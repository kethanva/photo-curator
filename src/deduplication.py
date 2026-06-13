"""
Duplicate and near-duplicate detection.

Two passes:
1. Perceptual hash (pHash) — fast Hamming distance check
2. CLIP cosine similarity — catches visually identical photos with minor edits

The sharpest (highest blur_score) copy is kept; others are marked duplicate.

For large libraries (>10k photos) use find_duplicates_ann() which replaces
the O(N²) CLIP scan with O(N·K) ANN queries via a VectorStore.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Set

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.vector_store import VectorStore


# Supported perceptual-hash algorithms. Each maps to an imagehash function that
# returns a 64-bit hash (16 hex chars) at the default hash_size=8, so any two
# hashes from the *same* algorithm are Hamming-comparable. ``whash`` needs the
# optional PyWavelets dependency; it is resolved lazily and falls back to phash.
# Mirrors the selectable HashAlgorithm set in photos-cleanup/src/hash.rs.
_HASH_ALGORITHMS = {
    "phash": lambda im: imagehash.phash(im),
    "dhash": lambda im: imagehash.dhash(im),
    "ahash": lambda im: imagehash.average_hash(im),
    "whash": lambda im: imagehash.whash(im),
}

DEFAULT_HASH_ALGORITHM = "phash"


def compute_phash(img: Image.Image, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """
    Return a perceptual hash as a hex string.

    ``algorithm`` selects the hashing method: ``phash`` (DCT, default),
    ``dhash`` (gradient), ``ahash`` (average/mean), or ``whash`` (wavelet).
    An unknown name falls back to phash. All produce equal-length hashes, so
    two hashes are only ever compared when they share an algorithm.
    """
    fn = _HASH_ALGORITHMS.get(algorithm, _HASH_ALGORITHMS[DEFAULT_HASH_ALGORITHM])
    try:
        return str(fn(img))
    except (OSError, ValueError, TypeError, ImportError) as exc:
        # imagehash can fail on unusual PIL image modes (1-bit BMP, 16-bit
        # TIFF); whash raises ImportError when PyWavelets is absent. Empty
        # string makes hamming_distance return 64 (max), so the photo is never
        # marked as a pHash duplicate. Logging keeps the otherwise-invisible
        # failure observable.
        logger.debug("compute_phash(algorithm=%s) failed (mode=%s, size=%s): %s",
                     algorithm, img.mode, img.size, exc)
        return ""


def hamming_distance(h1: str, h2: str) -> int:
    """
    Hamming distance between two pHash hex strings.
    Returns 64 (maximum) if either hash is invalid.
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 64
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return 64


class _BKTree:
    """
    BK-tree over equal-length perceptual-hash hex strings for Hamming radius
    queries.

    Hamming distance is a true metric, so triangle-inequality pruning is
    exact: a radius query only descends into children whose edge distance d
    satisfies |d − dist(query, node)| ≤ tolerance. This cuts the accepted-set
    pHash scan in find_duplicates_ann from O(N_accepted) per photo to roughly
    O(log N) for typical thresholds, removing the O(N²) cliff on large
    libraries.

    Hashes of different hex lengths are kept in separate trees (one root per
    length): mixing lengths would break the metric — hamming_distance returns
    a constant 64 for any length mismatch, which violates the triangle
    inequality and would make pruning skip true matches.

    Ported from photos-cleanup/src/group.rs::BkTree.
    """

    def __init__(self) -> None:
        # Flat node storage: [hash_int, payload, children {dist: node_idx}].
        self._nodes: List[list] = []
        self._roots: Dict[int, int] = {}   # hex length → root node index

    @staticmethod
    def _to_int(phash_hex: str):
        try:
            return int(phash_hex, 16)
        except ValueError:
            return None

    def insert(self, phash_hex: str, payload: object) -> None:
        """Insert a hash with an arbitrary payload; empty/invalid hashes are
        ignored (they can never match anything — hamming returns 64)."""
        if not phash_hex:
            return
        h = self._to_int(phash_hex)
        if h is None:
            return
        length = len(phash_hex)
        new_idx = len(self._nodes)
        if length not in self._roots:
            self._nodes.append([h, payload, {}])
            self._roots[length] = new_idx
            return
        cur = self._roots[length]
        while True:
            node = self._nodes[cur]
            dist = (node[0] ^ h).bit_count()
            child = node[2].get(dist)
            if child is None:
                self._nodes.append([h, payload, {}])
                node[2][dist] = new_idx
                return
            cur = child

    def search(self, phash_hex: str, tolerance: int) -> List[object]:
        """Return payloads of all stored hashes within ``tolerance`` Hamming
        bits of ``phash_hex`` (same hex length only)."""
        if not phash_hex:
            return []
        h = self._to_int(phash_hex)
        if h is None:
            return []
        root = self._roots.get(len(phash_hex))
        if root is None:
            return []
        out: List[object] = []
        stack = [root]
        while stack:
            node = self._nodes[stack.pop()]
            dist = (node[0] ^ h).bit_count()
            if dist <= tolerance:
                out.append(node[1])
            lo, hi = dist - tolerance, dist + tolerance
            for d, child in node[2].items():
                if lo <= d <= hi:
                    stack.append(child)
        return out


def _secondary_confirms(
    rec: dict,
    keeper: dict,
    dual_hash: bool,
    secondary_threshold: int,
) -> bool:
    """
    Dual-hash confirmation for a pair whose PRIMARY hashes already matched.

    Returns True when the gate is disabled, when either side lacks a
    secondary hash (inconclusive — fall back to the primary verdict rather
    than vetoing, see _is_phash_dup), or when the secondary Hamming distance
    is within ``secondary_threshold``.
    """
    if not dual_hash:
        return True
    h2a, h2b = rec.get("phash2", ""), keeper.get("phash2", "")
    if not h2a or not h2b:
        return True
    return hamming_distance(h2a, h2b) <= secondary_threshold


def _is_phash_dup(
    rec: dict,
    keeper: dict,
    phash_threshold: int,
    dual_hash: bool,
    secondary_threshold: int,
) -> bool:
    """
    Decide whether ``rec`` is a perceptual-hash duplicate of ``keeper``.

    Primary gate: Hamming distance of the primary pHash ≤ ``phash_threshold``.

    When ``dual_hash`` is enabled the primary match must be *confirmed* by a
    second, independent hash algorithm (stored as ``phash2``) also being within
    ``secondary_threshold``. Requiring both algorithms to agree dramatically
    cuts false-positive groupings at the cost of one extra stored hash.

    If either photo lacks a secondary hash (ingested before dual_hash was
    enabled, or the secondary algorithm failed on that image), the secondary
    check is inconclusive — fall back to the primary verdict rather than
    vetoing. Otherwise a mixed old/new library would silently stop detecting
    near-duplicates across the ingest boundary (hamming_distance returns 64
    for any empty hash).

    Mirrors the dual-hash confirmation edge gate in photos-cleanup/src/group.rs.
    """
    if hamming_distance(rec.get("phash", ""), keeper.get("phash", "")) > phash_threshold:
        return False
    return _secondary_confirms(rec, keeper, dual_hash, secondary_threshold)


def find_burst_duplicates(
    records: List[dict],
    gap_seconds: float = 3.0,
) -> Set[str]:
    """
    Group photos shot in rapid succession (EXIF-timestamp proximity) and keep
    only the sharpest frame of each burst, marking the rest as duplicates.

    Complements perceptual hashing: burst shots often differ at the pixel level
    (subject motion, exposure drift) so a pHash may not cluster them, yet they
    are clearly one moment and the user wants a single keeper.

    Algorithm (ported from photos-cleanup/src/burst.rs):
        1. Drop records with no usable timestamp (≤ 0).
        2. Sort by timestamp.
        3. Sliding window: a gap > ``gap_seconds`` between consecutive photos
           closes the current burst and starts a new one.
        4. For each burst of ≥ 2 photos, keep the max-blur_score frame; the
           others are returned as duplicates.

    Args:
        records: dicts with keys path (str), timestamp (float, Unix seconds),
            blur_score (float).
        gap_seconds: max gap between consecutive frames to stay in one burst.

    Returns:
        Set of paths to mark as duplicates (every burst member except the
        sharpest).
    """
    timed = sorted(
        (r for r in records if float(r.get("timestamp", 0.0) or 0.0) > 0.0),
        key=lambda r: float(r["timestamp"]),
    )
    if len(timed) < 2:
        return set()

    to_remove: Set[str] = set()

    def _close_burst(burst: List[dict]) -> None:
        if len(burst) < 2:
            return
        # Sharpest frame wins; ties (e.g. all 0.0 after a metric failure)
        # break on file size so the highest-fidelity copy is kept. Mirrors
        # the (sharpness, pixels, file_size) key in photos-cleanup
        # selection.rs::compute_best.
        keeper = max(
            burst,
            key=lambda r: (
                r.get("blur_score", 0.0) or 0.0,
                r.get("file_size", 0) or 0,
            ),
        )
        for r in burst:
            if r["path"] != keeper["path"]:
                to_remove.add(r["path"])

    burst: List[dict] = [timed[0]]
    for prev, cur in zip(timed, timed[1:]):
        gap = float(cur["timestamp"]) - float(prev["timestamp"])
        if gap > gap_seconds:
            _close_burst(burst)
            burst = [cur]
        else:
            burst.append(cur)
    _close_burst(burst)

    return to_remove


def find_duplicates(
    records: List[dict],
    phash_threshold: int = 8,
    embedding_threshold: float = 0.95,
    dual_hash: bool = False,
    secondary_threshold: int = 8,
) -> Set[str]:
    """
    Identify duplicate photos across all records.

    Strategy:
        - Sort by blur_score descending so the sharpest copy is processed first.
        - For each candidate, compare against all already-accepted photos.
        - Mark as duplicate if:
            * exact file hash match, OR
            * pHash Hamming distance ≤ phash_threshold, OR
            * CLIP cosine similarity ≥ embedding_threshold

    Args:
        records: list of dicts with keys:
            path (str), file_hash (str), phash (str),
            clip_emb (np.ndarray | None), blur_score (float)
        phash_threshold: max Hamming distance for pHash duplicate
        embedding_threshold: cosine similarity floor for visual duplicate

    Returns:
        Set of paths to mark as duplicates (the worse-quality copies).
    """
    # Sharpest copy first; ties broken on file size so the highest-fidelity
    # copy survives among equally-sharp duplicates (re-encodes/resizes share a
    # pHash but the larger file is the less-compressed original). Mirrors the
    # (sharpness, pixels, file_size) keeper key in find_burst_duplicates and
    # photos-cleanup selection.rs::compute_best.
    sorted_recs = sorted(
        records,
        key=lambda r: (r.get("blur_score", 0.0) or 0.0, r.get("file_size", 0) or 0),
        reverse=True,
    )

    to_remove: Set[str] = set()
    kept: List[dict] = []

    for rec in sorted_recs:
        if rec["path"] in to_remove:
            continue

        is_dup = False

        for keeper in kept:
            # 1. Exact byte-for-byte match
            if rec.get("file_hash") and rec["file_hash"] == keeper.get("file_hash"):
                is_dup = True
                break

            # 2. Perceptual hash similarity (optionally dual-hash confirmed)
            if _is_phash_dup(rec, keeper, phash_threshold, dual_hash, secondary_threshold):
                is_dup = True
                break

            # 3. CLIP embedding similarity
            e1 = rec.get("clip_emb")
            e2 = keeper.get("clip_emb")
            if e1 is not None and e2 is not None:
                n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if n1 > 1e-8 and n2 > 1e-8:
                    sim = float(np.dot(e1, e2) / (n1 * n2))
                    if sim >= embedding_threshold:
                        is_dup = True
                        break

        if is_dup:
            to_remove.add(rec["path"])
        else:
            kept.append(rec)

    return to_remove


def find_duplicates_ann(
    records: List[dict],
    store: "VectorStore",
    phash_threshold: int = 8,
    embedding_threshold: float = 0.95,
    ann_k: int = 100,
    dual_hash: bool = False,
    secondary_threshold: int = 8,
) -> Set[str]:
    """
    ANN-accelerated duplicate detection.

    For each candidate photo the vector store is queried for its K nearest
    CLIP neighbours.  Only those neighbours that have already been accepted
    are checked for similarity — reducing the comparison cost from O(N²) to
    O(N·K) where K is small (default 100).

    The exact file-hash check is an O(1) dict lookup; the pHash check uses a
    BK-tree radius query (roughly O(log N_accepted) per photo) instead of a
    linear scan over every accepted photo.

    Args:
        records:             same schema as find_duplicates()
        store:               initialised VectorStore
        phash_threshold:     max Hamming distance for pHash duplicate
        embedding_threshold: cosine similarity floor for visual duplicate
        ann_k:               ANN candidates to retrieve per photo

    Returns:
        Set of paths to mark as duplicates (the worse-quality copies).
    """
    # Sharpest copy first; ties broken on file size so the highest-fidelity
    # copy survives among equally-sharp duplicates (re-encodes/resizes share a
    # pHash but the larger file is the less-compressed original). Mirrors the
    # (sharpness, pixels, file_size) keeper key in find_burst_duplicates and
    # photos-cleanup selection.rs::compute_best.
    sorted_recs = sorted(
        records,
        key=lambda r: (r.get("blur_score", 0.0) or 0.0, r.get("file_size", 0) or 0),
        reverse=True,
    )

    to_remove: Set[str] = set()
    accepted_paths: Set[str] = set()
    file_hash_index: dict = {}   # file_hash  → path (first accepted copy)
    # BK-tree over accepted photos' primary hashes; payload is a minimal
    # keeper dict {phash, phash2} so the dual-hash gate can confirm against
    # the accepted copy's secondary hash too. Radius queries replace the old
    # linear scan over every accepted photo (O(N²) Python-level XOR loop).
    phash_tree = _BKTree()

    for rec in sorted_recs:
        path = rec["path"]
        if path in to_remove:
            continue

        is_dup = False

        # 1. Exact byte-for-byte match via hash dict  (O(1))
        fh = rec.get("file_hash", "")
        if fh and fh in file_hash_index:
            is_dup = True

        # 2. ANN CLIP search  (O(log N) query, O(K) candidate check)
        if not is_dup and accepted_paths:
            emb = rec.get("clip_emb")
            # Stale zero-vec blobs from previous failed extractions can sit
            # in SQLite even though the vector store now refuses them at the
            # boundary. ChromaDB cosine distance with a zero-norm query is
            # implementation-defined (NaN, garbage neighbours, or raise) —
            # short-circuit before search_clip and let pHash/exact-hash run.
            if emb is not None and float(np.linalg.norm(emb)) >= 1e-6:
                try:
                    candidates = store.search_clip(emb, n_results=ann_k)
                except (RuntimeError, ValueError, OSError) as exc:
                    # ChromaDB query can raise on dimension mismatch or
                    # transient HNSW issues. Skip ANN for this record only —
                    # the pHash + exact-hash gates still run, so dedup
                    # degrades gracefully instead of aborting the stage and
                    # losing all dedup work for the rest of the library.
                    logger.warning(
                        "ANN search failed for %s (%s); falling back to "
                        "pHash/exact-hash dedup for this record.",
                        path, exc,
                    )
                    candidates = []
                for cand_path, dist in candidates:
                    if cand_path not in accepted_paths:
                        continue
                    # ChromaDB cosine distance: dist = 1 − cosine_similarity
                    # (valid for normalised vectors)
                    sim = max(0.0, 1.0 - float(dist))
                    if sim >= embedding_threshold:
                        is_dup = True
                        break

        # 3. pHash Hamming radius query via BK-tree — optionally dual-hash
        #    confirmed. The tree returns only keepers already within
        #    phash_threshold on the primary hash.
        if not is_dup:
            for keeper in phash_tree.search(rec.get("phash", ""), phash_threshold):
                if _secondary_confirms(rec, keeper, dual_hash, secondary_threshold):
                    is_dup = True
                    break

        if is_dup:
            to_remove.add(path)
        else:
            accepted_paths.add(path)
            if fh:
                file_hash_index[fh] = path
            phash_tree.insert(rec.get("phash", ""), {
                "phash": rec.get("phash", ""),
                "phash2": rec.get("phash2", ""),
            })

    return to_remove
