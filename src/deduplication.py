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

from typing import TYPE_CHECKING, List, Set

import imagehash
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from src.vector_store import VectorStore


def compute_phash(img: Image.Image) -> str:
    """Return perceptual hash as a hex string."""
    try:
        return str(imagehash.phash(img))
    except Exception:
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


def find_duplicates(
    records: List[dict],
    phash_threshold: int = 8,
    embedding_threshold: float = 0.95,
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
    # Sharpest copy first — it will be the one that gets kept
    sorted_recs = sorted(
        records, key=lambda r: r.get("blur_score", 0.0), reverse=True
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

            # 2. Perceptual hash similarity
            if hamming_distance(rec.get("phash", ""), keeper.get("phash", "")) <= phash_threshold:
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
    ann_k: int = 30,
) -> Set[str]:
    """
    ANN-accelerated duplicate detection.

    For each candidate photo the vector store is queried for its K nearest
    CLIP neighbours.  Only those neighbours that have already been accepted
    are checked for similarity — reducing the comparison cost from O(N²) to
    O(N·K) where K is small (default 30).

    pHash and exact file-hash checks are still O(N·N_accepted) but those
    comparisons are cheap (integer XOR / dict lookup).

    Args:
        records:             same schema as find_duplicates()
        store:               initialised VectorStore
        phash_threshold:     max Hamming distance for pHash duplicate
        embedding_threshold: cosine similarity floor for visual duplicate
        ann_k:               ANN candidates to retrieve per photo

    Returns:
        Set of paths to mark as duplicates (the worse-quality copies).
    """
    sorted_recs = sorted(
        records, key=lambda r: r.get("blur_score", 0.0), reverse=True
    )

    to_remove: Set[str] = set()
    accepted_paths: Set[str] = set()
    file_hash_index: dict = {}   # file_hash  → path (first accepted copy)
    phash_index: dict = {}       # path       → phash (accepted only)

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
            if emb is not None:
                k = min(ann_k, len(accepted_paths))
                candidates = store.search_clip(emb, n_results=k)
                for cand_path, dist in candidates:
                    if cand_path not in accepted_paths:
                        continue
                    # ChromaDB cosine distance: dist = 1 − cosine_similarity
                    # (valid for normalised vectors)
                    sim = max(0.0, 1.0 - float(dist))
                    if sim >= embedding_threshold:
                        is_dup = True
                        break

        # 3. pHash Hamming distance  (fast int XOR, still O(N_accepted))
        if not is_dup:
            rec_phash = rec.get("phash", "")
            for ap_phash in phash_index.values():
                if hamming_distance(rec_phash, ap_phash) <= phash_threshold:
                    is_dup = True
                    break

        if is_dup:
            to_remove.add(path)
        else:
            accepted_paths.add(path)
            if fh:
                file_hash_index[fh] = path
            phash_index[path] = rec.get("phash", "")

    return to_remove
