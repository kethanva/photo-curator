"""
Duplicate and near-duplicate detection.

Two passes:
1. Perceptual hash (pHash) — fast Hamming distance check
2. CLIP cosine similarity — catches visually identical photos with minor edits

The sharpest (highest blur_score) copy is kept; others are marked duplicate.
"""

from __future__ import annotations

from typing import List, Set

import imagehash
import numpy as np
from PIL import Image


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
