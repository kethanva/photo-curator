"""
Ranking engine: computes a composite quality + importance score for each photo.

Score = weighted sum of seven normalised components:
    sharpness           — Laplacian blur score
    aesthetic           — CLIP-based aesthetic prediction (replaces old proxy)
    face_score          — log-scaled face count (people matter)
    sentiment           — smile + eyes-open score from MediaPipe
    uniqueness          — penalises duplicates / near-duplicates
    metadata_importance — GPS + timestamp signals a meaningful moment
    diversity_bonus     — rewards photos from larger event clusters
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def _minmax(values: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]; returns 0.5 if all values are equal."""
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-8:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


def score_photos(
    records: List[dict],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute a composite score for each photo.

    Components (sharpness … diversity_bonus) are min-max normalised to
    [0, 1] within the current eligible set, then combined as a weighted sum.

    Args:
        records: list of dicts from the database (all eligible photos)
        weights: component weights — see config.yaml ranking.weights

    Returns:
        {path: score}
    """
    if not records:
        return {}

    paths = [r["path"] for r in records]

    # ── Sharpness ────────────────────────────────────────────────
    blur_raw = np.array([r.get("blur_score", 0.0) for r in records], dtype=float)
    sharpness = _minmax(blur_raw)

    # ── Aesthetic ────────────────────────────────────────────────
    # Uses CLIP-based aesthetic_score stored in DB (0–1)
    aesthetic_raw = np.array([r.get("aesthetic_score", 0.5) for r in records], dtype=float)
    aesthetic = _minmax(aesthetic_raw)

    # ── Face score ───────────────────────────────────────────────
    face_raw = np.array([r.get("face_count", 0) for r in records], dtype=float)
    face_log = np.log1p(np.clip(face_raw, 0, 10))
    face_score = _minmax(face_log)

    # ── Sentiment ────────────────────────────────────────────────
    # Smile + eyes-open from MediaPipe; 0.5 for photos with no faces
    sentiment_raw = np.array([r.get("smile_score", 0.5) for r in records], dtype=float)
    sentiment = _minmax(sentiment_raw)

    # ── Uniqueness ───────────────────────────────────────────────
    uniqueness = np.array(
        [0.0 if r.get("is_duplicate", 0) else 1.0 for r in records]
    )

    # ── Metadata importance ──────────────────────────────────────
    meta = np.array(
        [
            (1.0 if r.get("has_gps", 0) else 0.4)
            * (1.0 if r.get("timestamp", 0.0) > 0 else 0.6)
            for r in records
        ],
        dtype=float,
    )

    # ── Diversity bonus ──────────────────────────────────────────
    cluster_ids = [r.get("cluster_id", -1) for r in records]
    cluster_sizes = Counter(cluster_ids)
    max_size = max(cluster_sizes.values(), default=1)
    diversity_raw = np.array(
        [cluster_sizes[cid] / max_size for cid in cluster_ids], dtype=float
    )
    diversity_bonus = _minmax(diversity_raw)

    # ── Base weighted sum ────────────────────────────────────────
    w = weights
    total = (
        w.get("sharpness",          0.15) * sharpness
        + w.get("aesthetic",        0.25) * aesthetic
        + w.get("face_score",       0.15) * face_score
        + w.get("sentiment",        0.15) * sentiment
        + w.get("uniqueness",       0.15) * uniqueness
        + w.get("metadata_importance", 0.08) * meta
        + w.get("diversity_bonus",  0.07) * diversity_bonus
    )

    return {path: float(score) for path, score in zip(paths, total)}
