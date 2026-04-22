"""
Ranking engine: computes a composite quality + importance score for each photo.

Score = weighted sum of nine normalised components:
    sharpness           — Laplacian blur score
    aesthetic           — CLIP-based aesthetic prediction
    face_score          — log-scaled face count
    face_prominence     — face bounding-box area / image area (close-ups score higher)
    face_confidence     — mean MTCNN detection probability (rewards clearly detected faces)
    sentiment           — smile + eyes-open score from MediaPipe
    uniqueness          — rewards photos from small / unique event clusters
    metadata_importance — GPS + timestamp signals a meaningful moment
    diversity_bonus     — rewards photos from well-formed event clusters
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

    # ── Face prominence ──────────────────────────────────────────
    # Fraction of frame area covered by qualifying faces.
    # Close-up portraits (face fills ~30 % of frame) score much higher than
    # crowd shots where faces are tiny. Capped at 1.0 by face_detection.
    face_prom_raw = np.array([r.get("face_prominence", 0.0) for r in records], dtype=float)
    face_prominence = _minmax(face_prom_raw)

    # ── Face confidence ──────────────────────────────────────────
    # Mean MTCNN detection probability. Rewards unambiguously detected faces
    # over borderline / partial detections.
    face_conf_raw = np.array([r.get("face_confidence", 0.0) for r in records], dtype=float)
    face_confidence = _minmax(face_conf_raw)

    # ── Sentiment ────────────────────────────────────────────────
    # Smile + eyes-open from MediaPipe; 0.5 for photos with no faces
    sentiment_raw = np.array([r.get("smile_score", 0.5) for r in records], dtype=float)
    sentiment = _minmax(sentiment_raw)

    # ── Uniqueness ───────────────────────────────────────────────
    # Duplicates are already filtered upstream, so a binary is_duplicate flag
    # is constant (all 1.0) for eligible photos and gives zero discrimination.
    # Define uniqueness as inverse event-cluster density: photos from small
    # clusters (or noise singletons) are more "unique moments" than the 500th
    # photo of a wedding.
    cluster_ids = [r.get("cluster_id", -1) for r in records]
    cluster_sizes = Counter(cluster_ids)
    # Noise (cluster_id == -1) is a collection of singletons, not a real
    # cluster, so each such photo should read as maximally unique.
    uniqueness_raw = np.array(
        [
            1.0 if cid < 0 else 1.0 / float(cluster_sizes[cid])
            for cid in cluster_ids
        ],
        dtype=float,
    )
    uniqueness = _minmax(uniqueness_raw)

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
    # Rewards photos from well-formed event clusters (weddings, trips) without
    # double-counting DBSCAN noise: cluster_id == -1 is a bag of unrelated
    # singletons, so it should not dominate the "max cluster size" reference.
    real_sizes = {cid: n for cid, n in cluster_sizes.items() if cid >= 0}
    max_size = max(real_sizes.values(), default=1)
    diversity_raw = np.array(
        [
            0.0 if cid < 0 else cluster_sizes[cid] / max_size
            for cid in cluster_ids
        ],
        dtype=float,
    )
    diversity_bonus = _minmax(diversity_raw)

    # ── Base weighted sum ────────────────────────────────────────
    w = weights
    total = (
        w.get("sharpness",           0.10) * sharpness
        + w.get("aesthetic",         0.20) * aesthetic
        + w.get("face_score",        0.18) * face_score
        + w.get("face_prominence",   0.10) * face_prominence
        + w.get("face_confidence",   0.05) * face_confidence
        + w.get("sentiment",         0.18) * sentiment
        + w.get("uniqueness",        0.10) * uniqueness
        + w.get("metadata_importance", 0.05) * meta
        + w.get("diversity_bonus",   0.04) * diversity_bonus
    )

    return {path: float(score) for path, score in zip(paths, total)}
