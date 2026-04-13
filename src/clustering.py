"""
Event clustering: groups photos into events using DBSCAN on a combined
feature space of timestamp, GPS coordinates, and CLIP visual embeddings.

CLIP dimensions are reduced via PCA (512 → 32) before combining with
scalar features to avoid the curse of dimensionality.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


_CLIP_DIM = 512
_PCA_DIM = 32   # Reduced CLIP dimensions for clustering


def _build_feature_matrix(
    records: List[dict],
    time_weight: float,
    gps_weight: float,
    visual_weight: float,
) -> np.ndarray:
    """
    Build a normalised feature matrix:
        col 0   : timestamp  (× time_weight)
        col 1-2 : lat, lon   (× gps_weight)
        col 3+  : PCA-reduced CLIP embedding  (× visual_weight)
    """
    n = len(records)

    # --- Scalar features ---
    scalars = np.zeros((n, 3), dtype=np.float32)
    for i, r in enumerate(records):
        scalars[i, 0] = r.get("timestamp", 0.0)
        scalars[i, 1] = r.get("lat", 0.0)
        scalars[i, 2] = r.get("lon", 0.0)

    # --- CLIP embeddings ---
    clip_matrix = np.zeros((n, _CLIP_DIM), dtype=np.float32)
    has_clip = False
    for i, r in enumerate(records):
        emb = r.get("clip_emb")
        if emb is not None and len(emb) == _CLIP_DIM:
            clip_matrix[i] = emb
            has_clip = True

    # --- Normalise scalars ---
    scaler = StandardScaler()
    scalars_norm = scaler.fit_transform(scalars)

    # Apply per-column weights
    scalars_norm[:, 0] *= time_weight
    scalars_norm[:, 1:3] *= gps_weight

    # --- Reduce CLIP with PCA ---
    if has_clip and n > _PCA_DIM:
        pca = PCA(n_components=min(_PCA_DIM, n - 1), random_state=42)
        clip_reduced = pca.fit_transform(clip_matrix)
        clip_scaler = StandardScaler()
        clip_reduced = clip_scaler.fit_transform(clip_reduced) * visual_weight
    else:
        # Fall back to raw (zero) visual features if no CLIP embeddings
        clip_reduced = np.zeros((n, _PCA_DIM), dtype=np.float32)

    return np.hstack([scalars_norm, clip_reduced]).astype(np.float32)


def cluster_events(
    records: List[dict],
    eps: float = 0.6,
    min_samples: int = 2,
    time_weight: float = 1.0,
    gps_weight: float = 2.0,
    visual_weight: float = 1.5,
) -> Dict[str, int]:
    """
    Assign event cluster IDs to photos.

    Args:
        records: list of dicts with keys: path, timestamp, lat, lon, clip_emb
        eps: DBSCAN neighbourhood radius
        min_samples: minimum cluster size
        time_weight, gps_weight, visual_weight: feature importance

    Returns:
        {path: cluster_id}  — cluster_id == -1 means noise/singleton
    """
    if not records:
        return {}

    if len(records) == 1:
        return {records[0]["path"]: 0}

    X = _build_feature_matrix(records, time_weight, gps_weight, visual_weight)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = db.fit_predict(X)

    return {rec["path"]: int(lbl) for rec, lbl in zip(records, labels)}
