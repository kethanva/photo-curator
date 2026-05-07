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
    min_ts = min((r.get("timestamp", 0.0) for r in records if r.get("timestamp", 0.0) > 0), default=0.0)
    
    for i, r in enumerate(records):
        ts = r.get("timestamp", 0.0)
        # Absolute scaling: 1 unit = 1 hour (3600 seconds)
        # Subtract min_ts to prevent float32 precision loss on large Unix timestamps
        if ts > 0:
            scalars[i, 0] = ((ts - min_ts) / 3600.0) * time_weight
        else:
            # Place photos missing timestamps infinitely far apart
            # so they never cluster based on time (singletons)
            scalars[i, 0] = 1_000_000.0 + (i * 100.0)
        
        # Absolute spatial scaling: 1 unit = 0.01 degrees (~1.1 km)
        scalars[i, 1] = (r.get("lat", 0.0) * 100.0) * gps_weight
        scalars[i, 2] = (r.get("lon", 0.0) * 100.0) * gps_weight

    # --- CLIP embeddings ---
    clip_matrix = np.zeros((n, _CLIP_DIM), dtype=np.float32)
    has_clip = False
    for i, r in enumerate(records):
        emb = r.get("clip_emb")
        if emb is not None and len(emb) == _CLIP_DIM:
            clip_matrix[i] = emb
            has_clip = True

    # --- Reduce CLIP with PCA ---
    # Below this sample count, PCA degenerates: with n samples we get at most
    # n-1 components, so the visual feature space collapses to a tiny number
    # of columns and the hstacked feature matrix is dominated by the 3 scalar
    # columns. Skip PCA in that regime — it produces less-meaningful clusters
    # than treating visual features as zeroed-out (which falls through to
    # time/GPS clustering only).
    if has_clip and n > _PCA_DIM:
        pca = PCA(n_components=_PCA_DIM, random_state=42)
        clip_reduced = pca.fit_transform(clip_matrix)
        # L2 normalize so visual vectors lie on a unit hypersphere
        # Expected max distance = sqrt(2) ~ 1.414, putting it on a comparable scale to time/gps
        norms = np.linalg.norm(clip_reduced, axis=1, keepdims=True)
        clip_reduced = np.divide(clip_reduced, norms, out=np.zeros_like(clip_reduced), where=norms>1e-8)
        clip_reduced *= visual_weight
    else:
        # Either no CLIP embeddings, or sample count too low for stable PCA:
        # fall back to zero visual features. Time + GPS still drive clustering.
        clip_reduced = np.zeros((n, _PCA_DIM), dtype=np.float32)

    return np.hstack([scalars, clip_reduced]).astype(np.float32)


def cluster_events(
    records: List[dict],
    eps: float = 3.0,
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
        # A single eligible photo is, by definition, a noise singleton — not
        # a real event cluster. Use DBSCAN's noise label so downstream code
        # (ranking uniqueness/diversity, selection cluster cap) treats it
        # consistently with the multi-photo path.
        return {records[0]["path"]: -1}

    X = _build_feature_matrix(records, time_weight, gps_weight, visual_weight)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)

    return {rec["path"]: int(lbl) for rec, lbl in zip(records, labels)}
