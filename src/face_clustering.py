"""
Cross-photo face identity clustering.

Clusters face embeddings (from face_detection.py) across all photos to
identify recurring individuals — distinguishing family/friends from
background people.

Two outputs:
  person_id    : integer cluster ID (−1 = background/unknown)
  is_frequent  : True if this person appears in ≥ frequent_threshold photos

Algorithm:
  1. Collect face embeddings for photos where face_count >= 1
  2. Run DBSCAN with euclidean distance (FaceNet embeddings are L2 normalised)
  3. Count occurrences per cluster → mark frequent vs background
  4. Assign the dominant person_id to each photo
     (for multi-face photos, pick the cluster with the most appearances)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_identities(
    records: List[dict],
    eps: float = 0.8,
    min_samples: int = 2,
    frequent_threshold: int = 5,
) -> Dict[str, Tuple[int, bool]]:
    """
    Cluster face embeddings across photos to identify recurring people.

    Args:
        records: list of dicts with keys:
            path (str), face_count (int), face_emb (np.ndarray | None)
        eps: DBSCAN radius — smaller = stricter identity matching
             Recommended: 0.6–0.9 for FaceNet 512-dim embeddings
        min_samples: minimum photos to form an identity cluster
        frequent_threshold: min photos for a person to be "frequent"

    Returns:
        {path: (person_id, is_frequent_person)}
        person_id == −1 means background / unrecognised individual
    """
    # Separate photos with face embeddings
    valid = [
        r for r in records
        if r.get("face_count", 0) >= 1 and r.get("face_emb") is not None
    ]

    if not valid:
        return {r["path"]: (-1, False) for r in records}

    # Build embedding matrix
    embeddings = np.vstack([r["face_emb"] for r in valid]).astype(np.float32)

    # L2-normalise (FaceNet embeddings should already be, but ensure it)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 1e-8, norms, 1.0)

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(embeddings)

    # Count appearances per cluster to identify frequent people
    cluster_counts = Counter(lbl for lbl in labels if lbl >= 0)

    frequent_ids = {
        cid for cid, count in cluster_counts.items()
        if count >= frequent_threshold
    }

    # Map paths to (person_id, is_frequent)
    result: Dict[str, Tuple[int, bool]] = {}
    for rec, lbl in zip(valid, labels):
        pid = int(lbl)
        is_freq = pid in frequent_ids
        result[rec["path"]] = (pid, is_freq)

    # Photos without face embeddings get background label
    valid_paths = {r["path"] for r in valid}
    for rec in records:
        if rec["path"] not in valid_paths:
            result[rec["path"]] = (-1, False)

    return result


def get_frequent_people(
    identity_map: Dict[str, Tuple[int, bool]],
) -> List[int]:
    """Return sorted list of person_ids marked as frequent."""
    return sorted({pid for pid, is_freq in identity_map.values() if is_freq and pid >= 0})


def photos_per_person(
    identity_map: Dict[str, Tuple[int, bool]],
) -> Dict[int, List[str]]:
    """
    Return {person_id: [list_of_paths]} for frequent people only.
    Useful for the per-person selection budget.
    """
    result: Dict[int, List[str]] = {}
    for path, (pid, is_freq) in identity_map.items():
        if is_freq and pid >= 0:
            result.setdefault(pid, []).append(path)
    return result
