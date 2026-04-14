"""
Unit tests for src/clustering.py — DBSCAN event clustering.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.clustering import cluster_events, _build_feature_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emb(seed: int = 0, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


def _rec(
    path: str,
    timestamp: float = 0.0,
    lat: float = 0.0,
    lon: float = 0.0,
    clip_emb: np.ndarray | None = None,
) -> dict:
    return {
        "path": path,
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon,
        "clip_emb": clip_emb,
    }


# ---------------------------------------------------------------------------
# _build_feature_matrix tests
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_float32_array(self):
        recs = [_rec("a.jpg", timestamp=1.0), _rec("b.jpg", timestamp=2.0)]
        X = _build_feature_matrix(recs, 1.0, 1.0, 1.0)
        assert X.dtype == np.float32

    def test_shape_correct_with_clip(self):
        """With CLIP: 3 scalar + 32 PCA dims = 35 cols when n > 32."""
        recs = [
            _rec(f"{i}.jpg", timestamp=float(i), clip_emb=_make_emb(i))
            for i in range(40)
        ]
        X = _build_feature_matrix(recs, 1.0, 1.0, 1.0)
        assert X.shape == (40, 3 + 32)

    def test_shape_without_clip(self):
        """Without CLIP embeddings: 3 scalar + 32 zero cols."""
        recs = [_rec(f"{i}.jpg") for i in range(10)]
        X = _build_feature_matrix(recs, 1.0, 1.0, 1.0)
        assert X.shape[0] == 10
        assert X.shape[1] == 3 + 32

    def test_no_nan_in_output(self):
        recs = [_rec(f"{i}.jpg", timestamp=float(i)) for i in range(5)]
        X = _build_feature_matrix(recs, 1.0, 2.0, 1.5)
        assert not np.any(np.isnan(X))


# ---------------------------------------------------------------------------
# cluster_events tests
# ---------------------------------------------------------------------------

class TestClusterEvents:
    def test_empty_input_returns_empty(self):
        result = cluster_events([])
        assert result == {}

    def test_single_record_gets_cluster_zero(self):
        result = cluster_events([_rec("a.jpg")])
        assert result == {"a.jpg": 0}

    def test_returns_dict_of_ints(self):
        recs = [_rec(f"{i}.jpg", timestamp=float(i * 1000)) for i in range(5)]
        result = cluster_events(recs)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, int)

    def test_all_paths_in_result(self):
        paths = [f"photo_{i}.jpg" for i in range(6)]
        recs = [_rec(p, timestamp=float(i)) for i, p in enumerate(paths)]
        result = cluster_events(recs)
        assert set(result.keys()) == set(paths)

    def test_temporally_close_photos_may_cluster(self):
        """Photos taken seconds apart should tend to cluster together."""
        recs = [
            _rec("a.jpg", timestamp=0.0),
            _rec("b.jpg", timestamp=10.0),
            _rec("c.jpg", timestamp=20.0),
            # Far apart in time
            _rec("d.jpg", timestamp=86400.0),  # 1 day later
        ]
        result = cluster_events(recs, eps=0.3, min_samples=2)
        # a, b, c should plausibly be closer to each other than to d
        # We just check it runs without error and returns correct keys
        assert set(result.keys()) == {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}

    def test_cluster_ids_are_integers(self):
        recs = [_rec(f"{i}.jpg") for i in range(4)]
        result = cluster_events(recs)
        for cid in result.values():
            assert isinstance(cid, int)

    def test_noise_points_get_minus_one(self):
        """With very tight clustering params, some photos may be noise (-1)."""
        recs = [
            _rec(f"{i}.jpg", timestamp=float(i * 1000))
            for i in range(5)
        ]
        result = cluster_events(recs, eps=0.01, min_samples=10)
        # With tight eps and high min_samples, all should be noise
        for cid in result.values():
            assert cid == -1

    def test_custom_weights_accepted(self):
        recs = [_rec(f"{i}.jpg") for i in range(3)]
        result = cluster_events(
            recs, eps=0.5, min_samples=2,
            time_weight=2.0, gps_weight=0.5, visual_weight=1.0
        )
        assert isinstance(result, dict)
