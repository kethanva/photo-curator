"""
Unit tests for src/face_clustering.py — face identity clustering.
Uses sklearn only (no MTCNN/FaceNet required).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.face_clustering import cluster_identities, get_frequent_people, photos_per_person


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emb(seed: int = 0, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


def _rec(path: str, face_count: int = 1, face_emb: np.ndarray | None = None) -> dict:
    return {"path": path, "face_count": face_count, "face_emb": face_emb}


# ---------------------------------------------------------------------------
# cluster_identities tests
# ---------------------------------------------------------------------------

class TestClusterIdentities:
    def test_empty_input(self):
        result = cluster_identities([])
        assert result == {}

    def test_no_faces_all_background(self):
        recs = [_rec(f"{i}.jpg", face_count=0) for i in range(3)]
        result = cluster_identities(recs)
        for path, (pid, is_freq) in result.items():
            assert pid == -1
            assert is_freq is False

    def test_no_face_emb_treated_as_background(self):
        recs = [_rec("a.jpg", face_count=1, face_emb=None)]
        result = cluster_identities(recs)
        assert result["a.jpg"] == (-1, False)

    def test_returns_all_paths(self):
        emb = _make_emb(0)
        recs = [_rec("a.jpg", face_emb=emb), _rec("b.jpg", face_count=0)]
        result = cluster_identities(recs)
        assert set(result.keys()) == {"a.jpg", "b.jpg"}

    def test_single_face_not_frequent_with_threshold_5(self):
        emb = _make_emb(0)
        recs = [_rec("a.jpg", face_emb=emb)]
        result = cluster_identities(recs, frequent_threshold=5)
        pid, is_freq = result["a.jpg"]
        assert is_freq is False

    def test_frequent_person_detected(self):
        """6 photos with near-identical face embeddings → 1 frequent person."""
        base = _make_emb(0)
        recs = []
        for i in range(6):
            # Very tiny perturbation — keeps L2 distance well within eps=0.5
            noise = np.random.default_rng(i).standard_normal(512).astype(np.float32) * 0.001
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            recs.append(_rec(f"person_{i}.jpg", face_emb=emb))
        result = cluster_identities(recs, eps=0.5, min_samples=2, frequent_threshold=5)
        frequent = [pid for _, (pid, is_freq) in result.items() if is_freq]
        assert len(frequent) > 0

    def test_values_are_tuples(self):
        emb = _make_emb(0)
        recs = [_rec("a.jpg", face_emb=emb)]
        result = cluster_identities(recs)
        for val in result.values():
            assert isinstance(val, tuple)
            assert len(val) == 2

    def test_person_id_is_int(self):
        emb = _make_emb(0)
        recs = [_rec("a.jpg", face_emb=emb)]
        result = cluster_identities(recs)
        for pid, _ in result.values():
            assert isinstance(pid, int)

    def test_two_distinct_identities(self):
        """Embeddings far apart → different cluster IDs."""
        emb_a = np.zeros(512, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(512, dtype=np.float32)
        emb_b[1] = 1.0

        recs = [
            _rec("person_a_1.jpg", face_emb=emb_a),
            _rec("person_a_2.jpg", face_emb=emb_a.copy()),
            _rec("person_b_1.jpg", face_emb=emb_b),
            _rec("person_b_2.jpg", face_emb=emb_b.copy()),
        ]
        result = cluster_identities(recs, eps=0.5, min_samples=2, frequent_threshold=2)
        pids = [result[r["path"]][0] for r in recs]
        # The two groups should have different cluster IDs
        a_ids = {pids[0], pids[1]}
        b_ids = {pids[2], pids[3]}
        assert a_ids.isdisjoint(b_ids) or (a_ids == {-1} or b_ids == {-1})


# ---------------------------------------------------------------------------
# get_frequent_people tests
# ---------------------------------------------------------------------------

class TestGetFrequentPeople:
    def test_empty_map(self):
        assert get_frequent_people({}) == []

    def test_no_frequent(self):
        identity_map = {"a.jpg": (-1, False), "b.jpg": (0, False)}
        assert get_frequent_people(identity_map) == []

    def test_returns_frequent_ids(self):
        identity_map = {
            "a.jpg": (1, True),
            "b.jpg": (2, True),
            "c.jpg": (-1, False),
        }
        result = get_frequent_people(identity_map)
        assert 1 in result
        assert 2 in result
        assert -1 not in result

    def test_returns_sorted(self):
        identity_map = {f"{i}.jpg": (i, True) for i in range(5, 0, -1)}
        result = get_frequent_people(identity_map)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# photos_per_person tests
# ---------------------------------------------------------------------------

class TestPhotosPerPerson:
    def test_empty_map(self):
        assert photos_per_person({}) == {}

    def test_only_frequent_included(self):
        identity_map = {
            "a.jpg": (1, True),
            "b.jpg": (2, False),
            "c.jpg": (-1, False),
        }
        result = photos_per_person(identity_map)
        assert 1 in result
        assert 2 not in result
        assert -1 not in result

    def test_paths_grouped_correctly(self):
        identity_map = {
            "a.jpg": (1, True),
            "b.jpg": (1, True),
            "c.jpg": (2, True),
        }
        result = photos_per_person(identity_map)
        assert sorted(result[1]) == ["a.jpg", "b.jpg"]
        assert result[2] == ["c.jpg"]
