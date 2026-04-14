"""
Unit tests for src/ranking.py — composite photo scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ranking import _minmax, score_photos


# ---------------------------------------------------------------------------
# _minmax tests
# ---------------------------------------------------------------------------

class TestMinmax:
    def test_single_element_returns_half(self):
        arr = np.array([42.0])
        result = _minmax(arr)
        assert result[0] == pytest.approx(0.5)

    def test_all_equal_returns_half(self):
        arr = np.array([7.0, 7.0, 7.0])
        result = _minmax(arr)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_min_maps_to_zero_max_maps_to_one(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = _minmax(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_values_in_zero_one_range(self):
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        result = _minmax(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_negative_values_handled(self):
        arr = np.array([-10.0, 0.0, 10.0])
        result = _minmax(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_photos tests
# ---------------------------------------------------------------------------

def _make_record(
    path: str,
    blur: float = 100.0,
    aesthetic: float = 0.5,
    faces: int = 0,
    smile: float = 0.5,
    is_dup: int = 0,
    has_gps: int = 1,
    timestamp: float = 1.0,
    cluster_id: int = 0,
) -> dict:
    return {
        "path": path,
        "blur_score": blur,
        "aesthetic_score": aesthetic,
        "face_count": faces,
        "smile_score": smile,
        "is_duplicate": is_dup,
        "has_gps": has_gps,
        "timestamp": timestamp,
        "cluster_id": cluster_id,
    }


DEFAULT_WEIGHTS = {
    "sharpness": 0.15,
    "aesthetic": 0.25,
    "face_score": 0.15,
    "sentiment": 0.15,
    "uniqueness": 0.15,
    "metadata_importance": 0.08,
    "diversity_bonus": 0.07,
}


class TestScorePhotos:
    def test_empty_returns_empty_dict(self):
        result = score_photos([], DEFAULT_WEIGHTS)
        assert result == {}

    def test_returns_dict_of_floats(self):
        recs = [_make_record("a.jpg"), _make_record("b.jpg")]
        result = score_photos(recs, DEFAULT_WEIGHTS)
        assert isinstance(result, dict)
        for path, score in result.items():
            assert isinstance(score, float)

    def test_scores_in_zero_one_range(self):
        recs = [_make_record(f"{i}.jpg", blur=float(i * 10)) for i in range(5)]
        result = score_photos(recs, DEFAULT_WEIGHTS)
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_all_paths_present_in_result(self):
        paths = ["a.jpg", "b.jpg", "c.jpg"]
        recs = [_make_record(p) for p in paths]
        result = score_photos(recs, DEFAULT_WEIGHTS)
        assert set(result.keys()) == set(paths)

    def test_sharper_photo_scores_higher_when_sharpness_weight_dominant(self):
        weights = {"sharpness": 1.0, "aesthetic": 0.0, "face_score": 0.0,
                   "sentiment": 0.0, "uniqueness": 0.0, "metadata_importance": 0.0,
                   "diversity_bonus": 0.0}
        recs = [
            _make_record("sharp.jpg", blur=1000.0),
            _make_record("blurry.jpg", blur=1.0),
        ]
        result = score_photos(recs, weights)
        assert result["sharp.jpg"] > result["blurry.jpg"]

    def test_duplicate_penalised(self):
        weights = {"sharpness": 0.0, "aesthetic": 0.0, "face_score": 0.0,
                   "sentiment": 0.0, "uniqueness": 1.0, "metadata_importance": 0.0,
                   "diversity_bonus": 0.0}
        recs = [
            _make_record("orig.jpg", is_dup=0),
            _make_record("dup.jpg", is_dup=1),
        ]
        result = score_photos(recs, weights)
        assert result["orig.jpg"] > result["dup.jpg"]

    def test_single_record_scores(self):
        """Single photo should still produce a score."""
        recs = [_make_record("only.jpg")]
        result = score_photos(recs, DEFAULT_WEIGHTS)
        assert "only.jpg" in result
        assert 0.0 <= result["only.jpg"] <= 1.0

    def test_more_faces_scores_higher_with_face_weight(self):
        weights = {"sharpness": 0.0, "aesthetic": 0.0, "face_score": 1.0,
                   "sentiment": 0.0, "uniqueness": 0.0, "metadata_importance": 0.0,
                   "diversity_bonus": 0.0}
        recs = [
            _make_record("group.jpg", faces=5),
            _make_record("solo.jpg",  faces=0),
        ]
        result = score_photos(recs, weights)
        assert result["group.jpg"] > result["solo.jpg"]

    def test_gps_boosts_metadata_score(self):
        weights = {"sharpness": 0.0, "aesthetic": 0.0, "face_score": 0.0,
                   "sentiment": 0.0, "uniqueness": 0.0, "metadata_importance": 1.0,
                   "diversity_bonus": 0.0}
        recs = [
            _make_record("gps.jpg",    has_gps=1, timestamp=1000.0),
            _make_record("no_gps.jpg", has_gps=0, timestamp=0.0),
        ]
        result = score_photos(recs, weights)
        assert result["gps.jpg"] > result["no_gps.jpg"]

    def test_missing_optional_fields_use_defaults(self):
        """Records without optional fields should not crash scoring."""
        recs = [{"path": "a.jpg"}, {"path": "b.jpg"}]
        result = score_photos(recs, DEFAULT_WEIGHTS)
        assert "a.jpg" in result
        assert "b.jpg" in result
