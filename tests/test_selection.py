"""
Unit tests for src/selection.py — three-bucket photo selection and output writing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.selection import select_photos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    path: str,
    file_size: int = 500_000,
    resolution: int = 1080,
    quality_pass: int = 1,
    is_duplicate: int = 0,
    is_private: int = 0,
    cluster_id: int = 0,
    person_id: int = -1,
    is_frequent: int = 0,
) -> dict:
    return {
        "path": path,
        "file_size": file_size,
        "resolution": resolution,
        "quality_pass": quality_pass,
        "is_duplicate": is_duplicate,
        "is_private": is_private,
        "cluster_id": cluster_id,
        "person_id": person_id,
        "is_frequent": is_frequent,
        "aesthetic_score": 0.5,
        "blur_score": 100.0,
        "face_count": 0,
        "smile_score": 0.5,
    }


def _scores(records: list[dict], base: float = 0.5) -> dict:
    return {r["path"]: base for r in records}


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------

class TestSelectPhotosFiltering:
    def test_empty_input_returns_empty(self):
        result = select_photos([], {})
        assert result == []

    def test_all_duplicates_returns_empty(self):
        recs = [_rec(f"{i}.jpg", is_duplicate=1) for i in range(3)]
        result = select_photos(recs, _scores(recs))
        assert result == []

    def test_all_private_returns_empty(self):
        recs = [_rec(f"{i}.jpg", is_private=1) for i in range(3)]
        result = select_photos(recs, _scores(recs))
        assert result == []

    def test_quality_fail_excluded(self):
        recs = [
            _rec("good.jpg", quality_pass=1),
            _rec("bad.jpg",  quality_pass=0),
        ]
        result = select_photos(recs, _scores(recs))
        paths = [r["path"] for r in result]
        assert "bad.jpg" not in paths

    def test_duplicate_excluded(self):
        recs = [
            _rec("orig.jpg", is_duplicate=0),
            _rec("dup.jpg",  is_duplicate=1),
        ]
        result = select_photos(recs, _scores(recs))
        paths = [r["path"] for r in result]
        assert "orig.jpg" in paths
        assert "dup.jpg" not in paths


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------

class TestSelectPhotosBudget:
    def test_respects_max_bytes(self):
        """With a very tight budget, few photos should be selected."""
        recs = [_rec(f"{i}.jpg", file_size=500_000) for i in range(20)]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=1_000_000)  # 1 MB only
        # With 500 KB per photo, at most 2 should fit (estimate is smaller due to resize calc)
        assert len(result) <= 5

    def test_all_fit_within_large_budget(self):
        """With a huge budget, all qualifying photos should be selected."""
        recs = [_rec(f"{i}.jpg", file_size=10_000) for i in range(5)]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=10_000_000_000)
        assert len(result) == 5

    def test_no_photo_selected_twice(self):
        """Photos selected in multiple buckets must not appear twice."""
        # Create photos that qualify for people and location buckets
        recs = [
            _rec(f"{i}.jpg", person_id=i % 2, is_frequent=1, cluster_id=i % 3)
            for i in range(10)
        ]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=100_000_000)
        paths = [r["path"] for r in result]
        assert len(paths) == len(set(paths)), "Duplicate paths in selection"


# ---------------------------------------------------------------------------
# Bucket-specific tests
# ---------------------------------------------------------------------------

class TestPeopleBucket:
    def test_frequent_people_included(self):
        recs = [
            _rec("people.jpg", person_id=1, is_frequent=1),
            _rec("solo.jpg",   person_id=-1, is_frequent=0),
        ]
        scores = _scores(recs, 0.9)
        result = select_photos(recs, scores, max_bytes=100_000_000)
        paths = [r["path"] for r in result]
        assert "people.jpg" in paths

    def test_max_per_person_respected_in_people_bucket(self):
        """The people bucket should not add more than max_per_person per identity.

        Budget is tight (3 × ~500KB = 1.5MB) so the aesthetic overflow bucket
        cannot add beyond the per-person cap. Each test record uses file_size=500_000
        and resolution=1080 whose estimated output size is ≤500_000 bytes.
        """
        recs = [_rec(f"p{i}.jpg", person_id=1, is_frequent=1) for i in range(10)]
        scores = _scores(recs, 0.9)
        result = select_photos(recs, scores, max_bytes=1_600_000, max_per_person=3)
        person_1_count = sum(1 for r in result if r.get("person_id") == 1)
        assert person_1_count <= 3


class TestLocationBucket:
    def test_max_per_location_respected_in_location_bucket(self):
        """Location bucket should not exceed max_per_location per cluster.

        Budget is tight (5 × ~500KB = 2.5MB) so the aesthetic bucket cannot
        overflow beyond the location cap.
        """
        recs = [_rec(f"loc{i}.jpg", cluster_id=0) for i in range(20)]
        scores = {r["path"]: 0.5 for r in recs}
        result = select_photos(recs, scores, max_bytes=2_600_000, max_per_location=5)
        cluster_0_count = sum(1 for r in result if r.get("cluster_id") == 0)
        assert cluster_0_count <= 5


class TestAestheticBucket:
    def test_high_score_photos_preferred(self):
        """Given tight budget, higher-scored photos should be in result."""
        recs = [
            _rec("great.jpg",  file_size=10_000),
            _rec("average.jpg", file_size=10_000),
        ]
        scores = {"great.jpg": 0.99, "average.jpg": 0.01}
        result = select_photos(recs, scores, max_bytes=15_000)
        paths = [r["path"] for r in result]
        # The great photo should be selected over the average one
        assert "great.jpg" in paths


# ---------------------------------------------------------------------------
# Size estimation tests
# ---------------------------------------------------------------------------

class TestSizeEstimation:
    def test_small_file_not_grown(self):
        """A photo already smaller than target is not enlarged."""
        recs = [_rec("tiny.jpg", file_size=50_000, resolution=200)]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=100_000_000)
        assert len(result) == 1

    def test_zero_resolution_doesnt_crash(self):
        recs = [_rec("zero.jpg", resolution=0)]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=100_000_000)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Percentage mode tests
# ---------------------------------------------------------------------------

class TestPercentageMode:
    def _make_pool(self, n: int) -> tuple:
        recs = [_rec(f"img{i}.jpg") for i in range(n)]
        scores = {r["path"]: float(i) / n for i, r in enumerate(recs)}
        return recs, scores

    def test_percentage_mode_selects_correct_count(self):
        """15% of 100 eligible photos = 15 photos."""
        recs, scores = self._make_pool(100)
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.15,
        )
        assert len(result) == 15

    def test_percentage_mode_rounds_down(self):
        """int(7 * 0.15) = 1; should return 1 photo."""
        recs, scores = self._make_pool(7)
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.15,
        )
        assert len(result) == max(1, int(7 * 0.15))

    def test_percentage_mode_at_least_one(self):
        """Even very small percentage of a tiny pool returns ≥ 1 photo."""
        recs, scores = self._make_pool(3)
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.01,
        )
        assert len(result) >= 1

    def test_percentage_mode_100_percent(self):
        """100% should return all eligible photos (subject to byte cap)."""
        recs, scores = self._make_pool(20)
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
        )
        assert len(result) == 20

    def test_bytes_mode_ignores_percentage(self):
        """In bytes mode, output_percentage is irrelevant."""
        recs, scores = self._make_pool(50)
        # Tight byte cap that allows ~5 photos (each ~500 KB → 2.5 MB limit)
        result = select_photos(
            recs, scores,
            max_bytes=2_500_000,
            output_mode="bytes",
            output_percentage=1.0,   # would allow all 50 if mode were percentage
        )
        assert len(result) < 50

    def test_byte_cap_still_enforced_in_percentage_mode(self):
        """Even in percentage mode the hard byte cap prevents overrun."""
        recs, scores = self._make_pool(100)
        # Cap so tight only ~1 photo fits
        result = select_photos(
            recs, scores,
            max_bytes=100_000,
            output_mode="percentage",
            output_percentage=0.50,
        )
        assert len(result) <= 2  # byte cap overrides photo count

    def test_percentage_selects_highest_scored_photos(self):
        """The selected subset should be the top-N by score."""
        n = 20
        recs = [_rec(f"img{i}.jpg") for i in range(n)]
        scores = {f"img{i}.jpg": i / n for i in range(n)}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.25,  # top 5
        )
        selected_paths = {r["path"] for r in result}
        # Top 5 by score: img15..img19
        for i in range(15, 20):
            assert f"img{i}.jpg" in selected_paths
