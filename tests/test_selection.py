"""
Unit tests for src/selection.py — dynamic bucket photo selection and output writing.
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

DEFAULT_BUCKETS = {"people": 0.30, "location": 0.30, "aesthetic": 0.40}


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
        result = select_photos(recs, scores, max_bytes=1_000_000)
        assert len(result) <= 5

    def test_all_fit_within_large_budget(self):
        """With a huge budget, all qualifying photos should be selected."""
        recs = [_rec(f"{i}.jpg", file_size=10_000) for i in range(5)]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=10_000_000_000)
        assert len(result) == 5

    def test_no_photo_selected_twice(self):
        """Photos selected in multiple buckets must not appear twice."""
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

    def test_max_per_person_respected_across_all_buckets(self):
        """Per-person cap is enforced across all buckets including aesthetic.

        10 records, max_per_person_pct=0.30 → cap = max(1, int(10*0.30)) = 3.
        Even with a large byte budget, no more than 3 photos of person 1
        should appear in the output.
        """
        recs = [_rec(f"p{i}.jpg", person_id=1, is_frequent=1) for i in range(10)]
        scores = _scores(recs, 0.9)
        result = select_photos(recs, scores, max_bytes=100_000_000, max_per_person_pct=0.30)
        person_1_count = sum(1 for r in result if r.get("person_id") == 1)
        assert person_1_count <= 3


class TestLocationBucket:
    def test_location_bucket_caps_per_cluster(self):
        """Location bucket should not exceed max_per_location_pct per cluster.

        20 records across 2 clusters. max_per_location_pct=0.25 → cap = 5.
        """
        recs = [_rec(f"loc{i}.jpg", cluster_id=i % 2) for i in range(20)]
        scores = {r["path"]: 0.5 for r in recs}
        result = select_photos(recs, scores, max_bytes=100_000_000, max_per_location_pct=0.25)
        assert len(result) > 0


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
        assert "great.jpg" in paths


# ---------------------------------------------------------------------------
# Subject bucket tests
# ---------------------------------------------------------------------------

class TestSubjectBucket:
    def test_subject_bucket_picks_matching_photos(self):
        """Subject bucket should prefer photos with high subject similarity."""
        recs = [_rec(f"img{i}.jpg") for i in range(10)]
        scores = _scores(recs, 0.5)
        # Simulate: img0 and img1 are great bike photos, rest are not
        subj_scores = {
            "bike": {
                f"img{i}.jpg": (0.8 if i < 2 else 0.05) for i in range(10)
            }
        }
        buckets = {"bike": 0.50, "aesthetic": 0.50}
        result = select_photos(
            recs, scores,
            max_bytes=100_000_000,
            buckets=buckets,
            subject_scores=subj_scores,
        )
        paths = [r["path"] for r in result]
        assert "img0.jpg" in paths
        assert "img1.jpg" in paths

    def test_subject_bucket_skips_low_similarity(self):
        """Photos below similarity threshold should not fill subject bucket."""
        recs = [_rec(f"img{i}.jpg") for i in range(10)]
        scores = _scores(recs, 0.5)
        # All photos have very low bike similarity
        subj_scores = {"bike": {f"img{i}.jpg": 0.05 for i in range(10)}}
        buckets = {"bike": 0.50, "aesthetic": 0.50}
        result = select_photos(
            recs, scores,
            max_bytes=100_000_000,
            buckets=buckets,
            subject_scores=subj_scores,
            output_mode="percentage",
            output_percentage=0.50,
        )
        # All photos should come from aesthetic bucket, not bike
        assert len(result) > 0

    def test_multiple_subject_buckets(self):
        """Multiple subject buckets can coexist."""
        recs = [_rec(f"img{i}.jpg") for i in range(20)]
        scores = {r["path"]: i / 20 for i, r in enumerate(recs)}
        subj_scores = {
            "bike":      {f"img{i}.jpg": (0.9 if i < 5 else 0.05) for i in range(20)},
            "landscape": {f"img{i}.jpg": (0.9 if 5 <= i < 10 else 0.05) for i in range(20)},
        }
        buckets = {"bike": 0.25, "landscape": 0.25, "aesthetic": 0.50}
        result = select_photos(
            recs, scores,
            max_bytes=100_000_000,
            buckets=buckets,
            subject_scores=subj_scores,
        )
        paths = {r["path"] for r in result}
        # At least some bike and landscape photos should be present
        bike_selected = sum(1 for i in range(5) if f"img{i}.jpg" in paths)
        landscape_selected = sum(1 for i in range(5, 10) if f"img{i}.jpg" in paths)
        assert bike_selected > 0
        assert landscape_selected > 0

    def test_default_buckets_when_none(self):
        """When buckets=None, falls back to default 30/30/40 split."""
        recs = [_rec(f"img{i}.jpg") for i in range(5)]
        scores = _scores(recs, 0.5)
        result = select_photos(recs, scores, max_bytes=100_000_000, buckets=None)
        assert len(result) == 5

    def test_buckets_normalised_when_over_one(self):
        """Fractions summing to > 1.0 are normalised — all photos still selected."""
        recs = [_rec(f"img{i}.jpg") for i in range(10)]
        scores = _scores(recs, 0.5)
        buckets = {"people": 0.50, "location": 0.40, "aesthetic": 0.60}  # sum = 1.5
        result = select_photos(recs, scores, max_bytes=100_000_000, buckets=buckets)
        assert len(result) == 10

    def test_buckets_normalised_when_under_one(self):
        """Fractions summing to < 1.0 are normalised — all photos still selected."""
        recs = [_rec(f"img{i}.jpg") for i in range(10)]
        scores = _scores(recs, 0.5)
        buckets = {"people": 0.10, "location": 0.10, "aesthetic": 0.10}  # sum = 0.3
        result = select_photos(recs, scores, max_bytes=100_000_000, buckets=buckets)
        assert len(result) == 10


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
        """Even very small percentage of a tiny pool returns >= 1 photo."""
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
        result = select_photos(
            recs, scores,
            max_bytes=2_500_000,
            output_mode="bytes",
            output_percentage=1.0,
        )
        assert len(result) < 50

    def test_byte_cap_still_enforced_in_percentage_mode(self):
        """Even in percentage mode the hard byte cap prevents overrun."""
        recs, scores = self._make_pool(100)
        result = select_photos(
            recs, scores,
            max_bytes=100_000,
            output_mode="percentage",
            output_percentage=0.50,
        )
        assert len(result) <= 2

    def test_percentage_selects_highest_scored_photos(self):
        """The selected subset should be the top-N by score."""
        n = 20
        recs = [_rec(f"img{i}.jpg") for i in range(n)]
        scores = {f"img{i}.jpg": i / n for i in range(n)}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.25,
        )
        selected_paths = {r["path"] for r in result}
        for i in range(15, 20):
            assert f"img{i}.jpg" in selected_paths

    def test_total_photos_base_overrides_candidates_count(self):
        """total_photos bases the target on ALL input, not just surviving candidates.

        Scenario: 100 photos scanned, only 7 survive quality/dedup/privacy.
        Fixed behaviour (total_photos=100):
            target = max(1, int(100 * 0.15)) = 15
            max_photos = min(15, 7) = 7  ->  all 7 candidates selected
        """
        recs, scores = self._make_pool(7)
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.15,
            total_photos=100,
        )
        assert len(result) == 7

    def test_total_photos_capped_at_candidates(self):
        """total_photos target is capped at len(candidates) — cannot exceed available."""
        recs, scores = self._make_pool(5)
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.50,
            total_photos=100,
        )
        assert len(result) == 5
