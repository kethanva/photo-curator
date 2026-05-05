"""
Unit tests for src/selection.py — dynamic bucket photo selection and output writing.
"""

from __future__ import annotations

import json
import tempfile
import time
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
    timestamp: int = 0,
    blur_score: float = 100.0,
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
        "timestamp": timestamp,
        "aesthetic_score": 0.5,
        "blur_score": blur_score,
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

    def test_percentage_backfill_respects_diversity_caps(self):
        """Backfill must NOT bypass diversity caps to hit the count target.

        Spread across distinct days/locations is the priority — under-deliver
        rather than dump score-clustered photos to fill the quota.

        Setup: 30 photos across 2 days, 2 hours, 3 clusters with caps of 5%.
        Target = 15. Diversity ceiling ≈ max(2 days × 1, 2 hours × 1, 3 clusters
        × 1) = 2–3. Result must be ≤ that ceiling, not 15.
        """
        base_ts = 1_700_000_000
        recs = [
            _rec(
                f"img{i}.jpg",
                cluster_id=i % 3,
                timestamp=base_ts + (i % 2) * 3600,
            )
            for i in range(30)
        ]
        scores = {r["path"]: 1.0 - (i / 100.0) for i, r in enumerate(recs)}
        result = select_photos(
            recs,
            scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.50,
            max_per_cluster_pct=0.05,
            max_per_day_pct=0.05,
            max_per_hour_pct=0.05,
        )
        assert len(result) <= 3, (
            "Diversity caps must constrain backfill — got "
            f"{len(result)} photos vs ceiling of 3"
        )

    def test_percentage_backfill_spreads_across_distinct_days(self):
        """Round-robin backfill should pull from many distinct days even
        when the highest-scored photos are concentrated on one day."""
        base_ts = 1_700_000_000
        recs = []
        # 5 days × 10 photos each = 50; day 0 has the highest scores.
        for day_idx in range(5):
            for slot in range(10):
                recs.append(
                    _rec(
                        f"d{day_idx}_p{slot}.jpg",
                        cluster_id=day_idx,
                        timestamp=base_ts + day_idx * 86_400 + slot * 60,
                    )
                )
        scores = {
            r["path"]: 0.9 - (i * 0.001)
            for i, r in enumerate(recs)
        }
        result = select_photos(
            recs,
            scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=0.20,           # target = 10
            max_per_cluster_pct=0.30,         # ≤ 3 per cluster
            max_per_day_pct=0.30,             # ≤ 3 per day
            max_per_hour_pct=1.0,
        )
        days_covered = {
            time.localtime(r["timestamp"]).tm_yday for r in result
        }
        assert len(days_covered) >= 4, (
            f"Expected spread across ≥4 days, got {len(days_covered)}: "
            f"{days_covered}"
        )

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


class TestGlobalTemporalGap:
    """Cross-cluster / noise-singleton temporal proximity guard.

    Replicates the user-reported failure mode: filenames like
    IMG_20190413_102752 / IMG_20190413_102755 (3 seconds apart) ending up
    in DBSCAN noise (cid=-1) or in adjacent clusters, then both being
    selected because per-cluster spacing only fires for cid>=0 same-cluster
    pairs.
    """

    def test_burst_in_noise_blocked_by_global_gap(self):
        """Two burst photos in DBSCAN noise (cid=-1) must not both be selected."""
        ts = 1_555_148_872  # 2019-04-13 10:27:52 UTC
        recs = [
            _rec("IMG_20190413_102752.jpg", cluster_id=-1, timestamp=ts),
            _rec("IMG_20190413_102755.jpg", cluster_id=-1, timestamp=ts + 3),
            _rec("IMG_20190413_103000.jpg", cluster_id=-1, timestamp=ts + 128),
        ]
        scores = {r["path"]: 1.0 - i * 0.01 for i, r in enumerate(recs)}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
            global_min_gap_seconds=60.0,
            max_per_day_pct=1.0,
            max_per_hour_pct=1.0,
            min_pool_fraction=0.0,
        )
        paths = {r["path"] for r in result}
        # Either of the two close-burst photos may be selected, but not both.
        burst_pair = {"IMG_20190413_102752.jpg", "IMG_20190413_102755.jpg"}
        assert len(paths & burst_pair) == 1, (
            f"Both burst photos selected — global gap not enforced. Got: {paths}"
        )
        # The far-apart photo (>60s away) must be allowed in.
        assert "IMG_20190413_103000.jpg" in paths

    def test_burst_across_adjacent_clusters_blocked(self):
        """Two burst photos that DBSCAN split into different clusters are
        still blocked by the global gap — the per-cluster spacing alone
        only checks within the same cid."""
        ts = 1_550_939_782  # 2019-02-23 16:36:22 UTC
        recs = [
            _rec("IMG_20190223_163622.jpg", cluster_id=0, timestamp=ts),
            _rec("IMG_20190223_163624.jpg", cluster_id=1, timestamp=ts + 2),
            _rec("IMG_20190223_164100.jpg", cluster_id=2, timestamp=ts + 280),
        ]
        scores = {r["path"]: 1.0 - i * 0.01 for i, r in enumerate(recs)}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
            global_min_gap_seconds=60.0,
            max_per_cluster_pct=1.0,
            max_per_day_pct=1.0,
            max_per_hour_pct=1.0,
            min_pool_fraction=0.0,
        )
        paths = {r["path"] for r in result}
        burst_pair = {"IMG_20190223_163622.jpg", "IMG_20190223_163624.jpg"}
        assert len(paths & burst_pair) == 1, (
            f"Cross-cluster burst not blocked by global gap. Got: {paths}"
        )

    def test_three_photo_burst_collapses_to_one(self):
        """20190518_152454 / 152510 / 152514 — within 20s of each other.
        Only one should be selected regardless of cluster assignment."""
        ts = 1_558_193_094  # 2019-05-18 15:24:54 UTC
        recs = [
            _rec("20190518_152454.jpg", cluster_id=-1, timestamp=ts),
            _rec("20190518_152510.jpg", cluster_id=0, timestamp=ts + 16),
            _rec("20190518_152514.jpg", cluster_id=-1, timestamp=ts + 20),
            _rec("20190518_153500.jpg", cluster_id=-1, timestamp=ts + 606),
        ]
        scores = {r["path"]: 1.0 - i * 0.01 for i, r in enumerate(recs)}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
            global_min_gap_seconds=60.0,
            max_per_cluster_pct=1.0,
            max_per_day_pct=1.0,
            max_per_hour_pct=1.0,
            min_pool_fraction=0.0,
        )
        paths = {r["path"] for r in result}
        burst = {
            "20190518_152454.jpg",
            "20190518_152510.jpg",
            "20190518_152514.jpg",
        }
        assert len(paths & burst) == 1, (
            f"Burst of 3 photos within 20s collapsed to {len(paths & burst)}: {paths}"
        )
        assert "20190518_153500.jpg" in paths

    def test_photos_far_apart_both_selected(self):
        """Photos comfortably separated in time must both pass."""
        ts = 1_700_000_000
        recs = [
            _rec("a.jpg", cluster_id=-1, timestamp=ts),
            _rec("b.jpg", cluster_id=-1, timestamp=ts + 120),
            _rec("c.jpg", cluster_id=-1, timestamp=ts + 240),
        ]
        scores = {r["path"]: 0.9 for r in recs}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
            global_min_gap_seconds=60.0,
            max_per_day_pct=1.0,
            max_per_hour_pct=1.0,
            min_pool_fraction=0.0,
        )
        assert len(result) == 3

    def test_global_gap_disabled_allows_burst(self):
        """global_min_gap_seconds=0 disables the guard (back-compat)."""
        ts = 1_700_000_000
        recs = [
            _rec("a.jpg", cluster_id=-1, timestamp=ts),
            _rec("b.jpg", cluster_id=-1, timestamp=ts + 3),
        ]
        scores = {r["path"]: 0.9 for r in recs}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
            global_min_gap_seconds=0.0,
            max_per_day_pct=1.0,
            max_per_hour_pct=1.0,
            min_pool_fraction=0.0,
        )
        assert len(result) == 2

    def test_backfill_path_respects_global_gap(self):
        """The round-robin backfill (which historically bypassed _add) must
        also enforce the global gap. Setup: 4 burst photos within 30s plus
        one photo >60s away. With a target of 5 and loose diversity caps,
        the backfill kicks in but only one of the burst can land."""
        ts = 1_700_000_000
        recs = [
            _rec("burst_a.jpg", cluster_id=-1, timestamp=ts),
            _rec("burst_b.jpg", cluster_id=-1, timestamp=ts + 5),
            _rec("burst_c.jpg", cluster_id=-1, timestamp=ts + 12),
            _rec("burst_d.jpg", cluster_id=-1, timestamp=ts + 25),
            _rec("far.jpg", cluster_id=-1, timestamp=ts + 600),
        ]
        scores = {r["path"]: 1.0 - i * 0.001 for i, r in enumerate(recs)}
        result = select_photos(
            recs, scores,
            max_bytes=500_000_000,
            output_mode="percentage",
            output_percentage=1.0,
            global_min_gap_seconds=60.0,
            max_per_day_pct=1.0,
            max_per_hour_pct=1.0,
            max_per_cluster_pct=1.0,
            min_pool_fraction=0.0,
        )
        paths = {r["path"] for r in result}
        burst = {"burst_a.jpg", "burst_b.jpg", "burst_c.jpg", "burst_d.jpg"}
        assert len(paths & burst) == 1, (
            f"Backfill leaked multiple burst photos. Got: {paths}"
        )
        assert "far.jpg" in paths


class TestBlurHardGate:
    """Blur is a hard exclusion, applied to BOTH strict and fallback pools.

    Reproduces the user-reported leak where a photo with quality_pass=0
    (failed blur threshold) got admitted via the soft-pool fallback because
    it had a high composite score from aesthetic / face signals.
    """

    def test_blur_failed_photo_never_admitted_via_fallback(self):
        """Photo failing blur AND in fallback pool must be excluded."""
        good = _rec("good.jpg", quality_pass=1, blur_score=120.0)
        blurry = _rec(
            "IMG_20190101_114159.jpg",
            quality_pass=0, blur_score=12.0,  # well below threshold
        )
        # Composite score puts the blurry one ahead — without the gate,
        # it would be admitted first.
        scores = {"good.jpg": 0.10, "IMG_20190101_114159.jpg": 0.99}
        result = select_photos(
            [good, blurry], scores,
            max_bytes=500_000_000,
            output_mode="percentage", output_percentage=1.0,
            min_pool_fraction=1.0,        # force fallback admission
            total_photos=10,              # makes pool_floor large
            min_blur_score=50.0,
        )
        paths = {r["path"] for r in result}
        assert "IMG_20190101_114159.jpg" not in paths
        assert "good.jpg" in paths

    def test_exposure_failed_photo_can_still_be_admitted(self):
        """Exposure / resolution failures remain salvageable via fallback —
        only blur is the hard gate."""
        good = _rec("good.jpg", quality_pass=1, blur_score=120.0)
        # Failed quality_pass for non-blur reason; blur is fine.
        underexposed = _rec(
            "dim.jpg", quality_pass=0, blur_score=80.0,
        )
        scores = {"good.jpg": 0.5, "dim.jpg": 0.4}
        result = select_photos(
            [good, underexposed], scores,
            max_bytes=500_000_000,
            output_mode="percentage", output_percentage=1.0,
            min_pool_fraction=1.0,
            total_photos=10,
            min_blur_score=50.0,
        )
        paths = {r["path"] for r in result}
        assert "good.jpg" in paths
        assert "dim.jpg" in paths

    def test_blur_failed_photo_excluded_even_when_quality_pass_is_one(self):
        """Defence-in-depth: a cached row marked quality_pass=1 under an
        older / looser threshold must still be excluded if its stored
        blur_score is below the current min_blur_score."""
        sharp = _rec("sharp.jpg", quality_pass=1, blur_score=120.0)
        # quality_pass=1 lying about itself — blur_score is below floor.
        legacy_blurry = _rec(
            "legacy.jpg", quality_pass=1, blur_score=15.0,
        )
        scores = {"sharp.jpg": 0.5, "legacy.jpg": 0.95}
        result = select_photos(
            [sharp, legacy_blurry], scores,
            max_bytes=500_000_000,
            output_mode="percentage", output_percentage=1.0,
            min_blur_score=50.0,
        )
        paths = {r["path"] for r in result}
        assert "legacy.jpg" not in paths
        assert "sharp.jpg" in paths

    def test_blur_gate_disabled_back_compat(self):
        """min_blur_score=0 fully disables the gate (back-compat for callers
        that don't pass the new parameter)."""
        blurry = _rec("blurry.jpg", quality_pass=0, blur_score=5.0)
        scores = {"blurry.jpg": 0.5}
        result = select_photos(
            [blurry], scores,
            max_bytes=500_000_000,
            output_mode="percentage", output_percentage=1.0,
            min_pool_fraction=1.0,
            total_photos=10,
            min_blur_score=0.0,
        )
        assert len(result) == 1


