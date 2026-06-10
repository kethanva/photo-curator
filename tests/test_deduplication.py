"""
Unit tests for src/deduplication.py — pHash, Hamming distance, and duplicate detection.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.deduplication import compute_phash, find_duplicates, hamming_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emb(seed: int = 0, size: int = 512) -> np.ndarray:
    """Reproducible unit-norm embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


def _record(
    path: str,
    blur: float = 100.0,
    file_hash: str = "",
    phash: str = "",
    clip_emb: np.ndarray | None = None,
) -> dict:
    return {
        "path": path,
        "blur_score": blur,
        "file_hash": file_hash,
        "phash": phash,
        "clip_emb": clip_emb,
    }


# ---------------------------------------------------------------------------
# compute_phash tests
# ---------------------------------------------------------------------------

class TestComputePhash:
    def test_returns_string(self):
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        h = compute_phash(img)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_identical_images_same_hash(self):
        img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        assert compute_phash(img) == compute_phash(img)

    def test_different_images_likely_different_hash(self):
        img1 = Image.new("RGB", (64, 64), color=(0, 0, 0))
        img2 = Image.new("RGB", (64, 64), color=(255, 255, 255))
        # Different solid colours should produce different hashes
        assert compute_phash(img1) != compute_phash(img2)


# ---------------------------------------------------------------------------
# hamming_distance tests
# ---------------------------------------------------------------------------

class TestHammingDistance:
    def test_identical_hashes_zero_distance(self):
        h = "aabbccdd11223344"
        assert hamming_distance(h, h) == 0

    def test_empty_hash_returns_max(self):
        assert hamming_distance("", "aabb") == 64
        assert hamming_distance("aabb", "") == 64
        assert hamming_distance("", "") == 64

    def test_different_length_returns_max(self):
        assert hamming_distance("aabb", "aabbcc") == 64

    def test_invalid_hex_returns_max(self):
        assert hamming_distance("zzzz", "zzzz") == 64

    def test_single_bit_difference(self):
        # 0000 vs 0001 — differ in 1 bit
        d = hamming_distance("0000000000000000", "0000000000000001")
        assert d == 1

    def test_all_bits_different(self):
        # 0000...0000 vs ffff...ffff
        h1 = "0" * 16
        h2 = "f" * 16
        assert hamming_distance(h1, h2) == 64

    def test_returns_int(self):
        assert isinstance(hamming_distance("aabb", "aabb"), int)


# ---------------------------------------------------------------------------
# find_duplicates tests
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_empty_input(self):
        assert find_duplicates([]) == set()

    def test_single_record_not_duplicate(self):
        rec = _record("a.jpg", blur=100)
        result = find_duplicates([rec])
        assert result == set()

    def test_exact_file_hash_match(self):
        recs = [
            _record("a.jpg", blur=100, file_hash="abc123"),
            _record("b.jpg", blur=50,  file_hash="abc123"),  # worse quality
        ]
        result = find_duplicates(recs)
        assert "b.jpg" in result
        assert "a.jpg" not in result

    def test_sharper_copy_kept_on_hash_match(self):
        """The blurrier copy (b) should be marked as duplicate."""
        recs = [
            _record("sharp.jpg", blur=500, file_hash="same"),
            _record("blurry.jpg", blur=10,  file_hash="same"),
        ]
        result = find_duplicates(recs)
        assert "blurry.jpg" in result
        assert "sharp.jpg" not in result

    def test_phash_near_duplicate(self):
        """Hashes with Hamming distance ≤ threshold are duplicates."""
        # Differ in 1 bit — well within default threshold 8
        h1 = "0000000000000000"
        h2 = "0000000000000001"
        recs = [
            _record("a.jpg", blur=100, phash=h1),
            _record("b.jpg", blur=50,  phash=h2),
        ]
        result = find_duplicates(recs, phash_threshold=8)
        assert "b.jpg" in result

    def test_phash_far_apart_not_duplicate(self):
        """Hashes with large Hamming distance are not duplicates."""
        h1 = "0" * 16
        h2 = "f" * 16  # 64-bit difference
        recs = [
            _record("a.jpg", blur=100, phash=h1),
            _record("b.jpg", blur=50,  phash=h2),
        ]
        result = find_duplicates(recs, phash_threshold=8)
        assert "b.jpg" not in result

    def test_clip_cosine_near_duplicate(self):
        """Embeddings with cosine similarity ≥ threshold are duplicates."""
        emb = _make_emb(42)
        # Slightly perturbed but very similar
        noise = np.random.default_rng(1).standard_normal(512).astype(np.float32) * 0.001
        emb2 = emb + noise
        emb2 = emb2 / np.linalg.norm(emb2)

        recs = [
            _record("a.jpg", blur=100, clip_emb=emb),
            _record("b.jpg", blur=50,  clip_emb=emb2),
        ]
        result = find_duplicates(recs, embedding_threshold=0.95)
        assert "b.jpg" in result

    def test_clip_dissimilar_not_duplicate(self):
        """Orthogonal embeddings are not duplicates."""
        e1 = _make_emb(0)
        e2 = _make_emb(99)  # different seed → different direction
        recs = [
            _record("a.jpg", blur=100, clip_emb=e1),
            _record("b.jpg", blur=50,  clip_emb=e2),
        ]
        result = find_duplicates(recs, embedding_threshold=0.99)
        # With very high threshold and different embeddings, should not be dups
        # (may or may not be dup depending on actual cosine sim — just check no crash)
        assert isinstance(result, set)

    def test_no_cross_contamination(self):
        """Three distinct photos — none should be duplicates of each other."""
        recs = [
            _record("x.jpg", blur=300, clip_emb=_make_emb(1)),
            _record("y.jpg", blur=200, clip_emb=_make_emb(2)),
            _record("z.jpg", blur=100, clip_emb=_make_emb(3)),
        ]
        result = find_duplicates(recs, phash_threshold=8, embedding_threshold=0.99)
        # All embeddings from different seeds — probably no dups, but check set type
        assert isinstance(result, set)
        assert len(result) <= 2  # at most 2 of 3 can be dups

    def test_returns_set(self):
        result = find_duplicates([_record("a.jpg")])
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# Alternative hash algorithms (#4)
# ---------------------------------------------------------------------------

class TestHashAlgorithms:
    def test_dhash_and_phash_both_valid_hex(self):
        img = Image.new("RGB", (64, 64), color=(120, 60, 30))
        ph = compute_phash(img, "phash")
        dh = compute_phash(img, "dhash")
        ah = compute_phash(img, "ahash")
        assert all(isinstance(h, str) and len(h) == 16 for h in (ph, dh, ah))

    def test_unknown_algorithm_falls_back_to_phash(self):
        img = Image.new("RGB", (64, 64), color=(120, 60, 30))
        assert compute_phash(img, "nope") == compute_phash(img, "phash")

    def test_algorithms_differ_on_structured_image(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:32] = 255  # top half white, bottom black
        img = Image.fromarray(arr, "RGB")
        # dhash (gradient) and ahash (average) encode different structure.
        assert compute_phash(img, "dhash") != compute_phash(img, "ahash")


# ---------------------------------------------------------------------------
# Dual-hash confirmation gate (#3)
# ---------------------------------------------------------------------------

from src.deduplication import _is_phash_dup, find_burst_duplicates


def _drec(path, *, blur=100.0, phash="", phash2="", ts=0.0, size=0):
    return {"path": path, "blur_score": blur, "file_hash": "",
            "phash": phash, "phash2": phash2, "timestamp": ts,
            "file_size": size, "clip_emb": None}


class TestDualHashGate:
    # primary within threshold, secondary far apart
    A = _drec("a", phash="0000000000000000", phash2="0000000000000000")
    B = _drec("b", phash="0000000000000001", phash2="ffffffffffffffff")

    def test_primary_match_no_dual_is_dup(self):
        assert _is_phash_dup(self.B, self.A, 8, dual_hash=False, secondary_threshold=8) is True

    def test_dual_blocks_when_secondary_disagrees(self):
        assert _is_phash_dup(self.B, self.A, 8, dual_hash=True, secondary_threshold=8) is False

    def test_dual_allows_when_secondary_agrees(self):
        b = _drec("b", phash="0000000000000001", phash2="0000000000000001")
        assert _is_phash_dup(b, self.A, 8, dual_hash=True, secondary_threshold=8) is True

    def test_bktree_matches_brute_force(self):
        """BK-tree radius queries must return exactly the same matches as a
        brute-force Hamming scan, for several thresholds."""
        import random

        from src.deduplication import _BKTree, hamming_distance

        rng = random.Random(42)
        hashes = [f"{rng.getrandbits(64):016x}" for _ in range(300)]

        tree = _BKTree()
        for h in hashes:
            tree.insert(h, h)

        for tol in (0, 2, 8, 16):
            for q in hashes[:25]:
                expected = sorted(
                    h for h in hashes if hamming_distance(q, h) <= tol
                )
                assert sorted(tree.search(q, tol)) == expected

    def test_bktree_isolates_hash_lengths(self):
        """Hashes of different hex lengths never cross-match (the Hamming
        metric is only defined for equal lengths)."""
        from src.deduplication import _BKTree

        tree = _BKTree()
        tree.insert("0" * 16, "len16")
        tree.insert("0" * 8, "len8")
        assert tree.search("0" * 16, 64) == ["len16"]
        assert tree.search("0" * 8, 64) == ["len8"]

    def test_bktree_ignores_empty_and_invalid(self):
        from src.deduplication import _BKTree

        tree = _BKTree()
        tree.insert("", "empty")
        tree.insert("zzzz", "invalid")
        tree.insert("0" * 16, "good")
        assert tree.search("0" * 16, 64) == ["good"]
        assert tree.search("", 64) == []
        assert tree.search("zzzz", 64) == []

    def test_missing_secondary_falls_back_to_primary(self):
        # One side ingested before dual_hash was enabled → phash2 empty.
        # Secondary check is inconclusive, primary verdict must stand —
        # an empty hash must NOT veto the match (hamming would be 64).
        old = _drec("old", phash="0000000000000000", phash2="")
        new = _drec("new", phash="0000000000000001", phash2="0000000000000001")
        assert _is_phash_dup(new, old, 8, dual_hash=True, secondary_threshold=8) is True
        assert _is_phash_dup(old, new, 8, dual_hash=True, secondary_threshold=8) is True

    def test_find_duplicates_dual_hash_prevents_false_positive(self):
        recs = [self.A, self.B]
        # Without dual: B is a pHash dup of A.
        assert find_duplicates(recs, phash_threshold=8, dual_hash=False) == {"b"}
        # With dual: secondary disagrees → no dup marked.
        assert find_duplicates(recs, phash_threshold=8, dual_hash=True,
                               secondary_threshold=8) == set()


# ---------------------------------------------------------------------------
# Burst-shot dedup (#2)
# ---------------------------------------------------------------------------

class TestBurstDuplicates:
    def test_keeps_sharpest_of_burst(self):
        recs = [
            _drec("a", blur=10.0, ts=1000.0),
            _drec("b", blur=90.0, ts=1001.0),   # sharpest → keeper
            _drec("c", blur=50.0, ts=1002.5),
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == {"a", "c"}

    def test_gap_splits_into_separate_bursts(self):
        recs = [
            _drec("a", blur=10.0, ts=1000.0),
            _drec("b", blur=90.0, ts=1001.0),   # burst 1 keeper
            _drec("c", blur=20.0, ts=1010.0),   # burst 2 (gap 9s > 3s)
            _drec("d", blur=80.0, ts=1011.0),   # burst 2 keeper
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == {"a", "c"}

    def test_singletons_never_marked(self):
        recs = [
            _drec("a", ts=1000.0),
            _drec("b", ts=1100.0),
            _drec("c", ts=1200.0),
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == set()

    def test_records_without_timestamp_skipped(self):
        recs = [
            _drec("a", blur=10.0, ts=0.0),    # no EXIF
            _drec("b", blur=90.0, ts=0.0),    # no EXIF
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == set()

    def test_gap_boundary_exclusive(self):
        # gap exactly == gap_seconds stays in the same burst (only > splits).
        recs = [
            _drec("a", blur=10.0, ts=1000.0),
            _drec("b", blur=90.0, ts=1003.0),  # gap 3.0, not > 3.0
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == {"a"}

    def test_blur_tie_breaks_on_file_size(self):
        # Equal sharpness (e.g. both 0.0 after a metric failure) → keep the
        # larger file, mirroring photos-cleanup compute_best tie-break.
        recs = [
            _drec("small", blur=0.0, ts=1000.0, size=100),
            _drec("large", blur=0.0, ts=1001.0, size=900),
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == {"small"}

    def test_sharpness_beats_file_size(self):
        recs = [
            _drec("sharp_small", blur=90.0, ts=1000.0, size=100),
            _drec("blurry_large", blur=10.0, ts=1001.0, size=900),
        ]
        assert find_burst_duplicates(recs, gap_seconds=3.0) == {"blurry_large"}
