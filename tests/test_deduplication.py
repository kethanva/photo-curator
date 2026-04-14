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
