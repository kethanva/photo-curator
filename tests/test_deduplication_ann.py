"""
Tests for src/deduplication.find_duplicates_ann() — ANN-accelerated dedup.
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.deduplication import find_duplicates_ann


# ---------------------------------------------------------------------------
# Mock VectorStore
# ---------------------------------------------------------------------------

def _make_store(hits: List[Tuple[str, float]] | None = None):
    """
    Return a mock VectorStore whose search_clip() returns ``hits``.
    hits: list of (path, distance) tuples — distance is 1 - cosine_similarity.
    """
    store = MagicMock()
    if hits is not None:
        store.search_clip.return_value = hits
    else:
        store.search_clip.return_value = []
    return store


def _make_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _rec(path, blur=100.0, file_hash="", phash="", clip_emb=None):
    return {
        "path": path,
        "blur_score": blur,
        "file_hash": file_hash,
        "phash": phash,
        "clip_emb": clip_emb,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindDuplicatesAnn:
    def test_empty_input_returns_empty(self):
        store = _make_store()
        result = find_duplicates_ann([], store)
        assert result == set()

    def test_single_record_not_duplicate(self):
        store = _make_store()
        result = find_duplicates_ann([_rec("a.jpg")], store)
        assert result == set()

    def test_exact_file_hash_duplicate(self):
        store = _make_store()
        recs = [
            _rec("orig.jpg", blur=200, file_hash="abc"),
            _rec("dup.jpg",  blur=10,  file_hash="abc"),
        ]
        result = find_duplicates_ann(recs, store)
        assert "dup.jpg" in result
        assert "orig.jpg" not in result

    def test_phash_near_duplicate(self):
        """Two records with Hamming distance 1 should be duplicates."""
        store = _make_store(hits=[])
        h1 = "0000000000000000"
        h2 = "0000000000000001"
        recs = [
            _rec("a.jpg", blur=100, phash=h1),
            _rec("b.jpg", blur=50,  phash=h2),
        ]
        result = find_duplicates_ann(recs, store, phash_threshold=8)
        assert "b.jpg" in result

    def test_ann_cosine_near_duplicate(self):
        """Store returns the first photo as a near-match of the second → duplicate."""
        emb_a = _make_emb(0)
        emb_b = _make_emb(1)

        # Store will report emb_a (distance ≈ 0 → cosine ≈ 1.0) for emb_b's query
        store = _make_store(hits=[("a.jpg", 0.01)])  # dist 0.01 → sim 0.99
        recs = [
            _rec("a.jpg", blur=200, clip_emb=emb_a),
            _rec("b.jpg", blur=50,  clip_emb=emb_b),
        ]
        result = find_duplicates_ann(recs, store, embedding_threshold=0.95, ann_k=5)
        assert "b.jpg" in result
        assert "a.jpg" not in result

    def test_ann_below_threshold_not_duplicate(self):
        """Store returns a match with distance 0.5 → sim 0.5, below threshold 0.95."""
        emb_a = _make_emb(0)
        emb_b = _make_emb(1)

        store = _make_store(hits=[("a.jpg", 0.5)])  # sim = 0.5
        recs = [
            _rec("a.jpg", blur=200, clip_emb=emb_a),
            _rec("b.jpg", blur=50,  clip_emb=emb_b),
        ]
        result = find_duplicates_ann(recs, store, embedding_threshold=0.95)
        assert "b.jpg" not in result

    def test_no_ann_called_for_first_record(self):
        """The first record has no accepted set yet — ANN should not be called."""
        emb = _make_emb(0)
        store = _make_store()
        recs = [_rec("only.jpg", clip_emb=emb)]
        find_duplicates_ann(recs, store)
        store.search_clip.assert_not_called()

    def test_returns_set(self):
        store = _make_store()
        result = find_duplicates_ann([_rec("a.jpg")], store)
        assert isinstance(result, set)

    def test_sharper_copy_kept(self):
        """Blurrier copy should be marked as duplicate when hash matches."""
        store = _make_store()
        recs = [
            _rec("sharp.jpg", blur=500, file_hash="same_hash"),
            _rec("blurry.jpg", blur=10,  file_hash="same_hash"),
        ]
        result = find_duplicates_ann(recs, store)
        assert "blurry.jpg" in result
        assert "sharp.jpg" not in result

    def test_non_accepted_candidate_ignored_in_ann_results(self):
        """ANN returns a path that is NOT in the accepted set — should not mark as dup."""
        emb_a = _make_emb(0)
        emb_b = _make_emb(1)
        emb_c = _make_emb(2)

        # Store returns "c.jpg" (not yet in accepted set) for b's query
        store = _make_store(hits=[("c.jpg", 0.01)])
        recs = [
            _rec("a.jpg", blur=300, clip_emb=emb_a),
            _rec("b.jpg", blur=200, clip_emb=emb_b),
            _rec("c.jpg", blur=100, clip_emb=emb_c),
        ]
        result = find_duplicates_ann(recs, store, embedding_threshold=0.95)
        # b was compared against only accepted set {a} — c not accepted yet
        # So b should not be flagged as duplicate of the not-yet-accepted c
        assert "b.jpg" not in result
