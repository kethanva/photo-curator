"""
Unit tests for src/aesthetic.py — aesthetic scoring without requiring CLIP/torch.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import src.aesthetic as aesthetic_mod
from src.aesthetic import _NEGATIVE_PROMPTS, _POSITIVE_PROMPTS, batch_score, score_from_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _fake_pos() -> np.ndarray:
    v = np.ones(512, dtype=np.float32)
    return v / np.linalg.norm(v)


def _fake_neg() -> np.ndarray:
    v = -np.ones(512, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Prompt bank sanity
# ---------------------------------------------------------------------------

class TestPromptBank:
    def test_positive_prompts_not_empty(self):
        assert len(_POSITIVE_PROMPTS) > 0

    def test_negative_prompts_not_empty(self):
        assert len(_NEGATIVE_PROMPTS) > 0

    def test_prompts_are_strings(self):
        for p in _POSITIVE_PROMPTS + _NEGATIVE_PROMPTS:
            assert isinstance(p, str) and len(p) > 0


# ---------------------------------------------------------------------------
# score_from_embedding tests (mocked prompt vectors)
# ---------------------------------------------------------------------------

class TestScoreFromEmbedding:
    def test_none_returns_half(self):
        assert score_from_embedding(None) == pytest.approx(0.5)

    def test_wrong_dim_returns_half(self):
        emb = np.ones(256, dtype=np.float32)
        assert score_from_embedding(emb) == pytest.approx(0.5)

    def test_score_in_zero_one_range(self):
        emb = _make_emb(0)
        with patch.object(aesthetic_mod, "_pos_vec", _fake_pos()), \
             patch.object(aesthetic_mod, "_neg_vec", _fake_neg()):
            score = score_from_embedding(emb)
        assert 0.0 <= score <= 1.0

    def test_positive_aligned_emb_high_score(self):
        """Embedding aligned with positive prompt → high aesthetic score."""
        emb = _fake_pos()  # same direction as pos_vec
        with patch.object(aesthetic_mod, "_pos_vec", _fake_pos()), \
             patch.object(aesthetic_mod, "_neg_vec", _fake_neg()):
            score = score_from_embedding(emb)
        assert score > 0.6

    def test_negative_aligned_emb_low_score(self):
        """Embedding aligned with negative prompt → low aesthetic score."""
        emb = _fake_neg()
        with patch.object(aesthetic_mod, "_pos_vec", _fake_pos()), \
             patch.object(aesthetic_mod, "_neg_vec", _fake_neg()):
            score = score_from_embedding(emb)
        assert score < 0.4

    def test_exception_returns_half(self):
        """If _build_prompt_vectors raises, fallback to 0.5."""
        emb = _make_emb(0)
        with patch.object(aesthetic_mod, "_pos_vec", None), \
             patch.object(aesthetic_mod, "_neg_vec", None), \
             patch("src.aesthetic._build_prompt_vectors", side_effect=RuntimeError("boom")):
            score = score_from_embedding(emb, use_laion=False)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# batch_score tests
# ---------------------------------------------------------------------------

class TestBatchScore:
    def test_empty_array_returns_empty(self):
        embs = np.empty((0, 512), dtype=np.float32)
        result = batch_score(embs)
        assert result.shape == (0,)

    def test_output_shape_matches_input(self):
        embs = np.stack([_make_emb(i) for i in range(5)])
        with patch.object(aesthetic_mod, "_pos_vec", _fake_pos()), \
             patch.object(aesthetic_mod, "_neg_vec", _fake_neg()):
            result = batch_score(embs)
        assert result.shape == (5,)

    def test_output_dtype_float32(self):
        embs = np.stack([_make_emb(i) for i in range(3)])
        with patch.object(aesthetic_mod, "_pos_vec", _fake_pos()), \
             patch.object(aesthetic_mod, "_neg_vec", _fake_neg()):
            result = batch_score(embs)
        assert result.dtype == np.float32

    def test_all_scores_in_range(self):
        embs = np.stack([_make_emb(i) for i in range(4)])
        with patch.object(aesthetic_mod, "_pos_vec", _fake_pos()), \
             patch.object(aesthetic_mod, "_neg_vec", _fake_neg()):
            result = batch_score(embs)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
