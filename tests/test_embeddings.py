"""
Unit tests for src/embeddings.py — CLIP embedding extraction.
torch/CLIP are mocked so tests run without the heavy ML stack.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Stub out torch and clip before importing the module under test
_torch_stub = MagicMock()
_torch_stub.no_grad.return_value.__enter__ = lambda s, *a: None
_torch_stub.no_grad.return_value.__exit__ = lambda s, *a: None
_torch_stub.device = MagicMock(return_value=MagicMock())
_torch_stub.backends.mps.is_available.return_value = False
_torch_stub.cuda.is_available.return_value = False

sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("clip", MagicMock())

import src.embeddings as emb_mod
from src.embeddings import cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity — no dependencies
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_is_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_is_zero(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_is_minus_one(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_returns_float(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = cosine_similarity(a, b)
        assert isinstance(result, float)

    def test_unnormalised_vectors_correct(self):
        a = np.array([3.0, 0.0])
        b = np.array([0.0, 4.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_symmetry(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


# ---------------------------------------------------------------------------
# extract — mocked torch + CLIP
# ---------------------------------------------------------------------------

def _mock_torch_setup():
    """Return a mock (tensor, model, preprocess, device) tuple."""
    device = MagicMock()

    # Model returns a fake embedding tensor
    fake_emb = np.ones((1, 512), dtype=np.float32) / np.sqrt(512)
    fake_tensor = MagicMock()
    fake_tensor.norm.return_value = MagicMock()
    fake_tensor.__truediv__ = lambda self, other: fake_tensor
    fake_tensor.cpu.return_value.numpy.return_value = fake_emb

    model = MagicMock()
    model.encode_image.return_value = fake_tensor

    preprocess = MagicMock(return_value=MagicMock())
    preprocess.return_value.unsqueeze.return_value.to.return_value = MagicMock()

    return model, preprocess, device


class TestExtract:
    def test_exception_returns_zero_vector(self):
        """If preprocess raises, extract falls back to zero vector."""
        img = Image.new("RGB", (224, 224))
        model = MagicMock()
        preprocess = MagicMock(side_effect=RuntimeError("preprocess failed"))
        device = MagicMock()

        result = emb_mod.extract(img, model, preprocess, device)

        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)
        assert np.all(result == 0.0)


class TestBatchExtract:
    def test_empty_input_returns_empty(self):
        # batch_extract returns early before importing torch for empty list
        # We need torch for the non-empty path, which is already stubbed via sys.modules
        result = emb_mod.batch_extract([])
        assert result.shape == (0, 512)

    def test_empty_dtype_float32(self):
        result = emb_mod.batch_extract([])
        assert result.dtype == np.float32
