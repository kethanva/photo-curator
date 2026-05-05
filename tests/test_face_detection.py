"""
Unit tests for src/face_detection.py — face detection and face_score.

The MTCNN/FaceNet model functions are mocked; the pure-Python face_score
function is tested directly.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Stub torch and facenet_pytorch before importing the module under test
_torch_stub = sys.modules.get("torch", MagicMock())
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("facenet_pytorch", MagicMock())

import src.face_detection as fd_mod
from src.face_detection import face_score


# ---------------------------------------------------------------------------
# face_score tests (pure Python — no ML required)
# ---------------------------------------------------------------------------

class TestFaceScore:
    def test_zero_faces_returns_zero(self):
        assert face_score(0) == pytest.approx(0.0)

    def test_one_face_returns_positive(self):
        score = face_score(1)
        assert score > 0.0
        assert score <= 1.0

    def test_score_increases_with_face_count(self):
        s1 = face_score(1)
        s3 = face_score(3)
        s6 = face_score(6)
        assert s1 < s3
        assert s3 <= s6

    def test_score_capped_at_one(self):
        """Very large face counts should still yield ≤ 1."""
        assert face_score(100) <= 1.0
        assert face_score(1000) <= 1.0

    def test_custom_max_faces(self):
        """With a smaller max_faces, score rises faster."""
        score_low_max = face_score(3, max_faces=3)
        score_high_max = face_score(3, max_faces=10)
        assert score_low_max >= score_high_max

    def test_returns_float(self):
        assert isinstance(face_score(2), float)

    def test_single_face_below_max(self):
        """A group photo should score higher than a solo shot."""
        solo = face_score(1)
        group = face_score(4)
        assert group > solo


# ---------------------------------------------------------------------------
# detect tests — exception path
# ---------------------------------------------------------------------------

class TestDetect:
    def test_exception_returns_zero_none(self):
        """If MTCNN raises, detect returns empty face data."""
        img = Image.new("RGB", (100, 100))

        mock_mtcnn = MagicMock()
        mock_mtcnn.side_effect = RuntimeError("MTCNN error")

        with patch.object(fd_mod, "_mtcnn", mock_mtcnn), \
             patch.object(fd_mod, "_facenet", MagicMock()), \
             patch.object(fd_mod, "_device", MagicMock()), \
             patch("src.face_detection.load_models",
                   return_value=(mock_mtcnn, MagicMock(), MagicMock())):
            count, emb, prominence, confidence = fd_mod.detect(img)

        assert count == 0
        assert emb is None
        assert prominence == 0.0
        assert confidence == 0.0

    def test_no_faces_returns_zero_none(self):
        """If MTCNN returns None (no faces), detect returns empty face data."""
        img = Image.new("RGB", (100, 100))

        mock_mtcnn = MagicMock(return_value=None)
        mock_facenet = MagicMock()
        mock_device = MagicMock()

        with patch("src.face_detection.load_models",
                   return_value=(mock_mtcnn, mock_facenet, mock_device)):
            count, emb, prominence, confidence = fd_mod.detect(img)

        assert count == 0
        assert emb is None
        assert prominence == 0.0
        assert confidence == 0.0
