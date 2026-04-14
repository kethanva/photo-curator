"""
Unit tests for src/sentiment.py — facial expression scoring.

The pure-Python helper functions (_ear, _smile_score) are tested directly.
The score_image() function is tested with a mocked mediapipe to avoid
needing the heavy ML dependency in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.sentiment import (
    _EAR_OPEN_THRESHOLD,
    _LEFT_EYE,
    _MOUTH_LEFT_CORNER,
    _MOUTH_LOWER_LIP,
    _MOUTH_RIGHT_CORNER,
    _MOUTH_UPPER_LIP,
    _RIGHT_EYE,
    _SMILE_SENSITIVITY,
    _ear,
    _smile_score,
    score_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _landmark(x: float = 0.0, y: float = 0.0) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    return lm


def _make_landmarks(n: int = 500) -> list:
    """Create a list of N zero-position landmarks."""
    return [_landmark(0.0, 0.0) for _ in range(n)]


# ---------------------------------------------------------------------------
# _ear tests
# ---------------------------------------------------------------------------

class TestEar:
    def test_open_eye_returns_positive(self):
        """Eye with vertical spread > 0 and horizontal spread > 0."""
        lm = _make_landmarks()
        # Indices for _LEFT_EYE: [362, 385, 387, 263, 373, 380]
        # outer=362, top1=385, top2=387, inner=263, bot1=373, bot2=380
        for i in _LEFT_EYE:
            lm[i] = _landmark(x=0.0, y=0.0)
        # Make outer and inner separated horizontally
        lm[_LEFT_EYE[0]] = _landmark(x=0.0, y=0.0)  # outer
        lm[_LEFT_EYE[3]] = _landmark(x=0.1, y=0.0)  # inner (horizontal spread)
        # Make top and bottom separated vertically
        lm[_LEFT_EYE[1]] = _landmark(x=0.05, y=-0.02)  # top1
        lm[_LEFT_EYE[2]] = _landmark(x=0.05, y=-0.02)  # top2
        lm[_LEFT_EYE[4]] = _landmark(x=0.05, y=0.02)   # bot1
        lm[_LEFT_EYE[5]] = _landmark(x=0.05, y=0.02)   # bot2

        result = _ear(lm, _LEFT_EYE)
        assert result > 0

    def test_closed_eye_near_zero(self):
        """Top and bottom landmarks at same y → EAR ≈ 0."""
        lm = _make_landmarks()
        for idx in _LEFT_EYE:
            lm[idx] = _landmark(x=float(_LEFT_EYE.index(idx)) * 0.05, y=0.5)
        # Override outer/inner for horizontal spread
        lm[_LEFT_EYE[0]] = _landmark(x=0.0, y=0.5)
        lm[_LEFT_EYE[3]] = _landmark(x=0.2, y=0.5)
        result = _ear(lm, _LEFT_EYE)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_returns_float(self):
        lm = _make_landmarks()
        result = _ear(lm, _LEFT_EYE)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _smile_score tests
# ---------------------------------------------------------------------------

class TestSmileScore:
    def test_smiling_returns_above_half(self):
        """Corners raised above lip centre → smiling."""
        lm = _make_landmarks()
        lm[_MOUTH_LEFT_CORNER]  = _landmark(y=0.3)   # corners high (small y)
        lm[_MOUTH_RIGHT_CORNER] = _landmark(y=0.3)
        lm[_MOUTH_UPPER_LIP]    = _landmark(y=0.5)   # lip centre lower
        lm[_MOUTH_LOWER_LIP]    = _landmark(y=0.55)
        score = _smile_score(lm)
        assert score > 0.5

    def test_neutral_returns_near_half(self):
        """Corners at same level as lip → neutral."""
        lm = _make_landmarks()
        lm[_MOUTH_LEFT_CORNER]  = _landmark(y=0.5)
        lm[_MOUTH_RIGHT_CORNER] = _landmark(y=0.5)
        lm[_MOUTH_UPPER_LIP]    = _landmark(y=0.5)
        score = _smile_score(lm)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_frown_returns_below_half(self):
        """Corners lower than lip centre → frowning."""
        lm = _make_landmarks()
        lm[_MOUTH_LEFT_CORNER]  = _landmark(y=0.7)   # corners low
        lm[_MOUTH_RIGHT_CORNER] = _landmark(y=0.7)
        lm[_MOUTH_UPPER_LIP]    = _landmark(y=0.5)   # lip higher
        score = _smile_score(lm)
        assert score < 0.5

    def test_score_clipped_to_zero_one(self):
        """Extreme values should be clipped."""
        lm = _make_landmarks()
        lm[_MOUTH_LEFT_CORNER]  = _landmark(y=0.0)   # extreme
        lm[_MOUTH_RIGHT_CORNER] = _landmark(y=0.0)
        lm[_MOUTH_UPPER_LIP]    = _landmark(y=1.0)
        score = _smile_score(lm)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_image tests
# ---------------------------------------------------------------------------

class TestScoreImage:
    def test_mediapipe_import_error_returns_half(self):
        """If mediapipe is not installed, returns neutral 0.5."""
        img = Image.new("RGB", (100, 100))
        with patch("builtins.__import__", side_effect=ImportError("no mediapipe")):
            # score_image catches ImportError internally
            pass
        # Verify the except ImportError path returns 0.5
        # We can test this by checking the return value when mediapipe is absent
        with patch.dict("sys.modules", {"mediapipe": None}):
            result = score_image(img)
        assert result == pytest.approx(0.5)

    def test_no_faces_returns_half(self):
        """No face landmarks detected → 0.5."""
        img = Image.new("RGB", (100, 100))

        mock_mp = MagicMock()
        mock_face_mesh_instance = MagicMock()
        mock_face_mesh_instance.__enter__ = lambda s: s
        mock_face_mesh_instance.__exit__ = MagicMock(return_value=False)
        results = MagicMock()
        results.multi_face_landmarks = None
        mock_face_mesh_instance.process.return_value = results
        mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh_instance

        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            result = score_image(img)

        assert result == pytest.approx(0.5)

    def test_exception_returns_half(self):
        """Unexpected exception → 0.5."""
        img = Image.new("RGB", (100, 100))
        mock_mp = MagicMock()
        mock_mp.solutions.face_mesh.FaceMesh.side_effect = RuntimeError("crash")

        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            result = score_image(img)

        assert result == pytest.approx(0.5)

    def test_returns_float(self):
        img = Image.new("RGB", (100, 100))
        mock_mp = MagicMock()
        mock_mp.solutions.face_mesh.FaceMesh.return_value.__enter__ = lambda s: s
        mock_mp.solutions.face_mesh.FaceMesh.return_value.__exit__ = MagicMock(return_value=False)
        results = MagicMock()
        results.multi_face_landmarks = None
        mock_mp.solutions.face_mesh.FaceMesh.return_value.process.return_value = results

        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            result = score_image(img)

        assert isinstance(result, float)
