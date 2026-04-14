"""
Unit tests for src/privacy.py — screenshot detection, home filtering, document CLIP check.
The CLIP-dependent functions are tested with mocks; pure-Python functions are tested directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.privacy import (
    _SCREEN_RESOLUTIONS,
    _haversine_km,
    assess,
    is_home_private,
    is_screenshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, color=(100, 100, 100))


# ---------------------------------------------------------------------------
# is_screenshot tests
# ---------------------------------------------------------------------------

class TestIsScreenshot:
    def test_gps_photo_not_screenshot(self):
        img = _img((1080, 1920))
        assert is_screenshot(img, camera_model="", has_gps=True) is False

    def test_camera_model_not_screenshot(self):
        img = _img((1080, 1920))
        assert is_screenshot(img, camera_model="iPhone 15", has_gps=False) is False

    def test_known_screen_resolution_is_screenshot(self):
        w, h = next(iter(_SCREEN_RESOLUTIONS))
        img = _img((w, h))
        assert is_screenshot(img, camera_model="", has_gps=False) is True

    def test_known_resolution_landscape_is_screenshot(self):
        """Landscape orientation of a known screen size also counts."""
        w, h = next(iter(_SCREEN_RESOLUTIONS))
        img = _img((h, w))  # swapped
        assert is_screenshot(img, camera_model="", has_gps=False) is True

    def test_non_screen_resolution_not_screenshot(self):
        img = _img((1234, 5678))  # unlikely screen resolution
        assert is_screenshot(img, camera_model="", has_gps=False) is False


# ---------------------------------------------------------------------------
# _haversine_km tests
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km(37.77, -122.41, 37.77, -122.41) == pytest.approx(0.0, abs=0.001)

    def test_known_distance_sf_to_la(self):
        """SF to LA is roughly 560 km."""
        dist = _haversine_km(37.77, -122.41, 34.05, -118.24)
        assert 540 <= dist <= 600

    def test_north_south_pole(self):
        """Two poles are ~20000 km apart (half circumference)."""
        dist = _haversine_km(90.0, 0.0, -90.0, 0.0)
        assert 19000 <= dist <= 21000


# ---------------------------------------------------------------------------
# is_home_private tests
# ---------------------------------------------------------------------------

class TestIsHomePrivate:
    def test_no_home_returns_false(self):
        assert is_home_private(37.77, -122.41, face_count=0, home=None) is False

    def test_no_gps_returns_false(self):
        home = (37.77, -122.41)
        assert is_home_private(0.0, 0.0, face_count=0, home=home) is False

    def test_at_home_solo_returns_true(self):
        home = (37.7749, -122.4194)
        assert is_home_private(37.7749, -122.4194, face_count=0, home=home) is True

    def test_at_home_with_people_returns_false(self):
        home = (37.7749, -122.4194)
        assert is_home_private(37.7749, -122.4194, face_count=2, home=home) is False

    def test_far_from_home_returns_false(self):
        home = (37.7749, -122.4194)
        assert is_home_private(48.8566, 2.3522, face_count=0, home=home) is False

    def test_custom_radius(self):
        home = (37.77, -122.41)
        # ~2 km away
        result_tight = is_home_private(37.79, -122.41, face_count=0, home=home, radius_km=1.0)
        result_loose = is_home_private(37.79, -122.41, face_count=0, home=home, radius_km=5.0)
        assert result_tight is False
        assert result_loose is True


# ---------------------------------------------------------------------------
# assess tests (CLIP path mocked)
# ---------------------------------------------------------------------------

class TestAssess:
    def _img(self):
        return _img((640, 480))

    def test_screenshot_excluded(self):
        """If is_screenshot returns True, assess returns True."""
        img = _img((1080, 1920))  # known screen resolution
        result = assess(
            img=img,
            camera_model="", has_gps=False,
            lat=0.0, lon=0.0, face_count=0,
            home=None, home_radius_km=0.5,
            filter_screenshots=True,
            filter_documents=False,
            filter_home_private=False,
        )
        assert result is True

    def test_screenshot_filter_off(self):
        """With filter_screenshots=False, screenshots are not excluded."""
        img = _img((1080, 1920))
        result = assess(
            img=img,
            camera_model="", has_gps=False,
            lat=0.0, lon=0.0, face_count=0,
            home=None, home_radius_km=0.5,
            filter_screenshots=False,
            filter_documents=False,
            filter_home_private=False,
        )
        assert result is False

    def test_home_private_excluded(self):
        home = (37.7749, -122.4194)
        img = self._img()
        result = assess(
            img=img,
            camera_model="iPhone", has_gps=True,
            lat=37.7749, lon=-122.4194, face_count=0,
            home=home, home_radius_km=0.5,
            filter_screenshots=False,
            filter_documents=False,
            filter_home_private=True,
        )
        assert result is True

    def test_document_clip_mock(self):
        """CLIP document check mocked to return True → assess returns True."""
        img = self._img()
        with patch("src.privacy.is_document_clip", return_value=True):
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is True

    def test_document_clip_mock_false(self):
        """CLIP document check mocked to return False → assess returns False."""
        img = self._img()
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=0.0, lon=0.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is False

    def test_normal_photo_not_excluded(self):
        """Photo with camera model, GPS, and CLIP not doc → should not be excluded."""
        img = self._img()
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=img,
                camera_model="Sony A7", has_gps=True,
                lat=48.8, lon=2.3, face_count=2,
                home=None, home_radius_km=0.5,
                filter_screenshots=True,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is False
