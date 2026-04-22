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
    _is_private_from_probs,
    assess,
    is_home_private,
    is_reshared_filename,
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

    def test_document_embedding_fast_path_used_when_provided(self):
        """When a clip_emb is supplied, assess routes to is_document_from_embedding,
        not is_document_clip (which would otherwise re-run CLIP on the image)."""
        img = self._img()
        emb = np.ones(512, dtype=np.float32)
        with patch("src.privacy.is_document_from_embedding", return_value=True) as emb_check, \
             patch("src.privacy.is_document_clip", return_value=False) as img_check:
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=0.0, lon=0.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
                clip_emb=emb,
            )
        assert result is True
        emb_check.assert_called_once()
        img_check.assert_not_called()

    def test_document_image_path_used_when_no_embedding(self):
        """Without clip_emb, assess falls back to is_document_clip (original path)."""
        img = self._img()
        with patch("src.privacy.is_document_from_embedding", return_value=False) as emb_check, \
             patch("src.privacy.is_document_clip", return_value=True) as img_check:
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=0.0, lon=0.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is True
        img_check.assert_called_once()
        emb_check.assert_not_called()


# ---------------------------------------------------------------------------
# Reshared-filename tests
# ---------------------------------------------------------------------------

class TestIsPrivateFromProbs:
    """Locks in the (threshold, ratio) gate that prevents over-flagging."""

    def _probs(self, best_private: float, normal: float) -> np.ndarray:
        # 10 private prompts + 1 normal = 11 slots; fill rest with uniform
        # residual so the array sums to 1.0 (as real softmax output would).
        n_priv = 10
        residual = max(0.0, 1.0 - best_private - normal) / (n_priv - 1)
        arr = np.full(n_priv + 1, residual, dtype=float)
        arr[0] = best_private
        arr[-1] = normal
        return arr

    def test_clear_document_flagged(self):
        # Genuine document: 0.80 private vs 0.05 normal → 16× ratio
        assert _is_private_from_probs(self._probs(0.80, 0.05)) is True

    def test_borderline_vacation_photo_not_flagged(self):
        # Vacation photo that drifts up on "social media screenshot"
        # but normal prompt is still close → caught by ratio gate.
        assert _is_private_from_probs(self._probs(0.47, 0.30)) is False

    def test_absolute_threshold_enforced(self):
        # Even with infinite ratio (normal ≈ 0), below threshold = pass.
        assert _is_private_from_probs(self._probs(0.40, 0.001)) is False

    def test_ratio_gate_enforced(self):
        # Above threshold but normal is comparable → pass (not dominant)
        assert _is_private_from_probs(self._probs(0.52, 0.40)) is False

    def test_custom_threshold_and_ratio(self):
        probs = self._probs(0.55, 0.20)
        assert _is_private_from_probs(probs, threshold=0.50, ratio=1.8) is True
        assert _is_private_from_probs(probs, threshold=0.60, ratio=1.8) is False
        assert _is_private_from_probs(probs, threshold=0.50, ratio=3.0) is False


class TestIsResharedFilename:
    @pytest.mark.parametrize("name", [
        "FB_IMG_1558616482459.jpg",
        "fb_img_1234.jpg",
        "IMG-WA0001.jpg",
        "IMG-WA20230101-WA0007.jpg",
        "WhatsApp Image 2024-01-01 at 10.00.00.jpeg",
        "received_1234567890.jpeg",
        "Screenshot_20240101-103000.png",
        "Screen Shot 2024-01-01 at 10.00.00 AM.png",
        "insta_save_123.jpg",
        "telegram_photo.jpg",
    ])
    def test_reshared_names_flagged(self, name):
        assert is_reshared_filename(name) is True

    @pytest.mark.parametrize("name", [
        "IMG_20190303_194319.jpg",   # legit Android camera
        "IMG_1234.HEIC",             # legit iOS camera
        "DSC00123.JPG",              # legit DSLR
        "PXL_20240101_103000.jpg",   # legit Pixel camera
        "vacation.jpg",
    ])
    def test_regular_names_not_flagged(self, name):
        assert is_reshared_filename(name) is False

    def test_full_path_uses_leaf_only(self):
        assert is_reshared_filename("/foo/bar/FB_IMG_42.jpg") is True
        assert is_reshared_filename("/foo/FB_IMG_/actual_photo.jpg") is False

    def test_custom_prefix_list(self):
        assert is_reshared_filename("CUSTOM_123.jpg", prefixes=["custom_"]) is True
        # Defaults no longer apply when a custom list is provided
        assert is_reshared_filename("FB_IMG_1.jpg", prefixes=["custom_"]) is False


class TestAssessReshared:
    """Reshared-filename gate inside assess()."""

    def _img(self) -> Image.Image:
        return Image.new("RGB", (1024, 768), color=(100, 100, 100))

    def test_reshared_filename_excluded(self):
        """A path matching a reshare prefix is rejected before CLIP runs."""
        with patch("src.privacy.is_document_clip", return_value=False) as doc_check, \
             patch("src.privacy.is_document_from_embedding", return_value=False) as emb_check:
            result = assess(
                img=self._img(),
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
                filter_reshared=True,
                path="/photos/FB_IMG_12345.jpg",
            )
        assert result is True
        # CLIP checks must be short-circuited since filename filter hit first.
        doc_check.assert_not_called()
        emb_check.assert_not_called()

    def test_reshared_disabled_allows_through(self):
        """filter_reshared=False skips the filename filter entirely."""
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=self._img(),
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
                filter_reshared=False,
                path="/photos/FB_IMG_12345.jpg",
            )
        assert result is False

    def test_missing_path_skips_filename_check(self):
        """Back-compat: old callers that don't pass path must still work."""
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=self._img(),
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is False
