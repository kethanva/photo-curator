"""
Unit tests for src/quality.py — sharpness, exposure, and resolution metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.quality import QualityResult, assess, blur_score, exposure_score, resolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_rgb(r: int, g: int, b: int, size: tuple[int, int] = (100, 100)) -> Image.Image:
    """Create a uniform-colour RGB image."""
    arr = np.full((*size, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _gradient_image(size: tuple[int, int] = (200, 200)) -> Image.Image:
    """Create a gradient image — has lots of edges so high blur_score."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        arr[i, :, :] = i * 255 // size[0]
    return Image.fromarray(arr, "RGB")


def _checkerboard(size: tuple[int, int] = (100, 100)) -> Image.Image:
    """Checkerboard — high-frequency image → very high Laplacian variance."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            val = 255 if (i + j) % 2 == 0 else 0
            arr[i, j, :] = val
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# blur_score tests
# ---------------------------------------------------------------------------

class TestBlurScore:
    def test_uniform_image_has_zero_blur(self):
        img = _solid_rgb(128, 128, 128)
        score = blur_score(img)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_checkerboard_has_high_blur_score(self):
        img = _checkerboard()
        score = blur_score(img)
        assert score > 1000, f"Expected high score, got {score}"

    def test_gradient_has_nonzero_score(self):
        img = _gradient_image()
        score = blur_score(img)
        assert score > 0

    def test_returns_float(self):
        img = _solid_rgb(10, 20, 30)
        assert isinstance(blur_score(img), float)

    def test_different_colours_same_uniformity(self):
        """Uniform images of different colours all have blur_score ≈ 0."""
        for r, g, b in [(0, 0, 0), (255, 255, 255), (100, 200, 50)]:
            img = _solid_rgb(r, g, b)
            assert blur_score(img) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# exposure_score tests
# ---------------------------------------------------------------------------

class TestExposureScore:
    def test_black_image_score_is_zero(self):
        img = _solid_rgb(0, 0, 0)
        score = exposure_score(img)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_white_image_score_is_one(self):
        img = _solid_rgb(255, 255, 255)
        score = exposure_score(img)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_midgrey_score_near_half(self):
        img = _solid_rgb(128, 128, 128)
        score = exposure_score(img)
        assert 0.48 <= score <= 0.52

    def test_score_in_zero_one_range(self):
        for r, g, b in [(0, 0, 0), (255, 255, 255), (128, 64, 200)]:
            score = exposure_score(_solid_rgb(r, g, b))
            assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        assert isinstance(exposure_score(_solid_rgb(100, 100, 100)), float)


# ---------------------------------------------------------------------------
# resolution tests
# ---------------------------------------------------------------------------

class TestResolution:
    def test_square_returns_side(self):
        img = Image.new("RGB", (256, 256))
        assert resolution(img) == 256

    def test_landscape_returns_height(self):
        img = Image.new("RGB", (1920, 1080))
        assert resolution(img) == 1080

    def test_portrait_returns_width(self):
        img = Image.new("RGB", (720, 1280))
        assert resolution(img) == 720

    def test_returns_int(self):
        img = Image.new("RGB", (100, 200))
        assert isinstance(resolution(img), int)


# ---------------------------------------------------------------------------
# assess tests
# ---------------------------------------------------------------------------

class TestAssess:
    def test_returns_quality_result(self):
        img = _checkerboard((800, 800))
        result = assess(img)
        assert isinstance(result, QualityResult)

    def test_fails_on_tiny_image(self):
        """Image too small should fail resolution check."""
        img = _checkerboard((100, 100))
        result = assess(img, min_resolution=640)
        assert result.resolution == 100
        assert result.passes is False

    def test_fails_on_uniform_black_image(self):
        """Zero blur and zero exposure should fail."""
        img = _solid_rgb(0, 0, 0, size=(800, 800))
        result = assess(img, min_blur_score=50.0, min_exposure_score=0.15)
        assert result.passes is False

    def test_passes_for_sharp_well_exposed_image(self):
        """Checkerboard at adequate size and mid-grey should pass."""
        # Build a checkerboard with mid-grey average (alternating 0 and 255 ≈ 128 mean)
        img = _checkerboard((800, 800))
        result = assess(
            img,
            min_blur_score=100.0,
            min_exposure_score=0.15,
            max_exposure_score=0.90,
            min_resolution=640,
        )
        assert result.passes is True

    def test_fails_over_exposed(self):
        img = _solid_rgb(255, 255, 255, size=(800, 800))
        result = assess(img, max_exposure_score=0.90)
        assert result.passes is False

    def test_quality_result_fields(self):
        img = _checkerboard((500, 500))
        result = assess(img)
        assert result.blur_score >= 0
        assert 0.0 <= result.exposure_score <= 1.0
        assert result.resolution > 0

    def test_custom_thresholds_respected(self):
        """Very loose thresholds should make even bad images pass."""
        img = _solid_rgb(5, 5, 5, size=(10, 10))
        result = assess(
            img,
            min_blur_score=0.0,
            min_exposure_score=0.0,
            max_exposure_score=1.0,
            min_resolution=1,
        )
        assert result.passes is True

    def test_widescreen_fails_without_orig_resolution(self):
        """1024×576 processing image (16:9) fails min_resolution=640 without orig info."""
        img = _checkerboard((1024, 576))
        result = assess(img, min_resolution=640)
        assert result.resolution == 576
        assert result.passes is False

    def test_widescreen_passes_with_orig_resolution(self):
        """Same 1024×576 processing image passes when orig shorter side is known (2268)."""
        img = _checkerboard((1024, 576))
        result = assess(img, min_resolution=640, orig_resolution=2268)
        assert result.resolution == 2268
        assert result.passes is True

    def test_orig_resolution_zero_falls_back_to_img_size(self):
        """orig_resolution=0 means 'not provided' — falls back to min(img.size)."""
        img = _checkerboard((800, 800))
        result = assess(img, orig_resolution=0)
        assert result.resolution == 800


# ---------------------------------------------------------------------------
# flesh_fraction tests (accidental close-up signal)
# ---------------------------------------------------------------------------

from src.quality import flesh_fraction


class TestFleshFraction:
    def test_skin_tone_image_high_fraction(self):
        """A frame filled with a flesh tone (hue ~25°) scores near 1.0."""
        # RGB (224, 150, 120): hue ≈ 17°, sat ≈ 0.46, val ≈ 0.88 → in band.
        img = _solid_rgb(224, 150, 120, size=(80, 80))
        assert flesh_fraction(img) > 0.9

    def test_gray_image_zero_fraction(self):
        """Neutral gray has zero saturation → never flesh."""
        img = _solid_rgb(128, 128, 128, size=(80, 80))
        assert flesh_fraction(img) == 0.0

    def test_blue_image_zero_fraction(self):
        """Pure blue (hue 240°) is far outside the flesh band."""
        img = _solid_rgb(20, 40, 220, size=(80, 80))
        assert flesh_fraction(img) == 0.0

    def test_half_skin_half_gray(self):
        """Half flesh, half gray → roughly 0.5 fraction."""
        arr = np.zeros((80, 80, 3), dtype=np.uint8)
        arr[:, :40] = [224, 150, 120]   # flesh
        arr[:, 40:] = [128, 128, 128]   # gray
        img = Image.fromarray(arr, "RGB")
        f = flesh_fraction(img)
        assert 0.4 <= f <= 0.6

    def test_assess_populates_flesh_fraction(self):
        img = _solid_rgb(224, 150, 120, size=(80, 80))
        result = assess(img)
        assert result.flesh_fraction > 0.9


# ---------------------------------------------------------------------------
# mundane_heuristic_score flesh-discount tests
# ---------------------------------------------------------------------------

from src.quality import mundane_heuristic_score


class TestMundaneFleshDiscount:
    def test_flesh_heavy_frame_not_mundane(self):
        """A skin-toned frame (person in shot) must score ~0 despite being
        uniform — the flesh discount overrides the uniformity signals."""
        img = _solid_rgb(224, 150, 120, size=(80, 80))
        assert mundane_heuristic_score(img) < 0.1

    def test_gray_wall_still_mundane(self):
        """A flat gray wall has no flesh → full mundane score retained."""
        img = _solid_rgb(128, 128, 128, size=(80, 80))
        assert mundane_heuristic_score(img) >= 0.62

    def test_explicit_flesh_param_overrides_recompute(self):
        """Passing flesh explicitly skips internal recomputation and applies
        the discount based on the supplied value."""
        img = _solid_rgb(128, 128, 128, size=(80, 80))
        undiscounted = mundane_heuristic_score(img, flesh=0.0)
        discounted = mundane_heuristic_score(img, flesh=0.30)
        assert discounted < undiscounted
        assert discounted == pytest.approx(max(undiscounted - 0.30 * 3.0, 0.0))

    def test_assess_uses_flesh_discounted_mundane(self):
        """quality.assess must feed its flesh fraction into the mundane score:
        a flesh-filled frame reports a near-zero mundane_score."""
        result = assess(_solid_rgb(224, 150, 120, size=(80, 80)))
        assert result.mundane_score < 0.1
