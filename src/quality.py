"""
Image quality metrics: sharpness (Laplacian variance), exposure, and resolution.
All operations are pure-numpy — no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class QualityResult:
    blur_score: float       # Laplacian variance — higher = sharper
    exposure_score: float   # 0–1, ideal range ≈ 0.15–0.90
    resolution: int         # shorter dimension in pixels
    passes: bool            # True if all thresholds met


def blur_score(img: Image.Image) -> float:
    """
    Laplacian variance of the grayscale image.
    Higher values indicate sharper images.
    Uses array slicing — no scipy required.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Discrete Laplacian via 4-neighbour finite difference
    lap = (
        gray[1:-1, :-2]    # left
        + gray[1:-1, 2:]   # right
        + gray[:-2, 1:-1]  # top
        + gray[2:, 1:-1]   # bottom
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def exposure_score(img: Image.Image) -> float:
    """Mean pixel brightness normalised to [0, 1]."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    return float(gray.mean() / 255.0)


def resolution(img: Image.Image) -> int:
    """Shorter side in pixels (conservative quality measure)."""
    return min(img.size)


def assess(
    img: Image.Image,
    min_blur_score: float = 50.0,
    min_exposure_score: float = 0.15,
    max_exposure_score: float = 0.90,
    min_resolution: int = 640,
    orig_resolution: int = 0,
) -> QualityResult:
    """
    Compute quality metrics and evaluate against thresholds.

    Args:
        img: PIL Image (RGB) — may be a downscaled processing copy.
        min_blur_score: minimum Laplacian variance to pass.
        min_exposure_score: minimum mean brightness (0–1).
        max_exposure_score: maximum mean brightness (0–1).
        min_resolution: minimum shorter dimension in pixels.
        orig_resolution: shorter dimension of the *original* image before any
            downscaling.  When > 0 this value is used for the resolution check
            and stored in the result instead of ``min(img.size)``.  Pass this
            whenever the image has been resized for processing so that
            widescreen originals (16:9, 16:10 …) are not incorrectly rejected
            because their processing copy is narrower than the threshold.

    Returns:
        QualityResult with individual metrics and overall pass/fail.
    """
    b = blur_score(img)
    e = exposure_score(img)
    r = orig_resolution if orig_resolution > 0 else resolution(img)
    passes = (
        b >= min_blur_score
        and min_exposure_score <= e <= max_exposure_score
        and r >= min_resolution
    )
    return QualityResult(blur_score=b, exposure_score=e, resolution=r, passes=passes)
