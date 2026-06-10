"""
Image quality metrics: sharpness (Laplacian variance), exposure, resolution,
detail (luma stddev), and mundane-object heuristic.
All operations are pure-numpy — no heavy dependencies.

mundane_heuristic_score and detail_stddev ported from photos-cleanup/src/quality.rs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class QualityResult:
    blur_score: float       # Laplacian variance — higher = sharper
    exposure_score: float   # 0–1, ideal range ≈ 0.15–0.90
    resolution: int         # shorter dimension in pixels
    passes: bool            # True if all thresholds met
    detail_stddev: float = 0.0   # luma stddev — low = flat/featureless frame
    mundane_score: float = 0.0   # 0–1 heuristic, ≥0.62 likely mundane object
    flesh_fraction: float = 0.0  # 0–1 — fraction of frame in human flesh-tone band


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


def detail_stddev(img: Image.Image, max_dim: int = 512) -> float:
    """
    Luma standard deviation of a downsampled grayscale copy.

    Low values (< 12) indicate a flat, featureless frame — blank wall,
    ceiling, floor — with no meaningful subject. High values indicate
    varied tonal content typical of real scenes.

    Ported from photos-cleanup/src/quality.rs::detail_stddev.
    """
    thumb = img.copy()
    thumb.thumbnail((max_dim, max_dim), Image.BILINEAR)
    gray = np.array(thumb.convert("L"), dtype=np.float64)
    return float(gray.std())


def flesh_fraction(img: Image.Image, max_dim: int = 64) -> float:
    """
    Fraction of pixels (0–1) that fall in the human flesh-tone band:
    hue 3–50°, saturation 0.08–0.90, value ≥ 0.12.

    A high value combined with a low blur_score signals an accidental
    close-up — the camera fired against a hand/skin and captured no
    intentional scene. Pure-image, no ML model. Note this is really a
    "warm-tone" fraction — wood, sand, terracotta also fall in this hue
    band, so the accidental-closeup gate also requires the frame to be
    blurry.

    Ported from photos-cleanup/src/quality.rs::flesh_fraction (tuned band).
    """
    thumb = img.copy()
    thumb.thumbnail((max_dim, max_dim), Image.BILINEAR)
    rgb = np.array(thumb.convert("RGB"), dtype=np.float32) / 255.0  # H×W×3
    if rgb.size == 0:
        return 0.0

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    val = cmax
    safe_cmax = np.maximum(cmax, 1e-6)
    sat = np.where(cmax > 0.0, delta / safe_cmax, 0.0)

    # Hue in [0, 360); flat (delta==0) pixels keep hue 0, excluded by the band.
    hue = np.zeros_like(r)
    safe = delta > 1e-6
    mask_r = (cmax == r) & safe
    mask_g = (cmax == g) & safe
    mask_b = (cmax == b) & safe
    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    hue[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0)
    hue[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0)

    # Lower hue bound 3° (not 0°) avoids hue=0 artifacts where dark
    # near-gray pixels get assigned h=0 because delta ≈ 0.
    flesh = (
        (hue >= 3.0) & (hue <= 50.0)
        & (sat >= 0.08) & (sat <= 0.90)
        & (val >= 0.12)
    )
    return float(flesh.mean())


def mundane_heuristic_score(
    img: Image.Image,
    flesh: Optional[float] = None,
) -> float:
    """
    Pure-image heuristic estimating how display-unworthy a photo is (0–1).

    Combines three cheap signals — higher score = more likely to be a door,
    wall, window, random household item, or other scene with no aesthetic subject:

      1. Hue entropy   (weight 0.50) — uniform surfaces have very low entropy;
                                       landscapes, faces, animals have high entropy.
      2. Mean saturation (weight 0.20) — indoor neutral objects tend to be grey/beige;
                                          outdoor and portrait photos tend to be vivid.
      3. Spatial uniformity (weight 0.30) — a scene dominated by one object has
                                             similar colour across blocks; interesting
                                             scenes have varied block means.

    ``flesh`` is the flesh-tone pixel fraction of the same image; pass it when
    already computed (quality.assess does) to avoid recomputation, otherwise it
    is derived here. A frame with substantial flesh content (> 0.03) very
    likely contains a person, so the score is aggressively discounted —
    portraits and skin close-ups must never be classed as mundane objects.

    Typical mundane photos score ≥ 0.55; interesting photos score ≤ 0.35.
    Threshold 0.62 used as a conservative junk gate alongside CLIP.

    Ported from photos-cleanup/src/quality.rs::mundane_score (incl. the
    flesh-content discount from mundane_from_rgb128).
    """
    thumb = img.copy()
    thumb.thumbnail((128, 128), Image.BILINEAR)
    rgb = np.array(thumb.convert("RGB"), dtype=np.float32) / 255.0  # H×W×3

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Saturation
    sat = np.where(cmax > 0.0, delta / np.maximum(cmax, 1e-6), 0.0)
    mean_sat = float(sat.mean())

    # Hue in [0, 360)
    hue = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    hue[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0)
    hue[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0)

    # Shannon entropy of 32-bin hue histogram
    bins = np.floor(hue / 360.0 * 32.0).clip(0, 31).astype(np.int32)
    counts = np.bincount(bins.ravel(), minlength=32).astype(np.float64)
    n = float(bins.size)
    probs = counts[counts > 0] / n
    entropy = float(-np.sum(probs * np.log2(probs)))  # max ≈ 5.0 bits

    # Spatial block variance (4×4 grid of mean RGB)
    h_px, w_px = rgb.shape[:2]
    bh, bw = max(h_px // 4, 1), max(w_px // 4, 1)
    block_means = []
    for by in range(4):
        for bx in range(4):
            block = rgb[by * bh : (by + 1) * bh, bx * bw : (bx + 1) * bw]
            if block.size > 0:
                block_means.append(block.mean(axis=(0, 1)))  # shape (3,)
    bm = np.array(block_means, dtype=np.float64)  # (≤16, 3)
    mean_rgb = bm.mean(axis=0)
    spatial_var = float(np.mean(np.sum((bm - mean_rgb) ** 2, axis=1)))
    # ~0 for flat surfaces, ~5000–20000 for varied scenes; normalise to [0,1]
    spatial_uniformity = float(np.clip(1.0 - spatial_var / 8_000.0, 0.0, 1.0))

    entropy_score = float(np.clip(1.0 - entropy / 5.0, 0.0, 1.0))
    sat_score = float(np.clip(1.0 - mean_sat * 5.0, 0.0, 1.0))

    base_score = float(np.clip(
        entropy_score * 0.50 + spatial_uniformity * 0.30 + sat_score * 0.20,
        0.0, 1.0,
    ))

    # Flesh-content discount: a person in the frame is highly unlikely to be
    # a mundane object, regardless of how uniform the background is.
    if flesh is None:
        flesh = flesh_fraction(img)
    if flesh > 0.03:
        return max(base_score - flesh * 3.0, 0.0)
    return base_score


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
    d = detail_stddev(img)
    f = flesh_fraction(img)
    m = mundane_heuristic_score(img, flesh=f)
    passes = (
        b >= min_blur_score
        and min_exposure_score <= e <= max_exposure_score
        and r >= min_resolution
    )
    return QualityResult(
        blur_score=b, exposure_score=e, resolution=r, passes=passes,
        detail_stddev=d, mundane_score=m, flesh_fraction=f,
    )
