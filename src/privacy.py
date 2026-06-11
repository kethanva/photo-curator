"""
Privacy filtering: removes screenshots, documents/IDs, and optionally
solo shots taken at the user's home location.

Two detection strategies:
  1. Heuristic — fast, no ML: checks screen resolutions + missing EXIF
  2. CLIP zero-shot — catches receipts, ID cards, paper documents

The CLIP check reuses the already-loaded model from embeddings.py.
"""

from __future__ import annotations

import logging
import re
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# detail_stddev (luma stddev) below this = a flat, featureless frame with no
# tonal variation (blank wall, ceiling). Used to gate the pure-image mundane
# heuristic so it only hard-excludes genuinely subject-less frames when CLIP
# is unavailable to corroborate.
_FLAT_FRAME_DETAIL_MAX = 12.0


# ---------------------------------------------------------------------------
# Heuristic: screenshot detection
# ---------------------------------------------------------------------------

# Portrait and landscape pairs of common screen resolutions (width × height)
_SCREEN_RESOLUTIONS: set = {
    (1080, 1920), (1080, 2160), (1080, 2340), (1080, 2400),
    (1080, 2640), (1170, 2532), (1179, 2556), (1284, 2778),
    (1290, 2796), (828, 1792), (750, 1334), (1440, 3040),
    (1440, 3200), (1080, 2280), (1080, 2408),
    # Desktop common resolutions
    (2560, 1440), (1920, 1080), (2560, 1600), (3840, 2160),
    (3024, 1964), (2880, 1800), (2560, 1664), (1366, 768),
}


def is_screenshot(
    img: Image.Image,
    camera_model: str,
    has_gps: bool,
) -> bool:
    """
    Heuristic: true photos have camera EXIF; screenshots usually don't.
    Also checks pixel dimensions against known screen resolutions.
    """
    if has_gps:
        return False   # Real outdoor photos have GPS
    if camera_model.strip():
        return False   # Real photos carry the camera/phone model

    w, h = img.size
    return (w, h) in _SCREEN_RESOLUTIONS or (h, w) in _SCREEN_RESOLUTIONS


# ---------------------------------------------------------------------------
# Heuristic: accidental close-up (camera fired against hand/skin)
# ---------------------------------------------------------------------------


def is_accidental_closeup(
    flesh_fraction: float,
    blur_score: float,
    min_flesh_fraction: float = 0.65,
    max_blur_score: float = 120.0,
) -> bool:
    """
    True when flesh-tone pixels dominate the frame AND the image is blurry —
    the signature of a camera fired by accident against a hand/skin with no
    intentional subject.

    Both gates must trip: a sharp, well-composed portrait close-up also has a
    high flesh fraction but is NOT blurry, so the blur ceiling protects it.

    Ported from photos-cleanup/src/junk.rs AccidentalCloseup gate.
    """
    return flesh_fraction >= min_flesh_fraction and blur_score <= max_blur_score


# ---------------------------------------------------------------------------
# Heuristic: pitch-black / pure-white frame
# ---------------------------------------------------------------------------


def is_pitch_black_or_pure_white(
    exposure_score: float,
    detail_stddev: float,
    max_detail_stddev: float = 2.0,
    dark_exposure: float = 0.05,
    bright_exposure: float = 0.95,
) -> bool:
    """
    True for frames that are almost completely pure black or pure white —
    lens-cap shots, pocket misfires in the dark, blown-out flashes against a
    wall. Requires BOTH extreme mean brightness and near-zero tonal detail, so
    a dark-but-textured night scene or a bright snow field still passes.

    This matters as a hard exclusion because the selection stage treats
    exposure failures as salvageable (fallback-pool admission); a true black
    frame must never re-enter that way.

    Ported from photos-cleanup/src/junk.rs PitchBlackOrPureWhite gate.
    """
    return detail_stddev < max_detail_stddev and (
        exposure_score < dark_exposure or exposure_score > bright_exposure
    )


# ---------------------------------------------------------------------------
# CLIP zero-shot: document / private content detection
# ---------------------------------------------------------------------------

_PRIVATE_PROMPTS = [
    "a photo of a document or paper",
    "a bank card or credit card",
    "a receipt or invoice",
    "an identity card or passport",
    "a screenshot of a phone or computer screen",
    "handwritten notes or a whiteboard",
    # Text-heavy / reshared content typically forwarded from messaging apps.
    # CLIP reads these as distinct from normal scenes that happen to contain
    # incidental text like a street sign.
    "a meme with large text overlay",
    "an image with a quote or inspirational text",
    "a social media post screenshot with text",
    "an advertisement or flyer with heavy text",
]
_NORMAL_PROMPT = "a natural photo taken with a camera outdoors or indoors"

# Cached tokenised text tensors — multi-class privacy prompts
_text_features = None
_clip_loaded = False

# ---------------------------------------------------------------------------
# Dedicated binary text-heavy / screenshot detection
# ---------------------------------------------------------------------------

# Multiple text-heavy variants vs a single real-photo baseline. We run a
# *separate* binary softmax for each (variant, real_photo) pair and OR
# the results together — that way, a real photo (which doesn't strongly
# match ANY single variant) stays below threshold, while a text-heavy
# image (which strongly matches at least one variant) clears it.
#
# A multi-class softmax across all variants would dilute the per-variant
# probability, which is why the previous single-prompt design was used;
# the per-variant binary trick keeps that property while broadening the
# kinds of text-heavy content we catch — chat screenshots, photos of
# printed pages and notebooks, signs/posters dominated by text, slides
# from a presentation, calendar/schedule pages, and meme images with
# heavy text overlay. Real Android camera filenames like
# ``IMG_20190125_110258.jpg`` of a calendar page slipped through the
# single-prompt design because "chat / text-only image" didn't capture
# camera photos of printed material.
_TEXT_HEAVY_PROMPTS = (
    "a screenshot of text messages, chat conversations, or a text-only image",
    "a photo of a printed page, document, notebook, or letter",
    "a photo of a calendar, schedule, or table dominated by text rows",
    "a photo of a sign, poster, or flyer where text fills most of the frame",
    "a screenshot of a slide, presentation, or social media post",
    "a meme or graphic dominated by overlaid text or captions",
)
_REAL_PHOTO_PROMPT = "a real photograph of people, places, or objects"

_text_heavy_features = None
_text_heavy_loaded = False


def _get_clip_text_features():
    """Encode privacy prompts once; return (text_features, model, preprocess, device)."""
    global _text_features, _clip_loaded
    import torch
    import clip

    from src.embeddings import load_model

    model, preprocess, device = load_model()

    if not _clip_loaded:
        prompts = _PRIVATE_PROMPTS + [_NORMAL_PROMPT]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        _text_features = feats
        _clip_loaded = True

    return _text_features, model, preprocess, device


def _get_text_heavy_features():
    """Encode every text-heavy variant + the real-photo baseline once.

    Layout: rows ``[0..N-1]`` are the text-heavy variants in
    ``_TEXT_HEAVY_PROMPTS`` order; row ``N`` is the real-photo baseline.
    ``is_text_heavy_from_embedding`` runs a separate binary softmax for
    each ``(variant_i, baseline)`` pair and ORs the results.
    """
    global _text_heavy_features, _text_heavy_loaded
    import torch
    import clip
    from src.embeddings import load_model

    model, preprocess, device = load_model()

    if not _text_heavy_loaded:
        prompts = list(_TEXT_HEAVY_PROMPTS) + [_REAL_PHOTO_PROMPT]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        _text_heavy_features = feats
        _text_heavy_loaded = True

    return _text_heavy_features, model, preprocess, device


def is_text_heavy_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.55,
) -> bool:
    """
    CLIP check: True when the image is dominated by text content.

    Runs a *per-variant* binary softmax against the real-photo baseline
    (text_heavy_i vs real_photo) and returns True if the max text-heavy
    probability across all variants clears ``threshold``. This sidesteps
    the multi-class dilution that capped the old single-prompt design at
    chat / quote screenshots only — camera photos of printed pages,
    posters, calendars, and presentation slides now score above 0.60 on
    at least one matching variant.

    The 0.55 default sits at the empirical separation point: real photos
    with incidental text (street signs, menus in the background) peak
    around 0.45–0.52 on the closest variant, while genuine text-heavy
    content (printed pages, chat captures, posters) sits at 0.60–0.85.
    """
    if clip_emb is None:
        return False
    try:
        import torch

        all_feats, _, _, device = _get_text_heavy_features()
        image_tensor = torch.tensor(clip_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)

        # Last row is the real-photo baseline; everything before it is a
        # text-heavy variant. Build a binary (variant, baseline) pair for
        # each variant and softmax separately so dilution can't sink the
        # text-heavy mass below threshold.
        baseline = all_feats[-1:]
        variants = all_feats[:-1]
        max_text_prob = 0.0
        for i in range(variants.shape[0]):
            pair = torch.cat([variants[i:i + 1], baseline], dim=0)
            logits = (image_tensor @ pair.T * 100.0).softmax(dim=-1)
            prob = float(logits[0, 0].item())
            if prob > max_text_prob:
                max_text_prob = prob
        return max_text_prob >= threshold
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        logger.warning("is_text_heavy_from_embedding failed: %s", exc)
        return False

# ---------------------------------------------------------------------------
# Boring Objects Detection (legacy — kept for cache reassess compatibility)
# ---------------------------------------------------------------------------

_BORING_PROMPTS = [
    "a photo of a ceiling light or tubelight",
    "a mundane photo of a door or window",
    "a close up of a blank wall",
    "a blurry or accidental photo of the floor or carpet",
    "an uninteresting photo of an empty room corner",
]

_boring_features = None
_boring_loaded = False

def _get_boring_features():
    """Encode boring object prompts once and cache them."""
    global _boring_features, _boring_loaded
    import torch
    import clip
    from src.embeddings import load_model

    model, preprocess, device = load_model()

    if not _boring_loaded:
        prompts = _BORING_PROMPTS + [_NORMAL_PROMPT]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        _boring_features = feats
        _boring_loaded = True

    return _boring_features, model, preprocess, device

def is_boring_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.25,
    ratio: float = 20.0,
) -> bool:
    """
    CLIP check: True when image strongly looks like a mundane single object
    (door, tubelight, wall). Kept conservative because this is an exclusion
    gate, not just a ranking signal.

    Gating design: the 6-way softmax (5 boring prompts + 1 normal prompt)
    means a single class rarely tops 0.5 even for a clear winner, so the
    primary gate here is the ratio check (best_boring must be >= ratio ×
    normal_prob). ``threshold`` is a safety floor that rejects incoherent
    weak-best matches; 0.25 admits a confidently dominant boring class
    without misfiring on diffuse softmax distributions. Earlier 0.90 default
    was effectively unreachable.
    """
    if clip_emb is None:
        return False
    try:
        import torch

        text_feats, _, _, device = _get_boring_features()
        image_tensor = torch.tensor(clip_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)
        logits = (image_tensor @ text_feats.T * 100.0).softmax(dim=-1)
        probs = logits.cpu().numpy()[0]

        normal_prob = float(probs[-1])
        best_boring = float(probs[:-1].max())

        if best_boring <= threshold:
            return False
        return best_boring >= ratio * max(normal_prob, 1e-6)
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        logger.warning("is_boring_from_embedding failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Mundane Object Detection — binary per-variant approach
# ---------------------------------------------------------------------------
# Catches photos where a random everyday object fills the frame with no
# meaningful photographic intent: a door, a switch, a pipe, an appliance,
# clutter on a counter, a wall, etc.  Uses the same per-variant binary
# softmax as is_text_heavy_from_embedding so adding more prompts doesn't
# dilute the signal.
#
# Intentionally does NOT fire on:
#   • landscapes, mountains, nature, sky
#   • people, faces, crowds
#   • bikes, motorcycles, vehicles in context
#   • architecture with interesting composition
#   • food / dining scenes
# Those are excluded by pairing each mundane prompt against a rich
# "meaningful photo" baseline in a binary contest.

_MUNDANE_PROMPTS = (
    # Structural surfaces
    "a close-up photo of a blank or featureless wall",
    "a close-up photo of a door or door handle",
    "a photo of a window frame or glass pane",
    "a photo of a floor or carpet with nothing else visible",
    "a close-up of ceiling tiles or a bare ceiling",
    # Fixtures and fittings
    "a photo of a light switch or electrical socket on a wall",
    "a photo of a ceiling fan, tubelight, or overhead light fixture",
    "a photo of a pipe, cable, or wire attached to a wall",
    "a photo of an air conditioner unit, vent, or grille",
    "a photo of a drain, tap, or plumbing fixture",
    # Appliances and furniture close-ups
    "a photo of a washing machine, refrigerator, or household appliance",
    "a close-up of furniture from an odd angle with no context",
    "a photo of an empty shelf, cabinet, or cupboard",
    "a photo of a mattress, bed frame, or pillow with no people",
    # Clutter and junk
    "a photo of a trash bin, garbage bag, or waste",
    "a photo of random clutter or miscellaneous junk",
    "a photo of construction material, bare concrete, or scaffolding",
    "a photo of random objects on a counter or shelf with no context",
    # Accidental / unintentional shots
    "a blurry or accidental close-up of a random everyday object",
    "an unintentional photo taken by mistake of a random surface",
    # Vehicle detail without context
    "a close-up photo of a car tire, wheel, or engine part with nothing else",
    # Other mundane singletons
    "a photo of a lock, padlock, or latch",
    "a photo of a staircase or railing with no people",
    "a photo of a parking lot or empty road with no interesting content",
)

# Baseline: what we want to KEEP — meaningful, display-worthy photos.
_MEANINGFUL_PHOTO_PROMPT = (
    "a meaningful photograph of people, nature, landscapes, mountains, "
    "animals, celebrations, travel, sports, food, or memorable moments"
)

_mundane_features = None
_mundane_loaded = False


def _get_mundane_features():
    """Encode mundane prompts + meaningful baseline once; rows[0..N-1] = mundane, row[N] = baseline."""
    global _mundane_features, _mundane_loaded
    import torch
    import clip
    from src.embeddings import load_model

    model, preprocess, device = load_model()

    if not _mundane_loaded:
        prompts = list(_MUNDANE_PROMPTS) + [_MEANINGFUL_PHOTO_PROMPT]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        _mundane_features = feats
        _mundane_loaded = True

    return _mundane_features, model, preprocess, device


def is_mundane_object_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.62,
) -> bool:
    """
    CLIP check: True when the image is dominated by a random mundane object
    that completely fills the frame with no display-worthy intent.

    Uses binary per-variant softmax: for each mundane prompt we run a
    2-class softmax (mundane_i vs meaningful_baseline) and return True if
    ANY variant's mundane probability clears ``threshold``.  This avoids the
    multi-class dilution of the legacy boring detector and scales cleanly
    with many prompts.

    threshold=0.62:  real people/landscape/event photos score 0.35–0.55 on
    even the closest mundane prompt; genuine random-object photos score
    0.65–0.90 on at least one matching variant.  0.62 sits in the gap.
    """
    if clip_emb is None:
        return False
    try:
        import torch

        all_feats, _, _, device = _get_mundane_features()
        image_tensor = torch.tensor(clip_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)

        baseline = all_feats[-1:]   # meaningful photo
        variants = all_feats[:-1]   # mundane prompts

        max_mundane_prob = 0.0
        for i in range(variants.shape[0]):
            pair = torch.cat([variants[i : i + 1], baseline], dim=0)
            logits = (image_tensor @ pair.T * 100.0).softmax(dim=-1)
            prob = float(logits[0, 0].item())
            if prob > max_mundane_prob:
                max_mundane_prob = prob

        return max_mundane_prob >= threshold
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        logger.warning("is_mundane_object_from_embedding failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Intimate Content Detection
# ---------------------------------------------------------------------------

_INTIMATE_PROMPTS = [
    "a highly private or intimate photo of a person",
    "a nude or partially nude photo of a man or woman",
    "a sexually explicit or inappropriate photo",
    "a photo of someone in their underwear or swimsuit indoors",
    "a suggestive or revealing photo",
]

_intimate_features = None
_intimate_loaded = False

def _get_intimate_features():
    """Encode intimate object prompts once and cache them."""
    global _intimate_features, _intimate_loaded
    import torch
    import clip
    from src.embeddings import load_model

    model, preprocess, device = load_model()

    if not _intimate_loaded:
        prompts = _INTIMATE_PROMPTS + [_NORMAL_PROMPT]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        _intimate_features = feats
        _intimate_loaded = True

    return _intimate_features, model, preprocess, device

def is_intimate_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.25,
    ratio: float = 20.0,
) -> bool:
    """
    CLIP check: True when image strongly looks like intimate/NSFW content.

    Gating design matches ``is_boring_from_embedding``: 6-way softmax over
    5 intimate prompts + 1 normal prompt. The ratio check is the primary
    gate; ``threshold`` is a safety floor at 0.25 (previous 0.80 was
    unreachable in practice).
    """
    if clip_emb is None:
        return False
    try:
        import torch

        text_feats, _, _, device = _get_intimate_features()
        image_tensor = torch.tensor(clip_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)
        logits = (image_tensor @ text_feats.T * 100.0).softmax(dim=-1)
        probs = logits.cpu().numpy()[0]

        normal_prob = float(probs[-1])
        best_intimate = float(probs[:-1].max())

        if best_intimate <= threshold:
            return False
        return best_intimate >= ratio * max(normal_prob, 1e-6)
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        logger.warning("is_intimate_from_embedding failed: %s", exc)
        return False


def _private_vs_normal_probs(image_emb: np.ndarray) -> Optional[np.ndarray]:
    """
    Softmax distribution across [private_1 … private_N, normal] for a given
    normalised CLIP image embedding.  Returns None on failure.
    """
    try:
        import torch

        text_feats, _model, _preprocess, device = _get_clip_text_features()

        image_tensor = torch.tensor(image_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)

        logits = (image_tensor @ text_feats.T * 100.0).softmax(dim=-1)
        return logits.cpu().numpy()[0]
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        # Returning None silently disables the document/private-content gate
        # on the calling photo (callers `is_document_from_embedding` and
        # `is_document_page_from_embedding_and_image` interpret None as "not
        # private"). Log so an MPS OOM or model-load issue is observable
        # instead of letting private content slip through unflagged.
        logger.warning("_private_vs_normal_probs failed: %s", exc)
        return None


def _is_private_from_probs(
    probs: np.ndarray,
    threshold: float = 0.40,
    ratio: float = 3.0,
) -> bool:
    """
    Decide if the prob distribution indicates a private/document image.

    A photo is flagged private only when the best-matching private prompt
    clears BOTH gates:

      1. absolute confidence    best_private > threshold
      2. dominance over normal  best_private > ratio * normal_prob

    Why dual gate: softmax over 11 prompts dilutes each private prompt's
    mass (text-heavy images typically land in the 0.30–0.50 range because
    their score is split across ten competing private prompts), while real
    vacation photos score ~0.02–0.20 on the normal prompt.  A lone absolute
    threshold either misses text-heavy content (if high) or over-flags
    vacation photos (if low).  The ratio gate is what actually carries
    signal: a genuine document scores 10–100× more on its matching private
    prompt than on the normal prompt.
    """
    normal_prob = float(probs[-1])
    best_private = float(probs[:-1].max())
    if best_private <= threshold:
        return False
    return best_private >= ratio * max(normal_prob, 1e-6)


def is_document_clip(
    img: Image.Image,
    threshold: float = 0.40,
    ratio: float = 3.0,
) -> bool:
    """CLIP zero-shot: True when image looks like a document / private item."""
    try:
        from src.embeddings import load_model, extract

        _, model, preprocess, device = _get_clip_text_features()
        image_emb = extract(img, model, preprocess, device)
        probs = _private_vs_normal_probs(image_emb)
        if probs is None:
            return False
        return _is_private_from_probs(probs, threshold, ratio)
    except (RuntimeError, ValueError, OSError, MemoryError, ImportError) as exc:
        logger.warning("is_document_clip failed: %s", exc)
        return False


def is_document_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.40,
    ratio: float = 3.0,
) -> bool:
    """
    CLIP zero-shot privacy check from a pre-computed embedding.

    Lets the pipeline re-assess cached photos without re-loading image files.
    """
    if clip_emb is None:
        return False
    probs = _private_vs_normal_probs(clip_emb)
    if probs is None:
        return False
    return _is_private_from_probs(probs, threshold, ratio)


def looks_like_text_document_page(img: Image.Image) -> bool:
    """
    Heuristic for camera photos of text-heavy pages, calendars, and tables.

    CLIP alone is too blunt here: it can confuse busy scenes with documents.
    This structural check looks for a large light page region containing many
    straight horizontal/vertical strokes, which is typical of printed tables
    and uncommon in normal people/action photos.
    """
    try:
        import cv2

        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]
        scale = 1000.0 / max(h, w)
        if scale < 1.0:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)))

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        light_low_sat = (
            ((hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 85)).astype("uint8") * 255
        )
        paper_frac = float(light_low_sat.mean() / 255.0)
        if paper_frac < 0.45:
            return False

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=light_low_sat)
        masked_edge_frac = float(masked_edges.mean() / 255.0)
        if masked_edge_frac < 0.06:
            return False

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=60,
            maxLineGap=8,
        )
        if lines is None:
            return False

        hv_lines = 0
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            dx = abs(int(x2) - int(x1))
            dy = abs(int(y2) - int(y1))
            if (dx * dx + dy * dy) ** 0.5 < 60:
                continue
            angle = np.degrees(np.arctan2(dy, dx)) if dx or dy else 0.0
            if angle < 8.0 or angle > 82.0:
                hv_lines += 1

        return hv_lines >= 50
    except Exception as exc:
        # Document page is a privacy gate — a silent False on cv2 import
        # error, OpenCV's own ``cv2.error`` (whose class is hidden inside
        # the cv2 module and not safely importable here), or pixel-decoding
        # failures would let sensitive content through. Log so any failure
        # is observable; intentional broad catch.
        logger.warning("looks_like_text_document_page failed: %s", exc)
        return False


def is_document_page_from_embedding_and_image(
    clip_emb: Optional[np.ndarray],
    img: Image.Image,
) -> bool:
    """True for text-heavy page photos that the conservative doc gate misses."""
    if clip_emb is None:
        return False
    probs = _private_vs_normal_probs(clip_emb)
    if probs is None:
        return False
    if not _is_private_from_probs(probs, threshold=0.60, ratio=50.0):
        return False
    return looks_like_text_document_page(img)


# ---------------------------------------------------------------------------
# Filename heuristic: reshared / forwarded content
# ---------------------------------------------------------------------------

# Case-insensitive filename prefixes used by messaging apps and social
# downloads.  These files are almost always forwarded content (memes,
# screenshots, stock images) rather than photos the user took themselves.
_RESHARED_PREFIXES: Tuple[str, ...] = (
    "fb_img_",       # Facebook downloads
    "img-wa",        # WhatsApp (IMG-WA0001, IMG-WA20230101-…)
    "whatsapp",      # WhatsApp iOS saves
    "received_",     # Facebook Messenger / Telegram
    "screenshot",    # explicit screenshot dumps
    "screen shot",
    "insta_",        # Instagram reshares
    "twitter_",
    "tweet_",
    "telegram_",
    "viber image",
    "signal-",
)


# Pure-numeric stems of 10–13 digits (optionally with a trailing -N or _N
# sub-index) match Unix-epoch filenames written by WhatsApp on iOS,
# Messenger, Telegram, and similar messaging exports — e.g.
# 1568579433476.jpg (13-digit ms epoch). These files carry no EXIF, no
# camera model, no GPS, and are almost always forwarded content the user
# never took themselves. 14+ digits are excluded because that's the range
# of YYYYMMDDHHMMSS dates that some cameras still write.
_MESSAGING_NUMERIC_RE = re.compile(r"^\d{10,13}(?:[-_]\d+)?$")


def is_reshared_filename(
    path: Union[str, Path],
    prefixes: Optional[Iterable[str]] = None,
) -> bool:
    """
    True when the filename matches a well-known messaging/social reshare
    pattern (FB_IMG_*, WhatsApp*, Screenshot_*, 1568579433476.jpg, …).

    Comparison is case-insensitive against the leaf filename only.

    Two checks run in order:
      1. Prefix match against the configured ``prefixes`` (defaults to the
         built-in ``_RESHARED_PREFIXES`` list).
      2. Pure-numeric stem of 10–13 digits — a strong messaging-app signal
         independent of any prefix list. Always runs even when callers
         supply a custom ``prefixes`` iterable, because the numeric pattern
         is universal and not something users typically want to opt out of.
    """
    name = Path(path).name.lower()
    active = tuple(p.lower() for p in prefixes) if prefixes else _RESHARED_PREFIXES
    if any(name.startswith(p) for p in active):
        return True

    stem = Path(path).stem.lower()
    if _MESSAGING_NUMERIC_RE.match(stem):
        return True

    return False


# ---------------------------------------------------------------------------
# Cache re-assessment — refreshes is_private for photos ingested with a
# previous (stale) version of the privacy rules, without re-loading the
# originals or re-running CLIP on them.
# ---------------------------------------------------------------------------

def reassess_is_private_from_cache(
    conn,
    filter_reshared: bool = True,
    reshared_prefixes: Optional[Iterable[str]] = None,
    threshold: float = 0.40,
    document_ratio: float = 3.0,
    filter_text_heavy: bool = True,
    filter_boring_objects: bool = True,
    mundane_object_threshold: float = 0.62,
    filter_intimate_content: bool = True,
    filter_accidental_closeup: bool = True,
    min_flesh_fraction: float = 0.65,
    max_accidental_blur_score: float = 120.0,
    filter_pitch_black: bool = True,
) -> Tuple[int, int]:
    """
    Recompute is_private for every cached photo using only the stored CLIP
    embedding, cached per-pixel metrics, and the filename; updates rows in
    place.

    The CLIP checks, the filename filter, and the metric-replayable gates
    (accidental close-up, pitch-black/pure-white — their raw inputs are
    persisted at ingest) are applied here. The screenshot and home-private
    checks rely on EXIF already captured correctly at ingest time and don't
    change between runs, so their cached flag contributions are preserved.

    Rows ingested before the metric columns existed carry the -1 sentinel;
    their metric gates are skipped rather than replayed against bogus zeros.

    Returns:
        (total_rows_examined, rows_changed)
    """
    from src import database

    rows = conn.execute(
        "SELECT path, clip_emb, is_private, private_reason, camera_model, "
        "has_gps, face_count, "
        "blur_score, exposure_score, detail_stddev, flesh_fraction "
        "FROM photos"
    ).fetchall()

    changed = 0
    for row in rows:
        path = row["path"]
        clip_emb = database.blob_to_emb(row["clip_emb"])
        prev = int(row["is_private"])
        prev_reason = str(row["private_reason"] or "")

        if clip_emb is None:
            continue

        doc_hit = is_document_from_embedding(clip_emb, threshold, document_ratio)
        reshared_hit = (
            filter_reshared and is_reshared_filename(path, reshared_prefixes)
        )
        text_hit = filter_text_heavy and is_text_heavy_from_embedding(clip_emb)
        boring_hit = filter_boring_objects and (
            is_boring_from_embedding(clip_emb)
            or is_mundane_object_from_embedding(clip_emb, threshold=mundane_object_threshold)
        )
        intimate_hit = (
            filter_intimate_content and is_intimate_from_embedding(clip_emb)
        )

        # Metric-replayable gates. Without these, the revocation branch below
        # would un-flag accidental close-ups and lens-cap shots on every
        # reassess run — those photos DO carry a camera model. -1 sentinel
        # (pre-migration row) disables the gate for that row.
        flesh = float(row["flesh_fraction"] if row["flesh_fraction"] is not None else -1.0)
        detail = float(row["detail_stddev"] if row["detail_stddev"] is not None else -1.0)
        blur = float(row["blur_score"] or 0.0)
        exposure = float(row["exposure_score"] if row["exposure_score"] is not None else 0.5)
        closeup_hit = (
            filter_accidental_closeup
            and flesh >= 0.0
            and is_accidental_closeup(
                flesh, blur, min_flesh_fraction, max_accidental_blur_score
            )
        )
        pitch_hit = (
            filter_pitch_black
            and detail >= 0.0
            and is_pitch_black_or_pure_white(exposure, detail)
        )

        # First gate that fired, in the same precedence order as
        # assess_with_reason — persisted so exclusions stay auditable.
        if reshared_hit:
            cache_reason = "reshared_filename"
        elif closeup_hit:
            cache_reason = "accidental_closeup"
        elif pitch_hit:
            cache_reason = "pitch_black_or_pure_white"
        elif doc_hit:
            cache_reason = "document"
        elif text_hit:
            cache_reason = "text_heavy"
        elif boring_hit:
            cache_reason = "mundane_object"
        elif intimate_hit:
            cache_reason = "intimate_content"
        else:
            cache_reason = ""

        new_flag_from_cache = bool(cache_reason)

        # Reassessment only sees CLIP-recomputable + filename signals. Two other
        # rules (is_screenshot — needs original image dims; is_home_private —
        # needs the home config + EXIF GPS) ran at ingest and aren't replayed
        # here. To avoid sticky CLIP false-positives from older loose
        # thresholds, we revoke a prior True only when the photo carries
        # strong "real photo" signals: a camera model, GPS, or detected faces.
        # Photos without any of those signals stay flagged — they are likely
        # screenshots / home-private hits we can't recompute from cache.
        if new_flag_from_cache:
            final_flag = 1
            final_reason = cache_reason
        elif prev:
            camera_model = (row["camera_model"] or "").strip()
            has_gps = int(row["has_gps"] or 0)
            face_count = int(row["face_count"] or 0)
            has_real_photo_signal = (
                bool(camera_model) or has_gps > 0 or face_count > 0
            )
            if has_real_photo_signal:
                final_flag = 0
                final_reason = ""
            else:
                final_flag = 1
                # Keep the original ingest-time reason if recorded; otherwise
                # mark it as a non-replayable gate preserved from ingest.
                final_reason = prev_reason or "preserved_from_ingest"
        else:
            final_flag = 0
            final_reason = ""

        if final_flag != prev or final_reason != prev_reason:
            conn.execute(
                "UPDATE photos SET is_private=?, private_reason=? WHERE path=?",
                (final_flag, final_reason, path),
            )
            if final_flag != prev:
                changed += 1

    # NOTE: commit is the caller's responsibility — this function is invoked
    # under a ``with database.connect(...) as conn`` block in main.py which
    # commits on clean exit. Committing here would be a redundant double-commit
    # (and raises in stricter SQLite modes).
    return len(rows), changed


# ---------------------------------------------------------------------------
# Home + solo heuristic
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlon1 = radians(lat1), radians(lon1)
    rlat2, rlon2 = radians(lat2), radians(lon2)
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return R * 2.0 * atan2(sqrt(a), sqrt(1.0 - a))


def is_home_private(
    lat: float,
    lon: float,
    face_count: int,
    home: Optional[Tuple[float, float]],
    radius_km: float = 0.5,
) -> bool:
    """
    True when the photo was taken at home (within radius) with ≤ 1 face.
    Designed to filter routine daily snapshots from private spaces.
    """
    if home is None:
        return False
    if lat == 0.0 and lon == 0.0:
        return False   # No GPS data
    dist = _haversine_km(home[0], home[1], lat, lon)
    return dist <= radius_km and face_count <= 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess(
    img: Image.Image,
    camera_model: str,
    has_gps: bool,
    lat: float,
    lon: float,
    face_count: int,
    home: Optional[Tuple[float, float]],
    home_radius_km: float,
    filter_screenshots: bool = True,
    filter_documents: bool = True,
    filter_text_heavy: bool = True,
    filter_boring_objects: bool = True,
    mundane_object_threshold: float = 0.62,
    filter_intimate_content: bool = True,
    filter_home_private: bool = False,
    filter_reshared: bool = True,
    path: Optional[Union[str, Path]] = None,
    reshared_prefixes: Optional[Iterable[str]] = None,
    clip_emb: Optional[np.ndarray] = None,
    mundane_heuristic: float = 0.0,
    detail_stddev: float = 0.0,
    filter_accidental_closeup: bool = True,
    flesh_fraction: float = 0.0,
    blur_score: float = float("inf"),
    min_flesh_fraction: float = 0.65,
    max_accidental_blur_score: float = 120.0,
    filter_pitch_black: bool = True,
    exposure_score: float = 0.5,
) -> bool:
    """
    Return True if the photo should be excluded from the curated output.

    Thin wrapper over :func:`assess_with_reason` for callers that don't need
    the audit reason.
    """
    return assess_with_reason(
        img=img,
        camera_model=camera_model,
        has_gps=has_gps,
        lat=lat,
        lon=lon,
        face_count=face_count,
        home=home,
        home_radius_km=home_radius_km,
        filter_screenshots=filter_screenshots,
        filter_documents=filter_documents,
        filter_text_heavy=filter_text_heavy,
        filter_boring_objects=filter_boring_objects,
        mundane_object_threshold=mundane_object_threshold,
        filter_intimate_content=filter_intimate_content,
        filter_home_private=filter_home_private,
        filter_reshared=filter_reshared,
        path=path,
        reshared_prefixes=reshared_prefixes,
        clip_emb=clip_emb,
        mundane_heuristic=mundane_heuristic,
        detail_stddev=detail_stddev,
        filter_accidental_closeup=filter_accidental_closeup,
        flesh_fraction=flesh_fraction,
        blur_score=blur_score,
        min_flesh_fraction=min_flesh_fraction,
        max_accidental_blur_score=max_accidental_blur_score,
        filter_pitch_black=filter_pitch_black,
        exposure_score=exposure_score,
    )[0]


def assess_with_reason(
    img: Image.Image,
    camera_model: str,
    has_gps: bool,
    lat: float,
    lon: float,
    face_count: int,
    home: Optional[Tuple[float, float]],
    home_radius_km: float,
    filter_screenshots: bool = True,
    filter_documents: bool = True,
    filter_text_heavy: bool = True,
    filter_boring_objects: bool = True,
    mundane_object_threshold: float = 0.62,
    filter_intimate_content: bool = True,
    filter_home_private: bool = False,
    filter_reshared: bool = True,
    path: Optional[Union[str, Path]] = None,
    reshared_prefixes: Optional[Iterable[str]] = None,
    clip_emb: Optional[np.ndarray] = None,
    mundane_heuristic: float = 0.0,
    detail_stddev: float = 0.0,
    filter_accidental_closeup: bool = True,
    flesh_fraction: float = 0.0,
    blur_score: float = float("inf"),
    min_flesh_fraction: float = 0.65,
    max_accidental_blur_score: float = 120.0,
    filter_pitch_black: bool = True,
    exposure_score: float = 0.5,
) -> Tuple[bool, str]:
    """
    Return ``(is_private, reason)``. ``reason`` names the first gate that
    flagged the photo ('' when not private) and is persisted to the DB so
    aggressive exclusion rates can be audited per-gate after a run.

    Checks are ordered cheapest-first to avoid unnecessary model calls.
    When a pre-computed CLIP embedding is supplied the document check reuses
    it instead of running CLIP a second time on the same image.
    """
    if filter_reshared and path is not None and is_reshared_filename(path, reshared_prefixes):
        return True, "reshared_filename"
    if filter_screenshots and is_screenshot(img, camera_model, has_gps):
        return True, "screenshot"
    # Accidental close-up (hand/skin against lens). Pure-image, no model — the
    # flesh fraction and blur score are precomputed in quality.assess and passed
    # in. blur_score defaults to +inf so a caller that omits it never trips this.
    if filter_accidental_closeup and is_accidental_closeup(
        flesh_fraction, blur_score, min_flesh_fraction, max_accidental_blur_score
    ):
        return True, "accidental_closeup"
    # Pitch-black / pure-white frame (lens cap, pocket misfire, blown flash).
    # Hard-excluded here because the selection stage can re-admit exposure
    # failures via its fallback pool. exposure_score defaults to 0.5 (neutral)
    # so a caller that omits it never trips this gate.
    if filter_pitch_black and is_pitch_black_or_pure_white(
        exposure_score, detail_stddev
    ):
        return True, "pitch_black_or_pure_white"
    if filter_home_private and is_home_private(lat, lon, face_count, home, home_radius_km):
        return True, "home_private"
    if filter_documents:
        if clip_emb is not None:
            if is_document_from_embedding(clip_emb):
                return True, "document"
        elif is_document_clip(img):
            return True, "document"
    # Binary text-heavy check — catches chat screenshots saved with camera filenames
    # that slip past the multi-class document detector due to probability dilution.
    if filter_text_heavy and clip_emb is not None:
        if is_text_heavy_from_embedding(clip_emb):
            return True, "text_heavy"
        # Structural backstop for camera photos of printed pages / calendars /
        # tables that CLIP misses (e.g. a phone shot of a wall calendar saved
        # as IMG_YYYYMMDD_HHMMSS.jpg). Runs only at ingest because it needs
        # the original image data — the cache reassess path can't replay it.
        if looks_like_text_document_page(img):
            return True, "text_document_page"
    # Boring / mundane object filter.
    if filter_boring_objects:
        if clip_emb is not None:
            # CLIP semantic detectors are the authority on "mundane object".
            # The pure-image heuristic (mundane_heuristic_score) is NOT a
            # standalone hard gate here: it scores snow fields, fog, clean
            # sky, beach horizons, minimalist and backlit frames ~1.0 and
            # would silently exclude legitimate photos. CLIP correctly reads
            # those as nature/landscape and leaves them in, so when an
            # embedding is available we defer to it entirely.
            if is_boring_from_embedding(clip_emb):
                return True, "boring_object"
            if is_mundane_object_from_embedding(clip_emb, threshold=mundane_object_threshold):
                return True, "mundane_object"
        elif (
            mundane_heuristic >= 0.80
            and detail_stddev < _FLAT_FRAME_DETAIL_MAX
            and face_count == 0
        ):
            # No embedding to corroborate (CLIP extraction failed). Fall back
            # to the structural heuristic, but only for a clearly flat,
            # subject-less frame — high uniformity AND no tonal detail AND no
            # detected face. Errs toward keeping ambiguous frames rather than
            # dropping them.
            return True, "mundane_heuristic"
    # Intimate content filter
    if filter_intimate_content and clip_emb is not None:
        if is_intimate_from_embedding(clip_emb):
            return True, "intimate_content"
    return False, ""
