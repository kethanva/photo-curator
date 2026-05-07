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
# Boring Objects Detection
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
    filter_intimate_content: bool = True,
) -> Tuple[int, int]:
    """
    Recompute is_private for every cached photo using only the stored CLIP
    embedding and the filename; updates rows in place.

    Only the document/text-overlay CLIP check and the filename filter are
    applied here — the screenshot and home-private checks rely on EXIF
    already captured correctly at ingest time and don't change between
    runs, so their cached flag contributions are preserved.

    Returns:
        (total_rows_examined, rows_changed)
    """
    from src import database

    rows = conn.execute(
        "SELECT path, clip_emb, is_private, camera_model, has_gps, face_count "
        "FROM photos"
    ).fetchall()

    changed = 0
    for row in rows:
        path = row["path"]
        clip_emb = database.blob_to_emb(row["clip_emb"])
        prev = int(row["is_private"])

        if clip_emb is None:
            continue

        doc_hit = is_document_from_embedding(clip_emb, threshold, document_ratio)
        reshared_hit = (
            filter_reshared and is_reshared_filename(path, reshared_prefixes)
        )
        text_hit = filter_text_heavy and is_text_heavy_from_embedding(clip_emb)
        boring_hit = filter_boring_objects and is_boring_from_embedding(clip_emb)
        intimate_hit = (
            filter_intimate_content and is_intimate_from_embedding(clip_emb)
        )
        new_flag_from_cache = bool(
            doc_hit or reshared_hit or text_hit or boring_hit or intimate_hit
        )

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
        elif prev:
            camera_model = (row["camera_model"] or "").strip()
            has_gps = int(row["has_gps"] or 0)
            face_count = int(row["face_count"] or 0)
            has_real_photo_signal = (
                bool(camera_model) or has_gps > 0 or face_count > 0
            )
            final_flag = 0 if has_real_photo_signal else 1
        else:
            final_flag = 0

        if final_flag != prev:
            conn.execute(
                "UPDATE photos SET is_private=? WHERE path=?",
                (final_flag, path),
            )
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
    filter_intimate_content: bool = True,
    filter_home_private: bool = False,
    filter_reshared: bool = True,
    path: Optional[Union[str, Path]] = None,
    reshared_prefixes: Optional[Iterable[str]] = None,
    clip_emb: Optional[np.ndarray] = None,
) -> bool:
    """
    Return True if the photo should be excluded from the curated output.

    Checks are ordered cheapest-first to avoid unnecessary model calls.
    When a pre-computed CLIP embedding is supplied the document check reuses
    it instead of running CLIP a second time on the same image.
    """
    if filter_reshared and path is not None and is_reshared_filename(path, reshared_prefixes):
        return True
    if filter_screenshots and is_screenshot(img, camera_model, has_gps):
        return True
    if filter_home_private and is_home_private(lat, lon, face_count, home, home_radius_km):
        return True
    if filter_documents:
        if clip_emb is not None:
            if is_document_from_embedding(clip_emb):
                return True
        elif is_document_clip(img):
            return True
    # Binary text-heavy check — catches chat screenshots saved with camera filenames
    # that slip past the multi-class document detector due to probability dilution.
    if filter_text_heavy and clip_emb is not None:
        if is_text_heavy_from_embedding(clip_emb):
            return True
        # Structural backstop for camera photos of printed pages / calendars /
        # tables that CLIP misses (e.g. a phone shot of a wall calendar saved
        # as IMG_YYYYMMDD_HHMMSS.jpg). Runs only at ingest because it needs
        # the original image data — the cache reassess path can't replay it.
        if looks_like_text_document_page(img):
            return True
    # Boring objects filter
    if filter_boring_objects and clip_emb is not None:
        if is_boring_from_embedding(clip_emb):
            return True
    # Intimate content filter
    if filter_intimate_content and clip_emb is not None:
        if is_intimate_from_embedding(clip_emb):
            return True
    return False
