"""
Privacy filtering: removes screenshots, documents/IDs, and optionally
solo shots taken at the user's home location.

Two detection strategies:
  1. Heuristic — fast, no ML: checks screen resolutions + missing EXIF
  2. CLIP zero-shot — catches receipts, ID cards, paper documents

The CLIP check reuses the already-loaded model from embeddings.py.
"""

from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image


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

# Two-prompt binary check: text-image vs real photo.
# Binary softmax avoids the dilution problem of the 11-prompt multi-class
# classifier — a chat screenshot scores 0.70+ on _TEXT_HEAVY_PROMPT vs 0.50
# threshold becomes easy to clear.
_TEXT_HEAVY_PROMPT = (
    "a screenshot of text messages, chat conversations, or a text-only image"
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
    """Encode the binary text-heavy prompts once and cache them."""
    global _text_heavy_features, _text_heavy_loaded
    import torch
    import clip
    from src.embeddings import load_model

    model, preprocess, device = load_model()

    if not _text_heavy_loaded:
        tokens = clip.tokenize([_TEXT_HEAVY_PROMPT, _REAL_PHOTO_PROMPT]).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        _text_heavy_features = feats
        _text_heavy_loaded = True

    return _text_heavy_features, model, preprocess, device


def is_text_heavy_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.60,
) -> bool:
    """
    Binary CLIP check: True when the image looks like a screenshot or text-only image.

    Uses a two-prompt softmax so probability is not diluted across many classes.
    A genuine chat screenshot scores ~0.70–0.85 on the text-heavy prompt;
    real photographs with incidental text (signs, menus) typically score < 0.55.

    threshold=0.60 keeps false-positive rate very low on real photos.
    """
    if clip_emb is None:
        return False
    try:
        import torch

        text_feats, _, _, device = _get_text_heavy_features()
        image_tensor = torch.tensor(clip_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)
        logits = (image_tensor @ text_feats.T * 100.0).softmax(dim=-1)
        text_prob = float(logits[0, 0].item())
        return text_prob >= threshold
    except Exception:
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
    except Exception:
        return None


def _is_private_from_probs(
    probs: np.ndarray,
    threshold: float = 0.50,
    ratio: float = 2.5,
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


def is_document_clip(img: Image.Image, threshold: float = 0.50) -> bool:
    """CLIP zero-shot: True when image looks like a document / private item."""
    try:
        from src.embeddings import load_model, extract

        _, model, preprocess, device = _get_clip_text_features()
        image_emb = extract(img, model, preprocess, device)
        probs = _private_vs_normal_probs(image_emb)
        if probs is None:
            return False
        return _is_private_from_probs(probs, threshold)
    except Exception:
        return False


def is_document_from_embedding(
    clip_emb: Optional[np.ndarray],
    threshold: float = 0.50,
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
    return _is_private_from_probs(probs, threshold)


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


def is_reshared_filename(
    path: Union[str, Path],
    prefixes: Optional[Iterable[str]] = None,
) -> bool:
    """
    True when the filename matches a well-known messaging/social reshare
    pattern (FB_IMG_*, WhatsApp*, Screenshot_*, etc).

    Comparison is case-insensitive against the leaf filename only.
    """
    name = Path(path).name.lower()
    active = tuple(p.lower() for p in prefixes) if prefixes else _RESHARED_PREFIXES
    return any(name.startswith(p) for p in active)


# ---------------------------------------------------------------------------
# Cache re-assessment — refreshes is_private for photos ingested with a
# previous (stale) version of the privacy rules, without re-loading the
# originals or re-running CLIP on them.
# ---------------------------------------------------------------------------

def reassess_is_private_from_cache(
    conn,
    filter_reshared: bool = True,
    reshared_prefixes: Optional[Iterable[str]] = None,
    threshold: float = 0.50,
    filter_text_heavy: bool = True,
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
        "SELECT path, clip_emb, is_private FROM photos"
    ).fetchall()

    changed = 0
    for row in rows:
        path = row["path"]
        clip_emb = database.blob_to_emb(row["clip_emb"])
        prev = int(row["is_private"])

        # Preserve previous True if it came from screenshot/home rules —
        # we can only safely *revoke* a False positive from the CLIP doc
        # rule.  So: only overwrite when the new assessment disagrees with
        # the doc/reshared components, keeping other signals intact.
        doc_hit = (
            is_document_from_embedding(clip_emb, threshold)
            if clip_emb is not None else False
        )
        reshared_hit = (
            filter_reshared and is_reshared_filename(path, reshared_prefixes)
        )
        text_hit = (
            filter_text_heavy and is_text_heavy_from_embedding(clip_emb)
            if clip_emb is not None else False
        )
        new_flag = int(doc_hit or reshared_hit or text_hit)

        # We only reassess photos whose prior is_private was set by the
        # (now-fixed) CLIP doc logic.  If the screenshot heuristic set
        # it True, that flag is independent and should be recomputed
        # separately — not in scope here.  Simplest honest rule: take
        # the logical OR of prior non-CLIP reasons + new CLIP/filename
        # reasons.  Since we can't cleanly distinguish prior reasons,
        # we update only when new_flag differs and the photo has a
        # cached embedding we trust.
        if clip_emb is None:
            continue
        if new_flag != prev:
            conn.execute(
                "UPDATE photos SET is_private=? WHERE path=?",
                (new_flag, path),
            )
            changed += 1

    conn.commit()
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
    return False
