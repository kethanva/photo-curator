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
from typing import Optional, Tuple

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
]
_NORMAL_PROMPT = "a natural photo taken with a camera outdoors or indoors"

# Cached tokenised text tensors
_text_features = None
_clip_loaded = False


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


def is_document_clip(img: Image.Image, threshold: float = 0.35) -> bool:
    """
    CLIP zero-shot: return True if image looks like a document/private item.
    The threshold controls sensitivity — lower = stricter.
    """
    try:
        import torch
        from src.embeddings import load_model, extract

        text_feats, model, preprocess, device = _get_clip_text_features()

        image_emb = extract(img, model, preprocess, device)
        image_tensor = torch.tensor(image_emb, device=device).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)

        # Softmax over all prompts
        logits = (image_tensor @ text_feats.T * 100.0).softmax(dim=-1)
        probs = logits.cpu().numpy()[0]

        # Last prompt is the "normal photo" — sum of private prompt probs
        private_score = float(probs[:-1].sum())
        return private_score > threshold

    except Exception:
        return False


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
    filter_home_private: bool = False,
) -> bool:
    """
    Return True if the photo should be excluded from the curated output.

    Checks are ordered cheapest-first to avoid unnecessary model calls.
    """
    if filter_screenshots and is_screenshot(img, camera_model, has_gps):
        return True
    if filter_home_private and is_home_private(lat, lon, face_count, home, home_radius_km):
        return True
    if filter_documents and is_document_clip(img):
        return True
    return False
