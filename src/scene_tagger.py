"""
Zero-shot scene classification using CLIP.

Works from pre-computed CLIP embeddings — no re-inference required.
Returns a ranked list of scene labels for each photo.

Labels cover common personal photography contexts: travel, celebrations,
nature, social events, and daily life.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scene vocabulary
# ---------------------------------------------------------------------------

SCENE_LABELS: List[str] = [
    # Nature / outdoor
    "beach or ocean",
    "mountain or hiking trail",
    "forest or woods",
    "lake or river",
    "sunset or sunrise",
    "snowy landscape",
    "desert or canyon",
    # Urban / travel
    "city street or urban scene",
    "famous landmark or tourist attraction",
    "airport or travel",
    # Social / celebrations
    "birthday party",
    "wedding ceremony or reception",
    "graduation ceremony",
    "concert or live music",
    "sports event",
    "holiday or Christmas celebration",
    "family gathering or reunion",
    # Food & dining
    "restaurant or dining",
    "food or meal",
    # Indoor / daily life
    "indoor home",
    "office or work",
    "gym or fitness",
    # Portrait / people
    "portrait of a person",
    "group photo with multiple people",
    "child or children playing",
    "pets or animals",
]

# Prompt template — framing helps CLIP
_PROMPT_TEMPLATE = "a photo of {}"

# Cached text feature matrix (len(SCENE_LABELS), 512)
_label_features: Optional[np.ndarray] = None


def _build_label_features() -> None:
    global _label_features
    if _label_features is not None:
        return

    import torch
    import clip
    from src.embeddings import load_model

    model, _, device = load_model()

    prompts = [_PROMPT_TEMPLATE.format(label) for label in SCENE_LABELS]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        feats = model.encode_text(tokens)

    feats = feats / feats.norm(dim=-1, keepdim=True)
    _label_features = feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    clip_emb: np.ndarray,
    top_n: int = 3,
    min_confidence: float = 0.05,
) -> List[Tuple[str, float]]:
    """
    Return top-N scene labels for a pre-computed CLIP embedding.

    Args:
        clip_emb: normalised 512-dim float32 CLIP embedding
        top_n: number of top labels to return
        min_confidence: minimum softmax probability to include a label

    Returns:
        List of (label, confidence) tuples, sorted by confidence descending.
    """
    if clip_emb is None or len(clip_emb) != 512:
        return []

    try:
        _build_label_features()

        emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
        logits = emb @ _label_features.T  # (N_labels,)

        # Softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()

        top_idx = np.argsort(probs)[::-1][:top_n]
        return [
            (SCENE_LABELS[i], float(probs[i]))
            for i in top_idx
            if probs[i] >= min_confidence
        ]
    except (RuntimeError, ValueError, OSError) as exc:
        # Empty tags silently routes the scene bucket's budget into aesthetic.
        # Log so the operator can investigate (CLIP load issue, MPS error).
        logger.error("scene_tagger.classify failed: %s", exc)
        return []


def top_label(clip_emb: np.ndarray) -> str:
    """Return the single most likely scene label, or empty string."""
    results = classify(clip_emb, top_n=1)
    return results[0][0] if results else ""


def tags_to_json(tags: List[Tuple[str, float]]) -> str:
    """Serialise tag list to compact JSON string for database storage."""
    import json
    return json.dumps([{"label": t[0], "confidence": round(t[1], 4)} for t in tags])


def tags_from_json(s: str) -> List[Tuple[str, float]]:
    """Deserialise tags from database JSON string."""
    import json
    if not s:
        return []
    try:
        return [(d["label"], d["confidence"]) for d in json.loads(s)]
    except (ValueError, KeyError, TypeError) as exc:
        # Cached row predates the current scene_tags JSON shape — treat as
        # untagged. Logged at debug level since it's expected during a
        # schema/format migration window.
        logger.debug("scene_tags JSON parse failed (%s): %r", exc, s[:80])
        return []
