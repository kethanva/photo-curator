"""
Subject priority scoring — CLIP zero-shot matching against user-defined subjects.

Works entirely from pre-computed CLIP embeddings stored in the database; no
re-inference on images is required.  Prompt vectors are built once per process
(same pattern as aesthetic.py / scene_tagger.py).

Built-in subject presets cover common photography categories.  Any subject can
also supply custom prompts via config.yaml to override or extend the preset.

Usage in ranking:
    subject_scores = compute_scores(records, cfg["subject_priority"])
    # {path: boost_score}  where boost_score ∈ [0, 1]

Config shape (config.yaml):
    subject_priority:
      enabled: true
      subjects:
        - name: human        # matches SUBJECT_PRESETS["human"]
          priority: high     # high / medium / low  →  1.0 / 0.6 / 0.3
        - name: bike
          priority: medium
          prompts:           # optional override (replaces preset)
            - "cycling or bicycle"
            - "mountain biking on a trail"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src import database

# ---------------------------------------------------------------------------
# Built-in subject vocabulary
# ---------------------------------------------------------------------------

SUBJECT_PRESETS: Dict[str, List[str]] = {
    "human": [
        "a photo of a person smiling",
        "portrait of a person",
        "people together enjoying themselves",
    ],
    "landscape": [
        "a beautiful scenic landscape",
        "dramatic mountains and open sky",
        "sweeping natural vista",
    ],
    "nature": [
        "nature photography with trees or plants",
        "peaceful forest or woodland",
        "wildlife in natural habitat",
    ],
    "bike": [
        "cycling or bicycle riding",
        "mountain biking on a trail",
        "road cycling with a bicycle",
    ],
    "food": [
        "delicious food or meal",
        "beautifully plated restaurant dish",
        "food photography close-up",
    ],
    "travel": [
        "travel photography at a famous landmark",
        "tourist exploring a new city",
        "iconic destination or attraction",
    ],
    "pets": [
        "cute pet dog or cat portrait",
        "domestic animal looking at the camera",
        "adorable pet photograph",
    ],
    "sports": [
        "athletic sports activity or competition",
        "outdoor fitness and exercise",
        "sport action shot",
    ],
    "celebration": [
        "birthday party or celebration",
        "wedding ceremony or reception",
        "festive gathering with friends and family",
    ],
    "architecture": [
        "impressive building or architectural structure",
        "historic or modern architecture",
        "urban architectural detail",
    ],
    "cars": [
        "sports car or classic car",
        "automobile or vehicle photography",
        "car on an open road",
    ],
    "night": [
        "night photography with city lights",
        "long exposure night scene",
        "stars and night sky astrophotography",
    ],
    "street": [
        "candid street photography",
        "urban life and street scene",
        "documentary street moment",
    ],
    "fashion": [
        "fashion photography with stylish outfit",
        "editorial fashion portrait",
        "clothing or style photography",
    ],
}

PRIORITY_MULTIPLIERS: Dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
}


# ---------------------------------------------------------------------------
# Prompt-vector cache  {subject_name: (priority_weight, direction_vector)}
# ---------------------------------------------------------------------------

_cache: Dict[str, Tuple[float, np.ndarray]] = {}


def _build_vectors(subjects: List[dict]) -> None:
    """
    Encode all configured subjects into CLIP text direction vectors.
    Only subjects not already cached are processed.
    """
    missing = [s for s in subjects if s["name"] not in _cache]
    if not missing:
        return

    import torch
    import clip
    from src.embeddings import load_model

    model, _, device = load_model()

    for subj in missing:
        name = subj["name"]
        priority = PRIORITY_MULTIPLIERS.get(subj.get("priority", "medium"), 0.6)

        # Custom prompts override presets; fall back to preset; skip if neither
        prompts: List[str] = subj.get("prompts") or SUBJECT_PRESETS.get(name, [])
        if not prompts:
            continue

        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        direction = feats.mean(dim=0)
        direction = direction / direction.norm()
        _cache[name] = (priority, direction.cpu().numpy().astype(np.float32))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scores(
    records: List[dict],
    subjects_cfg: dict,
) -> Dict[str, float]:
    """
    Compute a subject-priority boost score [0, 1] for each eligible photo.

    Args:
        records:      list of DB dicts; each must contain a 'clip_emb' BLOB
        subjects_cfg: the subject_priority section from config.yaml

    Returns:
        {path: boost_score}  — 0.0 for photos with no matching subjects or
        missing embeddings; higher values for strong subject matches.
    """
    subjects: List[dict] = subjects_cfg.get("subjects", [])
    if not subjects:
        return {r["path"]: 0.0 for r in records}

    _build_vectors(subjects)
    if not _cache:
        return {r["path"]: 0.0 for r in records}

    # Stack priority weights and direction matrix from cache
    active = [(n, p, v) for n, (p, v) in _cache.items()]
    priorities = np.array([p for _, p, _ in active], dtype=np.float32)    # (S,)
    directions = np.stack([v for _, _, v in active], axis=0)               # (S, 512)

    scores: Dict[str, float] = {}
    for rec in records:
        path = rec["path"]
        emb = database.blob_to_emb(rec.get("clip_emb"))

        if emb is None or emb.shape != (512,):
            scores[path] = 0.0
            continue

        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            scores[path] = 0.0
            continue

        emb = emb / norm
        # Cosine similarities against each subject direction: (S,)
        sims = directions @ emb

        # Weight by priority, take max weighted similarity
        weighted = sims * priorities
        best = float(np.clip(weighted.max(), 0.0, 1.0))
        scores[path] = best

    return scores


def list_presets() -> List[str]:
    """Return names of all built-in subject presets."""
    return sorted(SUBJECT_PRESETS.keys())
