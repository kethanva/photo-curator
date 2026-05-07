"""
Subject priority scoring — CLIP zero-shot matching against user-defined subjects.

Works entirely from pre-computed CLIP embeddings stored in the database; no
re-inference on images is required.  Prompt vectors are built once per process
(same pattern as aesthetic.py / scene_tagger.py).

Built-in subject presets cover common photography categories.  Any subject can
also supply custom prompts via config.yaml to override or extend the preset.

50 built-in presets grouped by category:
  People    : human, family, kids, friends
  Cycling   : bike, motorbike, skateboard, scooter
  Outdoor   : landscape, nature, hiking, beach, camping, snow, water, flowers
  Urban     : street, architecture, night, market, cafe
  Food      : food, drinks
  Events    : celebration, concert, festival
  Sport     : sports, gym, running, yoga, football, basketball, swimming
  Travel    : travel, cars, trains, planes, boats
  Animals   : pets, wildlife, birds
  Lifestyle : fashion, art, music, reading, gaming

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
          prompts:           # optional: override preset with custom prompts
            - "cycling or bicycle"
            - "mountain biking on a trail"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src import database

logger = logging.getLogger(__name__)

# CLIP's text encoder accepts up to 77 BPE tokens. Roughly: 1 token ≈ 4 chars
# of English text, but punctuation and unusual words can push the ratio
# higher. We pre-trim very long custom prompts to a safe character bound so
# `clip.tokenize` doesn't raise on a single user-supplied subject prompt
# and crash the entire ranking stage. 250 chars is a comfortable margin.
_MAX_PROMPT_CHARS = 250

# ---------------------------------------------------------------------------
# Built-in subject vocabulary
# ---------------------------------------------------------------------------

SUBJECT_PRESETS: Dict[str, List[str]] = {
    # ── People ───────────────────────────────────────────────────────────────
    "human": [
        "a photo of a person smiling",
        "portrait of a person",
        "people together enjoying themselves",
    ],
    "family": [
        "family portrait together",
        "parents and children spending time together",
        "family gathering at home or outdoors",
    ],
    "kids": [
        "children playing and having fun",
        "cute kids laughing",
        "child portrait outdoors",
    ],
    "friends": [
        "group of friends laughing together",
        "friends hanging out and socialising",
        "people enjoying time with their friends",
    ],

    # ── Cycling & wheeled sports ─────────────────────────────────────────────
    "bike": [
        "cycling or bicycle riding",
        "mountain biking on a trail",
        "road cycling with a bicycle",
    ],
    "motorbike": [
        "motorbike or motorcycle on a road",
        "motorcycle adventure riding",
        "biker on a motorbike",
    ],
    "skateboard": [
        "skateboarding tricks at a skatepark",
        "skateboarder performing a trick",
        "skateboard urban skating",
    ],
    "scooter": [
        "person riding a scooter or moped",
        "electric scooter commuting",
        "vespa scooter on a street",
    ],

    # ── Outdoor / adventure ──────────────────────────────────────────────────
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
    "hiking": [
        "hiking on a mountain trail",
        "trekking through nature with a backpack",
        "hiker enjoying a scenic view",
    ],
    "beach": [
        "beach with sand and ocean waves",
        "people relaxing on a sunny beach",
        "tropical beach paradise",
    ],
    "camping": [
        "camping in the wilderness with a tent",
        "campfire at night outdoors",
        "camping trip in the forest",
    ],
    "snow": [
        "snow covered landscape in winter",
        "skiing or snowboarding on a mountain",
        "winter snow scene",
    ],
    "water": [
        "water sports surfing or kayaking",
        "swimming in the ocean or lake",
        "sailing or boating on water",
    ],
    "flowers": [
        "beautiful flowers in bloom",
        "macro photography of flowers",
        "garden with colourful flowers",
    ],

    # ── Urban / social ───────────────────────────────────────────────────────
    "street": [
        "candid street photography",
        "urban life and street scene",
        "documentary street moment",
    ],
    "architecture": [
        "impressive building or architectural structure",
        "historic or modern architecture",
        "urban architectural detail",
    ],
    "night": [
        "night photography with city lights",
        "long exposure night scene",
        "stars and night sky astrophotography",
    ],
    "market": [
        "outdoor market or bazaar",
        "food market or farmers market",
        "busy street market with vendors",
    ],
    "cafe": [
        "cosy coffee shop or cafe",
        "person drinking coffee in a cafe",
        "cafe interior with latte art",
    ],

    # ── Food & drink ─────────────────────────────────────────────────────────
    "food": [
        "delicious food or meal",
        "beautifully plated restaurant dish",
        "food photography close-up",
    ],
    "drinks": [
        "colourful cocktail or drink",
        "coffee or tea photography",
        "wine or beer in a glass",
    ],

    # ── Events & celebration ─────────────────────────────────────────────────
    "celebration": [
        "birthday party or celebration",
        "wedding ceremony or reception",
        "festive gathering with friends and family",
    ],
    "concert": [
        "live music concert with stage lights",
        "musician performing on stage",
        "crowd at a music festival",
    ],
    "festival": [
        "outdoor festival or fair",
        "cultural festival with costumes and colour",
        "street festival celebration",
    ],

    # ── Sport & fitness ──────────────────────────────────────────────────────
    "sports": [
        "athletic sports activity or competition",
        "outdoor fitness and exercise",
        "sport action shot",
    ],
    "gym": [
        "gym workout or weight training",
        "fitness exercise indoors",
        "person lifting weights at the gym",
    ],
    "running": [
        "person running or jogging outdoors",
        "marathon or road race",
        "trail running in nature",
    ],
    "yoga": [
        "yoga pose or meditation outdoors",
        "yoga class or practice",
        "person doing yoga on a mat",
    ],
    "football": [
        "football or soccer match",
        "players kicking a football",
        "football game action shot",
    ],
    "basketball": [
        "basketball game or practice",
        "player shooting a basketball hoop",
        "basketball court action",
    ],
    "swimming": [
        "swimming in a pool or open water",
        "competitive swimming race",
        "person doing laps in a pool",
    ],

    # ── Travel & transport ───────────────────────────────────────────────────
    "travel": [
        "travel photography at a famous landmark",
        "tourist exploring a new city",
        "iconic destination or attraction",
    ],
    "cars": [
        "sports car or classic car",
        "automobile or vehicle photography",
        "car on an open road",
    ],
    "trains": [
        "train or railway photography",
        "vintage steam train or modern high-speed rail",
        "train station with passengers",
    ],
    "planes": [
        "aircraft or aeroplane photography",
        "plane taking off or landing",
        "airport terminal with planes",
    ],
    "boats": [
        "sailing boat or yacht on water",
        "boat trip on a river or sea",
        "harbour with boats and ships",
    ],

    # ── Animals & wildlife ───────────────────────────────────────────────────
    "pets": [
        "cute pet dog or cat portrait",
        "domestic animal looking at the camera",
        "adorable pet photograph",
    ],
    "wildlife": [
        "wild animal in its natural habitat",
        "wildlife photography in the savanna or jungle",
        "bird or animal in the wild",
    ],
    "birds": [
        "bird in flight or perched",
        "bird photography in nature",
        "colourful exotic bird",
    ],

    # ── Arts & lifestyle ─────────────────────────────────────────────────────
    "fashion": [
        "fashion photography with stylish outfit",
        "editorial fashion portrait",
        "clothing or style photography",
    ],
    "art": [
        "artwork or painting in a gallery",
        "street art or mural",
        "creative artistic photograph",
    ],
    "music": [
        "musician playing an instrument",
        "guitar or piano being played",
        "music practice or recording session",
    ],
    "reading": [
        "person reading a book",
        "cosy reading in a library or bookshop",
        "books and reading corner",
    ],
    "gaming": [
        "person playing video games",
        "gaming setup with monitor and controller",
        "esports or competitive gaming",
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

    Robust to per-subject failures: a bad prompt (too long for the 77-token
    CLIP context, broken Unicode, etc.) is skipped with a warning rather
    than crashing the entire ranking stage after eight other stages have
    completed.
    """
    missing = [s for s in subjects if s.get("name") and s["name"] not in _cache]
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

        # Truncate over-long prompts before tokenisation so a single bad
        # config entry can't crash the stage with a CLIP context overflow.
        safe_prompts = [str(p)[:_MAX_PROMPT_CHARS] for p in prompts]

        try:
            tokens = clip.tokenize(safe_prompts).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            direction = feats.mean(dim=0)
            direction = direction / direction.norm()
            _cache[name] = (
                priority, direction.cpu().numpy().astype(np.float32)
            )
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            # Skip this subject — its bucket will simply score 0 for every
            # photo. Other subjects continue to load. Logged at warning so
            # an operator can fix the offending prompt in config.yaml.
            logger.warning(
                "Subject '%s' could not be encoded (%s); bucket disabled "
                "for this run.", name, exc,
            )


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

    # Stack priority weights and direction matrix — restricted to the names
    # in the current ``subjects_cfg``. Without this filter, _cache (which is
    # module-level and accumulates across calls) would let a previous call's
    # subjects bleed into this call's scoring matrix.
    requested = {s["name"] for s in subjects}
    active = [
        (n, p, v) for n, (p, v) in _cache.items() if n in requested
    ]
    if not active:
        return {r["path"]: 0.0 for r in records}
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


def score_single_subject(
    records: List[dict],
    subject_name: str,
) -> Dict[str, float]:
    """
    Score each photo against a single named subject.

    Uses the built-in CLIP prompts for ``subject_name`` (from SUBJECT_PRESETS).
    Returns {path: cosine_similarity} — 0.0 for photos with missing embeddings
    or unknown subject names.
    """
    prompts = SUBJECT_PRESETS.get(subject_name)
    if not prompts:
        return {r["path"]: 0.0 for r in records}

    # Build the direction vector for this subject (cached)
    _build_vectors([{"name": subject_name, "priority": "high"}])
    entry = _cache.get(subject_name)
    if entry is None:
        return {r["path"]: 0.0 for r in records}

    _, direction = entry  # (priority_weight, direction_vector)

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

        sim = float(np.dot(direction, emb / norm))
        scores[path] = max(0.0, sim)

    return scores


def list_presets() -> List[str]:
    """Return names of all built-in subject presets."""
    return sorted(SUBJECT_PRESETS.keys())
