"""
Tests for src/subject_priority.py — subject priority boost scoring.

All CLIP model calls are mocked so no GPU or model weights are required.
"""

import numpy as np
import pytest

from src import subject_priority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _record(path: str, emb: np.ndarray) -> dict:
    return {"path": path, "clip_emb": _fake_blob(emb)}


def _record_no_emb(path: str) -> dict:
    return {"path": path, "clip_emb": None}


# ---------------------------------------------------------------------------
# SUBJECT_PRESETS
# ---------------------------------------------------------------------------

class TestSubjectPresets:
    def test_all_presets_have_prompts(self):
        for name, prompts in subject_priority.SUBJECT_PRESETS.items():
            assert isinstance(prompts, list), f"{name} should have a list"
            assert len(prompts) >= 1, f"{name} should have at least one prompt"

    def test_preset_names_match_list_presets(self):
        assert set(subject_priority.list_presets()) == set(subject_priority.SUBJECT_PRESETS.keys())

    def test_priority_multipliers_values(self):
        m = subject_priority.PRIORITY_MULTIPLIERS
        assert m["high"] == 1.0
        assert 0 < m["medium"] < 1.0
        assert 0 < m["low"] < m["medium"]


# ---------------------------------------------------------------------------
# compute_scores — structural / no-ML cases
# ---------------------------------------------------------------------------

class TestComputeScoresStructural:
    def test_returns_dict_keyed_by_path(self):
        records = [_record("a.jpg", _unit_vec(seed=1)),
                   _record("b.jpg", _unit_vec(seed=2))]
        cfg = {"enabled": True, "subjects": [{"name": "human", "priority": "high"}]}
        # Monkeypatching _build_vectors to avoid CLIP load
        subject_priority._cache.clear()
        # Inject a fake cached vector for "human"
        subject_priority._cache["human"] = (1.0, _unit_vec(seed=99))

        scores = subject_priority.compute_scores(records, cfg)
        assert set(scores.keys()) == {"a.jpg", "b.jpg"}

    def test_empty_subjects_returns_zeros(self):
        records = [_record("a.jpg", _unit_vec())]
        cfg = {"enabled": True, "subjects": []}
        scores = subject_priority.compute_scores(records, cfg)
        assert scores["a.jpg"] == 0.0

    def test_missing_clip_emb_returns_zero(self):
        records = [_record_no_emb("no_emb.jpg")]
        subject_priority._cache["landscape"] = (1.0, _unit_vec(seed=5))
        cfg = {"enabled": True, "subjects": [{"name": "landscape", "priority": "high"}]}
        scores = subject_priority.compute_scores(records, cfg)
        assert scores["no_emb.jpg"] == 0.0

    def test_score_range_zero_to_one(self):
        subject_priority._cache.clear()
        vec = _unit_vec(seed=10)
        subject_priority._cache["nature"] = (0.6, _unit_vec(seed=20))
        records = [_record(f"img{i}.jpg", _unit_vec(seed=i)) for i in range(5)]
        cfg = {"enabled": True, "subjects": [{"name": "nature", "priority": "medium"}]}
        scores = subject_priority.compute_scores(records, cfg)
        for v in scores.values():
            assert 0.0 <= v <= 1.0, f"Score out of range: {v}"

    def test_matching_photo_scores_higher(self):
        """Photo whose embedding aligns with a subject direction should score higher."""
        subject_priority._cache.clear()
        direction = _unit_vec(seed=42)
        subject_priority._cache["human"] = (1.0, direction)

        # near-match: same direction
        near = direction.copy()
        # far-match: orthogonal direction
        far = _unit_vec(seed=7)
        # Orthogonalise far from direction
        far = far - np.dot(far, direction) * direction
        far = far / (np.linalg.norm(far) + 1e-8)

        records = [_record("near.jpg", near), _record("far.jpg", far)]
        cfg = {"enabled": True, "subjects": [{"name": "human", "priority": "high"}]}
        scores = subject_priority.compute_scores(records, cfg)
        assert scores["near.jpg"] > scores["far.jpg"]

    def test_high_priority_beats_medium_same_similarity(self):
        """Same cosine similarity but different priority → high > medium."""
        subject_priority._cache.clear()
        shared_dir = _unit_vec(seed=55)
        subject_priority._cache["high_subj"]   = (1.0, shared_dir)
        subject_priority._cache["medium_subj"] = (0.6, shared_dir)

        # Build a photo that aligns with shared_dir
        emb = shared_dir.copy()
        records = [_record("img.jpg", emb)]
        cfg = {
            "enabled": True,
            "subjects": [
                {"name": "high_subj",   "priority": "high"},
                {"name": "medium_subj", "priority": "medium"},
            ],
        }
        scores = subject_priority.compute_scores(records, cfg)
        # Score should reflect the highest weighted match (high_subj wins)
        assert scores["img.jpg"] == pytest.approx(1.0, abs=0.01)

    def test_empty_records_returns_empty_dict(self):
        subject_priority._cache["human"] = (1.0, _unit_vec())
        cfg = {"enabled": True, "subjects": [{"name": "human", "priority": "high"}]}
        assert subject_priority.compute_scores([], cfg) == {}


# ---------------------------------------------------------------------------
# Custom prompts config path (no CLIP required — cache bypass)
# ---------------------------------------------------------------------------

class TestCustomPrompts:
    def test_unknown_preset_with_custom_prompts_uses_cache(self):
        """A subject with no preset but injected into cache should work."""
        subject_priority._cache.clear()
        subject_priority._cache["custom_sport"] = (0.6, _unit_vec(seed=77))
        records = [_record("img.jpg", _unit_vec(seed=77))]  # same vec → high sim
        cfg = {
            "enabled": True,
            "subjects": [{"name": "custom_sport", "priority": "medium",
                          "prompts": ["sport action shot"]}],
        }
        scores = subject_priority.compute_scores(records, cfg)
        assert scores["img.jpg"] > 0.0
