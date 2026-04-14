"""
Unit tests for src/scene_tagger.py — zero-shot scene classification helpers.

Note: Tests that require the CLIP model (classify, top_label) are exercised
      with mocked label features so the heavy model is not loaded during CI.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from src.scene_tagger import (
    SCENE_LABELS,
    classify,
    tags_from_json,
    tags_to_json,
    top_label,
)


# ---------------------------------------------------------------------------
# SCENE_LABELS sanity tests
# ---------------------------------------------------------------------------

class TestSceneLabels:
    def test_labels_is_list(self):
        assert isinstance(SCENE_LABELS, list)

    def test_labels_not_empty(self):
        assert len(SCENE_LABELS) > 0

    def test_labels_are_strings(self):
        for label in SCENE_LABELS:
            assert isinstance(label, str)
            assert len(label) > 0


# ---------------------------------------------------------------------------
# tags_to_json / tags_from_json tests
# ---------------------------------------------------------------------------

class TestTagsJson:
    def test_empty_list_serialises(self):
        result = tags_to_json([])
        assert json.loads(result) == []

    def test_round_trip(self):
        tags = [("beach or ocean", 0.8), ("city street or urban scene", 0.15)]
        serialised = tags_to_json(tags)
        recovered = tags_from_json(serialised)
        assert len(recovered) == 2
        assert recovered[0][0] == "beach or ocean"
        assert recovered[0][1] == pytest.approx(0.8, abs=0.001)

    def test_tags_to_json_returns_string(self):
        result = tags_to_json([("beach or ocean", 0.9)])
        assert isinstance(result, str)

    def test_tags_from_json_empty_string(self):
        result = tags_from_json("")
        assert result == []

    def test_tags_from_json_invalid_json(self):
        result = tags_from_json("not valid json{{{")
        assert result == []

    def test_confidence_rounded(self):
        tags = [("forest or woods", 0.123456789)]
        serialised = tags_to_json(tags)
        obj = json.loads(serialised)
        # confidence should be rounded to 4 decimal places
        assert len(str(obj[0]["confidence"]).split(".")[-1]) <= 4

    def test_round_trip_many_labels(self):
        tags = [(label, 1.0 / len(SCENE_LABELS)) for label in SCENE_LABELS[:5]]
        serialised = tags_to_json(tags)
        recovered = tags_from_json(serialised)
        assert len(recovered) == 5


# ---------------------------------------------------------------------------
# classify tests (mocked label features)
# ---------------------------------------------------------------------------

def _fake_label_features() -> np.ndarray:
    """Generate fake (N_labels, 512) float32 feature matrix."""
    rng = np.random.default_rng(42)
    feats = rng.standard_normal((len(SCENE_LABELS), 512)).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / norms


class TestClassify:
    def test_none_embedding_returns_empty(self):
        result = classify(None)
        assert result == []

    def test_wrong_dimension_returns_empty(self):
        emb = np.ones(256, dtype=np.float32)
        result = classify(emb)
        assert result == []

    def test_returns_list_of_tuples(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = classify(emb, top_n=3)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_top_n_respected(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = classify(emb, top_n=2)
        assert len(result) <= 2

    def test_confidence_descending(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = classify(emb, top_n=5)
        confidences = [c for _, c in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_min_confidence_filter(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = classify(emb, top_n=10, min_confidence=0.5)
        for _, conf in result:
            assert conf >= 0.5

    def test_labels_are_in_scene_labels(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = classify(emb, top_n=5)
        for label, _ in result:
            assert label in SCENE_LABELS


# ---------------------------------------------------------------------------
# top_label tests
# ---------------------------------------------------------------------------

class TestTopLabel:
    def test_none_embedding_returns_empty_string(self):
        result = top_label(None)
        assert result == ""

    def test_returns_string(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = top_label(emb)
        assert isinstance(result, str)

    def test_returns_valid_label(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        import src.scene_tagger as st
        with patch.object(st, "_label_features", _fake_label_features()):
            result = top_label(emb)
        if result:
            assert result in SCENE_LABELS
