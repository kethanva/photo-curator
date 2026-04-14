"""
Unit tests for src/vector_store.py — ChromaDB-backed vector store.
ChromaDB is mocked so tests run without the library installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Stub chromadb before importing the module under test
_chromadb_stub = MagicMock()
sys.modules.setdefault("chromadb", _chromadb_stub)

from src.vector_store import VectorStore, _CLIP_DIM, _UPSERT_CHUNK


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------

def _make_store(tmp_path) -> tuple:
    """Create a VectorStore backed by mock ChromaDB collections."""
    mock_clip_col = MagicMock()
    mock_face_col = MagicMock()

    mock_client = MagicMock()
    mock_client.get_or_create_collection.side_effect = [mock_clip_col, mock_face_col]
    _chromadb_stub.PersistentClient.return_value = mock_client

    mock_clip_col.count.return_value = 0
    mock_face_col.count.return_value = 0

    store = VectorStore(str(tmp_path / "store"))
    return store, mock_clip_col, mock_face_col


def _make_emb(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# clip_ids / face_ids tests
# ---------------------------------------------------------------------------

class TestIdListing:
    def test_clip_ids_returns_set(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.get.return_value = {"ids": ["a.jpg", "b.jpg"]}
        result = store.clip_ids()
        assert result == {"a.jpg", "b.jpg"}

    def test_face_ids_returns_set(self, tmp_path):
        store, _, face_col = _make_store(tmp_path)
        face_col.get.return_value = {"ids": ["c.jpg"]}
        result = store.face_ids()
        assert result == {"c.jpg"}

    def test_empty_collection_returns_empty_set(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.get.return_value = {"ids": []}
        assert store.clip_ids() == set()


# ---------------------------------------------------------------------------
# upsert_clip / upsert_face tests
# ---------------------------------------------------------------------------

class TestSingleUpsert:
    def test_upsert_clip_calls_collection(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        emb = _make_emb(0)
        store.upsert_clip("photo.jpg", emb)
        clip_col.upsert.assert_called_once()
        call_kwargs = clip_col.upsert.call_args
        assert call_kwargs[1]["ids"] == ["photo.jpg"]

    def test_upsert_face_calls_collection(self, tmp_path):
        store, _, face_col = _make_store(tmp_path)
        emb = _make_emb(1)
        store.upsert_face("face.jpg", emb)
        face_col.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# upsert_clip_batch / upsert_face_batch tests
# ---------------------------------------------------------------------------

class TestBatchUpsert:
    def test_batch_upsert_clip(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        paths = [f"{i}.jpg" for i in range(5)]
        embs = np.stack([_make_emb(i) for i in range(5)])
        store.upsert_clip_batch(paths, embs)
        clip_col.upsert.assert_called()

    def test_batch_chunks_at_limit(self, tmp_path):
        """Batches larger than _UPSERT_CHUNK should result in multiple calls."""
        store, clip_col, _ = _make_store(tmp_path)
        n = _UPSERT_CHUNK + 1
        paths = [f"{i}.jpg" for i in range(n)]
        embs = np.stack([_make_emb(i % 100) for i in range(n)])
        store.upsert_clip_batch(paths, embs)
        assert clip_col.upsert.call_count == 2

    def test_empty_batch_no_calls(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        store.upsert_clip_batch([], np.zeros((0, 512), dtype=np.float32))
        clip_col.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# search_clip / search_face tests
# ---------------------------------------------------------------------------

class TestSearch:
    def test_empty_collection_returns_empty_list(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 0
        result = store.search_clip(_make_emb(0), n_results=5)
        assert result == []

    def test_search_clip_returns_path_distance_pairs(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 3
        clip_col.query.return_value = {
            "ids": [["a.jpg", "b.jpg"]],
            "distances": [[0.1, 0.3]],
        }
        result = store.search_clip(_make_emb(0), n_results=2)
        assert result == [("a.jpg", 0.1), ("b.jpg", 0.3)]

    def test_search_respects_n_results(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 100
        clip_col.query.return_value = {
            "ids": [["a.jpg", "b.jpg"]],
            "distances": [[0.1, 0.2]],
        }
        store.search_clip(_make_emb(0), n_results=2)
        call_kwargs = clip_col.query.call_args[1]
        assert call_kwargs["n_results"] == 2

    def test_n_results_capped_by_collection_size(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 3  # only 3 items
        clip_col.query.return_value = {
            "ids": [["a.jpg"]],
            "distances": [[0.1]],
        }
        store.search_clip(_make_emb(0), n_results=100)
        call_kwargs = clip_col.query.call_args[1]
        assert call_kwargs["n_results"] == 3  # capped to collection size


# ---------------------------------------------------------------------------
# get_all_clip / get_all_face tests
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_empty_collection_returns_empty(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 0
        paths, matrix = store.get_all_clip()
        assert paths == []
        assert matrix.shape == (0, _CLIP_DIM)

    def test_returns_paths_and_matrix(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 2
        emb1 = _make_emb(0).tolist()
        emb2 = _make_emb(1).tolist()
        clip_col.get.return_value = {
            "ids": ["a.jpg", "b.jpg"],
            "embeddings": [emb1, emb2],
        }
        paths, matrix = store.get_all_clip()
        assert paths == ["a.jpg", "b.jpg"]
        assert matrix.shape == (2, 512)
        assert matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# clip_count / face_count tests
# ---------------------------------------------------------------------------

class TestCounts:
    def test_clip_count(self, tmp_path):
        store, clip_col, _ = _make_store(tmp_path)
        clip_col.count.return_value = 42
        assert store.clip_count() == 42

    def test_face_count(self, tmp_path):
        store, _, face_col = _make_store(tmp_path)
        face_col.count.return_value = 7
        assert store.face_count() == 7
