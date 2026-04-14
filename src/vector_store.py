"""
Unified vector store for CLIP and FaceNet embeddings.

Uses ChromaDB (HNSW-backed, local, no server required) to provide
O(log N) approximate nearest-neighbour (ANN) search.  Scales to 1M+ photos.

Replaces the in-memory numpy + scikit-learn approach that crashes past ~50,000
photos when all 512-dim embeddings are loaded at once.

ChromaDB persists to disk automatically — no Docker, no daemon, no network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

_CLIP_DIM = 512
_FACE_DIM = 512
_UPSERT_CHUNK = 5_000   # recommended batch ceiling for ChromaDB upserts


class VectorStore:
    """
    Wraps two ChromaDB collections:

    clip_embeddings
        512-dim CLIP ViT-B/32 vectors; cosine distance space.
        Used for semantic deduplication, event clustering, and scene search.

    face_embeddings
        512-dim FaceNet vectors; L2 (euclidean) distance space.
        Used for cross-photo identity clustering.

    Each item is keyed by the photo's filesystem path string.
    """

    def __init__(
        self,
        store_path: str,
        clip_collection: str = "clip_embeddings",
        face_collection: str = "face_embeddings",
    ) -> None:
        import chromadb

        Path(store_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=store_path)

        # CLIP: cosine distance (CLIP embeddings are L2-normalised)
        self._clip = self._client.get_or_create_collection(
            name=clip_collection,
            metadata={"hnsw:space": "cosine"},
        )
        # FaceNet: L2 distance (FaceNet embeddings are L2-normalised)
        self._face = self._client.get_or_create_collection(
            name=face_collection,
            metadata={"hnsw:space": "l2"},
        )

        log.info(
            "VectorStore ready — clip=%d  face=%d",
            self._clip.count(),
            self._face.count(),
        )

    # ------------------------------------------------------------------
    # ID listing
    # ------------------------------------------------------------------

    def clip_ids(self) -> set:
        """Return the set of all path IDs present in the CLIP collection."""
        result = self._clip.get(include=[])
        return set(result["ids"])

    def face_ids(self) -> set:
        """Return the set of all path IDs present in the face collection."""
        result = self._face.get(include=[])
        return set(result["ids"])

    # ------------------------------------------------------------------
    # Single upsert
    # ------------------------------------------------------------------

    def upsert_clip(self, path: str, embedding: np.ndarray) -> None:
        """Add or update a single CLIP embedding."""
        self._clip.upsert(
            ids=[path],
            embeddings=[embedding.astype(np.float32).tolist()],
        )

    def upsert_face(self, path: str, embedding: np.ndarray) -> None:
        """Add or update a single face embedding."""
        self._face.upsert(
            ids=[path],
            embeddings=[embedding.astype(np.float32).tolist()],
        )

    # ------------------------------------------------------------------
    # Batch upsert
    # ------------------------------------------------------------------

    def upsert_clip_batch(
        self,
        paths: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Batch-upsert CLIP embeddings; chunks internally to avoid ChromaDB limits."""
        self._upsert_batch(self._clip, paths, embeddings)

    def upsert_face_batch(
        self,
        paths: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Batch-upsert face embeddings; chunks internally."""
        self._upsert_batch(self._face, paths, embeddings)

    def _upsert_batch(self, col, paths: List[str], embeddings: np.ndarray) -> None:
        emb_list = embeddings.astype(np.float32).tolist()
        for start in range(0, len(paths), _UPSERT_CHUNK):
            end = start + _UPSERT_CHUNK
            col.upsert(
                ids=paths[start:end],
                embeddings=emb_list[start:end],
            )

    # ------------------------------------------------------------------
    # ANN search
    # ------------------------------------------------------------------

    def search_clip(
        self,
        query: np.ndarray,
        n_results: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Return up to n_results (path, distance) pairs nearest to `query`.

        Distance is cosine distance in [0, 2]; lower = more similar.
        For normalised vectors:  cosine_similarity ≈ 1 − distance.
        """
        return self._search(self._clip, query, n_results)

    def search_face(
        self,
        query: np.ndarray,
        n_results: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Return up to n_results (path, distance) pairs nearest to `query`.

        Distance is L2 (Euclidean).
        """
        return self._search(self._face, query, n_results)

    def _search(
        self, col, query: np.ndarray, n_results: int
    ) -> List[Tuple[str, float]]:
        count = col.count()
        if count == 0:
            return []
        n = min(n_results, count)
        result = col.query(
            query_embeddings=[query.astype(np.float32).tolist()],
            n_results=n,
            include=["distances"],
        )
        return list(zip(result["ids"][0], result["distances"][0]))

    # ------------------------------------------------------------------
    # Bulk fetch  (used by DBSCAN clustering stages)
    # ------------------------------------------------------------------

    def get_all_clip(self) -> Tuple[List[str], np.ndarray]:
        """
        Return (paths, matrix) for every CLIP embedding in the store.

        matrix shape: (N, 512),  dtype float32.
        Returned in arbitrary order — use the paths list as an index.
        """
        return self._get_all(self._clip)

    def get_all_face(self) -> Tuple[List[str], np.ndarray]:
        """
        Return (paths, matrix) for every face embedding in the store.

        matrix shape: (N, 512),  dtype float32.
        """
        return self._get_all(self._face)

    def _get_all(self, col) -> Tuple[List[str], np.ndarray]:
        if col.count() == 0:
            return [], np.zeros((0, _CLIP_DIM), dtype=np.float32)
        result = col.get(include=["embeddings"])
        paths = result["ids"]
        matrix = np.array(result["embeddings"], dtype=np.float32)
        return paths, matrix

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def clip_count(self) -> int:
        """Number of CLIP embeddings stored."""
        return self._clip.count()

    def face_count(self) -> int:
        """Number of face embeddings stored."""
        return self._face.count()
