"""
Unit tests for src/database.py — SQLite schema, CRUD, and blob helpers.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src import database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    database.init_db(db_path)
    return db_path


@pytest.fixture
def conn(tmp_db: str):
    with database.connect(tmp_db) as c:
        yield c


# ---------------------------------------------------------------------------
# init_db / connect tests
# ---------------------------------------------------------------------------

class TestConnect:
    def test_creates_file(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        database.init_db(db_path)
        assert Path(db_path).exists()

    def test_returns_connection(self, tmp_db):
        with database.connect(tmp_db) as c:
            assert isinstance(c, sqlite3.Connection)

    def test_row_factory_set(self, tmp_db):
        with database.connect(tmp_db) as c:
            assert c.row_factory == sqlite3.Row

    def test_idempotent_init(self, tmp_db):
        """Calling init_db twice should not raise."""
        database.init_db(tmp_db)
        database.init_db(tmp_db)

    def test_photos_table_created(self, tmp_db):
        with database.connect(tmp_db) as c:
            tables = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = [t["name"] for t in tables]
        assert "photos" in names


# ---------------------------------------------------------------------------
# upsert / get_by_path tests
# ---------------------------------------------------------------------------

def _base_record(path: str = "a.jpg") -> dict:
    return {
        "path": path,
        "file_hash": "deadbeef",
        "file_size": 100_000,
        "timestamp": 1700000000.0,
        "lat": 37.7,
        "lon": -122.4,
        "camera_model": "iPhone 15",
        "has_gps": 1,
        "blur_score": 200.0,
        "exposure_score": 0.55,
        "resolution": 1080,
        "quality_pass": 1,
        "clip_emb": None,
        "phash": "abc123",
        "face_count": 2,
        "face_emb": None,
        "is_duplicate": 0,
        "is_private": 0,
        "cluster_id": -1,
        "score": 0.75,
        "selected": 0,
        "processed_at": 1700000000.0,
        "aesthetic_score": 0.6,
        "scene_tags": "",
        "smile_score": 0.8,
        "person_id": -1,
        "is_frequent": 0,
    }


class TestUpsert:
    def test_insert_new_record(self, conn):
        database.upsert(conn, _base_record("img1.jpg"))
        conn.commit()
        row = database.get_by_path(conn, "img1.jpg")
        assert row is not None
        assert row["file_hash"] == "deadbeef"

    def test_update_existing_record(self, conn):
        rec = _base_record("img1.jpg")
        database.upsert(conn, rec)
        conn.commit()

        rec["blur_score"] = 999.0
        database.upsert(conn, rec)
        conn.commit()

        row = database.get_by_path(conn, "img1.jpg")
        assert row["blur_score"] == pytest.approx(999.0)

    def test_upsert_with_blob_embedding(self, conn):
        emb = np.ones(512, dtype=np.float32)
        rec = _base_record("img2.jpg")
        rec["clip_emb"] = database.emb_to_blob(emb)
        database.upsert(conn, rec)
        conn.commit()

        row = database.get_by_path(conn, "img2.jpg")
        recovered = database.blob_to_emb(row["clip_emb"])
        np.testing.assert_array_almost_equal(recovered, emb)


class TestGetByPath:
    def test_missing_path_returns_none(self, conn):
        result = database.get_by_path(conn, "nonexistent.jpg")
        assert result is None

    def test_existing_path_returns_row(self, conn):
        database.upsert(conn, _base_record("photo.jpg"))
        conn.commit()
        row = database.get_by_path(conn, "photo.jpg")
        assert row is not None
        assert row["path"] == "photo.jpg"


# ---------------------------------------------------------------------------
# get_all tests
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_empty_table_returns_empty_list(self, conn):
        result = database.get_all(conn)
        assert result == []

    def test_returns_all_records(self, conn):
        for i in range(3):
            r = _base_record(f"img{i}.jpg")
            r["file_hash"] = f"hash{i}"
            database.upsert(conn, r)
        conn.commit()
        result = database.get_all(conn)
        assert len(result) == 3

    def test_where_clause_filters(self, conn):
        r1 = _base_record("good.jpg")
        r1["quality_pass"] = 1
        r2 = _base_record("bad.jpg")
        r2["quality_pass"] = 0
        r2["file_hash"] = "other"
        database.upsert(conn, r1)
        database.upsert(conn, r2)
        conn.commit()

        result = database.get_all(conn, "quality_pass = 1")
        paths = [r["path"] for r in result]
        assert "good.jpg" in paths
        assert "bad.jpg" not in paths


# ---------------------------------------------------------------------------
# update_fields tests
# ---------------------------------------------------------------------------

class TestUpdateFields:
    def test_updates_specified_columns(self, conn):
        database.upsert(conn, _base_record("x.jpg"))
        conn.commit()
        database.update_fields(conn, "x.jpg", blur_score=500.0, face_count=3)
        conn.commit()
        row = database.get_by_path(conn, "x.jpg")
        assert row["blur_score"] == pytest.approx(500.0)
        assert row["face_count"] == 3

    def test_no_kwargs_is_noop(self, conn):
        database.upsert(conn, _base_record("y.jpg"))
        conn.commit()
        database.update_fields(conn, "y.jpg")  # should not raise


# ---------------------------------------------------------------------------
# Blob helpers
# ---------------------------------------------------------------------------

class TestBlobHelpers:
    def test_none_emb_to_blob_returns_none(self):
        assert database.emb_to_blob(None) is None

    def test_none_blob_to_emb_returns_none(self):
        assert database.blob_to_emb(None) is None

    def test_empty_blob_returns_none(self):
        assert database.blob_to_emb(b"") is None

    def test_round_trip_float32(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        blob = database.emb_to_blob(original)
        recovered = database.blob_to_emb(blob)
        np.testing.assert_array_equal(recovered, original)

    def test_round_trip_512_dim(self):
        original = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        recovered = database.blob_to_emb(database.emb_to_blob(original))
        np.testing.assert_array_almost_equal(recovered, original)

    def test_blob_is_bytes(self):
        arr = np.zeros(10, dtype=np.float32)
        blob = database.emb_to_blob(arr)
        assert isinstance(blob, bytes)

    def test_recovered_is_writable(self):
        """blob_to_emb should return a mutable copy, not a read-only memoryview."""
        arr = np.ones(8, dtype=np.float32)
        recovered = database.blob_to_emb(database.emb_to_blob(arr))
        recovered[0] = 99.0  # should not raise
        assert recovered[0] == pytest.approx(99.0)
