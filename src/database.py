"""
SQLite cache — stores extracted features to avoid reprocessing unchanged files.
Schema designed for append-heavy workloads with incremental updates.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: str) -> None:
    """Create tables if they don't exist; migrate existing tables with new columns."""
    with connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS photos (
                id              INTEGER PRIMARY KEY,
                path            TEXT UNIQUE NOT NULL,
                file_hash       TEXT NOT NULL,
                file_size       INTEGER DEFAULT 0,
                -- EXIF metadata
                timestamp       REAL    DEFAULT 0,
                lat             REAL    DEFAULT 0,
                lon             REAL    DEFAULT 0,
                camera_model    TEXT    DEFAULT '',
                has_gps         INTEGER DEFAULT 0,
                -- quality
                blur_score      REAL    DEFAULT 0,
                exposure_score  REAL    DEFAULT 0.5,
                resolution      INTEGER DEFAULT 0,
                quality_pass    INTEGER DEFAULT 1,
                -- features (binary blobs)
                clip_emb        BLOB,
                phash           TEXT    DEFAULT '',
                -- face
                face_count      INTEGER DEFAULT 0,
                face_emb        BLOB,
                face_prominence REAL    DEFAULT 0,
                face_confidence REAL    DEFAULT 0,
                -- flags
                is_duplicate    INTEGER DEFAULT 0,
                is_private      INTEGER DEFAULT 0,
                -- output stages
                cluster_id      INTEGER DEFAULT -1,
                score           REAL    DEFAULT 0,
                selected        INTEGER DEFAULT 0,
                processed_at    REAL    DEFAULT 0,
                -- NEW: aesthetic, scene, sentiment, identity
                aesthetic_score REAL    DEFAULT 0,
                scene_tags      TEXT    DEFAULT '',
                smile_score     REAL    DEFAULT 0.5,
                person_id       INTEGER DEFAULT -1,
                is_frequent     INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_hash    ON photos(file_hash);
            CREATE INDEX IF NOT EXISTS idx_cluster ON photos(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_person  ON photos(person_id);
        """)

        # Migrate existing databases that predate the new columns
        _add_column_if_missing(conn, "photos", "aesthetic_score",  "REAL DEFAULT 0")
        _add_column_if_missing(conn, "photos", "scene_tags",       "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "photos", "smile_score",      "REAL DEFAULT 0.5")
        _add_column_if_missing(conn, "photos", "person_id",        "INTEGER DEFAULT -1")
        _add_column_if_missing(conn, "photos", "is_frequent",      "INTEGER DEFAULT 0")
        _add_column_if_missing(conn, "photos", "face_prominence",  "REAL DEFAULT 0")
        _add_column_if_missing(conn, "photos", "face_confidence",  "REAL DEFAULT 0")


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """ALTER TABLE … ADD COLUMN — silently skips if column already exists.

    Only suppresses the "duplicate column name" error. Other OperationalError
    cases ("database is locked", "disk full", "attempt to write a readonly
    database") are re-raised so a broken migration fails loudly instead of
    silently leaving the schema half-applied.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError as exc:
        msg = (exc.args[0] if exc.args else "").lower()
        if "duplicate column name" in msg or "already exists" in msg:
            return
        raise


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def upsert(conn: sqlite3.Connection, data: dict) -> None:
    """Insert or update a photo record (keyed on path)."""
    cols = list(data.keys())
    placeholders = ",".join("?" * len(cols))
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "path")
    sql = (
        f"INSERT INTO photos({','.join(cols)}) VALUES({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}"
    )
    conn.execute(sql, [data[c] for c in cols])


def get_by_path(conn: sqlite3.Connection, path: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM photos WHERE path=?", (path,)).fetchone()


def get_all(
    conn: sqlite3.Connection, clause: str = "", params: tuple = ()
) -> list:
    """Return photos, optionally filtered by ``clause``.

    SECURITY: ``clause`` is interpolated directly into SQL — it MUST be a
    static, code-controlled string (never user input, file content, or any
    value derived from the photo library). Production code should prefer the
    typed helpers below (``get_with_clip_emb``, ``get_non_private``,
    ``get_kept``, ``get_with_face_count_at_least``); ``clause`` is retained
    for ad-hoc/test use against this internal SQLite cache only.
    """
    sql = "SELECT * FROM photos"
    if clause:
        sql += " WHERE " + clause
    return conn.execute(sql, params).fetchall()


# Typed filter helpers — preferred over raw `clause` strings. Each helper
# uses a fixed predicate; arguments flow through ``?`` placeholders only.

def get_with_clip_emb(conn: sqlite3.Connection) -> list:
    """Photos that have a CLIP embedding stored."""
    return conn.execute(
        "SELECT * FROM photos WHERE clip_emb IS NOT NULL"
    ).fetchall()


def get_non_private(conn: sqlite3.Connection) -> list:
    """Photos not flagged private."""
    return conn.execute(
        "SELECT * FROM photos WHERE is_private=0"
    ).fetchall()


def get_kept(conn: sqlite3.Connection) -> list:
    """Photos that survived privacy AND deduplication filters."""
    return conn.execute(
        "SELECT * FROM photos WHERE is_private=0 AND is_duplicate=0"
    ).fetchall()


def get_with_face_count_at_least(
    conn: sqlite3.Connection, min_faces: int
) -> list:
    """Photos containing at least ``min_faces`` detected faces."""
    return conn.execute(
        "SELECT * FROM photos WHERE face_count >= ?", (int(min_faces),)
    ).fetchall()


def update_fields(conn: sqlite3.Connection, path: str, **kwargs) -> None:
    if not kwargs:
        return
    sets = ",".join(f"{k}=?" for k in kwargs)
    conn.execute(
        f"UPDATE photos SET {sets} WHERE path=?",
        list(kwargs.values()) + [path],
    )


# ---------------------------------------------------------------------------
# Numpy ↔ BLOB helpers
# ---------------------------------------------------------------------------

def emb_to_blob(arr: Optional[np.ndarray]) -> Optional[bytes]:
    return None if arr is None else arr.astype(np.float32).tobytes()


def blob_to_emb(blob: Optional[bytes]) -> Optional[np.ndarray]:
    return None if not blob else np.frombuffer(blob, dtype=np.float32).copy()
