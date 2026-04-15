"""
Photo Curator — Intelligent local photo curation pipeline.

Processes a folder of photos (10 GB+) and produces a curated ~1 GB selection.
Optimised for Apple Silicon (M1/M2/M3) via MPS acceleration.

9-stage pipeline:
  1  Scan          — recursive folder scan, skip cached
  2  Extract       — EXIF, quality, CLIP, face, privacy
  3  Aesthetic     — CLIP zero-shot aesthetic scoring
  4  Scene tags    — zero-shot scene classification
  5  Sentiment     — smile + eyes-open via MediaPipe
  6  Deduplication — pHash + embedding similarity
  7  Clustering    — DBSCAN event groups (time + GPS + CLIP)
  8  Faces         — cross-photo identity clustering
  9  Rank & select — 30/30/40 diversity strategy + resize output

Usage:
    cd photo_curator
    python main.py                             # uses config.yaml
    python main.py --input /photos --output /best
    python main.py --dry-run                   # score only, no file copy
    python main.py --from-stage 6             # resume from deduplication
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src import (
    aesthetic,
    clustering,
    database,
    deduplication,
    embeddings,
    face_clustering,
    face_detection,
    ingestion,
    metadata,
    privacy,
    quality,
    ranking,
    scene_tagger,
    selection,
    sentiment,
    subject_priority,
    vector_store as vs,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _home_coords(cfg: dict):
    home = cfg["privacy"].get("home_coords")
    return tuple(home) if home else None


def _init_vector_store(cfg: dict) -> "vs.VectorStore":
    """Create or open the persistent ChromaDB vector store."""
    vs_cfg = cfg.get("vector_store", {})
    store_path = vs_cfg.get("path", "cache/vector_store")
    clip_col   = vs_cfg.get("clip_collection", "clip_embeddings")
    face_col   = vs_cfg.get("face_collection", "face_embeddings")
    return vs.VectorStore(store_path, clip_col, face_col)


def _sync_store_from_sqlite(db_path: str, store: "vs.VectorStore") -> int:
    """
    Migrate any CLIP / face BLOBs that are in SQLite but not yet in the
    vector store.  Runs once at pipeline start — subsequent runs are no-ops
    because ChromaDB persists to disk.

    Returns the number of CLIP embeddings synced.
    """
    existing_clip = store.clip_ids()
    existing_face = store.face_ids()

    with database.connect(db_path) as conn:
        rows = database.get_all(conn, "clip_emb IS NOT NULL")

    clip_paths, clip_embs = [], []
    face_paths, face_embs = [], []

    for row in rows:
        path = row["path"]
        if path not in existing_clip:
            emb = database.blob_to_emb(row["clip_emb"])
            if emb is not None:
                clip_paths.append(path)
                clip_embs.append(emb)

        if row["face_emb"] and path not in existing_face:
            femb = database.blob_to_emb(row["face_emb"])
            if femb is not None:
                face_paths.append(path)
                face_embs.append(femb)

    if clip_paths:
        store.upsert_clip_batch(clip_paths, np.vstack(clip_embs))
    if face_paths:
        store.upsert_face_batch(face_paths, np.vstack(face_embs))

    if clip_paths or face_paths:
        print(
            f"  Synced {len(clip_paths)} CLIP + {len(face_paths)} face "
            "embeddings from SQLite → vector store"
        )
    return len(clip_paths)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_extract(cfg: dict, paths, db_path: str, store: "vs.VectorStore") -> None:
    """Stage 2: EXIF + quality + CLIP + face + privacy."""
    max_dim = cfg["ingestion"]["max_dimension"]
    q_cfg   = cfg["quality"]
    p_cfg   = cfg["privacy"]
    home    = _home_coords(cfg)

    print("  Loading CLIP model…")
    clip_model, clip_preprocess, clip_device = embeddings.load_model()
    print(f"  CLIP ready on {clip_device}")

    processed = skipped = 0

    with database.connect(db_path) as conn:
        for path in tqdm(paths, desc="  Extracting", unit="img"):
            path_str  = str(path)
            file_hash = ingestion.compute_file_hash(path)

            cached = database.get_by_path(conn, path_str)
            if cached and cached["file_hash"] == file_hash:
                skipped += 1
                continue

            img, orig_shorter = ingestion.load_image_safe(path, max_dim)
            if img is None:
                continue

            exif       = metadata.extract_exif(path)
            q          = quality.assess(img, **{k: q_cfg[k] for k in (
                             "min_blur_score", "min_exposure_score",
                             "max_exposure_score", "min_resolution")},
                             orig_resolution=orig_shorter)
            clip_emb   = embeddings.extract(img, clip_model, clip_preprocess, clip_device)
            phash_str  = deduplication.compute_phash(img)
            face_count, face_emb = face_detection.detect(img)
            is_priv    = privacy.assess(
                img=img,
                camera_model=exif["camera_model"],
                has_gps=exif["has_gps"],
                lat=exif["lat"], lon=exif["lon"],
                face_count=face_count,
                home=home,
                home_radius_km=p_cfg["home_radius_km"],
                filter_screenshots=p_cfg["filter_screenshots"],
                filter_documents=p_cfg["filter_documents"],
                filter_home_private=p_cfg["filter_home_private"],
            )

            database.upsert(conn, {
                "path": path_str,
                "file_hash": file_hash,
                "file_size": path.stat().st_size,
                "timestamp": exif["timestamp"],
                "lat": exif["lat"], "lon": exif["lon"],
                "camera_model": exif["camera_model"],
                "has_gps": int(exif["has_gps"]),
                "blur_score": q.blur_score,
                "exposure_score": q.exposure_score,
                "resolution": q.resolution,
                "quality_pass": int(q.passes),
                "clip_emb": database.emb_to_blob(clip_emb),
                "phash": phash_str,
                "face_count": face_count,
                "face_emb": database.emb_to_blob(face_emb),
                "is_private": int(is_priv),
                "processed_at": time.time(),
            })

            # Mirror embeddings into the vector store for O(log N) search
            store.upsert_clip(path_str, clip_emb)
            if face_emb is not None:
                store.upsert_face(path_str, face_emb)

            processed += 1

        conn.commit()

    print(f"  Processed {processed} new | Skipped {skipped} cached")


def stage_aesthetic(cfg: dict, db_path: str) -> None:
    """Stage 3: CLIP-based aesthetic scoring from stored embeddings."""
    use_laion = cfg.get("aesthetic", {}).get("use_laion_predictor", False)

    with database.connect(db_path) as conn:
        rows = database.get_all(conn, "clip_emb IS NOT NULL")
        updates = 0
        for row in rows:
            emb = database.blob_to_emb(row["clip_emb"])
            score = aesthetic.score_from_embedding(emb, use_laion=use_laion)
            database.update_fields(conn, row["path"], aesthetic_score=score)
            updates += 1
        conn.commit()

    print(f"  Scored aesthetics for {updates} photos")


def stage_scene(cfg: dict, db_path: str) -> None:
    """Stage 4: Zero-shot scene classification from stored CLIP embeddings."""
    top_n = cfg.get("scene_tagging", {}).get("top_n", 3)

    with database.connect(db_path) as conn:
        rows = database.get_all(conn, "clip_emb IS NOT NULL")
        updates = 0
        for row in rows:
            emb = database.blob_to_emb(row["clip_emb"])
            tags = scene_tagger.classify(emb, top_n=top_n)
            json_str = scene_tagger.tags_to_json(tags)
            database.update_fields(conn, row["path"], scene_tags=json_str)
            updates += 1
        conn.commit()

    print(f"  Tagged scenes for {updates} photos")


def stage_sentiment(cfg: dict, db_path: str, paths) -> None:
    """Stage 5: MediaPipe smile + eyes-open scoring."""
    if not cfg.get("sentiment", {}).get("enabled", True):
        print("  Sentiment scoring disabled — skipping")
        return

    max_dim = cfg["ingestion"]["max_dimension"]
    min_faces = cfg.get("sentiment", {}).get("min_face_for_sentiment", 1)

    with database.connect(db_path) as conn:
        rows = database.get_all(conn, "face_count >= ?", (min_faces,))
        path_to_row = {row["path"]: row for row in rows}
        updates = 0

        for path in tqdm(
            [p for p in paths if str(p) in path_to_row],
            desc="  Sentiment",
            unit="img",
        ):
            img, _ = ingestion.load_image_safe(path, max_dim)
            if img is None:
                continue
            score = sentiment.score_image(img)
            database.update_fields(conn, str(path), smile_score=score)
            updates += 1

        conn.commit()

    print(f"  Scored sentiment for {updates} photos")


def stage_dedup(cfg: dict, db_path: str, store: "vs.VectorStore") -> int:
    """Stage 6: Mark near-duplicate photos.

    Uses ANN-accelerated deduplication via the vector store when embeddings
    are available (O(N·K) instead of O(N²)).  Falls back to the brute-force
    scan if the store is empty (e.g. first run before sync completes).
    """
    dup_cfg = cfg["deduplication"]

    with database.connect(db_path) as conn:
        rows = database.get_all(conn, "quality_pass=1 AND is_private=0")
        records = [
            {
                "path":       r["path"],
                "file_hash":  r["file_hash"],
                "phash":      r["phash"],
                "clip_emb":   database.blob_to_emb(r["clip_emb"]),
                "blur_score": r["blur_score"],
            }
            for r in rows
        ]

    if store.clip_count() > 0:
        dup_paths = deduplication.find_duplicates_ann(
            records,
            store,
            phash_threshold=dup_cfg["phash_threshold"],
            embedding_threshold=dup_cfg["embedding_similarity"],
        )
    else:
        # Fallback: brute-force O(N²) scan (small libraries or first run)
        dup_paths = deduplication.find_duplicates(
            records,
            phash_threshold=dup_cfg["phash_threshold"],
            embedding_threshold=dup_cfg["embedding_similarity"],
        )

    with database.connect(db_path) as conn:
        # Reset all before applying new flags to make stage idempotent
        conn.execute("UPDATE photos SET is_duplicate=0 WHERE quality_pass=1 AND is_private=0")
        for p in dup_paths:
            database.update_fields(conn, p, is_duplicate=1)
        conn.commit()

    print(f"  Marked {len(dup_paths)} duplicates")
    return len(dup_paths)


def stage_cluster(cfg: dict, db_path: str, store: "vs.VectorStore") -> int:
    """Stage 7: DBSCAN event clustering.

    CLIP embeddings are fetched from the vector store instead of decoding
    SQLite BLOBs, avoiding the memory spike that caused crashes at 50k photos.
    """
    cl_cfg = cfg["clustering"]

    # Fetch scalar metadata only — no BLOB columns needed
    with database.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT path, timestamp, lat, lon FROM photos "
            "WHERE quality_pass=1 AND is_private=0 AND is_duplicate=0"
        ).fetchall()

    # Pull all CLIP embeddings from the vector store in one batch
    clip_paths, clip_matrix = store.get_all_clip()
    clip_index: dict = dict(zip(clip_paths, clip_matrix))

    records = [
        {
            "path":      r["path"],
            "timestamp": r["timestamp"],
            "lat":       r["lat"],
            "lon":       r["lon"],
            "clip_emb":  clip_index.get(r["path"]),
        }
        for r in rows
    ]

    cluster_map = clustering.cluster_events(
        records,
        eps=cl_cfg["eps"],
        min_samples=cl_cfg["min_samples"],
        time_weight=cl_cfg["time_weight"],
        gps_weight=cl_cfg["gps_weight"],
        visual_weight=cl_cfg["visual_weight"],
    )

    with database.connect(db_path) as conn:
        for path, cid in cluster_map.items():
            database.update_fields(conn, path, cluster_id=cid)
        conn.commit()

    n_events = len({v for v in cluster_map.values() if v >= 0})
    print(f"  Found {n_events} event clusters  ({len(cluster_map)} photos)")
    return n_events


def stage_face_clustering(cfg: dict, db_path: str, store: "vs.VectorStore") -> None:
    """Stage 8: Cross-photo person identity clustering.

    Face embeddings are fetched from the vector store instead of decoding
    SQLite BLOBs.
    """
    fc_cfg = cfg.get("face_clustering", {})
    if not fc_cfg.get("enabled", True):
        print("  Face clustering disabled — skipping")
        return

    # Fetch scalar metadata only
    with database.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT path, face_count FROM photos WHERE face_count >= 1"
        ).fetchall()

    # Pull all face embeddings from the vector store in one batch
    face_paths, face_matrix = store.get_all_face()
    face_index: dict = dict(zip(face_paths, face_matrix))

    records = [
        {
            "path":       r["path"],
            "face_count": r["face_count"],
            "face_emb":   face_index.get(r["path"]),
        }
        for r in rows
    ]

    identity_map = face_clustering.cluster_identities(
        records,
        eps=fc_cfg.get("eps", 0.8),
        min_samples=fc_cfg.get("min_samples", 2),
        frequent_threshold=fc_cfg.get("frequent_person_threshold", 5),
    )

    with database.connect(db_path) as conn:
        for path, (pid, is_freq) in identity_map.items():
            database.update_fields(conn, path, person_id=pid, is_frequent=int(is_freq))
        conn.commit()

    frequent_ids = face_clustering.get_frequent_people(identity_map)
    print(f"  Identified {len(frequent_ids)} frequent people  ({len(identity_map)} face-tagged photos)")


def stage_rank_and_select(cfg: dict, db_path: str, dry_run: bool, total_photos: int = 0) -> tuple:
    """Stage 9: Rank, select via dynamic buckets, and write output."""
    rank_cfg = cfg["ranking"]["weights"]
    sel_cfg  = cfg["selection"]
    out_cfg  = cfg["output"]

    with database.connect(db_path) as conn:
        rows    = database.get_all(conn, "quality_pass=1 AND is_private=0 AND is_duplicate=0")
        records = [dict(r) for r in rows]

    # ── Score ─────────────────────────────────────────────────────
    scores = ranking.score_photos(records, rank_cfg)

    with database.connect(db_path) as conn:
        for path, score in scores.items():
            database.update_fields(conn, path, score=score)
        conn.commit()

    print(f"  Scored {len(scores)} photos")

    # ── Dynamic buckets ──────────────────────────────────────────
    bucket_cfg = sel_cfg.get("buckets", {
        "people": 0.30, "location": 0.30, "aesthetic": 0.40,
    })

    # Identify CLIP subject buckets (anything that isn't people/location/aesthetic)
    subject_bucket_names = [
        name for name in bucket_cfg
        if name not in ("people", "location", "aesthetic")
    ]

    # Compute per-subject CLIP similarity scores
    bucket_subject_scores: dict = {}
    if subject_bucket_names and records:
        print(f"  Computing CLIP scores for subject buckets: "
              f"{', '.join(subject_bucket_names)}…")
        for name in subject_bucket_names:
            bucket_subject_scores[name] = subject_priority.score_single_subject(
                records, name,
            )
            matched = sum(1 for v in bucket_subject_scores[name].values() if v > 0.15)
            print(f"    {name}: {matched}/{len(records)} photos matched")

    # ── Select ────────────────────────────────────────────────────
    output_mode = sel_cfg.get("output_mode", "percentage")
    output_pct  = sel_cfg.get("output_percentage", 0.15)

    selected = selection.select_photos(
        records,
        scores,
        max_bytes=sel_cfg["max_output_bytes"],
        max_per_cluster_pct=sel_cfg.get("max_per_cluster_pct", 0.20),
        max_per_person_pct=sel_cfg.get("max_per_person_pct", 0.10),
        max_per_location_pct=sel_cfg.get("max_per_location_pct", 0.30),
        buckets=bucket_cfg,
        subject_scores=bucket_subject_scores,
        output_long_side=sel_cfg.get("output_long_side", 2560),
        output_jpeg_quality=sel_cfg.get("output_jpeg_quality", 92),
        output_mode=output_mode,
        output_percentage=output_pct,
        total_photos=total_photos,
    )

    pct_label = (
        f" ({output_pct*100:.0f}% of {total_photos or len(records)} input photos"
        f", {len(records)} eligible)"
        if output_mode == "percentage" else ""
    )
    print(f"  Selected {len(selected)} photos{pct_label}")

    if not dry_run:
        selection.copy_to_output(
            selected,
            scores,
            output_dir=cfg["paths"]["output"],
            resize=sel_cfg.get("resize_output", True),
            long_side=sel_cfg.get("output_long_side", 2560),
            jpeg_quality=sel_cfg.get("output_jpeg_quality", 92),
            generate_report=out_cfg["generate_report"],
            report_filename=out_cfg["report_filename"],
        )
    else:
        print("  [dry-run] Skipping file write")

    return scores, selected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    config_path: str = "config.yaml",
    input_override: str = None,
    output_override: str = None,
    dry_run: bool = False,
    from_stage: int = 1,
) -> None:
    cfg = load_config(config_path)
    if input_override:
        cfg["paths"]["input"] = input_override
    if output_override:
        cfg["paths"]["output"] = output_override

    db_path    = cfg["paths"]["cache"]
    input_dir  = cfg["paths"]["input"]
    output_dir = cfg["paths"]["output"]

    database.init_db(db_path)

    bar = "=" * 60
    print(f"\n{bar}")
    print("  Photo Curator Pipeline")
    print(bar)
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Cache:   {db_path}")
    if dry_run:
        print("  Mode:    DRY RUN (no files written)")
    if from_stage > 1:
        print(f"  Resuming from stage {from_stage}")
    print()

    # ── Vector store (HNSW-backed ChromaDB) ───────────────────────
    print("[0/9] Initialising vector store…")
    store = _init_vector_store(cfg)
    _sync_store_from_sqlite(db_path, store)
    print(
        f"  Vector store ready — "
        f"CLIP={store.clip_count()}  face={store.face_count()}"
    )
    print()

    t0 = time.time()

    def _stage(n: int, label: str):
        if n < from_stage:
            print(f"[{n}/9] {label} — skipped (resuming from stage {from_stage})")
            return False
        print(f"[{n}/9] {label}…")
        return True

    # ── Stage 1: Scan ─────────────────────────────────────────────
    if _stage(1, "Scanning photos"):
        extensions = set(cfg["ingestion"]["supported_extensions"])
        paths = ingestion.scan_photos(input_dir, extensions)
        print(f"  Found {len(paths)} images")
        if not paths:
            print("  Nothing to process.")
            return
    else:
        from src import ingestion as _ing
        paths = _ing.scan_photos(input_dir, set(cfg["ingestion"]["supported_extensions"]))

    # ── Stage 2: Feature extraction ───────────────────────────────
    if _stage(2, "Extracting features"):
        stage_extract(cfg, paths, db_path, store)

    # ── Stage 3: Aesthetic scoring ────────────────────────────────
    if _stage(3, "Aesthetic scoring"):
        stage_aesthetic(cfg, db_path)

    # ── Stage 4: Scene tagging ────────────────────────────────────
    if _stage(4, "Scene tagging"):
        stage_scene(cfg, db_path)

    # ── Stage 5: Sentiment ────────────────────────────────────────
    if _stage(5, "Smile & eye detection"):
        stage_sentiment(cfg, db_path, paths)

    # ── Stage 6: Deduplication ────────────────────────────────────
    if _stage(6, "Deduplication"):
        n_dups = stage_dedup(cfg, db_path, store)
    else:
        n_dups = 0

    # ── Stage 7: Event clustering ─────────────────────────────────
    if _stage(7, "Event clustering"):
        n_events = stage_cluster(cfg, db_path, store)
    else:
        n_events = 0

    # ── Stage 8: Face identity clustering ────────────────────────
    if _stage(8, "Face identity clustering"):
        stage_face_clustering(cfg, db_path, store)

    # ── Stage 9: Rank + select + output ──────────────────────────
    if _stage(9, "Ranking & selecting"):
        scores, selected = stage_rank_and_select(cfg, db_path, dry_run, total_photos=len(paths))
    else:
        selected = []

    elapsed = time.time() - t0
    print(f"\n{bar}")
    print("  Done!")
    print(f"  Total scanned:    {len(paths)}")
    print(f"  Duplicates found: {n_dups}")
    print(f"  Event clusters:   {n_events}")
    print(f"  Selected:         {len(selected)}")
    print(f"  Elapsed:          {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"{bar}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Photo Curator — local, offline, privacy-preserving"
    )
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--input",       help="Override input folder")
    parser.add_argument("--output",      help="Override output folder")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help="Resume pipeline from stage N (1–9). Cached DB data must exist."
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        input_override=args.input,
        output_override=args.output,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
    )


if __name__ == "__main__":
    main()
