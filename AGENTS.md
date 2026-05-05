# Photo Curator — Codex Session Context

## Project Overview

**Photo Curator** is a local, offline photo intelligence system that processes 10 GB+ photo libraries
and produces a curated ~1 GB selection. Runs entirely on-device. No cloud uploads.
Optimised for Apple Silicon (M1/M2/M3) via MPS acceleration.

This project was originally developed under `/Volumes/SSD/projects/photo_viewer/photo_curator/`
and migrated to this dedicated repo on 2026-04-13. All future development continues here.

---

## Architecture Summary

### 9-Stage Pipeline

```
Stage 1  Ingestion       — scan + EXIF extract
Stage 2  Deduplication   — perceptual hash (pHash) + CLIP cosine similarity
Stage 3  Quality         — blur detection (Laplacian), exposure, noise scoring
Stage 4  Face Detection  — MTCNN face detection + MediaPipe pose
Stage 5  Embeddings      — CLIP ViT-B/32 visual + semantic embeddings
Stage 6  Scene Tagging   — CLIP zero-shot scene/object labels
Stage 7  Sentiment       — CLIP-based sentiment scoring
Stage 8  Clustering      — DBSCAN event clustering + DBSCAN face identity clustering
Stage 9  Selection       — 30/30/40 strategy → copy to output + report
```

### Dynamic Bucket Selection Strategy

- **Subject Buckets** (Configurable in config.yaml, e.g. People, Location, Bike, Landscape) — Allocates budget fraction to best photos matching these subjects, respecting strict diversity caps (e.g. `max_per_person_pct`).
- **Aesthetic Bucket** (Final Catch-All) — Absorbs all remaining unused budget and selects the highest-scoring photos across the entire library to ensure the requested output quota is met.

### Technology Stack

| Layer       | Technology                              |
|-------------|------------------------------------------|
| Runtime     | Python 3.11+, Apple MPS (Metal)          |
| ML models   | CLIP ViT-B/32, MTCNN, FaceNet, MediaPipe |
| Storage     | SQLite (via `src/database.py`)           |
| Config      | `config.yaml`                            |

---

## File & Directory Layout

```
photo-curator/
├── main.py                  # Entry point — orchestrates all 9 stages
├── config.yaml              # All tuneable parameters
├── requirements.txt
├── README.md                # Full HLD/LLD with ASCII diagrams
├── src/
│   ├── __init__.py
│   ├── database.py          # SQLite schema, insert/query helpers
│   ├── quality.py           # Blur, exposure, noise scoring
│   ├── embeddings.py        # CLIP embedding generation (MPS-accelerated)
│   ├── face_detection.py    # MTCNN face detection
│   ├── deduplication.py     # pHash + cosine-similarity dedup
│   ├── privacy.py           # Privacy filter chain (faces, nudity, etc.)
│   ├── clustering.py        # DBSCAN event clustering
│   ├── ranking.py           # Composite score formula
│   ├── selection.py         # 30/30/40 selection logic
│   ├── aesthetic.py         # Aesthetic scoring
│   ├── scene_tagger.py      # CLIP zero-shot scene tagging
│   ├── sentiment.py         # CLIP-based sentiment scoring
│   ├── face_clustering.py   # DBSCAN face identity clustering
│   ├── ingestion.py         # Photo scanning + EXIF extraction
│   └── metadata.py          # Metadata utilities
├── data/
│   ├── input_photos/        # Drop source photos here (gitignored)
│   └── output_photos/       # Curated output written here (gitignored)
├── models/                  # Downloaded model weights (gitignored)
└── cache/                   # SQLite DB + embedding cache (gitignored)
```

---

## Key Design Decisions

1. **Fully offline** — no API calls, no cloud services
2. **MPS acceleration** — CLIP and FaceNet use `torch.device("mps")` on Apple Silicon
3. **SQLite cache** — all embeddings and scores persisted so reruns skip already-processed photos
4. **CLIP zero-shot** — scene tagging and sentiment use zero-shot classification, no fine-tuning needed
5. **pHash + CLIP dedup** — two-pass deduplication: fast perceptual hash first, then semantic similarity
6. **DBSCAN clustering** — chosen over k-means because event count is unknown ahead of time

---

## Running the Pipeline

```bash
# Setup (first time)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Add photos
cp -r /path/to/photos data/input_photos/

# Run
python main.py

# Output
data/output_photos/          # curated photos
data/output_photos/output.json  # curation report
```

---

## Configuration (`config.yaml`)

Key tuneable parameters:

- `pipeline.target_size_gb` — target output size (default 1.0)
- `dedup.phash_threshold` — perceptual hash distance for near-duplicates
- `dedup.clip_threshold` — cosine similarity threshold
- `quality.blur_threshold` — Laplacian variance cutoff
- `clustering.eps` / `clustering.min_samples` — DBSCAN parameters
- `selection.buckets` — Dynamic percentage allocation mapping for subjects (e.g., people, location, aesthetic)
- `selection.max_per_person_pct` — Diversity cap for a single person
- `selection.max_per_day_pct` — Diversity cap for a single shooting day

---

## Session History (migrated from photo_viewer)

- **2026-04-13**: Initial build — all 9 pipeline stages implemented, README.md created with HLD/LLD/ASCII diagrams
- Migrated from `/Volumes/SSD/projects/photo_viewer/photo_curator/` to this dedicated repo
