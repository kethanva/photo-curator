# Photo Curator

A local, offline photo intelligence system that processes 10 GB+ photo libraries and produces a curated ~1 GB selection. Runs entirely on-device with no cloud uploads. Optimised for Apple Silicon (M1/M2/M3) via MPS acceleration.

## What it does

```
10 GB of photos  →  9-stage pipeline  →  ~1 GB curated output
```

| Stage | What happens |
|-------|-------------|
| 1. Scan | Recursive folder scan; skips files already in cache |
| 2. Extract | EXIF metadata, quality check, CLIP embeddings, face detection, privacy filter |
| 3. Aesthetic | CLIP zero-shot aesthetic scoring (no extra downloads) |
| 4. Scene tags | Zero-shot classification: beach, wedding, mountain, etc. |
| 5. Sentiment | MediaPipe smile + eye-openness detection |
| 6. Dedup | pHash Hamming distance + CLIP cosine similarity |
| 7. Cluster | DBSCAN event grouping via time + GPS + visual features |
| 8. Faces | Cross-photo person identity clustering (family vs strangers) |
| 9. Rank & select | 30/30/40 diversity strategy + resize to 2 560 px JPEG |

**30/30/40 selection strategy:**
- **30%** — best shots of each identified frequent person (family/friends)
- **30%** — best shots per location cluster (travel diversity)
- **40%** — highest overall aesthetic scores regardless of subject

## Quick start

```bash
# 1. Clone and enter
git clone <repo> photo-curator
cd photo-curator

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# 4. Add your photos
cp -r /path/to/your/photos data/input_photos/

# 5. Run
python main.py
```

Output lands in `data/output_photos/` with an `output.json` report.

## Usage

```bash
# Basic run (uses config.yaml)
python main.py

# Custom paths
python main.py --input /Volumes/Photos --output ~/Desktop/Best

# Dry run — score and select but don't write files
python main.py --dry-run

# Resume from a specific stage (skips expensive earlier stages)
python main.py --from-stage 6    # re-dedup and everything after
python main.py --from-stage 9    # just re-rank and re-select
```

## Configuration

All tunable parameters are in `config.yaml`. Key settings:

```yaml
quality:
  min_blur_score: 50.0        # raise to be stricter about sharpness

selection:
  resize_output: true
  output_long_side: 2560      # or 3840 for 4K output
  people_budget_fraction: 0.30
  location_budget_fraction: 0.30
  aesthetic_budget_fraction: 0.40

privacy:
  filter_screenshots: true
  filter_documents: true
  home_coords: [48.8566, 2.3522]   # set your home lat/lon
  filter_home_private: true        # filter solo shots at home

face_clustering:
  frequent_person_threshold: 5    # min photos to count as "frequent person"
```

## Project structure

```
photo-curator/
├── main.py              # Pipeline orchestrator (run this)
├── config.yaml          # All tunable parameters
├── requirements.txt
├── data/
│   ├── input_photos/    # Put your photos here
│   └── output_photos/   # Curated output lands here
├── cache/               # SQLite feature cache (auto-created)
├── models/              # Downloaded model weights (auto-created)
└── src/
    ├── ingestion.py     # Scan, hash, load images
    ├── metadata.py      # EXIF extraction
    ├── quality.py       # Blur, exposure, resolution
    ├── embeddings.py    # CLIP ViT-B/32 (MPS-aware)
    ├── face_detection.py# MTCNN + FaceNet
    ├── face_clustering.py# Cross-photo person identity
    ├── aesthetic.py     # CLIP aesthetic scoring
    ├── scene_tagger.py  # Zero-shot scene labels
    ├── sentiment.py     # MediaPipe smile + eye detection
    ├── deduplication.py # pHash + cosine similarity
    ├── clustering.py    # DBSCAN event clustering
    ├── privacy.py       # Screenshot + document filter
    ├── ranking.py       # Weighted composite scorer
    ├── selection.py     # 30/30/40 diversity + resize
    └── database.py      # SQLite cache layer
```

## Performance (Apple M1)

| Library size | Estimated time |
|---|---|
| 1 000 photos | ~8–15 min |
| 5 000 photos | ~40–80 min |
| 10 000 photos | ~1.5–3 hours |

Re-runs are fast — only new or changed photos are processed. Use `--from-stage 9` to re-tune selection in seconds.

## Requirements

- Python 3.10+
- macOS with Apple Silicon (MPS) or any machine with CUDA/CPU
- ~4 GB RAM minimum; 8 GB recommended for large libraries

See `requirements.txt` for Python packages. MediaPipe is required for smile detection; it can be disabled in `config.yaml` with `sentiment.enabled: false`.
