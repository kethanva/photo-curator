# Photo Curator — Requirements, Decisions & Build History

This document captures the original product requirements that were passed in to
initialise the project, every significant design decision, and all the important
fixes that were made during the initial build. It is the single source of truth
for *why* things are built the way they are.

---

## TABLE OF CONTENTS

1. [Original Product Brief](#1-original-product-brief)
2. [Functional Requirements](#2-functional-requirements)
3. [Non-Functional Requirements](#3-non-functional-requirements)
4. [Constraints & Platform Targets](#4-constraints--platform-targets)
5. [Pipeline Architecture Requirements](#5-pipeline-architecture-requirements)
6. [Module-Level Requirements](#6-module-level-requirements)
7. [Configuration Requirements](#7-configuration-requirements)
8. [Output Requirements](#8-output-requirements)
9. [Key Design Decisions (and Why)](#9-key-design-decisions-and-why)
10. [Important Fixes Applied During Build](#10-important-fixes-applied-during-build)
11. [What Was Added Beyond the Initial Brief](#11-what-was-added-beyond-the-initial-brief)
12. [Known Gaps / Future Work](#12-known-gaps--future-work)

---

## 1. Original Product Brief

The project was initialised from the following requirement:

> **"Intelligent Local Photo Curation System (Mac M1)"**
>
> Design and implement a local, offline photo intelligence system capable of:
>
> - Processing a large personal photo library (10 GB+, roughly 3,000–5,000 high-res photos)
> - Running entirely on-device — NO cloud uploads, NO external APIs
> - Producing a curated ~1 GB selection of the best photos
> - Optimised for Apple Silicon (M1/M2/M3) using MPS acceleration
>
> The system must be able to:
> - Detect and remove near-duplicate and exact-duplicate photos
> - Assess technical quality (sharpness, exposure, resolution)
> - Detect faces and identify frequent people across photos
> - Cluster photos into events/moments
> - Score photos aesthetically
> - Tag scenes/objects automatically
> - Detect smile and eyes-open sentiment in portraits
> - Filter out private content (screenshots, documents, ID cards, home solo shots)
> - Select a diverse, balanced final set using a 30/30/40 people/location/aesthetic strategy
> - Resize output photos to fit within the 1 GB target
> - Generate a JSON curation report

A secondary requirement was passed in from an additional LLM response regarding
**pipeline architecture for 10 GB+ libraries**, which added the following:

> Creating a project of this scale requires a robust "Pipeline" architecture:
>
> - Each stage must be independently resumable (no full reprocessing on reruns)
> - An SQLite cache must persist all extracted features between runs
> - File hash + content hash checks must be used to skip already-processed photos
> - Stages must be individually skippable via a `--from-stage N` flag
> - The pipeline must process photos incrementally as new ones are added
> - Memory usage must be controlled — load one image at a time, not the whole library
> - Models must be loaded once and reused across all photos (singleton pattern)

---

## 2. Functional Requirements

### FR-01: Photo Ingestion
- Recursively scan a folder for photos
- Support: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tiff`, `.bmp`, `.webp`
- HEIC/HEIF support is mandatory (iPhone photos are the primary use case)
- Extract EXIF metadata: timestamp, GPS coordinates, camera model
- Compute SHA-256 file hash for change detection
- Resize to a working resolution (1024 px max dimension) before feature extraction

### FR-02: Quality Assessment
- Compute **sharpness** using Laplacian variance (no scipy dependency)
- Compute **exposure** as mean pixel brightness, normalised to 0–1
- Measure **resolution** as shorter dimension in pixels
- Flag as quality-fail if any threshold is not met
- Configurable thresholds: min blur, min/max exposure, min resolution

### FR-03: Duplicate Detection
- **Pass 1 — pHash**: perceptual hash Hamming distance ≤ threshold → duplicate
- **Pass 2 — CLIP cosine similarity**: if cosine similarity ≥ threshold → duplicate
- **Exact match**: SHA-256 file hash equality → duplicate
- Keep the sharpest (highest blur_score) copy; mark all others as duplicate
- Both thresholds must be configurable

### FR-04: CLIP Embeddings
- Use OpenAI CLIP ViT-B/32
- Produce normalised 512-dim float32 embeddings per photo
- Support batch extraction for efficiency
- Persist embeddings as binary blobs in SQLite
- Automatically select device: MPS → CUDA → CPU

### FR-05: Face Detection & Embedding
- Detect faces using MTCNN from facenet-pytorch
- Run face detector on CPU (MPS has tensor-type issues with MTCNN kernels)
- Extract 512-dim face embeddings using InceptionResnetV1 (pretrained on VGGFace2)
- Store face count and averaged face embedding per photo

### FR-06: Aesthetic Scoring
- **Default (offline)**: CLIP zero-shot using curated positive/negative prompt pairs
- **Optional (LAION)**: Download improved-aesthetic-predictor MLP from HuggingFace
  - The LAION model is more accurate but requires internet on first run
  - Toggle via `config.yaml: aesthetic.use_laion_predictor: true`
- Score from stored CLIP embeddings — no second model pass needed

### FR-07: Scene Tagging
- CLIP zero-shot classification against a vocabulary of scene/object labels
- Store top-N tags per photo as a JSON string in the database
- Works from stored CLIP embeddings — no extra inference

### FR-08: Sentiment Scoring (Portrait Quality)
- Detect smile and eyes-open state using MediaPipe Face Mesh
- Score portraits 0–1 (higher = smiling with eyes open)
- Only run on photos that have ≥ min_face_for_sentiment faces
- Default to 0.5 for photos with no faces

### FR-09: Privacy Filtering
- **Screenshot detection**: check pixel dimensions against known screen resolutions;
  also flag if no camera EXIF and no GPS
- **Document/ID detection**: CLIP zero-shot against private content prompts
  (documents, bank cards, receipts, passports, handwritten notes)
- **Home private filter** (optional): filter solo shots taken within radius_km of home
  GPS coordinates — disabled by default
- Checks ordered cheapest-first to avoid unnecessary CLIP calls

### FR-10: Event Clustering
- Cluster photos into events using DBSCAN
- Feature space: timestamp + GPS lat/lon + CLIP embedding (PCA-reduced 512→32)
- StandardScaler normalisation per feature group
- Configurable weights: time_weight, gps_weight, visual_weight
- Configurable DBSCAN: eps, min_samples
- DBSCAN chosen over k-means because the number of events is unknown in advance
- Noise points (singletons) get cluster_id = -1

### FR-11: Face Identity Clustering
- Cluster face embeddings across photos using DBSCAN
- Assign person_id to each photo with detected faces
- Flag persons who appear in ≥ frequent_person_threshold photos as "frequent"
- Configurable: eps, min_samples, frequent_person_threshold

### FR-12: Ranking
- Composite score = weighted sum of 7 normalised components
- Components: sharpness, aesthetic, face_score, sentiment, uniqueness,
  metadata_importance, diversity_bonus
- All components min-max normalised to 0–1 before weighting
- Weights fully configurable in config.yaml

### FR-13: Photo Selection (30/30/40)
- Three-bucket diversity strategy:
  - **People bucket (30%)**: best N photos per identified frequent person
  - **Location bucket (30%)**: best M photos per GPS/event cluster
  - **Aesthetic bucket (40%)**: highest overall scores regardless of subject
- Pre-filter: only quality_pass=1, is_private=0, is_duplicate=0 candidates
- Estimated output sizes after resizing used for budget tracking
- Configurable per-bucket caps: max_per_person, max_per_location, max_per_cluster
- Stop when total estimated output reaches max_output_bytes (default 1 GB)

### FR-14: Output
- Resize selected photos: longest side capped at output_long_side (default 2560 px)
- Save as JPEG with configurable quality (default 92)
- Resolve filename collisions automatically
- Generate `output.json` report with per-photo metadata:
  filename, original path, score, aesthetic, sentiment, faces, person_id,
  is_frequent_person, scene_tags, cluster, lat/lon, timestamp, camera, blur

### FR-15: CLI Interface
```
python main.py                           # uses config.yaml defaults
python main.py --input /photos           # override input folder
python main.py --output /best            # override output folder
python main.py --dry-run                 # score only, no file writes
python main.py --from-stage 6           # resume from stage N (1–9)
```

---

## 3. Non-Functional Requirements

### NFR-01: Fully Offline
- Zero network calls during normal operation
- No cloud APIs, no telemetry
- CLIP and FaceNet weights cached locally after first download

### NFR-02: Apple Silicon Optimised
- MPS (Metal Performance Shaders) used for CLIP and FaceNet
- MTCNN intentionally run on CPU (see Fix #3 below)
- Performance target: process a 10 GB library in under 2 hours on M1

### NFR-03: Resumable / Incremental
- SQLite cache persists all embeddings and scores between runs
- File hash check skips unchanged photos on reruns
- `--from-stage N` allows resuming a crashed run without re-extracting features

### NFR-04: Memory Efficient
- Load and process one image at a time through the feature extraction stage
- Models loaded as module-level singletons — never reloaded
- SQLite WAL mode + NORMAL sync for safe concurrent writes

### NFR-05: Configurable Without Code Changes
- All thresholds, weights, paths, and toggles live in `config.yaml`
- No hardcoded values in source code

### NFR-06: No scipy Dependency
- Quality module implements Laplacian variance via pure numpy array slicing
- Avoids the heavy scipy dependency for a simple convolution

### NFR-07: Graceful Degradation
- Missing EXIF returns zero/empty defaults — never crashes
- Failed CLIP extraction returns zero vector — contributes neutral score
- Failed face detection returns (0, None) — photo still processed
- Failed aesthetic download falls back to CLIP zero-shot

---

## 4. Constraints & Platform Targets

| Constraint              | Value                                         |
|-------------------------|-----------------------------------------------|
| Input library size      | 10 GB+ (3,000–5,000 high-res photos)          |
| Target output size      | ~1 GB                                         |
| Platform                | macOS (Apple Silicon M1/M2/M3 primary)        |
| Python version          | 3.11+                                         |
| Must be fully offline   | Yes — no cloud calls during processing        |
| CLIP model              | ViT-B/32 (balance of speed and accuracy)      |
| Face model              | MTCNN + InceptionResnetV1 (VGGFace2)          |
| Storage format          | SQLite (single-file, zero-server)             |
| Output image format     | JPEG (smaller, universal)                     |
| Output resolution       | 2560 px long side (2K display quality)        |

---

## 5. Pipeline Architecture Requirements

The pipeline is a linear sequence of 9 stages. Each stage reads from and writes
to the SQLite cache. Stages are idempotent — safe to rerun.

```
Stage 1  Scan              src/ingestion.py
Stage 2  Feature Extract   src/embeddings.py, src/quality.py,
                           src/face_detection.py, src/deduplication.py,
                           src/privacy.py, src/metadata.py
Stage 3  Aesthetic Score   src/aesthetic.py
Stage 4  Scene Tagging     src/scene_tagger.py
Stage 5  Sentiment         src/sentiment.py
Stage 6  Deduplication     src/deduplication.py
Stage 7  Event Clustering  src/clustering.py
Stage 8  Face Identity     src/face_clustering.py
Stage 9  Rank & Select     src/ranking.py, src/selection.py
```

**Resumability contract**: if `--from-stage N` is passed, stages 1..(N-1) are
printed as skipped, and stage 1 (scan) is always re-run silently to rebuild the
path list needed by later stages.

---

## 6. Module-Level Requirements

### `src/database.py`
- Single connection factory with WAL mode
- `init_db()` creates schema on first run; also migrates existing DBs via
  `ALTER TABLE … ADD COLUMN IF NOT EXISTS` idiom
- Upsert on `path` primary key — safe to rerun on the same photo
- Binary blobs (CLIP, face embeddings) stored as raw float32 bytes via numpy

### `src/ingestion.py`
- Recursive glob with extension filter
- Safe image load: resize to max_dimension, convert to RGB, catch corrupt files
- SHA-256 file hash for change detection

### `src/metadata.py`
- Extract timestamp, GPS, camera model from EXIF via piexif
- Return safe defaults (0, '', False) for missing fields
- Parse EXIF GPS rational fractions to decimal degrees

### `src/quality.py`
- Pure numpy Laplacian (no scipy)
- Returns `QualityResult` dataclass with individual metrics and overall pass bool

### `src/embeddings.py`
- Module-level singleton: model loaded once, reused for all photos
- Device selection: MPS → CUDA → CPU
- Returns zero vector (not None) on extraction failure

### `src/face_detection.py`
- MTCNN on CPU; FaceNet on MPS/CUDA/CPU
- Returns averaged embedding across all detected faces (not per-face)
- face_score() converts count to 0–1 using log scale

### `src/deduplication.py`
- Three-tier check: file hash → pHash Hamming → CLIP cosine
- Sort by blur_score descending so the sharpest copy is always kept
- Returns a set of paths to mark as duplicates

### `src/privacy.py`
- Cheapest check first: screenshot heuristic (no ML)
- Then home-private GPS check (arithmetic only)
- Then CLIP zero-shot document detection (most expensive — last)
- Returns bool: True = exclude this photo

### `src/aesthetic.py`
- Two strategies: CLIP zero-shot (default) or LAION MLP (optional)
- LAION model downloaded from HuggingFace on first use; cached in `models/`
- Falls back to CLIP zero-shot if LAION download fails

### `src/scene_tagger.py`
- CLIP zero-shot classification from stored embeddings
- Tags stored as JSON string in database

### `src/sentiment.py`
- MediaPipe Face Mesh for smile + eyes-open detection
- Returns normalised score 0–1

### `src/clustering.py`
- DBSCAN on combined feature space: time + GPS + PCA-reduced CLIP
- PCA reduces 512 CLIP dims to 32 before combining with scalar features
- StandardScaler applied per feature group
- Per-feature weights applied after scaling

### `src/face_clustering.py`
- DBSCAN on 512-dim face embeddings
- Labels photos with person_id and is_frequent flag

### `src/ranking.py`
- 7-component weighted sum
- All components min-max normalised independently before weighting
- Returns {path: float} score map

### `src/selection.py`
- Three-bucket strategy with per-bucket byte budgets
- Estimated output sizes account for resizing (not original file sizes)
- Deduplicates across buckets — a photo can only be selected once
- `copy_to_output()` handles resize, JPEG encoding, collision resolution, JSON report

---

## 7. Configuration Requirements

All configuration lives in `config.yaml`. Required top-level sections:

| Section         | Purpose                                         |
|-----------------|-------------------------------------------------|
| `paths`         | input, output, cache, models directories        |
| `ingestion`     | extensions, max_dimension, batch_size           |
| `quality`       | blur, exposure, resolution thresholds           |
| `deduplication` | phash_threshold, embedding_similarity           |
| `privacy`       | filter flags, home_coords, home_radius_km       |
| `clustering`    | DBSCAN params, feature weights                  |
| `aesthetic`     | use_laion_predictor toggle                      |
| `scene_tagging` | enabled, top_n                                  |
| `sentiment`     | enabled, min_face_for_sentiment                 |
| `face_clustering`| enabled, DBSCAN params, frequent threshold     |
| `ranking`       | weights dict (7 keys)                           |
| `selection`     | max_output_bytes, bucket fractions, caps, resize|
| `output`        | copy_files, generate_report, report_filename    |

---

## 8. Output Requirements

```
data/output_photos/
├── IMG_1234.jpg           # resized, quality-92 JPEG
├── IMG_5678.jpg
├── ...
└── output.json            # curation report
```

`output.json` format — array of objects, one per selected photo:
```json
{
  "filename":             "IMG_1234.jpg",
  "original_path":        "/path/to/source/IMG_1234.JPG",
  "score":                0.7821,
  "aesthetic":            0.683,
  "sentiment":            0.812,
  "faces":                2,
  "person_id":            3,
  "is_frequent_person":   true,
  "scene_tags":           "[\"beach\", \"sunset\", \"landscape\"]",
  "cluster":              7,
  "lat":                  51.5074,
  "lon":                  -0.1278,
  "timestamp":            1706745600.0,
  "camera":               "Apple iPhone 15 Pro",
  "blur":                 312.45
}
```

---

## 9. Key Design Decisions (and Why)

### Decision 1: SQLite over a vector database
**Why:** The project must be fully offline and self-contained. Chroma, Qdrant, and
Faiss all add significant deployment complexity. SQLite can store float32 embedding
blobs directly. For 5,000 photos (5,000 × 512 × 4 bytes = ~10 MB of embeddings),
SQLite is fast enough. A vector DB would be the right call at 100,000+ photos.

### Decision 2: DBSCAN over k-means for event clustering
**Why:** k-means requires knowing the number of clusters (events) ahead of time,
which is impossible for a personal photo library. DBSCAN discovers clusters
automatically and handles noise points (singletons) gracefully.

### Decision 3: MTCNN on CPU, FaceNet on MPS
**Why:** MTCNN uses custom detection kernels that produce tensor type mismatches
on the MPS backend (a known facenet-pytorch issue on Apple Silicon). Running MTCNN
on CPU avoids the crash while still giving MPS-accelerated embedding extraction
via FaceNet.

### Decision 4: pHash → CLIP two-pass deduplication
**Why:** pHash is extremely fast (microseconds per pair) but misses near-duplicates
that differ in JPEG compression or minor colour edits. CLIP catches those but is
slower. Running pHash first eliminates the obvious cases; CLIP only runs on
survivors, keeping the overall dedup stage fast.

### Decision 5: CLIP zero-shot for aesthetics (default)
**Why:** The LAION improved-aesthetic-predictor is more accurate but requires
downloading weights from HuggingFace on the first run — which breaks the
"fully offline" requirement for cold starts. CLIP zero-shot against curated
positive/negative prompts requires no extra downloads and produces consistent
results. LAION is available as an opt-in for users who want the accuracy.

### Decision 6: PCA 512→32 before DBSCAN clustering
**Why:** DBSCAN with cosine distance on raw 512-dim CLIP vectors suffers from
the curse of dimensionality — all pairwise distances collapse toward the same
value. Reducing to 32 principal components keeps the meaningful variance while
making the distance metric discriminative again.

### Decision 7: Resize output to 2560 px long side
**Why:** A 12 MP iPhone photo is typically 4–6 MB as a JPEG. Resized to 2560 px,
it becomes ~1–1.5 MB. This means a 1 GB budget holds ~700–1000 photos at 2K
quality (sharp on any current display) vs. ~200 full-resolution originals.
The resize step dramatically increases the diversity of the final selection.

### Decision 8: Module-level model singletons
**Why:** CLIP ViT-B/32 and FaceNet both take 2–5 seconds to load. Loading them
per-photo would make the pipeline 10–25× slower. The singleton pattern (global
`_model = None`, lazy init on first call) loads each model exactly once per run
regardless of how many photos are processed.

### Decision 9: Three-bucket 30/30/40 selection
**Why:** Pure score-maximisation produces a final set dominated by whoever is most
photogenic or by a single beautiful location. The three-bucket strategy enforces
structural diversity:
- People bucket ensures key relationships are represented
- Location bucket ensures geographic/event variety
- Aesthetic bucket picks the overall best remaining photos

### Decision 10: WAL mode for SQLite
**Why:** Write-Ahead Logging allows reads to proceed concurrently with writes.
This is important for potential future multi-threaded processing and prevents
database locking errors if the pipeline is interrupted mid-write.

---

## 10. Important Fixes Applied During Build

### Fix 1: `quality.py` — removed scipy dependency
**Problem:** Initial implementation used `scipy.ndimage.laplace()` for blur
detection. Scipy is a large dependency (~80 MB) that adds significant install time.

**Fix:** Replaced with pure numpy 4-neighbour finite difference Laplacian:
```python
lap = (gray[1:-1, :-2] + gray[1:-1, 2:] +
       gray[:-2, 1:-1] + gray[2:, 1:-1] -
       4.0 * gray[1:-1, 1:-1])
return float(lap.var())
```

### Fix 2: `quality.py` — parameter name mismatch
**Problem:** `quality.assess()` was called in `main.py` with keyword arguments
`min_blur_score`, `min_exposure_score`, `max_exposure_score` from config, but the
function signature used `min_blur`, `min_exposure`, `max_exposure`.

**Fix:** Updated `main.py` to unpack only the config keys that match the function
signature:
```python
q = quality.assess(img, **{k: q_cfg[k] for k in (
    "min_blur_score", "min_exposure_score",
    "max_exposure_score", "min_resolution")})
```
And aligned the function parameters to match config key names.

### Fix 3: `face_detection.py` — MTCNN MPS crash
**Problem:** Running MTCNN on `torch.device("mps")` raised a RuntimeError about
tensor dtype incompatibility in the detection kernel on Apple Silicon.

**Fix:** MTCNN is explicitly forced to CPU regardless of available hardware:
```python
_mtcnn = MTCNN(image_size=160, keep_all=True, device="cpu", ...)
```
FaceNet (the embedding model) still uses MPS for speed.

### Fix 4: `database.py` — schema migration for new columns
**Problem:** The initial schema was missing the `aesthetic_score`, `scene_tags`,
`smile_score`, `person_id`, and `is_frequent` columns which were added in a later
iteration. Existing SQLite databases would crash on INSERT.

**Fix:** Added `_add_column_if_missing()` called from `init_db()`:
```python
def _add_column_if_missing(conn, table, column, definition):
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError:
        pass  # Column already present
```

### Fix 5: `face_detection.py` — single face tensor shape
**Problem:** MTCNN returns a 3D tensor `(3, 160, 160)` for a single detected face
but a 4D tensor `(N, 3, 160, 160)` for multiple faces. Passing a 3D tensor to
FaceNet caused a dimension error.

**Fix:**
```python
if faces.dim() == 3:
    faces = faces.unsqueeze(0)  # Single face → batch of 1
```

### Fix 6: `embeddings.py` — MPS float precision
**Problem:** MPS requires float32 tensors. CLIP preprocess produces float32, but
operations on the tensor were silently promoting to float64 in some torch versions,
causing MPS device errors.

**Fix:** Explicitly cast and normalise on device:
```python
emb = emb / emb.norm(dim=-1, keepdim=True)
return emb.cpu().numpy()[0].astype(np.float32)
```

### Fix 7: `clustering.py` — PCA min component guard
**Problem:** PCA with `n_components=32` fails when the number of samples is ≤ 32.
Small test runs with fewer than 33 photos would crash.

**Fix:**
```python
pca = PCA(n_components=min(_PCA_DIM, n - 1), random_state=42)
```

### Fix 8: `selection.py` — budget estimation uses resized sizes
**Problem:** The selection loop was tracking budget using original file sizes.
This caused the algorithm to stop early, selecting far fewer photos than the
1 GB budget could actually hold after resizing.

**Fix:** `_est_size()` estimates the post-resize JPEG size before adding to the
budget:
```python
scale = output_long_side / max_dim
return max(50_000, int(orig * scale * scale * 0.75))
```

### Fix 9: `main.py` — scan always runs when resuming
**Problem:** When `--from-stage N` was passed (N > 1), stage 1 (scan) was skipped,
leaving `paths` undefined. Later stages that need the path list (e.g. sentiment
which iterates photo files) would crash with `NameError: name 'paths' is not defined`.

**Fix:** Stage 1 is always executed silently (without printing "skipped") when
resuming, to rebuild the `paths` list:
```python
else:
    from src import ingestion as _ing
    paths = _ing.scan_photos(input_dir, set(cfg["ingestion"]["supported_extensions"]))
```

### Fix 10: `privacy.py` — CLIP document check ordering
**Problem:** CLIP inference was being called even for photos that had already been
flagged as screenshots by the fast heuristic, wasting time.

**Fix:** Checks are strictly ordered cheapest-first with early return:
```python
if filter_screenshots and is_screenshot(...):  return True
if filter_home_private and is_home_private(...): return True
if filter_documents and is_document_clip(...):  return True
return False
```

### Fix 11: `aesthetic.py` — LAION fallback on offline systems
**Problem:** If `use_laion_predictor: true` was set and the HuggingFace download
failed (offline machine, network issue), the function would raise an exception
and no aesthetic score would be produced.

**Fix:** Wrapped download in try/except with explicit fallback to CLIP zero-shot:
```python
except Exception as e:
    print(f"  Warning: could not download aesthetic model ({e}). Falling back to CLIP.")
    return None
```
The `score_from_embedding()` function falls through to the CLIP path when
`_load_laion_predictor()` returns None.

### Fix 12: `README.md` creation
**Problem:** The README.md file was originally created at the wrong location
(`/Volumes/SSD/projects/photo_viewer/photo_curator/README.md`) and was not
present at `/Volumes/SSD/projects/photo-curator/README.md` after the project
was forked to its dedicated repository.

**Fix:** README.md was recreated directly in `/Volumes/SSD/projects/photo-curator/`
with the full HLD, LLD, pipeline flow, ASCII diagrams, configuration reference,
module dependency graph, and usage guide.

---

## 11. What Was Added Beyond the Initial Brief

The following modules and capabilities were added during the build in response to
the secondary LLM architecture review:

| Addition                     | Reason                                                        |
|------------------------------|---------------------------------------------------------------|
| `src/ingestion.py`           | Separated scan + load from feature extraction for clarity     |
| `src/metadata.py`            | Isolated EXIF parsing (piexif API is verbose, isolation helps)|
| `src/scene_tagger.py`        | Added zero-shot scene labels for richer report output         |
| `src/sentiment.py`           | MediaPipe smile/eyes scoring for portrait quality             |
| `src/aesthetic.py`           | Separated aesthetic scoring from embedding extraction         |
| `src/face_clustering.py`     | Separated person identity clustering from event clustering    |
| LAION aesthetic predictor    | Optional higher-accuracy aesthetic scoring                    |
| `--dry-run` CLI flag         | Score-only mode for testing without writing files             |
| `--from-stage N` CLI flag    | Resume a crashed pipeline without re-running all stages       |
| Output collision resolution  | Auto-rename `stem__parent_N.jpg` on filename conflict         |
| SQLite migration helpers     | Safe `ALTER TABLE ADD COLUMN` for schema evolution            |
| `fix_readme.py`              | One-off script to regenerate the README at the correct path   |

---

## 12. Known Gaps / Future Work

| Gap                          | Notes                                                         |
|------------------------------|---------------------------------------------------------------|
| No video support             | Only still images; `.mov`/`.mp4` thumbnails not extracted     |
| No RAW format support        | `.cr2`, `.nef`, `.arw` not in the extension list              |
| MTCNN CPU-only               | Could be replaced with a MPS-compatible detector later        |
| No parallel stage execution  | All stages run sequentially; could parallelise extract stage  |
| Scene vocabulary is hardcoded| `scene_tagger.py` uses a fixed label list in the module       |
| No interactive UI            | CLI only; a web UI (e.g. Streamlit) was not in scope          |
| LAION download first-run     | Breaks pure offline requirement on first use if enabled       |
| No test suite                | Unit tests for each module not written during initial build   |
| GPS clustering independent   | Event clustering uses DBSCAN but does not do pure GPS grouping|
