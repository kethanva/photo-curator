# Photo Curator

A local, offline photo intelligence system that processes 10 GB+ photo libraries
and produces a curated ~1 GB selection. Runs entirely on-device. No cloud uploads.
Optimised for Apple Silicon (M1/M2/M3) via MPS acceleration.

===============================================================================

## TABLE OF CONTENTS

  1. Quick Start
  2. High Level Design (HLD)
     2.1  System Context
     2.2  Component Map
     2.3  Technology Stack
     2.4  Design Principles
  3. Pipeline Flow
     3.1  End-to-End Pipeline (9 stages)
     3.2  Single-Photo Data Flow
     3.3  Selection Strategy (30/30/40)
  4. Low Level Design (LLD)
     4.1  Module Reference
     4.2  Database Schema
     4.3  Ranking Formula
     4.4  Deduplication Logic
     4.5  Clustering Algorithm
     4.6  Face Identity Clustering
     4.7  Privacy Filter Chain
     4.8  Output Resizing
  5. Module Dependency Graph
  6. Configuration Reference
  7. File & Directory Layout
  8. Performance & Scaling
  9. Usage


===============================================================================
## 1. QUICK START
===============================================================================

    git clone <repo> photo-curator && cd photo-curator
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git

    # drop your photos in
    cp -r /path/to/photos data/input_photos/

    # run the full pipeline
    python main.py

    # output + report
    data/output_photos/
    data/output_photos/output.json


===============================================================================
## 2. HIGH LEVEL DESIGN (HLD)
===============================================================================

-------------------------------------------------------------------------------
2.1  SYSTEM CONTEXT
-------------------------------------------------------------------------------

                          +---------------------------+
                          |        User / Shell       |
                          +-------------|-------------+
                                        |
                                  python main.py
                                        |
                          +-------------|-------------+
                          |                           |
                          |      PHOTO CURATOR        |
                          |    (Local Process)        |
                          |                           |
                          |  +---------------------+  |
                          |  |   9-Stage Pipeline  |  |
                          |  +---------------------+  |
                          |           |               |
                          |  +--------v-----------+   |
                          |  |   SQLite Cache DB  |   |
                          |  +--------------------+   |
                          |           |               |
                          |  +--------v-----------+   |
                          |  |  ML Models (local) |   |
                          |  |  - CLIP ViT-B/32   |   |
                          |  |  - MTCNN + FaceNet |   |
                          |  |  - MediaPipe       |   |
                          |  +--------------------+   |
                          |                           |
                          +-----|---------------------|+
                                |                     |
               +----------------+        +-----------+----------+
               |                         |                      |
    +----------v---------+    +----------v--------+  +----------v--------+
    |   Input Photos     |    |  Output Photos    |  |  output.json      |
    |  data/input_photos |    | data/output_photos|  |  (curation report)|
    |  (10 GB+, any fmt) |    |  (~1 GB, JPEG)    |  |                   |
    +--------------------+    +-------------------+  +-------------------+

  Key constraints:
    - Fully offline: no API calls, no cloud storage
    - Incremental: SQLite cache skips already-processed files
    - Resumable: --from-stage N replays from any checkpoint
    - Privacy-first: screenshots, documents, and home shots filtered locally


-------------------------------------------------------------------------------
2.2  COMPONENT MAP
-------------------------------------------------------------------------------

  +===========================================================================+
  |                         PHOTO CURATOR SYSTEM                             |
  +===========================================================================+
  |                                                                           |
  |  +-------------------+    +------------------+    +--------------------+ |
  |  |  INGESTION LAYER  |    |   ML LAYER       |    |  ANALYSIS LAYER   | |
  |  |-------------------|    |------------------|    |--------------------| |
  |  | ingestion.py      |    | embeddings.py    |    | quality.py        | |
  |  |  - folder scan    |    |  - CLIP ViT-B/32 |    |  - blur           | |
  |  |  - file hashing   |    |  - MPS/CUDA/CPU  |    |  - exposure       | |
  |  |  - HEIC support   |    |                  |    |  - resolution     | |
  |  |  - image resize   |    | face_detection.py|    |                   | |
  |  |                   |    |  - MTCNN detect  |    | aesthetic.py      | |
  |  | metadata.py       |    |  - FaceNet embed |    |  - CLIP prompts   | |
  |  |  - EXIF parse     |    |                  |    |  - LAION MLP opt. | |
  |  |  - GPS decode     |    | sentiment.py     |    |                   | |
  |  |  - timestamp      |    |  - MediaPipe     |    | scene_tagger.py   | |
  |  |  - camera model   |    |  - smile detect  |    |  - 28 labels      | |
  |  +-------------------+    |  - eye openness  |    |  - zero-shot CLIP | |
  |                           +------------------+    +--------------------+ |
  |                                                                           |
  |  +-------------------+    +------------------+    +--------------------+ |
  |  |  DEDUP / FILTER   |    |  CLUSTERING      |    |  SELECTION LAYER  | |
  |  |-------------------|    |------------------|    |--------------------| |
  |  | deduplication.py  |    | clustering.py    |    | ranking.py        | |
  |  |  - pHash          |    |  - DBSCAN        |    |  - 7 components   | |
  |  |  - cosine sim     |    |  - PCA reduce    |    |  - weighted score | |
  |  |  - keep sharpest  |    |  - time+GPS+CLIP |    |                   | |
  |  |                   |    |                  |    | selection.py      | |
  |  | privacy.py        |    | face_clustering.py    |  - 30/30/40 split | |
  |  |  - screenshot     |    |  - DBSCAN faces  |    |  - per-person cap | |
  |  |  - document CLIP  |    |  - person_id     |    |  - resize output  | |
  |  |  - home heuristic |    |  - frequent flag |    |  - JPEG write     | |
  |  +-------------------+    +------------------+    +--------------------+ |
  |                                                                           |
  |  +=====================================================================+ |
  |  |                      PERSISTENCE LAYER                             | |
  |  |---------------------------------------------------------------------| |
  |  |  database.py  ->  cache/photo_db.sqlite                            | |
  |  |  SQLite WAL mode, one row per photo, all features stored as BLOBs  | |
  |  +=====================================================================+ |
  +===========================================================================+


-------------------------------------------------------------------------------
2.3  TECHNOLOGY STACK
-------------------------------------------------------------------------------

  Category          Library / Tool          Purpose
  ----------------  ----------------------  ---------------------------------
  ML Embeddings     CLIP ViT-B/32 (OpenAI)  512-dim visual embeddings
  Face Detection    MTCNN (facenet-pytorch)  Bounding box + alignment
  Face Embedding    InceptionResnetV1        512-dim identity vector
  Sentiment         MediaPipe Face Mesh      468 facial landmarks
  Clustering        scikit-learn DBSCAN      Event + identity grouping
  Dim Reduction     scikit-learn PCA         512 -> 32 for CLIP clustering
  Image I/O         Pillow + pillow-heif     JPEG, PNG, HEIC, TIFF, WebP
  EXIF              piexif                   Timestamp, GPS, camera model
  Perceptual Hash   ImageHash (pHash)        Near-duplicate detection
  Accelerator       PyTorch MPS/CUDA/CPU     M1/M2/M3 or NVIDIA or fallback
  Cache             SQLite 3 (WAL mode)      Feature store, incremental runs
  Config            PyYAML                   config.yaml parsing
  Progress          tqdm                     Pipeline progress bars


-------------------------------------------------------------------------------
2.4  DESIGN PRINCIPLES
-------------------------------------------------------------------------------

  INCREMENTAL
    Every photo is identified by its MD5 hash. On re-run, photos already
    in the SQLite cache are skipped entirely. Only new or changed files
    are processed. A 10,000-photo library with 50 new photos takes seconds
    for stages 1-2 on the second run.

  RESUMABLE
    All intermediate results are persisted to SQLite after each stage.
    --from-stage N skips all earlier stages and reads the DB state instead.
    Useful for re-tuning config.yaml without repeating ML inference.

  MEMORY-EFFICIENT
    Photos are resized to max_dimension (default 1024px) before any ML
    processing. Images are never held in memory across loop iterations.
    Embeddings are stored as raw float32 BLOBs (512 * 4 = 2 KB each).

  PRIVACY-FIRST
    All processing is local. Privacy filters run before ML scoring so
    sensitive documents never enter the ranking pool. Home-location
    filtering is opt-in via config.yaml.

  SEPARATION OF CONCERNS
    Each src/ module has one job. main.py is the only orchestrator.
    No module imports another except through the DB or explicit parameters.
    (Exception: aesthetic.py and scene_tagger.py reuse embeddings.py
    model singleton to avoid loading CLIP twice.)


===============================================================================
## 3. PIPELINE FLOW
===============================================================================

-------------------------------------------------------------------------------
3.1  END-TO-END PIPELINE (9 STAGES)
-------------------------------------------------------------------------------

  INPUT: data/input_photos/**  (JPG, PNG, HEIC, TIFF, WebP, BMP)
  OUTPUT: data/output_photos/  (resized JPEG + output.json)

  +-------+
  | START |
  +---+---+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 1: SCAN                              ingestion.py      |
  |  - Recursive rglob for supported extensions                  |
  |  - Sort paths for deterministic ordering                     |
  |  - Compute MD5 hash per file                                 |
  |  - If hash matches DB record -> SKIP (incremental)           |
  +---+----------------------------------------------------------+
      |  N image paths (only unprocessed)
      v
  +---+----------------------------------------------------------+
  | STAGE 2: FEATURE EXTRACTION               per-image loop    |
  |                                                              |
  |  For each image:                                             |
  |    metadata.py    -> timestamp, GPS lat/lon, camera model   |
  |    quality.py     -> blur_score, exposure_score, resolution  |
  |    embeddings.py  -> 512-dim CLIP vector (MPS)               |
  |    dedup          -> pHash string                            |
  |    face_detection -> face_count, avg 512-dim face vector     |
  |    privacy.py     -> is_screenshot / is_document / is_home   |
  |                                                              |
  |  -> upsert all fields into SQLite photos table               |
  +---+----------------------------------------------------------+
      |  All features in DB
      v
  +---+----------------------------------------------------------+
  | STAGE 3: AESTHETIC SCORING                aesthetic.py      |
  |  - Reads clip_emb BLOB from DB (no re-inference)            |
  |  - Computes cosine sim vs positive/negative prompt vectors   |
  |  - Writes aesthetic_score [0,1] back to DB                  |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 4: SCENE TAGGING                    scene_tagger.py   |
  |  - Reads clip_emb BLOB from DB                              |
  |  - Softmax over 28 scene label embeddings                   |
  |  - Writes JSON scene_tags to DB                             |
  |    e.g. [{"label":"beach","confidence":0.42}, ...]          |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 5: SENTIMENT                        sentiment.py      |
  |  - Only runs on photos where face_count >= 1                |
  |  - Reloads image, runs MediaPipe Face Mesh (468 landmarks)  |
  |  - Computes Eye Aspect Ratio (EAR) + smile angle            |
  |  - Writes smile_score [0,1] to DB                           |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 6: DEDUPLICATION                    deduplication.py  |
  |  - Loads (path, file_hash, phash, clip_emb, blur_score)     |
  |  - Sorts by blur_score DESC (sharpest copy kept)            |
  |  - For each photo, compares against all kept photos:         |
  |      Pass 1: exact MD5 match?        -> duplicate           |
  |      Pass 2: pHash Hamming dist <= 8? -> duplicate          |
  |      Pass 3: CLIP cosine sim >= 0.95? -> duplicate          |
  |  - Marks losers: UPDATE photos SET is_duplicate=1           |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 7: EVENT CLUSTERING                 clustering.py     |
  |  - WHERE quality_pass=1 AND is_private=0 AND is_duplicate=0 |
  |  - Builds feature matrix per photo:                         |
  |      col 0   : timestamp  (StandardScaler * time_weight)    |
  |      col 1-2 : lat, lon   (StandardScaler * gps_weight)     |
  |      col 3+  : PCA(clip_emb, 32 dims) * visual_weight       |
  |  - DBSCAN(eps=0.6, metric='cosine')                         |
  |  - Writes cluster_id to DB (-1 = noise/singleton)           |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 8: FACE IDENTITY CLUSTERING         face_clustering.py|
  |  - WHERE face_count >= 1                                    |
  |  - L2-normalises all face_emb vectors                       |
  |  - DBSCAN(eps=0.8, metric='euclidean') on face embeddings   |
  |  - person_id = cluster label                                |
  |  - is_frequent = 1 if person appears in >= 5 photos        |
  |  - Writes person_id + is_frequent to DB                    |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 9a: RANKING                         ranking.py        |
  |  - WHERE quality_pass=1, is_private=0, is_duplicate=0       |
  |  - Computes 7-component weighted score (see section 4.3)    |
  |  - Writes score [0,1] to DB                                 |
  +---+----------------------------------------------------------+
      |
      v
  +---+----------------------------------------------------------+
  | STAGE 9b: SELECTION + OUTPUT              selection.py      |
  |  - 30/30/40 three-bucket strategy (see section 3.3)         |
  |  - Resizes each selected photo to 2560px long side          |
  |  - Saves as JPEG quality 92                                 |
  |  - Writes output.json report                                |
  +---+----------------------------------------------------------+
      |
      v
  +--------+
  | OUTPUT |  data/output_photos/*.jpg  +  output.json
  +--------+


-------------------------------------------------------------------------------
3.2  SINGLE-PHOTO DATA FLOW
-------------------------------------------------------------------------------

  FILE: IMG_1234.HEIC  (4000x3000, 5.2 MB, iPhone 14 Pro)
  |
  +-- ingestion.load_image_safe()
  |     HEIC -> RGB PIL Image, resized to 1024x768
  |
  +-- metadata.extract_exif()
  |     timestamp = 1704067200.0  (2024-01-01 12:00:00)
  |     lat = 48.8566, lon = 2.3522  (Paris)
  |     camera_model = "iPhone 14 Pro"
  |     has_gps = True
  |
  +-- quality.assess()
  |     blur_score    = 312.4   (Laplacian variance, > 50 -> passes)
  |     exposure_score = 0.61   (0.15 - 0.90 range -> passes)
  |     resolution    = 768     (> 640 -> passes)
  |     quality_pass  = True
  |
  +-- embeddings.extract()   [MPS accelerated]
  |     CLIP ViT-B/32 -> 512-dim float32 normalised vector
  |     [0.021, -0.043, 0.118, ...]
  |
  +-- deduplication.compute_phash()
  |     "a7c3f2b1d4e89012"
  |
  +-- face_detection.detect()   [MTCNN + FaceNet]
  |     face_count   = 2
  |     face_emb     = averaged 512-dim vector for both faces
  |
  +-- privacy.assess()
  |     is_screenshot()?  No  (has GPS + camera model)
  |     is_home_private()? No  (Paris != home)
  |     is_document_clip()? No  (outdoor scene)
  |     is_private = False
  |
  +-- database.upsert()
  |     -> all fields written to photos table
  |
  [Later, Stage 3 - reading from DB, no image reload needed]
  +-- aesthetic.score_from_embedding(clip_emb)
  |     pos_sim = 0.38,  neg_sim = 0.12
  |     aesthetic_score = 0.73
  |
  [Stage 4]
  +-- scene_tagger.classify(clip_emb)
  |     [("city street or urban scene", 0.31),
  |      ("famous landmark", 0.24),
  |      ("travel", 0.19)]
  |     scene_tags = '[{"label":"city street...","confidence":0.31},...]'
  |
  [Stage 5 - image reloaded once for MediaPipe]
  +-- sentiment.score_image(img)
  |     left_EAR  = 0.31  (open)
  |     right_EAR = 0.29  (open)
  |     smile     = 0.78  (corners raised)
  |     smile_score = 0.84
  |
  [Stage 8 - face identity]
  +-- face_clustering result
  |     person_id  = 3   (same person seen in 12 other photos)
  |     is_frequent = True
  |
  [Stage 9a]
  +-- ranking.score_photos()
  |     sharpness  = 0.81 (normalised)
  |     aesthetic  = 0.73
  |     face_score = 0.69 (2 faces, log-scaled)
  |     sentiment  = 0.84
  |     uniqueness = 1.0  (not a duplicate)
  |     meta       = 1.0  (has GPS + timestamp)
  |     diversity  = 0.72
  |     score = 0.15*0.81 + 0.25*0.73 + 0.15*0.69 + 0.15*0.84
  |           + 0.15*1.0  + 0.08*1.0  + 0.07*0.72  = 0.784
  |
  [Stage 9b]
  +-- selection: score=0.784, person_id=3 (frequent), cluster=7 (Paris)
  |     -> included in "people" bucket  (person 3 has < 5 photos so far)
  |
  +-- output: resize 1024x768 -> 853x640  (long side 853 <= 2560, no resize)
        save as data/output_photos/IMG_1234.jpg  (JPEG quality 92)


-------------------------------------------------------------------------------
3.3  SELECTION STRATEGY (30 / 30 / 40)
-------------------------------------------------------------------------------

  Total budget: 1 073 741 824 bytes (1 GB)

  Pre-filter: quality_pass=1, is_duplicate=0, is_private=0
  Sort all candidates by score DESC

                    +---------------------------+
                    |   ELIGIBLE PHOTO POOL     |
                    |   (scored, sorted)        |
                    +-------------+-------------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
  +-----------+------+  +---------+--------+  +-------+---------+
  | BUCKET 1         |  | BUCKET 2         |  | BUCKET 3        |
  | People  (30%)    |  | Location (30%)   |  | Aesthetic (40%) |
  |                  |  |                  |  |                 |
  | For each person  |  | For each event   |  | Highest score   |
  | with is_frequent |  | cluster (GPS)    |  | regardless of   |
  | take top N       |  | take top M       |  | subject or      |
  | (max 5 per       |  | (max 15 per      |  | person          |
  |  person)         |  |  cluster)        |  |                 |
  | Budget: 300 MB   |  | Budget: 300 MB   |  | Budget: 400 MB  |
  +------------------+  +------------------+  +-----------------+
              |                   |                   |
              +-------------------+-------------------+
                                  |
                         Deduplicate across
                         the three buckets
                         (a photo can qualify
                          for all three but
                          is only copied once)
                                  |
                                  v
                    +-------------+-------------+
                    |   FINAL SELECTION         |
                    |   resized to 2560px JPEG  |
                    |   data/output_photos/     |
                    +---------------------------+

  NOTE: Estimated output size uses post-resize bytes, not originals.
        A 12 MP original (~4 MB) -> ~1.5 MB after resize = 2.5x more
        photos fit within the 1 GB budget compared to copying originals.


===============================================================================
## 4. LOW LEVEL DESIGN (LLD)
===============================================================================

-------------------------------------------------------------------------------
4.1  MODULE REFERENCE
-------------------------------------------------------------------------------

  src/ingestion.py
  ----------------
  scan_photos(folder, extensions) -> List[Path]
    Recursive rglob, sorted, filters by extension set.

  compute_file_hash(path) -> str
    MD5 hex digest in 64 KB chunks. Used as cache key.

  load_image_safe(path, max_dimension) -> PIL.Image | None
    Opens image, converts to RGB, resizes longest side to max_dimension
    using LANCZOS resampling. Returns None on corrupt/unreadable files.
    Registers pillow-heif opener for HEIC/HEIF format support.

  ----------------------------------------

  src/metadata.py
  ---------------
  extract_exif(path) -> dict
    Keys: timestamp (float), lat (float), lon (float),
          camera_model (str), has_gps (bool)
    Uses piexif. GPS DMS -> decimal via Haversine.
    Returns zeroed dict on missing/corrupt EXIF.

  is_rare_location(lat, lon, home, radius_km) -> bool
    Haversine distance check. True if outside home radius.

  ----------------------------------------

  src/quality.py
  --------------
  blur_score(img) -> float
    Laplacian variance via 4-neighbour finite difference (pure numpy,
    no scipy). Higher = sharper. Typical range: 10 (blurry) - 2000 (sharp).

    Formula:
      lap = left + right + top + bottom - 4 * center
      return lap.var()

  exposure_score(img) -> float
    Mean grayscale pixel / 255.  Range: [0, 1].
    0.15-0.90 is the acceptable window.

  resolution(img) -> int
    min(width, height). Conservative measure.

  assess(img, min_blur, min_exposure, max_exposure, min_resolution)
    -> QualityResult(blur_score, exposure_score, resolution, passes)

  ----------------------------------------

  src/embeddings.py
  -----------------
  load_model() -> (model, preprocess, device)
    Singleton. Loaded once per process. Device priority: MPS > CUDA > CPU.

  extract(img, model, preprocess, device) -> np.ndarray[512]
    Single image. Returns L2-normalised 512-dim float32 vector.
    Returns zeros on failure (image still proceeds through pipeline).

  batch_extract(images, batch_size=16) -> np.ndarray[N, 512]
    Stacks tensors, runs in batches. MPS benefits from batch >= 8.

  cosine_similarity(a, b) -> float
    dot(a,b) / (norm(a) * norm(b)). Guards against zero vectors.

  ----------------------------------------

  src/face_detection.py
  ---------------------
  load_models() -> (mtcnn, facenet, device)
    Singleton. MTCNN on CPU (stability). FaceNet on MPS/CUDA/CPU.

  detect(img) -> (face_count: int, avg_embedding: np.ndarray | None)
    MTCNN returns tensor (N, 3, 160, 160) or None.
    FaceNet encodes each face crop -> mean embedding across N faces.
    Pixel normalisation: (pixels / 127.5) - 1.0  =>  [-1, 1] range.

  face_score(face_count, max_faces=6) -> float
    log(face_count + 1) / log(max_faces + 1)  capped at 1.0

  ----------------------------------------

  src/aesthetic.py
  ----------------
  _build_prompt_vectors()
    Encodes POSITIVE_PROMPTS and NEGATIVE_PROMPTS through CLIP text encoder.
    Averages each list into a single 512-dim direction vector.
    Cached as module globals (built once per process).

  score_from_embedding(clip_emb, use_laion=False) -> float
    CLIP mode:  raw = (pos_sim - neg_sim + 2.0) / 4.0  -> [0,1]
    LAION mode: downloads MLP weights from HuggingFace on first run,
                outputs 1-10 scale, normalised to [0,1].

  ----------------------------------------

  src/scene_tagger.py
  -------------------
  _build_label_features()
    Tokenises 28 prompts ("a photo of {label}") and encodes via CLIP text.
    Stored as (28, 512) matrix. Built once per process.

  classify(clip_emb, top_n=3) -> List[(label, confidence)]
    Softmax over dot products:  logits = clip_emb @ label_matrix.T
    Returns top_n labels above min_confidence threshold.

  tags_to_json / tags_from_json
    Serialisation helpers for SQLite TEXT column storage.

  ----------------------------------------

  src/sentiment.py
  ----------------
  score_image(img) -> float
    MediaPipe FaceMesh: static_image_mode=True, max_num_faces=5.
    Returns 0.5 if no faces detected or MediaPipe not installed.

    EAR (Eye Aspect Ratio):
      EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
      Open eye: EAR >= 0.20
      Uses 6 landmarks per eye (3 pairs: outer/inner/top1/top2/bot1/bot2)

    Smile:
      corner_avg_y = mean(left_corner.y, right_corner.y)
      raw = (upper_lip.y - corner_avg_y) * 25.0 + 0.5
      Smiling: corners are higher (smaller y in norm coords) than upper lip

    Per-face score = 0.55 * eye_open + 0.45 * smile
    Returns max score across all detected faces.

  ----------------------------------------

  src/deduplication.py
  --------------------
  compute_phash(img) -> str
    imagehash.phash() -> 64-bit hex string.

  hamming_distance(h1, h2) -> int
    XOR the two hex integers, count set bits. Returns 64 on invalid input.

  find_duplicates(records, phash_threshold, embedding_threshold) -> Set[str]
    Input records sorted by blur_score DESC (sharpest first).
    For each photo, compare against all already-kept photos:
      1. file_hash exact match
      2. hamming_distance(phash) <= phash_threshold   (default 8)
      3. cosine_similarity(clip_emb) >= threshold     (default 0.95)
    Returns Set[path] of photos to mark as duplicates.

  ----------------------------------------

  src/clustering.py
  -----------------
  _build_feature_matrix(records, time_weight, gps_weight, visual_weight)
    -> np.ndarray[N, 3+PCA_DIM]

    Step 1: Build (N, 515) raw matrix [timestamp, lat, lon, clip_emb...]
    Step 2: StandardScaler on each feature group independently
    Step 3: PCA(32) on CLIP columns (512 -> 32 dims)
    Step 4: Apply weights to each group
    Result: (N, 35) matrix for DBSCAN

  cluster_events(records, eps, min_samples, ...) -> Dict[path, cluster_id]
    DBSCAN(metric='cosine') on the 35-dim matrix.
    cluster_id == -1: noise / singleton (no nearby neighbours).

  ----------------------------------------

  src/face_clustering.py
  ----------------------
  cluster_identities(records, eps, min_samples, frequent_threshold)
    -> Dict[path, (person_id, is_frequent)]

    Collects face_emb from all photos with face_count >= 1.
    L2-normalises all vectors.
    DBSCAN(eps=0.8, metric='euclidean').
    Counts photos per cluster: if count >= frequent_threshold -> is_frequent.
    Photos without face_emb get (person_id=-1, is_frequent=False).

  ----------------------------------------

  src/privacy.py
  --------------
  is_screenshot(img, camera_model, has_gps) -> bool
    Fast heuristic (no ML):
      - has_gps=True or camera_model non-empty -> not a screenshot
      - Check (width, height) against set of 15 known screen resolutions
        (both portrait and landscape orientations)

  is_document_clip(img) -> bool
    CLIP zero-shot: 6 private prompts + 1 normal photo prompt.
    Softmax probability of private prompts summed > 0.35 -> True.
    Uses cached text features (built once).

  is_home_private(lat, lon, face_count, home, radius_km) -> bool
    Haversine distance <= radius_km AND face_count <= 1.

  assess(img, ...) -> bool
    Ordered cheapest-first:
      1. is_screenshot   (no ML, instant)
      2. is_home_private (no ML, fast)
      3. is_document_clip (CLIP, slower)

  ----------------------------------------

  src/ranking.py
  --------------
  _minmax(values) -> np.ndarray
    (x - min) / (max - min). Returns 0.5 array if all values equal.

  score_photos(records, weights) -> Dict[path, float]
    See section 4.3 for full formula.

  ----------------------------------------

  src/selection.py
  ----------------
  select_photos(records, scores, ...) -> List[dict]
    Three-bucket strategy. See section 3.3.
    _est_size(rec): estimates post-resize JPEG size for budget math.
      est = orig_size * (long_side / max_orig_dim)^2 * 0.75

  copy_to_output(selected, scores, output_dir, resize, ...)
    _resize_and_save(src, dst, long_side, quality):
      Opens original, converts RGB, resizes with LANCZOS if needed,
      saves as JPEG with optimize=True.

  ----------------------------------------

  src/database.py
  ---------------
  connect(db_path) -> Connection
    WAL journal mode + NORMAL synchronous = fast writes, crash-safe reads.

  init_db(db_path)
    CREATE TABLE IF NOT EXISTS + _add_column_if_missing() migration.
    Safe to call on existing databases.

  upsert(conn, data)
    INSERT ... ON CONFLICT(path) DO UPDATE SET ...
    Dynamically builds SQL from dict keys.

  emb_to_blob / blob_to_emb
    np.ndarray <-> bytes via float32 tobytes / frombuffer.
    512 floats = 2 048 bytes per embedding.


-------------------------------------------------------------------------------
4.2  DATABASE SCHEMA
-------------------------------------------------------------------------------

  TABLE: photos

  Column            Type     Default    Description
  ----------------  -------  ---------  --------------------------------------
  id                INTEGER  PK         Auto-increment row ID
  path              TEXT     UNIQUE     Absolute file path (primary key)
  file_hash         TEXT               MD5 hex digest (cache key)
  file_size         INTEGER  0          Original file size in bytes

  -- EXIF / metadata --
  timestamp         REAL     0          Unix timestamp from DateTimeOriginal
  lat               REAL     0          GPS latitude (decimal degrees)
  lon               REAL     0          GPS longitude (decimal degrees)
  camera_model      TEXT     ''         Camera or phone model string
  has_gps           INTEGER  0          1 if real GPS data present

  -- Quality --
  blur_score        REAL     0          Laplacian variance (higher = sharper)
  exposure_score    REAL     0.5        Mean brightness 0-1
  resolution        INTEGER  0          Shorter dimension in pixels
  quality_pass      INTEGER  1          0 = failed quality gate, excluded

  -- Feature vectors (binary) --
  clip_emb          BLOB               512 x float32 = 2 048 bytes
  phash             TEXT     ''         64-bit pHash hex string
  face_emb          BLOB               512 x float32 = 2 048 bytes

  -- Face --
  face_count        INTEGER  0          Number of faces detected
  person_id         INTEGER  -1         Face cluster ID (-1 = unknown)
  is_frequent       INTEGER  0          1 = appears in >= N photos

  -- Content --
  aesthetic_score   REAL     0          CLIP aesthetic quality [0,1]
  scene_tags        TEXT     ''         JSON array of {label, confidence}
  smile_score       REAL     0.5        MediaPipe sentiment [0,1]

  -- Flags --
  is_duplicate      INTEGER  0          1 = marked as near-duplicate
  is_private        INTEGER  0          1 = screenshot / doc / home shot

  -- Output --
  cluster_id        INTEGER  -1         Event cluster from DBSCAN
  score             REAL     0          Final composite rank score [0,1]
  selected          INTEGER  0          1 = chosen for output folder
  processed_at      REAL     0          Unix timestamp of processing

  INDEXES:
    idx_hash    ON photos(file_hash)    -- fast cache lookup
    idx_cluster ON photos(cluster_id)  -- cluster stats
    idx_person  ON photos(person_id)   -- person queries


-------------------------------------------------------------------------------
4.3  RANKING FORMULA
-------------------------------------------------------------------------------

  All 7 components are min-max normalised to [0,1] before weighting.
  Normalisation is computed across the current eligible photo set
  (quality_pass=1, is_private=0, is_duplicate=0).

  score = w1 * sharpness
        + w2 * aesthetic
        + w3 * face_score
        + w4 * sentiment
        + w5 * uniqueness
        + w6 * metadata_importance
        + w7 * diversity_bonus

  Component            Default  Source
  -------------------  -------  -------------------------------------------
  sharpness            0.15     minmax(blur_score)
  aesthetic            0.25     minmax(aesthetic_score)
  face_score           0.15     minmax(log1p(clip(face_count, 0, 10)))
  sentiment            0.15     minmax(smile_score)
  uniqueness           0.15     0.0 if is_duplicate else 1.0
  metadata_importance  0.08     (has_gps ? 1.0 : 0.4) * (timestamp>0 ? 1.0 : 0.6)
  diversity_bonus      0.07     minmax(cluster_size / max_cluster_size)
                       ----
  Total                1.00

  aesthetic_score detail:
    pos_sim = cosine(clip_emb, avg_positive_prompts)
    neg_sim = cosine(clip_emb, avg_negative_prompts)
    raw = (pos_sim - neg_sim + 2.0) / 4.0  ->  [0, 1]

  face_score detail:
    log1p prevents very large groups from dominating.
    face_count=1 -> 0.39,  face_count=3 -> 0.73,  face_count=6 -> 1.0


-------------------------------------------------------------------------------
4.4  DEDUPLICATION LOGIC
-------------------------------------------------------------------------------

  The algorithm is O(N^2) in the worst case but fast in practice because
  pHash comparison (step 2) short-circuits before CLIP comparison (step 3).

  Input photos sorted by blur_score DESC:
  [sharp.jpg, medium.jpg, blurry.jpg, duplicate_of_sharp.jpg, ...]

  for each candidate in sorted list:
    if candidate.path in to_remove: skip

    for each keeper in accepted list:
      if candidate.file_hash == keeper.file_hash:     -> duplicate
      if hamming(candidate.phash, keeper.phash) <= 8: -> duplicate
      if cosine(candidate.clip_emb, keeper.clip_emb) >= 0.95: -> duplicate

    if no match found: add to accepted list
    else: add to to_remove set

  Result: to_remove contains paths of all weaker copies.
  The kept copy is always the sharpest (highest blur_score).

  Threshold meanings:
    pHash Hamming distance:
      0 = exact visual duplicate (different compression)
      8 = very similar (burst shots, slight crop)
      16 = visually related but distinct

    CLIP cosine similarity:
      0.95 = nearly identical content
      0.90 = same scene, slightly different framing
      0.80 = related scenes


-------------------------------------------------------------------------------
4.5  EVENT CLUSTERING ALGORITHM
-------------------------------------------------------------------------------

  Goal: group photos from the same occasion (trip, party, hike) together.

  Feature engineering:

    Raw features per photo:    [timestamp,  lat,    lon,    clip_emb (512)]
    After StandardScaler:      [t_norm,     lat_n,  lon_n,  emb_norm (512)]
    After PCA (32 components): [t_norm,     lat_n,  lon_n,  emb_pca  (32) ]
    After weights applied:     [t*1.0,      gps*2.0,        emb*1.5       ]
    Final matrix shape:        (N, 35)

  DBSCAN parameters:
    eps = 0.6          distance threshold (cosine metric)
    min_samples = 2    min photos to form a cluster

  Why cosine metric:
    The combined feature vector spans different scales and semantics.
    Cosine distance normalises for magnitude, focusing on direction —
    photos with similar relative feature patterns cluster together
    regardless of absolute feature values.

  Why PCA from 512 to 32:
    DBSCAN with cosine metric degrades in high-dimensional spaces
    (curse of dimensionality). PCA retains ~80% of variance in 32 dims
    while making cluster boundaries meaningful.

  Cluster -1 (noise):
    Singletons or outliers. These are still ranked and can be selected —
    they just don't benefit from the diversity_bonus weight.


-------------------------------------------------------------------------------
4.6  FACE IDENTITY CLUSTERING
-------------------------------------------------------------------------------

  Goal: identify "Person A" across hundreds of photos without any labelling.

  Input: face_emb (512-dim FaceNet vectors) for all photos with faces.

  Processing:
    1. L2-normalise all embeddings
    2. DBSCAN(eps=0.8, metric='euclidean')
       - Euclidean distance on L2-normalised vectors is equivalent to
         angular distance, matching FaceNet's training objective
    3. Count photos per cluster
    4. Clusters with count >= frequent_threshold -> is_frequent = True

  Output per photo:
    person_id = -1         -> background person (one-off appearance)
    person_id = 3          -> identified individual #3
    is_frequent = True     -> appears in >= 5 photos (family / friend)

  Limitation note:
    face_emb is the AVERAGE of all faces in a multi-face photo.
    This is imprecise for group shots. Only solo shots (face_count=1)
    produce clean individual embeddings. Group shots may cluster noisily.
    Improvement path: store per-face embeddings as separate DB rows.

  How is_frequent is used in selection:
    Bucket 1 (30% budget) only draws from photos where is_frequent=True.
    This ensures every important person in your library is represented
    in the final output.


-------------------------------------------------------------------------------
4.7  PRIVACY FILTER CHAIN
-------------------------------------------------------------------------------

  Three checks, ordered cheapest-first:

  Check 1: Screenshot heuristic (< 1ms, no ML)
  +------------------------------------------+
  | has_gps == True       -> NOT screenshot   |
  | camera_model != ''    -> NOT screenshot   |
  | (w,h) in SCREEN_RES   -> IS  screenshot   |
  | (h,w) in SCREEN_RES   -> IS  screenshot   |
  +------------------------------------------+

  Screen resolution set includes 15 common phone and desktop
  resolutions in both portrait and landscape orientations.

  Check 2: Home + solo heuristic (< 1ms, no ML)
  +------------------------------------------+
  | Only active if filter_home_private: true   |
  | haversine(photo_gps, home_coords)          |
  |   <= home_radius_km (default 0.5 km)       |
  | AND face_count <= 1                        |
  | -> IS private                              |
  +------------------------------------------+

  Check 3: CLIP document detection (~5-10ms, ML)
  +------------------------------------------+
  | Encodes 7 text prompts (6 private + 1     |
  | normal) through CLIP text encoder once.   |
  | Softmax over dot products with image emb. |
  | Sum of private prompt probabilities        |
  |   > 0.35 threshold -> IS document/private |
  +------------------------------------------+

  Private prompts:
    "a photo of a document or paper"
    "a bank card or credit card"
    "a receipt or invoice"
    "an identity card or passport"
    "a screenshot of a phone or computer screen"
    "handwritten notes or a whiteboard"

  Normal prompt:
    "a natural photo taken with a camera outdoors or indoors"


-------------------------------------------------------------------------------
4.8  OUTPUT RESIZING
-------------------------------------------------------------------------------

  Why resize?
    A 12 MP photo at full resolution is ~3.5-5 MB as JPEG.
    Most photo viewers display at 2K (2560x1440) or 4K (3840x2160).
    Resizing to 2560px long side reduces size to ~1.2-1.8 MB,
    fitting 2.5x more photos within the 1 GB budget.

  Algorithm:
    w, h = original image dimensions
    if max(w, h) > long_side:
        scale = long_side / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    save as JPEG, quality=92, optimize=True

  Quality 92 vs 95:
    Quality 95: ~2.0 MB average, visually indistinguishable from original
    Quality 92: ~1.5 MB average, no perceptible difference at normal viewing
    Quality 85: ~1.0 MB average, slight compression artefacts in gradients

  LANCZOS resampling:
    Highest quality downsampling filter. Slower than BICUBIC but produces
    sharper results, especially for text or fine detail in photos.

  Budget estimation (used in selection, before actual resize):
    est_bytes = orig_size * (long_side / max_orig_dim)^2 * 0.75
    The 0.75 factor accounts for JPEG's efficiency vs raw pixel size.
    This is an approximation — actual JPEG size varies by content.


===============================================================================
## 5. MODULE DEPENDENCY GRAPH
===============================================================================

  Arrows show "imports / calls". database.py is used by everything
  (omitted from arrows for clarity).

                         config.yaml
                              |
                         main.py  <-- you run this
                              |
        +---------------------+---------------------+
        |          |          |          |          |
        v          v          v          v          v
  ingestion   metadata    quality   embeddings  face_detection
                                       |              |
                              +--------+--------+     |
                              |        |        |     |
                              v        v        v     v
                          aesthetic  scene  privacy  face_clustering
                          (reads DB) tagger (CLIP)
                                      (reads DB)
                                           |
                              +------------+------------+
                              |            |            |
                              v            v            v
                        deduplication  clustering   sentiment
                        (pHash+CLIP)   (DBSCAN)   (MediaPipe)
                              |            |            |
                              +------------+------------+
                                           |
                                      ranking.py
                                           |
                                      selection.py
                                           |
                                      output/


  Shared model singletons (loaded once per process):
    CLIP model     : embeddings.py -> also used by aesthetic.py,
                                      scene_tagger.py, privacy.py
    MTCNN+FaceNet  : face_detection.py
    MediaPipe      : sentiment.py


===============================================================================
## 6. CONFIGURATION REFERENCE
===============================================================================

  All settings live in config.yaml. Run python main.py --config my.yaml
  to use a different file.

  paths:
    input                 Path to your photo library folder
    output                Path where curated photos are written
    cache                 SQLite database path
    models                Directory for downloaded model weights

  ingestion:
    supported_extensions  List of file extensions to include
    max_dimension         Resize longest side to this before processing
                          Lower = faster, less accurate. Recommended: 1024
    batch_size            CLIP batch size. 16 is good for 8 GB RAM.

  quality:
    min_blur_score        Laplacian variance threshold.
                          50 = moderate quality. 100 = stricter.
    min_exposure_score    Min mean brightness. 0.15 rejects very dark photos.
    max_exposure_score    Max mean brightness. 0.90 rejects overexposed.
    min_resolution        Minimum shorter side in pixels. 640 = standard.

  deduplication:
    phash_threshold       Hamming distance. 8 = burst shots. 4 = strict.
    embedding_similarity  Cosine threshold. 0.95 = near-identical.

  privacy:
    filter_screenshots    Remove screenshots (heuristic + resolution check)
    filter_documents      Remove IDs, receipts, docs (CLIP zero-shot)
    filter_home_private   Remove solo shots at home GPS location
    home_coords           [lat, lon] for home location. null to disable.
    home_radius_km        Radius around home. 0.5 km = one city block.

  clustering:
    time_weight           Importance of timestamp in event clustering
    gps_weight            Importance of GPS location
    visual_weight         Importance of CLIP visual similarity
    eps                   DBSCAN neighbourhood radius
    min_samples           Min photos to form an event cluster

  aesthetic:
    use_laion_predictor   false = CLIP zero-shot (offline, instant)
                          true  = download LAION MLP from HuggingFace

  scene_tagging:
    enabled               Set false to skip (saves a few seconds)
    top_n                 Number of scene labels to store per photo

  sentiment:
    enabled               Set false if MediaPipe not installed
    min_face_for_sentiment Only run on photos with >= N faces

  face_clustering:
    enabled               Set false to skip identity clustering
    eps                   DBSCAN eps for face embeddings. 0.6-1.0.
    min_samples           Min photos to form an identity cluster
    frequent_person_threshold  Min photos to mark someone as frequent

  ranking.weights:
    sharpness             Weight for blur quality component
    aesthetic             Weight for CLIP aesthetic score
    face_score            Weight for face count (log-scaled)
    sentiment             Weight for smile + eye openness
    uniqueness            Weight for non-duplicate penalty
    metadata_importance   Weight for GPS + timestamp presence
    diversity_bonus       Weight for event cluster diversity

  selection:
    max_output_bytes      Total output budget in bytes. 1 GB default.
    people_budget_fraction    Fraction for people bucket (0.30)
    location_budget_fraction  Fraction for location bucket (0.30)
    aesthetic_budget_fraction Fraction for aesthetic bucket (0.40)
    max_per_person        Max photos of any one person (5)
    max_per_location      Max photos from one location cluster (15)
    max_per_cluster       Backup event cap (10)
    resize_output         true = resize photos before writing
    output_long_side      Target pixel size for longest dimension
    output_jpeg_quality   JPEG quality 1-100. 92 recommended.

  output:
    generate_report       Write output.json alongside photos
    report_filename       Name of the JSON report file


===============================================================================
## 7. FILE & DIRECTORY LAYOUT
===============================================================================

  photo-curator/
  |
  +-- main.py                   Pipeline orchestrator. Run this.
  +-- config.yaml               All tunable parameters.
  +-- requirements.txt          Python dependencies.
  |
  +-- data/
  |   +-- input_photos/         [ PUT YOUR PHOTOS HERE ]
  |   |   +-- 2024/             Subdirectories are scanned recursively.
  |   |   +-- holidays/
  |   |   +-- ...
  |   +-- output_photos/        [ CURATED OUTPUT WRITTEN HERE ]
  |       +-- IMG_1234.jpg      Resized JPEG (2560px long side)
  |       +-- ...
  |       +-- output.json       Curation report (score, faces, GPS, etc.)
  |
  +-- cache/
  |   +-- photo_db.sqlite       SQLite feature cache.
  |                             Safe to delete — will be rebuilt on next run.
  |
  +-- models/
  |   +-- aesthetic_predictor.pth   (downloaded on demand if LAION enabled)
  |
  +-- src/
      +-- __init__.py
      +-- ingestion.py          Scan + hash + load images
      +-- metadata.py           EXIF extraction (timestamp, GPS, camera)
      +-- quality.py            Blur + exposure + resolution metrics
      +-- embeddings.py         CLIP ViT-B/32 (MPS / CUDA / CPU)
      +-- face_detection.py     MTCNN detector + FaceNet embedder
      +-- face_clustering.py    Cross-photo person identity via DBSCAN
      +-- aesthetic.py          CLIP zero-shot aesthetic scoring
      +-- scene_tagger.py       Zero-shot scene classification (28 labels)
      +-- sentiment.py          MediaPipe smile + eye openness
      +-- deduplication.py      pHash Hamming + cosine similarity
      +-- clustering.py         DBSCAN event grouping (time+GPS+CLIP)
      +-- privacy.py            Screenshot + document + home filter
      +-- ranking.py            7-component weighted score
      +-- selection.py          30/30/40 diversity + JPEG resize output
      +-- database.py           SQLite cache layer (upsert + migration)


===============================================================================
## 8. PERFORMANCE & SCALING
===============================================================================

  STAGE TIMING BREAKDOWN (Apple M1, 5 000 photos)

  Stage   Operation                    Time       Notes
  ------  ---------------------------  ---------  ----------------------------
  1       Scan + hash                  ~30 sec    Disk I/O bound
  2       EXIF + quality + CLIP        ~25 min    CLIP is the bottleneck
          + face detection             (~0.3s/img on MPS)
  3       Aesthetic scoring            ~10 sec    Reads from DB, no re-run
  4       Scene tagging                ~10 sec    Reads from DB, no re-run
  5       Sentiment (MediaPipe)        ~8 min     Only on face photos
  6       Deduplication                ~20 sec    O(N^2) pHash fast path
  7       Event clustering             ~15 sec    PCA + DBSCAN
  8       Face identity clustering     ~5 sec     DBSCAN on embeddings
  9       Ranking + selection          ~5 sec     Pure numpy math
          File copy + resize           ~10 min    Disk I/O + PIL resize
  ------  ---------------------------  ---------  ----------------------------
  TOTAL                                ~60 min    First run (all ML)
  RE-RUN (no new photos)               ~15 sec    Only stages 3-9 from DB

  LIBRARY SIZE ESTIMATES (Apple M1)

  Photos     Stage 2 time     Total first run
  ---------  ---------------  ---------------
  1 000      ~5 min           ~15 min
  5 000      ~25 min          ~60 min
  10 000     ~50 min          ~2 hours
  20 000     ~100 min         ~4 hours

  MEMORY USAGE

  Component                Approximate memory
  -----------------------  ------------------
  CLIP model (ViT-B/32)    ~350 MB
  MTCNN + FaceNet          ~250 MB
  MediaPipe                ~100 MB
  SQLite cache (5k photos) ~20 MB (on disk)
  Per-image working set    ~5-20 MB peak
  Total peak               ~800 MB - 1.2 GB

  SPEED-UP TIPS

  1. Increase batch_size to 32 if you have > 8 GB RAM.
  2. Set max_dimension: 512 for a 2x speed boost in exchange for
     slightly less accurate embeddings.
  3. Disable sentiment (sentiment.enabled: false) if you have few
     face photos — saves ~30% of total time.
  4. Use --from-stage to re-run only later stages after config changes.
  5. The SQLite cache is the biggest win for repeated runs. Never delete
     photo_db.sqlite unless you want to reprocess everything.


===============================================================================
## 9. USAGE
===============================================================================

  BASIC

    cd photo-curator
    python main.py

  CUSTOM PATHS

    python main.py --input /Volumes/Photos --output ~/Desktop/Best

  DRY RUN (score + select, no files written)

    python main.py --dry-run

  RESUME FROM STAGE (uses existing DB cache)

    python main.py --from-stage 6    # re-dedup + cluster + rank + select
    python main.py --from-stage 9    # just re-rank and re-select

  CUSTOM CONFIG

    python main.py --config configs/strict.yaml

  TYPICAL TUNING WORKFLOW

    1. Run full pipeline once: python main.py
    2. Review output.json — check scores, scene tags, person IDs
    3. Edit config.yaml (adjust weights, thresholds, budget)
    4. Re-run from stage 9: python main.py --from-stage 9  (seconds)
    5. Iterate until satisfied

  REQUIREMENTS

    Python 3.10+
    macOS with Apple Silicon (MPS), or NVIDIA GPU (CUDA), or CPU fallback
    4 GB RAM minimum, 8 GB recommended
    Disk space: ~100 MB models + cache size (~4 KB per photo in DB)

    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git   # required
    # pip install mediapipe                              # for sentiment
    # pip install huggingface_hub                       # for LAION aesthetic
