"""
Face detection and embedding extraction using facenet-pytorch.

Returns per-image face count, averaged 512-dim embedding, face prominence
(fraction of image area covered by qualifying faces), and mean detection
confidence.

Quality filters applied before counting:
  - Face bounding box must cover >= 0.5 % of the image area (excludes tiny
    background bystanders).
  - MTCNN detection probability must be >= 0.80 (rejects uncertain detections).

MTCNN runs on CPU for stability; FaceNet runs on MPS/CUDA/CPU.

Install:
    pip install facenet-pytorch
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Module-level singletons
_mtcnn = None
_facenet = None
_device = None

# Minimum MTCNN probability to count a face
_MIN_PROB: float = 0.80
# Minimum face area as a fraction of image area
_MIN_FACE_AREA_FRAC: float = 0.005

# FaceNet input size — same as the MTCNN crop size used by facenet-pytorch.
_FACENET_INPUT: int = 160

# Bumped whenever ``detect()``'s output semantics change in a way that makes
# previously-stored ``face_emb`` blobs incompatible with new ones written
# by the same code path. Compared against the per-row ``face_emb_version``
# column in SQLite — rows below the current version are re-extracted by
# ``main.stage_refresh_face_embeddings`` before face clustering runs.
#
# History:
#   0  Legacy (pre-versioning): suffered (a) Pass-2 crop/box-order mismatch
#      that embedded background bystanders in multi-face photos, and (b)
#      mean-of-N FaceNet averaging that produced chimera embeddings for
#      group photos.
#   1  Current: manual cropping from valid_boxes (fixes a) + largest-face
#      embedding only (fixes b).
FACE_EMB_VERSION: int = 1

# Process-level counter of detect() failures that fell back to "no faces".
# A non-zero value at stage end means face/identity signals are missing for
# those photos — surface it via ``detect_failure_count``.
_detect_failure_count = 0


def detect_failure_count() -> int:
    """Number of detect() calls that failed and returned the no-face default."""
    return _detect_failure_count


def reset_detect_failure_count() -> None:
    global _detect_failure_count
    _detect_failure_count = 0


def _select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_models() -> Tuple:
    """
    Load MTCNN (detector) and InceptionResnetV1 (embedder).
    Cached after first call.

    Returns:
        (mtcnn, facenet, device)
    """
    global _mtcnn, _facenet, _device
    if _mtcnn is not None:
        return _mtcnn, _facenet, _device

    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1

    _device = _select_device()
    # MTCNN on CPU — avoid MPS tensor-type issues with detection kernels
    _mtcnn = MTCNN(
        image_size=160,
        keep_all=True,
        device="cpu",
        post_process=False,
        select_largest=False,
        min_face_size=20,       # pixels — ignore sub-20px detections
    )
    _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
    return _mtcnn, _facenet, _device


def detect(
    img: Image.Image,
) -> Tuple[int, Optional[np.ndarray], float, float]:
    """
    Detect faces and compute quality metrics.

    Two-pass approach:
      Pass 1 — ``mtcnn.detect()`` returns bounding boxes + probabilities;
               used for quality filtering, prominence, and confidence.
      Pass 2 — ``mtcnn()`` returns cropped face tensors for FaceNet embedding.

    Args:
        img: PIL Image (RGB)

    Returns:
        face_count:      qualifying face count after size + confidence filtering
        avg_embedding:   mean 512-dim float32 FaceNet embedding (None if 0 faces)
        face_prominence: qualifying face area / image area, capped at 1.0
        face_confidence: mean MTCNN detection probability for qualifying faces
    """
    import torch

    mtcnn, facenet, device = load_models()

    img_w, img_h = img.size
    img_area = max(1, img_w * img_h)
    min_face_area = img_area * _MIN_FACE_AREA_FRAC

    try:
        # ── Pass 1: bounding boxes + probabilities ────────────────
        boxes, probs = mtcnn.detect(img)

        if boxes is None or probs is None:
            return 0, None, 0.0, 0.0

        # Filter to qualifying faces (large enough + high confidence)
        valid_boxes, valid_probs = [], []
        for box, prob in zip(boxes, probs):
            if prob is None or float(prob) < _MIN_PROB:
                continue
            bw = max(0.0, float(box[2] - box[0]))
            bh = max(0.0, float(box[3] - box[1]))
            if bw * bh >= min_face_area:
                valid_boxes.append(box)
                valid_probs.append(float(prob))

        if not valid_boxes:
            return 0, None, 0.0, 0.0

        face_count = len(valid_boxes)

        # Prominence: fraction of frame occupied by qualifying faces
        face_area = sum(
            max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
            for b in valid_boxes
        )
        face_prominence = float(min(1.0, face_area / img_area))
        face_confidence = float(np.mean(valid_probs))

        # ── Pass 2: FaceNet embedding from manual crops of valid_boxes ──
        # We must NOT use ``mtcnn(img)`` here: that returns crops for every
        # box that passed MTCNN's internal min_face_size filter, which is a
        # superset of ``valid_boxes`` (we additionally filter by _MIN_PROB
        # and area). Slicing the first ``face_count`` of mtcnn(img)'s output
        # is misaligned in any image where a low-prob/small box appears
        # earlier in detection order than a qualifying box.
        # Instead, manually crop the qualifying boxes from the original
        # image and resize to FaceNet's expected input.
        
        # To avoid averaging multiple faces (which creates an invalid embedding
        # that doesn't cluster with any individual person), we extract the
        # embedding of the largest (most prominent) face to represent the photo.
        best_box = max(valid_boxes, key=lambda b: max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1])))
        x1, y1, x2, y2 = (int(round(c)) for c in best_box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return face_count, None, face_prominence, face_confidence
            
        crop = img.crop((x1, y1, x2, y2)).resize(
            (_FACENET_INPUT, _FACENET_INPUT), Image.BILINEAR
        )
        arr = np.asarray(crop, dtype=np.float32)  # (H, W, 3) in [0, 255]

        # (H, W, 3) → (1, 3, H, W) for PyTorch
        batch = np.expand_dims(arr, axis=0).transpose(0, 3, 1, 2)
        faces_tensor = torch.from_numpy(batch)
        # Pixel range (0–255) → normalised to [-1, 1] for FaceNet
        faces_norm = (faces_tensor / 127.5) - 1.0
        faces_norm = faces_norm.to(device)

        with torch.no_grad():
            embedding_vecs = facenet(faces_norm)  # (1, 512)

        emb = embedding_vecs[0].cpu().numpy().astype(np.float32)
        return face_count, emb, face_prominence, face_confidence

    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        global _detect_failure_count
        _detect_failure_count += 1
        logger.warning(
            "face_detection.detect failed (img size %sx%s): %s",
            img_w, img_h, exc,
        )
        return 0, None, 0.0, 0.0


def face_score(
    face_count: int,
    face_prominence: float = 0.0,
    max_faces: int = 6,
) -> float:
    """
    Composite face presence score [0–1].

    Blends log-scaled face count (people matter) with subject prominence
    (how much of the frame the faces fill — a close-up portrait scores higher
    than distant background faces).

    65 % count (log-scaled) + 35 % prominence.
    """
    if face_count == 0:
        return 0.0
    count_s = min(1.0, math.log(face_count + 1) / math.log(max_faces + 1))
    # Prominence: ~25 % face coverage → prominence=0.25, scaled to ~0.75
    prom_s = min(1.0, face_prominence * 3.0)
    return 0.65 * count_s + 0.35 * prom_s
