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

import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Module-level singletons
_mtcnn = None
_facenet = None
_device = None

# Minimum MTCNN probability to count a face
_MIN_PROB: float = 0.80
# Minimum face area as a fraction of image area
_MIN_FACE_AREA_FRAC: float = 0.005


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

        # ── Pass 2: face crops for FaceNet embedding ──────────────
        faces = mtcnn(img)  # Tensor (N, 3, 160, 160) or None

        if faces is None:
            return face_count, None, face_prominence, face_confidence

        if faces.dim() == 3:
            faces = faces.unsqueeze(0)

        # Embed only up to face_count crops (skips background bystanders)
        n_embed = min(faces.shape[0], face_count)
        faces = faces[:n_embed]

        # Pixel range (0–255) → normalised to [-1, 1] for FaceNet
        faces_norm = (faces.float() / 127.5) - 1.0
        faces_norm = faces_norm.to(device)

        with torch.no_grad():
            embedding_vecs = facenet(faces_norm)  # (N, 512)

        avg_emb = embedding_vecs.mean(dim=0).cpu().numpy().astype(np.float32)
        return face_count, avg_emb, face_prominence, face_confidence

    except Exception:
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
