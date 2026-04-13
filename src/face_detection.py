"""
Face detection and embedding extraction using facenet-pytorch.

Returns per-image face count and an averaged 512-dim face embedding.
MTCNN runs on CPU for stability; FaceNet runs on MPS/CUDA/CPU.

Install:
    pip install facenet-pytorch
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Module-level singletons
_mtcnn = None
_facenet = None
_device = None


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
    )
    _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
    return _mtcnn, _facenet, _device


def detect(img: Image.Image) -> Tuple[int, Optional[np.ndarray]]:
    """
    Detect faces in a PIL Image and return count + averaged embedding.

    Args:
        img: PIL Image (RGB)

    Returns:
        face_count: number of faces found (0 if none)
        avg_embedding: mean 512-dim float32 vector, or None if no faces
    """
    import torch

    mtcnn, facenet, device = load_models()

    try:
        faces = mtcnn(img)  # Tensor (N,3,160,160) or None

        if faces is None:
            return 0, None

        if faces.dim() == 3:
            faces = faces.unsqueeze(0)  # Single face → batch of 1

        face_count = faces.shape[0]

        # Pixel range from MTCNN (0–255) → normalise to [-1, 1]
        faces_norm = (faces.float() / 127.5) - 1.0
        faces_norm = faces_norm.to(device)

        with torch.no_grad():
            embeddings = facenet(faces_norm)  # (N, 512)

        avg_emb = embeddings.mean(dim=0).cpu().numpy().astype(np.float32)
        return face_count, avg_emb

    except Exception:
        return 0, None


def face_score(face_count: int, max_faces: int = 6) -> float:
    """
    Convert face count to a 0–1 score.
    Peaks at ~3–4 faces (group photos), diminishes above max_faces.
    """
    if face_count == 0:
        return 0.0
    # Log scale so a single face already registers, groups score higher
    import math
    return min(1.0, math.log(face_count + 1) / math.log(max_faces + 1))
