"""
CLIP image embeddings for visual similarity and clustering.
Optimised for Apple Silicon via the MPS backend.

Install CLIP:
    pip install git+https://github.com/openai/CLIP.git
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Module-level singletons — loaded once, reused across calls
_model = None
_preprocess = None
_device = None


def _select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model() -> Tuple:
    """
    Load CLIP ViT-B/32 model.  Subsequent calls return the cached instance.

    Returns:
        (model, preprocess, device)
    """
    global _model, _preprocess, _device
    if _model is not None:
        return _model, _preprocess, _device

    import torch
    import clip  # pip install git+https://github.com/openai/CLIP.git

    _device = _select_device()
    _model, _preprocess = clip.load("ViT-B/32", device=_device)
    _model.eval()
    return _model, _preprocess, _device


def extract(
    img,
    model=None,
    preprocess=None,
    device=None,
) -> np.ndarray:
    """
    Extract a normalised 512-dim CLIP embedding from a PIL Image.
    Returns a zero vector on failure.
    """
    import torch

    if model is None:
        model, preprocess, device = load_model()

    try:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].astype(np.float32)
    except Exception:
        return np.zeros(512, dtype=np.float32)


def batch_extract(
    images: List,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract normalised CLIP embeddings for a list of PIL Images.

    Returns:
        Float32 array of shape (N, 512).
    """
    import torch

    if not images:
        return np.zeros((0, 512), dtype=np.float32)

    model, preprocess, device = load_model()
    results: List[np.ndarray] = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        try:
            tensors = torch.stack([preprocess(img) for img in batch]).to(device)
            with torch.no_grad():
                embs = model.encode_image(tensors)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            results.append(embs.cpu().numpy())
        except Exception:
            results.append(np.zeros((len(batch), 512), dtype=np.float32))

    return np.vstack(results).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (normalised or not)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
