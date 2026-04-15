"""
Image ingestion: scan folders, hash files, track processed state.
"""

import hashlib
import os
from pathlib import Path
from typing import Generator, List

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".bmp", ".webp"}


def scan_photos(folder: str, extensions: set = DEFAULT_EXTENSIONS) -> List[Path]:
    """Recursively scan folder and return sorted list of image paths."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    paths = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(paths)


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Return MD5 hex digest of file contents (fast identity check)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_safe(path: Path, max_dimension: int = 1024):
    """
    Open image as RGB PIL Image at its original resolution.

    Returns a 2-tuple ``(img, orig_shorter_side)``:
      - ``img``: full-resolution PIL Image (RGB), or ``None`` on failure.
      - ``orig_shorter_side``: shorter dimension of the image in pixels,
        or 0 on failure.

    Handles HEIC/HEIF via pillow-heif if installed.

    Note: ``max_dimension`` is accepted for API compatibility but the
    downscaling step is disabled — photos are loaded at full resolution.
    """
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass  # will fail at Image.open if not installed

    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        orig_shorter = min(w, h)
        # Downscaling disabled — process and store at original resolution.
        # scale = min(1.0, max_dimension / max(w, h))
        # if scale < 1.0:
        #     img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img, orig_shorter
    except Exception:
        return None, 0
