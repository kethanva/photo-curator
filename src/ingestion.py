"""
Image ingestion: scan folders, hash files, track processed state.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Generator, List

logger = logging.getLogger(__name__)

# Side effect: allow PIL to load partially-truncated images instead of raising.
# Set once at import time so the global mutation is explicit and not repeated
# per-call. Trade-off: any future code that wants to detect truncation will
# need to inspect the loaded image.
from PIL import ImageFile as _PILImageFile
_PILImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".bmp", ".webp"}

# Process-level counter of unreadable images. Caller surfaces this so the
# operator can investigate corrupted source files instead of silently losing
# them every rerun.
_unreadable_count = 0


def unreadable_count() -> int:
    """Number of images that failed to load via ``load_image_safe``."""
    return _unreadable_count


def reset_unreadable_count() -> None:
    global _unreadable_count
    _unreadable_count = 0


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
    """Return SHA-256 hex digest of file contents (collision-resistant identity)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_file_hash_md5(path: Path, chunk_size: int = 65536) -> str:
    """MD5 fallback used only to validate legacy cache entries written before
    the switch to SHA-256. New code should call ``compute_file_hash``."""
    h = hashlib.md5()  # noqa: S324 - legacy cache compat only
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_matches_cached(path: Path, cached_hash: str, current_sha256: str) -> bool:
    """Return True when the cache entry refers to the same file contents.

    Accepts both modern SHA-256 (64 hex chars) and legacy MD5 (32 hex chars)
    so existing databases keep working without a one-shot reprocess.
    """
    if not cached_hash:
        return False
    if len(cached_hash) == 64:
        return cached_hash == current_sha256
    if len(cached_hash) == 32:
        return cached_hash == compute_file_hash_md5(path)
    return False


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
    from PIL import Image, ImageOps

    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass  # will fail at Image.open if not installed

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        orig_shorter = min(w, h)
        # Downscaling disabled — process and store at original resolution.
        # scale = min(1.0, max_dimension / max(w, h))
        # if scale < 1.0:
        #     img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img, orig_shorter
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        # OSError covers PIL UnidentifiedImageError, file-system failures, and
        # truncated-image issues that LOAD_TRUNCATED_IMAGES couldn't paper over.
        # SyntaxError is what PIL raises on malformed PNG/JPEG headers.
        # DecompressionBombError inherits from Exception (not OSError); huge
        # panoramas / RAW conversions over PIL's ~356MP threshold trigger it
        # and would otherwise crash stage_extract uncaught.
        global _unreadable_count
        _unreadable_count += 1
        logger.warning("Cannot open image %s: %s", path, exc)
        return None, 0
