"""
Unit tests for src/ingestion.py — folder scanning, hashing, and image loading.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from src.ingestion import (
    DEFAULT_EXTENSIONS,
    compute_file_hash,
    load_image_safe,
    scan_photos,
)


# ---------------------------------------------------------------------------
# scan_photos tests
# ---------------------------------------------------------------------------

class TestScanPhotos:
    def test_raises_on_missing_folder(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            scan_photos(str(tmp_path / "nonexistent"))

    def test_empty_folder_returns_empty(self, tmp_path: Path):
        result = scan_photos(str(tmp_path))
        assert result == []

    def test_finds_jpg_files(self, tmp_path: Path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        result = scan_photos(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "photo.jpg"

    def test_finds_multiple_extensions(self, tmp_path: Path):
        for name in ["a.jpg", "b.png", "c.jpeg"]:
            (tmp_path / name).write_bytes(b"fake")
        result = scan_photos(str(tmp_path))
        assert len(result) == 3

    def test_ignores_non_image_files(self, tmp_path: Path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_bytes(b"text")
        (tmp_path / "data.csv").write_bytes(b"data")
        result = scan_photos(str(tmp_path))
        assert len(result) == 1

    def test_recursion_into_subdirs(self, tmp_path: Path):
        sub = tmp_path / "2023"
        sub.mkdir()
        (tmp_path / "top.jpg").write_bytes(b"x")
        (sub / "nested.jpg").write_bytes(b"x")
        result = scan_photos(str(tmp_path))
        names = {p.name for p in result}
        assert "top.jpg" in names
        assert "nested.jpg" in names

    def test_result_is_sorted(self, tmp_path: Path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"x")
        result = scan_photos(str(tmp_path))
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_custom_extensions(self, tmp_path: Path):
        (tmp_path / "photo.tiff").write_bytes(b"x")
        (tmp_path / "photo.jpg").write_bytes(b"x")
        result = scan_photos(str(tmp_path), extensions={".tiff"})
        assert all(p.suffix == ".tiff" for p in result)

    def test_default_extensions_set(self):
        assert ".jpg" in DEFAULT_EXTENSIONS
        assert ".heic" in DEFAULT_EXTENSIONS
        assert ".png" in DEFAULT_EXTENSIONS


# ---------------------------------------------------------------------------
# compute_file_hash tests
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_returns_string(self, tmp_path: Path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"hello world")
        result = compute_file_hash(f)
        assert isinstance(result, str)

    def test_hex_string(self, tmp_path: Path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"data")
        result = compute_file_hash(f)
        int(result, 16)  # should not raise

    def test_correct_sha256(self, tmp_path: Path):
        content = b"photo data here"
        f = tmp_path / "img.jpg"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_hash(f) == expected

    def test_identical_files_same_hash(self, tmp_path: Path):
        content = b"same content"
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_files_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"content one")
        f2.write_bytes(b"content two")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_file_hash(f) == expected


# ---------------------------------------------------------------------------
# load_image_safe tests
# ---------------------------------------------------------------------------

class TestLoadImageSafe:
    """load_image_safe returns (img, orig_shorter_side) or (None, 0)."""

    def _create_jpg(self, tmp_path: Path, name: str = "img.jpg", size=(200, 200)) -> Path:
        p = tmp_path / name
        img = Image.new("RGB", size, color=(100, 150, 200))
        img.save(p, "JPEG")
        return p

    def test_returns_tuple(self, tmp_path: Path):
        p = self._create_jpg(tmp_path)
        result = load_image_safe(p, max_dimension=1024)
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_pil_image(self, tmp_path: Path):
        p = self._create_jpg(tmp_path)
        img, shorter = load_image_safe(p, max_dimension=1024)
        assert isinstance(img, Image.Image)
        assert shorter == 200

    def test_converts_to_rgb(self, tmp_path: Path):
        p = tmp_path / "gray.jpg"
        Image.new("L", (50, 50), color=128).save(p, "JPEG")
        img, _ = load_image_safe(p, max_dimension=1024)
        assert img is not None
        assert img.mode == "RGB"

    def test_loads_at_original_resolution(self, tmp_path: Path):
        """Downscaling is disabled — large images stay full-resolution."""
        p = self._create_jpg(tmp_path, size=(4000, 3000))
        img, shorter = load_image_safe(p, max_dimension=1024)
        assert img is not None
        assert img.size == (4000, 3000)
        assert shorter == 3000

    def test_small_image_not_upscaled(self, tmp_path: Path):
        p = self._create_jpg(tmp_path, size=(100, 100))
        img, shorter = load_image_safe(p, max_dimension=1024)
        assert img is not None
        assert img.size == (100, 100)
        assert shorter == 100

    def test_missing_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "nonexistent.jpg"
        img, shorter = load_image_safe(p, max_dimension=1024)
        assert img is None
        assert shorter == 0

    def test_corrupted_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not a jpeg at all !!!!")
        img, shorter = load_image_safe(p, max_dimension=1024)
        assert img is None
        assert shorter == 0

    def test_orig_shorter_side_for_landscape(self, tmp_path: Path):
        """2000x1000 landscape — shorter side is 1000."""
        p = self._create_jpg(tmp_path, size=(2000, 1000))
        img, shorter = load_image_safe(p, max_dimension=1000)
        assert img is not None
        assert shorter == 1000
