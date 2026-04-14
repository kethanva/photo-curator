"""
Tests for src/selection.py — copy_to_output and _resize_and_save.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.selection import copy_to_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_jpeg(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path, "JPEG")
    return path


def _rec(path: str, **kwargs) -> dict:
    defaults = {
        "path": path,
        "aesthetic_score": 0.7,
        "smile_score": 0.8,
        "face_count": 1,
        "person_id": -1,
        "is_frequent": 0,
        "scene_tags": "",
        "cluster_id": 0,
        "lat": 37.77,
        "lon": -122.41,
        "timestamp": 1700000000.0,
        "camera_model": "TestCam",
        "blur_score": 200.0,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# copy_to_output tests
# ---------------------------------------------------------------------------

class TestCopyToOutput:
    def test_creates_output_dir(self, tmp_path: Path):
        out = tmp_path / "output"
        assert not out.exists()
        copy_to_output([], {}, str(out), generate_report=False)
        assert out.exists()

    def test_empty_selection_writes_nothing(self, tmp_path: Path):
        out = tmp_path / "output"
        copy_to_output([], {}, str(out), generate_report=False)
        files = list(out.iterdir())
        assert files == []

    def test_copies_file_without_resize(self, tmp_path: Path):
        src = tmp_path / "photo.jpg"
        _create_jpeg(src)
        out = tmp_path / "output"

        copy_to_output(
            [_rec(str(src))],
            {str(src): 0.8},
            str(out),
            resize=False,
            generate_report=False,
        )
        assert (out / "photo.jpg").exists()

    def test_resizes_file(self, tmp_path: Path):
        src = tmp_path / "large.jpg"
        _create_jpeg(src, size=(4000, 3000))
        out = tmp_path / "output"

        copy_to_output(
            [_rec(str(src))],
            {str(src): 0.8},
            str(out),
            resize=True,
            long_side=1000,
            generate_report=False,
        )
        result = out / "large.jpg"
        assert result.exists()
        with Image.open(result) as img:
            assert max(img.size) <= 1000

    def test_generates_report(self, tmp_path: Path):
        src = tmp_path / "photo.jpg"
        _create_jpeg(src)
        out = tmp_path / "output"

        copy_to_output(
            [_rec(str(src))],
            {str(src): 0.75},
            str(out),
            resize=False,
            generate_report=True,
            report_filename="output.json",
        )
        report_path = out / "output.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert len(report) == 1
        assert report[0]["score"] == pytest.approx(0.75, abs=0.001)

    def test_report_contains_expected_fields(self, tmp_path: Path):
        src = tmp_path / "photo.jpg"
        _create_jpeg(src)
        out = tmp_path / "output"

        copy_to_output(
            [_rec(str(src))],
            {str(src): 0.5},
            str(out),
            resize=False,
            generate_report=True,
        )
        report = json.loads((out / "output.json").read_text())
        entry = report[0]
        for field in ("filename", "original_path", "score", "aesthetic", "faces", "cluster"):
            assert field in entry

    def test_collision_renamed(self, tmp_path: Path):
        """Two source photos with the same stem but in different dirs get renamed."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        src1 = dir_a / "photo.jpg"
        src2 = dir_b / "photo.jpg"
        _create_jpeg(src1)
        _create_jpeg(src2)
        out = tmp_path / "output"

        copy_to_output(
            [_rec(str(src1)), _rec(str(src2))],
            {str(src1): 0.9, str(src2): 0.8},
            str(out),
            resize=False,
            generate_report=False,
        )
        files = {f.name for f in out.iterdir()}
        assert len(files) == 2  # both files present, one renamed

    def test_missing_source_skipped(self, tmp_path: Path):
        """A record pointing to a non-existent file should be skipped gracefully."""
        out = tmp_path / "output"
        copy_to_output(
            [_rec(str(tmp_path / "ghost.jpg"))],
            {},
            str(out),
            resize=True,
            generate_report=False,
        )
        # No files should be written
        assert list(out.iterdir()) == []
