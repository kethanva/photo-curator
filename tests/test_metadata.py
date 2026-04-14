"""
Unit tests for src/metadata.py — EXIF parsing and GPS distance helpers.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import piexif
import pytest
from PIL import Image

from src.metadata import _dms_to_decimal, _rational_to_float, extract_exif, is_rare_location


# ---------------------------------------------------------------------------
# _rational_to_float
# ---------------------------------------------------------------------------

class TestRationalToFloat:
    def test_tuple_ratio(self):
        assert _rational_to_float((1, 2)) == pytest.approx(0.5)

    def test_whole_number(self):
        assert _rational_to_float((10, 1)) == pytest.approx(10.0)

    def test_zero_denominator_returns_zero(self):
        assert _rational_to_float((5, 0)) == pytest.approx(0.0)

    def test_list_ratio(self):
        assert _rational_to_float([3, 4]) == pytest.approx(0.75)

    def test_plain_float(self):
        assert _rational_to_float(3.14) == pytest.approx(3.14)

    def test_plain_int(self):
        assert _rational_to_float(7) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# _dms_to_decimal
# ---------------------------------------------------------------------------

class TestDmsToDecimal:
    def _deg(self, d: float, m: float, s: float):
        return [(int(d), 1), (int(m), 1), (int(s * 100), 100)]

    def test_north_positive(self):
        dms = self._deg(37, 46, 29.64)
        result = _dms_to_decimal(dms, b"N")
        assert result > 0

    def test_south_negative(self):
        dms = self._deg(33, 51, 21.0)
        result = _dms_to_decimal(dms, b"S")
        assert result < 0

    def test_west_negative(self):
        dms = self._deg(122, 25, 9.72)
        result = _dms_to_decimal(dms, b"W")
        assert result < 0

    def test_east_positive(self):
        dms = self._deg(0, 0, 0)
        result = _dms_to_decimal(dms, b"E")
        assert result == pytest.approx(0.0)

    def test_bad_input_returns_zero(self):
        result = _dms_to_decimal([], b"N")
        assert result == pytest.approx(0.0)

    def test_known_coordinate(self):
        """37° 0' 0" N = 37.0 decimal degrees."""
        dms = [(37, 1), (0, 1), (0, 1)]
        result = _dms_to_decimal(dms, b"N")
        assert result == pytest.approx(37.0, abs=0.01)


# ---------------------------------------------------------------------------
# extract_exif
# ---------------------------------------------------------------------------

def _write_jpeg_with_exif(path: Path, exif_bytes: bytes) -> None:
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(path, "JPEG", exif=exif_bytes)


class TestExtractExif:
    def test_missing_file_returns_defaults(self, tmp_path: Path):
        result = extract_exif(tmp_path / "missing.jpg")
        assert result["timestamp"] == pytest.approx(0.0)
        assert result["has_gps"] is False

    def test_jpeg_without_exif_returns_defaults(self, tmp_path: Path):
        p = tmp_path / "no_exif.jpg"
        Image.new("RGB", (50, 50)).save(p, "JPEG")
        result = extract_exif(p)
        assert result["timestamp"] == pytest.approx(0.0)
        assert result["camera_model"] == ""
        assert result["has_gps"] is False

    def test_camera_model_extracted(self, tmp_path: Path):
        p = tmp_path / "camera.jpg"
        exif_dict = {"0th": {piexif.ImageIFD.Model: b"iPhone 15"}}
        exif_bytes = piexif.dump(exif_dict)
        _write_jpeg_with_exif(p, exif_bytes)
        result = extract_exif(p)
        assert result["camera_model"] == "iPhone 15"

    def test_datetime_extracted(self, tmp_path: Path):
        p = tmp_path / "dated.jpg"
        # "2023:07:04 12:00:00"
        exif_dict = {
            "0th": {},
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2023:07:04 12:00:00"
            },
        }
        exif_bytes = piexif.dump(exif_dict)
        _write_jpeg_with_exif(p, exif_bytes)
        result = extract_exif(p)
        assert result["timestamp"] > 0

    def test_gps_extracted(self, tmp_path: Path):
        p = tmp_path / "gps.jpg"
        # San Francisco approx: 37.77° N, 122.41° W
        exif_dict = {
            "0th": {},
            "GPS": {
                piexif.GPSIFD.GPSLatitude:    [(37, 1), (46, 1), (12, 1)],
                piexif.GPSIFD.GPSLatitudeRef: b"N",
                piexif.GPSIFD.GPSLongitude:   [(122, 1), (24, 1), (36, 1)],
                piexif.GPSIFD.GPSLongitudeRef: b"W",
            },
        }
        exif_bytes = piexif.dump(exif_dict)
        _write_jpeg_with_exif(p, exif_bytes)
        result = extract_exif(p)
        assert result["has_gps"] is True
        assert result["lat"] == pytest.approx(37.77, abs=0.1)
        assert result["lon"] == pytest.approx(-122.41, abs=0.1)

    def test_result_keys_always_present(self, tmp_path: Path):
        p = tmp_path / "any.jpg"
        Image.new("RGB", (50, 50)).save(p, "JPEG")
        result = extract_exif(p)
        for key in ("timestamp", "lat", "lon", "camera_model", "has_gps"):
            assert key in result


# ---------------------------------------------------------------------------
# is_rare_location
# ---------------------------------------------------------------------------

class TestIsRareLocation:
    def test_no_gps_returns_false(self):
        assert is_rare_location(0.0, 0.0, home=(37.0, -122.0)) is False

    def test_no_home_returns_true_if_gps_present(self):
        assert is_rare_location(37.7, -122.4, home=None) is True

    def test_within_radius_returns_false(self):
        home = (37.7749, -122.4194)  # SF
        assert is_rare_location(37.7749, -122.4194, home=home, radius_km=1.0) is False

    def test_outside_radius_returns_true(self):
        home = (37.7749, -122.4194)
        # Tokyo is ~9000 km away
        assert is_rare_location(35.6762, 139.6503, home=home, radius_km=1.0) is True

    def test_nearby_returns_false(self):
        home = (37.7749, -122.4194)
        # ~100m away from home
        assert is_rare_location(37.7750, -122.4195, home=home, radius_km=1.0) is False
