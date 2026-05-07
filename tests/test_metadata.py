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

from datetime import datetime

from src.metadata import (
    _dms_to_decimal,
    _rational_to_float,
    extract_exif,
    is_rare_location,
    parse_timestamp_from_filename,
)


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

    def test_bad_input_returns_nan(self):
        # NaN sentinel (not 0.0) so callers can distinguish a decode
        # failure from a real 0° coordinate and refuse to set has_gps.
        import math
        result = _dms_to_decimal([], b"N")
        assert math.isnan(result)

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


# ---------------------------------------------------------------------------
# parse_timestamp_from_filename
# ---------------------------------------------------------------------------


class TestParseTimestampFromFilename:
    """Filename → Unix timestamp recovery for files without EXIF."""

    def _expect(self, name: str, expected_dt: datetime) -> None:
        ts = parse_timestamp_from_filename(name)
        assert ts == pytest.approx(expected_dt.timestamp())

    def test_android_default(self):
        self._expect("20190518_152454.jpg", datetime(2019, 5, 18, 15, 24, 54))

    def test_img_prefixed(self):
        self._expect(
            "IMG_20190413_102752.jpg", datetime(2019, 4, 13, 10, 27, 52)
        )

    def test_pixel_with_millis_suffix(self):
        # PXL_20210501_123456789.jpg — millis run together with seconds.
        # We only require year/month/day/HH/MM/SS to land correctly.
        ts = parse_timestamp_from_filename("PXL_20210501_123456789.jpg")
        assert ts == pytest.approx(
            datetime(2021, 5, 1, 12, 34, 56).timestamp()
        )

    def test_screenshot_dash(self):
        self._expect(
            "Screenshot_20240101-103000.png",
            datetime(2024, 1, 1, 10, 30, 0),
        )

    def test_whatsapp_apple_dotted(self):
        self._expect(
            "2019-05-18 15.24.54.jpg", datetime(2019, 5, 18, 15, 24, 54)
        )

    def test_signal_dashed(self):
        self._expect(
            "Signal-2019-05-18-15-24-54.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_whatsapp_date_only_anchors_near_noon_with_sequence(self):
        # IMG-20190518-WA0001.jpg has no time — anchors at 12:00 with the
        # WA sequence injected as the seconds offset so successive photos
        # in a chain keep their export order inside the day bucket.
        self._expect(
            "IMG-20190518-WA0001.jpg", datetime(2019, 5, 18, 12, 0, 1)
        )

    def test_date_only_dashed(self):
        self._expect("2019-05-18.jpg", datetime(2019, 5, 18, 12, 0, 0))

    def test_date_only_compact(self):
        self._expect("20190518.jpg", datetime(2019, 5, 18, 12, 0, 0))

    @pytest.mark.parametrize("name", [
        "vacation.jpg",
        "IMG_1234.HEIC",
        "DSC00123.JPG",
        "photo.png",
    ])
    def test_unparseable_returns_zero(self, name: str):
        assert parse_timestamp_from_filename(name) == 0.0

    @pytest.mark.parametrize("name", [
        "IMG_20191332_120000.jpg",  # month=13 invalid
        "IMG_20190230_120000.jpg",  # Feb 30 invalid
        "IMG_18991201_120000.jpg",  # year 1899 < 1990 floor
    ])
    def test_invalid_dates_rejected(self, name: str):
        assert parse_timestamp_from_filename(name) == 0.0

    def test_invalid_time_falls_back_to_noon(self):
        # Date is valid, time is junk (hour=99). Function must keep the
        # date and anchor at 12:00 so the photo still lands in the right
        # day bucket rather than being discarded outright.
        self._expect(
            "IMG_20190413_990000.jpg", datetime(2019, 4, 13, 12, 0, 0)
        )

    def test_burst_pair_yields_distinct_timestamps(self):
        """The user-reported burst pair must parse to timestamps 3 seconds
        apart so the global temporal-proximity gate can fire."""
        a = parse_timestamp_from_filename("IMG_20190413_102752.jpg")
        b = parse_timestamp_from_filename("IMG_20190413_102755.jpg")
        assert a > 0 and b > 0
        assert b - a == pytest.approx(3.0)

    # ---------------------------------------------------------------
    # Compact 14-digit YYYYMMDDHHMMSS (separator-free)
    # ---------------------------------------------------------------

    def test_compact_full_timestamp(self):
        """Bare 14-digit timestamp with no separators."""
        self._expect(
            "20190518152454.jpg", datetime(2019, 5, 18, 15, 24, 54)
        )

    def test_compact_full_timestamp_with_prefix(self):
        """Compact timestamp with text prefix (no underscore between)."""
        self._expect(
            "IMG20190518152454.jpg", datetime(2019, 5, 18, 15, 24, 54)
        )

    def test_compact_full_timestamp_with_underscore_prefix(self):
        self._expect(
            "IMG_20190518152454.jpg", datetime(2019, 5, 18, 15, 24, 54)
        )

    def test_compact_full_timestamp_with_suffix(self):
        """Compact timestamp followed by a suffix (e.g. burst index)."""
        self._expect(
            "20190518152454_BURST001.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_compact_full_timestamp_with_millis(self):
        """Compact timestamp with trailing 3-digit millis."""
        self._expect(
            "20190518152454789.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_compact_invalid_hour_returns_zero(self):
        """A 14-digit run with an impossible hour (99) returns 0.0.
        The date regex can't recover the date because its trailing
        ``(?!\\d)`` lookahead refuses to terminate the day field when
        more digits follow — a filename this malformed has no
        trustworthy date anyway."""
        assert parse_timestamp_from_filename("20190413990000.jpg") == 0.0

    # ---------------------------------------------------------------
    # Dot-separated date / datetime
    # ---------------------------------------------------------------

    def test_dot_separated_date_only(self):
        self._expect("2019.05.18.jpg", datetime(2019, 5, 18, 12, 0, 0))

    def test_dot_separated_date_with_time(self):
        self._expect(
            "2019.05.18 15.24.54.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_dot_separated_date_with_dashed_time(self):
        self._expect(
            "2019.05.18-15-24-54.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    # ---------------------------------------------------------------
    # AM/PM markers (localised macOS screenshots, etc.)
    # ---------------------------------------------------------------

    def test_macos_screen_shot_pm(self):
        """Screen Shot 2024-01-15 at 10.30.45 PM.png → 22:30:45."""
        self._expect(
            "Screen Shot 2024-01-15 at 10.30.45 PM.png",
            datetime(2024, 1, 15, 22, 30, 45),
        )

    def test_macos_screen_shot_am(self):
        """Screen Shot 2024-01-15 at 10.30.45 AM.png → 10:30:45."""
        self._expect(
            "Screen Shot 2024-01-15 at 10.30.45 AM.png",
            datetime(2024, 1, 15, 10, 30, 45),
        )

    def test_meridiem_noon_edge_case(self):
        """12 PM is noon (12:00), 12 AM is midnight (00:00)."""
        self._expect(
            "Screen Shot 2024-01-15 at 12.00.00 PM.png",
            datetime(2024, 1, 15, 12, 0, 0),
        )
        self._expect(
            "Screen Shot 2024-01-15 at 12.30.45 AM.png",
            datetime(2024, 1, 15, 0, 30, 45),
        )

    def test_meridiem_lowercase(self):
        """Lowercase am/pm markers must work (filename case can vary)."""
        self._expect(
            "Screen Shot 2024-01-15 at 03.45.00 pm.png",
            datetime(2024, 1, 15, 15, 45, 0),
        )

    def test_meridiem_only_applies_to_12_hour_range(self):
        """A 24-hour value (e.g. 15) accidentally followed by 'AM' must
        not be downgraded — only 1–12 are meaningful for AM/PM."""
        self._expect(
            "2024-01-15 15.30.45 AM.png",
            datetime(2024, 1, 15, 15, 30, 45),
        )

    # ---------------------------------------------------------------
    # Misc. real-world patterns we should already cover
    # ---------------------------------------------------------------

    def test_telegram_pattern(self):
        self._expect(
            "photo_2019-05-18_15-24-54.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_underscored_full_datetime(self):
        self._expect(
            "2019_05_18_15_24_54.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_iso_8601_like_with_t_separator(self):
        self._expect(
            "2019-05-18T15:24:54.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    # ---------------------------------------------------------------
    # Compact 12-digit YYYYMMDDHHMM (no seconds)
    # ---------------------------------------------------------------

    def test_compact_no_seconds(self):
        """12-digit YYYYMMDDHHMM with seconds defaulting to 0."""
        self._expect(
            "201905181524.jpg", datetime(2019, 5, 18, 15, 24, 0)
        )

    def test_compact_no_seconds_with_prefix(self):
        self._expect(
            "IMG_201905181524.jpg", datetime(2019, 5, 18, 15, 24, 0)
        )

    def test_compact_no_seconds_does_not_steal_from_full_form(self):
        """A 14-digit run must be matched by the precise YYYYMMDDHHMMSS
        regex, not truncated by the 12-digit fallback."""
        self._expect(
            "20190518152454.jpg", datetime(2019, 5, 18, 15, 24, 54)
        )

    # ---------------------------------------------------------------
    # Single-digit hour (macOS Screen Shot at 3 PM, etc.)
    # ---------------------------------------------------------------

    def test_single_digit_hour_with_pm(self):
        """Localised macOS Screen Shot uses single-digit hour 1–9."""
        self._expect(
            "Screen Shot 2024-01-15 at 3.45.00 PM.png",
            datetime(2024, 1, 15, 15, 45, 0),
        )

    def test_single_digit_hour_with_am(self):
        self._expect(
            "Screen Shot 2024-01-15 at 9.05.30 AM.png",
            datetime(2024, 1, 15, 9, 5, 30),
        )

    def test_single_digit_hour_24h_no_meridiem(self):
        """Filenames with a single-digit hour and no AM/PM marker keep
        the parsed hour as-is (24-hour interpretation)."""
        self._expect(
            "2024-01-15 8.30.00.png",
            datetime(2024, 1, 15, 8, 30, 0),
        )

    # ---------------------------------------------------------------
    # Unix-epoch fallback (messaging-app exports)
    # ---------------------------------------------------------------

    def test_unix_ms_epoch_pure_numeric(self):
        """13-digit unix-ms stem (WhatsApp iOS) parses to its real instant.

        Privacy filter still flags this name as a messaging reshare —
        timestamp recovery and inclusion are independent concerns."""
        # 1568579433476 ms = 2019-09-15T17:50:33.476 UTC
        ts = parse_timestamp_from_filename("1568579433476.jpg")
        # Tolerate local-tz conversion; assert correct day at minimum.
        assert ts > 0
        assert datetime.fromtimestamp(ts).year == 2019
        assert datetime.fromtimestamp(ts).month == 9
        assert datetime.fromtimestamp(ts).day in (15, 16)  # tz drift

    def test_unix_seconds_epoch_pure_numeric(self):
        """10-digit unix-second stem parses correctly."""
        ts = parse_timestamp_from_filename("1568579433.jpg")
        assert ts == pytest.approx(1568579433.0)

    def test_unix_epoch_with_messaging_prefix(self):
        """WeChat-style ``mmexport`` prefix doesn't block epoch recovery."""
        a = parse_timestamp_from_filename("mmexport1568579433476.jpeg")
        b = parse_timestamp_from_filename("1568579433476.jpg")
        assert a > 0 and b > 0
        assert a == pytest.approx(b)

    def test_unix_epoch_with_dash_prefix(self):
        ts = parse_timestamp_from_filename("IMG-1568579433476.png")
        assert ts == pytest.approx(1568579433.476, rel=1e-6)

    def test_unix_epoch_with_sub_index_does_not_match(self):
        """A 13-digit run followed by ``-2`` is NOT a clean trailing
        epoch (the regex requires (?!\\d), but ``-`` is non-digit so the
        run terminates correctly). The trailing ``-2`` then doesn't
        re-trigger because it's only 1 digit."""
        ts = parse_timestamp_from_filename("1568579433476-2.jpg")
        assert ts > 0
        # And the same epoch as the un-suffixed form
        assert ts == pytest.approx(1568579433.476, rel=1e-6)

    def test_unix_epoch_rejected_when_starts_with_calendar_year(self):
        """A 13-digit run starting with 19xx or 20xx is ambiguous — could
        be epoch or partial-calendar — so we reject it rather than guess."""
        # 2019051812345 starts with "20" — refused.
        assert parse_timestamp_from_filename("xx2019051812345.jpg") == 0.0

    def test_unix_epoch_out_of_range_rejected(self):
        """Epochs outside [year 2000, year 2100] are refused."""
        # 9999999999 = year 2286, out of range
        assert parse_timestamp_from_filename("9999999999.jpg") == 0.0
        # 100000000 = year 1973, out of range AND only 9 digits → no match
        assert parse_timestamp_from_filename("100000000.jpg") == 0.0

    def test_calendar_date_takes_priority_over_epoch(self):
        """A filename with both a parseable date AND a trailing 13-digit
        sub-string must use the date — not silently fall to the epoch
        path. Construct: leading date + epoch-shaped trailer."""
        ts = parse_timestamp_from_filename(
            "IMG_20190518_152454_1234567890123.jpg"
        )
        # Result should be 2019-05-18 15:24:54, NOT some epoch translation.
        assert ts == pytest.approx(
            datetime(2019, 5, 18, 15, 24, 54).timestamp()
        )

    # ---------------------------------------------------------------
    # finditer: skip invalid leading match, find valid date later
    # ---------------------------------------------------------------

    def test_finditer_skips_leading_invalid_date(self):
        """Filename starts with an ID that looks date-shaped (month=13)
        followed by a real date. The parser must skip the invalid first
        match and recover the real date instead of returning 0.0."""
        self._expect(
            "20191318_20190518.jpg", datetime(2019, 5, 18, 12, 0, 0)
        )

    def test_finditer_skips_invalid_then_uses_valid_with_time(self):
        """Same as above but the valid date has a real time attached —
        time must be picked up alongside the recovered date."""
        self._expect(
            "20191318_20190518_152454.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )

    def test_finditer_skips_feb_30_to_next_match(self):
        """A leading Feb-30 (range-valid but calendar-invalid) followed
        by a real date should not abort Pass 3 — finditer should walk to
        the next match."""
        self._expect(
            "20190230_20190518.jpg", datetime(2019, 5, 18, 12, 0, 0)
        )

    # ---------------------------------------------------------------
    # Seconds-optional time formats (HH:MM / HHMM)
    # ---------------------------------------------------------------

    def test_separated_time_no_seconds_colon(self):
        """``YYYY-MM-DD HH:MM.ext`` (no seconds) keeps minute precision
        instead of falling back to noon."""
        self._expect(
            "2019-05-18 15:24.jpg", datetime(2019, 5, 18, 15, 24, 0)
        )

    def test_separated_time_no_seconds_dot(self):
        self._expect(
            "2019-05-18 15.24.jpg", datetime(2019, 5, 18, 15, 24, 0)
        )

    def test_compact_hhmm_with_underscore_separator(self):
        """``YYYYMMDD_HHMM.ext`` (date and time both compact, joined by
        an underscore) parses to minute-aligned with seconds=0."""
        self._expect(
            "20190518_1524.jpg", datetime(2019, 5, 18, 15, 24, 0)
        )

    def test_iso_t_separated_time_no_seconds(self):
        self._expect(
            "2019-05-18T15:24.jpg", datetime(2019, 5, 18, 15, 24, 0)
        )

    def test_no_secs_regex_does_not_steal_three_digit_burst(self):
        """``IMG_20190518_007.jpg`` is a burst index, not 0:07 — the
        seconds-optional regex must refuse the 3-digit run so sequence
        injection can claim it instead."""
        self._expect(
            "IMG_20190518_007.jpg", datetime(2019, 5, 18, 12, 0, 7)
        )

    # ---------------------------------------------------------------
    # Sequence-number injection for date-only filenames
    # ---------------------------------------------------------------

    def test_whatsapp_sequence_orders_chain(self):
        """Successive WhatsApp WA-numbered photos must produce
        timestamps separated by exactly the sequence delta so the
        downstream temporal-proximity gate can fire on the chain."""
        a = parse_timestamp_from_filename("IMG-20190518-WA0001.jpg")
        b = parse_timestamp_from_filename("IMG-20190518-WA0002.jpg")
        c = parse_timestamp_from_filename("IMG-20190518-WA0010.jpg")
        assert a > 0 and b > 0 and c > 0
        assert b - a == pytest.approx(1.0)
        assert c - a == pytest.approx(9.0)

    def test_copy_suffix_sequence_injects(self):
        """Generic ``(N)`` copy-suffix lands as seconds offset."""
        self._expect(
            "2019-05-18 (3).jpg", datetime(2019, 5, 18, 12, 0, 3)
        )

    def test_underscore_burst_sequence_injects(self):
        self._expect(
            "2019-05-18_007.jpg", datetime(2019, 5, 18, 12, 0, 7)
        )

    def test_dash_burst_sequence_injects(self):
        self._expect(
            "2019-05-18-15.jpg", datetime(2019, 5, 18, 12, 0, 15)
        )

    def test_sequence_too_large_falls_back_to_noon(self):
        """A sequence that doesn't fit into the seconds field (>59) must
        not wrap or modulo — just plain noon, preserving day ordering."""
        self._expect(
            "IMG-20190518-WA9999.jpg", datetime(2019, 5, 18, 12, 0, 0)
        )

    def test_wa_zero_sequence_keeps_noon(self):
        """``WA0000`` should produce 12:00:00 (seq=0 is the same as no
        injection — also a sanity check that WA prefix doesn't collide
        with the existing date-only-anchors-at-noon contract)."""
        self._expect(
            "IMG-20190518-WA0000.jpg", datetime(2019, 5, 18, 12, 0, 0)
        )

    def test_dimension_token_does_not_inject(self):
        """A trailing ``_4032x3024`` dimensions token must not be misread
        as a sequence number — the ``x`` breaks the sequence regex's
        lookahead, so the photo lands at plain noon."""
        self._expect(
            "IMG_20190518_4032x3024.jpg",
            datetime(2019, 5, 18, 12, 0, 0),
        )

    def test_sequence_does_not_overrule_real_time(self):
        """When a real time is present, sequence injection must not run —
        seconds carry the real seconds, not the burst index."""
        # 15:24:54 with a trailing _001 (would be seq=1 if injection ran)
        self._expect(
            "IMG_20190518_152454_001.jpg",
            datetime(2019, 5, 18, 15, 24, 54),
        )
