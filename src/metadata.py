"""
EXIF metadata extraction: timestamp, GPS, camera model.
Adapted from vision-photo-clusterer with additional fields.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import piexif

logger = logging.getLogger(__name__)


# Sentinel returned by ``_dms_to_decimal`` when conversion fails — distinct
# from a real 0.0° value so callers can tell "decode failure" from "Equator/
# Greenwich". ``extract_exif`` checks for this and refuses to set has_gps.
_DMS_DECODE_FAILED = float("nan")


# ---------------------------------------------------------------------------
# Filename-based timestamp recovery
# ---------------------------------------------------------------------------
#
# Used as a fallback when EXIF DateTimeOriginal is missing — common for
# WhatsApp transfers, screenshots, and exports that strip metadata. A
# substantial share of the diversity / dedup logic depends on a usable
# timestamp, so recovering it from the filename keeps day/hour caps and
# event clustering working.

# Compact 14-digit datetime: YYYYMMDDHHMMSS, optionally followed by a
# 3-digit millisecond tail. Catches separator-free names like:
#     20190518152454.jpg                    raw compact
#     IMG_20190518152454.jpg                prefixed compact
#     IMG20190518152454_001.jpg             prefixed + suffixed
#     20190518152454789.mp4                 with millis
# This is checked BEFORE _DATE_RE because the date-only regex's
# trailing (?!\d) lookahead would refuse the match (the day is followed
# by another digit), causing the whole filename to fall through.
_FULL_TS_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})(?:\d{3})?(?!\d)"
)

# Compact 12-digit datetime: YYYYMMDDHHMM (no seconds). Less common than
# the 14-digit form but produced by some camera tools and manual exports
# (e.g. ``IMG_201905181524.jpg``). Tried after _FULL_TS_RE so the more
# precise match wins when both could apply. Seconds default to 0.
_COMPACT_NO_SECS_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?!\d)"
)

# Date core: YYYYMMDD optionally separated by - _ or . (year 1990–2099).
# The ``.`` separator covers patterns like ``2019.05.18.jpg`` and
# ``2019.05.18 15.24.54.jpg`` produced by some Windows camera tools and
# manual exports.
_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})[-_.]?(?P<month>\d{2})[-_.]?(?P<day>\d{2})(?!\d)"
)

# Time after the date: tolerates up to a short non-digit gap (e.g. " at "
# in "WhatsApp Image 2024-01-01 at 10.00.00") then HHMMSS with any of
# the common separators (- _ : . space). Hour is 1–2 digits so localised
# macOS Screen Shot names with single-digit hours work — ``Screen Shot
# 2024-01-15 at 3.45.00 PM.png``. Optional trailing millis (\d{3}) are
# accepted but ignored — Pixel cameras append them directly to the
# seconds field (PXL_20210501_123456789.jpg). An optional AM/PM marker
# at the tail handles the localised macOS Screen Shot pattern so the
# hour bucket lands in the correct half of the day.
_TIME_AFTER_RE = re.compile(
    r"\D{0,6}(?P<hour>\d{1,2})[-_:.\s]?(?P<minute>\d{2})[-_:.\s]?(?P<second>\d{2})"
    r"(?:\d{3})?(?!\d)(?:\s*(?P<meridiem>[AaPp][Mm]))?"
)

# Seconds-optional time after the date. Used as a fallback when the
# with-seconds regex fails so ``2019-05-18 15:24.jpg`` and
# ``20190518_1524.jpg`` keep their hour/minute precision instead of
# falling back to noon. Two carefully constrained branches:
#
#   - branch a: compact ``HHMM`` (exactly 4 digits, no internal
#     separator). Anchored to a separator-only lead-in so messaging
#     sequence numbers like ``WA0001`` (which sit after letters, not a
#     separator alone) cannot be misread as a time.
#   - branch b: separated ``H[:.]MM`` / ``HH[:.]MM`` with an explicit
#     internal separator. The internal separator is what disambiguates
#     time from a 3-digit burst index like ``_007``.
#
# A trailing ``(?!\d)`` guard prevents partial matches against longer
# digit runs (e.g. refuses to consume just ``9900`` of a malformed
# 6-digit time like ``_990000``).
_TIME_NO_SECS_AFTER_RE = re.compile(
    r"[-_:.\sT]+(?:"
    r"(?P<hour_a>\d{2})(?P<minute_a>\d{2})"
    r"|"
    r"(?P<hour_b>\d{1,2})[-_:.](?P<minute_b>\d{2})"
    r")(?!\d)"
)

# Trailing burst / copy index. Used to spread out the seconds of
# date-only filenames so exported chains stay in chronological order
# inside the same day. Three forms in priority order:
#
#   - ``WA\d+`` (WhatsApp export: ``IMG-20190518-WA0001.jpg``)
#   - ``(\d+)`` (copy suffix: ``photo (3).jpg``)
#   - ``[_-]\d+`` at end-of-token (generic burst: ``photo_007.jpg``)
#
# Capped at 4 digits to avoid swallowing dimension tokens like
# ``_4032x3024``. Only injected into the seconds field when the value
# fits in 0–59 — larger sequences silently fall back to plain noon
# rather than corrupting day ordering with a wrap.
_SEQUENCE_RES = (
    re.compile(r"WA(?P<seq>\d{1,4})", re.IGNORECASE),
    re.compile(r"\((?P<seq>\d{1,4})\)"),
    re.compile(r"[_-](?P<seq>\d{1,4})(?=\D|$)"),
)

# Unix-epoch fallback: 10-digit seconds or 13-digit milliseconds with no
# adjacent digits. Catches messaging-app exports that encode the capture
# time as a raw epoch in the filename:
#     1568579433476.jpg               WhatsApp iOS (13-digit ms)
#     1568579433.jpg                  10-digit seconds
#     mmexport1568579433476.jpeg      WeChat export (with prefix)
#     IMG-1568579433476.png           Generic messaging variants
# Independent of the privacy filter — privacy decides INCLUSION (these
# are usually reshared content), this regex decides whether the photo
# has a usable timestamp for clustering / day-cap accounting in the
# stages that run before the privacy gate.
_UNIX_EPOCH_RE = re.compile(r"(?<!\d)(?P<digits>\d{13}|\d{10})(?!\d)")

# Plausible epoch range — year 2000 to year 2100, in seconds.
_EPOCH_MIN_S = 946_684_800             # 2000-01-01T00:00:00Z
_EPOCH_MAX_S = 4_102_444_800           # 2100-01-01T00:00:00Z


def _adjust_for_meridiem(hour: int, meridiem: Optional[str]) -> int:
    """Convert a 12-hour clock value to 24-hour using an AM/PM marker.

    Returns the input unchanged when no marker is present or when the
    hour is outside the 1–12 range that AM/PM is meaningful for.
    """
    if not meridiem or not (1 <= hour <= 12):
        return hour
    upper = meridiem.upper()
    if upper == "PM" and hour < 12:
        return hour + 12
    if upper == "AM" and hour == 12:
        return 0
    return hour


def _build_calendar_timestamp(
    year: int, month: int, day: int, hour: int, minute: int, second: int,
) -> float:
    """Validate a calendar tuple and convert to a Unix timestamp.

    Returns 0.0 on any out-of-range or impossible value (e.g. Feb 30,
    hour 99). Centralised so each pass shares the same validation.
    """
    if not (1990 <= year <= 2100):
        return 0.0
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return 0.0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return 0.0
    try:
        return datetime(year, month, day, hour, minute, second).timestamp()
    except (ValueError, OverflowError):
        return 0.0


def _parse_time_after(rest: str) -> Optional[Tuple[int, int, int]]:
    """Parse ``(hour, minute, second)`` from text following a date match.

    Tries the with-seconds regex first because it carries strictly more
    information (richer lead-in, AM/PM marker, optional millis) and then
    falls back to the seconds-optional regex only when the first form
    can't make a valid match. Returns ``None`` when neither regex
    produces an in-range time.
    """
    m = _TIME_AFTER_RE.match(rest)
    if m:
        h = _adjust_for_meridiem(int(m.group("hour")), m.group("meridiem"))
        mi = int(m.group("minute"))
        s = int(m.group("second"))
        if 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= s <= 59:
            return (h, mi, s)
    m = _TIME_NO_SECS_AFTER_RE.match(rest)
    if m:
        hour_str = m.group("hour_a") or m.group("hour_b")
        minute_str = m.group("minute_a") or m.group("minute_b")
        h = int(hour_str)
        mi = int(minute_str)
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return (h, mi, 0)
    return None


def _extract_sequence(rest: str) -> Optional[int]:
    """Find a trailing burst / copy index suitable for seconds-injection.

    Returns the numeric value when one of the recognised patterns matches
    AND the value is within the seconds field range (0–59). Returns
    ``None`` when no pattern matches or the index is too large to safely
    encode — caller anchors at noon with second=0 in that case rather
    than wrapping and corrupting day ordering.
    """
    for pattern in _SEQUENCE_RES:
        m = pattern.search(rest)
        if m:
            seq = int(m.group("seq"))
            if 0 <= seq <= 59:
                return seq
            return None
    return None


def _try_unix_epoch(name: str) -> float:
    """Parse a trailing 10-digit (s) or 13-digit (ms) unix epoch.

    Restricted to digit runs starting with ``1`` because realistic epochs
    in the 2001–2033 window all do; runs starting with ``19`` or ``20``
    are far more likely to be calendar-style ``YYYYMMDD…`` prefixes than
    epoch values, so we refuse them rather than guess. Range-checked to
    the year 2000–2100 epoch window after that.
    """
    m = _UNIX_EPOCH_RE.search(name)
    if not m:
        return 0.0
    digits = m.group("digits")
    if not digits.startswith("1"):
        return 0.0
    raw = int(digits)
    ts = raw / 1000.0 if len(digits) == 13 else float(raw)
    if not (_EPOCH_MIN_S <= ts <= _EPOCH_MAX_S):
        return 0.0
    try:
        datetime.fromtimestamp(ts)
    except (ValueError, OSError, OverflowError):
        return 0.0
    return ts


def parse_timestamp_from_filename(path: Union[str, Path]) -> float:
    """
    Recover a Unix timestamp from common camera/messaging filename patterns.

    Returns 0.0 when no plausible date can be parsed. Time is best-effort —
    if only a date is present, the timestamp anchors at 12:00 local so the
    photo lands in the correct day bucket without forcing it to midnight
    (which would falsely group it with the previous evening's photos).

    Pass order (first plausible match wins). Every calendar pass uses
    ``finditer`` so an unrelated digit run that happens to look date-like
    (e.g. an ID with month=13) is skipped rather than aborting the pass.

    1. Compact 14-digit ``YYYYMMDDHHMMSS`` (with optional 3-digit millis).
       Required as a separate pass because the date-only regex's trailing
       ``(?!\\d)`` lookahead refuses to terminate the day field when the
       hour digits are pressed up against it.
    2. Compact 12-digit ``YYYYMMDDHHMM`` (no seconds). Same reason; rare
       but real.
    3. Date with optional separated time:
       - First try the with-seconds regex (richer lead-in handles ``at``
         word gaps and AM/PM markers).
       - Then a seconds-optional fallback for ``YYYY-MM-DD HH:MM`` and
         ``YYYYMMDD_HHMM`` style filenames that drop the seconds field.
       - When neither time form matches, anchor at noon and inject a
         trailing burst / copy index (``WA0001``, ``(1)``, ``_007``) as
         the seconds field so chains keep their relative order inside
         the day bucket.
    4. Unix-epoch fallback (10-digit seconds, 13-digit milliseconds) for
       messaging-app exports like ``1568579433476.jpg``. Conservatively
       gated to digit runs starting with ``1`` so we don't collide with
       calendar 19xx/20xx prefixes.

    Recognised forms (non-exhaustive):
      20190518_152454.jpg                 Android default
      IMG_20190223_163622.jpg             prefixed (IMG_, VID_, PXL_, …)
      PXL_20210501_123456789.jpg          Pixel camera (millis suffix)
      20190518152454.jpg                  compact YYYYMMDDHHMMSS
      IMG20190518152454.jpg               compact, no separator
      IMG_201905181524.jpg                compact YYYYMMDDHHMM (no secs)
      Screenshot_20240101-103000.png      Android screenshot
      Screen Shot 2024-01-15 at 10.30.45 AM.png   macOS screenshot
      Screen Shot 2024-01-15 at 3.45.00 PM.png    macOS, single-digit hour
      2019-05-18 15.24.54.jpg             WhatsApp / Apple style
      Signal-2019-05-18-15-24-54.jpg      Signal
      photo_2019-05-18_15-24-54.jpg       Telegram
      IMG-20190518-WA0001.jpg             WhatsApp (date only)
      2019.05.18 15.24.54.jpg             dot-separated date + time
      2019.05.18.jpg                      dot-separated date only
      20190518.jpg                        compact date only
      2019-05-18.jpg                      dashed date only
      1568579433476.jpg                   WhatsApp iOS unix-ms epoch
      1568579433.jpg                      unix-seconds epoch
      mmexport1568579433476.jpeg          WeChat export (prefixed epoch)
    """
    name = Path(path).stem

    # All calendar passes use ``finditer`` rather than ``search`` so a
    # leading invalid match (e.g. a non-date ID like ``20191318_…`` whose
    # month is 13) doesn't poison the rest of the string — the parser
    # gracefully steps past the bad match and tries the next plausible
    # candidate instead of giving up on that pass.

    # Pass 1 — compact YYYYMMDDHHMMSS (with optional 3-digit millis tail).
    for full in _FULL_TS_RE.finditer(name):
        ts = _build_calendar_timestamp(
            int(full.group("year")),
            int(full.group("month")),
            int(full.group("day")),
            int(full.group("hour")),
            int(full.group("minute")),
            int(full.group("second")),
        )
        if ts > 0:
            return ts

    # Pass 2 — compact YYYYMMDDHHMM (no seconds).
    for no_sec in _COMPACT_NO_SECS_RE.finditer(name):
        ts = _build_calendar_timestamp(
            int(no_sec.group("year")),
            int(no_sec.group("month")),
            int(no_sec.group("day")),
            int(no_sec.group("hour")),
            int(no_sec.group("minute")),
            0,
        )
        if ts > 0:
            return ts

    # Pass 3 — date with optional separated time. Two-stage time parse
    # (with-secs then no-secs) and a sequence-injection fallback so
    # date-only burst chains keep their export order inside the day.
    for m in _DATE_RE.finditer(name):
        year = int(m.group("year"))
        month = int(m.group("month"))
        day = int(m.group("day"))
        if not (1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
            continue
        rest = name[m.end():]
        time_tuple = _parse_time_after(rest)
        if time_tuple is not None:
            h, mi, s = time_tuple
            ts = _build_calendar_timestamp(year, month, day, h, mi, s)
            if ts > 0:
                return ts
        # No usable time. Anchor at noon and inject any trailing burst
        # index as the seconds offset so successive photos in a chain
        # (WA0001/WA0002, photo (1)/photo (2), …) keep their relative
        # order inside the day bucket.
        seq = _extract_sequence(rest)
        ts = _build_calendar_timestamp(
            year, month, day, 12, 0, seq if seq is not None else 0
        )
        if ts > 0:
            return ts

    # Pass 4 — unix-epoch fallback (messaging-app exports).
    return _try_unix_epoch(name)


def _rational_to_float(value) -> float:
    """Convert piexif rational (tuple or list) to float."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        denom = value[1]
        return float(value[0]) / float(denom) if denom else 0.0
    return float(value)


def _dms_to_decimal(dms, ref: bytes) -> float:
    """Convert degrees/minutes/seconds tuple to decimal degrees.

    Returns ``_DMS_DECODE_FAILED`` (NaN) on a malformed EXIF GPS block so
    the caller can distinguish "decode failed" from a real 0° coordinate
    and refuse to set has_gps. Returning 0.0 silently was sending photos
    to Null Island for clustering and rare-location checks.
    """
    try:
        d = _rational_to_float(dms[0])
        m = _rational_to_float(dms[1])
        s = _rational_to_float(dms[2])
        value = d + m / 60.0 + s / 3600.0
        if ref in (b"S", b"W"):
            value = -value
        return value
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        logger.debug("GPS DMS decode failed (ref=%r): %s", ref, exc)
        return _DMS_DECODE_FAILED


def extract_exif(path: Path) -> dict:
    """
    Extract EXIF data from image.

    Returns dict with keys:
      timestamp (float): Unix timestamp, 0.0 if missing
      lat (float): GPS latitude, 0.0 if missing
      lon (float): GPS longitude, 0.0 if missing
      camera_model (str): Camera/device model, '' if missing
      has_gps (bool): True if real GPS data present
    """
    result = {
        "timestamp": 0.0,
        "lat": 0.0,
        "lon": 0.0,
        "camera_model": "",
        "has_gps": False,
    }

    try:
        exif = piexif.load(str(path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Bad/missing EXIF block: fall through with the zeroed default. Log
        # so an operator can investigate why a photo lost its timestamp —
        # silent zero-timestamps collapse temporal diversity and corrupt
        # the per-day/per-hour caps for the entire run.
        logger.warning("extract_exif failed for %s: %s", path, exc)
        return result

    # Timestamp
    ifd0 = exif.get("0th", {})
    exif_ifd = exif.get("Exif", {})

    dt_bytes = (
        exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
        or ifd0.get(piexif.ImageIFD.DateTime)
    )
    if dt_bytes:
        try:
            dt_str = dt_bytes.decode("ascii", errors="ignore").strip()
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            result["timestamp"] = dt.timestamp()
        except Exception:
            pass

    # Filename fallback — keeps temporal diversity & event clustering working
    # for files exported without EXIF (WhatsApp, screenshots, manual exports).
    if result["timestamp"] <= 0:
        result["timestamp"] = parse_timestamp_from_filename(path)

    # Camera model
    model_bytes = ifd0.get(piexif.ImageIFD.Model, b"")
    if model_bytes:
        result["camera_model"] = model_bytes.decode("utf-8", errors="ignore").strip()

    # GPS
    gps = exif.get("GPS", {})
    lat_data = gps.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_data = gps.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)

    if lat_data and lat_ref and lon_data and lon_ref:
        import math
        lat_val = _dms_to_decimal(lat_data, lat_ref)
        lon_val = _dms_to_decimal(lon_data, lon_ref)
        # Only mark has_gps when BOTH conversions succeeded. A NaN from
        # _dms_to_decimal means a malformed rational — treating that as
        # (0, 0) silently put corrupted photos at Null Island and made
        # them cluster with each other.
        if not (math.isnan(lat_val) or math.isnan(lon_val)):
            result["lat"] = lat_val
            result["lon"] = lon_val
            result["has_gps"] = True

    return result


def is_rare_location(lat: float, lon: float, home: Optional[Tuple[float, float]],
                     radius_km: float = 0.5) -> bool:
    """Return True if GPS coords are outside the home radius."""
    if not lat and not lon:
        return False
    if home is None:
        return True
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1 = radians(home[0]), radians(home[1])
    lat2, lon2 = radians(lat), radians(lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dist = R * 2 * atan2(sqrt(a), sqrt(1 - a))
    return dist > radius_km
