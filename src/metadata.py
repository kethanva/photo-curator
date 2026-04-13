"""
EXIF metadata extraction: timestamp, GPS, camera model.
Adapted from vision-photo-clusterer with additional fields.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import piexif


def _rational_to_float(value) -> float:
    """Convert piexif rational (tuple or list) to float."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        denom = value[1]
        return float(value[0]) / float(denom) if denom else 0.0
    return float(value)


def _dms_to_decimal(dms, ref: bytes) -> float:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    try:
        d = _rational_to_float(dms[0])
        m = _rational_to_float(dms[1])
        s = _rational_to_float(dms[2])
        value = d + m / 60.0 + s / 3600.0
        if ref in (b"S", b"W"):
            value = -value
        return value
    except Exception:
        return 0.0


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
    except Exception:
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
        result["lat"] = _dms_to_decimal(lat_data, lat_ref)
        result["lon"] = _dms_to_decimal(lon_data, lon_ref)
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
