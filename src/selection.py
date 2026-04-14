"""
Selection engine: three-bucket diversity strategy + output resizing.

Budget allocation (configurable, defaults to 30/30/40):
  ┌─────────────────────┬───────────────────────────────────────────────────┐
  │ People bucket  (30%)│ Best N photos for each identified frequent person  │
  │ Location bucket(30%)│ Best M photos per GPS location cluster             │
  │ Aesthetic bucket(40%)│ Top overall scores regardless of subject         │
  └─────────────────────┴───────────────────────────────────────────────────┘

Output budget modes (config.yaml → selection.output_mode):
  percentage  — select output_percentage % of eligible photos (default 15 %)
  bytes       — fill up to max_output_bytes (1 GB default)
  Both modes enforce max_output_bytes as a hard upper limit.

Output resizing:
  Photos are resized to target_long_side (default 2 560 px) before writing.
  A 12 MP original (~4 MB JPEG) becomes ~1.5 MB, fitting ~2.5× more photos
  within the 1 GB budget while staying sharp on 2 K/4 K displays.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Resizing helper
# ---------------------------------------------------------------------------

def _resize_and_save(
    src: Path,
    dst: Path,
    long_side: int,
    quality: int,
) -> bool:
    """
    Resize image so longest side ≤ long_side, save as JPEG.
    Returns True on success.
    """
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > long_side:
                scale = long_side / max(w, h)
                img = img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )
            img.save(dst, "JPEG", quality=quality, optimize=True)
        return True
    except Exception as exc:
        print(f"  Warning: could not resize {src.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Three-bucket selection
# ---------------------------------------------------------------------------

def select_photos(
    records: List[dict],
    scores: Dict[str, float],
    max_bytes: int = 1_073_741_824,
    max_per_cluster: int = 10,
    max_per_person: int = 5,
    max_per_location: int = 15,
    people_fraction: float = 0.30,
    location_fraction: float = 0.30,
    aesthetic_fraction: float = 0.40,
    output_long_side: int = 2560,
    output_jpeg_quality: int = 92,
    output_mode: str = "bytes",
    output_percentage: float = 0.15,
) -> List[dict]:
    """
    Select photos using a three-bucket diversity strategy.

    Pre-filters duplicates, private photos, and quality failures.

    Budget modes
    ------------
    percentage  Stop once output_percentage * len(eligible) photos are selected.
    bytes       Stop once resized estimates reach max_output_bytes.
    Both modes enforce max_output_bytes as a hard cap.

    Returns:
        Ordered list of selected records (deduplicated across buckets).
    """
    candidates = [
        r for r in records
        if r.get("quality_pass", 1)
        and not r.get("is_duplicate", 0)
        and not r.get("is_private", 0)
    ]

    if not candidates:
        return []

    # Derive photo count target for percentage mode
    max_photos: Optional[int] = None
    if output_mode == "percentage":
        max_photos = max(1, int(len(candidates) * output_percentage))

    # ── Estimate output sizes after resizing ──────────────────────
    # Approximation: resized JPEG ≈ original_size * (long_side/max_orig_dim)^2 * 0.75
    def _est_size(rec: dict) -> int:
        orig = rec.get("file_size", 500_000)
        res = rec.get("resolution", 1)
        if res <= 0:
            res = 1
        max_dim = max(res * 1.5, 1)
        if max_dim <= output_long_side:
            return orig
        scale = output_long_side / max_dim
        return max(50_000, int(orig * scale * scale * 0.75))

    # Sort all candidates by score descending
    sorted_cands = sorted(candidates, key=lambda r: scores.get(r["path"], 0.0), reverse=True)
    selected_paths: Set[str] = set()
    selected: List[dict] = []
    used_bytes: int = 0

    def _add(rec: dict) -> bool:
        nonlocal used_bytes
        if rec["path"] in selected_paths:
            return False
        # Hard byte cap always applies
        est = _est_size(rec)
        if used_bytes + est > max_bytes:
            return False
        # Photo count cap applies in percentage mode
        if max_photos is not None and len(selected) >= max_photos:
            return False
        selected_paths.add(rec["path"])
        selected.append(rec)
        used_bytes += est
        return True

    def _budget_exhausted() -> bool:
        if max_photos is not None and len(selected) >= max_photos:
            return True
        return used_bytes >= max_bytes

    # Per-bucket byte budgets: used to distribute capacity across buckets.
    # In percentage mode the buckets still split the photo count proportionally.
    people_budget   = int(max_bytes * people_fraction)
    location_budget = int(max_bytes * location_fraction)

    # ── Bucket 1: People (30%) ────────────────────────────────────
    used_people = 0
    person_counts: Dict[int, int] = defaultdict(int)

    if max_photos is not None:
        people_photo_cap = max(1, int(max_photos * people_fraction))
    else:
        people_photo_cap = None

    for rec in sorted_cands:
        if _budget_exhausted():
            break
        pid = rec.get("person_id", -1)
        if pid < 0 or not rec.get("is_frequent", 0):
            continue
        if person_counts[pid] >= max_per_person:
            continue
        if people_photo_cap is not None and sum(person_counts.values()) >= people_photo_cap:
            continue
        est = _est_size(rec)
        if used_people + est > people_budget:
            continue
        if _add(rec):
            person_counts[pid] += 1
            used_people += est

    # ── Bucket 2: Location diversity (30%) ───────────────────────
    used_location = 0
    loc_counts: Dict[int, int] = defaultdict(int)

    if max_photos is not None:
        location_photo_cap = max(1, int(max_photos * location_fraction))
    else:
        location_photo_cap = None

    location_selected = 0
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        if rec["path"] in selected_paths:
            continue
        cid = rec.get("cluster_id", -1)
        if loc_counts[cid] >= max_per_location:
            continue
        if location_photo_cap is not None and location_selected >= location_photo_cap:
            continue
        est = _est_size(rec)
        if used_location + est > location_budget:
            continue
        if _add(rec):
            loc_counts[cid] += 1
            used_location += est
            location_selected += 1

    # ── Bucket 3: Aesthetic top picks (40%) ──────────────────────
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        if rec["path"] in selected_paths:
            continue
        _add(rec)

    return selected


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def copy_to_output(
    selected: List[dict],
    scores: Dict[str, float],
    output_dir: str,
    resize: bool = True,
    long_side: int = 2560,
    jpeg_quality: int = 92,
    generate_report: bool = True,
    report_filename: str = "output.json",
) -> None:
    """
    Write selected photos to the output directory.

    When resize=True, photos are resized to long_side and saved as JPEG,
    significantly reducing file sizes while preserving visual quality.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Clean previous output to prevent accumulating duplicates with _1 suffixes
    for old_file in out.glob("*"):
        if old_file.is_file() and old_file.name != ".gitkeep":
            try:
                old_file.unlink()
            except Exception:
                pass

    report: List[dict] = []
    copied = 0
    total_bytes = 0

    for rec in selected:
        src = Path(rec["path"])

        # Always output as .jpg when resizing
        stem = src.stem
        suffix = ".jpg" if resize else src.suffix
        dst = out / f"{stem}{suffix}"

        # Resolve collisions
        counter = 1
        while dst.exists():
            dst = out / f"{stem}__{src.parent.name}_{counter}{suffix}"
            counter += 1

        if resize:
            ok = _resize_and_save(src, dst, long_side, jpeg_quality)
        else:
            import shutil
            try:
                shutil.copy2(src, dst)
                ok = True
            except Exception as exc:
                print(f"  Warning: could not copy {src.name}: {exc}")
                ok = False

        if not ok:
            continue

        copied += 1
        total_bytes += dst.stat().st_size if dst.exists() else 0

        if generate_report:
            report.append(
                {
                    "filename": dst.name,
                    "original_path": str(src),
                    "score": round(scores.get(rec["path"], 0.0), 4),
                    "aesthetic": round(rec.get("aesthetic_score", 0.0), 3),
                    "sentiment": round(rec.get("smile_score", 0.5), 3),
                    "faces": rec.get("face_count", 0),
                    "person_id": rec.get("person_id", -1),
                    "is_frequent_person": bool(rec.get("is_frequent", 0)),
                    "scene_tags": rec.get("scene_tags", ""),
                    "cluster": rec.get("cluster_id", -1),
                    "lat": rec.get("lat", 0.0),
                    "lon": rec.get("lon", 0.0),
                    "timestamp": rec.get("timestamp", 0.0),
                    "camera": rec.get("camera_model", ""),
                    "blur": round(rec.get("blur_score", 0.0), 2),
                }
            )

    if generate_report and report:
        with open(out / report_filename, "w") as f:
            json.dump(report, f, indent=2)

    total_mb = total_bytes / 1_048_576
    print(f"  Written {copied} photos ({total_mb:.1f} MB) → {output_dir}")
