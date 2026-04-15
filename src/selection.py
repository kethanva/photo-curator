"""
Selection engine: dynamic bucket diversity strategy + output resizing.

Budget allocation is configured via ``selection.buckets`` in config.yaml.
Built-in bucket types:
  people    — Best N photos for each identified frequent person
  location  — Best M photos per GPS location cluster
  aesthetic — Catch-all: top overall scores (must be last)

Any other bucket name is a **CLIP subject bucket** — it uses built-in presets
from subject_priority.py to match photos by visual content.  Adding a new
subject bucket requires only a one-line config change.

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


# Built-in bucket names with special selection logic
_BUILTIN_BUCKETS = {"people", "location", "aesthetic"}


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
# Dynamic bucket selection
# ---------------------------------------------------------------------------

def select_photos(
    records: List[dict],
    scores: Dict[str, float],
    max_bytes: int = 1_073_741_824,
    max_per_cluster_pct: float = 0.20,
    max_per_person_pct: float = 0.10,
    max_per_location_pct: float = 0.30,
    buckets: Optional[Dict[str, float]] = None,
    subject_scores: Optional[Dict[str, Dict[str, float]]] = None,
    output_long_side: int = 2560,
    output_jpeg_quality: int = 92,
    output_mode: str = "bytes",
    output_percentage: float = 0.15,
    total_photos: int = 0,
) -> List[dict]:
    """
    Select photos using a dynamic bucket diversity strategy.

    Pre-filters duplicates, private photos, and quality failures.

    Args:
        buckets:         ordered {name: fraction} — fractions should sum to 1.0.
                         Defaults to {"people": 0.30, "location": 0.30, "aesthetic": 0.40}.
        subject_scores:  {bucket_name: {path: similarity}} for CLIP subject buckets.

    Budget modes
    ------------
    percentage  Stop once output_percentage * total_photos photos are selected.
    bytes       Stop once resized estimates reach max_output_bytes.
    Both modes enforce max_output_bytes as a hard cap.

    Returns:
        Ordered list of selected records (deduplicated across buckets).
    """
    if buckets is None:
        buckets = {"people": 0.30, "location": 0.30, "aesthetic": 0.40}
    if subject_scores is None:
        subject_scores = {}

    # Normalise bucket fractions to sum to 1.0
    total_frac = sum(buckets.values())
    if total_frac > 0 and abs(total_frac - 1.0) > 1e-6:
        buckets = {k: v / total_frac for k, v in buckets.items()}

    candidates = [
        r for r in records
        if r.get("quality_pass", 1)
        and not r.get("is_duplicate", 0)
        and not r.get("is_private", 0)
    ]

    if not candidates:
        return []

    # Derive photo count target for percentage mode.
    max_photos: Optional[int] = None
    if output_mode == "percentage":
        base = total_photos if total_photos > 0 else len(candidates)
        target = max(1, int(base * output_percentage))
        max_photos = min(target, len(candidates))

    # ── Derive per-subject caps from the output target ────────────
    cap_base = max_photos if max_photos is not None else len(candidates)
    max_per_person_n   = max(1, int(cap_base * max_per_person_pct))
    max_per_location_n = max(1, int(cap_base * max_per_location_pct))

    # ── Estimate output sizes after resizing ──────────────────────
    def _est_size(rec: dict) -> int:
        orig = rec.get("file_size", 2_000_000)
        return min(orig, 5_000_000)

    # Sort all candidates by score descending
    sorted_cands = sorted(
        candidates, key=lambda r: scores.get(r["path"], 0.0), reverse=True
    )
    selected_paths: Set[str] = set()
    selected: List[dict] = []
    used_bytes: int = 0

    def _add(rec: dict) -> bool:
        nonlocal used_bytes
        if rec["path"] in selected_paths:
            return False
        est = _est_size(rec)
        if used_bytes + est > max_bytes:
            return False
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

    # Shared counters — enforced across all buckets
    person_counts: Dict[int, int] = defaultdict(int)
    loc_counts: Dict[int, int] = defaultdict(int)

    # ── Process each bucket in config order ──────────────────────
    # "aesthetic" is always processed last as the catch-all.
    bucket_order = [
        (name, frac) for name, frac in buckets.items() if name != "aesthetic"
    ]
    aesthetic_fraction = buckets.get("aesthetic", 0.0)

    for bucket_name, fraction in bucket_order:
        if _budget_exhausted():
            break

        bucket_byte_budget = int(max_bytes * fraction)
        bucket_photo_cap = (
            max(1, int(max_photos * fraction)) if max_photos is not None else None
        )

        if bucket_name == "people":
            _fill_people_bucket(
                sorted_cands, _add, _budget_exhausted, _est_size,
                person_counts, max_per_person_n,
                bucket_byte_budget, bucket_photo_cap,
            )

        elif bucket_name == "location":
            _fill_location_bucket(
                sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
                person_counts, max_per_person_n,
                loc_counts, max_per_location_n,
                bucket_byte_budget, bucket_photo_cap,
            )

        else:
            # CLIP subject bucket
            subj_scores = subject_scores.get(bucket_name, {})
            _fill_subject_bucket(
                sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
                person_counts, max_per_person_n,
                subj_scores, bucket_byte_budget, bucket_photo_cap,
            )

    # ── Aesthetic catch-all (always last) ────────────────────────
    if aesthetic_fraction > 0 and not _budget_exhausted():
        _fill_aesthetic_bucket(
            sorted_cands, selected_paths, _add, _budget_exhausted,
            person_counts, max_per_person_n,
        )

    return selected


# ---------------------------------------------------------------------------
# Bucket fill helpers
# ---------------------------------------------------------------------------

def _fill_people_bucket(
    sorted_cands, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    byte_budget, photo_cap,
):
    used = 0
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        pid = rec.get("person_id", -1)
        if pid < 0 or not rec.get("is_frequent", 0):
            continue
        if person_counts[pid] >= max_per_person_n:
            continue
        if photo_cap is not None and sum(person_counts.values()) >= photo_cap:
            continue
        est = _est_size(rec)
        if used + est > byte_budget:
            continue
        if _add(rec):
            person_counts[pid] += 1
            used += est


def _fill_location_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    loc_counts, max_per_location_n,
    byte_budget, photo_cap,
):
    used = 0
    bucket_selected = 0
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        if rec["path"] in selected_paths:
            continue
        cid = rec.get("cluster_id", -1)
        if loc_counts[cid] >= max_per_location_n:
            continue
        pid = rec.get("person_id", -1)
        if pid >= 0 and person_counts[pid] >= max_per_person_n:
            continue
        if photo_cap is not None and bucket_selected >= photo_cap:
            continue
        est = _est_size(rec)
        if used + est > byte_budget:
            continue
        if _add(rec):
            loc_counts[cid] += 1
            used += est
            bucket_selected += 1
            if pid >= 0:
                person_counts[pid] += 1


def _fill_subject_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    subj_scores, byte_budget, photo_cap,
):
    """Fill a CLIP subject bucket.

    Candidates are sorted by their subject similarity (descending) so the
    best matches for this subject are picked first.
    """
    # Re-sort by subject similarity for this bucket
    subj_sorted = sorted(
        sorted_cands,
        key=lambda r: subj_scores.get(r["path"], 0.0),
        reverse=True,
    )

    used = 0
    bucket_selected = 0
    for rec in subj_sorted:
        if _budget_exhausted():
            break
        if rec["path"] in selected_paths:
            continue
        # Skip photos with negligible subject match
        if subj_scores.get(rec["path"], 0.0) < 0.15:
            break  # sorted descending — rest will be even lower
        pid = rec.get("person_id", -1)
        if pid >= 0 and person_counts[pid] >= max_per_person_n:
            continue
        if photo_cap is not None and bucket_selected >= photo_cap:
            break
        est = _est_size(rec)
        if used + est > byte_budget:
            continue
        if _add(rec):
            used += est
            bucket_selected += 1
            if pid >= 0:
                person_counts[pid] += 1


def _fill_aesthetic_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted,
    person_counts, max_per_person_n,
):
    """Aesthetic catch-all — fills remaining budget with highest-scored photos."""
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        if rec["path"] in selected_paths:
            continue
        pid = rec.get("person_id", -1)
        if pid >= 0 and person_counts[pid] >= max_per_person_n:
            continue
        if _add(rec):
            if pid >= 0:
                person_counts[pid] += 1


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
