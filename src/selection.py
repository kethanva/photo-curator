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
import time
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
    max_per_day_pct: float = 0.15,
    max_per_hour_pct: float = 0.05,
    min_score_pct: float = 0.0,
    min_pool_fraction: float = 0.40,
    buckets: Optional[Dict[str, float]] = None,
    subject_scores: Optional[Dict[str, Dict[str, float]]] = None,
    output_mode: str = "bytes",
    output_percentage: float = 0.15,
    total_photos: int = 0,
    resize_output: bool = True,
    max_dynamic_spacing: float = 600.0,
    output_long_side: int = 2560,
    output_jpeg_quality: int = 92,
) -> List[dict]:
    """
    Select photos using a dynamic bucket diversity strategy.

    Pre-filters duplicates and private photos as HARD gates. ``quality_pass``
    is treated as a SOFT preference: strict (quality_pass=1) candidates are
    always preferred, but when the strict pool is smaller than
    ``min_pool_fraction * total_photos`` the highest-scored ``quality_pass=0``
    photos are admitted to top the pool up to that floor. This prevents
    aggressive quality thresholds from starving the output on libraries with
    many soft-focus or underexposed shots.

    Args:
        buckets:          ordered {name: fraction} — fractions should sum to 1.0.
                          Defaults to {"people": 0.30, "location": 0.30, "aesthetic": 0.40}.
        subject_scores:   {bucket_name: {path: similarity}} for CLIP subject buckets.
        max_per_day_pct:  max fraction of output from any single calendar day.
                          Strongest temporal diversity guard — prevents one shooting
                          session from dominating. 0.0 disables.
        max_per_hour_pct: max fraction of output from any single calendar hour.
                          0.0 disables the cap.
        min_score_pct:    drop the bottom fraction of candidates by composite score
                          before selection (e.g. 0.15 drops the lowest-scored 15%).
                          0.0 disables the filter.
        min_pool_fraction: minimum candidate pool size as a fraction of
                          ``total_photos``. If the strict (quality_pass=1)
                          pool is smaller than this floor, the best-scored
                          ``quality_pass=0`` photos are admitted to pad it.
                          0.0 disables the pad (strict filter only).

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

    # Hard gates: duplicates and privacy-flagged photos never enter the pool.
    eligible = [
        r for r in records
        if not r.get("is_duplicate", 0) and not r.get("is_private", 0)
    ]
    strict_candidates = [r for r in eligible if r.get("quality_pass", 1)]
    fallback_candidates = [r for r in eligible if not r.get("quality_pass", 1)]

    # Expand the pool with best-scored quality-fail photos if the strict
    # pool doesn't meet the configured floor. Without this, aggressive
    # blur/exposure thresholds can starve the output (e.g. 335 photos →
    # 11 eligible → only 5 selected after diversity caps).
    pool_floor_base = total_photos if total_photos > 0 else len(eligible)
    pool_floor = int(pool_floor_base * min_pool_fraction)
    candidates = list(strict_candidates)
    if min_pool_fraction > 0.0 and len(candidates) < pool_floor and fallback_candidates:
        needed = min(pool_floor - len(candidates), len(fallback_candidates))
        fallback_sorted = sorted(
            fallback_candidates,
            key=lambda r: scores.get(r["path"], 0.0),
            reverse=True,
        )
        admitted = fallback_sorted[:needed]
        candidates.extend(admitted)
        print(
            f"  Pool expansion: strict pool {len(strict_candidates)} < "
            f"floor {pool_floor} ({min_pool_fraction*100:.0f}% of "
            f"{pool_floor_base}). Admitted {len(admitted)} top-scored "
            f"quality_pass=0 photos."
        )

    if not candidates:
        return []

    # ── Estimate output sizes after resizing ──────────────────────
    # Dynamically estimate output JPEG size based on requested resolution and quality.
    def _est_size(rec: dict) -> int:
        orig = rec.get("file_size") or 2_000_000
        if resize_output:
            # Dynamically estimate bits per pixel based on JPEG quality
            if output_jpeg_quality >= 95: bpp = 3.5
            elif output_jpeg_quality >= 90: bpp = 2.5
            elif output_jpeg_quality >= 85: bpp = 1.5
            elif output_jpeg_quality >= 75: bpp = 1.0
            else: bpp = 0.5
            
            # Assume average 4:3 aspect ratio
            pixels = output_long_side * (output_long_side * 0.75)
            estimated_bytes = int((pixels * bpp) / 8.0)
            
            return min(orig, estimated_bytes)
        return min(orig, 8_000_000)

    # Derive photo count target for percentage mode BEFORE score filtering,
    # so the target reflects the eligible pool size, not the post-filter remainder.
    max_photos: Optional[int] = None
    if output_mode == "percentage":
        base = total_photos if total_photos > 0 else len(candidates)
        target = max(1, int(base * output_percentage))
        max_photos = min(target, len(candidates))

    # Drop bottom fraction by composite score (removes visually meaningless photos).
    # Done after max_photos so the target isn't shrunk by the filter.
    if min_score_pct > 0.0 and scores:
        cand_scores = [scores.get(r["path"], 0.0) for r in candidates]
        threshold = float(np.percentile(cand_scores, min_score_pct * 100.0))
        candidates = [r for r in candidates if scores.get(r["path"], 0.0) >= threshold]
        if not candidates:
            return []
        # Clamp max_photos to the surviving candidate count
        if max_photos is not None:
            max_photos = min(max_photos, len(candidates))

    # ── Derive per-subject caps from the output target ────────────
    if max_photos is not None:
        cap_base = max_photos
        target_for_selectivity = max_photos
    else:
        # Bytes mode estimation for better selectivity and diversity caps
        avg_size = sum(_est_size(r) for r in candidates) / len(candidates) if candidates else 2_000_000
        est_count = int(max_bytes / max(1, avg_size))
        cap_base = min(len(candidates), est_count)
        target_for_selectivity = cap_base

    # All diversity caps below are STRICT — no auto-raise. Previously the
    # cluster and hour caps silently relaxed themselves to ceil(target/buckets)
    # when strict caps would have starved the target, which let one event fill
    # ~50% of small outputs. Symmetrical strict-everywhere matches the day-cap
    # philosophy: diversity wins over hitting the exact count. If caps prevent
    # reaching target, a warning is logged at the end of selection.

    max_per_person_n   = max(1, int(np.ceil(cap_base * max_per_person_pct)))
    max_per_location_n = max(1, int(np.ceil(cap_base * max_per_location_pct)))
    max_per_cluster_n  = max(1, int(np.ceil(cap_base * max_per_cluster_pct)))

    # ── Dynamic Temporal Spacing ──────────────────────────────────
    global_selectivity = target_for_selectivity / max(1, len(candidates))

    cluster_stats = {}
    for r in candidates:
        cid = r.get("cluster_id", -1)
        ts = r.get("timestamp", 0)
        if cid >= 0 and ts > 0:
            if cid not in cluster_stats:
                cluster_stats[cid] = {"min_ts": ts, "max_ts": ts, "count": 0}
            else:
                if ts < cluster_stats[cid]["min_ts"]: cluster_stats[cid]["min_ts"] = ts
                if ts > cluster_stats[cid]["max_ts"]: cluster_stats[cid]["max_ts"] = ts
            cluster_stats[cid]["count"] += 1

    cluster_dynamic_spacing = {}
    for cid, stats in cluster_stats.items():
        if stats["count"] < 2:
            cluster_dynamic_spacing[cid] = 0.0
            continue
            
        duration = stats["max_ts"] - stats["min_ts"]
            
        # Target number of photos is constrained by the actual event duration.
        # This prevents "bursts" (high count, short duration) from yielding tiny gaps.
        # Assume a dense event justifies 1 photo per 120 seconds.
        duration_based_n = max(0.5, duration / 120.0)
        
        expected_n = stats["count"] * global_selectivity
        
        # Target N is bounded by duration to aggressively squash bursts
        target_n = min(expected_n, duration_based_n, max_per_cluster_n)
        
        if target_n <= 1.0:
            # Event is too short/sparse to justify multiple photos. Force massive gap.
            dynamic_sec = float('inf')
        else:
            avg_gap = duration / target_n
            # Allow some wiggle room (0.6x) so they don't have to be perfectly evenly spaced
            dynamic_sec = avg_gap * 0.6
        
        # Absolute floor increased to 60s to prevent back-to-back similar shots
        absolute_floor = 60.0
        cluster_dynamic_spacing[cid] = min(float(max_dynamic_spacing), max(absolute_floor, dynamic_sec))

    def _day_key(rec: dict) -> int:
        """Local-time calendar day. Photos missing a timestamp get a unique negative ID
        so they don't incorrectly pool together and get throttled."""
        ts = rec.get("timestamp", 0) or 0
        if ts <= 0:
            return -abs(hash(rec["path"])) - 1
        lt = time.localtime(ts)
        return lt.tm_year * 1000 + lt.tm_yday

    def _hour_key(rec: dict) -> int:
        """Local-time calendar hour. Missing-timestamp photos get unique negative IDs."""
        ts = rec.get("timestamp", 0) or 0
        if ts <= 0:
            return -abs(hash(rec["path"])) - 1
        lt = time.localtime(ts)
        return lt.tm_year * 1_000_000 + lt.tm_yday * 100 + lt.tm_hour

    # Cluster cap — strict.
    distinct_clusters = {
        r.get("cluster_id", -1) for r in candidates
        if r.get("cluster_id", -1) >= 0
    }
    enforce_cluster_cap = len(distinct_clusters) >= 2

    # Day cap — strict.
    max_per_day_n: Optional[int] = None
    if max_per_day_pct > 0.0:
        distinct_days = {_day_key(r) for r in candidates}
        if len(distinct_days) >= 2:
            max_per_day_n = max(1, int(np.ceil(cap_base * max_per_day_pct)))

    # Hour cap — strict.
    max_per_hour_n: Optional[int] = None
    if max_per_hour_pct > 0.0:
        distinct_hours = {_hour_key(r) for r in candidates}
        if len(distinct_hours) >= 2:
            max_per_hour_n = max(1, int(np.ceil(cap_base * max_per_hour_pct)))

    # Sort all candidates by score descending
    sorted_cands = sorted(
        candidates, key=lambda r: scores.get(r["path"], 0.0), reverse=True
    )
    selected_paths: Set[str] = set()
    selected: List[dict] = []
    used_bytes: int = 0
    hour_counts: Dict[int, int] = defaultdict(int)
    day_counts: Dict[int, int] = defaultdict(int)

    def _add(rec: dict) -> bool:
        nonlocal used_bytes
        if rec["path"] in selected_paths:
            return False

        # Check dynamic temporal spacing within the same event cluster
        ts = rec.get("timestamp", 0) or 0
        cid = rec.get("cluster_id", -1)
        
        if ts > 0 and cid >= 0:
            req_spacing = cluster_dynamic_spacing.get(cid, 0.0)
            if req_spacing > 0:
                for sel_rec in selected:
                    if sel_rec.get("cluster_id", -1) == cid:
                        sel_ts = sel_rec.get("timestamp", 0) or 0
                        if sel_ts > 0 and abs(ts - sel_ts) < req_spacing:
                            return False

        est = _est_size(rec)
        if used_bytes + est > max_bytes:
            return False
        if max_photos is not None and len(selected) >= max_photos:
            return False
        # Per-day temporal diversity cap — applies to the missing-timestamp
        # pool (-1) too, so untimed photos can't bypass it.
        if max_per_day_n is not None:
            if day_counts[_day_key(rec)] >= max_per_day_n:
                return False
        # Per-hour temporal diversity cap — same pooling for missing timestamps.
        if max_per_hour_n is not None:
            if hour_counts[_hour_key(rec)] >= max_per_hour_n:
                return False
        selected_paths.add(rec["path"])
        selected.append(rec)
        used_bytes += est
        if max_per_day_n is not None:
            day_counts[_day_key(rec)] += 1
        if max_per_hour_n is not None:
            hour_counts[_hour_key(rec)] += 1
        return True

    def _budget_exhausted() -> bool:
        if max_photos is not None and len(selected) >= max_photos:
            return True
        return used_bytes >= max_bytes

    # Shared counters — enforced across all buckets
    person_counts: Dict[int, int] = defaultdict(int)
    loc_counts: Dict[int, int] = defaultdict(int)
    cluster_counts: Dict[int, int] = defaultdict(int)

    def _cluster_capped(cid: int) -> bool:
        """True if event-cluster cap should block this record."""
        if not enforce_cluster_cap:
            return False
        if cid < 0:
            return False  # noise/singleton — diversity not meaningful
        return cluster_counts[cid] >= max_per_cluster_n

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
                cluster_counts, _cluster_capped,
                bucket_byte_budget, bucket_photo_cap,
            )

        elif bucket_name == "location":
            _fill_location_bucket(
                sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
                person_counts, max_per_person_n,
                loc_counts, max_per_location_n,
                cluster_counts, _cluster_capped,
                bucket_byte_budget, bucket_photo_cap,
            )

        else:
            # CLIP subject bucket
            subj_scores = subject_scores.get(bucket_name, {})
            _fill_subject_bucket(
                sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
                person_counts, max_per_person_n,
                cluster_counts, _cluster_capped,
                subj_scores, bucket_byte_budget, bucket_photo_cap,
            )

    # ── Aesthetic share (always last) ────────────────────────────
    # Aesthetic bucket takes its own fraction PLUS any unused budget from previous buckets.
    # This ensures we hit the requested target max_photos / max_bytes if possible,
    # rather than artificially under-delivering just because earlier buckets were sparse.
    if aesthetic_fraction > 0 and not _budget_exhausted():
        aesthetic_byte_budget = max_bytes - used_bytes
        aesthetic_photo_cap = (
            max(1, max_photos - len(selected))
            if max_photos is not None else None
        )
        _fill_aesthetic_bucket(
            sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
            person_counts, max_per_person_n,
            cluster_counts, _cluster_capped,
            _day_key, _hour_key, day_counts, hour_counts,
            max_per_day_n, max_per_hour_n,
            aesthetic_byte_budget, aesthetic_photo_cap,
        )

    # Make strict-cap undershoot visible. Hidden auto-raise was the recurring
    # bug we just removed — surfacing the gap is the honest replacement.
    if max_photos is not None and len(selected) < max_photos:
        active = [
            f"day={max_per_day_n}" if max_per_day_n is not None else None,
            f"hour={max_per_hour_n}" if max_per_hour_n is not None else None,
            f"cluster={max_per_cluster_n}" if enforce_cluster_cap else None,
            f"person={max_per_person_n}",
            f"location={max_per_location_n}",
        ]
        active_str = ", ".join(c for c in active if c)
        print(
            f"  Note: selected {len(selected)} / target {max_photos}. "
            f"Diversity caps capped the output ({active_str}). "
            f"Raise max_per_*_pct in config.yaml or supply more diverse input "
            f"to reach the target."
        )

    return selected


# ---------------------------------------------------------------------------
# Bucket fill helpers
# ---------------------------------------------------------------------------

def _fill_people_bucket(
    sorted_cands, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    cluster_counts, _cluster_capped,
    byte_budget, photo_cap,
):
    used = 0
    bucket_selected = 0
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        if photo_cap is not None and bucket_selected >= photo_cap:
            break
        pid = rec.get("person_id", -1)
        if pid < 0 or not rec.get("is_frequent", 0):
            continue
        if person_counts[pid] >= max_per_person_n:
            continue
        cid = rec.get("cluster_id", -1)
        if _cluster_capped(cid):
            continue
        est = _est_size(rec)
        if used + est > byte_budget:
            continue
        if _add(rec):
            person_counts[pid] += 1
            cluster_counts[cid] += 1
            used += est
            bucket_selected += 1


def _fill_location_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    loc_counts, max_per_location_n,
    cluster_counts, _cluster_capped,
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
        if cid >= 0 and loc_counts[cid] >= max_per_location_n:
            continue
        if _cluster_capped(cid):
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
            cluster_counts[cid] += 1
            used += est
            bucket_selected += 1
            if pid >= 0:
                person_counts[pid] += 1


def _fill_subject_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    cluster_counts, _cluster_capped,
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
        cid = rec.get("cluster_id", -1)
        if _cluster_capped(cid):
            continue
        if photo_cap is not None and bucket_selected >= photo_cap:
            break
        est = _est_size(rec)
        if used + est > byte_budget:
            continue
        if _add(rec):
            used += est
            bucket_selected += 1
            cluster_counts[cid] += 1
            if pid >= 0:
                person_counts[pid] += 1


def _fill_aesthetic_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    cluster_counts, _cluster_capped,
    _day_key, _hour_key, day_counts, hour_counts,
    max_per_day_n, max_per_hour_n,
    byte_budget, photo_cap,
):
    """Aesthetic share — fills the remaining byte/photo budget with the
    highest-scored remaining candidates.

    Enforces every diversity cap (person, cluster, day, hour) before the
    `_add` call so iteration doesn't bounce off saturated buckets.
    Skipping saturated day/hour buckets early naturally promotes
    under-represented buckets without an explicit round-robin pass.
    """
    used = 0
    bucket_selected = 0
    for rec in sorted_cands:
        if _budget_exhausted():
            break
        if rec["path"] in selected_paths:
            continue
        if photo_cap is not None and bucket_selected >= photo_cap:
            break
        # Early diversity skips — avoid the wasted _add call when caps already bind.
        if max_per_day_n is not None and day_counts[_day_key(rec)] >= max_per_day_n:
            continue
        if max_per_hour_n is not None and hour_counts[_hour_key(rec)] >= max_per_hour_n:
            continue
        pid = rec.get("person_id", -1)
        if pid >= 0 and person_counts[pid] >= max_per_person_n:
            continue
        cid = rec.get("cluster_id", -1)
        if _cluster_capped(cid):
            continue
        est = _est_size(rec)
        if used + est > byte_budget:
            continue
        if _add(rec):
            used += est
            bucket_selected += 1
            cluster_counts[cid] += 1
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
    collision_counts: Dict[str, int] = defaultdict(int)

    for rec in selected:
        src = Path(rec["path"])

        # Always output as .jpg when resizing
        stem = src.stem
        suffix = ".jpg" if resize else src.suffix
        dst = out / f"{stem}{suffix}"

        # Resolve collisions
        if dst.exists():
            base_key = f"{stem}__{src.parent.name}"
            counter = collision_counts[base_key] + 1
            dst = out / f"{base_key}_{counter}{suffix}"
            while dst.exists():
                counter += 1
                dst = out / f"{base_key}_{counter}{suffix}"
            collision_counts[base_key] = counter

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
