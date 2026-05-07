"""
Bucket fill helpers for the selection engine.

These are pure, top-level functions extracted from ``selection.py`` to keep
``select_photos`` close to a readable orchestrator. Every helper takes the
shared selection state (``_add``, counters, key functions, caps) as explicit
arguments — no module-level state, no closures captured here.

Routing through ``_add`` is the contract: ``_add`` is the single arbiter of
byte budget, ``max_photos`` cap, day/hour caps, per-cluster dynamic spacing,
and global temporal proximity. Helpers here only enforce extra caps that
``_add`` does not know about (``person_counts``, ``cluster_counts``,
``loc_counts``) and update those counters on success.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def fill_people_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
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
        # Skip already-selected paths up-front so a redundant candidate
        # does not consume a person/cluster slot. Without this guard, an
        # earlier bucket's pick could come back here and starve a fresh
        # photo from the same event.
        if rec["path"] in selected_paths:
            continue
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


def fill_location_bucket(
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


def fill_subject_bucket(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    cluster_counts, _cluster_capped,
    _day_key, _hour_key, day_counts, hour_counts,
    max_per_day_n, max_per_hour_n,
    subj_scores, byte_budget, photo_cap,
):
    """Fill a CLIP subject bucket.

    Candidates are sorted by their subject similarity (descending) so the
    best matches for this subject are picked first. Day/hour caps are
    pre-checked here (matching fill_aesthetic_bucket) so saturated buckets
    don't burn iterations or inflate the per-bucket photo cap on rejected
    records.
    """
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
        # Skip photos with weak subject match. 0.20 sits above the
        # background CLIP similarity floor (~0.15–0.18 for unrelated photos)
        # so subject buckets pull genuine matches rather than noise.
        if subj_scores.get(rec["path"], 0.0) < 0.20:
            break  # sorted descending — rest will be even lower
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


def fill_stratified_spread(
    sorted_cands, selected_paths, _add, _budget_exhausted, _est_size,
    person_counts, max_per_person_n,
    cluster_counts, _cluster_capped,
    _day_key, _hour_key, day_counts, hour_counts,
    max_per_day_n, max_per_hour_n,
    byte_budget, photo_cap,
):
    """Stratified spread pass — guarantees each distinct day AND each distinct
    location/event cluster contributes its best-scoring candidate before the
    greedy by-score aesthetic pass runs.

    This is the positive-spread mechanism that complements the defensive caps:
    instead of letting score-clustered candidates dominate, every timeline
    bucket and every location bucket gets a turn.
    """
    used = 0
    bucket_selected = 0

    # Pass A — one best photo from each distinct day not yet represented.
    seen_days_at_start = {d for d, c in day_counts.items() if c > 0}
    by_day_unrepresented: Dict[int, List[dict]] = {}
    for rec in sorted_cands:
        if rec["path"] in selected_paths:
            continue
        d = _day_key(rec)
        if d in seen_days_at_start:
            continue
        by_day_unrepresented.setdefault(d, []).append(rec)

    for _day, queue in by_day_unrepresented.items():
        if _budget_exhausted():
            break
        if photo_cap is not None and bucket_selected >= photo_cap:
            break
        for rec in queue:
            if rec["path"] in selected_paths:
                continue
            pid = rec.get("person_id", -1)
            if pid >= 0 and person_counts[pid] >= max_per_person_n:
                continue
            cid = rec.get("cluster_id", -1)
            if _cluster_capped(cid):
                continue
            est = _est_size(rec)
            if used + est > byte_budget:
                break
            if _add(rec):
                used += est
                bucket_selected += 1
                cluster_counts[cid] += 1
                if pid >= 0:
                    person_counts[pid] += 1
                break  # one per day in this pass

    # Pass B — one best photo per distinct GPS/event cluster not yet seen.
    seen_clusters_at_start = {c for c, n in cluster_counts.items() if n > 0}
    by_cluster_unrepresented: Dict[int, List[dict]] = {}
    for rec in sorted_cands:
        if rec["path"] in selected_paths:
            continue
        cid = rec.get("cluster_id", -1)
        if cid < 0 or cid in seen_clusters_at_start:
            continue
        by_cluster_unrepresented.setdefault(cid, []).append(rec)

    for _cid, queue in by_cluster_unrepresented.items():
        if _budget_exhausted():
            break
        if photo_cap is not None and bucket_selected >= photo_cap:
            break
        for rec in queue:
            if rec["path"] in selected_paths:
                continue
            pid = rec.get("person_id", -1)
            if pid >= 0 and person_counts[pid] >= max_per_person_n:
                continue
            cid = rec.get("cluster_id", -1)
            if _cluster_capped(cid):
                continue
            est = _est_size(rec)
            if used + est > byte_budget:
                break
            if _add(rec):
                used += est
                bucket_selected += 1
                cluster_counts[cid] += 1
                if pid >= 0:
                    person_counts[pid] += 1
                break


def backfill_round_robin(
    *,
    sorted_cands,
    selected_paths,
    selected,
    _add,
    _day_key,
    day_counts,
    cluster_counts,
    _cluster_capped,
    person_counts,
    max_per_person_n,
    max_per_day_n,
    max_photos,
):
    """Diversity-respecting backfill.

    Walks distinct days in round-robin order (under-represented days first),
    pulling each day's highest-scored remaining candidate.

    Every addition is routed through ``_add`` — the same closure the main
    selection loop uses — so per-cluster dynamic spacing, global temporal
    proximity, ``max_bytes``, ``max_photos``, and per-day/hour caps are all
    enforced consistently. This function only filters candidates by the
    cluster/person caps that ``_add`` does not know about, and updates the
    matching counters on success.
    """
    by_day: Dict[int, List[dict]] = defaultdict(list)
    for rec in sorted_cands:
        if rec["path"] in selected_paths:
            continue
        by_day[_day_key(rec)].append(rec)

    while len(selected) < max_photos:
        ordered_days = sorted(
            (d for d, q in by_day.items() if q),
            key=lambda d: (day_counts[d], -len(by_day[d])),
        )
        if not ordered_days:
            break

        progress = False
        for day in ordered_days:
            if len(selected) >= max_photos:
                break

            if max_per_day_n is not None and day_counts[day] >= max_per_day_n:
                by_day[day].clear()
                continue

            queue = by_day[day]
            picked = None
            while queue:
                rec = queue.pop(0)
                if rec["path"] in selected_paths:
                    continue
                pid = rec.get("person_id", -1)
                if pid >= 0 and person_counts[pid] >= max_per_person_n:
                    continue
                cid = rec.get("cluster_id", -1)
                if _cluster_capped(cid):
                    continue
                if not _add(rec):
                    continue
                picked = rec
                break

            if picked is None:
                continue

            cid = picked.get("cluster_id", -1)
            cluster_counts[cid] += 1
            pid = picked.get("person_id", -1)
            if pid >= 0:
                person_counts[pid] += 1
            progress = True

        if not progress:
            break


def fill_aesthetic_bucket(
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
