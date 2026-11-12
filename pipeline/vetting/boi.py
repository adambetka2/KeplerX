"""Catalogue BOI proposals without changing measurements or view eligibility."""

import math
from collections import Counter, defaultdict

EXPOSURE_DAYS = 1765.464 / 86400
MATCHING = {
    "id": "boi-ephemeris-groups-v1",
    "rule": "Same source, positive periodic ephemerides, shared-quarter event baseline, phase agreement at both ends and accumulated period drift within half the larger duration plus one exposure; complete-link groups prevent chaining.",
    "unknown_period": "Retain separate proposals; do not infer an orbit.",
    "harmonics": "Retain distinct ephemeris groups and original hypotheses.",
    "scope": "Catalogue grouping only. One passing view is sufficient; correlated views do not increase confidence. Groups are not counts of confirmed planets.",
}


def signal_record(row, signal, evidence):
    triage = signal.get("triage", {})
    hypotheses = signal.get("timing_hypotheses", [])
    hypothesis = next(
        (h for h in hypotheses if h["hypothesis_id"] == signal["signal_id"]),
        hypotheses[0] if hypotheses else {},
    )
    records = [
        h[name] for h in hypotheses for name in ("rust_record", "shallow_record")
        if name in h
    ]
    assigned = next(
        (item for item in row.get("disposition", {}).get("signals", [])
         if item["signal_id"] == signal["signal_id"]),
        {},
    )
    return {
        "source_id": str(row["source_id"]),
        "signal_id": signal["signal_id"],
        "label": assigned.get("disposition", "UNRESOLVED"),
        "discovery_view": signal.get("discovery_view", "psf_flat"),
        "detector_classifications": sorted({
            item["classification"] for item in records if "classification" in item
        }),
        "triage_state": triage.get("status"),
        "failure_stage": triage.get("held_stage", triage.get("dismissed_stage")),
        "reason": triage.get("reason"),
        "period_days": hypothesis.get("period_days", signal.get("period_days")),
        "epoch_bkjd": hypothesis.get("epoch_bkjd"),
        "duration_days": hypothesis.get("duration_days"),
        "event_ids": signal["event_ids"],
        "event_units": [
            {key: unit.get(key) for key in ("quarter", "centre_bkjd")}
            for unit in signal.get("events", [])
        ],
        "evidence": str(evidence),
        "catalogue_state": signal.get("catalogue_state"),
        "catalogue_matches": signal.get("catalogue_search", {}).get("matches", []),
    }


def periodic(record):
    values = [record.get(name) for name in (
        "period_days", "epoch_bkjd", "duration_days",
    )]
    return all(type(v) in (int, float) and math.isfinite(v) for v in values) and (
        values[0] > 0 and values[2] > 0
    )


def matching_signal(left, right):
    if left["source_id"] != right["source_id"] or not all(
        periodic(record) for record in (left, right)
    ):
        return False
    common = {u["quarter"] for u in left["event_units"]} & {
        u["quarter"] for u in right["event_units"]
    }
    times = [
        u["centre_bkjd"] for record in (left, right) for u in record["event_units"]
        if u["quarter"] in common
    ]
    if len(times) < 4 or any(
        type(t) not in (int, float) or not math.isfinite(t) for t in times
    ):
        return False
    lo, hi = min(times), max(times)
    pa, pb = left["period_days"], right["period_days"]
    tolerance = max(left["duration_days"], right["duration_days"]) / 2 + EXPOSURE_DAYS
    if hi <= lo or (hi - lo) / min(pa, pb) * abs(pa - pb) > tolerance:
        return False
    ta, tb = left["epoch_bkjd"], right["epoch_bkjd"]
    epochs = [ta + round((bound - ta) / pa) * pa for bound in (lo, hi)]
    return all(abs(t - (tb + round((t - tb) / pb) * pb)) <= tolerance for t in epochs)


def catalogue(records):
    records = list(records)
    by_source = defaultdict(list)
    identities = set()
    for record in records:
        identity = (record["source_id"], record["discovery_view"], record["signal_id"])
        if identity in identities:
            raise ValueError("Repeated signal identity in BOI catalogue")
        identities.add(identity)
        if record["label"] == "BOI":
            if record["triage_state"] != "ACTIVE":
                raise ValueError("BOI proposal is not ACTIVE")
            by_source[record["source_id"]].append(record)
    result = []
    for source_id in sorted(by_source, key=int):
        groups = []
        proposals = sorted(by_source[source_id], key=lambda r: (
            r["discovery_view"], r["signal_id"],
        ))
        for proposal in proposals:
            group = next(
                (g for g in groups if all(matching_signal(proposal, p) for p in g)),
                None,
            )
            if group is None:
                groups.append([proposal])
            else:
                group.append(proposal)
        for index, group in enumerate(groups, 1):
            result.append({
                "boi_id": f"BOI-{source_id}-{index}",
                "label": "BOI",
                "source_id": source_id,
                "views": sorted({p["discovery_view"] for p in group}),
                "proposals": group,
            })
    return {
        "schema": 1,
        "bois": len(result),
        "boi_proposals": sum(len(g["proposals"]) for g in result),
        "boi_sources": len(by_source),
        "signal_labels": dict(Counter(r["label"] for r in records)),
        "matching": MATCHING,
        "objects": result,
        "catalogue_recoveries": [r for r in records if r.get("catalogue_state") == "RECOVERED_CATALOGUE_ASSOCIATION"],
        "nearby_catalogue_associations": [r for r in records if r.get("catalogue_state") == "NEARBY_SOURCE_UNCONFIRMED"],
        "catalogue_search_incomplete": sum(r.get("catalogue_state") == "CATALOGUE_SEARCH_INCOMPLETE" for r in records),
    }
