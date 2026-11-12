"""Instrument evidence at supplied intervals, without an instrumental veto."""

from pathlib import Path

import numpy as np
from lc2 import input_manifest, load_curve, photometry
from lc2 import vetting_points as valid_points
from parent_index import parent_members
from vetting_models import linear_fit, measure_interval

ROOT = Path(__file__).resolve().parents[2]

KEPLER_FLAGS = {
    1: "Attitude tweak",
    2: "Safe mode",
    4: "Coarse point",
    8: "Earth point",
    16: "Zero crossing",
    32: "Desaturation",
    64: "Argabrightening",
    128: "Cosmic ray in aperture",
    256: "Manual exclude",
    1024: "Sudden sensitivity dropout",
    2048: "Impulsive outlier",
    4096: "Argabrightening on CCD",
    8192: "Cosmic ray in collateral",
    16384: "Detector anomaly",
    32768: "No fine point",
    65536: "No data",
    131072: "Rolling band in aperture",
    262144: "Rolling band in mask",
    524288: "Possible thruster firing",
    1048576: "Thruster firing",
}


def engineering_fit(time, values, centre, duration):
    use = np.isfinite(time) & np.isfinite(values) & (abs(time - centre) <= 4 * duration)
    x = time[use] - centre
    if len(x) < 4:
        return {"status": "INSUFFICIENT_COVERAGE"}
    base = np.column_stack([np.ones(len(x)), x])
    return {
        "status": "MEASURED",
        "baseline": linear_fit(base, values[use], np.ones(len(x))),
        "event": linear_fit(
            np.column_stack([base, abs(x) <= duration / 2]),
            values[use],
            np.ones(len(x)),
        ),
        "scope": "Engineering-channel least-squares change; unit weights, no astronomical significance claim",
    }


def quarter_instrument_data(raw, view, quarter):
    curve = photometry(raw, view)
    use = (curve["quarter"] == quarter) & np.isfinite(curve["time"])
    order = np.argsort(curve["time"][use])
    time = curve["time"][use][order]
    cadence = curve["cadence"][use][order]
    gaps = np.diff(cadence) != 1
    return {
        "time": time,
        "quality": curve["quality"][use][order].astype(np.int64),
        "gap_starts": time[:-1][gaps],
        "gap_ends": time[1:][gaps],
    }


def engineering_evidence(raw, centre, duration, diagnostic):
    if not diagnostic:
        return {}, "NOT_SAMPLED_DIAGNOSTIC"
    records = {
        name: engineering_fit(raw["time_bkjd"], raw[name], centre, duration)
        for name in ("background", "pos_corr1", "pos_corr2")
        if name in raw
    }
    return records, "AVAILABLE" if records else "NOT_IN_LIGHT_CURVE"


def event_instrument(raw, event, quarter=None, diagnostic=True):
    centre, duration = event["centre_bkjd"], event["duration_days"]
    if quarter is None:
        quarter = quarter_instrument_data(raw, event["photometry"], event["quarter"])
    time = quarter["time"]
    inside = abs(time - centre) <= duration / 2
    quality = quarter["quality"][inside]
    flags = {
        label: int(((quality & bit) != 0).sum()) for bit, label in KEPLER_FLAGS.items()
    }
    near_gap = (quarter["gap_starts"] <= centre + duration) & (
        quarter["gap_ends"] >= centre - duration
    )
    boundaries = [
        [float(start), float(end)]
        for start, end in zip(
            quarter["gap_starts"][near_gap], quarter["gap_ends"][near_gap]
        )
    ]
    engineering, availability = engineering_evidence(raw, centre, duration, diagnostic)
    return {
        "event_id": event["event_id"],
        "status": "MEASURED",
        "flags": flags,
        "in_event_points": int(inside.sum()),
        "nearby_gaps_bkjd": boundaries,
        "quarter_time_bounds_bkjd": [float(time.min()), float(time.max())]
        if len(time)
        else None,
        "engineering": engineering,
        "engineering_availability": availability,
        "shared_time": event.get("shared_time", {"status": "UNAVAILABLE"}),
    }


def sibling_measurements(sid, parent, events, view, manifest, config):
    curve = load_curve(ROOT, sid, view, manifest)
    if curve["provenance"]["parent_tpf"] != str(parent):
        raise ValueError(f"Sibling extraction has another parent TPF: {sid}")
    valid = valid_points(curve, config["quality_bitmask"])
    quarters = {}
    records = []
    for event in events:
        quarter = event["quarter"]
        if quarter not in quarters:
            take = curve["quarter"] == quarter
            data = np.column_stack(
                [
                    curve["flux"][take],
                    curve["error"][take],
                    curve["time"][take],
                    valid[take],
                ]
            )
            quarters[quarter] = data[np.argsort(data[:, 2])]
        result = measure_interval(
            None,
            event["centre_bkjd"],
            event["duration_days"],
            view,
            config,
            prepared=quarters[quarter],
            sorted_time=True,
        )
        records.append(
            dict(result, source_id=sid, event_id=event["event_id"], view=view)
        )
    return records


def comparison_measurements(row, config):
    selected = set(row.get("instrument", {}).get("selected_event_ids", []))
    events = [event for event in row["events"] if event["event_id"] in selected]
    parent = row["coordinates"]["origin_tpf_kic"]
    siblings = parent_members(
        ROOT / config["parent_index"],
        config["parent_index_sha256"],
        row["source_id"],
        parent,
    )
    manifest = config.get("input_manifest")
    experimental_sources = (
        set(input_manifest(ROOT, manifest)[1]["sources"]) if manifest else None
    )
    missing = (
        [sid for sid in siblings if sid not in experimental_sources]
        if siblings is not None and experimental_sources is not None
        else []
    )
    records = []
    for sid in siblings or ():
        if sid in missing:
            continue
        for view in sorted({event["photometry"] for event in events}):
            view_events = [event for event in events if event["photometry"] == view]
            records.extend(
                sibling_measurements(sid, parent, view_events, view, manifest, config)
            )
    if siblings is None:
        status = "SOURCE_NOT_INDEXED"
    elif not events:
        status = "NO_SELECTED_EVENTS"
    elif missing:
        status = "INCOMPLETE_EXPERIMENTAL_INPUTS"
    else:
        status = "MEASURED"
    return {
        "status": status,
        "parent_tpf": str(parent),
        "expected_sibling_ids": list(siblings) if siblings is not None else None,
        "missing_sibling_ids": missing,
        "selected_event_ids": sorted(selected),
        "records": records,
        "scope": "Every verified LC2 source sharing the parent TPF at selected event times. Experimental sibling inputs must be in the same input manifest. Shared extractions are correlated; coincidence alone is not causal attribution.",
    }


def verify(row, raw, config):
    from vetting_signals import representative_event_ids
    from vetting_triage import measured_events, measured_signals

    selected = representative_event_ids(
        measured_signals(row), config["diagnostic_max_units_per_signal"]
    )
    events = measured_events(row)
    quarters = {}
    records = []
    for event in events:
        key = event["photometry"], event["quarter"]
        if key not in quarters:
            quarters[key] = quarter_instrument_data(raw, *key)
        records.append(
            event_instrument(
                raw,
                event,
                quarters[key],
                diagnostic=event["event_id"] in selected,
            )
        )
    row["instrument"] = {
        "status": "MEASURED" if events else "NOT_APPLICABLE_NO_ACTIVE_SIGNALS",
        "events": records,
        "evaluated_events": len(events),
        "diagnostic_events": sum(event["event_id"] in selected for event in events),
        "selected_event_ids": sorted(selected),
        "selection": {
            "maximum_units_per_signal": config["diagnostic_max_units_per_signal"],
            "rule": "Engineering diagnostics use one central unit per quarter, then evenly spaced units across time",
        },
        "skipped_dismissed_events": len(row["events"]) - len(events),
        "scope": "Quality flags, edges, gaps and shared-time context cover every active event; engineering diagnostics use the declared representative sample. Unavailable channels remain explicit.",
    }
    return row


if __name__ == "__main__":
    from vetting import program_main

    program_main("instrument")
