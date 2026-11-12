"""Reconcile frozen pixel-level planet controls through the complete pipeline."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.deterministic import (
    catalogue_control_report,
    injections,
    pixel_parent_audit,
    run as detector_run,
)
from pipeline.runtime import EVENT_MATCHING_CONTRACT, OUTCOME_CONTRACT


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def truth_event_ledger(truth, views):
    events = {}
    duration = float(truth["duration_hours"]) / 24
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Physical truth duration is missing or invalid")
    for view, records in views.items():
        centres = np.asarray(
            injections.truth_centres(
                truth, float(records["time"].min()), float(records["time"].max())
            )
        )
        if not np.isfinite(centres).all() or np.any(np.diff(centres) <= 0):
            raise ValueError("Physical event clock is invalid or reordered")
        after = np.clip(np.searchsorted(centres, records["time"]), 0, len(centres) - 1)
        before = np.maximum(0, after - 1)
        nearest = np.where(
            abs(records["time"] - centres[before])
            <= abs(records["time"] - centres[after]),
            before,
            after,
        )
        inside = abs(records["time"] - centres[nearest]) <= duration / 2
        pairs, counts = np.unique(
            np.column_stack((nearest[inside], records["quarter"][inside])),
            axis=0,
            return_counts=True,
        )
        for (index, quarter), count in zip(pairs, counts):
            centre = float(centres[index])
            cycle = int(
                round((centre - truth["epoch_bkjd"]) / truth["shape_period_days"])
            )
            event = events.setdefault(
                cycle,
                {
                    "event_id": f"truth-{truth['source_id']}-c{cycle}",
                    "cycle": cycle,
                    "centre_bkjd": centre,
                    "support": {},
                },
            )
            event["support"].setdefault(str(int(quarter)), {})[view] = int(count)
    return sorted(events.values(), key=lambda event: event["centre_bkjd"])


def match_event_units(events, units, duration_hours):
    tolerance = float(duration_hours) / 48 + EVENT_MATCHING_CONTRACT["exposure_days"]
    if not np.isfinite(tolerance) or duration_hours <= 0:
        raise ValueError("Event-matching duration is invalid")
    identities = [unit["unit_id"] for unit in units]
    if len(identities) != len(set(identities)) or not all(identities):
        raise ValueError("Supplied unit identities are missing or duplicated")
    truth_ids = [event["event_id"] for event in events]
    if len(truth_ids) != len(set(truth_ids)) or not all(truth_ids):
        raise ValueError("Truth event identities are missing or duplicated")
    centres = np.asarray([event["centre_bkjd"] for event in events])
    if not np.isfinite(centres).all() or np.any(np.diff(centres) <= 0):
        raise ValueError("Truth event ledger is invalid or reordered")
    matches = []
    for unit in units:
        centre = float(unit["centre_bkjd"])
        if not np.isfinite(centre):
            raise ValueError("Supplied event timing is not finite")
        position = int(np.searchsorted(centres, centre))
        options = [
            events[index]
            for index in (position - 1, position)
            if 0 <= index < len(events)
            and str(unit["quarter"]) in events[index]["support"]
            and abs(centres[index] - centre) <= tolerance
        ]
        if len(options) == 1:
            event = options[0]
            matches.append(
                {
                    "unit_id": unit["unit_id"],
                    "truth_event_id": event["event_id"],
                    "timing_error_days": abs(centre - event["centre_bkjd"]),
                }
            )
    matched_ids = {match["truth_event_id"] for match in matches}
    complete = bool(units) and len(matches) == len(units) == len(matched_ids)
    return {
        "status": "MATCHED" if complete else "NOT_MATCHED",
        "supplied_units": len(units),
        "matched_units": len(matches),
        "unique_matched_truth_events": len(matched_ids),
        "available_truth_events": len(events),
        "truth_event_coverage": len(matched_ids) / len(events) if events else 0.0,
        "timing_tolerance_days": tolerance,
        "matches": matches,
    }


def fitted_event_units(row, signal, assigned):
    records = assigned.get("signal_shape", {}).get("records", [])
    if len(records) != 1:
        return None
    selected = records[0]
    if len(signal["events"]) == 1:
        event = next(
            (
                event
                for event in row["events"]
                if event["event_id"] == selected.get("event_id")
            ),
            None,
        )
        if event is None or event["fit"].get("status") != "FIT_COMPLETE":
            return None
        return [{**signal["events"][0], "centre_bkjd": event["fit"]["centre_bkjd"]}]
    if set(selected.get("unit_ids", [])) != {
        unit["unit_id"] for unit in signal["events"]
    }:
        return None
    result = (
        row.get("orbital", {}).get("hypotheses", {}).get(selected["hypothesis_id"], {})
    )
    refinement = selected.get("timing_refinement")
    measurement = result["timing_refinement"] if refinement is not None else result
    fitted = (
        measurement.get("views", {}).get(selected["view"], {}).get("joint_shape", {})
    )
    if fitted.get("status") != "FIT_COMPLETE" or not fitted.get("theta"):
        return None
    offset = float(fitted["theta"][0])
    if not np.isfinite(offset):
        raise ValueError("Selected physical fit timing is not finite")
    timing = refinement["timing"] if refinement is not None else None
    return [
        {
            **unit,
            "centre_bkjd": unit["centre_bkjd"]
            + offset
            + (
                timing["start_shift_days"]
                + timing["slope"] * (unit["centre_bkjd"] - timing["first_centre_bkjd"])
                if timing is not None
                else 0
            ),
        }
        for unit in signal["events"]
    ]


def disposition(row, signal_id):
    return next(
        (
            item
            for item in row.get("disposition", {}).get("signals", [])
            if item["signal_id"] == signal_id
        ),
        {},
    )


def downstream(row, signal):
    terminal = catalogue_control_report.terminal(signal)
    assigned = disposition(row, signal["signal_id"])
    return {
        "signal_id": signal["signal_id"],
        "triage": terminal,
        "gate_path": [
            {
                "stage": item["stage"],
                "decision": item["decision"],
                "reason": item["reason"],
            }
            for item in signal["triage"].get("history", [])
        ],
        "localization": signal.get(
            "localization", {"status": "NOT_REACHED", "quarters": {}}
        ),
        "automatic_disposition": assigned.get("disposition", "UNAVAILABLE"),
        "automatic_reason_codes": assigned.get("reason_codes", []),
        "risk_features": assigned.get("risk_features", {}),
    }


def pixel_support(source):
    pixels = list(source["pixels"].values())
    fractions = [float(pixel["prf_stamp_fraction"]) for pixel in pixels]
    return {
        "quarters": len(pixels),
        "resolved_parents": sorted({str(pixel["base_parent_kic"]) for pixel in pixels}),
        "off_stamp_quarters": sorted(
            int(quarter)
            for quarter, pixel in source["pixels"].items()
            if not pixel["target_inside_stamp"]
        ),
        "minimum_prf_stamp_fraction": min(fractions),
        "maximum_prf_stamp_fraction": max(fractions),
    }


def signal_metadata(source, signal, proposal, truth):
    supplied = next(
        (
            item
            for item in proposal["supplied_hypotheses"]
            if item["signal_id"] == signal["signal_id"]
        ),
        None,
    )
    periodic = supplied is not None
    origins = {
        event["original_alert"]["detector_origin"]
        for event in source["events"]
        if event["event_id"] in signal["event_ids"]
    }
    if len(origins) != 1 or not origins <= {"rust_v6_deterministic", "shallow_bls_v1"}:
        raise ValueError("Signal detector origin is missing or inconsistent")
    origin = origins.pop()
    if periodic and supplied["detector_origin"] != origin:
        raise ValueError("Signal detector origin differs from its supplied hypothesis")
    period = supplied["period_days"] if periodic else None
    return {
        "match_type": "PERIODIC" if periodic else "SPARSE",
        "detector_signal_id": signal["signal_id"],
        "detector_origin": origin,
        "classification": (
            supplied["rust_record"]["classification"]
            if origin == "rust_v6_deterministic" and periodic
            else "SHALLOW_BLS_PROPOSAL"
            if periodic
            else "EVENT_LIKE"
        ),
        "accepted_harmonic": catalogue_control_report.accepted_harmonic(
            period, truth["period_days"]
        )
        if periodic and truth["period_days"] is not None
        else None,
    }


def case(truth, proposal, source, pixel_source, events):
    signals = []
    for signal in source["signals"] if source else []:
        assigned = disposition(source, signal["signal_id"])
        fitted = fitted_event_units(source, signal, assigned)
        signals.append(
            {
                **signal_metadata(source, signal, proposal, truth),
                "signal_id": signal["signal_id"],
                "event_ids": signal["event_ids"],
                "proposal_event_match": match_event_units(
                    events, signal["events"], truth["duration_hours"]
                ),
                "fitted_units": fitted,
                "fitted_event_match": match_event_units(
                    events, fitted, truth["duration_hours"]
                )
                if fitted is not None
                else {"status": "UNAVAILABLE"},
                "downstream": downstream(source, signal),
            }
        )
    matches = [
        signal
        for signal in signals
        if signal["proposal_event_match"]["status"] == "MATCHED"
    ]
    return {
        "source_id": str(truth["source_id"]),
        "stratum": truth["stratum"],
        "expected_white_mes": truth["expected_white_mes"],
        "truth": {
            "period_days": truth["period_days"],
            "epoch_bkjd": truth["epoch_bkjd"],
            "duration_hours": truth["duration_hours"],
            "observed_events": truth["observed_events"],
            "category": truth["category"],
            "shape_period_days": truth["shape_period_days"],
            "ttv": truth["ttv"],
        },
        "truth_events": events,
        "signals": signals,
        "boi_catalogue": catalogue_control_report.emitted_candidate(
            source.get("disposition", {})
        )
        if source
        else False,
        "disposition_incomplete": catalogue_control_report.incomplete_disposition(
            source
        ),
        "scientific_abstention": catalogue_control_report.scientific_abstention(source),
        "pixel_support": pixel_support(pixel_source),
        "match": matches[0] if matches else None,
        "matches": matches,
    }


def case_matches(item):
    return item["matches"]


def exact_pc_recovered(item):
    return not item["disposition_incomplete"] and any(
        signal["downstream"]["automatic_disposition"] == "BOI"
        and signal["fitted_event_match"]["status"] == "MATCHED"
        for signal in item["signals"]
    )


def summarize(cases, boi_catalogue_enabled):
    matched = [item for item in cases if case_matches(item)]
    all_matches = [match for item in matched for match in case_matches(item)]
    periodic = [item for item in matched if item["match"]["match_type"] == "PERIODIC"]
    exact = [item for item in periodic if item["match"]["accepted_harmonic"] == 1]
    outcomes = Counter(
        item["match"]["downstream"]["triage"]["status"] for item in matched
    )
    stages = Counter(item["match"]["downstream"]["triage"]["stage"] for item in matched)
    reasons = Counter(
        item["match"]["downstream"]["triage"]["reason"] for item in matched
    )
    localization = Counter(
        item["match"]["downstream"]["localization"]["status"] for item in matched
    )
    retained = [
        item
        for item in matched
        if any(
            match["downstream"]["triage"]["status"] in {"ACTIVE", "HELD"}
            for match in case_matches(item)
        )
    ]
    active = [
        item
        for item in retained
        if any(
            match["downstream"]["triage"]["status"] == "ACTIVE"
            for match in case_matches(item)
        )
    ]
    matched_by_origin = Counter(
        origin
        for item in cases
        for origin in {match["detector_origin"] for match in case_matches(item)}
    )
    active_by_origin = Counter(
        origin
        for item in cases
        for origin in {
            match["detector_origin"]
            for match in case_matches(item)
            if match["downstream"]["triage"]["status"] == "ACTIVE"
        }
    )
    retained_by_origin = Counter(
        origin
        for item in cases
        for origin in {
            match["detector_origin"]
            for match in case_matches(item)
            if match["downstream"]["triage"]["status"] in {"ACTIVE", "HELD"}
        }
    )
    held_only = [item for item in retained if item not in active]
    total = len(cases)
    recovered = (
        sum(exact_pc_recovered(item) for item in cases)
        if boi_catalogue_enabled
        else None
    )
    return {
        "injections": total,
        "matched_detector_proposals": len(matched),
        "matching_detector_proposals": len(all_matches),
        "matched_fraction": len(matched) / total,
        "retained_injections": len(retained),
        "retained_fraction": len(retained) / total,
        "active_injections": len(active),
        "matched_injections_by_detector_origin": dict(matched_by_origin),
        "retained_injections_by_detector_origin": dict(retained_by_origin),
        "active_injections_by_detector_origin": dict(active_by_origin),
        "held_only_injections": len(held_only),
        "dismissed_only_injections": len(matched) - len(retained),
        "periodic_matches": len(periodic),
        "periodic_fraction": len(periodic) / total,
        "exact_period_matches": len(exact),
        "exact_period_fraction": len(exact) / total,
        "accepted_harmonic_matches": sum(
            item["match"]["accepted_harmonic"] not in {None, 1} for item in periodic
        ),
        "unclassified_period_matches": sum(
            item["match"]["accepted_harmonic"] is None for item in periodic
        ),
        "exact_signal_boi_catalogue_recovery": {
            "status": "MEASURED"
            if boi_catalogue_enabled
            else "NOT_MEASURED_BOI_CATALOGUE_DISABLED",
            "trials": total,
            "recovered": recovered,
            "fraction": recovered / total if recovered is not None else None,
        },
        "automatic_parent_outcomes": catalogue_control_report.parent_outcomes(cases)
        if boi_catalogue_enabled
        else {"status": "NOT_MEASURED_BOI_CATALOGUE_DISABLED"},
        "sparse_matches": len(matched) - len(periodic),
        "matched_classifications": dict(
            Counter(item["match"]["classification"] for item in matched)
        ),
        "matched_triage_outcomes": dict(outcomes),
        "matched_terminal_stages": dict(stages),
        "matched_terminal_reasons": dict(reasons),
        "matched_localization_status": dict(localization),
        "all_matching_triage_outcomes": dict(
            Counter(match["downstream"]["triage"]["status"] for match in all_matches)
        ),
    }


def validate_parent_audit(controls, audit_path, manifest):
    report = read(audit_path)
    repository = Path(report.get("repository_root", ""))
    if report.get("schema") != 2 or not repository.is_absolute():
        raise ValueError(
            "Parent audit schema 2 with an absolute repository root is required"
        )
    inputs = {
        (repository / path).resolve(): checksum
        for path, checksum in report.get("inputs", {}).items()
    }
    if len(inputs) != len(report.get("inputs", {})):
        raise ValueError("Parent audit has duplicate resolved input paths")
    expected = {
        source_id: sorted(
            {str(pixel["base_parent_kic"]) for pixel in item["pixels"].values()}
        )
        for source_id, item in manifest["sources"].items()
    }
    if (
        report.get("status") != "PARENT_INDEPENDENT"
        or (repository / report.get("materialization", "")).resolve()
        != controls.resolve()
        or report.get("source_parents") != expected
        or inputs.get((controls / "materialization.json").resolve())
        != digest(controls / "materialization.json")
        or inputs.get((controls / "input-manifest.json").resolve())
        != digest(controls / "input-manifest.json")
    ):
        raise ValueError("Resolved pixel-parent audit is absent, stale, or failed")
    return report


def validate_controls(controls, audit_path):
    report_path = controls / "materialization.json"
    report = read(report_path)
    protocols = {
        "fsc-kepler-pixel-planet-controls-v1": {"periodic_injection"},
        "fsc-kepler-pixel-morphology-controls-v2": {
            "sparse_injection",
            "ttv_injection",
        },
    }
    allowed_categories = protocols.get(report.get("protocol"))
    if report.get("status") != "FROZEN" or allowed_categories is None:
        raise ValueError("Pixel planet controls are not frozen")
    for relative, expected in report["artifacts"].items():
        if digest(controls / relative) != expected:
            raise ValueError(f"Frozen pixel-control artifact changed: {relative}")
    source_ids = list(map(str, read(controls / "source-ids.json")))
    truth = read_jsonl(controls / "truth.jsonl")
    manifest = read(controls / "input-manifest.json")
    truth_ids = [str(item["source_id"]) for item in truth]
    categories = {item["category"] for item in truth}
    category = next(iter(categories)) if len(categories) == 1 else None
    if (
        source_ids != truth_ids
        or len(source_ids) != len(set(source_ids))
        or set(source_ids) != set(manifest.get("sources", {}))
        or category is None
        or not categories <= allowed_categories
        or report.get("category", category) != category
    ):
        raise ValueError("Pixel-control truth, manifest, and source identities differ")
    parent_report = validate_parent_audit(controls, audit_path, manifest)
    return report, source_ids, truth, manifest, parent_report


def prepared_truth_events(truth, preparation):
    source_id = str(truth["source_id"])
    source = next(
        item for item in preparation["sources"] if str(item["source_id"]) == source_id
    )
    views = {}
    artifacts = {}
    for view, item in source["inputs"].items():
        path = root_path(item["fsc_path"])
        if digest(path) != item["fsc_sha256"]:
            raise ValueError("Prepared event-matching input changed")
        views[view] = detector_run.read_fsc(path, source_id, view)
        artifacts[path] = item["fsc_sha256"]
    if "psf_flat" not in views:
        raise ValueError("Planet control lacks its declared discovery view")
    return truth_event_ledger(truth, views), artifacts


def build_report(controls, run, audit_path):
    controls, run, audit_path = map(Path, (controls, run, audit_path))
    control_report, source_ids, truth, manifest, parent_report = validate_controls(
        controls, audit_path
    )
    preparation, detection, results, rows = catalogue_control_report.validate_run(
        run, source_ids
    )
    expected_manifest = (controls / "input-manifest.json").resolve()
    declared_manifest = preparation.get("input_manifest", {})
    if (
        root_path(declared_manifest.get("path", "")).resolve() != expected_manifest
        or declared_manifest.get("sha256") != digest(expected_manifest)
        or root_path(read(run / "vetting.json")["input_manifest"]).resolve()
        != expected_manifest
    ):
        raise ValueError("Planet replay physical input identity changed")
    boi_catalogue_enabled = results["disposition_execution"]["boi_catalogue_enabled"]
    cases = []
    prepared_artifacts = {}
    for item in truth:
        source_id = str(item["source_id"])
        proposal = read(run / "proposals" / f"{source_id}.json")
        source = rows.get(source_id)
        if (
            str(proposal.get("source_id")) != source_id
            or proposal["alerts"]
            and source is None
        ):
            raise ValueError(
                "Planet proposal identity or vetting evidence is incomplete"
            )
        events, paths = prepared_truth_events(item, preparation)
        prepared_artifacts.update(paths)
        cases.append(
            case(item, proposal, source, manifest["sources"][source_id], events)
        )
    timings = read(run / "vetting/stage-timings.json")
    stage_wall = {
        item["stage"]: item["wall_seconds"]
        for item in timings["stages"]
        if item.get("status") == "COMPLETE"
    }
    source_hashes = {
        path.name: digest(path)
        for path in sorted((run / "vetting/sources").glob("*.json"))
    }
    evidence_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy_path = root_path(read(run / "vetting.json")["disposition_policy"])
    policy = read(policy_path)
    artifacts = [
        controls / "materialization.json",
        controls / "input-manifest.json",
        controls / "truth.jsonl",
        controls / "source-ids.json",
        audit_path,
        run / "preparation.json",
        run / "detection.json",
        run / "discovery-manifest.json",
        run / "queue.json",
        run / "vetting/results.json",
        run / "vetting/manifest.json",
        run / "vetting/stage-timings.json",
        policy_path,
        Path(catalogue_control_report.__file__),
        Path(injections.__file__),
        Path(detector_run.__file__),
        Path(__file__).resolve().parents[1] / "runtime.py",
        Path(__file__),
    ]
    strata = {
        stratum: summarize(
            [item for item in cases if item["stratum"] == stratum], boi_catalogue_enabled
        )
        for stratum in sorted({item["stratum"] for item in cases})
    }
    return {
        "schema": 2,
        "outcome_contract": OUTCOME_CONTRACT,
        "event_matching_contract": EVENT_MATCHING_CONTRACT,
        "status": "COMPLETE_PLANET_CONTROL_REPORT",
        "claim_boundary": control_report["claim_boundary"],
        "population": {
            "declared_sources": len(source_ids),
            "prepared_sources": detection["sources"],
            "queued_sources": detection["alerted_sources"],
            "resolved_parents": parent_report["distinct_resolved_parents"],
        },
        "recovery": {
            "all": summarize(cases, boi_catalogue_enabled),
            "by_stratum": strata,
        },
        "vetting": {
            "processing_status": results["processing_status"],
            "boi_signals": results["boi_signals"],
            "unresolved_signals": results["unresolved_signals"],
            "disposition_policy": {
                "id": policy["policy_id"],
                "status": policy["status"],
                "boi_catalogue_enabled": results["disposition_execution"][
                    "boi_catalogue_enabled"
                ],
                "evaluation_role": results["disposition_execution"]["evaluation_role"],
                "production_boi_enabled": results["disposition_execution"][
                    "production_boi_enabled"
                ],
            },
        },
        "performance": {
            "recorded_stage_wall_seconds": stage_wall,
            "recorded_vetting_wall_seconds": sum(stage_wall.values()),
            "slowest_stage": max(stage_wall, key=stage_wall.get),
        },
        "cases": cases,
        "cases_sha256": catalogue_control_report.json_digest(cases),
        "inputs": {
            pixel_parent_audit.display_path(path): checksum
            for path, checksum in {
                **{path: digest(path) for path in artifacts},
                **prepared_artifacts,
            }.items()
        },
        "source_evidence": {
            "files": len(source_hashes),
            "digest_ledger_sha256": evidence_hash,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--parent-audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        root_path(args.controls), root_path(args.run), root_path(args.parent_audit)
    )
    pixel_parent_audit.write_json(root_path(args.out), report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
