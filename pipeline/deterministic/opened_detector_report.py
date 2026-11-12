"""Measure supplied detector events on the original opened injection controls."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline"), str(ROOT / "pipeline/vetting")]

import vetting
import vetting_signals
from runtime import EVENT_MATCHING_CONTRACT, dump_path, implementation_hashes

from pipeline.deterministic import cloud, cloud_opened, planet_control_report

CONTRACT = "original-opened-detector-event-recovery-v1"
ENDPOINTS = ("ungated_alerts", "alerts")
FIXED_COHORTS = {name: row[2] for name, row in cloud_opened.TRUTH_COHORTS.items()}
DETECTORS = {"rust_v6_deterministic", "shallow_bls_v1"}


def bind_file(path, expected, dependencies):
    path = Path(path)
    if path in dependencies and dependencies[path] != expected:
        raise ValueError(f"Conflicting detector report dependency: {path}")
    dependencies[path] = expected


def read_bound(path, dependencies):
    path = Path(path)
    cloud_opened.parent_paths(path)
    payload = path.read_bytes()
    bind_file(path, hashlib.sha256(payload).hexdigest(), dependencies)
    return json.loads(payload)


def original_truth(sid, reference, dependencies):
    path = Path(reference["path"])
    bind_file(path, reference["file_sha256"], dependencies)
    cloud_opened.recheck({path: reference["file_sha256"]})
    lines = path.read_bytes().splitlines(keepends=True)
    index = reference["truth_index"]
    if type(index) is not int or not 0 <= index < len(lines):
        raise ValueError("Original truth row index is invalid")
    line = lines[index]
    if hashlib.sha256(line).hexdigest() != reference["line_sha256"]:
        raise ValueError("Original truth row bytes changed")
    truth = json.loads(line)
    cohort = reference["cohort"]
    if (
        cohort not in FIXED_COHORTS
        or truth["source_id"] != sid
        or truth["category"] != cloud_opened.TRUTH_COHORTS[cohort][1]
    ):
        raise ValueError("Original truth source or cohort differs")
    return truth


def detector_group_identity(alert):
    origin, signal = alert.get("detector_origin"), alert.get("detector_signal_id")
    if (
        not isinstance(origin, str)
        or origin not in DETECTORS
        or not isinstance(signal, str)
        or not signal
    ):
        raise ValueError("Detector grouping identity is missing or unsupported")
    return origin, signal


def alert_groups(alerts):
    groups, seen = {}, set()
    for alert in alerts:
        identity = alert.get("event_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Detector event IDs are missing or duplicated")
        seen.add(identity)
        if type(alert.get("quarter")) is not int or alert["quarter"] < 0:
            raise ValueError("Detector event quarter is invalid")
        groups.setdefault(detector_group_identity(alert), []).append(alert)
    return groups


def psf_supported(events):
    return {
        event["event_id"]
        for event in events
        if any(views.get("psf_flat", 0) > 0 for views in event["support"].values())
    }


def supported_matches(events, units, matching):
    truth = {event["event_id"]: event for event in events}
    quarters = {unit["unit_id"]: str(unit["quarter"]) for unit in units}
    return {
        match["truth_event_id"]
        for match in matching["matches"]
        if truth[match["truth_event_id"]]["support"][quarters[match["unit_id"]]].get(
            "psf_flat", 0
        )
        > 0
    }


def coverage(supported, matched):
    return {
        "supported_truth_events": len(supported),
        "matched_truth_events": len(matched),
        "fraction": len(matched) / len(supported) if supported else None,
        "supported_truth_event_ids": sorted(supported),
        "matched_truth_event_ids": sorted(matched),
    }


def detector_endpoint(alerts, views, events, duration_hours):
    groups, covered = [], set()
    for (origin, signal), supplied in alert_groups(alerts).items():
        specs = vetting.event_specs({"alerts": supplied, "views": views})
        units = vetting_signals.event_units(specs)
        matching = planet_control_report.match_event_units(
            events, units, duration_hours
        )
        matched = supported_matches(events, units, matching)
        covered.update(matched)
        groups.append(
            {
                "detector_origin": origin,
                "detector_signal_id": signal,
                "alerts": supplied,
                "units": units,
                "event_match": matching,
                "psf_flat_matched_truth_event_ids": sorted(matched),
            }
        )
    return {
        "alert_count": len(alerts),
        "alert_order": [alert["event_id"] for alert in alerts],
        "group_count": len(groups),
        "completely_matched_groups": sum(
            group["event_match"]["status"] == "MATCHED" for group in groups
        ),
        "any_completely_matched_group": any(
            group["event_match"]["status"] == "MATCHED" for group in groups
        ),
        "psf_flat_event_coverage": coverage(psf_supported(events), covered),
        "groups": groups,
    }


def check_detector_rows(proposal, ledger, sid):
    for name in ("raw_rows", "shallow_rows"):
        rows = proposal["rust_rows" if name == "raw_rows" else name]
        if len(rows) != ledger[name] or any(
            row.get("source_id") != sid for row in rows
        ):
            raise ValueError("Detector row source or proposal counts differ")


def check_eligibility(proposal, ledger, eligibility):
    if (
        proposal["shallow_eligibility"] != ledger["shallow_eligibility"]
        or proposal["shallow_eligibility"] != eligibility
        or eligibility["status"] not in {"ELIGIBLE", "INELIGIBLE"}
        or len(proposal["alerts"]) != ledger["alerts"]
    ):
        raise ValueError("Detector eligibility or queue counts differ")


def check_source(proposal, prepared, ledger, eligibility):
    sid = prepared["source_id"]
    if proposal.get("source_id") != sid or proposal.get("views") != prepared["views"]:
        raise ValueError("Detector source or prepared views differ")
    check_detector_rows(proposal, ledger, sid)
    check_eligibility(proposal, ledger, eligibility)
    expected = "PROPOSALS" if proposal["alerts"] else "DETECTOR_REJECTION"
    if ledger["status"] != expected:
        raise ValueError("Detector source status differs from its queue alerts")
    ungated = {alert["event_id"]: alert for alert in proposal["ungated_alerts"]}
    for alert in proposal["alerts"]:
        if ungated.get(alert["event_id"]) != alert:
            raise ValueError("Queue alert is not an unchanged ungated alert")


def source_record(
    sid, reference, prepared, proposal, ledger, eligibility, preparation, dependencies
):
    for name in ENDPOINTS:
        alert_groups(proposal[name])
    check_source(proposal, prepared, ledger, eligibility)
    truth = original_truth(sid, reference, dependencies)
    events, artifacts = planet_control_report.prepared_truth_events(truth, preparation)
    for path, expected in artifacts.items():
        bind_file(path, expected, dependencies)
    return {
        "source_id": sid,
        "cohort": reference["cohort"],
        "truth_reference": reference,
        "original_truth": truth,
        "all_view_truth_event_ledger": events,
        "detector_accounting": {
            "rust_rows": len(proposal["rust_rows"]),
            "shallow_rows": len(proposal["shallow_rows"]),
            "shallow_eligibility": eligibility,
            "source_status": ledger["status"],
            "rust_classifications": dict(
                Counter(row["classification"] for row in proposal["rust_rows"])
            ),
        },
        "endpoints": {
            name: detector_endpoint(
                proposal[name], prepared["views"], events, truth["duration_hours"]
            )
            for name in ENDPOINTS
        },
    }


def endpoint_summary(cases, name):
    rows = [case["endpoints"][name] for case in cases]
    supported = sum(
        row["psf_flat_event_coverage"]["supported_truth_events"] for row in rows
    )
    matched = sum(
        row["psf_flat_event_coverage"]["matched_truth_events"] for row in rows
    )
    return {
        "measured_sources": len(cases),
        "any_completely_matched_group_sources": sum(
            row["any_completely_matched_group"] for row in rows
        ),
        "alert_count": sum(row["alert_count"] for row in rows),
        "group_count": sum(row["group_count"] for row in rows),
        "completely_matched_groups": sum(
            row["completely_matched_groups"] for row in rows
        ),
        "psf_flat_event_coverage": {
            "supported_truth_events": supported,
            "matched_truth_events": matched,
            "fraction": matched / supported if supported else None,
        },
    }


def cohort_summary(cases, cohort, count):
    selected = [case for case in cases if case["cohort"] == cohort]
    return {
        "fixed_sources": count,
        "unmeasured_sources": count - len(selected),
        "endpoints": {name: endpoint_summary(selected, name) for name in ENDPOINTS},
    }


def summary(cases):
    return {
        "endpoints": {name: endpoint_summary(cases, name) for name in ENDPOINTS},
        "by_cohort": {
            name: cohort_summary(cases, name, count)
            for name, count in FIXED_COHORTS.items()
        },
        "rust_rows": sum(case["detector_accounting"]["rust_rows"] for case in cases),
        "shallow_rows": sum(
            case["detector_accounting"]["shallow_rows"] for case in cases
        ),
        "shallow_eligibility": dict(
            Counter(
                case["detector_accounting"]["shallow_eligibility"]["status"]
                for case in cases
            )
        ),
        "detector_source_status": dict(
            Counter(case["detector_accounting"]["source_status"] for case in cases)
        ),
    }


def discovery_dependencies(run, preparation, dependencies):
    manifest = read_bound(run / "discovery-manifest.json", dependencies)
    detection = read_bound(run / "detection.json", dependencies)
    if (
        manifest.get("implementation") != preparation["implementation"]
        or manifest["implementation"] != implementation_hashes()
    ):
        raise ValueError(
            "Detector implementation differs from preparation or current code"
        )
    for path, expected in manifest["artifacts"].items():
        bind_file(Path(path), expected, dependencies)
    bind_file(
        Path(detection["binary"]["path"]), detection["binary"]["sha256"], dependencies
    )
    policy = ROOT / preparation["disposition_policy"]
    bind_file(policy, preparation["disposition_policy_sha256"], dependencies)
    cloud_opened.recheck(dependencies)
    return detection


def check_queue(run, ledger, detection, dependencies):
    queue = read_bound(run / "queue.json", dependencies)
    expected = [
        {
            "source_id": row["source_id"],
            "evidence": f"proposals/{row['source_id']}.json",
        }
        for row in ledger
        if row["alerts"]
    ]
    if queue != {"discovery_manifest": "discovery-manifest.json", "rows": expected}:
        raise ValueError("Detector queue differs from its complete source ledger")
    if detection.get("alerted_sources") != len(expected) or detection.get(
        "alerts"
    ) != sum(row["alerts"] for row in ledger):
        raise ValueError("Detector queue summary counts differ")


def report_metadata(run, cases, cohort_counts):
    return {
        "schema": 1,
        "contract": CONTRACT,
        "status": "OPENED_DETECTOR_EVENTS_MEASURED",
        "run": str(run),
        "role": "development",
        "event_matching_contract": EVENT_MATCHING_CONTRACT,
        **{flag: False for flag in cloud_opened.FLAGS},
        "scope": "Supplied detector event recovery only; no fitted signal or automatic candidate recovery is measured.",
        "endpoint_definitions": {
            "ungated_alerts": "All supplied emitted detector alerts",
            "alerts": "Unchanged supplied alerts accepted into the follow-up queue",
        },
        "downstream": {
            name: "NOT_MEASURED"
            for name in (
                "triage_retention",
                "fitted_timing",
                "pixels",
                "final_disposition",
            )
        },
        "original_opened_inventory_counts": cloud_opened.COUNTS,
        "population": {
            "original_opened_sources": cloud_opened.COUNTS["original"],
            "supported_opened_sources": cloud_opened.COUNTS["supported"],
            "fixed_injected_sources": sum(FIXED_COHORTS.values()),
            "fixed_cohorts": FIXED_COHORTS,
            "measured_sources": len(cases),
            "unmeasured_fixed_injected_sources": sum(FIXED_COHORTS.values())
            - len(cases),
            "full_fixed_cohort_measured": cohort_counts == Counter(FIXED_COHORTS),
        },
    }


def build_report(run):
    run = dump_path(Path(run))
    preparation = cloud.complete_run(run, False)
    references, dependencies = cloud_opened.truth_refs(run)
    ids = preparation["selection"]["source_ids"]
    if not references or set(references) != set(ids):
        raise ValueError(
            "Detector report requires original injected truth refs for every prepared source"
        )
    detection = discovery_dependencies(run, preparation, dependencies)
    ledger = read_bound(run / "source-ledger.json", dependencies)
    check_queue(run, ledger, detection, dependencies)
    eligibility = {
        row["source_id"]: row for row in cloud.csv_rows(run / "shallow-eligibility.csv")
    }
    cases = []
    for prepared, row in zip(preparation["sources"], ledger, strict=True):
        sid = prepared["source_id"]
        proposal = read_bound(run / "proposals" / f"{sid}.json", dependencies)
        cases.append(
            source_record(
                sid,
                references[sid],
                prepared,
                proposal,
                row,
                eligibility[sid],
                preparation,
                dependencies,
            )
        )
    cohort_counts = Counter(case["cohort"] for case in cases)
    if any(cohort_counts[name] > count for name, count in FIXED_COHORTS.items()):
        raise ValueError("Detector report exceeds a fixed opened cohort")
    value = {
        **report_metadata(run, cases, cohort_counts),
        "summary": summary(cases),
        "cases": cases,
        "dependencies": {
            str(path): {"sha256": expected, "bytes": Path(path).stat().st_size}
            for path, expected in sorted(dependencies.items())
        },
    }
    cloud_opened.recheck(dependencies)
    return value


def write_report(run, output):
    output = dump_path(Path(output))
    if output.exists():
        raise ValueError("Detector report output already exists")
    report = build_report(run)
    dependencies = {
        Path(path): row["sha256"] for path, row in report["dependencies"].items()
    }
    cloud_opened.recheck(dependencies)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    identity = {
        "path": str(output),
        "bytes": partial.stat().st_size,
        "sha256": cloud.digest(partial),
    }
    cloud_opened.recheck({**dependencies, partial: identity["sha256"]})
    partial.rename(output)
    try:
        cloud_opened.recheck({**dependencies, output: identity["sha256"]})
    except Exception:
        output.rename(partial)
        raise
    return {
        "status": report["status"],
        "output": identity,
        "population": report["population"],
        "summary": report["summary"],
        "scientific_acceptance": False,
        "boi_catalogue_enabled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_report(args.run, args.out), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
