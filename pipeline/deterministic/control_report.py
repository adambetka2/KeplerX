"""Summarize a parent-independent ordinary-control replay without tuning it."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.deterministic.catalogue_control_partition import parent_map_identities
from pipeline.deterministic.catalogue_control_report import (
    OUTCOME_CONTRACT,
    emitted_candidate,
    incomplete_disposition,
    opened_replay_scope,
    parent_outcomes,
    scientific_abstention,
    validate_execution,
)
from pipeline.runtime import artifact_path

PLAN = Path(__file__).with_name("control-evaluation-plan.json")


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def signal_kind(row, signal):
    events = {event["event_id"]: event for event in row["events"]}
    kinds = {
        events[event_id]["original_alert"]["kind"] for event_id in signal["event_ids"]
    }
    if kinds == {"PERIODIC_WINDOW"}:
        return "PERIODIC"
    if kinds == {"SPARSE_EVENT_WINDOW"}:
        return "SPARSE"
    return "+".join(sorted(kinds))


def increment(counter, *values):
    counter["|".join(str(value) for value in values)] += 1


def summarize_sources(source_paths):
    kinds = Counter()
    outcomes = Counter()
    stages = Counter()
    reasons = Counter()
    kind_outcomes = Counter()
    localization = Counter()
    signals = 0
    candidate_parents = 0
    unresolved_parents = 0
    for path in source_paths:
        row = read(path)
        disposition = row.get("disposition", {})
        candidate_parents += emitted_candidate(disposition)
        unresolved_parents += disposition.get("status") == "UNRESOLVED"
        for signal in row["signals"]:
            signals += 1
            kind = signal_kind(row, signal)
            triage = signal["triage"]
            status = triage["status"]
            stage = (
                triage.get("held_stage") or triage.get("dismissed_stage") or "survived"
            )
            kinds[kind] += 1
            outcomes[status] += 1
            stages[stage] += 1
            reasons[triage.get("reason", "NONE")] += 1
            increment(kind_outcomes, kind, status, triage.get("reason", "NONE"))
        for event in row.get("localization", {}).get("events", []):
            increment(
                localization,
                event["status"],
                event.get("attribution", {}).get("status", "NONE"),
            )
    return {
        "signals": signals,
        "signal_kinds": dict(kinds),
        "outcomes": dict(outcomes),
        "terminal_stages": dict(stages),
        "terminal_reasons": dict(reasons),
        "kind_outcomes": dict(kind_outcomes),
        "localization_events": dict(localization),
        "candidate_parents": candidate_parents,
        "unresolved_queued_parents": unresolved_parents,
    }


def trial_signal(row, signal):
    disposition = {
        item["signal_id"]: item
        for item in row.get("disposition", {}).get("signals", [])
    }.get(signal["signal_id"], {})
    triage = signal["triage"]
    return {
        "signal_id": signal["signal_id"],
        "kind": signal_kind(row, signal),
        "event_ids": signal["event_ids"],
        "triage_status": triage["status"],
        "terminal_stage": triage.get("held_stage")
        or triage.get("dismissed_stage")
        or "survived",
        "terminal_reason": triage.get("reason", "NONE"),
        "automatic_disposition": disposition.get("disposition", "UNAVAILABLE"),
        "automatic_reason_codes": disposition.get("reason_codes", []),
    }


def trial_ledger(run, source_ids, parents, rows, boi_catalogue_enabled):
    trials = []
    for source_id, parent in zip(source_ids, parents):
        proposal_path = run / "proposals" / f"{source_id}.json"
        if not proposal_path.is_file():
            raise ValueError("A declared control trial lacks detector evidence")
        proposal = read(proposal_path)
        if str(proposal.get("source_id")) != source_id:
            raise ValueError("Detector evidence source identity differs")
        classifications = Counter(
            row["classification"] for row in proposal.get("rust_rows", [])
        )
        source = rows.get(source_id)
        if source is None and proposal.get("alerts"):
            raise ValueError("A proposed control source lacks vetting evidence")
        disposition = source.get("disposition", {}) if source else {}
        signals = (
            [trial_signal(source, signal) for signal in source["signals"]]
            if source
            else []
        )
        trials.append(
            {
                "source_id": source_id,
                "parent_tpf": parent,
                "status": "VETTING_COMPLETE" if source else "DETECTOR_NO_PROPOSAL",
                "detector": {
                    "raw_rows": len(proposal.get("rust_rows", [])),
                    "alerts": len(proposal.get("alerts", [])),
                    "classifications": dict(classifications),
                },
                "signals": signals,
                "boi_catalogue": emitted_candidate(disposition)
                if boi_catalogue_enabled
                else None,
                "disposition_incomplete": incomplete_disposition(source)
                if boi_catalogue_enabled
                else None,
                "scientific_abstention": scientific_abstention(source)
                if boi_catalogue_enabled
                else None,
                "automatic_disposition_status": disposition.get(
                    "status", "NOT_APPLICABLE"
                ),
            }
        )
    return trials


def validate_evaluation_policy(run, partition):
    if partition != "evaluation":
        return
    policy = read(root_path(read(run / "vetting.json")["disposition_policy"]))
    if (
        policy.get("status") not in {"FROZEN_EVALUATION", "RELEASED"}
        or policy.get("boi_catalogue_enabled") is not True
    ):
        raise ValueError(
            "Ordinary evaluation requires a frozen automatic disposition policy"
        )


def validate_exposure_independence(plan, source_ids, measured):
    references = plan.get("independence", {}).get("exposure_inputs", [])
    if not references:
        raise ValueError("Ordinary evaluation requires declared exposure identities")
    selected = {
        "source_id": set(source_ids),
        "kic_id": {measured[sid][0] for sid in source_ids},
        "fits_sha256": {measured[sid][1] for sid in source_ids},
    }
    for reference in references:
        path = root_path(reference["path"])
        if digest(path) != reference["sha256"]:
            raise ValueError("Ordinary evaluation exposure identities changed")
        excluded, _ = parent_map_identities(path)
        if any(selected[name] & excluded[name] for name in selected):
            raise ValueError("Ordinary evaluation overlaps declared exposed identities")


def partition_execution(run, results, partition):
    execution = validate_execution(run, results, {})
    if partition == "evaluation" and (
        not execution["boi_catalogue_enabled"]
        or (
            execution["policy_status"] == "FROZEN_EVALUATION"
            and execution["evaluation_role"] != "final_replay"
        )
    ):
        raise ValueError("Ordinary evaluation requires final-evaluation execution")
    return execution


def frozen_population_identities(freeze_path, source_ids_path, source_ids_sha256):
    freeze = read(freeze_path)
    artifacts = freeze.get("artifacts", {})
    parent_map = freeze_path.parent / "parent-map.csv"
    if (
        source_ids_path.resolve().parent != freeze_path.resolve().parent
        or artifacts.get(source_ids_path.name) != source_ids_sha256
        or not parent_map.is_file()
        or digest(parent_map) != artifacts.get("parent-map.csv")
    ):
        raise ValueError(
            "Ordinary control identities are not bound to their partition freeze"
        )
    with parent_map.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = {
        row["source_id"]: (row["parent_tpf"], row.get("raw_fits_sha256"))
        for row in rows
    }
    if len(identities) != len(rows) or any(
        not parent or not raw for parent, raw in identities.values()
    ):
        raise ValueError(
            "Ordinary partition lacks unique parent and raw input identities"
        )
    return identities


def validate_population(run, plan, partition, scope=None):
    freeze_path = root_path(plan["parent_partition_freeze"]["path"])
    if digest(freeze_path) != plan["parent_partition_freeze"]["sha256"]:
        raise ValueError("Parent-partition freeze hash differs")
    specification = plan["partitions"][partition]
    source_ids_path = root_path(specification["source_ids"])
    if digest(source_ids_path) != specification["source_ids_sha256"]:
        raise ValueError("Control source-list hash differs")
    identities = frozen_population_identities(
        freeze_path, source_ids_path, specification["source_ids_sha256"]
    )
    expected = list(map(str, read(source_ids_path)))
    if scope is not None:
        expected = scope["selected_source_ids"]
    preparation = read(run / "preparation.json")
    prepared = [str(row["source_id"]) for row in preparation["sources"]]
    if len(prepared) != len(set(prepared)) or set(prepared) != set(expected):
        raise ValueError("Prepared sources differ from the frozen control partition")
    measured = {
        str(row["source_id"]): (str(row["parent_tpf"]), row.get("raw_fits_sha256"))
        for row in preparation["sources"]
    }
    if measured != {source_id: identities[source_id] for source_id in expected}:
        raise ValueError(
            "Prepared ordinary parents or raw FITS identities differ from their freeze"
        )
    parents = [measured[source_id][0] for source_id in expected]
    if len(parents) != len(set(parents)):
        raise ValueError("Prepared control sources are not parent-independent")
    if partition == "evaluation":
        validate_exposure_independence(plan, expected, measured)
    return expected, parents


def build_report(run, plan_path=PLAN, partition="development", opened_execution=None):
    run, plan_path = Path(run), Path(plan_path)
    plan = read(plan_path)
    if (
        plan.get("schema") != 3
        or plan.get("outcome_contract") != OUTCOME_CONTRACT
        or partition not in plan["partitions"]
    ):
        raise ValueError("Unsupported control-evaluation plan or partition")
    if (
        partition == "evaluation"
        and plan["partitions"][partition].get("status")
        != "DETECTOR_AND_VETTER_OUTCOMES_UNOPENED"
    ):
        raise ValueError(
            "Ordinary evaluation partition is not an unopened identity freeze"
        )
    validate_evaluation_policy(run, partition)
    scope = None
    if opened_execution is not None:
        if partition != "development":
            raise ValueError("Opened subset reporting requires ordinary development")
        original = list(
            map(str, read(root_path(plan["partitions"][partition]["source_ids"])))
        )
        scope = opened_replay_scope(opened_execution, "ordinary", run, original)
    expected, parents = validate_population(run, plan, partition, scope)
    detection = read(run / "detection.json")
    results = read(run / "vetting/results.json")
    execution = partition_execution(run, results, partition)
    timings = read(run / "vetting/stage-timings.json")
    source_paths = sorted((run / "vetting/sources").glob("*.json"))
    if (
        detection.get("status") != "DETECTION_COMPLETE"
        or results.get("processing_status") != "EVIDENCE_STAGES_COMPLETE"
        or detection.get("sources") != len(expected)
        or detection.get("alerted_sources") != len(source_paths)
        or results.get("sources") != len(source_paths)
    ):
        raise ValueError("Control replay is incomplete or internally inconsistent")
    stage_wall = {
        row["stage"]: row["wall_seconds"]
        for row in timings["stages"]
        if row.get("status") == "COMPLETE"
    }
    rows = {path.stem: read(path) for path in source_paths}
    if validate_execution(run, results, rows) != execution:
        raise ValueError(
            "Ordinary control disposition execution changed during reporting"
        )
    boi_catalogue_enabled = execution["boi_catalogue_enabled"]
    summary = summarize_sources(source_paths)
    candidate_incidence = {
        "status": "NOT_MEASURED_BOI_CATALOGUE_DISABLED",
        "boi_signals": results["boi_signals"],
        "interpretation": (
            "Proposal and triage rates are workload diagnostics, "
            "not automatic candidate incidence."
        ),
    }
    artifacts = [
        plan_path,
        run / "preparation.json",
        run / "detection.json",
        run / "discovery-manifest.json",
        run / "vetting/results.json",
        run / "vetting/manifest.json",
        run / "vetting/stage-timings.json",
        run / "vetting.json",
        root_path(read(run / "vetting.json")["disposition_policy"]),
        Path(__file__),
        Path(__file__).with_name("catalogue_control_report.py"),
        Path(__file__).resolve().parents[1] / "runtime.py",
    ]
    trials = trial_ledger(
        run,
        expected,
        parents,
        rows,
        boi_catalogue_enabled,
    )
    if boi_catalogue_enabled:
        candidate_incidence = parent_outcomes(
            trials, plan["ordinary_candidate_incidence"]["confidence"]
        )
    return {
        "schema": 3,
        "outcome_contract": OUTCOME_CONTRACT,
        "status": "PARTIAL_OPENED_CONTROL_REPLAY_REPORT"
        if scope
        else "COMPLETE_CONTROL_REPLAY_REPORT",
        "opened_execution_scope": scope,
        "claim_boundary": (
            "Executed opened subset of the original ordinary-parent ledger; "
            "protected metadata-only trials remain unmeasured."
            if scope
            else "Complete ordinary-parent ledger under the recorded disposition "
            "execution. Ordinary-field candidate incidence does not establish "
            "a false-BOI rate or catalogue reliability. Scientific abstentions "
            "are distinct from incomplete execution."
        ),
        "partition": partition,
        "population": {
            "declared_sources": scope["declared_sources"] if scope else len(expected),
            "prepared_sources": detection["sources"],
            "independent_parents": len(parents),
            "queued_sources": detection["alerted_sources"],
            "queued_parent_fraction": detection["alerted_sources"] / len(parents),
        },
        "detector": {
            "raw_rows": detection["raw_rows"],
            "alerts": detection["alerts"],
            "classifications": detection["classifications"],
            "binary_sha256": detection["binary"]["sha256"],
        },
        "vetting": {
            **summary,
            "processing_status": results["processing_status"],
            "disposition_execution": execution,
            "boi_signals": results["boi_signals"],
            "unresolved_signals": results["unresolved_signals"],
        },
        "candidate_incidence": candidate_incidence,
        "trial_ledger": {
            "unit": "independent_parent_tpf",
            "boi_catalogue_enabled": boi_catalogue_enabled,
            "trials": trials,
            "sha256": json_digest(trials),
        },
        "performance": {
            "recorded_stage_wall_seconds": stage_wall,
            "recorded_vetting_wall_seconds": sum(stage_wall.values()),
            "slowest_stage": max(stage_wall, key=stage_wall.get),
        },
        "inputs": {
            **{artifact_path(path, ROOT): digest(path) for path in artifacts},
            **(scope["inputs"] if scope else {}),
        },
        "control_plan_sha256": digest(plan_path),
        "source_evidence": {
            "files": len(source_paths),
            "digest_ledger_sha256": json_digest(
                {path.name: digest(path) for path in source_paths}
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument(
        "--partition",
        choices=("development", "evaluation"),
        default="development",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opened-execution", type=Path)
    args = parser.parse_args()
    report = build_report(
        root_path(args.run), args.plan, args.partition, args.opened_execution
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
