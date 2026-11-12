"""Reconcile frozen catalogue controls through detector and vetter outcomes."""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.runtime import (
    DISPOSITION_LABELS,
    OUTCOME_CONTRACT,
    artifact_path,
    disposition_definition_sha256,
    disposition_execution,
)

CADENCE_DAYS = 1 / 48.94
CONTROL_CONFIDENCE = 0.9875
NATIVE_EB_PROTOCOL = "fsc-kepler-catalogue-eb-controls-v1"
NATIVE_EB_CATEGORY = "catalogue_eclipsing_binary"


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def opened_scope_graph(binding):
    from pipeline.deterministic import cloud
    from pipeline.deterministic.control_partitions import joint_component_index

    specification = Path(cloud.SCOPE_SPECIFICATION)
    packed = Path(cloud.SCOPE_BUNDLE)
    expected = binding["current_graph"]
    if (
        digest(specification) != expected["specification"]["sha256"]
        or digest(packed) != expected["packed_graph"]["sha256"]
    ):
        raise ValueError("Opened execution component graph changed")
    spec = read(specification)
    artifact = spec["artifact"]
    with gzip.open(packed, "rb") as stream:
        payload = stream.read(artifact["canonical_json_bytes"] + 1)
    if (
        len(payload) != artifact["canonical_json_bytes"]
        or hashlib.sha256(payload).hexdigest() != expected["canonical_json_sha256"]
        or expected["canonical_json_sha256"] != artifact["canonical_json_sha256"]
    ):
        raise ValueError("Opened execution decoded component graph changed")
    graph = json.loads(payload)
    cloud.check_scope_metadata(graph, spec)
    trials, _ = joint_component_index(graph)
    stopped = {
        row["full_witness_component_id"]
        for row in trials.values()
        if row["role"] in {"evaluation", "holdout"}
    }
    return trials, stopped


def opened_scope_rows(scope, trials, stopped, expected):
    rows = scope["metadata_ledger"]
    original = [str(row["source_id"]) for row in rows]
    if original != expected or len(original) != len(set(original)):
        raise ValueError("Opened reporting changed the original cohort ledger")
    fields = (
        "source_id",
        "cohort",
        "origin_parent",
        "raw_fits_sha256",
        "full_witness_component_id",
        "role",
        "scope",
    )
    selected, metadata_only = [], []
    for row in rows:
        frozen = trials[row["trial_id"]]
        if any(row[key] != frozen[key] for key in fields):
            raise ValueError("Opened reporting changed a frozen trial identity")
        blocked = row["full_witness_component_id"] in stopped
        if blocked != (row["input_status"] == "EXCLUDED_STOPPED_SHARED_COMPONENT"):
            raise ValueError("Opened reporting changed a stopped component decision")
        if blocked:
            metadata_only.append(
                {
                    **{key: row[key] for key in fields},
                    "trial_id": row["trial_id"],
                    "status": "METADATA_ONLY_PROTECTED_COMPONENT",
                    "execution_complete": False,
                    "boi_catalogue": None,
                }
            )
        else:
            selected.append(str(row["source_id"]))
    return selected, metadata_only


def opened_replay_scope(path, cohort, run, expected):
    from pipeline.deterministic.control_bindings import verified_reference

    path = Path(path)
    execution = read(path)
    declaration = execution["cohorts"][cohort]
    scope_path = verified_reference(declaration["scope_ledger"])
    binding_path = verified_reference(execution["cloud_role_reconciliation"])
    source_ids_path = verified_reference(declaration["source_ids"])
    scope, binding = read(scope_path), read(binding_path)
    policy = read(root_path(read(run / "vetting.json")["disposition_policy"]))
    if (
        execution["status"] != "REVIEWED_OPENED_DEVELOPMENT_EXECUTION"
        or execution["sealed_roles_authorized"] is not False
        or execution["final_code_freeze"] is not False
        or scope["scope"] != "CURRENT_OPENED_DEVELOPMENT"
        or scope["role"] != "development"
        or scope["cohort"] != cohort.replace("-", "_")
        or policy.get("status") != "DEVELOPMENT"
        or policy.get("boi_catalogue_enabled") is not False
        or Path(declaration["run"]).resolve() != run.resolve()
    ):
        raise ValueError("Opened subset reporting requires the bound development run")
    if (
        declaration["scope_ledger"]["sha256"]
        not in {row["sha256"] for row in binding["scopes"]}
        or declaration["source_ids"]["sha256"] != scope["source_ids"]["sha256"]
        or read(run / "preparation.json")["implementation"]
        != execution["implementation"]
    ):
        raise ValueError("Opened reporting execution or scope pin changed")
    trials, stopped = opened_scope_graph(binding)
    selected, metadata_only = opened_scope_rows(scope, trials, stopped, expected)
    if (
        selected != list(map(str, read(source_ids_path)))
        or set(declaration["metadata_only_excluded"])
        != {row["source_id"] for row in metadata_only}
        or len(expected) != declaration["declared_sources"]
        or len(selected) != declaration["selected_sources"]
    ):
        raise ValueError("Opened reporting omitted or added an executable source")
    return {
        "selected_source_ids": selected,
        "declared_sources": len(expected),
        "metadata_only_trials": metadata_only,
        "complete_declared_execution": not metadata_only,
        "final_scoped_role_authorization": False,
        "inputs": {
            str(item): digest(item)
            for item in (path, scope_path, binding_path, source_ids_path)
        },
    }


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def accepted_harmonic(period, reference):
    if abs(period / reference - 1) <= 0.01:
        return 1
    for harmonic in range(2, 9):
        if abs(period / (reference * harmonic) - 1) <= 0.01:
            return harmonic
        if abs(period / (reference / harmonic) - 1) <= 0.01:
            return -harmonic
    return None


def phase_delta(time, epoch, period):
    return (time - epoch + period * 0.5) % period - period * 0.5


def periodic_option(signal, hypothesis):
    if signal["koi_period"] is None:
        return None
    reference = float(signal["koi_period"])
    period = float(hypothesis["period_days"])
    harmonic = accepted_harmonic(period, reference)
    if harmonic is None or not hypothesis.get("followup_eligible"):
        return None
    duration = float(hypothesis["duration_days"])
    truth_duration = float(signal["koi_duration"]) / 24
    phase_error = abs(
        phase_delta(
            float(hypothesis["epoch_bkjd"]),
            float(signal["koi_time0bk"]),
            min(period, reference),
        )
    )
    ttv_tolerance = float(signal.get("ttv_amplitude_days") or 0.0)
    timing_tolerance = (
        0.5 * max(duration, truth_duration) + CADENCE_DAYS + ttv_tolerance
    )
    if phase_error > timing_tolerance:
        return None
    target = reference * harmonic if harmonic > 0 else reference / -harmonic
    period_error = abs(period / target - 1)
    origin = hypothesis["detector_origin"]
    if origin == "rust_v6_deterministic":
        classification = hypothesis["rust_record"]["classification"]
    elif origin == "shallow_bls_v1":
        classification = "SHALLOW_BLS_PROPOSAL"
    else:
        raise ValueError(f"Unknown detector origin: {origin}")
    return {
        "key": f"periodic:{hypothesis['signal_id']}",
        "match_type": "PERIODIC",
        "detector_signal_id": hypothesis["signal_id"],
        "detector_origin": origin,
        "detector_followup_lane": hypothesis["detector_followup_lane"],
        "classification": classification,
        "accepted_harmonic": harmonic,
        "period_days": period,
        "epoch_bkjd": float(hypothesis["epoch_bkjd"]),
        "duration_hours": duration * 24,
        "period_error_fraction": period_error,
        "phase_error_days": phase_error,
        "timing_tolerance_days": timing_tolerance,
        "ttv_tolerance_days": ttv_tolerance,
        "sort": (0, harmonic != 1, period_error, phase_error),
    }


def sparse_option(signal, alert):
    if alert.get("kind") != "SPARSE_EVENT_WINDOW":
        return None
    start, end = float(alert["start_bkjd"]), float(alert["end_bkjd"])
    centre = (start + end) / 2
    duration = end - start
    truth_duration = float(signal["koi_duration"]) / 24
    epoch = float(signal["koi_time0bk"])
    phase_error = (
        abs(centre - epoch)
        if signal["koi_period"] is None
        else abs(phase_delta(centre, epoch, float(signal["koi_period"])))
    )
    ttv_tolerance = float(signal.get("ttv_amplitude_days") or 0.0)
    timing_tolerance = (
        0.5 * max(duration, truth_duration) + CADENCE_DAYS + ttv_tolerance
    )
    if phase_error > timing_tolerance:
        return None
    detector_signal_id = alert["detector_signal_id"]
    return {
        "key": f"sparse:{detector_signal_id}",
        "match_type": "SPARSE",
        "detector_signal_id": detector_signal_id,
        "detector_origin": alert["detector_origin"],
        "detector_followup_lane": "SPARSE_EVENT",
        "classification": "EVENT_LIKE",
        "accepted_harmonic": None,
        "period_days": None,
        "epoch_bkjd": centre,
        "duration_hours": duration * 24,
        "period_error_fraction": None,
        "phase_error_days": phase_error,
        "timing_tolerance_days": timing_tolerance,
        "ttv_tolerance_days": ttv_tolerance,
        "sort": (1, False, 0, phase_error),
    }


def match_options(signal, proposals):
    options = [
        periodic_option(signal, row) for row in proposals.get("supplied_hypotheses", [])
    ]
    options.extend(sparse_option(signal, row) for row in proposals.get("alerts", []))
    return sorted(
        (row for row in options if row is not None), key=lambda row: row["sort"]
    )


def assign_truth_matches(signals, proposals):
    options = [match_options(signal, proposals) for signal in signals]
    assignment = {}
    selected = {}

    def assign(index, seen):
        for option in options[index]:
            key = option["key"]
            if key in seen:
                continue
            seen.add(key)
            previous = assignment.get(key)
            if previous is None or assign(previous, seen):
                assignment[key] = index
                selected[index] = option
                return True
        return False

    for index in sorted(
        range(len(signals)), key=lambda value: (len(options[value]), value)
    ):
        assign(index, set())
    return selected


def terminal(signal):
    triage = signal["triage"]
    history = triage.get("history", [])
    shape = next(
        (
            row.get("evidence", {}).get("score")
            for row in history
            if row["stage"] == "shape"
        ),
        None,
    )
    return {
        "status": triage["status"],
        "stage": triage.get("dismissed_stage")
        or triage.get("held_stage")
        or "survived",
        "reason": triage.get("reason", "NONE"),
        "shape_score": shape,
    }


def detector_ids(row, signal):
    events = {event["event_id"]: event for event in row["events"]}
    return {
        events[event_id]["original_alert"]["detector_signal_id"]
        for event_id in signal["event_ids"]
    }


def downstream_match(row, match):
    if row is None:
        raise ValueError("Matched detector proposal is absent from vetting evidence")
    matches = [
        signal
        for signal in row["signals"]
        if match["detector_signal_id"] in detector_ids(row, signal)
    ]
    if len(matches) != 1:
        raise ValueError("Detector proposal does not map to one vetted signal")
    signal = matches[0]
    dispositions = {
        item["signal_id"]: item for item in row["disposition"].get("signals", [])
    }
    disposition = dispositions.get(signal["signal_id"], {})
    return {
        "signal_id": signal["signal_id"],
        "triage": terminal(signal),
        "automatic_disposition": disposition.get("disposition", "UNAVAILABLE"),
        "automatic_reason_codes": disposition.get("reason_codes", []),
        "evidence_assessment": disposition.get("evidence_assessment", {}).get(
            "status", "UNAVAILABLE"
        ),
    }


def summarize_cases(cases):
    by_category = {}
    for category in sorted({row["category"] for row in cases}):
        rows = [row for row in cases if row["category"] == category]
        matched = [row for row in rows if row["match"] is not None]
        outcomes = Counter(
            row["match"]["downstream"]["triage"]["status"] for row in matched
        )
        reasons = Counter(
            row["match"]["downstream"]["triage"]["reason"] for row in matched
        )
        by_category[category] = {
            "signals": len(rows),
            "matched_detector_proposals": len(matched),
            "periodic_matches": sum(
                row["match"]["match_type"] == "PERIODIC" for row in matched
            ),
            "sparse_matches": sum(
                row["match"]["match_type"] == "SPARSE" for row in matched
            ),
            "matched_fraction": len(matched) / len(rows),
            "matched_triage_outcomes": dict(outcomes),
            "matched_terminal_reasons": dict(reasons),
        }
    return by_category


def summarize_all_signals(rows, categories):
    summaries = {}
    for category in sorted(set(categories.values())):
        sources = [
            row for source_id, row in rows.items() if categories[source_id] == category
        ]
        signals = [signal for row in sources for signal in row["signals"]]
        outcomes = Counter(terminal(signal)["status"] for signal in signals)
        reasons = Counter(terminal(signal)["reason"] for signal in signals)
        summaries[category] = {
            "queued_sources": len(sources),
            "signals": len(signals),
            "triage_outcomes": dict(outcomes),
            "terminal_reasons": dict(reasons),
            "boi_catalogue_sources": sum(
                bool(row["disposition"].get("boi_signal_count"))
                for row in sources
            ),
            "unresolved_sources": sum(
                row["disposition"].get("status") == "UNRESOLVED" for row in sources
            ),
        }
    return summaries


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


def trial_signal(row, signal):
    dispositions = {
        item["signal_id"]: item
        for item in row.get("disposition", {}).get("signals", [])
    }
    disposition = dispositions.get(signal["signal_id"], {})
    outcome = terminal(signal)
    return {
        "signal_id": signal["signal_id"],
        "kind": signal_kind(row, signal),
        "event_ids": signal["event_ids"],
        "triage_status": outcome["status"],
        "terminal_stage": outcome["stage"],
        "terminal_reason": outcome["reason"],
        "automatic_disposition": disposition.get("disposition", "UNAVAILABLE"),
        "automatic_reason_codes": disposition.get("reason_codes", []),
    }


def emitted_candidate(disposition):
    count = disposition.get("boi_signal_count")
    return (type(count) is int and count > 0) or any(
        item.get("disposition") == "BOI"
        for item in (
            *disposition.get("signals", []),
            *disposition.get("attributed_signal_reviews", []),
        )
    )


def consistent_candidate_count(disposition):
    count = disposition.get("boi_signal_count")
    if type(count) is not int or count < 0:
        return False
    candidates = {
        item.get("signal_id")
        for item in (
            *disposition.get("signals", []),
            *disposition.get("attributed_signal_reviews", []),
        )
        if item.get("disposition") == "BOI"
    }
    return (
        None not in candidates
        and count == len(candidates)
        and (disposition.get("status") == "BOI") == bool(candidates)
    )


def incomplete_disposition(source):
    if source is None:
        return False
    disposition = source.get("disposition", {})
    signals = [item.get("signal_id") for item in source.get("signals", [])]
    assigned = disposition.get("signals", [])
    assigned_ids = [item.get("signal_id") for item in assigned]
    return (
        source.get("processing_status") != "EVIDENCE_STAGES_COMPLETE"
        or not signals
        or not all(signals)
        or len(signals) != len(set(signals))
        or len(assigned_ids) != len(set(assigned_ids))
        or set(signals) != set(assigned_ids)
        or any(item.get("disposition") not in DISPOSITION_LABELS for item in assigned)
        or disposition.get("status") not in DISPOSITION_LABELS
        or not consistent_candidate_count(disposition)
    )


def scientific_abstention(source):
    if source is None or incomplete_disposition(source):
        return False
    return source["disposition"]["status"] == "UNRESOLVED" or any(
        item["disposition"] == "UNRESOLVED" for item in source["disposition"]["signals"]
    )


def parent_outcomes(trials, confidence=CONTROL_CONFIDENCE):
    if not trials:
        raise ValueError("Outcome measurement requires declared parent trials")
    candidates = sum(trial["boi_catalogue"] is True for trial in trials)
    failures = sum(trial["disposition_incomplete"] is True for trial in trials)
    abstentions = sum(trial["scientific_abstention"] is True for trial in trials)
    return {
        "status": "INCOMPLETE_EXECUTION" if failures else "MEASURED",
        "parent_trials": len(trials),
        "boi_catalogue_parents": candidates,
        "scientific_abstention_parents": abstentions,
        "execution_failure_parents": failures,
        "completed_parent_fraction": (len(trials) - failures) / len(trials),
        "fraction": candidates / len(trials) if not failures else None,
        "upper_bound": exact_upper_bound(candidates, len(trials), confidence)
        if not failures
        else None,
        "confidence": confidence,
    }


def trial_ledger(run, truth, rows, cases, boi_catalogue_enabled):
    cases_by_source = {}
    for case in cases:
        cases_by_source.setdefault(case["source_id"], []).append(
            {
                "catalogue_record": case["catalogue_record"],
                "matching_status": case["matching_status"],
            }
            if case.get("category") == NATIVE_EB_CATEGORY
            else {"kepoi_name": case["kepoi_name"], "match": case["match"]}
        )
    trials = []
    for item in truth:
        source_id = str(item["source_id"])
        proposal_path = run / "proposals" / f"{source_id}.json"
        if not proposal_path.is_file():
            raise ValueError("A declared catalogue trial lacks detector evidence")
        proposal = read(proposal_path)
        if str(proposal.get("source_id")) != source_id:
            raise ValueError("Detector evidence source identity differs")
        classifications = Counter(
            row["classification"] for row in proposal.get("rust_rows", [])
        )
        shallow_rows = proposal.get("shallow_rows", [])
        source = rows.get(source_id)
        if source is None and proposal.get("alerts"):
            raise ValueError("A proposed catalogue source lacks vetting evidence")
        disposition = source.get("disposition", {}) if source else {}
        trials.append(
            {
                "source_id": source_id,
                "parent_tpf": str(item["kic_id"]),
                "category": item["category"],
                "pool_rank": item["pool_rank"],
                "status": "VETTING_COMPLETE" if source else "DETECTOR_NO_PROPOSAL",
                "detector": {
                    "raw_rows": len(proposal.get("rust_rows", [])),
                    "shallow_rows": len(shallow_rows),
                    "shallow_eligibility": proposal.get("shallow_eligibility"),
                    "alerts": len(proposal.get("alerts", [])),
                    "classifications": dict(classifications),
                },
                "catalogue_signals": cases_by_source.get(source_id, []),
                "signals": [
                    trial_signal(source, signal) for signal in source["signals"]
                ]
                if source
                else [],
                "boi_catalogue": (
                    emitted_candidate(disposition) if boi_catalogue_enabled else None
                ),
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


def exact_upper_bound(outcomes, trials, confidence=CONTROL_CONFIDENCE):
    if outcomes == trials:
        return 1.0
    return float(beta.ppf(confidence, outcomes + 1, trials - outcomes))


def validate_controls(controls, partition=None):
    freeze = read(controls / "freeze.json")
    protocol = freeze.get("protocol")
    native = protocol == NATIVE_EB_PROTOCOL
    statuses = (
        {"FROZEN_CATEGORY_INPUTS"}
        if native
        else {"PROBE_COMPLETE", "FROZEN_INPUTS", "FROZEN_CATEGORY_INPUTS"}
    )
    if (
        protocol not in {"fsc-kepler-catalogue-controls-v2", NATIVE_EB_PROTOCOL}
        or freeze.get("status") not in statuses
    ):
        raise ValueError("Catalogue controls are not frozen or probe-complete")
    if not {"source-ids.json", "truth.jsonl", "input-manifest.json"} <= set(
        freeze["artifacts"]
    ):
        raise ValueError("Catalogue freeze omits required input artifacts")
    for relative, expected in freeze["artifacts"].items():
        if digest(controls / relative) != expected:
            raise ValueError(f"Catalogue-control artifact changed: {relative}")
    source_ids = list(map(str, read(controls / "source-ids.json")))
    truth = read_jsonl(controls / "truth.jsonl")
    truth_ids = [str(row["source_id"]) for row in truth]
    parents = [str(row["kic_id"]) for row in truth]
    if (
        len(source_ids) != len(set(source_ids))
        or source_ids != truth_ids
        or len(parents) != len(set(parents))
    ):
        raise ValueError("Catalogue-control truth and source identities differ")
    if freeze["status"] == "FROZEN_CATEGORY_INPUTS":
        validate_category_inputs(controls, freeze, truth)
    if native:
        source_ids, truth = native_partition(controls, freeze, truth, partition)
    elif partition is not None:
        raise ValueError("Catalogue partition reporting requires native EB controls")
    return freeze, source_ids, truth


def validate_category_inputs(controls, freeze, truth):
    counts = dict(Counter(row["category"] for row in truth))
    categories = freeze.get("categories")
    if (
        not counts
        or counts != freeze.get("selected")
        or not isinstance(categories, list)
        or len(categories) != len(set(categories))
        or set(categories) != set(counts)
        or not set(counts)
        <= (
            {NATIVE_EB_CATEGORY}
            if freeze["protocol"] == NATIVE_EB_PROTOCOL
            else {"catalogue_candidate", "catalogue_false_positive"}
        )
    ):
        raise ValueError("Catalogue category input counts or classes differ")
    manifest = read(controls / "input-manifest.json")["sources"]
    if set(manifest) != {str(row["source_id"]) for row in truth}:
        raise ValueError("Catalogue category manifest sources differ")
    for row in truth:
        metadata = manifest[str(row["source_id"])]["metadata"]
        if (
            str(metadata["source_id"]) != str(row["source_id"])
            or str(metadata["origin_tpf_kic"]) != str(row["kic_id"])
            or metadata["raw_fits_sha256"] != row["fits_sha256"]
        ):
            raise ValueError("Catalogue category raw/source/parent identity differs")


def native_partition(controls, freeze, truth, partition):
    if partition not in {"development", "evaluation"}:
        raise ValueError("Native EB reporting requires an explicit partition")
    roles = {}
    for role in ("development", "evaluation"):
        name = f"{role}-source-ids.json"
        if name not in freeze["artifacts"]:
            raise ValueError("Native EB partition is not hash-bound")
        ids = list(map(str, read(controls / name)))
        if len(ids) != freeze["partitions"][role] or len(ids) != len(set(ids)):
            raise ValueError("Native EB partition count or identities differ")
        roles[role] = ids
    if roles["development"] + roles["evaluation"] != [
        str(row["source_id"]) for row in truth
    ]:
        raise ValueError("Native EB partitions differ from frozen truth order")
    selected = set(roles[partition])
    return roles[partition], [row for row in truth if str(row["source_id"]) in selected]


def validate_control_inputs(controls, run, truth):
    expected = (controls / "input-manifest.json").resolve()
    preparation = read(run / "preparation.json")
    declared = preparation.get("input_manifest", {})
    configured = read(run / "vetting.json").get("input_manifest")
    if (
        not isinstance(declared, dict)
        or not isinstance(declared.get("path"), str)
        or root_path(declared["path"]).resolve() != expected
        or declared.get("sha256") != digest(expected)
        or not isinstance(configured, str)
        or root_path(configured).resolve() != expected
    ):
        raise ValueError("Catalogue replay uses a different frozen input manifest")
    prepared = {str(row["source_id"]): row for row in preparation["sources"]}
    for row in truth:
        actual = prepared.get(str(row["source_id"]), {})
        if (
            str(actual.get("parent_tpf")) != str(row["kic_id"])
            or actual.get("raw_fits_sha256") != row["fits_sha256"]
        ):
            raise ValueError("Catalogue replay raw/source/parent identity differs")


def validate_execution(run, results, rows, evaluation_role=None):
    policy = read(root_path(read(run / "vetting.json")["disposition_policy"]))
    execution = results.get("disposition_execution")
    if not isinstance(execution, dict):
        raise TypeError("Control run has no disposition execution record")
    if policy.get("status") in {"FROZEN_EVALUATION", "RELEASED"} and (
        policy.get("definition_sha256") != disposition_definition_sha256(policy)
    ):
        raise ValueError("Control disposition definition changed")
    effective = dict(policy)
    if policy.get("status") == "FROZEN_EVALUATION":
        effective["_evaluation_role"] = execution.get("evaluation_role")
        effective["_evaluation_authorized"] = effective["_evaluation_role"] is not None
    if disposition_execution(effective) != execution:
        raise ValueError("Control disposition execution differs from the policy")
    if (
        policy.get("status") == "FROZEN_EVALUATION"
        and evaluation_role is not None
        and execution.get("evaluation_role") != evaluation_role
    ):
        raise ValueError(
            "Control disposition execution uses a different partition role"
        )
    for row in rows.values():
        source_policy = {
            key: value
            for key, value in row.get("disposition_policy", {}).items()
            if key not in {"_evaluation_role", "_evaluation_authorized"}
        }
        if source_policy != policy:
            raise ValueError("Control source disposition policy differs from the run")
        if (
            row.get("disposition_execution") != execution
            or disposition_execution(row.get("disposition_policy")) != execution
        ):
            raise ValueError(
                "Control source disposition execution differs from the run"
            )
    return execution


def validate_run(run, expected, evaluation_role=None):
    preparation = read(run / "preparation.json")
    prepared = [str(row["source_id"]) for row in preparation["sources"]]
    detection = read(run / "detection.json")
    queue = read(run / "queue.json")
    results = read(run / "vetting/results.json")
    queued = [str(row["source_id"]) for row in queue["rows"]]
    source_paths = sorted((run / "vetting/sources").glob("*.json"))
    rows = {str(row["source_id"]): row for row in map(read, source_paths)}
    if (
        prepared != expected
        or detection.get("status") != "DETECTION_COMPLETE"
        or detection.get("sources") != len(expected)
        or len(queued) != len(set(queued))
        or not set(queued) <= set(expected)
        or set(queued) != set(rows)
        or detection.get("alerted_sources") != len(queued)
        or results.get("processing_status") != "EVIDENCE_STAGES_COMPLETE"
        or results.get("sources") != len(queued)
        or results.get("original_alerts_preserved") != detection.get("alerts")
    ):
        raise ValueError("Catalogue-control replay is incomplete or inconsistent")
    validate_execution(run, results, rows, evaluation_role)
    return preparation, detection, results, rows


def false_boi_report(execution, truth, trials):
    sources = [
        row
        for row in truth
        if row["category"] in {"catalogue_false_positive", NATIVE_EB_CATEGORY}
    ]
    source_ids = {str(row["source_id"]) for row in sources}
    if not source_ids:
        return {
            "status": "NOT_APPLICABLE_NO_FALSE_POSITIVE_TRIALS",
            "trials": 0,
            "zero_error_design_upper_bound": None,
            "confidence": CONTROL_CONFIDENCE,
        }
    design_bound = exact_upper_bound(0, len(source_ids))
    if execution["boi_catalogue_enabled"] is not True:
        return {
            "status": "NOT_MEASURED_BOI_CATALOGUE_DISABLED",
            "trials": len(source_ids),
            "zero_error_design_upper_bound": design_bound,
            "confidence": CONTROL_CONFIDENCE,
            "interpretation": (
                "Proposal and triage outcomes are diagnostics, not a false-BOI rate."
            ),
        }
    measured = [trial for trial in trials if trial["source_id"] in source_ids]
    if (
        len(measured) != len(source_ids)
        or {trial["source_id"] for trial in measured} != source_ids
    ):
        raise ValueError("False-class parent trial accounting is incomplete")
    return parent_outcomes(measured)


def native_cases(truth):
    return [
        {
            "category": source["category"],
            "pool_rank": source["pool_rank"],
            "kic_id": source["kic_id"],
            "source_id": str(source["source_id"]),
            "catalogue_record": signal,
            "ephemeris": {
                "supplied_period_days": signal["Per"],
                "supplied_epoch_days": signal["BJD0"],
                "epoch_reference": "BJD - 2400000",
                "epoch_conversion_applied": False,
                "duration_hours": None,
            },
            "matching_status": "NOT_MEASURED_NO_FROZEN_EVENT_MATCHING_CONTRACT",
            "match": None,
        }
        for source in truth
        for signal in source["catalogue_signals"]
    ]


def koi_cases(truth, run, rows):
    cases = []
    for source in truth:
        source_id = str(source["source_id"])
        proposals = read(run / "proposals" / f"{source_id}.json")
        signals = source["catalogue_signals"]
        matches = assign_truth_matches(signals, proposals)
        for index, signal in enumerate(signals):
            match = matches.get(index)
            if match is not None:
                match = {
                    key: value
                    for key, value in match.items()
                    if key not in {"key", "sort"}
                }
                match["downstream"] = downstream_match(rows.get(source_id), match)
            cases.append(
                {
                    "category": source["category"],
                    "pool_rank": source["pool_rank"],
                    "kic_id": source["kic_id"],
                    "source_id": source_id,
                    "kepoi_name": signal["kepoi_name"],
                    "period_days": signal["koi_period"],
                    "epoch_bkjd": signal["koi_time0bk"],
                    "duration_hours": signal["koi_duration"],
                    "match": match,
                }
            )
    return cases


def native_replay_preflight(run, truth, partition):
    config = read(run / "vetting.json")
    policy_path = root_path(config["disposition_policy"])
    policy = read(policy_path)
    if partition == "evaluation" and policy.get("status") != "FROZEN_EVALUATION":
        raise ValueError("Native EB evaluation requires a frozen scientific policy")
    if policy.get("status") == "FROZEN_EVALUATION":
        from pipeline.deterministic.run import evaluation_preflight

        identities = [
            {
                "source_id": str(row["source_id"]),
                "parent_tpf": str(row["kic_id"]),
                "raw_fits_sha256": row["fits_sha256"],
            }
            for row in truth
        ]
        scope = evaluation_preflight(
            SimpleNamespace(disposition_policy=policy_path),
            identities,
            config["input_manifest"],
        )
        expected_role = (
            "final_replay" if partition == "evaluation" else "development_replay"
        )
        if scope["evaluation_role"] != expected_role:
            raise ValueError("Native EB replay uses a different partition role")


def build_report(controls, run, partition=None, opened_execution=None):
    controls, run = Path(controls), Path(run)
    freeze, source_ids, truth = validate_controls(controls, partition)
    scope = None
    if opened_execution is not None:
        if partition != "development" or freeze["protocol"] != NATIVE_EB_PROTOCOL:
            raise ValueError("Opened subset reporting requires native EB development")
        scope = opened_replay_scope(opened_execution, "native-eb", run, source_ids)
        source_ids = scope["selected_source_ids"]
        truth = [row for row in truth if str(row["source_id"]) in set(source_ids)]
    validate_control_inputs(controls, run, truth)
    native = freeze["protocol"] == NATIVE_EB_PROTOCOL
    if native:
        native_replay_preflight(run, truth, partition)
    role = {"development": "development_replay", "evaluation": "final_replay"}.get(
        partition
    )
    _, detection, results, rows = validate_run(run, source_ids, role)
    execution = results["disposition_execution"]
    categories = {str(row["source_id"]): row["category"] for row in truth}
    cases = native_cases(truth) if native else koi_cases(truth, run, rows)
    timings = read(run / "vetting/stage-timings.json")
    stage_wall = {
        row["stage"]: row["wall_seconds"]
        for row in timings["stages"]
        if row.get("status") == "COMPLETE"
    }
    policy_path = root_path(read(run / "vetting.json")["disposition_policy"])
    source_hashes = {
        path.name: digest(path)
        for path in sorted((run / "vetting/sources").glob("*.json"))
    }
    source_evidence_sha256 = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = [
        controls / "freeze.json",
        controls / "input-manifest.json",
        controls / "truth.jsonl",
        controls / "source-ids.json",
        run / "preparation.json",
        run / "detection.json",
        run / "discovery-manifest.json",
        run / "queue.json",
        run / "vetting/results.json",
        run / "vetting/manifest.json",
        run / "vetting/stage-timings.json",
        policy_path,
        Path(__file__),
        Path(__file__).resolve().parents[1] / "runtime.py",
    ]
    if native:
        artifacts.append(controls / f"{partition}-source-ids.json")
    category_sources = Counter(row["category"] for row in truth)
    category_queued = Counter(categories[source_id] for source_id in rows)
    trials = trial_ledger(
        run,
        truth,
        rows,
        cases,
        execution["boi_catalogue_enabled"],
    )
    return {
        "schema": 4 if native else 3,
        "outcome_contract": OUTCOME_CONTRACT,
        "status": "PARTIAL_OPENED_CATALOGUE_CONTROL_REPORT"
        if scope
        else "COMPLETE_CATALOGUE_CONTROL_REPORT",
        "opened_execution_scope": scope,
        "claim_boundary": freeze["claim_boundary"],
        **(
            {"partition": partition, "truth_protocol": NATIVE_EB_PROTOCOL}
            if native
            else {}
        ),
        "population": {
            "declared_sources": scope["declared_sources"] if scope else len(source_ids),
            "prepared_sources": detection["sources"],
            "queued_sources": detection["alerted_sources"],
            "declared_signals": len(cases),
            "by_category": {
                category: {
                    "sources": count,
                    "queued_sources": category_queued[category],
                }
                for category, count in sorted(category_sources.items())
            },
        },
        "detector": {
            "raw_rows": detection["raw_rows"],
            "alerts": detection["alerts"],
            "classifications": detection["classifications"],
            "binary_sha256": detection["binary"]["sha256"],
            "catalogue_signal_recovery": {
                "status": "NOT_MEASURED_NO_FROZEN_EVENT_MATCHING_CONTRACT",
                "matched_detector_proposals": None,
                "matched_fraction": None,
                "interpretation": (
                    "The native records supply no durations. No event tolerance "
                    "or BKJD conversion is assumed. Parent-level candidate "
                    "emission is assessed across every canonical signal "
                    "independently of truth matching."
                ),
            }
            if native
            else summarize_cases(cases),
        },
        "vetting": {
            "all_queued_signals": summarize_all_signals(rows, categories),
            "processing_status": results["processing_status"],
            "disposition_execution": execution,
            "boi_signals": results["boi_signals"],
            "unresolved_signals": results["unresolved_signals"],
        },
        "false_boi_rate": {
            "status": "NOT_MEASURED_METADATA_ONLY_TRIALS",
            "upper_bound": None,
        }
        if scope and scope["metadata_only_trials"]
        else false_boi_report(execution, truth, trials),
        "trial_ledger": {
            "unit": "independent_parent_tpf",
            "boi_catalogue_enabled": execution["boi_catalogue_enabled"],
            "trials": trials,
            "sha256": json_digest(trials),
        },
        "performance": {
            "recorded_stage_wall_seconds": stage_wall,
            "recorded_vetting_wall_seconds": sum(stage_wall.values()),
            "slowest_stage": max(stage_wall, key=stage_wall.get),
        },
        "cases": cases,
        "inputs": {
            **{artifact_path(path, ROOT): digest(path) for path in artifacts},
            **(scope["inputs"] if scope else {}),
        },
        "source_evidence": {
            "files": len(source_hashes),
            "digest_ledger_sha256": source_evidence_sha256,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", choices=["development", "evaluation"])
    parser.add_argument("--opened-execution", type=Path)
    args = parser.parse_args()
    report = build_report(
        root_path(args.controls),
        root_path(args.run),
        args.partition,
        args.opened_execution,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
