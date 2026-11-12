"""Assemble release replay ledgers from verified control reports and input roles."""

import argparse
import csv
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline/vetting")]

from pipeline.deterministic import catalogue_control_report as catalogue
from pipeline.deterministic import (
    control_report,
    pixel_parent_audit,
    planet_control_report,
)
from pipeline.deterministic import (
    native_prospective_controls as native,
)
from pipeline.deterministic.signal_contracts import (
    NULL_CONTRACT,
    exclusion_references,
    null_exclusion,
    null_manifest,
    null_population,
)
from pipeline.runtime import (
    EVENT_MATCHING_CONTRACT,
    OUTCOME_CONTRACT,
    disposition_definition_sha256,
    dump_path,
    implementation_hashes,
)

COHORTS = {
    "periodic_injection": ("periodic_planet", "planet", "cases"),
    "sparse_injection": ("sparse_injection", "planet", "morphology_cases"),
    "ttv_injection": ("ttv_injection", "planet", "morphology_cases"),
    "background_contaminant": (
        "background_contaminant",
        "background_contaminant",
        "cases",
    ),
    "catalogue_eclipsing_binary": ("native_eb", "eclipsing_binary", "cases"),
    "ordinary": ("ordinary", "ordinary_unknown", "diagnostic_cases"),
}


def verified_reference(reference):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Control report requires a path and SHA256 reference")
    path = catalogue.root_path(reference["path"])
    if catalogue.digest(path) != reference["sha256"]:
        raise ValueError(f"Bound report changed: {path}")
    return path, catalogue.read(path)


def verify_inputs(inputs):
    for name, expected in inputs.items():
        if catalogue.digest(catalogue.root_path(name)) != expected:
            raise ValueError(f"Replay evidence input changed: {name}")


def input_graph(graph, role):
    audit_path, ledger_path = graph / "audit.json", graph / "trials.csv"
    audit = catalogue.read(audit_path)
    verify_inputs({str(ledger_path): audit["trial_ledger_sha256"]})
    with ledger_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != audit["trials"]:
        raise ValueError("Input graph trial count changed")
    if role == "final_replay" and audit.get("joint_independence_complete") is not True:
        raise ValueError("Final replay requires the completed input independence audit")
    partition = "evaluation" if role == "final_replay" else "development"
    required = {
        cohort for cohort, _, group in COHORTS.values() if group != "diagnostic_cases"
    }
    required.add("null")
    selected = [
        row
        for row in rows
        if row["role"] == partition
        and (role != "final_replay" or row["cohort"] in required)
    ]
    identities = [(row["cohort"], row["source_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("Input graph repeats a cohort/source identity")
    if role == "final_replay" and any(
        row["historically_exposed_component"] != "False" for row in selected
    ):
        raise ValueError("Final input graph contains historically opened components")
    return (
        audit,
        dict(zip(identities, selected)),
        {
            str(audit_path): catalogue.digest(audit_path),
            str(ledger_path): catalogue.digest(ledger_path),
        },
    )


def report_records(report):
    if report.get("outcome_contract") != OUTCOME_CONTRACT:
        raise ValueError("Control report uses a legacy outcome contract")
    status = report.get("status")
    if status in native.REPORT_STATUSES:
        return [(case["category"], case) for case in native.verified_report(report)]
    if status == "COMPLETE_PLANET_CONTROL_REPORT":
        if report.get("event_matching_contract") != EVENT_MATCHING_CONTRACT:
            raise ValueError("Planet report uses a legacy event-matching contract")
        return [(case["truth"]["category"], case) for case in report["cases"]]
    if status == "COMPLETE_FALSE_CLASS_CONTROL_REPORT":
        if any(case["category"] not in COHORTS for case in report["trials"]):
            raise ValueError(
                "Dropped synthetic class is outside the launch release scope"
            )
        return [(case["category"], case) for case in report["trials"]]
    if (
        status == "COMPLETE_CATALOGUE_CONTROL_REPORT"
        and report.get("truth_protocol") == catalogue.NATIVE_EB_PROTOCOL
    ):
        return [(case["category"], case) for case in report["trial_ledger"]["trials"]]
    if status == "COMPLETE_CONTROL_REPLAY_REPORT":
        return [("ordinary", case) for case in report["trial_ledger"]["trials"]]
    raise ValueError("Unsupported release control report protocol")


def read_control_run(entry, role):
    path, report = verified_reference(entry["report"])
    verify_inputs(report["inputs"])
    records = report_records(report)
    ids = [str(record["source_id"]) for _, record in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Control report trials are empty or duplicated")
    run = catalogue.root_path(entry["run"])
    preparation, detection, results, sources = catalogue.validate_run(run, ids, role)
    binary = detection["binary"]
    verify_inputs({binary["path"]: binary["sha256"]})
    checksums = {
        path.name: catalogue.digest(path)
        for path in sorted((run / "vetting/sources").glob("*.json"))
    }
    if report["source_evidence"] != {
        "files": len(checksums),
        "digest_ledger_sha256": catalogue.json_digest(checksums),
    }:
        raise ValueError("Control report source evidence changed")
    manifest = catalogue.read(run / "vetting/manifest.json")
    verify_inputs(manifest["input_sha256"])
    prepared = {str(row["source_id"]): row for row in preparation["sources"]}
    physical = preparation.get("input_manifest")
    inputs = catalogue.read(physical["path"])["sources"] if physical else {}
    truth = (
        {
            str(row["source_id"]): row
            for row in catalogue.read_jsonl(
                Path(physical["path"]).parent / "truth.jsonl"
            )
        }
        if physical
        else {}
    )
    if physical:
        verify_inputs({physical["path"]: physical["sha256"]})
    policy_path = catalogue.root_path(
        catalogue.read(run / "vetting.json")["disposition_policy"]
    )
    return {
        "reference": entry["report"],
        "path": path,
        "report": report,
        "run": run,
        "records": records,
        "prepared": prepared,
        "physical": inputs,
        "truth": truth,
        "sources": sources,
        "policy": catalogue.read(policy_path),
        "execution": results["disposition_execution"],
        "preparation": preparation,
        "manifest": manifest,
        "binary": binary,
    }


def event_ledger(proposal, source):
    expected = proposal["alerts"]
    actual = source["alerts"] if source else []
    alert_ids = [alert["event_id"] for alert in expected]
    if len(alert_ids) != len(set(alert_ids)):
        raise ValueError("Detector proposal repeats an alert identity")
    signals = source["signals"] if source else []
    assigned = source["disposition"]["signals"] if source else []
    return {
        "emitted_alerts": len(expected),
        "adjudicated_alerts": len(actual),
        "exact_identity_match": actual == expected,
        "declared_signal_ids": [signal["signal_id"] for signal in signals],
        "signal_dispositions": [
            {"signal_id": item["signal_id"], "disposition": item["disposition"]}
            for item in assigned
        ],
        "boi_signals": source["disposition"].get("boi_signal_count")
        if source
        else 0,
    }


def verify_planet_record(record, control):
    sid = str(record["source_id"])
    truth = control["truth"][sid]
    if any(value != truth[name] for name, value in record["truth"].items()):
        raise ValueError("Planet report physical truth differs from frozen inputs")
    events, _ = planet_control_report.prepared_truth_events(
        truth, control["preparation"]
    )
    if record["truth_events"] != events:
        raise ValueError("Planet report physical event ledger changed")
    source = control["sources"].get(sid)
    signals = (
        {signal["signal_id"]: signal for signal in source["signals"]} if source else {}
    )
    for item in record["signals"]:
        signal = signals[item["signal_id"]]
        assigned = planet_control_report.disposition(source, signal["signal_id"])
        units = planet_control_report.fitted_event_units(source, signal, assigned)
        if item["fitted_units"] != units or item[
            "downstream"
        ] != planet_control_report.downstream(source, signal):
            raise ValueError("Planet report fitted or disposition evidence changed")
        measured = planet_control_report.match_event_units(
            events, signal["events"], truth["duration_hours"]
        )
        if item["proposal_event_match"] != measured:
            raise ValueError("Planet report proposal membership changed")
        fitted = (
            planet_control_report.match_event_units(
                events, units, truth["duration_hours"]
            )
            if units is not None
            else {"status": "UNAVAILABLE"}
        )
        if item["fitted_event_match"] != fitted:
            raise ValueError("Planet report fitted membership changed")


def assemble_case(category, record, control, graph):
    cohort, truth_class, group = COHORTS[category]
    sid = str(record["source_id"])
    declared = graph[(cohort, sid)]
    prepared = control["prepared"][sid]
    if (str(prepared["parent_tpf"]), prepared["raw_fits_sha256"]) != (
        declared["origin_parent"],
        declared["raw_fits_sha256"],
    ) or not declared["component_id"]:
        raise ValueError(
            "Replay raw/parent/component identity differs from the input graph"
        )
    verify_inputs(
        {item["fsc_path"]: item["fsc_sha256"] for item in prepared["inputs"].values()}
    )
    physical = control["physical"].get(sid)
    parents = sorted(
        {
            str(pixel["base_parent_kic"])
            for pixel in (physical or {}).get("pixels", {}).values()
        }
    )
    if parents and not set(parents) <= set(declared["resolved_parents"].split(";")):
        raise ValueError("Actual pixel parents are missing from the input graph")
    source = control["sources"].get(sid)
    declared_signals = [item["signal_id"] for item in record["signals"]]
    actual_signals = [item["signal_id"] for item in source["signals"]] if source else []
    if len(declared_signals) != len(set(declared_signals)) or set(
        declared_signals
    ) != set(actual_signals):
        raise ValueError("Control report canonical alias ledger is incomplete")
    if truth_class == "planet":
        verify_planet_record(record, control)
    else:
        signal_report = (
            control_report.trial_signal
            if category == "ordinary"
            else catalogue.trial_signal
        )
        measured = (
            [signal_report(source, signal) for signal in source["signals"]]
            if source
            else []
        )
        if record["signals"] != measured:
            raise ValueError(
                "Control report signal dispositions differ from source evidence"
            )
    proposal_path = control["run"] / "proposals" / f"{sid}.json"
    proposal = catalogue.read(proposal_path)
    source_ids = [
        sid,
        *(
            [declared["contaminant_source_id"]]
            if declared["contaminant_source_id"]
            else []
        ),
    ]
    raw = [
        declared["raw_fits_sha256"],
        *(
            [declared["contaminant_raw_fits_sha256"]]
            if declared["contaminant_raw_fits_sha256"]
            else []
        ),
    ]
    output = {
        "truth_cohort": cohort,
        "truth_class": truth_class,
        "source_id": sid,
        "source_ids": source_ids,
        "raw_fits_sha256": raw,
        "parent_tpf": declared["origin_parent"],
        "resolved_parent_tpfs": parents,
        "component_ids": [declared["component_id"]],
        "historically_exposed_component": declared["historically_exposed_component"]
        == "True",
        "input_sha256": catalogue.json_digest(
            {"physical": physical, "prepared_inputs": prepared["inputs"]}
        ),
        "status": "FAILED" if catalogue.incomplete_disposition(source) else "COMPLETE",
        "alert_ledger": event_ledger(proposal, source),
        "signals": record["signals"],
        "source_evidence": {
            "path": str(control["run"] / "vetting/sources" / f"{sid}.json"),
            "sha256": catalogue.digest(
                control["run"] / "vetting/sources" / f"{sid}.json"
            ),
        }
        if source
        else None,
        "proposal_evidence": {
            "path": str(proposal_path),
            "sha256": catalogue.digest(proposal_path),
        },
    }
    if truth_class == "planet":
        output.update(truth=record["truth"], truth_events=record["truth_events"])
        units = (
            {signal["signal_id"]: signal["events"] for signal in source["signals"]}
            if source
            else {}
        )
        output["signals"] = [
            {**signal, "proposal_units": units[signal["signal_id"]]}
            for signal in record["signals"]
        ]
    if category == "sparse_injection":
        output["single_quarter_assessment"] = single_quarter_assessment(output)
    return group, output


def frozen_execution(controls, audit, role, run_spec_path):
    enabled = [control["execution"]["boi_catalogue_enabled"] for control in controls]
    if not any(enabled):
        return None
    policies = [control["policy"] for control in controls]
    if not all(enabled) or any(policy != policies[0] for policy in policies):
        raise ValueError("Replay combines different automatic disposition policies")
    policy = policies[0]
    vetting = importlib.import_module("vetting")
    vetting.validate_operational_policy(policy)
    if (
        policy["status"] != "FROZEN_EVALUATION"
        or policy["implementation"] != implementation_hashes()
        or audit.get("joint_independence_complete") is not True
        or any(control["execution"]["evaluation_role"] != role for control in controls)
        or any(
            control["preparation"]["implementation"] != policy["implementation"]
            for control in controls
        )
        or any(
            control["manifest"]["code_sha256"] != policy["implementation"]
            for control in controls
        )
        or any(
            control["binary"]["sha256"]
            != policy["evaluation_binding"]["detector_sha256"]
            for control in controls
        )
        or run_spec_path is None
    ):
        raise ValueError("Release replay is not the frozen independent experiment")
    scopes = [control["preparation"].get("evaluation_scope") for control in controls]
    expected = policy["evaluation_binding"][role]["cohorts"]
    actual = [
        scope["cohort"]
        for scope in scopes
        if scope and scope["evaluation_role"] == role
    ]
    if Counter(map(catalogue.json_digest, actual)) != Counter(
        map(catalogue.json_digest, expected)
    ):
        raise ValueError("Replay does not cover every frozen authorized cohort")
    for control in controls:
        config_path = control["run"] / "vetting.json"
        if control["manifest"]["config_sha256"] != catalogue.digest(config_path):
            raise ValueError("Frozen replay configuration changed")
        if (
            vetting.pipeline_contract(catalogue.read(config_path))
            != policy["pipeline_contract"]
        ):
            raise ValueError("Frozen replay pipeline contract changed")
    verify_inputs({str(run_spec_path): policy["evaluation_binding"]["run_spec_sha256"]})
    spec = catalogue.read(run_spec_path)
    if spec.get("status") != "SEALED" or spec.get(
        "final_audit"
    ) != vetting.release_audit_spec(policy):
        raise ValueError("Run specification differs from the frozen release policy")
    return policy, spec


def publication_fraction(measured, confidence, limit, complete, minimum):
    vetting = importlib.import_module("vetting")
    trials, count = measured["trials"], measured["outcomes"]
    fraction = count / trials if complete and trials else None
    passed = fraction is not None and (
        fraction >= limit if minimum else fraction <= limit
    )
    return {
        "status": ("PASSED" if passed else "FAILED")
        if fraction is not None
        else "NOT_MEASURED_INCOMPLETE_EXECUTION",
        "trials": trials,
        "recovered" if minimum else "false_candidates": count,
        "measured_fraction": fraction,
        "lower_bound" if minimum else "upper_bound": vetting.exact_binomial(
            count, trials, confidence, not minimum
        )
        if fraction is not None
        else None,
        "required_minimum" if minimum else "required_maximum": limit,
    }


def single_quarter_check(report, outcomes, audit, complete):
    assessments = []
    for case in report["morphology_cases"]:
        if case["truth_cohort"] != "sparse_injection":
            continue
        assessed = single_quarter_assessment(case)
        if case["single_quarter_assessment"] != assessed:
            raise ValueError(
                "Single-quarter assessment differs from saved physical evidence"
            )
        assessments.append(assessed)
    conflicts = sum(
        row["applicable"] and row["status"] == "ROBUST_OFFSET_CONFLICT"
        for row in assessments
    )
    accepted = conflicts <= audit["maximum_single_event_offset_conflicts"]
    return {
        "status": "PASSED" if complete else "NOT_MEASURED_INCOMPLETE_EXECUTION",
        "trials": outcomes["morphology"]["sparse_injection"]["trials"],
        "robust_offset_conflicts": conflicts,
        "maximum_conflicts": audit["maximum_single_event_offset_conflicts"],
        "branch": ("ACCEPTED" if accepted else "DR2_UNRESOLVED")
        if complete
        else "UNMEASURED",
    }


def reported_fraction(
    measured, count_field, fraction_field, bound_field, confidence, complete, upper
):
    vetting = importlib.import_module("vetting")
    count, trials = measured[count_field], measured["trials"]
    measured[fraction_field] = count / trials if complete and trials else None
    measured[bound_field] = (
        vetting.exact_binomial(count, trials, confidence, upper)
        if complete and trials
        else None
    )


def release_checks(report, spec):
    vetting = importlib.import_module("vetting")
    outcomes = vetting.replay_outcomes(report, spec)
    audit = spec["final_audit"]
    ledger = outcomes["ledger"]
    complete = not ledger["failed_cases"] and not ledger["identity_mismatch_cases"]
    checks = {
        "planet_recovery": publication_fraction(
            outcomes["planet"],
            audit["confidence"],
            audit["minimum_periodic_recovery_fraction"],
            complete,
            True,
        ),
        "null_false_candidates": publication_fraction(
            outcomes["null"],
            audit["confidence"],
            audit["maximum_null_false_candidate_fraction"],
            complete,
            False,
        ),
        "single_quarter_branch": single_quarter_check(
            report, outcomes, audit, complete
        ),
        "complete_outcome_ledger": {
            "status": "PASSED" if complete else "FAILED",
            **ledger,
        },
    }
    for measured in [
        outcomes["eclipsing_binary"],
        outcomes["background_contaminant"],
        outcomes["null"],
        *outcomes["null"]["variants"].values(),
    ]:
        reported_fraction(
            measured,
            "outcomes",
            "measured_fraction",
            "upper_bound",
            audit["confidence"],
            complete,
            True,
        )
    for measured in outcomes["morphology"].values():
        reported_fraction(
            measured,
            "exact_signal_boi_catalogue_recoveries",
            "exact_recovery_fraction",
            "exact_recovery_lower_bound",
            audit["confidence"],
            complete,
            False,
        )
    return checks, outcomes


def single_quarter_assessment(case):
    matches = [
        signal
        for signal in case["signals"]
        if signal["proposal_event_match"]["status"] == "MATCHED"
    ]
    statuses = [
        signal["downstream"]
        .get("localization", {})
        .get("source_attribution", {})
        .get("status", "NOT_REACHED")
        for signal in matches
    ]
    return {
        "status": "ROBUST_OFFSET_CONFLICT"
        if "ROBUST_OFFSET_CONFLICT" in statuses
        else "NO_ROBUST_OFFSET_CONFLICT",
        "applicable": len(case["truth_events"]) == 1,
        "matched_signals": len(matches),
        "localization_statuses": statuses,
    }


def evidence_inputs(cases):
    return {
        reference["path"]: reference["sha256"]
        for case in cases
        for name in ("source_evidence", "proposal_evidence")
        if (reference := case[name]) is not None
    }


def null_lane_decisions(proposal):
    identities = sorted(
        {alert["detector_signal_id"] for alert in proposal["ungated_alerts"]}
    )
    if [row["proposal_id"] for row in proposal["lane_decisions"]] != identities:
        raise ValueError("Null lane ledger lost, reordered or duplicated a proposal")
    accepted = {
        row["proposal_id"] for row in proposal["lane_decisions"]
        if row["decision"] == "PASS"
    }
    if proposal["alerts"] != [
        alert for alert in proposal["ungated_alerts"]
        if alert["detector_signal_id"] in accepted
    ]:
        raise ValueError("Null emitted alerts differ from the declared lane decisions")
    return proposal["lane_decisions"]


def null_variant_record(proposal, source, specification, sid, variant):
    references = exclusion_references(specification, sid)
    records = []
    for decision in null_lane_decisions(proposal):
        identity = decision["proposal_id"]
        alerts = [
            alert
            for alert in proposal["ungated_alerts"]
            if alert["detector_signal_id"] == identity
        ]
        exclusion = null_exclusion(alerts, references)
        if decision.get("null_exclusion") != exclusion:
            raise ValueError("Saved null exclusion differs from exact event membership")
        if decision["decision"] not in {"PASS", "REJECT", "EXCLUDE"} or (
            (decision["decision"] == "EXCLUDE") != (exclusion["status"] == "EXCLUDED")
        ):
            raise ValueError("Null lane decision differs from its exclusion state")
        if decision["decision"] == "REJECT":
            continue
        paths = []
        if exclusion["status"] == "ELIGIBLE":
            if source is None:
                raise ValueError("A queued null proposal lost its source result")
            events = {
                event["event_id"]
                for event in source["events"]
                if event["original_alert"]["detector_signal_id"] == identity
            }
            for signal in source["signals"]:
                if events & set(signal["event_ids"]):
                    paths.append(
                        {
                            "signal_id": signal["signal_id"],
                            "gate_path": [
                                row
                                for row in signal["triage"]["history"]
                                if row["stage"] in {"shape", "orbital", "instrument"}
                            ],
                        }
                    )
        records.append(
            {"proposal_id": identity, "exclusion": exclusion, "gate_paths": paths}
        )
    return {
        "variant": variant,
        "status": "COMPLETE_NULL_FLUX_LEDGER"
        if source is None or source.get("processing_status") == "EVIDENCE_STAGES_COMPLETE"
        else "INCOMPLETE_NULL_SOURCE_EXECUTION",
        "declared_proposal_ids": [record["proposal_id"] for record in records],
        "proposals": records,
        "lane_decisions": proposal["lane_decisions"],
        "emitted_alerts": len(proposal["alerts"]),
        "adjudicated_alerts": len(source["alerts"]) if source else 0,
        "exact_identity_match": (source["alerts"] if source else [])
        == proposal["alerts"],
    }


def verified_null_run(folder, role):
    run = catalogue.root_path(folder)
    if (run / "empty-variant.json").exists():
        return verified_empty_null_run(run, role)
    preparation = catalogue.read(run / "preparation.json")
    reference = preparation["input_manifest"]
    verify_inputs({reference["path"]: reference["sha256"]})
    contract = null_manifest(ROOT, reference["path"])
    if contract is None or contract["id"] != NULL_CONTRACT:
        raise ValueError("Null replay lacks its frozen construction contract")
    specification = catalogue.read(reference["path"])
    ids = [row["source_id"] for row in preparation["sources"]]
    if ids != list(specification["sources"]):
        raise ValueError("Null replay changed the declared source order or population")
    exclusion_path, exclusions = verified_reference(specification["null_exclusions"])
    inputs = {str(exclusion_path): catalogue.digest(exclusion_path)}
    for row in exclusions["sources"].values():
        for name in ("real_signals", "catalogue_ephemerides"):
            verify_inputs({row[name]["path"]: row[name]["sha256"]})
            inputs[row[name]["path"]] = row[name]["sha256"]
    preparation, detection, results, sources = catalogue.validate_run(run, ids, role)
    from pipeline.deterministic.run import validate_preparation, verify_discovery

    validate_preparation(run)
    verify_discovery(
        run / "queue.json",
        catalogue.read(run / "discovery-manifest.json"),
        catalogue.read(run / "vetting.json"),
    )
    policy = catalogue.read(
        catalogue.root_path(catalogue.read(run / "vetting.json")["disposition_policy"])
    )
    manifest = catalogue.read(run / "vetting/manifest.json")
    control = {
        "reference": {
            "path": str(run / "vetting/results.json"),
            "sha256": catalogue.digest(run / "vetting/results.json"),
        },
        "policy": policy,
        "execution": results["disposition_execution"],
        "preparation": preparation,
        "manifest": manifest,
        "run": run,
        "binary": detection["binary"],
    }
    inputs.update(manifest["input_sha256"])
    inputs.update(
        {
            str(run / name): catalogue.digest(run / name)
            for name in (
                "preparation.json",
                "detection.json",
                "vetting.json",
                "discovery-manifest.json",
                "vetting/manifest.json",
                "vetting/results.json",
            )
        }
    )
    inputs.update(
        {
            reference["path"]: reference["sha256"],
            contract["base_manifest"]["path"]: contract["base_manifest"]["sha256"],
        }
    )
    return control, inputs, specification, ids, contract, sources


def verified_empty_null_run(run, role):
    receipt_path = run / "empty-variant.json"
    receipt = catalogue.read(receipt_path)
    path, specification = verified_reference(receipt["input_manifest"])
    contract = null_manifest(ROOT, path)
    if (
        role != "development_replay"
        or receipt.get("status") != "NOT_APPLICABLE_SHORT_COVERAGE"
        or receipt.get("detectors_or_vetting_run") is not False
        or specification["role"] != "development"
        or contract is None or contract["variant"] != "quarter_scrambled"
        or specification["sources"]
        or not specification["construction_ledger"]
        or any((run / name).exists() for name in ("detection.json", "vetting", "proposals"))
    ):
        raise ValueError("An empty null variant must retain only its inapplicable construction ledger")
    return None, {str(path): catalogue.digest(path), str(receipt_path): catalogue.digest(receipt_path)}, specification, [], contract, {}


def verify_null_recipe(recipe, first, base_reference):
    if recipe != first:
        raise ValueError("Null variants differ in base inputs, seed or exclusions")
    if base_reference is not None and (
        recipe["base_manifest"] != base_reference or recipe["seed"] != 20261002
    ):
        raise ValueError("Null replay changed the frozen development base or transform seed")


def read_null_runs(paths, graph, role, base_reference=None):
    controls, inputs, cases = [], {}, {}
    if len(paths) != 2:
        raise ValueError("Null replay requires inverted and quarter-scrambled runs")
    variants, recipes = [], []
    for folder in paths:
        control, run_inputs, specification, ids, contract, sources = verified_null_run(
            folder, role
        )
        if control is not None:
            controls.append(control)
        inputs.update(run_inputs)
        run, variant = catalogue.root_path(folder), contract["variant"]
        variants.append(variant)
        recipes.append(
            {
                "id": contract["id"],
                "seed": contract["seed"],
                "base_manifest": contract["base_manifest"],
                "null_exclusions": specification["null_exclusions"],
            }
        )
        verify_null_recipe(recipes[-1], recipes[0], base_reference)
        for sid in ids:
            original = graph[("null", sid)]
            item = specification["sources"][sid]
            if (
                str(item["metadata"]["origin_tpf_kic"]) != original["origin_parent"]
                or item["metadata"]["raw_fits_sha256"] != original["raw_fits_sha256"]
            ):
                raise ValueError(
                    "Null parent/raw identity differs from frozen input graph"
                )
            case = cases.setdefault(
                sid,
                {
                    "truth_cohort": "null",
                    "source_id": sid,
                    "source_ids": [sid],
                    "parent_tpf": original["origin_parent"],
                    "raw_fits_sha256": [original["raw_fits_sha256"]],
                    "component_ids": [original["component_id"]],
                    "historically_exposed_component": original[
                        "historically_exposed_component"
                    ]
                    == "True",
                    "input_sha256": catalogue.json_digest(
                        {"source_id": sid, "base_manifest": contract["base_manifest"]}
                    ),
                    "usable_quarters": item["null_quarters"],
                    "variants": [],
                    "scientific_abstention": False,
                    "source_evidence": None,
                    "proposal_evidence": None,
                },
            )
            if case["usable_quarters"] != item["null_quarters"]:
                raise ValueError("Null variants changed their source coverage")
            proposal_path = run / "proposals" / f"{sid}.json"
            proposal = catalogue.read(proposal_path)
            source = sources.get(sid)
            record = null_variant_record(proposal, source, specification, sid, variant)
            record["input_sha256"] = item["sha256"]
            record["proposal_evidence"] = {
                "path": str(proposal_path),
                "sha256": catalogue.digest(proposal_path),
            }
            record["source_evidence"] = (
                {
                    "path": str(run / "vetting/sources" / f"{sid}.json"),
                    "sha256": catalogue.digest(run / "vetting/sources" / f"{sid}.json"),
                }
                if source
                else None
            )
            inputs[record["proposal_evidence"]["path"]] = record["proposal_evidence"][
                "sha256"
            ]
            if record["source_evidence"]:
                inputs[record["source_evidence"]["path"]] = record["source_evidence"][
                    "sha256"
                ]
            case["variants"].append(record)
            case["scientific_abstention"] |= bool(
                source
                and any(
                    row["disposition"] == "UNRESOLVED"
                    for row in source["disposition"]["signals"]
                )
            )
    if variants != ["inverted", "quarter_scrambled"]:
        raise ValueError("Null runs are duplicated, reversed or missing a variant")
    null_population(list(cases.values()))
    return controls, list(cases.values()), inputs


def validate_report_trials(cases, graph):
    identities = [
        (case["truth_cohort"], case["source_id"])
        for case in cases
        if case.get("native_prospective_contract") != native.PROTOCOL
    ]
    cohorts = {identity[0] for identity in identities}
    if len(identities) != len(set(identities)) or set(identities) != {
        key for key in graph if key[0] in cohorts
    }:
        raise ValueError("Replay omits or duplicates declared cohort trials")


def assemble_native_case(record, graph):
    item = record["identity"]
    cohort, sid = item["cohort"], item["source_id"]
    declared = graph.get((cohort, sid))
    if declared is not None and (
        declared["origin_parent"] != item["parent_tpf"]
        or declared["raw_fits_sha256"] != item["raw_fits_sha256"]
        or declared["component_id"] != item["component_id"]
    ):
        raise ValueError("Native replay identity differs from the input graph")
    proposal = (
        catalogue.read(native.verified_reference(record["proposal_evidence"]))
        if record["proposal_evidence"]
        else {"alerts": []}
    )
    source = (
        catalogue.read(native.verified_reference(record["source_evidence"]))
        if record["source_evidence"]
        else None
    )
    return {
        "truth_cohort": cohort,
        "truth_class": cohort,
        "source_id": sid,
        "source_ids": [sid],
        "raw_fits_sha256": [item["raw_fits_sha256"]],
        "parent_tpf": item["parent_tpf"],
        "resolved_parent_tpfs": [],
        "component_ids": [item["component_id"]],
        "historically_exposed_component": True,
        "native_prospective_contract": native.PROTOCOL,
        "joint_independence_complete": False,
        "input_sha256": catalogue.json_digest(
            {"identity": item, "truth": record["truth"], "recipe": record["recipe"]}
        ),
        "status": "COMPLETE" if record["execution_complete"] else "FAILED",
        "native_binding_status": record["binding_status"],
        "native_execution_status": record["status"],
        "truth_reference": record["truth"],
        "recipe_reference": record["recipe"],
        "alert_ledger": event_ledger(proposal, source),
        "signals": record["signals"],
        "source_evidence": record["source_evidence"],
        "proposal_evidence": record["proposal_evidence"],
    }


def append_native_report(entry, role, graph, groups, controls, inputs):
    path, report = verified_reference(entry["report"])
    if report.get("protocol") != native.PROTOCOL:
        return False
    if role != "development_replay":
        raise ValueError("Native prospective record controls are development-only")
    verify_inputs(report["inputs"])
    records = native.verified_report(report)
    if any(
        case.get("native_prospective_contract") == native.PROTOCOL
        for case in groups["cases"]
    ):
        raise ValueError("Native prospective reports repeat the original 200 cases")
    groups["cases"].extend(assemble_native_case(record, graph) for record in records)
    inputs.update(report["inputs"])
    inputs[str(path)] = catalogue.digest(path)
    controls.append(
        {
            "reference": entry["report"],
            "policy": native.metadata(native.metadata(report["binding"])["policy"]),
            "execution": {"boi_catalogue_enabled": False},
            "binary": None,
        }
    )
    return True


def bind_replay_execution(report, frozen, role):
    if frozen is not None:
        policy, spec = frozen
        report.update(
            implementation=policy["implementation"],
            disposition_definition_sha256=disposition_definition_sha256(policy),
            controls_sha256=policy["evaluation_binding"][role]["controls_sha256"],
            **{
                name: policy["evaluation_binding"][name]
                for name in ("run_spec_sha256", "detector_sha256", "calibration_sha256")
            },
        )
        if role == "final_replay":
            report["release_checks"], report["outcomes"] = release_checks(report, spec)
            report["status"] = (
                "REPLAY_PASSED"
                if all(
                    check["status"] == "PASSED"
                    for check in report["release_checks"].values()
                )
                else "REPLAY_FAILED"
            )
        else:
            vetting = importlib.import_module("vetting")
            try:
                vetting.validate_development_execution(report)
            except ValueError:
                report["status"] = "REPLAY_FAILED"
            else:
                report["status"] = "REPLAY_PASSED"
    return report


def declared_null_graph(graph, role, base_reference):
    from pipeline.deterministic import null_exclusions as declaration

    if role != "development_replay":
        raise ValueError("First-declaration null assembly is development-only")
    base = declaration.development_base(catalogue.root_path(base_reference["path"]), base_reference["sha256"])
    retained = {key: row for key, row in graph.items() if key[0] != "null"}
    for sid, item in base["sources"].items():
        row = item["original_pool_row"]
        retained[("null", sid)] = {
            "origin_parent": row["origin_parent"], "raw_fits_sha256": row["raw_fits_sha256"],
            "component_id": row["metadata_dependency_component_id"],
            "historically_exposed_component": "True",
        }
    return retained, {
        "status": "FIRST_OUTCOME_BLIND_DEVELOPMENT_DECLARATION",
        "base_manifest": base_reference, "sources": len(base["sources"]),
        "membership_sha256": declaration.MEMBERSHIP_SHA,
        "ordered_source_ids_sha256": declaration.ORDER_SHA,
        "selection_seed": 20261002, "transform_seed": 20261002,
        "component_basis": "METADATA_DEPENDENCY; quarter/scene independence unresolved",
        "detector_exposure": "COMMITTED_STOPPED_DEVELOPMENT_DETECTOR",
        "original_input_graph_retained_as_history": True,
    }


def report_input_graph(entries, graph_path, role, null_runs, null_base):
    if role not in {"development_replay", "final_replay"} or not (entries or null_runs):
        raise ValueError("Replay requires a declared role and control reports")
    if null_base is not None and (role != "development_replay" or not null_runs):
        raise ValueError("A declared development null base requires its opened paired runs")
    audit, graph, graph_inputs = input_graph(graph_path, role)
    declaration = None
    if null_base is not None:
        graph, declaration = declared_null_graph(graph, role, null_base)
    return audit, graph, graph_inputs, declaration


def bind_null_measurement(report, declaration):
    if declaration is not None:
        report["null_population_declaration"] = declaration
        reference = declaration["base_manifest"]
        report["inputs"][reference["path"]] = reference["sha256"]
    if report["null_cases"]:
        report["null_measurement"] = null_population(report["null_cases"])
    return report


def build_report(entries, graph_path, role, run_spec_path=None, null_runs=None, null_base=None):
    audit, graph, graph_inputs, declaration = report_input_graph(entries, graph_path, role, null_runs, null_base)
    inputs = dict(graph_inputs)
    controls = []
    groups = {"cases": [], "morphology_cases": [], "diagnostic_cases": []}
    for entry in entries:
        if append_native_report(entry, role, graph, groups, controls, inputs):
            continue
        control = read_control_run(entry, role)
        for category, record in control["records"]:
            group, case = assemble_case(category, record, control, graph)
            groups[group].append(case)
        inputs.update({str(control["path"]): catalogue.digest(control["path"])})
        inputs.update(control["report"]["inputs"])
        inputs[str(control["run"] / "vetting.json")] = catalogue.digest(
            control["run"] / "vetting.json"
        )
        controls.append(
            {
                name: control[name]
                for name in (
                    "reference",
                    "policy",
                    "execution",
                    "preparation",
                    "manifest",
                    "run",
                    "binary",
                )
            }
        )
    groups["null_cases"] = []
    if null_runs:
        null_controls, groups["null_cases"], null_inputs = read_null_runs(
            null_runs, graph, role, null_base
        )
        controls.extend(null_controls)
        inputs.update(null_inputs)
    elif role == "final_replay":
        raise ValueError("Final replay requires the complete frozen null runs")
    frozen = frozen_execution(controls, audit, role, run_spec_path)
    if role == "final_replay" and frozen is None:
        raise ValueError("Final replay requires frozen automatic disposition execution")
    cases = [case for rows in groups.values() for case in rows]
    inputs.update(evidence_inputs(cases))
    validate_report_trials(cases, graph)
    report = {
        "schema": 1,
        "status": "OPENED_REPORT_ASSEMBLED",
        "replay_role": role,
        "outcome_contract": OUTCOME_CONTRACT,
        "event_matching_contract": EVENT_MATCHING_CONTRACT,
        **groups,
        "population": {
            "status": "MEASURED"
            if audit.get("joint_independence_complete") is True and declaration is None
            and not any(
                case.get("native_prospective_contract") == native.PROTOCOL
                for case in cases
            )
            else "INPUT_INDEPENDENCE_INCOMPLETE",
            "declared_cases": len(cases),
            "by_cohort": dict(Counter(case["truth_cohort"] for case in cases)),
            "overlap_with_detector_development": {
                "historically_exposed_components": {
                    "count": sum(
                        case["historically_exposed_component"] for case in cases
                    )
                }
            },
        },
        "boi_catalogue_enabled": frozen is not None,
        "input_graph": graph_inputs,
        "control_reports": [control["reference"] for control in controls],
        "detector_binaries": [
            control["binary"] for control in controls if control["binary"] is not None
        ],
        "inputs": {**inputs, str(Path(__file__)): catalogue.digest(Path(__file__))},
        "claim_boundary": (
            "Opened reporting is not release evidence. Final rates require the "
            "frozen independent BOI-enabled experiment; ordinary "
            "unknown-truth trials remain separate diagnostics."
        ),
    }
    bind_null_measurement(report, declaration)
    importlib.import_module("vetting").replay_identity_sets(report)
    return bind_replay_execution(report, frozen, role)


def bind_release(policy_path, references, output):
    vetting = importlib.import_module("vetting")
    policy = catalogue.read(policy_path)
    if policy.get("status") != "FROZEN_EVALUATION":
        raise ValueError("Release assembly requires the frozen evaluation policy")
    vetting.validate_operational_policy(policy)
    if policy.get("implementation") != implementation_hashes():
        raise ValueError(
            "Release assembly implementation differs from the frozen replay"
        )
    required = {
        "run_spec",
        "detector",
        "disposition_calibration",
        "development_replay",
        "final_replay",
    }
    if set(references) != required:
        raise ValueError("Release assembly requires the five frozen input artifacts")
    paths = {
        name: vetting.verified_release_artifact(reference)
        for name, reference in references.items()
    }
    development = catalogue.read(paths["development_replay"])
    final = catalogue.read(paths["final_replay"])
    spec = catalogue.read(paths["run_spec"])
    if spec.get("final_audit") != vetting.release_audit_spec(policy):
        raise ValueError("Run specification differs from the frozen release policy")
    vetting.validate_development_execution(development)
    vetting.validate_replay_independence(development, final)
    vetting.validate_release_results(final.get("release_checks", {}), final, spec)
    audit = vetting.detector_replay_audit(references, development, final)
    output = dump_path(output)
    audit_path = output / "detector-final-audit.json"
    policy_path = output / "release-policy.json"
    if audit_path.exists() or policy_path.exists():
        raise ValueError("Release assembly must use a new output directory")
    released = {
        **policy,
        "status": "RELEASED",
        "current_state": {
            "automatic_assignment": "ENABLED",
            "reason": (
                "Passed frozen independent audit; operational pilot required "
                "before the full survey."
            ),
        },
        "release_binding": references,
        "release_checks": final["release_checks"],
    }
    states = {
        name: catalogue.read(path) for name, path in paths.items() if name != "detector"
    }
    states["detector_final_audit"] = audit
    vetting.validate_release_artifact_states(states, references, released)
    pixel_parent_audit.write_json(audit_path, audit)
    released["release_binding"] = {
        **references,
        "detector_final_audit": {
            "path": str(audit_path),
            "sha256": catalogue.digest(audit_path),
        },
    }
    vetting.validate_release_contract(released)
    pixel_parent_audit.write_json(policy_path, released)
    return policy_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--input-graph", required=True, type=Path)
    parser.add_argument(
        "--role", required=True, choices=("development_replay", "final_replay")
    )
    parser.add_argument("--run-spec", type=Path)
    parser.add_argument(
        "--null-runs", type=Path, nargs=2, metavar=("INVERTED", "QUARTER_SCRAMBLED")
    )
    parser.add_argument("--null-base-manifest", type=Path)
    parser.add_argument("--null-base-sha256")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.null_base_manifest) != bool(args.null_base_sha256):
        parser.error("Both development null base path and SHA256 are required")
    report = build_report(
        catalogue.read(args.reports),
        args.input_graph,
        args.role,
        args.run_spec,
        args.null_runs,
        {"path": str(args.null_base_manifest.resolve()), "sha256": args.null_base_sha256}
        if args.null_base_manifest else None,
    )
    pixel_parent_audit.write_json(dump_path(args.out), report)
    print(json.dumps({"status": report["status"], "population": report["population"]}))
    if report["status"] == "REPLAY_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
