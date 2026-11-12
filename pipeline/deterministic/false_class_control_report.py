"""Reconcile exact false-class injections with detector and vetter outcomes."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.deterministic import catalogue_control_report as catalogue
from pipeline.deterministic import false_class_injections, pixel_parent_audit


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def periodic_matches(truth, proposal):
    reference = truth["period_days"]
    matches = []
    for hypothesis in proposal["supplied_hypotheses"]:
        period = hypothesis["period_days"]
        harmonic = catalogue.accepted_harmonic(period, reference)
        if harmonic is None:
            continue
        duration = hypothesis["duration_days"]
        phase_error = abs(
            catalogue.phase_delta(
                hypothesis["epoch_bkjd"],
                truth["epoch_bkjd"],
                min(period, reference),
            )
        )
        tolerance = max(duration, truth["duration_days"]) / 2 + catalogue.CADENCE_DAYS
        if phase_error > tolerance:
            continue
        rust = hypothesis.get("rust_record")
        shallow = hypothesis.get("shallow_record")
        if (rust is None) == (shallow is None):
            raise ValueError(
                "Periodic hypothesis must have exactly one detector record"
            )
        matches.append(
            {
                "detector_signal_id": hypothesis["signal_id"],
                "detector_origin": (
                    "rust_v6_deterministic" if rust is not None else "shallow_bls_v1"
                ),
                "classification": (
                    rust["classification"]
                    if rust is not None
                    else "SHALLOW_BLS_PROPOSAL"
                ),
                "followup_eligible": hypothesis["followup_eligible"],
                "period_days": period,
                "harmonic": harmonic,
                "phase_error_days": phase_error,
                "search_mes": float(rust["search_mes"]) if rust is not None else None,
                "shallow_search_snr": (
                    float(shallow["search_snr"]) if shallow is not None else None
                ),
            }
        )
    return sorted(
        matches,
        key=lambda item: (
            item["harmonic"] != 1,
            abs(item["period_days"] / reference - 1),
            item["phase_error_days"],
        ),
    )


def matched_vetting(source, detector_signal_id):
    if source is None:
        return None
    matches = [
        signal
        for signal in source["signals"]
        if detector_signal_id in catalogue.detector_ids(source, signal)
    ]
    if len(matches) > 1:
        raise ValueError("One detector signal maps to multiple vetted signals")
    if not matches:
        return None
    signal = matches[0]
    disposition = next(
        (
            row
            for row in source["disposition"].get("signals", [])
            if row["signal_id"] == signal["signal_id"]
        ),
        {},
    )
    return {
        "signal_id": signal["signal_id"],
        "triage": catalogue.terminal(signal),
        "automatic_disposition": disposition.get("disposition", "UNAVAILABLE"),
        "automatic_reason_codes": disposition.get("reason_codes", []),
    }


def validate_input_products(controls, item, row):
    pixels = item["pixels"]
    if set(map(int, pixels)) != set(row["injected_quarters"]):
        raise ValueError("False-class pixel quarters differ from truth")
    if any(
        str(pixel["base_parent_kic"]) != row["parent_tpf"] for pixel in pixels.values()
    ):
        raise ValueError("False-class resolved pixel parent changed")
    array_path = controls / item["arrays"]
    if catalogue.digest(array_path) != item["sha256"]:
        raise ValueError("False-class LC2 arrays changed")
    with np.load(array_path, allow_pickle=False) as arrays:
        if set(np.unique(arrays["quarter"])) != set(row["injected_quarters"]):
            raise ValueError("False-class LC2 quarters differ from truth")
        if not set(np.unique(arrays["qdiag_quarter"])) <= set(row["injected_quarters"]):
            raise ValueError("False-class diagnostic quarters differ from truth")
    for pixel in pixels.values():
        if catalogue.digest(controls / pixel["path"]) != pixel["sha256"]:
            raise ValueError("False-class pixel product changed")


def frozen_generator_inputs(report, archive=None):
    expected = report["generator"]
    if not isinstance(expected, dict) or set(expected) != set(
        false_class_injections.generator_identity()
    ):
        raise ValueError("Frozen false-class generator file set changed")
    root = (ROOT / archive).resolve() if archive is not None else ROOT
    inputs = {}
    for name, sha256 in expected.items():
        path = root / name
        if not path.is_file() or catalogue.digest(path) != sha256:
            raise ValueError(
                f"Frozen generator source is unavailable or changed: {path}; "
                "supply --generator-archive with the exact frozen source files"
            )
        inputs[path] = sha256
    return inputs


def background_truth(truth, background_only):
    if not background_only:
        return truth
    selected = [row for row in truth if row["category"] == "background_contaminant"]
    if not selected:
        raise ValueError("False-class partition has no background controls")
    return selected


def validate_controls(controls, generator_archive=None, background_only=False):
    report = read(controls / "materialization.json")
    if report["protocol"] != false_class_injections.MODEL or report["status"] not in {
        "FROZEN",
        "PROBE",
    }:
        raise ValueError("False-class materialization is incomplete")
    for name, expected in report["artifacts"].items():
        if catalogue.digest(controls / name) != expected:
            raise ValueError(f"False-class artifact changed: {name}")
    if generator_archive is not None and report["status"] != "FROZEN":
        raise ValueError("Generator archives require frozen false-class controls")
    if report["status"] == "FROZEN":
        frozen_generator_inputs(report, generator_archive)
        if report["candidate_pool_sha256"] != catalogue.digest(
            false_class_injections.POOL / "freeze.json"
        ):
            raise ValueError("Frozen false-class candidate pool changed")
        seed = report["development_seed"]
        seed_report = Path(seed["path"])
        if (
            catalogue.digest(seed_report) != seed["sha256"]
            or catalogue.digest(seed_report.with_name("parent-audit.json"))
            != seed["parent_audit_sha256"]
            or read(seed_report)["generator"] != seed["generator"]
            or read(controls / "development-source-ids.json")
            != read(seed_report.with_name("development-source-ids.json"))
        ):
            raise ValueError("Frozen false-class development seed changed")
    truth = read_jsonl(controls / "truth.jsonl")
    source_ids = list(map(str, read(controls / "source-ids.json")))
    manifest = read(controls / "input-manifest.json")
    if (
        manifest.get("schema") != 1
        or manifest.get("protocol") != report["protocol"]
        or "Four observed-quarter excerpts" not in manifest.get("measurement_scope", "")
        or len(source_ids) != len(set(source_ids))
        or source_ids != [row["source_id"] for row in truth]
        or set(source_ids) != set(manifest["sources"])
        or len({row["parent_tpf"] for row in truth}) != len(truth)
    ):
        raise ValueError("False-class source or parent identity changed")
    for row in background_truth(truth, background_only):
        item = manifest["sources"][row["source_id"]]
        validate_input_products(controls, item, row)
    return report, source_ids, truth


def matched_status(matches):
    if not matches:
        return "IDENTITY_NOT_RECOVERED"
    if any(
        match["vetting"] is None and match["followup_eligible"] for match in matches
    ):
        return "VETTING_MISSING"
    if any(
        match["vetting"] is not None
        and match["vetting"]["triage"]["status"] != "DISMISSED"
        for match in matches
    ):
        return "UNRESOLVED"
    if any(match["vetting"] is not None for match in matches):
        return "VETTING_DISMISSED"
    return "DETECTOR_REJECTED"


def trial(truth, proposal, source, boi_catalogue_enabled):
    matches = periodic_matches(truth, proposal)
    for match in matches:
        match["vetting"] = matched_vetting(source, match["detector_signal_id"])
    disposition = source.get("disposition", {}) if source else {}
    boi_catalogue = boi_catalogue_enabled and catalogue.emitted_candidate(disposition)
    incomplete = boi_catalogue_enabled and catalogue.incomplete_disposition(source)
    abstention = boi_catalogue_enabled and catalogue.scientific_abstention(source)
    truth_status = matched_status(matches)
    status = truth_status
    if boi_catalogue:
        status = "AUTOMATIC_PC"
    elif incomplete:
        status = "INCOMPLETE_EXECUTION"
    elif abstention:
        status = "SCIENTIFIC_ABSTENTION"
    return {
        "source_id": truth["source_id"],
        "parent_tpf": truth["parent_tpf"],
        "category": truth["category"],
        "partition": truth["partition"],
        "artifact_kind": truth["artifact_kind"],
        "injected_period_days": truth["period_days"],
        "expected_search_mes": truth["expected_search_mes"],
        "detector_classifications": dict(
            Counter(row["classification"] for row in proposal["rust_rows"])
        ),
        "matches": matches,
        "signals": [
            catalogue.trial_signal(source, signal) for signal in source["signals"]
        ]
        if source
        else [],
        "truth_match_status": truth_status,
        "automatic_disposition_status": disposition.get("status", "NOT_APPLICABLE"),
        "boi_catalogue_on_parent": boi_catalogue if boi_catalogue_enabled else None,
        "disposition_incomplete": incomplete if boi_catalogue_enabled else None,
        "scientific_abstention": abstention if boi_catalogue_enabled else None,
        "status": status,
    }


def false_boi_rate(cases, enabled, partition):
    if not enabled:
        return {"status": "NOT_MEASURED_BOI_CATALOGUE_DISABLED"}
    measured = catalogue.parent_outcomes(
        [{**row, "boi_catalogue": row["boi_catalogue_on_parent"]} for row in cases]
    )
    if partition != "evaluation":
        measured["upper_bound"] = None
    return measured


def partition_truth(controls, all_source_ids, all_truth, partition, background_only):
    if partition is None:
        source_ids, truth = all_source_ids, all_truth
    else:
        source_ids = list(map(str, read(controls / f"{partition}-source-ids.json")))
        truth = [row for row in all_truth if row["partition"] == partition]
        if source_ids != [row["source_id"] for row in truth]:
            raise ValueError("False-class partition identities changed")
    materialized_sources = len(source_ids)
    truth = background_truth(truth, background_only)
    return [row["source_id"] for row in truth], truth, materialized_sources


def build_report(
    controls, run, partition=None, generator_archive=None, background_only=False
):
    controls, run = Path(controls), Path(run)
    frozen, all_source_ids, all_truth = validate_controls(
        controls, generator_archive, background_only
    )
    policy_path = catalogue.root_path(read(run / "vetting.json")["disposition_policy"])
    policy = read(policy_path)
    if partition == "evaluation" and (
        frozen["status"] != "FROZEN"
        or policy.get("boi_catalogue_enabled") is not True
        or policy["status"] not in {"FROZEN_EVALUATION", "RELEASED"}
    ):
        raise ValueError("False-class evaluation requires frozen controls and policy")
    source_ids, truth, materialized_sources = partition_truth(
        controls, all_source_ids, all_truth, partition, background_only
    )
    _, detection, results, rows = catalogue.validate_run(run, source_ids)
    execution = results["disposition_execution"]
    enabled = execution["boi_catalogue_enabled"]
    if partition == "evaluation" and (
        frozen["status"] != "FROZEN"
        or not enabled
        or policy["status"] not in {"FROZEN_EVALUATION", "RELEASED"}
        or (
            policy["status"] == "FROZEN_EVALUATION"
            and execution["evaluation_role"] != "final_replay"
        )
    ):
        raise ValueError("False-class evaluation requires frozen controls and policy")
    trials = []
    for item in truth:
        source_id = item["source_id"]
        proposal = read(run / "proposals" / f"{source_id}.json")
        if str(proposal["source_id"]) != source_id:
            raise ValueError("False-class proposal source identity changed")
        source = rows.get(source_id)
        if proposal["alerts"] and source is None:
            raise ValueError("False-class alerted source lacks vetting evidence")
        trials.append(trial(item, proposal, source, enabled))
    categories = {}
    for category in sorted({row["category"] for row in trials}):
        cases = [row for row in trials if row["category"] == category]
        categories[category] = {
            "trials": len(cases),
            "period_or_harmonic_recovered": sum(bool(row["matches"]) for row in cases),
            "exact_period_recovered": sum(
                any(match["harmonic"] == 1 for match in row["matches"]) for row in cases
            ),
            "outcomes": dict(Counter(row["status"] for row in cases)),
            "matched_triage_failed_trials": sum(
                row["truth_match_status"]
                not in {"DETECTOR_REJECTED", "VETTING_DISMISSED"}
                for row in cases
            ),
            "false_boi_rate": false_boi_rate(cases, enabled, partition),
        }
    artifacts = [
        controls / "materialization.json",
        controls / "truth.jsonl",
        controls / "source-ids.json",
        controls / "input-manifest.json",
        run / "preparation.json",
        run / "detection.json",
        run / "discovery-manifest.json",
        run / "queue.json",
        run / "vetting.json",
        run / "vetting/results.json",
        run / "vetting/manifest.json",
        run / "vetting/stage-timings.json",
        policy_path,
        Path(__file__),
        Path(catalogue.__file__),
        Path(__file__).resolve().parents[1] / "runtime.py",
    ]
    if partition is not None:
        artifacts.append(controls / f"{partition}-source-ids.json")
    generator_inputs = (
        frozen_generator_inputs(frozen, generator_archive)
        if frozen["status"] == "FROZEN"
        else {}
    )
    artifacts.extend(generator_inputs)
    source_hashes = {
        path.name: catalogue.digest(path)
        for path in sorted((run / "vetting/sources").glob("*.json"))
    }
    evidence_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": 2,
        "outcome_contract": catalogue.OUTCOME_CONTRACT,
        "status": "COMPLETE_FALSE_CLASS_CONTROL_REPORT",
        "partition": partition or "all",
        "background_only": background_only,
        "frozen_generator_inputs": {
            pixel_parent_audit.display_path(path): sha256
            for path, sha256 in generator_inputs.items()
        },
        "claim_boundary": (
            "A false-class upper bound is available only for a frozen materialization "
            "under a frozen BOI-enabled policy with complete execution. "
            "Any BOI designation on any alias of a false parent counts. Scientific "
            "abstentions and periodic identity non-recovery are reported separately."
        ),
        "population": {
            "declared_sources": len(source_ids),
            "materialized_partition_sources": materialized_sources,
            "unmeasured_materialized_sources": materialized_sources - len(source_ids),
            "prepared_sources": detection["sources"],
            "queued_sources": detection["alerted_sources"],
            "vetting_sources": results["sources"],
        },
        "policy": {
            "id": policy["policy_id"],
            "status": policy["status"],
            "boi_catalogue_enabled": enabled,
            "evaluation_role": execution["evaluation_role"],
            "production_boi_enabled": execution["production_boi_enabled"],
        },
        "categories": categories,
        "trials": trials,
        "trials_sha256": catalogue.json_digest(trials),
        "source_evidence": {
            "files": len(source_hashes),
            "digest_ledger_sha256": evidence_hash,
        },
        "inputs": {
            pixel_parent_audit.display_path(path): catalogue.digest(path)
            for path in artifacts
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--partition", choices=("development", "evaluation"))
    parser.add_argument("--generator-archive", type=Path)
    parser.add_argument("--background-only", action="store_true")
    args = parser.parse_args()
    report = build_report(
        catalogue.root_path(args.controls),
        catalogue.root_path(args.run),
        args.partition,
        args.generator_archive,
        args.background_only,
    )
    pixel_parent_audit.write_json(catalogue.root_path(args.out), report)
    print(
        json.dumps(
            {
                "population": report["population"],
                "policy": report["policy"],
                "categories": report["categories"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
