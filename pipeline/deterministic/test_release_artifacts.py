"""Exercise artifact publication and launch guards with explicit synthetic outcomes."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.deterministic import full_survey, release_report
from pipeline.deterministic.test_run import (
    adapter,
    bound_outcome_release,
    opened_cap_cases,
)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    policy = bound_outcome_release(tmp_path)
    policy["status"] = "FROZEN_EVALUATION"
    path = tmp_path / "frozen-policy.json"
    adapter.save(path, policy)
    references = {
        name: reference
        for name, reference in policy["release_binding"].items()
        if name != "detector_final_audit"
    }
    monkeypatch.setattr(release_report, "dump_path", Path)
    return path, policy, references, tmp_path / "published"


def test_publication_passes_actual_launch_and_resume_guards(experiment):
    path, frozen, references, output = experiment
    frozen_hash = adapter.digest(path)
    published = release_report.bind_release(path, references, output)
    released = adapter.read(published)
    assert released["status"] == "RELEASED"
    assert released["definition_sha256"] == frozen["definition_sha256"]
    assert adapter.digest(path) == frozen_hash
    assert released["release_binding"]["final_replay"] == references["final_replay"]
    full_survey.verified_release_contract(released, adapter)
    full_survey.verify_frozen_release(published, adapter.digest(published), adapter)
    assert (
        adapter.evaluation_preflight(SimpleNamespace(disposition_policy=published), [])
        is None
    )


def test_publication_refuses_to_overwrite_a_release(experiment):
    path, _, references, output = experiment
    published = release_report.bind_release(path, references, output)
    original = adapter.digest(published)
    with pytest.raises(ValueError, match="new output directory"):
        release_report.bind_release(path, references, output)
    assert adapter.digest(published) == original


@pytest.mark.parametrize("defect", ["missing", "reordered", "last_failure"])
def test_changed_stage_contract_cannot_publish_even_after_rehash(experiment, defect):
    path, policy, references, output = experiment
    vetting = importlib.import_module("vetting")
    if defect == "missing":
        del policy["triage_contract"]
    elif defect == "reordered":
        policy["triage_contract"]["stage_order"].reverse()
    else:
        policy["triage_contract"]["state_rule"] = "LAST_NONPASS_IN_STAGE_ORDER"
    policy["definition_sha256"] = vetting.disposition_definition_sha256(policy)
    adapter.save(path, policy)
    with pytest.raises(ValueError, match="triage contract"):
        release_report.bind_release(path, references, output)
    with pytest.raises(ValueError, match="triage contract"):
        vetting.validate_release_contract(policy)
    assert not (output / "release-policy.json").exists()


@pytest.mark.parametrize(
    "defect", ["opened", "legacy", "missing_alias", "false_boi", "disabled"]
)
def test_unpassed_replays_do_not_publish_a_policy(experiment, defect):
    path, _, references, output = experiment
    role = "development_replay" if defect == "opened" else "final_replay"
    report_path = Path(references[role]["path"])
    report = adapter.read(report_path)
    if defect == "opened":
        report.update(status="OPENED_REPORT_ASSEMBLED", boi_catalogue_enabled=False)
    elif defect == "legacy":
        del report["outcome_contract"]
    elif defect == "missing_alias":
        report["cases"][300]["alert_ledger"]["signal_dispositions"].pop()
    elif defect == "disabled":
        report["boi_catalogue_enabled"] = False
    else:
        ledger = report["cases"][300]["alert_ledger"]
        ledger["signal_dispositions"][0]["disposition"] = "BOI"
        ledger["boi_signals"] += 1
    adapter.save(report_path, report)
    references[role]["sha256"] = adapter.digest(report_path)
    with pytest.raises(ValueError):
        release_report.bind_release(path, references, output)
    assert not (output / "release-policy.json").exists()


@pytest.mark.parametrize(
    "parameter",
    [
        "assessment_rules",
        "pipeline_contract",
        "implementation",
        "outcome_contract",
        "event_matching_contract",
        "lane_gates",
    ],
)
def test_calibration_must_describe_the_selected_parameters(experiment, parameter):
    _, policy, _, _ = experiment
    vetting = importlib.import_module("vetting")
    binding = policy["release_binding"]
    states = {
        name: adapter.read(ref["path"])
        for name, ref in binding.items()
        if name != "detector"
    }
    states["disposition_calibration"]["parameters"][parameter] = "changed"
    with pytest.raises(ValueError, match="differs from the frozen experiment"):
        vetting.validate_calibration_binding(states, binding, policy)


def change_cap_spec(spec, defect):
    if defect == "missing_cohort":
        del spec["false_boi_caps"]["guide_star_family"]
    elif defect == "cap":
        spec["false_boi_caps"]["null"] = 0.02
    elif defect == "confidence":
        spec["confidence"] = 0.9
    else:
        spec["status"] = "BOUND_METADATA_INPUTS_PENDING"


@pytest.mark.parametrize("defect", [
    "missing_spec", "spec_drift", "missing_cohort", "cap", "confidence", "pending",
    "trials", "false_boi", "upper_bound", "ledger_cap", "feasible", "incomplete",
    "float_count", "missing_null", "null_source_denominator", "over_cap",
])
def test_launch_caps_require_bound_spec_and_reconciled_opened_outcomes(tmp_path, defect):
    vetting = importlib.import_module("vetting")
    policy = adapter.read(adapter.ROOT / "pipeline/vetting/disposition-policy.json")
    audit = policy["release_plan"]["final_audit"]
    cases, nulls = opened_cap_cases()
    report = {"cases": cases, "null_cases": nulls}
    path = tmp_path / "cap-spec.json"
    spec = {"status": "BOUND_OPENED_TEST_SPEC", "confidence": audit["confidence"],
            "false_boi_caps": copy.deepcopy(audit["opened_false_boi_caps"])}
    adapter.save(path, spec)
    calibration = {
        "spec": {"path": str(path), "sha256": adapter.digest(path)},
        "outcomes": {"feasible": True, "incomplete_cases": 0,
                     "false_boi": vetting.calibration_cap_outcomes(
                         report, audit["opened_false_boi_caps"], audit["confidence"])},
    }
    vetting.validate_calibration_caps(calibration, report, policy)
    assert calibration["outcomes"]["false_boi"]["null"]["trials"] == 800
    if defect == "missing_spec":
        del calibration["spec"]
    elif defect == "spec_drift":
        path.write_text("{}")
    elif defect in {"missing_cohort", "cap", "confidence", "pending"}:
        change_cap_spec(spec, defect)
        adapter.save(path, spec)
        calibration["spec"]["sha256"] = adapter.digest(path)
    elif defect in {"feasible", "incomplete"}:
        calibration["outcomes"]["feasible" if defect == "feasible" else "incomplete_cases"] = (
            False if defect == "feasible" else 1
        )
    elif defect == "missing_null":
        report["null_cases"] = []
    elif defect == "over_cap":
        for row in nulls[0]["variants"]:
            for proposal in row["proposals"]:
                proposal["gate_path"] = [{"stage": stage, "decision": "PASS"}
                                         for stage in ("shape", "orbital", "instrument")]
        calibration["outcomes"]["false_boi"] = vetting.calibration_cap_outcomes(
            report, audit["opened_false_boi_caps"], audit["confidence"]
        )
    else:
        ledger = calibration["outcomes"]["false_boi"]["null"]
        key = {"ledger_cap": "cap", "float_count": "trials",
               "null_source_denominator": "trials"}.get(defect, defect)
        ledger[key] = (800.0 if defect == "float_count" else 1 if defect == "null_source_denominator"
                       else ledger[key] + 1)
    with pytest.raises(ValueError):
        vetting.validate_calibration_caps(calibration, report, policy)


@pytest.mark.parametrize("defect", ["missing_cohort", "cap", "confidence", "pending"])
def test_rehashed_cap_spec_cannot_publish_a_policy(experiment, defect):
    path, policy, references, output = experiment
    vetting = importlib.import_module("vetting")
    calibration_path = Path(references["disposition_calibration"]["path"])
    calibration = adapter.read(calibration_path)
    spec_path = Path(calibration["spec"]["path"])
    spec = adapter.read(spec_path)
    change_cap_spec(spec, defect)
    adapter.save(spec_path, spec)
    calibration["spec"]["sha256"] = adapter.digest(spec_path)
    adapter.save(calibration_path, calibration)
    references["disposition_calibration"]["sha256"] = adapter.digest(calibration_path)
    run_spec_path = Path(references["run_spec"]["path"])
    run_spec = adapter.read(run_spec_path)
    run_spec["disposition_calibration"] = references["disposition_calibration"]
    adapter.save(run_spec_path, run_spec)
    references["run_spec"]["sha256"] = adapter.digest(run_spec_path)
    identities = {
        "run_spec_sha256": references["run_spec"]["sha256"],
        "calibration_sha256": references["disposition_calibration"]["sha256"],
    }
    policy["evaluation_binding"].update(identities)
    policy["definition_sha256"] = vetting.disposition_definition_sha256(policy)
    for role in ("development_replay", "final_replay"):
        report_path = Path(references[role]["path"])
        report = adapter.read(report_path)
        report.update(identities, disposition_definition_sha256=policy["definition_sha256"])
        adapter.save(report_path, report)
        references[role]["sha256"] = adapter.digest(report_path)
    adapter.save(path, policy)
    with pytest.raises(ValueError, match="Calibration cap specification"):
        release_report.bind_release(path, references, output)
    assert not (output / "release-policy.json").exists()


@pytest.mark.parametrize(
    "defect", ["proposal_count", "replay_reference", "legacy_audit"]
)
def test_rehashed_detector_audit_must_match_the_replays(experiment, defect):
    _, original, _, _ = experiment
    policy = copy.deepcopy(original)
    policy["status"] = "RELEASED"
    reference = policy["release_binding"]["detector_final_audit"]
    path = Path(reference["path"])
    audit = adapter.read(path)
    if defect == "proposal_count":
        audit["proposal_measurements"]["final_replay"]["periodic_planet"][
            "proposed_parents"
        ] += 1
    elif defect == "replay_reference":
        audit["replays"]["final_replay"]["sha256"] = "0" * 64
    else:
        audit.update(
            status="DETECTOR_AUDIT_PASSED_DOWNSTREAM_AUDIT_REQUIRED",
            checks={"test": True},
        )
    adapter.save(path, audit)
    reference["sha256"] = adapter.digest(path)
    with pytest.raises(ValueError, match="audit differs|have not passed"):
        full_survey.verified_release_contract(policy, adapter)


@pytest.mark.parametrize("artifact", ["calibration_evidence", "cohort"])
def test_resume_rechecks_nested_frozen_inputs(experiment, artifact):
    path, _, references, output = experiment
    published = release_report.bind_release(path, references, output)
    checksum = adapter.digest(published)
    released = adapter.read(published)
    if artifact == "calibration_evidence":
        calibration = adapter.read(references["disposition_calibration"]["path"])
        changed = Path(calibration["development_evidence"]["path"])
    else:
        cohort = released["evaluation_binding"]["final_replay"]["cohorts"][0]
        changed = Path(cohort["path"])
    changed.write_text("changed retained bytes")
    with pytest.raises(ValueError, match="unavailable or changed"):
        full_survey.verify_frozen_release(published, checksum, adapter)


def test_rehashed_replay_cannot_substitute_a_different_detector(experiment):
    _, policy, references, _ = experiment
    vetting = importlib.import_module("vetting")
    binary = Path(references["detector"]["path"]).with_name("other-detector")
    binary.write_bytes(b"different detector bytes")
    path = Path(references["final_replay"]["path"])
    report = adapter.read(path)
    report["detector_binaries"] = [
        {"path": str(binary), "sha256": adapter.digest(binary)}
    ]
    adapter.save(path, report)
    policy["release_binding"]["final_replay"]["sha256"] = adapter.digest(path)
    policy["status"] = "RELEASED"
    with pytest.raises(ValueError, match="binary differs from the frozen artifact"):
        vetting.validate_release_contract(policy)


def test_frozen_assembly_rejects_a_different_replayed_binary(experiment):
    _, policy, references, output = experiment
    config = adapter.read(adapter.ROOT / "pipeline/vetting/config.json")
    config.update(quality_bitmask=-1, discovery_contract=adapter.detector_contract())
    output.mkdir()
    config_path = output / "vetting.json"
    adapter.save(config_path, config)
    role = "development_replay"
    control = {
        "policy": policy,
        "run": output,
        "binary": {"path": "different-detector", "sha256": "0" * 64},
        "execution": {"boi_catalogue_enabled": True, "evaluation_role": role},
        "preparation": {
            "implementation": policy["implementation"],
            "evaluation_scope": {
                "evaluation_role": role,
                "cohort": policy["evaluation_binding"][role]["cohorts"][0],
            },
        },
        "manifest": {
            "code_sha256": policy["implementation"],
            "config_sha256": adapter.digest(config_path),
        },
    }
    graph = {"joint_independence_complete": True}
    spec = Path(references["run_spec"]["path"])
    with pytest.raises(ValueError, match="not the frozen independent experiment"):
        release_report.frozen_execution([control], graph, role, spec)
    control["binary"] = references["detector"]
    assert release_report.frozen_execution([control], graph, role, spec)[0] == policy


@pytest.mark.parametrize("status", ["FROZEN_EVALUATION", "RELEASED"])
def test_detector_entry_checks_actual_bytes_before_execution(experiment, status):
    path, policy, references, _ = experiment
    policy["status"] = status
    adapter.save(path, policy)
    args = SimpleNamespace(disposition_policy=path)
    binary = Path(references["detector"]["path"])
    adapter.verify_frozen_detector(args, binary)
    binary.write_bytes(b"different compiled detector")
    with pytest.raises(ValueError, match="binary differs from the frozen"):
        adapter.verify_frozen_detector(args, binary)


def test_single_event_is_a_boi_without_a_period():
    disposition = importlib.import_module("vetting_disposition")
    signal = {
        "signal_id": "single", "period_days": None,
        "events": [{"event_ids": ["a"]}], "event_ids": ["a"],
        "triage": {"status": "ACTIVE", "history": [
            {"stage": stage, "decision": "PASS"} for stage in disposition.GATE_ORDER
        ]},
    }
    assert disposition.triage_disposition(signal) == (
        "BOI", ["ALL_PIPELINE_GATES_PASS"]
    )
