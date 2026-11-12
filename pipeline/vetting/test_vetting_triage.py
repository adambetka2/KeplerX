"""Record mode measures every gate; both modes label from the first failing gate."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import vetting_flux
import vetting_triage
from vetting_disposition import (
    GATE_ORDER,
    TRIAGE_CONTRACT,
    check_gate_record,
    first_failed_gate,
)

GATES = {
    "signals": "shape_gate",
    "orbital": "orbital_gate",
    "instrument": "instrument_gate",
    "localization": "localization_gate",
    "contamination": "contamination_gate",
}
FAILURES = {
    ("dismissed", "shape_gate"): "DISMISS",
    ("held", "instrument_gate"): "CONTINUE_INCONCLUSIVE",
}


def run_gates(monkeypatch, execution):
    for gate in GATES.values():
        monkeypatch.setattr(
            vetting_triage,
            gate,
            lambda row, signal, policy, gate=gate: {
                "decision": FAILURES.get((signal["signal_id"], gate), "PASS"),
                "reason": gate,
                "evidence": {},
            },
        )
    monkeypatch.setattr(
        vetting_flux, "append_refined_pixel_jobs", lambda row, config: None
    )
    row = {"signals": [{"signal_id": name} for name in ("dismissed", "held", "passed")]}
    config = {"triage": {"enabled": True}, "triage_execution": execution}
    measured = []
    for stage in GATES:
        row = vetting_triage.apply_gate(row, stage, config)
        measured.append([s["signal_id"] for s in vetting_triage.measured_signals(row)])
    return row, measured


def test_record_mode_measures_every_gate_with_the_skip_mode_label(monkeypatch):
    skip, skip_measured = run_gates(monkeypatch, "SKIP_AFTER_FIRST_FAILURE")
    record, record_measured = run_gates(monkeypatch, "RECORD_EVERY_GATE")
    assert skip_measured == [["held", "passed"]] * 2 + [["passed"]] * 3
    assert record_measured == [["dismissed", "held", "passed"]] * 5
    for row in (skip, record):
        assert [signal["triage"]["status"] for signal in row["signals"]] == [
            "DISMISSED",
            "HELD",
            "ACTIVE",
        ]
        assert [
            (first_failed_gate(signal) or {}).get("stage") for signal in row["signals"]
        ] == ["shape", "instrument", None]
    dismissed, held, _ = skip["signals"]
    assert [item["stage"] for item in held["triage"]["history"]] == list(GATE_ORDER[:3])
    assert dismissed["contamination"]["status"] == "SKIPPED_DISMISSED"
    assert held["host"]["status"] == "SKIPPED_HELD"
    for signal in record["signals"]:
        assert [item["stage"] for item in signal["triage"]["history"]] == list(
            GATE_ORDER
        )
        assert "contamination" not in signal and "host" not in signal


@pytest.mark.parametrize("execution", ["SKIP_AFTER_FIRST_FAILURE", "RECORD_EVERY_GATE"])
@pytest.mark.parametrize("stage,decision", [
    (stage, decision) for stage in GATE_ORDER
    for decision in ("DISMISS", "CONTINUE_INCONCLUSIVE")
    if (stage, decision) != ("instrument", "DISMISS")
])
def test_gate_record_preserves_each_emitted_first_failure(monkeypatch, execution, stage, decision):
    gate = GATES["signals" if stage == "shape" else stage]
    monkeypatch.setitem(FAILURES, ("passed", gate), decision)
    row, _ = run_gates(monkeypatch, execution)
    signal = row["signals"][-1]
    before = copy.deepcopy(signal)
    assert first_failed_gate(signal) == {
        "stage": stage, "decision": decision, "reason": gate, "evidence": {},
    }
    assert signal == before


def test_gate_record_must_follow_the_declared_order():
    passed = [{"stage": stage, "decision": "PASS"} for stage in GATE_ORDER]
    held = passed[:2] + [{"stage": "instrument", "decision": "CONTINUE_INCONCLUSIVE"}]
    check_gate_record({"signal_id": "s", "triage": {"status": "ACTIVE", "history": passed}})
    check_gate_record({"signal_id": "s", "triage": {
        "status": "HELD", "held_stage": "instrument", "history": held,
    }})
    for triage in (
        {"history": passed[:4]},
        {"history": held[1:]},
        {"history": held, "record_every_gate": True},
        {},
    ):
        with pytest.raises(ValueError, match="gate order"):
            check_gate_record({"signal_id": "s", "triage": triage})


def test_policy_binds_the_implemented_triage_contract():
    from runtime import disposition_definition, disposition_definition_sha256

    policy = json.loads(Path(__file__).with_name("disposition-policy.json").read_text())
    assert policy["triage_contract"] == TRIAGE_CONTRACT
    assert disposition_definition(policy)["triage_contract"] == TRIAGE_CONTRACT
    original = disposition_definition_sha256(policy)
    policy["triage_contract"]["state_rule"] = "LAST_NONPASS_IN_STAGE_ORDER"
    assert disposition_definition_sha256(policy) != original


@pytest.mark.parametrize("execution", ["SKIP_AFTER_FIRST_FAILURE", "RECORD_EVERY_GATE"])
@pytest.mark.parametrize("first,later", [
    ("DISMISS", "CONTINUE_INCONCLUSIVE"), ("CONTINUE_INCONCLUSIVE", "DISMISS"),
])
def test_later_failure_cannot_replace_the_first_gate_state(
    monkeypatch, execution, first, later
):
    monkeypatch.setitem(FAILURES, ("passed", "shape_gate"), first)
    monkeypatch.setitem(FAILURES, ("passed", "orbital_gate"), later)
    row, _ = run_gates(monkeypatch, execution)
    signal = row["signals"][-1]
    assert signal["triage"]["status"] == ("DISMISSED" if first == "DISMISS" else "HELD")
    assert first_failed_gate(signal)["stage"] == "shape"
    if execution == "RECORD_EVERY_GATE":
        assert signal["triage"]["history"][1]["decision"] == later


@pytest.mark.parametrize("available,decision", [
    (True, "PASS"), (False, "CONTINUE_INCONCLUSIVE"),
])
def test_real_instrument_gate_matches_the_declared_outcomes(available, decision):
    signal = {"events": [{"unit_id": "a", "centre_bkjd": 1, "event_ids": ["a"]}]}
    row = {"instrument": {
        "selection": {"maximum_units_per_signal": 1},
        "events": [{"event_id": "a", "status": "MEASURED", "in_event_points": 1}] if available else [],
    }}
    result = vetting_triage.instrument_gate(row, signal, {})
    assert result["decision"] == decision
    assert decision in TRIAGE_CONTRACT["allowed_decisions"]["instrument"]


@pytest.mark.parametrize("mutation", [
    "held_after_all_passes", "active_after_dismissal", "wrong_failure_stage",
    "wrong_failure_reason", "unknown_decision",
])
def test_gate_record_rejects_state_drift_from_actual_gate_history(monkeypatch, mutation):
    row, _ = run_gates(monkeypatch, "RECORD_EVERY_GATE")
    dismissed, held, passed = row["signals"]
    signal = held
    if mutation == "held_after_all_passes":
        signal = passed
        signal["triage"].update(status="HELD", held_stage="instrument")
    elif mutation == "active_after_dismissal":
        signal = dismissed
        signal["triage"]["status"] = "ACTIVE"
    elif mutation == "wrong_failure_stage":
        signal["triage"]["held_stage"] = "contamination"
    elif mutation == "wrong_failure_reason":
        signal["triage"]["reason"] = "later_gate"
    else:
        signal["triage"]["history"][0]["decision"] = "FAIL"
    with pytest.raises(ValueError, match="gate (state|decision)"):
        check_gate_record(signal)


def test_boi_cannot_use_an_active_history_with_a_held_state(monkeypatch):
    import vetting_disposition

    row, _ = run_gates(monkeypatch, "RECORD_EVERY_GATE")
    signal = row["signals"][-1]
    signal["events"] = [{}, {}]
    signal["triage"].update(status="HELD", held_stage="instrument")
    with pytest.raises(ValueError, match="gate state"):
        vetting_disposition.triage_disposition(signal)



def test_unknown_triage_execution_fails():
    config = {"triage": {"enabled": True}, "triage_execution": "SKIP"}
    with pytest.raises(ValueError, match="Unknown triage execution"):
        vetting_triage.apply_gate({"signals": []}, "orbital", config)


@pytest.mark.parametrize("execution", ["SKIP_AFTER_FIRST_FAILURE", "RECORD_EVERY_GATE"])
def test_instrument_gate_cannot_emit_a_dismissal(monkeypatch, execution):
    monkeypatch.setitem(FAILURES, ("passed", "instrument_gate"), "DISMISS")
    with pytest.raises(ValueError, match="gate decision.*instrument"):
        run_gates(monkeypatch, execution)


def test_gate_record_rejects_an_instrument_dismissal(monkeypatch):
    row, _ = run_gates(monkeypatch, "RECORD_EVERY_GATE")
    held = row["signals"][1]
    triage = held["triage"]
    triage.update(status="DISMISSED", dismissed_stage="instrument")
    del triage["held_stage"]
    triage["history"][2]["decision"] = "DISMISS"
    with pytest.raises(ValueError, match="gate decision.*instrument"):
        check_gate_record(held)


@pytest.mark.parametrize("execution", ["SKIP_AFTER_FIRST_FAILURE", "RECORD_EVERY_GATE"])
def test_null_counts_consume_emitted_triage_histories(monkeypatch, execution):
    from pipeline.deterministic.signal_contracts import (
        FLUX_GATES,
        null_proposal_outcome,
    )

    row, _ = run_gates(monkeypatch, execution)
    outcomes = []
    for signal in row["signals"]:
        path = [item for item in signal["triage"]["history"] if item["stage"] in FLUX_GATES]
        outcome = null_proposal_outcome({"exclusion": {"status": "ELIGIBLE"}, "gate_path": path})
        assert outcome["vetted"] is True
        outcomes.append(outcome["false_candidate"])
    assert outcomes == [False, False, True]


@pytest.mark.parametrize("execution", ["SKIP_AFTER_FIRST_FAILURE", "RECORD_EVERY_GATE"])
def test_null_rejects_a_corrupted_instrument_outcome(monkeypatch, execution):
    from pipeline.deterministic.signal_contracts import (
        FLUX_GATES,
        null_proposal_outcome,
    )

    row, _ = run_gates(monkeypatch, execution)
    held = row["signals"][1]
    path = [dict(item) for item in held["triage"]["history"] if item["stage"] in FLUX_GATES]
    path[-1]["decision"] = "DISMISS"
    with pytest.raises(ValueError, match="instrument.*decision is invalid"):
        null_proposal_outcome({"exclusion": {"status": "ELIGIBLE"}, "gate_path": path})


@pytest.mark.parametrize("automatic,reviewed", [
    ("UNRESOLVED", "BOI"),
    ("FALSE_POSITIVE", "BOI"),
    ("BOI", "FALSE_ALARM"),
])
def test_reviews_cannot_change_automatic_labels_or_counts(monkeypatch, automatic, reviewed):
    import vetting_disposition as disposition

    review = {"version": 1, "reviewer": "fixture", "rationale": "fixture", "evidence": ["fixture"],
              "signal_id": "unknown-attributed-alias", "event_ids": ["a"], "disposition": reviewed,
              "assessments": {name: {"availability": "AVAILABLE", "interpretation": "fixture", "evidence": ["fixture"]}
                              for name in disposition.ASSESSMENTS}}
    history = [review, dict(review, version=2, rationale="second audit annotation")]
    signal = {"signal_id": "automatic", "event_ids": ["a"], "events": [{"event_ids": ["a"]}],
              "signal_assessments": {"manual": {"automatic": False}}}
    row = {"events": [{"event_id": "a"}], "signals": [signal], "reviews": history}
    assessment = {"automatic_branch": {"automatic": True}}
    monkeypatch.setattr(disposition, "automatic_signal_assessments", lambda *args: assessment)
    monkeypatch.setattr(disposition, "automatic_signal", lambda row, supplied: {
        "signal_id": supplied["signal_id"], "disposition": automatic, "reason_codes": ["AUTOMATIC_BRANCH"],
    })
    result = disposition.adjudicate(row)
    assert result["status"] == result["signals"][0]["disposition"] == automatic
    assert result["boi_signal_count"] == result["boi_hypothesis_count"] == int(automatic == "BOI")
    assert result["signals"][0]["reason_codes"] == ["AUTOMATIC_BRANCH"]
    assert signal["signal_assessments"] == assessment
    assert result["attributed_signal_reviews"] == result["signals"][0]["reviews"] == history
    assert row["reviews"] == history


def test_a_review_without_automatic_signal_assessments_cannot_assign_a_candidate():
    import vetting_disposition as disposition

    review = {"reviewer": "fixture", "rationale": "fixture", "evidence": ["fixture"],
              "event_ids": ["a"], "signal_id": "unmeasured", "disposition": "BOI",
              "assessments": {name: {"availability": "AVAILABLE", "interpretation": "fixture", "evidence": ["fixture"]}
                              for name in disposition.ASSESSMENTS}}
    result = disposition.adjudicate({"events": [{"event_id": "a"}], "reviews": [review]})
    assert result["status"] == "UNRESOLVED"
    assert result["boi_signal_count"] == 0
    assert result["attributed_signal_reviews"] == [review]
