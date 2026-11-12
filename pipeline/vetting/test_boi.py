"""BOI eligibility, independent views and conservative catalogue identities."""

import copy

import pytest

from pipeline.vetting.boi import catalogue, matching_signal
from pipeline.vetting.vetting_disposition import GATE_ORDER, triage_disposition


def passed_signal():
    return {
        "signal_id": "single", "period_days": None,
        "triage": {"status": "ACTIVE", "history": [
            {"stage": stage, "decision": "PASS"} for stage in GATE_ORDER
        ]},
    }


def proposal(view, name, epoch=100.0, period=10.0, label="BOI"):
    return {
        "source_id": "123", "signal_id": name, "discovery_view": view,
        "label": label, "triage_state": "ACTIVE" if label == "BOI" else label,
        "period_days": period, "epoch_bkjd": epoch, "duration_days": 0.2,
        "event_units": [{"quarter": 1, "centre_bkjd": t} for t in (100.0, 110.0, 120.0)],
    }


def test_complete_gates_designate_a_single_event_boi_without_mutation():
    signal = passed_signal()
    before = copy.deepcopy(signal)
    assert triage_disposition(signal) == ("BOI", ["ALL_PIPELINE_GATES_PASS"])
    assert signal == before
    assert triage_disposition(signal, reproduced_mechanism=True)[0] == "FALSE_POSITIVE"


@pytest.mark.parametrize("count", range(5))
def test_incomplete_active_history_cannot_be_a_boi(count):
    signal = passed_signal()
    signal["triage"]["history"] = signal["triage"]["history"][:count]
    assert triage_disposition(signal) == ("UNRESOLVED", ["TRIAGE_INCOMPLETE"])


@pytest.mark.parametrize("stage", GATE_ORDER)
@pytest.mark.parametrize("decision,label", [("DISMISS", "DISMISSED"), ("CONTINUE_INCONCLUSIVE", "HELD")])
def test_first_failed_gate_keeps_its_label_stage_and_reason(stage, decision, label):
    signal = passed_signal()
    history = signal["triage"]["history"][:GATE_ORDER.index(stage) + 1]
    history[-1].update(decision=decision, reason="MEASURED_REASON")
    signal["triage"] = {
        "status": label, "history": history, "reason": "MEASURED_REASON",
        "dismissed_stage" if label == "DISMISSED" else "held_stage": stage,
    }
    if stage == "instrument" and decision == "DISMISS":
        with pytest.raises(ValueError, match="Unknown triage gate decision"):
            triage_disposition(signal)
    else:
        assert triage_disposition(signal) == (label, ["MEASURED_REASON"])


@pytest.mark.parametrize("view", ["psf_flat", "psf_qdiag", "sap_qdiag"])
def test_one_view_is_sufficient_and_other_failures_do_not_veto(view):
    records = [proposal(view, "positive"), proposal("another", "failed", label="HELD")]
    report = catalogue(records)
    assert report["bois"] == report["boi_proposals"] == 1
    assert report["objects"][0]["views"] == [view]
    assert report["signal_labels"] == {"BOI": 1, "HELD": 1}


def test_agreeing_views_link_without_adding_an_object_or_changing_proposals():
    records = [proposal("sap_qdiag", "sap"), proposal("psf_flat", "flat", epoch=110.01)]
    original = copy.deepcopy(records)
    report = catalogue(records)
    assert report["bois"] == 1 and report["boi_proposals"] == 2
    assert report["objects"][0]["views"] == ["psf_flat", "sap_qdiag"]
    assert records == original


@pytest.mark.parametrize("change", [
    {"period_days": None}, {"period_days": 20.0}, {"epoch_bkjd": 102.0},
    {"period_days": 10.2}, {"source_id": "456"}, {"event_units": []},
])
def test_unknown_period_harmonic_phase_drift_and_foreign_source_stay_separate(change):
    left, right = proposal("sap_qdiag", "sap"), proposal("psf_flat", "flat")
    right.update(change)
    assert not matching_signal(left, right)
    assert catalogue([left, right])["bois"] == 2


def test_grouping_requires_every_pair_to_match_and_rejects_duplicate_identities():
    records = [proposal("psf_flat", "a", epoch=100),
               proposal("psf_flat", "b", epoch=100.08),
               proposal("psf_flat", "c", epoch=100.16)]
    assert catalogue(records)["bois"] == 2
    with pytest.raises(ValueError, match="Repeated signal"):
        catalogue(records + records[:1])
