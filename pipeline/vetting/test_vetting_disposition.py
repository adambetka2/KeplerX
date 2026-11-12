"""Triage and final shape comparison share penalized timing evidence."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vetting_disposition import model_comparisons, signal_dispositions, signal_shape
from vetting_triage import repeated_shape_gate


def fit(score, status="FIT_COMPLETE"):
    alternatives = {
        name: {"status": "MEASURED", "bic": 100 + score + index}
        for index, name in enumerate((
            "linear", "quadratic", "step", "dropout", "single_cadence",
            "shared_single_cadence", "signed_box", "sinusoid",
        ))
    }
    return {"status": status, "bic": 100, "alternatives": alternatives}


def source(original, refined, penalty):
    signal = {
        "signal_id": "signal", "event_ids": ["event-a", "event-b"],
        "events": [{"unit_id": "a"}, {"unit_id": "b"}],
        "timing_hypotheses": [{
            "hypothesis_id": "original", "timing_basis": "ORIGINAL_PROPOSAL",
            "unit_ids": ["a", "b"],
        }],
    }
    result = {
        "views": {"psf_flat": {"joint_shape": fit(original)}},
        "timing_refinement": {
            "status": "SELECTED", "timing_bic_penalty": penalty,
            "timing": {"period_days": 1.01, "epoch_bkjd": 100.02},
            "views": {"psf_flat": {"joint_shape": fit(refined)}},
        },
    }
    return {"events": [], "orbital": {"hypotheses": {"original": result}}}, signal


def compare(row, signal):
    final = signal_shape(signal, model_comparisons(row, signal))
    gate = repeated_shape_gate(row, signal, {
        "shape_transit_bic_margin": 10, "shape_false_alarm_bic_margin": 10,
    })
    assert final["score"] == gate["evidence"]["score"]
    return final, gate


@pytest.mark.parametrize("original,refined,penalty,expected", [
    (-20, 30, 5, 25), (-20, 15, 20, -5), (-20, 20, 20, 0),
    (25, 30, 5, 25), (25, 30, 10, 25), (-20, 30, 90, -20),
])
def test_final_and_triage_choose_the_same_penalized_measurement(
    original, refined, penalty, expected
):
    row, signal = source(original, refined, penalty)
    unchanged = copy.deepcopy(row)
    final, gate = compare(row, signal)
    assert final["score"] == expected
    assert row == unchanged
    selected = expected > original
    assert (
        gate["evidence"]["measurement"] == "BOUNDED_REFINED_TIMING_SHAPE"
    ) == selected
    assert ("timing_refinement" in final["records"][0]) == selected
    if selected:
        record = final["records"][0]
        assert record["original_shape"]["delta_bic"] == original
        assert record["timing_refinement"]["timing_bic_penalty"] == penalty
        assert record["transit_bic"] == 100 + penalty
        assert record["timing_basis"] == "ORIGINAL_PROPOSAL"
        assert final["selection"]["measurement"] == "BOUNDED_REFINED_TIMING_SHAPE"


def candidate_source(name, centres):
    return {
        "signal_id": name,
        "event_ids": [f"{name}-{index}" for index in range(len(centres))],
        "events": [
            {"start_bkjd": centre - 0.1, "end_bkjd": centre + 0.1}
            for centre in centres
        ],
        "association_status": "DETECTOR_SUPPLIED_HYPOTHESIS",
    }


@pytest.mark.parametrize(
    "left,right,expected_count,expected_status",
    [
        (
            [0, 7, 14, 21, 28],
            list(range(35)),
            None,
            "OVERLAPPING_CANDIDATE_HYPOTHESES",
        ),
        (
            [0.5, 7.5, 14.5, 21.5, 28.5],
            list(range(35)),
            2,
            "EXPLICIT_SIGNAL_IDENTITIES",
        ),
        ([0], [0, 7, 14], 2, "EXPLICIT_SIGNAL_IDENTITIES"),
    ],
)
def test_candidate_count_retains_harmonic_identity_uncertainty(
    monkeypatch, left, right, expected_count, expected_status
):
    monkeypatch.setattr(
        "vetting_disposition.automatic_signal",
        lambda row, signal: {
            "signal_id": signal["signal_id"],
            "event_ids": signal["event_ids"],
            "disposition": "BOI",
        },
    )
    row = {"signals": [candidate_source("a", left), candidate_source("b", right)]}
    base = {"status": "UNRESOLVED", "boi_signal_count": 0}
    result = signal_dispositions(row, base)
    assert result["boi_signal_count"] == expected_count
    assert result["boi_hypothesis_count"] == 2
    assert result["count_status"] == expected_status


@pytest.mark.parametrize("missing", ["selection", "penalty", "view", "fit"])
def test_incomplete_or_unselected_refinement_cannot_replace_original(missing):
    row, signal = source(-20, 1000, 5)
    refinement = row["orbital"]["hypotheses"]["original"]["timing_refinement"]
    if missing == "selection":
        refinement["status"] = "REJECTED"
    elif missing == "penalty":
        refinement.pop("timing_bic_penalty")
    elif missing == "view":
        refinement["views"] = {}
    else:
        refinement["views"]["psf_flat"]["joint_shape"]["status"] = "OPTIMIZER_LIMIT"
    final, _ = compare(row, signal)
    assert final["score"] == -20
    assert "timing_refinement" not in final["records"][0]


@pytest.mark.parametrize("penalty", [-1, float("inf"), float("nan")])
def test_invalid_penalty_fails_clearly(penalty):
    row, signal = source(20, 100, penalty)
    with pytest.raises(ValueError, match="finite nonnegative BIC penalty"):
        model_comparisons(row, signal)
    with pytest.raises(ValueError, match="finite nonnegative BIC penalty"):
        repeated_shape_gate(row, signal, {})


def test_correlated_views_are_not_added_to_refined_score():
    row, signal = source(-20, 100, 20)
    result = row["orbital"]["hypotheses"]["original"]
    result["views"]["psf_qdiag"] = {"joint_shape": fit(500)}
    final, _ = compare(row, signal)
    assert final["score"] == 80
    assert final["selection"]["view"] == "psf_flat"


def test_refinement_retains_the_original_view_and_measured_view_identities():
    row, signal = source(-20, 100, 20)
    result = row["orbital"]["hypotheses"]["original"]
    result["views"]["psf_flat"]["joint_shape"]["status"] = "OPTIMIZER_LIMIT"
    result["views"]["psf_qdiag"] = {"joint_shape": fit(30)}
    final, _ = compare(row, signal)
    assert final["score"] == 80
    assert final["selection"]["view"] == "psf_qdiag"
    assert final["selection"]["measurement_view"] == "psf_flat"


@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
@pytest.mark.parametrize("score,decision", [(30, "PASS"), (-30, "DISMISS")])
def test_declared_view_is_vetted_independently(view, score, decision):
    row, signal = source(-score, 0, 0)
    result = row["orbital"]["hypotheses"]["original"]
    result.pop("timing_refinement")
    result["hypothesis"] = {"discovery_view": view}
    result["views"][view] = {"joint_shape": fit(score)}
    signal["discovery_view"] = view
    final, gate = compare(row, signal)
    assert final["score"] == score
    assert final["selection"]["view"] == view
    assert gate["decision"] == decision
    assert result["views"]["psf_flat"]["joint_shape"] == fit(-score)


@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
def test_missing_declared_view_cannot_use_another_view(view):
    row, signal = source(1000, 0, 0)
    result = row["orbital"]["hypotheses"]["original"]
    result.pop("timing_refinement")
    result["hypothesis"] = {"discovery_view": view}
    signal["discovery_view"] = view
    final = signal_shape(signal, model_comparisons(row, signal))
    gate = repeated_shape_gate(row, signal, {})
    assert final["status"] == "UNRESOLVED"
    assert gate["decision"] == "CONTINUE_INCONCLUSIVE"


def test_multiple_original_hypotheses_remain_unresolved():
    row, signal = source(-20, 100, 20)
    other = copy.deepcopy(signal["timing_hypotheses"][0])
    other["hypothesis_id"] = "other"
    signal["timing_hypotheses"].append(other)
    row["orbital"]["hypotheses"]["other"] = copy.deepcopy(
        row["orbital"]["hypotheses"]["original"]
    )
    final = signal_shape(signal, model_comparisons(row, signal))
    assert final["status"] == "UNRESOLVED"
    assert final["reason"] == "MULTIPLE_NON_EQUIVALENT_SIGNAL_MEASUREMENTS"


def test_single_event_keeps_its_own_shape_measurement():
    row, signal = source(-20, 100, 20)
    signal["events"] = [{"unit_id": "a"}]
    signal["event_ids"] = ["event-a"]
    row["events"] = [{"event_id": "event-a", "photometry": "psf_flat", "fit": fit(7)}]
    final = signal_shape(signal, model_comparisons(row, signal))
    assert final["score"] == 7
    assert "timing_refinement" not in final["records"][0]
