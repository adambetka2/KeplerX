"""A shorter event train can hold an alias without diagnosing a binary."""

import numpy as np
import vetting
import vetting_alias
import vetting_disposition
import vetting_triage


def fixture_signal():
    centres = [0.0, 3.0, 6.0, 9.0, 12.0]
    return {
        "signal_id": "candidate",
        "period_days": 3.0,
        "events": [{"centre_bkjd": centre, "duration_days": 0.1} for centre in centres],
        "timing_hypotheses": [
            {
                "hypothesis_id": "proposal",
                "period_days": 3.0,
                "epoch_bkjd": 0.0,
                "duration_days": 0.1,
            }
        ],
    }


def fixture_config():
    return {
        "alias": {
            "workers": 2,
            "max_eb_rows_per_signal": 3,
            "minimum_supplied_event_alignment": 0.8,
            "minimum_independent_events": 5,
            "maximum_single_event_information_fraction": 0.5,
            "minimum_independent_depth_snr": 7,
            "maximum_source_eb_harmonic_order": 8,
            "maximum_source_eb_harmonic_period_error_fraction": 0.01,
            "minimum_source_eb_harmonic_alignment": 0.5,
            "minimum_source_eb_odd_even_bic_gain": 10,
        }
    }


def fixture_row():
    return {
        "rust_rows": [
            {
                "photometry": "psf_flat",
                "classification": "EB_LIKE",
                "period_days": "1.0",
                "t0_bkjd": "0.0",
                "duration_hours": "2.4",
                "search_mes": "20",
                "signal_rank": "1",
            }
        ]
    }


def train(depths):
    return {
        "status": "MEASURED",
        "events": [
            {
                "centre_bkjd": float(index),
                "duration_days": 0.1,
                "depth_fraction": depth,
                "depth_error": 0.001,
            }
            for index, depth in enumerate(depths)
        ],
    }


def test_independent_shorter_period_dips_hold_alias(monkeypatch):
    signal = fixture_signal()
    primary = train([0.01] * 13)
    secondary = train([0.0] * 13)
    monkeypatch.setattr(
        vetting_alias, "measure_train", lambda *args: [primary, secondary]
    )

    result = vetting_alias.measure_signal(
        {}, fixture_row(), signal, fixture_config(), {}
    )

    assert result["status"] == "COHERENT_SHORTER_PERIOD"
    depth = result["records"][0]["independent_phases"][0]
    assert depth["measured_events"] == 8
    assert depth["snr"] > 7


def test_candidate_only_dips_do_not_create_shorter_period_conflict(monkeypatch):
    signal = fixture_signal()
    primary = train([0.01 if index % 3 == 0 else 0.0 for index in range(13)])
    secondary = train([0.0] * 13)
    monkeypatch.setattr(
        vetting_alias, "measure_train", lambda *args: [primary, secondary]
    )

    result = vetting_alias.measure_signal(
        {}, fixture_row(), signal, fixture_config(), {}
    )

    assert result["status"] == "NO_SUPPORTED_ALIAS_IN_SHORTLIST"


def test_unmeasured_competing_train_stays_incomplete(monkeypatch):
    signal = fixture_signal()
    unavailable = {"status": "COMPUTATION_LIMIT", "events": []}
    monkeypatch.setattr(vetting_alias, "measure_train", lambda *args: [unavailable] * 2)

    result = vetting_alias.measure_signal(
        {}, fixture_row(), signal, fixture_config(), {}
    )

    assert result["status"] == "INCOMPLETE_SHORTER_PERIOD_COMPARISON"


def test_underpowered_measured_train_stays_incomplete(monkeypatch):
    signal = fixture_signal()
    short = train([0.0] * 2)
    monkeypatch.setattr(vetting_alias, "measure_train", lambda *args: [short] * 2)

    result = vetting_alias.measure_signal(
        {}, fixture_row(), signal, fixture_config(), {}
    )

    assert result["status"] == "INCOMPLETE_SHORTER_PERIOD_COMPARISON"


def test_no_aligned_shorter_period_row_needs_no_raw_flux():
    signal = fixture_signal()
    row = fixture_row() | {"signals": [signal]}
    row["rust_rows"][0]["t0_bkjd"] = "0.25"

    assert not vetting_alias.needs_flux(row, fixture_config())
    measured = vetting_alias.measure(row, None, fixture_config())
    assert measured["alias"]["signals"]["candidate"]["status"] == (
        "NO_SUPPORTED_ALIAS_IN_SHORTLIST"
    )


def test_alias_comparison_uses_selected_refined_ephemeris():
    signal = fixture_signal()
    refined = {"period_days": 2.999, "epoch_bkjd": 0.02, "duration_days": 0.1}
    row = {
        "orbital": {
            "hypotheses": {
                "proposal": {
                    "timing_refinement": {
                        "status": "SELECTED",
                        "hypothesis": refined,
                    }
                }
            }
        }
    }

    assert vetting_alias.candidate_ephemeris(row, signal) == refined


def test_alias_holds_orbital_gate_without_false_positive_denial(monkeypatch):
    signal = fixture_signal()
    row = {"alias": {"signals": {"candidate": {"status": "COHERENT_SHORTER_PERIOD"}}}}
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    result = vetting_triage.orbital_gate(row, signal, {"assessment_rules": {}})

    assert result["decision"] == "CONTINUE_INCONCLUSIVE"
    assert result["reason"] == "LOWER_PERIOD_ALIAS_REQUIRES_RESOLUTION"


def test_source_two_eclipse_model_dismisses_an_aligned_alias(monkeypatch):
    signal = fixture_signal()
    signal["period_days"] = 1.0
    signal["events"] = [
        {"centre_bkjd": float(index), "duration_days": 0.1}
        for index in range(20)
    ]
    row = {
        "rust_rows": [
            {
                "photometry": "psf_flat",
                "classification": "TRANSIT_LIKE",
                "period_days": "3.0",
                "t0_bkjd": "0.0",
                "duration_hours": "2.4",
                "signal_rank": "1",
            }
        ],
        "signals": [signal],
    }
    events = [
        {
            "status": "MEASURED",
            "cycle": cycle,
            "depth_fraction": 0.01 if cycle % 2 == 0 else 0.007,
            "depth_error": 0.0001,
            "in_event_points": 5,
            "before_points": 10,
            "after_points": 10,
        }
        for cycle in range(20)
    ]
    monkeypatch.setattr(
        vetting_alias,
        "expected_events",
        lambda *args: {"status": "MEASURED", "events": events},
    )
    monkeypatch.setattr(
        vetting_alias,
        "independent_candidate_train",
        lambda *args: {
            "status": "MEASURED",
            "depth_fraction": 0.0001,
            "depth_error": 0.0001,
            "snr": 1,
            "measured_events": 20,
            "largest_information_fraction": 0.1,
        },
    )
    monkeypatch.setattr(
        vetting_alias,
        "photometry",
        lambda *args: {"time": [0.0]},
    )
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    assert vetting_alias.needs_flux(row, fixture_config())
    measured = vetting_alias.measure(row, {}, fixture_config())
    alternative = measured["alias"]["signals"]["candidate"]
    assert alternative["status"] == "REPRODUCED_SOURCE_TWO_ECLIPSE"
    assert alternative["source_two_eclipse"]["model"]["odd_even"]["two_depth_bic_gain"] > 10
    outcome = vetting_triage.orbital_gate(measured, signal, {"assessment_rules": {}})
    assert outcome["decision"] == "PASS"
    assert outcome["reason"] == "SOURCE_TWO_ECLIPSE_CONTEXT_PENDING"

    monkeypatch.setattr(
        vetting_triage,
        "contamination_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )
    monkeypatch.setattr(
        vetting_triage,
        "instrument_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )
    result = vetting_triage.contamination_gate(measured, signal, {"assessment_rules": {}})
    assert result["decision"] == "DISMISS"
    assert result["reason"] == "REPRODUCED_SOURCE_TWO_ECLIPSE_MECHANISM"


def test_source_two_eclipse_attribution_respects_context(monkeypatch):
    signal = fixture_signal()
    row = {
        "signals": [signal],
        "alias": {"signals": {"candidate": {"status": "REPRODUCED_SOURCE_TWO_ECLIPSE"}}},
    }
    policy = {"assessment_rules": {}}
    monkeypatch.setattr(
        vetting_triage,
        "instrument_assessment",
        lambda *args: {"status": "UNASSESSED"},
    )
    monkeypatch.setattr(
        vetting_triage,
        "contamination_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )
    result = vetting_triage.contamination_gate(row, signal, policy)
    assert result["decision"] == "CONTINUE_INCONCLUSIVE"
    assert result["reason"] == "SOURCE_TWO_ECLIPSE_ATTRIBUTION_INCOMPLETE"

    monkeypatch.setattr(
        vetting_triage,
        "contamination_assessment",
        lambda *args: {"status": "MATERIAL_CONFLICT"},
    )
    result = vetting_triage.contamination_gate(row, signal, policy)
    assert result["decision"] == "DISMISS"
    assert result["reason"] == "COMPETING_SOURCE_EVIDENCE"


def test_independent_candidate_dips_prevent_source_eb_dismissal(monkeypatch):
    signal = fixture_signal()
    signal["period_days"] = 1.0
    signal["events"] = [
        {"centre_bkjd": float(index), "duration_days": 0.1}
        for index in range(20)
    ]
    row = fixture_row() | {"signals": [signal]}
    row["rust_rows"][0]["period_days"] = "3.0"
    monkeypatch.setattr(vetting_alias, "photometry", lambda *args: {"time": [0.0]})
    monkeypatch.setattr(
        vetting_alias,
        "source_two_eclipse_model",
        lambda *args: {"status": "REPRODUCED_UNEQUAL_TWO_ECLIPSE"},
    )
    monkeypatch.setattr(
        vetting_alias,
        "independent_candidate_train",
        lambda *args: {
            "status": "MEASURED",
            "depth_fraction": 0.01,
            "depth_error": 0.001,
            "snr": 10,
            "measured_events": 20,
            "largest_information_fraction": 0.1,
        },
    )
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    measured = vetting_alias.measure(row, {}, fixture_config())
    alternative = measured["alias"]["signals"]["candidate"]
    assert alternative["status"] == "INDEPENDENT_CANDIDATE_DIP_SUPPORT"
    outcome = vetting_triage.orbital_gate(measured, signal, {"assessment_rules": {}})
    assert outcome["decision"] == "CONTINUE_INCONCLUSIVE"
    assert outcome["reason"] == "CANDIDATE_DIP_OUTSIDE_SOURCE_ECLIPSES"


def test_source_two_eclipse_requires_two_positive_event_trains(monkeypatch):
    signal = fixture_signal()
    signal["period_days"] = 1.0
    signal["events"] = [
        {"centre_bkjd": float(index), "duration_days": 0.1}
        for index in range(20)
    ]
    row = {
        "rust_rows": [
            {
                "photometry": "psf_flat",
                "classification": "EB_LIKE",
                "period_days": "3.0",
                "t0_bkjd": "0.0",
                "duration_hours": "2.4",
                "signal_rank": "1",
            }
        ],
        "signals": [signal],
    }
    events = [
        {
            "status": "MEASURED",
            "cycle": cycle,
            "depth_fraction": 0.01 if cycle % 2 == 0 else 0.0,
            "depth_error": 0.0001,
            "in_event_points": 5,
            "before_points": 10,
            "after_points": 10,
        }
        for cycle in range(20)
    ]
    monkeypatch.setattr(
        vetting_alias,
        "expected_events",
        lambda *args: {"status": "MEASURED", "events": events},
    )
    monkeypatch.setattr(vetting_alias, "photometry", lambda *args: {"time": [0.0]})

    measured = vetting_alias.measure(row, {}, fixture_config())
    alternative = measured["alias"]["signals"]["candidate"]
    assert alternative["status"] != "REPRODUCED_SOURCE_TWO_ECLIPSE"


def test_source_two_eclipse_does_not_link_a_half_period_alias():
    signal = fixture_signal()
    signal["events"] = [
        {"centre_bkjd": float(index), "duration_days": 0.1}
        for index in range(20)
    ]
    rank_one = {
        "period_days": "1.5",
        "t0_bkjd": "0.0",
        "duration_hours": "2.4",
    }
    assert vetting_alias.source_eb_alignment(
        signal, rank_one, fixture_config()["alias"]
    ) is None


def test_incomplete_rank_one_comparison_keeps_existing_orbital_result(monkeypatch):
    signal = fixture_signal()
    row = fixture_row() | {"signals": [signal]}
    row["rust_rows"][0]["period_days"] = "3.0"
    monkeypatch.setattr(vetting_alias, "photometry", lambda *args: {"time": [0.0]})
    monkeypatch.setattr(
        vetting_alias,
        "source_two_eclipse_model",
        lambda *args: {"status": "INCOMPLETE_TWO_ECLIPSE_COMPARISON"},
    )
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    measured = vetting_alias.measure(row, {}, fixture_config())
    alternative = measured["alias"]["signals"]["candidate"]
    assert alternative["status"] == "NO_SUPPORTED_ALIAS_IN_SHORTLIST"
    assert alternative["source_two_eclipse"]["model"]["status"] == (
        "INCOMPLETE_TWO_ECLIPSE_COMPARISON"
    )
    outcome = vetting_triage.orbital_gate(measured, signal, {"assessment_rules": {}})
    assert outcome["decision"] == "PASS"


def test_incomplete_longer_source_period_holds_alias(monkeypatch):
    signal = fixture_signal()
    row = {
        "alias": {
            "signals": {
                "candidate": {"status": "INCOMPLETE_SOURCE_TWO_ECLIPSE_COMPARISON"}
            }
        }
    }
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    outcome = vetting_triage.orbital_gate(row, signal, {"assessment_rules": {}})
    assert outcome["decision"] == "CONTINUE_INCONCLUSIVE"
    assert outcome["reason"] == "SOURCE_TWO_ECLIPSE_COMPARISON_INCOMPLETE"


def test_incomplete_alias_train_does_not_veto_period_uncertain_transit(monkeypatch):
    signal = fixture_signal()
    row = {
        "alias": {
            "signals": {"candidate": {"status": "INCOMPLETE_SHORTER_PERIOD_COMPARISON"}}
        }
    }
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    result = vetting_triage.orbital_gate(row, signal, {"assessment_rules": {}})

    assert result["decision"] == "PASS"


def test_missing_repeat_alias_measurement_holds(monkeypatch):
    monkeypatch.setattr(
        vetting_triage,
        "orbital_assessment",
        lambda *args: {"status": "ASSESSED_NO_MATERIAL_CONFLICT"},
    )

    result = vetting_triage.orbital_gate({}, fixture_signal(), {"assessment_rules": {}})

    assert result["decision"] == "CONTINUE_INCONCLUSIVE"
    assert result["reason"] == "ALIAS_MEASUREMENT_UNAVAILABLE"


def test_single_event_does_not_require_alias_measurement():
    result = vetting_triage.orbital_gate(
        {}, {"events": [{"unit_id": "only"}]}, {"assessment_rules": {}}
    )

    assert result["decision"] == "PASS"


def test_aligned_single_event_records_two_eclipse_context_without_a_veto(monkeypatch):
    signal = {
        "signal_id": "single",
        "events": [{"centre_bkjd": 10.0, "duration_days": 0.1}],
    }
    row = fixture_row() | {"signals": [signal]}
    monkeypatch.setattr(
        vetting_alias, "photometry", lambda *args: {"time": np.array([10.0])}
    )
    monkeypatch.setattr(
        vetting_alias,
        "source_two_eclipse_model",
        lambda *args: {"status": "REPRODUCED_UNEQUAL_TWO_ECLIPSE"},
    )

    assert vetting_alias.needs_flux(row, fixture_config())
    measured = vetting_alias.measure(row, object(), fixture_config())
    record = measured["alias"]["signals"]["single"]
    assert record["status"] == "SINGLE_EVENT_SOURCE_EB_DIAGNOSTIC"
    assert record["source_two_eclipse"]["model"]["status"] == (
        "REPRODUCED_UNEQUAL_TWO_ECLIPSE"
    )
    assert (
        vetting_triage.orbital_gate(measured, signal, {"assessment_rules": {}})[
            "decision"
        ]
        == "PASS"
    )


def test_unaligned_or_non_eb_single_event_skips_source_model():
    signal = {
        "signal_id": "single",
        "events": [{"centre_bkjd": 10.5, "duration_days": 0.1}],
    }
    row = fixture_row() | {"signals": [signal]}
    assert not vetting_alias.needs_flux(row, fixture_config())
    assert vetting_alias.measure(row, None, fixture_config())["alias"]["status"] == (
        "NOT_APPLICABLE"
    )

    row["rust_rows"][0]["classification"] = "TRANSIT_LIKE"
    signal["events"][0]["centre_bkjd"] = 10.0
    assert not vetting_alias.needs_flux(row, fixture_config())


def test_alias_stage_loads_raw_flux_for_aligned_single_event(monkeypatch, tmp_path):
    row = fixture_row() | {
        "signals": [
            {
                "signal_id": "single",
                "events": [{"centre_bkjd": 10.0, "duration_days": 0.1}],
            }
        ]
    }
    saved = []
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    monkeypatch.setattr(
        vetting,
        "read",
        lambda path: fixture_config() if path == vetting.CONFIG else row,
    )
    monkeypatch.setattr(vetting, "observation", lambda source_id: ({}, object()))
    monkeypatch.setattr(
        vetting, "save", lambda path, value, compact: saved.append(value)
    )
    monkeypatch.setattr(
        vetting_alias, "photometry", lambda *args: {"time": np.array([10.0])}
    )
    monkeypatch.setattr(
        vetting_alias,
        "source_two_eclipse_model",
        lambda *args: {"status": "NO_SUPPORTED_TWO_ECLIPSE"},
    )

    assert vetting.measure_alias_source("123")["source_id"] == "123"
    assert saved[0]["alias"]["signals"]["single"]["status"] == (
        "SINGLE_EVENT_SOURCE_EB_DIAGNOSTIC"
    )


def test_held_signal_keeps_its_first_failure_reason():
    history = [
        {"stage": "shape", "decision": "PASS"},
        {"stage": "orbital", "decision": "CONTINUE_INCONCLUSIVE"},
    ]
    signal = {"signal_id": "held", "triage": {"status": "HELD", "held_stage": "orbital", "history": history}}
    signal["triage"]["reason"] = "UNCONSTRAINED_ORBITAL_EVIDENCE"
    signal["triage"]["history"][-1]["reason"] = "UNCONSTRAINED_ORBITAL_EVIDENCE"
    result = vetting_disposition.triage_disposition(signal)
    assert result == ("HELD", ["UNCONSTRAINED_ORBITAL_EVIDENCE"])
