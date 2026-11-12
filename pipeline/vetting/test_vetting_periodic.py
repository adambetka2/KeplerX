"""Repeated-event window measurements preserve the original cadence selection."""

import json
from pathlib import Path

import numpy as np
import vetting_periodic
from vetting_disposition import shape_evidence
from vetting_models import event_data


def test_sorted_local_window_matches_full_quarter_selection():
    time = np.r_[np.arange(0, 2, 0.02), np.arange(4, 6, 0.02)]
    flux = 1000 + 2 * np.sin(time)
    error = np.ones(len(time))
    quality = np.ones(len(time))
    quality[::13] = 0
    x = np.column_stack([flux, error, time, quality])
    config = {"minimum_context_days": 0.5, "local_context_durations": 4}
    for centre in (0.0, 0.5, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0):
        old = event_data(x, centre, 0.1, config)
        new = event_data(x, centre, 0.1, config, sorted_time=True)
        if old is None:
            assert new is None
        else:
            assert new is not None
            for before, after in zip(old, new):
                np.testing.assert_array_equal(before, after)


def test_short_period_phase_mask_keeps_exposure_clear_out_of_event_flux(monkeypatch):
    time = np.arange(0, 12, 0.0204336)
    curve = {
        "time": time,
        "flux": np.full(len(time), 1000.0),
        "error": np.ones(len(time)),
        "quarter": np.ones(len(time), dtype=int),
        "quality": np.zeros(len(time), dtype=int),
        "flatten": np.zeros(len(time), dtype=int),
    }
    monkeypatch.setattr(vetting_periodic, "photometry", lambda raw, view: curve)
    hypothesis = {"period_days": 0.269, "epoch_bkjd": 0.05}
    config = {"quality_bitmask": -1, "exposure_days": 0.0204336}

    chunks = vetting_periodic.phase_curve_chunks(
        {}, hypothesis, 0.08, "psf_flat", config
    )

    assert len(chunks) == 1
    assert len(chunks[0]["time"]) > 20


def test_short_period_shape_samples_fit_eligible_events(monkeypatch):
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    config["short_period_joint_fit_max_units"] = 4
    config["short_period_joint_fit_max_missing_fraction"] = 0.3
    time = np.arange(0, 5, config["exposure_days"])
    centres = 0.2 + 0.4 * np.arange(14)
    phase = (time - 0.2 + 0.2) % 0.4 - 0.2
    flux = 1000 * (1 - 0.001 * (abs(phase) < 0.02))
    curve = {
        "time": time,
        "flux": flux,
        "error": np.full(len(time), 0.1),
        "quarter": np.ones(len(time), dtype=int),
    }
    monkeypatch.setattr(vetting_periodic, "photometry", lambda raw, view: curve)
    monkeypatch.setattr(
        vetting_periodic,
        "valid_points",
        lambda curve, quality_bitmask, mask_flatten=True: np.ones(len(time), bool),
    )
    members = [
        {
            "unit_id": f"event-{index}",
            "centre_bkjd": centre,
            "quarter": 1,
            "duration_days": 0.04,
        }
        for index, centre in enumerate(centres)
    ]
    hypothesis = {
        "period_days": 0.4,
        "detector_origin": "shallow_bls_v1",
        "shallow_record": {"search_band": "short"},
    }

    fit = vetting_periodic.folded_fit(
        {}, hypothesis, members, 0.04, "psf_flat", config
    )

    assert fit["status"] == "FIT_COMPLETE"
    assert len(fit["selected_unit_ids"]) == 4
    assert fit["selected_unit_ids"] == fit["fitted_unit_ids"]
    assert fit["selection"]["candidate_units"] > 4
    assert fit["selection"]["missing_candidate_fraction"] > 0.05

    guarded = vetting_periodic.folded_fit(
        {}, hypothesis, members, 0.04, "psf_flat",
        config | {"short_period_joint_fit_max_missing_fraction": 0.05},
    )
    assert guarded["status"] == "INCOMPLETE_EVENT_MEMBERSHIP"
    assert shape_evidence(guarded) == {
        "status": "UNRESOLVED",
        "reason": "SHAPE_EVENT_MEMBERSHIP_INCOMPLETE",
    }

    rust = vetting_periodic.folded_fit(
        {}, {"period_days": 0.4, "detector_origin": "rust_v6_deterministic"},
        members, 0.04, "psf_flat", config,
    )
    assert rust["selection"]["limit"] == config["joint_fit_max_units_per_signal"]


def test_bounded_refinement_corrects_drift_within_supplied_events(monkeypatch):
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    time = np.arange(0, 50, config["exposure_days"])
    truth = 0.2 + 0.9 * np.arange(52)
    distance = np.min(abs(time[:, None] - truth[None, :]), axis=1)
    curve = {
        "time": time,
        "flux": 1000 * (1 - 0.01 * (distance < 0.04)),
        "error": np.ones(len(time)),
        "quarter": np.ones(len(time), dtype=int),
    }
    monkeypatch.setattr(vetting_periodic, "photometry", lambda raw, view: curve)
    monkeypatch.setattr(
        vetting_periodic,
        "valid_points",
        lambda curve, quality_bitmask, mask_flatten=True: np.ones(len(time), bool),
    )
    members = [
        {
            "unit_id": f"event-{cycle}",
            "centre_bkjd": 0.185 + cycle * 0.9006,
            "quarter": 1,
            "duration_days": 0.08,
        }
        for cycle in range(52)
    ]
    config["timing_refinement_max_units"] = 32

    shifted, result = vetting_periodic.refine_event_timing(
        {}, members, 0.08, config
    )

    fitted_period = shifted[1]["centre_bkjd"] - shifted[0]["centre_bkjd"]
    assert abs(fitted_period - 0.9) < 0.0002
    assert result["refined_box_score"] > result["original_box_score"]
    assert all(
        abs(after["centre_bkjd"] - before["centre_bkjd"]) <= 0.04
        for before, after in zip(members, shifted)
    )


def test_event_train_support_exposes_sparse_real_dips():
    coherent = [
        {"status": "MEASURED", "depth_fraction": 3.0, "depth_error": 1.0}
        for _ in range(40)
    ]
    sparse = coherent[:25] + [
        {"status": "MEASURED", "depth_fraction": -0.1, "depth_error": 1.0}
        for _ in range(75)
    ]

    assert vetting_periodic.event_train_support(coherent)["positive_deficit_z"] > -1
    assert vetting_periodic.event_train_support(sparse)["positive_deficit_z"] < -10


def test_primary_train_retains_each_extraction_measurement(monkeypatch):
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    time = np.arange(0, 30, config["exposure_days"])
    phase = (time - 2 + 5) % 10 - 5
    pulse = abs(phase) < 0.15
    curves = {
        view: {
            "time": time,
            "flux": 1000 * (1 + sign * 0.01 * pulse),
            "error": np.full(len(time), 0.1),
            "quarter": np.ones(len(time), dtype=int),
            "quality": np.zeros(len(time), dtype=int),
            "flatten": np.zeros(len(time), dtype=int),
        }
        for view, sign in (("psf_flat", -1), ("sap_qdiag", 1))
    }
    monkeypatch.setattr(vetting_periodic, "photometry", lambda raw, view: curves[view])
    monkeypatch.setattr(vetting_periodic, "folded_fit", lambda *args: {})
    monkeypatch.setattr(vetting_periodic, "phase_modulation", lambda *args: {})
    measured = vetting_periodic.hypothesis_measurements(
        {}, {"period_days": 10, "epoch_bkjd": 2}, [], 0.3, list(curves), config
    )

    for view in curves:
        assert measured[view]["primary_train"]["status"] == "MEASURED"
        assert measured[view]["primary_train"]["events"] == 3
        assert measured[view]["primary"]["measured_windows"] == 3
    assert measured["psf_flat"]["primary_train"]["observed_positive_fraction"] == 1
    assert measured["sap_qdiag"]["primary_train"]["observed_positive_fraction"] == 0
    assert measured["psf_flat"]["primary_depth"]["depth_fraction"] > 0
    assert measured["sap_qdiag"]["primary_depth"]["depth_fraction"] < 0


def test_primary_train_retains_underpowered_event_coverage(monkeypatch):
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    time = np.arange(0, 15, config["exposure_days"])
    curve = {
        "time": time,
        "flux": np.full(len(time), 1000.0),
        "error": np.ones(len(time)),
        "quarter": np.ones(len(time), dtype=int),
        "quality": np.zeros(len(time), dtype=int),
        "flatten": np.zeros(len(time), dtype=int),
    }
    monkeypatch.setattr(vetting_periodic, "photometry", lambda raw, view: curve)
    monkeypatch.setattr(vetting_periodic, "folded_fit", lambda *args: {})
    monkeypatch.setattr(vetting_periodic, "phase_modulation", lambda *args: {})
    measured = vetting_periodic.hypothesis_measurements(
        {}, {"period_days": 10, "epoch_bkjd": 2}, [], 0.3, ["psf_flat"], config
    )["psf_flat"]

    assert measured["primary"]["measured_windows"] == 2
    assert measured["primary_train"] == {"status": "UNCONSTRAINED", "events": 2}
