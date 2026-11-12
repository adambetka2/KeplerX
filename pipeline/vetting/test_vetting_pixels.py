"""Pixel evidence must follow every event represented by a shared pixel job."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

import vetting_pixels as pixels
from vetting_pixels import aggregate_signals, pixel_job_active


def test_spatial_null_prefers_prf_planet_and_column_artifact():
    row, column = np.indices((5, 5))
    mask = np.ones((5, 5), dtype=bool)
    prf = np.exp(-((row - 2) ** 2 + (column - 2) ** 2) / 2).ravel()
    planet = 80 * prf
    stripe = 20 * (column == 2).astype(float).ravel()
    fitted_stripe_amplitude = np.sum(prf * stripe) / np.sum(prf**2)
    errors = np.ones(25)

    assert pixels.spatial_null_bic_advantage(planet, errors, planet, mask) < -10
    assert (
        pixels.spatial_null_bic_advantage(
            stripe, errors, fitted_stripe_amplitude * prf, mask
        )
        > 10
    )


def pixel(event_id, represented, quarter, offset=4.0, snr=20.0):
    return {
        "event_id": event_id,
        "represented_event_ids": represented,
        "quarter": quarter,
        "tangent_offset_arcsec": [offset, 0.0],
        "tangent_covariance_arcsec2": [[0.01, 0.0], [0.0, 0.01]],
        "target_inside_stamp": True,
        "prf_correlation": 0.9,
        "fixed_source_controls": {
            "measurements": [
                {
                    "offset_durations": 0,
                    "status": "MEASURED",
                    "amplitude": snr,
                    "formal_error": 1.0,
                }
            ]
        },
    }


def test_active_signal_can_use_another_signals_representative():
    job = {"event_id": "dismissed", "represented_event_ids": ["dismissed", "active"]}
    assert pixel_job_active(job, {"active"})
    assert not pixel_job_active(job, {"other"})


def test_refined_pixel_jobs_are_scoped_to_the_signal():
    from vetting_flux import append_refined_pixel_jobs

    row = {
        "events": [
            {
                "event_id": f"event-{index}",
                "alert_id": f"alert-{index}",
                "quarter": 1,
                "centre_bkjd": float(index),
                "duration_days": 0.2,
                "photometry": "psf_flat",
            }
            for index in range(2)
        ],
        "signals": [
            {
                "signal_id": "target",
                "triage": {"status": "ACTIVE"},
                "timing_hypotheses": [
                    {"hypothesis_id": "original", "timing_basis": "ORIGINAL_PROPOSAL"}
                ],
                "events": [
                    {
                        "unit_id": f"unit-{index}",
                        "event_ids": [f"event-{index}"],
                        "quarter": 1,
                        "centre_bkjd": float(index),
                        "duration_days": 0.2,
                    }
                    for index in range(2)
                ],
            }
        ],
        "orbital": {
            "hypotheses": {
                "original": {
                    "timing_refinement": {
                        "status": "SELECTED",
                        "timing": {
                            "start_shift_days": 0.1,
                            "slope": -0.2,
                            "first_centre_bkjd": 0.0,
                        },
                    }
                }
            }
        },
        "pixel_jobs": [],
    }
    append_refined_pixel_jobs(row, {"diagnostic_max_units_per_signal": 2})
    append_refined_pixel_jobs(row, {"diagnostic_max_units_per_signal": 2})

    assert len(row["pixel_jobs"]) == 2
    np.testing.assert_allclose(
        [job["centre_bkjd"] for job in row["pixel_jobs"]], [0.1, 0.9]
    )
    assert all(job["timing_refined_signal_id"] == "target" for job in row["pixel_jobs"])
    row["localization"] = {
        "events": [
            {"event_id": "original", "represented_event_ids": ["event-0"]},
            {"event_id": "original-only", "represented_event_ids": ["event-1"]},
            {
                "event_id": "refined",
                "represented_event_ids": ["event-0"],
                "timing_refined_signal_id": "target",
            },
        ]
    }
    assert (
        pixels.pixel_measurements_for_signal(row, "target")["event-0"]["event_id"]
        == "refined"
    )
    assert "event-1" not in pixels.pixel_measurements_for_signal(row, "target")
    assert (
        pixels.pixel_measurements_for_signal(row, "other")["event-0"]["event_id"]
        == "original"
    )


def test_scheduler_includes_active_represented_event(tmp_path, monkeypatch):
    monkeypatch.setattr(pixels, "RUN", tmp_path)
    pixels.save(
        tmp_path / "sources/7.json",
        {
            "signals": [{"event_ids": ["active"], "triage": {"status": "ACTIVE"}}],
            "pixel_jobs": [
                {
                    "event_id": "dismissed",
                    "represented_event_ids": ["dismissed", "active"],
                    "quarter": 1,
                }
            ],
        },
    )
    tasks = pixels.localization_tasks([{"source_id": "7"}])
    assert len(tasks) == 1
    assert tasks[0][1][0]["event_id"] == "dismissed"


def test_shared_pixel_is_attributed_once_per_quarter():
    row = {
        "signals": [
            {
                "signal_id": "active",
                "triage": {"status": "ACTIVE"},
                "events": [
                    {"quarter": 1, "event_ids": ["active-q1", "other-q1"]},
                    {"quarter": 2, "event_ids": ["active-q2"]},
                    {"quarter": 3, "event_ids": ["active-q3"]},
                ],
            }
        ],
        "localization": {
            "events": [
                pixel("other-q1", ["other-q1", "active-q1"], 1),
                pixel("active-q2", ["active-q2"], 2),
                pixel("other-q3", ["other-q3", "active-q3"], 3),
            ]
        },
    }
    aggregate_signals(row)
    localization = row["signals"][0]["localization"]
    assert len(localization["good_quarters"]) == 3
    assert len(localization["quarters"]["1"]["individual_offsets_arcsec"]) == 1
    assert localization["source_attribution"]["status"] == "ROBUST_OFFSET_CONFLICT"
    held = {
        "status": "SKIPPED_HELD",
        "reason": "LOWER_PERIOD_ALIAS_REQUIRES_RESOLUTION",
    }
    row["signals"].append(
        {"signal_id": "held", "triage": {"status": "HELD"}, "localization": held}
    )
    row["localization"]["events"] = []
    aggregate_signals(row)
    assert row["signals"][0]["localization"] is localization
    assert row["signals"][1]["localization"] == held


def attribution(events, pixels_):
    row = {
        "signals": [
            {
                "signal_id": "signal",
                "triage": {"status": "ACTIVE"},
                "events": [
                    {"quarter": 5, "event_ids": [event_id]} for event_id in events
                ],
            }
        ],
        "localization": {"events": pixels_},
    }
    aggregate_signals(row)
    return row["signals"][0]["localization"]["source_attribution"]


def test_single_event_is_localized_in_its_quarter():
    offset = attribution(["event"], [pixel("event", ["event"], 5)])
    assert offset["basis"] == "SINGLE_QUARTER"
    assert offset["status"] == "ROBUST_OFFSET_CONFLICT"
    assert offset["target_snr"] == 20.0
    assert np.isclose(offset["offset_uncertainty_arcsec"], np.sqrt(0.01 + 0.5**2))

    near = attribution(["event"], [pixel("event", ["event"], 5, offset=0.3)])
    assert near["status"] == "TARGET_COMPATIBLE"

    faint = attribution(["event"], [pixel("event", ["event"], 5, snr=5.0)])
    assert faint["status"] == "INSUFFICIENT_QUARTER_SNR"
    assert "offset_distance_arcsec" not in faint

    outside = pixel("event", ["event"], 5)
    outside["target_inside_stamp"] = False
    assert attribution(["event"], [outside])["status"] == "INSUFFICIENT_GOOD_QUARTERS"


def test_repeated_events_keep_the_multi_quarter_rule():
    two = attribution(
        ["one", "two"], [pixel("one", ["one"], 5), pixel("two", ["two"], 5)]
    )
    assert two["status"] == "INSUFFICIENT_GOOD_QUARTERS"
    assert two["minimum_good_quarters"] == 3
    assert "basis" not in two


def test_singular_position_covariance_remains_unconstrained():
    record = pixel("singular", ["singular"], 5)
    record["tangent_covariance_arcsec2"] = [
        [4.364300969038285e53, 2.4594887187491277e53],
        [2.4594887187491273e53, 1.386037489295152e53],
    ]
    result = pixels.combine_positions([record])
    assert result["status"] == "POSITION_UNCONSTRAINED"
    assert result["reason"] == "SINGULAR_POSITION_COVARIANCE"
    assert "offset_arcsec" not in result
    assert attribution(["singular"], [record])["status"] == "INSUFFICIENT_GOOD_QUARTERS"


def test_diagnostic_singular_position_does_not_discard_valid_attribution():
    invalid = pixel("invalid", ["invalid"], 9)
    invalid["tangent_covariance_arcsec2"] = [[1.0, 1.0], [1.0, 1.0]]
    invalid["prf_correlation"] = -0.2
    valid = [pixel(str(q), [str(q)], q, offset=0.3) for q in (1, 2, 3)]
    row = {
        "signals": [
            {
                "signal_id": "signal",
                "triage": {"status": "ACTIVE"},
                "events": [
                    {"quarter": p["quarter"], "event_ids": [p["event_id"]]}
                    for p in [*valid, invalid]
                ],
            }
        ],
        "localization": {"events": [*valid, invalid]},
    }
    aggregate_signals(row)
    result = row["signals"][0]["localization"]
    assert result["quarters"]["9"]["status"] == "POSITION_UNCONSTRAINED"
    assert len(result["good_quarters"]) == 3
    assert result["source_attribution"]["status"] == "TARGET_COMPATIBLE"


def test_valid_position_combination_preserves_original_calculation():
    records = [
        pixel("one", ["one"], 5, offset=0.3),
        pixel("two", ["two"], 5, offset=0.5),
    ]
    result = pixels.combine_positions(records)
    np.testing.assert_array_equal(result["offset_arcsec"], [0.4, 0.0])
    np.testing.assert_array_equal(
        result["formal_independent_covariance_arcsec2"], [[0.005, 0.0], [0.0, 0.005]]
    )
