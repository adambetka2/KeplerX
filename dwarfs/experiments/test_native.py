"""Ensure the paired native comparison isolates its declared configuration axes."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "native_experiment", Path(__file__).with_name("native.py")
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


def test_factorial_changes_only_declared_windows_and_knots():
    designs = NATIVE.designs()
    normalized = []
    for design in designs:
        config = design["config"]
        windows = (
            config["detrending"].pop("window_days"),
            config["long_period_search"]["detrending"].pop("window_days"),
        )
        knots = config["physical_template"].pop("knot_spacing_days")
        normalized.append((windows, knots, config))
    assert [item[:2] for item in normalized] == [
        ((0.5, 1.0), 3.0),
        ((1.0, 2.0), 3.0),
        ((0.5, 1.0), 1.5),
        ((1.0, 2.0), 1.5),
    ]
    assert all(item[2] == normalized[0][2] for item in normalized)


def test_production_scoring_rejects_wrong_alias():
    truth = {
        "period_days": 9.9731,
        "epoch_bkjd": 1002.731,
        "start_bkjd": 1000.0,
        "stop_bkjd": 1045.0,
        "duration_hours": 4.0,
        "exposure_days": 0.0204336,
    }
    assert NATIVE.evolution.matches({"period_days": 9.9731, "t0_bkjd": 1002.731}, truth)
    assert not NATIVE.evolution.matches(
        {"period_days": 4.98655, "t0_bkjd": 1002.731}, truth
    )
