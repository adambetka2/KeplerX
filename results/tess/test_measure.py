"""Frozen known planet, paired shallow increments and missing-data accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from tess.archive import digest
from tess.measure import fit_event, measure, overlap, phase

FIXTURE = Path(__file__).parent / "fixtures"


def gold():
    record = json.loads((FIXTURE / "hat-p-7-s14.json").read_text())
    path = FIXTURE / "hat-p-7-s14.npz"
    assert digest(path) == record["photometry_sha256"]
    with np.load(path) as z:
        data = dict(z)
    data.update(
        pixels=data["flux"][:, None, None],
        pixel_error=data["error"][:, None, None],
        aperture=np.ones((1, 1), bool),
        meta={"target_pixel_xy": [0, 0]},
    )
    return data, record["host"]["signals"][0]["clock"], record


def test_known_planet_native_tess():
    data, clock, reference = gold()
    result, _ = measure(data, clock, [], 3)
    primary = result["summary"]["primary"]
    low, high = reference["expected_tess_box_depth_fraction_interval"]
    assert primary["events"] == 12
    assert low < primary["depth_fraction"] < high
    assert primary["event_scatter_error"] > primary["formal_error"]
    # Formal odd/even alone would falsely flag this real planet.
    assert abs(result["summary"]["odd_minus_even_formal_sigma"]) > 4
    assert abs(result["summary"]["odd_minus_even_event_scatter_sigma"]) < 2


@pytest.mark.parametrize("depth", [40e-6, 100e-6, 1000e-6])
def test_paired_shallow_increments_on_native_noise(depth):
    data, known, _ = gold()
    clock = {"period_days": 3.71, "epoch_bjd_tdb": 2458684.1, "duration_days": 0.0833}
    before, _ = measure(data, clock, [known], 3)
    dt, _ = phase(data["time"], clock)
    changed = dict(data)
    changed["flux"] = data["flux"] - np.median(
        data["flux"][data["good"]]
    ) * depth * overlap(dt, clock["duration_days"], data["exposure"])
    changed["pixels"] = changed["flux"][:, None, None]
    after, _ = measure(changed, clock, [known], 3)
    increment = (
        after["summary"]["primary"]["depth_fraction"]
        - before["summary"]["primary"]["depth_fraction"]
    )
    assert increment == pytest.approx(depth, rel=0.05)


def test_partial_exposures_signed_depth_and_baseline():
    dt = np.array([-0.1, -0.06, -0.02, 0.02, 0.06, 0.1])
    exposure = np.full(6, 0.04)
    # A 0.02 d event is shorter than one exposure; two exposures each see 1/4.
    fraction = np.array([0, 0, 0.25, 0.25, 0, 0])
    np.testing.assert_allclose(overlap(dt, 0.02, exposure), fraction)
    for depth in [-40e-6, 40e-6]:
        flux = 100 + 2 * dt - 100 * depth * fraction
        result, _ = fit_event(dt, flux, np.full(6, 0.01), 0.02, exposure)
        assert result["depth_fraction"] == pytest.approx(depth, abs=1e-14)


def test_commensurate_signals_keep_overlap_explicit():
    data, clock, _ = gold()
    result, _ = measure(data, clock, [clock], 3)
    assert result["summary"]["primary"]["events"] == 12
    assert sum(e["other_signal_overlap_cadences"] for e in result["events"]) > 0


def test_no_coverage_retains_missing_measurement():
    data, clock, _ = gold()
    data["good"][:] = False
    data["time"][0] = np.nan
    result, _ = measure(data, clock, [], 3)
    assert result["summary"]["primary"]["status"] == "NO_MEASURABLE_DEPTH"
    assert result["summary"]["odd_minus_even_formal_sigma"] is None
    assert "depth_fraction" not in result["summary"]["primary"]
