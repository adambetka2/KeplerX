"""Gold cases for heterogeneous delivered units and observed-event clocks."""

import numpy as np

from dwarfs.experiments.compare_photometry import (
    calibrated_curves,
    clock_match,
    injected_curves,
    normalizations,
    null_draw,
    source_bootstrap,
)
from dwarfs.experiments.reprocess_validation import CommonSearch


def test_view_specific_units_do_not_change_fractional_signal_amplitude():
    data = {
        "source_flux": np.array(1000.0),
        "target_profiles": np.tile([0.2, 0.3], (10, 1)),
        "aperture": np.array([1.0, 0.0]),
        "delivered_psf": np.full(10, 1300.0),
        "delivered_psf_flat": np.full(10, 900.0),
    }
    np.testing.assert_allclose(
        normalizations(data), [200.0, 200.0, 200.0, 1300.0, 900.0]
    )


def test_joint_null_preserves_pixels_background_and_delivered_covariance():
    rng = np.random.default_rng(72)
    residual = rng.normal(size=(100, 7))
    trend = np.tile([100.0, 20.0, 30.0, 40.0, 3.0, 1200.0, 900.0], (100, 1))
    data = {
        "time": np.arange(100) / 5,
        "normalization": np.array([50.0, 50.0, 50.0, 1200.0, 900.0]),
    }
    draw, signs = null_draw(data, trend, residual, 2.0, rng)
    actual = (
        np.column_stack(
            (
                draw["pixel_flux"],
                draw["background"],
                draw["delivered_psf"],
                draw["delivered_psf_flat"],
            )
        )
        - trend
    )
    np.testing.assert_allclose(
        actual[:, :, None] * actual[:, None, :],
        residual[:, :, None] * residual[:, None, :],
        atol=1e-11,
    )
    np.testing.assert_array_equal(draw["normalization"], data["normalization"])
    assert set(signs) == {-1, 1}


def test_observed_clock_accepts_compensated_period_and_epoch():
    time = np.arange(0.0, 90.0, 0.02043361111111111)
    truth = {"period_days": 10.0, "epoch_bkjd": 5.0, "duration_hours": 4.0}
    proposal = {"period_days": 10.015, "epoch_bkjd": 4.94, "duration_hours": 4.0}
    expected = -1.0 * (np.abs((time - 5.0 + 5.0) % 10.0 - 5.0) < 2 / 24)
    result = clock_match(time, proposal, truth, expected)
    assert result["clock_match"] and result["supported_events"] == 9
    assert result["maximum_timing_offset_hours"] < result["timing_tolerance_hours"]
    assert (
        abs(proposal["period_days"] / 10.0 - 1) * np.ptp(time)
        > 4 / 48 + 0.02043361111111111
    )


def test_observed_clock_rejects_true_mistiming_or_fewer_than_three_events():
    time = np.arange(0.0, 45.0, 0.02043361111111111)
    truth = {"period_days": 10.0, "epoch_bkjd": 5.0, "duration_hours": 2.0}
    proposal = {"period_days": 10.05, "epoch_bkjd": 5.0, "duration_hours": 2.0}
    expected = -1.0 * (np.abs((time - 5.0 + 5.0) % 10.0 - 5.0) < 1 / 24)
    assert not clock_match(time, proposal, truth, expected)["clock_match"]
    short = time < 20
    proposal["period_days"] = 10.0
    result = clock_match(time[short], proposal, truth, expected[short])
    assert result["supported_events"] == 2 and not result["clock_match"]


def test_source_bootstrap_uses_one_effect_per_source():
    result = source_bootstrap([-0.1, 0.1, 0.2, 0.3])
    assert result["sources"] == 4
    np.testing.assert_allclose(result["mean_source_effect"], 0.125)
    lower, upper = result["source_bootstrap_95_interval"]
    assert lower < 0.125 < upper


def test_full_refit_and_conditional_psf_share_intrinsic_injection():
    time = np.arange(0.0, 45.0, 0.02043361111111111)
    y, x = np.mgrid[-1:2, -1:2]
    profile = np.exp(-(x * x + y * y) / 2).ravel()
    profile /= profile.sum()
    rng = np.random.default_rng(81)
    pixels = 1000 * profile + 4 * np.sin(time[:, None] / 0.5)
    pixels += rng.normal(0, 0.1, pixels.shape)
    aperture = np.zeros(9)
    aperture[[1, 3, 4, 5, 7]] = 1
    data = {
        "time": time,
        "pixel_flux": pixels,
        "source_flux": np.array(1000.0),
        "target_profiles": np.tile(profile, (len(time), 1)),
        "profile_derivatives": np.tile(
            np.column_stack((profile * x.ravel(), profile * y.ravel())),
            (len(time), 1, 1),
        ),
        "pointing": np.column_stack((0.01 * np.sin(time), 0.01 * np.cos(time))),
        "cbv": np.column_stack([np.sin(time / j) for j in (1.7, 2.3, 3.1, 4.3)]),
        "aperture": aperture,
        "background": np.zeros(len(time)),
        "pixel_error": np.ones_like(pixels),
        "delivered_psf": 1300 + rng.normal(0, 0.1, len(time)),
        "delivered_psf_flat": 900 + rng.normal(0, 0.1, len(time)),
    }
    data["normalization"] = normalizations(data)
    baseline, _ = calibrated_curves(data)
    loss = 0.001 * (np.abs((time - 2.731 + 5.0) % 10.0 - 5.0) < 2 / 24)
    injected, expected, _ = injected_curves(data, loss, data["target_profiles"])
    for k in range(5):
        np.testing.assert_allclose(expected[:, k], -loss, atol=1e-15)
    np.testing.assert_allclose(
        (injected - baseline)[:, 3:], expected[:, 3:], atol=1e-15
    )
    search = CommonSearch(time, np.ones(len(time)))
    assert (
        0.95 < search.transfer(injected[:, 1] - baseline[:, 1], expected[:, 1]) < 1.05
    )
