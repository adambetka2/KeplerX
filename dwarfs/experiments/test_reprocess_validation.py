"""Gold checks for paired pixel nulls, transfer and source-normalized comparisons."""

import numpy as np

from dwarfs.experiments.reprocess_validation import (
    CommonSearch,
    clock_match,
    empirical_p,
    observed_noise,
    rate_summary,
    source_bootstrap,
    wild_draw,
)


def test_wild_null_keeps_pixel_and_background_covariance_together():
    time = np.arange(24) / 4
    residual = np.arange(24 * 4).reshape(24, 4) / 100
    trend = np.full_like(residual, 30)
    data = {"time": time, "pixel_flux": trend[:, :-1], "background": trend[:, -1]}
    draw = wild_draw(data, trend, residual, 1, np.random.default_rng(19))
    actual = np.column_stack((draw["pixel_flux"], draw["background"])) - trend
    np.testing.assert_allclose(
        actual[:, :, None] * actual[:, None, :],
        residual[:, :, None] * residual[:, None, :],
        atol=1e-13,
    )
    assert np.any(actual < 0) and np.any(actual > 0)


def test_empirical_p_includes_plus_one_and_ties():
    calibration = np.arange(19)
    np.testing.assert_allclose(empirical_p(calibration, [30, 18, -1]), [0.05, 0.1, 1])


def test_combined_false_alarm_rule_requires_both_calibrations():
    held = np.array([20, 40])
    pvalues = np.maximum(
        empirical_p(np.arange(19), held), empirical_p(np.arange(19) + 20, held)
    )
    summary = rate_summary(pvalues)
    assert summary["held_false_alarms"] == 1
    assert summary["held_false_alarm_rate"] == 0.5


def test_common_blind_search_recovers_known_box_after_smooth_nuisance():
    time = np.arange(0, 65, 0.02043361111111111)
    period = 10.037
    phase = (time - 2.731 + period / 2) % period - period / 2
    transit = -10 * (np.abs(phase) < 2 / 24)
    flux = 1000 + np.sin(time / 7) + transit
    search = CommonSearch(time, np.ones(len(time)))
    proposal = search.search(flux)
    truth = {"period_days": period, "epoch_bkjd": 2.731, "duration_hours": 4}
    matched, energy = clock_match(time, proposal, truth, transit)
    assert matched and energy > 0.8
    assert proposal["score"] > 50
    assert abs(proposal["duration_hours"] - 4) < 0.3
    np.testing.assert_allclose(search.transfer(1.25 * transit, transit), 1.25)


def test_clock_match_rejects_small_relative_period_error_with_large_drift():
    time = np.linspace(0, 90, 1000)
    truth = {"period_days": 10, "epoch_bkjd": 1, "duration_hours": 2}
    proposal = {"period_days": 10.05, "epoch_bkjd": 1, "duration_hours": 2}
    expected = -1.0 * (np.abs((time - 1 + 5) % 10 - 5) < 1 / 24)
    assert not clock_match(time, proposal, truth, expected)[0]


def test_noise_comparison_accounts_for_aperture_dilution():
    time = np.arange(0, 35, 0.02043361111111111)
    residual = np.random.default_rng(29).normal(size=len(time))
    psf = 1000 + residual
    data = {
        "time": time,
        "delivered_psf": psf,
        "target_profiles": np.tile([0.2, 0.3], (len(time), 1)),
        "aperture": np.array([1.0, 0.0]),
    }
    pair = np.column_stack(
        (200 + 0.2 * residual, 200 + 0.1 * residual, 200 + 0.15 * residual)
    )
    search = CommonSearch(time, np.ones(len(time)))
    report = observed_noise(data, pair, search, 1000)
    for hours in ("2.0", "4.0", "6.0"):
        sap = report["delivered_sap"][hours]["source_relative_sigma_ppm"]
        psf = report["delivered_psf"][hours]["source_relative_sigma_ppm"]
        new = report["pixel_reprocessed"][hours]["source_relative_sigma_ppm"]
        np.testing.assert_allclose([sap, new], [psf, psf / 2], rtol=1e-9)


def test_uncertainty_resamples_sources_not_individual_injections():
    result = source_bootstrap([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["sources"] == 5
    np.testing.assert_allclose(result["mean_source_effect"], 0.3)
    low, high = result["source_bootstrap_95_interval"]
    assert 0.1 <= low < 0.3 < high <= 0.5


def test_full_pixel_refit_preserves_signal_with_independent_nuisance():
    from dwarfs.experiments.reprocess import fit_extract

    time = np.arange(0, 45, 0.02043361111111111)
    y, x = np.mgrid[-1:2, -1:2]
    profile = np.exp(-(x * x + y * y) / 2).ravel()
    profile /= profile.sum()
    derivatives = np.column_stack((profile * x.ravel(), profile * y.ravel()))
    rng = np.random.default_rng(31)
    common = 10 * np.sin(time * 2 * np.pi / 2.17)
    pixels = 1000 * profile + common[:, None] + rng.normal(0, 0.1, (len(time), 9))
    aperture = np.zeros(9)
    aperture[[1, 3, 4, 5, 7]] = 1
    kwargs = {
        "time": time,
        "pointing": np.column_stack((0.01 * np.sin(time), 0.01 * np.cos(time))),
        "target_profiles": np.tile(profile, (len(time), 1)),
        "aperture": aperture,
        "background": np.zeros(len(time)),
        "pixel_error": np.ones_like(pixels),
        "profile_derivatives": np.tile(derivatives, (len(time), 1, 1)),
    }
    base = fit_extract(pixel_flux=pixels, **kwargs)
    loss = 0.001 * (np.abs((time - 2.731 + 5) % 10 - 5) < 2 / 24)
    delta = -1000 * loss[:, None] * profile
    injected = fit_extract(pixel_flux=pixels + delta, **kwargs)
    search = CommonSearch(time, np.sqrt(np.sum(aperture)) * np.ones(len(time)))
    retained = search.transfer(injected["flux"] - base["flux"], delta @ aperture)
    assert 0.95 < retained < 1.05
    assert np.max(np.abs(np.diff(injected["flux"] - base["flux"]))) > 0
