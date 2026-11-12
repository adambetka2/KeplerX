"""Analytic target transfer and electron-count variance gold cases."""

import numpy as np

from dwarfs.experiments.pixels import (
    compare_delivered,
    fit_scene,
    inject_pixels,
    load_aperture,
    photon_budget,
    project_pixels,
    target_operator,
)


def test_fixed_scene_operator_removes_neighbor_and_background():
    target = np.array([0.7, 0.2, 0.08, 0.02])
    neighbor = target[::-1]
    nuisance = np.vstack((neighbor, np.ones(4)))
    variance = np.array([4.0, 9.0, 16.0, 25.0])
    operator = target_operator(target, nuisance, variance)
    baseline = 100 * target + 400 * neighbor + 20
    result, _ = project_pixels(baseline, np.sqrt(variance), operator)
    np.testing.assert_allclose(result, 100, atol=1e-11)
    np.testing.assert_allclose(nuisance @ operator, 0, atol=1e-12)
    fitted, residual = fit_scene(np.vstack((target, neighbor)), baseline, variance)
    np.testing.assert_allclose(fitted, [100, 400, 20], atol=1e-10)
    np.testing.assert_allclose(residual, 0, atol=1e-10)


def test_pixel_injection_commutes_with_frozen_operator_and_tracks_dilution():
    target = np.array([[0.6, 0.3, 0.1], [0.5, 0.35, 0.15], [0.4, 0.4, 0.2]])
    baseline = 100 * target + 40
    loss = np.array([0.0, 0.001, 0.002])
    injected, delta = inject_pixels(baseline, target, 100, loss)
    operator = np.array([1.0, 1.0, 0.0])
    extracted = injected @ operator
    source_flux = 100 * (target @ operator)
    np.testing.assert_allclose(extracted - baseline @ operator, -loss * source_flux)
    np.testing.assert_allclose(delta.sum(axis=1), -100 * loss)
    assert not np.allclose(source_flux, baseline @ operator)


def test_live_time_photon_and_read_budget_has_expected_electron_scaling():
    metadata = {
        "INT_TIME": {"value": 6.0},
        "NUM_FRM": {"value": 10},
        "READNOIS": {"value": 3.0, "comment": "[electrons] read noise"},
    }
    profiles = np.array([[0.75, 0.25], [0.1, 0.9]])
    coefficients = np.array([120.0, 60.0, 0.0])
    expected = (120 * profiles[0] + 60 * profiles[1] + 30) / 60 + 90 / 60**2
    data = {
        "flux_bkg": np.full((3, 2), 30.0),
        "flux_err": np.broadcast_to(np.sqrt(expected), (3, 2)),
        "flux_bkg_err": np.full((3, 2), 0.2),
    }
    budget, details = photon_budget(data, profiles, coefficients, metadata, {})
    np.testing.assert_allclose(budget["unexplained_signed"], 0, atol=1e-14)
    np.testing.assert_allclose(budget["read"], 0.025)
    assert details["live_seconds"] == 60
    assert not details["ultimate_bound_established"]


def test_read_noise_missing_does_not_become_zero_or_hide_signed_mismatch():
    metadata = {"INT_TIME": {"value": 1.0}, "NUM_FRM": {"value": 1}}
    data = {
        "flux_bkg": np.ones((2, 2)),
        "flux_err": np.ones((2, 2)),
        "flux_bkg_err": np.ones((2, 2)),
    }
    budget, details = photon_budget(
        data, np.array([[0.5, 0.5]]), np.array([4.0, 0.0]), metadata, {}
    )
    assert "read" not in budget
    np.testing.assert_allclose(budget["unexplained_signed"], -2)
    assert details["read_noise_status"] == "MISSING_READNOIS"


def test_delivered_equivalence_requires_absolute_flux_not_shape_agreement():
    cadence = np.arange(5)
    flux = np.array([100.0, 102.0, 99.0, 101.0, 98.0])
    error = np.ones(5)
    delivered = {"cadence": cadence, "flux": flux, "error": error}
    assert (
        compare_delivered(cadence, flux, error, delivered)["status"]
        == "MATCHED_DELIVERED_SAP"
    )
    delivered["flux"] = flux + 10
    assert (
        compare_delivered(cadence, flux, error, delivered)["status"]
        == "APERTURE_REPLAY_DIFFERS"
    )


def test_aperture_zero_padding_can_extend_beyond_stamp():
    case = {"aperture": [[1, 1, 0], [0, 0, 0]], "aperture_offset_in_tpf_yx": [0, 3]}
    actual = load_aperture(case, (2, 5))
    expected = np.zeros((2, 5), dtype=bool)
    expected[0, 3:5] = True
    np.testing.assert_array_equal(actual, expected)
