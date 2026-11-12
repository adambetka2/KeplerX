"""Injected pixel gold tests for adaptive spatial extraction."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "adaptive_pixels", Path(__file__).with_name("reprocess.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def gold():
    rng = np.random.default_rng(4151)
    time = np.arange(0, 32, 0.0204336111)
    yy, xx = np.mgrid[-2:3, -2:3]
    profile = np.exp(-(xx**2 + yy**2) / 1.3)
    profile /= profile.sum()
    derivatives = np.stack((2 * xx * profile / 1.3, 2 * yy * profile / 1.3), axis=-1)
    target = np.broadcast_to(profile.ravel(), (len(time), profile.size)).copy()
    derivatives = np.broadcast_to(derivatives.reshape(-1, 2), (*target.shape, 2)).copy()
    aperture = (xx**2 + yy**2 <= 1).ravel().astype(float)
    background_mode = (1 + 0.1 * xx + 0.04 * yy).ravel()
    instrument = 20 * np.sin(2 * np.pi * time / 0.71) + 12 * np.cos(
        2 * np.pi * time / 1.17
    )
    pixels = 1000 * target + instrument[:, None] * background_mode + 100
    error = np.ones_like(pixels)
    pixels += rng.normal(0, error)
    pointing = np.column_stack((np.sin(time) * 0.03, np.cos(time / 0.8) * 0.02))
    return {
        "pixel_flux": pixels,
        "time": time,
        "pointing": pointing,
        "target_profiles": target,
        "aperture": aperture,
        "background": np.full(len(time), 100 * aperture.sum()),
        "pixel_error": error,
        "profile_derivatives": derivatives,
    }


def test_pixel_common_mode_is_removed_without_erasing_injected_transit():
    data = gold()
    original = MODULE.fit_extract(**data)
    phase = (data["time"] - 1.31 + 4.6) % 9.2 - 4.6
    loss = 3.0 * (np.abs(phase) < 0.08)
    injected = dict(
        data, pixel_flux=data["pixel_flux"] - loss[:, None] * data["target_profiles"]
    )
    corrected = MODULE.fit_extract(**injected)
    expected = -loss * (data["target_profiles"] @ data["aperture"])
    observed = corrected["flux"] - original["flux"]
    transfer = expected @ observed / (expected @ expected)
    assert 0.95 < transfer < 1.05
    assert np.std(np.diff(original["flux"])) < 0.3 * np.std(
        np.diff(original["baseline_flux"])
    )
    assert original["diagnostics"]["projection_target_response_error"] < 1e-10


def test_held_block_target_flux_does_not_train_its_prediction():
    data = gold()
    original = MODULE.fit_extract(**data)
    selected = (data["time"] > 8.5) & (data["time"] < 9.5)
    changed = dict(
        data,
        pixel_flux=data["pixel_flux"]
        + selected[:, None] * 10 * data["target_profiles"],
    )
    result = MODULE.fit_extract(**changed)
    own_fold = original["folds"] == 2
    # Global median recentering is a constant and cannot change a transit contrast.
    difference = result["prediction"][own_fold] - original["prediction"][own_fold]
    assert np.ptp(difference) < 1e-8


def test_external_only_ablation_does_not_use_reference_pixel_variations():
    data = gold()
    original = MODULE.fit_extract(**data, reference_pixels=False)
    changed = data["pixel_flux"].copy()
    changed[:, data["aperture"] == 0] += np.sin(data["time"][:, None]) * 200
    result = MODULE.fit_extract(
        **dict(data, pixel_flux=changed), reference_pixels=False
    )
    np.testing.assert_allclose(result["flux"], original["flux"], atol=1e-10)
    np.testing.assert_allclose(
        result["pixel_weights"], np.broadcast_to(data["aperture"], changed.shape)
    )
