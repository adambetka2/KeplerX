"""Checks for time-slide box calibration."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from null_calibration import CADENCE_DAYS, box_depths, calibration


def series(red=0.0, seed=0, days=90.0):
    rng = np.random.default_rng(seed)
    time = np.arange(0, days, CADENCE_DAYS)
    flux = rng.normal(0, 1e-3, time.size)
    flux += red * np.repeat(rng.normal(0, 1, time.size // 8 + 1), 8)[: time.size]
    return time, flux, np.full(time.size, 1e-3)


def test_white_noise_is_calibrated():
    time, flux, noise = series()
    _, depth, error = box_depths(time, flux, noise, 2 / 24, 0.0)
    assert abs(calibration(depth / error)["kappa_mad"] - 1) < 0.08


def test_correlated_noise_inflates_kappa():
    time, flux, noise = series(red=1e-3)
    _, depth, error = box_depths(time, flux, noise, 4 / 24, 0.0)
    assert calibration(depth / error)["kappa_mad"] > 2


def test_box_recovers_depth_in_its_window():
    time, flux, noise = series()
    flux[np.floor(time / (2 / 24)) == 120] -= 5e-3
    window, depth, _ = box_depths(time, flux, noise, 2 / 24, 0.0)
    assert abs(depth[window == 120][0] - 5e-3) < 1e-3
