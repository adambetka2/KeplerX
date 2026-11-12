"""Checks for event-level odd/even and consistency statistics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolve_template import event_consistency

CADENCE = 0.02043365


def series(depths, period=3.0, duration=0.1, red=0.0, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0, period * len(depths), CADENCE)
    flux = rng.normal(0, 1e-3, len(time))
    flux += red * np.repeat(rng.normal(0, 1, len(time) // 12 + 1), 12)[: len(time)]
    cycle = np.rint((time - 1.0) / period).astype(int)
    inside = np.abs(time - 1.0 - cycle * period) < duration / 2
    flux[inside] -= np.asarray(depths)[np.clip(cycle[inside], 0, len(depths) - 1)]
    return time, flux, period, duration


def test_constant_transit_under_red_noise_is_consistent():
    time, flux, period, duration = series([2e-3] * 40, red=1e-3)
    event = event_consistency(time, flux, period, 1.0, duration, 0.25)
    assert event["odd_even_z"] < 3.5
    assert event["event_t"] > 6


def test_alternating_depths_fail_odd_even():
    time, flux, period, duration = series([4e-3, 1e-3] * 20)
    assert (
        event_consistency(time, flux, period, 1.0, duration, 0.25)["odd_even_z"] > 3.5
    )


def test_single_strong_event_is_inconsistent():
    time, flux, period, duration = series([3e-2] + [0.0] * 39)
    assert event_consistency(time, flux, period, 1.0, duration, 0.25)["event_t"] < 6


def test_too_few_events_returns_none():
    time, flux, period, duration = series([2e-3] * 4)
    assert event_consistency(time, flux, period, 1.0, duration, 0.25) is None
