"""Checks for the self-calibrated periodogram significance."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolve_template import periodogram_significance


def band(duration, peak_at=None, floor=3.0, seed=0):
    period = np.geomspace(30, 100, 5000)
    snr = floor + 0.7 * np.random.default_rng(seed).standard_normal(period.size)
    snr[::97] = np.nan
    if peak_at is not None:
        snr[np.argmin(np.abs(period - peak_at))] = floor + 14.0
    arrays = {"period_days": period, "search_snr": snr}
    return {"search_band": "long", "duration_trial_days": duration, "periodogram_arrays": arrays}


def test_peak_is_measured_against_its_own_nearest_duration_null():
    diagnostics = {"bands": [band(0.06), band(0.24, peak_at=50.0, floor=4.0), {"search_band": "long"}]}
    result = periodogram_significance(diagnostics, "long", 50.0, 0.2)
    assert result["duration_trial_days"] == 0.24
    assert abs(result["null_median"] - 4.0) < 0.05
    assert 18 < result["sde"] < 22


def test_raised_floor_does_not_raise_significance():
    low = periodogram_significance({"bands": [band(0.24, floor=3.0)]}, "long", 50.0, 0.24)
    high = periodogram_significance({"bands": [band(0.24, floor=5.0)]}, "long", 50.0, 0.24)
    assert abs(low["sde"] - high["sde"]) < 1e-9
