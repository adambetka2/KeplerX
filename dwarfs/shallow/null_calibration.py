"""Time-slide empirical calibration of box transit-depth estimators on uninjected light curves."""

import numpy as np

CADENCE_DAYS = 0.02043365


def box_depths(time, flux, noise, duration, start, coverage_min=0.5):
    index = np.floor((time - start) / duration).astype(np.int64)
    if index.min() < 0:
        raise ValueError("Start must not follow the first cadence")
    weight = 1 / noise**2
    count = np.bincount(index)
    total = np.bincount(index, weight)
    signal = np.bincount(index, weight * flux)
    use = count >= coverage_min * duration / CADENCE_DAYS
    window = np.flatnonzero(use)
    return window, -signal[use] / total[use], 1 / np.sqrt(total[use])


def calibration(z):
    centre = np.median(z)
    return {
        "windows": int(z.size),
        "median": float(centre),
        "kappa_mad": float(1.4826 * np.median(np.abs(z - centre))),
        "kappa_std": float(np.std(z)),
        "tail_5": float(np.mean(np.abs(z - centre) > 5)),
    }
