"""Cadence masks applied before the shallow transit search.

Two population systematics in faint KBonus SAP curves carry no transit information:

* Thermal settling. After every data gap (quarterly roll, monthly downlink, safe
  mode) many unrelated faint stars dip together for a few hours; the population
  rate of < -3 sigma cadences is 13x baseline in the first 0.1 d and returns to
  baseline by 0.5 d. These dips recur at the ~31 d and ~93 d gap spacing and build
  strong false periodic signals.
* Brightening outliers (flares, cosmic rays). A cadence whose selected seed-stage
  residual exceeds `threshold` local standard deviations is masked. Only positive
  residuals are masked, so a transit dip is never removed by this rule.
"""

import numpy as np
from evolve_detector import COMMON
from evolve_noise import noise_weights


def settling(records, gap_days, settle_days):
    """True for cadences within `settle_days` after the start of each data segment."""
    time = records["time"]
    starts = np.r_[True, np.diff(time) > gap_days]
    segment_start = time[
        np.maximum.accumulate(np.where(starts, np.arange(len(time)), 0))
    ]
    return time - segment_start < settle_days


def upper_outliers(records, config, threshold):
    """True for cadences whose seed-stage residual exceeds `threshold` local sigma."""
    if config["noise_model"]["method"] != "local_point":
        raise ValueError("Upper outlier masking requires the local_point noise model")
    quarters, counts = np.unique(records["quarter"], return_counts=True)
    retained = np.flatnonzero(
        np.isin(records["quarter"], quarters[counts >= config["min_quarter_cadences"]])
    )
    time, flux, base, quarter = COMMON.whiten(
        records,
        config["detrend_window_cadences"],
        config["min_quarter_cadences"],
        config.get("detrending"),
    )
    for value in np.unique(quarter):
        use = quarter == value
        flux[use] -= np.median(flux[use])
    noise = noise_weights(
        time, flux, quarter.astype(np.int64), base, 1.0, config["noise_model"]
    )
    mask = np.zeros(len(records), dtype=bool)
    mask[retained] = flux / noise > threshold
    return mask


def search_mask(records, config, gap_days=0.25, settle_days=0.5, threshold=4.0):
    """Settling cadences first, then brightening outliers of the remaining curve."""
    mask = settling(records, gap_days, settle_days)
    kept = np.flatnonzero(~mask)
    mask[kept[upper_outliers(records[kept], config, threshold)]] = True
    return mask
