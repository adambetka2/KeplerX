"""KBonus's flattening of the stitched PSF, emulated."""

import math

import numpy as np
from scipy.interpolate import BSpline

SMOOTHER = {"knot_spacing_days": 2.0, "clip_sigma": None, "iterations": 1}


def robust_sigma(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def spline_fit(time, flux, spacing, sigma, clip, iterations):
    """Iteratively clipped least-squares cubic B-spline with uniform knots."""
    span = time[-1] - time[0]
    inner = max(int(span / spacing) - 1, 0)
    if len(time) < 20 or inner < 1:
        return np.full(len(time), np.median(flux))
    knots = np.r_[
        [time[0]] * 4, np.linspace(time[0], time[-1], inner + 2)[1:-1], [time[-1]] * 4
    ]
    design = BSpline.design_matrix(time, knots, 3).tocsr()
    use = np.ones(len(time), bool)
    ridge = 1e-10 * len(time)
    for _ in range(iterations):
        b = design[use]
        normal = (b.T @ b).toarray()
        normal[np.diag_indices_from(normal)] += ridge
        coefficients = np.linalg.solve(normal, b.T @ flux[use])
        fit = design @ coefficients
        if clip is None:
            break
        keep = np.abs(flux - fit) < clip * sigma
        if np.array_equal(keep, use) or keep.sum() < 20:
            break
        use = keep
    return fit


def trend(time, raw, quarter, groups, smoother=SMOOTHER):
    """Emulated KBonus trend of `raw` on the cadences of `groups` (per quarter)."""
    out = np.ones(len(time))
    for q in groups:
        m = quarter == q
        t, y = time[m], raw[m]
        sigma = robust_sigma(np.diff(y)) / math.sqrt(2)
        out[m] = spline_fit(
            t,
            y,
            smoother["knot_spacing_days"],
            sigma,
            smoother["clip_sigma"],
            smoother["iterations"],
        )
    return out
