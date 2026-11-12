"""Transit templates as KBonus delivered transits: through its flattening."""

from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline

PHASES = 16  # knot positions averaged over one knot spacing
SHORTEST_DAYS = 0.25  # templates of 6 hours and longer
REACH = 3  # knot spacings either side of the transit that the fit sees
KNOT_DAYS = 2.0  # KBonus's spline knot spacing


def flattened(model, exposure, spacing):
    """The template minus the knot-phase average of its cubic B-spline fit (knots every `spacing` days)."""
    model = np.asarray(model, dtype=np.float64)
    inside = np.flatnonzero(model)
    if not len(inside):
        return model
    reach = int(np.ceil(REACH * spacing / exposure))
    lo = max(inside[0] - reach, 0)
    hi = min(inside[-1] + reach + 1, len(model))
    out = model.copy()
    window = _flattened_window(model[lo:hi].tobytes(), float(exposure), float(spacing))
    out[lo:hi] = np.frombuffer(window)
    return out


@lru_cache(maxsize=512)
def _flattened_window(values, exposure, spacing):
    """The fit's subtraction over one window (bytes of float64 in and out)."""
    y = np.frombuffer(values)
    t = np.arange(len(y)) * exposure
    fit = np.zeros(len(y))
    for k in range(PHASES):
        start = -spacing * (k + 0.5) / PHASES
        inner = np.arange(start + spacing, t[-1], spacing)
        inner = inner[inner > t[0]]
        knots = np.r_[[t[0]] * 4, inner, [t[-1]] * 4]
        design = BSpline.design_matrix(t, knots, 3).toarray()
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        fit += design @ coefficients
    return (y - fit / PHASES).tobytes()
