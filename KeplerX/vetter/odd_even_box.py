"""Odd and even transit depths with a fixed box shape: a linear estimate."""

import numpy as np

MIN_BASELINE = 3


def transit_depths(time, flux, error, period, t0, tdur, texp):
    """(epoch, depth, error) of every transit with in-transit and baseline cadences."""
    epoch = np.round((time - t0) / period)
    local = np.abs(time - (t0 + epoch * period))
    inner = (tdur - texp) / 2.0
    if inner <= 0 or not np.any(local <= inner):
        inner = tdur / 4.0
    inside = local <= inner
    around = (local >= tdur) & (local <= 3.0 * tdur)
    used = inside | around
    epochs, index = np.unique(epoch[used], return_inverse=True)
    weight = 1.0 / error[used] ** 2
    sums = []
    for part in (inside[used], around[used]):
        w = np.where(part, weight, 0.0)
        sums.append((
            np.bincount(index, w, len(epochs)),
            np.bincount(index, w * flux[used], len(epochs)),
            np.bincount(index, part, len(epochs)),
        ))  # fmt: skip
    (wa, fa, na), (wb, fb, nb) = sums
    ok = (na > 0) & (nb >= MIN_BASELINE)
    depth = fb[ok] / wb[ok] - fa[ok] / wa[ok]
    return np.column_stack([epochs[ok], depth, np.sqrt(1.0 / wa[ok] + 1.0 / wb[ok])])


def parity_depth(depths, errors):
    """Inverse-variance mean, its error widened by the scatter when above noise."""
    w = 1.0 / errors**2
    mean = (w @ depths) / w.sum()
    error = 1.0 / np.sqrt(w.sum())
    if len(depths) > 1:
        reduced = (w @ (depths - mean) ** 2) / (len(depths) - 1)
        error *= np.sqrt(max(1.0, reduced))
    return mean, error


KEYS = (
    "odd_even_box_stat", "odd_box_depth", "even_box_depth", "odd_box_depth_err",
    "even_box_depth_err", "odd_box_transits", "even_box_transits",
)  # fmt: skip


def odd_even_box(time, flux, error, period, t0, tdur, texp):
    """The odd/even box metrics; None where a parity has no measurable transit."""
    out = dict.fromkeys(KEYS)
    if not (period > 0 and tdur > 0 and np.isfinite([period, t0, tdur]).all()):
        return out
    rows = transit_depths(time, flux, error, period, t0, tdur, texp)
    odd = rows[rows[:, 0] % 2 == 1]
    even = rows[rows[:, 0] % 2 == 0]
    out["odd_box_transits"], out["even_box_transits"] = len(odd), len(even)
    if not len(odd) or not len(even):
        return out
    odd_depth, odd_error = parity_depth(odd[:, 1], odd[:, 2])
    even_depth, even_error = parity_depth(even[:, 1], even[:, 2])
    stat = abs(odd_depth - even_depth) / np.hypot(odd_error, even_error)
    return out | {
        "odd_even_box_stat": stat,
        "odd_box_depth": odd_depth,
        "even_box_depth": even_depth,
        "odd_box_depth_err": odd_error,
        "even_box_depth_err": even_error,
    }
