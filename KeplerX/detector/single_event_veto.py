"""Single-event veto as an outlier test on the strongest event."""

import numba
import numpy as np
from scipy.stats import norm

OUTLIER_MARGIN = 3.0
TABLE_DAYS = 0.004  # time-table step (about a fifth of a Kepler cadence)


def expected_maxima(count):
    """Expected maximum of n standard normal draws for n = 0..count (Blom)."""
    n = np.arange(1, count + 1, dtype=np.float64)
    return np.concatenate([[0.0], norm.ppf((n - 0.375) / (n + 0.25))])


@numba.njit(cache=True)
def time_table(t):
    """table[g] = searchsorted(t, g * TABLE_DAYS) for sorted t from 0."""
    n = len(t)
    size = int(t[n - 1] / TABLE_DAYS) + 3
    table = np.empty(size, dtype=np.int64)
    j = 0
    for g in range(size):
        while j < n and t[j] < g * TABLE_DAYS:
            j += 1
        table[g] = j
    return table


@numba.njit(cache=True)
def winner_kept(t, table, num, den, P, nb, d, b, expected, margin, threshold):
    """Whether one winner (period P, nb bins, duration row d, bin b) is kept."""
    n, size = len(t), len(table)
    width = P / nb
    total, strongest, count = 0.0, -np.inf, 0
    s_num, s_den, drop_num, drop_den = 0.0, 0.0, 0.0, 0.0
    k = 0
    while k * P + b * width <= t[n - 1]:
        start = k * P + b * width
        j = table[min(max(int(np.floor(start / TABLE_DAYS)) - 1, 0), size - 1)]
        while j < n and t[j] < start:
            j += 1
        while j < n and t[j] < start + width:
            if den[d, j] > 0:
                ses = num[d, j] / np.sqrt(den[d, j])
                total += ses
                count += 1
                s_num += num[d, j]
                s_den += den[d, j]
                if ses > strongest:
                    strongest = ses
                    drop_num, drop_den = num[d, j], den[d, j]
            j += 1
        k += 1
    if count >= 2:
        others = (total - strongest) / (count - 1)
        e = expected[min(count, len(expected) - 1)]
        outlier = strongest - others > e + margin
        rest = (s_num - drop_num) / np.sqrt(s_den - drop_den)
        return not outlier or rest >= threshold
    return False
