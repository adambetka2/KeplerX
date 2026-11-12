"""Epoch-aware candidate cleaning and ephemeris refinement."""

import math

import numba
import numpy as np

from detector import single_event_veto as sev

# upstream.clean_tces_of_harmonics's integer ratios.
MULTIPLES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], np.float64)
SAME_PERIOD = 0.02  # upstream's first-stage window
HARMONIC = 0.01  # upstream's harmonic tolerance, an upper bound here
EPOCH_DURATIONS = 2.0  # upstream remove_TCE_harmonics' same-epoch tolerance
PERIOD_STEPS = 10  # refinement steps per trial-period step, each side
EPOCH_STEPS = 6  # quarter cadences, each side
TABLE_DAYS = 0.004
CHECK_SHARE = 0.85  # upstream mask_highest_TCE_and_check_others' mes_frac
REFINED = 64  # survivors refined per pass, strongest first
ALIAS_Q = 3
# Alias ratios are exact to the refined periods' precision (about 1e-5).
ALIAS_TOLERANCE = 2e-4
# A fold whose next event is below this share of its k-th rests on its k strongest.
CARRIED_FRACTION = 0.2


@numba.njit(cache=True)
def _circular(x, m):
    return abs((x + 0.5 * m) % m - 0.5 * m)


@numba.njit(cache=True)
def _shared(period, mid, tdur, i, j, baseline, multiples, same, harmonic,
            epoch_durations):  # fmt: skip
    """Whether detection j's events are detection i's."""
    tol = epoch_durations * max(tdur[i], tdur[j])
    d = mid[j] - mid[i]
    r = period[j] / period[i]
    if abs(r - 1.0) < same:
        return _circular(d, period[i]) < tol
    ratio_tol = min(harmonic, 2.0 * tol / baseline)
    for x in multiples:
        if abs(r / x - 1.0) < ratio_tol and _circular(d, period[i]) < tol:
            return True
        if abs(r * x - 1.0) < ratio_tol and _circular(d, period[j]) < tol:
            return True
    return False


BUCKET = math.log(1.02)
WIDEN = 1e-9  # relative margin on every bucket window
CHEAP_TRANSITS = 48


@numba.njit(cache=True)
def _bucket(period, origin):
    return int(math.floor((math.log(period) - origin) / BUCKET))


@numba.njit(cache=True)
def _shared_any(lo, hi, origin, head, nxt, period, mid, tdur, j, baseline,
                multiples, same, harmonic, epoch_durations):  # fmt: skip
    """Whether an accepted detection with period in [lo, hi] shares j's events."""
    for b in range(
        max(_bucket(lo, origin), 0), min(_bucket(hi, origin), len(head) - 1) + 1
    ):
        a = head[b]
        while a >= 0:
            if _shared(period, mid, tdur, a, j, baseline, multiples, same, harmonic,
                       epoch_durations):  # fmt: skip
                return True
            a = nxt[a]
    return False


@numba.njit(cache=True)
def _suppressed(j, origin, head, nxt, period, mid, tdur, baseline, multiples, same,
                harmonic, epoch_durations):  # fmt: skip
    """Whether an accepted detection shares j's events."""
    p = period[j]
    if _shared_any(p / (1.0 + same) * (1.0 - WIDEN), p / (1.0 - same) * (1.0 + WIDEN),
                   origin, head, nxt, period, mid, tdur, j, baseline, multiples, same,
                   harmonic, epoch_durations):  # fmt: skip
        return True
    for x in multiples:
        for centre in (p / x, p * x):
            if _shared_any(centre / (1.0 + harmonic) * (1.0 - WIDEN),
                           centre / (1.0 - harmonic) * (1.0 + WIDEN), origin, head,
                           nxt, period, mid, tdur, j, baseline, multiples, same,
                           harmonic, epoch_durations):  # fmt: skip
                return True
    return False


@numba.njit(cache=True)
def _greedy(period, mes, t0, tdur, t_mid, baseline, multiples, same, harmonic,
            epoch_durations, t, num, den, nbins, rows, bins, expected, margin,
            threshold, veto):  # fmt: skip
    n = len(period)
    order = np.argsort(-mes, kind="mergesort")
    kept = np.zeros(n, np.bool_)
    if n == 0:
        return kept
    mid = np.empty(n)
    for i in range(n):
        mid[i] = t0[i] + math.floor((t_mid - t0[i]) / period[i] + 0.5) * period[i]
    top = multiples.max() if len(multiples) else 1.0
    origin = (
        math.log(period.min() / (top * (1.0 + harmonic)) / (1.0 + same)) - 2 * BUCKET
    )
    head = np.full(_bucket(period.max() * top / (1.0 - harmonic) / (1.0 - same), origin) + 3,
                   -1, np.int64)  # fmt: skip
    nxt = np.full(n, -1, np.int64)
    table = sev.time_table(t) if veto else np.zeros(1, np.int64)
    span = t[len(t) - 1]
    for jj in range(n):
        j = order[jj]
        cheap = veto and span / period[j] < CHEAP_TRANSITS
        if cheap and not sev.winner_kept(t, table, num, den, period[j], nbins[j], rows[j],
                                         bins[j], expected, margin, threshold):  # fmt: skip
            continue
        if _suppressed(j, origin, head, nxt, period, mid, tdur, baseline, multiples,
                       same, harmonic, epoch_durations):  # fmt: skip
            continue
        if veto and not cheap and not sev.winner_kept(
            t, table, num, den, period[j], nbins[j], rows[j], bins[j], expected, margin,
            threshold,
        ):  # fmt: skip
            continue
        b = _bucket(period[j], origin)
        nxt[j] = head[b]
        head[b] = j
        kept[j] = True
    return kept


def clean(period, mes, t0, tdur, t_mid, baseline, multiples, same, harmonic,
          epoch_durations):  # fmt: skip
    """Mask of detections kept by epoch-aware greedy suppression."""
    none = np.zeros((1, 1))
    return _greedy(period, mes, t0, tdur, t_mid, baseline, multiples, same, harmonic,
                   epoch_durations, np.zeros(1), none, none, np.zeros(0, np.int64),
                   np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(1), 0.0, 0.0,
                   False)  # fmt: skip


def vetted_clean(period, mes, t0, tdur, t_mid, baseline, multiples, same, harmonic,
                 epoch_durations, t, num, den, nbins, rows, bins, expected, margin,
                 threshold):  # fmt: skip
    """clean of the winners the outlier veto keeps."""
    return _greedy(period, mes, t0, tdur, t_mid, baseline, multiples, same, harmonic,
                   epoch_durations, t, num, den, nbins, rows, bins, expected, margin,
                   threshold, True)  # fmt: skip


@numba.njit(cache=True)
def _table(t):
    n = len(t)
    size = int(t[n - 1] / TABLE_DAYS) + 3
    table = np.empty(size, np.int64)
    j = 0
    for g in range(size):
        while j < n and t[j] < g * TABLE_DAYS:
            j += 1
        table[g] = j
    return table


@numba.njit(cache=True)
def _transit_sums(t, table, num, den, use, P, T, texp, lo, sum_num, sum_den):
    """Add each transit's first usable cadence within half an exposure of its time to the rows' sums."""
    n, size = len(t), len(table)
    first = math.ceil((0.5 * texp - T) / P)
    last = math.floor((t[n - 1] + 0.5 * texp - T) / P)
    sum_num[:] = 0.0
    sum_den[:] = 0.0
    transits = 0
    for k in range(first, last + 1):
        c = T + k * P
        start = c - 0.5 * texp
        g = int(math.floor(start / TABLE_DAYS)) - 1
        j = table[min(max(g, 0), size - 1)]
        while j < n and t[j] < start:
            j += 1
        while j < n and t[j] < c + 0.5 * texp and not use[j]:
            j += 1
        if j < n and t[j] < c + 0.5 * texp:
            transits += 1
            for r in range(len(sum_num)):
                sum_num[r] += num[lo + r, j]
                sum_den[r] += den[lo + r, j]
    return transits


@numba.njit(cache=True)
def refine(t, num, den, use, period, step, n_period, centre, texp, n_epoch, lo, hi,
           min_transits):  # fmt: skip
    """Best nearest-cadence MES over periods period + step * k / n_period."""
    table = _table(t)
    sum_num = np.empty(hi - lo)
    sum_den = np.empty(hi - lo)
    best = (-np.inf, period, centre, lo, 0)
    for kp in range(-n_period, n_period + 1):
        P = period + step * kp / max(n_period, 1)
        if P <= 0:
            continue
        for ke in range(-n_epoch, n_epoch + 1):
            T = centre + 0.25 * texp * ke
            transits = _transit_sums(
                t, table, num, den, use, P, T, texp, lo, sum_num, sum_den
            )
            if transits < min_transits:
                continue
            for r in range(hi - lo):
                if sum_den[r] > 0:
                    m = sum_num[r] / math.sqrt(sum_den[r])
                    if m > best[0]:
                        best = (m, P, T, lo + r, transits)
    return best


ALIAS_RATIOS = (1 / 2, 1 / 3, 2.0, 3.0)
# An alias replaces a detection only with this much more MES, with and without its strongest event.
ALIAS_MARGIN = 1.05


@numba.njit(cache=True)
def mes_without_strongest(t, num, den, use, period, epoch, texp, row):
    """MES of row over the events of event_sums, leaving out the event with the largest numerator."""
    n = len(t)
    first = math.ceil((-0.5 * texp - epoch) / period)
    last = math.floor((t[n - 1] + 0.5 * texp - epoch) / period)
    sum_num, sum_den, top_num, top_den = 0.0, 0.0, -math.inf, 0.0
    j = 0
    for k in range(first, last + 1):
        c = epoch + k * period
        while j < n and t[j] < c - 0.5 * texp:
            j += 1
        i = j
        while i < n and t[i] < c + 0.5 * texp and not use[i]:
            i += 1
        if i < n and t[i] < c + 0.5 * texp:
            sum_num += num[row, i]
            sum_den += den[row, i]
            if num[row, i] > top_num:
                top_num, top_den = num[row, i], den[row, i]
    if sum_den - top_den <= 0.0:
        return -math.inf
    return (sum_num - top_num) / math.sqrt(sum_den - top_den)


def strongest_alias(t, num, den, use, periods, lo, hi, texp, min_transits, found):
    """found = (mes, period, epoch, row) refined."""
    mes, period, epoch, _ = found
    ascending = periods[0] <= periods[-1]
    grid = periods if ascending else periods[::-1]
    best = found
    for ratio in ALIAS_RATIOS:
        alias = period * ratio
        k = int(np.searchsorted(grid, alias))
        if k <= 0 or k >= len(grid):
            continue
        step = float(grid[k] - grid[k - 1])
        index = k if ascending else len(grid) - 1 - k
        for j in range(int(ratio) if ratio > 1 else 1):
            m, p, e, r, _ = refine(
                t, num, den, use, alias, step, PERIOD_STEPS, epoch + j * period,
                texp, EPOCH_STEPS, int(lo[index]), int(hi[index]), min_transits,
            )  # fmt: skip
            if (
                m > mes * ALIAS_MARGIN
                and m > best[0]
                and mes_without_strongest(t, num, den, use, p, e, texp, r)
                > mes * ALIAS_MARGIN
            ):
                best = (m, p, e, r)
    return best


def mid_epoch(t, period, phase):
    """The transit time nearest the middle of t for phase (from t[0])."""
    middle = 0.5 * (t[0] + t[-1]) - t[0]
    return phase + np.floor((middle - phase) / period + 0.5) * period


def first_epoch(period, epoch):
    """The first transit at or after time 0 of epoch's train."""
    return epoch - np.floor(epoch / period) * period


def check_others(TCEs, time, num, den, tdurs, threshold, min_transits=2):
    """upstream mask_highest_TCE_and_check_others at the refined ephemerides with a depth-consistency test."""
    order = TCEs[:, 2].argsort(kind="stable")[::-1]
    tces_sorted = TCEs[order]
    texp = float(np.nanmin(np.abs(np.diff(time))))
    t = np.ascontiguousarray(time - time[0])
    every = np.ones(len(t), dtype=bool)
    use = transit_mask(t, tces_sorted[0], time[0])
    keep = np.zeros(len(tces_sorted), dtype=bool)
    keep[0] = True
    for i in range(1, len(tces_sorted)):
        _, period, _, t0, tdur = tces_sorted[i]
        if any(alias(tces_sorted[i], tces_sorted[k]) for k in np.flatnonzero(keep)):
            continue
        row = int(np.argmin(np.abs(tdur - tdurs)))
        epoch = float(t0 - time[0])
        n_all, d_all, k_all = event_sums(t, num, den, every, period, epoch, texp, row)
        n_rem, d_rem, k_rem = event_sums(t, num, den, use, period, epoch, texp, row)
        if k_rem < min_transits or d_rem <= 0 or d_all <= 0 or n_all <= 0:
            continue
        depth_all, depth_rem = n_all / d_all, n_rem / d_rem
        if depth_rem >= CHECK_SHARE * depth_all and n_rem / np.sqrt(d_rem) > threshold:
            keep[i] = True
            use &= transit_mask(t, tces_sorted[i], time[0])
    return tces_sorted[keep]


def alias(weak, strong):
    """Whether weak's events lie on strong's at a ratio p / q (q <= ALIAS_Q)."""
    _, p_weak, _, t_weak, d_weak = weak
    _, p_strong, _, t_strong, d_strong = strong
    short, long = sorted((p_weak, p_strong))
    for q in range(1, ALIAS_Q + 1):
        p = round(q * long / short)
        if p < 1 or abs(q * long / short - p) > ALIAS_TOLERANCE * p:
            continue
        spacing = short / q
        offset = (t_weak - t_strong + 0.5 * spacing) % spacing - 0.5 * spacing
        if abs(offset) < max(d_weak, d_strong):
            return True
    return False


@numba.njit(cache=True)
def event_sums(t, num, den, use, period, epoch, texp, row):
    """Sums of num and den of row over each transit's first usable cadence near its time."""
    n = len(t)
    first = math.ceil((-0.5 * texp - epoch) / period)
    last = math.floor((t[n - 1] + 0.5 * texp - epoch) / period)
    sum_num, sum_den, transits = 0.0, 0.0, 0
    j = 0
    for k in range(first, last + 1):
        c = epoch + k * period
        while j < n and t[j] < c - 0.5 * texp:
            j += 1
        i = j
        while i < n and t[i] < c + 0.5 * texp and not use[i]:
            i += 1
        if i < n and t[i] < c + 0.5 * texp:
            sum_num += num[row, i]
            sum_den += den[row, i]
            transits += 1
    return sum_num, sum_den, transits


@numba.njit(cache=True)
def event_terms(t, num, den, use, period, epoch, texp, row):
    """num and den of row at each transit event_sums reads, and the transits' centres."""
    n = len(t)
    first = math.ceil((-0.5 * texp - epoch) / period)
    last = math.floor((t[n - 1] + 0.5 * texp - epoch) / period)
    size = max(last - first + 1, 0)
    nums, dens, centres = np.empty(size), np.empty(size), np.empty(size)
    count = 0
    j = 0
    for k in range(first, last + 1):
        c = epoch + k * period
        while j < n and t[j] < c - 0.5 * texp:
            j += 1
        i = j
        while i < n and t[i] < c + 0.5 * texp and not use[i]:
            i += 1
        if i < n and t[i] < c + 0.5 * texp and den[row, i] > 0:
            nums[count], dens[count], centres[count] = num[row, i], den[row, i], c
            count += 1
    return nums[:count], dens[:count], centres[:count]


def carrying_events(nums, dens, threshold, min_transits):
    """The events a fold of events (num, den) rests on when they are fewer than the transit requirement."""
    ses = nums / np.sqrt(dens)
    order = np.argsort(-ses, kind="stable")
    for k in range(1, min_transits):
        if len(order) > k and ses[order[k]] < CARRIED_FRACTION * ses[order[k - 1]]:
            rest = order[k:]
            if (
                len(rest) < min_transits
                or nums[rest].sum() / math.sqrt(dens[rest].sum()) < threshold
            ):
                return order[:k]
            break
    return order[:0]


def transit_mask(t, tce, origin):
    """upstream make_transit_mask (cadences farther than one duration)."""
    _, period, _, t0, tdur = tce
    fold = (t + origin - t0 + period / 2.0) % period - period / 2.0
    return np.abs(fold) > tdur
