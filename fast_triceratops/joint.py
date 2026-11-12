"""Fused V3 joint likelihood, bitwise equal to post_boi.quarters.SourceLikelihood.

Production V3 (post_boi/patches) does not call lnL_*_p. Each draw goes through
SourceLikelihood.__call__: upstream's simulate_*_p model, then per row
0.5 * (residual + sum(precision * (model - 1 - centre)**2)) and, for non-twin
binaries, the 1.5 sigma secondary veto. kernels.costs computes that with
pytransit's own kernel in row chunks, and vetoed binaries skip their primary.

Under accelerated_v3(), pruned() does not model rows that cannot matter.
Upstream keeps each scenario's lnL only for argsort()[0] and _log_mean_exp,
whose logsumexp (scipy 1.13) sees a row only through exp(logw - max); the other
99 "best" draws of each scenario's res are discarded by calc_probs and may
differ. A deferred row returns the cost SENTINEL + id * STEP (so lnL = -cost
exactly, far below every real row) when both hold:
- a lower bound on its cost already exceeds the best exactly computed cost by
  the limit, so it is not the maximum argsort reports (a tied maximum disables
  deferral for the call). With EXACT the bound is pytransit's own terms at
  probed grid points; with CERTIFIED it is kernels.cheap_row;
- (EXACT limit only) kernels.finite_rows proves its model finite, so it stays
  in logsumexp's finite set exactly where the true row would be.
fast_log_mean_exp then models exactly any deferred row whose bound, with its
prior, comes within the limit of the maximum. With EXACT every remaining row's
exp() is exactly zero either way, so the result is bitwise upstream's. With
CERTIFIED each dropped weight is below e^-50 of the maximum, which moves the
evidence by at most N e^-50 (2e-16 for N = 1e6) relative: below one ulp of lnZ.
"""

import contextlib

import numpy as np

from fast_triceratops.kernels import (
    PREPARE,
    cheap_bounds,
    cost_inputs,
    costs,
    evaluate,
    finite_rows,
    screened,
    standard,
)
from post_boi.quarters import SourceLikelihood
from post_boi.records import require

ORIGINAL_CALL = SourceLikelihood.__call__  # captured before any swap
ROUNDS = (2, 6, 24)  # probe points per round, most informative (precision * centre**2) first
SEEDS = 32  # rows modelled exactly first (lowest bounds or guesses), to find the best cost
EXACT = 760.0  # exp(x) is exactly 0 below x = -745.14; this keeps 15 units to spare
CERTIFIED = 50.0  # each dropped weight < e^-50 of the maximum: <= 2e-16 relative for 1e6 draws
MINIMUM = 64  # smaller calls are modelled whole
SENTINEL, STEP = 2.0**200, 2.0**148  # deferred cost SENTINEL + id * STEP: lognorm - cost = -cost
DEFERRED = {}  # id of a call's first deferred row -> Deferred
STATE = {"next": 0, "pruning": False, "limit": EXACT}


class Deferred:
    """The deferred rows of one likelihood call and what it takes to model them later."""

    def __init__(self, first, r, rows, bound, time, cost, lognorm):
        self.first, self.r, self.rows, self.bound = first, r, rows, bound
        self.time, self.cost, self.lognorm = time, cost, lognorm


def rounds(r, rows, time, cost, limit, values, exact):
    """Exact-mode bounds. The cost is 0.5 * (residual + a sum of nonnegative terms, one per
    grid point), so pytransit's own terms at any probed points give a lower bound. Rounds
    probe the most informative points (precision * centre**2) for the rows still within
    limit of the best exact cost, seeding it with the lowest bounds each round; rows whose
    probed excess, extrapolated twice over the whole information profile, still stays within
    the limit are modelled at once (efficiency only)."""
    floor, info = 0.5 * cost[4], cost[3] * cost[2]**2
    order = np.argsort(-info, kind="stable")
    terms = cost[:4] + (0.0, True)
    bound = np.full(rows.size, floor)
    best, start, probed = np.inf, 0, 0.0
    for size in ROUNDS:
        undecided = np.flatnonzero(~exact & ~(bound > best + limit))
        points = np.sort(order[start:start + size])
        start += size
        if undecided.size == 0 or points.size == 0:
            break
        bound[undecided] += costs(r, rows[undecided], time, terms, points)
        probed += info[points].sum()
        ranked = bound[undecided]
        seeds = undecided[np.argpartition(ranked, SEEDS)[:SEEDS]] if ranked.size > SEEDS else undecided
        seeds = np.union1d(seeds, undecided[~np.isfinite(ranked)])
        values[seeds] = costs(r, rows[seeds], time, cost)
        exact[seeds] = True
        best = np.min(values[exact], initial=np.inf, where=np.isfinite(values[exact]))
        if 0.0 < probed < info.sum() and np.isfinite(best):
            open_ = np.flatnonzero(~exact & ~(bound > best + limit))
            reach = floor + 2.0 * info.sum() / probed * (bound[open_] - floor)
            doomed = open_[reach <= best + limit]
            values[doomed] = costs(r, rows[doomed], time, cost)
            exact[doomed] = True
    return bound * (1 - 1e-12), best  # the probed terms were summed in a different order


def certified(r, rows, time, cost, limit, values, exact):
    """Certified-mode bounds: kernels.cheap_row, valid while pytransit stays within ETA of
    the exact occultation. Rows are modelled in order of the cheap guess, in doubling
    batches, while their bound is within limit of the best exact cost so far."""
    bound, guess = cheap_bounds(r, rows, time, cost)
    bounded = np.isfinite(bound)
    guess = np.where(bounded, guess, -np.inf)  # unbounded rows first
    best, size = np.inf, SEEDS
    while True:
        todo = np.flatnonzero(~exact & ~(bound > best + limit))
        if todo.size == 0:
            return np.where(bounded, bound, np.nan), best
        take = todo[np.argpartition(guess[todo], size)[:size]] if todo.size > size else todo
        values[take] = costs(r, rows[take], time, cost)
        exact[take] = True
        best = min(best, np.min(values[take], initial=np.inf, where=np.isfinite(values[take])))
        size *= 2


def pruned(name, time, flux, sigma, args, kwargs, source):
    """SourceLikelihood.__call__'s costs with deferred rows (module docstring)."""
    r = PREPARE[name](*args, **kwargs)
    out = np.full(r["k"].size, np.inf)
    rows = screened(r, sigma)
    cost = cost_inputs(flux, sigma, source)
    lognorm = float(source.log_normalization)
    require(abs(lognorm) < 2.0**100, "Gaussian normalization out of sentinel range")
    if rows.size < MINIMUM:
        out[rows] = costs(r, rows, time, cost)
        return out
    limit = STATE["limit"]
    values, exact = np.full(rows.size, np.nan), np.zeros(rows.size, bool)
    bound, best = (rounds if limit == EXACT else certified)(r, rows, time, cost, limit, values, exact)
    last = ~exact & ~(bound > best + limit)
    values[last] = costs(r, rows[last], time, cost)
    exact |= last
    deferred = np.flatnonzero(~exact)
    if deferred.size and limit == EXACT:  # certified mode tolerates a dropped NaN row
        unproven = deferred[~finite_rows(r, rows[deferred], time)]
        values[unproven] = costs(r, rows[unproven], time, cost)
        exact[unproven] = True
        deferred = np.flatnonzero(~exact)
    lnL = lognorm - values[exact]
    top = np.max(lnL, initial=-np.inf, where=np.isfinite(lnL))
    if deferred.size and np.count_nonzero(lnL == top) != 1:  # argsort's tie order needs every row
        values[deferred] = costs(r, rows[deferred], time, cost)
        exact[deferred] = True
        deferred = deferred[:0]
    require(np.all(np.abs(values[exact & np.isfinite(values)]) < 2.0**150),
            "Exact cost in sentinel range")
    out[rows[exact]] = values[exact]
    if deferred.size:
        first = STATE["next"]
        STATE["next"] += deferred.size
        DEFERRED[first] = Deferred(first, r, rows[deferred], bound[deferred], time, cost, lognorm)
        out[rows[deferred]] = SENTINEL + (first + np.arange(deferred.size)) * STEP
    return out


def fast_log_mean_exp(lnL, prior, N_total):
    """upstream's _log_mean_exp(lnL + prior) (or of lnL), modelling only the deferred rows
    that can carry weight."""
    from triceratops._numerics import _log_mean_exp

    logw = lnL if prior is None else lnL + prior
    marked = np.flatnonzero(np.isfinite(lnL) & (lnL <= -SENTINEL))
    if marked.size == 0:
        return _log_mean_exp(logw, N_total=N_total)
    ids = (-lnL[marked] - SENTINEL) / STEP
    firsts = np.array(sorted(DEFERRED), dtype=np.int64)
    require(firsts.size > 0 and np.array_equal(ids, np.round(ids)), "Unknown deferred row")
    ids = ids.astype(np.int64)
    owner = firsts[np.searchsorted(firsts, ids, side="right") - 1]
    index = ids - owner
    calls, upper = {}, np.empty(marked.size)  # upper: bound on each deferred row's lnL
    for first in np.unique(owner):
        call = calls[first] = DEFERRED.pop(int(first))
        mine = owner == first
        require(np.array_equal(np.sort(index[mine]), np.arange(call.rows.size)),
                "Deferred rows of one call split across evidences")
        upper[mine] = call.lognorm - call.bound[index[mine]]
    if prior is not None:
        upper = upper + prior[marked]
    pending = np.isfinite(logw[marked])
    genuine = np.isfinite(logw)
    genuine[marked] = False
    while True:
        top = np.max(logw, initial=-np.inf, where=genuine)
        need = pending & ~(upper < top - STATE["limit"])
        if not need.any():
            break
        for first in np.unique(owner[need]):
            pick = need & (owner == first)
            call = calls[first]
            cost = costs(call.r, call.rows[index[pick]], call.time, call.cost)
            exact = call.lognorm - cost
            logw[marked[pick]] = exact if prior is None else exact + prior[marked[pick]]
        genuine[marked[need]] = np.isfinite(logw[marked[need]])
        pending &= ~need
    return _log_mean_exp(logw, N_total=N_total)


def resolved(scenario):
    """A scenario function that must resolve every row it defers; a failure discards them."""
    def run(*args, **kwargs):
        require(not DEFERRED, "Deferred rows left by an earlier scenario")
        try:
            result = scenario(*args, **kwargs)
        except BaseException:
            DEFERRED.clear()
            raise
        require(not DEFERRED, "Deferred rows not resolved by their evidence")
        return result
    return run


def call(source, function, time, flux, sigma, *args, **kwargs):
    """Drop-in for SourceLikelihood.__call__: fused for lnL_*_p rows, upstream otherwise."""
    name = function.__name__
    if name in PREPARE and standard(name, time, flux, sigma, args, kwargs):
        require(np.array_equal(time, source.time), "Likelihood phase grid differs")
        if STATE["pruning"]:
            return pruned(name, time, flux, sigma, args, kwargs, source)
        return evaluate(name, time, flux, sigma, args, kwargs, source)
    return ORIGINAL_CALL(source, function, time, flux, sigma, *args, **kwargs)


@contextlib.contextmanager
def pruning(limit=EXACT):
    """Let call() defer rows. Only for scenario functions whose evidence goes through
    fast_log_mean_exp (scenarios.repaired_v3), which resolves them."""
    STATE.update(pruning=True, limit=limit)
    try:
        yield
    finally:
        STATE.update(pruning=False, limit=EXACT)
        DEFERRED.clear()


@contextlib.contextmanager
def accelerated_joint():
    """Swap SourceLikelihood.__call__ for the fused call, then restore."""
    SourceLikelihood.__call__ = call
    try:
        yield
    finally:
        SourceLikelihood.__call__ = ORIGINAL_CALL
