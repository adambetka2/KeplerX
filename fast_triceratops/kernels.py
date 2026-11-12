"""Bitwise-equal fast replacements for pinned upstream lnL_*_p.

Every model flux comes from pytransit's own compiled quadratic_model_pv, called
with upstream's argument types (F-ordered float64 rows, at least two per call),
so it is the same machine code: pytransit compiles its point model with
fastmath, and a re-inlined copy can round differently. Around it:
- eclipsing binaries compute upstream's 25-point secondary first and model the
  primary only for rows that pass the 1.5 sigma veto; twins skip the secondary;
- rows go through in chunks, so no (draws, points) array is built;
- dilution and the cost (upstream's chi-square, or post_boi's SourceLikelihood)
  run in a strict numba kernel that reproduces numpy's pairwise row sum.
For joint.pruned it also proves a row's model finite (finite_row) and bounds a
row's cost without pytransit (cheap_row, certified mode only).
"""

import functools
import hashlib
import importlib.metadata
import inspect
import itertools

import numpy as np
from numba import njit, parallel_chunksize, prange
from triceratops import likelihoods as upstream  # installs pytransit's numpy shims

# isort: split
from pytransit import QuadraticModel
from pytransit.orbits.taylor_z import vajs_from_paiew

VERSIONS = {"numpy": "1.26.4", "PyTransit": "2.2.0"}  # numpy_sum copies numpy 1.26's sum
PINNED = {  # upstream sources whose preparation prepare_tp/prepare_eb copy
    "simulate_TP_transit_p": "3eaef7391d27ebf2c2801f1e98959488de6694b45aa32bc01a24d7ada4e6bfb7",
    "simulate_EB_transit_p": "8ef344a7e07ebb965aa2ec68161a3c6c5d5cec1c1a39914290cd8dfc55305b06",
    "lnL_TP_p": "071effb291e91add9190cf171c148658418f3999fc155598ed0d9322447097d3",
    "lnL_EB_p": "43a716a011eeb7799c2f54d7e8e4881bf590de7c2f0dff9eb22b2724ff7e876e",
    "lnL_EB_twin_p": "582d443361a13dfa1046cc5e72e79ea171b7028158a7082697869582f9c16744",
}


def require_pinned(module, pinned):
    """Fail unless each named function's source is the pinned one."""
    for name, checksum in pinned.items():
        source = inspect.getsource(getattr(module, name))
        if hashlib.sha256(source.encode()).hexdigest() != checksum:
            raise RuntimeError(f"{module.__name__}.{name} differs from pinned TRICERATOPS d20875c")


for package, version in VERSIONS.items():
    if importlib.metadata.version(package) != version:
        raise RuntimeError(f"{package} {importlib.metadata.version(package)} is not {version}")
require_pinned(upstream, PINNED)

Rearth, Rsun, pi = upstream.Rearth, upstream.Rsun, upstream.pi
tm, tm_sec = QuadraticModel(interpolate=False), QuadraticModel(interpolate=False)
SECONDARY_TIME = np.linspace(-0.05, 0.05, 25)
NUMPY_BUFFER = 8192  # np.sum adds one pairwise sum per 8192-element buffer chunk
CHUNK = 32  # rows per dynamically scheduled prange chunk
CELLS = 2**21  # model values per pytransit call (16 MiB)
NONE = np.empty(0)  # placeholder for the unused cost arrays


@njit(cache=True)
def pairwise_sum(a, lo, n):
    """numpy 1.26 DOUBLE_pairwise_sum: 8 accumulators, blocks of at most 128."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        while i < n - n % 8:
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return pairwise_sum(a, lo, n2) + pairwise_sum(a, lo + n2, n - n2)


@njit(cache=True)
def numpy_sum(a):
    """np.sum(a) for a contiguous float64 row: 0.0 plus one pairwise sum per buffer."""
    res = 0.0
    for lo in range(0, a.size, NUMPY_BUFFER):
        res += pairwise_sum(a, lo, min(NUMPY_BUFFER, a.size - lo))
    return res


@njit(parallel=True, cache=True)
def cost_kernel(model, flux, sigma2, centre, precision, residual, joint, eb, dilute, binary):
    """Per model row: upstream's 0.5 * chi-square, or with joint SourceLikelihood.cost."""
    out = np.empty(model.shape[0])
    for i in prange(model.shape[0]):
        terms = np.empty(model.shape[1])
        for j in range(model.shape[1]):
            f = model[i, j]
            if binary:
                f = (f + eb[i]) / (1 + eb[i])
            f = (f + dilute[i]) / (1 + dilute[i])
            if joint:  # quarters.py SourceLikelihood.cost
                d = f - 1.0 - centre[j]
                terms[j] = precision[j] * (d * d)
            else:  # likelihoods.py:496
                d = flux[j] - f
                terms[j] = d * d / sigma2
        out[i] = 0.5 * (residual + numpy_sum(terms)) if joint else 0.5 * numpy_sum(terms)
    return out


def prepare_tp(R_p, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
               companion_is_host=False, exptime=0.00139, nsamples=20):
    """Rows from simulate_TP_transit_p's arguments (likelihoods.py:340-358, unchanged)."""
    F_target = 1
    F_comp = companion_fluxratio/(1-companion_fluxratio)
    k = R_p*Rearth/(R_s*Rsun)
    P_orb = np.full_like(k, P_orb)
    a = a/(R_s*Rsun)
    inc *= (pi/180.)
    w = (90-argp)*(pi/180.)
    F_dilute = F_target / F_comp if companion_is_host else F_comp / F_target
    return {"k": k, "P_orb": P_orb, "a": a, "inc": inc, "ecc": ecc, "w": w, "u1": u1, "u2": u2,
            "eb": NONE, "dilute": F_dilute, "binary": False, "secondary": None,
            "exptime": exptime, "nsamples": nsamples}


def prepare_eb(R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
               companion_is_host=False, exptime=0.00139, nsamples=20, twin=False):
    """Rows from simulate_EB_transit_p's arguments (likelihoods.py:400-449, unchanged except
    that (n, 1) shapes are (n,)); twins never use their secondary."""
    F_target = 1
    F_comp = companion_fluxratio/(1 - companion_fluxratio)
    F_EB = EB_fluxratio/(1 - EB_fluxratio)
    k = R_EB/R_s
    k[(k - 1.0) < 1e-6] *= 0.999
    P_orb = np.full_like(k, P_orb)
    a = a/(R_s*Rsun)
    inc *= (pi/180.)
    w = (90-argp)*(pi/180.)
    ks = R_s/R_EB
    ks[(ks - 1.0) < 1e-6] *= 0.999
    ws = (90-argp+180)*(pi/180.)
    if companion_is_host:
        eb, eb_sec = F_EB/F_comp, F_comp/F_EB
        F_dilute = F_target/(F_comp + F_EB)
    else:
        eb, eb_sec = F_EB/F_target, F_target/F_EB
        F_dilute = F_comp/(F_target + F_EB)
    rows = {"k": k, "P_orb": P_orb, "a": a, "inc": inc, "ecc": ecc, "w": w, "u1": u1, "u2": u2,
            "eb": eb, "dilute": F_dilute, "binary": True, "secondary": None,
            "exptime": exptime, "nsamples": nsamples}
    if not twin:
        # likelihoods.py:426-449: the diluted depth of the 25-point secondary eclipse
        t0 = np.full_like(ks, 0.)
        tm_sec.set_data(SECONDARY_TIME)
        sec_flux = tm_sec.evaluate_pv(pvp=np.array([ks, t0, P_orb, a, inc, ecc, ws]).T,
                                      ldc=np.array([u1, u2]).T)
        sec_flux = np.min(sec_flux, axis=1)
        sec_flux = (sec_flux + eb_sec)/(1 + eb_sec)
        rows["secondary"] = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    return rows


PREPARE = {"lnL_TP_p": prepare_tp, "lnL_EB_p": prepare_eb,
           "lnL_EB_twin_p": functools.partial(prepare_eb, twin=True)}


def standard(name, time, flux, sigma, args, kwargs):
    """calc_probs' inputs: float64 rows of one length >= 2, a float64 grid of >= 2 points
    matching flux, a scalar sigma. Anything else goes to upstream, which raises or
    broadcasts as it always has."""
    try:
        bound = inspect.signature(PREPARE[name]).bind(*args, **kwargs).arguments
    except TypeError:
        return False
    rows = [v for key, v in bound.items()
            if key not in ("companion_is_host", "exptime", "nsamples", "twin")]
    n, npt = np.shape(rows[0])[0] if np.ndim(rows[0]) == 1 else 0, np.size(time)

    def float64(x, size):
        return isinstance(x, np.ndarray) and x.dtype == np.float64 and x.shape == (size,)

    return (n >= 2 and npt >= 2 and np.ndim(sigma) == 0 and float64(time, npt)
            and float64(flux, npt) and all(float64(v, n) for v in rows))


def chunks(rows, size):
    """Consecutive pieces of the row indices, none of exactly one row unless rows has one."""
    edges = [*range(0, rows.size, size), rows.size]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        del edges[-2]
    return [rows[lo:hi] for lo, hi in itertools.pairwise(edges)]


def cost_inputs(flux, sigma, source):
    """cost_kernel's data arguments: upstream's chi-square, or a SourceLikelihood's cost."""
    if source is None:
        return flux, sigma**2, NONE, NONE, 0.0, False
    return (NONE, 1.0, np.asarray(source.centre, np.float64),
            np.asarray(source.precision, np.float64), float(source.residual), True)


def screened(r, sigma):
    """Rows whose primary counts: all, or the binaries passing upstream's 1.5 sigma veto
    (the others are +inf whatever the primary)."""
    if r["secondary"] is None:
        return np.arange(r["k"].size)
    return np.flatnonzero(r["secondary"] < 1.5*sigma)


def costs(r, rows, time, cost, points=None):
    """The cost of each of rows (indices into r); with points, over those grid points only.

    pytransit models every (row, time) independently, so a subset of rows or times gives
    the same bits as the whole grid."""
    flux, sigma2, centre, precision, residual, joint = cost
    if points is not None:
        time = np.ascontiguousarray(time[points])
        flux = flux[points] if flux.size else flux
        centre, precision = (centre[points], precision[points]) if joint else (centre, precision)
    eb, dilute = np.asarray(r["eb"], np.float64), np.asarray(r["dilute"], np.float64)
    out = np.empty(rows.size)
    tm.set_data(time, exptimes=r["exptime"], nsamples=r["nsamples"])
    for chunk in chunks(np.arange(rows.size), max(2, CELLS // time.size)):
        # A one-row call would type as C-ordered and compile different code; repeat the row.
        take = rows[np.resize(chunk, max(2, chunk.size))]
        pvp = np.array([r["k"][take], np.full_like(r["k"][take], 0.), r["P_orb"][take], r["a"][take],
                        r["inc"][take], r["ecc"][take], r["w"][take]]).T
        ldc = np.array([r["u1"][take], r["u2"][take]]).T
        with parallel_chunksize(CHUNK):  # dynamic scheduling; rows are independent
            model = tm.evaluate_pv(pvp=pvp, ldc=ldc).reshape(take.size, time.size)
            values = cost_kernel(model, flux, sigma2, centre, precision, residual, joint,
                                 eb[take] if r["binary"] else NONE, dilute[take], r["binary"])
        out[chunk] = values[:chunk.size]
    return out


def evaluate(name, time, flux, sigma, args, kwargs, source=None):
    """lnL_*_p's result; with a SourceLikelihood, its cost instead."""
    r = PREPARE[name](*args, **kwargs)
    out = np.full(r["k"].size, np.inf)
    rows = screened(r, sigma)
    out[rows] = costs(r, rows, time, cost_inputs(flux, sigma, source))
    return out


EDGE = 1e-7  # stellar radii: margin around each z where eval_quad_z_s can turn non-finite


@njit(cache=True)
def clearance(z, k):
    """Distance from z to the nearest z where pytransit's point model may be non-finite
    (negative inside; see unsafe)."""
    return min(abs(z - (1.0 + k)), abs(z - (1.0 - k)), z - 1e-6) - EDGE


@njit(cache=True)
def unsafe(lo, hi, k):
    """Whether [lo, hi] meets a z at which pytransit's point model may be non-finite, for
    1e-6 <= k <= 1 - 1e-6 at least 2e-4 from its 0.5 snap. Elsewhere every square root,
    arccos and elliptic-integral argument of eval_quad_z_s stays inside its domain by much
    more than any rounding. The dangers are the contacts z = 1 +- k and z near 0 (where
    (k+z)^2 - (k-z)^2 can round negative). Near z = k there is none: after the 1e-6 shift,
    a z close enough to make (k-z)^2 tiny takes the edge-at-origin branch, whose elliptic
    moduli 0.5/k (z > 0.5) or 2k (z < 0.5) stay below 1 away from the snap."""
    return ((lo <= 1.0 + k + EDGE and hi >= 1.0 + k - EDGE)
            or (lo <= 1.0 - k + EDGE and hi >= 1.0 - k - EDGE)
            or lo <= 1e-6)


@njit(cache=True)
def provable(k, p, a, i, e, w, u1, u2, eb, dilute, tmax):
    """Rows finite_row can reason about: finite inputs, 1e-6 <= k <= 1 - 1e-6 at least 2e-4
    from pytransit's 0.5 snap, a well-converged Kepler solve (e < 0.95), limb darkening with
    1 - u1/3 - u2/6 >= 0.05 (pytransit divides by it), and every grid point at epoch 0."""
    for v in (k, p, a, i, e, w, u1, u2, eb, dilute):
        if not np.isfinite(v):
            return False
    return (1e-6 <= k <= 1.0 - 1e-6 and abs(k - 0.5) >= 2e-4 and 0.0 <= e < 0.95
            and 1.0 - u1 / 3.0 - u2 / 6.0 >= 0.05 and eb >= 0.0 and dilute >= 0.0 and a > 1.0
            and tmax < 0.5 * p * (1.0 - 1e-9))


@njit(cache=True)
def taylor_z(t, c):
    """pytransit's z_taylor_st in Horner form, c = (y0, vx, ax/2, jx/6, sx/24, vy, ay/2,
    jy/6, sy/24); equal to it within rounding (~1e-15), far below EDGE."""
    px = t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])))
    py = c[0] + t * (c[5] + t * (c[6] + t * (c[7] + t * c[8])))
    return np.sqrt(px * px + py * py)


@njit(cache=True)
def band_free(c, k, base, dt, start, stop):
    """No sample t = base + dt (s + 0.5), start <= s < stop, has z within EDGE of an unsafe
    value. Branch-free so that it vectorizes."""
    bad = False
    for s in range(start, stop):
        bad |= clearance(taylor_z(base + dt * (s + 0.5), c), k) <= 0.0
    return not bad


@njit(cache=True)
def samples_clear(tc, z, c, k, exptime, nsamples, slope):
    """No exposure sample of the point at tc (centre z) comes within EDGE of an unsafe z.
    Samples nearer tc than clearance(z) / slope in time cannot reach a band and are skipped;
    one extra sample on each side covers rounding of the sample index."""
    dt = exptime / nsamples
    skip = max(0.0, clearance(z, k)) / slope
    first = min(nsamples, max(0, int(np.floor((0.5 * exptime - skip) / dt - 0.5)) + 2))
    last = max(first, min(nsamples, int(np.ceil((0.5 * exptime + skip) / dt - 0.5)) - 1))
    base = tc - 0.5 * exptime
    return band_free(c, k, base, dt, 0, first) and band_free(c, k, base, dt, last, nsamples)


@njit(cache=True)
def finite_row(k, p, a, i, e, w, u1, u2, eb, dilute, time, exptime, nsamples):
    """True only if pytransit's model of this row is finite at every grid point (False means
    no proof, not a failure). Replays quadratic_model_pv's window and sample times; its z
    can differ from this one by rounding (~1e-12), far below EDGE."""
    tmax = np.max(np.abs(time))
    if not provable(k, p, a, i, e, w, u1, u2, eb, dilute, tmax):
        return False
    orbit = vajs_from_paiew(0.0, p, a, i, e, w)[1:]
    for v in orbit:
        if not np.isfinite(v):
            return False
    y0, vx, vy, ax, ay, jx, jy, sx, sy = orbit
    if vx == 0.0:
        return False
    window = max(0.125, (2.0 + k) / vx) * (1.0 + 1e-9) + 1e-12
    t = tmax + 0.5 * exptime
    # |dz/dt| <= |px'| + |py'| over |t| <= tmax + exptime/2, where every sample lies
    slope = (abs(vx) + abs(ax) * t + abs(jx) * t * t / 2 + abs(sx) * t**3 / 6
             + abs(vy) + abs(ay) * t + abs(jy) * t * t / 2 + abs(sy) * t**3 / 6)
    slope = slope * (1.0 + 1e-9) + 1e-12
    reach = 0.5 * exptime * slope
    c = (y0, vx, ax / 2, jx / 6, sx / 24, vy, ay / 2, jy / 6, sy / 24)
    for j in range(time.size):
        tc = time[j]
        if abs(tc) > window:  # pytransit sets exactly 1 without sampling
            continue
        z = taylor_z(tc, c)
        if z - reach > 1.0 + k + EDGE or not unsafe(z - reach, z + reach, k):
            continue
        if not samples_clear(tc, z, c, k, exptime, nsamples, slope):
            return False
    return True


@njit(parallel=True, cache=True)
def finite_kernel(k, p, a, i, e, w, u1, u2, eb, dilute, time, exptime, nsamples):
    out = np.empty(k.size, np.bool_)
    for n in prange(k.size):
        out[n] = finite_row(k[n], p[n], a[n], i[n], e[n], w[n], u1[n], u2[n], eb[n], dilute[n],
                            time, exptime, nsamples)
    return out


def columns(r, rows):
    """The per-row inputs of finite_row and cheap_row for rows (indices into r)."""
    def column(key):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(r[key], np.float64), r["k"].shape)[rows])

    eb = column("eb") if r["binary"] else np.zeros(rows.size)
    return (column("k"), column("P_orb"), column("a"), column("inc"), column("ecc"), column("w"),
            column("u1"), column("u2"), eb, column("dilute"))


def finite_rows(r, rows, time):
    """finite_row for each of rows (indices into r)."""
    with parallel_chunksize(CHUNK):
        return finite_kernel(*columns(r, rows), time, float(r["exptime"]), int(r["nsamples"]))


# pytransit's point model against the exact occultation (scratch/bound/eta.py, 33,000 random and
# near-edge cases): at most 6.4e-8 apart outside its two special bands, 1.3e-4 inside them.
ETA, ETA_BAND = 2e-7, 5e-4
GROUPS = 2  # exposure sample groups per grid point in cheap_row


@njit(cache=True)
def overlap(z, k):
    """Uniform-disc occulted fraction of the star, 0 < k < 1."""
    if z >= 1.0 + k:
        return 0.0
    if z <= 1.0 - k:
        return k * k
    k0 = np.arccos(min(1.0, max(-1.0, (k * k + z * z - 1.0) / (2.0 * k * z))))
    k1 = np.arccos(min(1.0, max(-1.0, (1.0 - k * k + z * z) / (2.0 * z))))
    return (k * k * k0 + k1 - 0.5 * np.sqrt(max(0.0, 4.0 * z * z - (1.0 + z * z - k * k) ** 2))) / np.pi


@njit(cache=True)
def intensity(r, u1, u2):
    mu = np.sqrt(max(0.0, 1.0 - r * r))
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


@njit(cache=True)
def banded(lo, hi, k):
    """Whether [lo, hi] meets pytransit's edge-at-origin (|z-k| < 1e-4 (z+k)) or contact
    (|z+k-1| < 1e-4) special formula."""
    w = 1.2e-4 * (hi + k) + 1e-9
    return (lo <= k + w and hi >= k - w) or (lo <= 1.0 - k + 1.2e-4 and hi >= 1.0 - k - 1.2e-4)


@njit(cache=True)
def deficit(tc, c, k, u1, u2, omega, slope, exptime, nsamples):
    """Bounds (lo, hi) on pytransit's mean occulted flux over the exposure samples of the
    point at tc, inside its window. Over each group of samples z stays within slope *
    half-width of the group's centre; the exact occulted flux is non-increasing in z and lies
    between the uniform-disc overlap times the limb-darkened intensity at the overlap's outer
    and inner radius; pytransit's model is within ETA of it (ETA_BAND in its special bands),
    and exactly 1 wherever z > 1 + k."""
    dt = exptime / nsamples
    lo = hi = 0.0
    for g in range(GROUPS):
        s0, s1 = g * nsamples // GROUPS, (g + 1) * nsamples // GROUPS
        if s1 <= s0:
            continue
        t0 = tc - 0.5 * exptime + dt * (s0 + 0.5)
        t1 = tc - 0.5 * exptime + dt * (s1 - 0.5)
        reach = slope * 0.5 * (t1 - t0) + 1e-12
        z = taylor_z(0.5 * (t0 + t1), c)
        zlo, zhi = max(0.0, z - reach), z + reach
        if zlo > 1.0 + k + EDGE:
            continue
        eta = ETA_BAND if banded(zlo, zhi, k) else ETA
        lo += (s1 - s0) * max(0.0, overlap(zhi, k) * intensity(min(1.0, zhi + k), u1, u2) / omega - eta)
        hi += (s1 - s0) * (overlap(zlo, k) * intensity(max(0.0, zlo - k), u1, u2) / omega + eta)
    return lo / nsamples, hi / nsamples


@njit(cache=True)
def cheap_row(k, p, a, i, e, w, u1, u2, eb, dilute, binary, time, centre, precision, residual,
              exptime, nsamples):
    """A lower bound on this row's SourceLikelihood cost without pytransit, and an estimate of
    it from the bound intervals' midpoints: (cheap, guess); NaN where no bound applies.

    Each point's diluted model lies in the interval that deficit() implies (limb darkening
    falling outward: u1 >= 0, u1 + 2 u2 >= 0, 1 - u1 - u2 >= 0), so its term is at least
    precision times the squared distance from 1 + centre to that interval. The bound holds
    only while pytransit honours ETA."""
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    tmax = np.max(np.abs(time))
    if (not provable(k, p, a, i, e, w, u1, u2, eb, dilute, tmax) or u1 < 0.0
            or u1 + 2.0 * u2 < 0.0 or 1.0 - u1 - u2 < 0.0):
        return np.nan, np.nan
    y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(0.0, p, a, i, e, w)[1:]
    c = (y0, vx, ax / 2, jx / 6, sx / 24, vy, ay / 2, jy / 6, sy / 24)
    for v in c:
        if not np.isfinite(v):
            return np.nan, np.nan
    window = max(0.125, (2.0 + k) / vx)
    t = tmax + 0.5 * exptime
    slope = (abs(vx) + abs(ax) * t + abs(jx) * t * t / 2 + abs(sx) * t**3 / 6
             + abs(vy) + abs(ay) * t + abs(jy) * t * t / 2 + abs(sy) * t**3 / 6) * (1 + 1e-9) + 1e-12
    total = middle = 0.0
    for j in range(time.size):
        lo = hi = 0.0
        # beyond the window, or with every sample at z > 1 + k, pytransit returns exactly 1
        if (abs(time[j]) <= window * (1 + 1e-9) + 1e-12
                and taylor_z(time[j], c) - 0.5 * exptime * slope <= 1.0 + k + EDGE):
            lo, hi = deficit(time[j], c, k, u1, u2, omega, slope, exptime, nsamples)
            if abs(abs(time[j]) - window) <= 1e-9 * window + 1e-12:
                lo = 0.0  # pytransit may skip this point and return exactly 1
        mlo, mhi = 1.0 - hi - 1e-15, 1.0 - lo + 1e-15
        if binary:
            mlo, mhi = (mlo + eb) / (1.0 + eb), (mhi + eb) / (1.0 + eb)
        flo, fhi = (mlo + dilute) / (1.0 + dilute), (mhi + dilute) / (1.0 + dilute)
        gap = max(0.0, flo - 1.0 - centre[j], 1.0 + centre[j] - fhi)
        total += precision[j] * gap * gap
        d = 0.5 * (flo + fhi) - 1.0 - centre[j]
        middle += precision[j] * d * d
    return 0.5 * (residual + total) * (1 - 1e-12), 0.5 * (residual + middle)


@njit(parallel=True, cache=True)
def cheap_kernel(k, p, a, i, e, w, u1, u2, eb, dilute, binary, time, centre, precision,
                 residual, exptime, nsamples):
    cheap, guess = np.empty(k.size), np.empty(k.size)
    for n in prange(k.size):
        cheap[n], guess[n] = cheap_row(k[n], p[n], a[n], i[n], e[n], w[n], u1[n], u2[n], eb[n],
                                       dilute[n], binary, time, centre, precision, residual,
                                       exptime, nsamples)
    return cheap, guess


def cheap_bounds(r, rows, time, cost):
    """cheap_row for each of rows (indices into r) against a SourceLikelihood's cost inputs:
    bounds and guesses, NaN where no bound applies."""
    with parallel_chunksize(CHUNK):
        return cheap_kernel(*columns(r, rows), r["binary"], time, cost[2], cost[3],
                            float(cost[4]), float(r["exptime"]), int(r["nsamples"]))


def fused(name):
    """Drop-in for likelihoods.<name>; non-standard inputs go to upstream."""
    original = getattr(upstream, name)

    def likelihood(time, flux, sigma, *args, **kwargs):
        if not standard(name, time, flux, sigma, args, kwargs):
            return original(time, flux, sigma, *args, **kwargs)
        return evaluate(name, time, flux, sigma, args, kwargs)

    likelihood.__name__ = likelihood.__qualname__ = name
    return likelihood


REPLACEMENTS = {name: fused(name) for name in PREPARE}
lnL_TP_p, lnL_EB_p, lnL_EB_twin_p = (REPLACEMENTS[name] for name in PREPARE)
