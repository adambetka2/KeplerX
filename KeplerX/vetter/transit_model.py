"""batman.TransitModel with the accurate limb-darkening integral."""

import copy
import math

import batman
import numpy as np
from batman import _nonlinear_ld, _rsky

from vetter import kernel_bindings

MARGIN = 1e-6
ANGLE_MARGIN = 1e-9
PHASE_MARGIN = 1e-12
# The runs reach this much further in phase than the in-transit limit.
RUN_MARGIN = 1e-9
TWO_PI = 2.0 * math.pi

# Chebyshev interpolants where the flux is analytic in d: within 3e-13 of the integral.
INTERPOLATE = True
INTERPOLATE_POINTS = 2048
# Sized as before runs existed: every sample when the in-transit window exceeds this share.
ROUTE_FRACTION = 0.08


def occulted(ds, params, model, interpolate):
    """Flux at sky separations ds: batman's limb-darkening integral, accurately."""
    return kernel_bindings.nonlinear_ld(
        ds, params.rp, *params.u, interpolate=INTERPOLATE and interpolate
    )


def batman_rule(ds, params, model, interpolate):
    """batman's own midpoint rule at the model's step."""
    return _nonlinear_ld._nonlinear_ld(
        ds, abs(params.rp), *params.u, model.fac, model.nthreads
    )


class TransitModel:
    """A batman.TransitModel for primary transits with nonlinear limb darkening."""

    def __init__(self, params, t, **kwargs):
        # batman's integration step is derived when first read, from the parameters as they were here.
        self.pending = None
        if kwargs.get("fac") is None:
            self.pending = (copy.deepcopy(params), t, dict(kwargs))
            kwargs = dict(kwargs, fac=1.0)
        self.model = batman.TransitModel(params, t, **kwargs)
        self.ts = np.ascontiguousarray(self.model.t_supersample, dtype=np.float64)
        t = np.ascontiguousarray(self.model.t, dtype=np.float64)
        self.factor = max(int(self.model.supersample_factor), 1)
        self.sorted = None
        if not len(t) or len(self.ts) != len(t) * self.factor:
            return
        if not (np.isfinite(t).all() and np.isfinite(self.ts).all()):
            return
        # Largest distance of a sample from its cadence's time.
        self.spread = float(
            np.max(np.abs(self.ts.reshape(-1, self.factor) - t[:, None]))
        )
        if np.all(t[1:] >= t[:-1]):
            self.order, self.sorted = None, t
        else:
            self.order = np.argsort(t, kind="stable")
            self.sorted = t[self.order]
        start, stop = self.sorted[0], self.sorted[-1]
        self.start = start
        self.width = (stop - start) / len(t) if stop > start else 1.0
        self.bucket = np.searchsorted(
            self.sorted, start + np.arange(len(t) + 2) * self.width, side="left"
        )
        self.crowd = int(np.diff(self.bucket).max(initial=0))

    @property
    def fac(self):
        if self.pending is not None:
            params, t, kwargs = self.pending
            self.model.fac = batman.TransitModel(params, t, **kwargs).fac
            self.pending = None
        return self.model.fac

    @property
    def nthreads(self):
        return self.model.nthreads

    @property
    def t_supersample(self):
        return self.model.t_supersample

    def _supported(self, params):
        model = self.model
        return (
            self.sorted is not None
            and model.transittype == 1
            and params.limb_dark == "nonlinear"
            and model.limb_dark == "nonlinear"
            and len(params.u) == 4
            and params.ecc == 0.0
            and params.w == 90.0
            and params.rp >= 0.0
            and params.a > 0.0
            and params.per > 0.0
            and 0.0 <= params.inc <= 180.0
            and np.isfinite([params.t0, params.per, params.a, params.inc]).all()
        )

    @staticmethod
    def _limit(params):
        """Phase fraction within which a point may be in transit."""
        ratio = (1.0 + params.rp + MARGIN) / params.a
        angle = math.pi / 2.0 if ratio >= 1.0 else math.asin(ratio)
        return (angle + ANGLE_MARGIN) / TWO_PI + PHASE_MARGIN

    def _slots(self, times):
        """Bucket indices of `times`, clipped to the table."""
        slot = np.floor((times - self.start) / self.width)
        return np.clip(slot, 0, len(self.bucket) - 2).astype(np.intp)

    def _runs(self, params, limit):
        """Increasing, disjoint positions in the sorted cadences of every cadence with a sample within `limit`."""
        t, t0, per = self.sorted, params.t0, params.per
        reach = (limit + RUN_MARGIN) * per + self.spread
        first = math.floor((t[0] - t0 - reach) / per)
        last = math.ceil((t[-1] - t0 + reach) / per)
        if last - first >= len(t):
            # More orbits than cadences: one pass over the cadences is cheaper.
            q = (t - t0) / per
            fraction = np.abs(q - np.trunc(q))
            wide = limit + RUN_MARGIN + self.spread / per
            near = (fraction < wide) | (fraction > 1.0 - wide)
            return np.flatnonzero(near), last - first + 1
        centres = t0 + np.arange(first, last + 1, dtype=np.float64) * per
        # Slot j's first cadence is at or before any time in it; the 1e-9 margin covers rounding.
        lo = self.bucket[self._slots(centres - reach)]
        hi = self.bucket[self._slots(centres + reach) + 1]
        lo = np.maximum(lo, np.concatenate(([0], hi[:-1])))
        length = hi - lo
        keep = length > 0
        lo, length = lo[keep], length[keep]
        if not len(lo):
            return np.zeros(0, dtype=np.intp), len(centres)
        shift = np.repeat(lo - (np.cumsum(length) - length), length)
        return np.arange(int(length.sum()), dtype=np.intp) + shift, len(centres)

    def _routed_size(self, params, limit, samples, orbits):
        """The point count the integral was passed before runs, which chooses interpolation."""
        if 2.0 * limit > ROUTE_FRACTION:
            return len(self.ts)
        n = len(samples)
        if n < INTERPOLATE_POINTS:
            return n
        if n >= INTERPOLATE_POINTS + 2 * orbits * self.factor * (self.crowd + 2):
            return n
        q = (self.ts[samples] - params.t0) / params.per
        fraction = np.abs(q - np.trunc(q))
        return int(np.count_nonzero((fraction < limit) | (fraction > 1.0 - limit)))

    def light_curve(self, params):
        if not self._supported(params):
            self.model.fac = self.fac
            return self.model.light_curve(params)
        model, factor = self.model, self.factor
        limit = self._limit(params)
        runs, orbits = self._runs(params, limit)
        cadences = runs if self.order is None else self.order[runs]
        if factor == 1:
            samples = cadences
        else:
            samples = (cadences[:, None] * factor + np.arange(factor)).ravel()
        interpolate = (
            self._routed_size(params, limit, samples, orbits) >= INTERPOLATE_POINTS
        )
        lc = np.ones(len(self.sorted))
        if not len(samples):
            return lc
        ds = _rsky._rsky(
            self.ts[samples], params.t0, params.per, params.a,
            params.inc * math.pi / 180.0, params.ecc, params.w * math.pi / 180.0,
            model.transittype, model.nthreads,
        )  # fmt: skip
        flux = occulted(ds, params, self, interpolate)
        if factor == 1:
            lc[cadences] = flux
        else:
            lc[cadences] = np.mean(flux.reshape(-1, factor), axis=1)
        return lc
