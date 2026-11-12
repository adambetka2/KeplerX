"""juliet's native transit likelihood and prior transform for one candidate."""

import math

import numpy as np
from scipy import special
from scipy.stats._continuous_distns import _log_gauss_mass, _norm_logcdf

from validator import kernel_bindings

INSTRUMENT = "Kepler"
# Order of the kernel's parameter vector.
KERNEL_ORDER = (
    "P_p1", "t0_p1", "p_p1", "b_p1", "rho", f"q1_{INSTRUMENT}", f"q2_{INSTRUMENT}",
    "ecc_p1", "omega_p1", f"mflux_{INSTRUMENT}", f"sigma_w_{INSTRUMENT}",
    f"mdilution_{INSTRUMENT}",
)  # fmt: skip
DISTRIBUTIONS = ("uniform", "loguniform", "normal", "truncatednormal", "beta", "fixed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_priors(priors):
    require(
        set(priors) == set(KERNEL_ORDER), "Validator fits juliet's one-planet model"
    )
    for name, prior in priors.items():
        require(
            prior["distribution"].lower() in DISTRIBUTIONS,
            f"Unsupported prior {prior['distribution']} for {name}",
        )


def free_parameters(priors):
    """juliet's model_parameters order, fixed parameters left out."""
    return [n for n, p in priors.items() if p["distribution"].lower() != "fixed"]


class PriorTransform:
    """juliet.utils transforms with scipy's own ppf arithmetic, per scalar."""

    def __init__(self, priors):
        check_priors(priors)
        self.names = free_parameters(priors)
        self.specs = [
            (priors[n]["distribution"].lower(), priors[n]["hyperparameters"])
            for n in self.names
        ]
        self._compile()

    def __getstate__(self):
        return dict(names=self.names, specs=self.specs)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._compile()

    def _compile(self):
        self.steps = [self._step(kind, h) for kind, h in self.specs]

    @staticmethod
    def _step(kind, h):
        if kind == "uniform":
            a, b = h
            return lambda x: a + (b - a) * x
        if kind == "loguniform":
            la, lb = np.log(h[0]), np.log(h[1])
            return lambda x: np.exp(la + x * (lb - la))
        if kind == "normal":
            mu, sigma = h
            return lambda x: special.ndtri(x) * sigma + mu
        if kind == "beta":
            a, b = h
            return lambda x: special.betaincinv(a, b, x) * 1.0 + 0.0
        mu, sigma, lo, hi = h
        ar, br = np.asarray([(lo - mu) / sigma]), np.asarray([(hi - mu) / sigma])
        mass = float(_log_gauss_mass(ar, br)[0])
        if ar[0] < 0:
            edge = float(_norm_logcdf(ar)[0])
            return lambda x: _ndtri_log_sum(edge, math.log(x) + mass) * sigma + mu
        edge = float(_norm_logcdf(-br)[0])
        return lambda x: -_ndtri_log_sum(edge, math.log1p(-x) + mass) * sigma + mu

    def __call__(self, cube):
        out = np.empty(len(self.steps))
        for i, step in enumerate(self.steps):
            out[i] = step(cube[i])
        return out


def _ndtri_log_sum(x, y):
    """ndtri_exp(logsumexp([x, y])) as scipy 1.13 computes it for two values."""
    top = max(x, y)
    if not math.isfinite(top):
        top = 0.0
    total = math.exp(x - top) + math.exp(y - top)
    return special.ndtri_exp(math.log(total) + top)


class NativeLikelihood:
    """Callable log-likelihood of juliet's model on fixed photons; picklable."""

    def __init__(self, priors, time, flux, flux_err, supersample, exposure):
        check_priors(priors)
        time, flux, flux_err = (
            np.ascontiguousarray(v, float) for v in (time, flux, flux_err)
        )
        require(time.ndim == 1 and time.size > 0, "Supply native photons")
        require(np.all(np.diff(time) > 0), "Native photon times must increase")
        require(np.isfinite([time, flux, flux_err]).all(), "Non-finite native photons")
        require(type(supersample) is int and supersample >= 1, "Invalid supersampling")
        self.names = free_parameters(priors)
        self.index = np.array(
            [self.names.index(n) if n in self.names else -1 for n in KERNEL_ORDER]
        )
        self.fixed = np.array(
            [
                float(priors[n]["hyperparameters"]) if i < 0 else np.nan
                for n, i in zip(KERNEL_ORDER, self.index)
            ]
        )
        self.time, self.flux, self.variance = time, flux, flux_err**2
        self.supersample = supersample
        self.exposure = float(exposure) if supersample > 1 else 0.0
        if supersample > 1:
            self.offsets = np.linspace(
                -self.exposure / 2.0, self.exposure / 2.0, supersample
            )
        else:
            self.offsets = np.zeros(1)
        self.half_span = float(np.max(np.abs(self.offsets)))
        self._buffers()

    def _buffers(self):
        self._lib = kernel_bindings.library()
        self._model = np.empty_like(self.time)
        self._terms = np.empty_like(self.time)
        self._count = np.zeros(1, dtype=np.int64)
        self._theta = self.fixed.copy()
        self._free = np.flatnonzero(self.index >= 0)
        self._source = self.index[self._free]
        k = len(self._free)
        self._prefix = (
            k
            if np.array_equal(self._free, np.arange(k))
            and np.array_equal(self._source, np.arange(k))
            else None
        )
        address = [
            a.ctypes.data for a in (self.time, self.flux, self.variance, self.offsets)
        ]
        self._args = (
            self.time.size, *address[:3], self.supersample, address[3], self.half_span,
            self._theta.ctypes.data, self._model.ctypes.data, self._terms.ctypes.data,
            self._count.ctypes.data,
        )  # fmt: skip

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in ("_lib", "_model", "_terms", "_count", "_theta", "_free", "_source",
                    "_prefix", "_args"):  # fmt: skip
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._buffers()

    def kernel_vector(self, values):
        theta = self.fixed.copy()
        free = self.index >= 0
        theta[free] = np.asarray(values, float)[self.index[free]]
        return theta

    def kernel_rows(self, values):
        values = np.atleast_2d(np.asarray(values, float))
        rows = np.repeat(self.fixed[None, :], len(values), axis=0)
        free = self.index >= 0
        rows[:, free] = values[:, self.index[free]]
        return np.ascontiguousarray(rows)

    def __call__(self, values):
        if self._prefix is not None:
            self._theta[: self._prefix] = values
        else:
            self._theta[self._free] = np.asarray(values)[self._source]
        return self._lib.validator_loglike_address(*self._args)

    def rows(self, values):
        """Log-likelihoods of many parameter vectors, and photons modelled for each."""
        theta = self.kernel_rows(values)
        out = np.empty(len(theta))
        counts = np.zeros(len(theta), dtype=np.int64)
        self._lib.validator_loglike_rows(
            len(theta), theta, self.time.size, self.time, self.flux, self.variance,
            self.supersample, self.offsets, self.half_span, self._model, self._terms,
            out, counts,
        )  # fmt: skip
        return out, counts

    def model(self, values, everywhere=False):
        """Exposure-averaged transit model (before dilution and mean flux)."""
        theta = self.kernel_vector(values)
        out = np.empty_like(self.time)
        count = np.zeros(1, dtype=np.int64)
        status = self._lib.validator_model(
            self.time.size, self.time, self.supersample, self.offsets, self.half_span,
            theta, out, count, int(everywhere),
        )  # fmt: skip
        return (out if status == 1 else None), int(count[0])
