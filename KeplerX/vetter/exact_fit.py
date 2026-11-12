"""lmfit.minimize for the Vetter fits, with the same numbers and less overhead."""

import copy
import math

import numpy as np
from lmfit.minimizer import (
    HAS_NUMDIFFTOOLS,
    AbortFitException,
    reduce_cauchylogpdf,
    reduce_chisquare,
)
from scipy.optimize import _lbfgsb
from scipy.optimize import leastsq as scipy_leastsq
from scipy.optimize import minimize as scipy_minimize

from vetter.kernel_bindings import ONE_BLOCK, Gradient, Window

# scipy L-BFGS-B without its wrappers; False routes the scalar fits through SciPy's own loop.
DRIVER = True
EPS = np.finfo(np.float64).eps
# _minimize_lbfgsb defaults: maxcor, ftol, gtol, eps, maxls.
LBFGSB_M, LBFGSB_FTOL, LBFGSB_PGTOL, LBFGSB_STEP, LBFGSB_MAXLS = (
    10,
    2.2204460492503131e-09,
    1e-5,
    1e-8,
    20,
)


def negative_cauchy_loglike(r):
    """lmfit's reduce_cauchylogpdf, -scipy.stats.cauchy.logpdf(r).sum()."""
    if not np.isfinite(r).all():
        return reduce_cauchylogpdf(r)
    with np.errstate(divide="ignore"):
        return -np.log(1.0 / np.pi / (1.0 + r * r)).sum()


class Driven:
    """The fields scipy's OptimizeResult carries into lmfit's result."""


class Scalar:
    """ScalarFunction for jac=None: the objective and its 2-point gradient."""

    def __init__(self, fun, x0, gradient=None, value_gradient=None):
        self.fun, self.n, self.nfev, self.ngev = fun, x0.size, 0, 0
        self.batch_gradient = gradient
        self.value_gradient = value_gradient
        self.x = np.array(x0, np.float64)
        self.f, self.g = self.calculate(self.x)

    def calculate(self, x):
        if self.value_gradient is not None:
            result = self.value_gradient(x)
            if result is not None:
                self.nfev += self.n + 1
                self.ngev += 1
                return result
        f = self.evaluate(x)
        return f, self.gradient(x, f)

    def evaluate(self, x):
        self.nfev += 1
        fx = self.fun(np.copy(x))
        if not np.isscalar(fx):
            fx = np.asarray(fx).item()
        return fx

    def gradient(self, x, f0):
        self.ngev += 1
        sign_x0 = (x >= 0).astype(float) * 2 - 1
        h = LBFGSB_STEP
        dx = (x + h) - x
        h = np.where(dx == 0, EPS**0.5 * sign_x0 * np.maximum(1.0, np.abs(x)), h)
        if self.batch_gradient is not None:
            result = self.batch_gradient(x, h, f0)
            if result is not None:
                self.nfev += self.n
                return result
        g = np.empty(self.n)
        for i in range(self.n):
            step = np.zeros(self.n)
            step[i] = h[i]
            xi = x + step
            g[i] = (self.evaluate(xi) - f0) / (xi[i] - x[i])
        return g

    def fun_and_grad(self, x):
        if not np.array_equal(x, self.x):
            self.x = np.array(x, np.float64)
            self.f, self.g = self.calculate(self.x)
        return self.f, self.g


def lbfgsb(fun, x0, maxiter, maxfun, gradient=None, value_gradient=None):
    """scipy.optimize._minimize_lbfgsb(fun, x0, jac=None, bounds=None, ...)."""
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    n = x0.size
    sf = Scalar(fun, x0, gradient, value_gradient)
    m = LBFGSB_M
    factr = LBFGSB_FTOL / np.finfo(float).eps
    x = np.array(x0, np.float64)
    bounds = (np.zeros(n, np.float64), np.zeros(n, np.float64), np.zeros(n, np.int32))
    wa = np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, np.float64)
    iwa = np.zeros(3 * n, np.int32)
    task, csave = np.zeros(1, "S60"), np.zeros(1, "S60")
    lsave, isave, dsave = np.zeros(4, np.int32), np.zeros(44, np.int32), np.zeros(29)
    task[:] = "START"
    f, g = np.array(0.0, np.float64), np.zeros((n,), np.float64)
    n_iterations = 0
    while 1:
        g = g.astype(np.float64)
        _lbfgsb.setulb(
            m, x, *bounds, f, g, factr, LBFGSB_PGTOL, wa, iwa, task, -1, csave, lsave,
            isave, dsave, LBFGSB_MAXLS,
        )  # fmt: skip
        task_str = task.tobytes()
        if task_str.startswith(b"FG"):
            f, g = sf.fun_and_grad(x)
        elif task_str.startswith(b"NEW_X"):
            n_iterations += 1
            if n_iterations >= maxiter:
                task[:] = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            elif sf.nfev > maxfun:
                task[:] = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
        else:
            break
    return driven(task, sf, f, g, x, n_iterations, maxiter, maxfun)


def driven(task, sf, f, g, x, n_iterations, maxiter, maxfun):
    """The OptimizeResult of _minimize_lbfgsb, without hess_inv."""
    task_str = task.tobytes().strip(b"\x00").strip()
    if task_str.startswith(b"CONV"):
        warnflag = 0
    elif sf.nfev > maxfun or n_iterations >= maxiter:
        warnflag = 1
    else:
        warnflag = 2
    ret = Driven()
    ret.fun, ret.jac, ret.nfev, ret.njev = f, g, sf.nfev, sf.ngev
    ret.nit, ret.status, ret.message = n_iterations, warnflag, task_str.decode()
    ret.x, ret.success = x, warnflag == 0
    return ret


NAN_MESSAGE = (
    "NaN values detected in your input data or the output of "
    "your objective/model function - fitting algorithms cannot "
    "handle this! Please read https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do "
    "for more information."
)
ERR_MAX_EVALS = (
    "Too many function calls (max set to {:i})! Use:"
    " minimize(func, params, ..., max_nfev=NNN)"
    " to increase this maximum."
)
LEASTSQ_KWS = dict(
    Dfun=None, full_output=1, col_deriv=0, ftol=1.5e-8, xtol=1.5e-8, gtol=0.0,
    epsfcn=1.0e-10, factor=100, diag=None,
)  # fmt: skip


class Values:
    """The parameter view the objective sees: valuesdict() only."""

    __slots__ = ("_values",)

    def __init__(self, values):
        self._values = values

    def valuesdict(self):
        return self._values


class Result:
    """The MinimizerResult fields the vetting code reads."""


def transform(par):
    """Parameter.from_internal as a plain-math function (same libm calls)."""
    low, high = par.min, par.max
    if low == -math.inf and high == math.inf:
        return float
    if high == math.inf:
        return lambda val: low - 1.0 + math.sqrt(val * val + 1)
    if low == -math.inf:
        return lambda val: high + 1 - math.sqrt(val * val + 1)
    return lambda val: low + (math.sin(val) + 1) * (high - low) / 2.0


class Fit:
    def __init__(self, fcn, params, args, reduce_fcn):
        self.fcn, self.args = fcn, args
        result = Result()
        result.params = copy.deepcopy(params)
        result.params.update_constraints()
        result.var_names, result.init_vals, internal = [], [], []
        result.nfev, result.errorbars, result.aborted = 0, False, False
        result.success, result.covar, result.message = True, None, None
        # Values in the Parameters' own order (valuesdict order is observable in prints).
        self.template, self.lower, self.upper, self.from_internal = {}, [], [], []
        for name, par in result.params.items():
            par.stderr = None
            par.correl = None
            if par.expr is not None:
                par.vary = False
            if par.vary:
                result.var_names.append(name)
                internal.append(par.setup_bounds())
                result.init_vals.append(par.value)
                self.lower.append(par.min)
                self.upper.append(par.max)
                self.from_internal.append(transform(par))
                self.template[name] = None
            elif par.expr is None:
                self.template[name] = float(par.value)
            par.init_value = par.value
            if par.name is None:
                par.name = name
        result.nvarys = len(result.var_names)
        result.init_values = dict(zip(result.var_names, result.init_vals))
        result._init_vals_internal = internal
        self.result = result
        self.max_nfev = 2000 * (result.nvarys + 1)
        if reduce_fcn is None:
            self.reduce_fcn = reduce_chisquare
        elif isinstance(reduce_fcn, str) and reduce_fcn.lower().startswith("neglogc"):
            self.reduce_fcn = negative_cauchy_loglike
        else:
            raise NotImplementedError(f"reduce_fcn {reduce_fcn!r}")
        self.direct, self.gradient_kernel = self._kernels(fcn, args)

    def _kernels(self, fcn, args):
        """Prepare exact residual and gradient kernels for supported windows."""
        kernel = getattr(fcn, "kernel", None)
        if kernel is None or self.reduce_fcn is not reduce_chisquare or args:
            return None, None
        window, kind, names = kernel
        if window.n > ONE_BLOCK or any(name not in self.template for name in names):
            return None, None
        gradient = None
        # Subclasses may observe residual_sums calls; retain their path.
        if type(window) is Window and 0 < self.result.nvarys <= 7 and len(names) <= 7:
            gradient = Gradient(
                window, kind, names, self.result.var_names, self.lower, self.upper,
                self.template,
            )  # fmt: skip
        return (window, kind, names), gradient

    def externals(self, fvars):
        """Clipped external values of the variables, in var_names order."""
        values = []
        for convert, low, high, val in zip(
            self.from_internal, self.lower, self.upper, fvars
        ):
            value = float(convert(val))
            if value > high:
                value = high
            elif value < low:
                value = low
            values.append(value)
        return values

    def count(self, fvars):
        """Minimizer.__residual's evaluation count and abort rule."""
        result = self.result
        result.nfev += 1
        result.last_internal_values = fvars
        if result.nfev > self.max_nfev:
            result.aborted = True
            result.message = (
                f"Fit aborted: number of function evaluations > {self.max_nfev}"
            )
            result.success = False
            raise AbortFitException(
                f"fit aborted: too many function evaluations {self.max_nfev}"
            )

    def residual(self, fvars):
        """Minimizer.__residual: external values, the objective, coerce_float64."""
        values = dict(self.template)
        values.update(zip(self.result.var_names, self.externals(fvars)))
        self.current = values
        self.count(fvars)
        out = np.asarray(self.fcn(Values(values), *self.args), dtype=np.float64)
        out = out.ravel()
        if not np.isfinite(np.sum(out)):
            raise ValueError(NAN_MESSAGE)
        return out

    def penalty(self, fvars):
        if self.direct is not None:
            return self.direct_penalty(fvars)
        r = self.residual(fvars)
        if isinstance(r, np.ndarray) and r.size > 1:
            r = self.reduce_fcn(r)
            if isinstance(r, np.ndarray) and r.size > 1:
                r = r.sum()
        return r

    def direct_penalty(self, fvars):
        """penalty() for a compiled residual: the same sums, from the kernel."""
        window, kind, names = self.direct
        values = dict(self.template)
        values.update(zip(self.result.var_names, self.externals(fvars)))
        self.current = values
        self.count(fvars)
        total, chisquare, nan = window.residual_sums(
            kind, [values[name] for name in names]
        )
        if nan and kind == 2:
            print("OH NO!!!!!! NANS!!!!!!!")
            print(values)
        if not math.isfinite(total):
            raise ValueError(NAN_MESSAGE)
        return np.float64(chisquare)

    def direct_gradient(self, x, h, f0):
        """Batch only complete, finite gradients; otherwise retain every evaluation."""
        if self.result.nfev + len(x) > self.max_nfev:
            return None
        result = self.gradient_kernel(x, h, f0)
        if result is not None:
            self.record_gradient(x, h[-1], len(x))
        return result

    def direct_value_gradient(self, x):
        """Fuse only a complete objective/gradient that fits the evaluation budget."""
        if self.result.nfev + len(x) + 1 > self.max_nfev:
            return None
        result = self.gradient_kernel.value_gradient(x)
        if result is None:
            return None
        f, g, last_step = result
        self.record_gradient(x, last_step, len(x) + 1)
        return f, g

    def record_gradient(self, x, last_step, count):
        self.result.nfev += count
        step = np.zeros(len(x))
        step[-1] = last_step
        self.result.last_internal_values = x + step
        self.current = dict(self.template)
        self.current.update(
            zip(self.result.var_names, map(float, self.gradient_kernel.last))
        )

    def finish(self):
        """Store the final values on lmfit's Parameters and the statistics."""
        result = self.result
        for name in result.var_names:
            result.params[name].value = self.current[name]
        result.params.update_constraints()
        residual = result.residual
        result.nvarys = len(result.init_vals)
        if isinstance(residual, np.ndarray):
            result.chisqr = (residual**2).sum()
            result.ndata = len(residual)
            result.nfree = result.ndata - result.nvarys
        else:
            result.chisqr = residual
            result.ndata = 1
            result.nfree = 1
        result.redchi = result.chisqr / max(1, result.nfree)
        result.chisqr = max(result.chisqr, 1.0e-250 * result.ndata)
        neg2_log_likel = result.ndata * np.log(result.chisqr / result.ndata)
        result.aic = neg2_log_likel + 2 * result.nvarys
        result.bic = neg2_log_likel + np.log(result.ndata) * result.nvarys

    def scalar(self, method):
        """Minimizer.scalar_minimize for L-BFGS-B without numdifftools."""
        if HAS_NUMDIFFTOOLS:
            raise NotImplementedError("lmfit would estimate a covariance here")
        result = self.result
        result.method = method
        options = {"maxiter": 2 * self.max_nfev, "maxfun": 2 * self.max_nfev}
        ret = None
        with np.errstate(invalid="ignore"):
            try:
                if DRIVER and method == "L-BFGS-B":
                    ret = lbfgsb(
                        self.penalty, result._init_vals_internal, **options,
                        gradient=self.direct_gradient if self.gradient_kernel is not None else None,
                        value_gradient=self.direct_value_gradient if self.gradient_kernel is not None else None,
                    )  # fmt: skip
                else:
                    ret = scipy_minimize(
                        self.penalty, result._init_vals_internal, method=method,
                        options=options,
                    )  # fmt: skip
            except AbortFitException:
                pass
            if not result.aborted:
                for attr in dir(ret):
                    if not attr.startswith("_"):
                        setattr(result, attr, getattr(ret, attr))
                result.x = np.atleast_1d(result.x)
                result.residual = self.residual(result.x)
                result.nfev -= 1
            else:
                result.x = result.last_internal_values
                result.nfev -= 2
                result.residual = self.residual(result.x)
                result.nfev += 1
        self.finish()
        result.uvars = None
        return result

    def leastsq(self):
        """Minimizer.leastsq."""
        result = self.result
        result.method = "leastsq"
        result.nfev -= 2
        kws = dict(LEASTSQ_KWS, maxfev=2 * self.max_nfev)
        result.call_kws = kws
        original = np.geterr()
        np.seterr(all="ignore")
        lsout = None
        try:
            lsout = scipy_leastsq(self.residual, result._init_vals_internal, **kws)
        except AbortFitException:
            pass
        result.uvars = None
        if not result.aborted:
            best, cov, _infodict, errmsg, ier = lsout
        else:
            best, cov, ier, errmsg = (
                result.last_internal_values,
                None,
                -1,
                "Fit aborted.",
            )
        result.nfev -= 1
        if result.nfev >= self.max_nfev:
            result.nfev = self.max_nfev - 1
        try:
            result.residual = self.residual(best)
            self.finish()
        except AbortFitException:
            pass
        result.ier = ier
        result.lmdif_message = errmsg
        result.success = ier in [1, 2, 3, 4]
        if ier in {1, 2, 3}:
            result.message = "Fit succeeded."
        elif ier == 0:
            result.message = (
                "Invalid Input Parameters. I.e. more variables "
                "than data points given, tolerance < 0.0, or "
                "no data provided."
            )
        elif ier == 4:
            result.message = "One or more variable did not affect the fit."
        elif ier == 5:
            result.message = ERR_MAX_EVALS.format(kws["maxfev"])
        else:
            result.message = "Tolerance seems to be too small."
        result.errorbars = cov is not None
        if result.errorbars:
            result.covar = self.int2ext_cov(cov, best)
            self.uncertainties()
        else:
            result.message = f"{result.message} Could not estimate error-bars."
        np.seterr(**original)
        return result

    def int2ext_cov(self, cov_int, fvars):
        """Minimizer._int2ext_cov_x."""
        params = self.result.params
        g = [
            params[name].scale_gradient(fvars[i])
            for i, name in enumerate(self.result.var_names)
        ]
        grad2d = np.atleast_2d(g)
        grad = np.dot(grad2d.T, grad2d)
        return cov_int * grad

    def uncertainties(self):
        """Minimizer._calculate_uncertainties_correlations without uvars."""
        result = self.result
        result.errorbars = True
        result.covar *= result.redchi
        for par in result.params.values():
            par.stderr, par.correl = 0, None
        for ivar, name in enumerate(result.var_names):
            par = result.params[name]
            par.stderr = float(np.sqrt(result.covar[ivar, ivar]))
            par.correl = {}
            try:
                result.errorbars = result.errorbars and (par.stderr > 0.0)
                for jvar, varn2 in enumerate(result.var_names):
                    if jvar != ivar:
                        par.correl[varn2] = float(
                            result.covar[ivar, jvar]
                            / (par.stderr * np.sqrt(result.covar[jvar, jvar]))
                        )
            except ZeroDivisionError:
                result.errorbars = False
        if result.errorbars:
            result.uvars = result.params.create_uvars(covar=result.covar)


def minimize(fcn, params, method="leastsq", args=None, reduce_fcn=None):
    """lmfit.minimize(fcn, params, method, args=args, reduce_fcn=reduce_fcn)."""
    fit = Fit(fcn, params, () if args is None else tuple(args), reduce_fcn)
    user_method = method.lower()
    if user_method.startswith("leasts"):
        return fit.leastsq()
    if user_method in ("lbfgs", "lbfgsb", "l-bfgs-b"):
        return fit.scalar("L-BFGS-B")
    raise NotImplementedError(f"method {method!r}")
