"""TRICERATOPS's stellar spline relations (triceratops/funcs.py), bitwise equal and fast.

Upstream evaluates scipy InterpolatedUnivariateSpline objects through FITPACK's Fortran
splev and fpbspl. scipy 1.13's arm64 build fuses two operations there (fmadd): the
de Boor update h(i) + f * (t(li) - x) and the sum sp + c(ll) * h(j); every other
operation rounds separately. splev() repeats that arithmetic with llvm.fma in exactly
those two places, so each value has the same bits; tests compare it with scipy.
"""

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
from triceratops import funcs


@intrinsic
def fma(typingctx, a, b, c):
    """a * b + c rounded once (llvm.fma, the fmadd scipy's Fortran uses)."""

    def codegen(context, builder, signature, args):
        double = ir.DoubleType()
        function = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(double, [double] * 3), "llvm.fma.f64"
        )
        return builder.call(function, args)

    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True, inline="always")
def basis(arg, t, h, hh):
    """FITPACK's cubic basis; both stellar relations have these same knots."""
    k1 = 4
    nk1 = t.size - k1
    lo, hi = k1, nk1 + 1
    if arg != arg:
        lo = nk1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid - 1] <= arg:
            lo = mid
        else:
            hi = mid
    h[0] = 1.0
    for j in range(1, k1):
        hh[:j] = h[:j]
        h[0] = 0.0
        for i in range(1, j + 1):
            li, lj = lo + i, lo + i - j
            if t[li - 1] == t[lj - 1]:
                h[i] = 0.0
                continue
            f = hh[i - 1] / (t[li - 1] - t[lj - 1])
            h[i - 1] = fma(t[li - 1] - arg, f, h[i - 1])
            h[i] = (arg - t[lj - 1]) * f
    return lo


@njit(cache=True, inline="always")
def value(c, interval, h):
    result = 0.0
    for j in range(4):
        result = fma(c[interval - 4 + j], h[j], result)
    return result


@njit(cache=True)
def splev(x, t, c):
    """FITPACK splev for a cubic spline with extrapolation (ext=0)."""
    out = np.empty(x.size)
    h, hh = np.empty(4), np.empty(4)
    for m in range(x.size):
        out[m] = value(c, basis(x[m], t, h, hh), h)
    return out


def evaluate(spline, x):
    """spline(x) for one of funcs' InterpolatedUnivariateSpline objects."""
    t, c, k = spline._eval_args
    if k != 3 or spline.ext != 0:
        raise RuntimeError("only cubic, extrapolating splines are replicated")
    x = np.asarray(x, dtype=np.float64)
    return splev(np.ascontiguousarray(x.ravel()), t, c).reshape(x.shape)


@njit(cache=True)
def stellar_kernel(
    masses, max_radii, max_teffs, hot_t, hot_r, hot_teff, cool_t, cool_r, cool_teff
):
    radii, teffs = np.zeros(masses.size), np.zeros(masses.size)
    h, hh = np.empty(4), np.empty(4)
    for i in range(masses.size):
        if masses[i] > 0.63:
            interval = basis(masses[i], hot_t, h, hh)
            radii[i], teffs[i] = value(hot_r, interval, h), value(hot_teff, interval, h)
        elif masses[i] <= 0.63:
            interval = basis(masses[i], cool_t, h, hh)
            radii[i], teffs[i] = (
                value(cool_r, interval, h),
                value(cool_teff, interval, h),
            )
        if radii[i] > max_radii[i]:
            radii[i] = max_radii[i]
        if teffs[i] > max_teffs[i]:
            teffs[i] = max_teffs[i]
        if radii[i] < 0.1:
            radii[i] = 0.1
        if teffs[i] < 2800:
            teffs[i] = 2800
    return radii, teffs


def pair(radius, teff):
    tr, cr, kr = radius._eval_args
    tt, ct, kt = teff._eval_args
    if (
        kr != 3
        or kt != 3
        or radius.ext != 0
        or teff.ext != 0
        or not np.array_equal(tr, tt)
    ):
        raise RuntimeError(
            "Require the pinned cubic stellar relations with identical knots"
        )
    return tr, cr, ct


def stellar_relations(Masses, max_Radii, max_Teffs):
    """Share the exact basis and clipping pass between radius and temperature."""
    masses, radii, teffs = (
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (Masses, max_Radii, max_Teffs)
    )
    if masses.ndim != 1 or radii.shape != masses.shape or teffs.shape != masses.shape:
        raise ValueError("Supply aligned one-dimensional stellar relation inputs")
    hot = pair(funcs.Rad_spline_Torres, funcs.Teff_spline_Torres)
    cool = pair(funcs.Rad_spline_cdwrf, funcs.Teff_spline_cdwrf)
    return stellar_kernel(masses, radii, teffs, *hot, *cool)


FLUX = {
    "TESS": funcs.flux_spline,
    "Vis": funcs.flux_spline,
    "J": funcs.flux_spline_J,
    "H": funcs.flux_spline_H,
    "K": funcs.flux_spline_K,
}


def flux_relation(Masses, filt="TESS"):
    """funcs.flux_relation with replicated splines; 10** stays numpy's."""
    return 10 ** evaluate(FLUX[filt], Masses)
