"""Vetter with the same measurements and less time per signal."""

import numpy as np
from lmfit import Parameters
from scipy.optimize import curve_fit

from detector import compiled_search, kernel_bindings
from detector.gap_padding import pad_time_series
from vetter import exact_fit, gap_filling
from vetter import kernel_bindings as kernels
from vetter.transit_model import TransitModel
from vetter.upstream import ses as upstream_ses
from vetter.upstream import vetting as upstream
from vetter.upstream.vetting import (
    fit_sinewave,
    get_p_tdur_t0,
    make_transit_mask,
)


def tce_masked_num_den(time, flux, t0, P, width, cadence, fill_mode="reflect",
                       nwindow=7.0, n_channels=16):  # fmt: skip
    """upstream.tce_masked_num_den with compiled medians and convolutions."""
    template_cadence = 0.020417  # get_transit_signal's unchanged upstream default
    pad_time, pad_flux, pad_boo = pad_time_series(
        time,
        flux,
        cadence,
        gap_filling.pad_time_series,
        in_mode=fill_mode,
        minimum=max(
            4, 2 * len(np.arange(template_cadence, width, template_cadence)) + 1
        ),
    )
    tce_mask = upstream.make_transit_mask(pad_time, P, t0, width)
    if np.sum(tce_mask) == len(tce_mask):
        pad_flux_tce = pad_flux.copy()
    else:
        pad_flux_tce = np.interp(pad_time, pad_time[tce_mask], pad_flux[tce_mask])
    template = upstream.get_transit_signal(
        width=width, depth=100.0, pad=len(pad_flux), b=0.2
    )
    level = n_channels - 1
    flux_transform = kernel_bindings.ocwt(
        pad_flux - np.median(pad_flux), max_level=level, upstream=upstream.ocwt
    )
    tce_masked_flux_transform = kernel_bindings.ocwt(
        pad_flux_tce - np.median(pad_flux_tce), max_level=level, upstream=upstream.ocwt
    )
    template_transform = compiled_search.Template(
        kernel_bindings.ocwt(template - 1.0, max_level=level, upstream=upstream.ocwt)
    )
    window_size = nwindow * width
    sig2 = [
        1.0
        / compiled_search.calc_var_stat(
            x, window_size * 2**i, method="mad", exp_time=cadence, ses=upstream_ses
        )
        for i, x in enumerate(tce_masked_flux_transform)
    ]
    sig2 = np.array(sig2)
    n_levels = flux_transform.shape[0]
    levels = np.concatenate([np.arange(1, n_levels), [n_levels - 1]])
    if pad_boo.dtype != bool or len(pad_boo) != len(pad_flux):
        np.zeros((n_levels, len(pad_flux)))[:, pad_boo]
        raise TypeError("pad_time_series returned an unusable keep mask")
    need = np.flatnonzero(pad_boo)
    N_i, D_i = [], []
    for i, level in enumerate(levels):
        weighted = flux_transform[i] * sig2[i]
        N_i.append(
            2.0 ** (-level)
            * compiled_search.convolve_same(
                weighted, template_transform, i, False, need, upstream_ses
            )
        )
        D_i.append(
            2.0 ** (-level)
            * compiled_search.convolve_same(
                sig2[i], template_transform, i, True, need, upstream_ses
            )
        )
    return np.array(N_i), np.array(D_i)


def tce_masked_num_den_sectors(time, flux, t0, P, width, cadence, fill_mode="reflect",
                               nwindow=7, sector_dates=None):  # fmt: skip
    """upstream.tce_masked_num_den_sectors over the compiled function above."""
    if sector_dates is None:
        return tce_masked_num_den(
            time, flux, t0, P, width, cadence, fill_mode=fill_mode, nwindow=nwindow
        )
    segment_inds = [np.argmin(np.abs(sd - time)) for sd in sector_dates]
    if time[segment_inds[-1]] != time[-1]:
        segment_inds = segment_inds + [None]
    segment_lengths = [
        int(np.log2((time[segment_inds[i + 1]] - time[segment_inds[i]]) / cadence))
        for i in range(len(segment_inds) - 2)
    ]
    max_n_channels = min(segment_lengths)
    N_i = np.zeros(shape=(max_n_channels, len(time)))
    D_i = np.zeros(shape=(max_n_channels, len(time)))
    for i, seg_init in enumerate(segment_inds[:-1]):
        seg_end = segment_inds[i + 1]
        N_i_segment, D_i_segment = tce_masked_num_den(
            time[seg_init:seg_end], flux[seg_init:seg_end], t0, P, width, cadence,
            fill_mode=fill_mode, nwindow=nwindow, n_channels=max_n_channels,
        )  # fmt: skip
        N_i[:, seg_init:seg_end] = N_i_segment
        D_i[:, seg_init:seg_end] = D_i_segment
    return N_i, D_i


def temporal_statistic(t, t0, P, cadence, N_i, D_i):
    """The arithmetic of upstream.temporal_chi2_statistic after its inputs."""
    foldtime = (t - t0 + P / 2.0) % P - P / 2.0
    t0_mask = np.abs(foldtime) <= cadence
    n_transits = np.sum(t0_mask)
    N = np.sum(N_i, axis=0)
    D = np.sum(D_i, axis=0)
    t0_num, t0_den = N[t0_mask], D[t0_mask]
    Z = np.sum(t0_num) / np.sqrt(np.sum(t0_den))
    Zj = t0_num / np.sqrt(np.sum(t0_den))
    Qj = t0_den / np.sum(t0_den)
    delta_Zj = (Zj - Qj * Z) ** 2.0
    chi2 = np.sum(delta_Zj / Qj)
    red_chi2 = chi2 / (n_transits - 1)
    chi2_stat = Z / np.sqrt(red_chi2)
    return red_chi2, chi2_stat


def channel_statistic(time, t0, P, cadence, N_i, D_i):
    """The arithmetic of upstream.channel_chi2_statistic after its inputs."""
    n_levels = N_i.shape[0]
    N = np.sum(N_i, axis=0)
    D = np.sum(D_i, axis=0)
    z = N / np.sqrt(D)
    zi = N_i / np.sqrt(D)
    qi = D_i / D
    delta_zi = zi - qi * z
    chi2_n = np.sum(delta_zi**2.0 / qi, axis=0)
    foldtime = (time - t0 + P / 2.0) % P - P / 2.0
    t0_mask = np.abs(foldtime) < cadence
    n_transits = np.sum(t0_mask)
    Z = np.sum(N[t0_mask]) / np.sqrt(np.sum(D[t0_mask]))
    chi2 = np.sum(chi2_n[t0_mask])
    red_chi2 = chi2 / (n_transits * (n_levels - 1))
    chi2_stat = Z / np.sqrt(red_chi2)
    return red_chi2, chi2_stat


def kernel_residual(window, kind, names, warn=False):
    """An lmfit-style residual over `window` for the named parameter order."""

    def residual(pars, *args):
        vals = pars.valuesdict()
        out, nan = window.residual(kind, [vals[name] for name in names])
        if warn and nan:
            print("OH NO!!!!!! NANS!!!!!!!")
            print(vals)
        return out

    residual.kernel = (window, kind, names)
    return residual


TRAP = ("a", "b", "tdur", "t0", "c0", "c1", "c2")
BOX = ("a", "tdur", "t0", "c0", "c1", "c2")
SPSD = ("a", "d", "t0", "tdur", "c0", "c1", "c2")
SINE = ("a", "tdur", "t0", "c0", "c1", "c2")


def fit_transit(x, y, yerr, period, limb_darkening, exptime, p0, t0):
    """upstream.fit_transit with TransitModel."""
    import batman

    params = batman.TransitParams()
    params.t0 = t0
    params.per = np.pi * (p0[1]) / np.arcsin(1.0 / 50.0)
    params.rp = 1e-3
    params.a = 50.0
    params.inc = 90.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = limb_darkening
    params.limb_dark = "nonlinear"
    m = TransitModel(params, x, exp_time=exptime, fac=0.001)

    def lc_func(t, *par):
        rprs, width, t0, c = par
        params.per = np.pi * (width) / np.arcsin(1.0 / 50.0)
        params.rp = 10.0**rprs
        params.t0 = t0
        return m.light_curve(params) - 1.0 + c

    p0 = np.append(p0, [0])
    result = curve_fit(
        lc_func, xdata=x, ydata=y, p0=p0, sigma=yerr, absolute_sigma=True,
        bounds=[[-8, p0[1] / 5.0, -p0[1], -1], [0.0, p0[1] * 5.0, p0[1], 1]],
        check_finite=True, jac=None, max_nfev=1000 * len(x),
    )  # fmt: skip
    resids = y - lc_func(x, *result[0])
    chi2 = np.sum(resids**2.0 / yerr**2.0)
    x_fold = (x + period / 2.0) % period - period / 2.0
    plot_x = np.linspace(min(x_fold), max(x_fold), 100)
    m = TransitModel(
        params, plot_x, exp_time=exptime, fac=0.001, supersample_factor=5, max_err=0.1
    )
    return chi2, resids, plot_x, lc_func(plot_x, *result[0])


def compare_cosine_and_transit_model(time, flux, t0, width, P, limb_dark_coeffs, exptime,
                                     flux_err=None, nwidth=4, depth=None):  # fmt: skip
    """upstream.compare_cosine_and_transit_model(global_fit=True, plot=False)."""
    if depth is None:
        depth = 1e-3
    foldtime = (time - t0 + P / 2.0) % P - P / 2.0
    mask = ~make_transit_mask(time, P, t0, width * nwidth / 2)
    transit_time = foldtime[mask]
    cosine_time = foldtime[mask]
    transit_flux = flux[mask] - 1.0
    cosine_flux = flux[mask] - 1.0
    transit_fluxerr = flux_err[mask]
    cosine_fluxerr = flux_err[mask]
    sine_chi2s = []
    all_sine_results = []
    for n in [0.5, 1, 2]:
        results = fit_sinewave(
            cosine_time,
            cosine_flux,
            yerr=cosine_fluxerr,
            p0=[0.0, depth, n * width, 0.0],
        )
        sine_chi2s.append(results[0])
        all_sine_results.append(results)
    sine_chi2, sine_resids, sine_x, sine_y = all_sine_results[np.argmin(sine_chi2s)]
    tran_chi2, tran_resids, tran_x, tran_y = fit_transit(
        transit_time, transit_flux, yerr=transit_fluxerr, p0=[np.log10(depth), width, 0.0],
        t0=0.0, period=P, limb_darkening=limb_dark_coeffs, exptime=exptime,
    )  # fmt: skip
    tran_chi2 = tran_chi2 / len(tran_resids)
    sine_chi2 = sine_chi2 / len(sine_resids)
    nfreedom = len(transit_time) - 3.0
    return sine_chi2, tran_chi2, nfreedom


def odd_even_transit_depths(t, f, ferr, P, t0, width, initial_fit_method="LBFGS",
                            fit_method="leastsq"):  # fmt: skip
    """upstream.odd_even_transit_depths with exactfit and the trapezoid kernel."""
    phase = (t - t0 + P / 4) % (2 * P)
    odd = phase < P
    even = phase >= P
    # One parity may have no data on a long period: the comparison is undefined (upstream raises TypeError).
    if min(np.count_nonzero(odd), np.count_nonzero(even)) <= 4:
        raise ValueError("Odd or even transits hold too few cadences to compare")
    odd_flux = f[odd]
    odd_phase = phase[odd]
    even_flux = f[even]
    even_phase = phase[even] - P
    bothparams = Parameters()
    bothparams.add("t0", value=0.25 * P, vary=True, min=0.2 * P, max=0.3 * P)
    bothparams.add("a", value=1e-3, vary=True, min=1e-6, max=0.99)
    bothparams.add("b", value=1e3, vary=True, min=100, max=1e4)
    bothparams.add("tdur", value=width, min=0.2 * width, max=5 * width, vary=True)
    bothparams.add("c0", value=1.0, vary=True)
    bothparams.add("c1", value=0.0, vary=False)
    bothparams.add("c2", value=0.0, vary=False)
    both = kernel_residual(kernels.Window(phase % P, f, ferr), kernels.TRAP, TRAP)
    initresult = exact_fit.minimize(
        both, bothparams, method=initial_fit_method, reduce_fcn="neglogcauchy"
    )
    bothresult = exact_fit.minimize(
        both, initresult.params, method=fit_method, reduce_fcn="neglogcauchy"
    )
    bothresult.params["b"].vary = False
    evenresult = exact_fit.minimize(
        kernel_residual(kernels.Window(even_phase, even_flux, ferr[even]), kernels.TRAP, TRAP),
        bothresult.params, method=fit_method, reduce_fcn="neglogcauchy",
    )  # fmt: skip
    oddresult = exact_fit.minimize(
        kernel_residual(kernels.Window(odd_phase, odd_flux, ferr[odd]), kernels.TRAP, TRAP),
        bothresult.params, method=fit_method, reduce_fcn="neglogcauchy",
    )  # fmt: skip
    return phase, bothresult, evenresult, oddresult


def morphology_parameters(f, tdur, depth):
    """The four lmfit Parameters sets of upstream.bic_morphology_test."""
    boxparams = Parameters()
    boxparams.add("a", value=depth, min=0.0)
    boxparams.add("t0", value=0.0, vary=True, min=-2 * tdur, max=2 * tdur)
    boxparams.add("tdur", value=tdur * 0.75, vary=True, min=0.5 * tdur, max=2 * tdur)
    boxparams.add("c0", value=np.median(f), vary=True)
    rampparams = Parameters()
    rampparams.add("a", value=depth * 2.0, max=1.0, min=0.0)
    rampparams.add("d", value=2, min=1, max=100)
    rampparams.add("t0", value=-tdur / 4.0, vary=True, min=-tdur, max=tdur)
    rampparams.add("tdur", value=tdur, vary=True, min=0, max=4 * tdur)
    rampparams.add("c0", value=np.median(f), vary=True)
    spsdparams = Parameters()
    spsdparams.add("a", value=depth * 2.0, max=1.0, min=0.0)
    spsdparams.add("d", value=2, min=1, max=100)
    spsdparams.add("t0", value=-tdur / 4.0, vary=True, min=-tdur, max=tdur)
    spsdparams.add("tdur", value=tdur, vary=True, min=0, max=2 * tdur)
    spsdparams.add("c0", value=np.median(f), vary=True)
    sineparams = Parameters()
    sineparams.add("a", value=depth * 2.0, max=1.0, min=0.0)
    sineparams.add("t0", value=0.0, vary=True, min=-tdur, max=tdur)
    sineparams.add("tdur", value=tdur / 4.0, vary=True, min=tdur / 8.0, max=2 * tdur)
    sineparams.add("c0", value=np.median(f), vary=True)
    for params in (sineparams, spsdparams, rampparams, boxparams):
        params.add("c1", value=0, vary=True)
        params.add("c2", value=0, vary=True)
    return boxparams, rampparams, spsdparams, sineparams


def morphology_statistics(n_transits, num_good_transits, frac, ramp, spsd, sine):
    """The output dictionaries of upstream.bic_morphology_test."""
    if num_good_transits != 0:
        return {
            "num_transits": n_transits,
            "num_good_transits": num_good_transits,
            "frac_points_in_tran": frac / num_good_transits,
            "ramp_median_bic_stat": np.median(ramp),
            "ramp_min_bic_stat": np.min(ramp),
            "ramp_bic_stat": np.sum(ramp),
            "spsd_median_bic_stat": np.median(spsd),
            "spsd_min_bic_stat": np.min(spsd),
            "spsd_bic_stat": np.sum(spsd),
            "sine_median_bic_stat": np.median(sine),
            "sine_min_bic_stat": np.min(sine),
            "sine_bic_stat": np.sum(sine),
        }
    return {
        "num_transits": n_transits,
        "num_good_transits": num_good_transits,
        "frac_points_in_tran": 0.0,
        "ramp_median_bic_stat": np.nan,
        "ramp_min_bic_stat": np.nan,
        "ramp_bic_stat": np.nan,
        "spsd_median_bic_stat": np.nan,
        "spsd_min_bic_stat": np.nan,
        "spsd_bic_stat": np.nan,
        "sine_median_bic_stat": np.nan,
        "sine_min_bic_stat": np.nan,
        "sine_bic_stat": np.nan,
    }


def bic_morphology_test(t, f, ferr, P, t0, tdur, depth=1e-4, fit_method="LBFGS", ntdur=3,
                        texp=0.0204, min_frac=0.8):  # fmt: skip
    """upstream.bic_morphology_test (no plots, no detrending) with exactfit and kernels."""
    if P < 0:
        P = t[-1] - t[0]
    boxparams, rampparams, spsdparams, sineparams = morphology_parameters(
        f, tdur, depth
    )
    ramp_bic_probs = []
    spsd_bic_probs = []
    sine_bic_probs = []
    num_good_transits = 0
    frac_good_transit_points = 0
    n_transits = 0
    tn = t0
    t_maximum = max(t)
    ordered = (
        tn < t_maximum
        and type(t) is np.ndarray
        and t.ndim == 1
        and t.dtype == np.float64
        and type(f) is np.ndarray
        and type(ferr) is np.ndarray
        and len(f) == len(t)
        and len(ferr) == len(t)
        and np.isfinite(t).all()
        and np.all(t[1:] >= t[:-1])
    )
    while tn < t_maximum:
        t_min, t_max = tn - ntdur * tdur, tn + ntdur * tdur
        if ordered and t_min <= t_max:
            start = np.searchsorted(t, t_min, side="right")
            stop = np.searchsorted(t, t_max, side="left")
            tce_cut = slice(start, stop)
            npoints = np.int64(stop - start)
        else:
            tce_cut = np.logical_and(t > t_min, t < t_max)
            npoints = sum(tce_cut)
        tce_time = t[tce_cut] - tn
        tce_flux = f[tce_cut].copy() if isinstance(tce_cut, slice) else f[tce_cut]
        tce_err = ferr[tce_cut].copy() if isinstance(tce_cut, slice) else ferr[tce_cut]
        n_transits += 1
        tn += P
        if npoints < min_frac * (2 * ntdur * tdur / texp):
            continue
        num_good_transits += 1
        in_transit = np.abs(tce_time) < tdur
        count = np.sum(in_transit) if ordered and len(in_transit) else sum(in_transit)
        frac_good_transit_points += count / (2 * tdur / texp)
        window = kernels.Window(tce_time, tce_flux, tce_err)
        ramp_window = kernels.Window(-1 * tce_time, tce_flux, tce_err)
        box_out = exact_fit.minimize(
            kernel_residual(window, kernels.BOX, BOX), boxparams, method=fit_method
        )
        ramp_out = exact_fit.minimize(
            kernel_residual(ramp_window, kernels.SPSD, SPSD, warn=True), rampparams,
            method=fit_method,
        )  # fmt: skip
        spsd_out = exact_fit.minimize(
            kernel_residual(window, kernels.SPSD, SPSD, warn=True), spsdparams,
            method=fit_method,
        )  # fmt: skip
        sine_out = exact_fit.minimize(
            kernel_residual(window, kernels.SINE, SINE), sineparams, method=fit_method
        )
        ramp_bic_probs.append((box_out.bic - ramp_out.bic) / 2.0)
        spsd_bic_probs.append((box_out.bic - spsd_out.bic) / 2.0)
        sine_bic_probs.append((box_out.bic - sine_out.bic) / 2.0)
    return morphology_statistics(
        n_transits, num_good_transits, frac_good_transit_points, ramp_bic_probs,
        spsd_bic_probs, sine_bic_probs,
    )  # fmt: skip


def grouped_events(t, f, ferr, P, t0, tdur, groups, ntdur=3):
    """Cadences near transits, stacked into at most `groups` pseudo-events."""
    epoch = np.round((t - t0) / P)
    local = t - (t0 + epoch * P)
    near = np.abs(local) < ntdur * tdur
    observed = np.unique(epoch[near])
    if not len(observed):
        return t[:0], f[:0], ferr[:0], 0
    size = int(np.ceil(len(observed) / groups))
    group = np.searchsorted(observed, epoch[near]) // size
    pseudo = t0 + group * P + local[near]
    order = np.argsort(pseudo, kind="stable")
    return pseudo[order], f[near][order], ferr[near][order], size


REFINEMENT_WEIGHTS = ("sigma", "sigma_squared")


class FastVetter(upstream.Vetter):
    """Native Vetter (the upstream class) with exact replacements in its fits and chi-square tests."""

    refinement_weight = "sigma"

    def __init__(self, Dump, TCEs, refinement_weight=None):
        super().__init__(Dump, TCEs)
        if refinement_weight is not None:
            if refinement_weight not in REFINEMENT_WEIGHTS:
                raise ValueError(f"Unknown refinement weight {refinement_weight}")
            self.refinement_weight = refinement_weight

    def _transit_resids(self, par, robust=False):
        """upstream._transit_resids, divided by sigma rather than sigma**2."""
        if self.refinement_weight == "sigma_squared":
            return super()._transit_resids(par, robust)
        try:
            vals = par.valuesdict()
        except AttributeError:
            vals = par
        b, tdur, t0 = vals["b"], vals["tdur"], vals["t0"]
        per, logdepth = vals["per"], vals["logdepth"]
        rp_rs = np.sqrt(10.0**logdepth)
        a_rs = np.sqrt((1.0 + rp_rs) ** 2.0 - b**2.0) / np.sin((np.pi * tdur) / per)
        if b / a_rs > 1.0:
            return np.ones_like(self.flux) * 1e20
        inc = np.rad2deg(np.arccos(b / a_rs))
        self.batman_params.t0 = t0
        self.batman_params.per = per
        self.batman_params.rp = rp_rs
        self.batman_params.a = a_rs
        self.batman_params.inc = inc
        lc = self._get_lightcurve()
        return (self.flux - lc) / self.flux_err

    def _get_batman(self, limb_law="nonlinear"):
        import batman

        params = batman.TransitParams()
        params.t0 = self.time[0]
        params.per = 1.0
        params.rp = 0.1
        params.a = 15.0
        params.inc = 90.0
        params.ecc = 0.0
        params.w = 90.0
        params.limb_dark = limb_law
        params.u = self.limb_coeff
        m = TransitModel(
            params, self.time, supersample_factor=5, exp_time=self.exptime, max_err=0.1
        )
        return m, params

    def _update_tce(
        self, tce_num, fit_method="leastsq", return_fit=False, robust=False
    ):
        P, width, t0 = get_p_tdur_t0(self.tces[tce_num])
        params = Parameters()
        params.add("logdepth", value=-3.0, vary=True, max=0, min=-6)
        if P > max(self.time) - min(self.time):
            params.add("per", value=max(self.time) - min(self.time), vary=False)
        else:
            params.add("per", value=P, vary=True, min=0.99 * P, max=1.01 * P)
        params.add("t0", value=t0, vary=True, min=t0 - 2 * width, max=t0 + 2 * width)
        params.add(
            "tdur", value=width, min=0.25 * width, max=min(2 * width, P / 2), vary=True
        )
        params.add("b", value=0.1, min=0.0, max=1.0, vary=True)
        params.add("rp_rs", expr="10**(0.5 * logdepth)")
        params.add(
            "a_rs", expr="sqrt( (1 + rp_rs)**2 - b**2)/sin(pi*tdur/per)", min=1.0
        )
        if fit_method == "emcee":
            raise NotImplementedError("emcee refinement has no exact replacement")
        tce_out = exact_fit.minimize(self._transit_resids, params, method=fit_method)
        out = tce_out.params.valuesdict()
        if P > max(self.time) - min(self.time):
            self.refined_tces[tce_num, 1] = P
        else:
            self.refined_tces[tce_num, 1] = out["per"]
        self.refined_tces[tce_num, 2] = 10.0 ** out["logdepth"]
        self.refined_tces[tce_num, 3] = out["t0"]
        self.refined_tces[tce_num, 4] = out["tdur"]
        if return_fit:
            return out

    def cosine_vs_transit_global(self, tce_num, showplot=False, nwidth=4, depth=None,
                                 return_plot_values=False):  # fmt: skip
        if showplot or return_plot_values:
            raise NotImplementedError("plot values have no exact replacement")
        P, width, t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        depth = self.refined_tces[tce_num, 2]
        chic, chit, nf = compare_cosine_and_transit_model(
            time=self.time, flux=self.flux, t0=t0, width=width, P=P, depth=depth**0.5,
            exptime=self.exptime, limb_dark_coeffs=self.limb_coeff, flux_err=self.flux_err,
            nwidth=nwidth,
        )  # fmt: skip
        return {
            "global_tran_red_chi2": chit,
            "global_sin_red_chi2": chic,
            "global_tran_chi2": chit * nf,
            "global_sin_chi2": chic * nf,
            "global_diff_red_chi2": chit - chic,
            "global_diff_chi2": (chit - chic) * nf,
        }

    def odd_even_depth_test(self, tce_num, return_dict=False, fit_method="leastsq"):
        P, width, t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        _, both, even, odd = odd_even_transit_depths(
            t=self.time, f=self.flux, ferr=self.flux_err, P=P, t0=t0, width=width,
            fit_method="leastsq", initial_fit_method="LBFGS",
        )  # fmt: skip
        if even.params["a"].stderr is not None and odd.params["a"].stderr is not None:
            stat = np.abs(odd.params["a"] - even.params["a"]) / np.sqrt(
                even.params["a"].stderr ** 2.0 + odd.params["a"].stderr ** 2.0
            )
        else:
            print("Odd-Even Test Failed")
            stat = np.nan
        if return_dict:
            return {
                "odd_even_depth_stat": stat,
                "odd_depth": odd.params["a"].value,
                "even_depth": even.params["a"].value,
                "odd_depth_err": odd.params["a"].stderr,
                "even_depth_err": even.params["a"].stderr,
            }
        return both, even, odd, stat

    def local_morphology_test(self, tce_num, **test_kw):
        P, width, t0 = get_p_tdur_t0(self.refined_tces[tce_num])
        depth = self.refined_tces[tce_num, 2]
        t = self.time
        f = self.flux * self.trend
        ferr = self.flux_err
        texp = self.exptime
        if len(self.time) * texp / P > 100:
            groups = getattr(self, "morphology_groups", None)
            if not groups:
                return {}
            # Upstream skips the test here; consecutive transits are grouped and each group tested as one event.
            t, f, ferr, size = grouped_events(t, f, ferr, P, t0, width, groups)
            if not len(t):
                return {}
            out = bic_morphology_test(
                t, f, ferr, P, t0, width, depth, texp=texp, **test_kw
            )
            if out.get("num_good_transits"):
                out["frac_points_in_tran"] /= size
            return out | {"morphology_transits_per_group": size}
        return bic_morphology_test(
            t, f, ferr, P, t0, width, depth, texp=texp, **test_kw
        )

    def chi2_tests(self, tce_num, calc_ses=False, use_mask=None, mask_other_tces=False):
        P, width, t0 = upstream.get_p_tdur_t0(self.refined_tces[tce_num])
        texp = self.exptime
        if use_mask is not None:
            time = self.time[use_mask]
            flux = self.flux[use_mask]
        else:
            time = self.dump.lc.time[self.dump.lc.mask]
            flux = self.dump.lc.flux[self.dump.lc.mask]
        # Channel and temporal statistics share this call; an IndexError gives NaN.
        try:
            N_i, D_i = tce_masked_num_den_sectors(
                time, flux, t0, P, width, texp, "reflect", 7,
                sector_dates=self.dump.sector_dates,
            )  # fmt: skip
        except IndexError:
            channel_red_chi2 = channel_chi2_stat = np.nan
            temporal_red_chi2 = temporal_chi2_stat = np.nan
        else:
            channel_red_chi2, channel_chi2_stat = channel_statistic(
                time, t0, P, texp, N_i, D_i
            )
            temporal_red_chi2, temporal_chi2_stat = temporal_statistic(
                time, t0, P, texp, N_i, D_i
            )
        return {
            "channel_red_chi2": np.round(channel_red_chi2, 3),
            "temporal_red_chi2": np.round(temporal_red_chi2, 3),
            "channel_chi2_stat": np.round(channel_chi2_stat, 3),
            "temporal_chi2_stat": np.round(temporal_chi2_stat, 3),
        }
