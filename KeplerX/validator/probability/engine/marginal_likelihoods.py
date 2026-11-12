import functools
from pathlib import Path

import numpy as np
import triceratops
from astropy import constants
from pandas import read_csv
from triceratops.funcs import (
    limiting_separation,
    parse_contrast_curves,
    trilegal_results,
)
from triceratops.likelihoods import lnL_EB, lnL_EB_twin, lnL_TP
from triceratops.priors import (
    lnprior_background,
    lnprior_bound_EB,
    lnprior_bound_TP,
    sample_ecc,
    sample_inc,
    sample_q,
    sample_q_companion,
    sample_rp,
    sample_w,
)

from validator.probability.engine.splines import flux_relation, stellar_relations
from validator.probability.joint import call as _joint_call
from validator.probability.joint import fast_log_mean_exp as _fast_log_mean_exp
from validator.probability.joint import resolved
from validator.probability.kernels import lnL_EB_p, lnL_EB_twin_p, lnL_TP_p
from validator.probability.scenarios import background_ldc as _fast_background_ldc
from validator.probability.scenarios import ldc_lookup as _fast_ldc_lookup
from validator.probability.scenarios import top as _fast_top


def _flux_fraction(masses, reference, filt="TESS"):
    flux = flux_relation(masses, filt)
    return flux / (flux + flux_relation(reference, filt))


def _scenario(function):
    return functools.wraps(function)(resolved(function))


# Array comparisons and inherited scenario branches follow the pinned source.
# ruff: noqa: E712


np.seterr(divide="ignore")

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi
ln2pi = np.log(2 * pi)

_DATA_DIR = Path(triceratops.__file__).parent / "data"

ldc_T = read_csv(_DATA_DIR / "ldc_tess.csv")
ldc_T_Zs = np.array(ldc_T.Z, dtype=float)
ldc_T_Teffs = np.array(ldc_T.Teff, dtype=int)
ldc_T_loggs = np.array(ldc_T.logg, dtype=float)
ldc_T_u1s = np.array(ldc_T.aLSM, dtype=float)
ldc_T_u2s = np.array(ldc_T.bLSM, dtype=float)

ldc_K = read_csv(_DATA_DIR / "ldc_kepler.csv")
ldc_K_Zs = np.array(ldc_K.Z, dtype=float)
ldc_K_Teffs = np.array(ldc_K.Teff, dtype=int)
ldc_K_loggs = np.array(ldc_K.logg, dtype=float)
ldc_K_u1s = np.array(ldc_K.a, dtype=float)
ldc_K_u2s = np.array(ldc_K.b, dtype=float)


def _log_likelihood(function, joint, time, flux, sigma, *args, **kwargs):
    if joint is None:
        return (
            -0.5 * ln2pi - np.log(sigma) - function(time, flux, sigma, *args, **kwargs)
        )
    return joint.log_normalization - _joint_call(
        joint, function, time, flux, sigma, *args, **kwargs
    )


@_scenario
def lnZ_TTP(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the TTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    a = ((G * M_s * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    logg = np.log10(G * (M_s * Msun) / (R_s * Rsun) ** 2)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    rps = sample_rp(np.random.rand(N), np.full(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (rps * Rearth + R_s * Rsun) / a * e_corr

    coll = (rps * Rearth + R_s * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False)
        a_arr = a
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a_arr[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1.0:
                inc_min = np.arccos(Ptra[i]) * 180.0 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, None, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_TEB(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the TEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    logg = np.log10(G * (M_s * Msun) / (R_s * Rsun) ** 2)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    masses = qs * M_s
    radii, Teffs = stellar_relations(masses, np.full(N, R_s), np.full(N, Teff))
    fluxratios = _flux_fraction(masses, np.array([M_s]))

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (radii * Rsun + R_s * Rsun) / a * e_corr
    a_twin = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (2 * P_orb * 86400) ** 2) ** (
        1 / 3
    )
    Ptra_twin = (radii * Rsun + R_s * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (R_s * Rsun)

    coll = (radii * Rsun + R_s * Rsun) > a * (1 - eccs)
    coll_twin = (2 * R_s * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        companion_fluxratio = np.zeros(N)
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] < 0.95) & (coll[i] == False):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] >= 0.95) & (coll_twin[i] == False):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, None, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, None, N_total=N)
    res_twin = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    return res, res_twin


@_scenario
def lnZ_PTP(  # noqa: C901 - inherited upstream scientific branches
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    plx: float,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    molusc_file: str = None,
    joint=None,
):
    """
    Calculates the marginal likelihood of the PTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    a = ((G * M_s * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    logg = np.log10(G * (M_s * Msun) / (R_s * Rsun) ** 2)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a * (1 - molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1 / M_s] = 0.1 / M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    masses_comp = qs_comp * M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
    )
    fluxratios_comp = _flux_fraction(masses_comp, np.array([M_s]))

    if molusc_file is None:
        if contrast_curve_file is None:
            delta_mags = 2.5 * np.log10(fluxratios_comp / (1 - fluxratios_comp))
            lnprior_companion = lnprior_bound_TP(
                M_s, plx, np.full(delta_mags.shape, 2.2)
            )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # evaluate each contrast curve in its own filter and adopt
            # the tightest constraint across all supplied curves
            cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
            delta_mags_list = []
            for cc_filt in cc_filts:
                fluxratios_comp_cc = _flux_fraction(
                    masses_comp, np.array([M_s]), cc_filt
                )
                delta_mags_list.append(
                    2.5 * np.log10(fluxratios_comp_cc / (1 - fluxratios_comp_cc))
                )
            seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
            lnprior_companion = lnprior_bound_TP(M_s, plx, seps)
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    rps = sample_rp(np.random.rand(N), np.full(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (rps * Rearth + R_s * Rsun) / a * e_corr

    coll = (rps * Rearth + R_s * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs_comp != 0.0)
        a_arr = a
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a_arr[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[mask],
            companion_is_host=False,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False) & (qs_comp[i] != 0.0):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=False,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": fluxratios_comp[idx],
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_PEB(  # noqa: C901 - inherited upstream scientific branches
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    plx: float,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    molusc_file: str = None,
    joint=None,
):
    """
    Calculates the marginal likelihood of the PEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    logg = np.log10(G * (M_s * Msun) / (R_s * Rsun) ** 2)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))
    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a * (1 - molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1 / M_s] = 0.1 / M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    masses = qs * M_s
    radii, Teffs = stellar_relations(masses, np.full(N, R_s), np.full(N, Teff))
    fluxratios = _flux_fraction(masses, np.array([M_s]))

    masses_comp = qs_comp * M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
    )
    fluxratios_comp = _flux_fraction(masses_comp, np.array([M_s]))

    if molusc_file is None:
        if contrast_curve_file is None:
            delta_mags = 2.5 * np.log10(fluxratios_comp / (1 - fluxratios_comp))
            lnprior_companion = lnprior_bound_EB(
                M_s, plx, np.full(delta_mags.shape, 2.2)
            )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # evaluate each contrast curve in its own filter and adopt
            # the tightest constraint across all supplied curves
            cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
            delta_mags_list = []
            for cc_filt in cc_filts:
                fluxratios_comp_cc = _flux_fraction(
                    masses_comp, np.array([M_s]), cc_filt
                )
                delta_mags_list.append(
                    2.5 * np.log10(fluxratios_comp_cc / (1 - fluxratios_comp_cc))
                )
            seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
            lnprior_companion = lnprior_bound_EB(M_s, plx, seps)
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (radii * Rsun + R_s * Rsun) / a * e_corr
    a_twin = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (2 * P_orb * 86400) ** 2) ** (
        1 / 3
    )
    Ptra_twin = (radii * Rsun + R_s * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (R_s * Rsun)

    coll = (radii * Rsun + R_s * Rsun) > a * (1 - eccs)
    coll_twin = (2 * R_s * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95) & (qs_comp != 0.0)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[mask],
            companion_is_host=False,
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95) & (qs_comp != 0.0)
        )
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[mask],
            companion_is_host=False,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] < 0.95)
                & (coll[i] == False)
                & (qs_comp[i] != 0)
            ):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=False,
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] >= 0.95)
                & (coll_twin[i] == False)
                & (qs_comp[i] != 0)
            ):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=False,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idx],
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, lnprior_companion, N_total=N)
    res_twin = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idx],
        "lnZ": lnZ,
    }
    return res, res_twin


@_scenario
def lnZ_STP(  # noqa: C901 - inherited upstream scientific branches
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    plx: float,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    molusc_file: str = None,
    joint=None,
):
    """
    Calculates the marginal likelihood of the STP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a * (1 - molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1 / M_s] = 0.1 / M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    masses_comp = qs_comp * M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
    )
    loggs_comp = np.log10(G * (masses_comp * Msun) / (radii_comp * Rsun) ** 2)
    fluxratios_comp = _flux_fraction(masses_comp, np.array([M_s]))

    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        _ldc_Teffs = ldc_T_Teffs
        _ldc_loggs = ldc_T_loggs
        _ldc_u1s = ldc_T_u1s
        _ldc_u2s = ldc_T_u2s
        ldc_at_Z = ldc_T[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
    else:
        ldc_Zs = ldc_K_Zs
        _ldc_Teffs = ldc_K_Teffs
        _ldc_loggs = ldc_K_loggs
        _ldc_u1s = ldc_K_u1s
        _ldc_u2s = ldc_K_u2s
        ldc_at_Z = ldc_K[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
    rounded_loggs_comp = np.round(loggs_comp / 0.5) * 0.5
    rounded_loggs_comp[rounded_loggs_comp < 3.5] = 3.5
    rounded_loggs_comp[rounded_loggs_comp > 5.0] = 5.0
    rounded_Teffs_comp = np.round(Teffs_comp / 250) * 250
    rounded_Teffs_comp[rounded_Teffs_comp < 3500] = 3500
    rounded_Teffs_comp[rounded_Teffs_comp > 10000] = 10000
    u1s_comp, u2s_comp = _fast_ldc_lookup(
        rounded_Teffs_comp,
        rounded_loggs_comp,
        Teffs_at_Z,
        loggs_at_Z,
        u1s_at_Z,
        u2s_at_Z,
    )

    if molusc_file is None:
        if contrast_curve_file is None:
            delta_mags = 2.5 * np.log10(fluxratios_comp / (1 - fluxratios_comp))
            lnprior_companion = lnprior_bound_TP(
                M_s, plx, np.full(delta_mags.shape, 2.2)
            )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # evaluate each contrast curve in its own filter and adopt
            # the tightest constraint across all supplied curves
            cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
            delta_mags_list = []
            for cc_filt in cc_filts:
                fluxratios_comp_cc = _flux_fraction(
                    masses_comp, np.array([M_s]), cc_filt
                )
                delta_mags_list.append(
                    2.5 * np.log10(fluxratios_comp_cc / (1 - fluxratios_comp_cc))
                )
            seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
            lnprior_companion = lnprior_bound_TP(M_s, plx, seps)
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    rps = sample_rp(np.random.rand(N), masses_comp, flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * masses_comp * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (rps * Rearth + radii_comp * Rsun) / a * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (radii_comp * Rsun)

    coll = (rps * Rearth + radii_comp * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs_comp != 0.0)
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            radii_comp[mask],
            u1s_comp[mask],
            u2s_comp[mask],
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[mask],
            companion_is_host=True,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False) & (qs_comp[i] != 0.0):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    radii_comp[i],
                    u1s_comp[i],
                    u2s_comp[i],
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=True,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": masses_comp[idx],
        "R_s": radii_comp[idx],
        "u1": u1s_comp[idx],
        "u2": u2s_comp[idx],
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": fluxratios_comp[idx],
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_SEB(  # noqa: C901 - inherited upstream scientific branches
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    plx: float,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    molusc_file: str = None,
    joint=None,
):
    """
    Calculates the marginal likelihood of the SEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))
    if molusc_file is None:
        qs_comp = sample_q_companion(np.random.rand(N), M_s)
    else:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a * (1 - molusc_e) > 10]
        qs_comp = molusc_df2["mass ratio"].values
        qs_comp[qs_comp < 0.1 / M_s] = 0.1 / M_s
        qs_comp = np.pad(qs_comp, (0, N - len(qs_comp)))

    masses_comp = qs_comp * M_s
    radii_comp, Teffs_comp = stellar_relations(
        masses_comp, np.full(N, R_s), np.full(N, Teff)
    )
    loggs_comp = np.log10(G * (masses_comp * Msun) / (radii_comp * Rsun) ** 2)
    fluxratios_comp = _flux_fraction(masses_comp, np.array([M_s]))

    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        _ldc_Teffs = ldc_T_Teffs
        _ldc_loggs = ldc_T_loggs
        _ldc_u1s = ldc_T_u1s
        _ldc_u2s = ldc_T_u2s
        ldc_at_Z = ldc_T[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
    else:
        ldc_Zs = ldc_K_Zs
        _ldc_Teffs = ldc_K_Teffs
        _ldc_loggs = ldc_K_loggs
        _ldc_u1s = ldc_K_u1s
        _ldc_u2s = ldc_K_u2s
        ldc_at_Z = ldc_K[(ldc_Zs == ldc_Zs[np.abs(ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
    rounded_loggs_comp = np.round(loggs_comp / 0.5) * 0.5
    rounded_loggs_comp[rounded_loggs_comp < 3.5] = 3.5
    rounded_loggs_comp[rounded_loggs_comp > 5.0] = 5.0
    rounded_Teffs_comp = np.round(Teffs_comp / 250) * 250
    rounded_Teffs_comp[rounded_Teffs_comp < 3500] = 3500
    rounded_Teffs_comp[rounded_Teffs_comp > 13000] = 13000
    u1s_comp, u2s_comp = _fast_ldc_lookup(
        rounded_Teffs_comp,
        rounded_loggs_comp,
        Teffs_at_Z,
        loggs_at_Z,
        u1s_at_Z,
        u2s_at_Z,
    )

    masses = qs * masses_comp
    radii, Teffs = stellar_relations(masses, radii_comp, Teffs_comp)
    fluxratios = _flux_fraction(masses, np.array([M_s]))

    if molusc_file is None:
        if contrast_curve_file is None:
            delta_mags = 2.5 * np.log10(
                (fluxratios_comp / (1 - fluxratios_comp))
                + (fluxratios / (1 - fluxratios))
            )
            lnprior_companion = lnprior_bound_EB(
                M_s, plx, np.full(delta_mags.shape, 2.2)
            )
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[delta_mags > 0.0] = -np.inf
        else:
            # evaluate each contrast curve in its own filter and adopt
            # the tightest constraint across all supplied curves
            cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
            delta_mags_list = []
            for cc_filt in cc_filts:
                fluxratios_cc = _flux_fraction(masses, np.array([M_s]), cc_filt)
                fluxratios_comp_cc = _flux_fraction(
                    masses_comp, np.array([M_s]), cc_filt
                )
                delta_mags_list.append(
                    2.5
                    * np.log10(
                        (fluxratios_comp_cc / (1 - fluxratios_comp_cc))
                        + (fluxratios_cc / (1 - fluxratios_cc))
                    )
                )
            seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
            lnprior_companion = lnprior_bound_EB(M_s, plx, seps)
            lnprior_companion[lnprior_companion > 0.0] = 0.0
            lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf
    else:
        lnprior_companion = np.zeros(N)

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * (masses_comp + masses) * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (
        1 / 3
    )
    Ptra = (radii * Rsun + radii_comp * Rsun) / a * e_corr
    a_twin = (
        (G * (masses_comp + masses) * Msun) / (4 * pi**2) * (2 * P_orb * 86400) ** 2
    ) ** (1 / 3)
    Ptra_twin = (radii * Rsun + radii_comp * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (radii_comp * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (radii_comp * Rsun)

    coll = (radii * Rsun + radii_comp * Rsun) > a * (1 - eccs)
    coll_twin = (2 * radii_comp * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95) & (qs_comp != 0.0)
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            radii_comp[mask],
            u1s_comp[mask],
            u2s_comp[mask],
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[mask],
            companion_is_host=True,
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95) & (qs_comp != 0.0)
        )
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            radii_comp[mask],
            u1s_comp[mask],
            u2s_comp[mask],
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[mask],
            companion_is_host=True,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] < 0.95)
                & (coll[i] == False)
                & (qs_comp[i] != 0.0)
            ):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    radii_comp[i],
                    u1s_comp[i],
                    u2s_comp[i],
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=True,
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] >= 0.95)
                & (coll_twin[i] == False)
                & (qs_comp[i] != 0.0)
            ):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    radii_comp[i],
                    u1s_comp[i],
                    u2s_comp[i],
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[i],
                    companion_is_host=True,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": masses_comp[idx],
        "R_s": radii_comp[idx],
        "u1": u1s_comp[idx],
        "u2": u2s_comp[idx],
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idx],
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, lnprior_companion, N_total=N)
    res_twin = {
        "M_s": masses_comp[idx],
        "R_s": radii_comp[idx],
        "u1": u1s_comp[idx],
        "u2": u2s_comp[idx],
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idx],
        "lnZ": lnZ,
    }
    return res, res_twin


@_scenario
def lnZ_DTP(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    Tmag: float,
    Jmag: float,
    Hmag: float,
    Kmag: float,
    trilegal_fname: str,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the DTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        trilegal_fname (string): File containing trilegal query results.
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    a = ((G * M_s * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    logg = np.log10(G * (M_s * Msun) / (R_s * Rsun) ** 2)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    (
        Tmags_comp,
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        Jmags_comp,
        Hmags_comp,
        Kmags_comp,
    ) = trilegal_results(trilegal_fname, Tmag)
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10 ** (delta_mags / 2.5) / (1 + 10 ** (delta_mags / 2.5))
    N_comp = Tmags_comp.shape[0]
    idxs = np.random.randint(0, N_comp - 1, N)

    if contrast_curve_file is None:
        delta_mags = 2.5 * np.log10(fluxratios_comp[idxs] / (1 - fluxratios_comp[idxs]))
        lnprior_companion = np.full(
            N, np.log((N_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2)
        )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        # evaluate each contrast curve in its own filter and adopt the
        # tightest constraint across all supplied curves
        cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
        delta_mags_by_filt = {"J": delta_Jmags, "H": delta_Hmags, "K": delta_Kmags}
        delta_mags_list = [
            delta_mags_by_filt.get(cc_filt, delta_mags)[idxs] for cc_filt in cc_filts
        ]
        seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
        lnprior_companion = lnprior_background(N_comp, seps)
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf

    rps = sample_rp(np.random.rand(N), np.full(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    Ptra = (rps * Rearth + R_s * Rsun) / a * e_corr

    coll = (rps * Rearth + R_s * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False)
        a_arr = a
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a_arr[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[idxs[mask]],
            companion_is_host=False,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=False,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": fluxratios_comp[idxs[idx]],
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_DEB(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Z: float,
    Tmag: float,
    Jmag: float,
    Hmag: float,
    Kmag: float,
    trilegal_fname: str,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the DEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        trilegal_fname (string): File containing trilegal query results.
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    logg = np.log10(G * (M_s * Msun) / (R_s * Rsun) ** 2)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    masses = qs * M_s
    radii, Teffs = stellar_relations(masses, np.full(N, R_s), np.full(N, Teff))
    fluxratios = _flux_fraction(masses, np.array([M_s]))

    (
        Tmags_comp,
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        Jmags_comp,
        Hmags_comp,
        Kmags_comp,
    ) = trilegal_results(trilegal_fname, Tmag)
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10 ** (delta_mags / 2.5) / (1 + 10 ** (delta_mags / 2.5))
    N_comp = Tmags_comp.shape[0]
    idxs = np.random.randint(0, N_comp - 1, N)

    if contrast_curve_file is None:
        delta_mags = 2.5 * np.log10(fluxratios_comp[idxs] / (1 - fluxratios_comp[idxs]))
        lnprior_companion = np.full(
            N, np.log((N_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2)
        )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        # evaluate each contrast curve in its own filter and adopt the
        # tightest constraint across all supplied curves
        cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
        delta_mags_by_filt = {"J": delta_Jmags, "H": delta_Hmags, "K": delta_Kmags}
        delta_mags_list = [
            delta_mags_by_filt.get(cc_filt, delta_mags)[idxs] for cc_filt in cc_filts
        ]
        seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
        lnprior_companion = lnprior_background(N_comp, seps)
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (radii * Rsun + R_s * Rsun) / a * e_corr
    a_twin = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (2 * P_orb * 86400) ** 2) ** (
        1 / 3
    )
    Ptra_twin = (radii * Rsun + R_s * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (R_s * Rsun)

    coll = (radii * Rsun + R_s * Rsun) > a * (1 - eccs)
    coll_twin = (2 * R_s * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[idxs[mask]],
            companion_is_host=False,
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[idxs[mask]],
            companion_is_host=False,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] < 0.95) & (coll[i] == False):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=False,
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] >= 0.95) & (coll_twin[i] == False):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=False,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idxs[idx]],
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, lnprior_companion, N_total=N)
    res_twin = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idxs[idx]],
        "lnZ": lnZ,
    }
    return res, res_twin


@_scenario
def lnZ_BTP(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Tmag: float,
    Jmag: float,
    Hmag: float,
    Kmag: float,
    trilegal_fname: str,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the BTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        trilegal_fname (string): File containing trilegal query results.
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    (
        Tmags_comp,
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        Jmags_comp,
        Hmags_comp,
        Kmags_comp,
    ) = trilegal_results(trilegal_fname, Tmag)
    radii_comp = np.sqrt(G * masses_comp * Msun / 10**loggs_comp) / Rsun
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10 ** (delta_mags / 2.5) / (1 + 10 ** (delta_mags / 2.5))
    N_comp = Tmags_comp.shape[0]
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_comp, u2s_comp = _fast_background_ldc(
        Teffs_comp, loggs_comp, Zs_comp, ldc_Teffs, ldc_loggs, ldc_Zs, ldc_u1s, ldc_u2s
    )
    idxs = np.random.randint(0, N_comp, N)

    if contrast_curve_file is None:
        delta_mags = 2.5 * np.log10(fluxratios_comp[idxs] / (1 - fluxratios_comp[idxs]))
        lnprior_companion = np.full(
            N, np.log((N_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2)
        )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        # evaluate each contrast curve in its own filter and adopt the
        # tightest constraint across all supplied curves
        cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
        delta_mags_by_filt = {"J": delta_Jmags, "H": delta_Hmags, "K": delta_Kmags}
        delta_mags_list = [
            delta_mags_by_filt.get(cc_filt, delta_mags)[idxs] for cc_filt in cc_filts
        ]
        seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
        lnprior_companion = lnprior_background(N_comp, seps)
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf

    rps = sample_rp(np.random.rand(N), masses_comp[idxs], flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * masses_comp[idxs] * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (rps * Rearth + radii_comp[idxs] * Rsun) / a * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (radii_comp[idxs] * Rsun)

    coll = (rps * Rearth + radii_comp[idxs] * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (loggs_comp[idxs] >= 3.5)
            & (Teffs_comp[idxs] <= 10000)
        )
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            radii_comp[idxs[mask]],
            u1s_comp[idxs[mask]],
            u2s_comp[idxs[mask]],
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[idxs[mask]],
            companion_is_host=True,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (loggs_comp[idxs[i]] >= 3.5)
                & (Teffs_comp[idxs[i]] <= 10000)
                & (coll[i] == False)
            ):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    radii_comp[idxs[i]],
                    u1s_comp[idxs[i]],
                    u2s_comp[idxs[i]],
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=True,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": masses_comp[idxs[idx]],
        "R_s": radii_comp[idxs[idx]],
        "u1": u1s_comp[idxs[idx]],
        "u2": u2s_comp[idxs[idx]],
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": fluxratios_comp[idxs[idx]],
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_BEB(  # noqa: C901 - inherited upstream scientific branches
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    M_s: float,
    R_s: float,
    Teff: float,
    Tmag: float,
    Jmag: float,
    Hmag: float,
    Kmag: float,
    trilegal_fname: str,
    contrast_curve_file=None,
    filt="TESS",
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the BEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        trilegal_fname (string): File containing trilegal query results.
        contrast_curve_file (str or list of str): Path to contrast curve
                         file, or a list of paths to combine more than
                         one contrast curve. A companion is ruled out if
                         any single curve rules it out.
        filt (str or list of str): Photometric filter(s) of the contrast
                         curve(s). Options are TESS, Vis, J, H, and K. A
                         single filter is applied to all curves.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), M_s)
    # Preserve the unused upstream draw and its RNG consumption.
    _qs_comp = sample_q_companion(np.random.rand(N), M_s)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    (
        Tmags_comp,
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        Jmags_comp,
        Hmags_comp,
        Kmags_comp,
    ) = trilegal_results(trilegal_fname, Tmag)
    radii_comp = np.sqrt(G * masses_comp * Msun / 10**loggs_comp) / Rsun
    delta_mags = Tmag - Tmags_comp
    delta_Jmags = Jmag - Jmags_comp
    delta_Hmags = Hmag - Hmags_comp
    delta_Kmags = Kmag - Kmags_comp
    fluxratios_comp = 10 ** (delta_mags / 2.5) / (1 + 10 ** (delta_mags / 2.5))
    fluxratios_comp_J = 10 ** (delta_Jmags / 2.5) / (1 + 10 ** (delta_Jmags / 2.5))
    fluxratios_comp_H = 10 ** (delta_Hmags / 2.5) / (1 + 10 ** (delta_Hmags / 2.5))
    fluxratios_comp_K = 10 ** (delta_Kmags / 2.5) / (1 + 10 ** (delta_Kmags / 2.5))
    N_comp = Tmags_comp.shape[0]
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_comp, u2s_comp = _fast_background_ldc(
        Teffs_comp, loggs_comp, Zs_comp, ldc_Teffs, ldc_loggs, ldc_Zs, ldc_u1s, ldc_u2s
    )
    idxs = np.random.randint(0, N_comp, N)

    masses = qs * masses_comp[idxs]
    radii, Teffs = stellar_relations(masses, radii_comp[idxs], Teffs_comp[idxs])
    fluxratios_comp_bound = _flux_fraction(masses_comp[idxs], np.array([M_s]))
    distance_correction = fluxratios_comp[idxs] / fluxratios_comp_bound
    fluxratios = _flux_fraction(masses, np.array([M_s])) * distance_correction
    if contrast_curve_file is None:
        delta_mags = 2.5 * np.log10(
            (fluxratios_comp[idxs] / (1 - fluxratios_comp[idxs]))
            + (fluxratios / (1 - fluxratios))
        )
        lnprior_companion = np.full(
            N, np.log((N_comp / 0.1) * (1 / 3600) ** 2 * 2.2**2)
        )
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[delta_mags > 0.0] = -np.inf
    else:
        # evaluate each contrast curve in its own filter and adopt the
        # tightest constraint across all supplied curves
        cc_files, cc_filts = parse_contrast_curves(contrast_curve_file, filt)
        fluxratios_comp_by_filt = {
            "J": fluxratios_comp_J,
            "H": fluxratios_comp_H,
            "K": fluxratios_comp_K,
        }
        delta_mags_list = []
        for cc_filt in cc_filts:
            # EB flux ratios in this contrast curve's filter
            fluxratios_comp_cc = fluxratios_comp_by_filt.get(cc_filt, fluxratios_comp)[
                idxs
            ]
            fluxratios_comp_bound_cc = _flux_fraction(
                masses_comp[idxs], np.array([M_s]), cc_filt
            )
            distance_correction_cc = fluxratios_comp_cc / fluxratios_comp_bound_cc
            fluxratios_cc = (
                _flux_fraction(masses, np.array([M_s]), cc_filt)
                * distance_correction_cc
            )
            delta_mags_list.append(
                2.5
                * np.log10(
                    (fluxratios_comp_cc / (1 - fluxratios_comp_cc))
                    + (fluxratios_cc / (1 - fluxratios_cc))
                )
            )
        seps = limiting_separation([np.abs(dm) for dm in delta_mags_list], cc_files)
        lnprior_companion = lnprior_background(N_comp, seps)
        lnprior_companion[lnprior_companion > 0.0] = 0.0
        lnprior_companion[np.any(np.array(delta_mags_list) > 0.0, axis=0)] = -np.inf

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = (
        (G * (masses_comp[idxs] + masses) * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2
    ) ** (1 / 3)
    Ptra = (radii * Rsun + radii_comp[idxs] * Rsun) / a * e_corr
    a_twin = (
        (G * (masses_comp[idxs] + masses) * Msun)
        / (4 * pi**2)
        * (2 * P_orb * 86400) ** 2
    ) ** (1 / 3)
    Ptra_twin = (radii * Rsun + radii_comp[idxs] * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (radii_comp[idxs] * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (radii_comp[idxs] * Rsun)

    coll = (radii * Rsun + radii_comp[idxs] * Rsun) > a * (1 - eccs)
    coll_twin = (2 * radii_comp[idxs] * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (qs < 0.95)
            & (loggs_comp[idxs] >= 3.5)
            & (Teffs_comp[idxs] <= 10000)
        )
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            radii_comp[idxs[mask]],
            u1s_comp[idxs[mask]],
            u2s_comp[idxs[mask]],
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[idxs[mask]],
            companion_is_host=True,
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min)
            & (coll_twin == False)
            & (qs >= 0.95)
            & (loggs_comp[idxs] >= 3.5)
            & (Teffs_comp[idxs] <= 10000)
        )
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            radii_comp[idxs[mask]],
            u1s_comp[idxs[mask]],
            u2s_comp[idxs[mask]],
            eccs[mask],
            argps[mask],
            companion_fluxratio=fluxratios_comp[idxs[mask]],
            companion_is_host=True,
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] < 0.95)
                & (loggs_comp[idxs[i]] >= 3.5)
                & (Teffs_comp[idxs[i]] <= 10000)
                & (coll[i] == False)
            ):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    radii_comp[idxs[i]],
                    u1s_comp[idxs[i]],
                    u2s_comp[idxs[i]],
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=True,
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] >= 0.95)
                & (loggs_comp[idxs[i]] >= 3.5)
                & (Teffs_comp[idxs[i]] <= 10000)
                & (coll_twin[i] == False)
            ):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    radii_comp[idxs[i]],
                    u1s_comp[idxs[i]],
                    u2s_comp[idxs[i]],
                    eccs[i],
                    argps[i],
                    companion_fluxratio=fluxratios_comp[idxs[i]],
                    companion_is_host=True,
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, lnprior_companion, N_total=N)
    res = {
        "M_s": masses_comp[idxs[idx]],
        "R_s": radii_comp[idxs[idx]],
        "u1": u1s_comp[idxs[idx]],
        "u2": u2s_comp[idxs[idx]],
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idxs[idx]],
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, lnprior_companion, N_total=N)
    res_twin = {
        "M_s": masses_comp[idxs[idx]],
        "R_s": radii_comp[idxs[idx]],
        "u1": u1s_comp[idxs[idx]],
        "u2": u2s_comp[idxs[idx]],
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": fluxratios_comp[idxs[idx]],
        "lnZ": lnZ,
    }
    return res, res_twin


@_scenario
def lnZ_NTP_unknown(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    Tmag: float,
    trilegal_fname: str,
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the NTP scenario for
    a star of unknown properties.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        Tmag (float): Target star TESS magnitude.
        trilegal_fname (string): File containing trilegal query results.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    (
        Tmags_nearby,
        masses_nearby,
        loggs_nearby,
        Teffs_nearby,
        Zs_nearby,
        Jmags_nearby,
        Hmags_nearby,
        Kmags_nearby,
    ) = trilegal_results(trilegal_fname, Tmag)
    mask = (Tmag - 1 < Tmags_nearby) & (Tmags_nearby < Tmag + 1)
    Tmags_possible = Tmags_nearby[mask]
    masses_possible = masses_nearby[mask]
    loggs_possible = loggs_nearby[mask]
    Teffs_possible = Teffs_nearby[mask]
    Zs_possible = Zs_nearby[mask]
    radii_possible = np.sqrt(G * masses_possible * Msun / 10**loggs_possible) / Rsun
    N_possible = Tmags_possible.shape[0]
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_possible = np.zeros(N_possible)
    u2s_possible = np.zeros(N_possible)
    for i in range(N_possible):
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teffs_possible[i]))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - loggs_possible[i]))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs - Zs_possible[i]))]
        mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        u1s_possible[i], u2s_possible[i] = ldc_u1s[mask].item(), ldc_u2s[mask].item()
    if N_possible > 0:
        idxs = np.random.randint(0, N_possible, N)
    else:
        res = {
            "M_s": 0,
            "R_s": 0,
            "u1": 0,
            "u2": 0,
            "P_orb": 0,
            "inc": 0,
            "R_p": 0,
            "ecc": 0,
            "argp": 0,
            "M_EB": 0,
            "R_EB": 0,
            "fluxratio_EB": 0,
            "fluxratio_comp": 0,
            "lnZ": -np.inf,
        }
        return res

    rps = sample_rp(np.random.rand(N), masses_possible[idxs], flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * masses_possible[idxs] * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (
        1 / 3
    )
    Ptra = (rps * Rearth + radii_possible[idxs] * Rsun) / a * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (radii_possible[idxs] * Rsun)

    coll = (rps * Rearth + radii_possible[idxs] * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (loggs_possible[idxs] >= 3.5)
            & (Teffs_possible[idxs] <= 10000)
        )
        companion_fluxratio = np.zeros(N)
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            radii_possible[idxs[mask]],
            u1s_possible[idxs[mask]],
            u2s_possible[idxs[mask]],
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (loggs_possible[idxs[i]] >= 3.5)
                & (Teffs_possible[idxs[i]] <= 10000)
                & (coll[i] == False)
            ):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    radii_possible[idxs[i]],
                    u1s_possible[idxs[i]],
                    u2s_possible[idxs[i]],
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, None, N_total=N)
    res = {
        "M_s": masses_possible[idxs[idx]],
        "R_s": radii_possible[idxs[idx]],
        "u1": u1s_possible[idxs[idx]],
        "u2": u2s_possible[idxs[idx]],
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_NEB_unknown(  # noqa: C901 - inherited upstream scientific branches
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    Tmag: float,
    trilegal_fname: str,
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the NEB scenario for a star
    of unknown properties.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        Tmag (float): Target star TESS magnitude.
        trilegal_fname (string): File containing trilegal query results.
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), 1.0)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    (
        Tmags_nearby,
        masses_nearby,
        loggs_nearby,
        Teffs_nearby,
        Zs_nearby,
        Jmags_nearby,
        Hmags_nearby,
        Kmags_nearby,
    ) = trilegal_results(trilegal_fname, Tmag)
    mask = (Tmag - 1 < Tmags_nearby) & (Tmags_nearby < Tmag + 1)
    Tmags_possible = Tmags_nearby[mask]
    masses_possible = masses_nearby[mask]
    loggs_possible = loggs_nearby[mask]
    Teffs_possible = Teffs_nearby[mask]
    Zs_possible = Zs_nearby[mask]
    radii_possible = np.sqrt(G * masses_possible * Msun / 10**loggs_possible) / Rsun
    N_possible = Tmags_possible.shape[0]
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    u1s_possible = np.zeros(N_possible)
    u2s_possible = np.zeros(N_possible)
    for i in range(N_possible):
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teffs_possible[i]))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - loggs_possible[i]))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs - Zs_possible[i]))]
        mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        u1s_possible[i], u2s_possible[i] = ldc_u1s[mask].item(), ldc_u2s[mask].item()
    if N_possible > 0:
        idxs = np.random.randint(0, N_possible, N)
    else:
        res = {
            "M_s": 0,
            "R_s": 0,
            "u1": 0,
            "u2": 0,
            "P_orb": 0,
            "inc": 0,
            "b": 0,
            "R_p": 0,
            "ecc": 0,
            "argp": 0,
            "M_EB": 0,
            "R_EB": 0,
            "fluxratio_EB": 0,
            "fluxratio_comp": 0,
            "lnZ": -np.inf,
        }
        return res

    masses = qs * masses_possible[idxs]
    radii, Teffs = stellar_relations(masses, radii_possible[idxs], Teffs_possible[idxs])
    fluxratios = _flux_fraction(masses, masses_possible[idxs])

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = (
        (G * (masses_possible[idxs] + masses) * Msun)
        / (4 * pi**2)
        * (P_orb * 86400) ** 2
    ) ** (1 / 3)
    Ptra = (radii * Rsun + radii_possible[idxs] * Rsun) / a * e_corr
    a_twin = (
        (G * (masses_possible[idxs] + masses) * Msun)
        / (4 * pi**2)
        * (2 * P_orb * 86400) ** 2
    ) ** (1 / 3)
    Ptra_twin = (radii * Rsun + radii_possible[idxs] * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (radii_possible[idxs] * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (radii_possible[idxs] * Rsun)

    coll = (radii * Rsun + radii_possible[idxs] * Rsun) > a * (1 - eccs)
    coll_twin = (2 * radii_possible[idxs] * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min)
            & (coll == False)
            & (qs < 0.95)
            & (loggs_possible[idxs] >= 3.5)
            & (Teffs_possible[idxs] <= 10000)
        )
        companion_fluxratio = np.zeros(N)
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            radii_possible[idxs[mask]],
            u1s_possible[idxs[mask]],
            u2s_possible[idxs[mask]],
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (
            (incs >= inc_min)
            & (coll_twin == False)
            & (qs >= 0.95)
            & (loggs_possible[idxs] >= 3.5)
            & (Teffs_possible[idxs] <= 10000)
        )
        companion_fluxratio = np.zeros(N)
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            radii_possible[idxs[mask]],
            u1s_possible[idxs[mask]],
            u2s_possible[idxs[mask]],
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] < 0.95)
                & (loggs_possible[idxs[i]] >= 3.5)
                & (Teffs_possible[idxs[i]] <= 10000)
                & (coll[i] == False)
            ):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    radii_possible[idxs[i]],
                    u1s_possible[idxs[i]],
                    u2s_possible[idxs[i]],
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (
                (incs[i] >= inc_min)
                & (qs[i] >= 0.95)
                & (loggs_possible[idxs[i]] >= 3.5)
                & (Teffs_possible[idxs[i]] <= 10000)
                & (coll_twin[i] == False)
            ):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    radii_possible[idxs[i]],
                    u1s_possible[idxs[i]],
                    u2s_possible[idxs[i]],
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, None, N_total=N)
    res = {
        "M_s": masses_possible[idxs[idx]],
        "R_s": radii_possible[idxs[idx]],
        "u1": u1s_possible[idxs[idx]],
        "u2": u2s_possible[idxs[idx]],
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, None, N_total=N)
    res_twin = {
        "M_s": masses_possible[idxs[idx]],
        "R_s": radii_possible[idxs[idx]],
        "u1": u1s_possible[idxs[idx]],
        "u2": u2s_possible[idxs[idx]],
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    return res, res_twin


@_scenario
def lnZ_NTP_evolved(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    R_s: float,
    Teff: float,
    Z: float,
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the NTP scenario for
    subgiant stars.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    logg = 3.0
    M_s = (10**logg) * (R_s * Rsun) ** 2 / G / Msun
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    rps = sample_rp(np.random.rand(N), np.full(N, M_s), flatpriors)
    incs = sample_inc(np.random.rand(N))
    eccs = sample_ecc(np.random.rand(N), planet=True, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * M_s * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (rps * Rearth + R_s * Rsun) / a * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)

    coll = (rps * Rearth + R_s * Rsun) > a * (1 - eccs)

    lnL = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False)
        a_arr = a
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = _log_likelihood(
            lnL_TP_p,
            joint,
            time,
            flux,
            sigma,
            rps[mask],
            P_orb[mask],
            incs[mask],
            a_arr[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (coll[i] == False):
                lnL[i] = _log_likelihood(
                    lnL_TP,
                    joint,
                    time,
                    flux,
                    sigma,
                    rps[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, None, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": rps[idx],
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": np.zeros(N_samples),
        "R_EB": np.zeros(N_samples),
        "fluxratio_EB": np.zeros(N_samples),
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    return res


@_scenario
def lnZ_NEB_evolved(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    P_orb: float,
    R_s: float,
    Teff: float,
    Z: float,
    N: int = 1000000,
    parallel: bool = False,
    mission: str = "TESS",
    flatpriors: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    joint=None,
):
    """
    Calculates the marginal likelihood of the NEB scenario
    for subgiant stars.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        parallel (bool): Whether or not to simulate light curves
                         in parallel.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # sample orbital periods if range is given
    if type(P_orb) not in [float, int]:
        P_orb = np.random.uniform(low=P_orb[0], high=P_orb[-1], size=N)
    else:
        P_orb = np.full(N, P_orb)

    logg = 3.0
    M_s = (10**logg) * (R_s * Rsun) ** 2 / G / Msun
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs - Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs - Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs - logg))]
    mask = (ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
    u1, u2 = ldc_u1s[mask].item(), ldc_u2s[mask].item()

    incs = sample_inc(np.random.rand(N))
    qs = sample_q(np.random.rand(N), 1.0)
    eccs = sample_ecc(np.random.rand(N), planet=False, P_orb=np.mean(P_orb))
    argps = sample_w(np.random.rand(N))

    masses = qs * M_s
    radii, Teffs = stellar_relations(masses, np.full(N, R_s), np.full(N, Teff))
    fluxratios = _flux_fraction(masses, np.array([M_s]))

    e_corr = (1 + eccs * np.sin(argps * pi / 180)) / (1 - eccs**2)
    a = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (P_orb * 86400) ** 2) ** (1 / 3)
    Ptra = (radii * Rsun + R_s * Rsun) / a * e_corr
    a_twin = ((G * (M_s + masses) * Msun) / (4 * pi**2) * (2 * P_orb * 86400) ** 2) ** (
        1 / 3
    )
    Ptra_twin = (R_s * Rsun + R_s * Rsun) / a_twin * e_corr

    r = a * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b = r * np.cos(incs * pi / 180) / (R_s * Rsun)
    r_twin = a_twin * (1 - eccs**2) / (1 + eccs * np.sin(argps * np.pi / 180))
    b_twin = r_twin * np.cos(incs * pi / 180) / (R_s * Rsun)

    coll = (radii * Rsun + R_s * Rsun) > a * (1 - eccs)
    coll_twin = (2 * R_s * Rsun) > a_twin * (1 - eccs)

    lnL = np.full(N, -np.inf)
    lnL_twin = np.full(N, -np.inf)
    if parallel:
        inc_min = np.full(N, 90.0)
        inc_min[Ptra <= 1.0] = np.arccos(Ptra[Ptra <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll == False) & (qs < 0.95)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        companion_fluxratio = np.zeros(N)
        lnL[mask] = _log_likelihood(
            lnL_EB_p,
            joint,
            time,
            flux,
            sigma,
            radii[mask],
            fluxratios[mask],
            P_orb[mask],
            incs[mask],
            a[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
        inc_min = np.full(N, 90.0)
        inc_min[Ptra_twin <= 1.0] = np.arccos(Ptra_twin[Ptra_twin <= 1.0]) * 180.0 / pi
        mask = (incs >= inc_min) & (coll_twin == False) & (qs >= 0.95)
        R_s_arr = np.full(np.count_nonzero(mask), R_s)
        u1_arr = np.full(np.count_nonzero(mask), u1)
        u2_arr = np.full(np.count_nonzero(mask), u2)
        companion_fluxratio = np.zeros(N)
        lnL_twin[mask] = _log_likelihood(
            lnL_EB_twin_p,
            joint,
            time,
            flux,
            sigma,
            R_s,
            fluxratios[mask],
            2 * P_orb[mask],
            incs[mask],
            a_twin[mask],
            R_s_arr,
            u1_arr,
            u2_arr,
            eccs[mask],
            argps[mask],
            companion_fluxratio=companion_fluxratio[mask],
            exptime=exptime,
            nsamples=nsamples,
        )
    else:
        for i in range(N):
            if Ptra[i] <= 1:
                inc_min = np.arccos(Ptra[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] < 0.95) & (coll[i] == False):
                lnL[i] = _log_likelihood(
                    lnL_EB,
                    joint,
                    time,
                    flux,
                    sigma,
                    radii[i],
                    fluxratios[i],
                    P_orb[i],
                    incs[i],
                    a[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )
            if Ptra_twin[i] <= 1:
                inc_min = np.arccos(Ptra_twin[i]) * 180 / pi
            else:
                continue
            if (incs[i] >= inc_min) & (qs[i] >= 0.95) & (coll_twin[i] == False):
                lnL_twin[i] = _log_likelihood(
                    lnL_EB_twin,
                    joint,
                    time,
                    flux,
                    sigma,
                    R_s,
                    fluxratios[i],
                    2 * P_orb[i],
                    incs[i],
                    a_twin[i],
                    R_s,
                    u1,
                    u2,
                    eccs[i],
                    argps[i],
                    exptime=exptime,
                    nsamples=nsamples,
                )

    N_samples = 100
    idx = _fast_top(-lnL, N_samples)
    lnZ = _fast_log_mean_exp(lnL, None, N_total=N)
    res = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": P_orb[idx],
        "inc": incs[idx],
        "b": b[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": radii[idx],
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    N_samples = 100
    idx = _fast_top(-lnL_twin, N_samples)
    lnZ = _fast_log_mean_exp(lnL_twin, None, N_total=N)
    res_twin = {
        "M_s": np.full(N_samples, M_s),
        "R_s": np.full(N_samples, R_s),
        "u1": np.full(N_samples, u1),
        "u2": np.full(N_samples, u2),
        "P_orb": 2 * P_orb[idx],
        "inc": incs[idx],
        "b": b_twin[idx],
        "R_p": np.zeros(N_samples),
        "ecc": eccs[idx],
        "argp": argps[idx],
        "M_EB": masses[idx],
        "R_EB": np.full(N_samples, R_s),
        "fluxratio_EB": fluxratios[idx],
        "fluxratio_comp": np.zeros(N_samples),
        "lnZ": lnZ,
    }
    return res, res_twin
