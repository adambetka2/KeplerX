"""Injected signals on delivered KBonus PSF Flat, through an emulated flattening."""

import math

import batman
import numpy as np

from photometry.flattening import SMOOTHER, robust_sigma, trend  # noqa: F401


def flattened(raw, time, quarter, valid, signal, smoother=SMOOTHER):
    """Relative flux change a multiplicative `signal` leaves in delivered Flat."""
    result = np.asarray(signal, float).copy()
    touched = valid & (np.abs(signal - 1) > 1e-12)
    groups = np.unique(quarter[touched])
    if not len(groups):
        return result
    scales = {g: np.median(raw[valid & (quarter == g)]) for g in groups}
    groups = np.array([g for g in groups if np.isfinite(scales[g]) and scales[g] > 0])
    if not len(groups):
        return result
    v = np.flatnonzero(valid & np.isin(quarter, groups))
    t, q = time[v], quarter[v]
    scale = np.ones(len(v))
    for g in groups:  # the smoother sees each quarter at its own normalisation
        scale[q == g] = scales[g]
    r = raw[v] / scale
    before = trend(t, r, q, groups, smoother)
    after = trend(t, r * signal[v], q, groups, smoother)
    result[v] = signal[v] * before / after
    return result


def supersampling(duration_days, exposure_days):
    """Subexposures so that ingress and egress are resolved (odd, 7-41)."""
    n = int(math.ceil(12 * exposure_days / max(duration_days, 1e-3)))
    return int(np.clip(n | 1, 7, 41))


def ttv_shift(time, body, epoch):
    """Per-cadence offset of the nearest transit's time (sinusoidal TTV)."""
    ttv = body.get("ttv")
    if not ttv:
        return np.zeros(len(time))
    n = np.rint((time - epoch) / body["period_days"])
    return ttv["amplitude_days"] * np.sin(
        2 * math.pi * n * body["period_days"] / ttv["super_period_days"]
        + ttv["phase_rad"]
    )


def batman_params(body, epoch):
    p = batman.TransitParams()
    p.t0 = epoch
    p.per = body["period_days"]
    p.rp = body["radius_ratio"]
    p.a = body["a_rs"]
    p.inc = body["inc_deg"]
    p.ecc = body["e"]
    p.w = math.degrees(body["omega_rad"])
    p.limb_dark = "nonlinear"
    p.u = list(body["limb_darkening"])
    return p


def light_curve(params, time, exposure, supersample, transittype="primary"):
    """batman light curve, with a fixed step where its step search does not converge."""
    try:
        model = batman.TransitModel(
            params,
            time,
            transittype=transittype,
            exp_time=exposure,
            supersample_factor=supersample,
        )
    except Exception:
        model = batman.TransitModel(
            params, time, transittype=transittype, exp_time=exposure,
            supersample_factor=supersample, fac=1e-3,
        )  # fmt: skip
    return model.light_curve(params)


def near(time, centre, period, half):
    return np.abs((time - centre + period / 2) % period - period / 2) <= half


def transit_flux(time, exposure, planet, epoch):
    """Relative flux of one transiting planet; TTVs shift each transit."""
    flux = np.ones(len(time))
    finite = np.isfinite(time)
    shifted = time - ttv_shift(np.where(finite, time, epoch), planet, epoch)
    pick = finite & near(
        shifted, epoch, planet["period_days"], planet["duration_days"] / 2 + exposure
    )
    if pick.any():
        params = batman_params(planet, epoch)
        flux[pick] = light_curve(
            params,
            shifted[pick],
            exposure,
            supersampling(planet["duration_days"], exposure),
        )
    return flux


def secondary_time(binary, epoch):
    """Mid-time of the secondary eclipse of an eccentric orbit."""
    params = batman_params(binary, epoch)
    params.fp = binary["light_ratio"]
    return float(batman.TransitModel(params, np.array([epoch])).get_t_secondary(params))


def binary_flux(time, exposure, binary, epoch):
    """Host plus companion star."""
    finite = np.isfinite(time)
    t = np.where(finite, time, epoch)
    period = binary["period_days"]
    sub = supersampling(binary["duration_days"], exposure)
    params = batman_params(binary, epoch)
    host = np.ones(len(t))
    pick = finite & near(t, epoch, period, binary["duration_days"] / 2 + exposure)
    if pick.any():
        host[pick] = light_curve(params, t[pick], exposure, sub)
    secondary = secondary_time(binary, epoch)
    e, w = binary["e"], binary["omega_rad"]
    stretch = (1 + e * math.sin(w)) / max(1 - e * math.sin(w), 1e-3)
    companion = np.full(len(t), binary["light_ratio"])
    pick = finite & near(
        t, secondary, period, stretch * binary["duration_days"] / 2 + exposure
    )
    if pick.any():
        params.fp = binary["light_ratio"]
        params.t_secondary = secondary
        companion[pick] = light_curve(params, t[pick], exposure, sub, "secondary") - 1
    ellipsoidal = -binary["ellipsoidal_amplitude"] * np.cos(
        4 * math.pi * (t - epoch) / period
    )
    system = (host + companion + ellipsoidal) / (1 + binary["light_ratio"])
    flux = np.ones(len(time))
    flux[finite] = 1 + binary["dilution"] * (system[finite] - 1)
    return flux
