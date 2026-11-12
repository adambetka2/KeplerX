"""Evidence that a detection is an eclipsing binary, not a planet."""

import math

import numpy as np

from vetter.upstream.dump import make_transit_mask
from vetter.upstream.vetting import get_p_tdur_t0

# Phases within this many transit durations of the transit are its wings.
NEAR_PRIMARY = 2.0
# Period ratios (to repeat_rule's tolerance) of detections that are the same signal.
SAME_SIGNAL_RATIOS = (1.0, 2.0, 3.0, 1 / 2, 1 / 3)
SAME_SIGNAL_TOLERANCE = 5e-4
# A dip at half phase this close, and this close in MES, is the transit at half the period.
HALF_PHASE_DURATIONS = 2.0
HALF_DEPTH_MES = 3.0
# A secondary at phase j/k (k up to this), as deep as the transit event by event, is the planet's own transit.
OWN_TRANSIT_MAX_K = 4
OWN_TRANSIT_SIGMA = 3.0
# Amplitude per unit q (R*/a)^3: 0.15 (15 + u)(1 + tau) / (3 - u), u = 0.6, tau = 0.4.
ELLIPSOIDAL_COEFFICIENT = 1.37
# KBonus flattening's transfer of a sinusoid by its period (days).
TRANSFER = ((0.25, 1.0), (1.0, 0.996), (1.5, 0.986), (2.0, 0.971), (3.0, 0.927),
            (4.0, 0.484), (6.0, 0.005))  # fmt: skip
MAX_MODULATION_DAYS = 3.0
# Transits need this share of their in-transit cadences to have a depth.
MIN_TRANSIT_COVERAGE = 0.5


def binary_evidence(vetter, index, metrics):
    """Secondary-eclipse measurements of detection `index` with every other detection masked."""
    P, width, t0 = get_p_tdur_t0(vetter.tces[index])
    time = vetter.dump.lc.time
    others = np.ones(len(time), dtype=bool)
    for i, row in enumerate(vetter.tces):
        p, w, e = get_p_tdur_t0(row)
        same = any(
            abs(p / (P * ratio) - 1.0) < SAME_SIGNAL_TOLERANCE
            for ratio in SAME_SIGNAL_RATIOS
        )
        if i != index and not same:
            others &= make_transit_mask(time, p, e, w)
    _, phase, mes = vetter._calc_mes(
        index, calc_ses=True, mask_tce=True, use_mask=others
    )
    phase = (np.asarray(phase) - 0.25 + 0.5) % 1.0 - 0.5
    mes = np.where(np.asarray(mes) == 0, np.nan, np.asarray(mes, dtype=float))
    near = np.abs(phase) < NEAR_PRIMARY * width / P
    mes[near] = np.nan
    if not np.isfinite(mes).any():
        return {"binary_secondary_mes": None}
    top = int(np.nanargmax(mes))
    secondary, where = float(mes[top]), float(phase[top])
    noise = float(1.4826 * np.nanmedian(np.abs(mes - np.nanmedian(mes))))
    offset = np.abs((phase - where + 0.5) % 1.0 - 0.5)
    elsewhere = mes[offset >= NEAR_PRIMARY * width / P]
    primary = metrics.get("max_mes")
    half = abs(abs(where) - 0.5) < HALF_PHASE_DURATIONS * width / P and (
        primary is not None and abs(secondary - primary) < HALF_DEPTH_MES
    )
    a_rs = 4.2066 * P ** (2 / 3) * vetter.dump.star_density ** (1 / 3)
    albedo = (
        secondary / primary * a_rs**2
        if primary and primary > 0 and math.isfinite(a_rs)
        else None
    )
    lc = vetter.dump.lc
    use = lc.mask & others & np.isfinite(lc.flux)
    return {
        "binary_secondary_mes": secondary,
        "binary_secondary_phase": where,
        "binary_secondary_noise": noise,
        "binary_next_dip_mes": float(np.nanmax(elsewhere))
        if np.isfinite(elsewhere).any()
        else None,
        "binary_brightening_mes": float(-np.nanmin(mes)),
        "binary_secondary_is_half_period": bool(half),
        "binary_occultation_albedo": albedo,
        **secondary_events(lc.time, lc.flux, use, P, t0, width, lc.exptime, where),
        **odd_even(lc.time, lc.flux, use, P, t0, width, lc.exptime),
        **ellipsoidal(lc.time, lc.flux, use, P, t0, width, where, a_rs),
    }


def transit_depths(time, flux, use, period, epoch, width, exposure):
    """Each transit's number and depth."""
    offset = (time - epoch + period / 2) % period - period / 2
    number = np.round((time - epoch) / period).astype(np.int64)
    near = use & (np.abs(offset) < 2.5 * width)
    inside = near & (np.abs(offset) < width / 2)
    around = near & (np.abs(offset) >= width)
    if not inside.any():
        return np.array([], dtype=np.int64), np.array([])
    base = number.min()
    size = number.max() - base + 1
    n_in = np.bincount(number[inside] - base, minlength=size)
    s_in = np.bincount(number[inside] - base, flux[inside], minlength=size)
    n_out = np.bincount(number[around] - base, minlength=size)
    s_out = np.bincount(number[around] - base, flux[around], minlength=size)
    keep = (n_in >= MIN_TRANSIT_COVERAGE * width / exposure) & (n_out > 0)
    depth = s_out[keep] / n_out[keep] - s_in[keep] / n_in[keep]
    return np.flatnonzero(keep) + base, depth


def secondary_events(time, flux, use, period, epoch, width, exposure, phase):
    """The secondary at `phase` in the light curve's own events."""
    out = {
        "binary_secondary_event_sigma": None,
        "binary_secondary_halves_sigma": None,
        "binary_secondary_is_own_transit": False,
    }
    at = epoch + phase * period
    number, depth = transit_depths(time, flux, use, period, at, width, exposure)
    scatter = float(np.std(depth, ddof=1)) if len(depth) >= 2 else 0.0
    if not scatter > 0:
        return out
    mean, error = float(np.mean(depth)), scatter / math.sqrt(len(depth))
    out["binary_secondary_event_sigma"] = mean / error
    when = at + number * period
    middle = 0.5 * (time.min() + time.max())
    halves = [depth[when < middle], depth[when >= middle]]
    if all(len(half) for half in halves):
        out["binary_secondary_halves_sigma"] = min(
            float(np.mean(half)) / (scatter / math.sqrt(len(half))) for half in halves
        )
    tolerance = HALF_PHASE_DURATIONS * width / period
    if any(
        abs((phase - j / k + 0.5) % 1.0 - 0.5) < tolerance
        for k in range(2, OWN_TRANSIT_MAX_K + 1)
        for j in range(1, k)
    ):
        _, primary = transit_depths(time, flux, use, period, epoch, width, exposure)
        if len(primary) >= 2:
            p_error = float(np.std(primary, ddof=1)) / math.sqrt(len(primary))
            out["binary_secondary_is_own_transit"] = bool(
                abs(mean - float(np.mean(primary)))
                < OWN_TRANSIT_SIGMA * math.hypot(error, p_error)
            )
    return out


def odd_even(time, flux, use, period, epoch, width, exposure):
    """Odd against even transit depths, with errors from each set's own scatter."""
    number, depth = transit_depths(time, flux, use, period, epoch, width, exposure)
    odd, even = depth[number % 2 == 1], depth[number % 2 == 0]
    out = {"binary_odd_transits": int(len(odd)), "binary_even_transits": int(len(even))}
    if len(odd) < 3 or len(even) < 3:
        return out | {"binary_odd_even_sigma": None}
    error = math.sqrt(np.var(odd, ddof=1) / len(odd) + np.var(even, ddof=1) / len(even))
    difference = abs(float(np.mean(odd) - np.mean(even)))
    return out | {"binary_odd_even_sigma": difference / error if error > 0 else None}


def transfer(days):
    """KBonus flattening's transfer of a sinusoid of this period."""
    periods, values = zip(*TRANSFER, strict=True)
    return float(np.interp(math.log(days), np.log(periods), values))


def ellipsoidal(time, flux, use, period, epoch, width, secondary_phase, a_rs):
    """The coherent -A cos(4 pi phase) term of the out-of-eclipse flux."""
    if period / 2 > MAX_MODULATION_DAYS:
        return {"binary_ellipsoidal_mass_ratio": None}
    phase = ((time - epoch) / period) % 1.0
    clear = width / period
    eclipses = [0.0, 0.5, secondary_phase % 1.0]
    out = use & np.all(
        [np.abs((phase - e + 0.5) % 1.0 - 0.5) > clear for e in eclipses], axis=0
    )
    if out.sum() < 100:
        return {"binary_ellipsoidal_mass_ratio": None}

    def fit(sel):
        x = 2 * math.pi * phase[sel]
        design = np.column_stack(
            [np.ones(len(x)), np.cos(x), np.sin(x), np.cos(2 * x), np.sin(2 * x)]
        )
        coef, *_ = np.linalg.lstsq(design, flux[sel], rcond=None)
        residual = flux[sel] - design @ coef
        scale = float(np.std(residual)) * math.sqrt(
            np.linalg.inv(design.T @ design)[3, 3]
        )
        return -float(coef[3]), scale

    amplitude, error = fit(out)
    middle = np.median(time[out])
    halves = [fit(out & (time < middle)), fit(out & (time >= middle))]
    agree = all(a > 3 * e for a, e in halves) and abs(
        halves[0][0] - halves[1][0]
    ) < 3 * (math.hypot(halves[0][1], halves[1][1]))
    corrected = amplitude / transfer(period / 2)
    return {
        "binary_ellipsoidal_amplitude_ppm": corrected * 1e6,
        "binary_ellipsoidal_sigma": amplitude / error if error > 0 else None,
        "binary_ellipsoidal_halves_agree": bool(agree),
        "binary_ellipsoidal_mass_ratio": corrected
        / (ELLIPSOIDAL_COEFFICIENT * a_rs**-3)
        if math.isfinite(a_rs)
        else None,
    }
