"""Physical signals for the KBonus calibration sample, with population weights."""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

G_SI = 6.674e-11
R_SUN = 6.957e8
M_SUN = 1.989e30
R_EARTH = 6.371e6
DAY = 86400.0
SOLAR_LOGG = 4.438067627303133
# Mamajek's dwarf sequence (copied from the K-dwarf selection of 6 October 2026).
ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = ROOT / "data/calibration-sample-design/mamajek-dwarf-sequence.txt"
SEQUENCE_FIXTURE = ROOT / "tests/fixtures/dwarf-sequence.txt"


def stellar_class(teff, logg):
    if not np.isfinite(teff) or not np.isfinite(logg):
        return "unknown"
    if logg < 3.5:
        return "giant"
    if logg < 4.1:
        return "subgiant"
    return ("M", "K", "G", "F", "hot")[
        int(np.searchsorted([3900, 5300, 6000, 7500], teff))
    ]


@lru_cache(maxsize=1)
def dwarf_sequence():
    """(Teff, mass) of the dwarf sequence, sorted by Teff."""
    rows = []
    source = SEQUENCE if SEQUENCE.is_file() else SEQUENCE_FIXTURE
    for line in source.read_text().splitlines():
        values = line.split()
        if values and values[0].startswith(("F", "G", "K", "M")) and len(values) == 32:
            rows.append([float(values[1]), float(values[30])])
    return np.asarray(sorted(rows))


def catalogue_stellar(teff, logg):
    """The campaign's declared stellar inputs."""
    missing = any(x is None or not np.isfinite(x) for x in (teff, logg))
    if missing:
        teff, logg = 5772.0, SOLAR_LOGG
    reference = dwarf_sequence()
    mass = float(np.interp(teff, reference[:, 0], reference[:, 1]))
    radius = float(np.sqrt(mass * 10 ** (SOLAR_LOGG - logg)))
    return {
        "teff": float(teff),
        "logg": float(logg),
        "density_solar": mass / radius**3,
        "mass_solar": mass,
        "radius_solar": radius,
        "luminosity_solar": radius**2 * (teff / 5772) ** 4,
        "missing_stellar_inputs": missing,
        "basis": (
            "Explicit solar surrogate: native Gaia temperature/gravity missing"
            if missing
            else "Frozen Gaia Teff/logg; fiducial sequence mass and gravity-derived radius, not measured mass/radius"
        ),
    }


def true_host(catalogue, rng, g_mag=None):
    """Physical truth for a host whose campaign inputs are `catalogue`."""
    missing = catalogue.get("missing_stellar_inputs", False)
    faint = float(
        np.clip(
            (g_mag if g_mag is not None and np.isfinite(g_mag) else 17.9) - 15.0,
            0.0,
            4.0,
        )
    )
    widen = 1.3 if catalogue["logg"] < 4.0 else 1.0
    teff_sd = 600.0 if missing else widen * (0.03 + 0.02 * faint) * catalogue["teff"]
    logg_sd = 0.4 if missing else widen * (0.12 + 0.05 * faint)
    teff = float(np.clip(catalogue["teff"] + rng.normal(0, teff_sd), 2800, 9500))
    logg = float(np.clip(catalogue["logg"] + rng.normal(0, logg_sd), 2.5, 5.2))
    mass = float(
        np.clip(catalogue["mass_solar"] * (teff / catalogue["teff"]) ** 1.6, 0.08, 3.0)
    )
    radius = float(np.sqrt(mass * 10 ** (4.438 - logg)))
    return {
        "teff": teff,
        "logg": logg,
        "mass_solar": mass,
        "radius_solar": radius,
        "density_solar": mass / radius**3,
    }


def limb_darkening_grid(grid):
    """Interpolators over Detector's nonlinear Kepler grid (teff, logg)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    points = np.c_[grid["teff"].to_numpy() / 1000.0, grid["logg"].to_numpy()]
    values = grid[["a1", "a2", "a3", "a4"]].to_numpy()
    return LinearNDInterpolator(points, values), NearestNDInterpolator(points, values)


def limb_darkening(teff, logg, interpolators, rng=None):
    """Grid coefficients at the true (teff, logg)."""
    linear, nearest = interpolators
    coeffs = linear(teff / 1000.0, logg)
    if not np.all(np.isfinite(coeffs)):
        coeffs = nearest(teff / 1000.0, logg)
    return [float(c) for c in np.asarray(coeffs).ravel()]


def occurrence(period, radius, teff):
    """Planets per star per ln P per ln R (SAG13 with an inner turnover)."""
    period, radius = np.asarray(period, float), np.asarray(radius, float)
    years = period / 365.25
    small = radius < 3.4
    rate = np.where(
        small, 0.38 * radius**-0.19 * years**0.26, 0.73 * radius**-1.18 * years**0.59
    )
    knee = np.select([radius < 2, radius < 4], [3.0, 7.0], 10.0)
    rate = rate / (1 + (period / knee) ** -1.5)
    hot = np.where(
        (radius > 8) & (period < 10), 0.004 / (math.log(25 / 8) * math.log(10)), 0.0
    )
    rate = rate + hot
    cool = np.asarray(teff, float) < 4000
    rate = np.where(cool & small, rate * 2.0, rate)
    return np.where(cool & ~small, rate * 0.3, rate)


def semi_major_axis(period, mass_solar):
    """a in metres from Kepler's third law (planet mass neglected)."""
    return (G_SI * mass_solar * M_SUN * (period * DAY) ** 2 / (4 * math.pi**2)) ** (
        1 / 3
    )


def transit_probability(a_rs, radius_ratio, e, omega):
    return min(1.0, (1 + radius_ratio) / a_rs * (1 + e * math.sin(omega)) / (1 - e * e))


def tidal(period):
    """Tides circularise close orbits."""
    return np.clip((np.asarray(period, float) - 2.0) / 8.0, 0.0, 1.0)


def eccentricity(rng, multi, period=None):
    e = min(rng.rayleigh(0.083), 0.6) if multi else min(abs(rng.normal(0, 0.32)), 0.9)
    return float(e * (tidal(period) if period is not None else 1.0))


def duration_days(period, a_rs, k, b, inc, e, omega):
    """Total (T14) duration of an eccentric transit (Winn 2010, eq. 14-16)."""
    arg = math.sqrt(max((1 + k) ** 2 - b * b, 0.0)) / (a_rs * math.sin(inc))
    t14 = period / math.pi * math.asin(min(arg, 1.0))
    return t14 * math.sqrt(1 - e * e) / (1 + e * math.sin(omega))


# Period-band selection relative to 1-100 d (Adam, 29 Oct 2026).
PERIOD_BANDS = (
    (0.0, 0.5, 0.2),
    (0.5, 1.0, 1.0),
    (100.0, 200.0, 0.9),
    (200.0, np.inf, 0.12),
)
FEW_TRANSITS = 0.06


def boost(
    expected_mes,
    duration_hours,
    period,
    e,
    b,
    k,
    events,
    period_range=(0.5, 200.0),
    min_transits=3,
):
    """Relative selection preference (vectorised); a case's weight carries its inverse."""
    mes = np.nan_to_num(np.asarray(expected_mes, float), nan=0.0)
    period = np.asarray(period, float)
    g = np.select(
        [mes < 1, mes < 3, mes < 5, mes <= 40, mes <= 100],
        [0.02, 0.08, 0.6, 1.0, 0.35],
        0.15,
    )
    for low, high, factor in PERIOD_BANDS:
        g = g * np.where((period >= low) & (period < high), factor, 1.0)
    inside = (period >= period_range[0]) & (period <= period_range[1])
    enough = np.asarray(events, float) >= min_transits
    g = g * np.where(inside & ~enough, FEW_TRANSITS, 1.0)
    regime = np.where(duration_hours > 10, 10.0, np.where(duration_hours > 6, 3.0, 1.0))
    regime = regime * np.where(period > 50, 2.0, 1.0)
    regime = regime * np.where(e > 0.3, 1.5, 1.0)
    regime = regime * np.where(b > 1 - k, 2.0, 1.0)
    return g * np.where(inside & enough & (mes >= 3), regime, 1.0)


def proposals(host, n, rng, multi, period_range, radius_range):
    """`n` candidate transiting planets for `host` from the log-uniform proposal, with their geometry."""
    period = np.exp(rng.uniform(*np.log(period_range), n))
    radius = np.exp(rng.uniform(*np.log(radius_range), n))
    if multi:
        e = np.minimum(rng.rayleigh(0.083, n), 0.6)
    else:
        e = np.minimum(np.abs(rng.normal(0, 0.32, n)), 0.9)
    omega = rng.uniform(0, 2 * math.pi, n)
    rs = host["radius_solar"] * R_SUN
    a_rs = semi_major_axis(period, host["mass_solar"]) / rs
    e = e * tidal(period)
    e = np.clip(np.minimum(e, 1 - 3 / a_rs), 0, None)
    k = radius * R_EARTH / rs
    b = rng.uniform(0, 1, n) * (1 + k)
    factor = (1 + e * np.sin(omega)) / (1 - e * e)
    cos_i = np.clip(b / a_rs * factor, 0, 1)
    sin_i = np.sqrt(1 - cos_i**2)
    chord = np.sqrt(np.clip((1 + k) ** 2 - b * b, 0, None))
    t14 = (
        period
        / math.pi
        * np.arcsin(np.clip(chord / (a_rs * np.maximum(sin_i, 1e-9)), 0, 1))
    )
    t14 = t14 * np.sqrt(1 - e * e) / (1 + e * np.sin(omega))
    p_tr = np.minimum(1.0, (1 + k) / a_rs * factor)
    alive = (a_rs > 2) & (b / a_rs * factor < 1) & (k < 0.5)
    return {
        "period": period, "radius": radius, "e": e, "omega": omega, "a_rs": a_rs,
        "k": k, "b": b, "duration": t14, "p_tr": p_tr * alive,
        "occurrence": occurrence(period, radius, host["teff"]),
    }  # fmt: skip


def approximate_depth(k, b):
    """Mean in-transit depth for selection: k^2 with limb brightening, reduced for grazing chords."""
    graze = np.clip((1 + k - b) / (2 * k), 0, 1) ** 1.5
    return 1.1 * k * k * np.where(b > 1 - k, graze, 1.0)


def planet(
    host, ld, rng, period=None, radius=None, e=None, omega=None, b=None, multi=False
):
    """One transiting planet's physical parameters around `host` (truth)."""
    period = float(
        period
        if period is not None
        else math.exp(rng.uniform(math.log(0.5), math.log(700)))
    )
    radius = float(
        radius
        if radius is not None
        else math.exp(rng.uniform(math.log(0.5), math.log(25)))
    )
    e = eccentricity(rng, multi, period) if e is None else e
    omega = rng.uniform(0, 2 * math.pi) if omega is None else omega
    rs = host["radius_solar"] * R_SUN
    a_rs = semi_major_axis(period, host["mass_solar"]) / rs
    k = radius * R_EARTH / rs
    while a_rs * (1 - e) < 3.0 and e > 0:
        e *= 0.5
    b = float(rng.uniform(0, 1 + k) if b is None else b)
    # b is the impact parameter at conjunction: b = a cos i / R* x (1-e^2)/(1+e sin w)
    cos_i = b / a_rs * (1 + e * math.sin(omega)) / (1 - e * e)
    if cos_i >= 1:
        return None
    inc = math.acos(cos_i)
    t14 = duration_days(period, a_rs, k, b, inc, e, omega)
    return {
        "period_days": period,
        "radius_earth": radius,
        "radius_ratio": k,
        "a_rs": a_rs,
        "e": e,
        "omega_rad": omega,
        "b": b,
        "inc_deg": math.degrees(inc),
        "duration_days": t14,
        "grazing": bool(b > 1 - k),
        "limb_darkening": ld,
        "transit_probability": transit_probability(a_rs, k, e, omega),
        "occurrence": float(occurrence(period, radius, host["teff"])),
    }


def companions(primary, host, ld, rng, count):
    """Further transiting planets of a multi-planet system (peas in a pod)."""
    out = []
    inc0 = math.radians(primary["inc_deg"])
    for _ in range(count):
        for _attempt in range(20):
            ratio = math.exp(rng.normal(math.log(1.9), 0.35))
            if ratio < 1.2:
                continue
            period = primary["period_days"] * (
                ratio if rng.random() < 0.5 else 1 / ratio
            )
            if not 0.5 <= period <= 700:
                continue
            radius = primary["radius_earth"] * 10 ** rng.normal(0, 0.15)
            inc = inc0 + math.radians(rng.normal(0, 1.5))
            e = eccentricity(rng, True, period)
            omega = rng.uniform(0, 2 * math.pi)
            rs = host["radius_solar"] * R_SUN
            a_rs = semi_major_axis(period, host["mass_solar"]) / rs
            b = a_rs * math.cos(inc) * (1 - e * e) / (1 + e * math.sin(omega))
            k = radius * R_EARTH / rs
            if abs(b) > 1 + k:
                break
            p = planet(host, ld, rng, period, radius, e, omega, abs(b), multi=True)
            if p:
                out.append(p)
            break
    return out


RESONANCES = ((2, 1), (3, 2), (4, 3), (5, 4))


def assign_ttvs(planets, rng, single_rate=0.03):
    """Sinusoidal TTVs for near-first-order-resonant pairs, plus unseen perturbers."""
    for p in planets:
        p["ttv"] = None
    ordered = sorted(planets, key=lambda p: p["period_days"])
    for inner, outer in zip(ordered, ordered[1:], strict=False):
        ratio = outer["period_days"] / inner["period_days"]
        for j, i in RESONANCES:
            if abs(ratio / (j / i) - 1) < 0.05:
                super_period = 1 / abs(
                    j / outer["period_days"] - i / inner["period_days"]
                )
                phase = rng.uniform(0, 2 * math.pi)
                for planet_, scale in ((inner, 1.0), (outer, -1.0)):
                    # TTV amplitudes after Holczer et al. (2016).
                    amplitude = math.exp(
                        rng.uniform(math.log(1 / 1440), math.log(1.5 / 24))
                    )
                    planet_["ttv"] = {
                        "amplitude_days": amplitude,
                        "super_period_days": super_period,
                        "phase_rad": phase + (0 if scale > 0 else math.pi),
                        "cause": f"near {j}:{i}",
                    }
                break
    if len(planets) == 1 and rng.random() < single_rate:
        planets[0]["ttv"] = {
            "amplitude_days": math.exp(
                rng.uniform(math.log(1 / 1440), math.log(1 / 24))
            ),
            "super_period_days": math.exp(rng.uniform(math.log(100), math.log(2000))),
            "phase_rad": rng.uniform(0, 2 * math.pi),
            "cause": "unseen perturber",
        }
    return planets


def eclipsing_binary(host, ld, rng, periods, dilution=None):
    """A detached, grazing or close eclipsing binary on (or blended with) the host."""
    period = float(rng.choice(periods))
    k = float(math.exp(rng.uniform(math.log(0.15), math.log(1.0))))
    j = float(np.clip(k**2.5 * math.exp(rng.normal(0, 0.3)), 0.002, 1.0))
    rs = host["radius_solar"] * R_SUN
    mass = host["mass_solar"] * (1 + k**1.25)
    a_rs = semi_major_axis(period, mass) / rs
    e = (
        float(min(abs(rng.normal(0, 0.25)), 0.8))
        if period > 5
        else float(min(rng.rayleigh(0.02), 0.2))
    )
    while a_rs * (1 - e) < 1.6 * (1 + k) and e > 0:
        e *= 0.5
    if a_rs * (1 - e) < 1.6 * (1 + k):
        return None
    omega = rng.uniform(0, 2 * math.pi)
    b = float(rng.uniform(0, 1 + k))
    cos_i = b / a_rs * (1 + e * math.sin(omega)) / (1 - e * e)
    if cos_i >= 1:
        return None
    inc = math.acos(cos_i)
    light_ratio = k * k * j
    ellipsoidal = 0.15 * (k**1.25) * (1 / a_rs) ** 3 * math.sin(inc) ** 2
    if dilution is None:
        dilution = 1.0
    return {
        "kind": "eclipsing_binary" if dilution >= 1 else "blended_eclipsing_binary",
        "period_days": period,
        "radius_ratio": k,
        "surface_brightness_ratio": j,
        "light_ratio": light_ratio,
        "a_rs": a_rs,
        "e": e,
        "omega_rad": omega,
        "b": b,
        "inc_deg": math.degrees(inc),
        "duration_days": duration_days(period, a_rs, k, b, inc, e, omega),
        "ellipsoidal_amplitude": ellipsoidal,
        "dilution": float(dilution),
        "limb_darkening": ld,
    }
