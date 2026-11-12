"""Golden: the calibration physics."""

import math

import numpy as np
from conftest import array_digests, golden, inject, load_lane_host, planet
from scipy.stats import gamma

from calibration.sample_builder import (
    EXPOSURE_DAYS,
    FLUX_AT_KP12,
    KEPLER_COLOUR_KNOT,
    KEPLER_COLOUR_SLOPE,
    hz_edges_au,
    information_snr,
    orbital_period,
    star_light,
)  # fmt: skip
from calibration.scoring import gamma_efficiency

BODY = {"period_days": 3.0, "epoch_bkjd": 101.0}


def white_curve(sigmas, cadences=20000, seed=0):
    rng = np.random.default_rng(seed)
    quarter = np.repeat(np.arange(1, len(sigmas) + 1), cadences)
    time = 100.0 + np.arange(len(quarter)) * EXPOSURE_DAYS
    noise = np.repeat(np.asarray(sigmas, float), cadences)
    return {
        "time": time, "quarter": quarter, "mask": np.ones(len(time), bool),
        "flux": 1 + noise * rng.standard_normal(len(time)),
    }  # fmt: skip


def box(curve, depth, cadences):
    offset = (curve["time"] - BODY["epoch_bkjd"] + 1.5) % 3.0 - 1.5
    transit = np.abs(offset) < cadences * EXPOSURE_DAYS / 2
    return np.where(transit, depth, 0.0), transit


def test_information_snr_is_the_matched_filter_in_white_noise():
    curve = white_curve([1e-3] * 4)
    loss, transit = box(curve, 2e-4, 6)
    expected = 2e-4 * math.sqrt(transit.sum()) / 1e-3
    assert abs(information_snr(curve, loss, transit, BODY, {}) / expected - 1) < 0.05


def test_information_snr_adds_quarters_with_their_own_noise():
    curve = white_curve([1e-3, 2e-3])
    loss, transit = box(curve, 2e-4, 6)
    first, second = (transit & (curve["quarter"] == q) for q in (1, 2))
    expected = 2e-4 * math.sqrt(first.sum() / 1e-3**2 + second.sum() / 2e-3**2)
    assert abs(information_snr(curve, loss, transit, BODY, {}) / expected - 1) < 0.05


def test_red_stars_deliver_less_than_their_g_implies():
    g = 18.0
    expected = FLUX_AT_KP12 * 10 ** (-0.4 * (g - 12))
    curve = {"quarter": np.repeat([1, 2], 400), "mask": np.ones(800, bool)}
    metadata = {"normalization_by_quarter": {"1": 4 * expected, "2": 0.5 * expected}}
    blue = star_light(g, KEPLER_COLOUR_KNOT, curve, metadata)
    assert blue["fraction_by_quarter"] == {"1": 0.25, "2": 1.0}
    red = star_light(g, KEPLER_COLOUR_KNOT + 1.0, curve, metadata)
    assert np.isclose(red["kepler_band_mag"], g + KEPLER_COLOUR_SLOPE)
    assert np.isclose(
        red["fraction_by_quarter"]["1"], 0.25 * 10 ** (-0.4 * KEPLER_COLOUR_SLOPE)
    )


def test_habitable_zone_edges_of_the_sun():
    inner, outer = hz_edges_au({"radius_solar": 1.0, "teff": 5772.0})
    assert abs(inner - 0.950) < 0.002 and abs(outer - 1.676) < 0.003
    assert abs(orbital_period(1.0, 1.0) - 365.25) < 1e-9


def test_efficiency_fit_recovers_a_known_curve():
    rng = np.random.default_rng(1)
    snr = rng.uniform(3, 30, 20000)
    hit = rng.random(len(snr)) < 0.95 * gamma.cdf(snr, 25.0, scale=0.36)
    fit = gamma_efficiency(snr, hit)
    assert abs(fit["c"] - 0.95) < 0.01
    assert abs(fit["snr_at_50pct"] - gamma.ppf(0.5 / 0.95, 25.0, scale=0.36)) < 0.15


def test_injection_is_golden():
    """A two-planet system (the inner one grazing) on the quiet K dwarf."""
    lane = load_lane_host()
    bodies = [planet(lane, 4.8, 1.5, 1.3), planet(lane, 31.7, 2.4, 9.0, b=1.0)]
    arrays, metadata, signals = inject(lane, bodies)
    keep = ("observed_events", "model_depth_ppm", "flat_depth_ppm", "flattening_transfer", "local_transfer",
            "information_snr", "white_snr", "star_flux_fraction", "detectable", "grazing", "duration_days")  # fmt: skip
    golden("injection", {
        "arrays": array_digests({k: arrays[k] for k in ("flux", "error", "native_flux", "mask")}),
        "normalization_by_quarter": metadata["normalization_by_quarter"],
        "signals": [{k: s[k] for k in keep} for s in signals],
    })  # fmt: skip
