"""KBonus-realistic calibration sample for the PSF Flat campaign."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import artifacts
from calibration import population as pop
from calibration.lane_census import (
    DESIGN,
    KOI_TABLE,
    LC2_COLUMNS,
    array_features,
    binned,
    lane,
    prepare_lc2,
    robust_std,
    single_threaded,
)
from calibration.signal_injection import SMOOTHER, binary_flux, flattened, transit_flux
from common import (
    ROOT,
    config_file,
    digest,
    load_packet,
    phase,
    read,
    receipt,
    require,
    save_packet,
    write,
)
from photometry.star_share import (  # noqa: F401 (tests import them from here)
    FLUX_AT_KP12,
    KEPLER_COLOUR_KNOT,
    KEPLER_COLOUR_SLOPE,
    kepler_band_mag,
)

SEED = "kbonus-real-20261029-v1"
EXPOSURE_DAYS = 0.02043361111111111
# Proposal-shaping S/N scale (the first build's Detector); host weights undo it.
PROPOSAL_SNR_SCALE = 0.80
MIN_QUARTER_BINS = 20
# Conservative habitable zone: Kopparapu et al. (2014) eq. 4, one Earth mass.
HZ_EDGES = {
    "runaway_greenhouse": (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
    "maximum_greenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
}
AU = 1.495978707e11
CONFIRMED = Path(
    "/Volumes/discPCIe/data/kepler-planets/catalogue/nasa-confirmed-kepler-planets-20260904.json"
)
KBONUS_FITS = Path("/Volumes/discPCIe/data/kepler-planets/kbonus-fits")
EB_CATALOGUE = Path(
    "/Volumes/discPCIe/data/reference-catalogues/kepler-eb-2016-0694ada5b64a.csv"
)
DISPOSITIONS = ROOT.parent / "results/boi-validation"
LD_GRID = ROOT / "detector/upstream/data/meta/limb_coeffs.h5"

PLAN = {
    "seed": SEED,
    "planet_hosts": 5000,
    "second_planet_hosts": 2500,
    # Floors for rare classes; host weights undo them.
    "class_floors": {"M": 500, "subgiant": 250, "F": 250, "giant": 250, "hot": 120},
    "multi_fractions": {"2": 0.20, "3": 0.07},
    "period_range_days": [0.25, 750.0],
    "radius_range_earth": [0.5, 25.0],
    "proposals_per_host": 20000,
    "inverted_controls": 800,
    "scrambled_controls": 400,
    "artifact_controls": 600,
    "artifact_copies_per_class": 400,
    "artifact_first_host": 800,
    "q2_comb_extra_hosts": 400,
    "scramble_block_cadences": 480,
    "synthetic_binaries": 600,
    "synthetic_blend_candidates": 800,
    "blend_dilution_range": [0.001, 0.3],
    "blend_min_foreign_light": 0.01,
    "contamination": {
        "eclipsing_binary": 450,
        "koi_false_positive": 200,
        "koi_planet": 150,
    },
    "contamination_added": {
        "eclipsing_binary": 250,
        "koi_false_positive": 400,
        "koi_planet": 50,
    },
    "contamination_min_snr": 7.1,
    # Binary rate: Kirk et al. (2016), 2,878 among ~207,000 Kepler targets.
    "eclipsing_binary_fraction": 0.0139,
    "known_planet_hosts": 1000,
    # Habitable-zone stratum: K and M curves where a mid-zone Earth reaches S/N 1.
    "hz_panel": {
        "classes": ["M", "K"],
        "min_expected_snr_earth": 1.0,
        "cases_per_host": 6,
        "radius_range_earth": [0.5, 2.0],
        "multi_fractions": {"2": 0.25, "3": 0.10},
        "max_mid_period_days": 200.0,
        "beyond_range_fraction": 0.05,
    },
    "quick_fraction": 0.15,
    "minimum_free_gib": 20,
    "smoother": SMOOTHER,
}


def keyed(*parts):
    return hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).hexdigest()


def rng_for(*parts):
    return np.random.default_rng(int(keyed(*parts)[:16], 16))


def partition(family):
    return (
        "development" if int(keyed("partition", family)[:8], 16) % 2 == 0 else "holdout"
    )


@lru_cache(maxsize=1)
def config():
    return config_file()


@lru_cache(maxsize=1)
def ld_interpolators():
    return pop.limb_darkening_grid(pd.read_hdf(LD_GRID, "KEPLER"))


@lru_cache(maxsize=1)
def eb_periods():
    eb = pd.read_csv(EB_CATALOGUE)
    return eb.Per[(eb.Per > 0.25) & (eb.Per < 750)].to_numpy()


def lc2_arrays(source_id):
    from photometry.program import locate_source, source_table

    folder = locate_source(str(source_id), "/Volumes/discPCIe/data/index.sqlite")
    table = source_table(
        folder / "cadence.parquet", str(source_id), list(LC2_COLUMNS.values())
    )
    arrays = {key: table[name].to_numpy() for key, name in LC2_COLUMNS.items()}
    return arrays, {"shard": str(folder)}


def fits_arrays(path):
    from astropy.io import fits

    from photometry.program import load_kbonus

    arrays, info = load_kbonus(path)
    with fits.open(path, memmap=False) as hdus:
        arrays["raw"] = np.array(hdus["LIGHTCURVE_STITCHED"].data["FLUX"], dtype=float)
    return arrays, info


def prepare(arrays, source, stellar, provenance):
    """The campaign's preparation plus the raw PSF the flattening emulator needs."""
    prepared_arrays, raw, scales = prepare_lc2(arrays)
    good = prepared_arrays["mask"]
    time = prepared_arrays["time"]
    metadata = {
        "schema": 1,
        "source_id": str(source["source_id"]),
        "kic": source.get("kic"),
        "parent_kic": source.get("parent_kic"),
        "ra_deg": source.get("ra_deg"),
        "dec_deg": source.get("dec_deg"),
        "view": "psf_flat",
        "time_system": "BKJD_TDB",
        "exposure_days": EXPOSURE_DAYS,
        "flux_convention": "delivered_psf_flat",
        "stellar": stellar,
        "provenance": provenance,
        "normalization_by_quarter": scales,
        "additional_detrending": False,
        "stored_cadences": len(time),
        "usable_cadences": int(good.sum()),
        "baseline_days": float(np.ptp(time[good])),
        "first_valid_bkjd": float(time[good].min()),
        "last_valid_bkjd": float(time[good].max()),
        "median_formal_error_ppm": float(
            np.median(prepared_arrays["error"][good]) * 1e6
        ),
        "exclusion_bit_definitions": {
            "1": "nonfinite",
            "2": "nonpositive_error",
            "4": "quality",
            "8": "flatten_mask",
            "16": "invalid_quarter_normalization",
            "32": "retained_flag_outlier",
            "64": "field_artifact_epoch",
        },
    }
    return prepared_arrays, raw, metadata


def star_light(g_mag, bp_rp, prepared, metadata):
    """The host star's share of each quarter's delivered flux."""
    levels = metadata["normalization_by_quarter"]
    if g_mag is None or not np.isfinite(g_mag):
        return {"expected_flux": None, "fraction_by_quarter": {}, "fraction": 1.0}
    kp = kepler_band_mag(g_mag, bp_rp)
    expected = FLUX_AT_KP12 * 10 ** (-0.4 * (kp - 12))
    by_quarter = {q: min(1.0, expected / level) for q, level in levels.items()}
    usable = star_fraction(prepared["quarter"], by_quarter)[prepared["mask"]]
    return {
        "kepler_band_mag": kp,
        "expected_flux": expected,
        "fraction_by_quarter": by_quarter,
        "fraction": float(np.mean(usable)),
    }


def star_fraction(quarter, by_quarter):
    """Per-cadence star share; quarters without a level (masked) keep 1."""
    return np.array([by_quarter.get(str(q), 1.0) for q in quarter])


def with_signal(
    prepared, metadata, raw, signal, fraction, raw_factor=None, flat_change=None
):
    """Prepared arrays plus a signal on the star passed through the emulated smoother."""
    time, quarter = prepared["time"], prepared["quarter"]
    valid = np.isfinite(time) & np.isfinite(raw) & np.isfinite(prepared["flux"])
    diluted = 1 + fraction * (np.asarray(signal, float) - 1)
    if raw_factor is not None:
        diluted = diluted * raw_factor
    change = flattened(raw, time, quarter, valid, diluted) - 1
    if flat_change is not None:
        change = change + flat_change
    scales = {int(q): v for q, v in metadata["normalization_by_quarter"].items()}
    scale = np.array([scales.get(int(q), 1.0) for q in quarter])
    native = prepared["native_flux"] + change * scale
    result = {name: value.copy() for name, value in prepared.items()}
    result["native_flux"] = native
    normalization = {}
    for q in np.unique(quarter):
        group = quarter == q
        usable = group & prepared["mask"]
        if not usable.any():
            result["flux"][group] = native[group]
            continue
        level = float(np.median(native[usable]))
        result["flux"][group] = native[group] / level
        result["error"][group] = prepared["native_error"][group] / level
        normalization[str(q)] = level
    return result, metadata | {"normalization_by_quarter": normalization}, change


def cdpp_at(features, hours):
    """Noise of a mean over `hours` from the host's binned-noise ladder."""
    widths = np.array([1.0, 3.0, 6.0, 12.0])
    values = (
        np.array([features.get(f"cdpp_{w:.0f}h_ppm", np.nan) for w in widths], float)
        / 1e6
    )
    ok = np.isfinite(values) & (values > 0)
    if not ok.any():
        return np.full(np.shape(hours), np.nan)
    hours = np.asarray(hours, float)
    lw, lv = np.log(widths[ok]), np.log(values[ok])
    inside = np.exp(
        np.interp(np.log(np.clip(hours, widths[ok][0], widths[ok][-1])), lw, lv)
    )
    below = values[ok][0] * np.sqrt(widths[ok][0] / np.maximum(hours, 1e-3))
    above = values[ok][-1] * np.sqrt(widths[ok][-1] / np.maximum(hours, 1e-3))
    return np.where(
        hours < widths[ok][0], below, np.where(hours > widths[ok][-1], above, inside)
    )


def observed_events(time, period, epoch, duration, exposure, minimum_coverage):
    """Indices of the transits whose duration the cadences cover at least `minimum_coverage`."""
    event_id = np.rint((time - epoch) / period).astype(np.int64)
    overlaps = np.maximum(
        0,
        np.minimum(time + exposure / 2, epoch + event_id * period + duration / 2)
        - np.maximum(time - exposure / 2, epoch + event_id * period - duration / 2),
    )
    ids, inverse = np.unique(event_id[overlaps > 0], return_inverse=True)
    covered = np.bincount(inverse, weights=overlaps[overlaps > 0], minlength=len(ids))
    return ids[covered >= duration * minimum_coverage]


def local_depth(time, flux, period, epoch, duration):
    """Median per-event depth against a local linear baseline."""
    depths = []
    n0, n1 = (
        int(np.floor((time.min() - epoch) / period)),
        int(np.ceil((time.max() - epoch) / period)),
    )
    for k in range(n0, n1 + 1):
        dt = time - (epoch + k * period)
        inside = np.abs(dt) < 0.35 * duration
        outside = (np.abs(dt) > 0.75 * duration) & (
            np.abs(dt) < 0.75 * duration + max(duration, 0.25)
        )
        if (
            inside.sum() < 2
            or (dt[outside] < 0).sum() < 3
            or (dt[outside] > 0).sum() < 3
        ):
            continue
        line = np.polyfit(dt[outside], flux[outside], 1)
        depths.append(1 - np.mean(flux[inside] / np.polyval(line, dt[inside])))
    return float(np.median(depths)) if depths else None


def noise_of_mean(prepared, quarter, cadences, features):
    """Noise of a mean of `cadences` consecutive cadences in one quarter of the prepared curve."""
    sel = prepared["mask"] & (prepared["quarter"] == quarter)
    width = cadences * EXPOSURE_DAYS
    if sel.sum() > MIN_QUARTER_BINS * cadences:
        _, means = binned(prepared["time"][sel], prepared["flux"][sel], width)
        if len(means) > MIN_QUARTER_BINS:
            return float(robust_std(np.diff(means)) / math.sqrt(2))
    return float(cdpp_at(features, width * 24))


def information_snr(prepared, loss, transit, body, features):
    """S/N of a matched filter with the signal's true shape, detector aside."""
    good = prepared["mask"] & transit
    if not good.any():
        return 0.0
    time, quarter, s = prepared["time"][good], prepared["quarter"][good], loss[good]
    event = np.rint((time - body["epoch_bkjd"]) / body["period_days"]).astype(np.int64)
    _, first, inverse, n = np.unique(
        event, return_index=True, return_inverse=True, return_counts=True
    )
    power = np.bincount(inverse, s**2) / n
    noise = {}
    total = 0.0
    for k, q in enumerate(quarter[first]):
        key = (int(q), int(n[k]))
        if key not in noise:
            noise[key] = noise_of_mean(prepared, *key, features)
        if noise[key] > 0:
            total += power[k] / noise[key] ** 2
    return float(math.sqrt(total))


def signal_statistics(prepared, change, model, body, features, fraction):
    """Observed events, depths on the star and in the delivered curve."""
    good = prepared["mask"]
    time = prepared["time"][good]
    duration = body["duration_days"]
    events = observed_events(
        time, body["period_days"], body["epoch_bkjd"], duration, EXPOSURE_DAYS,
        config()["calibration"]["sample_construction"]["minimum_event_coverage"],
    )  # fmt: skip
    centre = (
        np.abs(phase(time, body["period_days"], body["epoch_bkjd"])) < 0.25 * duration
    )
    loss_model = 1 - model[good]
    diluted = 1 + fraction[good] * (model[good] - 1)
    ratio = 1 + change[good]
    loss_flat = 1 - ratio
    if centre.any():
        depth = float(np.mean(loss_model[centre]))
        share = float(np.mean(fraction[good][centre]))
        flat_depth = float(np.mean(loss_flat[centre]))
    else:
        depth = float(np.max(loss_model, initial=0))
        share = float(np.mean(fraction[good])) if good.any() else 1.0
        flat_depth = float(np.max(loss_flat, initial=0))
    local_model = local_depth(
        time, diluted, body["period_days"], body["epoch_bkjd"], duration
    )
    local_flat = local_depth(
        time, ratio, body["period_days"], body["epoch_bkjd"], duration
    )
    point = features.get("point_noise_ppm", np.nan) / 1e6
    snr = information_snr(
        prepared, -np.asarray(change, float), model < 1, body, features
    )
    return {
        "observed_events": int(len(events)),
        "observed_event_indices": events.tolist(),
        "three_event_coverage": bool(len(events) >= 3),
        "model_depth_ppm": 1e6 * depth,
        "star_flux_fraction": share,
        "flat_depth_ppm": 1e6 * flat_depth,
        "flattening_transfer": flat_depth / (depth * share) if depth > 0 else None,
        "local_transfer": local_flat / local_model
        if local_model and local_flat is not None and local_model > 0
        else None,
        "information_snr": snr,
        "white_snr": float(np.sqrt(np.sum((loss_flat / point) ** 2)))
        if point > 0
        else None,
        "in_search_box": bool(
            config()["detection"]["period_min_days"]
            <= body["period_days"]
            <= config()["detection"]["period_max_days"]
            and len(events) >= config()["detection"]["min_observed_transits"]
        ),
    }


def draw_system(truth, features, metadata, rng, plan, share):
    """One transiting planet chosen by occurrence x transit probability x boost."""
    roll = rng.random()
    size = (
        3
        if roll < plan["multi_fractions"]["3"]
        else 2
        if roll < plan["multi_fractions"]["3"] + plan["multi_fractions"]["2"]
        else 1
    )
    multi = size > 1
    cand = pop.proposals(
        truth,
        plan["proposals_per_host"],
        rng,
        multi,
        plan["period_range_days"],
        plan["radius_range_earth"],
    )
    duty = features.get("duty_cycle", 0.5)
    baseline = metadata["baseline_days"]
    events = duty * baseline / cand["period"]
    noise = cdpp_at(features, cand["duration"] * 24)
    depth = share * pop.approximate_depth(cand["k"], cand["b"])
    mes = PROPOSAL_SNR_SCALE * depth / noise * np.sqrt(events)
    detection = config()["detection"]
    g = pop.boost(
        mes,
        cand["duration"] * 24,
        cand["period"],
        cand["e"],
        cand["b"],
        cand["k"],
        events,
        (detection["period_min_days"], detection["period_max_days"]),
        detection["min_observed_transits"],
    )
    base = cand["occurrence"] * cand["p_tr"]
    target = base * g
    require(target.sum() > 0, "No viable planet proposals")
    pick = int(rng.choice(len(target), p=target / target.sum()))
    volume = math.log(
        plan["period_range_days"][1] / plan["period_range_days"][0]
    ) * math.log(plan["radius_range_earth"][1] / plan["radius_range_earth"][0])
    selection = {
        "boost": float(g[pick]),
        "approximate_mes": float(mes[pick]),
        "transiting_rate": float(volume * base.mean()),
        "boosted_rate": float(volume * target.mean()),
        "proposals": int(len(target)),
        "system_size_drawn": size,
        "star_flux_fraction": share,
    }
    ld = pop.limb_darkening(truth["teff"], truth["logg"], ld_interpolators(), rng)
    primary = pop.planet(
        truth, ld, rng, period=cand["period"][pick], radius=cand["radius"][pick], e=float(cand["e"][pick]),
        omega=float(cand["omega"][pick]), b=float(cand["b"][pick]), multi=multi,
    )  # fmt: skip
    require(primary is not None, "Selected proposal has no physical orbit")
    planets = [primary | {"role": "primary"}]
    if multi:
        planets += [
            p | {"role": "companion"}
            for p in pop.companions(primary, truth, ld, rng, size - 1)
        ]
    pop.assign_ttvs(planets, rng)
    for body in planets:
        body["epoch_bkjd"] = float(
            metadata["first_valid_bkjd"] + rng.uniform(0, body["period_days"])
        )
        body |= hz_fields(body, truth)
    return planets, selection


def planet_case(host, prepared, raw, metadata, features, rng, plan):
    light = host["star_light"]
    planets, selection = draw_system(
        host["truth"], features, metadata, rng, plan, light["fraction"]
    )
    draws = host.get("planet_draws", 1)
    weight = (
        host["host_weight"] / draws * selection["boosted_rate"] / selection["boost"]
    )
    if (host.get("hz") or {}).get("stratum") and in_hz_box(
        planets[0], host["truth"], plan["hz_panel"]
    ):
        selection["in_panel_box"] = True
        weight = 0.0
    injected, metadata, signals, model = inject_system(
        host, prepared, raw, metadata, features, planets,
        lambda body: {"population_weight": weight},
    )  # fmt: skip
    truth = {"signals": signals, "selection": selection, "system_size": len(planets)}
    return injected, metadata, truth, model


def inject_system(host, prepared, raw, metadata, features, planets, extra):
    """`planets` diluted by the star's share and passed through the emulated smoother."""
    fraction = star_fraction(
        prepared["quarter"], host["star_light"]["fraction_by_quarter"]
    )
    model = np.ones(len(prepared["time"]))
    models = []
    for body in planets:
        m = transit_flux(prepared["time"], EXPOSURE_DAYS, body, body["epoch_bkjd"])
        models.append(m)
        model *= m
    injected, metadata, change = with_signal(prepared, metadata, raw, model, fraction)
    signals = []
    for body, m in zip(planets, models, strict=True):
        alone = (
            with_signal(prepared, metadata, raw, m, fraction)[2]
            if len(planets) > 1
            else change
        )
        stats = signal_statistics(prepared, alone, m, body, features, fraction)
        signals.append(
            {
                "kind": "planet",
                "period_days": body["period_days"],
                "epoch_bkjd": body["epoch_bkjd"],
                "duration_days": body["duration_days"],
                **{
                    k: v
                    for k, v in body.items()
                    if k not in ("period_days", "epoch_bkjd", "duration_days")
                },
                "limb_darkening_law": "nonlinear",
                **extra(body),
                **stats,
                **detectability(body, stats, host["catalogue_stellar"]),
            }
        )
    return injected, metadata, signals, model


def detectability(body, stats, catalogue):
    """What the photons allow inside the search box, the detector aside."""
    detection = config()["detection"]
    if (
        not detection["period_min_days"]
        <= body["period_days"]
        <= detection["period_max_days"]
    ):
        control = "outside_period_range"
    elif stats["observed_events"] < detection["min_observed_transits"]:
        control = "fewer_than_min_transits"
    else:
        control = None
    snr = stats.get("information_snr")
    density = catalogue["density_solar"]
    return {
        "control_class": control,
        "detectable": bool(
            control is None and snr is not None and snr >= detection["mes_threshold"]
        ),
        "catalogue_density_solar": density,
        "reference_duration_days": 13
        / 24
        * (body["period_days"] / 365) ** (1 / 3)
        * density ** (-1 / 3),
    }


def binary_case(host, prepared, raw, metadata, features, rng, plan, blended):
    truth = host["truth"]
    dilution = 1.0
    if blended:
        foreign = 1 - host["star_light"]["fraction"]
        if foreign < plan["blend_min_foreign_light"]:
            return None
        rng = rng_for("blend", host["source_id"])
        low, high = plan["blend_dilution_range"]
        dilution = float(
            math.exp(rng.uniform(math.log(low), math.log(min(high, foreign))))
        )
    ld = pop.limb_darkening(truth["teff"], truth["logg"], ld_interpolators(), rng)
    for _ in range(50):
        binary = pop.eclipsing_binary(truth, ld, rng, eb_periods(), dilution)
        if binary is not None:
            break
    require(binary is not None, "No physical binary drawn")
    binary["epoch_bkjd"] = float(
        metadata["first_valid_bkjd"] + rng.uniform(0, binary["period_days"])
    )
    model = binary_flux(prepared["time"], EXPOSURE_DAYS, binary, binary["epoch_bkjd"])
    # A host binary is diluted by the host's share; a blend's dilution is its own share.
    fraction = (
        np.ones(len(prepared["time"]))
        if blended
        else star_fraction(
            prepared["quarter"], host["star_light"]["fraction_by_quarter"]
        )
    )
    injected, metadata, change = with_signal(prepared, metadata, raw, model, fraction)
    stats = signal_statistics(prepared, change, model, binary, features, fraction)
    signal = {
        "kind": binary["kind"],
        "period_days": binary["period_days"],
        "epoch_bkjd": binary["epoch_bkjd"],
        "duration_days": binary["duration_days"],
        **{
            k: v
            for k, v in binary.items()
            if k not in ("kind", "period_days", "epoch_bkjd", "duration_days")
        },
        "limb_darkening_law": "nonlinear",
        "expected_disposition": "false_positive",
        **stats,
    }
    return injected, metadata, {"signals": [signal]}


def save_case(out, kind, host, arrays, metadata, truth, index=0):
    case_id = keyed("case", kind, host["source_id"], index)[:24]
    folder = Path(out) / "light-curves" / case_id
    folder.mkdir(parents=True)
    save_packet(
        folder, arrays, metadata | {"case_id": case_id, "calibration_case": True}
    )
    record = {
        "case_id": case_id,
        "kind": kind,
        "source_id": host["source_id"],
        "prepared": str(folder),
        "partition": host["partition"],
        "family": host["family"],
        "host": {k: v for k, v in host.items() if k not in ("truth",)}
        | {"truth": host.get("truth")},
        "signals": [],
        "known_signals": host.get("known_signals", []),
        **truth,
    }
    record["diagnostics"] = {
        "three_event_coverage": any(
            s.get("three_event_coverage") for s in record["signals"]
        )
        or None
    }
    path = Path(out) / "truth" / f"{case_id}.json"
    write(path, record)
    return {
        "case_id": case_id,
        "kind": kind,
        "source_id": host["source_id"],
        "partition": host["partition"],
        "family": host["family"],
        "population": host.get("population"),
        "g_mag": host.get("g_mag"),
        "bp_rp": host.get("bp_rp"),
        "star_flux_fraction": host["star_light"]["fraction"],
        "host_weight": host.get("host_weight"),
        "hz_stratum": bool((host.get("hz") or {}).get("stratum")),
        "n_signals": len(record["signals"]),
        "prepared": str(folder),
        "truth": str(path),
        "truth_sha256": digest(path),
        "packet_sha256": digest(folder / "packet.json"),
        "stored_cadences": metadata["stored_cadences"],
        "usable_cadences": metadata["usable_cadences"],
    }


def build_host(job):
    """All cases of one host; exceptions are returned, not raised."""
    try:
        return _build_host(job)
    except Exception as error:
        return {
            "source_id": job["host"]["source_id"],
            "error": f"{type(error).__name__}: {error}",
        }


def _build_host(job):
    host, out, plan = job["host"], job["out"], job["plan"]
    rng = rng_for("host", host["source_id"], job.get("role", "lane"))
    if job.get("fits"):
        arrays, info = fits_arrays(job["fits"])
        provenance = {"kbonus_fits": receipt(job["fits"])}
    else:
        arrays, info = lc2_arrays(host["source_id"])
        provenance = info
    stellar = pop.catalogue_stellar(host.get("teff_gspphot"), host.get("logg_gspphot"))
    prepared, raw, metadata = prepare(arrays, host, stellar, provenance)
    features = array_features(prepared, raw)
    host = host | {
        "features": features,
        "catalogue_stellar": stellar,
        "planet_draws": job["kinds"].count("synthetic_planet"),
    }
    host["truth"] = pop.true_host(stellar, rng, host.get("g_mag"))
    host["star_light"] = star_light(
        host.get("g_mag"), host.get("bp_rp"), prepared, metadata
    )
    host["hz"] = host_hz(host, stellar, features, metadata, plan["hz_panel"])
    if job.get("role") == "hz-panel":
        host["hz"] = (host["hz"] or {}) | {"stratum": True}
    context = {
        "job": job, "host": host, "prepared": prepared, "raw": raw,
        "metadata": metadata, "features": features, "rng": rng, "plan": plan,
    }  # fmt: skip
    rows, summaries, seen = [], [], {}
    for kind in job["kinds"]:
        index = seen[kind] = seen.get(kind, -1) + 1
        made = MAKERS[kind](context, kind)
        if made is None:
            continue
        arrays, case_metadata, truth = made
        row = save_case(out, kind, host, arrays, case_metadata, truth, index)
        row |= {
            "role": job.get("role", "lane"),
            "artifact_pair": job.get("artifact_pair")
            if kind.startswith("artifact_")
            else None,
        }
        rows.append(row)
        summaries += [
            {
                "case_id": row["case_id"],
                "case_kind": kind,
                "source_id": host["source_id"],
                **s,
            }
            for s in truth.get("signals", [])
        ]
    return {
        "source_id": host["source_id"],
        "rows": rows,
        "signals": summaries,
        "features": features,
    }


def make_original(c, kind):
    scope = "Uninjected lane light curve; may hold real signals"
    return c["prepared"], c["metadata"], {"scope": scope}


def make_planet(c, kind):
    arrays, metadata, truth, model = planet_case(
        c["host"], c["prepared"], c["raw"], c["metadata"], c["features"], c["rng"], c["plan"],
    )  # fmt: skip
    c.setdefault("planet_pair", (model, truth))
    return arrays, metadata, truth


def make_artifact_copy(c, kind):
    """A copy of the host's original or first planet case carrying one artifact class."""
    name = kind.removeprefix("artifact_")
    host, prepared = c["host"], c["prepared"]
    rng = rng_for("artifact", host["source_id"], name)
    features = c["features"]
    noise = {
        "point": features["point_noise_ppm"] / 1e6,
        "bin_noise": features["cdpp_3h_ppm"] / 1e6,
    }
    made = artifacts.artifact(name, prepared, noise, rng)
    if made is None:
        return None
    flat, raw_factor, truth = made
    paired = c["job"]["artifact_pair"]
    signal, signals = np.ones(len(prepared["time"])), []
    if paired == "synthetic_planet":
        model, planet_truth = c["planet_pair"]
        signal, signals = model, planet_truth["signals"]
    fraction = star_fraction(
        prepared["quarter"], host["star_light"]["fraction_by_quarter"]
    )
    arrays, metadata, _ = with_signal(
        prepared, c["metadata"], c["raw"], signal, fraction, raw_factor, flat
    )
    truth |= {
        "paired_kind": paired,
        "paired_case": keyed("case", paired, host["source_id"], 0)[:24],
    }
    scope = (
        "Copy of the paired case with one measured artifact class added; any candidate "
        "not matching a planet signal or known signal is an artifact"
    )
    return arrays, metadata, {"signals": signals, "artifact": truth, "scope": scope}


def make_binary(c, kind):
    return binary_case(
        c["host"], c["prepared"], c["raw"], c["metadata"], c["features"], c["rng"],
        c["plan"], kind == "synthetic_blend",
    )  # fmt: skip


def transformed_control(arrays, metadata, kind, seed, block):
    """DR25-style inverted curve, or blocks of usable cadences reordered within each quarter."""
    result = {name: value.copy() for name, value in arrays.items()}
    random = np.random.default_rng(seed)
    for quarter, scale in metadata["normalization_by_quarter"].items():
        group = np.flatnonzero((arrays["quarter"] == int(quarter)) & arrays["mask"])
        if kind == "inverted":
            result["flux"][group] = 2 - arrays["flux"][group]
            result["native_flux"][group] = scale * result["flux"][group]
        else:
            chunks = [group[i : i + block] for i in range(0, len(group), block)]
            shuffled = np.concatenate(
                [chunks[i] for i in random.permutation(len(chunks))]
            )
            for name in ("flux", "error", "native_flux", "native_error"):
                result[name][group] = arrays[name][shuffled]
    return result


def make_transformed(c, kind):
    transform = "inverted" if kind == "control_inverted" else "block_scrambled"
    seed = int(keyed(kind, c["host"]["source_id"])[:16], 16)
    control = transformed_control(
        c["prepared"],
        c["metadata"],
        transform,
        seed,
        c["plan"]["scramble_block_cadences"],
    )
    return control, c["metadata"], {"transformation": transform, "seed": seed}


def make_artifact(c, kind):
    truth = {
        "artifact": c["job"]["artifact"],
        "scope": "Uninjected light curve chosen for dips at the field-wide artifact epochs "
        "the production mask leaves; any candidate is an artifact unless it matches a known signal",
    }
    return c["prepared"], c["metadata"], truth


def measured_depth(prepared, body):
    """Depth of a real signal in its own delivered curve."""
    good = prepared["mask"]
    time, flux = prepared["time"][good], prepared["flux"][good]
    duration = body["duration_days"]
    distance = np.abs(phase(time, body["period_days"], body["epoch_bkjd"]))
    centre = distance < 0.25 * duration
    ring = (distance > 0.75 * duration) & (
        distance < 0.75 * duration + max(duration, 0.25)
    )
    if centre.sum() < 3 or ring.sum() < 10:
        return None
    return float(np.median(flux[ring]) - np.mean(flux[centre]))


def make_real(c, kind):
    """Real signals (leakage, known planets, validated BOIs) on the unchanged curve."""
    prepared = c["prepared"]
    none = np.zeros(len(prepared["time"]))
    signals = []
    for s in c["job"]["signals"]:
        body = dict(s)
        if body.get("duration_days") is None:
            body["duration_days"] = folded_duration(prepared, body)
        stats = signal_statistics(
            prepared, none, none + 1, body, c["features"], none + 1
        )
        keep = ("observed_events", "three_event_coverage", "in_search_box")
        depth = measured_depth(prepared, body)
        box = np.abs(phase(prepared["time"], body["period_days"], body["epoch_bkjd"]))
        snr = (
            information_snr(
                prepared,
                none + depth,
                box < body["duration_days"] / 2,
                body,
                c["features"],
            )
            if depth is not None
            else None
        )
        measured = {
            "measured_depth_ppm": None if depth is None else 1e6 * depth,
            "information_snr": snr,
        }
        stats |= measured
        signals.append(
            body | {k: stats[k] for k in keep} | measured
            | detectability(body, stats, c["host"]["catalogue_stellar"])
        )  # fmt: skip
    return (
        prepared,
        c["metadata"],
        {"signals": signals, "scope": c["job"].get("scope", "")},
    )


def hz_flux(teff, edge):
    """Stellar flux (Earth's = 1) at a conservative habitable-zone edge."""
    x = teff - 5780.0
    s, a, b, c, d = HZ_EDGES[edge]
    return s + a * x + b * x**2 + c * x**3 + d * x**4


def luminosity_solar(star):
    return star["radius_solar"] ** 2 * (star["teff"] / 5772) ** 4


def hz_edges_au(star):
    """Inner (runaway greenhouse) and outer (maximum greenhouse) edges, au."""
    return tuple(
        math.sqrt(luminosity_solar(star) / hz_flux(star["teff"], e)) for e in HZ_EDGES
    )


def orbital_period(a_au, mass_solar):
    return 365.25 * math.sqrt(a_au**3 / mass_solar)


def hz_expectation(star, features, kp, level):
    """An Earth at the middle of the conservative habitable zone."""
    middle = math.sqrt(math.prod(hz_edges_au(star)))
    period = orbital_period(middle, star["mass_solar"])
    rs_au = star["radius_solar"] * pop.R_SUN / AU
    duration = period / math.pi * math.asin(min(1.0, rs_au / middle))
    k = pop.R_EARTH / (star["radius_solar"] * pop.R_SUN)
    share = min(1.0, FLUX_AT_KP12 * 10 ** (-0.4 * (kp - 12)) / level)
    events = features["duty_cycle"] * features["baseline_days"] / period
    snr = (
        share
        * float(pop.approximate_depth(k, 0.0))
        / float(cdpp_at(features, duration * 24))
        * math.sqrt(events)
    )
    return {"mid_period_days": period, "expected_snr_earth": snr}


def in_hz_stratum(population, expectation, panel):
    """The habitable-zone stratum."""
    return bool(
        population in panel["classes"]
        and expectation["mid_period_days"] <= panel["max_mid_period_days"]
        and expectation["expected_snr_earth"] >= panel["min_expected_snr_earth"]
    )


def host_hz(host, stellar, features, metadata, panel):
    """A built K or M host's mid-zone Earth and stratum membership, from its own curve."""
    kp = host["star_light"].get("kepler_band_mag")
    levels = list(metadata["normalization_by_quarter"].values())
    if (
        host.get("population") not in panel["classes"]
        or kp is None
        or not levels
        or "duty_cycle" not in features
    ):
        return None
    expectation = hz_expectation(stellar, features, kp, float(np.median(levels)))
    return expectation | {
        "stratum": in_hz_stratum(host["population"], expectation, panel)
    }


def hz_census(frame, panel):
    """The realism census's K and M curves."""
    census = pd.read_parquet(DESIGN / "lane-noise-census/features.parquet")
    levels = pd.read_parquet(DESIGN / "lane-noise-census/levels.parquet").dropna(
        subset=["level"]
    )
    level = levels.groupby("source_id")["level"].median()
    colours = frame.set_index("source_id").bp_rp
    census = census[census.error.isna() & census.population.isin(panel["classes"])]
    rows = []
    for r in census.to_dict("records"):
        star = pop.catalogue_stellar(r["teff_gspphot"], r["logg_gspphot"])
        kp = kepler_band_mag(r["g_mag"], colours.get(r["source_id"]))
        expectation = hz_expectation(star, r, kp, level.get(r["source_id"], np.nan))
        rows.append({"source_id": r["source_id"], "population": r["population"],
                     **expectation})  # fmt: skip
    rows = pd.DataFrame(rows)
    return rows.assign(
        stratum=[
            in_hz_stratum(row.population, row._asdict(), panel)
            for row in rows.itertuples(index=False)
        ]
    )


def hz_fields(body, truth):
    """Insolation (Earth's = 1) at a planet's orbit, and whether it lies in the habitable zone."""
    a_au = body["a_rs"] * truth["radius_solar"] * pop.R_SUN / AU
    insolation = luminosity_solar(truth) / a_au**2
    return {
        "insolation_earth": insolation,
        "in_conservative_hz": bool(
            hz_flux(truth["teff"], "maximum_greenhouse")
            <= insolation
            <= hz_flux(truth["teff"], "runaway_greenhouse")
        ),
    }


def in_hz_box(body, truth, panel):
    """Whether a planet lies in the panel's box: the conservative zone and the panel's radius range."""
    low, high = panel["radius_range_earth"]
    return bool(
        hz_fields(body, truth)["in_conservative_hz"]
        and low <= body["radius_earth"] <= high
    )


def hz_system(truth, rng, panel, metadata):
    """A transiting small planet in the host's conservative habitable zone."""
    roll = rng.random()
    fractions = panel["multi_fractions"]
    size = (
        3
        if roll < fractions["3"]
        else 2
        if roll < fractions["3"] + fractions["2"]
        else 1
    )
    ld = pop.limb_darkening(truth["teff"], truth["logg"], ld_interpolators(), rng)
    inner, outer = hz_edges_au(truth)
    cut = (config()["detection"]["period_max_days"] / 365.25) ** (2 / 3) * truth[
        "mass_solar"
    ] ** (1 / 3)
    beyond = panel["beyond_range_fraction"]
    if inner < cut < outer:
        segments = [(inner, cut, 1 - beyond), (cut, outer, beyond)]
    else:
        segments = [(inner, outer, 1.0)]
    low, high = panel["radius_range_earth"]
    primary = None
    for _ in range(50):
        lo, hi, share = segments[0] if rng.random() < segments[0][2] else segments[-1]
        a = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        # Proposal density per ln R per ln P (ln P = 1.5 ln a).
        density = share / math.log(hi / lo) / 1.5 / math.log(high / low)
        radius = math.exp(rng.uniform(math.log(low), math.log(high)))
        period = orbital_period(a, truth["mass_solar"])
        primary = pop.planet(
            truth, ld, rng, period=period, radius=radius, multi=size > 1
        )
        if primary is not None:
            break
    require(primary is not None, "No physical habitable-zone orbit")
    planets = [primary | {"role": "primary"}]
    if size > 1:
        planets += [
            p | {"role": "companion"}
            for p in pop.companions(primary, truth, ld, rng, size - 1)
        ]
    pop.assign_ttvs(planets, rng)
    for body in planets:
        body["epoch_bkjd"] = float(
            metadata["first_valid_bkjd"] + rng.uniform(0, body["period_days"])
        )
        body |= hz_fields(body, truth)
    selection = {
        "hz_inner_au": inner,
        "hz_outer_au": outer,
        "search_range_edge_au": cut,
        "system_size_drawn": size,
        "box_rate": primary["occurrence"] * primary["transit_probability"] / density,
    }
    return planets, selection


def make_hz_planet(c, kind):
    """A habitable-zone stratum host's panel case."""
    host = c["host"]
    index = c["hz_index"] = c.get("hz_index", -1) + 1
    rng = rng_for("hz", host["source_id"], index)
    panel = c["plan"]["hz_panel"]
    planets, selection = hz_system(host["truth"], rng, panel, c["metadata"])
    weight = host["host_weight"] / panel["cases_per_host"] * selection["box_rate"]
    arrays, metadata, signals, _ = inject_system(
        host, c["prepared"], c["raw"], c["metadata"], c["features"], planets,
        lambda body: {"population_weight": weight},
    )  # fmt: skip
    scope = (
        "Small planet in the conservative habitable zone of a quiet K or M lane host "
        "(habitable-zone stratum, drawn in the zone)"
    )
    truth = {"signals": signals, "selection": selection | host["hz_selection"],
             "system_size": len(planets), "scope": scope}  # fmt: skip
    return arrays, metadata, truth


MAKERS = {
    "control_original": make_original,
    "synthetic_planet": make_planet,
    "synthetic_binary": make_binary,
    "synthetic_blend": make_binary,
    "control_inverted": make_transformed,
    "control_scrambled": make_transformed,
    "control_artifact": make_artifact,
    "contamination": make_real,
    "known_planet": make_real,
    "validated_boi": make_real,
    "hz_planet": make_hz_planet,
    **{f"artifact_{name}": make_artifact_copy for name in artifacts.CLASSES},
}


def folded_duration(prepared, body):
    """Total eclipse duration from the folded light curve (FWHM x 1.3)."""
    good = prepared["mask"]
    t, f = prepared["time"][good], prepared["flux"][good]
    period = body["period_days"]
    x = phase(t, period, body["epoch_bkjd"]) / period
    width = max(EXPOSURE_DAYS / period, 0.002)
    bins = np.arange(-0.5, 0.5 + width, width)
    index = np.digitize(x, bins)
    level = np.array(
        [
            np.median(f[index == i]) if (index == i).sum() >= 3 else np.nan
            for i in range(1, len(bins))
        ]
    )
    centres = 0.5 * (bins[1:] + bins[:-1])
    base = np.nanmedian(level[np.abs(centres) > 0.2])
    core = np.nanmin(level[np.abs(centres) < 0.05])
    half = base - 0.5 * (base - core)
    below = np.abs(centres) < 0.1
    low = centres[below & (level < half)]
    if not len(low):
        return float(min(0.08 * period, 0.4))
    return float(min(1.3 * (low.max() - low.min() + width) * period, 0.3 * period))


def choose_hosts(frame, plan):
    """Lane-proportional hosts with floors for rare classes; weights undo them."""
    rng = rng_for("hosts")
    shares = frame.population.value_counts()
    total = plan["planet_hosts"]
    wanted = {
        c: max(plan["class_floors"].get(c, 0), int(round(total * n / len(frame))))
        for c, n in shares.items()
    }
    chosen = []
    for c, n in wanted.items():
        group = frame[frame.population == c]
        take = group.sample(n=min(n, len(group)), random_state=int(rng.integers(2**31)))
        chosen.append(take.assign(host_weight=len(group) / len(take)))
    return pd.concat(chosen).sample(frac=1, random_state=int(rng.integers(2**31)))


def host_record(row):
    return {
        "source_id": str(row.source_id),
        "kic": None if pd.isna(row.kic) else int(row.kic),
        "parent_kic": int(row.origin_tpf_kic),
        "ra_deg": float(row.ra_deg),
        "dec_deg": float(row.dec_deg),
        "g_mag": float(row.phot_g_mean_mag),
        "bp_rp": None if pd.isna(row.bp_rp) else float(row.bp_rp),
        "teff_gspphot": None if pd.isna(row.teff_gspphot) else float(row.teff_gspphot),
        "logg_gspphot": None if pd.isna(row.logg_gspphot) else float(row.logg_gspphot),
        "population": row.population,
        "tpf_relation": row.tpf_relation,
        "family": str(int(row.origin_tpf_kic)),
        "partition": partition(str(int(row.origin_tpf_kic))),
        "host_weight": float(getattr(row, "host_weight", 1.0)),
    }


def known_signals():
    """Signals already found on lane sources (survey and M/K BOIs)."""
    import csv

    known = {}
    for name, key in (("survey", "boi_id"), ("mk", "signal_id")):
        for row in csv.DictReader(
            open(DISPOSITIONS / f"{name}-dispositions-20261025.csv")
        ):
            known.setdefault(row["source_id"], []).append(
                {
                    "id": row[key],
                    "period_days": float(row["period_days"]),
                    "category": row["category"],
                }
            )
    return known


def lane_jobs(frame, plan, out, known):
    hosts = choose_hosts(frame, plan)
    order = list(hosts.itertuples())
    jobs = []
    controls = set(range(plan["inverted_controls"]))
    scrambled = set(range(plan["scrambled_controls"]))
    binaries = set(range(len(order) - plan["synthetic_binaries"], len(order)))
    blends = set(
        range(
            len(order)
            - plan["synthetic_binaries"]
            - plan["synthetic_blend_candidates"],
            len(order) - plan["synthetic_binaries"],
        )
    )
    first = plan["artifact_first_host"]
    copies = len(artifacts.CLASSES) * plan["artifact_copies_per_class"]
    for i, row in enumerate(order):
        host = host_record(row) | {"known_signals": known.get(str(row.source_id), [])}
        job = {"host": host, "out": str(out), "plan": plan}
        kinds = ["control_original", "synthetic_planet"]
        if i < plan["second_planet_hosts"]:
            kinds.append("synthetic_planet")
        if i in controls:
            kinds.append("control_inverted")
        if i in scrambled:
            kinds.append("control_scrambled")
        if i in binaries:
            kinds.append("synthetic_binary")
        if i in blends:
            kinds.append("synthetic_blend")
        pairs = ("control_original", "synthetic_planet")
        if first <= i < first + copies:
            j = i - first
            kinds.append("artifact_" + artifacts.CLASSES[j % len(artifacts.CLASSES)])
            job["artifact_pair"] = pairs[(j // len(artifacts.CLASSES)) % 2]
        elif first + copies <= i < first + copies + plan["q2_comb_extra_hosts"]:
            kinds.append("artifact_q2_comb")
            job["artifact_pair"] = pairs[(i - first - copies) % 2]
        jobs.append(job | {"kinds": kinds})
    return jobs


def contamination_jobs(frame, plan, out, known, exclude, counts, key):
    """Real leakage of the TPF target's signal into lane sources: `counts` per label, apart from `exclude`."""
    by_id = frame.set_index("source_id")
    rng = rng_for(key)
    jobs = []
    eb = pd.read_parquet(DESIGN / "leakage-eclipsing-binaries.parquet")
    eb = eb[(eb.snr > plan["contamination_min_snr"]) & ~eb.source_id.isin(exclude)]
    ebcat = pd.read_csv(EB_CATALOGUE).drop_duplicates("KIC").set_index("KIC")
    koi = pd.read_parquet(DESIGN / "leakage-kois.parquet")
    koi = koi[(koi.snr > plan["contamination_min_snr"]) & ~koi.source_id.isin(exclude)]
    pools = {
        "eclipsing_binary": eb.assign(label="eclipsing_binary"),
        "koi_false_positive": koi[koi.disposition == "FALSE POSITIVE"].assign(
            label="koi_false_positive"
        ),
        "koi_planet": koi[koi.disposition != "FALSE POSITIVE"].assign(
            label="koi_planet"
        ),
    }
    taken = set()
    for label, pool in pools.items():
        pool = pool.drop_duplicates("source_id")
        pool = pool[~pool.source_id.isin(taken)]
        n = min(counts[label], len(pool))
        strata = pd.qcut(
            np.log10(pool.depth_ppm.clip(lower=1)), 5, labels=False, duplicates="drop"
        )
        take = pool.groupby(strata, group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), n // 5 + 1), random_state=int(rng.integers(2**31))
            )
        )
        for row in take.head(n).itertuples():
            taken.add(row.source_id)
            source = by_id.loc[row.source_id]
            host = host_record(source.to_frame().T.assign(source_id=row.source_id).iloc[0]) | {
                "known_signals": known.get(str(row.source_id), []), "host_weight": None,
            }  # fmt: skip
            if label == "eclipsing_binary":
                target = ebcat.loc[row.target]
                signal = {
                    "kind": "contamination_eclipsing_binary", "period_days": float(target.Per),
                    "epoch_bkjd": float(target.BJD0) - 54833.0, "duration_days": None,
                    "source_kic": int(row.target), "morphology": float(target.Morph),
                }  # fmt: skip
            else:
                signal = {
                    "kind": f"contamination_{label}", "period_days": float(row.period),
                    "epoch_bkjd": float(row.epoch), "duration_days": float(row.duration_h) / 24,
                    "source_kic": int(row.target), "koi": row.koi,
                    "fp_flags": {"ss": int(row.fp_ss), "co": int(row.fp_co), "nt": int(row.fp_nt), "ec": int(row.fp_ec)},
                }  # fmt: skip
            signal |= {
                "leak_depth_ppm": float(row.depth_ppm), "leak_snr": float(row.snr),
                "expected_disposition": "false_positive",
            }  # fmt: skip
            jobs.append(
                {
                    "host": host,
                    "kinds": ["contamination"],
                    "signals": [signal],
                    "out": str(out),
                    "plan": plan,
                    "role": f"contamination-{label}",
                    "scope": "Real KBonus light curve showing its TPF target's signal (leakage); the signal is not on this source.",
                }  # fmt: skip
            )
    return jobs


def artifact_jobs(frame, plan, out, known, exclude):
    """Census curves most affected by the field-wide artifacts the production mask leaves."""
    donors = pd.read_parquet(artifacts.MODELS / "donors.parquet")
    donors = donors.assign(excess=artifacts.unmasked_excess(donors))
    donors = donors[donors.point.notna() & ~donors.source_id.isin(exclude)]
    donors = donors.sort_values(["excess", "unmasked_active_dips"], ascending=False)
    by_id = frame.set_index("source_id")
    jobs = []
    for row in donors.head(plan["artifact_controls"]).itertuples():
        source = by_id.loc[row.source_id]
        host = host_record(source.to_frame().T.assign(source_id=row.source_id).iloc[0]) | {
            "known_signals": known.get(str(row.source_id), []), "host_weight": None,
        }  # fmt: skip
        artifact = {
            "unmasked_active_dips": int(row.unmasked_active_dips), "unmasked_active_bins": int(row.unmasked_active_bins),
            "quiet_dips": int(row.quiet_dips), "quiet_bins": int(row.quiet_bins), "excess_dips": float(row.excess),
        }  # fmt: skip
        jobs.append(
            {
                "host": host,
                "kinds": ["control_artifact"],
                "artifact": artifact,
                "out": str(out),
                "plan": plan,
                "role": "artifact",
            }
        )
    return jobs


def known_planet_jobs(plan, out):
    """KBonus light curves of confirmed Kepler planets."""
    confirmed = pd.DataFrame(json.loads(CONFIRMED.read_text()))
    koi = pd.DataFrame(json.loads(KOI_TABLE.read_text()))
    files = {int(p.name.split("kic-")[1][:9]): p for p in KBONUS_FITS.glob("*.fits")}
    confirmed = confirmed[
        confirmed.kic_id.isin(files)
        & confirmed.period_days.notna()
        & confirmed.epoch_bkjd.notna()
    ]
    stars = koi.drop_duplicates("kepid").set_index("kepid")
    rng = rng_for("known-planets")
    systems = []
    for kic, group in confirmed.groupby("kic_id"):
        hard = (
            (group.duration_hours > 6).any() or (group.depth_ppm < 300).any() or (group.period_days < 1).any()
            or (group.period_days > 50).any() or len(group) > 1
        )  # fmt: skip
        systems.append((kic, group, 3.0 if hard else 1.0))
    p = np.array([s[2] for s in systems])
    pick = rng.choice(
        len(systems),
        size=min(plan["known_planet_hosts"], len(systems)),
        replace=False,
        p=p / p.sum(),
    )
    jobs = []
    for i in sorted(pick):
        kic, group, _ = systems[i]
        star = stars.loc[kic] if kic in stars.index else None
        signals = [
            {
                "kind": "known_planet",
                "name": r.kepler_name,
                "period_days": float(r.period_days),
                "epoch_bkjd": float(r.epoch_bkjd),
                "duration_days": float(r.duration_hours) / 24
                if r.duration_hours == r.duration_hours
                else None,
                "catalogue_depth_ppm": None
                if pd.isna(r.depth_ppm)
                else float(r.depth_ppm),
                "radius_earth": None
                if pd.isna(r.planet_radius_earth)
                else float(r.planet_radius_earth),
                "impact_parameter": None
                if pd.isna(r.impact_parameter)
                else float(r.impact_parameter),
                "expected_disposition": "planet",
            }  # fmt: skip
            for r in group.itertuples()
        ]
        others = koi[(koi.kepid == kic) & (koi.koi_disposition == "CANDIDATE")]
        host = {
            "source_id": f"kic{kic}", "kic": int(kic), "parent_kic": int(kic), "family": f"kic{kic}",
            "partition": partition(f"kic{kic}"), "population": "kepler_target", "host_weight": None,
            "teff_gspphot": None if star is None or pd.isna(star.koi_steff) else float(star.koi_steff),
            "logg_gspphot": None if star is None or pd.isna(star.koi_slogg) else float(star.koi_slogg),
            "g_mag": None if star is None or pd.isna(star.koi_kepmag) else float(star.koi_kepmag),
            "known_signals": [{"id": r.kepoi_name, "period_days": float(r.koi_period), "category": "KOI_CANDIDATE"} for r in others.itertuples()],
        }  # fmt: skip
        jobs.append(
            {
                "host": host,
                "kinds": ["known_planet"],
                "signals": signals,
                "fits": str(files[kic]),
                "out": str(out),
                "plan": plan,
                "role": "known-planet",
                "scope": "KBonus light curve of a Kepler target with confirmed planets; stellar inputs from the KOI table.",
            }  # fmt: skip
        )
    return jobs


def hz_jobs(frame, plan, out, known, exclude):
    """The habitable-zone stratum's panel."""
    panel = plan["hz_panel"]
    census = hz_census(frame, panel)
    lane_counts = frame.population.value_counts()
    by_id = frame.set_index("source_id")
    jobs, sizes = [], {}
    for cls in panel["classes"]:
        group = census[census.population == cls]
        sizes[cls] = float(lane_counts.get(cls, 0) * group.stratum.mean())
        free = group[~group.source_id.isin(exclude)]
        pool = int(
            (
                free.expected_snr_earth.notna()
                & (free.mid_period_days <= panel["max_mid_period_days"])
            ).sum()
        )
        chosen = free[free.stratum].sort_values("expected_snr_earth", ascending=False)
        for rank, row in enumerate(chosen.itertuples(), 1):
            source = by_id.loc[row.source_id]
            selection = {
                "panel_rank": rank, "panel_pool": pool,
                "expected_snr_earth": float(row.expected_snr_earth),
                "mid_period_days": float(row.mid_period_days),
            }  # fmt: skip
            record = source.to_frame().T.assign(
                source_id=row.source_id, host_weight=sizes[cls] / len(chosen)
            )
            host = host_record(record.iloc[0]) | {
                "known_signals": known.get(row.source_id, []), "hz_selection": selection,
            }  # fmt: skip
            jobs.append({
                "host": host, "out": str(out), "plan": plan, "role": "hz-panel",
                "kinds": ["control_original"] + ["hz_planet"] * panel["cases_per_host"],
            })  # fmt: skip
    return jobs, sizes


def validated_jobs(frame, plan, out, known):
    path = DESIGN / "validated-planets.json"
    if not path.exists():
        return []
    by_id = frame.set_index("source_id")
    jobs = []
    for source_id, signals in json.loads(path.read_text()).items():
        source = by_id.loc[source_id]
        host = host_record(source.to_frame().T.assign(source_id=source_id).iloc[0]) | {
            "known_signals": known.get(source_id, []), "host_weight": None,
        }  # fmt: skip
        jobs.append(
            {
                "host": host,
                "kinds": ["validated_boi"],
                "signals": signals,
                "out": str(out),
                "plan": plan,
                "role": "validated",
                "scope": "Internally validated planets on lane sources (results/boi-validation).",
            }  # fmt: skip
        )
    return jobs


def git_state():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {"commit": head, "files": {p: digest(ROOT / p) for p in (
        "calibration/sample_builder.py", "calibration/signal_injection.py", "calibration/population.py", "calibration/lane_census.py",
        "calibration/artifacts.py",
    )}}  # fmt: skip


def build(out, plan, workers=3, limit=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    require(
        shutil.disk_usage(out).free > plan["minimum_free_gib"] * 1024**3,
        "Internal disk reserve reached",
    )
    (out / "light-curves").mkdir()
    (out / "truth").mkdir()
    frame = lane()
    frame["population"] = [
        pop.stellar_class(t, g)
        for t, g in zip(frame.teff_gspphot, frame.logg_gspphot, strict=True)
    ]
    known = known_signals()
    jobs = lane_jobs(frame, plan, out, known)
    used = {j["host"]["source_id"] for j in jobs}
    jobs += contamination_jobs(
        frame, plan, out, known, used, plan["contamination"], "contamination"
    )
    used |= {j["host"]["source_id"] for j in jobs}
    jobs += contamination_jobs(
        frame,
        plan,
        out,
        known,
        used,
        plan["contamination_added"],
        "contamination-added",
    )
    used |= {j["host"]["source_id"] for j in jobs}
    jobs += artifact_jobs(frame, plan, out, known, used)
    jobs += validated_jobs(frame, plan, out, known)
    jobs += known_planet_jobs(plan, out)
    panel_jobs, strata = hz_jobs(
        frame, plan, out, known, {j["host"]["source_id"] for j in jobs}
    )
    jobs += panel_jobs
    if limit:
        rng = rng_for("smoke")
        jobs = [
            jobs[i]
            for i in sorted(
                rng.choice(len(jobs), size=min(limit, len(jobs)), replace=False)
            )
        ]
    models = {
        p.name: receipt(p) for p in sorted(artifacts.MODELS.iterdir()) if p.is_file()
    }
    write(
        out / "plan.json",
        plan | {"software": git_state(), "jobs": len(jobs), "artifact_models": models,
                "hz_stratum_lane_curves": strata},
    )  # fmt: skip
    shutil.copy(artifacts.MODELS / "validation.json", out / "artifact-validation.json")
    shutil.copy(DESIGN / "validated-planets.json", out / "validated-planets.json")
    rows, signals, failures, hosts = [], [], [], []
    single_threaded()
    with ProcessPoolExecutor(workers, initializer=single_threaded) as pool:
        for i, result in enumerate(pool.map(build_host, jobs, chunksize=4)):
            if "error" in result:
                failures.append(result)
                continue
            rows += result["rows"]
            signals += result["signals"]
            hosts.append({"source_id": result["source_id"]} | result["features"])
            if i % 250 == 0:
                print(
                    f"{i}/{len(jobs)} hosts, {len(rows)} cases, {len(failures)} failures",
                    flush=True,
                )
                require(
                    shutil.disk_usage(out).free > plan["minimum_free_gib"] * 1024**3,
                    "Internal disk reserve reached",
                )
    cases = lane_weights(pd.DataFrame(rows), len(frame), plan, strata)
    flat = pd.DataFrame([flat_signal(s) for s in signals])
    flat.to_parquet(out / "signals.parquet")
    pd.DataFrame(hosts).drop_duplicates("source_id").to_parquet(
        out / "host-features.parquet"
    )
    write(out / "failures.json", failures)
    cases = write_batches(out, cases, plan)
    summary = {
        "cases": int(len(cases)),
        "by_kind": cases.kind.value_counts().to_dict(),
        "by_partition": cases.partition.value_counts().to_dict(),
        "signals": len(signals),
        "failures": len(failures),
        "habitable_zone": hz_summary(cases, flat, strata),
        "bytes": int(
            sum(
                f.stat().st_size
                for f in (out / "light-curves").rglob("*")
                if f.is_file()
            )
        ),
    }
    write(out / "summary.json", summary)
    return summary


def hz_summary(cases, signals, strata):
    """The habitable-zone stratum as built."""
    planets = signals[signals.kind == "planet"].merge(
        cases[["case_id", "population"]], on="case_id"
    )
    zone = planets[planets.in_conservative_hz.fillna(False).astype(bool)]
    originals = cases[cases.kind == "control_original"]
    panel = cases.kind == "hz_planet"

    def counts(frame):
        return {
            "planets": int(len(frame)),
            "primaries": int((frame.role == "primary").sum()),
            "beyond_search_range": int(
                (frame.control_class == "outside_period_range").sum()
            ),
            "detectable": int(frame.detectable.sum()),
            "detectable_by_class_radius": {
                f"{cls} {band}": int(g.detectable.sum())
                for (cls, band), g in frame.groupby(
                    ["population", pd.cut(frame.radius_earth, [0, 1, 1.25, 1.5, 1.75, 2.0, 30])],
                    observed=True,
                )
            },
        }  # fmt: skip

    return {
        "stratum_lane_curves": strata,
        "panel_hosts": cases[panel].drop_duplicates("source_id").population.value_counts().to_dict(),
        "main_sample_stratum_hosts": originals[originals.hz_stratum & ~originals.source_id.isin(cases.source_id[panel])]
        .population.value_counts().to_dict(),
        "panel_cases": int(panel.sum()),
        "panel_cases_by_partition": cases[panel].partition.value_counts().to_dict(),
        "drawn_in_the_zone": counts(zone[zone.case_kind == "hz_planet"]),
        "occurrence_drawn_in_a_zone": counts(zone[zone.case_kind == "synthetic_planet"]),
    }  # fmt: skip


def leak_pools(plan):
    """Lane curves leaking each label's TPF-target signal above the S/N cut."""
    cut = plan["contamination_min_snr"]
    eb = pd.read_parquet(DESIGN / "leakage-eclipsing-binaries.parquet")
    koi = pd.read_parquet(DESIGN / "leakage-kois.parquet")
    koi = koi[koi.snr > cut]
    taken = set(eb[eb.snr > cut].source_id)
    false_positives = set(koi[koi.disposition == "FALSE POSITIVE"].source_id) - taken
    planets = (
        set(koi[koi.disposition != "FALSE POSITIVE"].source_id)
        - taken
        - false_positives
    )
    return {
        "contamination-eclipsing_binary": len(taken),
        "contamination-koi_false_positive": len(false_positives),
        "contamination-koi_planet": len(planets),
    }


def lane_weights(cases, lane_size, plan, strata):
    """`lane_weight`: how many of the lane's curves a negative case stands for."""
    weight = pd.Series(np.nan, index=cases.index)
    original = cases.kind == "control_original"
    weight[original] = cases.host_weight[original]
    for cls, size in strata.items():
        sel = original & cases.hz_stratum & (cases.population == cls)
        weight[sel] = size / max(sel.sum(), 1)
    for role, count in leak_pools(plan).items():
        sel = cases.role.str.startswith(role)
        weight[sel] = count / max(sel.sum(), 1)
    binaries = plan["eclipsing_binary_fraction"] * lane_size
    drawn = original & (cases.role != "hz-panel")
    foreign = float((cases.star_flux_fraction[drawn] < 0.99).mean())
    for kind, total in (
        ("synthetic_binary", binaries),
        ("synthetic_blend", binaries * foreign),
    ):
        sel = cases.kind == kind
        weight[sel] = total / max(sel.sum(), 1)
    census = artifacts.models()["donors"].point.notna().sum()
    weight[cases.kind == "control_artifact"] = lane_size / census
    rates = artifacts.lane_rates()
    for name in artifacts.CLASSES:
        sel = (cases.kind == f"artifact_{name}") & (
            cases.artifact_pair == "control_original"
        )
        weight[sel] = rates[name] * lane_size / max(sel.sum(), 1)
    return cases.assign(lane_weight=weight)


def write_batches(out, cases, plan):
    """Runner manifests per partition."""
    out = Path(out)
    quick_hosts = {
        h for h in cases.source_id.unique()
        if int(keyed("quick", h)[:8], 16) / 0xFFFFFFFF < plan["quick_fraction"]
    }  # fmt: skip
    cases = cases.assign(
        quick=cases.source_id.isin(quick_hosts) | (cases.kind == "validated_boi")
    )
    cases.to_parquet(out / "cases.parquet")
    batches = out / "manifests"
    batches.mkdir(exist_ok=True)
    sets = [("all", cases), ("quick", cases[cases.quick])]
    for part in ("development", "holdout"):
        for label, subset in sets:
            chosen = subset[subset.partition == part]
            with open(batches / f"{part}-{label}.jsonl", "w") as stream:
                for row in chosen.itertuples():
                    stream.write(
                        json.dumps({"case_id": row.case_id, "prepared": row.prepared})
                        + "\n"
                    )
    return cases


def audit_case(row):
    """Rebuild one case from its truth record alone and compare with storage."""
    truth = read(row["truth"])
    arrays, meta = load_packet(row["prepared"])
    raw_arrays, info = lc2_arrays(truth["source_id"])
    prepared, raw, base = prepare(raw_arrays, truth["host"], meta["stellar"], info)
    kind = row["kind"]
    rebuilt, rebuilt_meta = prepared, base
    if kind in ("control_inverted", "control_scrambled"):
        transform = "inverted" if kind == "control_inverted" else "block_scrambled"
        rebuilt = transformed_control(
            prepared, base, transform, truth["seed"], PLAN["scramble_block_cadences"]
        )
    elif kind in (
        "synthetic_planet",
        "synthetic_binary",
        "synthetic_blend",
        "hz_planet",
    ) or kind.startswith("artifact_"):
        model = np.ones(len(prepared["time"]))
        for signal in truth["signals"]:
            maker = transit_flux if signal["kind"] == "planet" else binary_flux
            model *= maker(
                prepared["time"], EXPOSURE_DAYS, signal, signal["epoch_bkjd"]
            )
        light = truth["host"]["star_light"]["fraction_by_quarter"]
        fraction = star_fraction(prepared["quarter"], light)
        if kind == "synthetic_blend":
            fraction = np.ones(len(fraction))
        flat, raw_factor = (
            artifacts.replay(truth["artifact"], prepared)
            if kind.startswith("artifact_")
            else (None, None)
        )
        rebuilt, rebuilt_meta, _ = with_signal(
            prepared, base, raw, model, fraction, raw_factor, flat
        )
    good = arrays["mask"]
    return {
        "case_id": row["case_id"],
        "kind": kind,
        "arrays_identical": all(
            np.array_equal(arrays[n], rebuilt[n], equal_nan=True) for n in arrays
        ),
        "normalization_matches": rebuilt_meta["normalization_by_quarter"]
        == meta["normalization_by_quarter"],
        "max_abs_difference": float(
            np.nanmax(np.abs(rebuilt["flux"][good] - arrays["flux"][good]), initial=0.0)
        ),
    }


def audit(out, per_kind=200, workers=4):
    """audit.json: sampled cases of every lane kind rebuilt array by array from their truth record."""
    out = Path(out)
    cases = pd.read_parquet(out / "cases.parquet")
    picks = [
        g.sample(
            min(len(g), per_kind if kind == "synthetic_planet" else per_kind // 5),
            random_state=3,
        )
        for kind, g in cases[cases.kind != "known_planet"].groupby("kind")
    ]
    rows = pd.concat(picks).to_dict("records")
    single_threaded()
    with ProcessPoolExecutor(workers, initializer=single_threaded) as pool:
        d = pd.DataFrame(list(pool.map(audit_case, rows, chunksize=4)))
    summary = {
        "audited": len(d),
        "by_kind": d.kind.value_counts().to_dict(),
        "all_arrays_identical": bool(d.arrays_identical.all()),
        "normalization_matches": bool(d.normalization_matches.all()),
        "max_abs_flux_difference": float(d.max_abs_difference.max()),
    }
    write(out / "audit.json", summary)
    return summary


def register(out, registry=ROOT / "calibration/registered-sample.json"):
    """calibration/registered-sample.json: receipts of the sample's records."""
    out = Path(out)
    files = ["plan.json", "summary.json", "quality.json", "audit.json", "cases.parquet", "signals.parquet",
             "host-features.parquet", "failures.json", "artifact-validation.json",
             "validated-planets.json"]  # fmt: skip
    record = {
        "schema": 1,
        "name": out.name,
        "state": "BUILT",
        "root": str(out.resolve()),
        "design": "docs/keplerX-calibration-sample.md",
        "records": {
            name: receipt(out / name) for name in files if (out / name).exists()
        },
        "manifests": {
            p.stem: receipt(p) for p in sorted((out / "manifests").glob("*.jsonl"))
        },
    }
    write(registry, record)
    return record


def flat_signal(s):
    keep = {k: v for k, v in s.items() if not isinstance(v, (list, dict))}
    if isinstance(s.get("ttv"), dict):
        keep["ttv_amplitude_days"] = s["ttv"]["amplitude_days"]
    return keep


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("--hosts", type=int, default=PLAN["planet_hosts"])
    b.add_argument("--workers", type=int, default=3)
    b.add_argument(
        "--limit", type=int, default=None, help="Smoke run: a random subset of jobs"
    )
    w = sub.add_parser(
        "manifests", help="Rewrite the runner manifests of a built sample"
    )
    w.add_argument("out", type=Path)
    r = sub.add_parser(
        "register", help="Record the sample in calibration/registered-sample.json"
    )
    r.add_argument("out", type=Path)
    a = sub.add_parser(
        "audit", help="Rebuild cases from truth and compare (audit.json)"
    )
    a.add_argument("out", type=Path)
    a.add_argument("--per-kind", type=int, default=200)
    a.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(args.out, args.per_kind, args.workers), indent=1))
        return
    if args.command == "register":
        print(json.dumps(register(args.out)["records"], indent=1))
        return
    if args.command == "manifests":
        cases = write_batches(
            args.out, pd.read_parquet(args.out / "cases.parquet"), PLAN
        )
        print(cases.groupby(["partition", "quick"]).size().to_dict())
        return
    plan = PLAN | {"planet_hosts": args.hosts}
    print(json.dumps(build(args.out, plan, args.workers, args.limit), indent=1))


if __name__ == "__main__":
    main()
