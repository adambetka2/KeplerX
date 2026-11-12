"""Experimental, period-first shallow transit proposals from prepared LC2 curves."""

import argparse
import csv
import ctypes
import functools
import hashlib
import json
import os
import platform
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.constants import G
from scipy.ndimage import median_filter

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
CONFIG = FOLDER / "config.json"
BLS_SOURCE = FOLDER / "bls.c"
BLS_COMMAND = (
    "cc",
    "-O3",
    "-mcpu=native" if platform.machine() in {"arm64", "aarch64"} else "-march=native",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-ffp-contract=off",
    "-shared",
    "-fPIC",
)
RECORD = np.dtype(
    [
        ("time", "<f8"),
        ("flux", "<f4"),
        ("error", "<f4"),
        ("quarter", "<i2"),
        ("cadence", "<i4"),
    ]
)
FIELDS = (
    "source_id",
    "search_rank",
    "search_band",
    "period_days",
    "t0_bkjd",
    "duration_hours",
    "depth_ppm",
    "search_snr",
    "observed_events",
    "observed_quarters",
    "in_transit_points",
    "inverse_same_period_snr",
    "dip_inverse_snr_ratio",
    "harmonic_delta_log_likelihood",
    "odd_even_sigma",
    "positive_cycle_fraction",
)
ELIGIBILITY_FIELDS = ("source_id", "status", "reason", "elapsed_seconds")


class ShallowIneligible(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_fsc(path):
    with Path(path).open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise ValueError(f"Truncated FSC header: {path}")
        magic, source_id, view, count = struct.unpack("<8sQQQ", header)
        data = stream.read()
    if magic != b"FSC_LC2\0" or view != 0 or len(data) != count * RECORD.itemsize:
        raise ValueError(f"Invalid PSF_FLAT FSC: {path}")
    if path.name != f"{source_id}.psf_flat.fsc":
        raise ValueError(f"FSC source identity disagrees with filename: {path}")
    records = np.frombuffer(data, dtype=RECORD)
    if (
        count < 100
        or not np.isfinite(records["time"]).all()
        or not np.isfinite(records["flux"]).all()
        or not np.isfinite(records["error"]).all()
        or not (records["error"] > 0).all()
        or not (np.diff(records["time"]) >= 0).all()
    ):
        raise ValueError(f"Invalid FSC photometry: {path}")
    return source_id, records, hashlib.sha256(header + data).hexdigest()


@functools.cache
def bls_library():
    compiler = subprocess.run(
        [BLS_COMMAND[0], "--version"], capture_output=True, check=True
    ).stdout.decode()
    target = subprocess.run(
        [*BLS_COMMAND, "-dM", "-E", "-x", "c", "/dev/null"],
        capture_output=True,
        check=True,
    ).stdout
    compilation = {
        "command": list(BLS_COMMAND),
        "architecture": platform.machine(),
        "compiler": compiler,
        "target_macros_sha256": hashlib.sha256(target).hexdigest(),
        "source_sha256": digest(BLS_SOURCE),
    }
    key = hashlib.sha256(json.dumps(compilation, sort_keys=True).encode())
    library = ROOT / "dump/shallow-bls" / f"bls-{key.hexdigest()[:16]}.so"
    if not library.is_file():
        library.parent.mkdir(exist_ok=True)
        partial = library.with_name(f"{library.name}.{os.getpid()}")
        subprocess.run(
            [*BLS_COMMAND, "-o", str(partial), str(BLS_SOURCE), "-lm"], check=True
        )
        partial.replace(library)
    handle = ctypes.CDLL(str(library))
    handle.build_manifest = {
        "path": str(library),
        "sha256": digest(library),
        "compilation": compilation,
    }
    array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    handle.bls.restype = ctypes.c_int
    handle.bls.argtypes = [
        ctypes.c_int,
        array,
        array,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        array,
    ]
    return handle


def box_search(time, flux, noise, periods, durations, oversample=10):
    """Return the power, depth, duration and transit time that astropy's
    BoxLeastSquares(time, flux, dy=noise).power(periods, durations, objective="snr")
    returns on its arm64 build, bit for bit. Explicit fma() preserves that
    rounding contract on other architectures."""
    periods = np.asarray(periods, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    t_ref = np.min(time)
    out = np.empty((7, len(periods)))
    status = bls_library().bls(
        len(time),
        time - t_ref,
        flux - np.median(flux),
        1.0 / noise**2,
        len(periods),
        periods,
        len(durations),
        durations,
        oversample,
        out,
    )
    if status:
        raise ValueError(f"Box least squares rejected its inputs with status {status}")
    power, depth, _, duration, phase, _, _ = out
    return power, depth, duration, phase + t_ref


@functools.cache
def biweight_slider():
    """Use Wotan's location estimator without its flux-sign mask or empty gap windows."""
    from numba import njit
    from wotan.location_estimates import location_iter

    @njit(cache=True)
    def slide(time, flux, window, cval):
        trend = np.empty(len(flux))
        lower = upper = 0
        for index in range(len(time)):
            while time[lower] < time[index] - window / 2:
                lower += 1
            while upper < len(time) and time[upper] <= time[index] + window / 2:
                upper += 1
            trend[index] = location_iter(flux[lower:upper], cval, 1)
        return trend

    return slide


def detrend(time, flux, window, settings):
    if settings is None:
        return median_filter(flux, size=window, mode="nearest")
    if settings["method"] != "biweight":
        raise ValueError(f"Unknown detrending method: {settings['method']}")
    window_days, cval = settings["window_days"], settings["cval"]
    if not np.isfinite([window_days, cval]).all() or min(window_days, cval) <= 0:
        raise ValueError("Positive finite biweight window and tuning constant required")
    trend = biweight_slider()(time, flux, window_days, cval)
    if not np.isfinite(trend).all():
        raise ShallowIneligible("NONFINITE_BIWEIGHT_TREND")
    return trend


def whiten(records, window, min_quarter_cadences, detrending=None):
    quarter_values, counts = np.unique(records["quarter"], return_counts=True)
    retained = quarter_values[counts >= min_quarter_cadences]
    use = np.isin(records["quarter"], retained)
    records = records[use]
    time = records["time"].astype(np.float64)
    flux = records["flux"].astype(np.float64)
    quarter = records["quarter"]
    noise = np.empty(len(flux), dtype=np.float64)
    for value in np.unique(quarter):
        use = quarter == value
        flux[use] -= detrend(time[use], flux[use], window, detrending)
        centered = flux[use] - np.median(flux[use])
        sigma = 1.4826 * np.median(np.abs(centered))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ShallowIneligible("NONPOSITIVE_QUARTER_SCATTER")
        noise[use] = sigma
    return time, flux, noise, quarter


def duration_limit(periods, settings):
    density = settings["rho_min_g_cm3"] * 1000
    size = 1 + settings["radius_ratio_max"]
    exposure = settings["exposure_margin_days"]
    if not np.isfinite([density, size, exposure]).all() or min(density, size) <= 0:
        raise ValueError("Positive finite duration-envelope density and size required")
    if size < 1 or exposure < 0:
        raise ValueError("Nonnegative radius ratio and exposure margin required")
    axis = (G * density * (periods * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    return periods / np.pi * np.arcsin(np.minimum(1, size / axis)) + exposure


def search_grid(time, flux, noise, periods, config, oversample, diagnostics=None):
    durations = np.asarray(config["duration_days"], dtype=np.float64)
    envelope = config.get("duration_envelope")
    if envelope is None:
        return box_search(time, flux, noise, periods, durations, oversample)
    if np.any(np.diff(durations) <= 0) or durations[0] <= 0:
        raise ValueError("Duration envelope requires positive ascending durations")
    maximum = duration_limit(periods, envelope)
    counts = np.sum(durations[:, None] <= maximum[None, :], axis=0)
    if np.any(counts == 0):
        raise ValueError("Duration grid has no point below the declared envelope")
    result = [np.empty(len(periods)) for _ in range(4)]
    groups = []
    for count in np.unique(counts):
        selected = np.flatnonzero(counts == count)
        values = box_search(
            time, flux, noise, periods[selected], durations[:count], oversample
        )
        for combined, values_for_group in zip(result, values):
            combined[selected] = values_for_group
        groups.append({"durations": int(count), "period_trials": len(selected)})
    if diagnostics is not None:
        diagnostics["duration_envelope"] = {"settings": envelope, "groups": groups}
    return result


def coverage(time, quarter, period, epoch, duration):
    cycle = np.rint((time - epoch) / period).astype(np.int64)
    inside = np.abs(time - epoch - cycle * period) <= duration / 2
    return (
        len(np.unique(cycle[inside])),
        len(np.unique(quarter[inside])),
        int(inside.sum()),
    )


def transit_window(time, period, epoch, duration):
    cycle = np.rint((time - epoch) / period)
    return np.abs(time - epoch - cycle * period) <= duration / 2


def refine_peak(
    time, flux, noise, periods, index, config, diagnostics=None, periodogram_arrays=None
):
    lower = periods[max(0, index - 1)]
    upper = periods[min(len(periods) - 1, index + 1)]
    local = np.linspace(lower, upper, config["period_refine_points"])
    power, depth, duration, transit_time = search_grid(
        time,
        flux,
        noise,
        local,
        config,
        config.get("refine_phase_oversample", 10),
    )
    best = int(np.argmax(power))
    if diagnostics is not None:
        diagnostics["refinement"] = {
            "period_lower_days": float(lower),
            "period_upper_days": float(upper),
            "trial_count": len(local),
        }
    if periodogram_arrays is not None:
        periodogram_arrays.update(
            {
                f"refinement_{index}_{key}": value
                for key, value in {
                    "period_days": local,
                    "search_snr": power,
                    "depth_ppm": depth * 1e6,
                    "duration_days": duration,
                    "epoch_bkjd": transit_time,
                }.items()
            }
        )
    return (
        float(local[best]),
        float(transit_time[best]),
        float(duration[best]),
        float(depth[best]) * 1e6,
        float(power[best]),
    )


def decision(diagnostics, row, status, reasons=()):
    if diagnostics is not None:
        diagnostics.setdefault("decisions", []).append(
            {
                **row,
                "status": status,
                "reasons": list(reasons),
                "first_reason": next(iter(reasons), None),
            }
        )


def distinct_proposals(proposals, time, config, diagnostics=None):
    if config.get("sort_proposals_by_refined_snr", False):
        proposals = [
            {**row, "proposal_priority": priority}
            for priority, row in enumerate(
                sorted(proposals, key=lambda row: row["search_snr"], reverse=True), 1
            )
        ]
    chosen = []
    windows = []
    for index, row in enumerate(proposals):
        window = transit_window(
            time, row["period_days"], row["t0_bkjd"], row["duration_hours"] / 24
        )
        if any(
            np.count_nonzero(window & previous) / min(window.sum(), previous.sum())
            > config["maximum_transit_overlap"]
            for previous in windows
        ):
            decision(diagnostics, row, "REJECTED", ("OVERLAP",))
            continue
        chosen.append(row)
        windows.append(window)
        decision(diagnostics, row, "PROPOSAL")
        if len(chosen) == config["max_proposals_per_source"]:
            for remaining in proposals[index + 1 :]:
                decision(diagnostics, remaining, "REJECTED", ("PROPOSAL_QUOTA",))
            break
    return chosen


def overlaps_proposal(window, time, proposal, config):
    previous = transit_window(
        time,
        proposal["period_days"],
        proposal["t0_bkjd"],
        proposal["duration_hours"] / 24,
    )
    denominator = min(window.sum(), previous.sum())
    return bool(
        denominator
        and np.count_nonzero(window & previous) / denominator
        > config["maximum_transit_overlap"]
    )


def duration_controlled_grid(baseline, period_min, period_max, config):
    correlation = config["period_grid_correlation"]
    capacity = config["maximum_period_trials"]
    if type(capacity) is not int or capacity < 2:
        raise ValueError("Period trial capacity must be an integer of at least two")
    durations = np.asarray(config["duration_days"], dtype=np.float64)
    if (
        not np.isfinite(correlation)
        or not 0 < correlation < 1
        or not len(durations)
        or not np.isfinite(durations).all()
        or np.min(durations) <= 0
    ):
        raise ValueError("Finite correlation in (0, 1) and positive durations required")
    drift = 4 * (1 - correlation) * np.min(durations)
    size = (
        int(np.ceil(np.log(period_max / period_min) / np.log1p(drift / baseline))) + 1
    )
    if size > capacity:
        raise ValueError(f"Duration-controlled period grid requires {size} trials")
    return np.geomspace(period_min, period_max, size)


def period_grid(time, config):
    baseline = float(time[-1] - time[0])
    period_max = min(config["period_max_days"], baseline / 2)
    if period_max <= config["period_min_days"]:
        raise ShallowIneligible("INSUFFICIENT_BASELINE")
    if "period_grid_correlation" in config:
        return duration_controlled_grid(
            baseline, config["period_min_days"], period_max, config
        )
    size = config["period_grid_size"]
    if size < 2:
        raise ValueError("Period grid needs at least two points")
    return np.geomspace(config["period_min_days"], period_max, size)


def periodogram_diagnostics(diagnostics, time, periods, power, depth, duration, epoch):
    if diagnostics is None:
        return
    finite = np.isfinite(power) & np.isfinite(depth)
    diagnostics["grid"] = {
        "trial_count": len(periods),
        "baseline_days": float(time[-1] - time[0]),
        "period_min_days": float(periods[0]),
        "period_max_days": float(periods[-1]),
        "maximum_adjacent_phase_drift_days": float(
            np.max(np.diff(periods) / periods[:-1]) * (time[-1] - time[0])
        ),
    }
    diagnostics["periodogram"] = {
        "finite_trials": int(finite.sum()),
        "maximum_search_snr": float(np.max(power[finite])) if finite.any() else None,
    }
    if diagnostics.get("store_periodograms"):
        diagnostics["periodogram_arrays"] = {
            "period_days": periods,
            "search_snr": power,
            "depth_ppm": depth * 1e6,
            "duration_days": duration,
            "epoch_bkjd": epoch,
        }


def refined_hypothesis(time, quarter, fitted, hypotheses, config, diagnostics, peak):
    period, epoch, duration, depth, snr = fitted
    row = {
        "period_days": period,
        "t0_bkjd": epoch,
        "duration_hours": duration * 24,
        "depth_ppm": depth,
        "search_snr": snr,
    }
    reasons = []
    if depth < config["depth_ppm_min"] or depth > config["depth_ppm_max"]:
        reasons.append("DEPTH_DOMAIN")
    if snr < config["search_snr_min"]:
        reasons.append("SEARCH_SNR")
    if not reasons:
        if any(
            abs(period / prior["period_days"] - 1) < config["period_dedup_fraction"]
            for prior in hypotheses
        ):
            reasons.append("PERIOD_DUPLICATE")
        else:
            events, quarters, points = coverage(time, quarter, period, epoch, duration)
            row.update(
                observed_events=events,
                observed_quarters=quarters,
                in_transit_points=points,
            )
            if events < config["min_observed_events"]:
                reasons.append("COVERAGE_EVENTS")
            if quarters < config["min_observed_quarters"]:
                reasons.append("COVERAGE_QUARTERS")
    if diagnostics is not None:
        diagnostics["tested_peaks"].append({**peak, **row, "reasons": reasons})
    if reasons:
        decision(diagnostics, row, "REJECTED", reasons)
        return None
    return {"search_rank": len(hypotheses) + 1, **row}


def redundant_refinement(periods, index, hypotheses, config):
    endpoints = periods[[max(0, index - 1), min(len(periods) - 1, index + 1)]]
    return any(
        np.max(np.abs(endpoints / row["period_days"] - 1))
        < config["period_dedup_fraction"]
        for row in hypotheses
    )


def search_hypotheses(
    time, flux, noise, quarter, periods, power, depth, order, config, diagnostics
):
    hypotheses = []
    periodogram_arrays = (
        diagnostics["periodogram_arrays"]
        if diagnostics is not None and diagnostics.get("store_periodograms")
        else None
    )
    skipped = 0
    for position, index in enumerate(order):
        if position - skipped >= config.get("max_refined_peaks_per_band", len(order)):
            if diagnostics is not None:
                diagnostics["unrefined_eligible_trials_after_refinement_quota"] = (
                    len(order) - position
                )
            break
        peak = None
        if diagnostics is not None:
            peak = {
                "grid_index": int(index),
                "coarse_period_days": float(periods[index]),
                "coarse_search_snr": float(power[index]),
                "coarse_depth_ppm": float(depth[index]),
            }
        if config.get(
            "skip_redundant_refinement_intervals", False
        ) and redundant_refinement(periods, index, hypotheses, config):
            skipped += 1
            if diagnostics is not None:
                diagnostics.setdefault("excluded_refinement_intervals", []).append(
                    {
                        **peak,
                        "period_lower_days": float(periods[max(0, index - 1)]),
                        "period_upper_days": float(
                            periods[min(len(periods) - 1, index + 1)]
                        ),
                        "reason": "REFINEMENT_PERIOD_REDUNDANT",
                    }
                )
            continue
        fitted = refine_peak(
            time, flux, noise, periods, index, config, peak, periodogram_arrays
        )
        row = refined_hypothesis(
            time, quarter, fitted, hypotheses, config, diagnostics, peak
        )
        if row is None:
            continue
        hypotheses.append(row)
        if len(hypotheses) == config["max_search_hypotheses_per_source"]:
            if diagnostics is not None:
                diagnostics["unrefined_eligible_trials_after_quota"] = (
                    len(order) - position - 1
                )
            break
    return hypotheses


def measure_hypothesis(bls, time, flux, noise, row, band):
    period, duration, epoch = (
        row["period_days"],
        row["duration_hours"] / 24,
        row["t0_bkjd"],
    )
    stats = bls.compute_stats(period, duration, epoch)
    brightening = float(box_search(time, -flux, noise, [period], [duration])[0][0])
    odd, even = stats["depth_odd"], stats["depth_even"]
    odd_even_error = np.hypot(odd[1], even[1])
    odd_even_sigma = abs(odd[0] - even[0]) / odd_even_error
    covered = np.asarray(stats["per_transit_count"]) > 0
    cycle_support = np.asarray(stats["per_transit_log_likelihood"])[covered]
    return {
        **row,
        "search_band": band,
        "inverse_same_period_snr": brightening,
        "dip_inverse_snr_ratio": row["search_snr"] / max(brightening, 1e-6),
        "harmonic_delta_log_likelihood": float(stats["harmonic_delta_log_likelihood"]),
        "odd_even_sigma": float(odd_even_sigma),
        "positive_cycle_fraction": float(np.mean(cycle_support > 0))
        if len(cycle_support)
        else 0.0,
    }


def hypothesis_reasons(row, config):
    reasons = []
    ratio = row["dip_inverse_snr_ratio"]
    if ratio < config["dip_inverse_snr_ratio_min"]:
        reasons.append("INVERSE")
    if (
        row["search_rank"] >= config["high_rank_start"]
        and ratio < config["high_rank_inverse_snr_ratio_min"]
    ):
        reasons.append("HIGH_RANK_INVERSE")
    if (
        row["harmonic_delta_log_likelihood"]
        > config["harmonic_delta_log_likelihood_max"]
    ):
        reasons.append("HARMONIC")
    if row["odd_even_sigma"] > config["odd_even_sigma_max"]:
        reasons.append("ODD_EVEN")
    if row["positive_cycle_fraction"] < config.get("positive_cycle_fraction_min", 0):
        reasons.append("CYCLE_SUPPORT")
    if not np.isfinite(
        [ratio, row["harmonic_delta_log_likelihood"], row["odd_even_sigma"]]
    ).all():
        reasons.append("NONFINITE")
    return reasons


def search_order(power, depth_ppm, config, diagnostics=None):
    threshold = config.get("coarse_search_snr_min", config["search_snr_min"])
    depth_allowed = (depth_ppm >= config["depth_ppm_min"]) & (
        depth_ppm <= config["depth_ppm_max"]
    )
    allowed = depth_allowed & (power >= threshold)
    eligible_count = int(allowed.sum())
    if config.get("refine_local_maxima_only", False):
        allowed &= (power >= np.r_[-np.inf, power[:-1]]) & (
            power > np.r_[power[1:], -np.inf]
        )
    eligible = np.flatnonzero(allowed)
    if diagnostics is not None:
        diagnostics["coarse_exclusions"] = {
            "NONFINITE": int((~np.isfinite(power) | ~np.isfinite(depth_ppm)).sum()),
            "SEARCH_SNR": int((power < threshold).sum()),
            "DEPTH_DOMAIN": int((~depth_allowed).sum()),
            "NON_LOCAL_MAXIMUM": eligible_count - len(eligible),
            "eligible_trials": len(eligible),
        }
    return eligible[np.argsort(power[eligible])[::-1]]


def scan_range(time, flux, noise, quarter, config, band, diagnostics=None):
    band_diagnostics = None
    if diagnostics is not None:
        band_diagnostics = {
            "band": band,
            "status": "SEARCHING",
            "config": dict(config),
            "store_periodograms": diagnostics.get("store_periodograms", False),
            "tested_peaks": [],
            "whitening": {
                "detrending": config.get("detrending"),
                "retained_cadences": len(time),
                "quarter_noise_ppm": {
                    str(int(q)): float(noise[quarter == q][0] * 1e6)
                    for q in np.unique(quarter)
                },
            },
        }
        diagnostics.setdefault("bands", []).append(band_diagnostics)
    periods = period_grid(time, config)
    power, depth, coarse_duration, coarse_epoch = search_grid(
        time,
        flux,
        noise,
        periods,
        config,
        config.get("coarse_phase_oversample", 10),
        band_diagnostics,
    )
    periodogram_diagnostics(
        band_diagnostics, time, periods, power, depth, coarse_duration, coarse_epoch
    )
    depth_ppm = depth * 1e6
    order = search_order(power, depth_ppm, config, band_diagnostics)
    hypotheses = search_hypotheses(
        time,
        flux,
        noise,
        quarter,
        periods,
        power,
        depth_ppm,
        order,
        config,
        band_diagnostics,
    )
    bls = BoxLeastSquares(time, flux, dy=noise)
    proposals = []
    for row in hypotheses:
        measured = measure_hypothesis(bls, time, flux, noise, row, band)
        reasons = hypothesis_reasons(measured, config)
        if reasons:
            decision(band_diagnostics, measured, "REJECTED", reasons)
            continue
        proposals.append(measured)
    result = distinct_proposals(proposals, time, config, band_diagnostics)
    if band_diagnostics is not None:
        band_diagnostics.update(
            status="SEARCHED", hypotheses=len(hypotheses), proposals=len(result)
        )
    return result


def scan_long_periods(records, time, flux, noise, quarter, config, diagnostics):
    long_config = config | config["long_period_search"]
    independent = long_config.get("detrending") != config.get("detrending")
    try:
        if independent:
            time, flux, noise, quarter = whiten(
                records,
                long_config["detrend_window_cadences"],
                long_config["min_quarter_cadences"],
                long_config.get("detrending"),
            )
        return scan_range(time, flux, noise, quarter, long_config, "long", diagnostics)
    except ShallowIneligible as error:
        if not independent and error.reason != "INSUFFICIENT_BASELINE":
            raise
        if diagnostics is not None:
            bands = diagnostics.setdefault("bands", [])
            if not bands or bands[-1]["band"] != "long":
                bands.append({"band": "long", "config": long_config})
            bands[-1].update(status="INELIGIBLE", reason=error.reason)
        return []


def add_long_period_proposals(
    records, time, flux, noise, quarter, proposals, config, diagnostics=None
):
    long = scan_long_periods(records, time, flux, noise, quarter, config, diagnostics)
    next_rank = max((row["search_rank"] for row in proposals), default=0)
    added = 0
    merge_diagnostics = (
        None if diagnostics is None else diagnostics.setdefault("band_merge", {})
    )
    for index, row in enumerate(long):
        used = (
            added if config.get("proposal_budget_per_band", False) else len(proposals)
        )
        if used >= config["max_proposals_per_source"]:
            for remaining in long[index:]:
                decision(merge_diagnostics, remaining, "REJECTED", ("LONG_BAND_QUOTA",))
            break
        window = transit_window(
            records["time"],
            row["period_days"],
            row["t0_bkjd"],
            row["duration_hours"] / 24,
        )
        overlaps = [
            prior
            for prior in proposals
            if overlaps_proposal(window, records["time"], prior, config)
        ]
        preserve = config.get("preserve_cross_band_hypotheses", False) and all(
            prior["search_band"] != row["search_band"] for prior in overlaps
        )
        if overlaps and not preserve:
            decision(merge_diagnostics, row, "REJECTED", ("CROSS_BAND_OVERLAP",))
            continue
        proposals.append({**row, "search_rank": next_rank + row["search_rank"]})
        added += 1
        linked = (
            {
                "overlapping_core_proposals": [
                    {
                        key: prior[key]
                        for key in (
                            "search_band",
                            "search_rank",
                            "period_days",
                            "t0_bkjd",
                        )
                    }
                    for prior in overlaps
                ]
            }
            if overlaps
            else {}
        )
        decision(merge_diagnostics, row | linked, "PROPOSAL")
    return proposals


def scan(records, config, diagnostics=None):
    time, flux, noise, quarter = whiten(
        records,
        config["detrend_window_cadences"],
        config["min_quarter_cadences"],
        config.get("detrending"),
    )
    if diagnostics is not None:
        diagnostics["whitening"] = {
            "input_cadences": len(records),
            "retained_cadences": len(time),
            "retained_quarters": sorted(map(int, np.unique(quarter))),
            "quarter_noise_ppm": {
                str(int(q)): float(noise[quarter == q][0] * 1e6)
                for q in np.unique(quarter)
            },
        }
    if len(np.unique(quarter)) < config["min_observed_quarters"]:
        raise ShallowIneligible("INSUFFICIENT_USABLE_QUARTERS")
    proposals = scan_range(time, flux, noise, quarter, config, "baseline", diagnostics)
    if "short_period_search" in config:
        short_config = config | config["short_period_search"]
        short = scan_range(
            time, flux, noise, quarter, short_config, "short", diagnostics
        )
        next_rank = max((row["search_rank"] for row in proposals), default=0)
        for row in short:
            if row["period_days"] < config["period_min_days"]:
                row["search_rank"] += next_rank
                proposals.append(row)
    if "long_period_search" in config:
        return add_long_period_proposals(
            records, time, flux, noise, quarter, proposals, config, diagnostics
        )
    return proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-ids", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    paths = sorted(args.input_dir.glob("*.psf_flat.fsc"))
    if args.source_ids:
        selected = set(args.source_ids.read_text().splitlines())
        paths = [path for path in paths if path.name.split(".")[0] in selected]
        if len(paths) != len(selected):
            raise ValueError("Requested source identities missing from FSC directory")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Limit must be positive")
        paths = paths[: args.limit]
    if not paths or args.output_dir.exists():
        raise ValueError("Nonempty inputs and new output directory required")
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    counts = {
        "sources": 0,
        "searched_sources": 0,
        "ineligible_sources": 0,
        "proposals": 0,
        "sources_without_proposals": 0,
    }
    ineligible_by_reason = {}
    source_wall_times = []
    with (
        (args.output_dir / "proposals.csv").open("w", newline="") as stream,
        (args.output_dir / "inputs.csv").open("w", newline="") as input_stream,
        (args.output_dir / "eligibility.csv").open(
            "w", newline=""
        ) as eligibility_stream,
    ):
        fields = FIELDS
        if config.get("sort_proposals_by_refined_snr", False):
            fields += ("proposal_priority",)
        writer = csv.DictWriter(stream, fieldnames=fields)
        input_writer = csv.writer(input_stream)
        eligibility_writer = csv.DictWriter(
            eligibility_stream, fieldnames=ELIGIBILITY_FIELDS
        )
        writer.writeheader()
        input_writer.writerow(("source_id", "fsc_sha256"))
        eligibility_writer.writeheader()
        for path in paths:
            source_start = time.monotonic()
            source_id, records, fsc_sha = read_fsc(path)
            input_writer.writerow((source_id, fsc_sha))
            try:
                rows = scan(records, config)
            except ShallowIneligible as error:
                rows = []
                status = "INELIGIBLE"
                reason = error.reason
                counts["ineligible_sources"] += 1
                ineligible_by_reason[reason] = ineligible_by_reason.get(reason, 0) + 1
            else:
                status = "SEARCHED"
                reason = ""
                counts["searched_sources"] += 1
            writer.writerows({"source_id": source_id, **row} for row in rows)
            elapsed = time.monotonic() - source_start
            source_wall_times.append(elapsed)
            eligibility_writer.writerow(
                {
                    "source_id": source_id,
                    "status": status,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                }
            )
            counts["sources"] += 1
            counts["proposals"] += len(rows)
            counts["sources_without_proposals"] += status == "SEARCHED" and not rows
    manifest = {
        "status": "DEVELOPMENT_PROPOSALS_ONLY",
        "config_sha256": digest(CONFIG),
        "implementation_sha256": digest(Path(__file__)),
        "bls_sha256": digest(BLS_SOURCE),
        "bls_binary": bls_library().build_manifest,
        "input_dir": str(args.input_dir.resolve()),
        "sources_sha256": hashlib.sha256(
            "\n".join(path.name for path in paths).encode()
        ).hexdigest(),
        "proposals_sha256": digest(args.output_dir / "proposals.csv"),
        "inputs_sha256": digest(args.output_dir / "inputs.csv"),
        "eligibility_sha256": digest(args.output_dir / "eligibility.csv"),
        "elapsed_seconds": time.monotonic() - start,
        "per_source_wall_seconds_p50": float(np.quantile(source_wall_times, 0.5)),
        "per_source_wall_seconds_p95": float(np.quantile(source_wall_times, 0.95)),
        "ineligible_by_reason": ineligible_by_reason,
        **counts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
