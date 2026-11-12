"""Experimental native-bin event ranking before unchanged physical refinement.

The fixed diagonal box metric projects one global intercept. Its event scatter
is descriptive under correlated residuals, not a calibrated TPS chi-square.
Final physical fits retain their identical fixed-W spline projection and gates.
"""

import copy
import ctypes
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

FOLDER = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    __name__ + "_physical", FOLDER / "evolve_template.py"
)
PHYSICAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PHYSICAL
SPEC.loader.exec_module(PHYSICAL)
SEARCH, COMMON = PHYSICAL.SEARCH, PHYSICAL.COMMON
ShallowIneligible = PHYSICAL.ShallowIneligible
COHERENCE_CONFIG = {"method": "native_bin_shared_intercept", "minimum_events": 3}
COLUMNS = (
    "native_mask_signed_snr",
    "priority",
    "event_scatter_chi2",
    "observed_events",
    "template_information",
    "depth",
    "support_reason_code",
)
REASONS = {
    0: "SUPPORTED_THREE_OR_MORE_EVENTS",
    1: "NO_OBSERVED_WINDOW_INFORMATION",
    2: "NO_OUTSIDE_INFORMATION",
    3: "ONE_EVENT_ZERO_PRIORITY",
    4: "TWO_EVENTS_ZERO_PRIORITY",
    5: "PREFIX_INFORMATION_NUMERICALLY_UNSUPPORTED",
}
REFERENCE = (
    "https://github.com/nasa/kepler-pipeline/blob/"
    "f58b21df2c82969d8bd3e26a269bd7f5b9a770e1/source-code/matlab/"
    "tps/search/mfiles/compute_chisquare_veto.m"
)


def physical_config(config):
    return {
        key: value
        for key, value in config.items()
        if key not in {"coherence_ranking", "global_fundamental_conditioning"}
    }


def settings(config):
    if config["coherence_ranking"] != COHERENCE_CONFIG:
        raise ValueError("Explicit native-bin shared-intercept ranking required")
    if type(config["coherence_ranking"]["minimum_events"]) is not int:
        raise ValueError("Three observed events must be an integer")
    return SEARCH.settings(physical_config(config))


@functools.cache
def bls_library():
    original = SEARCH.bls_library()
    source = FOLDER / "coherence.c"
    compilation = copy.deepcopy(original.build_manifest["compilation"])
    compilation["source_sha256"] = COMMON.digest(source)
    compilation["included_bls_sha256"] = COMMON.digest(SEARCH.BLS_SOURCE)
    key = hashlib.sha256(json.dumps(compilation, sort_keys=True).encode()).hexdigest()
    path = SEARCH.ROOT / "dump/shallow-bls" / f"coherence-{key[:16]}.so"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}")
        subprocess.run(
            [*COMMON.BLS_COMMAND, "-o", str(partial), str(source), "-lm"], check=True
        )
        partial.replace(path)
    handle = ctypes.CDLL(str(path))
    double = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    handle.bls.restype = handle.bls_metadata.restype = handle.rank_events.restype = (
        ctypes.c_int
    )
    handle.bls.argtypes = original.bls.argtypes
    handle.bls_metadata.argtypes = [*original.bls.argtypes, integer]
    handle.rank_events.argtypes = [
        ctypes.c_int,
        double,
        double,
        double,
        ctypes.c_int,
        double,
        integer,
        ctypes.c_double,
        ctypes.c_int,
        double,
        double,
    ]
    handle.build_manifest = {
        "path": str(path),
        "sha256": COMMON.digest(path),
        "compilation": compilation,
    }
    return handle


def native_metadata(time_axis, flux, noise, periods, duration, oversample):
    reference = float(time_axis[0])
    output = np.empty((7, len(periods)), dtype=np.float64)
    metadata = np.empty((len(periods), 2), dtype=np.int64)
    status = bls_library().bls_metadata(
        len(time_axis),
        np.ascontiguousarray(time_axis - reference),
        np.ascontiguousarray(flux - np.median(flux)),
        np.ascontiguousarray(1 / noise**2),
        len(periods),
        np.ascontiguousarray(periods),
        1,
        np.array([duration], dtype=np.float64),
        oversample,
        output,
        metadata,
    )
    if status:
        raise ValueError(f"Native metadata search rejected input: {status}")
    return output, metadata


def cached_arrays(seed):
    arrays = seed["periodogram_arrays"]
    names = (
        "preprocessing_time",
        "preprocessing_flux",
        "preprocessing_duration_noise",
        "preprocessing_quarter",
        "period_days",
        "search_snr",
        "depth_ppm",
        "duration_days",
        "epoch_bkjd",
    )
    values = [np.asarray(arrays[name]) for name in names]
    if any(value.ndim != 1 for value in values):
        raise ValueError("One-dimensional sealed prefix arrays required")
    if any(not np.isfinite(value).all() for value in values[:5]):
        raise ValueError("Finite sealed cadence, weight and period axes required")
    if (
        len(values[0]) < 2
        or len(values[4]) < 2
        or any(len(value) != len(values[0]) for value in values[:4])
        or any(len(value) != len(values[4]) for value in values[4:])
        or np.any(np.diff(values[0]) <= 0)
        or np.any(np.diff(values[4]) <= 0)
        or np.any(values[2] <= 0)
        or np.any(values[4] <= 0)
    ):
        raise ValueError("Sealed prefix axes or fixed errors differ")
    return values


def checked_metadata(seed, values, raw, trace):
    time_axis, flux, noise, _, periods, power, depth, duration, epoch = values
    requested = seed["duration_trial_days"]
    oversample = seed["config"].get("coarse_phase_oversample", 3)
    started = time.process_time()
    native, metadata = (
        native_metadata(time_axis, flux, noise, periods[raw], requested, oversample)
        if len(raw)
        else (np.empty((7, 0)), np.empty((0, 2), dtype=np.int64))
    )
    trace["periodogram_arrays"].update(
        native_seven_outputs=native,
        native_selected_bins=metadata,
        coherence_eligible_grid_index=raw,
    )
    restored = (native[0], native[1] * 1e6, native[3], native[4] + time_axis[0])
    for actual, original in zip(
        restored, (power[raw], depth[raw], duration[raw], epoch[raw])
    ):
        if actual.dtype != original.dtype or actual.tobytes() != original.tobytes():
            raise ValueError("Metadata entry changed a sealed coarse numerical array")
    seed["metadata_cpu_seconds"] = time.process_time() - started
    return native, metadata


def rank_arrays(values, native, metadata, raw, duration, oversample, trace):
    time_axis, flux, noise = values[:3]
    weight = 1 / noise**2
    response = flux - np.median(flux)
    prefix_weight = np.r_[0.0, np.cumsum(weight)]
    prefix_response = np.r_[0.0, np.cumsum(weight * response)]
    if not np.isfinite(prefix_weight).all() or not np.isfinite(prefix_response).all():
        raise ValueError("Fixed-W prefix sums overflowed")
    stats = np.full((len(raw), len(COLUMNS)), np.nan, dtype=np.float64)
    trace["periodogram_arrays"]["coherence_statistics"] = stats
    status = bls_library().rank_events(
        len(time_axis),
        np.ascontiguousarray(time_axis - time_axis[0]),
        prefix_weight,
        prefix_response,
        len(raw),
        np.ascontiguousarray(values[4][raw]),
        np.ascontiguousarray(metadata),
        duration / oversample,
        oversample,
        np.ascontiguousarray(native[0]),
        stats,
    )
    if status or not np.isfinite(stats[:, 1]).all():
        raise ValueError(f"Native event ranking rejected input: {status}")
    return stats


def ranking(seed, trace, config):
    values = cached_arrays(seed)
    raw = COMMON.search_order(values[5], values[6], config, trace)
    native, metadata = checked_metadata(seed, values, raw, trace)
    trace["periodogram_arrays"].update(
        native_seven_outputs=native,
        native_selected_bins=metadata,
        coherence_eligible_grid_index=raw,
    )
    start = time.process_time()
    stats = rank_arrays(
        values,
        native,
        metadata,
        raw,
        seed["duration_trial_days"],
        config.get("coarse_phase_oversample", 3),
        trace,
    )
    order = np.argsort(-stats[:, 1], kind="stable")
    rank = np.empty(len(raw), dtype=np.int64)
    rank[order] = np.arange(1, len(raw) + 1)
    trace.update(
        coherence_statistics_columns=list(COLUMNS),
        coherence_support_reasons=REASONS,
        metadata_cpu_seconds=seed["metadata_cpu_seconds"],
        all_eligible_ranking_cpu_seconds=time.process_time() - start,
        native_membership="Source integer phase-bin and copied-bin membership",
    )
    trace["periodogram_arrays"].update(
        native_seven_outputs=native,
        native_selected_bins=metadata,
        coherence_eligible_grid_index=raw,
        coherence_statistics=stats,
        coherence_native_search_snr=values[5][raw],
        coherence_priority_rank=rank,
        coherence_order_grid_index=raw[order],
        coherence_refinement_state=np.zeros(len(raw), dtype=np.uint8),
    )
    return values, raw, order, stats


def fit_candidates(data, periods, values, raw, order, stats, config, selected, trace):
    time_axis, flux, noise, quarter = data
    hypotheses, accepted, tested = [], [], 0
    model = BoxLeastSquares(time_axis, flux, dy=noise)
    states = trace["periodogram_arrays"]["coherence_refinement_state"]
    for position, raw_position in enumerate(order):
        if tested == selected["max_refinements_per_duration"]:
            trace["unrefined_after_duration_budget"] = len(order) - position
            break
        index = int(raw[raw_position])
        peak = {
            "grid_index": index,
            "coarse_period_days": float(periods[index]),
            "coarse_search_snr": float(values[0][index]),
            "coarse_depth_ppm": float(values[1][index]) * 1e6,
            "coarse_raw_rank": int(raw_position) + 1,
            "coherence_rank": position + 1,
            "coherence_priority": float(stats[raw_position, 1]),
        }
        if selected["skip_redundant_intervals"] and COMMON.redundant_refinement(
            periods, index, hypotheses, config
        ):
            states[raw_position] = 1
            trace["excluded_refinement_intervals"].append(
                peak
                | {
                    "period_lower_days": float(periods[max(0, index - 1)]),
                    "period_upper_days": float(
                        periods[min(len(periods) - 1, index + 1)]
                    ),
                    "reason": "REFINEMENT_PERIOD_REDUNDANT",
                }
            )
            continue
        states[raw_position] = 2
        fitted = SEARCH.fitted_peak(
            time_axis, flux, noise, periods, index, config, selected, trace
        )
        tested += 1
        row = COMMON.refined_hypothesis(
            time_axis, quarter, fitted, hypotheses, config, trace, peak
        )
        if row is None:
            continue
        hypotheses.append(row)
        measured = COMMON.measure_hypothesis(
            model, time_axis, flux, noise, row, trace["search_band"]
        )
        reasons = COMMON.hypothesis_reasons(measured, config)
        if reasons:
            COMMON.decision(trace, measured, "REJECTED", reasons)
        else:
            accepted.append(measured)
            COMMON.decision(trace, measured, "SHAPE_ACCEPTED")
    trace.update(hypotheses=len(hypotheses), actual_refinements=tested)
    return accepted


def reranked_duration(seed, config, selected, evidence):
    trace = {
        key: copy.deepcopy(seed[key])
        for key in ("search_band", "duration_trial_days", "config", "grid")
    }
    trace.update(
        band=seed["band"] + "_native_coherence",
        status="SEARCHING",
        role="NATIVE_COHERENCE_ACTIVE_REFINEMENT",
        store_periodograms=True,
        tested_peaks=[],
        local_refinements=[],
        excluded_refinement_intervals=[],
        coherence_refinement_state_codes={
            0: "UNREFINED_AFTER_BUDGET",
            1: "REFINEMENT_PERIOD_REDUNDANT",
            2: "ACTUAL_REFINEMENT_STARTED",
        },
        periodogram_arrays={
            key: value
            for key, value in seed["periodogram_arrays"].items()
            if not key.startswith("refinement_")
        },
    )
    evidence["bands"].append(trace)
    chosen = config | {
        "duration_days": [seed["duration_trial_days"]],
        "period_grid_correlation": selected["coarse_correlation"],
        "maximum_period_trials": selected["maximum_period_trials"],
    }
    if chosen != seed["config"]:
        raise ValueError("Cached duration configuration differs")
    data, raw, order, stats = ranking(seed, trace, chosen)
    arrays = (data[5], data[6] / 1e6, data[7], data[8])
    accepted = fit_candidates(
        data[:4], data[4], arrays, raw, order, stats, chosen, selected, trace
    )
    trace.update(status="SEARCHED", shape_accepted=len(accepted))


def seed_identity(records, config):
    return PHYSICAL.seed_identity(records, physical_config(config))


def prefix_membership(prior, selected, config):
    for band, key in (("baseline", "duration_days"), ("long", "long_duration_days")):
        rows = [row for row in prior if row.get("search_band") == band]
        failed = [
            row
            for row in prior
            if row.get("band") == "long" and row["status"] == "INELIGIBLE"
        ]
        if band == "long" and ("long_period_search" not in config or failed):
            if rows:
                raise ValueError("Whole-band ineligibility has duration evidence")
            continue
        if [row["duration_trial_days"] for row in rows] != selected[key]:
            raise ValueError("Cached complete duration-band membership differs")


def reranked_prefix(records, config, prefix, diagnostics):
    selected = settings(config)
    if set(prefix) != {"identity", "diagnostics", "proposals"}:
        raise ValueError("Bound identity, diagnostics and proposals required")
    SEARCH.checked_prefix_identity(
        prefix["identity"],
        seed_identity(records, config),
        prefix["diagnostics"],
        records,
    )
    evidence = copy.deepcopy(prefix["diagnostics"])
    diagnostics.update(evidence)
    evidence = diagnostics
    if evidence["engine"]["name"] != "duration_partitioned_search":
        raise ValueError("Complete unchanged duration prefix required")
    prior = evidence.pop("bands")
    evidence["inactive_native_prefix_bands"] = prior
    prefix_membership(prior, selected, config)
    evidence["inactive_native_prefix_proposals"] = copy.deepcopy(prefix["proposals"])
    evidence["inactive_native_prefix_selection"] = evidence.pop("band_selection", {})
    evidence["inactive_native_prefix_merge"] = evidence.pop("band_merge", {})
    evidence["bands"] = []
    base = {
        key: value
        for key, value in physical_config(config).items()
        if key != "physical_template"
    }
    for seed in prior:
        seed["role"] = "INACTIVE_PRE_RANKING_NATIVE_PREFIX"
        band = seed.get("search_band")
        if seed["status"] == "SEARCHED" and band in ("baseline", "long"):
            chosen = (
                base | base.get("long_period_search", {}) if band == "long" else base
            )
            reranked_duration(seed, chosen, selected, evidence)
        elif seed["status"] == "INELIGIBLE":
            evidence["bands"].append(copy.deepcopy(seed))
        else:
            raise ValueError("Unexpected prefix band membership or incomplete status")
    return {"identity": prefix["identity"], "diagnostics": evidence, "proposals": []}


def scan_with_seed_diagnostics(records, config, prefix, diagnostics=None):
    evidence = {} if diagnostics is None else diagnostics
    ranked = reranked_prefix(records, config, prefix, evidence)
    evidence.update(ranked["diagnostics"])
    result = PHYSICAL.scan_with_seed_diagnostics(
        records, physical_config(config), ranked, evidence
    )
    evidence["engine"]["coherence_ranking"] = config["coherence_ranking"]
    evidence["engine"]["coherence_source_sha256"] = COMMON.digest(__file__)
    evidence["engine"]["coherence_native"] = bls_library().build_manifest
    evidence["engine"]["coherence_reference"] = REFERENCE
    evidence["engine"]["coherence_limitations"] = (
        "Native one-phase boxes; shared-intercept diagonal ranking, not spline or "
        "TPS whitening/chi-square calibration. Correlation, drift, TTV and real "
        "depth changes can penalize planets. Old grid, eligibility, period-only "
        "dedup, 64 actual fits, physical metrics and final gates are unchanged."
    )
    if "global_fundamental_conditioning" in config:
        conditioner = SEARCH.sibling("evolve_global_fundamental")
        result = conditioner.apply(result, evidence, config, PHYSICAL)
        evidence["engine"]["global_fundamental_source_sha256"] = COMMON.digest(
            FOLDER / "evolve_global_fundamental.py"
        )
    return result


def scan(records, config, diagnostics=None):
    settings(config)
    base = physical_config(config)
    seed_config = {
        key: value for key, value in base.items() if key != "physical_template"
    }
    evidence = {} if diagnostics is None else diagnostics
    evidence["store_periodograms"] = True
    proposals = SEARCH.scan(records, seed_config, evidence)
    prefix = {
        "identity": seed_identity(records, config),
        "diagnostics": evidence,
        "proposals": proposals,
    }
    return scan_with_seed_diagnostics(records, config, prefix, diagnostics)
