"""Development-only duration grids with bounded refinement and immediate vetoes.

The coarse grid permits 4(1-correlation)*duration accumulated phase drift.
Each blind coarse cell is refined at the separately declared fine correlation.
The unchanged weighted box model and detector gates are applied at every stage;
this is neither a physical transit template nor TPS covariance whitening.
"""

import ctypes
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
SURVEY = ROOT / "pipeline/shallow"
BLS_SOURCE = SURVEY / "bls.c"
ENGINE_CONFIG = {
    "coarse_correlation": 0.8,
    "refinement_correlation": 0.95,
    "maximum_period_trials": 2000000,
    "maximum_refinement_trials": 64,
    "max_refinements_per_duration": 64,
    "duration_days": [0.02, 0.028, 0.04, 0.056, 0.08, 0.112, 0.16, 0.224, 0.32],
    "long_duration_days": [0.06, 0.084, 0.12, 0.168, 0.24, 0.336, 0.48],
    "skip_redundant_intervals": True,
}
REFERENCES = {
    "period_grid": (
        "https://github.com/nasa/kepler-pipeline/blob/"
        "f58b21df2c82969d8bd3e26a269bd7f5b9a770e1/source-code/matlab/"
        "tps/search/mfiles/compute_search_periods.m"
    ),
    "weighted_boxes": "https://docs.astropy.org/en/stable/timeseries/bls.html",
}


def source_path(name):
    return (SURVEY if name in ("run.py", "bls.c") else FOLDER) / name


@functools.cache
def sibling(name):
    path = source_path(f"{name}.py")
    spec = importlib.util.spec_from_file_location(__name__ + "_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = sibling("run")
ShallowIneligible = COMMON.ShallowIneligible


def bls_library():
    COMMON.ROOT, COMMON.BLS_SOURCE = ROOT, BLS_SOURCE
    return COMMON.bls_library()


@functools.cache
def batch_library():
    fine = bls_library().build_manifest
    source = FOLDER / "batch_bls.c"
    compilation = fine["compilation"] | {
        "source_sha256": COMMON.digest(source),
        "included_bls_sha256": COMMON.digest(BLS_SOURCE),
    }
    key = hashlib.sha256(json.dumps(compilation, sort_keys=True).encode()).hexdigest()
    library = ROOT / "dump/shallow-bls" / f"batch-bls-{key[:16]}.so"
    if not library.is_file():
        library.parent.mkdir(parents=True, exist_ok=True)
        partial = library.with_name(f"{library.name}.{os.getpid()}")
        subprocess.run(
            [*COMMON.BLS_COMMAND, "-o", str(partial), str(source), "-lm"], check=True
        )
        partial.replace(library)
    handle = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    handle.bls_batch.restype = ctypes.c_int
    handle.bls_batch.argtypes = [
        ctypes.c_int,
        array,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        array,
    ]
    handle.build_manifest = {
        "path": str(library),
        "sha256": COMMON.digest(library),
        "compilation": compilation,
    }
    return handle


def array_identity(value):
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def checked_group(configs):
    if len(configs) != 3:
        raise ValueError("Exactly three declared seed configurations required")
    shared = [{k: v for k, v in c.items() if k != "noise_model"} for c in configs]
    if any(c != shared[0] for c in shared[1:]):
        raise ValueError("Grouped science configuration differs beyond fixed noise")
    if any(k in shared[0] for k in ("physical_template", "coherence_ranking")):
        raise ValueError("Pristine duration seed configurations required")
    first, second, third = (c.get("noise_model") for c in configs)
    if first is not None or second is None or third is None:
        raise ValueError("Baseline, floor1 and floor0.5 metrics required")
    if (
        second.get("method") != "local_point"
        or second.get("minimum_scale_fraction") != 1.0
        or third.get("minimum_scale_fraction") != 0.5
        or {k: v for k, v in second.items() if k != "minimum_scale_fraction"}
        != {k: v for k, v in third.items() if k != "minimum_scale_fraction"}
    ):
        raise ValueError("Exact separately declared local-point metrics required")
    for config in configs:
        settings(config)
    return shared[0]


def coarse_group_metadata(configs):
    return {
        "kind": "BATCHED_COARSE_THREE_METRICS",
        "configs": configs,
        "fine_native": bls_library().build_manifest,
        "coarse_native": batch_library().build_manifest,
        "source_sha256": {
            name: COMMON.digest(source_path(name))
            for name in (
                "evolve_detector.py",
                "evolve_noise.py",
                "run.py",
                "bls.c",
                "batch_bls.c",
            )
        },
    }


def coarse_execution(records, configs, member, runtime):
    return coarse_group_metadata(configs) | {
        "member": member,
        "prepared": array_identity(records),
        "runtime": runtime,
        "bands": [],
    }


def checked_coarse_header(witness, expected, records):
    if (
        set(witness)
        != {
            "kind",
            "member",
            "configs",
            "prepared",
            "runtime",
            "fine_native",
            "coarse_native",
            "source_sha256",
            "bands",
        }
        or witness["kind"] != "BATCHED_COARSE_THREE_METRICS"
    ):
        raise ValueError("Explicit complete grouped coarse execution witness required")
    checked_group(witness["configs"])
    member = witness["member"]
    if type(member) is not int or not 0 <= member < 3:
        raise ValueError("Exact coarse metric member required")
    config_sha = hashlib.sha256(
        json.dumps(witness["configs"][member], sort_keys=True).encode()
    ).hexdigest()
    if config_sha != expected["config_sha256"]:
        raise ValueError("Coarse group metric/config association differs")
    if witness["prepared"] != array_identity(records):
        raise ValueError("Coarse group prepared input differs")
    checked_coarse_runtime(witness["runtime"], expected["runtime"])
    checked_coarse_sources(witness)
    checked_coarse_native(witness["fine_native"], bls_library().build_manifest)
    checked_coarse_native(witness["coarse_native"], batch_library().build_manifest)


def checked_coarse_native(previous, current):
    if set(previous) != {"path", "sha256", "compilation"}:
        raise ValueError("Exact native build witness required")
    if COMMON.digest(previous["path"]) != previous["sha256"] or {
        k: v for k, v in previous.items() if k != "path"
    } != {k: v for k, v in current.items() if k != "path"}:
        raise ValueError("Actual coarse/fine native execution differs")


def checked_coarse_runtime(runtime, expected):
    if set(runtime) != {
        "python",
        "architecture",
        "platform",
        "executable_sha256",
        "libraries",
    } or set(runtime["libraries"]) != set(expected["libraries"]) | {"batman-package"}:
        raise ValueError("Complete grouped runtime witness required")
    common = {k: runtime[k] for k in ("python", "architecture", "executable_sha256")}
    common["libraries"] = {k: runtime["libraries"][k] for k in expected["libraries"]}
    if common != expected:
        raise ValueError("Coarse group runtime differs")


def checked_coarse_sources(witness):
    names = {"evolve_detector.py", "evolve_noise.py", "run.py", "bls.c", "batch_bls.c"}
    if set(witness["source_sha256"]) != names:
        raise ValueError("Exact coarse dependency witness required")
    if {name: COMMON.digest(source_path(name)) for name in names} != witness[
        "source_sha256"
    ]:
        raise ValueError("Coarse execution source changed")


def checked_coarse_entry(entry, row, configs, member_index):
    if set(entry) != {"band", "duration_days", "members", "oversample"}:
        raise ValueError("Exact coarse band witness schema required")
    if entry["band"] not in ("baseline", "long") or len(entry["members"]) != 3:
        raise ValueError("Exact coarse band/metric membership required")
    if (row["search_band"], row["duration_trial_days"]) != (
        entry["band"],
        entry["duration_days"],
    ) or entry["oversample"] != row["config"].get("coarse_phase_oversample", 3):
        raise ValueError("Coarse band/duration/oversample association differs")
    chosen = configs[member_index]
    if entry["band"] == "long":
        chosen = chosen | chosen["long_period_search"]
    chosen = chosen | {
        "duration_days": [entry["duration_days"]],
        "period_grid_correlation": settings(configs[0])["coarse_correlation"],
        "maximum_period_trials": settings(configs[0])["maximum_period_trials"],
    }
    if row["config"] != chosen:
        raise ValueError("Actual coarse per-band member configuration differs")
    key = "long_duration_days" if entry["band"] == "long" else "duration_days"
    if entry["duration_days"] not in settings(configs[0])[key]:
        raise ValueError("Coarse duration is outside the declared group")
    keys = {
        "preprocessing_time",
        "preprocessing_flux",
        "preprocessing_quarter",
        "preprocessing_base_noise",
        "period_days",
        "noise",
        "ivar",
        "seven_outputs",
    }
    for member in entry["members"]:
        if set(member) != keys or any(
            set(value) != {"dtype", "shape", "sha256"} for value in member.values()
        ):
            raise ValueError("Exact typed coarse numerical witness required")
    common = keys - {"noise", "ivar", "seven_outputs"}
    if any(
        {k: member[k] for k in common} != {k: entry["members"][0][k] for k in common}
        for member in entry["members"][1:]
    ):
        raise ValueError("Common coarse preprocessing/grid witness differs")


def checked_coarse_bands(witness, diagnostics):
    bands = diagnostics["bands"]
    if "inactive_native_prefix_bands" in diagnostics:
        bands = diagnostics["inactive_native_prefix_bands"]
        if any(
            row.get("role") != "INACTIVE_PRE_RANKING_NATIVE_PREFIX" for row in bands
        ):
            raise ValueError("Explicit immutable inactive seed ownership required")
        if diagnostics["engine"]["name"] != "duration_partitioned_search":
            raise ValueError("Coherence reuse requires its pristine duration engine")
    rows = [b for b in bands if b["status"] == "SEARCHED"]
    if len(rows) != len(witness["bands"]):
        raise ValueError("Complete grouped coarse band membership required")
    keys = (
        "preprocessing_time",
        "preprocessing_flux",
        "preprocessing_quarter",
        "preprocessing_base_noise",
        "period_days",
    )
    for row, entry in zip(rows, witness["bands"]):
        checked_coarse_entry(entry, row, witness["configs"], witness["member"])
        arrays = row["periodogram_arrays"]
        actual = {key: array_identity(arrays[key]) for key in keys}
        actual["noise"] = array_identity(arrays["preprocessing_duration_noise"])
        actual["ivar"] = array_identity(
            1.0 / arrays["preprocessing_duration_noise"] ** 2
        )
        actual["seven_outputs"] = array_identity(arrays["coarse_native_seven_outputs"])
        if actual != entry["members"][witness["member"]]:
            raise ValueError("Coarse input/grid/noise/numerical member differs")


def checked_coarse_engine(witness, diagnostics):
    engine = diagnostics["engine"]
    if (
        set(engine)
        != {
            "name",
            "settings",
            "noise_model",
            "source_sha256",
            "common_source_sha256",
            "references",
            "limitations",
            "coarse_execution",
        }
        or engine["name"] != "duration_partitioned_search"
        or engine["source_sha256"] != witness["source_sha256"]["evolve_detector.py"]
        or engine["common_source_sha256"] != witness["source_sha256"]["run.py"]
        or engine["settings"] != settings(witness["configs"][witness["member"]])
        or engine["noise_model"]
        != witness["configs"][witness["member"]].get("noise_model")
        or engine["references"] != REFERENCES
        or engine["limitations"]
        != "Boxes, blind biweight and period-only dedup; not TPS"
        or engine["coarse_execution"] != witness
    ):
        raise ValueError(
            "Exact pristine coarse engine/source/member ownership required"
        )


def checked_coarse_identity(witness, expected, diagnostics, records):
    checked_coarse_header(witness, expected, records)
    checked_coarse_engine(witness, diagnostics)
    checked_coarse_bands(witness, diagnostics)


def checked_prefix_identity(identity, expected, diagnostics, records):
    grouped = diagnostics["engine"].get("coarse_execution")
    if grouped is None:
        if identity != expected:
            raise ValueError("Standalone blind prefix identity differs")
        return
    if set(identity) != set(expected) | {"coarse_execution"}:
        raise ValueError("Grouped prefix requires its actual coarse witness")
    if {k: identity[k] for k in expected} != expected:
        raise ValueError("Grouped blind prefix base identity differs")
    checked_coarse_identity(
        identity["coarse_execution"], expected, diagnostics, records
    )


def settings(config):
    selected = config["experimental_search"]
    for key in (
        "maximum_period_trials",
        "maximum_refinement_trials",
        "max_refinements_per_duration",
    ):
        if type(selected[key]) is not int or selected[key] < 2:
            raise ValueError(f"Experimental search requires integer {key} >= 2")
    for key in ("coarse_correlation", "refinement_correlation"):
        if not np.isfinite(selected[key]) or not 0 < selected[key] < 1:
            raise ValueError(f"Experimental search requires {key} in (0, 1)")
    for key in ("duration_days", "long_duration_days"):
        durations = np.asarray(selected[key], dtype=np.float64)
        if (
            durations.ndim != 1
            or not len(durations)
            or not np.isfinite(durations).all()
            or not np.all(durations > 0)
            or not np.all(np.diff(durations) > 0)
        ):
            raise ValueError(f"Experimental search requires ordered positive {key}")
    return selected


def duration_grid(time, duration, config, selected):
    chosen = config | {
        "duration_days": [duration],
        "period_grid_correlation": selected["coarse_correlation"],
        "maximum_period_trials": selected["maximum_period_trials"],
    }
    periods = COMMON.period_grid(time, chosen)
    if "duration_envelope" in chosen:
        periods = periods[
            duration <= COMMON.duration_limit(periods, chosen["duration_envelope"])
        ]
    return chosen, periods


def metric(time, flux, quarter, base_noise, duration, config, diagnostic, cache):
    model = config.get("noise_model")
    if model is None:
        return base_noise.copy()
    model_diagnostic = {}
    weights = sibling("evolve_noise").noise_weights(
        time, flux, quarter, base_noise, duration, model, model_diagnostic, cache=cache
    )
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != flux.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Experimental noise metric must preserve finite positive cadence weights"
        )
    if diagnostic is not None:
        diagnostic["noise_diagnostics"] = {
            key: value for key, value in model_diagnostic.items() if key != "arrays"
        }
        diagnostic["noise_diagnostics"]["source_sha256"] = COMMON.digest(
            FOLDER / "evolve_noise.py"
        )
        diagnostic["noise_arrays"] = model_diagnostic.get("arrays", {})
    return weights


def fine_grid(periods, index, time, duration, selected):
    lower, upper = periods[max(0, index - 1)], periods[min(len(periods) - 1, index + 1)]
    local_config = {
        "duration_days": [duration],
        "period_grid_correlation": selected["refinement_correlation"],
        "maximum_period_trials": selected["maximum_refinement_trials"],
    }
    grid = COMMON.duration_controlled_grid(
        time[-1] - time[0], lower, upper, local_config
    )
    grid = np.unique(np.r_[grid, periods[index]])
    if len(grid) > selected["maximum_refinement_trials"]:
        raise ValueError("Including the coarse seed exceeds refinement capacity")
    return grid


def fitted_peak(time, flux, noise, periods, index, config, selected, diagnostic):
    duration = config["duration_days"][0]
    local = fine_grid(periods, index, time, duration, selected)
    values = COMMON.search_grid(
        time, flux, noise, local, config, config.get("refine_phase_oversample", 10)
    )
    best = int(np.argmax(values[0]))
    if diagnostic is not None:
        diagnostic["local_refinements"].append(
            {
                "grid_index": int(index),
                "period_lower_days": float(local[0]),
                "period_upper_days": float(local[-1]),
                "trial_count": len(local),
                "maximum_phase_drift_days": float(
                    np.max(np.diff(local) / local[:-1]) * (time[-1] - time[0])
                ),
            }
        )
        if diagnostic["store_periodograms"]:
            for name, data in zip(
                ("search_snr", "depth_ppm", "duration_days", "epoch_bkjd"), values
            ):
                scale = 1e6 if name == "depth_ppm" else 1
                diagnostic["periodogram_arrays"][f"refinement_{index}_{name}"] = (
                    data * scale
                )
            diagnostic["periodogram_arrays"][f"refinement_{index}_period_days"] = local
    return (
        float(local[best]),
        float(values[3][best]),
        float(values[2][best]),
        float(values[1][best]) * 1e6,
        float(values[0][best]),
    )


def candidates(
    time, flux, noise, quarter, periods, values, config, selected, band, diagnostic
):
    order = COMMON.search_order(values[0], values[1] * 1e6, config, diagnostic)
    hypotheses, accepted, tested = [], [], 0
    bls = BoxLeastSquares(time, flux, dy=noise)
    for position, index in enumerate(order):
        if tested == selected["max_refinements_per_duration"]:
            if diagnostic is not None:
                diagnostic["unrefined_after_duration_budget"] = len(order) - position
            break
        peak = {
            "grid_index": int(index),
            "coarse_period_days": float(periods[index]),
            "coarse_search_snr": float(values[0][index]),
            "coarse_depth_ppm": float(values[1][index]) * 1e6,
        }
        if selected["skip_redundant_intervals"] and COMMON.redundant_refinement(
            periods, index, hypotheses, config
        ):
            if diagnostic is not None:
                diagnostic["excluded_refinement_intervals"].append(
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
        fitted = fitted_peak(
            time, flux, noise, periods, index, config, selected, diagnostic
        )
        tested += 1
        row = COMMON.refined_hypothesis(
            time, quarter, fitted, hypotheses, config, diagnostic, peak
        )
        if row is None:
            continue
        hypotheses.append(row)
        measured = COMMON.measure_hypothesis(bls, time, flux, noise, row, band)
        reasons = COMMON.hypothesis_reasons(measured, config)
        if reasons:
            COMMON.decision(diagnostic, measured, "REJECTED", reasons)
        else:
            accepted.append(measured)
            COMMON.decision(diagnostic, measured, "SHAPE_ACCEPTED")
    if diagnostic is not None:
        diagnostic.update(hypotheses=len(hypotheses), actual_refinements=tested)
    return accepted


def duration_diagnostic(diagnostics, band, index, duration, config):
    if diagnostics is None:
        return None
    result = {
        "band": f"{band}_duration_{index}",
        "search_band": band,
        "duration_trial_days": duration,
        "config": config,
        "status": "SEARCHING",
        "store_periodograms": diagnostics.get("store_periodograms", False),
        "tested_peaks": [],
        "local_refinements": [],
        "excluded_refinement_intervals": [],
    }
    diagnostics.setdefault("bands", []).append(result)
    return result


def duration_inputs(data, duration, index, config, selected, band, diagnostics, cache):
    time, flux, base_noise, quarter = data
    chosen, periods = duration_grid(time, duration, config, selected)
    diagnostic = duration_diagnostic(diagnostics, band, index, duration, chosen)
    if len(periods) < 2:
        if diagnostic is not None:
            diagnostic.update(status="INELIGIBLE", reason="DURATION_ABOVE_ENVELOPE")
        return None
    noise = metric(time, flux, quarter, base_noise, duration, config, diagnostic, cache)
    return chosen, periods, noise, diagnostic


def record_duration(data, context, values, selected):
    time, flux, base_noise, quarter = data
    chosen, periods, noise, diagnostic = context
    COMMON.periodogram_diagnostics(diagnostic, time, periods, *values)
    if diagnostic is not None:
        diagnostic["grid"].update(coarse_correlation=selected["coarse_correlation"])
        if diagnostic["store_periodograms"]:
            if "coarse_native_seven_outputs" in diagnostic:
                diagnostic["periodogram_arrays"]["coarse_native_seven_outputs"] = (
                    diagnostic.pop("coarse_native_seven_outputs")
                )
            diagnostic["periodogram_arrays"].update(
                preprocessing_time=time,
                preprocessing_flux=flux,
                preprocessing_quarter=quarter,
                preprocessing_base_noise=base_noise,
                preprocessing_duration_noise=noise,
            )
            diagnostic["periodogram_arrays"].update(
                {
                    "noise_" + key: value
                    for key, value in diagnostic.pop("noise_arrays", {}).items()
                }
            )
        else:
            diagnostic.pop("noise_arrays", None)


def finish_duration(data, context, values, selected, band):
    time, flux, _, quarter = data
    chosen, periods, noise, diagnostic = context
    result = candidates(
        time, flux, noise, quarter, periods, values, chosen, selected, band, diagnostic
    )
    if diagnostic is not None:
        diagnostic.update(status="SEARCHED", shape_accepted=len(result))
    return result


def one_duration(data, duration, index, config, selected, band, diagnostics, cache):
    context = duration_inputs(
        data, duration, index, config, selected, band, diagnostics, cache
    )
    if context is None:
        return []
    chosen, periods, noise, _ = context
    values = COMMON.search_grid(
        data[0],
        data[1],
        noise,
        periods,
        chosen,
        chosen.get("coarse_phase_oversample", 3),
    )
    record_duration(data, context, values, selected)
    return finish_duration(data, context, values, selected, band)


def band_scan(records, config, selected, band, diagnostics):
    data = COMMON.whiten(
        records,
        config["detrend_window_cadences"],
        config["min_quarter_cadences"],
        config.get("detrending"),
    )
    if len(np.unique(data[3])) < config["min_observed_quarters"]:
        raise ShallowIneligible("INSUFFICIENT_USABLE_QUARTERS")
    key = "long_duration_days" if band == "long" else "duration_days"
    accepted, cache = [], {}
    for index, duration in enumerate(selected[key]):
        accepted.extend(
            one_duration(
                data, duration, index, config, selected, band, diagnostics, cache
            )
        )
    for rank, row in enumerate(accepted, 1):
        row["search_rank"] = rank
    final = (
        None
        if diagnostics is None
        else diagnostics.setdefault("band_selection", {}).setdefault(band, {})
    )
    return COMMON.distinct_proposals(accepted, data[0], config, final)


def merge(core, long, records, config, diagnostics):
    result = list(core)
    next_rank = max((row["search_rank"] for row in core), default=0)
    diagnostic = (
        None if diagnostics is None else diagnostics.setdefault("band_merge", {})
    )
    for row in long:
        window = COMMON.transit_window(
            records["time"],
            row["period_days"],
            row["t0_bkjd"],
            row["duration_hours"] / 24,
        )
        overlaps = [
            prior
            for prior in core
            if COMMON.overlaps_proposal(window, records["time"], prior, config)
        ]
        if overlaps and not config.get("preserve_cross_band_hypotheses", False):
            COMMON.decision(diagnostic, row, "REJECTED", ("CROSS_BAND_OVERLAP",))
            continue
        result.append(row | {"search_rank": next_rank + row["search_rank"]})
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
        COMMON.decision(diagnostic, row | linked, "PROPOSAL")
    return result


def batch_values(data, contexts):
    time, flux, _, _ = data
    periods = contexts[0][1]
    duration = np.asarray(contexts[0][0]["duration_days"], dtype=np.float64)
    oversample = contexts[0][0].get("coarse_phase_oversample", 3)
    for chosen, grid, _, _ in contexts[1:]:
        if (
            array_identity(grid) != array_identity(periods)
            or chosen["duration_days"] != contexts[0][0]["duration_days"]
            or chosen.get("coarse_phase_oversample", 3) != oversample
        ):
            raise ValueError("Common coarse grid/duration/oversample differs")
    noise = np.ascontiguousarray([c[2] for c in contexts], dtype=np.float64)
    ivar = np.ascontiguousarray(1.0 / noise**2)
    out = np.empty((3, 7, len(periods)), dtype=np.float64)
    reference = np.min(time)
    status = batch_library().bls_batch(
        len(time),
        np.ascontiguousarray(time - reference),
        np.ascontiguousarray(flux - np.median(flux)),
        3,
        ivar,
        len(periods),
        np.ascontiguousarray(periods),
        len(duration),
        duration,
        oversample,
        out,
    )
    if status:
        raise ValueError(f"Batched coarse BLS rejected its inputs with status {status}")
    out.flags.writeable = False
    return out, [(v[0], v[1], v[3], v[4] + reference) for v in out]


def group_duration(data, duration, index, configs, selected, band, diagnostics, caches):
    contexts = [
        duration_inputs(data, duration, index, c, selected, band, d, cache)
        for c, d, cache in zip(configs, diagnostics, caches)
    ]
    if all(context is None for context in contexts):
        return [[], [], []]
    if any(context is None for context in contexts):
        raise ValueError("Grouped coarse eligibility differs")
    native, values = batch_values(data, contexts)
    keys = (
        "preprocessing_time",
        "preprocessing_flux",
        "preprocessing_base_noise",
        "preprocessing_quarter",
    )
    inputs = {key: array_identity(value) for key, value in zip(keys, data)}
    members = []
    for context, raw in zip(contexts, native):
        members.append(
            inputs
            | {
                "period_days": array_identity(context[1]),
                "noise": array_identity(context[2]),
                "ivar": array_identity(1.0 / context[2] ** 2),
                "seven_outputs": array_identity(raw),
            }
        )
    entry = {
        "band": band,
        "duration_days": duration,
        "members": members,
        "oversample": contexts[0][0].get("coarse_phase_oversample", 3),
    }
    for diagnostic, context, raw in zip(diagnostics, contexts, native):
        diagnostic["engine"]["coarse_execution"]["bands"].append(entry)
        context[3]["coarse_native_seven_outputs"] = raw
    for context, result in zip(contexts, values):
        record_duration(data, context, result, selected)
    return [
        finish_duration(data, context, result, selected, band)
        for context, result in zip(contexts, values)
    ]


def group_band(records, configs, selected, band, diagnostics):
    data = [
        COMMON.whiten(
            records,
            c["detrend_window_cadences"],
            c["min_quarter_cadences"],
            c.get("detrending"),
        )
        for c in configs
    ]
    original = [array_identity(value) for value in data[0]]
    if any([array_identity(value) for value in item] != original for item in data[1:]):
        raise ValueError("Grouped per-band preprocessing differs")
    if len(np.unique(data[0][3])) < configs[0]["min_observed_quarters"]:
        raise ShallowIneligible("INSUFFICIENT_USABLE_QUARTERS")
    accepted, caches = [[], [], []], [{}, {}, {}]
    key = "long_duration_days" if band == "long" else "duration_days"
    for index, duration in enumerate(selected[key]):
        result = group_duration(
            data[0], duration, index, configs, selected, band, diagnostics, caches
        )
        for target, rows in zip(accepted, result):
            target.extend(rows)
    for rows in accepted:
        for rank, row in enumerate(rows, 1):
            row["search_rank"] = rank
    return [
        COMMON.distinct_proposals(
            rows,
            data[0][0],
            c,
            d.setdefault("band_selection", {}).setdefault(band, {}),
        )
        for rows, c, d in zip(accepted, configs, diagnostics)
    ]


def scan_group(records, configs, diagnostics, runtime):
    checked_group(configs)
    if len(diagnostics) != 3 or any(
        not d.get("store_periodograms") for d in diagnostics
    ):
        raise ValueError("Three independent full-evidence diagnostic owners required")
    selected = settings(configs[0])
    if "short_period_search" in configs[0]:
        raise ValueError("Experimental group requires explicit 0.5–100 day bands")
    bls_library()
    for member, (config, evidence) in enumerate(zip(configs, diagnostics)):
        evidence["engine"] = {
            "name": "duration_partitioned_search",
            "settings": selected,
            "noise_model": config.get("noise_model"),
            "source_sha256": COMMON.digest(__file__),
            "common_source_sha256": COMMON.digest(source_path("run.py")),
            "references": REFERENCES,
            "limitations": "Boxes, blind biweight and period-only dedup; not TPS",
            "coarse_execution": coarse_execution(records, configs, member, runtime),
        }
    core = group_band(records, configs, selected, "baseline", diagnostics)
    if "long_period_search" not in configs[0]:
        return core
    chosen = [c | c["long_period_search"] for c in configs]
    try:
        long = group_band(records, chosen, selected, "long", diagnostics)
    except ShallowIneligible as error:
        for evidence in diagnostics:
            evidence.setdefault("bands", []).append(
                {"band": "long", "status": "INELIGIBLE", "reason": error.reason}
            )
        return core
    return [
        merge(a, b, records, c, d)
        for a, b, c, d in zip(core, long, configs, diagnostics)
    ]


def scan(records, config, diagnostics=None):
    selected = settings(config)
    if "short_period_search" in config:
        raise ValueError(
            "The experimental engine requires explicit 0.5–100 day study bands"
        )
    bls_library()
    if diagnostics is not None:
        diagnostics["engine"] = {
            "name": "duration_partitioned_search",
            "settings": selected,
            "noise_model": config.get("noise_model"),
            "source_sha256": COMMON.digest(__file__),
            "common_source_sha256": COMMON.digest(source_path("run.py")),
            "references": REFERENCES,
            "limitations": "Boxes, blind biweight and period-only dedup; not TPS",
        }
    core = band_scan(records, config, selected, "baseline", diagnostics)
    if "long_period_search" not in config:
        return core
    long_config = config | config["long_period_search"]
    try:
        long = band_scan(records, long_config, selected, "long", diagnostics)
    except ShallowIneligible as error:
        if diagnostics is not None:
            diagnostics.setdefault("bands", []).append(
                {"band": "long", "status": "INELIGIBLE", "reason": error.reason}
            )
        return core
    return merge(core, long, records, config, diagnostics)
