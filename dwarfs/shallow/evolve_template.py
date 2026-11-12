"""Blind physical-template refinement with a joint, fixed spline nuisance model.

Duration-search boxes supply seeds only. All final likelihoods use the same
a fixed diagonal precision and profile the identical spline design from
data and templates. This is conditional GLS, not TPS covariance whitening.
"""

import copy
import functools
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import platform
import sys
from pathlib import Path

import batman
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import block_diag
from scipy.sparse.linalg import splu

FOLDER = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    __name__ + "_duration_search", FOLDER / "evolve_detector.py"
)
SEARCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEARCH
SPEC.loader.exec_module(SEARCH)
COMMON = SEARCH.COMMON
ShallowIneligible = COMMON.ShallowIneligible
bls_library = SEARCH.bls_library

TEMPLATE_CONFIG = {
    "knot_spacing_days": 3.0,
    "gap_days": 0.5,
    "exposure_days": 0.02043361111111111,
    "integration_samples": 11,
    "profile_samples": 1025,
    "maximum_seeds_per_band": 64,
    "maximum_profile_trials": 256,
    "duration_scales": [0.8, 1.0, 1.25],
    "epoch_offsets": [-0.5, -0.25, 0.0, 0.25, 0.5],
    "radius_ratio": 0.02,
    "density_min_g_cm3": 1.0,
    "density_max_g_cm3": 150.0,
    "profiles": [
        {"impact": 0.0, "limb_darkening": [0.5, 0.2]},
        {"impact": 0.75, "limb_darkening": [0.5, 0.2]},
        {"impact": 0.0, "limb_darkening": [0.7, 0.1]},
        {"impact": 0.75, "limb_darkening": [0.7, 0.1]},
    ],
}


def settings(config):
    chosen = config["physical_template"]
    if set(chosen) != set(TEMPLATE_CONFIG):
        raise ValueError("Explicit physical settings required")
    model = config.get("noise_model")
    if model is not None:
        if model.get("method") != "local_point":
            raise ValueError(
                "Physical metric supports duration-independent local_point only"
            )
        SEARCH.sibling("evolve_noise").settings_values(model)
    integers = (
        "integration_samples",
        "profile_samples",
        "maximum_seeds_per_band",
        "maximum_profile_trials",
    )
    for key in integers:
        if type(chosen[key]) is not int or chosen[key] < 2:
            raise ValueError(f"Physical refinement requires integer {key} >= 2")
    for key in (
        set(chosen) - set(integers) - {"profiles", "duration_scales", "epoch_offsets"}
    ):
        if not np.isfinite(chosen[key]) or chosen[key] <= 0:
            raise ValueError(f"Physical refinement requires finite positive {key}")
    if not 0 < chosen["radius_ratio"] < 0.2:
        raise ValueError("Physical small-planet radius ratio must lie in (0, .2)")
    if chosen["density_min_g_cm3"] >= chosen["density_max_g_cm3"]:
        raise ValueError("Ordered physical density bounds required")
    profile_settings(chosen)
    return chosen


def profile_settings(chosen):
    if chosen["integration_samples"] % 2 != 1 or chosen["profile_samples"] % 2 != 1:
        raise ValueError("Odd integration and profile sample counts required")
    for key in ("duration_scales", "epoch_offsets"):
        values = np.asarray(chosen[key], dtype=float)
        if (
            not len(values)
            or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(f"Ordered finite physical {key} required")
    if np.min(chosen["duration_scales"]) <= 0:
        raise ValueError("Positive physical duration scales required")
    if not chosen["profiles"]:
        raise ValueError("At least one declared physical profile required")
    for profile in chosen["profiles"]:
        impact = profile["impact"]
        coefficients = np.asarray(profile["limb_darkening"], dtype=float)
        if (
            coefficients.shape != (2,)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(impact)
            or not 0 <= impact < 1
        ):
            raise ValueError(
                "Finite quadratic limb darkening and impact in [0,1) required"
            )
        first, second = coefficients
        if first < 0 or first + second > 1 or first + 2 * second < 0:
            raise ValueError("Nonnegative, centrally increasing intensity required")


def spline_basis(time, quarter, selected):
    segment = (
        np.cumsum(
            np.r_[
                True, (np.diff(quarter) != 0) | (np.diff(time) > selected["gap_days"])
            ]
        )
        - 1
    )
    matrices, descriptions = [], []
    for label in np.unique(segment):
        index = np.flatnonzero(segment == label)
        values = time[index]
        degree = min(3, len(np.unique(values)) - 1)
        if degree == 0:
            matrix = np.ones((len(values), 1))
            knots = [float(values[0]), float(values[-1])]
        else:
            interior = np.arange(
                values[0] + selected["knot_spacing_days"],
                values[-1],
                selected["knot_spacing_days"],
            )
            knots = np.r_[
                np.repeat(values[0], degree + 1),
                interior,
                np.repeat(values[-1], degree + 1),
            ]
            matrix = BSpline.design_matrix(values, knots, degree)
        matrices.append(matrix)
        descriptions.append(
            {
                "segment": int(label),
                "rows": len(index),
                "degree": degree,
                "knots": np.asarray(knots).tolist(),
            }
        )
    return block_diag(matrices, format="csr"), segment, descriptions


class ProfileMetric:
    def __init__(self, time, flux, noise, quarter, selected):
        self.time, self.flux, self.noise, self.quarter = time, flux, noise, quarter
        self.weight = 1 / noise**2
        self.basis, self.segment, self.description = spline_basis(
            time, quarter, selected
        )
        normal = self.basis.T @ self.basis.multiply(self.weight[:, None])
        try:
            self.factor = splu(normal.tocsc())
        except RuntimeError as error:
            raise ShallowIneligible("SINGULAR_FIXED_NUISANCE_BASIS") from error
        self.beta = self.factor.solve(np.asarray(self.basis.T @ (self.weight * flux)))
        self.residual = flux - self.basis @ self.beta

    def projected_gram(self, index, design):
        weighted = self.weight[index, None] * design
        nuisance = np.asarray(self.basis[index].T @ weighted)
        gram = design.T @ weighted - nuisance.T @ self.factor.solve(nuisance)
        correlation = -design.T @ (self.weight[index] * self.residual[index])
        return (gram + gram.T) / 2, correlation

    def fit(self, index, values):
        gram, correlation = self.projected_gram(index, values[:, None])
        energy, numerator = float(gram[0, 0]), float(correlation[0])
        original = float(np.dot(values**2, self.weight[index]))
        tolerance = 128 * np.finfo(float).eps * original
        if energy < -tolerance or not np.isfinite([energy, numerator]).all():
            raise ValueError("Nonfinite or negative profiled template information")
        if energy <= tolerance:
            return None
        return {
            "C": numerator,
            "E": energy,
            "depth": numerator / energy,
            "depth_error": 1 / np.sqrt(energy),
            "snr": numerator / np.sqrt(energy),
            "delta_log_likelihood": max(0, numerator) ** 2 / (2 * energy),
        }

    def linear(self, index, design):
        nuisance = np.asarray(self.basis[index].T @ (self.weight[index, None] * design))
        residual = -self.basis @ self.factor.solve(nuisance)
        residual[index] += design
        gram = residual.T @ (self.weight[:, None] * residual)
        correlation = -residual.T @ (self.weight * self.residual)
        eigen = np.linalg.eigvalsh(gram)
        if eigen[0] <= 128 * np.finfo(float).eps * eigen[-1]:
            return None
        covariance = np.linalg.inv(gram)
        depth = covariance @ correlation
        return depth, covariance, float(correlation @ depth / 2)


def profile_table(period, duration, profile, selected):
    if not 0 < duration < period / 2:
        return None
    impact, radius = profile["impact"], selected["radius_ratio"]
    axis = np.sqrt(
        impact**2
        + ((1 + radius) ** 2 - impact**2) / np.sin(np.pi * duration / period) ** 2
    )
    density = 3 * np.pi * axis**3 / (COMMON.G * (period * 86400) ** 2) / 1000
    if not selected["density_min_g_cm3"] <= density <= selected["density_max_g_cm3"]:
        return None
    parameters = batman.TransitParams()
    parameters.t0, parameters.per, parameters.rp = 0.0, period, radius
    parameters.a, parameters.inc = axis, np.degrees(np.arccos(impact / axis))
    parameters.ecc, parameters.w = 0.0, 90.0
    parameters.u, parameters.limb_dark = profile["limb_darkening"], "quadratic"
    extent = (duration + selected["exposure_days"]) / 2
    phase = np.linspace(-extent, extent, selected["profile_samples"])
    loss = integrated_loss(parameters, phase, duration, profile, selected)
    normalization = float(loss[len(loss) // 2])
    if normalization <= 0 or not np.isfinite(loss).all():
        raise ValueError("Physical template has invalid integrated central depth")
    return phase, loss / normalization, float(density)


@functools.lru_cache(maxsize=1)
def integration_rule(samples):
    nodes, weights = np.polynomial.legendre.leggauss(samples)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def integrated_loss(parameters, phase, duration, profile, selected):
    impact, radius, axis = profile["impact"], parameters.rp, parameters.a
    inner = max(0, (1 - radius) ** 2 - impact**2) / (axis**2 - impact**2)
    full = parameters.per / np.pi * np.arcsin(np.sqrt(inner))
    contacts = np.array([-duration / 2, -full / 2, full / 2, duration / 2])
    exposure = selected["exposure_days"]
    lower, upper = phase - exposure / 2, phase + exposure / 2
    bounds = np.c_[lower, np.clip(contacts, lower[:, None], upper[:, None]), upper]
    middle = (bounds[:, :-1] + bounds[:, 1:]) / 2
    width = np.diff(bounds) / 2
    nodes, weights = integration_rule(selected["integration_samples"])
    times = (middle[:, :, None] + width[:, :, None] * nodes).ravel()
    model = batman.TransitModel(parameters, times, nthreads=1)
    loss = (1 - model.light_curve(parameters)).reshape(len(phase), 5, len(nodes))
    return np.sum((loss @ weights) * width, axis=1) / exposure


def sampled_template(time, period, epoch, table):
    phase = (time - epoch + period / 2) % period - period / 2
    index = np.flatnonzero(np.abs(phase) <= table[0][-1])
    return index, np.interp(phase[index], table[0], table[1])


def seeds(diagnostics, band):
    result = []
    for trace in diagnostics["bands"]:
        if trace.get("search_band") != band or trace["status"] != "SEARCHED":
            continue
        arrays = trace["periodogram_arrays"]
        for refinement in trace["local_refinements"]:
            index = refinement["grid_index"]
            prefix = f"refinement_{index}_"
            best = int(np.argmax(arrays[prefix + "search_snr"]))
            local = arrays[prefix + "period_days"]
            result.append(
                {
                    "seed_band": trace["band"],
                    "seed_grid_index": index,
                    "period_days": float(local[best]),
                    "period_trials": local[max(0, best - 1) : best + 2].copy(),
                    "t0_bkjd": float(arrays[prefix + "epoch_bkjd"][best]),
                    "duration_days": float(arrays[prefix + "duration_days"][best]),
                    "seed_search_snr": float(arrays[prefix + "search_snr"][best]),
                }
            )
    return sorted(result, key=lambda row: row["seed_search_snr"], reverse=True)


def chosen_seeds(pool, metric, config, selected, trace):
    chosen, windows = [], []
    for rank, seed in enumerate(pool, 1):
        row = seed | {"seed_rank": rank}
        window = COMMON.transit_window(
            metric.time, seed["period_days"], seed["t0_bkjd"], seed["duration_days"]
        )
        duplicate = any(
            abs(seed["period_days"] / prior["period_days"] - 1)
            < config["period_dedup_fraction"]
            and min(window.sum(), previous.sum()) > 0
            and np.count_nonzero(window & previous) / max(window.sum(), previous.sum())
            > 0.9
            for prior, previous in zip(chosen, windows)
        )
        if duplicate:
            COMMON.decision(trace, row, "REJECTED", ("SEED_CLOCK_REDUNDANT",))
        elif len(chosen) == selected["maximum_seeds_per_band"]:
            COMMON.decision(trace, row, "REJECTED", ("PHYSICAL_SEED_QUOTA",))
        else:
            chosen.append(row)
            windows.append(window)
    return chosen


def seed_template(time, period, epoch, table, windows):
    key = (float(period), float(epoch), float(table[0][-1]))
    if key not in windows:
        phase = (time - epoch + period / 2) % period - period / 2
        index = np.flatnonzero(np.abs(phase) <= table[0][-1])
        windows[key] = index, phase[index]
    index, phase = windows[key]
    return index, np.interp(phase, table[0], table[1])


def physical_trials(seed, metric, selected, trace):
    durations = seed["duration_days"] * np.asarray(selected["duration_scales"])
    combinations = list(
        itertools.product(
            seed["period_trials"], durations, range(len(selected["profiles"]))
        )
    )
    if (
        len(combinations) * len(selected["epoch_offsets"])
        > selected["maximum_profile_trials"]
    ):
        raise ValueError("Physical refinement exceeds its explicit trial capacity")
    rows, best, saved_tables, windows = [], None, {}, {}
    trace.update(trials=rows, profile_arrays=saved_tables)
    for number, (period, duration, shape) in enumerate(combinations):
        table = profile_table(period, duration, selected["profiles"][shape], selected)
        if table is None:
            rows.append(
                {
                    "period_days": float(period),
                    "duration_days": float(duration),
                    "profile_index": shape,
                    "status": "PHYSICAL_DENSITY_DOMAIN",
                }
            )
            continue
        saved_tables[f"table_{number}_phase_days"] = table[0]
        saved_tables[f"table_{number}_integrated_loss"] = table[1]
        for offset in selected["epoch_offsets"]:
            epoch = seed["t0_bkjd"] + offset * duration
            index, values = seed_template(metric.time, period, epoch, table, windows)
            fit = metric.fit(index, values)
            row = {
                "period_days": float(period),
                "t0_bkjd": float(epoch),
                "duration_days": float(duration),
                "profile_index": shape,
                "density_g_cm3": table[2],
                "table_index": number,
            }
            rows.append(row | (fit or {"status": "NO_TEMPLATE_INFORMATION"}))
            if fit is not None and (best is None or fit["snr"] > best["snr"]):
                best = row | fit | {"index": index, "values": values}
    return best


def inverse_statistic(metric, candidate, selected):
    period, duration = candidate["period_days"], candidate["duration_days"]
    inverse = COMMON.box_search(
        metric.time, -metric.residual, metric.noise, [period], [duration], oversample=10
    )
    epoch = float(inverse[3][0])
    fits = []
    for shape, profile in enumerate(selected["profiles"]):
        table = profile_table(period, duration, profile, selected)
        if table is None:
            continue
        for offset in selected["epoch_offsets"]:
            clock = epoch + offset * duration
            index, values = sampled_template(metric.time, period, clock, table)
            fit = metric.fit(index, values)
            if fit is not None:
                fits.append({"profile_index": shape, "epoch_bkjd": float(clock), **fit})
    return max((max(0, -fit["snr"]) for fit in fits), default=0.0), fits


def periodogram_significance(diagnostics, band, period, duration):
    traces = [
        trace
        for trace in diagnostics["bands"]
        if trace.get("search_band") == band
        and "duration_trial_days" in trace
        and "search_snr" in trace.get("periodogram_arrays", {})
    ]
    trace = min(
        traces, key=lambda trace: abs(np.log(trace["duration_trial_days"] / duration))
    )
    grid = trace["periodogram_arrays"]["period_days"]
    snr = trace["periodogram_arrays"]["search_snr"]
    finite = snr[np.isfinite(snr)]
    median = np.median(finite)
    scale = 1.4826 * np.median(np.abs(finite - median))
    nearest = int(np.argmin(np.abs(np.log(grid / period))))
    peak = np.nanmax(snr[max(0, nearest - 3) : nearest + 4])
    return {
        "duration_trial_days": trace["duration_trial_days"],
        "coarse_snr": float(peak),
        "null_median": float(median),
        "null_scale": float(scale),
        "sde": float((peak - median) / scale),
    }


def event_consistency(time, residual, period, epoch, duration, flank_min):
    """Odd/even and event-to-event consistency measured from per-event depths.

    Each event's depth is its local-baseline median minus its in-transit mean, so
    structured noise and edge artifacts enter the error through the event scatter
    rather than through a white-noise formal error.
    """
    cycle = np.rint((time - epoch) / period).astype(np.int64)
    offset = np.abs(time - epoch - cycle * period)
    flank = duration / 2 + max(duration, flank_min)
    depths, parity = [], []
    for event in np.unique(cycle[offset < flank]):
        use = cycle == event
        inside = use & (offset < duration / 2)
        base = use & (offset >= duration / 2) & (offset < flank)
        if inside.any() and base.sum() >= 4:
            depths.append(np.median(residual[base]) - np.mean(residual[inside]))
            parity.append(event % 2)
    depths, parity = np.asarray(depths), np.asarray(parity)
    counts = np.array([np.sum(parity == 0), np.sum(parity == 1)])
    if counts.min() < 3:
        return None
    medians = np.array([np.median(depths[parity == p]) for p in (0, 1)])
    split = np.median(np.abs(depths - medians[parity]))
    pooled = np.median(np.abs(depths - np.median(depths)))
    scale = 1.2533 * 1.4826
    if split <= 0 or pooled <= 0:
        return None
    return {
        "events": len(depths),
        "odd_even_z": float(
            abs(medians[0] - medians[1]) / (scale * split * np.sqrt(np.sum(1 / counts)))
        ),
        "event_t": float(np.median(depths) * np.sqrt(len(depths)) / (scale * pooled)),
    }


def consistency(metric, candidate, selected):
    index, values = candidate["index"], candidate["values"]
    cycle = np.rint(
        (metric.time[index] - candidate["t0_bkjd"]) / candidate["period_days"]
    ).astype(np.int64)
    design = np.c_[values * (cycle % 2 == 0), values * (cycle % 2 == 1)]
    odd_even = metric.linear(index, design)
    phase = 2 * np.pi * metric.time / candidate["period_days"]
    harmonic = metric.linear(np.arange(len(phase)), np.c_[np.sin(phase), np.cos(phase)])
    inverse_snr, inverse_fits = inverse_statistic(metric, candidate, selected)
    if odd_even is None or harmonic is None:
        return {"status": "CONSISTENCY_UNIDENTIFIABLE", "inverse_fits": inverse_fits}
    depth, covariance, likelihood = odd_even
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    per_cycle = []
    for event in np.unique(cycle):
        use = cycle == event
        fit = metric.fit(index[use], values[use])
        per_cycle.append({"cycle": int(event), "fit": fit})
    return {
        "status": "MEASURED",
        "odd_even_sigma": float(abs(depth[0] - depth[1]) / np.sqrt(variance)),
        "odd_even_depths_ppm": depth * 1e6,
        "odd_even_covariance": covariance,
        "odd_even_delta_log_likelihood": likelihood,
        "inverse_search_snr": inverse_snr,
        "dip_inverse_snr_ratio": candidate["snr"] / max(inverse_snr, 1e-6),
        "harmonic_delta_log_likelihood": harmonic[2]
        - candidate["delta_log_likelihood"],
        "harmonic_coefficients": harmonic[0],
        "harmonic_covariance": harmonic[1],
        "inverse_fits": inverse_fits,
        "per_cycle": per_cycle,
        "positive_cycle_fraction": float(
            np.mean(
                [row["fit"] is not None and row["fit"]["C"] > 0 for row in per_cycle]
            )
        ),
    }


def measured_row(candidate, stats, metric, band, rank, selected):
    observed = COMMON.coverage(
        metric.time,
        metric.quarter,
        candidate["period_days"],
        candidate["t0_bkjd"],
        candidate["duration_days"] + selected["exposure_days"],
    )
    return {
        "period_days": candidate["period_days"],
        "t0_bkjd": candidate["t0_bkjd"],
        "duration_hours": candidate["duration_days"] * 24,
        "depth_ppm": candidate["depth"] * 1e6,
        "search_snr": candidate["snr"],
        "search_band": band,
        "search_rank": rank,
        "observed_events": observed[0],
        "observed_quarters": observed[1],
        "in_transit_cadences": observed[2],
        "inverse_search_snr": stats.get("inverse_search_snr", 0.0),
        "inverse_same_period_snr": stats.get("inverse_search_snr", 0.0),
        "dip_inverse_snr_ratio": stats.get("dip_inverse_snr_ratio", 0.0),
        "odd_even_sigma": stats.get("odd_even_sigma", np.inf),
        "harmonic_delta_log_likelihood": stats.get(
            "harmonic_delta_log_likelihood", np.inf
        ),
        "positive_cycle_fraction": stats.get("positive_cycle_fraction", 0.0),
        "profile_index": candidate["profile_index"],
        "physical_density_g_cm3": candidate["density_g_cm3"],
        "physical_delta_log_likelihood": candidate["delta_log_likelihood"],
    }


def rejection_reasons(row, stats, config):
    reasons = []
    if not config["depth_ppm_min"] <= row["depth_ppm"] <= config["depth_ppm_max"]:
        reasons.append("DEPTH_DOMAIN")
    if row["search_snr"] < config["search_snr_min"]:
        reasons.append("SEARCH_SNR")
    if (
        row["observed_events"] < config["min_observed_events"]
        or row["observed_quarters"] < config["min_observed_quarters"]
    ):
        reasons.append("COVERAGE")
    if stats["status"] != "MEASURED":
        return reasons + [stats["status"]]
    reasons.extend(COMMON.hypothesis_reasons(row, config))
    return reasons


def raw_metric(records, config, selected):
    time, filtered, noise, quarter = COMMON.whiten(
        records,
        config["detrend_window_cadences"],
        config["min_quarter_cadences"],
        config.get("detrending"),
    )
    retained = np.isin(records["quarter"], np.unique(quarter))
    flux = records["flux"][retained].astype(np.float64)
    if not np.array_equal(time, records["time"][retained]):
        raise ValueError("Joint trend metric changed retained cadence identities")
    model = config.get("noise_model")
    if model is None:
        metric = ProfileMetric(time, flux, noise, quarter, selected)
        metric.filtered = filtered
        return metric
    diagnostic = {}
    local = SEARCH.metric(
        time,
        filtered,
        quarter,
        noise,
        config["experimental_search"]["duration_days"][0],
        config,
        diagnostic,
        {},
    )
    metric = ProfileMetric(time, flux, local, quarter, selected)
    metric.filtered = filtered
    metric.base_noise = noise
    metric.noise_diagnostics = diagnostic["noise_diagnostics"]
    metric.noise_arrays = diagnostic["noise_arrays"]
    return metric


def verify_seed_weights(metric, diagnostics, band):
    for trace in diagnostics["bands"]:
        if trace.get("search_band") != band or "duration_trial_days" not in trace:
            continue
        if trace["status"] != "SEARCHED":
            continue
        arrays = trace["periodogram_arrays"]
        if not np.array_equal(arrays["preprocessing_time"], metric.time):
            raise ValueError(
                "Seed and physical metrics have different cadence identities"
            )
        if not np.array_equal(arrays["preprocessing_duration_noise"], metric.noise):
            raise ValueError(
                "Seed and physical metrics have different local-point errors"
            )


def physical_band(records, config, selected, band, diagnostics):
    metric = raw_metric(records, config, selected)
    trace = {
        "band": band + "_physical",
        "search_band": band,
        "status": "REFINING",
        "metric": "profiled_quarter_MAD_diagonal",
        "nuisance_segments": metric.description,
        "decisions": [],
        "physical_refinements": [],
        "preprocessing_arrays": {
            "time": metric.time,
            "prepared_flux": metric.flux,
            "noise": metric.noise,
            "quarter": metric.quarter,
            "nuisance_coefficients": metric.beta,
            "nuisance_residual": metric.residual,
            "segment": metric.segment,
            "basis_data": metric.basis.data,
            "basis_indices": metric.basis.indices,
            "basis_indptr": metric.basis.indptr,
        },
        "basis_shape": list(metric.basis.shape),
    }
    diagnostics["bands"].append(trace)
    if config.get("noise_model") is not None:
        verify_seed_weights(metric, diagnostics, band)
        trace["metric"] = "profiled_fixed_local_point_MAD_diagonal"
        trace["noise_diagnostics"] = metric.noise_diagnostics
        trace["preprocessing_arrays"].update(
            base_noise=metric.base_noise,
            **{"noise_" + key: value for key, value in metric.noise_arrays.items()},
        )
    pool = seeds(diagnostics, band)
    chosen = chosen_seeds(pool, metric, config, selected, trace)
    accepted = []
    for rank, seed in enumerate(chosen, 1):
        detail = {key: value for key, value in seed.items() if key != "period_trials"}
        trace["physical_refinements"].append(detail)
        candidate = physical_trials(seed, metric, selected, detail)
        if candidate is None:
            COMMON.decision(trace, detail, "REJECTED", ("NO_PHYSICAL_MODEL",))
            continue
        stats = consistency(metric, candidate, selected)
        detail["consistency"] = stats
        row = measured_row(candidate, stats, metric, band, rank, selected)
        reasons = rejection_reasons(row, stats, config)
        rescue = config.get("odd_even_event_rescue")
        if rescue is not None and reasons and set(reasons) <= set(rescue["reasons"]):
            event = event_consistency(
                metric.time,
                metric.filtered,
                row["period_days"],
                row["t0_bkjd"],
                row["duration_hours"] / 24,
                rescue["flank_days_min"],
            )
            detail["odd_even_event_rescue"] = event
            if (
                event is not None
                and event["odd_even_z"] <= rescue["odd_even_z_max"]
                and event["event_t"] >= rescue["event_t_min"]
            ):
                row["odd_even_event_rescue"] = event
                reasons = []
        veto = config.get("event_veto")
        if veto is not None and not reasons:
            event = event_consistency(
                metric.time,
                metric.filtered,
                row["period_days"],
                row["t0_bkjd"],
                row["duration_hours"] / 24,
                veto["flank_days_min"],
            )
            detail["event_veto"] = event
            if event is not None and (
                event["odd_even_z"] > veto["odd_even_z_max"]
                or event["event_t"] < veto["event_t_min"]
            ):
                reasons = ["EVENT_INCONSISTENT"]
        null = config.get("periodogram_null_gate")
        if null is not None and not reasons and band in null["bands"]:
            significance = periodogram_significance(
                diagnostics, band, row["period_days"], row["duration_hours"] / 24
            )
            detail["periodogram_null_gate"] = significance
            if significance["sde"] < null["sde_min"]:
                reasons = ["PERIODOGRAM_NULL"]
        if reasons:
            COMMON.decision(trace, row, "REJECTED", reasons)
        else:
            accepted.append(row)
            COMMON.decision(trace, row, "SHAPE_ACCEPTED")
    trace.update(status="SEARCHED", seed_pool=len(pool), physical_seeds=len(chosen))
    return COMMON.distinct_proposals(accepted, metric.time, config, trace)


def batman_identity():
    distribution = importlib.metadata.distribution("batman-package")
    files = {
        str(path): hashlib.sha256(
            Path(distribution.locate_file(path)).read_bytes()
        ).hexdigest()
        for path in distribution.files or []
        if Path(path).suffix in (".py", ".so", ".dylib", ".pyd")
    }
    return {"version": distribution.version, "files": files}


@functools.cache
def common_runtime():
    libraries = {}
    for name in ("numpy", "scipy", "astropy", "wotan", "numba", "llvmlite"):
        distribution = importlib.metadata.distribution(name)
        files = {
            str(path): COMMON.digest(distribution.locate_file(path))
            for path in distribution.files or []
            if Path(path).suffix in (".py", ".so", ".dylib", ".pyd")
        }
        libraries[name] = {
            "version": distribution.version,
            "code_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()
            ).hexdigest(),
        }
    return {
        "python": sys.version,
        "architecture": platform.machine(),
        "executable_sha256": COMMON.digest(sys.executable),
        "libraries": libraries,
    }


def seed_identity(records, config):
    seed_config = {
        key: value for key, value in config.items() if key != "physical_template"
    }
    native = copy.deepcopy(
        {
            key: value
            for key, value in bls_library().build_manifest.items()
            if key != "path"
        }
    )
    return {
        "prepared_bytes_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(seed_config, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": {
            name: COMMON.digest(SEARCH.source_path(name))
            for name in ("evolve_detector.py", "evolve_noise.py", "run.py", "bls.c")
        },
        "runtime": copy.deepcopy(common_runtime()),
        "native": native,
    }


def scan_with_seed_diagnostics(records, config, prefix, diagnostics=None):
    """Refine a caller-sealed blind result; artifact hashes must precede hydration."""
    selected = settings(config)
    if set(prefix) != {"identity", "diagnostics", "proposals"}:
        raise ValueError(
            "Bound blind seed prefix requires identity, diagnostics and proposals"
        )
    SEARCH.checked_prefix_identity(
        prefix["identity"],
        seed_identity(records, config),
        prefix["diagnostics"],
        records,
    )
    evidence = copy.deepcopy(prefix["diagnostics"])
    if evidence["engine"]["name"] != "duration_partitioned_search":
        raise ValueError("Fresh duration-engine diagnostics required")
    evidence["seed_proposals"] = copy.deepcopy(prefix["proposals"])
    evidence["seed_identity"] = copy.deepcopy(prefix["identity"])
    for band in evidence["bands"]:
        band["role"] = "BLIND_BOX_SEED_SEARCH_ONLY"
    evidence["seed_engine"] = evidence.pop("engine")
    evidence["engine"] = {
        "name": "physical_template_joint_spline_GLS",
        "noise_model": config.get("noise_model"),
        "settings": selected,
        "source_sha256": COMMON.digest(__file__),
        "batman": batman_identity(),
        "references": ["https://arxiv.org/abs/1901.02015"],
        "limitations": (
            "Coarse box4 threshold and duration search budgets can exclude seeds; "
            "fixed quarter or local-point diagonal MAD is not covariance whitening; "
            "coarse biweight flux and final raw joint-spline flux differ; inverse "
            "physical phases are refined around one blind inverted-box seed; "
            "template and nuisance priors and profile interpolation are fixed."
        ),
    }
    if diagnostics is not None:
        diagnostics.update(evidence)
        evidence = diagnostics
    core = physical_band(records, config, selected, "baseline", evidence)
    long = []
    if "long_period_search" in config:
        try:
            long = physical_band(
                records,
                config | config["long_period_search"],
                selected,
                "long",
                evidence,
            )
        except ShallowIneligible as error:
            evidence["bands"].append(
                {
                    "band": "long_physical",
                    "status": "INELIGIBLE",
                    "reason": error.reason,
                }
            )
    result = SEARCH.merge(core, long, records, config, evidence)
    if diagnostics is not None:
        diagnostics.update(evidence)
    return result


def scan(records, config, diagnostics=None):
    settings(config)
    evidence = {} if diagnostics is None else diagnostics
    evidence["store_periodograms"] = True
    seed_config = {
        key: value for key, value in config.items() if key != "physical_template"
    }
    proposals = SEARCH.scan(records, seed_config, evidence)
    prefix = {
        "identity": seed_identity(records, config),
        "diagnostics": evidence,
        "proposals": proposals,
    }
    return scan_with_seed_diagnostics(records, config, prefix, diagnostics)
