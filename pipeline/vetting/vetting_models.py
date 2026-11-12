"""Exposure-integrated fits constrained to an existing detector event."""

import batman
import numpy as np
from lc2 import CADENCE, photometry, uses_flatten_mask
from lc2 import vetting_points as valid_points
from scipy.optimize import least_squares, minimize_scalar

NAMES = [
    "centre",
    "log_radius_ratio",
    "log_duration",
    "impact_fraction",
    "q1",
    "q2",
    "baseline",
    "slope",
]


def physical_input(raw, view, quality_bitmask, mask_flatten=True):
    data = photometry(raw, view)
    return np.column_stack(
        [
            data["flux"],
            data["error"],
            data["time"],
            valid_points(data, quality_bitmask, mask_flatten),
        ]
    )


class TransitCurve:
    def __init__(self, time, exposure=CADENCE, supersample=7):
        self.time = np.array(time, copy=True)
        self.exposure = exposure
        self.supersample = supersample
        self.model = None
        self.near = None
        self.near_model = None

    def __call__(self, theta):
        centre, log_r, log_d, impact, q1, q2, baseline, slope = theta
        radius, duration = np.exp(log_r), np.exp(log_d)
        b = impact * (1 + radius)
        period = 10000.0
        chord = (1 + radius) ** 2 - b * b
        a = np.sqrt(b * b + chord / np.sin(np.pi * duration / period) ** 2)
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a = centre, period, radius, a
        p.inc, p.ecc, p.w = np.degrees(np.arccos(b / a)), 0, 90
        p.limb_dark = "quadratic"
        p.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
        if self.model is None:
            self.model = batman.TransitModel(
                p,
                np.asarray(self.time, float),
                supersample_factor=self.supersample,
                exp_time=self.exposure,
            )
            curve = self.model.light_curve(p)
        else:
            # Integrate only exposures that can overlap the transit. Other exposures have unit flux.
            grid = np.asarray(self.time, float)
            near = abs(grid - centre) <= (duration + self.exposure) / 2 + CADENCE
            curve = np.ones(len(grid))
            if near.any():
                if self.near is None or not np.array_equal(near, self.near):
                    self.near = near
                    self.near_model = batman.TransitModel(
                        p,
                        grid[near],
                        supersample_factor=self.supersample,
                        exp_time=self.exposure,
                        fac=self.model.fac,
                    )
                curve[near] = self.near_model.light_curve(p)
        return baseline + slope * self.time + curve - 1


def transit(theta, time, exposure=CADENCE, supersample=7):
    return TransitCurve(time, exposure, supersample)(theta)


def median(values):
    """Median of a finite one-dimensional array, bitwise equal to np.median."""
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Median of an empty or non-finite array")
    half = len(values) // 2
    if len(values) % 2:
        return np.partition(values, half)[half]
    part = np.partition(values, [half - 1, half])
    return (part[half - 1] + part[half]) / 2


def event_data(x, centre, duration, config, sorted_time=False):
    half = max(
        config["minimum_context_days"], config["local_context_durations"] * duration
    )
    if sorted_time:
        time = x[:, 2]
        first = max(0, np.searchsorted(time, centre - half, side="left") - 1)
        last = min(len(x), np.searchsorted(time, centre + half, side="right") + 1)
        x = x[first:last]
    valid = (x[:, 3] > 0) & np.isfinite(x).all(axis=1) & (x[:, 1] > 0)
    take = valid & (abs(x[:, 2] - centre) <= half)
    if not take.any():
        return None
    baseline = median(x[take, 0])
    if baseline <= 0:
        return None
    return x[take, 2] - centre, x[take, 0] / baseline - 1, x[take, 1] / baseline


def bounds(time, y, duration):
    scatter = max(float(np.std(y)), 1e-6)
    slope = 20 * scatter / max(np.ptp(time), CADENCE)
    lower = [
        -duration / 2,
        np.log(1e-5),
        np.log(CADENCE / 4),
        0,
        0.001,
        0.001,
        -10 * scatter,
        -slope,
    ]
    upper = [
        duration / 2,
        np.log(1.5),
        np.log(max(np.ptp(time) / 2, CADENCE)),
        0.99,
        0.999,
        0.999,
        10 * scatter,
        slope,
    ]
    return np.array(lower), np.array(upper)


def event_errors(time, flux, error, duration):
    outside = abs(time) > duration
    if outside.sum() < 3:
        return error, {"status": "LOCAL_NOISE_UNCONSTRAINED", "errors_used": "FORMAL"}
    design = np.column_stack([np.ones(len(time)), time])
    coefficients = np.linalg.lstsq(
        design[outside] / error[outside, None],
        flux[outside] / error[outside],
        rcond=None,
    )[0]
    residual = flux - design @ coefficients
    scatter = 1.4826 * median(abs(residual[outside] - median(residual[outside])))
    jitter = np.sqrt(max(scatter**2 - median(error[outside] ** 2), 0))
    effective = np.hypot(error, jitter)
    bins = np.floor((time[outside] - time.min()) / max(duration, CADENCE)).astype(int)
    precision = np.bincount(bins, weights=1 / effective[outside] ** 2)
    total = np.bincount(bins, weights=residual[outside] / effective[outside] ** 2)
    use = np.bincount(bins) >= 2
    beta = 1.0
    if use.sum() >= 2:
        means = total[use] / precision[use]
        measured = 1.4826 * median(abs(means - median(means)))
        beta = float(max(1, measured / np.sqrt(median(1 / precision[use]))))
    return effective * beta, {
        "status": "LOCAL_NOISE_ESTIMATED",
        "white_jitter": float(jitter),
        "duration_beta": beta,
        "control_bins": int(use.sum()),
        "scope": "Local linear-baseline residuals, excess scatter and duration-bin noise. This noise estimator still needs injection tests; its output is not a significance cutoff.",
    }


def linear_fit(design, y, error, parameter_errors=True):
    return _linear_fit(
        design, y, error, design / error[:, None], y / error, parameter_errors
    )


def _linear_fit(design, y, error, weighted, values, parameter_errors):
    weights, _, rank, _ = np.linalg.lstsq(weighted, values, rcond=None)
    if rank < design.shape[1]:
        return {"status": "UNIDENTIFIABLE"}
    residual = (y - design @ weights) / error
    chi2 = residual @ residual
    result = {
        "status": "MEASURED",
        "chi2": float(chi2),
        "parameters": weights.tolist(),
        "bic": float(chi2 + len(weights) * np.log(len(y))),
    }
    if parameter_errors:
        result["parameter_errors"] = np.sqrt(
            np.diag(np.linalg.inv(weighted.T @ weighted))
        ).tolist()
    return result


class _VariableLinearFit:
    def __init__(self, time, y, error, columns):
        self.y, self.error = y, error
        self.design = np.empty((len(time), 2 + columns))
        self.design[:, 0], self.design[:, 1] = 1, time
        self.weighted = np.empty_like(self.design)
        self.weighted[:, :2] = self.design[:, :2] / error[:, None]
        self.values = y / error

    def __call__(self, *columns, parameter_errors=False):
        for i, column in enumerate(columns, 2):
            self.design[:, i] = column
            self.weighted[:, i] = self.design[:, i] / self.error
        return _linear_fit(
            self.design, self.y, self.error, self.weighted, self.values, parameter_errors
        )


def measure_interval(
    raw, centre, duration, view, config, quarter=None, prepared=None, sorted_time=False
):
    if prepared is None:
        x = physical_input(
            raw, view, config["quality_bitmask"], uses_flatten_mask(view, config)
        )
        if quarter is not None:
            x = x[photometry(raw, view)["quarter"] == quarter]
    else:
        x = prepared
    data = event_data(x, centre, duration, config, sorted_time=sorted_time)
    if data is None:
        return {
            "status": "NO_USABLE_COVERAGE",
            "centre_bkjd": float(centre),
            "duration_days": float(duration),
        }
    time, y, formal = data
    exposure = config["exposure_days"]
    pulse = (
        np.maximum(
            0,
            np.minimum(time + exposure / 2, duration / 2)
            - np.maximum(time - exposure / 2, -duration / 2),
        )
        / exposure
    )
    error, noise = event_errors(time, y, formal, duration)
    fit = linear_fit(np.column_stack([np.ones(len(time)), time, -pulse]), y, error)
    result = dict(
        fit,
        centre_bkjd=float(centre),
        duration_days=float(duration),
        in_event_points=int((pulse > 0).sum()),
        before_points=int((time < -duration / 2).sum()),
        after_points=int((time > duration / 2).sum()),
        noise=noise,
    )
    if fit["status"] == "MEASURED":
        result.update(
            depth_fraction=fit["parameters"][-1],
            depth_error=fit["parameter_errors"][-1],
        )
    return result


def alternatives(time, y, error, duration):
    base = np.column_stack([np.ones(len(time)), time])
    single = abs(time) == abs(time).min()
    box = abs(time) <= duration / 2
    models = {
        "linear": base,
        "quadratic": np.column_stack([base, time**2]),
        "step": np.column_stack([base, time >= 0]),
        "single_cadence": np.column_stack([base, single]),
        "signed_box": np.column_stack([base, box]),
    }
    fitted = {name: linear_fit(design, y, error) for name, design in models.items()}
    signed_box = fitted["signed_box"]
    if signed_box["status"] == "MEASURED":
        signed_box["depth_fraction"] = -signed_box["parameters"][-1]
    signed_box["equivalent_to"] = (
        ["single_cadence"]
        if np.array_equal(box, single)
        else (["transit_like_box"] if signed_box.get("depth_fraction", 0) > 0 else [])
    )
    return fitted


def dropout_alternative(time, y, error, duration):
    solver = _VariableLinearFit(time, y, error, 1)
    lo, hi = max(CADENCE / 4, np.finfo(float).eps), max(np.ptp(time), CADENCE)

    def solve(theta, parameter_errors=False):
        onset, log_tau = theta
        pulse = np.exp(-np.maximum(time - onset, 0) / np.exp(log_tau))
        pulse *= time >= onset
        return solver(pulse, parameter_errors=parameter_errors)

    onsets = np.unique(np.r_[-duration / 2, time[abs(time) <= duration / 2]])
    results = []
    for onset in onsets:
        result = minimize_scalar(
            lambda log_tau, onset=onset: solve([onset, log_tau]).get("chi2", np.inf),
            bounds=(np.log(lo), np.log(hi)),
            method="bounded",
        )
        results.append((solve([onset, result.x]), onset, result))
    fit, onset, result = min(results, key=lambda item: item[0].get("chi2", np.inf))
    fit = solve([onset, result.x], parameter_errors=True)
    if fit["status"] == "MEASURED":
        fit.update(
            bic=fit["chi2"] + 5 * np.log(len(time)),
            onset_days=float(onset),
            recovery_days=float(np.exp(result.x)),
            optimizer_success=bool(result.success),
        )
    return fit


def sinusoidal_alternative(time, y, error, duration):
    periods = np.geomspace(
        max(2 * duration, 4 * CADENCE), max(4 * np.ptp(time), 3 * duration), 64
    )

    solver = _VariableLinearFit(time, y, error, 2)
    scaled_time = 2 * np.pi * time

    def solve(period, parameter_errors=False):
        phase = scaled_time / period
        fit = solver(np.sin(phase), np.cos(phase), parameter_errors=parameter_errors)
        if fit["status"] != "MEASURED":
            return dict(fit, chi2=float("inf"))
        return dict(
            fit,
            bic=fit["chi2"] + 5 * np.log(len(time)),
            modulation_timescale_days=float(period),
        )

    rows = [solve(period) for period in periods]
    index = int(np.argmin([row["chi2"] for row in rows]))
    if not np.isfinite(rows[index]["chi2"]):
        return {"status": "UNIDENTIFIABLE"}
    fit = minimize_scalar(
        lambda period: solve(period)["chi2"],
        method="bounded",
        bounds=(periods[max(index - 1, 0)], periods[min(index + 1, len(periods) - 1)]),
    )
    refined = solve(fit.x)
    best_period = periods[index] if rows[index]["chi2"] <= refined["chi2"] else fit.x
    return {
        **solve(best_period, parameter_errors=True),
        "orbital_period_inferred": False,
        "scope": "Competing variability model on the same event context; not a transit search",
    }


def fit_uncertainty(fit, transform=None):
    if np.linalg.matrix_rank(fit.jac) < len(NAMES):
        return {"status": "PARAMETERS_DEGENERATE"}
    covariance = np.linalg.inv(fit.jac.T @ fit.jac)
    if transform is not None:
        covariance = transform @ covariance @ transform.T
    return {
        "status": "LOCAL_CURVATURE_ONLY",
        "covariance": covariance.tolist(),
        "parameter_names": NAMES,
    }


def optimize_event(time, y, error, duration, config):
    lo, hi = bounds(time, y, duration)
    lo[0], hi[0] = -1, 1
    inside = abs(time) <= duration / 2
    depth = max(float(-np.median(y[inside])), 1e-5)
    initial = [
        0,
        np.log(min(np.sqrt(depth), 0.9)),
        np.log(max(duration, CADENCE / 2)),
        0.3,
        0.25,
        0.4,
        np.median(y),
        0,
    ]
    initial = np.clip(initial, lo + 1e-6, hi - 1e-6)

    def physical(theta):
        converted = theta.copy()
        converted[0] *= (duration + np.exp(theta[2])) / 2
        return converted

    model = TransitCurve(time, config["exposure_days"], config["supersample_factor"])

    def residual(theta):
        return (model(physical(theta)) - y) / error

    starts = [initial]
    for width in (np.sqrt(np.exp(initial[2]) * np.exp(hi[2])), np.exp(hi[2]) * 0.9):
        for offset in (-0.5, 0, 0.5):
            start = initial.copy()
            start[0], start[2] = offset, np.log(width)
            starts.append(np.clip(start, lo + 1e-6, hi - 1e-6))
    fits = [
        least_squares(
            residual,
            start,
            bounds=(lo, hi),
            max_nfev=config["optimizer_max_evaluations"],
        )
        for start in starts
    ]
    matching = [
        result
        for result in fits
        if predicts_proposed_dip(physical(result.x), time[inside], config)
    ]
    fit = min(matching or fits, key=lambda result: result.fun @ result.fun)
    chi2 = float(fit.fun @ fit.fun)
    theta = physical(fit.x)
    transform = np.eye(len(NAMES))
    transform[0, 0] = (duration + np.exp(fit.x[2])) / 2
    transform[0, 2] = fit.x[0] * np.exp(fit.x[2]) / 2
    return {
        "status": ("FIT_COMPLETE" if fit.success else "OPTIMIZER_LIMIT")
        if matching
        else "PROPOSAL_NOT_EXPLAINED",
        "centre_offset_days": float(theta[0]),
        "duration_days": float(np.exp(fit.x[2])),
        "apparent_radius_ratio": float(np.exp(fit.x[1])),
        "theta": theta.tolist(),
        "parameter_names": NAMES,
        "bounds": [lo.tolist(), hi.tolist()],
        "optimizer_parameter_names": ["proposal_overlap_fraction", *NAMES[1:]],
        "active_bounds": fit.active_mask.tolist(),
        "optimizer_starts": len(starts),
        "solutions_without_proposed_dip": len(fits) - len(matching),
        "best_start_index": next(
            index for index, result in enumerate(fits) if result is fit
        ),
        "chi2": chi2,
        "bic": chi2 + len(NAMES) * np.log(len(time)),
        "uncertainty": fit_uncertainty(fit, transform),
        "proposal_constraint": "The fitted transit overlaps the original proposed interval. The centre may lie outside it, so an ingress-only proposal does not force an incorrect midpoint.",
        "model_scope": "Circular local-velocity approximation, with free physical limb darkening and impact parameter. Computational period 10000 d is not an inferred orbit. Apparent radius assumes unit dilution; no absolute companion radius or planetary mass is inferred. Multiple structures in the context can bias a single-transit fit; numerical starts and residuals are retained, not a marginalized posterior.",
    }


def predicts_proposed_dip(theta, time, config):
    shape = np.array(theta, copy=True)
    shape[-2:] = 0
    # A neighbouring dip cannot characterize an alert it predicts no flux loss in.
    return np.any(
        transit(shape, time, config["exposure_days"], config["supersample_factor"]) < 0
    )


def fit_event(x, centre, duration, config):
    data = event_data(x, centre, duration, config)
    if data is None:
        return {"status": "RELATIVE_FLUX_UNAVAILABLE", "period_days": None}
    time, y, formal_error = data
    error, noise = event_errors(time, y, formal_error, duration)
    alt = alternatives(time, y, error, duration)
    alt["shared_single_cadence"] = dict(alt["single_cadence"])
    base = {
        "time_origin_bkjd": centre,
        "n_points": len(time),
        "noise": noise,
        "alternatives": alt,
        "period_days": None,
        "original_interval_bkjd": [centre - duration / 2, centre + duration / 2],
        "context_bkjd": [float(time.min() + centre), float(time.max() + centre)],
        "edge_context": bool(time.min() > -duration or time.max() < duration),
        "automatic_disposition": False,
    }
    if len(time) <= len(NAMES) or np.ptp(time) == 0:
        return dict(base, status="PHYSICAL_FIT_UNDERCONSTRAINED")
    if not np.any(abs(time) <= duration / 2):
        return dict(base, status="NO_IN_EVENT_MEASUREMENTS")
    fit = optimize_event(time, y, error, duration, config)
    base["alternatives"]["sinusoid"] = sinusoidal_alternative(time, y, error, duration)
    base["alternatives"]["dropout"] = dropout_alternative(time, y, error, duration)
    fitted = transit(
        fit["theta"], time, config["exposure_days"], config["supersample_factor"]
    )
    residual = y - fitted
    base["residuals"] = {
        "rms": float(np.sqrt(np.mean(residual**2))),
        "lag_one_correlation": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
        "scope": "Local residual diagnostic; not a significance or acceptance test",
    }
    return {
        **base,
        **fit,
        "centre_bkjd": centre + fit["centre_offset_days"],
        "comparison_scope": "All fits use identical points, normalization and measured errors. BIC differences are diagnostic, not calibrated false-alarm probabilities or candidate cuts.",
    }


def project_baseline(time, flux, error):
    design = np.column_stack([np.ones(len(time)), time])
    fitted = np.linalg.lstsq(design / error[:, None], flux / error, rcond=None)[0]
    return flux - design @ fitted


class _BaselineProjection:
    def __init__(self, time, error):
        self.error = error
        self.design = np.column_stack([np.ones(len(time)), time])
        self.weighted = self.design / error[:, None]

    def __call__(self, flux):
        fitted = np.linalg.lstsq(self.weighted, flux / self.error, rcond=None)[0]
        return flux - self.design @ fitted


def joint_pulse(chunks, pulse_for):
    numerator, denominator, null_chi2 = 0.0, 0.0, 0.0
    for time, y, error, duration in chunks:
        project = _BaselineProjection(time, error)
        residual = project(y) / error
        pulse = project(pulse_for(time, duration)) / error
        numerator += pulse @ residual
        denominator += pulse @ pulse
        null_chi2 += residual @ residual
    if denominator <= 0:
        return {"status": "UNIDENTIFIABLE"}
    count = sum(len(c[0]) for c in chunks)
    chi2 = null_chi2 - numerator**2 / denominator
    return {
        "status": "MEASURED",
        "depth_fraction": float(-numerator / denominator),
        "chi2": float(chi2),
        "bic": float(chi2 + (2 * len(chunks) + 1) * np.log(count)),
    }


def joint_box(chunks):
    result = joint_pulse(
        chunks, lambda time, duration: (abs(time) <= duration / 2).astype(float)
    )
    if result["status"] == "MEASURED":
        result["equivalent_to"] = (
            ["shared_single_cadence"]
            if all(
                np.array_equal(abs(time) <= duration / 2, abs(time) == abs(time).min())
                for time, _, _, duration in chunks
            )
            else (["transit_like_box"] if result["depth_fraction"] > 0 else [])
        )
    return result


def joint_single_cadence(chunks):
    return joint_pulse(
        chunks, lambda time, duration: (abs(time) == abs(time).min()).astype(float)
    )


def unique_event_arrays(arrays):
    if not arrays:
        return []
    if not all(x.ndim == 2 and x.shape[1] == 5 for x in arrays):
        raise ValueError("Joint-event arrays require one absolute sample-ID column")
    owner = {}
    for chunk, x in enumerate(arrays):
        for row, sample in enumerate(x[:, 4].astype(np.int64)):
            candidate = (abs(x[row, 2]), chunk, row)
            if sample not in owner or candidate < owner[sample]:
                owner[sample] = candidate
    return [
        x[
            [
                owner[int(sample)][1:] == (chunk, row)
                for row, sample in enumerate(x[:, 4])
            ],
            :4,
        ]
        for chunk, x in enumerate(arrays)
    ]


def fitted_chunks(arrays, duration, config):
    chunks, used = [], []
    for index, x in enumerate(unique_event_arrays(arrays)):
        data = event_data(x, 0, duration, config)
        if data is None:
            continue
        time, y, formal = data
        error, _ = event_errors(time, y, formal, duration)
        chunks.append((time, y, error, duration))
        used.append(index)
    return chunks, used


def joint_event_fit(arrays, duration, config, *, units=None, limit=None):
    if (units is None) != (limit is None) or (
        units is not None and len(units) != len(arrays)
    ):
        raise ValueError("Joint-fit units and limit must match the candidate arrays")
    initial_chunks, initial_used = fitted_chunks(arrays, duration, config)
    if not initial_chunks:
        membership = {"chunks_used": 0, "used_chunk_indices": []}
        return {"status": "NO_USABLE_COVERAGE", "period_days": None, **membership}
    parameters = {
        "linear": 2,
        "quadratic": 3,
        "step": 3,
        "single_cadence": 3,
        "sinusoid": 5,
        "dropout": 5,
    }
    chunks, used, individual = [], [], []
    for chunk, index in zip(initial_chunks, initial_used):
        time, flux, error, _ = chunk
        models = alternatives(time, flux, error, duration)
        models["sinusoid"] = sinusoidal_alternative(time, flux, error, duration)
        models["dropout"] = dropout_alternative(time, flux, error, duration)
        if all(np.isfinite(models[name].get("chi2", np.inf)) for name in parameters):
            chunks.append(chunk)
            used.append(index)
            individual.append(models)
    rejected = len(initial_chunks) - len(chunks)
    eligible = len(chunks)
    if units is not None:
        from vetting_signals import representative_units

        chosen = {
            unit["unit_id"]
            for unit in representative_units([units[index] for index in used], limit)
        }
        keep = [units[index]["unit_id"] in chosen for index in used]
        chunks = [chunk for chunk, selected in zip(chunks, keep) if selected]
        individual = [models for models, selected in zip(individual, keep) if selected]
        used = [index for index, selected in zip(used, keep) if selected]
    membership = {
        "chunks_used": len(chunks),
        "used_chunk_indices": used,
        "chunks_rejected_for_incomplete_alternatives": rejected,
        "eligible_candidate_chunks": eligible,
    }
    if not chunks:
        return {
            "status": "SHAPE_ALTERNATIVES_UNAVAILABLE",
            "period_days": None,
            **membership,
        }
    time = np.concatenate([c[0] for c in chunks])
    y = np.concatenate([c[1] for c in chunks])
    if not np.any(abs(time) <= duration / 2) or len(time) <= 6 + 2 * len(chunks):
        return {
            "status": "PHYSICAL_FIT_UNDERCONSTRAINED",
            "period_days": None,
            **membership,
        }
    lo, hi = bounds(time, y, duration)
    radius = np.sqrt(max(-np.median(y[abs(time) <= duration / 2]), 1e-5))
    initial = np.clip(
        [0, np.log(radius), np.log(duration), 0.3, 0.25, 0.4],
        lo[:6] + 1e-6,
        hi[:6] - 1e-6,
    )
    boundaries = np.cumsum([0, *(len(chunk[0]) for chunk in chunks)])
    model = TransitCurve(time, config["exposure_days"], config["supersample_factor"])

    projections = [_BaselineProjection(t, err) for t, _, err, _ in chunks]

    def residual(theta):
        fitted = model([*theta, 0, 0])
        return np.concatenate(
            [
                projections[i](flux - fitted[boundaries[i] : boundaries[i + 1]])
                / err
                for i, (t, flux, err, _) in enumerate(chunks)
            ]
        )

    fit = least_squares(
        residual,
        initial,
        bounds=(lo[:6], hi[:6]),
        max_nfev=config["optimizer_max_evaluations"],
    )
    alts = {}
    for name, count in parameters.items():
        chi2 = sum(item[name].get("chi2", np.inf) for item in individual)
        if np.isfinite(chi2):
            alts[name] = {
                "status": "MEASURED",
                "chi2": float(chi2),
                "bic": float(chi2 + count * len(chunks) * np.log(len(time))),
            }
    alts["signed_box"] = joint_box(chunks)
    alts["shared_single_cadence"] = joint_single_cadence(chunks)
    return {
        "status": "FIT_COMPLETE" if fit.success else "OPTIMIZER_LIMIT",
        "theta": fit.x.tolist(),
        "chi2": float(fit.fun @ fit.fun),
        "bic": float(fit.fun @ fit.fun + (6 + 2 * len(chunks)) * np.log(len(time))),
        "alternatives": alts,
        "baseline_parameters": 2 * len(chunks),
        **membership,
        "sample_count": len(time),
        "period_days": None,
        "scope": "Shared exposure-integrated shape with a separately fitted line for each supplied event. Extraction views are never pooled. No period inferred by this fit.",
    }
