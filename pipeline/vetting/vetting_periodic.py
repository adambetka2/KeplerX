"""Verify timing hypotheses of supplied events; outputs are measurements, not alerts."""

import numpy as np
from lc2 import photometry, uses_flatten_mask
from lc2 import vetting_points as valid_points
from scipy.special import erfcinv, ndtr
from vetting_models import joint_event_fit, measure_interval


def weighted_depth(rows):
    measured = [r for r in rows if r.get("depth_error", 0) > 0]
    if not measured:
        return {"status": "UNCONSTRAINED"}
    depths = np.array([r["depth_fraction"] for r in measured])
    weights = 1 / np.array([r["depth_error"] for r in measured]) ** 2
    depth = float(np.sum(weights * depths) / weights.sum())
    return {
        "status": "MEASURED",
        "depth_fraction": depth,
        "depth_error": float(1 / np.sqrt(weights.sum())),
        "events": len(rows),
        "depth_chi2": float(np.sum(weights * (depths - depth) ** 2)),
        "degrees_of_freedom": len(measured) - 1,
        "largest_information_fraction": float(weights.max() / weights.sum()),
        "sample_count": sum(
            r.get("in_event_points", 0)
            + r.get("before_points", 0)
            + r.get("after_points", 0)
            for r in measured
        ),
    }


def event_train_support(events):
    measured = [
        event
        for event in events
        if event.get("status") == "MEASURED"
        and event.get("depth_error", 0) > 0
    ]
    if len(measured) < 3:
        return {"status": "UNCONSTRAINED", "events": len(measured)}
    aggregate = weighted_depth(measured)
    depth = np.array([event["depth_fraction"] for event in measured])
    error = np.array([event["depth_error"] for event in measured])
    expected = np.clip(ndtr(aggregate["depth_fraction"] / error), 1e-12, 1 - 1e-12)
    observed = depth > 0
    variance = np.sum(expected * (1 - expected))
    return {
        "status": "MEASURED",
        "events": len(measured),
        "observed_positive_fraction": float(np.mean(observed)),
        "expected_positive_fraction": float(np.mean(expected)),
        "positive_deficit_z": float(
            (observed.sum() - expected.sum()) / np.sqrt(variance)
        ),
        "scope": "Predicted-event dip signs under a shared measured depth and local errors; the z score is a coherence diagnostic, not a calibrated false-alarm probability",
    }


def depth_difference(first, second):
    if first["status"] != "MEASURED" or second["status"] != "MEASURED":
        return {"status": "NOT_APPLICABLE"}
    error = np.hypot(first["depth_error"], second["depth_error"])
    return {
        "status": "MEASURED",
        "difference": first["depth_fraction"] - second["depth_fraction"],
        "error": float(error),
        "sigma": float(abs(first["depth_fraction"] - second["depth_fraction"]) / error),
        "two_depth_bic_gain": float(
            ((first["depth_fraction"] - second["depth_fraction"]) / error) ** 2
            - np.log(max(first["sample_count"] + second["sample_count"], 1))
        ),
    }


def closest_indices(time, valid, centres):
    indices = np.flatnonzero(valid)
    values = time[indices]
    if len(values) < 2 or not np.all(values[:-1] <= values[1:]):
        return np.array(
            [indices[np.argmin(abs(values - centre))] for centre in centres]
        )
    right = np.searchsorted(values, centres)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(values) - 1)
    use_left = abs(values[left] - centres) <= abs(values[right] - centres)
    nearest_values = np.where(use_left, values[left], values[right])
    return indices[np.searchsorted(values, nearest_values, side="left")]


def expected_events(raw, hypothesis, duration, view, config, phase=0):
    curve = photometry(raw, view)
    time = curve["time"]
    valid = np.isfinite(time)
    period, epoch = hypothesis["period_days"], hypothesis["epoch_bkjd"]
    if not valid.any():
        return {"status": "NO_USABLE_COVERAGE", "events": []}
    first = int(np.ceil((time[valid].min() - epoch) / period - phase))
    last = int(np.floor((time[valid].max() - epoch) / period - phase))
    if last - first + 1 > config["orbital_max_windows"]:
        return {
            "status": "COMPUTATION_LIMIT",
            "expected_windows": last - first + 1,
            "events": [],
        }
    cycles = np.arange(first, last + 1)
    centres = epoch + (cycles + phase) * period
    nearest = closest_indices(time, valid, centres)
    x = np.column_stack(
        [
            curve["flux"],
            curve["error"],
            time,
            valid_points(
                curve, config["quality_bitmask"], uses_flatten_mask(view, config)
            ),
        ]
    )
    quarters = {
        int(quarter): x[curve["quarter"] == quarter]
        for quarter in np.unique(curve["quarter"])
    }
    ordered = {
        quarter: np.isfinite(data[:, 2]).all() and np.all(data[1:, 2] >= data[:-1, 2])
        for quarter, data in quarters.items()
    }
    records = []
    for cycle, centre, closest in zip(cycles, centres, nearest):
        quarter = int(curve["quarter"][closest])
        record = measure_interval(
            raw,
            centre,
            duration,
            view,
            config,
            quarter,
            prepared=quarters[quarter],
            sorted_time=ordered[quarter],
        )
        records.append(dict(record, cycle=int(cycle)))
    return {
        "status": "MEASURED",
        "events": records,
        "scope": "Flux at predicted times, including gaps and events not alerted by the detector. These measurements cannot create discoveries.",
    }


def compact_coverage(result):
    events = result.get("events", [])
    return {key: value for key, value in result.items() if key != "events"} | {
        "measured_windows": sum(event.get("status") == "MEASURED" for event in events),
        "covered_windows": sum(event.get("in_event_points", 0) > 0 for event in events),
        "expected_windows": result.get("expected_windows", len(events)),
    }


def phase_curve_chunks(raw, hypothesis, duration, view, config):
    curve = photometry(raw, view)
    valid = valid_points(
        curve, config["quality_bitmask"], uses_flatten_mask(view, config)
    )
    period, epoch = hypothesis["period_days"], hypothesis["epoch_bkjd"]
    phase = ((curve["time"] - epoch) / period + 0.5) % 1 - 0.5
    phase_guard = (duration + config["exposure_days"]) / (2 * period)
    valid &= (abs(phase) > phase_guard) & (
        abs(abs(phase) - 0.5) > phase_guard
    )
    chunks = []
    for quarter in np.unique(curve["quarter"][valid]):
        use = valid & (curve["quarter"] == quarter)
        y, error = curve["flux"][use], curve["error"][use]
        norm = np.median(y)
        if norm <= 0:
            continue
        time = curve["time"][use]
        y, error = y / norm - 1, error / norm
        centre, scatter = np.median(y), 1.4826 * np.median(abs(y - np.median(y)))
        if scatter > 0:
            threshold = np.sqrt(2) * erfcinv(1 / len(y)) * scatter
            keep = abs(y - centre) <= threshold
            time, y, error = time[keep], y[keep], error[keep]
        if len(y) < 4:
            continue
        base = np.column_stack([np.ones(len(y)), time - np.mean(time)])
        weighted = base / error[:, None]
        if np.linalg.matrix_rank(weighted) < 2:
            continue
        inverse = np.linalg.inv(weighted.T @ weighted)
        projection = inverse @ (weighted.T / error)
        chunks.append(
            {
                "quarter": int(quarter),
                "time": time,
                "y_residual": y - base @ (projection @ y),
                "error": error,
                "base": base,
                "projection": projection,
            }
        )
    return chunks


def coherent_sinusoid(chunks, epoch, period, period_ratio):
    normal = np.zeros((2, 2))
    rhs = np.zeros(2)
    baseline_chi2 = 0.0
    points = 0
    for chunk in chunks:
        angle = 2 * np.pi * (chunk["time"] - epoch) / (period * period_ratio)
        signal = np.column_stack([np.cos(angle), np.sin(angle)])
        residual_signal = signal - chunk["base"] @ (chunk["projection"] @ signal)
        weighted_signal = residual_signal / chunk["error"][:, None]
        weighted_y = chunk["y_residual"] / chunk["error"]
        normal += weighted_signal.T @ weighted_signal
        rhs += weighted_signal.T @ weighted_y
        baseline_chi2 += float(weighted_y @ weighted_y)
        points += len(weighted_y)
    if points < 3 or np.linalg.matrix_rank(normal) < 2:
        return {"status": "UNIDENTIFIABLE", "period_ratio": period_ratio}
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ rhs
    amplitude = float(np.hypot(*coefficients))
    direction = coefficients / amplitude if amplitude > 0 else np.ones(2) / np.sqrt(2)
    amplitude_error = float(np.sqrt(direction @ covariance @ direction))
    improvement = float(coefficients @ normal @ coefficients)
    return {
        "status": "MEASURED",
        "period_ratio": period_ratio,
        "period_days": float(period * period_ratio),
        "amplitude_fraction": amplitude,
        "amplitude_ppm": 1e6 * amplitude,
        "amplitude_error": amplitude_error,
        "signal_to_noise": amplitude / amplitude_error,
        "phase_radians": float(np.arctan2(coefficients[1], coefficients[0])),
        "bic_gain": improvement - 2 * np.log(points),
        "baseline_chi2": baseline_chi2,
        "modulation_chi2": baseline_chi2 - improvement,
        "points": points,
        "quarters": len(chunks),
    }


def phase_modulation(raw, hypothesis, duration, view, config):
    chunks = phase_curve_chunks(raw, hypothesis, duration, view, config)
    period, epoch = hypothesis["period_days"], hypothesis["epoch_bkjd"]
    signals = [
        coherent_sinusoid(chunks, epoch, period, ratio) for ratio in (0.5, 1.0, 2.0)
    ]
    measured = [signal for signal in signals if signal["status"] == "MEASURED"]
    return {
        "status": "MEASURED" if measured else "UNIDENTIFIABLE",
        "tce_period_days": float(period),
        "signals": signals,
        "strongest_signal": max(measured, key=lambda item: item["signal_to_noise"])
        if measured
        else None,
        "model": "Kepler SWEET-style coherent sine fits at half, equal and twice the TCE period; quarter-normalized out-of-event flux with independent quarter baselines and shared phase and amplitude",
    }


def folded_fit(raw, hypothesis, members, duration, view, config):
    from vetting_signals import representative_units

    curve = photometry(raw, view)
    valid = valid_points(
        curve, config["quality_bitmask"], uses_flatten_mask(view, config)
    )
    short = (
        hypothesis["period_days"] <= config["short_period_shape_max_days"]
        and hypothesis.get("detector_origin") == "shallow_bls_v1"
        and hypothesis.get("shallow_record", {}).get("search_band") == "short"
    )
    limit = config[
        "short_period_joint_fit_max_units"
        if short
        else "joint_fit_max_units_per_signal"
    ]
    candidates = representative_units(members, min(len(members), 2 * limit if short else limit))
    chunks, candidate_units = [], []
    for event in candidates:
        time = curve["time"] - event["centre_bkjd"]
        use = (
            valid
            & (curve["quarter"] == event["quarter"])
            & (
                abs(time)
                <= max(
                    config["minimum_context_days"],
                    config["local_context_durations"] * duration,
                )
            )
        )
        if not use.any():
            continue
        norm = np.median(curve["flux"][use])
        if norm > 0:
            rows = np.flatnonzero(use)
            chunks.append(
                np.column_stack(
                    [
                        curve["flux"][use] / norm,
                        curve["error"][use] / norm,
                        time[use],
                        np.ones(use.sum()),
                        rows,
                    ]
                )
            )
            candidate_units.append(event)
    if not chunks:
        return {"status": "NO_USABLE_COVERAGE"}
    fitted = joint_event_fit(
        chunks,
        duration,
        config,
        units=candidate_units if short else None,
        limit=limit if short else None,
    )
    fitted["requested_unit_ids"] = [event["unit_id"] for event in members]
    fitted["candidate_unit_ids"] = [event["unit_id"] for event in candidates]
    fitted["selected_unit_ids"] = [
        candidate_units[index]["unit_id"]
        for index in fitted.get("used_chunk_indices", [])
    ] if short else [event["unit_id"] for event in candidates]
    fitted["fitted_unit_ids"] = [
        candidate_units[index]["unit_id"]
        for index in fitted.get("used_chunk_indices", [])
    ]
    missing_fraction = (
        1 - fitted.get("eligible_candidate_chunks", 0) / len(candidates)
    )
    if short:
        if fitted["status"] == "FIT_COMPLETE" and (
            missing_fraction > config["short_period_joint_fit_max_missing_fraction"]
            or fitted.get("chunks_used", 0) < min(3, len(members))
        ):
            fitted["status"] = "INCOMPLETE_EVENT_MEMBERSHIP"
    elif fitted.get("chunks_used") != len(candidates):
        fitted["status"] = "INCOMPLETE_EVENT_MEMBERSHIP"
    fitted["selection"] = {
        "requested_units": len(members),
        "candidate_units": len(candidates),
        "eligible_candidate_units": fitted.get("eligible_candidate_chunks", 0),
        "missing_candidate_fraction": missing_fraction,
        "selected_units": len(fitted["selected_unit_ids"]),
        "limit": limit,
        "require_complete_membership": short,
        "rule": (
            "Quarter-covering, time-spread candidates; select from locally fit-eligible events"
            if short
            else "One central unit per quarter, then evenly spaced units across time"
        ),
    }
    fitted["scope"] = (
        "Joint local transit shape at a predeclared quarter-covering sample of supplied event centres, normalized per event. Independent event baselines are fitted simultaneously; this is not a new event search."
    )
    return fitted


def refine_event_timing(raw, members, duration, config, view="psf_flat"):
    from vetting_signals import representative_units

    selected = representative_units(
        members, min(len(members), config["timing_refinement_max_units"])
    )
    first_time = min(unit["centre_bkjd"] for unit in members)
    span = max(unit["centre_bkjd"] for unit in members) - first_time
    if span <= 0:
        return None
    curve = photometry(raw, view)
    valid = valid_points(
        curve, config["quality_bitmask"], uses_flatten_mask(view, config)
    )
    chunks = []
    for unit in selected:
        time = curve["time"] - unit["centre_bkjd"]
        use = (
            valid
            & (curve["quarter"] == unit["quarter"])
            & (abs(time) <= max(config["minimum_context_days"], 4 * duration))
        )
        if not use.any():
            continue
        local_time = time[use]
        flux = curve["flux"][use]
        error = curve["error"][use]
        outside = abs(local_time) > duration
        if outside.sum() < 4:
            continue
        baseline = np.column_stack([np.ones(outside.sum()), local_time[outside]])
        coefficients = np.linalg.lstsq(
            baseline / error[outside, None], flux[outside] / error[outside], rcond=None
        )[0]
        residual = flux - coefficients[0] - coefficients[1] * local_time
        position = (unit["centre_bkjd"] - first_time) / span
        chunks.append((local_time, residual, error, position))
    if len(chunks) < 3:
        return None
    offsets = np.linspace(
        -duration / 2, duration / 2, config["timing_refinement_grid_points"]
    )
    start, end = np.meshgrid(offsets, offsets, indexing="ij")
    numerator = np.zeros_like(start)
    denominator = np.zeros_like(start)
    for time, residual, error, position in chunks:
        shift = start + (end - start) * position
        inside = abs(time[None, None, :] - shift[:, :, None]) <= duration / 2
        weight = 1 / error**2
        numerator += np.sum(inside * (weight * residual), axis=2)
        denominator += np.sum(inside * weight, axis=2)
    score = np.divide(
        np.maximum(0, -numerator) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = np.unravel_index(np.argmax(score), score.shape)
    shift_start, shift_end = offsets[best[0]], offsets[best[1]]
    slope = (shift_end - shift_start) / span
    shifted = [
        unit | {
            "centre_bkjd": unit["centre_bkjd"]
            + shift_start
            + slope * (unit["centre_bkjd"] - first_time)
        }
        for unit in members
    ]
    return shifted, {
        "start_shift_days": float(shift_start),
        "end_shift_days": float(shift_end),
        "slope": float(slope),
        "first_centre_bkjd": float(first_time),
        "sampled_units": len(chunks),
        "original_box_score": float(score[len(offsets) // 2, len(offsets) // 2]),
        "refined_box_score": float(score[best]),
        "search": "Fixed-duration box on the supplied event windows; endpoint shifts limited to half the proposed duration",
    }


def hypothesis_measurements(raw, hypothesis, members, duration, views, config):
    measurements = {}
    for view in views:
        primary = expected_events(raw, hypothesis, duration, view, config)
        secondary = expected_events(raw, hypothesis, duration, view, config, 0.5)
        odd = weighted_depth([r for r in primary["events"] if r["cycle"] % 2])
        even = weighted_depth([r for r in primary["events"] if not r["cycle"] % 2])
        measurements[view] = {
            "primary": compact_coverage(primary),
            "primary_depth": weighted_depth(primary["events"]),
            "primary_train": event_train_support(primary["events"]),
            "odd": odd,
            "even": even,
            "odd_even": depth_difference(odd, even),
            "secondary_phase_half": weighted_depth(secondary["events"]),
            "phase_modulation": phase_modulation(
                raw, hypothesis, duration, view, config
            ),
            "joint_shape": folded_fit(raw, hypothesis, members, duration, view, config),
        }
    return measurements


def verify_hypothesis(raw, hypothesis, units, views, config):
    from vetting_disposition import shape_evidence
    from vetting_signals import timing_units

    units = timing_units(units, hypothesis.get("timing_basis", "ORIGINAL_PROPOSAL"))
    members = [u for u in units if u["unit_id"] in hypothesis["unit_ids"]]
    duration = float(np.median([m["duration_days"] for m in members]))
    measurements = hypothesis_measurements(
        raw, hypothesis, members, duration, views, config
    )
    result = {
        "status": "MEASURED",
        "hypothesis": hypothesis,
        "views": measurements,
        "scope": "Circular phase-half secondary is measured without rejecting planetary occultations. Eccentric secondary phases remain untested; TTVs retained in timing residuals. Views are correlated and never combined as independent evidence.",
    }
    discovery_view = hypothesis.get("discovery_view", "psf_flat")
    original = measurements.get(discovery_view, {}).get("joint_shape", {})
    original_score = shape_evidence(original).get("delta_bic")
    if (
        hypothesis.get("timing_basis") != "ORIGINAL_PROPOSAL"
        or hypothesis.get("detector_origin") != "rust_v6_deterministic"
        or hypothesis["period_days"] > config["timing_refinement_max_period_days"]
        or len(members) <= config["timing_refinement_max_units"]
        or original_score is None
        or original_score >= -config["triage"]["shape_false_alarm_bic_margin"]
    ):
        return result
    refinement = (
        refine_event_timing(raw, members, duration, config)
        if discovery_view == "psf_flat"
        else refine_event_timing(raw, members, duration, config, discovery_view)
    )
    if refinement is None:
        result["timing_refinement"] = {"status": "INSUFFICIENT_EVENT_COVERAGE"}
        return result
    shifted, timing = refinement
    shifted_fit = folded_fit(raw, hypothesis, shifted, duration, discovery_view, config)
    raw_score = shape_evidence(shifted_fit).get("delta_bic")
    penalty = 2 * np.log(shifted_fit["sample_count"]) if raw_score is not None else None
    result["timing_refinement"] = {
        "status": "DIAGNOSTIC",
        "timing": timing,
        "original_shape_score": original_score,
        "refined_shape_score": raw_score,
        "timing_bic_penalty": float(penalty) if penalty is not None else None,
        "adjusted_shape_score": raw_score - penalty if penalty is not None else None,
    }
    if (
        penalty is None
        or raw_score - penalty <= config["triage"]["shape_transit_bic_margin"]
    ):
        return result
    slope = timing["slope"]
    refined_hypothesis = hypothesis | {
        "period_days": hypothesis["period_days"] * (1 + slope),
        "epoch_bkjd": hypothesis["epoch_bkjd"]
        + timing["start_shift_days"]
        + slope * (hypothesis["epoch_bkjd"] - timing["first_centre_bkjd"]),
        "timing_basis": "BOUNDED_REFINED_PROPOSAL",
    }
    train = event_train_support(
        expected_events(raw, refined_hypothesis, duration, discovery_view, config)[
            "events"
        ]
    )
    result["timing_refinement"]["event_train"] = train
    if (
        train["status"] != "MEASURED"
        or train["positive_deficit_z"]
        < config["timing_refinement_minimum_event_train_positive_z"]
    ):
        result["timing_refinement"]["status"] = "EVENT_TRAIN_INCOHERENT"
        return result
    result["timing_refinement"].update(
        status="SELECTED",
        hypothesis=refined_hypothesis,
        views=hypothesis_measurements(
            raw, refined_hypothesis, shifted, duration, views, config
        ),
    )
    return result


def verify(row, raw, config):
    from vetting_flux import finite
    from vetting_signals import event_units
    from vetting_triage import measured_hypothesis_ids

    units = event_units(row["events"])
    if "supplied_hypotheses" in row:
        units = [unit for signal in row["signals"] for unit in signal["events"]]
    event_views = {e["photometry"] for e in row["events"]}
    active = measured_hypothesis_ids(row)
    hypotheses = [
        hypothesis
        for hypothesis in row["association"]["hypotheses"]
        if hypothesis["hypothesis_id"] in active
    ]
    results = {}
    for hypothesis in hypotheses:
        views = sorted(event_views | set(hypothesis.get("forced_photometry_views", [])))
        result = verify_hypothesis(raw, hypothesis, units, views, config)
        results[hypothesis["hypothesis_id"]] = finite(result)
    row["orbital"] = {
        "status": "MEASURED" if hypotheses else "NOT_APPLICABLE_UNKNOWN_PERIOD",
        "hypotheses": results,
        "evaluated_hypotheses": len(hypotheses),
        "skipped_dismissed_hypotheses": len(row["association"]["hypotheses"])
        - len(hypotheses),
    }
    for signal in row["signals"]:
        signal["orbital_hypothesis_ids"] = [
            hypothesis["hypothesis_id"]
            for hypothesis in signal.get("timing_hypotheses", [])
        ]
    return row


if __name__ == "__main__":
    from vetting import program_main

    program_main("orbital")
