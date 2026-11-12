"""Measure shorter-period eclipse trains competing with supplied repeat signals."""

import numpy as np
from lc2 import photometry
from vetting_periodic import depth_difference, expected_events, weighted_depth
from vetting_triage import measured_signals


def aligned_fraction(signal, row):
    centres = np.array([unit["centre_bkjd"] for unit in signal["events"]])
    if len(centres) < 2:
        return 0.0
    period = float(row["period_days"])
    phase = (centres - float(row["t0_bkjd"]) + period / 2) % period - period / 2
    distance = np.minimum(abs(phase), abs(abs(phase) - period / 2))
    duration = float(row["duration_hours"]) / 24
    candidate_duration = np.array([unit["duration_days"] for unit in signal["events"]])
    return float(np.mean(distance <= (duration + candidate_duration) / 2))


def candidate_ephemeris(row, signal):
    hypothesis = signal["timing_hypotheses"][0]
    measured = (
        row.get("orbital", {})
        .get("hypotheses", {})
        .get(hypothesis["hypothesis_id"], {})
    )
    refinement = measured.get("timing_refinement", {})
    return (
        refinement["hypothesis"]
        if refinement.get("status") == "SELECTED"
        else hypothesis
    )


def independent_depth(events, hypothesis):
    period = hypothesis["period_days"]
    epoch = hypothesis["epoch_bkjd"]
    duration = hypothesis["duration_days"]
    remaining = [
        event
        for event in events
        if abs((event["centre_bkjd"] - epoch + period / 2) % period - period / 2)
        > (duration + event["duration_days"]) / 2
    ]
    depth = weighted_depth(remaining)
    if depth["status"] == "MEASURED":
        depth["snr"] = depth["depth_fraction"] / depth["depth_error"]
        depth["measured_events"] = sum(
            event.get("depth_error", 0) > 0 for event in remaining
        )
    return depth


def measure_train(raw, row, config):
    hypothesis = {
        "period_days": float(row["period_days"]),
        "epoch_bkjd": float(row["t0_bkjd"]),
    }
    duration = float(row["duration_hours"]) / 24
    return [
        expected_events(raw, hypothesis, duration, row["photometry"], config, phase)
        for phase in (0, 0.5)
    ]


def eligible_rows(row, signal, config):
    return sorted(
        (
            candidate
            for candidate in row.get("rust_rows", [])
            if candidate["photometry"] == signal.get("discovery_view", "psf_flat")
            and candidate["classification"] == "EB_LIKE"
            and float(candidate["period_days"]) < signal["period_days"]
        ),
        key=lambda candidate: -float(candidate["search_mes"]),
    )[: config["max_eb_rows_per_signal"]]


def source_eb_rank_one(row):
    return next(
        (
            candidate
            for candidate in row.get("rust_rows", [])
            if candidate["photometry"] == row.get("discovery_view", "psf_flat")
            and int(candidate["signal_rank"]) == 1
            and candidate["classification"] in {"TRANSIT_LIKE", "EB_LIKE"}
        ),
        None,
    )


def source_eb_alignment(signal, candidate, rule):
    if candidate is None or signal.get("period_days") is None:
        return None
    period = float(candidate["period_days"])
    ratio = period / signal["period_days"]
    harmonic = round(ratio)
    if not (
        1 <= harmonic <= rule["maximum_source_eb_harmonic_order"]
        and abs(ratio / harmonic - 1)
        <= rule["maximum_source_eb_harmonic_period_error_fraction"]
    ):
        return None
    events = signal["events"]
    if not events:
        return None
    duration = float(candidate["duration_hours"]) / 24
    epoch = float(candidate["t0_bkjd"])
    aligned = sum(
        abs((event["centre_bkjd"] - epoch + period / 2) % period - period / 2)
        <= (duration + event["duration_days"]) / 2
        for event in events
    )
    fraction = aligned / len(events)
    if (
        aligned < rule["minimum_independent_events"]
        or fraction < rule["minimum_source_eb_harmonic_alignment"] / harmonic
    ):
        return None
    return {
        "harmonic_order": harmonic,
        "period_error_fraction": abs(ratio / harmonic - 1),
        "aligned_supplied_events": aligned,
        "supplied_events": len(events),
        "aligned_fraction": fraction,
    }


def source_eb_candidates(row, signals, rule):
    candidate = source_eb_rank_one(row)
    return candidate, {
        signal["signal_id"]: alignment
        for signal in signals
        if (alignment := source_eb_alignment(signal, candidate, rule)) is not None
    }


def single_event_source_eb_candidates(row, signals):
    candidate = source_eb_rank_one(row)
    if candidate is None or candidate["classification"] != "EB_LIKE":
        return {}
    period = float(candidate["period_days"])
    epoch = float(candidate["t0_bkjd"])
    duration = float(candidate["duration_hours"]) / 24
    aligned = {}
    for signal in signals:
        if len(signal["events"]) != 1:
            continue
        event = signal["events"][0]
        distance = abs(
            (event["centre_bkjd"] - epoch + period / 2) % period - period / 2
        )
        tolerance = (duration + event["duration_days"]) / 2
        if distance <= tolerance:
            aligned[signal["signal_id"]] = {
                "source_period_days": period,
                "source_epoch_bkjd": epoch,
                "phase_distance_days": distance,
                "event_window_tolerance_days": tolerance,
            }
    return aligned


def source_two_eclipse_model(raw, candidate, config):
    hypothesis = {
        "period_days": float(candidate["period_days"]),
        "epoch_bkjd": float(candidate["t0_bkjd"]),
    }
    train = expected_events(
        raw,
        hypothesis,
        float(candidate["duration_hours"]) / 24,
        candidate["photometry"],
        config,
    )
    if train["status"] != "MEASURED":
        return {
            "status": "INCOMPLETE_TWO_ECLIPSE_COMPARISON",
            "reason": train["status"],
            "rust_signal_rank": 1,
        }
    measured = [
        event for event in train["events"] if event["status"] == "MEASURED"
    ]
    depths = []
    for parity in (0, 1):
        events = [event for event in measured if event["cycle"] % 2 == parity]
        depth = weighted_depth(events)
        if depth["status"] == "MEASURED":
            depth["snr"] = depth["depth_fraction"] / depth["depth_error"]
            depth["measured_events"] = len(events)
        depths.append(depth)
    odd_even = (
        depth_difference(*depths)
        if all(depth["status"] == "MEASURED" for depth in depths)
        else {"status": "NOT_APPLICABLE"}
    )
    rule = config["alias"]
    status = (
        "INCOMPLETE_TWO_ECLIPSE_COMPARISON"
        if not all(adequately_sampled(depth, rule) for depth in depths)
        else (
            "REPRODUCED_UNEQUAL_TWO_ECLIPSE"
            if all(supported_phase(depth, rule) for depth in depths)
            and odd_even["two_depth_bic_gain"]
            > rule["minimum_source_eb_odd_even_bic_gain"]
            else "NO_SUPPORTED_TWO_ECLIPSE"
        )
    )
    return {
        "status": status,
        "rust_signal_rank": 1,
        "period_days": hypothesis["period_days"],
        "binary_period_days": 2 * hypothesis["period_days"],
        "epoch_bkjd": hypothesis["epoch_bkjd"],
        "even": depths[0],
        "odd": depths[1],
        "odd_even": odd_even,
        "scope": "EB-compatible alternating positive dip trains on original PSF_FLAT flux at the immutable rank-one Rust ephemeris; source attribution requires instrument and pixel context.",
    }


def independent_candidate_train(raw, row, signal, candidate, config):
    hypothesis = candidate_ephemeris(row, signal)
    train = expected_events(
        raw,
        hypothesis,
        hypothesis["duration_days"],
        signal.get("discovery_view", "psf_flat"),
        config,
    )
    if train["status"] != "MEASURED":
        return {"status": train["status"]}
    source_hypothesis = {
        "period_days": float(candidate["period_days"]),
        "epoch_bkjd": float(candidate["t0_bkjd"]),
        "duration_days": float(candidate["duration_hours"]) / 24,
    }
    return independent_depth(train["events"], source_hypothesis)


def needs_flux(row, config):
    rule = config["alias"]
    active = measured_signals(row)
    signals = [
        signal
        for signal in active
        if signal.get("period_days") is not None and len(signal["events"]) > 1
    ]
    _, source_candidates = source_eb_candidates(row, signals, rule)
    return (
        bool(source_candidates)
        or bool(single_event_source_eb_candidates(row, active))
        or any(
            aligned_fraction(signal, candidate)
            >= rule["minimum_supplied_event_alignment"]
            for signal in signals
            for candidate in eligible_rows(row, signal, rule)
        )
    )


def adequately_sampled(depth, rule):
    return (
        depth.get("status") == "MEASURED"
        and depth["measured_events"] >= rule["minimum_independent_events"]
        and depth["largest_information_fraction"]
        <= rule["maximum_single_event_information_fraction"]
    )


def supported_phase(depth, rule):
    return (
        adequately_sampled(depth, rule)
        and depth["snr"] >= rule["minimum_independent_depth_snr"]
    )


def shortlist_status(records, rule):
    if any(record["coherent_shorter_period"] for record in records):
        return "COHERENT_SHORTER_PERIOD"
    if any(
        not adequately_sampled(phase, rule)
        for record in records
        for phase in record["independent_phases"]
    ):
        return "INCOMPLETE_SHORTER_PERIOD_COMPARISON"
    return "NO_SUPPORTED_ALIAS_IN_SHORTLIST"


def measure_signal(raw, row, signal, config, cache):
    rule = config["alias"]
    records = []
    hypothesis = candidate_ephemeris(row, signal)
    for candidate in eligible_rows(row, signal, rule):
        fraction = aligned_fraction(signal, candidate)
        if fraction < rule["minimum_supplied_event_alignment"]:
            continue
        rank = int(candidate["signal_rank"])
        if rank not in cache:
            cache[rank] = measure_train(raw, candidate, config)
        trains = cache[rank]
        phases = []
        for phase, train in zip((0, 0.5), trains):
            depth = (
                independent_depth(train["events"], hypothesis)
                if train["status"] == "MEASURED"
                else {"status": train["status"]}
            )
            phases.append({"phase": phase, **depth})
        supported = any(supported_phase(depth, rule) for depth in phases)
        records.append(
            {
                "rust_signal_rank": rank,
                "period_days": float(candidate["period_days"]),
                "supplied_event_alignment": fraction,
                "independent_phases": phases,
                "coherent_shorter_period": supported,
            }
        )
    return {
        "status": shortlist_status(records, rule),
        "tested_eb_rows": len(records),
        "records": records,
        "scope": "Top detector-ranked EB_LIKE rows; a missing supported alias is not proof that the candidate period is unique.",
    }


def measured_repeated_aliases(
    row, raw, config, signals, source_candidate, source_candidates, source_model
):
    cache = {}
    records = {}
    for signal in signals:
        record = measure_signal(raw, row, signal, config, cache)
        association = source_candidates.get(signal["signal_id"])
        if association is not None:
            record["source_two_eclipse"] = {**association, "model": source_model}
            if source_model["status"] == "REPRODUCED_UNEQUAL_TWO_ECLIPSE":
                if association["harmonic_order"] == 1:
                    record["status"] = "REPRODUCED_SOURCE_TWO_ECLIPSE"
                else:
                    independent = independent_candidate_train(
                        raw, row, signal, source_candidate, config
                    )
                    record["source_two_eclipse"]["independent_candidate_train"] = (
                        independent
                    )
                    if not adequately_sampled(independent, config["alias"]):
                        record["status"] = "INCOMPLETE_SOURCE_TWO_ECLIPSE_COMPARISON"
                    elif supported_phase(independent, config["alias"]):
                        record["status"] = "INDEPENDENT_CANDIDATE_DIP_SUPPORT"
                    else:
                        record["status"] = "REPRODUCED_SOURCE_TWO_ECLIPSE"
            elif (
                source_model["status"] == "INCOMPLETE_TWO_ECLIPSE_COMPARISON"
                and association["harmonic_order"] > 1
            ):
                record["status"] = "INCOMPLETE_SOURCE_TWO_ECLIPSE_COMPARISON"
        records[signal["signal_id"]] = record
    return records


def measure(row, raw, config):
    active = measured_signals(row)
    signals = [
        signal
        for signal in active
        if signal.get("period_days") is not None and len(signal["events"]) > 1
    ]
    single_candidates = single_event_source_eb_candidates(row, active)
    if not signals and not single_candidates:
        row["alias"] = {"status": "NOT_APPLICABLE", "signals": {}}
        return row
    if needs_flux(row, config):
        if raw is None:
            raise ValueError("Aligned shorter-period rows require raw flux")
        time = photometry(
            raw, config.get("discovery_contract", {}).get("discovery_view", "psf_flat")
        )["time"]
        if not np.isfinite(time).any():
            row["alias"] = {"status": "NO_USABLE_COVERAGE", "signals": {}}
            return row
    source_candidate, source_candidates = source_eb_candidates(
        row, signals, config["alias"]
    )
    source_model = (
        source_two_eclipse_model(raw, source_candidate, config)
        if source_candidates or single_candidates
        else None
    )
    records = measured_repeated_aliases(
        row, raw, config, signals, source_candidate, source_candidates, source_model
    )
    for signal_id, alignment in single_candidates.items():
        records[signal_id] = {
            "status": "SINGLE_EVENT_SOURCE_EB_DIAGNOSTIC",
            "source_two_eclipse": {**alignment, "model": source_model},
            "scope": "Measured source eclipse context only; no automatic single-event attribution or disposition rule is calibrated.",
        }
    row["alias"] = {
        "status": "MEASURED",
        "signals": records,
    }
    return row
