"""Gate signals in declared order; survey runs skip later stages after the first failure."""

import math
from collections import Counter

from vetting_disposition import (
    GATE_ORDER,
    TRIAGE_CONTRACT,
    contamination_assessment,
    instrument_assessment,
    localization_assessment,
    orbital_assessment,
    shape_evidence,
    timing_shape_score,
    validate_gate_result,
)

TRIAGE_EXECUTIONS = tuple(TRIAGE_CONTRACT["execution_modes"])
SHAPE_VIEW_PRIORITY = ("psf_flat", "psf_qdiag", "psf", "sap_qdiag", "sap")


def signal_active(signal):
    return signal.get("triage", {}).get("status", "ACTIVE") == "ACTIVE"


def signal_measured(signal):
    triage = signal.get("triage", {})
    return triage.get("record_every_gate") is True or signal_active(signal)


def measured_signals(row):
    return [signal for signal in row.get("signals", []) if signal_measured(signal)]


def measured_event_ids(row):
    return {
        event_id for signal in measured_signals(row) for event_id in signal["event_ids"]
    }


def measured_events(row):
    event_ids = measured_event_ids(row)
    return [event for event in row["events"] if event["event_id"] in event_ids]


def measured_hypothesis_ids(row):
    return {
        hypothesis_id
        for signal in measured_signals(row)
        for hypothesis_id in [
            item["hypothesis_id"] for item in signal.get("timing_hypotheses", [])
        ]
    }


def gate_summary(states):
    dismissed, held = Counter(), Counter()
    for state in states:
        dismissed.update(state.get("dismissed_by_stage", {}))
        held.update(state.get("held_by_stage", {}))
    return {
        "sources": len(states),
        "signals_total": sum(state.get("signals_total", 0) for state in states),
        "signals_active": sum(state.get("signals_active", 0) for state in states),
        "signals_held": sum(state.get("signals_held", 0) for state in states),
        "signals_dismissed": sum(state.get("signals_dismissed", 0) for state in states),
        "sources_with_active_signals": sum(
            state.get("signals_active", 0) > 0 for state in states
        ),
        "held_by_stage": dict(held),
        "dismissed_by_stage": dict(dismissed),
    }


def canonical_unit_score(row, unit, discovery_view=None):
    events = [
        event for event in row["events"] if event["event_id"] in unit["event_ids"]
    ]
    records = [
        {
            "event_id": event["event_id"],
            "view": event["photometry"],
            **shape_evidence(event["fit"]),
        }
        for event in events
        if event["photometry"] in SHAPE_VIEW_PRIORITY
    ]
    measured = {
        record["view"]
        for record in records
        if record.get("delta_bic") is not None and math.isfinite(record["delta_bic"])
    }
    priority = (discovery_view,) if discovery_view is not None else SHAPE_VIEW_PRIORITY
    view = next((name for name in priority if name in measured), None)
    selected = [record for record in records if record["view"] == view]
    if len(selected) != 1:
        return {"status": "UNRESOLVED", "view": view, "available": records}
    return {**selected[0], "available": records}


def shape_gate(row, signal, policy):
    if len(signal.get("events", [])) > 1:
        return repeated_shape_gate(row, signal, policy)
    records = [
        canonical_unit_score(row, unit, signal.get("discovery_view"))
        for unit in signal["events"]
    ]
    measured = [record for record in records if record.get("delta_bic") is not None]
    score = sum(record["delta_bic"] for record in measured)
    evidence = {
        "score": score if measured else None,
        "measured_event_units": len(measured),
        "event_units": len(records),
        "records": records,
        "combination": (
            "Sum of canonical-view BIC differences across disjoint event units"
        ),
    }
    if len(measured) != len(records):
        return outcome("CONTINUE_INCONCLUSIVE", "SHAPE_EVIDENCE_INCOMPLETE", evidence)
    return shape_decision(score, evidence, policy)


def repeated_shape_gate(row, signal, policy):
    units = {unit["unit_id"] for unit in signal["events"]}
    hypotheses = [
        item
        for item in signal.get("timing_hypotheses", [])
        if item.get("timing_basis") == "ORIGINAL_PROPOSAL"
        and set(item.get("unit_ids", [])) == units
    ]
    if len(hypotheses) != 1:
        return outcome(
            "CONTINUE_INCONCLUSIVE",
            "TIMING_ALIAS_UNRESOLVED",
            {"original_timing_hypotheses": len(hypotheses)},
        )
    result = (
        row.get("orbital", {})
        .get("hypotheses", {})
        .get(hypotheses[0]["hypothesis_id"], {})
    )
    records = [
        {"view": view, **shape_evidence(data.get("joint_shape", {}))}
        for view, data in result.get("views", {}).items()
    ]
    measured = {
        record["view"]
        for record in records
        if record.get("delta_bic") is not None and math.isfinite(record["delta_bic"])
    }
    priority = (
        (signal["discovery_view"],)
        if "discovery_view" in signal
        else SHAPE_VIEW_PRIORITY
    )
    view = next((name for name in priority if name in measured), None)
    selected = [record for record in records if record["view"] == view]
    evidence = {
        "view": view,
        "records": records,
        "measurement": "JOINT_ORIGINAL_TIMING_SHAPE",
    }
    if len(selected) != 1:
        return outcome("CONTINUE_INCONCLUSIVE", "SHAPE_EVIDENCE_INCOMPLETE", evidence)
    score = selected[0]["delta_bic"]
    selected_score, refinement = timing_shape_score(result, score)
    if refinement is not None:
        evidence["timing_refinement"] = refinement
        if selected_score > score:
            score = selected_score
            evidence["measurement"] = "BOUNDED_REFINED_TIMING_SHAPE"
    evidence["score"] = score
    return shape_decision(score, evidence, policy)


def shape_decision(score, evidence, policy):
    if score < -policy["shape_false_alarm_bic_margin"]:
        return outcome("DISMISS", "NON_TRANSIT_MODEL_STRONGLY_PREFERRED", evidence)
    if score > policy["shape_transit_bic_margin"]:
        return outcome("PASS", "TRANSIT_MODEL_PREFERRED", evidence)
    return outcome("CONTINUE_INCONCLUSIVE", "SHAPE_MODEL_COMPARISON_CLOSE", evidence)


def assessment_policy(policy):
    return {
        "status": "RELEASED",
        "assessment_rules": policy["assessment_rules"],
    }


def assessment_gate(assessment, conflict_reason, pass_reason):
    if assessment["status"] == "MATERIAL_CONFLICT":
        return outcome("DISMISS", conflict_reason, assessment)
    if assessment["status"] == "ASSESSED_NO_MATERIAL_CONFLICT":
        return outcome("PASS", pass_reason, assessment)
    return outcome("CONTINUE_INCONCLUSIVE", "EVIDENCE_INCOMPLETE", assessment)


def orbital_gate(row, signal, policy):
    if len(signal.get("events", [])) == 1:
        return outcome("PASS", "SINGLE_EVENT_ORBITAL_CHECK_NOT_APPLICABLE", {})
    assessment = orbital_assessment(row, signal, assessment_policy(policy))
    if assessment["status"] == "MATERIAL_CONFLICT":
        return outcome("DISMISS", "EB_LIKE_ORBITAL_EVIDENCE", assessment)
    alias = row.get("alias", {}).get("signals", {}).get(signal["signal_id"], {})
    if alias.get("status") == "REPRODUCED_SOURCE_TWO_ECLIPSE":
        if assessment["status"] != "ASSESSED_NO_MATERIAL_CONFLICT":
            return assessment_gate(
                assessment,
                "EB_LIKE_ORBITAL_EVIDENCE",
                "NO_MATERIAL_ORBITAL_CONFLICT",
            )
        return outcome(
            "PASS",
            "SOURCE_TWO_ECLIPSE_CONTEXT_PENDING",
            {"orbital": assessment, "alias": alias},
        )
    if alias.get("status") == "INCOMPLETE_SOURCE_TWO_ECLIPSE_COMPARISON":
        return outcome(
            "CONTINUE_INCONCLUSIVE",
            "SOURCE_TWO_ECLIPSE_COMPARISON_INCOMPLETE",
            {"orbital": assessment, "alias": alias},
        )
    if alias.get("status") == "INDEPENDENT_CANDIDATE_DIP_SUPPORT":
        return outcome(
            "CONTINUE_INCONCLUSIVE",
            "CANDIDATE_DIP_OUTSIDE_SOURCE_ECLIPSES",
            {"orbital": assessment, "alias": alias},
        )
    if alias.get("status") == "COHERENT_SHORTER_PERIOD":
        return outcome(
            "CONTINUE_INCONCLUSIVE",
            "LOWER_PERIOD_ALIAS_REQUIRES_RESOLUTION",
            {"orbital": assessment, "alias": alias},
        )
    if alias.get("status") not in {
        "NO_SUPPORTED_ALIAS_IN_SHORTLIST",
        "INCOMPLETE_SHORTER_PERIOD_COMPARISON",
    }:
        return outcome(
            "CONTINUE_INCONCLUSIVE",
            "ALIAS_MEASUREMENT_UNAVAILABLE",
            {"orbital": assessment, "alias": alias},
        )
    return assessment_gate(
        assessment,
        "EB_LIKE_ORBITAL_EVIDENCE",
        "NO_MATERIAL_ORBITAL_CONFLICT",
    )


def instrument_gate(row, signal, policy):
    from vetting_signals import representative_units

    records = {
        record["event_id"]: record
        for record in row.get("instrument", {}).get("events", [])
    }
    selected = representative_units(
        signal["events"], row["instrument"]["selection"]["maximum_units_per_signal"]
    )
    complete = all(
        any(
            event_id in records
            and records[event_id].get("status") == "MEASURED"
            and records[event_id].get("in_event_points", 0) > 0
            for event_id in unit["event_ids"]
        )
        for unit in selected
    )
    evidence = {
        "event_units": len(signal["events"]),
        "evaluated_units": len(selected),
        "rule": "Every selected event unit must have a measured instrument record",
    }
    if complete:
        return outcome("PASS", "INSTRUMENT_MEASUREMENTS_AVAILABLE", evidence)
    return outcome("CONTINUE_INCONCLUSIVE", "INSTRUMENT_EVIDENCE_INCOMPLETE", evidence)


def localization_gate(row, signal, policy):
    assessment = localization_assessment(row, signal, assessment_policy(policy))
    return assessment_gate(
        assessment,
        "SOURCE_LOCALIZATION_CONFLICT",
        "SOURCE_LOCALIZATION_COMPATIBLE",
    )


def contamination_gate(row, signal, policy):
    from catalogue_search import admission

    state, reason = admission(signal)
    signal["catalogue_state"] = state
    if reason is not None:
        return outcome("CONTINUE_INCONCLUSIVE", reason, signal.get("catalogue_search", {}))
    index = row["signals"].index(signal)
    assessment = contamination_assessment(row, signal, index, assessment_policy(policy))
    result = assessment_gate(
        assessment,
        "COMPETING_SOURCE_EVIDENCE",
        "NO_MATERIAL_COMPETING_SOURCE_EVIDENCE",
    )
    if result["decision"] != "PASS":
        return result
    alias = row.get("alias", {}).get("signals", {}).get(signal["signal_id"], {})
    if alias.get("status") != "REPRODUCED_SOURCE_TWO_ECLIPSE":
        return result
    instrument = instrument_assessment(row, signal, assessment_policy(policy))
    evidence = {"alias": alias, "instrument": instrument, "contamination": assessment}
    if instrument["status"] != "ASSESSED_NO_MATERIAL_CONFLICT":
        return outcome(
            "CONTINUE_INCONCLUSIVE", "SOURCE_TWO_ECLIPSE_ATTRIBUTION_INCOMPLETE", evidence
        )
    return outcome("DISMISS", "REPRODUCED_SOURCE_TWO_ECLIPSE_MECHANISM", evidence)


def outcome(decision, reason, evidence):
    return {"decision": decision, "reason": reason, "evidence": evidence}


def fail(signal, stage, result, record):
    status = TRIAGE_CONTRACT["decision_states"][result["decision"]]
    signal["triage"]["status"] = status
    signal["triage"][f"{status.lower()}_stage"] = stage
    signal["triage"]["reason"] = result["reason"]
    if record:
        return
    for later in GATE_ORDER[GATE_ORDER.index(stage) + 1 :] + ("host",):
        signal[later] = {"status": f"SKIPPED_{status}", "reason": result["reason"]}


def apply_gate(row, stage, config):
    policy = config.get("triage")
    if not policy or policy.get("enabled") is not True:
        return row
    if config["triage_execution"] not in TRIAGE_EXECUTIONS:
        raise ValueError(f"Unknown triage execution: {config['triage_execution']}")
    record = config["triage_execution"] == "RECORD_EVERY_GATE"
    gates = {
        "signals": ("shape", shape_gate),
        "orbital": ("orbital", orbital_gate),
        "instrument": ("instrument", instrument_gate),
        "localization": ("localization", localization_gate),
        "contamination": ("contamination", contamination_gate),
    }
    name, gate = gates[stage]
    for signal in measured_signals(row):
        signal.setdefault("triage", {"status": "ACTIVE", "history": []})
        if record:
            signal["triage"]["record_every_gate"] = True
        result = gate(row, signal, policy)
        validate_gate_result(name, result)
        signal["triage"]["history"].append({"stage": name, **result})
        if signal_active(signal) and result["decision"] != "PASS":
            fail(signal, name, result, record)
    if stage == "signals":
        from vetting_flux import append_refined_pixel_jobs

        append_refined_pixel_jobs(row, config)
    dismissed = [
        signal for signal in row["signals"] if signal["triage"]["status"] == "DISMISSED"
    ]
    held = [signal for signal in row["signals"] if signal["triage"]["status"] == "HELD"]
    row["triage"] = {
        "status": "MEASURED",
        "last_stage": name,
        "signals_total": len(row["signals"]),
        "signals_active": len(row["signals"]) - len(dismissed) - len(held),
        "signals_held": len(held),
        "signals_dismissed": len(dismissed),
        "held_by_stage": dict(
            Counter(signal["triage"]["held_stage"] for signal in held)
        ),
        "dismissed_by_stage": dict(
            Counter(signal["triage"]["dismissed_stage"] for signal in dismissed)
        ),
    }
    return row
