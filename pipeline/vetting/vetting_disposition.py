import hashlib
import json
import math

POLICY = {
    "id": "deterministic-boi-v1",
    "status": "BOI_CATALOGUE",
    "shape": "Use one predeclared detector-view signal comparison. Repeated events use the detector-supplied ephemeris and joint fit. Tied or non-identifiable transit and nuisance models cannot support a candidate.",
    "reference": "https://arxiv.org/abs/1607.08237; https://arxiv.org/abs/1710.06758; https://arxiv.org/abs/2103.12538",
    "scope": "A Bonus Object of Interest (BOI) is a transit signal that passes every current pipeline triage gate. Single events remain eligible with unknown periods. BOI is a survey designation; statistical validation, confirmation and novelty require separate evidence.",
}

ASSESSMENTS = (
    "event_shape",
    "competing_explanations",
    "source_association",
    "identity_catalogues",
)

AUTOMATIC_ASSESSMENT_RULES = {
    "instrument": "instrument-quality-spatial-null-v3",
    "source_localization": "pixel-attribution-v2",
    "contamination": "catalogue-neighbour-v6",
    "identity_catalogues": "catalogue-signal-identity-v2",
    "orbital": "orbital-eb-v2",
}

GATE_ORDER = ("shape", "orbital", "instrument", "localization", "contamination")
GATE_DECISIONS = ("PASS", "DISMISS", "CONTINUE_INCONCLUSIVE")
TRIAGE_CONTRACT = {
    "id": "ordered-first-failure-triage-v1",
    "stage_order": list(GATE_ORDER),
    "allowed_decisions": {
        stage: (
            ["PASS", "CONTINUE_INCONCLUSIVE"]
            if stage == "instrument" else list(GATE_DECISIONS)
        )
        for stage in GATE_ORDER
    },
    "decision_states": {
        "PASS": "ACTIVE", "DISMISS": "DISMISSED", "CONTINUE_INCONCLUSIVE": "HELD",
    },
    "state_rule": "FIRST_NONPASS_IN_STAGE_ORDER",
    "execution_modes": ["SKIP_AFTER_FIRST_FAILURE", "RECORD_EVERY_GATE"],
    "record_mode": "Later measurements preserve the first failure stage, state and reason",
}


def validate_triage_contract(policy):
    if policy.get("triage_contract") != TRIAGE_CONTRACT:
        raise ValueError("Policy triage contract differs from the implemented gates")


def validate_gate_result(stage, result):
    if result.get("decision") not in TRIAGE_CONTRACT["allowed_decisions"][stage]:
        raise ValueError(
            f"Unknown triage gate decision for {stage}: {result.get('decision')}"
        )


def check_gate_state(signal, failed):
    expected = {"status": "ACTIVE"}
    if failed is not None:
        dismissed = failed["decision"] == "DISMISS"
        expected = {
            "status": TRIAGE_CONTRACT["decision_states"][failed["decision"]],
            "dismissed_stage" if dismissed else "held_stage": failed["stage"],
            "reason": failed.get("reason"),
        }
    triage = signal["triage"]
    fields = ("status", "held_stage", "dismissed_stage", "reason")
    if any(triage.get(field) != expected.get(field) for field in fields):
        raise ValueError(
            f"The triage gate state differs from its first failure: {signal['signal_id']}"
        )


def check_gate_record(signal):
    """Require the declared gate sequence and the state of its first failure."""
    triage = signal.get("triage", {})
    history = triage.get("history", [])
    stages = tuple(item.get("stage") for item in history)
    if stages != GATE_ORDER[: len(history)]:
        raise ValueError(
            f"The triage record differs from the gate order: {signal['signal_id']}"
        )
    for item in history:
        validate_gate_result(item["stage"], item)
    failed = next((item for item in history if item["decision"] != "PASS"), None)
    expected = GATE_ORDER
    if failed is not None and triage.get("record_every_gate") is not True:
        expected = GATE_ORDER[: GATE_ORDER.index(failed["stage"]) + 1]
    if stages != expected:
        raise ValueError(
            f"The triage record differs from the gate order: {signal['signal_id']}"
        )
    check_gate_state(signal, failed)


def first_failed_gate(signal):
    check_gate_record(signal)
    history = signal["triage"]["history"]
    return next((item for item in history if item["decision"] != "PASS"), None)


def validate_candidate_assessment(review):
    for name in ASSESSMENTS:
        assessment = review.get("assessments", {}).get(name, {})
        if assessment.get("availability") not in {
            "AVAILABLE",
            "PARTIAL",
            "UNAVAILABLE",
        }:
            raise ValueError(
                f"Candidate review must record evidence availability: {name}"
            )
        if not assessment.get("interpretation") or not assessment.get("evidence"):
            raise ValueError(
                f"Candidate review must cite and interpret the evidence: {name}"
            )


def validate_review(review, event_ids):
    allowed = {
        "TRANSIT_LIKE",
        "BOI",
        "FALSE_POSITIVE",
        "FALSE_ALARM",
        "UNRESOLVED",
    }
    if review["disposition"] not in allowed:
        raise ValueError("Unknown review disposition")
    if not all(review.get(key) for key in ("reviewer", "rationale", "evidence")):
        raise ValueError("A disposition requires an attributed, evidence-backed review")
    if not review.get("event_ids") or not set(review["event_ids"]) <= event_ids:
        raise ValueError("A review must identify existing detector-proposed events")
    if review["disposition"] == "BOI":
        validate_candidate_assessment(review)
    if review["disposition"] == "FALSE_POSITIVE" and not review.get(
        "reproduced_mechanism"
    ):
        raise ValueError("False-positive denial requires a reproduced mechanism")


def adjudicate(evidence):
    event_ids = {event["event_id"] for event in evidence["events"]}
    current = {}
    for review in evidence.get("reviews", []):
        validate_review(review, event_ids)
        current.update(dict.fromkeys(review["event_ids"], review))
    result = {
        "status": "UNRESOLVED",
        "method": "Automatic assessment with attributed review annotations",
        "reviewed_event_ids": sorted(current),
        "unreviewed_event_ids": sorted(event_ids - current.keys()),
        "boi_signal_count": 0,
        "new_boi_signal_count": 0,
        "established_new_boi_count": 0,
        "bois_with_unresolved_novelty": 0,
        "count_status": "EXPLICIT_SIGNAL_IDENTITIES",
        "attributed_signal_reviews": list(evidence.get("reviews", [])),
        "automatic_classification": "BOI_CATALOGUE",
        "statistically_validated": False,
        "confirmed": False,
        "retained": True,
        "scope": POLICY["scope"],
    }
    if evidence.get("signals") is not None:
        policy = evidence.get("disposition_policy")
        for index, signal in enumerate(evidence["signals"]):
            signal["signal_assessments"] = automatic_signal_assessments(
                evidence, signal, index, policy
            )
        result = signal_dispositions(evidence, result)
    return result


def measured_bic(model):
    value = model.get("bic")
    return value if value is not None and math.isfinite(value) else math.inf


def shape_evidence(fit):
    selection = fit.get("selection", {})
    if (
        selection.get("require_complete_membership")
        and fit.get("status") == "INCOMPLETE_EVENT_MEMBERSHIP"
    ):
        return {"status": "UNRESOLVED", "reason": "SHAPE_EVENT_MEMBERSHIP_INCOMPLETE"}
    required = {
        "linear",
        "quadratic",
        "step",
        "dropout",
        "single_cadence",
        "shared_single_cadence",
        "signed_box",
        "sinusoid",
    }
    models = fit.get("alternatives", {})
    if not required <= models.keys():
        return {"status": "UNRESOLVED", "reason": "SHAPE_ALTERNATIVES_INCOMPLETE"}
    physical = measured_bic(fit) if fit.get("status") == "FIT_COMPLETE" else math.inf
    box = models.get("signed_box", {})
    box_bic = (
        measured_bic(box)
        if "transit_like_box" in box.get("equivalent_to", [])
        else math.inf
    )
    transit_name, transit = min(
        (("physical_transit", physical), ("transit_like_box", box_bic)),
        key=lambda item: item[1],
    )
    model_bics = {name: measured_bic(value) for name, value in models.items()}
    if not all(math.isfinite(model_bics.get(name, math.inf)) for name in required):
        return {"status": "UNRESOLVED", "reason": "SHAPE_EVIDENCE_UNCONSTRAINED"}
    other = {
        name: value
        for name, value in model_bics.items()
        if name != "signed_box" and math.isfinite(value)
    }
    if not math.isfinite(transit):
        return {"status": "UNRESOLVED", "reason": "SHAPE_EVIDENCE_UNCONSTRAINED"}
    best = min(other, key=other.get)
    delta_bic = other[best] - transit
    if delta_bic == 0:
        status = "NON_IDENTIFIABLE"
    elif delta_bic > 0:
        status = "TRANSIT_COMPATIBLE"
    else:
        status = "ALTERNATIVE_PREFERRED"
    return {
        "status": status,
        "transit_model": transit_name,
        "transit_bic": transit,
        "alternative": best,
        "alternative_bic": other[best],
        "alternative_equivalent_to": models[best].get("equivalent_to", []),
        "delta_bic": delta_bic,
        "reason": "Physical-transit preference over explicit nuisance models; no universal significance claim",
    }


def timing_shape_score(result, original_score):
    refinement = result.get("timing_refinement", {})
    if refinement.get("status") != "SELECTED":
        return original_score, None
    view = result.get("hypothesis", {}).get("discovery_view", "psf_flat")
    refined = shape_evidence(
        refinement.get("views", {}).get(view, {}).get("joint_shape", {})
    )
    penalty = refinement.get("timing_bic_penalty")
    if refined.get("delta_bic") is None or penalty is None:
        return original_score, None
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Timing refinement requires a finite nonnegative BIC penalty")
    adjusted = refined["delta_bic"] - penalty
    return max(original_score, adjusted), {
        "raw_shape": refined,
        "timing_bic_penalty": penalty,
        "adjusted_score": adjusted,
        "timing": refinement["timing"],
    }


def joint_shape_records(result):
    records = [
        {"view": view, **shape_evidence(data["joint_shape"])}
        for view, data in result.get("views", {}).items()
    ]
    view = canonical_view(records, result.get("hypothesis", {}).get("discovery_view"))
    selected = [record for record in records if record["view"] == view]
    if len(selected) != 1:
        return records
    original = selected[0]
    score, refinement = timing_shape_score(result, original["delta_bic"])
    if refinement is not None and score > original["delta_bic"]:
        updated = {
            **refinement["raw_shape"],
            "view": view,
            "status": "NON_IDENTIFIABLE" if score == 0 else (
                "TRANSIT_COMPATIBLE" if score > 0 else "ALTERNATIVE_PREFERRED"
            ),
            "transit_bic": refinement["raw_shape"]["transit_bic"]
            + refinement["timing_bic_penalty"],
            "delta_bic": score,
            "original_shape": dict(original),
            "timing_refinement": refinement,
        }
        original.clear()
        original.update(updated)
    return records


def model_comparisons(row, signal):
    records = [
        {
            "event_id": event["event_id"],
            "view": event["photometry"],
            **shape_evidence(event["fit"]),
        }
        for event in row["events"]
        if event["event_id"] in signal["event_ids"]
    ]
    for hypothesis in signal.get("timing_hypotheses", []):
        result = (
            row.get("orbital", {})
            .get("hypotheses", {})
            .get(hypothesis["hypothesis_id"], {})
        )
        records.extend(
            {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "timing_basis": hypothesis.get("timing_basis", "ORIGINAL_PROPOSAL"),
                "unit_ids": hypothesis.get("unit_ids", []),
                **record,
            }
            for record in joint_shape_records(result)
        )
    return records


def usable_measurement(record):
    score = record.get("delta_bic")
    return score is not None and math.isfinite(score)


def canonical_view(records, discovery_view=None):
    views = {record["view"] for record in records if usable_measurement(record)}
    return next(
        (
            view
            for view in (
                (discovery_view,)
                if discovery_view is not None
                else ("psf_flat", "psf_qdiag", "psf", "sap_qdiag", "sap")
            )
            if view in views
        ),
        None,
    )


def measurement_identity(record):
    return (
        record.get("view"),
        record.get("timing_basis"),
        record.get("hypothesis_id"),
        record.get("event_id"),
        tuple(sorted(record.get("unit_ids", []))),
    )


def canonical_signal_records(signal, records):
    repeated = len(signal.get("events", [])) > 1
    required_units = {unit["unit_id"] for unit in signal.get("events", [])}
    selected = (
        [
            record
            for record in records
            if "hypothesis_id" in record
            and set(record.get("unit_ids", [])) == required_units
        ]
        if repeated
        else [
            record
            for record in records
            if record.get("event_id") in set(signal.get("event_ids", []))
        ]
    )
    if repeated:
        selected = [
            record
            for record in selected
            if record.get("timing_basis") == "ORIGINAL_PROPOSAL"
        ]
    view = canonical_view(selected, signal.get("discovery_view"))
    chosen = [
        record
        for record in selected
        if record["view"] == view and usable_measurement(record)
    ]
    unique = {}
    for record in chosen:
        identity = measurement_identity(record)
        if identity in unique:
            raise ValueError("Duplicate canonical signal measurement identity")
        unique[identity] = record
    return list(unique.values()), view, repeated


def signal_shape(signal, records):
    selected, view, repeated = canonical_signal_records(signal, records)
    if not selected:
        return {
            "status": "UNRESOLVED",
            "reason": "SIGNAL_SHAPE_UNMEASURED",
            "selection": {"view": view, "repeated": repeated},
            "records": [],
        }
    reference = selected[0]
    if len(selected) != 1:
        return {
            "status": "UNRESOLVED",
            "reason": "MULTIPLE_NON_EQUIVALENT_SIGNAL_MEASUREMENTS",
            "selection": {"view": view, "repeated": repeated},
            "records": selected,
        }
    selection = {
        "view": view,
        "repeated": repeated,
        "timing_basis": reference.get("timing_basis"),
        "rule": "DECLARED_DISCOVERY_VIEW_ORIGINAL_TIMING_JOINT_FOR_REPEATS"
        if "discovery_view" in signal
        else "PSF_BEFORE_SAP_ORIGINAL_TIMING_JOINT_FOR_REPEATS",
    }
    if reference.get("timing_refinement") is not None:
        selection.update(
            measurement="BOUNDED_REFINED_TIMING_SHAPE",
            measurement_view=signal.get("discovery_view", "psf_flat"),
            rule="CANONICAL_JOINT_WITH_PENALIZED_BOUNDED_TIMING",
        )
    return {
        "status": reference["status"],
        "score": reference.get("delta_bic"),
        "selection": selection,
        "records": [reference],
    }


def referenced_evidence(row, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    value = row
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            return None
    return value


def valid_evidence_ref(reference, row):
    value = referenced_evidence(row, reference.get("artifact"))
    if value is None:
        return False
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest() == reference.get("sha256")


def evidence_reference(row, pointer):
    value = referenced_evidence(row, pointer)
    if value is None:
        pointer = "/source_id"
        value = row["source_id"]
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"artifact": pointer, "sha256": hashlib.sha256(encoded).hexdigest()}


def assessment_record(row, signal, name, pointer, status, interpretation, rule_id):
    return {
        "status": status,
        "assessment_rule_id": rule_id,
        "signal_id": signal["signal_id"],
        "event_ids": signal["event_ids"],
        "evidence_refs": [evidence_reference(row, pointer)],
        "interpretation": interpretation,
        "automatic": True,
    }


def assessment_rule(policy, name):
    rule = (policy or {}).get("assessment_rules", {}).get(name)
    active = (policy or {}).get("status") == "RELEASED" or (
        (policy or {}).get("status") == "FROZEN_EVALUATION"
        and (policy or {}).get("_evaluation_authorized") is True
    )
    if (
        not active
        or not isinstance(rule, dict)
        or rule.get("status") != "RELEASED"
        or rule.get("id") != AUTOMATIC_ASSESSMENT_RULES[name]
    ):
        return None
    return rule


def instrument_assessment(row, signal, policy):
    rule = assessment_rule(policy, "instrument")
    spatial_null = signal.get("localization", {}).get("spatial_null", {})
    records = {
        record["event_id"]: record
        for record in row.get("instrument", {}).get("events", [])
    }
    selected = [records.get(event_id) for event_id in signal["event_ids"]]
    status = "UNASSESSED"
    interpretation = (
        "The frozen instrument rule is unavailable or event evidence is incomplete."
    )
    if rule is not None and all(
        record is not None
        and record.get("status") == "MEASURED"
        and record.get("in_event_points", 0) > 0
        and record.get("quarter_time_bounds_bkjd") is not None
        for record in selected
    ):
        shared_available = all(
            record.get("shared_time", {}).get("complete") is True
            for record in selected
        )
        flagged = {
            name
            for record in selected
            for name, count in record.get("flags", {}).items()
            if count and name in set(rule["material_quality_flags"])
        }
        if (
            spatial_null.get("status") == "MEASURED"
            and spatial_null["median_bic_advantage"] > rule["spatial_null_bic_margin"]
        ):
            status = "MATERIAL_CONFLICT"
            interpretation = (
                "A row, column, or uniform pixel artifact is strongly "
                "preferred to the source PRF."
            )
        elif flagged:
            status = "MATERIAL_CONFLICT"
            interpretation = f"Calibrated instrument quality flags: {sorted(flagged)}."
        elif not shared_available:
            interpretation = "Shared-time instrument evidence is unavailable."
        elif spatial_null.get("status") != "MEASURED":
            interpretation = "Pixel-shape instrument evidence is unavailable."
        else:
            status = "ASSESSED_NO_MATERIAL_CONFLICT"
            interpretation = (
                "Measured quality and pixel-shape evidence crosses no frozen instrument "
                "conflict threshold; shared-time counts are contextual, not a veto."
            )
    result = assessment_record(
        row,
        signal,
        "instrument",
        "/instrument",
        status,
        interpretation,
        rule["id"] if rule else "UNRELEASED_INSTRUMENT_RULE",
    )
    if spatial_null.get("status") == "MEASURED":
        index = next(
            index
            for index, item in enumerate(row["signals"])
            if item["signal_id"] == signal["signal_id"]
        )
        result["evidence_refs"].append(
            evidence_reference(row, f"/signals/{index}/localization/spatial_null")
        )
    return result


def localization_assessment(row, signal, policy):
    rule = assessment_rule(policy, "source_localization")
    attribution = signal.get("localization", {}).get("source_attribution", {})
    single = attribution.get("basis") == "SINGLE_QUARTER"
    position = "single-quarter" if single else "robust multi-quarter"
    status = "UNASSESSED"
    interpretation = (
        "The frozen single-quarter localization rule is unavailable, or the event "
        "quarter is not quality-qualified or lacks the required target S/N."
        if single
        else "The frozen robust localization rule is unavailable or has too few "
        "quality-qualified quarters."
    )
    if rule is not None:
        if attribution.get("status") in set(rule["material_statuses"]):
            status = "MATERIAL_CONFLICT"
            interpretation = (
                f"The {position} source position is materially offset "
                "from the propagated target position."
            )
        elif attribution.get("status") in set(rule["accepted_statuses"]):
            status = "ASSESSED_NO_MATERIAL_CONFLICT"
            interpretation = (
                f"The {position} source position is compatible with the "
                "propagated target position; this is not unique host attribution or validation."
            )
    return assessment_record(
        row,
        signal,
        "source_localization",
        "/localization",
        status,
        interpretation,
        rule["id"] if rule else "UNRELEASED_LOCALIZATION_RULE",
    )


def significant_neighbour(record, minimum_sigma):
    depth, error = record.get("depth_fraction"), record.get("depth_error")
    return (
        record.get("status") == "MEASURED"
        and depth is not None
        and error is not None
        and error > 0
        and depth / error >= minimum_sigma
    )


def catalogue_match_identity(match):
    return (
        match.get("catalogue"),
        match.get("source"),
        match.get("known_period_days"),
        round(match.get("separation_arcsec", 0), 6),
    )


def catalogue_match_groups(signal, timing_records, include):
    matched_events = {}
    examples = {}
    for record in timing_records:
        for match in record.get("matches", []):
            overlaps = match.get("overlaps_original_interval_nominally") is True
            if not overlaps or not include(match):
                continue
            identity = catalogue_match_identity(match)
            matched_events.setdefault(identity, set()).add(record["event_id"])
            examples.setdefault(identity, match)
    units = signal.get("events", [])
    groups = []
    for identity, event_ids in matched_events.items():
        matched_units = sum(
            any(event_id in event_ids for event_id in unit["event_ids"])
            for unit in units
        )
        fraction = matched_units / len(units) if units else 0
        known_period = examples[identity].get("known_period_days")
        proposed_period = signal.get("period_days")
        order = None
        error = None
        if (
            known_period
            and proposed_period
            and known_period > 0
            and proposed_period > 0
        ):
            ratio = max(known_period / proposed_period, proposed_period / known_period)
            order = round(ratio)
            error = abs(ratio / order - 1)
        groups.append(
            {
                "catalogue": examples[identity].get("catalogue"),
                "source": examples[identity].get("source"),
                "signal_class": examples[identity].get("signal_class"),
                "exact_identifier_match": examples[identity].get(
                    "exact_identifier_match"
                ),
                "separation_arcsec": examples[identity].get("separation_arcsec"),
                "known_period_days": examples[identity].get("known_period_days"),
                "harmonic_order": order,
                "harmonic_period_error_fraction": error,
                "harmonic_expected_fraction": (
                    1 / order
                    if order and known_period >= proposed_period
                    else 1
                    if order
                    else None
                ),
                "matched_event_units": matched_units,
                "signal_event_units": len(units),
                "matched_fraction": fraction,
            }
        )
    return groups


def calibrated_catalogue_matches(groups, rule):
    return [
        group
        for group in groups
        if (
            group["matched_event_units"] >= rule["minimum_catalogue_match_units"]
            and group["matched_fraction"] >= rule["minimum_catalogue_match_fraction"]
        )
        or harmonic_catalogue_match(group, rule)
    ]


def harmonic_catalogue_match(group, rule):
    order = group.get("harmonic_order")
    expected = group.get("harmonic_expected_fraction")
    return (
        order is not None
        and 1 <= order <= rule["maximum_harmonic_order"]
        and group["harmonic_period_error_fraction"]
        <= rule["maximum_harmonic_period_error_fraction"]
        and group["matched_event_units"] >= rule["minimum_harmonic_match_units"]
        and group["matched_fraction"] / expected
        >= rule["minimum_harmonic_expected_match_fraction"]
    )


def underpowered_catalogue_matches(groups, rule):
    return [
        group
        for group in groups
        if (
            group["matched_event_units"] < rule["minimum_catalogue_match_units"]
            and group["matched_fraction"] >= rule["minimum_catalogue_match_fraction"]
        )
        or (
            group.get("harmonic_order") is not None
            and group["harmonic_order"] <= rule["maximum_harmonic_order"]
            and group["harmonic_period_error_fraction"]
            <= rule["maximum_harmonic_period_error_fraction"]
            and group["matched_event_units"] < rule["minimum_harmonic_match_units"]
            and group["matched_fraction"] / group["harmonic_expected_fraction"]
            >= rule["minimum_harmonic_expected_match_fraction"]
        )
    ]


def catalogue_timing_groups(signal, timing_records, rule):
    return catalogue_match_groups(
        signal,
        timing_records,
        lambda match: (
            match.get("separation_arcsec", 0)
            >= rule["minimum_competing_source_separation_arcsec"]
        ),
    )


def catalogue_timing_conflicts(signal, timing_records, rule):
    return calibrated_catalogue_matches(
        catalogue_timing_groups(signal, timing_records, rule), rule
    )


def catalogue_identity_groups(signal, timing_records, rule):
    material = set(rule["material_signal_classes"])
    return catalogue_match_groups(
        signal,
        timing_records,
        lambda match: (
            match.get("exact_identifier_match") is True
            and match.get("signal_class") in material
        ),
    )


def catalogue_identity_conflicts(signal, timing_records, rule):
    return calibrated_catalogue_matches(
        catalogue_identity_groups(signal, timing_records, rule), rule
    )


def neighbour_comparison_complete(evidence, signal, rule):
    if rule is None:
        return False
    comparisons = evidence.get("neighbour_comparisons", {})
    siblings = comparisons.get("expected_sibling_ids")
    selected = set(comparisons.get("selected_event_ids", [])) & set(
        signal["event_ids"]
    )
    if comparisons.get("status") != "MEASURED" or siblings is None or not selected:
        return False
    expected = {(sid, event_id) for sid in siblings for event_id in selected}
    records = evidence.get("neighbour_flux", [])
    observed = {(r.get("source_id"), r.get("event_id")) for r in records}
    return (
        len(records) == len(expected)
        and observed == expected
        and all(
            record.get("status") in set(rule["accepted_neighbour_statuses"])
            for record in records
        )
    )


def target_prf_source_supported(signal, rule):
    localization = signal.get("localization", {})
    attribution = localization.get("source_attribution", {})
    prf = localization.get("scene_prf", {})
    compared = prf.get("compared_quarters")
    preferred = prf.get("target_preferred_quarters")
    return (
        attribution.get("status") == "TARGET_COMPATIBLE"
        and prf.get("status") == "MEASURED_DIAGNOSTIC"
        and type(compared) is int
        and compared >= rule["minimum_scene_prf_compared_quarters"]
        and type(preferred) is int
        and 0 <= preferred <= compared
        and 2 * preferred > compared
    )


def contamination_assessment(row, signal, signal_index, policy):
    rule = assessment_rule(policy, "contamination")
    evidence = signal.get("contamination", {})
    timing_records = evidence.get("catalogue_timing", [])
    timing_conflicts = (
        catalogue_timing_conflicts(signal, timing_records, rule)
        if rule is not None
        else []
    )
    underpowered_timing = (
        underpowered_catalogue_matches(
            catalogue_timing_groups(signal, timing_records, rule), rule
        )
        if rule is not None
        else []
    )
    neighbours = [
        record
        for record in evidence.get("neighbour_flux", [])
        if significant_neighbour(
            record, rule["minimum_neighbour_depth_sigma"] if rule else math.inf
        )
    ]
    neighbour_coverage = neighbour_comparison_complete(evidence, signal, rule)
    target_prf_support = (
        target_prf_source_supported(signal, rule) if rule is not None else False
    )
    timing_coverage = {record.get("event_id") for record in timing_records} == set(
        signal["event_ids"]
    )
    status = "UNASSESSED"
    interpretation = (
        "The frozen contamination rule is unavailable or scene evidence is incomplete."
    )
    if rule is not None:
        if evidence.get("mechanism_status") == "ESTABLISHED":
            status = "MATERIAL_CONFLICT"
            interpretation = (
                "A competing-source transfer mechanism is established."
            )
        elif timing_conflicts:
            interpretation = (
                "Coherent nearby catalogue timing requires source attribution; "
                "timing coincidence alone does not establish flux transfer."
            )
        elif neighbours:
            interpretation = (
                "Coincident sibling deficits need source attribution; shared target "
                "photons can dim multiple extractions."
            )
        elif underpowered_timing:
            interpretation = (
                "A catalogue source matches too few event units to establish or "
                "exclude a competing-source mechanism."
            )
        elif (
            evidence.get("status") == "EVIDENCE_ASSEMBLED"
            and timing_coverage
            and (neighbour_coverage or target_prf_support)
            and evidence.get("scene", {}).get("status")
            in set(rule["accepted_scene_statuses"])
        ):
            status = "ASSESSED_NO_MATERIAL_CONFLICT"
            interpretation = (
                "Complete sibling measurements and catalogue timing show no material "
                "competing-source conflict; uncatalogued blends remain possible."
                if neighbour_coverage
                else (
                    "Quality-qualified target pixels favor the target PRF over "
                    "modelled Gaia alternatives in a strict majority of compared "
                    "quarters. Sibling extraction evidence remains incomplete, and "
                    "uncatalogued blends remain possible."
                )
            )
    result = assessment_record(
        row,
        signal,
        "contamination",
        f"/signals/{signal_index}/contamination",
        status,
        interpretation,
        rule["id"] if rule else "UNRELEASED_CONTAMINATION_RULE",
    )
    if status == "ASSESSED_NO_MATERIAL_CONFLICT" and not neighbour_coverage:
        result["evidence_refs"].append(
            evidence_reference(row, f"/signals/{signal_index}/localization/scene_prf")
        )
    return result


def identity_catalogue_assessment(row, signal, signal_index, policy):
    rule = assessment_rule(policy, "identity_catalogues")
    evidence = signal.get("contamination", {})
    timing_records = evidence.get("catalogue_timing", [])
    conflicts = (
        catalogue_identity_conflicts(signal, timing_records, rule)
        if rule is not None
        else []
    )
    underpowered = (
        underpowered_catalogue_matches(
            catalogue_identity_groups(signal, timing_records, rule), rule
        )
        if rule is not None
        else []
    )
    timing_coverage = {record.get("event_id") for record in timing_records} == set(
        signal["event_ids"]
    )
    status = "UNASSESSED"
    interpretation = (
        "The frozen exact-signal catalogue rule is unavailable or catalogue "
        "timing coverage is incomplete."
    )
    if rule is not None:
        if conflicts:
            status = "MATERIAL_CONFLICT"
            interpretation = (
                "The signal coherently matches an exact-identifier catalogue "
                f"false class: matches={len(conflicts)}."
            )
        elif underpowered:
            interpretation = (
                "An exact-identifier false class matches too few event units to "
                "establish or exclude signal identity."
            )
        elif (
            evidence.get("status") == "EVIDENCE_ASSEMBLED"
            and timing_coverage
            and evidence.get("catalogue_status")
            in set(rule["accepted_catalogue_statuses"])
        ):
            status = "ASSESSED_NO_MATERIAL_CONFLICT"
            interpretation = (
                "No exact-identifier false-class ephemeris explains the required "
                "fraction of event units; catalogue absence is not a novelty claim."
            )
    return assessment_record(
        row,
        signal,
        "identity_catalogues",
        f"/signals/{signal_index}/contamination",
        status,
        interpretation,
        rule["id"] if rule else "UNRELEASED_CATALOGUE_IDENTITY_RULE",
    )


def orbital_assessment(row, signal, policy):
    rule = assessment_rule(policy, "orbital")
    original_hypotheses = [
        row.get("orbital", {}).get("hypotheses", {}).get(item["hypothesis_id"], {})
        for item in signal.get("timing_hypotheses", [])
        if item.get("timing_basis") == "ORIGINAL_PROPOSAL"
    ]
    hypotheses = list(original_hypotheses)
    hypotheses.extend(
        hypothesis["timing_refinement"]
        for hypothesis in original_hypotheses
        if hypothesis.get("timing_refinement", {}).get("status") == "SELECTED"
    )
    canonical = []
    for hypothesis in hypotheses:
        views = hypothesis.get("views", {})
        name = next(
            (
                view
                for view in (
                    (signal["discovery_view"],)
                    if "discovery_view" in signal
                    else ("psf_flat", "psf_qdiag", "psf", "sap_qdiag", "sap")
                )
                if view in views
            ),
            None,
        )
        if name is not None:
            canonical.append((name, views[name]))
    gains = [
        view.get("odd_even", {}).get("two_depth_bic_gain") for _, view in canonical
    ]
    gains = [value for value in gains if value is not None and math.isfinite(value)]
    secondary_conflicts = []
    secondary_complete = True
    modulation_conflicts = []
    modulation_complete = True
    modulation_evidence = []
    for view_name, view in canonical:
        primary = view.get("primary_depth", {})
        secondary = view.get("secondary_phase_half", {})
        depth = secondary.get("depth_fraction")
        error = secondary.get("depth_error")
        primary_depth = primary.get("depth_fraction")
        secondary_complete &= (
            secondary.get("status") == "MEASURED"
            and depth is not None
            and error is not None
            and error > 0
            and primary.get("status") == "MEASURED"
            and primary_depth is not None
            and primary_depth > 0
        )
        if (
            rule is not None
            and secondary.get("status") == "MEASURED"
            and depth is not None
            and error is not None
            and error > 0
            and primary_depth is not None
            and primary_depth > 0
            and depth / error >= rule["minimum_secondary_sigma"]
            and depth / primary_depth > rule["maximum_secondary_to_primary_depth_ratio"]
        ):
            secondary_conflicts.append(depth / primary_depth)
        modulation = view.get("phase_modulation", {})
        strongest = modulation.get("strongest_signal") or {}
        modulation_complete &= modulation.get("status") == "MEASURED" and all(
            value is not None and math.isfinite(value)
            for value in (
                modulation.get("tce_period_days"),
                strongest.get("amplitude_fraction"),
                strongest.get("signal_to_noise"),
            )
        )
        if rule is not None and strongest:
            amplitude = strongest.get("amplitude_fraction")
            signal_to_noise = strongest.get("signal_to_noise")
            period = modulation.get("tce_period_days")
            conflict = (
                period is not None
                and period < rule["maximum_phase_modulation_period_days"]
                and amplitude is not None
                and amplitude > rule["minimum_phase_modulation_amplitude_fraction"]
                and primary_depth is not None
                and primary_depth > 0
                and amplitude
                < rule["maximum_phase_modulation_to_primary_depth_ratio"]
                * primary_depth
                and signal_to_noise is not None
                and signal_to_noise > rule["minimum_phase_modulation_signal_to_noise"]
            )
            modulation_evidence.append(
                {
                    "view": view_name,
                    "tce_period_days": period,
                    "fitted_period_ratio": strongest.get("period_ratio"),
                    "amplitude_fraction": amplitude,
                    "primary_depth_fraction": primary_depth,
                    "signal_to_noise": signal_to_noise,
                    "coherent_conflict": conflict,
                }
            )
            if conflict:
                modulation_conflicts.append(modulation_evidence[-1])
    status = "UNASSESSED"
    interpretation = "The frozen orbital rule is unavailable or repeated-event evidence is incomplete."
    if (
        rule is not None
        and hypotheses
        and gains
        and len(gains) == len(canonical) == len(hypotheses)
        and secondary_complete
        and modulation_complete
    ):
        if (
            max(gains) > rule["maximum_odd_even_bic_gain"]
            or secondary_conflicts
            or modulation_conflicts
        ):
            status = "MATERIAL_CONFLICT"
            interpretation = (
                f"Stellar-eclipse conflict: maximum odd/even BIC gain={max(gains):.6g}, "
                f"secondary/primary ratios={secondary_conflicts}, "
                f"phase-modulation evidence={modulation_evidence}."
            )
        else:
            status = "ASSESSED_NO_MATERIAL_CONFLICT"
            interpretation = (
                "The assessed original and selected refined timing hypotheses cross no frozen "
                "odd/even, secondary or phase-modulation conflict threshold."
            )
    return assessment_record(
        row,
        signal,
        "orbital",
        "/orbital",
        status,
        interpretation,
        rule["id"] if rule else "UNRELEASED_ORBITAL_RULE",
    )


def automatic_signal_assessments(row, signal, signal_index, policy):
    result = {
        "instrument": instrument_assessment(row, signal, policy),
        "source_localization": localization_assessment(row, signal, policy),
        "contamination": contamination_assessment(row, signal, signal_index, policy),
        "identity_catalogues": identity_catalogue_assessment(
            row, signal, signal_index, policy
        ),
    }
    if len(signal.get("events", [])) > 1:
        result["orbital"] = orbital_assessment(row, signal, policy)
    return result


def valid_assessment(name, assessment, signal, row):
    status = assessment.get("status")
    return (
        status
        in {
            "ASSESSED_NO_MATERIAL_CONFLICT",
            "MATERIAL_CONFLICT",
            "UNASSESSED",
            "NOT_APPLICABLE",
        }
        and (status != "NOT_APPLICABLE" or name == "orbital")
        and bool(assessment.get("assessment_rule_id"))
        and assessment.get("signal_id") == signal["signal_id"]
        and set(assessment.get("event_ids", [])) == set(signal["event_ids"])
        and bool(assessment.get("evidence_refs"))
        and all(valid_evidence_ref(ref, row) for ref in assessment["evidence_refs"])
        and bool(assessment.get("interpretation"))
    )


def evidence_assessment(row, signal):
    required = {
        "instrument",
        "source_localization",
        "contamination",
        "identity_catalogues",
    }
    if len(signal.get("events", [])) > 1:
        required.add("orbital")
    supplied = signal.get("signal_assessments", {})
    assessments = {}
    for name in sorted(required):
        assessment = supplied.get(name, {})
        assessments[name] = (
            assessment
            if valid_assessment(name, assessment, signal, row)
            else {
                "status": "UNASSESSED",
                "reason": "MISSING_OR_MALFORMED_EXPLICIT_ASSESSMENT",
            }
        )
    statuses = {assessment["status"] for assessment in assessments.values()}
    status = (
        "MATERIAL_CONFLICT"
        if "MATERIAL_CONFLICT" in statuses
        else (
            "UNASSESSED"
            if "UNASSESSED" in statuses
            else "ASSESSED_NO_MATERIAL_CONFLICT"
        )
    )
    return {"status": status, "assessments": assessments}


def automatic_signal(row, signal):
    records = model_comparisons(row, signal)
    shape = signal_shape(signal, records)
    assessment = evidence_assessment(row, signal)
    mechanisms = [
        m
        for m in row.get("reproduced_mechanisms", [])
        if set(signal["event_ids"]) <= set(m.get("event_ids", []))
        and m.get("status") == "REPRODUCED"
        and m.get("evidence")
        and m.get("interpretation")
    ]
    status, reasons = triage_disposition(signal, bool(mechanisms))
    policy = row.get("disposition_policy") or {}
    if policy.get("status") == "FROZEN_EVALUATION" and not policy.get("_evaluation_authorized"):
        status, reasons = "UNRESOLVED", ["EVALUATION_NOT_AUTHORIZED"]
    return {
        "signal_id": signal["signal_id"],
        "event_ids": signal["event_ids"],
        "disposition": status,
        "reason_codes": reasons,
        "model_comparisons": records,
        "signal_shape": shape,
        "evidence_assessment": assessment,
        "reproduced_mechanisms": mechanisms,
        "policy": POLICY,
        "evidence_availability": {
            name: row.get(name, {}).get("status", "UNAVAILABLE")
            for name in ("association", "orbital", "instrument", "localization")
        },
        "source_association": signal.get("localization", {"status": "UNAVAILABLE"}),
        "competing_explanations": signal.get(
            "contamination", {"status": "UNAVAILABLE"}
        ),
        "host": signal.get("host", {"status": "UNAVAILABLE"}),
        "risk_features": risk_features(row, signal),
    }


def triage_disposition(signal, reproduced_mechanism=False):
    if reproduced_mechanism:
        return "FALSE_POSITIVE", ["REPRODUCED_ASTROPHYSICAL_MECHANISM"]
    history = signal.get("triage", {}).get("history", [])
    stages = tuple(item.get("stage") for item in history)
    if stages != GATE_ORDER[:len(history)]:
        raise ValueError("Signal triage does not follow the declared gate order")
    if len(history) < len(GATE_ORDER) and all(
        item.get("decision") == "PASS" for item in history
    ):
        return "UNRESOLVED", ["TRIAGE_INCOMPLETE"]
    failed = first_failed_gate(signal)
    if failed is None:
        return "BOI", ["ALL_PIPELINE_GATES_PASS"]
    return signal["triage"]["status"], [failed["reason"]]


def risk_features(row, signal):
    odd_even, target_offsets, relative_offsets = [], [], []
    for hypothesis in signal.get("timing_hypotheses", []):
        record = (
            row.get("orbital", {})
            .get("hypotheses", {})
            .get(hypothesis["hypothesis_id"], {})
        )
        odd_even.extend(
            v["odd_even"].get("two_depth_bic_gain")
            for v in record.get("views", {}).values()
        )
    for pixel in row.get("localization", {}).get("events", []):
        if pixel["event_id"] not in signal["event_ids"]:
            continue
        attribution = pixel.get("attribution", {})
        target = attribution.get("target_offset_sigma")
        neighbours = [
            n["offset_sigma"]
            for n in attribution.get("neighbours", [])
            if n.get("offset_sigma") is not None
        ]
        if target is not None:
            target_offsets.append(target)
        if target is not None and neighbours:
            relative_offsets.append(target**2 - min(neighbours) ** 2)
    return {
        name: max((v for v in values if v is not None), default=None)
        for name, values in (
            ("odd_even", odd_even),
            ("target_offset_sigma", target_offsets),
            ("relative_source_offset", relative_offsets),
        )
    }


def candidate_event_windows(signal):
    windows = []
    for event in signal.get("events", []):
        start, end = event.get("start_bkjd"), event.get("end_bkjd")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end < start
        ):
            return None
        windows.append((start, end))
    return sorted(windows)


def overlapping_candidate_hypotheses(signals):
    periodic = [signal for signal in signals if len(signal.get("events", [])) > 1]
    if len(periodic) < 2:
        return False
    windows = [candidate_event_windows(signal) for signal in periodic]
    if any(items is None for items in windows):
        return True
    for index, left in enumerate(windows):
        for right in windows[index + 1 :]:
            smaller, larger = sorted((left, right), key=len)
            position = 0
            overlapping = 0
            for start, end in smaller:
                while position < len(larger) and larger[position][1] < start:
                    position += 1
                overlapping += position < len(larger) and larger[position][0] <= end
            if overlapping >= 0.95 * len(smaller):
                return True
    return False


def annotated_signal_disposition(row, signal):
    assigned = automatic_signal(row, signal)
    reviews = [
        review for review in row.get("reviews", [])
        if set(review["event_ids"]).intersection(signal["event_ids"])
    ]
    if reviews:
        assigned["reviews"] = reviews
    return assigned


def signal_dispositions(row, base):
    dispositions = [annotated_signal_disposition(row, signal) for signal in row["signals"]]
    candidate_ids = {
        record["signal_id"] for record in dispositions
        if record["disposition"] == "BOI"
    }
    novelty = [
        novelty_status(row.get("novelty_review", {}).get(sid, {}))
        for sid in candidate_ids
    ]
    labels = {r["disposition"] for r in dispositions}
    status = (
        "BOI"
        if candidate_ids
        else (next(iter(labels)) if len(labels) == 1 else "UNRESOLVED")
    )
    two_event_uncertain = any(
        len(s["events"]) == 2
        and s["association_status"] == "TIMING_COMPATIBLE_REPEAT_HYPOTHESIS"
        and s["signal_id"] in candidate_ids
        for s in row["signals"]
    )
    overlapping = overlapping_candidate_hypotheses(
        [signal for signal in row["signals"] if signal["signal_id"] in candidate_ids]
    )
    if two_event_uncertain:
        count_status = "TWO_EVENT_MEMBERSHIP_UNRESOLVED"
    elif overlapping:
        count_status = "OVERLAPPING_CANDIDATE_HYPOTHESES"
    else:
        count_status = "EXPLICIT_SIGNAL_IDENTITIES"
    tentative = count_status != "EXPLICIT_SIGNAL_IDENTITIES"
    return {
        **base,
        "status": status,
        "signals": dispositions,
        "attributed_signal_reviews": list(row.get("reviews", [])),
        "method": "Automatic detector-selected end-to-end policy; attributed reviews retained for audit",
        "automatic_classification": "BOI_CATALOGUE",
        "policy_id": POLICY["id"],
        "boi_signal_count": None if tentative else len(candidate_ids),
        "boi_hypothesis_count": len(candidate_ids),
        "count_status": count_status,
        "new_boi_signal_count": None
        if tentative or "UNRESOLVED" in novelty
        else novelty.count("NEW"),
        "bois_with_unresolved_novelty": novelty.count("UNRESOLVED"),
        "established_new_boi_count": novelty.count("NEW"),
        "unresolved_signals": sum(
            r["disposition"] == "UNRESOLVED" for r in dispositions
        ),
        "scope": POLICY["scope"],
    }


def novelty_complete(review):
    checks = (
        "toi",
        "ctoi",
        "tic",
        "gaia",
        "nasa",
        "exofop",
        "ads_scix",
        "collier_raven",
    )
    return bool(
        review.get("status") == "NO_PRIOR_SIGNAL_IN_QUERIED_SOURCES"
        and review.get("reviewer")
        and review.get("source_identity_confirmed")
        and review.get("evidence")
        and all(review.get("checks", {}).get(key) for key in checks)
    )


def novelty_status(review):
    if novelty_complete(review):
        return "NEW"
    if (
        review.get("status") == "PREVIOUSLY_REPORTED_SIGNAL"
        and review.get("reviewer")
        and review.get("source_identity_confirmed")
        and review.get("evidence")
    ):
        return "KNOWN"
    return "UNRESOLVED"


if __name__ == "__main__":
    from vetting import finish

    finish()
