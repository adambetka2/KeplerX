"""Associate supplied event intervals; never search the light curve for new events."""

import hashlib
from itertools import pairwise

import numpy as np


def identity(prefix, ids):
    return prefix + hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]


def measured_timing(events):
    fitted = [
        e
        for e in events
        if e["fit"].get("status") == "FIT_COMPLETE"
        and e["photometry"] == e["original_alert"]["photometry"]
    ]
    if not fitted:
        return {}

    def support(event):
        fit = event["fit"]
        return fit.get("alternatives", {}).get("linear", {}).get(
            "bic", -np.inf
        ) - fit.get("bic", np.inf)

    event = max(fitted, key=support)
    return {
        "centre_bkjd": event["fit"]["centre_bkjd"],
        "duration_days": event["fit"]["duration_days"],
        "timing_measurement_event_id": event["event_id"],
        "timing_basis": "CHARACTERIZED_PROPOSAL",
        "timing_scope": "Use the best-supported original-extraction transit fit for timing, retaining all alternative measurements; detector intervals and scores remain immutable.",
    }


def event_units(events):
    ordered = sorted(events, key=lambda e: e["original_alert"]["start_bkjd"])
    groups = []
    for event in ordered:
        alert = event["original_alert"]
        if (
            groups
            and alert["quarter"] == groups[-1]["quarter"]
            and alert["start_bkjd"] < groups[-1]["end_bkjd"]
        ):
            group = groups[-1]
            group["end_bkjd"] = max(group["end_bkjd"], alert["end_bkjd"])
            group["events"].append(event)
        else:
            groups.append(
                {
                    "quarter": alert["quarter"],
                    "start_bkjd": alert["start_bkjd"],
                    "end_bkjd": alert["end_bkjd"],
                    "events": [event],
                }
            )
    for group in groups:
        events = group.pop("events")
        group["event_ids"] = sorted({e["event_id"] for e in events})
        group["unit_id"] = identity("event-", group["event_ids"])
        group["centre_bkjd"] = (group["start_bkjd"] + group["end_bkjd"]) / 2
        group["duration_days"] = group["end_bkjd"] - group["start_bkjd"]
        group["timing_basis"] = "ORIGINAL_PROPOSAL"
        group["characterized_timing"] = measured_timing(events)
    return groups


def evenly_spaced(items, limit):
    if limit == 0:
        return []
    if len(items) <= limit:
        return list(items)
    indices = [int((index + 0.5) * len(items) / limit) for index in range(limit)]
    return [items[min(index, len(items) - 1)] for index in indices]


def representative_units(units, limit):
    if limit < 1:
        raise ValueError("Representative event-unit limit must be positive")
    ordered = sorted(units, key=lambda unit: unit["centre_bkjd"])
    if len(ordered) <= limit:
        return ordered
    quarters = {}
    for unit in ordered:
        quarters.setdefault(unit["quarter"], []).append(unit)
    per_quarter = [values[len(values) // 2] for _, values in sorted(quarters.items())]
    selected = evenly_spaced(per_quarter, limit)
    selected_ids = {unit["unit_id"] for unit in selected}
    remaining = [unit for unit in ordered if unit["unit_id"] not in selected_ids]
    selected.extend(evenly_spaced(remaining, max(limit - len(selected), 0)))
    return sorted(selected, key=lambda unit: unit["centre_bkjd"])


def representative_event_ids(signals, limit):
    return {
        event_id
        for signal in signals
        for unit in representative_units(signal["events"], limit)
        for event_id in unit["event_ids"]
    }


def timing_fit(units, period, epoch):
    time = np.array([u["centre_bkjd"] for u in units])
    half = np.array([u["duration_days"] / 2 for u in units])
    cycles = np.rint((time - epoch) / period).astype(int)
    use = abs(time - epoch - cycles * period) <= half
    if use.sum() < 2 or len(np.unique(cycles[use])) != use.sum():
        return None
    design = np.column_stack([np.ones(use.sum()), cycles[use]])
    coefficients = np.linalg.lstsq(
        design / half[use, None], time[use] / half[use], rcond=None
    )[0]
    residual = time[use] - design @ coefficients
    if np.any(abs(residual) > half[use]):
        return None
    return {
        "unit_ids": [u["unit_id"] for u, take in zip(units, use) if take],
        "cycles": (cycles[use] - cycles[use].min()).tolist(),
        "epoch_bkjd": float(coefficients[0] + cycles[use].min() * coefficients[1]),
        "period_days": float(coefficients[1]),
        "timing_residual_days": residual.tolist(),
        "timing_uncertainty": "Characterized durations, or original intervals when no fit is available, bound this timing hypothesis; no Gaussian timing error inferred",
    }


def timing_units(units, basis):
    if basis == "ORIGINAL_PROPOSAL":
        return units
    return [dict(u, **u.get("characterized_timing", {})) for u in units]


def basis_hypotheses(units, basis):
    records, seen = [], set()
    for first, second in pairwise(units):
        period = second["centre_bkjd"] - first["centre_bkjd"]
        if period <= 0:
            continue
        fit = timing_fit(units, period, first["centre_bkjd"])
        if fit is None:
            continue
        key = tuple(fit["unit_ids"]), tuple(fit["cycles"])
        if key not in seen:
            seen.add(key)
            records.append(
                dict(
                    fit,
                    hypothesis_id=identity("timing-", [basis, str(key)]),
                    timing_basis=basis,
                    unresolved_aliases="Observed spacings permit P/n if intervening events were missed; no integer-cycle scan performed",
                )
            )
    return records


def period_hypotheses(units, config):
    records = basis_hypotheses(units, "ORIGINAL_PROPOSAL")
    if any(u["characterized_timing"] for u in units):
        characterized = timing_units(units, "CHARACTERIZED_PROPOSAL")
        characterized.sort(key=lambda u: u["centre_bkjd"])
        records.extend(basis_hypotheses(characterized, "CHARACTERIZED_PROPOSAL"))
    return records, {
        "scope": "Adjacent spacings of original proposals and characterized events separately seed hypotheses. Weak local fits cannot overwrite original timing evidence. No period grid, integer-cycle enumeration or trial budget; aliases and TTV interpretations remain possible."
    }


def repeat_groups(units, hypotheses):
    memberships = {frozenset(h["unit_ids"]) for h in hypotheses}
    maximal = [m for m in memberships if not any(m < other for other in memberships)]
    unambiguous = [
        m for m in maximal if not any(m != other and m & other for other in maximal)
    ]
    return [
        {
            "signal_id": identity(
                "signal-",
                [
                    eid
                    for u in units
                    if u["unit_id"] in members
                    for eid in u["event_ids"]
                ],
            ),
            "event_ids": sorted(
                eid for u in units if u["unit_id"] in members for eid in u["event_ids"]
            ),
            "association_status": "TIMING_COMPATIBLE_REPEAT_HYPOTHESIS",
        }
        for members in unambiguous
    ]


def validate_assignments(assignments, known):
    seen = set()
    for assignment in assignments:
        ids = set(assignment["event_ids"])
        if not ids or not ids <= known or ids & seen:
            raise ValueError(
                "Signal assignments must be disjoint sets of existing event IDs"
            )
        if not assignment.get("rationale") or not assignment.get("author"):
            raise ValueError("Explicit associations require attribution and rationale")
        seen |= ids
    return seen


def explicit_groups(row, units, hypotheses):
    assignments = row.get("signal_assignments", [])
    seen = validate_assignments(assignments, {e["event_id"] for e in row["events"]})
    groups = [dict(a, association_status="ATTRIBUTED_ASSOCIATION") for a in assignments]
    for group in repeat_groups(units, hypotheses):
        if not set(group["event_ids"]) & seen:
            groups.append(group)
            seen.update(group["event_ids"])
    for unit in units:
        ids = set(unit["event_ids"])
        if ids & seen and not ids <= seen:
            raise ValueError("An extraction duplicate cannot be partly assigned")
        if not ids <= seen:
            groups.append(
                {
                    "signal_id": identity("signal-", ids),
                    "event_ids": sorted(ids),
                    "association_status": "OBSERVED_EVENT_IDENTITY",
                }
            )
    return groups


def inferred_event_groups(events, config):
    units = event_units(events)
    hypotheses, search = period_hypotheses(units, config)
    groups = explicit_groups({"events": events}, units, hypotheses)
    for group in groups:
        members = [u for u in units if set(u["event_ids"]) <= set(group["event_ids"])]
        ids = {u["unit_id"] for u in members}
        group["events"] = members
        group["timing_hypotheses"] = [
            h for h in hypotheses if set(h["unit_ids"]) == ids
        ]
        group["related_timing_hypotheses"] = [
            h["hypothesis_id"] for h in hypotheses if ids & set(h["unit_ids"])
        ]
        group["period_days"] = None
    return groups, hypotheses, search


def associate(row, raw, config):
    if "supplied_hypotheses" in row:
        return associate_supplied(row, config)
    groups, hypotheses, search = inferred_event_groups(row["events"], config)
    row["signals"] = groups
    row["association"] = {
        "status": "MEASURED",
        "hypotheses": hypotheses,
        "computation": search,
        "scope": "Overlapping extraction intervals share one observed-event identity. Nonconflicting maximal timing memberships form repeat hypotheses; intersecting alternatives stay unresolved. TTVs, fragments and independent planets remain possible; no catalogue or event-count eligibility cut.",
    }
    return row


def associate_supplied(row, config):
    groups, hypotheses = [], []
    covered = set()
    for supplied in row["supplied_hypotheses"]:
        if not supplied["followup_eligible"]:
            continue
        ids = set(supplied["alert_ids"])
        events = [e for e in row["events"] if e["alert_id"] in ids]
        if {e["alert_id"] for e in events} != ids or not events:
            raise ValueError("Supplied detector hypothesis lost observed events")
        units = event_units(events)
        hypothesis = dict(
            supplied,
            unit_ids=[u["unit_id"] for u in units],
            timing_basis="ORIGINAL_PROPOSAL",
        )
        hypotheses.append(hypothesis)
        event_ids = sorted(e["event_id"] for e in events)
        covered.update(event_ids)
        groups.append(
            {
                "signal_id": supplied["signal_id"],
                "event_ids": event_ids,
                "association_status": "DETECTOR_SUPPLIED_HYPOTHESIS",
                "events": units,
                "period_days": supplied["period_days"],
                "timing_hypotheses": [hypothesis],
                "related_timing_hypotheses": [hypothesis["hypothesis_id"]],
                "unresolved_aliases": supplied["unresolved_aliases"],
            }
        )
    remaining = [e for e in row["events"] if e["event_id"] not in covered]
    inferred_groups, inferred_hypotheses, search = inferred_event_groups(
        remaining, config
    )
    groups.extend(inferred_groups)
    hypotheses.extend(inferred_hypotheses)
    discovery_view = config.get("discovery_contract", {}).get(
        "discovery_view", "psf_flat"
    )
    if discovery_view != "psf_flat":
        row["discovery_view"] = discovery_view
        for group in groups:
            group["discovery_view"] = discovery_view
        for hypothesis in hypotheses:
            hypothesis["discovery_view"] = discovery_view
    row["signals"] = groups
    row["association"] = {
        "status": "MEASURED",
        "hypotheses": hypotheses,
        "inferred_event_computation": search,
        "scope": "Unchanged detector ephemerides remain supplied hypotheses. Uncovered sparse-event intervals retain their observed identities and adjacent-spacing repeat hypotheses; aliases and TTV interpretations remain unresolved.",
    }
    return row


if __name__ == "__main__":
    from vetting import program_main

    program_main("signals")
