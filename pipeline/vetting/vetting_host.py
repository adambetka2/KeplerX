"""Interpret available host information without stellar or radius eligibility cuts."""

import numpy as np
from astropy.constants import R_earth, R_sun
from lc2 import photometry, uses_flatten_mask


def stellar_radius(record):
    value = record.get("radius_flame")
    if value is not None and np.isfinite(value) and value > 0:
        return {
            "value_solar": value,
            "lower_solar": record.get("radius_flame_lower"),
            "upper_solar": record.get("radius_flame_upper"),
            "source": "Gaia FLAME reported interval",
        }
    tic = record.get("tic_parameters", {})
    radius = tic.get("record", {}).get("Rad")
    if (
        tic.get("status") == "TIC_STELLAR_RECORD_IDENTIFIED"
        and radius is not None
        and radius > 0
    ):
        error = tic["record"].get("s_Rad")
        return {
            "value_solar": radius,
            "lower_solar": None if error is None else max(0, radius - error),
            "upper_solar": None if error is None else radius + error,
            "source": "Identified TIC record; symmetric reported error",
        }
    return {"status": "STELLAR_RADIUS_UNAVAILABLE"}


def interpret_fit(event, radius, dilution):
    fit = event["fit"]
    ratio = fit.get("apparent_radius_ratio")
    record = {
        "event_id": event["event_id"],
        "apparent_radius_ratio": ratio,
        "fit_status": fit["status"],
        "stellar_radius": radius,
        "dilution": dilution,
        "absolute_radius_status": "UNCONSTRAINED",
    }
    if ratio is None or radius.get("value_solar") is None:
        return record
    conversion = float(R_sun / R_earth)
    record["unit_dilution_radius_earth"] = ratio * radius["value_solar"] * conversion
    covariance = fit.get("uncertainty", {}).get("covariance")
    if covariance is not None:
        sigma = float(np.sqrt(np.asarray(covariance)[1, 1]))
        record["apparent_ratio_local_curvature_interval"] = [
            float(ratio * np.exp(-sigma)),
            float(ratio * np.exp(sigma)),
        ]
    record["interpretation"] = (
        "Conditional apparent size at unit dilution. A blended or grazing light curve needs a host-specific diluted transit fit; this estimate is not an absolute-size veto."
    )
    return record


def diluted_fit(raw, event, fraction, config):
    from vetting_models import fit_event, physical_input

    x = physical_input(
        raw,
        event["photometry"],
        config["quality_bitmask"],
        uses_flatten_mask(event["photometry"], config),
    )
    x = x[photometry(raw, event["photometry"])["quarter"] == event["quarter"]]
    usable = (x[:, 3] > 0) & np.isfinite(x).all(axis=1)
    if not usable.any():
        return {"status": "NO_USABLE_COVERAGE"}
    level = np.median(x[usable, 0])
    x[:, 0] = level + (x[:, 0] - level) / fraction
    x[:, 1] /= fraction
    return fit_event(x, event["centre_bkjd"], event["duration_days"], config)


def dilution_constraints(raw, event, dilution, radius, config):
    record = dilution.get("target_flux_fraction", {}).get(event["photometry"])
    if record is None:
        return {"status": "UNCONSTRAINED"}
    fractions = sorted(
        {
            record[key]
            for key in ("value", "lower", "upper")
            if record.get(key) is not None
        }
    )
    if any(not np.isfinite(f) or f <= 0 or f > 1 for f in fractions):
        raise ValueError("A host flux fraction must lie in (0, 1]")
    fits = []
    for fraction in fractions:
        fit = diluted_fit(raw, event, fraction, config)
        ratio = fit.get("apparent_radius_ratio")
        stellar = [
            radius.get(key) for key in ("lower_solar", "value_solar", "upper_solar")
        ]
        sizes = (
            [
                float(ratio * r * float(R_sun / R_earth))
                for r in stellar
                if r is not None
            ]
            if ratio is not None
            else []
        )
        fits.append(
            {"flux_fraction": fraction, "fit": fit, "conditional_radius_earth": sizes}
        )
    return {
        "status": "CONDITIONAL_MODEL_GRID",
        "fits": fits,
        "scope": "Refits at the supplied dilution estimate/endpoints and propagates the supplied stellar radius endpoints. Conditional sensitivity grid, not a joint credible interval; grazing/limb-darkening degeneracies remain.",
    }


def interpret(row, raw, config):
    from vetting_triage import signal_measured

    target = {
        "source_id": row["source_id"],
        "radius": stellar_radius(row.get("stellar", {})),
    }
    alternatives = [
        {"source_id": s["source_id"], "radius": stellar_radius(s)}
        for s in row.get("gaia_scene", {}).get("sources", [])
        if str(s["source_id"]) != str(row["source_id"])
    ]
    dilution = row.get("dilution", {"status": "UNCONSTRAINED"})
    events = {e["event_id"]: e for e in row["events"]}
    for signal in row["signals"]:
        if not signal_measured(signal):
            signal.setdefault(
                "host",
                {
                    "status": "SKIPPED_DISMISSED",
                    "reason": signal["triage"]["reason"],
                },
            )
            continue
        signal["host"] = {
            "status": "INTERPRETED_AVAILABLE_EVIDENCE",
            "possible_hosts": [target, *alternatives],
            "measurements": [
                interpret_fit(events[key], target["radius"], dilution)
                for key in signal["event_ids"]
            ],
            "scope": "Catalogue absence, unknown mass/radius/dilution and uncertain geometry never impose a BOI veto. Host alternatives are retained; no planetary validation probability is calculated.",
        }
        for record in signal["host"]["measurements"]:
            record["dilution_constraints"] = dilution_constraints(
                raw, events[record["event_id"]], dilution, target["radius"], config
            )
        signal["host"]["alternative_host_constraints"] = [
            host_constraints(host, signal["event_ids"], events, row, raw, config)
            for host in alternatives
        ]
    return row


def host_constraints(host, ids, events, row, raw, config):
    dilution = row.get("host_dilution", {}).get(
        str(host["source_id"]), {"status": "UNCONSTRAINED"}
    )
    if host["radius"].get("value_solar") is None and not dilution.get(
        "target_flux_fraction"
    ):
        return {
            "source_id": host["source_id"],
            "status": "UNCONSTRAINED",
            "event_count": len(ids),
            "measurements": [],
            "scope": "No neighbour-specific stellar radius or flux fraction is available for an event-level host fit.",
        }
    measurements = []
    for eid in ids:
        record = interpret_fit(events[eid], host["radius"], dilution)
        record["dilution_constraints"] = dilution_constraints(
            raw, events[eid], dilution, host["radius"], config
        )
        measurements.append(record)
    return {
        "source_id": host["source_id"],
        "measurements": measurements,
        "scope": "Conditional on this alternative host producing the supplied signal. Target dilution is not transferred to a neighbour; host-specific constraints must be supplied.",
    }


if __name__ == "__main__":
    from vetting import program_main

    program_main("host")
