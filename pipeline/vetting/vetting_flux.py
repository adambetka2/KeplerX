"""Characterize only the supplied detector events; no independent transit search."""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from lc2 import photometry, uses_flatten_mask
from vetting import CONFIG, ROOT, RUN, check_run, observation, read, save
from vetting_models import fit_event, physical_input
from vetting_signals import representative_event_ids


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        valid = math.isfinite(value) if type(value) is float else np.isfinite(value)
        return float(value) if valid else None
    return value


def characterize_event(raw, event, config):
    x = physical_input(
        raw,
        event["photometry"],
        config["quality_bitmask"],
        uses_flatten_mask(event["photometry"], config),
    )
    x = x[photometry(raw, event["photometry"])["quarter"] == event["quarter"]]
    try:
        result = fit_event(x, event["centre_bkjd"], event["duration_days"], config)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        result = {"status": "FIT_FAILED", "error": f"{type(exc).__name__}: {exc}"}
    return finite(result)


def pixel_jobs(row, config):
    selected = representative_event_ids(
        row.get("signals", []), config["diagnostic_max_units_per_signal"]
    )
    events = [event for event in row["events"] if event["event_id"] in selected]
    groups = []
    for event in sorted(events, key=lambda item: item["centre_bkjd"]):
        start = event["centre_bkjd"] - event["duration_days"] / 2
        end = event["centre_bkjd"] + event["duration_days"] / 2
        if (
            groups
            and event["quarter"] == groups[-1]["quarter"]
            and start < groups[-1]["end"]
        ):
            groups[-1]["events"].append(event)
            groups[-1]["end"] = max(groups[-1]["end"], end)
        else:
            groups.append(
                {
                    "quarter": event["quarter"],
                    "start": start,
                    "end": end,
                    "events": [event],
                }
            )
    jobs = []
    for group in groups:
        event = min(
            group["events"],
            key=lambda item: (
                {"psf": 0, "sap": 1}.get(item["photometry"], 2),
                item["photometry"],
            ),
        )
        job = {
            key: event[key]
            for key in (
                "event_id",
                "alert_id",
                "quarter",
                "centre_bkjd",
                "duration_days",
            )
        }
        job["represented_event_ids"] = sorted(
            item["event_id"] for item in group["events"]
        )
        fit = event["fit"]
        if fit.get("status") == "FIT_COMPLETE":
            job.update(
                centre_bkjd=fit["centre_bkjd"], duration_days=fit["duration_days"]
            )
        job["timing_basis"] = (
            "FITTED_PROPOSED_EVENT"
            if fit.get("status") == "FIT_COMPLETE"
            else "ORIGINAL_PROPOSAL"
        )
        jobs.append(job)
    return jobs


def append_refined_pixel_jobs(row, config):
    from vetting_signals import representative_units
    from vetting_triage import signal_measured

    events = {event["event_id"]: event for event in row["events"]}
    existing = {job["event_id"] for job in row["pixel_jobs"]}
    for signal in row["signals"]:
        if not signal_measured(signal):
            continue
        hypothesis = next(
            (
                item
                for item in signal.get("timing_hypotheses", [])
                if item.get("timing_basis") == "ORIGINAL_PROPOSAL"
            ),
            None,
        )
        if hypothesis is None:
            continue
        result = row["orbital"]["hypotheses"][hypothesis["hypothesis_id"]]
        refinement = result.get("timing_refinement", {})
        if refinement.get("status") != "SELECTED":
            continue
        timing = refinement["timing"]
        units = representative_units(
            signal["events"], config["diagnostic_max_units_per_signal"]
        )
        for index, unit in enumerate(units):
            represented = [event_id for event_id in unit["event_ids"] if event_id in events]
            if not represented:
                continue
            event = min(
                (events[event_id] for event_id in represented),
                key=lambda item: (item["photometry"] != "psf_flat", item["event_id"]),
            )
            job_id = f"{signal['signal_id']}-timing-refined-{index}"
            if job_id in existing:
                continue
            centre = (
                unit["centre_bkjd"]
                + timing["start_shift_days"]
                + timing["slope"]
                * (unit["centre_bkjd"] - timing["first_centre_bkjd"])
            )
            row["pixel_jobs"].append(
                {
                    "event_id": job_id,
                    "alert_id": event["alert_id"],
                    "quarter": unit["quarter"],
                    "centre_bkjd": centre,
                    "duration_days": unit["duration_days"],
                    "represented_event_ids": represented,
                    "timing_basis": "BOUNDED_REFINED_PROPOSAL",
                    "timing_refined_signal_id": signal["signal_id"],
                }
            )
            existing.add(job_id)


def fit_source(source_id):
    config = read(CONFIG)
    path = RUN / "sources" / f"{source_id}.json"
    row = read(path)
    _, raw = observation(source_id)
    cache = {}
    fit_ids = {
        event_id
        for signal in row.get("signals", [])
        if len(signal["events"]) == 1
        for event_id in signal["events"][0]["event_ids"]
    }
    for event in row["events"]:
        if event["event_id"] not in fit_ids:
            event["fit"] = {
                "status": "NOT_SELECTED_JOINT_SIGNAL_FIT",
                "reason": "Repeated signals use the bounded joint physical-transit fit",
            }
            continue
        key = (
            event["photometry"],
            event["quarter"],
            event["centre_bkjd"],
            event["duration_days"],
        )
        if key not in cache:
            cache[key] = characterize_event(raw, event, config)
        event["fit"] = cache[key]
    row["pixel_jobs"] = pixel_jobs(row, config)
    row["characterization_scope"] = (
        "Single event signals receive one predeclared PSF-before-SAP fit. Repeated signals use the detector-timed bounded joint fit; every original alert remains in the ledger. Pixel jobs retain representative event units across quarters. No period search, threshold selection or invented events."
    )
    save(path, row, compact=True)
    return {"source_id": source_id, "events": len(row["events"])}


def run():
    config = read(CONFIG)
    check_run(ROOT / config["queue"])
    sources = [r["source_id"] for r in read(RUN / "cohort.json")]
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        for result in pool.map(fit_source, sources):
            print(result, flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
