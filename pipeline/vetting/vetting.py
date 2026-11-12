"""Downstream evidence and attributed reviews for detector proposals."""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from deterministic.signal_contracts import (
    calibrated_lanes,
    null_manifest,
    null_population,
)
from evaluation_scope import (
    authorize,
    match_cohort,
    queue_source_identities,
    role_binding,
)
from lc2 import input_manifest, load_source, source_catalogue, verify_views
from runtime import (
    DISPOSITION_LABELS,
    EVENT_MATCHING_CONTRACT,
    OUTCOME_CONTRACT,
    PeakMemory,
    catalogue_search_paths,
    cpu_seconds,
    disposition_calibration_parameters,
    disposition_definition_sha256,
    disposition_evaluation_token,
    disposition_execution,
    dump_path,
    execution_workers,
    implementation_hashes,
    reference_catalogue_paths,
)
from vetting_disposition import (
    AUTOMATIC_ASSESSMENT_RULES,
    adjudicate,
    validate_triage_contract,
)

if __name__ in {"__main__", "__mp_main__"}:
    # pipeline/run.py and spawned workers execute this file by path; the
    # submodules must import this same instance to share stage names and counters.
    sys.modules["vetting"] = sys.modules[__name__]

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / os.environ.get("SPACELAB_VETTING_CONFIG", Path(__file__).with_name("config.json"))
).resolve()
RUN = dump_path(ROOT / json.loads(CONFIG.read_text())["run_directory"])
PRIMARY_CATALOGUES, EXTRA_CATALOGUES = reference_catalogue_paths(
    json.loads(CONFIG.read_text())
)
DETERMINISTIC_ORIGINS = {
    "rust_v1",
    "rust_v2_multiview",
    "rust_v4_primary_whitened",
    "rust_v5_deterministic",
    "rust_v6_deterministic",
    "rust_v6_shallow_v1",
}
PRODUCTION_TRANSFORMER_ORIGINS = {"production_transformer_full_curve_v1"}


def read(path):
    value = json.loads(path.read_text())
    return execution_workers(value) if Path(path) == CONFIG else value


def save(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


COUNTERS = Counter()
COUNTER_LOCK = threading.Lock()
STAGE = {"name": None}


def count(**deltas):
    with COUNTER_LOCK:
        COUNTERS.update(deltas)


def counter_snapshot():
    with COUNTER_LOCK:
        return dict(COUNTERS)


def counter_deltas(before):
    deltas = {}
    for key, value in sorted(counter_snapshot().items()):
        delta = value - before.get(key, 0)
        if delta:
            deltas[key] = round(delta, 3) if isinstance(delta, float) else delta
    return deltas


def started():
    return time.monotonic(), time.process_time()


def elapsed(started_at):
    return {
        "wall_seconds": round(time.monotonic() - started_at[0], 3),
        "cpu_seconds": round(time.process_time() - started_at[1], 3),
    }


def log_source_timing(source_id, **fields):
    line = {"stage": STAGE["name"], "source_id": source_id, **fields}
    with (RUN / "source-timings.jsonl").open("a") as handle:
        handle.write(json.dumps(line) + "\n")


def log_worker(result):
    count(**result["counters"])
    fields = {
        key: value
        for key, value in result.items()
        if key not in {"source_id", "timing", "counters", "triage"}
    }
    log_source_timing(
        result["source_id"], **fields, **result["timing"], **result["counters"]
    )


def timed_stage(name, program):
    path = RUN / "stage-timings.json"
    report = read(path) if path.exists() else {"schema": 2, "stages": []}
    config = read(CONFIG)
    record = {
        "stage": name,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "worker_limits": {
            "source": config["workers"],
            "alias": config["alias"]["workers"],
            "network": config["network_workers"],
        },
    }
    report["stages"].append(record)
    save(path, report)
    STAGE["name"] = name
    started_wall, started_cpu = time.monotonic(), cpu_seconds()
    before, memory = counter_snapshot(), PeakMemory()

    def usage():
        return {
            "wall_seconds": round(time.monotonic() - started_wall, 3),
            "cpu_seconds": round(cpu_seconds() - started_cpu, 3),
            "peak_rss_bytes": memory.stop(),
            **counter_deltas(before),
        }

    try:
        result = program()
    except BaseException as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", **usage())
        save(path, report)
        raise
    record.update(status="COMPLETE", **usage())
    save(path, report)
    return result


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observation(source_id):
    manifest = read(CONFIG).get("input_manifest")
    null_manifest(ROOT, manifest)
    return load_source(ROOT, source_id, manifest)


def catalogue_hashes():
    return {str(path): digest(path) for path in PRIMARY_CATALOGUES + EXTRA_CATALOGUES + catalogue_search_paths(read(CONFIG))}


def freeze_catalogue_file(name, paths):
    target = RUN / name
    current = [
        {
            "path": str(path),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "status": "LOCAL_SNAPSHOT",
        }
        for path in paths
    ]
    if target.exists():
        recorded = {row["path"]: row["sha256"] for row in read(target)}
        expected = {row["path"]: row["sha256"] for row in current}
        if recorded != expected:
            raise ValueError(f"Frozen catalogue provenance changed: {target}")
    else:
        save(target, current)


def freeze_catalogues():
    freeze_catalogue_file("catalogue-provenance.json", PRIMARY_CATALOGUES)
    freeze_catalogue_file("extra-catalogue-provenance.json", EXTRA_CATALOGUES)
    freeze_catalogue_file("complete-catalogue-provenance.json", catalogue_search_paths(read(CONFIG)))
    return catalogue_hashes()


def tap_rows(payload):
    if isinstance(payload, list):
        return payload
    columns = [item["name"] for item in payload["metadata"]]
    return [
        item if isinstance(item, dict) else dict(zip(columns, item, strict=True))
        for item in payload["data"]
    ]


def tap(name, endpoint, query):
    path = RUN / "queries" / f"{name}.json"
    cached = read(path) if path.exists() else None
    if cached is not None:
        if (cached.get("endpoint"), cached.get("query")) != (endpoint, query):
            raise ValueError(f"Cached query identity changed: {name}")
        if cached.get("status") == "COMPLETE":
            count(query_cache_hits=1)
            return cached
    result = {
        "endpoint": endpoint,
        "query": query,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }
    if cached is not None:
        result["previous_failure"] = {
            key: cached.get(key) for key in ("status", "error", "retrieved_utc")
        }
    if not read(CONFIG).get("network", True):
        result.update(status="NETWORK_DISABLED", rows=[])
        save(path, result)
        return result
    started_wall = time.monotonic()
    try:
        response = requests.get(
            endpoint,
            params={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "json",
                "QUERY": query,
            },
            timeout=read(CONFIG)["archive_timeout_seconds"],
        )
        count(network_bytes=len(response.content))
        response.raise_for_status()
        result.update(status="COMPLETE", rows=tap_rows(response.json()))
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        result.update(
            status="QUERY_FAILED", error=f"{type(exc).__name__}: {exc}", rows=[]
        )
    count(network_calls=1, network_seconds=time.monotonic() - started_wall)
    save(path, result)
    return result


SCENE_COLUMNS = ("ra_deg", "dec_deg", "origin_tpf_kic", "phot_g_mean_mag")
SCENE_FIELDS = (*SCENE_COLUMNS, "shard")


def scene_metadata_input(path, reference):
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or not isinstance(reference["path"], str)
        or not reference["path"]
        or not isinstance(reference["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
    ):
        raise ValueError("Experimental scene metadata needs a path and SHA256")
    scene = path.parent / reference["path"]
    payload = scene.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("Experimental scene metadata changed")
    return scene, payload


def manifest_scene(path, reference, metadata, shards, complete_index=False):
    _, payload = scene_metadata_input(path, reference)
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not {"source_id", *SCENE_FIELDS} <= row.keys():
            raise ValueError("Experimental scene metadata is missing source fields")
        sid = row["source_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"[1-9][0-9]*", sid):
            raise ValueError("Experimental scene source IDs must be decimal strings")
        if sid in indexed:
            raise ValueError("Experimental scene metadata repeats a source identity")
        if (
            type(row["origin_tpf_kic"]) is not int
            or row["origin_tpf_kic"] <= 0
            or type(row["shard"]) is not int
            or row["shard"] < 0
            or not all(
                isinstance(row[key], (int, float))
                and not isinstance(row[key], bool)
                and math.isfinite(row[key])
                for key in ("ra_deg", "dec_deg")
            )
            or not 0 <= row["ra_deg"] < 360
            or not -90 <= row["dec_deg"] <= 90
            or (
                row["phot_g_mean_mag"] is not None
                and (
                    not isinstance(row["phot_g_mean_mag"], (int, float))
                    or isinstance(row["phot_g_mean_mag"], bool)
                    or not math.isfinite(row["phot_g_mean_mag"])
                )
            )
        ):
            raise ValueError("Experimental scene source metadata is invalid")
        indexed[sid] = row
    if not set(shards) <= {row["shard"] for row in rows}:
        raise ValueError("Experimental scene metadata does not cover the cohort shards")
    for sid, source in metadata.items():
        if sid not in indexed or any(
            indexed[sid].get(key) != source.get(key)
            for key in (*SCENE_FIELDS, "raw_fits_sha256")
        ):
            raise ValueError(
                "Experimental scene metadata differs from a selected source"
            )
    return [(row["source_id"], row) for row in rows if complete_index or row["shard"] in shards]


def source_metadata(cohort):
    """Metadata rows of the cohort and the scene rows of every indexed source."""
    metadata = {r["source_id"]: observation(r["source_id"])[0] for r in cohort}
    scene, seen = [], set()
    for sid, row in scene_rows(cohort, metadata):
        if sid in seen:
            raise ValueError(f"Duplicate indexed source metadata: {sid}")
        seen.add(sid)
        scene.append({"source_id": sid, **{key: row[key] for key in SCENE_FIELDS}})
    return metadata, scene


def complete_scene_index(path, specification):
    base = specification
    binding = specification.get("null_construction", {}).get("base_manifest")
    if binding is not None:
        _, payload = scene_metadata_input(path, binding)
        base = json.loads(payload)
    complete = (
        base.get("schema") == 3
        and base.get("scene_coverage", {}).get("complete_original_index") is True
    )
    if complete and base["scene_metadata"]["sha256"] != specification["scene_metadata"]["sha256"]:
        raise ValueError("Complete null scene differs from its bound base manifest")
    return complete


def scene_rows(cohort, metadata):
    """Yield the identity and scene fields of every source indexed for the cohort."""
    config = read(CONFIG)
    manifest = config.get("input_manifest")
    if manifest is not None:
        path, specification = input_manifest(ROOT, manifest)
        if "scene_metadata" not in specification:
            return metadata.items()
        preparation = read((ROOT / config["queue"]).parent / "preparation.json")
        shards = {row["shard"] for row in preparation["sources"]}
        complete_index = complete_scene_index(path, specification)
        return manifest_scene(path, specification["scene_metadata"], metadata, shards, complete_index)
    if config.get("discovery_contract", {}).get("origin") in DETERMINISTIC_ORIGINS:
        preparation = read((ROOT / config["queue"]).parent / "preparation.json")
        shards = sorted({r["shard"] for r in preparation["sources"]})
    else:
        index = ROOT / "discSSD/data/index.sqlite"
        with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
            locations = [
                database.execute(
                    "select shard from sources where gaia_source_id = ?",
                    (str(r["source_id"]),),
                ).fetchone()
                for r in cohort
            ]
        if any(row is None for row in locations):
            raise ValueError("A proposed source is absent from the LC2 index")
        shards = sorted({row[0] for row in locations})
    columns = ["source_id", *SCENE_COLUMNS]
    return ((row["source_id"], row) for row in source_catalogue(ROOT, shards, columns))


def code_hashes():
    return implementation_hashes()


def input_hashes(source_ids):
    path = read(CONFIG).get("input_manifest")
    policy = read(CONFIG).get("disposition_policy")
    hashes = {str(ROOT / policy): digest(ROOT / policy)} if policy else {}
    checkpoint = read(CONFIG).get("pixel_checkpoint_ledger")
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint["path"])
        checksum = digest(checkpoint_path)
        if checksum != checkpoint["sha256"]:
            raise ValueError("Pixel checkpoint ledger changed")
        hashes[str(checkpoint_path)] = checksum
    if path is None:
        return hashes
    path, manifest = input_manifest(ROOT, path)
    source_ids = [str(source_id) for source_id in source_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Experimental input source identities are duplicated")
    inputs = [path]
    if "scene_metadata" in manifest:
        scene, _ = scene_metadata_input(path, manifest["scene_metadata"])
        inputs.append(scene)
    for source_id in source_ids:
        source = manifest["sources"][source_id]
        if manifest["schema"] == 3:
            from pipeline.photometry.lc2 import lc2_manifest_files

            inputs.extend(lc2_manifest_files(ROOT, manifest, source))
        else:
            inputs.append(path.parent / source.get("arrays", source.get("deltas")))
        inputs.extend(
            path.parent / p["path"] for p in source.get("pixels", {}).values()
        )
    return {
        **hashes,
        **{str(p): digest(p) for p in dict.fromkeys(inputs)},
    }


def pipeline_contract(config):
    discovery = config.get("discovery_contract")
    if discovery is None:
        raise ValueError("Vetting config must define its discovery contract")
    contract = {
        "contract_version": 3,
        "discovery": discovery,
        "vetting": {
            key: config[key]
            for key in (
                "quality_bitmask",
                "exposure_days",
                "supersample_factor",
                "local_context_durations",
                "minimum_context_days",
                "neighbour_radius_arcsec",
                "optimizer_max_evaluations",
                "joint_fit_max_units_per_signal",
                "timing_refinement_max_period_days",
                "timing_refinement_max_units",
                "timing_refinement_grid_points",
                "timing_refinement_minimum_event_train_positive_z",
                "short_period_shape_max_days",
                "short_period_joint_fit_max_units",
                "short_period_joint_fit_max_missing_fraction",
                "diagnostic_max_units_per_signal",
                "orbital_max_windows",
                "pixel_policy",
                "parent_index_sha256",
                "triage",
            )
        },
        "alias": {
            key: value for key, value in config["alias"].items() if key != "workers"
        },
    }

    contract["vetting"]["flatten_mask"] = (
        "EXCLUDE_NONZERO_PRIMARY_AND_CADENCE_MATCHED_DIAGNOSTICS"
    )
    if "flatten_mask_views" in config:
        contract["vetting"]["flatten_mask_views"] = config["flatten_mask_views"]
        contract["vetting"]["flatten_mask"] = "EXCLUDE_NONZERO_ONLY_IN_DECLARED_VIEWS"
    return contract


def verified_release_artifact(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Release artifacts require exact path and SHA256 fields")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or digest(path) != reference["sha256"]:
        raise ValueError(f"Release artifact is unavailable or changed: {path}")
    return path


def exact_binomial(successes, trials, confidence, upper):
    from scipy.stats import beta

    if upper:
        return (
            1.0
            if successes == trials
            else float(beta.ppf(confidence, successes + 1, trials - successes))
        )
    return (
        0.0
        if successes == 0
        else float(beta.ppf(1 - confidence, successes, trials - successes + 1))
    )


def replay_outcomes(report, run_spec):
    if any(
        value.get("outcome_contract") != OUTCOME_CONTRACT
        for value in (report, run_spec)
    ):
        raise ValueError("Replay uses a legacy or missing outcome contract")
    if any(
        value.get("event_matching_contract") != EVENT_MATCHING_CONTRACT
        for value in (report, run_spec)
    ):
        raise ValueError("Replay uses a legacy or changed event-matching contract")
    cases, morphology, expected = replay_populations(report, run_spec["final_audit"])
    spec = run_spec["final_audit"]
    nulls = null_population(report["null_cases"], spec["null_sources"])
    all_cases = [*cases, *morphology]
    ledgers = [replay_case_ledger(case) for case in all_cases]
    completed = sum(ledger["complete"] for ledger in ledgers)
    emitted = sum(ledger["emitted_alerts"] for ledger in ledgers)
    adjudicated = sum(ledger["adjudicated_alerts"] for ledger in ledgers)
    identity_mismatches = sum(
        case.get("alert_ledger", {}).get("exact_identity_match") is not True
        for case in all_cases
    )
    abstentions = sum(ledger["scientific_abstention"] for ledger in ledgers)
    result = {
        "planet": {
            "trials": expected["planet"],
            "outcomes": sum(
                ledger["complete"]
                and replay_case_planet_recovered(case)
                for case, ledger in zip(cases, ledgers[:len(cases)])
                if case["truth_class"] == "planet"
            ),
        },
        "ledger": {
            "declared_cases": len(all_cases),
            "completed_cases": completed,
            "failed_cases": len(all_cases) - completed,
            "emitted_alerts": emitted,
            "adjudicated_alerts": adjudicated,
            "identity_mismatch_cases": identity_mismatches,
            "scientific_abstention_cases": abstentions,
        },
    }
    result["null"] = nulls
    ledger = result["ledger"]
    ledger["declared_cases"] += nulls["sources"]
    ledger["completed_cases"] += nulls["completed_sources"]
    ledger["failed_cases"] += nulls["sources"] - nulls["completed_sources"]
    ledger["emitted_alerts"] += nulls["emitted_alerts"]
    ledger["adjudicated_alerts"] += nulls["adjudicated_alerts"]
    ledger["identity_mismatch_cases"] += nulls["identity_mismatch_sources"]
    ledger["scientific_abstention_cases"] += nulls["scientific_abstention_sources"]
    for name in tuple(expected)[1:]:
        rows = [case for case in cases if case["truth_class"] == name]
        result[name] = {
            "trials": len(rows),
            "outcomes": sum(replay_case_false_boi(case) for case in rows),
        }
    for name in expected:
        result[name]["coverage"] = replay_coverage(
            [case for case in cases if case["truth_class"] == name]
        )
    result["morphology"] = {
        name: morphology_outcomes(
            [case for case in morphology if case["truth_cohort"] == name]
        )
        for name in spec["morphology_trials"]
    }
    return result


def replay_populations(report, spec):
    cases = report.get("cases", [])
    classes = (
        "planet",
        "eclipsing_binary",
        "background_contaminant",
    )
    if any(case.get("truth_class") not in classes for case in cases):
        raise ValueError("Final replay contains an unsupported truth class")
    cohorts = {
        "planet": "periodic_planet",
        "eclipsing_binary": "native_eb",
        "background_contaminant": "background_contaminant",
    }
    if any(case.get("truth_cohort") != cohorts[case["truth_class"]] for case in cases):
        raise ValueError("Final replay core truth cohorts differ from the release plan")
    morphology = report.get("morphology_cases")
    if not isinstance(morphology, list):
        raise ValueError("Final replay lacks its morphology evaluations")
    if Counter(case.get("truth_cohort") for case in morphology) != Counter(
        spec["morphology_trials"]
    ):
        raise ValueError(
            "Final replay morphology populations differ from run specification"
        )
    if any(case.get("truth_class") != "planet" for case in morphology):
        raise ValueError("Final replay morphology truth classes are invalid")
    all_cases = [*cases, *morphology, *report.get("null_cases", [])]
    parents = [case.get("parent_tpf") for case in all_cases]
    if not all(parents) or len(parents) != len(set(parents)):
        raise ValueError("Final replay cases are not parent-disjoint")
    expected = {
        "planet": spec["injected_components"],
        **spec["reported_trials"],
    }
    if Counter(case["truth_class"] for case in cases) != Counter(expected):
        raise ValueError("Final replay populations differ from the run specification")
    return cases, morphology, expected


def replay_coverage(cases):
    ledgers = [replay_case_ledger(case) for case in cases]
    completed = sum(ledger["complete"] for ledger in ledgers)
    proposed = sum(ledger["emitted_alerts"] > 0 for ledger in ledgers)
    disposed = sum(
        ledger["complete"] and ledger["emitted_alerts"] > 0
        and not ledger["scientific_abstention"] for ledger in ledgers
    )
    return {
        "declared_cases": len(cases),
        "completed_cases": completed,
        "execution_coverage": completed / len(cases) if cases else None,
        "proposed_cases": proposed,
        "nonproposed_cases": len(cases) - proposed,
        "scientific_abstention_cases": sum(
            ledger["scientific_abstention"] for ledger in ledgers
        ),
        "fully_disposed_proposed_cases": disposed,
        "fully_disposed_proposed_fraction": disposed / proposed if proposed else None,
    }


def morphology_outcomes(cases):
    ledgers = [replay_case_ledger(case) for case in cases]
    return {
        "trials": len(cases),
        "exact_event_recovered_parents": sum(
            any(signal["proposal_event_match"]["status"] == "MATCHED"
                for signal in case["signals"]) for case in cases
        ),
        "retained_parents": sum(
            any(signal["downstream"]["triage"]["status"] in {"ACTIVE", "HELD"}
                for signal in case["signals"]) for case in cases
        ),
        "exact_signal_boi_catalogue_recoveries": sum(
            ledger["complete"] and replay_case_planet_recovered(case)
            for case, ledger in zip(cases, ledgers)
        ),
        "boi_catalogue_parents": sum(replay_case_false_boi(case) for case in cases),
        "scientific_abstention_parents": sum(
            ledger["scientific_abstention"] for ledger in ledgers
        ),
        "execution_failure_parents": sum(not ledger["complete"] for ledger in ledgers),
        "coverage": replay_coverage(cases),
    }


def replay_case_planet_recovered(case):
    from deterministic.planet_control_report import match_event_units

    signals = case.get("signals")
    events = case.get("truth_events")
    if not isinstance(signals, list) or not isinstance(events, list):
        raise ValueError("Planet replay lacks its exact physical-event evidence")
    dispositions = {
        item["signal_id"]: item["disposition"]
        for item in case["alert_ledger"]["signal_dispositions"]
    }
    identities = [signal["signal_id"] for signal in signals]
    if len(identities) != len(set(identities)) or set(identities) != set(dispositions):
        raise ValueError("Planet replay exact-event alias ledger is incomplete")
    recovered = False
    for signal in signals:
        if (
            signal["downstream"]["automatic_disposition"]
            != dispositions[signal["signal_id"]]
        ):
            raise ValueError("Planet replay signal disposition differs from its ledger")
        original = match_event_units(
            events, signal["proposal_units"], case["truth"]["duration_hours"]
        )
        if original != signal["proposal_event_match"]:
            raise ValueError("Planet replay proposal membership was misreported")
        units = signal["fitted_units"]
        measured = (
            match_event_units(events, units, case["truth"]["duration_hours"])
            if units is not None else {"status": "UNAVAILABLE"}
        )
        if measured != signal["fitted_event_match"]:
            raise ValueError("Planet replay exact-event recovery was misreported")
        recovered |= (
            dispositions[signal["signal_id"]] == "BOI"
            and measured["status"] == "MATCHED"
        )
    return recovered


def replay_case_false_boi(case):
    ledger = case.get("alert_ledger", {})
    count = ledger.get("boi_signals")
    return (type(count) is int and count > 0) or any(
        item.get("disposition") == "BOI"
        for item in ledger.get("signal_dispositions", [])
    )


def replay_case_ledger(case):
    ledger = case.get("alert_ledger", {})
    counts = {
        name: ledger.get(name)
        for name in ("emitted_alerts", "adjudicated_alerts", "boi_signals")
    }
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("Replay case outcome counts are missing or invalid")
    declared = ledger.get("declared_signal_ids")
    assigned = ledger.get("signal_dispositions")
    if not isinstance(declared, list) or not isinstance(assigned, list):
        raise ValueError("Replay case signal ledger is missing")
    identities = [item.get("signal_id") for item in assigned]
    if (
        len(declared) != len(set(declared))
        or len(identities) != len(set(identities))
        or not all(declared) or not all(identities)
    ):
        raise ValueError("Replay case signal identities are invalid or duplicated")
    complete = (
        case.get("status") == "COMPLETE"
        and ledger.get("exact_identity_match") is True
        and set(declared) == set(identities)
        and all(item.get("disposition") in DISPOSITION_LABELS for item in assigned)
        and counts["boi_signals"] == sum(
            item.get("disposition") == "BOI" for item in assigned
        )
        and counts["emitted_alerts"] == counts["adjudicated_alerts"]
        and bool(declared) == bool(counts["emitted_alerts"])
    )
    return {
        **counts,
        "complete": complete,
        "scientific_abstention": complete and any(
            item["disposition"] == "UNRESOLVED" for item in assigned
        ),
    }


def release_artifact_paths(policy):
    binding = policy.get("release_binding", {})
    expected_binding = {
        "run_spec",
        "detector",
        "disposition_calibration",
        "detector_final_audit",
        "development_replay",
        "final_replay",
    }
    if set(binding) != expected_binding:
        raise ValueError("Released disposition artifact binding is incomplete")
    return {
        name: verified_release_artifact(reference)
        for name, reference in binding.items()
    }


def replay_identity_sets(report):
    cases = [
        *report.get("cases", []), *report.get("morphology_cases", []),
        *report.get("diagnostic_cases", []), *report.get("null_cases", []),
    ]
    for name in ("source_ids", "raw_fits_sha256", "component_ids"):
        seen = set()
        for case in cases:
            values = set(case.get(name, []))
            if not values or not all(values) or values & seen:
                raise ValueError(
                    "Replay trials share or lack source/raw/component identities"
                )
            seen.update(values)
    parents = {str(case.get("parent_tpf")) for case in cases}
    inputs = {case.get("input_sha256") for case in cases}
    sources = {str(source) for case in cases for source in case.get("source_ids", [])}
    arrays = {str(array) for case in cases for array in case.get("raw_fits_sha256", [])}
    components = {
        str(component) for case in cases for component in case.get("component_ids", [])
    }
    if (
        not cases
        or None in inputs
        or "None" in parents
        or not sources
        or not arrays
        or not components
        or len(parents) != len(cases)
        or len(inputs) != len(cases)
    ):
        raise ValueError("Replay case identities are incomplete or duplicated")
    return parents, inputs, sources, arrays, components


def validate_replay_independence(development, final):
    for report in (development, final):
        population = report.get("population", {})
        if population.get("status") != "MEASURED":
            raise ValueError("Replay population independence has not been measured")
    overlaps = final["population"].get("overlap_with_detector_development", {})
    if not overlaps or any(
        value != 0 for role in overlaps.values() for value in role.values()
    ):
        raise ValueError("Final replay population overlaps detector development data")
    development_ids = replay_identity_sets(development)
    final_ids = replay_identity_sets(final)
    if any(left & right for left, right in zip(development_ids, final_ids)):
        raise ValueError("Development and final replay populations overlap")


def replay_detector_measurements(report):
    cases = [
        *report.get("cases", []),
        *report.get("morphology_cases", []),
        *report.get("diagnostic_cases", []),
    ]
    result = {}
    for cohort in sorted({case["truth_cohort"] for case in cases}):
        rows = [case for case in cases if case["truth_cohort"] == cohort]
        ledgers = [replay_case_ledger(case) for case in rows]
        result[cohort] = {
            "trials": len(rows),
            "proposed_parents": sum(row["emitted_alerts"] > 0 for row in ledgers),
            "emitted_alerts": sum(row["emitted_alerts"] for row in ledgers),
            "exact_event_proposal_parents": sum(
                any(
                    signal["proposal_event_match"]["status"] == "MATCHED"
                    for signal in case["signals"]
                )
                for case in rows
            )
            if all(case["truth_class"] == "planet" for case in rows)
            else None,
        }
    if report.get("null_cases"):
        measured = null_population(report["null_cases"])
        result["null"] = {
            name: measured[name]
            for name in (
                "sources", "completed_sources", "emitted_alerts", "adjudicated_alerts",
                "trials", "outcomes", "excluded_proposals", "variants",
            )
        }
    return result


def detector_replay_audit(binding, development, final):
    checks = final.get("release_checks", {})
    passed = (
        development.get("status") == final.get("status") == "REPLAY_PASSED"
        and bool(checks)
        and all(check["status"] == "PASSED" for check in checks.values())
    )
    return {
        "status": "END_TO_END_AUDIT_PASSED"
        if passed
        else "END_TO_END_AUDIT_INCOMPLETE",
        "outcome_contract": OUTCOME_CONTRACT,
        "event_matching_contract": EVENT_MATCHING_CONTRACT,
        "replays": {
            role: binding[role]
            for role in (
                "development_replay",
                "final_replay",
            )
        },
        **{
            f"{name}_sha256": binding[artifact]["sha256"]
            for name, artifact in (
                ("run_spec", "run_spec"),
                ("detector", "detector"),
                ("calibration", "disposition_calibration"),
            )
        },
        "proposal_measurements": {
            "development_replay": replay_detector_measurements(development),
            "final_replay": replay_detector_measurements(final),
        },
        "interpretation": (
            "Proposal yield and exact-event proposal recovery are separate from "
            "automatic disposition. The same independent end-to-end replay "
            "supplies both audits; no additional detector population is required."
        ),
    }


def calibration_cap_outcomes(report, caps, confidence):
    from deterministic.catalogue_control_report import exact_upper_bound

    cases = [
        *report.get("cases", []),
        *report.get("morphology_cases", []),
        *report.get("diagnostic_cases", []),
    ]
    if not all(replay_case_ledger(case)["complete"] for case in cases):
        raise ValueError("Opened cap replay has incomplete physical outcomes")
    nulls = null_population(report.get("null_cases", []))
    if nulls["completed_sources"] != nulls["sources"]:
        raise ValueError("Opened cap replay has incomplete null outcomes")
    result = {}
    for cohort, cap in caps.items():
        rows = [case for case in cases if case.get("truth_cohort") == cohort]
        trials = nulls["trials"] if cohort == "null" else len(rows)
        outcomes = nulls["outcomes"] if cohort == "null" else sum(
            replay_case_false_boi(case) for case in rows
        )
        result[cohort] = {
            "trials": trials, "false_boi": outcomes,
            "upper_bound": exact_upper_bound(outcomes, trials, confidence), "cap": cap,
        }
    return result


def validate_calibration_caps(calibration, report, policy):
    spec = read(verified_release_artifact(calibration.get("spec")))
    audit = policy["release_plan"]["final_audit"]
    caps = audit["opened_false_boi_caps"]
    if (
        spec.get("status") in {"DRAFT_UNBOUND", "BOUND_METADATA_INPUTS_PENDING"}
        or spec.get("false_boi_caps") != caps
        or spec.get("confidence") != audit["confidence"]
    ):
        raise ValueError("Calibration cap specification differs from the launch policy")
    expected = calibration_cap_outcomes(report, caps, audit["confidence"])
    outcomes = calibration.get("outcomes", {})
    measured = outcomes.get("false_boi", {})
    if (
        measured != expected
        or any(type(row.get(name)) is not int for row in measured.values()
               for name in ("trials", "false_boi"))
        or outcomes.get("feasible") is not True
        or type(outcomes.get("incomplete_cases")) is not int
        or outcomes["incomplete_cases"] != 0
        or any(
            row["trials"] < 1 or row["upper_bound"] > row["cap"]
            for row in expected.values()
        )
    ):
        raise ValueError(
            "Calibration cap outcomes differ from the complete opened replay "
            "or exceed a cap"
        )


def validate_calibration_binding(states, binding, policy):
    calibration = states["disposition_calibration"]
    spec = states["run_spec"]
    if (
        calibration.get("parameters") != disposition_calibration_parameters(policy)
        or spec.get("implementation") != policy.get("implementation")
        or spec.get("pipeline_contract") != policy.get("pipeline_contract")
        or spec.get("cohorts")
        != {
            role: policy["evaluation_binding"][role]
            for role in ("development_replay", "final_replay")
        }
        or spec.get("detector") != binding["detector"]
        or spec.get("disposition_calibration") != binding["disposition_calibration"]
    ):
        raise ValueError(
            "Calibration or run specification differs from the frozen experiment"
        )
    evidence = read(verified_release_artifact(calibration.get("development_evidence")))
    cases = [
        *evidence.get("cases", []),
        *evidence.get("morphology_cases", []),
        *evidence.get("diagnostic_cases", []),
    ]
    if (
        evidence.get("replay_role") != "development_replay"
        or evidence.get("status") != "OPENED_REPORT_ASSEMBLED"
        or evidence.get("outcome_contract") != OUTCOME_CONTRACT
        or evidence.get("event_matching_contract") != EVENT_MATCHING_CONTRACT
        or not cases
        or not all(replay_case_ledger(case)["complete"] for case in cases)
    ):
        raise ValueError(
            "Calibration evidence is not the declared opened development replay"
        )
    for path, expected in evidence.get("inputs", {}).items():
        verified_release_artifact({"path": path, "sha256": expected})
    validate_calibration_caps(calibration, states["development_replay"], policy)
    for role in ("development_replay", "final_replay"):
        for cohort in spec["cohorts"][role]["cohorts"]:
            verified_release_artifact(cohort)
        binaries = states[role].get("detector_binaries")
        if not binaries or any(
            binary["sha256"] != binding["detector"]["sha256"] for binary in binaries
        ):
            raise ValueError("Replay detector binary differs from the frozen artifact")
        for binary in binaries:
            verified_release_artifact(binary)


def validate_development_execution(report):
    cases = [
        *report.get("cases", []),
        *report.get("morphology_cases", []),
        *report.get("diagnostic_cases", []),
    ]
    nulls = null_population(report.get("null_cases", []))
    if (
        report.get("boi_catalogue_enabled") is not True
        or report.get("population", {}).get("declared_cases")
        != len(cases) + nulls["sources"]
        or not cases and not nulls["sources"]
        or nulls["sources"] != nulls["completed_sources"]
        or not all(replay_case_ledger(case)["complete"] for case in cases)
    ):
        raise ValueError("Frozen development replay execution is incomplete")


def release_artifact_states(paths, binding, policy):
    states = {name: read(path) for name, path in paths.items() if name != "detector"}
    return validate_release_artifact_states(states, binding, policy)


def validate_release_artifact_states(states, binding, policy):
    for role in ("development_replay", "final_replay"):
        for path, expected in states[role].get("inputs", {}).items():
            verified_release_artifact({"path": path, "sha256": expected})
    run_spec = states["run_spec"]
    calibration = states["disposition_calibration"]
    detector = states["detector_final_audit"]
    definition_hash = disposition_definition_sha256(policy)
    if any(
        value.get("outcome_contract") != OUTCOME_CONTRACT
        for value in (
            policy, run_spec, states["development_replay"], states["final_replay"]
        )
    ):
        raise ValueError(
            "Released disposition uses a legacy or missing outcome contract"
        )
    if any(
        value.get("event_matching_contract") != EVENT_MATCHING_CONTRACT
        for value in (
            policy, run_spec, states["development_replay"], states["final_replay"]
        )
    ):
        raise ValueError("Released disposition event-matching contract changed")
    if (
        run_spec.get("status") != "SEALED"
        or calibration.get("status") != "CALIBRATED"
        or detector.get("status") != "END_TO_END_AUDIT_PASSED"
        or states["development_replay"].get("status") != "REPLAY_PASSED"
        or states["final_replay"].get("status") != "REPLAY_PASSED"
        or states["development_replay"].get("replay_role") != "development_replay"
        or states["final_replay"].get("replay_role") != "final_replay"
        or any(states[role].get("boi_catalogue_enabled") is not True
               for role in ("development_replay", "final_replay"))
        or policy.get("definition_sha256") != definition_hash
        or any(
            states[name].get("disposition_definition_sha256") != definition_hash
            for name in ("development_replay", "final_replay")
        )
        or any(
            states[name].get("implementation") != policy.get("implementation")
            for name in ("development_replay", "final_replay")
        )
    ):
        raise ValueError("Released disposition artifacts have not passed")
    validate_development_execution(states["development_replay"])
    expected = {
        "run_spec_sha256": binding["run_spec"]["sha256"],
        "detector_sha256": binding["detector"]["sha256"],
        "calibration_sha256": binding["disposition_calibration"]["sha256"],
    }
    evaluation_binding = policy["evaluation_binding"]
    if (
        any(evaluation_binding.get(name) != value for name, value in expected.items())
        or any(
            states[role].get(name) != value
            for role in ("development_replay", "final_replay")
            for name, value in expected.items()
        )
        or any(
            states[role].get("controls_sha256")
            != evaluation_binding[role]["controls_sha256"]
            for role in ("development_replay", "final_replay")
        )
    ):
        raise ValueError("Replay artifact identities do not match the frozen policy")
    measured = {
        "run_spec_sha256": detector.get("run_spec_sha256"),
        "detector_sha256": detector.get("detector_sha256"),
        "calibration_sha256": detector.get("calibration_sha256"),
    }
    if (
        measured != expected
        or calibration.get("detector_sha256") != expected["detector_sha256"]
    ):
        raise ValueError("Detector release artifact identities do not match")
    validate_calibration_binding(states, binding, policy)
    if detector != detector_replay_audit(
        binding, states["development_replay"], states["final_replay"]
    ):
        raise ValueError("Detector audit differs from the actual replay ledgers")
    validate_replay_independence(states["development_replay"], states["final_replay"])
    return states


def validate_release_ledger(ledger, outcomes):
    if set(ledger) != {
        "status",
        "declared_cases",
        "completed_cases",
        "failed_cases",
        "emitted_alerts",
        "adjudicated_alerts",
        "identity_mismatch_cases",
        "scientific_abstention_cases",
    } or not (
        ledger.get("status") == "PASSED"
        and {name: ledger[name] for name in outcomes} == outcomes
        and ledger["declared_cases"] == ledger["completed_cases"]
        and ledger["failed_cases"] == 0
        and ledger["emitted_alerts"] == ledger["adjudicated_alerts"]
        and ledger["identity_mismatch_cases"] == 0
    ):
        raise ValueError("Released disposition outcome ledger is incomplete")


def validate_release_results(checks, final, run_spec):
    from deterministic.release_report import release_checks
    measured_checks, outcomes = release_checks(final, run_spec)
    if final.get("outcomes") != outcomes:
        raise ValueError("Final replay measured outcomes were misreported")
    if checks != measured_checks or checks != final.get("release_checks"):
        raise ValueError(
            "Policy checks differ from measured sealed publication endpoints"
        )
    if not all(check["status"] == "PASSED" for check in checks.values()):
        raise ValueError("Measured sealed publication endpoints failed")
    validate_release_ledger(checks["complete_outcome_ledger"], outcomes["ledger"])


def validate_release_contract(policy):
    validate_triage_contract(policy)
    if policy.get("boi_catalogue_enabled") is not True:
        raise ValueError("Qualified control policy must explicitly enable BOI outcomes")
    paths = release_artifact_paths(policy)
    states = release_artifact_states(paths, policy["release_binding"], policy)
    if states["run_spec"].get("final_audit") != release_audit_spec(policy):
        raise ValueError("Release audit specification differs from the frozen policy")
    validate_release_results(
        policy.get("release_checks", {}),
        states["final_replay"],
        states["run_spec"],
    )


def release_audit_spec(policy):
    plan = policy["release_plan"]
    spec = plan["final_audit"]
    return {
        "injected_components": spec["planet_trials"],
        **{name: spec[name] for name in (
            "confidence", "reported_trials", "null_sources",
            "minimum_periodic_recovery_fraction",
            "maximum_null_false_candidate_fraction",
            "maximum_single_event_offset_conflicts",
            "opened_false_boi_caps",
        )},
        "morphology_trials": plan["morphology_trials"],
    }


def frozen_assessment_rules(rules):
    fields = {
        "instrument": ("material_quality_flags", "spatial_null_bic_margin"),
        "source_localization": ("accepted_statuses", "material_statuses"),
        "contamination": (
            "minimum_scene_prf_compared_quarters",
            "minimum_neighbour_depth_sigma",
            "minimum_competing_source_separation_arcsec",
            "minimum_catalogue_match_units",
            "minimum_catalogue_match_fraction",
            "maximum_harmonic_order",
            "maximum_harmonic_period_error_fraction",
            "minimum_harmonic_match_units",
            "minimum_harmonic_expected_match_fraction",
            "accepted_scene_statuses",
            "accepted_neighbour_statuses",
        ),
        "identity_catalogues": (
            "minimum_catalogue_match_units",
            "minimum_catalogue_match_fraction",
            "maximum_harmonic_order",
            "maximum_harmonic_period_error_fraction",
            "minimum_harmonic_match_units",
            "minimum_harmonic_expected_match_fraction",
            "material_signal_classes",
            "accepted_catalogue_statuses",
        ),
        "orbital": (
            "maximum_odd_even_bic_gain",
            "minimum_secondary_sigma",
            "maximum_secondary_to_primary_depth_ratio",
            "maximum_phase_modulation_period_days",
            "minimum_phase_modulation_signal_to_noise",
            "minimum_phase_modulation_amplitude_fraction",
            "maximum_phase_modulation_to_primary_depth_ratio",
        ),
    }
    return set(rules) == set(fields) and all(
        frozen_assessment_rule(rules[name], AUTOMATIC_ASSESSMENT_RULES[name], names)
        for name, names in fields.items()
    )


def frozen_assessment_rule(rule, rule_id, fields):
    lists = {
        "material_quality_flags",
        "accepted_statuses",
        "material_statuses",
        "accepted_scene_statuses",
        "accepted_neighbour_statuses",
        "material_signal_classes",
        "accepted_catalogue_statuses",
    }
    if (
        not isinstance(rule, dict)
        or rule.get("id") != rule_id
        or rule.get("status") != "RELEASED"
    ):
        return False
    for name in fields:
        value = rule.get(name)
        if name in lists:
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item for item in value
            ):
                return False
            if name != "material_quality_flags" and not value:
                return False
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return False
    return True


def validate_operational_policy(policy):
    rules = policy.get("assessment_rules", {})
    binding = policy.get("evaluation_binding", {})
    digest_fields = (
        "run_spec_sha256",
        "detector_sha256",
        "calibration_sha256",
    )
    if (
        policy.get("schema") != 3
        or policy.get("policy_id") != "deterministic-boi-v1"
        or policy.get("boi_catalogue_enabled") is not True
        or not frozen_assessment_rules(rules)
        or set(binding) != {*digest_fields, "development_replay", "final_replay"}
        or any(
            not isinstance(binding.get(name), str) or len(binding[name]) != 64
            for name in digest_fields
        )
        or any(
            not role_binding(binding.get(role))
            for role in ("development_replay", "final_replay")
        )
        or policy.get("definition_sha256") != disposition_definition_sha256(policy)
    ):
        raise ValueError("Disposition policy scientific definition is not frozen")
    validate_triage_contract(policy)
    calibrated_lanes(policy, required=True)


def load_disposition_policy():
    config = read(CONFIG)
    path = config.get("disposition_policy")
    if path is None:
        return None
    policy = read(ROOT / path)
    if policy.get("policy_id") != "deterministic-boi-v1":
        raise ValueError("Unknown disposition policy")
    if policy.get("status") == "BOI_CATALOGUE":
        validate_triage_contract(policy)
        if policy.get("outcome_contract") != OUTCOME_CONTRACT:
            raise ValueError("Unknown BOI outcome contract")
    operational = policy.get("status") in {"FROZEN_EVALUATION", "RELEASED"}
    bound = operational or (policy.get("status") == "BOI_CATALOGUE" and "pipeline_contract" in policy)
    if bound and policy.get("pipeline_contract") != pipeline_contract(config):
        raise ValueError("Disposition policy and pipeline settings differ")
    if bound and policy.get("implementation") != implementation_hashes():
        raise ValueError(
            "Disposition calibration implementation changed; re-evaluate the policy"
        )
    if operational:
        validate_operational_policy(policy)
    if policy.get("status") == "FROZEN_EVALUATION":
        token = os.environ.get("SPACELAB_DISPOSITION_EVALUATION_TOKEN")
        policy["_evaluation_role"] = next(
            (
                role
                for role in ("development_replay", "final_replay")
                if token == disposition_evaluation_token(policy, role)
            ),
            None,
        )
        policy["_evaluation_authorized"] = policy["_evaluation_role"] is not None
    elif policy.get("status") == "RELEASED":
        validate_release_contract(policy)
    return policy


def discovery_provenance(queue_path):
    queue_path = Path(queue_path).resolve()
    queue = read(queue_path)
    reference = queue.get("discovery_manifest")
    if not reference:
        raise ValueError("Queue must cite its discovery manifest")
    manifest_path = Path(reference)
    if not manifest_path.is_absolute():
        manifest_path = queue_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = read(manifest_path)
    if manifest.get("origin") in DETERMINISTIC_ORIGINS:
        from deterministic.run import verify_discovery

        verify_discovery(queue_path, manifest, read(CONFIG))
    else:
        verify_transformer_discovery(manifest)
    return manifest_path, manifest


def verify_transformer_discovery(manifest):
    names = (
        (
            "production_survey.py",
            "production_model.py",
            "production_data.py",
            "production_eval.py",
            "production_train.py",
            "../photometry/lc2.py",
        )
        if manifest.get("origin") in PRODUCTION_TRANSFORMER_ORIGINS
        else ("survey.py", "model.py", "data.py", "../photometry/lc2.py")
    )
    source = ROOT / "releases/transformer-v1/source"
    code = {
        name: digest(
            ROOT / "pipeline/photometry/lc2.py"
            if name == "../photometry/lc2.py"
            else source / name
        )
        for name in names
    }
    if manifest.get("code") != code:
        raise ValueError("Queue was not produced by this Transformer implementation")
    expected_contract = pipeline_contract(read(CONFIG))["discovery"]
    if manifest.get("checkpoint_contract") != expected_contract:
        raise ValueError("Transformer checkpoint and pipeline signal contracts differ")
    if not all(
        manifest.get(key) for key in ("checkpoint_sha256", "calibration_sha256")
    ):
        raise ValueError("Discovery checkpoint or calibration provenance is missing")
    if not math.isfinite(manifest.get("threshold", math.nan)):
        raise ValueError("Discovery cadence threshold is missing or nonfinite")
    if manifest.get("decision") != (
        "Direct valid-cadence logits above frozen model threshold; no external "
        "reranking"
    ):
        raise ValueError(
            "Queue discovery decision is not the sole Transformer detector"
        )


def queue_digest(queue_path):
    manifest_path, manifest = discovery_provenance(queue_path)
    payload = {
        "queue": read(Path(queue_path).resolve()),
        "resolved_rows": read_queue(queue_path),
        "discovery_manifest": manifest,
        "discovery_manifest_path": str(manifest_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluation_queue_scope(queue_path):
    policy = load_disposition_policy()
    if not policy or policy.get("status") != "FROZEN_EVALUATION":
        return None
    config = read(CONFIG)
    role = authorize(policy, os.environ.get("SPACELAB_DISPOSITION_EVALUATION_TOKEN"))
    source_ids = [row["source_id"] for row in read_queue(queue_path)]
    evaluation = match_cohort(
        policy,
        role,
        queue_source_identities(source_ids, config.get("input_manifest")),
        config.get("input_manifest"),
        subset=True,
        pixel_inputs=config.get("frozen_pixel_inputs"),
    )
    expected = config.get("evaluation_scope")
    if expected is not None and evaluation != expected:
        raise ValueError("Vetting queue differs from its prepared evaluation scope")
    return evaluation


def check_run(queue_path):
    evaluation = evaluation_queue_scope(queue_path)
    path = RUN / "manifest.json"
    if not path.exists():
        return evaluation
    manifest = read(path)
    if manifest.get("evaluation_scope") != evaluation:
        raise ValueError("Vetting resume differs from its frozen evaluation scope")
    if manifest["queue_sha256"] != queue_digest(queue_path) or manifest[
        "config_sha256"
    ] != digest(CONFIG):
        raise ValueError("Frozen inputs/configuration changed; use a new run directory")
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("Implementation changed; use a new run directory")
    if manifest.get("catalogue_sha256") != catalogue_hashes():
        raise ValueError("Catalogue snapshots changed; use a new run directory")
    if manifest.get("input_sha256", {}) != input_hashes(manifest["source_ids"]):
        raise ValueError("Experimental input changed; use a new run directory")
    return evaluation


def prepare():
    config = read(CONFIG)
    queue_path = ROOT / config["queue"]
    evaluation = check_run(queue_path)
    discovery_path, discovery = discovery_provenance(queue_path)
    frozen_catalogues = freeze_catalogues()
    cohort = read_queue(queue_path)
    metadata, scene = source_metadata(cohort)
    save(RUN / "cohort.json", cohort)
    save(RUN / "metadata.json", metadata)
    save(RUN / "scene.json", scene, compact=True)
    if not (RUN / "manifest.json").exists():
        save(
            RUN / "manifest.json",
            {
                "queue_sha256": queue_digest(queue_path),
                "config_sha256": digest(CONFIG),
                "worker_limits": {
                    "source": config["workers"],
                    "alias": config["alias"]["workers"],
                    "network": config["network_workers"],
                },
                "source_ids": [r["source_id"] for r in cohort],
                "evaluation_scope": evaluation,
                "alerts": sum(len(alert_intervals(r)) for r in cohort),
                "selection": "Every supplied source and alert, including monotransits",
                "discovery": {
                    "manifest": str(discovery_path),
                    "manifest_sha256": digest(discovery_path),
                    "origin": discovery["origin"],
                    "detector_contract": discovery["detector_contract"],
                    "binary": discovery["binary"],
                }
                if discovery.get("origin") in DETERMINISTIC_ORIGINS
                else {
                    "manifest": str(discovery_path),
                    "manifest_sha256": digest(discovery_path),
                    "checkpoint_sha256": discovery["checkpoint_sha256"],
                    "calibration_sha256": discovery["calibration_sha256"],
                    "cadence_threshold": discovery["threshold"],
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha256": code_hashes(),
                "catalogue_sha256": frozen_catalogues,
                "input_sha256": input_hashes([r["source_id"] for r in cohort]),
            },
        )
    print(json.dumps({"prepared_sources": len(cohort), "queries": 0}), flush=True)


def local_catalogues():
    return [(path, read(path)) for path in PRIMARY_CATALOGUES]


def catalogue_matches(source, catalogues):
    matches = []
    radius = read(CONFIG)["neighbour_radius_arcsec"]
    target = SkyCoord(source["ra_deg"] * u.deg, source["dec_deg"] * u.deg)
    for path, rows in catalogues:
        for r in rows:
            if r.get("ra") is None or r.get("dec") is None:
                continue
            if abs(r["dec"] - source["dec_deg"]) > radius / 3600:
                continue
            separation = target.separation(
                SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
            ).arcsec
            if separation > radius:
                continue
            identity = bool(
                str(r.get("gaia_dr3_id", "")).split(" ")[-1] == source["source_id"]
            )
            matches.append(
                {
                    "catalogue": path.name,
                    "separation_arcsec": float(separation),
                    "exact_identifier_match": identity,
                    "record": r,
                }
            )
    return {
        "status": "LOCAL_SNAPSHOTS_CHECKED",
        "matches": matches,
        "radius_arcsec": radius,
        "coverage": (
            "NASA cumulative and DR25 KOI, PS and TOI snapshots in the configured "
            "scene. No match is not literature closure or exclusion of distant "
            "contamination."
        ),
    }


def query_row_matches_source(row, sid):
    return str(row.get("source_id", "")) == sid or row.get("id") == f"Gaia DR3 {sid}"


def query_evidence(sid, prefix):
    queries = [read(p) for p in sorted((RUN / "queries").glob(f"{prefix}-*.json"))]
    relevant = [
        query
        for query in queries
        if re.search(rf"(?<!\d){re.escape(sid)}(?!\d)", query["query"])
    ]
    if not relevant or any(q["status"] != "COMPLETE" for q in relevant):
        return {"status": "QUERY_UNAVAILABLE", "matches": []}
    rows = [
        row
        for query in relevant
        for row in query["rows"]
        if query_row_matches_source(row, sid)
    ]
    return {
        "status": "EXACT_IDENTIFIERS_QUERIED",
        "matches": rows,
        "coverage": (
            "Scoped exact-identifier query; absence does not establish absence "
            "of literature"
        ),
    }


def alert_intervals(proposal):
    if "alerts" not in proposal:
        raise ValueError("An explicit detector alert queue is required")
    return proposal["alerts"]


def read_queue(path):
    path = Path(path).resolve()
    discovery_provenance(path)
    rows, seen = [], set()
    for entry in read(path)["rows"]:
        row = read(path.parent / entry["evidence"]) if "evidence" in entry else entry
        sid = str(row["source_id"])
        if sid != str(entry["source_id"]) or sid in seen:
            raise ValueError("Queue source identities must be unique and consistent")
        ids = [alert["event_id"] for alert in alert_intervals(row)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate event IDs: {sid}")
        seen.add(sid)
        rows.append(dict(row, source_id=sid))
    return rows


def event_specs(proposal):
    available = [
        v["view"]
        for v in proposal["views"]
        if v["status"] in {"SCORED", "AVAILABLE", "COMBINED_SCORED"}
    ]
    scored = {
        v["view"]
        for v in proposal["views"]
        if v["status"] in {"SCORED", "COMBINED_SCORED"}
    }
    specs = []
    for alert in alert_intervals(proposal):
        start, end = alert["start_bkjd"], alert["end_bkjd"]
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
            or alert["photometry"] not in scored
        ):
            raise ValueError(
                "A detector event must have a positive interval and a scored extraction"
            )
        view = next(
            (view for view in ("psf", "sap", alert["photometry"]) if view in available),
            None,
        )
        if view is None:
            raise ValueError("A detector event has no available primary extraction")
        specs.append(
            {
                "event_id": f"{alert['event_id']}:{view}",
                "alert_id": alert["event_id"],
                "quarter": alert["quarter"],
                "photometry": view,
                "centre_bkjd": (start + end) / 2,
                "duration_days": end - start,
                "original_alert": alert,
                "fit": {"status": "NOT_RUN"},
            }
        )
    return specs


def neighbours(source, scene):
    """Scene rows within the neighbour radius, in scene order."""
    radius = read(CONFIG)["neighbour_radius_arcsec"]
    declinations = np.array([r["dec_deg"] for r in scene], dtype=float)
    if not np.isfinite(declinations).all():
        raise ValueError("Scene rows must have finite declinations")
    window = np.abs(declinations - source["dec_deg"]) <= radius / 3600
    rows = [
        scene[index]
        for index in np.flatnonzero(window)
        if scene[index]["source_id"] != source["source_id"]
    ]
    nearby = []
    if rows:
        target = SkyCoord(source["ra_deg"] * u.deg, source["dec_deg"] * u.deg)
        coordinates = SkyCoord(
            [r["ra_deg"] for r in rows] * u.deg, [r["dec_deg"] for r in rows] * u.deg
        )
        separations = target.separation(coordinates).arcsec
        nearby = [
            {
                "source_id": r["source_id"],
                "separation_arcsec": float(separation),
                "same_parent_tpf": r["origin_tpf_kic"] == source["origin_tpf_kic"],
                "phot_g_mean_mag": r["phot_g_mean_mag"],
            }
            for r, separation in zip(rows, separations, strict=True)
            if separation <= radius
        ]
    shards = sorted({row["shard"] for row in scene})
    return {
        "status": "EXTRACTED_SOURCE_SCENE",
        "sources": nearby,
        "shards": shards,
        "radius_arcsec": radius,
        "coverage": (
            f"KBonus sources in {len(shards)} indexed shards in the configured "
            "scene; not a complete Gaia scene"
        ),
    }


def fit_source(proposal, metadata, scene, catalogues, config):
    sid = proposal["source_id"]
    path = RUN / "sources" / f"{sid}.json"
    if path.exists():
        cached = read(path)
        if cached["raw_fits_sha256"] != metadata[sid]["raw_fits_sha256"]:
            raise ValueError(f"Source input changed: {sid}")
        return cached
    source, raw = observation(sid)
    verify_views(proposal, raw)
    gaia = query_evidence(sid, "gaia")
    result = {
        "source_id": sid,
        "raw_fits_sha256": source["raw_fits_sha256"],
        "coordinates": {
            k: source[k] for k in ["ra_deg", "dec_deg", "kic", "origin_tpf_kic"]
        },
        "catalogues": catalogue_matches(source, catalogues),
        "literature": query_evidence(sid, "simbad"),
        "stellar": gaia["matches"][0] if gaia["matches"] else {},
        "neighbours": neighbours(source, scene),
        "events": event_specs(proposal),
        "alerts": alert_intervals(proposal),
        "localization": {"status": "NOT_RUN", "events": []},
        "reviews": [],
    }
    if "supplied_hypotheses" in proposal:
        result["supplied_hypotheses"] = proposal["supplied_hypotheses"]
        result["rust_rows"] = proposal["rust_rows"]
        result["shallow_rows"] = proposal["shallow_rows"]
    result["disposition"] = adjudicate(result)
    save(path, result, compact=True)
    return result


def run():
    check_run(ROOT / read(CONFIG)["queue"])
    config, metadata, cohort = (
        read(CONFIG),
        read(RUN / "metadata.json"),
        read(RUN / "cohort.json"),
    )
    scene, catalogues = read(RUN / "scene.json"), local_catalogues()
    rows = []
    for proposal in cohort:
        rows.append(fit_source(proposal, metadata, scene, catalogues, config))
        save(
            RUN / "progress.json",
            {"sources_complete": len(rows), "sources_total": len(cohort)},
        )
        if len(rows) % 10 == 0:
            print(json.dumps(read(RUN / "progress.json")), flush=True)
    summarize()


def status_counts(rows):
    fits = Counter(
        event["fit"].get("status", "NOT_RUN") for row in rows for event in row["events"]
    )
    pixels = Counter(
        event["status"]
        for row in rows
        for event in row["localization"].get("events", [])
    )
    return dict(fits), dict(pixels)


def summary_index(rows):
    indexed = []
    omitted = {"signals", "attributed_signal_reviews", "event_missing_evidence"}
    for row in rows:
        disposition = row["disposition"]
        indexed.append(
            {
                "source_id": row["source_id"],
                "disposition": {
                    key: value
                    for key, value in disposition.items()
                    if key not in omitted
                },
                "boi_signals": [
                    {
                        "signal_id": signal["signal_id"],
                        "event_ids": signal["event_ids"],
                        "reason_codes": signal.get("reason_codes", []),
                    }
                    for signal in disposition.get("signals", [])
                    if signal.get("disposition") == "BOI"
                ],
                "evidence": str(RUN / "sources" / f"{row['source_id']}.json"),
            }
        )
    return indexed


def signal_ledger(row):
    from boi import signal_record

    path = RUN / "sources" / f"{row['source_id']}.json"
    return [signal_record(row, signal, path) for signal in row.get("signals", [])]


def summarize():
    from vetting_triage import signal_active

    cohort = read(RUN / "cohort.json")
    execution = disposition_execution(load_disposition_policy())
    found_ids = set()
    counts = Counter()
    fits, pixels = Counter(), Counter()
    held_by_stage, dismissed_by_stage = Counter(), Counter()
    dispositions, automatic_classification = Counter(), Counter()
    survivors, bois, indexed, ledger = [], [], [], []
    candidate_count = new_candidate_count = 0
    candidate_unknown = new_candidate_unknown = False
    all_complete = True
    for proposal in cohort:
        row = read(RUN / "sources" / f"{proposal['source_id']}.json")
        found_ids.add(row["source_id"])
        if not row["disposition"]["retained"]:
            raise ValueError("A frozen cohort source was not retained")
        expected_alerts = alert_intervals(proposal)
        if {event["alert_id"] for event in row["events"]} != {
            alert["event_id"] for alert in expected_alerts
        }:
            raise ValueError(f"Alert preservation failed for {row['source_id']}")
        counts["original_alerts_preserved"] += len(expected_alerts)
        counts["event_hypotheses"] += len(row["events"])
        source_fits, source_pixels = status_counts([row])
        fits.update(source_fits)
        pixels.update(source_pixels)
        signals = row.get("signals", [])
        active = [signal for signal in signals if signal_active(signal)]
        held = [
            signal
            for signal in signals
            if signal.get("triage", {}).get("status") == "HELD"
        ]
        dismissed = [
            signal
            for signal in signals
            if signal.get("triage", {}).get("status") == "DISMISSED"
        ]
        counts["signals"] += len(signals)
        counts["triage_surviving_signals"] += len(active)
        counts["triage_held_signals"] += len(held)
        counts["triage_dismissed_signals"] += len(dismissed)
        held_by_stage.update(signal["triage"]["held_stage"] for signal in held)
        dismissed_by_stage.update(
            signal["triage"]["dismissed_stage"] for signal in dismissed
        )
        counts["sources_with_surviving_signals"] += bool(active)
        counts["stellar_radius_sources"] += (
            row["stellar"].get("radius_flame") is not None
        )
        counts["sources_with_indexed_references"] += bool(
            row["literature"].get("references")
        )
        counts["excluded_sources"] += bool(signals) and not bool(active)
        counts["unresolved_signals"] += row["disposition"].get("unresolved_signals", 0)
        dispositions.update((row["disposition"]["status"],))
        automatic_classification.update(
            (row["disposition"]["automatic_classification"],)
        )
        complete = row.get("processing_status") == "EVIDENCE_STAGES_COMPLETE"
        if complete and row.get("disposition_execution") != execution:
            raise ValueError("Source disposition execution differs from the run")
        all_complete &= complete
        value = row["disposition"]["boi_signal_count"]
        candidate_unknown |= value is None
        if value is not None:
            candidate_count += value
        value = row["disposition"]["new_boi_signal_count"]
        new_candidate_unknown |= value is None
        if value is not None:
            new_candidate_count += value
        source_path = str(RUN / "sources" / f"{row['source_id']}.json")
        survivors.extend(
            {
                "source_id": row["source_id"],
                "signal_id": signal["signal_id"],
                "event_ids": signal["event_ids"],
                "event_units": len(signal.get("events", [])),
                "period_days": signal.get("period_days"),
                "triage_history": signal.get("triage", {}).get("history", []),
                "evidence": source_path,
            }
            for signal in active
        )
        bois.extend(
            {
                "source_id": row["source_id"],
                "signal_id": signal["signal_id"],
                "event_ids": signal["event_ids"],
                "reason_codes": signal.get("reason_codes", []),
                "evidence": source_path,
            }
            for signal in row["disposition"].get("signals", [])
            if signal.get("disposition") == "BOI"
        )
        indexed.extend(summary_index([row]))
        ledger.extend(signal_ledger(row))
        del row, signals, active, held, dismissed
    if found_ids != {item["source_id"] for item in cohort}:
        raise ValueError("Summary source identities do not match the frozen cohort")
    save(RUN / "surviving-signals.json", survivors)
    save(RUN / "bois.json", bois)
    save(RUN / "signal-ledger.json", ledger)
    summary = {
        "processing_status": "EVIDENCE_STAGES_COMPLETE" if all_complete else "PARTIAL",
        "disposition_execution": execution,
        "sources": len(cohort),
        "event_hypotheses": counts["event_hypotheses"],
        "original_alerts_preserved": counts["original_alerts_preserved"],
        "signals": counts["signals"],
        "associated_signals": counts["signals"],
        "triage_surviving_signals": counts["triage_surviving_signals"],
        "triage_held_signals": counts["triage_held_signals"],
        "triage_held_by_stage": dict(held_by_stage),
        "triage_dismissed_signals": counts["triage_dismissed_signals"],
        "triage_dismissed_by_stage": dict(dismissed_by_stage),
        "sources_with_surviving_signals": counts["sources_with_surviving_signals"],
        "fit_status_counts": dict(fits),
        "pixel_status_counts": dict(pixels),
        "stellar_radius_sources": counts["stellar_radius_sources"],
        "sources_with_indexed_references": counts["sources_with_indexed_references"],
        "dispositions": dict(dispositions),
        "boi_signals": len(bois),
        "distinct_boi_signal_count": None if candidate_unknown else candidate_count,
        "boi_counting_unit": "SOURCE_AND_VIEW_SIGNAL_PROPOSAL",
        "signal_ledger": str(RUN / "signal-ledger.json"),
        "boi_list": str(RUN / "bois.json"),
        "surviving_signal_list": str(RUN / "surviving-signals.json"),
        "new_boi_signals": (
            None if new_candidate_unknown else new_candidate_count
        ),
        "excluded_sources": counts["excluded_sources"],
        "statistical_validation_run": False,
        "automatic_classification": dict(automatic_classification),
        "unresolved_signals": counts["unresolved_signals"],
        "limitations": (
            "BOIs are final pipeline transit survivors. Keep source/view proposals "
            "distinct from unique astrophysical signals; associated aliases and "
            "correlated views retain their evidence. Follow-up establishes physical "
            "validation and confirmation. Attributed reviews remain annotations."
        ),
        "manifest": str(RUN / "manifest.json"),
        "rows": indexed,
    }
    save(RUN / "results.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


def finish_source(sid):
    from vetting_disposition import check_gate_record
    from vetting_evidence import ephemeris_diagnostics
    from vetting_triage import measured_event_ids

    path = RUN / "sources" / f"{sid}.json"
    row = read(path)
    for signal in row.get("signals", []):
        check_gate_record(signal)
    active = measured_event_ids(row)
    policy = load_disposition_policy()
    if policy is not None:
        row["disposition_policy"] = policy
    row["disposition_execution"] = disposition_execution(policy)
    literature = RUN / "literature" / f"{sid}.json"
    row["literature"] = (
        read(literature) if literature.exists() else {"status": "NOT_RUN"}
    )
    novelty = RUN / "novelty" / f"{sid}.json"
    if novelty.exists():
        row["novelty_review"] = read(novelty)
        for review in row["novelty_review"].values():
            verify_review_references(review)
    for mechanism in row.get("reproduced_mechanisms", []):
        verify_review_references(mechanism)
    localizations = row.get("localization", {}).get("events", [])
    for event in row["events"]:
        if event["event_id"] not in active:
            continue
        pixel = next(
            (
                record
                for record in localizations
                if event["event_id"]
                in record.get("represented_event_ids", [record["event_id"]])
            ),
            {"status": "NOT_RUN"},
        )
        event["source_association"] = {
            "status": pixel.get("attribution", {}).get("status", pixel["status"]),
            "evidence": pixel.get("evidence"),
        }
        event["known_ephemerides"] = ephemeris_diagnostics(event, row["catalogues"])
    row["localization"] = {**row.get("localization", {}), "events": localizations}
    row["disposition"] = adjudicate(row)
    row["processing_status"] = processing_status(row)
    save(path, row, compact=True)


def processing_status(row):
    from vetting_triage import measured_signals

    if row.get("signals") and not measured_signals(row):
        return "EVIDENCE_STAGES_COMPLETE"
    statuses = [p["status"] for p in row["localization"].get("events", [])]
    statuses += [e["fit"]["status"] for e in row["events"]]
    statuses += [
        row.get(name, {}).get("status", "NOT_RUN")
        for name in ("association", "orbital", "instrument", "gaia_scene", "literature")
    ]
    for signal in row.get("signals", []):
        statuses += [
            signal.get(name, {}).get("status", "NOT_RUN")
            for name in ("contamination", "host")
        ]
    return (
        "EVIDENCE_STAGES_COMPLETE"
        if statuses and "NOT_RUN" not in statuses
        else "PARTIAL"
    )


def finish_worker(sid):
    before = counter_snapshot()
    started_at = started()
    finish_source(sid)
    return {
        "source_id": sid,
        "timing": elapsed(started_at),
        "counters": counter_deltas(before),
    }


def finish():
    config = read(CONFIG)
    check_run(ROOT / config["queue"])
    ids = [proposal["source_id"] for proposal in read(RUN / "cohort.json")]
    with ProcessPoolExecutor(
        max_workers=min(config["workers"], max(1, len(ids)))
    ) as pool:
        for result in pool.map(finish_worker, ids):
            log_worker(result)
    seal_run()
    summarize()


def seal_run():
    manifest = read(RUN / "manifest.json")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["query_sha256"] = {
        p.name: digest(p) for p in (RUN / "queries").glob("*.json")
    }
    save(RUN / "manifest.json", manifest)


def run_all():
    from vetting_evidence import bibliography
    from vetting_evidence import run as evidence
    from vetting_flux import run as flux
    from vetting_pixels import localize, retrieve

    stages = [
        ("prepare", prepare),
        ("source_evidence", run),
        ("signal_association", lambda: run_program("signals", apply_triage=False)),
        ("flux_characterization", flux),
        ("orbital_measurement", lambda: run_program("orbital", apply_triage=False)),
        ("shape_gate", lambda: run_program("shape")),
        ("alias_measurement", lambda: run_program("alias", apply_triage=False)),
        ("orbital_gate", lambda: run_program("orbital_gate")),
        ("catalogue_evidence", evidence),
        ("instrument_gate", lambda: run_program("instrument")),
        ("pixel_retrieval", retrieve),
        ("pixel_localization", localize),
        ("contamination_gate", lambda: run_program("contamination")),
        ("host_interpretation", lambda: run_program("host")),
        ("bibliography", bibliography),
        ("finish", finish),
    ]
    from vetting_checkpoint import checkpoint, restore, verify_preserved

    selected = checkpoint(read(CONFIG), RUN)
    if selected is not None:
        timed_stage("restore_pre_pixel_checkpoint", lambda: restore(selected))
        stages = stages[10:]
    for name, program in stages:
        timed_stage(name, program)
    verify_preserved()


def retrieve_null_pixels():
    """Null flux has no matching TPF; never substitute its original pixels."""
    config = read(CONFIG)
    contract = null_manifest(ROOT, config.get("input_manifest"))
    if contract is None:
        return False
    check_run(ROOT / config["queue"])
    path = (ROOT / config["input_manifest"]).resolve()
    specification = read(path)
    if config.get("frozen_pixel_inputs") is not None or any(
        source.get("pixels") for source in specification["sources"].values()
    ):
        raise ValueError("Null flux cannot use original or controlled pixel inputs")
    metadata = read(RUN / "metadata.json")
    parents = {}
    for proposal in read(RUN / "cohort.json"):
        sid = proposal["source_id"]
        parent = metadata[sid]["origin_tpf_kic"]
        quarters = parents.setdefault(parent, {})
        row = read(RUN / "sources" / f"{sid}.json")
        for event in row["pixel_jobs"]:
            quarters[str(event["quarter"])] = {
                "status": "NULL_TRANSFORM_HAS_NO_MATCHING_PIXELS"
            }
    reference = {
        "path": str(path),
        "sha256": digest(path),
    }
    results = []
    for parent, quarters in parents.items():
        result = {
            "parent_kic": parent,
            "quarters": quarters,
            "null_input_manifest": reference,
            "null_construction": contract,
        }
        save(RUN / "pixel-manifests" / f"{parent}.json", result)
        results.append(result)
    save(RUN / "pixel-retrieval.json", {
        "status": "NULL_TRANSFORM_HAS_NO_MATCHING_PIXELS",
        "parents": len(results), "results": results,
        "null_input_manifest": reference, "null_construction": contract,
    })
    return True


def measure_orbital_source(source_id):
    from vetting_flux import finite
    from vetting_periodic import verify
    from vetting_triage import measured_signals

    started_at = started()
    config = read(CONFIG)
    path = RUN / "sources" / f"{source_id}.json"
    row = read(path)
    raw = observation(source_id)[1] if measured_signals(row) else None
    row = finite(verify(row, raw, config))
    save(path, row, compact=True)
    return {
        "source_id": source_id,
        "events": len(row["events"]),
        "timing": elapsed(started_at),
        "counters": {"fitted_hypotheses": row["orbital"]["evaluated_hypotheses"]},
    }


def run_pooled(measure, name, workers):
    """Measure every cohort source in parallel; results keep the cohort order."""
    source_ids = [source["source_id"] for source in read(RUN / "cohort.json")]
    dispatch = source_ids
    if name == "orbital":
        dispatch = sorted(
            source_ids,
            key=lambda sid: -len(read(RUN / "sources" / f"{sid}.json")["events"]),
        )
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {sid: pool.submit(measure, sid) for sid in dispatch}
        for sid in source_ids:
            result = futures[sid].result()
            log_worker(result)
            print(
                json.dumps(
                    {
                        "source_id": result["source_id"],
                        "program": name,
                        "status": "COMPLETE",
                    }
                ),
                flush=True,
            )
            results.append(result)
    return results


def run_orbital_program(config):
    run_pooled(measure_orbital_source, "orbital", config["workers"])


def measure_alias_source(source_id):
    from vetting_alias import measure, needs_flux
    from vetting_flux import finite
    from vetting_triage import measured_signals

    started_at = started()
    config = read(CONFIG)
    path = RUN / "sources" / f"{source_id}.json"
    row = read(path)
    needs_raw = needs_flux(row, config)
    measured = needs_raw or any(
        signal.get("period_days") is not None and len(signal["events"]) > 1
        for signal in measured_signals(row)
    )
    if measured:
        raw = observation(source_id)[1] if needs_raw else None
        save(path, finite(measure(row, raw, config)), compact=True)
    return {
        "source_id": source_id,
        "measured": measured,
        "timing": elapsed(started_at),
        "counters": {},
    }


def run_alias_program(config):
    run_pooled(measure_alias_source, "alias", config["alias"]["workers"])


def measure_program_source(name, apply_triage, source_id):
    from vetting_evidence import contamination
    from vetting_flux import finite
    from vetting_host import interpret
    from vetting_signals import associate
    from vetting_systematics import comparison_measurements
    from vetting_systematics import verify as instrument
    from vetting_triage import apply_gate, measured_signals

    programs = {
        "signals": associate,
        "shape": lambda row, raw, config: row,
        "orbital_gate": lambda row, raw, config: row,
        "instrument": instrument,
        "host": interpret,
        "contamination": contamination,
    }
    started_at, before = started(), counter_snapshot()
    config = read(CONFIG)
    path = RUN / "sources" / f"{source_id}.json"
    row = read(path)
    raw = (
        observation(source_id)[1]
        if name in {"instrument", "host"} and measured_signals(row)
        else None
    )
    row = programs[name](row, raw, config)
    if name == "instrument":
        row["instrument"]["comparisons"] = comparison_measurements(row, config)
    triage = None
    if name != "host" and apply_triage:
        gate_stage = {"shape": "signals", "orbital_gate": "orbital"}.get(name, name)
        row = apply_gate(row, gate_stage, config)
        triage = row.get("triage")
    save(path, finite(row), compact=True)
    return {
        "source_id": source_id,
        "triage": triage,
        "timing": elapsed(started_at),
        "counters": counter_deltas(before),
    }


def run_program(name, apply_triage=True):
    from vetting_triage import gate_summary

    config = read(CONFIG)
    check_run(ROOT / config["queue"])
    if name == "orbital":
        return run_orbital_program(config)
    if name == "alias":
        return run_alias_program(config)
    measure = partial(measure_program_source, name, apply_triage)
    results = run_pooled(measure, name, config["workers"])
    gate_states = [r["triage"] for r in results if r["triage"] is not None]
    if gate_states:
        summary = gate_summary(gate_states)
        stage = gate_states[-1]["last_stage"]
        save(RUN / "triage" / f"{stage}.json", summary)
        print(json.dumps({"program": name, "triage": summary}), flush=True)


def program_main(name):
    argparse.ArgumentParser(
        description=f"Run {name} on every supplied proposal"
    ).parse_args()
    run_program(name, apply_triage=name not in {"signals", "orbital", "alias"})


def review_sources(path):
    check_run(ROOT / read(CONFIG)["queue"])
    for review in read(path):
        source_path = RUN / "sources" / f"{review['source_id']}.json"
        row = read(source_path)
        verify_review_references(review)
        row["reviews"].append({k: v for k, v in review.items() if k != "source_id"})
        row["disposition"] = adjudicate(row)
        save(source_path, row, compact=True)
    summarize()


def verify_review_references(review):
    references = list(review.get("evidence", []))
    for assessment in review.get("assessments", {}).values():
        references.extend(assessment.get("evidence", []))
    for reference in references:
        path = (RUN / reference.split("#", 1)[0]).resolve()
        if RUN.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"Review evidence file is unavailable: {reference}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["prepare", "run", "summarize", "finish", "all", "review"],
    )
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.stage == "review":
        if args.reviews is None:
            raise ValueError("Review requires an evidence-backed review file")
        review_sources(args.reviews)
    else:
        {
            "prepare": prepare,
            "run": run,
            "summarize": summarize,
            "finish": finish,
            "all": run_all,
        }[args.stage]()
