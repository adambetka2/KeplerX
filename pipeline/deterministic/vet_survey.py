"""Vet every detector follow-up source from a completed bounded survey."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

from pipeline.runtime import (
    SERIAL_BLAS,
    catalogue_search_paths,
    reference_catalogue_paths,
    worker_count,
)

os.environ.update(SERIAL_BLAS)

from . import run as pipeline

MINIMUM_PIXEL_FREE_BYTES = 20 * 1024**3
REPLAY_CONTRACT = "frozen-survey-batch-replay-v1"


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_survey(survey):
    report = pipeline.read(survey / "survey.json")
    if report.get("schema") != 7 or report.get("status") != "COMPLETE":
        raise ValueError("A completed dual-detector survey is required")
    if (
        report["detector_contract"]
        != pipeline.bound_detector_contract(report["detector_contract"])
        or report["implementation"] != pipeline.implementation_hashes()
    ):
        raise ValueError("Survey detector contract or implementation changed")
    if not {"followup-diagnostics.csv", "detections.csv"} <= report["artifacts"].keys():
        raise ValueError(
            "Survey lacks frozen follow-up diagnostics; create a fresh survey"
        )
    for name, filename in (
        ("binary", "detector"),
        ("config", "detector-config.json"),
        ("shallow_config", "shallow-config.json"),
    ):
        path = survey / filename
        if (
            Path(report[name]["path"]).resolve() != path.resolve()
            or pipeline.digest(path) != report[name]["sha256"]
        ):
            raise ValueError(f"Frozen survey {name} changed")
    for name, expected in report["artifacts"].items():
        if pipeline.digest(survey / name) != expected:
            raise ValueError(f"Survey artifact changed: {name}")
    cohort = csv_rows(survey / "cohort.csv")
    proposals = csv_rows(survey / "proposals.csv")
    survivors = csv_rows(survey / "survivor-review.csv")
    shallow = csv_rows(survey / "shallow-proposals.csv")
    eligibility = pipeline.shallow_eligibility(
        survey / "shallow-eligibility.csv", [row["source_id"] for row in cohort]
    )
    if (
        len(cohort) != report["accepted_sources"]
        or len({row["source_id"] for row in cohort}) != len(cohort)
        or len(proposals) != report["proposal_rows"]
        or len({row["source_id"] for row in proposals}) != report["proposal_sources"]
        or len(survivors) != report["survivor_review_rows"]
        or len({row["source_id"] for row in survivors})
        != report["survivor_review_sources"]
        or len(shallow) != report["shallow_proposal_rows"]
        or len({row["source_id"] for row in shallow})
        != report["shallow_proposal_sources"]
        or sum(row["status"] == "INELIGIBLE" for row in eligibility)
        != report["shallow_ineligible_sources"]
    ):
        raise ValueError("Survey cohort and follow-up counts do not reconcile")
    pipeline.verify_profile_outputs(
        report["detector_contract"], proposals, survivors, shallow, eligibility
    )
    if report["detector_contract"].get("discovery_profile") and pipeline.read(
        survey / "detector-config.json"
    ) != pipeline.search_configuration(
        report["detector_contract"]["discovery_profile"]
    ):
        raise ValueError("Survey search configuration differs from its profile")
    pipeline.parse_shallow_rows(
        survey / "shallow-proposals.csv", {row["source_id"] for row in cohort}
    )
    return report, cohort, proposals, survivors, shallow


def followup_rows(survey):
    report, cohort, proposals, survivors, shallow = read_survey(survey)
    expected = defaultdict(lambda: {"rust": [], "shallow": []})
    for row in proposals:
        if row["classification"] not in {"TRANSIT_LIKE", "EVENT_LIKE"}:
            raise ValueError("Survey proposal has an invalid classification")
        expected[row["source_id"]]["rust"].append(row)
    for row in survivors:
        if pipeline.periodic_followup_lane(row) != "SURVIVOR_REVIEW":
            raise ValueError("Survey survivor has an invalid follow-up lane")
        expected[row["source_id"]]["rust"].append(row)
    for row in shallow:
        expected[row["source_id"]]["shallow"].append(row)
    if not expected or not set(expected) <= {row["source_id"] for row in cohort}:
        raise ValueError("Survey follow-up sources are absent from the cohort")
    eligibility = {
        row["source_id"]: row
        for row in pipeline.shallow_eligibility(
            survey / "shallow-eligibility.csv", [row["source_id"] for row in cohort]
        )
    }
    for sid, rows in expected.items():
        rows["shallow_eligibility"] = eligibility[sid]
        rows["discovery_profile"] = report["detector_contract"].get(
            "discovery_profile", "all-lanes-v1"
        )
    attach_diagnostics(survey, report, expected)
    ordered = [row for row in cohort if row["source_id"] in expected]
    return report, ordered, expected


def attach_diagnostics(survey, report, expected):
    discovery_view = report["detector_contract"]["discovery_view"]
    diagnostics = pipeline.parse_rows(
        survey / "followup-diagnostics.csv",
        {(sid, discovery_view) for sid in expected},
    )
    if (
        sum(map(len, diagnostics.values())) != report["followup_diagnostic_rows"]
        or len(diagnostics) != report["followup_diagnostic_sources"]
    ):
        raise ValueError("Survey follow-up diagnostics do not reconcile")
    for sid, rows in expected.items():
        rows["diagnostics"] = diagnostics[(sid, discovery_view)]
        selected = [
            row
            for row in rows["diagnostics"]
            if pipeline.admitted_row(row, rows.get("discovery_profile", "all-lanes-v1"))
        ]
        if sorted(selected, key=row_key) != sorted(rows["rust"], key=row_key):
            raise ValueError("Survey diagnostic and proposal rows differ")


def replay_plan(survey, batches, mode):
    if mode not in {"sample", "all"} or not batches:
        raise ValueError(
            "A nonempty cohort and declared detector replay mode are required"
        )
    survey_sha = pipeline.digest(survey / "survey.json")
    ranked = sorted(
        range(1, len(batches) + 1),
        key=lambda number: pipeline.json_sha256(
            {
                "contract": REPLAY_CONTRACT,
                "survey_sha256": survey_sha,
                "source_ids": [row["source_id"] for row in batches[number - 1]],
            }
        ),
    )
    count = len(batches) if mode == "all" else (len(batches) + 49) // 50
    return {
        "contract": REPLAY_CONTRACT,
        "mode": mode,
        "survey_sha256": survey_sha,
        "selection": "ascending SHA256 rank of frozen survey and ordered batch sources",
        "sample_numerator": 1,
        "sample_denominator": 50,
        "rounding": "ceiling",
        "total_batches": len(batches),
        "selected_batches": sorted(ranked[:count]),
        "actual_fraction": count / len(batches),
    }


def survey_alerts(source, rows, input_manifest, discovery_view=pipeline.DISCOVERY_VIEW):
    sid = source["source_id"]
    curve = pipeline.load_curve(pipeline.ROOT, sid, discovery_view, input_manifest)
    if (
        pipeline.curve_digest(curve) != source["input_arrays_sha256"]
        or str(curve["provenance"]["parent_tpf"]) != source["parent_tpf"]
    ):
        raise ValueError(f"Survey source arrays or parent changed: {sid}")
    records = pipeline.prepare_curve_records(
        curve,
        mask_flatten=discovery_view == pipeline.DISCOVERY_VIEW,
        relative_input=pipeline.null_manifest(pipeline.ROOT, input_manifest)
        is not None,
    )
    payload = (
        struct.pack(
            "<8sQQQ",
            b"FSC_LC2\0",
            int(sid),
            pipeline.VIEW_CODES[discovery_view],
            len(records),
        )
        + records.tobytes()
    )
    if hashlib.sha256(payload).hexdigest() != source["fsc_sha256"]:
        raise ValueError(f"Survey prepared light curve changed: {sid}")
    rust = [
        alert
        for row in sorted(rows["rust"], key=lambda row: int(row["signal_rank"]))
        for alert in pipeline.supplied_events(row, records)[0]
    ]
    periodic = [alert for alert in rust if alert["kind"] == "PERIODIC_WINDOW"]
    alerts = rust + [
        alert
        for row in sorted(rows["shallow"], key=lambda row: int(row["search_rank"]))
        for alert in pipeline.shallow_events(row, records, discovery_view)[0]
    ]
    alerts = [
        alert
        for alert in alerts
        if alert["kind"] != "SPARSE_EVENT_WINDOW"
        or not any(
            alert["photometry"] == window["photometry"]
            and alert["quarter"] == window["quarter"]
            and alert["start_bkjd"] < window["end_bkjd"]
            and window["start_bkjd"] < alert["end_bkjd"]
            for window in periodic
        )
    ]
    if not alerts or len({alert["event_id"] for alert in alerts}) != len(alerts):
        raise ValueError(f"Survey follow-up alerts are missing or duplicated: {sid}")
    return alerts


def shared_index_identity(survey, ordered):
    return {
        "contract": "survey-alert-index-v2",
        "survey_sha256": pipeline.digest(survey / "survey.json"),
        "source_ids_sha256": pipeline.json_sha256(
            [source["source_id"] for source in ordered]
        ),
        "sources": len(ordered),
    }


def initialize_shared_index(connection, identity):
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sources (
            ordinal INTEGER PRIMARY KEY, source_id TEXT UNIQUE NOT NULL,
            parent_tpf INTEGER NOT NULL, alerts_sha256 TEXT NOT NULL
        );
        CREATE TABLE alerts (
            id INTEGER PRIMARY KEY, source_id TEXT NOT NULL,
            parent_tpf INTEGER NOT NULL, quarter INTEGER NOT NULL,
            start_bkjd REAL NOT NULL, end_bkjd REAL NOT NULL
        );
        CREATE VIRTUAL TABLE windows USING rtree(
            id, start_bkjd, end_bkjd, quarter_min, quarter_max
        );
        """
    )
    connection.execute(
        "INSERT INTO metadata VALUES ('identity', ?)",
        (json.dumps(identity, sort_keys=True),),
    )
    connection.execute(
        "INSERT INTO metadata VALUES ('body_sha256', ?)",
        (hashlib.sha256().hexdigest(),),
    )
    connection.commit()


def hash_index_rows(connection, digest, source_id=None):
    query = """
        SELECT s.ordinal, s.source_id, s.parent_tpf, s.alerts_sha256,
               a.id, a.parent_tpf, a.quarter, a.start_bkjd, a.end_bkjd,
               w.id, w.start_bkjd, w.end_bkjd, w.quarter_min, w.quarter_max
        FROM sources s LEFT JOIN alerts a ON a.source_id=s.source_id
        LEFT JOIN windows w ON w.id=a.id
    """
    parameters = ()
    if source_id is not None:
        query += " WHERE s.source_id=?"
        parameters = (source_id,)
    for row in connection.execute(query + " ORDER BY s.ordinal, a.id", parameters):
        if (
            any(value is None for value in row)
            or row[2] != row[5]
            or row[4] != row[9]
            or not all(math.isfinite(value) for value in row[7:14])
            or not row[10] <= row[7] < row[8] <= row[11]
            or row[12] != row[6]
            or row[13] != row[6]
        ):
            raise ValueError(
                "Shared-alert index has missing or invalid interval records"
            )
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")


def verify_index_body(connection):
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Shared-alert index integrity check failed")
    orphaned = connection.execute("""
        SELECT COUNT(*) FROM alerts a LEFT JOIN sources s ON s.source_id=a.source_id
        WHERE s.source_id IS NULL
    """).fetchone()[0]
    counts = [
        connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in ("alerts", "windows")
    ]
    if orphaned or counts[0] != counts[1]:
        raise ValueError("Shared-alert index interval inventory changed")
    digest = hashlib.sha256()
    hash_index_rows(connection, digest)
    recorded = connection.execute(
        "SELECT value FROM metadata WHERE key='body_sha256'"
    ).fetchone()
    if recorded is None or digest.hexdigest() != recorded[0]:
        raise ValueError("Shared-alert index interval body changed")
    return digest


def commit_index_body(connection, digest):
    connection.execute(
        "UPDATE metadata SET value=? WHERE key='body_sha256'", (digest.hexdigest(),)
    )
    connection.commit()


def shared_alert_index(out, survey, report, ordered, expected):
    identity = shared_index_identity(survey, ordered)
    path = out / "shared-alerts.sqlite"
    building = out / "shared-alerts.sqlite.building"
    if path.exists() and building.exists():
        raise ValueError("Complete and incomplete shared-alert indexes coexist")
    if path.exists():
        building = path
    connection = sqlite3.connect(building)
    try:
        if not connection.execute(
            "SELECT name FROM sqlite_master WHERE name='metadata'"
        ).fetchone():
            initialize_shared_index(connection, identity)
        recorded = connection.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if recorded is None or json.loads(recorded[0]) != identity:
            raise ValueError("Shared-alert index identity differs from survey")
        body = verify_index_body(connection)
        done = [
            row[0]
            for row in connection.execute(
                "SELECT source_id FROM sources ORDER BY ordinal"
            )
        ]
        if done != [source["source_id"] for source in ordered[: len(done)]]:
            raise ValueError("Shared-alert index source prefix changed")
        if path.exists():
            if len(done) != len(ordered):
                raise ValueError("Complete shared-alert index lacks sources")
            return path
        manifest = report["selection"].get("input_manifest")
        input_manifest = manifest["path"] if manifest else None
        for ordinal, source in enumerate(ordered[len(done) :], len(done)):
            sid = source["source_id"]
            view = report["detector_contract"]["discovery_view"]
            alerts = (
                survey_alerts(source, expected[sid], input_manifest)
                if view == pipeline.DISCOVERY_VIEW
                else survey_alerts(source, expected[sid], input_manifest, view)
            )
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?)",
                (
                    ordinal,
                    sid,
                    int(source["parent_tpf"]),
                    pipeline.json_sha256(alerts),
                ),
            )
            for alert in alerts:
                cursor = connection.execute(
                    "INSERT INTO alerts "
                    "(source_id, parent_tpf, quarter, start_bkjd, end_bkjd) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        sid,
                        int(source["parent_tpf"]),
                        int(alert["quarter"]),
                        alert["start_bkjd"],
                        alert["end_bkjd"],
                    ),
                )
                connection.execute(
                    "INSERT INTO windows VALUES (?, ?, ?, ?, ?)",
                    (
                        cursor.lastrowid,
                        alert["start_bkjd"],
                        alert["end_bkjd"],
                        int(alert["quarter"]),
                        int(alert["quarter"]),
                    ),
                )
            hash_index_rows(connection, body, sid)
            if (ordinal + 1) % 100 == 0:
                commit_index_body(connection, body)
        commit_index_body(connection, body)
        verify_index_body(connection)
    finally:
        connection.close()
    building.replace(path)
    return path


def identity(survey, report, policy, args, ordered, index):
    config_path = pipeline.vetting_config_path(args)
    config = pipeline.read(config_path)
    pixels = pipeline.frozen_pixel_configuration(args).get("frozen_pixel_inputs")
    if config.get("frozen_pixel_inputs") not in (None, pixels):
        raise ValueError("The base configuration differs from the explicit frozen pixel inputs")
    primary, extra = reference_catalogue_paths(config)
    parent_index = pipeline.ROOT / config["parent_index"]
    if pipeline.digest(parent_index) != config["parent_index_sha256"]:
        raise ValueError("Configured parent index differs from its frozen identity")
    frozen = {
        "survey": str(survey),
        "survey_sha256": pipeline.digest(survey / "survey.json"),
        "survey_artifacts": report["artifacts"],
        "implementation": pipeline.implementation_hashes(),
        "disposition_policy": {
            "path": str(policy),
            "sha256": pipeline.digest(policy),
        },
        "vetting_configuration": {
            "path": str(config_path),
            "sha256": pipeline.digest(config_path),
        },
        "frozen_pixel_inputs": pixels,
        "pixel_minimum_free_bytes": getattr(args, "pixel_minimum_free_bytes", MINIMUM_PIXEL_FREE_BYTES),
        "record_every_gate": getattr(args, "record_every_gate", False),
        "detector_contract": report["detector_contract"],
        "catalogue_sha256": {
            str(path): pipeline.digest(path) for path in (*primary, *extra, *catalogue_search_paths(config))
        },
        "parent_index": {
            "path": str(parent_index),
            "sha256": config["parent_index_sha256"],
        },
        "input_manifest": report["selection"].get("input_manifest"),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "workers": args.workers,
        "concurrent_batches": args.concurrent_batches,
        "network": not args.no_network,
        "followup_sources": len(ordered),
        "followup_source_ids_sha256": pipeline.json_sha256(
            [row["source_id"] for row in ordered]
        ),
        "shared_alert_index": {"path": str(index), "sha256": pipeline.digest(index)},
        "detector_replay": args.detector_replay_plan,
    }
    checkpoint = getattr(args, "pixel_checkpoint_ledger", None)
    if checkpoint is not None:
        checkpoint = pipeline.dump_path(checkpoint)
        frozen["pixel_checkpoint_ledger"] = {
            "path": str(checkpoint), "sha256": pipeline.digest(checkpoint),
        }
    return frozen


def verify_execution_identity(args):
    frozen = args.execution_identity
    files = (
        frozen["disposition_policy"],
        frozen["vetting_configuration"],
        frozen["shared_alert_index"],
        frozen["parent_index"],
        {
            "path": str(Path(frozen["survey"]) / "survey.json"),
            "sha256": frozen["survey_sha256"],
        },
    )
    files = [
        *files,
        *(
            {"path": str(Path(frozen["survey"]) / name), "sha256": sha}
            for name, sha in frozen["survey_artifacts"].items()
        ),
        *(
            {"path": name, "sha256": sha}
            for name, sha in frozen["catalogue_sha256"].items()
        ),
    ]
    if frozen.get("pixel_checkpoint_ledger"):
        files.append(frozen["pixel_checkpoint_ledger"])
    if frozen["input_manifest"]:
        files.append(frozen["input_manifest"])
    if frozen.get("frozen_pixel_inputs"):
        binding = frozen["frozen_pixel_inputs"]
        files.append(binding)
        files.extend(
            {"path": path, "sha256": checksum}
            for path, checksum in pipeline.read(Path(binding["path"]))["inputs"].items()
        )
    if any(pipeline.digest(Path(item["path"])) != item["sha256"] for item in files):
        raise ValueError(
            "Frozen vetting policy, configuration, survey or shared index changed"
        )
    if (
        pipeline.implementation_hashes() != frozen["implementation"]
        or pipeline.bound_detector_contract(frozen["detector_contract"])
        != frozen["detector_contract"]
    ):
        raise ValueError("Frozen vetting implementation or detector contract changed")


def row_key(row):
    return row["source_id"], row["photometry"], int(row["signal_rank"])


def shallow_key(row):
    return row["source_id"], int(row["search_rank"])


def reconcile_detector(batch, expected):
    actual = csv_rows(batch / "raw.csv")
    diagnostics = [row for rows in expected.values() for row in rows["diagnostics"]]
    if len({row_key(row) for row in actual}) != len(actual) or sorted(
        actual, key=row_key
    ) != sorted(diagnostics, key=row_key):
        raise ValueError("Vetting detector diagnostics differ from the frozen survey")
    profiles = {
        rows.get("discovery_profile", "all-lanes-v1") for rows in expected.values()
    }
    if len(profiles) != 1:
        raise ValueError("Detector replay mixes discovery profiles")
    profile = profiles.pop()
    selected = [row for row in actual if pipeline.admitted_row(row, profile)]
    expected_rows = [row for rows in expected.values() for row in rows["rust"]]
    if len({row_key(row) for row in selected}) != len(selected) or sorted(
        selected, key=row_key
    ) != sorted(expected_rows, key=row_key):
        raise ValueError("Vetting detector rows differ from the frozen survey")
    actual_shallow = csv_rows(batch / "shallow-raw.csv")
    expected_shallow = [row for rows in expected.values() for row in rows["shallow"]]
    if len({shallow_key(row) for row in actual_shallow}) != len(
        actual_shallow
    ) or sorted(actual_shallow, key=shallow_key) != sorted(
        expected_shallow, key=shallow_key
    ):
        raise ValueError("Vetting shallow rows differ from the frozen survey")
    actual_eligibility = pipeline.shallow_eligibility(
        batch / "shallow-eligibility.csv", list(expected)
    )
    for row in actual_eligibility:
        frozen = expected[row["source_id"]]["shallow_eligibility"]
        if (row["status"], row["reason"]) != (
            frozen["status"],
            frozen["reason"],
        ):
            raise ValueError(
                "Vetting shallow eligibility differs from the frozen survey"
            )
    return pipeline.read(batch / "detection.json")


def verify_batch_inputs(batch, sources):
    preparation = pipeline.validate_preparation(batch)
    discovery_view = preparation["detector_contract"]["discovery_view"]
    if [row["source_id"] for row in preparation["sources"]] != [
        row["source_id"] for row in sources
    ]:
        raise ValueError("Vetting preparation differs from frozen survey source order")
    manifest = preparation.get("input_manifest")
    input_manifest = manifest["path"] if manifest else None
    for prepared, frozen in zip(preparation["sources"], sources):
        sid = frozen["source_id"]
        curve = pipeline.load_curve(pipeline.ROOT, sid, discovery_view, input_manifest)
        item = prepared["inputs"][discovery_view]
        flatten = curve["flatten"]
        flatten_sha = hashlib.sha256(
            str(flatten.dtype).encode() + flatten.tobytes()
        ).hexdigest()
        if (
            pipeline.curve_digest(curve) != frozen["input_arrays_sha256"]
            or flatten_sha != frozen["flatten_mask_sha256"]
            or any(
                str(prepared[name]) != str(frozen[name])
                for name in (
                    "parent_tpf",
                    "raw_fits_sha256",
                    "shard",
                )
            )
            or item["fsc_sha256"] != frozen["fsc_sha256"]
            or item["retained_cadences"] != int(frozen["retained_cadences"])
        ):
            raise ValueError(f"Vetting inputs differ from frozen survey: {sid}")
    return preparation


def restore_detection(batch, survey, preparation, expected):
    discovery_view = preparation["detector_contract"]["discovery_view"]
    with (survey / "followup-diagnostics.csv").open(newline="") as stream:
        fields = csv.DictReader(stream).fieldnames
    grouped = {
        (sid, discovery_view): rows["diagnostics"] for sid, rows in expected.items()
    }
    shallow = {sid: rows["shallow"] for sid, rows in expected.items()}
    tables = (
        ("raw.csv", fields, [row for rows in grouped.values() for row in rows]),
        (
            "shallow-raw.csv",
            pipeline.shallow_detector.FIELDS,
            [row for rows in shallow.values() for row in rows],
        ),
        (
            "shallow-eligibility.csv",
            pipeline.shallow_detector.ELIGIBILITY_FIELDS,
            [rows["shallow_eligibility"] for rows in expected.values()],
        ),
    )
    for name, columns, rows in tables:
        with (batch / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    grouped = pipeline.parse_rows(batch / "raw.csv", set(grouped))
    shallow = pipeline.parse_shallow_rows(batch / "shallow-raw.csv", set(expected))
    queue, ledger, artifacts = pipeline.build_queue(
        batch, preparation, grouped, shallow
    )
    artifacts += [
        batch / name
        for name in (
            "raw.csv",
            "shallow-raw.csv",
            "shallow-eligibility.csv",
            "preparation.json",
            "queue.json",
            "source-ledger.json",
        )
    ]
    binary = pipeline.read(survey / "survey.json")["binary"]
    pipeline.save(
        batch / "discovery-manifest.json",
        {
            "schema": 2,
            "origin": pipeline.bound_detector_contract(
                preparation["detector_contract"]
            )["origin"],
            "detector_contract": pipeline.bound_detector_contract(
                preparation["detector_contract"]
            ),
            "implementation": pipeline.implementation_hashes(),
            "binary": binary,
            "artifacts": {str(path): pipeline.digest(path) for path in artifacts},
            "frozen_survey": {
                "path": str(survey),
                "sha256": pipeline.digest(survey / "survey.json"),
            },
        },
    )
    report = {
        "status": "DETECTION_COMPLETE",
        "origin": "FROZEN_SURVEY_ROWS",
        "binary": binary,
        "sources": len(ledger),
        "alerted_sources": len(queue["rows"]),
        "raw_rows": sum(row["raw_rows"] for row in ledger),
        "shallow_rows": sum(row["shallow_rows"] for row in ledger),
        "shallow_ineligible_sources": sum(
            row["shallow_eligibility"]["status"] == "INELIGIBLE" for row in ledger
        ),
        "alerts": sum(row["alerts"] for row in ledger),
        "classifications": dict(
            Counter(row["classification"] for rows in grouped.values() for row in rows)
        ),
    }
    pipeline.save(batch / "detection.json", report)


def verify_replay_receipt(result, number, plan, execution_identity):
    if result.get("execution_identity_sha256") != pipeline.json_sha256(
        execution_identity
    ):
        raise ValueError(
            "Vetting batch belongs to a different frozen execution identity"
        )
    receipt = result.get("detector_replay", {})
    if receipt != {
        "plan_sha256": pipeline.json_sha256(plan),
        "selected": number in plan["selected_batches"],
        "status": "RECONCILED" if number in plan["selected_batches"] else "FROZEN_ROWS",
        "verified_input_sources": result["sources"],
    }:
        raise ValueError(
            "Vetting batch detector replay receipt differs from frozen plan"
        )


def reconcile_shared_alerts(batch, index, sources):
    queue = pipeline.read(batch / "queue.json")
    parents = {source["source_id"]: str(source["parent_tpf"]) for source in sources}
    ledger = pipeline.read(batch / "source-ledger.json")
    if [row["source_id"] for row in ledger] != list(parents):
        raise ValueError("Vetting ledger differs from shared-alert source cohort")
    accepted = []
    alert_count = 0
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        for item in ledger:
            sid = item["source_id"]
            proposal = pipeline.read(batch / "proposals" / f"{sid}.json")
            recorded = connection.execute(
                "SELECT parent_tpf, alerts_sha256 FROM sources WHERE source_id=?",
                (sid,),
            ).fetchone()
            if (
                recorded is None
                or str(recorded[0]) != parents[sid]
                or recorded[1] != pipeline.json_sha256(proposal["ungated_alerts"])
                or item["alerts"] != len(proposal["alerts"])
            ):
                raise ValueError(f"Vetting alerts differ from survey-wide index: {sid}")
            if proposal["alerts"]:
                accepted.append({"source_id": sid, "evidence": f"proposals/{sid}.json"})
                alert_count += len(proposal["alerts"])
    detection = pipeline.read(batch / "detection.json")
    if (
        queue["rows"] != accepted
        or detection["sources"] != len(parents)
        or detection["alerted_sources"] != len(accepted)
        or detection["alerts"] != alert_count
    ):
        raise ValueError("Vetting queue and gated-source counts do not reconcile")


def gzip_file(path):
    original_sha = pipeline.digest(path)
    target = path.with_suffix(path.suffix + ".gz")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with (
        path.open("rb") as source,
        temporary.open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0, compresslevel=6) as packed,
    ):
        shutil.copyfileobj(source, packed, 1024 * 1024)
    temporary.replace(target)
    with gzip.open(target, "rb") as packed:
        if hashlib.file_digest(packed, "sha256").hexdigest() != original_sha:
            raise ValueError(f"Compressed evidence changed: {path}")
    return {"source_sha256": original_sha, "archive_sha256": pipeline.digest(target)}


def relink_evidence(batch, sources):
    replacements = {str(path): str(path) + ".gz" for path in sources}
    summaries = {}
    for name in ("results.json", "bois.json", "surviving-signals.json", "signal-ledger.json"):
        path = batch / "vetting" / name
        before = pipeline.digest(path)
        value = pipeline.read(path)
        rows = value["rows"] if name == "results.json" else value
        if name == "results.json" and len(rows) != len(sources):
            raise ValueError("Vetting result index omits source evidence")
        for row in rows:
            evidence = row.get("evidence")
            if evidence not in replacements:
                raise ValueError(f"Vetting summary has unknown evidence: {evidence}")
            row["evidence"] = replacements[evidence]
        pipeline.save(path, value)
        summaries[name] = {
            "before_sha256": before,
            "after_sha256": pipeline.digest(path),
            "evidence_links": len(rows),
        }
    return summaries


def cached_pixel_references(batch, cache, namespace=None):
    referenced = {}
    for manifest in (batch / "vetting/pixel-manifests").glob("*.json"):
        for pixel in pipeline.read(manifest)["quarters"].values():
            if pixel.get("status") != "AVAILABLE":
                continue
            path = Path(pixel["path"])
            if namespace is not None and not path.resolve().is_relative_to(namespace.resolve()):
                raise ValueError("Frozen pixel product lies outside its cache namespace")
            if cache.resolve() in path.resolve().parents:
                if not pixel.get("sha256") or not (pixel.get("url") or pixel.get("transport_uri")):
                    raise ValueError("Downloaded pixel lacks archive provenance")
                previous = referenced.setdefault(path.resolve(), pixel)
                if previous["sha256"] != pixel["sha256"]:
                    raise ValueError("Batch pixel has conflicting hashes")
    return referenced


def pixel_cache_products(cache, config):
    if cache.is_symlink() or not cache.is_dir():
        raise ValueError("Batch pixel cache is missing or redirects its writes")
    binding = config.get("frozen_pixel_inputs")
    folders = [cache]
    if binding is not None:
        checksum = binding["sha256"]
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Frozen pixel cache identity is invalid")
        namespace = cache / f"frozen-{checksum}"
        if sorted(cache.iterdir()) != [namespace] or namespace.is_symlink() or not namespace.is_dir():
            raise ValueError("Frozen pixel cache has an unexpected namespace")
        folders = [namespace]
    products = []
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Unexpected batch pixel cache entry remains")
            if path.name.endswith(".fits.gz"):
                products.append(path)
            elif not path.name.endswith(".lock"):
                raise ValueError("Unexpected batch pixel cache entry remains")
    return products


def evict_batch_pixels(batch):
    cache = batch / "pixel-cache"
    config = pipeline.read(batch / "vetting.json")
    if Path(config["pixel_cache_directory"]).resolve() != cache.resolve():
        raise ValueError("Vetting did not use the batch pixel cache")
    products = pixel_cache_products(cache, config)
    binding = config.get("frozen_pixel_inputs")
    namespace = cache / f"frozen-{binding['sha256']}" if binding else None
    referenced = cached_pixel_references(batch, cache, namespace)
    rows = []
    for path in products:
        pixel = referenced.get(path.resolve())
        digest = pipeline.digest(path)
        if binding is not None and path.name != f"{digest}.fits.gz":
            raise ValueError("Frozen cached pixel filename differs from its content identity")
        if pixel is not None and digest != pixel["sha256"]:
            raise ValueError("Batch pixel differs from recorded archive product")
        rows.append(
            {
                "filename": str(path.relative_to(cache)),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "url": pixel.get("url") if pixel is not None else None,
                "transport_uri": pixel.get("transport_uri") if pixel is not None else None,
            }
        )
    if set(referenced) - {path.resolve() for path in products}:
        raise ValueError("Referenced batch pixel is missing")
    report = {
        "schema": 1,
        "products": rows,
        "bytes_evicted": sum(row["bytes"] for row in rows),
    }
    pipeline.save(batch / "cache-eviction.json", report)
    shutil.rmtree(cache)
    return report


def compact(batch, detection):
    source_dir = batch / "vetting/sources"
    paths = sorted(source_dir.glob("*.json"))
    if len(paths) != detection["alerted_sources"]:
        raise ValueError("Vetting source evidence count differs from detector queue")
    files = {}
    packed_paths = [
        *paths,
        batch / "vetting/cohort.json",
        batch / "vetting/metadata.json",
        batch / "vetting/scene.json",
    ]
    for path in packed_paths:
        if not path.is_file():
            raise ValueError(f"Missing vetting evidence: {path}")
        files[str(path.relative_to(batch))] = gzip_file(path)
    summaries = relink_evidence(batch, paths)
    prepared_inputs = {
        item["fsc_path"]: item["fsc_sha256"]
        for source in pipeline.read(batch / "preparation.json")["sources"]
        for item in source["inputs"].values()
    }
    for path in packed_paths:
        path.unlink()
    shutil.rmtree(batch / "inputs")
    evict_batch_pixels(batch)
    retained = {
        str(path.relative_to(batch)): pipeline.digest(path)
        for path in sorted(batch.rglob("*"))
        if path.is_file() and path.suffix != ".gz"
    }
    archive = {
        "schema": 1,
        "files": files,
        "prepared_inputs": prepared_inputs,
        "relinked_summaries": summaries,
        "retained_files": retained,
    }
    pipeline.save(batch / "compact.json", archive)
    return archive


def verify_archived_files(batch, archive):
    for name, checks in archive["files"].items():
        if pipeline.digest(batch / (name + ".gz")) != checks["archive_sha256"]:
            raise ValueError(f"Compressed vetting evidence changed: {name}")
        with gzip.open(batch / (name + ".gz"), "rb") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != checks["source_sha256"]
            ):
                raise ValueError(f"Archived original vetting evidence changed: {name}")


def verify_relinked_summaries(batch, archive):
    archived_sources = {
        str(batch / (name + ".gz"))
        for name in archive["files"]
        if name.startswith("vetting/sources/")
    }
    for name, checks in archive["relinked_summaries"].items():
        path = batch / "vetting" / name
        if pipeline.digest(path) != checks["after_sha256"]:
            raise ValueError(f"Relinked vetting summary changed: {name}")
        value = pipeline.read(path)
        rows = value["rows"] if name == "results.json" else value
        if len(rows) != checks["evidence_links"] or any(
            row.get("evidence") not in archived_sources for row in rows
        ):
            raise ValueError(f"Vetting evidence links do not reconcile: {name}")


def verify_archive(batch, record):
    if pipeline.digest(batch / "batch-complete.json") != record["sha256"]:
        raise ValueError(f"Completed vetting batch changed: {batch}")
    result = pipeline.read(batch / "batch-complete.json")
    if result["status"] != "COMPLETE" or result["batch"] != record["batch"]:
        raise ValueError("Completed vetting batch has an invalid status")
    archive = pipeline.read(batch / "compact.json")
    if pipeline.json_sha256(archive) != result["archive_sha256"]:
        raise ValueError("Vetting archive manifest changed")
    verify_archived_files(batch, archive)
    verify_relinked_summaries(batch, archive)
    if (batch / "inputs").exists():
        raise ValueError("Completed vetting batch retains transient inputs")
    if (batch / "pixel-cache").exists():
        raise ValueError("Completed vetting batch retains transient pixel cache")
    retained = archive["retained_files"]
    if any(pipeline.digest(batch / name) != sha for name, sha in retained.items()):
        raise ValueError("Retained vetting evidence changed")
    expected_files = set(retained) | {name + ".gz" for name in archive["files"]}
    expected_files |= {"compact.json", "batch-complete.json"}
    actual_files = {
        str(path.relative_to(batch)) for path in batch.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Vetting batch artifact set changed")
    return result


def verify_completed_inventory(batch, archive, sources, queued):
    required = {
        "preparation.json",
        "discovery-manifest.json",
        "detection.json",
        "queue.json",
        "source-ledger.json",
        "raw.csv",
        "shallow-raw.csv",
        "shallow-eligibility.csv",
        "vetting.json",
        "result.json",
        "cache-eviction.json",
        "vetting/manifest.json",
        "vetting/results.json",
        "vetting/bois.json",
        "vetting/surviving-signals.json",
        "vetting/signal-ledger.json",
        *(f"proposals/{source['source_id']}.json" for source in sources),
    }
    packed = {"vetting/cohort.json", "vetting/metadata.json", "vetting/scene.json"}
    packed |= {f"vetting/sources/{sid}.json" for sid in queued}
    prepared_inputs = {
        item["fsc_path"]: item["fsc_sha256"]
        for source in sources
        for item in source["inputs"].values()
    }
    if (
        not sources
        or not required <= set(archive["retained_files"])
        or set(archive["files"]) != packed
        or archive["prepared_inputs"] != prepared_inputs
        or set(archive["relinked_summaries"])
        != {"results.json", "bois.json", "surviving-signals.json", "signal-ledger.json"}
    ):
        raise ValueError(
            "Completed vetting batch lacks its required evidence inventory"
        )
    with gzip.open(batch / "vetting/cohort.json.gz", "rt") as stream:
        cohort = json.load(stream)
        if cohort != [pipeline.read(batch / f"proposals/{sid}.json") for sid in queued]:
            raise ValueError("Archived vetting cohort differs from the gated queue")
    for sid in queued:
        with gzip.open(batch / f"vetting/sources/{sid}.json.gz", "rt") as stream:
            evidence = json.load(stream)
        if (
            evidence.get("source_id") != sid
            or evidence.get("processing_status") != "EVIDENCE_STAGES_COMPLETE"
            or evidence.get("alerts")
            != pipeline.read(batch / f"proposals/{sid}.json")["alerts"]
            or {event["alert_id"] for event in evidence.get("events", [])}
            != {alert["event_id"] for alert in evidence["alerts"]}
        ):
            raise ValueError("Archived vetting source is missing or incomplete")


def verify_completed_counts(batch, result, sources, queued):
    ids = [source["source_id"] for source in sources]
    ledger = pipeline.read(batch / "source-ledger.json")
    detection = pipeline.read(batch / "detection.json")
    summary = pipeline.read(batch / "result.json")
    indexed = pipeline.read(batch / "vetting/results.json")
    if (
        len(set(ids)) != len(ids)
        or result["sources"] != len(ids)
        or result["source_ids_sha256"] != pipeline.json_sha256(ids)
        or [row["source_id"] for row in ledger] != ids
        or len(set(queued)) != len(queued)
        or not set(queued) <= set(ids)
        or [row["source_id"] for row in ledger if row["alerts"]] != queued
        or detection["sources"] != len(ids)
        or detection["alerted_sources"] != len(queued)
        or result["alerts"] != detection["alerts"]
        or detection["alerts"] != sum(row["alerts"] for row in ledger)
        or summary["status"] != "EVIDENCE_STAGES_COMPLETE"
        or indexed["processing_status"] != "EVIDENCE_STAGES_COMPLETE"
        or indexed["sources"] != len(queued)
        or indexed["original_alerts_preserved"] != detection["alerts"]
        or [row["source_id"] for row in indexed["rows"]] != queued
        or result["boi_signals"] != summary["boi_signals"]
        or result["unresolved_signals"] != summary["unresolved_signals"]
    ):
        raise ValueError(
            "Completed vetting batch source or outcome counts do not reconcile"
        )


def verify_completed_bindings(batch, execution_identity, queued, result):
    config = pipeline.read(batch / "vetting.json")
    manifest = pipeline.read(batch / "vetting/manifest.json")
    frozen_policy = execution_identity["disposition_policy"]
    if (
        result.get("execution_identity_sha256")
        != pipeline.json_sha256(execution_identity)
        or (pipeline.ROOT / config["disposition_policy"]).resolve()
        != Path(frozen_policy["path"]).resolve()
        or config["shared_alert_index"] != execution_identity["shared_alert_index"]
        or config["quality_bitmask"] != -1
        or config.get("frozen_pixel_inputs") != execution_identity.get("frozen_pixel_inputs")
        or config.get("pixel_minimum_free_bytes", MINIMUM_PIXEL_FREE_BYTES)
        != execution_identity.get("pixel_minimum_free_bytes", MINIMUM_PIXEL_FREE_BYTES)
        or execution_identity.get("record_every_gate")
        and config.get("triage_execution") != "RECORD_EVERY_GATE"
        or manifest["source_ids"] != queued
        or manifest["config_sha256"] != pipeline.digest(batch / "vetting.json")
        or manifest["code_sha256"] != execution_identity["implementation"]
        or manifest["catalogue_sha256"] != execution_identity["catalogue_sha256"]
        or manifest["discovery"]["manifest_sha256"]
        != pipeline.digest(batch / "discovery-manifest.json")
        or manifest["input_sha256"].get(
            str(pipeline.ROOT / config["disposition_policy"])
        )
        != frozen_policy["sha256"]
    ):
        raise ValueError(
            "Completed vetting batch differs from its frozen execution bindings"
        )
    if any(
        pipeline.digest(Path(name)) != sha
        for name, sha in manifest["input_sha256"].items()
    ):
        raise ValueError("Completed vetting batch original input changed")


def verify_batch(batch, record, execution_identity):
    result = verify_archive(batch, record)
    archive = pipeline.read(batch / "compact.json")
    minimum = {"preparation.json", "queue.json", "source-ledger.json"}
    if not minimum <= set(archive["retained_files"]):
        raise ValueError(
            "Completed vetting batch lacks its required evidence inventory"
        )
    sources = pipeline.read(batch / "preparation.json")["sources"]
    queued = [row["source_id"] for row in pipeline.read(batch / "queue.json")["rows"]]
    verify_completed_inventory(batch, archive, sources, queued)
    verify_completed_counts(batch, result, sources, queued)
    verify_completed_bindings(batch, execution_identity, queued, result)
    return result


def detect_batch(out, survey, number, sources, expected, args, index):
    verify_execution_identity(args)
    batch = out / "batches" / f"{number:06d}"
    if batch.exists():
        shutil.rmtree(batch)
    source_ids = out / "identities" / f"{number:06d}.json"
    ids = [row["source_id"] for row in sources]
    pipeline.save(source_ids, ids)
    survey_report = pipeline.read(survey / "survey.json")
    if (
        pipeline.digest(survey / "survey.json")
        != args.detector_replay_plan["survey_sha256"]
        or pipeline.digest(survey / "detector") != survey_report["binary"]["sha256"]
    ):
        raise ValueError("Frozen survey or detector changed during vetting")
    manifest = survey_report["selection"].get("input_manifest")
    input_manifest = Path(manifest["path"]) if manifest else None
    if manifest and pipeline.digest(input_manifest) != manifest["sha256"]:
        raise ValueError("Survey input manifest changed")
    started = time.monotonic()
    pipeline.prepare(
        SimpleNamespace(
            shards=sorted({int(row["shard"]) for row in sources}),
            limit=len(ids),
            source_ids=source_ids,
            discovery_view=survey_report["detector_contract"]["discovery_view"],
            discovery_profile=survey_report["detector_contract"].get(
                "discovery_profile", "all-lanes-v1"
            ),
            input_manifest=input_manifest,
            out=batch,
            disposition_policy=args.disposition_policy,
            frozen_pixel_inputs=getattr(args, "frozen_pixel_inputs", None),
            pixel_input_role=getattr(args, "pixel_input_role", None),
            pixel_cache_maximum_bytes=getattr(args, "pixel_cache_maximum_bytes", None),
        )
    )
    preparation = verify_batch_inputs(batch, sources)
    selected = {sid: expected[sid] for sid in ids}
    replayed = number in args.detector_replay_plan["selected_batches"]
    if replayed:
        pipeline.scan(
            SimpleNamespace(
                run=batch,
                binary=survey / "detector",
                threads=args.threads,
                disposition_policy=args.disposition_policy,
                frozen_pixel_inputs=getattr(args, "frozen_pixel_inputs", None),
                pixel_input_role=getattr(args, "pixel_input_role", None),
                pixel_cache_maximum_bytes=getattr(
                    args, "pixel_cache_maximum_bytes", None
                ),
            )
        )
    else:
        restore_detection(batch, survey, preparation, selected)
    detection = reconcile_detector(batch, selected)
    detection["detector_replay"] = {
        "plan_sha256": pipeline.json_sha256(args.detector_replay_plan),
        "selected": replayed,
        "status": "RECONCILED" if replayed else "FROZEN_ROWS",
        "verified_input_sources": len(ids),
    }
    pipeline.save(batch / "detection.json", detection)
    reconcile_shared_alerts(batch, index, sources)
    return batch, ids, detection, started


def vet_batch(number, batch, ids, detection, started, args, policy, index):
    verify_execution_identity(args)
    pipeline.vet(
        SimpleNamespace(
            run=batch,
            python=Path(sys.executable),
            workers=args.workers,
            no_network=args.no_network,
            disposition_policy=policy,
            vetting_config=getattr(args, "vetting_config", None),
            record_every_gate=getattr(args, "record_every_gate", False),
            pixel_cache_directory=batch / "pixel-cache",
            pixel_minimum_free_bytes=getattr(args, "pixel_minimum_free_bytes", MINIMUM_PIXEL_FREE_BYTES),
            frozen_pixel_inputs=getattr(args, "frozen_pixel_inputs", None),
            pixel_input_role=getattr(args, "pixel_input_role", None),
            pixel_cache_maximum_bytes=getattr(args, "pixel_cache_maximum_bytes", None),
            shared_alert_index=index,
            pixel_checkpoint_ledger=getattr(args, "pixel_checkpoint_ledger", None),
        )
    )
    result = pipeline.read(batch / "result.json")
    if result["status"] != "EVIDENCE_STAGES_COMPLETE":
        raise ValueError("Vetting batch did not finish completely")
    verify_execution_identity(args)
    archive = compact(batch, detection)
    completed = {
        "schema": 1,
        "status": "COMPLETE",
        "batch": number,
        "sources": len(ids),
        "source_ids_sha256": pipeline.json_sha256(ids),
        "alerts": detection["alerts"],
        "boi_signals": result["boi_signals"],
        "unresolved_signals": result["unresolved_signals"],
        "wall_seconds": round(time.monotonic() - started, 3),
        "archive_sha256": pipeline.json_sha256(archive),
        "detector_replay": detection["detector_replay"],
        "execution_identity_sha256": pipeline.json_sha256(args.execution_identity),
    }
    pipeline.save(batch / "batch-complete.json", completed, durable=True)
    return verify_batch(
        batch,
        {"batch": number, "sha256": pipeline.digest(batch / "batch-complete.json")},
        args.execution_identity,
    )


def run_batches(out, survey, batches, first, expected, args, policy, index):
    """Yield completed batches in order while vetting several at once.

    Detection runs here, one batch at a time, because it shares cached LC2
    readers. Each vetting stage is its own process and waits mostly on stage
    barriers, the network and single long sources, so batches overlap it.
    """
    pending = {}
    with ThreadPoolExecutor(args.concurrent_batches) as executor:
        for number in range(first, len(batches) + 1):
            batch = out / "batches" / f"{number:06d}"
            complete = batch / "batch-complete.json"
            if complete.exists():
                row = {"batch": number, "sha256": pipeline.digest(complete)}
                task = (verify_batch, batch, row, args.execution_identity)
            else:
                detected = detect_batch(
                    out, survey, number, batches[number - 1], expected, args, index
                )
                task = (vet_batch, number, *detected, args, policy, index)
            running = {future for future in pending.values() if not future.done()}
            while len(running) >= args.concurrent_batches:
                _, running = wait(running, return_when=FIRST_COMPLETED)
            for future in pending.values():
                if future.done() and future.exception():
                    raise future.exception()
            pending[number] = executor.submit(*task)
            while pending and pending[min(pending)].done():
                done = min(pending)
                yield done, pending.pop(done).result()
        for done in sorted(pending):
            yield done, pending.pop(done).result()


def resume_vetting_report(out, frozen, batches, args):
    report_path = out / "vet-survey.json"
    if not args.resume or not report_path.exists():
        if args.resume and any((out / "batches").glob("*")):
            raise ValueError(
                "Vetting resume lacks its frozen report but contains batch artifacts"
            )
        return {"schema": 1, "status": "RUNNING", "identity": frozen, "batches": []}
    report = pipeline.read(report_path)
    if report["identity"] != frozen or report["status"] not in {"RUNNING", "FAILED"}:
        raise ValueError("Vetting resume identity or status changed")
    if len(report["batches"]) > len(batches):
        raise ValueError("Vetting report exceeds the frozen source cohort")
    for number, entry in enumerate(report["batches"], start=1):
        if entry["batch"] != number:
            raise ValueError("Vetting batch sequence changed")
        batch = out / "batches" / f"{entry['batch']:06d}"
        result = verify_batch(batch, entry, frozen)
        verify_replay_receipt(result, number, args.detector_replay_plan, frozen)
        ids = [source["source_id"] for source in batches[number - 1]]
        if result["source_ids_sha256"] != pipeline.json_sha256(ids):
            raise ValueError("Completed vetting batch source identities changed")
    completed = sum(len(rows) for rows in batches[: len(report["batches"])])
    if report.get("completed_sources", 0) != completed:
        raise ValueError("Vetting resume source count differs from batches")
    report["status"] = "RUNNING"
    report.pop("error", None)
    return report


def vet_survey(args):
    if min(args.batch_size, args.threads, args.workers, args.concurrent_batches) < 1:
        raise ValueError(
            "Positive batch size, thread, worker and concurrent batch counts required"
        )
    survey, out = pipeline.dump_path(args.survey), pipeline.dump_path(args.run)
    policy = (pipeline.ROOT / args.disposition_policy).resolve()
    survey_report, ordered, expected = followup_rows(survey)
    args.discovery_view = survey_report["detector_contract"]["discovery_view"]
    if args.resume:
        if not out.is_dir():
            raise ValueError("Vetting run directory is missing")
    else:
        out.mkdir(parents=True, exist_ok=False)
    index = shared_alert_index(out, survey, survey_report, ordered, expected)
    batches = [
        ordered[start : start + args.batch_size]
        for start in range(0, len(ordered), args.batch_size)
    ]
    args.detector_replay_plan = replay_plan(survey, batches, args.detector_replay)
    frozen = identity(survey, survey_report, policy, args, ordered, index)
    args.execution_identity = frozen
    report_path = out / "vet-survey.json"
    report = resume_vetting_report(out, frozen, batches, args)
    pipeline.save(report_path, report, durable=True)
    try:
        first = len(report["batches"]) + 1
        for number, result in run_batches(
            out, survey, batches, first, expected, args, policy, index
        ):
            verify_replay_receipt(result, number, args.detector_replay_plan, frozen)
            complete = out / "batches" / f"{number:06d}" / "batch-complete.json"
            row = {"batch": number, "sha256": pipeline.digest(complete)}
            if result["source_ids_sha256"] != pipeline.json_sha256(
                [source["source_id"] for source in batches[number - 1]]
            ):
                raise ValueError("Vetting batch source identities changed")
            report["batches"].append(row)
            report["completed_sources"] = sum(
                len(batch_rows) for batch_rows in batches[:number]
            )
            pipeline.save(report_path, report, durable=True)
        report["status"] = "COMPLETE"
        verify_execution_identity(args)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pipeline.save(report_path, report, durable=True)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--threads", type=int, default=worker_count())
    parser.add_argument("--workers", type=int, default=worker_count())
    parser.add_argument("--concurrent-batches", type=int, default=1)
    parser.add_argument(
        "--detector-replay",
        choices=("sample", "all"),
        default="sample",
        help="Replay the frozen 2%% batch sample, or every batch for paired checks",
    )
    parser.add_argument(
        "--disposition-policy", type=Path, default=pipeline.DISPOSITION_POLICY
    )
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--vetting-config", type=Path)
    parser.add_argument("--pixel-checkpoint-ledger", type=Path)
    parser.add_argument("--record-every-gate", action="store_true")
    parser.add_argument("--frozen-pixel-inputs", type=Path)
    parser.add_argument("--pixel-input-role", choices=("development", "evaluation"))
    parser.add_argument("--pixel-cache-maximum-bytes", type=int)
    parser.add_argument("--pixel-minimum-free-bytes", type=int, default=MINIMUM_PIXEL_FREE_BYTES)
    parser.add_argument("--resume", action="store_true")
    vet_survey(parser.parse_args())


if __name__ == "__main__":
    main()
