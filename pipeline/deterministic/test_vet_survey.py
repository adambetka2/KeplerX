"""Sampled replay preserves complete diagnostic evidence and frozen inputs."""

import copy
import csv
import gzip
import hashlib
import json
import sqlite3
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deterministic import cloud
from deterministic import run as pipeline
from deterministic import test_run as contracts
from deterministic import vet_survey as vetter


def write_rows(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw = contracts.raw.__wrapped__()
    arrays = tmp_path / "123.npz"
    np.savez(arrays, **raw)
    manifest = tmp_path / "inputs.json"
    pipeline.save(
        manifest,
        {
            "schema": 1,
            "measurement_scope": "opened replay contract fixture",
            "sources": {
                "123": {
                    "arrays": arrays.name,
                    "sha256": pipeline.digest(arrays),
                    "metadata": {
                        "source_id": "123",
                        "shard": 0,
                        "shard_path": str(tmp_path),
                        "origin_tpf_kic": 10,
                        "raw_fits_sha256": "a" * 64,
                    },
                }
            },
        },
    )
    ids = tmp_path / "source-ids.json"
    pipeline.save(ids, ["123"])
    monkeypatch.setattr(pipeline, "dump_path", lambda path: Path(path))
    batch = tmp_path / "batch"
    pipeline.prepare(
        SimpleNamespace(
            shards=[0],
            limit=1,
            source_ids=ids,
            input_manifest=manifest,
            out=batch,
        )
    )
    preparation = pipeline.read(batch / "preparation.json")
    source = preparation["sources"][0]
    frozen = {
        name: source[name]
        for name in (
            "source_id",
            "parent_tpf",
            "raw_fits_sha256",
            "flatten_mask_sha256",
            "shard",
        )
    }
    frozen.update(
        input_arrays_sha256=pipeline.curve_digest(
            pipeline.load_curve(pipeline.ROOT, "123", "psf_flat", manifest)
        ),
        fsc_sha256=source["inputs"]["psf_flat"]["fsc_sha256"],
        retained_cadences=str(source["inputs"]["psf_flat"]["retained_cadences"]),
    )
    survey = tmp_path / "survey"
    survey.mkdir()
    binary = survey / "detector"
    binary.write_bytes(b"frozen detector fixture")
    pipeline.save(
        survey / "survey.json",
        {
            "binary": {"path": str(binary), "sha256": pipeline.digest(binary)},
            "detector_contract": pipeline.detector_contract(),
        },
    )
    return batch, survey, preparation, frozen


@pytest.mark.parametrize("view", ["psf_flat", "psf_qdiag", "sap_qdiag"])
def test_bounded_view_preparation_matches_standalone_and_rejects_mask_changes(
    prepared, view
):
    batch, survey, preparation, _ = prepared
    manifest = Path(preparation["input_manifest"]["path"])
    standalone = survey.parent / f"standalone-{view}"
    pipeline.prepare(
        SimpleNamespace(
            shards=[0],
            limit=1,
            source_ids=batch.parent / "source-ids.json",
            input_manifest=manifest,
            out=standalone,
            discovery_view=view,
        )
    )
    bounded = survey.parent / f"bounded-{view}"
    (bounded / "inputs").mkdir(parents=True)
    frozen = pipeline.prepare_discovery_source(
        bounded, {"source_id": "123", "shard": 0}, manifest, view
    )
    actual = vetter.verify_batch_inputs(standalone, [frozen])
    assert actual["sources"][0]["inputs"][view]["fsc_sha256"] == frozen["fsc_sha256"]
    frozen["flatten_mask_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Vetting inputs differ"):
        vetter.verify_batch_inputs(standalone, [frozen])


def expectations(kind):
    eb = contracts.rust_row(1, "EB_LIKE", period="1")
    if kind == "shallow":
        rust, shallow = [eb], [contracts.shallow_row()]
    elif kind == "sparse":
        rust = [eb, dict(contracts.rust_row(2, "EVENT_LIKE", period=""), t0_bkjd="1.5")]
        shallow = []
    elif kind == "survivor":
        rust = [
            eb,
            dict(contracts.rust_row(2, "SYSTEMATIC_LIKE"), quarter_loo_status="FAIL"),
        ]
        shallow = []
    else:
        rust, shallow = [eb, contracts.rust_row(2)], [contracts.shallow_row()]
    fields = list(dict.fromkeys(key for row in rust for key in row))
    rust = [{key: row.get(key, "") for key in fields} for row in rust]
    return {
        "123": {
            "diagnostics": rust,
            "rust": [
                row
                for row in rust
                if row["classification"] == "EVENT_LIKE"
                or pipeline.periodic_followup_lane(row) is not None
            ],
            "shallow": shallow,
            "shallow_eligibility": {
                "source_id": "123",
                "status": "ELIGIBLE",
                "reason": "",
                "elapsed_seconds": "0.1",
            },
        }
    }


def execution_identity(survey, index):
    paths = {
        "disposition_policy": pipeline.ROOT / pipeline.DISPOSITION_POLICY,
        "vetting_configuration": pipeline.ROOT / "pipeline/vetting/config.json",
        "shared_alert_index": index,
        "parent_index": index,
    }
    report = pipeline.read(survey / "survey.json")
    reference = survey / "catalogue.json"
    if not reference.exists():
        pipeline.save(reference, {"scope": "opened test fixture"})
    return {
        "survey": str(survey),
        "survey_sha256": pipeline.digest(survey / "survey.json"),
        "implementation": pipeline.implementation_hashes(),
        "detector_contract": pipeline.detector_contract(),
        "survey_artifacts": {"detector": pipeline.digest(survey / "detector")},
        "input_manifest": report.get("selection", {}).get("input_manifest"),
        "catalogue_sha256": {str(reference): pipeline.digest(reference)},
        **{
            name: {"path": str(path), "sha256": pipeline.digest(path)}
            for name, path in paths.items()
        },
    }


@pytest.mark.parametrize("kind", ["periodic", "shallow", "sparse", "survivor"])
def test_frozen_rows_preserve_full_queue_and_binary_diagnostics(prepared, kind):
    batch, survey, preparation, frozen = prepared
    expected = expectations(kind)
    rows = expected["123"]["diagnostics"]
    write_rows(survey / "followup-diagnostics.csv", list(rows[0]), rows)
    write_rows(survey / "detections.csv", list(rows[0]), rows)
    write_rows(
        batch / "shallow-eligibility.csv",
        pipeline.shallow_detector.ELIGIBILITY_FIELDS,
        [expected["123"]["shallow_eligibility"]],
    )
    pipeline.build_queue(
        batch,
        preparation,
        {("123", "psf_flat"): rows},
        {"123": expected["123"]["shallow"]},
    )
    baseline = pipeline.read(batch / "proposals/123.json")
    vetter.verify_batch_inputs(batch, [frozen])
    vetter.restore_detection(batch, survey, preparation, expected)
    assert pipeline.read(batch / "proposals/123.json") == baseline
    assert baseline["rust_rows"][0]["classification"] == "EB_LIKE"
    report = vetter.reconcile_detector(batch, expected)
    assert report["raw_rows"] == len(rows)
    pipeline.verify_discovery(
        batch / "queue.json",
        pipeline.read(batch / "discovery-manifest.json"),
        {"discovery_contract": pipeline.detector_contract()},
    )


@pytest.mark.parametrize(
    "field",
    [
        "input_arrays_sha256",
        "fsc_sha256",
        "flatten_mask_sha256",
        "raw_fits_sha256",
        "parent_tpf",
        "shard",
        "retained_cadences",
    ],
)
def test_every_batch_rejects_changed_frozen_inputs(prepared, field):
    batch, _, _, frozen = prepared
    frozen[field] = (
        "99" if field in {"parent_tpf", "shard", "retained_cadences"} else "0" * 64
    )
    with pytest.raises(ValueError, match="inputs differ from frozen survey"):
        vetter.verify_batch_inputs(batch, [frozen])


def test_prepared_file_mutation_is_rejected(prepared):
    batch, _, preparation, frozen = prepared
    path = Path(preparation["sources"][0]["inputs"]["psf_flat"]["fsc_path"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Prepared input changed"):
        vetter.verify_batch_inputs(batch, [frozen])


def test_replay_reconciles_nonproposal_diagnostics(prepared):
    batch, survey, preparation, _ = prepared
    expected = expectations("periodic")
    rows = expected["123"]["diagnostics"]
    write_rows(survey / "followup-diagnostics.csv", list(rows[0]), rows)
    write_rows(survey / "detections.csv", list(rows[0]), rows)
    vetter.restore_detection(batch, survey, preparation, expected)
    changed = copy.deepcopy(rows)
    changed[0]["search_mes"] = "90"
    write_rows(batch / "raw.csv", list(changed[0]), changed)
    with pytest.raises(ValueError, match="diagnostics differ from the frozen survey"):
        vetter.reconcile_detector(batch, expected)


@pytest.mark.parametrize(
    "count,selected", [(1, 1), (49, 1), (50, 1), (51, 2), (100, 2), (1001, 21)]
)
def test_sampling_is_frozen_and_reports_rounding(tmp_path, count, selected):
    pipeline.save(tmp_path / "survey.json", {"status": "COMPLETE", "cohort": "opened"})
    batches = [[{"source_id": str(number)}] for number in range(count)]
    plan = vetter.replay_plan(tmp_path, batches, "sample")
    assert vetter.replay_plan(tmp_path, batches, "sample") == plan
    assert len(plan["selected_batches"]) == selected
    assert plan["actual_fraction"] == selected / count
    assert (
        len(vetter.replay_plan(tmp_path, batches, "all")["selected_batches"]) == count
    )
    result = {
        "sources": 1,
        "execution_identity_sha256": pipeline.json_sha256({"policy": "frozen"}),
        "detector_replay": {
            "plan_sha256": pipeline.json_sha256(plan),
            "selected": True,
            "status": "RECONCILED",
            "verified_input_sources": 1,
        },
    }
    vetter.verify_replay_receipt(
        result, plan["selected_batches"][0], plan, {"policy": "frozen"}
    )
    with pytest.raises(ValueError, match="different frozen execution identity"):
        vetter.verify_replay_receipt(
            result, plan["selected_batches"][0], plan, {"policy": "other"}
        )
    result["detector_replay"]["verified_input_sources"] = 0
    with pytest.raises(ValueError, match="receipt differs from frozen plan"):
        vetter.verify_replay_receipt(
            result, plan["selected_batches"][0], plan, {"policy": "frozen"}
        )


def test_old_survey_cannot_silently_drop_binary_evidence(tmp_path):
    pipeline.save(
        tmp_path / "survey.json",
        {
            "schema": 7,
            "status": "COMPLETE",
            "artifacts": {},
            "detector_contract": pipeline.detector_contract(),
            "implementation": pipeline.implementation_hashes(),
        },
    )
    with pytest.raises(ValueError, match="lacks frozen follow-up diagnostics"):
        vetter.read_survey(tmp_path)


@pytest.mark.parametrize("replayed", [False, True])
def test_only_declared_batches_execute_detectors(prepared, monkeypatch, replayed):
    _, survey, preparation, frozen = prepared
    expected = expectations("periodic")
    rows = expected["123"]["diagnostics"]
    write_rows(survey / "followup-diagnostics.csv", list(rows[0]), rows)
    write_rows(survey / "detections.csv", list(rows[0]), rows)
    report = pipeline.read(survey / "survey.json")
    report["selection"] = {"input_manifest": preparation["input_manifest"]}
    pipeline.save(survey / "survey.json", report)
    out = survey.parent / "vetting"
    out.mkdir()
    index = vetter.shared_alert_index(out, survey, report, [frozen], expected)
    calls = []

    def scan(args):
        calls.append(args)
        vetter.restore_detection(
            args.run, survey, pipeline.validate_preparation(args.run), expected
        )

    monkeypatch.setattr(pipeline, "scan", scan)
    plan = {
        "contract": vetter.REPLAY_CONTRACT,
        "survey_sha256": pipeline.digest(survey / "survey.json"),
        "total_batches": 2,
        "selected_batches": [1] if replayed else [2],
    }
    args = SimpleNamespace(
        threads=1,
        disposition_policy=pipeline.DISPOSITION_POLICY,
        detector_replay_plan=plan,
        execution_identity=execution_identity(survey, index),
    )
    batch, ids, detection, _ = vetter.detect_batch(
        out, survey, 1, [frozen], expected, args, index
    )
    assert ids == ["123"] and len(calls) == int(replayed)
    assert pipeline.read(batch / "proposals/123.json")["rust_rows"] == rows
    if replayed:
        assert calls[0].disposition_policy == pipeline.DISPOSITION_POLICY
    assert detection["detector_replay"] == {
        "plan_sha256": pipeline.json_sha256(plan),
        "selected": replayed,
        "status": "RECONCILED" if replayed else "FROZEN_ROWS",
        "verified_input_sources": 1,
    }


@pytest.mark.parametrize("artifact", ["survey.json", "detector"])
def test_frozen_identity_mutation_stops_before_preparation(
    prepared, monkeypatch, artifact
):
    _, survey, _, frozen = prepared
    expected = expectations("periodic")
    plan = {
        "survey_sha256": pipeline.digest(survey / "survey.json"),
        "selected_batches": [],
    }
    index = survey / "index.sqlite"
    index.write_bytes(b"frozen index")
    frozen_identity = execution_identity(survey, index)
    path = survey / artifact
    if artifact.endswith("json"):
        report = pipeline.read(path)
        report["changed"] = True
        pipeline.save(path, report)
    else:
        path.write_bytes(b"changed frozen binary")

    def prepare(args):
        raise AssertionError("Changed frozen identities reached preparation")

    monkeypatch.setattr(pipeline, "prepare", prepare)
    args = SimpleNamespace(
        detector_replay_plan=plan, execution_identity=frozen_identity
    )
    with pytest.raises(ValueError, match="Frozen .*changed"):
        vetter.detect_batch(
            survey.parent / "vetting", survey, 1, [frozen], expected, args, None
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM windows",
        "DELETE FROM alerts",
        "UPDATE windows SET end_bkjd=99",
        "DELETE FROM sources",
    ],
)
def test_shared_index_rejects_lost_or_changed_interval_records(prepared, mutation):
    _, survey, preparation, frozen = prepared
    out = survey.parent / "shared-index"
    out.mkdir()
    report = {
        "selection": {"input_manifest": preparation["input_manifest"]},
        "detector_contract": pipeline.detector_contract(),
    }
    expected = expectations("periodic")
    index = vetter.shared_alert_index(out, survey, report, [frozen], expected)
    with sqlite3.connect(index) as connection:
        connection.execute(mutation)
    with pytest.raises(ValueError, match="Shared-alert index"):
        vetter.shared_alert_index(out, survey, report, [frozen], expected)


def index_fixture_sources(prepared, count, view):
    _, survey, preparation, frozen = prepared
    manifest = Path(preparation["input_manifest"]["path"])
    specification = pipeline.read(manifest)
    template = specification["sources"]["123"]
    sources, expected = [], {}
    for number in range(123, 123 + count):
        sid = str(number)
        item = copy.deepcopy(template)
        item["metadata"]["source_id"] = sid
        specification["sources"][sid] = item
        expected[sid] = expectations(("periodic", "sparse", "shallow")[number % 3])["123"]
        for rows in (expected[sid]["rust"], expected[sid]["shallow"]):
            for row in rows:
                row["source_id"] = sid
    pipeline.save(manifest, specification)
    for sid in expected:
        curve = pipeline.load_curve(pipeline.ROOT, sid, view, manifest)
        records = pipeline.prepare_curve_records(
            curve, mask_flatten=view == pipeline.DISCOVERY_VIEW
        )
        payload = struct.pack(
            "<8sQQQ", b"FSC_LC2\0", int(sid), pipeline.VIEW_CODES[view], len(records)
        ) + records.tobytes()
        sources.append(dict(
            frozen, source_id=sid, input_arrays_sha256=pipeline.curve_digest(curve),
            fsc_sha256=hashlib.sha256(payload).hexdigest(),
        ))
    report = {
        "selection": {"input_manifest": {"path": str(manifest)}},
        "detector_contract": pipeline.detector_contract(view),
    }
    return survey, report, sources, expected


@pytest.mark.parametrize("view", ["psf_flat", "psf_qdiag", "sap_qdiag"])
def test_parallel_shared_index_preserves_exact_rows_and_hash(prepared, view):
    survey, report, sources, expected = index_fixture_sources(prepared, 8, view)
    indexes = []
    for workers in (1, 4):
        out = survey.parent / f"index-{workers}"
        out.mkdir()
        indexes.append(vetter.shared_alert_index(
            out, survey, report, sources, expected, workers
        ))
    assert pipeline.digest(indexes[0]) == pipeline.digest(indexes[1])
    with sqlite3.connect(indexes[1]) as connection:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM alerts WHERE source_id=?", ("123",)
        ).fetchall()
        assert any("alerts_source" in row[3] for row in plan)


def test_parallel_shared_index_resumes_committed_prefix_after_worker_failure(prepared):
    survey, report, sources, expected = index_fixture_sources(prepared, 105, "psf_flat")
    out = survey.parent / "resume-index"
    out.mkdir()
    changed = copy.deepcopy(sources)
    changed[-1]["fsc_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prepared light curve changed"):
        vetter.shared_alert_index(out, survey, report, changed, expected, 4)
    with sqlite3.connect(out / "shared-alerts.sqlite.building") as connection:
        assert connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 100
        vetter.verify_index_body(connection)
    resumed = vetter.shared_alert_index(out, survey, report, sources, expected, 4)
    fresh = survey.parent / "fresh-index"
    fresh.mkdir()
    uninterrupted = vetter.shared_alert_index(fresh, survey, report, sources, expected, 1)
    with sqlite3.connect(resumed) as a, sqlite3.connect(uninterrupted) as b:
        assert vetter.verify_index_body(a).hexdigest() == vetter.verify_index_body(b).hexdigest()
        for table in ("sources", "alerts", "windows", "metadata"):
            assert a.execute(f"SELECT * FROM {table}").fetchall() == b.execute(
                f"SELECT * FROM {table}"
            ).fetchall()


def test_gated_rejections_preserve_ungated_survey_context(prepared):
    batch, survey, preparation, frozen = prepared
    expected = expectations("periodic")
    rows = expected["123"]["diagnostics"]
    write_rows(survey / "followup-diagnostics.csv", list(rows[0]), rows)
    write_rows(survey / "detections.csv", list(rows[0]), rows)
    report = {
        "selection": {"input_manifest": preparation["input_manifest"]},
        "detector_contract": pipeline.detector_contract(),
    }
    out = survey.parent / "shared-index"
    out.mkdir()
    index = vetter.shared_alert_index(out, survey, report, [frozen], expected)
    thresholds = {name: 10000 for name in ("periodic", "shallow", "sparse")}
    from deterministic.signal_contracts import LANE_CONTRACT

    calibration = survey.parent / "lane-calibration.json"
    pipeline.save(
        calibration,
        {
            "status": "CALIBRATED",
            "measurement_scope": "OPENED_DEVELOPMENT_NULLS",
            "lane_contract": LANE_CONTRACT,
            "thresholds": thresholds,
        },
    )
    policy = pipeline.read(pipeline.ROOT / pipeline.DISPOSITION_POLICY)
    policy["lane_gates"] = {
        "status": "CALIBRATED",
        "contract": LANE_CONTRACT,
        "thresholds": thresholds,
        "calibration": {
            "path": str(calibration),
            "sha256": pipeline.digest(calibration),
        },
    }
    policy_path = survey.parent / "gated-policy.json"
    pipeline.save(policy_path, policy)
    preparation["disposition_policy"] = str(policy_path)
    vetter.restore_detection(batch, survey, preparation, expected)
    vetter.reconcile_detector(batch, expected)
    vetter.reconcile_shared_alerts(batch, index, [frozen])
    proposal = pipeline.read(batch / "proposals/123.json")
    assert proposal["ungated_alerts"] and proposal["alerts"] == []
    assert (
        pipeline.read(batch / "source-ledger.json")[0]["status"] == "DETECTOR_REJECTION"
    )
    assert pipeline.read(batch / "queue.json")["rows"] == []


def test_empty_gated_queue_completes_actual_evidence_stages(prepared, monkeypatch):
    test_gated_rejections_preserve_ungated_survey_context(prepared)
    original, survey, preparation, _ = prepared
    batch = original.parent / "empty-gated-run"
    policy = original.parent / "gated-policy.json"
    references = cloud.catalogues()
    base_read = pipeline.read

    def read(path):
        value = base_read(path)
        if Path(path) == pipeline.ROOT / "pipeline/vetting/config.json":
            value.update(
                {name: references[name] for name in ("catalogues", "extra_catalogues")}
            )
        return value

    monkeypatch.setattr(pipeline, "read", read)
    pipeline.prepare(
        SimpleNamespace(
            shards=[0],
            limit=1,
            source_ids=original.parent / "source-ids.json",
            input_manifest=Path(preparation["input_manifest"]["path"]),
            disposition_policy=policy,
            out=batch,
        )
    )
    vetter.restore_detection(
        batch, survey, pipeline.validate_preparation(batch), expectations("periodic")
    )
    pipeline.vet(
        SimpleNamespace(
            run=batch,
            python=Path(sys.executable),
            workers=1,
            no_network=True,
            disposition_policy=policy,
            pixel_cache_directory=batch / "pixel-cache",
            pixel_minimum_free_bytes=vetter.MINIMUM_PIXEL_FREE_BYTES,
            shared_alert_index=original.parent / "shared-index/shared-alerts.sqlite",
        )
    )
    result = pipeline.read(batch / "result.json")
    assert result["status"] == "EVIDENCE_STAGES_COMPLETE"
    assert result["boi_signals"] == 0
    assert pipeline.read(batch / "source-ledger.json")[0]["source_id"] == "123"


@pytest.fixture
def completed(prepared):
    batch, survey, preparation, frozen = prepared
    expected = expectations("periodic")
    rows = expected["123"]["diagnostics"]
    write_rows(survey / "followup-diagnostics.csv", list(rows[0]), rows)
    write_rows(survey / "detections.csv", list(rows[0]), rows)
    vetter.restore_detection(batch, survey, preparation, expected)
    detection = pipeline.read(batch / "detection.json")
    index = survey / "index.sqlite"
    index.write_bytes(b"opened index fixture")
    frozen_identity = execution_identity(survey, index)
    proposal = pipeline.read(batch / "proposals/123.json")
    (batch / "pixel-cache").mkdir()
    pipeline.save(
        batch / "vetting.json",
        {
            "pixel_cache_directory": str(batch / "pixel-cache"),
            "disposition_policy": frozen_identity["disposition_policy"]["path"],
            "shared_alert_index": frozen_identity["shared_alert_index"],
            "quality_bitmask": -1,
        },
    )
    pipeline.save(
        batch / "vetting/manifest.json",
        {
            "source_ids": ["123"],
            "config_sha256": pipeline.digest(batch / "vetting.json"),
            "code_sha256": frozen_identity["implementation"],
            "catalogue_sha256": frozen_identity["catalogue_sha256"],
            "discovery": {
                "manifest_sha256": pipeline.digest(batch / "discovery-manifest.json")
            },
            "input_sha256": {
                frozen_identity["disposition_policy"]["path"]: frozen_identity[
                    "disposition_policy"
                ]["sha256"]
            },
        },
    )
    source = batch / "vetting/sources/123.json"
    pipeline.save(
        source,
        {
            "source_id": "123",
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "alerts": proposal["alerts"],
            "events": [{"alert_id": alert["event_id"]} for alert in proposal["alerts"]],
        },
    )
    pipeline.save(batch / "vetting/cohort.json", [proposal])
    pipeline.save(batch / "vetting/metadata.json", {"123": {}})
    pipeline.save(batch / "vetting/scene.json", [{"source_id": "123"}])
    pipeline.save(
        batch / "vetting/results.json",
        {
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "sources": 1,
            "original_alerts_preserved": detection["alerts"],
            "rows": [{"source_id": "123", "evidence": str(source)}],
        },
    )
    for name in ("bois.json", "surviving-signals.json", "signal-ledger.json"):
        pipeline.save(batch / "vetting" / name, [])
    pipeline.save(
        batch / "result.json",
        {
            "status": "EVIDENCE_STAGES_COMPLETE",
            "boi_signals": 0,
            "unresolved_signals": 1,
        },
    )
    archive = vetter.compact(batch, detection)
    result = {
        "status": "COMPLETE",
        "batch": 1,
        "sources": 1,
        "source_ids_sha256": pipeline.json_sha256([frozen["source_id"]]),
        "alerts": detection["alerts"],
        "boi_signals": 0,
        "unresolved_signals": 1,
        "archive_sha256": pipeline.json_sha256(archive),
        "execution_identity_sha256": pipeline.json_sha256(frozen_identity),
    }
    pipeline.save(batch / "batch-complete.json", result)
    return (
        batch,
        {"batch": 1, "sha256": pipeline.digest(batch / "batch-complete.json")},
        frozen_identity,
    )


def test_complete_batch_keeps_required_evidence_after_compaction(completed):
    batch, record, frozen = completed
    assert vetter.verify_batch(batch, record, frozen)["sources"] == 1


def test_empty_completed_archive_cannot_be_adopted(tmp_path):
    archive = {
        "files": {},
        "prepared_inputs": {},
        "relinked_summaries": {},
        "retained_files": {},
    }
    pipeline.save(tmp_path / "compact.json", archive)
    pipeline.save(
        tmp_path / "batch-complete.json",
        {
            "status": "COMPLETE",
            "batch": 1,
            "sources": 1,
            "alerts": 1,
            "archive_sha256": pipeline.json_sha256(archive),
        },
    )
    record = {"batch": 1, "sha256": pipeline.digest(tmp_path / "batch-complete.json")}
    with pytest.raises(ValueError, match="required evidence inventory"):
        vetter.verify_batch(tmp_path, record, {})


@pytest.mark.parametrize(
    "field", ["sources", "alerts", "boi_signals", "unresolved_signals"]
)
def test_complete_batch_reconciles_declared_counts_with_evidence(completed, field):
    batch, record, frozen = completed
    receipt = pipeline.read(batch / "batch-complete.json")
    receipt[field] += 1
    pipeline.save(batch / "batch-complete.json", receipt)
    record["sha256"] = pipeline.digest(batch / "batch-complete.json")
    with pytest.raises(ValueError, match="counts do not reconcile"):
        vetter.verify_batch(batch, record, frozen)


@pytest.mark.parametrize(
    "name", ["disposition_policy", "vetting_configuration", "shared_alert_index"]
)
def test_configuration_drift_between_batches_is_rejected(prepared, name):
    _, survey, _, _ = prepared
    index = survey / "index.sqlite"
    index.write_bytes(b"opened index fixture")
    frozen = execution_identity(survey, index)
    path = survey / f"{name}.fixture"
    path.write_bytes(Path(frozen[name]["path"]).read_bytes())
    frozen[name]["path"] = str(path)
    vetter.verify_execution_identity(SimpleNamespace(execution_identity=frozen))
    path.write_bytes(b"changed between batches")
    with pytest.raises(ValueError, match="Frozen vetting policy"):
        vetter.verify_execution_identity(SimpleNamespace(execution_identity=frozen))


def reseal_archive(batch, record):
    archive = pipeline.read(batch / "compact.json")
    archive["retained_files"] = {
        name: pipeline.digest(batch / name) for name in archive["retained_files"]
    }
    for name, checks in archive["files"].items():
        with gzip.open(batch / (name + ".gz"), "rb") as stream:
            checks["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        checks["archive_sha256"] = pipeline.digest(batch / (name + ".gz"))
    pipeline.save(batch / "compact.json", archive)
    receipt = pipeline.read(batch / "batch-complete.json")
    receipt["archive_sha256"] = pipeline.json_sha256(archive)
    pipeline.save(batch / "batch-complete.json", receipt)
    record["sha256"] = pipeline.digest(batch / "batch-complete.json")


def test_unrecorded_batch_with_lost_alert_events_cannot_be_adopted(completed):
    batch, record, frozen = completed
    path = batch / "vetting/sources/123.json.gz"
    with gzip.open(path, "rt") as stream:
        evidence = json.load(stream)
    evidence["events"] = []
    with gzip.open(path, "wt") as stream:
        json.dump(evidence, stream)
    reseal_archive(batch, record)
    with pytest.raises(ValueError, match="source is missing or incomplete"):
        vetter.verify_batch(batch, record, frozen)


def test_unrecorded_batch_with_different_catalogues_cannot_be_adopted(completed):
    batch, record, frozen = completed
    manifest = pipeline.read(batch / "vetting/manifest.json")
    manifest["catalogue_sha256"] = {}
    pipeline.save(batch / "vetting/manifest.json", manifest)
    reseal_archive(batch, record)
    with pytest.raises(ValueError, match="frozen execution bindings"):
        vetter.verify_batch(batch, record, frozen)


def test_catalogue_drift_after_last_batch_blocks_completion(prepared):
    _, survey, _, _ = prepared
    index = survey / "index.sqlite"
    index.write_bytes(b"opened index fixture")
    frozen = execution_identity(survey, index)
    Path(next(iter(frozen["catalogue_sha256"]))).write_bytes(b"changed catalogue")
    with pytest.raises(ValueError, match="Frozen vetting policy"):
        vetter.verify_execution_identity(SimpleNamespace(execution_identity=frozen))


def test_missing_global_report_cannot_adopt_old_batches(tmp_path):
    (tmp_path / "batches/000001").mkdir(parents=True)
    with pytest.raises(ValueError, match="lacks its frozen report"):
        vetter.resume_vetting_report(tmp_path, {}, [], SimpleNamespace(resume=True))


def test_archive_rechecks_original_uncompressed_identity(completed):
    batch, record, frozen = completed
    archive = pipeline.read(batch / "compact.json")
    archive["files"]["vetting/sources/123.json"]["source_sha256"] = "0" * 64
    pipeline.save(batch / "compact.json", archive)
    receipt = pipeline.read(batch / "batch-complete.json")
    receipt["archive_sha256"] = pipeline.json_sha256(archive)
    pipeline.save(batch / "batch-complete.json", receipt)
    record["sha256"] = pipeline.digest(batch / "batch-complete.json")
    with pytest.raises(ValueError, match="Archived original vetting evidence changed"):
        vetter.verify_batch(batch, record, frozen)


@pytest.mark.parametrize("status", ["FROZEN_EVALUATION", "RELEASED", "UNCONFIGURED"])
def test_development_configuration_override_rejects_other_policy_states(tmp_path, status):
    policy = tmp_path / "policy.json"
    pipeline.save(policy, {"status": status})
    with pytest.raises(ValueError, match="override requires a development policy"):
        pipeline.vetting_config_path(SimpleNamespace(
            disposition_policy=policy, vetting_config=tmp_path / "config.json"
        ))


def test_development_configuration_override_is_explicit(tmp_path):
    policy = tmp_path / "policy.json"
    pipeline.save(policy, {"status": "DEVELOPMENT"})
    path = tmp_path / "cloud-config.json"
    assert pipeline.vetting_config_path(SimpleNamespace(
        disposition_policy=policy, vetting_config=path
    )) == path
    assert pipeline.vetting_config_path(SimpleNamespace()) == (
        pipeline.ROOT / "pipeline/vetting/config.json"
    ).resolve()


@pytest.mark.parametrize("defect", ["local_profile", "external", "sealed_policy", "sealed_role"])
def test_reduced_pixel_reserve_requires_opened_cloud_storage(tmp_path, monkeypatch, defect):
    policy = tmp_path / "policy.json"
    pipeline.save(policy, {"status": "FROZEN_EVALUATION" if defect == "sealed_policy" else "DEVELOPMENT"})
    cache = tmp_path / "cloud-cache"
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "local" if defect == "local_profile" else "cloud-development")
    monkeypatch.setattr(pipeline, "EXTERNAL_DUMPS", (tmp_path,) if defect == "external" else ())
    args = SimpleNamespace(disposition_policy=policy,
                           pixel_input_role="evaluation" if defect == "sealed_role" else "development")
    with pytest.raises(ValueError, match="opened cloud development storage"):
        pipeline.verify_pixel_reserve(args, cache, 2 * 1024**3)


def test_reduced_cloud_reserve_preserves_local_default(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    pipeline.save(policy, {"status": "DEVELOPMENT"})
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(pipeline, "EXTERNAL_DUMPS", ())
    args = SimpleNamespace(disposition_policy=policy, pixel_input_role="development")
    pipeline.verify_pixel_reserve(args, tmp_path / "cloud-cache", 2 * 1024**3)
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "local")
    pipeline.verify_pixel_reserve(args, tmp_path / "external-cache", 20 * 1024**3)


def test_two_gib_reserve_on_mounted_control_storage(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    pipeline.save(policy, {"status": "FROZEN_EVALUATION"})
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "local")
    monkeypatch.setattr(pipeline, "EXTERNAL_DUMPS", (tmp_path,))
    args = SimpleNamespace(disposition_policy=policy, pixel_input_role="evaluation")
    cache = tmp_path / "controls-20260926/pixel-cache"
    pipeline.verify_pixel_reserve(args, cache, 2 * 1024**3)
    with pytest.raises(ValueError, match="opened cloud development storage"):
        pipeline.verify_pixel_reserve(args, cache, 1024**3)


def test_inherited_reduced_pixel_reserve_is_checked(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    pipeline.save(policy, {"status": "DEVELOPMENT"})
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(pipeline, "EXTERNAL_DUMPS", ())
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "local")
    args = SimpleNamespace(disposition_policy=policy)
    config = {"pixel_cache_directory": "dump/pixels", "pixel_minimum_free_bytes": 2 * 1024**3}
    with pytest.raises(ValueError, match="opened cloud development storage"):
        pipeline.pixel_cache_configuration(args, config)
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    assert pipeline.pixel_cache_configuration(args, config)["pixel_minimum_free_bytes"] == 2 * 1024**3


@pytest.fixture
def cloud_identity(prepared, monkeypatch):
    _, survey, _, source = prepared
    index = survey / "index.sqlite"
    index.write_bytes(b"opened index fixture")
    catalogue = survey / "catalogue.json"
    pipeline.save(catalogue, {"scope": "opened cloud fixture"})
    monkeypatch.setattr(vetter, "reference_catalogue_paths", lambda config: ([catalogue], []))
    config_path = survey / "cloud-config.json"
    pipeline.save(config_path, {"parent_index": str(index),
                                "parent_index_sha256": pipeline.digest(index)})
    receipt = survey / "pixel-receipt.json"
    pipeline.save(receipt, {"scientific_outcomes_read": False})
    ledger = survey / "pixel-ledger.json"
    pipeline.save(ledger, {"schema": 1, "status": "FROZEN_CONTROL_PIXEL_CONTENT_LEDGER",
                          "inputs": {str(receipt): pipeline.digest(receipt)}, "sources": {}})
    args = SimpleNamespace(
        vetting_config=config_path, disposition_policy=pipeline.DISPOSITION_POLICY,
        frozen_pixel_inputs=ledger, pixel_input_role="development",
        pixel_cache_maximum_bytes=4 * 1024**3, pixel_minimum_free_bytes=2 * 1024**3,
        record_every_gate=True, batch_size=1, threads=2, workers=1,
        concurrent_batches=1, no_network=True, detector_replay_plan={"mode": "all"},
    )
    report = {"artifacts": {"detector": pipeline.digest(survey / "detector")},
              "detector_contract": pipeline.detector_contract(), "selection": {}}
    frozen = vetter.identity(survey, report, pipeline.ROOT / pipeline.DISPOSITION_POLICY,
                             args, [source], index)
    args.execution_identity = frozen
    return args, config_path, ledger, receipt


def test_coordinator_binds_development_configuration_and_frozen_pixel_options(cloud_identity):
    args, config, ledger, _ = cloud_identity
    frozen = args.execution_identity
    assert frozen["vetting_configuration"] == {"path": str(config), "sha256": pipeline.digest(config)}
    assert frozen["frozen_pixel_inputs"] == {
        "path": str(ledger), "sha256": pipeline.digest(ledger),
        "role": "development", "maximum_cache_bytes": 4 * 1024**3,
    }
    assert frozen["pixel_minimum_free_bytes"] == 2 * 1024**3
    assert frozen["record_every_gate"] is True
    vetter.verify_execution_identity(args)


@pytest.mark.parametrize("position", [1, 2, 3], ids=["config", "ledger", "receipt"])
def test_coordinator_rejects_cloud_configuration_and_ledger_drift(cloud_identity, position):
    args = cloud_identity[0]
    path = cloud_identity[position]
    value = pipeline.read(path)
    value["changed_after_freeze"] = True
    pipeline.save(path, value)
    with pytest.raises(ValueError, match="Frozen vetting policy"):
        vetter.verify_execution_identity(args)


@pytest.fixture
def frozen_cache(tmp_path):
    cache = tmp_path / "pixel-cache"
    binding = {"sha256": "a" * 64}
    namespace = cache / f"frozen-{binding['sha256']}"
    namespace.mkdir(parents=True)
    payload = b"opened frozen pixel fixture"
    checksum = hashlib.sha256(payload).hexdigest()
    product = namespace / f"{checksum}.fits.gz"
    product.write_bytes(payload)
    (namespace / "cache.lock").touch()
    pipeline.save(tmp_path / "vetting.json", {
        "pixel_cache_directory": str(cache), "frozen_pixel_inputs": binding
    })
    manifests = tmp_path / "vetting/pixel-manifests"
    manifests.mkdir(parents=True)
    pipeline.save(manifests / "10.json", {"quarters": {"1": {
        "status": "AVAILABLE", "path": str(product), "sha256": checksum,
        "transport_uri": "mast:KEPLER/target_pixel_files/fixture.fits.gz",
    }}})
    return tmp_path, cache, namespace, product


def test_compaction_records_and_evicts_verified_frozen_pixel_namespace(frozen_cache):
    batch, cache, namespace, product = frozen_cache
    size = product.stat().st_size
    report = vetter.evict_batch_pixels(batch)
    assert report["bytes_evicted"] == size
    assert report["products"][0]["filename"] == f"{namespace.name}/{product.name}"
    assert report["products"][0]["transport_uri"].startswith("mast:KEPLER/")
    assert not cache.exists()
    assert pipeline.read(batch / "cache-eviction.json") == report


@pytest.mark.parametrize("defect", ["missing", "changed_hash", "other_namespace", "temporary", "symlink"])
def test_compaction_rejects_incomplete_or_redirected_frozen_cache(frozen_cache, defect):
    batch, cache, namespace, product = frozen_cache
    if defect == "missing":
        product.unlink()
    elif defect == "changed_hash":
        product.write_bytes(b"changed pixel")
    elif defect == "other_namespace":
        (cache / "frozen-unrecorded").mkdir()
    elif defect == "temporary":
        (namespace / "unfinished.tmp").write_bytes(b"unfinished transfer")
    else:
        external = batch / "retained-original-pixel.fits.gz"
        external.write_bytes(product.read_bytes())
        product.unlink()
        product.symlink_to(external)
    with pytest.raises(ValueError):
        vetter.evict_batch_pixels(batch)
    assert cache.exists()
    assert not (batch / "cache-eviction.json").exists()
    if defect == "symlink":
        assert external.read_bytes() == b"opened frozen pixel fixture"


def test_frozen_cache_rejects_available_product_outside_declared_namespace(frozen_cache):
    batch, cache, _, product = frozen_cache
    manifest = batch / "vetting/pixel-manifests/10.json"
    value = pipeline.read(manifest)
    value["quarters"]["1"]["path"] = str(batch / product.name)
    pipeline.save(manifest, value)
    with pytest.raises(ValueError, match="outside its cache namespace"):
        vetter.evict_batch_pixels(batch)
    assert cache.exists()
    assert product.exists()


def test_completed_batch_rejects_changed_frozen_pixel_binding(completed):
    batch, record, frozen = completed
    config = pipeline.read(batch / "vetting.json")
    config["frozen_pixel_inputs"] = {"sha256": "a" * 64}
    pipeline.save(batch / "vetting.json", config)
    manifest = pipeline.read(batch / "vetting/manifest.json")
    manifest["config_sha256"] = pipeline.digest(batch / "vetting.json")
    pipeline.save(batch / "vetting/manifest.json", manifest)
    reseal_archive(batch, record)
    with pytest.raises(ValueError, match="frozen execution bindings"):
        vetter.verify_batch(batch, record, frozen)
