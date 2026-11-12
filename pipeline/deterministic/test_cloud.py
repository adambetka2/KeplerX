"""Cloud comparisons and transfers must preserve complete, frozen evidence."""

import gzip
import hashlib
import io
import json
import sqlite3
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deterministic import cloud


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bind_discovery(run):
    path = run / "discovery-manifest.json"
    manifest = json.loads(path.read_text())
    required = [
        run / name for name in (
            "raw.csv", "shallow-raw.csv", "shallow-eligibility.csv",
            "preparation.json", "queue.json", "source-ledger.json",
        )
    ]
    required += sorted((run / "proposals").glob("*.json"))
    manifest["artifacts"] = {str(item): cloud.digest(item) for item in required}
    write(path, manifest)


def make_run(run):
    run.mkdir(parents=True)
    fsc = run / "inputs/123.psf_flat.fsc"
    fsc.parent.mkdir()
    fsc.write_bytes(b"complete frozen prepared records")
    binary = run / "detector"
    binary.write_bytes(b"fixture detector binary")
    contract = {
        "origin": "rust_v6_shallow_v1", "contract_version": 5,
        "files": {"config.json": "a" * 64, "src/main.rs": "b" * 64},
        "shallow_files": {"config.json": "c" * 64, "run.py": "d" * 64},
        "views": ["psf_flat"], "discovery_view": "psf_flat",
        "preparation": "frozen photometry preparation",
    }
    preparation = {
        "status": "PREPARED", "cohort_sha256": "e" * 64,
        "selection": {"source_ids": ["123"], "shards": [0]},
        "detector_contract": contract,
        "sources": [{
            "source_id": "123", "raw_fits_sha256": "f" * 64,
            "flatten_mask_sha256": "1" * 64, "views": [{"view": "psf_flat"}],
            "unavailable_views": {},
            "inputs": {"psf_flat": {
                "fsc_path": str(fsc), "fsc_sha256": cloud.digest(fsc),
                "retained_cadences": 100,
            }},
        }],
    }
    write(run / "preparation.json", preparation)
    binary_identity = {"path": str(binary), "sha256": cloud.digest(binary)}
    write(run / "detection.json", {
        "status": "DETECTION_COMPLETE", "sources": 1, "alerted_sources": 1,
        "alerts": 1, "raw_rows": 1, "shallow_rows": 1,
        "shallow_ineligible_sources": 0, "binary": binary_identity,
    })
    write(run / "source-ledger.json", [{
        "source_id": "123", "status": "QUEUED", "alerts": 1,
        "raw_rows": 1, "shallow_rows": 1,
        "evidence": str(run / "proposals/123.json"),
    }])
    write(run / "queue.json", {"discovery_manifest": "discovery-manifest.json", "rows": [{
        "source_id": "123", "evidence": "proposals/123.json",
    }]})
    write(run / "proposals/123.json", {
        "source_id": "123", "views": [{"view": "psf_flat"}],
        "alerts": [{"event_id": "event-1", "kind": "SPARSE_EVENT_WINDOW"}],
        "supplied_hypotheses": [],
    })
    for name in ("raw.csv", "shallow-raw.csv"):
        (run / name).write_text("source_id,score\n123,8.0\n")
    (run / "shallow-eligibility.csv").write_text(
        "source_id,status,reason,elapsed_seconds\n123,ELIGIBLE,,1.0\n"
    )
    write(run / "discovery-manifest.json", {
        "schema": 2, "origin": contract["origin"], "detector_contract": contract,
        "binary": binary_identity, "artifacts": {},
    })
    bind_discovery(run)
    config = json.loads((cloud.ROOT / "pipeline/vetting/config.json").read_text())
    config.update(
        run_directory=str(run / "vetting"), queue=str(run / "queue.json"),
        workers=1, network=False, disposition_policy="pipeline/vetting/disposition-policy.json",
        discovery_contract=contract, triage_execution="RECORD_EVERY_GATE",
    )
    write(run / "vetting.json", config)
    write(run / "result.json", {"status": "EVIDENCE_STAGES_COMPLETE"})
    write(run / "vetting/results.json", {
        "processing_status": "EVIDENCE_STAGES_COMPLETE", "sources": 1,
        "original_alerts_preserved": 1, "event_hypotheses": 1,
    })
    write(run / "vetting/sources/123.json", {
        "source_id": "123", "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "events": [{"event_id": "event-1:psf_flat", "alert_id": "event-1"}],
        "signals": [{"signal_id": "signal-1", "depth": 0.001}],
    })
    write(run / "vetting/manifest.json", {
        "config_sha256": cloud.digest(run / "vetting.json"), "code_sha256": {},
        "input_sha256": {str(cloud.ROOT / config["disposition_policy"]): "2" * 64},
        "catalogue_sha256": {"reference-catalogue.json": "4" * 64},
    })
    return run


@pytest.fixture
def pair(tmp_path):
    return make_run(tmp_path / "baseline"), make_run(tmp_path / "candidate")


def assert_rejected(program):
    try:
        report = program()
    except (ValueError, FileNotFoundError, KeyError):
        return
    assert report["status"] == "DIFFERENT"


def test_complete_identical_runs_compare(pair):
    assert cloud.compare(*pair)["status"] == "IDENTICAL"


def test_missing_and_empty_runs_cannot_compare(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        cloud.compare(tmp_path / "missing-a", tmp_path / "missing-b")


@pytest.mark.parametrize("lost", ["result.json", "vetting/results.json", "vetting/sources/123.json"])
def test_missing_vetting_evidence_cannot_compare(pair, lost):
    (pair[1] / lost).unlink()
    assert_rejected(lambda: cloud.compare(*pair))


@pytest.mark.parametrize("lost", ["proposals/123.json", "raw.csv", "shallow-eligibility.csv"])
def test_missing_from_both_runs_still_fails(pair, lost):
    for run in pair:
        (run / lost).unlink()
        bind_discovery(run) if (run / lost).suffix == ".json" else None
    assert_rejected(lambda: cloud.compare(*pair))


def test_empty_discovery_inventory_cannot_pass(pair):
    for run in pair:
        value = cloud.read(run / "discovery-manifest.json")
        value["artifacts"] = {}
        write(run / "discovery-manifest.json", value)
    assert_rejected(lambda: cloud.compare(*pair))


@pytest.mark.parametrize("failed", ["result.json", "vetting/results.json", "vetting/sources/123.json"])
def test_failed_vetting_cannot_pass_when_records_match(pair, failed):
    for run in pair:
        value = cloud.read(run / failed)
        key = "status" if failed == "result.json" else "processing_status"
        value[key] = "PARTIAL"
        write(run / failed, value)
    assert_rejected(lambda: cloud.compare(*pair))


def test_record_identity_must_agree_with_filename(pair):
    for run in pair:
        value = cloud.read(run / "vetting/sources/123.json")
        value["source_id"] = "456"
        write(run / "vetting/sources/123.json", value)
    assert_rejected(lambda: cloud.compare(*pair))


def test_lost_alerts_cannot_be_an_identical_success(pair):
    for run in pair:
        value = cloud.read(run / "vetting/sources/123.json")
        value["events"] = []
        write(run / "vetting/sources/123.json", value)
        value = cloud.read(run / "vetting/results.json")
        value["original_alerts_preserved"] = 0
        write(run / "vetting/results.json", value)
    assert_rejected(lambda: cloud.compare(*pair))


def test_actual_prepared_bytes_are_verified(pair):
    (pair[1] / "inputs/123.psf_flat.fsc").write_bytes(b"changed arrays")
    assert_rejected(lambda: cloud.compare(*pair))


def test_detector_csvs_are_byte_exact(pair):
    (pair[1] / "raw.csv").write_text("score,source_id\n8.0,123\n")
    bind_discovery(pair[1])
    assert cloud.compare(*pair)["status"] == "DIFFERENT"


def test_elapsed_detector_time_is_ignored(pair):
    (pair[1] / "shallow-eligibility.csv").write_text(
        "source_id,status,reason,elapsed_seconds\n123,ELIGIBLE,,2.0\n"
    )
    bind_discovery(pair[1])
    assert cloud.compare(*pair)["status"] == "IDENTICAL"


def test_recorded_policy_hash_must_match_even_when_current_file_is_shared(pair):
    value = cloud.read(pair[1] / "vetting/manifest.json")
    key = next(iter(value["input_sha256"]))
    value["input_sha256"][key] = "3" * 64
    write(pair[1] / "vetting/manifest.json", value)
    assert_rejected(lambda: cloud.compare(*pair))


def test_recorded_catalogue_snapshot_must_match(pair):
    value = cloud.read(pair[1] / "vetting/manifest.json")
    value["catalogue_sha256"]["reference-catalogue.json"] = "5" * 64
    write(pair[1] / "vetting/manifest.json", value)
    assert_rejected(lambda: cloud.compare(*pair))


def test_execution_workers_may_change(pair):
    value = cloud.read(pair[1] / "vetting.json")
    value["workers"] = 2
    value["alias"]["workers"] = 1
    write(pair[1] / "vetting.json", value)
    manifest = cloud.read(pair[1] / "vetting/manifest.json")
    manifest["config_sha256"] = cloud.digest(pair[1] / "vetting.json")
    write(pair[1] / "vetting/manifest.json", manifest)
    assert cloud.compare(*pair)["status"] == "IDENTICAL"


def test_zero_record_comparator_cannot_report_success(pair, monkeypatch):
    monkeypatch.setattr(cloud, "compare_runs", lambda *args: {
        "records": {"differing": 0, "missing_in_candidate": [],
                    "missing_in_baseline": [], "compared": 0},
    })
    assert cloud.compare(*pair)["status"] == "DIFFERENT"


def test_scientific_vetting_config_cannot_change(pair):
    candidate = pair[1]
    value = cloud.read(candidate / "vetting.json")
    value["quality_bitmask"] = 0
    write(candidate / "vetting.json", value)
    write(candidate / "vetting/manifest.json", {
        "config_sha256": cloud.digest(candidate / "vetting.json"), "code_sha256": {},
        "input_sha256": {str(cloud.ROOT / value["disposition_policy"]): "2" * 64},
        "catalogue_sha256": {"reference-catalogue.json": "4" * 64},
    })
    assert_rejected(lambda: cloud.compare(*pair))


def test_detector_rows_must_reconcile_declared_counts(pair):
    for run in pair:
        (run / "raw.csv").write_text("source_id,score\n")
        bind_discovery(run)
    assert_rejected(lambda: cloud.compare(*pair))


def archive(path, members):
    with tarfile.open(path, "w:gz") as bundle:
        for name, data, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            member.linkname = "../outside"
            bundle.addfile(member, io.BytesIO(data) if member.isfile() else None)


@pytest.mark.parametrize("name", ["../outside", "/outside", "."])
def test_unpack_rejects_path_escape(tmp_path, name):
    path = tmp_path / "bundle.tar.gz"
    archive(path, [(name, b"unsafe", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="escapes"):
        cloud.unpack(path, tmp_path / "dump")


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_unpack_rejects_nonfiles(tmp_path, kind):
    path = tmp_path / "bundle.tar.gz"
    archive(path, [("inputs/link", b"", kind)])
    with pytest.raises(ValueError, match="non-file"):
        cloud.unpack(path, tmp_path / "dump")


def test_unpack_rejects_duplicate_members(tmp_path):
    path = tmp_path / "bundle.tar.gz"
    archive(path, [("inputs/data", b"a", tarfile.REGTYPE), ("inputs/data", b"b", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="repeats"):
        cloud.unpack(path, tmp_path / "dump")


def test_unpack_cannot_follow_existing_escape_symlink(tmp_path):
    destination = tmp_path / "dump"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "inputs").symlink_to(outside, target_is_directory=True)
    path = tmp_path / "bundle.tar.gz"
    archive(path, [("inputs/data.json", b"unsafe", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="escapes"):
        cloud.unpack(path, destination)
    assert list(outside.iterdir()) == []


def test_oracle_rejects_changed_production_fixture(tmp_path, monkeypatch):
    path = tmp_path / "native-bls-oracle.npz"
    monkeypatch.setattr(cloud, "BLS_FIXTURE", path)
    path.write_bytes(b"changed expected results")
    with pytest.raises(ValueError, match="wrong-sized input"):
        cloud.oracle()


def test_public_preparation_rejects_changed_manifest(tmp_path, monkeypatch):
    import cloud_public

    monkeypatch.setattr(cloud, "DUMP", tmp_path)
    specification = tmp_path / "specification.json"
    monkeypatch.setattr(cloud_public, "SPECIFICATION", specification)
    write(specification, {"schema": 1})
    folder = tmp_path / "cloud-public"
    write(folder / "input-manifest.json", {"schema": 1, "sources": {}})
    write(folder / "source-ids.json", [])
    write(folder / "receipt.json", {
        "status": "PUBLIC_INPUTS_VERIFIED",
        "specification_sha256": cloud.digest(specification),
        "manifest_sha256": "0" * 64,
        "source_ids_sha256": cloud.digest(folder / "source-ids.json"),
    })
    with pytest.raises(ValueError, match="manifest changed"):
        cloud.prepare_public(tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_export_retains_every_evidence_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "DUMP", tmp_path)
    run = tmp_path / "run"
    write(run / "vetting/sources/123.json", {"source_id": "123"})
    write(run / "vetting/queries/gaia.json", {"status": "COMPLETE"})
    write(run / "measurements.jsonl", {"rejected_attempt": True})
    output = tmp_path / "exports/results.tar.gz"
    result = cloud.export_results([run], output)
    assert result["status"] == "EXPORTED"
    with tarfile.open(output) as bundle:
        names = set(bundle.getnames())
        assert {"run/vetting/sources/123.json", "run/vetting/queries/gaia.json", "run/measurements.jsonl"} <= names
        assert {"cloud-export/code.json", "cloud-export/code.patch", "cloud-export/code.diffstat"} <= names
        assert "cloud-export/source/pipeline/deterministic/cloud.py" in names
        for name in names:
            assert bundle.getmember(name).isfile()
    receipt = cloud.read(output.with_suffix(output.suffix + ".json"))
    assert receipt["archive"]["sha256"] == cloud.digest(output)
    assert receipt["files"]["run/measurements.jsonl"]["sha256"] == cloud.digest(run / "measurements.jsonl")


def test_export_cannot_follow_file_symlink_outside_dump(tmp_path, monkeypatch):
    dump = tmp_path / "dump"
    dump.mkdir()
    monkeypatch.setattr(cloud, "DUMP", dump)
    run = dump / "run"
    run.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("external content")
    (run / "escaped.txt").symlink_to(outside)
    with pytest.raises(ValueError):
        cloud.export_results([run], dump / "exports/results.tar.gz")


def test_export_rejects_empty_evidence_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "DUMP", tmp_path)
    run = tmp_path / "empty-run"
    run.mkdir()
    with pytest.raises(ValueError, match="evidence file"):
        cloud.export_results([run], tmp_path / "exports/results.tar.gz")


def test_export_materializes_linked_evidence_inside_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "DUMP", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    payload = tmp_path / "retained.json"
    payload.write_text("complete evidence")
    (run / "source.json").symlink_to(payload)
    output = tmp_path / "exports/results.tar.gz"
    cloud.export_results([run], output)
    with tarfile.open(output) as bundle:
        member = bundle.getmember("run/source.json")
        assert member.isfile()
        assert hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == cloud.digest(payload)


@pytest.fixture
def catalogue_bundle(tmp_path, monkeypatch):
    scratch = tmp_path / 'dump'
    scratch.mkdir()
    archive = tmp_path / 'catalogues.tar.gz'
    files = {}
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in cloud.PRIMARY_CATALOGUES + cloud.EXTRA_CATALOGUES:
            payload = b'[{"source":123}]' if name.endswith('.json') else b'public reference row\n'
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
            files[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            if name.endswith('.json'):
                files[name]['rows'] = 1
    specification = tmp_path / 'catalogues.json'
    write(specification, {
        'schema': 1, 'scope': 'complete existing public catalogue snapshots',
        'archive': {'bytes': archive.stat().st_size, 'sha256': cloud.digest(archive)},
        'files': files,
    })
    monkeypatch.setattr(cloud, 'DUMP', scratch)
    monkeypatch.setattr(cloud, 'CATALOGUE_BUNDLE', archive)
    monkeypatch.setattr(cloud, 'CATALOGUE_SPECIFICATION', specification)
    return scratch, archive, specification


def test_catalogue_bundle_preserves_every_original_snapshot(catalogue_bundle):
    scratch, _, _ = catalogue_bundle
    result = cloud.catalogues()
    assert result['status'] == 'SNAPSHOTS_VERIFIED'
    assert len(result['catalogues']) == 4 and len(result['extra_catalogues']) == 2
    assert cloud.catalogues() == result
    assert cloud.read(scratch / 'cloud-catalogues-receipt.json') == result
    path = Path(result['catalogues'][0])
    path.chmod(0o644)
    path.write_text('[]')
    with pytest.raises(ValueError, match='wrong-sized input|Changed input'):
        cloud.catalogues()


def test_catalogue_bundle_rejects_archive_mutation(catalogue_bundle):
    _, archive, _ = catalogue_bundle
    archive.write_bytes(b'changed snapshot bundle')
    with pytest.raises(ValueError, match='wrong-sized input|Changed input'):
        cloud.catalogues()


def test_catalogue_bundle_rejects_incomplete_specification(catalogue_bundle):
    _, _, specification = catalogue_bundle
    value = cloud.read(specification)
    value['files'].pop(cloud.PRIMARY_CATALOGUES[0])
    write(specification, value)
    with pytest.raises(ValueError, match='differs from configured snapshots'):
        cloud.catalogues()


def test_catalogue_bundle_rejects_inventory_addition(catalogue_bundle):
    scratch, _, _ = catalogue_bundle
    cloud.catalogues()
    (scratch / 'cloud-reference-catalogues/unregistered.json').write_text('[]')
    with pytest.raises(ValueError, match='inventory changed'):
        cloud.catalogues()


@pytest.fixture
def parent_bundle(tmp_path, monkeypatch):
    scratch = tmp_path / "dump"
    scratch.mkdir()
    database = tmp_path / "parents.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sources(source_id TEXT, parent_tpf INTEGER, shard INTEGER)")
        connection.executemany("INSERT INTO sources VALUES (?, ?, ?)", [("123", 10, 0), ("456", 10, 0)])
    archive = tmp_path / "parents.sqlite.gz"
    archive.write_bytes(gzip.compress(database.read_bytes(), mtime=0))
    specification = tmp_path / "parents.json"
    write(specification, {
        "schema": 1, "scope": "identity metadata only", "sources": 2, "parents": 1, "shards": 1,
        "archive": {"bytes": archive.stat().st_size, "sha256": cloud.digest(archive)},
        "database": {"bytes": database.stat().st_size, "sha256": cloud.digest(database)},
    })
    config = tmp_path / "config.json"
    write(config, {
        "parent_index": "dump/parent-index-20260925/parents.sqlite",
        "parent_index_sha256": cloud.digest(database),
    })
    monkeypatch.setattr(cloud, "DUMP", scratch)
    monkeypatch.setattr(cloud, "PARENT_BUNDLE", archive)
    monkeypatch.setattr(cloud, "PARENT_SPECIFICATION", specification)
    monkeypatch.setattr(cloud, "VETTING_CONFIGURATION", config)
    return scratch, archive, specification, config


def test_parent_index_preserves_exact_bytes_and_verifies_reuse(parent_bundle):
    scratch, _, specification, _ = parent_bundle
    result = cloud.parents()
    path = Path(result["parent_index"])
    assert cloud.digest(path) == cloud.read(specification)["database"]["sha256"]
    assert result["sources"] == 2 and result["parents"] == 1 and result["shards"] == 1
    assert cloud.parents() == result
    assert cloud.read(scratch / "cloud-parents-receipt.json") == result
    assert path.stat().st_mode & 0o222 == 0
    path.chmod(0o644)
    path.write_bytes(b"mutated index")
    with pytest.raises(ValueError, match="wrong-sized input|Changed input"):
        cloud.parents()


def test_parent_index_rejects_changed_configured_pin(parent_bundle):
    _, _, _, config = parent_bundle
    value = cloud.read(config)
    value["parent_index_sha256"] = "0" * 64
    write(config, value)
    with pytest.raises(ValueError, match="configured identity"):
        cloud.parents()


def test_parent_index_rejects_archive_mutation(parent_bundle):
    _, archive, _, _ = parent_bundle
    archive.write_bytes(b"mutated archive")
    with pytest.raises(ValueError, match="wrong-sized input|Changed input"):
        cloud.parents()


def test_parent_index_rejects_wrong_source_inventory(parent_bundle):
    _, _, specification, _ = parent_bundle
    value = cloud.read(specification)
    value["sources"] = 3
    write(specification, value)
    with pytest.raises(ValueError, match="source inventory"):
        cloud.parents()


def test_parent_index_cannot_write_outside_dump(parent_bundle, tmp_path):
    scratch, _, _, _ = parent_bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    (scratch / "parent-index-20260925").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="under dump"):
        cloud.parents()
    assert not list(outside.iterdir())


@pytest.fixture
def scope_bundle(tmp_path, monkeypatch):
    scratch = tmp_path / "scope-dump"
    scratch.mkdir()
    specification = tmp_path / cloud.SCOPE_SPECIFICATION.name
    write(specification, cloud.read(cloud.SCOPE_SPECIFICATION))
    monkeypatch.setattr(cloud, "DUMP", scratch)
    monkeypatch.setattr(cloud, "SCOPE_SPECIFICATION", specification)
    return scratch, specification


def test_joint_scope_provisioning_preserves_unresolved_boundaries(scope_bundle):
    scratch, _ = scope_bundle
    first = cloud.scopes()
    assert first == cloud.scopes()
    assert first["counts"]["trials"] == 9007
    assert first["counts"]["full_witness_components"] == 8960
    assert first["counts"]["holdout_overlap_witnesses"] == 19
    assert first["counts"]["quarter_states"]["REQUIRED_UNPROVISIONED"] == 7791
    assert first["joint_independence_complete"] is False
    assert first["scientific_outcomes_provisioned"] is False
    assert first["original_local_inputs_reverified"] is False
    path = scratch / "cloud-joint-scope.json"
    assert cloud.digest(path) == first["sha256"]
    assert path.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("defect", ["complete_claim", "counts", "raw_size"])
def test_joint_scope_rejects_changed_specification_boundaries(scope_bundle, defect):
    _, path = scope_bundle
    specification = cloud.read(path)
    if defect == "complete_claim":
        specification["joint_independence_complete"] = True
    elif defect == "counts":
        specification["counts"]["trials"] -= 1
    else:
        specification["artifact"]["canonical_json_bytes"] -= 1
    write(path, specification)
    with pytest.raises(ValueError):
        cloud.scopes()


def test_joint_scope_rejects_changed_compressed_metadata(scope_bundle, tmp_path, monkeypatch):
    path = tmp_path / cloud.SCOPE_BUNDLE.name
    path.write_bytes(b"changed metadata bundle")
    monkeypatch.setattr(cloud, "SCOPE_BUNDLE", path)
    with pytest.raises(ValueError, match="wrong-sized input"):
        cloud.scopes()


@pytest.mark.parametrize("name", [
    "cloud-joint-scope.json", "cloud-joint-scope.part", "cloud-joint-scope-receipt.json",
    "cloud-joint-scope-receipt.part",
])
def test_joint_scope_output_cannot_redirect_writes(scope_bundle, tmp_path, name):
    scratch, _ = scope_bundle
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"preserved original")
    (scratch / name).symlink_to(outside)
    with pytest.raises(ValueError, match="redirect its writes"):
        cloud.scopes()
    assert outside.read_bytes() == b"preserved original"
    assert (scratch / name).is_symlink()
    assert {item.name for item in scratch.iterdir()} == {name}


def test_joint_scope_rejects_changed_reused_metadata(scope_bundle):
    scratch, _ = scope_bundle
    cloud.scopes()
    path = scratch / "cloud-joint-scope.json"
    path.chmod(0o600)
    path.write_bytes(b"changed reused metadata")
    with pytest.raises(ValueError, match="wrong-sized input"):
        cloud.scopes()


def test_joint_scope_rejects_an_undeclared_unresolved_field(scope_bundle):
    _, path = scope_bundle
    specification = cloud.read(path)
    specification["whitelist"]["unresolved_scope"].remove("observed_header_parents")
    write(path, specification)
    with pytest.raises(ValueError, match="bindings or boundary"):
        cloud.scopes()


def test_joint_scope_binds_observed_parents_to_their_trials(scope_bundle):
    _, path = scope_bundle
    with gzip.open(cloud.SCOPE_BUNDLE, "rb") as stream:
        metadata = json.load(stream)
    edge = metadata["provenance"]["observed_native_header_parents"]["edges"]["edges"][0]
    trial = next(row for row in metadata["trials"] if row["trial_id"] == edge["trial_id"])
    trial["additional_parent_identities"].remove(edge["actual_raw_header_TPFORG"])
    with pytest.raises(ValueError, match="missing from unique additional"):
        cloud.check_scope_metadata(metadata, cloud.read(path))


def test_joint_scope_withdraws_the_receipt_when_new_metadata_fails(scope_bundle):
    scratch, path = scope_bundle
    cloud.scopes()
    specification = cloud.read(path)
    specification["joint_independence_complete"] = True
    write(path, specification)
    with pytest.raises(ValueError, match="Invalid joint scope"):
        cloud.scopes()
    assert not (scratch / "cloud-joint-scope-receipt.json").exists()


def test_joint_scope_rejects_stale_partials_and_withdraws_the_receipt(scope_bundle):
    scratch, _ = scope_bundle
    cloud.scopes()
    (scratch / "cloud-joint-scope.part").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Stale joint scope"):
        cloud.scopes()
    assert not (scratch / "cloud-joint-scope-receipt.json").exists()
    assert (scratch / "cloud-joint-scope.part").read_bytes() == b"incomplete"


def test_joint_scope_rejects_nonregular_outputs_and_withdraws_the_receipt(scope_bundle):
    scratch, _ = scope_bundle
    cloud.scopes()
    path = scratch / "cloud-joint-scope.json"
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="regular files"):
        cloud.scopes()
    assert not (scratch / "cloud-joint-scope-receipt.json").exists()


@pytest.mark.parametrize("moment", ["before-publication", "after-publication", "removed-after-publication"])
def test_joint_scope_rejects_input_drift_and_withdraws_the_receipt(scope_bundle, monkeypatch, moment):
    import cloud_null

    scratch, path = scope_bundle
    cloud.scopes()
    if moment == "before-publication":
        original = cloud.check_scope_metadata

        def changed(metadata, specification):
            original(metadata, specification)
            path.write_bytes(b"changed during verification")

        monkeypatch.setattr(cloud, "check_scope_metadata", changed)
    else:
        original = cloud_null.publish

        def changed(output, payload):
            original(output, payload)
            if moment == "removed-after-publication":
                path.unlink()
            else:
                path.write_bytes(b"changed after publication")

        monkeypatch.setattr(cloud_null, "publish", changed)
    error = FileNotFoundError if moment == "removed-after-publication" else ValueError
    with pytest.raises(error):
        cloud.scopes()
    assert not (scratch / "cloud-joint-scope-receipt.json").exists()


@pytest.fixture
def pixel_bundle(tmp_path, monkeypatch):
    scratch = tmp_path / "dump"
    scratch.mkdir()
    specification = tmp_path / "pixels.json"
    specification.write_bytes(cloud.PIXEL_SPECIFICATION.read_bytes())
    monkeypatch.setattr(cloud, "DUMP", scratch)
    monkeypatch.setattr(cloud, "PIXEL_SPECIFICATION", specification)
    return scratch, specification


def test_pixel_projection_preserves_all_opened_quarters_and_unresolved_inputs(pixel_bundle):
    scratch, _ = pixel_bundle
    result = cloud.pixels()
    assert result["sources"] == 16 and result["source_quarters"] == 201
    assert result["unique_payload_bytes"] == 950777235
    assert result["states"] == {
        "PUBLIC_BYTE_AND_HEADER_IDENTITIES_VERIFIED": 166,
        "FROZEN_PIXEL_INPUT_UNRESOLVED": 35,
    }
    assert not result["payloads_downloaded"]
    assert cloud.pixels() == result
    assert cloud.read(scratch / "cloud-pixels-receipt.json") == result
    ledger = cloud.read(result["frozen_pixel_inputs"]["path"])
    assert all(source["role"] == "development" for source in ledger["sources"].values())
    assert not ledger["automatic_pc_enabled"] and not ledger["joint_independence_complete"]


@pytest.mark.parametrize("field,value", [
    ("role", "evaluation"), ("origin_parent_kic", 1), ("raw_fits_sha256", "0" * 64),
])
def test_pixel_projection_rejects_source_identity_changes(pixel_bundle, field, value):
    _, specification = pixel_bundle
    item = cloud.read(specification)
    next(iter(item["sources"].values()))[field] = value
    write(specification, item)
    with pytest.raises(ValueError, match="source, role, parent or raw identity"):
        cloud.pixels()


@pytest.mark.parametrize("auxiliary", [False, True])
def test_pixel_projection_rejects_a_changed_receipt(pixel_bundle, auxiliary):
    scratch, specification = pixel_bundle
    cloud.pixels()
    item = cloud.read(specification)
    name = next(iter(item["files"]))
    if auxiliary:
        original = next(iter(item["origin"]["auxiliary_spatial_receipts"]))
        name = item["record_paths"][original]
    path = scratch / "cloud-pixel-records" / name
    path.chmod(0o644)
    path.write_bytes(b"changed receipt")
    with pytest.raises(ValueError, match="wrong-sized input|Changed input"):
        cloud.pixels()


def test_pixel_projection_rejects_undeclared_records(pixel_bundle):
    scratch, _ = pixel_bundle
    cloud.pixels()
    (scratch / "cloud-pixel-records/undeclared.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        cloud.pixels()


def test_pixel_projection_rejects_materialized_quarter_loss(pixel_bundle):
    scratch, _ = pixel_bundle
    cloud.pixels()
    path = scratch / "cloud-pixel-ledger.json"
    ledger = cloud.read(path)
    source = next(iter(ledger["sources"].values()))
    source["quarters"].pop(next(iter(source["quarters"])))
    write(path, ledger)
    with pytest.raises(ValueError, match="ledger changed"):
        cloud.pixels()


def test_pixel_projection_rejects_another_public_selection(pixel_bundle, monkeypatch, tmp_path):
    path = tmp_path / "public.json"
    item = cloud.read(cloud.PUBLIC_SPECIFICATION)
    item["sources"].pop(next(iter(item["sources"])))
    write(path, item)
    monkeypatch.setattr(cloud, "PUBLIC_SPECIFICATION", path)
    with pytest.raises(ValueError, match="opened public selection"):
        cloud.pixels()


def test_pixel_projection_rejects_changes_to_original_quarter_coverage(pixel_bundle):
    _, specification = pixel_bundle
    item = cloud.read(specification)
    source = next(iter(item["sources"].values()))
    source["quarters"].pop(next(iter(source["quarters"])))
    item["source_quarters"] -= 1
    write(specification, item)
    with pytest.raises(ValueError, match="source-quarter projection changed"):
        cloud.pixels()


def test_pixel_projection_cannot_write_outside_dump(pixel_bundle, tmp_path):
    scratch, _ = pixel_bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    (scratch / "cloud-pixel-records").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="under dump"):
        cloud.pixels()
    assert not list(outside.iterdir())


@pytest.fixture
def control_pixel_bundle(tmp_path, monkeypatch):
    scratch = tmp_path / "control-dump"
    scratch.mkdir()
    specification = tmp_path / "control-pixels.json"
    specification.write_bytes(cloud.CONTROL_PIXEL_SPECIFICATION.read_bytes())
    public = tmp_path / "public-controls.json"
    public.write_bytes(cloud.PUBLIC_CONTROL_SPECIFICATION.read_bytes())
    monkeypatch.setattr(cloud, "DUMP", scratch)
    monkeypatch.setattr(cloud, "CONTROL_PIXEL_SPECIFICATION", specification)
    monkeypatch.setattr(cloud, "PUBLIC_CONTROL_SPECIFICATION", public)
    return scratch, specification, public


def test_control_pixel_projection_preserves_available_and_unprovisioned_inputs(control_pixel_bundle):
    scratch, _, _ = control_pixel_bundle
    result = cloud.control_pixels()
    assert result["sources"] == 14 and result["source_quarters"] == 162
    assert result["states"] == {
        "PUBLIC_BYTE_AND_HEADER_IDENTITIES_VERIFIED": 51,
        "FROZEN_PIXEL_INPUT_UNRESOLVED": 111,
    }
    assert result["unique_payload_bytes"] == 116274370
    assert not result["payloads_downloaded"]
    assert cloud.control_pixels() == result
    assert cloud.read(scratch / "cloud-control-pixel-receipt.json") == result
    ledger = cloud.read(scratch / "cloud-control-pixel-ledger.json")
    assert {source["cohort"] for source in ledger["sources"].values()} == {
        "native_eb", "native_confirmed_extra", "periodic_planet", "shallow_planet_extra",
    }
    assert all(source["role"] == "development" for source in ledger["sources"].values())
    assert not ledger["automatic_pc_enabled"] and not ledger["joint_independence_complete"]
    for source in ledger["sources"].values():
        if source["cohort"] != "native_eb":
            assert all(item["kind"] == "UNAVAILABLE" for item in source["quarters"].values())
    assert {path.name for path in scratch.iterdir()} == {
        "cloud-control-pixel-records", "cloud-control-pixel-ledger.json", "cloud-control-pixel-receipt.json",
    }


@pytest.mark.parametrize("field,value", [
    ("role", "evaluation"), ("origin_parent_kic", 1), ("raw_fits_sha256", "0" * 64),
    ("source_id", "1"), ("cohort", "ordinary"),
])
def test_control_pixels_reject_source_binding_changes(control_pixel_bundle, field, value):
    scratch, specification, _ = control_pixel_bundle
    item = cloud.read(specification)
    next(iter(item["sources"].values()))[field] = value
    write(specification, item)
    with pytest.raises(ValueError, match="source, role, parent or raw identity"):
        cloud.control_pixels()
    assert not list(scratch.iterdir())


def test_control_pixels_reject_another_public_specification(control_pixel_bundle):
    scratch, _, public = control_pixel_bundle
    item = cloud.read(public)
    item["sources"].pop(next(iter(item["sources"])))
    write(public, item)
    with pytest.raises(ValueError, match="opened public selection"):
        cloud.control_pixels()
    assert not list(scratch.iterdir())


def test_control_pixels_bind_each_public_source_cohort(control_pixel_bundle):
    scratch, specification, public = control_pixel_bundle
    expected = cloud.read(public)
    next(iter(expected["sources"].values()))["opened_cohort"] = "shallow_planet_extra"
    write(public, expected)
    item = cloud.read(specification)
    item["public_inputs_sha256"] = cloud.digest(public)
    write(specification, item)
    with pytest.raises(ValueError, match="source, role, parent or raw identity"):
        cloud.control_pixels()
    assert not list(scratch.iterdir())


def test_control_pixels_reject_loss_of_an_unprovisioned_quarter(control_pixel_bundle):
    scratch, specification, _ = control_pixel_bundle
    item = cloud.read(specification)
    source = next(source for source in item["sources"].values() if source["cohort"] == "shallow_planet_extra")
    source["quarters"].pop(next(iter(source["quarters"])))
    item["source_quarters"] -= 1
    write(specification, item)
    with pytest.raises(ValueError, match="source-quarter projection changed"):
        cloud.control_pixels()
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("unprovisioned", [False, True])
def test_control_pixels_reject_changed_available_or_unprovisioned_receipts(control_pixel_bundle, unprovisioned):
    scratch, specification, _ = control_pixel_bundle
    cloud.control_pixels()
    item = cloud.read(specification)
    cohort = "shallow_planet_extra" if unprovisioned else "native_eb"
    source = next(source for source in item["sources"].values() if source["cohort"] == cohort)
    quarter = next(iter(source["quarters"].values()))
    original = quarter["nominal_path" if unprovisioned else "path"]
    path = scratch / "cloud-control-pixel-records" / item["record_paths"][original]
    path.chmod(0o644)
    path.write_bytes(b"changed receipt")
    with pytest.raises(ValueError, match="wrong-sized input|Changed input"):
        cloud.control_pixels()


def test_control_pixels_reject_undeclared_records(control_pixel_bundle):
    scratch, _, _ = control_pixel_bundle
    cloud.control_pixels()
    (scratch / "cloud-control-pixel-records/undeclared.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        cloud.control_pixels()


@pytest.mark.parametrize("name", [
    "cloud-control-pixel-records", "cloud-control-pixel-ledger.json",
    "cloud-control-pixel-ledger.part", "cloud-control-pixel-receipt.json",
    "cloud-control-pixel-receipt.part",
])
def test_control_pixel_outputs_cannot_redirect_writes(control_pixel_bundle, tmp_path, name):
    scratch, _, _ = control_pixel_bundle
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"preserved original")
    (scratch / name).symlink_to(outside)
    with pytest.raises(ValueError, match="redirect their writes"):
        cloud.control_pixels()
    assert outside.read_bytes() == b"preserved original"
    assert (scratch / name).is_symlink()
    assert {path.name for path in scratch.iterdir()} == {name}


def test_control_pixels_reject_an_undeclared_dangling_symlink(control_pixel_bundle, tmp_path):
    scratch, _, _ = control_pixel_bundle
    cloud.control_pixels()
    (scratch / "cloud-control-pixel-records/extra.json").symlink_to(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="regular files under dump"):
        cloud.control_pixels()


def test_public_controls_cli_dispatches_to_explicit_fetcher(monkeypatch, capsys):
    import cloud_public

    sections = []
    monkeypatch.setattr(cloud_public, "fetch_controls", lambda section: sections.append(section) or {"status": "PUBLIC_INPUTS_VERIFIED"}, raising=False)
    monkeypatch.setattr(sys, "argv", ["cloud.py", "public-controls", "2nd"])
    assert cloud.main() == 0
    assert sections == ["2nd"]
    assert json.loads(capsys.readouterr().out)["status"] == "PUBLIC_INPUTS_VERIFIED"


def test_control_pixels_cli_dispatches_separately(monkeypatch, capsys):
    monkeypatch.setattr(cloud, "control_pixels", lambda: {"status": "OPENED_PIXEL_RECORDS_VERIFIED"})
    monkeypatch.setattr(sys, "argv", ["cloud.py", "control-pixels"])
    assert cloud.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "OPENED_PIXEL_RECORDS_VERIFIED"


def test_opened_truths_cli_requires_verified_provisioning(monkeypatch, capsys):
    from deterministic import cloud_opened

    sections = []
    monkeypatch.setattr(cloud_opened, "provision_truths", lambda section: sections.append(section) or {"status": "OPENED_TRUTHS_VERIFIED"})
    monkeypatch.setattr(sys, "argv", ["cloud.py", "opened-truths", "2nd"])
    assert cloud.main() == 0
    assert sections == ["2nd"]
    assert json.loads(capsys.readouterr().out)["status"] == "OPENED_TRUTHS_VERIFIED"


@pytest.mark.parametrize("verify", [False, True])
def test_opened_corpus_cli_requires_explicit_destination(monkeypatch, capsys, tmp_path, verify):
    from deterministic import cloud_opened_corpus

    calls = []

    def provision(**options):
        calls.append(options)
        return {"status": "OPENED_ORIGINAL180_CORPUS_PARTIAL"}

    def checked(**options):
        return provision(**options), {}

    monkeypatch.setattr(cloud_opened_corpus, "cloud_destination", lambda _: False)
    monkeypatch.setattr(cloud_opened_corpus, "provision", provision)
    monkeypatch.setattr(cloud_opened_corpus, "verified_corpus", checked)
    args = ["cloud.py", "opened-corpus", "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args + (["--verify"] if verify else []))
    assert cloud.main() == 0
    expected = {"destination": tmp_path}
    if not verify:
        expected["minimum_free_bytes"] = 20 * 1024**3
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out)["status"] == "OPENED_ORIGINAL180_CORPUS_PARTIAL"


def test_opened_corpus_cli_does_not_report_failed_consumer_verification(monkeypatch, capsys, tmp_path):
    from deterministic import cloud_opened_corpus

    def fail(**_):
        raise ValueError("Corpus output drifted")

    monkeypatch.setattr(cloud_opened_corpus, "verified_corpus", fail)
    monkeypatch.setattr(sys, "argv", ["cloud.py", "opened-corpus", "--out", str(tmp_path), "--verify"])
    with pytest.raises(ValueError, match="output drifted"):
        cloud.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("path", ["dump/cloud-volumes", "dump/redirect", "dump/../outside"])
def test_opened_corpus_relative_cli_accepts_only_permanent_dump_link(monkeypatch, tmp_path, path):
    from deterministic import cloud_opened, cloud_opened_corpus

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "dump").symlink_to(scratch, target_is_directory=True)
    (scratch / "redirect").symlink_to(tmp_path / "outside", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    for module in (cloud, cloud_opened, cloud_opened_corpus):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(cloud_opened, "dump_path", lambda out: out.resolve())
    calls = []

    def checked(**options):
        calls.append(options)
        raise ValueError("Verified cache reached")

    monkeypatch.setattr(cloud_opened, "verified_inputs", checked)
    error = "Verified cache reached" if path == "dump/cloud-volumes" else "symlink|traverse"
    with pytest.raises(ValueError, match=error):
        cloud.opened_corpus(Path(path))
    assert calls == ([{"destination": scratch / "cloud-opened"}] if path == "dump/cloud-volumes" else [])


@pytest.mark.parametrize("cohorts", [[], ["periodic_planet", "sparse_injection", "ttv_injection"], ["ordinary", "native_eb"]])
def test_opened_controls_cli_passes_only_explicit_cohorts(monkeypatch, capsys, cohorts):
    from deterministic import cloud_opened

    calls = []

    def provision(section, **options):
        calls.append((section, options))
        return {"status": "OPENED_METADATA_DRY_RUN"}

    monkeypatch.setattr(cloud_opened, "provision", provision)
    arguments = ["cloud.py", "opened-controls", "2nd", "--dry-run", "--source-limit", "3"]
    for cohort in cohorts:
        arguments += ["--cohort", cohort]
    monkeypatch.setattr(sys, "argv", arguments)
    assert cloud.main() == 0
    assert calls == [("2nd", {"source_limit": 3, "dry_run": True, "cohorts": cohorts or None})]
    assert json.loads(capsys.readouterr().out)["status"] == "OPENED_METADATA_DRY_RUN"


def test_opened_truth_refs_cli_preserves_refs_and_dependencies(monkeypatch, capsys, tmp_path):
    from deterministic import cloud_opened

    run = tmp_path / "prepared"
    reference = {"123": {"truth_index": 0, "line_sha256": "a" * 64}}
    dependency = tmp_path / "truth.jsonl"
    calls = []
    monkeypatch.setattr(cloud_opened, "truth_refs", lambda path: calls.append(path) or (reference, {dependency: "b" * 64}))
    monkeypatch.setattr(sys, "argv", ["cloud.py", "opened-truth-refs", "--run", str(run)])
    assert cloud.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == [run]
    assert result["truth_refs"] == reference
    assert result["dependencies"] == {str(dependency): "b" * 64}
    assert result["sources"] == 1
    assert result["prepared_fsc_verified"] is True
    assert all(result[flag] is False for flag in cloud_opened.FLAGS)


def test_opened_truth_refs_cli_does_not_report_failed_verification(monkeypatch, capsys, tmp_path):
    from deterministic import cloud_opened

    def fail(path):
        raise ValueError("Prepared FSC changed")

    monkeypatch.setattr(cloud_opened, "truth_refs", fail)
    monkeypatch.setattr(sys, "argv", ["cloud.py", "opened-truth-refs", "--run", str(tmp_path)])
    with pytest.raises(ValueError, match="Prepared FSC changed"):
        cloud.main()
    assert capsys.readouterr().out == ""


@pytest.fixture
def prepared_controls(tmp_path, monkeypatch):
    import cloud_public

    scratch = tmp_path / "prepared-control-dump"
    folder = scratch / "cloud-controls"
    folder.mkdir(parents=True)
    raw = {
        "time_bkjd": np.arange(200, dtype=np.float64) * 0.02,
        "cadenceno": np.arange(200, dtype=np.int32),
        "quarter": np.ones(200, dtype=np.int16),
        "psf_flat_kbonus_flux": np.full(200, 1000.0),
        "psf_flat_kbonus_flux_err": np.ones(200),
        "sap_quality": np.zeros(200, dtype=np.int32),
        "flatten_mask": np.zeros(200, dtype=np.int32),
        "qdiag_time_bkjd": np.arange(240, dtype=np.float64) * 0.02,
        "qdiag_cadenceno": np.arange(240, dtype=np.int32),
    }
    arrays = {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                     "bytes": value.nbytes, "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
              for name, value in raw.items()}
    declaration = {
        "metadata": {"source_id": "123", "shard": 7, "origin_tpf_kic": 99,
                     "raw_fits_sha256": "1" * 64},
        "opened_cohort": "shallow_planet_extra", "opened_scope": "DEVELOPMENT_EXTRA",
        "origin": {"original_rank": 1}, "arrays": arrays,
        "raw_view_sha256": cloud_public.curve_digest(cloud_public.photometry(raw, "psf_flat")),
        "prepared_records_sha256": hashlib.sha256(cloud_public.prepare_records(raw).tobytes()).hexdigest(),
    }
    public = tmp_path / "prepared-public-controls.json"
    write(public, {"sources": {"123": declaration}})
    monkeypatch.setattr(cloud, "PUBLIC_CONTROL_SPECIFICATION", public)
    path = folder / "123.npz"
    np.savez_compressed(path, **raw)
    sources = {"123": {key: declaration[key] for key in ("metadata", "opened_cohort", "opened_scope", "origin")}}
    sources["123"].update(arrays=path.name, sha256=cloud.digest(path))
    manifest = folder / "input-manifest.json"
    ids = folder / "source-ids.json"
    boundaries = {"role": "development", "survey_context_complete": False,
                  "automatic_pc_enabled": False, "joint_independence_complete": False}
    write(manifest, {"schema": 1, "measurement_scope": "OPENED_DEVELOPMENT_ONLY", "sources": sources, **boundaries})
    write(ids, list(sources))
    write(folder / "receipt.json", {
        "status": "PUBLIC_INPUTS_VERIFIED",
        "scope": "OPENED_DEVELOPMENT_ONLY", **boundaries,
        "specification_sha256": cloud.digest(cloud.PUBLIC_CONTROL_SPECIFICATION),
        "manifest_sha256": cloud.digest(manifest), "source_ids_sha256": cloud.digest(ids),
    })
    monkeypatch.setattr(cloud, "DUMP", scratch)
    return folder


def test_control_preparation_uses_verified_separate_manifest(prepared_controls, monkeypatch, tmp_path):
    from deterministic import run

    calls = []
    monkeypatch.setattr(run, "prepare", calls.append)
    result = cloud.prepare_public(tmp_path / "output", controls=True)
    assert result["status"] == "PREPARED" and result["sources"] == 1
    assert len(calls) == 1 and calls[0].input_manifest == prepared_controls / "input-manifest.json"
    assert calls[0].source_ids == prepared_controls / "source-ids.json"


@pytest.mark.parametrize("name", ["input-manifest.json", "source-ids.json"])
def test_control_preparation_rejects_changed_hash_bound_inputs(prepared_controls, name, tmp_path):
    (prepared_controls / name).write_text("{}")
    with pytest.raises(ValueError, match="receipt or manifest changed"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("field,value", [
    ("scope", "SEALED"), ("role", "evaluation"), ("survey_context_complete", True),
    ("automatic_pc_enabled", True), ("joint_independence_complete", True),
])
def test_control_preparation_rejects_changed_release_boundaries(prepared_controls, field, value, tmp_path):
    path = prepared_controls / "receipt.json"
    receipt = cloud.read(path)
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(ValueError, match="opened development boundaries"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


def test_control_preparation_rejects_changed_cohort_even_with_new_receipt_hash(prepared_controls, tmp_path):
    path = prepared_controls / "input-manifest.json"
    manifest = cloud.read(path)
    next(iter(manifest["sources"].values()))["opened_cohort"] = "ordinary"
    write(path, manifest)
    receipt_path = prepared_controls / "receipt.json"
    receipt = cloud.read(receipt_path)
    receipt["manifest_sha256"] = cloud.digest(path)
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="source identity, cohort or scope"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


def test_control_preparation_rejects_changed_membership_even_with_new_receipt_hash(prepared_controls, tmp_path):
    path = prepared_controls / "source-ids.json"
    write(path, ["456"])
    receipt_path = prepared_controls / "receipt.json"
    receipt = cloud.read(receipt_path)
    receipt["source_ids_sha256"] = cloud.digest(path)
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="source membership or order"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


def rebind_control_container(folder):
    path = folder / "input-manifest.json"
    manifest = cloud.read(path)
    manifest["sources"]["123"]["sha256"] = cloud.digest(folder / "123.npz")
    write(path, manifest)
    receipt_path = folder / "receipt.json"
    receipt = cloud.read(receipt_path)
    receipt["manifest_sha256"] = cloud.digest(path)
    write(receipt_path, receipt)


@pytest.mark.parametrize("mutation", ["qdiag", "mask", "dtype"])
def test_control_preparation_reverifies_rehashed_array_contents(prepared_controls, mutation, tmp_path):
    path = prepared_controls / "123.npz"
    with np.load(path, allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    if mutation == "qdiag":
        raw["qdiag_time_bkjd"][0] += 1
    elif mutation == "mask":
        raw["flatten_mask"][0] = 1
    else:
        raw["cadenceno"] = raw["cadenceno"].astype(np.int64)
    np.savez_compressed(path, **raw)
    rebind_control_container(prepared_controls)
    with pytest.raises(ValueError, match="dtype, shape or content differs"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


def test_control_preparation_rejects_rehashed_duplicate_npz_keys(prepared_controls, tmp_path):
    path = prepared_controls / "123.npz"
    with zipfile.ZipFile(path, "a") as archive:
        content = archive.read("flatten_mask.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("flatten_mask.npy", content)
    rebind_control_container(prepared_controls)
    with pytest.raises(ValueError, match="duplicate keys"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("filename", ["another.npz", "../123.npz", "/tmp/123.npz"])
def test_control_preparation_rejects_unbound_array_paths(prepared_controls, filename, tmp_path):
    path = prepared_controls / "input-manifest.json"
    manifest = cloud.read(path)
    manifest["sources"]["123"]["arrays"] = filename
    write(path, manifest)
    receipt_path = prepared_controls / "receipt.json"
    receipt = cloud.read(receipt_path)
    receipt["manifest_sha256"] = cloud.digest(path)
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="regular source-bound paths"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()


def test_control_preparation_rejects_redirected_arrays(prepared_controls, tmp_path):
    path = prepared_controls / "123.npz"
    original = path.read_bytes()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="regular source-bound paths"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert outside.read_bytes() == original
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("field", ["raw_view_sha256", "prepared_records_sha256"])
def test_control_preparation_checks_frozen_measurement_hashes(prepared_controls, field, tmp_path):
    path = cloud.PUBLIC_CONTROL_SPECIFICATION
    specification = cloud.read(path)
    specification["sources"]["123"][field] = "0" * 64
    write(path, specification)
    receipt_path = prepared_controls / "receipt.json"
    receipt = cloud.read(receipt_path)
    receipt["specification_sha256"] = cloud.digest(path)
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="photometry differs from frozen"):
        cloud.prepare_public(tmp_path / "must-not-exist", controls=True)
    assert not (tmp_path / "must-not-exist").exists()
