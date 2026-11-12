import csv
import json
import os
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.deterministic import full_survey, source_ledger


def corpus(tmp_path):
    sources = [
        {
            "source_id": 1,
            "origin_tpf_kic": 11,
            "raw_fits_sha256": "a" * 64,
            "raw_fits_bytes": 100,
        },
        {
            "source_id": 2,
            "origin_tpf_kic": 22,
            "raw_fits_sha256": "b" * 64,
            "raw_fits_bytes": 200,
        },
    ]
    for volume in ("discSSD", "discPCIe"):
        (tmp_path / volume / "data").mkdir(parents=True)
    index = tmp_path / "discSSD/data/index.sqlite"
    with sqlite3.connect(index) as database:
        database.executescript(
            "create table sources (gaia_source_id text primary key, "
            "volume text, shard integer);"
            "create table files (volume text,path text,collection text,format text,"
            "bytes integer,sha256 text,status text);"
        )
        for shard, row in enumerate(sources, 1):
            volume = "discSSD" if shard == 1 else "discPCIe"
            database.execute(
                "insert into sources values (?,?,?)",
                (str(row["source_id"]), volume, shard),
            )
            folder = tmp_path / volume / "data/kbonus/lc2" / f"shard-{shard:04d}"
            folder.mkdir(parents=True)
            for name in sorted(source_ledger.PRODUCTS):
                path = folder / f"{name}.parquet"
                table = pa.Table.from_pylist(
                    [row] if name == "source" else [{"value": shard}]
                )
                pq.write_table(table, path)
                database.execute(
                    "insert into files values (?,?,?,?,?,?,?)",
                    (
                        volume,
                        str(path.relative_to(tmp_path / volume / "data")),
                        "kbonus",
                        "parquet",
                        path.stat().st_size,
                        source_ledger.digest(path),
                        "available",
                    ),
                )
    synchronize_indexes(tmp_path)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(["2", "1"]))
    return selection, sources


def synchronize_indexes(root):
    index = root / "discSSD/data/index.sqlite"
    shutil.copyfile(index, root / "discPCIe/data/index.sqlite")
    manifest = {
        "lookup_sha256": source_ledger.digest(index),
        "primary_collections": {"kbonus": {"sources": 2}},
    }
    for volume in ("discSSD", "discPCIe"):
        (root / volume / "data/index.json").write_text(json.dumps(manifest))


def test_complete_input_ledger_preserves_order_and_reverifies_resume(tmp_path):
    selection, sources = corpus(tmp_path)
    out = tmp_path / "audit"
    result = source_ledger.freeze(tmp_path, selection, out, expected=2)
    with (out / "sources.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["source_id"] for row in rows] == ["2", "1"]
    assert rows[0]["raw_fits_sha256"] == sources[1]["raw_fits_sha256"]
    assert result["sources"] == 2 and result["shards"] == 2 and result["products"] == 12
    assert result["verified_input_bytes"] == sum(
        row["bytes"] for row in source_ledger.read(out / "products.json")
    )
    assert not result["scientific_outcomes_opened"]
    assert (
        source_ledger.freeze(tmp_path, selection, out, expected=2, resume=True)
        == result
    )
    changed = tmp_path / "discPCIe/data/kbonus/lc2/shard-0002/qdiag.parquet"
    changed.write_bytes(changed.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="LC2 input bytes differ"):
        source_ledger.freeze(tmp_path, selection, out, expected=2, resume=True)


def test_mounted_index_mismatch_stops_before_freeze(tmp_path):
    selection, _ = corpus(tmp_path)
    path = tmp_path / "discPCIe/data/index.sqlite"
    with sqlite3.connect(path) as database:
        database.execute("update sources set shard=3 where gaia_source_id='2'")
    manifest = tmp_path / "discPCIe/data/index.json"
    value = source_ledger.read(manifest)
    value["lookup_sha256"] = source_ledger.digest(path)
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Mounted corpus indexes differ"):
        source_ledger.freeze(tmp_path, selection, tmp_path / "audit", expected=2)


@pytest.mark.parametrize("damage", ["missing_product", "wrong_source"])
def test_incomplete_shard_or_source_identity_cannot_freeze(tmp_path, damage):
    selection, _ = corpus(tmp_path)
    index = tmp_path / "discSSD/data/index.sqlite"
    with sqlite3.connect(index) as database:
        if damage == "missing_product":
            database.execute("delete from files where path like '%quarter.parquet'")
        else:
            path = tmp_path / "discSSD/data/kbonus/lc2/shard-0001/source.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "source_id": 9,
                            "origin_tpf_kic": 11,
                            "raw_fits_sha256": "a" * 64,
                            "raw_fits_bytes": 100,
                        }
                    ]
                ),
                path,
            )
            database.execute(
                "update files set bytes=?,sha256=? where volume=? and path=?",
                (
                    path.stat().st_size,
                    source_ledger.digest(path),
                    "discSSD",
                    "kbonus/lc2/shard-0001/source.parquet",
                ),
            )
    synchronize_indexes(tmp_path)
    out = tmp_path / "audit"
    reason = (
        "exactly six input products"
        if damage == "missing_product"
        else "metadata does not close"
    )
    with pytest.raises(ValueError, match=reason):
        source_ledger.freeze(tmp_path, selection, out, expected=2)
    assert not (out / "freeze.json").exists()


def test_resume_rejects_changed_order_without_replacing_artifacts(tmp_path):
    selection, _ = corpus(tmp_path)
    out = tmp_path / "audit"
    source_ledger.freeze(tmp_path, selection, out, expected=2)
    before = source_ledger.digest(out / "sources.csv")
    selection.write_text(json.dumps(["1", "2"]))
    with pytest.raises(ValueError, match="resume identity changed"):
        source_ledger.freeze(tmp_path, selection, out, expected=2, resume=True)
    assert source_ledger.digest(out / "sources.csv") == before


def test_interrupted_audit_rehashes_completed_inputs_before_resume(
    tmp_path, monkeypatch
):
    selection, _ = corpus(tmp_path)
    out = tmp_path / "audit"
    original = source_ledger.verify_product
    checked = []

    def interrupted(root, record):
        checked.append(record["path"])
        if len(checked) == 4:
            raise RuntimeError("interrupted")
        return original(root, record)

    monkeypatch.setattr(source_ledger, "verify_product", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        source_ledger.freeze(tmp_path, selection, out, expected=2)
    assert source_ledger.read(out / "progress.json")["complete"] == 3
    assert not (out / "freeze.json").exists()
    completed = checked[:3]
    checked.clear()

    def resumed(root, record):
        checked.append(record["path"])
        return original(root, record)

    monkeypatch.setattr(source_ledger, "verify_product", resumed)
    result = source_ledger.freeze(tmp_path, selection, out, expected=2, resume=True)
    assert checked[:3] == completed
    assert result["products"] == len(checked) == 12


def test_input_change_during_hash_cannot_complete_freeze(tmp_path, monkeypatch):
    selection, _ = corpus(tmp_path)
    out = tmp_path / "audit"
    changed = tmp_path / "discSSD/data/kbonus/lc2/shard-0001/qdiag.parquet"
    original = source_ledger.digest

    def concurrent_change(path):
        checksum = original(path)
        if path == changed:
            path.write_bytes(path.read_bytes() + b"x")
        return checksum

    monkeypatch.setattr(source_ledger, "digest", concurrent_change)
    with pytest.raises(ValueError, match="changed during verification"):
        source_ledger.freeze(tmp_path, selection, out, expected=2)
    assert not (out / "freeze.json").exists()


def test_survey_ledger_rehashes_current_bytes_and_detects_later_changes(
    tmp_path, monkeypatch
):
    selection, _ = corpus(tmp_path)
    out = tmp_path / "audit"
    source_ledger.freeze(tmp_path, selection, out, expected=2)
    monkeypatch.setattr(full_survey, "ROOT", tmp_path)
    binding, stamps = full_survey.verified_input_ledger(out, selection, 2)
    assert binding["sha256"] == source_ledger.digest(out / "freeze.json")
    assert binding["sources"] == 2 and binding["products"] == 12
    full_survey.verify_input_stamps(stamps)
    path = tmp_path / "discSSD/data/kbonus/lc2/shard-0001/cadence.parquet"
    data = path.read_bytes()
    stat = path.stat()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="changed during"):
        full_survey.verify_input_stamps(stamps)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="input bytes"):
        full_survey.verified_input_ledger(out, selection, 2)


@pytest.mark.parametrize("changed", ["selection", "sources", "status", "products"])
def test_survey_rejects_incomplete_or_changed_input_ledger(
    tmp_path, monkeypatch, changed
):
    selection, _ = corpus(tmp_path)
    out = tmp_path / "audit"
    source_ledger.freeze(tmp_path, selection, out, expected=2)
    monkeypatch.setattr(full_survey, "ROOT", tmp_path)
    if changed == "selection":
        selection.write_text('["1", "2"]')
    elif changed == "sources":
        (out / "sources.csv").write_text("source_id\n2\n")
    elif changed == "products":
        (out / "products.json").write_text("[]")
    else:
        path = out / "freeze.json"
        frozen = source_ledger.read(path)
        frozen["status"] = "VERIFYING_LC2_INPUT_BYTES"
        source_ledger.write(path, frozen)
    with pytest.raises(ValueError, match="frozen LC2 input ledger"):
        full_survey.verified_input_ledger(out, selection, 2)


def test_survey_stage_binding_precedes_and_preserves_resume_identity(tmp_path):
    run_dir = tmp_path / "survey"
    identity = {"input_ledger": {"sha256": "a" * 64}}
    full_survey.bind_stage_inputs(run_dir, identity)
    assert not run_dir.exists()
    run_dir.mkdir()
    full_survey.bind_stage_inputs(run_dir, identity)
    with pytest.raises(ValueError, match="Completed artifact changed"):
        full_survey.bind_stage_inputs(
            run_dir, {"input_ledger": {"sha256": "b" * 64}}
        )
    unbound = tmp_path / "old-survey"
    unbound.mkdir()
    with pytest.raises(ValueError, match="lacks its frozen input binding"):
        full_survey.bind_stage_inputs(unbound, identity)
