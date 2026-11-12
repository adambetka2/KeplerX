"""Metadata bindings preserve frozen roles, full scene context and flux guards."""

import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.deterministic import cloud_null as pool
from pipeline.deterministic import null_base_manifest as producer
from pipeline.deterministic import signal_contracts
from pipeline.photometry import lc2


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def synchronize(case):
    write_json(
        case.pool_spec,
        {
            "provenance": {
                "original_inputs": [
                    {"origin_path": "discSSD/data/index.sqlite", **producer.reference(case.index)}
                ]
            }
        },
    )
    case.declaration["inputs"] = [producer.reference(case.pool_spec)]
    write_json(case.declaration_path, case.declaration)
    case.declaration_sha = pool.digest(case.declaration_path)


def replace_metadata(case, shard, records):
    volume = "discPCIe" if shard == 1 else "discSSD"
    path = case.root / volume / f"data/kbonus/lc2/shard-{shard:04d}/source.parquet"
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=1)
    with sqlite3.connect(case.index) as database:
        database.execute(
            "update files set bytes=?,sha256=? where volume=? and path=?",
            (
                path.stat().st_size,
                pool.digest(path),
                volume,
                str(path.relative_to(case.root / volume / "data")),
            ),
        )
    synchronize(case)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    index = root / "discSSD/data/index.sqlite"
    index.parent.mkdir(parents=True)
    groups = {0: [1, 2], 1: [4, 5], 2: [3]}
    metadata = {
        sid: {
            "source_id": sid,
            "origin_tpf_kic": 10 if sid == 4 else sid * 10,
            "raw_fits_sha256": hashlib.sha256(str(sid).encode()).hexdigest(),
            "ra_deg": 280.0 + sid / 1000,
            "dec_deg": 40.0,
            "phot_g_mean_mag": 15.0,
            "original_header_cards": f"OBJECT  = Source {sid}\n" + "COMMENT Original header card\n" * 1024,
        }
        for ids in groups.values()
        for sid in ids
    }
    with sqlite3.connect(index) as database:
        database.executescript(
            "create table sources (gaia_source_id text,volume text,shard integer);"
            "create table files (volume text,path text,collection text,bytes integer,sha256 text,status text,format text);"
        )
        for shard, ids in groups.items():
            volume = "discPCIe" if shard == 1 else "discSSD"
            folder = root / volume / f"data/kbonus/lc2/shard-{shard:04d}"
            folder.mkdir(parents=True)
            for sid in ids:
                database.execute("insert into sources values (?,?,?)", (str(sid), volume, shard))
            for name in ("source",) if shard == 1 else producer.PRODUCTS:
                records = (
                    [metadata[sid] for sid in ids]
                    if name == "source"
                    else [
                        {
                            "source_id": sid,
                            "time_bkjd": 100.0 + i,
                            "quarter": 1,
                            "psf_flat_kbonus_flux": 1000.0,
                        }
                        for sid in ids
                        for i in range(3)
                    ]
                )
                path = folder / f"{name}.parquet"
                pq.write_table(
                    pa.Table.from_pylist(records), path, row_group_size=1 if name == "source" else 3
                )
                database.execute(
                    "insert into files values (?,?,?,?,?,?,?)",
                    (
                        volume,
                        str(path.relative_to(root / volume / "data")),
                        "kbonus",
                        path.stat().st_size,
                        pool.digest(path),
                        "available",
                        "parquet",
                    ),
                )
    rows = [
        {
            "source_id": str(sid),
            "shard": str(shard),
            "origin_parent": str(metadata[sid]["origin_tpf_kic"]),
            "raw_fits_sha256": metadata[sid]["raw_fits_sha256"],
            "metadata_dependency_component_id": hashlib.sha256(f"component:{sid}".encode()).hexdigest(),
            "linked_existing_scopes": "",
            "linked_full_witness_component_ids": "",
            **pool.EXPOSURE,
        }
        for sid, shard in [(3, 2), (1, 0)]
    ]
    csv_path = tmp_path / "development-membership.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pool.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    binding = {
        **producer.reference(csv_path),
        "sources": 2,
        "role": "development",
        "ordered_source_ids_sha256": hashlib.sha256(pool.canonical(["3", "1"])).hexdigest(),
    }
    declaration = {
        "schema": 1,
        "status": "FIRST_OUTCOME_BLIND_NULL_MEMBERSHIP_DECLARATION",
        "membership_and_seeds_frozen": True,
        "selection_seed": 20261002,
        "transform_seed": 20261002,
        "execution_authorized": False,
        "counts": {"development": 2, "evaluation": 2},
        "artifacts": {"development-membership.csv": binding},
        "same_base_recipes": {"id": "upper-clipped-quarter-nulls-v1", "seed": 20261002},
    }
    result = SimpleNamespace(
        root=root,
        index=index,
        pool_spec=tmp_path / "cloud-null-pool.json",
        declaration_path=tmp_path / "declaration.json",
        declaration=declaration,
        membership=csv_path,
        rows=rows,
        metadata=metadata,
        out=tmp_path / "dump/base",
    )
    synchronize(result)
    monkeypatch.setattr(producer, "POOL_SPECIFICATION", result.pool_spec)
    monkeypatch.setattr(producer, "QUOTA", 2)
    return result


def produce(case, role="development"):
    return producer.produce(
        case.declaration_path, case.declaration_sha, case.membership, role, case.root, case.index, case.out
    )


def seal_arrays(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name not in {"cadence.parquet", "qdiag.parquet"}, "Scientific product bytes were opened"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(lc2, "load_source", lambda *args: pytest.fail("Source arrays were accessed"))
    monkeypatch.setattr(lc2, "read_source", lambda *args: pytest.fail("Source arrays were accessed"))


def test_metadata_only_binding_preserves_order_and_all_original_shards(case, monkeypatch):
    seal_arrays(monkeypatch)
    manifest = produce(case)
    emitted = json.loads((case.out / "input-manifest.json").read_bytes())
    assert list(emitted["sources"]) == manifest["ordered_source_ids"] == ["3", "1"]
    assert [item["original_pool_row"] for item in emitted["sources"].values()] == case.rows
    assert manifest["schema"] == 3 and manifest["protocol"] == lc2.LC2_MANIFEST
    assert manifest["role"] == "development" and not manifest["execution_authorized"]
    assert manifest["scene_coverage"]["complete_original_index"] is True
    assert manifest["scene_coverage"]["sources"] == 5
    scene = [json.loads(line) for line in (case.out / "source-scene.jsonl").read_bytes().splitlines()]
    assert {row["source_id"] for row in scene} == {"1", "2", "3", "4", "5"}
    assert {row["shard"] for row in scene} == {0, 1, 2}
    assert all(
        set(row)
        == {"source_id", "ra_deg", "dec_deg", "origin_tpf_kic", "phot_g_mean_mag", "shard", "raw_fits_sha256"}
        for row in scene
    )
    for sid, item in manifest["sources"].items():
        assert item["metadata"]["original_header_cards"] == case.metadata[int(sid)]["original_header_cards"]
        path = case.root / item["lc2_folder"] / "source.parquet"
        assert manifest["lc2_products"][item["lc2_folder"]]["source"]["sha256"] == pool.digest(path)
    assert len(manifest["lc2_products"]) == 2
    assert all(set(products) == set(producer.PRODUCTS) for products in manifest["lc2_products"].values())
    assert not manifest["cadence_or_qdiag_arrays_read"] and not manifest["null_flux_constructed"]
    assert "INDEX_HASH_BOUND" in manifest["product_verification"]["cadence"]


@pytest.mark.parametrize("derived", [False, True])
def test_full_scene_consumer_retains_an_unselected_cross_shard_neighbour(case, monkeypatch, derived):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vetting"))
    from pipeline.vetting import vetting

    manifest = produce(case)
    input_path = case.out / "input-manifest.json"
    if derived:
        metadata_only = dict(
            manifest,
            schema=1,
            null_construction={"base_manifest": {"path": str(input_path), "sha256": pool.digest(input_path)}},
        )
        input_path = case.out / "derived-metadata.json"
        write_json(input_path, metadata_only)
    config = case.out / "vetting.json"
    write_json(
        config,
        {"input_manifest": str(input_path), "queue": str(case.out / "queue.json")},
    )
    selected = {sid: item["metadata"] for sid, item in manifest["sources"].items()}
    write_json(case.out / "preparation.json", {"sources": list(selected.values())})
    monkeypatch.setattr(vetting, "CONFIG", config)
    monkeypatch.setattr(vetting, "ROOT", case.root)
    scene = dict(vetting.scene_rows([], selected))
    assert set(scene) == {"1", "2", "3", "4", "5"}
    assert scene["4"]["origin_tpf_kic"] == selected["1"]["origin_tpf_kic"]
    assert scene["4"]["shard"] not in {row["shard"] for row in selected.values()}


@pytest.mark.parametrize("damage", ["base_hash", "scene_hash"])
def test_derived_complete_scene_requires_unchanged_base_and_scene_pins(case, damage):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vetting"))
    from pipeline.vetting import vetting

    manifest = produce(case)
    path = case.out / "input-manifest.json"
    derived = dict(
        manifest,
        schema=1,
        null_construction={"base_manifest": {"path": str(path), "sha256": pool.digest(path)}},
    )
    if damage == "base_hash":
        derived["null_construction"]["base_manifest"]["sha256"] = "f" * 64
    else:
        derived["scene_metadata"] = {"path": "source-scene.jsonl", "sha256": "f" * 64}
    with pytest.raises(ValueError, match="scene metadata changed|Complete null scene differs"):
        vetting.complete_scene_index(path, derived)


def test_produced_schema3_manifest_is_consumed_by_the_hash_bound_reader(case):
    manifest = produce(case)
    metadata, raw = lc2.load_source(case.root, "1", case.out / "input-manifest.json")
    assert metadata == {
        **manifest["sources"]["1"]["metadata"],
        "input_manifest": str(case.out / "input-manifest.json"),
    }
    assert raw["source_id"].tolist() == [1, 1, 1]


def test_evaluation_metadata_stays_unauthorized_and_cannot_construct_flux(case, monkeypatch):
    binding = dict(case.declaration["artifacts"]["development-membership.csv"], role="evaluation")
    case.declaration["artifacts"]["evaluation-membership.csv"] = binding
    synchronize(case)
    seal_arrays(monkeypatch)
    manifest = produce(case, "evaluation")
    assert manifest["role"] == "evaluation" and manifest["evaluation_authorization_required"]
    assert not manifest["execution_authorized"] and not manifest["quarter_scene_independence_certified"]
    monkeypatch.setattr(
        signal_contracts, "load_source", lambda *args: pytest.fail("Evaluation arrays were opened")
    )
    with pytest.raises(ValueError, match="frozen scoped evaluation policy"):
        signal_contracts.construct(
            SimpleNamespace(
                input_manifest=case.out / "input-manifest.json",
                variant="inverted",
                seed=20261002,
                disposition_policy=None,
                out=case.out / "flux",
            )
        )
    assert not (case.out / "flux").exists()


@pytest.mark.parametrize("damage", ["declaration", "csv", "index", "pool"])
def test_changed_frozen_input_bytes_refuse_before_output(case, damage):
    path = {
        "declaration": case.declaration_path,
        "csv": case.membership,
        "index": case.index,
        "pool": case.pool_spec,
    }[damage]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed|Bound metadata changed"):
        produce(case)
    assert not case.out.exists()


@pytest.mark.parametrize("damage", ["quota", "order", "role"])
def test_no_quota_shrink_order_change_or_role_reassignment(case, damage):
    binding = case.declaration["artifacts"]["development-membership.csv"]
    if damage == "quota":
        binding["sources"] = 1
    elif damage == "order":
        binding["ordered_source_ids_sha256"] = "f" * 64
    else:
        binding["role"] = "evaluation"
    synchronize(case)
    with pytest.raises(ValueError, match="quota or role|source order"):
        produce(case)
    assert not case.out.exists()


@pytest.mark.parametrize("field,value", [("origin_tpf_kic", 999), ("raw_fits_sha256", "f" * 64)])
def test_selected_source_must_match_original_parent_and_raw(case, field, value):
    changed = dict(case.metadata[3], **{field: value})
    replace_metadata(case, 2, [changed])
    with pytest.raises(ValueError, match="Original null source binding differs"):
        produce(case)
    assert not (case.out / "input-manifest.json").exists()


def test_indexed_shard_must_match_original_membership(case):
    with sqlite3.connect(case.index) as database:
        database.execute("update sources set shard=0 where gaia_source_id='3'")
    synchronize(case)
    with pytest.raises(ValueError, match="Original null shard differs"):
        produce(case)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "extra"])
def test_complete_scene_refuses_missing_repeated_or_unindexed_sources(case, damage):
    records = [case.metadata[4], case.metadata[5]]
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records.append(records[0])
    else:
        records.append(dict(records[0], source_id=99))
    replace_metadata(case, 1, records)
    with pytest.raises(ValueError, match="Complete original shard scene coverage differs"):
        produce(case)
    assert not (case.out / "input-manifest.json").exists()


@pytest.mark.parametrize("damage", ["missing", "duplicate", "invalid"])
def test_all_three_original_reader_products_are_required(case, damage):
    with sqlite3.connect(case.index) as database:
        if damage == "missing":
            database.execute("delete from files where path like 'kbonus/lc2/shard-0000/qdiag%'")
        elif damage == "duplicate":
            database.execute(
                "insert into files select * from files where path like 'kbonus/lc2/shard-0000/qdiag%'"
            )
        else:
            database.execute(
                "update files set sha256='invalid' where path like 'kbonus/lc2/shard-0000/qdiag%'"
            )
    synchronize(case)
    with pytest.raises(ValueError, match="indexed product|Invalid indexed"):
        produce(case)


def test_source_hash_precedes_metadata_and_original_products_stay_read_only(case, monkeypatch):
    source = case.root / "discSSD/data/kbonus/lc2/shard-0000/source.parquet"
    original = source.read_bytes()
    source.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    corrupt = source.read_bytes()
    parquet = pq.ParquetFile

    def guarded(buffer):
        assert buffer.getvalue() != corrupt, "Unverified metadata was decoded"
        return parquet(buffer)

    monkeypatch.setattr(pq, "ParquetFile", guarded)
    with pytest.raises(ValueError, match="Bound metadata changed"):
        produce(case)


def test_existing_output_and_unbound_sqlite_journal_refuse(case):
    Path(str(case.index) + "-wal").write_bytes(b"unbound")
    with pytest.raises(ValueError, match="unbound SQLite journal"):
        produce(case)
    assert not case.out.exists()
    Path(str(case.index) + "-wal").unlink()
    produce(case)
    with pytest.raises(FileExistsError):
        produce(case)
