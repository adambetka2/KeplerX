"""Bounded original-reader corpus transport and adversarial handoff checks."""

import gzip
import io
import json
import sqlite3
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deterministic import cloud_opened_corpus as corpus
from deterministic.test_cloud_opened import inventory, make_source

opened = corpus.opened
IDS = ["2052214955493700224", "2079311521898851200", "2052305184163296512"]


def bytes_table(table):
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def descriptor(payload):
    return {"bytes": len(payload), "sha256": corpus.checksum(payload)}


def pack(case, monkeypatch, members=None):
    case.spec["files"] = {
        name: descriptor(payload) for name, payload in case.members.items()
    }
    case.spec["uncompressed_member_bytes"] = sum(
        x["bytes"] for x in case.spec["files"].values()
    )
    for sid, row in case.spec["sources"].items():
        row["metadata"].update(case.spec["files"][row["metadata"]["member"]])
    for group in case.spec["original_lineage"].values():
        for row in group.values():
            row.update(case.spec["files"][row["member"]])
            row["original"].update({key: row[key] for key in ("bytes", "sha256")})
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload, kind in members or [
            (name, payload, tarfile.REGTYPE) for name, payload in case.members.items()
        ]:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.size = len(payload) if kind == tarfile.REGTYPE else 0
            archive.addfile(item, io.BytesIO(payload) if item.size else None)
    compressed = gzip.compress(raw.getvalue(), mtime=0)
    case.archive.write_bytes(compressed)
    case.spec["archive"] = {"path": case.archive.name, **descriptor(compressed)}
    payload = opened.serialized(case.spec)
    case.spec_path.write_bytes(payload)
    monkeypatch.setattr(corpus, "SPECIFICATION_SHA256", corpus.checksum(payload))


def source_row(sid, row, path):
    values = {
        "source_id": int(sid),
        "source_ordinal": 1,
        "gaia_designation": f"Gaia DR3 {sid}",
        "kic": row["product_kic"],
        "origin_tpf_kic": row["metadata"]["origin_tpf_kic"],
        "ra_deg": 12.3,
        "dec_deg": -4.5,
        "phot_g_mean_mag": 13.0,
        "file_stem": f"original-{sid}",
        "product_url": row["url"],
        "advertised_psf_avail": "True",
        "advertised_sap_avail": "True",
        "raw_fits_bytes": row["bytes"],
        "raw_fits_sha256": row["sha256"],
        "stitched_rows": 200,
        "quarter_count": 1,
        "primary_header_cards": fits.getheader(path).tostring(),
        **{
            name: 200
            for name in (
                "finite_psf_stitched",
                "finite_sap_stitched",
                "finite_psf_flat_kbonus",
                "quality_zero_rows",
                "flatten_mask_zero_rows",
            )
        },
    }
    return pa.Table.from_pylist([values], schema=corpus.SOURCE_SCHEMA)


def restrict_view(path, sid, row, view, count, delta):
    column = {"psf_qdiag": "FLUX_ERR", "psf_nova": "PSF_FLUX_ERR_NOVA"}[view]
    with fits.open(path, mode="update") as hdus:
        hdus["LIGHTCURVE_Q1"].data[column][count:] = 0.0
    row["sha256"] = opened.digest(path)
    row["metadata"]["raw_fits_sha256"] = row["sha256"]
    raw = opened.public.fits_arrays(path, sid, row)
    row["base_arrays"] = inventory(raw)
    with np.load(io.BytesIO(delta), allow_pickle=False) as values:
        for name in values.files:
            raw[name] = np.asarray(raw[name], dtype=np.float64) + values[name]
    row["arrays"] = inventory(raw)


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    monkeypatch.setattr(opened, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(opened, "EXTERNAL_DUMPS", ())
    monkeypatch.setattr(corpus, "COHORTS", dict.fromkeys(corpus.COHORTS, 1))
    monkeypatch.setattr(
        corpus.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100 * 1024**3, used=0, total=200 * 1024**3),
    )
    monkeypatch.delenv("SPACELAB_EXECUTION_PROFILE", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "deltas").mkdir()
    data = {
        "supported_order": IDS.copy(),
        "sources": {},
        "budgets": {
            "max_cloud_storage_bytes": 32_000_000_000,
            "paired_run_headroom_bytes": 4 * 1024**3,
        },
    }
    members, sources, lineage = {}, {}, {}
    schema_map = {}
    for name, schema in (
        ("cadence", corpus.CADENCE_SCHEMA),
        ("qdiag", corpus.QDIAG_SCHEMA),
    ):
        payload = schema.serialize().to_pybytes()
        member = f"schema/{corpus.checksum(payload)}.arrow"
        members[member] = payload
        schema_map[name] = member
    for sid, cohort in zip(IDS, corpus.COHORTS):
        row, payload, deltas = make_source(
            tmp_path, sid, "periodic_planet", 100 + IDS.index(sid), 200 + IDS.index(sid)
        )
        row["opened_cohort"] = cohort
        row["metadata"]["shard_path"] = "/original/unavailable/shard-0000"
        path = cache / f"{sid}.fits"
        path.write_bytes(payload)
        delta = deltas[f"{sid}.npz"]
        if hasattr(request, "param"):
            restrict_view(path, sid, row, *request.param, delta)
        (cache / "deltas" / f"{sid}.npz").write_bytes(delta)
        raw = opened.reconstruct(path, sid, row, cache / "deltas")
        np.savez_compressed(cache / f"{sid}.npz", **raw)
        data["sources"][sid] = row
        metadata = source_row(sid, row, path)
        member = f"source/{sid}.parquet"
        members[member] = bytes_table(metadata)
        sources[sid] = {
            "opened_cohort": cohort,
            "volume": "discSSD",
            "shard": 0,
            "metadata": {"member": member, **descriptor(members[member])},
            "schemas": schema_map.copy(),
            "original_source_table": {
                "path": "/never/read/source.parquet",
                **descriptor(members[member]),
            },
        }
        manifest = {
            "schema": 2,
            "measurement_scope": "OPENED_DEVELOPMENT_CONTROLS",
            "transform": "additive_flux_delta",
            "truth": "truth.jsonl",
            "sources": {
                sid: {
                    "source_id": sid,
                    "metadata": row["metadata"].copy(),
                    "base_raw_fits_sha256": row["sha256"],
                    "deltas": "/original/delta.npz",
                    "sha256": row["delta"]["sha256"],
                    "truth_index": 0,
                    "pixels": {},
                }
            },
        }
        values = {
            "input-manifest.json": opened.serialized(manifest),
            "source-ids.json": opened.serialized([sid]),
            "truth.jsonl": b'{"development_fixture":true}\n',
        }
        materialization = {
            "schema": 1,
            "status": "FROZEN",
            "partition": "development",
            "source_ids": [sid],
            "sources": 1,
            "artifacts": {
                name: corpus.checksum(payload) for name, payload in values.items()
            },
        }
        materialization["artifacts"]["gaia-astrometry.json"] = "a" * 64
        values["materialization.json"] = opened.serialized(materialization)
        lineage[cohort] = {}
        for name, payload in values.items():
            member = f"lineage/{cohort}/{name}"
            members[member] = payload
            lineage[cohort][name] = {
                "member": member,
                **descriptor(payload),
                "original": {
                    "path": f"/never/read/{cohort}/{name}",
                    **descriptor(payload),
                },
            }
        row["origin"]["original_input_manifest"] = {
            "path": "/never/read/input-manifest.json",
            "sha256": corpus.checksum(values["input-manifest.json"]),
        }
    spec = {
        "schema": 1,
        "status": "PORTABLE_ORIGINAL_OPENED_SOURCE_METADATA_ONLY",
        "scope": opened.public.SCOPE,
        "role": "development",
        "source_order": IDS.copy(),
        "sources": sources,
        "schemas": {name: descriptor(members[name]) for name in schema_map.values()},
        "original_lineage": lineage,
        "files": {},
        "archive": {},
        "uncompressed_member_bytes": 0,
        "source_rows": 3,
        "source_columns": 22,
        "original_cadence_or_qdiag_payload_rows_read": 0,
        "original_input_arrays_or_truths_changed": False,
        "full180_reconstruction_verified": False,
        "dependencies": {"/unavailable/historical/file": "b" * 64},
        "export_program": {"path": "/never/execute/export.py"},
        **{flag: False for flag in corpus.FLAGS},
    }
    selected, _ = opened.computational_selection(data, list(corpus.COHORTS))
    manifest = {
        "sources": {sid: {} for sid in IDS},
        "computational_selection": selected,
    }
    dependencies = {
        path: opened.digest(path) for path in cache.rglob("*") if path.is_file()
    }
    implementation = Path(corpus.runner.__file__)
    dependencies[implementation] = opened.digest(implementation)

    def inputs(**kwargs):
        opened.recheck(dependencies)
        return (
            opened.parse(opened.serialized(manifest)),
            data["sources"],
            dependencies.copy(),
        )

    monkeypatch.setattr(opened, "verified_inputs", inputs)
    monkeypatch.setattr(
        opened,
        "packet",
        lambda: (data, None, {implementation: dependencies[implementation]}),
    )
    case = SimpleNamespace(
        cache=cache,
        destination=tmp_path / "corpus",
        data=data,
        members=members,
        spec=spec,
        manifest=manifest,
        archive=tmp_path / "metadata.tar.gz",
        spec_path=tmp_path / "metadata.json",
        dependencies=dependencies,
    )
    pack(case, monkeypatch)
    return case


def run(case, **kwargs):
    return corpus.provision(
        specification=case.spec_path,
        cache=case.cache,
        destination=case.destination,
        **kwargs,
    )


def verify(case):
    return corpus.verified_corpus(
        specification=case.spec_path, cache=case.cache, destination=case.destination
    )


def assert_withdrawn(case):
    assert not (case.destination / "receipt.json").exists()
    assert not (case.destination / "source-ids.json").exists()


def test_original_reader_exact29_arrays_four_views_metadata_and_order(
    case, monkeypatch
):
    original_loader = corpus.lc2.load_source

    def local_only(root, sid, manifest=None):
        assert Path(root) == case.destination
        return original_loader(root, sid, manifest)

    monkeypatch.setattr(corpus.lc2, "load_source", local_only)
    receipt = run(case)
    assert receipt["status"] == "OPENED_ORIGINAL180_CORPUS_COMPLETE"
    assert receipt["selection"]["selected_source_ids"] == IDS
    assert all(not receipt[flag] for flag in corpus.FLAGS)
    assert receipt["full_frozen_materialization_created"] is False
    assert all(
        len(source["preparation_views"]) == 4 for source in receipt["sources"].values()
    )
    assert set(
        receipt["unprovisioned_original_materialization_artifacts"]["periodic_planet"]
    ) == {"gaia-astrometry.json"}
    parquet = pq.ParquetFile(
        case.destination / "discSSD/data/kbonus/lc2/shard-0000/qdiag.parquet"
    )
    assert parquet.num_row_groups == 3
    assert parquet.schema_arrow.field("source_id").type == pa.int64()
    assert parquet.read()["source_id"].to_pylist() == [
        int(sid) for sid in IDS for _ in range(200)
    ]
    assert verify(case)[0] == receipt
    for name, payload in case.members.items():
        assert (case.destination / "metadata" / name).read_bytes() == payload


@pytest.mark.parametrize(
    "case", [("psf_nova", 0), ("psf_qdiag", 99), ("psf_nova", 100)], indirect=True
)
def test_original_unavailable_views_are_preserved_without_fsc_hashes(case, request):
    view, count = request.node.callspec.params["case"]
    receipt = run(case)
    for sid, source in receipt["sources"].items():
        item = source["preparation_views"][view]
        assert item["retained_cadences"] == count
        assert item["status"] == ("UNAVAILABLE" if count < 100 else "PREPARED")
        if count < 100:
            assert set(item) == {
                "status",
                "reason",
                "raw_view_sha256",
                "retained_cadences",
            }
            assert item["reason"] == "InsufficientCadences"
            with np.load(case.cache / f"{sid}.npz", allow_pickle=False) as saved:
                raw = {name: saved[name] for name in saved.files}
            with pytest.raises(corpus.runner.InsufficientCadences) as error:
                corpus.runner.prepare_records(raw, view)
            assert error.value.count == count
        else:
            assert set(item) == {
                "status",
                "raw_view_sha256",
                "retained_cadences",
                "prepared_records_sha256",
                "fsc_sha256",
            }
        assert source["preparation_views"]["psf_flat"]["status"] == "PREPARED"
    assert verify(case)[0] == receipt


@pytest.mark.parametrize("side", ["raw", "baseline"])
@pytest.mark.parametrize("change", ["availability", "count", "error"])
def test_one_sided_preparation_availability_count_or_error_is_rejected(
    case, monkeypatch, side, change
):
    with np.load(case.cache / f"{IDS[0]}.npz", allow_pickle=False) as saved:
        baseline = {name: saved[name] for name in saved.files}
    raw = {name: value.copy() for name, value in baseline.items()}
    original = corpus.runner.prepare_records
    selected = raw if side == "raw" else baseline

    def prepare(value, view):
        if view == "psf_nova":
            if change == "error" and value is selected:
                raise KeyError("original failed read")
            if change == "count":
                raise corpus.runner.InsufficientCadences(1 if value is selected else 0)
            if value is selected:
                raise corpus.runner.InsufficientCadences(0)
        return original(value, view)

    monkeypatch.setattr(corpus.runner, "prepare_records", prepare)
    with pytest.raises(KeyError if change == "error" else ValueError):
        corpus.view_oracles(raw, baseline, case.data["sources"][IDS[0]])


def test_unavailable_discovery_view_never_counts_as_original_transport(
    case, monkeypatch
):
    original = corpus.runner.prepare_records

    def prepare(raw, view):
        if view == "psf_flat":
            raise corpus.runner.InsufficientCadences(0)
        return original(raw, view)

    monkeypatch.setattr(corpus.runner, "prepare_records", prepare)
    with pytest.raises(ValueError, match="discovery view"):
        run(case)
    assert_withdrawn(case)


@pytest.mark.parametrize("count", [True, -1, 100])
def test_impossible_unavailable_count_is_rejected(case, monkeypatch, count):
    def prepare(raw, view):
        raise corpus.runner.InsufficientCadences(count)

    monkeypatch.setattr(corpus.runner, "prepare_records", prepare)
    with pytest.raises(ValueError, match="count is invalid"):
        run(case)
    assert_withdrawn(case)


@pytest.mark.parametrize("side", [0, 1])
def test_consumer_one_sided_availability_drift_withdraws_old_success(
    case, monkeypatch, side
):
    run(case)
    original = corpus.runner.prepare_records
    calls = 0

    def prepare(raw, view):
        nonlocal calls
        if view == "psf_nova":
            calls += 1
            if calls - 1 == side:
                raise corpus.runner.InsufficientCadences(0)
        return original(raw, view)

    monkeypatch.setattr(corpus.runner, "prepare_records", prepare)
    with pytest.raises(ValueError, match="availability"):
        verify(case)
    assert_withdrawn(case)


@pytest.mark.parametrize("case", [("psf_nova", 0)], indirect=True)
@pytest.mark.parametrize(
    "change",
    [
        "count_bool",
        "count_negative",
        "count_eligible",
        "reason",
        "status",
        "fsc_hash",
        "records_hash",
        "raw_hash",
        "lost_view",
        "discovery",
    ],
)
def test_forged_unavailable_evidence_withdraws_success(case, change):
    receipt = run(case)
    views = receipt["sources"][IDS[0]]["preparation_views"]
    item = views["psf_nova"]
    if change == "count_bool":
        item["retained_cadences"] = False
    elif change == "count_negative":
        item["retained_cadences"] = -1
    elif change == "count_eligible":
        item["retained_cadences"] = 100
    elif change == "reason":
        item["reason"] = "KeyError"
    elif change == "status":
        item["status"] = "PREPARED"
    elif change == "fsc_hash":
        item["fsc_sha256"] = "a" * 64
    elif change == "records_hash":
        item["prepared_records_sha256"] = "a" * 64
    elif change == "raw_hash":
        item["raw_view_sha256"] = "bad"
    elif change == "lost_view":
        views.pop("psf_nova")
    else:
        views["psf_flat"] = item.copy()
    path = case.destination / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))
    with pytest.raises(ValueError):
        verify(case)
    assert_withdrawn(case)


def test_old_view_receipt_cannot_hide_missing_explicit_availability(case):
    receipt = run(case)
    for source in receipt["sources"].values():
        for item in source["preparation_views"].values():
            item.pop("status")
    path = case.destination / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))
    with pytest.raises((ValueError, KeyError)):
        verify(case)
    assert_withdrawn(case)


def test_prefix_retains_full_original_manifests_and_all_delta_bytes(case):
    case.manifest["sources"] = {IDS[0]: {}}
    case.manifest["computational_selection"], _ = opened.computational_selection(
        case.data, list(corpus.COHORTS), 1
    )
    receipt = run(case)
    assert receipt["status"] == "OPENED_ORIGINAL180_CORPUS_PARTIAL"
    assert receipt["selection"]["full180_provisioning_complete"] is False
    assert receipt["original_source_order"] == IDS
    assert list(receipt["sources"]) == IDS[:1]
    assert len(list((case.destination / "deltas").iterdir())) == 3
    for cohort, sid in zip(corpus.COHORTS, IDS):
        manifest = json.loads(Path(receipt["manifests"][cohort]).read_bytes())
        assert list(manifest["sources"]) == [sid]
        assert manifest["sources"][sid]["pixels"] == {}
    assert verify(case)[0] == receipt


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "extra", "traversal", "symlink"]
)
def test_archive_closed_regular_members(case, monkeypatch, change):
    members = [
        (name, payload, tarfile.REGTYPE) for name, payload in case.members.items()
    ]
    if change == "duplicate":
        members.append(members[0])
    elif change == "missing":
        members.pop()
    elif change in {"extra", "traversal"}:
        members.append(
            ("unknown" if change == "extra" else "../outside", b"no", tarfile.REGTYPE)
        )
    else:
        members[0] = (members[0][0], b"", tarfile.SYMTYPE)
    pack(case, monkeypatch, members)
    with pytest.raises(ValueError):
        run(case)
    assert_withdrawn(case)
    assert not (case.destination.parent / "outside").exists()


def test_bounded_gzip_decode_rejects_padding_bomb(case, monkeypatch):
    with gzip.open(case.archive, "rb") as stream:
        raw = stream.read()
    zipped = gzip.compress(raw + b"\0" * (corpus.MAX_DECODED + 1024**2), mtime=0)
    case.archive.write_bytes(zipped)
    case.spec["archive"].update(descriptor(zipped))
    payload = opened.serialized(case.spec)
    case.spec_path.write_bytes(payload)
    monkeypatch.setattr(corpus, "SPECIFICATION_SHA256", corpus.checksum(payload))
    with pytest.raises(ValueError, match="decoded stream"):
        run(case)
    assert_withdrawn(case)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "schema",
        "mixed_rows",
        "order",
        "volume",
        "lineage_delta",
        "frozen_hash",
    ],
)
def test_rebound_metadata_cannot_change_original_contract(case, monkeypatch, change):
    sid = IDS[0]
    member = case.spec["sources"][sid]["metadata"]["member"]
    table = pq.read_table(pa.BufferReader(case.members[member]))
    if change == "source":
        table = table.set_column(
            0, corpus.SOURCE_SCHEMA.field(0), pa.array([int(IDS[1])], type=pa.int64())
        )
    elif change == "schema":
        table = table.drop(["source_ordinal"])
    elif change == "mixed_rows":
        table = pa.concat_tables([table, table])
    elif change == "order":
        case.spec["source_order"].reverse()
    elif change == "volume":
        case.spec["sources"][sid]["volume"] = "discPCIe"
    else:
        name = (
            "input-manifest.json"
            if change == "lineage_delta"
            else "materialization.json"
        )
        name = f"lineage/periodic_planet/{name}"
        value = json.loads(case.members[name])
        if change == "lineage_delta":
            value["sources"][sid]["sha256"] = "f" * 64
        else:
            value["artifacts"]["truth.jsonl"] = "f" * 64
        case.members[name] = opened.serialized(value)
    if change in {"source", "schema", "mixed_rows"}:
        case.members[member] = bytes_table(table)
    pack(case, monkeypatch)
    with pytest.raises((ValueError, KeyError)):
        run(case)
    assert_withdrawn(case)


def test_absent_base_oracle_still_requires_full29_oracle(case):
    case.data["sources"][IDS[1]]["base_arrays"] = None
    case.data["sources"][IDS[1]]["base_checks"] = None
    assert run(case)["sources"][IDS[1]]["base_array_oracle_available"] is False


@pytest.mark.parametrize(
    "target",
    [
        "receipt",
        "index",
        "metadata",
        "delta",
        "manifest",
        "parquet",
        "unknown",
        "partial",
    ],
)
def test_failure_withdraws_previous_success(case, target):
    receipt = run(case)
    if target == "receipt":
        receipt["selection"]["full180_provisioning_complete"] = False
        (case.destination / "receipt.json").chmod(0o644)
        (case.destination / "receipt.json").write_bytes(opened.serialized(receipt))
    elif target == "unknown" or target == "partial":
        (
            case.destination / ("extra" if target == "unknown" else "receipt.json.part")
        ).write_bytes(b"x")
    else:
        names = {
            "index": "discSSD/data/index.sqlite",
            "metadata": f"metadata/source/{IDS[0]}.parquet",
            "delta": f"deltas/{IDS[0]}.npz",
            "manifest": "transport/periodic_planet/input-manifest.json",
            "parquet": "discSSD/data/kbonus/lc2/shard-0000/cadence.parquet",
        }
        path = case.destination / names[target]
        path.chmod(0o644)
        path.write_bytes(b"changed")
    with pytest.raises((ValueError, KeyError)):
        verify(case)
    assert_withdrawn(case)


def rebind_output(case, receipt, path):
    name = path.relative_to(case.destination).as_posix()
    receipt["outputs"][name] = corpus.identity(path)
    receipt["dependencies"][str(path)] = opened.digest(path)
    output = case.destination / "receipt.json"
    output.chmod(0o644)
    output.write_bytes(opened.serialized(receipt))


@pytest.mark.parametrize(
    "change",
    [
        "flux",
        "source_groups",
        "extra_source",
        "index",
        "index_schema",
        "path_only",
        "role",
        "lost_view",
    ],
)
def test_rehashed_outputs_do_not_replace_scientific_or_accounting_oracles(case, change):
    receipt = run(case)
    folder = case.destination / "discSSD/data/kbonus/lc2/shard-0000"
    if change in {"flux", "source_groups", "extra_source"}:
        path = folder / "cadence.parquet"
        table = pq.read_table(path)
        if change == "flux":
            index = table.schema.get_field_index("psf_flat_kbonus_flux")
            table = table.set_column(
                index, table.schema.field(index), pa.array(np.full(len(table), 998.0))
            )
        elif change == "source_groups":
            table = table.take(pa.array(np.roll(np.arange(len(table)), 1)))
        else:
            table = pa.concat_tables([table, table.slice(0, 1)])
        path.chmod(0o644)
        pq.write_table(table, path, row_group_size=200)
    elif change in {"index", "index_schema"}:
        path = case.destination / "discSSD/data/index.sqlite"
        with sqlite3.connect(path) as db:
            if change == "index":
                db.execute(
                    "UPDATE sources SET shard=2 WHERE gaia_source_id=?", (IDS[0],)
                )
            else:
                db.execute("ALTER TABLE sources ADD COLUMN hidden TEXT")
    elif change == "path_only":
        path = Path(receipt["manifests"]["periodic_planet"])
        value = json.loads(path.read_bytes())
        value["sources"][IDS[0]]["pixels"] = {"substituted": True}
        path.chmod(0o644)
        path.write_bytes(opened.serialized(value))
    elif change == "role":
        receipt["role"] = "evaluation"
        path = case.destination / "receipt.json"
    else:
        receipt["sources"][IDS[0]]["preparation_views"].pop("psf_nova")
        path = case.destination / "receipt.json"
    if path.name == "receipt.json":
        path.chmod(0o644)
        path.write_bytes(opened.serialized(receipt))
    else:
        rebind_output(case, receipt, path)
    with pytest.raises((ValueError, KeyError)):
        verify(case)
    assert_withdrawn(case)


@pytest.mark.parametrize(
    "stage", ["verify", "finish", "publication", "published_owner"]
)
def test_late_dependency_or_output_drift_never_leaves_success(case, monkeypatch, stage):
    if stage == "verify":
        original = corpus.verify_sources

        def altered(*args, **kwargs):
            result = original(*args, **kwargs)
            path = (
                case.destination / "discSSD/data/kbonus/lc2/shard-0000/cadence.parquet"
            )
            path.write_bytes(path.read_bytes() + b"x")
            return result

        monkeypatch.setattr(corpus, "verify_sources", altered)
    elif stage == "finish":
        original = corpus.finish

        def altered(*args, **kwargs):
            (case.cache / f"{IDS[0]}.fits").write_bytes(b"changed")
            return original(*args, **kwargs)

        monkeypatch.setattr(corpus, "finish", altered)
    else:
        original = opened.publish

        def altered(root, path, payload):
            result = original(root, path, payload)
            if path.name == "receipt.json":
                target = root / (
                    corpus.OWNER if stage == "published_owner" else "source-ids.json"
                )
                target.chmod(0o644)
                target.write_bytes(target.read_bytes() + b"\n")
            return result

        monkeypatch.setattr(opened, "publish", altered)
    with pytest.raises(ValueError):
        run(case)
    assert_withdrawn(case)


def test_existing_corpus_is_not_overwritten_and_success_is_withdrawn(case):
    run(case)
    path = case.destination / "discSSD/data/index.sqlite"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="fresh root"):
        run(case)
    assert path.read_bytes() == before
    assert_withdrawn(case)


@pytest.mark.parametrize("consumer", [False, True])
def test_unrelated_occupied_root_preserves_original_receipts_and_source_ids(
    case, consumer
):
    case.destination.mkdir()
    files = {
        "receipt.json": b"original unrelated receipt",
        "source-ids.json": b'["original_source"]',
        "keep": b"original data",
    }
    for name, payload in files.items():
        (case.destination / name).write_bytes(payload)
    with pytest.raises(ValueError):
        verify(case) if consumer else run(case)
    assert {name: (case.destination / name).read_bytes() for name in files} == files


@pytest.mark.parametrize(
    "change",
    [
        "boundary",
        "fields",
        "numeric_type",
        "growth",
        "view_bytes",
        "headroom",
        "free",
        "used",
        "reserve",
        "disk_growth",
        "space_flag",
        "cloud_flag",
        "history",
    ],
)
def test_receipt_resource_and_boundary_claims_are_checked(case, change):
    receipt = run(case)
    resources = receipt["resources"]
    changes = {
        "numeric_type": ("forecast_growth_bytes", True),
        "growth": ("forecast_growth_bytes", resources["forecast_growth_bytes"] + 1),
        "view_bytes": ("next_preparation_all_views_bytes", 0),
        "headroom": ("paired_run_headroom_bytes", 0),
        "free": ("available_bytes_before", resources["reserve_bytes"]),
        "used": ("current_filesystem_used_bytes", -1),
        "reserve": ("reserve_bytes", corpus.CLOUD_RESERVE),
        "disk_growth": ("disk_growth_bytes", -1),
        "space_flag": ("preparation_and_paired_run_space_reserved_here", True),
        "cloud_flag": ("cloud_budget_applied", True),
        "history": ("filesystem_measurements", "CURRENT_VERIFIED"),
    }
    if change == "boundary":
        receipt["boundary"] = "Scientific calibration complete"
    elif change == "fields":
        resources["new"] = 0
    else:
        key, value = changes[change]
        resources[key] = value
    path = case.destination / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))
    with pytest.raises(ValueError):
        verify(case)
    assert_withdrawn(case)


def test_cloud_bootstrap_accepts_only_exact_empty_managed_volume_links(
    case, monkeypatch
):
    root = case.destination.parent / "repo"
    destination = root / "dump/cloud-volumes"
    for name in ("discSSD", "discPCIe"):
        (destination / name).mkdir(parents=True)
        (root / name).symlink_to(destination / name)
    monkeypatch.setattr(corpus, "ROOT", root)
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    case.destination = destination
    assert (
        run(case, minimum_free_bytes=corpus.CLOUD_RESERVE)["selection"][
            "full180_provisioning_complete"
        ]
        is True
    )
    assert (root / "discSSD").is_symlink()
    assert (root / "discPCIe").is_symlink()


@pytest.mark.parametrize("bad", ["symlink", "unknown", "occupied_volume", "wrong_link"])
def test_destination_refuses_redirects_unknown_or_used_bootstrap(
    case, monkeypatch, bad
):
    outside = case.destination.parent / "outside"
    outside.mkdir()
    sentinel = outside / "retain"
    sentinel.write_bytes(b"unchanged")
    if bad == "symlink":
        case.destination.symlink_to(outside)
    elif bad == "unknown":
        case.destination.mkdir()
        (case.destination / "unknown").write_bytes(b"x")
    else:
        root = case.destination.parent / "repo"
        case.destination = root / "dump/cloud-volumes"
        for name in ("discSSD", "discPCIe"):
            (case.destination / name).mkdir(parents=True)
            (root / name).symlink_to(
                case.destination / name if bad != "wrong_link" else outside
            )
        if bad == "occupied_volume":
            (case.destination / "discSSD/data").mkdir()
        monkeypatch.setattr(corpus, "ROOT", root)
        monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    with pytest.raises(ValueError):
        run(case)
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("value", [0, -1, True, 512 * 1024**2, corpus.CLOUD_RESERVE])
def test_local_reserve_requires_twenty_gib(case, value):
    with pytest.raises(ValueError, match="reserve"):
        run(case, minimum_free_bytes=value)
    assert_withdrawn(case)


@pytest.mark.parametrize("edge", ["free", "cloud_used", "during_write"])
def test_capacity_counts_current_use_and_source_growth(case, monkeypatch, edge):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    if edge == "during_write":
        original = corpus.extract_metadata

        def altered(*args):
            original(*args)
            monkeypatch.setattr(
                corpus.shutil,
                "disk_usage",
                lambda path: SimpleNamespace(
                    free=corpus.CLOUD_RESERVE, used=0, total=32_000_000_000
                ),
            )

        monkeypatch.setattr(corpus, "extract_metadata", altered)
    else:
        monkeypatch.setattr(
            corpus.shutil,
            "disk_usage",
            lambda path: SimpleNamespace(
                free=corpus.CLOUD_RESERVE if edge == "free" else 30_000_000_000,
                used=31_000_000_000 if edge == "cloud_used" else 0,
                total=32_000_000_000,
            ),
        )
    with pytest.raises(ValueError):
        run(case, minimum_free_bytes=corpus.CLOUD_RESERVE)
    assert_withdrawn(case)


def test_cloud_small_reserve_remains_forbidden_on_external_storage(case, monkeypatch):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(opened, "EXTERNAL_DUMPS", (case.destination.parent,))
    with pytest.raises(ValueError, match="reserve"):
        run(case, minimum_free_bytes=corpus.CLOUD_RESERVE)


def test_manifest_cache_subset_requires_original_order_nonempty(case):
    case.manifest["sources"] = {}
    with pytest.raises(ValueError, match="no injected sources"):
        run(case)
    case.manifest["sources"] = dict.fromkeys(reversed(IDS), {})
    with pytest.raises(ValueError, match="original order"):
        run(case)


def test_guard_failure_never_follows_output_symlink(case):
    receipt = run(case)
    path = Path(receipt["manifests"]["periodic_planet"])
    outside = case.destination.parent / "outside.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    before = outside.read_bytes()
    with pytest.raises(ValueError, match="symlink"):
        verify(case)
    assert outside.read_bytes() == before
    assert path.is_symlink()
    assert_withdrawn(case)


def test_handoff_drift_withdraws_receipt_before_returning_paths(case, monkeypatch):
    run(case)
    original = corpus.verify_sources

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        (case.cache / f"{IDS[0]}.npz").write_bytes(b"changed")
        return result

    monkeypatch.setattr(corpus, "verify_sources", altered)
    with pytest.raises(ValueError):
        verify(case)
    assert_withdrawn(case)


@pytest.mark.parametrize("consumer", [False, True])
def test_redirected_cache_validation_withdraws_old_success(case, consumer):
    run(case)
    link = case.cache.parent / "cache-link"
    link.symlink_to(case.cache)
    case.cache = link
    with pytest.raises(ValueError, match="symlink"):
        verify(case) if consumer else run(case)
    assert_withdrawn(case)


def test_actual_closed180_metadata_matches_live_packet_and_original_lineage(tmp_path):
    descriptor = opened.parse(opened.SPECIFICATION.read_bytes())
    metadata, _ = opened.decoded_metadata(opened.SPECIFICATION, descriptor)
    if any(opened.digest(opened.ROOT / item["path"]) != item["sha256"] for item in metadata["reader_programs"]):
        with pytest.raises(ValueError, match="wrong-sized opened input|Opened input bytes changed"):
            opened.packet()
        return
    data, _, dependencies = opened.packet()
    spec, archive, metadata_dependencies = corpus.metadata_packet(
        corpus.SPECIFICATION, data
    )
    dependencies.update(metadata_dependencies)
    assert len(spec["source_order"]) == 180
    assert len(spec["files"]) == 194
    assert spec["uncompressed_member_bytes"] == 6_715_422
    assert corpus.identity(archive) == {
        "bytes": 573075,
        "sha256": "8385f2484b2e9b148043792f499ea531a5dba2b9411bc3306cd2eb91c8fe81fd",
    }
    root = tmp_path / "actual-metadata"
    root.mkdir()
    minimum = corpus.reserve(
        root, corpus.CLOUD_RESERVE if corpus.cloud_destination(root) else None
    )
    assert (
        corpus.shutil.disk_usage(root).free
        >= minimum + spec["uncompressed_member_bytes"]
    )
    corpus.extract_metadata(root, archive, spec, minimum)
    corpus.schemas(root, spec)
    originals, missing = corpus.lineage(root, spec, data)
    assert {
        cohort: len(value["sources"]) for cohort, value in originals.items()
    } == corpus.COHORTS
    assert all("gaia-astrometry.json" in files for files in missing.values())
    for sid in spec["source_order"]:
        table = corpus.source_metadata(root, spec, sid, data["sources"][sid])
        assert len(table) == 1
    opened.recheck(dependencies)


def test_conflicting_input_snapshots_cannot_be_rebound(case, monkeypatch):
    implementation = Path(corpus.runner.__file__)
    monkeypatch.setattr(
        opened, "packet", lambda: (case.data, None, {implementation: "f" * 64})
    )
    with pytest.raises(ValueError, match="between dependency snapshots"):
        run(case)
    assert_withdrawn(case)


def test_output_identity_descriptor_fields_are_closed(case):
    receipt = run(case)
    receipt["outputs"][corpus.OWNER]["unverified_extra"] = True
    path = case.destination / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))
    with pytest.raises(ValueError, match="file identity"):
        verify(case)
    assert_withdrawn(case)


@pytest.mark.parametrize(
    "change", ["completion", "selection_schema", "verified_flag", "source_order"]
)
def test_handoff_typed_selection_and_scientific_proofs_cannot_alias_numbers(
    case, change
):
    receipt = run(case)
    if change == "completion":
        receipt["selection"]["full180_provisioning_complete"] = 1
    elif change == "selection_schema":
        receipt["selection"]["cache_computational_selection"]["schema"] = True
    elif change == "verified_flag":
        receipt["sources"][IDS[0]]["full29_array_inventory_verified"] = 1
    else:
        receipt["sources"] = dict(reversed(list(receipt["sources"].items())))
    path = case.destination / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))
    with pytest.raises(ValueError):
        verify(case)
    assert_withdrawn(case)
