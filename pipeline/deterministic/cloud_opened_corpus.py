"""Build an isolated original LC2 corpus from verified opened public inputs."""

import gzip
import hashlib
import io
import os
import shutil
import sqlite3
import struct
import sys
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from deterministic import cloud_opened as opened
from deterministic import run as runner
from photometry import lc2

SPECIFICATION = Path(__file__).with_name("cloud-opened-source-metadata.json")
SPECIFICATION_SHA256 = (
    "8633603564573d5ffa96932b59e296be8e6f2a76b3b452b191bc6324b7808f9c"
)
COHORTS = {"periodic_planet": 100, "sparse_injection": 40, "ttv_injection": 40}
RESERVE = 20 * 1024**3
CLOUD_RESERVE = 2 * 1024**3
MAX_ARCHIVE = 8 * 1024**2
MAX_DECODED = 16 * 1024**2
SUCCESS = ("receipt.json", "source-ids.json")
OWNER = "corpus-owner.json"
BOUNDARY = "Exact selected opened LC2 transport. Preserved original lineage is inert; no active FROZEN materialization, pixel readiness, context closure or scientific acceptance."
FLAGS = (
    "scientific_acceptance",
    "automatic_pc_enabled",
    "joint_independence_complete",
    "pixel_readiness_complete",
)
SOURCE_SCHEMA = pa.schema(
    [
        ("source_id", pa.int64()),
        ("source_ordinal", pa.int64()),
        ("gaia_designation", pa.string()),
        ("kic", pa.int64()),
        ("origin_tpf_kic", pa.int64()),
        ("ra_deg", pa.float64()),
        ("dec_deg", pa.float64()),
        ("phot_g_mean_mag", pa.float64()),
        ("file_stem", pa.string()),
        ("product_url", pa.string()),
        ("advertised_psf_avail", pa.string()),
        ("advertised_sap_avail", pa.string()),
        ("raw_fits_bytes", pa.int64()),
        ("raw_fits_sha256", pa.string()),
        ("stitched_rows", pa.int64()),
        ("quarter_count", pa.int16()),
        ("finite_psf_stitched", pa.int64()),
        ("finite_sap_stitched", pa.int64()),
        ("finite_psf_flat_kbonus", pa.int64()),
        ("quality_zero_rows", pa.int64()),
        ("flatten_mask_zero_rows", pa.int64()),
        ("primary_header_cards", pa.string()),
    ]
)
CADENCE_SCHEMA = pa.schema(
    [
        ("source_id", pa.int64()),
        ("time_bkjd", pa.float64()),
        ("cadenceno", pa.int32()),
        ("quarter", pa.int16()),
        ("psf_stitched_flux", pa.float64()),
        ("psf_stitched_flux_err", pa.float64()),
        ("sap_stitched_flux", pa.float64()),
        ("sap_stitched_flux_err", pa.float64()),
        ("psf_flat_kbonus_flux", pa.float64()),
        ("psf_flat_kbonus_flux_err", pa.float64()),
        ("sap_quality", pa.int32()),
        ("flatten_mask", pa.int32()),
    ]
)
QDIAG_SCHEMA = pa.schema(
    [
        ("source_id", pa.int64()),
        ("quarter", pa.int16()),
        ("module", pa.int16()),
        ("output", pa.int16()),
        ("channel", pa.int16()),
        ("cadenceno", pa.int32()),
        ("time_bkjd", pa.float64()),
        ("psf_flux", pa.float64()),
        ("psf_flux_err", pa.float64()),
        ("sap_flux", pa.float64()),
        ("sap_flux_err", pa.float64()),
        ("centroid_column", pa.float64()),
        ("centroid_row", pa.float64()),
        ("sap_quality", pa.int32()),
        ("sap_background", pa.float64()),
        ("psf_reduced_chi2", pa.float64()),
        ("psf_flux_nova", pa.float64()),
        ("psf_flux_err_nova", pa.float64()),
    ]
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(payload):
    return hashlib.sha256(payload).hexdigest()


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": opened.digest(path)}


def bind_dependencies(dependencies, current):
    require(
        all(
            path not in dependencies or dependencies[path] == value
            for path, value in current.items()
        ),
        "Corpus input identities changed between dependency snapshots",
    )
    dependencies.update(current)


def owner_payload(root):
    return opened.serialized(
        {
            "schema": 1,
            "kind": "OPENED_ORIGINAL180_CORPUS_MANAGED",
            "corpus_root": str(root),
        }
    )


def owned(root):
    if root.is_symlink() or not root.is_dir():
        return False
    path = root / OWNER
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes() == owner_payload(root)


def ownership_token(root):
    if not owned(root):
        return None
    stat = root.stat()
    return stat.st_dev, stat.st_ino


def withdraw(root, token=None):
    if root.is_symlink() or not root.is_dir():
        return
    stat = root.stat()
    if token != (stat.st_dev, stat.st_ino) and not owned(root):
        return
    for name in SUCCESS:
        path = root / name
        if path.is_file() and not path.is_symlink():
            path.unlink()


def cloud_destination(root):
    external = any(
        root.resolve().is_relative_to(p.resolve()) for p in opened.EXTERNAL_DUMPS
    )
    return (
        os.getenv("SPACELAB_EXECUTION_PROFILE") == "cloud-development" and not external
    )


def reserve(root, requested):
    value = RESERVE if requested is None else requested
    require(
        type(value) is int and value > 0, "Corpus reserve must be a positive integer"
    )
    require(
        value >= RESERVE or value == CLOUD_RESERVE and cloud_destination(root),
        "Smaller corpus reserve requires explicit cloud development outside external storage",
    )
    return value


def bootstrap_root(root):
    require(
        cloud_destination(root)
        and root == opened.destination_path(ROOT / "dump/cloud-volumes"),
        "Only the declared cloud bootstrap may supply pre-existing empty volumes",
    )
    for name in ("discSSD", "discPCIe"):
        link, folder = ROOT / name, root / name
        require(
            link.is_symlink()
            and link.resolve() == folder.resolve()
            and folder.is_dir()
            and not folder.is_symlink()
            and not any(folder.iterdir()),
            "Cloud bootstrap volume links or empty directories changed",
        )


def begin(destination, cache):
    root = opened.destination_path(Path(destination))
    require(
        not root.is_relative_to(cache) and not cache.is_relative_to(root),
        "Corpus and verified cache directories must remain separate",
    )
    withdraw(root)
    if root.exists():
        require(root.is_dir(), "Corpus destination must be a managed directory")
        entries = list(root.iterdir())
        if entries:
            require(
                {p.name for p in entries} == {"discSSD", "discPCIe"},
                "Corpus construction requires a fresh root; prior evidence is retained",
            )
            bootstrap_root(root)
    root.mkdir(parents=True, exist_ok=True)
    opened.publish(root, root / OWNER, owner_payload(root))
    return root


def member_name(name):
    require(isinstance(name, str), "Metadata member path must be a string")
    path = PurePosixPath(name)
    require(
        name
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and str(path) == name,
        "Metadata member path escapes its closed inventory",
    )
    return name


def file_descriptor(item, limit=MAX_DECODED):
    require(
        set(item) == {"bytes", "sha256"}
        and type(item["bytes"]) is int
        and 0 < item["bytes"] <= limit
        and opened.public.sha256(item["sha256"]),
        "Metadata file identity or size is malformed",
    )


def check_sources(spec, data):
    ids = [
        sid
        for sid in data["supported_order"]
        if data["sources"][sid]["opened_cohort"] in COHORTS
    ]
    require(
        spec["source_order"] == ids
        and list(spec["sources"]) == ids
        and len(ids) == sum(COHORTS.values()),
        "Original injected-source order or membership changed",
    )
    require(
        {
            name: sum(data["sources"][sid]["opened_cohort"] == name for sid in ids)
            for name in COHORTS
        }
        == COHORTS,
        "Original injected cohort counts changed",
    )
    for sid, item in spec["sources"].items():
        row = data["sources"][sid]
        require(
            set(item)
            == {
                "opened_cohort",
                "volume",
                "shard",
                "metadata",
                "schemas",
                "original_source_table",
            }
            and set(item["metadata"]) == {"member", "bytes", "sha256"},
            "Original source metadata descriptor fields changed",
        )
        require(
            item["opened_cohort"] == row["opened_cohort"]
            and row["reader"] == "lc2"
            and row["role"] == "development"
            and "delta" in row
            and type(item["shard"]) is int
            and item["shard"] == row["metadata"]["shard"]
            and item["volume"] == ("discSSD" if item["shard"] % 2 == 0 else "discPCIe"),
            "Original metadata source, role or location changed",
        )
        require(
            item["metadata"]["member"] == f"source/{sid}.parquet"
            and {k: item["metadata"][k] for k in ("bytes", "sha256")}
            == spec["files"][f"source/{sid}.parquet"]
            and set(item["schemas"]) == {"cadence", "qdiag"}
            and set(item["schemas"].values()) <= spec["schemas"].keys(),
            "Original source metadata/schema references changed",
        )


def check_inventory(spec):
    names = {f"source/{sid}.parquet" for sid in spec["source_order"]} | set(
        spec["schemas"]
    )
    require(
        set(spec["original_lineage"]) == set(COHORTS) and len(spec["schemas"]) == 2,
        "Original three-cohort lineage and two schemas are required",
    )
    for cohort, files in spec["original_lineage"].items():
        require(
            set(files)
            == {
                "input-manifest.json",
                "materialization.json",
                "source-ids.json",
                "truth.jsonl",
            },
            "Original lineage member set changed",
        )
        for name, item in files.items():
            member = f"lineage/{cohort}/{name}"
            require(
                item["member"] == member
                and item["sha256"] == item["original"]["sha256"]
                and item["bytes"] == item["original"]["bytes"],
                "Original lineage identity changed",
            )
            require(
                {k: item[k] for k in ("bytes", "sha256")} == spec["files"][member],
                "Lineage differs from archive inventory",
            )
            names.add(member)
    require(set(spec["files"]) == names, "Metadata archive inventory is not closed")
    for name, item in spec["files"].items():
        member_name(name)
        file_descriptor(item)
    for name, item in spec["schemas"].items():
        require(
            name == f"schema/{item['sha256']}.arrow" and item == spec["files"][name],
            "Metadata schema content name changed",
        )
    require(
        spec["uncompressed_member_bytes"]
        == sum(x["bytes"] for x in spec["files"].values())
        and spec["uncompressed_member_bytes"] <= MAX_DECODED,
        "Metadata decoded-byte bound changed",
    )


def metadata_packet(path, data):
    path = Path(path).absolute()
    opened.checked_file(
        path, {"bytes": path.stat().st_size, "sha256": SPECIFICATION_SHA256}
    )
    require(path.stat().st_size <= 512 * 1024, "Metadata descriptor exceeds its bound")
    payload = path.read_bytes()
    spec = opened.parse(payload)
    require(
        set(spec)
        == {
            "schema",
            "status",
            "scope",
            "role",
            "source_order",
            "sources",
            "schemas",
            "original_lineage",
            "files",
            "archive",
            "uncompressed_member_bytes",
            "source_rows",
            "source_columns",
            "original_cadence_or_qdiag_payload_rows_read",
            "original_input_arrays_or_truths_changed",
            "full180_reconstruction_verified",
            "dependencies",
            "export_program",
            *FLAGS,
        },
        "Metadata descriptor field inventory changed",
    )
    require(
        checksum(payload) == SPECIFICATION_SHA256
        and spec.get("schema") == 1
        and type(spec["schema"]) is int
        and spec.get("status") == "PORTABLE_ORIGINAL_OPENED_SOURCE_METADATA_ONLY"
        and spec.get("scope") == opened.public.SCOPE
        and spec.get("role") == "development"
        and type(spec.get("source_rows")) is int
        and spec.get("source_rows") == sum(COHORTS.values())
        and type(spec.get("source_columns")) is int
        and spec.get("source_columns") == 22
        and type(spec.get("original_cadence_or_qdiag_payload_rows_read")) is int
        and spec.get("original_cadence_or_qdiag_payload_rows_read") == 0
        and spec.get("original_input_arrays_or_truths_changed") is False
        and spec.get("full180_reconstruction_verified") is False
        and all(spec.get(flag) is False for flag in FLAGS),
        "Metadata scientific boundary changed",
    )
    check_sources(spec, data)
    check_inventory(spec)
    archive = opened.artifact_path(path, spec["archive"])
    file_descriptor({k: spec["archive"][k] for k in ("bytes", "sha256")}, MAX_ARCHIVE)
    opened.checked_file(archive, spec["archive"])
    return spec, archive, {path: checksum(payload), archive: spec["archive"]["sha256"]}


def extract_metadata(root, archive, spec, minimum):
    folder = root / "metadata"
    seen = set()
    bound = MAX_DECODED + len(spec["files"]) * 1024 + 10240
    with gzip.open(archive, "rb") as compressed:
        payload = compressed.read(bound + 1)
    require(len(payload) <= bound, "Metadata archive decoded stream exceeds its bound")
    with tarfile.open(
        fileobj=io.BytesIO(payload), mode="r:", ignore_zeros=True
    ) as stream:
        for member in stream:
            require(
                member.name in spec["files"]
                and member.name not in seen
                and member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                and not member.issparse()
                and not member.pax_headers,
                "Metadata archive has an unknown, duplicate or nonregular member",
            )
            expected = spec["files"][member.name]
            require(
                member.size == expected["bytes"], "Metadata archive member size changed"
            )
            target = folder / member_name(member.name)
            opened.guard_path(root, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            require(
                shutil.disk_usage(root).free >= minimum + member.size,
                "Metadata extraction approaches disk reserve",
            )
            content = stream.extractfile(member)
            require(content is not None, "Metadata member payload absent")
            payload = content.read(member.size + 1)
            require(
                len(payload) == member.size and checksum(payload) == expected["sha256"],
                "Metadata member bytes changed",
            )
            target.write_bytes(payload)
            target.chmod(0o444)
            seen.add(member.name)
    require(seen == set(spec["files"]), "Metadata archive lost declared members")


def schemas(root, spec):
    result = {}
    for name, item in spec["schemas"].items():
        path = root / "metadata" / name
        opened.checked_file(path, item)
        require(
            path.stat().st_size <= 32 * 1024, "Arrow schema exceeds bounded metadata"
        )
        schema = pa.ipc.read_schema(pa.BufferReader(path.read_bytes()))
        require(
            schema.equals(CADENCE_SCHEMA, check_metadata=True)
            or schema.equals(QDIAG_SCHEMA, check_metadata=True),
            "Original table schema changed",
        )
        result[name] = schema
    require(
        {len(s) for s in result.values()} == {12, 18},
        "Both original table schemas are required",
    )
    return result


def source_metadata(root, spec, sid, row):
    item = spec["sources"][sid]
    path = root / "metadata" / item["metadata"]["member"]
    opened.checked_file(path, item["metadata"])
    parquet = pq.ParquetFile(path)
    require(
        parquet.metadata.num_rows == 1
        and parquet.num_row_groups == 1
        and parquet.schema_arrow.equals(SOURCE_SCHEMA, check_metadata=True),
        "Original source metadata requires exactly one row and its22-field schema",
    )
    table = lc2.read_source_rows(path, int(sid))
    metadata = table.to_pylist()[0]
    expected = row["metadata"]
    require(
        str(metadata["source_id"]) == sid
        and metadata["raw_fits_bytes"] == row["bytes"]
        and all(
            metadata[k] == expected[k]
            for k in expected
            if k not in {"source_id", "shard", "shard_path"}
        ),
        "Original22-field metadata disagrees with frozen source identity",
    )
    require(
        metadata["stitched_rows"] == row["arrays"]["time_bkjd"]["shape"][0],
        "Original source row count changed",
    )
    return table


def lineage(root, spec, data):
    manifests = {}
    missing = {}
    for cohort, files in spec["original_lineage"].items():
        folder = root / "metadata/lineage" / cohort
        manifest = opened.parse((folder / "input-manifest.json").read_bytes())
        materialization = opened.parse((folder / "materialization.json").read_bytes())
        ids = opened.parse((folder / "source-ids.json").read_bytes())
        expected = [
            sid
            for sid in spec["source_order"]
            if spec["sources"][sid]["opened_cohort"] == cohort
        ]
        require(
            ids == expected
            and list(manifest["sources"]) == expected
            and len(expected) == COHORTS[cohort]
            and manifest["schema"] == 2
            and manifest["transform"] == "additive_flux_delta"
            and materialization["status"] == "FROZEN"
            and materialization["partition"] == "development"
            and materialization["source_ids"] == expected
            and materialization["sources"] == len(expected),
            "Original cohort lineage/order changed",
        )
        require(
            all(
                materialization["artifacts"][name] == files[name]["sha256"]
                for name in ("input-manifest.json", "source-ids.json", "truth.jsonl")
            ),
            "Original frozen artifact bindings changed",
        )
        for sid, item in manifest["sources"].items():
            row = data["sources"][sid]
            require(
                item["base_raw_fits_sha256"] == row["sha256"]
                and item["sha256"] == row["delta"]["sha256"]
                and item["source_id"] == sid
                and item["metadata"] == row["metadata"]
                and row["origin"]["original_input_manifest"]["sha256"]
                == files["input-manifest.json"]["sha256"],
                "Original schema2 source or delta lineage changed",
            )
        missing[cohort] = sorted(set(materialization["artifacts"]) - set(files))
        manifests[cohort] = manifest
    return manifests, missing


def selection(manifest, spec):
    ids = [sid for sid in manifest["sources"] if sid in spec["sources"]]
    require(
        ids and ids == [sid for sid in spec["source_order"] if sid in ids],
        "Verified cache has no injected sources or differs from original order",
    )
    return {
        "cache_computational_selection": manifest["computational_selection"],
        "selected_source_ids": ids,
        "original_injected_source_count": len(spec["source_order"]),
        "selected_cache_injected_sources_complete": True,
        "full180_provisioning_complete": ids == spec["source_order"],
    }


def forecast_values(spec, data, ids):
    numeric = sum(
        sum(v["bytes"] for v in data["sources"][sid]["arrays"].values())
        + 8 * data["sources"][sid]["arrays"]["qdiag_time_bkjd"]["shape"][0]
        for sid in ids
    )
    shards = {
        (spec["sources"][sid]["volume"], spec["sources"][sid]["shard"]) for sid in ids
    }
    deltas = sum(data["sources"][sid]["delta"]["bytes"] for sid in spec["source_order"])
    growth = (
        spec["uncompressed_member_bytes"]
        + deltas
        + 2 * numeric
        + 3 * len(shards) * 1024**2
        + len(ids) * 1024**2
        + 16 * 1024**2
    )
    return {
        "forecast_growth_bytes": growth,
        "next_preparation_all_views_bytes": sum(
            opened.preparation_bytes(data["sources"][sid]) for sid in ids
        ),
        "paired_run_headroom_bytes": data["budgets"]["paired_run_headroom_bytes"],
        "preparation_and_paired_run_space_reserved_here": False,
    }


def forecast(root, spec, data, ids, minimum):
    values = forecast_values(spec, data, ids)
    growth = values["forecast_growth_bytes"]
    disk = shutil.disk_usage(root)
    within = (
        not cloud_destination(root)
        or disk.used + growth + minimum <= data["budgets"]["max_cloud_storage_bytes"]
    )
    require(
        disk.free >= minimum + growth and within,
        "Corpus forecast exceeds storage budget or free-space reserve",
    )
    return {
        **values,
        "reserve_bytes": minimum,
        "current_filesystem_used_bytes": disk.used,
        "available_bytes_before": disk.free,
        "within_declared_cloud32gb_budget": within,
        "cloud_budget_applied": cloud_destination(root),
        "filesystem_measurements": "HISTORICAL_AT_CONSTRUCTION",
    }


def copy_deltas(root, cache, spec, data, minimum):
    folder = root / "deltas"
    folder.mkdir()
    for sid in spec["source_order"]:
        item = data["sources"][sid]["delta"]
        source = cache / "deltas" / item["path"]
        opened.checked_file(source, item)
        target = folder / f"{sid}.npz"
        require(
            shutil.disk_usage(root).free >= minimum + item["bytes"],
            "Delta copying approaches disk reserve",
        )
        shutil.copyfile(source, target)
        opened.checked_file(target, item)
        target.chmod(0o444)


def transported_manifest(root, spec, manifest):
    result = opened.parse(opened.serialized(manifest))
    for sid, item in result["sources"].items():
        source = spec["sources"][sid]
        item["deltas"] = f"../../deltas/{sid}.npz"
        item["metadata"]["shard_path"] = str(
            root / source["volume"] / f"data/kbonus/lc2/shard-{source['shard']:04d}"
        )
    return result


def transport_manifests(root, spec, originals):
    result = {}
    for cohort, manifest in originals.items():
        folder = root / "transport" / cohort
        folder.mkdir(parents=True)
        path = folder / "input-manifest.json"
        opened.publish(
            root, path, opened.serialized(transported_manifest(root, spec, manifest))
        )
        result[cohort] = str(path)
    return result


def tables(raw, metadata, schema_map, names, sid):
    result = {"source": metadata}
    for name in ("cadence", "qdiag"):
        schema = schema_map[names[name]]
        require(
            schema.equals(
                CADENCE_SCHEMA if name == "cadence" else QDIAG_SCHEMA,
                check_metadata=True,
            ),
            "Source table schema was swapped",
        )
        arrays = []
        for field in schema:
            value = (
                np.full(len(raw["qdiag_time_bkjd"]), int(sid), dtype=np.int64)
                if name == "qdiag" and field.name == "source_id"
                else raw[f"qdiag_{field.name}" if name == "qdiag" else field.name]
            )
            arrays.append(pa.array(value, type=field.type, from_pandas=False))
        result[name] = pa.Table.from_arrays(arrays, schema=schema)
    return result


def source_tables(root, cache, spec, data, sid, schema_map, minimum):
    row, item = data["sources"][sid], spec["sources"][sid]
    growth = 2 * sum(v["bytes"] for v in row["arrays"].values()) + 3 * 1024**2
    require(
        shutil.disk_usage(root).free >= minimum + growth,
        "Corpus writing approaches free-space reserve",
    )
    path = cache / f"{sid}.fits"
    opened.checked_file(path, row)
    opened.checked_header(path, row)
    raw = opened.public.fits_arrays(path, sid, row)
    if row["base_arrays"] is not None:
        opened.public.verify_inventory(raw, row["base_arrays"])
        opened.public.verify_arrays(raw, row["base_checks"])
    return tables(
        raw, source_metadata(root, spec, sid, row), schema_map, item["schemas"], sid
    )


def build_shard(root, cache, spec, data, ids, schema_map, minimum):
    first = spec["sources"][ids[0]]
    folder = root / first["volume"] / f"data/kbonus/lc2/shard-{first['shard']:04d}"
    folder.mkdir(parents=True)
    writers = {}
    try:
        for sid in ids:
            values = source_tables(root, cache, spec, data, sid, schema_map, minimum)
            for name, table in values.items():
                if name not in writers:
                    writers[name] = pq.ParquetWriter(
                        folder / f"{name}.parquet", table.schema
                    )
                writers[name].write_table(table, row_group_size=len(table))
    finally:
        for writer in writers.values():
            writer.close()


def build_index(root, spec, ids):
    path = root / "discSSD/data/index.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE sources(gaia_source_id TEXT PRIMARY KEY,volume TEXT NOT NULL,shard INTEGER NOT NULL)"
        )
        database.executemany(
            "INSERT INTO sources VALUES(?,?,?)",
            [
                (sid, spec["sources"][sid]["volume"], spec["sources"][sid]["shard"])
                for sid in ids
            ],
        )


def prepare_view(raw, view):
    try:
        records = runner.prepare_records(raw, view)
    except runner.InsufficientCadences as exc:
        require(
            type(exc.count) is int and 0 <= exc.count < 100,
            "Unavailable preparation count is invalid",
        )
        return None, exc.count
    require(len(records) >= 100, "Prepared view has insufficient cadences")
    return records, len(records)


def view_oracles(raw, baseline, row):
    result = {}
    for view in runner.DETECTOR_VIEWS:
        raw_hash = lc2.curve_digest(lc2.photometry(raw, view))
        require(
            raw_hash == lc2.curve_digest(lc2.photometry(baseline, view)),
            "Corpus raw preparation view differs from verified original input",
        )
        records, count = prepare_view(raw, view)
        original, original_count = prepare_view(baseline, view)
        require(
            (records is None) == (original is None) and count == original_count,
            "Corpus preparation availability or cadence count differs from original",
        )
        if records is None:
            require(
                view != runner.DISCOVERY_VIEW,
                "Original discovery view must remain eligible",
            )
            result[view] = {
                "status": "UNAVAILABLE",
                "reason": "InsufficientCadences",
                "raw_view_sha256": raw_hash,
                "retained_cadences": count,
            }
            continue
        require(
            records.dtype == original.dtype and records.tobytes() == original.tobytes(),
            "Corpus preparation view differs from verified original input",
        )
        record_hash = checksum(records.tobytes())
        if view == runner.DISCOVERY_VIEW:
            require(
                raw_hash == row["raw_view_sha256"]
                and record_hash == row["prepared_records_sha256"]
                and len(records) == row["prepared_cadences"],
                "Original discovery/FSC oracle changed",
            )
        header = struct.pack(
            "<8sQQQ",
            b"FSC_LC2\0",
            int(row["metadata"]["source_id"]),
            runner.VIEW_CODES[view],
            len(records),
        )
        result[view] = {
            "status": "PREPARED",
            "raw_view_sha256": raw_hash,
            "prepared_records_sha256": record_hash,
            "fsc_sha256": checksum(header + records.tobytes()),
            "retained_cadences": len(records),
        }
    return result


def check_preparation_views(sources, ids):
    require(list(sources) == ids, "Corpus proof source identities or order changed")
    common = {"status", "raw_view_sha256", "retained_cadences"}
    for source in sources.values():
        views = source["preparation_views"]
        require(
            set(views) == set(runner.DETECTOR_VIEWS),
            "Corpus proof lost a preparation view",
        )
        for view, item in views.items():
            require(
                type(item["retained_cadences"]) is int
                and opened.public.sha256(item["raw_view_sha256"]),
                "Corpus preparation proof types changed",
            )
            if item["status"] == "UNAVAILABLE":
                require(
                    set(item) == common | {"reason"}
                    and item["reason"] == "InsufficientCadences"
                    and 0 <= item["retained_cadences"] < 100
                    and view != runner.DISCOVERY_VIEW,
                    "Corpus unavailable preparation evidence changed",
                )
            else:
                require(
                    item["status"] == "PREPARED"
                    and set(item) == common | {"prepared_records_sha256", "fsc_sha256"}
                    and item["retained_cadences"] >= 100
                    and all(
                        opened.public.sha256(item[key])
                        for key in ("prepared_records_sha256", "fsc_sha256")
                    ),
                    "Corpus eligible preparation evidence changed",
                )


def verify_sources(root, cache, spec, data, ids, manifests):
    lc2.source_row_groups.cache_clear()
    result = {}
    for sid in ids:
        row, item = data["sources"][sid], spec["sources"][sid]
        metadata, raw = lc2.manifest_source(root, sid, manifests[item["opened_cohort"]])
        opened.public.verify_inventory(raw, row["arrays"])
        expected = source_metadata(root, spec, sid, row).to_pylist()[0]
        expected.update(
            source_id=sid,
            shard=item["shard"],
            shard_path=str(
                root / item["volume"] / f"data/kbonus/lc2/shard-{item['shard']:04d}"
            ),
            input_manifest=str(Path(manifests[item["opened_cohort"]]).resolve()),
        )
        require(metadata == expected, "Corpus reader changed original22-field metadata")
        baseline = opened.npz_values(cache / f"{sid}.npz", row["arrays"])
        result[sid] = {
            "opened_cohort": item["opened_cohort"],
            "volume": item["volume"],
            "shard": item["shard"],
            "primary_rows": len(raw["time_bkjd"]),
            "qdiag_rows": len(raw["qdiag_time_bkjd"]),
            "full29_array_inventory_verified": True,
            "original22_metadata_verified": True,
            "base_array_oracle_available": row["base_arrays"] is not None,
            "preparation_views": view_oracles(raw, baseline, row),
        }
    return result


def output_inventory(root):
    result = {}
    for path in root.rglob("*"):
        opened.guard_path(root, path)
        require(path.is_dir() or path.is_file(), "Corpus output has a nonregular entry")
        if path.is_file() and path not in {root / name for name in SUCCESS}:
            result[str(path.relative_to(root))] = identity(path)
    return result


def shard_groups(spec, ids):
    groups = defaultdict(list)
    for sid in ids:
        item = spec["sources"][sid]
        groups[item["volume"], item["shard"]].append(sid)
    return groups


def expected_outputs(spec, ids):
    names = {f"metadata/{name}" for name in spec["files"]}
    names.add(OWNER)
    names.update(f"deltas/{sid}.npz" for sid in spec["source_order"])
    names.update(f"transport/{cohort}/input-manifest.json" for cohort in COHORTS)
    names.add("discSSD/data/index.sqlite")
    for volume, shard in shard_groups(spec, ids):
        names.update(
            f"{volume}/data/kbonus/lc2/shard-{shard:04d}/{name}.parquet"
            for name in ("source", "cadence", "qdiag")
        )
    return names


def check_output_tree(root, outputs, spec, ids):
    expected = expected_outputs(spec, ids)
    require(set(outputs) == expected, "Corpus file inventory is not closed")
    directories = {Path(name).parent.as_posix() for name in expected}
    directories.update(
        parent.as_posix() for name in expected for parent in Path(name).parents
    )
    directories.update({"discSSD", "discPCIe"})
    for path in root.rglob("*"):
        opened.guard_path(root, path)
        name = path.relative_to(root).as_posix()
        require(
            name in directories
            if path.is_dir()
            else name in expected or name in SUCCESS,
            "Corpus contains an unknown or partial output",
        )
    for name, item in outputs.items():
        file_descriptor(item, 2**63 - 1)
        opened.checked_file(root / name, item)


def check_parquet(root, spec, data, ids):
    schemas_by_name = {
        "source": SOURCE_SCHEMA,
        "cadence": CADENCE_SCHEMA,
        "qdiag": QDIAG_SCHEMA,
    }
    for (volume, shard), members in shard_groups(spec, ids).items():
        folder = root / volume / f"data/kbonus/lc2/shard-{shard:04d}"
        for name, schema in schemas_by_name.items():
            parquet = pq.ParquetFile(folder / f"{name}.parquet")
            require(
                parquet.schema_arrow.equals(schema, check_metadata=True)
                and parquet.num_row_groups == len(members),
                "Corpus table schema or source-specific row groups changed",
            )
            for number, sid in enumerate(members):
                group = parquet.metadata.row_group(number)
                rows = (
                    1
                    if name == "source"
                    else data["sources"][sid]["arrays"][
                        "qdiag_time_bkjd" if name == "qdiag" else "time_bkjd"
                    ]["shape"][0]
                )
                stat = group.column(schema.get_field_index("source_id")).statistics
                require(
                    group.num_rows == rows
                    and stat is not None
                    and stat.min == stat.max == int(sid)
                    and group.total_byte_size
                    <= 8 * spec["sources"][sid]["metadata"]["bytes"]
                    + 4
                    * sum(x["bytes"] for x in data["sources"][sid]["arrays"].values()),
                    "Corpus row group lost, mixed or expanded source rows",
                )


def check_index(root, spec, ids):
    path = root / "discSSD/data/index.sqlite"
    require(
        path.stat().st_size <= 16 * 1024**2, "Corpus subset index exceeds its bound"
    )
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        objects = database.execute("SELECT name,type FROM sqlite_master").fetchall()
        columns = database.execute("PRAGMA table_info(sources)").fetchall()
        expected_columns = [
            (0, "gaia_source_id", "TEXT", 0, None, 1),
            (1, "volume", "TEXT", 1, None, 0),
            (2, "shard", "INTEGER", 1, None, 0),
        ]
        require(
            sorted(objects)
            == [("sources", "table"), ("sqlite_autoindex_sources_1", "index")]
            and columns == expected_columns
            and database.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
            "Corpus subset index schema changed",
        )
        rows = database.execute(
            "SELECT gaia_source_id,volume,shard FROM sources ORDER BY rowid"
        ).fetchall()
    expected = [
        (sid, spec["sources"][sid]["volume"], spec["sources"][sid]["shard"])
        for sid in ids
    ]
    require(
        rows == expected,
        "Corpus subset index source identities, order or locations changed",
    )


def check_transport(root, cache, spec, data, originals, manifests):
    expected = {
        cohort: str(root / "transport" / cohort / "input-manifest.json")
        for cohort in COHORTS
    }
    require(manifests == expected, "Corpus transport manifest paths changed")
    for cohort, manifest in originals.items():
        expected_payload = opened.serialized(transported_manifest(root, spec, manifest))
        require(
            Path(manifests[cohort]).read_bytes() == expected_payload,
            "Corpus transport changed more than declared paths",
        )
    for sid in spec["source_order"]:
        item = data["sources"][sid]["delta"]
        opened.checked_file(root / "deltas" / f"{sid}.npz", item)
        opened.checked_file(cache / "deltas" / item["path"], item)


def checked_outputs(root, cache, spec, data, ids, originals, manifests, outputs):
    require(
        sum(item["bytes"] for item in outputs.values())
        <= forecast_values(spec, data, ids)["forecast_growth_bytes"],
        "Corpus output bytes exceed the canonical growth forecast",
    )
    check_output_tree(root, outputs, spec, ids)
    for name, item in spec["files"].items():
        opened.checked_file(root / "metadata" / name, item)
    schemas(root, spec)
    check_transport(root, cache, spec, data, originals, manifests)
    check_parquet(root, spec, data, ids)
    check_index(root, spec, ids)


def finish(
    root,
    cache,
    spec,
    selected,
    manifests,
    sources,
    dependencies,
    resources,
    missing,
    outputs,
):
    ids = selected["selected_source_ids"]
    require(owned(root), "Corpus managed ownership changed during construction")
    require(list(sources) == ids, "Corpus verification lost or reordered sources")
    resources["disk_growth_bytes"] = (
        resources["available_bytes_before"] - shutil.disk_usage(root).free
    )
    require(
        0 <= resources["disk_growth_bytes"] <= resources["forecast_growth_bytes"],
        "Corpus exceeded bounded storage growth",
    )
    output_dependencies = {
        root / name: item["sha256"] for name, item in outputs.items()
    }
    dependencies.update(output_dependencies)
    payload = opened.serialized(ids)
    opened.publish(root, root / "source-ids.json", payload)
    dependencies[root / "source-ids.json"] = checksum(payload)
    receipt = {
        "schema": 1,
        "status": "OPENED_ORIGINAL180_CORPUS_COMPLETE"
        if selected["full180_provisioning_complete"]
        else "OPENED_ORIGINAL180_CORPUS_PARTIAL",
        "scope": opened.public.SCOPE,
        "role": "development",
        **{flag: False for flag in FLAGS},
        "full_frozen_materialization_created": False,
        "corpus_root": str(root),
        "verified_cache": str(cache),
        "selection": selected,
        "original_source_order": spec["source_order"],
        "source_ids_sha256": checksum(payload),
        "manifests": manifests,
        "sources": sources,
        "outputs": outputs,
        "dependencies": {str(p): value for p, value in dependencies.items()},
        "resources": resources,
        "unprovisioned_original_materialization_artifacts": missing,
        "boundary": BOUNDARY,
    }
    opened.recheck(dependencies)
    payload = opened.serialized(receipt)
    opened.publish(root, root / "receipt.json", payload)
    opened.recheck(dependencies)
    require(
        (root / "receipt.json").read_bytes() == payload,
        "Corpus receipt changed during publication",
    )
    return receipt


def provision(
    specification=SPECIFICATION, cache=None, destination=None, minimum_free_bytes=None
):
    destination = (
        Path(destination)
        if destination is not None
        else ROOT / "dump/cloud-opened-corpus"
    )
    root = opened.destination_path(destination)
    token = ownership_token(root)
    withdraw(root)
    try:
        cache = opened.destination_path(
            Path(cache) if cache is not None else ROOT / "dump/cloud-opened"
        )
        manifest, _, dependencies = opened.verified_inputs(destination=cache)
        data, _, packet_dependencies = opened.packet()
        bind_dependencies(dependencies, packet_dependencies)
        dependencies[Path(__file__)] = opened.digest(Path(__file__))
        spec, archive, metadata_dependencies = metadata_packet(specification, data)
        bind_dependencies(dependencies, metadata_dependencies)
        selected = selection(manifest, spec)
        root = begin(destination, cache)
        token = ownership_token(root)
        minimum = reserve(root, minimum_free_bytes)
        resources = forecast(root, spec, data, selected["selected_source_ids"], minimum)
        extract_metadata(root, archive, spec, minimum)
        schema_map = schemas(root, spec)
        originals, missing = lineage(root, spec, data)
        copy_deltas(root, cache, spec, data, minimum)
        manifests = transport_manifests(root, spec, originals)
        groups = shard_groups(spec, selected["selected_source_ids"])
        for ids in groups.values():
            build_shard(root, cache, spec, data, ids, schema_map, minimum)
        build_index(root, spec, selected["selected_source_ids"])
        outputs = output_inventory(root)
        checked_outputs(
            root,
            cache,
            spec,
            data,
            selected["selected_source_ids"],
            originals,
            manifests,
            outputs,
        )
        sources = verify_sources(
            root, cache, spec, data, selected["selected_source_ids"], manifests
        )
        return finish(
            root,
            cache,
            spec,
            selected,
            manifests,
            sources,
            dependencies,
            resources,
            missing,
            outputs,
        )
    except BaseException:
        withdraw(root, token)
        raise


def check_receipt(receipt, root, cache, spec, selected, dependencies):
    fields = {
        "schema",
        "status",
        "scope",
        "role",
        *FLAGS,
        "full_frozen_materialization_created",
        "corpus_root",
        "verified_cache",
        "selection",
        "original_source_order",
        "source_ids_sha256",
        "manifests",
        "sources",
        "outputs",
        "dependencies",
        "resources",
        "unprovisioned_original_materialization_artifacts",
        "boundary",
    }
    status = (
        "OPENED_ORIGINAL180_CORPUS_COMPLETE"
        if selected["full180_provisioning_complete"]
        else "OPENED_ORIGINAL180_CORPUS_PARTIAL"
    )
    require(
        set(receipt) == fields
        and type(receipt["schema"]) is int
        and receipt["schema"] == 1
        and receipt["status"] == status
        and receipt["scope"] == opened.public.SCOPE
        and receipt["role"] == "development"
        and all(receipt[flag] is False for flag in FLAGS)
        and receipt["full_frozen_materialization_created"] is False
        and receipt["corpus_root"] == str(root)
        and receipt["verified_cache"] == str(cache)
        and opened.serialized(receipt["selection"]) == opened.serialized(selected)
        and receipt["original_source_order"] == spec["source_order"]
        and receipt["boundary"] == BOUNDARY,
        "Corpus receipt boundary or selected completion changed",
    )
    ids = selected["selected_source_ids"]
    check_preparation_views(receipt["sources"], ids)
    expected = opened.serialized(ids)
    require(
        receipt["source_ids_sha256"] == checksum(expected)
        and (root / "source-ids.json").read_bytes() == expected,
        "Corpus selected source identities changed",
    )
    output_dependencies = {
        str(root / name): item["sha256"] for name, item in receipt["outputs"].items()
    }
    required = {str(p): value for p, value in dependencies.items()}
    required.update(output_dependencies)
    required[str(root / "source-ids.json")] = checksum(expected)
    require(
        receipt["dependencies"] == required,
        "Corpus receipt input/implementation bindings changed",
    )


def check_resources(root, resources, spec, data, ids):
    canonical = forecast_values(spec, data, ids)
    expected_keys = {
        *canonical,
        "reserve_bytes",
        "current_filesystem_used_bytes",
        "available_bytes_before",
        "within_declared_cloud32gb_budget",
        "cloud_budget_applied",
        "filesystem_measurements",
        "disk_growth_bytes",
    }
    require(
        set(resources) == expected_keys
        and resources["filesystem_measurements"] == "HISTORICAL_AT_CONSTRUCTION"
        and resources["preparation_and_paired_run_space_reserved_here"] is False,
        "Corpus resource report field inventory or measurement boundary changed",
    )
    numeric = set(expected_keys) - {
        "within_declared_cloud32gb_budget",
        "cloud_budget_applied",
        "filesystem_measurements",
        "preparation_and_paired_run_space_reserved_here",
    }
    require(
        all(
            type(resources[name]) is int and 0 <= resources[name] <= 2**63 - 1
            for name in numeric
        )
        and all(resources[name] == value for name, value in canonical.items()),
        "Corpus resource forecast or types changed",
    )
    minimum = reserve(root, resources["reserve_bytes"])
    cloud = cloud_destination(root)
    within = (
        not cloud
        or resources["current_filesystem_used_bytes"]
        + resources["forecast_growth_bytes"]
        + minimum
        <= data["budgets"]["max_cloud_storage_bytes"]
    )
    require(
        resources["cloud_budget_applied"] is cloud
        and resources["within_declared_cloud32gb_budget"] is within
        and within
        and resources["available_bytes_before"]
        >= minimum + resources["forecast_growth_bytes"]
        and resources["disk_growth_bytes"] <= resources["forecast_growth_bytes"],
        "Corpus historical storage accounting is inconsistent",
    )


def verified_corpus(specification=SPECIFICATION, cache=None, destination=None):
    root = opened.destination_path(
        Path(destination)
        if destination is not None
        else ROOT / "dump/cloud-opened-corpus"
    )
    token = ownership_token(root)
    try:
        require(owned(root), "Corpus managed ownership marker absent or changed")
        cache = opened.destination_path(
            Path(cache) if cache is not None else ROOT / "dump/cloud-opened"
        )
        path = root / "receipt.json"
        opened.guard_path(root, path)
        require(
            path.is_file() and path.stat().st_size <= 4 * 1024**2,
            "Corpus receipt absent or oversized",
        )
        payload = path.read_bytes()
        receipt = opened.parse(payload)
        manifest, _, dependencies = opened.verified_inputs(destination=cache)
        data, _, current_dependencies = opened.packet()
        bind_dependencies(dependencies, current_dependencies)
        dependencies[Path(__file__)] = opened.digest(Path(__file__))
        spec, _, metadata_dependencies = metadata_packet(specification, data)
        bind_dependencies(dependencies, metadata_dependencies)
        selected = selection(manifest, spec)
        check_receipt(receipt, root, cache, spec, selected, dependencies)
        check_resources(
            root, receipt["resources"], spec, data, selected["selected_source_ids"]
        )
        originals, missing = lineage(root, spec, data)
        checked_outputs(
            root,
            cache,
            spec,
            data,
            selected["selected_source_ids"],
            originals,
            receipt["manifests"],
            receipt["outputs"],
        )
        sources = verify_sources(
            root,
            cache,
            spec,
            data,
            selected["selected_source_ids"],
            receipt["manifests"],
        )
        require(
            opened.serialized(sources) == opened.serialized(receipt["sources"])
            and missing == receipt["unprovisioned_original_materialization_artifacts"],
            "Corpus reader/original-lineage proofs changed",
        )
        dependencies.update(
            {Path(p): value for p, value in receipt["dependencies"].items()}
        )
        dependencies[path] = checksum(payload)
        opened.recheck(dependencies)
        return receipt, dependencies
    except BaseException:
        withdraw(root, token)
        raise
