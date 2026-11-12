"""Bind frozen null memberships to original LC2 metadata without reading flux."""

import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.deterministic import cloud_null as pool
from pipeline.photometry import lc2
from pipeline.runtime import dump_path

QUOTA = 10000
POOL_SPECIFICATION = Path(__file__).with_name("cloud-null-pool.json")
PRODUCTS = ("source", "cadence", "qdiag")
SCENE_FIELDS = (
    "source_id",
    "ra_deg",
    "dec_deg",
    "origin_tpf_kic",
    "phot_g_mean_mag",
    "shard",
    "raw_fits_sha256",
)


def reference(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": pool.digest(path)}


def verified_bytes(path, binding):
    data = path.read_bytes()
    pool.require(
        len(data) == binding["bytes"] and hashlib.sha256(data).hexdigest() == binding["sha256"],
        f"Bound metadata changed: {path}",
    )
    return data


def bound_reference(path, binding):
    return {"path": str(path.resolve()), "bytes": binding["bytes"], "sha256": binding["sha256"]}


def frozen_membership(declaration_path, expected_sha256, membership_path, role):
    data = declaration_path.read_bytes()
    pool.require(
        pool.valid_hash(expected_sha256) and hashlib.sha256(data).hexdigest() == expected_sha256,
        "Frozen null declaration hash changed",
    )
    declaration = json.loads(data)
    pool.require(
        declaration["schema"] == 1
        and declaration["status"] == "FIRST_OUTCOME_BLIND_NULL_MEMBERSHIP_DECLARATION"
        and declaration["membership_and_seeds_frozen"] is True
        and declaration["selection_seed"] == declaration["transform_seed"] == 20261002
        and declaration["execution_authorized"] is False,
        "Invalid frozen null declaration",
    )
    pool.require(role in {"development", "evaluation"}, "Null base role must be explicit")
    binding = declaration["artifacts"][f"{role}-membership.csv"]
    payload = verified_bytes(membership_path, binding)
    reader = csv.DictReader(io.StringIO(payload.decode(), newline=""), strict=True)
    pool.require(reader.fieldnames == list(pool.FIELDS), "Original null membership fields changed")
    rows = list(reader)
    pool.require(
        len(rows) == binding["sources"] == declaration["counts"][role] == QUOTA and binding["role"] == role,
        "Exact null role quota or role binding differs",
    )
    for row in rows:
        pool.row_identity(row)
        pool.require(
            not row["linked_existing_scopes"] and not row["linked_full_witness_component_ids"],
            "Existing control-linked source entered null membership",
        )
    for key in ("source_id", "origin_parent", "raw_fits_sha256", "metadata_dependency_component_id"):
        pool.require(len({row[key] for row in rows}) == QUOTA, f"Repeated null membership identity: {key}")
    ids = [row["source_id"] for row in rows]
    pool.require(
        hashlib.sha256(pool.canonical(ids)).hexdigest() == binding["ordered_source_ids_sha256"],
        "Frozen null source order changed",
    )
    return (
        declaration,
        rows,
        [
            bound_reference(declaration_path, {"bytes": len(data), "sha256": expected_sha256}),
            bound_reference(membership_path, binding),
        ],
    )


def original_index_binding(declaration):
    matches = [row for row in declaration["inputs"] if Path(row["path"]).name == POOL_SPECIFICATION.name]
    pool.require(len(matches) == 1, "Frozen null declaration lacks its original pool specification")
    specification = json.loads(verified_bytes(POOL_SPECIFICATION, matches[0]))
    bindings = [
        row
        for row in specification["provenance"]["original_inputs"]
        if row["origin_path"] == "discSSD/data/index.sqlite"
    ]
    pool.require(len(bindings) == 1, "Original scanned-half index binding is absent or ambiguous")
    return bindings[0], bound_reference(POOL_SPECIFICATION, matches[0])


def location(volume, shard):
    pool.require(
        volume in {"discSSD", "discPCIe"} and type(shard) is int and 0 <= shard <= 9999,
        "Invalid indexed LC2 volume or shard",
    )
    return f"{volume}/data/kbonus/lc2/shard-{shard:04d}"


def indexed_shards(database, rows):
    folders = {}
    for row in rows:
        matches = database.execute(
            "select volume,shard from sources where gaia_source_id=?", (row["source_id"],)
        ).fetchall()
        pool.require(len(matches) == 1, f"Source missing or repeated in original index: {row['source_id']}")
        volume, shard = matches[0]
        folder = location(volume, shard)
        pool.require(shard == int(row["shard"]), f"Original null shard differs: {row['source_id']}")
        folders[folder] = (volume, shard)
    return folders


def indexed_product(database, root, volume, shard, name):
    relative = f"kbonus/lc2/shard-{shard:04d}/{name}.parquet"
    matches = database.execute(
        "select bytes,sha256,status,format from files where collection='kbonus' and volume=? and path=?",
        (volume, relative),
    ).fetchall()
    pool.require(len(matches) == 1, f"Original indexed product missing or repeated: {relative}")
    size, sha256, status, codec = matches[0]
    pool.require(
        type(size) is int
        and size > 0
        and pool.valid_hash(sha256)
        and status == "available"
        and codec == "parquet",
        f"Invalid indexed LC2 product: {relative}",
    )
    path = root / volume / "data" / relative
    pool.require(path.is_file() and path.stat().st_size == size, f"Original LC2 product size differs: {path}")
    return {"bytes": size, "sha256": sha256}


def scene_source(row, folder, shard):
    source = str(row["source_id"])
    pool.require(
        pool.decimal_identity(source)
        and type(row["origin_tpf_kic"]) is int
        and row["origin_tpf_kic"] > 0
        and pool.valid_hash(row["raw_fits_sha256"]),
        "Original source metadata lacks source/parent/raw identities",
    )
    metadata = lc2.finite_metadata(dict(row, source_id=source, shard=shard, shard_path=str(folder)))
    pool.require(
        all(
            type(metadata[key]) in {int, float} and math.isfinite(metadata[key])
            for key in ("ra_deg", "dec_deg")
        )
        and 0 <= metadata["ra_deg"] < 360
        and -90 <= metadata["dec_deg"] <= 90,
        f"Original source lacks valid scene coordinates: {source}",
    )
    magnitude = metadata["phot_g_mean_mag"]
    pool.require(
        magnitude is None or (type(magnitude) in {int, float} and math.isfinite(magnitude)),
        f"Invalid scene magnitude: {source}",
    )
    return metadata


def shard_metadata(database, root, volume, shard, products):
    folder = root / location(volume, shard)
    path = folder / "source.parquet"
    data = verified_bytes(path, products["source"])
    records = pq.ParquetFile(io.BytesIO(data)).read().to_pylist()
    metadata = [scene_source(row, folder, shard) for row in records]
    expected = [
        str(row[0])
        for row in database.execute(
            "select gaia_source_id from sources where volume=? and shard=?", (volume, shard)
        )
    ]
    ids = [row["source_id"] for row in metadata]
    pool.require(
        len(expected) == len(set(expected)) and len(ids) == len(set(ids)) and set(ids) == set(expected),
        f"Complete original shard scene coverage differs: {folder}",
    )
    return sorted(metadata, key=lambda row: int(row["source_id"])), bound_reference(path, products["source"])


def bind_selected(row, metadata, folder):
    pool.require(
        str(metadata["origin_tpf_kic"]) == row["origin_parent"]
        and metadata["raw_fits_sha256"] == row["raw_fits_sha256"]
        and metadata["shard"] == int(row["shard"])
        and metadata["source_id"] == row["source_id"],
        f"Original null source binding differs: {row['source_id']}",
    )
    return {"lc2_folder": folder, "metadata": metadata, "original_pool_row": row}


def bind_shards(database, root, rows, scene):
    selected = {row["source_id"]: row for row in rows}
    selected_folders = indexed_shards(database, rows)
    sources, products, inputs, coverage, seen = {}, {}, [], [], set()
    for volume, shard in database.execute("select distinct volume,shard from sources order by volume,shard"):
        folder = location(volume, shard)
        names = PRODUCTS if folder in selected_folders else ("source",)
        bound = {name: indexed_product(database, root, volume, shard, name) for name in names}
        if folder in selected_folders:
            products[folder] = bound
        metadata, identity = shard_metadata(database, root, volume, shard, bound)
        inputs.append(identity)
        coverage.append({"lc2_folder": folder, "sources": len(metadata), "source_metadata": identity})
        for source in metadata:
            pool.require(
                source["source_id"] not in seen, "Source identity repeats across original scene shards"
            )
            seen.add(source["source_id"])
            scene.write(pool.canonical({key: source[key] for key in SCENE_FIELDS}) + b"\n")
            if source["source_id"] in selected:
                sid = source["source_id"]
                pool.require(sid not in sources, f"Null source repeats across indexed shards: {sid}")
                sources[sid] = bind_selected(selected[sid], source, folder)
    pool.require(
        set(sources) == set(selected)
        and len(seen) == database.execute("select count(*) from sources").fetchone()[0],
        "Frozen null membership or full original indexed scene is not completely bound",
    )
    return {row["source_id"]: sources[row["source_id"]] for row in rows}, products, inputs, coverage


def produce(declaration_path, expected_sha256, membership_path, role, root, index, output):
    declaration, rows, inputs = frozen_membership(declaration_path, expected_sha256, membership_path, role)
    index_binding, pool_identity = original_index_binding(declaration)
    verified_bytes(index, index_binding)
    pool.require(
        not any(Path(str(index) + suffix).exists() for suffix in ("-wal", "-journal")),
        "Original index has an unbound SQLite journal",
    )
    inputs.extend([pool_identity, bound_reference(index, index_binding)])
    output.mkdir(parents=True, exist_ok=False)
    scene_path = output / "source-scene.jsonl"
    temporary = scene_path.with_suffix(".jsonl.part")
    with (
        sqlite3.connect(f"{index.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as database,
        temporary.open("xb") as scene,
    ):
        sources, products, metadata_inputs, coverage = bind_shards(database, root, rows, scene)
    inputs.extend(metadata_inputs)
    for identity in inputs:
        verified_bytes(Path(identity["path"]), identity)
    temporary.replace(scene_path)
    scene_identity = reference(scene_path)
    manifest = {
        "schema": 3,
        "protocol": lc2.LC2_MANIFEST,
        "role": role,
        "measurement_scope": "OPENED_NULL_BASE" if role == "development" else "SEALED_NULL_BASE",
        "binding_status": "METADATA_ONLY_ORIGINAL_LC2_BINDING",
        "declaration": inputs[0],
        "membership": inputs[1],
        "selection_seed": 20261002,
        "transform_seed": 20261002,
        "ordered_source_ids": [row["source_id"] for row in rows],
        "ordered_source_ids_sha256": declaration["artifacts"][f"{role}-membership.csv"][
            "ordered_source_ids_sha256"
        ],
        "sources": sources,
        "lc2_products": products,
        "scene_metadata": {"path": scene_path.name, "sha256": scene_identity["sha256"]},
        "scene_coverage": {
            "sources": sum(row["sources"] for row in coverage),
            "shards": coverage,
            "bytes": scene_identity["bytes"],
            "complete_original_index": True,
            "verified_index": bound_reference(index, index_binding),
        },
        "metadata_inputs": inputs,
        "product_verification": {
            "source": "REHASHED_BEFORE_METADATA_READ",
            "cadence": "INDEX_HASH_BOUND_SIZE_CHECKED",
            "qdiag": "INDEX_HASH_BOUND_SIZE_CHECKED",
            "before_array_access": "READER_REHASHES_ALL_THREE_PRODUCTS",
        },
        "execution_authorized": False,
        "evaluation_authorization_required": role == "evaluation",
        "null_flux_constructed": False,
        "cadence_or_qdiag_arrays_read": False,
        "quarter_scene_independence_certified": False,
        "same_base_recipes": declaration["same_base_recipes"],
    }
    path = output / "input-manifest.json"
    path.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False) + "\n")
    scene_path.chmod(0o444)
    path.chmod(0o444)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--declaration-sha256", required=True)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("development", "evaluation"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, help="Defaults to ROOT/discSSD/data/index.sqlite")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = dump_path(args.out)
    result = produce(
        args.declaration,
        args.declaration_sha256,
        args.membership,
        args.role,
        args.root,
        args.index or args.root / "discSSD/data/index.sqlite",
        output,
    )
    print(
        json.dumps(
            {
                "input_manifest": reference(output / "input-manifest.json"),
                "role": result["role"],
                "selected_sources": len(result["sources"]),
                "scene_sources": result["scene_coverage"]["sources"],
                "binding_status": result["binding_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
