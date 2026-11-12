"""Freeze the complete LC2 source ledger and verify indexed input bytes."""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.runtime import dump_path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = {"source", "cadence", "qdiag", "quarter", "header", "aperture"}
SOURCE_FIELDS = ["source_id", "origin_tpf_kic", "raw_fits_sha256", "raw_fits_bytes"]
LEDGER_FIELDS = ["source_id", "volume", "shard", *SOURCE_FIELDS[1:]]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(".building")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Completed input artifact changed: {path}")
        return
    write(path, value)


def selected_sources(selection, expected):
    ids = read(selection)
    if (
        not isinstance(ids, list)
        or len(ids) != expected
        or any(not isinstance(sid, str) or not sid.isdecimal() for sid in ids)
        or len(set(ids)) != expected
    ):
        raise ValueError("Frozen source selection is incomplete or malformed")
    return ids


def indexes(root, expected):
    inputs = {}
    for volume in ("discSSD", "discPCIe"):
        folder = root / volume / "data"
        manifest = folder / "index.json"
        index = folder / "index.sqlite"
        declared = read(manifest)
        checksum = digest(index)
        if (
            checksum != declared["lookup_sha256"]
            or declared["primary_collections"]["kbonus"]["sources"] != expected
        ):
            raise ValueError(
                f"Mounted corpus index differs from its declaration: {volume}"
            )
        inputs[str(manifest.resolve())] = digest(manifest)
        inputs[str(index.resolve())] = checksum
    if (
        len(
            {
                inputs[str((root / v / "data/index.sqlite").resolve())]
                for v in ("discSSD", "discPCIe")
            }
        )
        != 1
    ):
        raise ValueError("Mounted corpus indexes differ")
    return inputs


def inventory(root, ids):
    index = root / "discSSD/data/index.sqlite"
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
        locations = {
            sid: (volume, int(shard))
            for sid, volume, shard in database.execute(
                "select gaia_source_id, volume, shard from sources"
            )
        }
        records = database.execute(
            "select volume,path,bytes,sha256,status,format from files "
            "where collection='kbonus' and path glob 'kbonus/lc2/shard-*/*.parquet'"
        ).fetchall()
    if set(locations) != set(ids):
        raise ValueError("Frozen selection differs from the complete LC2 index")
    groups = defaultdict(set)
    for sid, location in locations.items():
        groups[location].add(sid)
    products = {}
    for volume, path, size, checksum, status, format_name in records:
        if (
            volume not in {"discSSD", "discPCIe"}
            or status != "available"
            or format_name != "parquet"
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise ValueError(f"Invalid declared LC2 product: {path}")
        key = (volume, path)
        if key in products:
            raise ValueError("Repeated indexed LC2 product")
        products[key] = {
            "volume": volume,
            "path": path,
            "bytes": size,
            "sha256": checksum,
        }
    wanted = {
        (volume, f"kbonus/lc2/shard-{shard:04d}/{name}.parquet")
        for volume, shard in groups
        for name in PRODUCTS
    }
    if products.keys() != wanted:
        raise ValueError("Each declared LC2 shard requires exactly six input products")
    return groups, products


def file_stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_product(root, record):
    path = root / record["volume"] / "data" / record["path"]
    before = file_stamp(path)
    if before[2] != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"LC2 input bytes differ from the corpus index: {path}")
    if file_stamp(path) != before:
        raise ValueError(f"LC2 input changed during verification: {path}")
    return path, before


def source_rows(path, volume, shard, expected):
    rows = pq.read_table(path, columns=SOURCE_FIELDS, use_threads=False).to_pylist()
    ids = [str(row["source_id"]) for row in rows]
    if len(set(ids)) != len(rows) or set(ids) != expected:
        raise ValueError(f"Source metadata does not close its indexed shard: {path}")
    result = {}
    for row, sid in zip(rows, ids):
        checksum = row["raw_fits_sha256"]
        if (
            not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or not isinstance(row["raw_fits_bytes"], int)
            or row["raw_fits_bytes"] <= 0
        ):
            raise ValueError(f"Original FITS identity is missing: {sid}")
        result[sid] = {**row, "source_id": sid, "volume": volume, "shard": shard}
    return result


def write_ledger(path, ids, rows):
    temporary = path.with_suffix(".building")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows[sid] for sid in ids)
    if path.exists() and digest(path) != digest(temporary):
        raise ValueError("Completed source ledger changed")
    temporary.replace(path)


def verify_files(root, out, groups, products, identity):
    rows, stamps = {}, {}
    identity_stamps = {Path(path): file_stamp(Path(path)) for path in identity}
    ordered = sorted(products.values(), key=lambda r: (r["volume"], r["path"]))
    for number, record in enumerate(ordered, 1):
        if any(file_stamp(path) != stamp for path, stamp in identity_stamps.items()):
            raise ValueError("Frozen ledger identity changed during verification")
        path, stamp = verify_product(root, record)
        stamps[path] = stamp
        relative = Path(record["path"])
        shard = int(relative.parent.name.removeprefix("shard-"))
        volume = record["volume"]
        if relative.name == "source.parquet":
            rows.update(source_rows(path, volume, shard, groups[(volume, shard)]))
            if file_stamp(path) != stamp:
                raise ValueError(
                    "Source metadata changed while constructing the ledger"
                )
        checkpoint = (
            out / "products" / volume / relative.parent.name / (relative.stem + ".json")
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        write_once(checkpoint, record)
        write(
            out / "progress.json",
            {
                "status": "VERIFYING_LC2_INPUT_BYTES",
                "complete": number,
                "total": len(ordered),
                "identity": identity,
            },
        )
        print(
            json.dumps(
                {"complete": number, "total": len(ordered), "product": str(path)}
            ),
            flush=True,
        )
    if any(file_stamp(path) != stamp for path, stamp in stamps.items()):
        raise ValueError("A verified LC2 input changed before the completed freeze")
    return rows, ordered


def freeze(root, selection, out, expected=406547, resume=False):
    ids = selected_sources(selection, expected)
    identity = indexes(root, expected)
    identity[str(selection.resolve())] = digest(selection)
    identity[str(Path(__file__).resolve())] = digest(Path(__file__))
    groups, products = inventory(root, ids)
    out.mkdir(parents=True, exist_ok=resume)
    if resume and read(out / "progress.json")["identity"] != identity:
        raise ValueError("Input-ledger resume identity changed")
    write(
        out / "progress.json",
        {
            "status": "VERIFYING_LC2_INPUT_BYTES",
            "complete": 0,
            "total": len(products),
            "identity": identity,
        },
    )
    rows, ordered = verify_files(root, out, groups, products, identity)
    if set(rows) != set(ids):
        raise ValueError("Source ledger does not reconcile with the frozen selection")
    write_ledger(out / "sources.csv", ids, rows)
    write_once(out / "source-ids.json", ids)
    write_once(out / "products.json", ordered)
    if any(digest(Path(path)) != checksum for path, checksum in identity.items()):
        raise ValueError("Input-ledger identity changed before completion")
    result = {
        "schema": 1,
        "status": "FROZEN_LC2_INPUT_LEDGER",
        "sources": len(ids),
        "shards": len(groups),
        "products": len(ordered),
        "verified_input_bytes": sum(row["bytes"] for row in ordered),
        "inputs": identity,
        "artifacts": {
            name: digest(out / name)
            for name in ("sources.csv", "source-ids.json", "products.json")
        },
        "boi_catalogue_enabled": False,
        "scientific_implementation_frozen": False,
        "scientific_outcomes_opened": False,
        "original_raw_fits_bytes_rehashed": False,
        "boundary": (
            "Complete selected LC2 identities, source/origin/raw-FITS metadata "
            "and current bytes of all six indexed Parquet products per shard. "
            "Original archival FITS identities are retained from verified source "
            "metadata; their original FITS bytes are not read. This input freeze "
            "does not release a policy, freeze scientific implementation, or "
            "execute detection, vetting or scientific outcomes."
        ),
    }
    write_once(out / "freeze.json", result)
    write(
        out / "progress.json",
        {
            "status": "COMPLETE",
            "complete": len(ordered),
            "total": len(ordered),
            "identity": identity,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ids", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-sources", type=int, default=406547)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = dump_path(ROOT / args.out)
    try:
        result = freeze(
            ROOT,
            (ROOT / args.source_ids).resolve(),
            output,
            args.expected_sources,
            args.resume,
        )
    except Exception as error:
        if (output / "progress.json").exists() and not isinstance(
            error, FileExistsError
        ):
            progress = read(output / "progress.json")
            write(
                output / "progress.json",
                {
                    **progress,
                    "status": "FAILED",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "sources": result["sources"],
                "products": result["products"],
            }
        )
    )


if __name__ == "__main__":
    main()
