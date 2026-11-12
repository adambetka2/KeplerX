"""Rebuild an existing LC2 cohort from verified MAST originals."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.io import fits

TABLES = ("source", "cadence", "qdiag", "quarter", "header", "aperture")
STITCHED = {
    "time_bkjd": "TIME",
    "cadenceno": "CADENCENO",
    "quarter": "QUARTER",
    "psf_stitched_flux": "FLUX",
    "psf_stitched_flux_err": "FLUX_ERR",
    "sap_stitched_flux": "SAP_FLUX",
    "sap_stitched_flux_err": "SAP_FLUX_ERR",
    "psf_flat_kbonus_flux": "PSF_FLAT_FLUX",
    "psf_flat_kbonus_flux_err": "PSF_FLAT_FLUX_ERR",
    "sap_quality": "SAP_QUALITY",
    "flatten_mask": "FLATTEN_MASK",
}
DIAGNOSTIC = {
    "time_bkjd": "TIME",
    "cadenceno": "CADENCENO",
    "psf_flux": "FLUX",
    "psf_flux_err": "FLUX_ERR",
    "sap_flux": "SAP_FLUX",
    "sap_flux_err": "SAP_FLUX_ERR",
    "centroid_column": "CENTROID_COLUMN",
    "centroid_row": "CENTROID_ROW",
    "sap_quality": "SAP_QUALITY",
    "sap_background": "SAP_BKG",
    "psf_reduced_chi2": "RED_CHI2",
    "psf_flux_nova": "PSF_FLUX_NOVA",
    "psf_flux_err_nova": "PSF_FLUX_ERR_NOVA",
}
LOCAL = threading.local()
RESERVE = 20 * 1024**3


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def capacity(path, growth=0):
    if shutil.disk_usage(path).free < RESERVE + growth:
        raise OSError("Insufficient capacity to preserve the 20 GiB reserve")


def checked_raw(path, row):
    if (
        path.stat().st_size != row["raw_fits_bytes"]
        or digest(path) != row["raw_fits_sha256"]
    ):
        raise ValueError(f"Original FITS identity differs for {row['source_id']}")


def fetch(row, staging):
    path = staging / f"{row['source_id']}.fits"
    if path.exists():
        checked_raw(path, row)
        return path
    temporary = path.with_suffix(".part")
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(6):
        try:
            capacity(staging, row["raw_fits_bytes"])
            sha = hashlib.sha256()
            size = 0
            with LOCAL.session.get(
                row["product_url"], stream=True, timeout=(15, 120)
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        if size > row["raw_fits_bytes"]:
                            raise ValueError("MAST product exceeds its original size")
                        sha.update(block)
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if (
                size != row["raw_fits_bytes"]
                or sha.hexdigest() != row["raw_fits_sha256"]
            ):
                raise ValueError("MAST product differs from the local original")
            temporary.replace(path)
            path.chmod(0o444)
            return path
        except (requests.RequestException, OSError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == 5:
                raise RuntimeError(f"Download failed: {row['source_id']}") from error
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("Download exhausted retries")


def header_text(header):
    return header.tostring(sep="\n", endcard=True, padding=False)


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def scale(values):
    center = median(values)
    return (
        None if center is None else median(np.abs(np.asarray(values) - center)) * 1.4826
    )


def identifiers(header, quarter):
    return {
        name: scalar(header[name.upper()]) for name in ("module", "output", "channel")
    } | {"quarter": quarter}


def arrow_arrays(schema, arrays, sid, count):
    arrays = {"source_id": np.full(count, sid, dtype=np.int64), **arrays}
    return pa.Table.from_arrays(
        [pa.array(arrays[field.name], type=field.type) for field in schema],
        schema=schema,
    )


def quarter_summary(hdu, primary, sid):
    data = hdu.data
    quarter = int(hdu.name.split("Q")[-1])
    row = {"source_id": sid, **identifiers(hdu.header, quarter)}
    row.update(
        season=scalar(hdu.header["SEASON"]),
        origin_tpf_kic=int(primary["TPFORG"]),
        delivered_rows=len(data),
        cadenceno_min=int(np.min(data["CADENCENO"])),
        cadenceno_max=int(np.max(data["CADENCENO"])),
        finite_psf=int(np.isfinite(data["FLUX"]).sum()),
        finite_sap=int(np.isfinite(data["SAP_FLUX"]).sum()),
        quality_zero=int((data["SAP_QUALITY"] == 0).sum()),
        median_psf_flux=median(data["FLUX"]),
        robust_psf_scale=scale(data["FLUX"]),
        median_sap_flux=median(data["SAP_FLUX"]),
        robust_sap_scale=scale(data["SAP_FLUX"]),
        lightcurve_extname=hdu.name,
        aperture_extname=f"APERTURE_Q{quarter}",
    )
    for name, column in (
        ("median_psf_error", "FLUX_ERR"),
        ("median_sap_error", "SAP_FLUX_ERR"),
        ("median_centroid_column", "CENTROID_COLUMN"),
        ("median_centroid_row", "CENTROID_ROW"),
        ("median_sap_background", "SAP_BKG"),
        ("median_psf_reduced_chi2", "RED_CHI2"),
    ):
        row[name] = median(data[column])
    return row


def extract(path, row, schemas):
    sid = int(row["source_id"])
    with fits.open(path, memmap=True) as hdus:
        primary = hdus[0].header
        if (
            str(primary["GAIAID"]).split()[-1] != str(sid)
            or int(primary["TPFORG"]) != row["origin_tpf_kic"]
        ):
            raise ValueError(
                f"FITS source or extraction-parent identity differs: {sid}"
            )
        stitched = hdus["LIGHTCURVE_STITCHED"].data
        quarters = [hdu for hdu in hdus if hdu.name.startswith("LIGHTCURVE_Q")]
        counts = {
            "stitched_rows": len(stitched),
            "quarter_count": len(quarters),
            "finite_psf_stitched": int(np.isfinite(stitched["FLUX"]).sum()),
            "finite_sap_stitched": int(np.isfinite(stitched["SAP_FLUX"]).sum()),
            "finite_psf_flat_kbonus": int(np.isfinite(stitched["PSF_FLAT_FLUX"]).sum()),
            "quality_zero_rows": int((stitched["SAP_QUALITY"] == 0).sum()),
            "flatten_mask_zero_rows": int((stitched["FLATTEN_MASK"] == 0).sum()),
        }
        if any(row[name] != value for name, value in counts.items()):
            raise ValueError(
                f"Original cadence/quarter inventory differs: {sid}: {counts}"
            )
        metadata = {field.name: row.get(field.name) for field in schemas["source"]}
        metadata["primary_header_cards"] = header_text(primary)
        tables = {"source": pa.Table.from_pylist([metadata], schema=schemas["source"])}
        arrays = {
            name: np.asarray(stitched[column]).astype(
                schemas["cadence"].field(name).type.to_pandas_dtype()
            )
            for name, column in STITCHED.items()
        }
        tables["cadence"] = arrow_arrays(schemas["cadence"], arrays, sid, len(stitched))
        qtables = []
        for hdu in quarters:
            quarter = int(hdu.name.split("Q")[-1])
            arrays = {
                name: np.asarray(hdu.data[column]).astype(
                    schemas["qdiag"].field(name).type.to_pandas_dtype()
                )
                for name, column in DIAGNOSTIC.items()
            }
            arrays.update(
                {
                    name: np.full(len(hdu.data), value)
                    for name, value in identifiers(hdu.header, quarter).items()
                }
            )
            qtables.append(arrow_arrays(schemas["qdiag"], arrays, sid, len(hdu.data)))
        tables["qdiag"] = pa.concat_tables(qtables)
        tables["quarter"] = pa.Table.from_pylist(
            [quarter_summary(hdu, primary, sid) for hdu in quarters],
            schema=schemas["quarter"],
        )
        headers, apertures = [], []
        for number, hdu in enumerate(hdus):
            quarter = (
                int(hdu.name.split("Q")[-1])
                if hdu.name.startswith(("APERTURE_Q", "LIGHTCURVE_Q"))
                else None
            )
            headers.append(
                dict(
                    source_id=sid,
                    hdu_index=number,
                    extname=hdu.name,
                    quarter=hdu.header.get("QUARTER"),
                    header_cards=header_text(hdu.header),
                )
            )
            if hdu.name.startswith("APERTURE_Q"):
                info = identifiers(hdus[f"LIGHTCURVE_Q{quarter}"].header, quarter)
                height, width = hdu.data.shape
                apertures.append(
                    dict(
                        source_id=sid,
                        **info,
                        height=height,
                        width=width,
                        mask_uint8_c_order=np.asarray(hdu.data, dtype=np.uint8).tobytes(
                            order="C"
                        ),
                        header_cards=header_text(hdu.header),
                    )
                )
        tables["header"] = pa.Table.from_pylist(headers, schema=schemas["header"])
        tables["aperture"] = pa.Table.from_pylist(apertures, schema=schemas["aperture"])
    return tables


def fingerprint(table):
    sha = hashlib.sha256()
    for field in table.schema:
        column = table[field.name].combine_chunks()
        sha.update(column.is_null().to_numpy(zero_copy_only=False).tobytes())
        if pa.types.is_string(field.type) or pa.types.is_binary(field.type):
            for value in column.to_pylist():
                block = value.encode() if isinstance(value, str) else value or b""
                sha.update(len(block).to_bytes(8, "little"))
                sha.update(block)
        else:
            values = column.fill_null(0).to_numpy(zero_copy_only=False)
            sha.update(values.tobytes())
    return sha.hexdigest()


def check_tables(folder, schemas, expected, fingerprints):
    for name in TABLES:
        parquet = pq.ParquetFile(folder / f"{name}.parquet")
        if not parquet.schema_arrow.equals(
            schemas[name]
        ) or parquet.num_row_groups != len(expected):
            raise ValueError(f"Schema or source row groups differ: {name}")
        sid_column = schemas[name].get_field_index("source_id")
        for number, row in enumerate(expected):
            group = parquet.metadata.row_group(number)
            stat = group.column(sid_column).statistics
            if (
                fingerprint(parquet.read_row_group(number))
                != fingerprints[number][name]
            ):
                raise ValueError(f"Parquet readback differs from FITS: {name}")
            if name == "aperture" and group.num_rows == 0:
                continue
            if (
                stat is None
                or stat.min != row["source_id"]
                or stat.max != row["source_id"]
            ):
                raise ValueError(f"Source row-group identity differs: {name}")
            if name == "cadence" and group.num_rows != row["stitched_rows"]:
                raise ValueError("Stitched cadence count differs")


def completed(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for item in manifest["files"].values():
        path = folder / item["path"]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Published LC2 file changed: {path}")
    return manifest


def parquet_writer(path, schema, name):
    encoding = None
    if name in {"cadence", "qdiag"}:
        encoding = {
            field.name: "BYTE_STREAM_SPLIT"
            for field in schema
            if pa.types.is_floating(field.type)
        }
        encoding["cadenceno"] = "DELTA_BINARY_PACKED"
    return pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=encoding is None,
        column_encoding=encoding,
    )


def build_shard(shard, args):
    cohort = Path(args.metadata)
    source_path = cohort / f"shard-{shard:04d}.json.gz"
    with gzip.open(source_path, "rt") as stream:
        rows = json.load(stream)
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    parent = Path(args.data) / volume / "data/kbonus/lc2"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / f"shard-{shard:04d}"
    if final.exists():
        manifest = completed(final)
        if manifest["cohort_metadata_sha256"] != digest(source_path) or manifest[
            "sources"
        ] != len(rows):
            raise ValueError("Completed shard belongs to another source cohort")
        return shard, rows, manifest
    schemas = {
        name: pa.ipc.read_schema(
            pa.BufferReader((cohort / f"{name}.arrow").read_bytes())
        )
        for name in TABLES
    }
    staging = Path(args.staging) / f"shard-{shard:04d}"
    staging.mkdir(parents=True, exist_ok=True)
    partial = parent / f".shard-{shard:04d}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    writers = {
        name: parquet_writer(partial / f"{name}.parquet", schemas[name], name)
        for name in TABLES
    }
    started = time.monotonic()
    fingerprints = []
    try:
        with ThreadPoolExecutor(args.download_workers) as pool:
            downloads = pool.map(lambda row: fetch(row, staging), rows)
            for number, (row, path) in enumerate(zip(rows, downloads), 1):
                capacity(parent)
                tables = extract(path, row, schemas)
                fingerprints.append(
                    {name: fingerprint(table) for name, table in tables.items()}
                )
                for name, table in tables.items():
                    writers[name].write_table(table, row_group_size=max(1, len(table)))
                if number % 100 == 0:
                    print(
                        json.dumps(
                            dict(
                                shard=shard,
                                converted=number,
                                total=len(rows),
                                elapsed_s=round(time.monotonic() - started),
                            )
                        ),
                        flush=True,
                    )
    finally:
        for writer in writers.values():
            writer.close()
    check_tables(partial, schemas, rows, fingerprints)
    files = {}
    for name in TABLES:
        path = partial / f"{name}.parquet"
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        files[name] = dict(
            path=path.name,
            bytes=path.stat().st_size,
            rows=pq.ParquetFile(path).metadata.num_rows,
            sha256=digest(path),
        )
        path.chmod(0o444)
    manifest = dict(
        format="lc2",
        shard=shard,
        sources=len(rows),
        files=files,
        cohort_metadata_sha256=digest(source_path),
        raw_bytes=sum(row["raw_fits_bytes"] for row in rows),
        origin="MAST KBONUS-BKG v1.0",
        elapsed_s=round(time.monotonic() - started, 3),
    )
    save(partial / "manifest.json", manifest)
    partial.rename(final)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    completed(final)
    for row in rows:
        (staging / f"{row['source_id']}.fits").unlink(missing_ok=True)
    return shard, rows, manifest


def database(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS files(
            id INTEGER PRIMARY KEY, volume TEXT, path TEXT, collection TEXT,
            format TEXT, bytes INTEGER, sha256 TEXT, status TEXT,
            kic INTEGER, gaia_source_id TEXT, UNIQUE(volume,path));
        CREATE TABLE IF NOT EXISTS sources(
            gaia_source_id TEXT PRIMARY KEY,kic INTEGER,volume TEXT,shard INTEGER);
        CREATE TABLE IF NOT EXISTS memberships(
            collection TEXT,file_id INTEGER,kic INTEGER,
            PRIMARY KEY(collection,file_id));
    """)
    return connection


def publish_index(connection, result):
    shard, rows, manifest = result
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO sources VALUES(?,?,?,?)",
            [(str(row["source_id"]), row["kic"], volume, shard) for row in rows],
        )
        for name, item in manifest["files"].items():
            path = f"kbonus/lc2/shard-{shard:04d}/{item['path']}"
            connection.execute(
                "INSERT OR REPLACE INTO files"
                "(volume,path,collection,format,bytes,sha256,status)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    volume,
                    path,
                    "kbonus",
                    "parquet",
                    item["bytes"],
                    item["sha256"],
                    "available",
                ),
            )


def status(connection, args, cohort, started, errors):
    count = connection.execute("SELECT count(*) FROM sources").fetchone()[0]
    shards = connection.execute("SELECT count(DISTINCT shard) FROM sources").fetchone()[
        0
    ]
    raw_bytes = 0
    for path in Path(args.data).glob("disc*/data/kbonus/lc2/shard-*/manifest.json"):
        raw_bytes += json.loads(path.read_text())["raw_bytes"]
    state = dict(
        schema=1,
        state="complete"
        if count == cohort["sources"] and not errors
        else "downloading",
        sources=count,
        target_sources=cohort["sources"],
        shards=shards,
        target_shards=cohort["shards"],
        raw_bytes_verified=raw_bytes,
        raw_bytes_total=cohort["raw_bytes"],
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        invocation_elapsed_s=round(time.monotonic() - started),
        errors=errors,
    )
    save(Path(args.metadata).parent / "status.json", state)
    for volume in ("discSSD", "discPCIe"):
        save(
            Path(args.data) / volume / "data/index.json",
            dict(
                schema=1,
                lookup="index.sqlite",
                paths_relative_to="data/",
                lookup_sha256=digest(Path(args.data) / "discSSD/data/index.sqlite"),
                primary_collections=dict(
                    kbonus=dict(
                        preferred_format="lc2",
                        sources=count,
                        shards=shards,
                        target_sources=cohort["sources"],
                        shard_assignment="even:discSSD; odd:discPCIe",
                    )
                ),
                ingestion_state=state["state"],
                cohort_sha256=digest(Path(args.metadata) / "cohort.json"),
            ),
        )
    print(json.dumps(state), flush=True)


def validate_metadata(args):
    cohort = json.loads((Path(args.metadata) / "cohort.json").read_text())
    if cohort["sources"] != 406547 or cohort["shards"] != 814:
        raise ValueError("Expected the existing 406,547-source, 814-shard local cohort")
    seen = set()
    total_bytes = 0
    for shard in range(cohort["shards"]):
        path = Path(args.metadata) / f"shard-{shard:04d}.json.gz"
        if digest(path) != cohort["files"][path.name]:
            raise ValueError("Cohort metadata identity changed")
        with gzip.open(path, "rt") as stream:
            rows = json.load(stream)
        for row in rows:
            if (
                row["source_id"] in seen
                or row["shard"] != shard
                or row["source_ordinal"] != len(seen)
            ):
                raise ValueError("Cohort source membership or ordering differs")
            if not row["product_url"].startswith(
                "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AHLSP%2Fkbonus-bkg%2Flcs%2F"
            ):
                raise ValueError(
                    "Source is outside the declared MAST KBONUS-BKG collection"
                )
            seen.add(row["source_id"])
            total_bytes += row["raw_fits_bytes"]
    if len(seen) != cohort["sources"] or total_bytes != cohort["raw_bytes"]:
        raise ValueError("Cohort totals differ")
    return cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--shard-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--shards", type=int, nargs="+")
    args = parser.parse_args()
    if args.shard_workers < 1 or args.download_workers < 1:
        parser.error("Worker counts must be positive")
    Path(args.staging).mkdir(parents=True, exist_ok=True)
    with (Path(args.metadata).parent / "ingest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cohort = validate_metadata(args)
        index = Path(args.data) / "discSSD/data/index.sqlite"
        index.parent.mkdir(parents=True, exist_ok=True)
        other = Path(args.data) / "discPCIe/data/index.sqlite"
        other.parent.mkdir(parents=True, exist_ok=True)
        if not other.exists():
            other.symlink_to(index)
        started = time.monotonic()
        errors = []
        with database(index) as connection:
            status(connection, args, cohort, started, errors)
            with ProcessPoolExecutor(args.shard_workers) as pool:
                futures = {
                    pool.submit(build_shard, shard, args): shard
                    for shard in (args.shards or range(cohort["shards"]))
                }
                for future in as_completed(futures):
                    try:
                        publish_index(connection, future.result())
                    except Exception as error:
                        errors.append(dict(shard=futures[future], error=repr(error)))
                    status(connection, args, cohort, started, errors)
        if errors:
            raise RuntimeError(f"Unfinished shards: {errors}")


if __name__ == "__main__":
    main()
