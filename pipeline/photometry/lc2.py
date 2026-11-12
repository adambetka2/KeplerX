"""Read complete LC2 sources and their extraction views from the mounted corpus."""

import hashlib
import json
import math
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CADENCE = 1765.464 / 86400
QUALITY_BITMASK = 1130799
VIEWS = {
    "psf": "psf_stitched",
    "sap": "sap_stitched",
    "psf_flat": "psf_flat_kbonus",
    "psf_qdiag": "psf",
    "sap_qdiag": "sap",
    "psf_nova": "psf_nova",
}
QDIAG_FLUX = {
    "psf_qdiag": ("qdiag_psf_flux", "qdiag_psf_flux_err"),
    "sap_qdiag": ("qdiag_sap_flux", "qdiag_sap_flux_err"),
    "psf_nova": ("qdiag_psf_flux_nova", "qdiag_psf_flux_err_nova"),
}
DELTA_FLUX = {
    "psf_flat": "psf_flat_kbonus_flux",
    "psf_qdiag": "qdiag_psf_flux",
    "sap_qdiag": "qdiag_sap_flux",
    "psf_nova": "qdiag_psf_flux_nova",
}
LC2_MANIFEST = "hash-bound-lc2-base-v1"
KEYS = ["time", "cadence", "quarter", "flux", "error", "quality", "flatten"]
COLUMNS = {
    "time": "time_bkjd",
    "cadence": "cadenceno",
    "quarter": "quarter",
    "flux": "psf_stitched_flux",
    "error": "psf_stitched_flux_err",
    "quality": "sap_quality",
}


def curve_digest(curve):
    digest = hashlib.sha256()
    for key in COLUMNS:
        array = np.asarray(curve[key])
        digest.update(json.dumps([key, str(array.dtype), array.shape]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def verify_views(proposal, raw):
    for view in proposal["views"]:
        measured = curve_digest(photometry(raw, view["view"]))
        if measured != view["input_arrays_sha256"]:
            raise ValueError(
                "Discovery and downstream input arrays differ: "
                f"{proposal['source_id']} {view['view']}"
            )


def valid_points(data, quality_bitmask=QUALITY_BITMASK):
    return (
        np.isfinite(data["time"])
        & np.isfinite(data["flux"])
        & np.isfinite(data["error"])
        & (data["error"] > 0)
        & ((data["quality"].astype(np.int64) & quality_bitmask) == 0)
    )


def vetting_points(data, quality_bitmask, mask_flatten=True):
    good = valid_points(data, quality_bitmask)
    return good & (data["flatten"] == 0) if mask_flatten else good


def uses_flatten_mask(view, config):
    return "flatten_mask_views" not in config or view in config["flatten_mask_views"]


@lru_cache(maxsize=6)
def source_row_groups(path):
    """Map each source to its row groups; LC2 shard files hold one source per group."""
    parquet = pq.ParquetFile(path)
    column = parquet.schema_arrow.get_field_index("source_id")
    groups = {}
    for i in range(parquet.num_row_groups):
        stat = parquet.metadata.row_group(i).column(column).statistics
        if stat is None or stat.min != stat.max:
            raise ValueError(f"Expected source-specific LC2 row groups: {path}")
        groups.setdefault(int(stat.min), []).append(i)
    return parquet, groups


def read_source_rows(path, sid, columns=None):
    parquet, groups = source_row_groups(str(path))
    table = parquet.read_row_groups(groups[sid], columns=columns)
    if not len(table) or not np.all(table["source_id"].to_numpy() == sid):
        raise ValueError(f"Missing or mixed source rows: {sid} in {path}")
    return table


def source_row(folder, sid):
    table = read_source_rows(Path(folder) / "source.parquet", sid)
    if len(table) != 1:
        raise ValueError(f"Expected one LC2 metadata row: {sid} in {folder}")
    return table.to_pylist()[0]


def read_source(folder, source_id):
    folder = Path(folder)
    sid = int(source_id)
    table = read_source_rows(folder / "cadence.parquet", sid)
    diagnostics = read_source_rows(folder / "qdiag.parquet", sid)
    raw = {name: table[name].to_numpy() for name in table.column_names}
    raw.update(
        {
            f"qdiag_{name}": diagnostics[name].to_numpy()
            for name in diagnostics.column_names
            if name != "source_id"
        }
    )
    return dict(
        source_row(folder, sid),
        source_id=str(sid),
        shard_path=str(folder),
        shard=int(folder.name.rsplit("-", 1)[-1]),
    ), raw


def file_stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=4)
def read_manifest(path, stamp):
    manifest = json.loads(Path(path).read_text())
    if file_stamp(path) != stamp:
        raise ValueError(f"Input manifest changed while reading: {path}")
    return manifest


def input_manifest(root, path):
    path = (Path(root) / path).resolve()
    manifest = read_manifest(str(path), file_stamp(path))
    if manifest.get("schema") not in {1, 2, 3} or not manifest.get("measurement_scope"):
        raise ValueError("An input manifest must declare schema and measurement scope")
    if "native_control_binding" in manifest:
        from pipeline.deterministic.native_prospective_controls import runtime_binding

        runtime_binding(manifest)
    return path, manifest


def finite_metadata(metadata):
    result = {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in metadata.items()
    }
    result["shard_path"] = str(Path(result["shard_path"]).resolve())
    return result


@lru_cache(maxsize=4096)
def verified_lc2_file(path, stamp, expected):
    with Path(path).open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != expected or file_stamp(path) != stamp:
        raise ValueError(f"Bound LC2 product changed: {path}")


def lc2_manifest_files(root, manifest, item):
    folder = item["lc2_folder"]
    if manifest.get("protocol") != LC2_MANIFEST or not re.fullmatch(
        r"disc(?:SSD|PCIe)/data/kbonus/lc2/shard-[0-9]{4}", folder
    ):
        raise ValueError("Invalid hash-bound LC2 source storage")
    products = manifest["lc2_products"][folder]
    if set(products) != {"source", "cadence", "qdiag"}:
        raise ValueError("Bound LC2 source requires all three reader products")
    paths = []
    for name, record in products.items():
        path = Path(root) / folder / f"{name}.parquet"
        stamp = file_stamp(path)
        if stamp[2] != record["bytes"]:
            raise ValueError(f"Bound LC2 product size changed: {path}")
        verified_lc2_file(str(path), stamp, record["sha256"])
        paths.append(path)
    return paths


def lc2_manifest_source(root, source_id, manifest, item):
    lc2_manifest_files(root, manifest, item)
    metadata, raw = read_source(Path(root) / item["lc2_folder"], source_id)
    metadata = finite_metadata(metadata)
    if metadata != item["metadata"]:
        raise ValueError(f"Bound LC2 source metadata changed: {source_id}")
    return metadata, raw


def delta_path(path, item):
    path = Path(path).parent / item["deltas"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Experimental delta changed: {path}")
    return path


def manifest_source(root, source_id, path):
    path, manifest = input_manifest(root, path)
    if "native_control_binding" in manifest:
        from pipeline.deterministic.native_prospective_controls import runtime_source

        runtime_source(manifest, str(source_id))
    item = manifest["sources"][str(source_id)]
    if manifest["schema"] == 3:
        metadata, raw = lc2_manifest_source(root, source_id, manifest, item)
        return dict(metadata, input_manifest=str(path.resolve())), raw
    if manifest["schema"] == 2:
        if manifest.get("transform") != "additive_flux_delta":
            raise ValueError("Unknown experimental input transform")
        metadata, raw = load_source(root, source_id)
        if metadata["raw_fits_sha256"] != item["base_raw_fits_sha256"]:
            raise ValueError(f"Experimental base input changed: {source_id}")
        transformed = delta_path(path, item)
        with np.load(transformed, allow_pickle=False) as data:
            if not data.files or not set(data.files) <= set(DELTA_FLUX.values()):
                raise ValueError("Experimental delta contains unknown arrays")
            raw = dict(raw)
            for key in data.files:
                delta = data[key]
                if delta.shape != raw[key].shape or not np.isfinite(delta).all():
                    raise ValueError("Experimental delta shape or values are invalid")
                raw[key] = np.asarray(raw[key], dtype=np.float64) + delta
        return dict(metadata, input_manifest=str(path.resolve())), raw
    return array_manifest_source(path, item, source_id)


def array_manifest_source(path, item, source_id):
    array_path = path.parent / item["arrays"]
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Experimental input changed: {array_path}")
    with np.load(array_path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    lengths = {
        "primary": len(raw["time_bkjd"]),
        "qdiag": len(raw["qdiag_time_bkjd"]) if "qdiag_time_bkjd" in raw else 0,
    }
    if any(
        len(value) != lengths["qdiag" if key.startswith("qdiag_") else "primary"]
        for key, value in raw.items()
    ):
        raise ValueError("Experimental arrays have inconsistent row counts")
    if str(item["metadata"]["source_id"]) != str(source_id):
        raise ValueError("Experimental source identity mismatch")
    if "source_id" in raw and not np.all(
        raw["source_id"].astype(str) == str(source_id)
    ):
        raise ValueError("Experimental arrays contain another source identity")
    metadata = dict(item["metadata"], input_manifest=str(path.resolve()))
    return metadata, raw


def load_source(root, source_id, manifest=None):
    if manifest is not None:
        return manifest_source(root, source_id, manifest)
    index = Path(root) / "discSSD/data/index.sqlite"
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
        row = database.execute(
            "select volume, shard from sources where gaia_source_id = ?",
            (str(source_id),),
        ).fetchone()
    if row is None:
        raise ValueError(f"Source absent from LC2 index: {source_id}")
    volume, shard = row
    return read_source(
        Path(root) / volume / f"data/kbonus/lc2/shard-{shard:04d}", source_id
    )


def source_catalogue(root, shards, columns=None):
    for shard in shards:
        volume = "discSSD" if shard % 2 == 0 else "discPCIe"
        folder = Path(root) / volume / f"data/kbonus/lc2/shard-{shard:04d}"
        parquet = pq.ParquetFile(folder / "source.parquet")
        rows = parquet.read(columns=columns).to_pylist()
        for row in {int(row["source_id"]): row for row in rows}.values():
            yield dict(
                row,
                source_id=str(row["source_id"]),
                shard_path=str(folder),
                shard=shard,
            )


def photometry(raw, view):
    if view in {"psf_qdiag", "sap_qdiag", "psf_nova"}:
        flux, error = QDIAG_FLUX[view]
        time = raw["qdiag_time_bkjd"]
        return dict(
            zip(
                KEYS,
                [
                    time,
                    raw["qdiag_cadenceno"],
                    raw["qdiag_quarter"],
                    raw[flux],
                    raw[error],
                    raw["qdiag_sap_quality"],
                    np.isin(
                        raw["qdiag_cadenceno"],
                        raw["cadenceno"][raw["flatten_mask"] != 0],
                    ).astype(np.int32),
                ],
            )
        )
    prefix = VIEWS[view]
    return dict(
        zip(
            KEYS,
            [
                raw[name]
                for name in [
                    "time_bkjd",
                    "cadenceno",
                    "quarter",
                    f"{prefix}_flux",
                    f"{prefix}_flux_err",
                    "sap_quality",
                    "flatten_mask",
                ]
            ],
        )
    )


def load_curve(root, source_id, view="psf", manifest=None):
    if manifest is not None:
        path, specification = input_manifest(root, manifest)
        if specification["schema"] == 2:
            if specification.get("transform") != "additive_flux_delta":
                raise ValueError("Unknown experimental input transform")
            item = specification["sources"][str(source_id)]
            curve = load_curve(root, source_id, view)
            if curve["provenance"]["raw_fits_sha256"] != item["base_raw_fits_sha256"]:
                raise ValueError(f"Experimental base input changed: {source_id}")
            transformed = delta_path(path, item)
            with np.load(transformed, allow_pickle=False) as data:
                if not data.files or not set(data.files) <= set(DELTA_FLUX.values()):
                    raise ValueError("Experimental delta contains unknown arrays")
                key = DELTA_FLUX.get(view)
                if key in data:
                    delta = data[key]
                    if (
                        delta.shape != curve["flux"].shape
                        or not np.isfinite(delta).all()
                    ):
                        raise ValueError(
                            "Experimental delta shape or values are invalid"
                        )
                    curve["flux"] = np.asarray(curve["flux"], dtype=np.float64) + delta
            curve["provenance"]["input_manifest"] = str(path.resolve())
            return curve
        source, raw = load_source(root, source_id, manifest)
        curve = photometry(raw, view)
        curve["provenance"] = {
            "source_id": str(source_id),
            "view": view,
            "parent_tpf": str(source["origin_tpf_kic"]),
            "raw_fits_sha256": source["raw_fits_sha256"],
            "input_manifest": source["input_manifest"],
            "stored_rows": len(curve["time"]),
        }
        return curve
    if view in {"psf_qdiag", "sap_qdiag", "psf_nova"}:
        source, raw = load_source(root, source_id)
        curve = photometry(raw, view)
        curve["provenance"] = {
            "source_id": str(source_id),
            "view": view,
            "parent_tpf": str(source["origin_tpf_kic"]),
            "raw_fits_sha256": source["raw_fits_sha256"],
            "shard_path": str(source["shard_path"]),
            "stored_rows": len(curve["time"]),
        }
        return curve
    prefix = VIEWS[view]
    index = Path(root) / "discSSD/data/index.sqlite"
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
        row = database.execute(
            "SELECT volume, shard FROM sources WHERE gaia_source_id = ?",
            (str(source_id),),
        ).fetchone()
    if row is None:
        raise ValueError(f"Source absent from LC2 index: {source_id}")
    volume, shard = row
    folder = Path(root) / volume / f"data/kbonus/lc2/shard-{shard:04d}"
    sid = int(source_id)
    columns = dict(
        COLUMNS,
        flux=f"{prefix}_flux",
        error=f"{prefix}_flux_err",
        flatten="flatten_mask",
    )
    table = read_source_rows(
        folder / "cadence.parquet", sid, ["source_id", *columns.values()]
    )
    source = source_row(folder, sid)
    curve = {key: table[column].to_numpy() for key, column in columns.items()}
    curve["provenance"] = {
        "source_id": str(sid),
        "view": view,
        "parent_tpf": str(source["origin_tpf_kic"]),
        "raw_fits_sha256": source["raw_fits_sha256"],
        "shard_path": str(folder),
        "stored_rows": len(table),
    }
    return curve


def load_kepler_curve(path, source_id, parent_tpf, view):
    from astropy.io import fits

    columns = dict(COLUMNS)
    columns.update(
        time="TIME", cadence="CADENCENO", quarter="QUARTER", quality="SAP_QUALITY"
    )
    columns["flux"] = "FLUX" if view == "psf" else "SAP_FLUX"
    columns["error"] = "FLUX_ERR" if view == "psf" else "SAP_FLUX_ERR"
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header
        if str(header["GAIAID"]).split()[-1] != str(source_id):
            raise ValueError(f"Confirmed-host FITS identity mismatch: {path}")
        if int(header["TPFORG"]) != int(parent_tpf):
            raise ValueError(f"Confirmed-host originating TPF mismatch: {path}")
        table = hdus["LIGHTCURVE_STITCHED"].data
        curve = {key: np.array(table[column]) for key, column in columns.items()}
    curve["provenance"] = {
        "source_id": str(source_id),
        "parent_tpf": str(parent_tpf),
        "view": view,
        "fits_path": str(path),
        "stored_rows": len(table),
    }
    return curve
