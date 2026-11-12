"""Fetch a bounded, frozen opened-development cohort directly from MAST."""

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic.catalogue_controls import (
    QUARTER_COLUMNS,
    STITCHED_COLUMNS,
    product_uri,
    read_product,
)
from deterministic.run import DISCOVERY_VIEW, prepare_records
from lc2 import DELTA_FLUX, curve_digest, photometry
from runtime import EXTERNAL_DUMPS, dump_path

SPECIFICATION = Path(__file__).with_name("cloud-public-inputs.json")
CONTROL_SPECIFICATION = Path(__file__).with_name("cloud-public-controls.json")
CONTROL_DELTAS = Path(__file__).with_name("cloud-control-deltas.tar.gz")
DUMP = ROOT / "dump"
LIMIT = 512 * 1024**2
SCOPE = "OPENED_DEVELOPMENT_ONLY"
CONTROL_COHORTS = {
    "native_eb": ("catalogue", "CURRENT_OPENED_DEVELOPMENT"),
    "native_confirmed_extra": (
        "catalogue",
        "DEVELOPMENT_EXTRA_HISTORICALLY_OPENED_CONFIRMED_HOST",
    ),
    "periodic_planet": ("lc2", "CURRENT_OPENED_DEVELOPMENT"),
    "shallow_planet_extra": ("lc2", "DEVELOPMENT_EXTRA_OPENED_SHALLOW_PILOT"),
}
DIAGNOSTIC_COLUMNS = QUARTER_COLUMNS | {
    "qdiag_centroid_column": "CENTROID_COLUMN",
    "qdiag_centroid_row": "CENTROID_ROW",
    "qdiag_sap_background": "SAP_BKG",
    "qdiag_psf_reduced_chi2": "RED_CHI2",
}
INTEGER_TYPES = {
    "source_id": np.int64,
    "quarter": np.int16,
    "cadenceno": np.int32,
    "sap_quality": np.int32,
    "flatten_mask": np.int32,
    "qdiag_quarter": np.int16,
    "qdiag_module": np.int16,
    "qdiag_output": np.int16,
    "qdiag_channel": np.int16,
    "qdiag_cadenceno": np.int32,
    "qdiag_sap_quality": np.int32,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Public manifest output is a symlink")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)


def declared_uri(row):
    if row["product_kic"]:
        return product_uri(row["product_kic"])
    sid = row["metadata"]["source_id"]
    return (
        f"mast:HLSP/kbonus-bkg/lcs/{sid[:4]}/{sid}/"
        f"hlsp_kbonus-bkg_kepler_kepler_gaia-dr3-{sid}_kepler_v1.0_lc.fits"
    )


def checked_url(row):
    url = row["url"]
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "mast.stsci.edu"
        or parsed.path != "/api/v0.1/Download/file"
        or parsed.fragment
        or parse_qs(parsed.query, strict_parsing=True, keep_blank_values=True)
        != {"uri": [declared_uri(row)]}
    ):
        raise ValueError(f"Noncanonical MAST product URL: {url}")
    return url


def checked_source(sid, row):
    metadata = row["metadata"]
    if (
        not re.fullmatch("[1-9][0-9]*", sid)
        or metadata["source_id"] != sid
        or metadata["raw_fits_sha256"] != row["sha256"]
        or metadata["kic"] != row["product_kic"]
        or type(row["product_kic"]) is not int
        or row["product_kic"] < 0
        or type(row["bytes"]) is not int
        or not 0 < row["bytes"] <= LIMIT
        or any(
            not sha256(row[name])
            for name in ("sha256", "raw_view_sha256", "prepared_records_sha256")
        )
    ):
        raise ValueError(f"Invalid frozen source identity: {sid}")
    checked_url(row)


def specification(path=None):
    data = json.loads((SPECIFICATION if path is None else path).read_text())
    if (
        type(data.get("schema")) is not int
        or data["schema"] != 1
        or data.get("scope") != SCOPE
        or type(data.get("max_download_bytes")) is not int
        or not 0 < data["max_download_bytes"] <= LIMIT
        or not isinstance(data.get("sources"), dict)
        or not data["sources"]
    ):
        raise ValueError("Invalid opened-development public input specification")
    for sid, row in data["sources"].items():
        checked_source(sid, row)
    if (
        sum(row["bytes"] for row in data["sources"].values())
        > data["max_download_bytes"]
    ):
        raise ValueError("Frozen public downloads exceed their byte budget")
    parents = [row["metadata"]["origin_tpf_kic"] for row in data["sources"].values()]
    if any(type(parent) is not int or parent <= 0 for parent in parents) or len(
        set(parents)
    ) != len(parents):
        raise ValueError("Public development sources must have distinct valid parents")
    return data


def control_specification():
    data = specification(CONTROL_SPECIFICATION)
    if (
        data.get("status") != "BOUNDED_OPENED_CONTROL_INPUTS"
        or data.get("automatic_pc_enabled") is not False
        or data.get("joint_independence_complete") is not False
    ):
        raise ValueError("Control inputs must retain their development boundary")
    files = {}
    for sid, row in data["sources"].items():
        contract = CONTROL_COHORTS.get(row.get("opened_cohort"))
        if contract != (row.get("reader"), row.get("opened_scope")):
            raise ValueError("Control cohort, scope or reader changed")
        injected = row["reader"] == "lc2"
        if injected != ("delta" in row) or injected != ("base_arrays" in row):
            raise ValueError("Control injection binding is absent or unexpected")
        if injected:
            delta = row["delta"]
            if delta["path"] != f"{sid}.npz":
                raise ValueError("Control delta path differs from source identity")
            files[delta["path"]] = {key: delta[key] for key in ("bytes", "sha256")}
    if files != data["deltas"]["files"]:
        raise ValueError("Control delta inventory differs from injected sources")
    return data


def verify_inventory(arrays, expected):
    if set(arrays) != set(expected):
        raise ValueError("Control array inventory differs from original inputs")
    for name, array in arrays.items():
        identity = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes": array.nbytes,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
        if identity != expected[name]:
            raise ValueError(f"Control array dtype, shape or content differs: {name}")


def extract_deltas(out, files):
    seen = set()
    with tarfile.open(CONTROL_DELTAS, "r|gz") as bundle:
        for member in bundle:
            if (
                member.name not in files
                or member.name in seen
                or not member.isfile()
                or Path(member.name).name != member.name
                or member.size != files[member.name]["bytes"]
            ):
                raise ValueError("Control delta archive member differs from inventory")
            seen.add(member.name)
            with (
                bundle.extractfile(member) as source,
                (out / member.name).open("xb") as target,
            ):
                shutil.copyfileobj(source, target)
    if seen != set(files):
        raise ValueError("Control delta archive inventory is incomplete")


def control_deltas(data, folder):
    files = data["deltas"]["files"]
    identity = data["deltas"]["archive"]
    if (
        type(identity["bytes"]) is not int
        or not 0 < identity["bytes"] <= LIMIT
        or any(
            type(item["bytes"]) is not int or not 0 < item["bytes"] <= LIMIT
            for item in files.values()
        )
        or sum(item["bytes"] for item in files.values()) > LIMIT
    ):
        raise ValueError("Control delta archive exceeds its byte budget")
    checked_file(CONTROL_DELTAS, identity)
    out = folder / "deltas"
    if out.is_symlink():
        raise ValueError("Control delta output is a symlink")
    if not out.exists():
        checkpoint(folder, sum(item["bytes"] for item in files.values()))
        out.mkdir()
        extract_deltas(out, files)
    if {item.name for item in out.iterdir()} != set(files):
        raise ValueError("Materialized control delta inventory changed")
    for name, identity in files.items():
        checked_file(out / name, identity)
        (out / name).chmod(0o444)
    return out


def checkpoint(folder, growth=0):
    folder = dump_path(folder)
    external = any(root == folder or root in folder.parents for root in EXTERNAL_DUMPS)
    cloud = os.environ.get("SPACELAB_EXECUTION_PROFILE") == "cloud-development"
    reserve = LIMIT if cloud and not external else 20 * 1024**3
    if shutil.disk_usage(folder).free < reserve + growth:
        raise ValueError(
            f"Insufficient space for public inputs and {reserve} byte reserve"
        )


def checked_file(path, row):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
        raise ValueError(f"Missing or wrong-sized public input: {path}")
    if digest(path) != row["sha256"]:
        raise ValueError(f"Public input hash differs: {path}")


def download(path, row):
    if path.exists() or path.is_symlink():
        checked_file(path, row)
        return
    checkpoint(path.parent, row["bytes"])
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.is_symlink():
        raise ValueError(f"Public download partial is a symlink: {temporary}")
    with requests.get(
        checked_url(row), stream=True, timeout=(15, 120), allow_redirects=False
    ) as response:
        response.raise_for_status()
        if response.status_code != 200 or response.url != row["url"]:
            raise ValueError("MAST changed the declared product URL or response status")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != row["bytes"]:
            raise ValueError("MAST Content-Length differs from frozen product size")
        count = 0
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                count += len(block)
                if count > row["bytes"]:
                    raise ValueError("MAST stream exceeds frozen product size")
                checkpoint(path.parent, len(block))
                stream.write(block)
    checked_file(temporary, row)
    temporary.replace(path)


def verify_arrays(raw, row):
    actual = curve_digest(photometry(raw, DISCOVERY_VIEW))
    records = prepare_records(raw, DISCOVERY_VIEW)
    prepared = hashlib.sha256(records.tobytes()).hexdigest()
    if actual != row["raw_view_sha256"] or prepared != row["prepared_records_sha256"]:
        raise ValueError(
            "Public photometry differs from frozen ordinary-control arrays"
        )
    return len(records)


def fits_columns(table, columns):
    return {
        name: np.array(table[column], dtype=INTEGER_TYPES.get(name, np.float64))
        for name, column in columns.items()
    }


def quarter_arrays(hdu):
    quarter = int(hdu.name.rsplit("Q", 1)[-1])
    if int(hdu.header["QUARTER"]) != quarter:
        raise ValueError("FITS quarter identity differs between extension and header")
    arrays = fits_columns(hdu.data, DIAGNOSTIC_COLUMNS)
    for name in ("quarter", "module", "output", "channel"):
        key = f"qdiag_{name}"
        arrays[key] = np.full(
            len(hdu.data), int(hdu.header[name.upper()]), dtype=INTEGER_TYPES[key]
        )
    return arrays


def fits_arrays(path, sid, row):
    with fits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        same_kic = (
            int(header["KEPLERID"]) == row["product_kic"]
            if row["product_kic"]
            else header["KEPLERID"] == ""
        )
        if not same_kic or str(header["GAIAID"]).split()[-1] != sid:
            raise ValueError("Public FITS source identity differs from frozen metadata")
        raw = fits_columns(hdus["LIGHTCURVE_STITCHED"].data, STITCHED_COLUMNS)
        raw["source_id"] = np.full(len(raw["time_bkjd"]), int(sid), dtype=np.int64)
        quarters = sorted(
            (hdu for hdu in hdus if re.fullmatch("LIGHTCURVE_Q[0-9]+", hdu.name)),
            key=lambda hdu: int(hdu.name.rsplit("Q", 1)[-1]),
        )
        if not quarters or len({hdu.name for hdu in quarters}) != len(quarters):
            raise ValueError("Public FITS has absent or duplicated quarter extensions")
        arrays = [quarter_arrays(hdu) for hdu in quarters]
        raw.update(
            {
                name: np.concatenate([item[name] for item in arrays])
                for name in arrays[0]
            }
        )
    return raw


def control_arrays(path, sid, row, deltas):
    if row["reader"] == "catalogue":
        metadata, raw = read_product(path, row["product_kic"])
        if metadata != row["metadata"]:
            raise ValueError("Control FITS metadata differs from original source")
    else:
        raw = fits_arrays(path, sid, row)
    verify_inventory(raw, row.get("base_arrays", row["arrays"]))
    if "delta" in row:
        verify_arrays(raw, row["base_checks"])
        item = row["delta"]
        path = deltas / item["path"]
        checked_file(path, item)
        with np.load(path, allow_pickle=False) as saved:
            if len(saved.files) != len(set(saved.files)):
                raise ValueError("Control delta contains duplicated arrays")
            values = {name: saved[name] for name in saved.files}
        verify_inventory(values, item["arrays"])
        if not values or not set(values) <= set(DELTA_FLUX.values()):
            raise ValueError("Control delta contains unknown flux arrays")
        for name, delta in values.items():
            if delta.shape != raw[name].shape or not np.isfinite(delta).all():
                raise ValueError("Control delta shape or values are invalid")
            raw[name] = np.asarray(raw[name], dtype=np.float64) + delta
    verify_inventory(raw, row["arrays"])
    return raw


def materialize(path, sid, row, deltas=None):
    raw = (
        fits_arrays(path, sid, row)
        if deltas is None
        else control_arrays(path, sid, row, deltas)
    )
    cadences = verify_arrays(raw, row)
    output = path.parent / f"{sid}.npz"
    checkpoint(path.parent, sum(array.nbytes for array in raw.values()) + 1024**2)
    temporary = output.with_suffix(".npz.part")
    if output.is_symlink() or temporary.is_symlink():
        raise ValueError("Public array output is a symlink")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **raw)
    temporary.replace(output)
    return {
        "arrays": output.name,
        "sha256": digest(output),
        "metadata": row["metadata"],
    }, cadences


def control_receipt(data, out, sources, manifest, receipt):
    boundary = {
        "automatic_pc_enabled": False,
        "joint_independence_complete": False,
        "boundary": data["boundary"],
    }
    manifest.update(boundary)
    receipt.update(boundary)
    receipt["cohorts"] = {}
    for cohort in CONTROL_COHORTS:
        ids = [sid for sid, row in sources.items() if row["opened_cohort"] == cohort]
        path = out / f"source-ids-{cohort}.json"
        save(path, ids)
        receipt["cohorts"][cohort] = {
            "sources": len(ids),
            "source_ids": path.name,
            "sha256": digest(path),
        }


def fetch_public(section, controls=False):
    if section not in {"1st", "2nd"}:
        raise ValueError("Public input section must be 1st or 2nd")
    specification_path = CONTROL_SPECIFICATION if controls else SPECIFICATION
    specification_hash = digest(specification_path)
    data = control_specification() if controls else specification()
    out = DUMP / ("cloud-controls" if controls else "cloud-public")
    if out.is_symlink():
        raise ValueError("Public input output is a symlink")
    out = dump_path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "receipt.json"
    names = ["receipt.json", "input-manifest.json", "source-ids.json"]
    if controls:
        names += [f"source-ids-{cohort}.json" for cohort in CONTROL_COHORTS]
    if any((out / name).is_symlink() for name in names):
        raise ValueError("Public input receipt or manifest output is a symlink")
    for name in names:
        (out / name).unlink(missing_ok=True)
    checkpoint(out, sum(row["bytes"] for row in data["sources"].values()))
    deltas = control_deltas(data, out) if controls else None
    sources, records = {}, []
    for sid, row in data["sources"].items():
        path = out / f"{sid}.fits"
        download(path, row)
        sources[sid], count = materialize(path, sid, row, deltas)
        if controls:
            sources[sid].update(
                {key: row[key] for key in ("opened_cohort", "opened_scope", "origin")}
            )
        records.append(
            {
                "source_id": sid,
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "prepared_cadences": count,
            }
        )
    if digest(specification_path) != specification_hash:
        raise ValueError("Frozen public specification changed during materialization")
    manifest = {
        "schema": 1,
        "measurement_scope": SCOPE,
        "role": "development",
        "survey_context_complete": False,
        "sources": sources,
    }
    receipt = {
        "status": "PUBLIC_INPUTS_VERIFIED",
        "scope": SCOPE,
        "role": "development",
        "survey_context_complete": False,
        "section": section,
        "specification_sha256": specification_hash,
        "sources": records,
    }
    if controls:
        control_receipt(data, out, sources, manifest, receipt)
    save(out / "input-manifest.json", manifest)
    save(out / "source-ids.json", list(sources))
    receipt["manifest_sha256"] = digest(out / "input-manifest.json")
    receipt["source_ids_sha256"] = digest(out / "source-ids.json")
    save(receipt_path, receipt)
    return receipt


def fetch_controls(section):
    return fetch_public(section, controls=True)
