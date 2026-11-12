"""Replay the fixed opened controls from pinned public FITS and original deltas."""

import gzip
import hashlib
import io
import json
import math
import os
import re
import resource
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic import cloud_public as public
from deterministic.catalogue_controls import (
    QUARTER_COLUMNS,
    STITCHED_COLUMNS,
    read_product,
)
from lc2 import DELTA_FLUX
from runtime import EXTERNAL_DUMPS, dump_path

SPECIFICATION = Path(__file__).with_name("cloud-opened-inputs.json")
TRUTH_SPECIFICATION = Path(__file__).with_name("cloud-opened-truths.json")
TRUTH_SPECIFICATION_SHA256 = "c71702a1a0d938e9bffd6647aaa0f0b0be93ac691d47d759d31d36ece36a264e"
TRUTH_COHORTS = {"periodic_planet": ("planets", "periodic_injection", 100),
                 "sparse_injection": ("sparse", "sparse_injection", 40),
                 "ttv_injection": ("ttv", "ttv_injection", 40)}
TRUTH_FIELDS = {"category", "depth_ppm", "duration_hours", "epoch_bkjd", "expected_white_mes",
                "impact_parameter", "in_transit_cadences", "inclination_degrees", "observed_events",
                "observed_quarters", "period_days", "point_sigma_ppm", "pool_rank", "radius_ratio",
                "scaled_semimajor_axis", "shape_period_days", "source_id", "stellar_density_solar",
                "stratum", "ttv"}
TRUTH_CLOCKS = {"d9549237cbdd55a0950177705f383d331606fca37fd67575f0ba130b0a2f7aae":
                (31952, "106ee4d3fc76ba160e30896b4f396f3a668f3f32"),
                "cf931ff504da934534338083c5bd74f91cc37af542bb99de8a3889d6c6f30420":
                (32178, "ab81661a3b9c7dd97824d40f66746e786dfb3dcc")}
TRUTH_CLOCK_FUNCTIONS = {"event_centres": "a3e6cb35fbbcaac99448dbf809a8ac8177090dd45b9f2a87acde0863e2748809",
                         "truth_centres": "044740e304a5779bbf4a01fb8dbbf9cd7217c39d175f6691f9094cf6018fe70f"}
LIMIT = 8 * 1024**2
FLAGS = ("scientific_acceptance", "automatic_pc_enabled", "joint_independence_complete",
         "survey_context_complete", "pixel_readiness_complete", "partitions_frozen", "seeds_frozen")
TRUTH_PACKET_FIELDS = {"archive", *FLAGS, "boundary", "cohorts", "counts", "fixed_array_packet",
                       "historical_fixed_array_packet", "input_inventory", "members",
                       "new_sources_or_seeds_or_truths_generated", "original_clock_generation",
                       "original_opened876_selection_unchanged", "original_route_bindings",
                       "original_truth_bytes_preserved", "program", "protected_metadata_only_sources_excluded",
                       "role", "schema", "scope", "status", "units_and_convention"}
COHORTS = {"ordinary": ("lc2", False), "periodic_planet": ("lc2", True),
           "native_eb": ("catalogue", False), "sparse_injection": ("lc2", True),
           "ttv_injection": ("lc2", True)}
COUNTS = {"original": 876, "supported": 771, "protected": 5,
          "background_recipe_unavailable": 100,
          "cohorts": {"ordinary": 492, "periodic_planet": 100, "native_eb": 99,
                      "sparse_injection": 40, "ttv_injection": 40},
          "arrays": 21567, "delta_sources": 180, "delta_arrays": 720,
          "native_parent_qualified": 6, "local_exact": 108,
          "remote_canonical_proof_pending": 663, "local_injected_base_oracles": 5,
          "unavailable_local_injected_base_oracles": 175}
PREPARATION_PROFILES = [
    {"name": "original-main", "files": [{"path": "pipeline/deterministic/run.py", "bytes": 73979,
        "sha256": "3ffa249a51a277fdc9884960b48209ef691e7f20d5948478e978d66f4e113d3b"}]},
    {"name": "coordinated-default-preparation", "files": [
        {"path": "pipeline/deterministic/run.py", "bytes": 79856,
         "sha256": "6aaa279fb8ae3117fb9ba107de2c979178644cdbb0a669e6ef0f5e44b286d4f5"},
        {"path": "pipeline/deterministic/signal_contracts.py", "bytes": 20158,
         "sha256": "2328a24fbb48432da1d61a0d634883d8835af42ad7c5ce036a1126011b03f8be"}]},
]
SCOPE_METADATA = [
    {"path": "pipeline/deterministic/cloud-joint-scope.json", "bytes": 50483,
     "sha256": "c305e1ffefdbd8d815c702b9543792d00660bbd362cdc5af781e9b18de988b4c"},
    {"path": "pipeline/deterministic/cloud-joint-scope.json.gz", "bytes": 3271755,
     "sha256": "04a51858ddd4b6d891bc97bed3749e8aaaebb2c8729e599814ba41cf913b8df0"},
]
JOINT_SCOPE_DECODED_SHA256 = "aff2e0baea45bd89dd4e9b36eb93e4e441a3310ad96c15f2f7cbe6cd7ce1049f"
OUTPUT_NAMES = ("receipt.json", "input-manifest.json", "source-ids.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse(payload):
    return json.loads(payload, object_pairs_hook=unique_object)


def integer(value, minimum=0, maximum=32_000_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Invalid bounded integer in opened packet")
    return value


def parent_paths(path):
    for current in (path, *path.parents):
        if current == ROOT / "dump":
            break
        if current.is_symlink():
            raise ValueError(f"Opened path redirects through a symlink: {current}")


def checked_file(path, item):
    parent_paths(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError(f"Missing, redirected or wrong-sized opened input: {path}")
    if digest(path) != item["sha256"]:
        raise ValueError(f"Opened input bytes changed: {path}")


def boundary(value):
    if (value.get("scope") != public.SCOPE or value.get("role") != "development"
        or value.get("original_opened876_selection_unchanged") is not True
        or any(value.get(flag) is not False for flag in FLAGS)):
        raise ValueError("Opened packet must retain all development boundaries")


def artifact_path(specification, item):
    name = item["path"]
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Opened artifact name escapes its specification directory")
    return specification.parent / name


def decoded_metadata(specification, descriptor):
    item = descriptor["artifact"]
    integer(item["bytes"], 1, LIMIT)
    expected = integer(item["decoded_bytes"], 1, LIMIT)
    path = artifact_path(specification, item)
    checked_file(path, item)
    with gzip.open(path, "rb") as stream:
        payload = stream.read(expected + 1)
    if len(payload) != expected or hashlib.sha256(payload).hexdigest() != item["decoded_sha256"]:
        raise ValueError("Opened metadata decompression differs from pinned bytes")
    return parse(payload), path


def array_names(reader):
    names = set(STITCHED_COLUMNS) | set(QUARTER_COLUMNS) | {"qdiag_quarter"}
    if reader == "lc2":
        names |= set(public.DIAGNOSTIC_COLUMNS) | {"source_id", "qdiag_module", "qdiag_output", "qdiag_channel"}
    return names


def array_descriptor(item, delta=False):
    if set(item) != {"dtype", "shape", "bytes", "sha256"} or not public.sha256(item["sha256"]):
        raise ValueError("Array descriptor fields or content hash changed")
    shape = item["shape"]
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError("Opened arrays require one bounded cadence dimension")
    integer(shape[0], 1, 2_000_000)
    dtype = np.dtype(item["dtype"])
    if dtype.kind not in "fi" or dtype.itemsize not in {2, 4, 8} or delta and dtype != np.dtype("float32"):
        raise ValueError("Opened array dtype is invalid")
    if integer(item["bytes"], 1, 64 * 1024**2) != shape[0] * dtype.itemsize:
        raise ValueError("Array dimensions disagree with bounded byte inventory")


def array_inventory(arrays, reader):
    if set(arrays) != array_names(reader):
        raise ValueError("Opened source differs from its original21/29 array contract")
    for item in arrays.values():
        array_descriptor(item)
    if sum(item["bytes"] for item in arrays.values()) > 256 * 1024**2:
        raise ValueError("Opened source exceeds the one-source memory budget")


def source_header(sid, row):
    header = row["raw_header"]
    if set(header) != {"GAIAID", "KEPLERID", "TPFORG", "HLSPID", "HLSPVER"}:
        raise ValueError("Original public header fields changed")
    kic = header["KEPLERID"]
    if kic != "":
        integer(kic, 1)
    integer(header["TPFORG"], 1)
    if (str(header["GAIAID"]).split()[-1] != sid
        or (0 if kic == "" else kic) != row["product_kic"]
        or header["HLSPID"] != "KBONUS-BKG" or header["HLSPVER"].upper() != "V1.0"):
        raise ValueError("Pinned public header identity differs from source metadata")
    qualified = header["TPFORG"] != row["metadata"]["origin_tpf_kic"]
    if qualified != (row["local_canonical_proof_state"] == "LOCAL_EXACT_PARENT_QUALIFIED"):
        raise ValueError("Unexpected public header parent qualification")
    if qualified and row["reader"] != "catalogue":
        raise ValueError("Only the preserved native reader may retain qualified parents")


def source_delta(sid, row):
    if "delta" not in row:
        return
    item = row["delta"]
    if item["path"] != f"{sid}.npz" or item["transform"] != "additive_flux_delta":
        raise ValueError("Original delta identity or additive semantics changed")
    integer(item["bytes"], 1, LIMIT)
    if not public.sha256(item["sha256"]) or set(item["arrays"]) != set(DELTA_FLUX.values()):
        raise ValueError("Original delta hash or closed four-view inventory changed")
    for name, descriptor in item["arrays"].items():
        array_descriptor(descriptor, delta=True)
        if descriptor["shape"] != row["arrays"][name]["shape"]:
            raise ValueError("Original delta cadence dimensions differ from final arrays")


def source(sid, row):
    public.checked_source(sid, row)
    if (row.get("role") != "development" or row.get("opened_scope") != "CURRENT_OPENED_DEVELOPMENT"
        or COHORTS.get(row.get("opened_cohort")) != (row.get("reader"), "delta" in row)
        or str(row["metadata"]["origin_tpf_kic"]) != row["parents"]["origin"]):
        raise ValueError("Original opened source role, cohort or reader changed")
    integer(row["metadata"]["origin_tpf_kic"], 1)
    integer(row["metadata"]["shard"])
    integer(row["prepared_cadences"], 100, 2_000_000)
    integer(row["original_rank"], 1, 876)
    array_inventory(row["arrays"], row["reader"])
    if (row["base_arrays"] is None) != (row["base_checks"] is None):
        raise ValueError("Original base-array oracle availability changed")
    if row["base_arrays"] is not None:
        array_inventory(row["base_arrays"], row["reader"])
    source_header(sid, row)
    source_delta(sid, row)


def metadata_counts(data):
    rows = data["sources"].values()
    injected = [row for row in rows if "delta" in row]
    states = Counter(row["local_canonical_proof_state"] for row in rows)
    unavailable = Counter(row["state"] for row in data["unavailable"].values())
    return {"original": len(data["original_order"]), "supported": len(data["sources"]),
            "protected": unavailable["PROTECTED_LINK_METADATA_ONLY"],
            "background_recipe_unavailable": unavailable["UNAVAILABLE_ORIGINAL_BACKGROUND_RECIPE"],
            "cohorts": dict(Counter(row["opened_cohort"] for row in rows)),
            "arrays": sum(len(row["arrays"]) for row in rows),
            "delta_sources": len(injected), "delta_arrays": sum(len(row["delta"]["arrays"]) for row in injected),
            "native_parent_qualified": states["LOCAL_EXACT_PARENT_QUALIFIED"],
            "local_exact": states["LOCAL_EXACT"] + states["LOCAL_EXACT_PARENT_QUALIFIED"],
            "remote_canonical_proof_pending": states["REMOTE_CANONICAL_PROOF_PENDING"],
            "local_injected_base_oracles": sum(row["base_arrays"] is not None for row in injected),
            "unavailable_local_injected_base_oracles": sum(row["base_arrays"] is None for row in injected)}


def check_metadata(data, descriptor):
    boundary(data)
    if type(data.get("schema")) is not int or data["schema"] != 1 or data["status"] != "FIXED_OPENED876_PUBLIC_ARRAY_REPLAY_SPECIFICATION":
        raise ValueError("Opened metadata schema or status changed")
    original, supported = data["original_order"], data["supported_order"]
    if (len(original) != len(set(original)) or set(original) != set(data["sources"]) | set(data["unavailable"])
        or set(data["sources"]) & set(data["unavailable"])
        or supported != [sid for sid in original if sid in data["sources"]]):
        raise ValueError("Original876 source membership or order changed")
    for sid, row in data["sources"].items():
        source(sid, row)
    check_unavailable(data)
    check_parent_edges(data)
    check_budgets(data)
    for key, count in data["counts"].items():
        for value in count.values() if key == "cohorts" else (count,):
            integer(value)
    if data["counts"] != COUNTS or metadata_counts(data) != COUNTS:
        raise ValueError("Original876 supported/unavailable denominators changed")
    for field in (*FLAGS, "counts", "budgets", "reader_programs", "preparation_profiles", "scope_metadata", "joint_scope_decoded_sha256", "boundary"):
        if descriptor[field] != data[field]:
            raise ValueError(f"Opened descriptor and metadata disagree: {field}")
    if data["deltas"]["archive"] != descriptor["deltas"]:
        raise ValueError("Original delta archive binding changed")


def check_unavailable(data):
    states = {"PROTECTED_LINK_METADATA_ONLY", "UNAVAILABLE_ORIGINAL_BACKGROUND_RECIPE"}
    for sid, row in data["unavailable"].items():
        if (not re.fullmatch("[1-9][0-9]*", sid) or row["source_id"] != sid
            or row["state"] not in states or row["role"] != "development"
            or row["opened_scope"] != "CURRENT_OPENED_DEVELOPMENT"
            or not public.sha256(row["raw"]["sha256"])):
            raise ValueError("Unavailable source identity or metadata boundary changed")
        if (row["state"] == "UNAVAILABLE_ORIGINAL_BACKGROUND_RECIPE") != (row["cohort"] == "background"):
            raise ValueError("Unavailable background/protected class binding changed")


def check_parent_edges(data):
    qualified = {sid: row for sid, row in data["sources"].items()
                 if row["local_canonical_proof_state"] == "LOCAL_EXACT_PARENT_QUALIFIED"}
    edges = data["actual_header_parent_edges"]
    if len(edges) != len(qualified) or {edge["source_id"] for edge in edges} != set(qualified):
        raise ValueError("Qualified native actual-header parent edge inventory changed")
    for edge in edges:
        row = qualified[edge["source_id"]]
        if (edge["original_reader_parent"] != row["parents"]["origin"]
            or edge["raw_fits_sha256"] != row["sha256"]
            or edge["actual_raw_header_TPFORG"] != str(row["raw_header"]["TPFORG"])
            or edge["known_protected_link_found"] is not False
            or edge["joint_independence_complete"] is not False
            or edge["must_enter_future_joint_null_connectivity_before_selection_or_freezing"] is not True):
            raise ValueError("Qualified native parent provenance or protection state changed")


def preparation_bytes(row):
    discovery = row["prepared_cadences"] * 22 + 32
    quarter_views = 3 * (row["arrays"]["qdiag_time_bkjd"]["shape"][0] * 22 + 32)
    return discovery + quarter_views


def check_budgets(data):
    rows, budgets = list(data["sources"].values()), data["budgets"]
    expected = {"raw_fits_bytes": sum(row["bytes"] for row in rows),
                "original_compressed_array_bytes": sum(row["original_prepared_array_bytes"] for row in rows),
                "array_uncompressed_bytes": sum(item["bytes"] for row in rows for item in row["arrays"].values()),
                "fsc_forecast_bytes": sum(preparation_bytes(row) for row in rows),
                "paired_run_headroom_bytes": 4 * 1024**3, "max_archive_bytes": LIMIT,
                "max_cloud_storage_bytes": 32_000_000_000,
                "external_minimum_free_bytes": 20 * 1024**3,
                "cloud_development_minimum_free_bytes": 2 * 1024**3}
    if budgets != expected:
        raise ValueError("Opened full-scope storage forecast or reserve policy changed")
    for value in budgets.values():
        integer(value)


def packet(specification=SPECIFICATION):
    specification = Path(specification).absolute()
    parent_paths(specification)
    if specification.is_symlink() or not specification.is_file() or specification.stat().st_size > 64 * 1024:
        raise ValueError("Opened packet descriptor must be a bounded regular file")
    payload = specification.read_bytes()
    descriptor = parse(payload)
    boundary(descriptor)
    if type(descriptor.get("schema")) is not int or descriptor["schema"] != 1 or descriptor.get("status") != "CLOUD_OPENED876_ARRAY_REPLAY_PACKET":
        raise ValueError("Opened descriptor schema or status changed")
    data, metadata_path = decoded_metadata(specification, descriptor)
    check_metadata(data, descriptor)
    archive = artifact_path(specification, descriptor["deltas"])
    integer(descriptor["deltas"]["bytes"], 1, LIMIT)
    checked_file(archive, descriptor["deltas"])
    dependencies = {specification: hashlib.sha256(payload).hexdigest(), metadata_path: descriptor["artifact"]["sha256"],
                    archive: descriptor["deltas"]["sha256"], Path(__file__): digest(Path(__file__))}
    required = {"pipeline/deterministic/cloud_public.py", "pipeline/deterministic/catalogue_controls.py",
                "pipeline/photometry/lc2.py"}
    if len(data["reader_programs"]) != len(required) or {item["path"] for item in data["reader_programs"]} != required:
        raise ValueError("Original reader/preparation code inventory is not closed")
    for item in data["reader_programs"]:
        path = ROOT / item["path"]
        if path.resolve() != path or ROOT not in path.parents:
            raise ValueError("Reader program binding escapes the repository")
        checked_file(path, item)
        dependencies[path] = item["sha256"]
    profile = preparation_profile(data["preparation_profiles"])
    data["selected_preparation_profile"] = profile
    dependencies.update({ROOT / item["path"]: item["sha256"] for item in profile["files"]})
    if data["scope_metadata"] != SCOPE_METADATA or data["joint_scope_decoded_sha256"] != JOINT_SCOPE_DECODED_SHA256:
        raise ValueError("Reviewed actual-parent scope overlay binding changed")
    for item in SCOPE_METADATA:
        path = ROOT / item["path"]
        checked_file(path, item)
        dependencies[path] = item["sha256"]
    return data, archive, dependencies


def preparation_profile(profiles):
    if profiles != PREPARATION_PROFILES:
        raise ValueError("Reviewed preparation version inventory changed")
    matches = []
    for profile in profiles:
        paths = [(ROOT / item["path"], item) for item in profile["files"]]
        if all(path.is_file() and not path.is_symlink() and identity(path) == {key: item[key] for key in ("bytes", "sha256")}
               for path, item in paths):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError("Current preparation code does not match exactly one reviewed version")
    return matches[0]


def guard_path(root, path):
    if root != path and root not in path.parents or ".." in path.relative_to(root).parts:
        raise ValueError("Opened output escapes its managed destination")
    for current in (path, *path.parents):
        if current == root.parent:
            break
        if current.is_symlink():
            raise ValueError(f"Opened output redirects through a symlink: {current}")


def destination_path(destination):
    if ".." in destination.parts:
        raise ValueError("Opened destination must not traverse parent directories")
    parent_paths(destination)
    return dump_path(destination)


def begin(destination):
    root = destination_path(destination)
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / name for name in OUTPUT_NAMES]
    bad = any(path.is_symlink() or path.exists() and not path.is_file() for path in paths)
    withdraw(root)
    if bad:
        raise ValueError("Opened metadata output must not redirect writes")
    for path in root.rglob("*"):
        guard_path(root, path)
        if path.name.endswith(".part"):
            raise ValueError(f"Stale opened partial requires explicit removal: {path}")
    return root


def withdraw(root):
    for name in OUTPUT_NAMES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            path.unlink()
    withdraw_truths(root)


def check_cache(root, data):
    allowed = {f"{sid}.{suffix}" for sid in data["sources"] for suffix in ("fits", "npz")}
    allowed |= set(OUTPUT_NAMES) | {"deltas", "truths"}
    if any(path.name not in allowed or path.name not in {"deltas", "truths"} and not path.is_file()
           for path in root.iterdir()):
        raise ValueError("Opened cache contains unknown or wrong-type entries")
    folder = root / "deltas"
    if folder.exists() and not folder.is_dir():
        raise ValueError("Opened recipe cache is not a directory")
    truth_cache(root)


def reserve(root, data):
    external = any(path == root or path in root.parents for path in EXTERNAL_DUMPS)
    cloud = os.getenv("SPACELAB_EXECUTION_PROFILE") == "cloud-development"
    key = "cloud_development_minimum_free_bytes" if cloud and not external else "external_minimum_free_bytes"
    value = data["budgets"][key]
    if value != (2 if cloud and not external else 20) * 1024**3:
        raise ValueError("Opened destination reserve policy changed")
    return value


def checkpoint(root, data, growth=0):
    if shutil.disk_usage(root).free < reserve(root, data) + growth:
        raise ValueError("Insufficient opened destination capacity and free-space reserve")


def forecast(root, data, ids=None):
    ids = data["supported_order"] if ids is None else ids
    raw, arrays = 0, 0
    for sid in ids:
        row = data["sources"][sid]
        path = root / f"{sid}.fits"
        guard_path(root, path)
        if path.exists():
            checked_file(path, row)
        else:
            raw += row["bytes"]
        output = root / f"{sid}.npz"
        guard_path(root, output)
        if not output.exists():
            arrays += sum(item["bytes"] for item in row["arrays"].values()) + 64 * 1024
    fsc = sum(preparation_bytes(data["sources"][sid]) for sid in ids)
    growth = raw + arrays + fsc + data["budgets"]["paired_run_headroom_bytes"] + LIMIT
    disk = shutil.disk_usage(root if root.exists() else root.parent)
    free = disk.free
    cloud = os.getenv("SPACELAB_EXECUTION_PROFILE") == "cloud-development" and reserve(root, data) < 20 * 1024**3
    within_vm = not cloud or disk.used + growth + reserve(root, data) <= data["budgets"]["max_cloud_storage_bytes"]
    return {"remaining_raw_bytes": raw, "array_upper_bound_bytes": arrays,
            "fsc_forecast_bytes": fsc, "requested_sources": len(ids),
            "paired_run_headroom_bytes": data["budgets"]["paired_run_headroom_bytes"],
            "forecast_growth_bytes": growth, "available_bytes": free, "reserve_bytes": reserve(root, data),
            "current_filesystem_used_bytes": disk.used, "filesystem_total_bytes": disk.total,
            "within_declared_cloud32gb_budget": within_vm,
            "capacity_sufficient": free >= reserve(root, data) + growth and within_vm}


def publish(root, path, payload):
    temporary = path.with_suffix(path.suffix + ".part")
    guard_path(root, path)
    guard_path(root, temporary)
    if temporary.exists():
        raise ValueError("Opened publication partial already exists")
    with temporary.open("xb") as stream:
        stream.write(payload)
    temporary.chmod(0o444)
    temporary.replace(path)
    expected = hashlib.sha256(payload).hexdigest()
    if digest(path) != expected:
        raise ValueError("Opened published bytes drifted")
    return expected


def recipes(root, archive, data):
    folder = root / "deltas"
    guard_path(root, folder)
    folder.mkdir(exist_ok=True)
    files = data["deltas"]["files"]
    expected = {row["delta"]["path"]: {key: row["delta"][key] for key in ("bytes", "sha256")}
                for row in data["sources"].values() if "delta" in row}
    if files != expected:
        raise ValueError("Original180 recipe source inventory changed")
    seen = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if member.name in seen or member.name not in files or not member.isfile() or member.size != files[member.name]["bytes"]:
                raise ValueError("Original recipe archive member changed")
            seen.add(member.name)
            path = folder / member.name
            if not path.exists():
                with bundle.extractfile(member) as stream:
                    payload = stream.read(member.size + 1)
                if len(payload) != member.size or hashlib.sha256(payload).hexdigest() != files[member.name]["sha256"]:
                    raise ValueError("Original recipe archive content changed")
                publish(root, path, payload)
            checked_file(path, files[member.name])
            path.chmod(0o444)
    if seen != set(files) or {path.name for path in folder.iterdir()} != seen:
        raise ValueError("Original recipe inventory is not closed")
    return folder


def npz_values(path, expected):
    parent_paths(path)
    if not path.is_file() or path.stat().st_size > sum(item["bytes"] for item in expected.values()) + 64 * 1024:
        raise ValueError("Opened NPZ container exceeds its bounded array inventory")
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != {f"{name}.npy" for name in expected}:
            raise ValueError("Original delta NPZ member inventory changed")
        for name, descriptor in expected.items():
            member = archive.getinfo(f"{name}.npy")
            if member.file_size > descriptor["bytes"] + 4096:
                raise ValueError("Original delta NPY exceeds its declared memory budget")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, dtype = np.lib.format._read_array_header(stream, version)
            if list(shape) != descriptor["shape"] or str(dtype) != descriptor["dtype"]:
                raise ValueError("Original delta NPY dimensions or dtype changed")
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    public.verify_inventory(values, expected)
    return values


def delta_values(path, item):
    checked_file(path, item)
    values = npz_values(path, item["arrays"])
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Original delta values are not finite")
    return values


def checked_header(path, row):
    with fits.open(path, memmap=True) as hdus:
        actual = {key: hdus[0].header[key] for key in row["raw_header"]}
        tables = [hdu for hdu in hdus if hdu.name == "LIGHTCURVE_STITCHED" or re.fullmatch("LIGHTCURVE_Q[0-9]+", hdu.name)]
        table_bytes = sum(integer(hdu.header["NAXIS1"], 1) * integer(hdu.header["NAXIS2"], 1, 2_000_000) for hdu in tables)
        if table_bytes > row["bytes"] or hdus["LIGHTCURVE_STITCHED"].header["NAXIS2"] != row["arrays"]["time_bkjd"]["shape"][0]:
            raise ValueError("Public FITS table dimensions exceed original bounded inventory")
    if actual != row["raw_header"] or any(type(actual[key]) is not type(row["raw_header"][key]) for key in actual):
        raise ValueError("Unexpected canonical raw header changed before photometry access")


def reconstruct(path, sid, row, deltas):
    checked_header(path, row)
    if row["reader"] == "catalogue":
        metadata, raw = read_product(path, row["product_kic"])
        if metadata != row["metadata"]:
            raise ValueError("Original catalogue21 reader metadata changed")
    else:
        raw = public.fits_arrays(path, sid, row)
    if row["base_arrays"] is not None:
        public.verify_inventory(raw, row["base_arrays"])
        public.verify_arrays(raw, row["base_checks"])
    if "delta" in row:
        values = delta_values(deltas / row["delta"]["path"], row["delta"])
        for name, delta in values.items():
            if delta.shape != raw[name].shape:
                raise ValueError("Original delta dimensions differ from public base")
            raw[name] = np.asarray(raw[name], dtype=np.float64) + delta
    public.verify_inventory(raw, row["arrays"])
    count = public.verify_arrays(raw, row)
    if count != row["prepared_cadences"]:
        raise ValueError("Original prepared cadence accounting changed")
    return raw


def download(root, path, row, data, metrics):
    guard_path(root, path)
    if path.exists():
        checked_file(path, row)
        metrics["raw_cache_hits"] += 1
        return
    temporary = path.with_suffix(".fits.part")
    guard_path(root, temporary)
    start = time.monotonic()
    with public.requests.get(public.checked_url(row), stream=True, timeout=(15, 120), allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200 or response.url != row["url"]:
            raise ValueError("MAST changed the pinned response or canonical URL")
        count = 0
        with temporary.open("xb") as stream:
            for block in response.iter_content(1024 * 1024):
                count += len(block)
                if count > row["bytes"]:
                    raise ValueError("MAST response exceeds pinned raw size")
                checkpoint(root, data, len(block))
                stream.write(block)
    checked_file(temporary, row)
    temporary.chmod(0o444)
    temporary.replace(path)
    metrics["network_bytes"] += count
    metrics["network_seconds"] += time.monotonic() - start


def cached_arrays(root, path, raw, row):
    guard_path(root, path)
    if path.exists():
        expected = digest(path)
        values = npz_values(path, row["arrays"])
        public.verify_arrays(values, row)
        if digest(path) != expected:
            raise ValueError("Cached opened container drifted during verification")
        return True, expected
    temporary = path.with_suffix(".npz.part")
    guard_path(root, temporary)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **raw)
    expected = digest(temporary)
    temporary.chmod(0o444)
    temporary.replace(path)
    if digest(path) != expected:
        raise ValueError("Generated opened array container drifted during publication")
    return False, expected


def recheck(dependencies):
    for path, expected in dependencies.items():
        parent_paths(path)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Opened dependency changed during verification: {path}")


def serialized(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def finish(root, data, section, selection, sources, dependencies, metrics, resources):
    ids = checked_selection(data, selection)
    if list(sources) != ids:
        raise ValueError("Opened publication lost or reordered requested batch sources")
    complete, cohort_complete, status = selection_completion(selection, ids)
    common = {"scope": public.SCOPE, "role": "development", **{flag: False for flag in FLAGS},
              "original_opened876_selection_unchanged": True, "provisioning_complete": complete,
              "selected_cohorts_complete": cohort_complete, "computational_selection": selection}
    manifest = {"schema": 1, "measurement_scope": public.SCOPE, **common, "sources": sources}
    outputs = {"input-manifest.json": serialized(manifest), "source-ids.json": serialized(list(sources))}
    for name, payload in outputs.items():
        dependencies[root / name] = hashlib.sha256(payload).hexdigest()
        publish(root, root / name, payload)
    recheck(dependencies)
    receipt = {"status": status,
               **common, "section": section, "original_counts": data["counts"],
               "provisioned_sources": len(sources), "unavailable": data["unavailable"],
               "manifest_sha256": dependencies[root / "input-manifest.json"],
               "source_ids_sha256": dependencies[root / "source-ids.json"],
               "dependencies": {str(path): expected for path, expected in dependencies.items()},
               "selected_preparation_profile": data["selected_preparation_profile"],
               "scope_metadata": data["scope_metadata"],
               "joint_scope_decoded_sha256": data["joint_scope_decoded_sha256"],
               "config": {"input_manifest": str(root / "input-manifest.json"),
                          "source_ids": str(root / "source-ids.json"), "frozen_pixel_inputs": None},
               "metrics": metrics, "resources": resources, "boundary": data["boundary"]}
    payload = serialized(receipt)
    expected = publish(root, root / "receipt.json", payload)
    recheck(dependencies)
    if digest(root / "receipt.json") != expected:
        raise ValueError("Opened success receipt bytes changed after publication")
    return receipt


def computational_selection(data, cohorts=None, limit=None):
    if cohorts is None:
        names = list(COHORTS)
    else:
        if (not isinstance(cohorts, (list, tuple)) or not cohorts
            or any(not isinstance(name, str) or name not in COHORTS for name in cohorts)
            or len(cohorts) != len(set(cohorts))):
            raise ValueError("Opened computational cohorts must be unique known names")
        names = [name for name in COHORTS if name in cohorts]
    ids = [sid for sid in data["supported_order"] if data["sources"][sid]["opened_cohort"] in names]
    if not ids:
        raise ValueError("Opened computational cohort selection is empty")
    count = len(ids)
    if limit is not None:
        integer(limit, 1, len(ids))
        ids = ids[:limit]
    return {"schema": 1, "cohorts": names, "source_limit": limit,
            "cohort_sources": count, "requested_sources": len(ids)}, ids


def checked_selection(data, selection):
    if (not isinstance(selection, dict)
        or set(selection) != {"schema", "cohorts", "source_limit", "cohort_sources", "requested_sources"}
        or type(selection["schema"]) is not int or selection["schema"] != 1
        or not isinstance(selection["cohorts"], list)):
        raise ValueError("Opened computational selection schema or fields changed")
    integer(selection["cohort_sources"], 1, COUNTS["supported"])
    integer(selection["requested_sources"], 1, COUNTS["supported"])
    expected, ids = computational_selection(data, selection["cohorts"], selection["source_limit"])
    if selection != expected:
        raise ValueError("Opened computational selection order or denominators changed")
    return ids


def selection_completion(selection, ids):
    complete = len(ids) == COUNTS["supported"]
    cohort_complete = len(ids) == selection["cohort_sources"]
    status = ("OPENED_PUBLIC_INPUTS_VERIFIED" if complete else "OPENED_PUBLIC_INPUTS_COHORTS_VERIFIED"
              if cohort_complete else "OPENED_PUBLIC_INPUTS_PARTIAL_VERIFIED")
    return complete, cohort_complete, status


def handoff_metadata(root, data):
    payloads = {}
    for name in OUTPUT_NAMES:
        path = root / name
        guard_path(root, path)
        if not path.is_file() or path.stat().st_size > LIMIT:
            raise ValueError("Opened handoff metadata must remain bounded regular files")
        payloads[name] = path.read_bytes()
    receipt, manifest, ids = (parse(payloads[name]) for name in OUTPUT_NAMES)
    boundary(receipt)
    boundary(manifest)
    expected_ids = checked_selection(data, manifest.get("computational_selection"))
    receipt_ids = checked_selection(data, receipt.get("computational_selection"))
    if (type(manifest.get("schema")) is not int or manifest["schema"] != 1
        or manifest.get("measurement_scope") != public.SCOPE
        or not isinstance(ids, list) or not ids or ids != expected_ids or ids != receipt_ids
        or manifest["computational_selection"] != receipt["computational_selection"]
        or list(manifest["sources"]) != ids):
        raise ValueError("Opened handoff source membership or original order changed")
    complete, cohort_complete, status = selection_completion(manifest["computational_selection"], ids)
    if (receipt.get("status") != status or receipt.get("section") not in {"1st", "2nd"}
        or receipt.get("provisioning_complete") is not complete or manifest.get("provisioning_complete") is not complete
        or receipt.get("selected_cohorts_complete") is not cohort_complete or manifest.get("selected_cohorts_complete") is not cohort_complete
        or type(receipt.get("provisioned_sources")) is not int or receipt["provisioned_sources"] != len(ids)
        or receipt.get("original_counts") != data["counts"] or receipt.get("unavailable") != data["unavailable"]
        or receipt.get("selected_preparation_profile") != data["selected_preparation_profile"]
        or receipt.get("scope_metadata") != data["scope_metadata"]
        or receipt.get("joint_scope_decoded_sha256") != data["joint_scope_decoded_sha256"]):
        raise ValueError("Opened handoff receipt completeness or binding changed")
    hashes = {root / name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    if (receipt.get("manifest_sha256") != hashes[root / "input-manifest.json"]
        or receipt.get("source_ids_sha256") != hashes[root / "source-ids.json"]
        or receipt.get("config") != {"input_manifest": str(root / "input-manifest.json"),
                                    "source_ids": str(root / "source-ids.json"), "frozen_pixel_inputs": None}):
        raise ValueError("Opened handoff metadata bytes or configuration changed")
    return receipt, manifest, ids, hashes


def handoff_arrays(root, manifest, data):
    dependencies = {}
    fields = ("metadata", "opened_cohort", "opened_scope", "origin")
    for sid, saved in manifest["sources"].items():
        row = data["sources"][sid]
        if (set(saved) != {"arrays", "sha256", *fields} or saved["arrays"] != f"{sid}.npz"
            or any(saved[field] != row[field] for field in fields) or not public.sha256(saved["sha256"])):
            raise ValueError("Opened handoff source identity or array path changed")
        path, raw_path = root / f"{sid}.npz", root / f"{sid}.fits"
        guard_path(root, path)
        checked_file(raw_path, row)
        checked_header(raw_path, row)
        values = npz_values(path, row["arrays"])
        if digest(path) != saved["sha256"] or public.verify_arrays(values, row) != row["prepared_cadences"]:
            raise ValueError("Opened handoff array bytes or prepared oracle changed")
        dependencies.update({path: saved["sha256"], raw_path: row["sha256"]})
    return dependencies


def verified_inputs(specification=SPECIFICATION, destination=None):
    destination = Path(destination) if destination is not None else ROOT / "dump/cloud-opened"
    root = destination_path(destination)
    if not root.is_dir():
        raise ValueError("Verified opened input directory required")
    try:
        data, _, dependencies = packet(specification)
        check_cache(root, data)
        receipt, manifest, ids, hashes = handoff_metadata(root, data)
        dependencies.update(handoff_arrays(root, manifest, data))
        for name, item in data["deltas"]["files"].items():
            path = root / "deltas" / name
            checked_file(path, item)
            dependencies[path] = item["sha256"]
        dependencies.update({path: value for path, value in hashes.items() if path.name != "receipt.json"})
        if receipt.get("dependencies") != {str(path): value for path, value in dependencies.items()}:
            raise ValueError("Opened handoff dependency inventory is not closed")
        dependencies.update(hashes)
        recheck(dependencies)
        return manifest, {sid: data["sources"][sid] for sid in ids}, dependencies
    except BaseException:
        withdraw(root)
        raise


def provision(section, specification=SPECIFICATION, destination=None, source_limit=None, dry_run=False, cohorts=None):
    if section not in {"1st", "2nd"}:
        raise ValueError("Opened cloud section must be 1st or 2nd")
    destination = Path(destination) if destination is not None else ROOT / "dump/cloud-opened"
    if dry_run:
        data, _, _ = packet(specification)
        root = destination_path(destination)
        selection, ids = computational_selection(data, cohorts, source_limit)
        return {"status": "OPENED_METADATA_DRY_RUN", "counts": data["counts"],
                "requested_sources": len(ids), "computational_selection": selection,
                "scientific_acceptance": False, "joint_independence_complete": False,
                "provisioning_complete": False, "selected_cohorts_complete": False,
                "resources": forecast(root, data, ids)}
    root = begin(destination)
    start = time.monotonic()
    cpu_start = time.process_time()
    disk_start = shutil.disk_usage(root).free
    try:
        data, archive, dependencies = packet(specification)
        check_cache(root, data)
        selection, ids = computational_selection(data, cohorts, source_limit)
        resources = forecast(root, data, ids)
        if not resources["capacity_sufficient"]:
            raise ValueError("Opened requested-batch forecast exceeds capacity or declared32GB cloud budget")
        checkpoint(root, data, resources["forecast_growth_bytes"])
        deltas = recipes(root, archive, data)
        dependencies.update({path: digest(path) for path in deltas.iterdir()})
        metrics = {"network_bytes": 0, "network_seconds": 0.0, "raw_cache_hits": 0, "array_cache_hits": 0}
        sources = {}
        for sid in ids:
            row = data["sources"][sid]
            path = root / f"{sid}.fits"
            download(root, path, row, data, metrics)
            raw = reconstruct(path, sid, row, deltas)
            output = root / f"{sid}.npz"
            checkpoint(root, data, sum(array.nbytes for array in raw.values()) + 64 * 1024)
            cache_hit, array_hash = cached_arrays(root, output, raw, row)
            metrics["array_cache_hits"] += cache_hit
            dependencies.update({path: row["sha256"], output: array_hash})
            sources[sid] = {"arrays": output.name, "sha256": dependencies[output],
                            **{key: row[key] for key in ("metadata", "opened_cohort", "opened_scope", "origin")}}
        resources["available_bytes_after"] = shutil.disk_usage(root).free
        metrics.update({"wall_seconds": time.monotonic() - start,
                        "cpu_seconds": time.process_time() - cpu_start,
                        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                        "disk_growth_bytes": disk_start - resources["available_bytes_after"]})
        return finish(root, data, section, selection, sources, dependencies, metrics, resources)
    except BaseException:
        withdraw(root)
        raise


def withdraw_truths(root):
    folder = root / "truths"
    if folder.is_symlink():
        return
    path = folder / "receipt.json"
    if path.is_file() and not path.is_symlink():
        path.unlink()


def truth_cache(root):
    folder = root / "truths"
    guard_path(root, folder)
    if not folder.exists():
        return
    if not folder.is_dir():
        raise ValueError("Opened truth cache must be a managed directory")
    files = {folder / "receipt.json", *(folder / cohort / "truth.jsonl" for cohort in TRUTH_COHORTS)}
    directories = {folder / cohort for cohort in TRUTH_COHORTS}
    for path in folder.rglob("*"):
        guard_path(root, path)
        if path in files and path.is_file() or path in directories and path.is_dir():
            continue
        raise ValueError("Opened truth cache has unknown entries or stale partials")


def truth_json(payload):
    def reject(value):
        raise ValueError(f"Nonfinite opened truth JSON: {value}")
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject)


def check_truth_packet(spec, specification, data):
    boundary(spec)
    if (set(spec) != TRUTH_PACKET_FIELDS or type(spec.get("schema")) is not int or spec["schema"] != 1
        or spec.get("status") != "CLOUD_OPENED_ORIGINAL_PHYSICAL_TRUTHS_PACKET"
        or spec.get("original_truth_bytes_preserved") is not True
        or spec.get("new_sources_or_seeds_or_truths_generated") is not False):
        raise ValueError("Opened truth packet schema or development boundary changed")
    fixed = spec["fixed_array_packet"]
    checked_file(specification, fixed["descriptor"])
    descriptor = truth_json(specification.read_bytes())
    for key in ("artifact", "deltas", "counts", "reader_programs", "preparation_profiles", "joint_scope_decoded_sha256"):
        if fixed[key] != descriptor[key]:
            raise ValueError("Opened truths are bound to a different current array packet")
    protected = sorted(sid for sid, row in data["unavailable"].items() if row["state"] == "PROTECTED_LINK_METADATA_ONLY")
    if spec["protected_metadata_only_sources_excluded"] != protected:
        raise ValueError("Opened truth protected identity exclusions changed")
    if [group["opened_cohort"] for group in spec["cohorts"]] != list(TRUTH_COHORTS):
        raise ValueError("Opened truth cohort inventory or order changed")
    for group in spec["cohorts"]:
        check_truth_cohort(group, data)
    check_truth_clocks(spec)
    members = [group["truth"] for group in spec["cohorts"]]
    expected_counts = {"sources": sum(value[2] for value in TRUTH_COHORTS.values()),
                       **{key: value[2] for key, value in TRUTH_COHORTS.items()},
                       "members": len(members), "truth_bytes": sum(item["bytes"] for item in members),
                       "bound_original_float32_delta_arrays": COUNTS["delta_arrays"],
                       "protected_truths_read": 0, "mixed_sealed_truth_files_read": 0}
    if spec["members"] != members or spec["counts"] != expected_counts:
        raise ValueError("Opened truth member or source denominator inventory changed")
    for value in spec["counts"].values():
        integer(value)


def check_truth_cohort(group, data):
    name = group["opened_cohort"]
    original, category, count = TRUTH_COHORTS[name]
    ids = [sid for sid in data["supported_order"] if data["sources"][sid]["opened_cohort"] == name]
    if (set(group) != {"category", "opened_cohort", "original_cohort", "original_inputs",
                      "original_manifest_generator", "original_partition_status", "original_truth_fields",
                      "source_order", "source_truth_bindings", "sources", "truth"}
        or group["original_cohort"] != original or group["category"] != category
        or type(group["sources"]) is not int or group["sources"] != count
        or len(ids) != count or group["source_order"] != ids
        or [row["source_id"] for row in group["source_truth_bindings"]] != ids
        or group["original_truth_fields"] != sorted(TRUTH_FIELDS)
        or group["truth"]["path"] != f"{name}/truth.jsonl"):
        raise ValueError("Opened truth source membership, category or original order changed")
    integer(group["truth"]["bytes"], 1, 256 * 1024)
    if set(group["truth"]) != {"path", "bytes", "sha256"} or not public.sha256(group["truth"]["sha256"]):
        raise ValueError("Opened truth file descriptor changed")
    for index, binding in enumerate(group["source_truth_bindings"]):
        check_truth_binding(binding, data["sources"][ids[index]], index, name)
    check_truth_origins(group, data)


def check_truth_origins(group, data):
    files = {Path(item["path"]).name: item for item in group["original_inputs"]}
    if len(files) != 4 or len(group["original_inputs"]) != 4 or set(files) != {"input-manifest.json", "materialization.json", "source-ids.json", "truth.jsonl"}:
        raise ValueError("Opened original truth metadata inventory is not closed")
    if ({key: files["truth.jsonl"][key] for key in ("bytes", "sha256")}
        != {key: group["truth"][key] for key in ("bytes", "sha256")}
        or group["original_partition_status"] not in {"OPENED_DEVELOPMENT", "OPENED_DEVELOPMENT_PRESERVED"}):
        raise ValueError("Opened original truth bytes or exposure changed")
    for sid in group["source_order"]:
        original = data["sources"][sid]["origin"]["original_input_manifest"]
        if any(original[key] != files["input-manifest.json"][key] for key in ("path", "sha256")):
            raise ValueError("Opened truth original source manifest identity changed")


def check_truth_clocks(spec):
    clocks = spec["original_clock_generation"]
    if len(clocks) != len(TRUTH_CLOCKS) or {row["sha256"] for row in clocks} != set(TRUTH_CLOCKS):
        raise ValueError("Opened original clock generation inventory changed")
    for row in clocks:
        size, commit = TRUTH_CLOCKS[row["sha256"]]
        if (set(row) != {"path", "bytes", "sha256", "git_commit", "functions"}
            or row["path"] != "pipeline/deterministic/injections.py" or row["bytes"] != size
            or row["git_commit"] != commit or set(row["functions"]) != set(TRUTH_CLOCK_FUNCTIONS)):
            raise ValueError("Opened original clock code binding changed")
        for name, expected in TRUTH_CLOCK_FUNCTIONS.items():
            item = row["functions"][name]
            if set(item) != {"source", "sha256"} or item["sha256"] != expected or hashlib.sha256(item["source"].encode()).hexdigest() != expected:
                raise ValueError("Opened original clock snapshot changed")
    expected = {group["original_manifest_generator"]["pipeline/deterministic/injections.py"] for group in spec["cohorts"]}
    if expected - set(TRUTH_CLOCKS):
        raise ValueError("Opened cohort clock generator binding changed")


def check_truth_binding(binding, row, index, cohort):
    expected = {"truth_index": index, "opened_cohort": cohort, "role": "development", "reader": "lc2",
                "reader_parent": str(row["metadata"]["origin_tpf_kic"]),
                "base_raw": {"bytes": row["bytes"], "sha256": row["sha256"]},
                "original_delta": row["delta"], "original_injected_array": row["origin"]["original_array"],
                "raw_view_sha256": row["raw_view_sha256"], "prepared_records_sha256": row["prepared_records_sha256"]}
    if (set(binding) != {*expected, "source_id", "line_bytes", "line_sha256", "original_delta_reference"}
        or binding["source_id"] != row["metadata"]["source_id"]
        or type(binding["truth_index"]) is not int
        or any(binding[key] != value for key, value in expected.items())
        or not public.sha256(binding["line_sha256"])):
        raise ValueError("Opened truth raw, reader-parent, original delta or FSC binding changed")
    integer(binding["line_bytes"], 1, 16 * 1024)
    source_delta(binding["source_id"], row)


def truth_packet(specification, truth_specification, data):
    path = Path(truth_specification).absolute()
    checked_file(path, {"bytes": integer(path.stat().st_size, 1, 512 * 1024), "sha256": TRUTH_SPECIFICATION_SHA256})
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TRUTH_SPECIFICATION_SHA256:
        raise ValueError("Opened truth descriptor changed during capture")
    spec = truth_json(payload)
    check_truth_packet(spec, Path(specification).absolute(), data)
    archive = artifact_path(path, spec["archive"])
    integer(spec["archive"]["bytes"], 1, 256 * 1024)
    checked_file(archive, spec["archive"])
    dependencies = {path: TRUTH_SPECIFICATION_SHA256, archive: spec["archive"]["sha256"]}
    return spec, archive, dependencies


def truth_members(archive, spec):
    item = spec["archive"]
    expected = integer(item["decoded_bytes"], 1, 256 * 1024)
    if (set(item) != {"path", "bytes", "sha256", "decoded_bytes", "decoded_sha256", "codec", "tar_format", "mtime", "closed_regular_members"}
        or item["codec"] != "gzip" or item["tar_format"] != "USTAR" or type(item["mtime"]) is not int or item["mtime"] != 0
        or item["closed_regular_members"] is not True):
        raise ValueError("Opened truth archive codec or header contract changed")
    with gzip.open(archive, "rb") as stream:
        payload = stream.read(expected + 1)
    if len(payload) != expected or hashlib.sha256(payload).hexdigest() != item["decoded_sha256"]:
        raise ValueError("Opened truth decompression differs from its bounded inventory")
    members = {item["path"]: item for item in spec["members"]}
    values = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as bundle:
        for member in bundle:
            check_truth_member(member, members, values)
            with bundle.extractfile(member) as stream:
                value = stream.read(member.size + 1)
            if len(value) != member.size or hashlib.sha256(value).hexdigest() != members[member.name]["sha256"]:
                raise ValueError("Opened original truth member bytes changed")
            values[member.name] = value
    if list(values) != list(members):
        raise ValueError("Opened truth archive is missing or reorders original members")
    for group in spec["cohorts"]:
        check_truth_rows(values[group["truth"]["path"]], group)
    return values


def check_truth_member(member, members, seen):
    if (member.name not in members or member.name in seen or member.type != tarfile.REGTYPE
        or member.size != members[member.name]["bytes"] or member.mode != 0o444
        or member.uid != 0 or member.gid != 0 or member.mtime != 0
        or member.uname or member.gname or member.linkname or member.pax_headers):
        raise ValueError("Opened truth archive member, order or regular header changed")


def finite_truth(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_truth(item)
    elif isinstance(value, list):
        for item in value:
            finite_truth(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Opened truth numeric values must remain finite")


def check_truth_rows(payload, group):
    lines = payload.splitlines(keepends=True)
    if len(lines) != group["sources"] or not payload.endswith(b"\n"):
        raise ValueError("Opened truth rows or original newline framing changed")
    for line, binding in zip(lines, group["source_truth_bindings"], strict=True):
        row = truth_json(line)
        finite_truth(row)
        if (len(line) != binding["line_bytes"] or hashlib.sha256(line).hexdigest() != binding["line_sha256"]
            or set(row) != TRUTH_FIELDS or row["source_id"] != binding["source_id"]
            or row["category"] != group["category"]):
            raise ValueError("Opened truth original row identity, category or line bytes changed")


def truth_files(root, spec):
    dependencies, refs = {}, {}
    for group in spec["cohorts"]:
        item = group["truth"]
        path = root / "truths" / item["path"]
        guard_path(root, path)
        checked_file(path, item)
        check_truth_rows(path.read_bytes(), group)
        dependencies[path] = item["sha256"]
        for binding in group["source_truth_bindings"]:
            refs[binding["source_id"]] = {"cohort": group["opened_cohort"], "path": str(path),
                "file_sha256": item["sha256"], "truth_index": binding["truth_index"],
                "line_sha256": binding["line_sha256"]}
    return refs, dependencies


def truth_receipt(spec, ids, section, dependencies, selection):
    injected = {sid for group in spec["cohorts"] for sid in group["source_order"]}
    selected = [sid for sid in ids if sid in injected]
    return {"schema": 1, "status": "OPENED_TRUTHS_VERIFIED", "scope": public.SCOPE, "role": "development",
            **{flag: False for flag in FLAGS}, "original_opened876_selection_unchanged": True,
            "original_truth_bytes_preserved": True, "section": section, "counts": spec["counts"],
            "computational_selection": selection,
            "array_source_ids": ids, "selected_truth_source_ids": selected,
            "omitted_truth_sources": len(injected) - len(selected),
            "full180_metadata_verified": True, "prepared_fsc_verified": False,
            "dependencies": {str(path): value for path, value in dependencies.items()}}


def provision_truths(section, specification=SPECIFICATION, truth_specification=TRUTH_SPECIFICATION, destination=None):
    root = destination_path(Path(destination) if destination is not None else ROOT / "dump/cloud-opened")
    withdraw_truths(root)
    try:
        if section not in {"1st", "2nd"}:
            raise ValueError("Opened truth section must be 1st or 2nd")
        manifest, _, dependencies = verified_inputs(specification, root)
        data, _, packet_dependencies = packet(specification)
        dependencies.update(packet_dependencies)
        spec, archive, truth_dependencies = truth_packet(specification, truth_specification, data)
        dependencies.update(truth_dependencies)
        values = truth_members(archive, spec)
        checkpoint(root, data, sum(map(len, values.values())) + 512 * 1024)
        folder = root / "truths"
        guard_path(root, folder)
        folder.mkdir(exist_ok=True)
        truth_cache(root)
        for name, payload in values.items():
            path = folder / name
            guard_path(root, path.parent)
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                checked_file(path, next(item for item in spec["members"] if item["path"] == name))
            else:
                publish(root, path, payload)
        _, written = truth_files(root, spec)
        dependencies.update(written)
        receipt = truth_receipt(spec, list(manifest["sources"]), section, dependencies, manifest["computational_selection"])
        recheck(dependencies)
        payload = serialized(receipt)
        expected = publish(root, folder / "receipt.json", payload)
        recheck({**dependencies, folder / "receipt.json": expected})
        return receipt
    except BaseException:
        withdraw_truths(root)
        raise


def prepared_truth_views(out, row, original, raw):
    from deterministic import run as runner

    identities, unavailable = {}, {}
    for view in runner.DETECTOR_VIEWS:
        try:
            records = runner.prepare_records(raw, view)
        except (runner.InsufficientCadences, KeyError) as exc:
            unavailable[view] = type(exc).__name__
            continue
        item = row["inputs"].get(view, {})
        path = out / "inputs" / f"{row['source_id']}.{view}.fsc"
        guard_path(out, path)
        expected = {"bytes": 32 + records.nbytes, "sha256": item.get("fsc_sha256")}
        checked_file(path, expected)
        if (set(item) != {"view", "retained_cadences", "fsc_path", "fsc_sha256"}
            or item.get("view") != view or item.get("fsc_path") != str(path)
            or type(item.get("retained_cadences")) is not int
            or item.get("retained_cadences") != len(records)
            or runner.read_fsc(path, row["source_id"], view).tobytes() != records.tobytes()):
            raise ValueError("Opened truth FSC view, source or exact prepared records changed")
        if view == runner.DISCOVERY_VIEW and hashlib.sha256(records.tobytes()).hexdigest() != original["prepared_records_sha256"]:
            raise ValueError("Opened truth discovery FSC differs from the original oracle")
        identities[path] = expected["sha256"]
    if (row.get("unavailable_views") != unavailable
        or set(row["inputs"]) != set(runner.DETECTOR_VIEWS) - set(unavailable)):
        raise ValueError("Opened truth FSC available/unavailable view accounting changed")
    return identities


def truth_preparation(preparation, root, manifest, rows):
    from deterministic import run as runner

    out = destination_path(Path(preparation).absolute())
    path = out / "preparation.json"
    guard_path(out, path)
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError("Opened truth preparation must be a bounded regular report")
    payload = path.read_bytes()
    report = truth_json(payload)
    ids = list(manifest["sources"])
    if (type(report.get("schema")) is not int or report["schema"] != 2
        or report["status"] != "PREPARED" or report.get("evaluation_scope") is not None
        or report.get("input_manifest") != {"path": str(root / "input-manifest.json"), "sha256": digest(root / "input-manifest.json")}
        or report["selection"]["source_ids"] != ids or [row["source_id"] for row in report["sources"]] != ids
        or report["preparation_counts"]["attempted_sources"] != len(ids)
        or report["preparation_counts"]["accepted_sources"] != len(ids)
        or report["preparation_counts"]["ineligible_sources"] != 0):
        raise ValueError("Opened truth preparation changed its supplied prefix or development scope")
    if report.get("implementation") != runner.implementation_hashes() or report.get("detector_contract") != runner.detector_contract():
        raise ValueError("Opened truth preparation implementation or contract changed")
    dependencies = {path: hashlib.sha256(payload).hexdigest()}
    dependencies.update({runner.ROOT / name: expected for name, expected in report["implementation"].items()})
    dependencies.update({runner.FOLDER / name: expected for name, expected in report["detector_contract"]["files"].items()})
    dependencies.update({runner.ROOT / "pipeline/shallow" / name: expected for name, expected in report["detector_contract"]["shallow_files"].items()})
    for row in report["sources"]:
        sid = row["source_id"]
        original = rows[sid]
        if row.get("status") != "PREPARED" or row["parent_tpf"] != original["parents"]["origin"] or row["raw_fits_sha256"] != original["sha256"]:
            raise ValueError("Opened truth preparation parent or raw identity changed")
        raw = npz_values(root / f"{sid}.npz", original["arrays"])
        dependencies.update(prepared_truth_views(out, row, original, raw))
    counts = {"attempted_sources": len(ids), "accepted_sources": len(ids), "ineligible_sources": 0,
              "prepared_views": sum(len(row["inputs"]) for row in report["sources"]),
              "unavailable_views": sum(len(row["unavailable_views"]) for row in report["sources"])}
    if report["preparation_counts"] != counts:
        raise ValueError("Opened truth preparation view denominators changed")
    for value in report["preparation_counts"].values():
        integer(value)
    return dependencies


def truth_refs(preparation, specification=SPECIFICATION, truth_specification=TRUTH_SPECIFICATION, destination=None):
    root = destination_path(Path(destination) if destination is not None else ROOT / "dump/cloud-opened")
    try:
        manifest, rows, dependencies = verified_inputs(specification, root)
        data, _, packet_dependencies = packet(specification)
        dependencies.update(packet_dependencies)
        spec, _, truth_dependencies = truth_packet(specification, truth_specification, data)
        dependencies.update(truth_dependencies)
        refs, written = truth_files(root, spec)
        dependencies.update(written)
        path = root / "truths/receipt.json"
        guard_path(root, path)
        if not path.is_file() or path.stat().st_size > 512 * 1024:
            raise ValueError("Opened truth success receipt must be a bounded regular file")
        payload = path.read_bytes()
        receipt = truth_json(payload)
        expected = truth_receipt(spec, list(manifest["sources"]), receipt.get("section"), dependencies, manifest["computational_selection"])
        if receipt != expected or payload != serialized(expected) or receipt.get("section") not in {"1st", "2nd"}:
            raise ValueError("Opened truth receipt or closed dependencies changed")
        dependencies[path] = hashlib.sha256(payload).hexdigest()
        dependencies.update(truth_preparation(preparation, root, manifest, rows))
        recheck(dependencies)
        return {sid: refs[sid] for sid in manifest["sources"] if sid in refs}, dependencies
    except BaseException:
        withdraw_truths(root)
        raise
