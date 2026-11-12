"""Replay exact opened physical pixels in a separate six-source experiment."""

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from deterministic.cloud_public import CONTROL_COHORTS, verify_inventory
from deterministic.pixel_injections import deterministic_fits, inject_flux
from runtime import EXTERNAL_DUMPS

FLAGS = ("automatic_pc_enabled", "joint_independence_complete", "scientific_acceptance")
NAMES = ("input-manifest.json", "source-ids.json", "receipt.json")
LIMIT = 1024**3
S3 = "https://stpubdata.s3.us-east-1.amazonaws.com/kepler/public/target_pixel_files"
PRF = "https://archive.stsci.edu/missions/kepler/fpc/prf"
MAX_TPF_BYTES = 128 * 1024**2
PIXEL_FIELDS = {
    "path", "sha256", "base_path", "base_parent_kic", "base_sha256", "prf_calibration",
    "target_pixel", "target_inside_stamp", "target_sky_deg", "target_astrometry",
    "source_flux_electrons_per_second", "maximum_relative_loss", "prf_stamp_fraction",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def size(value, maximum):
    return type(value) is int and 0 < value <= maximum


def checked_file(path, identity):
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == identity["bytes"] and digest(path) == identity["sha256"],
            f"Pinned pixel input changed: {path}")


def guard_path(path, root, file=True):
    require(path == root or root in path.parents, "Injected pixel path escapes its experiment")
    require(".." not in path.relative_to(root).parts, "Injected pixel path escapes its experiment")
    for item in (path, *path.parents):
        require(not item.is_symlink(), f"Injected pixel path must not redirect: {item}")
        expected_file = item == path and file
        require(not item.exists() or (item.is_file() if expected_file else item.is_dir()),
                f"Injected pixel path has the wrong file type: {item}")
        if item == root:
            break


def begin(controls):
    require(controls.is_dir() and not controls.is_symlink(), "Verified public-controls directory required")
    folder = controls / "injected-pixels"
    guard_path(folder, controls, file=False)
    withdraw(folder, controls)
    if folder.exists():
        require(not any(path.is_symlink() for path in folder.rglob("*")),
                "Injected pixel outputs must not redirect their writes")
    for name in NAMES:
        path = folder / name
        guard_path(path, controls)
        guard_path(path.with_suffix(path.suffix + ".part"), controls)
    require(not any(path.name.endswith((".part", ".tmp")) for path in folder.rglob("*")),
            "Stale injected pixel partial; inspect and remove it before retrying")
    folder.mkdir(exist_ok=True)
    return folder


def withdraw(folder, controls):
    if controls.is_symlink() or not controls.is_dir() or folder.is_symlink() or not folder.is_dir():
        return
    for name in NAMES:
        path = folder / name
        if not path.is_symlink() and path.is_file():
            path.unlink()


def checkpoint(folder, growth):
    path = folder.resolve()
    external = any(root == path or root in path.parents for root in EXTERNAL_DUMPS)
    cloud = os.environ.get("SPACELAB_EXECUTION_PROFILE") == "cloud-development"
    reserve = 512 * 1024**2 if cloud and not external else 20 * 1024**3
    require(shutil.disk_usage(path).free >= reserve + growth,
            f"Insufficient space for exact pixel replay and {reserve} byte reserve")


def check_source(sid, source, public):
    require(sid in public and source["source_id"] == sid, "Injected pixel source is outside public controls")
    declared = public[sid]
    require(source["opened_cohort"] in {"periodic_planet", "shallow_planet_extra"}
            and all(source[key] == declared[key] for key in ("opened_cohort", "opened_scope"))
            and source["opened_scope"] == CONTROL_COHORTS[source["opened_cohort"]][1]
            and source["origin_parent_kic"] == declared["metadata"]["origin_tpf_kic"]
            and source["raw_fits_sha256"] == declared["metadata"]["raw_fits_sha256"],
            "Injected pixel raw, parent, cohort or scope binding changed")
    require(all(source[key] == declared["origin"][key]
                for key in ("original_manifest", "truth_reference", "generator_reference")),
            "Original opened pixel input provenance changed")


def check_quarter(sid, quarter, row, specification):
    from lkprf.utils import channel_to_module_output

    require(re.fullmatch(r"[0-9]|1[0-7]", quarter) and type(row["base"]["quarter"]) is int
            and row["base"]["quarter"] == int(quarter),
            "Original injected pixel quarter changed")
    require(row["recipe"] == f"{sid}/q{int(quarter):02d}.npz", "Injected pixel recipe path changed")
    base, pixel = row["base"], row["original_pixel_record"]
    require(set(pixel) == PIXEL_FIELDS and type(pixel["target_inside_stamp"]) is bool,
            "Original pixel record fields or target-stamp state changed")
    require(type(base["parent_kic"]) is int and base["parent_kic"] > 0
            and type(base["channel"]) is int and 1 <= base["channel"] <= 84,
            "Invalid base TPF parent or channel")
    parent = f'{base["parent_kic"]:09d}'
    require(re.fullmatch(rf"kplr{parent}-[0-9]{{13}}_lpd-targ\.fits\.gz", base["filename"])
            and base["url"] == f'{S3}/{parent[:4]}/{parent}/{base["filename"]}',
            "Base TPF URL is not its exact canonical parent product")
    require(pixel["path"] == f"pixels/{sid}/q{int(quarter):02d}.fits.gz"
            and pixel["sha256"] == row["injected"]["sha256"]
            and pixel["base_parent_kic"] == base["parent_kic"] and pixel["base_sha256"] == base["sha256"],
            "Original injected pixel metadata binding changed")
    calibration = pixel["prf_calibration"]
    module, output = channel_to_module_output(base["channel"])
    require(calibration["filename"] in specification["prf_calibrations"]
            and calibration["filename"] == f"kplr{module:02}.{output}_2011265_prf.fits"
            and calibration["sha256"] == specification["prf_calibrations"][calibration["filename"]]["sha256"],
            "Original PRF calibration binding changed")


def check_specification(specification, archive, public, public_path):
    require(type(specification.get("schema")) is int and specification["schema"] == 1
            and specification.get("status") == "EXACT_ORIGINAL_OPENED_INJECTED_PIXEL_REPLAY"
            and specification.get("scope") == "OPENED_DEVELOPMENT_ONLY"
            and all(specification.get(key) is False for key in FLAGS)
            and specification["public_controls_sha256"] == digest(public_path),
            "Opened injected pixel scientific boundary changed")
    require(specification["archive"]["path"] == archive.name
            and size(specification["archive"]["bytes"], 2 * 1024**2), "Invalid bounded recipe archive")
    sources, files, calibrations = specification["sources"], specification["files"], specification["prf_calibrations"]
    require(len(sources) == 6 and len(files) == 88 and len(calibrations) == 12,
            "Exact original six-source 88-quarter inventory required")
    require(list(sources) == [sid for sid in public if sid in sources]
            and Counter(source["opened_cohort"] for source in sources.values()) == {"periodic_planet": 4, "shallow_planet_extra": 2},
            "Already selected injected source order or cohort counts changed")
    quarters = [row for source in sources.values() for row in source["quarters"].values()]
    require(len(quarters) == 88 and {row["recipe"] for row in quarters} == set(files),
            "Original injected source-quarter recipe inventory changed")
    for sid, source in sources.items():
        check_source(sid, source, public)
        for quarter, row in source["quarters"].items():
            check_quarter(sid, quarter, row, specification)
    check_inventory(specification)


def check_inventory(specification):
    files, calibrations = specification["files"], specification["prf_calibrations"]
    quarters = [row for source in specification["sources"].values() for row in source["quarters"].values()]
    identities = list(files.values()) + list(calibrations.values())
    identities += [row[key] for row in quarters for key in ("base", "injected")]
    require(all(size(row["bytes"], 32 * 1024**2) and valid_hash(row["sha256"]) for row in identities),
            "Invalid bounded original pixel byte identities")
    for row in quarters:
        check_array_descriptor(row["base"]["flux"], ">f4", 3, 32 * 1024**2)
        factor_rows = files[row["recipe"]]["arrays"]
        require(set(factor_rows) == {"profile", "loss", "amplitude"}, "Physical recipe arrays changed")
        for name, dimensions in (("profile", 2), ("loss", 1), ("amplitude", 0)):
            check_array_descriptor(factor_rows[name], "float64", dimensions, 2 * 1024**2)
        require(factor_rows["profile"]["shape"] == row["base"]["flux"]["shape"][1:]
                and factor_rows["loss"]["shape"] == row["base"]["flux"]["shape"][:1],
                "Preflight recipe and TPF dimensions differ")
    for name, row in calibrations.items():
        require(re.fullmatch(r"kplr[0-9]{2}\.[1-4]_2011265_prf\.fits", name)
                and row["filename"] == name and row["url"] == f"{PRF}/{name}",
                "PRF URL differs from exact original calibration")
    actual = {
        "sources": len(specification["sources"]), "quarters": len(quarters),
        "recipe_bytes": sum(row["bytes"] for row in files.values()),
        "base_pixel_bytes": sum(row["base"]["bytes"] for row in quarters),
        "injected_pixel_bytes": sum(row["injected"]["bytes"] for row in quarters),
        "prf_files": len(calibrations), "prf_bytes": sum(row["bytes"] for row in calibrations.values()),
    }
    require(actual == specification["counts"] and all(type(value) is int for value in specification["counts"].values())
            and actual["recipe_bytes"] <= 8 * 1024**2
            and actual["base_pixel_bytes"] + actual["injected_pixel_bytes"] + actual["prf_bytes"] <= LIMIT,
            "Exact pixel replay counts or cache bound changed")


def check_array_descriptor(row, dtype, dimensions, maximum):
    require(set(row) == {"dtype", "shape", "bytes", "sha256"}
            and row["dtype"] == dtype and isinstance(row["shape"], list)
            and len(row["shape"]) == dimensions and all(type(value) is int and 0 < value <= 20000 for value in row["shape"])
            and size(row["bytes"], maximum) and valid_hash(row["sha256"]),
            "Invalid bounded original pixel array descriptor")
    require(int(np.prod(row["shape"], dtype=np.int64)) * np.dtype(dtype).itemsize == row["bytes"],
            "Original pixel array byte count differs from its shape")


def factors(payload, identity):
    require(len(payload) == identity["bytes"] and hashlib.sha256(payload).hexdigest() == identity["sha256"],
            "Original injection recipe bytes changed")
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        members = bundle.infolist()
        require(len(members) == 3 and {row.filename for row in members} == {f"{name}.npy" for name in ("profile", "loss", "amplitude")}
                and all(size(row.file_size, 2 * 1024**2) for row in members), "Recipe NPZ inventory is not closed and bounded")
        for member in members:
            recipe_array_header(bundle, member, identity["arrays"][member.filename.removesuffix(".npy")])
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    require(set(identity["arrays"]) == {"profile", "loss", "amplitude"}, "Recipe array declarations changed")
    verify_inventory(arrays, identity["arrays"])
    require(all(array.dtype == np.dtype("float64") and np.isfinite(array).all() for array in arrays.values())
            and arrays["profile"].ndim == 2 and arrays["loss"].ndim == 1 and arrays["amplitude"].ndim == 0,
            "Physical pixel factors must preserve finite float64 shapes")
    return arrays


def recipe_array_header(bundle, member, identity):
    require(stat.S_IFMT(member.external_attr >> 16) in {0, stat.S_IFREG}, "Recipe NPZ contains a nonregular member")
    with bundle.open(member) as stream:
        require(np.lib.format.read_magic(stream) == (1, 0), "Original recipe array format changed")
        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
        require(list(shape) == identity["shape"] and str(dtype) == identity["dtype"] and order is False
                and stream.tell() + identity["bytes"] == member.file_size,
                "Recipe array header dtype, shape, order or bounded bytes changed")


def recipe_archive(archive, specification):
    checked_file(archive, specification["archive"])
    recipes = {}
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            require(member.isfile() and member.name in specification["files"] and member.name not in recipes
                    and member.size == specification["files"][member.name]["bytes"],
                    "Recipe archive contains an unknown, doubled or nonregular member")
            payload = bundle.extractfile(member).read(member.size + 1)
            factors(payload, specification["files"][member.name])
            recipes[member.name] = payload
    require(set(recipes) == set(specification["files"]), "Recipe archive inventory is incomplete")
    return recipes


def publish(path, payload, root):
    partial = path.with_suffix(path.suffix + ".part")
    guard_path(path, root)
    guard_path(partial, root)
    require(not partial.exists(), f"Stale pixel partial; inspect before retrying: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
        partial.chmod(0o444)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def materialize(path, payload, identity, root):
    require(len(payload) == identity["bytes"] and hashlib.sha256(payload).hexdigest() == identity["sha256"],
            "Copied verified pixel or array payload changed")
    guard_path(path, root)
    if path.exists():
        checked_file(path, identity)
    else:
        publish(path, payload, root)
    path.chmod(0o444)


def download(path, row, root, metrics):
    guard_path(path, root)
    partial = path.with_suffix(path.suffix + ".part")
    guard_path(partial, root)
    require(not partial.exists(), f"Stale pixel download partial; inspect before retrying: {partial}")
    if path.exists():
        checked_file(path, row)
        path.chmod(0o444)
        metrics["cache_hits"] += 1
        return
    checkpoint(root, row["bytes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        transfer(path, partial, row, metrics)
    finally:
        metrics["network_seconds"] += time.monotonic() - started
        partial.unlink(missing_ok=True)
    path.chmod(0o444)


def transfer(path, partial, row, metrics):
    with requests.get(row["url"], stream=True, timeout=(15, 120), allow_redirects=False) as response:
        response.raise_for_status()
        require(response.status_code == 200 and response.url == row["url"], "Exact pixel product URL or response changed")
        length = response.headers.get("Content-Length")
        require(length is None or int(length) == row["bytes"], "Exact pixel product Content-Length changed")
        metrics["network_calls"] += 1
        count = 0
        with partial.open("xb") as stream:
            for block in response.iter_content(1024 * 1024):
                count += len(block)
                require(count <= row["bytes"], "Pixel product transfer exceeds its pinned size")
                metrics["network_bytes"] += len(block)
                stream.write(block)
    checked_file(partial, row)
    partial.replace(path)


def header_identity(hdus, base, sid=None):
    hdus.verify("exception")
    header = hdus[0].header
    require(all(type(header.get(key)) is int and header.get(key) == value for key, value in (
        ("QUARTER", base["quarter"]), ("KEPLERID", base["parent_kic"]), ("CHANNEL", base["channel"]),
    )), "TPF quarter, parent or channel header changed")
    if sid is not None:
        require(header.get("FSCSRC") == sid and header.get("FSCMODEL") == "physical-prf-v1",
                "Injected TPF scientific identity headers changed")


def tpf_hdus(path):
    with gzip.open(path, "rb") as stream:
        payload = stream.read(MAX_TPF_BYTES + 1)
    require(len(payload) <= MAX_TPF_BYTES, "TPF exceeds bounded uncompressed pixel memory")
    return fits.open(io.BytesIO(payload), memmap=False)


def tpf_dimensions(hdus, identity):
    header = hdus[1].header
    require(type(header["NAXIS1"]) is int and size(header["NAXIS1"], MAX_TPF_BYTES)
            and type(header["NAXIS2"]) is int and header["NAXIS2"] == identity["shape"][0]
            and header["NAXIS1"] * header["NAXIS2"] <= MAX_TPF_BYTES
            and header.get("XTENSION") == "BINTABLE" and header.get("PCOUNT") == 0,
            "TPF table exceeds frozen row dimensions or memory bound")
    require(type(header.get("TFIELDS")) is int and size(header["TFIELDS"], 64), "Invalid bounded TPF columns")
    widths = [column_width(header[f"TFORM{index}"], header["NAXIS1"])
              for index in range(1, header["TFIELDS"] + 1)]
    require(sum(widths) == header["NAXIS1"], "TPF column storage differs from its bounded row width")
    for name in ("FLUX", "FLUX_ERR"):
        indices = [index for index in range(1, header["TFIELDS"] + 1) if header.get(f"TTYPE{index}") == name]
        require(len(indices) == 1, "Original TPF flux column inventory changed")
        index = indices[0]
        rows, columns = identity["shape"][1:]
        require(header.get(f"TFORM{index}") == f"{rows * columns}E"
                and header.get(f"TDIM{index}") == f"({columns},{rows})",
                "TPF flux column dimensions differ from frozen array inventory")


def column_width(form, maximum):
    match = re.fullmatch(r"([0-9]*)([LXBIJKAEDCM])", form)
    require(match is not None, "Original TPF must use bounded fixed-width columns")
    repeat = int(match[1] or "1")
    require(0 <= repeat <= maximum * 8, "TPF column repeat exceeds bounded row width")
    widths = {"L": 1, "X": 0, "B": 1, "I": 2, "J": 4, "K": 8, "A": 1, "E": 4, "D": 8, "C": 8, "M": 16}
    return (repeat + 7) // 8 if match[2] == "X" else repeat * widths[match[2]]


def replay(base_path, output, sid, row, arrays, root):
    checked_file(base_path, row["base"])
    guard_path(output, root)
    partial = output.with_suffix(output.suffix + ".part")
    temporary = partial.with_suffix(partial.suffix + ".tmp")
    guard_path(partial, root)
    guard_path(temporary, root)
    require(not partial.exists() and not temporary.exists(), "Stale injected TPF partial; inspect before retrying")
    with tpf_hdus(base_path) as hdus:
        tpf_dimensions(hdus, row["base"]["flux"])
        header_identity(hdus, row["base"])
        flux = hdus[1].data["FLUX"]
        verify_inventory({"flux": flux}, {"flux": row["base"]["flux"]})
        require(arrays["profile"].shape == flux.shape[1:] and arrays["loss"].shape == flux.shape[:1],
                "Recipe factors do not match original TPF dimensions")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            original_error = hdus[1].data["FLUX_ERR"].copy()
            hdus[1].data["FLUX"] = inject_flux(flux, arrays["profile"], arrays["loss"], arrays["amplitude"].item())
            require(np.array_equal(hdus[1].data["FLUX_ERR"], original_error, equal_nan=True), "Pixel errors changed during replay")
            hdus[0].header["FSCSRC"] = (sid, "Injected Gaia DR3 source")
            hdus[0].header["FSCMODEL"] = ("physical-prf-v1", "Forgotten Stars control")
            deterministic_fits(partial, hdus)
            checked_file(partial, row["injected"])
            partial.chmod(0o444)
            partial.replace(output)
    checked_file(output, row["injected"])
    with tpf_hdus(output) as hdus:
        tpf_dimensions(hdus, row["base"]["flux"])
        header_identity(hdus, row["base"], sid)
        require(all(hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1 for hdu in hdus),
                "Injected TPF checksums changed")
    output.chmod(0o444)


def record_paths(folder, specification):
    paths = []
    for sid, source in specification["sources"].items():
        paths.append(folder / f"{sid}.npz")
        for row in source["quarters"].values():
            output = folder / row["original_pixel_record"]["path"]
            paths.extend((output, folder / "base" / row["base"]["filename"], folder / "recipes" / row["recipe"]))
            paths.append(output.with_suffix(output.suffix + ".part.tmp"))
    paths.extend(folder / "prf" / name for name in specification["prf_calibrations"])
    for path in paths:
        guard_path(path, folder)
        guard_path(path.with_suffix(path.suffix + ".part"), folder)
    return paths


def experiment(folder, specification, original):
    sources = {}
    for sid, declared in specification["sources"].items():
        source = dict(original["sources"][sid])
        source["pixels"] = {}
        for quarter, row in declared["quarters"].items():
            pixel = dict(row["original_pixel_record"])
            pixel["base_path"] = str(folder / "base" / row["base"]["filename"])
            source["pixels"][quarter] = pixel
        sources[sid] = source
    return {"schema": 1, "measurement_scope": "OPENED_DEVELOPMENT_ONLY", "role": "development",
            "survey_context_complete": False, **{key: False for key in FLAGS}, "sources": sources,
            "boundary": specification["boundary"]}


def stable_inputs(identities):
    require(all(path.is_file() and not path.is_symlink() and digest(path) == expected
                for path, expected in identities.items()), "Pixel replay dependency changed during provisioning")


def provision(specification_path, archive, public_path, controls, load_controls):
    folder = controls / "injected-pixels"
    try:
        begin(controls)
        return provision_checked(specification_path, archive, public_path, controls, folder, load_controls)
    except BaseException:
        withdraw(folder, controls)
        raise


def provision_checked(specification_path, archive, public_path, controls, folder, load_controls):
    started, cpu = time.monotonic(), time.process_time()
    initial_bytes = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    require(initial_bytes <= LIMIT, "Existing injected pixel cache exceeds its bound")
    require(specification_path.is_file() and not specification_path.is_symlink()
            and size(specification_path.stat().st_size, 2 * 1024**2), "Bounded regular injected pixel specification required")
    payload = specification_path.read_bytes()
    specification = json.loads(payload)
    canonical(specification)
    original, public, dependencies = load_controls()
    dependencies = dict(dependencies)
    dependencies[specification_path] = hashlib.sha256(payload).hexdigest()
    dependencies[archive] = specification["archive"]["sha256"]
    dependencies[public_path] = specification["public_controls_sha256"]
    dependencies[Path(__file__)] = digest(__file__)
    dependencies[Path(deterministic_fits.__code__.co_filename)] = digest(deterministic_fits.__code__.co_filename)
    dependencies[Path(verify_inventory.__code__.co_filename)] = digest(verify_inventory.__code__.co_filename)
    check_specification(specification, archive, public, public_path)
    record_paths(folder, specification)
    checkpoint(folder, LIMIT)
    recipes = recipe_archive(archive, specification)
    metrics = {"network_calls": 0, "network_bytes": 0, "network_seconds": 0.0, "cache_hits": 0}
    for name, row in specification["prf_calibrations"].items():
        download(folder / "prf" / name, row, folder, metrics)
        dependencies[folder / "prf" / name] = row["sha256"]
    pixel_hashes = {}
    for sid, source in specification["sources"].items():
        path = controls / f"{sid}.npz"
        materialize(folder / path.name, path.read_bytes(), {"bytes": path.stat().st_size, "sha256": original["sources"][sid]["sha256"]}, folder)
        dependencies[folder / path.name] = original["sources"][sid]["sha256"]
        for row in source["quarters"].values():
            materialize(folder / "recipes" / row["recipe"], recipes[row["recipe"]], specification["files"][row["recipe"]], folder)
            dependencies[folder / "recipes" / row["recipe"]] = specification["files"][row["recipe"]]["sha256"]
            base = folder / "base" / row["base"]["filename"]
            download(base, row["base"], folder, metrics)
            dependencies[base] = row["base"]["sha256"]
            output = folder / row["original_pixel_record"]["path"]
            replay(base, output, sid, row, factors(recipes[row["recipe"]], specification["files"][row["recipe"]]), folder)
            dependencies[output] = row["injected"]["sha256"]
            pixel_hashes[str(output.relative_to(folder))] = row["injected"]["sha256"]
    return finish(folder, specification, original, dependencies, pixel_hashes, metrics, started, cpu, initial_bytes)


def finish(folder, specification, original, dependencies, pixel_hashes, metrics, started, cpu, initial_bytes):
    stable_inputs(dependencies)
    for name, row in specification["prf_calibrations"].items():
        checked_file(folder / "prf" / name, row)
    for name, expected in pixel_hashes.items():
        require(digest(folder / name) == expected, "Injected pixels changed before experiment handoff")
    manifest = experiment(folder, specification, original)
    payloads = (json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n",
                canonical(list(manifest["sources"])) + b"\n")
    for name, payload in zip(NAMES[:2], payloads):
        path = folder / name
        dependencies[path] = hashlib.sha256(payload).hexdigest()
        publish(path, payload, folder)
        checked_file(path, {"bytes": len(payload), "sha256": dependencies[path]})
    metrics.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
                   disk_growth_bytes=sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) - initial_bytes,
                   peak_pixel_memory_estimate_bytes=max(row["base"]["flux"]["bytes"] for source in specification["sources"].values()
                                                        for row in source["quarters"].values()) * 12 + MAX_TPF_BYTES,
                   uncompressed_tpf_ceiling_bytes=MAX_TPF_BYTES)
    result = {
        "status": "EXACT_OPENED_INJECTED_PIXELS_VERIFIED", "scope": "OPENED_DEVELOPMENT_ONLY", "role": "development",
        "counts": specification["counts"], "cohort_sources": dict(Counter(source["opened_cohort"] for source in specification["sources"].values())),
        "source_scopes": {sid: source["opened_scope"] for sid, source in specification["sources"].items()},
        "config": {"input_manifest": str(folder / NAMES[0]), "frozen_pixel_inputs": None, "prf_cache_directory": str(folder / "prf")},
        "manifest_sha256": dependencies[folder / NAMES[0]], "source_ids_sha256": dependencies[folder / NAMES[1]],
        "dependencies": {str(path): expected for path, expected in dependencies.items()}, "injected_pixel_sha256": pixel_hashes,
        "raw_fits_sha256": {sid: source["raw_fits_sha256"] for sid, source in specification["sources"].items()},
        "base_products": {row["base"]["filename"]: row["base"] for source in specification["sources"].values()
                          for row in source["quarters"].values()},
        "prf_calibrations": specification["prf_calibrations"], "recipes": specification["files"], "origin": specification["origin"],
        "program_sha256": {str(path): expected for path, expected in dependencies.items() if path.suffix == ".py"},
        "metrics": metrics, "survey_context_complete": False, **{key: False for key in FLAGS}, "boundary": specification["boundary"],
    }
    check_handoff(result, manifest, folder, specification)
    stable_inputs(dependencies)
    payload = json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"
    publish(folder / NAMES[2], payload, folder)
    checked_file(folder / NAMES[2], {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    stable_inputs(dependencies)
    return result


def check_handoff(receipt, manifest, folder, specification):
    require(json.loads((folder / NAMES[0]).read_bytes()) == manifest
            and json.loads((folder / NAMES[1]).read_bytes()) == list(manifest["sources"]),
            "Published injected pixel manifest or source IDs changed")
    require(receipt["status"] == "EXACT_OPENED_INJECTED_PIXELS_VERIFIED"
            and receipt["scope"] == manifest["measurement_scope"] == "OPENED_DEVELOPMENT_ONLY"
            and receipt["role"] == manifest["role"] == "development"
            and receipt["survey_context_complete"] is manifest["survey_context_complete"] is False
            and all(receipt[key] is manifest[key] is False for key in FLAGS), "Injected pixel handoff boundary changed")
    require(receipt["config"] == {"input_manifest": str(folder / NAMES[0]), "frozen_pixel_inputs": None,
                                  "prf_cache_directory": str(folder / "prf")}
            and manifest["schema"] == 1 and list(manifest["sources"]) == list(specification["sources"])
            and all(set(row["pixels"]) == set(specification["sources"][sid]["quarters"])
                    for sid, row in manifest["sources"].items()), "Separate six-source pixel handoff changed")
