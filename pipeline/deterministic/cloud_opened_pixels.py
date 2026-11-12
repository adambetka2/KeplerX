"""Transport exact original opened180 pixels through the verified LC2 corpus."""

import copy
import gzip
import hashlib
import io
import re
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from deterministic import cloud_injected_pixels as pixels
from deterministic import cloud_opened as opened
from deterministic import cloud_opened_corpus as corpus

ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION = Path(__file__).with_name("cloud-opened-pixels.json")
COHORTS = {"periodic_planet": (100, 1074), "sparse_injection": (40, 40), "ttv_injection": (40, 426)}
FLAGS = ("scientific_acceptance", "joint_independence_complete", "automatic_pc_enabled",
         "pixel_readiness_complete", "survey_context_complete", "partitions_frozen", "seeds_frozen")
STATUS = "EXACT_ORIGINAL_OPENED180_PIXEL_TRANSPORT"
OWNER = "pixel-owner.json"
SUCCESS = ("receipt.json", "source-ids.json", "config.json")
MAX_METADATA = 16 * 1024**2
MAX_ARCHIVE = 8 * 1024**2
MAX_FILE = 16 * 1024**2
METADATA_HEADROOM = 32 * 1024**2
SHARED_INPUTS = tuple(f"pipeline/deterministic/cloud-opened-{name}" for name in (
    "inputs.json", "inputs.json.gz", "source-metadata.json", "source-metadata.tar.gz",
    "truths.json", "truths.tar.gz"))
PROGRAMS = tuple(f"pipeline/deterministic/{name}.py" for name in (
    "cloud_injected_pixels", "pixel_injections", "injections"))
BOUNDARY = "Exact original opened input transport only; no active FROZEN materialization, new trial, scene closure, independence or scientific readiness."

require = pixels.require


def serialized(value):
    return opened.serialized(value)


def identity(path):
    return {"bytes": Path(path).stat().st_size, "sha256": opened.digest(path)}


def canonical_hash(value):
    return hashlib.sha256(pixels.canonical(value)).hexdigest()


def bounded_identity(item, maximum):
    require(isinstance(item, dict) and pixels.size(item.get("bytes"), maximum)
            and pixels.valid_hash(item.get("sha256")), "Invalid bounded pixel identity")


def file_identity(path, item, dependencies):
    bounded_identity(item, 32_000_000_000)
    opened.checked_file(path, item)
    dependencies[path] = item["sha256"]


def metadata(specification):
    dependencies = {}
    opened.parent_paths(specification)
    require(specification.is_file() and not specification.is_symlink()
            and specification.stat().st_size <= 64 * 1024, "Pixel descriptor must be a bounded regular file")
    payload = specification.read_bytes()
    descriptor = opened.parse(payload)
    dependencies[specification] = hashlib.sha256(payload).hexdigest()
    item = descriptor["artifact"]
    require(item["path"] == "cloud-opened-pixels.json.gz", "Pixel metadata artifact name changed")
    bounded_identity(item, MAX_ARCHIVE)
    require(pixels.size(item["decoded_bytes"], MAX_METADATA) and pixels.valid_hash(item["decoded_sha256"]),
            "Invalid decoded metadata bound")
    path = specification.parent / item["path"]
    file_identity(path, item, dependencies)
    with gzip.open(path, "rb") as stream:
        decoded = stream.read(item["decoded_bytes"] + 1)
    require(len(decoded) == item["decoded_bytes"] and hashlib.sha256(decoded).hexdigest() == item["decoded_sha256"],
            "Pixel decoded metadata changed")
    data = opened.parse(decoded)
    keys = ["schema", "status", "scope", "role", *FLAGS, "counts", "cohorts", "boundary", "archive"]
    require(all(descriptor[key] == data[key] for key in keys), "Pixel descriptor and metadata disagree")
    require(type(data["schema"]) is int and data["schema"] == 1 and data["status"] == STATUS
            and data["scope"] == "OPENED_DEVELOPMENT_ONLY" and data["role"] == "development"
            and data["original_opened876_selection_unchanged"] is True
            and all(data[flag] is False for flag in FLAGS), "Pixel scientific boundary changed")
    require([row["path"] for row in data["shared_inputs"]] == list(SHARED_INPUTS)
            and [row["path"] for row in data["runtime_programs"]] == list(PROGRAMS),
            "Closed pixel dependency inventory changed")
    for row in data["shared_inputs"] + data["runtime_programs"]:
        require(re.fullmatch(r"pipeline/(deterministic|photometry)/[a-zA-Z0-9_.-]+", row["path"]),
                "Pixel dependency path is outside the closed program inputs")
        file_identity(ROOT / row["path"], row, dependencies)
    require(data["archive"]["path"] == "cloud-opened-pixels.tar.gz", "Pixel recipe archive name changed")
    file_identity(specification.parent / data["archive"]["path"], data["archive"], dependencies)
    require(data["archive"]["bytes"] <= MAX_ARCHIVE, "Pixel recipe archive is oversized")
    check_inventory(data)
    opened.recheck(dependencies)
    return data, dependencies


def check_flux(base, recipe):
    flux = base["flux"]
    require(set(flux) == {"dtype", "shape", "bytes", "sha256"} and flux["dtype"] == ">f4"
            and len(flux["shape"]) == 3 and all(pixels.size(n, 10000) for n in flux["shape"])
            and flux["bytes"] == 4 * int(np.prod(flux["shape"])) and flux["bytes"] <= 32 * 1024**2
            and pixels.valid_hash(flux["sha256"]), "Original FLUX dimensions or memory bounds changed")
    expected = {"profile": flux["shape"][1:], "loss": flux["shape"][:1], "amplitude": []}
    require(set(recipe["arrays"]) == set(expected), "Physical factor array inventory changed")
    for name, shape in expected.items():
        row = recipe["arrays"][name]
        count = int(np.prod(shape)) if shape else 1
        require(set(row) == {"dtype", "shape", "bytes", "sha256"} and row["dtype"] == "float64"
                and row["shape"] == shape and type(row["bytes"]) is int and row["bytes"] == count * 8
                and pixels.valid_hash(row["sha256"]), "Physical factor dtype, shape or content declaration changed")


def check_quarter(sid, quarter, row, data):
    base, original = row["base"], row["original_pixel_record"]
    require(re.fullmatch(r"[0-9]|1[0-7]", quarter) and type(base["quarter"]) is int
            and base["quarter"] == int(quarter), "Original quarter changed")
    require(row["recipe"] == f"{sid}-q{int(quarter):02d}.npz" and row["recipe"] in data["files"],
            "Original factor recipe path changed")
    require(set(original) == pixels.PIXEL_FIELDS and type(original["target_inside_stamp"]) is bool,
            "Original pixel metadata fields or stamp state changed")
    require(type(base["parent_kic"]) is int and base["parent_kic"] > 0
            and type(base["channel"]) is int and 1 <= base["channel"] <= 84, "Original pixel parent or channel changed")
    parent = f'{base["parent_kic"]:09d}'
    require(re.fullmatch(rf"kplr{parent}-[0-9]{{13}}_lpd-targ\.fits\.gz", base["filename"])
            and base["url"] == f'{pixels.S3}/{parent[:4]}/{parent}/{base["filename"]}', "Canonical base TPF product changed")
    require(original["path"] == f"pixels/{sid}/q{int(quarter):02d}.fits.gz"
            and original["sha256"] == row["injected"]["sha256"]
            and original["base_sha256"] == base["sha256"] and original["base_parent_kic"] == base["parent_kic"],
            "Original injected pixel binding changed")
    bounded_identity(base, MAX_FILE)
    bounded_identity(row["injected"], MAX_FILE)
    check_flux(base, data["files"][row["recipe"]])
    calibration = original["prf_calibration"]
    require(calibration["filename"] in data["prf_calibrations"]
            and calibration["sha256"] == data["prf_calibrations"][calibration["filename"]]["sha256"],
            "Original PRF calibration binding changed")
    from lkprf.utils import channel_to_module_output

    module, output = channel_to_module_output(base["channel"])
    require(calibration["filename"] == f"kplr{module:02}.{output}_2011265_prf.fits", "Quarter PRF channel changed")


def check_inventory(data):
    order, sources = data["source_order"], data["sources"]
    require(len(order) == len(set(order)) == len(sources) == sum(n[0] for n in COHORTS.values())
            and set(order) == set(sources), "Original180 source order or inventory changed")
    require(data["cohorts"] == {name: {"sources": n[0], "quarters": n[1]} for name, n in COHORTS.items()},
            "Original cohort denominator changed")
    require(Counter(source["opened_cohort"] for source in sources.values()) == {name: n[0] for name, n in COHORTS.items()},
            "Original source cohort counts changed")
    quarters = []
    for sid in order:
        source = sources[sid]
        require(re.fullmatch(r"[1-9][0-9]{18}", sid) and source["source_id"] == sid
                and source["opened_scope"] == "CURRENT_OPENED_DEVELOPMENT"
                and source["original_source_sha256"] == canonical_hash(source["original_source"]),
                "Original source lineage changed")
        require(source["original_source"]["source_id"] == sid
                and source["raw_fits_sha256"] == source["original_source"]["base_raw_fits_sha256"]
                and source["origin_parent_kic"] == source["original_source"]["metadata"]["origin_tpf_kic"]
                and set(source["quarters"]) == set(source["original_source"]["pixels"]), "Original raw/parent/quarter source binding changed")
        for quarter, row in source["quarters"].items():
            require(row["original_pixel_record"] == source["original_source"]["pixels"][quarter], "Original pixel row changed")
            check_quarter(sid, quarter, row, data)
            quarters.append((source, row))
    check_counts(data, quarters)


def check_counts(data, quarters):
    recipes = {row["recipe"] for _, row in quarters}
    ancillary = {row["member"] for lineage in data["lineage"].values() for row in lineage["ancillary"].values()}
    require(set(data["files"]) == recipes | ancillary and not recipes & ancillary, "Factor/lineage member inventory is not closed")
    for name, row in data["files"].items():
        require(re.fullmatch(r"[1-9][0-9]{18}-q[0-1][0-9]\.npz|lineage/(periodic_planet|sparse_injection|ttv_injection)/(gaia-astrometry|repair-identity)\.json", name),
                "Unknown or unsafe recipe archive member")
        bounded_identity(row, 2 * 1024**2)
    for name, row in data["prf_calibrations"].items():
        require(re.fullmatch(r"kplr[0-9]{2}\.[1-4]_2011265_prf\.fits", name)
                and row["url"] == f"{pixels.PRF}/{name}", "Original PRF product changed")
        bounded_identity(row, MAX_FILE)
    expected = {"sources": len(data["sources"]), "quarters": len(quarters), "recipes": len(recipes),
                "ancillary_files": len(ancillary), "prfs": len(data["prf_calibrations"]),
                "nonorigin_parent_quarters": sum(row["base"]["parent_kic"] != source["origin_parent_kic"] for source, row in quarters),
                "outside_stamp_quarters": sum(row["original_pixel_record"]["target_inside_stamp"] is False for _, row in quarters),
                "recipe_bytes": sum(data["files"][name]["bytes"] for name in recipes),
                "ancillary_bytes": sum(data["files"][name]["bytes"] for name in ancillary),
                "base_bytes": sum(row["base"]["bytes"] for _, row in quarters),
                "injected_bytes": sum(row["injected"]["bytes"] for _, row in quarters),
                "prf_bytes": sum(row["bytes"] for row in data["prf_calibrations"].values())}
    require(data["counts"] == expected and all(type(value) is int for value in data["counts"].values()), "Original pixel totals changed")
    require(len(quarters) == sum(n[1] for n in COHORTS.values()), "Original quarter denominator changed")
    require({row["original_pixel_record"]["prf_calibration"]["filename"] for _, row in quarters} == set(data["prf_calibrations"]),
            "Original PRF inventory has missing or extra calibrations")
    require(expected["nonorigin_parent_quarters"] == 73 and expected["outside_stamp_quarters"] == 207
            and expected["prfs"] == 67 and expected["ancillary_files"] == 5, "Original pixel boundary counts changed")
    require(Counter(source["opened_cohort"] for source, _ in quarters) == {name: n[1] for name, n in COHORTS.items()},
            "Original per-cohort quarter counts changed")


def bundle(specification, data):
    path = specification.parent / data["archive"]["path"]
    pixels.checked_file(path, data["archive"])
    contents = {}
    with gzip.open(path, "rb") as stream:
        decoded = stream.read(MAX_METADATA + 1)
    require(len(decoded) <= MAX_METADATA and len(decoded) % 512 == 0, "Recipe TAR decoded size exceeds its bound")
    with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:") as archive:
        for member in archive:
            require(member.isreg() and not member.pax_headers and not member.sparse
                    and member.name in data["files"] and member.name not in contents
                    and member.size == data["files"][member.name]["bytes"], "Recipe archive contains an unsafe, unknown or duplicate member")
            value = archive.extractfile(member).read(member.size + 1)
            row = data["files"][member.name]
            require(len(value) == row["bytes"] and hashlib.sha256(value).hexdigest() == row["sha256"], "Original recipe/ancillary bytes changed")
            if member.name.endswith(".npz"):
                pixels.factors(value, row)
            contents[member.name] = value
        require(not any(decoded[archive.offset:]), "Recipe archive contains orphan bytes after its closed inventory")
    require(set(contents) == set(data["files"]), "Original recipe archive is incomplete")
    return contents


def owner_payload(root):
    return serialized({"schema": 1, "kind": "OPENED_ORIGINAL180_PIXELS_MANAGED", "root": str(root)})


def owned(root):
    path = root / OWNER
    return (root.is_dir() and not root.is_symlink() and path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 1024 and path.read_bytes() == owner_payload(root))


def token(root):
    return (root.stat().st_dev, root.stat().st_ino) if owned(root) else None


def withdraw(root, ownership=None):
    if not root.is_dir() or root.is_symlink():
        return
    current = root.stat()
    if ownership != (current.st_dev, current.st_ino) and not owned(root):
        return
    paths = [root / name for name in SUCCESS]
    paths += [root / "transport" / cohort / "input-manifest.json" for cohort in COHORTS]
    for path in paths:
        if path.is_file() and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)):
            path.unlink()


def begin(root):
    opened.parent_paths(root)
    require(not root.exists() or root.is_dir(), "Pixel output root has the wrong type")
    if root.exists() and any(root.iterdir()):
        require(owned(root), "Occupied pixel destination is not unambiguously owned")
    root.mkdir(parents=True, exist_ok=True)
    if not owned(root):
        pixels.publish(root / OWNER, owner_payload(root), root)
    ownership = token(root)
    withdraw(root, ownership)
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Owned pixel output redirects through a symlink")
        require(path.is_dir() or path.is_file(), "Owned pixel output has an unsupported file type")
        require(not path.name.endswith((".part", ".tmp")), "Stale owned pixel partial requires review")
    return ownership


def context(preparation, opened_cache, corpus_root, data):
    result, dependencies = corpus.verified_corpus(cache=opened_cache, destination=corpus_root)
    refs, truth_dependencies = opened.truth_refs(preparation, destination=opened_cache)
    corpus.bind_dependencies(dependencies, truth_dependencies)
    packet, _, packet_dependencies = opened.packet()
    corpus.bind_dependencies(dependencies, packet_dependencies)
    ids = result["selection"]["selected_source_ids"]
    require(result["original_source_order"] == data["source_order"]
            and ids == data["source_order"][:len(ids)] and set(refs) == set(ids),
            "Verified corpus/truth selection is not an original180 prefix")
    require(data["provenance"]["original_scope_counts"] == packet["counts"], "Original876 denominator changed")
    manifests = original_lineage(corpus_root, data)
    for sid in ids:
        source = data["sources"][sid]
        row = packet["sources"][sid]
        require(row["role"] == "development" and row["opened_scope"] == source["opened_scope"]
                and row["opened_cohort"] == source["opened_cohort"] and row["sha256"] == source["raw_fits_sha256"],
                "Opened source/raw/role binding changed")
        truth_reference(refs[sid], source, data["lineage"][source["opened_cohort"]])
    return result, manifests, packet, dependencies



def truth_reference(reference, source, lineage):
    index = reference["truth_index"]
    require(type(index) is int and index == source["truth"]["original_truth_line_index"]
            and index == source["original_source"]["truth_index"]
            and reference["cohort"] == source["opened_cohort"]
            and reference["file_sha256"] == lineage["truth"]["sha256"], "Original truth reference binding changed")
    path = Path(reference["path"])
    opened.parent_paths(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2, "Original truth reference must be a bounded regular file")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == reference["file_sha256"], "Original truth reference file changed")
    lines, kept = payload.splitlines(), payload.splitlines(keepends=True)
    require(0 <= index < len(lines) and hashlib.sha256(kept[index]).hexdigest() == reference["line_sha256"]
            and hashlib.sha256(lines[index]).hexdigest() == source["truth"]["original_truth_line_sha256"]
            and canonical_hash(opened.parse(lines[index])) == source["truth"]["original_truth_clock_payload_sha256"],
            "Original truth line bytes, line ending or clock changed")

def original_lineage(root, data):
    manifests = {}
    for cohort, lineage in data["lineage"].items():
        require(cohort in COHORTS, "Unknown original pixel cohort lineage")
        folder = root / "metadata/lineage" / cohort
        for name, item in lineage["original_files"].items():
            pixels.checked_file(folder / name, item)
        manifest = opened.parse((folder / "input-manifest.json").read_bytes())
        frozen = opened.parse((folder / "materialization.json").read_bytes())
        ids = opened.parse((folder / "source-ids.json").read_bytes())
        expected = [sid for sid in data["source_order"] if data["sources"][sid]["opened_cohort"] == cohort]
        require(ids == expected == list(manifest["sources"]) and frozen["source_ids"] == ids
                and frozen["status"] == "FROZEN" and frozen["partition"] == "development"
                and manifest["generator"] == lineage["generator"]
                and manifest["truth"] == lineage["truth"] and manifest["gaia_astrometry"] == lineage["gaia_astrometry"],
                "Original inert frozen lineage changed")
        truth_path = folder / "truth.jsonl"
        require(identity(truth_path)["sha256"] == lineage["truth"]["sha256"], "Original truth file changed")
        lines = truth_path.read_bytes().splitlines()
        for sid in ids:
            source = data["sources"][sid]
            require(source["original_source"] == manifest["sources"][sid], "Original source/pixel record changed")
            line = lines[source["truth"]["original_truth_line_index"]]
            require(hashlib.sha256(line).hexdigest() == source["truth"]["original_truth_line_sha256"]
                    and canonical_hash(opened.parse(line)) == source["truth"]["original_truth_clock_payload_sha256"],
                    "Original truth line or clock changed")
        for name, row in lineage["ancillary"].items():
            require(frozen["artifacts"][name] == row["sha256"] and data["files"][row["member"]] == {key: row[key] for key in ("bytes", "sha256")},
                    "Original Gaia/repair artifact binding changed")
        manifests[cohort] = manifest
    require(set(manifests) == set(COHORTS), "Original180 lineage cohort inventory is incomplete")
    return manifests


def select(data, available, source_limit):
    maximum = len(available)
    require(maximum > 0 and available == data["source_order"][:maximum], "Original prefix is empty or reordered")
    if source_limit is not None:
        require(type(source_limit) is int and 1 <= source_limit <= maximum, "Invalid computational prefix limit")
    return available[:source_limit] if source_limit is not None else list(available)


def products(data, ids):
    bases, prfs = {}, {}
    for sid in ids:
        for row in data["sources"][sid]["quarters"].values():
            base = row["base"]
            require(base["filename"] not in bases or bases[base["filename"]] == base, "Conflicting canonical base identities")
            bases[base["filename"]] = base
            name = row["original_pixel_record"]["prf_calibration"]["filename"]
            prfs[name] = data["prf_calibrations"][name]
    return {"base": bases, "prf": prfs}


def retained_paths(mapping, expected, ids, root, dependencies):
    require(isinstance(mapping, dict) and set(mapping) == {"schema", "source_ids", "base", "prf"}
            and type(mapping["schema"]) is int and mapping["schema"] == 1 and mapping["source_ids"] == ids,
            "Retained input mapping must bind the exact original computational prefix")
    resolved = {"base": {}, "prf": {}}
    for kind, rows in expected.items():
        require(set(mapping[kind]) == set(rows), "Retained input mapping is not closed")
        for name, wanted in rows.items():
            row = mapping[kind][name]
            require(set(row) == {"path", "bytes", "sha256"} and all(row[key] == wanted[key] for key in ("bytes", "sha256")),
                    "Retained original identity changed")
            path = Path(row["path"]).resolve(strict=True)
            require(not path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "Retained original input must remain outside owned outputs")
            pixels.checked_file(path, wanted)
            dependencies[path] = wanted["sha256"]
            resolved[kind][name] = path
    return resolved


def outputs(root, data, ids, expected, mode):
    result = {}
    for kind, rows in expected.items():
        if kind == "base" and mode == "retained":
            continue
        for name, row in rows.items():
            result[root / kind / name] = row
    for sid in ids:
        for row in data["sources"][sid]["quarters"].values():
            result[root / row["original_pixel_record"]["path"]] = row["injected"]
            result[root / "recipes" / row["recipe"]] = data["files"][row["recipe"]]
    for lineage in data["lineage"].values():
        for row in lineage["ancillary"].values():
            result[root / row["member"]] = row
    return result


def capacity(root, packet, planned):
    growth = 0
    for path, row in planned.items():
        pixels.guard_path(path, root)
        if path.exists():
            require(path.stat().st_size == row["bytes"], "Existing planned pixel size changed")
        else:
            growth += row["bytes"]
    growth += METADATA_HEADROOM
    budgets = packet["budgets"]
    require(budgets["max_cloud_storage_bytes"] == 32_000_000_000, "Exact32GB packet storage budget required")
    cloud = corpus.cloud_destination(root)
    reserve = budgets["cloud_development_minimum_free_bytes"] if cloud else budgets["external_minimum_free_bytes"]
    headroom = budgets["paired_run_headroom_bytes"]
    usage = shutil.disk_usage(root)
    require(usage.free >= growth + reserve + headroom, "Insufficient storage for remaining pixel growth, reserve and paired headroom")
    if cloud:
        require(usage.total - usage.free + growth + reserve + headroom <= budgets["max_cloud_storage_bytes"],
                "Whole cloud filesystem would exceed exact32GB budget")
    return {"available_bytes": usage.free, "filesystem_used_bytes": usage.total - usage.free,
            "remaining_growth_bytes": growth, "minimum_free_bytes": reserve,
            "paired_headroom_bytes": headroom, "max_cloud_storage_bytes": budgets["max_cloud_storage_bytes"]}


def ancillary_files(root, data, contents, dependencies):
    for cohort, lineage in data["lineage"].items():
        gaia = None
        for name, row in lineage["ancillary"].items():
            path = root / row["member"]
            pixels.materialize(path, contents[row["member"]], row, root)
            dependencies[path] = row["sha256"]
            if name == "gaia-astrometry.json":
                gaia = opened.parse(contents[row["member"]])
        require(gaia is not None and gaia["status"] == "COMPLETE", "Original Gaia response is absent or incomplete")
        rows = {str(row["source_id"]): row for row in gaia["rows"]}
        ids = [sid for sid in data["source_order"] if data["sources"][sid]["opened_cohort"] == cohort]
        require(set(rows) == set(ids) and len(rows) == len(gaia["rows"]), "Original Gaia source inventory changed")
        require(all(canonical_hash(rows[sid]) == data["sources"][sid]["gaia_row_sha256"] for sid in ids),
                "Original Gaia row changed")


def acquire(root, expected, mode, retained, planned, packet, dependencies, metrics, reuse):
    paths = {"base": {}, "prf": {}}
    for kind, rows in expected.items():
        for name, row in rows.items():
            capacity(root, packet, planned)
            path = root / kind / name
            if mode == "download":
                require(not reuse or path.is_file(), "Cached base or PRF input is missing")
                pixels.download(path, row, root, metrics)
            elif kind == "prf":
                source = retained[kind][name]
                pixels.checked_file(source, row)
                pixels.materialize(path, source.read_bytes(), row, root)
                metrics["retained_inputs_verified"] += 1
            else:
                path = retained[kind][name]
                pixels.checked_file(path, row)
                metrics["retained_inputs_verified"] += 1
            dependencies[path] = row["sha256"]
            paths[kind][name] = path
    return paths


def preserved_pixel(output, row):
    with pixels.tpf_hdus(output) as hdus:
        expected = row["preservation"]
        ignore = {"CHECKSUM", "DATASUM", "FSCSRC", "FSCMODEL"}
        headers = [{"hdu": i, "original_noninjection_cards_sha256": canonical_hash(
                    [card.image for card in hdu.header.cards if card.keyword not in ignore])}
                   for i, hdu in enumerate(hdus)]
        require(headers == expected["headers"], "Original noninjection headers changed")
        measured = {}
        for i, hdu in enumerate(hdus):
            if hdu.data is None:
                continue
            names = hdu.columns.names if hasattr(hdu, "columns") else [None]
            for name in names:
                array = np.asarray(hdu.data[name] if name else hdu.data)
                measured[f'{i}:{name or "data"}'] = {"dtype": str(array.dtype), "shape": list(array.shape),
                    "bytes": array.nbytes, "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
        require(measured == {name: item["injected"] for name, item in expected["arrays"].items()},
                "Original error, aperture or other pixel payload changed")


def replay_source(root, sid, data, contents, paths, packet, planned, dependencies, reuse):
    source = data["sources"][sid]
    rows = []
    for quarter, row in source["quarters"].items():
        capacity(root, packet, planned)
        recipe = root / "recipes" / row["recipe"]
        pixels.materialize(recipe, contents[row["recipe"]], data["files"][row["recipe"]], root)
        factors = pixels.factors(recipe.read_bytes(), data["files"][row["recipe"]])
        output = root / row["original_pixel_record"]["path"]
        require(not reuse or output.is_file(), "Cached injected pixel is missing")
        pixels.replay(paths["base"][row["base"]["filename"]], output, sid, row, factors, root)
        preserved_pixel(output, row)
        dependencies[recipe] = data["files"][row["recipe"]]["sha256"]
        dependencies[output] = row["injected"]["sha256"]
        rows.append({"quarter": int(quarter), "pixel_sha256": row["injected"]["sha256"], "recipe_sha256": dependencies[recipe]})
    checkpoint = {"schema": 1, "status": "SOURCE_PIXEL_INPUT_BYTES_VERIFIED", "source_id": sid,
                  "opened_cohort": source["opened_cohort"], "quarters": rows,
                  "original_source_sha256": source["original_source_sha256"],
                  "raw_fits_sha256": source["raw_fits_sha256"], "publication_is_not_complete_transport": True,
                  "program_sha256": {str(path): value for path, value in dependencies.items() if path.suffix == ".py"},
                  **{flag: False for flag in FLAGS}}
    payload = serialized(checkpoint)
    path = root / "checkpoints" / f"{sid}.json"
    pixels.publish(path, payload, root)
    require(path.read_bytes() == payload, "Successful source checkpoint changed during publication")
    dependencies[path] = hashlib.sha256(payload).hexdigest()
    return checkpoint


def mapping_stable(mapping, paths):
    if mapping is None:
        return
    for kind, rows in paths.items():
        for name, path in rows.items():
            require(Path(mapping[kind][name]["path"]).resolve(strict=True) == path,
                    "Retained original input path was redirected during verification")
            pixels.checked_file(path, mapping[kind][name])


def write_bound(root, path, payload, dependencies):
    expected = hashlib.sha256(payload).hexdigest()
    pixels.publish(path, payload, root)
    require(path.read_bytes() == payload, "Published pixel metadata bytes changed")
    dependencies[path] = expected


def transported_manifest(root, corpus_root, template, ids, data, paths):
    result = copy.deepcopy(template)
    result["sources"] = {}
    for sid in ids:
        row = copy.deepcopy(data["sources"][sid]["original_source"])
        row["deltas"] = str(corpus_root / "deltas" / f"{sid}.npz")
        row["metadata"]["shard_path"] = template["sources"][sid]["metadata"]["shard_path"]
        for quarter, pixel in row["pixels"].items():
            original = data["sources"][sid]["quarters"][quarter]
            pixel["path"] = str(root / pixel["path"])
            pixel["base_path"] = str(paths["base"][original["base"]["filename"]])
        result["sources"][sid] = row
    cohort = data["sources"][ids[0]]["opened_cohort"]
    result["truth"] = dict(result["truth"], path=str(corpus_root / "metadata/lineage" / cohort / "truth.jsonl"))
    result["gaia_astrometry"] = dict(result["gaia_astrometry"], path=str(root / "lineage" / cohort / "gaia-astrometry.json"))
    result.update(role="development", **{flag: False for flag in FLAGS},
                  original_source_order=[sid for sid in data["source_order"] if data["sources"][sid]["opened_cohort"] == cohort],
                  computational_source_ids=ids, original_opened876_selection_unchanged=True,
                  full_frozen_materialization_created=False)
    require(result["schema"] == 2 and result["transform"] == "additive_flux_delta", "Original schema2 reader contract changed")
    return result


def manifests(root, corpus_root, original, ids, data, paths, dependencies):
    result, witnesses = {}, {}
    for cohort, filename in original["manifests"].items():
        selected = [sid for sid in ids if data["sources"][sid]["opened_cohort"] == cohort]
        if not selected:
            continue
        path = Path(filename)
        require(path in dependencies, "Verified corpus manifest dependency is absent")
        template = opened.parse(path.read_bytes())
        manifest = transported_manifest(root, corpus_root, template, selected, data, paths)
        output = root / "transport" / cohort / "input-manifest.json"
        write_bound(root, output, serialized(manifest), dependencies)
        result[cohort] = str(output)
        witnesses[cohort] = {
            "original_manifest_sha256": data["lineage"][cohort]["original_files"]["input-manifest.json"]["sha256"],
            "verified_corpus_manifest_sha256": dependencies[path], "transport_manifest_sha256": dependencies[output],
            "source_ids": selected, "relocated_fields": ["deltas", "metadata.shard_path", "pixels.*.path", "pixels.*.base_path", "truth.path", "gaia_astrometry.path"],
            "original_source_sha256": {sid: data["sources"][sid]["original_source_sha256"] for sid in selected}}
    return result, witnesses


def check_handoff(root, receipt, data):
    ids = receipt["selection"]["selected_source_ids"]
    require(receipt["status"] == "ORIGINAL_OPENED_PIXEL_INPUTS_VERIFIED" and receipt["role"] == "development"
            and receipt["scope"] == "OPENED_DEVELOPMENT_ONLY" and all(receipt[flag] is False for flag in FLAGS)
            and receipt["full_frozen_materialization_created"] is False, "Pixel handoff scientific boundary changed")
    require(receipt["counts"] == data["counts"] and ids == data["source_order"][:len(ids)]
            and receipt["original_source_order"] == data["source_order"]
            and opened.parse((root / "source-ids.json").read_bytes()) == ids, "Pixel handoff source order changed")
    config = {"corpus_root": receipt["corpus_root"], "input_manifests": receipt["manifests"],
              "frozen_pixel_inputs": None, "prf_cache_directory": str(root / "prf"),
              "source_ids": str(root / "source-ids.json"), "role": "development", "scope": receipt["scope"],
              **{flag: False for flag in FLAGS}}
    require(receipt["config"] == config and opened.parse((root / "config.json").read_bytes()) == config,
            "Pixel handoff runtime config changed")
    require(set(receipt["manifests"]) == {data["sources"][sid]["opened_cohort"] for sid in ids}, "Pixel handoff cohort inventory changed")
    for cohort, filename in receipt["manifests"].items():
        path = Path(filename)
        require(path == root / "transport" / cohort / "input-manifest.json", "Pixel handoff manifest path changed")
        manifest = opened.parse(path.read_bytes())
        expected = [sid for sid in ids if data["sources"][sid]["opened_cohort"] == cohort]
        require(manifest["schema"] == 2 and manifest["transform"] == "additive_flux_delta"
                and list(manifest["sources"]) == expected and manifest["role"] == "development"
                and all(manifest[flag] is False for flag in FLAGS), "Pixel handoff manifest boundary changed")


def finish(root, data, original, corpus_root, ids, paths, dependencies, metrics, resources, timing, mapping, preparation, cache):
    opened.recheck(dependencies)
    current, witnesses = manifests(root, corpus_root, original, ids, data, paths, dependencies)
    write_bound(root, root / "source-ids.json", serialized(ids), dependencies)
    config = {"corpus_root": str(corpus_root), "input_manifests": current, "frozen_pixel_inputs": None,
              "prf_cache_directory": str(root / "prf"), "source_ids": str(root / "source-ids.json"),
              "role": "development", "scope": "OPENED_DEVELOPMENT_ONLY", **{flag: False for flag in FLAGS}}
    write_bound(root, root / "config.json", serialized(config), dependencies)
    metrics.update(wall_seconds=time.monotonic() - timing[0], cpu_seconds=time.process_time() - timing[1],
                   disk_growth_bytes=disk_bytes(root) - timing[2],
                   quarter_array_and_tpf_bound_bytes=pixels.MAX_TPF_BYTES + 12 * max(
                       row["base"]["flux"]["bytes"] for sid in ids for row in data["sources"][sid]["quarters"].values()))
    result = {"schema": 1, "status": "ORIGINAL_OPENED_PIXEL_INPUTS_VERIFIED", "scope": "OPENED_DEVELOPMENT_ONLY",
              "role": "development", **{flag: False for flag in FLAGS}, "full_frozen_materialization_created": False,
              "original_opened876_selection_unchanged": True, "original_source_order": data["source_order"],
              "counts": data["counts"], "original_scope_counts": data["provenance"]["original_scope_counts"],
              "selection": {"selected_source_ids": ids, "original_injected_sources": 180,
                            "full180_transport_complete": ids == data["source_order"],
                            "verified_corpus_source_ids": original["selection"]["selected_source_ids"],
                            "selected_quarters": sum(len(data["sources"][sid]["quarters"]) for sid in ids)},
              "corpus_root": str(corpus_root), "opened_cache": str(cache), "preparation": str(Path(preparation).resolve()), "config": config, "manifests": current, "relocation_witnesses": witnesses,
              "resources": resources, "metrics": metrics, "mode": metrics["mode"], "retained_input_mapping": mapping,
              "dependencies": {str(path): expected for path, expected in dependencies.items()}, "boundary": BOUNDARY}
    check_handoff(root, result, data)
    opened.recheck(dependencies)
    payload = serialized(result)
    pixels.publish(root / "receipt.json", payload, root)
    require((root / "receipt.json").read_bytes() == payload, "Published pixel receipt bytes changed")
    opened.recheck(dependencies)
    return result


def disk_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def provision(preparation, *, corpus_root, opened_cache, destination, specification=SPECIFICATION,
              mode="download", retained_inputs=None, source_limit=None, reuse=False):
    root = opened.destination_path(Path(destination))
    ownership = token(root)
    try:
        ownership = begin(root)
        require(mode in {"download", "retained"} and type(reuse) is bool, "Unsupported original pixel transport mode")
        require((mode == "retained") == (retained_inputs is not None), "Retained mode requires an explicit closed mapping")
        return provision_checked(root, preparation, Path(corpus_root), Path(opened_cache), Path(specification),
                                 mode, retained_inputs, source_limit, reuse)
    except BaseException:
        withdraw(root, ownership)
        raise


def provision_checked(root, preparation, corpus_root, cache, specification, mode, mapping, source_limit, reuse):
    timing = time.monotonic(), time.process_time(), disk_bytes(root)
    corpus_root, cache = opened.destination_path(corpus_root), opened.destination_path(cache)
    dependencies = {path: opened.digest(path) for path in (Path(__file__), Path(corpus.__file__), Path(opened.__file__), root / OWNER)}
    data, inputs = metadata(specification)
    corpus.bind_dependencies(dependencies, inputs)
    original, _, packet, upstream = context(preparation, cache, corpus_root, data)
    corpus.bind_dependencies(dependencies, upstream)
    ids = select(data, original["selection"]["selected_source_ids"], source_limit)
    contents, expected = bundle(specification, data), products(data, ids)
    mapping = copy.deepcopy(mapping)
    retained = retained_paths(mapping, expected, ids, root, dependencies) if mode == "retained" else {"base": {}, "prf": {}}
    planned = outputs(root, data, ids, expected, mode)
    for path, row in planned.items():
        if path.exists():
            pixels.checked_file(path, row)
    resources = capacity(root, packet, planned)
    metrics = {"mode": mode, "network_calls": 0, "network_bytes": 0, "network_seconds": 0.0,
               "cache_hits": 0, "retained_inputs_verified": 0}
    ancillary_files(root, data, contents, dependencies)
    paths = acquire(root, expected, mode, retained, planned, packet, dependencies, metrics, reuse)
    for sid in ids:
        replay_source(root, sid, data, contents, paths, packet, planned, dependencies, reuse)
    mapping_stable(mapping, retained)
    resources["after"] = capacity(root, packet, planned)
    result = finish(root, data, original, corpus_root, ids, paths, dependencies, metrics, resources, timing, mapping, preparation, cache)
    mapping_stable(mapping, retained)
    opened.recheck(dependencies)
    require((root / "receipt.json").read_bytes() == serialized(result), "Published pixel receipt changed after handoff")
    return result


def verified_pixels(preparation, *, corpus_root, opened_cache, destination, specification=SPECIFICATION):
    root = opened.destination_path(Path(destination))
    ownership = token(root)
    try:
        require(owned(root), "Pixel managed ownership marker absent or changed")
        path = root / "receipt.json"
        pixels.guard_path(path, root)
        require(path.is_file() and path.stat().st_size <= MAX_METADATA, "Pixel success receipt absent or oversized")
        payload = path.read_bytes()
        receipt = opened.parse(payload)
        data, inputs = metadata(Path(specification))
        check_handoff(root, receipt, data)
        require(receipt["corpus_root"] == str(opened.destination_path(Path(corpus_root)))
                and receipt["opened_cache"] == str(opened.destination_path(Path(opened_cache)))
                and receipt["preparation"] == str(Path(preparation).resolve()), "Pixel verified upstream paths changed")
        require(payload == serialized(receipt), "Pixel receipt serialization changed")
        original, _, _, upstream = context(preparation, opened.destination_path(Path(opened_cache)),
                                           opened.destination_path(Path(corpus_root)), data)
        corpus.bind_dependencies(inputs, upstream)
        checked_cache(root, data, receipt, original, inputs)
        return provision(preparation, corpus_root=corpus_root, opened_cache=opened_cache, destination=root,
                         specification=specification, mode=receipt["mode"], retained_inputs=receipt["retained_input_mapping"],
                         source_limit=len(receipt["selection"]["selected_source_ids"]), reuse=True)
    except BaseException:
        withdraw(root, ownership)
        raise


def checked_cache(root, data, receipt, original, dependencies):
    ids, mode = receipt["selection"]["selected_source_ids"], receipt["mode"]
    require(mode in {"download", "retained"} and ids == original["selection"]["selected_source_ids"][:len(ids)],
            "Cached pixel selection or mode changed")
    expected = products(data, ids)
    if mode == "retained":
        retained = retained_paths(receipt["retained_input_mapping"], expected, ids, root, dependencies)
        paths = {"base": retained["base"], "prf": {name: root / "prf" / name for name in expected["prf"]}}
    else:
        require(receipt["retained_input_mapping"] is None, "Downloaded cache acquired an unrelated retained mapping")
        paths = {kind: {name: root / kind / name for name in rows} for kind, rows in expected.items()}
    for path, row in outputs(root, data, ids, expected, mode).items():
        dependencies[path] = row["sha256"]
    for path in (Path(__file__), Path(corpus.__file__), Path(opened.__file__), root / OWNER):
        dependencies[path] = opened.digest(path)
    dependencies[root / "source-ids.json"] = hashlib.sha256(serialized(ids)).hexdigest()
    dependencies[root / "config.json"] = hashlib.sha256(serialized(receipt["config"])).hexdigest()
    for cohort, filename in receipt["manifests"].items():
        selected = [sid for sid in ids if data["sources"][sid]["opened_cohort"] == cohort]
        template = opened.parse(Path(original["manifests"][cohort]).read_bytes())
        wanted = transported_manifest(root, Path(receipt["corpus_root"]), template, selected, data, paths)
        dependencies[Path(filename)] = hashlib.sha256(serialized(wanted)).hexdigest()
    recorded = {Path(name): value for name, value in receipt["dependencies"].items()}
    checkpoints = {root / "checkpoints" / f"{sid}.json" for sid in ids}
    require(set(recorded) == set(dependencies) | checkpoints and all(pixels.valid_hash(value) for value in recorded.values()),
            "Cached pixel dependency inventory is not closed")
    require(all(recorded[path] == value for path, value in dependencies.items()), "Cached pixel bound dependencies changed")
    opened.recheck(recorded)
