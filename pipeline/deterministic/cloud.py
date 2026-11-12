"""Check cloud numerical contracts, public inputs and complete paired evidence."""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import VOLATILE, Differences, compare_runs, run_text

DUMP = ROOT / "dump"
BLS_ORACLE = {
    "bytes": 380262,
    "sha256": "a837d471e7fb7bfd90e26aa998651058b66b550252a322ac36acca0df548aa54",
}
BLS_FIXTURE = ROOT / "pipeline/shallow/native-bls-oracle.npz"
CATALOGUE_BUNDLE = ROOT / "pipeline/vetting/cloud-catalogues.tar.gz"
CATALOGUE_SPECIFICATION = ROOT / "pipeline/vetting/cloud-catalogues.json"
PRIMARY_CATALOGUES = tuple(
    f"{name}-rows.json" for name in ("cumulative", "q1_q17_dr25_koi", "ps", "toi")
)
EXTRA_CATALOGUES = (
    "kepler-eb-2016-0694ada5b64a.csv", "exofop-ctoi-84a829ccae3d.txt",
)
PARENT_BUNDLE = ROOT / "pipeline/vetting/cloud-parents.sqlite.gz"
PARENT_SPECIFICATION = ROOT / "pipeline/vetting/cloud-parents.json"
VETTING_CONFIGURATION = ROOT / "pipeline/vetting/config.json"
PIXEL_BUNDLE = ROOT / "pipeline/vetting/cloud-pixels.tar.gz"
PIXEL_SPECIFICATION = ROOT / "pipeline/vetting/cloud-pixels.json"
PUBLIC_SPECIFICATION = ROOT / "pipeline/deterministic/cloud-public-inputs.json"
PUBLIC_CONTROL_SPECIFICATION = ROOT / "pipeline/deterministic/cloud-public-controls.json"
CONTROL_PIXEL_BUNDLE = ROOT / "pipeline/vetting/cloud-control-pixels.tar.gz"
CONTROL_PIXEL_SPECIFICATION = ROOT / "pipeline/vetting/cloud-control-pixels.json"
SCOPE_BUNDLE = ROOT / "pipeline/deterministic/cloud-joint-scope.json.gz"
SCOPE_SPECIFICATION = ROOT / "pipeline/deterministic/cloud-joint-scope.json"
NULL_POOL_BUNDLE = ROOT / "pipeline/deterministic/cloud-null-pool.csv.gz"
NULL_POOL_SPECIFICATION = ROOT / "pipeline/deterministic/cloud-null-pool.json"
INJECTED_PIXEL_SPECIFICATION = ROOT / "pipeline/vetting/cloud-injected-pixels.json"
INJECTED_PIXEL_BUNDLE = ROOT / "pipeline/vetting/cloud-injected-pixels.tar.gz"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(path, identity):
    if not path.is_file() or path.stat().st_size != identity["bytes"]:
        raise ValueError(f"Missing or wrong-sized input: {path}")
    if digest(path) != identity["sha256"]:
        raise ValueError(f"Changed input: {path}")


def scope_counts(metadata):
    trials = metadata["trials"]
    states = Counter(quarter["state"] for trial in trials for quarter in trial["pixel_quarters"])
    return {
        "trials": len(trials), "scope_counts": dict(Counter(trial["scope"] for trial in trials)),
        "trial_only_components": len(metadata["trial_only_components"]),
        "full_witness_components": len(metadata["full_witness_components"]),
        "qualified_external_connectivity_links": len(metadata["qualified_external_connectivity"]),
        "quarter_rows": sum(states.values()), "quarter_states": dict(states),
        "holdout_overlap_witnesses": len(metadata["holdout_overlap_witnesses"]),
        "preserved_exclusions": len(metadata["holdout_excluded_identities_preserved"]),
    }


def check_scope_metadata(metadata, specification):
    import cloud_null

    if (
        metadata.get("schema") != 1
        or metadata.get("status") != "PORTABLE_METADATA_SCOPE_PREPARATION_ONLY"
        or metadata.get("joint_independence_complete") is not False
        or scope_counts(metadata) != specification["counts"]
        or metadata["provenance"] != specification["provenance"]
        or metadata["unresolved"] != specification["unresolved"]
        or set(metadata["unresolved"]) != set(specification["whitelist"]["unresolved_scope"])
        or metadata["boundary"] != specification["boundary"]
    ):
        raise ValueError("Joint scope metadata bindings or boundary changed")
    cloud_null.check_observed_parent_edges(metadata)
    allowed = set(specification["whitelist"]["trial"])
    if any(set(trial) != allowed for trial in metadata["trials"]):
        raise ValueError("Joint scope trial metadata fields changed")
    if any(
        not set(quarter) <= set(specification["whitelist"]["quarter"])
        for trial in metadata["trials"] for quarter in trial["pixel_quarters"]
    ):
        raise ValueError("Joint scope quarter metadata fields changed")
    identities = [trial["trial_id"] for trial in metadata["trials"]]
    for name in ("trial_only_components", "full_witness_components"):
        members = [trial for component in metadata[name] for trial in component["trial_ids"]]
        if len(set(identities)) != len(identities) or sorted(members) != sorted(identities):
            raise ValueError("Joint scope component membership changed")


def scopes():
    path = DUMP / "cloud-joint-scope.json"
    temporary = path.with_suffix(".part")
    receipt = DUMP / "cloud-joint-scope-receipt.json"
    receipt_temporary = receipt.with_suffix(".part")
    if receipt.is_file() and not receipt.is_symlink():
        receipt.unlink()
    if any(item.is_symlink() for item in (path, temporary, receipt, receipt_temporary)):
        raise ValueError("Joint scope metadata output must not redirect its writes")
    if any(item.exists() and not item.is_file() for item in (path, temporary, receipt, receipt_temporary)):
        raise ValueError("Joint scope metadata outputs must be regular files")
    if temporary.exists() or receipt_temporary.exists():
        raise ValueError("Stale joint scope .part file; inspect and remove it before retrying")
    specification_bytes = SCOPE_SPECIFICATION.read_bytes()
    specification_sha256 = hashlib.sha256(specification_bytes).hexdigest()
    specification = json.loads(specification_bytes)
    artifact = specification["artifact"]
    if (
        specification.get("schema") != 1
        or specification.get("status") != "CLOUD_METADATA_PROJECTION_ONLY"
        or specification.get("joint_independence_complete") is not False
        or artifact["path"] != SCOPE_BUNDLE.name
        or type(artifact["canonical_json_bytes"]) is not int
        or not 0 < artifact["canonical_json_bytes"] <= 32 * 1024**2
    ):
        raise ValueError("Invalid joint scope metadata specification")
    checked_file(SCOPE_BUNDLE, artifact)
    with gzip.open(SCOPE_BUNDLE, "rb") as stream:
        payload = stream.read(artifact["canonical_json_bytes"] + 1)
    if (
        len(payload) != artifact["canonical_json_bytes"]
        or hashlib.sha256(payload).hexdigest() != artifact["canonical_json_sha256"]
    ):
        raise ValueError("Joint scope metadata decompression changed")
    check_scope_metadata(json.loads(payload), specification)
    import cloud_null

    dependencies = ((SCOPE_SPECIFICATION, specification_sha256), (SCOPE_BUNDLE, artifact["sha256"]))
    cloud_null.check_input_stability(dependencies)
    identity = {"bytes": len(payload), "sha256": artifact["canonical_json_sha256"]}
    if path.exists():
        checked_file(path, identity)
    else:
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.replace(path)
    path.chmod(0o444)
    result = {
        "status": "JOINT_SCOPE_METADATA_VERIFIED", "path": str(path), **identity,
        "specification_sha256": specification_sha256, "counts": specification["counts"],
        "joint_independence_complete": False, "scientific_outcomes_provisioned": False,
        "original_local_inputs_reverified": False, "unresolved": specification["unresolved"],
        "boundary": specification["boundary"],
    }
    cloud_null.check_input_stability(dependencies)
    cloud_null.publish(receipt, json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
    try:
        cloud_null.check_input_stability(dependencies)
    except Exception:
        receipt.unlink(missing_ok=True)
        raise
    return result


def null_pool():
    import cloud_null

    return cloud_null.provision(NULL_POOL_SPECIFICATION, NULL_POOL_BUNDLE, SCOPE_SPECIFICATION, SCOPE_BUNDLE, DUMP)


def unpack(archive, destination):
    destination = destination.resolve()
    seen = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            path = destination / member.name
            if (
                Path(member.name).is_absolute()
                or path.resolve() == destination
                or destination not in path.resolve().parents
            ):
                raise ValueError(f"Archive path escapes dump: {member.name}")
            if not member.isfile():
                raise ValueError(f"Archive contains a non-file: {member.name}")
            if path.resolve() in seen:
                raise ValueError(f"Archive repeats a path: {member.name}")
            seen.add(path.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)


def catalogues():
    specification = read(CATALOGUE_SPECIFICATION)
    names = set(PRIMARY_CATALOGUES + EXTRA_CATALOGUES)
    if specification.get("schema") != 1 or set(specification["files"]) != names:
        raise ValueError("Cloud catalogue specification differs from configured snapshots")
    checked_file(CATALOGUE_BUNDLE, specification["archive"])
    out = (DUMP / "cloud-reference-catalogues").resolve()
    if DUMP.resolve() not in out.parents:
        raise ValueError("Cloud catalogue directory must stay under dump")
    if not out.exists():
        out.mkdir(parents=True)
        unpack(CATALOGUE_BUNDLE, out)
    if {path.name for path in out.iterdir()} != names:
        raise ValueError("Cloud catalogue snapshot inventory changed")
    for name, expected in specification["files"].items():
        path = out / name
        if path.is_symlink():
            raise ValueError(f"Cloud catalogue snapshot must be a regular file: {path}")
        checked_file(path, expected)
        if "rows" in expected and len(read(path)) != expected["rows"]:
            raise ValueError(f"Cloud catalogue row count changed: {path}")
        path.chmod(0o444)
    result = {
        "status": "SNAPSHOTS_VERIFIED",
        "specification_sha256": digest(CATALOGUE_SPECIFICATION),
        "scope": specification["scope"],
        "files": specification["files"],
        "catalogues": [str(out / name) for name in PRIMARY_CATALOGUES],
        "extra_catalogues": [str(out / name) for name in EXTRA_CATALOGUES],
    }
    save(DUMP / "cloud-catalogues-receipt.json", result)
    return result


def parents():
    specification = read(PARENT_SPECIFICATION)
    config = read(VETTING_CONFIGURATION)
    if (
        specification.get("schema") != 1
        or specification["database"]["sha256"] != config["parent_index_sha256"]
        or config["parent_index"] != "dump/parent-index-20260925/parents.sqlite"
    ):
        raise ValueError("Cloud parent index differs from the configured identity")
    checked_file(PARENT_BUNDLE, specification["archive"])
    path = DUMP / "parent-index-20260925/parents.sqlite"
    if not path.exists():
        if DUMP.resolve() not in path.resolve().parents:
            raise ValueError("Cloud parent index must be written under dump")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".sqlite.building")
        with gzip.open(PARENT_BUNDLE, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        checked_file(temporary, specification["database"])
        temporary.chmod(0o444)
        temporary.replace(path)
    checked_file(path, specification["database"])
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        counts = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT parent_tpf), COUNT(DISTINCT shard) FROM sources"
        ).fetchone()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok" or list(counts) != [
        specification[key] for key in ("sources", "parents", "shards")
    ]:
        raise ValueError("Cloud parent index source inventory or integrity changed")
    result = {
        "status": "PARENT_INDEX_VERIFIED",
        "parent_index": str(path),
        "parent_index_sha256": specification["database"]["sha256"],
        "scope": specification["scope"],
        **{key: specification[key] for key in ("sources", "parents", "shards")},
    }
    save(DUMP / "cloud-parents-receipt.json", result)
    return result


def pixel_source(sid, source, expected):
    metadata = expected["metadata"]
    if (
        source["source_id"] != sid
        or metadata["source_id"] != sid
        or source["role"] != "development"
        or source["cohort"] != expected.get("opened_cohort", "ordinary")
        or source["origin_parent_kic"] != metadata["origin_tpf_kic"]
        or source["raw_fits_sha256"] != metadata["raw_fits_sha256"]
        or not source["quarters"]
    ):
        raise ValueError("A cloud pixel source, role, parent or raw identity differs")


def pixel_sources(specification, public_path=None):
    public_path = PUBLIC_SPECIFICATION if public_path is None else public_path
    public = read(public_path)
    if (
        specification.get("schema") != 1
        or specification["scope"] != "OPENED_DEVELOPMENT_ONLY"
        or public.get("scope") != "OPENED_DEVELOPMENT_ONLY"
        or specification["public_inputs_sha256"] != digest(public_path)
        or set(specification["sources"]) != set(public["sources"])
    ):
        raise ValueError("Cloud pixel inputs differ from the opened public selection")
    for sid, source in specification["sources"].items():
        pixel_source(sid, source, public["sources"][sid])
    projection = json.dumps(specification["sources"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(projection.encode()).hexdigest() != specification["origin"]["selected_sources_sha256"]:
        raise ValueError("The original cloud pixel source-quarter projection changed")


def relocated_pixel_sources(specification, out):
    sources = json.loads(json.dumps(specification["sources"]))
    used = set()
    for source in sources.values():
        for item in source["quarters"].values():
            for field in ("path", "nominal_path"):
                if field not in item:
                    continue
                original = item[field]
                name = specification["record_paths"][original]
                expected = item["sha256" if field == "path" else "nominal_sha256"]
                if specification["files"][name]["sha256"] != expected:
                    raise ValueError("A cloud pixel receipt differs from its original binding")
                item[field] = str(out / name)
                used.add(name)
            relocated = []
            for original in item.get("spatial_paths", []):
                name = specification["record_paths"][original]
                expected = specification["origin"]["auxiliary_spatial_receipts"][original]
                if specification["files"][name]["sha256"] != expected:
                    raise ValueError("An unresolved spatial receipt differs from its binding")
                relocated.append(str(out / name))
                used.add(name)
            if "spatial_paths" in item:
                item["spatial_paths"] = relocated
    if used != set(specification["files"]):
        raise ValueError("The cloud pixel receipt inventory differs from declared quarters")
    return sources


def checked_pixel_records(specification, out, archive=None):
    archive = PIXEL_BUNDLE if archive is None else archive
    if not out.exists():
        out.mkdir(parents=True)
        unpack(archive, out)
    files = specification["files"]
    if any(path.is_symlink() for path in out.rglob("*")):
        raise ValueError("Cloud pixel receipts must be regular files under dump")
    if {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} != set(files):
        raise ValueError("Cloud pixel receipt inventory changed")
    for name, expected in files.items():
        path = out / name
        if path.is_symlink() or out not in path.resolve().parents:
            raise ValueError("Cloud pixel receipts must be regular files under dump")
        checked_file(path, expected)
        path.chmod(0o444)


def pixel_content_counts(frozen):
    states, products, quarters = Counter(), {}, 0
    for source in frozen.ledger["sources"].values():
        for quarter in source["quarters"]:
            record = frozen.record(source, int(quarter))
            states[record["status"]] += 1
            quarters += 1
            if record["status"] == "PUBLIC_BYTE_AND_HEADER_IDENTITIES_VERIFIED":
                checksum = record["transfer"]["sha256"]
                size = record["request"]["size_bytes"]
                if checksum in products and products[checksum] != size:
                    raise ValueError("A cloud pixel payload has conflicting byte identities")
                products[checksum] = size
    return states, products, quarters


def pixel_output_paths(records_name, ledger_name, receipt_name):
    paths = tuple(DUMP / name for name in (records_name, ledger_name, receipt_name))
    guarded = (*paths, *(path.with_suffix(".part") for path in paths[1:]))
    if any(path.is_symlink() for path in guarded):
        raise ValueError("Cloud pixel outputs must stay under dump and not redirect their writes")
    if any(DUMP.resolve() not in path.resolve().parents for path in paths):
        raise ValueError("Cloud pixel outputs must stay under dump")
    return paths[0].resolve(), paths[1], paths[2]


def save_pixel_output(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("The materialized cloud pixel ledger changed")
        return
    temporary = path.with_suffix(".part")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pixel_inputs(public_path, specification_path, archive, names):
    from frozen_pixel_inputs import FrozenPixelInputs

    specification = read(specification_path)
    pixel_sources(specification, public_path)
    checked_file(archive, specification["archive"])
    out, path, receipt_path = pixel_output_paths(*names)
    checked_pixel_records(specification, out, archive)
    ledger = {
        "schema": 2, "status": "FROZEN_CONTROL_PIXEL_CONTENT_LEDGER",
        "inputs": {
            str(specification_path): digest(specification_path),
            str(public_path): digest(public_path),
            **{str(out / name): expected["sha256"] for name, expected in specification["files"].items()},
        },
        "sources": relocated_pixel_sources(specification, out),
        "source_quarters": specification["source_quarters"],
        "origin": specification["origin"],
        "scientific_outcomes_opened": False,
        "joint_independence_complete": False,
        "automatic_pc_enabled": False,
    }
    save_pixel_output(path, ledger)
    frozen = FrozenPixelInputs(path, digest(path))
    states, products, quarters = pixel_content_counts(frozen)
    if quarters != specification["source_quarters"]:
        raise ValueError("The cloud pixel source-quarter count changed")
    result = {
        "status": "OPENED_PIXEL_RECORDS_VERIFIED",
        "sources": len(ledger["sources"]), "source_quarters": quarters,
        "states": dict(states), "unique_payload_bytes": sum(products.values()),
        "payloads_downloaded": False, "origin": specification["origin"],
        "frozen_pixel_inputs": {
            "path": str(path), "sha256": digest(path), "role": "development",
            "maximum_cache_bytes": 4 * 1024**3,
        },
    }
    save_pixel_output(receipt_path, result)
    return result


def pixels():
    return pixel_inputs(
        PUBLIC_SPECIFICATION, PIXEL_SPECIFICATION, PIXEL_BUNDLE,
        ("cloud-pixel-records", "cloud-pixel-ledger.json", "cloud-pixels-receipt.json"),
    )


def control_pixels():
    return pixel_inputs(
        PUBLIC_CONTROL_SPECIFICATION, CONTROL_PIXEL_SPECIFICATION, CONTROL_PIXEL_BUNDLE,
        ("cloud-control-pixel-records", "cloud-control-pixel-ledger.json", "cloud-control-pixel-receipt.json"),
    )


def oracle():
    import astropy
    import numpy as np
    from astropy.timeseries import BoxLeastSquares
    from astropy.timeseries.periodograms.bls import _impl

    sys.path.insert(0, str(ROOT / "pipeline"))
    from shallow.run import bls_library, box_search

    path = BLS_FIXTURE
    checked_file(path, BLS_ORACLE)
    with np.load(path, allow_pickle=False) as fixed:
        origin = json.loads(str(fixed["origin"]))
        if astropy.__version__ != origin["astropy"]:
            raise ValueError("Astropy version differs from the production oracle")
        for band in (0, 1):
            periods, durations = fixed[f"periods_{band}"], fixed[f"durations_{band}"]
            for sign in (1, -1):
                time, flux, noise = fixed["time"], sign * fixed["flux"], fixed["noise"]
                reference = BoxLeastSquares(time, flux, dy=noise).power(
                    periods, durations, objective="snr"
                )
                outputs = {
                    "astropy": (reference.power, reference.depth, reference.duration, reference.transit_time),
                    "kernel": box_search(time, flux, noise, periods, durations),
                }
                for implementation, arrays in outputs.items():
                    if any(not np.array_equal(array.view(np.uint64), fixed[f"bits_{band}_{sign}_{i}"])
                           for i, array in enumerate(arrays)):
                        raise ValueError(f"Production BLS bits differ: {implementation}/{band}/{sign}")
    receipt = {
        "status": "BLS_ORACLE_PASSED", "cases": 4, "fixture": BLS_ORACLE,
        "origin": origin, "platform": platform.platform(),
        "astropy_bls_sha256": digest(Path(_impl.__file__)),
        "kernel_sha256": digest(Path(bls_library()._name)),
    }
    save(DUMP / "cloud-bls-oracle.json", receipt)
    return receipt


def control_preparation_scope(receipt, manifest):
    for value, scope_key in ((receipt, "scope"), (manifest, "measurement_scope")):
        if (
            value.get(scope_key) != "OPENED_DEVELOPMENT_ONLY"
            or value.get("role") != "development"
            or value.get("survey_context_complete") is not False
            or value.get("automatic_pc_enabled") is not False
            or value.get("joint_independence_complete") is not False
        ):
            raise ValueError("Public controls must preserve the opened development boundaries")


def control_preparation_sources(manifest, ids):
    expected = read(PUBLIC_CONTROL_SPECIFICATION)["sources"]
    if list(manifest["sources"]) != list(expected) or read(ids) != list(expected):
        raise ValueError("Public control source membership or order changed")
    for sid, source in manifest["sources"].items():
        declared = expected[sid]
        if any(source.get(field) != declared[field] for field in ("metadata", "opened_cohort", "opened_scope", "origin")):
            raise ValueError("Public control source identity, cohort or scope changed")
    return expected


def control_preparation_arrays(manifest, folder, expected):
    import numpy as np
    from cloud_public import verify_arrays, verify_inventory

    for sid, source in manifest["sources"].items():
        path = folder / f"{sid}.npz"
        if (
            source.get("arrays") != path.name or path.is_symlink()
            or path.resolve().parent != folder.resolve()
        ):
            raise ValueError("Public control arrays must use regular source-bound paths")
        if digest(path) != source["sha256"]:
            raise ValueError("Public control array container hash changed")
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != len(set(data.files)):
                raise ValueError("Public control array archive contains duplicate keys")
            if set(data.files) != set(expected[sid]["arrays"]):
                raise ValueError("Control array inventory differs from original inputs")
            arrays = {name: data[name] for name in data.files}
        verify_inventory(arrays, expected[sid]["arrays"])
        verify_arrays(arrays, expected[sid])


def verified_control_inputs():
    folder = DUMP / "cloud-controls"
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Verified public-controls directory required")
    paths = (folder / "receipt.json", folder / "input-manifest.json", folder / "source-ids.json", PUBLIC_CONTROL_SPECIFICATION)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Public control metadata must use regular original paths")
    payloads = {path: path.read_bytes() for path in paths}
    identities = {path: hashlib.sha256(payload).hexdigest() for path, payload in payloads.items()}
    receipt, manifest = (json.loads(payloads[path]) for path in paths[:2])
    if (receipt["status"] != "PUBLIC_INPUTS_VERIFIED" or receipt["specification_sha256"] != identities[paths[3]]
        or receipt["manifest_sha256"] != identities[paths[1]] or receipt["source_ids_sha256"] != identities[paths[2]]):
        raise ValueError("Public control receipt or original manifest changed")
    control_preparation_scope(receipt, manifest)
    expected = control_preparation_sources(manifest, paths[2])
    control_preparation_arrays(manifest, folder, expected)
    identities.update({folder / f"{sid}.npz": source["sha256"] for sid, source in manifest["sources"].items()})
    identities[Path(__file__)] = digest(__file__)
    return manifest, expected, identities


def injected_pixels():
    sys.path.insert(0, str(ROOT / "pipeline"))
    from deterministic import cloud_injected_pixels

    return cloud_injected_pixels.provision(INJECTED_PIXEL_SPECIFICATION, INJECTED_PIXEL_BUNDLE,
                                          PUBLIC_CONTROL_SPECIFICATION, DUMP / "cloud-controls", verified_control_inputs)


def prepare_public(out, controls=False):
    from cloud_public import SPECIFICATION

    sys.path.insert(0, str(ROOT / "pipeline"))
    from deterministic.run import prepare

    public_specification = PUBLIC_CONTROL_SPECIFICATION if controls else SPECIFICATION
    folder = DUMP / ("cloud-controls" if controls else "cloud-public")
    if controls and folder.is_symlink():
        raise ValueError("Public control input directory must not redirect its reads")
    receipt = read(folder / "receipt.json")
    manifest = folder / "input-manifest.json"
    ids = folder / "source-ids.json"
    if (
        receipt["status"] != "PUBLIC_INPUTS_VERIFIED"
        or receipt["specification_sha256"] != digest(public_specification)
        or receipt["manifest_sha256"] != digest(manifest)
        or receipt["source_ids_sha256"] != digest(ids)
    ):
        raise ValueError("Public input receipt or manifest changed")
    specification = read(manifest)
    if controls:
        control_preparation_scope(receipt, specification)
        expected = control_preparation_sources(specification, ids)
        control_preparation_arrays(specification, folder, expected)
    prepare(SimpleNamespace(
        shards=sorted({item["metadata"]["shard"] for item in specification["sources"].values()}),
        limit=len(specification["sources"]), source_ids=ids, input_manifest=manifest, out=out,
    ))
    return {"status": "PREPARED", "run": str(out), "sources": len(specification["sources"])}


def check_opened_preparation(out, manifest, expected):
    from deterministic import cloud_opened
    from deterministic.run import DISCOVERY_VIEW, read_fsc

    path = out / "preparation.json"
    cloud_opened.guard_path(out, path)
    payload = path.read_bytes()
    preparation = json.loads(payload)
    identities = {path: hashlib.sha256(payload).hexdigest()}
    ids = list(manifest["sources"])
    source_manifest = (DUMP / "cloud-opened/input-manifest.json").resolve()
    if (preparation["status"] != "PREPARED"
        or preparation.get("input_manifest") != {"path": str(source_manifest), "sha256": digest(source_manifest)}
        or preparation["selection"]["source_ids"] != ids
        or [row["source_id"] for row in preparation["sources"]] != ids
        or preparation["preparation_counts"]["attempted_sources"] != len(ids)
        or preparation["preparation_counts"]["ineligible_sources"] != 0):
        raise ValueError("Opened preparation changed the fixed supplied cohort")
    for row in preparation["sources"]:
        sid = row["source_id"]
        original = expected[sid]
        item = row["inputs"][DISCOVERY_VIEW]
        path = out / "inputs" / f"{sid}.{DISCOVERY_VIEW}.fsc"
        if (Path(item["fsc_path"]).resolve() != path.resolve()
            or item["fsc_sha256"] != digest(path)
            or row["parent_tpf"] != original["parents"]["origin"]
            or row["raw_fits_sha256"] != original["sha256"]
            or item["retained_cadences"] != original["prepared_cadences"]):
            raise ValueError("Opened preparation changed original input identities")
        records = read_fsc(path, sid, DISCOVERY_VIEW)
        if hashlib.sha256(records.tobytes()).hexdigest() != original["prepared_records_sha256"]:
            raise ValueError("Opened preparation changed the exact original FSC oracle")
        identities.update(opened_fsc_identities(out, sid, row["inputs"]))
    return identities


def opened_fsc_identities(out, sid, inputs):
    from deterministic import cloud_opened
    from deterministic.run import DETECTOR_VIEWS

    identities = {}
    for view, item in inputs.items():
        if view not in DETECTOR_VIEWS:
            raise ValueError("Opened preparation output view changed")
        path = out / "inputs" / f"{sid}.{view}.fsc"
        cloud_opened.guard_path(out, path)
        if Path(item["fsc_path"]).resolve() != path or not path.is_file() or digest(path) != item["fsc_sha256"]:
            raise ValueError("Opened preparation output identity changed")
        identities[path] = item["fsc_sha256"]
    return identities


def opened_preparation_capacity(out, expected):
    from deterministic import cloud_opened

    data, _, _ = cloud_opened.packet()
    growth = sum(cloud_opened.preparation_bytes(row) for row in expected.values()) + data["budgets"]["paired_run_headroom_bytes"]
    parent = next(path for path in out.parents if path.is_dir())
    disk = shutil.disk_usage(parent)
    reserve = cloud_opened.reserve(parent, data)
    if (disk.free < growth + reserve
        or reserve == 2 * 1024**3 and disk.used + growth + reserve > data["budgets"]["max_cloud_storage_bytes"]):
        raise ValueError("All-view opened preparation exceeds storage capacity and reserve")
    return {"forecast_growth_bytes": growth, "reserve_bytes": reserve, "available_bytes": disk.free,
            "current_filesystem_used_bytes": disk.used}


def prepare_opened(out):
    sys.path.insert(0, str(ROOT / "pipeline"))
    from deterministic import cloud_opened
    from deterministic.run import prepare
    from runtime import dump_path

    manifest, expected, dependencies = cloud_opened.verified_inputs()
    dependencies[Path(__file__)] = digest(__file__)
    folder = DUMP / "cloud-opened"
    out = dump_path(out)
    if out.exists():
        raise ValueError("Opened preparation requires a fresh output directory")
    try:
        resources = opened_preparation_capacity(out, expected)
        cloud_opened.recheck(dependencies)
        prepare(SimpleNamespace(
            shards=sorted({row["metadata"]["shard"] for row in manifest["sources"].values()}),
            limit=len(expected), source_ids=folder / "source-ids.json",
            input_manifest=folder / "input-manifest.json", out=out,
        ))
        dependencies.update(check_opened_preparation(out, manifest, expected))
        cloud_opened.recheck(dependencies)
    except BaseException:
        path = out / "preparation.json"
        if path.is_file() and not path.is_symlink():
            preparation = read(path)
            preparation["status"] = "PREPARATION_FAILED"
            save(path, preparation)
        raise
    return {"status": "PREPARED", "run": str(out), "sources": len(expected),
            "original_counts": cloud_opened.COUNTS,
            "provisioning_complete": manifest["provisioning_complete"],
            "scientific_acceptance": False, "joint_independence_complete": False, "resources": resources}


def complete_run(run, vetting):
    preparation = read(run / "preparation.json")
    detection = read(run / "detection.json")
    ids = preparation["selection"]["source_ids"]
    ledger = read(run / "source-ledger.json")
    if (
        preparation["status"] != "PREPARED"
        or detection["status"] != "DETECTION_COMPLETE"
        or not ids
        or any(not isinstance(sid, str) or not sid.isascii() or not sid.isdecimal() or sid.startswith("0") for sid in ids)
        or len(ids) != len(set(ids))
        or ids != [row["source_id"] for row in ledger]
        or ids != [row["source_id"] for row in preparation["sources"]]
        or detection["sources"] != len(ids)
    ):
        raise ValueError(f"Incomplete source accounting: {run}")
    verify_discovery_inputs(run, preparation, detection)
    if vetting:
        verify_vetting(run, detection)
    return preparation


def verify_discovery_inputs(run, preparation, detection):
    ids = preparation["selection"]["source_ids"]
    required = {run / "proposals" / f"{sid}.json" for sid in ids}
    required.update(run / name for name in (
        "raw.csv", "shallow-raw.csv", "shallow-eligibility.csv",
        "preparation.json", "queue.json", "source-ledger.json",
    ))
    manifest = read(run / "discovery-manifest.json")
    artifacts = manifest["artifacts"]
    if {Path(name).resolve() for name in artifacts} != {path.resolve() for path in required}:
        raise ValueError(f"Incomplete discovery artifact inventory: {run}")
    if manifest["detector_contract"] != preparation["detector_contract"]:
        raise ValueError(f"Discovery preparation contract changed: {run}")
    for name, expected in artifacts.items():
        path = Path(name)
        if run.resolve() not in path.resolve().parents or digest(path) != expected:
            raise ValueError(f"Invalid discovery artifact: {name}")
    binary = manifest["binary"]
    if binary != detection["binary"] or digest(Path(binary["path"])) != binary["sha256"]:
        raise ValueError(f"Detector binary changed: {run}")
    for source in preparation["sources"]:
        verify_prepared_inputs(source, run)
        proposal = read(run / "proposals" / f"{source['source_id']}.json")
        if proposal["source_id"] != source["source_id"]:
            raise ValueError(f"Proposal source identity changed: {run}")
    verify_detector_counts(run, detection, ids)


def verify_prepared_inputs(source, run):
    for view, identity in source["inputs"].items():
        path = Path(identity["fsc_path"])
        if (
            run.resolve() not in path.resolve().parents
            or path.name != f"{source['source_id']}.{view}.fsc"
            or digest(path) != identity["fsc_sha256"]
        ):
            raise ValueError(f"Prepared source input changed: {path}")


def verify_detector_counts(run, detection, ids):
    ledger = read(run / "source-ledger.json")
    for name, field in (("raw.csv", "raw_rows"), ("shallow-raw.csv", "shallow_rows")):
        rows = csv_rows(run / name)
        counts = Counter(row["source_id"] for row in rows)
        expected = {row["source_id"]: row[field] for row in ledger}
        if (
            len(rows) != detection[field]
            or any(row["source_id"] not in ids for row in rows)
            or any(counts[sid] != count for sid, count in expected.items())
        ):
            raise ValueError(f"Detector row accounting differs: {run / name}")
    eligibility = csv_rows(run / "shallow-eligibility.csv")
    if sorted(row["source_id"] for row in eligibility) != sorted(ids):
        raise ValueError(f"Shallow eligibility accounting differs: {run}")


def verify_vetting(run, detection):
    result, measured = read(run / "result.json"), read(run / "vetting/results.json")
    queue = read(run / "queue.json")["rows"]
    queued = {row["source_id"] for row in queue}
    records = sorted((run / "vetting/sources").glob("*.json"))
    alerts = sum(len(read(run / "proposals" / f"{sid}.json")["alerts"]) for sid in queued)
    if (
        result["status"] != "EVIDENCE_STAGES_COMPLETE"
        or measured["processing_status"] != "EVIDENCE_STAGES_COMPLETE"
        or len(queue) != len(queued)
        or queued != {path.stem for path in records}
        or measured["sources"] != len(queued)
        or detection["alerted_sources"] != len(queued)
        or measured["original_alerts_preserved"] != detection["alerts"]
        or alerts != detection["alerts"]
    ):
        raise ValueError(f"Incomplete vetting source accounting: {run}")
    events = 0
    for path in records:
        row, proposal = read(path), read(run / "proposals" / path.name)
        events += len(row["events"])
        expected = {alert["event_id"] for alert in proposal["alerts"]}
        if (
            row["source_id"] != path.stem
            or row["processing_status"] != "EVIDENCE_STAGES_COMPLETE"
            or {event["alert_id"] for event in row["events"]} != expected
        ):
            raise ValueError(f"Incomplete source and alert preservation: {path}")
    if measured["event_hypotheses"] != events:
        raise ValueError(f"Vetting event accounting differs: {run}")
    config, manifest = read(run / "vetting.json"), read(run / "vetting/manifest.json")
    if manifest["config_sha256"] != digest(run / "vetting.json"):
        raise ValueError(f"Vetting configuration changed: {run}")
    recorded_policy(config, manifest)


def csv_rows(path, timed=False):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if timed:
        for row in rows:
            row.pop("elapsed_seconds", None)
    return rows


def compare(baseline, candidate, vetting=True):
    left, right = complete_run(baseline, vetting), complete_run(candidate, vetting)
    paired_inputs(left, right)
    differences = Differences(VOLATILE | {"elapsed_seconds"}, 0, 0)
    for name in ("raw.csv", "shallow-raw.csv"):
        differences.compare(
            name, digest(baseline / name), digest(candidate / name)
        )
    differences.compare(
        "shallow-eligibility.csv", csv_rows(baseline / "shallow-eligibility.csv", True),
        csv_rows(candidate / "shallow-eligibility.csv", True),
    )
    compare_proposals(baseline, candidate, differences)
    report = {"detector_differences": differences.report(differences.material)}
    if vetting:
        paired_vetting(baseline, candidate)
        report["vetting"] = compare_runs(baseline, candidate, [], 0, 0)
    failed = bool(report["detector_differences"])
    if vetting:
        records = report["vetting"]["records"]
        failed |= bool(
            records["differing"]
            or records["missing_in_candidate"]
            or records["missing_in_baseline"]
            or not records["compared"]
        )
    report["status"] = "DIFFERENT" if failed else "IDENTICAL"
    return report


def paired_inputs(left, right):
    for key in ("cohort_sha256", "input_manifest", "selection"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Paired input changed: {key}")
    contracts = [
        {key: value for key, value in item["detector_contract"].items()
         if key not in {"files", "shallow_files"}}
        for item in (left, right)
    ]
    if contracts[0] != contracts[1]:
        raise ValueError("Detector scientific contract changed")
    if left["detector_contract"]["files"]["config.json"] != right["detector_contract"]["files"]["config.json"]:
        raise ValueError("Detector scientific configuration changed")
    if (
        left["detector_contract"]["shallow_files"]["config.json"]
        != right["detector_contract"]["shallow_files"]["config.json"]
    ):
        raise ValueError("Shallow scientific configuration changed")
    for before, after in zip(left["sources"], right["sources"], strict=True):
        paired_source(before, after)


def paired_source(before, after):
    for key in ("source_id", "raw_fits_sha256", "flatten_mask_sha256", "views", "unavailable_views"):
        if before.get(key) != after.get(key):
            raise ValueError(f"Paired source input changed: {key}")
    before_hashes = {view: identity["fsc_sha256"] for view, identity in before["inputs"].items()}
    after_hashes = {view: identity["fsc_sha256"] for view, identity in after["inputs"].items()}
    if before_hashes != after_hashes:
        raise ValueError("Paired prepared arrays changed")


def recorded_policy(config, manifest):
    configured = config["disposition_policy"]
    matches = [
        value for name, value in manifest["input_sha256"].items()
        if name == configured or name.endswith("/" + configured)
    ]
    if len(matches) != 1:
        raise ValueError("Vetting manifest must bind exactly one disposition policy input")
    return matches[0]


def paired_vetting(baseline, candidate):
    rows, policies, catalogues = [], [], []
    ignored = {"run_directory", "queue", "workers", "pixel_cache_directory", "prf_cache_directory",
               "pixel_minimum_free_bytes", "disposition_policy", "discovery_contract"}
    for run in (baseline, candidate):
        config = json.loads(run_text(run / "vetting.json", run))
        manifest = read(run / "vetting/manifest.json")
        policies.append(recorded_policy(read(run / "vetting.json"), manifest))
        catalogues.append(sorted(manifest["catalogue_sha256"].values()))
        config = {key: value for key, value in config.items() if key not in ignored}
        config["alias"] = {key: value for key, value in config["alias"].items() if key != "workers"}
        rows.append(config)
    if rows[0] != rows[1] or policies[0] != policies[1] or catalogues[0] != catalogues[1]:
        raise ValueError("Paired vetting scientific configuration or policy changed")


def compare_proposals(baseline, candidate, differences):
    files = {"source-ledger.json", "queue.json"}
    for run in (baseline, candidate):
        files.update(
            str(path.relative_to(run)) for path in (run / "proposals").glob("*.json")
        )
    for name in sorted(files):
        for run in (baseline, candidate):
            if not (run / name).is_file():
                raise ValueError(f"Missing paired evidence: {run / name}")
        differences.compare(
            name,
            json.loads(run_text(baseline / name, baseline)),
            json.loads(run_text(candidate / name, candidate)),
        )


def export_results(paths, output):
    paths = [path.resolve() for path in paths]
    for path in paths:
        if not path.exists() or DUMP.resolve() not in path.parents:
            raise ValueError(f"Result export must select existing dump paths: {path}")
        if output.resolve() == path or path in output.resolve().parents:
            raise ValueError("Export archive must be outside selected inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with tarfile.open(output, "w:gz", dereference=True) as bundle:
        for path in paths:
            selected = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in selected:
                if item.is_symlink() and not item.is_file():
                    raise ValueError(f"Linked evidence must be materialized as regular files: {item}")
                if not item.is_file():
                    continue
                if DUMP.resolve() not in item.resolve().parents:
                    raise ValueError(f"Result evidence escapes dump: {item}")
                name = str(item.relative_to(DUMP.resolve()))
                files[name] = {"sha256": digest(item), "bytes": item.stat().st_size}
                bundle.add(item, arcname=name, recursive=False)
        if not files:
            raise ValueError("Result export requires at least one evidence file")
        export_code(bundle, files)
    receipt = {
        "schema": 1,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "archive": {
            "path": str(output),
            "sha256": digest(output),
            "bytes": output.stat().st_size,
        },
        "files": files,
    }
    save(output.with_suffix(output.suffix + ".json"), receipt)
    return {"status": "EXPORTED", "files": len(files), "archive": receipt["archive"]}


def export_code(bundle, files):
    import io

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT)

    metadata = {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--short", "--", "pipeline", "docs").decode(),
    }
    entries = {
        "cloud-export/code.json": json.dumps(metadata, indent=2).encode(),
        "cloud-export/code.patch": git("diff", "--binary", "HEAD", "--", "pipeline", "docs"),
        "cloud-export/code.diffstat": git("diff", "--stat", "HEAD", "--", "pipeline", "docs"),
    }
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "pipeline", "docs").decode().split("\0")
    for name in filter(None, names):
        path = ROOT / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Code export requires regular files: {path}")
        entries["cloud-export/source/" + name] = path.read_bytes()
    for name, payload in entries.items():
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def opened_corpus(out, verify=False):
    sys.path.insert(0, str(ROOT / "pipeline"))
    from deterministic import cloud_opened_corpus

    out = out.absolute()
    if verify:
        receipt, _ = cloud_opened_corpus.verified_corpus(destination=out)
        return receipt
    minimum = cloud_opened_corpus.CLOUD_RESERVE if cloud_opened_corpus.cloud_destination(out) else cloud_opened_corpus.RESERVE
    return cloud_opened_corpus.provision(destination=out, minimum_free_bytes=minimum)


def opened_pixels(args):
    sys.path.insert(0, str(ROOT / "pipeline"))
    from deterministic import cloud_opened_pixels

    paths = {
        "corpus_root": args.corpus_root.absolute(),
        "opened_cache": args.opened_cache.absolute(),
        "destination": args.out.absolute(),
    }
    if args.verify:
        if args.retained_inputs is not None or args.source_limit is not None:
            raise ValueError("Pixel verification uses the published selection and input mapping")
        return cloud_opened_pixels.verified_pixels(args.run.absolute(), **paths)
    mapping = read(args.retained_inputs) if args.retained_inputs is not None else None
    return cloud_opened_pixels.provision(
        args.run.absolute(), **paths, mode="retained" if mapping is not None else "download",
        retained_inputs=mapping, source_limit=args.source_limit,
    )


def execute(args):
    commands = {
        "oracle": oracle, "catalogues": catalogues, "parents": parents,
        "pixels": pixels, "control-pixels": control_pixels, "scopes": scopes, "null-pool": null_pool,
        "injected-pixels": injected_pixels,
        "opened-pixels": lambda: opened_pixels(args),
    }
    if args.command in {"public-data", "public-controls"}:
        import cloud_public

        fetcher = cloud_public.fetch_controls if args.command == "public-controls" else cloud_public.fetch_public
        return fetcher(args.section)
    if args.command in commands:
        return commands[args.command]()
    if args.command == "opened-corpus":
        return opened_corpus(args.out, args.verify)
    if args.command in {"opened-controls", "opened-truths", "opened-truth-refs"}:
        sys.path.insert(0, str(ROOT / "pipeline"))
        from deterministic import cloud_opened

        if args.command == "opened-controls":
            return cloud_opened.provision(args.section, source_limit=args.source_limit, dry_run=args.dry_run, cohorts=args.cohort)
        if args.command == "opened-truths":
            return cloud_opened.provision_truths(args.section)
        refs, dependencies = cloud_opened.truth_refs(args.run)
        return {"status": "OPENED_PREPARED_TRUTH_REFERENCES_VERIFIED", "role": "development",
                "original_truth_bytes_preserved": True, "prepared_fsc_verified": True,
                **{flag: False for flag in cloud_opened.FLAGS}, "sources": len(refs),
                "truth_refs": refs, "dependencies": {str(path): value for path, value in dependencies.items()}}
    if args.command == "prepare":
        if args.opened:
            return prepare_opened(args.out)
        return prepare_public(args.out, args.controls)
    if args.command == "compare":
        result = compare(args.baseline.resolve(), args.candidate.resolve(), not args.detectors_only)
        save(args.out, result)
        return result
    return export_results(args.paths, args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("public-data")
    download.add_argument("section", choices=("1st", "2nd"))
    controls = sub.add_parser("public-controls")
    controls.add_argument("section", choices=("1st", "2nd"))
    opened = sub.add_parser("opened-controls")
    opened.add_argument("section", choices=("1st", "2nd"))
    opened.add_argument("--source-limit", type=int)
    opened.add_argument("--dry-run", action="store_true")
    opened.add_argument("--cohort", action="append", choices=("ordinary", "periodic_planet", "native_eb", "sparse_injection", "ttv_injection"))
    truths = sub.add_parser("opened-truths")
    truths.add_argument("section", choices=("1st", "2nd"))
    truth_refs = sub.add_parser("opened-truth-refs")
    truth_refs.add_argument("--run", type=Path, required=True)
    corpus = sub.add_parser("opened-corpus")
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--verify", action="store_true")
    pixels = sub.add_parser("opened-pixels")
    pixels.add_argument("--run", type=Path, required=True)
    pixels.add_argument("--corpus-root", type=Path, required=True)
    pixels.add_argument("--opened-cache", type=Path, default=DUMP / "cloud-opened")
    pixels.add_argument("--out", type=Path, required=True)
    pixels.add_argument("--retained-inputs", type=Path)
    pixels.add_argument("--source-limit", type=int)
    pixels.add_argument("--verify", action="store_true")
    sub.add_parser("oracle")
    sub.add_parser("catalogues")
    sub.add_parser("parents")
    sub.add_parser("pixels")
    sub.add_parser("control-pixels")
    sub.add_parser("scopes")
    sub.add_parser("null-pool")
    sub.add_parser("injected-pixels")
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--out", type=Path, required=True)
    selection = preparation.add_mutually_exclusive_group()
    selection.add_argument("--controls", action="store_true")
    selection.add_argument("--opened", action="store_true")
    paired = sub.add_parser("compare")
    paired.add_argument("baseline", type=Path)
    paired.add_argument("candidate", type=Path)
    paired.add_argument("--detectors-only", action="store_true")
    paired.add_argument("--out", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("paths", type=Path, nargs="+")
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args)
    print(json.dumps(result))
    return int(result["status"] == "DIFFERENT")


if __name__ == "__main__":
    sys.exit(main())
