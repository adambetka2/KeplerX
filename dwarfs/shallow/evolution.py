"""Blind, paired detector competition on bound end-to-end control cohorts."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import pairwise
from multiprocessing import get_context
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import numpy as np
from scipy.stats import beta, binomtest

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
SURVEY = ROOT / "pipeline/shallow"
SURVEY_FILES = ("run.py", "bls.c")
ENGINE = None
SETTINGS = None
# Calibration keeps all diagnostics. Survey mode keeps arrays only for proposals (~0.1 TB per 50,000 sources).
ARRAY_POLICY = "all"
TRIMMED_KEYS = frozenset(
    {
        "store_periodograms",
        "periodogram_arrays",
        "preprocessing_arrays",
        "nuisance_segments",
        "excluded_refinement_intervals",
        "local_refinements",
        "trials",
        "profile_arrays",
        "per_cycle",
        "inverse_fits",
        "contexts",
        "arrays",
        "inactive_native_prefix_bands",
    }
)
INLINE_ELEMENTS = 16
PHYSICAL_EVIDENCE = (
    ("time", np.float64),
    ("prepared_flux", np.float32),
    ("nuisance_residual", np.float32),
    ("noise", np.float32),
    ("quarter", np.int16),
)
SEED_EVIDENCE = ("period_days", "search_snr", "depth_ppm")
DROPPED = object()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def save(path, value, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    separators = None if indent else (",", ":")
    text = json.dumps(serial(value), allow_nan=False, indent=indent, separators=separators)
    temporary.write_text(text + "\n")
    temporary.replace(path)


def bind_files(paths):
    return {str(path): digest(path) for path in paths}


def check_bindings(bindings):
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("Bound input, source, configuration or result changed")


def case_bindings(row, folder):
    bindings = {}
    names = ["receipt"]
    if row["preparation_status"] == "PREPARED":
        names.append("input")
    names.extend(
        name
        for name in ("observability", "instrument_observability")
        if row.get(name + "_path")
    )
    for name in names:
        path = Path(row[name + "_path"])
        path = path if path.is_absolute() else folder / path
        bindings[str(path)] = row[name + "_sha256"]
    return bindings


def cohort_inputs(folder):
    manifest = folder / "manifest.json"
    case_path = folder / "cases.csv"
    bindings = bind_files([manifest, case_path])
    declared = read(manifest)
    for name in ("clocks_csv", "plan", "contamination"):
        entry = declared[name]
        path = Path(entry["path"])
        path = path if path.is_absolute() else folder / path
        bindings[str(path)] = entry["sha256"]
    expected = declared["cases_csv"]["sha256"]
    if bindings[str(case_path)] != expected:
        raise ValueError("Cohort case identity disagrees with its manifest")
    cases = rows(case_path)
    if "component_resolution" in declared:
        entry = declared["component_resolution"]
        resolution = folder / entry["path"]
        bindings[str(resolution)] = entry["sha256"]
        check_bindings({str(resolution): entry["sha256"]})
        component_mapping(cases, read(resolution), corrected=True)
    identifiers = [row["case_id"] for row in cases]
    if not cases or len(identifiers) != len(set(identifiers)):
        raise ValueError("Nonempty, unique case identities required")
    for row in cases:
        if row["positive"] not in ("true", "false"):
            raise ValueError("Case positive must be explicitly true or false")
        if not row["parent_tpf"]:
            raise ValueError("Every case requires a correlation parent identity")
        if not row["component_id"]:
            raise ValueError("Every case requires its joint identity component")
        bindings.update(case_bindings(row, folder))
    check_bindings(bindings)
    return cases, bindings


def component_mapping(cases, resolution, corrected=False):
    if resolution["status"] != "COMPLETED_INPUT_ONLY_TARGET_KIC_CLOSURE_AUDIT":
        raise ValueError("Completed input-only identity resolution required")
    identifiers = {row["case_id"] for row in cases}
    mapping = {}
    for group in resolution["corrected_components"].values():
        if not identifiers.intersection(group):
            continue
        if not set(group) <= identifiers or mapping.keys() & group.keys():
            raise ValueError("Complete, disjoint identity-resolution stages required")
        mapping.update(group)
    if set(mapping) != identifiers:
        raise ValueError("Identity resolution does not cover exact case membership")
    for case in cases:
        row = mapping[case["case_id"]]
        original = case["original_component_id"] if corrected else case["component_id"]
        if (
            original != row["original_component"]
            or case["source_id"] != row["source_id"]
            or case["kind"] != row["kind"]
        ):
            raise ValueError("Identity-resolution case metadata changed")
        if corrected and case["component_id"] != row["corrected_component_id"]:
            raise ValueError("Corrected case component disagrees with input-only audit")
    return mapping


def resolve_components(args):
    cases, bindings = cohort_inputs(args.cohort)
    if args.out.exists() or any("original_component_id" in row for row in cases):
        raise ValueError(
            "Fresh component-resolution destination and original cases required"
        )
    manifest = read(args.cohort / "manifest.json")
    if "portable_export" not in manifest:
        raise ValueError(
            "Resolve exported portable cohorts to preserve all input paths"
        )
    check_bindings({str(args.mapping): args.mapping_sha256})
    mapping = component_mapping(cases, read(args.mapping))
    args.out.mkdir(parents=True)
    for path in args.cohort.iterdir():
        if path.name not in ("cases.csv", "manifest.json"):
            (args.out / path.name).symlink_to(
                path.resolve(), target_is_directory=path.is_dir()
            )
    for case in cases:
        row = mapping[case["case_id"]]
        case.update(
            original_component_id=case["component_id"],
            component_id=row["corrected_component_id"],
            native_target_kic=row["index_target_kic"] or "",
        )
    write_cases(args.out / "cases.csv", cases)
    shutil.copy2(args.mapping, args.out / "component-resolution.json")
    manifest.update(
        cases_csv={"path": "cases.csv", "sha256": digest(args.out / "cases.csv")},
        component_resolution={
            "path": "component-resolution.json",
            "sha256": args.mapping_sha256,
            "original_cohort": str(args.cohort.resolve()),
            "original_bindings": bindings,
            "contract": (
                "Only joint identity metadata changes; inputs, draws, clocks "
                "and original receipts remain unchanged."
            ),
        },
    )
    save(args.out / "manifest.json", manifest)
    check_bindings(bindings | {str(args.mapping): args.mapping_sha256})
    verified, _ = cohort_inputs(args.out)
    print(
        json.dumps(
            {
                "status": "COMPONENTS_RESOLVED_WITH_INPUTS_UNCHANGED",
                "cases": len(verified),
            }
        )
    )


def job_key(row):
    value = row["input_sha256"] + ":" + row["array_key"]
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def jobs_for(cases, folder):
    jobs = {}
    for row in cases:
        if row["preparation_status"] != "PREPARED":
            if not row["first_failure"]:
                raise ValueError("Ineligible input lacks its first failure")
            continue
        path = Path(row["input_path"])
        path = path if path.is_absolute() else folder / path
        job = {
            "key": job_key(row),
            "path": str(path),
            "sha256": row["input_sha256"],
            "array_key": row["array_key"],
        }
        prior = jobs.setdefault(job["key"], job)
        if prior["sha256"] != job["sha256"] or prior["array_key"] != job["array_key"]:
            raise ValueError("Shared input key has conflicting paths or identities")
    return list(jobs.values())


def export_input(job, folder):
    check_bindings({job["path"]: job["sha256"]})
    with np.load(job["path"], allow_pickle=False) as stored:
        records = stored[job["array_key"]].copy()
    target = folder / "inputs" / f"{job['key']}.npz"
    np.savez_compressed(target, records=records)
    check_bindings({job["path"]: job["sha256"]})
    return {
        "path": str(target.relative_to(folder)),
        "sha256": digest(target),
        "prepared_bytes_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
        "source_input": job,
        "rows": len(records),
    }


def write_cases(path, cases):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)


def export_receipts(case, cohort, destination):
    names = ["receipt"]
    names.extend(
        name
        for name in ("observability", "instrument_observability")
        if case.get(name + "_path")
    )
    for name in names:
        receipt = Path(case[name + "_path"])
        receipt = receipt if receipt.is_absolute() else cohort / receipt
        sha256 = case[name + "_sha256"]
        check_bindings({str(receipt): sha256})
        target = destination / "receipts" / f"{sha256}.json"
        if not target.exists():
            shutil.copy2(receipt, target)
        case[name + "_path"] = str(target.relative_to(destination))


def export_cohort(args):
    cases, bindings = cohort_inputs(args.cohort)
    jobs = jobs_for(cases, args.cohort)
    if args.out.exists():
        raise ValueError("Portable input cohort requires a fresh destination")
    growth = sum(Path(job["path"]).stat().st_size for job in jobs) * 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    capacity(args.out.parent, max(growth, 1024**2))
    (args.out / "inputs").mkdir(parents=True)
    (args.out / "receipts").mkdir()
    exported = {job["key"]: export_input(job, args.out) for job in jobs}
    for case in cases:
        export_receipts(case, args.cohort, args.out)
        if case["preparation_status"] == "PREPARED":
            product = exported[job_key(case)]
            case.update(
                input_path=product["path"],
                input_sha256=product["sha256"],
                array_key="records",
            )
    write_cases(args.out / "cases.csv", cases)
    manifest = read(args.cohort / "manifest.json")
    clocks = args.cohort / "clocks.csv"
    check_bindings({str(clocks): manifest["clocks_csv"]["sha256"]})
    shutil.copy2(clocks, args.out / "clocks.csv")
    for name in (
        "contamination",
        "plan",
        *(["component_resolution"] if "component_resolution" in manifest else []),
    ):
        entry = manifest[name]
        source = Path(entry["path"])
        source = source if source.is_absolute() else args.cohort / source
        check_bindings({str(source): entry["sha256"]})
        target = args.out / f"{name}.json"
        shutil.copy2(source, target)
        manifest[name] = entry | {"path": target.name, "sha256": digest(target)}
    check_bindings(bindings)
    manifest.update(
        cases_csv={"path": "cases.csv", "sha256": digest(args.out / "cases.csv")},
        clocks_csv={"path": "clocks.csv", "sha256": digest(args.out / "clocks.csv")},
        portable_export={
            "original_cohort": str(args.cohort.resolve()),
            "original_bindings": bindings,
            "inputs": exported,
            "source_sha256": digest(__file__),
            "contract": (
                "Only byte-identical prepared records are passed to the detector. "
                "Raw injection models and scorer clocks are absent from input "
                "arrays; original provenance receipts remain unchanged."
            ),
        },
    )
    save(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PORTABLE_PREPARED_COHORT_EXPORTED",
                "cases": len(cases),
                "inputs": len(jobs),
            }
        )
    )


def combine_cohorts(args):
    if args.out.exists():
        raise ValueError("Combined cohort requires a fresh destination")
    cases, clocks, sources = [], [], []
    for index, folder in enumerate(args.cohort):
        additions, bindings = cohort_inputs(folder)
        if index and any(row["positive"] == "true" for row in additions):
            raise ValueError("Supplements may add controls, never planet draws")
        manifest = read(folder / "manifest.json")
        clock_file = folder / "clocks.csv"
        check_bindings({str(clock_file): manifest["clocks_csv"]["sha256"]})
        cases.extend(additions)
        clocks.extend(rows(clock_file))
        sources.append(
            {
                "folder": str(folder.resolve()),
                "bindings": bindings,
                "manifest": manifest,
            }
        )
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Supplement duplicates an existing case identity")
    args.out.mkdir(parents=True)
    write_cases(args.out / "cases.csv", cases)
    with (args.out / "clocks.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(clocks[0]))
        writer.writeheader()
        writer.writerows(clocks)
    for name in ("contamination", "plan"):
        documents = []
        for source in sources:
            entry = source["manifest"][name]
            path = Path(entry["path"])
            path = path if path.is_absolute() else Path(source["folder"]) / path
            check_bindings({str(path): entry["sha256"]})
            documents.append(
                {"path": str(path), "sha256": entry["sha256"], "document": read(path)}
            )
        save(args.out / f"{name}.json", {"original_documents": documents})
    manifest = {
        "status": "COMBINED_PLANET_COHORT_WITH_PROSPECTIVE_CONTROL_SUPPLEMENT",
        "sources": sources,
        "positive_signals": sum(row["positive"] == "true" for row in cases),
        "requested_cases": len(cases),
        "kind_counts": dict(Counter(row["kind"] for row in cases)),
        "preparation_counts": dict(Counter(row["preparation_status"] for row in cases)),
    }
    for name, filename in (
        ("cases_csv", "cases.csv"),
        ("clocks_csv", "clocks.csv"),
        ("contamination", "contamination.json"),
        ("plan", "plan.json"),
    ):
        path = args.out / filename
        manifest[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    save(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "cases": len(cases),
                "positive_signals": manifest["positive_signals"],
            }
        )
    )


def capacity(destination, anticipated_bytes):
    for disk in (Path("/Volumes/discSSD"), Path("/Volumes/discPCIe")):
        if disk.exists() and shutil.disk_usage(disk).free < 2 * 1024**3:
            raise ValueError(f"External reserve below 2 GiB: {disk}")
    if (
        anticipated_bytes <= 0
        or shutil.disk_usage(destination).free < anticipated_bytes + 40 * 1024**3
    ):
        raise ValueError("Insufficient admitted storage with 40 GiB working reserve")


def source_file(source, name):
    if name in SURVEY_FILES:
        return source.parents[1] / "pipeline/shallow" / name
    return source / name


def snapshot(folder, designs):
    source = folder / "source/dwarfs/shallow"
    source.mkdir(parents=True)
    for path in FOLDER.iterdir():
        if path.suffix in (".py", ".c", ".json") and not path.name.startswith("test_"):
            shutil.copy2(path, source / path.name)
    survey = folder / "source/pipeline/shallow"
    survey.mkdir(parents=True)
    for name in SURVEY_FILES:
        shutil.copy2(SURVEY / name, survey / name)
    (folder / "source/dump").symlink_to(ROOT / "dump", target_is_directory=True)
    settings = folder / "designs.json"
    save(settings, designs)
    return bind_files([settings, *sorted(source.iterdir()), *sorted(survey.iterdir())])


def runtime_identity():
    libraries = {}
    for name in (
        "numpy",
        "scipy",
        "astropy",
        "wotan",
        "numba",
        "llvmlite",
        "batman-package",
    ):
        distribution = importlib.metadata.distribution(name)
        code = {
            str(path): digest(distribution.locate_file(path))
            for path in distribution.files or []
            if Path(path).suffix in (".py", ".so", ".dylib", ".pyd")
        }
        if not code:
            raise ValueError(f"Unbound numerical runtime: {name}")
        libraries[name] = {
            "version": distribution.version,
            "code_sha256": hashlib.sha256(
                json.dumps(code, sort_keys=True).encode()
            ).hexdigest(),
        }
    return {
        "python": sys.version,
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "executable_sha256": digest(sys.executable),
        "libraries": libraries,
    }


def import_engine(source, engine):
    for folder in (source, source_file(source, "run.py").parent):
        sys.path.insert(0, str(folder))
    module = importlib.import_module(engine)
    if Path(module.__file__).resolve() != source_file(source, engine + ".py").resolve():
        raise ValueError("Worker engine differs from the immutable round snapshot")
    return module


def initialize(source, engine, config, array_policy="all"):
    global ENGINE, SETTINGS, ARRAY_POLICY
    if array_policy not in ("all", "proposals"):
        raise ValueError(f"Unknown array policy: {array_policy}")
    ENGINE = import_engine(source, engine)
    SETTINGS = config
    ARRAY_POLICY = array_policy


def separate_arrays(value, prefix="", arrays=None):
    arrays = {} if arrays is None else arrays
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in numerical evidence")
        arrays[prefix] = value
        return {
            "array": prefix,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }, arrays
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key], arrays = separate_arrays(item, f"{prefix}/{key}", arrays)
        return result, arrays
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            converted, arrays = separate_arrays(item, f"{prefix}/{index}", arrays)
            result.append(converted)
        return result, arrays
    return value, arrays


def hydrate_arrays(value, arrays):
    if isinstance(value, dict) and set(value) == {"array", "shape", "dtype"}:
        array = arrays[value["array"]]
        if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
            raise ValueError("Saved blind numerical array metadata differs")
        return array.copy()
    if isinstance(value, dict):
        return {key: hydrate_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [hydrate_arrays(item, arrays) for item in value]
    return value


def scan_with_prefix(job, records, diagnostics):
    entry = job["seed_prefix"]
    check_bindings(entry["bindings"])
    result = read(entry["result"])
    if "coarse_execution" in entry["identity"]:
        if entry["runtime"] != runtime_identity():
            raise ValueError("Grouped prefix requires identical full runtime")
        ENGINE.SEARCH.checked_coarse_header(
            entry["identity"]["coarse_execution"], entry["identity"], records
        )
        ENGINE.SEARCH.checked_coarse_engine(
            entry["identity"]["coarse_execution"], result["diagnostics"]
        )
    with IndexedArrays(entry["arrays"], allow_pickle=False) as stored:
        if "coarse_execution" in entry["identity"] and set(stored.files) != (
            declared_array_names(
                {"diagnostics": result["diagnostics"], "proposals": result["proposals"]}
            )
            | {"prepared_records"}
        ):
            raise ValueError("Grouped prefix numerical membership differs")
        if (
            stored["prepared_records"].dtype != records.dtype
            or stored["prepared_records"].shape != records.shape
            or stored["prepared_records"].tobytes() != records.tobytes()
        ):
            raise ValueError("Saved blind seed has different prepared records")
        evidence = hydrate_arrays(result["diagnostics"], stored)
        proposals = hydrate_arrays(result["proposals"], stored)
    prefix = {
        "identity": entry["identity"],
        "diagnostics": evidence,
        "proposals": proposals,
    }
    if (
        "coarse_execution" in entry["identity"]
        and result["status"] == "DETECTOR_INELIGIBLE"
    ):
        diagnostics.update(evidence)
        check_bindings(entry["bindings"])
        raise ENGINE.ShallowIneligible(result["first_failure"])
    proposals = ENGINE.scan_with_seed_diagnostics(
        records, SETTINGS, prefix, diagnostics=diagnostics
    )
    check_bindings(entry["bindings"])
    return proposals


def save_unexpected_failure(target, job, records, diagnostics, error, before, start):
    failure = {
        "status": "UNEXPECTED_ERROR",
        "first_failure": f"{type(error).__name__}: {error}",
        "exception_module": type(error).__module__,
        "exception_traceback": traceback.format_exc(),
        "input": job,
        "prepared_bytes_sha256_before_search": before,
        "prepared_bytes_sha256_at_failure": hashlib.sha256(
            records.tobytes()
        ).hexdigest(),
        "elapsed_seconds": time.monotonic() - start,
    }
    save(target, failure)
    result, arrays = separate_arrays({"diagnostics": diagnostics})
    arrays["prepared_records_at_failure"] = records
    array_path = target.with_suffix(".npz")
    np.savez_compressed(array_path, **arrays)
    save(
        target,
        result
        | failure
        | {
            "partial_arrays": {
                "path": array_path.name,
                "sha256": digest(array_path),
                "bytes": array_path.stat().st_size,
            }
        },
    )


def trim_evidence(value, arrays):
    """Drop bulk diagnostics and array descriptors; inline arrays of at most 16 elements."""
    if isinstance(value, dict) and set(value) == {"array", "shape", "dtype"}:
        array = arrays[value["array"]]
        return array.tolist() if array.size <= INLINE_ELEMENTS else DROPPED
    if isinstance(value, dict):
        trimmed = {
            key: trim_evidence(item, arrays)
            for key, item in value.items()
            if key not in TRIMMED_KEYS
        }
        return {key: item for key, item in trimmed.items() if item is not DROPPED}
    if isinstance(value, list):
        trimmed = (trim_evidence(item, arrays) for item in value)
        return [item for item in trimmed if item is not DROPPED]
    return value


def proposal_evidence(result, arrays, records):
    """Arrays that let a reviewer see each proposal: the prepared records, the physical
    stage's preprocessing and the periodograms of the bands that seeded a proposal."""
    evidence = {"prepared_records": records}
    bands = result["diagnostics"].get("bands", [])
    seeds = set()
    for band in bands:
        if "preprocessing_arrays" not in band:
            continue
        # A proposal row's search_rank is the seed_rank of its refinement in the same band.
        ranks = {row["search_rank"] for row in band.get("decisions", []) if row.get("status") == "PROPOSAL"}
        seeds.update(ref["seed_band"] for ref in band.get("physical_refinements", []) if ref["seed_rank"] in ranks)
        for name, dtype in PHYSICAL_EVIDENCE:
            evidence[f"{band['band']}/{name}"] = arrays[band["preprocessing_arrays"][name]["array"]].astype(dtype)
    for band in bands:
        if "periodogram_arrays" in band and band["band"] in seeds:
            for name in SEED_EVIDENCE:
                evidence[f"{band['band']}/{name}"] = arrays[band["periodogram_arrays"][name]["array"]].astype(np.float32)
    return evidence


def policy_outputs(result, arrays, records, proposals):
    """Apply the array policy to a separated result: every array and the full JSON under
    "all"; under "proposals" the trimmed JSON, and arrays only when there are proposals.
    The evidence is read before trimming, which removes the array descriptors."""
    if ARRAY_POLICY == "all":
        return result, arrays | {"prepared_records": records}
    evidence = proposal_evidence(result, arrays, records) if proposals else None
    return trim_evidence(result, arrays) | {"array_policy": ARRAY_POLICY}, evidence


def scan_job(task):
    job, target = task
    if digest(job["path"]) != job["sha256"]:
        raise ValueError("Prepared input hash changed before search")
    with np.load(job["path"], allow_pickle=False) as stored:
        records = stored[job["array_key"]].copy()
    return scan_records(job, records, target)


def scan_records(job, records, target):
    """Search prepared records and save the result and its arrays under the array policy."""
    target = Path(target)
    prepared_digest = hashlib.sha256(records.tobytes()).hexdigest()
    if records.dtype.names != ("time", "flux", "error", "quarter", "cadence"):
        raise ValueError("Declared LC2 records required")
    if (
        not all(np.isfinite(records[key]).all() for key in ("time", "flux", "error"))
        or np.any(records["error"] <= 0)
        or np.any(np.diff(records["time"]) < 0)
    ):
        raise ValueError("Prepared records have invalid photometry or clocks")
    start = time.monotonic()
    cpu_start = time.process_time()
    diagnostics = {"store_periodograms": True}
    try:
        try:
            proposals = (
                scan_with_prefix(job, records, diagnostics)
                if "seed_prefix" in job
                else ENGINE.scan(records, SETTINGS, diagnostics=diagnostics)
            )
            status, failure = "SEARCHED", None
        except ENGINE.ShallowIneligible as error:
            proposals, status, failure = [], "DETECTOR_INELIGIBLE", str(error)
        if hashlib.sha256(records.tobytes()).hexdigest() != prepared_digest:
            raise ValueError("Detector mutated its prepared input records")
    except Exception as error:
        save_unexpected_failure(
            target, job, records, diagnostics, error, prepared_digest, start
        )
        raise RuntimeError(
            f"{type(error).__module__}.{type(error).__name__}: {error}; "
            f"saved failure: {target}"
        ) from None
    search_seconds = time.monotonic() - start
    search_cpu_seconds = time.process_time() - cpu_start
    result, arrays = separate_arrays({"diagnostics": diagnostics, "proposals": proposals})
    result, arrays = policy_outputs(result, arrays, records, proposals)
    saved = None
    if arrays is not None:
        array_path = target.with_suffix(".npz")
        np.savez_compressed(array_path, **arrays)
        saved = {
            "path": array_path.name,
            "sha256": digest(array_path),
            "bytes": array_path.stat().st_size,
        }
    result.update(
        status=status,
        first_failure=failure,
        input=job,
        compiled_library=ENGINE.bls_library().build_manifest,
        search_seconds=search_seconds,
        search_cpu_seconds=search_cpu_seconds,
        elapsed_seconds=time.monotonic() - start,
        arrays=saved,
    )
    if "path" in job and digest(job["path"]) != job["sha256"]:
        raise ValueError("Prepared input hash changed during search")
    save(target, result, indent=2 if ARRAY_POLICY == "all" else None)
    return {"key": job["key"], "result": str(target), "sha256": digest(target)}


def run_tasks(jobs, folder, source, design, workers, task_function=scan_job):
    value = os.environ.get("SPACELAB_SHALLOW_NUMBA_CACHE_ROOT")
    if not value or not Path(value).is_absolute():
        raise ValueError("Explicit absolute per-round Numba cache root required")
    root = Path(value)
    engine = design["engine"]
    if Path(engine).name != engine or engine in (".", ".."):
        raise ValueError("Single declared engine cache namespace required")
    cache = root / engine
    cache.mkdir(parents=True, exist_ok=True)
    prior = os.environ["NUMBA_CACHE_DIR"]
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    try:
        folder.mkdir()
        iterator = iter(jobs)
        completed = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize,
            initargs=(source, design["engine"], design["config"]),
        ) as pool:
            pending = {}
            for _ in range(min(workers, len(jobs))):
                job = next(iterator)
                pending[
                    pool.submit(task_function, (job, folder / f"{job['key']}.json"))
                ] = job
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                try:
                    for future in finished:
                        completed.append(future.result())
                        del pending[future]
                except BaseException:
                    for future in pending:
                        future.cancel()
                    save(
                        folder / "failure.json",
                        {"completed": completed, "pending": list(pending.values())},
                    )
                    raise
                for _ in finished:
                    job = next(iterator, None)
                    if job is not None:
                        pending[
                            pool.submit(
                                task_function, (job, folder / f"{job['key']}.json")
                            )
                        ] = job
                save(
                    folder / "progress.json",
                    {"completed": len(completed), "total": len(jobs)},
                )
                print(
                    json.dumps(
                        {
                            "design": design["name"],
                            "completed": len(completed),
                            "total": len(jobs),
                        }
                    ),
                    flush=True,
                )
        return sorted(completed, key=lambda row: row["key"])
    finally:
        os.environ["NUMBA_CACHE_DIR"] = prior


def common_runtime(runtime):
    return {
        key: value
        for key, value in runtime.items()
        if key in ("python", "architecture", "executable_sha256")
    } | {
        "libraries": {
            name: runtime["libraries"][name]
            for name in ("numpy", "scipy", "astropy", "wotan", "numba", "llvmlite")
        }
    }


def incomplete_parent_bindings(entry, identity, folder):
    admission = entry["incomplete_parent"]
    failure_path = Path(admission["failure_path"])
    failed_design = failure_path.resolve().parent.relative_to(folder.resolve())
    design_names = {row["name"] for row in identity["designs"]}
    if (
        identity["status"] != "RUNNING"
        or identity["mode"] != "PAIRED_COMPETITION"
        or len(failed_design.parts) != 1
        or str(failed_design) not in design_names
        or str(failed_design) in identity["blind_results"]
        or entry["design"] not in identity["blind_results"]
        or failure_path.name != "failure.json"
        or (folder / entry["design"] / "failure.json").exists()
    ):
        raise ValueError(
            "Incomplete parent requires a complete, unfailed design ledger"
        )
    bindings = {str(failure_path): admission["failure_sha256"]}
    check_bindings(bindings)
    failure = read(failure_path)
    if set(failure) != {"completed", "pending"} or not failure["pending"]:
        raise ValueError("Incomplete parent lacks a bound failed-design receipt")
    return bindings


def saved_blind_round(entry, jobs, allow_incomplete=False):
    folder = Path(entry["round"])
    identity_path = folder / "identity.json"
    identity = read(identity_path)
    incomplete = "incomplete_parent" in entry
    if incomplete and not allow_incomplete:
        raise ValueError("Incomplete parents cannot supply seed prefixes")
    if not incomplete and identity["status"] != "BLIND_SEARCH_COMPLETE_UNSCORED":
        raise ValueError("Saved blind evidence requires a completed round")
    bindings = (
        identity["source_bindings"]
        | identity["cohort_bindings"]
        | {str(identity_path): entry.get("identity_sha256", digest(identity_path))}
    )
    if incomplete:
        bindings.update(incomplete_parent_bindings(entry, identity, folder))
    check_bindings(bindings)
    if common_runtime(identity["runtime"]) != common_runtime(runtime_identity()):
        raise ValueError("Saved blind numerical runtime differs")
    design = next(row for row in identity["designs"] if row["name"] == entry["design"])
    measured = identity["blind_results"][entry["design"]]
    results = {row["key"]: row for row in measured}
    if len(results) != len(measured):
        raise ValueError("Saved blind results contain duplicated prepared jobs")
    if (
        set(results) != {job["key"] for job in jobs}
        or len(results) != identity["jobs_per_design"]
    ):
        raise ValueError("Saved blind jobs differ from the complete requested cohort")
    return identity, design, results, bindings


def saved_blind_product(entry):
    path = Path(entry["result"])
    check_bindings({str(path): entry["sha256"]})
    result = read(path)
    arrays = path.parent / result["arrays"]["path"]
    bindings = {str(path): entry["sha256"], str(arrays): result["arrays"]["sha256"]}
    check_bindings(bindings)
    return result, arrays, bindings


def config_digest(config):
    return hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reuse_dependencies(engine, identity, source):
    registry = {
        "run": ("run.py", "bls.c"),
        "evolve_detector": ("evolve_detector.py", "evolve_noise.py", "run.py", "bls.c"),
        "evolve_template": (
            "evolve_template.py",
            "evolve_detector.py",
            "evolve_noise.py",
            "run.py",
            "bls.c",
        ),
    }
    if engine not in registry:
        raise ValueError("Whole-result reuse requires a reviewed engine dependency set")
    if identity.get("mode") == "PRISTINE_COARSE_GROUP":
        raise ValueError(
            "Grouped bank supplies checked seed prefixes, not whole results"
        )
    dependencies = {}
    for name in registry[engine]:
        values = {
            value
            for path, value in identity["source_bindings"].items()
            if Path(path).name == name
        }
        if len(values) != 1:
            raise ValueError("Saved engine dependency is missing or ambiguous")
        dependencies[name] = values.pop()
    check_bindings(
        {str(source_file(source, name)): value for name, value in dependencies.items()}
    )
    return dependencies


def declared_array_names(value):
    if isinstance(value, dict) and set(value) == {"array", "shape", "dtype"}:
        return {value["array"]}
    if isinstance(value, dict):
        return set().union(*(declared_array_names(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(declared_array_names(item) for item in value))
    return set()


class IndexedArrays:
    def __init__(self, path, *, allow_pickle=False):
        if allow_pickle is not False:
            raise ValueError("Pickled numerical arrays are forbidden")
        self.archive = zipfile.ZipFile(path)
        names = self.archive.namelist()
        if len(names) != len(set(names)) or any(
            not name.endswith(".npy") for name in names
        ):
            self.archive.close()
            raise ValueError("Duplicate or non-NPY numerical archive member")
        self.files = [name[:-4] for name in names]
        self.members = frozenset(self.files)

    def __enter__(self):
        return self

    def __getitem__(self, key):
        if not isinstance(key, str) or key not in self.members:
            raise ValueError("Exact numerical archive member missing")
        with self.archive.open(key + ".npy") as stream:
            array = np.lib.format.read_array(stream, allow_pickle=False)
        if not isinstance(array, np.ndarray):
            raise ValueError("Numerical archive member is not an array")
        return array

    def __exit__(self, *_):
        self.archive.close()


def reuse_array_identity(result, arrays, job):
    check_bindings({job["path"]: job["sha256"]})
    with IndexedArrays(job["path"], allow_pickle=False) as current:
        records = current[job["array_key"]].copy()
    with IndexedArrays(arrays, allow_pickle=False) as stored:
        if set(stored.files) != declared_array_names(
            {"diagnostics": result["diagnostics"], "proposals": result["proposals"]}
        ) | {"prepared_records"}:
            raise ValueError("Saved diagnostic array membership differs")
        prior = stored["prepared_records"]
        if (
            prior.dtype != records.dtype
            or prior.shape != records.shape
            or prior.tobytes() != records.tobytes()
        ):
            raise ValueError("Reused result has different complete prepared records")
        hydrate_arrays(result["diagnostics"], stored)
        hydrate_arrays(result["proposals"], stored)
        members = {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name in sorted(stored.files)
            for array in (stored[name],)
        }
    return {
        "prepared_records": members["prepared_records"],
        "numerical_members": len(members),
        "numerical_members_identity_sha256": config_digest(members),
    }


def reused_product(entry, job, native):
    result, arrays, bindings = saved_blind_product(entry)
    if "coarse_execution" in result["input"].get("seed_prefix", {}).get("identity", {}):
        raise ValueError(
            "Grouped consumer whole-result reuse requires a separate admission"
        )
    if result["status"] not in ("SEARCHED", "DETECTOR_INELIGIBLE"):
        raise ValueError("Only completed blind results can be reused")
    if result["status"] == "DETECTOR_INELIGIBLE" and not result["first_failure"]:
        raise ValueError("Reused ineligible result lacks its first failure")
    if any(
        result["input"][name] != job[name] for name in ("key", "sha256", "array_key")
    ):
        raise ValueError("Reused prepared input identity differs")
    previous_native = result["compiled_library"]
    if {key: value for key, value in native.items() if key != "path"} != {
        key: value for key, value in previous_native.items() if key != "path"
    }:
        raise ValueError("Reused native binary or compilation identity differs")
    bindings[previous_native["path"]] = previous_native["sha256"]
    array_identity = reuse_array_identity(result, arrays, job)
    timings = {
        "original_search_seconds": result["search_seconds"],
        "original_search_cpu_seconds": result["search_cpu_seconds"],
    }
    if not all(np.isfinite(value) and value >= 0 for value in timings.values()):
        raise ValueError("Reused result lacks valid measured search times")
    check_bindings(bindings | {job["path"]: job["sha256"]})
    return {
        "result": entry,
        "arrays": result["arrays"] | {"path": str(arrays)},
        "input": {name: job[name] for name in ("sha256", "array_key")},
    } | array_identity | timings, bindings


def reused_product_task(arguments):
    return reused_product(*arguments)


def reuse_runtime_admission(engine, previous, current):
    if previous == current:
        method = "IDENTICAL_FULL_RUNTIME"
    else:
        libraries = {"numpy", "scipy", "astropy", "wotan", "numba", "llvmlite"}
        if (
            engine not in ("run", "evolve_detector")
            or set(previous["libraries"]) != libraries
            or set(current["libraries"]) != libraries | {"batman-package"}
            or previous
            != current
            | {"libraries": {name: current["libraries"][name] for name in libraries}}
        ):
            raise ValueError("Reused engine numerical runtime differs")
        method = "LEGACY_BOX_RUNTIME_WITH_ADDED_UNUSED_BATMAN_BINDING"
    return {"method": method, "previous": previous, "current": current}


def reuse_results(design, jobs, source):
    start = time.monotonic()
    if "seed_prefix" in design:
        raise ValueError("Whole-result reuse cannot also use a seed prefix")
    keys = [job["key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Whole-result reuse requires unique prepared jobs")
    entry = design["reuse_results"]
    identity_path = Path(entry["round"]) / "identity.json"
    check_bindings({str(identity_path): entry["identity_sha256"]})
    identity, previous, results, bindings = saved_blind_round(
        entry, jobs, allow_incomplete=True
    )
    if previous["engine"] != design["engine"] or config_digest(
        previous["config"]
    ) != config_digest(design["config"]):
        raise ValueError("Reused engine or typed configuration differs")
    dependencies = reuse_dependencies(design["engine"], identity, source)
    admitted_runtime = runtime_identity()
    runtime_admission = reuse_runtime_admission(
        design["engine"], identity["runtime"], admitted_runtime
    )
    bindings.update(
        {str(source_file(source, name)): value for name, value in dependencies.items()}
    )
    engine = import_engine(source, design["engine"])
    native = engine.bls_library().build_manifest
    bindings[native["path"]] = native["sha256"]
    products = {}
    tasks = [(results[job["key"]], job, native) for job in jobs]
    with ProcessPoolExecutor(max_workers=8, mp_context=get_context("spawn")) as pool:
        for job, (product, product_bindings) in zip(
            jobs, pool.map(reused_product_task, tasks)
        ):
            products[job["key"]] = product
            bindings.update(product_bindings)
    check_bindings(bindings)
    if runtime_identity() != admitted_runtime:
        raise ValueError("Runtime changed during whole-result admission")
    return (
        list(results.values()),
        {
            "method": "REUSED_IMMUTABLE_COMPLETED_BLIND_RESULTS",
            "parent_status": identity["status"],
            "parent_admission": (
                "COMPLETE_DESIGN_FROM_INCOMPLETE_PARENT"
                if "incomplete_parent" in entry
                else "COMPLETE_ROUND"
            ),
            "incomplete_parent": entry.get("incomplete_parent"),
            "source_round": str(Path(entry["round"]).resolve()),
            "source_identity": {
                "path": str(identity_path),
                "sha256": entry["identity_sha256"],
            },
            "engine": design["engine"],
            "config_sha256": config_digest(design["config"]),
            "source_dependencies": dependencies,
            "runtime": identity["runtime"],
            "runtime_admission": runtime_admission,
            "native_build": native,
            "jobs": products,
            "fresh_searches": 0,
            "validation_seconds": time.monotonic() - start,
        },
        bindings,
    )


def prefix_jobs(design, jobs, source):
    extensions = {
        "evolve_template": ("physical_template",),
        "evolve_coherence": ("coherence_ranking",),
        "evolve_native_coherence": ("physical_template", "coherence_ranking"),
    }
    if design["engine"] not in extensions:
        raise ValueError("Saved box prefixes require a declared duration extension")
    identity, seed_design, results, bindings = saved_blind_round(
        design["seed_prefix"], jobs
    )
    seed_config = {
        key: value
        for key, value in design["config"].items()
        if key not in extensions[design["engine"]]
    }
    if (
        seed_design["engine"] != "evolve_detector"
        or seed_config != seed_design["config"]
    ):
        raise ValueError("Saved box prefix configuration differs")
    if identity.get("mode") == "PRISTINE_COARSE_GROUP":
        if set(design["seed_prefix"]) != {"round", "design", "identity_sha256"}:
            raise ValueError(
                "Grouped prefixes require an explicit sealed bank identity"
            )
        return grouped_prefix_jobs(
            identity, seed_design, results, bindings, jobs, source
        )
    sources = {
        Path(path).name: value
        for path, value in identity["source_bindings"].items()
        if Path(path).name
        in ("evolve_detector.py", "evolve_noise.py", "run.py", "bls.c")
    }
    check_bindings(
        {str(source_file(source, name)): value for name, value in sources.items()}
    )
    if len(sources) != 4:
        raise ValueError("Saved box prefix source is incomplete")
    prepared = []
    for job in jobs:
        result, arrays, product_bindings = saved_blind_product(results[job["key"]])
        bindings.update(product_bindings)
        if result["status"] == "DETECTOR_INELIGIBLE":
            prepared.append(job)
            continue
        if result["status"] != "SEARCHED":
            raise ValueError("Saved box prefix is incomplete")
        with IndexedArrays(arrays, allow_pickle=False) as stored:
            prepared_sha = hashlib.sha256(
                stored["prepared_records"].tobytes()
            ).hexdigest()
        native = {
            key: value
            for key, value in result["compiled_library"].items()
            if key != "path"
        }
        prepared.append(
            job
            | {
                "seed_prefix": {
                    "result": results[job["key"]]["result"],
                    "arrays": str(arrays),
                    "bindings": product_bindings,
                    "identity": {
                        "prepared_bytes_sha256": prepared_sha,
                        "config_sha256": hashlib.sha256(
                            json.dumps(seed_config, sort_keys=True).encode()
                        ).hexdigest(),
                        "source_sha256": sources,
                        "runtime": common_runtime(identity["runtime"]),
                        "native": native,
                    },
                }
            }
        )
    return prepared, bindings


def grouped_bank_metadata(identity, source):
    if (
        identity["mode"] != "PRISTINE_COARSE_GROUP"
        or identity["status"] != "BLIND_SEARCH_COMPLETE_UNSCORED"
        or len(identity["designs"]) != 3
        or identity["coarse_group"]["kind"] != "BATCHED_COARSE_THREE_METRICS"
        or identity["runtime"] != runtime_identity()
    ):
        raise ValueError(
            "Complete three-metric bank with identical full runtime required"
        )
    designs = identity["designs"]
    if any(
        set(d) != {"name", "engine", "config"} or d["engine"] != "evolve_detector"
        for d in designs
    ):
        raise ValueError("Exact pristine duration-engine group designs required")
    group = identity["coarse_group"]
    if set(group) != {
        "kind",
        "configs",
        "fine_native",
        "coarse_native",
        "source_sha256",
    }:
        raise ValueError("Exact grouped bank execution schema required")
    if group["configs"] != [d["config"] for d in designs]:
        raise ValueError("Coarse bank design/metric mapping differs")
    names = {"evolve_detector.py", "evolve_noise.py", "run.py", "bls.c", "batch_bls.c"}
    if set(group["source_sha256"]) != names:
        raise ValueError("Complete coarse dependency mapping required")
    bindings = {str(source_file(source, n)): group["source_sha256"][n] for n in names}
    for native in (group["fine_native"], group["coarse_native"]):
        if set(native) != {"path", "sha256", "compilation"}:
            raise ValueError("Actual coarse/fine build witness required")
        bindings[native["path"]] = native["sha256"]
    grouped_build_metadata(group, identity["source_bindings"])
    check_bindings(bindings)
    return group, bindings


def grouped_build_metadata(group, source_bindings):
    for name, sha in group["source_sha256"].items():
        if {
            value for path, value in source_bindings.items() if Path(path).name == name
        } != {sha}:
            raise ValueError("Grouped execution differs from its frozen source ledger")
    fine = group["fine_native"]["compilation"]
    if fine["source_sha256"] != group["source_sha256"]["bls.c"]:
        raise ValueError("Fine kernel source claim differs")
    if group["coarse_native"]["compilation"] != fine | {
        "source_sha256": group["source_sha256"]["batch_bls.c"],
        "included_bls_sha256": group["source_sha256"]["bls.c"],
    }:
        raise ValueError("Coarse compiler/source/include claim differs")


def grouped_bank_products(identity, jobs, group):
    expected = {job["key"] for job in jobs}
    names = {design["name"] for design in identity["designs"]}
    if len(names) != 3 or set(identity["blind_results"]) != names:
        raise ValueError("Exact declared three-design bank ledger required")
    bindings, products = {}, {}
    for member, design in enumerate(identity["designs"]):
        ledger = identity["blind_results"][design["name"]]
        if len(ledger) != len(jobs) or {row["key"] for row in ledger} != expected:
            raise ValueError(
                "All three bank members require complete unique job ledgers"
            )
        products[design["name"]] = {}
        for entry in ledger:
            result, arrays, bound = saved_blind_product(entry)
            bindings.update(bound)
            witness = result["diagnostics"]["engine"]["coarse_execution"]
            if (
                set(witness) != set(group) | {"member", "prepared", "runtime", "bands"}
                or type(witness["member"]) is not int
                or witness["member"] != member
                or {k: witness[k] for k in group} != group
                or witness["runtime"] != identity["runtime"]
                or result["compiled_library"] != group["fine_native"]
            ):
                raise ValueError("Actual member/coarse/fine/runtime witness differs")
            if result["status"] not in ("SEARCHED", "DETECTOR_INELIGIBLE"):
                raise ValueError("Every pristine prefix must be complete")
            if (
                result["status"] == "DETECTOR_INELIGIBLE"
                and not result["first_failure"]
            ):
                raise ValueError("Ineligible grouped prefix lacks its first failure")
            grouped_duration_membership(result, group["configs"][member])
            products[design["name"]][entry["key"]] = (entry, result, arrays)
    for job in jobs:
        members = [products[d["name"]][job["key"]][1] for d in identity["designs"]]
        grouped_job_metadata(members, job)
    return products, bindings


def grouped_duration_row(row, band, index, duration, config):
    chosen = config | {
        "duration_days": [duration],
        "period_grid_correlation": config["experimental_search"]["coarse_correlation"],
        "maximum_period_trials": config["experimental_search"]["maximum_period_trials"],
    }
    if (
        row.get("band") != f"{band}_duration_{index}"
        or row.get("search_band") != band
        or row.get("duration_trial_days") != duration
        or row.get("config") != chosen
        or row.get("store_periodograms") is not True
    ):
        raise ValueError("Exact declared duration/config/order/evidence required")
    if row["status"] == "INELIGIBLE":
        if row.get("reason") != "DURATION_ABOVE_ENVELOPE":
            raise ValueError("Explicit unchanged duration-envelope failure required")
    elif row["status"] != "SEARCHED" or "reason" in row:
        raise ValueError("Complete declared duration outcome required")


def grouped_duration_membership(result, config):
    rows = result["diagnostics"].get("bands", [])
    witness = result["diagnostics"]["engine"]["coarse_execution"]
    if result["status"] == "DETECTOR_INELIGIBLE":
        if rows or witness["bands"]:
            raise ValueError("Core ineligibility must precede all duration work")
        return
    expected = [
        ("baseline", i, duration, config)
        for i, duration in enumerate(config["experimental_search"]["duration_days"])
    ]
    if "long_period_search" in config:
        if rows and rows[-1].get("band") == "long":
            failure = rows[-1]
            if (
                set(failure) != {"band", "status", "reason"}
                or failure["status"] != "INELIGIBLE"
                or not isinstance(failure["reason"], str)
                or not failure["reason"]
            ):
                raise ValueError("Exact generic long ineligibility ledger required")
            rows = rows[:-1]
        else:
            chosen = config | config["long_period_search"]
            expected.extend(
                ("long", i, duration, chosen)
                for i, duration in enumerate(
                    config["experimental_search"]["long_duration_days"]
                )
            )
    if len(rows) != len(expected):
        raise ValueError("Every declared core/long duration must remain in the bank")
    for row, (band, index, duration, chosen) in zip(rows, expected):
        grouped_duration_row(row, band, index, duration, chosen)
    grouped_duration_witness(rows, witness)


def grouped_band_witness(row, entry, member):
    if (
        set(entry) != {"band", "duration_days", "members", "oversample"}
        or entry["band"] != row["search_band"]
        or entry["duration_days"] != row["duration_trial_days"]
        or entry["oversample"] != row["config"].get("coarse_phase_oversample", 3)
        or len(entry["members"]) != 3
    ):
        raise ValueError("Exact searched duration/witness order required")
    keys = {
        "preprocessing_time",
        "preprocessing_flux",
        "preprocessing_quarter",
        "preprocessing_base_noise",
        "period_days",
        "noise",
        "ivar",
        "seven_outputs",
    }
    for item in entry["members"]:
        if set(item) != keys or any(
            set(value) != {"dtype", "shape", "sha256"} for value in item.values()
        ):
            raise ValueError("Complete three-metric typed witness required")
    arrays = row["periodogram_arrays"]
    mapping = {
        "noise": "preprocessing_duration_noise",
        "seven_outputs": "coarse_native_seven_outputs",
    }
    selected = entry["members"][member]
    for key, value in selected.items():
        name = mapping.get(key, key)
        descriptor = arrays["preprocessing_duration_noise" if key == "ivar" else name]
        if (
            value["shape"] != descriptor["shape"]
            or value["dtype"] != descriptor["dtype"]
        ):
            raise ValueError("Serialized searched-array/witness schema differs")
    common = keys - {"noise", "ivar", "seven_outputs"}
    if any(
        {k: item[k] for k in common} != {k: entry["members"][0][k] for k in common}
        for item in entry["members"][1:]
    ):
        raise ValueError("Common searched-input/grid witness differs")


def grouped_duration_witness(rows, witness):
    searched = [row for row in rows if row["status"] == "SEARCHED"]
    if len(searched) != len(witness["bands"]):
        raise ValueError("Every searched duration must have its ordered witness")
    for row, entry in zip(searched, witness["bands"]):
        grouped_band_witness(row, entry, witness["member"])


def grouped_job_metadata(members, job):
    for result in members:
        if result["input"] != job:
            raise ValueError("Coarse bank job/input mapping differs")
    witnesses = [r["diagnostics"]["engine"]["coarse_execution"] for r in members]
    shared = [{k: v for k, v in w.items() if k != "member"} for w in witnesses]
    if any(w != shared[0] for w in shared[1:]):
        raise ValueError("Coarse bank common input/grid/band mapping differs")
    if len({(r["status"], r["first_failure"]) for r in members}) != 1:
        raise ValueError("Grouped eligibility/failure ledger differs")
    ledgers = [
        [
            {
                k: row[k]
                for k in (
                    "band",
                    "search_band",
                    "duration_trial_days",
                    "status",
                    "reason",
                )
                if k in row
            }
            for row in result["diagnostics"].get("bands", [])
        ]
        for result in members
    ]
    if any(ledger != ledgers[0] for ledger in ledgers[1:]):
        raise ValueError("Complete eligible/ineligible grouped duration ledger differs")


def grouped_prefix_jobs(identity, seed_design, results, bindings, jobs, source):
    group, bound = grouped_bank_metadata(identity, source)
    bindings.update(bound)
    products, bound = grouped_bank_products(identity, jobs, group)
    local = {
        path: sha
        for path, sha in bindings.items()
        if Path(path).name == "identity.json"
    }
    local.update(
        {str(source_file(source, n)): sha for n, sha in group["source_sha256"].items()}
    )
    local.update(
        {group[k]["path"]: group[k]["sha256"] for k in ("fine_native", "coarse_native")}
    )
    bindings.update(bound)
    sources = {
        n: sha for n, sha in group["source_sha256"].items() if n != "batch_bls.c"
    }
    prepared = []
    for job in jobs:
        entry, result, arrays = products[seed_design["name"]][job["key"]]
        witness = result["diagnostics"]["engine"]["coarse_execution"]
        native = {k: v for k, v in group["fine_native"].items() if k != "path"}
        owned = local | {job["path"]: job["sha256"]}
        for design in identity["designs"]:
            sibling_entry, sibling_result, sibling_arrays = products[design["name"]][
                job["key"]
            ]
            owned.update(
                {
                    sibling_entry["result"]: sibling_entry["sha256"],
                    str(sibling_arrays): sibling_result["arrays"]["sha256"],
                }
            )
        prepared.append(
            job
            | {
                "seed_prefix": {
                    "result": entry["result"],
                    "arrays": str(arrays),
                    "bindings": owned,
                    "runtime": identity["runtime"],
                    "identity": {
                        "prepared_bytes_sha256": witness["prepared"]["sha256"],
                        "config_sha256": hashlib.sha256(
                            json.dumps(seed_design["config"], sort_keys=True).encode()
                        ).hexdigest(),
                        "source_sha256": sources,
                        "runtime": common_runtime(identity["runtime"]),
                        "native": native,
                        "coarse_execution": witness,
                    },
                }
            }
        )
    check_bindings(bindings)
    return prepared, bindings


def group_records(job):
    check_bindings({job["path"]: job["sha256"]})
    with np.load(job["path"], allow_pickle=False) as stored:
        records = stored[job["array_key"]].copy()
    if records.dtype.names != ("time", "flux", "error", "quarter", "cadence"):
        raise ValueError("Declared LC2 records required")
    if (
        not all(np.isfinite(records[k]).all() for k in ("time", "flux", "error"))
        or np.any(records["error"] <= 0)
        or np.any(np.diff(records["time"]) < 0)
    ):
        raise ValueError("Prepared records have invalid photometry or clocks")
    return records


def save_group_product(
    target, job, records, evidence, proposals, status, failure, timing
):
    result, arrays = separate_arrays({"diagnostics": evidence, "proposals": proposals})
    arrays["prepared_records"] = records
    array_path = target.with_suffix(".npz")
    np.savez_compressed(array_path, **arrays)
    result.update(
        status=status,
        first_failure=failure,
        input=job,
        compiled_library=ENGINE.bls_library().build_manifest,
        search_seconds=None,
        search_cpu_seconds=None,
        group_execution_time=timing,
        arrays={
            "path": array_path.name,
            "sha256": digest(array_path),
            "bytes": array_path.stat().st_size,
        },
    )
    save(target, result)
    return {"key": job["key"], "result": str(target), "sha256": digest(target)}


def persist_group_products(
    targets, job, records, diagnostics, proposals, status, failure, timing
):
    products = []
    try:
        for path, evidence, rows in zip(targets, diagnostics, proposals):
            products.append(
                save_group_product(
                    path, job, records, evidence, rows, status, failure, timing
                )
            )
    except Exception as error:
        before = hashlib.sha256(records.tobytes()).hexdigest()
        for path, evidence in zip(
            targets[len(products) :], diagnostics[len(products) :]
        ):
            save_unexpected_failure(
                path, job, records, evidence, error, before, time.monotonic()
            )
        raise RuntimeError(
            "Grouped serialization failed; complete prior products retained"
        ) from error
    return products


def scan_group_job(task):
    job, target = task
    target = Path(target)
    if runtime_identity() != SETTINGS["runtime"]:
        raise ValueError("Grouped worker numerical runtime differs from admission")
    records = group_records(job)
    before = hashlib.sha256(records.tobytes()).hexdigest()
    diagnostics = [{"store_periodograms": True} for _ in range(3)]
    targets = [target.with_name(f"{job['key']}.metric{i}.json") for i in range(3)]
    start, cpu = time.monotonic(), time.process_time()
    try:
        try:
            proposals = ENGINE.scan_group(
                records, SETTINGS["configs"], diagnostics, SETTINGS["runtime"]
            )
            status, failure = "SEARCHED", None
        except ENGINE.ShallowIneligible as error:
            proposals, status, failure = [[], [], []], "DETECTOR_INELIGIBLE", str(error)
        if hashlib.sha256(records.tobytes()).hexdigest() != before:
            raise ValueError("Grouped detector mutated its prepared records")
        if runtime_identity() != SETTINGS["runtime"]:
            raise ValueError("Grouped worker numerical runtime changed during search")
    except Exception as error:
        for path, evidence in zip(targets, diagnostics):
            save_unexpected_failure(path, job, records, evidence, error, before, start)
        raise RuntimeError(
            f"Grouped failure saved for all metrics: {target}"
        ) from error
    timing = {
        "wall_seconds": time.monotonic() - start,
        "cpu_seconds": time.process_time() - cpu,
        "attribution": "ONE_SHARED_GROUP_EXECUTION_NOT_PER_MEMBER",
    }
    products = persist_group_products(
        targets, job, records, diagnostics, proposals, status, failure, timing
    )
    check_bindings({job["path"]: job["sha256"]})
    return {"key": job["key"], "products": products}


def initialize_coarse_group(source, configs):
    value = os.environ.get("SPACELAB_SHALLOW_NUMBA_CACHE_ROOT")
    if not value or not Path(value).is_absolute():
        raise ValueError("Explicit absolute per-round Numba cache root required")
    cache = Path(value) / "evolve_detector"
    cache.mkdir(parents=True, exist_ok=True)
    prior = os.environ["NUMBA_CACHE_DIR"]
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    try:
        initialize(source, "evolve_detector", None)
        ENGINE.checked_group(configs)
        return ENGINE.coarse_group_metadata(configs)
    finally:
        os.environ["NUMBA_CACHE_DIR"] = prior


def run_coarse_group(args):
    cases, cohort_bindings = cohort_inputs(args.cohort)
    designs = read(args.designs)
    if (
        len(designs) != 3
        or len({d["name"] for d in designs}) != 3
        or any(
            set(d) != {"name", "engine", "config"} or d["engine"] != "evolve_detector"
            for d in designs
        )
        or args.workers < 1
        or args.out.exists()
    ):
        raise ValueError(
            "Fresh bank and three uniquely named pristine seed designs required"
        )
    jobs = jobs_for(cases, args.cohort)
    capacity(args.out.parent, args.reserve_gib * 1024**3)
    args.out.mkdir()
    bindings = snapshot(args.out, designs)
    source = args.out / "source/dwarfs/shallow"
    runtime = runtime_identity()
    check_bindings(bindings | cohort_bindings)
    group = initialize_coarse_group(source, [d["config"] for d in designs])
    bindings.update(
        {group[k]["path"]: group[k]["sha256"] for k in ("fine_native", "coarse_native")}
    )
    identity = {
        "status": "RUNNING",
        "mode": "PRISTINE_COARSE_GROUP",
        "cohort": str(args.cohort.resolve()),
        "cohort_bindings": cohort_bindings,
        "source_bindings": bindings,
        "runtime": runtime,
        "cases": len(cases),
        "jobs_per_design": len(jobs),
        "workers": args.workers,
        "designs": designs,
        "admitted_bytes": args.reserve_gib * 1024**3,
        "coarse_group": group,
        "blind_results": {},
    }
    save(args.out / "identity.json", identity)
    results = run_tasks(
        jobs,
        args.out / "grouped_jobs",
        source,
        {
            "name": "three_metric_coarse_group",
            "engine": "evolve_detector",
            "config": {"configs": group["configs"], "runtime": runtime},
        },
        args.workers,
        task_function=scan_group_job,
    )
    identity["blind_results"] = {
        d["name"]: [row["products"][i] for row in results]
        for i, d in enumerate(designs)
    }
    check_bindings(bindings | cohort_bindings)
    if runtime_identity() != runtime:
        raise ValueError("Numerical runtime changed during grouped execution")
    identity["status"] = "BLIND_SEARCH_COMPLETE_UNSCORED"
    save(args.out / "identity.json", identity)


def final_mode(cases, designs):
    expected = {
        "synthetic_planet": 6000,
        "real_planet": 1000,
        "catalogue_fp": 200,
        "native_eb": 250,
        "instrumental_artifact": 600,
        "original": 650,
        "inverted": 650,
        "year_scrambled": 650,
    }
    if len(designs) != 1 or Counter(row["kind"] for row in cases) != expected:
        raise ValueError("Final mode requires one design and the approved 10000 mix")
    if any("seed_prefix" in row or "reuse_results" in row for row in designs):
        raise ValueError("The first final mixed test requires fresh blind searches")
    if any(
        (row["positive"] == "true")
        != (row["kind"] in ("synthetic_planet", "real_planet"))
        for row in cases
    ):
        raise ValueError("Final mode requires all 7000 planet signals as positives")
    for row in cases:
        final_recovery(row, False)
    return "FINAL_MIXED_TEST"


def sequential_mode(cases, designs):
    expected = {
        "synthetic_planet": 172,
        "real_planet": 28,
        "catalogue_fp": 14,
        "native_eb": 14,
        "instrumental_artifact": 14,
        "original": 14,
        "inverted": 14,
        "year_scrambled_unconditioned": 16,
        "year_scrambled_metadata_coverage_panel": 16,
    }
    if len(designs) != 1 or Counter(row["kind"] for row in cases) != expected:
        raise ValueError("Sequential mode requires one design and the fixed 302 mix")
    if any(
        (row["positive"] == "true")
        != (row["kind"] in ("synthetic_planet", "real_planet"))
        for row in cases
    ):
        raise ValueError("Sequential development requires all 200 planet positives")
    for row in cases:
        final_recovery(row, False)
    return "SEQUENTIAL_DEVELOPMENT200"


def mixed_diagnostic_mode(cases, designs):
    expected = {
        "synthetic_planet": 600,
        "real_planet": 100,
        "catalogue_fp": 50,
        "native_eb": 50,
        "instrumental_artifact": 50,
        "original": 50,
        "inverted": 50,
        "year_scrambled_unconditioned": 25,
        "year_scrambled_metadata_coverage_panel": 25,
    }
    if len(designs) != 1 or Counter(row["kind"] for row in cases) != expected:
        raise ValueError(
            "Mixed diagnostic mode requires one design and the fixed 1000 mix"
        )
    if any(
        (row["positive"] == "true")
        != (row["kind"] in ("synthetic_planet", "real_planet"))
        for row in cases
    ):
        raise ValueError("Mixed diagnostic mode requires all 700 planet positives")
    for row in cases:
        final_recovery(row, False)
    return "MIXED_DIAGNOSTIC1000"


def round_mode(cases, designs, golden, final, sequential=False, mixed=False):
    if sum((golden, final, sequential, mixed)) > 1:
        raise ValueError("Golden, final, sequential and mixed modes are exclusive")
    if mixed:
        return mixed_diagnostic_mode(cases, designs)
    if sequential:
        return sequential_mode(cases, designs)
    if golden:
        if len(designs) != 1 or any(
            row["kind"] != "real_planet" or row["positive"] != "true" for row in cases
        ):
            raise ValueError("Golden mode requires one design and real-planet controls")
        return "GOLDEN"
    if final:
        return final_mode(cases, designs)
    for case in cases:
        final_recovery(case, False)
    if not 3 <= len(designs) <= 4:
        raise ValueError("Three or four competing designs required")
    return "PAIRED_COMPETITION"


def run_round(args):
    cases, cohort_bindings = cohort_inputs(args.cohort)
    designs = read(args.designs)
    if len({row["name"] for row in designs}) != len(designs):
        raise ValueError("Uniquely named designs required")
    mode = round_mode(
        cases, designs, args.golden, args.final, args.sequential, args.mixed
    )
    if args.workers < 1 or args.out.exists():
        raise ValueError("Positive worker count and fresh round directory required")
    jobs = jobs_for(cases, args.cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    capacity(args.out.parent, args.reserve_gib * 1024**3)
    reused, reuse_bindings, reuse_receipts = {}, {}, {}
    for design in designs:
        if "reuse_results" in design:
            results, receipt, admitted = reuse_results(design, jobs, FOLDER)
            reused[design["name"]] = results
            reuse_receipts[design["name"]] = receipt
            reuse_bindings.update(admitted)
    args.out.mkdir()
    bindings = snapshot(args.out, designs) | reuse_bindings
    identity = {
        "status": "RUNNING",
        "mode": mode,
        "cohort": str(args.cohort.resolve()),
        "cohort_bindings": cohort_bindings,
        "source_bindings": bindings,
        "runtime": runtime_identity(),
        "cases": len(cases),
        "jobs_per_design": len(jobs),
        "workers": args.workers,
        "designs": designs,
        "admitted_bytes": args.reserve_gib * 1024**3,
        "blind_results": {},
        "result_reuse": reuse_receipts,
    }
    save(args.out / "identity.json", identity)
    source = args.out / "source/dwarfs/shallow"
    prepared_jobs = {}
    for design in designs:
        if "seed_prefix" in design:
            prepared_jobs[design["name"]], cached_bindings = prefix_jobs(
                design, jobs, source
            )
            bindings.update(cached_bindings)
        else:
            prepared_jobs[design["name"]] = jobs
    identity["source_bindings"] = bindings
    save(args.out / "identity.json", identity)
    for design in designs:
        check_bindings(bindings | cohort_bindings)
        if design["name"] in reused:
            identity["blind_results"][design["name"]] = reused[design["name"]]
            print(
                json.dumps({"design": design["name"], "reused_jobs": len(jobs)}),
                flush=True,
            )
        else:
            identity["blind_results"][design["name"]] = run_tasks(
                prepared_jobs[design["name"]],
                args.out / design["name"],
                source,
                design,
                args.workers,
            )
        save(args.out / "identity.json", identity)
    check_bindings(bindings | cohort_bindings)
    if runtime_identity() != identity["runtime"]:
        raise ValueError("Numerical runtime changed during the round")
    identity["status"] = "BLIND_SEARCH_COMPLETE_UNSCORED"
    save(args.out / "identity.json", identity)


def matches(proposal, clock):
    period, epoch = float(proposal["period_days"]), float(proposal["t0_bkjd"])
    true_period, true_epoch = float(clock["period_days"]), float(clock["epoch_bkjd"])
    start, stop = float(clock["start_bkjd"]), float(clock["stop_bkjd"])
    tolerance = float(clock["duration_hours"]) / 48 + float(clock["exposure_days"])
    if (
        not np.isfinite(
            [period, epoch, true_period, true_epoch, start, stop, tolerance]
        ).all()
        or min(period, true_period, tolerance) <= 0
    ):
        raise ValueError("Finite positive clocks required")
    centres = true_epoch + true_period * np.arange(
        np.ceil((start - true_epoch) / true_period),
        np.floor((stop - true_epoch) / true_period) + 1,
    )
    timing = np.abs(centres - epoch - np.rint((centres - epoch) / period) * period)
    return bool(
        len(timing)
        and abs(period / true_period - 1) * (stop - start) <= tolerance
        and abs((epoch - true_epoch + true_period / 2) % true_period - true_period / 2)
        <= tolerance
        and np.max(timing) <= tolerance
    )


def final_recovery(case, matched):
    if case["kind"] != "synthetic_planet" or case["preparation_status"] != "PREPARED":
        return matched, ""
    depth = float(case["sampled_prepared_depth_ppm"])
    identical = str(case["prepared_records_identical"]).lower()
    support = int(case["prepared_nonzero_loss_cadences"])
    if not np.isfinite(depth) or depth < 0 or support < 0:
        raise ValueError("Synthetic sampled depth/support must be nonnegative")
    if identical not in ("true", "false") or (depth == 0) != (support == 0):
        raise ValueError("Synthetic model/encoded support identity is inconsistent")
    if depth == 0:
        return False, "ZERO_PREPARED_MODEL_SUPPORT"
    if identical == "true":
        return False, "IDENTICAL_ORIGINAL_INJECTED_RECORDS"
    return matched, ""


def score_case(case, result, clocks, zero_support_guard):
    matched = any(
        matches(proposal, clock)
        for proposal in result["proposals"]
        for clock in clocks.get(case["case_id"], [])
    )
    recovered, reason = (
        final_recovery(case, matched) if zero_support_guard else (matched, "")
    )
    row = {
        **case,
        "recovered": recovered,
        "proposal_count": len(result["proposals"]),
        "detector_status": result["status"],
        "detector_first_failure": result["first_failure"],
    }
    if zero_support_guard:
        row.update(exact_clock_match=matched, recovery_exclusion=reason)
    return row


def score_design(cases, clocks, entries, zero_support_guard=False):
    results = {}
    for entry in entries:
        check_bindings({entry["result"]: entry["sha256"]})
        result = read(entry["result"])
        product = Path(entry["result"]).parent / result["arrays"]["path"]
        check_bindings({str(product): result["arrays"]["sha256"]})
        results[entry["key"]] = result
    scored = []
    for case in cases:
        eligible = case["preparation_status"] == "PREPARED"
        result = (
            results[job_key(case)]
            if eligible
            else {
                "proposals": [],
                "status": "INPUT_INELIGIBLE",
                "first_failure": case["first_failure"],
            }
        )
        scored.append(score_case(case, result, clocks, zero_support_guard))
    return scored


def sampled_depth_groups(groups, row):
    boundaries = [40, 60, 80, 100, 150, 200, 350, 500, 750, 1000]
    for field in ("sampled_raw_depth_ppm", "sampled_prepared_depth_ppm"):
        if row.get(field) in (None, ""):
            continue
        depth = float(row[field])
        if not np.isfinite(depth) or depth < 0:
            raise ValueError("Sampled physical depth must be finite and nonnegative")
        label = "0" if depth == 0 else "0-40" if depth < 40 else ">1000"
        for lower, upper in pairwise(boundaries):
            if lower <= depth < upper or depth == upper == 1000:
                label = f"{lower}-{upper}"
                break
        groups[f"{field}:{label}"].append(row)
        groups[f"{field}:population:{row['population']}:{label}"].append(row)


def gaia_group(row):
    raw = row.get("phot_g_mean_mag", "")
    if raw in (None, "", "N/A"):
        return "UNKNOWN"
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError("Gaia magnitude must be finite or explicit N/A")
    if value < 16:
        return "<16"
    if value < 17:
        return "16-17"
    if value <= 18:
        return "17-18"
    return ">18"


def binary_uncertainty(successes, count):
    if not count:
        return {"fraction": None, "interval95": [None, None], "upper95": None}
    lower = (
        float(beta.ppf(0.025, successes, count - successes + 1)) if successes else 0.0
    )
    upper = (
        float(beta.ppf(0.975, successes + 1, count - successes))
        if successes < count
        else 1.0
    )
    upper_one = (
        float(beta.ppf(0.95, successes + 1, count - successes))
        if successes < count
        else 1.0
    )
    return {
        "fraction": successes / count,
        "interval95": [lower, upper],
        "upper95": upper_one,
    }


def group_uncertainty(name, group):
    components = defaultdict(list)
    for row in group:
        components[row["component_id"]].append(row)
    ordered = [components[key] for key in sorted(components)]
    totals = np.asarray([len(rows) for rows in ordered], dtype=np.int64)
    recovery = np.asarray([sum(row["recovered"] for row in rows) for rows in ordered])
    trigger = np.asarray(
        [sum(row["proposal_count"] > 0 for row in rows) for rows in ordered]
    )
    seed = int.from_bytes(
        hashlib.sha256(("20261011:" + name).encode()).digest()[:8], "little"
    )
    indices = np.random.default_rng(seed).integers(
        0, len(ordered), size=(3000, len(ordered))
    )
    denominators = totals[indices].sum(axis=1)
    recovery_interval = np.quantile(
        recovery[indices].sum(axis=1) / denominators, [0.025, 0.975]
    ).tolist()
    trigger_interval = np.quantile(
        trigger[indices].sum(axis=1) / denominators, [0.025, 0.975]
    ).tolist()
    return {
        "recovery_case_binomial_descriptive": binary_uncertainty(
            int(recovery.sum()), len(group)
        ),
        "trigger_case_binomial_descriptive": binary_uncertainty(
            int(trigger.sum()), len(group)
        ),
        "component_count": len(ordered),
        "component_any_proposal": binary_uncertainty(
            int(np.count_nonzero(trigger)), len(ordered)
        ),
        "recovery_component_bootstrap_interval95": recovery_interval,
        "trigger_component_bootstrap_interval95": trigger_interval,
        "component_bootstrap_replicates": 3000,
        "uncertainty_scope": (
            "Conditional cohort uncertainty. Case binomial intervals assume "
            "independent cases; linked cases use component resampling. "
            "Bootstrap degenerates at all-zero/all-one outcomes; exact "
            "component upper bounds remain reported."
        ),
    }


def fractions(scored):
    groups = defaultdict(list)
    for row in scored:
        groups[f"kind:{row['kind']}"].append(row)
        groups[f"photometry:{row['photometry_input_class']}"].append(row)
        groups[f"conditioning:{row['selection_conditioning']}"].append(row)
        magnitude = gaia_group(row)
        groups[f"gaia_g:{magnitude}"].append(row)
        groups[f"view:{row.get('photometry_view', 'UNKNOWN')}"].append(row)
        if row["positive"] == "true":
            groups["planet_signals"].append(row)
            groups[f"population:{row['population']}"].append(row)
            groups[f"planet_gaia_g:{magnitude}"].append(row)
            groups[f"population_gaia_g:{row['population']}:{magnitude}"].append(row)
            sampled_depth_groups(groups, row)
            depth = float(row["depth_ppm"])
            boundaries = [40, 60, 80, 100, 150, 200, 350, 500, 750, 1000]
            for lower, upper in pairwise(boundaries):
                if lower <= depth < upper or depth == upper == 1000:
                    groups[f"depth_ppm:{lower}-{upper}"].append(row)
                    groups[f"depth_kind:{row['kind']}:{lower}-{upper}"].append(row)
                    groups[f"depth_gaia_g:{lower}-{upper}:{magnitude}"].append(row)
                    groups[
                        f"depth_population:{row['population']}:{lower}-{upper}"
                    ].append(row)
                    break
        else:
            groups["proposal_burden_controls"].append(row)
            if row["kind"] == "instrumental_artifact":
                mechanism = row.get("requested_instrument_kind", "NOT_RECORDED")
                support = row.get("instrument_observability_status", "NOT_RECORDED")
                groups[f"instrument_observability:{mechanism}:{support}"].append(row)
    return {
        name: {
            "count": len(group),
            "recovered": sum(row["recovered"] for row in group),
            "any_proposal": sum(row["proposal_count"] > 0 for row in group),
            "total_proposals": sum(row["proposal_count"] for row in group),
            "statuses": dict(Counter(row["detector_status"] for row in group)),
            **group_uncertainty(name, group),
        }
        for name, group in sorted(groups.items())
    }


def paired_statistics(baseline, candidate):
    deltas = defaultdict(int)
    old_control = defaultdict(bool)
    new_control = defaultdict(bool)
    old_count = defaultdict(int)
    new_count = defaultdict(int)
    wins = losses = positives = 0
    for old, new in zip(baseline, candidate, strict=True):
        if old["case_id"] != new["case_id"]:
            raise ValueError("Paired case order changed")
        if old["positive"] == "true":
            positives += 1
            wins += new["recovered"] and not old["recovered"]
            losses += old["recovered"] and not new["recovered"]
            deltas[old["component_id"]] += int(new["recovered"]) - int(old["recovered"])
        elif old["preparation_status"] == "PREPARED":
            component = old["component_id"]
            old_control[component] |= old["proposal_count"] > 0
            new_control[component] |= new["proposal_count"] > 0
            old_count[component] += old["proposal_count"]
            new_count[component] += new["proposal_count"]
    changes = np.asarray([value for value in deltas.values() if value])
    if len(changes) and np.all(np.abs(changes) == 1):
        pvalue = binomtest(
            int((changes > 0).sum()), len(changes), alternative="greater"
        ).pvalue
    elif len(changes):
        generator = np.random.default_rng(20261010)
        samples = generator.choice([-1, 1], (25000, len(changes))) @ changes
        pvalue = (1 + np.count_nonzero(samples >= changes.sum())) / 25001
    else:
        pvalue = 1.0
    added = sum(new_control[key] and not old_control[key] for key in old_control)
    count = len(old_control)
    upper = float(beta.ppf(0.95, added + 1, count - added)) if added < count else 1.0
    mean_increase = (
        (sum(new_count.values()) - sum(old_count.values())) / count if count else None
    )
    return {
        "wins": int(wins),
        "losses": int(losses),
        "positive_cases": positives,
        "net_gain_fraction": (wins - losses) / positives,
        "parent_cluster_one_sided_p": float(pvalue),
        "prepared_control_parents": count,
        "parents_with_new_proposals": added,
        "new_proposal_parent_fraction_upper95": upper,
        "mean_control_proposal_increase_per_component": mean_increase,
    }


def component_outcomes(scored, endpoint):
    groups = defaultdict(list)
    for row in scored:
        if row["kind"] == endpoint:
            groups[row["component_id"]].append(row["proposal_count"] > 0)
    if not groups:
        raise ValueError("No cases for a required control endpoint")
    return {key: any(value) for key, value in groups.items()}


def control_endpoint(baseline, candidate, kind):
    old = component_outcomes(baseline, kind)
    new = component_outcomes(candidate, kind)
    if old.keys() != new.keys():
        raise ValueError("Paired endpoint components differ")
    wins = sum(old[key] and not new[key] for key in old)
    losses = sum(new[key] and not old[key] for key in old)
    discordant = wins + losses
    pvalue = (
        binomtest(wins, discordant, alternative="greater").pvalue if discordant else 1.0
    )
    return {
        "components": len(old),
        "old_triggered": sum(old.values()),
        "new_triggered": sum(new.values()),
        "removed_trigger_components": wins,
        "added_trigger_components": losses,
        "net_trigger_reduction_fraction": (wins - losses) / len(old),
        "paired_component_one_sided_p": float(pvalue),
    }


def competition_endpoints(baseline, candidate):
    for old, new in zip(baseline, candidate, strict=True):
        if (old["case_id"], old["component_id"]) != (
            new["case_id"],
            new["component_id"],
        ):
            raise ValueError("Paired case/component identity changed")
    losses = defaultdict(bool)
    for old, new in zip(baseline, candidate, strict=True):
        if old["positive"] == "true":
            losses[old["component_id"]] |= old["recovered"] and not new["recovered"]
    eligibility = sum(
        old["detector_status"] == "SEARCHED" and new["detector_status"] != "SEARCHED"
        for old, new in zip(baseline, candidate, strict=True)
    )
    return {
        "catalogue_fp": control_endpoint(baseline, candidate, "catalogue_fp"),
        "native_eb": control_endpoint(baseline, candidate, "native_eb"),
        "planet_components_with_any_recovery_loss": binary_uncertainty(
            sum(losses.values()), len(losses)
        ),
        "searched_case_eligibility_regressions": eligibility,
    }


def holm_endpoints(comparisons):
    ordered = sorted(
        (item[field], name, endpoint)
        for name, item in comparisons.items()
        for endpoint, field in (
            ("recovery", "parent_cluster_one_sided_p"),
            ("catalogue_fp", "catalogue_fp_p"),
        )
    )
    adjusted = 0.0
    for index, (pvalue, name, endpoint) in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (len(ordered) - index) * pvalue))
        comparisons[name][endpoint + "_holm_adjusted_p"] = adjusted


def admit_promotions(comparisons):
    holm_endpoints(comparisons)
    for item in comparisons.values():
        endpoints = item["endpoints"]
        common = (
            endpoints["searched_case_eligibility_regressions"] == 0
            and endpoints["native_eb"]["new_triggered"]
            <= endpoints["native_eb"]["old_triggered"]
            and item["new_proposal_parent_fraction_upper95"] <= 0.05
            and item["mean_control_proposal_increase_per_component"] <= 0.25
        )
        recovery = (
            common
            and item["net_gain_fraction"] >= 0.05
            and item["recovery_holm_adjusted_p"] <= 0.05
        )
        loss_upper = endpoints["planet_components_with_any_recovery_loss"]["upper95"]
        fp = (
            common
            and item["net_gain_fraction"] >= 0
            and loss_upper <= 0.05
            and endpoints["catalogue_fp"]["net_trigger_reduction_fraction"] >= 0.20
            and item["catalogue_fp_holm_adjusted_p"] <= 0.05
        )
        item["promotion_routes"] = {
            "recovery": bool(recovery),
            "catalogue_fp": bool(fp),
        }
        item["promote_to_1000"] = bool(recovery or fp)


def report_status(identity):
    return {
        "GOLDEN": "SCORED_GOLDEN_END_TO_END",
        "FINAL_MIXED_TEST": "SCORED_FINAL_MIXED_END_TO_END",
        "PAIRED_COMPETITION": "SCORED_PAIRED_END_TO_END",
        "SEQUENTIAL_DEVELOPMENT200": "SCORED_SEQUENTIAL_DEVELOPMENT200",
        "MIXED_DIAGNOSTIC1000": "SCORED_MIXED_DIAGNOSTIC1000",
    }[identity.get("mode", "PAIRED_COMPETITION")]


def report_round(args):
    identity = read(args.out / "identity.json")
    if identity.get("mode") == "PRISTINE_COARSE_GROUP":
        raise ValueError("A pristine grouped bank supplies prefixes, never scores")
    if (
        identity["status"] != "BLIND_SEARCH_COMPLETE_UNSCORED"
        or (args.out / "comparison.json").exists()
    ):
        raise ValueError(
            "Only complete, previously unscored blind rounds can be scored"
        )
    check_bindings(identity["source_bindings"] | identity["cohort_bindings"])
    cohort = Path(identity["cohort"])
    cases, _ = cohort_inputs(cohort)
    manifest = read(cohort / "manifest.json")
    clock_path = cohort / "clocks.csv"
    check_bindings({str(clock_path): manifest["clocks_csv"]["sha256"]})
    clocks = defaultdict(list)
    for row in rows(clock_path):
        clocks[row["case_id"]].append(row)
    if any(not clocks[row["case_id"]] for row in cases if row["positive"] == "true"):
        raise ValueError("Every planet signal requires a scorer-only clock")
    scores = {
        name: score_design(
            cases,
            clocks,
            entries,
            zero_support_guard=True,
        )
        for name, entries in identity["blind_results"].items()
    }
    incumbent = identity["designs"][0]["name"]
    comparisons = {
        name: paired_statistics(scores[incumbent], value)
        for name, value in scores.items()
        if name != incumbent
    }
    for name, item in comparisons.items():
        item["endpoints"] = competition_endpoints(scores[incumbent], scores[name])
        item["catalogue_fp_p"] = item["endpoints"]["catalogue_fp"][
            "paired_component_one_sided_p"
        ]
    admit_promotions(comparisons)
    report = {
        "status": report_status(identity),
        "identity_sha256": digest(args.out / "identity.json"),
        "clock_sha256": digest(clock_path),
        "incumbent": incumbent,
        "promotion_rule": {
            "version": "PROSPECTIVE_FAINT_MULTIOBJECTIVE_V1",
            "recovery_route": (
                "Net planet gain >=0.05 and component-paired one-sided Holm p<=0.05."
            ),
            "catalogue_fp_route": (
                "FP any-proposal component reduction >=0.20 and paired one-sided "
                "Holm p<=0.05, with net planet gain>=0 and exact upper95 fraction "
                "of planet components with any loss<=0.05."
            ),
            "common_gates": (
                "No SEARCHED-case eligibility loss, no EB trigger-component count "
                "increase, upper95 newly triggered prepared control-component "
                "fraction<=0.05, mean control proposals/component increase<=0.25."
            ),
            "multiplicity": (
                "Holm across both endpoints for every non-incumbent design in this "
                "round."
            ),
            "scope": (
                "Qualification for a larger1000-case comparison only. Final targets"
                " and all unconditional denominators remain unchanged."
            ),
        },
        "summaries": {name: fractions(value) for name, value in scores.items()},
        "comparisons": comparisons,
    }
    if identity.get("mode") == "SEQUENTIAL_DEVELOPMENT200":
        report["historical_promotion_rule"] = report.pop("promotion_rule")
        report["adoption"] = {
            "status": "WORKING_DEVELOPMENT_ONLY",
            "comparison": (
                "Working adoption requires a separate bound, manual paired review."
            ),
            "qualification": (
                "The historical five-point/Holm rule does not decide adoption here; "
                "larger confirmation and final performance remain unqualified."
            ),
            "criteria": {
                "planet_net_gain_min": 1,
                "positive_denominator": 200,
                "searched_case_eligibility_regressions_max": 0,
                "native_eb_any_proposal_case_increase_max": 0,
                "catalogue_fp_any_proposal_case_increase_max": 0,
                "control_any_activity_case_increase_max": 0,
                "control_total_proposals_increase_max": 0,
                "control_denominator": 102,
                "tie": "Zero net planet gain requires lower aggregate control burden.",
                "reporting": "Record paired gains/losses and depth/M/K strata.",
                "scope": "Development only; no significance or independence claim.",
            },
        }
    if identity.get("mode") == "MIXED_DIAGNOSTIC1000":
        report["historical_promotion_rule"] = report.pop("promotion_rule")
        report["diagnostic_scoring"] = {
            "rule": "EXACT_CLOCK_WITH_ZERO_PREPARED_MODEL_SUPPORT_AS_MISS_V2",
            "positive_denominator": 700,
            "control_denominator": 300,
            "requested_cases": 1000,
            "synthetic_signals": 600,
            "real_signals": 100,
            "target_planet_recovery_fraction": 0.20,
            "scope": (
                "Fresh faint synthetic hosts and nulls; previously opened real "
                "and historical EB/FP anchors. Not a sealed final holdout."
            ),
            "eligibility": "Keep every input and detector failure in denominators.",
            "false_positive_measure": (
                "Any proposal on a historical control is activity, distinct "
                "from designated-clock recovery or astrophysical classification."
            ),
        }
    if identity.get("mode") == "FINAL_MIXED_TEST":
        signals = scores[incumbent]
        zeroes = [
            row
            for row in signals
            if row["recovery_exclusion"] == "ZERO_PREPARED_MODEL_SUPPORT"
        ]
        report["final_scoring"] = {
            "rule": "EXACT_CLOCK_WITH_ZERO_PREPARED_MODEL_SUPPORT_AS_MISS_V2",
            "positive_denominator": 7000,
            "target_recovered_signals": 1400,
            "aim_recovered_signals": 1750,
            "target_planet_recovery_fraction": 0.20,
            "aim_planet_recovery_fraction": 0.25,
            "zero_support_synthetic_draws": len(zeroes),
            "zero_support_exact_clock_coincidences": sum(
                row["exact_clock_match"] for row in zeroes
            ),
            "zero_support_native_proposals": sum(
                row["proposal_count"] for row in zeroes
            ),
        }
    for name, value in scores.items():
        save(args.out / f"{name}-scored-cases.json", value)
    save(args.out / "comparison.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--designs", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--reserve-gib", type=int, required=True)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--golden", action="store_true")
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--sequential", action="store_true")
    mode.add_argument("--mixed", action="store_true")
    group = commands.add_parser("coarse-group")
    group.add_argument("--cohort", type=Path, required=True)
    group.add_argument("--designs", type=Path, required=True)
    group.add_argument("--out", type=Path, required=True)
    group.add_argument("--workers", type=int, default=1)
    group.add_argument("--reserve-gib", type=int, required=True)
    report = commands.add_parser("report")
    report.add_argument("--out", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--cohort", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    combine = commands.add_parser("combine")
    combine.add_argument("--cohort", type=Path, action="append", required=True)
    combine.add_argument("--out", type=Path, required=True)
    resolve = commands.add_parser("resolve-components")
    resolve.add_argument("--cohort", type=Path, required=True)
    resolve.add_argument("--mapping", type=Path, required=True)
    resolve.add_argument("--mapping-sha256", required=True)
    resolve.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {
        "run": run_round,
        "coarse-group": run_coarse_group,
        "report": report_round,
        "export": export_cohort,
        "combine": combine_cohorts,
        "resolve-components": resolve_components,
    }[args.command](args)


if __name__ == "__main__":
    main()
