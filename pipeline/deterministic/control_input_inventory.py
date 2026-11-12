"""Inventory original real-control views and pixel quarters without outcomes."""

import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.photometry import lc2

COHORTS = {"native_eb", "catalogue_fp", "ordinary"}
VIEWS = ("psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova")
RESERVE = 20 * 1024**3
MAXIMUM_RSS = 2 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_digest(array):
    array = np.asarray(array)
    result = hashlib.sha256()
    result.update(json.dumps([str(array.dtype), array.shape]).encode())
    result.update(array.tobytes(order="C"))
    return result.hexdigest()


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def declared_sources(graph):
    audit = read(graph / "audit.json")
    ledger = graph / "trials.csv"
    if digest(ledger) != audit["trial_ledger_sha256"]:
        raise ValueError("The declared component ledger changed")
    with ledger.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != audit["trials"]:
        raise ValueError("The declared component trial count does not reconcile")
    selected = [row for row in rows if row["cohort"] in COHORTS]
    if not selected or len({row["source_id"] for row in selected}) != len(selected):
        raise ValueError("Real-control source identities are empty or duplicated")
    for row in selected:
        if row["role"] not in {"development", "evaluation"}:
            raise ValueError("A real-control source has an undeclared role")
        if row["historically_exposed_component"] not in {"True", "False"}:
            raise ValueError("Historical source exposure is not declared")
        if row["role"] == "evaluation" and (
            row["historically_exposed_component"] != "False"
        ):
            raise ValueError("An evaluation source is historically exposed")
    return selected


def corpus_index(index):
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as database:
        locations = {
            str(sid): (volume, int(shard))
            for sid, volume, shard in database.execute(
                "select gaia_source_id,volume,shard from sources"
            )
        }
        hashes = {
            (volume, path): checksum
            for volume, path, checksum in database.execute(
                "select volume,path,sha256 from files where collection='kbonus' "
                "and path like 'kbonus/lc2/shard-%/source.parquet'"
            )
        }
    return locations, hashes


def source_order(row, locations):
    if row["cohort"] == "ordinary":
        volume, shard = locations[row["source_id"]]
        return (1, volume, shard, int(row["source_id"]))
    return (0, row["cohort"], row["role"], int(row["source_id"]))


def bind_source_inputs(root, selected, manifests, inputs, locations, hashes):
    manifest_data = {name: read(path) for name, path in manifests.items()}
    for path in manifests.values():
        inputs[str(path.resolve())] = digest(path)
    for row in selected:
        if row["cohort"] == "ordinary":
            volume, shard = locations[row["source_id"]]
            relative = f"kbonus/lc2/shard-{shard:04d}/source.parquet"
            path = (root / volume / "data" / relative).resolve()
            if str(path) in inputs:
                continue
            expected = hashes[volume, relative]
        else:
            manifest = manifests[row["cohort"]]
            source = manifest_data[row["cohort"]]["sources"][row["source_id"]]
            path = (manifest.parent / source["arrays"]).resolve()
            expected = source["sha256"]
        if digest(path) != expected:
            raise ValueError(f"The declared source input changed: {path}")
        inputs[str(path)] = expected


def view_inventory(curve):
    valid = lc2.valid_points(curve, -1)
    discovery = valid & (curve["flatten"] == 0)
    counts = {}
    quarters = set()
    for quarter in np.unique(curve["quarter"]):
        use = curve["quarter"] == quarter
        raw_count = int(np.sum(use & valid))
        counts[str(int(quarter))] = {
            "original_rows": int(use.sum()),
            "quality_zero_usable_cadences": raw_count,
            "quality_zero_flatten_zero_usable_cadences": int(
                np.sum(use & discovery)
            ),
        }
        if raw_count:
            quarters.add(int(quarter))
    return {
        "array_sha256": {key: array_digest(curve[key]) for key in lc2.KEYS},
        "cadences_by_quarter": counts,
    }, quarters


def source_inventory(root, row, manifest):
    sid = row["source_id"]
    metadata, raw = lc2.load_source(root, sid, manifest)
    observed = (
        str(metadata["source_id"]), str(metadata["origin_tpf_kic"]),
        metadata["raw_fits_sha256"],
    )
    expected = (sid, row["origin_parent"], row["raw_fits_sha256"])
    domain = "KBonus_LC2" if manifest is None else "KBonus_LC1"
    if observed != expected or row["source_domain"] != domain:
        raise ValueError(f"The source/parent/raw/domain identity differs: {sid}")
    views, quarters = {}, set()
    for view in VIEWS:
        views[view], usable = view_inventory(lc2.photometry(raw, view))
        quarters.update(usable)
    return {
        "source_id": sid, "cohort": row["cohort"], "role": row["role"],
        "source_domain": row["source_domain"],
        "origin_parent_kic": int(row["origin_parent"]),
        "raw_fits_sha256": row["raw_fits_sha256"],
        "input_manifest": str(manifest.resolve()) if manifest else None,
        "required_pixel_quarters": sorted(quarters), "views": views,
        "pixel_coverage_status": "NOT_YET_VERIFIED",
        "scientific_outcomes_opened": False,
    }


def inventory_sources(root, selected, manifests, output):
    completed, requested = Counter(), Counter()
    with (output / "sources.jsonl").open("x") as stream:
        for number, row in enumerate(selected, 1):
            result = source_inventory(root, row, manifests.get(row["cohort"]))
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            key = row["cohort"] + ":" + row["role"]
            completed[key] += 1
            requested[key] += len(result["required_pixel_quarters"])
            if number % 10 == 0 or number == len(selected):
                save(output / "progress.json", {
                    "status": "RUNNING", "pid": os.getpid(),
                    "completed_sources": number, "declared_sources": len(selected),
                    "completed_by_cohort": dict(completed),
                    "requested_quarters_by_cohort": dict(requested),
                })
            if peak_rss_bytes() > MAXIMUM_RSS:
                raise MemoryError("The input inventory exceeded its memory limit")
            if shutil.disk_usage(output).free < RESERVE:
                raise OSError("The input inventory approached its disk reserve")
    return completed, requested


def run_inventory(args):
    root, output, graph = args.root.resolve(), args.output, args.graph.resolve()
    for volume in ("discSSD", "discPCIe"):
        if not (root / volume).resolve().is_mount():
            raise OSError(f"The original corpus volume is not mounted: {volume}")
    if shutil.disk_usage(output.parent).free < RESERVE + 1024**3:
        raise OSError("Insufficient output allowance above the disk reserve")
    selected = declared_sources(graph)
    index = (root / "discSSD/data/index.sqlite").resolve()
    inputs = {str(path.resolve()): digest(path) for path in (
        Path(__file__), Path(lc2.__file__), graph / "audit.json",
        graph / "trials.csv", index, root / "discSSD/data/index.json",
        root / "discPCIe/data/index.json",
    )}
    manifests = {
        "native_eb": args.native_manifest.resolve(),
        "catalogue_fp": args.catalogue_fp_manifest.resolve(),
    }
    locations, hashes = corpus_index(index)
    selected.sort(key=lambda row: source_order(row, locations))
    bind_source_inputs(root, selected, manifests, inputs, locations, hashes)
    output.mkdir()
    save(output / "experiment.json", {
        "status": "PREDECLARED_OUTCOME_BLIND_REAL_CONTROL_QUARTER_INVENTORY",
        "inputs": inputs, "declared_sources": len(selected),
        "cohorts": dict(Counter(row["cohort"] + ":" + row["role"]
                                for row in selected)),
        "ordered_sources": [{key: row[key] for key in
                             ("source_id", "cohort", "role")}
                            for row in selected],
        "workers": 1, "maximum_self_rss_bytes": MAXIMUM_RSS,
        "minimum_free_bytes": RESERVE,
        "measurement_policy": (
            "Read original input arrays and verify source, parent, raw and domain "
            "identities. Fingerprint every view array and count quarter cadence "
            "availability under quality-zero and flatten-zero masks separately. "
            "Request the union of usable quarters across supplied views. Only "
            "PSF_FLAT is the discovery view; other views are correlated checks."
        ),
        "boundary": (
            "Input identities and availability only. No detector, transit fit, "
            "eligibility choice or scientific outcome is computed. Actual TPF "
            "identity, stamp coverage and joint independence remain unverified."
        ),
        "scientific_outcomes_opened": False, "boi_catalogue_enabled": False,
    })
    start, cpu = time.monotonic(), time.process_time()
    completed, requested = inventory_sources(root, selected, manifests, output)
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError(f"A frozen inventory input changed: {path}")
    save(output / "audit.json", {
        "status": "COMPLETE_OUTCOME_BLIND_REAL_CONTROL_QUARTER_INVENTORY",
        "declared_sources": len(selected), "completed_by_cohort": dict(completed),
        "requested_quarters_by_cohort": dict(requested),
        "requested_parent_quarters": sum(requested.values()),
        "experiment_sha256": digest(output / "experiment.json"),
        "sources_sha256": digest(output / "sources.jsonl"),
        "wall_seconds": time.monotonic() - start,
        "cpu_seconds": time.process_time() - cpu,
        "peak_self_rss_bytes": peak_rss_bytes(),
        "actual_tpf_coverage_verified": False, "joint_independence_complete": False,
        "scientific_outcomes_opened": False, "boi_catalogue_enabled": False,
    })
    save(output / "progress.json", {
        "status": "COMPLETE", "pid": os.getpid(),
        "completed_sources": len(selected), "declared_sources": len(selected),
    })
    print("Complete real-control quarter inventory", len(selected),
          sum(requested.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--catalogue-fp-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.nice(10)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"The inventory output already exists: {args.output}")
    try:
        run_inventory(args)
    except BaseException as error:
        if args.output.exists():
            save(args.output / "failure.json", {
                "status": "FAILED_INCOMPLETE_INPUT_INVENTORY",
                "error": f"{type(error).__name__}: {error}",
            })
        raise


if __name__ == "__main__":
    main()
