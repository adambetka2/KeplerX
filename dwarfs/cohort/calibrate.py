"""Freeze and replay detector-only physical injections in native M/K-dwarf curves."""


import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from itertools import islice, pairwise
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"
)

import numpy as np
from scipy.optimize import brentq

ROOT = Path(
    os.environ.get("SPACELAB_CALIBRATION_ROOT", Path(__file__).resolve().parents[2])
)
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from pipeline.deterministic import injections
from pipeline.deterministic.run import InsufficientCadences, prepare_curve_records
from pipeline.photometry.lc2 import curve_digest, load_curve

PLAN = Path(__file__).with_name("calibration-plan.json")
RAW_FIELDS = ("time", "flux", "error", "quality", "flatten", "cadence", "quarter")
DETECTOR = None
SCAN_CONFIG = None
DIAGNOSTICS_ENABLED = False
STORE_PERIODOGRAMS = False
PREPARED_RECEIPTS = None


class YearScrambleIneligible(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_input(path):
    source = read(path)
    for product, entry in (
        source.get("native_input_verification", {}).get("products", {}).items()
    ):
        if file_stamp(product) != entry["stamp"]:
            raise ValueError("Verified original LC2 product changed after preparation")
    for key in ("arrays", "raw_arrays"):
        entry = source.get(key)
        if entry and digest(Path(path).parent / entry["path"]) != entry["sha256"]:
            raise ValueError("Prepared or raw immutable arrays changed")
    if source["status"] == "PREPARED" and not source.get("arrays"):
        raise ValueError("Prepared trial lacks arrays")
    if (
        source.get("schema", 1) >= 2
        and source["status"] == "INPUT_INELIGIBLE"
        and (
            not source.get("raw_arrays")
            or source.get("first_failure_stage")
            not in (
                "ORIGINAL_PREPARATION",
                "INJECTED_PREPARATION",
            )
        )
    ):
        raise ValueError("Ineligible trial lacks raw arrays or its first failure stage")
    return source


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), allow_nan=False, indent=2) + "\n"
    )
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    number = float(value) if value not in (None, "") else None
    return number if number is not None and math.isfinite(number) else None


def storage_check(out, anticipated_bytes=0):
    volumes = {name: Path("/Volumes") / name for name in ("discSSD", "discPCIe")}
    for name, volume in volumes.items():
        if not volume.is_mount() or (ROOT / name).resolve() != volume:
            raise ValueError(f"Original corpus volume is unavailable: {name}")
        if shutil.disk_usage(volume).free < 2 * 1024**3:
            raise ValueError(f"Original corpus reserve is below 2 GiB: {name}")
    destination = out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(destination).free
    if available < 40 * 1024**3 + anticipated_bytes:
        raise ValueError("Actual output destination lacks the 40 GiB working reserve")
    return {
        "destination": str(destination),
        "free_bytes": available,
        "anticipated_bytes": anticipated_bytes,
        "input_volume_free_bytes": {
            name: shutil.disk_usage(path).free for name, path in volumes.items()
        },
    }


def periodogram_storage_bound(config, count, null_count, store_periodograms):
    """Bound dense-grid numerical arrays without assuming file compression."""
    if not store_periodograms:
        return count * 512 * 1024
    bands = [config] + [
        config | config[name]
        for name in ("short_period_search", "long_period_search")
        if name in config
    ]
    if not any("period_grid_correlation" in band for band in bands):
        return count * 5 * 1024**2
    per_source = 0
    for band in bands:
        capacity_key = (
            "maximum_period_trials"
            if "period_grid_correlation" in band
            else "period_grid_size"
        )
        sizes = {
            capacity_key: band[capacity_key],
            "max_refined_peaks_per_band": band.get(
                "max_refined_peaks_per_band", band[capacity_key]
            ),
            "period_refine_points": band["period_refine_points"],
        }
        for name, value in sizes.items():
            minimum = 2 if name == capacity_key else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"Dense periodogram bound requires integral {name}")
        per_source += (
            (
                int(sizes[capacity_key])
                + int(sizes["max_refined_peaks_per_band"])
                * int(sizes["period_refine_points"])
            )
            * 5
            * 8
        )
    return per_source * (count + null_count)


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def year_scramble(records):
    """Use the DR25-style SCR1 year order with explicitly reconstructed clocks."""
    required = {"time", "cadence", "quarter", "flux", "error"}
    if records.ndim != 1 or not required.issubset(records.dtype.names or ()):
        raise ValueError("Year scrambling requires structured photometry records")
    if not np.array_equal(np.unique(records["quarter"]), np.arange(1, 18)):
        raise YearScrambleIneligible("SCRAMBLE_REQUIRES_EXACT_QUARTERS_1_TO_17")
    if (
        not np.isfinite(records["time"]).all()
        or not np.isfinite(records["flux"]).all()
        or not np.isfinite(records["error"]).all()
        or not (records["error"] > 0).all()
        or not np.issubdtype(records["time"].dtype, np.floating)
        or not np.issubdtype(records["cadence"].dtype, np.integer)
        or (records["cadence"] < 0).any()
        or (records["cadence"] > np.iinfo(np.int64).max).any()
        or not (np.diff(records["time"]) > 0).all()
        or not (np.diff(records["cadence"].astype(np.int64)) > 0).all()
        or not (np.diff(records["quarter"].astype(np.int64)) >= 0).all()
    ):
        raise ValueError("Year scrambling requires finite ordered original records")
    groups = (records["quarter"] - 1) // 4
    blocks = [np.flatnonzero(groups == year) for year in range(5)]
    starts = np.array([block[0] for block in blocks])
    stops = np.array([block[-1] for block in blocks])
    time_gap = records["time"][starts[1:]] - records["time"][stops[:-1]]
    cadence_gap = records["cadence"][starts[1:]].astype(np.int64) - records["cadence"][
        stops[:-1]
    ].astype(np.int64)
    order = np.concatenate([blocks[year] for year in (3, 2, 1, 0, 4)])
    result = records[order].copy()
    time_cursor = float(records["time"][0])
    cadence_cursor = int(records["cadence"][0])
    offsets = []
    position = 0
    for slot, year in enumerate((3, 2, 1, 0, 4)):
        block = blocks[year]
        time_offset = time_cursor - float(records["time"][block[0]])
        cadence_offset = cadence_cursor - int(records["cadence"][block[0]])
        if year == 4:
            time_offset = 0.0
        selected = slice(position, position + len(block))
        shifted = records["cadence"][block].astype(np.int64) + cadence_offset
        limits = np.iinfo(records["cadence"].dtype)
        if np.any(shifted < limits.min) or np.any(shifted > limits.max):
            raise ValueError("Scrambled cadence identifiers exceed their input dtype")
        result["time"][selected] = records["time"][block] + time_offset
        result["cadence"][selected] = shifted
        offsets.append(
            {
                "input_year": year + 1,
                "output_slot": slot + 1,
                "quarters": sorted(map(int, np.unique(records["quarter"][block]))),
                "rows": len(block),
                "time_offset_days": time_offset,
                "cadence_offset": cadence_offset,
            }
        )
        position += len(block)
        if slot < 4:
            time_cursor = float(result["time"][position - 1]) + float(time_gap[slot])
            cadence_cursor = int(result["cadence"][position - 1]) + int(
                cadence_gap[slot]
            )
    mapping = {
        "input_row_index": order,
        "output_row_index": np.arange(len(records)),
        "input_time": records["time"][order].copy(),
        "output_time": result["time"].copy(),
        "input_cadence": records["cadence"][order].copy(),
        "output_cadence": result["cadence"].copy(),
        "input_quarter": records["quarter"][order].copy(),
    }
    if (
        not (np.diff(result["time"]) > 0).all()
        or not (np.diff(result["cadence"].astype(np.int64)) > 0).all()
        or result[result["quarter"] == 17].tobytes()
        != records[records["quarter"] == 17].tobytes()
    ):
        raise ValueError("Scrambled clocks are unordered or Q17 changed")
    return (
        result,
        mapping,
        {
            "variant": "dr25_style_scr1_year_permutation",
            "year_order": [4, 3, 2, 1],
            "q17_last": True,
            "quarter_order": [
                *range(13, 17),
                *range(9, 13),
                *range(5, 9),
                *range(1, 5),
                17,
            ],
            "blocks": offsets,
            "input_boundary_time_gaps_days": time_gap.tolist(),
            "input_boundary_cadence_gaps": cadence_gap.tolist(),
        },
    )


def verified_native_inputs(out, row):
    path = out / "native-input-verification.json"
    payload = path.read_bytes()
    ledger = json.loads(payload)
    index = read(out / "freeze.json")["native_index"]
    if (
        ledger["index"]["sha256"] != index["sha256"]
        or file_stamp(index["path"]) != index["stamp"]
    ):
        raise ValueError("Frozen native index identity changed")
    verified = {}
    for name in ("source.parquet", "cadence.parquet", "qdiag.parquet"):
        product = (Path(row["lc2_path"]) / name).resolve()
        entry = ledger["products"].get(str(product))
        if (
            not entry
            or entry["status"] != "VERIFIED"
            or entry["stamp"] != file_stamp(product)
        ):
            raise ValueError(
                f"Native LC2 product has no current verified identity: {product}"
            )
        receipt = entry["receipt"]
        if digest(receipt["path"]) != receipt["sha256"] or read(receipt["path"]) != {
            key: value for key, value in entry.items() if key != "receipt"
        }:
            raise ValueError("Native input verification receipt changed")
        verified[str(product)] = entry
    return {"ledger_sha256": hashlib.sha256(payload).hexdigest(), "products": verified}


def freeze(args):
    out = args.out
    storage = storage_check(out)
    if (out / "freeze.json").exists():
        raise ValueError("This experiment is already frozen")
    population = read_rows(args.population)
    source_ids = [row["source_id"] for row in population]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Population identities are duplicated")
    parents = {name: set() for name in ("development", "confirmation")}
    selected = []
    for partition, count in (("development", 5000), ("confirmation", 10000)):
        declared = "evaluation" if partition == "confirmation" else partition
        pools = {
            kind: sorted(
                (
                    row
                    for row in population
                    if row["population"] == kind and row["split"] == declared
                ),
                key=lambda row: row["source_rank_sha256"],
            )
            for kind in ("M", "K")
        }
        if any(len(rows) < count // 2 for rows in pools.values()):
            raise ValueError(f"Insufficient frozen {partition} population capacity")
        for index in range(count):
            kind = "M" if index % 2 == 0 else "K"
            row = pools[kind][index // 2]
            if not row["parent_tpf"] or not row["raw_fits_sha256"]:
                raise ValueError("Population lacks original input/parent identity")
            parents[partition].add(row["parent_tpf"])
            selected.append(
                dict(
                    row,
                    partition=partition,
                    trial_index=index + 1,
                    depth_bin=(index // 2) % 6,
                )
            )
    if parents["development"] & parents["confirmation"]:
        raise ValueError("Development and confirmation share recorded parent TPFs")
    if len({row["source_id"] for row in selected}) != len(selected):
        raise ValueError("A source was selected more than once")
    out.mkdir(parents=True, exist_ok=True)
    generator = out / "generator"
    generator.mkdir()
    shutil.copyfile(Path(__file__), generator / "calibrate.py")
    shutil.copyfile(PLAN, generator / "calibration-plan.json")
    shutil.copyfile(PLAN, out / "plan.json")
    write_rows(out / "trials.csv", selected)
    save(
        out / "freeze.json",
        {
            "schema": 1,
            "status": "FROZEN_DETECTOR_INJECTION_TRIALS",
            "plan_sha256": digest(out / "plan.json"),
            "population": {
                "path": str(args.population.resolve()),
                "sha256": digest(args.population),
                "sources": len(population),
            },
            "trials_sha256": digest(out / "trials.csv"),
            "generator_sha256": digest(Path(__file__)),
            "generator_archive": str((generator / "calibrate.py").resolve()),
            "preparation_wrapper_sha256": digest(
                ROOT / "pipeline/deterministic/run.py"
            ),
            "preparation_sha256": digest(
                ROOT / "pipeline/deterministic/signal_contracts.py"
            ),
            "transit_machinery_sha256": digest(
                ROOT / "pipeline/deterministic/injections.py"
            ),
            "reader_sha256": digest(ROOT / "pipeline/photometry/lc2.py"),
            "native_index": {
                "path": str((ROOT / "discSSD/data/index.sqlite").resolve()),
                "sha256": digest(ROOT / "discSSD/data/index.sqlite"),
                "stamp": file_stamp(ROOT / "discSSD/data/index.sqlite"),
            },
            "environment": {
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "scipy",
                    "batman-package",
                    "astropy",
                    "pyarrow",
                )
            },
            "selected": dict(Counter(row["partition"] for row in selected)),
            "storage": storage,
            "population_references": {
                path.name: {"path": str(path.resolve()), "sha256": digest(path)}
                for path in (
                    args.population.with_name("population.json"),
                    args.population.with_name("population-verification.json"),
                    args.population.with_name("host-geometry-model.json"),
                    args.population.with_name("host-geometry-verification.json"),
                )
            },
            "nonfinite_diagnostic_encoding": (
                "JSON strings NaN, +Infinity and -Infinity preserve nonfinite "
                "values explicitly."
            ),
            "parent_counts": {name: len(values) for name, values in parents.items()},
            "claim_boundary": read(out / "plan.json")["claim_boundary"],
            "parent_independence": (
                "Recorded original parent TPF IDs; shared spacecraft epochs "
                "remain correlated across partitions."
            ),
        },
    )
    print(json.dumps({"status": "FROZEN", "selected": len(selected)}), flush=True)


def frozen_trials(out, partition, count):
    freeze = read(out / "freeze.json")
    if (
        digest(out / "trials.csv") != freeze["trials_sha256"]
        or digest(out / "plan.json") != freeze["plan_sha256"]
    ):
        raise ValueError("Frozen experiment identities changed")
    rows = [
        row for row in read_rows(out / "trials.csv") if row["partition"] == partition
    ]
    if count < 1 or count > len(rows):
        raise ValueError("Trial count exceeds the frozen partition")
    return rows[:count], read(out / "plan.json"), freeze


def year_scramble_admission(args, rows, freeze):
    if args.partition != "development" or Path(args.label).name != args.label:
        raise ValueError(
            "Year controls require development and a single directory label"
        )
    if args.label in ("", ".", ".."):
        raise ValueError("Year control label is empty or reserved")
    folder = args.out / "checks" / f"year-scramble-{args.label}"
    if folder.exists():
        raise ValueError("Year controls require a new output directory")
    binding_path = args.out / "checks/native-input-binding.json"
    binding = read(binding_path)
    ledger_path = args.out / "native-input-verification.json"
    ledger = read(ledger_path)
    if (
        binding["experiment_freeze"]["sha256"] != digest(args.out / "freeze.json")
        or binding["native_input_ledger"]["sha256"] != digest(ledger_path)
        or ledger["status"] != "COMPLETE"
        or binding["index"] != ledger["index"]
        or ledger["index"]["sha256"] != freeze["native_index"]["sha256"]
        or file_stamp(freeze["native_index"]["path"]) != freeze["native_index"]["stamp"]
        or digest(freeze["native_index"]["path"]) != freeze["native_index"]["sha256"]
    ):
        raise ValueError("Year control native input binding changed")
    manifest_path = (
        args.out / f"prepared-{args.view}-{args.partition}-{args.count}.json"
    )
    manifest = read(manifest_path)
    if manifest["count"] != len(rows) or set(manifest["receipt_sha256"]) != {
        row["source_id"] for row in rows
    }:
        raise ValueError("Year control prepared membership changed")
    paths = [
        Path(__file__),
        args.out / "freeze.json",
        args.out / "plan.json",
        args.out / "trials.csv",
        ledger_path,
        binding_path,
        manifest_path,
        Path(freeze["native_index"]["path"]),
    ]
    return {
        "folder": folder,
        "manifest": manifest,
        "ledger": ledger,
        "bindings": {str(path): digest(path) for path in paths},
        "input_folder": args.out / "inputs" / args.view / args.partition,
    }


def checked_year_original(row, context):
    path = context["input_folder"] / f"{row['source_id']}.json"
    expected = context["manifest"]["receipt_sha256"][row["source_id"]]
    if digest(path) != expected:
        raise ValueError("Year control input receipt changed")
    source = checked_input(path)
    native = source["native_input_verification"]
    ledger_hash = context["bindings"][
        str(path.parents[3] / "native-input-verification.json")
    ]
    expected_products = {
        str((Path(row["lc2_path"]) / name).resolve())
        for name in ("source.parquet", "cadence.parquet", "qdiag.parquet")
    }
    if (
        source["trial"] != row
        or native["ledger_sha256"] != ledger_hash
        or set(native["products"]) != expected_products
    ):
        raise ValueError("Year control source or native ledger changed")
    for product, entry in native["products"].items():
        if (
            context["ledger"]["products"].get(product) != entry
            or digest(entry["receipt"]["path"]) != entry["receipt"]["sha256"]
        ):
            raise ValueError("Year control original product receipt changed")
    if digest(path) != expected:
        raise ValueError("Year control receipt changed while reading")
    return source


def construct_year_control(row, source, context):
    sid, folder = row["source_id"], context["folder"]
    result = {
        "schema": 1,
        "source_id": sid,
        "trial": row,
        "variant": "dr25_style_scr1_year_permutation",
        "input_receipt_sha256": context["manifest"]["receipt_sha256"][sid],
        "status": "INPUT_INELIGIBLE",
        "reason": source["reason"],
        "first_failure_stage": "SOURCE_INPUT_PREPARATION",
        "native_input_verification": source["native_input_verification"],
        "detector_searched": False,
    }
    if source["status"] == "PREPARED":
        array_path = context["input_folder"] / source["arrays"]["path"]
        with np.load(array_path, allow_pickle=False) as archive:
            original = archive["original"]
        if (
            hashlib.sha256(original.tobytes()).hexdigest()
            != source["original_prepared_sha256"]
        ):
            raise ValueError("Year control original prepared bytes changed")
        result["input_original_prepared_sha256"] = source["original_prepared_sha256"]
        try:
            records, mapping, construction = year_scramble(original)
        except YearScrambleIneligible as error:
            result.update(
                status="SCRAMBLE_INELIGIBLE",
                reason=error.reason,
                first_failure_stage="YEAR_PERMUTATION",
            )
        else:
            arrays = {"records": records, **mapping}
            result.update(
                status="CONSTRUCTED",
                reason=None,
                first_failure_stage=None,
                construction=construction,
                output_prepared_sha256=hashlib.sha256(records.tobytes()).hexdigest(),
                arrays=save_arrays(folder / f"{sid}.npz", **arrays),
                array_identity={
                    name: {
                        "dtype": array.dtype.descr,
                        "shape": list(array.shape),
                        "bytes": array.nbytes,
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    }
                    for name, array in arrays.items()
                },
            )
    if checked_year_original(row, context) != source:
        raise ValueError("Year control source changed during construction")
    save(folder / f"{sid}.json", result)
    return result


def year_scramble_inputs(args):
    rows, _, freeze = frozen_trials(args.out, args.partition, args.count)
    context = year_scramble_admission(args, rows, freeze)
    sources, anticipated = {}, 0
    for row in rows:
        source = checked_year_original(row, context)
        sources[row["source_id"]] = source
        if source["status"] == "PREPARED":
            with zipfile.ZipFile(
                context["input_folder"] / source["arrays"]["path"]
            ) as archive:
                anticipated += 3 * archive.getinfo("original.npy").file_size + 8192
    storage = storage_check(args.out / "checks", anticipated_bytes=anticipated)
    folder = context["folder"]
    folder.mkdir()
    shutil.copyfile(__file__, folder / "calibrate.py")
    identity = {
        "schema": 1,
        "label": args.label,
        "partition": args.partition,
        "view": args.view,
        "requested": len(rows),
        "source_ids": [row["source_id"] for row in rows],
        "bindings": context["bindings"],
        "prepared_receipt_sha256": context["manifest"]["receipt_sha256"],
        "storage_admission": storage,
        "runtime": {
            "python": sys.version,
            "python_executable_sha256": digest(sys.executable),
            "numpy": np.__version__,
        },
        "reference": "https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19114-002.pdf",
        "claim_boundary": "Construction only; no search, recovery or false-alarm rate",
    }
    save(folder / "identity.json", identity)
    results = [
        construct_year_control(row, sources[row["source_id"]], context) for row in rows
    ]
    for row in rows:
        if checked_year_original(row, context) != sources[row["source_id"]]:
            raise ValueError("Year control input changed before completion")
    if any(digest(path) != value for path, value in context["bindings"].items()):
        raise ValueError("Year control dependency changed during construction")
    if file_stamp(freeze["native_index"]["path"]) != freeze["native_index"]["stamp"]:
        raise ValueError("Year control native index stamp changed")
    result = {
        "status": "COMPLETE_CONTROL_CONSTRUCTION",
        "requested": len(rows),
        "constructed": sum(item["status"] == "CONSTRUCTED" for item in results),
        "statuses": dict(Counter(item["status"] for item in results)),
        "ineligibility_reasons": dict(
            Counter(item["reason"] for item in results if item["reason"])
        ),
        "receipt_sha256": {
            row["source_id"]: digest(folder / f"{row['source_id']}.json")
            for row in rows
        },
        "identity_sha256": digest(folder / "identity.json"),
        "before_after_bindings_equal": True,
        "detector_searched": 0,
        "claim_boundary": identity["claim_boundary"],
    }
    save(folder / "manifest.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "requested",
                    "constructed",
                    "statuses",
                    "detector_searched",
                )
            }
        ),
        flush=True,
    )
    return result


def limb_darkening_references(model):
    if (
        model["mode"] != "CLARET_PHOENIX_KEPLER_QUADRATIC_V1"
        or model["selection"]
        != {"filter": "Kp", "method": "L", "model": "qs", "Z": 0, "xi_kms": 2}
        or model["interpolation"] != "ACTUAL_GRAVITY_ROWS_THEN_TEMPERATURE"
        or model["solar_logg"] != math.log10(1.3271244e20 / 6.957e8**2 * 100)
        or [item["year"] for item in model["references"]] != [2012, 2013]
    ):
        raise ValueError("Unsupported host-conditioned limb-darkening recipe")
    bindings = {
        str(Path(item[key + "_path"]).resolve()): item[key + "_sha256"]
        for item in model["references"]
        for key in ("table", "readme")
    }
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("Host-conditioned primary reference changed")
    return bindings


@lru_cache(maxsize=2)
def limb_darkening_grid(references):
    nodes = {}
    for year, path, expected in references:
        if digest(path) != expected:
            raise ValueError("Limb-darkening table changed before cached parsing")
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) != 9:
                raise ValueError("Expected nine-field PHOENIX coefficient row")
            if fields[6:] != ["Kp", "L", "qs"]:
                continue
            gravity, temperature, metallicity, velocity, u1, u2 = map(float, fields[:6])
            if metallicity != 0 or velocity != 2 or (gravity, temperature) in nodes:
                raise ValueError("Atmosphere selection or node uniqueness changed")
            nodes[gravity, temperature] = {
                "logg": gravity,
                "teff_k": temperature,
                "u1": u1,
                "u2": u2,
                "year": year,
                "table_line": number,
                "table_sha256": expected,
            }
    if len(nodes) != 427:
        raise ValueError("Kepler PHOENIX node membership changed")
    return nodes


def model_bracket(values, target):
    values = sorted(set(values))
    if not math.isfinite(target) or not values[0] <= target <= values[-1]:
        raise ValueError("Host coordinate is outside the declared atmosphere grid")
    index = int(np.searchsorted(values, target))
    if values[index] == target:
        return target, target, 0.0
    lower, upper = values[index - 1], values[index]
    return lower, upper, (target - lower) / (upper - lower)


def host_limb_darkening(row, model):
    bindings = limb_darkening_references(model)
    references = tuple(
        (item["year"], str(Path(item["table_path"]).resolve()), item["table_sha256"])
        for item in model["references"]
    )
    nodes = limb_darkening_grid(references)
    mass, radius, temperature = (
        finite(row.get(key))
        for key in ("inferred_mass_solar", "inferred_radius_solar", "source_teff_k")
    )
    if any(value is None or value <= 0 for value in (mass, radius, temperature)):
        raise ValueError("Host-conditioned recipe requires fiducial mass/radius/Teff")
    gravity = model["solar_logg"] + math.log10(mass) - 2 * math.log10(radius)
    lower, upper, fraction = model_bracket([key[0] for key in nodes], gravity)
    selected = []
    for value, gravity_weight in ((lower, 1 - fraction), (upper, fraction)):
        left, right, weight = model_bracket(
            [key[1] for key in nodes if key[0] == value], temperature
        )
        for coordinate, temperature_weight in ((left, 1 - weight), (right, weight)):
            selected.append(
                dict(
                    nodes[value, coordinate], weight=gravity_weight * temperature_weight
                )
            )
    coefficients = [
        sum(node["weight"] * node[key] for node in selected) for key in ("u1", "u2")
    ]
    if (
        any(node["weight"] < 0 for node in selected)
        or abs(sum(node["weight"] for node in selected) - 1) >= 1e-14
        or abs(sum(node["weight"] * node["teff_k"] for node in selected) - temperature)
        >= 1e-9
        or abs(sum(node["weight"] * node["logg"] for node in selected) - gravity)
        >= 1e-12
    ):
        raise ValueError(
            "Actual-node interpolation does not reproduce host coordinates"
        )
    u1, u2 = coefficients
    if not (u1 >= 0 and u1 + 2 * u2 >= 0 and u1 + u2 <= 1):
        raise ValueError("Nonphysical quadratic limb-darkening intensity")
    if bindings != limb_darkening_references(model):
        raise ValueError("Primary reference changed during host interpolation")
    return coefficients, {
        "model": model,
        "teff_k": temperature,
        "fiducial_logg": gravity,
        "gravity_basis": "FROZEN_FIDUCIAL_MASS_RADIUS_NOT_MEASURED_HOST_GRAVITY",
        "nodes": selected,
        "maximum_temperature_span_k": max(
            selected[index + 1]["teff_k"] - selected[index]["teff_k"]
            for index in (0, 2)
        ),
        "cross_2012_2013_bridge": len(
            {node["year"] for node in selected if node["weight"] > 0}
        )
        > 1,
        "reference_sha256": bindings,
    }


def synthetic_truth(row, plan):
    seed, sid, partition = plan["seed"], row["source_id"], row["partition"]
    parts = (partition, sid)
    period = injections.log_uniform(seed, *plan["period_days"], *parts, "period")
    density = finite(row.get("inferred_density_solar"))
    if density is None or density <= 0:
        raise ValueError("Frozen trial lacks its declared fiducial stellar density")
    impact = injections.uniform(seed, *plan["impact_parameter"], *parts, "impact")
    depth = injections.log_uniform(
        seed, *plan["depth_bins_ppm"][int(row["depth_bin"])], *parts, "depth"
    )
    epoch = 120.0 + period * injections.unit(seed, *parts, "phase")
    model_provenance = None
    if "limb_darkening_model" in plan:
        coefficients, model_provenance = host_limb_darkening(
            row, plan["limb_darkening_model"]
        )
    else:
        coefficients = [
            injections.uniform(seed, *plan[key], *parts, key)
            for key in ("limb_darkening_u1", "limb_darkening_u2")
        ]
    signal = {
        "exposure_days": plan["exposure_days"],
        "supersample_factor": plan["supersample_factor"],
        "limb_darkening": coefficients,
    }

    def integrated_depth(radius):
        axis, inclination, duration = injections.orbit(period, density, impact, radius)
        return injections.transit_loss(
            np.array([epoch]),
            [epoch],
            period,
            (radius, axis, inclination, duration),
            signal,
            epoch,
        )[0]

    radius = brentq(
        lambda value: integrated_depth(value) - depth * 1e-6, 1e-5, 0.3, xtol=1e-13
    )
    axis, inclination, duration = injections.orbit(period, density, impact, radius)
    truth = {
        "source_id": sid,
        "population": row["population"],
        "partition": partition,
        "trial_index": int(row["trial_index"]),
        "depth_bin": int(row["depth_bin"]),
        "period_days": period,
        "shape_period_days": period,
        "epoch_bkjd": epoch,
        "category": "periodic_injection",
        "radius_ratio": radius,
        "scaled_semimajor_axis": axis,
        "inclination_degrees": inclination,
        "duration_hours": duration * 24,
        "impact_parameter": impact,
        "stellar_density_solar": density,
        "stellar_density_origin": row["geometry_basis"],
        "inferred_mass_solar": finite(row.get("inferred_mass_solar")),
        "inferred_radius_solar": finite(row.get("inferred_radius_solar")),
        "target_integrated_depth_ppm": depth,
        "integrated_depth_ppm": integrated_depth(radius) * 1e6,
        "signal": signal,
        "parent_tpf": row["parent_tpf"],
        "base_raw_fits_sha256": row["raw_fits_sha256"],
        "phot_g_mean_mag": finite(row.get("phot_g_mean_mag")),
        "teff_gspphot": finite(row.get("teff_gspphot")),
    }
    if model_provenance is not None:
        truth["limb_darkening_provenance"] = model_provenance
    return truth


def loss(times, truth):
    times = np.asarray(times, dtype=np.float64)
    result = np.zeros(len(times))
    phase = np.abs(
        (times - truth["epoch_bkjd"] + truth["period_days"] / 2) % truth["period_days"]
        - truth["period_days"] / 2
    )
    extent = truth["duration_hours"] / 48 + truth["signal"]["exposure_days"] / 2 + 1e-9
    use = np.isfinite(times) & (phase <= extent)
    if not use.any():
        return result
    geometry = (
        truth["radius_ratio"],
        truth["scaled_semimajor_axis"],
        truth["inclination_degrees"],
        truth["duration_hours"] / 24,
    )
    result[use] = injections.transit_loss(
        times[use],
        [],
        truth["period_days"],
        geometry,
        truth["signal"],
        truth["epoch_bkjd"],
    )
    return result


def load_native(row, requested_view):
    availability = {}
    views = (
        ("psf_qdiag", "sap_qdiag")
        if requested_view == "preferred_full"
        else (requested_view,)
    )
    for view in views:
        curve = load_curve(ROOT, row["source_id"], view)
        usable = (
            np.isfinite(curve["time"])
            & np.isfinite(curve["flux"])
            & np.isfinite(curve["error"])
        )
        usable &= (curve["error"] > 0) & (curve["quality"] == 0)
        if view == "psf_flat":
            usable &= curve["flatten"] == 0
        availability[view] = {
            "stored_cadences": len(usable),
            "usable_cadences": int(usable.sum()),
            "curve_sha256": curve_digest(curve),
        }
        if requested_view != "preferred_full" or usable.sum() >= 100:
            return curve, view, availability
    return curve, view, availability


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}


def preserve_failed_preparation(path, saved, curve, injected, loss, delta, original):
    arrays = {f"raw_{key}": np.asarray(curve[key]) for key in RAW_FIELDS}
    arrays.update(raw_loss=loss, raw_delta=delta, injected_flux=injected["flux"])
    if original is not None:
        arrays["original"] = original
        saved["original_prepared_sha256"] = hashlib.sha256(
            original.tobytes()
        ).hexdigest()
    saved.update(
        raw_arrays=save_arrays(path, **arrays),
        available_prepared_arrays=["original"] if original is not None else [],
    )


def prepare_trial(task):
    out, row, plan, requested_view = task
    out = Path(out)
    folder = out / "inputs" / requested_view / row["partition"]
    receipt = folder / f"{row['source_id']}.json"
    native_verification = verified_native_inputs(out, row)
    if receipt.exists():
        saved = checked_input(receipt)
        if saved["trial"] != row or saved["plan_sha256"] != digest(out / "plan.json"):
            raise ValueError("Prepared trial identity changed")
        return saved
    start = time.monotonic()
    truth = synthetic_truth(row, plan)
    curve, view, availability = load_native(row, requested_view)
    if any(
        file_stamp(path) != entry["stamp"]
        for path, entry in native_verification["products"].items()
    ):
        raise ValueError("Native LC2 product changed while loading a source")
    provenance = curve["provenance"]
    if (
        provenance["raw_fits_sha256"] != row["raw_fits_sha256"]
        or provenance["parent_tpf"] != row["parent_tpf"]
    ):
        raise ValueError("Native source/parent identity changed")
    native_hash = curve_digest(curve)
    native_loss = loss(curve["time"], truth)
    delta = np.zeros(len(native_loss))
    for quarter in np.unique(curve["quarter"]):
        use = curve["quarter"] == quarter
        valid = use & np.isfinite(curve["flux"]) & (curve["quality"] == 0)
        if valid.any():
            delta[use] = -float(np.median(curve["flux"][valid])) * native_loss[use]
    injected_curve = dict(
        curve, flux=np.asarray(curve["flux"], dtype=np.float64) + delta
    )
    truth["photometry_view"] = view
    saved = {
        "schema": 2,
        "trial": row,
        "truth": truth,
        "provenance": provenance,
        "requested_view": requested_view,
        "view_availability": availability,
        "view_selection_reason": "FULL_PSF_UNAVAILABLE"
        if requested_view == "preferred_full" and view == "sap_qdiag"
        else "DECLARED_VIEW",
        "plan_sha256": digest(out / "plan.json"),
        "native_curve_sha256": native_hash,
        "native_input_verification": native_verification,
        "injected_curve_sha256": curve_digest(injected_curve),
    }
    original = None
    stage = "ORIGINAL_PREPARATION"
    try:
        original = prepare_curve_records(curve, mask_flatten=view == "psf_flat")
        stage = "INJECTED_PREPARATION"
        injected = prepare_curve_records(
            injected_curve, mask_flatten=view == "psf_flat"
        )
    except InsufficientCadences as error:
        saved.update(
            status="INPUT_INELIGIBLE", reason=str(error), first_failure_stage=stage
        )
        preserve_failed_preparation(
            folder / f"{row['source_id']}.raw.npz",
            saved,
            curve,
            injected_curve,
            native_loss,
            delta,
            original,
        )
    else:
        sampled = loss(injected["time"], truth)
        truth["sampled_depth_ppm"] = float(sampled.max() * 1e6)
        truth["start_bkjd"] = float(injected["time"].min())
        truth["stop_bkjd"] = float(injected["time"].max())
        centres = [
            value
            for _, value in injections.event_centres(
                truth["period_days"],
                truth["epoch_bkjd"],
                truth["start_bkjd"],
                truth["stop_bkjd"],
            )
        ]
        events, quarters, points = injections.observed_events(
            injected,
            centres,
            truth["duration_hours"] / 24,
            truth["period_days"],
            truth["epoch_bkjd"],
        )
        truth.update(
            observed_events=events,
            observed_quarters=sorted(map(int, quarters)),
            in_transit_cadences=points,
        )
        try:
            sigma = injections.robust_point_sigma(original)
        except injections.InjectionIneligible:
            sigma = None
        truth["point_sigma_ppm"] = sigma * 1e6 if sigma else None
        truth["expected_white_mes"] = (
            float(np.linalg.norm(sampled) / sigma) if sigma else None
        )
        array_path = folder / f"{row['source_id']}.npz"
        array_receipt = save_arrays(
            array_path,
            original=original,
            injected=injected,
            raw_delta=delta,
            raw_loss=native_loss,
            prepared_loss=sampled,
            **{f"raw_{key}": np.asarray(curve[key]) for key in RAW_FIELDS},
        )
        saved.update(
            status="PREPARED",
            reason=None,
            first_failure_stage=None,
            arrays=array_receipt,
            original_prepared_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
            injected_prepared_sha256=hashlib.sha256(injected.tobytes()).hexdigest(),
        )
    saved["elapsed_seconds"] = time.monotonic() - start
    save(receipt, saved)
    return saved


def remodel_admission(args):
    bindings = {
        str(args.prior_manifest.resolve()): args.prior_sha256,
        str(args.plan.resolve()): args.plan_sha256,
        str(Path(__file__).resolve()): digest(__file__),
    }
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("Raw remodel prior manifest or recipe changed")
    prior, recipe = read(args.prior_manifest), read(args.plan)
    if (
        Path(recipe["prior_manifest"]["path"]).resolve()
        != args.prior_manifest.resolve()
        or recipe["prior_manifest"]["sha256"] != args.prior_sha256
    ):
        raise ValueError("Remodel recipe binds a different prior cohort manifest")
    design_entry = prior["design"]
    bindings[design_entry["path"]] = design_entry["sha256"]
    design = read(design_entry["path"])
    if recipe["prior_plan"] != design["physical_plan"]:
        raise ValueError("Remodel recipe does not bind the original physical plan")
    plan_entry = recipe["prior_plan"]
    bindings[plan_entry["path"]] = plan_entry["sha256"]
    physical = read(plan_entry["path"]) | {
        "limb_darkening_model": recipe["limb_darkening_model"],
        "geometry": recipe["geometry"],
        "protocol": recipe["protocol"],
        "limb_darkening_uncertainty": recipe["limb_darkening_uncertainty"],
    }
    bindings.update(limb_darkening_references(recipe["limb_darkening_model"]))
    for entry in prior["files"].values():
        bindings[entry["path"]] = entry["sha256"]
    cases = read_rows(prior["files"]["cases.csv"]["path"])
    if (
        prior["status"] != "COMPLETE_PREPARED_COHORT_NO_DETECTOR_SEARCH"
        or prior["requested_cases"] != 6000
        or len(cases) != 6000
        or not 1 <= args.count <= len(cases)
        or not 1 <= args.workers <= 2
        or any(row["kind"] != "synthetic_planet" for row in cases)
        or len({row["source_id"] for row in cases}) != len(cases)
    ):
        raise ValueError("Remodel requires the complete declared synthetic6000 parent")
    remodel_provenance(prior, recipe, bindings)
    remodel_selected_inputs(cases[: args.count], bindings)
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("Raw replay admission dependency differs")
    return cases[: args.count], physical, prior, bindings


def remodel_selected_inputs(cases, bindings):
    for case in cases:
        path = Path(case["receipt_path"])
        if digest(path) != case["receipt_sha256"]:
            raise ValueError("Prior selected source receipt changed")
        source = checked_input(path)
        array = source.get("arrays") or source["raw_arrays"]
        array_path = path.parent / array["path"]
        if (
            case["source_id"] != source["trial"]["source_id"]
            or source["trial"]["partition"] != "fresh_final"
            or source["requested_view"] != "preferred_full"
            or case["preparation_status"] != source["status"]
            or (
                source["status"] == "PREPARED"
                and (
                    Path(case["input_path"]).resolve() != array_path.resolve()
                    or case["input_sha256"] != array["sha256"]
                    or case["array_key"] != "injected"
                )
            )
        ):
            raise ValueError("Prior selected case/view/array identity differs")
        bindings[str(path)] = case["receipt_sha256"]
        bindings[str(array_path)] = array["sha256"]
        for product in source["native_input_verification"]["products"].values():
            entry = product["receipt"]
            bindings[entry["path"]] = entry["sha256"]


def remodel_provenance(prior, recipe, bindings):
    original = prior["preparation_identity"]
    dependencies = original["dependencies"]
    legacy = recipe["prior_generator"]
    expected = next(
        value
        for path, value in dependencies.items()
        if path.endswith("/shallow/calibrate.py")
    )
    if legacy["sha256"] != expected:
        raise ValueError("Archived original generator identity differs")
    bindings[legacy["path"]] = expected
    bindings.update(
        {
            path: value
            for path, value in dependencies.items()
            if not path.endswith(
                ("/shallow/calibrate.py", "/shallow/evolve_controls.py")
            )
        }
    )
    for path, entry in prior["input_bindings"].items():
        if Path(path).name in (
            "freeze.json",
            "native-input-verification.json",
            "native-input-binding.json",
            "index.sqlite",
        ):
            bindings[path] = entry["sha256"]
    index = original["native_index"]
    if file_stamp(index["path"]) != index["stamp"]:
        raise ValueError("Native index changed before raw replay")
    runtime = runtime_identity(
        {},
        {
            "environment": {
                name: value["version"]
                for name, value in original["runtime"]["libraries"].items()
            }
        },
    )
    if runtime != original["runtime"]:
        raise ValueError(
            "Raw replay numerical runtime differs from original construction"
        )


def replay_original_records(archive, source, curve):
    try:
        original = prepare_curve_records(
            curve, mask_flatten=source["truth"]["photometry_view"] == "psf_flat"
        )
    except InsufficientCadences as error:
        if (
            source["status"] != "INPUT_INELIGIBLE"
            or source["first_failure_stage"] != "ORIGINAL_PREPARATION"
            or source["reason"] != str(error)
            or "original" in archive
        ):
            raise ValueError(
                "Original preparation failure did not replay exactly"
            ) from error
        return None, str(error)
    if "original" not in archive:
        raise ValueError(
            "Original prepared records were not retained; no silent replay substitute"
        )
    retained = archive["original"]
    if (
        original.dtype != retained.dtype
        or original.shape != retained.shape
        or original.tobytes() != retained.tobytes()
        or hashlib.sha256(original.tobytes()).hexdigest()
        != source["original_prepared_sha256"]
    ):
        raise ValueError("Original prepared byte identity did not replay exactly")
    return original, None


def remodel_truth_statistics(truth, original, injected):
    sampled = loss(injected["time"], truth)
    truth.update(
        sampled_depth_ppm=float(sampled.max() * 1e6),
        start_bkjd=float(injected["time"].min()),
        stop_bkjd=float(injected["time"].max()),
    )
    events, quarters, points = injections.observed_events(
        injected,
        [],
        truth["duration_hours"] / 24,
        truth["period_days"],
        truth["epoch_bkjd"],
    )
    truth.update(
        observed_events=events,
        observed_quarters=sorted(map(int, quarters)),
        in_transit_cadences=points,
    )
    try:
        sigma = injections.robust_point_sigma(original)
    except injections.InjectionIneligible:
        sigma = None
    truth["point_sigma_ppm"] = sigma * 1e6 if sigma else None
    truth["expected_white_mes"] = (
        float(np.linalg.norm(sampled) / sigma) if sigma else None
    )
    return sampled


def raw_remodel_curve(curve, truth):
    native_loss = loss(curve["time"], truth)
    if not np.isfinite(native_loss).all() or native_loss.min() < -1e-10:
        raise ValueError("Raw remodel produced material invalid physical loss")
    delta = np.zeros(len(native_loss))
    for quarter in np.unique(curve["quarter"]):
        use = curve["quarter"] == quarter
        valid = use & np.isfinite(curve["flux"]) & (curve["quality"] == 0)
        if valid.any():
            delta[use] = -float(np.median(curve["flux"][valid])) * native_loss[use]
    return (
        dict(curve, flux=np.asarray(curve["flux"], dtype=np.float64) + delta),
        native_loss,
        delta,
    )


def remodel_trial(task):
    out, case, physical, recipe_sha = task
    out, prior_path = Path(out), Path(case["receipt_path"])
    if digest(prior_path) != case["receipt_sha256"]:
        raise ValueError("Original source receipt changed before remodel")
    source = checked_input(prior_path)
    array_entry = source.get("arrays") or source["raw_arrays"]
    prior_array = prior_path.parent / array_entry["path"]
    with np.load(prior_array, allow_pickle=False) as archive:
        curve = {key: archive["raw_" + key].copy() for key in RAW_FIELDS}
        curve["provenance"] = source["provenance"]
        if curve_digest(curve) != source["native_curve_sha256"]:
            raise ValueError(
                "Seven-field raw replay differs from the original native curve"
            )
        original, original_failure = replay_original_records(archive, source, curve)
        old_mask = (
            archive["injected"][["time", "quarter", "cadence"]].copy()
            if "injected" in archive
            else None
        )
    truth = synthetic_truth(source["trial"], physical)
    preserved = (
        "source_id",
        "population",
        "partition",
        "trial_index",
        "depth_bin",
        "period_days",
        "shape_period_days",
        "epoch_bkjd",
        "category",
        "impact_parameter",
        "stellar_density_solar",
        "stellar_density_origin",
        "inferred_mass_solar",
        "inferred_radius_solar",
        "target_integrated_depth_ppm",
        "parent_tpf",
        "base_raw_fits_sha256",
        "phot_g_mean_mag",
        "teff_gspphot",
    )
    if any(truth[key] != source["truth"][key] for key in preserved) or any(
        truth["signal"][key] != source["truth"]["signal"][key]
        for key in ("exposure_days", "supersample_factor")
    ):
        raise ValueError("Raw remodel changed a frozen source or signal draw")
    truth["photometry_view"] = source["truth"]["photometry_view"]
    injected_curve, native_loss, delta = raw_remodel_curve(curve, truth)
    folder = out / "inputs" / source["requested_view"] / source["trial"]["partition"]
    target = folder / f"{case['source_id']}.json"
    if target.exists():
        raise FileExistsError("Preserve prior remodel source receipt")
    saved = {
        key: source[key]
        for key in (
            "trial",
            "provenance",
            "requested_view",
            "view_availability",
            "view_selection_reason",
            "native_input_verification",
            "native_curve_sha256",
        )
    }
    saved.update(
        schema=2,
        truth=truth,
        plan_sha256=digest(out / "plan.json"),
        injected_curve_sha256=curve_digest(injected_curve),
        remodel={
            "recipe_sha256": recipe_sha,
            "prior_receipt": {
                "path": str(prior_path),
                "sha256": case["receipt_sha256"],
            },
            "prior_arrays": {"path": str(prior_array), "sha256": array_entry["sha256"]},
            "original_preparation_replayed_exactly": True,
            "retained_fields": list(preserved),
            "generator_sha256": digest(__file__),
        },
    )
    if original_failure is not None:
        injected_records = None
        saved.update(
            status="INPUT_INELIGIBLE",
            reason=original_failure,
            first_failure_stage="ORIGINAL_PREPARATION",
        )
        preserve_failed_preparation(
            folder / f"{case['source_id']}.raw.npz",
            saved,
            curve,
            injected_curve,
            native_loss,
            delta,
            None,
        )
    else:
        injected_records = remodel_prepared_output(
            saved, folder, curve, injected_curve, original, native_loss, delta
        )
    saved["remodel"]["prepared_mask_comparison"] = remodel_mask_comparison(
        old_mask, injected_records
    )
    if (
        digest(prior_path) != case["receipt_sha256"]
        or checked_input(prior_path) != source
    ):
        raise ValueError("Original inputs changed during raw replay")
    limb_darkening_references(physical["limb_darkening_model"])
    save(target, saved)
    return saved


def remodel_prepared_output(
    saved, folder, curve, injected_curve, original, native_loss, delta
):
    sid = saved["truth"]["source_id"]
    try:
        injected = prepare_curve_records(
            injected_curve, mask_flatten=saved["truth"]["photometry_view"] == "psf_flat"
        )
    except InsufficientCadences as error:
        saved.update(
            status="INPUT_INELIGIBLE",
            reason=str(error),
            first_failure_stage="INJECTED_PREPARATION",
        )
        preserve_failed_preparation(
            folder / f"{sid}.raw.npz",
            saved,
            curve,
            injected_curve,
            native_loss,
            delta,
            original,
        )
    else:
        sampled = remodel_truth_statistics(saved["truth"], original, injected)
        saved.update(
            status="PREPARED",
            reason=None,
            first_failure_stage=None,
            arrays=save_arrays(
                folder / f"{sid}.npz",
                original=original,
                injected=injected,
                raw_delta=delta,
                raw_loss=native_loss,
                prepared_loss=sampled,
                **{"raw_" + key: curve[key] for key in RAW_FIELDS},
            ),
            original_prepared_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
            injected_prepared_sha256=hashlib.sha256(injected.tobytes()).hexdigest(),
        )
        return injected
    return None


def remodel_mask_comparison(prior, new):
    if prior is None or new is None:
        return {"available": False, "reason": "PRIOR_OR_NEW_PREPARATION_INELIGIBLE"}
    before = set(zip(prior["quarter"].tolist(), prior["cadence"].tolist()))
    after = set(zip(new["quarter"].tolist(), new["cadence"].tolist()))
    return {
        "available": True,
        "prior_cadences": len(prior),
        "new_cadences": len(new),
        "prior_only_cadences": len(before - after),
        "new_only_cadences": len(after - before),
        "ordered_time_quarter_cadence_exact": all(
            np.array_equal(prior[key], new[key])
            for key in ("time", "quarter", "cadence")
        ),
        "scope": "Masks may legitimately change after physical raw injection.",
    }


def array_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy") or name[:-4] in headers:
                raise ValueError("Unexpected raw replay archive member")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported raw replay NPY header version")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = reader(stream)
                if dtype.hasobject:
                    raise ValueError("Object arrays are not raw replay inputs")
                headers[name[:-4]] = (shape, dtype)
    return headers


def remodel_storage_bound(cases):
    total = 0
    for case in cases:
        receipt = Path(case["receipt_path"])
        source = checked_input(receipt)
        entry = source.get("arrays") or source["raw_arrays"]
        headers = array_headers(receipt.parent / entry["path"])
        length = headers["raw_time"][0][0]
        raw_bytes = sum(
            math.prod(headers["raw_" + key][0]) * headers["raw_" + key][1].itemsize
            for key in RAW_FIELDS
        )
        if any(headers["raw_" + key][0] != (length,) for key in RAW_FIELDS):
            raise ValueError("Raw replay field header lengths differ")
        original_bytes = (
            math.prod(headers["original"][0]) * headers["original"][1].itemsize
            if "original" in headers
            else 0
        )
        record_size = headers["original"][1].itemsize if original_bytes else 32
        total += (
            raw_bytes
            + original_bytes
            + length * (record_size + 3 * 8)
            + 2 * receipt.stat().st_size
            + 65536
        )
    return {
        "anticipated_bytes": total,
        "method": (
            "Uncompressed seven native fields + original prepared bytes + "
            "new prepared records bounded by every native cadence + raw loss/"
            "delta/prepared loss float64 + twice prior receipt bytes +64KiB "
            "per source for new metadata, archive headers and cohort records. "
            "No compression savings assumed."
        ),
    }


def remodel(args):
    if args.out.exists():
        raise FileExistsError("Raw remodel requires a fresh immutable output namespace")
    cases, physical, prior, bindings = remodel_admission(args)
    growth = remodel_storage_bound(cases)
    admission = storage_check(args.out.parent, growth["anticipated_bytes"])
    admission["growth_bound"] = growth
    args.out.mkdir()
    save(args.out / "plan.json", physical)
    tasks = [(str(args.out), row, physical, args.plan_sha256) for row in cases]
    execute(tasks, remodel_trial, args.workers, args.out / "progress.json")
    receipts = [
        checked_input(
            args.out
            / "inputs"
            / "preferred_full"
            / "fresh_final"
            / f"{row['source_id']}.json"
        )
        for row in cases
    ]
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise ValueError("Raw remodel dependency changed during construction")
    index = prior["preparation_identity"]["native_index"]
    if file_stamp(index["path"]) != index["stamp"]:
        raise ValueError("Native index changed during raw replay")
    runtime = runtime_identity(
        {},
        {
            "environment": {
                name: item["version"]
                for name, item in prior["preparation_identity"]["runtime"][
                    "libraries"
                ].items()
            }
        },
    )
    if runtime != prior["preparation_identity"]["runtime"]:
        raise ValueError("Raw replay runtime changed during construction")
    manifest = {
        "status": "COMPLETE_RAW_REMODEL_CONSTRUCTION_NO_SEARCH",
        "requested_cases": len(cases),
        "prior_manifest_sha256": args.prior_sha256,
        "recipe_sha256": args.plan_sha256,
        "source_sha256": digest(__file__),
        "preparation_counts": dict(Counter(row["status"] for row in receipts)),
        "receipt_sha256": {
            row["source_id"]: digest(
                args.out
                / "inputs/preferred_full/fresh_final"
                / f"{row['source_id']}.json"
            )
            for row in cases
        },
        "original_preparation_replayed_exactly": len(cases),
        "runtime": runtime,
        "storage_admission": admission,
        "before_after_sha256": bindings,
        "scope": (
            "Fixed parent prefix; all failures/zero support retained without "
            "replacement. Original prepared arrays verified exactly. New raw "
            "loss precedes unchanged preparation; no detector search or "
            "prepared-flux rescaling."
        ),
    }
    save(args.out / "manifest.json", manifest)
    for path in args.out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return manifest


def prepare(args):
    rows, plan, freeze = frozen_trials(args.out, args.partition, args.count)
    if digest(Path(__file__)) != freeze["generator_sha256"]:
        raise ValueError("Injection generator differs from the frozen experiment")
    dependencies = {
        "preparation_wrapper_sha256": ROOT / "pipeline/deterministic/run.py",
        "preparation_sha256": ROOT / "pipeline/deterministic/signal_contracts.py",
        "transit_machinery_sha256": ROOT / "pipeline/deterministic/injections.py",
        "reader_sha256": ROOT / "pipeline/photometry/lc2.py",
    }
    if any(digest(path) != freeze[key] for key, path in dependencies.items()):
        raise ValueError("Frozen preparation, transit machinery or reader changed")
    storage_check(args.out, anticipated_bytes=args.count * 8 * 1024**2)
    tasks = [(str(args.out), row, plan, args.view) for row in rows]
    execute(
        tasks,
        prepare_trial,
        args.workers,
        args.out / f"prepare-{args.view}-{args.partition}-progress.json",
    )
    receipts = [
        read(
            args.out
            / "inputs"
            / args.view
            / args.partition
            / f"{row['source_id']}.json"
        )
        for row in rows
    ]
    save(
        args.out / f"prepared-{args.view}-{args.partition}-{args.count}.json",
        {
            "schema": 1,
            "count": len(receipts),
            "statuses": dict(Counter(row["status"] for row in receipts)),
            "receipt_sha256": {
                row["trial"]["source_id"]: digest(
                    args.out
                    / "inputs"
                    / args.view
                    / args.partition
                    / f"{row['trial']['source_id']}.json"
                )
                for row in receipts
            },
        },
    )


def completed_tasks(pool, function, tasks, workers):
    remaining = iter(tasks)
    pending = {pool.submit(function, task) for task in islice(remaining, workers)}
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            results = [future.result() for future in done]
            yield from results
            pending.update(
                pool.submit(function, task) for task in islice(remaining, len(done))
            )
    finally:
        for future in pending:
            future.cancel()


def execute(tasks, function, workers, progress_path):
    start = time.monotonic()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in completed_tasks(pool, function, tasks, workers):
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                progress = {
                    "status": "COMPLETE" if completed == len(tasks) else "RUNNING",
                    "completed": completed,
                    "total": len(tasks),
                    "elapsed_seconds": time.monotonic() - start,
                }
                save(progress_path, progress)
                print(json.dumps(progress), flush=True)


def detector_initializer(
    module_path, config_path, store_periodograms=False, prepared_manifest=None
):
    global \
        DETECTOR, \
        SCAN_CONFIG, \
        DIAGNOSTICS_ENABLED, \
        STORE_PERIODOGRAMS, \
        PREPARED_RECEIPTS
    path = Path(module_path)
    specification = importlib.util.spec_from_file_location("calibration_detector", path)
    DETECTOR = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = DETECTOR
    specification.loader.exec_module(DETECTOR)
    DETECTOR.ROOT = ROOT
    DETECTOR.BLS_SOURCE = path.with_name("bls.c")
    SCAN_CONFIG = read(config_path)
    DIAGNOSTICS_ENABLED = "diagnostics" in inspect.signature(DETECTOR.scan).parameters
    STORE_PERIODOGRAMS = store_periodograms
    PREPARED_RECEIPTS = (
        read(prepared_manifest)["receipt_sha256"] if prepared_manifest else None
    )
    if store_periodograms and not DIAGNOSTICS_ENABLED:
        raise ValueError(
            "Periodogram preservation requires the blind diagnostic detector API"
        )


def runtime_identity(config, freeze):
    packages = list(freeze["environment"])
    settings = [config] + [
        config.get(name, {}) for name in ("short_period_search", "long_period_search")
    ]
    if any(row.get("detrending", {}).get("method") == "biweight" for row in settings):
        packages += ["wotan", "numba", "llvmlite"]
    libraries = {}
    for package in packages:
        distribution = importlib.metadata.distribution(package)
        files = {
            str(item): digest(distribution.locate_file(item))
            for item in distribution.files or []
            if Path(item).suffix in {".py", ".so", ".dylib", ".pyd"}
        }
        if not files:
            raise ValueError(f"Runtime package has no bound numerical code: {package}")
        libraries[package] = {
            "version": distribution.version,
            "numerical_files": len(files),
            "numerical_code_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    differences = {
        name: {"frozen": version, "runtime": libraries[name]["version"]}
        for name, version in freeze["environment"].items()
        if libraries[name]["version"] != version
    }
    if differences:
        print(
            json.dumps(
                {
                    "status": "RUNTIME_DIFFERS_FROM_FROZEN_GENERATOR",
                    "differences": differences,
                }
            ),
            flush=True,
        )
    executable = Path(sys.executable).resolve()
    return {
        "python": {
            "version": sys.version,
            "executable": str(executable),
            "executable_sha256": digest(executable),
            "implementation": sys.implementation.name,
        },
        "libraries": libraries,
        "frozen_generator_version_differences": differences,
        "code_binding": (
            "SHA256 of the sorted wheel-relative mapping of all Python and "
            "compiled numerical module file SHA256 values; bytecode and JIT "
            "caches excluded."
        ),
    }


def clock_match(proposal, truth):
    period = float(proposal["period_days"])
    epoch = float(proposal["t0_bkjd"])
    true_period, true_epoch = truth["period_days"], truth["epoch_bkjd"]
    tolerance = truth["duration_hours"] / 48 + truth["signal"]["exposure_days"]
    start, stop = truth["start_bkjd"], truth["stop_bkjd"]
    drift = abs(period / true_period - 1) * (stop - start)
    wrapped_epoch = abs(
        (epoch - true_epoch + true_period / 2) % true_period - true_period / 2
    )
    centres = np.asarray(
        [
            value
            for _, value in injections.event_centres(
                true_period, true_epoch, start, stop
            )
            if start <= value <= stop
        ]
    )
    timing = abs(centres - epoch - np.rint((centres - epoch) / period) * period)
    worst = float(timing.max()) if len(timing) else None
    exact = bool(
        len(timing)
        and drift <= tolerance
        and wrapped_epoch <= tolerance
        and worst <= tolerance
    )
    harmonics = (0.5, 2.0, 1 / 3, 3.0)
    harmonic = next(
        (
            value
            for value in harmonics
            if abs(period / (true_period * value) - 1) * (stop - start) <= tolerance
        ),
        None,
    )
    return {
        "exact": exact,
        "period_drift_days": drift,
        "wrapped_epoch_error_days": wrapped_epoch,
        "worst_clock_error_days": worst,
        "timing_tolerance_days": tolerance,
        "harmonic_period_ratio": harmonic,
    }


def scan_records(records):
    diagnostics = {"store_periodograms": True} if STORE_PERIODOGRAMS else {}
    try:
        proposals = (
            DETECTOR.scan(records, SCAN_CONFIG, diagnostics=diagnostics)
            if DIAGNOSTICS_ENABLED
            else DETECTOR.scan(records, SCAN_CONFIG)
        )
    except DETECTOR.ShallowIneligible as error:
        return {
            "status": "DETECTOR_INELIGIBLE",
            "reason": error.reason,
            "proposals": [],
            "diagnostics": diagnostics,
        }
    return {
        "status": "SEARCHED",
        "reason": None,
        "proposals": proposals,
        "diagnostics": diagnostics,
    }


def preserve_periodograms(result, target):
    arrays = {}
    for name in ("injected", "null_control"):
        if result.get(name) is None:
            continue
        for band in result[name]["diagnostics"].get("bands", []):
            values = band.pop("periodogram_arrays", {})
            arrays.update(
                {f"{name}_{band['band']}_{key}": array for key, array in values.items()}
            )
    if not arrays:
        return
    path = target.with_suffix(".periodograms.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    result["periodograms"] = {
        "path": path.name,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "arrays": {
            key: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for key, array in arrays.items()
        },
    }


def searched_spectra(row):
    fields = ("period_days", "search_snr", "depth_ppm", "duration_days", "epoch_bkjd")
    return {
        f"{name}_{band['band']}_{field}"
        for name in ("injected", "null_control")
        if row.get(name) is not None
        for band in row[name]["diagnostics"].get("bands", [])
        if band["status"] == "SEARCHED"
        for field in fields
    }


def verify_periodograms(row, folder, required=False):
    entry = row.get("periodograms")
    expected = searched_spectra(row) if required else set()
    if expected and not entry:
        raise ValueError("Searched detector band lacks its mandatory blind spectra")
    if not entry:
        return
    path = folder / entry["path"]
    if digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError("Preserved blind periodogram arrays changed")
    with np.load(path, allow_pickle=False) as arrays:
        if not expected <= set(arrays.files) or set(arrays.files) != set(
            entry["arrays"]
        ):
            raise ValueError("Preserved blind periodogram array membership changed")
        for name in arrays.files:
            values = arrays[name]
            metadata = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
            if metadata != entry["arrays"][name]:
                raise ValueError("Preserved blind periodogram array metadata changed")


def validate_side(side, truth, store_periodograms):
    if side["status"] not in ("SEARCHED", "DETECTOR_INELIGIBLE"):
        raise ValueError("Unknown detector replay status")
    if side["status"] == "DETECTOR_INELIGIBLE":
        if (
            side["proposals"]
            or not isinstance(side["reason"], str)
            or not side["reason"]
        ):
            raise ValueError(
                "Detector-ineligible replay has proposals or lacks a reason"
            )
    elif side["reason"] is not None:
        raise ValueError("Searched detector replay records an ineligibility reason")
    if (
        store_periodograms
        and side["status"] == "SEARCHED"
        and not any(
            band["status"] == "SEARCHED"
            for band in side["diagnostics"].get("bands", [])
        )
    ):
        raise ValueError("Searched replay lacks its declared diagnostic band")
    exact = False
    for proposal in side["proposals"]:
        values = [float(proposal[key]) for key in ("period_days", "t0_bkjd")]
        if values[0] <= 0 or not np.isfinite(values).all():
            raise ValueError("Replay proposal has an invalid measured clock")
        measured = clock_match(proposal, truth)
        if (
            type(proposal["truth_match"]["exact"]) is not bool
            or proposal["truth_match"] != measured
        ):
            raise ValueError(
                "Replay proposal clock match differs from its immutable truth"
            )
        exact |= measured["exact"]
    return exact


def validate_result(row, source, null_count, folder, store_periodograms):
    if row["trial"] != source["trial"] or row["truth"] != source["truth"]:
        raise ValueError("Replay trial or physical truth differs from its input")
    if source["status"] == "INPUT_INELIGIBLE":
        expected = {
            "status": "INPUT_INELIGIBLE",
            "reason": source["reason"],
            "proposals": [],
            "diagnostics": {},
        }
        if (
            row["injected"] != expected
            or row["null_control"] is not None
            or row.get("periodograms")
        ):
            raise ValueError(
                "Input-ineligible replay differs from its preserved failure"
            )
        exact = False
    elif source["status"] == "PREPARED":
        exact = validate_side(row["injected"], source["truth"], store_periodograms)
    else:
        raise ValueError("Unknown immutable input eligibility status")
    if row["recovered_exact"] is not exact:
        raise ValueError(
            "Replay exact-recovery flag differs from its measured proposals"
        )
    requested = (
        source["status"] == "PREPARED"
        and int(row["trial"]["trial_index"]) <= null_count
    )
    if (row["null_control"] is not None) != requested:
        raise ValueError("Replay original-control membership differs from its request")
    if requested:
        original_exact = validate_side(
            row["null_control"], source["truth"], store_periodograms
        )
        if row["preexisting_exact_clock"] is not original_exact:
            raise ValueError(
                "Replay original-clock flag differs from its measured proposals"
            )
    elif (
        row.get("preexisting_exact_clock") is not None
        and row["preexisting_exact_clock"] is not False
    ):
        raise ValueError("Replay original clock flagged without an original scan")
    verify_periodograms(row, folder, store_periodograms)


def run_trial(task):
    run_folder, input_folder, row, null_count = task
    run_folder, input_folder = Path(run_folder), Path(input_folder)
    target = run_folder / f"{row['source_id']}.json"
    input_path = input_folder / f"{row['source_id']}.json"
    source = checked_input(input_path)
    if (
        source["trial"] != row
        or PREPARED_RECEIPTS is None
        or PREPARED_RECEIPTS[row["source_id"]] != digest(input_path)
    ):
        raise ValueError("Trial input differs from its frozen prepared-stage manifest")
    if target.exists():
        saved = read(target)
        if saved["trial"] != row or saved["input_receipt_sha256"] != digest(input_path):
            raise ValueError("Detector replay trial identity changed")
        validate_result(saved, source, null_count, run_folder, STORE_PERIODOGRAMS)
        return saved
    start = time.monotonic()
    result = {
        "schema": 1,
        "trial": row,
        "truth": source["truth"],
        "input_receipt_sha256": digest(input_path),
        "recovered_exact": False,
        "null_control": None,
    }
    if source["status"] != "PREPARED":
        result["injected"] = {
            "status": "INPUT_INELIGIBLE",
            "reason": source["reason"],
            "proposals": [],
            "diagnostics": {},
        }
    else:
        array_path = input_folder / source["arrays"]["path"]
        with np.load(array_path, allow_pickle=False) as archive:
            injected = scan_records(archive["injected"])
            for proposal in injected["proposals"]:
                proposal["truth_match"] = clock_match(proposal, source["truth"])
            result["injected"] = injected
            result["recovered_exact"] = any(
                row["truth_match"]["exact"] for row in injected["proposals"]
            )
            if int(row["trial_index"]) <= null_count:
                original = scan_records(archive["original"])
                for proposal in original["proposals"]:
                    proposal["truth_match"] = clock_match(proposal, source["truth"])
                result["null_control"] = original
                result["preexisting_exact_clock"] = any(
                    row["truth_match"]["exact"] for row in original["proposals"]
                )
    result["elapsed_seconds"] = time.monotonic() - start
    if STORE_PERIODOGRAMS:
        preserve_periodograms(result, target)
    validate_result(result, source, null_count, run_folder, STORE_PERIODOGRAMS)
    save(target, result)
    return result


def run(args):
    rows, _, freeze = frozen_trials(args.out, args.partition, args.count)
    if digest(injections.__file__) != freeze["transit_machinery_sha256"]:
        raise ValueError("Frozen physical clock-matching dependency changed")
    folder = args.out / "runs" / args.label
    binding_path = args.out / "checks/native-input-binding.json"
    binding = read(binding_path)
    if binding["experiment_freeze"]["sha256"] != digest(
        args.out / "freeze.json"
    ) or binding["native_input_ledger"]["sha256"] != digest(
        args.out / "native-input-verification.json"
    ):
        raise ValueError(
            "Complete native-input binding differs from the frozen experiment"
        )
    prepared_path = (
        args.out / f"prepared-{args.view}-{args.partition}-{args.count}.json"
    )
    prepared = read(prepared_path)
    if prepared["count"] != len(rows) or set(prepared["receipt_sha256"]) != {
        row["source_id"] for row in rows
    }:
        raise ValueError(
            "Prepared-stage manifest source identities differ from the frozen trials"
        )
    identity = {
        "partition": args.partition,
        "count": args.count,
        "null_count": args.null_count,
        "view": args.view,
        "config_sha256": digest(args.config),
        "module_sha256": digest(args.module),
        "bls_sha256": digest(args.module.with_name("bls.c")),
        "experiment_freeze_sha256": digest(args.out / "freeze.json"),
        "matching_code_sha256": digest(Path(__file__)),
        "prepared_manifest_sha256": digest(prepared_path),
        "store_periodograms": args.store_periodograms,
        "native_input_binding_sha256": digest(binding_path),
        "nonfinite_diagnostic_encoding": "NaN, +Infinity and -Infinity JSON strings",
        "runtime": runtime_identity(read(args.config), freeze),
    }
    storage_check(
        args.out,
        anticipated_bytes=periodogram_storage_bound(
            read(args.config), args.count, args.null_count, args.store_periodograms
        ),
    )
    if folder.exists():
        if read(folder / "identity.json") != identity:
            raise ValueError(
                "Detector replay label already binds a different experiment"
            )
    else:
        folder.mkdir(parents=True)
        shutil.copyfile(args.module, folder / "run.py")
        shutil.copyfile(args.module.with_name("bls.c"), folder / "bls.c")
        shutil.copyfile(args.config, folder / "config.json")
        shutil.copyfile(Path(__file__), folder / "harness.py")
        shutil.copyfile(prepared_path, folder / "prepared-manifest.json")
        save(folder / "identity.json", identity)
    copied = {
        "module_sha256": folder / "run.py",
        "config_sha256": folder / "config.json",
        "bls_sha256": folder / "bls.c",
        "prepared_manifest_sha256": folder / "prepared-manifest.json",
        "matching_code_sha256": folder / "harness.py",
    }
    if any(digest(path) != identity[key] for key, path in copied.items()):
        raise ValueError(
            "Frozen detector replay source, configuration or kernel changed"
        )
    start, completed = time.monotonic(), 0
    tasks = [
        (
            str(folder / "trials"),
            str(args.out / "inputs" / args.view / args.partition),
            row,
            args.null_count,
        )
        for row in rows
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=detector_initializer,
        initargs=(
            str(folder / "run.py"),
            str(folder / "config.json"),
            args.store_periodograms,
            str(folder / "prepared-manifest.json"),
        ),
    ) as pool:
        for _ in completed_tasks(pool, run_trial, tasks, args.workers):
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(rows):
                progress = {
                    "status": "COMPLETE" if completed == len(rows) else "RUNNING",
                    "label": args.label,
                    "completed": completed,
                    "total": len(rows),
                    "elapsed_seconds": time.monotonic() - start,
                }
                save(folder / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    report(args.out, args.label)


def recovery(rows):
    success = sum(row["recovered_exact"] for row in rows)
    count = len(rows)
    if not count:
        return {"recovered": 0, "trials": 0, "fraction": None, "wilson_95": None}
    proportion, z = success / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
        / denominator
    )
    return {
        "recovered": success,
        "trials": count,
        "fraction": proportion,
        "wilson_95": [center - half, center + half],
    }


def report(out, label):
    folder = out / "runs" / label
    identity = read(folder / "identity.json")
    prepared_manifest = folder / "prepared-manifest.json"
    if digest(prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("Detector report prepared-stage identity changed")
    prepared_receipts = read(prepared_manifest)["receipt_sha256"]
    rows = [read(path) for path in sorted((folder / "trials").glob("*.json"))]
    expected, _, _ = frozen_trials(out, identity["partition"], identity["count"])
    declared = {row["source_id"]: row for row in expected}
    seen = set()
    for row in rows:
        sid = row["trial"]["source_id"]
        if sid in seen or declared.get(sid) != row["trial"]:
            raise ValueError(
                "Replay has duplicate, changed or undeclared trial outcomes"
            )
        seen.add(sid)
        input_path = (
            out / "inputs" / identity["view"] / identity["partition"] / f"{sid}.json"
        )
        source = checked_input(input_path)
        validate_result(
            row,
            source,
            identity["null_count"],
            folder / "trials",
            identity["store_periodograms"],
        )
        if (
            digest(input_path) != row["input_receipt_sha256"]
            or prepared_receipts[sid] != row["input_receipt_sha256"]
            or source["truth"] != row["truth"]
        ):
            raise ValueError(
                "Replay input or physical truth differs from its immutable receipt"
            )
    strata = {}
    for axis in ("population", "depth_bin", "photometry_view"):
        for value in sorted({row["truth"][axis] for row in rows}, key=str):
            strata[f"{axis}:{value}"] = recovery(
                [row for row in rows if row["truth"][axis] == value]
            )
    for low, high in ((0, 3), (3, 8), (8, 12), (12, 20), (20, None)):
        selected = [
            row
            for row in rows
            if row["truth"].get("expected_white_mes") is not None
            and row["truth"]["expected_white_mes"] >= low
            and (high is None or row["truth"]["expected_white_mes"] < high)
        ]
        strata[f"expected_white_mes:{low}-{high}"] = recovery(selected)
    bins = {
        "period_days": [0.5, 3, 10, 30, 100.01],
        "duration_hours": [0, 0.5, 1, 2, 4, 8, 16, 100],
        "phot_g_mean_mag": [0, 16, 17, 18, 19, 25],
        "observed_events": [0, 3, 5, 10, 30, 100, 10000],
    }
    for axis, edges in bins.items():
        for low, high in pairwise(edges):
            selected = [
                row
                for row in rows
                if row["truth"].get(axis) is not None
                and low <= row["truth"][axis] < high
            ]
            strata[f"{axis}:{low}-{high}"] = recovery(selected)
    eligible = [row for row in rows if row["injected"]["status"] == "SEARCHED"]
    nulls = [row for row in rows if row["null_control"] is not None]
    result = {
        "schema": 1,
        "label": label,
        "identity": identity,
        "status": "COMPLETE" if len(rows) == identity["count"] else "INCOMPLETE",
        "completed": len(rows),
        "requested": identity["count"],
        "unmeasured_trials": identity["count"] - len(rows),
        "requested_recovery_lower_bound": sum(row["recovered_exact"] for row in rows)
        / identity["count"],
        "unconditional_exact_recovery": recovery(rows),
        "searched_exact_recovery": recovery(eligible),
        "injected_statuses": dict(Counter(row["injected"]["status"] for row in rows)),
        "ineligibility_reasons": dict(
            Counter(
                row["injected"]["reason"] for row in rows if row["injected"]["reason"]
            )
        ),
        "strata": strata,
        "parent_count": len({row["trial"]["parent_tpf"] for row in rows}),
        "null_original_controls": {
            "trials": len(nulls),
            "with_any_proposal": sum(
                bool(row["null_control"]["proposals"]) for row in nulls
            ),
            "preexisting_exact_injection_clock": sum(
                row.get("preexisting_exact_clock", False) for row in nulls
            ),
            "proposal_count": sum(
                len(row["null_control"]["proposals"]) for row in nulls
            ),
        },
        "source_seconds": sum(row["elapsed_seconds"] for row in rows),
        "preserved_periodogram_trials": sum(
            bool(row.get("periodograms")) for row in rows
        ),
        "preserved_periodogram_bytes": sum(
            row.get("periodograms", {}).get("bytes", 0) for row in rows
        ),
        "claim_boundary": read(out / "plan.json")["claim_boundary"],
        "uncertainty_boundary": (
            "Wilson intervals summarize trial counts; shared parent/spacecraft "
            "noise means they are descriptive, not independence-adjusted "
            "confidence bounds."
        ),
    }
    result["confirmation_target_observed"] = bool(
        identity["partition"] == "confirmation"
        and identity["count"] == 10000
        and len(rows) == 10000
        and result["unconditional_exact_recovery"]["fraction"] >= 0.85
    )
    save(folder / "report.json", result)
    print(
        json.dumps(
            {
                "label": label,
                "status": result["status"],
                "recovery": result["unconditional_exact_recovery"],
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--population", type=Path, required=True)
    frozen.add_argument("--out", type=Path, required=True)
    prepared = commands.add_parser("prepare")
    replay = commands.add_parser("run")
    for command in (prepared, replay):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument(
            "--partition", choices=("development", "confirmation"), required=True
        )
        command.add_argument("--count", type=int, required=True)
        command.add_argument("--workers", type=int, default=10)
        command.add_argument(
            "--view",
            choices=("preferred_full", "psf_flat", "psf_qdiag", "sap_qdiag"),
            default="preferred_full",
        )
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument(
        "--module", type=Path, default=ROOT / "pipeline/shallow/run.py"
    )
    replay.add_argument("--label", required=True)
    replay.add_argument("--null-count", type=int, default=0)
    replay.add_argument("--store-periodograms", action="store_true")
    summary = commands.add_parser("report")
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--label", required=True)
    scrambled = commands.add_parser("year-scramble")
    scrambled.add_argument("--out", type=Path, required=True)
    scrambled.add_argument("--partition", choices=("development",), required=True)
    scrambled.add_argument("--count", type=int, required=True)
    scrambled.add_argument(
        "--view",
        choices=("preferred_full", "psf_flat", "psf_qdiag", "sap_qdiag"),
        default="preferred_full",
    )
    scrambled.add_argument("--label", required=True)
    remodeled = commands.add_parser("remodel")
    remodeled.add_argument("--prior-manifest", type=Path, required=True)
    remodeled.add_argument("--prior-sha256", required=True)
    remodeled.add_argument("--plan", type=Path, required=True)
    remodeled.add_argument("--plan-sha256", required=True)
    remodeled.add_argument("--out", type=Path, required=True)
    remodeled.add_argument("--count", type=int, required=True)
    remodeled.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1 or getattr(args, "null_count", 0) < 0:
        raise ValueError("Positive worker count and nonnegative null count required")
    if args.command == "report":
        report(args.out, args.label)
    else:
        {
            "freeze": freeze,
            "prepare": prepare,
            "run": run,
            "year-scramble": year_scramble_inputs,
            "remodel": remodel,
        }[args.command](args)


if __name__ == "__main__":
    main()
