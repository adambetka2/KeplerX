"""Prepare parent-disjoint detector evolution inputs; clocks stay with the scorer."""


import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(
    os.environ.get("SPACELAB_CALIBRATION_ROOT", Path(__file__).resolve().parents[2])
)
os.environ["SPACELAB_CALIBRATION_ROOT"] = str(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import calibrate
import numpy as np

from pipeline.deterministic.cloud_public import fits_arrays
from pipeline.deterministic.run import (
    InsufficientCadences,
    prepare_curve_records,
    read_fsc,
)
from pipeline.photometry.lc2 import load_source, photometry

STUDY = ROOT / "dump/shallow-calibration-20261010"
SEED = "shallow-evolution-controls-20261010-v1"
DEPTH_EDGES = (40, 60, 80, 100, 150, 200, 350, 500, 750, 1000)
INPUTS = {}
PREPARATION = {}
NATIVE_PRODUCTS = {}
TARGET_KICS = {}
FINAL_SEED = "shallow-evolution-fresh-final-20261010-v1"
COMPONENT_NAMESPACE = "shallow-evolution-joint-native-target-kic-v1"
FINAL_CONTROLS = {
    "catalogue_fp": 200,
    "native_eb": 250,
    "instrumental_artifact": 600,
    "original": 650,
    "inverted": 650,
    "year_scrambled": 650,
}
FIELDS = (
    "case_id",
    "kind",
    "positive",
    "source_id",
    "parent_tpf",
    "input_path",
    "input_sha256",
    "array_key",
    "preparation_status",
    "first_failure",
    "receipt_path",
    "receipt_sha256",
    "population",
    "depth_ppm",
    "phot_g_mean_mag",
    "input_was_previously_detector_opened",
    "selection_conditioning",
    "photometry_input_class",
    "component_id",
    "native_target_kic",
    "native_target_kic_status",
    "sampled_depth_ppm",
    "observed_events",
    "raw_nonzero_loss_cadences",
    "prepared_nonzero_loss_cadences",
    "first_injection_observability_failure",
    "observability_path",
    "observability_sha256",
    "requested_instrument_kind",
    "instrument_observability_status",
    "instrument_observability_path",
    "instrument_observability_sha256",
)
CLOCK_FIELDS = (
    "case_id",
    "signal_id",
    "period_days",
    "epoch_bkjd",
    "duration_hours",
    "start_bkjd",
    "stop_bkjd",
    "exposure_days",
    "depth_ppm",
    "population",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(path, expected=None):
    path = Path(path)
    sha = calibrate.digest(path)
    require(
        expected is None or sha == expected, "Immutable input differs: " + str(path)
    )
    INPUTS[str(path)] = {"sha256": sha, "stamp": calibrate.file_stamp(path)}
    return path


def read(path, expected=None):
    return json.loads(bind(path, expected).read_text())


def rows(path, expected=None):
    with bind(path, expected).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    path = Path(path)
    require(not path.exists(), "Refusing to replace evidence: " + str(path))
    calibrate.save(path, value)
    path.chmod(0o444)


def write_csv(path, values, fields):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in values)
    Path(path).chmod(0o444)


def rank(identity):
    return hashlib.sha256(f"{SEED}:{identity}".encode()).hexdigest()


def identities(row):
    sources = [row["source_id"], *row.get("extra_source_ids", [])]
    values = {"s:" + str(value) for value in sources if value not in (None, "", 0, "0")}
    parents = [
        row.get("parent_tpf", row.get("origin_tpf_kic")),
        row.get("kic_id"),
        row.get("native_target_kic"),
        *row.get("extra_parent_tpfs", []),
        *(TARGET_KICS.get(str(sid)) for sid in sources),
    ]
    values.update(
        "p:" + str(value) for value in parents if value not in (None, "", 0, "0")
    )
    raw = row.get("indexed_fits_sha256", row.get("raw_fits_sha256"))
    if raw:
        values.add("r:" + raw)
    values.update(
        "r:" + value for value in row.get("extra_raw_fits_sha256", []) if value
    )
    return values


def native_host_metadata(row):
    sid = str(row["source_id"])
    value = TARGET_KICS.get(sid)
    return {
        **row,
        "native_target_kic": value if value else None,
        "native_target_kic_status": "INDEX_CATALOGUE_TARGET_KIC"
        if value
        else "UNKNOWN_INDEX_KIC_ZERO"
        if sid in TARGET_KICS
        else "NOT_IN_NATIVE_LC2_INDEX",
    }


def protected_components():
    path = STUDY / "checks/prepare-real-kepler-cohort.py"
    bind(path, "64cfceebcab29cf649f7d9b769c0d5b6a124e92dfa7d32f9ab132a16e345a70a")
    spec = importlib.util.spec_from_file_location("evolution_protected_metadata", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    value = module.protected_identities()
    for p, receipt in module.INPUTS.items():
        bind(p, receipt["sha256"])
    return {
        f"{prefix}:{key}"
        for name, prefix in (("source", "s"), ("parent", "p"), ("raw", "r"))
        for key in value[name]
        if key not in (None, "", 0, "0")
    } | {
        "p:" + str(TARGET_KICS[sid]) for sid in value["source"] if TARGET_KICS.get(sid)
    }


def case(stage, kind, row, positive=False):
    key = row.get("signal_id", row["source_id"])
    return {
        "case_id": f"{stage}:{kind}:{key}",
        "kind": kind,
        "positive": "true" if positive else "false",
        "source_id": row["source_id"],
        "parent_tpf": row["parent_tpf"],
        "population": row.get(
            "population", "KEPLER_REAL_HOST" if positive else "UNKNOWN"
        ),
        "depth_ppm": row.get("depth_ppm", row.get("catalogue_depth_ppm", "")),
        "input_was_previously_detector_opened": True,
        "selection_conditioning": "INPUT_ONLY_HASH_RANK_AND_DECLARED_DEPTH_STRATA"
        if kind == "synthetic_planet"
        else "FROZEN_CATALOGUE_OR_INPUT_RANK",
        "origin": row,
        "photometry_input_class": row.get(
            "photometry_input_class", "PREFERRED_FULL_NATIVE"
        ),
    }


def preparation_identity():
    freeze = read(
        STUDY / "freeze.json",
        "5881d86e808d815fd4ad42957c59d682985422cc86e8394e48ea2e3f64c9f39f",
    )
    ledger = read(
        STUDY / "native-input-verification.json",
        "d507b0719b07e504acc3eab5771e6128a166274d8a565c6d622f783e1be97405",
    )
    certificate = read(
        STUDY / "checks/native-input-binding.json",
        "ab8152a0169b085ba3bc04d662f4a39907ef9411032d1e87086a897471a491f7",
    )
    require(
        ledger["status"] == "COMPLETE" and certificate["index"] == ledger["index"],
        "Native ledger incomplete or certificate differs",
    )
    index = freeze["native_index"]
    require(
        calibrate.file_stamp(index["path"]) == index["stamp"],
        "Frozen native index stamp changed",
    )
    bind(index["path"], index["sha256"])
    with sqlite3.connect(f"file:{index['path']}?mode=ro", uri=True) as database:
        TARGET_KICS.update(database.execute("SELECT gaia_source_id,kic FROM sources"))
    source_root = Path(__file__).resolve().parents[2]
    dependencies = {
        str(path): calibrate.digest(bind(path))
        for path in [Path(__file__)]
        + [
            source_root / "pipeline" / name
            for name in (
                "deterministic/cloud_public.py",
                "deterministic/injections.py",
                "deterministic/run.py",
                "deterministic/signal_contracts.py",
                "photometry/lc2.py",
                "shallow/calibrate.py",
            )
        ]
    }
    runtime = calibrate.runtime_identity(
        {}, {"environment": {**freeze["environment"], "requests": "2.32.5"}}
    )
    differences = {
        name: row["version"]
        for name, row in runtime["libraries"].items()
        if row["version"] != {**freeze["environment"], "requests": "2.32.5"}[name]
    }
    require(
        not differences,
        "Preparation numerical environment differs: " + str(differences),
    )
    return {
        "dependencies": dependencies,
        "runtime": runtime,
        "native_index": index,
        "native_ledger_sha256": calibrate.digest(
            STUDY / "native-input-verification.json"
        ),
    }


def assign_components(stages, connections=()):
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return parents[key]

    values = list(connections) + [
        entry["origin"] for entries in stages.values() for entry in entries
    ]
    for row in values:
        keys = sorted(identities(row))
        roots = [find(key) for key in keys]
        first = min(roots)
        for key in roots:
            parents[key] = first
    for entries in stages.values():
        for entry in entries:
            entry.update(
                {
                    key: entry["origin"][key]
                    for key in ("native_target_kic", "native_target_kic_status")
                }
            )
            root = find(next(iter(identities(entry["origin"]))))
            entry["component_id"] = hashlib.sha256(
                f"{COMPONENT_NAMESPACE}:{root}".encode()
            ).hexdigest()


def plan(out):
    legacy = read(
        out / "cohorts/plan.json",
        "f046d244d7a279c0418456c48ba71e2dae7543a05d1acc58e67376ebb174ee05",
    )
    bind(legacy["source_snapshot"]["path"], legacy["source_snapshot"]["sha256"])
    value = read(
        out / "cohorts/plan-v2.json",
        "db29a4699bb404cabe65fe7b36d8c37068b15d32381c6f681b85bbbb65520a6a",
    )
    bind(value["source_snapshot"]["path"], value["source_snapshot"]["sha256"])
    return value


def native_real(row):
    path = Path(row["native_fits_path"])
    require(
        calibrate.file_stamp(path) == json.loads(row["current_stamp"]),
        "Native real FITS identity changed",
    )
    bind(path, row["indexed_fits_sha256"])
    raw = fits_arrays(path, row["source_id"], {"product_kic": int(row["kic_id"])})
    require(
        calibrate.file_stamp(path) == INPUTS[str(path)]["stamp"],
        "Native FITS changed during parsing",
    )
    return raw


def selected_curve(raw):
    views = []
    for view in ("psf_qdiag", "sap_qdiag"):
        curve = photometry(raw, view)
        good = (
            np.isfinite(curve["time"])
            & np.isfinite(curve["flux"])
            & np.isfinite(curve["error"])
            & (curve["error"] > 0)
            & (curve["quality"] == 0)
        )
        views.append((view, curve, int(good.sum())))
    return views[0] if views[0][2] >= 100 else views[1]


def save_input(
    out, key, raw, origin, selected=None, extra_arrays=None, model=None, native=None
):
    path = out / "inputs" / f"{key}.json"
    if path.exists():
        value = read(path)
        require(value["origin"] == origin, "Existing prepared input origin differs")
        if value.get("arrays"):
            bind(path.with_suffix(".npz"), value["arrays"]["sha256"])
        return value, path
    view, curve, usable = selected if selected is not None else selected_curve(raw)
    receipt = {
        "schema": 1,
        "origin": origin,
        "photometry_view": view,
        "usable_quality_cadences_before_preparation": usable,
        "preparation": (
            "Existing fixed quality0/finite-positive-error/positive- "
            "quarter-median/upper12MAD/time-cadence-sort-and- "
            "dedup/FSCfloat32; mask_flatten=False."
        ),
        "source_sha256": calibrate.digest(__file__),
        "preparation_identity": PREPARATION,
        "instrument_model": model,
        "native_input_verification": native,
        "native_or_array_identity": {
            k: v
            for k, v in origin.items()
            if k
            in (
                "native_fits_path",
                "indexed_fits_sha256",
                "current_stamp",
                "raw_path",
                "raw_sha256",
                "input_manifest",
            )
        },
    }
    records = None
    try:
        records = prepare_curve_records(curve)
        receipt.update(preparation_status="PREPARED", first_failure="")
    except InsufficientCadences as error:
        receipt.update(preparation_status="INPUT_INELIGIBLE", first_failure=str(error))
    arrays = {"raw_" + name: curve[name] for name in calibrate.RAW_FIELDS}
    arrays.update(extra_arrays or {})
    if records is not None:
        arrays["records"] = records
    target = path.with_suffix(".npz")
    require(not target.exists(), "Unbound existing prepared NPZ: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    target.chmod(0o444)
    receipt["arrays"] = {
        "path": str(target),
        "sha256": calibrate.digest(target),
        "field_hashes": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "bytes_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for k, v in arrays.items()
        },
    }
    save(path, receipt)
    return receipt, path


def instrument_parameters(origin):
    sid = origin["source_id"]

    def draw(name, low, high):
        fraction = (int(rank(sid + ":instrument:" + name)[:16], 16) + 0.5) / 2**64
        return low + (high - low) * fraction

    return {
        "kind": origin["artifact_kind"],
        "seed": SEED,
        "amplitude_fraction": draw("amplitude", 0.001, 0.006),
        "recovery_days": draw("recovery", 0.25, 3.0),
        "residual_fraction": 0.2,
        "period_days": draw("period", 2.5, 16.0),
        "pulse_duration_days": draw("duration", 0.08, 0.25),
        "quarter_phase_fraction": draw("quarter_phase", 0.1, 0.8),
        "scope": (
            "Predeclared surrogate mechanism parameters; not fitted observed "
            "systematics or empirical occurrence rates."
        ),
    }


def instrument_shape(times, model):
    t = times - times[0]
    kind = model["kind"]
    if kind == "thermal_recovery":
        starts = times[np.r_[True, np.diff(times) > 1.0]]
        age = times - starts[np.searchsorted(starts, times, side="right") - 1]
        return np.exp(-age / model["recovery_days"])
    if kind == "sensitivity_dropout":
        age = t - model["quarter_phase_fraction"] * t[-1]
        return np.where(
            age >= 0,
            model["residual_fraction"]
            + (1 - model["residual_fraction"])
            * np.exp(-np.maximum(age, 0) / model["recovery_days"]),
            0,
        )
    require(kind == "column_pulse", "Unknown instrumental mechanism: " + kind)
    phase = (
        t
        - model["quarter_phase_fraction"] * model["period_days"]
        + model["period_days"] / 2
    ) % model["period_days"] - model["period_days"] / 2
    return np.clip(
        (model["pulse_duration_days"] / 2 - np.abs(phase))
        / (model["pulse_duration_days"] / 5),
        0,
        1,
    )


def inject_instrument(curve, origin):
    model = instrument_parameters(origin)
    modified = {k: np.asarray(curve[k]).copy() for k in calibrate.RAW_FIELDS}
    loss, delta = np.zeros(len(curve["time"])), np.zeros(len(curve["time"]))
    quarters = []
    for quarter in np.unique(curve["quarter"]):
        use = (
            (curve["quarter"] == quarter)
            & np.isfinite(curve["time"])
            & np.isfinite(curve["flux"])
        )
        if not use.any():
            continue
        median = float(np.median(curve["flux"][use]))
        valid = np.isfinite(median) and median > 0
        if valid:
            shape = instrument_shape(curve["time"][use], model)
            loss[use] = model["amplitude_fraction"] * shape
            delta[use] = -median * loss[use]
            modified["flux"][use] += delta[use]
        quarters.append(
            {
                "quarter": int(quarter),
                "raw_baseline": median,
                "instrument_injected": bool(valid),
            }
        )
    model.update(
        quarters=quarters,
        additive_decrement=(
            "modified_flux=original_flux-quarter_median*instrument_loss; "
            "only flux changed before fixed preparation"
        ),
    )
    return (
        modified,
        {
            "original_raw_flux": curve["flux"],
            "instrument_loss": loss,
            "instrument_flux_delta": delta,
        },
        model,
    )


def checked_fresh_native(origin):
    proof = calibrate.verified_native_inputs(STUDY, origin)
    source, raw = load_source(ROOT, origin["source_id"])
    require(
        str(source["origin_tpf_kic"]) == origin["parent_tpf"]
        and source["raw_fits_sha256"] == origin["raw_fits_sha256"],
        "Fresh native source identity differs from frozen population",
    )
    require(
        proof == calibrate.verified_native_inputs(STUDY, origin),
        "Fresh native products changed during loading",
    )
    for path, entry in proof["products"].items():
        NATIVE_PRODUCTS[path] = entry
        bind(entry["receipt"]["path"], entry["receipt"]["sha256"])
    return raw, proof


def prepare_fresh_instrument(out, entry):
    origin, key = entry["origin"], "fresh-instrument-" + entry["source_id"]
    if (out / "inputs" / f"{key}.json").exists():
        value, path = save_input(out, key, None, origin)
        for product, proof in value["native_input_verification"]["products"].items():
            require(
                calibrate.file_stamp(product) == proof["stamp"],
                "Reused fresh native input stamp differs",
            )
            NATIVE_PRODUCTS[product] = proof
        return value, path
    raw, proof = checked_fresh_native(origin)
    view, curve, usable = selected_curve(raw)
    modified, arrays, model = inject_instrument(curve, origin)
    try:
        arrays["original"] = prepare_curve_records(curve)
        model["original_preparation_status"] = "PREPARED"
        model["original_first_failure"] = ""
    except InsufficientCadences as error:
        model["original_preparation_status"] = "INPUT_INELIGIBLE"
        model["original_first_failure"] = str(error)
    value, path = save_input(
        out, key, None, origin, (view, modified, usable), arrays, model, proof
    )
    return value, path


def instrument_support(archive):
    loss, delta = archive["instrument_loss"], archive["instrument_flux_delta"]
    require(np.isfinite(loss).all() and np.all(loss >= 0), "Invalid artifact loss")
    require(
        np.array_equal(
            archive["raw_flux"], archive["original_raw_flux"] + delta, equal_nan=True
        ),
        "Instrument raw flux arithmetic differs",
    )
    for quarter in np.unique(archive["raw_quarter"]):
        use = (
            (archive["raw_quarter"] == quarter)
            & np.isfinite(archive["raw_time"])
            & np.isfinite(archive["original_raw_flux"])
        )
        if use.any():
            baseline = float(np.median(archive["original_raw_flux"][use]))
            expected = -baseline * loss[use] if baseline > 0 else np.zeros(use.sum())
            require(np.array_equal(delta[use], expected), "Artifact decrement differs")
    records = archive["records"] if "records" in archive else None
    original = archive["original"] if "original" in archive else None
    prepared_loss = None
    if records is not None:
        lookup = {
            (time, int(cadence), int(quarter)): value
            for time, cadence, quarter, value in zip(
                archive["raw_time"],
                archive["raw_cadence"],
                archive["raw_quarter"],
                loss,
                strict=True,
            )
        }
        prepared_loss = np.asarray(
            [
                lookup[(row["time"], int(row["cadence"]), int(row["quarter"]))]
                for row in records
            ]
        )
    equal = (
        original.dtype == records.dtype
        and original.shape == records.shape
        and original.tobytes() == records.tobytes()
        if original is not None and records is not None
        else None
    )
    require(
        np.any(loss > 0) or equal is not False,
        "Zero raw artifact changed prepared records",
    )
    return {
        "raw_rows": len(loss),
        "raw_support_cadences": int(np.count_nonzero(loss > 0)),
        "raw_maximum_loss": float(np.max(loss, initial=0)),
        "prepared_rows": len(records) if records is not None else None,
        "prepared_support_cadences": int(np.count_nonzero(prepared_loss > 0))
        if prepared_loss is not None
        else None,
        "prepared_maximum_nominal_loss": float(np.max(prepared_loss, initial=0))
        if prepared_loss is not None
        else None,
        "prepared_original_byte_equal": equal,
    }


def instrument_observability(out, value, receipt_path):
    arrays = bind(value["arrays"]["path"], value["arrays"]["sha256"])
    identity = {
        "builder_sha256": calibrate.digest(__file__),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": calibrate.digest(bind(receipt_path)),
        "arrays_sha256": value["arrays"]["sha256"],
    }
    path = out / "inputs/instrument-observability-v6" / receipt_path.name
    if path.exists():
        result = read(path)
        require(
            result["identity"] == identity, "Artifact observability binding differs"
        )
        return result, path
    with np.load(arrays, allow_pickle=False) as archive:
        for key, expected in value["arrays"]["field_hashes"].items():
            require(
                str(archive[key].dtype) == expected["dtype"]
                and list(archive[key].shape) == expected["shape"]
                and hashlib.sha256(archive[key].tobytes()).hexdigest()
                == expected["bytes_sha256"],
                "Artifact array field identity differs",
            )
        support = instrument_support(archive)
    status = (
        "ZERO_SAMPLED_ARTIFACT"
        if support["raw_support_cadences"] == 0
        else "INPUT_PREPARATION_FAILED"
        if value["preparation_status"] != "PREPARED"
        else "NO_PREPARED_ARTIFACT"
        if support["prepared_support_cadences"] == 0
        else "REALIZED_PREPARED_ARTIFACT"
    )
    result = {
        "identity": identity,
        "requested_mechanism": value["instrument_model"]["kind"],
        "status": status,
        "support": support,
        "preparation_status": value["preparation_status"],
        "first_preparation_failure": value["first_failure"],
        "scope": (
            "Nominal imposed raw artifact support on retained cadences, before "
            "detector whitening; not a fitted artifact depth or certified FP. "
            "Zero/removed-support draws retain ordinary-input proposal burden."
        ),
    }
    save(path, result)
    return result, path


def prepare_archived_fp(out, entry):
    origin = entry["origin"]
    target = out / "inputs" / f"catalogue_fp-{entry['source_id']}.json"
    if target.exists():
        value = read(target)
        require(value["origin"] == origin, "Archived FP origin differs")
        bind(value["arrays"]["path"], value["arrays"]["sha256"])
        return value, target
    path = bind(origin["fsc_path"])
    records = read_fsc(path, entry["source_id"], "psf_flat").copy()
    require(
        len(records) >= 100
        and np.all(np.isfinite(records["time"]))
        and np.all(np.isfinite(records["flux"]))
        and np.all(np.isfinite(records["error"]) & (records["error"] > 0)),
        "Invalid archived FP FSC numerical payload",
    )
    raw = native_real(origin)
    reproduced = prepare_curve_records(photometry(raw, "psf_flat"), mask_flatten=True)
    require(
        records.dtype == reproduced.dtype and records.tobytes() == reproduced.tobytes(),
        "Archived FP FSC differs from independently prepared native PSF_FLAT",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.with_suffix(".npz").exists(), "Unbound existing archived FP NPZ")
    np.savez_compressed(target.with_suffix(".npz"), records=records)
    target.with_suffix(".npz").chmod(0o444)
    value = {
        "origin": origin,
        "preparation_status": "PREPARED",
        "first_failure": "",
        "photometry_view": "psf_flat",
        "photometry_input_class": "ARCHIVED_PSF_FLAT",
        "native_fits_to_archived_fsc_records_byte_equal": True,
        "archived_fsc_sha256": INPUTS[str(path)]["sha256"],
        "preparation_identity": PREPARATION,
        "arrays": {
            "path": str(target.with_suffix(".npz")),
            "sha256": calibrate.digest(target.with_suffix(".npz")),
            "records_bytes_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
        },
    }
    save(target, value)
    return value, target


def prepare_catalogue(out, entry):
    origin = entry["origin"]
    if origin.get("fresh_instrument"):
        return prepare_fresh_instrument(out, entry)
    if entry["kind"] == "catalogue_fp":
        return prepare_archived_fp(out, entry)
    if entry["kind"] == "real_planet":
        key = "real-" + entry["source_id"]
        host = {
            k: origin[k]
            for k in (
                "source_id",
                "parent_tpf",
                "kic_id",
                "native_fits_path",
                "indexed_fits_sha256",
                "indexed_bytes",
                "current_stamp",
                "product_family",
            )
        }
        if (out / "inputs" / f"{key}.json").exists():
            return save_input(out, key, None, host)
        raw = native_real(origin)
        origin = host
    else:
        path = bind(origin["raw_path"], origin["raw_sha256"])
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
        key = entry["kind"] + "-" + entry["source_id"]
    return save_input(out, key, raw, origin)


def prepare_study(out, entry):
    origin = entry["origin"]
    path = bind(origin["receipt_path"], origin["receipt_sha256"])
    source = calibrate.checked_input(path)
    value = {
        "origin": origin,
        "preparation_status": source["status"],
        "first_failure": source["reason"],
        "photometry_view": source["truth"]["photometry_view"],
        "preparation_identity": PREPARATION,
    }
    if not source.get("arrays"):
        return value, path
    array_path = path.parent / source["arrays"]["path"]
    bind(array_path, source["arrays"]["sha256"])
    kind = entry["kind"]
    if kind in ("synthetic_planet", "original"):
        value["arrays"] = {
            "path": str(array_path),
            "sha256": source["arrays"]["sha256"],
        }
        return value, path
    with np.load(array_path, allow_pickle=False) as archive:
        original = archive["original"].copy()
    target = out / "inputs" / f"{kind}-{entry['source_id']}.json"
    if target.exists():
        value = read(target)
        require(value["origin"] == origin, "Transformed input origin differs")
        if value.get("arrays"):
            bind(target.with_suffix(".npz"), value["arrays"]["sha256"])
        return value, target
    transformed = original.copy()
    arrays = {}
    if kind == "inverted":
        transformed["flux"] = -transformed["flux"]
        value["transform"] = (
            "Reflection of prepared relative flux around zero; all nonflux "
            "fields identical."
        )
    else:
        try:
            transformed, arrays, metadata = calibrate.year_scramble(original)
            value["transform"] = metadata
        except calibrate.YearScrambleIneligible as error:
            value.update(
                preparation_status="SCRAMBLE_INELIGIBLE", first_failure=error.reason
            )
            save(target, value)
            return value, target
    arrays["records"] = transformed
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.with_suffix(".npz").exists(), "Unbound existing transformed NPZ")
    np.savez_compressed(target.with_suffix(".npz"), **arrays)
    target.with_suffix(".npz").chmod(0o444)
    value["arrays"] = {
        "path": str(target.with_suffix(".npz")),
        "sha256": calibrate.digest(target.with_suffix(".npz")),
    }
    save(target, value)
    return value, target


def finish_case(entry, value, receipt_path):
    entry = {k: v for k, v in entry.items() if k != "origin"}
    entry.update(
        input_path=value.get("arrays", {}).get("path", ""),
        input_sha256=value.get("arrays", {}).get("sha256", ""),
        array_key="injected"
        if entry["kind"] == "synthetic_planet"
        else "original"
        if entry["kind"] == "original"
        else "records",
        preparation_status=value["preparation_status"],
        first_failure=value["first_failure"] or "",
        receipt_path=str(receipt_path),
        receipt_sha256=calibrate.digest(receipt_path),
    )
    if entry["preparation_status"] != "PREPARED":
        entry.update(input_path="", input_sha256="", array_key="")
    return entry


def clock(entry, prepared, receipt):
    origin = entry["origin"]
    truth = origin.get("truth", origin)
    start, stop = "", ""
    if prepared["preparation_status"] == "PREPARED":
        with np.load(prepared["input_path"], allow_pickle=False) as archive:
            record = archive[prepared["array_key"]]
            start, stop = float(record["time"][0]), float(record["time"][-1])
    return {
        "case_id": entry["case_id"],
        "signal_id": origin.get("signal_id", entry["source_id"]),
        "period_days": truth["period_days"],
        "epoch_bkjd": truth["epoch_bkjd"],
        "duration_hours": truth["duration_hours"],
        "start_bkjd": start,
        "stop_bkjd": stop,
        "exposure_days": truth.get("signal", {}).get(
            "exposure_days", 0.02043361111111111
        ),
        "depth_ppm": entry["depth_ppm"],
        "population": entry["population"],
    }


def contamination(entries):
    koi_path = ROOT / "discPCIe/data/reference-catalogues/q1_q17_dr25_koi-rows.json"
    eb_path = (
        ROOT / "discPCIe/data/reference-catalogues/kepler-eb-2016-0694ada5b64a.csv"
    )
    koi, eb = defaultdict(list), defaultdict(list)
    for row in read(koi_path):
        koi[str(row["kepid"])].append(
            {
                key: row.get(key)
                for key in (
                    "kepoi_name",
                    "koi_disposition",
                    "koi_period",
                    "koi_time0bk",
                    "koi_duration",
                    "koi_depth",
                )
            }
        )
    for row in rows(eb_path):
        eb[str(int(row["KIC"]))].append(
            {key: row.get(key) for key in ("KIC", "Per", "BJD0", "Morph")}
        )
    result = []
    for entry in entries:
        origin = entry["origin"]
        keys = {
            str(value)
            for value in (
                entry["parent_tpf"],
                origin.get("kic_id"),
                origin.get("native_target_kic"),
            )
            if value not in (None, "", 0, "0")
        }
        result.append(
            {
                "case_id": entry["case_id"],
                "catalogue_identity_keys": sorted(keys),
                "known_koi_by_identity": {key: koi[key] for key in keys if koi[key]},
                "known_eb_by_identity": {key: eb[key] for key in keys if eb[key]},
                "control_truth": origin.get("control_truth"),
                "scope": (
                    "Target/extraction-parent catalogue matches, not an aperture- "
                    "complete contamination analysis; unknown originals/transforms "
                    "measure proposal burden, not certified astrophysical "
                    "negatives."
                ),
            }
        )
    return result


def expand_forbidden(population, forbidden):
    forbidden = set(forbidden)
    while True:
        before = len(forbidden)
        for row in population:
            keys = identities(row)
            if keys & forbidden:
                forbidden.update(keys)
        if len(forbidden) == before:
            return forbidden


def population_description(population):
    result = {}
    for kind in ("ALL", "M", "K"):
        subset = [r for r in population if kind == "ALL" or r["population"] == kind]
        result[kind] = {
            "sources": len(subset),
            "parents": len({r["parent_tpf"] for r in subset}),
            "stored_quarter_counts": dict(Counter(r["quarter_count"] for r in subset)),
            "quantile_probabilities": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
            "quantiles": {
                field: np.quantile(
                    [float(r[field]) for r in subset], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                ).tolist()
                for field in ("phot_g_mean_mag", "teff_gspphot")
            },
        }
    return result


def final_physical_plan():
    previous = read(
        STUDY / "plan.json",
        "bda6659188b13b5d19636bc18d9b2cee3b69e885fc05be6e51c4b712dea86213",
    )
    return {
        **{
            key: previous[key]
            for key in (
                "period_days",
                "impact_parameter",
                "limb_darkening_u1",
                "limb_darkening_u2",
                "exposure_days",
                "supersample_factor",
                "geometry",
                "injection",
                "claim_boundary",
                "matching",
            )
        },
        "schema": 1,
        "protocol": "detector-evolution-fresh-final-inputs-v1",
        "seed": FINAL_SEED,
        "view": "preferred_full",
        "depth_bins_ppm": list(zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:])),
    }


def identity_connections(population):
    historical = []
    for path in (
        STUDY / "trials.csv",
        ROOT / "dump/dwarfs-6000-20261008/verification/sources.csv",
        ROOT / "dump/v4-survey-50000-20260921/cohort.csv",
    ):
        historical.extend(native_host_metadata(r) for r in rows(path))
    catalogue = []
    for folder in (
        "known-object-benchmark-20261003/eb-inputs",
        "release-catalogue-controls-v2",
        "release-catalogue-evaluation-v2",
        "deadline-controls-20260925/catalogue-eb-independent-v3-inputs",
    ):
        manifest = read(ROOT / "dump" / folder / "input-manifest.json")
        for sid, item in manifest["sources"].items():
            metadata = item["metadata"]
            catalogue.append(
                native_host_metadata(
                    {
                        "source_id": sid,
                        "parent_tpf": str(metadata["origin_tpf_kic"]),
                        "kic_id": str(metadata["kic"]),
                        "raw_fits_sha256": metadata["raw_fits_sha256"],
                        "extra_parent_tpfs": [
                            str(v["base_parent_kic"])
                            for v in item.get("pixels", {}).values()
                        ],
                    }
                )
            )
    return population + historical + catalogue, historical


def corrected_stages(previous, old_design, connections, mapping):
    original = {**previous["stages"], **old_design["stages"]}
    stages = json.loads(json.dumps(original))
    for entries in stages.values():
        for entry in entries:
            entry["origin"] = native_host_metadata(entry["origin"])
    assign_components(stages, connections)
    differences = {}
    for stage, entries in stages.items():
        differences[stage] = []
        proofs = mapping["corrected_components"].get(stage, {})
        overlap_proofs = {
            row["case_id"]: row
            for overlap in mapping["cross_stage_overlaps"]
            for row in overlap["cases"].get(stage, [])
        }
        require(
            len({r["component_id"] for r in entries})
            == mapping["stages"][stage]["components"],
            "Independent stage component count differs",
        )
        for before, after in zip(original[stage], entries, strict=True):
            proof = proofs.get(after["case_id"], overlap_proofs.get(after["case_id"]))
            if proof is not None:
                expected = hashlib.sha256(
                    f"{COMPONENT_NAMESPACE}:{proof['component']}".encode()
                ).hexdigest()
                require(
                    after["component_id"] == expected,
                    "Independent joint-component mapping differs",
                )
            preserved = {key: after[key] for key in before}
            preserved["component_id"] = before["component_id"]
            preserved["origin"] = {
                key: after["origin"][key] for key in before["origin"]
            }
            require(
                preserved == before, "Selected case or numerical parameters changed"
            )
            differences[stage].append(
                {
                    "case_id": after["case_id"],
                    "previous_component_id": before["component_id"],
                    "corrected_component_id": after["component_id"],
                    "native_target_kic": after["native_target_kic"],
                    "native_target_kic_status": after["native_target_kic_status"],
                }
            )
    names = list(stages)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if {name, other} == {"final-synthetic6000", "final-controls3000"}:
                continue
            require(
                not {r["component_id"] for r in stages[name]}
                & {r["component_id"] for r in stages[other]},
                "Joint identity overlaps across stages",
            )
    return stages, differences


def final_design(out):
    PREPARATION.update(preparation_identity())
    target = out / "cohorts/final-synthetic-plan-v4.json"
    require(not target.exists(), "Final V4 design already exists; preserve it")
    old_path = out / "cohorts/final-synthetic-plan.json"
    old_design = read(
        old_path, "6008525ac38db457dae787065f51cde0d2e340dd973d636fa6d08bc6cc17945c"
    )
    bind(old_design["source_snapshot"]["path"], old_design["source_snapshot"]["sha256"])
    previous = plan(out)
    mapping_path = STUDY / "checks/evolution/target-kic-corrected-components-v2.json"
    mapping = read(
        mapping_path, "2d66eb9be21964a4d475f6d59f8c64d38f603c511fd2b74ef2639a587150c058"
    )
    require(
        mapping["status"] == "COMPLETED_INPUT_ONLY_TARGET_KIC_CLOSURE_AUDIT"
        and mapping["mapping_namespace"] == COMPONENT_NAMESPACE,
        "Joint identity audit is not complete or namespace differs",
    )
    for cohort in mapping["cohort_bindings"].values():
        for key in ("manifest", "cases_csv", "clocks_csv"):
            bind(cohort[key]["path"], cohort[key]["sha256"])
    bind(
        STUDY / "checks/evolution/target-kic-closure-audit.json",
        "54d17f1b901a9b2e6ca0a784ae1948d94cdfaeb65ef97dd0fd889e4875ffa16b",
    )
    bind(
        STUDY / "checks/evolution/fresh-final-prefix24-peer-review.json",
        "6672eb983bbc81ac1032e6f795b8ccda3b964534f036a7d2a642dec2e6211328",
    )
    bind(
        out / "cohorts/final-synthetic-prefix24/manifest.json",
        "41809528a25c7488fd7cfbda9ff8007e72c8f3f6c9bc487974825452351347ee",
    )
    population = [
        native_host_metadata(r)
        for r in rows(
            STUDY / "population-host-model.csv",
            "fceb1923296fe08eeff475236a229c3edcb0a75f19c0b953bc12574d3a5d6838",
        )
    ]
    connections, historical = identity_connections(population)
    stages, differences = corrected_stages(previous, old_design, connections, mapping)
    prior_keys = set().union(
        *(
            identities(r["origin"])
            for entries in previous["stages"].values()
            for r in entries
        )
    )
    catalogue_keys = set().union(
        *(
            identities(r["origin"])
            for r in stages["final-controls3000"]
            if r["kind"] in ("native_eb", "catalogue_fp")
        )
    )
    history_keys = set().union(*(identities(r) for r in historical))
    forbidden = expand_forbidden(
        connections, protected_components() | prior_keys | catalogue_keys | history_keys
    )
    synthetic = [r["origin"] for r in stages["final-synthetic6000"]]
    final_keys = set().union(*(identities(r) for r in synthetic))
    require(not final_keys & forbidden, "Corrected fresh synthetic exclusion failed")
    require(
        len({r["component_id"] for r in stages["final-synthetic6000"]}) == 6000,
        "Corrected final synthetic components are not distinct",
    )
    available = [r for r in population if not identities(r) & forbidden]
    physical = read(
        old_design["physical_plan"]["path"], old_design["physical_plan"]["sha256"]
    )
    require(
        physical == json.loads(json.dumps(final_physical_plan())),
        "Physical draws changed at identity transition",
    )
    storage = calibrate.storage_check(out, anticipated_bytes=60 * 1024**3)
    experiment = out / "inputs/final-synthetic-v4"
    snapshot = out / "cohorts/evolve-controls-final-v4.py"
    require(
        not experiment.exists() and not snapshot.exists(),
        "Unbound existing V4 artifacts",
    )
    snapshot.write_bytes(Path(__file__).read_bytes())
    snapshot.chmod(0o444)
    save(experiment / "plan.json", physical)
    save(experiment / "freeze.json", {"native_index": PREPARATION["native_index"]})
    (experiment / "native-input-verification.json").symlink_to(
        STUDY / "native-input-verification.json"
    )
    transition_path = out / "cohorts/final-identity-transition-v3-v4.json"
    transition = {
        "status": "METADATA_ONLY_HOST_IDENTITY_CORRECTION_SELECTION_PRESERVED",
        "previous_design": {
            "path": str(old_path),
            "sha256": calibrate.digest(old_path),
        },
        "corrected_mapping": {
            "path": str(mapping_path),
            "sha256": calibrate.digest(mapping_path),
        },
        "source_snapshot": {
            "path": str(snapshot),
            "sha256": calibrate.digest(snapshot),
        },
        "component_namespace": COMPONENT_NAMESPACE,
        "selected_case_order_and_numerical_parameters_unchanged": True,
        "case_identity_changes": differences,
        "scope": (
            "Explicit native target KIC distinct from extraction TPF; unknown "
            "zero KIC creates no alias. Prior cohorts and V3 inputs remain "
            "unchanged. No redraw or outcome access."
        ),
    }
    save(transition_path, transition)
    bind(transition_path)
    for path, receipt in INPUTS.items():
        require(
            calibrate.file_stamp(path) == receipt["stamp"]
            and calibrate.digest(path) == receipt["sha256"],
            "Identity transition metadata changed during read",
        )
    value = {
        **{
            key: val
            for key, val in old_design.items()
            if key
            not in (
                "schema",
                "plan_path",
                "stages",
                "physical_plan",
                "source_snapshot",
                "preparation_identity",
                "input_bindings",
                "storage_admission",
                "population_comparison",
                "excluded_identity_count",
                "final_synthetic_components",
            )
        },
        "schema": 4,
        "plan_path": str(target),
        "stages": {key: stages[key] for key in old_design["stages"]},
        "physical_plan": {
            "path": str(experiment / "plan.json"),
            "sha256": calibrate.digest(experiment / "plan.json"),
        },
        "source_snapshot": {
            "path": str(snapshot),
            "sha256": calibrate.digest(snapshot),
        },
        "preparation_identity": PREPARATION,
        "input_bindings": dict(INPUTS),
        "storage_admission": storage,
        "identity_transition": {
            "path": str(transition_path),
            "sha256": calibrate.digest(transition_path),
        },
        "component_namespace": COMPONENT_NAMESPACE,
        "population_comparison": {
            "full_listed": population_description(population),
            "available_after_exclusions": population_description(available),
            "selected_synthetic": population_description(synthetic),
        },
        "excluded_identity_count": len(forbidden),
        "final_synthetic_components": sorted(
            {r["component_id"] for r in stages["final-synthetic6000"]}
        ),
        "native_target_kic_status_counts": dict(
            Counter(r["native_target_kic_status"] for r in synthetic)
        ),
    }
    save(target, value)
    print(
        json.dumps(
            {
                "design": str(target),
                "sha256": calibrate.digest(target),
                "available": value["population_comparison"][
                    "available_after_exclusions"
                ],
                "transition": calibrate.digest(transition_path),
            }
        ),
        flush=True,
    )


def loss_description(values):
    return {
        "cadences": len(values),
        "nonzero_loss_cadences": int(np.count_nonzero(values)),
        "positive_signal_cadences_above_roundoff": int(
            np.count_nonzero(values > 1e-10)
        ),
        "sampled_depth_ppm": float(values.max() * 1e6) if len(values) else None,
        "minimum_loss": float(values.min()) if len(values) else None,
    }


def raw_injection_description(archive, truth):
    flux, loss, delta = archive["raw_flux"], archive["raw_loss"], archive["raw_delta"]
    quarters = []
    for quarter in np.unique(archive["raw_quarter"]):
        use = archive["raw_quarter"] == quarter
        good = use & np.isfinite(flux) & (archive["raw_quality"] == 0)
        median = float(np.median(flux[good])) if good.any() else None
        expected = -median * loss[use] if median is not None else np.zeros(use.sum())
        require(
            np.array_equal(delta[use], expected),
            "Raw decrement differs from declared injection",
        )
        quarters.append(
            {
                "quarter": int(quarter),
                "generator_baseline": median,
                "nonpositive_baseline": median is not None and median <= 0,
                "raw_brightening_cadences": int(np.count_nonzero(delta[use] > 0)),
            }
        )
    require(
        np.isfinite(loss).all() and (not len(loss) or np.min(loss) >= -1e-10),
        "Material invalid transit loss",
    )
    usable = (
        np.isfinite(archive["raw_time"])
        & np.isfinite(flux)
        & np.isfinite(archive["raw_error"])
        & (archive["raw_error"] > 0)
        & (archive["raw_quality"] == 0)
    )
    records = {
        "time": archive["raw_time"][usable],
        "quarter": archive["raw_quarter"][usable],
    }
    events, _, points = calibrate.injections.observed_events(
        records,
        [],
        truth["duration_hours"] / 24,
        truth["period_days"],
        truth["epoch_bkjd"],
    )
    return {
        **loss_description(loss),
        "usable_quality_cadences": int(usable.sum()),
        "usable_nonzero_loss_cadences": int(np.count_nonzero(loss[usable])),
        "usable_geometric_events": events,
        "usable_in_transit_cadences": points,
        "quarter_baselines": quarters,
    }


def injection_observability(receipt_path, source, out, folder=None):
    array_entry = source.get("arrays", source.get("raw_arrays"))
    require(
        array_entry is not None, "Selected draw lacks preserved raw injection arrays"
    )
    array_path = receipt_path.parent / array_entry["path"]
    bind(array_path, array_entry["sha256"])
    receipt_sha = calibrate.digest(bind(receipt_path))
    folder = folder or out / "inputs/final-synthetic-v4/observability"
    path = folder / f"{source['trial']['source_id']}.json"
    identity = {
        "input_receipt_path": str(receipt_path),
        "input_receipt_sha256": receipt_sha,
        "input_arrays_sha256": array_entry["sha256"],
        "builder_sha256": calibrate.digest(__file__),
    }
    if path.exists():
        result = read(path)
        require(result["identity"] == identity, "Observability receipt binding differs")
        return result, path
    with np.load(array_path, allow_pickle=False) as archive:
        raw = raw_injection_description(archive, source["truth"])
        prepared = (
            loss_description(archive["prepared_loss"])
            if "prepared_loss" in archive
            else None
        )
        records = archive["injected"] if "injected" in archive else None
        original = archive["original"] if "original" in archive else None
        byte_equal = bool(
            original is not None
            and records is not None
            and original.tobytes() == records.tobytes()
        )
        for name, values in (("original", original), ("injected", records)):
            if values is not None:
                require(
                    hashlib.sha256(values.tobytes()).hexdigest()
                    == source[name + "_prepared_sha256"],
                    "Prepared record byte hash differs",
                )
        if records is not None:
            retained = set(records["quarter"].tolist())
            require(
                not any(
                    q["nonpositive_baseline"] and q["quarter"] in retained
                    for q in raw["quarter_baselines"]
                ),
                "Nonpositive generator baseline in retained quarter",
            )
    events = source["truth"].get("observed_events")
    failure = (
        "NO_EXPOSURE_INTEGRATED_SIGNAL_ABOVE_NUMERICAL_FLOOR_ON_RAW_CADENCES"
        if not raw["positive_signal_cadences_above_roundoff"]
        else "INPUT_PREPARATION_FAILED"
        if source["status"] != "PREPARED"
        else "NO_EXPOSURE_INTEGRATED_SIGNAL_ABOVE_NUMERICAL_FLOOR_ON_PREPARED_CADENCES"
        if not prepared["positive_signal_cadences_above_roundoff"]
        else ""
    )
    result = {
        "identity": identity,
        "preparation_status": source["status"],
        "first_preparation_failure_stage": source["first_failure_stage"],
        "first_preparation_failure": source["reason"],
        "target_integrated_depth_ppm": source["truth"]["target_integrated_depth_ppm"],
        "raw": raw,
        "prepared": prepared,
        "observed_geometric_events": events,
        "fewer_than_three_geometric_events": events < 3 if events is not None else None,
        "original_injected_prepared_bytes_identical": byte_equal,
        "first_injection_observability_failure": failure,
        "claim_boundary": (
            "No replacement or period/phase adjustment. Events use "
            "physicalT14 and can omit exposure wings. Model-positive "
            "threshold1e-10 separates known analytic roundoff; exact nonzero "
            "counts are retained. Input failures remain in the positive "
            "denominator."
        ),
    }
    require(
        calibrate.digest(receipt_path) == receipt_sha
        and calibrate.digest(array_path) == array_entry["sha256"],
        "Injection inputs changed during observability accounting",
    )
    calibrate.checked_input(receipt_path)
    save(path, result)
    return result, path


def final_bindings(out, design_sha, control_boundary_sha=None):
    require(design_sha, "An externally bound final design SHA256 is required")
    target = bind(out / "cohorts/final-synthetic-plan-v4.json", design_sha)
    design = read(target)
    bind(design["source_snapshot"]["path"], design["source_snapshot"]["sha256"])
    PREPARATION.update(preparation_identity())
    expected = design["preparation_identity"]
    if control_boundary_sha:
        boundary = read(
            out / "cohorts/control-construction-boundary-v6.json", control_boundary_sha
        )
        require(
            boundary["design_sha256"] == design_sha,
            "Control boundary changes selected design",
        )
        bind(boundary["source_snapshot"]["path"], boundary["source_snapshot"]["sha256"])
        require(
            boundary["source_snapshot"]["sha256"] == calibrate.digest(__file__),
            "Control builder differs from bound snapshot",
        )
        for path, sha in boundary["preserved_before_after"].items():
            bind(path, sha)
        expected = boundary["preparation_identity"]
    else:
        require(
            design["source_snapshot"]["sha256"] == calibrate.digest(__file__),
            "Final builder differs from declared snapshot",
        )
    require(
        PREPARATION == expected,
        "Final preparation identity changed",
    )
    for path, entry in design["input_bindings"].items():
        if control_boundary_sha and Path(path).resolve() == Path(__file__).resolve():
            require(
                entry["sha256"] == design["source_snapshot"]["sha256"],
                "Historical builder replacement is not declared",
            )
            continue
        bind(path, entry["sha256"])
        require(INPUTS[path]["stamp"] == entry["stamp"], "Final metadata stamp changed")
    physical_path = bind(
        design["physical_plan"]["path"], design["physical_plan"]["sha256"]
    )
    experiment = out / "inputs/final-synthetic-v4"
    require(
        read(experiment / "freeze.json")
        == {"native_index": PREPARATION["native_index"]}
        and (experiment / "native-input-verification.json").resolve()
        == (STUDY / "native-input-verification.json").resolve(),
        "Fresh native index/ledger wiring differs",
    )
    bind(
        experiment / "native-input-verification.json",
        PREPARATION["native_ledger_sha256"],
    )
    return design, read(physical_path)


def final_prepared_row(out, entry):
    receipt_path = (
        out
        / "inputs/final-synthetic-v4/inputs/preferred_full/fresh_final"
        / f"{entry['source_id']}.json"
    )
    source = calibrate.checked_input(receipt_path)
    array = source.get("arrays", source.get("raw_arrays"))
    for path in (receipt_path, receipt_path.parent / array["path"]):
        if path.stat().st_mode & 0o222:
            path.chmod(0o444)
    require(
        source["trial"] == entry["origin"]
        and source["plan_sha256"]
        == calibrate.digest(out / "inputs/final-synthetic-v4/plan.json"),
        "Fresh trial/physical plan changed",
    )
    require(
        source["truth"]["target_integrated_depth_ppm"] == entry["depth_ppm"],
        "Fresh depth draw differs",
    )
    observability, obs_path = injection_observability(receipt_path, source, out)
    value = {
        "preparation_status": source["status"],
        "first_failure": source["reason"],
        "arrays": {
            "path": str(receipt_path.parent / source["arrays"]["path"]),
            "sha256": source["arrays"]["sha256"],
        }
        if "arrays" in source
        else {},
    }
    updated = {**entry, "origin": {**entry["origin"], "truth": source["truth"]}}
    prepared = finish_case(updated, value, receipt_path)
    prepared.update(
        sampled_depth_ppm=source["truth"].get("sampled_depth_ppm", ""),
        observed_events=source["truth"].get("observed_events", ""),
        raw_nonzero_loss_cadences=observability["raw"]["nonzero_loss_cadences"],
        prepared_nonzero_loss_cadences=(observability["prepared"] or {}).get(
            "nonzero_loss_cadences", ""
        ),
        first_injection_observability_failure=observability[
            "first_injection_observability_failure"
        ],
        observability_path=str(obs_path),
        observability_sha256=calibrate.digest(obs_path),
    )
    for product, proof in source["native_input_verification"]["products"].items():
        NATIVE_PRODUCTS[product] = proof
        bind(proof["receipt"]["path"], proof["receipt"]["sha256"])
    return prepared, clock(updated, prepared, value), observability


def preserved_prefix_numerics(out, entries):
    comparisons = []
    for entry in entries:
        sid = entry["source_id"]
        relative = Path("inputs/preferred_full/fresh_final") / f"{sid}.json"
        before_path = out / "inputs/final-synthetic" / relative
        after_path = out / "inputs/final-synthetic-v4" / relative
        before, after = (
            calibrate.checked_input(path) for path in (before_path, after_path)
        )
        bind(before_path)
        require(
            before["status"] == after["status"]
            and before["truth"] == after["truth"]
            and before["reason"] == after["reason"],
            "V4 changed the preserved physical truth or eligibility",
        )
        paths = []
        for path, receipt in ((before_path, before), (after_path, after)):
            product = receipt.get("arrays", receipt.get("raw_arrays"))
            paths.append(bind(path.parent / product["path"], product["sha256"]))
        with np.load(paths[0], allow_pickle=False) as old:
            with np.load(paths[1], allow_pickle=False) as new:
                require(set(old.files) == set(new.files), "V4 array fields changed")
                for field in old.files:
                    require(
                        old[field].dtype == new[field].dtype
                        and old[field].shape == new[field].shape
                        and old[field].tobytes() == new[field].tobytes(),
                        "V4 numerical array changed: " + field,
                    )
                comparisons.append(
                    {"source_id": sid, "equal_array_fields": len(old.files)}
                )
    return {
        "status": "EXACT_V3_V4_SAVED_NUMERICAL_ARRAY_AND_TRUTH_PARITY",
        "cases": comparisons,
        "scope": (
            "Same selected prefix24 regenerated; metadata-only identity correction."
        ),
    }


def prepare_final_synthetic(out, count, workers, design_sha):
    require(
        1 <= count <= 6000 and 1 <= workers <= 2,
        "Fresh preparation is bounded to6000 draws/two workers",
    )
    design, physical = final_bindings(out, design_sha)
    folder = (
        out
        / "cohorts"
        / (
            "final-synthetic6000-v4"
            if count == 6000
            else f"final-synthetic-prefix{count}-v4"
        )
    )
    require(
        not (folder / "manifest.json").exists(),
        "Fresh prepared cohort already complete",
    )
    calibrate.storage_check(
        out, anticipated_bytes=60 * 1024**3 if count == 6000 else count * 16 * 1024**2
    )
    entries = design["stages"]["final-synthetic6000"][:count]
    experiment = out / "inputs/final-synthetic-v4"
    tasks = [
        (str(experiment), r["origin"], physical, "preferred_full") for r in entries
    ]
    calibrate.execute(
        tasks,
        calibrate.prepare_trial,
        workers,
        experiment / f"prepare-{count}-progress.json",
    )
    cases, clocks, observations = [], [], []
    for entry in entries:
        prepared, timing, observation = final_prepared_row(out, entry)
        cases.append(prepared)
        clocks.append(timing)
        observations.append(observation)
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "cases.csv", cases, FIELDS)
    write_csv(folder / "clocks.csv", clocks, CLOCK_FIELDS)
    save(folder / "contamination.json", contamination(entries))
    numerical_parity = preserved_prefix_numerics(out, entries) if count == 24 else None
    require(
        PREPARATION == preparation_identity(),
        "Final runtime/code changed during preparation",
    )
    for path, entry in INPUTS.items():
        require(
            calibrate.file_stamp(path) == entry["stamp"]
            and calibrate.digest(path) == entry["sha256"],
            "Final input changed during preparation",
        )
    for path, entry in NATIVE_PRODUCTS.items():
        require(
            calibrate.file_stamp(path) == entry["stamp"],
            "Final native product stamp changed",
        )
    manifest = {
        "status": "COMPLETE_PREPARED_COHORT_NO_DETECTOR_SEARCH",
        "stage": folder.name,
        "requested_cases": count,
        "positive_signals": count,
        "preparation_counts": dict(Counter(r["preparation_status"] for r in cases)),
        "population_counts": dict(Counter(r["population"] for r in cases)),
        "observability_failures": dict(
            Counter(r["first_injection_observability_failure"] for r in observations)
        ),
        "prepared_sampled_zero": sum(
            r["prepared"] is not None
            and r["prepared"]["positive_signal_cadences_above_roundoff"] == 0
            for r in observations
        ),
        "prepared_fewer_than_three_geometric_events": sum(
            r["fewer_than_three_geometric_events"] is True for r in observations
        ),
        "positive_denominator": (
            "All selected draws, including input failures, zero sampled loss "
            "and fewer-than-three-event draws; no replacements."
        ),
        "design": {
            "path": str(out / "cohorts/final-synthetic-plan-v4.json"),
            "sha256": calibrate.digest(out / "cohorts/final-synthetic-plan-v4.json"),
        },
        "source_sha256": calibrate.digest(__file__),
        "preparation_identity": PREPARATION,
        "v3_v4_prefix_numerical_parity": numerical_parity,
        "files": {
            name: {
                "path": str(folder / name),
                "sha256": calibrate.digest(folder / name),
            }
            for name in ("cases.csv", "clocks.csv", "contamination.json")
        },
        "input_bindings": INPUTS,
    }
    save(folder / "manifest.json", manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "stage",
                    "requested_cases",
                    "preparation_counts",
                    "prepared_sampled_zero",
                    "prepared_fewer_than_three_geometric_events",
                )
            }
        ),
        flush=True,
    )


def build_controls(out, design_sha, control_boundary_sha):
    require(control_boundary_sha, "A bound control-construction boundary is required")
    PREPARATION.update(preparation_identity())
    specification, _ = final_bindings(out, design_sha, control_boundary_sha)
    prior = read(out / "cohorts/final-synthetic6000-v4/manifest.json")
    require(
        prior["status"] == "COMPLETE_PREPARED_COHORT_NO_DETECTOR_SEARCH"
        and prior["requested_cases"] == 6000
        and prior["design"]["sha256"] == design_sha,
        "Final synthetic inputs are not complete and bound",
    )
    for item in prior["files"].values():
        bind(item["path"], item["sha256"])
    stage = "final-controls3000"
    entries = specification["stages"][stage]
    require(len(entries) == 3000, "Final control denominator differs")
    folder = out / "cohorts" / (stage + "-v6")
    require(
        not (folder / "manifest.json").exists(), "Cohort already complete; preserve it"
    )
    calibrate.storage_check(out, anticipated_bytes=4 * 1024**3)
    folder.mkdir(parents=True, exist_ok=True)
    cases, clocks, observations = [], [], []
    for index, entry in enumerate(entries, 1):
        if entry["kind"] in ("original", "inverted", "year_scrambled"):
            receipt_path = (
                out
                / "inputs/final-synthetic-v4/inputs/preferred_full/fresh_final"
                / f"{entry['source_id']}.json"
            )
            source = calibrate.checked_input(receipt_path)
            require(
                source["trial"] == entry["origin"],
                "Final paired-control origin differs",
            )
            entry = {
                **entry,
                "origin": {
                    **entry["origin"],
                    "truth": source["truth"],
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": calibrate.digest(receipt_path),
                },
            }
            value, receipt_path = prepare_study(out, entry)
        else:
            value, receipt_path = prepare_catalogue(out, entry)
        prepared = finish_case(entry, value, receipt_path)
        if entry["origin"].get("fresh_instrument"):
            observation, obs_path = instrument_observability(out, value, receipt_path)
            observations.append(observation)
            prepared.update(
                requested_instrument_kind=observation["requested_mechanism"],
                instrument_observability_status=observation["status"],
                instrument_observability_path=str(obs_path),
                instrument_observability_sha256=calibrate.digest(obs_path),
            )
        cases.append(prepared)
        if entry["positive"] == "true":
            clocks.append(clock(entry, prepared, value))
        if index % 25 == 0 or index == len(entries):
            print(
                stage,
                index,
                "/",
                len(entries),
                Counter(row["preparation_status"] for row in cases),
                flush=True,
            )
    controls = contamination(entries)
    write_csv(folder / "cases.csv", cases, FIELDS)
    write_csv(folder / "clocks.csv", clocks, CLOCK_FIELDS)
    save(folder / "contamination.json", controls)
    value = {
        "schema": 1,
        "status": "COMPLETE_PREPARED_COHORT_NO_DETECTOR_SEARCH",
        "stage": stage,
        "positive_signals": len(clocks),
        "requested_cases": len(cases),
        "kind_counts": dict(Counter(row["kind"] for row in cases)),
        "preparation_counts": dict(Counter(row["preparation_status"] for row in cases)),
        "positive_denominator": (
            "Every requested synthetic/real planet signal, including all "
            "input failures; one blind search may serve several real "
            "clocks."
        ),
        "controls_boundary": (
            "Proposal burden reported separately; originals/inversion/SCR1 "
            "are not certified astrophysical false positives."
        ),
        "paths": (
            "Absolute local paths; transfer must rewrite input/receipt "
            "paths using explicit SHA-bound file mapping."
        ),
        "previous_outcome_dependence": specification["real_golden_boundary"],
        "depth_edges_ppm": DEPTH_EDGES,
        "declared_control_allocations": FINAL_CONTROLS,
        "control_construction_boundary_sha256": control_boundary_sha,
        "instrument_observability_counts": dict(
            Counter(row["status"] for row in observations)
        ),
        "instrument_observability_by_mechanism": dict(
            Counter(
                row["requested_mechanism"] + ":" + row["status"] for row in observations
            )
        ),
        "cases_csv": {
            "path": str(folder / "cases.csv"),
            "sha256": calibrate.digest(folder / "cases.csv"),
        },
        "clocks_csv": {
            "path": str(folder / "clocks.csv"),
            "sha256": calibrate.digest(folder / "clocks.csv"),
        },
        "contamination": {
            "path": str(folder / "contamination.json"),
            "sha256": calibrate.digest(folder / "contamination.json"),
        },
        "plan": {
            "path": specification.get("plan_path", str(out / "cohorts/plan.json")),
            "sha256": calibrate.digest(
                specification.get("plan_path", out / "cohorts/plan.json")
            ),
        },
        "source_sha256": calibrate.digest(__file__),
        "preparation_identity": PREPARATION,
        "input_bindings": INPUTS,
    }
    for path, item in INPUTS.items():
        require(
            calibrate.file_stamp(path) == item["stamp"]
            and calibrate.digest(path) == item["sha256"],
            "Input changed during preparation",
        )
    require(
        PREPARATION == preparation_identity(),
        "Runtime or preparation dependencies changed",
    )
    for product, proof in NATIVE_PRODUCTS.items():
        require(
            calibrate.file_stamp(product) == proof["stamp"],
            "Native input product changed during cohort preparation",
        )
    save(folder / "manifest.json", value)
    print(
        json.dumps(
            {
                "stage": stage,
                "status": value["status"],
                "manifest_sha256": calibrate.digest(folder / "manifest.json"),
            }
        ),
        flush=True,
    )


def remodeled_case(entry, experiment, out):
    receipt_path = (
        experiment / "inputs/preferred_full/fresh_final" / f"{entry['source_id']}.json"
    )
    source = calibrate.checked_input(bind(receipt_path))
    array = source.get("arrays") or source["raw_arrays"]
    bind(receipt_path.parent / array["path"], array["sha256"])
    observation, path = injection_observability(
        receipt_path, source, out, experiment / "observability"
    )
    bind(path)
    value = {
        "preparation_status": source["status"],
        "first_failure": source["reason"],
        "arrays": {
            "path": str(receipt_path.parent / array["path"]),
            "sha256": array["sha256"],
        }
        if source["status"] == "PREPARED"
        else {},
    }
    prepared = finish_case(entry, value, receipt_path)
    prepared.update(
        phot_g_mean_mag=source["truth"]["phot_g_mean_mag"],
        sampled_depth_ppm=source["truth"].get("sampled_depth_ppm", ""),
        sampled_raw_depth_ppm=observation["raw"]["sampled_depth_ppm"],
        sampled_prepared_depth_ppm=(observation["prepared"] or {}).get(
            "sampled_depth_ppm", ""
        ),
        observed_events=source["truth"].get("observed_events", ""),
        raw_nonzero_loss_cadences=observation["raw"]["nonzero_loss_cadences"],
        prepared_nonzero_loss_cadences=(observation["prepared"] or {}).get(
            "nonzero_loss_cadences", ""
        ),
        first_injection_observability_failure=observation[
            "first_injection_observability_failure"
        ],
        observability_path=str(path),
        observability_sha256=calibrate.digest(path),
    )
    timing = clock({**entry, "origin": {"truth": source["truth"]}}, prepared, value)
    return prepared, timing, observation


def remodel_cohort(args):
    require(
        args.label and Path(args.label).name == args.label,
        "Remodeled cohort requires a single fresh label",
    )
    folder, experiment = (
        args.out / "cohorts" / args.label,
        args.out / "inputs" / args.label,
    )
    require(
        not folder.exists() and not experiment.exists(),
        "Preserve all existing remodeled inputs and cohort evidence",
    )
    request = argparse.Namespace(
        prior_manifest=args.prior_manifest,
        prior_sha256=args.prior_sha256,
        plan=args.recipe,
        plan_sha256=args.recipe_sha256,
        out=experiment,
        count=args.count,
        workers=args.workers,
    )
    cases, _, prior, _ = calibrate.remodel_admission(request)
    prior_path = bind(args.prior_manifest, args.prior_sha256)
    for item in prior["files"].values():
        bind(item["path"], item["sha256"])
    identity = preparation_identity()
    construction = calibrate.remodel(request)
    bind(experiment / "manifest.json")
    bind(args.recipe, args.recipe_sha256)
    bind(experiment / "plan.json")
    values = [remodeled_case(entry, experiment, args.out) for entry in cases]
    prepared, clocks, observations = map(list, zip(*values))
    folder.mkdir(parents=True)
    fields = list(dict.fromkeys(key for row in prepared for key in row))
    write_csv(folder / "cases.csv", prepared, fields)
    write_csv(folder / "clocks.csv", clocks, CLOCK_FIELDS)
    contamination_path = Path(prior["files"]["contamination.json"]["path"])
    with (folder / "contamination.json").open("xb") as stream:
        stream.write(contamination_path.read_bytes())
    (folder / "contamination.json").chmod(0o444)
    manifest = {
        "status": "COMPLETE_PREPARED_COHORT_NO_DETECTOR_SEARCH",
        "stage": args.label,
        "requested_cases": len(cases),
        "positive_signals": len(clocks),
        "kind_counts": {"synthetic_planet": len(cases)},
        "population_counts": dict(Counter(row["population"] for row in prepared)),
        "preparation_counts": dict(
            Counter(row["preparation_status"] for row in prepared)
        ),
        "prepared_sampled_zero": sum(
            row["prepared"] is not None and row["prepared"]["sampled_depth_ppm"] == 0
            for row in observations
        ),
        "positive_denominator": (
            "Same selected prefix and case/component identities; every requested "
            "draw retained, including preparation failures and zero model support."
        ),
        "prior_manifest": {"path": str(prior_path), "sha256": args.prior_sha256},
        "recipe": {"path": str(args.recipe), "sha256": args.recipe_sha256},
        "construction": {
            "path": str(experiment / "manifest.json"),
            "sha256": calibrate.digest(experiment / "manifest.json"),
            "status": construction["status"],
        },
        "source_sha256": calibrate.digest(__file__),
        "preparation_identity": identity,
        "input_bindings": INPUTS,
        **{
            key: {"path": str(path), "sha256": calibrate.digest(path)}
            for key, path in (
                ("cases_csv", folder / "cases.csv"),
                ("clocks_csv", folder / "clocks.csv"),
                ("contamination", folder / "contamination.json"),
                ("plan", experiment / "plan.json"),
            )
        },
    }
    if identity != preparation_identity():
        raise ValueError("Remodel cohort runtime/source/index changed")
    for path, entry in list(INPUTS.items()):
        require(
            calibrate.digest(path) == entry["sha256"]
            and calibrate.file_stamp(path) == entry["stamp"],
            "Remodel cohort input changed: " + path,
        )
    for entry in prepared:
        calibrate.checked_input(entry["receipt_path"])
    save(folder / "manifest.json", manifest)
    print(json.dumps({"manifest": str(folder / "manifest.json"), "cases": len(cases)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "evolution")
    parser.add_argument(
        "--stage",
        choices=(
            "final-synthetic-plan",
            "final-synthetic6000",
            "final-controls3000",
            "remodel-synthetic",
        ),
        required=True,
    )
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--design-sha256")
    parser.add_argument("--control-boundary-sha256")
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--prior-sha256")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--recipe-sha256")
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.stage == "final-synthetic-plan":
        final_design(args.out)
    elif args.stage == "final-synthetic6000":
        prepare_final_synthetic(args.out, args.count, args.workers, args.design_sha256)
    elif args.stage == "remodel-synthetic":
        require(
            all(
                (
                    args.prior_manifest,
                    args.prior_sha256,
                    args.recipe,
                    args.recipe_sha256,
                )
            ),
            "Remodel requires explicit prior manifest and recipe hash bindings",
        )
        remodel_cohort(args)
    else:
        build_controls(args.out, args.design_sha256, args.control_boundary_sha256)


if __name__ == "__main__":
    main()
