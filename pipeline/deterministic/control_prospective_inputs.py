"""Materialize unchanged prospective inputs without granting scientific roles."""

import argparse
import hashlib
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.deterministic.control_bindings import (
    protected_sources,
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_eligibility import (
    BOUNDARY,
    COHORTS,
    header_identity,
    mask_exposure,
    save,
    verify_plan,
)
from pipeline.deterministic.control_execution_inputs import reviewed_scopes
from pipeline.deterministic.control_pair_inputs import (
    protection_sets,
    require_permitted,
    sealed_raw_hashes,
)
from pipeline.deterministic.control_physical_readiness import checked_report

MINIMUM_FREE_BYTES = 2 * 1024**3


def admission(directory, additional):
    require(type(additional) is int and additional >= 0, "Invalid storage forecast")
    free = shutil.disk_usage(directory).free
    require(free - additional >= MINIMUM_FREE_BYTES, "Cloud disk reserve crossed")
    return {"filesystem_free_bytes": free, "admitted_additional_bytes": additional}


def exact_index(rows, expected):
    ids = [r["source_id"] for r in rows]
    require(ids == expected and len(ids) == len(set(ids)), "Source order/scope differs")
    return {r["source_id"]: r for r in rows}


def source_inputs(plan, guide, pairs):
    guide_ids = [
        s["source_id"] for s in plan["sources"] if s["cohort"] == "guide_star_family"
    ]
    pair_ids = [
        s["source_id"] for s in plan["sources"] if s["cohort"] == "contamination_pair"
    ]
    guide_rows = exact_index(
        [r for r in guide["sources"] if r["cohort"] == "guide_star_family"], guide_ids
    )
    pair_rows = exact_index(pairs["targets"], pair_ids)
    inputs = {}
    for sid, row in guide_rows.items():
        record = read_json(verified_reference(row["archive_record"]))
        require(
            record["status"] == "VERIFIED_FULL_PRODUCT", "Guide payload unavailable"
        )
        inputs[sid] = {
            "record": row["archive_record"],
            "payload": record["payload"],
            "preparation": row["witness"]["preparation"],
        }
    for sid, row in pair_rows.items():
        record = read_json(verified_reference(row["record"]))
        require(
            record["status"] == "ORIGINAL_TARGET_RAW_BYTES_AND_PREPARATION_VERIFIED",
            "Contamination target payload unavailable",
        )
        inputs[sid] = {
            "record": row["record"],
            **{k: record[k] for k in ("payload", "preparation")},
        }
    return inputs


def original_payload(source, item, protected, protected_hashes):
    require(source["source_id"] not in protected, "Protected sealed source")
    require(
        source["pool"]["raw_fits_sha256"] not in protected_hashes,
        "Protected sealed raw identity",
    )
    path = verified_reference(item["payload"])
    require(
        item["payload"]["sha256"] == source["pool"]["raw_fits_sha256"],
        "Original raw hash changed",
    )
    return path


def source_metadata(source, record):
    quarters = header_identity(source, record["headers"])
    cards = record["headers"][0]["cards"]
    magnitude = cards.get("GMAG")
    require(
        magnitude is None or math.isfinite(float(magnitude)), "Nonfinite G magnitude"
    )
    return {
        "source_id": source["source_id"],
        "origin_tpf_kic": int(source["parent_tpf"]),
        "kic": int(source["product"]["kic"]),
        "raw_fits_sha256": source["pool"]["raw_fits_sha256"],
        "shard": int(source["shard"]),
        "shard_path": source["uri"],
        "ra_deg": float(cards["RA_OBJ"]),
        "dec_deg": float(cards["DEC_OBJ"]),
        "phot_g_mean_mag": None if magnitude is None else float(magnitude),
        "quarter_parent_bindings": quarters,
        "quarter_parent_complete": False,
        "prospective_cohort": source["cohort"],
    }


def array_inventory(raw):
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": value.nbytes,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in raw.items()
    }


def save_arrays(path, raw):
    require(not path.exists(), "Array output already exists")
    np.savez_compressed(path, **raw)
    with np.load(path, allow_pickle=False) as saved:
        require(
            len(saved.files) == len(set(saved.files)) and set(saved.files) == set(raw),
            "Encoded array inventory changed",
        )
        require(
            array_inventory({name: saved[name] for name in saved.files})
            == array_inventory(raw),
            "Encoded array content changed",
        )
    return reference(path)


def materialize_source(source, item, directory, protected, protected_hashes):
    from pipeline.deterministic.cloud_public import fits_arrays

    payload = original_payload(source, item, protected, protected_hashes)
    record = read_json(verified_reference(item["record"]))
    require(record["source"] == source, "Archive source metadata changed")
    metadata = source_metadata(source, record)
    raw = fits_arrays(payload, source["source_id"], {"product_kic": metadata["kic"]})
    require(
        mask_exposure(raw)[0] == item["preparation"], "Preparation exposure changed"
    )
    inventory = array_inventory(raw)
    admission(directory, sum(v.nbytes for v in raw.values()) + 1024**2)
    path = directory / "arrays" / f"{source['source_id']}.npz"
    arrays = save_arrays(path, raw)
    return {
        "source_id": source["source_id"],
        "cohort": source["cohort"],
        "status": "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND",
        "original_payload": item["payload"],
        "archive_record": item["record"],
        "arrays": arrays,
        "array_inventory": inventory,
        "metadata": metadata,
        "preparation": item["preparation"],
        "component_id": source["pool"]["metadata_dependency_component_id"],
        "final_scoped_role_authorization": False,
    }


def write_manifest(plan, records, directory, plan_path, protection_receipt):
    sources = {
        row["source_id"]: {
            "arrays": f"arrays/{row['source_id']}.npz",
            "sha256": row["arrays"]["sha256"],
            "metadata": row["metadata"],
        }
        for row in records
        if row["status"] == "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND"
    }
    manifest = {
        "schema": 1,
        "role": "development",
        "measurement_scope": "OPENED_DEVELOPMENT_ONLY",
        "status": "UNFROZEN_PROSPECTIVE_INPUTS_ONLY",
        "original_plan": reference(plan_path),
        "protected_inputs": protection_receipt,
        "sources": sources,
        "boundary": BOUNDARY,
    }
    save(directory / "input-manifest.json", manifest)
    for cohort in sorted(COHORTS):
        ids = [s["source_id"] for s in plan["sources"] if s["cohort"] == cohort]
        save(directory / f"{cohort}-source-ids.json", ids)
    save(directory / "source-ids.json", [s["source_id"] for s in plan["sources"]])
    return reference(directory / "input-manifest.json")


def validate_materialization(report_path):
    report = read_json(report_path)
    require(
        report["schema"] == 1
        and report["boundary"] == BOUNDARY
        and report["runtime_policy_required"] == "DEVELOPMENT"
        and report["final_scoped_role_authorization"] is False,
        "Materialized input scope changed",
    )
    require(
        "protected_inputs" in report["inputs"],
        "Current full-component protection receipt required",
    )
    protection = protection_sets(report["inputs"]["protected_inputs"])
    for row in report["inputs"].values():
        verified_reference(row)
    verified_reference(report["implementation"])
    plan = read_json(verified_reference(report["inputs"]["plan"]))
    verify_plan(plan)
    manifest_path = verified_reference(report["input_manifest"])
    manifest = read_json(manifest_path)
    require(
        manifest["schema"] == 1
        and manifest["role"] == "development"
        and manifest["measurement_scope"] == "OPENED_DEVELOPMENT_ONLY"
        and manifest["boundary"] == BOUNDARY
        and manifest["original_plan"] == report["inputs"]["plan"]
        and manifest["protected_inputs"] == report["inputs"]["protected_inputs"],
        "Runtime input manifest scope changed",
    )
    rows = [read_json(verified_reference(row)) for row in report["records"]]
    exact_index(rows, [s["source_id"] for s in plan["sources"]])
    originals = {s["source_id"]: s for s in plan["sources"]}
    inputs = source_inputs(
        plan,
        read_json(verified_reference(report["inputs"]["guide"])),
        read_json(verified_reference(report["inputs"]["pairs"])),
    )
    verified = {}
    for row in rows:
        if row["status"] == "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND":
            sid = row["source_id"]
            validate_source_proof(originals[sid], inputs[sid], row, protection)
            validate_encoded_source(row, manifest["sources"][sid], manifest_path)
            verified[sid] = row
    require(set(verified) == set(manifest["sources"]), "Runtime source scope differs")
    return {"attempted_sources": len(rows), "bound_sources": len(verified)}


def validate_source_proof(source, original, row, protection):
    from pipeline.deterministic.cloud_public import fits_arrays

    require_permitted(source, protection)
    require(
        row["archive_record"] == original["record"]
        and row["original_payload"] == original["payload"]
        and row["cohort"] == source["cohort"]
        and row["component_id"] == source["pool"]["metadata_dependency_component_id"]
        and row["final_scoped_role_authorization"] is False,
        "Original prospective source provenance changed",
    )
    record = read_json(verified_reference(original["record"]))
    require(
        row["metadata"] == source_metadata(source, record), "Source metadata changed"
    )
    path = original_payload(
        source,
        original,
        protection["source_ids"],
        protection["raw_or_pixel_content_hashes"],
    )
    raw = fits_arrays(
        path, source["source_id"], {"product_kic": row["metadata"]["kic"]}
    )
    require(
        row["array_inventory"] == array_inventory(raw),
        "Array content proof differs from original raw reader",
    )
    require(
        row["preparation"] == original["preparation"] == mask_exposure(raw)[0],
        "Original preparation exposure changed",
    )


def validate_encoded_source(row, runtime, manifest_path):
    require(
        runtime
        == {
            "arrays": f"arrays/{row['source_id']}.npz",
            "sha256": row["arrays"]["sha256"],
            "metadata": row["metadata"],
        },
        "Runtime source metadata or input binding differs",
    )
    verified_reference(row["original_payload"])
    verified_reference(row["archive_record"])
    path = verified_reference(row["arrays"])
    require(
        path.resolve() == (manifest_path.parent / runtime["arrays"]).resolve(),
        "Runtime array path differs from its exact content proof",
    )
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "Repeated encoded array")
        require(
            array_inventory({name: saved[name] for name in saved.files})
            == row["array_inventory"],
            "Encoded arrays differ from original content proof",
        )


def materialize(
    plan_path, guide_path, pair_path, inventory, bindings, directory, protection_receipt
):
    require(not directory.exists(), "Output exists; preserve the original attempt")
    protection = protection_sets(protection_receipt)
    plan = read_json(plan_path)
    verify_plan(plan)
    guide = checked_report(guide_path, plan_path)
    pairs = read_json(pair_path)
    require(
        pairs["schema"] == 1
        and pairs["boundary"] == BOUNDARY
        and pairs["inputs"]["plan"] == reference(plan_path)
        and pairs["inputs"]["target_report"] == reference(guide_path),
        "Pair report scope or input changed",
    )
    for row in pairs["inputs"].values():
        verified_reference(row)
    inputs = source_inputs(plan, guide, pairs)
    protected = protected_sources(
        read_json(verified_reference(plan["inputs"]["sealed_identity_exclusions"]))
    )
    protected_hashes = sealed_raw_hashes(
        reviewed_scopes(inventory, bindings), protected
    )
    admission(directory.parent, 0)
    directory.mkdir()
    (directory / "arrays").mkdir()
    (directory / "records").mkdir()
    records = []
    for source in plan["sources"]:
        sid = source["source_id"]
        try:
            require_permitted(source, protection)
            row = materialize_source(
                source, inputs[sid], directory, protected, protected_hashes
            )
        except (ValueError, OSError, KeyError) as error:
            row = {
                "source_id": sid,
                "cohort": source["cohort"],
                "status": "INPUT_BINDING_FAILED",
                "error": type(error).__name__,
                "message": str(error),
            }
        save(directory / "records" / f"{sid}.json", row)
        records.append(row)
        print(f"Retained prospective input {sid}: {row['status']}", flush=True)
    report = {
        "schema": 1,
        "status": "UNFROZEN_PROSPECTIVE_EXECUTION_BINDINGS",
        "inputs": {
            "plan": reference(plan_path),
            "guide": reference(guide_path),
            "pairs": reference(pair_path),
            "reviewed_inventory": reference(inventory),
            "protected_inputs": protection_receipt,
        },
        "implementation": reference(Path(__file__)),
        "input_manifest": write_manifest(
            plan, records, directory, plan_path, protection_receipt
        ),
        "records": [
            reference(directory / "records" / f"{r['source_id']}.json") for r in records
        ],
        "counts": dict(Counter(r["status"] for r in records)),
        "capacity_after": admission(directory, 0),
        "runtime_policy_required": "DEVELOPMENT",
        "final_scoped_role_authorization": False,
        "boundary": BOUNDARY,
    }
    save(directory / "execution-bindings.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "guide",
        "pairs",
        "reviewed-inventory",
        "bindings",
        "directory",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--protected-inputs", type=Path, required=True)
    parser.add_argument("--protected-inputs-sha256", required=True)
    args = parser.parse_args()
    result = materialize(
        args.plan,
        args.guide,
        args.pairs,
        args.reviewed_inventory,
        args.bindings,
        args.directory,
        {"path": str(args.protected_inputs), "sha256": args.protected_inputs_sha256},
    )
    print(result["counts"])


if __name__ == "__main__":
    main()
