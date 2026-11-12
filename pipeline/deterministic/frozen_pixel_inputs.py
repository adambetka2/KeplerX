"""Freeze pixel content identities and retrieve those bytes within a cache bound."""

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.deterministic.control_pixel_inputs import (
    RESERVE,
    digest,
    header_record,
    read,
    receive_product,
    request_specification,
    validate_request,
)
from pipeline.runtime import EXTERNAL_DUMPS

DEFAULT_CACHE_BYTES = 8 * 1024**3
VERIFIED = "PUBLIC_BYTE_AND_HEADER_IDENTITIES_VERIFIED"


def verify_bindings(inputs):
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError(f"A frozen pixel ledger input changed: {path}")


def checked_record(path, expected, request):
    if digest(path) != expected:
        raise ValueError("A frozen pixel content record changed")
    record = read(path)
    if record["request"] != request or record["payload_retained"]:
        raise ValueError("The bounded pixel record request or retention state differs")
    if record["status"] == VERIFIED:
        transfer, header = record["transfer"], record["header"]
        if (
            request["status"] != "PUBLIC_PRODUCT_REQUEST"
            or transfer["bytes_received"] != request["size_bytes"]
            or not re.fullmatch(r"[0-9a-f]{64}", transfer["sha256"])
            or header["parent_kic"] != request["parent_kic"]
            or header["quarter"] != request["quarter"]
            or header["pixel_flux_values_read"]
        ):
            raise ValueError("The verified pixel byte/header identities differ")
        if request.get("expected_sha256") not in (None, transfer["sha256"]):
            raise ValueError("The pixel record differs from its predeclared content")
    elif record["status"] not in {
        "ORIGIN_PRODUCT_UNRESOLVED",
        "PUBLIC_PRODUCT_VERIFICATION_UNRESOLVED",
    }:
        raise ValueError("The pixel input record has an undeclared state")
    return record


def source_binding(source, request):
    if any(
        source[key] != request[key]
        for key in (
            "source_id",
            "role",
            "cohort",
            "origin_parent_kic",
            "raw_fits_sha256",
        )
    ):
        raise ValueError("A pixel request differs from the frozen source cohort")
    return {
        key: source[key]
        for key in (
            "source_id",
            "role",
            "cohort",
            "origin_parent_kic",
            "raw_fits_sha256",
        )
    }


def inventory_sources(folder, rows, records, source_inventory):
    sources = {}
    inventory = list(map(json.loads, source_inventory.read_text().splitlines()))
    declared = {row["source_id"]: row for row in inventory}
    if len(declared) != len(inventory):
        raise ValueError("The frozen source inventory repeats an identity")
    if len(records) != len(rows) or [r["request_index"] for r in records] != list(
        range(len(rows))
    ):
        raise ValueError("Every declared pixel request must be retained exactly once")
    for item, request in zip(records, rows):
        path = folder / "products" / f"product-{item['request_index']:06d}.json"
        record = checked_record(path, item["sha256"], request)
        if record["request_index"] != item["request_index"]:
            raise ValueError("The pixel content ordinal differs")
        sid, quarter = request["source_id"], str(request["quarter"])
        binding = source_binding(declared[sid], request)
        source = sources.setdefault(sid, {**binding, "quarters": {}})
        if quarter in source["quarters"]:
            raise ValueError("The frozen source repeats a requested pixel quarter")
        source["quarters"][quarter] = {
            "path": str(path.resolve()),
            "sha256": item["sha256"],
            "status": record["status"],
        }
    return sources


def freeze_inventory(folder, requests_path, source_inventory, output):
    if output.exists():
        raise FileExistsError("The frozen pixel content ledger already exists")
    audit, experiment = read(folder / "audit.json"), read(folder / "experiment.json")
    if (
        audit["status"] != "COMPLETE_BOUNDED_PUBLIC_PIXEL_INPUT_RECORDS"
        or audit["temporary_payloads_remaining"] != 0
        or audit["scientific_outcomes_opened"]
        or audit["boi_catalogue_enabled"]
        or any((folder / "spool").iterdir())
    ):
        raise ValueError("A completed input-only bounded pixel inventory is required")
    inputs = {
        str((folder / name).resolve()): audit[key]
        for name, key in (
            ("experiment.json", "experiment_sha256"),
            ("results.json", "results_sha256"),
        )
    }
    verify_bindings(inputs)
    verify_bindings(experiment["inputs"])
    rows, bindings = request_specification(requests_path)
    experiment_paths = {
        Path(path).resolve(): expected
        for path, expected in experiment["inputs"].items()
    }
    source_paths = {
        Path(path).resolve(): expected for path, expected in bindings.items()
    }
    if (
        experiment_paths.get(requests_path.resolve()) != digest(requests_path)
        or source_paths.get(source_inventory.resolve()) != digest(source_inventory)
        or len(rows) != audit["requests"]
    ):
        raise ValueError("The inventory source/request freeze bindings differ")
    sources = inventory_sources(
        folder, rows, read(folder / "results.json"), source_inventory
    )
    inputs.update(
        {
            str(path.resolve()): digest(path)
            for path in (folder / "audit.json", requests_path, source_inventory)
        }
    )
    verify_bindings(inputs)
    ledger = {
        "schema": 1,
        "status": "FROZEN_CONTROL_PIXEL_CONTENT_LEDGER",
        "inputs": inputs,
        "sources": sources,
        "source_quarters": len(rows),
        "scientific_outcomes_opened": False,
        "joint_independence_complete": False,
        "boi_catalogue_enabled": False,
        "boundary": (
            "Frozen content/header identity and explicit unresolved inputs. "
            "Byte retrieval does not assess PRF sensitivity, attribution, "
            "control independence or scientific release."
        ),
    }
    with output.open("x") as stream:
        stream.write(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    return ledger


def freeze_reconciled(folder, output):
    if output.exists():
        raise FileExistsError("The frozen pixel content ledger already exists")
    audit_path = folder / "audit.json"
    proof_path = folder / "independent-verification.json"
    quarters_path = folder / "quarters.jsonl"
    audit, proof = read(audit_path), read(proof_path)
    if (
        audit["status"] != "CURRENT_REQUIRED_REAL_PIXEL_RECEIPTS_RECONCILED"
        or proof["status"]
        != "INDEPENDENT_REQUIRED_PIXEL_RECEIPTS_AND_COMPONENTS_VERIFIED"
        or proof["ledger_audit_sha256"] != digest(audit_path)
        or proof["ledger_quarters_sha256"] != audit["quarter_ledger_sha256"]
        or proof["required_component_conflicts"]
        or audit["scientific_outcomes_opened"]
        or proof["scientific_outcomes_opened"]
    ):
        raise ValueError(
            "An independently verified required pixel receipt audit is required"
        )
    verify_bindings(audit["inputs"])
    inputs = {
        str(path.resolve()): digest(path)
        for path in (audit_path, proof_path, quarters_path)
    }
    sources = {}
    quarters = 0
    with quarters_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            sid, quarter = row["source_id"], str(row["quarter"])
            binding = {
                "source_id": sid,
                "role": row["role"],
                "cohort": row["cohort"],
                "origin_parent_kic": row["origin_parent_kic"],
                "raw_fits_sha256": row["raw_fits_sha256"],
            }
            source = sources.setdefault(sid, {**binding, "quarters": {}})
            if any(source[key] != value for key, value in binding.items()):
                raise ValueError(
                    "The reconciled source identity changes between quarters"
                )
            if quarter in source["quarters"]:
                raise ValueError("The reconciled source repeats a pixel quarter")
            source["quarters"][quarter] = row
            quarters += 1
    if (
        len(sources) != audit["required_real_sources"]
        or quarters != audit["required_real_source_quarters"]
        or quarters != proof["required_real_source_quarters"]
    ):
        raise ValueError("The reconciled source-quarter counts differ")
    verify_bindings(inputs)
    ledger = {
        "schema": 2,
        "status": "FROZEN_CONTROL_PIXEL_CONTENT_LEDGER",
        "inputs": inputs,
        "sources": sources,
        "source_quarters": quarters,
        "scientific_outcomes_opened": False,
        "joint_independence_complete": False,
        "boi_catalogue_enabled": False,
        "boundary": (
            "Independently audited public pixel receipts and explicit unavailable "
            "quarters. Retrieval rechecks source, parent, role transition, byte "
            "and header identities. This does not assess PRF sensitivity, "
            "attribution or scientific release."
        ),
    }
    with output.open("x") as stream:
        stream.write(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    return ledger


def unavailable_record(item, source, quarter):
    path = Path(item["nominal_path"])
    if digest(path) != item["nominal_sha256"]:
        raise ValueError("An unavailable pixel input receipt changed")
    nominal = read(path)
    request = nominal["request"]
    if nominal["status"] == VERIFIED and nominal["header"]["target_geometry"].get(
        "target_point_inside_pixel_footprint"
    ):
        raise ValueError("A frozen unavailable pixel quarter has public support")
    if any(request[key] != source[key] for key in source if key != "quarters"):
        raise ValueError("An unavailable pixel receipt has another source identity")
    return {
        "status": "FROZEN_PIXEL_INPUT_UNRESOLVED",
        "request": {**request, "quarter": quarter},
        "input_reason": item["reason"],
    }


def reconciled_record(item, source, quarter):
    if item["kind"] == "UNAVAILABLE":
        return unavailable_record(item, source, quarter)
    path = Path(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError("A reconciled pixel receipt changed")
    record = read(path)
    if item["kind"] == "SPATIAL":
        matches = [row for row in record["quarters"] if row["quarter"] == quarter]
        if len(matches) != 1:
            raise ValueError("The frozen spatial quarter differs")
        record = matches[0]["attempts"][item["attempt_index"]]
    request = record["request"]
    if item["kind"] == "REPLACEMENT_RESERVE_ROLE_BOUND":
        if (request["cohort"], request["role"]) != (
            "catalogue_eclipsing_binary_unassigned_reserve",
            "UNASSIGNED_RESERVE",
        ):
            raise ValueError("A frozen replacement lost its original reserve role")
        request = {
            **request,
            "cohort": source["cohort"],
            "role": source["role"],
            "status": "PUBLIC_PRODUCT_REQUEST",
        }
    elif item["kind"] not in {"NOMINAL", "SPATIAL"}:
        raise ValueError("The frozen pixel receipt has an undeclared selection")
    validate_request(request)
    source_binding(source, request)
    if (
        request["quarter"] != quarter
        or record["status"] != VERIFIED
        or record["payload_retained"]
        or request["parent_kic"] != item["actual_parent_kic"]
        or record["transfer"]["sha256"] != item["content_sha256"]
        or request.get("expected_sha256")
        not in (
            None,
            record["transfer"]["sha256"],
        )
        or record["transfer"]["bytes_received"] != request["size_bytes"]
        or record["header"]["parent_kic"] != item["actual_parent_kic"]
        or record["header"]["quarter"] != quarter
        or record["header"]["pixel_flux_values_read"]
        or not record["header"]["target_geometry"].get(
            "target_point_inside_pixel_footprint"
        )
    ):
        raise ValueError(
            "A frozen public pixel receipt differs from its selected input"
        )
    return {**record, "request": request}


class FrozenPixelInputs:
    def __init__(self, path, expected_sha256):
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or digest(path) != expected_sha256
        ):
            raise ValueError("The frozen pixel ledger SHA256 differs")
        self.ledger = read(path)
        if (
            self.ledger["schema"] not in {1, 2}
            or self.ledger["status"] != "FROZEN_CONTROL_PIXEL_CONTENT_LEDGER"
        ):
            raise ValueError("The pixel content ledger is not frozen")
        verify_bindings(self.ledger["inputs"])
        self.sha256 = expected_sha256

    def source(self, source_id, metadata, role):
        source = self.ledger["sources"][source_id]
        if (
            source["source_id"] != source_id
            or source["role"] != role
            or source["origin_parent_kic"] != metadata["origin_tpf_kic"]
            or source["raw_fits_sha256"] != metadata["raw_fits_sha256"]
        ):
            raise ValueError(
                "The frozen pixel source, role, parent or raw identity differs"
            )
        return source

    def record(self, source, quarter):
        item = source["quarters"][str(quarter)]
        if self.ledger["schema"] == 2:
            if any(item[key] != source[key] for key in source if key != "quarters"):
                raise ValueError("A reconciled pixel source-quarter identity differs")
            if item["quarter"] != quarter:
                raise ValueError("A reconciled pixel quarter identity differs")
            return reconciled_record(item, source, quarter)
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("A frozen pixel content record changed")
        row = read(item["path"])
        request = row["request"]
        validate_request(request)
        if (
            source_binding(source, request)
            != {k: v for k, v in source.items() if k != "quarters"}
            or request["quarter"] != quarter
            or row["status"] != item["status"]
        ):
            raise ValueError("The frozen pixel source-quarter request binding differs")
        return checked_record(item["path"], item["sha256"], request)


def cache_usage(cache):
    return sum(path.stat().st_size for path in cache.glob("*.fits.gz"))


def verified_payload(path, record):
    request = record["request"]
    if (
        path.is_symlink()
        or path.stat().st_size != request["size_bytes"]
        or digest(path) != record["transfer"]["sha256"]
    ):
        raise ValueError(
            "A cached frozen pixel payload differs from its content identity"
        )
    header = header_record(path, request)
    if header["header_sha256"] != record["header"]["header_sha256"]:
        raise ValueError("A frozen pixel payload header identity differs")


def cached_product(record, cache, network, reserve, maximum_bytes):
    if record["status"] != VERIFIED:
        return {
            "status": "FROZEN_PIXEL_INPUT_UNRESOLVED",
            "input_status": record["status"],
            **(
                {"input_reason": record["input_reason"]}
                if "input_reason" in record
                else {}
            ),
        }
    request = dict(record["request"], expected_sha256=record["transfer"]["sha256"])
    path = cache / f"{request['expected_sha256']}.fits.gz"
    with (cache / "cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache_usage(cache) > maximum_bytes:
            raise OSError("The frozen pixel cache exceeds its byte allowance")
        hit = path.exists() or path.is_symlink()
        if hit:
            verified_payload(path, record)
        elif not network:
            raise OSError(
                "Declared available frozen pixels are absent from the offline cache"
            )
        else:
            transfer_product(record, request, path, reserve, maximum_bytes)
    return {
        "status": "AVAILABLE",
        "path": str(path),
        "sha256": request["expected_sha256"],
        "base_parent_kic": request["parent_kic"],
        "transport_uri": request["transport_uri"],
        "cache_hit": hit,
        "downloaded_bytes": 0 if hit else request["size_bytes"],
    }


def transfer_product(record, request, path, reserve, maximum_bytes):
    if cache_usage(path.parent) + request["size_bytes"] > maximum_bytes:
        raise OSError(
            "The frozen pixel cache byte allowance is exhausted; use a smaller batch"
        )
    if shutil.disk_usage(path.parent).free < reserve + request["size_bytes"]:
        raise OSError("The frozen pixel transfer cannot preserve the disk reserve")
    temporary = path.with_suffix(".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("The frozen pixel temporary path is already occupied")
    try:
        receive_product(request, temporary, reserve=reserve)
        verified_payload(temporary, record)
        if shutil.disk_usage(path.parent).free < reserve:
            raise OSError("The frozen pixel transfer approached the disk reserve")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def retrieve_records(
    records, cache, network=True, reserve=RESERVE, maximum_bytes=DEFAULT_CACHE_BYTES
):
    if type(reserve) is not int or reserve <= 0:
        raise ValueError("The frozen pixel disk reserve must be a positive integer")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("The frozen pixel cache bound must be a positive integer")
    if reserve < RESERVE and (
        os.environ.get("SPACELAB_EXECUTION_PROFILE") != "cloud-development"
        or any(cache.resolve().is_relative_to(root.resolve()) for root in EXTERNAL_DUMPS)
        or any(record["request"]["role"] != "development" for record in records)
    ):
        raise ValueError(
            "Frozen pixel retrieval requires the original reserve unless an explicit "
            "cloud-development cache uses opened development inputs"
        )
    cache.mkdir(parents=True, exist_ok=True)
    if cache.is_symlink() or (cache / "cache.lock").is_symlink():
        raise ValueError(
            "The frozen pixel cache namespace must not redirect its writes"
        )
    return [
        cached_product(record, cache, network, reserve, maximum_bytes)
        for record in records
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--reconciled", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reconciled:
        if any((args.inventory, args.requests, args.source_inventory)):
            parser.error("A reconciled audit cannot be combined with an inventory")
        ledger = freeze_reconciled(args.reconciled, args.out)
    else:
        if not all((args.inventory, args.requests, args.source_inventory)):
            parser.error("An inventory, request plan and source inventory are required")
        ledger = freeze_inventory(
            args.inventory, args.requests, args.source_inventory, args.out
        )
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "sources": len(ledger["sources"]),
                "source_quarters": ledger["source_quarters"],
                "sha256": digest(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
