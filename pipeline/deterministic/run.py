"""Run the deterministic Rust search and shared vetting pipeline."""

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic.signal_contracts import (
    calibrated_lanes,
    exclusion_references,
    lane_decision,
    null_exclusion,
    null_manifest,
    relative_curve,
)
from evaluation_scope import authorize, match_cohort
from lc2 import (
    curve_digest,
    input_manifest,
    load_curve,
    load_source,
    photometry,
    source_catalogue,
)
from runtime import (
    EXTERNAL_DUMPS,
    dump_path,
    execution_workers,
    implementation_hashes,
    json_sha256,
    worker_count,
)
from shallow import run as shallow_detector

FOLDER = Path(__file__).resolve().parent
CONFIG = FOLDER / "config.json"
DISPOSITION_POLICY = Path("pipeline/vetting/disposition-policy.json")
ARCHIVE = {
    "repository": "spacelab-archive (private repository, unlisted)",
    "commit": "2850e1a9681311fb337e950408ca1e609332a381",
    "path": "projects/exoplanets/forgotten-stars/experiments/fsc-v1-survey-80k/archive/rust_classifier_v1",
    "tree": "04a43e93855851174d91aeccdcd92899e645cc15",
}
ARCHIVED_FROZEN = {
    "src/main.rs": "40a634d40c6a72060c07a21e4371e31458436dbf720bec8a64833adff57109fe",
    "Cargo.toml": "8eb4a4616b80945f845fc15e975fbf8623a3e741479962e7e214d52afb9dc9b3",
    "Cargo.lock": "40e22b168053d6e606c63928bccd939d0a6aacd07a32677f21fc5aafbbcbb946",
    "preregistration_v3_balanced.json": "7809ec69d1b41743e0e606725ee99de618a4307aa2c8aa930aa5114b1feaae9c",
}
DETECTOR_VIEWS = ("psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova")
DISCOVERY_VIEW = "psf_flat"
DISCOVERY_PROFILES = ("all-lanes-v1", "rust-transit-only-v1")
VIEW_CODES = {view: code for code, view in enumerate(DETECTOR_VIEWS)}
RECORD = np.dtype(
    [
        ("time", "<f8"),
        ("flux", "<f4"),
        ("error", "<f4"),
        ("quarter", "<i2"),
        ("cadence", "<i4"),
    ],
    align=False,
)


class InsufficientCadences(ValueError):
    def __init__(self, count):
        self.count = count
        super().__init__(
            f"Archived input preparation retained {count} cadences; Rust requires 100"
        )


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def detector_contract(discovery_view=DISCOVERY_VIEW, discovery_profile="all-lanes-v1"):
    if discovery_view not in {"psf_flat", "psf_qdiag", "sap_qdiag"}:
        raise ValueError("Unsupported discovery photometry")
    settings = search_configuration(discovery_profile)
    contract = {
        "origin": "rust_v6_shallow_v1",
        "contract_version": 5,
        "archive": ARCHIVE,
        "archived_files": ARCHIVED_FROZEN,
        "files": {
            name: digest(FOLDER / name)
            for name in (
                "src/main.rs",
                "Cargo.toml",
                "Cargo.lock",
                "config.json",
            )
        },
        "shallow_files": {
            name: digest(ROOT / "pipeline/shallow" / name)
            for name in ("run.py", "bls.c", "config.json", "requirements.txt")
        },
        "views": list(DETECTOR_VIEWS),
        "discovery_view": discovery_view,
        "preparation": "Each available view independently: finite time/flux/error, positive error, SAP_QUALITY=0; PSF_FLAT also FLATTEN_MASK=0; per-quarter median; upper-only 12 MAD sigma clipping; time/cadence sort and cadence deduplication; FSC_LC2 f32 flux/error",
        "protocol_version": read(CONFIG)["protocol_version"],
        "decision": "Rust and shallow BLS search the same prepared PSF_FLAT input. Their immutable periodic proposals and Rust sparse intervals enter one shared vetting queue with detector identities and diagnostics retained."
        if discovery_view == DISCOVERY_VIEW
        else f"Rust and shallow BLS search the same prepared {discovery_view} input. Photometric vetting uses this discovery view independently; correlated checks and immutable proposals remain retained.",
        "release": "DUAL_SEARCH; final all-gate survivors receive BOI designations",
    }

    if discovery_profile == "rust-transit-only-v1":
        if discovery_view == DISCOVERY_VIEW:
            raise ValueError("Transit-only continuation is restricted to PSF and SAP")
        contract.update(
            origin="rust_v6_deterministic",
            contract_version=6,
            discovery_profile=discovery_profile,
            search_config_sha256=json_sha256(settings),
            admission="Final Rust TRANSIT_LIKE only; at least three sampled transits",
            excluded_lanes=["SHALLOW", "SPARSE_EVENT", "SURVIVOR_REVIEW"],
            decision="Unchanged Rust periodic core, extensions and residual search. Only final TRANSIT_LIKE proposals enter independent-view vetting; all Rust diagnostic rows remain retained.",
            release="TRANSIT_ONLY; final all-gate survivors receive BOI designations",
        )
    return contract


def search_configuration(profile):
    if profile not in DISCOVERY_PROFILES:
        raise ValueError("Unknown discovery profile")
    config = read(CONFIG)
    if profile == "rust-transit-only-v1":
        config["sparse_event_snr_min"] = None
        config["sparse_event_min_points"] = 0
        config["max_sparse_events_per_target"] = 0
    return config


def bound_detector_contract(contract):
    profile = contract.get("discovery_profile", "all-lanes-v1")
    if profile == "all-lanes-v1":
        return detector_contract(contract["discovery_view"])
    return detector_contract(contract["discovery_view"], profile)


def args_detector_contract(args):
    return bound_detector_contract(
        {
            "discovery_view": getattr(args, "discovery_view", DISCOVERY_VIEW),
            "discovery_profile": getattr(args, "discovery_profile", "all-lanes-v1"),
        }
    )


def admitted_row(row, profile="all-lanes-v1"):
    if profile == "rust-transit-only-v1":
        if row["classification"] != "TRANSIT_LIKE":
            return False
        if int(row["n_transits"]) < 3:
            raise ValueError("Transit-only proposal has fewer than three transits")
        return True
    if profile != "all-lanes-v1":
        raise ValueError("Unknown discovery profile")
    return (
        row["classification"] == "EVENT_LIKE" or periodic_followup_lane(row) is not None
    )


def verify_profile_outputs(contract, proposals, survivors, shallow, eligibility):
    profile = contract.get("discovery_profile", "all-lanes-v1")
    if profile == "rust-transit-only-v1":
        if (
            survivors
            or shallow
            or not all(admitted_row(row, profile) for row in proposals)
        ):
            raise ValueError("Excluded discovery lane entered transit-only follow-up")
        if any(row["status"] != "NOT_RUN" for row in eligibility):
            raise ValueError("Transit-only profile executed the shallow detector")
    elif any(row["status"] == "NOT_RUN" for row in eligibility):
        raise ValueError("All-lane discovery omitted the shallow detector")


def scan_discovery_shallow(input_dir, output, source_ids, profile, **options):
    if profile == "all-lanes-v1":
        return scan_shallow(input_dir, output, source_ids, **options)
    if profile != "rust-transit-only-v1":
        raise ValueError("Unknown discovery profile")
    with Path(output).open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=shallow_detector.FIELDS).writeheader()
    with (
        Path(output)
        .with_name("shallow-eligibility.csv")
        .open("w", newline="") as stream
    ):
        writer = csv.DictWriter(stream, fieldnames=shallow_detector.ELIGIBILITY_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "source_id": sid,
                "status": "NOT_RUN",
                "reason": "DISABLED_BY_DISCOVERY_PROFILE",
                "elapsed_seconds": 0,
            }
            for sid in source_ids
        )
    return {str(sid): [] for sid in source_ids}


def verify_frozen_detector(args, binary):
    policy = read(ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY))
    if (
        policy.get("status") in {"FROZEN_EVALUATION", "RELEASED"}
        and digest(Path(binary)) != policy["evaluation_binding"]["detector_sha256"]
    ):
        raise ValueError(
            "Detector binary differs from the frozen disposition artifact"
        )


def prepare_curve_records(curve, mask_flatten=False, relative_input=False):
    _, arrays = relative_curve(curve, mask_flatten, relative_input)
    records = np.empty(len(arrays[0]), dtype=RECORD)
    for name, array in zip(RECORD.names, arrays):
        records[name] = array
    if len(records) < 100:
        raise InsufficientCadences(len(records))
    if (
        not np.isfinite(records["flux"]).all()
        or not np.isfinite(records["error"]).all()
        or not (records["error"] > 0).all()
    ):
        raise ValueError("FSC float32 conversion produced invalid photometry")
    return records


def prepare_records(raw, view="psf_flat", relative_input=False):
    return prepare_curve_records(
        photometry(raw, view), mask_flatten=view == "psf_flat", relative_input=relative_input
    )


def write_fsc(path, source_id, view, records):
    with Path(path).open("xb") as stream:
        stream.write(
            struct.pack(
                "<8sQQQ",
                b"FSC_LC2\0",
                int(source_id),
                VIEW_CODES[view],
                len(records),
            )
        )
        stream.write(records.tobytes())


def read_fsc(path, source_id, view):
    with Path(path).open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise ValueError("Truncated FSC header")
        magic, identity, view_code, count = struct.unpack("<8sQQQ", header)
        payload = stream.read()
    if (
        magic != b"FSC_LC2\0"
        or identity != int(source_id)
        or view_code != VIEW_CODES[view]
        or len(payload) != count * RECORD.itemsize
    ):
        raise ValueError("FSC identity, view or payload length mismatch")
    return np.frombuffer(payload, dtype=RECORD)


def views(raw, prepared=None):
    prepared = set(DETECTOR_VIEWS if prepared is None else prepared)
    return [
        {
            "view": view,
            "status": "SCORED" if view in prepared else "UNAVAILABLE",
            "input_arrays_sha256": curve_digest(photometry(raw, view)),
        }
        for view in DETECTOR_VIEWS
    ]


def flatten_digest(raw):
    array = np.asarray(raw["flatten_mask"])
    return hashlib.sha256(str(array.dtype).encode() + array.tobytes()).hexdigest()


def select_sources(
    shards,
    limit,
    source_ids=None,
    complete_pool=False,
    experimental_manifest=None,
):
    if limit < 1 or len(set(shards)) != len(shards):
        raise ValueError("Positive limit and distinct shards required")
    if experimental_manifest:
        _, specification = input_manifest(ROOT, experimental_manifest)
    else:
        specification = None
    if specification and specification["schema"] == 1:
        if source_ids is None:
            raise ValueError(
                "Standalone input manifests require explicit source identities"
            )
        available = {}
        for source_id, item in specification["sources"].items():
            metadata = dict(item["metadata"], source_id=str(source_id))
            if int(metadata["shard"]) in shards:
                available[str(source_id)] = metadata
    else:
        available = {
            row["source_id"]: row for row in source_catalogue(ROOT, sorted(shards))
        }
    ids = (
        sorted(available, key=int) if source_ids is None else list(map(str, source_ids))
    )
    if len(ids) != len(set(ids)) or any(sid not in available for sid in ids):
        raise ValueError(
            "Selected source identities must be distinct and present in the declared shards"
        )
    if len(ids) < limit or (source_ids is not None and len(ids) != limit):
        raise ValueError(
            "The selected cohort does not contain the requested source count"
        )
    return [available[sid] for sid in (ids if complete_pool else ids[:limit])]


def evaluation_preflight(
    args, sources, experimental_manifest=None, null_construction=None
):
    policy = read(ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY))
    if policy.get("status") not in {"FROZEN_EVALUATION", "RELEASED"}:
        return None
    sys.path.insert(0, str(ROOT / "pipeline/vetting"))
    vetting = importlib.import_module("vetting")
    vetting.validate_operational_policy(policy)
    config = read(ROOT / "pipeline/vetting/config.json")
    discovery_view = getattr(args, "discovery_view", DISCOVERY_VIEW)
    config.update(quality_bitmask=-1, discovery_contract=args_detector_contract(args))
    if discovery_view != DISCOVERY_VIEW:
        config["flatten_mask_views"] = ["psf_flat", "psf_nova"]
        config["quality_selection"] = (
            "Require SAP_QUALITY == 0. Apply FLATTEN_MASK only to PSF_FLAT "
            "and PSF_NOVA; full PSF and SAP retain their own quality-selected cadences."
        )
    if policy.get("implementation") != implementation_hashes() or policy.get(
        "pipeline_contract"
    ) != vetting.pipeline_contract(config):
        raise ValueError(
            "Frozen evaluation implementation or pipeline configuration changed"
        )
    if policy["status"] == "RELEASED":
        vetting.validate_release_contract(policy)
        return None
    role = authorize(policy, os.environ.get("SPACELAB_DISPOSITION_EVALUATION_TOKEN"))
    if null_construction is not None:
        return match_cohort(
            policy,
            role,
            sources,
            experimental_manifest,
            null_construction=null_construction,
        )
    pixel_inputs = frozen_pixel_configuration(args).get("frozen_pixel_inputs")
    if pixel_inputs is None:
        return match_cohort(policy, role, sources, experimental_manifest)
    return match_cohort(
        policy, role, sources, experimental_manifest, pixel_inputs=pixel_inputs
    )


def prepare_source(out, source, input_manifest=None, discovery_view=DISCOVERY_VIEW):
    sid = source["source_id"]
    metadata, raw = load_source(ROOT, sid, input_manifest)
    relative_input = null_manifest(ROOT, input_manifest) is not None
    inputs, unavailable = {}, {}
    for view in DETECTOR_VIEWS:
        path = out / "inputs" / f"{sid}.{view}.fsc"
        try:
            records = prepare_records(raw, view, relative_input)
        except (InsufficientCadences, KeyError) as exc:
            unavailable[view] = type(exc).__name__
            continue
        write_fsc(path, sid, view, records)
        if read_fsc(path, sid, view).tobytes() != records.tobytes():
            raise ValueError("FSC round-trip changed prepared records")
        inputs[view] = {
            "view": view,
            "retained_cadences": len(records),
            "fsc_path": str(path),
            "fsc_sha256": digest(path),
        }
    if discovery_view not in inputs:
        for item in inputs.values():
            Path(item["fsc_path"]).unlink()
        return {"status": "INELIGIBLE", "unavailable_views": len(DETECTOR_VIEWS)}
    return {
        "source_id": sid,
        "status": "PREPARED",
        "parent_tpf": str(metadata["origin_tpf_kic"]),
        "raw_fits_sha256": metadata["raw_fits_sha256"],
        "shard": metadata["shard"],
        "shard_path": metadata["shard_path"],
        "views": views(raw, inputs),
        "flatten_mask_sha256": flatten_digest(raw),
        "inputs": inputs,
        "unavailable_views": unavailable,
    }


def prepare(args):
    discovery_view = getattr(args, "discovery_view", DISCOVERY_VIEW)
    contract = args_detector_contract(args)
    input_manifest = getattr(args, "input_manifest", None)
    selected = select_sources(
        args.shards,
        args.limit,
        read(args.source_ids) if args.source_ids else None,
        complete_pool=args.source_ids is None,
        experimental_manifest=input_manifest,
    )
    evaluation = evaluation_preflight(args, selected, input_manifest)
    out = dump_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    manifest = {
        "schema": 2,
        "status": "PREPARING",
        "evaluation_scope": evaluation,
        "detector_contract": contract,
        "disposition_policy": str(
            getattr(args, "disposition_policy", DISPOSITION_POLICY)
        ),
        "disposition_policy_sha256": digest(
            ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY)
        ),
        "implementation": implementation_hashes(),
        "requested_sources": args.limit,
        "selection": {
            "shards": sorted(args.shards),
            "ordering": "explicit supplied identities"
            if args.source_ids
            else "numeric Gaia source ID",
            "source_ids": [],
            "eligibility": f"{discovery_view.upper()} has 100 prepared cadences. Attempt every explicit source identity; otherwise retain the first requested eligible count. Availability only, before detection.",
        },
        "novelty": "Prior processing or catalogue membership is not inferred from deterministic selection",
        "sources": [],
        "ineligible": [],
        "preparation_counts": {
            "attempted_sources": 0,
            "accepted_sources": 0,
            "ineligible_sources": 0,
            "prepared_views": 0,
            "unavailable_views": 0,
        },
    }
    if args.source_ids:
        manifest["selection"]["requested_source_ids"] = [
            source["source_id"] for source in selected
        ]
        manifest["selection"]["source_ids_file"] = {
            "path": str(args.source_ids.resolve()),
            "sha256": digest(args.source_ids),
        }
    if input_manifest:
        input_path = (ROOT / input_manifest).resolve()
        manifest["input_manifest"] = {
            "path": str(input_path),
            "sha256": digest(input_path),
        }
    save(out / "preparation.json", manifest)
    for source in selected:
        entry = (
            prepare_source(out, source, input_manifest)
            if discovery_view == DISCOVERY_VIEW
            else prepare_source(out, source, input_manifest, discovery_view)
        )
        counts = manifest["preparation_counts"]
        counts["attempted_sources"] += 1
        unavailable = entry["unavailable_views"]
        counts["unavailable_views"] += (
            unavailable if isinstance(unavailable, int) else len(unavailable)
        )
        if entry["status"] == "PREPARED":
            manifest["sources"].append(entry)
            manifest["selection"]["source_ids"].append(entry["source_id"])
            counts["accepted_sources"] += 1
            counts["prepared_views"] += len(entry["inputs"])
        else:
            counts["ineligible_sources"] += 1
            manifest["ineligible"].append(
                {
                    "source_id": source["source_id"],
                    "reason": "DISCOVERY_VIEW_UNAVAILABLE",
                    "unavailable_views": unavailable,
                }
            )
        if counts["attempted_sources"] % 100 == 0:
            save(out / "preparation.json", manifest)
        if len(manifest["sources"]) == args.limit:
            break
    complete = len(manifest["sources"]) == args.limit or (
        args.source_ids is not None
        and counts["attempted_sources"] == args.limit
        and counts["accepted_sources"] > 0
    )
    manifest["status"] = "PREPARED" if complete else "PREPARATION_FAILED"
    manifest["cohort_sha256"] = json_sha256(manifest["selection"]["source_ids"])
    save(out / "preparation.json", manifest)
    if manifest["status"] != "PREPARED":
        raise ValueError("Too few usable selected inputs")
    print(json.dumps(manifest["preparation_counts"]), flush=True)


def validate_preparation_accounting(preparation):
    accepted = preparation["selection"]["source_ids"]
    requested = preparation["selection"].get("requested_source_ids")
    if requested is None:
        if len(accepted) != preparation["requested_sources"]:
            raise ValueError("Preparation source count differs from requested quota")
        return
    rejected = [row["source_id"] for row in preparation["ineligible"]]
    counts = preparation["preparation_counts"]
    if (
        len(requested) != preparation["requested_sources"]
        or len(set(requested)) != len(requested)
        or len(set(accepted)) != len(accepted)
        or len(set(rejected)) != len(rejected)
        or set(accepted) & set(rejected)
        or set(accepted) | set(rejected) != set(requested)
        or counts["attempted_sources"] != len(requested)
        or counts["accepted_sources"] != len(accepted)
        or counts["ineligible_sources"] != len(rejected)
    ):
        raise ValueError("Explicit cohort preparation accounting is incomplete")
    reference = preparation["selection"]["source_ids_file"]
    if (
        digest(reference["path"]) != reference["sha256"]
        or read(reference["path"]) != requested
    ):
        raise ValueError("Explicit source selection changed after preparation")


def validate_preparation(out):
    preparation = read(out / "preparation.json")
    if preparation["status"] != "PREPARED" or preparation[
        "detector_contract"
    ] != bound_detector_contract(preparation["detector_contract"]):
        raise ValueError("Preparation is incomplete or detector contract changed")
    if preparation["implementation"] != implementation_hashes():
        raise ValueError("Implementation changed after preparation; use a fresh run")
    if (
        preparation.get("disposition_policy")
        and digest(ROOT / preparation["disposition_policy"])
        != preparation["disposition_policy_sha256"]
    ):
        raise ValueError("Disposition policy changed after preparation")
    experimental = preparation.get("input_manifest")
    if experimental and digest(experimental["path"]) != experimental["sha256"]:
        raise ValueError("Experimental input manifest changed after preparation")
    sources = preparation["sources"]
    validate_preparation_accounting(preparation)
    if [r["source_id"] for r in sources] != preparation["selection"][
        "source_ids"
    ] or preparation["cohort_sha256"] != json_sha256(
        preparation["selection"]["source_ids"]
    ):
        raise ValueError("Preparation source ledger differs from selected cohort")
    expected = {
        str(Path(item["fsc_path"]).resolve())
        for source in sources
        for item in source["inputs"].values()
    }
    if expected != {str(p.resolve()) for p in (out / "inputs").iterdir()}:
        raise ValueError("Detector input directory differs from frozen cohort")
    for source in sources:
        for item in source["inputs"].values():
            if digest(item["fsc_path"]) != item["fsc_sha256"]:
                raise ValueError(f"Prepared input changed: {source['source_id']}")
    return preparation


def parse_rows(path, source_views):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "source_id",
            "photometry",
            "signal_rank",
            "classification",
            "search_pass",
            "protocol_version",
            "period_days",
            "t0_bkjd",
            "duration_hours",
            "search_mes",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Rust output has missing or invalid columns")
        rows = list(reader)
    grouped = {key: [] for key in source_views}
    keys = set()
    for row in rows:
        sid, view, rank = (
            row["source_id"],
            row["photometry"],
            int(row["signal_rank"]),
        )
        key = (sid, view)
        if key not in grouped or (*key, rank) in keys:
            raise ValueError(
                "Rust output has unknown or duplicate source/view/rank identities"
            )
        if row["protocol_version"] != read(CONFIG)["protocol_version"]:
            raise ValueError("Rust output protocol differs from frozen configuration")
        if row["classification"] not in {
            "TRANSIT_LIKE",
            "EVENT_LIKE",
            "EB_LIKE",
            "SYSTEMATIC_LIKE",
            "LOW_SIGNIFICANCE",
            "NO_SIGNAL",
        }:
            raise ValueError("Rust output contains an unknown classification")
        periodic = row["classification"] not in {"EVENT_LIKE", "NO_SIGNAL"}
        if periodic != bool(row["search_mes"]):
            raise ValueError("Rust output has invalid periodic search MES")
        if periodic and not np.isfinite(float(row["search_mes"])):
            raise ValueError("Rust output has nonfinite periodic search MES")
        if periodic != bool(row["search_pass"]):
            raise ValueError("Rust output has invalid search pass")
        if periodic and int(row["search_pass"]) not in {1, 2}:
            raise ValueError("Rust output has unknown search pass")
        keys.add((*key, rank))
        grouped[key].append(row)
    if any(not values for values in grouped.values()):
        raise ValueError("Rust output omitted selected sources")
    return grouped


def scan_shallow_source(task):
    input_dir, sid, settings, *selected = task
    view = selected[0] if selected else DISCOVERY_VIEW
    path = Path(input_dir) / f"{sid}.{view}.fsc"
    records = read_fsc(path, sid, view)
    started = time.monotonic()
    try:
        rows = shallow_detector.scan(records, settings)
    except shallow_detector.ShallowIneligible as exc:
        rows, status, reason = [], "INELIGIBLE", exc.reason
    else:
        status, reason = "ELIGIBLE", ""
    return (
        sid,
        rows,
        {
            "source_id": sid,
            "status": status,
            "reason": reason,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        },
    )


def scan_shallow(
    input_dir,
    output,
    source_ids,
    config=shallow_detector.CONFIG,
    workers=1,
    discovery_view=DISCOVERY_VIEW,
):
    if workers < 1:
        raise ValueError("Shallow worker count must be positive")
    settings = read(config)
    grouped = {str(sid): [] for sid in source_ids}
    if not grouped:
        raise ValueError("Shallow scan requires at least one prepared source")
    shallow_detector.bls_library()
    eligibility_path = Path(output).with_name("shallow-eligibility.csv")
    with Path(output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=shallow_detector.FIELDS)
        writer.writeheader()
        with eligibility_path.open("w", newline="") as eligibility_stream:
            eligibility_writer = csv.DictWriter(
                eligibility_stream, fieldnames=shallow_detector.ELIGIBILITY_FIELDS
            )
            eligibility_writer.writeheader()
            tasks = ((input_dir, sid, settings, discovery_view) for sid in grouped)
            if workers == 1:
                results = map(scan_shallow_source, tasks)
                executor = None
            else:
                executor = ProcessPoolExecutor(max_workers=min(workers, len(grouped)))
                results = executor.map(scan_shallow_source, tasks)
            try:
                for sid, rows, eligibility in results:
                    writer.writerows({"source_id": sid, **row} for row in rows)
                    grouped[sid] = rows
                    eligibility_writer.writerow(eligibility)
            finally:
                if executor is not None:
                    executor.shutdown()
    return parse_shallow_rows(output, set(grouped))


def shallow_eligibility(path, source_ids):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != shallow_detector.ELIGIBILITY_FIELDS:
            raise ValueError("Shallow eligibility has invalid columns")
        rows = list(reader)
    if len(rows) != len(source_ids) or {row["source_id"] for row in rows} != set(
        source_ids
    ):
        raise ValueError("Shallow eligibility must cover each prepared source once")
    for row in rows:
        if row["status"] not in {"ELIGIBLE", "INELIGIBLE", "NOT_RUN"} or (
            row["status"] == "ELIGIBLE"
        ) != (row["reason"] == ""):
            raise ValueError("Shallow eligibility has invalid status or reason")
        if row["status"] == "NOT_RUN" and (
            row["reason"] != "DISABLED_BY_DISCOVERY_PROFILE"
            or float(row["elapsed_seconds"]) != 0
        ):
            raise ValueError("Disabled shallow scan has invalid scope evidence")
        elapsed = float(row["elapsed_seconds"])
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Shallow eligibility has invalid elapsed time")
    return rows


def parse_shallow_rows(path, source_ids):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != shallow_detector.FIELDS:
            raise ValueError("Shallow output has invalid columns")
        rows = list(reader)
    grouped = {str(sid): [] for sid in source_ids}
    identities = set()
    for row in rows:
        sid = row["source_id"]
        rank = int(row["search_rank"])
        if sid not in grouped or rank < 1 or (sid, rank) in identities:
            raise ValueError("Shallow output has unknown or duplicate source/rank")
        fields = (
            "period_days",
            "t0_bkjd",
            "duration_hours",
            "depth_ppm",
            "search_snr",
            "dip_inverse_snr_ratio",
            "harmonic_delta_log_likelihood",
            "odd_even_sigma",
            "positive_cycle_fraction",
        )
        inverse_snr = float(row["inverse_same_period_snr"])
        # BLS uses -inf when no positive-depth inverse model is supported.
        inverse_valid = np.isfinite(inverse_snr) or np.isneginf(inverse_snr)
        if not inverse_valid or not np.isfinite(
            [float(row[name]) for name in fields]
        ).all():
            raise ValueError("Shallow output has nonfinite diagnostics")
        if row["search_band"] not in {"baseline", "short", "long"} or not 0 <= float(
            row["positive_cycle_fraction"]
        ) <= 1:
            raise ValueError("Shallow output has invalid search band or cycle support")
        period, duration = float(row["period_days"]), float(row["duration_hours"]) / 24
        if period <= 0 or duration <= 0 or duration >= period:
            raise ValueError("Shallow output has invalid ephemeris")
        identities.add((sid, rank))
        grouped[sid].append(row)
    return grouped


def periodic_windows(records, cycle, inside):
    windows, counts = np.unique(
        np.column_stack((cycle[inside], records["quarter"][inside])),
        axis=0,
        return_counts=True,
    )
    return [
        (int(number), int(quarter), int(count))
        for (number, quarter), count in zip(windows, counts)
    ]


def rust_identity(row, sparse=False):
    prefix = "rust-v6-event" if sparse else "rust-v6"
    identity = f"{prefix}-{row['source_id']}-{row['signal_rank']}"
    if row["photometry"] != DISCOVERY_VIEW:
        identity += f"-{row['photometry']}"
    return identity


def supplied_events(row, records):
    if row["classification"] == "EVENT_LIKE":
        if row["period_days"] or not row["t0_bkjd"] or not row["duration_hours"]:
            raise ValueError("Rust event fields are invalid")
        epoch, hours = map(float, (row["t0_bkjd"], row["duration_hours"]))
        duration = hours / 24
        if not np.isfinite([epoch, duration]).all() or duration <= 0:
            raise ValueError("Rust emitted a nonfinite or invalid event")
        inside = abs(records["time"] - epoch) <= duration / 2
        quarters = sorted(set(records["quarter"][inside].tolist()))
        identity = rust_identity(row, sparse=True)
        alerts = [
            {
                "event_id": f"{identity}-q{quarter}",
                "kind": "SPARSE_EVENT_WINDOW",
                "photometry": row["photometry"],
                "quarter": quarter,
                "start_bkjd": epoch - duration / 2,
                "end_bkjd": epoch + duration / 2,
                "cycle": None,
                "detector_signal_id": identity,
                "detector_origin": "rust_v6_deterministic",
                "timing_origin": "SUPPLIED_RUST_EVENT",
                "in_window_detector_cadences": int(
                    (inside & (records["quarter"] == quarter)).sum()
                ),
            }
            for quarter in quarters
        ]
        return alerts, None
    values = [row[name] for name in ("period_days", "t0_bkjd", "duration_hours")]
    if not any(values):
        return [], None
    if not all(values):
        raise ValueError("Rust ephemeris fields are only partially populated")
    period, epoch, hours = map(float, values)
    duration = hours / 24
    if (
        not np.isfinite([period, epoch, duration]).all()
        or period <= 0
        or duration <= 0
        or duration >= period
    ):
        raise ValueError("Rust emitted a nonfinite or invalid ephemeris")
    identity = rust_identity(row)
    hypothesis = {
        "signal_id": identity,
        "hypothesis_id": identity,
        "period_days": period,
        "epoch_bkjd": epoch,
        "duration_days": duration,
        "alert_ids": [],
        "rust_record": row,
        "detector_origin": "rust_v6_deterministic",
        "detector_followup_lane": periodic_followup_lane(row),
        "unresolved_aliases": "All emitted ranks retained; ranks and harmonic alternatives are not independent planets",
    }
    if hypothesis["detector_followup_lane"] is None:
        return [], hypothesis
    view = row["photometry"]
    cycle = np.rint((records["time"] - epoch) / period).astype(np.int64)
    inside = abs(records["time"] - epoch - cycle * period) <= duration / 2
    alerts = []
    for number, quarter, count in periodic_windows(records, cycle, inside):
        centre = epoch + number * period
        alerts.append(
            {
                "event_id": f"{identity}-{view}-c{number}-q{quarter}",
                "kind": "PERIODIC_WINDOW",
                "photometry": view,
                "quarter": quarter,
                "start_bkjd": centre - duration / 2,
                "end_bkjd": centre + duration / 2,
                "cycle": number,
                "detector_signal_id": identity,
                "detector_origin": "rust_v6_deterministic",
                "timing_origin": "SUPPLIED_RUST_EPHEMERIS",
                "in_window_detector_cadences": int(count),
            }
        )
    hypothesis["alert_ids"] = [a["event_id"] for a in alerts]
    return alerts, hypothesis


def shallow_events(row, records, discovery_view=DISCOVERY_VIEW):
    sid, rank = row["source_id"], int(row["search_rank"])
    period = float(row["period_days"])
    epoch = float(row["t0_bkjd"])
    duration = float(row["duration_hours"]) / 24
    identity = f"shallow-v1-{sid}-{rank}"
    if discovery_view != DISCOVERY_VIEW:
        identity += f"-{discovery_view}"
    cycle = np.rint((records["time"] - epoch) / period).astype(np.int64)
    inside = abs(records["time"] - epoch - cycle * period) <= duration / 2
    alerts = []
    for number, quarter, count in periodic_windows(records, cycle, inside):
        centre = epoch + number * period
        alerts.append(
            {
                "event_id": f"{identity}-{discovery_view}-c{number}-q{quarter}",
                "kind": "PERIODIC_WINDOW",
                "photometry": discovery_view,
                "quarter": quarter,
                "start_bkjd": centre - duration / 2,
                "end_bkjd": centre + duration / 2,
                "cycle": number,
                "detector_signal_id": identity,
                "detector_origin": "shallow_bls_v1",
                "timing_origin": "SUPPLIED_SHALLOW_EPHEMERIS",
                "in_window_detector_cadences": int(count),
            }
        )
    if not alerts:
        raise ValueError(f"Shallow proposal has no observed cadence windows: {sid}")
    hypothesis = {
        "signal_id": identity,
        "hypothesis_id": identity,
        "period_days": period,
        "epoch_bkjd": epoch,
        "duration_days": duration,
        "alert_ids": [alert["event_id"] for alert in alerts],
        "shallow_record": row,
        "detector_origin": "shallow_bls_v1",
        "detector_followup_lane": "SHALLOW_BLS",
        "followup_eligible": True,
        "unresolved_aliases": "Detector periods and harmonics may describe the same source signal",
    }
    return alerts, hypothesis


def periodic_followup_lane(row):
    if row["classification"] == "TRANSIT_LIKE":
        return "PRIMARY_TRANSIT_LIKE"
    if row["classification"] == "SYSTEMATIC_LIKE" and (
        row.get("duration_consistency_status") == "FAIL"
        or row.get("quarter_loo_status") == "FAIL"
    ):
        return "SURVIVOR_REVIEW"
    return None


def select_followup_alerts(alerts, hypotheses, rows, gates, references, lane_ledger):
    decisions = []
    accepted = set()
    for identity in sorted({alert["detector_signal_id"] for alert in alerts}):
        emitted = [alert for alert in alerts if alert["detector_signal_id"] == identity]
        hypothesis = next((item for item in hypotheses if item["signal_id"] == identity), None)
        if hypothesis is not None:
            shallow = hypothesis["detector_origin"] == "shallow_bls_v1"
            lane = "shallow" if shallow else "periodic"
            row = hypothesis["shallow_record" if shallow else "rust_record"]
        else:
            lane = "sparse"
            row = next(item for item in rows if rust_identity(item, sparse=True) == identity)
        decision = lane_decision(identity, lane, row, gates)
        if references is not None:
            decision["null_exclusion"] = null_exclusion(emitted, references)
            if decision["null_exclusion"]["status"] == "EXCLUDED":
                decision["decision"] = "EXCLUDE"
        decisions.append(decision)
        if decision["decision"] == "PASS":
            accepted.add(identity)
        elif hypothesis is not None:
            hypothesis["followup_eligible"] = False
    selected = [alert for alert in alerts if alert["detector_signal_id"] in accepted]
    if lane_ledger is not None:
        lane_ledger.update(ungated_alerts=alerts, lane_decisions=decisions)
    return selected


def expand_source_rows(
    source,
    grouped,
    shallow_grouped,
    gates=None,
    references=None,
    lane_ledger=None,
    discovery_view=DISCOVERY_VIEW,
    discovery_profile="all-lanes-v1",
):
    sid, alerts, hypotheses, rows = source["source_id"], [], [], []
    prepared = {
        view: read_fsc(item["fsc_path"], sid, view)
        for view, item in source["inputs"].items()
    }
    for row in grouped[(sid, discovery_view)]:
        rows.append(row)
        if discovery_profile == "rust-transit-only-v1" and not admitted_row(
            row, discovery_profile
        ):
            continue
        events, hypothesis = supplied_events(row, prepared[discovery_view])
        if (
            row["classification"] == "EVENT_LIKE"
            or periodic_followup_lane(row) is not None
        ) and not events:
            raise ValueError(
                f"Rust proposal row has no observed cadence windows: {sid}"
            )
        alerts.extend(events)
        if hypothesis is not None:
            hypothesis["followup_eligible"] = periodic_followup_lane(row) is not None
            hypothesis["forced_photometry_views"] = sorted(prepared)
            hypotheses.append(hypothesis)
    rust_periodic = [alert for alert in alerts if alert["kind"] == "PERIODIC_WINDOW"]
    shallow_rows = shallow_grouped[sid]
    for row in shallow_rows:
        events, hypothesis = shallow_events(
            row, prepared[discovery_view], discovery_view
        )
        alerts.extend(events)
        hypothesis["forced_photometry_views"] = sorted(prepared)
        hypotheses.append(hypothesis)
    alerts = [
        alert
        for alert in alerts
        if alert["kind"] != "SPARSE_EVENT_WINDOW"
        or not any(
            alert["photometry"] == window["photometry"]
            and alert["quarter"] == window["quarter"]
            and alert["start_bkjd"] < window["end_bkjd"]
            and window["start_bkjd"] < alert["end_bkjd"]
            for window in rust_periodic
        )
    ]
    selected = select_followup_alerts(
        alerts, hypotheses, rows, gates, references, lane_ledger
    )
    return rows, shallow_rows, selected, hypotheses


def build_queue(out, preparation, grouped, shallow_grouped):
    discovery_view = preparation["detector_contract"]["discovery_view"]
    profile = preparation["detector_contract"].get("discovery_profile", "all-lanes-v1")
    queue, ledger = {"discovery_manifest": "discovery-manifest.json", "rows": []}, []
    artifacts = []
    policy = read(ROOT / preparation.get("disposition_policy", str(DISPOSITION_POLICY)))
    gates = calibrated_lanes(
        policy, policy.get("status") in {"FROZEN_EVALUATION", "RELEASED"}
    )
    experimental = preparation.get("input_manifest")
    specification = read(experimental["path"]) if experimental else {}
    eligibility = {
        row["source_id"]: row
        for row in shallow_eligibility(
            out / "shallow-eligibility.csv",
            [source["source_id"] for source in preparation["sources"]],
        )
    }
    verify_profile_outputs(
        preparation["detector_contract"],
        [row for values in grouped.values() for row in values if admitted_row(row, profile)],
        [],
        [row for values in shallow_grouped.values() for row in values],
        list(eligibility.values()),
    )
    for source in preparation["sources"]:
        sid = source["source_id"]
        experimental = preparation.get("input_manifest")
        _, raw = (
            load_source(ROOT, sid, experimental["path"])
            if experimental
            else load_source(ROOT, sid)
        )
        if (
            views(raw, source["inputs"]) != source["views"]
            or flatten_digest(raw) != source["flatten_mask_sha256"]
        ):
            raise ValueError(f"LC2 changed after input preparation: {sid}")
        references = (
            exclusion_references(specification, sid)
            if specification.get("null_construction")
            else None
        )
        lane_ledger = {}
        rows, shallow_rows, alerts, hypotheses = expand_source_rows(
            source,
            grouped,
            shallow_grouped,
            gates,
            references,
            lane_ledger,
            discovery_view,
            profile,
        )
        proposal = {
            "source_id": sid,
            "views": source["views"],
            "alerts": alerts,
            **lane_ledger,
            "supplied_hypotheses": hypotheses,
            "rust_rows": rows,
            "shallow_rows": shallow_rows,
            "shallow_eligibility": eligibility[sid],
        }
        path = out / "proposals" / f"{sid}.json"
        save(path, proposal)
        artifacts.append(path)
        if alerts:
            queue["rows"].append(
                {"source_id": sid, "evidence": str(path.relative_to(out))}
            )
        ledger.append(
            {
                "source_id": sid,
                "status": "PROPOSALS" if alerts else "DETECTOR_REJECTION",
                "raw_rows": len(rows),
                "shallow_rows": len(shallow_rows),
                "shallow_eligibility": eligibility[sid],
                "hypotheses": len(hypotheses),
                "classifications": dict(Counter(row["classification"] for row in rows)),
                "alerts": len(alerts),
                "evidence": str(path),
            }
        )
    save(out / "queue.json", queue)
    save(out / "source-ledger.json", ledger)
    return queue, ledger, artifacts


def run_command(command, log, environment=None):
    with Path(log).open("x") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Program exited {result.returncode}; inspect {log}")


def build_binary(out):
    target = out / "build"
    command = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        str(FOLDER / "Cargo.toml"),
        "--target-dir",
        str(target),
    ]
    environment = dict(os.environ)
    environment["RUSTFLAGS"] = (
        environment.get("RUSTFLAGS", "") + " -C target-cpu=native"
    ).strip()
    run_command(command, out / "build.log", environment)
    binary = target / "release" / "forgotten-stars-search"
    if not binary.is_file():
        raise RuntimeError("Cargo completed without producing the detector binary")
    return binary


def prepare_discovery_source(
    out, source, input_manifest=None, discovery_view=DISCOVERY_VIEW
):
    sid = source["source_id"]
    curve = load_curve(ROOT, sid, discovery_view, input_manifest)
    records = prepare_curve_records(
        curve,
        mask_flatten=discovery_view == DISCOVERY_VIEW,
        relative_input=null_manifest(ROOT, input_manifest) is not None,
    )
    path = out / "inputs" / f"{sid}.{discovery_view}.fsc"
    write_fsc(path, sid, discovery_view, records)
    if read_fsc(path, sid, discovery_view).tobytes() != records.tobytes():
        raise ValueError("FSC round-trip changed prepared records")
    provenance = curve["provenance"]
    flatten = np.asarray(curve["flatten"])
    return {
        "source_id": sid,
        "shard": source["shard"],
        "parent_tpf": provenance["parent_tpf"],
        "raw_fits_sha256": provenance["raw_fits_sha256"],
        "input_arrays_sha256": curve_digest(curve),
        "flatten_mask_sha256": hashlib.sha256(
            str(flatten.dtype).encode() + flatten.tobytes()
        ).hexdigest(),
        "retained_cadences": len(records),
        "fsc_sha256": digest(path),
    }


def run_survey_batch(
    out,
    binary,
    config,
    shallow_config,
    entries,
    threads,
    batch_number,
    discovery_view=DISCOVERY_VIEW,
    discovery_profile="all-lanes-v1",
):
    if discovery_profile != "all-lanes-v1" and read(config) != search_configuration(
        discovery_profile
    ):
        raise ValueError("Survey search configuration differs from its profile")
    raw = out / "raw.csv"
    command = [
        str(binary),
        "scan",
        "--config",
        str(config),
        "--input-dir",
        str(out / "inputs"),
        "--output",
        str(raw),
        "--photometry",
        discovery_view,
    ]
    started = time.monotonic()
    run_command(
        command,
        out / "rust.log",
        dict(os.environ, RAYON_NUM_THREADS=str(threads)),
    )
    source_views = {(entry["source_id"], discovery_view) for entry in entries}
    grouped = parse_rows(raw, source_views)
    view_options = (
        {} if discovery_view == DISCOVERY_VIEW else {"discovery_view": discovery_view}
    )
    shallow_grouped = scan_discovery_shallow(
        out / "inputs",
        out / "shallow-raw.csv",
        [entry["source_id"] for entry in entries],
        discovery_profile,
        config=shallow_config,
        workers=threads,
        **view_options,
    )
    eligibility = shallow_eligibility(
        out / "shallow-eligibility.csv", [entry["source_id"] for entry in entries]
    )
    rows = [row for values in grouped.values() for row in values]
    shallow_rows = [row for values in shallow_grouped.values() for row in values]
    rows.sort(
        key=lambda row: (
            int(row["source_id"]),
            row["photometry"],
            int(row["signal_rank"]),
        )
    )
    shallow_rows.sort(key=lambda row: (int(row["source_id"]), int(row["search_rank"])))
    return (
        {
            "batch": batch_number,
            "sources": len(entries),
            "rows": len(rows),
            "shallow_rows": len(shallow_rows),
            "shallow_ineligible_sources": sum(
                row["status"] == "INELIGIBLE" for row in eligibility
            ),
            "seconds": round(time.monotonic() - started, 3),
            "classifications": dict(Counter(row["classification"] for row in rows)),
        },
        rows,
        shallow_rows,
    )


SURVEY_TABLES = (
    "cohort.csv",
    "detections.csv",
    "ineligible.csv",
    "proposals.csv",
    "survivor-review.csv",
    "followup-diagnostics.csv",
    "shallow-proposals.csv",
    "shallow-eligibility.csv",
)
COHORT_FIELDS = (
    "source_id",
    "shard",
    "parent_tpf",
    "raw_fits_sha256",
    "input_arrays_sha256",
    "flatten_mask_sha256",
    "retained_cadences",
    "fsc_sha256",
)
INELIGIBLE_FIELDS = ("source_id", "shard", "retained_cadences", "reason")


def read_survey_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def append_survey_rows(path, fields, rows):
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as stream:
            if next(csv.reader(stream)) != list(fields):
                raise ValueError(f"Survey table header changed: {path.name}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    return path.stat().st_size


# survey.json is the commit record. Table bytes beyond its committed sizes
# belong to an interrupted batch; a resume drops them and reruns that batch.
def commit_survey(out, report, tables):
    for name, (fields, rows) in tables.items():
        report["committed_bytes"][name] = append_survey_rows(out / name, fields, rows)
    save(out / "survey.json", report, durable=True)


def restore_survey_tables(out, report):
    committed = report.get("committed_bytes")
    if not isinstance(committed, dict) or not set(committed) <= set(SURVEY_TABLES):
        raise ValueError("Survey report lacks its committed table sizes")
    discarded = {}
    for name in SURVEY_TABLES:
        size = (out / name).stat().st_size if (out / name).exists() else 0
        if size != committed.get(name, 0):
            discarded[name] = size - committed.get(name, 0)
    short = sorted(name for name, extra in discarded.items() if extra < 0)
    if short:
        raise ValueError(
            f"Survey tables are shorter than their committed sizes: {short}"
        )
    for name in discarded:
        os.truncate(out / name, committed.get(name, 0))
    return discarded


def survey_selection(args):
    source_ids = getattr(args, "source_ids", None)
    input_manifest = getattr(args, "input_manifest", None)
    discovery_view = getattr(args, "discovery_view", DISCOVERY_VIEW)
    selection = {
        "shards": sorted(args.shards),
        "ordering": "explicit supplied identities"
        if source_ids
        else "numeric Gaia source ID",
        "eligibility": "PSF_FLAT has at least 100 prepared cadences before detection"
        if discovery_view == DISCOVERY_VIEW
        else f"{discovery_view} has at least 100 prepared cadences before detection",
    }
    if source_ids:
        path = (ROOT / source_ids).resolve()
        selection["source_ids"] = {"path": str(path), "sha256": digest(path)}
        selection["execution_order"] = "shard then numeric Gaia source ID"
    if input_manifest:
        path = (ROOT / input_manifest).resolve()
        selection["input_manifest"] = {"path": str(path), "sha256": digest(path)}
    return selection


def ordered_survey_sources(sources, explicit):
    if not explicit:
        return sources
    return sorted(sources, key=lambda row: (int(row["shard"]), int(row["source_id"])))


def validate_survey_identity(out, args, report):
    policy = getattr(args, "disposition_policy", DISPOSITION_POLICY)
    if report.get("disposition_policy") != str(policy) or report.get(
        "disposition_policy_sha256"
    ) != digest(ROOT / policy):
        raise ValueError("Survey disposition policy identity changed or is missing")
    if report.get("schema") != 7 or report.get("status") not in {
        "RUNNING",
        "FAILED",
    }:
        raise ValueError("Survey is not a resumable incomplete dual-detector run")
    expected = {
        "detector_contract": args_detector_contract(args),
        "disposition_policy": str(
            getattr(args, "disposition_policy", DISPOSITION_POLICY)
        ),
        "disposition_policy_sha256": digest(
            ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY)
        ),
        "evaluation_scope": getattr(args, "evaluation_scope", None),
        "implementation": implementation_hashes(),
        "requested_sources": args.limit,
        "selection": survey_selection(args),
        "batch_size": args.batch_size,
        "threads": args.threads,
    }
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError("Survey resume arguments or implementation changed")
    binary = out / "detector"
    config = out / "detector-config.json"
    shallow_config = out / "shallow-config.json"
    files = {"binary": binary, "config": config, "shallow_config": shallow_config}
    valid_files = all(
        path.is_file()
        and report.get(name, {}).get("path") == str(path)
        and digest(path) == report.get(name, {}).get("sha256")
        for name, path in files.items()
    )
    if report["detector_contract"].get("discovery_profile") and read(
        config
    ) != search_configuration(report["detector_contract"]["discovery_profile"]):
        raise ValueError(
            "Frozen survey configuration differs from its discovery profile"
        )
    if not valid_files:
        raise ValueError("Frozen survey detector or config changed")
    return binary, config, shallow_config


def restore_survey_checkpoint(report):
    batches = report.get("batches", [])
    checkpoint = (
        (
            batches[-1].get("cumulative_attempted_sources"),
            batches[-1].get("cumulative_accepted_sources"),
            batches[-1].get("cumulative_ineligible_sources"),
        )
        if batches
        else (0, 0, 0)
    )
    if None in checkpoint:
        raise ValueError("Survey batches lack cumulative resume checkpoints")
    (
        report["attempted_sources"],
        report["accepted_sources"],
        report["ineligible_sources"],
    ) = checkpoint
    classifications = Counter()
    for batch in batches:
        classifications.update(batch["classifications"])
    return batches, classifications


def validate_survey_resume(out, args):
    report = read(out / "survey.json")
    binary, config, shallow_config = validate_survey_identity(out, args, report)
    discarded = restore_survey_tables(out, report)

    cohort_path = out / "cohort.csv"
    ineligible_path = out / "ineligible.csv"
    proposal_path = out / "proposals.csv"
    survivor_path = out / "survivor-review.csv"
    cohort = read_survey_rows(cohort_path) if cohort_path.exists() else []
    ineligible = read_survey_rows(ineligible_path) if ineligible_path.exists() else []
    proposals = read_survey_rows(proposal_path) if proposal_path.exists() else []
    survivors = read_survey_rows(survivor_path) if survivor_path.exists() else []
    shallow_path = out / "shallow-proposals.csv"
    shallow_rows = read_survey_rows(shallow_path) if shallow_path.exists() else []
    eligibility = (
        shallow_eligibility(
            out / "shallow-eligibility.csv", [row["source_id"] for row in cohort]
        )
        if cohort
        else []
    )
    proposal_sources = {row["source_id"] for row in proposals}
    survivor_sources = {row["source_id"] for row in survivors}
    shallow_sources = {row["source_id"] for row in shallow_rows}
    diagnostics_path = out / "followup-diagnostics.csv"
    if (
        report.get("batches")
        and "followup-diagnostics.csv" not in report["committed_bytes"]
    ):
        raise ValueError(
            "Survey lacks frozen follow-up diagnostics; a new survey is required"
        )
    diagnostics = (
        read_survey_rows(diagnostics_path) if diagnostics_path.exists() else []
    )
    diagnostic_sources = {row["source_id"] for row in diagnostics}
    queued_sources = proposal_sources | survivor_sources | shallow_sources
    profile = report["detector_contract"].get("discovery_profile", "all-lanes-v1")
    selected = [
        row
        for row in diagnostics
        if row["classification"] in {"TRANSIT_LIKE", "EVENT_LIKE"}
        and admitted_row(row, profile)
    ]
    reviews = [
        row
        for row in diagnostics
        if profile == "all-lanes-v1"
        and periodic_followup_lane(row) == "SURVIVOR_REVIEW"
    ]
    verify_profile_outputs(
        report["detector_contract"], proposals, survivors, shallow_rows, eligibility
    )
    batches, classifications = restore_survey_checkpoint(report)
    report.setdefault("resumes", []).append(
        {"committed_batches": len(batches), "discarded_bytes": discarded}
    )
    checks = (
        len(cohort) == report.get("accepted_sources"),
        len(ineligible) == report.get("ineligible_sources"),
        len({row["source_id"] for row in cohort}) == len(cohort),
        len({row["source_id"] for row in ineligible}) == len(ineligible),
        not {row["source_id"] for row in cohort}
        & {row["source_id"] for row in ineligible},
        sum(batch["sources"] for batch in batches) == len(cohort),
        sum(batch["rows"] for batch in batches) == sum(classifications.values()),
        dict(classifications) == report.get("classifications"),
        len(diagnostics) == report.get("followup_diagnostic_rows"),
        len(diagnostic_sources) == report.get("followup_diagnostic_sources"),
        diagnostic_sources == queued_sources,
        selected == proposals,
        reviews == survivors,
        len({(row["source_id"], row["signal_rank"]) for row in diagnostics})
        == len(diagnostics),
        sum(batch["followup_diagnostic_rows"] for batch in batches) == len(diagnostics),
        len(proposals) == report.get("proposal_rows"),
        len(proposal_sources) == report.get("proposal_sources"),
        len(survivors) == report.get("survivor_review_rows"),
        len(survivor_sources) == report.get("survivor_review_sources"),
        len(shallow_rows) == report.get("shallow_proposal_rows"),
        len(shallow_sources) == report.get("shallow_proposal_sources"),
        sum(batch.get("shallow_rows", 0) for batch in batches) == len(shallow_rows),
        sum(batch.get("shallow_ineligible_sources", 0) for batch in batches)
        == report.get("shallow_ineligible_sources")
        == sum(row["status"] == "INELIGIBLE" for row in eligibility),
        shallow_sources <= {row["source_id"] for row in cohort},
        len({(row["source_id"], row["search_rank"]) for row in shallow_rows})
        == len(shallow_rows),
        all(periodic_followup_lane(row) == "SURVIVOR_REVIEW" for row in survivors),
        report.get("attempted_sources")
        == report.get("accepted_sources") + report.get("ineligible_sources"),
    )
    if not all(checks):
        raise ValueError("Persisted survey counts do not reconcile")
    return (
        report,
        binary,
        config,
        shallow_config,
        classifications,
        proposal_sources,
        survivor_sources,
        shallow_sources,
    )


def start_survey(out, args):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": 7,
        "disposition_policy": str(
            getattr(args, "disposition_policy", DISPOSITION_POLICY)
        ),
        "disposition_policy_sha256": digest(
            ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY)
        ),
        "status": "BUILDING",
        "evaluation_scope": getattr(args, "evaluation_scope", None),
        "detector_contract": args_detector_contract(args),
        "implementation": implementation_hashes(),
        "requested_sources": args.limit,
        "selection": survey_selection(args),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "attempted_sources": 0,
        "accepted_sources": 0,
        "ineligible_sources": 0,
        "proposal_rows": 0,
        "proposal_sources": 0,
        "survivor_review_rows": 0,
        "survivor_review_sources": 0,
        "followup_diagnostic_rows": 0,
        "followup_diagnostic_sources": 0,
        "shallow_proposal_rows": 0,
        "shallow_proposal_sources": 0,
        "shallow_ineligible_sources": 0,
        "classifications": {},
        "batches": [],
        "committed_bytes": {},
    }
    save(out / "survey.json", report)
    binary = build_binary(out)
    frozen_binary = out / "detector"
    frozen_config = out / "detector-config.json"
    frozen_shallow_config = out / "shallow-config.json"
    shutil.copy2(binary, frozen_binary)
    profile = report["detector_contract"].get("discovery_profile", "all-lanes-v1")
    if profile == "all-lanes-v1":
        shutil.copy2(CONFIG, frozen_config)
    else:
        save(frozen_config, search_configuration(profile))
    shutil.copy2(shallow_detector.CONFIG, frozen_shallow_config)
    shutil.rmtree(out / "build")
    report["binary"] = {
        "path": str(frozen_binary),
        "sha256": digest(frozen_binary),
    }
    report["config"] = {
        "path": str(frozen_config),
        "sha256": digest(frozen_config),
    }
    report["shallow_config"] = {
        "path": str(frozen_shallow_config),
        "sha256": digest(frozen_shallow_config),
    }
    report["status"] = "RUNNING"
    save(out / "survey.json", report)
    return (
        report,
        frozen_binary,
        frozen_config,
        frozen_shallow_config,
        Counter(),
        set(),
        set(),
        set(),
        started,
    )


def resume_survey(out, args):
    (
        report,
        binary,
        config,
        shallow_config,
        classifications,
        proposal_sources,
        survivor_sources,
        shallow_sources,
    ) = validate_survey_resume(out, args)
    started = time.monotonic() - report.get("elapsed_seconds", 0.0)
    report.pop("error", None)
    report["status"] = "RUNNING"
    save(out / "survey.json", report, durable=True)
    return (
        report,
        binary,
        config,
        shallow_config,
        classifications,
        proposal_sources,
        survivor_sources,
        shallow_sources,
        started,
    )


def record_survey_batch(
    out,
    working,
    rows,
    shallow_rows,
    entries,
    batch,
    report,
    classifications,
    proposal_sources,
    survivor_sources,
    shallow_sources,
    ineligible_entries,
    started,
):
    with (working / "raw.csv").open(newline="") as stream:
        fields = csv.DictReader(stream).fieldnames
    if not fields:
        raise ValueError("Rust output has no CSV header")
    eligibility = shallow_eligibility(
        working / "shallow-eligibility.csv", [entry["source_id"] for entry in entries]
    )
    shallow_ineligible = sum(row["status"] == "INELIGIBLE" for row in eligibility)
    if batch["shallow_ineligible_sources"] != shallow_ineligible:
        raise ValueError("Shallow batch eligibility count differs from source ledger")
    profile = report["detector_contract"].get("discovery_profile", "all-lanes-v1")
    proposals = [
        row
        for row in rows
        if row["classification"] in {"TRANSIT_LIKE", "EVENT_LIKE"}
        and admitted_row(row, profile)
    ]
    survivors = [
        row
        for row in rows
        if profile == "all-lanes-v1"
        and periodic_followup_lane(row) == "SURVIVOR_REVIEW"
    ]
    verify_profile_outputs(
        report["detector_contract"], proposals, survivors, shallow_rows, eligibility
    )
    queued = {row["source_id"] for row in [*proposals, *survivors, *shallow_rows]}
    diagnostics = [row for row in rows if row["source_id"] in queued]
    batch["followup_diagnostic_rows"] = len(diagnostics)
    batch["followup_diagnostic_sources"] = len(queued)
    report["followup_diagnostic_rows"] += len(diagnostics)
    classifications.update(row["classification"] for row in rows)
    proposal_sources.update(row["source_id"] for row in proposals)
    survivor_sources.update(row["source_id"] for row in survivors)
    shallow_sources.update(row["source_id"] for row in shallow_rows)
    report["followup_diagnostic_sources"] = len(
        proposal_sources | survivor_sources | shallow_sources
    )
    report["proposal_rows"] += len(proposals)
    report["proposal_sources"] = len(proposal_sources)
    report["survivor_review_rows"] += len(survivors)
    report["survivor_review_sources"] = len(survivor_sources)
    report["shallow_proposal_rows"] += len(shallow_rows)
    report["shallow_proposal_sources"] = len(shallow_sources)
    report["shallow_ineligible_sources"] += shallow_ineligible
    report["classifications"] = dict(classifications)
    batch["cumulative_attempted_sources"] = report["attempted_sources"]
    batch["cumulative_accepted_sources"] = report["accepted_sources"]
    batch["cumulative_ineligible_sources"] = report["ineligible_sources"]
    report["batches"].append(batch)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    commit_survey(
        out,
        report,
        {
            "cohort.csv": (COHORT_FIELDS, entries),
            "detections.csv": (fields, rows),
            "ineligible.csv": (INELIGIBLE_FIELDS, ineligible_entries),
            "proposals.csv": (fields, proposals),
            "survivor-review.csv": (fields, survivors),
            "followup-diagnostics.csv": (fields, diagnostics),
            "shallow-proposals.csv": (shallow_detector.FIELDS, shallow_rows),
            "shallow-eligibility.csv": (
                shallow_detector.ELIGIBILITY_FIELDS,
                eligibility,
            ),
        },
    )


def survey(args):
    if args.limit < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("Positive source, batch and thread counts are required")
    source_ids = getattr(args, "source_ids", None)
    input_manifest = getattr(args, "input_manifest", None)
    discovery_view = getattr(args, "discovery_view", DISCOVERY_VIEW)
    available = select_sources(
        args.shards,
        args.limit,
        read(source_ids) if source_ids else None,
        complete_pool=source_ids is None,
        experimental_manifest=input_manifest,
    )
    args.evaluation_scope = evaluation_preflight(args, available, input_manifest)
    out = dump_path(args.run)
    if args.resume:
        (
            report,
            frozen_binary,
            frozen_config,
            frozen_shallow_config,
            classifications,
            proposal_sources,
            survivor_sources,
            shallow_sources,
            started,
        ) = resume_survey(out, args)
    else:
        (
            report,
            frozen_binary,
            frozen_config,
            frozen_shallow_config,
            classifications,
            proposal_sources,
            survivor_sources,
            shallow_sources,
            started,
        ) = start_survey(out, args)

    verify_frozen_detector(args, frozen_binary)
    available = ordered_survey_sources(available, source_ids is not None)
    complete_selection = len(available) == args.limit
    selected_sources = len(available)
    if report["attempted_sources"] > len(available):
        raise ValueError("Persisted survey attempt count exceeds the source pool")
    available = available[report["attempted_sources"] :]
    batch_entries = []
    ineligible_entries = []
    working = out / "working"
    if working.exists():
        shutil.rmtree(working)
    (working / "inputs").mkdir(parents=True)
    try:
        for index, source in enumerate(available):
            report["attempted_sources"] += 1
            try:
                entry = (
                    prepare_discovery_source(working, source, input_manifest)
                    if discovery_view == DISCOVERY_VIEW
                    else prepare_discovery_source(
                        working, source, input_manifest, discovery_view
                    )
                )
            except InsufficientCadences as exc:
                report["ineligible_sources"] += 1
                ineligible_entries.append(
                    {
                        "source_id": source["source_id"],
                        "shard": source["shard"],
                        "retained_cadences": exc.count,
                        "reason": "INSUFFICIENT_CADENCES",
                    }
                )
            else:
                batch_entries.append(entry)
                report["accepted_sources"] += 1
            if not batch_entries:
                continue
            if (
                len(batch_entries) < args.batch_size
                and report["accepted_sources"] < args.limit
                and index + 1 < len(available)
            ):
                continue

            number = len(report["batches"]) + 1
            view_options = (
                {}
                if discovery_view == DISCOVERY_VIEW
                else {"discovery_view": discovery_view}
            )
            if report["detector_contract"].get("discovery_profile"):
                view_options["discovery_profile"] = report["detector_contract"][
                    "discovery_profile"
                ]
            batch, rows, shallow_rows = run_survey_batch(
                working,
                frozen_binary,
                frozen_config,
                frozen_shallow_config,
                batch_entries,
                args.threads,
                number,
                **view_options,
            )
            record_survey_batch(
                out,
                working,
                rows,
                shallow_rows,
                batch_entries,
                batch,
                report,
                classifications,
                proposal_sources,
                survivor_sources,
                shallow_sources,
                ineligible_entries,
                started,
            )
            shutil.rmtree(working)
            batch_entries = []
            ineligible_entries = []
            if report["accepted_sources"] == args.limit:
                break
            (working / "inputs").mkdir(parents=True)
        if ineligible_entries and report["accepted_sources"] > 0:
            report["batches"].append(
                {
                    "batch": len(report["batches"]) + 1,
                    "sources": 0,
                    "rows": 0,
                    "followup_diagnostic_rows": 0,
                    "followup_diagnostic_sources": 0,
                    "shallow_rows": 0,
                    "shallow_ineligible_sources": 0,
                    "seconds": 0.0,
                    "classifications": {},
                    "cumulative_attempted_sources": report["attempted_sources"],
                    "cumulative_accepted_sources": report["accepted_sources"],
                    "cumulative_ineligible_sources": report["ineligible_sources"],
                }
            )
            commit_survey(
                out, report, {"ineligible.csv": (INELIGIBLE_FIELDS, ineligible_entries)}
            )
            ineligible_entries = []
        if complete_selection and report["attempted_sources"] != selected_sources:
            raise ValueError("Not every selected source was attempted")
        if not complete_selection and report["accepted_sources"] != args.limit:
            raise ValueError("Too few usable selected inputs")
        if ineligible_entries:
            raise ValueError("Ineligible identities remain outside a committed batch")
        if report["accepted_sources"] == 0:
            raise ValueError("No selected source had enough prepared cadences")
        report["status"] = "COMPLETE"
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["artifacts"] = {
            name: digest(out / name)
            for name in (
                *SURVEY_TABLES,
                "detector",
                "detector-config.json",
                "shallow-config.json",
            )
        }
        save(out / "survey.json", report, durable=True)
    except Exception as exc:
        report = read(out / "survey.json")
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(out / "survey.json", report, durable=True)
        raise
    print(json.dumps(report), flush=True)


def scan(args):
    out, binary = dump_path(args.run), args.binary.resolve()
    preparation = validate_preparation(out)
    if preparation.get("disposition_policy_sha256") != digest(
        ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY)
    ):
        raise ValueError("Detection policy differs from preparation")
    discovery_view = preparation["detector_contract"]["discovery_view"]
    args.discovery_view = discovery_view
    args.discovery_profile = preparation["detector_contract"].get(
        "discovery_profile", "all-lanes-v1"
    )
    experimental = preparation.get("input_manifest")
    evaluation = evaluation_preflight(
        args, preparation["sources"], experimental["path"] if experimental else None
    )
    if preparation.get("evaluation_scope") != evaluation:
        raise ValueError("Detection differs from its prepared evaluation scope")
    if args.threads < 1 or not binary.is_file():
        raise ValueError(
            "An existing Rust binary and positive thread count are required"
        )
    verify_frozen_detector(args, binary)
    if args.discovery_profile != "all-lanes-v1":
        save(out / "detector-config.json", search_configuration(args.discovery_profile))
    command = [
        str(binary),
        "scan",
        "--config",
        str(
            CONFIG
            if args.discovery_profile == "all-lanes-v1"
            else out / "detector-config.json"
        ),
        "--input-dir",
        str(out / "inputs"),
        "--output",
        str(out / "raw.csv"),
        "--photometry",
        discovery_view,
    ]
    report = {
        "status": "RUNNING",
        "command": command,
        "threads": args.threads,
        "binary": {"path": str(binary), "sha256": digest(binary)},
    }
    if (out / "detection.json").exists():
        raise ValueError("Detection already exists; use a fresh run")
    save(out / "detection.json", report)
    try:
        run_command(
            command,
            out / "rust.log",
            dict(os.environ, RAYON_NUM_THREADS=str(args.threads)),
        )
        source_views = {
            (source["source_id"], discovery_view)
            for source in preparation["sources"]
            if discovery_view in source["inputs"]
        }
        grouped = parse_rows(out / "raw.csv", source_views)
        shallow_grouped = scan_discovery_shallow(
            out / "inputs",
            out / "shallow-raw.csv",
            [source["source_id"] for source in preparation["sources"]],
            args.discovery_profile,
            workers=args.threads,
            discovery_view=discovery_view,
        )
        queue, ledger, artifacts = build_queue(
            out, preparation, grouped, shallow_grouped
        )
        artifacts += [
            out / name
            for name in (
                "raw.csv",
                "shallow-raw.csv",
                "shallow-eligibility.csv",
                "preparation.json",
                "queue.json",
                "source-ledger.json",
            )
        ]
        if args.discovery_profile != "all-lanes-v1":
            artifacts.append(out / "detector-config.json")
        manifest = {
            "schema": 2,
            "origin": bound_detector_contract(preparation["detector_contract"])[
                "origin"
            ],
            "detector_contract": bound_detector_contract(
                preparation["detector_contract"]
            ),
            "implementation": implementation_hashes(),
            "binary": report["binary"],
            "shallow_binary": shallow_detector.bls_library().build_manifest
            if args.discovery_profile == "all-lanes-v1"
            else None,
            "artifacts": {str(p): digest(p) for p in artifacts},
        }
        save(out / "discovery-manifest.json", manifest)
        report.update(
            status="DETECTION_COMPLETE",
            sources=len(ledger),
            alerted_sources=len(queue["rows"]),
            raw_rows=sum(r["raw_rows"] for r in ledger),
            shallow_rows=sum(r["shallow_rows"] for r in ledger),
            shallow_ineligible_sources=sum(
                row["shallow_eligibility"]["status"] == "INELIGIBLE" for row in ledger
            ),
            alerts=sum(r["alerts"] for r in ledger),
            classifications=dict(
                Counter(
                    row["classification"] for rows in grouped.values() for row in rows
                )
            ),
            view_classifications={
                view: dict(
                    Counter(
                        row["classification"]
                        for (source_id, row_view), rows in grouped.items()
                        if row_view == view
                        for row in rows
                    )
                )
                for view in DETECTOR_VIEWS
            },
        )
    except (OSError, ValueError, RuntimeError) as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save(out / "detection.json", report)
        raise
    save(out / "detection.json", report)
    print(json.dumps(report), flush=True)


def verify_discovery(queue_path, manifest, config):
    contract = bound_detector_contract(manifest["detector_contract"])
    if manifest.get("schema") != 2 or manifest.get("origin") != contract["origin"]:
        raise ValueError("Invalid Rust discovery manifest")
    if (
        manifest.get("detector_contract") != contract
        or config.get("discovery_contract") != contract
    ):
        raise ValueError("Rust discovery and vetting contracts differ")
    if manifest.get("implementation") != implementation_hashes():
        raise ValueError("Rust discovery implementation changed; use a fresh run")
    artifacts = manifest.get("artifacts", {})
    if not any(
        Path(path).resolve() == Path(queue_path).resolve() for path in artifacts
    ):
        raise ValueError("Rust discovery manifest does not bind this queue")
    for path, expected in artifacts.items():
        if digest(path) != expected:
            raise ValueError(f"Rust discovery artifact changed: {path}")
    binary = manifest["binary"]
    if digest(binary["path"]) != binary["sha256"]:
        raise ValueError("Frozen Rust binary changed")
    shallow = manifest.get("shallow_binary")
    if shallow is not None and digest(shallow["path"]) != shallow["sha256"]:
        raise ValueError("Frozen shallow BLS library changed")


def local_pixel_cache(path):
    cache = dump_path(ROOT / path)
    roots = (
        (ROOT / "dump").resolve(),
        *((root / "controls-20260926").resolve() for root in EXTERNAL_DUMPS),
    )
    if not any(cache.is_relative_to(root) for root in roots):
        raise ValueError(
            "Pixel downloads require a local cache or declared external control storage"
        )
    return cache


def verify_pixel_reserve(args, cache, reserve):
    from deterministic.control_pixel_inputs import RESERVE

    if type(reserve) is not int or reserve <= 0:
        raise ValueError("Pixel disk reserve must be a positive integer")
    if reserve >= RESERVE:
        return
    if reserve >= 2 * 1024**3 and any(
        cache.is_relative_to((root / "controls-20260926").resolve())
        for root in EXTERNAL_DUMPS
    ):
        return
    policy = read(ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY))
    if (
        os.environ.get("SPACELAB_EXECUTION_PROFILE") != "cloud-development"
        or any(cache.is_relative_to(root.resolve()) for root in EXTERNAL_DUMPS)
        or policy.get("status") not in {"DEVELOPMENT", "BOI_CATALOGUE"}
        or getattr(args, "pixel_input_role", None) not in {None, "development"}
    ):
        raise ValueError("A reduced pixel reserve requires opened cloud development storage")


def pixel_cache_configuration(args, config):
    path = (
        getattr(args, "pixel_cache_directory", None) or config["pixel_cache_directory"]
    )
    cache = local_pixel_cache(path)
    result = {"pixel_cache_directory": str(cache)}
    reserve = getattr(args, "pixel_minimum_free_bytes", None)
    if reserve is None:
        reserve = config.get("pixel_minimum_free_bytes")
    if reserve is not None:
        verify_pixel_reserve(args, cache, reserve)
        result["pixel_minimum_free_bytes"] = reserve
    return result


def frozen_pixel_configuration(args):
    from deterministic.frozen_pixel_inputs import DEFAULT_CACHE_BYTES, FrozenPixelInputs

    path = getattr(args, "frozen_pixel_inputs", None)
    role = getattr(args, "pixel_input_role", None)
    allowance = getattr(args, "pixel_cache_maximum_bytes", None)
    if path is None:
        if role is not None or allowance is not None:
            raise ValueError(
                "Frozen pixel role/cache arguments require a content ledger"
            )
        return {}
    if role not in {"development", "evaluation"}:
        raise ValueError(
            "A frozen pixel content ledger requires an explicit input role"
        )
    allowance = DEFAULT_CACHE_BYTES if allowance is None else allowance
    if type(allowance) is not int or allowance <= 0:
        raise ValueError("The frozen pixel cache byte allowance must be positive")
    path = (ROOT / path).resolve()
    checksum = digest(path)
    FrozenPixelInputs(path, checksum)
    return {
        "frozen_pixel_inputs": {
            "path": str(path), "sha256": checksum,
            "role": role, "maximum_cache_bytes": allowance,
        }
    }


def vetting_config_path(args):
    override = getattr(args, "vetting_config", None)
    if override is not None:
        policy = read(ROOT / getattr(args, "disposition_policy", DISPOSITION_POLICY))
        if policy.get("status") not in {"DEVELOPMENT", "BOI_CATALOGUE"}:
            raise ValueError("A vetting configuration override requires a development policy")
    return (ROOT / (override or "pipeline/vetting/config.json")).resolve()


def bind_vetting_preparation(out, args, policy_path, config):
    preparation_path = out / "preparation.json"
    if not preparation_path.exists():
        return
    preparation = read(preparation_path)
    if preparation.get("disposition_policy_sha256") != digest(ROOT / policy_path):
        raise ValueError("Vetting policy differs from preparation")
    experimental = preparation.get("input_manifest")
    evaluation = evaluation_preflight(
        args, preparation["sources"], experimental["path"] if experimental else None
    )
    if preparation.get("evaluation_scope") != evaluation:
        raise ValueError("Vetting differs from its prepared evaluation scope")
    config["evaluation_scope"] = evaluation
    if experimental:
        config["input_manifest"] = experimental["path"]


def vet(args):
    out = dump_path(args.run)
    detection = read(out / "detection.json")
    if detection["status"] != "DETECTION_COMPLETE":
        raise ValueError("Detection is incomplete")
    discovery = read(out / "discovery-manifest.json")
    discovery_view = discovery["detector_contract"]["discovery_view"]
    args.discovery_profile = discovery["detector_contract"].get(
        "discovery_profile", "all-lanes-v1"
    )
    args.discovery_view = discovery_view
    if args.workers < 1:
        raise ValueError("Vetting worker count must be positive")
    policy_path = getattr(args, "disposition_policy", DISPOSITION_POLICY)
    config = read(vetting_config_path(args))
    config.update(
        queue=str(out / "queue.json"),
        run_directory=str(out / "vetting"),
        workers=args.workers,
        network=not args.no_network,
        quality_bitmask=-1,
        quality_selection=read(ROOT / "pipeline/vetting/config.json")[
            "quality_selection"
        ],
        discovery_contract=bound_detector_contract(discovery["detector_contract"]),
        disposition_policy=str(policy_path),
    )
    config = execution_workers(config)
    if discovery_view != DISCOVERY_VIEW:
        config["flatten_mask_views"] = ["psf_flat", "psf_nova"]
        config["quality_selection"] = (
            "Require SAP_QUALITY == 0. Apply FLATTEN_MASK only to PSF_FLAT "
            "and PSF_NOVA; full PSF and SAP retain their own quality-selected cadences."
        )
    if getattr(args, "record_every_gate", False):
        config["triage_execution"] = "RECORD_EVERY_GATE"
    shared_index = getattr(args, "shared_alert_index", None)
    if shared_index is not None:
        shared_index = dump_path(shared_index)
        config["shared_alert_index"] = {
            "path": str(shared_index),
            "sha256": digest(shared_index),
        }
    else:
        config["shared_alert_scope_complete"] = True
    bind_vetting_preparation(out, args, policy_path, config)
    verify_discovery(out / "queue.json", read(out / "discovery-manifest.json"), config)
    config.update(pixel_cache_configuration(args, config))
    config.update(frozen_pixel_configuration(args))
    checkpoint = getattr(args, "pixel_checkpoint_ledger", None)
    if checkpoint is not None:
        checkpoint = dump_path(checkpoint)
        config["pixel_checkpoint_ledger"] = {
            "path": str(checkpoint),
            "sha256": digest(checkpoint),
        }
    if (out / "vetting.json").exists():
        raise ValueError(
            "Vetting already configured; invoke pipeline/run.py for a frozen-stage resume"
        )
    save(out / "vetting.json", config)
    environment = dict(os.environ, SPACELAB_VETTING_CONFIG=str(out / "vetting.json"))
    run_command(
        [str(args.python.absolute()), str(ROOT / "pipeline/run.py"), "vetting", "all"],
        out / "vetting.log",
        environment,
    )
    result = read(out / "vetting/results.json")
    stage_timings = read(out / "vetting/stage-timings.json")
    policy = read(ROOT / policy_path)
    if (
        result["sources"] != detection["alerted_sources"]
        or result["original_alerts_preserved"] != detection["alerts"]
    ):
        raise ValueError(
            "Detection-to-disposition source or alert reconciliation failed"
        )
    save(
        out / "result.json",
        {
            "status": result["processing_status"],
            "detection": detection,
            "boi_signals": result["boi_signals"],
            "unresolved_signals": result["unresolved_signals"],
            "vetting_results": str(out / "vetting/results.json"),
            "vetting_stage_timings": stage_timings,
            "disposition_policy": {
                "id": policy["policy_id"],
                "status": policy["status"],
            },
            "disposition_execution": result["disposition_execution"],
            "boi_catalogue_enabled": result["disposition_execution"][
                "boi_catalogue_enabled"
            ],
            "production_boi_enabled": result["disposition_execution"][
                "production_boi_enabled"
            ],
        },
    )


def run_all(args):
    prepare(
        argparse.Namespace(
            shards=args.shards,
            limit=args.limit,
            source_ids=args.source_ids,
            discovery_view=getattr(args, "discovery_view", DISCOVERY_VIEW),
            discovery_profile=getattr(args, "discovery_profile", "all-lanes-v1"),
            input_manifest=getattr(args, "input_manifest", None),
            disposition_policy=getattr(args, "disposition_policy", DISPOSITION_POLICY),
            frozen_pixel_inputs=getattr(args, "frozen_pixel_inputs", None),
            pixel_input_role=getattr(args, "pixel_input_role", None),
            pixel_cache_maximum_bytes=getattr(args, "pixel_cache_maximum_bytes", None),
            out=args.run,
        )
    )
    out = dump_path(args.run)
    binary = build_binary(out)
    scan(
        argparse.Namespace(
            run=args.run,
            binary=binary,
            threads=args.threads,
            disposition_policy=getattr(args, "disposition_policy", DISPOSITION_POLICY),
            frozen_pixel_inputs=getattr(args, "frozen_pixel_inputs", None),
            pixel_input_role=getattr(args, "pixel_input_role", None),
            pixel_cache_maximum_bytes=getattr(args, "pixel_cache_maximum_bytes", None),
        )
    )
    vet(
        argparse.Namespace(
            run=args.run,
            python=args.python,
            workers=args.workers,
            no_network=args.no_network,
            record_every_gate=getattr(args, "record_every_gate", False),
            vetting_config=getattr(args, "vetting_config", None),
            disposition_policy=getattr(args, "disposition_policy", DISPOSITION_POLICY),
            pixel_cache_directory=getattr(args, "pixel_cache_directory", None),
            pixel_minimum_free_bytes=getattr(args, "pixel_minimum_free_bytes", None),
            frozen_pixel_inputs=getattr(args, "frozen_pixel_inputs", None),
            pixel_input_role=getattr(args, "pixel_input_role", None),
            pixel_cache_maximum_bytes=getattr(args, "pixel_cache_maximum_bytes", None),
        )
    )
    print(json.dumps(read(out / "result.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--shards", type=int, nargs="+", required=True)
    prep.add_argument("--limit", type=int, required=True)
    prep.add_argument("--source-ids", type=Path)
    prep.add_argument("--input-manifest", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--disposition-policy", type=Path, default=DISPOSITION_POLICY)
    search = sub.add_parser("scan")
    search.add_argument("--run", type=Path, required=True)
    search.add_argument("--binary", type=Path, required=True)
    search.add_argument("--threads", type=int, default=worker_count())
    search.add_argument("--disposition-policy", type=Path, default=DISPOSITION_POLICY)
    followup = sub.add_parser("vet")
    followup.add_argument("--run", type=Path, required=True)
    followup.add_argument("--python", type=Path, required=True)
    followup.add_argument("--workers", type=int, default=worker_count())
    followup.add_argument("--no-network", action="store_true")
    followup.add_argument("--record-every-gate", action="store_true")
    followup.add_argument("--vetting-config", type=Path)
    followup.add_argument("--disposition-policy", type=Path, default=DISPOSITION_POLICY)
    followup.add_argument("--pixel-cache-directory", type=Path)
    followup.add_argument("--pixel-minimum-free-bytes", type=int)
    complete = sub.add_parser("all")
    complete.add_argument("--shards", type=int, nargs="+", required=True)
    complete.add_argument("--limit", type=int, required=True)
    complete.add_argument("--source-ids", type=Path)
    complete.add_argument("--input-manifest", type=Path)
    complete.add_argument("--run", type=Path, required=True)
    complete.add_argument("--threads", type=int, default=worker_count())
    complete.add_argument("--workers", type=int, default=worker_count())
    complete.add_argument("--python", type=Path, default=Path(sys.executable))
    complete.add_argument("--no-network", action="store_true")
    complete.add_argument("--record-every-gate", action="store_true")
    complete.add_argument("--vetting-config", type=Path)
    complete.add_argument("--disposition-policy", type=Path, default=DISPOSITION_POLICY)
    complete.add_argument("--pixel-cache-directory", type=Path)
    complete.add_argument("--pixel-minimum-free-bytes", type=int)
    proposals = sub.add_parser("survey")
    proposals.add_argument("--shards", type=int, nargs="+", required=True)
    proposals.add_argument("--limit", type=int, required=True)
    proposals.add_argument("--batch-size", type=int, default=5000)
    proposals.add_argument("--run", type=Path, required=True)
    proposals.add_argument("--threads", type=int, default=worker_count())
    proposals.add_argument("--source-ids", type=Path)
    proposals.add_argument("--input-manifest", type=Path)
    proposals.add_argument("--resume", action="store_true")
    proposals.add_argument(
        "--disposition-policy", type=Path, default=DISPOSITION_POLICY
    )
    for stage in (prep, search, followup, complete, proposals):
        stage.add_argument("--frozen-pixel-inputs", type=Path)
        stage.add_argument("--pixel-input-role", choices=("development", "evaluation"))
        stage.add_argument("--pixel-cache-maximum-bytes", type=int)
    for stage in (prep, complete, proposals):
        stage.add_argument(
            "--discovery-view",
            choices=("psf_flat", "psf_qdiag", "sap_qdiag"),
            default=DISCOVERY_VIEW,
        )
    for stage in (prep, complete, proposals):
        stage.add_argument(
            "--discovery-profile", choices=DISCOVERY_PROFILES, default="all-lanes-v1"
        )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "scan": scan,
        "vet": vet,
        "all": run_all,
        "survey": survey,
    }[args.stage](args)


if __name__ == "__main__":
    main()
