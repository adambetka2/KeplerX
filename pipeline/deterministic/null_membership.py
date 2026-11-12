"""First outcome-blind null membership declaration; no scientific input access."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from pipeline.deterministic import cloud_null as pool

SELECTION_SEED = TRANSFORM_SEED = 20261002
QUOTA = 10000
ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "pipeline/deterministic"
PACKET_COMMIT = "c9cb77fa353c0518af6ce74d8a3c5dadc78e0ca8"
PACKET_PATH = "dump/cloud-return-holdout-complete-20261002"
MANIFEST_SHA = "defcf42a328c91ad1ee9f12193ad9397cab2f81aeeeade0485b87efb50827314"
GRAPH_SHA = "aff2e0baea45bd89dd4e9b36eb93e4e441a3310ad96c15f2f7cbe6cd7ce1049f"
ORIGINALS = {
    "cloud-null-pool.json": "83ac6c839f75d7fbdc50bc3027c9cffdf23c3cd804f5a57029f3ed7c57030064",
    "cloud-null-pool.csv.gz": "2b580f42c3fa2093f35639342186c62eee50b21b1ef7b512284a9a046570fa16",
    "cloud-joint-scope.json": "c305e1ffefdbd8d815c702b9543792d00660bbd362cdc5af781e9b18de988b4c",
    "cloud-joint-scope.json.gz": "04a51858ddd4b6d891bc97bed3749e8aaaebb2c8729e599814ba41cf913b8df0",
    "cloud-control-metadata.json": "ce675727ec3dc9f0aa92b2ea924534be98ec843a06cf2d53a749a5e367dc0119",
}
SET_FIELDS = {
    "source_ids": pool.decimal_identity,
    "parent_identities": pool.decimal_identity,
    "raw_or_pixel_content_hashes": pool.valid_hash,
    "protected_full_component_ids": pool.valid_hash,
    "additional_original_stopped_components_protected": pool.valid_hash,
}
RECIPES = {
    "id": "upper-clipped-quarter-nulls-v1",
    "seed": TRANSFORM_SEED,
    "shared_base": "Both variants reference the identical ordered role membership and original raw bindings.",
    "inverted": "Original finite/quality/flatten masks, per-quarter normalization, upper-only clip, then inversion.",
    "quarter_scrambled": "Rotate paired flux/errors within usable quarters by source/quarter/seed; require >2 usable discovery quarters.",
    "usable_quarters": "UNRESOLVED; no array access and no assumption that all bases supply both variants.",
    "program": "pipeline/deterministic/signal_contracts.py",
    "real_signal_exclusions": "Required hashed real/catalogue event references and proposal exclusions remain unmeasured.",
    "execution_authorized": False,
}


def reference(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": pool.digest(path)}


def verify_file(path, identity):
    pool.require(path.is_file() and not path.is_symlink(), f"Not a regular metadata file: {path}")
    pool.require(
        path.stat().st_size == identity["bytes"] and pool.digest(path) == identity["sha256"],
        f"Metadata bytes changed: {path}",
    )


def verified_packet(directory):
    manifest_path = directory / "manifest.json"
    pool.require(
        manifest_path.is_file()
        and not manifest_path.is_symlink()
        and pool.digest(manifest_path) == MANIFEST_SHA,
        "Wrong completed-holdout packet manifest",
    )
    manifest = json.loads(manifest_path.read_bytes())
    archive = directory / "reports.tar.gz"
    verify_file(archive, manifest["archive"])
    payloads = {}
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [item.name for item in members]
        pool.require(
            len(names) == len(set(names)) and set(names) == set(manifest["members"]),
            "Archive membership is not closed and unique",
        )
        for member in members:
            name = PurePosixPath(member.name)
            pool.require(
                member.isfile()
                and not name.is_absolute()
                and ".." not in name.parts
                and str(name) == member.name,
                "Unsafe archive member",
            )
            expected = manifest["members"][member.name]
            pool.require(member.size == expected["bytes"], "Archive member size changed")
            data = stream.extractfile(member).read()
            pool.require(
                hashlib.sha256(data).hexdigest() == expected["sha256"], "Archive member hash changed"
            )
            if member.name.startswith("reports/"):
                payloads[member.name] = data
    return manifest, payloads, [reference(manifest_path), reference(archive)]


def protection_sets(guard, aliases):
    result = {}
    for key, valid in SET_FIELDS.items():
        values = guard[key]
        pool.require(
            isinstance(values, list)
            and values == sorted(set(values))
            and all(valid(value) for value in values),
            f"Invalid protection set: {key}",
        )
        result[key] = set(values)
    original = aliases["aliases"]
    observed = guard["observed_public_global_parent_aliases"]
    pool.require(len(original) == len(observed) == 39, "Original conservative alias inventory changed")
    result["alias_parents"] = {row["witnessed_original_primary_global_parent"] for row in original}
    result["alias_parents"].update(row["observed_public_global_parent"] for row in observed)
    pool.require(
        result["alias_parents"] <= result["parent_identities"], "Conservative parent alias lacks protection"
    )
    for row in original:
        pool.require(
            set(row["potential_original_stopped_component_ids"])
            <= result["additional_original_stopped_components_protected"],
            "Alias component lacks protection",
        )
    for row in guard["potential_evaluation_scene_links"]:
        pool.require(
            row["target_source_id"] in result["source_ids"]
            and row["neighbor_identity"]["source_id"] in result["source_ids"],
            "Scene alias lacks protection",
        )
    return result


def load_inputs(directory):
    identities = [reference(STAGE / name) for name in ORIGINALS]
    pool.require(
        all(row["sha256"] == ORIGINALS[Path(row["path"]).name] for row in identities),
        "Original committed pool or graph inputs changed",
    )
    spec = json.loads((STAGE / "cloud-null-pool.json").read_bytes())
    bundle = STAGE / "cloud-null-pool.csv.gz"
    pool.check_specification(spec, bundle)
    payload = pool.decoded_bundle(bundle, spec["artifact"], "csv_bytes", "csv_sha256", 64 * 1024**2)
    scope_spec, scope_bundle = STAGE / "cloud-joint-scope.json", STAGE / "cloud-joint-scope.json.gz"
    scope = pool.linked_scope(spec, scope_spec, scope_bundle)
    pool.check_csv(payload, spec, scope)
    joint = json.loads(gzip.decompress(scope_bundle.read_bytes()))
    pool.require(
        hashlib.sha256(pool.canonical(joint)).hexdigest() == GRAPH_SHA
        and len(joint["trials"]) == 9007
        and len(joint["full_witness_components"]) == 8960
        and len(joint["holdout_overlap_witnesses"]) == 19,
        "Full original joint graph changed",
    )
    manifest, files, packet_refs = verified_packet(directory)
    guard = json.loads(files["reports/protected-input-sets-after-complete-headers.json"])
    aliases = json.loads(files["reports/original-parent-witness-reconciliation.json"])
    bindings = guard["inputs"]
    pool.require(
        bindings["current_graph"]["sha256"] == GRAPH_SHA
        and bindings["stopped_identity_pool"]["sha256"] == spec["artifact"]["csv_sha256"]
        and bindings["graph_specification"]["sha256"] == ORIGINALS["cloud-joint-scope.json"]
        and bindings["metadata_raw_exclusions"]["sha256"] == ORIGINALS["cloud-control-metadata.json"],
        "Protection packet refers to different original inputs",
    )
    protected = protection_sets(guard, aliases)
    rows = list(csv.DictReader(io.StringIO(payload.decode(), newline="")))
    pool.require(len(rows) == 122000, "Original scanned-half denominator changed")
    return rows, protected, manifest, identities + packet_refs


def selection_hash(row):
    identity = [SELECTION_SEED, row["metadata_dependency_component_id"], row["source_id"]]
    return hashlib.sha256(pool.canonical(identity)).hexdigest()


def exclusions(row, protected):
    reasons = []
    fields = {
        "source_id": "source_ids",
        "origin_parent": "parent_identities",
        "raw_fits_sha256": "raw_or_pixel_content_hashes",
        "metadata_dependency_component_id": "additional_original_stopped_components_protected",
    }
    for field, key in fields.items():
        if row[field] in protected[key]:
            reasons.append({"reason": key, "source_id": row["source_id"], "identity": row[field]})
    if row["origin_parent"] in protected["alias_parents"]:
        reasons.append(
            {
                "reason": "conservative_parent_alias",
                "source_id": row["source_id"],
                "identity": row["origin_parent"],
            }
        )
    for field in ("linked_existing_scopes", "linked_full_witness_component_ids"):
        for value in pool.ordered_links(row[field]):
            reasons.append({"reason": field, "source_id": row["source_id"], "identity": value})
    for value in pool.ordered_links(row["linked_full_witness_component_ids"]):
        if value in protected["protected_full_component_ids"]:
            reasons.append(
                {"reason": "protected_full_component_ids", "source_id": row["source_id"], "identity": value}
            )
    return reasons


def select(rows, protected, quota=QUOTA):
    pool.require(type(quota) is int and quota > 0, "Quota must be a positive integer")
    components = defaultdict(list)
    seen, owners = set(), {}
    for row in rows:
        pool.row_identity(row)
        source, component = row["source_id"], row["metadata_dependency_component_id"]
        pool.require(source not in seen, "Repeated pool source")
        seen.add(source)
        for kind in ("source_id", "origin_parent", "raw_fits_sha256"):
            pool.require(
                owners.setdefault((kind, row[kind]), component) == component,
                "Known identity split across metadata components",
            )
        components[component].append(row)
    decisions, candidates = [], []
    for component, members in sorted(components.items()):
        witnesses = [
            reason
            for row in sorted(members, key=lambda row: row["source_id"])
            for reason in exclusions(row, protected)
        ]
        best = min(members, key=lambda row: (selection_hash(row), row["source_id"]))
        decisions.append(
            {
                "component": component,
                "source_ids": sorted(row["source_id"] for row in members),
                "exclusions": witnesses,
                "representative": None if witnesses else best["source_id"],
            }
        )
        if not witnesses:
            candidates.append(best)
    candidates.sort(
        key=lambda row: (selection_hash(row), row["metadata_dependency_component_id"], row["source_id"])
    )
    pool.require(
        len(candidates) >= 2 * quota,
        f"Exact development/evaluation quotas refused: {len(candidates)} eligible components; require {2 * quota}",
    )
    membership = [
        {
            **row,
            "selection_position": index,
            "role_position": index // 2,
            "role": "development" if index % 2 == 0 else "evaluation",
            "selection_sha256": selection_hash(row),
        }
        for index, row in enumerate(candidates[: 2 * quota])
    ]
    return membership, decisions


def pool_ledger(rows, membership, decisions):
    selected = {row["source_id"]: row for row in membership}
    lookup = {row["component"]: row for row in decisions}
    for row in rows:
        decision = lookup[row["metadata_dependency_component_id"]]
        source = row["source_id"]
        if decision["exclusions"]:
            status = "EXCLUDED_COMPONENT"
        elif source in selected:
            status = "SELECTED"
        elif source != decision["representative"]:
            status = "NOT_COMPONENT_REPRESENTATIVE"
        else:
            status = "ELIGIBLE_COMPONENT_BEYOND_QUOTA"
        yield {
            **row,
            "status": status,
            "selection_sha256": selection_hash(row),
            "selected_role": selected.get(source, {}).get("role"),
        }


def write_gzip(path, rows):
    data = b"".join(pool.canonical(row) + b"\n" for row in rows)
    path.write_bytes(gzip.compress(data, mtime=0))
    return {**reference(path), "decoded_bytes": len(data), "decoded_sha256": hashlib.sha256(data).hexdigest()}


def write_membership(path, rows):
    rows = list(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=pool.FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())
    return {
        **reference(path),
        "sources": len(rows),
        "role": path.stem.removesuffix("-membership"),
        "ordered_source_ids_sha256": hashlib.sha256(
            pool.canonical([row["source_id"] for row in rows])
        ).hexdigest(),
    }


def implementation():
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT
    ).strip()
    pool.require(not tracked, "Commit the declaration implementation before freezing memberships")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    paths = [
        Path(__file__),
        STAGE / "cloud_null.py",
        STAGE / "signal_contracts.py",
        STAGE / "config.json",
        ROOT / "pipeline/shallow/config.json",
        ROOT / "pipeline/vetting/config.json",
    ]
    return {"commit": commit, "files": [reference(path) for path in paths]}


def declare(directory, output):
    code = implementation()
    rows, protected, manifest, inputs = load_inputs(directory)
    membership, decisions = select(rows, protected)
    for ref in inputs + code["files"]:
        verify_file(Path(ref["path"]), ref)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "pool-ledger.jsonl.gz": write_gzip(
            output / "pool-ledger.jsonl.gz", pool_ledger(rows, membership, decisions)
        ),
        "component-ledger.jsonl.gz": write_gzip(output / "component-ledger.jsonl.gz", decisions),
    }
    for role in ("development", "evaluation"):
        name = f"{role}-membership.csv"
        artifacts[name] = write_membership(output / name, (row for row in membership if row["role"] == role))
    result = {
        "schema": 1,
        "status": "FIRST_OUTCOME_BLIND_NULL_MEMBERSHIP_DECLARATION",
        "historical_membership_recovered": False,
        "selection_seed": SELECTION_SEED,
        "transform_seed": TRANSFORM_SEED,
        "selection": "SHA256 of canonical UTF-8 JSON [seed, component_id, source_id]; minimum per eligible component; global hash/component/source sort; first 20000; zero-based even positions development, odd evaluation.",
        "implementation": code,
        "inputs": inputs,
        "protection_packet": {
            "commit": PACKET_COMMIT,
            "path": PACKET_PATH,
            "closed_members": len(manifest["members"]),
            "manifest_sha256": MANIFEST_SHA,
            "archive": manifest["archive"],
        },
        "counts": {
            "original_pool_sources": len(rows),
            "original_pool_components": len(decisions),
            "excluded_components": sum(bool(row["exclusions"]) for row in decisions),
            "exclusion_witnesses_by_reason": dict(
                Counter(reason["reason"] for row in decisions for reason in row["exclusions"])
            ),
            "eligible_components": sum(not row["exclusions"] for row in decisions),
            "development": QUOTA,
            "evaluation": QUOTA,
            "original_graph_trials": 9007,
            "original_graph_components": 8960,
            "original_holdout_links": 19,
        },
        "artifacts": artifacts,
        "same_base_recipes": RECIPES,
        "original_stage_exposures": pool.EXPOSURE,
        "boundary": "Membership and seeds are first declared before selected flux or outcomes; existing memberships, exposures, graph, roles and guards are unchanged.",
        "unresolved": [
            "Actual quarter-parent, pixel and scene dependency witnesses and cross-role independence",
            "Original local base files and usable discovery-quarter counts",
            "Real/catalogue signal exclusions and measured per-vetted-proposal endpoints",
            "Final scoped code/configuration/policy authorization; no evaluation flux may be prepared here",
        ],
        "protected_opened_denominators": {
            "ordinary": 496,
            "native_eb": 100,
            "original_metadata_only_ordinary": 4,
            "original_metadata_only_native_eb": 1,
            "denominator_exception": False,
            "shared_reservation_history": "DEFERRED_FINAL; every old linked component remains excluded from these null memberships even if its deferred reservation is retired",
            "unexecuted_units_are_measured_outcomes": False,
        },
        "membership_and_seeds_frozen": True,
        "execution_authorized": False,
        "null_arrays_generated": False,
        "selected_flux_or_outcomes_read": False,
        "joint_independence_complete": False,
        "scientific_acceptance": False,
    }
    for ref in inputs + code["files"]:
        verify_file(Path(ref["path"]), ref)
    (output / "declaration.json").write_bytes(pool.canonical(result) + b"\n")
    for path in output.iterdir():
        path.chmod(0o444)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet", type=Path, required=True, help="Pinned manifest.json and reports.tar.gz directory"
    )
    parser.add_argument("--out", type=Path, required=True, help="New metadata-only declaration directory")
    args = parser.parse_args()
    result = declare(args.packet, args.out)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
