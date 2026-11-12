"""Prepare executable development source slices from reviewed fixed scopes."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.deterministic.control_bindings import (
    BOUNDARY,
    LAUNCH_SCOPES,
    digest,
    protected_sources,
    read_json,
    reference,
    require,
    unique_object,
    verified_reference,
)
from pipeline.deterministic.control_eligibility import save
from pipeline.deterministic.control_partitions import joint_component_index

INVENTORY_SHA = "49b22dc26f095b8a2305c057bc6dd78edaf90c42daa38e5bf0dff3b19abe4817"
CONTROL_SPEC_SHA = "7e796243c839cfd67c4981c8ffdbf086dc10138951b649ee5892b23638bb454a"


def reviewed_scopes(inventory_path, directory):
    require(
        digest(inventory_path) == INVENTORY_SHA, "Reviewed binding inventory changed"
    )
    inventory = read_json(inventory_path)["complete_original_report_member_inventory"]
    scopes = []
    for (scope, role), cohorts in LAUNCH_SCOPES.items():
        for cohort, count in cohorts.items():
            name = f"scopes/{scope.lower()}--{cohort}--{role}.json"
            expected = inventory[f"cloud-2nd-todo8/final/{name}"]
            path = verified_reference({"path": str(directory / name), **expected})
            value = read_json(path)
            require(
                value["scope"] == scope
                and value["role"] == role
                and value["cohort"] == cohort
                and len(value["trials"]) == count,
                "Reviewed scope role, cohort or denominator differs",
            )
            scopes.append((path, value))
    return scopes


def opened_identities(scopes, metadata):
    arrays = {
        row["source_id"]: row["cloud_arrays_sha256"]
        for cohort in metadata["opened_cohorts"].values()
        for row in cohort["rows"]
    }
    result = {}
    for _, scope in scopes:
        if scope["scope"] != "CURRENT_OPENED_DEVELOPMENT":
            continue
        for trial in scope["trials"]:
            sid = trial["source_id"]
            require(sid not in result, "Repeated opened source identity")
            result[sid] = {**trial, "original_cloud_arrays_sha256": arrays[sid]}
    return result


def component_exclusions(scopes):
    from pipeline.deterministic.cloud import check_scope_metadata

    specification = Path(__file__).with_name("cloud-joint-scope.json")
    spec = read_json(specification)
    artifact = spec["artifact"]
    packed = verified_reference(
        {**artifact, "path": str(specification.with_name(artifact["path"]))}
    )
    with gzip.open(packed, "rb") as stream:
        payload = stream.read(artifact["canonical_json_bytes"] + 1)
    require(
        len(payload) == artifact["canonical_json_bytes"]
        and hashlib.sha256(payload).hexdigest() == artifact["canonical_json_sha256"],
        "Current component graph decoded identity changed",
    )
    joint = json.loads(payload, object_pairs_hook=unique_object)
    check_scope_metadata(joint, spec)
    trials, _ = joint_component_index(joint)
    reviewed = [row for _, scope in scopes for row in scope["trials"]]
    require(
        len(reviewed) == len(trials)
        and {row["trial_id"] for row in reviewed} == trials.keys()
        and all(row == trials[row["trial_id"]] for row in reviewed),
        "Reviewed scopes differ from the current full component graph",
    )
    stopped = {
        row["full_witness_component_id"]
        for row in trials.values()
        if row["role"] in {"evaluation", "holdout"}
    }
    return stopped, {
        "specification": reference(specification),
        "packed_graph": reference(packed),
        "canonical_json_sha256": artifact["canonical_json_sha256"],
        "canonical_json_bytes": len(payload),
        "stopped_component_ids": sorted(stopped),
    }


def array_reference(manifest_path, row):
    relative = Path(row["arrays"])
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Input array path escapes its manifest",
    )
    path = verified_reference(
        {"path": str(manifest_path.parent / relative), "sha256": row["sha256"]}
    )
    return reference(path)


def control_array_proof(sid, original, arrays, specification):
    from pipeline.deterministic.cloud_public import verify_inventory

    require(digest(specification) == CONTROL_SPEC_SHA, "Original array oracle changed")
    oracle = read_json(specification)["sources"].get(sid)
    if oracle is None:
        return None
    require(
        oracle["opened_scope"] == "CURRENT_OPENED_DEVELOPMENT"
        and oracle["opened_cohort"] == original["cohort"]
        and oracle["metadata"]["source_id"] == sid
        and str(oracle["metadata"]["origin_tpf_kic"]) == original["origin_parent"]
        and oracle["sha256"] == original["raw_fits_sha256"],
        "Original array oracle source/parent/raw/cohort differs",
    )
    with np.load(arrays["path"], allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate array key")
        verify_inventory(
            {name: archive[name] for name in archive.files}, oracle["arrays"]
        )
    return {
        "status": "ALL_ORIGINAL_CONTROL_ARRAY_BYTES_VERIFIED",
        "execution_ready": True,
        "original_array_oracle": reference(specification),
        "array_count": len(oracle["arrays"]),
    }


def array_proof(manifest_path, sid, row, original, arrays):
    from pipeline.deterministic.cloud_public import fits_arrays

    if arrays["sha256"] == original["original_cloud_arrays_sha256"]:
        return {"status": "ORIGINAL_NPZ_BYTES_VERIFIED", "execution_ready": True}
    specification = Path(__file__).with_name("cloud-public-controls.json")
    proof = control_array_proof(sid, original, arrays, specification)
    if proof is not None:
        return proof
    fits_path = manifest_path.parent / f"{sid}.fits"
    if original["cohort"] != "ordinary" or not fits_path.is_file():
        return {
            "status": "ORIGINAL_ARRAY_CONTENT_PROOF_NOT_PROVISIONED",
            "execution_ready": False,
        }
    require(
        digest(fits_path) == original["raw_fits_sha256"],
        "Original raw reader input changed",
    )
    raw = fits_arrays(fits_path, sid, {"product_kic": row["metadata"]["kic"]})
    with np.load(arrays["path"], allow_pickle=False) as archive:
        require(set(archive.files) == set(raw), "Raw-reader array inventory differs")
        for name, expected in raw.items():
            actual = archive[name]
            require(
                actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and hashlib.sha256(actual.tobytes()).digest()
                == hashlib.sha256(expected.tobytes()).digest(),
                "Input array data differ from the original raw-reader witness",
            )
    return {
        "status": "ALL_RAW_READER_ARRAY_BYTES_VERIFIED",
        "execution_ready": True,
        "original_raw_fits": reference(fits_path),
        "array_count": len(raw),
    }


def bind_manifest(path, identities, protected, stopped_components):
    manifest = read_json(path)
    require(
        manifest["schema"] == 1 and manifest["role"] == "development",
        "Development input role changed",
    )
    require(
        manifest.get("measurement_scope") == "OPENED_DEVELOPMENT_ONLY",
        "Development input measurement scope changed",
    )
    require(
        isinstance(manifest["sources"], dict) and bool(manifest["sources"]),
        "Empty input manifest",
    )
    for sid in manifest["sources"]:
        require(sid not in protected, "Manifest includes protected sealed raw identity")
        if sid in identities:
            require(
                identities[sid]["full_witness_component_id"] not in stopped_components,
                "Manifest includes stopped/shared protected component",
            )
    bound, extras = {}, []
    for sid, row in manifest["sources"].items():
        metadata = row["metadata"]
        require(metadata["source_id"] == sid, "Manifest source key and metadata differ")
        arrays = array_reference(path, row)
        if sid not in identities:
            extras.append(
                {
                    "source_id": sid,
                    "arrays": arrays,
                    "status": "OUTSIDE_DECLARED_OPENED876_SCOPE",
                }
            )
            continue
        original = identities[sid]
        require(
            str(metadata["origin_tpf_kic"]) == original["origin_parent"]
            and metadata["raw_fits_sha256"] == original["raw_fits_sha256"],
            "Original source/parent/raw input binding differs",
        )
        if "opened_cohort" in row:
            require(
                row["opened_cohort"] == original["cohort"], "Manifest cohort changed"
            )
        bound[sid] = {
            "arrays": arrays,
            "input_manifest": reference(path),
            "trial_id": original["trial_id"],
            "cohort": original["cohort"],
            "array_proof": array_proof(path, sid, row, original, arrays),
        }
    return {
        "input_manifest": reference(path),
        "bound": bound,
        "unassigned_development_extras": extras,
    }


def scope_sources(scope, manifests, protected, stopped_components):
    rows = []
    for trial in scope["trials"]:
        sid = trial["source_id"]
        bindings = [m["bound"][sid] for m in manifests if sid in m["bound"]]
        status = "INPUTS_NOT_PROVISIONED"
        if scope["scope"] != "CURRENT_OPENED_DEVELOPMENT":
            status = "IDENTITY_ONLY_NOT_AUTHORIZED_FOR_THIS_DEVELOPMENT_EXECUTION"
            bindings = []
        elif trial["full_witness_component_id"] in stopped_components:
            status = "EXCLUDED_STOPPED_SHARED_COMPONENT"
            bindings = []
        elif sid in protected:
            status = "EXCLUDED_PROTECTED_RAW_IDENTITY"
            bindings = []
        elif bindings:
            status = "ORIGINAL_ARRAY_PROOF_PENDING"
            if any(b["array_proof"]["execution_ready"] for b in bindings):
                status = "HASH_VERIFIED_DEVELOPMENT_INPUTS"
        rows.append(
            {
                "trial_id": trial["trial_id"],
                "source_id": sid,
                "parent_tpf": trial["origin_parent"],
                "raw_fits_sha256": trial["raw_fits_sha256"],
                "full_witness_component_id": trial["full_witness_component_id"],
                "status": status,
                "input_variants": bindings,
            }
        )
    return rows


def execution_slice(scope, manifest, out, protected, stopped_components):
    require(
        scope["scope"] == "CURRENT_OPENED_DEVELOPMENT",
        "Execution slices require the original opened development scope",
    )
    ids = []
    for trial in scope["trials"]:
        sid = trial["source_id"]
        binding = manifest["bound"].get(sid)
        if binding is None:
            continue
        require(
            sid not in protected
            and trial["full_witness_component_id"] not in stopped_components,
            "Execution slice includes protected raw identity "
            "or stopped/shared component",
        )
        if binding["array_proof"]["execution_ready"]:
            ids.append(sid)
    if not ids:
        return None
    prefix = f"{scope['cohort']}--{manifest['input_manifest']['sha256'][:12]}"
    path = out / "development-source-ids" / f"{prefix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids, indent=2) + "\n")
    return {
        "source_ids": reference(path),
        "input_manifest": manifest["input_manifest"],
        "selected_sources": len(ids),
        "declared_cohort_sources": len(scope["trials"]),
        "measurement_scope": "BOUNDED_EXISTING_OPENED_DEVELOPMENT_SUBSET",
        "runtime_policy_required": "DEVELOPMENT",
        "final_scoped_role_authorization": False,
    }


def build(inventory_path, directory, manifests, out):
    require(not out.exists(), "Execution binding output exists; retain it")
    scopes = reviewed_scopes(inventory_path, directory)
    metadata_path = Path(__file__).with_name("cloud-control-metadata.json")
    metadata = read_json(metadata_path)
    identities = opened_identities(scopes, metadata)
    protected = protected_sources(metadata)
    stopped_components, graph = component_exclusions(scopes)
    bound = [
        bind_manifest(path, identities, protected, stopped_components)
        for path in manifests
    ]
    groups, slices = [], []
    for path, scope in scopes:
        rows = scope_sources(scope, bound, protected, stopped_components)
        groups.append(
            {
                "scope": scope["scope"],
                "role": scope["role"],
                "cohort": scope["cohort"],
                "reviewed_scope": reference(path),
                "sources": rows,
                "status_counts": dict(Counter(r["status"] for r in rows)),
            }
        )
        if scope["scope"] == "CURRENT_OPENED_DEVELOPMENT":
            slices.extend(
                slice_
                for m in bound
                if (
                    slice_ := execution_slice(
                        scope, m, out, protected, stopped_components
                    )
                )
            )
    result = {
        "schema": 1,
        "status": "CONTROL_EXECUTION_INPUT_BINDINGS_PREPARATION_ONLY",
        "reviewed_inventory": reference(inventory_path),
        "sealed_identity_exclusions": reference(metadata_path),
        "stopped_shared_component_exclusions": graph,
        "input_manifests": bound,
        "scopes": groups,
        "development_execution_slices": slices,
        "declared_trials": sum(len(s["sources"]) for s in groups),
        "opened_denominator": len(identities),
        "verified_opened_inputs": len({sid for m in bound for sid in m["bound"]}),
        "execution_ready_opened_inputs": sum(
            row["status"] == "HASH_VERIFIED_DEVELOPMENT_INPUTS"
            for scope in groups
            if scope["scope"] == "CURRENT_OPENED_DEVELOPMENT"
            for row in scope["sources"]
        ),
        "pending_scopes": [
            "two_10000_null_partitions",
            "real_10000_rehearsal",
            "7000_flux_injections",
            "declared_1000_periodic_stratified_subset",
            "unchanged_100_guide_and_100_contamination_physical_eligibility",
        ],
        "implementation": reference(Path(__file__)),
        "boundary": BOUNDARY,
    }
    save(out / "execution-input-bindings.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reviewed-inventory", "bindings-directory", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--input-manifest", action="append", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.reviewed_inventory, args.bindings_directory, args.input_manifest, args.out
    )
    print(
        {
            k: result[k]
            for k in ("declared_trials", "opened_denominator", "verified_opened_inputs")
        }
    )


if __name__ == "__main__":
    main()
