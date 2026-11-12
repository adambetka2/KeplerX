"""Bind actual prospective target/donor products without selecting replacements."""

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

import requests

from pipeline.deterministic.control_bindings import (
    protected_sources,
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_donor_headers import (
    bind_donor_header,
    donor_product,
    verify_targets,
)
from pipeline.deterministic.control_eligibility import (
    BOUNDARY,
    header_identity,
    mask_exposure,
    save,
    verify_plan,
)
from pipeline.deterministic.control_eligibility_archive import (
    local_headers,
    range_request,
    verify_headers,
)
from pipeline.deterministic.control_execution_inputs import reviewed_scopes
from pipeline.deterministic.control_physical_readiness import checked_report, donor_rows

MAXIMUM_RETAINED_BYTES = 4 * 1024**3
MINIMUM_FREE_BYTES = 2 * 1024**3
RETIREMENT_SPECIFICATION = Path(__file__).with_name(
    "native-protection-retirements.json"
)
RETIREMENT_SPECIFICATION_SHA256 = (
    "7da6ef0679652211169d53d328232f017234984939c963372bddf0200a7d8fbd"
)


def approved_retirement_guard(guard, receipt):
    specification = read_json(
        verified_reference(
            {
                "path": str(RETIREMENT_SPECIFICATION),
                "sha256": RETIREMENT_SPECIFICATION_SHA256,
            }
        )
    )
    require(
        receipt["sha256"] == specification["protection_guard_sha256"]
        and guard["status"]
        == "AUTHORIZED_RETIREMENT_FULL_COMPONENT_AND_INDEXED_PARENT_CLOSURE"
        and guard["authorization_scope"]
        == "ORIGINAL_NATIVE_PROSPECTIVE_DEVELOPMENT_ONLY"
        and guard["approval"]["sha256"] == specification["approval_sha256"]
        and guard["exposure_overlay"]["sha256"]
        == specification["exposure_overlay_sha256"]
        and guard["complete_development_exposure_before_access"]["sha256"]
        == specification["complete_development_exposure_sha256"]
        and guard["derivation"]["sha256"] == specification["derivation_sha256"]
        and guard["historical_null_exclusion_guard"]["sha256"]
        == specification["historical_protection_guard_sha256"]
        and guard["native_development_component_application"]["sha256"]
        == specification["native_component_application_sha256"]
        and guard["historical_source_seeds_covered_by_whole_component_approval"]
        == specification["historical_source_seeds_covered_by_whole_component_approval"]
        and guard["retired_trial_ids"] == specification["retired_trial_ids"]
        and guard["native_targets_reopened_by_authorization"]
        == specification["native_source_ids"]
        and guard["remaining_protected_native_source_ids"]
        == specification["remaining_protected_native_source_ids"]
        and guard["six_scene_link_witnesses_retained"] is True
        and guard["historical_null_exclusions_not_released"] is True
        and guard["original_native_trials_per_class"] == 100
        and guard["current_roles_launch_counts_and_caps_rewritten"] is False,
        "Unapproved prospective retirement or protection closure changed",
    )
    application_receipt = {
        **guard["native_development_component_application"],
        "path": str(
            Path(receipt["path"]).with_name("native-component-application.json")
        ),
    }
    application = read_json(verified_reference(application_receipt))
    require(
        application["status"]
        == "NATIVE_DEVELOPMENT_ONLY_APPROVED_WHOLE_COMPONENT_APPLICATION"
        and application["approval"]["sha256"] == specification["approval_sha256"]
        and application[
            "exact_historical_source_seeds_covered_by_whole_component_approval"
        ]
        == specification["historical_source_seeds_covered_by_whole_component_approval"]
        and application["all_covered_seeds_absent_from_current_sealed_list"] is True
        and application["native_protection_seed_application_only"] is True
        and application[
            "historical_exclusion_lists_or_null_guard_or_memberships_changed"
        ]
        is False
        and application["new_authorization_or_additional_retirements_claimed"] is False
        and application["scientific_arrays_or_outcomes_read"] is False,
        "Unapproved native component application or null exclusion release",
    )
    return set(specification["remaining_protected_native_source_ids"])


def protection_scene_targets(guard, receipt):
    if guard["schema"] == 2:
        return approved_retirement_guard(guard, receipt)
    require(
        guard["schema"] == 1
        and guard["status"]
        == "FULL_COMPONENT_AND_INDEXED_PARENT_CLOSURE_BEFORE_PIXEL_PROVISIONING"
        and guard["six_scene_links_remain_protected"] is True,
        "Full-component prospective protection receipt changed",
    )
    return {
        row["target_source_id"] for row in guard["potential_evaluation_scene_links"]
    }


def protection_sets(receipt):
    guard = read_json(verified_reference(receipt))
    protected_scene_targets = protection_scene_targets(guard, receipt)
    specification = read_json(Path(__file__).with_name("cloud-joint-scope.json"))
    require(
        guard["inputs"]["current_graph"]["sha256"]
        == specification["artifact"]["canonical_json_sha256"]
        and guard["five_retired_reservations_remain_historical_exclusions"] is True
        and guard["arrays_or_outcomes_read"] is False,
        "Full-component prospective protection receipt changed",
    )
    names = (
        "source_ids",
        "raw_or_pixel_content_hashes",
        "parent_identities",
        "additional_original_stopped_components_protected",
        "protected_full_component_ids",
    )
    result = {}
    for name in names:
        values = guard[name]
        require(
            isinstance(values, list)
            and all(isinstance(value, str) and value for value in values)
            and values == sorted(set(values)),
            "Protection identities are not ordered distinct strings",
        )
        result[name] = set(values)
    require(
        bool(result["protected_full_component_ids"])
        and len(guard["potential_evaluation_scene_links"]) == 6
        and len(
            {
                row["target_source_id"]
                for row in guard["potential_evaluation_scene_links"]
            }
        )
        == 6
        and protected_scene_targets <= result["source_ids"],
        "Protected scene targets absent from full-component closure",
    )
    return result


def require_permitted(source, protection):
    require(
        source["source_id"] not in protection["source_ids"]
        and source["pool"]["raw_fits_sha256"]
        not in protection["raw_or_pixel_content_hashes"]
        and source["parent_tpf"] not in protection["parent_identities"]
        and source["pool"]["metadata_dependency_component_id"]
        not in protection["additional_original_stopped_components_protected"],
        "Protected full-component prospective input",
    )


def capacity(root, additional=0):
    retained = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    free = shutil.disk_usage(root).free
    require(additional >= 0, "Invalid storage admission size")
    require(
        retained + additional <= MAXIMUM_RETAINED_BYTES,
        "Revised dispatch retained-data limit exhausted",
    )
    require(
        free - additional >= MINIMUM_FREE_BYTES, "Cloud disk reserve would be crossed"
    )
    return {"retained_dispatch_bytes": retained, "filesystem_free_bytes": free}


def sealed_raw_hashes(scopes, protected):
    return {
        row["raw_fits_sha256"]
        for _, scope in scopes
        for row in scope["trials"]
        if row["source_id"] in protected or scope["role"] in {"evaluation", "holdout"}
    }


def verify_product_bytes(payload, expected_sha, protected_hashes):
    identity = reference(payload)
    require(identity["sha256"] not in protected_hashes, "Protected sealed raw payload")
    if expected_sha is not None:
        require(identity["sha256"] == expected_sha, "Original target raw hash differs")
    return identity


def receive_product(
    session, source, total, path, expected_sha, root, protected_hashes, attempts
):
    require(
        not path.exists(), "Payload output exists; use its unchanged verified record"
    )
    capacity(root, total + 16 * 1024**2)
    data, observed = range_request(session, source, 0, total - 1, attempts)
    require(observed == total, "Canonical archive product size changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    identity = verify_product_bytes(path, expected_sha, protected_hashes)
    headers = local_headers(path)
    verify_headers(headers, total)
    return identity, headers


def preparation(source_id, product_kic, payload, catalogue=False):
    from pipeline.deterministic.catalogue_controls import read_product
    from pipeline.deterministic.cloud_public import fits_arrays

    if catalogue:
        metadata, raw = read_product(payload, product_kic)
        require(
            metadata["source_id"] == source_id, "Donor catalogue reader Gaia differs"
        )
    else:
        raw = fits_arrays(payload, source_id, {"product_kic": product_kic})
    exposure, _, _ = mask_exposure(raw)
    return exposure


def target_record(session, source, previous, directory, protected, protected_hashes):
    result = {
        "source": source,
        "prior_archive_record": previous["archive_record"],
        "status": "UNAVAILABLE",
        "attempts": [],
    }
    started = time.monotonic()
    try:
        require(
            source["source_id"] not in protected, "Protected sealed target identity"
        )
        prior = read_json(verified_reference(previous["archive_record"]))
        require(prior["source"] == source, "Previous target source changed")
        path = directory / "payloads" / f"target-{source['source_id']}.fits"
        payload, headers = receive_product(
            session,
            source,
            prior["archive_product_bytes"],
            path,
            source["pool"]["raw_fits_sha256"],
            directory.parent,
            protected_hashes,
            result["attempts"],
        )
        result.update(payload=payload, headers=headers)
        require(
            headers == prior["headers"],
            "Full target headers differ from original audit",
        )
        header_identity(source, headers)
        result["preparation"] = preparation(
            source["source_id"], source["product"]["kic"], path
        )
        result["status"] = "ORIGINAL_TARGET_RAW_BYTES_AND_PREPARATION_VERIFIED"
    except (ValueError, OSError, KeyError, requests.RequestException) as error:
        result.update(error=type(error).__name__, message=str(error))
    result["wall_seconds"] = time.monotonic() - started
    return result


def donor_record(session, binding, previous, directory, protected, protected_hashes):
    result = {
        "binding": binding,
        "prior_archive_record": previous["record"],
        "status": "UNAVAILABLE",
        "attempts": [],
    }
    started = time.monotonic()
    try:
        require(binding["source_id"] not in protected, "Protected sealed pair target")
        prior = read_json(verified_reference(previous["record"]))
        require(prior["binding"] == binding, "Previous donor source changed")
        if prior["status"] == "UNAVAILABLE":
            result["status"] = "CANONICAL_DONOR_UNAVAILABLE_IN_ORIGINAL_HEADER_AUDIT"
            return result
        sid = previous["donor"]["archive_gaia_source_id"]
        require(sid not in protected, "Protected sealed donor identity")
        path = directory / "payloads" / f"donor-{binding['source_id']}.fits"
        payload, headers = receive_product(
            session,
            donor_product(binding),
            prior["archive_product_bytes"],
            path,
            None,
            directory.parent,
            protected_hashes,
            result["attempts"],
        )
        result.update(payload=payload, headers=headers)
        require(
            headers == prior["headers"], "Full donor headers differ from original audit"
        )
        require(
            bind_donor_header(binding, headers) == previous["donor"],
            "Donor identity changed",
        )
        result["preparation"] = preparation(
            sid, int(binding["parent_tpf"]), path, catalogue=True
        )
        result.update(
            status="CANONICAL_DONOR_RAW_BYTES_AND_PREPARATION_BOUND",
            original_donor_raw_hash_provisioned=False,
            provenance="FIRST_FULL_CANONICAL_ARCHIVE_BINDING_FOR_UNFROZEN_PROSPECTIVE_PAIR",
        )
    except (ValueError, OSError, KeyError, requests.RequestException) as error:
        result.update(error=type(error).__name__, message=str(error))
    finally:
        result["wall_seconds"] = time.monotonic() - started
    return result


def collect(
    plan_path,
    target_path,
    donor_path,
    storage_path,
    directory,
    inventory,
    bindings_directory,
    protection_receipt,
):
    protection = protection_sets(protection_receipt)
    require(not directory.exists(), "Pair input output exists; retain the attempt")
    plan = read_json(plan_path)
    verify_plan(plan)
    targets = checked_report(target_path, plan_path)
    verify_targets(targets, plan, plan_path)
    donors = checked_report(donor_path, plan_path, target_path)
    donor_rows(plan, targets, donors)
    storage = read_json(storage_path)
    require(
        storage["maximum_dispatch_retained_bytes"] == MAXIMUM_RETAINED_BYTES
        and storage["minimum_filesystem_free_bytes"] == MINIMUM_FREE_BYTES
        and storage["raw_payload_git_upload"] is False
        and storage["source_reselection"] is False,
        "Revised storage scope or reserve changed",
    )
    directory.mkdir(parents=True)
    capacity(directory.parent)
    metadata = read_json(
        verified_reference(plan["inputs"]["sealed_identity_exclusions"])
    )
    protected = protected_sources(metadata) | protection["source_ids"]
    scopes = reviewed_scopes(inventory, bindings_directory)
    protected_hashes = (
        sealed_raw_hashes(scopes, protected) | protection["raw_or_pixel_content_hashes"]
    )
    target_index = {r["source_id"]: r for r in targets["sources"]}
    donor_index = {r["target_source_id"]: r for r in donors["donors"]}
    sources, counterpart_rows = [], []
    with requests.Session() as session:
        for source, binding in zip(
            [r for r in plan["sources"] if r["cohort"] == "contamination_pair"],
            plan["donor_bindings"],
            strict=True,
        ):
            require(source["source_id"] == binding["source_id"], "Pair order changed")
            sid = source["source_id"]
            try:
                require_permitted(source, protection)
            except ValueError:
                protected.add(sid)
            target = target_record(
                session,
                source,
                target_index[sid],
                directory,
                protected,
                protected_hashes,
            )
            donor = donor_record(
                session,
                binding,
                donor_index[sid],
                directory,
                protected,
                protected_hashes,
            )
            for kind, row in (("target", target), ("donor", donor)):
                path = directory / "records" / f"{kind}-{sid}.json"
                save(path, row)
                (sources if kind == "target" else counterpart_rows).append(
                    {
                        "source_id": sid,
                        "status": row["status"],
                        "record": reference(path),
                    }
                )
            print(f"Retained fixed target/donor payload attempt {sid}", flush=True)
    result = {
        "schema": 1,
        "status": "UNFROZEN_PAIR_INPUT_PROVENANCE_ONLY",
        "inputs": {
            "plan": reference(plan_path),
            "target_report": reference(target_path),
            "donor_report": reference(donor_path),
            "storage_plan": reference(storage_path),
            "reviewed_scope_inventory": reference(inventory),
            "protected_inputs": protection_receipt,
        },
        "targets": sources,
        "donors": counterpart_rows,
        "target_status_counts": dict(Counter(r["status"] for r in sources)),
        "donor_status_counts": dict(Counter(r["status"] for r in counterpart_rows)),
        "capacity_after": capacity(directory.parent),
        "implementation": reference(Path(__file__)),
        "boundary": BOUNDARY,
    }
    save(directory / "pair-input-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "plan",
        "target",
        "donor",
        "storage-plan",
        "directory",
        "reviewed-inventory",
        "bindings-directory",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--protected-inputs", type=Path, required=True)
    parser.add_argument("--protected-inputs-sha256", required=True)
    args = parser.parse_args()
    result = collect(
        args.plan,
        args.target,
        args.donor,
        args.storage_plan,
        args.directory,
        args.reviewed_inventory,
        args.bindings_directory,
        {"path": str(args.protected_inputs), "sha256": args.protected_inputs_sha256},
    )
    print(result["target_status_counts"], result["donor_status_counts"])


if __name__ == "__main__":
    main()
