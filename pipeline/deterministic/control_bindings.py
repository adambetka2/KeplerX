"""Reconcile launch control identities without authorizing a scientific replay.

Inputs are hash-pinned metadata, including the existing prospective selection.
Every output remains a preparation artifact; missing inputs never become passes.
"""

import argparse
import csv
import hashlib
import io
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from pipeline.deterministic import cloud_null
from pipeline.deterministic.control_partitions import (
    bind_joint_components,
    digest,
    joint_component_index,
    select_opened_parents,
)

OPENED_COHORTS = {
    "ordinary": ("ordinary", 496),
    "planets": ("periodic_planet", 100),
    "native-eb": ("native_eb", 100),
    "sparse": ("sparse_injection", 40),
    "ttv": ("ttv_injection", 40),
    "background": ("background_contaminant", 100),
}
LAUNCH_SCOPES = {
    ("CURRENT_OPENED_DEVELOPMENT", "development"): dict(OPENED_COHORTS.values()),
    ("CURRENT_CORE_FINAL", "evaluation"): {
        "periodic_planet": 300,
        "sparse_injection": 160,
        "ttv_injection": 160,
        "native_eb": 450,
        "background_contaminant": 450,
    },
    ("DEFERRED_FINAL", "evaluation"): {"catalogue_fp": 500, "ordinary": 4500},
    ("DROPPED_CLASS_IDENTITY_PRESERVED", "development"): {
        "eclipsing_binary": 100,
        "instrumental_artifact": 100,
    },
    ("DROPPED_CLASS_IDENTITY_PRESERVED", "evaluation"): {
        "eclipsing_binary": 450,
        "instrumental_artifact": 450,
    },
    ("HOLDOUT_VETTING_DISPOSITION_ONLY", "holdout"): {"real_host_holdout": 511},
}
POOLS = (
    ("guide-star-opened-pool-current-sealed-safe.csv", "guide_star_family"),
    ("catalogued-parent-opened-pool-current-sealed-safe.csv", "contamination_pair"),
)
PARENT_INDEX_SHA256 = "9d5ca5f4e7dc34ed60312991d9bcb02f96bbeb255e775c0fdece9e9d276b483e"
DRAFT_SHA256 = "72cf8759cd01b4c93560f0157ea2e170b93c764010f58f3caf4ac77388ef1370"
PROGRAM_NAMES = (
    "deterministic/control_bindings.py",
    "deterministic/control_binding_inputs.py",
    "deterministic/control_partitions.py",
    "deterministic/cloud_null.py",
    "vetting/calibrate.py",
)
INPUT_NAMES = {
    "control_metadata",
    "joint_scope_specification",
    "joint_scope",
    "null_pool_specification",
    "null_pool",
    "parent_index",
    "calibration_bindings",
    "opened_truth_descriptor",
}
REPORT_NAMES = {
    "opened_source_bindings",
    "input_graph",
    "quarter_readiness",
    "holdout_links",
    "prospective_cohorts",
    "prospective_full_stopped_component_members",
    "null_pool_verified_counts",
    "calibration",
    "required_inputs",
    "unresolved_original_joint_scope",
    "observed_native_parent_supplement",
    "retained_prepared_inputs",
}
BOUNDARY = {
    "scientific_acceptance": False,
    "joint_independence_complete": False,
    "boi_catalogue_enabled": False,
    "scope_frozen": False,
    "prospective_cohorts_frozen": False,
    "null_partitions_selected": False,
    "seeds_frozen": False,
    "sealed_science_read": False,
    "original_local_inputs_reverified": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON identity: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)


def reference(path):
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def verified_reference(row):
    require(
        isinstance(row, dict)
        and {"path", "sha256"} <= row.keys()
        and isinstance(row["path"], str)
        and row["path"].strip()
        and cloud_null.valid_hash(row["sha256"]),
        "Invalid input reference",
    )
    path = Path(row["path"])
    require(
        path.is_file() and not path.is_symlink() and digest(path) == row["sha256"],
        f"Input unavailable or changed: {path}",
    )
    if "bytes" in row:
        require(
            type(row["bytes"]) is int and row["bytes"] == path.stat().st_size,
            f"Input size changed: {path}",
        )
    return path


def checked_inputs(path):
    recipe = read_json(path)
    require(
        type(recipe.get("schema")) is int
        and recipe["schema"] == 1
        and recipe.get("status") == "CONTROL_BINDING_PREPARATION_INPUTS"
        and set(recipe.get("inputs", {})) == INPUT_NAMES,
        "Binding preparation input schema or scope changed",
    )
    require(
        set(recipe) <= {"schema", "status", "inputs", "opened_preparations"},
        "Undeclared binding recipe fields or scope changes",
    )
    refs = recipe["inputs"]
    files = {name: verified_reference(row) for name, row in refs.items()}
    require(
        refs["parent_index"]["sha256"] == PARENT_INDEX_SHA256,
        "Production parent index pin changed",
    )
    return recipe, files


def positive_identity(value):
    return cloud_null.decimal_identity(value)


def check_trial(row):
    fields = {
        "trial_id",
        "scope",
        "role",
        "cohort",
        "source_domain",
        "source_id",
        "origin_parent",
        "raw_fits_sha256",
        "resolved_parents",
        "additional_parent_identities",
        "required_pixel_quarters",
        "pixel_quarters",
        "pixel_quarter_metadata_state",
        "stage_exposure",
    }
    require(
        isinstance(row, dict) and fields <= row.keys(),
        "Trial identity fields are missing",
    )
    require(
        positive_identity(row["source_id"])
        and positive_identity(row["origin_parent"])
        and cloud_null.valid_hash(row["raw_fits_sha256"])
        and row["source_domain"] in {"KBonus_LC1", "KBonus_LC2"}
        and row["trial_id"] == f"{row['cohort']}:{row['role']}:{row['source_id']}",
        "Trial source, raw input or declared identity differs",
    )
    for key in ("resolved_parents", "additional_parent_identities"):
        values = row[key]
        require(
            isinstance(values, list)
            and len(values) == len(set(values))
            and all(positive_identity(value) for value in values),
            "Invalid resolved parent identity",
        )
    require(
        isinstance(row["stage_exposure"], dict)
        and {"detector", "vetting", "disposition", "basis"}
        <= row["stage_exposure"].keys()
        and all(
            isinstance(row["stage_exposure"][key], str)
            and row["stage_exposure"][key].strip()
            for key in ("detector", "vetting", "disposition")
        ),
        "Stage exposure metadata is missing",
    )
    return quarter_readiness(row)


def quarter_readiness(row):
    required, quarters = row["required_pixel_quarters"], row["pixel_quarters"]
    require(
        isinstance(required, list)
        and len(required) == len(set(required))
        and all(type(q) is int and 0 <= q <= 17 for q in required)
        and isinstance(quarters, list),
        "Invalid required quarter inventory",
    )
    seen, unresolved = set(), []
    for quarter in quarters:
        require(
            isinstance(quarter, dict)
            and type(quarter.get("quarter")) is int
            and quarter["quarter"] in required
            and quarter["quarter"] not in seen,
            "Duplicate or undeclared quarter identity",
        )
        seen.add(quarter["quarter"])
        parent = quarter.get("actual_parent_kic")
        if parent is None:
            unresolved.append(quarter["quarter"])
            continue
        require(
            type(parent) is int
            and parent > 0
            and str(parent)
            in {
                row["origin_parent"],
                *row["resolved_parents"],
                *row["additional_parent_identities"],
            },
            "Quarter parent is absent from declared dependency identities",
        )
        if not all(
            cloud_null.valid_hash(quarter.get(key))
            for key in ("content_sha256", "receipt_sha256")
        ):
            unresolved.append(quarter["quarter"])
    unresolved = sorted(set(required) - seen | set(unresolved))
    return {
        "declared_state": row["pixel_quarter_metadata_state"],
        "required_quarters": required,
        "unresolved_quarters": unresolved,
        "quarter_inventory_known": bool(required),
        "payloads_reverified": False,
        "scientific_context_complete": False,
    }


def scope_groups(metadata):
    trials, components = joint_component_index(metadata)
    groups = defaultdict(list)
    readiness = {}
    for row in trials.values():
        readiness[row["trial_id"]] = check_trial(row)
        groups[(row["scope"], row["role"], row["cohort"])].append(row["trial_id"])
    counts = {key: len(rows) for key, rows in groups.items()}
    expected = {
        (scope, role, cohort): count
        for (scope, role), cohorts in LAUNCH_SCOPES.items()
        for cohort, count in cohorts.items()
    }
    require(
        counts == expected, "Declared launch scope, role, cohort or denominator changed"
    )
    return trials, components, groups, readiness


def metadata_rows(metadata):
    require(
        type(metadata.get("schema")) is int
        and metadata["schema"] == 1
        and metadata.get("status") == "METADATA_ONLY",
        "Control metadata boundary changed",
    )
    require(
        set(metadata["opened_cohorts"]) == set(OPENED_COHORTS), "Opened cohorts changed"
    )
    rows = []
    for name, (_, count) in OPENED_COHORTS.items():
        cohort = metadata["opened_cohorts"][name]
        require(
            cohort["sources"] == count and len(cohort["rows"]) == count,
            "Opened cohort denominator or identity is missing",
        )
        for row in cohort["rows"]:
            require(
                set(row)
                == {
                    "source_id",
                    "parent_tpf",
                    "raw_fits_sha256",
                    "cloud_arrays_sha256",
                    "shard",
                }
                and positive_identity(row["source_id"])
                and positive_identity(row["parent_tpf"])
                and cloud_null.valid_hash(row["raw_fits_sha256"])
                and cloud_null.valid_hash(row["cloud_arrays_sha256"])
                and type(row["shard"]) is int
                and 0 <= row["shard"] <= 65535,
                "Opened source, parent, shard or input hash is invalid",
            )
            rows.append(row)
    for key in ("source_id", "parent_tpf", "raw_fits_sha256", "cloud_arrays_sha256"):
        require(len({row[key] for row in rows}) == len(rows), f"Repeated opened {key}")
    return rows


def protected_sources(metadata):
    exclusions = metadata["sealed_exclusions"]
    require(
        set(exclusions)
        == {
            "current-sealed-raw-exclusion-source-ids.json",
            "sealed-raw-exclusion-source-ids.json",
        },
        "Current or historical sealed exclusions are missing",
    )
    protected = set()
    for row in exclusions.values():
        ids = row["source_ids"]
        require(
            cloud_null.valid_hash(row["sha256"])
            and isinstance(ids, list)
            and len(ids) == len(set(ids))
            and all(positive_identity(value) for value in ids),
            "Invalid protected source identity or exclusion binding",
        )
        protected.update(ids)
    return protected


def opened_bindings(metadata, trials):
    metadata_rows(metadata)
    bindings = []
    protected = protected_sources(metadata)
    for name, (cohort, _) in OPENED_COHORTS.items():
        for row in metadata["opened_cohorts"][name]["rows"]:
            trial_id = f"{cohort}:development:{row['source_id']}"
            trial = trials.get(trial_id, {})
            expected = {
                "source_id": row["source_id"],
                "origin_parent": row["parent_tpf"],
                "raw_fits_sha256": row["raw_fits_sha256"],
                "scope": "CURRENT_OPENED_DEVELOPMENT",
            }
            require(
                all(trial.get(key) == value for key, value in expected.items()),
                f"Opened source/parent/raw/scope reconciliation failed: {trial_id}",
            )
            bindings.append(
                {
                    **row,
                    "trial_id": trial_id,
                    "cohort": cohort,
                    "full_witness_component_id": trial["full_witness_component_id"],
                    "declared_raw_exclusion": row["source_id"] in protected,
                    "execution_authorized": False,
                }
            )
    return bindings


def reproducible_selections(metadata, prior):
    require(
        set(metadata["prospective_pools"]) == {name for name, _ in POOLS},
        "Prospective pools changed",
    )
    parents = {row["parent_tpf"] for row in metadata_rows(metadata)}
    protected = protected_sources(metadata)
    selected = {}
    for name, cohort in POOLS:
        pool = metadata["prospective_pools"][name]
        require(
            pool["status"] == "PROSPECTIVE_NOT_FROZEN"
            and cloud_null.valid_hash(pool["original_sha256"]),
            "Prospective pool freeze or origin identity changed",
        )
        selection = select_opened_parents(pool["rows"], 100, protected, parents)
        require(
            selection == prior["prospective_selections"].get(name),
            "Existing prospective selection or exclusion recipe changed",
        )
        parents.update(row["parent_tpf"] for row in selection["selection"])
        selected[cohort] = {
            **selection,
            "pool": name,
            "original_pool_sha256": pool["original_sha256"],
            "ordered_pool": pool["rows"],
        }
    return selected


def parent_locations(path, identities):
    result = {}
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as database:
        for source in identities:
            rows = database.execute(
                "select parent_tpf,shard from sources where source_id=?", (source,)
            ).fetchall()
            require(
                len(rows) == 1,
                f"Prospective source missing or duplicated in parent index: {source}",
            )
            result[source] = {"parent_tpf": str(rows[0][0]), "shard": str(rows[0][1])}
    return result


def pool_component_members(payload, selected_ids):
    rows = list(csv.DictReader(io.StringIO(payload.decode(), newline="")))
    lookup = {row["source_id"]: row for row in rows}
    require(
        len(lookup) == len(rows) and set(selected_ids) <= lookup.keys(),
        "Prospective identities are missing or repeated in stopped-survey metadata",
    )
    touched = {
        lookup[source]["metadata_dependency_component_id"] for source in selected_ids
    }
    members = defaultdict(list)
    for row in rows:
        if row["metadata_dependency_component_id"] in touched:
            members[row["metadata_dependency_component_id"]].append(row)
    return lookup, dict(members)


def candidate_binding(row, stopped, indexed, scopes, links):
    cloud_null.row_identity(stopped)
    require(
        row["source_id"] == stopped["source_id"]
        and row["parent_tpf"] == stopped["origin_parent"] == indexed["parent_tpf"]
        and str(row["shard"]) == stopped["shard"] == indexed["shard"],
        "Prospective source, parent or shard identities disagree",
    )
    components = cloud_null.ordered_links(stopped["linked_full_witness_component_ids"])
    known = sorted(
        {
            component
            for token in (
                ("source", stopped["source_id"]),
                ("parent", stopped["origin_parent"]),
                ("raw", stopped["raw_fits_sha256"]),
            )
            for component in links.get(token, ())
        }
    )
    require(
        set(known) <= set(components), "Prospective known component links were omitted"
    )
    linked_scopes = sorted(
        {scope for component in components for scope in scopes[component]}
    )
    require(
        linked_scopes
        == list(cloud_null.ordered_links(stopped["linked_existing_scopes"])),
        "Prospective full component scopes differ",
    )
    forbidden = sorted(set(linked_scopes) - {"CURRENT_OPENED_DEVELOPMENT"})
    return {
        **row,
        "raw_fits_sha256": stopped["raw_fits_sha256"],
        "metadata_dependency_component_id": stopped["metadata_dependency_component_id"],
        "linked_full_witness_component_ids": list(components),
        "linked_existing_scopes": linked_scopes,
        "forbidden_scope_links": forbidden,
        "known_direct_component_links": known,
        "stage_exposure": {key: stopped[key] for key in cloud_null.EXPOSURE},
        "status": "NOT_FROZEN",
        "scientifically_eligible": False,
        "independence_certified": False,
        "required_quarter_parents": "NOT_PROVISIONED",
        "scene_inputs": "NOT_PROVISIONED",
        "raw_product_location": "NOT_PROVISIONED",
    }


def prospective_bindings(selections, payload, parent_index, joint):
    ids = [
        row["source_id"] for value in selections.values() for row in value["selection"]
    ]
    require(len(ids) == len(set(ids)), "Prospective cohorts repeat a source identity")
    lookup, members = pool_component_members(payload, ids)
    indexed = parent_locations(parent_index, ids)
    scopes, links = cloud_null.scope_indices(joint)
    results = {}
    for cohort, value in selections.items():
        rows = [
            candidate_binding(
                row, lookup[row["source_id"]], indexed[row["source_id"]], scopes, links
            )
            for row in value["selection"]
        ]
        results[cohort] = {
            **value,
            "binding_rows": rows,
            "scientifically_eligible_sources": 0,
            "scientifically_verified_pairs": 0,
            "status": "NOT_FROZEN",
        }
    return results, members


def identity_graph(trials, components):
    owners = defaultdict(list)
    for trial in trials.values():
        for token in sorted(cloud_null.trial_links(trial)):
            owners[token].append(trial["trial_id"])
    shared = []
    for (kind, identity), members in sorted(owners.items()):
        bound = {trials[trial]["full_witness_component_id"] for trial in members}
        require(
            len(bound) == 1, "Known identity split across full dependency components"
        )
        if len(members) > 1:
            scopes = sorted({trials[trial]["scope"] for trial in members})
            require(
                not {"CURRENT_OPENED_DEVELOPMENT", "CURRENT_CORE_FINAL"} <= set(scopes),
                "Opened/sealed identity overlap prevents control binding",
            )
            shared.append(
                {
                    "kind": kind,
                    "identity": identity,
                    "trial_ids": members,
                    "full_witness_component_id": next(iter(bound)),
                    "scopes": scopes,
                }
            )
    return {
        "status": "KNOWN_METADATA_DEPENDENCIES_ONLY",
        "unique_known_identities": len(owners),
        "shared_identity_edges": shared,
        "full_witness_components": list(components.values()),
        "joint_independence_complete": False,
    }


def verify_holdout_links(joint, trials):
    owners = {
        "parent": lambda row: {
            str(value)
            for kind, value in cloud_null.trial_links(row)
            if kind == "parent"
        }
    }
    observed = set()
    for witness in joint["holdout_overlap_witnesses"]:
        holdout = trials.get(witness["holdout_trial_id"], {})
        require(
            holdout.get("scope") == "HOLDOUT_VETTING_DISPOSITION_ONLY",
            "Holdout witness scope changed",
        )
        for edge in witness["shared_identity_edges"]:
            control = trials.get(edge["control_trial_id"], {})
            kind, identity = edge["identity_kind"], edge["identity_value"]
            require(
                kind in owners
                and control.get("scope") == edge["control_scope"]
                and identity in owners[kind](holdout) & owners[kind](control)
                and holdout["full_witness_component_id"]
                == control["full_witness_component_id"],
                "Holdout shared-parent witness differs from complete component",
            )
            key = witness["holdout_trial_id"], edge["control_trial_id"], kind, identity
            require(key not in observed, "Repeated holdout dependency witness")
            observed.add(key)
    expected = set()
    for component in joint["full_witness_components"]:
        holdouts = [
            trials[t]
            for t in component["trial_ids"]
            if trials[t]["scope"] == "HOLDOUT_VETTING_DISPOSITION_ONLY"
        ]
        controls = [
            trials[t]
            for t in component["trial_ids"]
            if trials[t]["scope"] != "HOLDOUT_VETTING_DISPOSITION_ONLY"
        ]
        for holdout in holdouts:
            for control in controls:
                expected.update(
                    (holdout["trial_id"], control["trial_id"], "parent", identity)
                    for identity in owners["parent"](holdout)
                    & owners["parent"](control)
                )
    require(observed == expected, "Missing holdout shared-parent witness")
    return {
        "witnesses": joint["holdout_overlap_witnesses"],
        "edges": len(observed),
        "independent_holdout_acceptance": False,
    }


def calibration_summary(metadata, prior, truth_descriptor):
    from pipeline.vetting.calibrate import LAUNCH_FALSE_BOI_CAPS, OBJECTIVE_METADATA

    draft = metadata["draft_specification"]
    require(
        draft["sha256"] == DRAFT_SHA256
        and hashlib.sha256(draft["text"].encode()).hexdigest() == DRAFT_SHA256,
        "Unchanged draft specification hash differs",
    )
    original = json.loads(draft["text"])
    require(
        prior.get("status") == "BOUND_METADATA_INPUTS_PENDING"
        and prior.get("original_draft_sha256") == DRAFT_SHA256
        and prior.get("false_boi_caps") == LAUNCH_FALSE_BOI_CAPS
        and all(
            prior.get(key) == original[key]
            for key in ("schema", "confidence", "grid", "start", "diagnostic_cohorts")
        ),
        "Calibration grid, confidence, cap or initial policy changed",
    )
    require(
        len(prior["objective"]) == len(original["objective"]), "Objective scope changed"
    )
    objectives = []
    for term, frozen in zip(prior["objective"], original["objective"], strict=True):
        require(
            all(
                term.get(key) == value
                for key, value in frozen.items()
                if key not in {"opened_truth", "sealed_truth"}
            ),
            "Objective cohort changed",
        )
        cohort, family, category = OBJECTIVE_METADATA[term["opened_cohort"]]
        for role, partition in (("opened", "development"), ("sealed", "evaluation")):
            origin = metadata["strata"][f"{family}-{partition}-strata.jsonl"]
            expected = [row for row in origin["rows"] if row["category"] == category]
            ref = term[f"{role}_truth"]
            path = verified_reference(ref)
            actual = [
                json.loads(line, object_pairs_hook=unique_object)
                for line in path.read_text().splitlines()
            ]
            require(
                actual == expected
                and ref["origin_metadata_sha256"] == origin["sha256"]
                and all(
                    set(row) == {"source_id", "category", "stratum"} for row in actual
                ),
                "Calibration identity/stratum binding changed",
            )
            if role == "opened":
                require(
                    {row["source_id"] for row in actual}
                    == {
                        row["source_id"]
                        for row in metadata["opened_cohorts"][cohort]["rows"]
                    },
                    "Opened objective identities differ",
                )
        objectives.append(term)
    truth_files = {
        key: value
        for key, value in truth_descriptor["files"].items()
        if key.endswith("/truth.jsonl")
    }
    require(
        set(truth_files)
        == {f"lineage/{name}/truth.jsonl" for name in OBJECTIVE_METADATA},
        "Original opened truth descriptor cohort bindings changed",
    )
    require(
        truth_descriptor.get("scope") == "OPENED_DEVELOPMENT_ONLY"
        and truth_descriptor.get("original_input_arrays_or_truths_changed") is False,
        "Opened truth descriptor boundary changed",
    )
    expected_sources = {
        row["source_id"]
        for name in ("planets", "sparse", "ttv")
        for row in metadata["opened_cohorts"][name]["rows"]
    }
    require(
        set(truth_descriptor["source_order"]) == expected_sources
        and len(truth_descriptor["source_order"]) == len(expected_sources)
        and set(truth_descriptor["sources"]) == expected_sources,
        "Original opened truth/source identity scope changed",
    )
    for name in OBJECTIVE_METADATA:
        original_truth = truth_descriptor["original_lineage"][name]["truth.jsonl"]
        require(
            original_truth["member"] == f"lineage/{name}/truth.jsonl"
            and {key: original_truth[key] for key in ("bytes", "sha256")}
            == truth_files[original_truth["member"]]
            and all(
                original_truth[key] == original_truth["original"][key]
                for key in ("bytes", "sha256")
            ),
            "Original truth bytes or lineage identity changed",
        )
    return {
        "status": "BOUND_METADATA_INPUTS_PENDING",
        "original_draft_sha256": DRAFT_SHA256,
        "unchanged_grid": original["grid"],
        "unchanged_start": original["start"],
        "confidence": original["confidence"],
        "false_boi_caps": dict(LAUNCH_FALSE_BOI_CAPS),
        "objective": objectives,
        "original_opened_truth_descriptors": truth_files,
        "opened_record_cases_verified_for_calibration": 0,
        "record_evidence": (
            "COMPLETE_CURRENT_IMPLEMENTATION_RECORD_EVERY_GATE_RUNS_NOT_PROVISIONED"
        ),
        "holdout_stratum": "IDENTITY_METADATA_ONLY_STRATUM_RULE_NOT_PROVISIONED",
        "null_denominators": "SOURCE_VARIANT_PROPOSAL_ENDPOINT_LEDGER_NOT_PROVISIONED",
    }


def required_inputs(prospective):
    common = [
        "required_quarters",
        "actual_parent_by_quarter",
        "raw_product_path_or_uri",
        "scene_catalogue_identity",
        "sibling_source_input_hashes",
        "pixel_receipt_hashes",
        "prf_dependency_hashes",
        "historical_parent_component_witnesses",
        "stage_exposure_by_parent",
    ]
    result = []
    for cohort, value in prospective.items():
        fields = common + (
            [
                "contaminant_source_id",
                "contaminant_raw_fits_sha256",
                "contaminant_parent_by_quarter",
                "pair_scene_eligibility",
            ]
            if cohort == "contamination_pair"
            else [
                "original_Q0_Q1_family_membership",
                "declared_ephemeris_association",
                "family_eligibility",
            ]
        )
        result.append(
            {
                "cohort": cohort,
                "status": "NOT_PROVISIONED",
                "sha256": None,
                "source_ids": [row["source_id"] for row in value["selection"]],
                "required_fields": fields,
                "origin_path": (
                    "dump/outlook-20260930/cohort-outlook.json"
                    if cohort == "guide_star_family"
                    else "NOT_DECLARED"
                ),
                "existing_candidate_pool": {
                    "origin_path": value["pool"],
                    "sha256": value["original_pool_sha256"],
                },
                "correction": (
                    "Supply the hash-bound original eligibility/scene projection "
                    "for these unchanged candidates; no reselection."
                ),
            }
        )
    result.extend(
        [
            {
                "scope": "null",
                "origin_path": "NOT_DECLARED",
                "sha256": None,
                "required_fields": [
                    "two_10000_source_memberships",
                    "base_manifest",
                    "variant_recipe",
                    "seed",
                    "exclusion_graph",
                ],
                "status": "NOT_FROZEN",
            },
            {
                "scope": "flux_injections",
                "origin_path": "dump/release-injections-v1/materialization.json",
                "sha256": None,
                "required_fields": [
                    "original_5000_1000_1000_identity_metadata",
                    "sealed_overlap_exclusions",
                    "declared_1000_periodic_stratified_subset",
                ],
                "status": "METADATA_NOT_PROVISIONED_DO_NOT_READ_PROTECTED_TRUTHS",
            },
            {
                "scope": "real_rehearsal_10000",
                "origin_path": "NOT_DECLARED",
                "sha256": None,
                "required_fields": [
                    "ordered_source_identities",
                    "raw_parent_component_hashes",
                    "role_and_exposure",
                ],
                "status": "NOT_FROZEN",
            },
            {
                "scope": "opened_record_runs",
                "origin_path": "NOT_DECLARED",
                "sha256": None,
                "required_fields": [
                    "complete_record_evidence_report",
                    "code_config_policy_hashes",
                    "all_opened_cases",
                    "proposal_endpoint_ledger",
                ],
                "status": "NOT_PROVISIONED",
            },
        ]
    )
    return result


def verify_joint_specification(spec, joint_path, joint):
    artifact = spec["artifact"]
    require(
        joint_path.stat().st_size == artifact["canonical_json_bytes"]
        and digest(joint_path) == artifact["canonical_json_sha256"],
        "Joint metadata decoded pin changed",
    )
    require(
        joint["provenance"] == spec["provenance"]
        and joint["boundary"] == spec["boundary"]
        and joint["unresolved"] == spec["unresolved"],
        "Joint metadata provenance or unresolved boundaries changed",
    )
    require(
        all(set(row) == set(spec["whitelist"]["trial"]) for row in joint["trials"]),
        "Unexpected fields outside allowed trial identity metadata",
    )
    require(
        all(
            set(row) <= set(spec["whitelist"]["quarter"])
            for trial in joint["trials"]
            for row in trial["pixel_quarters"]
        ),
        "Unexpected fields outside allowed quarter identity metadata",
    )
    require(
        all(
            set(row) == set(spec["whitelist"]["full_component"])
            for row in joint["full_witness_components"]
        ),
        "Unexpected full component witness fields",
    )


def assemble(files):
    metadata, joint = (
        read_json(files["control_metadata"]),
        read_json(files["joint_scope"]),
    )
    spec, null_spec = (
        read_json(files["joint_scope_specification"]),
        read_json(files["null_pool_specification"]),
    )
    verify_joint_specification(spec, files["joint_scope"], joint)
    trials, components, groups, quarters = scope_groups(joint)
    opened = opened_bindings(metadata, trials)
    prior = read_json(files["calibration_bindings"])
    require(
        prior["metadata"]["sha256"] == digest(files["control_metadata"]),
        "Prior calibration metadata pin changed",
    )
    payload = files["null_pool"].read_bytes()
    require(
        len(payload) == null_spec["artifact"]["csv_bytes"]
        and hashlib.sha256(payload).hexdigest() == null_spec["artifact"]["csv_sha256"],
        "Null metadata decoded pin changed",
    )
    binding = null_spec["provenance"]["joint_scope"]
    require(
        binding["specification_sha256"] == digest(files["joint_scope_specification"])
        and binding["canonical_json_sha256"] == digest(files["joint_scope"])
        and binding["gzip_sha256"] == spec["artifact"]["sha256"],
        "Null/joint specification linkage changed",
    )
    cloud_null.check_specification(null_spec, Path(null_spec["artifact"]["path"]))
    null_counts = cloud_null.check_csv(
        payload, null_spec, cloud_null.scope_indices(joint)
    )
    prospective, members = prospective_bindings(
        reproducible_selections(metadata, prior), payload, files["parent_index"], joint
    )
    return {
        "opened_source_bindings": opened,
        "input_graph": identity_graph(trials, components),
        "quarter_readiness": quarters,
        "holdout_links": verify_holdout_links(joint, trials),
        "prospective_cohorts": prospective,
        "prospective_full_stopped_component_members": members,
        "null_pool_verified_counts": null_counts,
        "calibration": calibration_summary(
            metadata, prior, read_json(files["opened_truth_descriptor"])
        ),
        "required_inputs": required_inputs(prospective),
        "unresolved_original_joint_scope": joint["unresolved"],
        "observed_native_parent_supplement": joint["provenance"].get(
            "observed_native_header_parents"
        ),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build(inputs, out):
    from pipeline.deterministic.control_binding_inputs import bind_preparations

    require(
        not out.exists(),
        "Binding output already exists; retain it and choose a fresh directory",
    )
    recipe, files = checked_inputs(inputs)
    initial = reference(inputs)
    root = Path(__file__).resolve().parents[2]
    programs = [reference(root / f"pipeline/{name}") for name in PROGRAM_NAMES]
    result = assemble(files)
    result["retained_prepared_inputs"] = bind_preparations(
        result["opened_source_bindings"],
        recipe.get("opened_preparations", []),
        Path(__file__).resolve().parents[2],
    )
    nested_refs = [
        term[key]
        for term in result["calibration"]["objective"]
        for key in ("opened_truth", "sealed_truth")
    ]
    nested_refs.extend(result["retained_prepared_inputs"]["input_file_references"])
    for ref in [initial, *recipe["inputs"].values(), *nested_refs, *programs]:
        verified_reference(ref)
    joint = read_json(files["joint_scope"])
    trials, _, groups, _ = scope_groups(joint)
    for (scope, role, cohort), ids in groups.items():
        value = {
            "schema": 1,
            "status": "CONTROL_SCOPE_METADATA_PREPARATION_ONLY",
            **BOUNDARY,
            "scope": scope,
            "cohort": cohort,
            "role": role,
            "original_order_retained": True,
            "trial_ids": ids,
            "trials": [trials[trial] for trial in ids],
            "component_binding": bind_joint_components(joint, ids),
        }
        write_json(out / f"scopes/{scope.lower()}--{cohort}--{role}.json", value)
    for name, value in result.items():
        write_json(out / f"{name}.json", value)
    outputs = [reference(path) for path in sorted(out.rglob("*.json"))]
    receipt = {
        "schema": 1,
        "status": "CONTROL_BINDINGS_PREPARATION_ONLY",
        **BOUNDARY,
        "inputs_recipe": initial,
        "inputs": recipe["inputs"],
        "programs": programs,
        "dependent_inputs": nested_refs,
        "output_directory": str(out.absolute()),
        "outputs": outputs,
        "declared_trials": len(trials),
        "opened_identities": len(result["opened_source_bindings"]),
        "prospective_candidates": {
            cohort: len(value["selection"])
            for cohort, value in result["prospective_cohorts"].items()
        },
        "complete_calibration_cases": 0,
        "boundary": (
            "Metadata preparation only; scoped replay authorization, "
            "scientific eligibility and independence are unresolved."
        ),
    }
    for ref in [initial, *recipe["inputs"].values(), *nested_refs, *programs]:
        verified_reference(ref)
    write_json(out / "readiness.json", receipt)
    return receipt


def verify(receipt_path):
    receipt = read_json(receipt_path)
    require(
        type(receipt.get("schema")) is int
        and receipt["schema"] == 1
        and receipt.get("status") == "CONTROL_BINDINGS_PREPARATION_ONLY"
        and all(receipt.get(key) is value for key, value in BOUNDARY.items()),
        "Binding readiness boundary changed",
    )
    output_root = Path(receipt["output_directory"])
    expected = {f"{name}.json" for name in REPORT_NAMES}
    expected.update(
        f"scopes/{scope.lower()}--{cohort}--{role}.json"
        for (scope, role), cohorts in LAUNCH_SCOPES.items()
        for cohort in cohorts
    )
    names = [
        str(Path(ref["path"]).relative_to(output_root)) for ref in receipt["outputs"]
    ]
    require(
        len(names) == len(set(names)) and set(names) == expected,
        "Complete binding output inventory changed",
    )
    require(
        receipt["declared_trials"] == 9007
        and receipt["opened_identities"] == 876
        and receipt["prospective_candidates"]
        == {"guide_star_family": 100, "contamination_pair": 100}
        and receipt["complete_calibration_cases"] == 0,
        "Binding readiness counts changed",
    )
    program_names = [
        str(Path(row["path"]).relative_to(Path(row["path"]).parents[2]))
        for row in receipt["programs"]
    ]
    require(
        program_names == [f"pipeline/{name}" for name in PROGRAM_NAMES],
        "Complete producer program inventory changed",
    )
    for ref in [
        receipt["inputs_recipe"],
        *receipt["inputs"].values(),
        *receipt["dependent_inputs"],
        *receipt["programs"],
        *receipt["outputs"],
    ]:
        verified_reference(ref)
    recipe, _ = checked_inputs(receipt["inputs_recipe"]["path"])
    require(
        recipe["inputs"] == receipt["inputs"] and receipt["outputs"],
        "Bound input inventory changed",
    )
    calibration = read_json(output_root / "calibration.json")
    prepared = read_json(output_root / "retained_prepared_inputs.json")
    expected_dependencies = [
        term[key]
        for term in calibration["objective"]
        for key in ("opened_truth", "sealed_truth")
    ]
    expected_dependencies.extend(prepared["input_file_references"])
    require(
        receipt["dependent_inputs"] == expected_dependencies,
        "Complete dependent input inventory changed",
    )
    return {
        "status": "CONTROL_BINDING_FILES_UNCHANGED",
        "report": reference(receipt_path),
        **BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.inputs:
        if args.out is None:
            parser.error("--out is required with --inputs")
        receipt = build(args.inputs, args.out)
        result = {
            "status": receipt["status"],
            "readiness": reference(args.out / "readiness.json"),
            **BOUNDARY,
        }
    else:
        require(args.out is None, "--out is not accepted with --verify")
        result = verify(args.verify)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
