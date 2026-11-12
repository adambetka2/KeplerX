"""Reconcile unchanged prospective controls with verified archive witnesses."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.deterministic import cloud_null
from pipeline.deterministic.control_bindings import (
    digest,
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_donor_headers import (
    bind_donor_header,
    quarter_association,
    verify_targets,
)
from pipeline.deterministic.control_eligibility import (
    BOUNDARY,
    guide_witness,
    save,
    verify_plan,
)
from pipeline.deterministic.control_eligibility_archive import verify_headers
from pipeline.deterministic.control_execution_inputs import reviewed_scopes
from pipeline.deterministic.control_parent_candidates import (
    exact_stamp_match,
    reconcile_candidates,
)
from pipeline.deterministic.control_parent_headers import (
    stamp_witness,
    verify_parent_headers,
)


def checked_report(path, plan_path, target_path=None):
    value = read_json(path)
    require(
        type(value.get("schema")) is int and value["schema"] == 1,
        "Archive report schema changed",
    )
    require(value["boundary"] == BOUNDARY, "Archive report acceptance scope changed")
    require(value["plan"] == reference(plan_path), "Archive report plan changed")
    if target_path is not None:
        require(
            value["target_report"] == reference(target_path),
            "Archive report source input changed",
        )
    return value


def index_quarters(rows):
    result = {}
    for row in rows:
        key = row["source_id"], row["quarter"]
        require(key not in result, "Repeated source/quarter identity")
        require(
            type(row["quarter"]) is int and 0 <= row["quarter"] <= 17,
            "Invalid quarter identity",
        )
        result[key] = row
    return result


def verify_programs(report):
    programs = report["implementation"]
    if "path" in programs:
        verified_reference(programs)
        return
    root = Path(__file__).resolve().parents[2]
    for relative, expected in programs.items():
        path = Path(relative)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and path.parts[0] == "pipeline",
            "Archive program path escapes pipeline",
        )
        require(
            digest(root / path) == expected, "Archive measured implementation changed"
        )


def verify_preparation(plan, targets):
    for source, target in zip(plan["sources"], targets["sources"], strict=True):
        record = read_json(verified_reference(target["archive_record"]))
        if record["status"] == "UNAVAILABLE":
            continue
        total = record.get("product_bytes", record.get("archive_product_bytes"))
        if total is not None:
            verify_headers(record["headers"], total)
        if source["cohort"] != "guide_star_family":
            continue
        payload = verified_reference(record["payload"])
        expected = guide_witness(
            source, payload, record["headers"], plan["original_family_proposals"]
        )
        require(
            target["witness"] == expected,
            "Original preparation/ephemeris witness changed",
        )


def target_quarters(target):
    return target.get(
        "quarter_witnesses", target.get("witness", {}).get("quarter_witnesses", [])
    )


def nominal_witness(source, quarter, aperture, row):
    require(row["cohort"] == source["cohort"], "Quarter cohort changed")
    record = read_json(verified_reference(row["record"]))
    require(
        record["parent_kic"] == source["parent_tpf"]
        and record["quarter"] == quarter["quarter"],
        "Quarter parent identity changed",
    )
    if record["status"] == "UNAVAILABLE":
        expected = {
            "quarter": quarter["quarter"],
            "status": "CANONICAL_PARENT_HEADERS_UNAVAILABLE",
        }
        exact = None
    else:
        require(
            record["status"] == "VERIFIED_PARENT_HEADERS"
            and record["geometry"]
            == verify_parent_headers(
                record["headers"], source["parent_tpf"], quarter["quarter"]
            ),
            "Parent header geometry changed",
        )
        expected = stamp_witness(quarter, record)
        exact = exact_stamp_match(quarter, aperture, record)
    require(
        row
        == {
            "source_id": source["source_id"],
            "cohort": source["cohort"],
            "record": row["record"],
            **expected,
        },
        "Nominal parent witness changed",
    )
    return record, exact


def candidate_witness(source, quarter, aperture, row, projection):
    require(
        [r["parent_kic"] for r in row["candidates"]]
        == [str(p) for p in projection["candidate_parent_ids"]],
        "Candidate parent inventory changed",
    )
    records = []
    for declared in row["candidates"]:
        path = verified_reference(declared["record"])
        record = read_json(path)
        require(
            record["parent_kic"] == declared["parent_kic"],
            "Candidate cached parent identity changed",
        )
        records.append((path, record))
    expected = reconcile_candidates(source, quarter, aperture, records)
    require(row == expected, "Candidate stamp witness changed")
    return expected


def quarter_rows(plan, targets, nominal, candidates, projection):
    nominal_index = index_quarters(nominal["quarters"])
    candidate_index = index_quarters(candidates["quarters"])
    coverage = candidates.get("quarter_coverage")
    require(
        coverage in {None, "ALL_ORIGINAL_OBSERVED_QUARTERS"},
        "Unknown candidate quarter coverage",
    )
    rows, expected, pending = [], set(), set()
    for source, target in zip(plan["sources"], targets["sources"], strict=True):
        record = read_json(verified_reference(target["archive_record"]))
        headers = {h["name"]: h for h in record.get("headers", [])}
        for quarter in target_quarters(target):
            key = source["source_id"], quarter["quarter"]
            require(key not in expected, "Repeated source quarter in LC input")
            expected.add(key)
            require(key in nominal_index, "Missing nominal source/quarter witness")
            aperture = headers.get(f"APERTURE_Q{quarter['quarter']}")
            raw, exact = nominal_witness(source, quarter, aperture, nominal_index[key])
            parents = []
            if exact and exact["exact_declared_geometry_match"]:
                parents.append(source["parent_tpf"])
            candidate = None
            if not exact or not exact["exact_declared_geometry_match"]:
                pending.add(key)
                require(key in candidate_index, "Missing unresolved quarter witness")
            if coverage is not None or key in pending:
                require(key in candidate_index, "Missing original quarter candidates")
                candidate = candidate_witness(
                    source,
                    quarter,
                    aperture,
                    candidate_index[key],
                    projection[source["source_id"]],
                )
                parents.extend(candidate["matching_candidate_parents"])
            rows.append(
                {
                    "source_id": source["source_id"],
                    "cohort": source["cohort"],
                    "quarter": quarter["quarter"],
                    "original_global_parent": source["parent_tpf"],
                    "original_raw_fits_sha256": source["pool"]["raw_fits_sha256"],
                    "nominal_record": nominal_index[key]["record"],
                    "nominal_exact_geometry": exact,
                    "additional_candidates": candidate,
                    "matching_header_parent_ids": sorted(set(parents)),
                    "nominal_archive_product_bytes": raw.get("product_bytes"),
                    "original_extraction_parent_uniqueness_verified": False,
                    "physical_scene_measured": False,
                }
            )
    require(set(nominal_index) == expected, "Nominal source/quarter scope changed")
    require(
        set(candidate_index) == (expected if coverage is not None else pending),
        "Candidate quarter scope changed",
    )
    return rows


def donor_rows(plan, targets, report):
    expected = plan["donor_bindings"]
    require(
        [r["target_source_id"] for r in report["donors"]]
        == [r["source_id"] for r in expected],
        "Donor target identities or order changed",
    )
    target_index = {r["source_id"]: r for r in targets["sources"]}
    for binding, row in zip(expected, report["donors"], strict=True):
        record = read_json(verified_reference(row["record"]))
        require(record["binding"] == binding, "Donor binding changed")
        require(row["status"] == record["status"], "Donor status changed")
        donor = None
        if record["status"] == "VERIFIED_DONOR_ARCHIVE_HEADERS":
            verify_headers(record["headers"], record["archive_product_bytes"])
            donor = bind_donor_header(binding, record["headers"])
        else:
            require(record["status"] == "UNAVAILABLE", "Invalid donor failure status")
        require(row["donor"] == donor, "Donor product identity changed")
        require(
            row["target_donor_quarters"]
            == quarter_association(target_index[binding["source_id"]], record),
            "Donor quarter association changed",
        )
    require(
        report["status_counts"] == dict(Counter(r["status"] for r in report["donors"])),
        "Donor denominator changed",
    )
    return report["donors"]


def existing_parent_index(scopes):
    result = defaultdict(list)
    for _, scope in scopes:
        for trial in scope["trials"]:
            for kind, identity in cloud_null.trial_links(trial):
                if kind == "parent":
                    result[str(identity)].append(
                        {
                            "trial_id": trial["trial_id"],
                            "scope": trial["scope"],
                            "full_witness_component_id": trial[
                                "full_witness_component_id"
                            ],
                        }
                    )
    return result


def parent_links(plan, quarters, scopes):
    index = existing_parent_index(scopes)
    source_index = {r["source_id"]: r for r in plan["sources"]}
    links = []
    for row in quarters:
        source = source_index[row["source_id"]]
        for parent in row["matching_header_parent_ids"]:
            linked = index.get(parent, [])
            links.append(
                {
                    "source_id": source["source_id"],
                    "quarter": row["quarter"],
                    "observed_header_parent": parent,
                    "original_global_parent": source["parent_tpf"],
                    "original_stopped_component_id": source["pool"][
                        "metadata_dependency_component_id"
                    ],
                    "linked_existing_trials": linked,
                    "opened_or_sealed_link_unresolved": bool(linked),
                    "original_component_reassigned": False,
                    "identity_disjointness_certified": False,
                }
            )
    return links


def stopped_component_closure(plan, quarters, pool):
    identities = [r["source_id"] for r in pool]
    require(
        len(identities) == len(set(identities)), "Repeated stopped-pool source identity"
    )
    parent_components = defaultdict(set)
    members = defaultdict(list)
    for row in pool:
        component = row["metadata_dependency_component_id"]
        parent_components[row["origin_parent"]].add(component)
        members[component].append(row)
    sources = {r["source_id"]: r for r in plan["sources"]}
    original = {r["pool"]["metadata_dependency_component_id"] for r in plan["sources"]}
    require(original <= members.keys(), "Original stopped component disappeared")
    adjacency = {component: set() for component in original}
    edges = []
    for row in quarters:
        source = sources[row["source_id"]]
        component = source["pool"]["metadata_dependency_component_id"]
        for parent in row["matching_header_parent_ids"]:
            linked = sorted(parent_components.get(parent, []))
            for other in linked:
                adjacency[component].add(other)
                adjacency.setdefault(other, set()).add(component)
            edges.append(
                {
                    "source_id": row["source_id"],
                    "quarter": row["quarter"],
                    "observed_header_parent": parent,
                    "original_component_id": component,
                    "linked_stopped_component_ids": linked,
                    "unknown_parent_in_stopped_pool": not linked,
                    "identity_disjointness_certified": False,
                }
            )
    groups = component_groups(adjacency)
    return {
        "status": "CONSERVATIVE_OBSERVED_PARENT_LINKAGE_NOT_SCIENTIFIC_INDEPENDENCE",
        "original_component_ids": sorted(original),
        "additional_component_ids": sorted(adjacency.keys() - original),
        "full_original_stopped_component_members": {
            key: members[key] for key in sorted(adjacency)
        },
        "observed_parent_edges": edges,
        "conservative_linkage_groups": groups,
        "original_component_ids_or_members_reassigned": False,
        "null_population_or_seed_selected": False,
        "joint_independence_complete": False,
    }


def component_groups(adjacency):
    pending = set(adjacency)
    groups = []
    while pending:
        frontier, seen = [min(pending)], set()
        while frontier:
            component = frontier.pop()
            if component in seen:
                continue
            seen.add(component)
            frontier.extend(adjacency[component] - seen)
        pending -= seen
        components = sorted(seen)
        groups.append(
            {
                "metadata_linkage_group_id": hashlib.sha256(
                    json.dumps(components).encode()
                ).hexdigest(),
                "original_stopped_component_ids": components,
                "independent_trial_count_asserted": False,
            }
        )
    return groups


def readiness_sources(plan, targets, quarters, donors):
    by_source = defaultdict(list)
    for row in quarters:
        by_source[row["source_id"]].append(row)
    donor_index = {r["target_source_id"]: r for r in donors}
    results = []
    for source, target in zip(plan["sources"], targets["sources"], strict=True):
        sid = source["source_id"]
        guide = source["cohort"] == "guide_star_family"
        witness = target.get("witness", {})
        required = (
            [r for r in by_source[sid] if r["quarter"] in (0, 1)]
            if guide
            else by_source[sid]
        )
        observed = {r["quarter"] for r in required}
        results.append(
            {
                "source_id": sid,
                "cohort": source["cohort"],
                "original_parent_tpf": source["parent_tpf"],
                "source_archive_record": target["archive_record"],
                "raw_input_sha256_reverified": guide
                and witness.get("raw_hash_verified", False),
                "q0_q1_both_available": {0, 1} <= observed if guide else None,
                "production_preparation": witness.get("preparation") if guide else None,
                "original_family_proposals": witness.get("original_family_proposals")
                if guide
                else None,
                "required_observed_quarters": [r["quarter"] for r in required],
                "quarters_with_matching_parent_headers": sum(
                    bool(r["matching_header_parent_ids"]) for r in required
                ),
                "donor": donor_index.get(sid),
                "missing_dependencies": [
                    "authoritative_original_quarter_parent_and_complete_scene_inventory",
                    "full_original_pixel_and_PRF_hash_bindings",
                    "scene_astrometry_and_sibling_curve_inputs",
                    "physical_eligibility_measurement_and_historical_stage_exposure",
                ]
                + (["original_donor_raw_input_hash_and_flux"] if not guide else []),
                "physical_eligibility_verified": False,
                "cohort_frozen": False,
            }
        )
    return results


def build(paths, inventory, bindings_directory, out):
    require(not out.exists(), "Physical readiness output exists; retain it")
    plan = read_json(paths["plan"])
    verify_plan(plan)
    targets = checked_report(paths["target"], paths["plan"])
    verify_targets(targets, plan, paths["plan"])
    verify_preparation(plan, targets)
    reports = {
        k: checked_report(paths[k], paths["plan"], paths["target"])
        for k in ("donor", "parent", "candidate")
    }
    for report in (targets, *reports.values()):
        verify_programs(report)
    require(
        reports["candidate"]["nominal_parent_report"] == reference(paths["parent"]),
        "Candidate report nominal input changed",
    )
    require(
        reports["candidate"].get("nominal_match_rule")
        == "ORIGIN_CHANNEL_AND_APERTURE_DIMENSIONS",
        "Candidate reconciliation omits aperture dimensions",
    )
    projection = read_json(
        verified_reference(reports["candidate"]["public_candidates"])
    )
    require(
        list(projection) == [s["source_id"] for s in plan["sources"]],
        "Candidate source scope changed",
    )
    scopes = reviewed_scopes(inventory, bindings_directory)
    quarters = quarter_rows(
        plan, targets, reports["parent"], reports["candidate"], projection
    )
    donors = donor_rows(plan, targets, reports["donor"])
    sources = readiness_sources(plan, targets, quarters, donors)
    links = parent_links(plan, quarters, scopes)
    pool_path = verified_reference(plan["inputs"]["stopped_identity_pool"])
    with pool_path.open(newline="") as stream:
        pool = list(csv.DictReader(stream))
    closure = stopped_component_closure(plan, quarters, pool)
    result = {
        "schema": 1,
        "status": "ACTUAL_ARCHIVE_WITNESSES_PHYSICAL_ELIGIBILITY_PENDING",
        "inputs": {k: reference(p) for k, p in paths.items()},
        "reviewed_inventory": reference(inventory),
        "original_reviewed_scopes": [reference(p) for p, _ in scopes],
        "sources": sources,
        "quarter_parent_witnesses": quarters,
        "conservative_parent_links": links,
        "stopped_pool_parent_closure": closure,
        "counts": {
            "declared_sources": len(sources),
            "observed_quarters": len(quarters),
            "guide_q0_q1_available": sum(
                s["q0_q1_both_available"] is True for s in sources
            ),
            "quarters_with_matching_header_parents": sum(
                bool(r["matching_header_parent_ids"]) for r in quarters
            ),
            "links_to_existing_trial_scope": sum(
                bool(r["linked_existing_trials"]) for r in links
            ),
            "physical_eligibility_verified": 0,
            "scientific_scene_measurements": 0,
        },
        "implementation": reference(Path(__file__)),
        "boundary": BOUNDARY,
    }
    save(out / "physical-readiness.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "plan",
        "target",
        "donor",
        "parent",
        "candidate",
        "inventory",
        "bindings-directory",
        "out",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        key: getattr(args, key)
        for key in ("plan", "target", "donor", "parent", "candidate")
    }
    print(build(paths, args.inventory, args.bindings_directory, args.out)["counts"])


if __name__ == "__main__":
    main()
