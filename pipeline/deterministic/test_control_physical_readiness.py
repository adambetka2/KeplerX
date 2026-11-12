import copy
import json

import pytest

from pipeline.deterministic import control_physical_readiness as readiness
from pipeline.deterministic.control_bindings import reference
from pipeline.deterministic.control_eligibility import BOUNDARY
from pipeline.deterministic.control_parent_candidates import reconcile_candidates
from pipeline.deterministic.control_parent_headers import stamp_witness
from pipeline.deterministic.test_control_parent_candidates import (
    aperture,
    parent_record,
    quarter,
    record,
    selected,
)


def write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def reports(tmp_path):
    plan = write(tmp_path / "plan.json", {"selected": 1})
    target = write(tmp_path / "target.json", {"selected": 1})
    report = {
        "schema": 1,
        "boundary": BOUNDARY,
        "plan": reference(plan),
        "target_report": reference(target),
    }
    path = write(tmp_path / "report.json", report)
    return plan, target, path, report


@pytest.mark.parametrize("change", ["schema", "boundary", "plan", "target"])
def test_report_scope_and_post_assembly_input_drift_rejected(reports, change):
    plan, target, path, value = reports
    if change == "schema":
        value["schema"] = 99
    elif change == "boundary":
        value["boundary"] = {**BOUNDARY, "scope_frozen": True}
    else:
        (plan if change == "plan" else target).write_text("changed")
    write(path, value)
    with pytest.raises(ValueError):
        readiness.checked_report(path, plan, target)


def test_real_eligibility_boundary_is_preserved(reports):
    plan, target, path, _ = reports
    value = readiness.checked_report(path, plan, target)
    assert value["boundary"] == BOUNDARY
    assert value["boundary"]["cohorts_frozen"] is False
    assert value["boundary"]["guide_mechanism_attributed"] is False


def test_missing_and_repeated_quarter_identities_rejected():
    row = {"source_id": "1", "quarter": 1}
    with pytest.raises(ValueError, match="Repeated"):
        readiness.index_quarters([row, dict(row)])
    with pytest.raises(KeyError):
        readiness.index_quarters([{"source_id": "1"}])


@pytest.mark.parametrize("value", [-1, 18, "1", True])
def test_unknown_or_invalid_quarter_identity_is_rejected(value):
    with pytest.raises(ValueError, match="Invalid quarter"):
        readiness.index_quarters([{"source_id": "1", "quarter": value}])


def nominal_case(tmp_path):
    source = selected()
    source["parent_tpf"] = "7"
    raw = record()
    path = write(tmp_path / "parent.json", raw)
    row = {
        "source_id": source["source_id"],
        "cohort": source["cohort"],
        "record": reference(path),
        **stamp_witness(quarter(), raw),
    }
    return source, raw, path, row


def test_origin_match_does_not_pass_different_aperture_dimensions(tmp_path):
    source, _, _, row = nominal_case(tmp_path)
    different = aperture()
    different["cards"]["NAXIS1"] = 3
    _, witness = readiness.nominal_witness(source, quarter(), different, row)
    assert witness["stamp_origin_and_channel_match"] is True
    assert witness["exact_declared_geometry_match"] is False
    assert witness["physical_attribution_measured"] is False


@pytest.mark.parametrize(
    "change", ["raw_hash", "parent", "quarter", "cohort", "witness"]
)
def test_nominal_parent_input_or_identity_drift_rejected(tmp_path, change):
    source, raw, path, row = nominal_case(tmp_path)
    if change == "raw_hash":
        path.write_text("changed")
    elif change == "parent":
        source["parent_tpf"] = "9"
    elif change == "quarter":
        raw["quarter"] = 2
        write(path, raw)
        row["record"] = reference(path)
    elif change == "cohort":
        row["cohort"] = "contamination_pair"
    else:
        row["target_header_position_inside_candidate_stamp"] = False
    with pytest.raises(ValueError):
        readiness.nominal_witness(source, quarter(), aperture(), row)


def test_candidate_inventory_cannot_drop_failed_parent(tmp_path):
    path = write(tmp_path / "candidate.json", record())
    row = reconcile_candidates(selected(), quarter(), aperture(), [(path, record())])
    with pytest.raises(ValueError, match="inventory changed"):
        readiness.candidate_witness(
            selected(), quarter(), aperture(), row, {"candidate_parent_ids": [7, 9]}
        )


def test_candidate_match_cannot_be_changed_after_assembly(tmp_path):
    path = write(tmp_path / "candidate.json", record())
    row = reconcile_candidates(selected(), quarter(), aperture(), [(path, record())])
    row["matching_candidate_parents"] = ["9"]
    with pytest.raises(ValueError, match="witness changed"):
        readiness.candidate_witness(
            selected(), quarter(), aperture(), row, {"candidate_parent_ids": [7]}
        )


def all_quarters_case(tmp_path):
    source, raw, path, nominal = nominal_case(tmp_path)
    second = parent_record(9)
    second_path = write(tmp_path / "second.json", second)
    archive = write(
        tmp_path / "archive.json", {"headers": [{"name": "APERTURE_Q1", **aperture()}]}
    )
    candidates = {
        "quarter_coverage": "ALL_ORIGINAL_OBSERVED_QUARTERS",
        "quarters": [
            reconcile_candidates(
                source, quarter(), aperture(), [(path, raw), (second_path, second)]
            )
        ],
    }
    return (
        {"sources": [source]},
        {
            "sources": [
                {"archive_record": reference(archive), "quarter_witnesses": [quarter()]}
            ]
        },
        {"quarters": [nominal]},
        candidates,
        {"1": {"candidate_parent_ids": [7, 9]}},
    )


def test_readiness_retains_all_matching_parents_when_nominal_matches(tmp_path):
    row = readiness.quarter_rows(*all_quarters_case(tmp_path))[0]
    assert row["matching_header_parent_ids"] == ["7", "9"]
    assert (
        row["additional_candidates"]["status"] == "AMBIGUOUS_MATCHING_PUBLIC_CANDIDATES"
    )
    assert row["original_global_parent"] == "7"
    assert row["original_extraction_parent_uniqueness_verified"] is False


@pytest.mark.parametrize("change", ["missing", "extra", "coverage"])
def test_all_quarter_candidate_scope_drift_refused(tmp_path, change):
    args = all_quarters_case(tmp_path)
    candidates = args[3]
    if change == "missing":
        candidates["quarters"] = []
    elif change == "extra":
        extra = copy.deepcopy(candidates["quarters"][0])
        extra["quarter"] = 2
        candidates["quarters"].append(extra)
    else:
        candidates["quarter_coverage"] = "UNKNOWN"
    with pytest.raises(ValueError):
        readiness.quarter_rows(*args)


def test_absent_donor_keeps_original_selection_and_unavailable_result(tmp_path):
    binding = {"source_id": "1", "parent_tpf": "7"}
    raw = {"binding": binding, "status": "UNAVAILABLE", "attempts": [{"status": 404}]}
    path = write(tmp_path / "donor.json", raw)
    row = {
        "target_source_id": "1",
        "record": reference(path),
        "status": "UNAVAILABLE",
        "donor": None,
        "target_donor_quarters": [],
    }
    result = readiness.donor_rows(
        {"donor_bindings": [binding]},
        {"sources": [{"source_id": "1"}]},
        {"donors": [row], "status_counts": {"UNAVAILABLE": 1}},
    )
    assert len(result) == 1 and result[0]["donor"] is None


def test_donor_reselection_or_omission_is_rejected():
    plan = {"donor_bindings": [{"source_id": "1"}]}
    for rows in ([], [{"target_source_id": "2"}]):
        with pytest.raises(ValueError, match="identities or order changed"):
            readiness.donor_rows(plan, {"sources": []}, {"donors": rows})


def trial(scope, sid, parent, component):
    return {
        "trial_id": scope + ":" + sid,
        "scope": scope,
        "source_id": sid,
        "origin_parent": parent,
        "resolved_parents": [parent],
        "additional_parent_identities": [],
        "pixel_quarters": [],
        "raw_fits_sha256": "a" * 64,
        "full_witness_component_id": component,
    }


def test_new_header_parent_preserves_link_to_sealed_and_holdout_components():
    scope = {
        "trials": [
            trial("CURRENT_CORE_FINAL", "2", "7", "shared"),
            trial("HOLDOUT_VETTING_DISPOSITION_ONLY", "3", "7", "shared"),
        ]
    }
    source = {**selected(), "pool": {"metadata_dependency_component_id": "old"}}
    before = copy.deepcopy(scope)
    links = readiness.parent_links(
        {"sources": [source]},
        [{"source_id": "1", "quarter": 1, "matching_header_parent_ids": ["7"]}],
        [(None, scope)],
    )
    assert len(links[0]["linked_existing_trials"]) == 2
    assert links[0]["opened_or_sealed_link_unresolved"] is True
    assert links[0]["identity_disjointness_certified"] is False
    assert scope == before and links[0]["original_component_reassigned"] is False


def test_empty_known_trial_links_do_not_certify_independence():
    source = {**selected(), "pool": {"metadata_dependency_component_id": "old"}}
    links = readiness.parent_links(
        {"sources": [source]},
        [{"source_id": "1", "quarter": 1, "matching_header_parent_ids": ["9"]}],
        [],
    )
    assert links[0]["linked_existing_trials"] == []
    assert links[0]["identity_disjointness_certified"] is False


def test_guide_missing_q0_retained_without_reselection_or_freeze():
    source = selected()
    row = {"source_id": "1", "quarter": 1, "matching_header_parent_ids": ["7"]}
    result = readiness.readiness_sources(
        {"sources": [source]},
        {"sources": [{"archive_record": {}, "witness": {"raw_hash_verified": True}}]},
        [row],
        [],
    )
    assert len(result) == 1 and result[0]["q0_q1_both_available"] is False
    assert result[0]["physical_eligibility_verified"] is False
    assert result[0]["cohort_frozen"] is False


def test_observed_quarter_parent_preserves_full_other_stopped_component():
    source = {**selected(), "pool": {"metadata_dependency_component_id": "original"}}
    pool = [
        {
            "source_id": "1",
            "origin_parent": "8",
            "metadata_dependency_component_id": "original",
            "vetting_exposure": "UNRESOLVED",
        },
        {
            "source_id": "2",
            "origin_parent": "7",
            "metadata_dependency_component_id": "other",
            "vetting_exposure": "UNRESOLVED",
        },
        {
            "source_id": "3",
            "origin_parent": "9",
            "metadata_dependency_component_id": "other",
            "vetting_exposure": "UNRESOLVED",
        },
    ]
    before = copy.deepcopy(pool)
    result = readiness.stopped_component_closure(
        {"sources": [source]},
        [{"source_id": "1", "quarter": 1, "matching_header_parent_ids": ["7"]}],
        pool,
    )
    assert result["additional_component_ids"] == ["other"]
    assert result["full_original_stopped_component_members"]["other"] == pool[1:]
    assert result["conservative_linkage_groups"][0][
        "original_stopped_component_ids"
    ] == ["original", "other"]
    assert result["joint_independence_complete"] is False and pool == before
    assert result["original_component_ids_or_members_reassigned"] is False


def test_unmapped_parent_does_not_certify_independence():
    source = {**selected(), "pool": {"metadata_dependency_component_id": "original"}}
    pool = [
        {
            "source_id": "1",
            "origin_parent": "8",
            "metadata_dependency_component_id": "original",
        }
    ]
    result = readiness.stopped_component_closure(
        {"sources": [source]},
        [{"source_id": "1", "quarter": 1, "matching_header_parent_ids": ["7"]}],
        pool,
    )
    assert result["observed_parent_edges"][0]["unknown_parent_in_stopped_pool"] is True
    assert (
        result["observed_parent_edges"][0]["identity_disjointness_certified"] is False
    )


def test_repeated_stopped_source_or_missing_original_component_is_rejected():
    source = {**selected(), "pool": {"metadata_dependency_component_id": "original"}}
    row = {
        "source_id": "1",
        "origin_parent": "7",
        "metadata_dependency_component_id": "original",
    }
    with pytest.raises(ValueError, match="Repeated stopped-pool"):
        readiness.stopped_component_closure({"sources": [source]}, [], [row, dict(row)])
    with pytest.raises(ValueError, match="Original stopped component disappeared"):
        readiness.stopped_component_closure(
            {"sources": [source]},
            [],
            [{**row, "metadata_dependency_component_id": "other"}],
        )
