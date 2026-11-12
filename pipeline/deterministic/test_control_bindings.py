import copy
import csv
import gzip
import io
import json
import sqlite3
from pathlib import Path

import pytest

from pipeline.deterministic import cloud_null
from pipeline.deterministic import control_bindings as bindings
from pipeline.deterministic.control_partitions import joint_component_index

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def original_joint():
    with gzip.open(ROOT / "pipeline/deterministic/cloud-joint-scope.json.gz") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def original_metadata():
    return bindings.read_json(
        ROOT / "pipeline/deterministic/cloud-control-metadata.json"
    )


@pytest.fixture
def calibration(tmp_path, original_metadata):
    from pipeline.vetting.calibrate import prepare_bindings

    path = ROOT / "pipeline/deterministic/cloud-control-metadata.json"
    return prepare_bindings(path, tmp_path / "calibration.json")


def test_original_scope_preserves_all_trials_components_and_holdout_links(
    original_joint,
):
    trials, components, groups, readiness = bindings.scope_groups(original_joint)
    assert len(trials) == 9007 and len(components) == 8960
    assert len(groups) == 18 and set(readiness) == set(trials)
    graph = bindings.identity_graph(trials, components)
    assert graph["full_witness_components"] == original_joint["full_witness_components"]
    assert graph["joint_independence_complete"] is False
    result = bindings.verify_holdout_links(original_joint, trials)
    assert result["edges"] == 19
    assert result["witnesses"] == original_joint["holdout_overlap_witnesses"]
    assert result["independent_holdout_acceptance"] is False


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "scope", "role", "parent", "raw"]
)
def test_launch_scope_rejects_invalid_or_changed_identities(original_joint, mutation):
    joint = copy.deepcopy(original_joint)
    row = joint["trials"][0]
    if mutation == "duplicate":
        joint["trials"].append(copy.deepcopy(row))
    elif mutation == "missing":
        del row["source_id"]
    elif mutation == "scope":
        row["scope"] = "CURRENT_CORE_FINAL"
    elif mutation == "role":
        row["role"] = "evaluation"
    elif mutation == "parent":
        row["origin_parent"] = ""
    else:
        row["raw_fits_sha256"] = "bad-hash"
    with pytest.raises(ValueError):
        bindings.scope_groups(joint)


def test_opened_reconciliation_retains_all_original_and_protected_ids(
    original_metadata, original_joint
):
    trials, _ = joint_component_index(original_joint)
    rows = bindings.opened_bindings(original_metadata, trials)
    assert len(rows) == 876
    assert sum(row["declared_raw_exclusion"] for row in rows) == 2
    assert all(row["execution_authorized"] is False for row in rows)


@pytest.mark.parametrize(
    "field", ["source_id", "parent_tpf", "raw_fits_sha256", "cloud_arrays_sha256"]
)
def test_opened_reconciliation_rejects_duplicate_identities(original_metadata, field):
    metadata = copy.deepcopy(original_metadata)
    rows = metadata["opened_cohorts"]["ordinary"]["rows"]
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match="Repeated opened"):
        bindings.metadata_rows(metadata)


@pytest.mark.parametrize("mutation", ["raw", "parent", "missing", "sealed_scope"])
def test_opened_source_binding_rejects_contradictory_joint_identity(
    original_metadata, original_joint, mutation
):
    trials, _ = joint_component_index(copy.deepcopy(original_joint))
    source = original_metadata["opened_cohorts"]["ordinary"]["rows"][0]
    trial_id = f"ordinary:development:{source['source_id']}"
    if mutation == "missing":
        del trials[trial_id]
    else:
        field, value = {
            "raw": ("raw_fits_sha256", "1" * 64),
            "parent": ("origin_parent", "1"),
            "sealed_scope": ("scope", "CURRENT_CORE_FINAL"),
        }[mutation]
        trials[trial_id][field] = value
    with pytest.raises(ValueError, match="reconciliation"):
        bindings.opened_bindings(original_metadata, trials)


def test_prospective_order_recipe_and_original_100_selections(
    original_metadata, calibration
):
    result = bindings.reproducible_selections(original_metadata, calibration)
    assert list(result) == ["guide_star_family", "contamination_pair"]
    all_parents = []
    for name, value in result.items():
        assert value["status"] == "NOT_FROZEN"
        assert len(value["selection"]) == 100
        assert len(value["selection"]) + len(value["excluded"]) == len(
            value["ordered_pool"]
        )
        all_parents.extend(row["parent_tpf"] for row in value["selection"])
    assert len(set(all_parents)) == 200
    assert (
        result["guide_star_family"]["selection"][0]["source_id"]
        == "2051015590158677632"
    )


@pytest.mark.parametrize(
    "mutation", ["order", "duplicate", "selection", "exclusion", "frozen"]
)
def test_prospective_recipe_rejects_silent_reselection_or_scope_change(
    original_metadata, calibration, mutation
):
    metadata = copy.deepcopy(original_metadata)
    name = bindings.POOLS[0][0]
    if mutation == "order":
        rows = metadata["prospective_pools"][name]["rows"]
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "duplicate":
        metadata["prospective_pools"][name]["rows"].append(
            metadata["prospective_pools"][name]["rows"][0]
        )
    elif mutation == "selection":
        calibration["prospective_selections"][name]["selection"].reverse()
    elif mutation == "exclusion":
        source = calibration["prospective_selections"][name]["selection"][0][
            "source_id"
        ]
        metadata["sealed_exclusions"]["current-sealed-raw-exclusion-source-ids.json"][
            "source_ids"
        ].append(source)
    else:
        metadata["prospective_pools"][name]["status"] = "FROZEN"
    with pytest.raises(ValueError):
        bindings.reproducible_selections(metadata, calibration)


def stopped_row(source="1", parent="10", component="a" * 64, links="", scopes=""):
    return {
        "source_id": source,
        "shard": "0",
        "origin_parent": parent,
        "raw_fits_sha256": source.zfill(64),
        "metadata_dependency_component_id": component,
        "linked_existing_scopes": scopes,
        "linked_full_witness_component_ids": links,
        **cloud_null.EXPOSURE,
    }


def candidate(row):
    return {
        "source_id": row["source_id"],
        "parent_tpf": row["origin_parent"],
        "shard": row["shard"],
    }


def test_prospective_forbidden_scope_and_exposure_remain_ineligible():
    component = "b" * 64
    row = stopped_row(links=component, scopes="CURRENT_CORE_FINAL")
    result = bindings.candidate_binding(
        candidate(row),
        row,
        candidate(row),
        {component: {"CURRENT_CORE_FINAL"}},
        {("parent", "10"): {component}},
    )
    assert result["forbidden_scope_links"] == ["CURRENT_CORE_FINAL"]
    assert result["status"] == "NOT_FROZEN"
    assert result["scientifically_eligible"] is False
    assert result["independence_certified"] is False
    assert result["stage_exposure"] == cloud_null.EXPOSURE


def test_unlinked_candidate_remains_unresolved():
    row = stopped_row()
    result = bindings.candidate_binding(candidate(row), row, candidate(row), {}, {})
    assert result["linked_existing_scopes"] == []
    assert result["scene_inputs"] == "NOT_PROVISIONED"
    assert result["scientifically_eligible"] is False


@pytest.mark.parametrize("mutation", ["parent", "shard", "raw", "missing_component"])
def test_prospective_identity_reconciliation_rejects_missing_or_changed_fields(
    mutation,
):
    row = stopped_row()
    index, links = candidate(row), {}
    if mutation in {"parent", "shard"}:
        index["parent_tpf" if mutation == "parent" else "shard"] = "20"
    elif mutation == "raw":
        row["raw_fits_sha256"] = ""
    else:
        links[("parent", "10")] = {"b" * 64}
    with pytest.raises(ValueError):
        bindings.candidate_binding(candidate(row), row, index, {}, links)


def test_full_stopped_component_preserves_unselected_sibling_identities():
    rows = [stopped_row(), stopped_row("2"), stopped_row("3", "30", "c" * 64)]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=cloud_null.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    lookup, components = bindings.pool_component_members(
        stream.getvalue().encode(), ["1"]
    )
    assert lookup["1"] == rows[0]
    assert components == {"a" * 64: rows[:2]}
    with pytest.raises(ValueError, match="missing or repeated"):
        bindings.pool_component_members(stream.getvalue().encode(), ["missing"])


def test_parent_index_rejects_missing_and_duplicate_source_bindings(tmp_path):
    path = tmp_path / "parents.sqlite"
    with sqlite3.connect(path) as database:
        database.execute(
            "create table sources(source_id text, parent_tpf int, shard int)"
        )
        database.executemany(
            "insert into sources values(?,?,?)",
            [("1", 10, 0), ("2", 20, 0), ("2", 21, 0)],
        )
    assert bindings.parent_locations(path, ["1"]) == {
        "1": {"parent_tpf": "10", "shard": "0"}
    }
    for source in ["2", "3"]:
        with pytest.raises(ValueError, match="missing or duplicated"):
            bindings.parent_locations(path, [source])


def quarter_trial():
    return {
        "origin_parent": "10",
        "resolved_parents": ["10"],
        "additional_parent_identities": [],
        "required_pixel_quarters": [0, 1],
        "pixel_quarters": [],
        "pixel_quarter_metadata_state": "UNRESOLVED",
    }


def test_unknown_and_absent_quarter_parents_never_pass_readiness():
    row = quarter_trial()
    row["pixel_quarters"] = [
        {"quarter": 0, "actual_parent_kic": None, "state": "UNAVAILABLE"}
    ]
    result = bindings.quarter_readiness(row)
    assert result["unresolved_quarters"] == [0, 1]
    assert result["scientific_context_complete"] is False
    assert result["payloads_reverified"] is False
    row["required_pixel_quarters"], row["pixel_quarters"] = [], []
    assert bindings.quarter_readiness(row)["quarter_inventory_known"] is False


@pytest.mark.parametrize("mutation", ["unknown_parent", "duplicate", "undeclared"])
def test_quarter_identity_contradictions_are_rejected(mutation):
    row = quarter_trial()
    quarter = {
        "quarter": 0,
        "actual_parent_kic": 10,
        "content_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
    }
    row["pixel_quarters"] = [quarter]
    if mutation == "unknown_parent":
        quarter["actual_parent_kic"] = 20
    elif mutation == "duplicate":
        row["pixel_quarters"].append(copy.deepcopy(quarter))
    else:
        quarter["quarter"] = 2
    with pytest.raises(ValueError):
        bindings.quarter_readiness(row)


def test_holdout_graph_rejects_omitted_or_changed_overlap_witness(original_joint):
    joint = copy.deepcopy(original_joint)
    trials, _ = joint_component_index(joint)
    joint["holdout_overlap_witnesses"].pop()
    with pytest.raises(ValueError, match="Missing holdout"):
        bindings.verify_holdout_links(joint, trials)


def test_opened_sealed_shared_identity_is_rejected(original_joint):
    joint = copy.deepcopy(original_joint)
    trials, components = joint_component_index(joint)
    opened = next(
        row for row in trials.values() if row["scope"] == "CURRENT_OPENED_DEVELOPMENT"
    )
    sealed = next(
        row for row in trials.values() if row["scope"] == "CURRENT_CORE_FINAL"
    )
    sealed["origin_parent"] = opened["origin_parent"]
    sealed["full_witness_component_id"] = opened["full_witness_component_id"]
    with pytest.raises(ValueError, match="Opened/sealed"):
        bindings.identity_graph(trials, components)


def test_unchanged_calibration_grid_and_strata_remain_metadata_only(
    original_metadata, calibration
):
    descriptor = bindings.read_json(
        ROOT / "pipeline/deterministic/cloud-opened-source-metadata.json"
    )
    result = bindings.calibration_summary(original_metadata, calibration, descriptor)
    assert result["opened_record_cases_verified_for_calibration"] == 0
    assert len(result["original_opened_truth_descriptors"]) == 3
    assert result["status"] == "BOUND_METADATA_INPUTS_PENDING"


@pytest.mark.parametrize(
    "mutation",
    [
        "grid",
        "cap",
        "objective",
        "strata",
        "status",
        "truth_scope",
        "truth_ids",
        "truth_hash",
    ],
)
def test_calibration_binding_rejects_changed_grid_truth_scope_or_input(
    original_metadata, calibration, mutation
):
    descriptor = bindings.read_json(
        ROOT / "pipeline/deterministic/cloud-opened-source-metadata.json"
    )
    if mutation == "grid":
        calibration["grid"][next(iter(calibration["grid"]))] = [123]
    elif mutation == "cap":
        calibration["false_boi_caps"]["ordinary"] = 0.5
    elif mutation == "objective":
        calibration["objective"].pop()
    elif mutation == "strata":
        path = Path(calibration["objective"][0]["opened_truth"]["path"])
        path.write_text(path.read_text() + "\n")
    elif mutation == "status":
        calibration["status"] = "CALIBRATED"
    elif mutation == "truth_scope":
        descriptor["scope"] = "SEALED"
    elif mutation == "truth_ids":
        descriptor["source_order"].pop()
    else:
        descriptor["original_lineage"]["periodic_planet"]["truth.jsonl"]["sha256"] = (
            "c" * 64
        )
    with pytest.raises(ValueError):
        bindings.calibration_summary(original_metadata, calibration, descriptor)


def test_changed_input_hash_and_symlink_are_rejected(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{}")
    ref = bindings.reference(path)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        bindings.verified_reference(ref)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="changed"):
        bindings.verified_reference(bindings.reference(alias))


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"source_id":"1","source_id":"2"}')
    with pytest.raises(ValueError, match="Duplicate JSON identity"):
        bindings.read_json(path)
