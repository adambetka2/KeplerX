import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.deterministic import control_prospective_inputs as prospective
from pipeline.deterministic.control_bindings import reference


def write(path, value):
    path.write_text(json.dumps(value))
    return reference(path)


@pytest.fixture
def original(tmp_path):
    path = tmp_path / "raw.fits"
    path.write_bytes(b"original")
    payload = reference(path)
    source = {
        "source_id": "1",
        "parent_tpf": "7",
        "shard": "3",
        "cohort": "guide_star_family",
        "product": {"kic": 9},
        "pool": {
            "raw_fits_sha256": payload["sha256"],
            "metadata_dependency_component_id": "component-7",
        },
        "uri": "original-product",
    }
    item = {"payload": payload}
    return source, item


def test_original_bytes_are_bound_before_reader_access(original):
    source, item = original
    assert prospective.original_payload(
        source, item, set(), set()
    ) == prospective.verified_reference(item["payload"])


@pytest.mark.parametrize("kind", ["source", "raw"])
def test_protected_identity_rejected_before_any_payload_access(original, kind):
    source, item = original
    prospective.Path(item["payload"]["path"]).unlink()
    protected = {"1"} if kind == "source" else set()
    hashes = {source["pool"]["raw_fits_sha256"]} if kind == "raw" else set()
    with pytest.raises(ValueError, match="Protected sealed"):
        prospective.original_payload(source, item, protected, hashes)


def test_changed_raw_bytes_fail_even_when_output_hash_is_updated(original):
    source, item = original
    path = prospective.Path(item["payload"]["path"])
    path.write_bytes(b"changed")
    item["payload"] = reference(path)
    with pytest.raises(ValueError, match="Original raw hash changed"):
        prospective.original_payload(source, item, set(), set())


@pytest.mark.parametrize(
    "rows",
    [
        [{"source_id": "1"}, {"source_id": "1"}],
        [{"source_id": "2"}, {"source_id": "1"}],
        [{"source_id": "1"}],
    ],
)
def test_missing_duplicate_and_reordered_attempts_are_rejected(rows):
    with pytest.raises(ValueError, match="Source order/scope differs"):
        prospective.exact_index(rows, ["1", "2"])


def test_array_encoding_preserves_integer_identities_and_nan_bits(tmp_path):
    raw = {
        "source_id": np.full(2, 2051015590158677632, dtype=np.int64),
        "flux": np.array([float("nan"), -0.0]),
    }
    output = prospective.save_arrays(tmp_path / "source.npz", raw)
    with np.load(output["path"], allow_pickle=False) as saved:
        assert prospective.array_inventory(
            {k: saved[k] for k in saved.files}
        ) == prospective.array_inventory(raw)


def test_existing_arrays_are_not_overwritten(tmp_path):
    path = tmp_path / "source.npz"
    path.write_bytes(b"preserved-attempt")
    with pytest.raises(ValueError, match="already exists"):
        prospective.save_arrays(path, {"x": np.arange(3)})
    assert path.read_bytes() == b"preserved-attempt"


def test_array_shape_dtype_and_content_all_enter_identity():
    original = prospective.array_inventory({"x": np.array([1, 2], dtype=np.int32)})
    for changed in (
        np.array([[1, 2]], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int32),
    ):
        assert prospective.array_inventory({"x": changed}) != original


def test_storage_admission_uses_full_filesystem_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prospective.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=prospective.MINIMUM_FREE_BYTES + 9),
    )
    with pytest.raises(ValueError, match="reserve crossed"):
        prospective.admission(tmp_path, 10)
    assert prospective.admission(tmp_path, 9)["admitted_additional_bytes"] == 9


@pytest.mark.parametrize("additional", [-1, True, 1.0])
def test_invalid_storage_forecasts_rejected(tmp_path, additional):
    with pytest.raises(ValueError, match="Invalid storage forecast"):
        prospective.admission(tmp_path, additional)


def test_unavailable_sources_remain_in_original_requested_scope(tmp_path):
    plan = {
        "sources": [
            {"source_id": "2", "cohort": "guide_star_family"},
            {"source_id": "1", "cohort": "contamination_pair"},
        ]
    }
    path = tmp_path / "plan.json"
    write(path, plan)
    rows = [
        {"source_id": "2", "status": "INPUT_BINDING_FAILED"},
        {
            "source_id": "1",
            "status": "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND",
            "arrays": {"sha256": "a" * 64},
            "metadata": {"source_id": "1"},
        },
    ]
    manifest_ref = prospective.write_manifest(
        plan, rows, tmp_path, path, {"path": "guard.json", "sha256": "b" * 64}
    )
    manifest = prospective.read_json(manifest_ref["path"])
    assert list(manifest["sources"]) == ["1"]
    assert prospective.read_json(tmp_path / "source-ids.json") == ["2", "1"]
    assert prospective.read_json(tmp_path / "guide_star_family-source-ids.json") == [
        "2"
    ]
    assert manifest["status"] == "UNFROZEN_PROSPECTIVE_INPUTS_ONLY"
    assert (
        manifest["role"] == "development"
        and manifest["boundary"] == prospective.BOUNDARY
    )


def test_unknown_quarter_parent_is_not_filled_with_global_parent(original, monkeypatch):
    source, _ = original
    quarter = {"quarter": 0, "quarter_parent": None}
    monkeypatch.setattr(prospective, "header_identity", lambda *_: [quarter])
    record = {"headers": [{"cards": {"RA_OBJ": 290.0, "DEC_OBJ": 38.0, "GMAG": 17.2}}]}
    metadata = prospective.source_metadata(source, record)
    assert metadata["origin_tpf_kic"] == 7
    assert metadata["quarter_parent_bindings"] == [quarter]
    assert metadata["quarter_parent_complete"] is False


def test_changed_preparation_rejected_before_output_arrays(
    original, tmp_path, monkeypatch
):
    from pipeline.deterministic import cloud_public

    source, item = original
    item["record"] = write(tmp_path / "record.json", {"source": source})
    item["preparation"] = {"prepared_cadences": 7}
    monkeypatch.setattr(prospective, "source_metadata", lambda *_: {"kic": 9})
    monkeypatch.setattr(
        cloud_public, "fits_arrays", lambda *_: {"time_bkjd": np.arange(3)}
    )
    monkeypatch.setattr(
        prospective, "mask_exposure", lambda _: ({"prepared_cadences": 8}, None, None)
    )
    with pytest.raises(ValueError, match="Preparation exposure changed"):
        prospective.materialize_source(source, item, tmp_path, set(), set())
    assert not (tmp_path / "arrays").exists()


@pytest.fixture
def encoded(tmp_path):
    (tmp_path / "arrays").mkdir()
    raw = {"x": np.array([1.0, 2.0])}
    row = {
        "source_id": "1",
        "metadata": {"source_id": "1", "origin_tpf_kic": 7},
        "original_payload": write(tmp_path / "raw.json", {"original": True}),
        "archive_record": write(tmp_path / "archive.json", {"source_id": "1"}),
        "arrays": prospective.save_arrays(tmp_path / "arrays/1.npz", raw),
        "array_inventory": prospective.array_inventory(raw),
    }
    runtime = {
        "arrays": "arrays/1.npz",
        "sha256": row["arrays"]["sha256"],
        "metadata": dict(row["metadata"]),
    }
    return row, runtime, tmp_path / "input-manifest.json"


def test_encoded_source_can_be_verified_before_and_after_preparation(encoded):
    row, runtime, manifest = encoded
    prospective.validate_encoded_source(row, runtime, manifest)


@pytest.mark.parametrize("field", ["arrays", "sha256", "metadata"])
def test_runtime_binding_drift_is_rejected(encoded, field):
    row, runtime, manifest = encoded
    runtime[field] = {} if field == "metadata" else "changed"
    with pytest.raises(ValueError, match="binding differs"):
        prospective.validate_encoded_source(row, runtime, manifest)


def test_updated_npz_and_manifest_hash_still_require_original_array_bytes(encoded):
    row, runtime, manifest = encoded
    path = prospective.Path(row["arrays"]["path"])
    np.savez_compressed(path, x=np.array([1.0, 3.0]))
    row["arrays"] = reference(path)
    runtime["sha256"] = row["arrays"]["sha256"]
    with pytest.raises(ValueError, match="original content proof"):
        prospective.validate_encoded_source(row, runtime, manifest)


def test_original_payload_drift_remains_fatal_after_encoding(encoded):
    row, runtime, manifest = encoded
    prospective.Path(row["original_payload"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input unavailable or changed"):
        prospective.validate_encoded_source(row, runtime, manifest)


@pytest.mark.parametrize("change", ["none", "array", "mask", "component", "role"])
def test_updated_content_receipts_still_bind_original_source_proof(
    original, tmp_path, monkeypatch, change
):
    from pipeline.deterministic import cloud_public

    source, item = original
    raw = {"x": np.array([1.0, 2.0])}
    metadata = {"kic": 9}
    item["record"] = write(tmp_path / "record.json", {})
    item["preparation"] = {"prepared_cadences": 2}
    row = {
        "source_id": "1",
        "cohort": source["cohort"],
        "component_id": source["pool"]["metadata_dependency_component_id"],
        "final_scoped_role_authorization": False,
        "archive_record": item["record"],
        "original_payload": item["payload"],
        "metadata": metadata,
        "preparation": dict(item["preparation"]),
        "array_inventory": prospective.array_inventory(raw),
    }
    monkeypatch.setattr(prospective, "source_metadata", lambda *_: metadata)
    monkeypatch.setattr(cloud_public, "fits_arrays", lambda *_: raw)
    monkeypatch.setattr(
        prospective, "mask_exposure", lambda _: (item["preparation"], None, None)
    )
    if change == "array":
        row["array_inventory"] = prospective.array_inventory(
            {"x": np.array([1.0, 3.0])}
        )
    elif change == "mask":
        row["preparation"]["prepared_cadences"] = 3
    elif change == "component":
        row["component_id"] = "unlinked-component"
    elif change == "role":
        row["final_scoped_role_authorization"] = True
    if change == "none":
        prospective.validate_source_proof(source, item, row, empty_protection())
        return
    with pytest.raises(ValueError):
        prospective.validate_source_proof(source, item, row, empty_protection())


def empty_protection():
    return {
        key: set()
        for key in (
            "source_ids",
            "raw_or_pixel_content_hashes",
            "parent_identities",
            "additional_original_stopped_components_protected",
        )
    }


@pytest.fixture
def guard(tmp_path):
    graph = prospective.read_json(
        prospective.Path(prospective.__file__).with_name("cloud-joint-scope.json")
    )["artifact"]["canonical_json_sha256"]
    data = {
        "schema": 1,
        "status": "FULL_COMPONENT_AND_INDEXED_PARENT_CLOSURE_BEFORE_PIXEL_PROVISIONING",
        "inputs": {"current_graph": {"sha256": graph}},
        "six_scene_links_remain_protected": True,
        "five_retired_reservations_remain_historical_exclusions": True,
        "arrays_or_outcomes_read": False,
        **{key: [] for key in empty_protection()},
        "protected_full_component_ids": ["full-witness-component"],
        "source_ids": [str(sid) for sid in range(10, 16)],
        "potential_evaluation_scene_links": [
            {"target_source_id": str(sid)} for sid in range(10, 16)
        ],
    }
    return data, tmp_path / "guard.json"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_ids", "1"),
        ("raw_or_pixel_content_hashes", None),
        ("parent_identities", "7"),
        ("additional_original_stopped_components_protected", "component-7"),
    ],
)
def test_full_component_validation_refuses_before_reader_or_file_access(
    original, guard, monkeypatch, field, value
):
    from pipeline.deterministic import cloud_public

    source, item = original
    data, path = guard
    value = value or source["pool"]["raw_fits_sha256"]
    data[field] = sorted([*data[field], value])
    protection = prospective.protection_sets(write(path, data))
    prospective.Path(item["payload"]["path"]).unlink()
    monkeypatch.setattr(
        cloud_public, "fits_arrays", lambda *_: pytest.fail("Protected array read")
    )
    with pytest.raises(ValueError, match="Protected full-component"):
        prospective.validate_source_proof(source, item, {}, protection)


@pytest.mark.parametrize(
    "change",
    ["bytes", "graph", "lost_scene", "lost_link", "no_components", "duplicates"],
)
def test_protection_receipt_drift_refused(guard, change):
    data, path = guard
    receipt = write(path, data)
    if change == "bytes":
        path.write_text(path.read_text() + " ")
    else:
        if change == "graph":
            data["inputs"]["current_graph"]["sha256"] = "f" * 64
        elif change == "lost_scene":
            data["source_ids"].pop()
        elif change == "lost_link":
            data["potential_evaluation_scene_links"].pop()
        elif change == "no_components":
            data["protected_full_component_ids"] = []
        else:
            data["source_ids"].append(data["source_ids"][-1])
        receipt = write(path, data)
    with pytest.raises(ValueError):
        prospective.protection_sets(receipt)


def test_scene_linked_target_stays_metadata_only_during_materialization(
    original, guard, tmp_path, monkeypatch
):
    source, _ = original
    data, guard_path = guard
    data["source_ids"] = sorted([*data["source_ids"], source["source_id"]])
    receipt = write(guard_path, data)
    exclusions = write(tmp_path / "exclusions.json", {})
    plan_path = tmp_path / "plan.json"
    plan = {"sources": [source], "inputs": {"sealed_identity_exclusions": exclusions}}
    plan_ref = write(plan_path, plan)
    guide_path = tmp_path / "guide.json"
    guide_ref = write(guide_path, {})
    pair_path = tmp_path / "pairs.json"
    write(
        pair_path,
        {
            "schema": 1,
            "boundary": prospective.BOUNDARY,
            "inputs": {"plan": plan_ref, "target_report": guide_ref},
        },
    )
    inventory = tmp_path / "inventory.json"
    write(inventory, {})
    monkeypatch.setattr(prospective, "verify_plan", lambda _: None)
    monkeypatch.setattr(prospective, "checked_report", lambda *_: {})
    monkeypatch.setattr(prospective, "source_inputs", lambda *_: {"1": {}})
    monkeypatch.setattr(prospective, "protected_sources", lambda _: set())
    monkeypatch.setattr(prospective, "reviewed_scopes", lambda *_: [])
    monkeypatch.setattr(
        prospective,
        "materialize_source",
        lambda *_: pytest.fail("Protected array read"),
    )
    out = tmp_path / "out"
    report = prospective.materialize(
        plan_path, guide_path, pair_path, inventory, tmp_path, out, receipt
    )
    assert report["counts"] == {"INPUT_BINDING_FAILED": 1}
    row = prospective.read_json(out / "records/1.json")
    assert row["message"] == "Protected full-component prospective input"
    manifest = prospective.read_json(out / "input-manifest.json")
    assert manifest["sources"] == {} and manifest["protected_inputs"] == receipt
    assert prospective.read_json(out / "source-ids.json") == ["1"]
    assert not list((out / "arrays").iterdir())


def test_legacy_validation_requires_protection_before_any_dependencies(tmp_path):
    path = tmp_path / "report.json"
    write(
        path,
        {
            "schema": 1,
            "boundary": prospective.BOUNDARY,
            "runtime_policy_required": "DEVELOPMENT",
            "final_scoped_role_authorization": False,
            "inputs": {},
        },
    )
    with pytest.raises(ValueError, match="full-component protection receipt required"):
        prospective.validate_materialization(path)
