"""Portable null metadata preserves unselected identities and known dependencies."""

import copy
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloud_null
from deterministic import cloud


def write_json(path, value):
    path.write_text(json.dumps(value))


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_pool(fixture, fields=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or cloud_null.FIELDS)
    writer.writeheader()
    writer.writerows(fixture.rows)
    payload = stream.getvalue().encode()
    fixture.bundle.write_bytes(gzip.compress(payload, mtime=0))
    fixture.specification["artifact"] = {
        "path": fixture.bundle.name, "bytes": fixture.bundle.stat().st_size,
        "sha256": cloud_null.digest(fixture.bundle), "csv_bytes": len(payload), "csv_sha256": sha(payload),
    }
    write_json(fixture.spec, fixture.specification)
    return payload


def write_scope(fixture):
    payload = cloud_null.canonical(fixture.scope_metadata)
    fixture.scope_bundle.write_bytes(gzip.compress(payload, mtime=0))
    specification = {"artifact": {
        "path": fixture.scope_bundle.name, "bytes": fixture.scope_bundle.stat().st_size,
        "sha256": cloud_null.digest(fixture.scope_bundle), "canonical_json_bytes": len(payload),
        "canonical_json_sha256": sha(payload),
    }}
    write_json(fixture.scope_spec, specification)
    fixture.specification["provenance"]["joint_scope"] = {
        "specification_sha256": cloud_null.digest(fixture.scope_spec),
        "gzip_sha256": cloud_null.digest(fixture.scope_bundle), "canonical_json_sha256": sha(payload),
    }
    write_json(fixture.spec, fixture.specification)


@pytest.fixture
def null_pool(tmp_path, monkeypatch):
    scratch = tmp_path / "dump"
    scratch.mkdir()
    component, separate, witness = "a" * 64, "b" * 64, "c" * 64
    scopes = ("CURRENT_OPENED_DEVELOPMENT", "HOLDOUT_VETTING_DISPOSITION_ONLY")
    rows = [{
        "source_id": str(source), "shard": "0", "origin_parent": "10" if source < 3 else "30",
        "raw_fits_sha256": str(source) * 64, "metadata_dependency_component_id": component if source < 3 else separate,
        "linked_existing_scopes": ";".join(scopes) if source < 3 else "",
        "linked_full_witness_component_ids": witness if source < 3 else "", **cloud_null.EXPOSURE,
    } for source in (1, 2, 3)]
    identity = {"origin_path": "metadata.json", "bytes": 10, "sha256": "d" * 64}
    originals = [{"source_id": row["source_id"], "shard": row["shard"],
                  "parent_tpf": row["origin_parent"], "raw_fits_sha256": row["raw_fits_sha256"]} for row in rows]
    specification = {
        "schema": 1, "status": "CLOUD_NULL_POOL_METADATA_ONLY", "csv_fields": list(cloud_null.FIELDS),
        "exposure": dict(cloud_null.EXPOSURE), "unresolved": {"dependencies": "Not complete"},
        "boundary": "Metadata only; historical inputs were verified at origin, not here.",
        **{key: False for key in cloud_null.FLAGS},
        "counts": {
            "sources": 3, "origin_parents": 2, "raw_fits_hashes": 3, "metadata_dependency_components": 2,
            "shards": [0], "ordered_identity_metadata_sha256": sha(cloud_null.canonical(originals)),
            "linked_scope_sources": {key: 2 for key in scopes}, "linked_scope_components": {key: 1 for key in scopes},
            "component_scope_combinations": {";".join(scopes): 1, "NO_EXISTING_CONTROL_TRIAL_LINK": 1},
            "no_existing_control_link_sources": 1, "no_existing_control_link_components": 1,
        },
        "provenance": {"inventory": dict(identity), "program": dict(identity), "identity_audit": dict(identity),
                       "original_inputs": [dict(identity)], "joint_scope": {}},
    }
    trials = [{
        "trial_id": name, "scope": scope, "full_witness_component_id": witness,
        "source_id": source, "raw_fits_sha256": raw, "origin_parent": parent,
        "resolved_parents": ["11"], "additional_parent_identities": ["12"],
        "contaminant_source_id": "20", "contaminant_raw_fits_sha256": "e" * 64,
        "pixel_quarters": [{"actual_parent_kic": "15", "content_sha256": "f" * 64}],
    } for name, scope, source, raw, parent in (
        ("opened:1", scopes[0], "1", "1" * 64, "10"),
        ("heldout:100", scopes[1], "100", "d" * 64, "99"),
    )]
    fixture = SimpleNamespace(
        dump=scratch, rows=rows, specification=specification,
        spec=tmp_path / "cloud-null-pool.json", bundle=tmp_path / "cloud-null-pool.csv.gz",
        scope_spec=tmp_path / "cloud-joint-scope.json", scope_bundle=tmp_path / "cloud-joint-scope.json.gz",
        scope_metadata={"schema": 1, "status": "PORTABLE_METADATA_SCOPE_PREPARATION_ONLY",
                        "joint_independence_complete": False, "qualified_external_connectivity": [], "trials": trials,
                        "full_witness_components": [{"component_id": witness,
                            "trial_ids": [row["trial_id"] for row in trials],
                            "identity_map_members": {"historical": ["reserved:200"]},
                            "historical_stage_exposure": "UNRESOLVED"}]},
    )
    write_pool(fixture)
    write_scope(fixture)
    for key, value in {"DUMP": scratch, "NULL_POOL_SPECIFICATION": fixture.spec,
                       "NULL_POOL_BUNDLE": fixture.bundle, "SCOPE_SPECIFICATION": fixture.scope_spec,
                       "SCOPE_BUNDLE": fixture.scope_bundle}.items():
        monkeypatch.setattr(cloud, key, value)
    return fixture


def test_null_pool_preserves_exact_readonly_metadata_and_false_boundaries(null_pool):
    first = cloud.null_pool()
    assert first == cloud.null_pool()
    assert first["status"] == "NULL_CANDIDATE_METADATA_VERIFIED"
    assert first["counts"] == null_pool.specification["counts"]
    assert first["provenance"] == null_pool.specification["provenance"]
    assert first["exposure"] == cloud_null.EXPOSURE
    assert all(first[key] is False for key in cloud_null.FLAGS)
    path = null_pool.dump / "cloud-null-pool.csv"
    assert path.read_bytes() == gzip.decompress(null_pool.bundle.read_bytes())
    receipt = null_pool.dump / "cloud-null-pool-receipt.json"
    assert json.loads(receipt.read_bytes()) == first
    assert path.stat().st_mode & 0o222 == receipt.stat().st_mode & 0o222 == 0
    assert not list(null_pool.dump.glob("*.part"))
    assert cloud.execute(SimpleNamespace(command="null-pool")) == first


@pytest.mark.parametrize("flag", cloud_null.FLAGS)
def test_null_pool_rejects_boundary_changes_and_invalidates_old_receipt(null_pool, flag):
    cloud.null_pool()
    null_pool.specification[flag] = True
    write_json(null_pool.spec, null_pool.specification)
    with pytest.raises(ValueError, match="boundary"):
        cloud.null_pool()
    assert not (null_pool.dump / "cloud-null-pool-receipt.json").exists()


@pytest.mark.parametrize("name", [
    "cloud-null-pool.csv", "cloud-null-pool.part", "cloud-null-pool-receipt.json", "cloud-null-pool-receipt.part",
])
def test_null_pool_rejects_all_output_symlinks_before_writes(null_pool, tmp_path, name):
    outside = tmp_path / "original.json"
    outside.write_bytes(b"unchanged")
    (null_pool.dump / name).symlink_to(outside)
    with pytest.raises(ValueError, match="redirect"):
        cloud.null_pool()
    assert outside.read_bytes() == b"unchanged"
    assert (null_pool.dump / name).is_symlink()
    assert {path.name for path in null_pool.dump.iterdir()} == {name}


@pytest.mark.parametrize("name", ["cloud-null-pool.part", "cloud-null-pool-receipt.part"])
def test_null_pool_rejects_stale_partial_and_invalidates_receipt(null_pool, name):
    cloud.null_pool()
    (null_pool.dump / name).write_bytes(b"inspect this partial")
    with pytest.raises(ValueError, match="Stale"):
        cloud.null_pool()
    assert (null_pool.dump / name).read_bytes() == b"inspect this partial"
    assert not (null_pool.dump / "cloud-null-pool-receipt.json").exists()


@pytest.mark.parametrize("field,value", [
    ("bytes", True), ("bytes", 16 * 1024**2 + 1), ("csv_bytes", True),
    ("csv_bytes", 64 * 1024**2 + 1), ("csv_sha256", "0" * 64), ("sha256", "0" * 64),
])
def test_null_pool_checks_bounded_compressed_and_decoded_identities(null_pool, field, value):
    null_pool.specification["artifact"][field] = value
    write_json(null_pool.spec, null_pool.specification)
    with pytest.raises(ValueError, match="identities|metadata changed"):
        cloud.null_pool()


@pytest.mark.parametrize("field,value", [
    ("source_id", "01"), ("source_id", "-1"), ("shard", "0.0"), ("shard", "65536"),
    ("origin_parent", ""), ("raw_fits_sha256", "bad"), ("metadata_dependency_component_id", "bad"),
    ("detector_exposure", "UNEXPOSED"), ("vetting_exposure", "UNEXPOSED"),
])
def test_null_pool_rechecks_typed_rows_even_with_rehashed_payload(null_pool, field, value):
    null_pool.rows[0][field] = value
    write_pool(null_pool)
    with pytest.raises(ValueError, match="typed|exposure"):
        cloud.null_pool()


@pytest.mark.parametrize("defect", ["duplicate", "cross_shard_duplicate", "reordered", "missing", "extra_column"])
def test_null_pool_rejects_changed_membership_order_or_schema(null_pool, defect):
    if defect == "duplicate":
        null_pool.rows[1]["source_id"] = "1"
    elif defect == "cross_shard_duplicate":
        null_pool.rows[2]["source_id"] = "1"
        null_pool.rows[2]["shard"] = "1"
    elif defect == "reordered":
        null_pool.rows.reverse()
    elif defect == "missing":
        null_pool.rows.pop()
    else:
        for row in null_pool.rows:
            row["outcome"] = "forbidden"
        write_pool(null_pool, (*cloud_null.FIELDS, "outcome"))
    if defect != "extra_column":
        write_pool(null_pool)
    with pytest.raises(ValueError, match="unique|counts|columns"):
        cloud.null_pool()


@pytest.mark.parametrize("defect", ["missing_direct", "unknown", "wrong_scopes", "duplicates", "inconsistent", "split"])
def test_null_pool_checks_full_witness_links_and_component_closure(null_pool, defect):
    row = null_pool.rows[0]
    if defect == "missing_direct":
        row["linked_full_witness_component_ids"] = row["linked_existing_scopes"] = ""
    elif defect == "unknown":
        row["linked_full_witness_component_ids"] = "9" * 64
    elif defect == "wrong_scopes":
        row["linked_existing_scopes"] = "CURRENT_OPENED_DEVELOPMENT"
    elif defect == "duplicates":
        row["linked_full_witness_component_ids"] += ";" + row["linked_full_witness_component_ids"]
    elif defect == "inconsistent":
        null_pool.rows[2]["metadata_dependency_component_id"] = row["metadata_dependency_component_id"]
    else:
        null_pool.rows[1]["metadata_dependency_component_id"] = "9" * 64
    write_pool(null_pool)
    with pytest.raises(ValueError, match="missing|Unknown|scopes|sorted|disagree|split"):
        cloud.null_pool()


@pytest.mark.parametrize("field,value", [
    ("source_id", "20"), ("origin_parent", "11"), ("origin_parent", "12"),
    ("origin_parent", "15"), ("raw_fits_sha256", "e" * 64),
])
def test_null_pool_preserves_contaminant_resolved_and_actual_quarter_links(null_pool, field, value):
    null_pool.rows[2][field] = value
    write_pool(null_pool)
    with pytest.raises(ValueError, match="Direct known"):
        cloud.null_pool()


@pytest.mark.parametrize("field", ["sources", "origin_parents", "raw_fits_hashes", "metadata_dependency_components"])
def test_null_pool_rejects_bool_counts(null_pool, field):
    null_pool.specification["counts"][field] = True
    write_json(null_pool.spec, null_pool.specification)
    with pytest.raises(ValueError, match="counts"):
        cloud.null_pool()


def test_null_pool_rejects_wrong_joint_provenance(null_pool):
    null_pool.specification["provenance"]["joint_scope"]["specification_sha256"] = "9" * 64
    write_json(null_pool.spec, null_pool.specification)
    with pytest.raises(ValueError, match="Bound joint scope"):
        cloud.null_pool()


def test_null_pool_preserves_changed_reused_csv_but_removes_stale_receipt(null_pool):
    cloud.null_pool()
    path = null_pool.dump / "cloud-null-pool.csv"
    path.chmod(0o600)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Previously provisioned"):
        cloud.null_pool()
    assert path.read_bytes() == b"changed"
    assert not (null_pool.dump / "cloud-null-pool-receipt.json").exists()


def test_null_pool_rejects_a_truncated_original_full_witness(null_pool):
    del null_pool.scope_metadata["full_witness_components"][0]["identity_map_members"]
    write_scope(null_pool)
    with pytest.raises(ValueError, match="Incomplete full joint"):
        cloud.null_pool()


def test_null_pool_requires_new_audit_for_added_external_connectivity(null_pool):
    null_pool.scope_metadata["qualified_external_connectivity"] = [{"external": "new dependency"}]
    write_scope(null_pool)
    with pytest.raises(ValueError, match="Additional qualified external connectivity"):
        cloud.null_pool()


def test_null_pool_rejects_cross_shard_duplicate_with_all_counts_and_hashes_rebound(null_pool):
    row = null_pool.rows[2]
    row.update(source_id="1", shard="1", metadata_dependency_component_id="a" * 64,
               linked_existing_scopes=null_pool.rows[0]["linked_existing_scopes"],
               linked_full_witness_component_ids="c" * 64)
    scopes = null_pool.rows[0]["linked_existing_scopes"].split(";")
    counts = null_pool.specification["counts"]
    counts.update(metadata_dependency_components=1, shards=[0, 1],
                  linked_scope_sources={key: 3 for key in scopes},
                  component_scope_combinations={";".join(scopes): 1},
                  no_existing_control_link_sources=0, no_existing_control_link_components=0)
    identities = [{"source_id": value["source_id"], "shard": value["shard"],
                   "parent_tpf": value["origin_parent"], "raw_fits_sha256": value["raw_fits_sha256"]}
                  for value in null_pool.rows]
    counts["ordered_identity_metadata_sha256"] = sha(cloud_null.canonical(identities))
    write_pool(null_pool)
    with pytest.raises(ValueError, match="unique and in committed order"):
        cloud.null_pool()


def test_null_pool_rejects_specification_drift_before_publishing(null_pool, monkeypatch):
    original = cloud_null.check_csv

    def change_specification(*args):
        result = original(*args)
        null_pool.specification["boundary"] = "Changed during verification"
        write_json(null_pool.spec, null_pool.specification)
        return result

    monkeypatch.setattr(cloud_null, "check_csv", change_specification)
    with pytest.raises(ValueError, match="changed during verification"):
        cloud.null_pool()
    assert not list(null_pool.dump.iterdir())


def test_null_pool_rejects_joint_specification_drift_before_publishing(null_pool, monkeypatch):
    original = cloud_null.scope_indices

    def change_scope(metadata):
        result = original(metadata)
        null_pool.scope_spec.write_text("Changed during verification")
        return result

    monkeypatch.setattr(cloud_null, "scope_indices", change_scope)
    with pytest.raises(ValueError, match="changed during verification"):
        cloud.null_pool()
    assert not list(null_pool.dump.iterdir())


@pytest.mark.parametrize("attribute", ["bundle", "scope_bundle"])
def test_null_pool_rejects_bundle_drift_and_invalidates_previous_receipt(null_pool, monkeypatch, attribute):
    cloud.null_pool()
    original = cloud_null.check_csv

    def change_bundle(*args):
        result = original(*args)
        getattr(null_pool, attribute).write_bytes(b"Changed during verification")
        return result

    monkeypatch.setattr(cloud_null, "check_csv", change_bundle)
    with pytest.raises(ValueError, match="changed during verification"):
        cloud.null_pool()
    assert not (null_pool.dump / "cloud-null-pool-receipt.json").exists()


def test_null_pool_rechecks_inputs_after_csv_publish_before_success_receipt(null_pool, monkeypatch):
    original = cloud_null.publish

    def change_bundle(path, payload):
        original(path, payload)
        if path.suffix == ".csv":
            null_pool.bundle.write_bytes(b"Changed after metadata publication")

    monkeypatch.setattr(cloud_null, "publish", change_bundle)
    with pytest.raises(ValueError, match="changed during verification"):
        cloud.null_pool()
    assert (null_pool.dump / "cloud-null-pool.csv").exists()
    assert not (null_pool.dump / "cloud-null-pool-receipt.json").exists()


def test_null_pool_invalidates_receipt_before_rejecting_a_nonfile_output(null_pool):
    cloud.null_pool()
    path = null_pool.dump / "cloud-null-pool.csv"
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="regular files"):
        cloud.null_pool()
    assert path.is_dir()
    assert not (null_pool.dump / "cloud-null-pool-receipt.json").exists()


@pytest.fixture
def observed_scope(null_pool):
    edges = []
    metadata = null_pool.scope_metadata
    for position in range(6):
        source, reader, parent = str(10000 + position), str(1000 + position), str(30 + position)
        raw = sha(f"native raw {position}".encode())
        trial_id = f"native_eb:development:{source}"
        witness = sha(trial_id.encode())
        metadata["trials"].append({
            "trial_id": trial_id, "source_id": source, "source_domain": "KBonus_LC1",
            "cohort": "native_eb", "role": "development", "scope": "CURRENT_OPENED_DEVELOPMENT",
            "origin_parent": reader, "raw_fits_sha256": raw, "additional_parent_identities": [parent],
            "resolved_parents": [reader], "contaminant_source_id": "", "contaminant_raw_fits_sha256": "",
            "pixel_quarters": [], "full_witness_component_id": witness,
        })
        metadata["full_witness_components"].append({
            "component_id": witness, "trial_ids": [trial_id],
            "identity_map_members": {"ordinary-expanded-exposure-reserved-parent-map.csv": 1},
            "historical_stage_exposure": "UNRESOLVED_BY_STAGE_IN_ORIGINAL_IDENTITY_MAPS",
        })
        edges.append({
            "source_id": source, "trial_id": trial_id, "source_domain": "KBonus_LC1",
            "original_reader_parent": reader, "actual_raw_header_TPFORG": parent,
            "raw_fits_sha256": raw, "raw_file": {"path": f"original/{source}.fits", "bytes": 2880, "sha256": raw},
            "raw_header": {"GAIAID": f"Gaia DR3 {source}", "KEPLERID": int(reader), "TPFORG": int(parent),
                "HLSPID": "KBONUS-BKG", "HLSPVER": "V1.0", "original_declared_parent": int(reader),
                "actual_header_parent_matches_original_declared": False, "header_origin_binding": "ORIGINAL_CATALOGUE_PRODUCT",
                "extensions": ["PRIMARY", "LIGHTCURVE_STITCHED", "LIGHTCURVE_Q1", "APERTURE_Q1"]},
            "native_reader_contract": "catalogue_controls.read_product, exact original21 arrays",
            "supplemental_identity_edge": ["parent", parent],
            "historical_reservation_namespace_counts": {
                "opened-resolved-parent-map.csv": 0, "historical-opened-survey-parent-map.csv": 0,
                "paused-survey-opened-parent-map.csv": 0, "ordinary-expanded-exposure-reserved-parent-map.csv": 1},
            "historical_reservation_identity_rows": [],
            "historical_stage_exposure": "UNRESOLVED_BY_STAGE_IN_ORIGINAL_IDENTITY_MAPS",
            "known_protected_link_found": False, "would_connect_previously_separate_metadata_components": position == 0,
            "must_enter_future_joint_null_connectivity_before_selection_or_freezing": True,
            "scientific_acceptance": False, "joint_independence_complete": False,
        })
    metadata["provenance"] = {"observed_native_header_parents": {"edges": {
        "schema": 1, "status": "SUPPLEMENTAL_NATIVE_RAW_HEADER_PARENT_METADATA", "edges": edges,
        "inputs": [{"path": "original-header-audit.json", "bytes": 100, "sha256": "0" * 64}],
        "boundary": "Observed input metadata; no new roles or scientific acceptance.",
        "scientific_acceptance": False, "joint_independence_complete": False,
    }}}
    null_pool.rows[2].update(linked_existing_scopes="CURRENT_OPENED_DEVELOPMENT",
                             linked_full_witness_component_ids=metadata["trials"][2]["full_witness_component_id"])
    counts = null_pool.specification["counts"]
    counts.update(linked_scope_sources={"CURRENT_OPENED_DEVELOPMENT": 3, "HOLDOUT_VETTING_DISPOSITION_ONLY": 2},
                  linked_scope_components={"CURRENT_OPENED_DEVELOPMENT": 2, "HOLDOUT_VETTING_DISPOSITION_ONLY": 1},
                  component_scope_combinations={"CURRENT_OPENED_DEVELOPMENT;HOLDOUT_VETTING_DISPOSITION_ONLY": 1,
                                               "CURRENT_OPENED_DEVELOPMENT": 1},
                  no_existing_control_link_sources=0, no_existing_control_link_components=0)
    write_pool(null_pool)
    write_scope(null_pool)
    return SimpleNamespace(pool=null_pool, metadata=metadata, edges=edges, trial=metadata["trials"][2],
                           supplement=metadata["provenance"]["observed_native_header_parents"]["edges"])


def test_observed_parents_bind_new_null_links_without_mutating_reader_or_roles(observed_scope):
    before = cloud_null.canonical(observed_scope.metadata)
    cloud_null.check_observed_parent_edges(observed_scope.metadata)
    receipt = cloud.null_pool()
    assert receipt["counts"]["linked_scope_sources"] == {
        "CURRENT_OPENED_DEVELOPMENT": 3, "HOLDOUT_VETTING_DISPOSITION_ONLY": 2}
    assert receipt["counts"]["no_existing_control_link_sources"] == 0
    assert all(receipt[key] is False for key in cloud_null.FLAGS)
    assert before == cloud_null.canonical(observed_scope.metadata)
    assert observed_scope.trial["origin_parent"] == "1000"
    assert observed_scope.trial["resolved_parents"] == ["1000"]


@pytest.mark.parametrize("field,value", [
    ("source_id", "10001"), ("raw_fits_sha256", "0" * 64), ("origin_parent", "30"),
    ("source_domain", "KBonus_LC2"), ("cohort", "periodic_planet"), ("role", "evaluation"),
    ("scope", "CURRENT_CORE_FINAL"), ("additional_parent_identities", []),
    ("additional_parent_identities", ["30", "30"]), ("additional_parent_identities", [30]),
])
def test_observed_parent_trial_misbinding_rejects_rehashed_scope_and_withdraws_receipt(observed_scope, field, value):
    fixture = observed_scope.pool
    cloud.null_pool()
    observed_scope.trial[field] = value
    write_scope(fixture)
    with pytest.raises(ValueError, match="Observed parent"):
        cloud.null_pool()
    assert not (fixture.dump / "cloud-null-pool-receipt.json").exists()


@pytest.mark.parametrize("defect", ["missing", "repeated", "extra_field", "unknown_trial", "mismatched_edge"])
def test_observed_parent_inventory_is_closed_and_unique(observed_scope, defect):
    if defect == "missing":
        observed_scope.edges.pop()
    elif defect == "repeated":
        observed_scope.edges[1] = copy.deepcopy(observed_scope.edges[0])
    elif defect == "extra_field":
        observed_scope.edges[0]["eligibility_pass"] = True
    elif defect == "unknown_trial":
        observed_scope.edges[0].update(source_id="10006", trial_id="native_eb:development:10006")
    else:
        observed_scope.edges[0]["supplemental_identity_edge"] = ["parent", "31"]
    write_scope(observed_scope.pool)
    with pytest.raises(ValueError, match="Observed parent|Repeated observed"):
        cloud.null_pool()


@pytest.mark.parametrize("field,value", [
    ("GAIAID", "Gaia DR3 10001"), ("KEPLERID", 1001), ("TPFORG", 31),
    ("original_declared_parent", 30), ("KEPLERID", True), ("TPFORG", "30"),
    ("actual_header_parent_matches_original_declared", True),
    ("header_origin_binding", "SUBSTITUTED_PARENT"),
])
def test_observed_parent_primary_header_is_bound_to_source_and_both_parents(observed_scope, field, value):
    observed_scope.edges[0]["raw_header"][field] = value
    write_scope(observed_scope.pool)
    with pytest.raises(ValueError, match="Observed parent primary header"):
        cloud.null_pool()


@pytest.mark.parametrize("field,value", [
    ("bytes", True), ("bytes", 0), ("bytes", 512 * 1024**2 + 1),
    ("path", ""), ("sha256", "not a hash"), ("sha256", "0" * 64),
])
def test_observed_parent_raw_descriptor_rejects_unbound_or_invalid_identities(observed_scope, field, value):
    observed_scope.edges[0]["raw_file"][field] = value
    write_scope(observed_scope.pool)
    with pytest.raises(ValueError, match="observed parent|Observed parent raw"):
        cloud.null_pool()


@pytest.mark.parametrize("field", ["scientific_acceptance", "joint_independence_complete"])
@pytest.mark.parametrize("level", ["supplement", "edge"])
def test_observed_parent_boundaries_cannot_promote_acceptance(observed_scope, field, level):
    target = observed_scope.supplement if level == "supplement" else observed_scope.edges[0]
    target[field] = True
    write_scope(observed_scope.pool)
    with pytest.raises(ValueError, match="Observed parent.*boundary"):
        cloud.null_pool()


def test_observed_parent_protected_link_is_preserved_without_qualification_or_acceptance(observed_scope):
    observed_scope.edges[0]["known_protected_link_found"] = True
    write_scope(observed_scope.pool)
    before = cloud_null.canonical(observed_scope.metadata)
    result = cloud.null_pool()
    assert all(result[key] is False for key in cloud_null.FLAGS)
    assert before == cloud_null.canonical(observed_scope.metadata)
    assert observed_scope.edges[0]["known_protected_link_found"] is True


def test_observed_parent_presence_requires_a_complete_supplement(observed_scope):
    observed_scope.metadata["provenance"]["observed_native_header_parents"] = None
    write_scope(observed_scope.pool)
    with pytest.raises(ValueError, match="Missing observed parent"):
        cloud.null_pool()


@pytest.mark.parametrize("defect", ["missing", "wrong_parent", "repeated", "wrong_namespace"])
def test_observed_parent_historical_witness_inventory_is_preserved(observed_scope, defect):
    edge = observed_scope.edges[0]
    witness = {"namespace": "historical-opened-survey-parent-map.csv", "parent_tpf": "30",
               "source_id": "20000", "raw_fits_sha256": "a" * 64, "row": 10}
    edge["historical_reservation_namespace_counts"][witness["namespace"]] = 1
    edge["historical_reservation_identity_rows"] = [witness]
    if defect == "missing":
        edge["historical_reservation_identity_rows"] = []
    elif defect == "wrong_parent":
        witness["parent_tpf"] = "31"
    elif defect == "repeated":
        edge["historical_reservation_identity_rows"].append(dict(witness))
        edge["historical_reservation_namespace_counts"][witness["namespace"]] = 2
    else:
        witness["namespace"] = "unbound-map.csv"
    write_scope(observed_scope.pool)
    with pytest.raises(ValueError, match="observed parent historical|Observed parent historical"):
        cloud.null_pool()


def test_null_pool_cannot_erase_new_observed_parent_links_even_with_rehashed_csv(observed_scope):
    fixture = observed_scope.pool
    fixture.rows[2].update(linked_existing_scopes="", linked_full_witness_component_ids="")
    write_pool(fixture)
    with pytest.raises(ValueError, match="Direct known joint identities"):
        cloud.null_pool()


def test_null_pool_cannot_split_sources_sharing_an_observed_parent(observed_scope):
    fixture = observed_scope.pool
    fixture.rows[1].update(origin_parent="30", metadata_dependency_component_id="9" * 64,
                           linked_existing_scopes="CURRENT_OPENED_DEVELOPMENT",
                           linked_full_witness_component_ids=fixture.rows[2]["linked_full_witness_component_ids"])
    fixture.specification["counts"].update(metadata_dependency_components=3,
        linked_scope_sources={"CURRENT_OPENED_DEVELOPMENT": 3, "HOLDOUT_VETTING_DISPOSITION_ONLY": 1},
        linked_scope_components={"CURRENT_OPENED_DEVELOPMENT": 3, "HOLDOUT_VETTING_DISPOSITION_ONLY": 1},
        component_scope_combinations={"CURRENT_OPENED_DEVELOPMENT;HOLDOUT_VETTING_DISPOSITION_ONLY": 1,
                                     "CURRENT_OPENED_DEVELOPMENT": 2})
    identities = [{"source_id": row["source_id"], "shard": row["shard"],
                   "parent_tpf": row["origin_parent"], "raw_fits_sha256": row["raw_fits_sha256"]} for row in fixture.rows]
    fixture.specification["counts"]["ordered_identity_metadata_sha256"] = sha(cloud_null.canonical(identities))
    write_pool(fixture)
    with pytest.raises(ValueError, match="Known null dependency split"):
        cloud.null_pool()
