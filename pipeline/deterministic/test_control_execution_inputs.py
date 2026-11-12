import json

import numpy as np
import pytest

from pipeline.deterministic import control_execution_inputs as execution
from pipeline.deterministic import run as runner
from pipeline.deterministic.control_bindings import digest


@pytest.fixture
def inputs(tmp_path):
    arrays = tmp_path / "1.npz"
    np.savez_compressed(arrays, source_id=np.array([1]), time_bkjd=np.array([120.0]))
    original = {
        "source_id": "1",
        "origin_parent": "7",
        "raw_fits_sha256": "a" * 64,
        "cohort": "native_eb",
        "trial_id": "native_eb:development:1",
        "full_witness_component_id": "component-1",
        "original_cloud_arrays_sha256": digest(arrays),
    }
    row = {
        "arrays": "1.npz",
        "sha256": digest(arrays),
        "opened_cohort": "native_eb",
        "metadata": {
            "source_id": "1",
            "origin_tpf_kic": 7,
            "raw_fits_sha256": "a" * 64,
        },
    }
    manifest = {
        "schema": 1,
        "role": "development",
        "measurement_scope": "OPENED_DEVELOPMENT_ONLY",
        "sources": {"1": row},
    }
    path = tmp_path / "input-manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, original


def test_real_input_bytes_bind_to_original_identity(inputs):
    path, _, original = inputs
    result = execution.bind_manifest(path, {"1": original}, set(), set())
    assert result["bound"]["1"]["trial_id"] == original["trial_id"]
    assert result["bound"]["1"]["array_proof"]["execution_ready"] is True


@pytest.mark.parametrize(
    "change", ["role", "scope", "source", "parent", "raw", "cohort"]
)
def test_changed_scope_or_original_identity_rejected(inputs, change):
    path, manifest, original = inputs
    row = manifest["sources"]["1"]
    if change == "role":
        manifest["role"] = "evaluation"
    elif change == "scope":
        manifest["measurement_scope"] = "SEALED"
    elif change == "source":
        row["metadata"]["source_id"] = "2"
    elif change == "parent":
        row["metadata"]["origin_tpf_kic"] = 9
    elif change == "raw":
        row["metadata"]["raw_fits_sha256"] = "b" * 64
    else:
        row["opened_cohort"] = "ordinary"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        execution.bind_manifest(path, {"1": original}, set(), set())


def test_protected_raw_overlap_rejected_before_array_access(inputs):
    path, _, original = inputs
    path.with_name("1.npz").unlink()
    with pytest.raises(ValueError, match="protected sealed raw"):
        execution.bind_manifest(path, {"1": original}, {"1"}, set())


def test_changed_array_with_stale_manifest_hash_rejected(inputs):
    path, _, original = inputs
    path.with_name("1.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input unavailable or changed"):
        execution.bind_manifest(path, {"1": original}, set(), set())


def test_changed_array_and_updated_manifest_hash_do_not_prove_original_content(inputs):
    path, manifest, original = inputs
    np.savez_compressed(
        path.with_name("1.npz"), source_id=np.array([1]), time_bkjd=np.array([121.0])
    )
    manifest["sources"]["1"]["sha256"] = digest(path.with_name("1.npz"))
    path.write_text(json.dumps(manifest))
    result = execution.bind_manifest(path, {"1": original}, set(), set())
    assert result["bound"]["1"]["array_proof"]["execution_ready"] is False
    scope = {
        "scope": "CURRENT_OPENED_DEVELOPMENT",
        "cohort": "native_eb",
        "trials": [original],
    }
    assert (
        execution.execution_slice(scope, result, path.parent / "out", set(), set())
        is None
    )


def test_unprovisioned_and_sealed_sources_stay_in_denominator(inputs):
    path, _, original = inputs
    result = execution.bind_manifest(path, {"1": original}, set(), set())
    missing = {**original, "source_id": "2", "trial_id": "native_eb:development:2"}
    scope = {"scope": "CURRENT_OPENED_DEVELOPMENT", "trials": [original, missing]}
    rows = execution.scope_sources(scope, [result], set(), set())
    assert len(rows) == 2 and rows[1]["status"] == "INPUTS_NOT_PROVISIONED"
    scope["scope"] = "CURRENT_CORE_FINAL"
    rows = execution.scope_sources(scope, [result], set(), set())
    assert len(rows) == 2 and all(not row["input_variants"] for row in rows)
    assert all(
        row["status"] == "IDENTITY_ONLY_NOT_AUTHORIZED_FOR_THIS_DEVELOPMENT_EXECUTION"
        for row in rows
    )


def test_development_slice_retains_original_scope_order_and_requires_development(
    inputs,
):
    path, _, original = inputs
    result = execution.bind_manifest(path, {"1": original}, set(), set())
    scope = {
        "scope": "CURRENT_OPENED_DEVELOPMENT",
        "cohort": "native_eb",
        "trials": [original, {**original, "source_id": "2"}],
    }
    slice_ = execution.execution_slice(scope, result, path.parent / "out", set(), set())
    assert slice_["selected_sources"] == 1 and slice_["declared_cohort_sources"] == 2
    assert slice_["runtime_policy_required"] == "DEVELOPMENT"
    assert slice_["final_scoped_role_authorization"] is False


def test_multi_source_slice_is_consumable_by_actual_runner(inputs):
    path, manifest, original = inputs
    arrays = path.parent / "9007199254740993.npz"
    np.savez_compressed(
        arrays, source_id=np.array([9007199254740993]), time_bkjd=np.array([120.0])
    )
    second = {
        **original,
        "source_id": "9007199254740993",
        "trial_id": "native_eb:development:9007199254740993",
        "full_witness_component_id": "component-2",
        "original_cloud_arrays_sha256": digest(arrays),
    }
    manifest["sources"][second["source_id"]] = {
        **manifest["sources"]["1"],
        "arrays": arrays.name,
        "sha256": digest(arrays),
        "metadata": {
            **manifest["sources"]["1"]["metadata"],
            "source_id": second["source_id"],
        },
    }
    for row in manifest["sources"].values():
        row["metadata"]["shard"] = 0
    path.write_text(json.dumps(manifest))
    bound = execution.bind_manifest(
        path, {"1": original, second["source_id"]: second}, set(), set()
    )
    scope = {
        "scope": "CURRENT_OPENED_DEVELOPMENT",
        "cohort": "native_eb",
        "trials": [second, original],
    }
    slice_ = execution.execution_slice(scope, bound, path.parent / "out", set(), set())
    ids = runner.read(slice_["source_ids"]["path"])
    assert ids == [second["source_id"], "1"]
    assert all(isinstance(sid, str) for sid in ids)
    selected = runner.select_sources([0], 2, ids, experimental_manifest=path)
    assert [row["source_id"] for row in selected] == ids


def test_duplicate_source_keys_are_rejected(inputs):
    path, manifest, original = inputs
    text = json.dumps(manifest)
    row = json.dumps(manifest["sources"]["1"])
    path.write_text(text.replace(f'"1": {row}', f'"1": {row}, "1": {row}'))
    with pytest.raises(ValueError, match="Duplicate JSON identity"):
        execution.bind_manifest(path, {"1": original}, set(), set())


def test_shared_component_refused_before_any_manifest_array_access(inputs, monkeypatch):
    path, manifest, original = inputs
    manifest["sources"]["2"] = {
        **manifest["sources"]["1"],
        "metadata": {**manifest["sources"]["1"]["metadata"], "source_id": "2"},
    }
    path.write_text(json.dumps(manifest))
    identities = {
        "1": original,
        "2": {**original, "source_id": "2", "full_witness_component_id": "shared"},
    }
    calls = []
    monkeypatch.setattr(execution, "array_reference", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="stopped/shared protected component"):
        execution.bind_manifest(path, identities, set(), {"shared"})
    assert calls == []


@pytest.mark.parametrize("protection", ["raw", "component", "scope"])
def test_execution_slice_refuses_protected_binding_without_emitting(inputs, protection):
    path, _, original = inputs
    manifest = execution.bind_manifest(path, {"1": original}, set(), set())
    scope = {
        "scope": "CURRENT_CORE_FINAL"
        if protection == "scope"
        else "CURRENT_OPENED_DEVELOPMENT",
        "cohort": "native_eb",
        "trials": [original],
    }
    out = path.parent / "out"
    with pytest.raises(ValueError):
        execution.execution_slice(
            scope,
            manifest,
            out,
            {"1"} if protection == "raw" else set(),
            {"component-1"} if protection == "component" else set(),
        )
    assert not out.exists()


@pytest.fixture
def current_scopes():
    import gzip
    from collections import defaultdict
    from pathlib import Path

    path = Path(execution.__file__).with_name("cloud-joint-scope.json.gz")
    graph = json.loads(gzip.decompress(path.read_bytes()))
    groups = defaultdict(list)
    for trial in graph["trials"]:
        groups[trial["scope"], trial["role"], trial["cohort"]].append(trial)
    return [
        (
            path,
            {"scope": scope, "role": role, "cohort": cohort, "trials": trials},
        )
        for (scope, role, cohort), trials in groups.items()
    ]


def test_current_graph_preserves_all_five_shared_opened_trials(current_scopes):
    stopped, receipt = execution.component_exclusions(current_scopes)
    assert receipt["canonical_json_sha256"] == (
        "aff2e0baea45bd89dd4e9b36eb93e4e441a3310ad96c15f2f7cbe6cd7ce1049f"
    )
    rows = [
        row
        for _, scope in current_scopes
        if scope["scope"] == "CURRENT_OPENED_DEVELOPMENT"
        for row in execution.scope_sources(scope, [], set(), stopped)
    ]
    excluded = [r for r in rows if r["status"] == "EXCLUDED_STOPPED_SHARED_COMPONENT"]
    assert len(rows) == 876
    assert {r["source_id"] for r in excluded} == {
        "2051287276908976000",
        "2086582218628879616",
        "2107266540612318848",
        "2130351298336248192",
        "2132455561730848384",
    }
    assert all(not row["input_variants"] for row in excluded)


@pytest.mark.parametrize(
    "sid",
    [
        "2051287276908976000",
        "2086582218628879616",
        "2107266540612318848",
        "2130351298336248192",
        "2132455561730848384",
    ],
)
def test_each_shared_opened_source_refused_even_without_raw_id_exclusion(
    inputs, current_scopes, sid
):
    stopped, _ = execution.component_exclusions(current_scopes)
    original = next(
        row
        for _, scope in current_scopes
        if scope["scope"] == "CURRENT_OPENED_DEVELOPMENT"
        for row in scope["trials"]
        if row["source_id"] == sid
    )
    path, manifest, _ = inputs
    row = manifest["sources"].pop("1")
    row["metadata"]["source_id"] = sid
    manifest["sources"][sid] = row
    path.write_text(json.dumps(manifest))
    path.with_name("1.npz").unlink()
    with pytest.raises(ValueError, match="stopped/shared protected component"):
        execution.bind_manifest(path, {sid: original}, set(), stopped)


def test_changed_reviewed_component_cannot_escape_current_graph(current_scopes):
    current_scopes[0][1]["trials"][0]["full_witness_component_id"] = "detached"
    with pytest.raises(ValueError, match="current full component graph"):
        execution.component_exclusions(current_scopes)


@pytest.fixture
def oracle(inputs, monkeypatch):
    import hashlib

    path, _, original = inputs
    with np.load(path.with_name("1.npz"), allow_pickle=False) as saved:
        arrays = {
            name: {
                "dtype": str(saved[name].dtype),
                "shape": list(saved[name].shape),
                "bytes": saved[name].nbytes,
                "sha256": hashlib.sha256(saved[name].tobytes()).hexdigest(),
            }
            for name in saved.files
        }
    value = {
        "sources": {
            "1": {
                "opened_scope": "CURRENT_OPENED_DEVELOPMENT",
                "opened_cohort": "native_eb",
                "metadata": {"source_id": "1", "origin_tpf_kic": 7},
                "sha256": original["raw_fits_sha256"],
                "arrays": arrays,
            }
        }
    }
    oracle_path = path.with_name("oracle.json")
    oracle_path.write_text(json.dumps(value))
    monkeypatch.setattr(execution, "CONTROL_SPEC_SHA", digest(oracle_path))
    return oracle_path, value


def test_approved_array_hashes_allow_different_npz_encoding(inputs, oracle):
    path, _, original = inputs
    oracle_path, _ = oracle
    np.savez(
        path.with_name("1.npz"), source_id=np.array([1]), time_bkjd=np.array([120.0])
    )
    assert digest(path.with_name("1.npz")) != original["original_cloud_arrays_sha256"]
    proof = execution.control_array_proof(
        "1", original, execution.reference(path.with_name("1.npz")), oracle_path
    )
    assert proof["execution_ready"] is True and proof["array_count"] == 2


def test_modified_data_cannot_pass_original_array_oracle(inputs, oracle):
    path, _, original = inputs
    oracle_path, _ = oracle
    np.savez(
        path.with_name("1.npz"), source_id=np.array([1]), time_bkjd=np.array([121.0])
    )
    with pytest.raises(ValueError):
        execution.control_array_proof(
            "1", original, execution.reference(path.with_name("1.npz")), oracle_path
        )


def test_changed_original_oracle_is_rejected(inputs, oracle):
    path, _, original = inputs
    oracle_path, value = oracle
    value["sources"]["1"]["opened_cohort"] = "periodic_planet"
    oracle_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="oracle changed"):
        execution.control_array_proof(
            "1", original, execution.reference(path.with_name("1.npz")), oracle_path
        )


@pytest.mark.parametrize("field", ["opened_scope", "opened_cohort", "sha256"])
def test_original_oracle_identity_cannot_change(inputs, oracle, monkeypatch, field):
    path, _, original = inputs
    oracle_path, value = oracle
    value["sources"]["1"][field] = "changed"
    oracle_path.write_text(json.dumps(value))
    monkeypatch.setattr(execution, "CONTROL_SPEC_SHA", digest(oracle_path))
    with pytest.raises(ValueError, match="source/parent/raw/cohort differs"):
        execution.control_array_proof(
            "1", original, execution.reference(path.with_name("1.npz")), oracle_path
        )
