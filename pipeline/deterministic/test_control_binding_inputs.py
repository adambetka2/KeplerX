import copy
import json
from pathlib import Path

import pytest

from pipeline.deterministic import control_binding_inputs as prepared
from pipeline.deterministic import control_bindings as bindings


def retained_files(tmp_path):
    raw, arrays = "a" * 64, "b" * 64
    opened = {
        "source_id": "1",
        "parent_tpf": "10",
        "raw_fits_sha256": raw,
        "cloud_arrays_sha256": arrays,
        "cohort": "ordinary",
        "shard": 0,
        "trial_id": "ordinary:development:1",
        "declared_raw_exclusion": False,
    }
    array_path = tmp_path / "1.npz"
    array_path.write_bytes(b"opaque opened array fixture")
    fsc = tmp_path / "1.fsc"
    fsc.write_bytes(b"opaque prepared fixture")
    source = {
        "source_id": "1",
        "parent_tpf": "10",
        "raw_fits_sha256": raw,
        "shard": 0,
        "status": "PREPARED",
        "inputs": {},
        "unavailable_views": {},
    }
    for view in ("psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova"):
        source["inputs"][view] = {
            "view": view,
            "fsc_path": str(fsc),
            "fsc_sha256": bindings.digest(fsc),
        }
    supplied = {
        "arrays": "1.npz",
        "sha256": bindings.digest(array_path),
        "opened_cohort": "ordinary",
        "metadata": {
            "source_id": "1",
            "origin_tpf_kic": 10,
            "raw_fits_sha256": raw,
            "shard": 0,
        },
        "origin": {"original_array": {"sha256": arrays}},
    }
    manifest = {
        "scope": "OPENED_DEVELOPMENT_ONLY",
        "role": "development",
        "original_opened876_selection_unchanged": True,
        "sources": {"1": supplied},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    preparation = {
        "sources": [source],
        "implementation": {},
        "input_manifest": bindings.reference(manifest_path),
    }
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation))
    return (
        opened,
        manifest,
        preparation,
        {
            "manifest": bindings.reference(manifest_path),
            "preparation": bindings.reference(preparation_path),
        },
    )


def test_retained_inputs_bind_exact_bytes_without_calibration_or_remeasurement(
    tmp_path,
):
    opened, _, _, ref = retained_files(tmp_path)
    result = prepared.bind_preparations([opened], [ref], tmp_path)
    assert len(result["input_file_references"]) == 7
    assert len(result["sources"]["1"]["prepared_views"]) == 4
    assert result["sources"]["1"]["current_calibration_case"] is False
    assert result["calibration_cases"] == 0
    assert result["scientific_acceptance"] is False
    assert result["unprovisioned_or_unchecked_source_ids"] == []
    assert result["runs"][0]["scientific_run_repeated"] is False


@pytest.mark.parametrize("changed", ["npz", "fsc", "preparation", "manifest"])
def test_retained_input_hash_change_is_rejected(tmp_path, changed):
    opened, _, _, ref = retained_files(tmp_path)
    path = (
        tmp_path
        / {
            "npz": "1.npz",
            "fsc": "1.fsc",
            "preparation": "preparation.json",
            "manifest": "manifest.json",
        }[changed]
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepared.bind_preparations([opened], [ref], tmp_path)


@pytest.mark.parametrize(
    "changed",
    ["role", "scope", "source", "parent", "raw", "original_array", "protected"],
)
def test_retained_bindings_reject_forbidden_scope_or_identity(tmp_path, changed):
    opened, manifest, preparation, _ = retained_files(tmp_path)
    if changed in {"role", "scope"}:
        manifest[changed] = "evaluation"
    elif changed == "source":
        preparation["sources"][0]["source_id"] = "2"
    elif changed == "parent":
        manifest["sources"]["1"]["metadata"]["origin_tpf_kic"] = 20
    elif changed == "raw":
        manifest["sources"]["1"]["metadata"]["raw_fits_sha256"] = "c" * 64
    elif changed == "original_array":
        manifest["sources"]["1"]["origin"]["original_array"]["sha256"] = "c" * 64
    else:
        opened["declared_raw_exclusion"] = True
    manifest_path, preparation_path = (
        tmp_path / "manifest.json",
        tmp_path / "preparation.json",
    )
    manifest_path.write_text(json.dumps(manifest))
    preparation["input_manifest"] = bindings.reference(manifest_path)
    preparation_path.write_text(json.dumps(preparation))
    ref = {
        "manifest": bindings.reference(manifest_path),
        "preparation": bindings.reference(preparation_path),
    }
    with pytest.raises(ValueError):
        prepared.bind_preparations([opened], [ref], tmp_path)


def test_retained_repeated_sources_and_missing_views_are_rejected(tmp_path):
    opened, _, preparation, ref = retained_files(tmp_path)
    with pytest.raises(ValueError, match="repeats an opened source"):
        prepared.bind_preparations([opened], [ref, ref], tmp_path)
    del preparation["sources"][0]["inputs"]["psf_nova"]
    Path(ref["preparation"]["path"]).write_text(json.dumps(preparation))
    ref["preparation"] = bindings.reference(ref["preparation"]["path"])
    with pytest.raises(ValueError, match="view inventory"):
        prepared.bind_preparations([opened], [ref], tmp_path)


def test_frozen_implementation_changes_are_explicit_and_paths_are_bounded(tmp_path):
    path = tmp_path / "pipeline/code.py"
    path.parent.mkdir()
    path.write_text("changed source code")
    current, differences = prepared.implementation_differences(
        {"pipeline/code.py": "a" * 64}, tmp_path
    )
    assert differences["pipeline/code.py"] == {
        "measured": "a" * 64,
        "current": current["pipeline/code.py"],
    }
    with pytest.raises(ValueError, match="escapes pipeline"):
        prepared.implementation_differences({"../outside.py": "a" * 64}, tmp_path)


def receipt_files(tmp_path, monkeypatch):
    input_path = tmp_path / "metadata.json"
    input_path.write_text("{}")
    ref = bindings.reference(input_path)
    monkeypatch.setattr(bindings, "PARENT_INDEX_SHA256", ref["sha256"])
    recipe_path = tmp_path / "inputs.json"
    recipe = {
        "schema": 1,
        "status": "CONTROL_BINDING_PREPARATION_INPUTS",
        "inputs": {name: ref for name in bindings.INPUT_NAMES},
    }
    recipe_path.write_text(json.dumps(recipe))
    out = tmp_path / "bound"
    names = {f"{name}.json" for name in bindings.REPORT_NAMES}
    names.update(
        f"scopes/{scope.lower()}--{cohort}--{role}.json"
        for (scope, role), cohorts in bindings.LAUNCH_SCOPES.items()
        for cohort in cohorts
    )
    for name in names:
        value = (
            {"objective": []}
            if name == "calibration.json"
            else {"input_file_references": []}
        )
        bindings.write_json(out / name, value)
    receipt = {
        "schema": 1,
        "status": "CONTROL_BINDINGS_PREPARATION_ONLY",
        **bindings.BOUNDARY,
        "output_directory": str(out),
        "inputs_recipe": bindings.reference(recipe_path),
        "inputs": recipe["inputs"],
        "dependent_inputs": [],
        "programs": [],
        "outputs": [bindings.reference(out / name) for name in sorted(names)],
        "declared_trials": 9007,
        "opened_identities": 876,
        "prospective_candidates": {"guide_star_family": 100, "contamination_pair": 100},
        "complete_calibration_cases": 0,
    }
    path = out / "readiness.json"
    for name in bindings.PROGRAM_NAMES:
        program = tmp_path / "pipeline" / name
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_text("opaque measured program fixture")
        receipt["programs"].append(bindings.reference(program))
    bindings.write_json(path, receipt)
    return path, receipt, recipe_path


def test_closed_report_rejects_post_assembly_input_and_output_drift(
    tmp_path, monkeypatch
):
    path, receipt, recipe_path = receipt_files(tmp_path, monkeypatch)
    assert bindings.verify(path)["status"] == "CONTROL_BINDING_FILES_UNCHANGED"
    recipe_path.write_text(recipe_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        bindings.verify(path)
    recipe_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "CONTROL_BINDING_PREPARATION_INPUTS",
                "inputs": receipt["inputs"],
            }
        )
    )
    output = Path(receipt["outputs"][0]["path"])
    output.write_text(output.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        bindings.verify(path)


@pytest.mark.parametrize(
    "changed",
    [
        "boundary",
        "count",
        "omitted_output",
        "duplicate_output",
        "dependent_input",
        "omitted_program",
    ],
)
def test_closed_report_cannot_drop_members_or_claim_acceptance(
    tmp_path, monkeypatch, changed
):
    path, receipt, _ = receipt_files(tmp_path, monkeypatch)
    if changed == "boundary":
        receipt["joint_independence_complete"] = True
    elif changed == "count":
        receipt["complete_calibration_cases"] = 100
    elif changed == "omitted_output":
        receipt["outputs"].pop()
    elif changed == "duplicate_output":
        receipt["outputs"].append(copy.deepcopy(receipt["outputs"][0]))
    elif changed == "dependent_input":
        receipt["dependent_inputs"] = [receipt["inputs_recipe"]]
    else:
        receipt["programs"].pop()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        bindings.verify(path)
