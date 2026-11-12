"""Scoped evaluation must reject foreign inputs before scientific computation."""

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

import evaluation_scope as scope
import runtime
from deterministic import run as adapter


def save(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def calibrated_test_lanes(tmp_path):
    from deterministic.signal_contracts import LANE_CONTRACT

    thresholds = {name: 8.0 for name in ("periodic", "shallow", "sparse")}
    calibration = save(tmp_path / "lanes.json", {
        "status": "CALIBRATED", "measurement_scope": "OPENED_DEVELOPMENT_NULLS",
        "lane_contract": LANE_CONTRACT, "thresholds": thresholds,
    })
    return {
        "contract": LANE_CONTRACT, "status": "CALIBRATED",
        "thresholds": thresholds, "calibration": calibration,
    }


@pytest.fixture
def frozen_scientific_settings(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    config = adapter.read(ROOT / "pipeline/vetting/config.json")
    config.update(quality_bitmask=-1, discovery_contract=adapter.detector_contract())
    policy = adapter.read(ROOT / "pipeline/vetting/disposition-policy.json")
    policy.update(
        status="FROZEN_EVALUATION",
        boi_catalogue_enabled=True,
        assessment_rules=copy.deepcopy(config["triage"]["assessment_rules"]),
        implementation={},
        pipeline_contract=copy.deepcopy(vetting.pipeline_contract(config)),
        evaluation_binding={
            **{
                name: "a" * 64
                for name in ("run_spec_sha256", "detector_sha256", "calibration_sha256")
            },
            **{
                role: {
                    "controls_sha256": "b" * 64,
                    "cohorts": [{"path": "dump/test-cohort.json", "sha256": "c" * 64}],
                }
                for role in ("development_replay", "final_replay")
            },
        },
        lane_gates=calibrated_test_lanes(tmp_path),
    )
    policy["definition_sha256"] = runtime.disposition_definition_sha256(policy)
    policy_path = tmp_path / "policy.json"
    save(policy_path, policy)
    original_read = adapter.read
    monkeypatch.setattr(adapter, "implementation_hashes", dict)
    monkeypatch.setattr(
        adapter,
        "read",
        lambda path: (
            copy.deepcopy(config)
            if Path(path) == ROOT / "pipeline/vetting/config.json"
            else original_read(path)
        ),
    )
    return policy, config, policy_path, SimpleNamespace(disposition_policy=policy_path)


@pytest.mark.parametrize("status", ["FROZEN_EVALUATION", "RELEASED"])
@pytest.mark.parametrize(
    "keys,value",
    [
        (("timing_refinement_max_period_days",), 4),
        (("timing_refinement_max_units",), 32),
        (("timing_refinement_grid_points",), 63),
        (("timing_refinement_minimum_event_train_positive_z",), -5),
        (("short_period_shape_max_days",), 0.4),
        (("short_period_joint_fit_max_units",), 512),
        (("short_period_joint_fit_max_missing_fraction",), 0.1),
        (("parent_index_sha256",), "d" * 64),
        (("alias", "max_eb_rows_per_signal"), 5),
        (("alias", "minimum_supplied_event_alignment"), 0.9),
        (("alias", "maximum_single_event_information_fraction"), 0.6),
        (("triage", "enabled"), False),
        (("triage", "shape_transit_bic_margin"), 20),
        (("triage", "assessment_rules", "instrument", "spatial_null_bic_margin"), 20),
    ],
)
def test_scientific_config_changes_stop_before_scope_or_release_launch(
    frozen_scientific_settings, monkeypatch, status, keys, value
):
    policy, config, path, args = frozen_scientific_settings
    policy["status"] = status
    save(path, policy)
    changed = config
    for key in keys[:-1]:
        changed = changed[key]
    changed[keys[-1]] = value
    monkeypatch.setattr(
        adapter,
        "authorize",
        lambda *args: pytest.fail("Changed settings reached scope authorization"),
    )
    vetting = importlib.import_module("vetting")
    monkeypatch.setattr(
        vetting,
        "validate_release_contract",
        lambda *args: pytest.fail("Changed settings reached release launch"),
    )
    with pytest.raises(ValueError, match="pipeline configuration changed"):
        adapter.evaluation_preflight(args, [])


@pytest.mark.parametrize("defect", [
    "missing", "reordered", "instrument_dismissal", "last_failure",
])
def test_rehashed_policy_requires_the_implemented_triage_contract(
    frozen_scientific_settings, monkeypatch, defect
):
    policy, _, path, args = frozen_scientific_settings
    vetting = importlib.import_module("vetting")
    contract = policy.setdefault("triage_contract", {})
    if defect == "missing":
        del policy["triage_contract"]
    elif defect == "reordered":
        contract["stage_order"] = [
            "orbital", "shape", "instrument", "localization", "contamination",
        ]
    elif defect == "instrument_dismissal":
        decisions = contract.setdefault("allowed_decisions", {}).setdefault(
            "instrument", ["PASS", "CONTINUE_INCONCLUSIVE"]
        )
        decisions.append("DISMISS")
    else:
        contract["state_rule"] = "LAST_NONPASS_IN_STAGE_ORDER"
    policy["definition_sha256"] = runtime.disposition_definition_sha256(policy)
    save(path, policy)
    monkeypatch.setattr(
        adapter, "authorize",
        lambda *args: pytest.fail("Invalid triage contract reached authorization"),
    )
    with pytest.raises(ValueError, match="triage contract"):
        vetting.validate_operational_policy(policy)
    with pytest.raises(ValueError, match="triage contract"):
        adapter.evaluation_preflight(args, [])


def test_worker_and_verified_cache_locations_do_not_change_scientific_contract(
    frozen_scientific_settings, monkeypatch
):
    _, config, _, args = frozen_scientific_settings
    config.update(
        workers=1,
        pixel_cache_directory="dump/other-pixel-cache",
        prf_cache_directory="dump/other-prf-cache",
    )
    config["alias"]["workers"] = 1
    monkeypatch.setattr(adapter, "authorize", lambda *args: "development_replay")
    monkeypatch.setattr(adapter, "match_cohort", lambda *args: {"scope": "accepted"})
    assert adapter.evaluation_preflight(args, []) == {"scope": "accepted"}


@pytest.fixture
def scoped_policy(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    sources = [{"source_id": "123", "parent_tpf": "45", "raw_fits_sha256": "a" * 64}]
    other = [{"source_id": "456", "parent_tpf": "67", "raw_fits_sha256": "b" * 64}]
    manifest = save(
        tmp_path / "input-manifest.json",
        {"schema": 1, "sources": {"123": {"metadata": sources[0]}}},
    )
    refs = {}
    for role, identities in (("development_replay", other), ("final_replay", sources)):
        refs[role] = save(
            tmp_path / f"{role}.json",
            {
                "schema": 1,
                "status": "FROZEN_EVALUATION_COHORT",
                "role": role,
                "cohort_id": role,
                "sources": identities,
                "input_manifest": manifest,
            },
        )
    policy = {
        "schema": 3,
        "triage_contract": adapter.read(
            ROOT / "pipeline/vetting/disposition-policy.json"
        )["triage_contract"],
        "policy_id": "deterministic-boi-v1",
        "status": "FROZEN_EVALUATION",
        "boi_catalogue_enabled": True,
        "assessment_rules": adapter.read(ROOT / "pipeline/vetting/config.json")[
            "triage"
        ]["assessment_rules"],
        "implementation": {},
        "pipeline_contract": {},
        "lane_gates": calibrated_test_lanes(tmp_path),
        "evaluation_binding": {
            "run_spec_sha256": "c" * 64,
            "detector_sha256": "d" * 64,
            "calibration_sha256": "e" * 64,
            **{
                role: {"controls_sha256": "f" * 64, "cohorts": [ref]}
                for role, ref in refs.items()
            },
        },
    }
    policy["definition_sha256"] = runtime.disposition_definition_sha256(policy)
    monkeypatch.setattr(adapter, "implementation_hashes", dict)
    monkeypatch.setattr(vetting, "pipeline_contract", lambda config: {})
    monkeypatch.setenv(
        "SPACELAB_DISPOSITION_EVALUATION_TOKEN",
        runtime.disposition_evaluation_token(policy, "final_replay"),
    )
    return (
        policy,
        sources,
        Path(manifest["path"]),
        save(tmp_path / "policy.json", policy),
    )


def test_scoped_token_survives_computed_runtime_flags(scoped_policy):
    policy, _, _, _ = scoped_policy
    token = runtime.disposition_evaluation_token(policy, "final_replay")
    policy.update(_evaluation_authorized=True, _evaluation_role="final_replay")
    assert scope.authorize(policy, token) == "final_replay"


def test_released_policy_is_checked_before_detector_entry(scoped_policy, monkeypatch):
    policy, sources, manifest, reference = scoped_policy
    policy["status"] = "RELEASED"
    save(Path(reference["path"]), policy)
    vetting = importlib.import_module("vetting")
    calls = []

    def reject(value):
        calls.append(value)
        raise ValueError("Released audit evidence changed")

    monkeypatch.setattr(vetting, "validate_release_contract", reject)
    with pytest.raises(ValueError, match="Released audit evidence changed"):
        adapter.evaluation_preflight(
            SimpleNamespace(disposition_policy=Path(reference["path"])),
            sources,
            manifest,
        )
    assert calls == [policy]


def test_valid_released_policy_does_not_impose_an_evaluation_cohort(
    scoped_policy, monkeypatch
):
    policy, sources, manifest, reference = scoped_policy
    policy["status"] = "RELEASED"
    save(Path(reference["path"]), policy)
    vetting = importlib.import_module("vetting")
    monkeypatch.setattr(vetting, "validate_release_contract", lambda value: None)
    assert (
        adapter.evaluation_preflight(
            SimpleNamespace(disposition_policy=Path(reference["path"])),
            sources,
            manifest,
        )
        is None
    )


def test_complete_evaluation_preflight_accepts_frozen_cohort(scoped_policy):
    policy, sources, manifest, reference = scoped_policy
    matched = adapter.evaluation_preflight(
        SimpleNamespace(disposition_policy=Path(reference["path"])), sources, manifest
    )
    assert matched["evaluation_role"] == "final_replay"
    assert matched["definition_sha256"] == policy["definition_sha256"]


def test_changed_policy_definition_rejected_after_preparation(
    scoped_policy, tmp_path, monkeypatch
):
    policy, sources, manifest, reference = scoped_policy
    args = SimpleNamespace(
        disposition_policy=Path(reference["path"]),
        run=tmp_path,
        binary=tmp_path / "binary",
        threads=1,
    )
    prepared = adapter.evaluation_preflight(args, sources, manifest)
    policy["assessment_rules"]["instrument"]["spatial_null_bic_margin"] += 1
    policy["definition_sha256"] = runtime.disposition_definition_sha256(policy)
    save(Path(reference["path"]), policy)
    monkeypatch.setenv(
        "SPACELAB_DISPOSITION_EVALUATION_TOKEN",
        runtime.disposition_evaluation_token(policy, "final_replay"),
    )
    monkeypatch.setattr(adapter, "dump_path", Path)
    monkeypatch.setattr(
        adapter,
        "validate_preparation",
        lambda _: {
            "sources": sources,
            "input_manifest": {"path": str(manifest)},
            "evaluation_scope": prepared,
            "disposition_policy": reference["path"],
            "disposition_policy_sha256": reference["sha256"],
        },
    )
    monkeypatch.setattr(
        adapter,
        "run_command",
        lambda *args: pytest.fail("Changed-policy detector execution started"),
    )
    with pytest.raises(ValueError, match="policy differs from preparation"):
        adapter.scan(args)


@pytest.mark.parametrize(
    "mutation", ["source", "parent", "raw", "duplicate", "manifest", "role"]
)
def test_foreign_cohort_rejected_even_with_valid_role_token(
    scoped_policy, tmp_path, mutation
):
    policy, sources, manifest, _ = scoped_policy
    rows = copy.deepcopy(sources)
    role = "final_replay"
    if mutation == "source":
        rows[0]["source_id"] = "456"
    elif mutation == "parent":
        rows[0]["parent_tpf"] = "999"
    elif mutation == "raw":
        rows[0]["raw_fits_sha256"] = "0" * 64
    elif mutation == "duplicate":
        rows *= 2
    elif mutation == "manifest":
        other = tmp_path / "other.json"
        other.write_bytes(manifest.read_bytes())
        manifest = other
    else:
        role = "development_replay"
    with pytest.raises(ValueError, match="scope|identity"):
        scope.match_cohort(policy, role, rows, manifest)


def test_changed_frozen_manifest_rejected(scoped_policy):
    policy, sources, manifest, _ = scoped_policy
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        scope.match_cohort(policy, "final_replay", sources, manifest)


def test_scoped_queue_accepts_only_declared_identity_subset(scoped_policy):
    policy, sources, manifest, _ = scoped_policy
    matched = scope.match_cohort(policy, "final_replay", sources, manifest, subset=True)
    assert matched["evaluation_role"] == "final_replay"
    assert (
        scope.match_cohort(policy, "final_replay", [], manifest, subset=True) == matched
    )
    foreign = [{**sources[0], "source_id": "999"}]
    with pytest.raises(ValueError, match="authorized scope"):
        scope.match_cohort(policy, "final_replay", foreign, manifest, subset=True)


@pytest.mark.parametrize("stage", ["prepare", "scan", "survey"])
def test_foreign_scope_rejected_before_detector_or_flux_preparation(
    scoped_policy, tmp_path, monkeypatch, stage
):
    _, sources, manifest, reference = scoped_policy
    foreign = [{**sources[0], "parent_tpf": "999"}]
    monkeypatch.setattr(adapter, "select_sources", lambda *args, **kwargs: foreign)
    monkeypatch.setattr(adapter, "dump_path", Path)
    monkeypatch.setattr(
        adapter,
        "validate_preparation",
        lambda out: {
            "sources": foreign, "input_manifest": {"path": str(manifest)},
            "disposition_policy": reference["path"],
            "disposition_policy_sha256": reference["sha256"],
        },
    )

    def unexpected(*args, **kwargs):
        pytest.fail(
            "Scientific computation began before the evaluation scope was checked"
        )

    for name in (
        "prepare_source",
        "prepare_discovery_source",
        "run_command",
        "start_survey",
        "resume_survey",
    ):
        monkeypatch.setattr(adapter, name, unexpected)
    out = tmp_path / "run"
    args = SimpleNamespace(
        shards=[1],
        limit=1,
        source_ids=None,
        input_manifest=manifest,
        disposition_policy=Path(reference["path"]),
        out=out,
        run=out,
        threads=1,
        binary=tmp_path / "binary",
        batch_size=1,
        resume=False,
    )
    with pytest.raises(ValueError, match="authorized scope"):
        getattr(adapter, stage)(args)
    assert not out.exists()


def test_unauthorized_role_rejected_before_preparation(
    scoped_policy, tmp_path, monkeypatch
):
    _, sources, manifest, reference = scoped_policy
    monkeypatch.setattr(adapter, "select_sources", lambda *args, **kwargs: sources)
    monkeypatch.setenv("SPACELAB_DISPOSITION_EVALUATION_TOKEN", "incorrect")
    with pytest.raises(ValueError, match="valid scoped role token"):
        adapter.prepare(
            SimpleNamespace(
                shards=[1],
                limit=1,
                source_ids=None,
                input_manifest=manifest,
                disposition_policy=Path(reference["path"]),
                out=tmp_path / "run",
            )
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("stage", ["prepare", "check_run"])
def test_foreign_vetting_queue_rejected_before_network_or_measurement(
    scoped_policy, tmp_path, monkeypatch, stage
):
    policy, _, manifest, _ = scoped_policy
    vetting = importlib.import_module("vetting")
    config = tmp_path / "config.json"
    save(
        config, {"queue": str(tmp_path / "queue.json"), "input_manifest": str(manifest)}
    )
    monkeypatch.setattr(vetting, "CONFIG", config)
    monkeypatch.setattr(vetting, "load_disposition_policy", lambda: policy)
    monkeypatch.setattr(vetting, "read_queue", lambda path: [{"source_id": "123"}])
    monkeypatch.setattr(
        vetting,
        "queue_source_identities",
        lambda *args: [
            {"source_id": "123", "parent_tpf": "999", "raw_fits_sha256": "a" * 64}
        ],
    )

    def unexpected(*args):
        pytest.fail("Vetting began before the evaluation queue was checked")

    monkeypatch.setattr(vetting, "freeze_catalogues", unexpected)
    with pytest.raises(ValueError, match="authorized scope"):
        if stage == "prepare":
            vetting.prepare()
        else:
            vetting.check_run(tmp_path / "queue.json")


def bind_null_recipe(policy, tmp_path, sources, manifest, variant="inverted"):
    from deterministic.signal_contracts import NULL_CONTRACT

    recipe = {"id": NULL_CONTRACT, "variant": variant, "seed": 5}
    reference = save(tmp_path / "null-cohort.json", {
        "schema": 1, "status": "FROZEN_EVALUATION_COHORT", "role": "final_replay",
        "cohort_id": "sealed-nulls", "sources": sources,
        "input_manifest": {
            "path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "null_construction": recipe,
    })
    policy["evaluation_binding"]["final_replay"]["cohorts"] = [reference]
    policy["definition_sha256"] = runtime.disposition_definition_sha256(policy)
    return recipe


def test_sealed_null_scope_requires_exact_frozen_recipe_and_construction_receipt(
    scoped_policy, tmp_path
):
    policy, sources, manifest, _ = scoped_policy
    recipe = bind_null_recipe(policy, tmp_path, sources, manifest)
    matched = scope.match_cohort(policy, "final_replay", sources, manifest, null_construction=recipe)
    with pytest.raises(ValueError, match="exactly one authorized"):
        scope.match_cohort(policy, "final_replay", sources, manifest)
    derived = {
        "role": "evaluation", "construction_scope": matched,
        "null_construction": {
            **recipe, "normalized_relative": True,
            "base_manifest": {
                "path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        },
    }
    path = tmp_path / "derived.json"
    save(path, derived)
    assert scope.match_cohort(policy, "final_replay", sources, path) == matched
    derived["null_construction"]["seed"] += 1
    save(path, derived)
    with pytest.raises(ValueError, match="exactly one authorized"):
        scope.match_cohort(policy, "final_replay", sources, path)
    derived["null_construction"]["seed"] -= 1
    derived.pop("construction_scope")
    save(path, derived)
    with pytest.raises(ValueError, match="construction scope changed"):
        scope.match_cohort(policy, "final_replay", sources, path)


def test_sealed_scramble_scope_retains_short_coverage_population_ledger(
    scoped_policy, tmp_path
):
    policy, sources, manifest, _ = scoped_policy
    sources = sources + [{"source_id": "456", "parent_tpf": "67", "raw_fits_sha256": "b" * 64}]
    recipe = bind_null_recipe(policy, tmp_path, sources, manifest, "quarter_scrambled")
    matched = scope.match_cohort(policy, "final_replay", sources, manifest, null_construction=recipe)
    derived = {
        "role": "evaluation", "construction_scope": matched,
        "null_construction": {
            **recipe, "normalized_relative": True,
            "base_manifest": {
                "path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        },
        "construction_ledger": [
            {"source_id": "123", "quarters": 3, "status": "CONSTRUCTED"},
            {"source_id": "456", "quarters": 2, "status": "INVERTED_ONLY_SHORT_COVERAGE"},
        ],
    }
    path = tmp_path / "scrambled.json"
    save(path, derived)
    assert scope.match_cohort(policy, "final_replay", sources[:1], path) == matched
    derived["construction_ledger"][1]["status"] = "CONSTRUCTED"
    save(path, derived)
    with pytest.raises(ValueError, match="short-coverage rule"):
        scope.match_cohort(policy, "final_replay", sources[:1], path)


def test_unauthorized_sealed_null_construction_stops_before_array_access(
    scoped_policy, tmp_path, monkeypatch
):
    from deterministic import signal_contracts as signals

    policy, _, manifest, policy_reference = scoped_policy
    original = json.loads(manifest.read_text())
    original.update(role="evaluation", measurement_scope="SYNTHETIC_CONTRACT_TEST")
    save(manifest, original)
    monkeypatch.setattr(signals, "load_source", lambda *args: pytest.fail("Unauthorized sealed arrays opened"))
    args = SimpleNamespace(input_manifest=manifest, disposition_policy=Path(policy_reference["path"]),
                           variant="inverted", seed=5, out=tmp_path / "null", exclusions=None)
    monkeypatch.delenv("SPACELAB_DISPOSITION_EVALUATION_TOKEN")
    with pytest.raises(ValueError, match="valid scoped role token"):
        signals.construct(args)
    monkeypatch.setenv("SPACELAB_DISPOSITION_EVALUATION_TOKEN", runtime.disposition_evaluation_token(policy, "final_replay"))
    with pytest.raises(ValueError, match="input changed"):
        signals.construct(args)
    assert not args.out.exists()


def test_authorized_sealed_null_constructor_binds_recipe_before_synthetic_flux_access(
    scoped_policy, tmp_path, monkeypatch
):
    import numpy as np
    from deterministic import signal_contracts as signals

    policy, sources, manifest, policy_reference = scoped_policy
    count = 360
    arrays = {
        "time_bkjd": np.arange(count) * 0.02,
        "cadenceno": np.arange(count, dtype=np.int32),
        "quarter": np.repeat([1, 2, 3], 120),
        "sap_quality": np.zeros(count, dtype=np.int64),
        "flatten_mask": np.zeros(count, dtype=np.int64),
        "psf_flat_kbonus_flux": 1000 + np.sin(np.arange(count)),
        "psf_flat_kbonus_flux_err": np.ones(count),
    }
    array_path = tmp_path / "123.npz"
    np.savez_compressed(array_path, **arrays)
    metadata = {**sources[0], "origin_tpf_kic": "45", "shard": 1}
    save(manifest, {
        "schema": 1, "role": "evaluation", "measurement_scope": "SYNTHETIC_CONTRACT_TEST",
        "sources": {"123": {
            "arrays": array_path.name, "metadata": metadata,
            "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        }},
    })
    recipe = bind_null_recipe(policy, tmp_path, sources, manifest)
    save(Path(policy_reference["path"]), policy)
    monkeypatch.setenv("SPACELAB_DISPOSITION_EVALUATION_TOKEN", runtime.disposition_evaluation_token(policy, "final_replay"))
    monkeypatch.setattr(signals, "dump_path", Path)
    args = SimpleNamespace(input_manifest=manifest, disposition_policy=Path(policy_reference["path"]),
                           variant="inverted", seed=5, out=tmp_path / "null", exclusions=None)
    signals.construct(args)
    derived = json.loads((args.out / "input-manifest.json").read_text())
    assert derived["role"] == "evaluation"
    assert derived["control_role"] == "SEALED_FINAL_NULLS"
    assert {key: derived["null_construction"][key] for key in recipe} == recipe
    assert scope.match_cohort(policy, "final_replay", sources, args.out / "input-manifest.json") == derived["construction_scope"]
