import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.vetting import (
    statistical_validation,
    validation_campaign,
    validation_results,
)


def write_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_canonical_plan_is_exactly_20_tic_runs_per_boi():
    plan = statistical_validation.load_plan()

    assert plan["catalogue"] == "TIC"
    assert plan["ephemeris"] == "DISCOVERY"
    assert tuple(target["source_id"] for target in plan["targets"]) == (
        "2077554781499350144",
        "2127186564572468352",
    )
    assert len(plan["targets"]) * statistical_validation.RUNS_PER_TARGET == 40


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.update(runs=20),
        lambda plan: plan.update(catalogue="GAIA"),
        lambda plan: plan["thresholds"].update(FPP_lt=0.01),
        lambda plan: plan["targets"].append(copy.deepcopy(plan["targets"][0])),
        lambda plan: plan["targets"][0].update(family="TIC"),
        lambda plan: plan["targets"][0].update(
            packet=plan["targets"][0]["packet"].replace("tic", "gaia")
        ),
    ],
)
def test_old_model_matrix_shapes_are_rejected(tmp_path, mutate):
    plan = copy.deepcopy(statistical_validation.load_plan())
    mutate(plan)

    with pytest.raises(ValueError):
        statistical_validation.load_plan(write_plan(tmp_path, plan))


def test_weight_audit_reconstructs_the_omitted_evidence_bound(monkeypatch):
    storage = io.BytesIO()
    np.savez(
        storage,
        log_weight=np.array([0.0, -np.inf]),
        bounded=np.array([False, True]),
        bounded_log_weight_upper=np.log(0.2),
    )
    payload = storage.getvalue()
    load = np.load
    monkeypatch.setattr(
        validation_results.np, "load", lambda _: load(io.BytesIO(payload))
    )
    monkeypatch.setattr(validation_results, "digest", lambda _: "unchanged")
    row = {
        "draws": 2,
        "weight_file": "weights.npz",
        "weight_sha256": "unchanged",
        "log_evidence": np.log(0.5),
        "bounded_physical_draws": 1,
        "omitted_log_evidence_upper": np.log(0.1),
        "log_evidence_upper": np.log(0.6),
    }
    validation_results.validate_weight(Path("result.json"), row, 2, True)
    row["log_evidence_upper"] = np.log(0.5)
    with pytest.raises(ValueError, match="bound does not reproduce"):
        validation_results.validate_weight(Path("result.json"), row, 2, True)


def test_reported_probability_cannot_disagree_with_scenario_evidence(monkeypatch):
    plan = statistical_validation.load_plan()
    target = dict(plan["targets"][0], scenario_count=3)
    evidence = [
        {"scenario": "TP", "log_evidence": np.log(0.9)},
        {"scenario": "EB", "log_evidence": np.log(0.09)},
        {"scenario": "NTP:neighbour", "log_evidence": np.log(0.01)},
    ]
    seed = int(
        np.random.SeedSequence(plan["seed_entropy"] + [0, 0]).generate_state(
            1, dtype=np.uint64
        )[0]
    )
    data = {
        "status": "COMPLETE_INDEPENDENT_RUN",
        "identity": {
            "source_id": target["source_id"],
            "run_index": 0,
            "seed": seed,
            "draws_per_scenario": 1_000_000,
            "freeze_sha256": "unchanged",
        },
        "scenario_names": ["TP", "EB", "NTP:neighbour"],
        "scenario_evidences": evidence,
        "probabilities": statistical_validation.probabilities(evidence),
    }
    monkeypatch.setattr(validation_results, "digest", lambda _: "unchanged")
    monkeypatch.setattr(validation_results, "validate_weight", lambda *args: None)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _: json.dumps(statistical_validation.serializable(data)),
    )
    path = Path("result.json")
    validation_results.validate_result(path, target, 0, 0, plan, path, True)
    data["probabilities"]["FPP"]["probability"] = 0.001
    with pytest.raises(ValueError, match="probabilities do not reproduce"):
        validation_results.validate_result(path, target, 0, 0, plan, path, True)


def test_worker_failure_preserves_first_failure_and_stops_queue(monkeypatch):
    import threading

    barrier = threading.Barrier(10)
    calls = []

    def fail(job, *args):
        calls.append(job)
        barrier.wait(timeout=5)
        return {"exit_code": 1, "job": job}

    monkeypatch.setattr(validation_campaign, "execute", fail)
    monkeypatch.setattr(statistical_validation, "write_json", lambda *args: None)
    state = {"completed_jobs": [], "first_failure": None}
    validation_campaign.run_queue(list(range(40)), Path("."), {}, "test", state)
    assert sorted(calls) == list(range(10))
    assert len(state["completed_jobs"]) == 10
    assert state["first_failure"]["exit_code"] == 1
