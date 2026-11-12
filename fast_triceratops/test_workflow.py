"""Exercise allocation, immutable inputs and the real frozen-source replay path."""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from fast_triceratops import workflow


@pytest.mark.parametrize(
    "fpp,nfpp,stops", [(0.10, 0, True), (0, 0.01, True), (0.099, 0.009, False)]
)
def test_first_two_gate_precedes_later_seeds(monkeypatch, fpp, nfpp, stops):
    finished, lock = set(), threading.Lock()
    config = dict(workers=2, repeats=20)
    monkeypatch.setitem(workflow.STATE, "config", config)
    monkeypatch.setitem(workflow.STATE, "root", Path(__file__).parent)

    def attempt(stage, index):
        with lock:
            if stage == "recovered" or index >= 2:
                assert {("final", 0), ("final", 1)} <= finished
            finished.add((stage, index))
        return dict(stage=stage, index=index, FPP=fpp, NFPP=nfpp)

    monkeypatch.setattr(workflow, "attempt", attempt)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        status, primary, diagnostic = workflow.schedule(pool, config, 3)
    assert [r["index"] for r in primary] == list(range(2 if stops else 20))
    assert [r["index"] for r in diagnostic] == ([] if stops else [0, 1, 2])
    assert status.startswith("STOPPED" if stops else "COMPLETE")


def test_failure_does_not_allocate_the_remaining_seeds(monkeypatch):
    submitted = []
    monkeypatch.setitem(workflow.STATE, "config", dict(workers=2))
    monkeypatch.setitem(workflow.STATE, "root", Path(__file__).parent)

    class FailedPool:
        def submit(self, function, stage, index):
            submitted.append(index)
            future = concurrent.futures.Future()
            future.set_exception(ValueError("instrument failure"))
            return future

    with pytest.raises(ValueError, match="instrument failure"):
        workflow.batch(FailedPool(), "final", range(20))
    assert submitted == [0, 1]


def minimal_plan(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    plan = dict(
        protocol="TRICERATOPS_KEPLER_V3",
        cpu_budget=2,
        draws_per_scenario=1_000_000,
        seeds=list(range(20)),
    )
    workflow.write(source / "plan.json", plan)
    workflow.write(source / "freeze.json", dict(files=[]))
    return source / "plan.json"


def test_experiment_snapshots_source_and_rejects_changed_config(tmp_path):
    plan = minimal_plan(tmp_path)
    source_receipt = workflow.receipt(plan)
    config = workflow.create(plan, tmp_path / "experiment", draws=100, repeats=2)
    _, frozen = workflow.verify_experiment(config)
    assert len(frozen) == 64
    assert (config.parent / "source/fast_triceratops/workflow.py").is_file()
    original = config.read_text()
    config.write_text(original + "\n")
    with pytest.raises(ValueError, match="Experiment input changed"):
        workflow.verify_experiment(config)
    assert workflow.receipt(plan) == source_receipt


@pytest.mark.parametrize("changed", ["new_source", "source_freeze"])
def test_changed_or_unbound_source_is_rejected(tmp_path, changed):
    plan = minimal_plan(tmp_path)
    config = workflow.create(plan, tmp_path / "experiment", draws=100)
    if changed == "new_source":
        (config.parent / "source/unbound.py").write_text("raise RuntimeError\n")
    else:
        (plan.parent / "freeze.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Unbound|input changed"):
        workflow.verify_experiment(config)


def test_low_disk_reserve_fails_before_creating_output(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path)
    usage = shutil.disk_usage(tmp_path)._replace(free=190 * 2**20)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage)
    with pytest.raises(ValueError, match="2 GiB"):
        workflow.create(plan, tmp_path / "experiment")
    assert not (tmp_path / "experiment").exists()


def test_explicit_stop_prevents_worker_start(tmp_path):
    config = workflow.create(minimal_plan(tmp_path), tmp_path / "experiment")
    workflow.write(config.parent / "stop.json", dict(reason="Explicit test stop"))
    with pytest.raises(subprocess.CalledProcessError):
        workflow.launch(config)
    assert not (config.parent / "runs").exists()


@pytest.mark.parametrize(
    "option",
    [dict(workers=3), dict(draws=1), dict(draws=1_000_001), dict(mode="unknown")],
)
def test_invalid_resource_or_science_overrides_fail_before_output(tmp_path, option):
    plan = minimal_plan(tmp_path)
    output = tmp_path / "experiment"
    with pytest.raises(ValueError):
        workflow.create(plan, output, **option)
    assert not output.exists()


def gold_campaign(tmp_path):
    """Kepler-10 photons through real preparation, freeze and quarter likelihood.

    Three interleaved quarters and their responses are a declared adapter control,
    not measured Kepler-10 quarter fractions or a new scientific fit.
    """
    from fast_triceratops.gold import EXPTIME, FIXTURE, P_ORB
    from post_boi.prepare import prepare
    from post_boi.protocol import freeze
    from post_boi.records import read, receipt, write

    raw = np.loadtxt(FIXTURE / "lightcurve.csv", delimiter=",")
    root = tmp_path / "gold"
    root.mkdir()
    shutil.copyfile(FIXTURE / "stars.csv", root / "stars.csv")
    shutil.copyfile(FIXTURE / "background.csv", root / "background.csv")
    with np.load(FIXTURE / "geometry.npz") as archive:
        geometry = dict(archive)
    geometry["quarter"] = np.array([3, 4, 5])
    for q, fraction in zip([3, 4, 5], [0.9, 0.7, 0.95], strict=True):
        geometry[f"xy_q{q}"] = geometry["xy_q3"]
        geometry[f"aperture_q{q}"] = geometry["aperture_q3"]
        geometry[f"flux_fraction_q{q}"] = np.array([fraction, 1 - fraction, 0, 0, 0, 0])
    np.savez_compressed(root / "geometry.npz", **geometry)
    np.savez_compressed(
        root / "photons.npz",
        source_id="377780790",
        time_bkjd=raw[:, 0],
        flux=raw[:, 1],
        flux_err=raw[:, 2],
        quarter=3 + np.arange(len(raw)) % 3,
        valid=np.ones(len(raw), dtype=bool),
    )
    np.savez_compressed(
        root / "noise.npz",
        source_id="377780790",
        time_bkjd=raw[:, 0],
        sigma=raw[:, 2],
        applied_fraction=np.ones(len(raw)),
    )
    c = read(Path(__file__).resolve().parents[1] / "post_boi/candidate-template.json")
    c.update(
        source_id="377780790",
        signal_id="workflow-gold-control",
        input_basis="Known-object photons; artificial quarter adapter control, no admission",
        photometry="photons.npz",
        prepared="prepared",
        scene="stars.csv",
        geometry="geometry.npz",
        background="background.csv",
        aperture_response="measured_kepler",
        native_review="review.json",
        readiness="review.json",
        flux_convention="raw_aperture",
        exposure_days=EXPTIME,
        recovered=dict(period_days=P_ORB, epoch_bkjd=0, duration_days=0.065),
        timing_review="One fixed fixture clock; no FPP-selected clock",
        cpu_budget=2,
    )
    c["detrending"]["window_days"] = 0.2
    c["native"]["priors"] = "priors.json"
    c["probability"].update(
        noise="noise.npz",
        noise_basis="Delivered fixture errors for adapter control",
        supersample=20,
    )
    write(root / "candidate.json", c)
    write(root / "priors.json", dict(scope="Adapter control, no native inference"))
    write(
        root / "review.json",
        dict(
            source_id=c["source_id"],
            signal_id=c["signal_id"],
            accepted=True,
            evidence=[receipt(root / "priors.json")],
            period_days=P_ORB,
            epoch_bkjd=0,
            duration_days=0.065,
            observed_depth_fraction=0.00019,
            probability_noise_sigma=float(raw[:, 2].mean()),
            probability_noise_basis="Delivered fixture errors for adapter control",
        ),
    )
    prepare(root / "candidate.json", root / "prepared")
    return freeze(root / "candidate.json", root / "campaign")


def v3_runtime():
    import inspect

    from triceratops.triceratops import target

    return "joint" in inspect.signature(target.calc_probs).parameters


def seed_child(config, inject_failure):
    code = (
        "import sys\nfrom fast_triceratops import workflow as w\n"
        "from post_boi import campaign\nw.initialize(sys.argv[1], True)\n"
    )
    if inject_failure:
        code += (
            "def fail(*args, **kwargs):\n    raise RuntimeError('injected implementation failure')\n"
            "campaign.calculate_quarters = fail\n"
        )
    code += "w.attempt('final', 0)\n"
    source_plan = Path(workflow.read(config)["source_plan"]["path"])
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(
            [str(config.parent / "source"), str(source_plan.parent / "source")]
        ),
        NUMBA_CACHE_DIR=str(config.parent / "numba-cache"),
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", code, str(config)],
        cwd=config.parent,
        env=env,
        capture_output=True,
        text=True,
    )


def partial_replay(config):
    seed = config.parent / "runs/final/seed-00"
    failed = seed_child(config, inject_failure=True)
    assert failed.returncode != 0 and "injected implementation failure" in failed.stderr
    assert len(list(seed.glob("attempt-*/failure.json"))) == 1
    assert not (seed / "result.json").exists()
    completed = seed_child(config, inject_failure=False)
    assert completed.returncode == 0, completed.stderr
    assert not (config.parent / "summary.json").exists()
    assert len(list(seed.glob("attempt-*"))) == 2
    return workflow.receipt(seed / "result.json")


@pytest.mark.skipif(not v3_runtime(), reason="needs the pinned V3 runtime")
def test_gold_frozen_process_replay_and_integrity(tmp_path):
    plan = gold_campaign(tmp_path)
    original = workflow.receipt(plan.parent / "freeze.json")
    # One low-priority numerical worker in both runs; parallel allocation is
    # exercised above without competing with the active validation's CPU budget.
    configs = []
    for mode in ("stock", "engine"):
        config = workflow.create(
            plan,
            tmp_path / mode,
            mode=mode,
            draws=2048,
            repeats=2,
            profile=mode == "engine",
        )
        saved = partial_replay(config) if mode == "engine" else None
        workflow.launch(config)
        if saved:
            assert workflow.receipt(saved["path"]) == saved
            assert (
                len(list((config.parent / "runs/final/seed-00").glob("attempt-*"))) == 2
            )
        configs.append(config)
    for index in range(2):
        paths = [p.parent / f"runs/final/seed-{index:02d}/result.json" for p in configs]
        a, b = (workflow.read(p) for p in paths)
        assert (a["FPP"], a["NFPP"], a["seed"]) == (b["FPP"], b["NFPP"], b["seed"])
        files = [
            {Path(r["path"]).name: Path(r["path"]) for r in saved["files"]}
            for saved in (a, b)
        ]
        with (
            np.load(files[0]["evidence.npz"]) as left,
            np.load(files[1]["evidence.npz"]) as right,
        ):
            np.testing.assert_array_equal(
                left["log_evidence"].view(np.uint64),
                right["log_evidence"].view(np.uint64),
            )
        assert (
            files[0]["scenarios.csv"].read_bytes()
            == files[1]["scenarios.csv"].read_bytes()
        )
        assert files[0]["stars.csv"].read_bytes() == files[1]["stars.csv"].read_bytes()
    assert workflow.receipt(plan.parent / "freeze.json") == original
    assert not (plan.parent / "runs").exists()
    # A changed original photon file must fail even when a complete replay exists.
    photons = plan.parent.parent / "photons.npz"
    with photons.open("ab") as stream:
        stream.write(b"changed")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    failed = subprocess.run(
        [sys.executable, "-B", "-m", "post_boi.campaign", "verify", str(plan)],
        cwd=plan.parent / "source",
        env=env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0 and "Frozen file changed" in failed.stderr
