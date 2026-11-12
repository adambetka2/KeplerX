"""Cost-aware dispatch keeps cohort results, accounting and failures intact."""

import json
import sys
from collections import Counter
from concurrent.futures import Future
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry"), str(ROOT / "pipeline/vetting")]

import vetting


@pytest.mark.parametrize("program", ["orbital", "alias"])
def test_dispatch_preserves_cohort_results_and_worker_accounting(
    tmp_path, monkeypatch, capsys, program
):
    ids = ["small", "large", "tied", "middle"]
    sizes = {"small": 1, "large": 5, "tied": 5, "middle": 3}
    (tmp_path / "sources").mkdir()
    (tmp_path / "cohort.json").write_text(
        json.dumps([{"source_id": sid} for sid in ids])
    )
    for sid in ids:
        (tmp_path / "sources" / f"{sid}.json").write_text(
            json.dumps({"events": list(range(sizes[sid]))})
        )
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    monkeypatch.setattr(vetting, "COUNTERS", Counter())
    monkeypatch.setattr(vetting, "STAGE", {"name": "orbital_measurement"})
    dispatched = []

    class Pool:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, measure, sid):
            dispatched.append(sid)
            future = Future()
            future.set_result(measure(sid))
            return future

    def measure(sid):
        return {
            "source_id": sid,
            "events": sizes[sid],
            "timing": {"wall_seconds": 1.0, "cpu_seconds": 1.0},
            "counters": {"fitted_hypotheses": sizes[sid]},
        }

    monkeypatch.setattr(vetting, "ProcessPoolExecutor", Pool)
    results = vetting.run_pooled(measure, program, 2)
    assert dispatched == (["large", "tied", "middle", "small"] if program == "orbital" else ids)
    assert [row["source_id"] for row in results] == ids
    logs = [json.loads(line) for line in (tmp_path / "source-timings.jsonl").read_text().splitlines()]
    assert [row["source_id"] for row in logs] == ids
    assert [row["events"] for row in logs] == [sizes[sid] for sid in ids]
    assert vetting.COUNTERS["fitted_hypotheses"] == sum(sizes.values())
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["source_id"] for row in progress] == ids
