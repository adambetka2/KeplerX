"""Check metadata sampling and physical injections before flat cadence masking."""

import hashlib
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_retrieval as pilot


def worker_noise_receipt():
    engine = pilot.API.ENGINE
    search = engine.NATIVE.SEARCH if hasattr(engine, "NATIVE") else engine.SEARCH
    noise = search.sibling("evolve_noise")
    time = np.arange(257, dtype=float) / 48
    flux = np.sin(np.arange(257, dtype=float)) * 0.0001
    result = noise.noise_weights(
        time,
        flux,
        np.ones(257, dtype=np.int64),
        np.full(257, 0.0001),
        0.08,
        {"method": "local_point", **noise.POINT_DEFAULTS},
    )
    return {
        "sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "cache_hits": sum(noise.window_mad.stats.cache_hits.values()),
    }


def test_spawned_workers_survive_engine_switch_and_warm_restart(tmp_path, monkeypatch):
    out = tmp_path / "round"
    designs = [
        {"name": engine, "engine": engine, "config": {}}
        for engine in ("evolve_native_coherence", "evolve_psf_flat")
    ]
    bindings = pilot.API.snapshot(out, designs)
    pilot.API.save(
        out / "identity.json",
        {
            "designs": designs,
            "source_bindings": bindings,
            "runtime": pilot.API.runtime_identity(),
        },
    )
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "shared-parent-cache"))
    receipts = []
    for index in (0, 1, 0, 1):
        engine = designs[index]["engine"]
        with (
            pilot.worker_cache(out, engine),
            ProcessPoolExecutor(
                1,
                mp_context=get_context("spawn"),
                initializer=pilot.initialize,
                initargs=(out, index),
            ) as pool,
        ):
            receipts.append(pool.submit(worker_noise_receipt).result())
        assert os.environ["NUMBA_CACHE_DIR"] == str(
            tmp_path / "shared-parent-cache"
        )
    assert {row["sha256"] for row in receipts} == {
        "6c30b7c1890e460c7fad466262f2fc575802bf23581f96c0d5bf078ff16846ea"
    }
    assert all(row["cache_hits"] > 0 for row in receipts[2:])


def test_runner_imports_candidate_from_its_snapshot(tmp_path):
    out = tmp_path / "round"
    pilot.API.snapshot(
        out, [{"name": "flat", "engine": "evolve_psf_flat", "config": {}}]
    )
    source = """import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import flat_retrieval as pilot
out=Path(sys.argv[2])
pilot.API.initialize(out/'source/dwarfs/shallow','evolve_psf_flat',{},array_policy='proposals')
assert Path(pilot.API.ENGINE.__file__).resolve()==(out/'source/dwarfs/shallow/evolve_psf_flat.py').resolve()
"""
    subprocess.run(
        [sys.executable, "-c", source, str(Path(pilot.__file__).parent), str(out)],
        check=True,
    )


def test_balanced_selection_keeps_population_and_depth_quotas():
    items = [
        {
            "case": {
                "case_id": str(index),
                "kind": "synthetic_planet",
                "population": "M" if index % 2 else "K",
                "preparation_status": "PREPARED",
            },
            "origin": {"depth_bin": index % 5},
            "truth": {},
            **{field: float(index) for field in pilot.NUMERIC},
        }
        for index in range(200)
    ]
    selected = pilot.sample(items, 20)
    assert len(selected) == len({item["case"]["case_id"] for item in selected}) == 20
    assert Counter(pilot.stratum(item) for item in selected) == Counter(
        {pilot.stratum(item): 2 for item in items}
    )
    assert pilot.sample(items, 20) == selected
    assert (
        max(
            pilot.ks_2samp(
                [item[key] for item in items], [item[key] for item in selected]
            ).statistic
            for key in pilot.NUMERIC
        )
        <= 0.08
    )


def test_flat_physical_loss_uses_source_baseline_and_keeps_masked_rows(monkeypatch):
    curve = {
        "time": np.arange(100) / 48,
        "flux": np.full(100, 900.0),
        "error": np.ones(100),
        "quality": np.zeros(100, dtype=int),
        "flatten": np.zeros(100, dtype=int),
        "cadence": np.arange(100),
        "quarter": np.ones(100, dtype=int),
    }
    curve["flatten"][20] = 1
    loss = np.zeros(100)
    loss[20:23] = 0.001
    monkeypatch.setattr(pilot.calibrate, "loss", lambda time, truth: loss.copy())
    injected, arrays = pilot.injected_flat(curve, {})
    np.testing.assert_allclose(injected["flux"][20:23], 899.1)
    np.testing.assert_array_equal(injected["flatten"], curve["flatten"])
    np.testing.assert_array_equal(curve["flux"], 900)
    np.testing.assert_array_equal(arrays["physical_loss"], loss)
