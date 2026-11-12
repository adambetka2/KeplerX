"""Frozen injected gold checks for the restricted blind-search experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "preparation_experiment", Path(__file__).with_name("preparation.py")
)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)


@pytest.fixture(scope="module")
def gold():
    time = np.arange(0, 45, 0.02043361111111111) + 1000
    rng = np.random.default_rng(613)
    error = np.full(len(time), 200e-6)
    baseline = 0.003 * np.sin(2 * np.pi * time / 13) + rng.normal(0, error)
    engine = EXPERIMENT.Search(time, error)
    return time, baseline, engine


@pytest.mark.parametrize(
    "period,phase,hours", [(9.80313, 0.3783, 2), (10.1931, 0.7127, 6)]
)
def test_off_grid_physical_gold_recovery(gold, period, phase, hours):
    time, baseline, engine = gold
    epoch = time[0] + period * phase
    loss = EXPERIMENT.physical_loss(time, period, epoch, hours, 3000)
    result = engine.search(baseline - loss, EXPERIMENT.METHODS[2])
    assert abs(result["period_days"] / period - 1) < 0.002
    offset = (result["epoch_bkjd"] - epoch + period / 2) % period - period / 2
    assert abs(offset) < hours / 48
    assert result["score"] > 40


def test_joint_fit_preserves_known_clock_signal(gold):
    time, baseline, engine = gold
    loss = EXPERIMENT.physical_loss(time, 9.903, time[0] + 2.13, 6, 100)
    method = EXPERIMENT.METHODS[2]
    retained = engine.known_depth(baseline - loss, loss, method) - engine.known_depth(
        baseline, loss, method
    )
    assert retained == pytest.approx(1, abs=1e-8)


def test_null_is_not_a_global_phase_shift(gold):
    time, flux, engine = gold
    curve = {"time": time, "flux": 1 + flux}
    null = next(
        EXPERIMENT.null_curves(
            curve, engine, np.arange(len(time)), flux, np.random.default_rng(91), 1, 1
        )
    )
    assert not np.array_equal(null["flux"], curve["flux"])
    original = engine.search(flux, EXPERIMENT.METHODS[2])["score"]
    changed = engine.search(null["flux"] - 1, EXPERIMENT.METHODS[2])["score"]
    assert abs(changed - original) > 0.01


def test_normalization_rescales_formal_score(gold):
    time, flux, engine = gold
    method = EXPERIMENT.METHODS[2]
    original = engine.search(flux, method)
    normalized = EXPERIMENT.normalized_search(
        engine, time, flux, engine.error * 2, method
    )
    assert normalized["score"] == pytest.approx(original["score"] / 2)
    changed_error = engine.error.copy()
    changed_error[0] *= 2
    with pytest.raises(ValueError, match="scalar error normalization"):
        EXPERIMENT.normalized_search(engine, time, flux, changed_error, method)


def test_wider_biweight_preserves_more_shallow_long_transit(gold):
    time, baseline, engine = gold
    loss = EXPERIMENT.physical_loss(time, 9.903, time[0] + 2.13, 6, 100)
    retained = [
        engine.known_depth(baseline - loss, loss, method)
        - engine.known_depth(baseline, loss, method)
        for method in EXPERIMENT.METHODS[:2]
    ]
    assert 0 < retained[0] < retained[1] < 1
