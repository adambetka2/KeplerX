"""Transit model reuse preserves parameter updates and independent time grids."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline/photometry"), str(ROOT / "pipeline/vetting")]

import vetting_models as models


def parameters():
    rng = np.random.default_rng(426)
    return [
        np.array(
            [
                rng.uniform(-0.1, 0.1),
                np.log(rng.uniform(0.01, 0.9)),
                np.log(rng.uniform(0.03, 0.6)),
                rng.uniform(0.05, 0.95),
                rng.uniform(0.01, 0.99),
                rng.uniform(0.01, 0.99),
                rng.uniform(-0.02, 0.02),
                rng.uniform(-0.05, 0.05),
            ]
        )
        for _ in range(20)
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("supersample", [1, 7, 51])
def test_reused_model_updates_every_parameter_exactly(dtype, supersample):
    time = np.linspace(-1, 1, 111, dtype=dtype)
    curve = models.TransitCurve(time, supersample=supersample)
    sequence = parameters()
    expected = [
        models.transit(theta, time, supersample=supersample) for theta in sequence
    ]
    for order in (range(len(sequence)), reversed(range(len(sequence)))):
        for index in order:
            np.testing.assert_array_equal(curve(sequence[index]), expected[index])


def test_each_fit_owns_one_full_model_and_its_time_grid(monkeypatch):
    factory = models.batman.TransitModel
    constructed = []

    def construct(*args, **kwargs):
        result = factory(*args, **kwargs)
        constructed.append((result, kwargs.get("fac")))
        return result

    monkeypatch.setattr(models.batman, "TransitModel", construct)
    time = np.linspace(-1, 1, 121)
    original = time.copy()
    curve = models.TransitCurve(time)
    sequence = parameters()
    before = curve(sequence[0])
    full = curve.model
    time += 5
    for theta in sequence:
        curve(theta)
    assert curve.model is full and constructed[0] == (full, None)
    for window, fac in constructed[1:]:
        assert fac == full.fac and 0 < len(window.t) < len(original)
        assert np.isin(window.t, original).all()
    np.testing.assert_array_equal(curve(sequence[0]), before)
    other = models.TransitCurve(original, exposure=0.05, supersample=51)
    other(sequence[0])
    assert other.model is constructed[-1][0] and curve.model is not other.model
    np.testing.assert_array_equal(
        curve(sequence[0]), models.transit(sequence[0], original)
    )
    np.testing.assert_array_equal(
        other(sequence[0]), models.transit(sequence[0], original, 0.05, 51)
    )


def test_fixed_exposure_window_reuses_model_and_updates_geometry(monkeypatch):
    factory = models.batman.TransitModel
    constructed = []

    def construct(*args, **kwargs):
        result = factory(*args, **kwargs)
        constructed.append(result)
        return result

    monkeypatch.setattr(models.batman, "TransitModel", construct)
    time = np.linspace(-1, 1, 1001)
    curve = models.TransitCurve(time, supersample=51)
    theta = np.array([0.0, np.log(0.04), np.log(0.2), 0.3, 0.25, 0.4, 0.0, 0.0])
    curve(theta)
    curve(theta)
    cached = curve.near_model
    assert len(constructed) == 2
    for index in range(len(theta)):
        changed = theta.copy()
        changed[index] += 1e-8
        actual = curve(changed)
        assert curve.near_model is cached
        np.testing.assert_array_equal(
            actual, models.transit(changed, time, supersample=51)
        )
    shifted = theta.copy()
    shifted[0] = 0.4
    actual = curve(shifted)
    assert curve.near_model is not cached
    np.testing.assert_array_equal(actual, models.transit(shifted, time, supersample=51))


@pytest.mark.parametrize("length", [1, 2, 3, 4, 7, 8, 101, 1000])
def test_median_is_bitwise_equal_to_numpy(length):
    rng = np.random.default_rng(length)
    samples = [
        rng.normal(size=length),
        rng.integers(-3, 3, length).astype(float),
        np.full(length, 0.1) * np.arange(length),
    ]
    for values in samples:
        original = values.copy()
        assert models.median(values) == np.median(values)
        np.testing.assert_array_equal(values, original)


def test_median_rejects_non_finite_and_empty_input():
    with pytest.raises(ValueError):
        models.median(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        models.median(np.array([]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("parameter_errors", [False, True])
def test_variable_fit_preserves_casts_updates_and_independent_results(
    dtype, columns, parameter_errors
):
    rng = np.random.default_rng(20261005)
    time = np.linspace(-1, 1, 101, dtype=dtype)
    y = rng.normal(size=len(time)).astype(dtype)
    error = rng.uniform(0.01, 2, len(time)).astype(dtype)
    original = [value.copy() for value in (time, y, error)]
    solver = models._VariableLinearFit(time, y, error, columns)
    proposals = [rng.normal(size=(columns, len(time))).astype(dtype) for _ in range(4)]
    results = []
    for proposal in proposals + proposals[::-1]:
        design = np.column_stack([np.ones(len(time)), time, *proposal])
        expected = models.linear_fit(design, y, error, parameter_errors)
        actual = solver(*proposal, parameter_errors=parameter_errors)
        assert actual.keys() == expected.keys()
        for key in expected:
            if key == "status":
                assert actual[key] == expected[key]
            else:
                assert (
                    np.asarray(actual[key]).tobytes()
                    == np.asarray(expected[key]).tobytes()
                )
        results.append((actual, expected))
    assert all(actual == expected for actual, expected in results)
    for actual, expected in zip((time, y, error), original):
        np.testing.assert_array_equal(actual, expected)


def test_variable_fit_recovers_after_rank_deficient_proposal():
    time = np.linspace(-1, 1, 101)
    y = np.sin(time)
    error = np.full(len(time), 0.1)
    solver = models._VariableLinearFit(time, y, error, 1)
    for pulse in (np.sin(time), time, np.ones(len(time)), np.cos(time)):
        expected = models.linear_fit(
            np.column_stack([np.ones(len(time)), time, pulse]), y, error
        )
        assert solver(pulse, parameter_errors=True) == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("constant_time", [False, True])
def test_baseline_projection_reuses_fixed_inputs_without_changing_bits(
    dtype, constant_time
):
    rng = np.random.default_rng(20261005)
    time = np.linspace(-1, 1, 202, dtype=dtype)[::2]
    if constant_time:
        time[:] = -0.0
    error = rng.uniform(0.01, 2, len(time)).astype(dtype)
    original = [value.copy() for value in (time, error)]
    project = models._BaselineProjection(time, error)
    fluxes = [rng.normal(size=len(time)).astype(dtype) for _ in range(4)]
    expected = [models.project_baseline(time, flux, error) for flux in fluxes]
    for index in [0, 1, 2, 3, 2, 1, 0]:
        actual = project(fluxes[index])
        assert actual.tobytes() == expected[index].tobytes()
        actual[:] = 99
    for actual, want in zip((time, error), original):
        assert actual.tobytes() == want.tobytes()
