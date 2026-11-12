"""Cached PRFs preserve upstream profiles and gradients without rounding positions."""

import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
from lkprf import KeplerPRF as ArchiveKeplerPRF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kepler_prf import KeplerPRF


def model(cls, channel=1):
    prf = cls.__new__(cls)
    axis = np.linspace(-2.5, 2.5, 51)
    row, column = np.meshgrid(axis, axis, indexing="ij")
    prf.PRFrow = axis
    prf.PRFcol = axis
    prf.PRFdata = np.array(
        [
            np.exp(-((row - i / 20) ** 2 + (column + i / 30) ** 2) / channel)
            for i in range(5)
        ]
    )
    prf.PRFdata /= prf.PRFdata.sum(axis=(1, 2))[:, None, None]
    prf.crval1p = np.array([12.0, 12.0, 1112.0, 1112.0, 550.0])
    prf.crval2p = np.array([20.0, 1044.0, 20.0, 1044.0, 530.0])
    prf.cdelt1p = prf.cdelt2p = np.full(5, 0.1)
    prf._spline_cache = OrderedDict()
    return prf


def test_exact_profiles_and_gradients_at_varied_positions_and_shapes():
    original, cached = model(ArchiveKeplerPRF), model(KeplerPRF)
    positions = [(100.01 + i * 0.173, 150.2 + i * 0.083) for i in range(30)]
    positions += [(19.0, 11.0), (1045.0, 1113.0), (530.0, 550.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row, column in positions:
            for shape in ((5, 5), (7, 9)):
                targets = [(row, column), (row + 1.2, column + 0.7)]
                origin = (int(row) - 2, int(column) - 2)
                np.testing.assert_array_equal(
                    cached.evaluate(targets, origin, shape),
                    original.evaluate(targets, origin, shape),
                )
                for actual, expected in zip(
                    cached.gradient(targets, origin, shape),
                    original.gradient(targets, origin, shape),
                ):
                    np.testing.assert_array_equal(actual, expected)
                assert len(cached._spline_cache) <= 4


def test_evicted_and_mutated_profile_outputs_remain_correct():
    original, cached = model(ArchiveKeplerPRF), model(KeplerPRF)
    initial = [(100.1, 150.2)]
    expected = original.evaluate(initial, (98, 148), (5, 5))
    result = cached.evaluate(initial, (98, 148), (5, 5))
    initial_spline = cached.interpolate
    result[:] = -99
    np.testing.assert_array_equal(cached.evaluate(initial, (98, 148), (5, 5)), expected)
    assert cached.interpolate is initial_spline
    for i in range(10):
        targets = [(110.1 + i * 20, 180.2 + i * 30)]
        cached.evaluate(
            targets, (int(targets[0][0]) - 2, int(targets[0][1]) - 2), (5, 5)
        )
    assert len(cached._spline_cache) == 4
    assert all(spline is not initial_spline for spline in cached._spline_cache.values())
    np.testing.assert_array_equal(cached.evaluate(initial, (98, 148), (5, 5)), expected)


def test_calibration_models_do_not_share_splines():
    first, second = model(KeplerPRF, 1), model(KeplerPRF, 2)
    original_second = model(ArchiveKeplerPRF, 2)
    targets, origin, shape = [(100.1, 150.2)], (98, 148), (5, 5)
    first_profile = first.evaluate(targets, origin, shape)
    second_profile = second.evaluate(targets, origin, shape)
    assert not np.array_equal(first_profile, second_profile)
    assert first.interpolate is not second.interpolate
    np.testing.assert_array_equal(
        second_profile, original_second.evaluate(targets, origin, shape)
    )
