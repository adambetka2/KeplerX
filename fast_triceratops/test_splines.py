"""Paired spline/stellar controls against the pinned scipy/FITPACK evaluations."""

import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline
from triceratops import funcs

from fast_triceratops.engine import splines


@pytest.mark.parametrize("seed", [12, 139, 20261025])
def test_shared_stellar_basis_matches_original_bits(seed):
    rng = np.random.default_rng(seed)
    nodes = np.unique(np.r_[funcs.Mass_nodes_Torres, funcs.Mass_nodes_cdwrf, 0.63])
    masses = np.r_[
        rng.uniform(-0.2, 5, 20000),
        nodes,
        np.nextafter(nodes, -np.inf),
        np.nextafter(nodes, np.inf),
        np.nan,
    ]
    rng.shuffle(masses)
    radius, teff = (
        rng.uniform(0.05, 4, masses.size),
        rng.uniform(2000, 15000, masses.size),
    )
    expected = funcs.stellar_relations(masses, radius, teff)
    actual = splines.stellar_relations(masses, radius, teff)
    for left, right in zip(actual, expected, strict=True):
        assert np.array_equal(left.view(np.uint64), right.view(np.uint64))


@pytest.mark.parametrize("filt", ["TESS", "Vis", "J", "H", "K"])
def test_flux_spline_keeps_numpy_power_and_bits(filt):
    rng = np.random.default_rng(20261025)
    masses = np.r_[rng.uniform(-0.3, 7, 20000), 0.1, 0.63, 1.0, 3.0, np.nan]
    expected, actual = (
        funcs.flux_relation(masses, filt),
        splines.flux_relation(masses, filt),
    )
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_different_knots_do_not_share_a_basis():
    x = np.arange(6.0)
    radius = InterpolatedUnivariateSpline(x, x)
    teff = InterpolatedUnivariateSpline(x + 0.1, x)
    with pytest.raises(RuntimeError, match="identical knots"):
        splines.pair(radius, teff)
