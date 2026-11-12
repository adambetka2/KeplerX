"""Exercise real repaired dependencies against independent probability/orbit code."""

from types import SimpleNamespace

import batman
import juliet
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import beta


@pytest.mark.parametrize("shapes", [(0.867, 3.03), (2, 5), (0.5, 0.5), (2, 0.7)])
def test_beta_density_and_support(shapes):
    x = np.r_[np.linspace(1e-8, 1 - 1e-8, 201), -0.1, 1.1]
    actual = [juliet.utils.evaluate_beta(v, shapes) for v in x]
    np.testing.assert_array_equal(actual, beta.logpdf(x, *shapes))


def test_mcmc_rejects_prior_before_evaluating_model():
    def forbidden(theta):
        raise AssertionError("A zero-prior proposal reached the transit solver")

    invalid = SimpleNamespace(logprior=lambda theta: -np.inf, loglike=forbidden)
    assert juliet.fit.logprob(invalid, [1]) == -np.inf
    supported = SimpleNamespace(logprior=lambda theta: -3.0, loglike=lambda theta: 8.0)
    assert juliet.fit.logprob(supported, [1]) == 5.0


@pytest.mark.parametrize("ecc", [0.01, 0.8, 0.95, 0.999, 1 - 1e-8])
def test_eccentric_solver_against_bracketed_reference(ecc):
    # omega=pi/2 puts conjunction at periastron, so M=2*pi*time for P=1.
    mean = np.r_[np.linspace(-np.pi, np.pi, 401), [-1e-8, 1e-8]]
    time = mean / (2 * np.pi)
    actual = batman._rsky._getf(time, 0.0, 1.0, 20.0, np.pi / 2, ecc, np.pi / 2, 1, 1)
    anomaly = np.array(
        [
            brentq(lambda e: e - ecc * np.sin(e) - m, m - ecc, m + ecc, xtol=1e-15)
            for m in mean
        ]
    )
    expected = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(anomaly / 2),
        np.sqrt(1 - ecc) * np.cos(anomaly / 2),
    )
    np.testing.assert_allclose(np.sin(actual), np.sin(expected), rtol=0, atol=2e-8)
    np.testing.assert_allclose(np.cos(actual), np.cos(expected), rtol=0, atol=2e-8)
    shifted = batman._rsky._getf(
        time + 2000, 0.0, 1.0, 20.0, np.pi / 2, ecc, np.pi / 2, 1, 1
    )
    # Time rounding near periastron is magnified at extreme eccentricity.
    away = abs(mean) > 0.02
    np.testing.assert_allclose(np.cos(shifted[away]), np.cos(expected[away]), atol=2e-9)
