import numpy as np

from post_boi.kbonus import closure, passing_quarters, quarter_noise


def test_closure_reproduces_a_complete_flux_budget():
    throughput = np.array([0.6, 0.05, 0.0])
    relative = np.array([1.0, 4.0, 10.0])
    sap = 1000 * (0.6 + 0.2)
    assert np.isclose(closure(throughput, relative, 1000, sap), 1.0)
    assert np.isnan(closure(throughput, relative, np.nan, sap))
    assert passing_quarters({1: 1.0, 2: 0.79, 3: 1.2, 4: np.nan}) == [1, 3]


def test_white_noise_keeps_formal_errors_and_red_noise_is_inflated():
    rng = np.random.default_rng(1)
    cadence, n = 0.0204, 4000
    time = np.arange(n) * cadence
    quarter = np.repeat([1, 2], n // 2)
    error = np.full(n, 1e-3)
    flux = 1 + rng.normal(0, 1e-3, n)
    red = np.convolve(rng.normal(0, 1e-3, n + 40), np.ones(40) / np.sqrt(40), "valid")[
        :n
    ]
    flux[quarter == 2] = 1 + rng.normal(0, 1e-3, n // 2) + red[: n // 2]
    sigma, rows = quarter_noise(
        flux, error, time, quarter, np.zeros(n, bool), cadence, 0.1
    )
    white, coloured = rows
    assert white["beta"] < 1.15 and white["applied_scale"] < 1.15
    assert coloured["beta"] > 1.3 and coloured["applied_scale"] > white["applied_scale"]
    assert np.all(sigma >= error)
