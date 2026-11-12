"""Golden: the Validator's copy of the TRICERATOPS V3 probability engine."""

import contextlib

import numpy as np
import pytest


def v3_runtime():
    try:
        from triceratops import marginal_likelihoods

        from validator.probability.kernels import require_pinned
        from validator.probability.scenarios import PINNED_V3

        require_pinned(marginal_likelihoods, PINNED_V3)
    except (ImportError, RuntimeError):
        return False
    return True


def kepler10_v3():
    from validator.probability.gold import FIXTURE, kepler10_target
    from validator.quarters import QuarterLikelihood

    target = kepler10_target()
    raw = np.loadtxt(FIXTURE / "lightcurve.csv", delimiter=",")
    labels = np.minimum(np.arange(len(raw)) * 200 // len(raw), 199)
    weight = 1 / raw[:, 2] ** 2
    total = np.bincount(labels, weights=weight)
    time = np.bincount(labels, weights=weight * raw[:, 0]) / total
    binned = np.bincount(labels, weights=weight * raw[:, 1]) / total
    ids = np.array(target.stars["ID"].values[:2], dtype=str)
    fractions = np.array([[0.9, 0.1], [0.7, 0.3], [0.95, 0.05]])
    packet = {
        "time": time, "flux": raw[:, 1], "sigma": raw[:, 2], "bin_index": labels,
        "quarter": 3 + np.arange(len(raw)) % 3, "applied_fraction": np.ones(len(raw)),
    }  # fmt: skip
    quarters = QuarterLikelihood(packet, ids, [3, 4, 5], fractions)
    target.stars["fluxratio"] = 0.0
    target.stars.loc[:1, "fluxratio"] = fractions.mean(axis=0)
    target.stars["tdepth"] = 0.0
    target.stars.loc[:1, "tdepth"] = 0.00019 / fractions.max(axis=0)
    return target, quarters, time, binned, float(np.median(total**-0.5))


@pytest.mark.skipif(not v3_runtime(), reason="needs the frozen V3 TRICERATOPS runtime")
def test_probability_engine_copy_matches_upstream_triceratops():
    from validator.probability import engine
    from validator.probability.gold import EXPTIME, P_ORB

    results = []
    for context in (contextlib.nullcontext(), engine()):
        with context:
            target, quarters, time, binned, sigma = kepler10_v3()
            np.random.seed(20261025)
            target.calc_probs(
                time=time, flux_0=binned, flux_err_0=sigma, P_orb=P_ORB, N=8192,
                parallel=True, verbose=0, flatpriors=False, exptime=EXPTIME,
                nsamples=20, contrast_curve_file=None, filt=None, molusc_file=None,
                dilution_corrected=False, joint=quarters,
            )  # fmt: skip
            results.append(target)
    stock, copied = results
    assert len(copied.probs) == 18
    assert np.array_equal(
        np.asarray(copied.lnZ).view(np.uint64), np.asarray(stock.lnZ).view(np.uint64)
    )
    assert copied.probs.equals(stock.probs)
    assert copied.stars.equals(stock.stars)
    assert (copied.FPP, copied.NFPP) == (stock.FPP, stock.NFPP)
