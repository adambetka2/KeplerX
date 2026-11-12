"""The static engine copy must retain the complete probability-call outputs."""

import contextlib

import numpy as np
import pytest

from fast_triceratops import engine
from fast_triceratops.test_joint import kepler10_v3, v3_runtime


@pytest.mark.skipif(not v3_runtime(), reason="requires the pinned V3 runtime")
def test_static_engine_kepler10_quarter_and_neighbour_parity():
    from fast_triceratops.gold import EXPTIME, P_ORB

    results = []
    for context in (contextlib.nullcontext(), engine()):
        with context:
            target, quarters, time, binned, sigma = kepler10_v3()
            np.random.seed(20261025)
            target.calc_probs(
                time=time,
                flux_0=binned,
                flux_err_0=sigma,
                P_orb=P_ORB,
                N=8192,
                parallel=True,
                verbose=0,
                flatpriors=False,
                exptime=EXPTIME,
                nsamples=20,
                contrast_curve_file=None,
                filt=None,
                molusc_file=None,
                dilution_corrected=False,
                joint=quarters,
            )
            results.append(target)
    stock, fast = results
    assert len(fast.probs) == 18
    assert np.array_equal(
        np.asarray(fast.lnZ).view(np.uint64), np.asarray(stock.lnZ).view(np.uint64)
    )
    assert fast.probs.equals(stock.probs)
    assert fast.stars.equals(stock.stars)
    assert (fast.FPP, fast.NFPP) == (stock.FPP, stock.NFPP)
