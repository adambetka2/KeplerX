"""Exact-cadence export contract for independent photometry comparison."""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
import comparison_inputs as inputs


def test_delivered_rows_align_by_cadence_not_row_order():
    table = pa.table({"cadenceno": [30, 10], "flux": [3.0, 1.0]})
    aligned, present = inputs.aligned(table, np.array([10, 20, 30]))
    np.testing.assert_array_equal(present, [True, False, True])
    np.testing.assert_allclose(aligned["flux"], [1, np.nan, 3], equal_nan=True)


def test_common_support_retains_only_joint_good_flattened_rows():
    n = 2000
    arrays = {
        "time": np.arange(n) * 0.0204336,
        "quality": np.zeros(n),
        "pos_corr1": np.zeros(n),
        "pos_corr2": np.zeros(n),
    }
    qdiag = {key: np.ones(n) for key in inputs.QDIAG}
    qdiag["sap_quality"] = np.zeros(n)
    stitched = {key: np.ones(n) for key in inputs.STITCHED}
    stitched.update(sap_quality=np.zeros(n), flatten_mask=np.zeros(n))
    stitched["psf_flat_kbonus_flux"][2] = np.nan
    stitched["flatten_mask"][4] = 1
    qdiag["psf_flux_err"][6] = 0
    mask = inputs.common_support(arrays, qdiag, stitched, np.ones(n, dtype=bool))
    np.testing.assert_array_equal(np.flatnonzero(~mask), [2, 4, 6])
