"""Check flat preparation contracts and likelihood transfer across sparse segments."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evolve_native_coherence as incumbent
import evolve_psf_flat as flat


def test_flat_engine_does_not_change_incumbent_preprocessing():
    assert incumbent.COMMON.detrend is not flat.NATIVE.COMMON.detrend
    assert incumbent.PHYSICAL.spline_basis is not flat.NATIVE.PHYSICAL.spline_basis
    with pytest.raises(ValueError, match="PSF Flat"):
        flat.scan(np.empty(0), {"input_view": "sap_qdiag"})


def test_sparse_segment_intercept_has_full_rank_and_preserves_transit_information():
    time = np.r_[np.arange(100) / 48, 10.0]
    quarter = np.ones(len(time), dtype=int)
    selected = flat.NATIVE.PHYSICAL.TEMPLATE_CONFIG
    basis, segment, _ = flat.intercept_basis(time, quarter, selected)
    assert basis.shape == (101, 2)
    np.testing.assert_array_equal(np.bincount(segment), [100, 1])
    template = np.zeros(101)
    template[20:24] = 1
    flux = 0.003 - 0.0008 * template
    metric = flat.NATIVE.PHYSICAL.ProfileMetric(
        time, flux, np.full(101, 0.0001), quarter, selected
    )
    fitted = metric.fit(np.arange(101), template)
    assert fitted["depth"] == pytest.approx(0.0008)
    expected = 0.0008 / 0.0001 * np.sqrt(4 * (1 - 4 / 100))
    assert fitted["snr"] == pytest.approx(expected)
    np.testing.assert_allclose(
        flat.constant_trend(time, flux, 97, {"method": "quarter_median"}), 0.003
    )
