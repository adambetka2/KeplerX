"""Frozen official Kepler-10b example versus the real stock probability call."""

from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads
from triceratops.triceratops import target

from post_boi.campaign import calculate, verify_evidence
from post_boi.records import read, receipt


def test_official_kepler10_stock_evidence():
    root = Path(__file__).parent / "fixtures/kepler10"
    provenance = read(root / "sources.json")
    for name, checksum in provenance["sha256"].items():
        assert receipt(root / name)["sha256"] == checksum
    raw = np.loadtxt(root / "lightcurve.csv", delimiter=",")
    config = dict(
        source_id="377780790",
        scene=root / "stars.csv",
        geometry=root / "geometry.npz",
        background=root / "background.csv",
        aperture_response="upstream_gaussian",
        flux_convention="raw_aperture",
        exposure_days=0.020433598216918748,
        probability=dict(
            supersample=20, contrast_curve=None, contrast_filter=None, molusc=None
        ),
    )
    clock = dict(
        period_days=0.837490,
        observed_depth_fraction=0.00019,
        probability_noise_sigma=float(raw[:, 2].mean()),
    )
    set_num_threads(10)
    np.random.seed(20261020)
    result = calculate(
        config,
        clock,
        dict(time=raw[:, 0], flux=raw[:, 1], flux_err=raw[:, 2]),
        draws=32768,
    )
    reference = target.__new__(target)
    reference.ID, reference.mission = 11904151, "Kepler"
    reference.stars = pd.read_csv(root / "stars.csv", dtype={"ID": str})
    reference.trilegal_fname, reference.trilegal_url = (
        str(root / "background.csv"),
        None,
    )
    with np.load(root / "geometry.npz") as geometry:
        reference.sectors = geometry["quarter"]
        reference.pix_coords = [geometry["xy_q3"]]
        apertures = [geometry["aperture_q3"]]
    reference.calc_depths(0.00019, all_ap_pixels=apertures, dilution_corrected=False)
    np.random.seed(20261020)
    reference.calc_probs(
        time=raw[:, 0],
        flux_0=raw[:, 1],
        flux_err_0=float(raw[:, 2].mean()),
        P_orb=0.837490,
        N=32768,
        parallel=True,
        verbose=0,
        flatpriors=False,
        exptime=0.020433598216918748,
        nsamples=20,
        contrast_curve_file=None,
        filt=None,
        molusc_file=None,
        dilution_corrected=False,
    )
    np.testing.assert_array_equal(result.lnZ, reference.lnZ)
    np.testing.assert_allclose(
        result.probs.prob, reference.probs.prob, rtol=1e-12, atol=1e-12
    )
    with np.load(root / "expected.npz") as expected:
        np.testing.assert_array_equal(
            np.isfinite(result.lnZ), np.isfinite(expected["reference_lnZ"])
        )
        np.testing.assert_allclose(
            result.lnZ, expected["reference_lnZ"], rtol=1e-12, atol=1e-8
        )
        # Cross-architecture probability error follows the existing lnZ budget:
        # |delta logZ_i|<=epsilon bounds posterior L1 by exp(2*epsilon)-1.
        finite = np.isfinite(expected["reference_lnZ"])
        epsilon = float(np.max(1e-8 + 1e-12 * abs(expected["reference_lnZ"][finite])))
        assert np.abs(result.probs.prob - expected["reference_prob"]).sum() <= np.expm1(
            2 * epsilon
        )


def test_saved_evidence_cannot_disagree_with_reported_probability(tmp_path):
    import pandas as pd
    import pytest
    from triceratops._numerics import _normalize_probabilities

    lnz = np.arange(15.0) - 20
    probability, _ = _normalize_probabilities(lnz)
    np.savez(tmp_path / "evidence.npz", log_evidence=lnz)
    pd.DataFrame(dict(prob=probability)).to_csv(tmp_path / "scenarios.csv", index=False)
    saved = dict(
        files=[receipt(tmp_path / name) for name in ("evidence.npz", "scenarios.csv")],
        FPP=1 - probability[[0, 3, 9]].sum(),
        NFPP=0.0,
    )
    verify_evidence(saved)
    saved["FPP"] = 0.0
    with pytest.raises(ValueError, match="differ from upstream evidence"):
        verify_evidence(saved)
