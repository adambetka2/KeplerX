"""Offline Kepler-10b gold case: the stock target of post_boi/test_gold.py."""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE = Path(__file__).resolve().parent.parent / "post_boi/fixtures/kepler10"
P_ORB, EXPTIME, DEPTH, SEED = 0.837490, 0.020433598216918748, 0.00019, 20261020


def kepler10_target():
    from triceratops.triceratops import target

    reference = target.__new__(target)
    reference.ID, reference.mission = 11904151, "Kepler"
    reference.stars = pd.read_csv(FIXTURE / "stars.csv", dtype={"ID": str})
    reference.trilegal_fname, reference.trilegal_url = str(FIXTURE / "background.csv"), None
    with np.load(FIXTURE / "geometry.npz") as geometry:
        reference.sectors = geometry["quarter"]
        reference.pix_coords = [geometry["xy_q3"]]
        apertures = [geometry["aperture_q3"]]
    reference.calc_depths(DEPTH, all_ap_pixels=apertures, dilution_corrected=False)
    return reference


def typical_lightcurve():
    """A typical BOI: 12.3 d, 500 ppm, 0.17 d, 200 bins over +-2 durations, 200 ppm noise."""
    time = np.linspace(-0.34, 0.34, 200)
    depth = 5e-4 * np.clip((0.085 - np.abs(time)) / 0.02, 0, 1)
    flux = 1 - depth + np.random.default_rng(12).normal(0, 2e-4, time.size)
    return time, flux, 2e-4, 12.3, 40


def case_probs(draws, seed=SEED, nsamples=20, case="kepler10"):
    """Run stock calc_probs (or the accelerated one, inside accelerated())."""
    if case == "kepler10":
        raw = np.loadtxt(FIXTURE / "lightcurve.csv", delimiter=",")
        time, flux, sigma, period = raw[:, 0], raw[:, 1], float(raw[:, 2].mean()), P_ORB
    else:
        time, flux, sigma, period, nsamples = typical_lightcurve()
    reference = kepler10_target()
    np.random.seed(seed)
    reference.calc_probs(
        time=time, flux_0=flux, flux_err_0=sigma,
        P_orb=period, N=draws, parallel=True, verbose=0, flatpriors=False,
        exptime=EXPTIME, nsamples=nsamples, contrast_curve_file=None, filt=None,
        molusc_file=None, dilution_corrected=False,
    )
    return reference
