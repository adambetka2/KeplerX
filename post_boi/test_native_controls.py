"""Exercise the real native entrypoint beyond both old physical-domain boxes.

Short inference chains test integration/domain support, not convergence/admission.
"""

import pickle
from pathlib import Path

import batman
import numpy as np
import pytest

from post_boi.native import duration_samples, fit
from post_boi.prepare import prepare
from post_boi.records import read, write


@pytest.mark.parametrize(
    "period,k,rho,ecc",
    [(40.0, 0.02, 0.3, 0.0), (3.0, 0.4, 58.0, 0.0), (10.0, 0.08, 1.0, 0.2)],
)
def test_real_native_preparation_and_juliet_domain(tmp_path, period, k, rho, ecc):
    c = read(Path(__file__).with_name("candidate-template.json"))
    time = np.arange(-2, 2, 0.020433598216918748)
    density = rho * 1409.779824307526
    axis = (6.67430e-11 * density * (period * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, period, k, axis
    params.inc = np.rad2deg(np.arccos(0.2 / axis))
    params.ecc, params.w, params.u, params.limb_dark = (
        ecc,
        90.0,
        [0.3, 0.2],
        "quadratic",
    )
    model = batman.TransitModel(
        params, time, supersample_factor=20, exp_time=0.020433598216918748
    )
    flux = model.light_curve(params)
    source = "2102521666898377600"
    np.savez_compressed(
        tmp_path / "photons.npz",
        source_id=source,
        time_bkjd=time,
        flux=flux,
        flux_err=np.full(len(time), 50e-6),
        quarter=np.full(len(time), 3),
        valid=np.ones(len(time), bool),
    )

    def normal(x, s):
        return dict(distribution="normal", hyperparameters=[x, s])

    def fixed(x):
        return dict(distribution="fixed", hyperparameters=x)

    priors = dict(
        P_p1=normal(period, 0.001),
        t0_p1=normal(0.0, 0.005),
        p_p1=normal(k, k * 0.05),
        b_p1=dict(distribution="uniform", hyperparameters=[0.0, 0.8]),
        rho=normal(density, density * 0.05),
        q1_Kepler=fixed(0.25),
        q2_Kepler=fixed(0.3),
        ecc_p1=fixed(0.0),
        omega_p1=fixed(90.0),
        mdilution_Kepler=fixed(1.0),
        mflux_Kepler=fixed(0.0),
        sigma_w_Kepler=fixed(0.0),
    )
    if ecc:
        priors["ecc_p1"] = dict(distribution="beta", hyperparameters=[0.867, 3.03])
    write(tmp_path / "priors.json", priors)
    d = float(
        duration_samples(
            dict(
                P_p1=np.array([period]),
                p_p1=np.array([k]),
                b_p1=np.array([0.2]),
                rho=np.array([density]),
                ecc_p1=np.array([ecc]),
            ),
            priors,
        )[0]
    )
    c.update(
        source_id=source,
        signal_id="domain-control",
        input_basis="Independent batman synthetic domain control; not an admission",
        photometry="photons.npz",
        flux_convention="raw_aperture",
        exposure_days=0.020433598216918748,
        prepared="prepared",
        recovered=dict(period_days=period, epoch_bkjd=0.0, duration_days=d),
    )
    c["detrending"]["window_days"] = 2.0
    c["native"].update(
        priors="priors.json",
        prior_basis="Domain-control measurements with explicit five-percent errors",
        seed=20261020,
        starting_point=dict(P_p1=period, t0_p1=0.0, p_p1=k, b_p1=0.2, rho=density),
        supersample=20,
        fit_options=dict(
            sampler="emcee", nwalkers=20, nsteps=40, nburnin=10, nthreads=None
        ),
    )
    if ecc:
        c["native"]["starting_point"]["ecc_p1"] = ecc
    write(tmp_path / "candidate.json", c)
    prepare(tmp_path / "candidate.json", tmp_path / "prepared")
    fit(tmp_path / "candidate.json", tmp_path / "prepared", tmp_path / "native")
    result = read(tmp_path / "native/summary.json")
    if ecc:
        with (tmp_path / "native/juliet/emcee_posteriors.pkl").open("rb") as stream:
            chains = pickle.load(stream)["posteriors_per_walker"]
        assert np.all(np.any(np.diff(chains, axis=0) != 0, axis=(0, 2)))
    assert np.isfinite(result["radius_ratio_snr_quantile"])
    assert read(tmp_path / "native/review.json")["accepted"] is None
    with np.load(tmp_path / "native/posterior.npz") as samples:
        assert (
            np.isfinite(samples["loglike"]).all()
            if "loglike" in samples
            else len(samples["p_p1"]) > 0
        )
        if k > 0.3:
            assert np.any(samples["p_p1"] > 0.3)
        if rho > 50:
            assert np.any(samples["rho"] > 50 * 1409.779824307526)
    if period == 40:
        assert result["quantiles_16_50_84"]["duration_days"][1] * 24 > 6
