"""Exercise authenticated PSF routing and an exact declared aperture limit.

The FITS source, extraction coefficients and acceptances here are synthetic
software controls, never instrument qualification or physical admission.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numba import set_num_threads

from post_boi.campaign import calculate_quarters
from post_boi.prepare import prepare
from post_boi.protocol import freeze
from post_boi.psf_flat import ingest
from post_boi.psf_response import load_response
from post_boi.quarters import fold_quarters
from post_boi.records import candidate, digest, read, receipt, write
from post_boi.test_psf_flat import SOURCE
from post_boi.test_psf_flat import product as product


@pytest.fixture
def psf_case(product, tmp_path):
    fixtures = Path(__file__).parent / "fixtures/kepler10"
    photons = ingest(product[0], SOURCE, digest(product[0]), tmp_path / "input")
    stars = pd.read_csv(fixtures / "stars.csv", dtype={"ID": str})
    stars.loc[0, "ID"] = SOURCE
    stars.to_csv(tmp_path / "stars.csv", index=False)
    shutil.copyfile(fixtures / "background.csv", tmp_path / "background.csv")
    ids = stars["ID"].to_numpy(str)
    fractions = np.array([0.8, 0.2, 0, 0, 0, 0])
    with np.load(fixtures / "geometry.npz") as old:
        geometry = dict(source_id=ids, quarter=np.array([1]), xy_q1=old["xy_q3"])
        footprint = old["aperture_q3"]
    np.savez(tmp_path / "geometry.npz", **geometry, footprint_q1=footprint)
    np.savez(
        tmp_path / "aperture-limit.npz",
        **geometry,
        aperture_q1=footprint,
        flux_fraction_q1=fractions,
    )
    c = read(Path(__file__).with_name("candidate-template.json"))
    c.update(
        source_id=SOURCE,
        signal_id="synthetic-psf-integration-control",
        input_basis=__doc__,
        photometry=str(photons),
        flux_convention="psf_flat",
        exposure_days=1765.464 / 86400,
        prepared="prepared",
        native_review="native-review.json",
        readiness="readiness.json",
        scene="stars.csv",
        geometry="geometry.npz",
        aperture_response="signed_psf_flat",
        background="background.csv",
        recovered=dict(period_days=0.7, epoch_bkjd=130.4, duration_days=0.05),
        timing_review="One declared synthetic-control clock",
        cpu_budget=1,
        psf_response=dict(
            data="response.npz",
            review="response-review.json",
            input_review="input/input-review.json",
        ),
    )
    c["detrending"].update(window_days=0.5, polyorder=2)
    c["native"]["priors"] = "priors.json"
    c["probability"].update(
        supersample=20, noise="noise.npz", noise_basis="Synthetic constant noise"
    )
    write(tmp_path / "priors.json", dict(note=__doc__))
    clock = dict(
        source_id=SOURCE,
        signal_id=c["signal_id"],
        accepted=True,
        evidence=[receipt(tmp_path / "priors.json")],
        **c["recovered"],
        observed_depth_fraction=0.005,
        probability_noise_sigma=1e-4,
        probability_noise_basis="Synthetic control noise",
    )
    write(tmp_path / "native-review.json", clock)
    write(tmp_path / "readiness.json", clock)
    write(tmp_path / "candidate.json", c)
    prepare(tmp_path / "candidate.json", tmp_path / "prepared")
    c = candidate(tmp_path / "candidate.json")
    with np.load(photons) as native:
        n = len(native["time_bkjd"])
        sigma = np.full(n, 1e-4)
        np.savez(
            tmp_path / "noise.npz",
            source_id=SOURCE,
            time_bkjd=native["time_bkjd"],
            sigma=sigma,
            applied_fraction=np.ones(n),
        )
        # Reverse rows deliberately: lookup must use original index, not rank.
        order = np.arange(n)[::-1]
        np.savez(
            tmp_path / "response.npz",
            source_id=ids,
            original_index=order,
            time_bkjd=native["time_bkjd"][order],
            quarter=native["quarter"][order],
            coefficient=np.tile(fractions, (n, 1))[order],
            secondary_sigma=np.array([1e-4 / 0.8, 1e-4 / 0.2, *[np.nan] * 4]),
        )
    evidence = [receipt(tmp_path / "priors.json")]
    write(
        tmp_path / "method-review.json",
        dict(accepted=True, evidence=evidence, scope=__doc__),
    )
    inputs = dict(
        data=tmp_path / "response.npz",
        input_review=tmp_path / "input/input-review.json",
        photometry=photons,
        prepared=tmp_path / "prepared/detrended.npz",
        scene=tmp_path / "stars.csv",
        geometry=tmp_path / "geometry.npz",
        noise=tmp_path / "noise.npz",
    )
    write(
        tmp_path / "response-review.json",
        dict(
            accepted=True,
            source_id=SOURCE,
            signal_id=c["signal_id"],
            evidence=evidence,
            clocks=[c["recovered"]],
            inputs={k: receipt(v) for k, v in inputs.items()},
            method_review=receipt(tmp_path / "method-review.json"),
            normalization_basis="Declared synthetic aperture-limit coefficients",
            transfer_basis="Identity operator in this software control only",
            scene_basis="Kepler10 fixture with synthetic source ID, not a real scene",
            secondary_noise_basis="Exact aperture limit: observed sigma / response",
            uncertainty_basis="Declared synthetic constant noise",
        ),
    )
    with (
        np.load(tmp_path / "prepared/detrended.npz") as prepared,
        np.load(tmp_path / "noise.npz") as noise,
    ):
        packet = fold_quarters(
            prepared, clock, [], c["exposure_days"], noise, "psf_flat"
        )
    return c, clock, packet


def test_response_binds_original_rows_and_retains_missing_noise(psf_case):
    c, clock, packet = psf_case
    joint = load_response(c, clock, packet)
    np.testing.assert_array_equal(joint.response[:, 0], 0.8)
    assert np.isnan(joint.secondary_sigma[2:]).all()
    changed = dict(packet, quarter=packet["quarter"] + 1)
    with pytest.raises(ValueError, match="Prepared photon indices, times or quarters"):
        load_response(c, clock, changed)
    changed = dict(packet, time_bkjd=packet["time_bkjd"] + 1e-6)
    with pytest.raises(ValueError, match="Prepared photon indices, times or quarters"):
        load_response(c, clock, changed)


def test_unaccepted_method_cannot_freeze(psf_case, tmp_path):
    c, _, _ = psf_case
    path = tmp_path / "method-review.json"
    review = read(path)
    review["accepted"] = None
    path.write_text(json.dumps(review))
    path = Path(c["psf_response"]["review"])
    review = read(path)
    review["method_review"] = receipt(tmp_path / "method-review.json")
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="review remains unresolved"):
        freeze(tmp_path / "candidate.json", tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_psf_freeze_executes_own_source_and_binds_response(psf_case, tmp_path):
    freeze(tmp_path / "candidate.json", tmp_path / "campaign")
    root = tmp_path / "campaign/source"
    command = [
        sys.executable,
        "-B",
        "-m",
        "post_boi.campaign",
        "verify",
        "../plan.json",
    ]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    ok = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    with (tmp_path / "response.npz").open("ab") as stream:
        stream.write(b"changed")
    bad = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "Frozen file changed" in bad.stderr


def test_upstream_probability_call_matches_exact_aperture_limit(psf_case, tmp_path):
    c, clock, packet = psf_case
    reference = dict(
        c,
        flux_convention="raw_aperture",
        aperture_response="measured_kepler",
        geometry=str(tmp_path / "aperture-limit.npz"),
        psf_response=None,
    )
    set_num_threads(1)
    np.random.seed(20261027)
    psf = calculate_quarters(c, clock, packet, draws=4096)
    np.random.seed(20261027)
    aperture = calculate_quarters(reference, clock, packet, draws=4096)
    np.testing.assert_array_equal(psf.lnZ, aperture.lnZ)
    np.testing.assert_array_equal(psf.probs.prob, aperture.probs.prob)
    # Legacy scalar bookkeeping must preserve the actual secondary-screen noise.
    observed_sigma = clock["probability_noise_sigma"]
    for i in (0, 1):
        np.testing.assert_allclose(
            observed_sigma / psf.stars["fluxratio"].iloc[i],
            load_response(c, clock, packet).secondary_sigma[i],
            rtol=1e-15,
        )


def test_exact_fast_engine_preserves_signed_nonnormalized_evidence(psf_case):
    from fast_triceratops import engine

    c, clock, packet = psf_case
    path = Path(c["psf_response"]["data"])
    with np.load(path) as archive:
        response = dict(archive)
    rows = np.arange(len(response["original_index"]))
    response["coefficient"][:, 0] = 0.99 + 0.01 * np.sin(rows)
    response["coefficient"][:, 1] = 0.95 + 0.01 * np.cos(rows)
    response["coefficient"][:, 2] = 0.02 * np.sin(rows)
    response["secondary_sigma"][:3] = [1e-4 / 0.99, 1e-4 / 0.95, 1e-4 / 0.02]
    np.savez(path, **response)
    review_path = Path(c["psf_response"]["review"])
    review = read(review_path)
    review["inputs"]["data"] = receipt(path)
    review["normalization_basis"] = (
        "Declared synthetic nonnormalized signed-response control; no qualification"
    )
    review["secondary_noise_basis"] = "Declared synthetic intrinsic-source noise"
    review_path.write_text(json.dumps(review))
    set_num_threads(1)
    np.random.seed(20261027)
    stock = calculate_quarters(c, clock, packet, draws=4096)
    np.random.seed(20261027)
    with engine():
        fast = calculate_quarters(c, clock, packet, draws=4096)
    np.testing.assert_array_equal(stock.lnZ, fast.lnZ)
    np.testing.assert_array_equal(stock.probs.prob, fast.probs.prob)
    assert len(fast.probs) == 21


def test_frozen_psf_inputs_run_through_existing_fast_coordinator(psf_case, tmp_path):
    from fast_triceratops import workflow

    plan = freeze(tmp_path / "candidate.json", tmp_path / "campaign")
    experiment = workflow.create(
        plan,
        tmp_path / "software-control",
        mode="engine",
        workers=1,
        draws=256,
        repeats=2,
    )
    workflow.launch(experiment)
    summary = read(experiment.parent / "summary.json")
    assert len(summary["repeats"]) == 2
    assert [r["index"] for r in summary["repeats"]] == [0, 1]
    assert all(r["draws_per_scenario"] == 256 for r in summary["repeats"])
