"""Test frozen-source execution and a real changed-input rejection."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from post_boi.prepare import prepare
from post_boi.protocol import freeze
from post_boi.records import read, receipt, write


def test_freeze_executes_own_source_and_rejects_changed_input(tmp_path):
    fixtures = Path(__file__).parent / "fixtures/kepler10"
    for name in ("stars.csv", "geometry.npz", "background.csv"):
        shutil.copyfile(fixtures / name, tmp_path / name)
    c = read(Path(__file__).with_name("candidate-template.json"))
    time = np.arange(-3, 3, 0.020433598216918748)
    np.savez_compressed(
        tmp_path / "photons.npz",
        source_id="377780790",
        time_bkjd=time,
        flux=np.ones(len(time)),
        flux_err=np.full(len(time), 1e-4),
        quarter=np.full(len(time), 3),
        valid=np.ones(len(time), bool),
    )
    c.update(
        source_id="377780790",
        signal_id="freeze-control",
        input_basis="Integrity control only; not a physical admission",
        photometry="photons.npz",
        flux_convention="raw_aperture",
        exposure_days=0.020433598216918748,
        prepared="prepared",
        native_review="native-review.json",
        readiness="readiness.json",
        scene="stars.csv",
        geometry="geometry.npz",
        aperture_response="measured_kepler",
        background="background.csv",
        recovered=dict(period_days=2.0, epoch_bkjd=0.0, duration_days=0.1),
        timing_review="One clock in this integrity control",
    )
    c["detrending"]["window_days"] = 1.0
    c["native"]["priors"] = "priors.json"
    c["probability"]["supersample"] = 20
    c["probability"]["noise"] = "noise.npz"
    c["probability"]["noise_basis"] = "Synthetic constant noise for integrity control"
    np.savez(
        tmp_path / "noise.npz",
        source_id="377780790",
        time_bkjd=time,
        sigma=np.full(len(time), 1e-4),
        applied_fraction=np.ones(len(time)),
    )
    with np.load(tmp_path / "geometry.npz") as archive:
        geometry = dict(archive)
    geometry["flux_fraction_q3"] = np.array([1.0, 0, 0, 0, 0, 0])
    np.savez(tmp_path / "geometry.npz", **geometry)
    write(
        tmp_path / "priors.json",
        dict(note="Integrity test fixture, never used for inference"),
    )
    write(tmp_path / "candidate.json", c)
    evidence = [receipt(tmp_path / "priors.json")]
    review = dict(
        source_id=c["source_id"],
        signal_id=c["signal_id"],
        accepted=True,
        evidence=evidence,
        period_days=2.0,
        epoch_bkjd=0.0,
        duration_days=0.1,
        observed_depth_fraction=0.00019,
        probability_noise_sigma=1e-4,
        probability_noise_basis="Integrity-control constant noise",
    )
    write(tmp_path / "native-review.json", review)
    write(tmp_path / "readiness.json", review)
    prepare(tmp_path / "candidate.json", tmp_path / "prepared")
    plan = freeze(tmp_path / "candidate.json", tmp_path / "campaign")
    env = dict(
        os.environ,
        PYTHONPATH=str(tmp_path / "campaign/source"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = [sys.executable, "-B", "-m", "post_boi.campaign", "verify", str(plan)]
    ok = subprocess.run(
        command,
        cwd=tmp_path / "campaign/source",
        env=env,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "VERIFIED" in ok.stdout
    with (tmp_path / "priors.json").open("a") as f:
        f.write("\n")
    bad = subprocess.run(
        command,
        cwd=tmp_path / "campaign/source",
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "Frozen file changed" in bad.stderr
