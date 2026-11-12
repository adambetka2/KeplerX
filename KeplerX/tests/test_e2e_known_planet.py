"""End-to-end: TrES-2's KBonus light curve through the command line, `run.py batch`."""

import json
import subprocess
import sys

import numpy as np

from common import ROOT, phase, read, write
from runner.outputs import RunOutputs


def batch(fixtures, folder, config, isolation):
    folder.mkdir()
    truth = read(fixtures / "tres2-q0.json")
    config["execution"]["isolation"] = isolation
    settings = folder / "config.json"
    write(settings, config)
    request = {
        "input_kind": "native_arrays", "path": str(fixtures / "tres2-q0.npz"), "view": "psf_flat",
        "source_id": truth["source_id"], "time_system": "BKJD_TDB", "exposure_days": truth["exposure_days"],
        "flux_convention": "delivered_psf_flat", "provenance": truth,
        "stellar": {"teff": 5819, "logg": 4.4, "density_solar": 1.0, "basis": "Known-object software template"},
    }  # fmt: skip
    manifest = folder / "one-known-planet.jsonl"
    manifest.write_text(json.dumps({"case_id": "tres2", "photometry": request}) + "\n")
    output = folder / "batch"
    command = [
        sys.executable,
        "-B",
        str(ROOT / "run.py"),
        "--config",
        str(settings),
        "batch",
        str(manifest),
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read(output / "run.json")["state"] == "COMPLETE"
    outputs = RunOutputs(output)
    assert not (output / "tres2").exists()
    (state,) = outputs.database.execute("SELECT state FROM jobs").fetchone()
    assert state == "COMPLETE" and outputs.case_ids() == {"tres2"}
    detection = outputs.read_json("tres2", "detections/detections.json")
    epochs = truth["expected_epoch_bkjd"] + np.arange(4) * truth["expected_period_days"]
    found = [
        c for c in detection["candidates"]
        if np.all(abs(phase(epochs, c["period"], c["t0"])) < (c["tdur"] + truth["exposure_days"]) / 2)
    ]  # fmt: skip
    assert found
    for stage in ("prepared", "detections", "vetting"):
        record = outputs.read_json("tres2", f"{stage}/run.json")
        assert record["state"] == "COMPLETE" and record["isolation"] == isolation
    vetting = outputs.read_json("tres2", "vetting/vetting.json")["results"]
    planet = next(r for r in vetting if r["signal_id"] == found[0]["signal_id"])
    assert planet["metrics"]["depth_ppt"] > 6
    assert planet["disposition"]["state"] == "KEPT"
    bois = outputs.read_json("tres2", "bois.json")
    assert bois["results"][0]["state"] == "UNRESOLVED"
    assert bois["results"][0]["localization"]["state"] == "NOT_RUN"
    assert bois["decision_policy_applied"]
    return detection["candidates"], [r["metrics"] for r in vetting]


def test_known_planet_through_the_command_line(fixtures, tmp_path, config):
    config["detection"]["max_passes"] = 1
    runs = [
        batch(fixtures, tmp_path / isolation, config, isolation)
        for isolation in ("process", "worker")
    ]
    assert runs[0] == runs[1]
