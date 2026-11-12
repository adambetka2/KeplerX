"""End-to-end: what the pipeline does with broken inputs."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from common import ROOT, load_packet, read, receipt, save_packet, write
from vetter.program import run as vet


def cli(*args):
    return subprocess.run(
        [sys.executable, "-B", str(ROOT / "run.py"), *map(str, args)],
        capture_output=True,
        text=True,
    )


def test_a_failed_input_keeps_its_record_and_its_output_is_not_reused(tmp_path):
    output = tmp_path / "failed"
    first = cli("prepare", tmp_path / "missing.json", output)
    assert first.returncode != 0
    record = read(output / "run.json")
    assert record["state"] == "ERROR" and "FileNotFoundError" in record["error"]
    assert record["source"]
    second = cli("prepare", tmp_path / "missing.json", output)
    assert "Output already exists" in second.stderr
    assert read(output / "run.json") == record


def test_a_changed_packet_is_refused(tmp_path, fixtures):
    folder = tmp_path / "packet"
    folder.mkdir()
    with np.load(fixtures / "tres2-q0.npz") as archive:
        save_packet(
            folder,
            {name: archive[name] for name in archive.files},
            {"source_id": "2131314401800665344"},
        )
    with (folder / "photometry.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_packet(folder)


def test_failing_fits_keep_every_signal_and_accept_none(fixtures, tmp_path, config):
    record = read(fixtures / "vetting-failure.json")
    prepared, detections, output = (
        tmp_path / name for name in ("prepared", "detections", "vetting")
    )
    for directory in (prepared, detections, output):
        directory.mkdir()
    with np.load(fixtures / "vetting-failure.npz") as archive:
        save_packet(
            prepared, {name: archive[name] for name in archive.files}, record["packet"]
        )
    write(
        detections / "detections.json",
        record["detections"] | {"prepared": receipt(prepared / "packet.json")},
    )
    # Signal 8's cosine fit fails from upstream's sigma**2 refinement.
    config = config | {
        "vetting": config["vetting"] | {"refinement_weight": "sigma_squared"}
    }
    result = vet(
        SimpleNamespace(prepared=prepared, detections=detections, out=output), config
    )
    assert len(result["results"]) == 9
    errors = [row for row in result["results"] if row["metric_errors"]]
    assert errors and all(row["disposition"]["state"] == "UNRESOLVED" for row in errors)
    assert any(
        e["component"] == "cosine_vs_transit"
        for row in errors
        for e in row["metric_errors"]
    )
    assert all(
        row["disposition"]["state"] in ("UNRESOLVED", "KEPT", "DISMISSED")
        for row in result["results"]
    )
