"""Cohort failures, resume integrity and explicit clock trials."""

import argparse
import json
from pathlib import Path

import pytest

from tess import __main__ as cli
from tess.archive import digest, write


def arguments(tmp_path):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/hat-p-7-s14.json").read_text()
    )
    source = tmp_path / "input.json"
    write(source, {"hosts": [fixture["host"]]})
    return argparse.Namespace(
        command="coverage",
        input=source,
        output=tmp_path / "output",
        settings=Path(__file__).with_name("settings.json"),
        sectors=None,
        source_ids=None,
        reserve_gib=0,
    )


def test_query_failure_is_not_zero_coverage_and_can_resume(tmp_path, monkeypatch):
    args = arguments(tmp_path)

    def unavailable(host):
        raise OSError("archive unavailable")

    monkeypatch.setattr(cli, "coverage", unavailable)
    assert cli.run(args) == 2
    assert "archive unavailable" in (args.output / "failures.jsonl").read_text()
    monkeypatch.setattr(cli, "coverage", lambda host: {"response": {"results": []}})
    assert cli.run(args) == 0
    result = json.loads((args.output / "summary.json").read_text())
    assert result["hosts"][0]["available_sectors"] == []
    assert (args.output / "failures.jsonl").exists()
    args.sectors = [14]
    with pytest.raises(ValueError, match="changed"):
        cli.run(args)


def test_changed_completed_photons_are_rejected(tmp_path):
    photons = tmp_path / "native.fits"
    photons.write_bytes(b"original photons")
    result = {
        "files": {},
        "inputs": [{"path": str(photons), "sha256": digest(photons)}],
    }
    cli.verify_files(tmp_path, result)
    photons.write_bytes(b"changed photons")
    with pytest.raises(ValueError, match="Changed FITS"):
        cli.verify_files(tmp_path, result)


def test_all_explicit_clocks_preserved_and_epoch_units_checked(tmp_path):
    args = arguments(tmp_path)
    catalogue = json.loads(args.input.read_text())
    signal = catalogue["hosts"][0]["signals"][0]
    alternative = {**signal["clock"], "period_days": signal["clock"]["period_days"] * 2}
    signal["clock_trials"] = [alternative]
    assert cli.validate(catalogue)["hosts"][0]["signals"][0]["clock_trials"] == [
        alternative
    ]
    signal["clock"]["epoch_bjd_tdb"] = 121.3585417
    with pytest.raises(ValueError, match="BJD_TDB"):
        cli.validate(catalogue)
