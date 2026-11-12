"""Replay a real Flat proposal into the shared vetting input and event contracts."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_subset as subset
import flat_vetting as followup


def test_real_proposal_handoff_preserves_frozen_records_and_supplied_clocks(tmp_path):
    root = Path(__file__).resolve().parents[2]
    policy = followup.read(root / "results/psf-flat-real2501-local-20261015.json")
    detector = Path(policy["out"])
    if not (detector / "prepared-cases.json").is_file():
        pytest.skip("Prepared real-source run is unavailable")
    cases = subset.API.read(detector / "prepared-cases.json")
    design = policy["design"]["name"]
    task = None
    for case in cases:
        path = detector / "pilot" / design / (subset.API.job_key(case) + ".json")
        if (
            path.is_file()
            and (
                root
                / "discSSD/data/kbonus/lc2"
                / f"shard-{int(case['shard']):04d}/cadence.parquet"
            ).is_file()
        ):
            result = followup.read(path)
            if result.get("proposals"):
                task = case, path, result
                break
    if task is None:
        pytest.skip("A completed local real proposal is unavailable")
    proposal, item = followup.export_source(
        task, {"pipeline_root": str(root), "out": str(tmp_path)}
    )
    case, _, result = task
    assert proposal["shallow_rows"] == result["proposals"]
    assert followup.digest(item["arrays"]) == item["sha256"]
    assert len(proposal["supplied_hypotheses"]) == len(result["proposals"])
    identities = [event["event_id"] for event in proposal["alerts"]]
    assert len(set(identities)) == len(identities)
    for original, hypothesis in zip(
        result["proposals"], proposal["supplied_hypotheses"]
    ):
        assert hypothesis["detector_origin"] == followup.ORIGIN
        assert hypothesis["signal_id"].startswith("dwarfs-physical-v1-")
        assert hypothesis["period_days"] == original["period_days"]
        assert hypothesis["epoch_bkjd"] == original["t0_bkjd"]
        assert hypothesis["duration_days"] == original["duration_hours"] / 24
        assert hypothesis["forced_photometry_views"] == ["psf_flat"]
        events = [
            event
            for event in proposal["alerts"]
            if event["event_id"] in hypothesis["alert_ids"]
        ]
        assert events
        for event in events:
            centre = original["t0_bkjd"] + event["cycle"] * original["period_days"]
            duration = original["duration_hours"] / 24
            assert event["photometry"] == "psf_flat"
            assert event["start_bkjd"] == centre - duration / 2
            assert event["end_bkjd"] == centre + duration / 2
    with np.load(item["arrays"], allow_pickle=False) as raw:
        curve = subset.pilot.photometry(
            {key: raw[key] for key in raw.files}, "psf_flat"
        )
        records = subset.pilot.prepare_curve_records(curve, mask_flatten=True)
        assert hashlib.sha256(records.tobytes()).hexdigest() == case["records_sha256"]
