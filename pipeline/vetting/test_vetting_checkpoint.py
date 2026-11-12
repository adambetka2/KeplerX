"""Checkpoint replays retain science while allowing fresh detector timings."""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]
import vetting
import vetting_checkpoint as recovery


@pytest.fixture
def checkpoint_inputs(monkeypatch):
    config = {
        "disposition_policy": "/policy",
        "triage_execution": "SKIP_AFTER_FIRST_FAILURE",
    }
    cohort = [{
        "source_id": "123",
        "alerts": [{"depth": 0.001}],
        "shallow_eligibility": {
            "status": "ELIGIBLE",
            "reason": "SEARCH_COMPLETE",
            "elapsed_seconds": "1.0",
        },
    }]
    current = copy.deepcopy(cohort)
    current[0]["shallow_eligibility"]["elapsed_seconds"] = "2.0"
    data = {
        Path("/old/vetting.json"): config,
        Path("/policy"): {"threshold": 3, "implementation": {}},
        Path("/old/preparation.json"): {"sources": []},
        Path("/new/preparation.json"): {"sources": []},
        Path("/old/vetting/manifest.json"): {"catalogue_sha256": {}},
        Path("/new/vetting/manifest.json"): {"catalogue_sha256": {}},
        Path("/new/vetting/cohort.json"): current,
        Path("/new/vetting/metadata.json"): {"123": {"mask_sha256": "a"}},
        Path("/new/vetting/scene.json"): {"neighbours": []},
    }
    previous = {
        Path("/old/vetting/cohort.json.gz"): cohort,
        Path("/old/vetting/metadata.json.gz"): {"123": {"mask_sha256": "a"}},
        Path("/old/vetting/scene.json.gz"): {"neighbours": []},
    }
    monkeypatch.setattr(vetting, "read", data.__getitem__)
    monkeypatch.setattr(recovery, "packed", previous.__getitem__)
    return config, data, previous


def test_native_replay_timing_changes_preserve_both_records(checkpoint_inputs):
    config, data, previous = checkpoint_inputs
    before = copy.deepcopy((data, previous))
    recovery.compare_inputs(Path("/old"), Path("/new"), config)
    assert (data, previous) == before


@pytest.mark.parametrize("field", ["status", "reason", "source_id", "alerts"])
def test_native_replay_scientific_changes_remain_rejected(checkpoint_inputs, field):
    config, data, _ = checkpoint_inputs
    row = data[Path("/new/vetting/cohort.json")][0]
    target = row["shallow_eligibility"] if field in {"status", "reason"} else row
    target[field] = "changed"
    with pytest.raises(ValueError, match="Checkpoint cohort changed"):
        recovery.compare_inputs(Path("/old"), Path("/new"), config)


@pytest.mark.parametrize("filename", ["metadata", "scene"])
def test_other_evidence_timing_fields_remain_protected(checkpoint_inputs, filename):
    config, data, _ = checkpoint_inputs
    data[Path(f"/new/vetting/{filename}.json")]["elapsed_seconds"] = 1.0
    with pytest.raises(ValueError, match=f"Checkpoint {filename} changed"):
        recovery.compare_inputs(Path("/old"), Path("/new"), config)
