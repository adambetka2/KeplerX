"""A null flux survivor must not consume nominal pixels or vanish from accounting."""

import json
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
import vetting_pixels as pixels
from deterministic.signal_contracts import null_proposal_outcome


@pytest.fixture
def null_run(tmp_path, monkeypatch):
    manifest = tmp_path / "null.json"
    manifest.write_text(json.dumps({"sources": {"123": {}, "456": {}}}))
    config = {
        "input_manifest": str(manifest),
        "queue": str(tmp_path / "queue.json"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(vetting, "CONFIG", path)
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    monkeypatch.setattr(pixels, "RUN", tmp_path)
    contract = {"id": "upper-clipped-quarter-nulls-v1", "variant": "inverted"}
    monkeypatch.setattr(vetting, "null_manifest", lambda root, path: contract)
    monkeypatch.setattr(vetting, "check_run", lambda queue: None)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "123": {"origin_tpf_kic": "45"},
                "456": {"origin_tpf_kic": "45"},
            }
        )
    )
    (tmp_path / "cohort.json").write_text(
        json.dumps(
            [
                {"source_id": "123"},
                {"source_id": "456"},
            ]
        )
    )
    (tmp_path / "sources").mkdir()
    for sid, quarters in (("123", [1, 2]), ("456", [2, 3])):
        (tmp_path / "sources" / f"{sid}.json").write_text(
            json.dumps(
                {
                    "pixel_jobs": [{"quarter": quarter} for quarter in quarters],
                }
            )
        )
    return tmp_path, manifest, path


def test_null_dispatch_never_reads_nominal_inventory(null_run, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A transformed null accessed nominal pixels")

    monkeypatch.setattr(pixels, "indexed_pixels", forbidden)
    monkeypatch.setattr(pixels, "retrieve_frozen_pixels", forbidden)
    monkeypatch.setattr(pixels, "retrieve_experiment", forbidden)
    pixels.retrieve()
    folder, manifest, _ = null_run
    result = json.loads((folder / "pixel-retrieval.json").read_text())
    assert result["parents"] == 1
    assert result["null_input_manifest"]["sha256"] == vetting.digest(manifest)
    quarters = result["results"][0]["quarters"]
    assert set(quarters) == {"1", "2", "3"}
    assert all(
        item == {"status": "NULL_TRANSFORM_HAS_NO_MATCHING_PIXELS"}
        for item in quarters.values()
    )


@pytest.mark.parametrize("binding", ["frozen", "controlled"])
def test_null_dispatch_rejects_pixel_substitution(null_run, binding):
    folder, manifest, path = null_run
    if binding == "frozen":
        config = json.loads(path.read_text())
        config["frozen_pixel_inputs"] = {"path": "original.json"}
        path.write_text(json.dumps(config))
    else:
        specification = json.loads(manifest.read_text())
        specification["sources"]["123"]["pixels"] = {"1": "original.fits"}
        manifest.write_text(json.dumps(specification))
    with pytest.raises(ValueError, match="cannot use original or controlled"):
        pixels.retrieve()
    assert not (folder / "pixel-retrieval.json").exists()


def test_invalid_null_contract_cannot_fall_back_to_original_pixels(
    null_run, monkeypatch
):
    def invalid(*args):
        raise ValueError("Invalid null construction")

    monkeypatch.setattr(vetting, "null_manifest", invalid)
    with pytest.raises(ValueError, match="Invalid null construction"):
        pixels.retrieve()


def test_null_localization_preserves_flux_survivor_without_reading_a_tpf(
    null_run, monkeypatch
):
    pixels.retrieve()
    folder, _, _ = null_run
    quarters = json.loads((folder / "pixel-manifests/45.json").read_text())["quarters"]

    def forbidden(*args, **kwargs):
        pytest.fail("Missing null pixels were treated as available")

    monkeypatch.setattr(pixels, "localize_event", forbidden)
    result = pixels.localized_event(
        "123", {"event_id": "event", "quarter": 1}, quarters, {}, {}
    )
    assert result["status"] == "NULL_TRANSFORM_HAS_NO_MATCHING_PIXELS"
    proposal = {
        "exclusion": {"status": "ELIGIBLE"},
        "gate_path": [
            {"stage": stage, "decision": "PASS"}
            for stage in ("shape", "orbital", "instrument")
        ],
    }
    assert null_proposal_outcome(proposal) == {
        "vetted": True,
        "false_candidate": True,
        "excluded": False,
    }


def test_physical_inputs_keep_the_original_pixel_route(null_run, monkeypatch):
    monkeypatch.setattr(vetting, "null_manifest", lambda root, path: None)
    assert vetting.retrieve_null_pixels() is False
    assert not (null_run[0] / "pixel-retrieval.json").exists()


def test_null_empty_queue_records_zero_parents(null_run):
    folder, _, _ = null_run
    (folder / "cohort.json").write_text("[]\n")
    pixels.retrieve()
    result = json.loads((folder / "pixel-retrieval.json").read_text())
    assert result["parents"] == 0 and result["results"] == []
