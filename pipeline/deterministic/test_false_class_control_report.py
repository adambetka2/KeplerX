import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.deterministic import release_report

from pipeline.deterministic import false_class_control_report as report
from pipeline.runtime import disposition_execution


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def background_run(tmp_path):
    controls, run = tmp_path / "controls", tmp_path / "run"
    controls.mkdir()
    truth = [
        {
            "source_id": str(index),
            "parent_tpf": str(100 + index),
            "category": category,
            "partition": "development",
            "injected_quarters": [0],
            "artifact_kind": "test",
            "period_days": 10.0,
            "epoch_bkjd": 100.0,
            "duration_days": 0.1,
            "expected_search_mes": 100.0,
        }
        for index, category in enumerate(
            ["instrumental_artifact", "eclipsing_binary", "background_contaminant"], 1
        )
    ]
    (controls / "truth.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in truth)
    )
    save(controls / "source-ids.json", ["1", "2", "3"])
    save(controls / "development-source-ids.json", ["1", "2", "3"])
    arrays = controls / "3.npz"
    np.savez(arrays, quarter=np.array([0]), qdiag_quarter=np.array([0]))
    pixel = controls / "pixel.bin"
    pixel.write_bytes(b"original-pixel-bytes")
    manifest = {
        "schema": 1,
        "protocol": report.false_class_injections.MODEL,
        "measurement_scope": "Four observed-quarter excerpts",
        "sources": {
            str(index): {
                "arrays": f"{index}.npz",
                "sha256": report.catalogue.digest(arrays),
                "pixels": {
                    "0": {
                        "base_parent_kic": str(100 + index),
                        "path": "pixel.bin",
                        "sha256": report.catalogue.digest(pixel),
                    }
                },
            }
            for index in (1, 2, 3)
        },
    }
    save(controls / "input-manifest.json", manifest)
    save(
        controls / "materialization.json",
        {
            "protocol": manifest["protocol"],
            "status": "PROBE",
            "artifacts": {
                name: report.catalogue.digest(controls / name)
                for name in ("truth.jsonl", "source-ids.json", "input-manifest.json")
            },
        },
    )
    policy = json.loads(
        Path(report.ROOT / "pipeline/vetting/disposition-policy.json").read_text()
    )
    policy.update(status="DEVELOPMENT", boi_catalogue_enabled=False)
    save(run / "policy.json", policy)
    save(run / "vetting.json", {"disposition_policy": str(run / "policy.json")})
    save(run / "preparation.json", {"sources": [{"source_id": "3"}]})
    save(
        run / "detection.json",
        {
            "status": "DETECTION_COMPLETE",
            "sources": 1,
            "alerted_sources": 0,
            "alerts": 0,
        },
    )
    save(run / "queue.json", {"rows": []})
    save(run / "discovery-manifest.json", {})
    save(run / "vetting/manifest.json", {})
    save(run / "vetting/stage-timings.json", {})
    save(
        run / "vetting/results.json",
        {
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "sources": 0,
            "original_alerts_preserved": 0,
            "disposition_execution": disposition_execution(policy),
        },
    )
    save(
        run / "proposals/3.json",
        {"source_id": "3", "alerts": [], "rust_rows": [], "supplied_hypotheses": []},
    )
    return controls, run


def test_background_report_retains_the_materialized_denominator(background_run):
    controls, run = background_run
    original = hashlib.sha256((controls / "truth.jsonl").read_bytes()).hexdigest()
    result = report.build_report(controls, run, "development", background_only=True)
    assert result["population"] == {
        "declared_sources": 1,
        "materialized_partition_sources": 3,
        "unmeasured_materialized_sources": 2,
        "prepared_sources": 1,
        "queued_sources": 0,
        "vetting_sources": 0,
    }
    assert [row["source_id"] for row in result["trials"]] == ["3"]
    assert result["categories"]["background_contaminant"]["false_boi_rate"] == {
        "status": "NOT_MEASURED_BOI_CATALOGUE_DISABLED"
    }
    assert (
        hashlib.sha256((controls / "truth.jsonl").read_bytes()).hexdigest() == original
    )


def test_unselected_classes_are_not_silently_removed_without_the_option(background_run):
    controls, run = background_run
    with pytest.raises(FileNotFoundError):
        report.build_report(controls, run, "development")


def test_background_report_still_refuses_a_missing_background_source(background_run):
    controls, run = background_run
    save(run / "preparation.json", {"sources": []})
    with pytest.raises(ValueError, match="replay is incomplete"):
        report.build_report(controls, run, "development", background_only=True)


def test_background_report_still_checks_payload_hashes(background_run):
    controls, run = background_run
    (controls / "3.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="arrays changed"):
        report.build_report(controls, run, "development", background_only=True)


@pytest.mark.parametrize(
    "status",
    ["PARTIAL_OPENED_CONTROL_REPLAY_REPORT", "PARTIAL_OPENED_CATALOGUE_CONTROL_REPORT"],
)
def test_partial_opened_reports_cannot_enter_release(status):
    with pytest.raises(ValueError, match="Unsupported release control report"):
        release_report.report_records(
            {"status": status, "outcome_contract": release_report.OUTCOME_CONTRACT}
        )
