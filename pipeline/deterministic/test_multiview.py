"""Independent view completion and review boundaries."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from deterministic import multiview


def view_record(view, survivors, held=0):
    return {
        "discovery_view": view,
        "boi_catalogue_enabled": False,
        "active": len(survivors),
        "held": held,
        "survivors": [
            {"source_id": 1, "signal_id": signal, "discovery_view": view}
            for signal in survivors
        ],
    }


def test_negative_other_view_does_not_remove_flat_survivor():
    report = multiview.review_union(
        [
            view_record("psf_flat", ["positive"]),
            view_record("sap_qdiag", [], held=1),
            view_record("psf_qdiag", ["correlated"]),
        ]
    )
    assert report["sources_with_any_active_view"] == 1
    assert report["view_signal_counts"] == {"psf_flat": 1, "psf_qdiag": 1}
    assert [row["signal_id"] for row in report["survivors"]] == [
        "positive",
        "correlated",
    ]
    assert report["views"][1]["held"] == 1
    assert not report["boi_catalogue_enabled"]
    with pytest.raises(ValueError, match="Repeated discovery view"):
        multiview.review_union(
            [view_record("psf_flat", []), view_record("psf_flat", [])]
        )


def test_incomplete_or_changed_cohort_cannot_enter_review(tmp_path):
    save = multiview.pipeline.save
    ids = tmp_path / "ids.json"
    save(ids, [1, 2])
    preparation = {
        "requested_sources": 2,
        "selection": {
            "requested_source_ids": [1, 2],
            "source_ids": [1],
            "source_ids_file": {
                "path": str(ids),
                "sha256": multiview.pipeline.digest(ids),
            },
        },
        "ineligible": [{"source_id": 2}],
        "preparation_counts": {
            "attempted_sources": 2,
            "accepted_sources": 1,
            "ineligible_sources": 1,
        },
        "detector_contract": {"discovery_view": "psf_flat"},
    }
    save(tmp_path / "preparation.json", preparation)
    save(tmp_path / "result.json", {"status": "EVIDENCE_STAGES_COMPLETE"})
    save(tmp_path / "vetting/results.json", {"processing_status": "PARTIAL"})
    save(
        tmp_path / "vetting/stage-timings.json",
        {"stages": [{"status": "COMPLETE"}] * 16},
    )
    save(tmp_path / "vetting/surviving-signals.json", [])
    with pytest.raises(ValueError, match="incomplete evidence"):
        multiview.completed_view(tmp_path, "psf_flat", ids)
    save(ids, [1, 3])
    with pytest.raises(ValueError, match="selection changed"):
        multiview.completed_view(tmp_path, "psf_flat", ids)


def test_duplicate_signal_cannot_inflate_review():
    with pytest.raises(ValueError, match="Repeated final signal"):
        multiview.review_union([view_record("psf_flat", ["same", "same"])])


@pytest.mark.parametrize("view", multiview.VIEWS)
def test_sparse_followup_uses_full_view_identity(view):
    row = {"source_id": "123", "signal_rank": "7", "photometry": view, "mes": "12"}
    identity = multiview.pipeline.rust_identity(row, sparse=True)
    alerts = [{"detector_signal_id": identity}]
    assert (
        multiview.pipeline.select_followup_alerts(alerts, [], [row], None, None, None)
        == alerts
    )
