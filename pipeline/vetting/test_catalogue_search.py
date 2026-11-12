"""Known-source admission preserves timing coverage and first-failure evidence."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from catalogue_search import (
    REQUIRED,
    RULE,
    Snapshot,
    admission,
    compare,
    digest,
    measure_row,
    search,
)
from vetting_disposition import GATE_ORDER, check_gate_record, triage_disposition
from vetting_triage import apply_gate


@pytest.fixture
def inputs():
    source = {"source_id": "123", "ra_deg": 290.0, "dec_deg": 40.0, "kic": "999"}
    known = {
        "catalogue": "NASA_confirmed_planets_PS",
        "name": "Gold b",
        "row_index": 0,
        "gaia_dr3": "123",
        "kic": "888",
        "ra": 290.0,
        "dec": 40.0,
        "period": 4.0,
        "epoch": 0.0,
        "duration_hours": 2.0,
        "classification": "CONFIRMED",
        "time_reference": "BJD-TDB",
        "position_source": "Target catalogue",
    }
    hypothesis = {
        "hypothesis_id": "original",
        "timing_basis": "ORIGINAL_PROPOSAL",
        "period_days": 4.0,
        "epoch_bkjd": 0.0,
        "duration_days": 2 / 24,
        "unit_ids": ["u0", "u1", "u2"],
    }
    events = [
        {"unit_id": f"u{i}", "centre_bkjd": time}
        for i, time in enumerate((0.0, 4.0, 8.0))
    ]
    signal = {
        "signal_id": "signal",
        "event_ids": ["e0", "e1", "e2"],
        "events": events,
        "period_days": 4.0,
        "timing_hypotheses": [hypothesis],
    }
    return source, known, signal


def catalogue(rows):
    return SimpleNamespace(
        rows=rows,
        candidates=lambda source: list(range(len(rows))),
        manifest_sha256="frozen",
        manifest={"counts": {}, "position_qualification": "Recorded"},
    )


@pytest.mark.parametrize(
    "period,centres,expected",
    [
        (4.0, [0.0, 4.0, 8.0], True),
        (3.998, [0.0, 3.998, 7.996], True),
        (4.0, [0.0, 0.001, 8.0], False),
        (4.0, [0.0, 4.0, 8.5], False),
    ],
)
def test_three_independent_clock_cycles(inputs, period, centres, expected):
    source, known, signal = inputs
    hypothesis = {**signal["timing_hypotheses"][0], "period_days": period}
    result = compare(source, hypothesis, np.array(centres), known)
    assert result["flag"] is expected
    assert result["best_phase_class"]["independent_common_cycles"] == len(
        {round(t / period) for t in centres}
    )


def test_aperture_kic_and_centroid_proxy_do_not_establish_ownership(inputs):
    source, known, signal = inputs
    hypothesis = signal["timing_hypotheses"][0]
    centres = np.array([0.0, 4.0, 8.0])
    proxy = {
        **known,
        "gaia_dr3": None,
        "kic": source["kic"],
        "position_source": "TCE flux-weighted centroid proxy",
    }
    result = compare(source, hypothesis, centres, proxy)
    assert result["flag"]
    assert result["host_relation"] == "NEARBY_CATALOGUED_SOURCE"
    assert not result["exact_gaia_dr3_identity"]
    assert (
        admission({"catalogue_search": {"status": "COMPLETE", "matches": [result]}})[0]
        == "NEARBY_SOURCE_UNCONFIRMED"
    )


def test_missing_literature_epoch_preserves_row_with_qualified_object_coverage(inputs):
    source, known, signal = inputs
    missing = {**known, "row_index": 1, "epoch": None}
    report = search(source, signal, catalogue([known, missing]))
    assert report["status"] == "COMPLETE"
    unavailable = report["covered_unavailable_literature_pairs"][0]
    assert unavailable["epoch"] is None
    assert unavailable["object_coverage_closed_by"][0]["row_index"] == 0
    assert report["reviewed_period_compatible_pairs"][1]["epoch"] is None


def test_periodic_missing_hypothesis_is_incomplete_but_unknown_period_remains_eligible(
    inputs,
):
    source, known, signal = inputs
    signal["timing_hypotheses"] = []
    assert (
        search(source, signal, catalogue([known]))["status"]
        == "EPHEMERIS_COVERAGE_INCOMPLETE"
    )
    signal["period_days"] = None
    assert search(source, signal, catalogue([known]))["status"] == "COMPLETE"


@pytest.mark.parametrize("first_failure", [None, *GATE_ORDER[:-1]])
def test_known_recovery_retains_first_failure_without_false_positive_denial(
    inputs, first_failure
):
    source, known, signal = inputs
    signal["catalogue_search"] = search(source, signal, catalogue([known]))
    history = [
        {
            "stage": stage,
            "decision": "CONTINUE_INCONCLUSIVE" if stage == first_failure else "PASS",
            "reason": "EARLIER_EVIDENCE_INCOMPLETE"
            if stage == first_failure
            else "ORIGINAL_PASS",
            "evidence": {},
        }
        for stage in GATE_ORDER[:-1]
    ]
    signal["triage"] = {
        "status": "HELD" if first_failure else "ACTIVE",
        "history": history,
        "record_every_gate": True,
    }
    if first_failure:
        signal["triage"].update(
            held_stage=first_failure, reason="EARLIER_EVIDENCE_INCOMPLETE"
        )
    row = {"signals": [signal]}
    apply_gate(
        row,
        "contamination",
        {"triage": {"enabled": True}, "triage_execution": "RECORD_EVERY_GATE"},
    )
    check_gate_record(signal)
    label, reasons = triage_disposition(signal)
    assert label == "HELD"
    assert signal["triage"]["held_stage"] == (first_failure or "contamination")
    assert reasons == [
        "EARLIER_EVIDENCE_INCOMPLETE" if first_failure else "KNOWN_SIGNAL_ASSOCIATION"
    ]
    assert signal["catalogue_state"] == "RECOVERED_CATALOGUE_ASSOCIATION"


def test_snapshot_byte_changes_and_actual_row_schema(inputs, tmp_path):
    source, known, signal = inputs
    rows = [{**known, "catalogue": name} for name in sorted(REQUIRED)]
    data = tmp_path / "signals.json"
    data.write_text(json.dumps(rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "required_catalogues": sorted(REQUIRED),
                "counts": {"normalized_signal_rows": len(rows)},
                "signals": {"path": data.name, "sha256": digest(data)},
                "position_qualification": "Recorded",
            }
        )
    )
    config = {
        "catalogue_search": {
            "rule": RULE["id"],
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
        }
    }
    row = {
        "source_id": source["source_id"],
        "coordinates": {k: v for k, v in source.items() if k != "source_id"},
        "signals": [copy.deepcopy(signal)],
    }
    measure_row(row, source, config, tmp_path)
    assert row["signals"][0]["catalogue_search"]["status"] == "COMPLETE"
    assert admission(row["signals"][0])[0] == "RECOVERED_CATALOGUE_ASSOCIATION"
    data.write_text("[]")
    with pytest.raises(ValueError, match="rows hash changed"):
        Snapshot(manifest, digest(manifest))
