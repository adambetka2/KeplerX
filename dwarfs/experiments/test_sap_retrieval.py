"""Protect paired identities, SAP source units, masks and failure denominators."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sap_retrieval as sap


def test_sap_injection_uses_corrected_source_flux_and_keeps_flat_flagged_rows(
    monkeypatch,
):
    curve = {
        "time": np.arange(1000) / 48,
        "flux": np.full(1000, 1000.0),
        "error": np.ones(1000),
        "quality": np.zeros(1000, dtype=int),
        "flatten": np.zeros(1000, dtype=int),
        "cadence": np.arange(1000),
        "quarter": np.ones(1000, dtype=int),
    }
    curve["flatten"][20:23] = 1
    loss = np.zeros(1000)
    loss[20:23] = 0.001
    monkeypatch.setattr(sap.pilot.calibrate, "loss", lambda time, truth: loss.copy())
    source, _ = sap.source_curve(curve, {1: 0.25})
    injected, arrays = sap.inject(source, {})
    records = sap.pilot.prepare_curve_records(injected)
    np.testing.assert_allclose(source["flux"], 250)
    np.testing.assert_allclose(injected["flux"][20:23], 249.75)
    np.testing.assert_array_equal(arrays["flux_delta"][20:23], -0.25)
    np.testing.assert_array_equal(injected["error"], curve["error"])
    assert {20, 21, 22}.issubset(records["cadence"])
    np.testing.assert_allclose(records["flux"][20:23], -0.001)


def test_overlap_keeps_unavailable_sap_in_the_frozen_denominator():
    rows = [
        {"case_id": str(i), "positive": "true", "recovered": recovered}
        for i, recovered in enumerate((True, True, False, False))
    ]
    flat = [
        row | {"recovered": recovered}
        for row, recovered in zip(rows, (True, False, True, False), strict=True)
    ]
    rows[2].update(preparation_status="INPUT_INELIGIBLE", first_failure="missing SAP")
    result = sap.overlap(rows, flat)
    assert result["counts"] == {"both": 1, "sap_only": 1, "flat_only": 1, "neither": 1}
    assert result["case_ids"]["flat_only"] == ["2"]
    with pytest.raises(ValueError, match="Exact paired IDs"):
        sap.overlap(rows, flat[::-1])


def test_sap_preparation_identity_mismatch_is_not_an_eligibility_miss(
    tmp_path, monkeypatch
):
    (tmp_path / "prepared").mkdir()
    curve = {
        "time": np.arange(1000) / 48,
        "flux": np.full(1000, 1000.0),
        "error": np.ones(1000),
        "quality": np.zeros(1000, dtype=int),
        "flatten": np.zeros(1000, dtype=int),
        "cadence": np.arange(1000),
        "quarter": np.ones(1000, dtype=int),
    }
    monkeypatch.setattr(
        sap,
        "checked_sap",
        lambda item, folder: (
            curve,
            {1: 0.25},
            {"unmodified_preparation": {"records_sha256": "incorrect"}},
        ),
    )
    item = {
        "case": {
            "case_id": "same-frozen-case",
            "receipt_sha256": "original-receipt",
            "kind": "original",
        },
        "phot_g_mean_mag": 17.5,
    }
    with pytest.raises(RuntimeError, match="preparation differs"):
        sap.prepare_case(item, tmp_path)
    assert not list((tmp_path / "prepared").iterdir())
