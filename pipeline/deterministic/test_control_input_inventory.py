import csv
import json

import numpy as np
import pytest

from pipeline.deterministic import control_input_inventory as inventory


def curve():
    return {
        "time": np.arange(5, dtype=float), "cadence": np.arange(5),
        "quarter": np.array([1, 1, 2, 3, 4]),
        "flux": np.array([1.0, 1.0, 1.0, np.nan, 1.0]),
        "error": np.array([0.1, 0.1, 0.1, 0.1, 0.0]),
        "quality": np.array([0, 8, 0, 0, 0]),
        "flatten": np.array([1, 0, 0, 0, 0]),
    }


def test_availability_keeps_raw_and_discovery_masks_separate():
    result, quarters = inventory.view_inventory(curve())

    assert quarters == {1, 2}
    assert result["cadences_by_quarter"]["1"] == {
        "original_rows": 2, "quality_zero_usable_cadences": 1,
        "quality_zero_flatten_zero_usable_cadences": 0,
    }
    assert result["cadences_by_quarter"]["3"][
        "quality_zero_usable_cadences"
    ] == 0
    assert result["cadences_by_quarter"]["4"][
        "quality_zero_usable_cadences"
    ] == 0
    changed = curve()
    changed["flatten"][0] = 0
    fingerprint, _ = inventory.view_inventory(changed)
    assert fingerprint["array_sha256"]["flatten"] != (
        result["array_sha256"]["flatten"]
    )


def test_source_quarters_include_usable_correlated_views(monkeypatch, tmp_path):
    row = {
        "source_id": "12", "cohort": "ordinary", "role": "evaluation",
        "source_domain": "KBonus_LC2", "origin_parent": "9",
        "raw_fits_sha256": "raw",
    }
    metadata = {"source_id": "12", "origin_tpf_kic": 9,
                "raw_fits_sha256": "raw"}
    monkeypatch.setattr(inventory.lc2, "load_source",
                        lambda *args: (metadata, {}))

    def view(raw, name):
        result = curve()
        if name == "sap_qdiag":
            result["flux"][3] = 1.0
        return result

    monkeypatch.setattr(inventory.lc2, "photometry", view)
    result = inventory.source_inventory(tmp_path, row, None)

    assert result["required_pixel_quarters"] == [1, 2, 3]
    assert result["pixel_coverage_status"] == "NOT_YET_VERIFIED"
    assert result["scientific_outcomes_opened"] is False
    metadata["origin_tpf_kic"] = 10
    with pytest.raises(ValueError, match="source/parent/raw/domain"):
        inventory.source_inventory(tmp_path, row, None)


def test_declared_graph_rejects_exposed_evaluation_sources(tmp_path):
    ledger = tmp_path / "trials.csv"
    rows = [{"source_id": "12", "cohort": "ordinary", "role": "evaluation",
             "historically_exposed_component": "True"}]
    with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"trials": 1, "trial_ledger_sha256": inventory.digest(ledger)}
    (tmp_path / "audit.json").write_text(json.dumps(audit))

    with pytest.raises(ValueError, match="historically exposed"):
        inventory.declared_sources(tmp_path)
    ledger.write_text(ledger.read_text().replace("True", "False"))
    with pytest.raises(ValueError, match="ledger changed"):
        inventory.declared_sources(tmp_path)
