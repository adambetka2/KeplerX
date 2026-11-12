"""Keep the conditional survey tied to complete scores and frozen real Flat records."""

import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_real as real


def scored_report(sap):
    return {
        "status": "SCORED_SAP_VERSUS_SAVED_FLAT",
        "summaries": {
            "sap": {
                "planets": 700,
                "controls": 300,
                "recovered": sap,
                "statuses": {"UNEXPECTED_ERROR": 3},
            },
            "flat_reused": {
                "planets": 700,
                "controls": 300,
                "recovered": 163,
                "statuses": {"UNEXPECTED_ERROR": 2},
            },
        },
    }


@pytest.mark.parametrize(
    ("sap", "expected"),
    [(141, "QUALIFIES"), (142, "CRITERION_NOT_MET"), (163, "CRITERION_NOT_MET")],
)
def test_fifteen_percent_is_relative_and_uses_unrounded_counts(sap, expected):
    result = real.decision({"stage": "report", "state": "COMPLETE"}, scored_report(sap))
    assert result["status"] == expected
    assert result["flat_errors"] == 2 and result["sap_errors"] == 3


def test_interim_recovery_never_triggers_the_real_survey():
    assert real.decision({"stage": "run", "state": "RUNNING"}, scored_report(1)) == {
        "status": "WAITING_COMPARISON"
    }
    report = scored_report(141)
    report["summaries"]["sap"]["planets"] = 699
    with pytest.raises(ValueError, match="700-signal"):
        real.decision({"stage": "report", "state": "COMPLETE"}, report)


@pytest.mark.parametrize("population", ["M", "K"])
def test_spawned_preparation_reproduces_frozen_server_flat_records(
    tmp_path, population
):
    selection = (
        real.ROOT
        / "dump/psf-flat-retrieval-20261014-v1/real5000-usable-v1/real5000.csv"
    )
    if not selection.is_file():
        pytest.skip("Frozen external real-source selection is unavailable")
    rows = real.API.rows(selection)
    row = next(
        (
            row
            for row in rows
            if row["population"] == population
            and (real.shard_folder(real.ROOT, row) / "cadence.parquet").is_file()
        ),
        None,
    )
    if row is None:
        pytest.skip("Frozen population inputs are not mounted")
    (tmp_path / "prepared").mkdir()
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        case = pool.submit(real.prepare_one, (real.ROOT, tmp_path, row)).result()
    with np.load(case["input_path"], allow_pickle=False) as data:
        records = data["records"]
        assert hashlib.sha256(records.tobytes()).hexdigest() == row["records_sha256"]
        assert len(records) == int(row["prepared_cadences"])
        assert np.all(data["raw_flatten"][data["retained_mask"]] == 0)
    with pytest.raises(ValueError, match="Frozen local Flat input differs"):
        real.native_records(real.ROOT, row | {"records_sha256": "0" * 64})
