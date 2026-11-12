"""Replay frozen M/K records through the disk-independent local preparation."""

import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_local as local


@pytest.mark.parametrize("population", ["M", "K"])
def test_spawned_preparation_matches_frozen_records_without_workspace_disk(
    tmp_path, population
):
    rows = local.API.rows(local.real.ROOT / "results/psf-flat-real5000-20261014.csv")
    row = next(
        (
            row
            for row in rows
            if row["population"] == population
            and local.real.availability([row], local.real.ROOT)["available"]
        ),
        None,
    )
    if row is None:
        pytest.skip("Frozen population corpus is not mounted")
    (tmp_path / "prepared").mkdir()
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        case = pool.submit(local.prepare_one, (local.real.ROOT, tmp_path, row)).result()
    receipt = local.API.read(case["receipt_path"])
    assert receipt["frozen_selection"] == row
    assert local.API.digest(case["input_path"]) == case["input_sha256"]
    with np.load(case["input_path"], allow_pickle=False) as stored:
        records = stored["records"]
        assert len(records) == int(row["prepared_cadences"])
        assert hashlib.sha256(records.tobytes()).hexdigest() == row["records_sha256"]
        assert np.all(stored["raw_flatten"][stored["retained_mask"]] == 0)
        assert set(stored.files) == {
            "records",
            "retained_mask",
            *["raw_" + field for field in local.pilot.FIELDS],
        }
    with pytest.raises(ValueError, match="Frozen Flat records differ"):
        local.native_records(local.real.ROOT, row | {"records_sha256": "0" * 64})
