"""Regression checks for saved calibration replay consistency."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import calibrate
import numpy as np
import pytest


class CountingPool(ThreadPoolExecutor):
    def __init__(self, workers):
        super().__init__(max_workers=workers)
        self.submitted = []
        self.peak_pending = 0

    def submit(self, function, *args):
        future = super().submit(function, *args)
        self.submitted.append(future)
        self.peak_pending = max(
            self.peak_pending, sum(not item.done() for item in self.submitted)
        )
        return future


def test_completed_tasks_bounds_pending_work():
    def work(value):
        time.sleep(0.01)
        return value

    with CountingPool(2) as pool:
        results = list(calibrate.completed_tasks(pool, work, range(21), 2))
    assert sorted(results) == list(range(21))
    assert pool.peak_pending <= 2


def test_completed_tasks_does_not_submit_after_failure():
    def work(value):
        if value == 0:
            raise ValueError("First task failed")
        time.sleep(0.01)
        return value

    with CountingPool(2) as pool, pytest.raises(ValueError, match="First task failed"):
        list(calibrate.completed_tasks(pool, work, range(21), 2))
    assert len(pool.submitted) == 2
    assert pool.peak_pending <= 2


@pytest.fixture
def saved_result():
    truth = {
        "period_days": 3.0,
        "epoch_bkjd": 1.0,
        "start_bkjd": 0.0,
        "stop_bkjd": 30.0,
        "duration_hours": 2.0,
        "signal": {"exposure_days": 0.02043361111111111},
    }
    proposal = {"period_days": 3.0, "t0_bkjd": 1.0}
    proposal["truth_match"] = calibrate.clock_match(proposal, truth)
    trial = {"trial_index": "1", "source_id": "regression"}
    side = {
        "status": "SEARCHED",
        "reason": None,
        "proposals": [proposal],
        "diagnostics": {},
    }
    source = {"status": "PREPARED", "trial": trial, "truth": truth}
    row = {
        "trial": trial,
        "truth": truth,
        "injected": side,
        "recovered_exact": True,
        "null_control": copy.deepcopy(side),
        "preexisting_exact_clock": True,
    }
    return row, source


@pytest.mark.parametrize("field", ["recovered_exact", "preexisting_exact_clock"])
@pytest.mark.parametrize("value", [0, 1])
def test_replay_requires_boolean_flags(saved_result, tmp_path, field, value):
    row, source = saved_result
    calibrate.validate_result(row, source, 1, tmp_path, False)
    row[field] = value
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 1, tmp_path, False)


def test_replay_recounts_proposal_clock(saved_result, tmp_path):
    row, source = saved_result
    proposal = row["injected"]["proposals"][0]
    assert type(calibrate.clock_match(proposal, source["truth"])["exact"]) is bool
    proposal["truth_match"]["exact"] = 1
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 1, tmp_path, False)
    proposal["truth_match"]["exact"] = True
    proposal["t0_bkjd"] += 1.0
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 1, tmp_path, False)


def test_historical_empty_and_ineligible_replay(saved_result, tmp_path):
    row, source = saved_result
    row["injected"]["proposals"] = []
    row.update(recovered_exact=False, null_control=None, preexisting_exact_clock=None)
    calibrate.validate_result(row, source, 0, tmp_path, False)
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 0, tmp_path, True)
    row["injected"].update(status="DETECTOR_INELIGIBLE", reason="FEWER_THAN_100")
    calibrate.validate_result(row, source, 0, tmp_path, True)
    row["injected"]["reason"] = None
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 0, tmp_path, False)


def test_searched_band_requires_all_preserved_spectra(saved_result, tmp_path):
    row, source = saved_result
    row.update(null_control=None, preexisting_exact_clock=None)
    fields = ("period_days", "search_snr", "depth_ppm", "duration_days", "epoch_bkjd")
    band = {
        "band": "core",
        "status": "SEARCHED",
        "periodogram_arrays": {key: np.ones(2) for key in fields},
    }
    row["injected"]["diagnostics"] = {"bands": [band]}
    calibrate.preserve_periodograms(row, tmp_path / "regression.json")
    calibrate.validate_result(row, source, 0, tmp_path, True)
    row["periodograms"]["arrays"].pop("injected_core_period_days")
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 0, tmp_path, True)
    path = tmp_path / row["periodograms"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        retained = {key: archive[key] for key in row["periodograms"]["arrays"]}
    np.savez_compressed(path, **retained)
    row["periodograms"].update(sha256=calibrate.digest(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError):
        calibrate.validate_result(row, source, 0, tmp_path, True)
