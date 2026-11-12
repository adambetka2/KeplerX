"""Checks for the pre-search cadence masks."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolve_detector import COMMON
from masks import search_mask, settling, upper_outliers

RECORD = COMMON.RECORD
DESIGNS = json.loads(Path(__file__).with_name("current-designs.json").read_text())
CONFIG = DESIGNS[0]["config"]


def curve(rng, days=60.0, quarter_days=30.0):
    time = np.arange(0, days, 0.02043365)
    records = np.empty(len(time), dtype=RECORD)
    records["time"] = time
    records["flux"] = 1 + rng.normal(0, 1e-3, len(time))
    records["error"] = 1e-3
    records["quarter"] = np.floor(time / quarter_days).astype(int)
    records["cadence"] = np.arange(len(time))
    return records


def test_masks_flares_and_keeps_deep_transit():
    rng = np.random.default_rng(20261013)
    records = curve(rng)
    flares = np.array([500, 1200, 2100])
    records["flux"][flares] += 8e-3
    transit = np.arange(1700, 1704)
    records["flux"][transit] -= 10e-3
    mask = upper_outliers(records, CONFIG, 4.0)
    assert mask[flares].all()
    assert not mask[transit].any()
    assert mask.sum() <= len(flares) + 2


def test_short_quarter_is_never_masked():
    rng = np.random.default_rng(7)
    records = curve(rng, days=32.0, quarter_days=30.0)
    short = records["quarter"] == 1
    assert 0 < short.sum() < CONFIG["min_quarter_cadences"]
    records["flux"][np.flatnonzero(short)[0]] += 20e-3
    assert not upper_outliers(records, CONFIG, 4.0)[short].any()


def test_requires_local_point_noise():
    noise = CONFIG["noise_model"] | {"method": "duration_block"}
    config = CONFIG | {"noise_model": noise}
    with pytest.raises(ValueError):
        upper_outliers(curve(np.random.default_rng(1)), config, 4.0)


def test_settling_masks_only_after_gaps():
    records = curve(np.random.default_rng(3))
    records = records[(records["time"] < 20) | (records["time"] > 21)]
    resume = records["time"][records["time"] > 21][0]
    mask = settling(records, 0.25, 0.5)
    after_start = records["time"] < 0.5
    after_gap = (records["time"] >= resume) & (records["time"] < resume + 0.5)
    assert np.array_equal(mask, after_start | after_gap)


def test_search_mask_combines_rules():
    rng = np.random.default_rng(11)
    records = curve(rng)
    records["flux"][1500] += 8e-3
    mask = search_mask(records, CONFIG)
    assert mask[1500]
    assert mask[records["time"] < 0.5].all()
    assert not mask[(records["time"] > 1) & (records["time"] < 29)][:10].any()
