"""End-to-end: the Detector finds a planet with 7.5-hour transits on a 70-day orbit."""

from types import SimpleNamespace

import numpy as np
from conftest import inject, load_lane_host, packet, planet

from calibration.scoring import match_signal
from detector.program import run as detect

PERIOD = 70.0


def test_flattened_templates_find_a_long_transit(tmp_path, config):
    lane = load_lane_host()
    arrays, metadata, signals = inject(lane, [planet(lane, PERIOD, 2.0, 15.0)])
    (signal,) = signals
    assert signal["detectable"] and signal["duration_days"] > 0.3
    time = arrays["time"][arrays["mask"]]
    start, baseline = float(time.min()), float(np.ptp(time))
    prepared = packet(tmp_path / "prepared", arrays, metadata)
    out = tmp_path / "detections"
    out.mkdir()
    found = detect(SimpleNamespace(prepared=prepared, out=out), config)["candidates"]
    outcomes = [match_signal(c, signal, start, baseline)[0] for c in found]
    assert "exact" in outcomes, found
