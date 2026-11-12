"""End-to-end: production Detector on two injected planets."""

from types import SimpleNamespace

from conftest import inject, load_lane_host, packet, planet

from detector.program import run as detect


def test_two_planets_have_refined_ephemerides(tmp_path, config):
    lane = load_lane_host()
    bodies = [planet(lane, 4.8, 1.5, 1.3), planet(lane, 7.2, 1.3, 2.9)]
    arrays, metadata, signals = inject(lane, bodies)
    prepared = packet(tmp_path / "prepared", arrays, metadata)
    out = tmp_path / "detections"
    out.mkdir()
    candidates = detect(SimpleNamespace(prepared=prepared, out=out), config)[
        "candidates"
    ]
    half_cadence = metadata["exposure_days"] / 2
    for signal in signals:
        assert signal["detectable"]
        period, epoch = signal["period_days"], signal["epoch_bkjd"]
        found = [c for c in candidates if abs(c["period"] / period - 1) < 0.01]
        assert len(found) == 1, (period, candidates)
        candidate = found[0]
        assert abs(candidate["period"] / period - 1) < 2e-5
        offset = (candidate["t0"] - epoch + period / 2) % period - period / 2
        assert abs(offset) < half_cadence, (period, offset)
        assert candidate["mes"] > config["detection"]["mes_threshold"]
