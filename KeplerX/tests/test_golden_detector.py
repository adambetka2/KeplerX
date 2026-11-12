"""Golden: whole Detector searches of real photometry."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import inject, load_lane_host, packet, planet, tres2_packet

from common import load_packet
from detector import gpu_screening
from detector.program import run as detect
from detector.upstream_adapter import make_dump


def search(prepared, out, config, reference=None):
    out.mkdir()
    args = SimpleNamespace(prepared=prepared, out=out)
    try:
        return detect(args, config, reference)["candidates"]
    except Exception as error:  # a failure must be the same failure
        return f"{type(error).__name__}: {error}"


def bits(candidates):
    return json.dumps(candidates, sort_keys=True)


def lane_with_planets(tmp_path, quarters=None):
    """The quiet K dwarf with a strong planet."""
    lane = load_lane_host(quarters)
    bodies = [planet(lane, 4.8, 1.5, 1.3), planet(lane, 9.7, 1.0, 2.1)]
    arrays, metadata, _ = inject(lane, bodies)
    return packet(tmp_path / "prepared", arrays, metadata)


def degenerate(tmp_path):
    """One quarter of the K dwarf cut into islands of 40 cadences every two days."""
    lane = load_lane_host((2,))
    arrays = {name: value.copy() for name, value in lane.prepared.items()}
    phase = (arrays["time"] - arrays["time"][0]) % 2.0
    arrays["mask"] &= phase < 40 * lane.metadata["exposure_days"]
    return packet(tmp_path / "prepared", arrays, lane.metadata)


CURVES = {
    "tres2": lambda tmp_path, config: tres2_packet(tmp_path / "prepared", config)[0],
    "k_dwarf_two_quarters_with_planets": lambda tmp_path, config: lane_with_planets(
        tmp_path, (2, 3)
    ),
    "k_dwarf_degenerate_segments": lambda tmp_path, config: degenerate(tmp_path),
}


# The whitening sums in another order (about 1e-15): agree to this share of the template's largest.
FFT_TOLERANCE = 1e-12


@pytest.mark.parametrize("curve", list(CURVES))
def test_compiled_search_is_native_bit_for_bit(curve, tmp_path, config, monkeypatch):
    """The references share the Detector's template bank, period grid, segments, templates and repeat rule."""
    monkeypatch.setenv("KEPLERX_GPU", "1")
    prepared = CURVES[curve](tmp_path, config)
    expected = search(prepared, tmp_path / "native", config, "native")
    got = search(prepared, tmp_path / "compiled", config, "compiled")
    assert bits(got) == bits(expected)
    if curve != "k_dwarf_degenerate_segments":
        assert expected and not isinstance(expected, str)


@pytest.mark.skipif(
    not gpu_screening.available(), reason="No Metal GPU on this machine"
)
def test_gpu_screened_search_is_the_cpu_search(tmp_path, config, monkeypatch):
    """Production settings on four quarters (long periods fold in windows)."""
    prepared = lane_with_planets(tmp_path)
    monkeypatch.setenv("KEPLERX_GPU", "0")
    cpu = search(prepared, tmp_path / "cpu", config)
    monkeypatch.setenv("KEPLERX_GPU", "1")
    gpu = search(prepared, tmp_path / "gpu", config)
    assert bits(gpu) == bits(cpu)
    assert any(np.isclose(c["period"], 4.8, rtol=1e-3) for c in cpu)


@pytest.mark.parametrize("curve", ["tres2", "k_dwarf_two_quarters_with_planets"])
def test_fft_whitening_is_upstreams_to_rounding(curve, tmp_path, config):
    """Every template's ses, numerator and denominator over a whole curve."""
    arrays, metadata = load_packet(CURVES[curve](tmp_path, config))
    detector, upstream = (
        make_dump(
            arrays, metadata, config["detection"], reference=reference
        ).Calculate_SES_by_Segment()
        for reference in (None, "compiled")
    )
    for got, expected in zip(detector, upstream, strict=True):
        scale = np.abs(expected).max(axis=1, keepdims=True)
        assert np.all(np.abs(got - expected) <= FFT_TOLERANCE * scale)
