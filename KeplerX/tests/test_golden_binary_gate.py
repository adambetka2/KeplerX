"""Golden: the binary gate."""

from conftest import FIXTURES, golden

from common import read
from vetter.policy import assess


def test_binary_gate_verdicts_are_golden():
    signals = read(FIXTURES / "binary-gate-signals.json")["signals"]
    verdicts = {
        name: assess(signal["metrics"], "linear-score-binary-gate")
        for name, signal in signals.items()
    }
    assert verdicts["kepler-840-b"]["state"] == "DISMISSED"
    assert verdicts["kepler-158-c"]["state"] == "KEPT"
    assert "binary_evidence" not in verdicts["kepler-158-c"]
    assert verdicts["weak-synthetic-planet"]["state"] == "KEPT"
    assert verdicts["blended-binary"]["state"] == "DISMISSED"
    assert verdicts["alias-giant-planet"]["state"] == "KEPT"
    assert [
        p["feature"] for p in verdicts["alias-giant-planet"]["binary_evidence"]
    ] == ["depth_ppt"]
    assert verdicts["deep-blended-binary"]["state"] == "DISMISSED"
    assert {p["feature"] for p in verdicts["deep-blended-binary"]["failed"]} >= {
        "binary_occultation_albedo",
        "depth_ppt",
    }
    golden("binary-gate", verdicts)
