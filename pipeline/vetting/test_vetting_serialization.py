"""Evidence normalization preserves scalar types, missingness and binary64 values."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]
from vetting_flux import finite


@pytest.mark.parametrize(
    "dtype", [float, np.float16, np.float32, np.float64, np.longdouble]
)
@pytest.mark.parametrize(
    "value", [0.0, -0.0, 1.25, np.nextafter(0.0, 1.0), np.finfo(float).max]
)
def test_finite_scalar_conversion_preserves_binary64_bits(dtype, value):
    with np.errstate(over="ignore", under="ignore"):
        original = dtype(value)
    result = finite(original)
    if np.isfinite(original):
        assert type(result) is float
        assert struct.pack(">d", result) == struct.pack(">d", float(original))
    else:
        assert result is None


@pytest.mark.parametrize(
    "dtype", [float, np.float16, np.float32, np.float64, np.longdouble]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_evidence_remains_missing(dtype, value):
    assert finite(dtype(value)) is None


def test_nested_evidence_preserves_types_and_negative_zero():
    original = {
        "count": np.int64(3),
        "flag": np.bool_(True),
        "samples": (np.float32(1.25), -0.0, float("nan")),
    }
    result = finite(original)
    assert type(result["count"]) is int and type(result["flag"]) is bool
    assert type(result["samples"]) is list
    assert result["samples"][0] == 1.25 and result["samples"][2] is None
    assert struct.pack(">d", result["samples"][1]) == struct.pack(">d", -0.0)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert type(original["samples"]) is tuple
