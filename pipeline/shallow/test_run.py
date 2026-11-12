"""Checks for the experimental shallow proposal search."""

import csv
import ctypes
import json
import struct
import subprocess
import sys
from pathlib import Path

import astropy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (
    CONFIG,
    RECORD,
    ShallowIneligible,
    box_search,
    distinct_proposals,
    main,
    scan,
)


def test_recovers_shallow_repeated_transit():
    config = json.loads(CONFIG.read_text())
    period = 3.271
    epoch = 0.37
    duration = 0.12
    time = np.arange(0, 100, 0.02043365)
    rng = np.random.default_rng(20260924)
    flux = rng.normal(0, 45e-6, len(time))
    cycle = np.rint((time - epoch) / period)
    transit = abs(time - epoch - cycle * period) < duration / 2
    flux[transit] -= 150e-6
    records = np.empty(len(time), dtype=RECORD)
    records["time"] = time
    records["flux"] = flux
    records["error"] = 45e-6
    records["quarter"] = np.floor(time / 25).astype(int)
    records["cadence"] = np.arange(len(time))

    proposals = scan(records, config)
    matched = [row for row in proposals if abs(row["period_days"] / period - 1) < 0.001]
    assert matched
    assert matched[0]["depth_ppm"] < 500
    assert matched[0]["observed_events"] >= 3
    assert matched[0]["observed_quarters"] >= 2
    assert [row for row in proposals if row["search_band"] == "baseline"] == scan(
        records,
        {
            key: value for key, value in config.items()
            if key not in {"short_period_search", "long_period_search"}
        },
    )


def test_recovers_short_repeated_transit():
    config = json.loads(CONFIG.read_text())
    period = 0.317
    epoch = 0.13
    duration = 0.04
    time = np.arange(0, 100, 0.02043365)
    rng = np.random.default_rng(20260925)
    flux = rng.normal(0, 45e-6, len(time))
    cycle = np.rint((time - epoch) / period)
    flux[abs(time - epoch - cycle * period) < duration / 2] -= 150e-6
    records = np.empty(len(time), dtype=RECORD)
    records["time"] = time
    records["flux"] = flux
    records["error"] = 45e-6
    records["quarter"] = np.floor(time / 25).astype(int)
    records["cadence"] = np.arange(len(time))

    proposals = scan(records, config)
    matched = [
        row for row in proposals if abs(row["period_days"] / period - 1) < 0.001
    ]
    assert matched
    assert matched[0]["search_band"] == "short"
    assert matched[0]["positive_cycle_fraction"] >= 0.5


def test_box_search_matches_original_astropy_fma_bit_for_bit(tmp_path):
    # Fix the four fused operations of the declared arm64 arithmetic contract.
    path = Path(astropy.__file__).parent / "timeseries/periodograms/bls/bls.c"
    text = path.read_text()
    substitutions = {
        "return x - period * floor(x / period);": "return fma(-period, floor(x / period), x);",
        "sum_y += y[i] * ivar[i];": "sum_y = fma(y[i], ivar[i], sum_y);",
        "mean_y[ind] += y[n] * ivar[n];": "mean_y[ind] = fma(y[n], ivar[n], mean_y[ind]);",
        "fmod(n*bin_duration + 0.5*best_duration[p] + min_t, period)": "fmod(fma(n, bin_duration, 0.5*best_duration[p]) + min_t, period)",
    }
    for old, new in substitutions.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    reference_source = tmp_path / "astropy-fma.c"
    reference_source.write_text(text)
    library = tmp_path / "astropy-fma.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            str(reference_source),
            "-lm",
        ],
        check=True,
    )
    reference = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    reference.run_bls.argtypes = [
        ctypes.c_int,
        array,
        array,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        array,
        ctypes.c_int,
        ctypes.c_int,
    ] + [array] * 7
    reference.run_bls.restype = ctypes.c_int
    config = json.loads(CONFIG.read_text())
    rng = np.random.default_rng(20260930)
    time = np.sort(rng.uniform(0, 90, 6000))
    time = time[time % 30 > 4]
    noise = rng.uniform(40e-6, 60e-6, len(time))
    flux = rng.normal(0, 50e-6, len(time))
    for band in (
        config,
        config | config["short_period_search"],
        config | config["long_period_search"],
    ):
        periods = np.geomspace(band["period_min_days"], band["period_max_days"], 3001)
        durations = np.asarray(band["duration_days"])
        for sign in (1, -1):
            want = np.full((7, len(periods)), np.nan)
            centred_flux = sign * flux - np.median(sign * flux)
            assert (
                reference.run_bls(
                    len(time),
                    time - time.min(),
                    centred_flux,
                    1 / noise**2,
                    len(periods),
                    periods,
                    len(durations),
                    durations,
                    10,
                    0,
                    *want,
                )
                == 0
            )
            want[4] += time.min()
            got = box_search(time, sign * flux, noise, periods, durations)
            for value, expected in zip(got, want[[0, 1, 3, 4]]):
                assert value.tobytes() == expected.tobytes()


def test_sparse_quarters_do_not_define_noise():
    config = json.loads(CONFIG.read_text())
    records = np.empty(102, dtype=RECORD)
    records["time"] = np.arange(102) * 0.02
    records["flux"] = np.sin(np.arange(102)) * 1e-4
    records["error"] = 1e-4
    records["quarter"] = np.r_[np.zeros(100), np.ones(2)]
    records["cadence"] = np.arange(102)

    with pytest.raises(ShallowIneligible, match="INSUFFICIENT_USABLE_QUARTERS"):
        scan(records, config)


def test_degenerate_quarter_has_explicit_ineligible_reason():
    config = json.loads(CONFIG.read_text())
    records = np.empty(200, dtype=RECORD)
    records["time"] = np.arange(200) * 0.02043365
    records["flux"] = 0
    records["error"] = 1e-4
    records["quarter"] = np.r_[np.zeros(100), np.ones(100)]
    records["cadence"] = np.arange(200)

    with pytest.raises(ShallowIneligible, match="NONPOSITIVE_QUARTER_SCATTER"):
        scan(records, config)


def test_transit_overlap_retains_independent_signal():
    config = json.loads(CONFIG.read_text())
    time = np.arange(0, 100, 0.02043365)
    proposals = [
        {"period_days": 3.2, "t0_bkjd": 0.4, "duration_hours": 2.4},
        {"period_days": 6.4, "t0_bkjd": 0.4, "duration_hours": 2.4},
        {"period_days": 4.7, "t0_bkjd": 1.3, "duration_hours": 2.4},
    ]

    assert distinct_proposals(proposals, time, config) == [proposals[0], proposals[2]]


def test_cli_records_ineligible_source_and_continues(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    records = np.empty(200, dtype=RECORD)
    records["time"] = np.r_[
        np.arange(100) * 0.02043365,
        50 + np.arange(100) * 0.02043365,
    ]
    records["error"] = 1e-4
    records["quarter"] = np.r_[np.zeros(100), np.ones(100)]
    records["cadence"] = np.arange(200)
    for source_id, flux in (
        (1, np.zeros(200)),
        (2, np.random.default_rng(42).normal(0, 1e-4, 200)),
    ):
        records["flux"] = flux
        with (inputs / f"{source_id}.psf_flat.fsc").open("wb") as stream:
            stream.write(
                struct.pack("<8sQQQ", b"FSC_LC2\0", source_id, 0, len(records))
            )
            stream.write(records.tobytes())
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run.py", str(inputs), str(output)])

    main()

    with (output / "eligibility.csv").open(newline="") as stream:
        eligibility = list(csv.DictReader(stream))
    statuses = [(row["source_id"], row["status"], row["reason"]) for row in eligibility]
    assert statuses == [
        ("1", "INELIGIBLE", "NONPOSITIVE_QUARTER_SCATTER"),
        ("2", "SEARCHED", ""),
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["ineligible_sources"] == 1
    assert manifest["searched_sources"] == 1
