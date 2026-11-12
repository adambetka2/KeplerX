"""Contract checks for the Rust detector and current verification stages."""

import csv
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

import lc2
from deterministic import run as adapter
from deterministic import vet_survey as batch_vetter
from lc2 import valid_points


@pytest.mark.parametrize(
    ("stage", "program", "arguments", "limits"),
    [
        ("scan", "scan", ["--binary", "detector"], ("threads",)),
        ("vet", "vet", ["--python", "python"], ("workers",)),
        ("all", "run_all", ["--shards", "0", "--limit", "1500"], ("threads", "workers")),
        ("survey", "survey", ["--shards", "0", "--limit", "4000"], ("threads",)),
    ],
)
def test_compute_cli_uses_all_available_cpus(stage, program, arguments, limits, monkeypatch):
    monkeypatch.setattr(adapter, "worker_count", lambda: 32)
    captured = []
    monkeypatch.setattr(adapter, program, captured.append)
    monkeypatch.setattr(sys, "argv", ["run.py", stage, "--run", "dump/probe", *arguments])
    adapter.main()
    assert all(getattr(captured[0], name) == 32 for name in limits)
    monkeypatch.setattr(sys, "argv", [*sys.argv, f"--{limits[0]}", "8"])
    adapter.main()
    assert getattr(captured[1], limits[0]) == 8


def test_full_cpu_configuration_preserves_scientific_contract(monkeypatch):
    import vetting
    from runtime import execution_workers

    original = adapter.read(ROOT / "pipeline/vetting/config.json")
    original["discovery_contract"] = adapter.detector_contract()
    original.update(workers=4, network_workers=4)
    original["alias"]["workers"] = 2
    automatic = {**original, "workers": None, "network_workers": None,
                 "alias": {**original["alias"], "workers": None}}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    resolved = execution_workers(automatic)
    assert resolved["workers"] == resolved["alias"]["workers"] == resolved["network_workers"] == 32
    assert vetting.pipeline_contract(original) == vetting.pipeline_contract(resolved)


def test_frozen_pixel_configuration_in_isolated_module_context(tmp_path):
    program = """
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from pipeline.deterministic import vet_survey

assert str(root / "pipeline/deterministic") not in sys.path
assert vet_survey.pipeline.frozen_pixel_configuration(SimpleNamespace()) == {}
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program, str(ROOT)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_pixel_reserve_guard_in_direct_script_import_context(tmp_path):
    program = """
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

script = Path(sys.argv[1])
assert str(script.parents[2]) not in sys.path
namespace = runpy.run_path(str(script), run_name="reserve_contract")
args = SimpleNamespace(disposition_policy=namespace["DISPOSITION_POLICY"], pixel_input_role="development")
guard = namespace["verify_pixel_reserve"]
cache = Path(sys.argv[2])
os.environ["SPACELAB_EXECUTION_PROFILE"] = "cloud-development"
guard(args, cache, 2147483648)
external = namespace["EXTERNAL_DUMPS"][0] / "reserve-contract-cache"
for profile, role, target in (
    ("production", "development", cache),
    ("cloud-development", "evaluation", cache),
    ("cloud-development", "development", external),
):
    os.environ["SPACELAB_EXECUTION_PROFILE"] = profile
    args.pixel_input_role = role
    try:
        guard(args, target, 2147483648)
    except ValueError as error:
        assert "requires opened cloud development storage" in str(error)
    else:
        raise AssertionError("Reduced reserve accepted outside opened cloud development")
os.environ["SPACELAB_EXECUTION_PROFILE"] = "production"
args.pixel_input_role = "evaluation"
approved = namespace["EXTERNAL_DUMPS"][0] / "controls-20260926" / "pixel-cache"
guard(args, approved, 2147483648)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(ROOT / "pipeline/deterministic/run.py"),
         "/tmp/spacelab-reserve-contract-cache"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def raw():
    count = 320
    flux = 1000 * (1 + 0.001 * np.sin(np.arange(count)))
    data = {
        "time_bkjd": np.arange(count) * 0.02,
        "cadenceno": np.arange(count, dtype=np.int32),
        "quarter": np.repeat([1, 2], count // 2).astype(np.int16),
        "sap_quality": np.zeros(count, dtype=np.int64),
        "flatten_mask": np.zeros(count, dtype=np.int64),
    }
    for prefix in ("psf_flat_kbonus", "psf_stitched", "sap_stitched"):
        data[f"{prefix}_flux"] = flux.copy()
        data[f"{prefix}_flux_err"] = np.full(count, 0.3)
    data.update(
        qdiag_time_bkjd=data["time_bkjd"].copy(),
        qdiag_cadenceno=data["cadenceno"].copy(),
        qdiag_quarter=data["quarter"].copy(),
        qdiag_sap_quality=data["sap_quality"].copy(),
    )
    for prefix in ("psf", "sap"):
        data[f"qdiag_{prefix}_flux"] = flux.copy()
        data[f"qdiag_{prefix}_flux_err"] = np.full(count, 0.3)
    data["qdiag_psf_flux_nova"] = flux.copy()
    data["qdiag_psf_flux_err_nova"] = np.full(count, 0.3)
    return data


def rust_row(rank=1, classification="TRANSIT_LIKE", period="2"):
    return {
        "source_id": "123",
        "photometry": "psf_flat",
        "signal_rank": str(rank),
        "classification": classification,
        "search_pass": "" if classification in {"EVENT_LIKE", "NO_SIGNAL"} else "1",
        "protocol_version": adapter.read(adapter.CONFIG)["protocol_version"],
        "period_days": period,
        "t0_bkjd": "1",
        "duration_hours": "2.4",
        "mes": "9",
        "search_mes": "" if classification in {"EVENT_LIKE", "NO_SIGNAL"} else "9",
        "harmonic_of_rank1": "false",
    }


def shallow_row(rank=1, source_id="123"):
    return {
        "source_id": source_id,
        "search_rank": str(rank),
        "search_band": "baseline",
        "period_days": "2",
        "t0_bkjd": "1",
        "duration_hours": "2.4",
        "depth_ppm": "200",
        "search_snr": "9",
        "observed_events": "3",
        "observed_quarters": "2",
        "in_transit_points": "15",
        "inverse_same_period_snr": "2",
        "dip_inverse_snr_ratio": "4.5",
        "harmonic_delta_log_likelihood": "-1",
        "odd_even_sigma": "0.5",
        "positive_cycle_fraction": "0.6",
    }


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rust_row()))
        writer.writeheader()
        writer.writerows(rows)


def write_shallow_eligibility(path, source_ids):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=adapter.shallow_detector.ELIGIBILITY_FIELDS
        )
        writer.writeheader()
        writer.writerows(
            {
                "source_id": sid,
                "status": "ELIGIBLE",
                "reason": "",
                "elapsed_seconds": "0.1",
            }
            for sid in source_ids
        )


def test_multiview_contract_preserves_archived_identity():
    contract = adapter.detector_contract()
    assert contract["origin"] == "rust_v6_shallow_v1"
    assert contract["archived_files"] == adapter.ARCHIVED_FROZEN
    assert set(contract["shallow_files"]) == {
        "run.py",
        "bls.c",
        "config.json",
        "requirements.txt",
    }
    assert contract["views"] == list(adapter.DETECTOR_VIEWS)
    assert adapter.RECORD.itemsize == 22


def test_shallow_ineligibility_is_recorded_without_hiding_scan_errors(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(adapter, "read_fsc", lambda path, sid, view: object())

    def ineligible(records, settings):
        raise adapter.shallow_detector.ShallowIneligible("INSUFFICIENT_USABLE_QUARTERS")

    monkeypatch.setattr(adapter.shallow_detector, "scan", ineligible)
    output = tmp_path / "shallow-raw.csv"
    assert adapter.scan_shallow(tmp_path, output, ["123"], config) == {"123": []}
    eligibility = adapter.shallow_eligibility(
        tmp_path / "shallow-eligibility.csv", ["123"]
    )
    assert (eligibility[0]["status"], eligibility[0]["reason"]) == (
        "INELIGIBLE",
        "INSUFFICIENT_USABLE_QUARTERS",
    )

    def failed(records, settings):
        raise RuntimeError("unexpected BLS failure")

    monkeypatch.setattr(adapter.shallow_detector, "scan", failed)
    with pytest.raises(RuntimeError, match="unexpected BLS failure"):
        adapter.scan_shallow(tmp_path, output, ["123"], config)


def test_preparation_preserves_negative_dips_and_fsc_roundtrip(raw, tmp_path):
    raw["psf_flat_kbonus_flux"][10] = 500
    raw["psf_flat_kbonus_flux"][11] = 1500
    raw["sap_quality"][12] = 1
    raw["flatten_mask"][13] = 1
    raw["psf_flat_kbonus_flux_err"][14] = -1
    raw["cadenceno"][15] = raw["cadenceno"][16]
    records = adapter.prepare_records(raw)
    assert records["cadence"][records["flux"] < -0.4].tolist() == [10]
    assert not {11, 12, 13, 14} & set(records["cadence"])
    assert len(records) == len(set(records["cadence"])) == 315
    assert (np.diff(records["time"]) >= 0).all()
    path = tmp_path / "123.fsc"
    adapter.write_fsc(path, "123", "psf_flat", records)
    assert adapter.read_fsc(path, "123", "psf_flat").tobytes() == records.tobytes()
    with pytest.raises(ValueError, match="identity"):
        adapter.read_fsc(path, "456", "psf_flat")
    with pytest.raises(ValueError, match="view"):
        adapter.read_fsc(path, "123", "sap_qdiag")
    with path.open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(ValueError, match="payload"):
        adapter.read_fsc(path, "123", "psf_flat")


def test_preparation_matches_quarter_normalization(raw):
    records = adapter.prepare_records(raw)
    for quarter in [1, 2]:
        use = raw["quarter"] == quarter
        expected = (
            raw["psf_flat_kbonus_flux"][use]
            / np.median(raw["psf_flat_kbonus_flux"][use])
            - 1
        ).astype("<f4")
        np.testing.assert_array_equal(
            records["flux"][records["quarter"] == quarter], expected
        )


def test_additive_delta_manifest_preserves_base_provenance(raw, tmp_path, monkeypatch):
    metadata = {
        "source_id": "123",
        "raw_fits_sha256": "a" * 64,
        "origin_tpf_kic": 123,
    }
    base = {key: value.copy() for key, value in raw.items()}
    monkeypatch.setattr(lc2, "load_source", lambda root, sid: (metadata, base))
    delta_path = tmp_path / "123.npz"
    np.savez_compressed(
        delta_path,
        psf_flat_kbonus_flux=np.full(len(raw["time_bkjd"]), -2.0),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "measurement_scope": "test flux injection",
                "transform": "additive_flux_delta",
                "sources": {
                    "123": {
                        "base_raw_fits_sha256": "a" * 64,
                        "deltas": delta_path.name,
                        "sha256": adapter.digest(delta_path),
                    }
                },
            }
        )
    )
    found_metadata, found = lc2.manifest_source(tmp_path, "123", manifest_path)
    np.testing.assert_array_equal(
        found["psf_flat_kbonus_flux"], raw["psf_flat_kbonus_flux"] - 2.0
    )
    np.testing.assert_array_equal(found["sap_stitched_flux"], raw["sap_stitched_flux"])
    assert found_metadata["input_manifest"] == str(manifest_path.resolve())


def test_additive_delta_curve_reads_only_requested_view(raw, tmp_path, monkeypatch):
    original = lc2.load_curve
    base = lc2.photometry(raw, "psf_flat")
    base["provenance"] = {
        "source_id": "123",
        "raw_fits_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        lc2,
        "load_curve",
        lambda root, source_id, view: {
            key: value.copy() if isinstance(value, np.ndarray) else dict(value)
            for key, value in base.items()
        },
    )
    delta_path = tmp_path / "123.npz"
    np.savez_compressed(
        delta_path,
        psf_flat_kbonus_flux=np.full(len(base["flux"]), -2.0),
        qdiag_psf_flux=np.zeros(len(raw["qdiag_psf_flux"])),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "measurement_scope": "test flux injection",
                "transform": "additive_flux_delta",
                "sources": {
                    "123": {
                        "base_raw_fits_sha256": "a" * 64,
                        "deltas": delta_path.name,
                        "sha256": adapter.digest(delta_path),
                    }
                },
            }
        )
    )
    found = original(tmp_path, "123", "psf_flat", manifest_path)
    np.testing.assert_array_equal(found["flux"], base["flux"] - 2.0)
    assert found["provenance"]["input_manifest"] == str(manifest_path.resolve())


def test_native_views_use_qdiag_and_ignore_flatten_mask(raw):
    raw["flatten_mask"][:] = 1
    raw["qdiag_psf_flux_nova"][10] = 500
    with pytest.raises(ValueError, match="requires 100"):
        adapter.prepare_records(raw, "psf_flat")
    records = adapter.prepare_records(raw, "psf_nova")
    assert records["cadence"][records["flux"] < -0.4].tolist() == [10]


def test_short_curve_fails_without_replacement(raw):
    raw["flatten_mask"][:250] = 1
    with pytest.raises(ValueError, match="requires 100"):
        adapter.prepare_records(raw)


def test_selection_is_fixed_and_explicit(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "source_catalogue",
        lambda root, shards: iter(
            [{"source_id": "30"}, {"source_id": "2"}, {"source_id": "10"}]
        ),
    )
    assert [r["source_id"] for r in adapter.select_sources([1], 2)] == ["2", "10"]
    assert [r["source_id"] for r in adapter.select_sources([1], 2, ["30", "2"])] == [
        "30",
        "2",
    ]
    with pytest.raises(ValueError, match="requested source count"):
        adapter.select_sources([1], 4)
    with pytest.raises(ValueError, match="distinct"):
        adapter.select_sources([1], 2, ["2", "2"])


def test_standalone_manifest_selects_sources_outside_lc2(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "measurement_scope": "external catalogue control",
                "sources": {
                    "123": {
                        "metadata": {"source_id": "123", "shard": 0},
                        "arrays": "123.npz",
                        "sha256": "a" * 64,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(
        adapter,
        "source_catalogue",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LC2 opened")),
    )
    selected = adapter.select_sources(
        [0],
        1,
        ["123"],
        experimental_manifest=manifest,
    )
    assert selected == [{"source_id": "123", "shard": 0}]


def test_standalone_manifest_requires_declared_shard(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "measurement_scope": "external catalogue control",
                "sources": {
                    "123": {
                        "metadata": {"source_id": "123", "shard": 7},
                        "arrays": "123.npz",
                        "sha256": "a" * 64,
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="present in the declared shards"):
        adapter.select_sources(
            [0],
            1,
            ["123"],
            experimental_manifest=manifest,
        )


def test_prepare_freezes_first_usable_inputs_and_preflight_ledger(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        adapter,
        "source_catalogue",
        lambda root, shards: [{"source_id": str(i)} for i in [4, 1, 3, 2]],
    )
    monkeypatch.setattr(adapter, "dump_path", lambda path: path)

    def prepared(out, source, input_manifest=None):
        if source["source_id"] == "1":
            return {"status": "INELIGIBLE", "unavailable_views": 4}
        return {
            **source,
            "status": "PREPARED",
            "inputs": {"psf_flat": {}},
            "unavailable_views": 3,
        }

    monkeypatch.setattr(adapter, "prepare_source", prepared)
    out = tmp_path / "cohort"
    adapter.prepare(
        SimpleNamespace(shards=[200, 201, 202, 203], limit=2, source_ids=None, out=out)
    )
    result = adapter.read(out / "preparation.json")
    assert result["status"] == "PREPARED"
    assert result["selection"]["source_ids"] == ["2", "3"]
    assert result["preparation_counts"] == {
        "attempted_sources": 3,
        "accepted_sources": 2,
        "ineligible_sources": 1,
        "prepared_views": 2,
        "unavailable_views": 10,
    }
    assert result["cohort_sha256"] == adapter.json_sha256(["2", "3"])


def test_explicit_preparation_preserves_ineligible_identities(tmp_path, monkeypatch):
    ids = ["1", "2", "3"]
    selection = tmp_path / "source-ids.json"
    adapter.save(selection, ids)
    monkeypatch.setattr(
        adapter,
        "source_catalogue",
        lambda root, shards: [{"source_id": sid} for sid in ids],
    )
    monkeypatch.setattr(adapter, "dump_path", lambda path: path)

    def prepared(out, source, input_manifest=None):
        if source["source_id"] == "2":
            return {"status": "INELIGIBLE", "unavailable_views": 4}
        path = out / "inputs" / f"{source['source_id']}.psf_flat.fsc"
        path.write_bytes(source["source_id"].encode())
        return {
            **source,
            "status": "PREPARED",
            "inputs": {
                "psf_flat": {"fsc_path": str(path), "fsc_sha256": adapter.digest(path)}
            },
            "unavailable_views": 3,
        }

    monkeypatch.setattr(adapter, "prepare_source", prepared)
    out = tmp_path / "cohort"
    adapter.prepare(SimpleNamespace(shards=[0], limit=3, source_ids=selection, out=out))
    result = adapter.validate_preparation(out)
    assert result["selection"]["requested_source_ids"] == ids
    assert result["selection"]["source_ids"] == ["1", "3"]
    assert result["ineligible"] == [
        {
            "source_id": "2",
            "reason": "DISCOVERY_VIEW_UNAVAILABLE",
            "unavailable_views": 4,
        }
    ]
    assert result["preparation_counts"]["attempted_sources"] == 3
    assert result["preparation_counts"]["accepted_sources"] == 2
    result["ineligible"][0]["source_id"] = "1"
    adapter.save(out / "preparation.json", result)
    with pytest.raises(ValueError, match="accounting is incomplete"):
        adapter.validate_preparation(out)
    result["ineligible"][0]["source_id"] = "2"
    adapter.save(out / "preparation.json", result)
    adapter.save(selection, list(reversed(ids)))
    with pytest.raises(ValueError, match="selection changed"):
        adapter.validate_preparation(out)


def test_raw_csv_reconciles_every_source_and_rank(tmp_path):
    path = tmp_path / "raw.csv"
    csv_file(path, [rust_row(), rust_row(2, "EB_LIKE")])
    assert (
        len(adapter.parse_rows(path, {("123", "psf_flat")})[("123", "psf_flat")]) == 2
    )
    with pytest.raises(ValueError, match="omitted"):
        adapter.parse_rows(path, {("123", "psf_flat"), ("456", "psf_flat")})
    csv_file(path, [rust_row(), rust_row()])
    with pytest.raises(ValueError, match="duplicate"):
        adapter.parse_rows(path, {("123", "psf_flat")})


@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
def test_full_view_preparation_does_not_require_psf_flat(
    raw, tmp_path, monkeypatch, view
):
    raw["psf_flat_kbonus_flux"][:] = np.nan
    metadata = {
        "origin_tpf_kic": 42,
        "raw_fits_sha256": "a" * 64,
        "shard": 0,
        "shard_path": str(tmp_path),
    }
    monkeypatch.setattr(adapter, "load_source", lambda *args: (metadata, raw))
    (tmp_path / "inputs").mkdir()
    result = adapter.prepare_source(tmp_path, {"source_id": "123"}, discovery_view=view)
    assert result["status"] == "PREPARED"
    assert view in result["inputs"]
    assert "psf_flat" not in result["inputs"]
    assert result["inputs"][view]["retained_cadences"] >= 100


def test_full_view_vetting_keeps_its_own_quality_masks(raw):
    from lc2 import uses_flatten_mask, vetting_points
    from vetting_models import physical_input

    raw["flatten_mask"][:] = 1
    raw["qdiag_sap_quality"][0] = 1
    config = {"flatten_mask_views": ["psf_flat", "psf_nova"]}
    for view in ("psf_qdiag", "sap_qdiag"):
        curve = lc2.photometry(raw, view)
        assert not vetting_points(curve, -1).any()
        mask = uses_flatten_mask(view, config)
        assert not mask
        good = vetting_points(curve, -1, mask)
        assert not good[0] and good[1:].all()
        np.testing.assert_array_equal(physical_input(raw, view, -1, mask)[:, 3], good)
    assert uses_flatten_mask("psf_flat", config)
    assert uses_flatten_mask("psf_qdiag", {})


def test_shallow_inverse_without_positive_model_preserves_proposal(tmp_path):
    path = tmp_path / "shallow-raw.csv"
    row = shallow_row() | {
        "inverse_same_period_snr": "-inf",
        "dip_inverse_snr_ratio": "9000000.0",
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=adapter.shallow_detector.FIELDS)
        writer.writeheader()
        writer.writerow(row)
    original_bytes = path.read_bytes()

    assert adapter.parse_shallow_rows(path, {"123"}) == {"123": [row]}
    assert path.read_bytes() == original_bytes


def test_shallow_long_band_enters_the_queue(tmp_path):
    path = tmp_path / "shallow-raw.csv"
    row = shallow_row() | {
        "search_band": "long",
        "period_days": "62.8",
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=adapter.shallow_detector.FIELDS)
        writer.writeheader()
        writer.writerow(row)

    assert adapter.parse_shallow_rows(path, {"123"}) == {"123": [row]}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "period_days", "t0_bkjd", "duration_hours", "depth_ppm", "search_snr",
            "inverse_same_period_snr", "dip_inverse_snr_ratio",
            "harmonic_delta_log_likelihood", "odd_even_sigma", "positive_cycle_fraction",
        )
        for value in ("nan", "inf", "-inf")
        if (field, value) != ("inverse_same_period_snr", "-inf")
    ],
)
def test_shallow_nonfinite_diagnostics_still_refuse(tmp_path, field, value):
    path = tmp_path / "shallow-raw.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=adapter.shallow_detector.FIELDS)
        writer.writeheader()
        writer.writerow(shallow_row() | {field: value})

    with pytest.raises(ValueError, match="nonfinite"):
        adapter.parse_shallow_rows(path, {"123"})


def test_supplied_windows_preserve_epoch_cycles_and_quarter_gaps(raw):
    records = adapter.prepare_records(raw)
    records = records[(records["time"] < 2) | (records["time"] > 4)]
    alerts, hypothesis = adapter.supplied_events(rust_row(), records)
    assert [a["cycle"] for a in alerts] == [0, 2]
    assert [a["quarter"] for a in alerts] == [1, 2]
    assert hypothesis["period_days"] == 2
    assert hypothesis["epoch_bkjd"] == 1
    assert all(a["in_window_detector_cadences"] > 0 for a in alerts)
    shallow, _ = adapter.shallow_events(shallow_row(), records)
    assert [
        (a["cycle"], a["quarter"], a["in_window_detector_cadences"]) for a in shallow
    ] == [(a["cycle"], a["quarter"], a["in_window_detector_cadences"]) for a in alerts]
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.supplied_events(rust_row(period="nan"), records)


def test_detector_alerts_remain_on_the_primary_view(raw):
    records = adapter.prepare_records(raw, adapter.DISCOVERY_VIEW)
    alerts, hypothesis = adapter.supplied_events(rust_row(), records)
    assert {alert["photometry"] for alert in alerts} == {adapter.DISCOVERY_VIEW}
    assert {alert["timing_origin"] for alert in alerts} == {"SUPPLIED_RUST_EPHEMERIS"}
    assert set(hypothesis["alert_ids"]) == {alert["event_id"] for alert in alerts}


def test_sparse_event_window_has_no_forced_period(raw):
    records = adapter.prepare_records(raw, adapter.DISCOVERY_VIEW)
    row = rust_row(classification="EVENT_LIKE", period="")
    alerts, hypothesis = adapter.supplied_events(row, records)
    assert hypothesis is None
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "SPARSE_EVENT_WINDOW"
    assert alerts[0]["cycle"] is None
    assert alerts[0]["timing_origin"] == "SUPPLIED_RUST_EVENT"


def test_survivor_failure_enters_review_lane_without_reclassification(raw):
    records = adapter.prepare_records(raw, adapter.DISCOVERY_VIEW)
    row = dict(
        rust_row(classification="SYSTEMATIC_LIKE"),
        duration_consistency_status="FAIL",
        quarter_loo_status="PASS",
    )
    alerts, hypothesis = adapter.supplied_events(row, records)
    assert alerts
    assert row["classification"] == "SYSTEMATIC_LIKE"
    assert hypothesis["detector_followup_lane"] == "SURVIVOR_REVIEW"
    assert adapter.periodic_followup_lane(row) == "SURVIVOR_REVIEW"
    row["duration_consistency_status"] = "NOT_RUN"
    assert adapter.periodic_followup_lane(row) is None


def test_followup_expands_periodic_and_sparse_proposals(raw, tmp_path, monkeypatch):
    write_shallow_eligibility(tmp_path / "shallow-eligibility.csv", ["123"])
    inputs = {}
    for view in adapter.DETECTOR_VIEWS:
        records = adapter.prepare_records(raw, view)
        path = tmp_path / f"123.{view}.fsc"
        adapter.write_fsc(path, "123", view, records)
        inputs[view] = {
            "fsc_path": str(path),
            "fsc_sha256": adapter.digest(path),
        }
    source = {
        "source_id": "123",
        "inputs": inputs,
        "views": adapter.views(raw, inputs),
        "flatten_mask_sha256": adapter.flatten_digest(raw),
    }
    rejected = [
        rust_row(i + 2, status)
        for i, status in enumerate(
            ["EB_LIKE", "SYSTEMATIC_LIKE", "LOW_SIGNIFICANCE", "NO_SIGNAL"]
        )
    ]
    review = dict(
        rust_row(8, "SYSTEMATIC_LIKE"),
        duration_consistency_status="PASS",
        quarter_loo_status="FAIL",
    )
    shallow = dict(shallow_row(), t0_bkjd="1.5")
    monkeypatch.setattr(adapter, "load_source", lambda root, sid: ({}, raw))
    queue, ledger, _ = adapter.build_queue(
        tmp_path,
        {"sources": [source], "detector_contract": adapter.detector_contract()},
        {
            ("123", "psf_flat"): [
                rust_row(),
                dict(rust_row(6, "EVENT_LIKE", period=""), t0_bkjd="1.5"),
                dict(rust_row(7, "EVENT_LIKE", period=""), t0_bkjd="1"),
                *rejected,
                review,
            ]
        },
        {"123": [shallow]},
    )
    proposal = adapter.read(tmp_path / queue["rows"][0]["evidence"])
    assert len(proposal["rust_rows"]) == ledger[0]["raw_rows"] == 8
    assert len(proposal["supplied_hypotheses"]) == 7
    assert proposal["shallow_rows"] == [shallow]
    assert all(
        not h["alert_ids"]
        for h in proposal["supplied_hypotheses"]
        if not h["followup_eligible"]
    )
    assert {a["detector_signal_id"] for a in proposal["alerts"]} == {
        "rust-v6-123-1",
        "rust-v6-event-123-6",
        "rust-v6-123-8",
        "shallow-v1-123-1",
    }
    assert {a["photometry"] for a in proposal["alerts"]} == {adapter.DISCOVERY_VIEW}
    eligible = [h for h in proposal["supplied_hypotheses"] if h["followup_eligible"]]
    assert {h["detector_followup_lane"] for h in eligible} == {
        "PRIMARY_TRANSIT_LIKE",
        "SURVIVOR_REVIEW",
        "SHALLOW_BLS",
    }
    assert all(
        h["forced_photometry_views"] == sorted(adapter.DETECTOR_VIEWS) for h in eligible
    )
    assert {v["view"] for v in proposal["views"] if v["status"] == "SCORED"} == {
        *adapter.DETECTOR_VIEWS
    }
    queue, ledger, _ = adapter.build_queue(
        tmp_path,
        {"sources": [source], "detector_contract": adapter.detector_contract()},
        {("123", "psf_flat"): rejected},
        {"123": []},
    )
    assert not queue["rows"]
    assert ledger[0]["status"] == "DETECTOR_REJECTION"


def test_current_vetting_preserves_supplied_aliases_and_primary_view(raw, monkeypatch):
    vetting = importlib.import_module("vetting")
    periodic = importlib.import_module("vetting_periodic")
    signals = importlib.import_module("vetting_signals")
    records = adapter.prepare_records(raw)
    alerts, hypotheses = [], []
    for row in [rust_row(), rust_row(2, period="4")]:
        events, hypothesis = adapter.supplied_events(row, records)
        alerts.extend(events)
        hypotheses.append(
            dict(
                hypothesis,
                followup_eligible=True,
                forced_photometry_views=list(adapter.DETECTOR_VIEWS),
            )
        )
    sparse, hypothesis = adapter.supplied_events(
        rust_row(3, "EVENT_LIKE", period=""), records
    )
    assert hypothesis is None
    alerts.extend(sparse)
    proposal = {"views": adapter.views(raw), "alerts": alerts}
    events = vetting.event_specs(proposal)
    assert len(events) == len(alerts)
    assert {event["photometry"] for event in events} == {"psf_flat"}
    result = signals.associate(
        {"events": events, "supplied_hypotheses": hypotheses}, raw, {}
    )
    assert [s["period_days"] for s in result["signals"]] == [2, 4, None]
    assert {eid for s in result["signals"] for eid in s["event_ids"]} == {
        e["event_id"] for e in events
    }
    assert {h["epoch_bkjd"] for h in result["association"]["hypotheses"]} == {1}
    assert all(
        h["forced_photometry_views"] == list(adapter.DETECTOR_VIEWS)
        for h in result["association"]["hypotheses"]
    )
    measured_views = []

    def verify_hypothesis(raw, hypothesis, units, views, config):
        measured_views.append(views)
        return {"status": "MEASURED", "views": {}}

    monkeypatch.setattr(periodic, "verify_hypothesis", verify_hypothesis)
    periodic.verify(result, raw, {})
    assert measured_views
    assert all(set(views) == set(adapter.DETECTOR_VIEWS) for views in measured_views)


def test_shallow_only_source_reaches_shared_signal_association(
    raw, tmp_path, monkeypatch
):
    write_shallow_eligibility(tmp_path / "shallow-eligibility.csv", ["123"])
    vetting = importlib.import_module("vetting")
    signals = importlib.import_module("vetting_signals")
    records = adapter.prepare_records(raw)
    path = tmp_path / "123.psf_flat.fsc"
    adapter.write_fsc(path, "123", "psf_flat", records)
    source = {
        "source_id": "123",
        "inputs": {"psf_flat": {"fsc_path": str(path)}},
        "views": adapter.views(raw, {"psf_flat": {"fsc_path": str(path)}}),
        "flatten_mask_sha256": adapter.flatten_digest(raw),
    }
    monkeypatch.setattr(adapter, "load_source", lambda root, sid: ({}, raw))
    queue, ledger, _ = adapter.build_queue(
        tmp_path,
        {"sources": [source], "detector_contract": adapter.detector_contract()},
        {
            ("123", "psf_flat"): [
                dict(
                    rust_row(classification="NO_SIGNAL", period=""),
                    t0_bkjd="",
                    duration_hours="",
                )
            ]
        },
        {"123": [shallow_row()]},
    )
    assert ledger[0]["status"] == "PROPOSALS"
    assert len(queue["rows"]) == 1
    proposal = adapter.read(tmp_path / queue["rows"][0]["evidence"])
    events = vetting.event_specs(proposal)
    associated = signals.associate(
        {"events": events, "supplied_hypotheses": proposal["supplied_hypotheses"]},
        raw,
        {},
    )
    assert len(associated["signals"]) == 1
    assert associated["signals"][0]["signal_id"] == "shallow-v1-123-1"
    assert associated["signals"][0]["period_days"] == 2
    assert (
        associated["signals"][0]["timing_hypotheses"][0]["shallow_record"]
        == shallow_row()
    )


def test_ordered_closest_indices_match_linear_search():
    periodic = importlib.import_module("vetting_periodic")
    time = np.array([np.nan, 1.0, 1.0, 3.0, 8.0, 10.0])
    valid = np.isfinite(time)
    centres = np.array([-1.0, 1.0, 2.0, 7.0, 12.0])
    expected = np.array(
        [
            np.flatnonzero(valid)[np.argmin(abs(time[valid] - centre))]
            for centre in centres
        ]
    )
    np.testing.assert_array_equal(
        periodic.closest_indices(time, valid, centres), expected
    )


def test_unordered_closest_indices_match_linear_search():
    periodic = importlib.import_module("vetting_periodic")
    time = np.array([3.0, 1.0, 8.0])
    valid = np.ones(len(time), dtype=bool)
    centres = np.array([2.0, 7.0])
    np.testing.assert_array_equal(
        periodic.closest_indices(time, valid, centres), np.array([0, 2])
    )


def test_prepared_interval_input_preserves_measurement(raw):
    models = importlib.import_module("vetting_models")
    config = {
        "quality_bitmask": 0,
        "minimum_context_days": 0.5,
        "local_context_durations": 4,
        "exposure_days": lc2.CADENCE,
    }
    expected = models.measure_interval(raw, 1.0, 0.1, "psf_flat", config, 1)
    prepared = models.physical_input(raw, "psf_flat", 0)[raw["quarter"] == 1]
    measured = models.measure_interval(
        raw, 1.0, 0.1, "psf_flat", config, 1, prepared=prepared
    )
    assert measured == expected


def test_repeated_signal_work_is_quarter_covering_and_bounded():
    signals = importlib.import_module("vetting_signals")
    units = [
        {
            "unit_id": f"unit-{index}",
            "event_ids": [f"event-{index}"],
            "quarter": index % 17,
            "centre_bkjd": float(index),
        }
        for index in range(500)
    ]
    selected = signals.representative_units(units, 64)
    assert len(selected) == 64
    assert {unit["quarter"] for unit in selected} == set(range(17))
    assert selected == signals.representative_units(units, 64)


def test_repeated_shape_gate_uses_joint_original_timing_measurement():
    triage = importlib.import_module("vetting_triage")
    fit = {
        "status": "FIT_COMPLETE",
        "bic": 0.0,
        "alternatives": {
            name: {"bic": 20.0}
            for name in (
                "linear",
                "quadratic",
                "step",
                "dropout",
                "single_cadence",
                "shared_single_cadence",
                "signed_box",
                "sinusoid",
            )
        },
    }
    signal = {
        "events": [
            {"unit_id": "u1", "event_ids": ["e1"]},
            {"unit_id": "u2", "event_ids": ["e2"]},
        ],
        "timing_hypotheses": [
            {
                "hypothesis_id": "h1",
                "timing_basis": "ORIGINAL_PROPOSAL",
                "unit_ids": ["u1", "u2"],
            }
        ],
    }
    row = {
        "orbital": {"hypotheses": {"h1": {"views": {"psf_flat": {"joint_shape": fit}}}}}
    }
    policy = {"shape_false_alarm_bic_margin": 10, "shape_transit_bic_margin": 10}
    result = triage.shape_gate(row, signal, policy)
    assert result["decision"] == "PASS"
    assert result["evidence"]["view"] == "psf_flat"


def test_transit_like_dip_box_is_not_a_nuisance_model():
    disposition = importlib.import_module("vetting_disposition")
    alternatives = {
        name: {"bic": 20.0}
        for name in (
            "linear",
            "quadratic",
            "step",
            "dropout",
            "single_cadence",
            "shared_single_cadence",
            "sinusoid",
        )
    }
    alternatives["signed_box"] = {
        "bic": 0.0,
        "equivalent_to": ["transit_like_box"],
    }
    result = disposition.shape_evidence(
        {"status": "FIT_COMPLETE", "bic": 50.0, "alternatives": alternatives}
    )
    assert result["status"] == "TRANSIT_COMPATIBLE"
    assert result["transit_model"] == "transit_like_box"
    assert result["delta_bic"] == 20.0


def test_orbital_gate_uses_primary_view_and_kepler_sweet_thresholds():
    disposition = importlib.import_module("vetting_disposition")
    rule = {
        "id": "orbital-eb-v2",
        "status": "RELEASED",
        "maximum_odd_even_bic_gain": 10,
        "minimum_secondary_sigma": 7,
        "maximum_secondary_to_primary_depth_ratio": 0.5,
        "maximum_phase_modulation_period_days": 10,
        "minimum_phase_modulation_signal_to_noise": 50,
        "minimum_phase_modulation_amplitude_fraction": 0.005,
        "maximum_phase_modulation_to_primary_depth_ratio": 1,
    }

    def view(amplitude, signal_to_noise, period=3.0):
        return {
            "odd_even": {"two_depth_bic_gain": 0.0},
            "primary_depth": {"status": "MEASURED", "depth_fraction": 0.01},
            "secondary_phase_half": {
                "status": "MEASURED",
                "depth_fraction": 0.001,
                "depth_error": 0.001,
            },
            "phase_modulation": {
                "status": "MEASURED",
                "tce_period_days": period,
                "strongest_signal": {
                    "period_ratio": 0.5,
                    "amplitude_fraction": amplitude,
                    "signal_to_noise": signal_to_noise,
                },
            },
        }

    signal = {
        "signal_id": "signal-1",
        "event_ids": ["event-1", "event-2"],
        "timing_hypotheses": [
            {"hypothesis_id": "h1", "timing_basis": "ORIGINAL_PROPOSAL"}
        ],
    }
    row = {
        "source_id": "source-1",
        "orbital": {
            "hypotheses": {
                "h1": {
                    "views": {
                        "psf_flat": view(0.004, 100),
                        "psf_qdiag": view(0.006, 100),
                    }
                }
            }
        },
    }
    policy = {"status": "RELEASED", "assessment_rules": {"orbital": rule}}
    result = disposition.orbital_assessment(row, signal, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    row["orbital"]["hypotheses"]["h1"]["views"]["psf_flat"] = view(0.006, 51)
    result = disposition.orbital_assessment(row, signal, policy)
    assert result["status"] == "MATERIAL_CONFLICT"

    row["orbital"]["hypotheses"]["h1"]["views"]["psf_flat"] = view(0.006, 50)
    result = disposition.orbital_assessment(row, signal, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    row["orbital"]["hypotheses"]["h1"]["views"]["psf_flat"] = view(0.011, 100)
    result = disposition.orbital_assessment(row, signal, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    row["orbital"]["hypotheses"]["h1"]["views"]["psf_flat"] = view(
        0.006, 100, period=10.0
    )
    result = disposition.orbital_assessment(row, signal, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"


def test_phase_modulation_requires_one_phase_across_quarters(raw):
    periodic = importlib.import_module("vetting_periodic")
    time = raw["time_bkjd"]
    period = 0.8
    wave = 0.006 * np.cos(2 * np.pi * time / period)
    for prefix in ("psf_flat_kbonus", "psf_stitched", "sap_stitched"):
        raw[f"{prefix}_flux"] = 1000 * (1 + wave)
        raw[f"{prefix}_flux_err"] = np.full(len(time), 0.02)
    hypothesis = {"period_days": period, "epoch_bkjd": 0.0}
    config = {"quality_bitmask": lc2.QUALITY_BITMASK, "exposure_days": 0.0204336}
    coherent = periodic.phase_modulation(
        raw, hypothesis, duration=0.04, view="psf_flat", config=config
    )
    assert coherent["strongest_signal"]["period_ratio"] == 1.0
    assert coherent["strongest_signal"]["amplitude_fraction"] == pytest.approx(
        0.006, abs=2e-4
    )
    assert coherent["strongest_signal"]["signal_to_noise"] > 50

    raw["psf_flat_kbonus_flux"][raw["quarter"] == 2] = 1000 * (
        1 - wave[raw["quarter"] == 2]
    )
    incoherent = periodic.phase_modulation(
        raw, hypothesis, duration=0.04, view="psf_flat", config=config
    )
    assert incoherent["strongest_signal"]["amplitude_fraction"] < 0.0002


def test_exact_zero_quality_mask_rejects_every_flag(raw):
    curve = adapter.photometry(raw, "psf")
    curve["quality"][:3] = [1, 1 << 25, -1]
    good = valid_points(curve, quality_bitmask=-1)
    assert not good[:3].any()
    assert good[3:].all()


def test_vetting_mask_matches_primary_cadences_without_changing_flags(raw):
    raw["flatten_mask"][[17, 43]] = 1
    raw["sap_quality"][8] = 1 << 25
    original = {key: value.copy() for key, value in raw.items()}
    for view in ("psf", "sap", *adapter.DETECTOR_VIEWS):
        curve = lc2.photometry(raw, view)
        use = lc2.vetting_points(curve, -1)
        excluded = {17, 43} | ({8} if view in {"psf", "sap", "psf_flat"} else set())
        assert set(curve["cadence"][~use]) == excluded
        np.testing.assert_array_equal(curve["quality"], original[
            "qdiag_sap_quality" if view in {"psf_qdiag", "sap_qdiag", "psf_nova"}
            else "sap_quality"
        ])
    for key, value in raw.items():
        np.testing.assert_array_equal(value, original[key])
    primary = lc2.photometry(raw, "psf_flat")
    assert valid_points(primary, -1)[17]
    assert not lc2.vetting_points(primary, -1)[17]


def test_vetting_diagnostic_mask_uses_cadence_identity_not_row_order(raw):
    raw["flatten_mask"][17] = 1
    raw["qdiag_cadenceno"] = np.array([43, 17, 90000], dtype=np.int32)
    for key in list(raw):
        if key.startswith("qdiag_") and key != "qdiag_cadenceno":
            raw[key] = raw[key][:3]
    curve = lc2.photometry(raw, "psf_qdiag")
    np.testing.assert_array_equal(lc2.vetting_points(curve, -1), [True, False, True])


def test_rust_manifest_rejects_tampered_artifacts(tmp_path):
    queue, binary = tmp_path / "queue.json", tmp_path / "binary"
    queue.write_text("{}")
    binary.write_bytes(b"frozen binary")
    shallow = tmp_path / "bls.so"
    shallow.write_bytes(b"frozen shallow library")
    contract = adapter.detector_contract()
    manifest = {
        "schema": 2,
        "origin": contract["origin"],
        "detector_contract": contract,
        "implementation": adapter.implementation_hashes(),
        "artifacts": {str(queue): adapter.digest(queue)},
        "binary": {"path": str(binary), "sha256": adapter.digest(binary)},
        "shallow_binary": {"path": str(shallow), "sha256": adapter.digest(shallow)},
    }
    adapter.verify_discovery(queue, manifest, {"discovery_contract": contract})
    relocated = tmp_path / "external-queue.json"
    queue.rename(relocated)
    queue.symlink_to(relocated)
    adapter.verify_discovery(queue, manifest, {"discovery_contract": contract})
    queue.write_text('{"rows": []}')
    with pytest.raises(ValueError, match="artifact changed"):
        adapter.verify_discovery(queue, manifest, {"discovery_contract": contract})

    queue.write_text("{}")
    shallow.write_bytes(b"changed shallow library")
    with pytest.raises(ValueError, match="Frozen shallow BLS library changed"):
        adapter.verify_discovery(queue, manifest, {"discovery_contract": contract})


def test_offline_pixel_retrieval_never_contacts_archive(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    config = tmp_path / "config.json"
    adapter.save(config, {"network": False})
    monkeypatch.setattr(pixels, "CONFIG", config)
    monkeypatch.setattr(pixels, "RUN", tmp_path)

    def unexpected(*args):
        pytest.fail("Offline retrieval contacted MAST")

    monkeypatch.setattr(pixels, "add_archive_pixels", unexpected)
    result = pixels.retrieve_parent((123, {1, 2}), {}, {(123, 1): "local.fits"})
    assert result["quarters"]["1"]["status"] == "AVAILABLE"
    assert result["quarters"]["2"]["status"] == "NETWORK_DISABLED"


def test_parallel_pixel_retrieval_preserves_archive_order(tmp_path, monkeypatch):
    from threading import Barrier

    from astropy.io import fits

    pixels = importlib.import_module("vetting_pixels")
    filenames = [
        f"kplr000000123-{timestamp}_lpd-targ.fits.gz"
        for timestamp in ("100", "200", "300")
    ]
    mapping = {"100": 2, "200": 1, "300": 3}
    for filename in filenames:
        header = fits.Header({"QUARTER": mapping[filename.split("-")[1].split("_")[0]]})
        fits.PrimaryHDU(header=header).writeto(tmp_path / filename)
    barrier = Barrier(2)
    base = "https://archive.stsci.edu/pub/kepler/target_pixel_files/0000/000000123/"

    def download(_base, filename, _parent):
        if filename in filenames[:2]:
            barrier.wait(timeout=3)
        return tmp_path / filename

    monkeypatch.setattr(pixels, "archive_filenames", lambda _base: filenames)
    monkeypatch.setattr(pixels, "download_pixel", download)
    result = {"quarters": {}}
    pixels.add_archive_pixels(result, base, 123, {1, 2, 3}, mapping)
    assert list(result["quarters"]) == ["2", "1", "3"]
    for filename, quarter in zip(filenames, (2, 1, 3)):
        item = result["quarters"][str(quarter)]
        assert item["sha256"] == adapter.digest(tmp_path / filename)
        assert item["url"] == base + filename
        assert item["transport_uri"] == pixels.cloud_pixel_url(base, filename)
    monkeypatch.setattr(
        pixels,
        "download_pixel",
        lambda _base, filename, _parent: tmp_path / filename,
    )
    sequential = {"quarters": {}}
    pixels.add_archive_pixels(sequential, base, 123, {1, 2, 3}, {"100": 2, "300": 3})
    assert sequential == result
    mismapped = {"quarters": {}}
    pixels.add_archive_pixels(
        mismapped, base, 123, {1, 2, 3}, {"100": 1, "200": 2, "300": 3}
    )
    assert list(mismapped["quarters"]) == ["2", "3"]
    assert mismapped["index_disagreement"] == {
        "filename": filenames[0],
        "indexed_quarter": 1,
        "fits_quarter": 2,
    }


def test_cloud_pixel_url_rejects_unexpected_archive():
    pixels = importlib.import_module("vetting_pixels")
    with pytest.raises(ValueError, match="Unexpected Kepler TPF archive URL"):
        pixels.cloud_pixel_url("https://example.test/", "pixel.fits.gz")


def test_cloud_pixel_download_uses_mast_mirror(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    monkeypatch.setattr(pixels.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 1024**3))
    archive = "https://archive.stsci.edu/pub/kepler/target_pixel_files/0000/000000123/"
    cloud = pixels.cloud_pixel_url(archive, "pixel.fits.gz")
    calls = []

    def fail(url, _path, _deadline):
        calls.append(url)
        raise OSError("transfer failed")

    monkeypatch.setattr(pixels, "bounded_download", fail)
    with pytest.raises(OSError, match="after 1 attempts"):
        pixels.locked_download(
            archive,
            "pixel.fits.gz",
            123,
            tmp_path / "pixel.fits.gz",
            attempts=1,
        )
    assert calls == [cloud]


def test_standalone_catalogue_controls_use_archive_pixels(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "measurement_scope": "external catalogue control",
                "sources": {},
            }
        )
    )
    config = tmp_path / "config.json"
    adapter.save(config, {"input_manifest": str(manifest), "network": False})
    adapter.save(tmp_path / "metadata.json", {})
    adapter.save(tmp_path / "cohort.json", [])
    monkeypatch.setattr(pixels, "CONFIG", config)
    monkeypatch.setattr(pixels, "ROOT", tmp_path)
    monkeypatch.setattr(pixels, "RUN", tmp_path)
    monkeypatch.setattr(pixels, "PIXELS", tmp_path / "pixels")
    pixels.retrieve()
    assert adapter.read(tmp_path / "pixel-retrieval.json") == {
        "parents": 0,
        "results": [],
    }


def test_prf_calibration_path_is_channel_specific():
    pixels = importlib.import_module("vetting_pixels")
    assert pixels.kepler_prf_path(9, Path("cache")) == Path(
        "cache/kplr04.1_2011265_prf.fits"
    )


def test_prf_download_is_installed_atomically(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    verified = []

    def download(path):
        path.write_bytes(b"calibration")

    def kepler_prf(channel, cache_dir):
        return {"channel": channel}

    monkeypatch.setattr(pixels, "download_prf", download)
    monkeypatch.setattr(pixels, "KeplerPRF", kepler_prf)
    monkeypatch.setattr(pixels, "verify_prf", lambda path: verified.append(path))
    prf, calibration = pixels.load_kepler_prf(9, tmp_path)

    path = tmp_path / "kplr04.1_2011265_prf.fits"
    assert prf == {"channel": 9}
    assert path.read_bytes() == b"calibration"
    assert calibration == {"filename": path.name, "sha256": adapter.digest(path)}
    assert len(verified) == 1
    assert not [item for item in tmp_path.iterdir() if item.is_dir()]


def test_prf_download_retries_a_bounded_transfer(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    calls = []

    def download(url, path, deadline_seconds):
        calls.append((url, path, deadline_seconds))
        if len(calls) == 1:
            path.write_bytes(b"partial")
            raise OSError("slow archive")
        assert path.read_bytes() == b"partial"
        path.write_bytes(b"calibration")

    monkeypatch.setattr(pixels, "bounded_download", download)
    monkeypatch.setattr(
        pixels, "verify_prf", lambda path: path.read_bytes() == b"calibration"
    )
    path = tmp_path / "kplr04.1_2011265_prf.fits"
    pixels.download_prf(path, attempts=2)
    assert path.read_bytes() == b"calibration"
    assert len(calls) == 2
    assert not path.with_suffix(".fits.download").exists()


def test_prf_download_preserves_partial_after_bounded_attempts(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    calls = []

    def download(url, path, deadline_seconds):
        calls.append((url, path, deadline_seconds))
        path.write_bytes(path.read_bytes() + b"part" if path.exists() else b"part")
        raise OSError("slow archive")

    monkeypatch.setattr(pixels, "bounded_download", download)
    path = tmp_path / "kplr04.1_2011265_prf.fits"
    with pytest.raises(OSError, match="after 20 attempts"):
        pixels.download_prf(path, deadline_seconds=17)
    partial = path.with_suffix(".fits.download")
    assert len(calls) == 20
    assert {call[2] for call in calls} == {17}
    assert partial.read_bytes() == b"part" * 20


def test_tpf_download_preserves_partial_after_bounded_attempts(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    monkeypatch.setattr(pixels.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 1024**3))
    calls = []

    def download(url, path, deadline_seconds):
        calls.append((url, path, deadline_seconds))
        path.write_bytes(path.read_bytes() + b"part" if path.exists() else b"part")
        raise OSError("slow archive")

    monkeypatch.setattr(pixels, "bounded_download", download)
    path = tmp_path / "pixel.fits.gz"
    archive = "https://archive.stsci.edu/pub/kepler/target_pixel_files/0000/000000001/"
    with pytest.raises(OSError, match="after 20 attempts"):
        pixels.locked_download(
            archive,
            "pixel.fits.gz",
            1,
            path,
            deadline_seconds=17,
        )
    partial = path.with_suffix(path.suffix + ".tmp")
    assert len(calls) == 20
    assert {call[0] for call in calls} == {
        pixels.cloud_pixel_url(archive, "pixel.fits.gz")
    }
    assert {call[2] for call in calls} == {17}
    assert partial.read_bytes() == b"part" * 20


def test_archive_transfer_uses_bounded_curl(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    calls = []

    def run(command, check):
        calls.append((command, check))
        Path(command[command.index("--output") + 1]).write_bytes(b"archive")

    monkeypatch.setattr(pixels.subprocess, "run", run)
    target = tmp_path / "archive.fits"
    pixels.bounded_download("https://example.test/archive.fits", target, 17)
    command, check = calls[0]
    assert check is True
    assert command[command.index("--connect-timeout") + 1] == "10"
    assert command[command.index("--max-time") + 1] == "17"
    assert command[command.index("--continue-at") + 1] == "-"
    assert target.read_bytes() == b"archive"


def test_gaia_scene_order_is_deterministic(tmp_path, monkeypatch):
    evidence = importlib.import_module("vetting_evidence")
    config = tmp_path / "config.json"
    adapter.save(config, {"neighbour_radius_arcsec": 60})
    monkeypatch.setattr(evidence, "CONFIG", config)
    source = {"source_id": "3", "ra_deg": 10, "dec_deg": 20}
    rows = [
        {
            "source_id": source_id,
            "ra": 10 + offset,
            "dec": 20,
            "phot_g_mean_mag": 15,
        }
        for source_id, offset in (("2", 0.001), ("1", -0.001))
    ]
    scene = evidence.source_scene(source, [{"status": "COMPLETE", "rows": rows}])
    assert [row["source_id"] for row in scene["sources"]] == ["1", "2"]


def test_localized_event_cache_preserves_result_and_failure(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    monkeypatch.setattr(pixels, "RUN", tmp_path)
    events = [{"event_id": "one", "quarter": 1}, {"event_id": "two", "quarter": 1}]
    quarters = {"1": {"status": "AVAILABLE", "path": "tpf.fits"}}
    monkeypatch.setattr(pixels, "localize_event", lambda *_: {"status": "MEASURED"})
    monkeypatch.setattr(pixels, "attribute_pixels", lambda *_: {"status": "MATCH"})
    first = pixels.localized_event("7", events[0], quarters, {}, {"gaia_scene": {}})
    assert first == {
        "status": "MEASURED",
        "tpf_path": "tpf.fits",
        "attribution": {"status": "MATCH"},
    }
    monkeypatch.setattr(
        pixels,
        "localize_event",
        lambda *_: (_ for _ in ()).throw(ValueError("bad fit")),
    )
    assert (
        pixels.localized_event("7", events[0], quarters, {}, {"gaia_scene": {}})
        == first
    )
    second = pixels.localized_event("7", events[1], quarters, {}, {"gaia_scene": {}})
    assert second == {"status": "LOCALIZATION_FAILED", "error": "ValueError: bad fit"}
    assert adapter.read(tmp_path / "localizations-final/7-two.json") == second
    monkeypatch.setattr(pixels, "localize_event", lambda *_: {"status": "MEASURED"})
    third = pixels.localized_event(
        "7", {"event_id": "three", "quarter": 1}, quarters, {}, {}
    )
    assert third == {
        "status": "LOCALIZATION_FAILED",
        "error": "KeyError: 'gaia_scene'",
    }


def test_localization_tasks_preserve_first_event_and_order(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    triage = importlib.import_module("vetting_triage")
    monkeypatch.setattr(pixels, "RUN", tmp_path)
    monkeypatch.setattr(
        triage, "measured_event_ids", lambda row: {"one", "two", "three"}
    )
    adapter.save(
        tmp_path / "sources/7.json",
        {
            "pixel_jobs": [
                {"event_id": "one", "quarter": 1},
                {"event_id": "skip", "quarter": 1},
                {"event_id": "two", "quarter": 2},
                {"event_id": "one", "quarter": 3},
                {"event_id": "three", "quarter": 3},
            ],
            "gaia_scene": {"sources": []},
        },
    )
    tasks = pixels.localization_tasks([{"source_id": "7"}], events_per_task=2)
    assert [[event["event_id"] for event in task[1]] for task in tasks] == [
        ["one", "two"],
        ["three"],
    ]
    assert tasks[0][1][0]["quarter"] == 1


def localization_record(event_id, quarter, offset, correlation=0.9, inside=True):
    return {
        "event_id": event_id,
        "quarter": quarter,
        "target_inside_stamp": inside,
        "prf_correlation": correlation,
        "tangent_offset_arcsec": offset,
        "tangent_covariance_arcsec2": [[0.04, 0.0], [0.0, 0.04]],
    }


def localization_policy():
    return {
        "minimum_prf_correlation": 0.3,
        "minimum_good_quarters": 3,
        "aggregate_systematic_arcsec": 0.16,
        "minimum_material_offset_arcsec": 0.2,
        "large_offset_arcsec": 2.0,
        "small_offset_sigma": 4.0,
        "large_offset_sigma": 3.0,
    }


def test_prf_fit_uses_the_full_edge_pixel_footprint():
    pixels = importlib.import_module("vetting_pixels")
    shape = (8, 7)

    class Prf:
        def evaluate(self, targets, origin, shape):
            target_row, target_column = targets[0]
            row = target_row - origin[0]
            column = target_column - origin[1]
            rows, columns = np.indices(shape)
            profile = np.exp(
                -((rows - row) ** 2 + (columns - column) ** 2) / (2 * 0.8**2)
            )
            return profile[None, :, :]

    target_column, target_row = 3.02, 7.43
    prf = Prf()
    profile = prf.evaluate([(target_row, target_column)], (0, 0), shape)[0]
    local = SimpleNamespace(
        shape=shape,
        origin_row=0,
        origin_col=0,
        prf=prf,
        good_pix_mask=np.ones(shape, dtype=bool),
        transit_weights=[5 * profile],
        transit_weights_err=[np.full(shape, 0.01)],
    )
    fit, metrics = pixels.fit_prf_heatmap(local, target_column, target_row)
    assert fit.success
    assert metrics["centerx"] == pytest.approx(target_column, abs=1e-5)
    assert metrics["centery"] == pytest.approx(target_row, abs=1e-5)
    assert pixels.inside_pixel_footprint(target_column, target_row, shape)
    assert not pixels.inside_pixel_footprint(target_column, 7.51, shape)


def test_localization_uses_robust_quality_qualified_quarters(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    config = tmp_path / "config.json"
    adapter.save(config, {"pixel_policy": localization_policy()})
    monkeypatch.setattr(pixels, "CONFIG", config)
    signal = {
        "signal_id": "s1",
        "event_ids": ["e1", "e2", "e3", "bad"],
        "events": [
            {"unit_id": name, "event_ids": [name], "quarter": quarter}
            for name, quarter in (("e1", 1), ("e2", 2), ("e3", 3), ("bad", 4))
        ],
        "triage": {"status": "ACTIVE", "history": []},
    }
    row = {
        "signals": [signal],
        "localization": {
            "events": [
                localization_record("e1", 1, [0.08, 0.02]),
                localization_record("e2", 2, [0.02, -0.03]),
                localization_record("e3", 3, [-0.04, 0.01]),
                localization_record("bad", 4, [12.0, 8.0], correlation=0.1),
            ]
        },
    }
    pixels.aggregate_signals(row)
    attribution = signal["localization"]["source_attribution"]
    assert attribution["status"] == "TARGET_COMPATIBLE"
    assert attribution["good_quarters"] == 3


def test_localization_requires_inside_stamp_and_reproduces_robust_offset(
    tmp_path, monkeypatch
):
    pixels = importlib.import_module("vetting_pixels")
    config = tmp_path / "config.json"
    adapter.save(config, {"pixel_policy": localization_policy()})
    monkeypatch.setattr(pixels, "CONFIG", config)
    signal = {
        "signal_id": "s1",
        "event_ids": ["e1", "e2", "e3", "outside"],
        "events": [
            {"unit_id": name, "event_ids": [name], "quarter": quarter}
            for name, quarter in (
                ("e1", 1),
                ("e2", 2),
                ("e3", 3),
                ("outside", 4),
            )
        ],
        "triage": {"status": "ACTIVE", "history": []},
    }
    row = {
        "signals": [signal],
        "localization": {
            "events": [
                localization_record("e1", 1, [7.8, 0.1]),
                localization_record("e2", 2, [8.0, -0.1]),
                localization_record("e3", 3, [7.9, 0.0]),
                localization_record("outside", 4, [0.0, 0.0], inside=False),
            ]
        },
    }
    pixels.aggregate_signals(row)
    attribution = signal["localization"]["source_attribution"]
    assert attribution["status"] == "ROBUST_OFFSET_CONFLICT"
    assert attribution["good_quarters"] == 3
    assert attribution["offset_distance_arcsec"] == pytest.approx(7.9)

    result = localization_record("outside", 4, [0.0, 0.0], inside=False)
    result.update(
        prf_fit_success=True,
        centroid_uncertainty={"status": "BLOCK_SANDWICH_ESTIMATED"},
    )
    assert "TARGET_OUTSIDE_STAMP" in pixels.pixel_quality(result)


def test_localization_assessment_uses_only_aggregate_status():
    disposition = importlib.import_module("vetting_disposition")
    rule = {
        "id": "pixel-attribution-v2",
        "status": "RELEASED",
        "accepted_statuses": ["TARGET_COMPATIBLE"],
        "material_statuses": ["ROBUST_OFFSET_CONFLICT"],
    }
    policy = {"status": "RELEASED", "assessment_rules": {"source_localization": rule}}
    signal = {
        "signal_id": "s1",
        "event_ids": ["e1"],
        "localization": {"source_attribution": {"status": "TARGET_COMPATIBLE"}},
    }
    row = {"source_id": "1", "localization": {}, "signals": [signal]}
    result = disposition.localization_assessment(row, signal, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    signal["localization"]["source_attribution"]["status"] = "ROBUST_OFFSET_CONFLICT"
    result = disposition.localization_assessment(row, signal, policy)
    assert result["status"] == "MATERIAL_CONFLICT"


def contamination_policy():
    return {
        "id": "catalogue-neighbour-v6",
        "status": "RELEASED",
        "minimum_scene_prf_compared_quarters": 3,
        "minimum_neighbour_depth_sigma": 7,
        "minimum_competing_source_separation_arcsec": 1,
        "minimum_catalogue_match_units": 2,
        "minimum_catalogue_match_fraction": 0.5,
        "maximum_harmonic_order": 8,
        "maximum_harmonic_period_error_fraction": 0.001,
        "minimum_harmonic_match_units": 10,
        "minimum_harmonic_expected_match_fraction": 0.25,
        "accepted_scene_statuses": ["GAIA_SCENE_QUERIED"],
        "accepted_neighbour_statuses": ["MEASURED"],
    }


def identity_catalogue_policy():
    return {
        "id": "catalogue-signal-identity-v2",
        "status": "RELEASED",
        "minimum_catalogue_match_units": 2,
        "minimum_catalogue_match_fraction": 0.5,
        "maximum_harmonic_order": 8,
        "maximum_harmonic_period_error_fraction": 0.001,
        "minimum_harmonic_match_units": 10,
        "minimum_harmonic_expected_match_fraction": 0.25,
        "material_signal_classes": ["ECLIPSING_BINARY", "FALSE_POSITIVE"],
        "accepted_catalogue_statuses": ["SCOPED_CATALOGUES_CHECKED"],
    }


def catalogue_timing(event_id, matched):
    matches = []
    if matched:
        matches.append(
            {
                "catalogue": "kepler-eb.csv",
                "source": "KIC 123",
                "separation_arcsec": 47.0,
                "known_period_days": 1.3,
                "overlaps_original_interval_nominally": True,
            }
        )
    return {"event_id": event_id, "matches": matches}


def identity_timing(event_id, matched, signal_class="ECLIPSING_BINARY"):
    record = catalogue_timing(event_id, matched)
    for match in record["matches"]:
        match.update(
            exact_identifier_match=True,
            signal_class=signal_class,
            separation_arcsec=0.02,
        )
    return record


def test_contamination_requires_signal_level_catalogue_coherence():
    disposition = importlib.import_module("vetting_disposition")
    event_ids = [f"e{index}" for index in range(6)]
    signal = {
        "signal_id": "s1",
        "event_ids": event_ids,
        "events": [
            {"unit_id": event_id, "event_ids": [event_id]} for event_id in event_ids
        ],
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_timing": [
                catalogue_timing(event_id, index < 2)
                for index, event_id in enumerate(event_ids)
            ],
            "neighbour_flux": [],
            "neighbour_comparisons": {
                "status": "MEASURED",
                "expected_sibling_ids": [],
                "selected_event_ids": event_ids,
            },
            "scene": {"status": "GAIA_SCENE_QUERIED"},
            "mechanism_status": "NOT_ESTABLISHED",
        },
    }
    row = {"source_id": "1", "signals": [signal]}
    policy = {
        "status": "RELEASED",
        "assessment_rules": {"contamination": contamination_policy()},
    }
    result = disposition.contamination_assessment(row, signal, 0, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    for record in signal["contamination"]["catalogue_timing"][:3]:
        record["matches"] = catalogue_timing(record["event_id"], True)["matches"]
    result = disposition.contamination_assessment(row, signal, 0, policy)
    assert result["status"] == "UNASSESSED"

    signal["contamination"]["mechanism_status"] = "ESTABLISHED"
    result = disposition.contamination_assessment(row, signal, 0, policy)
    assert result["status"] == "MATERIAL_CONFLICT"


def test_single_event_catalogue_coincidence_does_not_establish_conflict():
    disposition = importlib.import_module("vetting_disposition")
    signal = {
        "signal_id": "s1",
        "event_ids": ["e1"],
        "events": [{"unit_id": "e1", "event_ids": ["e1"]}],
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_timing": [catalogue_timing("e1", True)],
            "neighbour_flux": [],
            "scene": {"status": "GAIA_SCENE_QUERIED"},
            "mechanism_status": "NOT_ESTABLISHED",
        },
    }
    row = {"source_id": "1", "signals": [signal]}
    policy = {
        "status": "RELEASED",
        "assessment_rules": {"contamination": contamination_policy()},
    }
    result = disposition.contamination_assessment(row, signal, 0, policy)
    assert result["status"] == "UNASSESSED"


def test_catalogue_identity_requires_coherent_exact_false_class():
    disposition = importlib.import_module("vetting_disposition")
    event_ids = [f"e{index}" for index in range(6)]
    signal = {
        "signal_id": "s1",
        "event_ids": event_ids,
        "events": [
            {"unit_id": event_id, "event_ids": [event_id]} for event_id in event_ids
        ],
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_status": "SCOPED_CATALOGUES_CHECKED",
            "catalogue_timing": [
                identity_timing(event_id, index == 0)
                for index, event_id in enumerate(event_ids)
            ],
        },
    }
    row = {"source_id": "1", "signals": [signal]}
    policy = {
        "status": "RELEASED",
        "assessment_rules": {"identity_catalogues": identity_catalogue_policy()},
    }
    result = disposition.identity_catalogue_assessment(row, signal, 0, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    for record in signal["contamination"]["catalogue_timing"][:3]:
        record["matches"] = identity_timing(record["event_id"], True)["matches"]
    result = disposition.identity_catalogue_assessment(row, signal, 0, policy)
    assert result["status"] == "MATERIAL_CONFLICT"

    for record in signal["contamination"]["catalogue_timing"]:
        record["matches"] = identity_timing(
            record["event_id"], True, "PLANET_OR_CANDIDATE"
        )["matches"]
    result = disposition.identity_catalogue_assessment(row, signal, 0, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"

    single = {
        **signal,
        "event_ids": ["single"],
        "events": [{"unit_id": "single", "event_ids": ["single"]}],
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_status": "SCOPED_CATALOGUES_CHECKED",
            "catalogue_timing": [identity_timing("single", True)],
        },
    }
    row["signals"] = [single]
    result = disposition.identity_catalogue_assessment(row, single, 0, policy)
    assert result["status"] == "UNASSESSED"


def test_catalogue_identity_recognizes_exact_source_period_harmonic():
    disposition = importlib.import_module("vetting_disposition")
    event_ids = [f"e{index}" for index in range(150)]
    records = [
        identity_timing(event_id, index % 3 == 0, "FALSE_POSITIVE")
        for index, event_id in enumerate(event_ids)
    ]
    for record in records:
        for match in record["matches"]:
            match["known_period_days"] = 16.0035
    signal = {
        "signal_id": "alias",
        "period_days": 5.3345,
        "event_ids": event_ids,
        "events": [
            {"unit_id": event_id, "event_ids": [event_id]} for event_id in event_ids
        ],
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_status": "SCOPED_CATALOGUES_CHECKED",
            "catalogue_timing": records,
        },
    }
    row = {"source_id": "1", "signals": [signal]}
    policy = {
        "status": "RELEASED",
        "assessment_rules": {"identity_catalogues": identity_catalogue_policy()},
    }
    result = disposition.identity_catalogue_assessment(row, signal, 0, policy)
    assert result["status"] == "MATERIAL_CONFLICT"

    signal["period_days"] = 5.5
    result = disposition.identity_catalogue_assessment(row, signal, 0, policy)
    assert result["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"


def test_ephemeris_match_retains_catalogue_source_identity():
    evidence = importlib.import_module("vetting_evidence")
    event = {"centre_bkjd": 170.0, "duration_days": 0.2}
    match = {
        "catalogue": "cumulative-rows.json",
        "separation_arcsec": 12.0,
        "exact_identifier_match": True,
        "record": {
            "kepoi_name": "K01234.01",
            "koi_pdisposition": "FALSE POSITIVE",
            "koi_period": 5.0,
            "koi_time0bk": 170.0,
            "koi_period_err1": 0.001,
            "koi_time0bk_err1": 0.001,
        },
    }
    result = evidence.ephemeris_match(event, match)
    assert result["source"] == "kepoi_name:K01234.01"
    assert result["signal_class"] == "FALSE_POSITIVE"
    assert result["exact_identifier_match"] is True
    assert result["overlaps_original_interval_nominally"] is True


def test_catalogue_enrichment_processes_every_source(tmp_path, monkeypatch):
    evidence = importlib.import_module("vetting_evidence")
    config = tmp_path / "config.json"
    adapter.save(config, {"queue": "queue.json", "workers": 2})
    adapter.save(tmp_path / "metadata.json", {})
    monkeypatch.setattr(evidence, "CONFIG", config)
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "RUN", tmp_path)
    monkeypatch.setattr(evidence, "check_run", lambda path: None)
    monkeypatch.setattr(
        evidence,
        "promoted_cohort",
        lambda: [{"source_id": sid} for sid in ["1", "2", "3"]],
    )
    monkeypatch.setattr(evidence, "exact_queries", lambda cohort, metadata: None)
    monkeypatch.setattr(evidence, "scenes", lambda cohort, metadata: None)
    monkeypatch.setattr(
        evidence,
        "scene_queries_for_sources",
        lambda cohort: {proposal["source_id"]: [] for proposal in cohort},
    )
    monkeypatch.setattr(evidence, "extra_catalogues", list)
    monkeypatch.setattr(evidence, "shared_alerts", lambda: ([], np.array([]), False))
    monkeypatch.setattr(evidence, "log_worker", lambda result: None)

    class EvidencePool:
        def __init__(self, max_workers, initializer, initargs):
            assert max_workers == 2
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, program, proposals):
            return map(program, proposals)

    monkeypatch.setattr(evidence, "ProcessPoolExecutor", EvidencePool)
    prefetched = []
    processed = []
    monkeypatch.setattr(
        evidence,
        "prefetch_stellar",
        lambda cohort: prefetched.extend(proposal["source_id"] for proposal in cohort),
    )
    monkeypatch.setattr(
        evidence,
        "enrich_source",
        lambda proposal, *args: processed.append(proposal["source_id"]),
    )
    evidence.run()
    assert sorted(prefetched) == ["1", "2", "3"]
    assert sorted(processed) == ["1", "2", "3"]


def test_completeness_requires_all_evidence_stages():
    vetting = importlib.import_module("vetting")
    row = {
        "events": [{"fit": {"status": "FIT_COMPLETE"}}],
        "localization": {"events": [{"status": "TPF_UNAVAILABLE"}]},
        "signals": [
            {
                "contamination": {"status": "EVIDENCE_ASSEMBLED"},
                "host": {"status": "INTERPRETED_AVAILABLE_EVIDENCE"},
            }
        ],
    }
    assert vetting.processing_status(row) == "PARTIAL"
    for name in ("association", "orbital", "instrument", "gaia_scene", "literature"):
        row[name] = {"status": "MEASURED"}
    assert vetting.processing_status(row) == "EVIDENCE_STAGES_COMPLETE"
    del row["signals"][0]["host"]
    assert vetting.processing_status(row) == "PARTIAL"


def test_vetting_stage_timing_records_completion(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    assert vetting.timed_stage("probe", lambda: "complete") == "complete"
    record = adapter.read(tmp_path / "stage-timings.json")["stages"][0]
    assert record["stage"] == "probe"
    assert record["status"] == "COMPLETE"
    assert record["wall_seconds"] >= 0
    assert record["cpu_seconds"] >= 0


def test_vetting_stage_timing_records_failure(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    monkeypatch.setattr(vetting, "RUN", tmp_path)

    def fail():
        raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        vetting.timed_stage("probe", fail)
    record = adapter.read(tmp_path / "stage-timings.json")["stages"][0]
    assert record["status"] == "FAILED"
    assert record["error"] == "RuntimeError: expected"


def replay_outcome_case(label="UNRESOLVED"):
    return {
        "status": "COMPLETE",
        "alert_ledger": {
            "emitted_alerts": 2,
            "adjudicated_alerts": 2,
            "exact_identity_match": True,
            "boi_signals": int(label == "BOI"),
            "declared_signal_ids": ["matched", "other-alias"],
            "signal_dispositions": [
                {"signal_id": "matched", "disposition": "FALSE_ALARM"},
                {"signal_id": "other-alias", "disposition": label},
            ],
        },
    }


@pytest.mark.parametrize("label", ["UNRESOLVED", "FALSE_ALARM", "BOI"])
def test_release_case_separates_completed_abstention_and_false_boi(label):
    vetting = importlib.import_module("vetting")
    case = replay_outcome_case(label)
    measured = vetting.replay_case_ledger(case)
    assert measured["complete"] is True
    assert measured["scientific_abstention"] is (label == "UNRESOLVED")
    assert vetting.replay_case_false_boi(case) is (label == "BOI")


@pytest.mark.parametrize("defect", ["lost_alias", "label", "identity", "job", "count"])
def test_release_case_execution_failures_are_not_false_bois(defect):
    vetting = importlib.import_module("vetting")
    case = replay_outcome_case()
    ledger = case["alert_ledger"]
    if defect == "lost_alias":
        ledger["signal_dispositions"].pop()
    elif defect == "label":
        ledger["signal_dispositions"][1]["disposition"] = "UNAVAILABLE"
    elif defect == "identity":
        ledger["exact_identity_match"] = False
    elif defect == "job":
        case["status"] = "FAILED"
    else:
        ledger["adjudicated_alerts"] = 1
    assert vetting.replay_case_ledger(case)["complete"] is False
    assert vetting.replay_case_false_boi(case) is False


@pytest.mark.parametrize("count", [None, True, -1, "0"])
def test_release_case_rejects_invalid_counts(count):
    vetting = importlib.import_module("vetting")
    case = replay_outcome_case()
    case["alert_ledger"]["boi_signals"] = count
    with pytest.raises(ValueError, match="counts are missing or invalid"):
        vetting.replay_case_ledger(case)


def add_exact_event_evidence(case):
    from deterministic.planet_control_report import match_event_units

    case["truth"] = {"duration_hours": 2.4}
    case["truth_events"] = [{
        "event_id": f"truth-{index}", "centre_bkjd": centre,
        "support": {"1": {"psf_flat": 3}},
    } for index, centre in enumerate((10.0, 20.0))]
    case["signals"] = []
    for index, assigned in enumerate(case["alert_ledger"]["signal_dispositions"]):
        units = [{"unit_id": f"unit-{index}", "quarter": 1,
                  "centre_bkjd": 10.0 + 10 * index}]
        case["signals"].append({
            "signal_id": assigned["signal_id"], "fitted_units": units,
            "fitted_event_match": match_event_units(case["truth_events"], units, 2.4),
            "proposal_units": [dict(unit) for unit in units],
            "proposal_event_match": match_event_units(case["truth_events"], units, 2.4),
            "downstream": {"automatic_disposition": assigned["disposition"], "triage": {"status": "ACTIVE"}},
        })


def test_release_planet_recovery_recomputes_physical_event_membership():
    vetting = importlib.import_module("vetting")
    from deterministic.planet_control_report import match_event_units

    case = replay_outcome_case("BOI")
    add_exact_event_evidence(case)
    assert vetting.replay_case_planet_recovered(case) is True
    signal = case["signals"][1]
    signal["fitted_units"][0]["centre_bkjd"] += 0.5
    with pytest.raises(ValueError, match="exact-event recovery was misreported"):
        vetting.replay_case_planet_recovered(case)
    signal["fitted_event_match"] = match_event_units(
        case["truth_events"], signal["fitted_units"], 2.4
    )
    case["recovery"] = {"extraction_trials": [{"pc_recovered": True}]}
    assert vetting.replay_case_planet_recovered(case) is False


def test_release_planet_recovery_refuses_legacy_recovery_flags():
    vetting = importlib.import_module("vetting")
    case = replay_outcome_case("BOI")
    case["recovery"] = {"extraction_trials": [{"pc_recovered": True}]}
    with pytest.raises(ValueError, match="exact physical-event evidence"):
        vetting.replay_case_planet_recovered(case)


def lane_policy_fixture(tmp_path):
    from deterministic.signal_contracts import LANE_CONTRACT
    thresholds = {name: 8.0 for name in ("periodic", "shallow", "sparse")}
    path = tmp_path / "test-lane-calibration.json"
    adapter.save(path, {"status": "CALIBRATED", "measurement_scope": "OPENED_DEVELOPMENT_NULLS",
                        "lane_contract": LANE_CONTRACT, "thresholds": thresholds})
    return {"contract": LANE_CONTRACT, "status": "CALIBRATED", "thresholds": thresholds,
            "calibration": {"path": str(path), "sha256": adapter.digest(path)}}


def final_release_cases():
    cases = []
    for truth_class, cohort, count in (
        ("planet", "periodic_planet", 300),
        ("eclipsing_binary", "native_eb", 450),
        ("background_contaminant", "background_contaminant", 450),
    ):
        for index in range(count):
            case = replay_outcome_case(
                "BOI" if truth_class == "planet" else "UNRESOLVED"
            )
            identity = f"final-{truth_class}-{index}"
            case.update(
                truth_class=truth_class,
                parent_tpf=identity,
                input_sha256=identity,
                source_ids=[identity],
                raw_fits_sha256=[identity],
                component_ids=[identity],
                truth_cohort=cohort,
            )
            if truth_class == "planet":
                add_exact_event_evidence(case)
            cases.append(case)
    return cases


def opened_cap_cases():
    cases = []
    for cohort, count in {
        "native_eb": 100, "ordinary": 496, "background_contaminant": 100,
        "contamination_pair": 100, "guide_star_family": 100,
    }.items():
        for index in range(count):
            identity = f"development-{cohort}-{index}"
            cases.append({
                **replay_outcome_case("UNRESOLVED"),
                "truth_cohort": cohort, "truth_class": cohort,
                "parent_tpf": identity, "input_sha256": identity,
                "source_ids": [identity], "raw_fits_sha256": [identity],
                "component_ids": [identity],
            })
    variants = []
    for variant in ("inverted", "quarter_scrambled"):
        ids = [f"proposal-{index}" for index in range(400)]
        variants.append({
            "variant": variant, "status": "COMPLETE_NULL_FLUX_LEDGER",
            "declared_proposal_ids": ids, "emitted_alerts": 400,
            "adjudicated_alerts": 400, "exact_identity_match": True,
            "proposals": [{"proposal_id": identity,
                           "exclusion": {"status": "ELIGIBLE"},
                           "gate_path": [{"stage": "shape", "decision": "DISMISS"}]}
                          for identity in ids],
        })
    identity = "development-null"
    nulls = [{"truth_cohort": "null", "source_id": identity,
              "parent_tpf": identity, "input_sha256": identity,
              "source_ids": [identity], "raw_fits_sha256": [identity],
              "component_ids": [identity], "usable_quarters": 3,
              "scientific_abstention": True, "variants": variants}]
    return cases, nulls


def bound_outcome_release(tmp_path):
    vetting = importlib.import_module("vetting")
    spec = {
        "status": "SEALED",
        "outcome_contract": vetting.OUTCOME_CONTRACT,
        "event_matching_contract": vetting.EVENT_MATCHING_CONTRACT,
        "final_audit": vetting.release_audit_spec(adapter.read(ROOT / "pipeline/vetting/disposition-policy.json")),
    }
    documents = {
        "run_spec": spec,
        "detector": {"synthetic_test_artifact": True},
        "disposition_calibration": {"status": "CALIBRATED"},
    }
    references = {}
    for name, value in documents.items():
        path = tmp_path / f"{name}.json"
        adapter.save(path, value)
        references[name] = {"path": str(path), "sha256": vetting.digest(path)}
    identities = {
        "run_spec_sha256": references["run_spec"]["sha256"],
        "detector_sha256": references["detector"]["sha256"],
        "calibration_sha256": references["disposition_calibration"]["sha256"],
    }
    documents["disposition_calibration"]["detector_sha256"] = identities[
        "detector_sha256"
    ]
    path = tmp_path / "disposition_calibration.json"
    adapter.save(path, documents["disposition_calibration"])
    references["disposition_calibration"]["sha256"] = vetting.digest(path)
    identities["calibration_sha256"] = vetting.digest(path)
    policy = {
        "schema": 3,
        "triage_contract": adapter.read(
            ROOT / "pipeline/vetting/disposition-policy.json"
        )["triage_contract"],
        "policy_id": "deterministic-boi-v1",
        "status": "RELEASED",
        "lane_gates": lane_policy_fixture(tmp_path),
        "boi_catalogue_enabled": True,
        "assessment_rules": adapter.read(ROOT / "pipeline/vetting/config.json")[
            "triage"
        ]["assessment_rules"],
        "outcome_contract": vetting.OUTCOME_CONTRACT,
        "event_matching_contract": vetting.EVENT_MATCHING_CONTRACT,
        "release_plan": adapter.read(ROOT / "pipeline/vetting/disposition-policy.json")[
            "release_plan"
        ],
        "implementation": adapter.implementation_hashes(),
        "evaluation_binding": {
            **identities,
            "development_replay": {"controls_sha256": "d" * 64},
            "final_replay": {"controls_sha256": "f" * 64},
        },
    }
    config = adapter.read(ROOT / "pipeline/vetting/config.json")
    config.update(quality_bitmask=-1, discovery_contract=adapter.detector_contract())
    policy["pipeline_contract"] = vetting.pipeline_contract(config)
    for role in ("development_replay", "final_replay"):
        path = tmp_path / f"{role}-cohort.json"
        adapter.save(path, {"schema": 1, "role": role})
        policy["evaluation_binding"][role]["cohorts"] = [
            {"path": str(path), "sha256": vetting.digest(path)}
        ]
    opened_cases, opened_nulls = opened_cap_cases()
    audit = policy["release_plan"]["final_audit"]
    cap_spec = tmp_path / "calibration-spec.json"
    adapter.save(cap_spec, {"status": "BOUND_OPENED_TEST_SPEC",
                            "false_boi_caps": audit["opened_false_boi_caps"],
                            "confidence": audit["confidence"]})
    documents["disposition_calibration"].update(
        spec={"path": str(cap_spec), "sha256": vetting.digest(cap_spec)},
        outcomes={"feasible": True, "incomplete_cases": 0,
                  "false_boi": vetting.calibration_cap_outcomes(
                      {"cases": opened_cases, "null_cases": opened_nulls},
                      audit["opened_false_boi_caps"], audit["confidence"])},
    )
    evidence_path = tmp_path / "opened-development-evidence.json"
    adapter.save(
        evidence_path,
        {
            "status": "OPENED_REPORT_ASSEMBLED",
            "replay_role": "development_replay",
            "outcome_contract": vetting.OUTCOME_CONTRACT,
            "event_matching_contract": vetting.EVENT_MATCHING_CONTRACT,
            "cases": opened_cases,
            "null_cases": opened_nulls,
        },
    )
    documents["disposition_calibration"].update(
        parameters=vetting.disposition_calibration_parameters(policy),
        development_evidence={
            "path": str(evidence_path),
            "sha256": vetting.digest(evidence_path),
        },
    )
    adapter.save(
        tmp_path / "disposition_calibration.json", documents["disposition_calibration"]
    )
    references["disposition_calibration"]["sha256"] = vetting.digest(
        tmp_path / "disposition_calibration.json"
    )
    spec.update(
        detector=references["detector"],
        disposition_calibration=references["disposition_calibration"],
        implementation=policy["implementation"],
        pipeline_contract=policy["pipeline_contract"],
        cohorts={
            role: policy["evaluation_binding"][role]
            for role in ("development_replay", "final_replay")
        },
    )
    adapter.save(tmp_path / "run_spec.json", spec)
    references["run_spec"]["sha256"] = vetting.digest(tmp_path / "run_spec.json")
    identities.update(
        run_spec_sha256=references["run_spec"]["sha256"],
        calibration_sha256=references["disposition_calibration"]["sha256"],
    )
    policy["evaluation_binding"].update(identities)
    policy["definition_sha256"] = vetting.disposition_definition_sha256(policy)
    final_cases = final_release_cases()
    documents["detector_final_audit"] = {
        "status": "DETECTOR_AUDIT_PASSED_DOWNSTREAM_AUDIT_REQUIRED",
        "checks": {"test": True},
        **identities,
    }
    morphology_cases = []
    for cohort in ("sparse_injection", "ttv_injection"):
        for index in range(160):
            case = replay_outcome_case()
            identity = f"final-{cohort}-{index}"
            case.update(
                truth_class="planet",
                truth_cohort=cohort,
                parent_tpf=identity,
                input_sha256=identity,
                source_ids=[identity],
                raw_fits_sha256=[identity],
                component_ids=[identity],
            )
            add_exact_event_evidence(case)
            if cohort == "sparse_injection":
                from deterministic.release_report import single_quarter_assessment
                case["single_quarter_assessment"] = single_quarter_assessment(case)
            morphology_cases.append(case)
    for role, cases in (
        (
            "development_replay",
            opened_cases,
        ),
        ("final_replay", final_cases),
    ):
        documents[role] = {
            "status": "REPLAY_PASSED",
            "replay_role": role,
            "boi_catalogue_enabled": True,
            "detector_binaries": [references["detector"]],
            "outcome_contract": vetting.OUTCOME_CONTRACT,
            "event_matching_contract": vetting.EVENT_MATCHING_CONTRACT,
            **identities,
            "controls_sha256": policy["evaluation_binding"][role]["controls_sha256"],
            "disposition_definition_sha256": policy["definition_sha256"],
            "implementation": policy["implementation"],
            "population": {
                "status": "MEASURED",
                "declared_cases": len(cases)
                + (len(morphology_cases) if role == "final_replay" else 0),
                "overlap_with_detector_development": {"source_ids": {"count": 0}},
            },
            "cases": cases,
            "morphology_cases": morphology_cases if role == "final_replay" else [],
        }
    documents["development_replay"]["null_cases"] = opened_nulls
    documents["development_replay"]["population"]["declared_cases"] += len(opened_nulls)
    null_cases = []
    for index in range(10000):
        identity = f"final-null-{index}"
        variants = [{"variant": variant, "status": "COMPLETE_NULL_FLUX_LEDGER",
                     "declared_proposal_ids": ["proposal"], "emitted_alerts": 1,
                     "adjudicated_alerts": 1, "exact_identity_match": True,
                     "proposals": [{"proposal_id": "proposal", "exclusion": {"status": "ELIGIBLE"},
                                    "gate_path": [{"stage": "shape", "decision": "DISMISS"}]}]}
                    for variant in ("inverted", "quarter_scrambled")]
        null_cases.append({"truth_cohort": "null", "source_id": identity,
                           "parent_tpf": identity, "input_sha256": identity,
                           "source_ids": [identity], "raw_fits_sha256": [identity],
                           "component_ids": [identity], "usable_quarters": 3,
                           "scientific_abstention": True, "variants": variants})
    documents["final_replay"]["null_cases"] = null_cases
    documents["final_replay"]["population"]["declared_cases"] += len(null_cases)
    from deterministic.release_report import release_checks
    checks, outcomes = release_checks(documents["final_replay"], spec)
    documents["final_replay"]["release_checks"] = checks
    documents["final_replay"]["outcomes"] = outcomes
    policy["release_checks"] = checks
    for name, value in documents.items():
        path = tmp_path / f"{name}.json"
        adapter.save(path, value)
        references[name] = {"path": str(path), "sha256": vetting.digest(path)}
    path = tmp_path / "detector_final_audit.json"
    adapter.save(
        path,
        vetting.detector_replay_audit(
            references, documents["development_replay"], documents["final_replay"]
        ),
    )
    references["detector_final_audit"]["sha256"] = vetting.digest(path)
    policy["release_binding"] = references
    return policy


def test_sealed_publication_uses_raw_fractions_with_bounds_reported_separately(tmp_path):
    from deterministic.release_report import release_checks
    policy = bound_outcome_release(tmp_path)
    report = adapter.read(policy["release_binding"]["final_replay"]["path"])
    spec = adapter.read(policy["release_binding"]["run_spec"]["path"])
    for case in report["cases"][90:300]:
        case["alert_ledger"]["signal_dispositions"][1]["disposition"] = "UNRESOLVED"
        case["alert_ledger"]["boi_signals"] = 0
        case["signals"][1]["downstream"]["automatic_disposition"] = "UNRESOLVED"
    for case in report["null_cases"][:200]:
        for variant in case["variants"]:
            variant["proposals"][0]["gate_path"] = [
                {"stage": stage, "decision": "PASS"} for stage in ("shape", "orbital", "instrument")]
    false = report["cases"][300]
    false["alert_ledger"]["signal_dispositions"][1]["disposition"] = "BOI"
    false["alert_ledger"]["boi_signals"] = 1
    checks, outcomes = release_checks(report, spec)
    assert checks["planet_recovery"]["measured_fraction"] == 0.3
    assert checks["planet_recovery"]["lower_bound"] < 0.3
    assert checks["null_false_candidates"]["measured_fraction"] == 0.02
    assert checks["null_false_candidates"]["upper_bound"] > 0.02
    assert all(row["status"] == "PASSED" for row in checks.values())
    assert outcomes["eclipsing_binary"]["outcomes"] == 1
    variant = report["null_cases"][200]["variants"][0]
    variant["proposals"][0]["gate_path"] = [
        {"stage": stage, "decision": "PASS"} for stage in ("shape", "orbital", "instrument")]
    checks, _ = release_checks(report, spec)
    assert checks["null_false_candidates"]["status"] == "FAILED"
    case = report["cases"][89]
    case["alert_ledger"]["signal_dispositions"][1]["disposition"] = "UNRESOLVED"
    case["alert_ledger"]["boi_signals"] = 0
    case["signals"][1]["downstream"]["automatic_disposition"] = "UNRESOLVED"
    checks, _ = release_checks(report, spec)
    assert checks["planet_recovery"]["status"] == "FAILED"


def refresh_bound_detector_audit(policy):
    vetting = importlib.import_module("vetting")
    binding = policy["release_binding"]
    report = vetting.detector_replay_audit(
        binding,
        adapter.read(binding["development_replay"]["path"]),
        adapter.read(binding["final_replay"]["path"]),
    )
    path = Path(binding["detector_final_audit"]["path"])
    adapter.save(path, report)
    binding["detector_final_audit"]["sha256"] = vetting.digest(path)


def test_bound_release_counts_scientific_abstentions_without_false_bois(tmp_path):
    vetting = importlib.import_module("vetting")
    policy = bound_outcome_release(tmp_path)
    assert policy["release_checks"]["complete_outcome_ledger"]["scientific_abstention_cases"] == 11220
    vetting.validate_release_contract(policy)


def test_release_reports_complete_execution_with_zero_scientific_false_class_coverage(tmp_path):
    policy = bound_outcome_release(tmp_path)
    final = adapter.read(policy["release_binding"]["final_replay"]["path"])
    for name in ("eclipsing_binary", "background_contaminant"):
        assert final["outcomes"][name]["outcomes"] == 0
        coverage = final["outcomes"][name]["coverage"]
        assert coverage["declared_cases"] == coverage["completed_cases"] == 450
        assert coverage["execution_coverage"] == 1
        assert coverage["scientific_abstention_cases"] == 450
        assert coverage["fully_disposed_proposed_fraction"] == 0
    assert final["outcomes"]["planet"]["coverage"]["scientific_abstention_cases"] == 0
    assert final["outcomes"]["planet"]["coverage"]["fully_disposed_proposed_fraction"] == 1


def test_coverage_keeps_nonproposals_and_failed_execution_separate():
    vetting = importlib.import_module("vetting")
    unresolved = replay_outcome_case("UNRESOLVED")
    failed = replay_outcome_case("UNRESOLVED")
    failed["status"] = "FAILED"
    nonproposal = {
        "status": "COMPLETE", "alert_ledger": {
            "emitted_alerts": 0, "adjudicated_alerts": 0,
            "boi_signals": 0, "declared_signal_ids": [],
            "signal_dispositions": [], "exact_identity_match": True,
        },
    }
    coverage = vetting.replay_coverage([unresolved, failed, nonproposal])
    assert coverage["declared_cases"] == 3
    assert coverage["completed_cases"] == 2
    assert coverage["execution_coverage"] == 2 / 3
    assert coverage["proposed_cases"] == 2
    assert coverage["nonproposed_cases"] == 1
    assert coverage["scientific_abstention_cases"] == 1
    assert coverage["fully_disposed_proposed_cases"] == 0
    assert vetting.replay_coverage([nonproposal])["fully_disposed_proposed_fraction"] is None


def test_abstention_coverage_can_overlap_a_parent_emitting_an_boi_catalogue():
    vetting = importlib.import_module("vetting")
    case = replay_outcome_case("BOI")
    case["alert_ledger"]["signal_dispositions"][0]["disposition"] = "UNRESOLVED"
    assert vetting.replay_case_false_boi(case) is True
    coverage = vetting.replay_coverage([case])
    assert coverage["scientific_abstention_cases"] == 1
    assert coverage["fully_disposed_proposed_fraction"] == 0


def test_release_rechecks_nested_source_evidence(tmp_path):
    vetting = importlib.import_module("vetting")
    policy = bound_outcome_release(tmp_path)
    evidence = tmp_path / "retained-source.json"
    evidence.write_text("original")
    reference = policy["release_binding"]["final_replay"]
    path = Path(reference["path"])
    report = adapter.read(path)
    report["inputs"] = {str(evidence): vetting.digest(evidence)}
    adapter.save(path, report)
    reference["sha256"] = vetting.digest(path)
    refresh_bound_detector_audit(policy)
    vetting.validate_release_contract(policy)
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="unavailable or changed"):
        vetting.validate_release_contract(policy)


def test_release_rejects_shared_morphology_and_core_components(tmp_path):
    vetting = importlib.import_module("vetting")
    policy = bound_outcome_release(tmp_path)
    reference = policy["release_binding"]["final_replay"]
    path = Path(reference["path"])
    report = adapter.read(path)
    report["morphology_cases"][0]["component_ids"] = report["cases"][0]["component_ids"]
    adapter.save(path, report)
    reference["sha256"] = vetting.digest(path)
    refresh_bound_detector_audit(policy)
    with pytest.raises(ValueError, match="share or lack"):
        vetting.validate_release_contract(policy)


def test_release_assembly_does_not_measure_bounds_with_a_failed_trial(tmp_path):
    from pipeline.deterministic import release_report
    policy = bound_outcome_release(tmp_path)
    final = adapter.read(policy["release_binding"]["final_replay"]["path"])
    spec = adapter.read(policy["release_binding"]["run_spec"]["path"])
    final["cases"][300]["status"] = "FAILED"
    checks, outcomes = release_report.release_checks(final, spec)
    assert outcomes["ledger"]["failed_cases"] == 1
    assert outcomes["eclipsing_binary"]["trials"] == 450
    assert outcomes["eclipsing_binary"]["coverage"]["completed_cases"] == 449
    assert checks["complete_outcome_ledger"]["status"] == "FAILED"
    for name in ("planet_recovery", "null_false_candidates"):
        assert checks[name]["status"] == "NOT_MEASURED_INCOMPLETE_EXECUTION"
        assert checks[name].get("upper_bound", checks[name].get("lower_bound")) is None


@pytest.mark.parametrize("defect", ["legacy", "lost_alias", "job", "changed_bytes", "event_contract", "missing_sparse", "native_truth", "changed_quota"])
def test_bound_release_rejects_invalid_outcome_artifacts(tmp_path, defect):
    vetting = importlib.import_module("vetting")
    policy = bound_outcome_release(tmp_path)
    reference = policy["release_binding"]["final_replay"]
    path = Path(reference["path"])
    report = adapter.read(path)
    if defect == "legacy":
        del report["outcome_contract"]
    elif defect == "event_contract":
        del report["event_matching_contract"]
    elif defect == "missing_sparse":
        report["morphology_cases"].pop(0)
    elif defect == "native_truth":
        report["cases"][300]["truth_cohort"] = "eclipsing_binary"
    elif defect == "changed_quota":
        policy["release_plan"]["morphology_trials"]["sparse_injection"] = 159
    elif defect == "lost_alias":
        report["cases"][300]["alert_ledger"]["signal_dispositions"].pop()
    elif defect == "job":
        report["cases"][300]["status"] = "FAILED"
    else:
        report["edited"] = True
    adapter.save(path, report)
    if defect != "changed_bytes":
        reference["sha256"] = vetting.digest(path)
    with pytest.raises(ValueError):
        vetting.validate_release_contract(policy)


def test_bound_release_rejects_null_false_candidate_excess(tmp_path):
    vetting = importlib.import_module("vetting")
    policy = bound_outcome_release(tmp_path)
    reference = policy["release_binding"]["final_replay"]
    path = Path(reference["path"])
    report = adapter.read(path)
    for case in report["null_cases"][:201]:
        for variant in case["variants"]:
            variant["proposals"][0]["gate_path"] = [
                {"stage": stage, "decision": "PASS"}
                for stage in ("shape", "orbital", "instrument")
            ]
    adapter.save(path, report)
    reference["sha256"] = vetting.digest(path)
    with pytest.raises(ValueError):
        vetting.validate_release_contract(policy)


def test_frozen_policy_requires_every_automatic_assessment_rule(tmp_path):
    vetting = importlib.import_module("vetting")
    policy = {
        "schema": 3,
        "triage_contract": adapter.read(
            ROOT / "pipeline/vetting/disposition-policy.json"
        )["triage_contract"],
        "policy_id": "deterministic-boi-v1",
        "status": "FROZEN_EVALUATION",
        "lane_gates": lane_policy_fixture(tmp_path),
        "boi_catalogue_enabled": True,
        "assessment_rules": json.loads(
            (ROOT / "pipeline/vetting/config.json").read_text()
        )["triage"]["assessment_rules"],
        "evaluation_binding": {
            "run_spec_sha256": "a" * 64,
            "detector_sha256": "b" * 64,
            "calibration_sha256": "c" * 64,
            "development_replay": {
                "controls_sha256": "d" * 64,
                "cohorts": [{"path": "dump/development-cohort.json", "sha256": "f" * 64}],
            },
            "final_replay": {
                "controls_sha256": "e" * 64,
                "cohorts": [{"path": "dump/final-cohort.json", "sha256": "f" * 64}],
            },
        },
    }
    policy["definition_sha256"] = vetting.disposition_definition_sha256(policy)
    vetting.validate_operational_policy(policy)

    for value in (None, "10", True, float("nan")):
        invalid = json.loads(json.dumps(policy))
        invalid["assessment_rules"]["instrument"]["spatial_null_bic_margin"] = value
        invalid["definition_sha256"] = vetting.disposition_definition_sha256(invalid)
        with pytest.raises(ValueError, match="scientific definition"):
            vetting.validate_operational_policy(invalid)

    invalid = json.loads(json.dumps(policy))
    invalid["assessment_rules"]["orbital"] = {"status": "RELEASED"}
    invalid["definition_sha256"] = vetting.disposition_definition_sha256(invalid)
    with pytest.raises(ValueError, match="scientific definition"):
        vetting.validate_operational_policy(invalid)

    del policy["assessment_rules"]["identity_catalogues"]
    policy["definition_sha256"] = vetting.disposition_definition_sha256(policy)
    with pytest.raises(ValueError, match="scientific definition"):
        vetting.validate_operational_policy(policy)


def test_instrument_assesses_all_active_events_beyond_diagnostic_limit(raw):
    systematics = importlib.import_module("vetting_systematics")
    disposition = importlib.import_module("vetting_disposition")
    triage = importlib.import_module("vetting_triage")
    config = json.loads((ROOT / "pipeline/vetting/config.json").read_text())
    raw["background"] = np.zeros(len(raw["time_bkjd"]))
    events = [
        {
            "event_id": f"event-{index}",
            "photometry": "psf_flat",
            "quarter": 1,
            "centre_bkjd": float(raw["time_bkjd"][10 + 5 * index]),
            "duration_days": 0.04,
            "shared_time": {
                "other_alerted_sources": 0,
                "same_parent_sources": 0,
                "complete": True,
            },
        }
        for index in range(25)
    ]
    signal = {
        "signal_id": "signal-instrument-coverage",
        "event_ids": [event["event_id"] for event in events],
        "localization": {
            "spatial_null": {
                "status": "MEASURED",
                "median_bic_advantage": -20,
            }
        },
        "events": [
            {
                "unit_id": f"unit-{index}",
                "event_ids": [event["event_id"]],
                "centre_bkjd": event["centre_bkjd"],
                "quarter": 1,
            }
            for index, event in enumerate(events)
        ],
        "triage": {"status": "ACTIVE"},
    }
    row = systematics.verify({"events": events, "signals": [signal]}, raw, config)
    records = {record["event_id"]: record for record in row["instrument"]["events"]}

    assert set(records) == set(signal["event_ids"])
    assert row["instrument"]["diagnostic_events"] == 18
    assert all(record["in_event_points"] > 0 for record in records.values())
    assert all(
        bool(record["engineering"])
        == (event_id in row["instrument"]["selected_event_ids"])
        for event_id, record in records.items()
    )
    policy = {
        "status": "FROZEN_EVALUATION",
        "_evaluation_authorized": True,
        "assessment_rules": config["triage"]["assessment_rules"],
    }
    assessed = disposition.instrument_assessment(row, signal, policy)
    assert assessed["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"
    assert all(
        disposition.valid_evidence_ref(reference, row)
        for reference in assessed["evidence_refs"]
    )
    row["instrument"]["events"][0]["shared_time"]["complete"] = False
    assert (
        disposition.instrument_assessment(row, signal, policy)["status"] == "UNASSESSED"
    )
    assert triage.instrument_gate(row, signal, config["triage"])["reason"] == (
        "INSTRUMENT_MEASUREMENTS_AVAILABLE"
    )
    row["instrument"]["events"][0]["shared_time"]["complete"] = True

    signal["localization"]["spatial_null"]["median_bic_advantage"] = 20
    assert (
        disposition.instrument_assessment(row, signal, policy)["status"]
        == "MATERIAL_CONFLICT"
    )
    signal["localization"]["spatial_null"]["status"] = "UNASSESSED"
    assert disposition.instrument_assessment(row, signal, policy)["status"] == (
        "UNASSESSED"
    )
    signal["localization"]["spatial_null"].update(
        status="MEASURED", median_bic_advantage=-20
    )
    row["instrument"]["events"][0]["shared_time"]["other_alerted_sources"] = 1000001
    assert (
        disposition.instrument_assessment(row, signal, policy)["status"]
        == "ASSESSED_NO_MATERIAL_CONFLICT"
    )
    policy["assessment_rules"]["instrument"]["material_quality_flags"] = ["Safe mode"]
    row["instrument"]["events"][0]["flags"] = {"Safe mode": 1}
    assert disposition.instrument_assessment(row, signal, policy)["status"] == (
        "MATERIAL_CONFLICT"
    )


def test_bounded_cohort_shared_alert_scope(tmp_path, monkeypatch):
    import sqlite3

    evidence = importlib.import_module("vetting_evidence")
    config = tmp_path / "config.json"
    cohort = [{"source_id": "1"}, {"source_id": "2"}]
    metadata = {source["source_id"]: {"origin_tpf_kic": "42"} for source in cohort}
    (tmp_path / "cohort.json").write_text(json.dumps(cohort))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(evidence, "CONFIG", config)
    monkeypatch.setattr(evidence, "RUN", tmp_path)
    monkeypatch.setattr(
        evidence,
        "alert_intervals",
        lambda source: [{"start_bkjd": 1.0, "end_bkjd": 2.0}],
    )

    config.write_text(json.dumps({"shared_alert_scope_complete": True}))
    shared, times, complete = evidence.shared_alerts()
    assert complete is True
    assert evidence.coincident_sources(shared, times, 1.2, 1.8, "1", "42") == {
        "2": True
    }

    config.write_text(json.dumps({}))
    assert evidence.shared_alerts()[2] is False

    index = tmp_path / "shared-alerts.sqlite"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE marker (value INTEGER)")
        connection.execute("INSERT INTO marker VALUES (1)")
    config.write_text(
        json.dumps(
            {
                "shared_alert_index": {
                    "path": str(index),
                    "sha256": adapter.digest(index),
                }
            }
        )
    )
    shared, times, complete = evidence.shared_alerts()
    assert complete is True and times is None
    with shared:
        assert shared.execute("SELECT value FROM marker").fetchone() == (1,)

    config.write_text(
        json.dumps({"shared_alert_index": {"path": str(index), "sha256": "wrong"}})
    )
    with pytest.raises(ValueError, match="Shared-alert index changed"):
        evidence.shared_alerts()


class ImmediatePool:
    """A stand-in process pool that runs the program in the test process."""

    scheduled: ClassVar[list] = []

    def __init__(self, max_workers):
        assert max_workers == 2

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, program, source_ids):
        self.scheduled.extend(source_ids)
        return map(program, source_ids)

    def submit(self, program, source_id):
        from concurrent.futures import Future

        self.scheduled.append(source_id)
        future = Future()
        future.set_result(program(source_id))
        return future


def test_orbital_stage_schedules_whole_sources_across_workers(
    tmp_path, monkeypatch, capsys
):
    vetting = importlib.import_module("vetting")
    adapter.save(
        tmp_path / "cohort.json",
        [{"source_id": source_id} for source_id in ("1", "2", "3")],
    )
    for source_id in ("1", "2", "3"):
        adapter.save(tmp_path / "sources" / f"{source_id}.json", {"events": []})
    scheduled = ImmediatePool.scheduled = []
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    monkeypatch.setattr(vetting, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(
        vetting,
        "measure_orbital_source",
        lambda source_id: {"source_id": source_id, "timing": {}, "counters": {}},
    )
    vetting.run_orbital_program({"workers": 2})

    assert scheduled == ["1", "2", "3"]
    assert [
        json.loads(line)["source_id"] for line in capsys.readouterr().out.splitlines()
    ] == [
        "1",
        "2",
        "3",
    ]


def test_program_stages_pool_sources_and_summarize_triage(
    tmp_path, monkeypatch, capsys
):
    vetting = importlib.import_module("vetting")
    triage = importlib.import_module("vetting_triage")
    config = tmp_path / "config.json"
    adapter.save(config, {"queue": str(tmp_path / "queue.json"), "workers": 2})
    adapter.save(tmp_path / "cohort.json", [{"source_id": "1"}, {"source_id": "2"}])
    ImmediatePool.scheduled = []
    monkeypatch.setattr(vetting, "CONFIG", config)
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    monkeypatch.setattr(vetting, "check_run", lambda path: None)
    monkeypatch.setattr(vetting, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(
        vetting,
        "measure_program_source",
        lambda name, apply_triage, sid: {
            "source_id": sid,
            "triage": {"last_stage": name, "state": f"{name}-{sid}"},
            "timing": {"wall_seconds": 0.5},
            "counters": {"pixel_files_opened": 1},
        },
    )
    monkeypatch.setattr(
        triage, "gate_summary", lambda states: {"states": [s["state"] for s in states]}
    )
    before = vetting.counter_snapshot().get("pixel_files_opened", 0)
    vetting.run_program("contamination")
    assert ImmediatePool.scheduled == ["1", "2"]
    assert adapter.read(tmp_path / "triage" / "contamination.json") == {
        "states": ["contamination-1", "contamination-2"]
    }
    assert vetting.counter_snapshot()["pixel_files_opened"] == before + 2
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line.get("source_id") for line in lines] == ["1", "2", None]
    timings = [
        json.loads(line)
        for line in (tmp_path / "source-timings.jsonl").read_text().splitlines()
    ]
    assert timings[0] == {
        "stage": vetting.STAGE["name"],
        "source_id": "1",
        "wall_seconds": 0.5,
        "pixel_files_opened": 1,
    }


def test_rust_scene_includes_unalerted_preparation_shards(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    config = tmp_path / "config.json"
    adapter.save(
        config,
        {
            "queue": str(tmp_path / "queue.json"),
            "discovery_contract": {"origin": "rust_v5_deterministic"},
        },
    )
    adapter.save(
        tmp_path / "preparation.json", {"sources": [{"shard": 0}, {"shard": 1}]}
    )
    monkeypatch.setattr(vetting, "CONFIG", config)

    def catalogue(root, shards, columns):
        assert shards == [0, 1] and columns == ["source_id", *vetting.SCENE_COLUMNS]
        return [
            {"source_id": sid, "ra_deg": 1.0, "dec_deg": 2.0, "origin_tpf_kic": 7,
             "phot_g_mean_mag": 12.5, "shard": 1}
            for sid in ("123", "456")
        ]

    full_row = {"source_id": "123", "shard": 1, "primary_header_cards": ["x"]}
    monkeypatch.setattr(vetting, "source_catalogue", catalogue)
    monkeypatch.setattr(vetting, "observation", lambda sid: (full_row, None))
    metadata, scene = vetting.source_metadata([{"source_id": "123"}])
    assert metadata == {"123": full_row}
    assert [row["source_id"] for row in scene] == ["123", "456"]
    assert set(scene[1]) == {"source_id", *vetting.SCENE_FIELDS}


def test_neighbours_come_from_the_scene_rows(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    config = tmp_path / "config.json"
    adapter.save(config, {"neighbour_radius_arcsec": 60})
    monkeypatch.setattr(vetting, "CONFIG", config)
    scene = [
        {"source_id": sid, "ra_deg": ra, "dec_deg": dec, "origin_tpf_kic": kic,
         "phot_g_mean_mag": mag, "shard": shard}
        for sid, ra, dec, kic, mag, shard in (
            ("1", 290.0, 45.0, 7, 12.0, 3),
            ("2", 290.0, 45.01, 7, 14.0, 3),
            ("3", 290.0, 45.1, 8, 15.0, 5),
        )
    ]
    target = {"source_id": "1", "ra_deg": 290.0, "dec_deg": 45.0, "origin_tpf_kic": 7}
    result = vetting.neighbours(target, scene)
    assert result["status"] == "EXTRACTED_SOURCE_SCENE"
    assert result["shards"] == [3, 5]
    assert [row["source_id"] for row in result["sources"]] == ["2"]
    assert result["sources"][0]["same_parent_tpf"] is True
    assert abs(result["sources"][0]["separation_arcsec"] - 36.0) < 0.01
    scene[2]["dec_deg"] = float("nan")
    with pytest.raises(ValueError, match="finite declinations"):
        vetting.neighbours(target, scene)


def test_manifest_scene_preserves_unalerted_sources_and_input_binding(tmp_path, monkeypatch):
    vetting = importlib.import_module("vetting")
    rows = [
        {"source_id": sid, "ra_deg": 290.0, "dec_deg": 45.0,
         "origin_tpf_kic": 7, "phot_g_mean_mag": 12.0, "shard": shard,
         "raw_fits_sha256": "a" * 64}
        for sid, shard in (("123", 0), ("456", 1), ("789", 0))
    ]
    scene_path = tmp_path / "scene.jsonl"
    scene_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    arrays = tmp_path / "arrays.npz"
    arrays.write_bytes(b"hash-bound arrays")
    manifest = tmp_path / "input-manifest.json"
    adapter.save(manifest, {
        "schema": 1, "measurement_scope": "opened control portability",
        "scene_metadata": {"path": scene_path.name, "sha256": adapter.digest(scene_path)},
        "sources": {"123": {"arrays": arrays.name}},
    })
    config = tmp_path / "config.json"
    adapter.save(config, {"input_manifest": str(manifest), "queue": str(tmp_path / "queue.json")})
    adapter.save(tmp_path / "preparation.json", {"sources": [{"shard": 0}, {"shard": 1}]})
    monkeypatch.setattr(vetting, "CONFIG", config)
    monkeypatch.setattr(vetting, "observation", lambda sid: (rows[0], None))
    monkeypatch.setattr(vetting, "source_catalogue", lambda *args: pytest.fail("Needs local corpus"))
    metadata, scene = vetting.source_metadata([{"source_id": "123"}])
    assert metadata == {"123": rows[0]}
    assert [row["source_id"] for row in scene] == ["123", "456", "789"]
    assert vetting.input_hashes(["123"])[str(scene_path)] == adapter.digest(scene_path)
    scene_path.write_text(scene_path.read_text() + " ")
    with pytest.raises(ValueError, match="scene metadata changed"):
        vetting.source_metadata([{"source_id": "123"}])
    with pytest.raises(ValueError, match="scene metadata changed"):
        vetting.input_hashes(["123"])


@pytest.mark.parametrize("change", ["duplicate", "numeric_id", "missing_field", "coordinates", "parent", "selected", "shards"])
def test_manifest_scene_rejects_invalid_identity_and_context(tmp_path, change):
    vetting = importlib.import_module("vetting")
    source = {"source_id": "123", "ra_deg": 290.0, "dec_deg": 45.0,
              "origin_tpf_kic": 7, "phot_g_mean_mag": 12.0, "shard": 0,
              "raw_fits_sha256": "a" * 64}
    rows = [source.copy()]
    if change == "duplicate":
        rows.append(source.copy())
    elif change == "numeric_id":
        rows[0]["source_id"] = 123
    elif change == "missing_field":
        rows[0].pop("origin_tpf_kic")
    elif change == "coordinates":
        rows[0]["dec_deg"] = 100.0
    elif change == "parent":
        rows[0]["origin_tpf_kic"] = True
    elif change == "selected":
        rows[0]["raw_fits_sha256"] = "b" * 64
    else:
        rows[0]["shard"] = 1
    path = tmp_path / "scene.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reference = {"path": path.name, "sha256": adapter.digest(path)}
    with pytest.raises(ValueError, match="Experimental scene"):
        vetting.manifest_scene(tmp_path / "input-manifest.json", reference, {"123": source}, {0})


def test_manifest_scene_filters_full_pack_to_prepared_shards(tmp_path):
    vetting = importlib.import_module("vetting")
    source = {"source_id": "123", "ra_deg": 290.0, "dec_deg": 45.0,
              "origin_tpf_kic": 7, "phot_g_mean_mag": 12.0, "shard": 0,
              "raw_fits_sha256": "a" * 64}
    sibling = dict(source, source_id="456")
    other = dict(source, source_id="789", shard=1)
    path = tmp_path / "scene.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in (other, source, sibling)))
    reference = {"path": path.name, "sha256": adapter.digest(path)}
    rows = vetting.manifest_scene(tmp_path / "input-manifest.json", reference, {"123": source}, {0})
    assert [sid for sid, _ in rows] == ["123", "456"]


def test_nonzero_program_exit_fails(tmp_path):
    with pytest.raises(RuntimeError, match="exited 7"):
        adapter.run_command(
            [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failed.log"
        )


def test_survey_batch_uses_frozen_config(tmp_path, monkeypatch):
    config = tmp_path / "detector-config.json"
    shallow_config = tmp_path / "shallow-config.json"
    binary = tmp_path / "detector"
    config.write_text("{}")
    shallow_config.write_text("{}")
    binary.write_bytes(b"binary")
    (tmp_path / "inputs").mkdir()

    def run(command, log, environment):
        assert command[3] == str(config)
        assert environment["RAYON_NUM_THREADS"] == "7"

    monkeypatch.setattr(adapter, "run_command", run)
    monkeypatch.setattr(
        adapter,
        "parse_rows",
        lambda path, views: {
            ("456", "psf_flat"): [dict(rust_row(), source_id="456")],
            ("123", "psf_flat"): [rust_row()],
        },
    )

    def scan_shallow(input_dir, output, source_ids, config, workers):
        assert workers == 7
        write_shallow_eligibility(tmp_path / "shallow-eligibility.csv", source_ids)
        return {sid: [shallow_row(source_id=sid)] for sid in source_ids}

    monkeypatch.setattr(adapter, "scan_shallow", scan_shallow)
    batch, rows, shallow = adapter.run_survey_batch(
        tmp_path,
        binary,
        config,
        shallow_config,
        [{"source_id": "456"}, {"source_id": "123"}],
        7,
        3,
    )
    assert batch["batch"] == 3
    assert batch["sources"] == 2
    assert [row["source_id"] for row in rows] == ["123", "456"]
    assert [row["source_id"] for row in shallow] == ["123", "456"]


@pytest.mark.parametrize(("eligible", "batch_size"), [({"1", "3"}, 3), ({"1", "2"}, 2)])
def test_full_survey_accounts_for_ineligible_and_final_partial_batch(
    tmp_path, monkeypatch, eligible, batch_size
):
    sources = [{"source_id": str(number), "shard": 1} for number in range(1, 5)]
    monkeypatch.setattr(adapter, "select_sources", lambda *args, **kwargs: sources)
    monkeypatch.setattr(adapter, "dump_path", lambda path: path)

    def start(out, args):
        out.mkdir()
        binary = out / "detector"
        config = out / "detector-config.json"
        shallow_config = out / "shallow-config.json"
        binary.write_bytes(b"detector")
        config.write_text("{}")
        shallow_config.write_text("{}")
        report = {
            "schema": 7,
        "disposition_policy": str(adapter.DISPOSITION_POLICY),
        "disposition_policy_sha256": adapter.digest(ROOT / adapter.DISPOSITION_POLICY),
            "status": "RUNNING",
            "requested_sources": args.limit,
            "attempted_sources": 0,
            "accepted_sources": 0,
            "ineligible_sources": 0,
            "proposal_rows": 0,
            "proposal_sources": 0,
            "followup_diagnostic_rows": 0,
        "followup_diagnostic_sources": 0,
        "survivor_review_rows": 0,
            "survivor_review_sources": 0,
            "shallow_proposal_rows": 0,
            "shallow_proposal_sources": 0,
            "shallow_ineligible_sources": 0,
            "classifications": {},
            "batches": [],
            "committed_bytes": {},
        }
        report["detector_contract"] = adapter.detector_contract()
        adapter.save(out / "survey.json", report)
        return report, binary, config, shallow_config, Counter(), set(), set(), set(), 0

    def prepare(out, source, input_manifest):
        if source["source_id"] not in eligible:
            raise adapter.InsufficientCadences(20)
        return {
            "source_id": source["source_id"],
            "shard": 1,
            "parent_tpf": "123",
            "raw_fits_sha256": "a" * 64,
            "input_arrays_sha256": "b" * 64,
            "flatten_mask_sha256": "c" * 64,
            "retained_cadences": 100,
            "fsc_sha256": "d" * 64,
        }

    def scan(out, binary, config, shallow_config, entries, threads, batch_number):
        rows = [dict(rust_row(), source_id=entry["source_id"]) for entry in entries]
        csv_file(out / "raw.csv", rows)
        write_shallow_eligibility(
            out / "shallow-eligibility.csv", [entry["source_id"] for entry in entries]
        )
        return (
            {
                "batch": batch_number,
                "sources": len(entries),
                "rows": len(rows),
                "shallow_rows": 0,
                "shallow_ineligible_sources": 0,
                "seconds": 1.0,
                "classifications": {"TRANSIT_LIKE": len(rows)},
            },
            rows,
            [],
        )

    monkeypatch.setattr(adapter, "start_survey", start)
    monkeypatch.setattr(adapter, "prepare_discovery_source", prepare)
    monkeypatch.setattr(adapter, "run_survey_batch", scan)
    out = tmp_path / "survey"
    adapter.survey(
        SimpleNamespace(
            run=out,
            resume=False,
            source_ids=None,
            input_manifest=None,
            shards=[1],
            limit=4,
            batch_size=batch_size,
            threads=1,
        )
    )
    report = adapter.read(out / "survey.json")
    assert report["status"] == "COMPLETE"
    assert (
        report["attempted_sources"],
        report["accepted_sources"],
        report["ineligible_sources"],
    ) == (4, 2, 2)
    assert [
        row["source_id"] for row in adapter.read_survey_rows(out / "cohort.csv")
    ] == sorted(eligible)
    assert [
        row["source_id"] for row in adapter.read_survey_rows(out / "ineligible.csv")
    ] == sorted({"1", "2", "3", "4"} - eligible)
    assert report["batches"][0]["sources"] == 2
    assert report["batches"][-1]["cumulative_attempted_sources"] == 4
    assert report["artifacts"]["ineligible.csv"] == adapter.digest(
        out / "ineligible.csv"
    )


def test_survey_selection_binds_explicit_experimental_inputs(tmp_path):
    source_ids = tmp_path / "source-ids.json"
    manifest = tmp_path / "input-manifest.json"
    source_ids.write_text('["123"]')
    manifest.write_text('{"schema":2}')
    selection = adapter.survey_selection(
        SimpleNamespace(
            shards=[2, 1],
            source_ids=source_ids,
            input_manifest=manifest,
        )
    )
    assert selection["ordering"] == "explicit supplied identities"
    assert selection["execution_order"] == "shard then numeric Gaia source ID"
    assert selection["source_ids"]["sha256"] == adapter.digest(source_ids)
    assert selection["input_manifest"]["sha256"] == adapter.digest(manifest)


def test_explicit_survey_execution_is_shard_ordered():
    sources = [
        {"source_id": "9", "shard": 2},
        {"source_id": "10", "shard": 1},
        {"source_id": "2", "shard": 1},
    ]
    ordered = adapter.ordered_survey_sources(sources, explicit=True)
    assert [(row["shard"], row["source_id"]) for row in ordered] == [
        (1, "2"),
        (1, "10"),
        (2, "9"),
    ]
    assert adapter.ordered_survey_sources(sources, explicit=False) is sources


def test_survey_resume_reconciles_and_rejects_changed_files(tmp_path, monkeypatch):
    contract = {"protocol_version": "test"}
    implementation = {"pipeline/deterministic/run.py": "test"}
    binary = tmp_path / "detector"
    config = tmp_path / "detector-config.json"
    shallow_config = tmp_path / "shallow-config.json"
    binary.write_bytes(b"binary")
    config.write_text("{}")
    shallow_config.write_text("{}")
    with (tmp_path / "cohort.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id"])
        writer.writeheader()
        writer.writerow({"source_id": "123"})
    with (tmp_path / "ineligible.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id"])
        writer.writeheader()
        writer.writerow({"source_id": "456"})
    csv_file(tmp_path / "proposals.csv", [rust_row()])
    csv_file(tmp_path / "followup-diagnostics.csv", [rust_row()])
    csv_file(tmp_path / "detections.csv", [rust_row()])
    with (tmp_path / "shallow-proposals.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=adapter.shallow_detector.FIELDS).writeheader()
    write_shallow_eligibility(tmp_path / "shallow-eligibility.csv", ["123"])
    args = SimpleNamespace(shards=[2, 1], limit=10, batch_size=1, threads=7)
    report = {
        "schema": 7,
        "disposition_policy": str(adapter.DISPOSITION_POLICY),
        "disposition_policy_sha256": adapter.digest(ROOT / adapter.DISPOSITION_POLICY),
        "status": "RUNNING",
        "detector_contract": contract,
        "implementation": implementation,
        "requested_sources": 10,
        "selection": {
            "shards": [1, 2],
            "ordering": "numeric Gaia source ID",
            "eligibility": "PSF_FLAT has at least 100 prepared cadences before detection",
        },
        "batch_size": 1,
        "threads": 7,
        "attempted_sources": 2,
        "accepted_sources": 1,
        "ineligible_sources": 1,
        "proposal_rows": 1,
        "proposal_sources": 1,
        "followup_diagnostic_rows": 1,
        "followup_diagnostic_sources": 1,
        "survivor_review_rows": 0,
        "survivor_review_sources": 0,
        "shallow_proposal_rows": 0,
        "shallow_proposal_sources": 0,
        "shallow_ineligible_sources": 0,
        "classifications": {"TRANSIT_LIKE": 1},
        "batches": [
            {
                "batch": 1,
                "sources": 1,
                "rows": 1,
                "followup_diagnostic_rows": 1,
                "followup_diagnostic_sources": 1,
                "shallow_rows": 0,
                "shallow_ineligible_sources": 0,
                "seconds": 1.0,
                "classifications": {"TRANSIT_LIKE": 1},
                "cumulative_attempted_sources": 2,
                "cumulative_accepted_sources": 1,
                "cumulative_ineligible_sources": 1,
            }
        ],
        "binary": {"path": str(binary), "sha256": adapter.digest(binary)},
        "config": {"path": str(config), "sha256": adapter.digest(config)},
        "shallow_config": {
            "path": str(shallow_config),
            "sha256": adapter.digest(shallow_config),
        },
        "committed_bytes": {
            name: (tmp_path / name).stat().st_size
            for name in adapter.SURVEY_TABLES
            if (tmp_path / name).exists()
        },
    }
    adapter.save(tmp_path / "survey.json", report)
    monkeypatch.setattr(adapter, "detector_contract", lambda *args: contract)
    monkeypatch.setattr(adapter, "implementation_hashes", lambda: implementation)
    with (tmp_path / "proposals.csv").open("a") as stream:
        stream.write("999,psf_flat,1,PC_")
    (tmp_path / "survivor-review.csv").write_text("uncommitted")

    resumed = adapter.validate_survey_resume(tmp_path, args)
    assert resumed[0] == dict(
        report,
        resumes=[
            {
                "committed_batches": 1,
                "discarded_bytes": {"proposals.csv": 18, "survivor-review.csv": 11},
            }
        ],
    )
    assert resumed[4] == {"TRANSIT_LIKE": 1}
    assert resumed[5] == {"123"}
    assert resumed[6] == set()
    assert resumed[7] == set()

    config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="detector or config changed"):
        adapter.validate_survey_resume(tmp_path, args)
    config.write_text("{}")

    os.truncate(tmp_path / "cohort.csv", 5)
    with pytest.raises(ValueError, match=r"shorter than their committed.*cohort\.csv"):
        adapter.validate_survey_resume(tmp_path, args)

    del report["committed_bytes"]
    adapter.save(tmp_path / "survey.json", report)
    with pytest.raises(ValueError, match="lacks its committed table sizes"):
        adapter.validate_survey_resume(tmp_path, args)


def test_survey_preserves_survivor_review_without_changing_proposal_count(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    fields = [*rust_row(), "duration_consistency_status"]
    with (working / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
    pc = dict(rust_row(), duration_consistency_status="PASS")
    survivor = dict(
        rust_row(classification="SYSTEMATIC_LIKE"),
        source_id="456",
        duration_consistency_status="FAIL",
    )
    report = {
        "attempted_sources": 2,
        "accepted_sources": 2,
        "ineligible_sources": 0,
        "proposal_rows": 0,
        "followup_diagnostic_rows": 0,
        "followup_diagnostic_sources": 0,
        "survivor_review_rows": 0,
        "shallow_proposal_rows": 0,
        "shallow_ineligible_sources": 0,
        "batches": [],
        "committed_bytes": {},
    }
    write_shallow_eligibility(working / "shallow-eligibility.csv", ["123", "456"])
    report["detector_contract"] = adapter.detector_contract()
    adapter.record_survey_batch(
        tmp_path,
        working,
        [pc, survivor],
        [shallow_row(source_id="456")],
        [{"source_id": "123"}, {"source_id": "456"}],
        {
            "sources": 2,
            "rows": 2,
            "shallow_rows": 1,
            "shallow_ineligible_sources": 0,
            "classifications": {},
        },
        report,
        Counter(),
        set(),
        set(),
        set(),
        [],
        0,
    )
    assert report["proposal_rows"] == 1
    assert report["survivor_review_rows"] == 1
    assert report["shallow_proposal_rows"] == 1
    assert adapter.read_survey_rows(tmp_path / "proposals.csv")[0]["source_id"] == "123"
    assert (
        adapter.read_survey_rows(tmp_path / "survivor-review.csv")[0]["source_id"]
        == "456"
    )
    assert report["committed_bytes"] == {
        name: (tmp_path / name).stat().st_size for name in adapter.SURVEY_TABLES
    }
    assert adapter.read(tmp_path / "survey.json") == report


def test_survey_freezes_all_diagnostics_for_shallow_only_and_rust_sources(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    rows = [rust_row(), rust_row(2, "EB_LIKE"),
            dict(rust_row(1, "EB_LIKE"), source_id="456"),
            dict(rust_row(1, "EB_LIKE"), source_id="789")]
    csv_file(working / "raw.csv", rows)
    write_shallow_eligibility(working / "shallow-eligibility.csv", ["123", "456", "789"])
    report = {"attempted_sources": 3, "accepted_sources": 3, "ineligible_sources": 0,
              "proposal_rows": 0, "survivor_review_rows": 0, "shallow_proposal_rows": 0,
              "shallow_ineligible_sources": 0, "followup_diagnostic_rows": 0,
              "batches": [], "committed_bytes": {}}
    report["detector_contract"] = adapter.detector_contract()
    adapter.record_survey_batch(
        tmp_path, working, rows, [shallow_row(source_id="456")],
        [{"source_id": sid} for sid in ("123", "456", "789")],
        {"sources": 3, "rows": 4, "shallow_rows": 1, "shallow_ineligible_sources": 0},
        report, Counter(), set(), set(), set(), [], 0,
    )
    assert adapter.read_survey_rows(tmp_path / "followup-diagnostics.csv") == rows[:3]
    assert adapter.read_survey_rows(tmp_path / "detections.csv") == rows
    assert report["followup_diagnostic_rows"] == 3
    assert report["followup_diagnostic_sources"] == 2
    assert report["batches"][0]["followup_diagnostic_rows"] == 3
    assert report["committed_bytes"]["followup-diagnostics.csv"] > 0


def build_survey_binary(out):
    binary = out / "build/release/forgotten-stars-search"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"detector")
    return binary


def prepare_survey_source(out, source, input_manifest):
    if source["source_id"] in {"3", "8"}:
        raise adapter.InsufficientCadences(20)
    return {
        "source_id": source["source_id"],
        "shard": 1,
        "parent_tpf": "123",
        "raw_fits_sha256": "a" * 64,
        "input_arrays_sha256": "b" * 64,
        "flatten_mask_sha256": "c" * 64,
        "retained_cadences": 100,
        "fsc_sha256": "d" * 64,
    }


@pytest.fixture
def interruptible_survey(monkeypatch):
    sources = [{"source_id": str(number), "shard": 1} for number in range(1, 9)]
    fields = [*rust_row(), "duration_consistency_status"]
    interruption = {}

    def interrupt(stage, number):
        if interruption["point"] == (stage, number):
            interruption["point"] = None
            interruption["action"]()

    def scan(out, binary, config, shallow_config, entries, threads, batch_number):
        interrupt("detect", batch_number)
        ids = [entry["source_id"] for entry in entries]
        rows = [
            dict(
                rust_row(rank, classification),
                source_id=sid,
                duration_consistency_status="FAIL",
            )
            for sid in ids
            for rank, classification in ((1, "TRANSIT_LIKE"), (2, "SYSTEMATIC_LIKE"))
        ]
        with (out / "raw.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_shallow_eligibility(out / "shallow-eligibility.csv", ids)
        return (
            {
                "batch": batch_number,
                "sources": len(ids),
                "rows": len(rows),
                "shallow_rows": len(ids),
                "shallow_ineligible_sources": 0,
                "seconds": 1.0,
                "classifications": {"TRANSIT_LIKE": len(ids), "SYSTEMATIC_LIKE": len(ids)},
            },
            rows,
            [shallow_row(source_id=sid) for sid in ids],
        )

    append = adapter.append_survey_rows
    save = adapter.save

    def append_rows(path, fields, rows):
        size = append(path, fields, rows)
        interruption["appends"] += 1
        interrupt("append", interruption["appends"])
        return size

    def commit(path, value, **options):
        if path.name == "survey.json" and value["status"] == "RUNNING":
            interrupt("commit", len(value["batches"]))
        save(path, value, **options)

    monkeypatch.setattr(adapter, "select_sources", lambda *args, **kwargs: sources)
    monkeypatch.setattr(adapter, "dump_path", lambda path: path)
    monkeypatch.setattr(adapter, "evaluation_preflight", lambda *args: None)
    monkeypatch.setattr(adapter, "verify_frozen_detector", lambda *args: None)
    monkeypatch.setattr(adapter, "build_binary", build_survey_binary)
    monkeypatch.setattr(adapter, "prepare_discovery_source", prepare_survey_source)
    monkeypatch.setattr(adapter, "run_survey_batch", scan)
    monkeypatch.setattr(adapter, "append_survey_rows", append_rows)
    monkeypatch.setattr(adapter, "save", commit)

    def run(out, resume=False, point=None, action=None):
        interruption.update(point=point, action=action, appends=0)
        adapter.survey(
            SimpleNamespace(
                run=out,
                resume=resume,
                source_ids=None,
                input_manifest=None,
                shards=[1],
                limit=8,
                batch_size=2,
                threads=1,
            )
        )

    return run


def survey_outcome(out):
    report = adapter.read(out / "survey.json")
    for name in ("elapsed_seconds", "binary", "config", "shallow_config", "resumes"):
        report.pop(name, None)
    return report, {name: (out / name).read_bytes() for name in adapter.SURVEY_TABLES}


def killed_survey(run, out, point):
    pid = os.fork()
    if pid == 0:
        try:
            run(out, point=point, action=lambda: os._exit(9))
        finally:
            os._exit(1)
    return os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])


@pytest.mark.parametrize(
    ("point", "committed_batches"),
    [
        (("detect", 1), 0),
        (("append", 3), 0),
        (("commit", 1), 0),
        (("detect", 2), 1),
        (("append", 10), 1),
        (("commit", 2), 1),
        (("append", 25), 3),
        (("commit", 4), 3),
    ],
)
def test_survey_resumes_after_a_kill(
    tmp_path, interruptible_survey, point, committed_batches
):
    interruptible_survey(tmp_path / "reference")
    assert killed_survey(interruptible_survey, tmp_path / "killed", point) == 9
    interruptible_survey(tmp_path / "killed", resume=True)
    assert survey_outcome(tmp_path / "killed") == survey_outcome(tmp_path / "reference")
    (resume,) = adapter.read(tmp_path / "killed/survey.json")["resumes"]
    assert resume["committed_batches"] == committed_batches
    assert bool(resume["discarded_bytes"]) == (point[0] != "detect")


def test_survey_resumes_after_a_failed_batch_commit(tmp_path, interruptible_survey):
    def fail():
        raise OSError(28, "No space left on device")

    interruptible_survey(tmp_path / "reference")
    with pytest.raises(OSError, match="No space left"):
        interruptible_survey(tmp_path / "failed", point=("append", 10), action=fail)
    failed = adapter.read(tmp_path / "failed/survey.json")
    assert failed["status"] == "FAILED"
    assert failed["error"] == "OSError: [Errno 28] No space left on device"
    assert len(failed["batches"]) == 1
    interruptible_survey(tmp_path / "failed", resume=True)
    assert survey_outcome(tmp_path / "failed") == survey_outcome(tmp_path / "reference")
    (resume,) = adapter.read(tmp_path / "failed/survey.json")["resumes"]
    assert set(resume["discarded_bytes"]) == {"cohort.csv", "detections.csv"}


def test_vetting_reconciles_shallow_only_survey_source(tmp_path, monkeypatch):
    cohort = [{"source_id": "123"}]
    diagnostics = [dict(rust_row(classification="NO_SIGNAL", period=""), t0_bkjd="", duration_hours="")]
    csv_file(tmp_path / "followup-diagnostics.csv", diagnostics)
    write_shallow_eligibility(tmp_path / "shallow-eligibility.csv", ["123"])
    monkeypatch.setattr(
        batch_vetter,
        "read_survey",
        lambda survey: (
            {
                "detector_contract": adapter.detector_contract(),
                "followup_diagnostic_rows": 1,
                "followup_diagnostic_sources": 1,
            },
            cohort,
            [],
            [],
            [shallow_row()],
        ),
    )
    _, ordered, expected = batch_vetter.followup_rows(tmp_path)
    assert ordered == cohort
    assert expected["123"]["rust"] == []
    assert expected["123"]["shallow"] == [shallow_row()]
    csv_file(
        tmp_path / "raw.csv",
        [
            dict(
                rust_row(classification="NO_SIGNAL", period=""),
                t0_bkjd="",
                duration_hours="",
            )
        ],
    )
    with (tmp_path / "shallow-raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=adapter.shallow_detector.FIELDS)
        writer.writeheader()
        writer.writerow(shallow_row())
    adapter.save(tmp_path / "detection.json", {"alerted_sources": 1})
    assert batch_vetter.reconcile_detector(tmp_path, expected)["alerted_sources"] == 1
    with (tmp_path / "shallow-raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=adapter.shallow_detector.FIELDS)
        writer.writeheader()
    with pytest.raises(ValueError, match="shallow rows differ"):
        batch_vetter.reconcile_detector(tmp_path, expected)


def test_survey_shared_alert_index_sees_other_batches(tmp_path, monkeypatch):
    import sqlite3

    from vetting_evidence import coincident_sources

    survey = tmp_path / "survey"
    survey.mkdir()
    adapter.save(survey / "survey.json", {"status": "COMPLETE"})
    run = tmp_path / "vetting"
    run.mkdir()
    sources = [
        {"source_id": "123", "parent_tpf": "10"},
        {"source_id": "456", "parent_tpf": "10"},
        {"source_id": "789", "parent_tpf": "20"},
    ]
    intervals = {
        "123": (1.0, 2.0),
        "456": (1.5, 2.5),
        "789": (2.0001, 3.0),
    }

    def alerts(source, rows, manifest):
        start, end = intervals[source["source_id"]]
        return [
            {
                "event_id": source["source_id"],
                "kind": "SPARSE_EVENT_WINDOW",
                "photometry": "psf_flat",
                "quarter": 1,
                "start_bkjd": start,
                "end_bkjd": end,
            }
        ]

    monkeypatch.setattr(batch_vetter, "survey_alerts", alerts)
    index = batch_vetter.shared_alert_index(
        run,
        survey,
        {"selection": {}, "detector_contract": adapter.detector_contract()},
        sources,
        {source["source_id"]: {} for source in sources},
    )
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        assert coincident_sources(connection, None, 1.0, 2.0, "123", 10) == {
            "456": True
        }
    assert (
        batch_vetter.shared_alert_index(
            run,
            survey,
            {"selection": {}, "detector_contract": adapter.detector_contract()},
            sources,
            {source["source_id"]: {} for source in sources},
        )
        == index
    )
    batch = tmp_path / "batch"
    (batch / "proposals").mkdir(parents=True)
    adapter.save(
        batch / "queue.json",
        {"rows": [{"source_id": "123", "evidence": "proposals/123.json"}]},
    )
    adapter.save(batch / "source-ledger.json", [{"source_id": "123", "alerts": 1}])
    adapter.save(batch / "detection.json", {"sources": 1, "alerted_sources": 1, "alerts": 1})
    adapter.save(batch / "proposals/123.json", {"alerts": alerts(sources[0], {}, None), "ungated_alerts": alerts(sources[0], {}, None)})
    batch_vetter.reconcile_shared_alerts(batch, index, sources[:1])
    adapter.save(batch / "proposals/123.json", {"alerts": [], "ungated_alerts": []})
    with pytest.raises(ValueError, match="differ from survey-wide index"):
        batch_vetter.reconcile_shared_alerts(batch, index, sources[:1])


def test_vetting_archive_detects_mutated_evidence(tmp_path):
    batch = tmp_path / "batch"
    (batch / "vetting/sources").mkdir(parents=True)
    (batch / "inputs").mkdir()
    cache = batch / "pixel-cache"
    cache.mkdir()
    (batch / "vetting/pixel-manifests").mkdir()
    pixel = cache / "example.fits.gz"
    pixel.write_bytes(b"archived target pixels")
    adapter.save(batch / "vetting.json", {"pixel_cache_directory": str(cache)})
    adapter.save(
        batch / "vetting/pixel-manifests/10.json",
        {
            "quarters": {
                "1": {
                    "status": "AVAILABLE",
                    "path": str(pixel),
                    "sha256": adapter.digest(pixel),
                    "url": "https://archive.stsci.edu/example.fits.gz",
                }
            }
        },
    )
    source = batch / "vetting/sources/123.json"
    source.write_text('{"events":[' + ",".join(["123"] * 1000) + "]}")
    (batch / "vetting/cohort.json").write_text('{"sources":["123"]}')
    (batch / "vetting/metadata.json").write_text('{"123":{"shard":229}}')
    (batch / "vetting/scene.json").write_text('[{"source_id":"123","shard":229}]')
    adapter.save(batch / "vetting/results.json", {"rows": [{"evidence": str(source)}]})
    adapter.save(batch / "vetting/bois.json", [{"evidence": str(source)}])
    adapter.save(batch / "vetting/surviving-signals.json", [{"evidence": str(source)}])
    adapter.save(batch / "vetting/signal-ledger.json", [{"evidence": str(source)}])
    prepared = batch / "inputs/123.psf_flat.fsc"
    prepared.write_bytes(b"prepared")
    adapter.save(
        batch / "preparation.json",
        {
            "sources": [
                {
                    "inputs": {
                        "psf_flat": {
                            "fsc_path": str(prepared),
                            "fsc_sha256": adapter.digest(prepared),
                        }
                    }
                }
            ]
        },
    )
    archive = batch_vetter.compact(batch, {"alerted_sources": 1})
    assert not source.exists()
    assert not (batch / "vetting/metadata.json").exists()
    assert (batch / "vetting/metadata.json.gz").exists()
    assert (batch / "vetting/scene.json.gz").exists()
    assert not prepared.exists()
    assert not cache.exists()
    assert adapter.read(batch / "cache-eviction.json")["bytes_evicted"] == 22
    assert adapter.read(batch / "vetting/results.json")["rows"][0]["evidence"] == str(
        source.with_suffix(".json.gz")
    )
    adapter.save(
        batch / "batch-complete.json",
        {
            "status": "COMPLETE",
            "batch": 1,
            "archive_sha256": adapter.json_sha256(archive),
        },
    )
    record = {"batch": 1, "sha256": adapter.digest(batch / "batch-complete.json")}
    batch_vetter.verify_archive(batch, record)
    source.with_suffix(".json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Compressed vetting evidence changed"):
        batch_vetter.verify_archive(batch, record)


def test_batch_pixel_eviction_rejects_unrecorded_change(tmp_path):
    batch = tmp_path / "batch"
    cache = batch / "pixel-cache"
    cache.mkdir(parents=True)
    (batch / "vetting/pixel-manifests").mkdir(parents=True)
    pixel = cache / "example.fits.gz"
    pixel.write_bytes(b"changed")
    adapter.save(batch / "vetting.json", {"pixel_cache_directory": str(cache)})
    adapter.save(
        batch / "vetting/pixel-manifests/10.json",
        {
            "quarters": {
                "1": {
                    "status": "AVAILABLE",
                    "path": str(pixel),
                    "sha256": "0" * 64,
                    "url": "https://archive.stsci.edu/example.fits.gz",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="differs from recorded archive product"):
        batch_vetter.evict_batch_pixels(batch)
    assert pixel.exists()


def concurrent_batches(tmp_path, monkeypatch, failing=None):
    lock, active, trace = threading.Lock(), [0], {"detected": [], "peak": 0}

    def detect(out, survey, number, sources, expected, args, index):
        trace["detected"].append(number)
        return out / "batches" / f"{number:06d}", sources, {}, 0.0

    def vet(number, batch, ids, detection, started, args, policy, index):
        with lock:
            active[0] += 1
            trace["peak"] = max(trace["peak"], active[0])
        time.sleep(0.05 if number == 1 else 0.01)
        with lock:
            active[0] -= 1
        if number == failing:
            raise ValueError("Vetting batch did not finish completely")
        return {"batch": number}

    monkeypatch.setattr(batch_vetter, "detect_batch", detect)
    monkeypatch.setattr(batch_vetter, "vet_batch", vet)
    monkeypatch.setattr(
        batch_vetter, "verify_batch", lambda batch, row, identity: {"batch": row["batch"]}
    )
    complete = tmp_path / "batches/000004/batch-complete.json"
    complete.parent.mkdir(parents=True)
    complete.write_text("{}")
    args = SimpleNamespace(concurrent_batches=2, execution_identity={})
    batches = [[str(number)] for number in range(1, 7)]
    run = batch_vetter.run_batches(tmp_path, None, batches, 1, {}, args, None, None)
    return run, trace


def test_survey_vetting_overlaps_batches_and_records_them_in_order(
    tmp_path, monkeypatch
):
    run, trace = concurrent_batches(tmp_path, monkeypatch)
    assert [number for number, _ in run] == [1, 2, 3, 4, 5, 6]
    assert trace["detected"] == [1, 2, 3, 5, 6]
    assert trace["peak"] == 2


def test_survey_vetting_stops_detecting_after_a_failed_batch(tmp_path, monkeypatch):
    run, trace = concurrent_batches(tmp_path, monkeypatch, failing=2)
    with pytest.raises(ValueError, match="did not finish"):
        list(run)
    assert trace["detected"] == [1, 2, 3]


def test_prf_profile_caches_only_exact_positions_within_one_event():
    pixels = importlib.import_module("vetting_pixels")

    class PRF:
        def __init__(self):
            self.calls = 0

        def evaluate(self, targets, origin, shape):
            self.calls += 1
            return np.arange(4).reshape(1, 2, 2) + sum(targets[0])

    local = SimpleNamespace(
        prf=PRF(),
        origin_row=10,
        origin_col=20,
        shape=(2, 2),
        good_pix_mask=np.array([[True, False], [False, True]]),
    )
    first = pixels.prf_profile(local, np.float64(0.25), np.float64(0.5))
    assert pixels.prf_profile(local, 0.25, 0.5) is first
    assert local.prf.calls == 1
    assert not np.array_equal(pixels.prf_profile(local, 0.5, 0.5), first)
    assert local.prf.calls == 2
    other = SimpleNamespace(
        prf=PRF(),
        origin_row=10,
        origin_col=20,
        shape=(2, 2),
        good_pix_mask=local.good_pix_mask,
    )
    pixels.prf_profile(other, 0.25, 0.5)
    assert other.prf.calls == 1


def test_all_runs_prepare_build_scan_and_vet_in_order(tmp_path, monkeypatch):
    calls = []
    binary = tmp_path / "build/release/forgotten-stars-search"

    monkeypatch.setattr(adapter, "dump_path", lambda path: path)
    monkeypatch.setattr(
        adapter, "prepare", lambda args: calls.append(("prepare", args))
    )

    def build(out):
        calls.append(("build", out))
        return binary

    monkeypatch.setattr(adapter, "build_binary", build)
    monkeypatch.setattr(adapter, "scan", lambda args: calls.append(("scan", args)))
    monkeypatch.setattr(adapter, "vet", lambda args: calls.append(("vet", args)))
    monkeypatch.setattr(adapter, "read", lambda path: {"status": "COMPLETE"})

    adapter.run_all(
        SimpleNamespace(
            shards=[200, 201],
            limit=10,
            source_ids=None,
            run=tmp_path,
            threads=8,
            workers=4,
            python=Path(sys.executable),
            no_network=True,
            pixel_cache_directory=tmp_path / "pixels",
            pixel_minimum_free_bytes=20 * 1024**3,
        )
    )

    assert [name for name, _ in calls] == ["prepare", "build", "scan", "vet"]
    assert calls[0][1].out == tmp_path
    assert calls[2][1].binary == binary
    assert calls[3][1].no_network is True
    assert calls[3][1].pixel_cache_directory == tmp_path / "pixels"
    assert calls[3][1].pixel_minimum_free_bytes == 20 * 1024**3


def test_vetting_hashes_full_manifest_and_only_selected_source_files(
    tmp_path, monkeypatch
):
    vetting = importlib.import_module("vetting")
    manifest = tmp_path / "input-manifest.json"
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    first_pixel, second_pixel = tmp_path / "first.fits.gz", tmp_path / "second.fits.gz"
    for path in (first, second, first_pixel, second_pixel):
        path.write_bytes(path.name.encode())
    adapter.save(
        manifest,
        {
            "schema": 1,
            "measurement_scope": "Test exact batch input identities",
            "sources": {
                "1": {
                    "arrays": first.name,
                    "pixels": {"1": {"path": first_pixel.name}},
                },
                "2": {
                    "arrays": second.name,
                    "pixels": {"1": {"path": second_pixel.name}},
                },
            },
        },
    )
    config = tmp_path / "vetting.json"
    adapter.save(config, {"input_manifest": str(manifest)})
    monkeypatch.setattr(vetting, "ROOT", tmp_path)
    monkeypatch.setattr(vetting, "CONFIG", config)

    original = vetting.input_hashes(["1"])
    assert set(original) == {str(manifest), str(first), str(first_pixel)}
    second.write_bytes(b"changed outside this batch")
    assert vetting.input_hashes(["1"]) == original
    specification = adapter.read(manifest)
    specification["measurement_scope"] = "Changed control definition"
    adapter.save(manifest, specification)
    changed_manifest = vetting.input_hashes(["1"])
    assert changed_manifest != original
    first.write_bytes(b"changed in this batch")
    assert vetting.input_hashes(["1"]) != changed_manifest
    with pytest.raises(KeyError):
        vetting.input_hashes(["missing"])


@pytest.mark.parametrize(
    ("record", "execution"),
    [
        ({}, "SKIP_AFTER_FIRST_FAILURE"),
        ({"record_every_gate": True}, "RECORD_EVERY_GATE"),
    ],
)
@pytest.mark.parametrize("view", ["psf_flat", "psf_qdiag", "sap_qdiag"])
def test_vetting_preserves_venv_launcher_and_exact_zero_quality(
    tmp_path, monkeypatch, record, execution, view
):
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(sys.executable)
    adapter.save(
        tmp_path / "detection.json",
        {"status": "DETECTION_COMPLETE", "alerted_sources": 0, "alerts": 0},
    )
    adapter.save(
        tmp_path / "discovery-manifest.json",
        {"detector_contract": adapter.detector_contract(view)},
    )
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(adapter, "verify_discovery", lambda *args: None)

    def run(command, log, environment):
        assert command[0] == str(launcher.absolute())
        config = adapter.read(environment["SPACELAB_VETTING_CONFIG"])
        assert config["quality_bitmask"] == -1
        if view == "psf_flat":
            assert config["quality_selection"] == adapter.read(
                adapter.ROOT / "pipeline/vetting/config.json"
            )["quality_selection"]
        else:
            assert config["flatten_mask_views"] == ["psf_flat", "psf_nova"]
            assert "full PSF and SAP retain their own" in config["quality_selection"]
        assert config["triage_execution"] == execution
        assert config["disposition_policy"] == str(adapter.DISPOSITION_POLICY)
        assert config["pixel_cache_directory"] == str(
            (ROOT / "dump/unit-test-pixel-cache").resolve()
        )
        assert config["pixel_minimum_free_bytes"] == 20 * 1024**3
        adapter.save(
            tmp_path / "vetting/results.json",
            {
                "disposition_execution": importlib.import_module(
                    "runtime"
                ).disposition_execution(
                    adapter.read(ROOT / adapter.DISPOSITION_POLICY)
                ),
                "sources": 0,
                "original_alerts_preserved": 0,
                "processing_status": "EVIDENCE_STAGES_COMPLETE",
                "boi_signals": 0,
                "unresolved_signals": 0,
            },
        )
        adapter.save(
            tmp_path / "vetting/stage-timings.json",
            {"schema": 1, "stages": []},
        )

    monkeypatch.setattr(adapter, "run_command", run)
    adapter.vet(
        SimpleNamespace(
            run=tmp_path,
            python=launcher,
            workers=1,
            no_network=True,
            pixel_cache_directory=ROOT / "dump/unit-test-pixel-cache",
            pixel_minimum_free_bytes=20 * 1024**3,
            **record,
        )
    )


def test_pixel_cache_configuration_rejects_archive_directory_before_launch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "EXTERNAL_DUMPS", ())
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    cache = tmp_path / "dump/pixel-cache"
    cache.parent.mkdir()
    outside = tmp_path / "read-only-archive"
    outside.mkdir()
    cache.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="local cache"):
        adapter.pixel_cache_configuration(
            SimpleNamespace(), {"pixel_cache_directory": "dump/pixel-cache"}
        )
    assert list(outside.iterdir()) == []


def test_pixel_cache_configuration_accepts_local_directory_with_read_only_seed_links(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    cache = tmp_path / "dump/local-pixels"
    cache.mkdir(parents=True)
    public = tmp_path / "read-only-public.fits.gz"
    public.write_bytes(b"declared public input")
    (cache / public.name).symlink_to(public)
    result = adapter.pixel_cache_configuration(
        SimpleNamespace(
            pixel_cache_directory=cache, pixel_minimum_free_bytes=20 * 1024**3
        ),
        {"pixel_cache_directory": "dump/archive-pixels"},
    )
    assert result == {
        "pixel_cache_directory": str(cache),
        "pixel_minimum_free_bytes": 20 * 1024**3,
    }
    assert public.read_bytes() == b"declared public input"


def test_pixel_cache_configuration_accepts_resolved_dump_symlink(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    workspace = tmp_path / "cloud-workspace"
    root.mkdir()
    workspace.mkdir()
    (root / "dump").symlink_to(workspace, target_is_directory=True)
    monkeypatch.setattr(adapter, "ROOT", root)
    monkeypatch.setattr(adapter, "EXTERNAL_DUMPS", ())
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    assert adapter.local_pixel_cache("dump/pixels") == workspace / "pixels"


@pytest.mark.parametrize("linked", [False, True])
def test_pixel_cache_configuration_accepts_declared_external_control_storage(
    tmp_path, monkeypatch, linked
):
    root = tmp_path / "repository"
    external = tmp_path / "external/forgotten-stars-workspace/dump"
    cache = external / "controls-20260926/pixels"
    cache.mkdir(parents=True)
    monkeypatch.setattr(adapter, "ROOT", root)
    monkeypatch.setattr(adapter, "EXTERNAL_DUMPS", (external,))
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    selected = cache
    if linked:
        selected = root / "dump/pixels"
        selected.parent.mkdir(parents=True)
        selected.symlink_to(cache, target_is_directory=True)
    result = adapter.pixel_cache_configuration(
        SimpleNamespace(
            pixel_cache_directory=selected, pixel_minimum_free_bytes=20 * 1024**3
        ),
        {"pixel_cache_directory": "dump/archive-pixels"},
    )
    assert result == {
        "pixel_cache_directory": str(cache),
        "pixel_minimum_free_bytes": 20 * 1024**3,
    }


def test_pixel_cache_configuration_rejects_existing_external_archive(
    tmp_path, monkeypatch
):
    external = tmp_path / "external/forgotten-stars-workspace/dump"
    archive = external / "pixel-cache-20260924"
    archive.mkdir(parents=True)
    marker = archive / "existing-input.fits.gz"
    marker.write_bytes(b"read-only archive bytes")
    monkeypatch.setattr(adapter, "ROOT", tmp_path / "repository")
    monkeypatch.setattr(adapter, "EXTERNAL_DUMPS", (external,))
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    with pytest.raises(ValueError, match="declared external control storage"):
        adapter.pixel_cache_configuration(
            SimpleNamespace(pixel_cache_directory=archive),
            {"pixel_cache_directory": "dump/pixels"},
        )
    assert marker.read_bytes() == b"read-only archive bytes"
    assert list(archive.iterdir()) == [marker]


@pytest.mark.parametrize("reserve", [0, -1, True, 1.5])
def test_pixel_disk_reserve_rejects_invalid_values(tmp_path, monkeypatch, reserve):
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    with pytest.raises(ValueError, match="positive integer"):
        adapter.pixel_cache_configuration(
            SimpleNamespace(pixel_minimum_free_bytes=reserve),
            {"pixel_cache_directory": "dump/local-pixels"},
        )


@pytest.mark.parametrize("stage", ["vet", "all"])
def test_pixel_cache_options_reach_cli_stage(tmp_path, monkeypatch, stage):
    calls = []
    monkeypatch.setattr(adapter, "vet" if stage == "vet" else "run_all", calls.append)
    cache = tmp_path / "dump/local-pixels"
    argv = [
        "run.py",
        stage,
        "--run",
        str(tmp_path / "run"),
        "--python",
        sys.executable,
        "--pixel-cache-directory",
        str(cache),
        "--pixel-minimum-free-bytes",
        str(20 * 1024**3),
    ]
    if stage == "all":
        argv.extend(["--shards", "1", "--limit", "100"])
    monkeypatch.setattr(sys, "argv", argv)
    adapter.main()
    assert len(calls) == 1
    assert calls[0].pixel_cache_directory == cache
    assert calls[0].pixel_minimum_free_bytes == 20 * 1024**3


@pytest.fixture
def frozen_evaluation_policy(tmp_path):
    runtime = importlib.import_module("runtime")
    policy = {
        "schema": 3,
        "triage_contract": adapter.read(
            ROOT / "pipeline/vetting/disposition-policy.json"
        )["triage_contract"],
        "policy_id": "deterministic-boi-v1",
        "status": "FROZEN_EVALUATION",
        "lane_gates": lane_policy_fixture(tmp_path),
        "boi_catalogue_enabled": True,
        "pipeline_contract": {},
        "implementation": {},
        "assessment_rules": json.loads(
            (ROOT / "pipeline/vetting/config.json").read_text()
        )["triage"]["assessment_rules"],
        "evaluation_binding": {
            "run_spec_sha256": "a" * 64,
            "detector_sha256": "b" * 64,
            "calibration_sha256": "c" * 64,
            "development_replay": {
                "controls_sha256": "d" * 64,
                "cohorts": [{"path": "dump/development-cohort.json", "sha256": "f" * 64}],
            },
            "final_replay": {
                "controls_sha256": "e" * 64,
                "cohorts": [{"path": "dump/final-cohort.json", "sha256": "f" * 64}],
            },
        },
    }
    policy["definition_sha256"] = runtime.disposition_definition_sha256(policy)
    return policy


@pytest.mark.parametrize("role", [None, "development_replay", "final_replay"])
def test_frozen_policy_records_verified_token_role(
    tmp_path, monkeypatch, frozen_evaluation_policy, role
):
    vetting = importlib.import_module("vetting")
    runtime = importlib.import_module("runtime")
    policy_path, config_path = tmp_path / "policy.json", tmp_path / "config.json"
    adapter.save(policy_path, frozen_evaluation_policy)
    adapter.save(config_path, {"disposition_policy": str(policy_path)})
    monkeypatch.setattr(vetting, "CONFIG", config_path)
    monkeypatch.setattr(vetting, "pipeline_contract", lambda config: {})
    monkeypatch.setattr(vetting, "implementation_hashes", dict)
    token = (
        runtime.disposition_evaluation_token(frozen_evaluation_policy, role)
        if role
        else "invalid"
    )
    monkeypatch.setenv("SPACELAB_DISPOSITION_EVALUATION_TOKEN", token)
    loaded = vetting.load_disposition_policy()
    execution = runtime.disposition_execution(loaded)
    assert loaded["_evaluation_authorized"] is (role is not None)
    assert execution["evaluation_role"] == role
    assert execution["boi_catalogue_enabled"] is (role is not None)
    assert execution["production_boi_enabled"] is False



@pytest.mark.parametrize("role", [None, "final_replay"])
def test_outer_vetting_result_reports_effective_evaluation(
    tmp_path, monkeypatch, frozen_evaluation_policy, role
):
    runtime = importlib.import_module("runtime")
    policy_path = tmp_path / "policy.json"
    adapter.save(policy_path, frozen_evaluation_policy)
    effective = dict(
        frozen_evaluation_policy,
        _evaluation_authorized=role is not None,
        _evaluation_role=role,
    )
    execution = runtime.disposition_execution(effective)
    adapter.save(
        tmp_path / "detection.json",
        {"status": "DETECTION_COMPLETE", "alerted_sources": 1, "alerts": 1},
    )
    adapter.save(tmp_path / "queue.json", {})
    adapter.save(
        tmp_path / "discovery-manifest.json",
        {"detector_contract": adapter.detector_contract()},
    )
    monkeypatch.setattr(adapter, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(adapter, "detector_contract", lambda *args: {})
    monkeypatch.setattr(adapter, "verify_discovery", lambda *args: None)

    def complete(*args):
        adapter.save(
            tmp_path / "vetting/results.json",
            {
                "processing_status": "EVIDENCE_STAGES_COMPLETE",
                "sources": 1,
                "original_alerts_preserved": 1,
                "boi_signals": int(role is not None),
                "unresolved_signals": int(role is None),
                "disposition_execution": execution,
            },
        )
        adapter.save(tmp_path / "vetting/stage-timings.json", {"stages": []})

    monkeypatch.setattr(adapter, "run_command", complete)
    adapter.vet(
        SimpleNamespace(
            run=tmp_path,
            workers=1,
            no_network=True,
            python=Path(sys.executable),
            disposition_policy=policy_path,
            pixel_cache_directory=ROOT / "dump/unit-test-pixel-cache",
        )
    )
    result = adapter.read(tmp_path / "result.json")
    assert result["boi_catalogue_enabled"] is (role is not None)
    assert result["boi_signals"] == int(role is not None)
    assert result["production_boi_enabled"] is False
    assert result["disposition_execution"] == execution


def test_control_reports_validate_executed_policy(tmp_path, frozen_evaluation_policy):
    from pipeline.deterministic import catalogue_control_report as report

    runtime = importlib.import_module("runtime")
    policy_path = tmp_path / "policy.json"
    adapter.save(policy_path, frozen_evaluation_policy)
    adapter.save(tmp_path / "vetting.json", {"disposition_policy": str(policy_path)})
    effective = dict(
        frozen_evaluation_policy,
        _evaluation_authorized=True,
        _evaluation_role="final_replay",
    )
    execution = runtime.disposition_execution(effective)
    source = {"disposition_execution": execution, "disposition_policy": effective}
    results = {"disposition_execution": execution}
    assert report.validate_execution(tmp_path, results, {"123": source}) == execution
    source["disposition_execution"] = dict(execution, boi_catalogue_enabled=False)
    with pytest.raises(ValueError, match="source disposition execution"):
        report.validate_execution(tmp_path, results, {"123": source})
    with pytest.raises(TypeError, match="no disposition execution record"):
        report.validate_execution(tmp_path, {}, {})
    with pytest.raises(ValueError, match="differs from the policy"):
        report.validate_execution(
            tmp_path,
            {"disposition_execution": dict(execution, production_boi_enabled=True)},
            {},
        )
    frozen_evaluation_policy["assessment_rules"]["instrument"]["spatial_null_bic_margin"] = 20
    adapter.save(policy_path, frozen_evaluation_policy)
    with pytest.raises(ValueError, match="definition changed"):
        report.validate_execution(tmp_path, results, {})


@pytest.mark.parametrize("role", [None, "final_replay"])
def test_empty_vetting_cohort_still_records_execution(
    tmp_path, monkeypatch, frozen_evaluation_policy, role
):
    vetting = importlib.import_module("vetting")
    effective = dict(
        frozen_evaluation_policy,
        _evaluation_authorized=role is not None,
        _evaluation_role=role,
    )
    adapter.save(tmp_path / "cohort.json", [])
    monkeypatch.setattr(vetting, "RUN", tmp_path)
    monkeypatch.setattr(vetting, "load_disposition_policy", lambda: effective)
    vetting.summarize()
    results = adapter.read(tmp_path / "results.json")
    assert results["sources"] == 0
    assert results["disposition_execution"]["boi_catalogue_enabled"] is (
        role is not None
    )
    assert results["disposition_execution"]["production_boi_enabled"] is False


@pytest.mark.parametrize("role", [None, "development_replay"])
def test_false_class_evaluation_requires_executed_final_role(
    tmp_path, monkeypatch, frozen_evaluation_policy, role
):
    from pipeline.deterministic import false_class_control_report as report

    runtime = importlib.import_module("runtime")
    policy_path = tmp_path / "policy.json"
    adapter.save(policy_path, frozen_evaluation_policy)
    adapter.save(tmp_path / "vetting.json", {"disposition_policy": str(policy_path)})
    adapter.save(tmp_path / "evaluation-source-ids.json", ["42"])
    effective = dict(
        frozen_evaluation_policy,
        _evaluation_authorized=role is not None,
        _evaluation_role=role,
    )
    execution = runtime.disposition_execution(effective)
    monkeypatch.setattr(
        report,
        "validate_controls",
        lambda *args: (
            {"status": "FROZEN"},
            ["42"],
            [{"source_id": "42", "partition": "evaluation"}],
        ),
    )
    monkeypatch.setattr(
        report.catalogue,
        "validate_run",
        lambda *_: ({}, {}, {"disposition_execution": execution}, {}),
    )
    with pytest.raises(ValueError, match="requires frozen controls and policy"):
        report.build_report(
            tmp_path, tmp_path, "evaluation", tmp_path / "frozen-generator"
        )


@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
def test_transit_only_profile_changes_only_sparse_emission(view):
    baseline = adapter.read(adapter.CONFIG)
    selected = adapter.search_configuration("rust-transit-only-v1")
    assert {key for key in baseline if baseline[key] != selected[key]} == {"sparse_event_snr_min", "sparse_event_min_points", "max_sparse_events_per_target"}
    assert selected["sparse_event_snr_min"] is None
    assert baseline["gates"]["n_transits_min"] == 3
    contract = adapter.detector_contract(view, "rust-transit-only-v1")
    assert adapter.bound_detector_contract(contract) == contract
    assert contract["search_config_sha256"] == adapter.json_sha256(selected)
    assert contract["origin"] == "rust_v6_deterministic"
    assert adapter.detector_contract(view).get("discovery_profile") is None
    with pytest.raises(ValueError, match="restricted"):
        adapter.detector_contract("psf_flat", "rust-transit-only-v1")


def test_transit_only_does_not_call_shallow_or_its_library(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Excluded shallow detector executed")
    monkeypatch.setattr(adapter, "scan_shallow", unexpected)
    monkeypatch.setattr(adapter.shallow_detector, "bls_library", unexpected)
    output = tmp_path / "shallow-raw.csv"
    assert adapter.scan_discovery_shallow(tmp_path, output, ["123", "456"], "rust-transit-only-v1") == {"123": [], "456": []}
    eligibility = adapter.shallow_eligibility(tmp_path / "shallow-eligibility.csv", ["123", "456"])
    assert all(row["status"] == "NOT_RUN" for row in eligibility)
    contract = adapter.detector_contract("psf_qdiag", "rust-transit-only-v1")
    adapter.verify_profile_outputs(contract, [], [], [], eligibility)
    with pytest.raises(ValueError, match="omitted"):
        adapter.verify_profile_outputs(adapter.detector_contract(), [], [], [], eligibility)


@pytest.mark.parametrize("classification", ["EVENT_LIKE", "EB_LIKE", "SYSTEMATIC_LIKE", "LOW_SIGNIFICANCE", "NO_SIGNAL"])
def test_transit_only_excludes_every_other_classification(classification):
    row = dict(rust_row(classification=classification), n_transits="3", quarter_loo_status="FAIL")
    assert not adapter.admitted_row(row, "rust-transit-only-v1")
    transit = dict(rust_row(), n_transits="3")
    assert adapter.admitted_row(transit, "rust-transit-only-v1")
    with pytest.raises(ValueError, match="fewer"):
        adapter.admitted_row(dict(transit, n_transits="2"), "rust-transit-only-v1")


@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
def test_transit_only_filters_co_resident_hypotheses_and_retains_eb_diagnostics(raw, tmp_path, view):
    records = adapter.prepare_records(raw, view)
    path = tmp_path / f"123.{view}.fsc"
    adapter.write_fsc(path, "123", view, records)
    source = {"source_id": "123", "inputs": {view: {"fsc_path": str(path)}}}
    primary = dict(rust_row(), photometry=view, n_transits="3")
    excluded = [
        dict(rust_row(2, "EB_LIKE"), photometry=view),
        dict(rust_row(3, "SYSTEMATIC_LIKE"), photometry=view, quarter_loo_status="FAIL"),
        dict(rust_row(4, "EVENT_LIKE", period=""), photometry=view),
    ]
    diagnostics, shallow, alerts, hypotheses = adapter.expand_source_rows(
        source, {("123", view): [primary, *excluded]}, {"123": []},
        discovery_view=view, discovery_profile="rust-transit-only-v1",
    )
    original_alerts, original_hypothesis = adapter.supplied_events(primary, records)
    assert diagnostics == [primary, *excluded]
    assert shallow == []
    assert alerts == original_alerts
    assert len(hypotheses) == 1
    assert hypotheses[0]["period_days"] == original_hypothesis["period_days"]
    assert hypotheses[0]["detector_followup_lane"] == "PRIMARY_TRANSIT_LIKE"


def test_transit_only_cannot_admit_survivor_review_or_shallow(tmp_path):
    contract = adapter.detector_contract("sap_qdiag", "rust-transit-only-v1")
    primary = dict(rust_row(), n_transits="3")
    eligibility = [{"status": "NOT_RUN"}]
    with pytest.raises(ValueError, match="Excluded"):
        adapter.verify_profile_outputs(contract, [primary], [rust_row(2, "SYSTEMATIC_LIKE")], [], eligibility)
    with pytest.raises(ValueError, match="Excluded"):
        adapter.verify_profile_outputs(contract, [primary], [], [shallow_row()], eligibility)



@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
def test_transit_only_resume_rejects_profile_and_configuration_changes(tmp_path, monkeypatch, view):
    monkeypatch.setattr(adapter, "build_binary", build_survey_binary)
    args = SimpleNamespace(
        shards=[0], limit=50, batch_size=50, threads=32, source_ids=None,
        input_manifest=None, disposition_policy=adapter.DISPOSITION_POLICY,
        discovery_view=view, discovery_profile="rust-transit-only-v1",
    )
    out = tmp_path / "survey"
    report, *_ = adapter.start_survey(out, args)
    adapter.validate_survey_identity(out, args, report)
    args.discovery_profile = "all-lanes-v1"
    with pytest.raises(ValueError, match="arguments or implementation"):
        adapter.validate_survey_identity(out, args, report)
    args.discovery_profile = "rust-transit-only-v1"
    config = adapter.read(out / "detector-config.json")
    config["gates"]["n_transits_min"] = 4
    adapter.save(out / "detector-config.json", config)
    with pytest.raises(ValueError, match="configuration differs"):
        adapter.validate_survey_identity(out, args, report)


@pytest.mark.parametrize("view", ["psf_qdiag", "sap_qdiag"])
def test_transit_only_policy_binds_actual_quality_mask_scope(tmp_path, view):
    import vetting
    from deterministic import full_survey
    policy_path = tmp_path / "policy.json"
    full_survey.freeze_boi_policy(policy_path, adapter, view, "rust-transit-only-v1")
    policy = adapter.read(policy_path)
    config = adapter.read(ROOT / "pipeline/vetting/config.json")
    config["discovery_contract"] = adapter.detector_contract(view, "rust-transit-only-v1")
    config["flatten_mask_views"] = ["psf_flat", "psf_nova"]
    assert policy["pipeline_contract"] == vetting.pipeline_contract(config)
    full_survey.verify_frozen_release(policy_path, adapter.digest(policy_path), adapter)
