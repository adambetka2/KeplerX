"""Nulls and queue gates must preserve physical inputs and trial accounting."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

from lc2 import photometry

from pipeline.deterministic import run
from pipeline.deterministic import signal_contracts as signals


@pytest.fixture
def raw():
    count = 360
    quarter = np.repeat([1, 2, 3], 120).astype(np.int16)
    flux = 1000 + np.sin(np.arange(count))
    flux[10], flux[20] = 1200, 800
    result = {
        "time_bkjd": np.arange(count) * 0.02,
        "cadenceno": np.arange(count, dtype=np.int32),
        "quarter": quarter,
        "sap_quality": np.zeros(count, dtype=np.int64),
        "flatten_mask": np.zeros(count, dtype=np.int64),
    }
    for prefix in ("psf_flat_kbonus", "psf_stitched", "sap_stitched"):
        result[prefix + "_flux"] = flux.copy()
        result[prefix + "_flux_err"] = np.linspace(0.3, 0.4, count)
    for key in ("time_bkjd", "cadenceno", "quarter", "sap_quality"):
        result["qdiag_" + key] = result[key].copy()
    for prefix in ("psf", "sap"):
        result["qdiag_" + prefix + "_flux"] = flux.copy()
        result["qdiag_" + prefix + "_flux_err"] = np.linspace(0.3, 0.4, count)
    result["qdiag_psf_flux_nova"] = flux.copy()
    result["qdiag_psf_flux_err_nova"] = np.linspace(0.3, 0.4, count)
    return result


def test_inversion_follows_upper_clip_without_clipping_the_former_dip(raw):
    original = raw["psf_flat_kbonus_flux"].copy()
    transformed, quarters = signals.null_raw(raw, "123", "inverted", 5)
    assert quarters == 3
    assert np.array_equal(raw["psf_flat_kbonus_flux"], original)
    curve = photometry(transformed, "psf_flat")
    assert np.isnan(curve["flux"][10])
    assert curve["flux"][20] > 1.19
    records = run.prepare_records(transformed, relative_input=True)
    assert 10 not in records["cadence"]
    assert records["flux"][records["cadence"] == 20][0] > 0.19
    assert np.array_equal(
        records["flux"], (curve["flux"][records["cadence"]] - 1).astype("f4")
    )
    assert np.array_equal(
        records["error"], curve["error"][records["cadence"]].astype("f4")
    )


def test_survey_preparation_keeps_inverted_dip_and_matches_standalone(
    raw, tmp_path, monkeypatch
):
    monkeypatch.setattr(signals, "dump_path", Path)
    arrays = tmp_path / "original.npz"
    np.savez_compressed(arrays, **raw)
    metadata = {
        "source_id": "123",
        "origin_tpf_kic": "45",
        "raw_fits_sha256": "a" * 64,
        "shard": 1,
        "shard_path": "original",
    }
    manifest = tmp_path / "original.json"
    manifest.write_text(json.dumps({
        "schema": 1,
        "role": "development",
        "measurement_scope": "SYNTHETIC_CONTRACT_TEST",
        "sources": {"123": {
            "arrays": arrays.name,
            "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
            "metadata": metadata,
        }},
    }))
    out = tmp_path / "null"
    signals.construct(SimpleNamespace(
        input_manifest=manifest, variant="inverted", seed=5,
        exclusions=None, out=out,
    ))
    prepared = []
    source = {"source_id": "123", "shard": 1}
    for name in ("survey", "standalone"):
        target = tmp_path / name
        (target / "inputs").mkdir(parents=True)
        manifest = out / "input-manifest.json"
        if name == "survey":
            receipt = run.prepare_discovery_source(target, source, manifest)
            assert receipt["retained_cadences"] == 359
        else:
            run.prepare_source(target, source, manifest)
        prepared.append(run.read_fsc(target / "inputs/123.psf_flat.fsc", "123", "psf_flat"))
    assert prepared[0].tobytes() == prepared[1].tobytes()
    assert 10 not in prepared[0]["cadence"]
    assert prepared[0]["flux"][prepared[0]["cadence"] == 20][0] > 0.19
    payload = json.loads(manifest.read_text())
    payload["construction_ledger"][0]["quarters"] = 2
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source coverage changed"):
        signals.null_manifest(ROOT, manifest)


def test_scrambling_is_replayable_and_moves_flux_with_errors_inside_quarters(raw):
    a, _ = signals.null_raw(raw, "123", "quarter_scrambled", 5)
    b, _ = signals.null_raw(raw, "123", "quarter_scrambled", 5)
    for key in raw:
        assert np.array_equal(a[key], b[key], equal_nan=True)
    indices, (_, flux, error, quarter, _) = signals.relative_curve(
        photometry(raw, "psf_flat"), True
    )
    for value in np.unique(quarter):
        use = quarter == value
        actual = np.column_stack(
            [
                a["psf_flat_kbonus_flux"][indices[use]],
                a["psf_flat_kbonus_flux_err"][indices[use]],
            ]
        )
        expected = np.column_stack([1 + flux[use], error[use]])
        assert sorted(map(tuple, actual)) == sorted(map(tuple, expected))
        assert not np.array_equal(actual, expected)
    assert np.array_equal(a["time_bkjd"], raw["time_bkjd"])


def test_two_quarters_have_only_inversion(raw):
    raw["quarter"][:] = np.repeat([1, 2], 180)
    result, count = signals.null_raw(raw, "123", "quarter_scrambled", 0)
    assert result is None and count == 2
    assert signals.null_raw(raw, "123", "inverted", 0)[0] is not None


def test_lane_threshold_is_inclusive_and_records_metric_and_identity():
    gates = {
        "thresholds": {"periodic": 10},
        "calibration": {"path": "opened", "sha256": "a" * 64},
    }
    passed = signals.lane_decision("signal", "periodic", {"search_mes": "10"}, gates)
    assert passed["decision"] == "PASS" and passed["threshold"] == 10
    assert passed["proposal_id"] == "signal" and passed["metric"] == "search_mes"
    assert (
        signals.lane_decision("signal", "periodic", {"search_mes": "9.9"}, gates)[
            "decision"
        ]
        == "REJECT"
    )
    assert (
        signals.lane_decision("signal", "periodic", {"search_mes": "9.9"}, None)[
            "status"
        ]
        == "DEVELOPMENT_UNGATED"
    )


def test_frozen_lanes_require_verified_opened_calibration(tmp_path):
    with pytest.raises(ValueError, match="opened-null"):
        signals.calibrated_lanes({}, required=True)
    thresholds = {name: 10 for name in signals.LANE_METRICS}
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "status": "CALIBRATED",
                "measurement_scope": "OPENED_DEVELOPMENT_NULLS",
                "lane_contract": signals.LANE_CONTRACT,
                "thresholds": thresholds,
            }
        )
    )
    policy = {
        "lane_gates": {
            "status": "CALIBRATED",
            "contract": signals.LANE_CONTRACT,
            "thresholds": thresholds,
            "calibration": {"path": str(path), "sha256": run.digest(path)},
        }
    }
    assert signals.calibrated_lanes(policy, True) == policy["lane_gates"]
    path.write_text(
        path.read_text().replace("OPENED_DEVELOPMENT_NULLS", "SEALED_NULLS")
    )
    policy["lane_gates"]["calibration"]["sha256"] = run.digest(path)
    with pytest.raises(ValueError, match="opened-null"):
        signals.calibrated_lanes(policy, True)


def test_null_exclusion_uses_exact_event_membership_and_quarter_support():
    alerts = [
        {"event_id": str(i), "quarter": 1, "start_bkjd": t - 0.05, "end_bkjd": t + 0.05}
        for i, t in enumerate((1, 3))
    ]
    reference = {
        "reference_id": "known",
        "origin": "REAL_SIGNAL",
        "duration_hours": 2.4,
        "events": [
            {"event_id": str(i), "centre_bkjd": t, "support": {"1": [1]}}
            for i, t in enumerate((1, 3))
        ],
    }
    assert signals.null_exclusion(alerts, [reference])["status"] == "EXCLUDED"
    alerts[1]["start_bkjd"] += 1
    alerts[1]["end_bkjd"] += 1
    assert signals.null_exclusion(alerts, [reference])["status"] == "ELIGIBLE"


def test_null_all_flux_passes_count_without_a_pixel_or_pc_disposition():
    proposal = {
        "exclusion": {"status": "ELIGIBLE"},
        "gate_path": [
            {"stage": stage, "decision": "PASS"} for stage in signals.FLUX_GATES
        ],
    }
    assert signals.null_proposal_outcome(proposal) == {
        "vetted": True,
        "false_candidate": True,
        "excluded": False,
    }
    proposal["gate_path"] = [{"stage": "shape", "decision": "DISMISS"}]
    assert signals.null_proposal_outcome(proposal)["vetted"]
    assert not signals.null_proposal_outcome(proposal)["false_candidate"]
    proposal["gate_path"][0]["decision"] = "PASS"
    with pytest.raises(ValueError, match="incomplete"):
        signals.null_proposal_outcome(proposal)
    proposal["exclusion"]["status"] = "EXCLUDED"
    assert not signals.null_proposal_outcome(proposal)["vetted"]


def test_null_rejects_a_nonexistent_gate_decision():
    with pytest.raises(ValueError, match="decision is invalid"):
        signals.null_proposal_outcome({
            "exclusion": {"status": "ELIGIBLE"},
            "gate_path": [{"stage": "shape", "decision": "FAIL"}],
        })


@pytest.mark.parametrize("path", [None, [], [{"stage": "shape", "decision": "CONTINUE_INCONCLUSIVE"}]])
def test_missing_or_inconclusive_truncated_null_path_cannot_supply_a_vetted_trial(path):
    with pytest.raises(ValueError, match="missing|incomplete"):
        signals.null_proposal_outcome({"exclusion": {"status": "ELIGIBLE"}, "gate_path": path})


def test_failed_associated_fit_cannot_hide_behind_a_passing_null_fit():
    paths = [
        {"signal_id": "passed", "gate_path": [{"stage": stage, "decision": "PASS"} for stage in signals.FLUX_GATES]},
        {"signal_id": "failed", "gate_path": []},
    ]
    with pytest.raises(ValueError, match="missing"):
        signals.null_proposal_outcome({"exclusion": {"status": "ELIGIBLE"}, "gate_paths": paths})


@pytest.mark.parametrize("fitted_paths", [None, 1, 2])
def test_null_rejects_instrument_dismissal_even_with_a_passing_alias(fitted_paths):
    path = [{"stage": stage, "decision": "PASS"} for stage in signals.FLUX_GATES]
    invalid = [dict(row) for row in path]
    invalid[-1]["decision"] = "DISMISS"
    proposal = {"exclusion": {"status": "ELIGIBLE"}, "gate_path": invalid}
    if fitted_paths is not None:
        proposal = {
            "exclusion": proposal["exclusion"],
            "gate_paths": [{"signal_id": "invalid", "gate_path": invalid}],
        }
        if fitted_paths == 2:
            proposal["gate_paths"].insert(0, {"signal_id": "passed", "gate_path": path})
    with pytest.raises(ValueError, match="instrument.*decision is invalid"):
        signals.null_proposal_outcome(proposal)


def test_null_denominator_counts_proposals_and_reports_sources_separately():
    def proposal(identity, passed, excluded=False):
        return {
            "proposal_id": identity,
            "exclusion": {"status": "EXCLUDED" if excluded else "ELIGIBLE"},
            "gate_path": (
                [{"stage": stage, "decision": "PASS"} for stage in signals.FLUX_GATES]
                if passed
                else [{"stage": "shape", "decision": "DISMISS"}]
            ),
        }

    variants = [
        {
            "variant": variant,
            "status": "COMPLETE_NULL_FLUX_LEDGER",
            "emitted_alerts": 2,
            "adjudicated_alerts": 2,
            "exact_identity_match": True,
            "declared_proposal_ids": ["a", "b"],
            "proposals": [
                proposal("a", variant == "inverted"),
                proposal("b", True, True),
            ],
        }
        for variant in ("inverted", "quarter_scrambled")
    ]
    cases = [
        {"usable_quarters": 3, "variants": variants, "scientific_abstention": True}
    ]
    measured = signals.null_population(cases, 1)
    assert measured["trials"] == 2 and measured["outcomes"] == 1
    assert measured["false_candidate_sources"] == 1 and measured["source_fraction"] == 1
    assert measured["excluded_proposals"] == 2 and measured["completed_sources"] == 1
    variants[0]["proposals"].pop()
    with pytest.raises(ValueError, match="lost, reordered or duplicated"):
        signals.null_population(cases, 1)


def test_each_null_proposal_counts_once_when_several_associated_fits_survive():
    paths = [
        {
            "signal_id": identity,
            "gate_path": [
                {"stage": stage, "decision": "PASS"} for stage in signals.FLUX_GATES
            ],
        }
        for identity in ("fit-a", "fit-b")
    ]
    result = signals.null_proposal_outcome(
        {"exclusion": {"status": "ELIGIBLE"}, "gate_paths": paths}
    )
    assert result == {"vetted": True, "false_candidate": True, "excluded": False}


def test_null_variant_coverage_is_complete_even_without_proposals():
    row = {
        "variant": "inverted",
        "status": "COMPLETE_NULL_FLUX_LEDGER",
        "emitted_alerts": 0,
        "adjudicated_alerts": 0,
        "exact_identity_match": True,
        "declared_proposal_ids": [],
        "proposals": [],
    }
    case = {"usable_quarters": 2, "variants": [row], "scientific_abstention": False}
    assert signals.null_population([case], 1)["completed_sources"] == 1
    assert signals.null_population([case], 1)["trials"] == 0
    case["usable_quarters"] = 3
    with pytest.raises(ValueError, match="short-coverage"):
        signals.null_population([case], 1)
