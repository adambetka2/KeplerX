"""Checks for the survey evidence policy: trimmed JSON and compact proposal arrays."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evolution as API

RECORD = np.dtype([("time", "<f8"), ("flux", "<f4"), ("error", "<f4"), ("quarter", "<i2"), ("cadence", "<i4")])
SEED = "baseline_duration_1_native_coherence"


def fake_result():
    rng = np.random.default_rng(20261014)
    periodogram = {name: rng.normal(size=5000) for name in ("period_days", "search_snr", "depth_ppm", "duration_days", "epoch_bkjd")}
    coherence = {
        "band": SEED,
        "status": "SEARCHED",
        "store_periodograms": True,
        "config": {"grid": 20000},
        "periodogram_arrays": periodogram,
        "tested_peaks": [{"period_days": 2.5, "search_snr": 8.1}],
        "decisions": [{"status": "SHAPE_ACCEPTED", "reasons": [], "first_reason": None}],
        "local_refinements": [1, 2, 3],
        "excluded_refinement_intervals": [[1.0, 2.0]],
    }
    preprocessing = {
        "time": np.linspace(131.5, 160.9, 1200),
        "prepared_flux": np.zeros(1200),
        "nuisance_residual": np.zeros(1200),
        "noise": np.ones(1200),
        "quarter": np.ones(1200, dtype=np.int64),
        "basis_data": np.zeros(50),
    }
    refinement = {
        "seed_band": SEED,
        "seed_rank": 1,
        "trials": [{"snr": 1.0}],
        "profile_arrays": {"table_0_phase_days": np.zeros(9)},
        "consistency": {
            "status": "MEASURED",
            "odd_even_depths_ppm": np.array([1.0, 2.0]),
            "odd_even_covariance": np.eye(2),
            "per_cycle": [{"cycle": 1}],
            "inverse_fits": [1],
            "positive_cycle_fraction": 0.7,
        },
    }
    physical = {
        "band": "baseline_physical",
        "status": "SEARCHED",
        "nuisance_segments": [1, 2],
        "preprocessing_arrays": preprocessing,
        "decisions": [
            {"seed_band": SEED, "seed_rank": 1, "status": "REJECTED", "period_trials": np.array([2.5, 2.6])},
            {"search_band": "baseline", "search_rank": 1, "status": "PROPOSAL", "period_days": 2.5},
            {"search_band": "baseline", "search_rank": 2, "status": "SHAPE_ACCEPTED", "period_days": 5.0},
        ],
        "physical_refinements": [refinement, {"seed_band": "other_band", "seed_rank": 2, "trials": []}],
    }
    conditioning = {
        "method": "global_fundamental",
        "contexts": [np.zeros(100)],
        "decisions": [{"retained": True, "arrays": {"big": np.zeros(400)}, "saved_best_trial": {"period_days": 2.5}}],
    }
    diagnostics = {
        "store_periodograms": True,
        "engine": {"name": "fake"},
        "bands": [coherence, physical],
        "global_fundamental_conditioning": conditioning,
        "inactive_native_prefix_bands": [coherence],
    }
    proposals = [{"period_days": 2.5, "depth_ppm": 300.0, "search_band": "baseline"}]
    return diagnostics, proposals


def test_trim_keeps_decisions_and_inlines_small_arrays():
    diagnostics, proposals = fake_result()
    result, arrays = API.separate_arrays({"diagnostics": diagnostics, "proposals": proposals})
    trimmed = API.trim_evidence(result, arrays)
    assert trimmed["proposals"] == proposals
    assert API.declared_array_names(trimmed) == set()
    coherence, physical = trimmed["diagnostics"]["bands"]
    assert "periodogram_arrays" not in coherence and "local_refinements" not in coherence
    assert coherence["tested_peaks"] and coherence["decisions"][0]["first_reason"] is None
    assert physical["decisions"][0]["period_trials"] == [2.5, 2.6]
    refinement = physical["physical_refinements"][0]
    assert "trials" not in refinement and "profile_arrays" not in refinement
    assert refinement["consistency"]["odd_even_covariance"] == [[1.0, 0.0], [0.0, 1.0]]
    assert "per_cycle" not in refinement["consistency"]
    assert "preprocessing_arrays" not in physical and "nuisance_segments" not in physical
    assert "inactive_native_prefix_bands" not in trimmed["diagnostics"]
    conditioning = trimmed["diagnostics"]["global_fundamental_conditioning"]
    assert "contexts" not in conditioning
    assert conditioning["decisions"] == [{"retained": True, "saved_best_trial": {"period_days": 2.5}}]


def test_proposal_evidence_keeps_seed_band_and_physical_preprocessing():
    diagnostics, proposals = fake_result()
    records = np.zeros(1200, dtype=RECORD)
    result, arrays = API.separate_arrays({"diagnostics": diagnostics, "proposals": proposals})
    evidence = API.proposal_evidence(result, arrays, records)
    expected = {"prepared_records"}
    expected.update(f"baseline_physical/{name}" for name, _ in API.PHYSICAL_EVIDENCE)
    expected.update(f"{SEED}/{name}" for name in API.SEED_EVIDENCE)
    assert set(evidence) == expected
    assert evidence["prepared_records"] is records
    assert evidence["baseline_physical/quarter"].dtype == np.int16
    assert evidence["baseline_physical/time"].dtype == np.float64
    assert evidence[f"{SEED}/search_snr"].dtype == np.float32
    assert np.allclose(evidence[f"{SEED}/period_days"], arrays["/diagnostics/bands/0/periodogram_arrays/period_days"])


def test_policy_outputs_trims_the_json_and_keeps_evidence_for_proposals(monkeypatch):
    diagnostics, proposals = fake_result()
    records = np.zeros(1200, dtype=RECORD)
    result, arrays = API.separate_arrays({"diagnostics": diagnostics, "proposals": proposals})
    monkeypatch.setattr(API, "ARRAY_POLICY", "proposals")
    trimmed, evidence = API.policy_outputs(result, arrays, records, proposals)
    assert trimmed["array_policy"] == "proposals" and API.declared_array_names(trimmed) == set()
    assert "baseline_physical/prepared_flux" in evidence and f"{SEED}/search_snr" in evidence
    assert API.policy_outputs(result, arrays, records, [])[1] is None
    monkeypatch.setattr(API, "ARRAY_POLICY", "all")
    full, kept = API.policy_outputs(result, arrays, records, proposals)
    assert full is result and kept["prepared_records"] is records
    assert "/diagnostics/bands/0/periodogram_arrays/period_days" in kept


def test_initialize_rejects_unknown_policy():
    try:
        API.initialize(Path("."), "evolve_native_coherence", {}, array_policy="some")
    except ValueError as error:
        assert "array policy" in str(error)
    else:
        raise AssertionError("unknown policy accepted")
