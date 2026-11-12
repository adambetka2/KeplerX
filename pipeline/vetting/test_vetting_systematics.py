"""Parent-source comparisons cannot pass when sibling evidence is missing."""

import json
from pathlib import Path

from vetting_disposition import (
    catalogue_timing_conflicts,
    contamination_assessment,
    neighbour_comparison_complete,
)


def test_neighbour_comparisons_require_verified_parent_membership():
    signal = {"event_ids": ["event-1"]}
    rule = {"accepted_neighbour_statuses": ["MEASURED"]}
    evidence = {"neighbour_flux": []}
    assert not neighbour_comparison_complete(evidence, signal, rule)

    evidence["neighbour_comparisons"] = {
        "status": "MEASURED",
        "expected_sibling_ids": [],
        "selected_event_ids": ["event-1"],
    }
    assert neighbour_comparison_complete(evidence, signal, rule)

    evidence["neighbour_comparisons"]["expected_sibling_ids"] = ["sibling-1"]
    assert not neighbour_comparison_complete(evidence, signal, rule)

    evidence["neighbour_flux"] = [
        {"source_id": "sibling-1", "event_id": "event-1", "status": "MEASURED"}
    ]
    assert neighbour_comparison_complete(evidence, signal, rule)

    evidence["neighbour_comparisons"]["status"] = "INCOMPLETE_EXPERIMENTAL_INPUTS"
    assert not neighbour_comparison_complete(evidence, signal, rule)


def test_coincident_sibling_dip_needs_source_attribution():
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    policy = {
        "status": "RELEASED",
        "assessment_rules": config["triage"]["assessment_rules"],
    }
    signal = {
        "signal_id": "transit",
        "event_ids": ["event-1"],
        "events": [{"event_ids": ["event-1"]}],
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_timing": [{"event_id": "event-1", "matches": []}],
            "neighbour_flux": [
                {
                    "source_id": "sibling-1",
                    "event_id": "event-1",
                    "status": "MEASURED",
                    "depth_fraction": 0.02,
                    "depth_error": 0.001,
                }
            ],
            "neighbour_comparisons": {
                "status": "MEASURED",
                "expected_sibling_ids": ["sibling-1"],
                "selected_event_ids": ["event-1"],
            },
            "scene": {"status": "GAIA_SCENE_QUERIED"},
        },
    }
    row = {"source_id": "target", "signals": [signal]}

    coincident = contamination_assessment(row, signal, 0, policy)
    assert coincident["status"] == "UNASSESSED"

    signal["contamination"]["neighbour_flux"][0]["depth_fraction"] = 0.0
    clean = contamination_assessment(row, signal, 0, policy)
    assert clean["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"


def test_target_prf_resolves_missing_sibling_input_only_with_complete_scene():
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    policy = {
        "status": "RELEASED",
        "assessment_rules": config["triage"]["assessment_rules"],
    }
    signal = {
        "signal_id": "transit",
        "event_ids": ["event-1"],
        "events": [{"event_ids": ["event-1"]}],
        "localization": {
            "source_attribution": {"status": "TARGET_COMPATIBLE"},
            "scene_prf": {
                "status": "MEASURED_DIAGNOSTIC",
                "compared_quarters": 4,
                "target_preferred_quarters": 3,
            },
        },
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_timing": [{"event_id": "event-1", "matches": []}],
            "neighbour_flux": [],
            "neighbour_comparisons": {"status": "INCOMPLETE_EXPERIMENTAL_INPUTS"},
            "scene": {"status": "GAIA_SCENE_QUERIED"},
        },
    }
    row = {"source_id": "target", "signals": [signal]}

    assessed = contamination_assessment(row, signal, 0, policy)
    assert assessed["status"] == "ASSESSED_NO_MATERIAL_CONFLICT"
    assert len(assessed["evidence_refs"]) == 2

    signal["localization"]["scene_prf"]["target_preferred_quarters"] = 2
    assert contamination_assessment(row, signal, 0, policy)["status"] == "UNASSESSED"
    signal["localization"]["scene_prf"]["target_preferred_quarters"] = 3

    signal["contamination"]["scene"]["status"] = "UNAVAILABLE"
    assert contamination_assessment(row, signal, 0, policy)["status"] == "UNASSESSED"
    signal["contamination"]["scene"]["status"] = "GAIA_SCENE_QUERIED"

    signal["contamination"]["neighbour_flux"] = [
        {
            "source_id": "sibling-1",
            "event_id": "event-1",
            "status": "MEASURED",
            "depth_fraction": 0.02,
            "depth_error": 0.001,
        }
    ]
    assert contamination_assessment(row, signal, 0, policy)["status"] == "UNASSESSED"


def test_nearby_eb_timing_cannot_dismiss_target_compatible_planet():
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    rule = config["triage"]["assessment_rules"]["contamination"]
    policy = {"status": "RELEASED", "assessment_rules": {"contamination": rule}}
    event_ids = ["event-1", "event-2"]
    match = {
        "catalogue": "kepler-eb.csv",
        "source": "KIC 5980800",
        "separation_arcsec": 14.6,
        "known_period_days": 0.4141629,
        "overlaps_original_interval_nominally": True,
    }
    signal = {
        "signal_id": "planet-period-37d",
        "period_days": 37.3033197,
        "event_ids": event_ids,
        "events": [{"event_ids": [event_id]} for event_id in event_ids],
        "localization": {
            "source_attribution": {"status": "TARGET_COMPATIBLE"},
            "scene_prf": {
                "status": "MEASURED_DIAGNOSTIC",
                "compared_quarters": 14,
                "target_preferred_quarters": 10,
            },
        },
        "contamination": {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_timing": [
                {"event_id": event_id, "matches": [match]} for event_id in event_ids
            ],
            "neighbour_flux": [],
            "neighbour_comparisons": {"status": "INCOMPLETE_EXPERIMENTAL_INPUTS"},
            "scene": {"status": "GAIA_SCENE_QUERIED"},
            "mechanism_status": "NOT_ESTABLISHED",
        },
    }
    row = {"source_id": "target", "signals": [signal]}
    conflicts = catalogue_timing_conflicts(
        signal, signal["contamination"]["catalogue_timing"], rule
    )
    assert len(conflicts) == 1 and conflicts[0]["harmonic_order"] == 90
    assert contamination_assessment(row, signal, 0, policy)["status"] == "UNASSESSED"

    signal["contamination"]["mechanism_status"] = "ESTABLISHED"
    assert contamination_assessment(row, signal, 0, policy)["status"] == "MATERIAL_CONFLICT"
