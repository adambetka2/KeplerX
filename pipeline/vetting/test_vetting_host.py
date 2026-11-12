"""Host evidence retains constraints without expanding absent neighbour data."""

from vetting_host import host_constraints


def test_unconstrained_neighbour_is_summarized():
    host = {"source_id": "2", "radius": {"status": "STELLAR_RADIUS_UNAVAILABLE"}}
    result = host_constraints(host, ["a", "b"], {}, {}, None, {})
    assert result == {
        "source_id": "2",
        "status": "UNCONSTRAINED",
        "event_count": 2,
        "measurements": [],
        "scope": "No neighbour-specific stellar radius or flux fraction is available for an event-level host fit.",
    }


def test_supplied_neighbour_radius_keeps_event_measurement():
    host = {"source_id": "2", "radius": {"value_solar": 1.0}}
    event = {
        "event_id": "a",
        "photometry": "psf_flat",
        "fit": {"status": "MEASURED", "apparent_radius_ratio": 0.01},
    }
    result = host_constraints(host, ["a"], {"a": event}, {}, None, {})
    assert len(result["measurements"]) == 1
    assert result["measurements"][0]["absolute_radius_status"] == "UNCONSTRAINED"
    assert result["measurements"][0]["unit_dilution_radius_earth"] > 0
    assert result["measurements"][0]["dilution_constraints"] == {
        "status": "UNCONSTRAINED"
    }
