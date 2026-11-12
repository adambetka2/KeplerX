import pytest

from pipeline.deterministic import holdout_controls as holdout


def graph_row(role, **values):
    row = dict.fromkeys((
        "source_id", "contaminant_source_id", "raw_fits_sha256",
        "contaminant_raw_fits_sha256", "origin_parent", "resolved_parents",
        "spatial_pixel_parent_by_quarter",
    ), "")
    return {**row, "role": role, **values}


def test_opened_identities_cover_every_parent_edge():
    graph = [
        graph_row(
            "development", source_id="a", contaminant_source_id="c",
            raw_fits_sha256="ra", origin_parent="1", resolved_parents="1;2",
            spatial_pixel_parent_by_quarter="0:3;1:4",
        ),
        graph_row("development", source_id="b", origin_parent="8"),
        graph_row("evaluation", source_id="e", origin_parent="9"),
    ]
    signals = [
        {"split": "development", "source_id": "d", "kic_id": "5"},
        {"split": "holdout", "source_id": "h", "kic_id": "6"},
    ]
    published = [{"source_id": "p", "parent_tpf": "7"}]
    catalogue = {"d": {"originating_tpf": "10"}}
    opened = holdout.opened_identities(graph, signals, published, catalogue)
    assert opened == {
        "source_id": {"a", "b", "c", "d", "p"},
        "kic_id": {"1", "2", "3", "4", "5", "7", "8", "10"},
        "fits_sha256": {"ra"},
    }
    host = {"source_id": {"h"}, "kic_id": {"6", "3"}, "fits_sha256": {"rh"}}
    assert holdout.conflicts(host, opened) == ["kic_id"]
    assert holdout.conflicts({**host, "kic_id": {"6", "9"}}, opened) == []


def test_holdout_selection_requires_the_declared_signals():
    sealed = {"split": "holdout", "input_eligible": "true", "in_domain": "true"}
    rows = [sealed] * holdout.HOLDOUT_SIGNALS + [
        {**sealed, "split": "development"},
        {**sealed, "in_domain": "false"},
    ]
    assert len(holdout.holdout_signals(rows)) == holdout.HOLDOUT_SIGNALS
    with pytest.raises(ValueError, match="686"):
        holdout.holdout_signals(rows[1:])
