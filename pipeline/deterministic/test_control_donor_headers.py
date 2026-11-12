import copy
import json

import pytest

from pipeline.deterministic import control_donor_headers as donors
from pipeline.deterministic.control_bindings import reference


def binding():
    return {
        "source_id": "1",
        "parent_tpf": "7",
        "public_donor_product_identities": [{"gaia_designation": "Gaia DR3 2"}],
    }


def headers():
    return [
        {
            "name": "PRIMARY",
            "cards": {"KEPLERID": 7, "GAIAID": "Gaia DR3 2", "TPFORG": 8},
        },
        {"name": "LIGHTCURVE_Q0", "sha256": "a" * 64, "cards": {"QUARTER": 0}},
    ]


def test_donor_header_binds_distinct_counterpart_without_reading_flux():
    result = donors.bind_donor_header(binding(), headers())
    assert result["archive_gaia_source_id"] == "2"
    assert result["catalogued_kic"] == "7"
    assert result["global_extraction_parent"] == "8"
    assert result["quarters"][0]["quarter_parent"] is None
    assert result["donor_flux_read"] is False
    assert result["donor_raw_hash_reverified"] is False
    assert result["contamination_attributed"] is False


@pytest.mark.parametrize("field,value", [("KEPLERID", 9), ("GAIAID", "Gaia DR3 9")])
def test_donor_identity_mismatch_rejected(field, value):
    changed = headers()
    changed[0]["cards"][field] = value
    with pytest.raises(ValueError):
        donors.bind_donor_header(binding(), changed)


def test_same_target_donor_rejected():
    selected = binding()
    selected["source_id"] = "2"
    with pytest.raises(ValueError, match="same source"):
        donors.bind_donor_header(selected, headers())


def test_missing_public_donor_match_keeps_actual_archive_identity_explicit():
    selected = binding()
    selected["public_donor_product_identities"] = []
    assert donors.donor_product(selected)["product"]["kic"] == 7
    result = donors.bind_donor_header(selected, headers())
    assert result["public_catalogue_gaia_match"] is False
    assert result["archive_gaia_source_id"] == "2"


def test_ambiguous_donor_product_rejected_without_selection():
    selected = binding()
    selected["public_donor_product_identities"] *= 2
    with pytest.raises(ValueError, match="Ambiguous"):
        donors.donor_product(selected)


def test_only_common_quarters_are_bound_and_unknown_parents_stay_unknown():
    target_header = {
        "sha256": "b" * 64,
        "cards": {"MODULE": 2, "OUTPUT": 3, "CHANNEL": 7},
    }
    target = {
        "quarter_witnesses": [
            {"quarter": 0, "quarter_parent": None, "archive_header": target_header}
        ]
    }
    donor = {"headers": headers()}
    extra = copy.deepcopy(donor["headers"][1])
    extra["name"] = "LIGHTCURVE_Q1"
    extra["cards"]["QUARTER"] = 1
    donor["headers"].append(extra)
    rows = donors.quarter_association(target, donor)
    assert len(rows) == 1 and rows[0]["quarter"] == 0
    assert rows[0]["target_quarter_parent"] is None
    assert rows[0]["donor_quarter_parent"] is None
    assert rows[0]["attribution_measured"] is False


def test_post_assembly_archive_record_drift_is_rejected(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    record_path = tmp_path / "archive.json"
    record_path.write_text(
        json.dumps({"source": {"source_id": "1"}, "status": "UNAVAILABLE"})
    )
    report = {
        "plan": reference(plan_path),
        "boundary": donors.BOUNDARY,
        "sources": [{"source_id": "1", "archive_record": reference(record_path)}],
    }
    plan = {"sources": [{"source_id": "1"}]}
    donors.verify_targets(report, plan, plan_path)
    record_path.write_text(
        json.dumps({"source": {"source_id": "2"}, "status": "UNAVAILABLE"})
    )
    with pytest.raises(ValueError, match="Input unavailable or changed"):
        donors.verify_targets(report, plan, plan_path)


def test_post_assembly_target_inventory_change_is_rejected(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    report = {"plan": reference(plan_path), "boundary": donors.BOUNDARY, "sources": []}
    with pytest.raises(ValueError, match="inventory or order changed"):
        donors.verify_targets(report, {"sources": [{"source_id": "1"}]}, plan_path)
