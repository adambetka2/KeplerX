import copy
import json

import pytest
from astropy.io import fits

from pipeline.deterministic import control_parent_candidates as candidates
from pipeline.deterministic.control_eligibility_archive import header_record
from pipeline.deterministic.control_parent_headers import verify_parent_headers
from pipeline.deterministic.test_control_parent_headers import original_headers


def quarter():
    return {
        "quarter": 1,
        "archive_header": {
            "sha256": "a" * 64,
            "cards": {
                "MODULE": 2,
                "OUTPUT": 3,
                "CHANNEL": 7,
                "TPF_ROW": 22,
                "TPF_COL": 11,
                "ROW": 22.5,
                "COLUMN": 11.5,
            },
        },
    }


def record():
    headers = original_headers()
    return {
        "status": "VERIFIED_PARENT_HEADERS",
        "quarter": 1,
        "parent_kic": "7",
        "headers": headers,
        "geometry": verify_parent_headers(headers, "7", 1),
    }


def aperture():
    return {"sha256": "b" * 64, "cards": {"NAXIS1": 2, "NAXIS2": 2}}


def selected():
    return {
        "source_id": "1",
        "cohort": "guide_star_family",
        "parent_tpf": "8",
        "pool": {"raw_fits_sha256": "c" * 64},
    }


def test_candidate_parent_preserves_original_parent_and_does_not_claim_uniqueness(
    tmp_path,
):
    path = tmp_path / "candidate.json"
    row = record()
    path.write_text(json.dumps(row))
    result = candidates.reconcile_candidates(
        selected(), quarter(), aperture(), [(path, row)]
    )
    assert result["original_global_parent"] == "8"
    assert result["matching_candidate_parents"] == ["7"]
    assert result["status"] == "ONE_MATCH_WITHIN_PUBLIC_CANDIDATE_SET"
    assert result["original_extraction_parent_uniqueness_verified"] is False
    assert result["complete_archive_neighbourhood"] is False
    assert result["physical_attribution_measured"] is False


def test_equal_corner_with_different_stamp_dimensions_does_not_match():
    changed = aperture()
    changed["cards"]["NAXIS1"] = 3
    result = candidates.exact_stamp_match(quarter(), changed, record())
    assert result["stamp_origin_and_channel_match"] is True
    assert result["exact_declared_geometry_match"] is False


def test_multiple_geometric_matches_remain_ambiguous(tmp_path):
    row = record()
    second = record()
    primary = fits.Header.fromstring(second["headers"][0]["raw_header_ascii"])
    primary["KEPLERID"] = 9
    second["headers"][0] = header_record(primary.tostring().encode("ascii"), 0)
    second["parent_kic"] = "9"
    second["geometry"] = verify_parent_headers(second["headers"], "9", 1)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(row))
    result = candidates.reconcile_candidates(
        selected(), quarter(), aperture(), [(path, row), (path, second)]
    )
    assert result["status"] == "AMBIGUOUS_MATCHING_PUBLIC_CANDIDATES"
    assert result["original_extraction_parent_uniqueness_verified"] is False


def test_repeated_candidate_parent_is_rejected(tmp_path):
    row = record()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Repeated candidate"):
        candidates.reconcile_candidates(
            selected(), quarter(), aperture(), [(path, row), (path, copy.deepcopy(row))]
        )


def test_all_unavailable_candidates_are_retained(tmp_path):
    row = {"status": "UNAVAILABLE", "quarter": 1, "parent_kic": "7"}
    path = tmp_path / "failed.json"
    path.write_text(json.dumps(row))
    result = candidates.reconcile_candidates(
        selected(), quarter(), aperture(), [(path, row)]
    )
    assert result["status"] == "NO_MATCHING_PUBLIC_CANDIDATE"
    assert (
        len(result["candidates"]) == 1
        and result["candidates"][0]["status"] == "UNAVAILABLE"
    )


def test_candidate_quarter_drift_rejected(tmp_path):
    row = record()
    row["quarter"] = 0
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Candidate quarter changed"):
        candidates.reconcile_candidates(
            selected(), quarter(), aperture(), [(path, row)]
        )


def test_candidate_geometry_drift_rejected(tmp_path):
    row = record()
    row["geometry"]["column_origin"] += 1
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="geometry changed"):
        candidates.reconcile_candidates(
            selected(), quarter(), aperture(), [(path, row)]
        )


def test_missing_aperture_dimensions_are_retained_as_unresolved(tmp_path):
    row = record()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(row))
    result = candidates.reconcile_candidates(selected(), quarter(), None, [(path, row)])
    assert result["status"] == "LC_APERTURE_HEADER_UNAVAILABLE"
    assert result["aperture_header_sha256"] is None
    assert result["matching_candidate_parents"] == []
    witness = result["candidates"][0]["witness"]
    assert witness["stamp_origin_and_channel_match"] is True
    assert witness["aperture_stamp_dimensions_match"] is None
    assert witness["exact_declared_geometry_match"] is False


def test_equal_origin_with_different_dimensions_requires_candidate_search(tmp_path):
    raw = record()
    path = tmp_path / "nominal.json"
    path.write_text(json.dumps(raw))
    source = selected()
    source["parent_tpf"] = "7"
    declared = {"cohort": source["cohort"], "record": candidates.reference(path)}
    larger = aperture()
    larger["cards"]["NAXIS2"] = 3
    assert candidates.nominal_exact_match(source, quarter(), larger, declared) is False
    assert (
        candidates.nominal_exact_match(source, quarter(), aperture(), declared) is True
    )


@pytest.mark.parametrize("change", ["parent", "quarter", "geometry", "hash"])
def test_nominal_search_input_drift_is_rejected(tmp_path, change):
    raw = record()
    path = tmp_path / "nominal.json"
    path.write_text(json.dumps(raw))
    source = selected()
    source["parent_tpf"] = "7"
    declared = {"cohort": source["cohort"], "record": candidates.reference(path)}
    if change == "parent":
        source["parent_tpf"] = "9"
    elif change == "quarter":
        raw["quarter"] = 2
    elif change == "geometry":
        raw["geometry"]["stamp_rows"] += 1
    else:
        path.write_text("changed")
    if change in {"quarter", "geometry"}:
        path.write_text(json.dumps(raw))
        declared["record"] = candidates.reference(path)
    with pytest.raises(ValueError):
        candidates.nominal_exact_match(source, quarter(), aperture(), declared)


def parent_record(parent):
    value = record()
    primary = fits.Header.fromstring(value["headers"][0]["raw_header_ascii"])
    primary["KEPLERID"] = parent
    value["headers"][0] = header_record(primary.tostring().encode("ascii"), 0)
    value["parent_kic"] = str(parent)
    value["geometry"] = verify_parent_headers(value["headers"], parent, 1)
    return value


def test_collector_checks_alternatives_even_when_nominal_stamp_matches(
    tmp_path, monkeypatch
):
    source = selected()
    source["parent_tpf"] = "7"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"sources": [source], "inputs": {"public_catalogue": {"path": "unused"}}}
        )
    )
    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps({"headers": [{"name": "APERTURE_Q1", **aperture()}]}))
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "archive_record": candidates.reference(archive),
                        "quarter_witnesses": [quarter()],
                    }
                ]
            }
        )
    )
    cache = tmp_path / "cache"
    (cache / "records").mkdir(parents=True)
    for parent in (7, 9):
        (cache / "records" / f"{parent}-q1.json").write_text(
            json.dumps(parent_record(parent))
        )
    (cache / "records/11-q1.json").write_text(
        json.dumps({"parent_kic": "11", "quarter": 1, "status": "UNAVAILABLE"})
    )
    nominal = tmp_path / "nominal.json"
    nominal.write_text(
        json.dumps(
            {
                "target_report": candidates.reference(target),
                "quarters": [
                    {
                        "source_id": "1",
                        "quarter": 1,
                        "cohort": source["cohort"],
                        "record": candidates.reference(cache / "records/7-q1.json"),
                    }
                ],
            }
        )
    )
    (tmp_path / "quarter-mapping.json").write_text(json.dumps({"1": "2009166043257"}))
    monkeypatch.setattr(candidates, "verify_plan", lambda *args: None)
    monkeypatch.setattr(candidates, "verify_targets", lambda *args: None)
    monkeypatch.setattr(candidates, "capacity", lambda *args: None)
    monkeypatch.setattr(
        candidates,
        "candidate_projection",
        lambda *args: {"1": {"candidate_parent_ids": [7, 9, 11]}},
    )
    monkeypatch.setattr(
        candidates,
        "candidate_record",
        lambda *args: pytest.fail("Verified retained headers must be reused"),
    )
    result = candidates.collect(plan, target, nominal, tmp_path / "result", cache)
    assert result["quarter_coverage"] == "ALL_ORIGINAL_OBSERVED_QUARTERS"
    assert len(result["quarters"]) == 1
    row = result["quarters"][0]
    assert row["status"] == "AMBIGUOUS_MATCHING_PUBLIC_CANDIDATES"
    assert row["matching_candidate_parents"] == ["7", "9"]
    assert [r["parent_kic"] for r in row["candidates"]] == ["7", "9", "11"]
    assert row["candidates"][-1]["status"] == "UNAVAILABLE"
    assert row["original_global_parent"] == "7"
    assert row["original_extraction_parent_uniqueness_verified"] is False
