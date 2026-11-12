"""Exact local-cohort coincidence bounds and query identity checks."""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [
    str(Path(__file__).parent),
    str(Path(__file__).parents[1] / "photometry"),
    str(Path(__file__).parents[1]),
]
import vetting
import vetting_evidence as evidence
from vetting_evidence import alert_time_bounds, coincident_sources


def test_evidence_worker_opens_its_own_readonly_survey_index(tmp_path, monkeypatch):
    index = tmp_path / "alerts.sqlite"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE marker (value INTEGER)")
        connection.execute("INSERT INTO marker VALUES (1)")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"shared_alert_index": {
        "path": str(index), "sha256": vetting.digest(index)
    }}))
    monkeypatch.setattr(evidence, "CONFIG", config)
    monkeypatch.setattr(evidence, "EVIDENCE_CONTEXT", ())
    parent, times, complete = evidence.shared_alerts()
    try:
        evidence.init_evidence_worker({}, [], {}, None, times, complete)
        child = evidence.EVIDENCE_CONTEXT[3]
        try:
            assert child is not parent
            assert child.execute("SELECT value FROM marker").fetchone() == (1,)
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                child.execute("INSERT INTO marker VALUES (2)")
            assert evidence.EVIDENCE_CONTEXT[-1] is True
        finally:
            child.close()
    finally:
        parent.close()


def reference(records, start, end, sid, parent):
    return {
        source: other_parent == parent
        for begin, stop, source, other_parent in records
        if begin <= end and stop >= start and source != sid
    }


def test_bounds_preserve_touching_intervals_and_duplicate_source_overwrite():
    records = sorted(
        [
            (-1.0, 1.0, "own", 1),
            (0.0, 0.5, "outside", 1),
            (0.0, 10.0, "long", 2),
            (1.0, 1.0, "point", 1),
            (1.0, 2.0, "repeated", 2),
            (1.5, 2.0, "repeated", 1),
            (2.0, 2.1, "edge", 1),
            (2.1, 3.0, "late", 2),
        ]
    )
    bounds = alert_time_bounds(records)
    result = coincident_sources(records, bounds, 1.0, 2.0, "own", 1)
    assert result == reference(records, 1.0, 2.0, "own", 1)
    assert result == {"long": False, "point": True, "repeated": True, "edge": True}


@pytest.mark.parametrize("long_window", [False, True])
def test_bounds_equal_full_scan_for_dense_randomized_cohorts(long_window):
    rng = np.random.default_rng(20260927)
    starts = rng.uniform(-100, 2000, 4000)
    stops = starts + rng.uniform(0, 4, len(starts))
    records = [
        (float(start), float(stop), str(index % 37), index % 4)
        for index, (start, stop) in enumerate(zip(starts, stops))
    ]
    if long_window:
        records.append((-200.0, 3000.0, "persistent", 2))
    records.sort()
    bounds = alert_time_bounds(records)
    for start, width in zip(rng.uniform(-200, 2100, 200), rng.uniform(0, 8, 200)):
        assert coincident_sources(
            records, bounds, start, start + width, "12", 2
        ) == reference(records, start, start + width, "12", 2)


def test_empty_cohort_has_no_coincident_sources():
    assert coincident_sources([], alert_time_bounds([]), 1, 2, "own", 1) == {}


@pytest.mark.parametrize(
    "windows", [[(2, 1)], [(np.nan, 2)], [(1, np.inf)], [(2, 3), (1, 2)]]
)
def test_invalid_interval_identities_fail_clearly(windows):
    with pytest.raises(ValueError, match="finite, ordered intervals"):
        alert_time_bounds([(start, stop, "1", 1) for start, stop in windows])


def test_cached_query_requires_exact_endpoint_and_query(tmp_path, monkeypatch):
    run = tmp_path / "run"
    queries = run / "queries"
    queries.mkdir(parents=True)
    record = {
        "endpoint": "https://example.org/tap/sync",
        "query": "select source_id from gaia_source",
        "status": "COMPLETE",
        "rows": [],
    }
    (queries / "gaia-000.json").write_text(json.dumps(record))
    monkeypatch.setattr(vetting, "RUN", run)

    assert vetting.tap("gaia-000", record["endpoint"], record["query"]) == record
    with pytest.raises(ValueError, match="Cached query identity changed"):
        vetting.tap("gaia-000", "https://other.example.org/tap/sync", record["query"])
    with pytest.raises(ValueError, match="Cached query identity changed"):
        vetting.tap("gaia-000", record["endpoint"], "select source_id from other")


def test_failed_tap_query_retries_and_reads_json_object_rows(tmp_path, monkeypatch):
    run = tmp_path / "run"
    queries = run / "queries"
    queries.mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"network": True, "archive_timeout_seconds": 5}))
    endpoint = "https://example.org/tap/sync"
    query = "select source_id from gaiadr3.gaia_source"
    (queries / "gaia-000.json").write_text(json.dumps({
        "endpoint": endpoint,
        "query": query,
        "status": "QUERY_FAILED",
        "error": "ReadTimeout",
        "retrieved_utc": "2026-10-03T00:00:00Z",
        "rows": [],
    }))

    class Response:
        content = b"{}"

        def raise_for_status(self):
            pass

        def json(self):
            return {"metadata": [{"name": "source_id"}], "data": [{"source_id": 123}]}

    monkeypatch.setattr(vetting, "CONFIG", config)
    monkeypatch.setattr(vetting, "RUN", run)
    monkeypatch.setattr(vetting.requests, "get", lambda *args, **kwargs: Response())
    result = vetting.tap("gaia-000", endpoint, query)
    assert result["status"] == "COMPLETE"
    assert result["rows"] == [{"source_id": 123}]
    assert result["previous_failure"]["error"] == "ReadTimeout"
    assert vetting.tap("gaia-000", endpoint, query) == result


def test_each_source_uses_only_its_own_scene_batch(tmp_path, monkeypatch):
    run = tmp_path / "run"
    queries = run / "queries"
    queries.mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"neighbour_radius_arcsec": 60}))
    (queries / "scene-000.json").write_text(json.dumps({
        "status": "COMPLETE",
        "rows": [
            {"source_id": 1, "ra": 10, "dec": 20},
            {"source_id": 3, "ra": 10.01, "dec": 20},
        ],
    }))
    (queries / "scene-010.json").write_text(json.dumps({
        "status": "QUERY_FAILED", "rows": [],
    }))
    cohort = [{"source_id": str(index)} for index in range(1, 12)]
    monkeypatch.setattr(evidence, "CONFIG", config)
    monkeypatch.setattr(evidence, "RUN", run)
    per_source = evidence.scene_queries_for_sources(cohort)
    first = evidence.source_scene({"source_id": "1", "ra_deg": 10, "dec_deg": 20}, per_source["1"])
    second = evidence.source_scene({"source_id": "11", "ra_deg": 30, "dec_deg": 40}, per_source["11"])
    assert first["status"] == "GAIA_SCENE_QUERIED"
    assert [row["source_id"] for row in first["sources"]] == ["3"]
    assert second["status"] == "PARTIAL_QUERY_COVERAGE"


def fake_tap(responses, calls):
    def tap(name, endpoint, query):
        calls.append((name, query))
        return {
            "endpoint": endpoint,
            "query": query,
            **responses[name.rsplit("-", 1)[0]],
        }

    return tap


def test_batched_tic_records_follow_the_reciprocal_identity_chain(monkeypatch):
    responses = {
        "stellar-dr3": {
            "status": "COMPLETE",
            "rows": [
                {"dr3_source_id": 1, "dr2_source_id": 11},
                {"dr3_source_id": 2, "dr2_source_id": 21},
                {"dr3_source_id": 2, "dr2_source_id": 22},
                {"dr3_source_id": 3, "dr2_source_id": 33},
                {"dr3_source_id": 5, "dr2_source_id": 55},
            ],
        },
        "stellar-dr2": {
            "status": "COMPLETE",
            "rows": [
                {"dr3_source_id": 1, "dr2_source_id": 11},
                {"dr3_source_id": 3, "dr2_source_id": 33},
                {"dr3_source_id": 4, "dr2_source_id": 33},
                {"dr3_source_id": 5, "dr2_source_id": 55},
            ],
        },
        "stellar-tic": {
            "status": "COMPLETE",
            "rows": [
                {"TIC": 100, "GAIA": 11, "Disp": None},
                {"TIC": 101, "GAIA": 11, "Disp": "DUPLICATE"},
                {"TIC": 500, "GAIA": 55, "Disp": None},
                {"TIC": 501, "GAIA": 55, "Disp": None},
            ],
        },
    }
    calls = []
    monkeypatch.setattr(evidence, "tap", fake_tap(responses, calls))
    records = evidence.tic_stellar_records(0, ["1", "2", "3", "5", "6"])
    assert [name for name, _ in calls] == [
        "stellar-dr3-0000",
        "stellar-dr2-0000",
        "stellar-tic-0000",
    ]
    assert "dr3_source_id in (1,2,3,5,6)" in calls[0][1]
    assert "dr2_source_id in (11,33,55)" in calls[1][1]
    assert '"GAIA" in (11,55)' in calls[2][1]
    assert records["1"]["status"] == "TIC_STELLAR_RECORD_IDENTIFIED"
    assert records["1"]["record"]["TIC"] == 100
    assert records["1"]["gaia_dr2_id"] == "11"
    unresolved = {"status": "GAIA_RELEASE_IDENTITY_UNRESOLVED"}
    assert records["2"] == unresolved and records["3"] == unresolved
    assert records["6"] == unresolved
    assert records["5"]["status"] == "TIC_IDENTITY_UNRESOLVED"
    assert [row["TIC"] for row in records["5"]["records"]] == [500, 501]


def test_batched_tic_records_report_incomplete_queries(monkeypatch):
    calls = []
    unresolved = {"status": "GAIA_RELEASE_IDENTITY_UNRESOLVED"}
    failed = {"stellar-dr3": {"status": "QUERY_FAILED", "rows": []}}
    monkeypatch.setattr(evidence, "tap", fake_tap(failed, calls))
    assert evidence.tic_stellar_records(50, ["7"]) == {"7": unresolved}
    assert [name for name, _ in calls] == ["stellar-dr3-0050"]
    responses = {
        "stellar-dr3": {
            "status": "COMPLETE",
            "rows": [{"dr3_source_id": 7, "dr2_source_id": 77}],
        },
        "stellar-dr2": {
            "status": "COMPLETE",
            "rows": [{"dr3_source_id": 7, "dr2_source_id": 77}],
        },
        "stellar-tic": {"status": "NETWORK_DISABLED", "rows": []},
    }
    monkeypatch.setattr(evidence, "tap", fake_tap(responses, calls))
    incomplete = {"status": "TIC_QUERY_INCOMPLETE"}
    assert evidence.tic_stellar_records(50, ["7"]) == {"7": incomplete}


def test_prefetch_resolves_only_sources_without_flame_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "RUN", tmp_path)
    complete = {"radius_flame": 1.0, "mass_flame": 1.0, "teff_gspphot": 5000.0}
    stellar = {"1": complete, "2": {"radius_flame": None}, "3": {}}
    monkeypatch.setattr(evidence, "source_stellar", stellar.__getitem__)
    batches = []

    def records(index, sids):
        batches.append((index, sids))
        return {sid: {"status": "TIC_QUERY_INCOMPLETE", "sid": sid} for sid in sids}

    monkeypatch.setattr(evidence, "tic_stellar_records", records)
    evidence.prefetch_stellar([{"source_id": sid} for sid in ("1", "2", "3")])
    assert batches == [(0, ["2", "3"])]
    written = sorted(path.name for path in (tmp_path / "stellar").glob("*.json"))
    assert written == ["2.json", "3.json"]
    row = dict(stellar["2"])
    evidence.complete_stellar(row, "2")
    assert row["tic_parameters"] == {"status": "TIC_QUERY_INCOMPLETE", "sid": "2"}
    row = dict(complete)
    evidence.complete_stellar(row, "1")
    assert "tic_parameters" not in row


def test_digest_reads_every_byte_without_whole_file_allocation(tmp_path, monkeypatch):
    import hashlib

    path = tmp_path / "index.sqlite"
    payload = bytes(range(256)) * 3072 + b"tail"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    with monkeypatch.context() as patch:
        def forbid_whole_file_read(path):
            pytest.fail("Integrity hashing allocated the whole file")

        patch.setattr(Path, "read_bytes", forbid_whole_file_read)
        assert vetting.digest(path) == expected
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"X")
        assert vetting.digest(path) == hashlib.sha256(payload[:-1] + b"X").hexdigest()
        assert vetting.digest(path) != expected
    path.write_bytes(b"")
    assert vetting.digest(path) == hashlib.sha256(b"").hexdigest()
