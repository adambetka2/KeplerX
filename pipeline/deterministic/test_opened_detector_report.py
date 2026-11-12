"""Detector recovery preserves aliases and verifies its immutable dependencies."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.deterministic import opened_detector_report as report
from pipeline.deterministic import run as runner


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def csv_file(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def alert(
    identity,
    centre,
    signal="native",
    origin="rust_v6_deterministic",
    view="psf_flat",
    quarter=1,
):
    return {
        "event_id": identity,
        "detector_signal_id": signal,
        "detector_origin": origin,
        "quarter": quarter,
        "photometry": view,
        "start_bkjd": centre - 0.04,
        "end_bkjd": centre + 0.04,
    }


@pytest.fixture
def closed_run(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / "inputs").mkdir()
    (out / "proposals").mkdir()
    records = np.zeros(80, dtype=runner.RECORD)
    records["time"] = np.concatenate(
        [np.linspace(c - 0.05, c + 0.05, 20) for c in (1, 3, 5, 7)]
    )
    records["quarter"] = 1
    source = {"source_id": "123", "inputs": {}, "views": [], "parent_tpf": "45"}
    for view in runner.DETECTOR_VIEWS:
        path = out / "inputs" / f"123.{view}.fsc"
        runner.write_fsc(path, "123", view, records)
        source["inputs"][view] = {
            "fsc_path": str(path),
            "fsc_sha256": report.cloud.digest(path),
        }
        source["views"].append({"view": view, "status": "SCORED"})
    preparation = {
        "status": "PREPARED",
        "selection": {"source_ids": ["123"]},
        "sources": [source],
        "implementation": report.implementation_hashes(),
        "detector_contract": {},
        "disposition_policy": str(tmp_path / "policy.json"),
    }
    write(tmp_path / "policy.json", {"status": "DEVELOPMENT"})
    preparation["disposition_policy_sha256"] = report.cloud.digest(
        tmp_path / "policy.json"
    )
    write(out / "preparation.json", preparation)
    eligibility = {
        "source_id": "123",
        "status": "ELIGIBLE",
        "reason": "",
        "elapsed_seconds": "0",
    }
    supplied = [alert("a", 1), alert("b", 3)]
    proposal = {
        "source_id": "123",
        "views": source["views"],
        "alerts": supplied,
        "ungated_alerts": supplied,
        "rust_rows": [{"source_id": "123", "classification": "TRANSIT_LIKE"}],
        "shallow_rows": [],
        "shallow_eligibility": eligibility,
    }
    ledger = [
        {
            "source_id": "123",
            "status": "PROPOSALS",
            "raw_rows": 1,
            "shallow_rows": 0,
            "alerts": 2,
            "shallow_eligibility": eligibility,
        }
    ]
    write(out / "proposals/123.json", proposal)
    write(out / "source-ledger.json", ledger)
    write(
        out / "queue.json",
        {"rows": [{"source_id": "123", "evidence": "proposals/123.json"}]},
    )
    csv_file(out / "raw.csv", proposal["rust_rows"])
    csv_file(out / "shallow-raw.csv", [], ["source_id"])
    csv_file(out / "shallow-eligibility.csv", [eligibility])
    binary = tmp_path / "detector"
    binary.write_bytes(b"frozen fixture detector")
    binary_identity = {"path": str(binary), "sha256": report.cloud.digest(binary)}
    write(
        out / "detection.json",
        {
            "status": "DETECTION_COMPLETE",
            "sources": 1,
            "raw_rows": 1,
            "shallow_rows": 0,
            "binary": binary_identity,
        },
    )
    truth = {
        "source_id": "123",
        "category": "periodic_injection",
        "duration_hours": 2.4,
        "epoch_bkjd": 1,
        "shape_period_days": 2,
    }
    truth_path = tmp_path / "truth.jsonl"
    write(truth_path, truth)
    references = {
        "123": {
            "cohort": "periodic_planet",
            "path": str(truth_path),
            "truth_index": 0,
            "file_sha256": report.cloud.digest(truth_path),
            "line_sha256": report.cloud.digest(truth_path),
        }
    }
    dependencies = {truth_path: report.cloud.digest(truth_path)}
    monkeypatch.setattr(report, "dump_path", lambda path: Path(path))
    monkeypatch.setattr(
        report.cloud_opened, "truth_refs", lambda run: (references, dict(dependencies))
    )
    case = SimpleNamespace(
        out=out,
        proposal=proposal,
        preparation=preparation,
        source=source,
        ledger=ledger,
        eligibility=eligibility,
        truth=truth,
        truth_path=truth_path,
        references=references,
        dependencies=dependencies,
        binary=binary_identity,
    )

    def bind_artifacts():
        paths = [
            out / name
            for name in (
                "preparation.json",
                "raw.csv",
                "shallow-raw.csv",
                "shallow-eligibility.csv",
                "queue.json",
                "source-ledger.json",
                "proposals/123.json",
            )
        ]
        write(
            out / "discovery-manifest.json",
            {
                "artifacts": {str(path): report.cloud.digest(path) for path in paths},
                "binary": case.binary,
                "implementation": case.preparation["implementation"],
                "detector_contract": {},
            },
        )

    def refresh():
        write(out / "proposals/123.json", case.proposal)
        case.ledger[0]["alerts"] = len(case.proposal["alerts"])
        case.ledger[0]["status"] = (
            "PROPOSALS" if case.proposal["alerts"] else "DETECTOR_REJECTION"
        )
        write(out / "source-ledger.json", case.ledger)
        rows = (
            [{"source_id": "123", "evidence": "proposals/123.json"}]
            if case.proposal["alerts"]
            else []
        )
        write(
            out / "queue.json",
            {"discovery_manifest": "discovery-manifest.json", "rows": rows},
        )
        write(
            out / "detection.json",
            {
                "status": "DETECTION_COMPLETE",
                "sources": 1,
                "raw_rows": 1,
                "shallow_rows": 0,
                "binary": case.binary,
                "alerts": len(case.proposal["alerts"]),
                "alerted_sources": len(rows),
            },
        )
        bind_artifacts()

    case.refresh = refresh
    case.bind_artifacts = bind_artifacts
    refresh()
    return case


def test_harmonic_subset_has_complete_group_and_partial_physical_coverage(closed_run):
    value = report.build_report(closed_run.out)
    case = value["cases"][0]
    endpoint = case["endpoints"]["ungated_alerts"]
    assert endpoint["any_completely_matched_group"]
    assert endpoint["groups"][0]["event_match"]["status"] == "MATCHED"
    assert endpoint["groups"][0]["event_match"]["truth_event_coverage"] == 0.5
    assert endpoint["psf_flat_event_coverage"]["supported_truth_events"] == 4
    assert endpoint["psf_flat_event_coverage"]["matched_truth_events"] == 2
    assert endpoint["psf_flat_event_coverage"]["fraction"] == 0.5
    assert case["all_view_truth_event_ledger"][0]["support"]["1"].keys() == set(
        runner.DETECTOR_VIEWS
    )
    assert value["population"]["fixed_injected_sources"] == 180
    assert value["population"]["original_opened_sources"] == 876
    assert value["population"]["unmeasured_fixed_injected_sources"] == 179
    assert value["downstream"] == {
        name: "NOT_MEASURED"
        for name in ("triage_retention", "fitted_timing", "pixels", "final_disposition")
    }
    assert value["original_opened_inventory_counts"]["protected"] == 5
    assert (
        value["original_opened_inventory_counts"]["background_recipe_unavailable"]
        == 100
    )
    assert not any(value[flag] for flag in report.cloud_opened.FLAGS)


def test_false_partial_unit_does_not_make_a_recovered_source(closed_run):
    closed_run.proposal["ungated_alerts"].append(alert("false", 6))
    closed_run.refresh()
    measured = report.build_report(closed_run.out)["cases"][0]["endpoints"]["alerts"]
    assert not measured["any_completely_matched_group"]
    assert measured["groups"][0]["event_match"]["matched_units"] == 2
    assert measured["groups"][0]["event_match"]["supplied_units"] == 3
    assert measured["psf_flat_event_coverage"]["matched_truth_events"] == 2


def test_views_detectors_and_aliases_do_not_multiply_physical_events(closed_run):
    alerts = closed_run.proposal["ungated_alerts"]
    alerts.extend(
        [
            alert("psf", 1, view="psf_qdiag"),
            alert("alias", 1, signal="alias"),
            alert("shallow", 3, signal="native", origin="shallow_bls_v1"),
        ]
    )
    closed_run.refresh()
    endpoint = report.build_report(closed_run.out)["cases"][0]["endpoints"][
        "ungated_alerts"
    ]
    assert endpoint["alert_count"] == 5 and endpoint["group_count"] == 3
    assert endpoint["completely_matched_groups"] == 3
    assert len(endpoint["groups"][0]["units"]) == 2
    assert endpoint["psf_flat_event_coverage"]["matched_truth_events"] == 2
    assert len(endpoint["groups"][0]["units"][0]["event_ids"]) == 2
    assert endpoint["alert_order"] == [item["event_id"] for item in alerts]
    assert (
        endpoint["groups"][0]["detector_signal_id"]
        == endpoint["groups"][2]["detector_signal_id"]
    )
    assert (
        endpoint["groups"][0]["detector_origin"]
        != endpoint["groups"][2]["detector_origin"]
    )


def test_ungated_recovery_is_separate_from_the_queue(closed_run):
    closed_run.proposal["alerts"] = []
    closed_run.refresh()
    value = report.build_report(closed_run.out)
    assert (
        value["summary"]["endpoints"]["ungated_alerts"][
            "any_completely_matched_group_sources"
        ]
        == 1
    )
    assert (
        value["summary"]["endpoints"]["alerts"]["any_completely_matched_group_sources"]
        == 0
    )
    assert value["summary"]["endpoints"]["alerts"]["alert_count"] == 0


def test_detector_rejection_and_ineligible_source_remain_in_denominator(closed_run):
    closed_run.proposal.update(alerts=[], ungated_alerts=[])
    closed_run.eligibility.update(status="INELIGIBLE", reason="fixed input coverage")
    csv_file(closed_run.out / "shallow-eligibility.csv", [closed_run.eligibility])
    closed_run.ledger[0]["status"] = "DETECTOR_REJECTION"
    closed_run.refresh()
    value = report.build_report(closed_run.out)
    assert value["population"]["measured_sources"] == 1
    assert value["summary"]["shallow_eligibility"] == {"INELIGIBLE": 1}
    assert value["summary"]["detector_source_status"] == {"DETECTOR_REJECTION": 1}
    assert (
        value["cases"][0]["endpoints"]["alerts"]["psf_flat_event_coverage"]["fraction"]
        == 0
    )


@pytest.mark.parametrize(
    "mutation", ["missing_event", "duplicate_event", "origin", "signal"]
)
def test_missing_or_duplicate_grouping_data_fails(closed_run, mutation):
    item = closed_run.proposal["ungated_alerts"][0]
    if mutation == "missing_event":
        item.pop("event_id")
    elif mutation == "duplicate_event":
        closed_run.proposal["ungated_alerts"].append(dict(item))
    else:
        item.pop("detector_" + ("origin" if mutation == "origin" else "signal_id"))
    closed_run.refresh()
    with pytest.raises(ValueError, match="event IDs|grouping identity"):
        report.build_report(closed_run.out)


def test_quarter_mismatch_does_not_match_a_truth_event(closed_run):
    for item in closed_run.proposal["alerts"]:
        item["quarter"] = 2
    closed_run.refresh()
    endpoint = report.build_report(closed_run.out)["cases"][0]["endpoints"]["alerts"]
    assert not endpoint["any_completely_matched_group"]
    assert endpoint["psf_flat_event_coverage"]["matched_truth_events"] == 0


def test_match_in_another_view_does_not_supply_psf_flat_quarter_coverage():
    events = [
        {
            "event_id": "truth",
            "centre_bkjd": 1,
            "support": {"1": {"psf": 3}, "2": {"psf_flat": 4}},
        }
    ]
    views = [{"view": view, "status": "SCORED"} for view in ("psf", "psf_flat")]
    measured = report.detector_endpoint([alert("a", 1)], views, events, 2.4)
    assert measured["any_completely_matched_group"]
    assert measured["psf_flat_event_coverage"]["supported_truth_events"] == 1
    assert measured["psf_flat_event_coverage"]["matched_truth_events"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "truth_source",
        "truth_bytes",
        "truth_cohort",
        "truth_index",
        "fsc",
        "fsc_source",
        "code",
        "zero_refs",
    ],
)
def test_wrong_source_truth_fsc_or_code_cannot_produce_a_report(closed_run, mutation):
    if mutation == "source":
        closed_run.proposal["source_id"] = "456"
        closed_run.refresh()
    elif mutation == "truth_source":
        closed_run.truth["source_id"] = "456"
        write(closed_run.truth_path, closed_run.truth)
        expected = report.cloud.digest(closed_run.truth_path)
        closed_run.references["123"].update(file_sha256=expected, line_sha256=expected)
        closed_run.dependencies[closed_run.truth_path] = expected
    elif mutation == "truth_bytes":
        closed_run.truth_path.write_text("{}\n")
    elif mutation == "truth_cohort":
        closed_run.references["123"]["cohort"] = "ttv_injection"
    elif mutation == "truth_index":
        closed_run.references["123"]["truth_index"] = 1
    elif mutation == "fsc":
        path = Path(closed_run.source["inputs"]["psf_flat"]["fsc_path"])
        path.write_bytes(path.read_bytes()[:-1])
    elif mutation == "fsc_source":
        identity = closed_run.source["inputs"]["psf_flat"]
        path = Path(identity["fsc_path"])
        payload = path.read_bytes()
        path.write_bytes(payload[:8] + (999).to_bytes(8, "little") + payload[16:])
        identity["fsc_sha256"] = report.cloud.digest(path)
        write(closed_run.out / "preparation.json", closed_run.preparation)
        closed_run.refresh()
    elif mutation == "code":
        manifest = report.cloud.read(closed_run.out / "discovery-manifest.json")
        manifest["implementation"] = {}
        write(closed_run.out / "discovery-manifest.json", manifest)
    else:
        closed_run.references.clear()
    output = closed_run.out / "detector-report.json"
    with pytest.raises(ValueError):
        report.write_report(closed_run.out, output)
    assert not output.exists()


def test_queue_cannot_reassign_an_emitted_alert(closed_run):
    closed_run.proposal["alerts"] = copy.deepcopy(closed_run.proposal["alerts"])
    closed_run.proposal["alerts"][0]["detector_signal_id"] = "new-alias"
    closed_run.refresh()
    with pytest.raises(ValueError, match="unchanged ungated"):
        report.build_report(closed_run.out)


@pytest.mark.parametrize(
    "mutation", ["extra_source", "wrong_path", "missing_source", "count"]
)
def test_frozen_queue_and_detection_counts_are_reconciled(closed_run, mutation):
    path = closed_run.out / "queue.json"
    queue = report.cloud.read(path)
    if mutation == "extra_source":
        queue["rows"].append({"source_id": "456", "evidence": "proposals/456.json"})
    elif mutation == "wrong_path":
        queue["rows"][0]["evidence"] = "another.json"
    elif mutation == "missing_source":
        queue["rows"] = []
    else:
        detection = report.cloud.read(closed_run.out / "detection.json")
        detection["alerts"] = 0
        write(closed_run.out / "detection.json", detection)
    write(path, queue)
    closed_run.bind_artifacts()
    with pytest.raises(ValueError, match="Detector queue"):
        report.build_report(closed_run.out)


@pytest.mark.parametrize("ids", [["a", "a"], ["a", ""]])
def test_truth_event_ids_cannot_be_duplicate_or_missing(ids):
    events = [
        {"event_id": identity, "centre_bkjd": centre, "support": {"1": {"psf_flat": 1}}}
        for identity, centre in zip(ids, (1, 3), strict=True)
    ]
    with pytest.raises(ValueError, match="Truth event identities"):
        report.detector_endpoint(
            [alert("proposal", 1)],
            [{"view": "psf_flat", "status": "SCORED"}],
            events,
            2.4,
        )


def test_two_disjoint_units_matching_one_truth_event_do_not_make_a_complete_group():
    events = [
        {"event_id": "truth", "centre_bkjd": 1, "support": {"1": {"psf_flat": 3}}}
    ]
    alerts = [alert("a", 0.98), alert("b", 1.02)]
    for item in alerts:
        centre = (item["start_bkjd"] + item["end_bkjd"]) / 2
        item.update(start_bkjd=centre - 0.01, end_bkjd=centre + 0.01)
    value = report.detector_endpoint(
        alerts, [{"view": "psf_flat", "status": "SCORED"}], events, 2.4
    )
    assert not value["any_completely_matched_group"]
    assert value["groups"][0]["event_match"]["matched_units"] == 2
    assert value["groups"][0]["event_match"]["unique_matched_truth_events"] == 1
    assert value["psf_flat_event_coverage"]["matched_truth_events"] == 1


def test_dependency_drift_during_matching_is_rejected(closed_run, monkeypatch):
    original = report.detector_endpoint

    def drift(*args):
        result = original(*args)
        closed_run.truth_path.write_text("{}\n")
        return result

    monkeypatch.setattr(report, "detector_endpoint", drift)
    with pytest.raises(ValueError, match="dependency changed"):
        report.build_report(closed_run.out)


@pytest.mark.parametrize("stage", ["partial", "published"])
def test_output_verification_rejects_dependency_drift_and_retains_partial(
    closed_run, monkeypatch, stage
):
    original = report.cloud_opened.recheck
    output = closed_run.out / "detector-report.json"

    def drift(dependencies):
        target = (
            output.with_name(output.name + ".partial") if stage == "partial" else output
        )
        if target in dependencies:
            closed_run.truth_path.write_text("{}\n")
        original(dependencies)

    monkeypatch.setattr(report.cloud_opened, "recheck", drift)
    with pytest.raises(ValueError, match="dependency changed"):
        report.write_report(closed_run.out, output)
    assert not output.exists() and output.with_name(output.name + ".partial").is_file()


def test_cli_writes_a_hash_bound_complete_report(closed_run, monkeypatch):
    output = closed_run.out / "detector-report.json"
    monkeypatch.setattr(
        report.sys,
        "argv",
        ["opened_detector_report", "--run", str(closed_run.out), "--out", str(output)],
    )
    assert report.main() == 0
    value = report.cloud.read(output)
    assert len(value["cases"]) == 1 and not value["scientific_acceptance"]
    for name, identity in value["dependencies"].items():
        assert Path(name).stat().st_size == identity["bytes"]
        assert report.cloud.digest(Path(name)) == identity["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        report.write_report(closed_run.out, output)
