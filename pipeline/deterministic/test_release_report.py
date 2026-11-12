"""Release assembly must retain aliases and reject unproven final inputs."""

import csv
import importlib
import json
import sys

import pytest

from pipeline.deterministic import release_report as release


@pytest.fixture
def native_trial(tmp_path):
    run = tmp_path / "run"
    (run / "proposals").mkdir(parents=True)
    alerts = [{"event_id": "a", "quarter": 1}, {"event_id": "b", "quarter": 2}]
    (run / "proposals/123.json").write_text(
        json.dumps({"source_id": "123", "alerts": alerts})
    )
    signals = [
        {
            "signal_id": label,
            "event_ids": [event],
            "triage": {"status": "ACTIVE", "history": []},
        }
        for label, event in (("catalogue-alias", "a"), ("unmatched-alias", "b"))
    ]
    source = {
        "signals": signals,
        "alerts": alerts,
        "events": [
            {
                "event_id": item["event_id"],
                "original_alert": {"kind": "PERIODIC_WINDOW"},
            }
            for item in alerts
        ],
        "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "disposition": {
            "status": "BOI",
            "boi_signal_count": 1,
            "signals": [
                {"signal_id": "catalogue-alias", "disposition": "FALSE_ALARM"},
                {"signal_id": "unmatched-alias", "disposition": "BOI"},
            ],
        },
    }
    (run / "vetting/sources").mkdir(parents=True)
    (run / "vetting/sources/123.json").write_text(json.dumps(source))
    control = {
        "run": run,
        "sources": {"123": source},
        "physical": {},
        "prepared": {
            "123": {"parent_tpf": "45", "raw_fits_sha256": "a" * 64, "inputs": {}}
        },
    }
    graph = {
        ("native_eb", "123"): {
            "origin_parent": "45",
            "raw_fits_sha256": "a" * 64,
            "resolved_parents": "",
            "component_id": "component",
            "historically_exposed_component": "True",
            "contaminant_source_id": "",
            "contaminant_raw_fits_sha256": "",
        }
    }
    record = {
        "source_id": "123",
        "signals": [
            release.catalogue.trial_signal(source, signal) for signal in signals
        ],
    }
    return record, control, graph


def test_native_eb_without_a_truth_match_retains_every_automatic_alias(native_trial):
    record, control, graph = native_trial
    group, case = release.assemble_case(
        "catalogue_eclipsing_binary", record, control, graph
    )
    vetting = importlib.import_module("vetting")
    assert group == "cases"
    assert case["truth_cohort"] == "native_eb"
    assert "truth_events" not in case
    assert case["alert_ledger"]["declared_signal_ids"] == [
        "catalogue-alias",
        "unmatched-alias",
    ]
    assert vetting.replay_case_ledger(case)["complete"] is True
    assert vetting.replay_case_false_boi(case) is True


def test_assembly_rejects_a_lost_unmatched_alias(native_trial):
    record, control, graph = native_trial
    record["signals"].pop()
    with pytest.raises(ValueError, match="alias ledger is incomplete"):
        release.assemble_case("catalogue_eclipsing_binary", record, control, graph)


def test_changed_alert_timing_is_an_execution_failure(native_trial):
    record, control, graph = native_trial
    control["sources"]["123"]["alerts"][0]["quarter"] = 3
    _, case = release.assemble_case(
        "catalogue_eclipsing_binary", record, control, graph
    )
    vetting = importlib.import_module("vetting")
    assert vetting.replay_case_ledger(case)["complete"] is False
    assert vetting.replay_case_false_boi(case) is True


@pytest.mark.parametrize("complete,exposed", [(False, "False"), (True, "True")])
def test_final_input_failures_stop_before_reading_outcomes(
    tmp_path, monkeypatch, complete, exposed
):
    ledger = tmp_path / "trials.csv"
    row = {
        "cohort": "native_eb",
        "source_id": "123",
        "role": "evaluation",
        "historically_exposed_component": exposed,
    }
    with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (tmp_path / "audit.json").write_text(
        json.dumps(
            {
                "trials": 1,
                "trial_ledger_sha256": release.catalogue.digest(ledger),
                "joint_independence_complete": complete,
            }
        )
    )

    def read_outcomes(*args):
        pytest.fail("Final outcomes were read before input authorization")

    monkeypatch.setattr(release, "read_control_run", read_outcomes)
    with pytest.raises(ValueError, match="independence audit|historically opened"):
        release.build_report([{"report": {}}], tmp_path, "final_replay")


@pytest.mark.parametrize("category", ["eclipsing_binary", "instrumental_artifact"])
def test_dropped_synthetic_reports_cannot_enter_launch_scope(category):
    report = {
        "outcome_contract": release.OUTCOME_CONTRACT,
        "status": "COMPLETE_FALSE_CLASS_CONTROL_REPORT",
        "trials": [{"category": category, "source_id": "123"}],
    }
    with pytest.raises(ValueError, match="Dropped synthetic class"):
        release.report_records(report)
    assert category not in release.COHORTS


def test_final_selection_preserves_deferred_exclusions_without_requiring_their_pixels(
    tmp_path,
):
    rows = [
        {
            "cohort": cohort,
            "source_id": str(index),
            "role": "evaluation",
            "historically_exposed_component": exposed,
        }
        for index, (cohort, exposed) in enumerate(
            (
                ("native_eb", "False"),
                ("ordinary", "True"),
                ("catalogue_false_positive", "True"),
                ("eclipsing_binary", "True"),
            )
        )
    ]
    ledger = tmp_path / "trials.csv"
    with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "audit.json").write_text(
        json.dumps(
            {
                "trials": len(rows),
                "trial_ledger_sha256": release.catalogue.digest(ledger),
                "joint_independence_complete": True,
            }
        )
    )
    _, selected, inputs = release.input_graph(tmp_path, "final_replay")
    assert set(selected) == {("native_eb", "0")}
    assert inputs[str(ledger)] == release.catalogue.digest(ledger)


def test_native_report_uses_the_parent_trial_ledger():
    trial = {"category": "catalogue_eclipsing_binary", "source_id": "123"}
    report = {
        "outcome_contract": release.OUTCOME_CONTRACT,
        "status": "COMPLETE_CATALOGUE_CONTROL_REPORT",
        "truth_protocol": release.catalogue.NATIVE_EB_PROTOCOL,
        "trial_ledger": {"trials": [trial]},
    }
    assert release.report_records(report) == [("catalogue_eclipsing_binary", trial)]


def test_native_report_requires_the_frozen_native_protocol():
    report = {
        "outcome_contract": release.OUTCOME_CONTRACT,
        "status": "COMPLETE_CATALOGUE_CONTROL_REPORT",
        "trial_ledger": [],
    }
    with pytest.raises(ValueError, match="protocol"):
        release.report_records(report)


def test_null_assembly_counts_a_detector_alias_once_and_preserves_exclusions(monkeypatch):
    alerts = [
        {"event_id": name, "detector_signal_id": name, "quarter": 1,
         "start_bkjd": centre - 0.05, "end_bkjd": centre + 0.05}
        for name, centre in (("a", 5), ("b", 10))
    ]
    reference = {"origin": "REAL_SIGNAL", "reference_id": "real-b", "duration_hours": 2.4,
                 "events": [{"event_id": "real-b", "centre_bkjd": 10, "support": {"1": [20]}}]}
    monkeypatch.setattr(release, "exclusion_references", lambda *args: [reference])
    proposal = {
        "ungated_alerts": alerts, "alerts": alerts[:1],
        "lane_decisions": [{"proposal_id": alert["event_id"],
                            "decision": "PASS" if alert["event_id"] == "a" else "EXCLUDE",
                            "null_exclusion": release.null_exclusion([alert], [reference])}
                           for alert in alerts],
    }
    path = [{"stage": name, "decision": "PASS"} for name in ("shape", "orbital", "instrument")]
    source = {
        "alerts": alerts[:1], "events": [{"event_id": "unit", "original_alert": alerts[0]}],
        "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "signals": [{"signal_id": name, "event_ids": ["unit"], "triage": {"history": path}}
                    for name in ("fit-a", "fit-a-alternative")],
    }
    variant = release.null_variant_record(proposal, source, {}, "123", "inverted")
    measured = release.null_population([{
        "usable_quarters": 2, "variants": [variant], "scientific_abstention": True,
    }])
    assert measured["trials"] == measured["outcomes"] == 1
    assert measured["excluded_proposals"] == 1
    assert variant["declared_proposal_ids"] == ["a", "b"]
    assert len(variant["proposals"][0]["gate_paths"]) == 2
    proposal["lane_decisions"].pop()
    with pytest.raises(ValueError, match="lost, reordered or duplicated"):
        release.null_variant_record(proposal, source, {}, "123", "inverted")


def test_single_quarter_uses_nested_truth_matched_source_attribution():
    case = {
        "truth_events": [{"event_id": "truth"}],
        "signals": [
            {"proposal_event_match": {"status": status}, "downstream": {
                "localization": {"status": "MEASURED", "source_attribution": {
                    "status": attribution,
                }},
            }}
            for status, attribution in (
                ("MATCHED", "ROBUST_OFFSET_CONFLICT"),
                ("UNMATCHED", "ROBUST_OFFSET_CONFLICT"),
            )
        ],
    }
    assessed = release.single_quarter_assessment(case)
    assert assessed["applicable"] is True
    assert assessed["status"] == "ROBUST_OFFSET_CONFLICT"
    assert assessed["localization_statuses"] == ["ROBUST_OFFSET_CONFLICT"]
    case["signals"][0]["downstream"]["localization"]["source_attribution"]["status"] = "TARGET_COMPATIBLE"
    assert release.single_quarter_assessment(case)["status"] == "NO_ROBUST_OFFSET_CONFLICT"


@pytest.mark.parametrize("outcome", ["complete", "lost_alert", "instrument_dismissal"])
def test_development_cli_assembles_physical_and_both_null_ledgers(
    native_trial, tmp_path, monkeypatch, outcome
):
    lost_alert = outcome == "lost_alert"
    record, control, graph = native_trial
    control.update(
        path=tmp_path / "physical-report.json", report={"inputs": {}},
        records=[("catalogue_eclipsing_binary", record)],
        reference={}, policy={}, execution={}, preparation={}, manifest={}, binary={},
    )
    control["path"].write_text("{}")
    (control["run"] / "vetting.json").write_text("{}")
    graph[("null", "456")] = {
        "origin_parent": "78", "raw_fits_sha256": "b" * 64,
        "component_id": "null-parent", "historically_exposed_component": "False",
    }
    monkeypatch.setattr(release, "input_graph", lambda *args: ({}, graph, {}))
    monkeypatch.setattr(release, "read_control_run", lambda *args: control)
    monkeypatch.setattr(release, "exclusion_references", lambda *args: [])
    policy = {"implementation": {}, "evaluation_binding": {
        "development_replay": {"controls_sha256": "c" * 64},
        "run_spec_sha256": "d" * 64, "detector_sha256": "e" * 64,
        "calibration_sha256": "f" * 64,
    }}
    monkeypatch.setattr(release, "frozen_execution", lambda *args: (policy, {}))
    nulls = {}
    folders = []
    for variant in ("inverted", "quarter_scrambled"):
        folder = tmp_path / variant
        (folder / "proposals").mkdir(parents=True)
        (folder / "vetting/sources").mkdir(parents=True)
        alert = {"event_id": "event", "detector_signal_id": "proposal", "quarter": 1,
                 "start_bkjd": 4.9, "end_bkjd": 5.1}
        proposal = {"alerts": [alert], "ungated_alerts": [alert], "lane_decisions": [
            {"proposal_id": "proposal", "decision": "PASS",
             "null_exclusion": {"status": "ELIGIBLE", "matches": []}},
        ]}
        source = {"alerts": [] if lost_alert else [alert],
                  "processing_status": "EVIDENCE_STAGES_COMPLETE",
                  "events": [{"event_id": "event", "original_alert": alert}],
                  "signals": [{"signal_id": "fit", "event_ids": ["event"],
                               "triage": {"history": [{"stage": "shape", "decision": "DISMISS"}]}}],
                  "disposition": {"signals": [{"disposition": "UNRESOLVED"}]}}
        if outcome == "instrument_dismissal":
            source["signals"][0]["triage"]["history"] = [
                {"stage": stage, "decision": "DISMISS" if stage == "instrument" else "PASS"}
                for stage in ("shape", "orbital", "instrument")
            ]
        (folder / "proposals/456.json").write_text(json.dumps(proposal))
        (folder / "vetting/sources/456.json").write_text(json.dumps(source))
        specification = {"null_exclusions": {}, "sources": {"456": {
            "metadata": {"origin_tpf_kic": "78", "raw_fits_sha256": "b" * 64},
            "null_quarters": 3, "sha256": "1" * 64,
        }}}
        contract = {"id": release.NULL_CONTRACT, "variant": variant, "seed": 7,
                    "base_manifest": {"path": "original.json", "sha256": "2" * 64}}
        nulls[folder] = (dict(control, run=folder), {}, specification, ["456"], contract, {"456": source})
        folders.append(folder)
    monkeypatch.setattr(release, "verified_null_run", lambda folder, role: nulls[folder])
    monkeypatch.setattr(release, "dump_path", lambda path: path)
    reports = tmp_path / "entries.json"
    reports.write_text(json.dumps([{"report": {
        "path": str(control["path"]), "sha256": release.catalogue.digest(control["path"]),
    }}]))
    output = tmp_path / "assembled.json"
    monkeypatch.setattr(sys, "argv", [
        "release_report", "--reports", str(reports), "--input-graph", str(tmp_path),
        "--role", "development_replay", "--null-runs", *map(str, folders), "--out", str(output),
    ])
    if outcome == "instrument_dismissal":
        with pytest.raises(ValueError, match="instrument.*decision is invalid"):
            release.main()
        assert not output.exists()
        return
    if lost_alert:
        with pytest.raises(SystemExit) as failure:
            release.main()
        assert failure.value.code == 1
    else:
        release.main()
    report = json.loads(output.read_text())
    assert report["status"] == ("REPLAY_FAILED" if lost_alert else "REPLAY_PASSED")
    assert report["population"]["declared_cases"] == 2
    assert len(report["cases"]) == len(report["null_cases"]) == 1
    assert release.null_population(report["null_cases"])["trials"] == 2
    assert release.null_population(report["null_cases"])["outcomes"] == 0
