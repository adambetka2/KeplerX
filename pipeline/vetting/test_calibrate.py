"""Calibration search, sealed-weighted objective, false-BOI caps and policy vectors."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import calibrate

from pipeline.deterministic.catalogue_control_report import exact_upper_bound

HERE = Path(__file__).parent


def development_start():
    policy = json.loads((HERE / "disposition-policy.json").read_text())
    config = json.loads((HERE / "config.json").read_text())
    base = {**policy, "pipeline_contract": {"vetting": {"triage": config["triage"]}}}
    start = {"triage.shape_transit_bic_margin": config["triage"]["shape_transit_bic_margin"]}
    for name, rule in config["triage"]["assessment_rules"].items():
        for field, value in rule.items():
            if policy["assessment_rules"][name].get(field, 0) is None:
                start[f"policy.assessment_rules.{name}.{field}"] = value
    return base, start


def test_search_moves_only_on_strict_improvement_and_prefers_fewer_candidates():
    evaluated = []

    def evaluate(vectors):
        evaluated.extend(vectors)
        return [
            {
                "score": [
                    v["triage.b"] == 20,
                    0,
                    0.0,
                    min(v["policy.a"], 2),
                    -v["policy.a"],
                ],
                "false_boi": {},
            }
            for v in vectors
        ]

    grid = {"policy.a": [1, 2, 3], "triage.b": [10, 20]}
    vector, best, record = calibrate.search(
        evaluate, grid, {"policy.a": 1, "triage.b": 10}
    )
    assert vector == {"policy.a": 2, "triage.b": 20}
    assert best["score"] == [True, 0, 0.0, 2, -2]
    assert record["sweeps"] == 2
    assert record["vectors_evaluated"] == len(evaluated) == len(record["trace"]) == 6
    assert [(m["parameter"], m["to"]) for m in record["moves"]] == [
        ("policy.a", 2),
        ("triage.b", 20),
    ]


def test_objective_weights_opened_recovery_to_sealed_strata_under_exact_caps():
    cases = [
        {"truth_cohort": "periodic_planet", "source_id": sid}
        for sid in ("p1", "p2", "p3", "p4")
    ]
    cases += [{"truth_cohort": "native_eb", "source_id": f"e{i}"} for i in range(3)]
    cases += [{"truth_cohort": "ordinary", "source_id": "o1"}]
    hits = [True, False, True, True, True, False, False, True]
    outcomes = [(hit, int(hit), False) for hit in hits]
    spec = {
        "objective": [{"name": "pixel", "opened_cohort": "periodic_planet"}],
        "false_boi_caps": {"native_eb": 0.02},
        "diagnostic_cohorts": ["ordinary"],
        "confidence": 0.95,
    }
    strata = {
        "pixel": (
            {"p1": "A", "p2": "A", "p3": "B", "p4": "B"},
            Counter({"A": 30, "B": 10}),
        )
    }
    summary = calibrate.summarize(cases, outcomes, spec, strata)
    bound = exact_upper_bound(1, 3, 0.95)
    assert summary["expected_sealed_recoveries"] == 25
    assert summary["objective"]["pixel"]["opened_recovered"] == 3
    assert summary["false_boi"]["native_eb"]["upper_bound"] == bound
    assert summary["diagnostic"] == {"ordinary": {"trials": 1, "candidate_cases": 1}}
    assert summary["score"] == [False, 0, 0.02 - bound, 25, -5]


def test_policy_vector_sets_declared_paths_and_one_rule_set():
    base, start = development_start()
    policy, triage = calibrate.candidate(
        base, start, {"triage.shape_transit_bic_margin": 5}
    )
    assert triage["shape_transit_bic_margin"] == 5
    assert policy["status"] == "FROZEN_EVALUATION" and policy["boi_catalogue_enabled"]
    assert triage["assessment_rules"] == policy["assessment_rules"]
    assert policy["pipeline_contract"]["vetting"]["triage"] == triage
    assert {rule["status"] for rule in policy["assessment_rules"].values()} == {
        "RELEASED"
    }
    assert "boi_bic_margin" not in base
    with pytest.raises(ValueError, match="Unknown calibration parameter"):
        calibrate.candidate(base, start, {"policy.boi_margin": 20})
    del start["policy.assessment_rules.orbital.minimum_secondary_sigma"]
    with pytest.raises(ValueError, match="undeclared"):
        calibrate.candidate(base, start, {})


@pytest.mark.parametrize("count", [99, 101])
def test_false_boi_summary_rejects_missing_or_extra_outcomes(count):
    cases = [{"truth_cohort": "native_eb"} for _ in range(100)]
    spec = {
        "objective": [],
        "false_boi_caps": {"native_eb": 0.05},
        "diagnostic_cohorts": [],
        "confidence": 0.95,
    }
    with pytest.raises(ValueError, match="outcomes do not reconcile"):
        calibrate.summarize(cases, [(False, 0, False)] * count, spec, {})
    complete = calibrate.summarize(cases, [(False, 0, False)] * 100, spec, {})
    assert complete["false_boi"]["native_eb"]["trials"] == 100
    assert complete["feasible"]


def test_shard_outcomes_cover_every_record_once_for_every_vector():
    cases = [
        {"source_evidence": {"path": "first"}},
        {"source_evidence": None},
        {"source_evidence": {"path": "second"}},
    ]
    first, second = (True, 1, False), (False, 0, True)
    parts = [[[(2, second)], [(0, second)]], [[(0, first)], [(2, first)]]]
    assert calibrate.reconcile_outcomes(cases, 2, parts) == [
        [first, (False, 0, False), second],
        [second, (False, 0, False), first],
    ]
    for bad in (
        parts[:1],
        [*parts, parts[0]],
        [[[(0, first)], [(0, first)]], [[(1, second)], [(2, second)]]],
        [[[(0, first)], [(0, first)]], [[(-1, second)], [(2, second)]]],
        [[[(0, first)], [(0, first)]], [[(2, second)]]],
    ):
        with pytest.raises(ValueError, match="Calibration|calibration"):
            calibrate.reconcile_outcomes(cases, 2, bad)


@pytest.mark.parametrize("count", [0, 2])
def test_search_rejects_missing_or_extra_policy_evaluations(count):
    def evaluate(vectors):
        return [{"score": [True], "false_boi": {}}] * count

    with pytest.raises(ValueError, match="zip"):
        calibrate.search(evaluate, {"policy.a": [1]}, {"policy.a": 1})


def test_spec_must_classify_every_cohort_and_start_on_the_grid():
    cases = [
        {"truth_cohort": "periodic_planet", "truth_class": "planet"},
        {"truth_cohort": "native_eb", "truth_class": "eclipsing_binary"},
    ]
    spec = {
        "objective": [{"opened_cohort": "periodic_planet"}],
        "false_boi_caps": {"native_eb": 0.02},
        "diagnostic_cohorts": [],
        "confidence": 0.95,
        "grid": {"triage.shape_transit_bic_margin": [5, 10]},
        "start": {"triage.shape_transit_bic_margin": 10},
    }
    calibrate.check_spec(spec, cases)
    for change in (
        {"false_boi_caps": {}},
        {"false_boi_caps": {"native_eb": 0.02, "null": 0.005}},
        {"objective": [{"opened_cohort": "native_eb"}], "false_boi_caps": {}},
    ):
        with pytest.raises(ValueError, match="exactly one"):
            calibrate.check_spec({**spec, **change}, cases)
    for grid in (
        {"triage.shape_transit_bic_margin": [5, 20]},
        {"triage.assessment_rules.orbital.minimum_secondary_sigma": [7]},
    ):
        with pytest.raises(ValueError, match="grid parameter"):
            calibrate.check_spec({**spec, "grid": grid}, cases)


def test_planet_recovery_needs_a_matched_candidate_in_a_complete_disposition():
    def row(count):
        return {
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "signals": [{"signal_id": "s1"}, {"signal_id": "s2"}],
            "disposition": {
                "status": "BOI",
                "boi_signal_count": count,
                "signals": [
                    {"signal_id": "s1", "disposition": "BOI"},
                    {"signal_id": "s2", "disposition": "UNRESOLVED"},
                ],
            },
        }

    def planet(matched):
        return {
            "truth_class": "planet",
            "signals": [
                {"signal_id": sid, "fitted_event_match": {"status": status}}
                for sid, status in zip(("s1", "s2"), matched)
            ],
        }

    assert calibrate.outcome(planet(["MATCHED", "UNAVAILABLE"]), row(1)) == (
        True,
        1,
        False,
    )
    assert not calibrate.outcome(planet(["UNAVAILABLE", "MATCHED"]), row(1))[0]
    assert calibrate.outcome(planet(["MATCHED", "MATCHED"]), row(None)) == (
        False,
        1,
        True,
    )
    assert calibrate.outcome({"truth_class": "eclipsing_binary"}, row(1))[0]


def test_null_cap_counts_vetted_proposals_and_reports_sources_separately():
    spec = {
        "objective": [], "false_boi_caps": {"null": 0.005},
        "diagnostic_cohorts": [], "confidence": 0.95,
    }
    cases = [{"truth_cohort": "null"} for _ in range(2)]
    rejected = {"vetted": True, "false_candidate": False, "excluded": False}
    excluded = {"vetted": False, "false_candidate": False, "excluded": True}
    outcomes = [
        calibrate.null_outcome([rejected] * 400 + [excluded] * 3),
        calibrate.null_outcome([rejected] * 200),
    ]
    summary = calibrate.summarize(cases, outcomes, spec, {})
    assert summary["feasible"]
    assert summary["false_boi"]["null"]["trials"] == 600
    assert summary["false_boi"]["null"]["upper_bound"] == exact_upper_bound(0, 600, 0.95)
    assert summary["null_sources"] == {
        "trials": 2, "vetted_sources": 2,
        "false_candidate_sources": 0, "excluded_proposals": 3,
    }
    surviving = {"vetted": True, "false_candidate": True, "excluded": False}
    outcomes[0] = calibrate.null_outcome([surviving] * 2 + [rejected] * 398 + [excluded] * 3)
    summary = calibrate.summarize(cases, outcomes, spec, {})
    assert summary["false_boi"]["null"]["false_boi"] == 2
    assert summary["null_sources"]["false_candidate_sources"] == 1
    assert summary["boi_signals"] == 2
    assert summary["incomplete_cases"] == 0


@pytest.mark.parametrize("endpoint", [
    {"vetted": True, "false_candidate": False},
    {"vetted": False, "false_candidate": False, "excluded": False},
    {"vetted": False, "false_candidate": True, "excluded": True},
    {"vetted": 1, "false_candidate": False, "excluded": False},
])
def test_null_endpoint_counts_reject_missing_or_inconsistent_execution(endpoint):
    with pytest.raises(ValueError, match="Null proposal endpoint"):
        calibrate.null_outcome([endpoint])


def test_null_sources_cannot_fall_back_to_final_disposition_counting():
    with pytest.raises(ValueError, match="replayed proposal flux endpoints"):
        calibrate.outcome({"truth_cohort": "null"}, {})
    for invalid in (True, calibrate.NullCounts(-1, 0, 0), calibrate.NullCounts(1, 2, 0)):
        with pytest.raises(ValueError, match="vetted-proposal counts"):
            calibrate.cohort_counts({"truth_cohort": "null"}, invalid)
    cases = [{"truth_cohort": "null", "source_evidence": None}]
    with pytest.raises(ValueError, match="omit a recorded case"):
        calibrate.reconcile_outcomes(cases, 1, [])
    empty = calibrate.null_outcome([])
    assert calibrate.reconcile_outcomes(cases, 1, [[[(0, empty)]]]) == [[empty]]


def test_null_source_without_flux_records_still_enters_execution_shards(tmp_path):
    proposal = tmp_path / "run/proposals/123.json"
    proposal.parent.mkdir(parents=True)
    proposal.write_text("{}")
    variant = {"source_evidence": None, "proposal_evidence": calibrate.reference(proposal)}
    case = {"truth_cohort": "null", "source_evidence": None, "variants": [variant]}
    assert calibrate.run_directory(variant) == str(tmp_path / "run")
    assert calibrate.shards([case], 2) == [[(0, case, None)]]


def test_null_cohort_needs_no_invented_planet_truth_class():
    spec = {"objective": [], "false_boi_caps": {"null": 0.005}, "diagnostic_cohorts": [],
            "confidence": 0.95, "grid": {}, "start": {}}
    calibrate.check_spec(spec, [{"truth_cohort": "null"}])


def test_development_runs_must_match_the_frozen_implementation(tmp_path):
    config = {
        **json.loads((HERE / "config.json").read_text()),
        "discovery_contract": {"origin": "test"},
        "triage_execution": "RECORD_EVERY_GATE",
    }
    base = {
        "implementation": calibrate.implementation_hashes(),
        "pipeline_contract": calibrate.pipeline_contract(config),
    }

    def run(name, detector, changes=()):
        folder = tmp_path / name
        (folder / "vetting/sources").mkdir(parents=True)
        (folder / "vetting.json").write_text(json.dumps({**config, **dict(changes)}))
        (folder / "detection.json").write_text(
            json.dumps({"binary": {"sha256": detector}})
        )
        manifest = {
            "code_sha256": base["implementation"],
            "config_sha256": calibrate.reference(folder / "vetting.json")["sha256"],
        }
        (folder / "vetting/manifest.json").write_text(json.dumps(manifest))
        path = str(folder / "vetting/sources/1.json")
        return {"source_evidence": {"path": path, "sha256": ""}}

    first, second = run("a", "d1"), run("b", "d1")
    configs, detector = calibrate.development_runs(
        [first, second, {"source_evidence": None}], base
    )
    assert set(configs) == {str(tmp_path / "a"), str(tmp_path / "b")}
    assert detector == "d1"
    with pytest.raises(ValueError, match="different detector"):
        calibrate.development_runs([first, run("c", "d2")], base)
    skip = run("d", "d1", {"triage_execution": "SKIP_AFTER_FIRST_FAILURE"})
    with pytest.raises(ValueError, match="record-every-gate"):
        calibrate.development_runs([skip], base)
    with pytest.raises(ValueError, match="implementation differs"):
        calibrate.development_runs([first], {**base, "implementation": {"a": "b"}})


def test_metadata_binding_preserves_grid_and_keeps_unexecuted_controls_pending(tmp_path):
    path = HERE.parent / "deterministic/cloud-control-metadata.json"
    metadata = json.loads(path.read_text())
    draft = json.loads(metadata["draft_specification"]["text"])
    result = calibrate.prepare_bindings(path, tmp_path / "binding.json")
    assert result["grid"] == draft["grid"] and result["start"] == draft["start"]
    assert result["confidence"] == 0.95
    assert result["false_boi_caps"] == {
        "null": 0.005, "native_eb": 0.05, "ordinary": 0.01,
        "background_contaminant": 0.03, "contamination_pair": 0.03,
        "guide_star_family": 0.03,
    }
    assert sum(item["sources"] for item in result["opened_cohorts"].values()) == 876
    assert {item["status"] for item in result["opened_cohorts"].values()} == {"INPUTS_NOT_PROVISIONED"}
    excluded = set().union(*(set(item["source_ids"]) for item in metadata["sealed_exclusions"].values()))
    parents = {row["parent_tpf"] for cohort in metadata["opened_cohorts"].values() for row in cohort["rows"]}
    for selection in result["prospective_selections"].values():
        assert selection["status"] == "NOT_FROZEN" and len(selection["selection"]) == 100
        chosen = selection["selection"]
        assert not {row["source_id"] for row in chosen} & excluded
        new_parents = {row["parent_tpf"] for row in chosen}
        assert len(new_parents) == 100 and not new_parents & parents
        parents.update(new_parents)
    for term in result["objective"]:
        opened = calibrate.read_strata(term["opened_truth"])
        sealed = calibrate.read_strata(term["sealed_truth"])
        assert not opened.keys() & sealed.keys()
    assert result["status"] == "BOUND_METADATA_INPUTS_PENDING"
    assert result["opened_truth_descriptor"] == calibrate.reference(
        HERE.parent / "deterministic/cloud-opened-source-metadata.json"
    )
    with pytest.raises(ValueError, match="lack verified opened records"):
        calibrate.check_spec(result, [])


@pytest.mark.parametrize("keys,value", [
    (("grid", "triage.shape_transit_bic_margin"), [10]),
    (("start", "triage.shape_transit_bic_margin"), 20),
    (("confidence",), 0.90),
    (("false_boi_caps", "native_eb"), 0.10),
    (("objective",), []),
    (("diagnostic_cohorts",), ["ordinary"]),
    (("original_draft_sha256",), "0" * 64),
    (("metadata", "sha256"), "0" * 64),
    (("opened_truth_descriptor", "sha256"), "0" * 64),
    (("objective", 0, "opened_truth", "sha256"), "0" * 64),
    (("status",), "BOUND_METADATA_INPUTS_PENDING"),
])
def test_calibration_refuses_specification_drift_before_loading_records(tmp_path, monkeypatch, keys, value):
    metadata = HERE.parent / "deterministic/cloud-control-metadata.json"
    path = tmp_path / "spec.json"
    spec = calibrate.prepare_bindings(metadata, path)
    spec["status"] = "BOUND_OPENED_RECORDS"
    target = spec
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(spec))
    report, base = tmp_path / "report.json", tmp_path / "policy.json"
    report.write_text("{}")
    base.write_text("{}")

    def forbidden(_):
        pytest.fail("Specification drift reached record loading")

    monkeypatch.setattr(calibrate, "development_cases", forbidden)
    with pytest.raises(ValueError, match="Calibration|Objective|Input unavailable or changed"):
        calibrate.calibrate(report, path, base, tmp_path / "result.json", 1)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("change", ["draft", "cohort", "strata", "parent"])
def test_binding_rejects_changed_grid_missing_cohort_and_unreconciled_strata(tmp_path, change):
    metadata = json.loads((HERE.parent / "deterministic/cloud-control-metadata.json").read_text())
    if change == "draft":
        metadata["draft_specification"]["text"] += " "
    elif change == "cohort":
        del metadata["opened_cohorts"]["native-eb"]
    elif change == "strata":
        metadata["strata"]["periodic_planet-development-strata.jsonl"]["rows"].pop()
    else:
        metadata["opened_cohorts"]["planets"]["rows"][0]["parent_tpf"] = metadata["opened_cohorts"]["ordinary"]["rows"][0]["parent_tpf"]
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="invalid|do not reconcile|identities repeat"):
        calibrate.prepare_bindings(path, tmp_path / "binding.json")


def null_adapter_fixture(tmp_path, monkeypatch):
    from pipeline.deterministic import release_report, signal_contracts

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) + "\n")
        return calibrate.reference(path)

    metadata = {"source_id": "123", "origin_tpf_kic": 41, "raw_fits_sha256": "f" * 64}
    base_manifest = write(tmp_path / "base.json", {
        "schema": 1, "measurement_scope": "UNIT_METADATA_ONLY", "role": "development",
        "sources": {"123": {"metadata": metadata}},
    })
    common = {"source_id": "123", "coverage_complete": True,
              "coverage_scope": signal_contracts.EXCLUSION_COVERAGE,
              "unknown_proposals": "RETAIN_IN_NULL_TRIAL_COUNT",
              "base_manifest_sha256": base_manifest["sha256"]}
    snapshots = {f"snapshot-{index}": {"reference": write(tmp_path / f"snapshot-{index}.json", [])}
                 for index in range(6)}
    snapshot_specification = write(tmp_path / "snapshots.json", {
        "files": {name: {"sha256": row["reference"]["sha256"]} for name, row in snapshots.items()},
    })
    real = write(tmp_path / "real.json", {**common, "references": []})
    known = {
        "origin": "CATALOGUE_EPHEMERIS", "reference_id": "known-c", "duration_hours": 2.4,
        "events": [{"event_id": "known-c", "centre_bkjd": 10, "support": {"1": {"psf_flat": 20}}}],
    }
    support = write(tmp_path / "support.json", {
        "source_id": "123", "status": "PREPARED_DISCOVERY_SUPPORT",
        "base_manifest_sha256": base_manifest["sha256"], "original_preparation": None,
        "prepared_fsc": snapshots["snapshot-0"]["reference"],
        "supported_references": {"known-c": {key: known[key] for key in ("duration_hours", "events")}},
    })
    known["verification"] = {
        "status": "FROZEN_CATALOGUE_EPHEMERIS", "catalogue": "snapshot-0", "row_index": 0,
        "snapshot": snapshots["snapshot-0"]["reference"], "prepared_support": support,
        "base_manifest_sha256": base_manifest["sha256"],
    }
    catalogue = write(tmp_path / "catalogue.json", {
        **common, "references": [known],
        "lookups": {name: {"snapshot": row["reference"], "status": "SCOPED_SNAPSHOT_LOOKUP",
                           "matched_rows": [0] if name == "snapshot-0" else []}
                    for name, row in snapshots.items()},
    })
    exclusions = write(tmp_path / "exclusions.json", {
        "contract": signal_contracts.NULL_CONTRACT, "exclusion_rule": signal_contracts.EXCLUSION_RULE,
        "coverage_scope": signal_contracts.EXCLUSION_COVERAGE, "unknown_proposals": "RETAIN_IN_NULL_TRIAL_COUNT",
        "role": "development", "base_manifest": base_manifest,
        "catalogue_snapshots": snapshots, "catalogue_specification": snapshot_specification,
        "original_preparation": None, "sources": {"123": {
            "coverage_complete": True, "real_signals": real, "catalogue_ephemerides": catalogue,
        }},
    })
    alerts = [{
        "event_id": name, "detector_signal_id": name, "quarter": 1,
        "start_bkjd": centre - 0.05, "end_bkjd": centre + 0.05,
    } for name, centre in (("a", 5), ("b", 20), ("c", 10))]
    config = {"triage_execution": "RECORD_EVERY_GATE", "triage": {
        "enabled": True, "shape_transit_bic_margin": 10, "shape_false_alarm_bic_margin": 10,
    }}
    monkeypatch.setattr(calibrate.vetting_triage, "shape_gate", lambda row, signal, policy:
                        calibrate.vetting_triage.shape_decision(signal["unit_score"], {}, policy))
    for name in ("orbital_gate", "instrument_gate", "localization_gate", "contamination_gate"):
        monkeypatch.setattr(calibrate.vetting_triage, name,
                            lambda *args: calibrate.vetting_triage.outcome("PASS", "UNIT", {}))
    monkeypatch.setattr(calibrate, "adjudicate", lambda row: {"status": "UNRESOLVED"})
    case = {
        "truth_cohort": "null", "source_id": "123", "parent_tpf": "41",
        "raw_fits_sha256": ["f" * 64], "usable_quarters": 4, "variants": [],
        "scientific_abstention": True, "source_evidence": None,
        "source_ids": ["123"], "component_ids": ["unit-41"], "input_sha256": "d" * 64,
    }
    for variant in ("inverted", "quarter_scrambled"):
        run = tmp_path / variant
        specification = {
            "schema": 1, "role": "development", "measurement_scope": "UNIT_RECORDS_ONLY",
            "null_construction": {
                "id": signal_contracts.NULL_CONTRACT, "variant": variant,
                "seed": 7, "normalized_relative": True, "base_manifest": base_manifest,
            },
            "construction_ledger": [{"source_id": "123", "quarters": 4, "status": "CONSTRUCTED"}],
            "null_exclusions": exclusions, "sources": {"123": {
                "sha256": "a" * 64, "null_quarters": 4,
                "metadata": metadata,
            }},
        }
        write(run / "preparation.json", {"input_manifest": write(run / "input.json", specification)})
        proposal = {"source_id": "123", "ungated_alerts": alerts, "alerts": alerts[:2],
                    "lane_decisions": [{
                        "proposal_id": alert["detector_signal_id"],
                        "decision": "EXCLUDE" if alert["event_id"] == "c" else "PASS",
                        "null_exclusion": signal_contracts.null_exclusion([alert], [known]),
                    } for alert in alerts]}
        row = {"source_id": "123", "processing_status": "EVIDENCE_STAGES_COMPLETE",
               "alerts": alerts[:2], "pixel_jobs": [], "events": [
            {"event_id": "unit-" + alert["event_id"], "original_alert": alert} for alert in alerts[:2]
        ], "signals": [{"signal_id": name, "event_ids": ["unit-" + original], "unit_score": score}
                       for name, original, score in (("fit-a", "a", 15), ("alias-a", "a", 15),
                                                     ("fit-b", "b", -15))]}
        calibrate.replay(row, config, None)
        record = release_report.null_variant_record(proposal, row, specification, "123", variant)
        record.update(input_sha256="a" * 64,
                      proposal_evidence=write(run / "proposals/123.json", proposal),
                      source_evidence=write(run / "vetting/sources/123.json", row))
        case["variants"].append(record)
        monkeypatch.setitem(calibrate.RUN_CONFIGS, calibrate.run_directory(record), config)
    return case, config


def test_real_null_adapter_replays_production_gates_and_counts_aliases_once(tmp_path, monkeypatch):
    case, config = null_adapter_fixture(tmp_path, monkeypatch)
    assert calibrate.null_identity(case)
    candidates = [(None, {**config["triage"], "shape_transit_bic_margin": margin})
                  for margin in (10, 20)]
    expected = [(calibrate.NullCounts(4, 2, 2), 2, False),
                (calibrate.NullCounts(4, 0, 2), 0, False)]
    assert calibrate.evaluate_null(case, candidates) == expected
    assert calibrate.evaluate_shard([(0, case, None)], candidates) == [
        [(0, value)] for value in expected
    ]
    case["variants"][0]["declared_proposal_ids"] = []
    with pytest.raises(ValueError, match="fixed null proposal population"):
        calibrate.evaluate_null(case, candidates)


@pytest.mark.parametrize("field,value", [
    ("parent_tpf", "42"), ("raw_fits_sha256", ["e" * 64]), ("usable_quarters", 5),
])
def test_null_adapter_rejects_changed_original_identity(tmp_path, monkeypatch, field, value):
    case, _ = null_adapter_fixture(tmp_path, monkeypatch)
    case[field] = value
    with pytest.raises(ValueError, match="original input identity changed"):
        list(calibrate.null_records(case))


def test_development_report_includes_only_completed_null_execution(tmp_path, monkeypatch):
    case, _ = null_adapter_fixture(tmp_path, monkeypatch)
    report = {
        "replay_role": "development_replay", "status": "OPENED_REPORT_ASSEMBLED",
        "outcome_contract": calibrate.OUTCOME_CONTRACT,
        "event_matching_contract": calibrate.EVENT_MATCHING_CONTRACT,
        "null_cases": [case],
        "inputs": {str(path): calibrate.reference(path)["sha256"]
                   for path in tmp_path.rglob("*.json")},
    }
    assert calibrate.development_cases(report) == [case]
    case["variants"][0]["status"] = "INCOMPLETE_NULL_FLUX_LEDGER"
    with pytest.raises(ValueError, match="complete null proposal execution"):
        calibrate.development_cases(report)


def null_run_policy(case):
    config = {
        **json.loads((HERE / "config.json").read_text()),
        "discovery_contract": {"origin": "unit"}, "triage_execution": "RECORD_EVERY_GATE",
    }
    base = {"implementation": calibrate.implementation_hashes(),
            "pipeline_contract": calibrate.pipeline_contract(config)}
    for variant in case["variants"]:
        run = Path(calibrate.run_directory(variant))
        (run / "vetting.json").write_text(json.dumps(config))
        (run / "detection.json").write_text(json.dumps({"binary": {"sha256": "b" * 64}}))
        (run / "vetting/manifest.json").write_text(json.dumps({
            "code_sha256": base["implementation"],
            "config_sha256": calibrate.reference(run / "vetting.json")["sha256"],
        }))
    return base


def mutate_null_recipe(case, field):
    run = Path(calibrate.run_directory(case["variants"][1]))
    preparation = run / "preparation.json"
    reference = json.loads(preparation.read_text())["input_manifest"]
    path = calibrate.resolve(reference["path"])
    specification = json.loads(path.read_text())
    if field == "seed":
        specification["null_construction"]["seed"] = 8
    else:
        target = specification["null_construction"] if field == "base_manifest" else specification
        original = calibrate.resolve(target[field]["path"])
        changed = original.with_name(original.stem + "-changed.json")
        changed.write_bytes(original.read_bytes())
        target[field] = calibrate.reference(changed)
    path.write_text(json.dumps(specification))
    preparation.write_text(json.dumps({"input_manifest": calibrate.reference(path)}))


@pytest.mark.parametrize("field", ["seed", "base_manifest", "null_exclusions"])
def test_null_variants_bind_the_common_construction_recipe(tmp_path, monkeypatch, field):
    case, _ = null_adapter_fixture(tmp_path, monkeypatch)
    base = null_run_policy(case)
    calibrate.development_runs([case], base)
    mutate_null_recipe(case, field)
    with pytest.raises(ValueError, match="base inputs, seed or exclusions"):
        calibrate.development_runs([case], base)


def test_post_assembly_preparation_recipe_drift_is_rejected(tmp_path, monkeypatch):
    case, _ = null_adapter_fixture(tmp_path, monkeypatch)
    base = null_run_policy(case)
    report = {
        "replay_role": "development_replay", "status": "OPENED_REPORT_ASSEMBLED",
        "outcome_contract": calibrate.OUTCOME_CONTRACT,
        "event_matching_contract": calibrate.EVENT_MATCHING_CONTRACT,
        "null_cases": [case], "inputs": {
            str(path): calibrate.reference(path)["sha256"] for path in tmp_path.rglob("*.json")
        },
    }
    assert calibrate.development_cases(report) == [case]
    calibrate.development_runs([case], base)
    mutate_null_recipe(case, "seed")
    with pytest.raises(ValueError, match="Calibration input changed"):
        calibrate.development_cases(report)
    report["inputs"] = {}
    with pytest.raises(ValueError, match="lacks input hash bindings"):
        calibrate.development_cases(report)


@pytest.mark.parametrize("change", ["missing_report", "empty_report", "missing_record", "changed_record"])
def test_absent_or_stale_final_record_inventory_never_reaches_search(tmp_path, monkeypatch, change):
    case, _ = null_adapter_fixture(tmp_path, monkeypatch)
    base = null_run_policy(case)
    report = {
        "replay_role": "development_replay", "status": "OPENED_REPORT_ASSEMBLED",
        "outcome_contract": calibrate.OUTCOME_CONTRACT,
        "event_matching_contract": calibrate.EVENT_MATCHING_CONTRACT,
        "null_cases": [case], "inputs": {
            str(path): calibrate.reference(path)["sha256"] for path in tmp_path.rglob("*.json")
        },
    }
    report_path, base_path = tmp_path / "report.json", tmp_path / "policy.json"
    if change != "missing_report":
        report_path.write_text(json.dumps({} if change == "empty_report" else report))
    base_path.write_text(json.dumps(base))
    if change in {"missing_record", "changed_record"}:
        record = calibrate.resolve(case["variants"][0]["source_evidence"]["path"])
        if change == "missing_record":
            record.unlink()
        else:
            record.write_text("{}\n")
    spec_path = tmp_path / "spec.json"
    metadata = HERE.parent / "deterministic/cloud-control-metadata.json"
    spec = calibrate.prepare_bindings(metadata, spec_path)
    spec["status"] = "BOUND_OPENED_RECORDS"
    spec_path.write_text(json.dumps(spec))

    def forbidden(*args):
        pytest.fail("Absent or stale final records reached calibration search")

    monkeypatch.setattr(calibrate, "optimize", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        calibrate.calibrate(report_path, spec_path, base_path, tmp_path / "result.json", 1)
    assert not (tmp_path / "result.json").exists()
