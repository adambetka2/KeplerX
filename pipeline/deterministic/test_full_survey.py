"""Checks for the unattended survey release boundary and final catalogue."""

import csv
import gzip
import hashlib
import json
import sqlite3
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.deterministic import full_survey


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_boi_catalogue_requires_an_explicit_policy(tmp_path):
    policy = tmp_path / "policy.json"
    for status, enabled in [("DEVELOPMENT", False), ("RELEASED", False), ("BOI_CATALOGUE", False)]:
        save(policy, {"status": status, "boi_catalogue_enabled": enabled})
        with pytest.raises(ValueError, match="not configured"):
            full_survey.released_policy(policy)
    save(policy, {"status": "BOI_CATALOGUE", "boi_catalogue_enabled": True})
    assert full_survey.released_policy(policy)["status"] == "BOI_CATALOGUE"


def test_release_preflight_uses_scientific_artifact_validator():
    from pipeline.deterministic import run

    with pytest.raises(ValueError, match="triage contract"):
        full_survey.verified_release_contract({}, run)


@pytest.mark.parametrize("change", [None, "path", "hash"])
def test_full_run_forwards_and_verifies_survey_policy(tmp_path, monkeypatch, change):
    from pipeline.deterministic import run

    source_ids = tmp_path / "source-ids.json"
    policy = tmp_path / "released-policy.json"
    save(source_ids, ["123"])
    save(policy, {"status": "RELEASED"})
    monkeypatch.setattr(full_survey, "dump_path", lambda path: Path(path))
    monkeypatch.setattr(full_survey, "ledger_stage", lambda *args: nullcontext())
    monkeypatch.setattr(full_survey, "verify_frozen_release", lambda *args: None)
    monkeypatch.setattr(full_survey, "verified_input_ledger", lambda *args: ({}, {}))
    monkeypatch.setattr(full_survey, "frozen_sources", lambda *args: [3])
    monkeypatch.setattr(full_survey, "bind_stage_inputs", lambda *args: None)

    def stage(command, *args):
        if "survey" not in command:
            raise RuntimeError("Reached vetting with the verified survey policy")
        assert command[command.index("--disposition-policy") + 1] == str(policy)
        return {
            "detector_contract": run.detector_contract(),
            "requested_sources": 1, "attempted_sources": 1,
            "selection": {"source_ids": {"sha256": full_survey.digest(source_ids)}, "shards": [3]},
            "implementation": run.implementation_hashes(),
            "disposition_policy": str(tmp_path / "other.json") if change == "path" else str(policy),
            "disposition_policy_sha256": "0" * 64 if change == "hash" else full_survey.digest(policy),
        }

    monkeypatch.setattr(full_survey, "run_stage", stage)
    args = SimpleNamespace(
        source_ids=source_ids, survey=tmp_path / "survey", vetting=tmp_path / "vetting",
        disposition_policy=policy, input_ledger=tmp_path / "ledger.json", expected_sources=1,
        survey_batch_size=1, threads=1, vetting_batch_size=1, workers=1,
        concurrent_vetting_batches=1,
    )
    error, message = (ValueError, "frozen source identities") if change else (
        RuntimeError, "Reached vetting with the verified survey policy"
    )
    with pytest.raises(error, match=message):
        full_survey.full_survey(args)


def test_full_run_rejects_policy_change_before_reusing_released_stages(tmp_path):
    path = tmp_path / "policy.json"
    save(path, {"status": "RELEASED", "boi_catalogue_enabled": True})
    checksum = full_survey.digest(path)
    save(path, {"status": "RELEASED", "boi_catalogue_enabled": True, "revision": 2})
    with pytest.raises(ValueError, match="policy changed during"):
        full_survey.verify_frozen_release(path, checksum, None)


def test_frozen_release_revalidates_scientific_contract(tmp_path):
    from pipeline.deterministic import run

    path = tmp_path / "policy.json"
    save(
        path,
        {
            "status": "BOI_CATALOGUE",
            "boi_catalogue_enabled": True,
                    "assessment_rules": {"orbital": {"status": "BOI_CATALOGUE", "threshold": 7}},
        },
    )
    with pytest.raises(ValueError, match="triage contract"):
        full_survey.verify_frozen_release(path, full_survey.digest(path), run)


def test_frozen_sources_reconcile_with_read_only_index(tmp_path, monkeypatch):
    monkeypatch.setattr(full_survey, "ROOT", tmp_path)
    index = tmp_path / "discSSD/data/index.sqlite"
    index.parent.mkdir(parents=True)
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE sources (gaia_source_id TEXT, shard INTEGER)")
        connection.executemany(
            "INSERT INTO sources VALUES (?, ?)", [("100", 1), ("200", 0)]
        )
    other = tmp_path / "discPCIe/data/index.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}")
    identities = tmp_path / "source-ids.json"
    save(identities, ["100", "200"])
    assert full_survey.frozen_sources(identities, 2) == [0, 1]

    save(identities, ["100", "100"])
    with pytest.raises(ValueError, match="repeated"):
        full_survey.frozen_sources(identities, 2)


def test_completed_stage_is_not_restarted(tmp_path, monkeypatch):
    run = tmp_path / "survey"
    report = run / "survey.json"
    save(report, {"status": "COMPLETE"})

    def unexpected(*args, **kwargs):
        raise AssertionError("A complete stage was restarted")

    monkeypatch.setattr(full_survey.subprocess, "run", unexpected)
    assert full_survey.run_stage(["survey"], report, run)["status"] == "COMPLETE"


def test_run_ledger_records_every_stage_start(tmp_path):
    ledger = tmp_path / "vetting.full-survey-ledger.json"
    report = tmp_path / "survey/survey.json"
    stage = partial(full_survey.ledger_stage, ledger, {"survey": tmp_path})
    progress = partial(full_survey.stage_progress, report, "attempted_sources")

    with pytest.raises(RuntimeError, match="expected"), stage("survey", progress):
        assert full_survey.read(ledger)["stages"][0]["status"] == "RUNNING"
        save(report, {"batches": [{}], "attempted_sources": 2})
        raise RuntimeError("expected")
    with stage("survey", progress):
        save(report, {"batches": [{}, {}, {}], "attempted_sources": 5})
    with stage("labels"):
        pass

    failed, resumed, labels = full_survey.read(ledger)["stages"]
    assert (failed["status"], failed["error"]) == ("FAILED", "RuntimeError: expected")
    assert (resumed["status"], labels["status"]) == ("COMPLETE", "COMPLETE")
    assert [
        (record[point]["batches"], record[point]["sources"])
        for record in (failed, resumed)
        for point in ("start", "stop")
    ] == [(0, 0), (1, 2), (1, 2), (3, 5)]
    for record in (failed, resumed, labels):
        assert record["started_utc"] <= record["stopped_utc"]
        assert record["wall_seconds"] >= 0
        assert record["cpu_seconds"] >= 0
        assert record["peak_rss_bytes"] >= 0
        assert record["stop"]["free_bytes"]["survey"] > 0
    assert set(labels["start"]) == {"free_bytes"}


def test_candidate_catalogue_reconciles_batch_and_evidence(tmp_path):
    batch = tmp_path / "batches/000001"
    evidence = batch / "vetting/sources/100.json.gz"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"evidence")
    identities = ["100"]
    save(tmp_path / "identities/000001.json", identities)
    candidate_path = batch / "vetting/bois.json"
    candidates = [{"source_id": "100", "signal_id": "s1", "evidence": str(evidence)}]
    save(candidate_path, candidates)
    archive = {
        "relinked_summaries": {
            "bois.json": {
                "after_sha256": full_survey.digest(candidate_path)
            }
        },
        "files": {
            "vetting/sources/100.json": {"archive_sha256": full_survey.digest(evidence)}
        },
    }
    save(batch / "compact.json", archive)
    complete = batch / "batch-complete.json"
    save(
        complete,
        {
            "status": "COMPLETE",
            "sources": 1,
            "boi_signals": 1,
            "source_ids_sha256": full_survey.json_digest(identities),
            "archive_sha256": full_survey.json_digest(archive),
        },
    )
    report = {"batches": [{"batch": 1, "sha256": full_survey.digest(complete)}]}
    assert [
        (row["signal_id"], row["label"])
        for row in full_survey.boi_proposals(tmp_path, report)
    ] == [("s1", "BOI")]

    save(candidate_path, [])
    with pytest.raises(ValueError, match="batch does not reconcile"):
        full_survey.boi_proposals(tmp_path, report)

    save(candidate_path, candidates)
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        full_survey.boi_proposals(tmp_path, report)


def test_full_survey_paths_accept_authorized_external_outputs(tmp_path, monkeypatch):
    from pipeline import runtime

    root = tmp_path / "repository"
    local = root / "dump"
    external = tmp_path / "volume/forgotten-stars-workspace/dump"
    original = tmp_path / "volume/data"
    local.mkdir(parents=True)
    external.mkdir(parents=True)
    original.mkdir()
    (local / "survey").symlink_to(external / "survey", target_is_directory=True)
    (local / "original-data").symlink_to(original, target_is_directory=True)
    monkeypatch.setattr(full_survey, "ROOT", root)
    monkeypatch.setattr(runtime, "DUMP", local)
    monkeypatch.setattr(runtime, "EXTERNAL_DUMPS", (external,))
    monkeypatch.setattr(Path, "is_mount", lambda path: path == external.parents[1])

    assert full_survey.dump_path("dump/local") == local / "local"
    assert full_survey.dump_path(external / "survey") == external / "survey"
    assert full_survey.dump_path("dump/survey") == external / "survey"
    for path in (original, "dump/original-data"):
        with pytest.raises(ValueError, match="Pipeline outputs must stay"):
            full_survey.dump_path(path)


def disposition_fixture(tmp_path):
    survey = tmp_path / "survey"
    out = tmp_path / "vetting"
    batch = out / "batches/000001"
    rows = {}
    for source_id, labels in (
        ("100", ["BOI", "UNRESOLVED", "FALSE_ALARM"]),
        ("200", ["FALSE_POSITIVE"]),
    ):
        signals = [
            {
                "signal_id": f"s{index}",
                "period_days": None,
                "events": [{"unit_id": "e1"}],
            }
            for index in range(len(labels))
        ]
        rows[source_id] = {
            "source_id": source_id,
            "signals": signals,
            "disposition_policy": {"status": "BOI_CATALOGUE", "boi_catalogue_enabled": True},
            "disposition": {
                "status": "BOI"
                if source_id == "100"
                else "FALSE_POSITIVE",
                "policy_id": "test-policy",
                "attributed_signal_reviews": [],
                "signals": [
                    {
                        "signal_id": f"s{index}",
                        "disposition": label,
                        "reason_codes": [label + "_EVIDENCE"],
                        "signal_shape": {"status": "MEASURED"},
                        "evidence_assessment": {
                            "status": "UNASSESSED"
                            if label == "UNRESOLVED"
                            else "ASSESSED"
                        },
                        "reproduced_mechanisms": [
                            {"status": "REPRODUCED", "evidence": "measured eclipse"}
                        ]
                        if label == "FALSE_POSITIVE"
                        else [],
                        "risk_features": {"unknown_period": True},
                    }
                    for index, label in enumerate(labels)
                ],
            },
        }
    archive = {"files": {}}
    for source_id, row in rows.items():
        name = f"vetting/sources/{source_id}.json"
        path = batch / (name + ".gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(row).encode()
        path.write_bytes(gzip.compress(payload, mtime=0))
        archive["files"][name] = {
            "archive_sha256": full_survey.digest(path),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
        }
    ids = list(rows)
    save(out / "identities/000001.json", ids)
    save(batch / "compact.json", archive)
    save(
        batch / "batch-complete.json",
        {
            "status": "COMPLETE",
            "sources": 2,
            "source_ids_sha256": full_survey.json_digest(ids),
            "archive_sha256": full_survey.json_digest(archive),
        },
    )
    survey.mkdir()
    (survey / "cohort.csv").write_text("source_id\n100\n200\n300\n")
    (survey / "ineligible.csv").write_text(
        "source_id,reason\n400,INSUFFICIENT_CADENCES\n"
    )
    return (
        survey,
        out,
        {
            "accepted_sources": 3,
            "attempted_sources": 4,
            "ineligible_sources": 1,
            "artifacts": {
                name: full_survey.digest(survey / name)
                for name in ("cohort.csv", "ineligible.csv")
            },
        },
        {
            "completed_sources": 2,
            "batches": [
                {
                    "batch": 1,
                    "sha256": full_survey.digest(batch / "batch-complete.json"),
                }
            ],
        },
        [{"source_id": "100", "signal_id": "s0"}],
    )


def test_complete_disposition_export_preserves_all_aliases_and_source_stages(tmp_path):
    args = disposition_fixture(tmp_path)
    result = full_survey.export_dispositions(*args)
    assert result["sources"] == result["signals"] == 4
    assert result["source_stages"] == {
        "VETTED": 2,
        "SCANNED_NO_FOLLOWUP": 1,
        "INELIGIBLE": 1,
    }
    assert result["signal_dispositions"] == {
        "BOI": 1,
        "UNRESOLVED": 1,
        "FALSE_ALARM": 1,
        "FALSE_POSITIVE": 1,
    }
    with (args[1] / "source-dispositions.csv").open() as stream:
        sources = {r["source_id"]: r for r in csv.DictReader(stream)}
    assert sources["100"]["disposition"] == "BOI"
    assert sources["300"]["disposition"] == sources["400"]["disposition"] == ""
    with (args[1] / "signal-dispositions.csv").open() as stream:
        signals = list(csv.DictReader(stream))
    assert len(signals) == 4
    assert all(r["period_days"] == "" and r["event_units"] == "1" for r in signals)
    assert json.loads(signals[1]["evidence_assessment"])["status"] == "UNASSESSED"
    assert full_survey.export_dispositions(*args) == result


@pytest.mark.parametrize(
    "change, message",
    [
        ("tamper", "evidence changed"),
        ("wrong_original_hash", "Original disposition evidence changed"),
        ("manual_review", "Automatic disposition identities"),
        ("missing_alias", "Automatic disposition identities"),
        ("wrong_source_status", "Source/signal dispositions"),
    ],
)
def test_disposition_export_rejects_changed_or_nonautomatic_evidence(
    tmp_path, change, message
):
    args = disposition_fixture(tmp_path)
    batch = args[1] / "batches/000001"
    path = batch / "vetting/sources/100.json.gz"
    archive = full_survey.read(batch / "compact.json")
    row = json.loads(gzip.decompress(path.read_bytes()))
    if change == "tamper":
        path.write_bytes(b"changed")
    elif change == "wrong_original_hash":
        archive["files"]["vetting/sources/100.json"]["source_sha256"] = "0" * 64
    else:
        if change == "manual_review":
            row["disposition"]["signals"][0]["reviews"] = [
                {"disposition": "BOI"}
            ]
        elif change == "missing_alias":
            row["disposition"]["signals"].pop()
        else:
            row["disposition"]["status"] = "FALSE_ALARM"
        payload = json.dumps(row).encode()
        path.write_bytes(gzip.compress(payload, mtime=0))
        archive["files"]["vetting/sources/100.json"] = {
            "archive_sha256": full_survey.digest(path),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
        }
    save(batch / "compact.json", archive)
    complete = full_survey.read(batch / "batch-complete.json")
    complete["archive_sha256"] = full_survey.json_digest(archive)
    save(batch / "batch-complete.json", complete)
    args[3]["batches"][0]["sha256"] = full_survey.digest(batch / "batch-complete.json")
    with pytest.raises(ValueError, match=message):
        full_survey.export_dispositions(*args)


def test_disposition_export_rejects_missing_candidates_and_source_coverage(tmp_path):
    args = disposition_fixture(tmp_path)
    with pytest.raises(ValueError, match="Candidate/disposition mismatch"):
        full_survey.export_dispositions(*args[:-1], [])
    (args[0] / "ineligible.csv").write_text("source_id,reason\n")
    with pytest.raises(ValueError, match="source ledger changed"):
        full_survey.export_dispositions(*args)
    args[2]["artifacts"]["ineligible.csv"] = full_survey.digest(
        args[0] / "ineligible.csv"
    )
    with pytest.raises(ValueError, match="disposition counts"):
        full_survey.export_dispositions(*args)
    assert not (args[1] / "signal-dispositions.csv").exists()
