"""Run a frozen LC2 survey through BOI designation and complete disposition export."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from pipeline.runtime import PeakMemory, cpu_seconds, worker_count
from pipeline.runtime import dump_path as runtime_dump_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "pipeline/vetting/disposition-policy.json"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dump_path(path):
    return runtime_dump_path(ROOT / path)


def has_null(value):
    if isinstance(value, dict):
        return any(has_null(item) for item in value.values())
    if isinstance(value, list):
        return any(has_null(item) for item in value)
    return value is None


def released_policy(path):
    policy = read(path)
    if (
        policy.get("status") not in {"BOI_CATALOGUE", "RELEASED"}
        or policy.get("boi_catalogue_enabled") is not True
    ):
        raise ValueError("The BOI catalogue policy is not configured")
    return policy


def verified_release_contract(policy, pipeline):
    sys.path.insert(0, str(ROOT / "pipeline/vetting"))
    from pipeline.vetting import vetting

    config = read(ROOT / "pipeline/vetting/config.json")
    view = (
        policy.get("pipeline_contract", {})
        .get("discovery", {})
        .get("discovery_view", "psf_flat")
    )
    declared = policy.get("pipeline_contract", {}).get("discovery", {})
    config["discovery_contract"] = (
        pipeline.bound_detector_contract(declared)
        if declared
        else pipeline.detector_contract(view)
    )
    if declared.get("discovery_profile") == "rust-transit-only-v1":
        config["flatten_mask_views"] = ["psf_flat", "psf_nova"]
    vetting.validate_triage_contract(policy)
    if policy.get("status") == "RELEASED":
        vetting.validate_operational_policy(policy)
        vetting.validate_release_contract(policy)
    if (
        policy.get("pipeline_contract") != vetting.pipeline_contract(config)
        or policy.get("implementation") != pipeline.implementation_hashes()
    ):
        raise ValueError("Released policy does not match the current pipeline")


def freeze_boi_policy(
    path, pipeline, view="psf_flat", discovery_profile="all-lanes-v1"
):
    sys.path.insert(0, str(ROOT / "pipeline/vetting"))
    from pipeline.vetting import vetting

    policy = released_policy(DEFAULT_POLICY)
    config = read(ROOT / "pipeline/vetting/config.json")
    config["discovery_contract"] = (
        pipeline.detector_contract(view)
        if discovery_profile == "all-lanes-v1"
        else pipeline.detector_contract(view, discovery_profile)
    )
    if discovery_profile == "rust-transit-only-v1":
        config["flatten_mask_views"] = ["psf_flat", "psf_nova"]
    policy["pipeline_contract"] = vetting.pipeline_contract(config)
    policy["implementation"] = pipeline.implementation_hashes()
    verified_release_contract(policy, pipeline)
    path = dump_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, policy)
    return path


def verify_frozen_release(path, expected_hash, pipeline):
    if digest(path) != expected_hash:
        raise ValueError("Frozen disposition policy changed during the full survey")
    verified_release_contract(released_policy(path), pipeline)


def frozen_sources(path, expected):
    source_ids = read(path)
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != expected
        or any(
            not isinstance(source_id, str) or not source_id.isdecimal()
            for source_id in source_ids
        )
        or len(set(source_ids)) != expected
    ):
        raise ValueError("Frozen source identities are missing, repeated or malformed")
    index = ROOT / "discSSD/data/index.sqlite"
    if not index.is_file() or not (ROOT / "discPCIe/data/index.json").is_file():
        raise ValueError("Both read-only corpus volumes must be mounted")
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        indexed = dict(connection.execute("SELECT gaia_source_id, shard FROM sources"))
    if set(source_ids) != indexed.keys():
        raise ValueError("Frozen source list differs from the LC2 index")
    return sorted(set(indexed.values()))


def verified_input_ledger(path, selection, expected):
    from . import source_ledger

    frozen = read(path / "freeze.json")
    if (
        frozen.get("status") != "FROZEN_LC2_INPUT_LEDGER"
        or frozen.get("sources") != expected
        or set(frozen.get("artifacts", {}))
        != {"sources.csv", "source-ids.json", "products.json"}
        or any(
            digest(path / name) != checksum
            for name, checksum in frozen["artifacts"].items()
        )
        or read(path / "source-ids.json") != read(selection)
    ):
        raise ValueError("Complete frozen LC2 input ledger is missing or changed")
    # Rehash current corpus bytes; a completed historical freeze is insufficient.
    if source_ledger.freeze(ROOT, selection, path, expected, resume=True) != frozen:
        raise ValueError("Current corpus differs from the frozen LC2 input ledger")
    files = [
        ROOT / row["volume"] / "data" / row["path"]
        for row in read(path / "products.json")
    ]
    files += [Path(name) for name in frozen["inputs"]]
    files += [path / name for name in (*frozen["artifacts"], "freeze.json")]
    stamps = {file: source_ledger.file_stamp(file) for file in files}
    return {
        "path": str(path),
        "sha256": digest(path / "freeze.json"),
        "sources": frozen["sources"],
        "products": frozen["products"],
        "verified_input_bytes": frozen["verified_input_bytes"],
    }, stamps


def verify_input_stamps(stamps):
    from . import source_ledger

    if any(source_ledger.file_stamp(path) != stamp for path, stamp in stamps.items()):
        raise ValueError("Frozen LC2 input changed during the full survey")


def bind_stage_inputs(run_dir, identity):
    path = run_dir.with_name(run_dir.name + ".full-survey-inputs.json")
    if run_dir.exists() and not path.is_file():
        raise ValueError(
            f"Existing survey stage lacks its frozen input binding: {run_dir}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, identity)


def run_stage(command, report_path, run_dir):
    if report_path.is_file() and read(report_path).get("status") == "COMPLETE":
        return read(report_path)
    if run_dir.exists():
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)
    report = read(report_path)
    if report.get("status") != "COMPLETE":
        raise ValueError(f"Stage did not complete: {report_path}")
    return report


def stage_progress(report_path, sources):
    if not report_path.is_file():
        return {"batches": 0, "sources": 0}
    report = read(report_path)
    return {"batches": len(report["batches"]), "sources": report.get(sources, 0)}


@contextmanager
def ledger_stage(path, outputs, name, progress=dict):
    def state():
        free = {key: shutil.disk_usage(out).free for key, out in outputs.items()}
        return {**progress(), "free_bytes": free}

    def save():
        building = path.with_name(path.name + ".building")
        building.write_text(json.dumps(ledger, indent=2) + "\n")
        building.replace(path)

    ledger = read(path) if path.exists() else {"schema": 1, "stages": []}
    record = {
        "stage": name,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "start": state(),
    }
    ledger["stages"].append(record)
    save()
    wall, cpu, memory = time.monotonic(), cpu_seconds(), PeakMemory(interval=10)
    try:
        yield
        record["status"] = "COMPLETE"
    except BaseException as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(
            stopped_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=round(time.monotonic() - wall, 3),
            cpu_seconds=round(cpu_seconds() - cpu, 3),
            peak_rss_bytes=memory.stop(),
            stop=state(),
        )
        save()


def write_once(path, value):
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Completed artifact changed: {path}")
        return
    temporary = path.with_name(path.name + ".building")
    temporary.write_bytes(content)
    temporary.replace(path)


def boi_proposals(out, report):
    candidates = []
    seen = set()
    for entry in report["batches"]:
        number = entry["batch"]
        batch = out / "batches" / f"{number:06d}"
        complete = batch / "batch-complete.json"
        if digest(complete) != entry["sha256"]:
            raise ValueError(f"Completed batch changed: {number}")
        batch_report = read(complete)
        archive = read(batch / "compact.json")
        ids = read(out / "identities" / f"{number:06d}.json")
        candidate_path = batch / "vetting/bois.json"
        rows = read(candidate_path)
        if (
            batch_report["status"] != "COMPLETE"
            or batch_report["sources"] != len(ids)
            or batch_report["boi_signals"] != len(rows)
            or batch_report["source_ids_sha256"] != json_digest(ids)
            or batch_report["archive_sha256"] != json_digest(archive)
            or digest(candidate_path)
            != archive["relinked_summaries"]["bois.json"]["after_sha256"]
        ):
            raise ValueError(f"Candidate batch does not reconcile: {number}")
        for row in rows:
            key = (row["source_id"], row["signal_id"])
            if row["source_id"] not in ids or key in seen:
                raise ValueError(f"Candidate identity does not reconcile: {number}")
            evidence = Path(row["evidence"]).resolve()
            try:
                original = str(evidence.relative_to(batch))[:-3]
                expected = archive["files"][original]["archive_sha256"]
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Candidate evidence is outside the batch: {number}"
                ) from exc
            if evidence.suffix != ".gz" or digest(evidence) != expected:
                raise ValueError(f"Candidate evidence changed: {number}")
            seen.add(key)
            candidates.append({**row, "label": "BOI"})
    candidates.sort(key=lambda row: (int(row["source_id"]), row["signal_id"]))
    return candidates


def disposition_source(batch, source_id, archive):
    name = f"vetting/sources/{source_id}.json"
    path = batch / (name + ".gz")
    checks = archive["files"].get(name)
    if checks is None or digest(path) != checks["archive_sha256"]:
        raise ValueError(f"Disposition evidence changed: {source_id}")
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != checks["source_sha256"]:
        raise ValueError(f"Original disposition evidence changed: {source_id}")
    row = json.loads(payload)
    assigned = row["disposition"]
    signals = row["signals"]
    policy = row["disposition_policy"]
    signal_ids = [signal["signal_id"] for signal in signals]
    labels = [signal["signal_id"] for signal in assigned["signals"]]
    if (
        row["source_id"] != source_id
        or len(set(signal_ids)) != len(signal_ids)
        or len(set(labels)) != len(labels)
        or set(signal_ids) != set(labels)
        or policy["status"] != "BOI_CATALOGUE"
        or policy["boi_catalogue_enabled"] is not True
        or assigned.get("attributed_signal_reviews")
        or any(signal.get("reviews") for signal in assigned["signals"])
    ):
        raise ValueError(
            f"Automatic disposition identities do not reconcile: {source_id}"
        )
    allowed = {"BOI", "HELD", "DISMISSED", "UNRESOLVED", "FALSE_ALARM", "FALSE_POSITIVE"}
    if assigned["status"] not in allowed or any(
        signal["disposition"] not in allowed for signal in assigned["signals"]
    ):
        raise ValueError(f"Unknown automatic disposition: {source_id}")
    outcomes = {signal["disposition"] for signal in assigned["signals"]}
    expected = (
        "BOI"
        if "BOI" in outcomes
        else next(iter(outcomes))
        if len(outcomes) == 1
        else "UNRESOLVED"
    )
    if assigned["status"] != expected:
        raise ValueError(f"Source/signal dispositions do not reconcile: {source_id}")
    return row, path, checks["archive_sha256"]


def export_vetted(batch, ids, archive, sources, signals, candidate_keys):
    counts = Counter()
    for source_id in ids:
        row, evidence, evidence_hash = disposition_source(batch, source_id, archive)
        assigned = row["disposition"]
        common = {
            "source_id": source_id,
            "batch": int(batch.name),
            "evidence": str(evidence),
            "evidence_sha256": evidence_hash,
            "policy_id": assigned["policy_id"],
        }
        sources.writerow(
            {
                **common,
                "stage": "VETTED",
                "disposition": assigned["status"],
                "signal_count": len(assigned["signals"]),
                "reason": "",
            }
        )
        measured = {signal["signal_id"]: signal for signal in row["signals"]}
        for signal in assigned["signals"]:
            label = signal["disposition"]
            key = (source_id, signal["signal_id"])
            if (label == "BOI") != (key in candidate_keys):
                raise ValueError(f"Candidate/disposition mismatch: {key}")
            original = measured[signal["signal_id"]]
            signals.writerow(
                {
                    **common,
                    "signal_id": signal["signal_id"],
                    "disposition": label,
                    "triage_state": original.get("triage", {}).get("status"),
                    "failure_stage": original.get("triage", {}).get("held_stage", original.get("triage", {}).get("dismissed_stage")),
                    "failure_reason": original.get("triage", {}).get("reason"),
                    "detector_classifications": json.dumps(sorted({h["rust_record"]["classification"] for h in original.get("timing_hypotheses", []) if "rust_record" in h}), separators=(",", ":")),
                    "period_days": original.get("period_days"),
                    "event_units": len(original["events"]),
                    "reason_codes": json.dumps(
                        signal["reason_codes"], separators=(",", ":")
                    ),
                    "signal_shape": json.dumps(
                        signal["signal_shape"], separators=(",", ":")
                    ),
                    "evidence_assessment": json.dumps(
                        signal["evidence_assessment"], separators=(",", ":")
                    ),
                    "reproduced_mechanisms": json.dumps(
                        signal["reproduced_mechanisms"], separators=(",", ":")
                    ),
                    "risk_features": json.dumps(
                        signal["risk_features"], separators=(",", ":")
                    ),
                }
            )
            counts[label] += 1
    return counts


def finalize_csv(path):
    building = path.with_suffix(path.suffix + ".building")
    if path.exists():
        if digest(path) != digest(building):
            raise ValueError(f"Completed disposition export changed: {path}")
        building.unlink()
    else:
        building.replace(path)
    return digest(path)


def export_unvetted(survey, sources, seen):
    stage_counts = Counter({"VETTED": len(seen)})
    with (survey / "cohort.csv").open(newline="") as stream:
        cohort = [row["source_id"] for row in csv.DictReader(stream)]
    if len(set(cohort)) != len(cohort) or not seen <= set(cohort):
        raise ValueError("Vetted sources do not reconcile with the scanned cohort")
    for source_id in cohort:
        if source_id in seen:
            continue
        sources.writerow(
            {
                "source_id": source_id,
                "stage": "SCANNED_NO_FOLLOWUP",
                "signal_count": 0,
            }
        )
        seen.add(source_id)
        stage_counts["SCANNED_NO_FOLLOWUP"] += 1
    with (survey / "ineligible.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source_id"] in seen:
                raise ValueError("Ineligible source identity repeats")
            sources.writerow(
                {
                    "source_id": row["source_id"],
                    "stage": "INELIGIBLE",
                    "reason": row["reason"],
                }
            )
            seen.add(row["source_id"])
            stage_counts["INELIGIBLE"] += 1
    return stage_counts, len(cohort)


def export_dispositions(survey, out, survey_report, vet_report, candidates):
    for name in ("cohort.csv", "ineligible.csv"):
        if digest(survey / name) != survey_report["artifacts"][name]:
            raise ValueError(f"Frozen disposition source ledger changed: {name}")
    source_path = out / "source-dispositions.csv"
    signal_path = out / "signal-dispositions.csv"
    common = ["source_id", "batch", "evidence", "evidence_sha256", "policy_id"]
    source_fields = common + ["stage", "disposition", "signal_count", "reason"]
    signal_fields = common + [
        "triage_state", "failure_stage", "failure_reason", "detector_classifications",
        "signal_id",
        "disposition",
        "period_days",
        "event_units",
        "reason_codes",
        "signal_shape",
        "evidence_assessment",
        "reproduced_mechanisms",
        "risk_features",
    ]
    candidate_keys = {(row["source_id"], row["signal_id"]) for row in candidates}
    seen = set()
    counts = Counter()
    with (
        source_path.with_suffix(".csv.building").open("w", newline="") as source_stream,
        signal_path.with_suffix(".csv.building").open("w", newline="") as signal_stream,
    ):
        sources = csv.DictWriter(source_stream, fieldnames=source_fields)
        signals = csv.DictWriter(signal_stream, fieldnames=signal_fields)
        sources.writeheader()
        signals.writeheader()
        for entry in vet_report["batches"]:
            batch = out / "batches" / f"{entry['batch']:06d}"
            if digest(batch / "batch-complete.json") != entry["sha256"]:
                raise ValueError("Completed disposition batch changed")
            complete = read(batch / "batch-complete.json")
            archive = read(batch / "compact.json")
            ids = read(out / "identities" / f"{entry['batch']:06d}.json")
            if (
                complete["status"] != "COMPLETE"
                or complete["sources"] != len(ids)
                or complete["source_ids_sha256"] != json_digest(ids)
                or complete["archive_sha256"] != json_digest(archive)
                or len(set(ids)) != len(ids)
                or seen.intersection(ids)
                or {
                    name
                    for name in archive["files"]
                    if name.startswith("vetting/sources/")
                }
                != {f"vetting/sources/{source_id}.json" for source_id in ids}
            ):
                raise ValueError("Disposition batch identities do not reconcile")
            counts.update(
                export_vetted(batch, ids, archive, sources, signals, candidate_keys)
            )
            seen.update(ids)
        vetted = len(seen)
        stage_counts, scanned = export_unvetted(survey, sources, seen)
    if (
        vetted != vet_report["completed_sources"]
        or scanned != survey_report["accepted_sources"]
        or len(seen) != survey_report["attempted_sources"]
        or stage_counts["INELIGIBLE"] != survey_report["ineligible_sources"]
        or counts["BOI"] != len(candidate_keys)
    ):
        raise ValueError("Final source/signal disposition counts do not reconcile")
    return {
        "schema": 1,
        "sources": len(seen),
        "source_stages": dict(stage_counts),
        "signals": sum(counts.values()),
        "signal_dispositions": dict(counts),
        "artifacts": {
            path.name: finalize_csv(path) for path in (source_path, signal_path)
        },
        "boundary": (
            "Automatic frozen-policy labels from hash-verified source evidence. "
            "Scanned sources without follow-up and ineligible sources carry stage "
            "status, not an invented false-signal disposition."
        ),
    }


def full_survey(args):
    from . import run as pipeline

    source_ids = (ROOT / args.source_ids).resolve()
    survey = dump_path(args.survey)
    vetting = dump_path(args.vetting)
    policy = (ROOT / args.disposition_policy).resolve()
    ledger = dump_path(args.input_ledger)
    view = getattr(args, "discovery_view", "psf_flat")
    profile = getattr(args, "discovery_profile", "all-lanes-v1")
    if survey == vetting or survey in vetting.parents or vetting in survey.parents:
        raise ValueError("Survey and vetting runs need separate directories")
    outputs = {"survey": survey.parent, "vetting": vetting.parent}
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
    stage = partial(
        ledger_stage,
        vetting.with_name(vetting.name + ".full-survey-ledger.json"),
        outputs,
    )
    policy_hash = digest(policy)
    with stage("verify_inputs"):
        verify_frozen_release(policy, policy_hash, pipeline)
        declared = read(policy).get("pipeline_contract", {}).get("discovery", {})
        if declared and (
            declared.get("discovery_view") != view
            or declared.get("discovery_profile", "all-lanes-v1") != profile
        ):
            raise ValueError("Survey discovery view differs from the frozen policy")
        inputs, stamps = verified_input_ledger(
            ledger, source_ids, args.expected_sources
        )
        shards = frozen_sources(source_ids, args.expected_sources)
    selection_hash = digest(source_ids)
    identity = {
        "schema": 1,
        "input_ledger": inputs,
        "source_ids_sha256": selection_hash,
        "disposition_policy_sha256": policy_hash,
        "runner_sha256": digest(Path(__file__)),
    }
    bind_stage_inputs(survey, identity)
    bind_stage_inputs(vetting, identity)
    verify_frozen_release(policy, policy_hash, pipeline)
    progress = partial(stage_progress, survey / "survey.json", "attempted_sources")
    with stage("survey", progress):
        survey_report = run_stage(
            [
                sys.executable,
                "pipeline/run.py",
                "deterministic",
                "survey",
                "--source-ids",
                str(source_ids),
                "--shards",
                *map(str, shards),
                "--limit",
                str(args.expected_sources),
                "--batch-size",
                str(args.survey_batch_size),
                "--threads",
                str(args.threads),
                "--disposition-policy",
                str(policy),
                "--discovery-view",
                view,
                "--discovery-profile",
                profile,
                "--run",
                str(survey),
            ],
            survey / "survey.json",
            survey,
        )
    if (
        survey_report["requested_sources"] != args.expected_sources
        or survey_report["attempted_sources"] != args.expected_sources
        or survey_report["selection"]["source_ids"]["sha256"] != selection_hash
        or survey_report["selection"]["shards"] != shards
        or survey_report["detector_contract"] != pipeline.args_detector_contract(args)
        or survey_report["implementation"] != pipeline.implementation_hashes()
        or survey_report.get("disposition_policy") != str(policy)
        or survey_report.get("disposition_policy_sha256") != policy_hash
    ):
        raise ValueError("Completed survey does not match frozen source identities")
    verify_input_stamps(stamps)
    verify_frozen_release(policy, policy_hash, pipeline)
    progress = partial(stage_progress, vetting / "vet-survey.json", "completed_sources")
    with stage("vetting", progress):
        vet_report = run_stage(
            [
                sys.executable,
                "-m",
                "pipeline.deterministic.vet_survey",
                "--survey",
                str(survey),
                "--run",
                str(vetting),
                "--batch-size",
                str(args.vetting_batch_size),
                "--threads",
                str(args.threads),
                "--workers",
                str(args.workers),
                "--concurrent-batches",
                str(args.concurrent_vetting_batches),
                "--disposition-policy",
                str(policy),
            ]
            + (
                ["--pixel-checkpoint-ledger", str(args.pixel_checkpoint_ledger)]
                if getattr(args, "pixel_checkpoint_ledger", None) is not None
                else []
            ),
            vetting / "vet-survey.json",
            vetting,
        )
    identity = vet_report["identity"]
    if (
        identity["survey_sha256"] != digest(survey / "survey.json")
        or identity["disposition_policy"]["sha256"] != policy_hash
        or identity["implementation"] != pipeline.implementation_hashes()
        or identity["batch_size"] != args.vetting_batch_size
        or identity["threads"] != args.threads
        or identity["workers"] != args.workers
        or identity["concurrent_batches"] != args.concurrent_vetting_batches
        or vet_report["completed_sources"] != identity["followup_sources"]
    ):
        raise ValueError("Completed vetting does not match the frozen survey or policy")
    verify_input_stamps(stamps)
    verify_frozen_release(policy, policy_hash, pipeline)
    with stage("labels"):
        candidates = boi_proposals(vetting, vet_report)
        catalogue = vetting / "bois.json"
        write_once(catalogue, candidates)
        dispositions = export_dispositions(
            survey, vetting, survey_report, vet_report, candidates
        )
        write_once(vetting / "dispositions.json", dispositions)
        from pipeline.vetting.boi import catalogue as group_bois

        records = []
        for entry in vet_report["batches"]:
            batch = vetting / "batches" / f"{entry['batch']:06d}"
            archive = read(batch / "compact.json")
            path = batch / "vetting/signal-ledger.json"
            if (
                digest(path)
                != archive["relinked_summaries"]["signal-ledger.json"]["after_sha256"]
            ):
                raise ValueError("Completed signal ledger changed")
            records.extend(read(path))
        grouped = group_bois(records)
        write_once(vetting / "boi-catalogue.json", grouped)
    summary = {
        "schema": 1,
        "status": "COMPLETE",
        "source_ids_sha256": selection_hash,
        "input_ledger": inputs,
        "attempted_sources": survey_report["attempted_sources"],
        "scanned_sources": survey_report["accepted_sources"],
        "ineligible_sources": survey_report["ineligible_sources"],
        "vetted_sources": vet_report["completed_sources"],
        "boi_signals": len(candidates),
        "bois": grouped["bois"],
        "discovery_view": view,
        "boi_catalogue_sha256": digest(vetting / "boi-catalogue.json"),
        "survey_sha256": digest(survey / "survey.json"),
        "vetting_sha256": digest(vetting / "vet-survey.json"),
        "disposition_policy_sha256": policy_hash,
        "bois_sha256": digest(catalogue),
        "dispositions_sha256": digest(vetting / "dispositions.json"),
        "runner_sha256": digest(Path(__file__)),
    }
    write_once(vetting / "full-survey.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-policy", type=Path)
    parser.add_argument(
        "--discovery-view",
        choices=("psf_flat", "psf_qdiag", "sap_qdiag"),
        default="psf_flat",
    )
    parser.add_argument(
        "--discovery-profile",
        choices=("all-lanes-v1", "rust-transit-only-v1"),
        default="all-lanes-v1",
    )
    parser.add_argument("--source-ids", type=Path)
    parser.add_argument("--input-ledger", type=Path)
    parser.add_argument("--survey", type=Path)
    parser.add_argument("--vetting", type=Path)
    parser.add_argument("--disposition-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--expected-sources", type=int, default=406547)
    parser.add_argument("--survey-batch-size", type=int, default=500)
    parser.add_argument("--vetting-batch-size", type=int, default=50)
    parser.add_argument("--threads", type=int, default=worker_count())
    parser.add_argument("--workers", type=int, default=worker_count())
    parser.add_argument("--concurrent-vetting-batches", type=int, default=1)
    parser.add_argument("--pixel-checkpoint-ledger", type=Path)
    args = parser.parse_args()
    if args.freeze_policy:
        from . import run as pipeline

        path = freeze_boi_policy(
            args.freeze_policy, pipeline, args.discovery_view, args.discovery_profile
        )
        print(json.dumps({"policy": str(path), "sha256": digest(path)}))
        return
    if any(
        getattr(args, name) is None
        for name in ("source_ids", "input_ledger", "survey", "vetting")
    ):
        parser.error(
            "Survey execution requires source IDs, input ledger, survey and vetting paths"
        )
    if (
        min(
            args.expected_sources,
            args.survey_batch_size,
            args.vetting_batch_size,
            args.threads,
            args.workers,
            args.concurrent_vetting_batches,
        )
        < 1
    ):
        parser.error("Counts, batch sizes and workers must be positive")
    full_survey(args)


if __name__ == "__main__":
    main()
