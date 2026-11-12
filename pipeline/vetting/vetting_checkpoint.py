"""Checked transfer of completed evidence to the pre-pixel stage of a repair."""

import copy
import gzip
import json
import shutil
from collections import Counter
from pathlib import Path

from runtime import implementation_hashes, json_sha256
from vetting_disposition import check_gate_record

PRE_PIXEL = ("shape", "orbital", "instrument")
CONFIG_PATHS = {
    "run_directory",
    "queue",
    "pixel_cache_directory",
    "shared_alert_index",
    "disposition_policy",
    "pixel_checkpoint_ledger",
}
ROW_DERIVED = {
    "localization",
    "triage",
    "disposition",
    "disposition_policy",
    "disposition_execution",
    "processing_status",
    "literature",
}
SIGNAL_DERIVED = {
    "triage",
    "localization",
    "contamination",
    "host",
    "signal_assessments",
}
EVENT_DERIVED = {"source_association", "known_ephemerides"}


def without(value, fields):
    return {key: item for key, item in value.items() if key not in fields}


def scientific_projection(row):
    result = without(row, ROW_DERIVED)
    result["events"] = [without(event, EVENT_DERIVED) for event in row["events"]]
    result["signals"] = [without(signal, SIGNAL_DERIVED) for signal in row["signals"]]
    result["pre_pixel_gates"] = [
        [gate for gate in signal["triage"]["history"] if gate["stage"] in PRE_PIXEL]
        for signal in row["signals"]
    ]
    return result


def triage_summary(signals):
    states = Counter(signal["triage"]["status"] for signal in signals)
    return {
        "status": "MEASURED",
        "last_stage": "instrument",
        "signals_total": len(signals),
        "signals_active": states["ACTIVE"],
        "signals_held": states["HELD"],
        "signals_dismissed": states["DISMISSED"],
        "held_by_stage": dict(
            Counter(
                signal["triage"]["held_stage"]
                for signal in signals
                if signal["triage"]["status"] == "HELD"
            )
        ),
        "dismissed_by_stage": dict(
            Counter(
                signal["triage"]["dismissed_stage"]
                for signal in signals
                if signal["triage"]["status"] == "DISMISSED"
            )
        ),
    }


def rewind_signal(signal):
    check_gate_record(signal)
    history = signal["triage"]["history"]
    prefix = history[:3]
    if len(prefix) == 3 and all(gate["decision"] == "PASS" for gate in prefix):
        signal["triage"] = {"status": "ACTIVE", "history": prefix}
        for field in SIGNAL_DERIVED - {"triage"}:
            signal.pop(field, None)
        return 1
    if any(gate["stage"] not in PRE_PIXEL for gate in history):
        raise ValueError("Unsupported checkpoint gate history")
    return 0


def rewind_row(original):
    row = copy.deepcopy(original)
    reopened = sum(rewind_signal(signal) for signal in row["signals"])
    for event in row["events"]:
        for field in EVENT_DERIVED:
            event.pop(field, None)
    for field in ROW_DERIVED:
        row.pop(field, None)
    row["localization"] = {"status": "NOT_RUN", "events": []}
    row["triage"] = triage_summary(row["signals"])
    if scientific_projection(row) != scientific_projection(original):
        raise ValueError("Pre-pixel scientific evidence changed during transfer")
    return row, reopened


def packed(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def checkpoint(config, run):
    from vetting import digest, read

    binding = config.get("pixel_checkpoint_ledger")
    if binding is None:
        return None
    path = Path(binding["path"])
    if digest(path) != binding["sha256"]:
        raise ValueError("Pixel checkpoint ledger changed")
    ledger = read(path)
    if (
        ledger["implementation"] != implementation_hashes()
        or ledger["status"] != "VERIFIED_PRE_PIXEL_REPAIR"
        or not run.is_relative_to(Path(ledger["destination"]))
    ):
        raise ValueError("Pixel checkpoint implementation or destination changed")
    report_path = Path(ledger["original_report"]["path"])
    if digest(report_path) != ledger["original_report"]["sha256"]:
        raise ValueError("Original checkpoint report changed")
    report = read(report_path)
    records = {item["batch"]: item for item in report["batches"]}
    number = int(run.parent.name)
    if number not in records:
        return None
    return ledger, report, records[number]


def source_identity(preparation):
    sources = copy.deepcopy(preparation["sources"])
    for source in sources:
        for item in source["inputs"].values():
            item.pop("fsc_path")
    return sources


def cohort_identity(cohort):
    rows = copy.deepcopy(cohort)
    for row in rows:
        if "shallow_eligibility" in row:
            row["shallow_eligibility"].pop("elapsed_seconds", None)
    return rows


def compare_inputs(original, destination, config):
    from vetting import read

    old_config = read(original / "vetting.json")
    if config["triage_execution"] != "SKIP_AFTER_FIRST_FAILURE":
        raise ValueError("Checkpoint repair requires the first-failure execution contract")
    if without(old_config, CONFIG_PATHS) != without(config, CONFIG_PATHS):
        raise ValueError("Checkpoint upstream configuration changed")
    old_policy = read(Path(old_config["disposition_policy"]))
    new_policy = read(Path(config["disposition_policy"]))
    if without(old_policy, {"implementation"}) != without(
        new_policy, {"implementation"}
    ):
        raise ValueError("Checkpoint scientific policy changed")
    if source_identity(read(original / "preparation.json")) != source_identity(
        read(destination / "preparation.json")
    ):
        raise ValueError("Checkpoint source arrays, masks or prepared inputs changed")
    old_manifest = read(original / "vetting/manifest.json")
    new_manifest = read(destination / "vetting/manifest.json")
    if old_manifest["catalogue_sha256"] != new_manifest["catalogue_sha256"]:
        raise ValueError("Checkpoint catalogue snapshots changed")
    for filename in ("cohort", "metadata", "scene"):
        previous = packed(original / f"vetting/{filename}.json.gz")
        current = read(destination / f"vetting/{filename}.json")
        if filename == "cohort":
            previous, current = cohort_identity(previous), cohort_identity(current)
        if previous != current:
            raise ValueError(f"Checkpoint {filename} changed")


def restore(selected):
    from vetting import CONFIG, RUN, digest, prepare, read, save

    from pipeline.deterministic.vet_survey import verify_batch

    ledger, report, record = selected
    original = Path(ledger["original_directory"]) / "batches" / f"{record['batch']:06d}"
    if original == RUN.parent:
        raise ValueError("A repair must retain its original generation")
    verify_batch(original, record, report["identity"])
    prepare()
    compare_inputs(original, RUN.parent, read(CONFIG))
    rows, reopened = {}, 0
    for sid in read(RUN / "manifest.json")["source_ids"]:
        original_row = packed(original / f"vetting/sources/{sid}.json.gz")
        restored, count = rewind_row(original_row)
        save(RUN / "sources" / f"{sid}.json", restored, compact=True)
        rows[sid] = json_sha256(scientific_projection(original_row))
        reopened += count
    for directory in ("queries", "stellar"):
        for path in (original / "vetting" / directory).glob("*.json"):
            destination = RUN / directory / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if digest(path) != digest(destination):
                raise ValueError("Checkpoint query transfer changed")
    save(
        RUN / "pixel-checkpoint.json",
        {
            "schema": 1,
            "stage": "END_OF_INSTRUMENT_GATE",
            "original": str(original),
            "original_batch_receipt": record,
            "original_archive_sha256": digest(original / "compact.json"),
            "reopened_signals": reopened,
            "upstream_science_sha256": rows,
            "inherited_stage_timings": read(original / "vetting/stage-timings.json"),
            "timing_scope": "Inherited upstream timings are provenance; repair wall time is recorded separately.",
        },
    )


def verify_preserved():
    from vetting import RUN, read

    path = RUN / "pixel-checkpoint.json"
    if not path.exists():
        return
    receipt = read(path)
    for sid, expected in receipt["upstream_science_sha256"].items():
        row = read(RUN / "sources" / f"{sid}.json")
        if json_sha256(scientific_projection(row)) != expected:
            raise ValueError(
                f"Repaired downstream stages changed upstream science: {sid}"
            )
