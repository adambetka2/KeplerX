"""Annotate completed vetting: BOI period flags, then catalogue-only other signals."""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PERIOD_RULE = {
    "id": "same-source-period-ratio-flags-v1",
    "maximum_numerator_or_denominator": 15,
    "maximum_relative_period_error": 0.001,
    "scope": "BOI proposals with the same Gaia source only. Period ratios flag possible aliases or harmonics; transit timings, residuals and independence are not tested. No proposal is removed or relabelled.",
}
HOST_RELATIONS = {"SAME_GAIA_ID", "POSITION_COMPATIBLE_WITHIN_2_ARCSEC"}
REQUIRED_CATALOGUES = {
    "NASA_cumulative_KOI", "NASA_Kepler_DR25_TCE",
    "NASA_confirmed_planets_PS", "Villanova_Kepler_EB_live",
}
RATIOS = [(m, n) for m in range(1, 16) for n in range(1, m + 1)
          if math.gcd(m, n) == 1]
SIGNAL_FIELDS = (
    "source_id", "signal_id", "discovery_view", "label", "triage_state",
    "failure_stage", "reason", "period_days", "epoch_bkjd", "duration_days",
    "evidence", "catalogue_state",
)
PAIR_FIELDS = (
    "source_id", "long_signal_id", "short_signal_id", "long_period_days",
    "short_period_days", "numerator", "denominator", "relative_period_error",
)
MATCH_FIELDS = (
    "source_id", "signal_id", "label", "catalogue", "name", "planet_name",
    "classification", "host_relation", "period", "epoch", "time_reference",
    "separation_arcsec", "rational_numerator", "rational_denominator",
    "rational_relative_mismatch", "evidence_url", "evidence",
)
WORKER_MATCHER = None
WORKER_SNAPSHOT = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_bound(path, bindings, expected=None):
    path = Path(path).resolve()
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"Input hash differs: {path}")
    bindings[str(path)] = actual
    return json.loads(data)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def period_pair(left, right):
    if left["source_id"] != right["source_id"]:
        return None
    if left["label"] != "BOI" or right["label"] != "BOI":
        raise ValueError("Period flags are restricted to BOIs")
    values = [left["period_days"], right["period_days"]]
    if any(value is None for value in values):
        return None
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Invalid periodic proposal")
    long, short = sorted((left, right), key=lambda row: -row["period_days"])
    ratio = long["period_days"] / short["period_days"]
    m, n = min(RATIOS, key=lambda pair: abs(ratio / (pair[0] / pair[1]) - 1))
    error = abs(ratio / (m / n) - 1)
    if error > PERIOD_RULE["maximum_relative_period_error"]:
        return None
    return dict(zip(PAIR_FIELDS, (
        left["source_id"], long["signal_id"], short["signal_id"],
        long["period_days"], short["period_days"], m, n, error,
    )))


def period_flags(records):
    by_source = defaultdict(list)
    for row in records:
        if row["label"] != "BOI":
            raise ValueError("Period flags are restricted to BOIs")
        by_source[row["source_id"]].append(row)
    return [pair for rows in by_source.values()
            for left, right in itertools.combinations(rows, 2)
            if (pair := period_pair(left, right)) is not None]


def load_batch(batch, bindings):
    result = read_bound(batch["result"], bindings, batch["result_sha256"])
    if result["processing_status"] != "EVIDENCE_STAGES_COMPLETE":
        raise ValueError("Batch evidence is incomplete")
    root = Path(batch["result"]).parent
    config = read_bound(root.parent / "config.json", bindings)
    metadata = read_bound(root / "metadata.json", bindings)
    rows = read_bound(result["signal_ledger"], bindings)
    if len(rows) != batch["proposals_supplied"]:
        raise ValueError("Batch proposal accounting differs")
    selected = []
    for row in rows:
        source = metadata[row["source_id"]]
        selected.append({**{key: row.get(key) for key in SIGNAL_FIELDS},
                         "host_kic": source["kic"],
                         "aperture_kic": source["origin_tpf_kic"],
                         "ra_deg": source["ra_deg"], "dec_deg": source["dec_deg"]})
    return selected, config["catalogue_search"]


def load_run(run):
    bindings = {}
    progress = read_bound(run / "progress.json", bindings)
    if progress["state"] != "COMPLETE":
        raise ValueError("Wait until the entire vetting pipeline finishes")
    final = read_bound(run / "catalogue.json", bindings)
    if final["status"] != "COMPLETE":
        raise ValueError("Wait until the entire vetting pipeline finishes")
    records, specs = [], []
    for batch in final["batches"]:
        rows, spec = load_batch(batch, bindings)
        records.extend(rows)
        specs.append(spec)
    if any(spec != specs[0] for spec in specs):
        raise ValueError("Catalogue snapshot differs between batches")
    identities = {(r["source_id"], r["discovery_view"], r["signal_id"]) for r in records}
    if len(identities) != len(records) or len(records) != final["proposals_supplied"]:
        raise ValueError("Final proposal accounting differs")
    if len({r["source_id"] for r in records}) != final["sources"]:
        raise ValueError("Final source accounting differs")
    spec = specs[0]
    manifest = read_bound(spec["manifest"], bindings, spec["manifest_sha256"])
    if not REQUIRED_CATALOGUES <= set(manifest["required_catalogues"]):
        raise ValueError("A required known-signal catalogue is absent")
    signals = manifest["signals"]
    path = Path(spec["manifest"]).parent / signals["path"]
    if digest(path) != signals["sha256"]:
        raise ValueError("Catalogue signal table hash differs")
    bindings[str(path.resolve())] = signals["sha256"]
    code = run / "source/pipeline/vetting/catalogue_search.py"
    bindings[str(code.resolve())] = digest(code)
    return records, bindings, manifest, {**spec, "code": str(code.resolve())}


def initialize_catalogue(spec):
    global WORKER_MATCHER, WORKER_SNAPSHOT
    module = importlib.util.spec_from_file_location("frozen_catalogue_search", spec["code"])
    WORKER_MATCHER = importlib.util.module_from_spec(module)
    module.loader.exec_module(WORKER_MATCHER)
    WORKER_SNAPSHOT = WORKER_MATCHER.Snapshot(spec["manifest"], spec["manifest_sha256"])


def catalogue_report(source, signal, snapshot_sha256):
    saved = signal.get("catalogue_search", {})
    if saved.get("status") == "COMPLETE":
        if saved["manifest_sha256"] != snapshot_sha256:
            raise ValueError("Source uses a different catalogue snapshot")
        if saved["rule"]["id"] != "complete-catalogue-common-recurrence-v1":
            raise ValueError("Source uses a different catalogue rule")
        report, origin = saved, "REUSED_COMPLETE_COMPARISON"
    else:
        report = WORKER_MATCHER.search(source, signal, WORKER_SNAPSHOT)
        origin = "COMPUTED_FROM_FROZEN_INPUTS"
    return {**{key: report[key] for key in (
        "status", "matches", "incomplete_pairs", "hypotheses_compared",
    )}, "comparison_origin": origin, "previous_status": saved.get("status")}


def source_reports(task):
    path, snapshot_sha256, source, selected_ids = task
    data = Path(path).read_bytes()
    row = json.loads(data)
    if row["source_id"] != source["source_id"]:
        raise ValueError(f"Source identity differs: {path}")
    reports = {signal["signal_id"]: catalogue_report(source, signal, snapshot_sha256)
               for signal in row["signals"] if signal["signal_id"] in selected_ids}
    if set(reports) != selected_ids:
        raise ValueError(f"Missing requested signal identity: {path}")
    return str(Path(path).resolve()), hashlib.sha256(data).hexdigest(), reports


def collect_reports(records, reports, bindings, spec, workers):
    sources = {r["evidence"]: {"source_id": r["source_id"], "kic": r["host_kic"],
                              "ra_deg": r["ra_deg"], "dec_deg": r["dec_deg"]}
               for r in records if r["signal_id"] not in reports}
    selected = defaultdict(set)
    for row in records:
        selected[row["evidence"]].add(row["signal_id"])
    tasks = [(path, spec["manifest_sha256"], source, selected[path])
             for path, source in sorted(sources.items())]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_catalogue,
                             initargs=(spec,)) as pool:
        for count, (path, sha256, found) in enumerate(pool.map(source_reports, tasks), 1):
            if set(found) & set(reports):
                raise ValueError("Repeated signal identity across source evidence")
            if path in bindings and bindings[path] != sha256:
                raise ValueError(f"Source evidence changed between stages: {path}")
            bindings[path] = sha256
            reports.update(found)
            if count % 100 == 0:
                print(json.dumps({"utc": utc(), "source_files_read": count,
                                  "source_files_in_stage": len(tasks)}), flush=True)


def match_rows(records, reports):
    return [{**match, "source_id": row["source_id"], "signal_id": row["signal_id"],
             "label": row["label"], "evidence": row["evidence"]}
            for row in records for match in reports[row["signal_id"]]["matches"]]


def known_confirmed(report):
    return [match for match in report["matches"]
            if match["classification"] == "CONFIRMED"
            and match["host_relation"] in HOST_RELATIONS]


def annotated_rows(records, reports, pairs):
    partners = defaultdict(set)
    for pair in pairs:
        left, right = pair["long_signal_id"], pair["short_signal_id"]
        partners[left].add(right)
        partners[right].add(left)
    for row in records:
        report = reports[row["signal_id"]]
        yield {**row, "possible_harmonic_or_alias": bool(partners[row["signal_id"]]) if row["label"] == "BOI" else None,
               "possible_harmonic_or_alias_partners": ";".join(sorted(partners[row["signal_id"]])),
               "period_flag_scope": "BOI_PERIOD_RATIOS_ONLY" if row["label"] == "BOI" else "NOT_RUN_AS_REQUESTED",
               "catalogue_search_status": report["status"],
               "catalogue_matches": len(report["matches"]),
               "host_compatible_confirmed_planet_associations": len(known_confirmed(report))}


def stage_summary(records, reports, pairs):
    confirmed = [row for row in records if known_confirmed(reports[row["signal_id"]])]
    names = {match.get("planet_name") or match["name"] for row in confirmed
             for match in known_confirmed(reports[row["signal_id"]])}
    return {
        "proposals": len(records), "sources": len({r["source_id"] for r in records}),
        "labels": dict(Counter(row["label"] for row in records)),
        "catalogue_statuses": dict(Counter(reports[r["signal_id"]]["status"] for r in records)),
        "comparison_origins": dict(Counter(reports[r["signal_id"]]["comparison_origin"] for r in records)),
        "catalogue_matched_proposals": sum(bool(reports[r["signal_id"]]["matches"]) for r in records),
        "host_compatible_confirmed_planet_proposals": len(confirmed),
        "host_compatible_confirmed_planet_sources": len({r["source_id"] for r in confirmed}),
        "host_compatible_confirmed_planet_names": sorted(names),
        "possible_harmonic_or_alias_pairs": len(pairs),
        "possible_harmonic_or_alias_proposals": len({p[k] for p in pairs for k in ("long_signal_id", "short_signal_id")}),
        "confirmed_planet_proposals_by_disposition": dict(Counter(r["label"] for r in confirmed)),
    }


def write_stage(output, records, reports, pairs, scope):
    output.mkdir()
    rows = list(annotated_rows(records, reports, pairs))
    fields = list(SIGNAL_FIELDS) + ["host_kic", "aperture_kic", "ra_deg", "dec_deg"] + [
        "possible_harmonic_or_alias", "possible_harmonic_or_alias_partners",
        "period_flag_scope", "catalogue_search_status", "catalogue_matches",
        "host_compatible_confirmed_planet_associations",
    ]
    write_csv(output / "signals.csv", rows, fields)
    write_csv(output / "catalogue-matches.csv", match_rows(records, reports), MATCH_FIELDS)
    write_csv(output / "possible-harmonic-alias-pairs.csv", pairs, PAIR_FIELDS)
    with (output / "catalogue-checks.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps({"source_id": row["source_id"],
                                     "signal_id": row["signal_id"], "label": row["label"],
                                     **reports[row["signal_id"]]}, allow_nan=False) + "\n")
    summary = {"status": "COMPLETE", "utc": utc(), "scope": scope,
               **stage_summary(records, reports, pairs)}
    save(output / "summary.json", summary)
    return summary


def annotate(run, output, workers=4):
    records, bindings, manifest, spec = load_run(run)
    if output.exists():
        raise FileExistsError(f"Derived output already exists: {output}")
    output.mkdir(parents=True)
    reports = {}
    bois = [r for r in records if r["label"] == "BOI"]
    other = [r for r in records if r["label"] != "BOI"]
    collect_reports(bois, reports, bindings, spec, workers)
    pairs = period_flags(bois)
    first = write_stage(output / "bois", bois, reports, pairs,
                        "BOIs only: possible period aliases/harmonics and catalogue checks")
    print(json.dumps({"stage": "bois", **first}), flush=True)
    collect_reports(other, reports, bindings, spec, workers)
    second = write_stage(output / "other-signals", other, reports, [],
                         "Catalogue checks only; no harmonic/alias tests on non-BOIs")
    if len(records) != first["proposals"] + second["proposals"]:
        raise ValueError("Derived stage accounting differs")
    summary = {
        "status": "COMPLETE", "utc": utc(), "vetting_run": str(run.resolve()),
        "period_rule": PERIOD_RULE, "catalogue_snapshot_prepared_utc": manifest["prepared_utc"],
        "catalogue_snapshot_sha256": spec["manifest_sha256"],
        "catalogue_sources": manifest["required_catalogues"],
        "bois": first, "other_signals": second,
        "all_proposals": stage_summary(records, reports, pairs),
        "inputs": bindings,
        "limitations": "Complete saved comparisons are reused; missing or incomplete comparisons run the frozen matcher on original signal inputs. No FITS reads or new network queries. Same-host confirmed-planet associations are candidate recovery evidence; nearby objects are separate. The denominator is detected proposals, not all known planets in the searched population. Original dispositions, measurements and timing hypotheses remain unchanged. No physical validation, novelty, residual subtraction or independence claim.",
    }
    summary["outputs"] = {str(p.relative_to(output)): digest(p)
                          for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "summary.json", summary)
    print(json.dumps({"stage": "complete", "utc": utc(),
                      "output": str(output), "proposals": len(records)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    annotate(args.run, args.output, args.workers)


if __name__ == "__main__":
    main()
