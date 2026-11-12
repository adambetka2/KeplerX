"""Freeze untouched deterministic-pipeline control pools."""

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("control-plan.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ordered(values, seed, label):
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{label}:{value}".encode()).digest(),
    )


def source_rows(index):
    uri = f"file:{index.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        rows = database.execute(
            "select gaia_source_id, volume, shard from sources"
        ).fetchall()
    return {
        source_id: {"source_id": source_id, "volume": volume, "shard": shard}
        for source_id, volume, shard in rows
    }


def excluded_sources(specifications):
    excluded = set()
    evidence = []
    for specification in specifications:
        path = ROOT / specification["path"]
        rows = read_csv(path)
        values = {row[specification["column"]] for row in rows}
        excluded.update(values)
        evidence.append(
            {
                "path": specification["path"],
                "sha256": digest(path),
                "rows": len(rows),
                "unique_sources": len(values),
            }
        )
    return excluded, evidence


def select_source_pools(available, excluded, cohorts, seed):
    candidates = ordered(set(available) - excluded, seed, "source")
    required = sum(specification["pool"] for specification in cohorts.values())
    if len(candidates) < required:
        raise ValueError("Too few untouched LC2 sources for the control pools")
    rows = []
    offset = 0
    for category, specification in cohorts.items():
        pool = candidates[offset : offset + specification["pool"]]
        offset += len(pool)
        for rank, source_id in enumerate(pool, 1):
            rows.append(
                {
                    "category": category,
                    "pool_rank": rank,
                    "target_count": specification["target"],
                    "pool_count": specification["pool"],
                    **available[source_id],
                }
            )
    return rows


def catalogue_groups(path):
    rows = json.loads(path.read_text())
    groups = defaultdict(list)
    for row in rows:
        if row.get("kepid"):
            groups[int(row["kepid"])].append(row)
    return groups


def listed_values(specifications, cast=str):
    values, evidence = set(), []
    for specification in specifications:
        path = ROOT / specification["path"]
        rows = read_csv(path)
        found = {cast(row[specification["column"]]) for row in rows}
        values.update(found)
        evidence.append(
            {
                "path": specification["path"],
                "sha256": digest(path),
                "rows": len(rows),
                "unique_values": len(found),
            }
        )
    return values, evidence


def select_catalogue_pools(groups, excluded, cohorts, seed):
    rows = []
    used = set(excluded)
    for category, specification in cohorts.items():
        disposition = specification["disposition"]
        candidates = [
            kic
            for kic, signals in groups.items()
            if kic not in used
            and {signal.get("koi_disposition") for signal in signals} == {disposition}
        ]
        candidates = ordered(candidates, seed, category)
        if len(candidates) < specification["pool"]:
            raise ValueError(f"Too few untouched {disposition} KIC controls")
        pool = candidates[: specification["pool"]]
        used.update(pool)
        for rank, kic in enumerate(pool, 1):
            rows.append(
                {
                    "category": category,
                    "pool_rank": rank,
                    "target_count": specification["target"],
                    "pool_count": specification["pool"],
                    "kic_id": kic,
                    "signal_count": len(groups[kic]),
                }
            )
    return rows


def freeze(out, plan_path=PLAN):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != 1:
        raise ValueError("Unsupported control plan schema")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    index = ROOT / plan["source_index"]
    available = source_rows(index)
    excluded, exclusion_evidence = excluded_sources(plan["excluded_source_manifests"])
    sources = select_source_pools(
        available,
        excluded,
        plan["source_cohorts"],
        plan["selection_seed"],
    )
    source_path = out / "source-pools.csv"
    write_csv(
        source_path,
        sources,
        (
            "category",
            "pool_rank",
            "target_count",
            "pool_count",
            "source_id",
            "volume",
            "shard",
        ),
    )

    catalogue_path = ROOT / plan["koi_catalogue"]
    groups = catalogue_groups(catalogue_path)
    opened_confirmed = ROOT / plan["opened_confirmed_manifest"]
    opened_false_positive = ROOT / plan["opened_false_positive_manifest"]
    excluded_catalogue, catalogue_exclusion_evidence = listed_values(
        [
            {"path": plan["opened_confirmed_manifest"], "column": "kic_id"},
            {"path": plan["opened_false_positive_manifest"], "column": "kic_id"},
            *plan.get("excluded_catalogue_manifests", []),
        ],
        int,
    )
    excluded_catalogue.update(map(int, plan.get("excluded_catalogue_kics", [])))
    catalogue = select_catalogue_pools(
        groups,
        excluded_catalogue,
        plan["catalogue_cohorts"],
        plan["selection_seed"],
    )
    catalogue_output = out / "catalogue-pools.csv"
    write_csv(
        catalogue_output,
        catalogue,
        (
            "category",
            "pool_rank",
            "target_count",
            "pool_count",
            "kic_id",
            "signal_count",
        ),
    )

    report = {
        "schema": 1,
        "status": "FROZEN_SOURCE_POOLS",
        "protocol": plan["protocol"],
        "selection_seed": plan["selection_seed"],
        "plan": {"path": str(plan_path), "sha256": digest(plan_path)},
        "source_index": {
            "path": plan["source_index"],
            "sha256": digest(index),
            "available_sources": len(available),
        },
        "exclusions": exclusion_evidence,
        "excluded_unique_sources": len(excluded),
        "source_pool_counts": dict(Counter(row["category"] for row in sources)),
        "catalogue": {
            "path": plan["koi_catalogue"],
            "sha256": digest(catalogue_path),
            "opened_confirmed_sha256": digest(opened_confirmed),
            "opened_false_positive_sha256": digest(opened_false_positive),
            "exclusions": catalogue_exclusion_evidence,
        },
        "catalogue_pool_counts": dict(Counter(row["category"] for row in catalogue)),
        "artifacts": {
            source_path.name: digest(source_path),
            catalogue_output.name: digest(catalogue_output),
        },
        "claim_boundary": {
            "source_pools": "Untouched ordered pools; materialization must accept the first eligible rows until each target count is reached.",
            "injections": "Flux-level periodic, sparse, and TTV controls do not establish pixel-localization completeness.",
            "catalogue_candidates": "Catalogue candidates are real-signal regressions, not confirmed planets or reliability truth.",
            "confirmed_planets": "The complete local confirmed-planet cohort was opened during development and is not an independent release audit.",
        },
    }
    report_path = out / "freeze.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out, args.plan), indent=2))


if __name__ == "__main__":
    main()
