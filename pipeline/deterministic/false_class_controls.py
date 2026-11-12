"""Freeze outcome-blind Kepler Bonus parents for three false-class injections."""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("false-class-control-plan.json")
INDEX = ROOT / "discSSD/data/index.sqlite"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rank(seed, category, source_id):
    return hashlib.sha256(f"{seed}:{category}:{source_id}".encode()).digest()


def indexed_sources(index=INDEX):
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
        files = database.execute(
            "select volume,path,sha256 from files where collection='kbonus' "
            "and path like 'kbonus/lc2/shard-%/source.parquet' order by volume,path"
        ).fetchall()
        indexed_ids = {
            source_id
            for (source_id,) in database.execute("select gaia_source_id from sources")
        }
    sources = {}
    provenance = []
    fields = [
        "source_id",
        "origin_tpf_kic",
        "ra_deg",
        "dec_deg",
        "phot_g_mean_mag",
        "quarter_count",
        "finite_psf_flat_kbonus",
        "raw_fits_sha256",
    ]
    for volume, relative, sha256 in files:
        path = ROOT / volume / "data" / relative
        for row in pq.read_table(path, columns=fields).to_pylist():
            source_id = str(row["source_id"])
            if source_id in sources:
                raise ValueError(f"Repeated LC2 source: {source_id}")
            sources[source_id] = dict(row, source_id=source_id, volume=volume)
        provenance.append({"volume": volume, "path": relative, "sha256": sha256})
    if set(sources) != indexed_ids:
        raise ValueError("LC2 source metadata and read-only index disagree")
    return sources, provenance


def excluded_parents(plan, sources):
    parents = set()
    inputs = {}
    for item in plan["exclude_csv"]:
        path = ROOT / item["path"]
        inputs[item["path"]] = digest(path)
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if item["column"] not in reader.fieldnames:
                raise ValueError(f"Exclusion column missing: {item['path']}")
            values = [row[item["column"]] for row in reader]
        if item["kind"] == "parent":
            parents.update(value for value in values if value)
        elif item["kind"] == "source":
            parents.update(
                str(sources[value]["origin_tpf_kic"])
                for value in values
                if value in sources
            )
        else:
            raise ValueError(f"Unknown exclusion kind: {item['kind']}")
    for item in plan["exclude_jsonl"]:
        path = ROOT / item["path"]
        inputs[item["path"]] = digest(path)
        with path.open() as stream:
            parents.update(str(json.loads(line)[item["field"]]) for line in stream)
    for name in plan["exclude_parent_audits"]:
        path = ROOT / name
        audit = read(path)
        if audit["status"] != "PARENT_INDEPENDENT":
            raise ValueError(f"Parent audit is incomplete: {name}")
        parents.update(
            str(parent)
            for resolved in audit["source_parents"].values()
            for parent in resolved
        )
        inputs[name] = digest(path)
    return parents, inputs


def eligible(source, plan):
    return (
        source["origin_tpf_kic"] is not None
        and int(source["origin_tpf_kic"]) > 0
        and source["quarter_count"] >= plan["minimum_quarters"]
        and source["finite_psf_flat_kbonus"] >= plan["minimum_finite_primary_cadences"]
        and source["raw_fits_sha256"]
        and source["ra_deg"] is not None
        and source["dec_deg"] is not None
    )


def separation_arcsec(a, b):
    ra1, ra2 = map(math.radians, (a["ra_deg"], b["ra_deg"]))
    dec1, dec2 = map(math.radians, (a["dec_deg"], b["dec_deg"]))
    angle = math.hypot(
        math.cos(dec2) * math.sin(ra2 - ra1),
        math.cos(dec1) * math.sin(dec2)
        - math.sin(dec1) * math.cos(dec2) * math.cos(ra2 - ra1),
    )
    cosine = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(
        dec2
    ) * math.cos(ra2 - ra1)
    return math.degrees(math.atan2(angle, cosine)) * 3600


def background_neighbour(source, siblings, plan):
    if source["phot_g_mean_mag"] is None:
        return None
    low, high = plan["background_separation_arcsec"]
    options = []
    for other in siblings:
        if other["source_id"] == source["source_id"] or not eligible(other, plan):
            continue
        if other["phot_g_mean_mag"] is None:
            continue
        distance = separation_arcsec(source, other)
        if (
            low <= distance <= high
            and abs(source["phot_g_mean_mag"] - other["phot_g_mean_mag"])
            <= plan["background_max_magnitude_difference"]
        ):
            options.append((distance, other["source_id"], other))
    return min(options) if options else None


def select_pool(plan, sources, exclusions):
    by_parent = defaultdict(list)
    for source in sources.values():
        if eligible(source, plan):
            by_parent[str(source["origin_tpf_kic"])].append(source)
    chosen = []
    occupied = set(exclusions)
    count = plan["candidate_pool_per_class"]
    for category in plan["classes"]:
        ordered = sorted(
            sources, key=lambda source_id: rank(plan["seed"], category, source_id)
        )
        selected = 0
        for source_id in ordered:
            source = sources[source_id]
            if not eligible(source, plan):
                continue
            parent = str(source["origin_tpf_kic"])
            if parent in occupied:
                continue
            neighbour = (
                background_neighbour(source, by_parent[parent], plan)
                if category == "background_contaminant"
                else None
            )
            if category == "background_contaminant" and neighbour is None:
                continue
            occupied.add(parent)
            selected += 1
            row = {
                "category": category,
                "pool_rank": selected,
                "source_id": source_id,
                "parent_tpf": parent,
                "volume": source["volume"],
                "raw_fits_sha256": source["raw_fits_sha256"],
                "contaminant_source_id": neighbour[1] if neighbour else "",
                "contaminant_raw_fits_sha256": neighbour[2]["raw_fits_sha256"]
                if neighbour
                else "",
                "contaminant_separation_arcsec": f"{neighbour[0]:.6f}"
                if neighbour
                else "",
            }
            chosen.append(row)
            if selected == count:
                break
        if selected != count:
            raise ValueError(
                f"Insufficient parent-disjoint {category} candidates: {selected}/{count}"
            )
    return chosen


def freeze(out, plan_path=PLAN):
    out = Path(out)
    plan_path = Path(plan_path)
    plan = read(plan_path)
    if (
        plan["schema"] != 1
        or len(plan["classes"]) != 3
        or len(set(plan["classes"])) != 3
        or plan["candidate_pool_per_class"]
        < plan["development_per_class"] + plan["evaluation_per_class"]
    ):
        raise ValueError("Invalid false-class control plan")
    sources, metadata_files = indexed_sources()
    exclusions, exclusion_inputs = excluded_parents(plan, sources)
    rows = select_pool(plan, sources, exclusions)
    out.mkdir(parents=True, exist_ok=False)
    pool_path = out / "candidate-pool.csv"
    with pool_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": 1,
        "status": "CANDIDATE_POOL_FROZEN",
        "protocol": plan["protocol"],
        "classes": {
            category: plan["candidate_pool_per_class"] for category in plan["classes"]
        },
        "distinct_parents": len({row["parent_tpf"] for row in rows}),
        "excluded_parents": len(exclusions),
        "source_count": len(sources),
        "inputs": {
            "generator": {
                "path": str(Path(__file__).relative_to(ROOT)),
                "sha256": digest(__file__),
            },
            "plan": {
                "path": str(plan_path.relative_to(ROOT)),
                "sha256": digest(plan_path),
            },
            "index": {"path": str(INDEX.relative_to(ROOT)), "sha256": digest(INDEX)},
            "exclusions": exclusion_inputs,
            "source_metadata": metadata_files,
        },
        "artifacts": {"candidate-pool.csv": digest(pool_path)},
        "claim_boundary": plan["claim_boundary"],
    }
    (out / "freeze.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out, args.plan), indent=2))


if __name__ == "__main__":
    main()
