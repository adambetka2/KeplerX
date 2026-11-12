"""Replace input failures with usable Flat cases without consulting search outcomes."""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import flat_retrieval as H
import numpy as np


def group(item):
    case = item["case"]
    depth = (
        item["origin"].get("depth_bin", item["truth"].get("depth_bin", "UNKNOWN"))
        if case["kind"] == "synthetic_planet"
        else "NOT_APPLICABLE"
    )
    return case["kind"], case["population"], depth


def load_population(cohort):
    clocks = defaultdict(list)
    for clock in H.API.rows(cohort / "clocks.csv"):
        clocks[clock["case_id"]].append(clock)
    items = []
    for case in H.API.rows(cohort / "cases.csv"):
        path = cohort / case["receipt_path"]
        H.API.check_bindings({str(path): case["receipt_sha256"]})
        item = H.recipe(case, H.API.read(path))
        if clocks[case["case_id"]]:
            for field in ("period_days", "duration_hours"):
                item[field] = H.number(clocks[case["case_id"]][0][field])
        items.append(item)
    return items


def potential(item):
    if item["case"]["kind"] in ("real_planet", "native_eb", "catalogue_fp"):
        return True
    if item["flat_cadences"] is None or item["flat_cadences"] < 100:
        return False
    return item["case"]["kind"] != "year_scrambled" or (
        item["quarter_count"] is not None and item["quarter_count"] >= 17
    )


def restore_case(item, out):
    """Restore case metadata from unchanged prepared records and their receipt."""
    case = item["case"].copy()
    path = out / "prepared" / (H.rank(case["case_id"])[:24] + ".npz")
    receipt = H.API.read(path.with_suffix(".json"))
    H.require(
        receipt["status"] == "PREPARED", "Only a prepared receipt can be restored"
    )
    H.API.check_bindings({str(path): receipt["archive_sha256"]})
    case.update(
        photometry_view="psf_flat",
        photometry_input_class="DERIVED_DELIVERED_PSF_FLAT",
        preparation_status="PREPARED",
        first_failure="",
        input_path=str(path),
        input_sha256=receipt["archive_sha256"],
        array_key="records",
        receipt_path=str(path.with_suffix(".json")),
        receipt_sha256=H.API.digest(path.with_suffix(".json")),
    )
    if item["phot_g_mean_mag"] is not None:
        case["phot_g_mean_mag"] = str(item["phot_g_mean_mag"])
    if case["kind"] == "synthetic_planet":
        with np.load(path, allow_pickle=False) as archive:
            support = H.calibrate.loss(archive["records"]["time"], item["truth"])
            case.update(
                sampled_prepared_depth_ppm=float(support.max()) * 1e6,
                prepared_nonzero_loss_cadences=int(np.count_nonzero(support)),
                prepared_records_identical=str(
                    archive["original"].tobytes() == archive["records"].tobytes()
                ).lower(),
            )
    return case


def restore_progress(out, kept, selected, missing, attempted):
    path = out / "replacement-ledger.json"
    if not path.exists():
        return []
    items = {}
    for batch in out.glob("candidate-batch-*.json"):
        items.update({item["case"]["case_id"]: item for item in H.API.read(batch)})
    ledger = H.API.read(path)
    for entry in ledger:
        item = items[entry["case_id"]]
        attempted.add(entry["case_id"])
        if entry["status"] != "PREPARED":
            continue
        kept.append(restore_case(item, out))
        selected.append(item)
        key = group(item)
        missing[key] -= 1
        if not missing[key]:
            del missing[key]
    return ledger


def pick_candidates(pool, missing):
    groups = defaultdict(list)
    for item in pool:
        groups[group(item)].append(item)
    chosen = []
    for key, count in sorted(missing.items(), key=lambda entry: str(entry[0])):
        H.require(
            len(groups[key]) >= count,
            f"Insufficient usable candidates in {key}: {len(groups[key])} for {count}",
        )
        # Extra input-only probes cover quality/mask or year-scramble failures.
        chosen.extend(
            H.sample(groups[key], min(len(groups[key]), count + max(2, count // 4)))
        )
    return chosen


def acquire(items, out):
    sources = {
        item["case"]["source_id"]: {
            "source_id": item["case"]["source_id"],
            "shard": item["origin"]["shard"],
        }
        for item in items
    }
    H.retrieve(list(sources.values()), out / "sources")


def derive(args):
    H.require(
        not (args.out / "plan.json").exists(),
        "A frozen usable cohort cannot be extended",
    )
    if not args.out.exists():
        args.out.mkdir(parents=True)
        (args.out / "logs").mkdir()
        (args.out / "prepared").mkdir()
        (args.out / "sources").symlink_to(
            args.parent / "sources", target_is_directory=True
        )
        (args.out / "fits").symlink_to(args.parent / "fits", target_is_directory=True)
    plan = H.API.read(args.parent / "plan.json")
    H.API.check_bindings(plan["bindings"])
    population = load_population(Path(plan["cohort"]))
    original = H.API.read(args.parent / "pilot-recipes.json")
    prepared = H.API.read(args.parent / "prepared-cases.json")
    lookup = {item["case"]["case_id"]: item for item in original}
    kept = [row for row in prepared if row["preparation_status"] == "PREPARED"]
    selected = [lookup[row["case_id"]] for row in kept]
    missing = Counter(
        group(lookup[row["case_id"]])
        for row in prepared
        if row["preparation_status"] != "PREPARED"
    )
    attempted = {row["case_id"] for row in prepared}
    ledger = restore_progress(args.out, kept, selected, missing, attempted)
    bindings = H.API.bind_files(
        [
            Path(__file__),
            Path(H.__file__),
            Path(H.calibrate.__file__),
            Path(H.CONTROLS.__file__),
            H.ROOT / "pipeline/deterministic/injections.py",
            H.ROOT / "pipeline/deterministic/run.py",
            H.ROOT / "pipeline/deterministic/cloud_public.py",
            H.ROOT / "pipeline/deterministic/catalogue_controls.py",
            H.ROOT / "pipeline/photometry/lc2.py",
            args.parent / "runner-v2/preparation-source/flat_retrieval.py",
        ]
    )
    archived = args.out / "first-failure/usable_flat.py"
    if archived.exists():
        bindings.update(H.API.bind_files([archived]))
    runtime = H.API.runtime_identity()
    while missing:
        pool = [
            item
            for item in population
            if item["case"]["case_id"] not in attempted and potential(item)
        ]
        candidates = pick_candidates(pool, missing)
        H.API.save(args.out / f"candidate-batch-{len(ledger)}.json", candidates)
        acquire(candidates, args.out)
        for item in candidates:
            key = group(item)
            if not missing[key]:
                continue
            attempted.add(item["case"]["case_id"])
            case = H.prepare_case(item, args.out)
            entry = {
                "case_id": case["case_id"],
                "status": case["preparation_status"],
                "first_failure": case["first_failure"],
                "group": key,
            }
            ledger.append(entry)
            if case["preparation_status"] == "PREPARED":
                kept.append(case)
                selected.append(item)
                missing[key] -= 1
                if not missing[key]:
                    del missing[key]
            H.API.save(args.out / "replacement-ledger.json", ledger)
            print(json.dumps({"usable": len(kept), **entry}), flush=True)
    H.require(
        len(kept) == 1000 and Counter(row["kind"] for row in kept) == Counter(H.COUNTS),
        "Exactly 1000 usable cases with the original kind mix required",
    )
    H.API.check_bindings(bindings)
    H.require(
        runtime == H.API.runtime_identity(),
        "Numerical runtime changed during replacement",
    )
    freeze(args, plan, kept, selected, population, prepared, bindings, runtime, ledger)


def freeze(args, plan, kept, selected, population, previous, bindings, runtime, ledger):
    H.API.save(args.out / "prepared-cases.json", kept)
    H.API.save(args.out / "pilot-recipes.json", selected)
    H.write_csv(args.out / "selected-cases.csv", [item["case"] for item in selected])
    for name in ("designs.json", "real5000.csv"):
        shutil.copyfile(args.parent / name, args.out / name)
    H.API.save(
        args.out / "preparation-identity.json",
        {
            "bindings": bindings,
            "runtime": runtime,
            "view": "psf_flat",
            "mask_flatten": True,
            "inherited_prepared_cases": 604,
            "parent_bindings": H.API.bind_files(
                [
                    args.parent / "prepared-cases.json",
                    args.parent / "preparation-identity.json",
                    args.parent / "runner-v2/preparation-identity.json",
                ]
            ),
            "selection": "Only preparation eligibility; detector results were not read",
        },
    )
    H.API.save(
        args.out / "representativeness.json",
        {
            "scope": "Conditional on usable delivered Flat and valid null construction; no longer the all-draw denominator",
            "against_original10000": H.distributions(population, selected),
            "against_availability_prefilter": H.distributions(
                [item for item in population if potential(item)], selected
            ),
            "limits": "The metadata prefilter is not a complete exact preparation census; unobserved eligibility and missing anchor metadata remain explicit",
        },
    )
    plan.update(
        status="FROZEN_1000_USABLE_PSF_FLAT_CASES",
        cohort_selection="Retain 604 prepared cases; replace 363 Flat failures and 33 invalid year scrambles within kind/M/K/depth-bin quotas",
        positive_denominator=700,
        control_denominator=300,
        parent=str(args.parent),
        replacement_attempts=len(ledger),
        excluded_parent_cases=[
            {"case_id": row["case_id"], "first_failure": row["first_failure"]}
            for row in previous
            if row["preparation_status"] != "PREPARED"
        ],
        limits="Recovery is conditional on 1000 usable Flat inputs; unavailable originals are recorded separately and do not enter this denominator. This is a development pilot, not the 10000-case programme gate. Delivered flattening transfer and astrophysical false-positive rates remain unmeasured.",
    )
    plan["bindings"] = plan["bindings"] | H.API.bind_files(
        [
            args.out / "prepared-cases.json",
            args.out / "pilot-recipes.json",
            args.out / "selected-cases.csv",
            args.out / "representativeness.json",
            args.out / "designs.json",
            args.out / "real5000.csv",
            args.out / "replacement-ledger.json",
        ]
    )
    H.API.save(args.out / "plan.json", plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "prepared": len(kept),
                "replacement_attempts": len(ledger),
                "counts": dict(Counter(row["kind"] for row in kept)),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.parent, args.out = args.parent.resolve(), args.out.resolve()
    H.API.capacity(args.out.parent, 16 * 1024**3)
    derive(args)


if __name__ == "__main__":
    main()
