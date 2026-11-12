"""Stage shallow pixel-control inputs without opening detector outcomes."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from deterministic import shallow_pixel_support as support_code

PLAN = Path(__file__).with_name("shallow-pixel-stage-plan.json")
BASE_PLAN = Path(__file__).with_name("shallow-pixel-support-plan.json")
PENDING = {"PENDING_TPF_INPUT", "PENDING_PIXEL_SUPPORT"}
SUPPORTED = "PIXEL_SUPPORTED"
UNSUPPORTED = "PIXEL_UNSUPPORTED"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def base_identity(identity):
    return {key: value for key, value in identity.items() if key != "tpf_map_sha256"}


def staged_cells(plan):
    cells = [cell for cell in support_code.CELLS if cell != plan["coverage_gap_cell"]]
    if (
        len(cells) != plan["staged_cells"]
        or len(cells) * plan["trials_per_cell"] != plan["staged_target"]
    ):
        raise ValueError("Staged-cell coverage plan changed")
    return cells


def checked_coverage_gap(args, plan):
    path = args.extra_options.parent / "rare-full-audit.json"
    audit = read(path)
    if (
        digest(path) != plan["coverage_gap_audit_sha256"]
        or audit["status"] != "COMPLETE_INPUT_ONLY_RARE_FULL_AUDIT"
        or audit["candidate_cell_counts"][plan["coverage_gap_cell"]] != 0
    ):
        raise ValueError("Audited shallow input coverage gap changed")


def checked_support(args, plan, prior):
    support_path = args.support / "support-report.json"
    support = read(support_path)
    identity = support["identity"]
    expected = {
        "plan_sha256": digest(BASE_PLAN),
        "resolver_sha256": digest(support_code.__file__),
        "exclusions_sha256": digest(args.exclusions),
        "extra_options_sha256": digest(args.extra_options),
        "selection_sha256": digest(args.selection),
        "gaia_sha256": digest(args.gaia),
        "tpf_map_sha256": digest(args.tpf_map) if args.tpf_map else None,
    }
    if (
        support["status"] != "COMPLETE_INPUT_ONLY_PIXEL_SUPPORT_ASSESSMENT"
        or any(identity.get(key) != sha for key, sha in expected.items())
        or digest(BASE_PLAN) != plan["support_plan_sha256"]
        or digest(support_code.__file__) != plan["support_resolver_sha256"]
        or digest(args.extra_options) != plan["combined_options_sha256"]
        or digest(args.gaia) != plan["gaia_sha256"]
        or digest(args.exclusions) != plan["exclusions_sha256"]
        or digest(args.selection) != plan["selection_sha256"]
        or identity["limit"] != 0
        or identity["candidate_pool_per_cell"]
        != read(BASE_PLAN)["candidate_pool_per_cell"]
    ):
        raise ValueError("Staged pixel support provenance changed")
    if prior is None and digest(support_path) != plan["base_support_report_sha256"]:
        raise ValueError("Initial staged support report is not the frozen Gaia input")
    if prior is not None and base_identity(identity) != prior["base_identity"]:
        raise ValueError("Staged support source population changed")
    if args.tpf_map:
        mapping = read(args.tpf_map)
        support_code.validate_tpf_map(mapping)
    return support, digest(support_path)


def checked_truths(args, support):
    exclusions = read(args.exclusions)
    exclusions["path"] = str(args.exclusions)
    selected_shards = read(args.selection)["selected_shards"]
    inputs = SimpleNamespace(
        screens=args.screens,
        enrichment=args.enrichment,
        extra_options=args.extra_options,
        exclusions=args.exclusions,
    )
    truths, screens, enrichment, extra = support_code.merged_screen_candidates(
        inputs, exclusions, selected_shards
    )
    pool, counts = support_code.candidate_pool(truths, read(BASE_PLAN), 0)
    identity = support["identity"]
    candidate_hash = hashlib.sha256(
        "\n".join(
            support_code.candidate_id(row["source_id"], row["cell"]) for row in pool
        ).encode()
    ).hexdigest()
    if (
        identity["screen_summaries_sha256"] != screens
        or identity["enrichment_summaries_sha256"] != enrichment
        or identity["candidate_ids_sha256"] != candidate_hash
        or counts != support["candidate_pool_cell_counts"]
        or len(pool) != support["sources"]
        or len({row["source_id"] for row in pool}) != support["distinct_sources"]
        or extra is None
    ):
        raise ValueError("Staged shallow input option pool changed")
    return {
        support_code.candidate_id(row["source_id"], row["cell"]): row for row in pool
    }


def terminal_unsupported(record, tpf_sources):
    reasons = record.get("quarter_reasons", {})
    complete = set(
        map(
            str,
            tpf_sources.get(record["source_id"], {}).get("complete_quarters", []),
        )
    )
    return (
        bool(reasons)
        and set(reasons) <= complete
        and not set(reasons.values())
        - {"NO_AVAILABLE_TPF", "TARGET_CENTER_OFF_STAMP", "RESOLVED_PARENT_EXCLUDED"}
    )


def checked_options(args, support, truths, excluded_parents):
    if set(truths) != set(support["records_sha256"]):
        raise ValueError("Support record set differs from screened options")
    options = {}
    statuses = Counter()
    tpf_sources = read(args.tpf_map)["sources"] if args.tpf_map else {}
    for key, expected in support["records_sha256"].items():
        path = args.support / "records" / f"{key}.json"
        if digest(path) != expected:
            raise ValueError(f"Pixel-support record changed: {path}")
        record, truth = read(path), truths[key]
        if (
            record["identity"] != support["identity"]
            or record["candidate_id"] != key
            or record["source_id"] != truth["source_id"]
            or record["cell"] != truth["cell"]
            or record["truth_sha256"] != support_code.truth_digest(truth)
            or record["base_raw_fits_sha256"] != truth["base_raw_fits_sha256"]
            or record["observed_quarters"] != truth["observed_quarters"]
        ):
            raise ValueError(f"Pixel-support truth changed: {path}")
        status = record["status"]
        if status not in PENDING | {SUPPORTED, UNSUPPORTED, "PENDING_ASTROMETRY"}:
            raise ValueError(f"Unexpected pixel-support status: {path}")
        if status == UNSUPPORTED and (
            support["identity"]["tpf_map_sha256"] is None
            or not terminal_unsupported(record, tpf_sources)
        ):
            raise ValueError(f"Unsupported choice lacks terminal TPF evidence: {path}")
        parents = (
            record["resolved_parents"]
            if status == SUPPORTED
            else [truth["source_parent_tpf"]]
        )
        if set(parents) & excluded_parents:
            raise ValueError(f"Choice overlaps a prior-control parent: {path}")
        options[key] = {
            "candidate_id": key,
            "source_id": record["source_id"],
            "cell": record["cell"],
            "base_raw_fits_sha256": record["base_raw_fits_sha256"],
            "truth_sha256": record["truth_sha256"],
            "origin_parent": truth["source_parent_tpf"],
            "resolved_parents": parents,
            "observed_quarters": record["observed_quarters"],
            "status": status,
            "support_record_sha256": expected,
            "pixel_evidence_sha256": support_code.truth_digest(record["pixels"])
            if status == SUPPORTED
            else None,
        }
        statuses[status] += 1
    if statuses != Counter(support["statuses"]):
        raise ValueError("Pixel-support status totals changed")
    supported = support_code.verified_supported_records(
        support, args.support, args.prf_cache
    )
    if {row["candidate_id"] for row in supported} != {
        key for key, row in options.items() if row["status"] == SUPPORTED
    }:
        raise ValueError("Resolved pixel-support verification changed")
    return options


def check_locked(rows, locked):
    source_ids = [row["source_id"] for row in rows if row["candidate_id"] in locked]
    raw_hashes = [
        row["base_raw_fits_sha256"] for row in rows if row["candidate_id"] in locked
    ]
    parents = [
        parent
        for row in rows
        if row["candidate_id"] in locked
        for parent in row["resolved_parents"]
    ]
    if (
        len(source_ids) != len(set(source_ids))
        or len(raw_hashes) != len(set(raw_hashes))
        or len(parents) != len(set(parents))
    ):
        raise ValueError(
            "Previously supported choices conflict on source, raw input, or resolved parent"
        )


def staged_choice(options, cells, per_cell, target, seed, locked=()):
    cell_set = set(cells)
    rows = sorted(
        (row for row in options if row["cell"] in cell_set),
        key=lambda row: (support_code.rank(row["source_id"], seed), row["cell"]),
    )
    ids = [row["candidate_id"] for row in rows]
    locked = set(locked)
    if len(ids) != len(set(ids)) or not locked <= set(ids):
        raise ValueError("Staged choice IDs changed")
    check_locked(rows, locked)
    if not rows:
        return []
    lower = np.array([int(row["candidate_id"] in locked) for row in rows])
    bounds = Bounds(lower, np.ones(len(rows)))
    linear = support_code.selection_constraints(rows, cells, per_cell)
    maximum = milp(
        -np.ones(len(rows)),
        integrality=np.ones(len(rows)),
        bounds=bounds,
        constraints=linear,
        options={"mip_rel_gap": 0},
    )
    if maximum.status != 0:
        raise RuntimeError(
            f"Staged maximum-cardinality selection failed: {maximum.message}"
        )
    count = round(-maximum.fun)
    if count > target:
        raise ValueError("Staged selection exceeded declared capacity")
    costs = np.arange(1, len(rows) + 1, dtype=float)
    exact = LinearConstraint(np.ones((1, len(rows))), count, count)
    ranked = milp(
        costs,
        integrality=np.ones(len(rows)),
        bounds=bounds,
        constraints=(linear, exact),
        options={"mip_rel_gap": 0},
    )
    if ranked.status != 0:
        raise RuntimeError(f"Staged deterministic selection failed: {ranked.message}")
    chosen = [row for row, value in zip(rows, ranked.x, strict=True) if value > 0.5]
    if len(chosen) != count or not locked <= {row["candidate_id"] for row in chosen}:
        raise ValueError("Staged selection lost a supported choice")
    return chosen


def checked_prior(path, plan, identity, options):
    if path is None:
        return None
    prior = read(path)
    if (
        prior["plan_sha256"] != digest(PLAN)
        or prior["selector_sha256"] != digest(__file__)
        or prior["base_identity"] != base_identity(identity)
        or prior["status"] not in {"ASSESS_SELECTED_PIXEL_INPUTS", "WAIT_INPUT"}
        or len(prior["selected"]) != plan["staged_target"]
    ):
        raise ValueError("Prior staged selection identity changed")
    if not set(prior["retired_ids"]).isdisjoint(
        {row["candidate_id"] for row in prior["selected"]}
    ):
        raise ValueError("Retired option remains selected")
    for previous in prior["selected"]:
        current = options[previous["candidate_id"]]
        if (
            current["truth_sha256"] != previous["truth_sha256"]
            or current["source_id"] != previous["source_id"]
            or current["cell"] != previous["cell"]
            or previous["status"] == SUPPORTED
            and (
                current["status"] != SUPPORTED
                or current["resolved_parents"] != previous["resolved_parents"]
                or current["pixel_evidence_sha256"] != previous["pixel_evidence_sha256"]
            )
            or previous["status"] == UNSUPPORTED
            and current["status"] != UNSUPPORTED
        ):
            raise ValueError("Previously staged choice changed or regressed")
    return prior


def advance_selection(options, prior, plan, cells):
    eligible_statuses = PENDING | {SUPPORTED}
    if prior:
        prior_ids = {row["candidate_id"] for row in prior["selected"]}
        pending = prior_ids & {
            key for key, row in options.items() if row["status"] in PENDING
        }
        if pending:
            return (
                [options[row["candidate_id"]] for row in prior["selected"]],
                prior["retired_ids"],
                "WAIT_INPUT",
            )
        unsupported = prior_ids & {
            key for key, row in options.items() if row["status"] == UNSUPPORTED
        }
        retired = sorted(set(prior["retired_ids"]) | unsupported)
        locked = prior_ids - unsupported
        eligible = [
            row
            for key, row in options.items()
            if key not in retired and row["status"] in eligible_statuses
        ]
        chosen = staged_choice(
            eligible,
            cells,
            plan["trials_per_cell"],
            plan["staged_target"],
            plan["seed"],
            locked,
        )
    else:
        retired = []
        eligible = [
            row for row in options.values() if row["status"] in eligible_statuses
        ]
        chosen = staged_choice(
            eligible,
            cells,
            plan["trials_per_cell"],
            plan["staged_target"],
            plan["seed"],
        )
    return chosen, retired, "ASSESS_SELECTED_PIXEL_INPUTS"


def run(args):
    plan = read(PLAN)
    if plan["status"] != "INPUT_ONLY_DEVELOPMENT":
        raise ValueError("Staging plan is not an input-only development plan")
    cells = staged_cells(plan)
    checked_coverage_gap(args, plan)
    prior_raw = read(args.prior) if args.prior else None
    support, support_sha = checked_support(args, plan, prior_raw)
    truths = checked_truths(args, support)
    options = checked_options(
        args, support, truths, set(read(args.exclusions)["parents"])
    )
    prior = checked_prior(args.prior, plan, support["identity"], options)
    if support["candidate_pool_cell_counts"][plan["coverage_gap_cell"]] != 0:
        raise ValueError("Predeclared shallow coverage gap changed")
    chosen, retired, status = advance_selection(options, prior, plan, cells)
    counts = Counter(row["cell"] for row in chosen)
    full = len(chosen) == plan["staged_target"] and all(
        counts[cell] == plan["trials_per_cell"] for cell in cells
    )
    if not full:
        status = "NO_FULL_STRUCTURAL_REPLACEMENT"
    elif any(row["status"] in PENDING for row in chosen):
        status = "WAIT_INPUT" if prior else "ASSESS_SELECTED_PIXEL_INPUTS"
    elif all(row["status"] == SUPPORTED for row in chosen):
        status = "SUPPORTED_35_CELL_STAGING_SET"
    selected_ids = {row["candidate_id"] for row in chosen}
    previous_ids = (
        {row["candidate_id"] for row in prior["selected"]} if prior else set()
    )
    report = {
        "schema": 1,
        "status": status,
        "stage": prior["stage"] + 1 if prior else 0,
        "plan_sha256": digest(PLAN),
        "selector_sha256": digest(__file__),
        "base_identity": base_identity(support["identity"]),
        "support_report_sha256": support_sha,
        "prior_sha256": digest(args.prior) if prior else None,
        "coverage_gap_cell": plan["coverage_gap_cell"],
        "selected": chosen,
        "retired_ids": retired,
        "added_ids": sorted(selected_ids - previous_ids),
        "removed_ids": sorted(previous_ids - selected_ids),
        "selected_cell_counts": {cell: counts[cell] for cell in cells},
        "selected_observed_quarter_files": sum(
            len(row["observed_quarters"]) for row in chosen
        ),
        "pending_selected": sum(row["status"] in PENDING for row in chosen),
        "supported_selected": sum(row["status"] == SUPPORTED for row in chosen),
        "unsupported_selected": sum(row["status"] == UNSUPPORTED for row in chosen),
        "claim_boundary": plan["claim_boundary"],
    }
    if prior and status != "WAIT_INPUT":
        if set(report["removed_ids"]) != set(retired) - set(prior["retired_ids"]):
            raise ValueError(
                "A selected option was replaced without unsupported evidence"
            )
    save(args.out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "screens",
        "selection",
        "enrichment",
        "extra_options",
        "exclusions",
        "gaia",
        "support",
        "prf_cache",
        "out",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--tpf-map", type=Path)
    parser.add_argument("--prior", type=Path)
    report = run(parser.parse_args())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "stage",
                    "coverage_gap_cell",
                    "selected_observed_quarter_files",
                    "pending_selected",
                    "supported_selected",
                    "unsupported_selected",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
