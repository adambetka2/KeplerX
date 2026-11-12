"""Post-search audit of the physical-template stage against scorer clocks (truth used only here).

For every planet case of a scored round, find the decisions near the true period in the box
bands and in the physical-template band, and record the statistics each stage computed.
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def near(period, true_period, tolerance=0.002):
    return abs(period / true_period - 1) < tolerance


def index_outputs(design_dir):
    out = {}
    for path in glob.glob(str(design_dir / "*.json")):
        with open(path) as handle:
            record = json.load(handle)
        if "input" in record:
            out[record["input"]["sha256"]] = path
    return out


def physical_bands(bands):
    return [band for band in bands if str(band.get("band", "")).endswith("_physical")]


def alias_of(seed, accepted, tolerance=0.004):
    """True when the seed's transit events are a subset or superset of an accepted seed's events."""
    for other in accepted:
        ratio = seed["period_days"] / other["period_days"]
        window = max(seed["duration_days"], other["duration_days"])
        k = round(ratio)
        if k >= 2 and abs(ratio - k) < tolerance:
            offset = (seed["t0_bkjd"] - other["t0_bkjd"]) % other["period_days"]
            if min(offset, other["period_days"] - offset) < window:
                return True
        inverse = round(1 / ratio)
        if inverse >= 2 and abs(1 / ratio - inverse) < tolerance:
            offset = (other["t0_bkjd"] - seed["t0_bkjd"]) % seed["period_days"]
            if min(offset, seed["period_days"] - offset) < window:
                return True
    return False


def dedup_rank(seeds, true_period):
    """Rank of the first near-truth seed after removing alias-family seeds of stronger seeds."""
    accepted = []
    for seed in sorted(seeds, key=lambda d: d["seed_rank"]):
        if alias_of(seed, accepted):
            continue
        accepted.append(seed)
        if near(seed["period_days"], true_period):
            return len(accepted)
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--clocks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    clocks = {}
    for row in csv.DictReader(open(args.clocks)):
        clocks.setdefault(row["case_id"], []).append(row)
    scored = json.load(open(args.round / f"{args.design}-scored-cases.json"))
    index = index_outputs(args.round / args.design)
    rows = []
    band_names = set()
    for case in scored:
        if case["kind"] not in ("real_planet", "synthetic_planet") or case["case_id"] not in clocks:
            continue
        clock = clocks[case["case_id"]][0]
        true_period = float(clock["period_days"])
        with open(index[case["input_sha256"]]) as handle:
            output = json.load(handle)
        bands = output["diagnostics"].get("bands", [])
        phys = physical_bands(bands)
        band_names.update(b.get("band", "") for b in bands)
        row = {"case_id": case["case_id"], "kind": case["kind"], "population": case["population"],
               "depth_ppm": case["depth_ppm"], "true_period_days": true_period,
               "true_duration_hours": clock["duration_hours"], "recovered": case["recovered"],
               "proposal_count": case["proposal_count"]}
        box = [d for b in bands if b not in phys for d in b.get("decisions", []) if near(d.get("period_days", -1), true_period)]
        row["box_near_truth"] = len(box)
        row["box_max_snr"] = max((d.get("search_snr", np.nan) for d in box), default=np.nan)
        row["box_odd_even_max"] = max((abs(d.get("odd_even_sigma", 0)) for d in box), default=np.nan)
        row["box_statuses"] = ";".join(sorted({d.get("status", "") for d in box}))
        phys_decisions = [d for b in phys for d in b.get("decisions", []) if near(d.get("period_days", -1), true_period)]
        row["physical_near_truth"] = len(phys_decisions)
        seeds = [d for d in phys_decisions if "seed_rank" in d]
        row["seed_rank_min"] = min((d["seed_rank"] for d in seeds), default=-1)
        all_seeds = [d for b in phys for d in b.get("decisions", []) if "seed_rank" in d and "duration_days" in d]
        row["seed_rank_after_alias_dedup"] = dedup_rank(all_seeds, true_period) if all_seeds else -1
        row["seed_pool"] = sum(int(b.get("seed_pool", 0) or 0) for b in phys)
        row["alias_seeds_in_top64"] = sum(1 for b in phys for d in sorted([x for x in b.get("decisions", []) if "seed_rank" in x and "duration_days" in x], key=lambda x: x["seed_rank"])[:64] if alias_of(d, [y for y in b.get("decisions", []) if "seed_rank" in y and "duration_days" in y and y["seed_rank"] < d["seed_rank"]]))
        fitted = [d for d in phys_decisions if "physical_delta_log_likelihood" in d]
        row["physical_fitted"] = len(fitted)
        best = max(fitted, key=lambda d: d.get("search_snr", 0), default=None)
        row["physical_snr"] = best["search_snr"] if best else np.nan
        row["physical_odd_even"] = best.get("odd_even_sigma", np.nan) if best else np.nan
        row["physical_harmonic"] = best.get("harmonic_delta_log_likelihood", np.nan) if best else np.nan
        row["physical_inverse_ratio"] = best.get("dip_inverse_snr_ratio", np.nan) if best else np.nan
        row["physical_status"] = best.get("status", "") if best else ""
        row["physical_reasons"] = ";".join(sorted({r for d in phys_decisions for r in d.get("reasons", [])}))
        row["physical_any_proposal_near_truth"] = any(d.get("status") == "PROPOSAL" for d in phys_decisions)
        row["final_near_truth"] = sum(near(p["period_days"], true_period) for p in output["proposals"])
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("band names:", sorted(band_names))
    print("planets audited", len(rows))
    reached = [r for r in rows if r["physical_near_truth"]]
    print("reached physical stage near truth", len(reached), "| fitted", sum(1 for r in rows if r["physical_fitted"]))
    for kind in ("synthetic_planet", "real_planet"):
        sel = [r for r in rows if r["kind"] == kind and r["physical_fitted"]]
        oe = np.array([abs(r["physical_odd_even"]) for r in sel])
        boxoe = np.array([r["box_odd_even_max"] for r in sel])
        print(f"{kind}: fitted near truth {len(sel)}; physical |odd_even| median {np.nanmedian(oe):.2f} >3.5: {int((oe>3.5).sum())}; box |odd_even| max median {np.nanmedian(boxoe):.2f} >3.5: {int((boxoe>3.5).sum())}")
        statuses = {}
        for r in sel:
            statuses[r["physical_status"]] = statuses.get(r["physical_status"], 0) + 1
        print("   best-near-truth physical status:", statuses)
        reasons = {}
        for r in sel:
            for reason in r["physical_reasons"].split(";"):
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
        print("   reasons among near-truth physical decisions:", reasons)
        quota = [r for r in rows if r["kind"] == kind and r["physical_near_truth"] and not r["physical_fitted"]]
        print("   near truth but never fitted (seed quota/dedup):", len(quota), [ (r['case_id'][-20:], r['seed_rank_min'], r['seed_rank_after_alias_dedup']) for r in quota][:12])
        print("   of those, within 64 after alias-family dedup:", sum(1 for r in quota if 0 < r["seed_rank_after_alias_dedup"] <= 64))
        print("   alias seeds among top-64 physical seeds (sum over planets of this kind):", sum(r["alias_seeds_in_top64"] for r in rows if r["kind"] == kind), "of", 64 * sum(1 for r in rows if r["kind"] == kind and r["seed_pool"]))
        lost_after_fit = [r for r in sel if not r["recovered"]]
        print("   fitted near truth but NOT recovered:", len(lost_after_fit))
        for r in lost_after_fit[:20]:
            print("     ", r["case_id"][-24:], "snr", round(r["physical_snr"],1), "oe", round(r["physical_odd_even"],2), "harm", round(r["physical_harmonic"],1), "inv", round(r["physical_inverse_ratio"],2), r["physical_status"], r["physical_reasons"], "final_near", r["final_near_truth"])


if __name__ == "__main__":
    main()
