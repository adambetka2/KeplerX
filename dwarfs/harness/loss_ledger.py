"""Truth-conditioned loss ledger for a scored development round.

Reads saved blind receipts only; never searches. For every planet signal it
records the expected white-noise SNR of the injected clock, the coarse box
statistic at the true period and its aliases, how far the true clock travelled
through the pipeline, and the first stage that lost it.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ALIASES = {
    "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "1/2": 0.5, "1/3": 1 / 3,
    "1/4": 0.25, "3/2": 1.5, "2/3": 2 / 3,
}
EXPOSURE = 0.02043361111111111


def job_key(row):
    return hashlib.sha256((row["input_sha256"] + ":" + row["array_key"]).encode()).hexdigest()[:24]


def alias_of(period, true_period, tolerance=0.004):
    ratio = period / true_period
    for name, value in ALIASES.items():
        if abs(ratio / value - 1) < tolerance:
            return name
    return ""


def matches(period, epoch, clock):
    true_period, true_epoch = float(clock["period_days"]), float(clock["epoch_bkjd"])
    start, stop = float(clock["start_bkjd"]), float(clock["stop_bkjd"])
    tolerance = float(clock["duration_hours"]) / 48 + float(clock["exposure_days"])
    centres = true_epoch + true_period * np.arange(
        np.ceil((start - true_epoch) / true_period),
        np.floor((stop - true_epoch) / true_period) + 1,
    )
    timing = np.abs(centres - epoch - np.rint((centres - epoch) / period) * period)
    return bool(
        len(timing)
        and abs(period / true_period - 1) * (stop - start) <= tolerance
        and abs((epoch - true_epoch + true_period / 2) % true_period - true_period / 2) <= tolerance
        and np.max(timing) <= tolerance
    )


def expected_snr(records, clock, depth_ppm, min_quarter_cadences=100):
    quarters, counts = np.unique(records["quarter"], return_counts=True)
    keep = np.isin(records["quarter"], quarters[counts >= min_quarter_cadences])
    time = records["time"][keep].astype(np.float64)
    flux = records["flux"][keep].astype(np.float64)
    error = records["error"][keep].astype(np.float64)
    quarter = records["quarter"][keep]
    period, epoch = float(clock["period_days"]), float(clock["epoch_bkjd"])
    duration = float(clock["duration_hours"]) / 24
    cycle = np.rint((time - epoch) / period)
    inside = np.abs(time - epoch - cycle * period) <= duration / 2
    sigma_fd = np.empty(len(time))
    sigma_mad = np.empty(len(time))
    for value in np.unique(quarter):
        use = quarter == value
        diff = np.diff(flux[use])
        sigma_fd[use] = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)
        centred = flux[use] - np.median(flux[use])
        sigma_mad[use] = 1.4826 * np.median(np.abs(centred))
    depth = depth_ppm * 1e-6
    information_fd = float(np.sum(1 / sigma_fd[inside] ** 2))
    information_err = float(np.sum(1 / error[inside] ** 2))
    events = np.unique(cycle[inside])
    per_event = np.array([np.sum(inside & (cycle == c)) for c in events])
    return {
        "cadences": len(time),
        "quarters": len(np.unique(quarter)),
        "span_days": float(time[-1] - time[0]) if len(time) else 0.0,
        "in_transit_cadences": int(inside.sum()),
        "events": len(events),
        "full_events": int(np.sum(per_event >= max(1, np.floor(duration / EXPOSURE)))),
        "sigma_fd_ppm": float(np.median(sigma_fd) * 1e6),
        "sigma_mad_ppm": float(np.median(sigma_mad) * 1e6),
        "sigma_error_ppm": float(np.median(error) * 1e6),
        "snr_white_fd": depth * np.sqrt(information_fd),
        "snr_white_error": depth * np.sqrt(information_err),
    }


def coarse_at_period(arrays, prefix, true_period, config):
    """Coarse box statistic at the grid period nearest the true one."""
    periods = arrays[prefix + "period_days"]
    if not (periods[0] <= true_period <= periods[-1]):
        return None
    power = arrays[prefix + "search_snr"]
    depth = arrays[prefix + "depth_ppm"]
    index = int(np.argmin(np.abs(np.log(periods / true_period))))
    lo, hi = max(0, index - 3), min(len(periods), index + 4)
    local = int(lo + np.argmax(power[lo:hi]))
    threshold = config.get("coarse_search_snr_min", config["search_snr_min"])
    is_max = bool(
        (power[local] >= (power[local - 1] if local > 0 else -np.inf))
        and (power[local] > (power[local + 1] if local + 1 < len(power) else -np.inf))
    )
    depth_ok = config["depth_ppm_min"] <= depth[local] <= config["depth_ppm_max"]
    return {
        "grid_index": local,
        "period": float(periods[local]),
        "snr": float(power[local]),
        "depth_ppm": float(depth[local]),
        "rank": int(np.sum(power > power[local]) + 1),
        "above_threshold": bool(power[local] >= threshold),
        "depth_allowed": bool(depth_ok),
        "local_maximum": is_max,
        "eligible": bool(power[local] >= threshold and depth_ok and is_max),
    }


def band_arrays(npz, path, names):
    out = {}
    for name in names:
        key = f"{path}/{name}"
        if key in npz.files:
            out[name] = npz[key]
    return out


def near(period, true_period, tolerance=0.003):
    return abs(period / true_period - 1) < tolerance


def analyse_case(case, clocks, result_path, config):
    with open(result_path) as stream:
        result = json.load(stream)
    npz = np.load(str(Path(result_path).with_suffix(".npz")), allow_pickle=False)
    records = npz["prepared_records"]
    diagnostics = result["diagnostics"]
    clock = clocks[0]
    true_period = float(clock["period_days"])
    depth = float(case["sampled_prepared_depth_ppm"] or case["depth_ppm"])
    row = {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "population": case["population"],
        "gaia_g": case["phot_g_mean_mag"],
        "depth_ppm": depth,
        "period_days": true_period,
        "duration_hours": float(clock["duration_hours"]),
        "recovered": bool(case["recovered"]),
        "proposal_count": len(result["proposals"]),
        "status": result["status"],
    }
    row.update(expected_snr(records, clock, depth, config["min_quarter_cadences"]))

    # Coarse stage: every duration band, true period and its aliases.
    coarse = {}
    bands = diagnostics.get("inactive_native_prefix_bands") or diagnostics["bands"]
    for index, band in enumerate(bands):
        if "duration_trial_days" not in band or band["status"] != "SEARCHED":
            continue
        path = f"/diagnostics/inactive_native_prefix_bands/{index}/periodogram_arrays"
        if "inactive_native_prefix_bands" not in diagnostics:
            path = f"/diagnostics/bands/{index}/periodogram_arrays"
        arrays = band_arrays(npz, path, ["period_days", "search_snr", "depth_ppm"])
        if "period_days" not in arrays:
            continue
        for alias, ratio in ALIASES.items():
            stat = coarse_at_period(arrays, "", true_period * ratio, band["config"])
            if stat is None:
                continue
            stat.update(band=band["band"], duration=band["duration_trial_days"])
            best = coarse.get(alias)
            if best is None or stat["snr"] > best["snr"]:
                coarse[alias] = stat
    for alias in ("1", "2", "1/2", "3", "1/3"):
        stat = coarse.get(alias)
        tag = alias.replace("/", "_")
        row[f"coarse_snr_{tag}"] = stat["snr"] if stat else np.nan
        row[f"coarse_rank_{tag}"] = stat["rank"] if stat else -1
    true_coarse = coarse.get("1")
    row["coarse_band"] = true_coarse["band"] if true_coarse else ""
    row["coarse_duration"] = true_coarse["duration"] if true_coarse else np.nan
    row["coarse_depth_ppm"] = true_coarse["depth_ppm"] if true_coarse else np.nan
    row["coarse_eligible"] = bool(true_coarse and true_coarse["eligible"])
    row["coarse_above_threshold"] = bool(true_coarse and true_coarse["above_threshold"])
    row["coarse_local_maximum"] = bool(true_coarse and true_coarse["local_maximum"])
    row["coarse_depth_allowed"] = bool(true_coarse and true_coarse["depth_allowed"])

    # Coherence ranking and box refinement near the true period.
    coherence = []
    for index, band in enumerate(diagnostics["bands"]):
        if "duration_trial_days" not in band or band["status"] != "SEARCHED":
            continue
        path = f"/diagnostics/bands/{index}/periodogram_arrays"
        arrays = band_arrays(
            npz, path,
            ["period_days", "coherence_eligible_grid_index", "coherence_priority_rank",
             "coherence_refinement_state", "coherence_statistics"],
        )
        if "coherence_eligible_grid_index" not in arrays:
            continue
        periods = arrays["period_days"]
        eligible = arrays["coherence_eligible_grid_index"]
        if not len(eligible):
            continue
        candidate_periods = periods[eligible]
        close = np.flatnonzero(np.abs(candidate_periods / true_period - 1) < 0.003)
        for position in close:
            coherence.append({
                "band": band["band"],
                "duration": band["duration_trial_days"],
                "period": float(candidate_periods[position]),
                "priority_rank": int(arrays["coherence_priority_rank"][position]),
                "priority": float(arrays["coherence_statistics"][position, 1]),
                "state": int(arrays["coherence_refinement_state"][position]),
                "eligible_count": len(eligible),
            })
    row["coherence_candidates"] = len(coherence)
    row["coherence_best_rank"] = min((c["priority_rank"] for c in coherence), default=-1)
    row["coherence_refined"] = any(c["state"] == 2 for c in coherence)
    row["coherence_redundant"] = any(c["state"] == 1 for c in coherence)

    tested = []
    for band in diagnostics["bands"]:
        for peak in band.get("tested_peaks", []):
            if near(peak["period_days"], true_period) or near(peak["coarse_period_days"], true_period):
                tested.append({"band": band["band"], **{k: peak[k] for k in ("period_days", "search_snr", "depth_ppm", "reasons", "coherence_rank") if k in peak}})
    row["box_tested"] = len(tested)
    row["box_best_snr"] = max((t["search_snr"] for t in tested), default=np.nan)
    row["box_accepted"] = any(not t["reasons"] for t in tested)
    row["box_reasons"] = ";".join(sorted({r for t in tested for r in t["reasons"]}))

    box_decisions = []
    for band in diagnostics["bands"]:
        if "duration_trial_days" not in band:
            continue
        for decision in band.get("decisions", []):
            if "period_days" in decision and near(decision["period_days"], true_period) and "dip_inverse_snr_ratio" in decision:
                box_decisions.append(decision)
    row["box_shape_accepted"] = any(d["status"] == "SHAPE_ACCEPTED" for d in box_decisions)
    row["box_shape_reasons"] = ";".join(sorted({r for d in box_decisions for r in d["reasons"]}))

    # Physical stage near the true period.
    seeds, physical = [], []
    for band in diagnostics["bands"]:
        if not band.get("band", "").endswith("_physical"):
            continue
        for decision in band.get("decisions", []):
            if "period_days" not in decision or not near(decision["period_days"], true_period):
                continue
            if "seed_rank" in decision and "search_snr" not in decision or "seed_search_snr" in decision and "search_snr" not in decision:
                seeds.append(decision)
            else:
                physical.append(decision)
        for seed in band.get("physical_refinements", []):
            if near(seed["period_days"], true_period):
                seeds.append({"status": "PROCESSED", "reasons": [], **{k: seed[k] for k in ("seed_rank", "seed_search_snr", "period_days")}})
    row["physical_seed_present"] = bool(seeds)
    row["physical_seed_best_rank"] = min((s.get("seed_rank", 10**6) for s in seeds), default=-1)
    row["physical_seed_reasons"] = ";".join(sorted({r for s in seeds for r in s.get("reasons", [])}))
    row["physical_seed_snr"] = max((s.get("seed_search_snr", np.nan) for s in seeds), default=np.nan)
    row["physical_tested"] = len(physical)
    row["physical_best_snr"] = max((d.get("search_snr", np.nan) for d in physical), default=np.nan)
    row["physical_best_depth_ppm"] = max((d.get("depth_ppm", np.nan) for d in physical), default=np.nan)
    row["physical_statuses"] = ";".join(sorted({d["status"] for d in physical}))
    row["physical_reasons"] = ";".join(sorted({r for d in physical for r in d.get("reasons", [])}))
    stats_keys = ("dip_inverse_snr_ratio", "odd_even_sigma", "harmonic_delta_log_likelihood", "positive_cycle_fraction", "observed_events", "search_rank")
    best_physical = max(physical, key=lambda d: d.get("search_snr", -np.inf), default=None)
    for key in stats_keys:
        row[f"physical_{key}"] = best_physical.get(key, np.nan) if best_physical else np.nan

    merge = [d for d in diagnostics.get("band_merge", {}).get("decisions", []) if near(d["period_days"], true_period)]
    row["merge_statuses"] = ";".join(sorted({f"{d['status']}:{','.join(d['reasons'])}" for d in merge}))
    gf = diagnostics.get("global_fundamental_conditioning", {})
    gf_near = []
    for decision in gf.get("decisions", []):
        proposal = gf["source_proposals"][decision["proposal_index"]]
        if near(proposal["period_days"], true_period):
            gf_near.append(decision)
    row["global_fundamental_near"] = len(gf_near)
    row["global_fundamental_rejected"] = any(not d["retained"] for d in gf_near)
    row["global_fundamental_reasons"] = ";".join(sorted({r for d in gf_near for r in d["reasons"]}))
    row["global_fundamental_snr"] = max((d["statistic"].get("signed_snr") or np.nan for d in gf_near), default=np.nan)

    proposals = result["proposals"]
    row["proposal_aliases"] = ";".join(sorted({alias_of(p["period_days"], true_period) or "other" for p in proposals}))
    exact = [p for p in proposals if near(p["period_days"], true_period)]
    row["proposal_near_true"] = len(exact)
    row["proposal_near_true_matches"] = any(matches(p["period_days"], p["t0_bkjd"], c) for p in exact for c in clocks)
    row["proposal_best_snr"] = max((p["search_snr"] for p in proposals), default=np.nan)
    row["first_loss_stage"] = loss_stage(row, config)
    return row


def loss_stage(row, config):
    if row["recovered"]:
        return "RECOVERED"
    if row["status"] != "SEARCHED":
        return "DETECTOR_INELIGIBLE"
    if row["events"] < config["min_observed_events"]:
        return "UNOBSERVABLE_FEWER_THAN_3_EVENTS"
    if np.isnan(row["coarse_snr_1"]):
        return "TRUE_PERIOD_OUTSIDE_GRID"
    if not row["coarse_above_threshold"]:
        return "COARSE_BELOW_THRESHOLD"
    if not row["coarse_depth_allowed"]:
        return "COARSE_DEPTH_DOMAIN"
    if not row["coarse_local_maximum"] and not row["coherence_candidates"]:
        return "COARSE_NOT_LOCAL_MAXIMUM"
    if not row["coherence_candidates"]:
        return "COARSE_NOT_ELIGIBLE"
    if not row["coherence_refined"]:
        return "COHERENCE_REDUNDANT" if row["coherence_redundant"] else "COHERENCE_UNREFINED_BUDGET"
    if not row["box_tested"]:
        return "BOX_REFINEMENT_DRIFTED"
    if not row["box_accepted"]:
        return "BOX_REJECTED:" + row["box_reasons"]
    if not row["physical_seed_present"]:
        return "PHYSICAL_SEED_ABSENT"
    if not row["physical_tested"]:
        return "PHYSICAL_SEED_NOT_CHOSEN:" + row["physical_seed_reasons"]
    if "SHAPE_ACCEPTED" not in row["physical_statuses"] and "PROPOSAL" not in row["physical_statuses"]:
        return "PHYSICAL_REJECTED:" + row["physical_reasons"]
    if "PROPOSAL" not in row["physical_statuses"]:
        return "DISTINCT_PROPOSAL_LOSS:" + row["physical_reasons"]
    if row["merge_statuses"] and "PROPOSAL" not in row["merge_statuses"]:
        return "MERGE_LOSS:" + row["merge_statuses"]
    if row["global_fundamental_rejected"]:
        return "GLOBAL_FUNDAMENTAL:" + row["global_fundamental_reasons"]
    if row["proposal_near_true"] and not row["proposal_near_true_matches"]:
        return "PROPOSED_CLOCK_MISMATCH"
    if row["proposal_near_true"]:
        return "PROPOSED_MATCH_UNSCORED"
    return "UNEXPLAINED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    scored = json.load(open(args.round / f"{args.design}-scored-cases.json"))
    designs = {d["name"]: d for d in json.load(open(args.round / "designs.json"))}
    config = designs[args.design]["config"]
    clocks = defaultdict(list)
    for row in csv.DictReader(open(args.cohort / "clocks.csv")):
        clocks[row["case_id"]].append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in scored:
        if case["positive"] != "true":
            continue
        if args.only and args.only not in case["case_id"]:
            continue
        if case["preparation_status"] != "PREPARED":
            rows.append({"case_id": case["case_id"], "kind": case["kind"], "first_loss_stage": "INPUT_INELIGIBLE", "recovered": False})
            continue
        path = args.round / args.design / f"{job_key(case)}.json"
        try:
            rows.append(analyse_case(case, clocks[case["case_id"]], path, config))
        except Exception as error:
            rows.append({"case_id": case["case_id"], "kind": case["kind"], "first_loss_stage": f"LEDGER_ERROR:{type(error).__name__}:{error}", "recovered": bool(case["recovered"])})
        print(rows[-1]["case_id"].split(":")[-1], rows[-1]["first_loss_stage"], f"snr_fd={rows[-1].get('snr_white_fd', float('nan')):.1f}", f"events={rows[-1].get('events', -1)}", flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "case_id", k))
    with open(args.out / "ledger.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    summary = {
        "round": str(args.round), "design": args.design, "cases": len(rows),
        "recovered": sum(bool(r.get("recovered")) for r in rows),
        "stages_all": dict(Counter(r["first_loss_stage"] for r in rows).most_common()),
        "stages_snr_fd_ge_7": dict(Counter(r["first_loss_stage"] for r in rows if r.get("snr_white_fd", 0) >= 7.1 and r.get("events", 0) >= 3).most_common()),
        "stages_snr_fd_ge_10": dict(Counter(r["first_loss_stage"] for r in rows if r.get("snr_white_fd", 0) >= 10 and r.get("events", 0) >= 3).most_common()),
    }
    bins = [0, 3, 5, 7.1, 10, 15, 25, 50, 1e9]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if lo <= r.get("snr_white_fd", -1) < hi and r.get("events", 0) >= 3]
        table.append({"snr_bin": f"{lo}-{hi}", "cases": len(sel), "recovered": sum(bool(r["recovered"]) for r in sel), "synthetic": sum(r["kind"] == "synthetic_planet" for r in sel), "real": sum(r["kind"] == "real_planet" for r in sel)})
    summary["efficiency_by_snr_fd_events_ge_3"] = table
    summary["events_lt_3"] = sum(1 for r in rows if 0 <= r.get("events", -1) < 3)
    json.dump(summary, open(args.out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
