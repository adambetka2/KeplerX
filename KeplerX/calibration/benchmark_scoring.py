"""Score a batch run on the survey benchmark (calibration/survey_benchmark.py)."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.scoring import epochs, match
from common import ROOT, read, write

MES_BINS = (0, 10, 20, 50, 100, 300, np.inf)
RATIOS = (1.0, 0.5, 2.0, 1 / 3, 3.0)
NEIGHBOUR_DEG = 2 / 60


def stage_times(result):
    times = {}
    for stage in result["stages"]:
        cpu = stage.get("cpu_user_seconds", 0) + stage.get("cpu_system_seconds", 0)
        times[f"{stage['stage']}_cpu"] = cpu
        times[f"{stage['stage']}_wall"] = stage.get("wall_seconds")
    return times


def load(run):
    with sqlite3.connect(Path(run) / "results.sqlite") as con:
        jobs = pd.read_sql("select case_id, state, result from jobs", con)
        candidates = pd.read_sql("select * from candidates", con)
        vetting = pd.read_sql(
            "select case_id, signal_id, state, metrics from vetting", con
        )
    times = pd.DataFrame([stage_times(json.loads(r)) for r in jobs.result])
    jobs = pd.concat([jobs.drop(columns="result"), times], axis=1)
    vetting["max_mes"] = [json.loads(m or "{}").get("max_mes") for m in vetting.metrics]
    candidates = candidates.merge(
        vetting.drop(columns="metrics").rename(columns={"state": "disposition"}),
        on=["case_id", "signal_id"], how="left",
    )  # fmt: skip
    return jobs, candidates


def known_planets(cases, candidates):
    rows = []
    by_case = dict(tuple(candidates.groupby("case_id")))
    for case_id, case in cases.iterrows():
        packet = read(Path(case.prepared) / "packet.json")
        start, baseline = packet["first_valid_bkjd"], packet["baseline_days"]
        found = by_case.get(case_id, candidates.iloc[:0])
        for planet in json.loads(case.planets):
            if not planet["duration_hours"] or np.isnan(planet["duration_hours"]):
                continue
            truth = {
                "period_days": planet["period_days"], "epoch_bkjd": planet["epoch_bkjd"],
                "duration_days": planet["duration_hours"] / 24,
            }  # fmt: skip
            best, hit = "miss", None
            for row in found.itertuples():
                outcome, _ = match(
                    {"period": row.period, "t0": row.t0}, truth, start, baseline
                )
                if outcome == "exact" or (outcome == "harmonic" and best == "miss"):
                    best, hit = outcome, row
                if outcome == "exact":
                    break
            rows.append({
                "case_id": case_id, "planet": planet["name"], "period": planet["period_days"],
                "depth_ppm": planet["depth_ppm"], "outcome": best,
                "transits": len(epochs(planet["epoch_bkjd"], planet["period_days"], start, baseline)),
                "max_mes": None if hit is None else hit.max_mes,
                "disposition": None if hit is None else hit.disposition,
            })  # fmt: skip
    return pd.DataFrame(rows)


def known_planet_summary(known, candidates):
    """Planets with at least three transits in the curve, the Detector's minimum."""
    planets = known_planets(known, candidates)
    planets = planets[planets.transits >= 3]
    exact = planets[planets.outcome == "exact"]
    kept = exact.disposition.isin(["KEPT", "UNCLASSIFIED"])
    bins = pd.cut(exact.max_mes, MES_BINS)
    return {
        "planets": len(planets),
        "outcome": planets.outcome.value_counts().to_dict(),
        "exact_rate": round(float((planets.outcome == "exact").mean()), 4),
        "exact_kept_by_vetting": round(float(kept.mean()), 4),
        "exact_kept_by_mes": {
            str(k): {"n": int(len(g)), "kept": round(float(kept[g.index].mean()), 4)}
            for k, g in exact.groupby(bins, observed=True)
        },
        "dismissed": exact[~kept][["planet", "period", "depth_ppm", "max_mes"]]
        .round(3).to_dict("records"),
    }  # fmt: skip


def same_clock(candidate, period, epoch):
    """Period equal or harmonic within 0.3%."""
    for ratio in RATIOS:
        if abs(candidate.period / (period * ratio) - 1) < 0.003:
            if ratio != 1.0:
                return True
            offset = ((candidate.t0 - epoch) / period + 0.5) % 1 - 0.5
            return abs(offset * period) < max(0.1, candidate.tdur)
    return False


def references():
    """KOIs, Kepler EBs (BKJD epochs) and earlier findings by source."""
    import csv

    from calibration.lane_census import INVENTORY, KOI_TABLE
    from calibration.sample_builder import DISPOSITIONS, EB_CATALOGUE

    koi = pd.DataFrame(json.loads(KOI_TABLE.read_text()))
    koi = koi.dropna(subset=["koi_period"]).rename(
        columns={"kepid": "kic", "koi_period": "period", "koi_time0bk": "epoch"}
    )[["kic", "period", "epoch", "ra", "dec", "koi_disposition"]]
    eb = pd.read_csv(EB_CATALOGUE)
    eb = eb[eb.Per > 0].assign(epoch=eb.BJD0 + 2400000 - 2454833)
    eb = eb.rename(columns={"KIC": "kic", "Per": "period", "_RA": "ra", "_DE": "dec"})
    earlier = {}
    for path in sorted(DISPOSITIONS.glob("*-dispositions-*.csv")):
        for row in csv.DictReader(path.open()):
            earlier.setdefault(row["source_id"], []).append(float(row["period_days"]))
    positions = pd.read_parquet(INVENTORY, columns=["source_id", "ra_deg", "dec_deg"])
    return (
        koi,
        eb[["kic", "period", "epoch", "ra", "dec"]],
        earlier,
        positions.set_index("source_id"),
    )


def explanation(candidate, case, koi, eb, earlier, positions):
    parent = int(case.origin_tpf_kic)
    for label, table in (
        ("parent KOI", koi[koi.kic == parent]),
        ("parent EB", eb[eb.kic == parent]),
    ):
        if any(same_clock(candidate, r.period, r.epoch) for r in table.itertuples()):
            return label
    ra, dec = positions.loc[str(case.source_id), ["ra_deg", "dec_deg"]]
    for label, table in (("neighbour KOI", koi), ("neighbour EB", eb)):
        near = np.hypot((table.ra - ra) * np.cos(np.radians(dec)), table.dec - dec)
        if any(
            same_clock(candidate, r.period, r.epoch)
            for r in table[near < NEIGHBOUR_DEG].itertuples()
        ):
            return label
    if any(
        abs(candidate.period / (p * r) - 1) < 0.003
        for p in earlier.get(str(case.source_id), [])
        for r in RATIOS
    ):
        return "earlier finding"
    return "unexplained"


def kept_explained(kept, cases):
    """Kept candidates of survey curves by what explains them, and their MES."""
    koi, eb, earlier, positions = references()
    labels = [
        explanation(row, cases.loc[row.case_id], koi, eb, earlier, positions)
        for row in kept.itertuples()
    ]
    kept = kept.assign(explained=labels)
    return {
        "by_explanation": kept.explained.value_counts().to_dict(),
        "unexplained_by_mes": pd.cut(kept.max_mes[kept.explained == "unexplained"], (0, 8, 10, 12, 20, 50, np.inf))
        .value_counts(sort=False).rename(str).to_dict(),
    }  # fmt: skip


def summary(jobs, candidates, cases):
    out = {"cases": jobs.state.value_counts().to_dict()}
    for name in sorted(n for n in cases.set.unique() if n.startswith("known-planets")):
        known = cases[(cases.set == name) & cases.index.isin(jobs.case_id)]
        if len(known):
            out[name.replace("-", "_")] = known_planet_summary(known, candidates)
    diverse = cases[(cases.set == "diverse") & cases.index.isin(jobs.case_id)]
    if len(diverse):
        weight = diverse.weight
        with_candidate = diverse.index.isin(candidates.case_id)
        kept = candidates[candidates.disposition.isin(["KEPT", "UNCLASSIFIED"])]
        with_kept = diverse.index.isin(kept.case_id)
        times = jobs.set_index("case_id").loc[diverse.index]
        out["survey_curves"] = {
            "curves": len(diverse),
            "with_candidate": round(float(with_candidate.mean()), 4),
            "with_candidate_weighted": round(float(np.average(with_candidate, weights=weight)), 4),
            "with_kept_candidate": round(float(with_kept.mean()), 4),
            "with_kept_candidate_weighted": round(float(np.average(with_kept, weights=weight)), 4),
            "candidates_per_curve": round(len(candidates[candidates.case_id.isin(diverse.index)]) / len(diverse), 3),
            "kept_candidates": kept_explained(kept[kept.case_id.isin(diverse.index)], cases),
            "seconds_per_curve": {
                column: {"mean": round(float(times[column].mean()), 3),
                         "weighted": round(float(np.average(times[column].fillna(0), weights=weight)), 3)}
                for column in times.columns if column.endswith(("_cpu", "_wall"))
            },
        }  # fmt: skip
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="Batch output folder (results.sqlite)")
    parser.add_argument(
        "--benchmark", type=Path, default=ROOT / "data/survey-benchmark"
    )
    args = parser.parse_args(argv)
    cases = pd.read_parquet(args.benchmark / "cases.parquet").set_index("case_id")
    jobs, candidates = load(args.run)
    jobs = jobs.join(cases[["set"]], on="case_id")
    result = summary(jobs, candidates, cases)
    write(args.run / "benchmark-score.json", result)
    print(json.dumps(result, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
