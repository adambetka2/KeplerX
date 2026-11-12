"""Fit and test the vetting policies of vetter/policy.py on a scored run."""

import argparse
import math

import numpy as np
import pandas as pd

from common import write
from vetter.policy import FEATURES, POLICIES, features

PLANETS = ("planet", "known_planet", "validated_planet")
NEGATIVES = (
    "unexplained", "eclipsing_binary", "blended_eclipsing_binary",
    "contamination_eclipsing_binary", "contamination_koi_false_positive",
)  # fmt: skip
FOLDS = 5
STRONGER = (2, 4, 8, 16)


def stronger(metrics, k):
    """The metrics of the same signal k times stronger."""
    m = dict(metrics)
    for name in ("max_mes", "max_sec_mes", "min_sec_mes", "cross_lane_mes"):
        if m.get(name) is not None:
            m[name] *= k
    for stat, red in (
        ("temporal_chi2_stat", "temporal_red_chi2"),
        ("channel_chi2_stat", "channel_red_chi2"),
    ):
        if m.get(stat) is not None and (m.get(red) or 0) > 0:
            signal = m[stat] * math.sqrt(m[red]) * k
            m[red] = 1.0 + (m[red] - 1.0) * k * k if m[red] > 1 else m[red]
            m[stat] = signal / math.sqrt(m[red])
    if (m.get("global_tran_red_chi2") or 0) > 1:
        m["global_tran_red_chi2"] = 1.0 + (m["global_tran_red_chi2"] - 1.0) * k * k
    return m


def feature_frame(candidates, strength=1):
    """Policy features of every candidate, from the scorer's m_ columns."""
    rows = []
    for row in candidates.to_dict("records"):
        metrics = {k[2:]: _number(v) for k, v in row.items() if k.startswith("m_")}
        rows.append(features(stronger(metrics, strength) if strength != 1 else metrics))
    return pd.DataFrame(rows, index=candidates.index, columns=FEATURES, dtype=float)


def folds(stars):
    """Fold of each candidate: stars, largest first, to the smallest fold."""
    load, fold_of = np.zeros(FOLDS), {}
    for star, size in stars.value_counts().items():
        k = int(np.argmin(load))
        fold_of[star] = k
        load[k] += size
    return stars.map(fold_of).to_numpy()


def logistic(x, y, nonnegative=()):
    """Penalised logistic regression, the weights at the indices `nonnegative` held >= 0."""
    free = np.ones(x.shape[1], bool)
    while True:
        intercept, weights = _newton(x[:, free], y)
        full = np.zeros(x.shape[1])
        full[free] = weights
        negative = [j for j in nonnegative if free[j] and full[j] < 0]
        if not negative:
            return intercept, full
        free[min(negative, key=lambda j: full[j])] = False


def _newton(x, y, iterations=50):
    """Penalised logistic regression by Newton steps: (intercept, weights)."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0] + [1.0] * x.shape[1])
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        gradient = design.T @ (p - y) + penalty @ beta
        hessian = (design * (p * (1 - p))[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.abs(step).max() < 1e-10:
            break
    return beta[0], beta[1:]


def fit(values, planet, policy, budget):
    """The policy's parameters fitted to these candidates."""
    names = list(policy["weights"])
    x = values[names].to_numpy()
    ok = np.isfinite(x).all(axis=1)
    nonnegative = policy.get("nonnegative", ())
    intercept, weights = logistic(
        x[ok], planet[ok].astype(float), [names.index(n) for n in nonnegative]
    )
    fitted = {
        "intercept": float(intercept),
        "weights": dict(zip(names, map(float, weights))),
        "nonnegative": tuple(nonnegative),
    }
    planet_scores = intercept + x[ok & planet] @ weights
    return fitted | {"cut": float(np.quantile(planet_scores, budget))}  # fmt: skip


def dismissed(values, fitted):
    """Which candidates the fitted policy dismisses (unavailable values never do)."""
    with np.errstate(invalid="ignore"):
        names = list(fitted["weights"])
        scores = fitted["intercept"] + values[names].to_numpy() @ np.array(
            [fitted["weights"][n] for n in names]
        )
        return scores < fitted["cut"]


def evaluate(candidates, policy, budget):
    values = feature_frame(candidates)
    planet = candidates.label.isin(PLANETS).to_numpy()
    used = planet | candidates.label.isin(NEGATIVES).to_numpy()
    fold = folds(candidates.signal_id.str.split(":").str[0])
    scaled = {k: feature_frame(candidates, k) for k in STRONGER}
    gone = np.zeros(len(candidates), bool)
    gone_stronger = {k: np.zeros(len(candidates), bool) for k in STRONGER}
    for k in range(FOLDS):
        train, test = used & (fold != k), fold == k
        fitted = fit(values[train], planet[train], policy, budget)
        gone[test] = dismissed(values[test], fitted)
        for strength, frame in scaled.items():
            gone_stronger[strength][test] = dismissed(frame[test], fitted)
    kept = candidates.assign(kept=~gone)
    nulls = kept[(kept.kind == "control_original") & (kept.label == "unexplained")]
    return {
        "budget": budget,
        "out_of_fold": {
            "planets_kept": round(float((~gone[planet]).mean()), 4),
            "negatives_dismissed": round(float(gone[used & ~planet].mean()), 4),
            "kept_by_label": kept.groupby("label").kept.mean().round(4).to_dict(),
            "planets_kept_by_mes": kept[planet].groupby(
                pd.cut(kept.mes[planet], [7.1, 8.5, 10, 12, 15, 20, np.inf]), observed=True
            ).kept.mean().round(4).rename(str).to_dict(),
            "null_original_cases_with_unexplained": {
                "before": int(nulls.case_id.nunique()),
                "after": int(nulls[nulls.kept].case_id.nunique()),
            },
            "kept_planets_dismissed_when_stronger": {
                str(k): round(float(g[planet & ~gone].mean()), 4)
                for k, g in gone_stronger.items()
            },
        },
        "fitted_on_all_candidates": fit(values[used], planet[used], policy, budget),
    }  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "score", help="Scoring output folder holding candidates.parquet"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", nargs="+", default=list(POLICIES))
    parser.add_argument("--budget", type=float, nargs="+", default=[0.02, 0.03, 0.04])
    parser.add_argument("--metrics", help="Parquet of added metrics by case and signal")
    args = parser.parse_args(argv)
    candidates = pd.read_parquet(f"{args.score}/candidates.parquet")
    if args.metrics:
        extra = pd.read_parquet(args.metrics).set_index(["case_id", "signal_id"])
        extra = extra.add_prefix("m_").reindex(
            pd.MultiIndex.from_frame(candidates[["case_id", "signal_id"]])
        )
        for column in extra:
            candidates[column] = extra[column].to_numpy()
    record = {
        "score": str(args.score),
        "added_metrics": args.metrics,
        "folds": f"{FOLDS} by source star",
        "planet_labels": PLANETS,
        "negative_labels": NEGATIVES,
        "policies": {
            name: [evaluate(candidates, POLICIES[name], b) for b in args.budget]
            for name in args.policy
        },
    }
    write(f"{args.out}/vetting-policy.json", record)
    for name, rows in record["policies"].items():
        for row in rows:
            oof = row["out_of_fold"]
            print(name, row["budget"], oof["planets_kept"], oof["negatives_dismissed"],
                  oof["kept_by_label"].get("unexplained"),
                  oof["kept_planets_dismissed_when_stronger"],
                  row["fitted_on_all_candidates"])  # fmt: skip


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
