"""Audit the complete 40-run TIC validation census."""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from pipeline.vetting.statistical_validation import (
    RUNS_PER_TARGET,
    digest,
    probabilities,
    restore,
    verify,
    write_json,
)


def result_directory(plan, source_id, index):
    return Path(plan["output_root"]) / "runs" / source_id / f"seed-{index:02d}"


def validate_weight(path, row, draws, deep):
    if row["draws"] != draws:
        raise ValueError("Wrong family denominator")
    weight = path.parent / row["weight_file"]
    if digest(weight) != row["weight_sha256"]:
        raise ValueError("Stored weights changed: " + str(weight))
    if not deep:
        return
    with np.load(weight) as arrays:
        logs, bounded = arrays["log_weight"], arrays["bounded"]
        if logs.shape != (draws,) or bounded.shape != logs.shape:
            raise ValueError("Attempted-draw array differs")
        measured = float(logsumexp(logs) - np.log(draws))
        if not np.isclose(measured, row["log_evidence"], rtol=0, atol=1e-8):
            raise ValueError("Evidence does not reproduce from weights")
        if (
            int(bounded.sum()) != row["bounded_physical_draws"]
            or np.isfinite(logs[bounded]).any()
        ):
            raise ValueError("Bound indicators differ")
        omitted = (
            float(arrays["bounded_log_weight_upper"]) + np.log(bounded.sum() / draws)
            if bounded.any()
            else -np.inf
        )
        for key, value in {
            "omitted_log_evidence_upper": omitted,
            "log_evidence_upper": float(np.logaddexp(measured, omitted)),
        }.items():
            if not np.isclose(value, row[key], rtol=0, atol=1e-8):
                raise ValueError("Evidence bound does not reproduce from weights")


def validate_result(path, target, target_index, run_index, plan, receipt, deep):
    data = restore(json.loads(path.read_text()))
    identity = data["identity"]
    expected_seed = int(
        np.random.SeedSequence(
            plan["seed_entropy"] + [target_index, run_index]
        ).generate_state(1, dtype=np.uint64)[0]
    )
    expected = {
        "source_id": target["source_id"],
        "run_index": run_index,
        "seed": expected_seed,
        "draws_per_scenario": plan["draws_per_scenario"],
    }
    if data["status"] != "COMPLETE_INDEPENDENT_RUN" or any(
        identity[key] != value for key, value in expected.items()
    ):
        raise ValueError("Production identity differs: " + str(path))
    if identity["freeze_sha256"] != digest(receipt):
        raise ValueError("Frozen integration identity differs: " + str(path))
    names = data["scenario_names"]
    rows = data["scenario_evidences"]
    if (
        len(names) != target["scenario_count"]
        or len(set(names)) != len(names)
        or names != [row["scenario"] for row in rows]
    ):
        raise ValueError("Incomplete scenario census: " + str(path))
    for row in rows:
        validate_weight(path, row, plan["draws_per_scenario"], deep)
    recomputed = probabilities(rows)
    for key, value in recomputed.items():
        expected_values = value if isinstance(value, dict) else {key: value}
        saved_values = (
            data["probabilities"][key]
            if isinstance(value, dict)
            else data["probabilities"]
        )
        if any(
            not np.isclose(number, saved_values[name], rtol=0, atol=1e-12)
            for name, number in expected_values.items()
        ):
            raise ValueError("Reported probabilities do not reproduce from evidence")
    return data


def distribution(values):
    values = np.asarray(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = sd / np.sqrt(len(values))
    interval = [
        mean - float(t.ppf(0.975, len(values) - 1)) * sem,
        mean + float(t.ppf(0.975, len(values) - 1)) * sem,
    ]
    return {
        "mean": mean,
        "sample_standard_deviation": sd,
        "standard_error_of_mean": sem,
        "observed_repeat_range": [float(values.min()), float(values.max())],
        "approximate_mean_95_percent_interval": interval,
        "interval_scope": "Observed independent-run numerical scatter. Heavy tails, unvisited modes and model uncertainty are not certified by this interval.",
    }


def target_summary(target, results, thresholds):
    metrics = {}
    for measure in ["probabilities"]:
        metrics[measure] = {}
        for category in ["FPP", "NFPP", "EB"]:
            values = [row[measure][category] for row in results]
            summary = distribution([value["probability"] for value in values])
            summary["mean_numerical_lower"] = float(
                np.mean(
                    [
                        min(value["probability"], value["probability_lower"])
                        for value in values
                    ]
                )
            )
            summary["mean_numerical_upper"] = float(
                np.mean(
                    [
                        max(value["probability"], value["probability_upper"])
                        for value in values
                    ]
                )
            )
            metrics[measure][category] = summary
    return {
        "source_id": target["source_id"],
        "catalogue": "TIC",
        "ephemeris": "DISCOVERY",
        "complete_runs": len(results),
        "required_runs": RUNS_PER_TARGET,
        "scenario_families": target["scenario_count"],
        "status": "COMPLETE_TARGET",
        "metrics": metrics,
        "thresholds": thresholds,
        "fpp_below_threshold": metrics["probabilities"]["FPP"]["mean_numerical_upper"]
        < thresholds["FPP_lt"],
        "nfpp_below_threshold": metrics["probabilities"]["NFPP"]["mean_numerical_upper"]
        < thresholds["NFPP_lt"],
        "maximum_bounded_posterior_mass_upper": max(
            row["probabilities"]["bounded_posterior_mass_upper"] for row in results
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    plan, _ = verify()
    base = Path(plan["output_root"])
    receipt = base / "freeze.json"
    targets, missing, files = [], [], []
    for target_index, target in enumerate(plan["targets"]):
        rows = []
        for index in range(RUNS_PER_TARGET):
            directory = result_directory(plan, target["source_id"], index)
            path = directory / "result.json"
            if not path.exists():
                missing.append({"source_id": target["source_id"], "run_index": index})
                continue
            rows.append(
                validate_result(
                    path,
                    target,
                    target_index,
                    index,
                    plan,
                    receipt,
                    args.deep,
                )
            )
            files.append({"path": str(path), "sha256": digest(path)})
        if rows and len(rows) != RUNS_PER_TARGET:
            continue
        if rows:
            if any(row["scenario_names"] != rows[0]["scenario_names"] for row in rows):
                raise ValueError("Independent runs have different scenario censuses")
            targets.append(target_summary(target, rows, plan["thresholds"]))
    result = {
        "status": "COMPLETE_NUMERICAL_CAMPAIGN_PENDING_PHYSICAL_AUDIT"
        if not missing
        else "IN_PROGRESS_NO_ADMISSION",
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "catalogue": "TIC",
        "ephemeris": "DISCOVERY",
        "complete_runs": len(files),
        "required_runs": len(plan["targets"]) * RUNS_PER_TARGET,
        "targets": targets,
        "missing": missing,
        "results": files,
        "deep_weight_replay": args.deep,
        "scope": "Conditional astrophysical FPP and NFPP. Native convergence, acquisition checks, stellar/RUWE gates and catalogue/vetting evidence remain separate requirements.",
    }
    write_json(base / "numerical-campaign-summary.json", result)
    with (base / "numerical-campaign-summary.csv").open("w", newline="") as stream:
        fields = [
            "source_id",
            "complete_runs",
            "scenario_families",
            "FPP_mean_percent",
            "FPP_SD_percentage_points",
            "NFPP_upper_mean_percent",
            "status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in targets:
            fpp = row["metrics"]["probabilities"]["FPP"]
            writer.writerow(
                {
                    "source_id": row["source_id"],
                    "complete_runs": row["complete_runs"],
                    "scenario_families": row["scenario_families"],
                    "FPP_mean_percent": 100 * fpp["mean"],
                    "FPP_SD_percentage_points": 100 * fpp["sample_standard_deviation"],
                    "NFPP_upper_mean_percent": 100
                    * row["metrics"]["probabilities"]["NFPP"]["mean_numerical_upper"],
                    "status": row["status"],
                }
            )
    print(
        json.dumps(
            {
                "status": result["status"],
                "complete_runs": len(files),
                "required_runs": result["required_runs"],
            }
        )
    )


if __name__ == "__main__":
    main()
