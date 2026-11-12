"""Package the frozen twenty-source comparison without changing its decisions."""

import argparse
import json
from pathlib import Path

import numpy as np

from dwarfs.experiments import compare_photometry as comparison
from dwarfs.experiments.reprocess_summary import artifact

METHODS = comparison.METHODS
PIXELS = "pixel_reprocessed"
DELIVERED = ("delivered_sap", "delivered_psf", "delivered_psf_flat")
LABELS = {
    "delivered_sap": "SAP",
    "pixel_reprocessed": "Pixel correction",
    "cbv_pointing_only": "Mission vectors only",
    "delivered_psf": "PSF",
    "delivered_psf_flat": "PSF Flat",
}
COLORS = ("#38598b", "#278563", "#81858b", "#c07938", "#8d61a8")


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != "held_pvalues"}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def total_recovery(cases):
    result = {}
    for depth in comparison.DESIGN["depth_ppm"]:
        result[str(depth)] = {}
        for method in METHODS:
            rows = [case["nominal_recovery"][str(depth)][method] for case in cases]
            result[str(depth)][method] = {
                key: sum(row[key] for row in rows)
                for key in ("trials", "recovered", "incremental_recovery")
            }
    return result


def total_false_alarms(cases):
    result = {}
    for block in comparison.DESIGN["block_days"]:
        result[str(block)] = {}
        for method in METHODS:
            rows = [case["joint_nulls"][str(block)][method] for case in cases]
            alarms = sum(row["held_false_alarms"] for row in rows)
            count = sum(row["held_count"] for row in rows)
            result[str(block)][method] = {
                "held_false_alarms": alarms,
                "held_count": count,
                "rate": alarms / count,
                "per_source_rates": [row["held_false_alarm_rate"] for row in rows],
                "source_cluster_uncertainty": comparison.source_bootstrap(
                    [row["held_false_alarm_rate"] for row in rows]
                ),
            }
        pixel_rates = np.array(result[str(block)][PIXELS]["per_source_rates"])
        for method in METHODS:
            result[str(block)][method]["pixel_minus_comparator_rate"] = (
                comparison.source_bootstrap(
                    pixel_rates - result[str(block)][method]["per_source_rates"]
                )
            )
    return result


def retention_summary(trials):
    result = {}
    for family in sorted({row["profile_family"] for row in trials}):
        result[family] = {}
        for method in METHODS:
            values = np.array(
                [
                    row["known_clock_retention"]
                    for row in trials
                    if row["profile_family"] == family and row["method"] == method
                ]
            )
            result[family][method] = {
                "probes": len(values),
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
    return result


def pilot_decision(aggregate, false_alarms, retention):
    comparisons = aggregate["all20"]["primary_faint_recovery_difference"]
    recovery = {
        method: comparisons[method]["source_bootstrap_95_interval"][0] > 0
        for method in DELIVERED
    }
    nominal = retention["nominal"][PIXELS]
    guards = {
        "improved_faint_recovery_vs_each_delivered_product": all(recovery.values()),
        "pooled_held_false_alarm_rate_at_most_5_percent_both_families": all(
            cell[PIXELS]["rate"] <= 0.05 for cell in false_alarms.values()
        ),
        "nominal_projected_amplitude_retention_90_to_110_percent": (
            nominal["min"] >= 0.9 and nominal["max"] <= 1.1
        ),
    }
    return {
        "pilot_win": all(guards.values()),
        "guards": guards,
        "recovery_comparisons": recovery,
        "limits": [
            "Source-bootstrap intervals describe this selected development sample",
            "Equal nominal calibration does not guarantee equal empirical false alarms",
            "PSF products receive ideal conditional signal preservation",
            "Passing this pilot is not a detector calibration or adoption gate",
        ],
    }


def summarize(folder):
    comparison.verify_plan(folder)
    completion = json.loads((folder / "completion.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    cases = completion["cases"]
    if len(cases) != 20 or plan["design"] != comparison.DESIGN:
        raise ValueError("Require the completed twenty-source frozen comparison")
    evidence = [artifact(folder / name) for name in ("completion.json", "plan.json")]
    trials = []
    for case in cases:
        path = folder / case["name"] / "injections.json"
        if comparison.previous.digest(path) != case["injections_sha256"]:
            raise ValueError("Completed injection evidence changed")
        trials.extend(json.loads(path.read_text()))
        evidence.extend([artifact(path), artifact(path.parent / "report.json")])
    false_alarms = total_false_alarms(cases)
    retention = retention_summary(trials)
    return {
        "schema": 1,
        "experiment": "Twenty new background sources, five photometry arms",
        "design": plan["design"],
        "cases": compact(cases),
        "aggregate": completion["aggregate"],
        "nominal_recovery": total_recovery(cases),
        "combined_held_false_alarms": false_alarms,
        "known_clock_projected_retention": retention,
        "decision": pilot_decision(completion["aggregate"], false_alarms, retention),
        "evidence": evidence,
    }


def plot(report, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for i, method in enumerate(DELIVERED):
        for population, color in (("M", "#247f82"), ("K", "#bd8240")):
            cases = [c for c in report["cases"] if c["source_class"] == population]
            values = [
                c["observed_noise"][PIXELS]["4.0"]["source_relative_sigma_ppm"]
                / c["observed_noise"][method]["4.0"]["source_relative_sigma_ppm"]
                for c in cases
            ]
            offset = -0.13 if population == "M" else 0.13
            x = i + offset + np.linspace(-0.08, 0.08, len(values))
            axes[0].scatter(
                x, values, color=color, label=population if i == 0 else None
            )
    axes[0].axhline(1, color="black", lw=0.8)
    axes[0].set(
        xticks=range(3),
        xticklabels=[LABELS[m] for m in DELIVERED],
        yscale="log",
        ylabel="Pixel / delivered 4-hour scatter",
        title="Each point is one source",
    )
    axes[0].legend(title="Photometric selection", fontsize=8)
    recovery = report["nominal_recovery"]
    depths = sorted(recovery, key=float)
    for method, color in zip(METHODS, COLORS, strict=True):
        rates = [
            recovery[d][method]["incremental_recovery"] / recovery[d][method]["trials"]
            for d in depths
        ]
        axes[1].plot(
            list(map(float, depths)), rates, "o-", color=color, label=LABELS[method]
        )
    axes[1].axvspan(40, 1000, color="gray", alpha=0.08)
    axes[1].set(
        xscale="log",
        ylim=(-0.03, 1.03),
        xlabel="Nominal source depth (ppm)",
        ylabel="Incremental recovery fraction",
        title="Shaded: primary faint objective",
    )
    for offset, block, marker in ((-0.1, "1.0", "o"), (0.1, "2.0", "^")):
        cells = report["combined_held_false_alarms"][block]
        for index, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
            axes[2].scatter(
                index + offset, cells[method]["rate"] * 100, marker=marker, color=color
            )
    axes[2].axhline(5, color="black", lw=0.8, ls="--")
    axes[2].set(
        xticks=range(5),
        xticklabels=[LABELS[m] for m in METHODS],
        ylabel="Pooled held false alarms (%)",
        title="Circles: 1-day; triangles: 2-day nulls",
    )
    axes[2].tick_params(axis="x", labelrotation=35, labelsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle(
        "Twenty selected star-quarters; conditional PSF response; identical search",
        fontsize=11,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = summarize(args.run)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot(report, args.out)


if __name__ == "__main__":
    main()
