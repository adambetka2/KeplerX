"""Summarize completed paired pixel-reprocessing experiments without rescoring."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dwarfs.experiments.reprocess_validation import METHODS, source_bootstrap


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_runs(paths):
    sources, trials, artifacts = [], [], []
    for path in paths:
        completion = path / "completion.json"
        result = json.loads(completion.read_text())
        artifacts.extend([artifact(completion), artifact(path / "plan.json")])
        for source in result["cases"]:
            sources.append(source)
            records = path / source["name"] / "injections.json"
            trials.extend(
                dict(row, source_id=source["source_id"])
                for row in json.loads(records.read_text())
            )
            artifacts.extend(
                [artifact(records), artifact(records.parent / "report.json")]
            )
    ids = [source["source_id"] for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("A source appears in more than one completed run")
    return sources, trials, artifacts


def summarize_cell(rows, sources):
    result = {}
    effects, mission_effects = [], []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        retention = np.array([row["known_clock_retention"] for row in selected])
        result[method] = {
            "trials": len(selected),
            "recoveries": sum(row["recovered"] for row in selected),
            "incremental_recoveries": sum(
                row["incremental_recovery"] for row in selected
            ),
            "baseline_coincident": sum(row["baseline_recovered"] for row in selected),
            "retention_p10_median_p90": np.quantile(
                retention, [0.1, 0.5, 0.9]
            ).tolist(),
            "retention_min_max": [float(retention.min()), float(retention.max())],
        }
    for source in sources:
        by_method = [
            [
                row
                for row in rows
                if row["source_id"] == source and row["method"] == method
            ]
            for method in METHODS
        ]
        if any(not records for records in by_method):
            continue
        rates = [
            np.mean([row["incremental_recovery"] for row in records])
            for records in by_method
        ]
        effects.append(float(rates[1] - rates[0]))
        mission_effects.append(float(rates[1] - rates[2]))
    result["paired_source_recovery_change"] = source_bootstrap(effects)
    result["paired_source_recovery_change_vs_mission_only"] = source_bootstrap(
        mission_effects
    )
    return result


def recovery_summary(trials, ids):
    result = {}
    for family in sorted({row["profile_family"] for row in trials}):
        family_rows = [row for row in trials if row["profile_family"] == family]
        result[family] = {
            str(depth): summarize_cell(
                [row for row in family_rows if row["truth"]["depth_ppm"] == depth], ids
            )
            for depth in sorted({row["truth"]["depth_ppm"] for row in family_rows})
        }
    return result


def summarize(paths):
    sources, trials, artifacts = load_runs(paths)
    ids = [source["source_id"] for source in sources]
    return {
        "schema": 1,
        "experiment": "Adaptive pixel reprocessing, full-quarter development pilot",
        "sources": sources,
        "recovery": recovery_summary(trials, ids),
        "evidence": artifacts,
        "boundaries": [
            "Five metadata-selected sources; probes are dependent within source",
            "Nominal depths use delivered PSF flux and an adopted spatial profile",
            "Recovery requires the correct clock, at least three observed events "
            "and significance under both block-null calibrations",
            "Conditional surrogate false alarms do not establish survey reliability",
            "Compare arms within this run; the earlier pilot used shorter spans "
            "and a different source-flux calibration",
            "Delivered PSF has observed-noise measurements only; "
            "its adaptive extraction was not replayed",
            "No physical photon limit, discovery, validation or detector adoption "
            "follows from this pilot",
        ],
    }


def plot(report, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sources = report["sources"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    x = np.arange(len(sources))
    labels = [
        f"…{s['source_id'][-6:]}\n{s['name'].rsplit('-', 1)[-1]}" for s in sources
    ]
    for offset, method, color, label in (
        (-0.25, METHODS[0], "#38598b", "Replayed SAP"),
        (0, METHODS[2], "#926da4", "Mission vectors only"),
        (0.25, "delivered_psf", "#bd8240", "Delivered PSF"),
    ):
        ratios = [
            s["observed_noise"][METHODS[1]]["4.0"]["source_relative_sigma_ppm"]
            / s["observed_noise"][method]["4.0"]["source_relative_sigma_ppm"]
            for s in sources
        ]
        axes[0].bar(x + offset, ratios, 0.23, color=color, label=label)
    axes[0].axhline(1, color="black", lw=0.8)
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="New / baseline 4-hour noise",
        title="Source-equivalent precision",
    )
    nominal = report["recovery"]["nominal"]
    depths = sorted(nominal, key=float)
    for method, color, label in (
        (METHODS[0], "#38598b", "Replayed SAP"),
        (METHODS[1], "#318562", "Pixel reprocessing"),
        (METHODS[2], "#926da4", "Mission vectors only"),
    ):
        rates = [
            nominal[d][method]["incremental_recoveries"] / nominal[d][method]["trials"]
            for d in depths
        ]
        axes[1].plot(list(map(float, depths)), rates, "o-", color=color, label=label)
    axes[1].set(
        xscale="log",
        ylim=(-0.03, 1.03),
        xlabel="Injected nominal source depth (ppm)",
        ylabel="Incremental recovery fraction",
        title="Same search; calibrated thresholds",
    )
    for offset, method, color, label in (
        (-0.2, METHODS[0], "#38598b", "Replayed SAP"),
        (0, METHODS[1], "#318562", "Pixel reprocessing"),
        (0.2, METHODS[2], "#926da4", "Mission vectors only"),
    ):
        for block, marker in (("1.0", "o"), ("2.0", "^")):
            rates = [
                s["joint_nulls"][block][method]["held_false_alarm_rate"] * 100
                for s in sources
            ]
            axes[2].scatter(
                x + offset,
                rates,
                marker=marker,
                color=color,
                label=f"{label}, {block} d",
            )
    axes[2].axhline(5, color="black", lw=0.8, ls="--", label="Calibration target")
    axes[2].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Held-out false alarms (%)",
        title="128 held surrogates per cell",
    )
    axes[2].legend(
        handles=[
            Line2D([], [], color="gray", marker="o", ls="", label="1-day nulls"),
            Line2D([], [], color="gray", marker="^", ls="", label="2-day nulls"),
            Line2D([], [], color="black", ls="--", label="5% calibration target"),
        ],
        loc="upper left",
        fontsize=8,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=color, lw=4, label=label)
            for color, label in (
                ("#38598b", "Replayed SAP"),
                ("#318562", "Pixel reprocessing"),
                ("#926da4", "Mission vectors only"),
                ("#bd8240", "Delivered PSF"),
            )
        ],
        loc="outside lower center",
        ncol=4,
        fontsize=9,
    )
    figure.suptitle(
        "Pixel reprocessing: five full quarters; conditional development evidence",
        fontsize=11,
    )
    figure.savefig(output / "pixel-reprocessing.png", dpi=180)
    figure.savefig(output / "pixel-reprocessing.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    report = summarize(args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot(report, args.output)


if __name__ == "__main__":
    main()
