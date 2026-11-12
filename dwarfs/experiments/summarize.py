"""Summarize completed paired preparation trials without changing their decisions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("biweight_0.5d", "biweight_1.0d", "joint_spline_1.5d")
LABELS = ("0.5-day biweight", "1-day biweight", "Joint spline search")
OPERATORS = ("aperture", "prf_background")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median(rows, key):
    return float(np.median([row[key] for row in rows]))


def load_runs(paths):
    sources, bindings = [], {}
    for path in paths:
        result_path = path / "results.json"
        design = json.loads((path / "design.json").read_text())
        results = json.loads(result_path.read_text())
        if len(results) != len(design["inputs"]):
            raise ValueError("Partial preparation run cannot be summarized")
        if digest(path / "preparation.py") != design["source_sha256"]:
            raise ValueError("Preparation source snapshot differs")
        bindings[str(result_path)] = digest(result_path)
        for result in results:
            if digest(result["source"]["path"]) != result["input_sha256"]:
                raise ValueError("Preparation input differs")
            sources.append(result)
    identities = [
        (row["source"]["source_id"], row["source"]["operator"]) for row in sources
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source/operator result")
    return sources, bindings


def summarize(sources):
    rows, nulls = [], []
    for source in sources:
        identity = {key: source["source"][key] for key in ("source_id", "operator")}
        rows.extend(identity | row for row in source["results"])
        for block, methods in source["nulls"].items():
            nulls.extend(
                identity
                | {
                    "block_days": float(block),
                    "method": method,
                    "held_false_alarms": values["held_false_alarms"],
                    "held_count": len(values["held_maxima"]),
                    "interval": values["conditional_binomial_95_interval"],
                }
                for method, values in methods.items()
            )
    groups = []
    for operator in OPERATORS:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["operator"] == operator and row["method"] == method
            ]
            for label, low, high in (
                ("science", 40, 1000),
                ("positive_controls", 3000, 10000),
            ):
                subset = [
                    row for row in selected if low <= row["truth"]["depth_ppm"] <= high
                ]
                groups.append(
                    {
                        "operator": operator,
                        "method": method,
                        "ladder": label,
                        "trials": len(subset),
                        "recovered": sum(row["recovered"] for row in subset),
                        "median_retained_amplitude": median(
                            subset, "known_clock_retained_amplitude"
                        ),
                        "median_formal_score_delta": median(
                            subset, "known_clock_formal_score_delta"
                        ),
                    }
                )
    return rows, nulls, groups


def paired(rows):
    comparisons = []
    for operator in OPERATORS:
        selected = [
            row
            for row in rows
            if row["operator"] == operator and row["truth"]["depth_ppm"] <= 1000
        ]
        indexed = {
            (row["source_id"], row["injection"], row["method"]): row for row in selected
        }
        baseline = [row for row in selected if row["method"] == METHODS[0]]
        for method in METHODS[1:]:
            wins, losses, gains = 0, 0, []
            for old in baseline:
                new = indexed[old["source_id"], old["injection"], method]
                wins += int(new["recovered"] and not old["recovered"])
                losses += int(old["recovered"] and not new["recovered"])
                gains.append(
                    new["known_clock_formal_score_delta"]
                    / old["known_clock_formal_score_delta"]
                )
            comparisons.append(
                {
                    "operator": operator,
                    "candidate": method,
                    "reference": METHODS[0],
                    "wins": wins,
                    "losses": losses,
                    "median_formal_score_ratio": float(np.median(gains)),
                }
            )
    return comparisons


def figure(rows, nulls, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    source_count = len({row["source_id"] for row in rows})
    largest_held_rate = max(
        row["held_false_alarms"] / row["held_count"]
        for row in nulls
        if row["block_days"] == 1.0
    )
    for column, operator in enumerate(OPERATORS):
        selected = [row for row in rows if row["operator"] == operator]
        for method, label in zip(METHODS, LABELS):
            group = [row for row in selected if row["method"] == method]
            retention = [
                median(
                    [
                        row
                        for row in group
                        if row["truth"]["duration_hours"] == hours
                        and row["truth"]["depth_ppm"] <= 1000
                    ],
                    "known_clock_retained_amplitude",
                )
                for hours in (2, 4, 6)
            ]
            axes[0, column].plot((2, 4, 6), retention, "o-", label=label)
            depths = sorted({row["truth"]["depth_ppm"] for row in group})
            recall = [
                np.mean(
                    [
                        row["recovered"]
                        for row in group
                        if row["truth"]["depth_ppm"] == depth
                    ]
                )
                for depth in depths
            ]
            axes[1, column].plot(depths, recall, "o-", label=label)
        axes[0, column].set(
            title="Pixel aperture"
            if column == 0
            else "Fixed PRF + background operator",
            xlabel="Injected duration (hours)",
            ylabel="Median retained amplitude (fraction)",
            ylim=(0, 1.1),
            xticks=(2, 4, 6),
        )
        axes[1, column].set(
            xscale="log",
            xlabel="Nominal area-ratio depth (ppm)",
            ylabel="Conditional recovery (fraction)",
            ylim=(0, 1.05),
        )
        axes[1, column].axvspan(2000, 12000, color="grey", alpha=0.1)
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{source_count} scenes; paired transfer probes and a restricted 9.75–10.25-day search\n"
        f"Nominal 5% sign-block threshold; largest held 1-day false-alarm rate: {largest_held_rate:.1%}\n"
        "Shaded depths are strong controls, not the science target",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        fig.savefig(out / f"preparation-comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources, bindings = load_runs(args.runs)
    rows, nulls, groups = summarize(sources)
    report = {
        "summary_source_sha256": digest(__file__),
        "source_clusters": len({row["source_id"] for row in rows}),
        "bindings": bindings,
        "groups": groups,
        "paired": paired(rows),
        "nulls": nulls,
        "limits": [
            "Repeated probes are not independent population trials",
            "Formal scores and amplitude retention are not photon bounds",
            "Conditional sign-block nulls do not establish survey reliability",
        ],
    }
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.out / "trials.csv").open("w", newline="") as stream:
        flat = [
            {k: v for k, v in row.items() if not isinstance(v, dict)} | row["truth"]
            for row in rows
        ]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    figure(rows, nulls, args.out)


if __name__ == "__main__":
    main()
