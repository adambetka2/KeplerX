"""Bright-first target list of the M/K SAP survey from the census and the transit-noise table.

Targets are the census M and K dwarfs sorted by Gaia G. A source is hopeless when a 1,000 ppm
transit cannot reach search SNR 7.1 even in white noise at the SAP median noise of its G bin:
SNR = depth / noise * sqrt(events per quarter * quarters), with 30 three-hour events per quarter
(P = 3 d, 3 h binned noise) or 180 one-hour events per quarter (P = 0.5 d, 1 h binned noise).
Hopeless sources stay in the list, flagged; the survey queue skips them.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "dump/mk-cohort-census-20261013"
G_EDGES = [0, 16, 16.5, 17, 17.5, 18, 18.5, 19]
EVENTS = {"binned_3h": 30, "binned_1h": 180}  # events per quarter at P = 3 d and P = 0.5 d
DEPTH = 1000e-6
SNR_MIN = 7.1
COLUMNS = ["source_id", "shard", "kic", "population", "phot_g_mean_mag", "quarter_count"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--out", type=Path, default=ROOT / "results/mk-survey-targets-20261014.csv")
    args = parser.parse_args()
    census_path = args.census / "census/cohort-census.parquet"
    noise_path = args.census / "noise/transit-noise.parquet"
    census = pq.read_table(census_path, columns=COLUMNS).to_pandas()
    targets = census[census["population"].isin(["M", "K"])].copy()
    if targets["phot_g_mean_mag"].isna().any() or targets["quarter_count"].isna().any():
        raise ValueError("Every M/K target needs a Gaia G magnitude and a quarter count")
    noise = pq.read_table(noise_path, columns=["view", "phot_g_mean_mag", *EVENTS]).to_pandas()
    sap = noise[noise["view"] == "sap_stitched"]
    medians = sap.groupby(pd.cut(sap["phot_g_mean_mag"], G_EDGES), observed=True)[list(EVENTS)].median()
    if len(medians) != len(G_EDGES) - 1:
        raise ValueError("The noise table must cover every G bin")
    bins = pd.cut(targets["phot_g_mean_mag"].clip(upper=G_EDGES[-1] - 0.01), G_EDGES)
    quarters = targets["quarter_count"].astype(float)
    snr = pd.DataFrame(
        {
            column: DEPTH / bins.map(medians[column]).astype(float) * np.sqrt(events * quarters)
            for column, events in EVENTS.items()
        }
    )
    targets["white_noise_snr_1000ppm"] = snr.max(axis=1).round(1)
    targets["hopeless"] = (targets["white_noise_snr_1000ppm"] < SNR_MIN).astype(int)
    targets["g_bin"] = bins.astype(str)
    targets = targets.sort_values(["phot_g_mean_mag", "source_id"]).reset_index(drop=True)
    targets["order"] = np.arange(len(targets))
    targets["phot_g_mean_mag"] = targets["phot_g_mean_mag"].round(3)
    targets["kic"] = targets["kic"].astype("Int64")
    columns = ["order", *COLUMNS, "white_noise_snr_1000ppm", "hopeless"]
    targets[columns].to_csv(args.out, index=False)
    by_bin = targets.groupby("g_bin", observed=True)["hopeless"].agg(["size", "sum"])
    by_population = targets.groupby("population")["hopeless"].agg(["size", "sum"])
    summary = {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "census": {"path": str(census_path), "sha256": digest(census_path)},
        "noise": {"path": str(noise_path), "sha256": digest(noise_path), "view": "sap_stitched"},
        "criterion": {
            "depth_ppm": DEPTH * 1e6,
            "snr_min": SNR_MIN,
            "events_per_quarter": EVENTS,
            "g_edges": G_EDGES,
            "g_clip": G_EDGES[-1] - 0.01,
        },
        "noise_median_ppm_by_g_bin": {
            str(index): {column: round(float(row[column]) * 1e6) for column in EVENTS}
            for index, row in medians.iterrows()
        },
        "rows": len(targets),
        "hopeless": int(targets["hopeless"].sum()),
        "queued": int((targets["hopeless"] == 0).sum()),
        "by_population": {
            population: {"rows": int(row["size"]), "hopeless": int(row["sum"])}
            for population, row in by_population.iterrows()
        },
        "by_g_bin": {
            label: {"rows": int(row["size"]), "hopeless": int(row["sum"])}
            for label, row in by_bin.iterrows()
        },
        "targets": {"path": str(args.out), "sha256": digest(args.out)},
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("rows", "hopeless", "queued", "by_population")}))


if __name__ == "__main__":
    main()
