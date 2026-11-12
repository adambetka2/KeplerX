"""An out-of-sample benchmark of real KBonus survey curves."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import population as pop
from calibration import sample_builder as sb
from calibration.lane_census import INVENTORY, KOI_TABLE
from common import ROOT, load_packet, receipt, save_packet, write

SEED = 20261102
PER_STRATUM = 32
QUICK = 8
G_BINS = (0.0, 15.0, 17.0, 18.0, 30.0)
G_LABELS = ("G<15", "G15-17", "G17-18", "G>18")
CALIBRATION_CASES = ROOT / "data/calibration-sample/cases.parquet"
FP_FLAGS = ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec")


def eligible():
    inv = pd.read_parquet(INVENTORY)
    hosts = set(pd.read_parquet(CALIBRATION_CASES).source_id.astype(str))
    keep = (
        (inv.finite_psf_flat_kbonus > 0)
        & (inv.partition == "development")
        & ~inv.source_id.astype(str).isin(hosts)
    )
    frame = inv[keep].copy()
    frame["g_bin"] = pd.cut(frame.phot_g_mean_mag, G_BINS, labels=G_LABELS)
    frame["stratum"] = frame.population.astype(str) + "/" + frame.g_bin.astype(str)
    return frame, len(inv)


def diverse(frame, rng):
    rows = []
    for stratum, group in frame.groupby("stratum", sort=True):
        take = group.sample(n=min(PER_STRATUM, len(group)), random_state=rng)
        take = take.assign(weight=len(group) / len(take), set="diverse")
        take["quick"] = np.arange(len(take)) < QUICK
        rows.append(take)
    return pd.concat(rows)


def known_planets():
    """Development KBonus curves of confirmed-planet hosts that the calibration sample did not use."""
    confirmed = pd.DataFrame(json.loads(sb.CONFIRMED.read_text()))
    files = {int(p.name.split("kic-")[1][:9]): p for p in sb.KBONUS_FITS.glob("*.fits")}
    confirmed = confirmed[
        confirmed.kic_id.isin(files)
        & confirmed.period_days.notna()
        & confirmed.epoch_bkjd.notna()
    ]
    cases = pd.read_parquet(CALIBRATION_CASES)
    used = set(cases.source_id[cases.kind == "known_planet"])
    koi = pd.DataFrame(json.loads(KOI_TABLE.read_text()))
    stars = koi.drop_duplicates("kepid").set_index("kepid")
    rows = []
    for kic, group in confirmed.groupby("kic_id"):
        family = f"kic{kic}"
        if family in used or sb.partition(family) != "development":
            continue
        star = stars.loc[kic] if kic in stars.index else None
        planets = [
            {"name": r.kepler_name, "koi": r.koi_name, "period_days": r.period_days,
             "epoch_bkjd": r.epoch_bkjd, "duration_hours": r.duration_hours,
             "depth_ppm": r.depth_ppm, "radius_earth": r.planet_radius_earth}
            for r in group.itertuples()
        ]  # fmt: skip
        others = [
            {"koi": r.kepoi_name, "disposition": r.koi_disposition,
             "period_days": r.koi_period, "epoch_bkjd": r.koi_time0bk,
             "false_positive_flags": [f[-2:] for f in FP_FLAGS if getattr(r, f)]}
            for r in koi[(koi.kepid == kic) & (koi.koi_disposition != "CONFIRMED")].itertuples()
        ]  # fmt: skip
        rows.append({
            "set": "known-planets", "source_id": family, "kic": kic, "origin_tpf_kic": kic,
            "family": family, "population": "kepler_target", "weight": 1.0, "quick": False,
            "ra_deg": None, "dec_deg": None, "fits": str(files[kic]),
            "phot_g_mean_mag": None if star is None else star.koi_kepmag,
            "teff_gspphot": None if star is None else star.koi_steff,
            "logg_gspphot": None if star is None else star.koi_slogg,
            "planets": json.dumps(planets), "other_kois": json.dumps(others),
        })  # fmt: skip
    return pd.DataFrame(rows)


def case_id(row):
    return f"{row['set']}-{row['source_id']}"


def build_one(job):
    row, out = job
    host = {
        "source_id": str(row["source_id"]),
        "kic": None if pd.isna(row["kic"]) else int(row["kic"]),
        "parent_kic": int(row["origin_tpf_kic"]),
        "ra_deg": None if pd.isna(row["ra_deg"]) else float(row["ra_deg"]),
        "dec_deg": None if pd.isna(row["dec_deg"]) else float(row["dec_deg"]),
    }
    folder = Path(out) / "light-curves" / case_id(row)
    try:
        if isinstance(row.get("fits"), str):
            arrays, _ = sb.fits_arrays(row["fits"])
            provenance = {"kbonus_fits": receipt(row["fits"])}
        else:
            arrays, provenance = sb.lc2_arrays(host["source_id"])
        stellar = pop.catalogue_stellar(row["teff_gspphot"], row["logg_gspphot"])
        prepared, _, metadata = sb.prepare(arrays, host, stellar, provenance)
    except ValueError as error:
        return case_id(row), None, str(error)
    folder.mkdir(parents=True, exist_ok=True)
    save_packet(
        folder, prepared, metadata | {"case_id": case_id(row), "benchmark": "survey"}
    )
    return case_id(row), str(folder), None


def one_quarter(row, out):
    """The known-planet packet cut to its quarter with most usable cadences."""
    arrays, metadata = load_packet(row["prepared"])
    quarters, counts = np.unique(arrays["quarter"][arrays["mask"]], return_counts=True)
    quarter = int(quarters[np.argmax(counts)])
    keep = arrays["quarter"] == quarter
    arrays = {name: values[keep] for name, values in arrays.items()}
    good, time = arrays["mask"], arrays["time"]
    case = row["case_id"].replace("known-planets-", "known-planets-one-quarter-")
    metadata = metadata | {
        "case_id": case, "quarter_kept": quarter, "stored_cadences": int(len(time)),
        "usable_cadences": int(good.sum()), "baseline_days": float(np.ptp(time[good])),
        "first_valid_bkjd": float(time[good].min()),
        "last_valid_bkjd": float(time[good].max()),
        "median_formal_error_ppm": float(np.median(arrays["error"][good]) * 1e6),
        "normalization_by_quarter": {
            k: v for k, v in metadata["normalization_by_quarter"].items()
            if int(k) == quarter
        },
    }  # fmt: skip
    metadata.pop("arrays")
    folder = Path(out) / "light-curves" / case
    folder.mkdir(parents=True, exist_ok=True)
    save_packet(folder, arrays, metadata)
    return case, str(folder)


def derive_one_quarter(out):
    cases = pd.read_parquet(out / "cases.parquet")
    cases = cases[cases.set != "known-planets-one-quarter"]
    known = cases[(cases.set == "known-planets") & cases.prepared.notna()]
    derived = known.copy()
    made = [one_quarter(row, out) for row in known.to_dict("records")]
    derived["case_id"] = [c for c, _ in made]
    derived["prepared"] = [f for _, f in made]
    derived["set"] = "known-planets-one-quarter"
    cases = pd.concat([cases, derived], ignore_index=True)
    cases.to_parquet(out / "cases.parquet")
    manifest(out / "manifests/known-planets-one-quarter.jsonl", derived)
    print(len(derived), "one-quarter curves")


def derive_sweep(out, count, workers):
    cases = pd.read_parquet(out / "cases.parquet")
    cases = cases[cases.set != "sweep"]
    frame, _ = eligible()
    frame = frame[~frame.source_id.astype(str).isin(cases.source_id.astype(str))]
    rng = np.random.default_rng([SEED, count])
    drawn = frame.sample(n=min(count, len(frame)), random_state=rng)
    drawn = drawn.assign(set="sweep", weight=len(frame) / count, quick=False)
    drawn["case_id"] = [case_id(r) for r in drawn.to_dict("records")]
    jobs = [(r, str(out)) for r in drawn.to_dict("records")]
    with ProcessPoolExecutor(workers) as pool:
        done = {c: (f, e) for c, f, e in pool.map(build_one, jobs, chunksize=8)}
    drawn["prepared"] = [done[c][0] for c in drawn.case_id]
    drawn["preparation_error"] = [done[c][1] for c in drawn.case_id]
    drawn["g_bin"] = drawn.g_bin.astype(str)
    cases = pd.concat([cases, drawn.reindex(columns=cases.columns)], ignore_index=True)
    cases.to_parquet(out / "cases.parquet")
    manifest(out / "manifests/sweep.jsonl", drawn[drawn.prepared.notna()])
    print(
        int(drawn.prepared.notna().sum()),
        "sweep curves,",
        int(drawn.preparation_error.notna().sum()),
        "not prepared",
    )


def manifest(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in frame.itertuples():
            stream.write(
                json.dumps({"case_id": row.case_id, "prepared": row.prepared}) + "\n"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--one-quarter", action="store_true")
    parser.add_argument("--sweep", type=int)
    args = parser.parse_args(argv)
    if args.one_quarter:
        return derive_one_quarter(args.out)
    if args.sweep:
        return derive_sweep(args.out, args.sweep, args.workers)
    rng = np.random.default_rng(SEED)
    frame, total = eligible()
    cases = pd.concat([diverse(frame, rng), known_planets()], ignore_index=True)
    cases["case_id"] = [case_id(r) for r in cases.to_dict("records")]
    jobs = [(r, str(args.out)) for r in cases.to_dict("records")]
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(build_one, jobs, chunksize=4))
    folders = {c: (f, e) for c, f, e in done}
    cases["prepared"] = [folders[c][0] for c in cases.case_id]
    cases["preparation_error"] = [folders[c][1] for c in cases.case_id]
    columns = [
        "case_id", "set", "source_id", "kic", "origin_tpf_kic", "family", "population",
        "phot_g_mean_mag", "g_bin", "stratum", "quarter_count", "teff_gspphot",
        "logg_gspphot", "weight", "quick", "planets", "other_kois", "prepared",
        "preparation_error",
    ]  # fmt: skip
    cases = cases.reindex(columns=columns)
    cases["g_bin"] = cases.g_bin.astype(str)
    cases["quick"] = cases.quick.astype(bool)
    cases.to_parquet(args.out / "cases.parquet")
    ok = cases[cases.prepared.notna()]
    manifest(args.out / "manifests/diverse.jsonl", ok[ok.set == "diverse"])
    manifest(
        args.out / "manifests/diverse-quick.jsonl", ok[(ok.set == "diverse") & ok.quick]
    )
    manifest(args.out / "manifests/known-planets.jsonl", ok[ok.set == "known-planets"])
    write(args.out / "build.json", {
        "seed": SEED, "inventory_sources": total, "eligible_sources": len(frame),
        "per_stratum": PER_STRATUM, "quick_per_stratum": QUICK,
        "cases": cases.groupby("set").size().to_dict(),
        "prepared": ok.groupby("set").size().to_dict(),
        "preparation_errors": int(cases.preparation_error.notna().sum()),
        "strata": frame.groupby("stratum").size().to_dict(),
    })  # fmt: skip
    print(json.dumps(json.loads((args.out / "build.json").read_text())["prepared"]))


if __name__ == "__main__":
    main()
