"""How close a calibration sample is to the KBonus PSF Flat campaign."""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

INDEX = "/Volumes/discPCIe/data/index.sqlite"
DESIGN = Path(__file__).resolve().parents[1] / "data/calibration-sample-design"
# LC2 inventory and KOI table, copied from kepler-diverse-20261028-v1 on 31 Oct 2026.
INVENTORY = DESIGN / "lane-inventory.parquet"
# Gaia DR3 photometry of every inventory source (ARI Heidelberg mirror, 1 Nov 2026).
COLOURS = DESIGN / "gaia-colours.parquet"
CATALOGUES = Path("/Volumes/discPCIe/data/reference-catalogues")
KOI_TABLE = DESIGN / "koi-cumulative-table.json"
CADENCE_DAYS = 1765.464 / 86400
LC2_COLUMNS = {
    "time": "time_bkjd",
    "cadence": "cadenceno",
    "quarter": "quarter",
    "flux": "psf_flat_kbonus_flux",
    "error": "psf_flat_kbonus_flux_err",
    "quality": "sap_quality",
    "flatten_mask": "flatten_mask",
    "raw": "psf_stitched_flux",
}
LOG_FEATURES = (
    "point_noise_ppm", "cdpp_1h_ppm", "cdpp_3h_ppm", "cdpp_6h_ppm",
    "cdpp_12h_ppm", "variability_ppm", "usable_cadences",
)  # fmt: skip
LINEAR_FEATURES = (
    "g_mag", "red_noise", "duty_cycle", "quarters", "seasons", "dip_rate",
    "excursion_rate", "jump_rate", "masked_fraction", "variability_period",
)  # fmt: skip
CATEGORIES = ("population", "tpf_relation")


def single_threaded():
    for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):  # fmt: skip
        os.environ[name] = "1"


def lane():
    """The PSF Flat lane: every LC2 source with any finite PSF Flat flux."""
    inv = pd.read_parquet(INVENTORY)
    inv = inv[inv.finite_psf_flat_kbonus > 0].copy()
    colours = pd.read_parquet(COLOURS, columns=["source_id", "bp_rp"])
    inv = inv.merge(colours, on="source_id", how="left")
    koi = pd.DataFrame(json.loads(KOI_TABLE.read_text()))
    eb = pd.read_csv(next(CATALOGUES.glob("kepler-eb-2016-*.csv")))
    relation = pd.Series("none", index=inv.index)
    relation[inv.origin_tpf_kic.isin(set(koi.kepid))] = "koi_other"
    planets = koi[koi.koi_disposition != "FALSE POSITIVE"].kepid
    relation[inv.origin_tpf_kic.isin(set(planets))] = "koi_planet_host"
    fps = koi[koi.koi_disposition == "FALSE POSITIVE"].kepid
    relation[inv.origin_tpf_kic.isin(set(fps))] = "koi_false_positive"
    relation[inv.origin_tpf_kic.isin(set(eb.KIC))] = "eclipsing_binary"
    inv["tpf_relation"] = relation
    inv["tpf_neighbours"] = inv.groupby("origin_tpf_kic").source_id.transform("size")
    return inv


def read_lc2(source_id, index=INDEX):
    from photometry.program import locate_source, source_table

    folder = locate_source(str(source_id), index)
    table = source_table(
        folder / "cadence.parquet", str(source_id), list(LC2_COLUMNS.values())
    )
    return {key: table[name].to_numpy() for key, name in LC2_COLUMNS.items()}


def prepare_lc2(arrays, epochs=True):
    """The campaign's preparation of LC2-like arrays."""
    from common import config_file
    from photometry.program import prepare_arrays

    arrays = dict(arrays)
    raw = np.asarray(arrays.pop("raw"), float)
    settings = dict(config_file()["photometry"])
    if not epochs:
        settings.pop("field_artifact_epochs")
    result, scales = prepare_arrays(arrays, settings)
    return result, raw[result["native_index"]], scales


def robust_std(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def binned(time, flux, width):
    """Means of flux in time bins of `width` days holding >= 70% of their cadences."""
    edges = np.floor((time - time.min()) / width).astype(np.int64)
    counts = np.bincount(edges)
    sums = np.bincount(edges, flux)
    full = counts >= 0.7 * width / CADENCE_DAYS
    centres = (np.flatnonzero(full) + 0.5) * width + time.min()
    return centres, sums[full] / counts[full]


def features(source_id):
    """Per-source realism features of one lane light curve (None if unusable)."""
    try:
        arrays, raw, _ = prepare_lc2(read_lc2(source_id))
    except Exception as error:
        return {
            "source_id": str(source_id),
            "error": f"{type(error).__name__}: {error}",
        }
    return {"source_id": str(source_id)} | array_features(arrays, raw)


def array_features(arrays, raw):
    """Realism features of prepared arrays."""
    ok, flux = arrays["mask"], arrays["flux"]
    out = {"usable_cadences": int(ok.sum()), "stored_cadences": int(len(ok))}
    if ok.sum() < 100:
        out["error"] = "fewer than 100 usable cadences"
        return out
    t, f = arrays["time"][ok], flux[ok]
    quarters = np.unique(arrays["quarter"][ok])
    span = t.max() - t.min()
    point = robust_std(np.diff(f)) / np.sqrt(2)
    out.update(
        quarters=len(quarters),
        seasons=len(set(int(q) % 4 for q in quarters)),
        baseline_days=float(span),
        duty_cycle=float(ok.sum() * CADENCE_DAYS / max(span, CADENCE_DAYS)),
        masked_fraction=float(1 - ok.mean()),
        point_noise_ppm=1e6 * point,
    )
    for label, hours in (("1h", 1), ("3h", 3), ("6h", 6), ("12h", 12)):
        _, means = binned(t, f, hours / 24)
        out[f"cdpp_{label}_ppm"] = 1e6 * robust_std(np.diff(means)) / np.sqrt(2)
    out["red_noise"] = out["cdpp_6h_ppm"] / (out["point_noise_ppm"] / np.sqrt(12))
    centres, half_day = binned(t, f, 0.5)
    if len(half_day) > 10:
        lo, hi = np.percentile(half_day, [5, 95])
        out["variability_ppm"] = 1e6 * (hi - lo)
        dev = np.abs(half_day - np.median(half_day)) / max(
            out["cdpp_12h_ppm"] / 1e6, 1e-9
        )
        out["excursion_rate"] = float((dev > 5).sum() / max(span, 1) * 100)
        out["variability_period"] = dominant_period(centres, half_day)
    resid = (f - np.median(f)) / max(point, 1e-9)
    out["dip_rate"] = float((resid < -5).sum() / max(span, 1) * 100)
    r = raw[ok]
    jumps = np.abs(np.diff(r / np.median(r))) / max(point, 1e-9)
    out["jump_rate"] = float((jumps > 20).sum() / max(span, 1) * 100)
    return out


def dominant_period(time, flux, low=1.0, high=40.0):
    """Lomb-Scargle peak period of half-day means between `low` and `high` days."""
    from astropy.timeseries import LombScargle

    if len(time) < 30 or time.max() - time.min() < 2 * low:
        return np.nan
    frequency = np.geomspace(1 / high, 1 / low, 2000)
    power = LombScargle(time, flux).power(frequency)
    return float(1 / frequency[np.argmax(power)])


# Mission cadence clock: cadence numbers 0..CENSUS_CADENCES-1 (KBonus spans 568-72,531).
CENSUS_CADENCES = 72600
STORED, USABLE, EPOCH, BAND_APERTURE, BAND_MASK, RETAINED, CLIPPED = (
    1 << i for i in range(7)
)


def census_matrices(out, rows, mode="r"):
    """Memory maps of the census."""
    out = Path(out)
    shape = (rows, CENSUS_CADENCES)
    return (
        np.memmap(out / "flux.f32", np.float32, mode, shape=shape),
        np.memmap(out / "raw.f32", np.float32, mode, shape=shape),
        np.memmap(out / "flags.u8", np.uint8, mode, shape=shape),
    )


def quarter_table(source_id):
    """Channel, module, output and season of each quarter of a source."""
    from photometry.program import locate_source, source_table

    folder = locate_source(str(source_id), INDEX)
    columns = ["quarter", "module", "output", "channel", "season"]
    table = source_table(folder / "quarter.parquet", str(source_id), columns)
    return table.to_pandas()


def census_row(job):
    """One reference curve into the census matrices."""
    row, source_id, out, rows = job
    flux_map, raw_map, flag_map = census_matrices(out, rows, "r+")
    try:
        arrays = read_lc2(source_id)
        view, raw, scales = prepare_lc2(arrays, epochs=False)
        masked, masked_raw, _ = prepare_lc2(arrays)
        quarters = quarter_table(source_id)
    except Exception as error:
        return {
            "row": row,
            "source_id": str(source_id),
            "error": f"{type(error).__name__}: {error}",
        }
    cadence = view["cadence"].astype(np.int64)
    usable = view["mask"]
    flux = np.where(usable, view["flux"], np.nan)
    raw_rel = np.full(len(raw), np.nan)
    for q in np.unique(view["quarter"][usable]):
        group = usable & (view["quarter"] == q)
        level = np.median(raw[group])
        if np.isfinite(level) and level > 0:
            raw_rel[group] = raw[group] / level
    quality = view["quality"].astype(np.int64)
    flags = np.full(len(cadence), STORED, np.uint8)
    flags[usable] |= USABLE
    flags[(view["exclusion_bits"] & 32) > 0] |= CLIPPED
    flags[~masked["mask"] & usable] |= EPOCH
    flags[(quality & 131072) > 0] |= BAND_APERTURE
    flags[(quality & 262144) > 0] |= BAND_MASK
    flags[(quality & 401552) > 0] |= RETAINED
    flux_map[row, cadence] = flux
    raw_map[row, cadence] = raw_rel
    flag_map[row, cadence] = flags
    flux_map.flush(), raw_map.flush(), flag_map.flush()
    levels = [
        {"row": row, "source_id": str(source_id), "quarter": int(r.quarter), "level": scales.get(str(r.quarter)),
         "channel": int(r.channel), "module": int(r.module), "output": int(r.output), "season": int(r.season)}
        for r in quarters.itertuples()
    ]  # fmt: skip
    return {"row": row, "source_id": str(source_id), "levels": levels} | array_features(
        masked, masked_raw
    )


def census(reference_path, out, workers=8):
    """The reference curves under production preparation."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ref = pd.read_parquet(reference_path)
    rows = len(ref)
    for array in census_matrices(out, rows, "w+"):
        array[:] = np.nan if array.dtype == np.float32 else 0
        array.flush()
    jobs = [(i, s, str(out), rows) for i, s in enumerate(ref.source_id)]
    results, levels = [], []
    single_threaded()
    with ProcessPoolExecutor(workers, initializer=single_threaded) as pool:
        for i, result in enumerate(pool.map(census_row, jobs, chunksize=16)):
            levels += result.pop("levels", [])
            results.append(result)
            if i % 2000 == 0:
                print(f"{i}/{rows}", flush=True)
    keep = ["source_id", "g_mag", "population", "tpf_relation", "tpf_neighbours",
            "teff_gspphot", "logg_gspphot", "family"]  # fmt: skip
    data = ref[keep].merge(pd.DataFrame(results), on="source_id", how="left")
    if "error" not in data:
        data["error"] = None
    data.to_parquet(out / "features.parquet")
    pd.DataFrame(levels).to_parquet(out / "levels.parquet")
    return data


def measure(source_ids, workers=3):
    single_threaded()
    with ProcessPoolExecutor(workers, initializer=single_threaded) as pool:
        rows = list(pool.map(features, source_ids, chunksize=32))
    return pd.DataFrame(rows)


def weighted_quantiles(x, w, qs):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    return np.interp(np.asarray(qs) * w.sum(), cw, x)


def ks_distance(a, wa, b, wb):
    grid = np.sort(np.r_[a, b])
    ca = np.searchsorted(np.sort(a), grid, side="right")
    fa = np.r_[0, np.cumsum(wa[np.argsort(a)])][ca] / wa.sum()
    cb = np.searchsorted(np.sort(b), grid, side="right")
    fb = np.r_[0, np.cumsum(wb[np.argsort(b)])][cb] / wb.sum()
    return float(np.max(np.abs(fa - fb)))


def compare(sample, reference, weights=None):
    """Feature-by-feature comparison of sample hosts against the reference."""
    weights = np.ones(len(sample)) if weights is None else np.asarray(weights, float)
    rows = []
    for name in LOG_FEATURES + LINEAR_FEATURES:
        if name not in sample or name not in reference:
            continue
        a, b = sample[name].to_numpy(float), reference[name].to_numpy(float)
        keep_a, keep_b = np.isfinite(a), np.isfinite(b)
        if name in LOG_FEATURES:
            keep_a &= a > 0
            keep_b &= b > 0
        a, wa, b = a[keep_a], weights[keep_a], b[keep_b]
        if len(a) < 10 or len(b) < 10:
            continue
        wb = np.ones(len(b))
        la, lb = (np.log10(a), np.log10(b)) if name in LOG_FEATURES else (a, b)
        ma, mb = np.average(la, weights=wa), lb.mean()
        sa = np.sqrt(np.average((la - ma) ** 2, weights=wa))
        pooled = np.sqrt((sa**2 + lb.var()) / 2)
        qs = (0.05, 0.25, 0.5, 0.75, 0.95)
        rows.append(
            {
                "feature": name,
                "scale": "log10" if name in LOG_FEATURES else "linear",
                "smd": float((ma - mb) / pooled) if pooled > 0 else 0.0,
                "ks": ks_distance(a, wa, b, wb),
                "sample_quantiles": weighted_quantiles(a, wa, qs).round(4).tolist(),
                "reference_quantiles": np.quantile(b, qs).round(4).tolist(),
                "n_sample": int(len(a)),
            }
        )
    categories = {}
    for name in CATEGORIES:
        if name in sample and name in reference:
            s = pd.Series(weights, index=sample.index).groupby(sample[name]).sum()
            r = reference[name].value_counts()
            categories[name] = {
                level: {
                    "sample": float(s.get(level, 0) / s.sum()),
                    "reference": float(r.get(level, 0) / r.sum()),
                }
                for level in sorted(set(s.index) | set(r.index))
            }
    return {"features": rows, "categories": categories}


HARD_REGIMES = {
    "duration_over_10h": lambda s: s.duration_days * 24 > 10,
    "duration_over_16h": lambda s: s.duration_days * 24 > 16,
    "over_100_transits": lambda s: s.observed_events > 100,
    "period_over_50d_three_events": lambda s: (
        (s.period_days > 50) & (s.observed_events >= 3)
    ),
    "usp_under_1d": lambda s: s.period_days < 1,
    "snr_7_to_10": lambda s: s.information_snr.between(7.1, 10),
    "snr_10_to_15": lambda s: s.information_snr.between(10, 15),
    "eccentric_over_0p3": lambda s: s.e > 0.3,
    "grazing": lambda s: s.grazing.astype(bool),
    "ttv": lambda s: (
        s.ttv_amplitude_days.notna() if "ttv_amplitude_days" in s else False
    ),
    "multi_system": lambda s: s.system_size > 1,
    "depth_under_200ppm": lambda s: s.flat_depth_ppm < 200,
    "flattening_loss_over_5pct": lambda s: s.local_transfer < 0.95,
}
HOST_REGIMES = {
    "M": lambda h: h.population == "M",
    "giant": lambda h: h.population == "giant",
    "subgiant": lambda h: h.population == "subgiant",
    "faint_g_over_18p5": lambda h: h.g_mag > 18.5,
    "few_quarters_under_6": lambda h: h.quarters < 6,
    "high_variability_top_decile": None,
    "jumps_top_decile": None,
}


def report(sample, reference_path, out=None):
    """Quality of a built sample against the lane reference (quality.json)."""
    sample = Path(sample)
    cases = pd.read_parquet(sample / "cases.parquet")
    signals = pd.read_parquet(sample / "signals.parquet")
    features = pd.read_parquet(sample / "host-features.parquet")
    ref = pd.read_parquet(reference_path)
    ref = ref[ref.error.isna()]
    hosts = cases[cases.kind == "control_original"].merge(
        features, on="source_id", how="left"
    )
    hosts["g_mag"] = hosts.g_mag.astype(float)
    result = {"cases": cases.kind.value_counts().to_dict()}
    result["host_balance_weighted"] = compare(
        hosts, ref, hosts.lane_weight.to_numpy(float)
    )
    result["host_balance_unweighted"] = compare(hosts, ref)
    worst = max(result["host_balance_weighted"]["features"], key=lambda r: r["ks"])
    result["host_balance_worst"] = {k: worst[k] for k in ("feature", "smd", "ks")}
    lane_cut = {
        "high_variability_top_decile": (
            "variability_ppm",
            ref.variability_ppm.quantile(0.9),
        ),
        "jumps_top_decile": ("jump_rate", ref.jump_rate.quantile(0.9)),
    }
    host_counts = {}
    for name, rule in HOST_REGIMES.items():
        if rule is None:
            column, cut = lane_cut[name]
            host_counts[name] = int((hosts[column] > cut).sum())
        else:
            host_counts[name] = int(rule(hosts).sum())
    result["host_regime_counts"] = host_counts
    planets = signals[
        (signals.kind == "planet")
        & signals.case_kind.isin(("synthetic_planet", "hz_planet"))
    ].merge(cases[["case_id", "partition", "quick"]], on="case_id", how="left")
    sizes = planets.groupby("case_id").size()
    planets["system_size"] = planets.case_id.map(sizes)
    w = planets.population_weight.to_numpy(float)
    from common import config_file

    detection = config_file()["detection"]
    searchable = planets.in_search_box.astype(bool)
    detectable = planets.detectable.astype(bool)
    result["planets"] = {
        "signals": int(len(planets)),
        "in_search_box": int(searchable.sum()),
        "detectable_information_snr_7p1": int(detectable.sum()),
        "weighted_share_detectable_of_searchable": float(
            w[detectable].sum() / w[searchable].sum()
        ),
        "period_days_bins": {
            str(k): int(v)
            for k, v in pd.cut(
                planets.period_days, [0, 0.5, 1, 10, 50, 100, 200, np.inf]
            )
            .value_counts(sort=False)
            .items()
        },
        "fewer_than_3_transits_in_range": int(
            (
                (planets.observed_events < 3)
                & planets.period_days.between(
                    detection["period_min_days"], detection["period_max_days"]
                )
            ).sum()
        ),
        "detectable_radius_bins": {
            str(k): int(v)
            for k, v in pd.cut(planets.radius_earth[detectable], [0, 2, 4, 8, 15, 30])
            .value_counts(sort=False)
            .items()
        },
    }
    real = signals[
        signals.kind.isin(["known_planet", "validated_planet"])
        | signals.kind.str.startswith("contamination")
    ]
    result["real_signals"] = {
        kind: {
            "signals": int(len(g)),
            "in_search_box": int(g.in_search_box.astype(bool).sum()),
            "detectable_on_own_depth": int(g.detectable.astype(bool).sum()),
            "measured_over_catalogue_depth_median": float(
                (g.measured_depth_ppm / g.catalogue_depth_ppm).median()
            )
            if "catalogue_depth_ppm" in g and g.catalogue_depth_ppm.notna().any()
            else None,
        }
        for kind, g in real.groupby("kind")
    }
    result["artifact_copies"] = {
        kind.removeprefix("artifact_"): g.artifact_pair.value_counts().to_dict()
        for kind, g in cases[cases.kind.str.startswith("artifact_")].groupby("kind")
    }
    result["lane_weight_sums"] = {
        kind: round(float(v), 1)
        for kind, v in cases.groupby("kind")
        .lane_weight.sum(min_count=1)
        .dropna()
        .items()
    }
    regimes = {}
    for name, rule in HARD_REGIMES.items():
        m = np.asarray(rule(planets), bool) & searchable.to_numpy()
        regimes[name] = {
            "searchable": int(m.sum()),
            "detectable": int((m & detectable.to_numpy()).sum()),
        }
    result["planet_regime_counts"] = regimes
    if "star_flux_fraction" in hosts:
        g = pd.cut(hosts.g_mag, [0, 16, 17, 18, 18.5, 19.5])
        result["star_flux_fraction_by_g"] = {
            str(k): np.nanquantile(v.to_numpy(float), (0.1, 0.25, 0.5))
            .round(3)
            .tolist()
            for k, v in hosts.groupby(g, observed=True).star_flux_fraction
        }
    snr_bins = [0, 1, 3, 5, 7.1, 10, 15, 20, 40, 100, np.inf]  # information S/N
    cut = pd.cut(planets.information_snr, snr_bins, right=False)
    result["snr_distribution_searchable"] = {
        str(k): int(v) for k, v in cut[searchable].value_counts(sort=False).items()
    }
    dur = pd.cut(planets.duration_days * 24, [0, 2, 4, 6, 8, 10, 13, 16, 24, 100])
    result["flattening_transfer_by_duration"] = {
        str(k): round(float(v), 4)
        for k, v in planets[planets.model_depth_ppm > 100]
        .groupby(dur, observed=True)
        .flattening_transfer.median()
        .items()
    }
    result["local_transfer_by_duration"] = {
        str(k): round(float(v), 4)
        for k, v in planets[planets.model_depth_ppm > 100]
        .groupby(dur, observed=True)
        .local_transfer.median()
        .items()
    }
    result["local_transfer_real_kbonus"] = {
        "(0, 2]": 0.998,
        "(2, 4]": 0.997,
        "(4, 6]": 0.991,
        "(6, 8]": 0.989,
        "(8, 10]": 0.970,
        "(10, 13]": 0.981,
        "(13, 16]": 0.876,
        "(16, 24]": 0.932,
        "(24, 100]": 0.856,
    }
    koi = pd.DataFrame(json.loads(KOI_TABLE.read_text()))
    kp = koi[
        (koi.koi_disposition != "FALSE POSITIVE")
        & (koi.koi_period <= 200)
        & (koi.koi_num_transits >= 3)
    ]
    d = planets[detectable]
    dw = d.population_weight.to_numpy(float)
    pairs = {
        "period_days": (d.period_days, kp.koi_period),
        "duration_hours": (d.duration_days * 24, kp.koi_duration),
        "transits": (
            d.observed_events.astype(float),
            kp.koi_num_transits.astype(float),
        ),
        "depth_ppm": (d.flat_depth_ppm, kp.koi_depth),
        "radius_earth": (d.radius_earth, kp.koi_prad),
    }
    result["detectable_vs_kepler_planets"] = {
        name: {
            "sample_weighted_quantiles": weighted_quantiles(
                a.to_numpy(float), dw, (0.1, 0.5, 0.9)
            )
            .round(3)
            .tolist(),
            "kepler_quantiles": np.nanquantile(b.to_numpy(float), (0.1, 0.5, 0.9))
            .round(3)
            .tolist(),
        }
        for name, (a, b) in pairs.items()
    }
    n_det = int(detectable.sum())
    expected_found = 0.35 * n_det
    result["power"] = {
        "detection_half_width_at_35pct": float(
            1.96 * math.sqrt(0.35 * 0.65 / max(n_det, 1))
        ),
        "expected_detected_signals": expected_found,
        "retention_half_width_at_96pct": float(
            1.96 * math.sqrt(0.96 * 0.04 / max(expected_found, 1))
        ),
    }
    negatives = cases[
        ~cases.kind.isin(["synthetic_planet", "known_planet", "validated_boi"])
    ]
    result["negatives"] = negatives.kind.value_counts().to_dict()
    others = signals[
        ~signals.kind.isin(["planet", "known_planet", "validated_planet"])
        & ~signals.case_kind.str.startswith("artifact_")
    ]
    result["negative_signal_depths_ppm"] = {
        kind: np.nanquantile(
            g.flat_depth_ppm.fillna(
                g.get("leak_depth_ppm", pd.Series(dtype=float))
            ).to_numpy(float),
            (0.1, 0.5, 0.9),
        )
        .round(0)
        .tolist()
        for kind, g in others.groupby("kind")
        if g.flat_depth_ppm.notna().any() or "leak_depth_ppm" in g
    }
    families = cases.groupby("family").partition.nunique()
    result["partition"] = {
        "families_in_both": int((families > 1).sum()),
        "by_partition_kind": cases.groupby(["partition", "kind"])
        .size()
        .unstack(0)
        .fillna(0)
        .astype(int)
        .to_dict(),
    }
    result["recovery_ceiling"] = recovery_ceiling(planets)
    known = signals[signals.kind == "known_planet"].merge(
        cases[["case_id", "partition", "quick"]], on="case_id", how="left"
    )
    result["recovery_ceiling_known_planets"] = {
        part: {
            "planets": int(len(g)),
            "in_search_box": int(g.in_search_box.astype(bool).sum()),
            "detectable_on_kbonus_depth": int(g.detectable.astype(bool).sum()),
        }
        for part, g in known.groupby("partition")
    }
    if out:
        Path(out).write_text(json.dumps(result, indent=1, default=float))
    return result


def template_match(planets, exposure_days=1765.464 / 86400):
    """How each planet's true duration meets the detector's templates at its period."""
    from detector.upstream_adapter import DURATION_WINDOW, MIN_TEMPLATE_CADENCES

    low, high = DURATION_WINDOW
    shortest = MIN_TEMPLATE_CADENCES * exposure_days
    ref = planets.reference_duration_days.to_numpy(float)
    period = planets.period_days.to_numpy(float)
    duration = planets.duration_days.to_numpy(float)
    floor = np.maximum(shortest, low * ref)
    top = np.minimum.reduce([high * ref, np.full(len(ref), 1.5), period / 3])
    return pd.Series(
        np.select(
            [top < floor, duration > top, duration < shortest],
            ["no_template", "longer_than_window", "shorter_than_templates"],
            "matched",
        ),
        index=planets.index,
    )


def recovery_ceiling(planets):
    """Per set, what the photons allow inside the search box."""
    from common import config_file
    from detector.upstream_adapter import DURATION_WINDOW, MIN_TEMPLATE_CADENCES

    detection = config_file()["detection"]
    match = template_match(planets)
    sets = {
        "all": planets.index == planets.index,
        "development": planets.partition == "development",
        "development_quick": (planets.partition == "development") & planets.quick,
        "holdout": planets.partition == "holdout",
        "holdout_quick": (planets.partition == "holdout") & planets.quick,
    }
    rows = {}
    for name, sel in sets.items():
        p, m = planets[sel], match[sel]
        detectable = p.detectable.astype(bool)
        rows[name] = {
            "planets": int(len(p)),
            "control_outside_period_range": int(
                (p.control_class == "outside_period_range").sum()
            ),
            "control_fewer_than_min_transits": int(
                (p.control_class == "fewer_than_min_transits").sum()
            ),
            "below_snr_threshold": int((p.control_class.isna() & ~detectable).sum()),
            "detectable": int(detectable.sum()),
            "detectable_template_matched": int((detectable & (m == "matched")).sum()),
            "detectable_longer_than_window": int(
                (detectable & (m == "longer_than_window")).sum()
            ),
            "detectable_shorter_than_templates": int(
                (detectable & (m == "shorter_than_templates")).sum()
            ),
            "detectable_no_template": int((detectable & (m == "no_template")).sum()),
            "ceiling_share_of_planets": float(detectable.mean()) if len(p) else None,
        }
    return {
        "settings": {
            "period_days": [detection["period_min_days"], detection["period_max_days"]],
            "min_observed_transits": detection["min_observed_transits"],
            "mes_threshold": detection["mes_threshold"],
            "duration_window": list(DURATION_WINDOW),
            "min_template_cadences": MIN_TEMPLATE_CADENCES,
        },
        "sets": rows,
    }


def reference(n, seed, out):
    frame = lane()
    chosen = frame.sample(n=n, random_state=seed)
    measured = measure(chosen.source_id.tolist())
    keep = ["source_id", "phot_g_mean_mag", "population", "tpf_relation",
            "tpf_neighbours", "teff_gspphot", "logg_gspphot", "family"]  # fmt: skip
    data = (
        chosen[keep]
        .rename(columns={"phot_g_mean_mag": "g_mag"})
        .merge(measured, on="source_id", how="left")
    )
    data.to_parquet(out)
    return data


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("out", type=Path)
    ref.add_argument("--n", type=int, default=20000)
    ref.add_argument("--seed", type=int, default=20261029)
    rep = sub.add_parser("report")
    rep.add_argument("sample", type=Path)
    rep.add_argument("--reference", type=Path, required=True)
    cen = sub.add_parser("census")
    cen.add_argument("reference", type=Path)
    cen.add_argument("out", type=Path)
    cen.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "report":
        print(
            json.dumps(
                report(args.sample, args.reference, args.sample / "quality.json"),
                indent=1,
                default=float,
            )
        )
        return
    if args.command == "census":
        data = census(args.reference, args.out, args.workers)
        print(len(data), "sources;", int(data.error.notna().sum()), "unusable")
        return
    if args.command == "reference":
        data = reference(args.n, args.seed, args.out)
        print(
            len(data),
            "sources;",
            int(data.get("error", pd.Series()).notna().sum()),
            "unusable",
        )


if __name__ == "__main__":
    main()
