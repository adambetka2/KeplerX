"""Spacecraft and KBonus artifacts."""

import argparse
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import lane_census as R
from calibration.signal_injection import flattened
from common import config_file
from photometry.program import field_artifacts

CLASSES = (
    "field_excursions",
    "q2_comb",
    "momentum_dumps",
    "rolling_band",
    "gap_ramps",
    "sensitivity_drops",
)
MODELS = R.DESIGN / "artifact-models"
BIN = 6  # cadences per 3-hour stack bin
COMB_BKJD = (132.0, 160.0)
ACTIVE_FACTOR = 2.0
DUMP_EDGES = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        13,
        16,
        20,
        25,
        30,
        36,
        44,
        52,
        62,
        74,
        88,
        104,
        122,
        146,
        175,
    ]
)
HOOK_CADENCES = 24
HOOK_KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)  # cadences
NULL_OFFSET = 300  # cadences from a gap edge or flagged run to its null twin
DROP_RECOVERY = (0.3, 0.9)
DROP_TAU_DAYS = (0.1, 2.0)
DROPS_PER_COPY = 2
NOISE_MATCHED = 200


def robust_sd(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def mission_clock(feats):
    """Cadence -> BKJD and quarter, from the census curve with most quarters."""
    best = feats.sort_values(["quarters", "usable_cadences"], ascending=False)
    arrays = R.read_lc2(best.source_id.iloc[0])
    cadence = arrays["cadence"].astype(int)
    grid = np.arange(R.CENSUS_CADENCES)
    time = np.interp(grid, cadence, arrays["time"])
    quarter = arrays["quarter"][
        np.clip(np.searchsorted(cadence, grid), 0, len(cadence) - 1)
    ]
    return time, quarter.astype(int)


def missing_runs(flags):
    """Mission-wide missing cadences."""
    from scipy.ndimage import maximum_filter1d

    stored = np.zeros(R.CENSUS_CADENCES)
    for i in range(0, flags.shape[0], 2000):
        stored += ((np.asarray(flags[i : i + 2000]) & R.STORED) > 0).sum(0)
    expect = maximum_filter1d(stored, 101)
    missing = np.flatnonzero((expect > 1000) & (stored < 0.2 * expect))
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    return [(int(r[0]), len(r)) for r in runs if len(r)]


def since_marks(marks):
    """Cadences since the last of `marks`, -1 beyond the last dump bin."""
    since = np.full(R.CENSUS_CADENCES, -1)
    last, marks = -(10**9), set(marks)
    for c in range(R.CENSUS_CADENCES):
        if c in marks:
            last = c
        since[c] = c - last if c - last < DUMP_EDGES[-1] else -1
    return since


def bin_noise(row):
    """3-hour bin means of one census curve and their bin-to-bin scatter."""
    nb = R.CENSUS_CADENCES // BIN
    fb = np.asarray(row[: nb * BIN], float).reshape(nb, BIN)
    count = np.isfinite(fb).sum(1)
    with np.errstate(all="ignore"):
        mean = np.where(count >= BIN // 2, np.nanmean(fb, 1), np.nan)
    good = np.isfinite(mean)
    if good.sum() < 30:
        return mean, np.nan
    return mean, robust_sd(np.diff(mean[good])) / math.sqrt(2)


def bin_z(row):
    """3-hour bin means of one census curve in units of their bin-to-bin scatter, as the stack counts them."""
    mean, sd = bin_noise(row)
    if not sd > 0:
        return None
    return (mean - np.nanmedian(mean)) / sd


def stack(flux):
    """Per 3-hour bin: sources with a value, and with a 3-sigma dip."""
    nb = R.CENSUS_CADENCES // BIN
    valid, low = np.zeros(nb, np.int64), np.zeros(nb, np.int64)
    for i in range(flux.shape[0]):
        z = bin_z(flux[i])
        if z is None:
            continue
        good = np.isfinite(z)
        valid += good
        low += good & (z < -3)
    return valid, low


def excursion_runs(row, z, bins, keep_share, rng):
    """A donor's significant excursions inside `bins`."""
    empty = np.zeros(0, np.int32), np.zeros(0, np.float32)
    if z is None:
        return empty
    kept = rng.random(len(z)) < keep_share
    strong = bins & kept & np.isfinite(z) & (np.abs(z) >= 3)
    if not strong.any():
        return empty
    weak = bins & np.isfinite(z) & (np.abs(z) >= 1.5)
    keep = strong.copy()
    for _ in range(4):
        grown = keep | (weak & (np.r_[False, keep[:-1]] | np.r_[keep[1:], False]))
        if np.array_equal(grown, keep):
            break
        keep = grown
    cadences = (np.flatnonzero(keep)[:, None] * BIN + np.arange(BIN)).ravel()
    values = np.asarray(row, float)[cadences] - 1
    smooth = pd.Series(values).rolling(5, center=True, min_periods=1).mean().to_numpy()
    good = np.isfinite(values)
    return cadences[good].astype(np.int32), smooth[good].astype(np.float32)


def dump_profile(d, label, sel, point):
    """Mean residual per since-dump bin, centred, and its chi^2."""
    nbins = len(DUMP_EDGES) - 1
    n = np.bincount(label[sel], None, nbins)
    s = np.bincount(label[sel], d[sel], nbins)
    mean = np.where(n >= 10, s / np.maximum(n, 1), np.nan)
    centre = mean - np.nanmedian(mean)
    chi2 = float(np.nansum((centre / (point / np.sqrt(np.maximum(n, 1)))) ** 2))
    return centre, chi2, int(np.isfinite(centre).sum())


def dump_profiles(d, point, since, false_since, quarter):
    """Per quarter, the donor's mean residual in bins of cadences since the last dump."""
    nbins = len(DUMP_EDGES) - 1
    label = np.digitize(since, DUMP_EDGES) - 1
    null = np.digitize(false_since, DUMP_EDGES) - 1
    out = []
    for q in np.unique(quarter[np.isfinite(d)]):
        sel = np.isfinite(d) & (quarter == q) & (label >= 0) & (label < nbins)
        sel_null = np.isfinite(d) & (quarter == q) & (null >= 0) & (null < nbins)
        if sel.sum() < 500 or sel_null.sum() < 500:
            continue
        centre, chi2, dof = dump_profile(d, label, sel, point)
        _, chi2_null, _ = dump_profile(d, null, sel_null, point)
        if dof >= 10 and chi2 > 4 * max(dof, chi2_null):
            out.append((int(q), np.nan_to_num(centre).astype(np.float32)))
    return out


def mean_noise(d, lengths=(3, 6, 12, 24, 48)):
    """Robust scatter of non-overlapping n-cadence means of a curve's usable cadences, for each n."""
    x = d[np.isfinite(d)]
    out = {}
    for n in lengths:
        k = len(x) // n
        out[n] = robust_sd(x[: k * n].reshape(k, n).mean(1)) if k >= 20 else np.nan
    return out


def run_offset(d, run, usable, avoid, scatter, point):
    """Mean offset of `run` from the surrounding usable cadences outside `avoid`."""
    lo, hi = int(run[0]), int(run[-1])
    around = np.r_[max(lo - 48, 0) : lo, hi + 1 : min(hi + 49, len(d))]
    around = around[usable[around] & ~avoid[around]]
    if len(around) < 12:
        return None
    offset = float(np.nanmean(d[run]) - np.nanmedian(d[around]))
    lengths = np.array(list(scatter))
    n = lengths[np.argmin(np.abs(np.log(lengths / len(run))))]
    sd = max(scatter[n] * math.sqrt(n / len(run)), point / math.sqrt(len(run)))
    return offset, (offset / sd if np.isfinite(sd) and sd > 0 else 0.0)


def band_runs(d, flags_row, point):
    """A donor's rolling-band-flagged runs (at least 3 cadences) offset by at least 3 sigma."""
    usable = np.isfinite(d)
    band = usable & ((flags_row & (R.BAND_APERTURE | R.BAND_MASK)) > 0)
    index = np.flatnonzero(band)
    counts = np.zeros(4, np.int64)
    if not len(index):
        return [], counts
    scatter = mean_noise(d)
    out = []
    for run in np.split(index, np.flatnonzero(np.diff(index) > 2) + 1):
        if len(run) < 3:
            continue
        result = run_offset(d, run, usable, band, scatter, point)
        if result:
            counts[0] += 1
            if abs(result[1]) >= 3:
                counts[1] += 1
                out.append((int(run[0]), len(run), result[0]))
        twin = run + NULL_OFFSET
        if twin[-1] < len(d) and usable[twin].all() and not band[twin].any():
            result = run_offset(d, twin, usable, band, scatter, point)
            if result:
                counts[2] += 1
                counts[3] += abs(result[1]) >= 3
    return out, counts


def fit_hook(y, k):
    """Best A exp(-k/kappa) by least squares over the kappa grid: (A, kappa, sum of squared weights)."""
    best = None
    for kappa in HOOK_KAPPAS:
        w = np.exp(-k / kappa)
        ww = float(w @ w)
        amp = float(w @ y) / ww
        cost = float(((y - amp * w) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, amp, kappa, ww)
    return best[1:]


def hook_at(d, edge, side, point, scatter):
    """The fitted hook beside one gap edge."""
    k = np.arange(HOOK_CADENCES)
    far = np.arange(HOOK_CADENCES, 4 * HOOK_CADENCES)
    if side == "after":
        near_idx, far_idx = edge + k, edge + far
    else:
        near_idx, far_idx = edge - k, edge - far
    if far_idx.min() < 0 or far_idx.max() >= len(d):
        return None
    y, base = d[near_idx], d[far_idx]
    good = np.isfinite(y)
    if (
        good.sum() < 0.8 * HOOK_CADENCES
        or not np.isfinite(y[:3]).all()
        or np.isfinite(base).sum() < 36
    ):
        return None
    level = np.nanmedian(base)
    amp, kappa, _ = fit_hook(y[good] - level, k[good])
    z3 = (np.mean(y[:3]) - level) / max(scatter[3], point / math.sqrt(3))
    return amp, kappa, abs(z3)


def gap_hooks(d, point, gaps):
    """A donor's hooks at gap edges whose three nearest cadences depart by at least 6 sigma."""
    counts = np.zeros(4, np.int64)
    scatter = mean_noise(d, (3,))
    out = []
    for start, length in gaps:
        for side, edge, twin in (
            ("after", start + length, start + length + NULL_OFFSET),
            ("before", start - 1, start - 1 - NULL_OFFSET),
        ):
            hook = hook_at(d, edge, side, point, scatter)
            if hook:
                counts[0] += 1
                if hook[2] >= 6:
                    counts[1] += 1
                    out.append((int(edge), side, hook[0], hook[1]))
            null = hook_at(d, twin, side, point, scatter)
            if null:
                counts[2] += 1
                counts[3] += null[2] >= 6
    return out, counts


def raw_steps(raw_row, point, near_gap):
    """Sharp persistent declines of the raw flux (relative size)."""
    x = pd.Series(np.asarray(raw_row, float))
    left = x.rolling(24, min_periods=19).median().to_numpy()
    right = np.r_[left[24:], np.full(24, np.nan)]
    later = np.r_[
        x.rolling(48, min_periods=38).median().to_numpy()[72:], np.full(72, np.nan)
    ]
    l3 = x.rolling(3, min_periods=3).median().to_numpy()
    r3 = np.r_[l3[3:], np.full(3, np.nan)]
    step = right - left
    sd = 1.25 * point * math.sqrt(2 / 24)
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(
            (step < -6 * sd)
            & (r3 - l3 < 0.6 * step)
            & (later - left < 0.4 * step)
            & ~near_gap
        )
    if not len(cand):
        return []
    groups = np.split(cand, np.flatnonzero(np.diff(cand) > 48) + 1)
    return [float(step[g[np.argmin(step[g])]]) for g in groups]


SHARED = {}


def load_shared(path):
    """Pool initializer: the census maps, clock and class windows, once per worker."""
    R.single_threaded()
    with open(path, "rb") as stream:
        SHARED.update(pickle.load(stream))
    SHARED["maps"] = R.census_matrices(SHARED["census"], SHARED["rows"])


def measure_donor(row):
    """Every class's measurements for one census curve."""
    s = SHARED
    flux, raw, flags = s["maps"]
    f, r, fl = (
        np.asarray(flux[row], float),
        np.asarray(raw[row], float),
        np.asarray(flags[row]),
    )
    usable = np.isfinite(f)
    if usable.sum() < 100:
        return {"row": row}
    d = f - 1
    point = robust_sd(np.diff(f[usable])) / math.sqrt(2)
    z = bin_z(f)
    rng = np.random.default_rng(row)
    _, binned = bin_noise(f)
    left = s["unmasked_active"] & np.isfinite(z) if z is not None else None
    quiet = s["quiet"] & np.isfinite(z) if z is not None else None
    band, band_counts = band_runs(d, fl, point)
    hooks, hook_counts = gap_hooks(d, point, s["gaps"])
    return {
        "row": row,
        "point": point,
        "bin_noise": binned,
        "unmasked_active_bins": int(left.sum()) if left is not None else 0,
        "unmasked_active_dips": int((left & (z < -3)).sum()) if left is not None else 0,
        "quiet_bins": int(quiet.sum()) if quiet is not None else 0,
        "quiet_dips": int((quiet & (z < -3)).sum()) if quiet is not None else 0,
        "field": excursion_runs(f, z, s["field_bins"], s["keep_share"], rng),
        "comb": excursion_runs(f, z, s["comb_bins"], s["keep_share"], rng),
        "dumps": dump_profiles(d, point, s["since"], s["false_since"], s["quarter"]),
        "band": band,
        "band_counts": band_counts,
        "hooks": hooks,
        "hook_counts": hook_counts,
        "steps": raw_steps(r, point, s["near_gap"]),
    }


def measure(census, out, workers=8):
    census, out = Path(census), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(census / "features.parquet")
    rows = len(feats)
    flux, raw, flags = R.census_matrices(census, rows)
    runs = missing_runs(flags)
    dumps = [s for s, n in runs if n <= 2]
    gaps = [(s, n) for s, n in runs if n >= 12]
    time, quarter = mission_clock(feats)
    since = since_marks(dumps)
    offsets = np.random.default_rng(0).integers(30, 117, len(dumps))
    false_since = since_marks(int(c + o) for c, o in zip(dumps, offsets, strict=True))
    near_gap = np.zeros(R.CENSUS_CADENCES, bool)
    for s, n in runs:
        near_gap[max(s - 30, 0) : s + n + 30] = True
    np.savez(
        out / "clock.npz", time=time, quarter=quarter, since=since, near_gap=near_gap
    )
    valid, low = stack(flux)
    rate = low / np.maximum(valid, 1)
    populated = valid > 0.25 * rows
    median = float(np.median(rate[populated]))
    active = populated & (rate >= ACTIVE_FACTOR * median)
    centres = time[np.arange(len(active)) * BIN + BIN // 2]
    comb = active & (centres >= COMB_BKJD[0]) & (centres <= COMB_BKJD[1])
    epochs = field_artifacts(
        centres, config_file()["photometry"]["field_artifact_epochs"]
    )
    shared = {
        "census": str(census), "rows": rows, "quarter": quarter, "since": since,
        "false_since": false_since, "near_gap": near_gap, "field_bins": active & ~comb,
        "comb_bins": comb, "gaps": gaps, "keep_share": np.clip(1 - median / np.maximum(rate, 1e-9), 0, 1),
        "unmasked_active": active & ~epochs, "quiet": populated & ~active,
    }  # fmt: skip
    with open(out / "shared.pkl", "wb") as stream:
        pickle.dump(shared, stream)
    R.single_threaded()
    results = []
    with ProcessPoolExecutor(
        workers, initializer=load_shared, initargs=(str(out / "shared.pkl"),)
    ) as pool:
        for i, result in enumerate(pool.map(measure_donor, range(rows), chunksize=64)):
            results.append(result)
            if i % 4000 == 0:
                print(f"{i}/{rows}", flush=True)
    (out / "shared.pkl").unlink()
    levels = pd.read_parquet(census / "levels.parquet")
    save_models(out, feats, results, levels)
    excess = {}
    for name, key in (("rolling_band", "band_counts"), ("gap_ramps", "hook_counts")):
        c = np.sum([r[key] for r in results if key in r], axis=0)
        flagged, null = c[1] / max(c[0], 1), c[3] / max(c[2], 1)
        excess[name] = {
            "evaluated": int(c[0]), "significant_share": float(flagged),
            "null_evaluated": int(c[2]), "null_significant_share": float(null),
            "keep_share": float(max(0.0, 1 - null / flagged)) if flagged > 0 else 0.0,
        }  # fmt: skip
    mission = {
        "census": str(census),
        "sources": rows,
        "dumps": dumps,
        "gaps": [list(g) for g in gaps],
        "median_dip_rate": median,
        "active_factor": ACTIVE_FACTOR,
        "field_bins": np.flatnonzero(active & ~comb).tolist(),
        "comb_bins": np.flatnonzero(comb).tolist(),
        "epoch_masked_bins": np.flatnonzero(epochs).tolist(),
        "bin_cadences": BIN,
        "excess": excess,
    }
    (out / "mission.json").write_text(json.dumps(mission))
    np.savez(out / "stack.npz", valid=valid, low=low)
    return mission


def save_models(out, feats, results, levels):
    """donors.parquet (one row per census curve: point noise, level, G and what each class holds for it)."""
    donors, field, comb, dumps, band, hooks, steps = [], [], [], [], [], [], []
    level = levels.groupby("row")["level"].median()
    observed = (
        levels[levels.level.notna()]
        .groupby("row")
        .quarter.agg(lambda q: int(sum(1 << int(x) for x in set(q))))
    )
    for r in results:
        row = r["row"]
        entry = {"row": row, "source_id": feats.source_id.iloc[row], "g_mag": feats.g_mag.iloc[row],
                 "point": r.get("point", np.nan), "bin_noise": r.get("bin_noise", np.nan),
                 "level": float(level.get(row, np.nan)), "quarters": int(observed.get(row, 0)),
                 "unmasked_active_bins": r.get("unmasked_active_bins", 0),
                 "unmasked_active_dips": r.get("unmasked_active_dips", 0),
                 "quiet_bins": r.get("quiet_bins", 0), "quiet_dips": r.get("quiet_dips", 0)}  # fmt: skip
        if "field" in r:
            for name, store in (("field", field), ("comb", comb)):
                cad, val = r[name]
                entry[f"{name}_cadences"] = len(cad)
                store += [
                    (row, c, v) for c, v in zip(cad.tolist(), val.tolist(), strict=True)
                ]
            entry["dump_quarters"] = len(r["dumps"])
            dumps += [(row, q, *p.tolist()) for q, p in r["dumps"]]
            entry["band_runs"] = len(r["band"])
            band += [(row, *b) for b in r["band"]]
            entry["hooks"] = len(r["hooks"])
            hooks += [(row, *h) for h in r["hooks"]]
            entry["steps"] = len(r["steps"])
            steps += [(row, s) for s in r["steps"]]
        donors.append(entry)
    pd.DataFrame(donors).to_parquet(out / "donors.parquet")
    pd.DataFrame(field, columns=["row", "cadence", "value"]).to_parquet(
        out / "field_excursions.parquet"
    )
    pd.DataFrame(comb, columns=["row", "cadence", "value"]).to_parquet(
        out / "q2_comb.parquet"
    )
    bins = [f"b{i}" for i in range(len(DUMP_EDGES) - 1)]
    pd.DataFrame(dumps, columns=["row", "quarter", *bins]).to_parquet(
        out / "momentum_dumps.parquet"
    )
    pd.DataFrame(band, columns=["row", "start", "length", "value"]).to_parquet(
        out / "rolling_band.parquet"
    )
    pd.DataFrame(
        hooks, columns=["row", "edge", "side", "amplitude", "kappa"]
    ).to_parquet(out / "gap_ramps.parquet")
    pd.DataFrame(steps, columns=["row", "step"]).to_parquet(
        out / "sensitivity_drops.parquet"
    )


@lru_cache(maxsize=1)
def models(root=str(MODELS)):
    root = Path(root)
    loaded = {
        "mission": json.loads((root / "mission.json").read_text()),
        "clock": dict(np.load(root / "clock.npz")),
        "donors": pd.read_parquet(root / "donors.parquet"),
    }
    for name in CLASSES:
        table = pd.read_parquet(root / f"{name}.parquet").sort_values(
            "row", kind="stable"
        )
        loaded[name] = table.reset_index(drop=True)
        loaded[f"{name}_rows"] = loaded[name].row.to_numpy()
    return loaded


def of_donor(name, donor, m):
    """Rows of class `name`'s table that belong to `donor`."""
    rows = m[f"{name}_rows"]
    return m[name].iloc[
        np.searchsorted(rows, donor) : np.searchsorted(rows, donor, "right")
    ]


def donor_pool(name, m):
    """Census rows holding an instance of class `name`."""
    if name == "sensitivity_drops":
        return np.sort(m["donors"].row[m["donors"].point.notna()].to_numpy())
    return np.unique(m[f"{name}_rows"])


def candidate_donors(name, noise, quarters, rng, m):
    """Donors of class `name` in the order they are tried."""
    pool = donor_pool(name, m)
    if name == "sensitivity_drops":
        return [int(rng.choice(pool))]
    key = "bin_noise" if name in ("field_excursions", "q2_comb") else "point"
    donors = m["donors"].set_index("row").reindex(pool)
    near = np.argsort(np.abs(np.log(donors[key].to_numpy() / noise[key])))[
        :NOISE_MATCHED
    ]
    mask = sum(1 << int(q) for q in quarters)
    shared = np.array(
        [bin(int(x) & mask).count("1") for x in donors.quarters.to_numpy()[near]]
    )
    order = np.lexsort((rng.random(len(near)), -shared))
    return [int(d) for d in pool[near][order]]


def positions(cadence):
    """Index of each mission cadence in a curve's arrays (-1 where absent)."""
    inside = (cadence >= 0) & (cadence < R.CENSUS_CADENCES)
    where = np.full(R.CENSUS_CADENCES, -1)
    where[cadence[inside]] = np.flatnonzero(inside)
    return where


def draw_events(name, donor, arrays, rng, m):
    """The instances a copy receives, with every parameter needed to replay them."""
    if name in ("field_excursions", "q2_comb", "momentum_dumps"):
        return []
    if name in ("rolling_band", "gap_ramps"):
        keep = m["mission"]["excess"][name]["keep_share"]
        table = of_donor(name, donor, m)
        for _ in range(50):
            kept = table[rng.random(len(table)) < keep]
            if len(kept):
                break
        return kept.drop(columns="row").to_dict("records")
    steps = m[name].step.to_numpy()
    cadence = arrays["cadence"].astype(int)
    near = m["clock"]["near_gap"][np.clip(cadence, 0, R.CENSUS_CADENCES - 1)]
    good = np.flatnonzero(arrays["mask"] & ~near)
    events = []
    for _ in range(DROPS_PER_COPY if len(good) else 0):
        events.append({
            "t0_bkjd": float(arrays["time"][rng.choice(good)]), "size": float(-rng.choice(steps)),
            "recovery": float(rng.uniform(*DROP_RECOVERY)),
            "tau_days": float(math.exp(rng.uniform(*np.log(DROP_TAU_DAYS)))),
        })  # fmt: skip
    return events


def apply(name, donor, events, arrays, m):
    """(change relative to the delivered level, raw multiplier) of a class's instances on a prepared curve."""
    cadence = arrays["cadence"].astype(int)
    n = len(cadence)
    flat, rawm = np.zeros(n), np.ones(n)
    where = positions(cadence)
    if name in ("field_excursions", "q2_comb"):
        table = of_donor(name, donor, m)
        at = where[table.cadence.to_numpy()]
        flat[at[at >= 0]] = table.value.to_numpy()[at >= 0]
    elif name == "momentum_dumps":
        since = m["clock"]["since"][np.clip(cadence, 0, R.CENSUS_CADENCES - 1)]
        centres = 0.5 * (DUMP_EDGES[1:] + DUMP_EDGES[:-1])
        for profile in of_donor(name, donor, m).itertuples(index=False):
            sel = (arrays["quarter"] == profile.quarter) & (since > 0)
            flat[sel] = np.interp(since[sel], centres, np.array(profile[2:], float))
    elif name == "rolling_band":
        for run in events:
            span = np.arange(run["start"], run["start"] + run["length"])
            at = where[span[span < R.CENSUS_CADENCES]]
            flat[at[at >= 0]] += run["value"]
    elif name == "gap_ramps":
        k = np.arange(HOOK_CADENCES)
        for hook in events:
            span = hook["edge"] + (k if hook["side"] == "after" else -k)
            ok = (span >= 0) & (span < R.CENSUS_CADENCES)
            at = where[span[ok]]
            flat[at[at >= 0]] += (hook["amplitude"] * np.exp(-k[ok] / hook["kappa"]))[
                at >= 0
            ]
    elif name == "sensitivity_drops":
        time = arrays["time"]
        for drop in events:
            q = arrays["quarter"][np.argmin(np.abs(time - drop["t0_bkjd"]))]
            sel = (time >= drop["t0_bkjd"]) & (arrays["quarter"] == q)
            dt = time[sel] - drop["t0_bkjd"]
            loss = (1 - drop["recovery"]) + drop["recovery"] * np.exp(
                -dt / drop["tau_days"]
            )
            rawm[sel] *= 1 - drop["size"] * loss
    return np.where(arrays["mask"], flat, 0.0), rawm


def artifact(name, arrays, noise, rng, m=None, tries=50):
    """One class's artifact for a prepared curve."""
    m = m or models()
    quarters = np.unique(arrays["quarter"][arrays["mask"]])
    for donor in candidate_donors(name, noise, quarters, rng, m)[:tries]:
        events = draw_events(name, donor, arrays, rng, m)
        flat, rawm = apply(name, donor, events, arrays, m)
        if np.any(flat != 0) or np.any(rawm != 1):
            break
    else:
        return None
    source = m["donors"].set_index("row").source_id.get(donor)
    truth = {"class": name, "donor_row": donor, "donor_source_id": str(source),
             "events": events, "changed_cadences": int((flat != 0).sum() + (rawm != 1).sum())}  # fmt: skip
    return flat, rawm, truth


def replay(truth, arrays, m=None):
    """Rebuild an artifact from its truth record (audit)."""
    return apply(
        truth["class"], truth["donor_row"], truth["events"], arrays, m or models()
    )


def unmasked_excess(donors):
    """3-sigma dips in the active bins the production mask leaves."""
    own = donors.quiet_dips / donors.quiet_bins.clip(lower=1)
    return donors.unmasked_active_dips - own * donors.unmasked_active_bins


def lane_rates(m=None):
    """Share of lane curves a copy of each class stands for."""
    m = m or models()
    donors = m["donors"][m["donors"].point.notna()]
    rates = {}
    for name in CLASSES[:3]:
        rates[name] = float(len(donor_pool(name, m)) / len(donors))
    for name, column in (("rolling_band", "band_runs"), ("gap_ramps", "hooks")):
        keep = m["mission"]["excess"][name]["keep_share"]
        counts = donors[column].fillna(0).to_numpy()
        rates[name] = float(np.mean(1 - (1 - keep) ** counts))
    rates["sensitivity_drops"] = float((donors.steps > 0).mean())
    return rates


def dominant(time, flux):
    centres, half = R.binned(time, flux, 0.5)
    return R.dominant_period(centres, half) if len(half) >= 30 else np.nan


def edge_pairs(row, crow, point, gaps):
    """(original, copy) z of the cadences beside every gap edge."""
    edges = {}
    for start, length in gaps:
        for o in (0, 1, 2, 6):
            for c, key in (
                (start + length + o, f"after+{o}"),
                (start - 1 - o, f"before-{o}"),
            ):
                if 0 <= c < R.CENSUS_CADENCES and np.isfinite(row[c]):
                    pair = ((row[c] - 1) / point, (row[c] - 1 + crow[c]) / point)
                    edges.setdefault(key, []).append(pair)
    return edges


def copy_record(name, change, truth, curve, m):
    """What the census measures of one class's change on one curve."""
    arrays, row, means, noise = (
        curve["arrays"],
        curve["row"],
        curve["means"],
        curve["noise"],
    )
    mask, cadence, point = arrays["mask"], curve["cadence"], noise["point"]
    crow = np.zeros(R.CENSUS_CADENCES)
    crow[cadence[mask]] = change[mask]
    rec = {
        "changed": int((change != 0).sum()),
        "max_abs_z": float(np.abs(change).max() / point),
    }
    if name in ("field_excursions", "q2_comb"):
        nb = R.CENSUS_CADENCES // BIN
        bins = np.array(
            m["mission"]["field_bins" if name == "field_excursions" else "comb_bins"]
        )
        covered = bins[np.isfinite(means[bins])]
        zc = crow[: nb * BIN].reshape(nb, BIN).mean(1)[covered] / noise["bin_noise"]
        rec |= {
            "covered": covered.astype(np.int32),
            "dips": int((zc < -3).sum()),
            "rises": int((zc > 3).sum()),
        }
    elif name == "momentum_dumps":
        rec["period_copy"] = dominant(
            arrays["time"][mask], arrays["flux"][mask] + change[mask]
        )
    elif name == "gap_ramps":
        rec["edges"] = edge_pairs(row, crow, point, m["mission"]["gaps"])
    elif name == "sensitivity_drops":
        time = arrays["time"]
        rec["drops"] = [
            (e["size"] / point, float(change[(time >= e["t0_bkjd"]) & (time < e["t0_bkjd"] + 3) & mask].min(initial=0) / point))
            for e in truth["events"]
        ]  # fmt: skip
    return rec


def validate_curve(job):
    """Every class injected into one census curve, re-prepared from LC2."""
    source_id, seed = job
    arrays, raw, _ = R.prepare_lc2(R.read_lc2(source_id))
    mask = arrays["mask"]
    if mask.sum() < 2000:
        return None
    m = models()
    cadence = arrays["cadence"].astype(int)
    row = np.full(R.CENSUS_CADENCES, np.nan)
    row[cadence[mask]] = arrays["flux"][mask]
    means, sd_bin = bin_noise(row)
    point = robust_sd(np.diff(arrays["flux"][mask])) / math.sqrt(2)
    curve = {"arrays": arrays, "row": row, "means": means, "cadence": cadence,
             "noise": {"point": point, "bin_noise": sd_bin}}  # fmt: skip
    rng = np.random.default_rng(seed)
    valid = np.isfinite(arrays["time"]) & np.isfinite(raw) & np.isfinite(arrays["flux"])
    out = {
        "source_id": source_id,
        "period_original": dominant(arrays["time"][mask], arrays["flux"][mask]),
    }
    for name in CLASSES:
        made = artifact(name, arrays, curve["noise"], rng, m)
        if made is None:
            out[name] = None
            continue
        flat, rawm, truth = made
        change = (
            flattened(raw, arrays["time"], arrays["quarter"], valid, rawm) - 1 + flat
        )
        out[name] = copy_record(name, np.where(mask, change, 0.0), truth, curve, m)
    return out


def validate(curves=600, workers=8, root=MODELS):
    """validation.json: each class injected into random census curves, compared with the census."""
    root = Path(root)
    m = models(str(root))
    census = Path(m["mission"]["census"])
    feats = pd.read_parquet(census / "features.parquet")
    pick = feats[feats.usable_cadences > 5000].sample(curves, random_state=5)
    seeds = np.random.default_rng(11).integers(2**31, size=len(pick))
    R.single_threaded()
    with ProcessPoolExecutor(workers, initializer=R.single_threaded) as pool:
        results = [
            r
            for r in pool.map(
                validate_curve, zip(pick.source_id, seeds.tolist()), chunksize=4
            )
            if r
        ]
    st = np.load(root / "stack.npz")
    rate = st["low"] / np.maximum(st["valid"], 1)
    median = m["mission"]["median_dip_rate"]
    report = {
        "curves": len(results),
        "lane_rates": lane_rates(m),
        "excess": m["mission"]["excess"],
    }

    def landed(name):
        return [r[name] for r in results if r[name] is not None]

    report["copies_landed"] = {name: len(landed(name)) for name in CLASSES}
    for name in ("field_excursions", "q2_comb"):
        exposure = np.zeros(len(rate))
        for r in landed(name):
            exposure[r["covered"]] += 1
        report[name] = {
            "injected_3sigma_dip_bins": int(sum(r["dips"] for r in landed(name))),
            "census_excess_dip_bins_expected": round(float(np.sum(exposure * np.clip(rate - median, 0, None))), 1),
            "census_excess_per_hit_curve_expected": round(
                float(np.sum(exposure * np.clip(rate - median, 0, None)) / report["lane_rates"][name]), 1
            ),
            "injected_3sigma_rise_bins": int(sum(r["rises"] for r in landed(name))),
        }  # fmt: skip

    def share(periods):
        p = np.asarray(periods, float)
        return round(float(np.nanmean((p >= 2.8) & (p <= 3.0))), 3)

    report["momentum_dumps"] = {
        "dominant_period_2p8_3p0_original": share(
            [r["period_original"] for r in results if r["momentum_dumps"] is not None]
        ),
        "dominant_period_2p8_3p0_copy": share([r["period_copy"] for r in landed("momentum_dumps")]),
        "dominant_period_2p8_3p0_donors": share(feats.variability_period.to_numpy()[donor_pool("momentum_dumps", m)]),
        "dominant_period_2p8_3p0_census": share(feats.variability_period.to_numpy()),
    }  # fmt: skip
    for name in ("rolling_band", "gap_ramps", "sensitivity_drops"):
        z = np.array([r["max_abs_z"] for r in landed(name)])
        report[name] = {
            "max_abs_change_over_point_quantiles_25_50_75_90_99": np.quantile(
                z, [0.25, 0.5, 0.75, 0.9, 0.99]
            )
            .round(2)
            .tolist()
        }
    edges = {}
    for r in landed("gap_ramps"):
        for key, pairs in r["edges"].items():
            edges.setdefault(key, []).extend(pairs)
    report["gap_ramps"]["edge_z_quantiles_1_5_95_99"] = {
        key: {"original": np.quantile([p[0] for p in v], [0.01, 0.05, 0.95, 0.99]).round(2).tolist(),
              "copy": np.quantile([p[1] for p in v], [0.01, 0.05, 0.95, 0.99]).round(2).tolist()}
        for key, v in sorted(edges.items())
    }  # fmt: skip
    drops = np.array([d for r in landed("sensitivity_drops") for d in r["drops"]])
    report["sensitivity_drops"][
        "size_over_point_and_deepest_flat_z_quantiles_10_50_90"
    ] = np.quantile(drops, [0.1, 0.5, 0.9], axis=0).round(2).tolist()
    (root / "validation.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("census", type=Path)
    m.add_argument("out", type=Path, nargs="?", default=MODELS)
    m.add_argument("--workers", type=int, default=8)
    v = sub.add_parser("validate")
    v.add_argument("--curves", type=int, default=600)
    v.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "measure":
        mission = measure(args.census, args.out, args.workers)
        print(
            json.dumps(
                {k: (len(v) if isinstance(v, list) else v) for k, v in mission.items()},
                indent=1,
            )
        )
        print(json.dumps(lane_rates(models(str(args.out))), indent=1))
        return
    print(json.dumps(validate(args.curves, args.workers), indent=1))


if __name__ == "__main__":
    main()
