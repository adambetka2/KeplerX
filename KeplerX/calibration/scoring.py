"""Score a pipeline run on the KBonus-realistic calibration sample."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.lane_census import single_threaded
from calibration.vetting_policy import PLANETS
from common import config_file, phase, require, write
from detector.upstream_adapter import DURATION_WINDOW, MIN_TEMPLATE_CADENCES
from runner.outputs import RunOutputs
from vetter.policy import POLICIES, assess

EXPOSURE = 1765.464 / 86400
# Candidate period = p/q of the true period.
HARMONICS = ((1, 4), (1, 3), (1, 2), (2, 3), (3, 2), (2, 1), (3, 1), (4, 1))
SYNTHETIC_PLANETS = ("synthetic_planet", "hz_planet")
BINARIES = (
    "eclipsing_binary",
    "blended_eclipsing_binary",
    "contamination_eclipsing_binary",
)
STRENGTH = {"exact": 3, "harmonic": 2, "secondary": 1, "secondary_harmonic": 1}
# Efficiency targets by information S/N (DR25: about 50% at expected MES 8.5-9).
EFFICIENCY_TARGETS = {8.5: 0.5, 12.0: 0.9}
RATIO_MIN_SNR = 12.5


def period_edges(*inner):
    """Period bin edges from 0.5 days to the search's top."""
    top = config_file()["detection"]["period_max_days"]
    edges = [0.5, *inner] + ([200] if top > 200 else [])
    return edges + [top + 0.001]


def epochs(t0, period, start, baseline):
    k = np.arange(
        math.ceil((start - t0) / period),
        math.floor((start + baseline - t0) / period) + 1,
    )
    return t0 + k * period


def match(candidate, signal, start, baseline):
    """'exact', 'harmonic' (with its ratio) or 'miss'."""
    P, t0 = signal["period_days"], signal["epoch_bkjd"]
    ttv = (
        signal["ttv"].get("amplitude_days", 0.0)
        if isinstance(signal.get("ttv"), dict)
        else 0.0
    )
    tol = max(signal["duration_days"] / 2, 1.5 * EXPOSURE) + ttv
    true = epochs(t0, P, start, baseline)
    if not len(true):
        return "miss", None
    cP, ct0 = candidate["period"], candidate["t0"]
    if abs(cP / P - 1) <= 0.02 and np.max(np.abs(phase(true, cP, ct0))) <= tol:
        return "exact", 1.0
    for p, q in HARMONICS:
        if abs(cP / P - p / q) > 0.02:
            continue
        for offset in range(q):
            own = epochs(ct0 + offset * cP, q * cP, start, baseline)
            if len(own) and np.max(np.abs(phase(own, P, t0))) <= tol:
                return "harmonic", round(p / q, 4)
    return "miss", None


def secondary_epoch(signal):
    """Mid-time of a binary's secondary eclipse."""
    if signal["kind"] == "contamination_eclipsing_binary" or "e" not in signal:
        return signal["epoch_bkjd"] + signal["period_days"] / 2
    from calibration.signal_injection import secondary_time

    return secondary_time(signal, signal["epoch_bkjd"])


def match_signal(candidate, signal, start, baseline):
    """match(), and for binaries the secondary eclipse's clock as well."""
    outcome, ratio = match(candidate, signal, start, baseline)
    if outcome == "miss" and signal["kind"] in BINARIES:
        shifted = signal | {"epoch_bkjd": secondary_epoch(signal)}
        outcome, ratio = match(candidate, shifted, start, baseline)
        outcome = {"exact": "secondary", "harmonic": "secondary_harmonic"}.get(
            outcome, outcome
        )
    return outcome, ratio


def window_keep(signal):
    """MES kept by the best template the detector admits at this period."""
    if signal.get("reference_duration_days") is None:
        return 1.0
    low, high = DURATION_WINDOW
    shortest = MIN_TEMPLATE_CADENCES * EXPOSURE
    ref, P, T = (
        signal["reference_duration_days"],
        signal["period_days"],
        signal["duration_days"],
    )
    floor, top = max(shortest, low * ref), min(high * ref, 1.5, P / 3)
    if top < floor:
        return 0.0
    if T > top:
        return math.sqrt(top / T)
    if T < floor:
        return math.sqrt(T / floor)
    return 1.0


def score_case(job):
    case, run, detection = job
    outputs = RunOutputs(run)
    out = {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "state": "MISSING",
        "signals": [],
        "candidates": [],
        "paired_case": None,
    }
    detections = outputs.read_json(case["case_id"], "detections/detections.json")
    if detections is None:
        return out
    vetting = outputs.read_json(case["case_id"], "vetting/vetting.json")
    vetted = {row["signal_id"]: row for row in vetting["results"]} if vetting else {}
    out["state"] = "COMPLETE" if vetting else "DETECTED_ONLY"
    truth = json.loads(Path(case["truth"]).read_text())
    out["paired_case"] = (truth.get("artifact") or {}).get("paired_case")
    with np.load(Path(case["prepared"]) / "photometry.npz") as archive:
        time = archive["time"][archive["mask"]]
    start, baseline = float(time.min()), float(np.ptp(time))
    candidates = detections["candidates"]
    labels = {c["signal_id"]: ("unexplained", None, None) for c in candidates}
    density = (truth["host"].get("catalogue_stellar") or {}).get("density_solar")
    for index, signal in enumerate(truth.get("signals", [])):
        if (
            signal.get("reference_duration_days") is None
            and density
            and signal["kind"] == "planet"
        ):
            signal["reference_duration_days"] = (
                13 / 24 * (signal["period_days"] / 365) ** (1 / 3) * density ** (-1 / 3)
            )
        best, ratio, matched, mes = "miss", None, None, None
        for c in candidates:
            outcome, r = match_signal(c, signal, start, baseline)
            if outcome == "miss":
                continue
            if STRENGTH[outcome] > STRENGTH.get(best, 0):
                best, ratio, matched, mes = outcome, r, c["signal_id"], c["mes"]
            if STRENGTH[outcome] > STRENGTH.get(labels[c["signal_id"]][1], 0):
                labels[c["signal_id"]] = (signal["kind"], outcome, index)
        out["signals"].append(signal_row(case, index, signal, detection) | {
            "outcome": best, "harmonic_ratio": ratio, "matched_signal_id": matched,
            "detected_mes": mes,
        })  # fmt: skip
    for c in candidates:
        out["candidates"].append(
            candidate_row(
                case, c, labels[c["signal_id"]], truth, vetted.get(c["signal_id"], {})
            )
        )
    return out


def signal_row(case, index, signal, detection):
    keep = window_keep(signal)
    snr = signal.get("information_snr")
    inside = (
        detection["period_min_days"]
        <= signal["period_days"]
        <= detection["period_max_days"]
    )
    enough = (signal.get("observed_events") or 0) >= detection["min_observed_transits"]
    return {
        "case_id": case["case_id"], "kind": case["kind"], "signal": index, "signal_kind": signal["kind"],
        "role": signal.get("role"), "period_days": signal["period_days"], "duration_days": signal["duration_days"],
        "radius_earth": signal.get("radius_earth"), "observed_events": signal.get("observed_events"),
        "information_snr": snr, "template_keep": keep, "flat_depth_ppm": signal.get("flat_depth_ppm"),
        # Companions carry no weight: a primary's weight counts all its star's planets.
        "population_weight": signal.get("population_weight") if signal.get("role") != "companion" else None,
        "e": signal.get("e"), "grazing": signal.get("grazing"),
        "insolation_earth": signal.get("insolation_earth"),
        "in_conservative_hz": bool(signal.get("in_conservative_hz")),
        "ttv": bool(signal.get("ttv")), "inside_range": inside, "enough_transits": enough,
        "detectable": bool(inside and enough and snr is not None and snr >= detection["mes_threshold"]),
    }  # fmt: skip


def candidate_row(case, candidate, label, truth, vetted):
    """One candidate with what it really is and every vetting metric."""
    kind, outcome, index = label
    if kind == "unexplained":
        for known in truth.get("known_signals", []):
            ratio = candidate["period"] / known["period_days"]
            for p, q in ((1, 1), *HARMONICS):
                if abs(ratio - p / q) < 0.002 * p / q:
                    kind, outcome = (
                        "known_signal",
                        "period" if p == q else f"period_{p}/{q}",
                    )
                    break
    metrics = {
        f"m_{k}": v
        for k, v in vetted.get("metrics", {}).items()
        if isinstance(v, (int, float)) or v is None
    }
    return {
        "case_id": case["case_id"], "kind": case["kind"], "signal_id": candidate["signal_id"],
        "period": candidate["period"], "mes": candidate["mes"], "tdur": candidate["tdur"], "label": kind,
        "match": outcome, "signal_index": index, "lane": truth.get("lane", "psf_flat"),
        "metric_errors": ",".join(e["component"] for e in vetted.get("metric_errors", [])), **metrics,
    }  # fmt: skip


def rate(frame, column="outcome", weights=None):
    """Shares found exactly, and found at all (exact, harmonic or, for a binary, its secondary eclipse)."""
    if not len(frame):
        return {"n": 0}
    exact = frame[column] == "exact"
    found = frame[column] != "miss"
    result = {
        "n": int(len(frame)),
        "exact": round(float(exact.mean()), 4),
        "found": round(float(found.mean()), 4),
    }
    if weights is not None:
        w = frame[weights].astype(float).fillna(0).to_numpy()
        if w.sum() > 0:
            result["weighted_exact"] = round(float((w * exact).sum() / w.sum()), 4)
            result["weighted_found"] = round(float((w * found).sum() / w.sum()), 4)
    return result


def by(frame, column, bins=None, weights="population_weight"):
    groups = (
        pd.cut(frame[column], bins, right=False) if bins is not None else frame[column]
    )
    return {
        str(k): rate(g, weights=weights)
        for k, g in frame.groupby(groups, observed=True)
    }


def gamma_efficiency(snr, hit):
    """Maximum-likelihood fit of c x GammaCDF(S/N; a, scale b), the form of DR25's detection efficiency."""
    from scipy.optimize import minimize
    from scipy.stats import gamma

    x, y = np.asarray(snr, float), np.asarray(hit, float)

    def loss(theta):
        c, a, b = 1 / (1 + np.exp(-theta[0])), np.exp(theta[1]), np.exp(theta[2])
        p = np.clip(c * gamma.cdf(x, a, scale=b), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    fit = minimize(loss, [3.0, math.log(20.0), math.log(0.45)], method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})  # fmt: skip
    c, a, b = 1 / (1 + math.exp(-fit.x[0])), math.exp(fit.x[1]), math.exp(fit.x[2])

    def at(level):
        return float(gamma.ppf(level / c, a, scale=b)) if level < c else None

    return {
        "c": c, "a": a, "b": b, "snr_at_50pct": at(0.5), "snr_at_90pct": at(0.9),
        "efficiency_at": {str(k): float(c * gamma.cdf(k, a, scale=b)) for k in EFFICIENCY_TARGETS},
    }  # fmt: skip


def wilson(k, n):
    if not n:
        return None
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def bootstrap_families(frame, n_boot, seed=0):
    """Resampled copies of `frame`, whole TPF families at a time."""
    rng = np.random.default_rng(seed)
    groups = frame.groupby("family").indices
    keys = list(groups)
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        yield frame.iloc[np.concatenate([groups[keys[i]] for i in pick])]


def efficiency_bins(frame):
    """Exact detections by information S/N, with Wilson intervals."""
    edges = [0, 3, 5, 6, 7.1, 8.5, 10, 12, 15, 20, 30, 50, np.inf]
    bins = {}
    for k, g in frame.groupby(
        pd.cut(frame.information_snr, edges, right=False), observed=True
    ):
        n, e = len(g), int((g.outcome == "exact").sum())
        bins[str(k)] = {"n": n, "exact": round(e / n, 4), "interval": wilson(e, n)}
    return bins


def efficiency_curve(searchable, n_boot=100):
    """Exact-detection efficiency against the information S/N of every searchable planet, unweighted."""
    s = searchable[searchable.information_snr > 0]
    hit = (s.outcome == "exact").to_numpy()
    bins = efficiency_bins(s)
    fit = gamma_efficiency(s.information_snr, hit)
    draws = [
        gamma_efficiency(b.information_snr, b.outcome == "exact")
        for b in bootstrap_families(s, n_boot)
    ]
    for key in ("snr_at_50pct", "snr_at_90pct"):
        values = [d[key] for d in draws if d[key] is not None]
        fit[f"{key}_interval"] = (
            np.quantile(values, [0.025, 0.975]).round(3).tolist() if values else None
        )
        fit[f"{key}_reached_in"] = round(len(values) / max(len(draws), 1), 3)
    targets = {
        str(snr): {"target": goal, "fitted": fit["efficiency_at"][str(snr)],
                   "met": fit["efficiency_at"][str(snr)] >= goal}
        for snr, goal in EFFICIENCY_TARGETS.items()
    }  # fmt: skip
    return {"planets": int(len(s)), "bins": bins, "gamma_fit": fit, "targets": targets}


def completeness_design(frame, classes):
    x = np.column_stack(
        [np.ones(len(frame)), np.log(frame.information_snr), np.log(frame.period_days),
         np.log(frame.radius_earth)]
        + [(frame.population == c).to_numpy(float) for c in classes[1:]]
    )  # fmt: skip
    return x


def fit_completeness(frame, classes):
    """C = c * sigmoid(alpha + beta ln S/N + gamma_P ln P + gamma_R ln R + delta_class)."""
    from scipy.optimize import minimize

    x = completeness_design(frame, classes)
    y = (frame.outcome == "exact").to_numpy(float)

    def loss(theta):
        c = 1 / (1 + np.exp(-theta[0]))
        p = np.clip(c / (1 + np.exp(-(x @ theta[1:]))), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 1e-3 * np.sum(
            theta[2:] ** 2
        )

    start = np.zeros(x.shape[1] + 1)
    start[0], start[1], start[2] = 3.0, -8.0, 4.0
    return minimize(loss, start, method="BFGS").x


def predict_completeness(theta, frame, classes):
    x = completeness_design(frame, classes)
    return 1 / (1 + np.exp(-theta[0])) / (1 + np.exp(-(x @ theta[1:])))


def completeness_model(searchable, n_boot=100):
    """C(S/N, P, R, class) fitted unweighted to every searchable planet, then integrated against occurrence."""
    s = completeness_frame(searchable)
    classes = s.population.value_counts().index.tolist()
    theta = fit_completeness(s, classes)
    predicted = predict_completeness(theta, s, classes)
    s = s.assign(predicted=predicted)

    def integrated(frame, values):
        w = frame.population_weight.astype(float).fillna(0).to_numpy()
        return float((w * values).sum() / w.sum()) if w.sum() > 0 else None

    detectable = s.detectable.to_numpy()
    result = {
        "planets": int(len(s)), "classes": classes,
        "coefficients": dict(zip(["logit_c", "alpha", "ln_snr", "ln_period", "ln_radius"]
                                 + [f"class_{c}" for c in classes[1:]], theta.round(4).tolist())),
        "asymptote_c": float(1 / (1 + math.exp(-theta[0]))),
        "occurrence_integrated": {
            "detectable": integrated(s[detectable], predicted[detectable]),
            "searchable": integrated(s, predicted),
        },
    }  # fmt: skip
    boot = {"detectable": [], "searchable": []}
    for b in bootstrap_families(s, n_boot, seed=1):
        values = predict_completeness(fit_completeness(b, classes), b, classes)
        keep = b.detectable.to_numpy()
        boot["detectable"].append(integrated(b[keep], values[keep]))
        boot["searchable"].append(integrated(b, values))
    result["occurrence_integrated_interval"] = {
        k: np.quantile([v for v in vals if v is not None], [0.025, 0.975]).round(4).tolist()
        for k, vals in boot.items() if any(v is not None for v in vals)
    }  # fmt: skip
    w = s.population_weight[detectable].astype(float).fillna(0).to_numpy()
    exact = (s.outcome[detectable] == "exact").to_numpy()
    result["indicator_weighted_detectable"] = {
        "exact": float((w * exact).sum() / w.sum()) if w.sum() > 0 else None,
        "kish_effective_size": float(w.sum() ** 2 / (w**2).sum())
        if (w**2).sum() > 0
        else 0.0,
    }
    calibration = {}
    for name, groups in (
        (
            "snr",
            pd.cut(s.information_snr, [0, 5, 7.1, 8.5, 10, 12, 15, 20, 40, np.inf]),
        ),
        ("period_days", pd.cut(s.period_days, period_edges(1, 10, 50, 100))),
        ("radius_earth", pd.cut(s.radius_earth, [0, 2, 4, 8, 15, 30])),
        ("class", s.population),
        ("template", np.where(s.template_keep < 1, "mismatched", "matched")),
    ):
        calibration[name] = {
            str(k): {"n": int(len(g)), "observed": round(float((g.outcome == "exact").mean()), 4),
                     "model": round(float(g.predicted.mean()), 4)}
            for k, g in s.groupby(groups, observed=True)
        }  # fmt: skip
    result["calibration"] = calibration
    return result


def completeness_frame(searchable):
    return searchable[
        (searchable.information_snr > 0) & searchable.radius_earth.notna()
    ].assign(population=searchable.population.fillna("unknown"))


def habitable_zone(planets, searchable, candidates):
    """Synthetic planets in a conservative habitable zone."""
    from calibration.sample_builder import PLAN

    zone = planets[planets.in_conservative_hz]
    if not len(zone):
        return None
    inside = zone[zone.inside_range & zone.enough_transits]
    detectable = inside[inside.detectable]
    main = completeness_frame(searchable)
    classes = main.population.value_counts().index.tolist()
    theta = fit_completeness(main, classes)
    known = completeness_frame(inside)
    known = known[known.population.isin(classes)]
    known = known.assign(predicted=predict_completeness(theta, known, classes))
    edges = [0, 3, 5, 7.1, 10, 15, np.inf]
    panel = PLAN["hz_panel"]
    low, high = panel["radius_range_earth"]
    box = zone[
        zone.population.isin(panel["classes"])
        & zone.radius_earth.between(low, high)
        & (zone.role == "primary")
    ]
    covered = completeness_frame(box)
    covered = covered[covered.population.isin(classes)]
    found = (
        box.inside_range & box.enough_transits & box.index.isin(covered.index)
    ).to_numpy()
    values = np.zeros(len(box))
    values[found] = predict_completeness(theta, box[found], classes)
    w = box.population_weight.astype(float).fillna(0).to_numpy()
    keys = set(zip(zone.case_id, zone.signal, strict=True))
    matched = np.array(
        [(c, i) in keys for c, i in zip(candidates.case_id, candidates.signal_index)],
        bool,
    )
    detections = candidates[
        (candidates.label == "planet") & (candidates.match == "exact") & matched
    ]
    return {
        "planets": int(len(zone)),
        "by_draw": zone.kind.value_counts().to_dict(),
        "primaries": int((zone.role == "primary").sum()),
        "outside_range": int((~zone.inside_range).sum()),
        "fewer_than_min_transits": int((zone.inside_range & ~zone.enough_transits).sum()),
        "detectable": rate(detectable),
        "efficiency_bins": efficiency_bins(inside),
        "detectable_by_class": by(detectable, "population", weights=None),
        "detectable_by_radius": by(detectable, "radius_earth", [0, 1, 1.25, 1.5, 1.75, 2.0, 30], weights=None),
        "detectable_by_role": by(detectable, "role", weights=None),
        "detectable_by_draw": by(detectable, "kind", weights=None),
        "model_against_observed": {
            str(k): {"n": int(len(g)), "observed": round(float((g.outcome == "exact").mean()), 4),
                     "model": round(float(g.predicted.mean()), 4)}
            for k, g in known.groupby(pd.cut(known.information_snr, edges, right=False), observed=True)
        },
        "box_occurrence_integrated": {
            "classes": panel["classes"], "radius_earth": [low, high],
            "primaries": int(len(box)), "searchable": int(found.sum()),
            "completeness": float((w * values).sum() / w.sum()) if w.sum() > 0 else None,
            "kish_effective_size": float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0,
        },
        "vetting": {
            "exact_detections": int(len(detections)),
            "with_metric_errors": int((detections.metric_errors.fillna("") != "").sum()),
            "kept_by_policy": {
                name: round(float(flags.mean()), 4)
                for name, flags in policy_kept(detections).items()
            } if len(detections) else {},
        },
    }  # fmt: skip


def snr_ratio(planets):
    """Detector MES over information S/N for exact detections above RATIO_MIN_SNR."""
    d = planets[(planets.outcome == "exact") & (planets.information_snr >= RATIO_MIN_SNR)
                & planets.detected_mes.notna()]  # fmt: skip
    ratio = d.detected_mes / d.information_snr

    def quantiles(v):
        return (
            {
                "n": int(len(v)),
                "quantiles_25_50_75": np.quantile(v, [0.25, 0.5, 0.75])
                .round(3)
                .tolist(),
            }
            if len(v)
            else {"n": 0}
        )

    return {
        "all": quantiles(ratio),
        "by_period": {str(k): quantiles(v) for k, v in ratio.groupby(pd.cut(d.period_days, period_edges(1, 10, 50)), observed=True)},
        "by_duration_hours": {str(k): quantiles(v) for k, v in ratio.groupby(pd.cut(d.duration_days * 24, [0, 1, 2, 4, 8, 100]), observed=True)},
    }  # fmt: skip


def false_alarms(ids, candidates, cases):
    """Cases with an unexplained candidate, plain and weighted to the lane."""
    unexplained = candidates[candidates.label == "unexplained"]
    hit = ids.isin(unexplained.case_id)
    out = {
        "cases": int(len(ids)),
        "with_unexplained": round(float(hit.mean()), 4) if len(ids) else None,
    }
    weights = cases.set_index("case_id").reindex(ids).lane_weight.to_numpy(float)
    if len(ids) and np.isfinite(weights).all() and weights.sum() > 0:
        out["lane_weighted_with_unexplained"] = round(
            float((weights * hit.to_numpy()).sum() / weights.sum()), 4
        )
    return out


def artifact_pairs(signals, candidates, cases, pairs):
    """Per artifact class, what the copies add or lose against the cases they copy."""
    out = {}
    copies = cases[cases.kind.str.startswith("artifact_")].merge(pairs, on="case_id")
    for kind, group in copies.groupby("kind"):
        nulls = group[group.artifact_pair == "control_original"]
        planets = group[group.artifact_pair == "synthetic_planet"]
        copied = signals[signals.case_id.isin(planets.case_id) & signals.detectable]
        paired = signals[signals.case_id.isin(planets.paired_case) & signals.detectable]
        out[kind.removeprefix("artifact_")] = {
            "null_copies": false_alarms(nulls.case_id, candidates, cases),
            "their_originals": false_alarms(nulls.paired_case, candidates, cases),
            "planet_copies_detectable": rate(copied),
            "their_planet_cases_detectable": rate(paired),
        }
    return out


def policy_kept(candidates):
    """Each proposed policy's KEPT flag for every candidate row."""
    rows = candidates.to_dict("records")
    metrics = [{k[2:]: v for k, v in row.items() if k.startswith("m_")} for row in rows]
    lanes = [row.get("lane") or "psf_flat" for row in rows]
    return {
        name: np.array(
            [
                assess(m, name, lane)["state"] == "KEPT"
                for m, lane in zip(metrics, lanes, strict=True)
            ],
            bool,
        )
        for name in POLICIES
    }


def efficiency_after(searchable, candidates, kept, n_boot):
    """The efficiency curve of planets found exactly and kept by a policy."""
    keys = set(zip(candidates.case_id[kept], candidates.signal_id[kept]))
    survived = np.array(
        [(c, m) in keys for c, m in zip(searchable.case_id, searchable.matched_signal_id)],
        bool,
    )  # fmt: skip
    vetted = searchable.assign(
        outcome=np.where((searchable.outcome == "exact") & survived, "exact", "miss")
    )
    curve = efficiency_curve(vetted, n_boot)
    return {"gamma_fit": curve["gamma_fit"], "targets": curve["targets"]}


def vetting_policies(candidates, searchable=None, n_boot=100):
    """What each proposed vetting policy would keep, by label, and the efficiency of detection then vetting."""
    if not len(candidates):
        return {}
    planet = candidates.label.isin(PLANETS)
    out = {}
    for name, flags in policy_kept(candidates).items():
        kept = candidates.assign(kept=flags)
        nulls = kept[(kept.kind == "control_original") & (kept.label == "unexplained")]
        out[name] = {
            "planets_kept": round(float(kept.kept[planet].mean()), 4) if planet.any() else None,
            "kept_by_label": kept.groupby("label").kept.mean().round(4).to_dict(),
            "null_original_cases_with_unexplained": {
                "before": int(nulls.case_id.nunique()),
                "after": int(nulls[nulls.kept].case_id.nunique()),
            },
        }  # fmt: skip
        if searchable is not None and len(searchable):
            out[name]["efficiency_after_policy"] = efficiency_after(
                searchable, candidates, flags, n_boot
            )
    return out


def summarise(signals, candidates, cases, states, pairs, n_boot=100):
    hosts = cases[["case_id", "population", "g_mag", "family"]]
    planets = signals[
        (signals.signal_kind == "planet") & signals.kind.isin(SYNTHETIC_PLANETS)
    ].merge(hosts, on="case_id", how="left")
    detectable = planets[planets.detectable]
    controls = planets[~planets.inside_range | ~planets.enough_transits]
    searchable = planets[planets.inside_range & planets.enough_transits]
    below = searchable[~searchable.detectable]
    require(len(searchable) > 0, "No searchable synthetic planets in these runs")
    summary = {
        "cases": states,
        "efficiency_curve": efficiency_curve(searchable, n_boot),
        "completeness_model": completeness_model(searchable, n_boot),
        "detector_mes_over_information_snr": snr_ratio(planets),
        "detection_on_detectable_planets": rate(
            detectable, weights="population_weight"
        ),
        "detectable_by_snr": by(
            detectable, "information_snr", [7.1, 8.5, 10, 12, 15, 20, 40, 100, np.inf]
        ),
        "detectable_by_period": by(
            detectable, "period_days", period_edges(1, 10, 50, 100)
        ),
        "detectable_by_duration_hours": by(
            detectable.assign(h=detectable.duration_days * 24),
            "h",
            [0, 1, 2, 4, 6, 10, 16, 100],
        ),
        "detectable_by_template_match": by(
            detectable.assign(
                t=np.where(detectable.template_keep < 1, "mismatched", "matched")
            ),
            "t",
        ),
        "detectable_by_host": by(detectable, "population"),
        "detectable_by_g_mag": by(detectable, "g_mag", [0, 16, 17, 18, 18.5, 25]),
        "detectable_by_role": by(detectable, "role"),
        "below_threshold_planets": rate(below),
        "structural_controls": {
            "outside_range": rate(controls[~controls.inside_range]),
            "fewer_than_min_transits": rate(
                controls[controls.inside_range & ~controls.enough_transits]
            ),
            "note": "Outside the period range an exact detection points to a software fault; with too few "
            "transits it may also mean the detector counted partial transits that the 50%-coverage rule "
            "does not. Harmonics are expected.",
        },
        "binaries": {
            k: rate(g)
            for k, g in signals[
                signals.signal_kind.str.contains("eclipsing_binary")
                & ~signals.signal_kind.str.startswith("contamination")
            ].groupby("signal_kind")
        },
        "leaks": {
            k: rate(g)
            for k, g in signals[
                signals.signal_kind.str.startswith("contamination")
            ].groupby("signal_kind")
        },
        "real_planets": {
            "known_planet_in_range": rate(
                signals[
                    (signals.signal_kind == "known_planet")
                    & signals.inside_range
                    & signals.enough_transits
                ]
            ),
            "known_planet_detectable_on_kbonus_depth": rate(
                signals[(signals.signal_kind == "known_planet") & signals.detectable]
            ),
            "validated_planet": rate(
                signals[signals.signal_kind == "validated_planet"]
            ),
        },
        "artifact_pairs": artifact_pairs(signals, candidates, cases, pairs),
        "lane_false_alarms": {
            kind: false_alarms(group.case_id, candidates, cases)
            for kind, group in cases[cases.lane_weight.notna()].groupby("kind")
        },
    }
    nulls = {}
    for kind in (
        "control_original",
        "control_artifact",
        "control_inverted",
        "control_scrambled",
    ):
        ids = cases[cases.kind == kind].case_id
        cand = candidates[candidates.case_id.isin(ids)]
        unexplained = cand[cand.label == "unexplained"]
        nulls[kind] = {
            "cases": int(len(ids)),
            "cases_with_candidates": round(float(ids.isin(cand.case_id).mean()), 4),
            "candidates_per_case": round(len(cand) / max(len(ids), 1), 3),
            "unexplained_candidates": int(len(unexplained)),
            "unexplained_mes_quantiles": np.quantile(unexplained.mes, [0.1, 0.5, 0.9])
            .round(2)
            .tolist()
            if len(unexplained)
            else None,
        }
    summary["null_cases"] = nulls
    zone = habitable_zone(planets, searchable, candidates)
    if zone:
        summary["habitable_zone"] = zone
    summary["vetting_policies"] = vetting_policies(candidates, searchable, n_boot)
    summary["candidate_labels"] = (
        candidates.groupby(["kind", "label"]).size().unstack(fill_value=0).to_dict()
    )
    return summary


def case_runs(runs):
    """Which run each case is scored from: the run that completed it, else the last run that recorded it."""
    run_of, complete = {}, set()
    for run in runs:
        jobs = dict(RunOutputs(run).database.execute("SELECT case_id, state FROM jobs"))
        done = {case_id for case_id, state in jobs.items() if state == "COMPLETE"}
        require(not done & complete, f"{run} repeats completed cases of another run")
        complete |= done
        run_of |= {c: str(run) for c in jobs if c in done or c not in complete}
    return run_of


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()
    detection = config_file()["detection"]
    cases = pd.read_parquet(args.sample / "cases.parquet")
    run_of = case_runs(args.runs)
    chosen = cases[cases.case_id.isin(run_of)]
    jobs = [
        (row, run_of[row["case_id"]], detection) for row in chosen.to_dict("records")
    ]
    single_threaded()
    with ProcessPoolExecutor(args.workers, initializer=single_threaded) as pool:
        results = list(pool.map(score_case, jobs, chunksize=16))
    args.out.mkdir(parents=True, exist_ok=False)
    signals = pd.DataFrame([s for r in results for s in r["signals"]])
    candidates = pd.DataFrame([c for r in results for c in r["candidates"]])
    signals.to_parquet(args.out / "signals.parquet")
    candidates.to_parquet(args.out / "candidates.parquet")
    pairs = pd.DataFrame(
        [{"case_id": r["case_id"], "paired_case": r["paired_case"]} for r in results]
    )
    state = pd.Series({r["case_id"]: r["state"] for r in results})
    # Development tunes, holdout accepts: each partition is reported alone.
    partitions = {}
    for part, group in chosen.groupby("partition"):
        ids = set(group.case_id)
        partitions[part] = summarise(
            signals[signals.case_id.isin(ids)], candidates[candidates.case_id.isin(ids)],
            group, state[list(ids)].value_counts().to_dict(),
            pairs[pairs.case_id.isin(ids)], args.bootstrap,
        )  # fmt: skip
    settings = {
        k: detection.get(k)
        for k in (
            "period_min_days",
            "period_max_days",
            "min_observed_transits",
            "mes_threshold",
        )
    } | {
        "duration_window": list(DURATION_WINDOW),
        "min_template_cadences": MIN_TEMPLATE_CADENCES,
    }
    write(args.out / "summary.json", {"partitions": partitions, "settings": settings})
    report = {
        part: {
            "cases": s["cases"],
            "detectable_planets": s["detection_on_detectable_planets"],
            "efficiency": s["efficiency_curve"]["gamma_fit"],
            "occurrence_integrated": s["completeness_model"]["occurrence_integrated"],
        }
        for part, s in partitions.items()
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
