"""Paired pixel-refit validation with independent multivariate surrogate nulls."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.interpolate import BSpline
from scipy.stats import beta

from dwarfs.experiments.preparation import physical_loss

DESIGN = {
    "schema": 1,
    "seed": 2026101402,
    "calibration_nulls": 128,
    "held_nulls": 128,
    "block_days": [1.0, 2.0],
    "alpha": 0.05,
    "period_min_days": 9.75,
    "period_max_days": 10.25,
    "duration_hours": [2.0, 4.0, 6.0],
    "depth_ppm": [40.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
    "phase_repeats": 2,
    "spline_days": 1.5,
    "phase_oversample": 10,
    "period_step": "Dmin * Pmin / (5 * observed span)",
    "null": "Whole pixel vector and aperture background share day-block signs",
    "refit": "Fit the complete pixel extractor for every null and injection",
    "source_rate": "Median delivered corrected PSF flux, frozen before scoring",
    "search_error": "Same supplied aperture errors for both methods; null calibrated",
    "limits": [
        "Restricted period search, not production completeness",
        "Conditional block-wild surrogate nulls, not guaranteed astrophysical nulls",
        "Delivered PSF is an observed-noise comparator, not replayed extraction",
        "Source flux and injection profiles remain calibrated-model assumptions",
    ],
}
METHODS = ("delivered_sap", "pixel_reprocessed", "cbv_pointing_only")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def robust_sigma(values):
    values = np.asarray(values)
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def basis(time, error, pointing=None, cbv=None):
    knots = np.r_[
        np.repeat(time[0], 4),
        np.arange(time[0] + DESIGN["spline_days"], time[-1], DESIGN["spline_days"]),
        np.repeat(time[-1], 4),
    ]
    design = BSpline.design_matrix(time, knots, 3).toarray()
    if pointing is not None:
        centered = pointing - np.median(pointing, axis=0)
        design = np.column_stack((design, centered, centered**2))
    if cbv is not None:
        design = np.column_stack((design, cbv))
    weighted = design / error[:, None]
    u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
    return u[:, singular > singular[0] * 1e-10]


class CommonSearch:
    def __init__(self, time, error):
        self.time, self.error = np.asarray(time), np.asarray(error)
        self.q = basis(self.time, self.error)
        low, high = DESIGN["period_min_days"], DESIGN["period_max_days"]
        step = min(DESIGN["duration_hours"]) / 24 * low / (5 * np.ptp(time))
        self.periods = np.linspace(low, high, int(np.ceil((high - low) / step)) + 1)

    def residual(self, flux):
        weighted = np.asarray(flux) / self.error
        return (weighted - self.q @ (self.q.T @ weighted)) * self.error

    def search(self, flux):
        result = BoxLeastSquares(self.time, self.residual(flux), dy=self.error).power(
            self.periods,
            np.asarray(DESIGN["duration_hours"]) / 24,
            objective="snr",
            oversample=DESIGN["phase_oversample"],
        )
        index = int(np.argmax(result.power))
        return {
            "score": float(result.power[index]),
            "period_days": float(result.period[index]),
            "epoch_bkjd": float(result.transit_time[index]),
            "duration_hours": float(result.duration[index] * 24),
        }

    def transfer(self, delta, expected):
        signal = self.residual(expected) / self.error
        denominator = float(signal @ signal)
        if denominator <= 0:
            raise ValueError("Injection has no measurable template support")
        return float(signal @ (self.residual(delta) / self.error) / denominator)


def empirical_p(calibration, values):
    calibration, values = np.asarray(calibration), np.atleast_1d(values)
    return (1 + (calibration[:, None] >= values).sum(axis=0)) / (len(calibration) + 1)


def false_alarm_summary(calibration, held):
    pvalues = empirical_p(calibration, held)
    return rate_summary(pvalues)


def rate_summary(pvalues):
    failures, total = int(np.sum(pvalues <= DESIGN["alpha"])), len(pvalues)
    lower = beta.ppf(0.025, failures, total - failures + 1) if failures else 0.0
    upper = beta.ppf(0.975, failures + 1, total - failures) if failures < total else 1.0
    return {
        "held_pvalues": pvalues.tolist(),
        "held_false_alarms": failures,
        "held_count": total,
        "held_false_alarm_rate": failures / total,
        "conditional_binomial_95_interval": [float(lower), float(upper)],
    }


def joint_null_summary(calibrations, folder):
    result = {}
    for block in DESIGN["block_days"]:
        with np.load(folder / f"nulls-{block:g}d.npz") as stored:
            held = stored["held_scores"]
        result[str(block)] = {}
        for index, method in enumerate(METHODS):
            pvalues = np.max(
                [
                    empirical_p(scores[:, index], held[:, index])
                    for scores in calibrations.values()
                ],
                axis=0,
            )
            result[str(block)][method] = rate_summary(pvalues)
    return result


def wild_components(data, error):
    pixels = np.column_stack((data["pixel_flux"], data["background"]))
    q = basis(data["time"], error, data["pointing"], data.get("cbv"))
    trend = (q @ (q.T @ (pixels / error[:, None]))) * error[:, None]
    return trend, pixels - trend


def wild_draw(data, trend, residual, block_days, rng):
    blocks = np.floor((data["time"] - data["time"][0]) / block_days).astype(int)
    signs = rng.choice([-1.0, 1.0], int(blocks.max()) + 1)
    simulated = trend + residual * signs[blocks, None]
    return dict(
        data,
        pixel_flux=simulated[:, :-1],
        background=simulated[:, -1],
        null_signs=signs,
    )


def extractor_arguments(data):
    names = (
        "pixel_flux",
        "time",
        "pointing",
        "target_profiles",
        "aperture",
        "background",
        "pixel_error",
        "profile_derivatives",
    )
    arguments = {name: data[name] for name in names}
    if "cbv" in data:
        arguments["cbv"] = data["cbv"]
    return arguments


def extract_pair(data, extractor):
    result = extractor(**extractor_arguments(data))
    ablation = extractor(**extractor_arguments(data), reference_pixels=False)
    if not np.array_equal(result["baseline_flux"], ablation["baseline_flux"]):
        raise ValueError("Pixel and CBV-only arms have different baselines")
    values = np.column_stack(
        (result["baseline_flux"], result["flux"], ablation["flux"])
    )
    result["ablation"] = ablation
    if not np.isfinite(values).all():
        raise ValueError("Extractor returned nonfinite photometry")
    return values, result


def compact_fit(result_input):
    diagnostics = dict(result_input["diagnostics"])
    diagnostics["folds"] = [
        {key: value for key, value in fold.items() if key != "train_indices"}
        for fold in diagnostics["folds"]
    ]
    result = {
        "diagnostics": diagnostics,
        "array_hashes": {
            key: hashlib.sha256(np.asarray(result_input[key]).tobytes()).hexdigest()
            for key in ("flux", "prediction", "folds", "pixel_weights")
        },
    }
    if "ablation" in result_input:
        result["ablation"] = compact_fit(result_input["ablation"])
    return result


def null_block(data, search, extractor, folder, block_days, seed):
    trend, residual = wild_components(data, search.error)
    np.savez_compressed(
        folder / f"null-model-{block_days:g}d.npz", trend=trend, residual=residual
    )
    populations = {}
    for population, count in (
        ("calibration", DESIGN["calibration_nulls"]),
        ("held", DESIGN["held_nulls"]),
    ):
        offset = 0 if population == "calibration" else 1
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(block_days), offset])
        )
        curves, scores, signs, fits = [], [], [], []
        for draw in range(count):
            simulated = wild_draw(data, trend, residual, block_days, rng)
            pair, extraction = extract_pair(simulated, extractor)
            curves.append(pair.astype(np.float32))
            proposals = [search.search(pair[:, k]) for k in range(len(METHODS))]
            scores.append([proposal["score"] for proposal in proposals])
            signs.append(simulated["null_signs"])
            fits.append(
                {
                    "draw": draw,
                    "proposals": proposals,
                    "extraction": compact_fit(extraction),
                }
            )
        write(
            folder / f"null-fits-{block_days:g}d-{population}.json",
            {"seed_sequence": [seed, int(block_days), offset], "fits": fits},
        )
        populations[population] = {
            "curves": np.asarray(curves),
            "scores": np.asarray(scores),
            "signs": np.asarray(signs),
        }
    path = folder / f"nulls-{block_days:g}d.npz"
    np.savez_compressed(
        path,
        **{
            f"{population}_{field}": array
            for population, values in populations.items()
            for field, array in values.items()
        },
    )
    results = {
        method: false_alarm_summary(
            populations["calibration"]["scores"][:, index],
            populations["held"]["scores"][:, index],
        )
        for index, method in enumerate(METHODS)
    }
    write(
        folder / f"nulls-{block_days:g}d.json",
        {
            "block_days": block_days,
            "results": results,
            "arrays_sha256": digest(path),
            "covariance": "Ensemble permits full-refit temporal covariance estimation",
        },
    )
    return populations["calibration"]["scores"], results


def noise_at_timescale(time, residual, hours):
    bins = np.floor((time - time[0]) / (hours / 24)).astype(int)
    counts = np.bincount(bins)
    sums = np.bincount(bins, weights=residual)
    expected = hours / 24 / np.median(np.diff(time))
    selected = counts >= max(2, expected / 2)
    return {
        "sigma_e_per_s": robust_sigma(sums[selected] / counts[selected]),
        "bins": int(selected.sum()),
    }


def observed_noise(data, pair, search, source_rate):
    curves = {method: pair[:, k] for k, method in enumerate(METHODS)}
    curves["delivered_psf"] = data["delivered_psf"]
    report = {}
    for method, flux in curves.items():
        rate = (
            source_rate
            if method == "delivered_psf"
            else source_rate * np.median(data["target_profiles"] @ data["aperture"])
        )
        residual = search.residual(flux)
        report[method] = {}
        for hours in DESIGN["duration_hours"]:
            value = noise_at_timescale(data["time"], residual, hours)
            value["source_relative_sigma_ppm"] = value["sigma_e_per_s"] / rate * 1e6
            value["normalization_e_per_s"] = float(rate)
            value["delivered_relative_sigma_ppm"] = (
                value["sigma_e_per_s"] / float(np.median(flux)) * 1e6
            )
            report[method][str(hours)] = value
    return report


def truth_rows(data, rng):
    start = float(data["time"][0])
    for duration in DESIGN["duration_hours"]:
        for depth in DESIGN["depth_ppm"]:
            for repeat in range(DESIGN["phase_repeats"]):
                period = float(rng.uniform(9.8, 10.2))
                yield {
                    "period_days": period,
                    "epoch_bkjd": start + float(rng.uniform(0, period)),
                    "duration_hours": duration,
                    "depth_ppm": depth,
                    "repeat": repeat,
                }


def clock_match(time, proposal, truth, expected):
    period, epoch = proposal["period_days"], proposal["epoch_bkjd"]
    true_period, true_epoch = truth["period_days"], truth["epoch_bkjd"]
    phase = (time - epoch + period / 2) % period - period / 2
    window = np.abs(phase) < proposal["duration_hours"] / 48
    energy = float(np.sum(expected[window] ** 2) / np.sum(expected**2))
    drift = abs(period / true_period - 1) * np.ptp(time)
    offset = abs((epoch - true_epoch + true_period / 2) % true_period - true_period / 2)
    tolerance = truth["duration_hours"] / 48 + 0.02043361111111111
    centers = true_epoch + true_period * np.arange(
        np.ceil((time[0] - true_epoch) / true_period),
        np.floor((time[-1] - true_epoch) / true_period) + 1,
    )
    timing = np.abs(centers - epoch - np.rint((centers - epoch) / period) * period)
    return bool(
        drift <= tolerance
        and offset <= tolerance
        and energy >= 0.5
        and len(timing)
        and timing.max() <= tolerance
    ), energy


def trial_rows(data, pair, search, extractor, truth, profile, source_rate, nulls):
    loss = physical_loss(
        data["time"],
        truth["period_days"],
        truth["epoch_bkjd"],
        truth["duration_hours"],
        truth["depth_ppm"],
    )
    delta = -source_rate * loss[:, None] * profile
    simulated = dict(data, pixel_flux=data["pixel_flux"] + delta)
    injected, extraction = extract_pair(simulated, extractor)
    expected = delta @ data["aperture"]
    support = loss > 0
    events = np.rint((data["time"] - truth["epoch_bkjd"]) / truth["period_days"])
    rows = []
    for index, method in enumerate(METHODS):
        proposal = search.search(injected[:, index])
        pvalues = {
            str(block): float(empirical_p(scores[:, index], proposal["score"])[0])
            for block, scores in nulls.items()
        }
        matched, energy = clock_match(data["time"], proposal, truth, expected)
        base_proposal = search.search(pair[:, index])
        baseline_match, _ = clock_match(data["time"], base_proposal, truth, expected)
        baseline_p = max(
            float(empirical_p(scores[:, index], base_proposal["score"])[0])
            for scores in nulls.values()
        )
        baseline_recovered = bool(baseline_match and baseline_p <= DESIGN["alpha"])
        recovered = bool(
            matched
            and max(pvalues.values()) <= DESIGN["alpha"]
            and len(np.unique(events[support])) >= 3
        )
        rows.append(
            {
                "method": method,
                "truth": truth,
                "proposal": proposal,
                "empirical_p": pvalues,
                "known_clock_retention": search.transfer(
                    injected[:, index] - pair[:, index], expected
                ),
                "supported_events": int(len(np.unique(events[support]))),
                "energy_covered": energy,
                "baseline_clock_match": baseline_match,
                "baseline_recovered": baseline_recovered,
                "recovered": recovered,
                "incremental_recovery": recovered and not baseline_recovered,
            }
        )
    return rows, loss, compact_fit(extraction)


def injection_families(data):
    return {
        "nominal": data["target_profiles"],
        **{
            key.removeprefix("profile_family_"): value
            for key, value in data.items()
            if key.startswith("profile_family_")
        },
    }


def injections(data, pair, search, extractor, folder, source_rate, nulls, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    truths, families = list(truth_rows(data, rng)), injection_families(data)
    rows, losses, fits = [], [], []
    for number, truth in enumerate(truths):
        names = (
            list(families)
            if truth["duration_hours"] == 4 and truth["depth_ppm"] in (1000, 3000)
            else ["nominal"]
        )
        for name in names:
            result, loss, fitted = trial_rows(
                data, pair, search, extractor, truth, families[name], source_rate, nulls
            )
            rows.extend(
                dict(row, injection=number, profile_family=name) for row in result
            )
            fits.append(
                {"injection": number, "profile_family": name, "extraction": fitted}
            )
        losses.append(loss)
    write(folder / "injections.json", rows)
    write(folder / "injection-fits.json", fits)
    np.savez_compressed(folder / "injection-losses.npz", loss=np.asarray(losses))
    return rows


def recovery_counts(rows):
    summary = {}
    for depth in DESIGN["depth_ppm"]:
        summary[str(depth)] = {}
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["profile_family"] == "nominal"
                and row["method"] == method
                and row["truth"]["depth_ppm"] == depth
            ]
            summary[str(depth)][method] = {
                "trials": len(selected),
                "recovered": sum(row["recovered"] for row in selected),
                "incremental_recovery": sum(
                    row["incremental_recovery"] for row in selected
                ),
            }
    return summary


def check_delivered_sap(data, reconstructed):
    if "delivered_sap" not in data:
        raise ValueError("Exact delivered SAP comparator is required")
    reference = data["delivered_sap"]
    difference = np.abs(reconstructed - reference)
    scale = max(float(np.median(np.abs(reference))), 1)
    maximum = float(difference.max())
    if maximum > 1e-5 * scale:
        raise ValueError("Pixel baseline does not reproduce delivered SAP")
    return {
        "maximum_absolute_difference_e_per_s": maximum,
        "maximum_relative_difference": maximum / scale,
        "relative_tolerance": 1e-5,
    }


def run_source(entry, folder, extractor):
    folder.mkdir()
    if digest(entry["path"]) != entry["sha256"]:
        raise ValueError("Frozen input hash changed")
    with np.load(entry["path"], allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    pair, extraction = extract_pair(data, extractor)
    replay = check_delivered_sap(data, pair[:, 0])
    error = np.sqrt(data["pixel_error"] ** 2 @ data["aperture"] ** 2)
    search = CommonSearch(data["time"], error)
    source_rate = float(np.median(data["delivered_psf"]))
    if not np.isfinite(source_rate) or source_rate <= 0:
        raise ValueError("Positive delivered PSF calibration required")
    identity = f"{entry['source_id']}:{entry['name']}"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
    seed += DESIGN["seed"]
    np.savez_compressed(
        folder / "baseline.npz",
        time=data["time"],
        pair=pair,
        delivered_psf=data["delivered_psf"],
        error=error,
        periods=search.periods,
        **{
            name: extraction[name]
            for name in ("prediction", "folds", "pixel_weights", "conditional_error")
            if name in extraction
        },
    )
    write(folder / "extraction.json", extraction["diagnostics"])
    write(
        folder / "observed-noise.json", observed_noise(data, pair, search, source_rate)
    )
    calibrations, null_results = {}, {}
    for block in DESIGN["block_days"]:
        calibrations[block], null_results[str(block)] = null_block(
            data, search, extractor, folder, block, seed
        )
    joint_nulls = joint_null_summary(calibrations, folder)
    write(folder / "joint-nulls.json", joint_nulls)
    rows = injections(
        data, pair, search, extractor, folder, source_rate, calibrations, seed
    )
    report = {
        "name": entry["name"],
        "source_id": entry["source_id"],
        "cadences": len(data["time"]),
        "span_days": float(np.ptp(data["time"])),
        "source_rate_e_per_s": source_rate,
        "source_rate_policy": DESIGN["source_rate"],
        "delivered_sap_replay": replay,
        "observed_noise": observed_noise(data, pair, search, source_rate),
        "nulls": null_results,
        "joint_nulls": joint_nulls,
        "injection_rows": len(rows),
        "profile_families": list(injection_families(data)),
        "aperture_response_by_profile": {
            name: float(np.median(profile @ data["aperture"]))
            for name, profile in injection_families(data).items()
        },
        "nominal_recovery": recovery_counts(rows),
        "injections_sha256": digest(folder / "injections.json"),
    }
    write(folder / "report.json", report)
    return report


def source_bootstrap(effects):
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(DESIGN["seed"] + 100)
    bootstrap = rng.choice(values, (4096, len(values)), replace=True).mean(axis=1)
    return {
        "sources": len(values),
        "source_effects": values.tolist(),
        "mean_source_effect": float(values.mean()),
        "source_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "limitation": "Tiny selected source sample; no population generalization",
    }


def recovery_effects(results, comparator):
    recovery = {}
    for depth in DESIGN["depth_ppm"]:
        effects = []
        for result in results:
            counts = result["nominal_recovery"][str(depth)]
            effects.append(
                counts[METHODS[1]]["incremental_recovery"]
                / counts[METHODS[1]]["trials"]
                - counts[comparator]["incremental_recovery"]
                / counts[comparator]["trials"]
            )
        recovery[str(depth)] = source_bootstrap(effects)
    return recovery


def aggregate(results):
    noise = {}
    for hours in DESIGN["duration_hours"]:
        by_method = {
            name: np.array(
                [
                    r["observed_noise"][name][str(hours)]["source_relative_sigma_ppm"]
                    for r in results
                ]
            )
            for name in (*METHODS, "delivered_psf")
        }
        noise[str(hours)] = {
            name: source_bootstrap(1 - by_method[METHODS[1]] / by_method[name])
            for name in (METHODS[0], METHODS[2], "delivered_psf")
        }
    return {
        "fractional_noise_reduction": noise,
        "paired_incremental_recovery_fraction_change": recovery_effects(
            results, METHODS[0]
        ),
        "paired_incremental_recovery_vs_cbv_only": recovery_effects(
            results, METHODS[2]
        ),
        "sign": "Positive means new extraction has lower source-equivalent noise",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["cases"] if isinstance(manifest, dict) else manifest
    module = importlib.import_module("dwarfs.experiments.reprocess")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path(module.__file__),
        Path(physical_loss.__code__.co_filename),
    ]
    for source in sources:
        shutil.copyfile(source, args.out / source.name)
    write(
        args.out / "plan.json",
        {
            "design": DESIGN,
            "cases": entries,
            "sources": {str(p): digest(p) for p in sources},
            "runtime": {
                "python": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "astropy", "batman-package")
                },
            },
            "manifest_sha256": digest(args.manifest),
        },
    )
    results = []
    for entry in entries:
        try:
            result = run_source(entry, args.out / entry["name"], module.fit_extract)
        except Exception as error:
            write(
                args.out / "first-failure.json",
                {
                    "case": entry["name"],
                    "exception": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        results.append(result)
        print(json.dumps({"name": entry["name"], "status": "COMPLETE"}), flush=True)
    write(
        args.out / "completion.json",
        {"cases": results, "aggregate": aggregate(results)},
    )


if __name__ == "__main__":
    main()
