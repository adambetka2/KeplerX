"""Frozen five-product comparison with explicit conditional PSF signal response."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dwarfs.experiments import reprocess, reprocess_validation as previous
from dwarfs.experiments.preparation import physical_loss

METHODS = (*previous.METHODS, "delivered_psf", "delivered_psf_flat")
OLD_SOURCES = {
    "2133106575693656832",
    "2105449558988715264",
    "2101239670706093696",
    "2077139784581832064",
    "2126093581293485184",
}
DESIGN = {
    **previous.DESIGN,
    "schema": 2,
    "seed": 2026101420,
    "methods": list(METHODS),
    "primary_depth_ppm": [40.0, 100.0, 300.0, 1000.0],
    "positive_control_depth_ppm": [3000.0, 10000.0],
    "search_error": "Uniform cadence weights after source-response normalization",
    "source_rate": "Native corrected PSF median calibrates pixel source amplitude",
    "delivered_normalization": "Each delivered PSF and PSF Flat common-mask median",
    "clock": "Actual supported centers within D/2+exposure; energy>=0.5; >=3 events",
    "null": "Shared block signs for whole pixels, background, PSF and PSF Flat",
    "comparison": {
        "delivered_sap": "Target pixel injection; exact aperture and background replay",
        "pixel_reprocessed": "Target pixel injection; complete adaptive refit",
        "cbv_pointing_only": "Target pixel injection; complete adaptive refit",
        "delivered_psf": "Conditional delivered-series injection; ideal unit response",
        "delivered_psf_flat": "Conditional delivered injection; ideal unit response",
    },
    "secondary_control": "Ideal post-delivery new/CBV loss; same primary thresholds",
    "bootstrap": "4096 source resamples; exploratory intervals, multiple comparisons",
    "pilot_decision": {
        "primary": "Pooled20, nominal40-1000ppm; other depths and M/K are descriptive",
        "recovery": "Lower95% source-bootstrap bound>0 versus SAP, PSF and PSF Flat",
        "false_alarm": "Pooled pixel held rate<=5% under each combined null decision",
        "transfer": "Every nominal full-refit amplitude between90% and110%",
        "meaning": "Common calibrated alarm rule; not identical empirical false alarms",
    },
    "limits": [
        "Restricted period search, not production completeness",
        "Original KBonus PSF and PSF Flat extraction is not replayed",
        "Conditional block-wild surrogate nulls are not an astrophysical null census",
        "Source flux, profile and Flat response normalization are model assumptions",
        "Selected source sample; source uncertainty does not prove population benefit",
    ],
}


def normalizations(data):
    source = float(data["source_flux"])
    response = float(np.median(data["target_profiles"] @ data["aperture"]))
    psf = float(np.median(data["delivered_psf"]))
    flat = float(np.median(data["delivered_psf_flat"]))
    values = np.array([source * response] * 3 + [psf, flat])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("All five source-response normalizations must be positive")
    return values


def null_components(data):
    augmented = np.column_stack(
        (
            data["pixel_flux"],
            data["background"],
            data["delivered_psf"],
            data["delivered_psf_flat"],
        )
    )
    q = previous.basis(
        data["time"], np.ones(len(augmented)), data["pointing"], data["cbv"]
    )
    trend = q @ (q.T @ augmented)
    return trend, augmented - trend


def null_draw(data, trend, residual, block_days, rng):
    blocks = np.floor((data["time"] - data["time"][0]) / block_days).astype(int)
    signs = rng.choice([-1.0, 1.0], int(blocks.max()) + 1)
    values = trend + residual * signs[blocks, None]
    simulated = dict(
        data,
        pixel_flux=values[:, :-3],
        background=values[:, -3],
        delivered_psf=values[:, -2],
        delivered_psf_flat=values[:, -1],
    )
    # Null perturbations must not change the source-response calibration.
    simulated["normalization"] = data["normalization"]
    return simulated, signs


def calibrated_curves(data):
    pair, fitted = previous.extract_pair(data, reprocess.fit_extract)
    curves = np.column_stack((pair, data["delivered_psf"], data["delivered_psf_flat"]))
    return curves / data["normalization"], fitted


def null_population(data, search, trend, residual, block, seed, population, folder):
    offset = 0 if population == "calibration" else 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(block), offset]))
    curves, scores, signs, fits = [], [], [], []
    count = DESIGN[f"{population}_nulls"]
    for draw in range(count):
        simulated, sign = null_draw(data, trend, residual, block, rng)
        pair, fitted = calibrated_curves(simulated)
        proposals = [search.search(pair[:, k]) for k in range(len(METHODS))]
        curves.append(pair.astype(np.float32))
        scores.append([p["score"] for p in proposals])
        signs.append(sign)
        fits.append(
            {
                "draw": draw,
                "proposals": proposals,
                "extraction": previous.compact_fit(fitted),
            }
        )
    previous.write(
        folder / f"null-fits-{block:g}d-{population}.json",
        {
            "seed_sequence": [seed, int(block), offset],
            "fits": fits,
        },
    )
    return {
        "curves": np.asarray(curves),
        "scores": np.asarray(scores),
        "signs": np.asarray(signs),
    }


def null_block(data, search, folder, block, seed):
    trend, residual = null_components(data)
    np.savez_compressed(
        folder / f"null-model-{block:g}d.npz", trend=trend, residual=residual
    )
    populations = {
        name: null_population(data, search, trend, residual, block, seed, name, folder)
        for name in ("calibration", "held")
    }
    path = folder / f"nulls-{block:g}d.npz"
    np.savez_compressed(
        path,
        **{
            f"{name}_{key}": value
            for name, fields in populations.items()
            for key, value in fields.items()
        },
    )
    results = {
        method: previous.false_alarm_summary(
            populations["calibration"]["scores"][:, k],
            populations["held"]["scores"][:, k],
        )
        for k, method in enumerate(METHODS)
    }
    previous.write(
        folder / f"nulls-{block:g}d.json",
        {
            "block_days": block,
            "results": results,
            "arrays_sha256": previous.digest(path),
        },
    )
    return populations["calibration"]["scores"], results


def joint_nulls(calibrations, folder):
    result = {}
    for block in DESIGN["block_days"]:
        with np.load(folder / f"nulls-{block:g}d.npz") as saved:
            held = saved["held_scores"]
        result[str(block)] = {
            method: previous.rate_summary(
                np.max(
                    [
                        previous.empirical_p(scores[:, k], held[:, k])
                        for scores in calibrations.values()
                    ],
                    axis=0,
                )
            )
            for k, method in enumerate(METHODS)
        }
    return result


def clock_match(time, proposal, truth, expected):
    period, epoch = proposal["period_days"], proposal["epoch_bkjd"]
    true_period, true_epoch = truth["period_days"], truth["epoch_bkjd"]
    support = np.abs(expected) > 0
    events = np.unique(np.rint((time[support] - true_epoch) / true_period))
    centers = true_epoch + events * true_period
    timing = np.abs(centers - epoch - np.rint((centers - epoch) / period) * period)
    phase = (time - epoch + period / 2) % period - period / 2
    window = np.abs(phase) < proposal["duration_hours"] / 48
    total = float(expected @ expected)
    energy = float(expected[window] @ expected[window] / total) if total > 0 else 0.0
    tolerance = truth["duration_hours"] / 48 + 0.02043361111111111
    maximum = float(timing.max()) if len(timing) else None
    return {
        "clock_match": bool(
            len(timing) >= 3 and maximum <= tolerance and energy >= 0.5
        ),
        "supported_events": len(events),
        "energy_covered": energy,
        "maximum_timing_offset_hours": None if maximum is None else maximum * 24,
        "timing_tolerance_hours": tolerance * 24,
    }


def injected_curves(data, loss, profile):
    source = float(data["source_flux"])
    delta = -source * loss[:, None] * profile
    simulated = dict(
        data,
        pixel_flux=data["pixel_flux"] + delta,
        delivered_psf=data["delivered_psf"] - loss * data["normalization"][3],
        delivered_psf_flat=(
            data["delivered_psf_flat"] - loss * data["normalization"][4]
        ),
    )
    pair, fitted = calibrated_curves(simulated)
    aperture_loss = delta @ data["aperture"] / data["normalization"][0]
    expected = np.column_stack(
        (aperture_loss, aperture_loss, aperture_loss, -loss, -loss)
    )
    return pair, expected, fitted


def trial_rows(data, baseline, search, truth, profile, calibrations):
    loss = physical_loss(
        data["time"],
        truth["period_days"],
        truth["epoch_bkjd"],
        truth["duration_hours"],
        truth["depth_ppm"],
    )
    pair, expected, fitted = injected_curves(data, loss, profile)
    rows = []
    for k, method in enumerate(METHODS):
        proposal, base = search.search(pair[:, k]), search.search(baseline[:, k])
        pvalues = {
            str(b): float(previous.empirical_p(s[:, k], proposal["score"])[0])
            for b, s in calibrations.items()
        }
        base_p = max(
            float(previous.empirical_p(s[:, k], base["score"])[0])
            for s in calibrations.values()
        )
        matched = clock_match(data["time"], proposal, truth, expected[:, k])
        base_match = clock_match(data["time"], base, truth, expected[:, k])
        base_recovered = bool(base_match["clock_match"] and base_p <= DESIGN["alpha"])
        recovered = bool(
            matched["clock_match"] and max(pvalues.values()) <= DESIGN["alpha"]
        )
        row = {
            "method": method,
            "response_policy": DESIGN["comparison"][method],
            "truth": truth,
            "proposal": proposal,
            "empirical_p": pvalues,
            **matched,
            "known_clock_retention": search.transfer(
                pair[:, k] - baseline[:, k], expected[:, k]
            ),
            "baseline_recovered": base_recovered,
            "recovered": recovered,
            "incremental_recovery": recovered and not base_recovered,
        }
        if k in (1, 2):
            row["post_delivery_control"] = conditional_control(
                data,
                baseline[:, k],
                expected[:, k],
                search,
                truth,
                calibrations,
                k,
                base_recovered,
            )
        rows.append(row)
    return rows, loss, previous.compact_fit(fitted)


def conditional_control(
    data, baseline, expected, search, truth, calibrations, index, baseline_recovered
):
    proposal = search.search(baseline + expected)
    matched = clock_match(data["time"], proposal, truth, expected)
    pvalues = {
        str(b): float(previous.empirical_p(s[:, index], proposal["score"])[0])
        for b, s in calibrations.items()
    }
    recovered = bool(
        matched["clock_match"] and max(pvalues.values()) <= DESIGN["alpha"]
    )
    return {
        "proposal": proposal,
        **matched,
        "empirical_p": pvalues,
        "recovered": recovered,
        "incremental_recovery": recovered and not baseline_recovered,
        "purpose": DESIGN["secondary_control"],
    }


def injections(data, baseline, search, calibrations, folder, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 999]))
    rows, losses, fits = [], [], []
    families = previous.injection_families(data)
    for number, truth in enumerate(previous.truth_rows(data, rng)):
        names = (
            list(families)
            if (
                truth["duration_hours"] == 4.0
                and truth["depth_ppm"] in (1000.0, 3000.0)
            )
            else ["nominal"]
        )
        for name in names:
            result, loss, fitted = trial_rows(
                data, baseline, search, truth, families[name], calibrations
            )
            rows.extend(
                dict(row, injection=number, profile_family=name) for row in result
            )
            fits.append(
                {"injection": number, "profile_family": name, "extraction": fitted}
            )
        losses.append(loss)
    previous.write(folder / "injections.json", rows)
    previous.write(folder / "injection-fits.json", fits)
    np.savez_compressed(folder / "injection-losses.npz", loss=np.asarray(losses))
    return rows


def recovery_counts(rows):
    result = {}
    for depth in DESIGN["depth_ppm"]:
        result[str(depth)] = {}
        for method in METHODS:
            selected = [
                r
                for r in rows
                if r["profile_family"] == "nominal"
                and r["truth"]["depth_ppm"] == depth
                and r["method"] == method
            ]
            result[str(depth)][method] = {
                "trials": len(selected),
                "recovered": sum(r["recovered"] for r in selected),
                "incremental_recovery": sum(
                    r["incremental_recovery"] for r in selected
                ),
            }
    return result


def observed_noise(data, pair, search):
    result = {}
    for k, method in enumerate(METHODS):
        residual = search.residual(pair[:, k])
        result[method] = {}
        for hours in DESIGN["duration_hours"]:
            value = previous.noise_at_timescale(data["time"], residual, hours)
            result[method][str(hours)] = {
                "bins": value["bins"],
                "source_relative_sigma_ppm": value["sigma_e_per_s"] * 1e6,
                "normalization_native_units": float(data["normalization"][k]),
            }
    return result


def run_source(entry, root):
    started = time.perf_counter(), time.process_time()
    folder = Path(root) / entry["name"]
    folder.mkdir()
    if previous.digest(entry["path"]) != entry["sha256"]:
        raise ValueError("Frozen input hash changed")
    with np.load(entry["path"], allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    data["normalization"] = normalizations(data)
    extraction_started = time.perf_counter(), time.process_time()
    pair, fitted = calibrated_curves(data)
    timings = {"baseline_two_extractions": elapsed(extraction_started)}
    replay = previous.check_delivered_sap(data, pair[:, 0] * data["normalization"][0])
    search = previous.CommonSearch(data["time"], np.ones(len(pair)))
    identity = f"{entry['source_id']}:{entry['name']}"
    seed = DESIGN["seed"] + int.from_bytes(
        hashlib.sha256(identity.encode()).digest()[:4], "little"
    )
    np.savez_compressed(
        folder / "baseline.npz",
        time=data["time"],
        pair=pair,
        normalization=data["normalization"],
        periods=search.periods,
        **{
            k: fitted[k]
            for k in ("prediction", "folds", "pixel_weights", "conditional_error")
        },
    )
    previous.write(folder / "extraction.json", fitted["diagnostics"])
    calibrations, nulls = {}, {}
    null_started = time.perf_counter(), time.process_time()
    for block in DESIGN["block_days"]:
        calibrations[block], nulls[str(block)] = null_block(
            data, search, folder, block, seed
        )
    joint = joint_nulls(calibrations, folder)
    timings["null_refits_and_search"] = elapsed(null_started)
    previous.write(folder / "joint-nulls.json", joint)
    injection_started = time.perf_counter(), time.process_time()
    rows = injections(data, pair, search, calibrations, folder, seed)
    timings["pixel_injections_refits_and_search"] = elapsed(injection_started)
    timings["case_total"] = elapsed(started)
    report = {
        "name": entry["name"],
        "source_id": entry["source_id"],
        "source_class": entry["source_class"],
        "cadences": len(pair),
        "parent_id": entry["parent_id"],
        "timings": timings,
        "span_days": float(np.ptp(data["time"])),
        "delivered_sap_replay": replay,
        "normalizations": dict(
            zip(METHODS, data["normalization"].tolist(), strict=True)
        ),
        "observed_noise": observed_noise(data, pair, search),
        "nulls": nulls,
        "joint_nulls": joint,
        "nominal_recovery": recovery_counts(rows),
        "injection_rows": len(rows),
        "injections_sha256": previous.digest(folder / "injections.json"),
    }
    previous.write(folder / "report.json", report)
    return entry["name"]


def elapsed(started):
    return {
        "wall_seconds": time.perf_counter() - started[0],
        "process_cpu_seconds": time.process_time() - started[1],
    }


def recovery_effects(results, comparator, depths):
    effects = []
    for result in results:
        counts = result["nominal_recovery"]
        effects.append(
            sum(
                counts[str(d)][METHODS[1]]["incremental_recovery"]
                - counts[str(d)][comparator]["incremental_recovery"]
                for d in depths
            )
            / sum(counts[str(d)][METHODS[1]]["trials"] for d in depths)
        )
    return source_bootstrap(effects)


def source_bootstrap(effects):
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(DESIGN["seed"] + 100)
    bootstrap = rng.choice(values, (4096, len(values)), replace=True).mean(axis=1)
    return {
        "sources": len(values),
        "source_effects": values.tolist(),
        "mean_source_effect": float(values.mean()),
        "source_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "limitation": "Selected source sample; exploratory unadjusted intervals",
    }


def aggregate_group(results):
    comparators = [method for method in METHODS if method != METHODS[1]]
    noise = {}
    for hours in DESIGN["duration_hours"]:
        sigma = {
            m: np.array(
                [
                    r["observed_noise"][m][str(hours)]["source_relative_sigma_ppm"]
                    for r in results
                ]
            )
            for m in METHODS
        }
        noise[str(hours)] = {
            m: source_bootstrap(1 - sigma[METHODS[1]] / sigma[m]) for m in comparators
        }
    return {
        "sources": len(results),
        "source_ids": [r["source_id"] for r in results],
        "fractional_noise_reduction": noise,
        "primary_faint_recovery_difference": {
            m: recovery_effects(results, m, DESIGN["primary_depth_ppm"])
            for m in comparators
        },
        "recovery_difference_by_depth": {
            str(d): {m: recovery_effects(results, m, [d]) for m in comparators}
            for d in DESIGN["depth_ppm"]
        },
    }


def source_paths():
    return [
        Path(__file__),
        Path(__file__).with_name("comparison_summary.py"),
        Path(reprocess.__file__),
        Path(previous.__file__),
        Path(physical_loss.__code__.co_filename),
    ]


def make_plan(manifest_path, folder):
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["cases"]
    identifiers = [str(entry["source_id"]) for entry in entries]
    if (
        len(entries) != 20
        or len(set(identifiers)) != 20
        or OLD_SOURCES.intersection(identifiers)
    ):
        raise ValueError(
            "Require exactly20 new independent sources, excluding all old five"
        )
    if len({str(e["parent_id"]) for e in entries}) != 20:
        raise ValueError("Require exactly one source per original Kepler TPF parent")
    if not manifest.get("complete") or any(
        e["source_class"] not in ("M", "K") for e in entries
    ):
        raise ValueError("Completed metadata-frozen M/K manifest required")
    if sum(e["source_class"] == "M" for e in entries) != 10:
        raise ValueError("Frozen selection requires ten M and ten K sources")
    folder.mkdir(parents=True, exist_ok=False)
    for source in source_paths():
        shutil.copyfile(source, folder / source.name)
    protocol = Path(__file__).parents[2] / "docs" / "pixel-comparison-20.md"
    shutil.copyfile(protocol, folder / protocol.name)
    previous.write(
        folder / "plan.json",
        {
            "design": DESIGN,
            "cases": entries,
            "sources": {str(p): previous.digest(p) for p in source_paths()},
            "protocol_document": {
                "name": protocol.name,
                "sha256": previous.digest(protocol),
            },
            "manifest_sha256": previous.digest(manifest_path),
            "runtime": {
                "python": platform.python_version(),
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "scipy", "astropy", "batman-package")
                },
            },
        },
    )


def verify_plan(folder):
    plan = json.loads((folder / "plan.json").read_text())
    if plan["design"] != DESIGN:
        raise ValueError("Frozen protocol changed")
    document = plan["protocol_document"]
    if previous.digest(folder / document["name"]) != document["sha256"]:
        raise ValueError("Frozen protocol document changed")
    for path, expected in plan["sources"].items():
        if (
            previous.digest(path) != expected
            or previous.digest(folder / Path(path).name) != expected
        ):
            raise ValueError("Frozen numerical source changed")
    for entry in plan["cases"]:
        if previous.digest(entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen input changed")
    return plan


def run_plan(folder, workers):
    plan = verify_plan(folder)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Require {name}=1 before importing numerical runtime")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(run_source, entry, str(folder)): entry
            for entry in plan["cases"]
        }
        for future in as_completed(jobs):
            try:
                print(
                    json.dumps({"name": future.result(), "status": "COMPLETE"}),
                    flush=True,
                )
            except Exception as error:
                previous.write(
                    folder / "first-failure.json",
                    {
                        "case": jobs[future]["name"],
                        "exception": type(error).__name__,
                        "message": str(error),
                    },
                )
                raise
    verify_plan(folder)
    results = [
        json.loads((folder / e["name"] / "report.json").read_text())
        for e in plan["cases"]
    ]
    groups = {
        "all20": results,
        **{k: [r for r in results if r["source_class"] == k] for k in ("M", "K")},
    }
    previous.write(
        folder / "completion.json",
        {
            "cases": results,
            "aggregate": {k: aggregate_group(v) for k, v in groups.items() if v},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.manifest, args.out)
    else:
        run_plan(args.out, args.workers)


if __name__ == "__main__":
    main()
