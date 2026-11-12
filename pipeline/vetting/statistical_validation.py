"""Run the paper-aligned TIC validation census for the two active BOIs."""

import argparse
import datetime
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from pipeline.vetting import validation_scene
from pipeline.vetting.scenario_bounds import LikelihoodBound
from pipeline.vetting.validation_models import evaluate_sample

PLAN_PATH = Path(__file__).with_name("statistical-validation.json")
RUNS_PER_TARGET = 20
SOURCE_IDS = ("2077554781499350144", "2127186564572468352")
PLAN_KEYS = {
    "background_area_multiplier",
    "catalogue",
    "draws_per_scenario",
    "ephemeris",
    "evidence_root",
    "likelihood_bound_log_budget",
    "output_root",
    "scope",
    "seed_entropy",
    "targets",
    "thresholds",
}
TARGET_KEYS = {
    "closure",
    "config",
    "field",
    "geometry",
    "native_summary",
    "packet",
    "route",
    "scenario_count",
    "source_id",
    "unresolved_geometry",
}
THRESHOLD_KEYS = {
    "FPP_lt",
    "NFPP_lt",
    "RUWE_le",
    "epoch_ess_ge",
    "radius_ratio_snr_gt",
    "rank_rhat_le",
}


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text())
    if set(plan) != PLAN_KEYS:
        raise ValueError("Validation plan schema differs from the TIC-only protocol")
    if plan["catalogue"] != "TIC" or plan["ephemeris"] != "DISCOVERY":
        raise ValueError("Only the TIC discovery-ephemeris protocol is supported")
    if plan["draws_per_scenario"] != 1_000_000:
        raise ValueError("Each scenario family requires one million attempted draws")
    thresholds = plan["thresholds"]
    if set(thresholds) != THRESHOLD_KEYS:
        raise ValueError("Validation threshold schema differs")
    if thresholds["FPP_lt"] != 0.015 or thresholds["NFPP_lt"] != 0.001:
        raise ValueError("Validation thresholds differ from the published protocol")
    targets = plan["targets"]
    if tuple(target["source_id"] for target in targets) != SOURCE_IDS:
        raise ValueError("Validation plan must contain exactly the two active BOIs")
    for target in targets:
        if set(target) != TARGET_KEYS:
            raise ValueError(
                "Validation target schema differs from the TIC-only protocol"
            )
        packet = str(target["packet"]).lower()
        if "tic" not in packet or "original" not in Path(packet).name:
            raise ValueError(
                "Validation target is not the TIC discovery-ephemeris input"
            )
    return plan


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable(value), indent=2) + "\n")
    temporary.replace(path)


def restore(value):
    if isinstance(value, dict):
        if set(value) == {"nonfinite"}:
            return float(value["nonfinite"])
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore(item) for item in value]
    return value


def frozen_files(plan):
    root = Path(plan["evidence_root"])
    files = {
        Path(__file__).resolve(),
        Path(__file__).with_name("scenario_bounds.py").resolve(),
        Path(__file__).with_name("validation_results.py").resolve(),
        Path(__file__).with_name("validation_campaign.py").resolve(),
        Path(__file__).with_name("validation_models.py").resolve(),
        Path(__file__).with_name("validation_scene.py").resolve(),
        Path(__file__).with_name("validation_runtime.py").resolve(),
    }
    files.add(Path(plan["output_root"]) / "audit/requirements.lock.txt")
    files.add(Path(plan["output_root"]) / "audit/physical-evidence.json")
    files.add(Path(plan["output_root"]) / "audit/generation-lineage.json")
    files.update((Path(plan["output_root"]) / "audit/catalogues").glob("*"))
    for target in plan["targets"]:
        for key in (
            "packet",
            "native_summary",
            "config",
            "geometry",
            "unresolved_geometry",
            "closure",
        ):
            files.add(Path(target[key]).resolve())
        manifest = json.loads(Path(target["closure"]).read_text())
        for item in manifest["base_files"] + manifest["delta_files"]:
            path = root / item["relative_path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Historical scientific dependency changed: {path}")
            files.add(path.resolve())
    for folder in (
        root / "statistical",
        root / "rv/kepler-route",
        root / "evidence/secondary-2127186564572468352",
    ):
        files.update(path.resolve() for path in folder.glob("*.py"))
    return sorted(files)


def freeze():
    plan = load_plan()
    output = Path(plan["output_root"]) / "freeze.json"
    if output.exists():
        raise FileExistsError(output)
    files = frozen_files(plan)
    receipt = {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "plan": str(PLAN_PATH.resolve()),
        "plan_sha256": digest(PLAN_PATH),
        "implementation_sha256": digest(__file__),
        "files": [
            {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "platform": {
            "hostname": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "draws_per_scenario": plan["draws_per_scenario"],
        "runs_per_target": RUNS_PER_TARGET,
        "targets": [
            {
                "source_id": target["source_id"],
                "scenario_count": target["scenario_count"],
            }
            for target in plan["targets"]
        ],
    }
    write_json(output, receipt)
    return receipt


def verify():
    plan = load_plan()
    receipt = json.loads((Path(plan["output_root"]) / "freeze.json").read_text())
    if (
        digest(PLAN_PATH) != receipt["plan_sha256"]
        or digest(__file__) != receipt["implementation_sha256"]
    ):
        raise ValueError("Frozen plan or integration implementation changed")
    for item in receipt["files"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen scientific dependency changed: " + item["path"])
    return plan, receipt


def context(plan, target, batch):
    root = Path(plan["evidence_root"])
    sys.path[:0] = [
        str(root / "statistical"),
        str(root / "rv/kepler-route"),
        str(root / "evidence/secondary-2127186564572468352"),
    ]
    core = importlib.import_module("kepler_scenarios")
    prepared = importlib.import_module(
        "prepared_total_binned_likelihood_v4"
    ).PreparedTotalBinnedLikelihood
    packet = prepared(target["packet"])
    field = Path(target["field"])
    common = [
        field,
        "TIC",
        field / "trilegal-gaia.dat",
        target["geometry"],
        target["unresolved_geometry"],
        "EXP_MEDIAN_FIT",
        "INDEPENDENT_QUARTER_DISCRETE_UNIFORM",
        0.1,
    ]
    if target["route"] == "DR2_FIELD":
        prior = importlib.import_module("field_aperture_prior_v18").FieldAperturePrior(
            *common, "MIST_BC_DR2REV_HISTORICAL_G", 0.003
        )
        kernel = importlib.import_module("kepler_scenarios_v22")
    elif target["route"] == "PAIRED_FIELD":
        prior = importlib.import_module("field_prior_v2").FieldPriorV2(
            *common, "EXACT_PAIRED_NATIVE_TRILEGAL_COLORS"
        )
        kernel = importlib.import_module("kepler_scenarios_v3")
    else:
        raise ValueError("Unknown TIC stellar-scene route")
    inventory = json.loads(Path(target["config"]).read_text())
    config = {
        key: inventory[key]
        for key in (
            "source_id",
            "flat_planet_priors",
            "unresolved_radius_arcsec",
            "background_area_factor",
            "population_area_deg2",
        )
    }
    config.update(
        period_days=float(packet.arrays["period_days"]),
        epoch_bkjd=float(packet.arrays["epoch_bkjd_tdb"]),
        exposure_days=float(
            packet.arrays["frames"] * packet.arrays["frame_seconds"] / 86400
        ),
        integration_samples=54,
        noise_quadrature_nodes=128,
        renderer="PIECEWISE_LINEAR_PHASE_GRID",
        phase_grid_nodes_per_cycle=800,
        observed_gaia_component_flux="COMPOSITE_OBSERVED_G",
    )
    offset = (
        0.122666
        if target["route"] == "PAIRED_FIELD"
        else inventory["kepler_AB_minus_Vega_mag"]
    )
    background = core.population(field / "trilegal-kepler.dat", offset)
    if config["background_area_factor"] != 1:
        raise ValueError("The historical background-area convention changed")
    area = config["unresolved_radius_arcsec"] ** 2 / 3600**2
    maximum_prior = (
        len(background["Kepler_Vega"]) * area / config["population_area_deg2"]
    )
    if maximum_prior * plan["background_area_multiplier"] >= 1:
        raise ValueError(
            "Background clipping prevents exact area sensitivity reweighting"
        )
    config["background_area_factor"] = plan["background_area_multiplier"]
    likelihood = packet
    names = core.BASE_SCENARIOS + [
        f"{kind}:{sid}"
        for sid in prior.ids
        if sid != target["source_id"]
        for kind in ("NTP", "NEB", "NEBx2P")
    ]
    if len(names) != target["scenario_count"] or len(set(names)) != len(names):
        raise ValueError("Frozen complete scenario census differs")
    validation_scene.prepare(
        prior, target["source_id"], config["unresolved_radius_arcsec"]
    )
    return (
        prior,
        kernel,
        packet,
        likelihood,
        background,
        config,
        proposal_radius(target),
        names,
    )


def proposal_radius(target):
    summary = json.loads(Path(target["native_summary"]).read_text())
    inference = (
        summary["families"]["tic"]["admissible_inference"]
        if "families" in summary
        else summary["admissible_inference"]
    )
    shape = inference["quantiles_16_50_84"]
    key = next(
        key
        for key in ("apparent_radius_ratio_unit_dilution", "radius_ratio_unit_dilution")
        if key in shape
    )
    return shape[key][1]


def accumulate(total, values):
    for key, value in values.items():
        if isinstance(value, dict):
            accumulate(total.setdefault(key, {}), value)
        elif isinstance(value, (int, np.integer)):
            total[key] = total.get(key, 0) + int(value)
        else:
            total.setdefault(key, value)


def physical_sample(state, scenario, size, rng):
    prior, kernel, packet, _, background, config, radius, _ = state
    args = (
        (size, rng)
        if type(prior).__name__ == "FieldPriorV2"
        else (size, rng, config["unresolved_radius_arcsec"])
    )
    scene, geometry, supported, indices, states, position, kp = prior.sample(*args)
    if not supported.all():
        raise ValueError("Unsupported prior scene; no attempted draw may be discarded")
    config["target_kp_mag"] = kp
    physical = kernel.scenario_draws(
        scenario,
        scene[config["source_id"]]["star"],
        scene,
        background,
        config,
        size,
        rng,
    )
    if physical["binary"]:
        proposal = importlib.import_module(
            "binary_proposals_v3"
        ).propose_binary_inclination
        physical, ratio, counts = proposal(
            physical,
            packet,
            scene,
            geometry,
            config,
            rng,
            float((1 - packet.m0).max()),
            float(packet.arrays["duration_days"]),
        )
    else:
        proposal = importlib.import_module("planet_proposals_v3").propose
        physical, ratio, counts = proposal(
            physical,
            packet,
            scene,
            geometry,
            config,
            rng,
            radius,
            float(packet.arrays["duration_days"]),
        )
    return physical, ratio, counts, scene, geometry, indices, states, position


def best_draw(physical, ratio, indices, states, position, selected, begin, maximum):
    keys = (
        "inc",
        "ecc",
        "argp",
        "a_cm",
        "host_radius",
        "occultor_radius",
        "host_flux",
        "extra_flux",
        "secondary_flux",
        "u1",
        "u2",
        "secondary_u1",
        "secondary_u2",
        "logprior",
    )
    return {
        "draw_index": begin + selected,
        "log_weight": maximum,
        "physical": {key: float(np.asarray(physical[key])[selected]) for key in keys},
        "log_importance_ratio": float(ratio[selected]),
        "stellar_particle_indices": {
            sid: int(value[selected]) for sid, value in indices.items()
        },
        "pointing_states": states[selected],
        "unresolved_position_arcsec": position[selected],
    }


def subset(value, selected, size):
    if isinstance(value, dict):
        return {key: subset(item, selected, size) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim and len(value) == size:
        return value[selected]
    return value


def integrate_scenario(
    state,
    scenario,
    rng,
    draws,
    batch,
    directory,
    index,
    bound=None,
    cutoff=-np.inf,
    render_batch=64,
):
    logs = np.full(draws, -np.inf)
    omitted = np.zeros(draws, dtype=bool)
    planet_radius = np.empty(draws) if scenario in ("TP", "PTP", "DTP") else None
    proposal_counts, domains = {}, {}
    accepted, best = 0, None
    begun = time.monotonic()
    for begin in range(0, draws, batch):
        size = min(batch, draws - begin)
        physical, ratio, proposal, scene, geometry, indices, states, position = (
            physical_sample(state, scenario, size, rng)
        )
        if planet_radius is not None:
            core = importlib.import_module("kepler_scenarios")
            planet_radius[begin : begin + size] = (
                physical["occultor_radius"] * core.RSUN / core.REARTH
            )
        accumulate(proposal_counts, proposal)
        accepted += int(physical["valid"].sum())
        values, bounded = evaluate_sample(
            state, physical, ratio, scene, geometry, bound, cutoff, render_batch
        )
        logs[begin : begin + size] = values
        omitted[begin : begin + size] = bounded
        if np.isfinite(values).any():
            selected = int(np.argmax(values))
            maximum = float(values[selected])
            if best is None or maximum > best["log_weight"]:
                best = best_draw(
                    physical, ratio, indices, states, position, selected, begin, maximum
                )
        for key in (
            "host_domain_report",
            "secondary_domain_report",
            "companion_stellar_relation_counts",
            "secondary_stellar_relation_counts",
        ):
            if physical[key] is not None:
                accumulate(domains.setdefault(key, {}), physical[key])
        for key in (
            "cold_identical_component_count",
            "native_background_primary_count",
        ):
            if key in physical:
                domains[key] = domains.get(key, 0) + int(physical[key])
    total = float(logsumexp(logs))
    finite = int(np.isfinite(logs).sum())
    ess = float(np.exp(2 * total - logsumexp(2 * logs))) if finite else 0.0
    weights_path = directory / f"weights-{index:03d}.npz"
    temporary = weights_path.with_suffix(".tmp")
    arrays = {
        "log_weight": logs,
        "bounded": omitted,
        "bounded_log_weight_upper": cutoff,
    }
    if planet_radius is not None:
        arrays["planet_radius_earth"] = planet_radius
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(weights_path)
    bounded_count = int(omitted.sum())
    omitted_log_evidence_upper = (
        cutoff + np.log(bounded_count / draws) if bounded_count else -np.inf
    )
    ess_scope = (
        "ALL_ATTEMPTED_DRAWS"
        if not bounded_count
        else "EVALUATED_WEIGHTS_ONLY; UNEVALUATED_WEIGHTS_HAVE_EXPLICIT_EVIDENCE_BOUNDS"
    )
    return {
        "scenario": scenario,
        "draws": draws,
        "accepted_physical_draws": accepted,
        "finite_likelihood_draws": finite,
        "log_evidence": total - np.log(draws),
        "log_evidence_upper": float(
            np.logaddexp(total - np.log(draws), omitted_log_evidence_upper)
        ),
        "omitted_log_evidence_upper": omitted_log_evidence_upper,
        "bounded_physical_draws": bounded_count,
        "likelihood_weight_effective_draws": ess,
        "weight_diagnostic_scope": ess_scope,
        "maximum_normalized_likelihood_weight": float(np.exp(np.max(logs) - total))
        if finite
        else None,
        "iid_relative_standard_error": float(np.sqrt(max(1 / ess - 1 / draws, 0)))
        if finite
        else None,
        "elapsed_seconds": time.monotonic() - begun,
        "weight_file": weights_path.name,
        "weight_sha256": digest(weights_path),
        "proposal_counts": proposal_counts,
        "domain_receipts": domains,
        "best_draw": best,
        "zero_evidence_scope": None
        if finite
        else "ONLY_BOUNDED_LIKELIHOODS"
        if bounded_count
        else "NO_FINITE_DRAW_IN_THIS_SAMPLE; NOT_EXACT_ZERO",
    }


def probabilities(rows, background_multiplier=1.0):
    names = [row["scenario"] for row in rows]
    evidence = np.array([row["log_evidence"] for row in rows])
    upper_evidence = np.array(
        [row.get("log_evidence_upper", row["log_evidence"]) for row in rows]
    )
    background = np.array([name.startswith(("D", "B")) for name in names])
    evidence[background] += np.log(background_multiplier)
    upper_evidence[background] += np.log(background_multiplier)
    denominator = logsumexp(evidence)
    if not np.isfinite(denominator):
        raise ValueError("Complete scenario census has no finite evidence")
    category = {
        "FPP": np.array([name not in ("TP", "PTP", "DTP") for name in names]),
        "NFPP": np.array([name.startswith("N") for name in names]),
        "EB": np.array([not name.split(":")[0].endswith("TP") for name in names]),
    }
    result = {
        key: {
            "log_probability": float(logsumexp(evidence[mask]) - denominator),
            "probability": float(np.exp(logsumexp(evidence[mask]) - denominator)),
        }
        for key, mask in category.items()
    }
    for key, mask in category.items():
        lower, upper = logsumexp(evidence[mask]), logsumexp(upper_evidence[mask])
        other_lower, other_upper = (
            logsumexp(evidence[~mask]),
            logsumexp(upper_evidence[~mask]),
        )
        result[key]["probability_lower"] = float(
            np.exp(lower - np.logaddexp(lower, other_upper))
        )
        result[key]["probability_upper"] = float(
            np.exp(upper - np.logaddexp(upper, other_lower))
        )
    missing = logsumexp(
        [row.get("omitted_log_evidence_upper", -np.inf) for row in rows]
    )
    result["bounded_posterior_mass_upper"] = float(np.exp(missing - denominator))
    return result


def resume(checkpoint, directory, identity, names, rng):
    if not checkpoint.exists():
        return []
    saved = restore(json.loads(checkpoint.read_text()))
    if saved["identity"] != identity:
        raise ValueError("Checkpoint arguments or frozen scientific model differ")
    rows = saved["completed_scenarios"]
    for row in rows:
        if digest(directory / row["weight_file"]) != row["weight_sha256"]:
            raise ValueError("Retained numerical weights changed")
    if [row["scenario"] for row in rows] != names[: len(rows)]:
        raise ValueError("Checkpoint scenario census differs")
    rng.bit_generator.state = saved["rng_state"]
    return rows


def run(
    source_id,
    run_index,
    batch,
    render_batch,
):
    from pipeline.vetting.validation_runtime import application_qos

    application_qos()
    if any(
        os.environ.get(key) != "1"
        for key in (
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    ):
        raise ValueError("Numerical thread counts must be pinned to one")
    plan, frozen = verify()
    output_root = Path(plan["output_root"])
    if shutil.disk_usage(output_root).free < 2 * 1024**3:
        raise ValueError("Output disk reserve is insufficient: " + str(output_root))
    target_index = next(
        index
        for index, target in enumerate(plan["targets"])
        if target["source_id"] == source_id
    )
    target = plan["targets"][target_index]
    if not 0 <= run_index < RUNS_PER_TARGET or batch <= 0 or render_batch <= 0:
        raise ValueError("Invalid run index or batch size")
    seed = int(
        np.random.SeedSequence(
            plan["seed_entropy"] + [target_index, run_index]
        ).generate_state(1, dtype=np.uint64)[0]
    )
    draws = plan["draws_per_scenario"]
    directory = Path(plan["output_root"]) / "runs" / source_id / f"seed-{run_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    output, checkpoint = directory / "result.json", directory / "checkpoint.json"
    if output.exists():
        raise FileExistsError(output)
    identity = {
        "freeze_sha256": digest(Path(plan["output_root"]) / "freeze.json"),
        "source_id": source_id,
        "seed": seed,
        "run_index": run_index,
        "draws_per_scenario": draws,
        "batch": batch,
        "render_batch": render_batch,
    }
    rng = np.random.default_rng(seed)
    begun = time.monotonic()
    state = context(plan, target, batch)
    if set(np.unique(state[3].quarter)) - set(state[0].quarters):
        raise ValueError(
            "Prepared photometry is outside the authenticated aperture quarters"
        )
    bound = LikelihoodBound(state[3], False)
    names = state[-1]
    rows = resume(checkpoint, directory, identity, names, rng)
    for index in range(len(rows), len(names)):
        cutoff = (
            rows[0]["log_evidence"]
            - plan["likelihood_bound_log_budget"]
            - np.log(len(names))
            if rows
            else -np.inf
        )
        row = integrate_scenario(
            state,
            names[index],
            rng,
            draws,
            batch,
            directory,
            index,
            bound,
            cutoff,
            render_batch,
        )
        rows.append(row)
        write_json(
            checkpoint,
            {
                "identity": identity,
                "status": "PARTIAL_NO_POSTERIOR",
                "complete_scenario_names": names,
                "completed_scenarios": rows,
                "rng_state": rng.bit_generator.state,
            },
        )
        print(
            json.dumps(
                serializable(
                    {
                        key: row[key]
                        for key in (
                            "scenario",
                            "draws",
                            "log_evidence",
                            "likelihood_weight_effective_draws",
                            "elapsed_seconds",
                        )
                    }
                )
            ),
            flush=True,
        )
    verify()
    receipt = {
        "identity": identity,
        "status": "COMPLETE_INDEPENDENT_RUN",
        "scenario_names": names,
        "scenario_evidences": rows,
        "probabilities": probabilities(rows),
        "config": {
            key: value for key, value in state[5].items() if key != "target_kp_mag"
        },
        "likelihood": state[3].provenance,
        "elapsed_seconds": time.monotonic() - begun,
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": frozen["platform"],
        "scope": (
            "Conditional astrophysical probabilities. Instrumental false alarms "
            "and omitted model/systematic uncertainty are not probabilities "
            "measured by this calculation."
        ),
    }
    write_json(output, receipt)
    print(json.dumps(receipt["probabilities"]), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run"))
    parser.add_argument("--source-id", choices=SOURCE_IDS)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--render-batch", type=int, default=64)
    args = parser.parse_args()
    if args.action == "freeze":
        receipt = freeze()
        print(
            json.dumps(
                {
                    "files": len(receipt["files"]),
                    "bytes": sum(item["bytes"] for item in receipt["files"]),
                }
            )
        )
    elif args.action == "verify":
        _, receipt = verify()
        print(json.dumps({"status": "PASS", "files": len(receipt["files"])}))
    else:
        if args.source_id is None:
            parser.error("run requires --source-id")
        run(
            args.source_id,
            args.run_index,
            args.batch,
            args.render_batch,
        )


if __name__ == "__main__":
    main()
