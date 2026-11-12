"""Freeze Kepler-informed flux-level release injections."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import batman
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic.run import InsufficientCadences, prepare_curve_records
from lc2 import load_curve, load_source

PLAN = Path(__file__).with_name("injection-plan.json")
RHO_SUN = 1408.0
GRAVITATIONAL_CONSTANT = 6.67430e-11
PRIMARY_FLUX = "psf_flat_kbonus_flux"
DELTA_GROUPS = (
    (
        "time_bkjd",
        "quarter",
        (PRIMARY_FLUX,),
    ),
    (
        "qdiag_time_bkjd",
        "qdiag_quarter",
        (
            "qdiag_psf_flux",
            "qdiag_sap_flux",
            "qdiag_psf_flux_nova",
        ),
    ),
)


class InjectionIneligible(ValueError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_csv(path, fields, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def unit(seed, *parts):
    payload = ":".join(map(str, (seed, *parts))).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value + 0.5) / 2**64


def uniform(seed, low, high, *parts):
    return low + (high - low) * unit(seed, *parts)


def log_uniform(seed, low, high, *parts):
    return math.exp(uniform(seed, math.log(low), math.log(high), *parts))


def periodic_schedule(strata, seed):
    labelled = [
        (item["name"], occurrence)
        for item in strata
        for occurrence in range(item["count"])
    ]
    return [
        label
        for label, _ in sorted(
            labelled,
            key=lambda item: hashlib.sha256(
                f"{seed}:schedule:{item[0]}:{item[1]}".encode()
            ).digest(),
        )
    ]


def scaled_strata(strata, count):
    total = sum(item["count"] for item in strata)
    exact = [count * item["count"] / total for item in strata]
    counts = [math.floor(value) for value in exact]
    order = sorted(
        range(len(strata)), key=lambda index: exact[index] - counts[index], reverse=True
    )
    for index in order[: count - sum(counts)]:
        counts[index] += 1
    return [dict(item, count=value) for item, value in zip(strata, counts)]


def robust_point_sigma(records):
    values = []
    for quarter in np.unique(records["quarter"]):
        flux = records["flux"][records["quarter"] == quarter].astype(np.float64)
        if len(flux) < 20:
            continue
        differences = np.diff(flux)
        median = np.median(differences)
        sigma = 1.4826 * np.median(np.abs(differences - median)) / math.sqrt(2)
        if np.isfinite(sigma) and sigma > 0:
            values.append(sigma)
    if not values:
        raise InjectionIneligible("No finite quarter point-noise estimate")
    return float(np.median(values))


def orbit(period_days, density_solar, impact, radius_ratio):
    period_seconds = period_days * 86400.0
    scaled_axis = (
        GRAVITATIONAL_CONSTANT
        * density_solar
        * RHO_SUN
        * period_seconds**2
        / (3.0 * math.pi)
    ) ** (1.0 / 3.0)
    if scaled_axis <= 1.0 + radius_ratio or impact >= scaled_axis:
        raise InjectionIneligible("Unphysical circular transit geometry")
    inclination = math.degrees(math.acos(impact / scaled_axis))
    numerator = (1.0 + radius_ratio) ** 2 - impact**2
    denominator = scaled_axis**2 - impact**2
    argument = math.sqrt(numerator / denominator)
    if not 0 < argument < 1:
        raise InjectionIneligible("Unphysical transit duration")
    duration_days = period_days * math.asin(argument) / math.pi
    return scaled_axis, inclination, duration_days


def event_centres(period, epoch, start, stop):
    first = math.floor((start - epoch) / period) - 1
    last = math.ceil((stop - epoch) / period) + 1
    return [(cycle, epoch + cycle * period) for cycle in range(first, last + 1)]


def truth_centres(truth, start, stop):
    if truth["category"] == "sparse_injection":
        return [truth["epoch_bkjd"]]
    linear = event_centres(truth["shape_period_days"], truth["epoch_bkjd"], start, stop)
    if truth["category"] != "ttv_injection":
        return [centre for _, centre in linear]
    ttv = truth["ttv"]
    return [
        centre
        + ttv["amplitude_days"]
        * math.sin(
            2.0 * math.pi * cycle / ttv["superperiod_cycles"] + ttv["phase_radians"]
        )
        for cycle, centre in linear
    ]


def transit_loss(times, centres, period, geometry, signal, periodic_epoch=None):
    radius_ratio, scaled_axis, inclination, duration_days = geometry
    loss = np.zeros(len(times), dtype=np.float64)
    params = batman.TransitParams()
    params.per = period
    params.rp = radius_ratio
    params.a = scaled_axis
    params.inc = inclination
    params.ecc = 0.0
    params.w = 90.0
    params.u = signal["limb_darkening"]
    params.limb_dark = "quadratic"
    if periodic_epoch is not None:
        params.t0 = periodic_epoch
        return 1.0 - batman.TransitModel(
            params,
            times,
            supersample_factor=signal["supersample_factor"],
            exp_time=signal["exposure_days"],
        ).light_curve(params)
    extent = max(1.5 * duration_days, 2.0 * signal["exposure_days"])
    centres = np.sort(np.asarray(centres, dtype=np.float64))
    if not len(centres):
        return loss
    right = np.searchsorted(centres, times)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(centres) - 1)
    nearest = np.where(
        np.abs(times - centres[left]) <= np.abs(times - centres[right]), left, right
    )
    relative_time = times - centres[nearest]
    use = np.abs(relative_time) <= extent
    if use.any():
        params.t0 = 0.0
        model = batman.TransitModel(
            params,
            relative_time[use],
            supersample_factor=signal["supersample_factor"],
            exp_time=signal["exposure_days"],
        ).light_curve(params)
        loss[use] = 1.0 - model
    return loss


def measure_signal(
    records,
    centres,
    period,
    density,
    impact,
    radius_ratio,
    signal,
    periodic_epoch=None,
):
    sigma = robust_point_sigma(records)
    scaled_axis, inclination, duration = orbit(period, density, impact, radius_ratio)
    geometry = (radius_ratio, scaled_axis, inclination, duration)
    loss = transit_loss(
        records["time"].astype(np.float64),
        centres,
        period,
        geometry,
        signal,
        periodic_epoch,
    )
    expected = float(np.linalg.norm(loss) / sigma)
    depth_ppm = float(loss.max() * 1e6)
    duration_hours = geometry[3] * 24.0
    if not signal["depth_ppm"][0] <= depth_ppm <= signal["depth_ppm"][1]:
        raise InjectionIneligible(
            "Applied transit depth is outside the detector domain"
        )
    if not signal["duration_hours"][0] <= duration_hours <= signal["duration_hours"][1]:
        raise InjectionIneligible(
            "Physical transit duration is outside the detector domain"
        )
    return sigma, expected, depth_ppm, geometry, loss


def observed_events(records, centres, duration_days, period=None, epoch=None):
    if period is not None:
        cycles = np.rint((records["time"] - epoch) / period).astype(np.int64)
        separation = np.abs(records["time"] - (epoch + cycles * period))
        inside = separation <= duration_days / 2.0
        quarters = set(records["quarter"][inside].tolist())
        return len(np.unique(cycles[inside])), quarters, int(inside.sum())
    events, quarters, points = 0, set(), 0
    for centre in centres:
        inside = np.abs(records["time"] - centre) <= duration_days / 2.0
        count = int(inside.sum())
        if count:
            events += 1
            points += count
            quarters.update(records["quarter"][inside].tolist())
    return events, quarters, points


def latent_parameters(source_id, category, stratum, plan):
    seed = plan["seed"]
    signal = plan["signal"]
    density = log_uniform(
        seed, *signal["stellar_density_solar"], category, source_id, "density"
    )
    impact = uniform(seed, *signal["impact_parameter"], category, source_id, "impact")
    radius_ratio = log_uniform(
        seed, *signal["radius_ratio"], category, source_id, "radius_ratio"
    )
    if category == "periodic_injection":
        specification = next(
            item
            for item in plan["cohorts"][category]["strata"]
            if item["name"] == stratum
        )
        period = log_uniform(
            seed, *specification["period_days"], category, source_id, "period"
        )
    elif category == "ttv_injection":
        period = log_uniform(
            seed,
            *plan["cohorts"][category]["period_days"],
            category,
            source_id,
            "period",
        )
    else:
        period = log_uniform(
            seed,
            *plan["cohorts"][category]["shape_period_days"],
            category,
            source_id,
            "period",
        )
    return period, density, impact, radius_ratio


def make_signal(source_id, pool_rank, category, stratum, records, plan):
    period, density, impact, radius_ratio = latent_parameters(
        source_id, category, stratum, plan
    )
    signal = plan["signal"]
    start, stop = map(float, (records["time"].min(), records["time"].max()))
    phase = unit(plan["seed"], category, source_id, "phase")
    epoch = 120.0 + phase * period
    linear = event_centres(period, epoch, start, stop)
    ttv = None
    if category == "sparse_injection":
        candidates = np.flatnonzero(
            (np.arange(len(records)) > 0)
            & (np.arange(len(records)) + 1 < len(records))
            & (np.roll(records["quarter"], 1) == records["quarter"])
            & (np.roll(records["quarter"], -1) == records["quarter"])
        )
        if not len(candidates):
            raise InjectionIneligible("No interior cadence for a sparse injection")
        chosen = candidates[min(int(phase * len(candidates)), len(candidates) - 1)]
        epoch = float(records["time"][chosen])
        centres = [epoch]
    elif category == "ttv_injection":
        base_geometry = orbit(period, density, impact, radius_ratio)
        amplitude = (
            uniform(
                plan["seed"],
                *plan["cohorts"][category]["amplitude_durations"],
                category,
                source_id,
                "amplitude",
            )
            * base_geometry[2]
        )
        if amplitude >= 0.15 * period:
            raise InjectionIneligible(
                "Requested strong TTV would reorder transit events"
            )
        superperiod = log_uniform(
            plan["seed"],
            *plan["cohorts"][category]["superperiod_cycles"],
            category,
            source_id,
            "superperiod",
        )
        ttv_phase = 2.0 * math.pi * unit(plan["seed"], category, source_id, "ttv_phase")
        centres = [
            centre
            + amplitude * math.sin(2.0 * math.pi * cycle / superperiod + ttv_phase)
            for cycle, centre in linear
        ]
        ttv = {
            "amplitude_days": amplitude,
            "amplitude_durations": amplitude / base_geometry[2],
            "superperiod_cycles": superperiod,
            "phase_radians": ttv_phase,
        }
    else:
        centres = [centre for _, centre in linear]

    periodic_epoch = epoch if category == "periodic_injection" else None
    sigma, expected, depth, geometry, loss = measure_signal(
        records,
        centres,
        period,
        density,
        impact,
        radius_ratio,
        signal,
        periodic_epoch,
    )
    events, quarters, points = observed_events(
        records,
        centres,
        geometry[3],
        period if periodic_epoch is not None else None,
        periodic_epoch,
    )
    if category == "sparse_injection":
        if points < signal["minimum_sparse_cadences"]:
            raise InjectionIneligible("Sparse injection has too few observed cadences")
    elif (
        events < signal["minimum_periodic_events"]
        or len(quarters) < signal["minimum_periodic_quarters"]
    ):
        raise InjectionIneligible(
            "Periodic injection has insufficient observed coverage"
        )
    truth = {
        "source_id": str(source_id),
        "pool_rank": int(pool_rank),
        "category": category,
        "stratum": stratum,
        "period_days": None if category == "sparse_injection" else period,
        "shape_period_days": period,
        "epoch_bkjd": epoch,
        "expected_white_mes": expected,
        "point_sigma_ppm": sigma * 1e6,
        "depth_ppm": depth,
        "duration_hours": geometry[3] * 24.0,
        "radius_ratio": geometry[0],
        "scaled_semimajor_axis": geometry[1],
        "inclination_degrees": geometry[2],
        "impact_parameter": impact,
        "stellar_density_solar": density,
        "observed_events": events,
        "observed_quarters": sorted(map(int, quarters)),
        "in_transit_cadences": points,
        "ttv": ttv,
    }
    return truth, loss


def delta_arrays(raw, truth, plan):
    signal = plan["signal"]
    geometry = (
        truth["radius_ratio"],
        truth["scaled_semimajor_axis"],
        truth["inclination_degrees"],
        truth["duration_hours"] / 24.0,
    )
    deltas = {}
    for time_key, quarter_key, flux_keys in DELTA_GROUPS:
        times = np.asarray(raw[time_key], dtype=np.float64)
        quarters = np.asarray(raw[quarter_key])
        loss = transit_loss(
            times,
            truth_centres(truth, float(times.min()), float(times.max())),
            truth["shape_period_days"],
            geometry,
            signal,
            truth["epoch_bkjd"] if truth["category"] == "periodic_injection" else None,
        )
        for flux_key in flux_keys:
            flux = np.asarray(raw[flux_key], dtype=np.float64)
            delta = np.zeros(len(flux), dtype=np.float32)
            for quarter in np.unique(quarters):
                use = quarters == quarter
                finite = use & np.isfinite(flux)
                if finite.any():
                    delta[use] = (-np.median(flux[finite]) * loss[use]).astype(
                        np.float32
                    )
            deltas[flux_key] = delta
    return deltas


def metadata_record(metadata):
    keys = (
        "source_id",
        "kic",
        "origin_tpf_kic",
        "ra_deg",
        "dec_deg",
        "phot_g_mean_mag",
        "raw_fits_sha256",
        "shard",
        "shard_path",
    )
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key in keys
        if (value := metadata.get(key)) is not None
    }


def frozen_inputs(plan):
    freeze_path = ROOT / plan["source_freeze"]
    source_path = ROOT / plan["source_pools"]
    freeze = read(freeze_path)
    if freeze.get("status") != "FROZEN_SOURCE_POOLS" or freeze["artifacts"].get(
        "source-pools.csv"
    ) != digest(source_path):
        raise ValueError("Injection plan does not bind the frozen source pools")
    return freeze_path, source_path, freeze


def generator_identity():
    paths = (
        Path(__file__),
        Path(__file__).with_name("run.py"),
        ROOT / "pipeline/photometry/lc2.py",
        ROOT / "pipeline/vetting/requirements.txt",
    )
    return {
        "files": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "batman_version": getattr(batman, "__version__", "UNKNOWN"),
        "numpy_version": np.__version__,
    }


def materialization_identity(
    plan_path, freeze_path, source_path, categories, limit_per_cohort
):
    return {
        "plan": {"path": str(plan_path), "sha256": digest(plan_path)},
        "source_freeze": {
            "path": str(freeze_path),
            "sha256": digest(freeze_path),
        },
        "source_pools": {"path": str(source_path), "sha256": digest(source_path)},
        "categories": categories,
        "limit_per_cohort": limit_per_cohort,
        "generator": generator_identity(),
    }


def read_jsonl(path, key):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {key(row): row for row in rows}


def validate_delta(path):
    expected = {flux_key for _, _, flux_keys in DELTA_GROUPS for flux_key in flux_keys}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(f"Injection delta has unexpected arrays: {path}")
        if any(array.ndim != 1 for array in archive.values()):
            raise ValueError(f"Injection delta contains a non-vector array: {path}")


def pool_work(pools, plan, categories, limit_per_cohort):
    work = []
    for category in categories:
        rows = pools[category]
        if limit_per_cohort is not None:
            rows = rows[: min(len(rows), max(100, 20 * limit_per_cohort))]
        schedule = None
        if category == "periodic_injection":
            strata = plan["cohorts"][category]["strata"]
            pool_strata = scaled_strata(strata, len(pools[category]))
            schedule = periodic_schedule(pool_strata, plan["seed"])
        for row in rows:
            index = int(row["pool_rank"]) - 1
            stratum = (
                schedule[index]
                if schedule is not None
                else category.removesuffix("_injection")
            )
            work.append({**row, "stratum": stratum})
    return work


def evaluate_eligibility(item, plan):
    sid, category = item["source_id"], item["category"]
    curve = load_curve(ROOT, sid, "psf_flat")
    records = prepare_curve_records(curve, mask_flatten=True)
    truth = None
    if category != "ordinary":
        truth, _ = make_signal(
            sid,
            item["pool_rank"],
            category,
            item["stratum"],
            records,
            plan,
        )
    return {
        "category": category,
        "pool_rank": int(item["pool_rank"]),
        "source_id": sid,
        "shard": int(item["shard"]),
        "status": "ELIGIBLE",
        "truth": truth,
    }


def evaluate_work(out, work, plan, progress, results):
    path = out / "eligibility.jsonl"
    if not path.exists():
        path.open("x").close()
    pending = [
        item
        for item in work
        if (item["category"], int(item["pool_rank"])) not in results
    ]
    pending.sort(
        key=lambda item: (
            int(item["shard"]),
            item["category"],
            int(item["pool_rank"]),
        )
    )
    with path.open("a") as stream:
        for index, item in enumerate(pending, 1):
            try:
                result = evaluate_eligibility(item, plan)
            except (InsufficientCadences, InjectionIneligible) as exc:
                result = {
                    "category": item["category"],
                    "pool_rank": int(item["pool_rank"]),
                    "source_id": item["source_id"],
                    "shard": int(item["shard"]),
                    "status": "INELIGIBLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            key = (result["category"], result["pool_rank"])
            results[key] = result
            stream.write(
                json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
            )
            if index % 100 == 0:
                stream.flush()
                progress.update(stage="ELIGIBILITY", eligibility_rows=len(results))
                write_json(out / "progress.json", progress)
                print(
                    json.dumps(
                        {
                            "stage": "ELIGIBILITY",
                            "complete": len(results),
                            "scheduled": len(results) + len(pending) - index,
                        }
                    ),
                    flush=True,
                )
    progress.update(stage="ELIGIBILITY_COMPLETE", eligibility_rows=len(results))
    write_json(out / "progress.json", progress)
    return results


def target_counts(plan, category, limit_per_cohort):
    target = plan["cohorts"][category]["target"]
    if limit_per_cohort is not None:
        target = min(target, limit_per_cohort)
    if category != "periodic_injection":
        return {category.removesuffix("_injection"): target}
    return {
        item["name"]: item["count"]
        for item in scaled_strata(plan["cohorts"][category]["strata"], target)
    }


def selection_state(results, plan, categories, limit_per_cohort):
    accepted = {category: [] for category in categories}
    deficits = {}
    for category in categories:
        remaining = target_counts(plan, category, limit_per_cohort)
        rows = sorted(
            (
                row
                for (found_category, _), row in results.items()
                if found_category == category and row["status"] == "ELIGIBLE"
            ),
            key=lambda row: row["pool_rank"],
        )
        for row in rows:
            stratum = (
                row["truth"]["stratum"]
                if row["truth"] is not None
                else category.removesuffix("_injection")
            )
            if remaining.get(stratum, 0) > 0:
                accepted[category].append(row)
                remaining[stratum] -= 1
        deficits[category] = {key: value for key, value in remaining.items() if value}
    return accepted, deficits


def select_accepted(results, plan, categories, limit_per_cohort):
    accepted, deficits = selection_state(results, plan, categories, limit_per_cohort)
    if any(deficits.values()):
        raise ValueError(f"Frozen pools lack eligible controls: {deficits}")
    return accepted


def evaluate_until_selected(out, work, plan, categories, limit_per_cohort, progress):
    results = read_jsonl(
        out / "eligibility.jsonl",
        lambda row: (row["category"], row["pool_rank"]),
    )
    category_work = {
        category: [item for item in work if item["category"] == category]
        for category in categories
    }
    prefixes = {}
    for category, rows in category_work.items():
        target = sum(target_counts(plan, category, limit_per_cohort).values())
        initial = len(rows) if limit_per_cohort is not None else max(100, 2 * target)
        prefixes[category] = min(len(rows), initial)

    while True:
        batch = [
            item
            for category, rows in category_work.items()
            for item in rows[: prefixes[category]]
            if (category, int(item["pool_rank"])) not in results
        ]
        results = evaluate_work(out, batch, plan, progress, results)
        accepted, deficits = selection_state(
            results, plan, categories, limit_per_cohort
        )
        incomplete = [category for category in categories if deficits[category]]
        if not incomplete:
            return results, accepted
        expanded = False
        for category in incomplete:
            target = sum(target_counts(plan, category, limit_per_cohort).values())
            new_prefix = min(
                len(category_work[category]),
                prefixes[category] + max(100, target // 2),
            )
            expanded |= new_prefix > prefixes[category]
            prefixes[category] = new_prefix
        if not expanded:
            raise ValueError(f"Frozen pools lack eligible controls: {deficits}")


def materialize_deltas(out, accepted, plan, progress, resume):
    path = out / "delta-ledger.jsonl"
    ledger = read_jsonl(path, lambda row: row["source_id"])
    if not resume:
        path.open("x").close()
    rows = [
        row
        for category, values in accepted.items()
        if category != "ordinary"
        for row in values
        if row["source_id"] not in ledger
    ]
    rows.sort(key=lambda row: (row["shard"], row["category"], row["pool_rank"]))
    with path.open("a") as stream:
        for index, row in enumerate(rows, 1):
            sid = row["source_id"]
            metadata, raw = load_source(ROOT, sid)
            delta_path = out / "deltas" / f"{sid}.npz"
            if delta_path.exists():
                validate_delta(delta_path)
            else:
                temporary = delta_path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **delta_arrays(raw, row["truth"], plan))
                temporary.replace(delta_path)
            record = {
                "source_id": sid,
                "metadata": metadata_record(metadata),
                "base_raw_fits_sha256": metadata["raw_fits_sha256"],
                "deltas": str(delta_path.relative_to(out)),
                "sha256": digest(delta_path),
            }
            ledger[sid] = record
            stream.write(
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
            if index % 100 == 0:
                stream.flush()
                progress.update(stage="DELTAS", delta_rows=len(ledger))
                write_json(out / "progress.json", progress)
                print(
                    json.dumps(
                        {
                            "stage": "DELTAS",
                            "complete": len(ledger),
                            "total": sum(
                                len(values)
                                for category, values in accepted.items()
                                if category != "ordinary"
                            ),
                        }
                    ),
                    flush=True,
                )
    progress.update(stage="DELTAS_COMPLETE", delta_rows=len(ledger))
    write_json(out / "progress.json", progress)
    return ledger


def materialize(
    out,
    plan_path=PLAN,
    categories=None,
    limit_per_cohort=None,
    resume=False,
):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    if plan.get("schema") != 1:
        raise ValueError("Unsupported injection plan schema")
    freeze_path, source_path, freeze = frozen_inputs(plan)
    categories = list(plan["cohorts"] if categories is None else categories)
    unknown = set(categories) - set(plan["cohorts"])
    if unknown:
        raise ValueError(f"Unknown control cohorts: {sorted(unknown)}")
    out = Path(out)
    identity = materialization_identity(
        plan_path, freeze_path, source_path, categories, limit_per_cohort
    )
    if resume:
        progress = read(out / "progress.json")
        if (
            progress.get("identity") != identity
            or (out / "materialization.json").exists()
        ):
            raise ValueError(
                "Materialization resume identity changed or run is complete"
            )
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "deltas").mkdir()
        progress = {"status": "RUNNING", "stage": "STARTING", "identity": identity}
        write_json(out / "progress.json", progress)
    with source_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    pools = {
        category: [row for row in rows if row["category"] == category]
        for category in categories
    }
    work = pool_work(pools, plan, categories, limit_per_cohort)
    results, accepted = evaluate_until_selected(
        out, work, plan, categories, limit_per_cohort, progress
    )
    ledger = materialize_deltas(out, accepted, plan, progress, resume)
    truths = [
        row["truth"]
        for category in categories
        for row in sorted(accepted[category], key=lambda item: item["pool_rank"])
        if row["truth"] is not None
    ]
    manifest_sources = {}
    for index, truth in enumerate(truths):
        record = dict(ledger[truth["source_id"]], truth_index=index)
        manifest_sources[truth["source_id"]] = record

    for category, values in accepted.items():
        write_json(
            out / f"{category}-source-ids.json",
            [row["source_id"] for row in values],
        )
    truth_path = out / "truth.jsonl"
    write_jsonl(truth_path, truths)
    audit_path = out / "periodic-audit.csv"
    fields = ("source_id", "period_days", "epoch_bkjd", "duration_hours")
    write_csv(
        audit_path,
        fields,
        (
            {key: truth[key] for key in fields}
            for truth in truths
            if truth["category"] == "periodic_injection"
        ),
    )
    manifest = {
        "schema": 2,
        "measurement_scope": plan["claim_boundary"],
        "transform": "additive_flux_delta",
        "protocol": plan["protocol"],
        "generator": identity["generator"],
        "truth": {"path": truth_path.name, "sha256": digest(truth_path)},
        "sources": manifest_sources,
    }
    manifest_path = out / "input-manifest.json"
    write_json(manifest_path, manifest)
    artifacts = {
        path.name: digest(path)
        for path in out.iterdir()
        if path.is_file() and path.name not in {"materialization.json", "progress.json"}
    }
    report = {
        "schema": 1,
        "status": "FROZEN" if limit_per_cohort is None else "PROBE",
        "protocol": plan["protocol"],
        **identity,
        "source_index_sha256": freeze["source_index"]["sha256"],
        "accepted": {key: len(value) for key, value in accepted.items()},
        "accepted_strata": dict(Counter(truth["stratum"] for truth in truths)),
        "attempt_failures": {
            category: dict(
                sorted(
                    Counter(
                        row["reason"]
                        for (found_category, _), row in results.items()
                        if found_category == category and row["status"] == "INELIGIBLE"
                    ).items()
                )
            )
            for category in categories
        },
        "truth_rows": len(truths),
        "delta_bytes": sum(path.stat().st_size for path in (out / "deltas").iterdir()),
        "artifacts": artifacts,
        "claim_boundary": plan["claim_boundary"],
    }
    write_json(out / "materialization.json", report)
    progress.update(status="COMPLETE", stage="FROZEN")
    write_json(out / "progress.json", progress)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--limit-per-cohort", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit_per_cohort is not None and args.limit_per_cohort < 1:
        raise ValueError("Probe cohort limit must be positive")
    print(
        json.dumps(
            materialize(
                args.out,
                args.plan,
                args.categories,
                args.limit_per_cohort,
                args.resume,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
