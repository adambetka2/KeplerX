"""Materialize three source-coherent false-class controls in LC2 and target pixels."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

import lc2
from deterministic import injections, pixel_injections
from lc2 import load_source, photometry

POOL = ROOT / "dump/false-class-controls-v5"
PLAN = Path(__file__).with_name("false-class-control-plan.json")
MODEL = "synthetic-false-classes-v2"
QUARTERS_PER_TRIAL = 4
MINIMUM_EXPECTED_WHITE_MES = 8.0
MINIMUM_EXPECTED_SEARCH_MES = 14.0


class ControlUnavailable(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def uniform(seed, category, source_id, name, low, high):
    score = hashlib.sha256(f"{seed}:{category}:{source_id}:{name}".encode()).digest()
    fraction = (int.from_bytes(score[:8], "big") + 0.5) / 2**64
    return low + (high - low) * fraction


def select_quarters(raw):
    time = np.asarray(raw["time_bkjd"], dtype=float)
    quarter = np.asarray(raw["quarter"])
    flux = np.asarray(raw[injections.PRIMARY_FLUX], dtype=float)
    supported = [
        int(q)
        for q in np.unique(quarter)
        if np.count_nonzero((quarter == q) & np.isfinite(time + flux)) >= 1000
    ]
    if len(supported) < QUARTERS_PER_TRIAL:
        raise ControlUnavailable("Fewer than four well-observed quarters")
    positions = (
        np.linspace(0, len(supported) - 1, QUARTERS_PER_TRIAL).round().astype(int)
    )
    return [supported[index] for index in positions]


def trapezoid(offset, duration):
    half = duration / 2
    ingress = duration / 5
    return np.clip((half - np.abs(offset)) / ingress, 0, 1)


def waveform(times, truth):
    time = np.asarray(times, dtype=float)
    finite = np.isfinite(time)
    time = np.where(finite, time, truth["epoch_bkjd"] + truth["period_days"] / 3)
    period = truth["period_days"]
    primary = (time - truth["epoch_bkjd"] + period / 2) % period - period / 2
    shape = trapezoid(primary, truth["duration_days"])
    if truth["category"] == "instrumental_artifact":
        phase = (time - truth["epoch_bkjd"]) % period
        kind = truth["artifact_kind"]
        if kind == "column_pulse":
            return truth["depth_fraction"] * np.where(finite, shape, 0)
        decay = truth["duration_days"] * (4 if kind == "common_mode_recovery" else 1)
        tail = np.exp(-phase / decay)
        return truth["depth_fraction"] * np.where(
            finite & (phase <= 4 * decay), tail, 0
        )
    if truth["category"] == "eclipsing_binary":
        secondary = (primary + period) % period - period / 2
        return truth["depth_fraction"] * np.where(
            finite,
            (
                shape
                + truth["secondary_depth_ratio"]
                * trapezoid(secondary, truth["duration_days"])
            ),
            0,
        )
    return truth["depth_fraction"] * np.where(finite, shape, 0)


def trial_truth(row, raw, seed):
    category, source_id = row["category"], row["source_id"]
    quarters = select_quarters(raw)
    period = uniform(seed, category, source_id, "period", 2.5, 16.0)
    duration = uniform(seed, category, source_id, "duration", 0.08, 0.25)
    primary_quarter = quarters[0]
    first = np.asarray(raw["time_bkjd"])[np.asarray(raw["quarter"]) == primary_quarter]
    epoch = float(np.nanmin(first)) + period * uniform(
        seed, category, source_id, "phase", 0.1, 0.9
    )
    depth_limits = {
        "instrumental_artifact": (0.001, 0.006),
        "eclipsing_binary": (0.008, 0.08),
        "background_contaminant": (0.05, 0.35),
    }
    depth = uniform(seed, category, source_id, "depth", *depth_limits[category])
    artifact_kind = (
        (
            "stripe_exponential",
            "column_pulse",
            "common_mode_recovery",
        )[(int(row["pool_rank"]) - 1) % 3]
        if category == "instrumental_artifact"
        else None
    )
    return {
        "category": category,
        "source_id": source_id,
        "parent_tpf": row["parent_tpf"],
        "contaminant_source_id": row["contaminant_source_id"] or None,
        "period_days": period,
        "epoch_bkjd": epoch,
        "duration_days": duration,
        "depth_fraction": depth,
        "artifact_kind": artifact_kind,
        "secondary_depth_ratio": uniform(
            seed, category, source_id, "secondary", 0.3, 1.0
        )
        if category == "eclipsing_binary"
        else None,
        "injected_quarters": quarters,
        "mechanism": {
            "instrumental_artifact": f"periodic detector {artifact_kind}",
            "eclipsing_binary": "target-source primary and secondary eclipses",
            "background_contaminant": "transit-like eclipse on a distinct Gaia source in the same TPF",
        }[category],
    }


def astrometry(source):
    return {
        "ra": source["ra_deg"],
        "dec": source["dec_deg"],
        "pmra": None,
        "pmdec": None,
    }


def transfer(target_profile, injected_profile):
    denominator = float(np.sum(target_profile**2))
    if denominator <= 0:
        raise ControlUnavailable("Target PRF is degenerate")
    value = float(np.sum(target_profile * injected_profile) / denominator)
    if value <= 0 or not math.isfinite(value):
        raise ControlUnavailable("Injected pixel pattern has no target coupling")
    return value


def stripe_profile(target_profile):
    column = int(np.unravel_index(np.argmax(target_profile), target_profile.shape)[1])
    stripe = np.zeros_like(target_profile)
    stripe[:, column] = 1 / target_profile.shape[0]
    return stripe


def pixel_pattern(hdus, target, companion, truth):
    target_profile, calibration, target_pixel, target_sky, position_status, inside = (
        pixel_injections.pixel_profile(hdus, astrometry(target))
    )
    if not inside:
        raise ControlUnavailable("Target is outside the selected TPF stamp")
    category = truth["category"]
    injected_pixel = target_pixel if category == "eclipsing_binary" else None
    injected_sky = target_sky if category == "eclipsing_binary" else None
    if category == "eclipsing_binary":
        injected_profile = target_profile
    elif category == "instrumental_artifact":
        injected_profile = (
            np.full_like(target_profile, 1 / target_profile.size)
            if truth["artifact_kind"] == "common_mode_recovery"
            else stripe_profile(target_profile)
        )
    else:
        (
            injected_profile,
            companion_calibration,
            injected_pixel,
            injected_sky,
            _,
            inside,
        ) = pixel_injections.pixel_profile(hdus, astrometry(companion))
        if companion_calibration != calibration:
            raise ValueError("Target and contaminant PRF calibrations differ")
        if not inside:
            raise ControlUnavailable("Contaminating source is outside the TPF stamp")
    return (
        injected_profile,
        transfer(target_profile, injected_profile),
        {
            "prf_calibration": calibration,
            "target_pixel": target_pixel,
            "target_sky_deg": target_sky,
            "injected_pixel": injected_pixel,
            "injected_sky_deg": injected_sky,
            "target_astrometry": position_status,
        },
    )


def required_source_flux(raw, quarter, source):
    try:
        return pixel_injections.source_flux(raw, quarter)
    except ValueError as error:
        raise ControlUnavailable(
            f"{source} has no usable flux in quarter {quarter}: {error}"
        ) from error


def quarter_plan(tpfs, target, companion, truth, raw, companion_raw):
    result = {}
    for quarter, path in sorted(tpfs.items()):
        with fits.open(path, memmap=False) as hdus:
            if int(hdus[0].header["KEPLERID"]) != int(truth["parent_tpf"]):
                raise ControlUnavailable(
                    "Resolved TPF parent differs from frozen parent"
                )
            profile, coupling, provenance = pixel_pattern(
                hdus, target, companion, truth
            )
        target_flux = required_source_flux(raw, quarter, "Target")
        if truth["category"] == "background_contaminant":
            amplitude = required_source_flux(companion_raw, quarter, "Contaminant")
            flux_ratio = amplitude * coupling / target_flux
        elif truth["category"] == "instrumental_artifact":
            amplitude = target_flux / coupling
            flux_ratio = 1.0
        else:
            amplitude = target_flux
            flux_ratio = 1.0
        result[quarter] = {
            "path": path,
            "profile": profile,
            "coupling": coupling,
            "amplitude": amplitude,
            "flux_ratio": flux_ratio,
            **provenance,
        }
    return result


def delta_arrays(raw, truth, quarters):
    deltas = {}
    for time_key, quarter_key, flux_keys in injections.DELTA_GROUPS:
        times = np.asarray(raw[time_key], dtype=float)
        numbers = np.asarray(raw[quarter_key])
        loss = waveform(times, truth)
        for flux_key in flux_keys:
            flux = np.asarray(raw[flux_key], dtype=float)
            delta = np.zeros(len(flux), dtype=np.float32)
            for quarter, specification in quarters.items():
                use = numbers == quarter
                finite = use & np.isfinite(flux)
                if finite.any():
                    baseline = float(np.median(flux[finite]))
                    delta[use] = (
                        -baseline * specification["flux_ratio"] * loss[use]
                    ).astype(np.float32)
            deltas[flux_key] = delta
    return deltas


def excerpt_arrays(raw, deltas, quarters):
    masks = {
        "primary": np.isin(raw["quarter"], quarters),
        "qdiag": np.isin(raw["qdiag_quarter"], quarters),
    }
    return {
        name: (
            np.asarray(values, dtype=np.float64) + deltas[name]
            if name in deltas
            else np.asarray(values)
        )[masks["qdiag" if name.startswith("qdiag_") else "primary"]]
        for name, values in raw.items()
    }


def expected_white_mes(raw, delta):
    flux = np.asarray(raw[injections.PRIMARY_FLUX], dtype=float)
    quarter = np.asarray(raw["quarter"])
    terms = []
    for number in np.unique(quarter):
        use = (quarter == number) & np.isfinite(flux)
        values = flux[use]
        if len(values) < 20:
            continue
        differences = np.diff(values)
        sigma = (
            1.4826
            * np.median(np.abs(differences - np.median(differences)))
            / math.sqrt(2)
        )
        if sigma > 0 and math.isfinite(sigma):
            terms.append(float(np.sum((delta[use] / sigma) ** 2)))
    return math.sqrt(sum(terms))


def expected_search_mes(raw, truth, quarters):
    records = injections.prepare_curve_records(
        photometry(raw, "psf_flat"), mask_flatten=True
    )
    terms = []
    for quarter, specification in quarters.items():
        use = records["quarter"] == quarter
        flux = np.asarray(records["flux"][use], dtype=float)
        if len(flux) < 100:
            raise ControlUnavailable(
                "Prepared quarter is too short for search-noise calibration"
            )
        residual = flux - median_filter(flux, size=97, mode="nearest")
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if not 0 < sigma < math.inf:
            raise ControlUnavailable(
                "Prepared quarter has no finite search-noise scale"
            )
        relative_signal = specification["flux_ratio"] * waveform(
            records["time"][use], truth
        )
        terms.append(float(np.sum((relative_signal / sigma) ** 2)))
    return math.sqrt(sum(terms))


def write_pixel(base, target, source_id, truth, quarter, specification):
    with fits.open(base, memmap=False) as hdus:
        hdus.verify("exception")
        if int(hdus[0].header["QUARTER"]) != quarter:
            raise ValueError("TPF quarter identity changed")
        flux = hdus[1].data["FLUX"]
        error = np.array(hdus[1].data["FLUX_ERR"], copy=True)
        loss = waveform(np.asarray(hdus[1].data["TIME"], dtype=float), truth)
        hdus[1].data["FLUX"] = pixel_injections.inject_flux(
            flux, specification["profile"], loss, specification["amplitude"]
        )
        if not np.array_equal(hdus[1].data["FLUX_ERR"], error, equal_nan=True):
            raise ValueError("TPF errors changed during false-class injection")
        hdus[0].header["FSCSRC"] = (source_id, "Controlled LC2 source")
        hdus[0].header["FSCMODEL"] = (MODEL, "Forgotten Stars control")
        pixel_injections.deterministic_fits(target, hdus)
    with fits.open(target, memmap=False, checksum=True) as hdus:
        hdus.verify("exception")


def validate_pixel(path, source_id, quarter):
    with fits.open(path, memmap=False, checksum=True) as hdus:
        hdus.verify("exception")
        if (
            str(hdus[0].header.get("FSCSRC")) != source_id
            or hdus[0].header.get("FSCMODEL") != MODEL
            or int(hdus[0].header["QUARTER"]) != quarter
        ):
            raise ValueError(f"Existing controlled pixel identity changed: {path}")


def materialize_trial(out, row, plan, mapping, existing):
    source_id = row["source_id"]
    target, raw = load_source(ROOT, source_id)
    if target["raw_fits_sha256"] != row["raw_fits_sha256"]:
        raise ValueError(f"Frozen raw source changed: {source_id}")
    companion = companion_raw = None
    if row["contaminant_source_id"]:
        companion, companion_raw = load_source(ROOT, row["contaminant_source_id"])
        if companion["raw_fits_sha256"] != row["contaminant_raw_fits_sha256"]:
            raise ValueError(f"Frozen contaminant changed: {source_id}")
        if companion["origin_tpf_kic"] != target["origin_tpf_kic"]:
            raise ValueError("Background pair no longer shares its TPF parent")
    truth = trial_truth(row, raw, plan["seed"])
    query_path = out / "spatial-tpf-queries" / f"{source_id}.json"
    tpfs = pixel_injections.available_tpfs(
        target,
        astrometry(target),
        truth["injected_quarters"],
        mapping,
        existing,
        query_path,
    )
    pixel_injections.prefetch_prfs(
        fits.getheader(path, 0)["CHANNEL"] for path in tpfs.values()
    )
    quarters = quarter_plan(tpfs, target, companion, truth, raw, companion_raw)
    search_mes = expected_search_mes(raw, truth, quarters)
    if search_mes <= 0:
        raise ControlUnavailable("Injected signal has no measurable LC2 support")
    mes_range = (
        (160.0, 250.0)
        if truth["category"] == "instrumental_artifact"
        else (90.0, 120.0)
    )
    intended_mes = uniform(
        plan["seed"], truth["category"], source_id, "expected_search_mes", *mes_range
    )
    truth["depth_fraction"] *= intended_mes / search_mes
    maximum_depth = {
        "instrumental_artifact": 0.1,
        "eclipsing_binary": 0.12,
        "background_contaminant": 0.5,
    }[truth["category"]]
    if not 0 < truth["depth_fraction"] <= maximum_depth:
        raise ControlUnavailable(
            "Required false-class depth is outside its declared range"
        )
    deltas = delta_arrays(raw, truth, quarters)
    white_mes = expected_white_mes(raw, deltas[injections.PRIMARY_FLUX])
    search_mes = expected_search_mes(raw, truth, quarters)
    if (
        white_mes < MINIMUM_EXPECTED_WHITE_MES
        or search_mes < MINIMUM_EXPECTED_SEARCH_MES
    ):
        raise ControlUnavailable("Injected signal falls below its declared MES floor")
    truth["intended_search_mes"] = intended_mes
    truth["expected_search_mes"] = search_mes
    truth["expected_white_mes"] = white_mes
    truth["raw_fits_sha256"] = target["raw_fits_sha256"]
    truth["contaminant_raw_fits_sha256"] = (
        companion["raw_fits_sha256"] if companion is not None else None
    )
    truth["pixel_parent_tpf"] = row["parent_tpf"]
    truth["flux_ratio_by_quarter"] = {
        str(quarter): specification["flux_ratio"]
        for quarter, specification in quarters.items()
    }
    pixel_folder = out / "pixels" / source_id
    pixel_folder.mkdir(parents=True, exist_ok=True)
    pixels = {}
    for quarter, specification in quarters.items():
        path = pixel_folder / f"q{quarter:02d}.fits.gz"
        if not path.exists():
            write_pixel(
                specification["path"], path, source_id, truth, quarter, specification
            )
        validate_pixel(path, source_id, quarter)
        pixels[str(quarter)] = {
            "path": os.path.relpath(path, out),
            "sha256": pixel_injections.digest(path),
            "base_path": str(specification["path"]),
            "base_sha256": pixel_injections.digest(specification["path"]),
            "base_parent_kic": int(row["parent_tpf"]),
            "prf_coupling": specification["coupling"],
            "prf_calibration": specification["prf_calibration"],
            "target_pixel": specification["target_pixel"],
            "target_sky_deg": specification["target_sky_deg"],
            "injected_pixel": specification["injected_pixel"],
            "injected_sky_deg": specification["injected_sky_deg"],
            "target_astrometry": specification["target_astrometry"],
        }
    arrays = excerpt_arrays(raw, deltas, truth["injected_quarters"])
    array_path = out / "arrays" / f"{source_id}.npz"
    if not array_path.exists():
        temporary = array_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(array_path)
    with np.load(array_path, allow_pickle=False) as stored:
        if set(stored.files) != set(arrays) or any(
            not np.array_equal(stored[name], values, equal_nan=True)
            for name, values in arrays.items()
        ):
            raise ValueError(f"Existing false-class arrays changed: {source_id}")
    return {
        "truth": truth,
        "manifest": {
            "metadata": injections.metadata_record(target),
            "base_raw_fits_sha256": target["raw_fits_sha256"],
            "arrays": os.path.relpath(array_path, out),
            "sha256": pixel_injections.digest(array_path),
            "pixels": pixels,
        },
    }


def pool_rows(pool, plan, category):
    freeze = read(pool / "freeze.json")
    if freeze["status"] != "CANDIDATE_POOL_FROZEN":
        raise ValueError("False-class candidate pool is not frozen")
    if freeze["inputs"]["plan"]["sha256"] != pixel_injections.digest(PLAN):
        raise ValueError("False-class plan changed since candidate selection")
    path = pool / "candidate-pool.csv"
    if freeze["artifacts"][path.name] != pixel_injections.digest(path):
        raise ValueError("False-class candidate identities changed")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if category is not None:
        rows = [row for row in rows if row["category"] == category]
    if len({row["parent_tpf"] for row in rows}) != len(rows):
        raise ValueError("False-class candidates share a parent TPF")
    return rows


def generator_identity():
    return {
        str(Path(path).relative_to(ROOT)): pixel_injections.digest(path)
        for path in (
            __file__,
            pixel_injections.__file__,
            injections.__file__,
            lc2.__file__,
        )
    }


def verified_development_probe(path, pool, plan):
    report = read(path / "materialization.json")
    audit = read(path / "parent-audit.json")
    expected_count = plan["development_per_class"]
    if (
        report["status"] != "PROBE"
        or report["candidate_pool_sha256"]
        != pixel_injections.digest(pool / "freeze.json")
        or report["counts"] != dict.fromkeys(plan["classes"], expected_count)
    ):
        raise ValueError("Development seed is not the audited candidate-pool probe")
    expected_partition = {"development": expected_count, "evaluation": 0}
    for category in plan["classes"]:
        if report["partition_counts"][category] != expected_partition:
            raise ValueError("Development seed partition counts changed")
    for name, expected in report["artifacts"].items():
        if pixel_injections.digest(path / name) != expected:
            raise ValueError(f"Development seed artifact changed: {name}")
    if (
        audit["status"] != "PASS"
        or audit["sources"] != expected_count * len(plan["classes"])
        or audit["distinct_resolved_parents"] != audit["sources"]
        or any(audit["reference_parent_overlaps"].values())
        or audit["materialization_sha256"]
        != pixel_injections.digest(path / "materialization.json")
        or audit["candidate_pool_sha256"] != report["candidate_pool_sha256"]
    ):
        raise ValueError("Development seed is not the audited candidate-pool probe")
    return report


def development_prefix(rows, category, accepted, skipped, expected_count):
    prefix = []
    for row in (row for row in rows if row["category"] == category):
        source_id = row["source_id"]
        if source_id in skipped:
            if skipped[source_id]["category"] != category:
                raise ValueError(f"Development skip category changed: {source_id}")
            continue
        if source_id not in accepted:
            raise ValueError(
                f"Development selection has an unrecorded trial: {source_id}"
            )
        if accepted[source_id]["category"] != category:
            raise ValueError(f"Development source category changed: {source_id}")
        prefix.append(source_id)
        if len(prefix) == expected_count:
            break
    return prefix


def development_seed(path, pool, rows, plan):
    path = Path(path)
    report = verified_development_probe(path, pool, plan)
    expected_count = plan["development_per_class"]
    truth = read_jsonl(path / "truth.jsonl")
    source_ids = read(path / "source-ids.json")
    manifest = read(path / "input-manifest.json")["sources"]
    unavailable = read_jsonl(path / "unavailable.jsonl")
    if (
        source_ids != [item["source_id"] for item in truth]
        or len(source_ids) != len(set(source_ids))
        or set(source_ids) != set(manifest)
        or len(unavailable) != len({item["source_id"] for item in unavailable})
        or set(source_ids) & {item["source_id"] for item in unavailable}
    ):
        raise ValueError("Development seed identities are incomplete")
    accepted = {item["source_id"]: item for item in truth}
    skipped = {item["source_id"]: item for item in unavailable}
    by_category = {}
    for category in plan["classes"]:
        expected = [item["source_id"] for item in truth if item["category"] == category]
        prefix = development_prefix(rows, category, accepted, skipped, expected_count)
        if prefix != expected:
            raise ValueError(f"Development source order changed: {category}")
        by_category[category] = expected
    return {
        "binding": {
            "path": str((path / "materialization.json").resolve()),
            "sha256": pixel_injections.digest(path / "materialization.json"),
            "parent_audit_sha256": pixel_injections.digest(path / "parent-audit.json"),
            "generator": report["generator"],
        },
        "accepted": accepted,
        "manifest": manifest,
        "by_category": by_category,
        "unavailable": unavailable,
    }


def start_materialization(out, identity, resume):
    if resume:
        progress = read(out / "progress.json")
        if progress["status"] != "RUNNING" or progress["identity"] != identity:
            raise ValueError("False-class resume identity changed")
        progress["complete"] = len(list((out / "records").glob("*.json")))
    else:
        out.mkdir(parents=True, exist_ok=False)
        for name in (
            "records",
            "unavailable-records",
            "pixels",
            "arrays",
            "spatial-tpf-queries",
        ):
            (out / name).mkdir()
        progress = {"status": "RUNNING", "identity": identity, "complete": 0}
        write(out / "progress.json", progress)
    return progress


def stored_unavailable(out, source_id, control_class):
    path = out / "unavailable-records" / f"{source_id}.json"
    if not path.exists():
        return None
    if (out / "records" / f"{source_id}.json").exists():
        raise ValueError(f"False-class trial has two outcomes: {source_id}")
    skipped = read(path)
    if skipped["source_id"] != source_id or skipped["category"] != control_class:
        raise ValueError(f"Unavailable false-class trial changed: {source_id}")
    return skipped


def collect_class(
    out, rows, plan, control_class, target, mapping, existing, progress, seed=None
):
    accepted = []
    unavailable = []
    for row in (row for row in rows if row["category"] == control_class):
        if len(accepted) == target:
            break
        source_id = row["source_id"]
        path = out / "records" / f"{source_id}.json"
        unavailable_path = out / "unavailable-records" / f"{source_id}.json"
        skipped = stored_unavailable(out, source_id, control_class)
        if skipped is not None:
            unavailable.append(skipped)
            continue
        already_stored = path.exists()
        if already_stored:
            record = read(path)
            if (
                not pixel_injections.validate_record(
                    {"source_id": source_id, "pixels": record["manifest"]["pixels"]},
                    out,
                    source_id,
                    record["truth"]["injected_quarters"],
                )
                or pixel_injections.digest(out / record["manifest"]["arrays"])
                != record["manifest"]["sha256"]
            ):
                raise ValueError(f"Completed false-class record changed: {source_id}")
        else:
            try:
                record = materialize_trial(out, row, plan, mapping, existing)
            except (ControlUnavailable, pixel_injections.PixelUnavailable) as error:
                if seed and len(accepted) < plan["development_per_class"]:
                    raise ValueError(
                        f"Frozen development trial became unavailable: {source_id}"
                    ) from error
                skipped = {
                    "source_id": source_id,
                    "category": control_class,
                    "reason": str(error),
                }
                write(unavailable_path, skipped)
                unavailable.append(skipped)
                continue
            write(path, record)
        record["truth"]["partition"] = (
            "development"
            if len(accepted) < plan["development_per_class"]
            else "evaluation"
        )
        if seed and len(accepted) < plan["development_per_class"]:
            expected_id = seed["by_category"][control_class][len(accepted)]
            if source_id != expected_id:
                raise ValueError(
                    f"Frozen development source changed: {source_id} != {expected_id}"
                )
            if (
                record["truth"] != seed["accepted"][source_id]
                or record["manifest"] != seed["manifest"][source_id]
            ):
                raise ValueError(f"Frozen development trial changed: {source_id}")
        accepted.append(record)
        if not already_stored:
            progress["complete"] += 1
        write(out / "progress.json", progress)
        print(
            json.dumps(
                {"category": control_class, "complete": len(accepted), "target": target}
            ),
            flush=True,
        )
    if len(accepted) != target:
        raise ValueError(
            f"Candidate pool cannot fill {control_class}: {len(accepted)}/{target}"
        )
    return accepted, unavailable


def finish_materialization(
    out, pool, accepted, unavailable, selected_classes, frozen, seed=None
):
    if len({record["truth"]["parent_tpf"] for record in accepted}) != len(accepted):
        raise ValueError("Materialized false-class trials share a parent TPF")
    partition_counts = {
        category: {
            partition: sum(
                record["truth"]["category"] == category
                and record["truth"]["partition"] == partition
                for record in accepted
            )
            for partition in ("development", "evaluation")
        }
        for category in selected_classes
    }
    if frozen:
        plan = read(PLAN)
        expected = {
            "development": plan["development_per_class"],
            "evaluation": plan["evaluation_per_class"],
        }
        if any(counts != expected for counts in partition_counts.values()):
            raise ValueError("False-class evaluation quotas are incomplete")
        kinds = [
            record["truth"]["artifact_kind"]
            for record in accepted
            if record["truth"]["category"] == "instrumental_artifact"
            and record["truth"]["partition"] == "evaluation"
        ]
        if any(kinds.count(kind) < 100 for kind in set(kinds)) or len(set(kinds)) != 3:
            raise ValueError("Instrument artifact morphologies are underrepresented")
    truth_path = out / "truth.jsonl"
    with truth_path.open("x") as stream:
        for record in accepted:
            stream.write(json.dumps(record["truth"], allow_nan=False) + "\n")
    with (out / "unavailable.jsonl").open("x") as stream:
        for row in unavailable:
            stream.write(json.dumps(row) + "\n")
    source_ids = [record["truth"]["source_id"] for record in accepted]
    write(out / "source-ids.json", source_ids)
    for partition in ("development", "evaluation"):
        write(
            out / f"{partition}-source-ids.json",
            [
                record["truth"]["source_id"]
                for record in accepted
                if record["truth"]["partition"] == partition
            ],
        )
    manifest = {
        "schema": 1,
        "protocol": MODEL,
        "measurement_scope": "Four observed-quarter excerpts with source-coherent synthetic instrumental stripe, on-target eclipsing binary, and offset-source contaminant signals in LC2 and target pixels; candidate-level false-class rates require an independent detector and vetter replay.",
        "sources": {
            record["truth"]["source_id"]: record["manifest"] for record in accepted
        },
    }
    write(out / "input-manifest.json", manifest)
    report = {
        "schema": 1,
        "status": "FROZEN" if frozen else "PROBE",
        "protocol": MODEL,
        "candidate_pool_sha256": pixel_injections.digest(pool / "freeze.json"),
        "generator": generator_identity(),
        "sources": len(accepted),
        "counts": {
            name: sum(row["truth"]["category"] == name for row in accepted)
            for name in selected_classes
        },
        "partition_counts": partition_counts,
        "unavailable": len(unavailable),
        "seeded_unavailable": len(seed["unavailable"]) if seed else 0,
        "artifacts": {
            name: pixel_injections.digest(out / name)
            for name in (
                "truth.jsonl",
                "source-ids.json",
                "development-source-ids.json",
                "evaluation-source-ids.json",
                "input-manifest.json",
                "unavailable.jsonl",
            )
        },
        "claim_boundary": manifest["measurement_scope"],
    }
    if seed:
        report["development_seed"] = seed["binding"]
    write(out / "materialization.json", report)
    return report


def materialize(
    out, pool=POOL, category=None, limit=None, resume=False, development_seed_path=None
):
    out, pool = Path(out), Path(pool)
    plan = read(PLAN)
    rows = pool_rows(pool, plan, category)
    selected_classes = [category] if category else plan["classes"]
    quota = plan["development_per_class"] + plan["evaluation_per_class"]
    target = limit if limit is not None else quota
    if target < 1 or target > quota:
        raise ValueError("Invalid false-class materialization limit")
    frozen = category is None and limit is None
    if frozen != (development_seed_path is not None):
        raise ValueError("Only the complete frozen cohort takes a development seed")
    seed = development_seed(development_seed_path, pool, rows, plan) if frozen else None
    identity = {
        "pool_sha256": pixel_injections.digest(pool / "freeze.json"),
        "plan_sha256": pixel_injections.digest(PLAN),
        "generator": generator_identity(),
        "category": category,
        "limit": limit,
        "development_seed": seed["binding"] if seed else None,
    }
    progress = start_materialization(out, identity, resume)
    if seed:
        for skipped in seed["unavailable"]:
            path = out / "unavailable-records" / f"{skipped['source_id']}.json"
            if path.exists() and read(path) != skipped:
                raise ValueError(f"Frozen development skip changed: {path.name}")
            write(path, skipped)
    mapping, existing = pixel_injections.indexed_pixels()
    accepted, unavailable = [], []
    for control_class in selected_classes:
        complete, skipped = collect_class(
            out, rows, plan, control_class, target, mapping, existing, progress, seed
        )
        accepted.extend(complete)
        unavailable.extend(skipped)
    report = finish_materialization(
        out,
        pool,
        accepted,
        unavailable,
        selected_classes,
        frozen,
        seed,
    )
    progress["status"] = "COMPLETE"
    write(out / "progress.json", progress)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pool", default=POOL, type=Path)
    parser.add_argument("--category", choices=read(PLAN)["classes"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development-seed", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.out,
                args.pool,
                args.category,
                args.limit,
                args.resume,
                args.development_seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
