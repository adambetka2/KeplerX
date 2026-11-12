"""Frozen pixel operators, target-only injections and explicit electron budgets."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear

POLICY = {
    "schema": 1,
    "span_days": 45.0,
    "weights": "Fixed median-image PRF and provided-error diagonal variance",
    "scene": "Nonnegative catalogue-source amplitudes plus signed constant background",
    "injection": "Subtract target_total_rate * time-dependent PRF * physical loss",
    "noise": "Retain observed noise; no fresh photon-noise draw after injection",
    "aperture_match_relative_tolerance": 1e-5,
    "pixel_region": "Aperture union target +/- 7.5 pixels, clipped to stamp",
    "prf_support": "Clip negative lobes; normalize full 17x17 support before cropping",
    "prf_field": "One fixed local mission calibration; translate by quality0 POS_CORR",
    "limits": [
        "Provided-error propagation is formal; empirical search nulls are separate",
        "Catalogue scene and fixed PRF weights are conditional, not a pixel optimum",
        "Signed unexplained variance is retained, not labelled read noise",
        "No irreducible information-bound claim",
    ],
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def finite_summary(values):
    values = np.asarray(values)
    good = values[np.isfinite(values)]
    if not len(good):
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return dict(
        zip(("p10", "median", "p90"), np.quantile(good, [0.1, 0.5, 0.9]).tolist()),
        count=len(good),
    )


def header_metadata(hdus):
    names = (
        "INT_TIME",
        "READTIME",
        "FRAMETIM",
        "NUM_FRM",
        "TIMEDEL",
        "GAIN",
        "READNOIS",
        "DEADC",
        "BJDREFI",
        "BJDREFF",
        "CHANNEL",
        "QUARTER",
    )
    result = {}
    for name in names:
        found = next((h.header for h in hdus if name in h.header), None)
        if found is not None:
            result[name] = {"value": found[name], "comment": found.comments[name]}
    return result


def read_tpf(case, span_days):
    with fits.open(case["tpf_path"], memmap=False) as hdus:
        table, header = hdus[1].data, hdus[1].header
        time = np.asarray(table["TIME"], dtype=float)
        start = float(case.get("start_bkjd", np.nanmin(time)))
        use = np.isfinite(time) & (time >= start) & (time <= start + span_days)
        fields = ("FLUX", "FLUX_ERR", "FLUX_BKG", "FLUX_BKG_ERR")
        arrays = {
            name.lower(): np.asarray(table[name][use], dtype=float) for name in fields
        }
        arrays.update(
            time=time[use],
            cadence=np.asarray(table["CADENCENO"][use]),
            quality=np.asarray(table["QUALITY"][use]),
            column_shift=np.asarray(table["POS_CORR1"][use], dtype=float),
            row_shift=np.asarray(table["POS_CORR2"][use], dtype=float),
        )
        units = {name: hdus[1].columns[name].unit for name in fields}
        origin = (float(header["2CRV5P"]), float(header["1CRV5P"]))
        metadata = header_metadata(hdus)
    if len(arrays["time"]) < 100:
        raise ValueError("Fewer than 100 finite-time pixel cadences")
    if any(unit not in ("e-/s", "electron/s") for unit in units.values()):
        raise ValueError(f"Unverified pixel rate units: {units}")
    return arrays, metadata | {"origin_row_column": origin, "units": units}


def crop_region(data, metadata, case, aperture):
    origin = np.asarray(metadata["origin_row_column"])
    position = np.array([case["target"]["row"], case["target"]["column"]]) - origin
    occupied = np.argwhere(aperture)
    lower = np.minimum(occupied.min(axis=0), np.floor(position - 7.5)).astype(int)
    upper = np.maximum(occupied.max(axis=0) + 1, np.ceil(position + 7.5)).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, aperture.shape)
    rows, columns = slice(lower[0], upper[0]), slice(lower[1], upper[1])
    for name in ("flux", "flux_err", "flux_bkg", "flux_bkg_err"):
        data[name] = data[name][:, rows, columns]
    metadata["crop_bounds_yx"] = [lower.tolist(), upper.tolist()]
    metadata["origin_row_column"] = (origin + lower).tolist()
    return aperture[rows, columns]


def supported_sources(case, origin, shape):
    kept, omitted = [], []
    for source in case.get("neighbors", []):
        if "row" not in source or "column" not in source:
            omitted.append(source | {"omission_reason": "MISSING_POSITION"})
            continue
        offset = np.array([source["row"], source["column"]]) - origin
        if np.all(offset >= -7.5) and np.all(offset <= np.array(shape) + 7.5):
            kept.append(source)
        else:
            omitted.append(source | {"omission_reason": "OUTSIDE_FIXED_PRF_SUPPORT"})
    return [case["target"], *kept], omitted


def positive_profile(prf, position, origin, shape):
    full_origin = np.floor(position).astype(int) - 8
    raw = prf._evaluate(
        targets=[tuple(position)], origin=tuple(full_origin), shape=(17, 17), dx=0, dy=0
    )[0]
    negative = float(-raw[raw < 0].sum())
    positive = np.maximum(raw, 0)
    positive /= positive.sum()
    result = np.zeros(shape)
    lower = np.maximum(full_origin, origin)
    upper = np.minimum(full_origin + 17, np.array(origin) + shape)
    if np.all(upper > lower):
        out = tuple(slice(a, b) for a, b in zip(lower - origin, upper - origin))
        own = tuple(
            slice(a, b) for a, b in zip(lower - full_origin, upper - full_origin)
        )
        result[out] = positive[own]
    return result, negative


def prf_cube(case, data, metadata, cache):
    from lkprf import KeplerPRF
    from lkprf.utils import channel_to_module_output

    channel = int(metadata["CHANNEL"]["value"])
    module, output = channel_to_module_output(channel)
    path = Path(cache) / f"kplr{module:02}.{output}_2011265_prf.fits"
    if not path.is_file():
        raise FileNotFoundError(f"Pre-cache and freeze the mission PRF: {path}")
    prf = KeplerPRF(channel, cache_dir=str(Path(cache)) + "/")
    origin = tuple(map(int, metadata["origin_row_column"]))
    shape = data["flux"].shape[1:]
    sources, omitted = supported_sources(case, origin, shape)
    positions = np.asarray([(s["row"], s["column"]) for s in sources], dtype=float)
    shifts = np.column_stack((data["row_shift"], data["column_shift"]))
    prf.evaluate(targets=list(map(tuple, positions)), origin=origin, shape=shape)
    evaluated = [
        positive_profile(prf, position, origin, shape) for position in positions
    ]
    retained = [0] + [i for i in range(1, len(evaluated)) if evaluated[i][0].sum() > 0]
    omitted.extend(
        sources[i] | {"omission_reason": "NO_CALIBRATED_PRF_SUPPORT"}
        for i in range(1, len(evaluated))
        if i not in retained
    )
    sources = [sources[i] for i in retained]
    evaluated = [evaluated[i] for i in retained]
    reference = np.asarray([item[0] for item in evaluated])
    dynamic = np.full(data["flux"].shape, np.nan)
    negative = [item[1] for item in evaluated]
    valid = np.isfinite(shifts).all(axis=1) & (data["quality"] == 0)
    for index in np.flatnonzero(valid):
        target = positions[0] + shifts[index]
        dynamic[index], clipped = positive_profile(prf, target, origin, shape)
        negative.append(clipped)
    return (
        reference,
        dynamic,
        {
            "path": str(path),
            "sha256": digest(path),
            "sources": sources,
            "omitted_sources": omitted,
            "clipped_negative_full_support_mass": finite_summary(negative),
        },
    )


def common_pixels(data, reference, aperture):
    good_rows = (data["quality"] == 0) & np.isfinite(data["row_shift"])
    good_rows &= np.isfinite(data["column_shift"])
    finite = np.isfinite(data["flux"][good_rows]) & np.isfinite(
        data["flux_err"][good_rows]
    )
    finite &= data["flux_err"][good_rows] > 0
    pixels = finite.all(axis=0) & np.isfinite(reference).all(axis=0)
    if np.any(aperture & ~pixels):
        raise ValueError("Delivered aperture includes unavailable calibration pixels")
    if good_rows.sum() < 100:
        raise ValueError(
            f"Only {good_rows.sum()} quality0 finite-pointing rows; require 100"
        )
    if pixels.sum() < len(reference) + 2:
        raise ValueError(
            f"Only {pixels.sum()} common pixels for {len(reference)} scene sources"
        )
    return pixels, good_rows


def fit_scene(reference, image, variance):
    design = np.column_stack((reference.T, np.ones(len(image))))
    weighted = design / np.sqrt(variance)[:, None]
    rank = np.linalg.matrix_rank(weighted)
    if rank != design.shape[1]:
        raise ValueError(f"Source scene rank {rank} below {design.shape[1]} columns")
    lower = np.r_[np.zeros(len(reference)), -np.inf]
    result = lsq_linear(weighted, image / np.sqrt(variance), bounds=(lower, np.inf))
    if not result.success or result.x[0] <= 0:
        raise ValueError("Positive target scene fit failed")
    return result.x, image - design @ result.x


def target_operator(profile, nuisance, variance):
    """Unit response to target, zero response to every frozen nuisance profile."""
    sigma = np.sqrt(variance)
    basis = np.asarray(nuisance, dtype=float).T / sigma[:, None]
    q, r = np.linalg.qr(basis, mode="reduced")
    if np.min(np.abs(np.diag(r))) <= 1e-10 * np.linalg.norm(basis):
        raise ValueError("Rank-deficient pixel nuisance model")
    whitened = profile / sigma
    residual = whitened - q @ (q.T @ whitened)
    information = float(residual @ residual)
    if information <= 1e-12 * float(whitened @ whitened):
        raise ValueError("Target is degenerate with pixel nuisances")
    return residual / sigma / information


def project_pixels(flux, error, operator):
    extracted = np.asarray(flux) @ operator
    propagated = np.sqrt(np.asarray(error) ** 2 @ operator**2)
    return extracted, propagated


def inject_pixels(flux, target_profile, target_rate, loss):
    delta = -np.asarray(loss)[:, None] * target_rate * np.asarray(target_profile)
    return np.asarray(flux) + delta, delta


def read_noise_electrons(metadata, case):
    if "read_noise_e_per_read" in case:
        return float(
            case["read_noise_e_per_read"]
        ), "EXPLICIT_MANIFEST_ELECTRONS_PER_READ"
    card = metadata.get("READNOIS")
    if card is None:
        return None, "MISSING_READNOIS"
    comment = card["comment"].lower()
    if "e-/" in comment or "electron" in comment or "e-" in comment:
        return float(card["value"]), "HEADER_ELECTRONS"
    if "adu" in comment and "GAIN" in metadata:
        return float(card["value"]) * float(
            metadata["GAIN"]["value"]
        ), "HEADER_ADU_TIMES_GAIN"
    return None, "READNOIS_UNIT_UNVERIFIED"


def photon_budget(data, profile, coefficients, metadata, case):
    target = coefficients[0] * data.get("target_prf", profile[0])
    neighbors = coefficients[1:-1] @ profile[1:]
    rates = {"target": target, "neighbors": neighbors, "background": data["flux_bkg"]}
    integration = metadata.get("INT_TIME", {}).get("value")
    frames = metadata.get("NUM_FRM", {}).get("value")
    read, read_status = read_noise_electrons(metadata, case)
    details = {
        "rate_units": "electron/s",
        "variance_units": "(electron/s)^2",
        "read_noise_status": read_status,
        "read_noise_assumption": "READNOIS electron noise is independent per frame",
        "ultimate_bound_established": False,
    }
    if integration is None or frames is None or integration <= 0 or frames <= 0:
        return {}, details | {"status": "MISSING_LIVE_EXPOSURE"}
    live = float(integration * frames)
    variances = {name: value / live for name, value in rates.items()}
    if read is not None:
        variances["read"] = np.full_like(target, frames * read**2 / live**2)
    total = sum(variances.values())
    variances["reported_total"] = data["flux_err"] ** 2
    variances["background_model_reported"] = data["flux_bkg_err"] ** 2
    variances["unexplained_signed"] = variances["reported_total"] - total
    if "observed_source_rate" in data:
        variances["observed_total_poisson_proxy"] = (
            data["observed_source_rate"] + data["flux_bkg"]
        ) / live
    return variances, details | {
        "status": "CONDITIONAL_SCENE_BUDGET",
        "live_seconds": live,
        "read_noise_e_per_read": read,
        "background_model_error_added_to_total": False,
        "negative_background_rate_pixels": int(np.sum(rates["background"] < 0)),
        "background_constant_fit_rate": float(coefficients[-1]),
    }


def compare_delivered(cadence, flux, error, delivered):
    _, source, target = np.intersect1d(
        cadence, delivered["cadence"], return_indices=True
    )
    a, b = flux[source], delivered["flux"][target]
    good = np.isfinite(a) & np.isfinite(b)
    if not good.any():
        return {"status": "NO_MATCHED_FINITE_DELIVERED_CADENCES"}
    difference = a[good] - b[good]
    scale = max(float(np.median(np.abs(b[good]))), 1.0)
    maximum = float(np.max(np.abs(difference)) / scale)
    passed = maximum <= POLICY["aperture_match_relative_tolerance"]
    return {
        "status": "MATCHED_DELIVERED_SAP" if passed else "APERTURE_REPLAY_DIFFERS",
        "cadences": int(good.sum()),
        "median_difference_e_per_s": float(np.median(difference)),
        "maximum_relative_difference": maximum,
        "error_ratio": finite_summary(
            error[source][good] / delivered["error"][target][good]
        ),
    }


def load_aperture(case, shape):
    if "aperture" in case:
        aperture = np.asarray(case["aperture"], dtype=bool)
    else:
        with np.load(case["aperture_path"], allow_pickle=False) as stored:
            aperture = np.asarray(stored["aperture"], dtype=bool)
    if "aperture_offset_in_tpf_yx" in case:
        placed = np.zeros(shape, dtype=bool)
        row, column = case["aperture_offset_in_tpf_yx"]
        rows, columns = np.nonzero(aperture)
        rows, columns = rows + row, columns + column
        if np.any(rows < 0) or np.any(rows >= shape[0]):
            raise ValueError("Active source aperture row outside target stamp")
        if np.any(columns < 0) or np.any(columns >= shape[1]):
            raise ValueError("Active source aperture column outside target stamp")
        placed[rows, columns] = True
        aperture = placed
    if aperture.shape != shape or not aperture.any():
        raise ValueError("Source aperture does not match target pixel stamp")
    return aperture


def delivered_series(stored):
    if "cadence" in stored:
        return {key: stored[key] for key in ("cadence", "flux", "error")}
    return {
        "cadence": stored["cadenceno"],
        "flux": stored["sap_flux"],
        "error": stored["sap_flux_err"],
    }


def baseline_arrays(data, flux, error, source_flux, quarter):
    return {
        "time": data["time"],
        "cadence": data["cadence"],
        "quality": data["quality"],
        "quarter": np.full(len(flux), quarter, dtype=np.int16),
        "flux": flux,
        "error": error,
        "source_flux": source_flux,
    }


def operator_products(
    case, data, profiles, coefficients, pixels, operator, folder, name
):
    flux, error = project_pixels(
        data["flux"][:, pixels], data["flux_err"][:, pixels], operator
    )
    response = profiles[:, pixels] @ operator
    source_flux = coefficients[0] * response
    path = folder / f"{name}.npz"
    quarter = int(case["quarter"])
    np.savez_compressed(
        path, **baseline_arrays(data, flux, error, source_flux, quarter)
    )
    row = {
        "name": f"{case['name']}_{name}",
        "path": str(path.resolve()),
        "source_id": case["source_id"],
        "quarter": quarter,
        "operator": name,
        "source_flux": "Scene target amplitude times measured PRF operator response",
        "source_response": finite_summary(response),
        "sha256": digest(path),
    }
    if name == "aperture" and case.get("delivered_path"):
        with np.load(case["delivered_path"], allow_pickle=False) as delivered:
            row["delivered_comparison"] = compare_delivered(
                data["cadence"], flux, error, delivered_series(delivered)
            )
    return row


def shared_trials(case, data):
    path = Path(__file__).with_name("preparation.py")
    spec = importlib.util.spec_from_file_location("pixel_shared_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = f"{case['source_id']}:{int(case['quarter'])}"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([module.DESIGN["seed"], seed]))
    truths = list(module.truth_rows(data, rng))
    profiles = np.array(
        [
            module.physical_loss(
                data["time"],
                truth["period_days"],
                truth["epoch_bkjd"],
                truth["duration_hours"],
                truth["depth_ppm"],
            )
            for truth in truths
        ]
    )
    return truths, profiles, {"path": str(path), "sha256": digest(path)}


def injection_parity(case, data, dynamic, coefficients, pixels, operators, folder):
    truths, losses, generator = shared_trials(case, data)
    original, target = data["flux"][:, pixels], dynamic[:, pixels]
    results = []
    for number, loss in enumerate(losses):
        injected, delta = inject_pixels(original, target, coefficients[0], loss)
        for name, operator in operators.items():
            actual = injected @ operator - original @ operator
            expected = -loss * coefficients[0] * (target @ operator)
            discrepancy = float(np.nanmax(np.abs(actual - expected)))
            scale = float(np.nanmax(np.abs(original) @ np.abs(operator)))
            tolerance = 64 * np.finfo(float).eps * max(scale, 1)
            if discrepancy > tolerance:
                raise ValueError("Pixel injection and extracted injection differ")
            results.append(
                {
                    "trial": number,
                    "operator": name,
                    "maximum_absolute_difference": discrepancy,
                    "floating_point_tolerance": tolerance,
                }
            )
        if number == 0:
            np.savez_compressed(
                folder / "pixel-injection-00.npz",
                injected_pixels=injected,
                pixel_delta=delta,
                loss=loss,
            )
    np.savez_compressed(folder / "physical-loss-profiles.npz", loss=losses)
    write(
        folder / "pixel-injection-parity.json",
        {
            "truths": truths,
            "generator": generator,
            "comparisons": results,
            "status": "ALL_TRIALS_AGREE",
        },
    )


def run_case(case, folder, prf_cache):
    folder.mkdir()
    data, metadata = read_tpf(case, POLICY["span_days"])
    aperture = load_aperture(case, data["flux"].shape[1:])
    aperture = crop_region(data, metadata, case, aperture)
    reference, dynamic, calibration = prf_cube(case, data, metadata, prf_cache)
    pixels, good = common_pixels(data, reference, aperture)
    image = np.median(data["flux"][good][:, pixels], axis=0)
    variance = np.median(data["flux_err"][good][:, pixels] ** 2, axis=0)
    profiles = reference[:, pixels]
    coefficients, residual = fit_scene(profiles, image, variance)
    nuisance = np.vstack((profiles[1:], np.ones(len(image))))
    operators = {
        "aperture": aperture[pixels].astype(float),
        "prf_background": target_operator(profiles[0], nuisance, variance),
    }
    budget_data = {
        name: value[good][:, pixels]
        for name, value in data.items()
        if name in ("flux_err", "flux_bkg", "flux_bkg_err")
    }
    budget_data["target_prf"] = dynamic[good][:, pixels]
    budget_data["observed_source_rate"] = image
    budget, budget_details = photon_budget(
        budget_data, profiles, coefficients, metadata, case
    )
    products = [
        operator_products(case, data, dynamic, coefficients, pixels, op, folder, name)
        for name, op in operators.items()
    ]
    injection_parity(case, data, dynamic, coefficients, pixels, operators, folder)
    budget_details["operators"] = {
        name: {
            term: finite_summary(np.asarray(values) @ op**2)
            for term, values in budget.items()
        }
        for name, op in operators.items()
    }
    saved = {
        "pixel_flux": data["flux"],
        "pixel_error": data["flux_err"],
        "target_prf": dynamic,
        "scene_prfs": reference,
        "coefficients": coefficients,
        "common_pixels": pixels,
        "aperture": aperture,
        "scene_residual": residual,
        "calibration_rows": good,
        **{f"operator_{name}": op for name, op in operators.items()},
        **{f"variance_{name}": values for name, values in budget.items()},
        **{name: data[name] for name in ("time", "cadence", "quality")},
    }
    np.savez_compressed(folder / "pixel-stage.npz", **saved)
    report = {
        "case": case,
        "tpf_sha256": digest(case["tpf_path"]),
        "metadata": metadata,
        "calibration": calibration,
        "budget": budget_details,
        "products": products,
        "scene_coefficients_e_per_s": coefficients.tolist(),
        "scene_residual": finite_summary(residual),
        "support": {
            "native_cadences": len(good),
            "quality0_cadences": int(np.sum(data["quality"] == 0)),
            "calibration_cadences": int(good.sum()),
            "common_pixels": int(pixels.sum()),
            "scene_sources": len(profiles),
            "scene_rank": len(coefficients),
            "nuisance_rank": len(nuisance),
            "rank_status": "FULL_RANK_CHECKED",
        },
        "pixel_stage_sha256": digest(folder / "pixel-stage.npz"),
        "scene_completeness": "CONDITIONAL_ON_LISTED_CATALOGUE_SOURCES",
    }
    write(folder / "report.json", report)
    return products


def normalize_case(case):
    result = dict(case)
    result["name"] = case.get("name", case.get("case_id"))
    if not result["name"]:
        raise ValueError("Explicit unique case name required")
    if "delivered_npz" in case:
        record = case["delivered_npz"]
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("Delivered array hash changed")
        result["delivered_path"] = record["path"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prf-cache", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    original = manifest["cases"] if isinstance(manifest, dict) else manifest
    cases = [normalize_case(case) for case in original]
    args.out.mkdir(parents=True, exist_ok=True)
    write(
        args.out / "plan.json",
        {
            "policy": POLICY,
            "cases": cases,
            "manifest_sha256": digest(args.manifest),
            "code_sha256": digest(__file__),
        },
    )
    products, failures = [], []
    for case in cases:
        try:
            products.extend(run_case(case, args.out / case["name"], args.prf_cache))
        except ValueError as error:
            failure = {
                "case": case["name"],
                "status": "PIXEL_CASE_FAILED",
                "first_failure": str(error),
            }
            failures.append(failure)
            write(args.out / case["name"] / "failure.json", failure)
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "products": len(products),
                    "failed_cases": len(failures),
                }
            ),
            flush=True,
        )
    write(args.out / "preparation-inputs.json", products)
    write(
        args.out / "completion.json",
        {
            "input_cases": len(cases),
            "failures": failures,
            "operator_products": len(products),
        },
    )


if __name__ == "__main__":
    main()
