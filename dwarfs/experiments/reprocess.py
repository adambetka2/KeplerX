"""Adaptive, target-protected pixel regression with blocked out-of-sample fits."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

POLICY = {
    "method": "target-and-position-protected pixel regression",
    "block_days": 4.0,
    "guard_days": 0.5,
    "ridge": 10.0,
    "objective": "sum((median_aperture_error/error_t)*(y_t-X_t beta))^2 + 10 sum(beta_nonintercept^2)",
    "features": "target-projected pixels plus x,y,x^2,xy,y^2 pointing and first four measured mission CBVs",
    "training": "Every held block excluded with 0.5-day guard; fold-local median/MAD scales",
    "projection": "Whitened span of source profile and two spatial derivatives; SVD relative cutoff 1e-8",
    "covariance": "Returned pixel weights/errors condition on trained coefficients; use full-refit null covariance for inference",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_features(pixel_flux, pixel_error, target_profiles, derivatives):
    scale = np.median(pixel_error, axis=0)
    templates = np.concatenate((target_profiles[:, :, None], derivatives), axis=2)
    templates = templates / scale[None, :, None]
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero target or spatial-derivative support")
    u, singular, _ = np.linalg.svd(templates / norms, full_matrices=False)
    keep = singular > singular[:, :1] * 1e-8
    u *= keep[:, None, :]
    weighted = pixel_flux / scale
    features = weighted - np.einsum(
        "npi,ni->np", u, np.einsum("npi,np->ni", u, weighted)
    )
    return features, u, scale, singular


def regress_fold(features, response, errors, train, test):
    center = np.median(features[train], axis=0)
    scale = 1.4826 * np.median(np.abs(features[train] - center), axis=0)
    active = scale > np.maximum(np.abs(center), 1) * 1e-12
    x = (features[:, active] - center[active]) / scale[active]
    x = np.column_stack((np.ones(len(x)), x))
    weight = np.median(errors[train]) / errors[train]
    design = x[train] * weight[:, None]
    target = response[train] * weight
    penalty = np.diag(np.r_[0, np.repeat(POLICY["ridge"], x.shape[1] - 1)])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    slope = np.zeros(features.shape[1])
    slope[active] = coefficients[1:] / scale[active]
    prediction = (features[test] - center) @ slope
    return (
        prediction,
        slope,
        {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "active": active.tolist(),
            "coefficients": coefficients.tolist(),
            "training_rows": int(train.sum()),
            "held_rows": int(test.sum()),
        },
    )


def fit_extract(
    pixel_flux,
    time,
    pointing,
    target_profiles,
    aperture,
    background,
    pixel_error,
    profile_derivatives,
    cbv=None,
    reference_pixels=True,
):
    arrays = (
        pixel_flux,
        time,
        pointing,
        target_profiles,
        aperture,
        background,
        pixel_error,
        profile_derivatives,
    )
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("Extractor requires a predeclared common finite support")
    if np.any(pixel_error <= 0) or np.any(np.diff(time) <= 0):
        raise ValueError("Positive pixel errors and increasing native times required")
    baseline = pixel_flux @ aperture - background
    error = np.sqrt(pixel_error**2 @ aperture**2)
    reference, u, whitening, singular = protected_features(
        pixel_flux, pixel_error, target_profiles, profile_derivatives
    )
    x, y = pointing.T
    if not reference_pixels:
        reference = np.zeros_like(reference)
    features = np.column_stack((reference, x, y, x * x, x * y, y * y))
    if cbv is not None:
        if cbv.shape != (len(time), 4) or not np.isfinite(cbv).all():
            raise ValueError("Four finite aligned mission CBVs required")
        features = np.column_stack((features, cbv))
    folds = np.floor((time - time[0]) / POLICY["block_days"]).astype(int)
    prediction, weights, diagnostics = (
        np.empty(len(time)),
        np.empty_like(pixel_flux),
        [],
    )
    for fold in np.unique(folds):
        test = folds == fold
        start = time[0] + fold * POLICY["block_days"]
        train = (time < start - POLICY["guard_days"]) | (
            time >= start + POLICY["block_days"] + POLICY["guard_days"]
        )
        if train.sum() < max(100, 3 * features.shape[1]):
            raise ValueError(
                "Insufficient independent rows for blocked pixel regression"
            )
        prediction[test], slope, details = regress_fold(
            features, baseline, error, train, test
        )
        pixel_slope = np.broadcast_to(
            slope[: pixel_flux.shape[1]], (test.sum(), pixel_flux.shape[1])
        )
        projected = pixel_slope - np.einsum(
            "npi,ni->np", u[test], np.einsum("npi,np->ni", u[test], pixel_slope)
        )
        weights[test] = aperture - projected / whitening
        diagnostics.append(
            details
            | {"fold": int(fold), "train_indices": np.flatnonzero(train).tolist()}
        )
    prediction -= np.median(prediction)
    return {
        "flux": baseline - prediction,
        "baseline_flux": baseline,
        "formal_error": error,
        "conditional_error": np.sqrt(np.sum(weights**2 * pixel_error**2, axis=1)),
        "prediction": prediction,
        "folds": folds,
        "pixel_weights": weights,
        "diagnostics": {
            "policy": POLICY,
            "reference_pixels": reference_pixels,
            "folds": diagnostics,
            "projection_min_relative_singular": float(
                np.min(singular[:, -1] / singular[:, 0])
            ),
            "projection_target_response_error": float(
                np.max(
                    np.abs(
                        np.sum(weights * target_profiles, axis=1)
                        - target_profiles @ aperture
                    )
                )
            ),
        },
    }


def author_basis(path):
    from patsy import dmatrix
    from scipy import sparse

    expected = "7bdd5581bd05838ef38e649afbc38f2ce87b8426ae9ae0e2bbf3a1dd7cfd9640"
    if digest(path) != expected:
        raise ValueError("Author basis source differs from reviewed psfmachine v1.1.4")
    tree = ast.parse(Path(path).read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_make_A_polar", "wrapped_spline"}
    ]
    if len(selected) != 2:
        raise ValueError(
            "Pinned author source does not expose the two declared spline functions"
        )
    namespace = {"np": np, "sparse": sparse, "dmatrix": dmatrix}
    exec(  # noqa: S102 -- Only the two reviewed, hash-pinned numerical functions.
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_make_A_polar"]


def learned_profile(
    model_path, basis, rows, columns, target_row, target_column, wcs, width=1.0
):
    from astropy.io import fits

    with fits.open(model_path) as hdus:
        header, coefficients = hdus[1].header.copy(), np.asarray(hdus[1].data["psf_w"])
    radius = float(header["RMAX"])
    options = {
        "rmin": header["RMIN"],
        "rmax": radius,
        "cut_r": header["CUT_R"],
        "n_r_knots": header["N_RKNOTS"],
        "n_phi_knots": header["N_PKNOTS"],
    }

    def evaluate(pixel_rows, pixel_columns, source_rows, source_columns):
        sky = wcs.all_pix2world(
            np.column_stack((pixel_columns.ravel(), pixel_rows.ravel())), 0
        )
        center = wcs.all_pix2world(np.column_stack((source_columns, source_rows)), 0)
        sky = sky.reshape((*pixel_rows.shape, 2))
        # Match the author's sky axes exactly: delta RA, without a cos(dec) factor.
        dx = ((sky[:, :, 0] - center[:, None, 0] + 180) % 360 - 180) * 3600
        dy = (sky[:, :, 1] - center[:, None, 1]) * 3600
        r, phi = np.hypot(dx, dy) / width, np.arctan2(dy, dx)
        use = r <= radius
        values = np.zeros(r.shape)
        matrix = basis(np.r_[phi[use], 0, 0], np.r_[r[use], 0, radius], **options)
        values[use] = 10 ** (matrix @ coefficients)[:-2]
        return values

    target_row, target_column = np.asarray(target_row), np.asarray(target_column)
    shape = (len(target_row), len(rows))
    raw = evaluate(
        np.broadcast_to(rows, shape),
        np.broadcast_to(columns, shape),
        target_row,
        target_column,
    )
    # This is an explicitly truncated model, not a claim of zero physical wings.
    grid = np.arange(-18, 19)
    gy, gx = np.meshgrid(grid, grid, indexing="ij")
    norm = np.empty(len(target_row))
    for start in range(0, len(target_row), 128):
        selected = slice(start, start + 128)
        py = np.floor(target_row[selected])[:, None] + gy.ravel()
        px = np.floor(target_column[selected])[:, None] + gx.ravel()
        norm[selected] = evaluate(
            py, px, target_row[selected], target_column[selected]
        ).sum(axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("Nonpositive learned-PSF normalization")
    return raw / norm[:, None], norm


def mission_evaluator(case, cache, shape):
    from lkprf import KeplerPRF

    spec = importlib.util.spec_from_file_location(
        "reprocess_mission", Path(__file__).with_name("pixels.py")
    )
    pixels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pixels)
    prf = KeplerPRF(int(case["headers"]["CHANNEL"]), cache_dir=str(cache) + "/")
    origin = (int(case["headers"]["2CRV5P"]), int(case["headers"]["1CRV5P"]))
    position = (case["target"]["row"], case["target"]["column"])
    prf.evaluate(targets=[position], origin=origin, shape=shape)

    def evaluate(row, column, width=1.0):
        if width != 1:
            raise ValueError("Mission profile has no declared width-rescaling operator")
        values = [
            pixels.positive_profile(prf, np.array([y, x]) + origin, origin, shape)[
                0
            ].ravel()
            for y, x in zip(row, column)
        ]
        return np.asarray(values), np.ones(len(row))

    return evaluate


def profile_bundle(case, data, use, pixel_mask, options):
    from astropy.io import fits
    from astropy.wcs import WCS

    shape = data["flux"].shape[1:]
    rows, columns = np.indices(shape)
    y = case["target"]["local_row"] + data["pos_corr2"][use]
    x = case["target"]["local_column"] + data["pos_corr1"][use]
    mission = mission_evaluator(case, options.prf_cache, shape)
    model_path = (
        options.learned_dir
        / f"Kepler_ffi_shape_model_ch{case['headers']['CHANNEL']:02}_q{case['quarter']:02}.fits"
    )
    with fits.open(case["tpf_path"], memmap=False) as hdus:
        wcs = WCS(hdus[2].header)
    learned = case["quarter"] != 1
    if learned and not model_path.is_file():
        raise FileNotFoundError(model_path)
    if learned:
        basis = author_basis(options.basis_source)

        def evaluate(row, column, width=1.0):
            return learned_profile(
                model_path,
                basis,
                rows.ravel(),
                columns.ravel(),
                row,
                column,
                wcs,
                width,
            )
    else:
        evaluate = mission
    primary, normalization = evaluate(y, x)
    derivative_y = (evaluate(y + 0.005, x)[0] - evaluate(y - 0.005, x)[0]) / 0.01
    derivative_x = (evaluate(y, x + 0.005)[0] - evaluate(y, x - 0.005)[0]) / 0.01
    families = {
        "profile_family_independent_mission": mission(y, x)[0],
        "profile_family_x_plus_0p1": evaluate(y, x + 0.1)[0],
        "profile_family_x_minus_0p1": evaluate(y, x - 0.1)[0],
        "profile_family_y_plus_0p1": evaluate(y + 0.1, x)[0],
        "profile_family_y_minus_0p1": evaluate(y - 0.1, x)[0],
    }
    if learned:
        families["profile_family_width_1p05"] = evaluate(y, x, 1.05)[0]
    else:
        del families["profile_family_independent_mission"]
    arrays = {
        "target_profiles": primary[:, pixel_mask],
        "profile_derivatives": np.stack(
            (derivative_x[:, pixel_mask], derivative_y[:, pixel_mask]), axis=-1
        ),
        **{name: value[:, pixel_mask] for name, value in families.items()},
    }
    metadata = {
        "policy": "Published learned sky-coordinate FFI shape"
        if learned
        else "Q1 learned model missing; explicitly declared mission profile",
        "model_path": str(model_path) if learned else "mission PRF in declared cache",
        "model_sha256": digest(model_path) if learned else None,
        "normalization": "37x37 discrete grid; author spline truncated at RMAX arcsec, not a physical zero-wing assertion"
        if learned
        else "Mission calibrated 17x17 positive support",
        "normalization_min_max": [
            float(normalization.min()),
            float(normalization.max()),
        ],
        "median_stamp_mass": float(np.median(primary.sum(axis=1))),
        "median_aperture_mass": float(np.median(primary @ data["aperture"].ravel())),
        "catalog_PSFFRAC": case["target_header"]["PSFFRAC"],
    }
    return arrays, metadata


def prepare_case(case, cbv_entry, options):
    if digest(case["path"]) != case["sha256"]:
        raise ValueError("Exported native pixel input changed")
    with np.load(case["path"], allow_pickle=False) as archive:
        data = dict(archive)
    with np.load(cbv_entry["path"], allow_pickle=False) as archive:
        if not np.array_equal(data["cadence"], archive["cadenceno"]):
            raise ValueError("CBVs do not exactly match native cadence numbers")
        cbv = np.column_stack([archive[f"vector_{index}"] for index in range(1, 5)])
        cbv_good = ~archive["gapflag"].astype(bool)
    use = (data["quality"] == 0) & data["delivered_present"]
    fields = (
        "time",
        "pos_corr1",
        "pos_corr2",
        "delivered_psf_flux",
        "delivered_psf_flux_err",
        "delivered_sap_flux",
        "delivered_sap_background",
    )
    use &= np.isfinite(np.column_stack([data[name] for name in fields])).all(axis=1)
    use &= (
        cbv_good & np.isfinite(cbv).all(axis=1) & (data["delivered_psf_flux_err"] > 0)
    )
    # RAW_CNTS is uncalibrated digital counts and cannot enter this e/s operator.
    flux = (
        data["flux"] + data["flux_bkg"] if case["background_replaced"] else data["flux"]
    )
    flux, error = (
        flux[use].reshape(use.sum(), -1),
        data["flux_err"][use].reshape(use.sum(), -1),
    )
    pixels = (np.isfinite(flux) & np.isfinite(error) & (error > 0)).all(axis=0)
    if np.any(data["aperture"].ravel() & ~pixels) or use.sum() < 100:
        raise ValueError("Insufficient common native aperture support")
    background = (
        data["delivered_sap_background"][use]
        if case["background_replaced"]
        else np.zeros(use.sum())
    )
    arrays = {
        "pixel_flux": flux[:, pixels],
        "pixel_error": error[:, pixels],
        "time": data["time"][use],
        "cadence": data["cadence"][use],
        "pointing": np.column_stack((data["pos_corr1"][use], data["pos_corr2"][use])),
        "aperture": data["aperture"].ravel()[pixels].astype(float),
        "background": background,
        "cbv": cbv[use],
        "delivered_psf": data["delivered_psf_flux"][use],
        "delivered_psf_error": data["delivered_psf_flux_err"][use],
        "delivered_sap": data["delivered_sap_flux"][use],
        "source_flux": np.array(np.median(data["delivered_psf_flux"][use])),
        "native_row_indices": np.flatnonzero(use),
        "native_pixel_indices": np.flatnonzero(pixels),
    }
    replay = arrays["pixel_flux"] @ arrays["aperture"] - background
    discrepancy = float(
        np.max(
            np.abs(replay - arrays["delivered_sap"]) / np.abs(arrays["delivered_sap"])
        )
    )
    if discrepancy > 1e-5 or arrays["source_flux"] <= 0:
        raise ValueError(
            "Failed delivered-SAP replay or nonpositive source-flux calibration"
        )
    if np.any(np.diff(arrays["time"]) <= 0):
        raise ValueError("Native time is not strictly increasing")
    profiles, profile_metadata = profile_bundle(case, data, use, pixels, options)
    arrays.update(profiles)
    path = options.out / f"{case['case_id']}.npz"
    np.savez_compressed(path, **arrays)
    return {
        "name": case["case_id"],
        "source_id": case["source_id"],
        "path": str(path.resolve()),
        "sha256": digest(path),
        "input_sha256": case["sha256"],
        "pixel_units": "calibrated electrons/second",
        "pixel_convention": "FLUX+FLUX_BKG" if case["background_replaced"] else "FLUX",
        "cbv_gap_rows_excluded": int(np.sum(~cbv_good)),
        "cbv_sha256": digest(cbv_entry["path"]),
        "rows": int(use.sum()),
        "pixels": int(pixels.sum()),
        "span_days": float(np.ptp(arrays["time"])),
        "profile_policy": profile_metadata,
        "source_flux_e_per_s": float(arrays["source_flux"]),
        "source_flux_policy": "Median delivered corrected-PSF amplitude; absolute calibration systematic unquantified",
        "sap_replay_max_relative_error": discrepancy,
    }


def prepare(options):
    options.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(options.manifest.read_text())["cases"]
    cbvs = {
        row["case_id"]: row["selected_channel"][0]
        for row in json.loads(options.cbv_manifest.read_text())
    }
    plan = {
        "policy": POLICY,
        "source_sha256": digest(__file__),
        "input_manifest_sha256": digest(options.manifest),
        "basis_source_sha256": digest(options.basis_source),
        "coordinate_audit_sha256": digest(options.coordinate_audit),
        "mission_prf_files": {
            str(p.resolve()): digest(p)
            for p in sorted(options.prf_cache.glob("*.fits"))
        },
    }
    (options.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    output = []
    for case in cases:
        try:
            output.append(prepare_case(case, cbvs[case["case_id"]], options))
        except Exception as error:
            (options.out / "first-failure.json").write_text(
                json.dumps({"case": case["case_id"], "error": repr(error)}, indent=2)
                + "\n"
            )
            raise
        (options.out / "manifest.json").write_text(
            json.dumps(plan | {"cases": output, "complete": False}, indent=2) + "\n"
        )
        print(case["case_id"], "profiles prepared", flush=True)
    (options.out / "manifest.json").write_text(
        json.dumps(plan | {"cases": output, "complete": True}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-policy", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prf-cache", type=Path)
    parser.add_argument("--cbv-manifest", type=Path)
    parser.add_argument("--basis-source", type=Path)
    parser.add_argument("--learned-dir", type=Path)
    parser.add_argument("--coordinate-audit", type=Path)
    args = parser.parse_args()
    if args.show_policy:
        print(json.dumps(POLICY, indent=2))
    else:
        prepare(args)


if __name__ == "__main__":
    main()
