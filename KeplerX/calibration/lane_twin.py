"""The SAP lane's calibration sample, built as a twin of the PSF Flat one."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from calibration import sample_builder as sb
from calibration.lane_census import INDEX, array_features
from calibration.signal_injection import binary_flux, transit_flux
from common import ROOT, config_file, digest, read, receipt, save_packet, write
from photometry.program import (
    EXCLUSION_BITS,
    prepare_arrays,
    share_by_quarter,
    star_shares,
    view_columns,
)  # fmt: skip
from photometry.star_share import (
    expected_flux,
    fits_quarters,
    kepler_band_mag,
    lc2_quarters,
)

SAMPLE = ROOT / "data/calibration-sample"
VIEW = "sap_flat"
TWINNED = (
    "control_original", "control_artifact", "control_inverted", "control_scrambled",
    "synthetic_planet", "hz_planet", "synthetic_binary", "contamination",
    "known_planet", "validated_boi",
)  # fmt: skip
# The share model's spread between quarters (ln), from Gaia binaries' eclipses at S/N > 50.
SHARE_ERROR = 0.075
FITS_NAMES = {"flux": "SAP_FLUX", "error": "SAP_FLUX_ERR", "time": "TIME",
              "cadence": "CADENCENO", "quarter": "QUARTER", "quality": "SAP_QUALITY",
              "flatten_mask": "FLATTEN_MASK"}  # fmt: skip


def sap_arrays(packet):
    """The host's stitched SAP and its cadence columns, from the PSF Flat packet's own source."""
    from photometry.program import locate_source, source_table

    provenance = packet["provenance"]
    if "kbonus_fits" in provenance:
        path = provenance["kbonus_fits"]["path"]
        with fits.open(path, memmap=False) as hdus:
            table = hdus["LIGHTCURVE_STITCHED"].data
            arrays = {k: np.array(table[n]) for k, n in FITS_NAMES.items()}
            measured = fits_quarters(hdus)
        provenance = {"kbonus_fits": receipt(path)}
    else:
        columns = view_columns(VIEW)
        folder = locate_source(packet["source_id"], INDEX)
        table = source_table(
            folder / "cadence.parquet", packet["source_id"], list(columns.values())
        )
        arrays = {k: table[n].to_numpy() for k, n in columns.items()}
        measured = lc2_quarters(folder, packet["source_id"])
        provenance = {"shard": str(folder)}
    arrays["star_share"], provenance["star_share"] = star_shares(
        arrays["quarter"], *measured
    )
    return arrays, provenance


def sap_lane(packet, arrays=None, provenance=None):
    """Prepared SAP Flat, the raw stitched SAP in its order, and metadata."""
    if arrays is None:
        arrays, provenance = sap_arrays(packet)
    raw = np.asarray(arrays["flux"], float)
    prepared, scales = prepare_arrays(arrays, config_file()["photometry"], VIEW)
    good = prepared["mask"]
    metadata = {k: v for k, v in packet.items() if k not in ("arrays", "provenance")}
    metadata |= {
        "view": VIEW,
        "flux_convention": "stitched SAP flattened as KBonus flattened PSF, decrowded per quarter",
        "provenance": provenance,
        "normalization_by_quarter": scales,
        "star_share_by_quarter": share_by_quarter(prepared),
        "exclusion_bit_definitions": EXCLUSION_BITS,
        "stored_cadences": len(prepared["time"]),
        "usable_cadences": int(good.sum()),
        "baseline_days": float(np.ptp(prepared["time"][good])),
        "first_valid_bkjd": float(prepared["time"][good].min()),
        "last_valid_bkjd": float(prepared["time"][good].max()),
        "median_formal_error_ppm": float(np.median(prepared["error"][good]) * 1e6),
    }
    return prepared, raw[prepared["native_index"]], metadata


def true_shares(arrays, case_id):
    """Each cadence's share as the sky gives it."""
    seed = int(hashlib.sha256(f"sap-share|{case_id}".encode()).hexdigest()[:16], 16)
    quarters = np.unique(arrays["quarter"])
    error = dict(
        zip(
            quarters.tolist(),
            np.exp(np.random.default_rng(seed).normal(0, SHARE_ERROR, len(quarters))),
            strict=True,
        )
    )
    return np.minimum(
        1.0,
        arrays["star_share"] * np.array([error[q] for q in arrays["quarter"].tolist()]),
    )


def inject(packet, arrays, provenance, clean, features, bodies, model_of, case_id):
    """`bodies` on the star injected into the raw stitched SAP."""
    share = true_shares(arrays, case_id)
    order = clean["native_index"]
    residual = (share / arrays["star_share"])[order]

    def prepared_with(models):
        factor = np.ones(len(arrays["flux"]))
        for model in models:
            factor *= 1 + share * (model - 1)
        return sap_lane(packet, arrays | {"flux": arrays["flux"] * factor}, provenance)

    models = [model_of(arrays["time"], body) for body in bodies]
    injected = prepared_with(models)
    stats = []
    for body, model in zip(bodies, models, strict=True):
        alone = injected[0] if len(bodies) == 1 else prepared_with([model])[0]
        change = np.nan_to_num(alone["flux"] - clean["flux"])
        stats.append(
            sb.signal_statistics(clean, change, model[order], body, features, residual)
        )
    return *injected, stats


def planet_model(time, body):
    return transit_flux(time, sb.EXPOSURE_DAYS, body, body["epoch_bkjd"])


def binary_model(time, body):
    return binary_flux(time, sb.EXPOSURE_DAYS, body, body["epoch_bkjd"])


def star_light(host, metadata, prepared):
    """The host's star_light record on SAP: its quarters' modelled shares."""
    shares = metadata["star_share_by_quarter"]
    usable = prepared["star_share"][prepared["mask"]]
    return {
        "kepler_band_mag": kepler_band_mag(host.get("g_mag"), host.get("bp_rp")),
        "expected_flux": expected_flux(host.get("g_mag"), host.get("bp_rp")),
        "fraction_by_quarter": shares,
        "fraction": float(np.mean(usable)) if len(usable) else 1.0,
    }


def twin(job):
    """One case's SAP twin; its row for cases.parquet and its flat signals."""
    case, out = job
    try:
        truth = read(case["truth"])
        packet = read(Path(case["prepared"]) / "packet.json")
        sap, provenance = sap_arrays(packet)
        prepared, raw, metadata = sap_lane(packet, sap, provenance)
        features = array_features(prepared, raw)
        host = truth["host"] | {"features": features}
        host["star_light"] = star_light(host, metadata, prepared)
        kind = case["kind"]
        record = {
            k: v
            for k, v in truth.items()
            if k not in ("host", "signals", "diagnostics")
        }
        signals = []
        arrays = prepared
        if kind in ("control_inverted", "control_scrambled"):
            transform = "inverted" if kind == "control_inverted" else "block_scrambled"
            block = read(SAMPLE / "plan.json")["scramble_block_cadences"]
            arrays = sb.transformed_control(
                prepared, metadata, transform, truth["seed"], block
            )
        elif kind in ("synthetic_planet", "hz_planet"):
            bodies = [dict(s) for s in truth["signals"]]
            arrays, _, metadata, stats = inject(
                packet, sap, provenance, prepared, features, bodies, planet_model, case["case_id"]
            )  # fmt: skip
            signals = [
                old | new | sb.detectability(old, new, host["catalogue_stellar"])
                for old, new in zip(truth["signals"], stats, strict=True)
            ]
        elif kind == "synthetic_binary":
            bodies = [dict(s) for s in truth["signals"]]
            arrays, _, metadata, stats = inject(
                packet, sap, provenance, prepared, features, bodies, binary_model, case["case_id"]
            )  # fmt: skip
            signals = [
                old | new for old, new in zip(truth["signals"], stats, strict=True)
            ]
        elif kind in ("contamination", "known_planet", "validated_boi"):
            context = {"prepared": prepared, "metadata": metadata, "features": features,
                       "host": host, "job": {"signals": truth["signals"], "scope": truth.get("scope", "")}}  # fmt: skip
            arrays, metadata, made = sb.make_real(context, kind)
            signals = [
                old | new
                for old, new in zip(truth["signals"], made["signals"], strict=True)
            ]
        folder = Path(out) / "light-curves" / case["case_id"]
        folder.mkdir(parents=True, exist_ok=True)
        save_packet(
            folder,
            arrays,
            metadata | {"case_id": case["case_id"], "calibration_case": True},
        )
        record |= {"lane": VIEW, "host": host, "signals": signals,
                   "prepared": str(folder), "psf_flat_twin": case["case_id"]}  # fmt: skip
        path = Path(out) / "truth" / f"{case['case_id']}.json"
        write(path, record)
        row = dict(case) | {
            "prepared": str(folder), "truth": str(path), "truth_sha256": digest(path),
            "packet_sha256": digest(folder / "packet.json"),
            "star_flux_fraction": host["star_light"]["fraction"],
            "n_signals": len(signals), "stored_cadences": metadata["stored_cadences"],
            "usable_cadences": metadata["usable_cadences"],
        }  # fmt: skip
        flat = [{"case_id": case["case_id"], "case_kind": kind, "source_id": case["source_id"],
                 **sb.flat_signal(s)} for s in signals]  # fmt: skip
        return row, flat, None
    except Exception as error:
        return None, [], {"case_id": case["case_id"], "kind": case["kind"],
                          "error": f"{type(error).__name__}: {error}"}  # fmt: skip


def merge(path, frame, case_ids):
    """`frame` joined to the table at `path` less its rows of `case_ids`."""
    if not path.exists():
        return frame
    old = pd.read_parquet(path)
    return pd.concat([old[~old.case_id.isin(set(case_ids))], frame], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    wanted = [json.loads(line)["case_id"] for line in open(args.manifest)]
    cases = pd.read_parquet(SAMPLE / "cases.parquet").set_index("case_id")
    cases = cases.loc[[c for c in wanted if c in cases.index]].reset_index()
    cases = cases[cases.kind.isin(TWINNED)]
    (args.out / "truth").mkdir(parents=True, exist_ok=True)
    rows, signals, errors = [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for row, flat, error in pool.map(
            twin, [(c, str(args.out)) for c in cases.to_dict("records")]
        ):
            if row is not None:
                rows.append(row)
                signals += flat
            if error is not None:
                errors.append(error)
    frame = pd.DataFrame(rows)
    merged = merge(args.out / "cases.parquet", frame, frame.case_id)
    merged.to_parquet(args.out / "cases.parquet")
    flat = pd.DataFrame(signals)
    merge(args.out / "signals.parquet", flat, frame.case_id).to_parquet(
        args.out / "signals.parquet"
    )
    (args.out / "manifests").mkdir(exist_ok=True)
    with open(args.out / "manifests" / args.manifest.name, "w") as handle:
        for r in frame.itertuples():
            handle.write(
                json.dumps({"case_id": r.case_id, "prepared": r.prepared}) + "\n"
            )
    write(args.out / f"failures-{args.manifest.stem}.json", errors)
    build = read(args.out / "build.json") if (args.out / "build.json").exists() else {}
    build = {"view": VIEW, "manifests": build.get("manifests", {})}
    build["manifests"][args.manifest.stem] = {
        "manifest": str(args.manifest), "twinned": len(rows), "failures": len(errors),
        "skipped_kinds": sorted(set(pd.read_parquet(SAMPLE / "cases.parquet").kind) - set(TWINNED)),
    }  # fmt: skip
    build["cases"] = len(merged)
    write(args.out / "build.json", build)
    print(json.dumps({"twinned": len(rows), "failures": len(errors)}))


if __name__ == "__main__":
    main()
