"""Verify physical transfer against declared flux truth, without vetting outcomes."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]
from deterministic import injections
from deterministic import pixel_injections as pixels
from lc2 import load_source
from vetting_pixels import kepler_prf_path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_hash(path, expected, products):
    path = str(Path(path).resolve())
    if path in products:
        if products[path] != expected:
            raise ValueError(f"Conflicting product identity: {path}")
        return
    if digest(path) != expected:
        raise ValueError(f"Product bytes changed: {path}")
    products[path] = expected


def arrays_equal(first, second):
    if first.dtype.kind in "fc":
        return np.array_equal(first, second, equal_nan=True)
    return np.array_equal(first, second)


def verify_unchanged(base, controlled):
    if len(base) != len(controlled):
        raise ValueError("FITS extension count changed")
    for index, (old, new) in enumerate(zip(base, controlled)):
        if index == 1:
            if old.columns.names != new.columns.names:
                raise ValueError("FITS table columns changed")
            for field in old.columns.names:
                if field != "FLUX" and not arrays_equal(
                    old.data[field], new.data[field]
                ):
                    raise ValueError(f"Non-flux FITS column changed: {field}")
        elif old.data is not None and not arrays_equal(old.data, new.data):
            raise ValueError("FITS image data changed")


def verify_pixel(folder, sid, truth, item, raw, astrometry, signal, cache, products):
    quarter, pixel = item
    controlled = (folder / pixel["path"]).resolve()
    public = Path(pixel["base_path"])
    verify_hash(controlled, pixel["sha256"], products)
    verify_hash(public, pixel["base_sha256"], products)
    calibration = pixel["prf_calibration"]
    verify_hash(cache / calibration["filename"], calibration["sha256"], products)
    with (
        fits.open(public, memmap=False) as base,
        fits.open(controlled, memmap=False, checksum=True) as injected,
    ):
        base.verify("exception")
        injected.verify("exception")
        for hdus in (base, injected):
            if (
                int(hdus[0].header["QUARTER"]) != int(quarter)
                or int(hdus[0].header["KEPLERID"]) != pixel["base_parent_kic"]
            ):
                raise ValueError("TPF parent or quarter differs")
        if (
            injected[0].header["FSCSRC"] != sid
            or injected[0].header["FSCMODEL"] != "physical-prf-v1"
        ):
            raise ValueError("Injected source or physical model differs")
        if any(
            hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1 for hdu in injected
        ):
            raise ValueError("Controlled FITS checksum failed")
        verify_unchanged(base, injected)
        if (
            kepler_prf_path(int(base[0].header["CHANNEL"]), cache).name
            != calibration["filename"]
        ):
            raise ValueError("PRF calibration does not match the pixel channel")
        profile, measured_calibration, pixel_position, sky, status, inside = (
            pixels.pixel_profile(base, astrometry, cache)
        )
        if (
            measured_calibration != calibration
            or status != pixel["target_astrometry"]
            or inside != pixel["target_inside_stamp"]
        ):
            raise ValueError("PRF or astrometry provenance differs")
        np.testing.assert_allclose(
            pixel_position, pixel["target_pixel"], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(sky, pixel["target_sky_deg"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            profile.sum(), pixel["prf_stamp_fraction"], rtol=0, atol=1e-10
        )
        x, y = pixel_position
        height, width = profile.shape
        stamp_distance = float(
            np.hypot(max(-x, 0, x - (width - 1)), max(-y, 0, y - (height - 1)))
        )
        times = np.asarray(base[1].data["TIME"], dtype=np.float64)
        loss = pixels.transit_loss(times, truth, signal)
        flux = np.asarray(raw[injections.PRIMARY_FLUX], dtype=np.float64)
        use = (raw["quarter"] == int(quarter)) & np.isfinite(flux)
        amplitude = float(np.median(flux[use]))
        if (
            not np.isfinite(amplitude)
            or amplitude <= 0
            or amplitude != pixel["source_flux_electrons_per_second"]
        ):
            raise ValueError("LC2 to TPF source-flux transfer differs")
        original = base[1].data["FLUX"]
        expected = (
            original.astype(np.float64)
            - amplitude * loss[:, None, None] * profile[None]
        ).astype(original.dtype)
        observed = injected[1].data["FLUX"]
        mismatch = (
            np.isfinite(expected) & np.isfinite(observed) & (expected != observed)
        )
        difference = abs(expected.astype(np.float64) - observed.astype(np.float64))
        return {
            "quarter": int(quarter),
            "parent_tpf": str(pixel["base_parent_kic"]),
            "transfer_matches_declared_truth": arrays_equal(expected, observed),
            "mismatched_pixel_values": int(mismatch.sum()),
            "maximum_flux_difference": float(np.nanmax(difference)),
            "declared_maximum_loss": float(loss.max()),
            "recorded_maximum_loss": pixel["maximum_relative_loss"],
            "target_inside_stamp": inside,
            "stamp_distance_pixels": stamp_distance,
            "structural_support": stamp_distance <= 2.0,
            "prf_stamp_fraction": float(profile.sum()),
        }


def verify_source(folder, sid, truth, item, astrometry, plan, cache, products):
    metadata, raw = load_source(ROOT, sid)
    if metadata["raw_fits_sha256"] != item["base_raw_fits_sha256"] or str(
        metadata["origin_tpf_kic"]
    ) != str(item["metadata"]["origin_tpf_kic"]):
        raise ValueError("Live source identity differs")
    if set(map(int, item["pixels"])) != set(truth["observed_quarters"]):
        raise ValueError("Declared quarter coverage differs")
    delta_path = (folder / item["deltas"]).resolve()
    verify_hash(delta_path, item["sha256"], products)
    expected_deltas = injections.delta_arrays(raw, truth, plan)
    with np.load(delta_path, allow_pickle=False) as deltas:
        if set(deltas.files) != set(expected_deltas):
            raise ValueError("Flux-delta fields differ")
        for field, expected in expected_deltas.items():
            if not arrays_equal(deltas[field], expected):
                raise ValueError(
                    f"Flux injection differs from declared morphology: {sid} {field}"
                )
    quarter_results = [
        verify_pixel(
            folder, sid, truth, entry, raw, astrometry, plan["signal"], cache, products
        )
        for entry in item["pixels"].items()
    ]
    failures = []
    if any(not row["transfer_matches_declared_truth"] for row in quarter_results):
        failures.append("PIXEL_TIMING_TRANSFER_MISMATCH")
    if any(not row["structural_support"] for row in quarter_results):
        failures.append("PIXEL_GEOMETRY_UNSUPPORTED")
    return {
        "source_id": sid,
        "category": truth["category"],
        "status": "VERIFIED" if not failures else "PHYSICAL_INPUT_FAILED",
        "failure_reasons": failures,
        "quarters": quarter_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--prf-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    folder = args.materialization.resolve()
    report = read(folder / "materialization.json")
    if report["status"] != "FROZEN":
        raise ValueError("Physical materialization incomplete")
    products = {}
    for name, expected in report["artifacts"].items():
        verify_hash(folder / name, expected, products)
    verify_hash(
        args.partitions / "freeze.json", report["partition_freeze_sha256"], products
    )
    manifest = read(folder / "input-manifest.json")
    truth = rows(folder / "truth.jsonl")
    expected_truth = pixels.partition_rows(
        args.partitions, report["partition"], category=report.get("category")
    )
    if (
        truth != expected_truth
        or [row["source_id"] for row in truth] != report["source_ids"]
        or len(truth) != report["sources"]
    ):
        raise ValueError("Physical truth or ordered input scope differs")
    astrometry = {
        str(row["source_id"]): row
        for row in read(folder / "gaia-astrometry.json")["rows"]
    }
    plan = read(injections.PLAN)
    results = []
    with (args.out / "trials.jsonl").open("x") as stream:
        for row in truth:
            sid = row["source_id"]
            result = verify_source(
                folder,
                sid,
                row,
                manifest["sources"][sid],
                astrometry[sid],
                plan,
                args.prf_cache.resolve(),
                products,
            )
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "source_id": sid,
                        "complete": len(results),
                        "total": len(truth),
                        "status": result["status"],
                    }
                ),
                flush=True,
            )
    summary = {
        "schema": 1,
        "status": "PHYSICAL_TRANSFER_VERIFIED"
        if all(row["status"] == "VERIFIED" for row in results)
        else "PHYSICAL_TRANSFER_FAILED",
        "sources": len(results),
        "categories": dict(Counter(row["category"] for row in results)),
        "source_states": dict(Counter(row["status"] for row in results)),
        "pixel_products": sum(len(row["quarters"]) for row in results),
        "pixel_transfer_mismatches": sum(
            not pixel["transfer_matches_declared_truth"]
            for row in results
            for pixel in row["quarters"]
        ),
        "unsupported_pixel_products": sum(
            not pixel["structural_support"]
            for row in results
            for pixel in row["quarters"]
        ),
        "products_hashed": len(products),
        "materialization_sha256": digest(folder / "materialization.json"),
        "manifest_sha256": digest(folder / "input-manifest.json"),
        "trial_ledger_sha256": digest(args.out / "trials.jsonl"),
        "audit_implementation_sha256": digest(__file__),
        "physics_implementation": pixels.generator_identity(),
        "signal_plan_sha256": digest(injections.PLAN),
        "scientific_outcomes_opened": False,
        "boundary": (
            "Input transfer only: immutable truth, flux deltas, TPF/PRF bytes, "
            "source geometry, exact quarter coverage, unchanged non-flux data "
            "and rounded pixel amplitudes. No detector, vetter, difference-image "
            "attribution or disposition outcomes read. Cross-cohort independence "
            "remains a separate audit."
        ),
    }
    (args.out / "audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary))
    if summary["status"] != "PHYSICAL_TRANSFER_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
