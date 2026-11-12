"""Verify public control pixel bytes and headers with bounded temporary storage."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.deterministic import pixel_injections
from pipeline.runtime import EXTERNAL_DUMPS

RESERVE = 20 * 1024**3
MAXIMUM_PRODUCT = 512 * 1024**2
MAXIMUM_RSS = 2 * 1024**3
DEADLINE_SECONDS = 90


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_request(row):
    if not isinstance(row["source_id"], str) or not row["source_id"].isdigit():
        raise ValueError("A public pixel request has an invalid source identity")
    if row["role"] not in {"development", "evaluation"}:
        raise ValueError("A public pixel request has an undeclared role")
    if type(row["quarter"]) is not int or not 0 <= row["quarter"] <= 17:
        raise ValueError("A public pixel request has an undeclared quarter")
    if row["status"] == "UNRESOLVED_ORIGIN_PRODUCT":
        return
    if row["status"] != "PUBLIC_PRODUCT_REQUEST":
        raise ValueError("A public pixel request has an undeclared state")
    parent = f"{int(row['parent_kic']):09d}"
    pattern = (
        r"https://stpubdata\.s3\.us-east-1\.amazonaws\.com/kepler/public/"
        rf"target_pixel_files/{parent[:4]}/{parent}/kplr{parent}-\d+_lpd-targ\.fits\.gz"
    )
    if not re.fullmatch(pattern, row["transport_uri"]):
        raise ValueError("The public pixel URI differs from its declared parent")
    if not 0 < row["size_bytes"] <= MAXIMUM_PRODUCT:
        raise ValueError("The declared pixel payload exceeds its byte allowance")
    expected = row.get("expected_sha256")
    if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("The declared pixel SHA256 is malformed")


def validate_response(response, row):
    response.raise_for_status()
    if response.status_code != 200:
        raise OSError("The public pixel response is not a complete payload")
    if int(response.headers.get("Content-Length", -1)) != row["size_bytes"]:
        raise OSError("The public pixel length differs from frozen transport metadata")
    expected = row.get("etag")
    if expected is not None and response.headers.get("ETag") != expected:
        raise OSError("The public pixel ETag differs from frozen transport metadata")


def receive_product(row, target, *, reserve=RESERVE):
    if type(reserve) is not int or reserve <= 0:
        raise ValueError("The public pixel disk reserve must be a positive integer")
    if reserve < RESERVE and (
        os.environ.get("SPACELAB_EXECUTION_PROFILE") != "cloud-development"
        or any(target.resolve().is_relative_to(root.resolve()) for root in EXTERNAL_DUMPS)
        or row["role"] != "development"
    ):
        raise ValueError("Public pixel transfers require the original reserve outside cloud development")
    began, size = time.monotonic(), 0
    checksum = hashlib.sha256()
    headers = {"If-Match": row["etag"]} if row.get("etag") else {}
    with requests.get(
        row["transport_uri"], headers=headers, stream=True, timeout=(10, 20)
    ) as response:
        validate_response(response, row)
        with target.open("xb") as stream:
            for chunk in response.iter_content(1024**2):
                size += len(chunk)
                if (
                    size > row["size_bytes"]
                    or time.monotonic() - began > DEADLINE_SECONDS
                ):
                    raise OSError(
                        "The public pixel transfer exceeded its declared bound"
                    )
                if shutil.disk_usage(target.parent).free < reserve:
                    raise OSError(
                        "The public pixel transfer approached the disk reserve"
                    )
                stream.write(chunk)
                checksum.update(chunk)
    if size != row["size_bytes"]:
        raise OSError("The complete public payload byte count differs")
    actual = checksum.hexdigest()
    if row.get("expected_sha256") is not None and actual != row["expected_sha256"]:
        raise ValueError("The public payload differs from its frozen SHA256")
    return {
        "sha256": actual,
        "bytes_received": size,
        "download_wall_seconds": time.monotonic() - began,
        "etag": response.headers.get("ETag"),
    }


def footprint(table, aperture):
    width, height = int(aperture["NAXIS1"]), int(aperture["NAXIS2"])
    columns = [
        i for i in range(1, table["TFIELDS"] + 1) if table[f"TTYPE{i}"] == "FLUX"
    ]
    if len(columns) != 1 or table[f"TDIM{columns[0]}"] != f"({width},{height})":
        raise ValueError("The aperture and FLUX column dimensions differ")
    begin, end = float(table["TSTART"]), float(table["TSTOP"])
    reference = float(table["BJDREFI"]) + float(table["BJDREFF"])
    if not all(math.isfinite(v) for v in (begin, end, reference)) or end <= begin:
        raise ValueError("The public exposure interval is invalid")
    return (height, width), reference + (begin + end) / 2


def target_geometry(aperture, shape, epoch, row):
    astrometry = row.get("astrometry")
    if astrometry is None:
        return {"status": "ASTROMETRY_UNAVAILABLE"}
    if str(astrometry["source_id"]) != row["source_id"]:
        raise ValueError("The target astrometry belongs to another source")
    ra, dec, status = pixel_injections.propagated_position(astrometry, epoch)
    wcs = WCS(aperture)
    if not wcs.has_celestial:
        raise ValueError("The public aperture has no celestial WCS")
    x, y = map(float, wcs.world_to_pixel_values(ra, dec))
    if not all(math.isfinite(v) for v in (x, y, ra, dec)):
        raise ValueError("The propagated target position is nonfinite")
    height, width = shape
    return {
        "status": "TARGET_POINT_GEOMETRY_MEASURED",
        "epoch_jd": epoch,
        "target_sky_deg": [ra, dec],
        "astrometry_status": status,
        "target_pixel_xy": [x, y],
        "stamp_shape_yx": [height, width],
        "target_point_inside_pixel_footprint": (
            -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5
        ),
    }


def header_record(path, row):
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdus:
        primary, table, aperture = (hdus[i].header for i in range(3))
        if int(primary["KEPLERID"]) != row["parent_kic"]:
            raise ValueError("The public TPF header has another parent identity")
        if int(primary["QUARTER"]) != row["quarter"]:
            raise ValueError("The public TPF header has another quarter identity")
        shape, epoch = footprint(table, aperture)
        hashes = {
            name: hashlib.sha256(header.tostring().encode("ascii")).hexdigest()
            for name, header in zip(
                ("primary", "flux_table", "aperture"), (primary, table, aperture)
            )
        }
        try:
            geometry = target_geometry(aperture, shape, epoch, row)
        except (KeyError, ValueError, TypeError) as error:
            geometry = {
                "status": "TARGET_POINT_GEOMETRY_UNRESOLVED",
                "error": f"{type(error).__name__}: {error}",
            }
        return {
            "parent_kic": int(primary["KEPLERID"]),
            "quarter": int(primary["QUARTER"]),
            "channel": int(primary["CHANNEL"]),
            "stamp_shape_yx": list(shape),
            "header_sha256": hashes,
            "aperture_header": aperture.tostring(),
            "target_geometry": geometry,
            "pixel_flux_values_read": False,
        }


def verify_product(index, row, output):
    result = {"request_index": index, "request": row, "payload_retained": False}
    if row["status"] == "UNRESOLVED_ORIGIN_PRODUCT":
        return {**result, "status": "ORIGIN_PRODUCT_UNRESOLVED"}
    target = output / "spool" / f"product-{index:06d}.fits.gz"
    if target.exists() or target.is_symlink():
        raise FileExistsError("The temporary pixel payload path is already occupied")
    try:
        if shutil.disk_usage(output).free < RESERVE + MAXIMUM_PRODUCT:
            raise OSError("The next payload cannot preserve the original disk reserve")
        transfer = receive_product(row, target)
        header = header_record(target, row)
        result.update(
            status="PUBLIC_BYTE_AND_HEADER_IDENTITIES_VERIFIED",
            transfer=transfer,
            header=header,
        )
    except (
        requests.RequestException,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        result.update(
            status="PUBLIC_PRODUCT_VERIFICATION_UNRESOLVED",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        target.unlink(missing_ok=True)
    if target.exists():
        raise OSError("A completed temporary pixel payload was not removed")
    return result


def guard(output):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak if sys.platform == "darwin" else peak * 1024
    if peak > MAXIMUM_RSS:
        raise MemoryError(
            "The public pixel input inventory exceeded its memory allowance"
        )
    if shutil.disk_usage(output).free < RESERVE:
        raise OSError("The public pixel input inventory approached its disk reserve")


def run_requests(rows, output, workers):
    states, records = Counter(), []
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(verify_product, i, row, output): i for i, row in enumerate(rows)
    }
    try:
        for future in as_completed(futures):
            result = future.result()
            index = futures[future]
            if result["request_index"] != index or result["request"] != rows[index]:
                raise ValueError("A completed public pixel request identity differs")
            path = output / "products" / f"product-{index:06d}.json"
            save(path, result)
            records.append({"request_index": index, "sha256": digest(path)})
            states[result["status"]] += 1
            save(
                output / "progress.json",
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_requests": len(records),
                    "declared_requests": len(rows),
                    "states": dict(states),
                },
            )
            guard(output)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return sorted(records, key=lambda row: row["request_index"]), states


def request_specification(path):
    specification = read(path)
    if (
        specification["schema"] != 1
        or specification["status"] != "FROZEN_PUBLIC_PIXEL_INPUT_REQUESTS"
    ):
        raise ValueError("The public pixel request specification is not frozen")
    rows = specification["requests"]
    if not rows or len({(r["source_id"], r["quarter"]) for r in rows}) != len(rows):
        raise ValueError("The public source-quarter requests are empty or duplicated")
    for row in rows:
        validate_request(row)
    inputs = specification["inputs"]
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError(f"A frozen public pixel request input changed: {path}")
    return rows, inputs


def run_inventory(args):
    if args.output.exists():
        raise FileExistsError("The public pixel inventory output already exists")
    rows, inputs = request_specification(args.requests)
    inputs = {
        **inputs,
        **{
            str(path.resolve()): digest(path)
            for path in (
                args.requests,
                Path(__file__),
                Path(pixel_injections.__file__),
                ROOT / "pipeline/vetting/vetting_pixels.py",
            )
        },
    }
    if shutil.disk_usage(args.output.parent).free < RESERVE + 4 * 1024**3:
        raise OSError(
            "The bounded pixel inventory cannot preserve its output allowance"
        )
    args.output.mkdir()
    (args.output / "spool").mkdir()
    (args.output / "products").mkdir()
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            args.output / "experiment.json",
            {
                "status": "PREDECLARED_BOUNDED_PUBLIC_PIXEL_INPUT_VERIFICATION",
                "inputs": inputs,
                "requests": len(rows),
                "workers": args.workers,
                "maximum_product_bytes": MAXIMUM_PRODUCT,
                "minimum_free_bytes": RESERVE,
                "maximum_self_rss_bytes": MAXIMUM_RSS,
                "download_deadline_seconds": DEADLINE_SECONDS,
                "scientific_outcomes_opened": False,
                "boi_catalogue_enabled": False,
                "boundary": (
                    "Complete public payload SHA256, parent/quarter headers and input "
                    "target-point geometry only. Each newly created temporary payload "
                    "is removed after inspection. Original data and caches remain "
                    "read-only. A retained content ledger needs a hash-enforcing "
                    "bounded retrieval loader before scientific evaluation. Geometry "
                    "is not PRF sensitivity, source attribution or a release result."
                ),
            },
        )
        began, cpu = time.monotonic(), time.process_time()
        records, states = run_requests(rows, args.output, args.workers)
        for path, expected in inputs.items():
            if digest(path) != expected:
                raise ValueError("A frozen pixel input changed during verification")
        if any((args.output / "spool").iterdir()):
            raise OSError("The completed inventory still has temporary payloads")
        save(args.output / "results.json", records)
        save(
            args.output / "audit.json",
            {
                "status": "COMPLETE_BOUNDED_PUBLIC_PIXEL_INPUT_RECORDS",
                "requests": len(rows),
                "states": dict(states),
                "experiment_sha256": digest(args.output / "experiment.json"),
                "results_sha256": digest(args.output / "results.json"),
                "wall_seconds": time.monotonic() - began,
                "cpu_seconds": time.process_time() - cpu,
                "temporary_payloads_remaining": 0,
                "scientific_outcomes_opened": False,
                "boi_catalogue_enabled": False,
                "all_actual_pixels_retained": False,
                "bounded_redownload_loader_implemented": False,
                "joint_independence_complete": False,
            },
        )
        save(
            args.output / "progress.json",
            {"status": "COMPLETE", "pid": os.getpid(), "completed_requests": len(rows)},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    os.nice(10)
    run_inventory(args)


if __name__ == "__main__":
    main()
