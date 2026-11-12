"""Resolve structural target-pixel support without reading science outcomes."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

from deterministic import pixel_injections as pixels
from lc2 import load_source


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    pixels.write_json(path, value)


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def generator_identity():
    identity = pixels.generator_identity()
    identity[str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    return dict(sorted(identity.items()))


def preflight_identity(partitions, base, partition, category, rows):
    return {
        "mode": "PIXEL_SUPPORT_PREFLIGHT",
        "partition": partition,
        "category": category,
        "source_ids": [row["source_id"] for row in rows],
        "partition_freeze_sha256": digest(partitions / "freeze.json"),
        "base_materialization_sha256": digest(base / "materialization.json"),
        "generator": generator_identity(),
    }


def query_provenance(path, out):
    return pixels.spatial_query_provenance(path, out)


def validate_query(query, out):
    if query is None:
        return True
    path = (out / query["path"]).resolve()
    return (
        out.resolve() in path.parents
        and path.exists()
        and digest(path) == query["sha256"]
    )


def validate_record(record, out, row):
    if (
        record.get("source_id") != row["source_id"]
        or record.get("category") != row["category"]
        or record.get("observed_quarters") != row["observed_quarters"]
    ):
        return False
    if not validate_query(record.get("spatial_tpf_query"), out):
        return False
    if record.get("status") == "PIXELS_UNAVAILABLE":
        return bool(record.get("reason"))
    if record.get("status") != "PIXEL_SUPPORTED" or set(
        record.get("pixels", {})
    ) != set(map(str, row["observed_quarters"])):
        return False
    for item in record["pixels"].values():
        path = Path(item["base_path"])
        calibration = pixels.PRF_CACHE / item["prf_calibration"]["filename"]
        if (
            not path.exists()
            or digest(path) != item["base_sha256"]
            or not calibration.exists()
            or digest(calibration) != item["prf_calibration"]["sha256"]
        ):
            return False
    return True


def quarter_support(path, astrometry, quarter):
    base_hash = digest(path)
    with fits.open(path, memmap=False) as hdus:
        if int(hdus[0].header["QUARTER"]) != quarter:
            raise ValueError("Target-pixel quarter identity changed")
        profile, calibration, pixel, sky, status, inside = pixels.pixel_profile(
            hdus, astrometry
        )
        parent = int(hdus[0].header["KEPLERID"])
    return {
        "base_path": str(path),
        "base_parent_kic": parent,
        "base_sha256": base_hash,
        "prf_calibration": calibration,
        "target_pixel": pixel,
        "target_inside_stamp": inside,
        "target_sky_deg": sky,
        "target_astrometry": status,
        "prf_stamp_fraction": float(profile.sum()),
    }


def assess_source(out, row, base_manifest, astrometry, mapping, existing):
    source_id = row["source_id"]
    base_item = base_manifest["sources"][source_id]
    metadata, _ = load_source(ROOT, source_id)
    if metadata["raw_fits_sha256"] != base_item["base_raw_fits_sha256"]:
        raise ValueError(f"Base LC2 source changed: {source_id}")
    query_path = out / "spatial-tpf-queries" / f"{source_id}.json"
    try:
        tpfs = pixels.available_tpfs(
            metadata,
            astrometry,
            row["observed_quarters"],
            mapping,
            existing,
            query_path,
        )
    except pixels.PixelUnavailable as error:
        return {
            "source_id": source_id,
            "category": row["category"],
            "observed_quarters": row["observed_quarters"],
            "status": "PIXELS_UNAVAILABLE",
            "reason": str(error),
            "spatial_tpf_query": query_provenance(query_path, out),
        }
    pixels.prefetch_prfs(fits.getheader(path, 0)["CHANNEL"] for path in tpfs.values())
    return {
        "source_id": source_id,
        "category": row["category"],
        "observed_quarters": row["observed_quarters"],
        "status": "PIXEL_SUPPORTED",
        "pixels": {
            str(quarter): quarter_support(path, astrometry, quarter)
            for quarter, path in sorted(tpfs.items())
        },
        "spatial_tpf_query": query_provenance(query_path, out),
    }


def preflight(
    out,
    partition="development",
    source_id=None,
    category=None,
    limit=None,
    resume=False,
    partitions=pixels.PARTITIONS,
    base=pixels.BASE_INJECTIONS,
):
    out, partitions, base = map(Path, (out, partitions, base))
    freeze, base_report, base_manifest = pixels.validate_inputs(partitions, base)
    rows = pixels.partition_rows(partitions, partition, source_id, category, limit)
    identity = preflight_identity(partitions, base, partition, category, rows)
    if resume:
        progress = pixels.read(out / "progress.json")
        if progress.get("identity") != identity or (out / "preflight.json").exists():
            raise ValueError("Pixel-support resume identity changed or run is complete")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "records").mkdir()
        (out / "spatial-tpf-queries").mkdir()
        progress = {
            "status": "RUNNING",
            "complete": 0,
            "total": len(rows),
            "identity": identity,
        }
        write_json(out / "progress.json", progress)
    astrometry_path = out / "gaia-astrometry.json"
    astrometry = pixels.gaia_astrometry(
        astrometry_path, [row["source_id"] for row in rows]
    )
    mapping, existing = pixels.indexed_pixels()
    records = []
    for row in rows:
        source_id = row["source_id"]
        record_path = out / "records" / f"{source_id}.json"
        if record_path.exists():
            record = pixels.read(record_path)
            if not validate_record(record, out, row):
                raise ValueError(f"Completed pixel-support record changed: {source_id}")
        else:
            record = assess_source(
                out, row, base_manifest, astrometry[source_id], mapping, existing
            )
            write_json(record_path, record)
        records.append(record)
        progress["complete"] = len(records)
        write_json(out / "progress.json", progress)
        print(
            json.dumps(
                {
                    "source_id": source_id,
                    "status": record["status"],
                    "complete": len(records),
                    "total": len(rows),
                }
            ),
            flush=True,
        )
    source_path = out / "source-ids.json"
    support_path = out / "support.jsonl"
    write_json(source_path, [row["source_id"] for row in rows])
    write_jsonl(support_path, records)
    artifacts = {
        path.name: digest(path) for path in (source_path, astrometry_path, support_path)
    }
    statuses = Counter(record["status"] for record in records)
    report = {
        "schema": 1,
        "status": "COMPLETE_PIXEL_SUPPORT_PREFLIGHT",
        "protocol": freeze["protocol"],
        **identity,
        "partition_status": freeze["partitions"][partition]["status"],
        "base_generator": base_report["generator"],
        "sources": len(records),
        "support_statuses": dict(statuses),
        "categories": dict(Counter(record["category"] for record in records)),
        "artifacts": artifacts,
        "claim_boundary": (
            "Resolves public target-pixel and PRF support only. It does not inject "
            "pixels or inspect detector, vetter, localization, or disposition outcomes."
        ),
    }
    write_json(out / "preflight.json", report)
    progress["status"] = "COMPLETE"
    write_json(out / "progress.json", progress)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--partition", choices=("development", "evaluation"), default="development"
    )
    parser.add_argument("--source-id")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partitions", default=pixels.PARTITIONS, type=Path)
    parser.add_argument("--base-injections", default=pixels.BASE_INJECTIONS, type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("Pixel-support limit must be positive")
    print(
        json.dumps(
            preflight(
                args.out,
                args.partition,
                args.source_id,
                args.category,
                args.limit,
                args.resume,
                args.partitions,
                args.base_injections,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
