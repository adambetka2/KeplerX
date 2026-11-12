"""Witness canonical parent stamp geometry using bounded TPF headers only."""

import argparse
import hashlib
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import requests

from pipeline.deterministic.control_bindings import read_json, reference, require
from pipeline.deterministic.control_donor_headers import verify_targets
from pipeline.deterministic.control_eligibility import BOUNDARY, save, verify_plan
from pipeline.deterministic.control_eligibility_archive import (
    BLOCK,
    capacity,
    header_end,
    header_record,
)

MAXIMUM_PREFIX = 64 * 1024
MAXIMUM_HEADER = 128 * 1024
ARCHIVE = "https://archive.stsci.edu/pub/kepler/target_pixel_files/"


def decode_headers(prefix):
    inflater = zlib.decompressobj(31)
    pending, decoded, headers = prefix, b"", []
    for _ in range(MAXIMUM_HEADER // BLOCK):
        block = inflater.decompress(pending, BLOCK)
        pending = inflater.unconsumed_tail
        require(block, "Compressed prefix does not contain both TPF headers")
        decoded += block
        end = header_end(decoded)
        if end is None or len(decoded) < end:
            continue
        offset = sum(h["bytes"] for h in headers)
        header = header_record(decoded[:end], offset)
        headers.append(header)
        decoded = decoded[end:]
        if len(headers) == 2:
            return headers
        require(
            header["cards"]["NAXIS"] == 0, "TPF primary has an unexpected data body"
        )
    raise ValueError("TPF headers exceed metadata byte bound")


def parent_url(parent, timestamp):
    require(re.fullmatch("[0-9]{13}", timestamp), "Invalid canonical TPF timestamp")
    name = f"{int(parent):09d}"
    require(int(parent) > 0, "Invalid canonical TPF parent")
    return f"{ARCHIVE}{name[:4]}/{name}/kplr{name}-{timestamp}_lpd-targ.fits.gz"


def archive_prefix(session, url, attempts):
    for _ in range(2):
        receipt = {"url": url, "requested_range": f"bytes=0-{MAXIMUM_PREFIX - 1}"}
        started = time.monotonic()
        try:
            with session.get(
                url,
                headers={"Range": receipt["requested_range"]},
                stream=True,
                timeout=(10, 20),
                allow_redirects=False,
            ) as response:
                receipt.update(
                    status=response.status_code,
                    etag=response.headers.get("ETag"),
                    content_range=response.headers.get("Content-Range"),
                )
                require(
                    response.status_code == 206,
                    "TPF archive did not honor metadata Range",
                )
                match = re.fullmatch(
                    r"bytes 0-(\d+)/(\d+)", receipt["content_range"] or ""
                )
                require(
                    match and int(match[1]) < MAXIMUM_PREFIX, "Invalid TPF prefix Range"
                )
                data = response.raw.read(MAXIMUM_PREFIX + 1)
                require(len(data) == int(match[1]) + 1, "TPF prefix byte count differs")
                receipt.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                return decode_headers(data), int(match[2])
        except (requests.RequestException, OSError, ValueError, zlib.error) as error:
            receipt.update(error=type(error).__name__, message=str(error))
        finally:
            receipt["wall_seconds"] = time.monotonic() - started
            attempts.append(receipt)
    raise ValueError("Both canonical TPF header requests failed")


def verify_parent_headers(headers, parent, quarter):
    offset = 0
    for header in headers:
        require(
            header_record(header["raw_header_ascii"].encode("ascii"), offset) == header,
            "TPF header content or identity changed",
        )
        offset += header["bytes"]
    primary, table = (h["cards"] for h in headers)
    require(int(primary["KEPLERID"]) == int(parent), "Canonical TPF parent differs")
    require(int(primary["QUARTER"]) == quarter, "Canonical TPF quarter differs")
    require(table["XTENSION"] == "BINTABLE", "TPF second HDU is not its target table")
    shape = re.fullmatch(r"\((\d+),(\d+)\)", table["TDIM4"])
    require(shape, "TPF stamp dimensions missing")
    return {
        "parent_kic": str(parent),
        "quarter": quarter,
        "module_output_channel": [primary[k] for k in ("MODULE", "OUTPUT", "CHANNEL")],
        "column_origin": int(table["1CRV4P"]),
        "row_origin": int(table["2CRV4P"]),
        "stamp_columns": int(shape[1]),
        "stamp_rows": int(shape[2]),
        "whole_product_hash_verified": False,
        "pixel_flux_read": False,
    }


def quarter_mapping(session, parent, directory):
    name = f"{int(parent):09d}"
    url = f"{ARCHIVE}{name[:4]}/{name}/"
    with session.get(url, timeout=(10, 20), allow_redirects=False) as response:
        require(
            response.status_code == 200 and len(response.content) <= 128 * 1024,
            "Canonical parent directory metadata unavailable or too large",
        )
        names = sorted(
            set(re.findall(r'href="(kplr\d+-(\d+)_lpd-targ.fits.gz)"', response.text))
        )
        listing = {
            "url": url,
            "status": response.status_code,
            "bytes": len(response.content),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "filenames": names,
        }
    save(directory / "directory-listing.json", listing)
    mapping = {}
    for filename, timestamp in names:
        attempts = []
        headers, total = archive_prefix(
            session, parent_url(parent, timestamp), attempts
        )
        quarter = int(headers[0]["cards"]["QUARTER"])
        geometry = verify_parent_headers(headers, parent, quarter)
        require(
            quarter not in mapping, "Canonical timestamp has ambiguous quarter identity"
        )
        mapping[quarter] = timestamp
        save(
            directory / f"mapping-q{quarter}.json",
            {
                "filename": filename,
                "attempts": attempts,
                "headers": headers,
                "product_bytes": total,
                "geometry": geometry,
            },
        )
    require(set(mapping) == set(range(18)), "Quarter timestamp mapping is incomplete")
    save(directory / "quarter-mapping.json", {str(k): v for k, v in mapping.items()})
    return mapping


def stamp_witness(quarter, record):
    cards = quarter["archive_header"]["cards"]
    geometry = record["geometry"]
    same = (
        [cards[k] for k in ("MODULE", "OUTPUT", "CHANNEL")]
        == geometry["module_output_channel"]
        and cards["TPF_ROW"] == geometry["row_origin"]
        and cards["TPF_COL"] == geometry["column_origin"]
    )
    row, col = cards.get("ROW"), cards.get("COLUMN")
    inside = (
        None
        if row is None or col is None
        else bool(
            geometry["row_origin"]
            <= row
            < geometry["row_origin"] + geometry["stamp_rows"]
            and geometry["column_origin"]
            <= col
            < geometry["column_origin"] + geometry["stamp_columns"]
        )
    )
    return {
        "quarter": quarter["quarter"],
        "canonical_parent": geometry["parent_kic"],
        "lightcurve_header_sha256": quarter["archive_header"]["sha256"],
        "canonical_tpf_geometry": geometry,
        "stamp_origin_and_channel_match": bool(same),
        "target_header_position_inside_candidate_stamp": inside if same else None,
        "status": "CANONICAL_PARENT_STAMP_MATCH"
        if same
        else "NOMINAL_PARENT_STAMP_MISMATCH",
        "original_extraction_parent_uniqueness_verified": False,
        "physical_attribution_measured": False,
    }


def collect(plan_path, target_report_path, directory):
    plan = read_json(plan_path)
    verify_plan(plan)
    report = read_json(target_report_path)
    verify_targets(report, plan, plan_path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with requests.Session() as session:
        mapping_path = directory / "quarter-mapping.json"
        mapping = (
            {int(k): v for k, v in read_json(mapping_path).items()}
            if mapping_path.exists()
            else quarter_mapping(session, report["sources"][0]["parent_tpf"], directory)
        )
        for source in report["sources"]:
            quarters = source.get(
                "quarter_witnesses",
                source.get("witness", {}).get("quarter_witnesses", []),
            )
            for quarter in quarters:
                capacity(directory.parent)
                q, parent = quarter["quarter"], source["parent_tpf"]
                path = directory / "records" / f"{parent}-q{q}.json"
                if path.exists():
                    record = read_json(path)
                else:
                    record = {
                        "parent_kic": parent,
                        "quarter": q,
                        "attempts": [],
                        "status": "UNAVAILABLE",
                    }
                    try:
                        headers, total = archive_prefix(
                            session, parent_url(parent, mapping[q]), record["attempts"]
                        )
                        record.update(
                            headers=headers,
                            product_bytes=total,
                            geometry=verify_parent_headers(headers, parent, q),
                            status="VERIFIED_PARENT_HEADERS",
                        )
                    except (
                        ValueError,
                        OSError,
                        KeyError,
                        requests.RequestException,
                    ) as error:
                        record.update(error=type(error).__name__, message=str(error))
                    save(path, record)
                require(
                    record["parent_kic"] == parent and record["quarter"] == q,
                    "Retained canonical parent/quarter identity changed",
                )
                if record["status"] == "VERIFIED_PARENT_HEADERS":
                    require(
                        verify_parent_headers(record["headers"], parent, q)
                        == record["geometry"],
                        "Retained canonical parent geometry changed",
                    )
                witness = (
                    stamp_witness(quarter, record)
                    if record["status"] == "VERIFIED_PARENT_HEADERS"
                    else {
                        "quarter": q,
                        "status": "CANONICAL_PARENT_HEADERS_UNAVAILABLE",
                    }
                )
                rows.append(
                    {
                        "source_id": source["source_id"],
                        "cohort": source["cohort"],
                        "record": reference(path),
                        **witness,
                    }
                )
            print(f"Retained parent witnesses for {source['source_id']}", flush=True)
    result = {
        "schema": 1,
        "plan": reference(plan_path),
        "target_report": reference(target_report_path),
        "implementation": reference(Path(__file__)),
        "quarters": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "boundary": BOUNDARY,
    }
    save(directory / "parent-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "target-report", "directory"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    collect(args.plan, args.target_report, args.directory)


if __name__ == "__main__":
    main()
