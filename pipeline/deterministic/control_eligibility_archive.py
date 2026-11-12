"""Bounded canonical archive transport for prospective eligibility witnesses."""

import hashlib
import math
import re
import shutil
import time
from pathlib import Path

import requests
from astropy.io import fits
from urllib3.exceptions import HTTPError

from pipeline.deterministic.control_bindings import read_json, reference, require
from pipeline.deterministic.control_eligibility import (
    digest,
    header_identity,
    product_identity,
    save,
    verify_plan,
)

MAXIMUM_BYTES = 3 * 1024**3 // 2
MINIMUM_FREE = 2 * 1024**3
BLOCK = 2880
MAX_HEADER = 128 * 1024


def capacity(root, additional=0):
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    free = shutil.disk_usage(root).free
    require(
        used + additional <= MAXIMUM_BYTES, "Dispatch retained-data budget exhausted"
    )
    require(
        free - additional >= MINIMUM_FREE, "Cloud free-space reserve would be crossed"
    )
    return {
        "retained_dispatch_bytes": used,
        "filesystem_free_bytes": free,
        "maximum_retained_bytes": MAXIMUM_BYTES,
        "minimum_free_bytes": MINIMUM_FREE,
    }


def range_request(session, source, start, end, attempts):
    require(
        source["url"] == product_identity(source["product"])["url"],
        "Archive URL changed",
    )
    headers = {"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache"}
    for _ in range(2):
        started = time.monotonic()
        receipt = {"url": source["url"], "request_headers": headers}
        try:
            with session.get(
                source["url"],
                headers=headers,
                timeout=(10, 20),
                stream=True,
                allow_redirects=False,
            ) as response:
                receipt.update(
                    status=response.status_code,
                    content_range=response.headers.get("Content-Range"),
                    etag=response.headers.get("ETag"),
                )
                require(
                    response.status_code == 206, "Archive did not honor bounded Range"
                )
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", receipt["content_range"] or ""
                )
                require(
                    match and tuple(map(int, match.groups()[:2])) == (start, end),
                    "Archive Range identity differs",
                )
                data = response.raw.read(end - start + 2)
                require(
                    len(data) == end - start + 1, "Archive Range byte count differs"
                )
                receipt.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                return data, int(match[3])
        except (requests.RequestException, HTTPError, OSError, ValueError) as error:
            receipt.update(error=type(error).__name__, message=str(error))
        finally:
            receipt["wall_seconds"] = time.monotonic() - started
            attempts.append(receipt)
    raise ValueError("Both bounded archive requests failed; see retained attempts")


def header_record(data, offset):
    text = data.decode("ascii")
    header = fits.Header.fromstring(text, sep="")
    name = header.get("EXTNAME", "PRIMARY")
    return {
        "name": name,
        "offset": offset,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "raw_header_ascii": text,
        "cards": {
            key: header[key]
            for key in header
            if key and key not in {"COMMENT", "HISTORY"}
        },
    }


def header_end(data):
    for offset in range(0, len(data), 80):
        if data[offset : offset + 8] == b"END     ":
            return math.ceil((offset + 80) / BLOCK) * BLOCK
    return None


def read_header(session, source, offset, total, attempts, initial=None):
    data = initial or b""
    while not header_end(data):
        require(len(data) < MAX_HEADER, "FITS header exceeds metadata bound")
        start = offset + len(data)
        require(start + BLOCK <= total, "Truncated archive FITS header")
        block, observed_total = range_request(
            session, source, start, start + BLOCK - 1, attempts
        )
        require(
            observed_total == total, "Archive product size changed between requests"
        )
        data += block
    return header_record(data[: header_end(data)], offset)


def next_header(header, total):
    cards = header["cards"]
    axes = [cards[f"NAXIS{i}"] for i in range(1, cards["NAXIS"] + 1)]
    elements = math.prod(axes) if axes else 0
    body = (
        abs(cards["BITPIX"])
        // 8
        * cards.get("GCOUNT", 1)
        * (elements + cards.get("PCOUNT", 0))
    )
    next_offset = header["offset"] + header["bytes"] + math.ceil(body / BLOCK) * BLOCK
    require(
        header["offset"] < next_offset <= total, "FITS HDU size escapes archive product"
    )
    return next_offset


def remote_headers(session, source, first, total, attempts):
    headers, offset = [], 0
    while offset < total:
        require(len(headers) < 256, "FITS extension inventory exceeds metadata bound")
        header = read_header(
            session, source, offset, total, attempts, first if offset == 0 else None
        )
        headers.append(header)
        offset = next_header(header, total)
    return headers


def local_headers(path):
    result = []
    with fits.open(path, memmap=True) as hdus, path.open("rb") as stream:
        for hdu in hdus:
            info = hdu.fileinfo()
            stream.seek(info["hdrLoc"])
            result.append(
                header_record(
                    stream.read(info["datLoc"] - info["hdrLoc"]), info["hdrLoc"]
                )
            )
    return result


def verify_headers(headers, total):
    offset = 0
    for header in headers:
        decoded = header_record(header["raw_header_ascii"].encode("ascii"), offset)
        require(decoded == header, "Archive header content, cards or hash changed")
        require(
            header_end(header["raw_header_ascii"].encode("ascii")) == header["bytes"],
            "Archive header padding/inventory changed",
        )
        offset = next_header(header, total)
    require(offset == total, "Archive header inventory is incomplete")


def full_product(session, source, path, total, attempts, root):
    capacity(root, total + 16 * 1024**2)
    if path.exists():
        require(
            path.stat().st_size == total
            and digest(path) == source["pool"]["raw_fits_sha256"],
            "Retained source product changed",
        )
        return reference(path)
    data, observed = range_request(session, source, 0, total - 1, attempts)
    require(observed == total, "Archive product size changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    require(
        digest(path) == source["pool"]["raw_fits_sha256"],
        "Original source raw FITS hash differs",
    )
    return reference(path)


def source_record(session, source, directory):
    root = directory.parent
    record = {"source": source, "attempts": [], "status": "UNAVAILABLE"}
    started = time.monotonic()
    try:
        record["capacity_before"] = capacity(root)
        first, total = range_request(session, source, 0, 5759, record["attempts"])
        record["archive_product_bytes"] = total
        require(total >= 5760 and total % BLOCK == 0, "Invalid archive FITS size")
        if source["cohort"] == "guide_star_family":
            path = directory / "payloads" / f"{source['source_id']}.fits"
            record["payload"] = full_product(
                session, source, path, total, record["attempts"], root
            )
            headers = local_headers(path)
            record["status"] = "VERIFIED_FULL_PRODUCT"
        else:
            headers = remote_headers(session, source, first, total, record["attempts"])
            record["status"] = "VERIFIED_ARCHIVE_HEADERS"
        record["headers"] = headers
        verify_headers(headers, total)
        header_identity(source, headers)
    except (requests.RequestException, OSError, ValueError, KeyError) as error:
        record.update(
            status="UNAVAILABLE", error=type(error).__name__, message=str(error)
        )
    record["wall_seconds"] = time.monotonic() - started
    record["capacity_after"] = capacity(root)
    return record


def collect(plan_path, directory):
    plan = read_json(plan_path)
    verify_plan(plan)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    capacity(directory.parent)
    with requests.Session() as session:
        for index, source in enumerate(plan["sources"], 1):
            path = directory / "records" / f"{source['source_id']}.json"
            if path.exists():
                require(
                    read_json(path)["source"] == source,
                    "Existing attempt scope changed",
                )
                continue
            save(path, source_record(session, source, directory))
            if index % 10 == 0:
                print(
                    f"Retained {index}/{len(plan['sources'])} "
                    "unchanged source attempts",
                    flush=True,
                )
