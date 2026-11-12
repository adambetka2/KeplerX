"""Parent TPFs streamed from MAST into memory."""

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BUCKET = "https://stpubdata.s3.us-east-1.amazonaws.com"
PREFIX = "kepler/public/target_pixel_files"
# Each long-cadence delivery's file stamp and quarter; Q4 was delivered twice.
QUARTERS = {
    "2009131105131": 0, "2009166043257": 1, "2009259160929": 2,
    "2009350155506": 3, "2010009091648": 4, "2010078095331": 4,
    "2010174085026": 5, "2010265121752": 6, "2010355172524": 7,
    "2011073133259": 8, "2011177032512": 9, "2011271113734": 10,
    "2012004120508": 11, "2012088054726": 12, "2012179063303": 13,
    "2012277125453": 14, "2013011073258": 15, "2013098041711": 16,
    "2013131215648": 17,
}  # fmt: skip
ATTEMPTS = 4


class PixelsUnreachable(RuntimeError):
    """MAST did not return a parent's listing or file after every attempt."""


def retried(get, what):
    error = None
    for attempt in range(ATTEMPTS):
        try:
            return get()
        except (requests.RequestException, ValueError, ET.ParseError) as exc:
            error = exc
            time.sleep(2**attempt)
    raise PixelsUnreachable(f"MAST {what} failed {ATTEMPTS} times: {error}")


def listing(parent_kic, session):
    """The parent's long-cadence TPFs on MAST, one per quarter."""
    name = f"{int(parent_kic):09d}"
    prefix = f"{PREFIX}/{name[:4]}/{name}/"

    def get():
        response = session.get(
            BUCKET + "/", params={"list-type": "2", "prefix": prefix}, timeout=60
        )
        response.raise_for_status()
        document = ET.fromstring(response.content)
        if document.findtext("{*}IsTruncated") != "false":
            raise ValueError(f"Truncated listing of {prefix}")
        return [
            (item.findtext("{*}Key"), int(item.findtext("{*}Size")))
            for item in document.findall("{*}Contents")
        ]

    items = {}
    for key, size in sorted(retried(get, f"listing of {prefix}")):
        match = re.fullmatch(
            rf"kplr{name}-(\d{{13}})_lpd-targ\.fits\.gz", key[len(prefix) :]
        )
        if match is None:
            continue  # short cadence, tarballs
        quarter = QUARTERS[match[1]]
        items.setdefault(
            quarter, {"url": f"{BUCKET}/{key}", "bytes": size, "quarter": quarter}
        )
    return [items[quarter] for quarter in sorted(items)]


def session(connections):
    client = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=connections)
    client.mount("https://", adapter)
    return client


def fetch(items, client, connections):
    """(item, bytes) for each item as it arrives, the item given its sha256."""

    def one(item):
        def get():
            response = client.get(item["url"], timeout=600)
            response.raise_for_status()
            if len(response.content) != item["bytes"]:
                raise ValueError(
                    f"{item['url']}: {len(response.content)} bytes, listed {item['bytes']}"
                )
            return response.content

        raw = retried(get, item["url"])
        return item | {"sha256": hashlib.sha256(raw).hexdigest()}, raw

    with ThreadPoolExecutor(min(connections, max(len(items), 1))) as pool:
        for future in as_completed([pool.submit(one, item) for item in items]):
            yield future.result()
