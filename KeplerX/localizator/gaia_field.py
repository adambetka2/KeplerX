"""Gaia DR3 cone searches from a local table of a whole field."""

import argparse
import io
import json
import math
import shutil
from pathlib import Path

import numpy as np

from common import write

COLUMNS = ("source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag")
ZONE_ARRAYS = ("zone_start", "zone_ra")
DTYPES = {"source_id": np.int64, "phot_g_mean_mag": np.float32}
ZONE_DEG = 1.0 / 60.0
MAXREC = 3_000_000  # the archive's anonymous ceiling; its default is 50,000


class Field:
    """Sources sorted into declination zones, with their zone offsets."""

    def __init__(self, columns, zone_start, zone_ra, meta):
        self.columns, self.zone_start, self.zone_ra = columns, zone_start, zone_ra
        self.meta = meta
        self.origin, self.height = meta["dec_origin"], meta["zone_deg"]

    @classmethod
    def load(cls, folder):
        """A field folder, its columns memory mapped."""
        folder = Path(folder)
        columns = {
            name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in COLUMNS
        }
        meta = json.loads((folder / "field.json").read_text())
        zone_start, zone_ra = (np.load(folder / f"{n}.npy") for n in ZONE_ARRAYS)
        return cls(columns, zone_start, zone_ra, meta)

    @classmethod
    def of(cls, columns, ra, dec):
        """A field in memory from columns, covering the ra and dec ranges."""
        columns, zone_start = zoned(columns, dec)
        meta = {
            "ra": list(ra),
            "dec": list(dec),
            "dec_origin": dec[0],
            "zone_deg": ZONE_DEG,
            "catalogue": "gaiadr3.gaia_source",
        }
        return cls(columns, zone_start, box(ra, len(zone_start) - 1), meta)

    def covers(self, ra, dec, radius_deg):
        dec_lo, dec_hi = self.meta["dec"]
        if not dec_lo <= dec - radius_deg <= dec + radius_deg <= dec_hi:
            return False
        first, last = self.zones(dec, radius_deg)
        spans = self.zone_ra[first : last + 1]
        spread = radius_deg / max(math.cos(math.radians(abs(dec) + radius_deg)), 1e-9)
        return bool(
            (spans[:, 0] <= ra - spread).all() and (ra + spread <= spans[:, 1]).all()
        )

    def zones(self, dec, radius_deg):
        """First and last zone a cone overlaps."""
        last = len(self.zone_start) - 2
        first_zone = max(int((dec - radius_deg - self.origin) // self.height), 0)
        return first_zone, min(
            int((dec + radius_deg - self.origin) // self.height), last
        )

    def cone(self, ra, dec, radius_deg):
        """Row indices of the sources within radius_deg of (ra, dec), in source_id order (VizieR's)."""
        ra_column, dec_column = self.columns["ra"], self.columns["dec"]
        first_zone, last_zone = self.zones(dec, radius_deg)
        # RA half-width at the circle's highest |dec|, widened so the exact test decides every edge.
        spread = radius_deg / max(math.cos(math.radians(abs(dec) + radius_deg)), 1e-9)
        spread *= 1.0 + 1e-9
        rows = []
        for zone in range(first_zone, last_zone + 1):
            start, stop = self.zone_start[zone], self.zone_start[zone + 1]
            zone_ra = ra_column[start:stop]
            lo = np.searchsorted(zone_ra, ra - spread, "left")
            hi = np.searchsorted(zone_ra, ra + spread, "right")
            rows.append(np.arange(start + lo, start + hi))
        rows = np.concatenate(rows) if rows else np.empty(0, np.int64)
        near = separation(ra, dec, ra_column[rows], dec_column[rows]) <= radius_deg
        rows = rows[near]
        return rows[np.argsort(self.columns["source_id"][rows], kind="stable")]

    def take(self, rows):
        return {name: np.asarray(column[rows]) for name, column in self.columns.items()}


def separation(ra0, dec0, ra, dec):
    """Great-circle distance in degrees (haversine)."""
    ra0, dec0 = math.radians(ra0), math.radians(dec0)
    ra, dec = np.radians(ra), np.radians(dec)
    h = np.sin((dec - dec0) / 2) ** 2
    h += math.cos(dec0) * np.cos(dec) * np.sin((ra - ra0) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.minimum(h, 1.0))))


def zoned(columns, dec):
    """Sources within the declination range sorted by zone and right ascension, and each zone's first row."""
    zone = np.floor((columns["dec"] - dec[0]) / ZONE_DEG).astype(np.int64)
    keep = (zone >= 0) & (columns["dec"] < dec[1])
    zone = zone[keep]
    order = np.lexsort((np.asarray(columns["ra"])[keep], zone))
    columns = {
        name: np.asarray(columns[name])[keep][order].astype(DTYPES.get(name, np.float64))
        for name in COLUMNS
    }  # fmt: skip
    zones = int(math.ceil((dec[1] - dec[0]) / ZONE_DEG))
    return columns, np.searchsorted(zone[order], np.arange(zones + 1)).astype(np.int64)


def box(ra, zones):
    """Every zone holding the same right-ascension span."""
    return np.tile(np.array(ra, np.float64), (zones, 1))


def save(out, columns, ra, dec, sources, zone_ra=None):
    """Sort sources into zones and write a field folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    columns, zone_start = zoned(columns, dec)
    zone_ra = box(ra, len(zone_start) - 1) if zone_ra is None else zone_ra
    for name in COLUMNS:
        np.save(out / f"{name}.npy", columns[name])
    np.save(out / "zone_start.npy", zone_start)
    np.save(out / "zone_ra.npy", zone_ra)
    write(out / "field.json", {
        "ra": list(ra), "dec": list(dec), "dec_origin": dec[0], "zone_deg": ZONE_DEG,
        "sources": int(len(columns["source_id"])), "magnitude_limit": 25,
        "catalogue": "gaiadr3.gaia_source", "inputs": sources,
    })  # fmt: skip


def votable_columns(votable):
    """The six columns of an archive VOTable response, NaN where masked."""
    from astropy.io.votable import parse_single_table

    raw = parse_single_table(io.BytesIO(votable)).to_table()
    named = {name.lower(): raw[name] for name in raw.colnames}
    return {
        name: np.ma.filled(named[name].astype(DTYPES.get(name, np.float64)), np.nan)
        if name != "source_id" else np.asarray(named[name], np.int64)
        for name in COLUMNS
    }  # fmt: skip


def archive_job(query, poll=15, timeout=4 * 3600):
    """An ADQL query run as an archive job."""
    import time

    import requests

    from localizator.gaia_cone_search import ARCHIVE, ArchiveUnreachable, complete

    data = {
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "PHASE": "RUN",
        "MAXREC": MAXREC,
    }  # fmt: skip
    job = requests.post(
        ARCHIVE.replace("/sync", "/async"), data=data | {"QUERY": query},
        allow_redirects=False, timeout=300,
    )  # fmt: skip
    job.raise_for_status()
    url = job.headers["Location"]
    deadline = time.monotonic() + timeout
    while (
        phase := requests.get(f"{url}/phase", timeout=300).text.strip()
    ) != "COMPLETED":
        if phase in ("ERROR", "ABORTED") or time.monotonic() > deadline:
            raise ArchiveUnreachable(f"Archive job {url} ended {phase}")
        time.sleep(poll)
    result = requests.get(f"{url}/results/result", timeout=3600)
    result.raise_for_status()
    return complete(result.content)


def spans(ra, dec, strip, margin):
    """Declination strips covering the sources with margin to spare."""
    lo = math.floor((dec.min() - margin) / strip) * strip
    hi = math.ceil((dec.max() + margin) / strip) * strip
    edges = lo + strip * np.arange(round((hi - lo) / strip) + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        near = (dec >= a - margin) & (dec < b + margin)
        if near.any():
            spread = margin / math.cos(math.radians(max(abs(a), abs(b)) + margin))
            ra_lo = math.floor((ra[near].min() - spread) * 100) / 100
            ra_hi = math.ceil((ra[near].max() + spread) * 100) / 100
            rows.append((float(a), float(b), ra_lo, ra_hi))
    return (float(edges[0]), float(edges[-1])), rows


def build(out, ra, dec, strip, margin):
    """Download the strips the sources need from the archive."""
    out = Path(out)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    dec_range, strips = spans(ra, dec, strip, margin)
    zones = int(math.ceil((dec_range[1] - dec_range[0]) / ZONE_DEG))
    zone_ra = np.tile([np.inf, -np.inf], (zones, 1))
    parts, sources = [], []
    for lo, hi, ra_lo, ra_hi in strips:
        query = (
            f"SELECT {', '.join(COLUMNS)} FROM gaiadr3.gaia_source "
            f"WHERE dec >= {lo!r} AND dec < {hi!r} AND ra >= {ra_lo!r} "
            f"AND ra < {ra_hi!r} AND phot_g_mean_mag <= 25"
        )
        path = out / "strips" / f"dec-{lo:.4f}-ra-{ra_lo:.2f}-{ra_hi:.2f}.npz"
        if not path.exists():
            columns = votable_columns(archive_job(query))
            temporary = path.with_suffix(".partial.npz")
            np.savez(temporary, **columns)
            temporary.replace(path)
        with np.load(path) as stored:
            parts.append({name: stored[name] for name in COLUMNS})
        first = round((lo - dec_range[0]) / ZONE_DEG)
        zone_ra[first : round((hi - dec_range[0]) / ZONE_DEG)] = (ra_lo, ra_hi)
        sources.append({"query": query, "rows": len(parts[-1]["source_id"])})
        print(f"dec {lo:.2f}-{hi:.2f} ra {ra_lo:.2f}-{ra_hi:.2f}: "
              f"{sources[-1]['rows']} sources", flush=True)  # fmt: skip
    columns = {name: np.concatenate([p[name] for p in parts]) for name in COLUMNS}
    ra_range = (min(r[2] for r in strips), max(r[3] for r in strips))
    save(out, columns, ra_range, dec_range, sources, zone_ra)
    shutil.rmtree(out / "strips")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--strip", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.25)
    args = parser.parse_args(argv)
    import pandas as pd

    from calibration.lane_census import INVENTORY

    sources = pd.read_parquet(INVENTORY, columns=["ra_deg", "dec_deg"])
    ra, dec = sources.ra_deg.to_numpy(), sources.dec_deg.to_numpy()
    build(args.out, ra, dec, args.strip, args.margin)


if __name__ == "__main__":
    main()
