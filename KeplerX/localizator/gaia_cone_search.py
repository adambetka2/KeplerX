"""Gaia DR3 cone searches from the ESA Gaia archive, in VizieR's table form."""

import os
import time
from pathlib import Path

import diskcache
import numpy as np
import requests
from astropy.table import MaskedColumn, Table

from common import ROOT
from localizator import gaia_field

ARCHIVE = "https://gea.esac.esa.int/tap-server/tap/sync"
DECIMALS = {"Gmag": 6, "pmRA": 3, "pmDE": 3, "RA_ICRS": 11, "DE_ICRS": 11}
UNITS = {
    "Gmag": "mag",
    "pmRA": "mas / yr",
    "pmDE": "mas / yr",
    "RA_ICRS": "deg",
    "DE_ICRS": "deg",
}


def radius_arcsec(tpf_shape, mission, radius=60):
    """Upstream's search radius: the TPF's half-extent plus two pixels."""
    if mission == "kepler":
        return ((np.nanmax(tpf_shape) + 4) * 0.5) * 4
    if mission == "tess":
        return ((np.nanmax(tpf_shape) + 4) * 0.5) * 21
    return radius


def adql(ra, dec, radius):
    return (
        "SELECT source_id, phot_g_mean_mag, pmra, pmdec, ra, dec FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra!r}, {dec!r}, {radius / 3600!r})) "
        "AND phot_g_mean_mag <= 25"
    )


def published(values, decimals):
    """Values as VizieR publishes them: rounded in decimal, read back as doubles."""
    return np.array([float(f"{v:.{decimals}f}") for v in np.asarray(values, float)])


def table(columns, served_by="ESA Gaia archive, gaiadr3.gaia_source"):
    """VizieR's I/355/gaiadr3 table from gaiadr3.gaia_source columns."""
    if len(columns["source_id"]) == 0:
        return None
    order = np.argsort(columns["source_id"], kind="stable")  # VizieR's row order
    sources = {
        "Gmag": "phot_g_mean_mag", "pmRA": "pmra", "pmDE": "pmdec",
        "RA_ICRS": "ra", "DE_ICRS": "dec",
    }  # fmt: skip
    result = Table(masked=True)
    names = np.array(
        [f"Gaia DR3 {s}" for s in columns["source_id"][order]], dtype="<U28"
    )
    result["DR3Name"] = MaskedColumn(names, mask=np.zeros(len(names), bool))
    for name, source in sources.items():
        values = np.asarray(columns[source], dtype=float)[order]
        mask = np.isnan(values)
        values = published(np.where(mask, 0.0, values), DECIMALS[name])
        result[name] = MaskedColumn(values, mask=mask, unit=UNITS[name])
    result.meta = {"name": "I/355/gaiadr3", "served_by": served_by}
    return result


def cache_directory():
    return (
        Path(os.environ.get("KEPLERX_KERNEL_CACHE", ROOT / "runtime" / "cache"))
        / "gaia-dr3"
    )


class ArchiveUnreachable(RuntimeError):
    """The ESA archive did not answer a cone search after every attempt."""


OVERFLOW = b'<INFO name="QUERY_STATUS" value="OVERFLOW"/>'


def complete(votable):
    """The archive's VOTable, refused when it holds only the first MAXREC rows."""
    if OVERFLOW in votable:
        raise ValueError("The archive truncated the result at its row limit")
    return votable


def fetch(ra, dec, radius, attempts=4, timeout=60):
    """The archive's VOTable for one cone, retried with backoff."""
    error = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                ARCHIVE,
                data={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "votable",
                    "QUERY": adql(ra, dec, radius),
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return complete(response.content)
        except requests.RequestException as exc:
            error = exc
            time.sleep(2**attempt)
    raise ArchiveUnreachable(
        f"ESA Gaia archive unreachable after {attempts} attempts: {error}"
    )


def cone_key(ra_targ, dec_targ, tpf_shape, mission, radius=60):
    """(ra, dec, radius in arcsec) of upstream's cone for this TPF."""
    rad = float(radius_arcsec(tpf_shape, mission, radius))
    return round(float(ra_targ), 9), round(float(dec_targ), 9), rad


def archive_columns(ra, dec, rad):
    """The archive's columns for one cone (radius in arcsec), from the kept responses or the archive."""
    key = (round(ra, 9), round(dec, 9), rad)
    with diskcache.Cache(str(cache_directory())) as cache:
        votable = cache.get(key)
        if votable is None:
            votable = fetch(ra, dec, rad)
            cache.set(key, votable)
    return gaia_field.votable_columns(votable)


class Cones:
    """One case's cone searches, as upstream asks them."""

    def __init__(self, field=None):
        self.field, self.widest, self.tables = field, {}, {}

    def __call__(self, ra_targ, dec_targ, tpf_shape, mission, radius=60):
        key = cone_key(ra_targ, dec_targ, tpf_shape, mission, radius)
        if key not in self.tables:
            self.tables[key] = self.search(float(ra_targ), float(dec_targ), key)
        return self.tables[key]

    def search(self, ra, dec, key):
        rad = key[2]
        if self.field is not None and self.field.covers(ra, dec, rad / 3600):
            rows = self.field.cone(ra, dec, rad / 3600)
            return table(
                self.field.take(rows), f"Field table of {self.field.meta['catalogue']}"
            )
        widest = self.widest.get(key[:2])
        if widest is None or widest[0] < rad:
            columns = archive_columns(ra, dec, rad)
            widest = (
                rad,
                gaia_field.Field.of(columns, (ra - 1, ra + 1), (dec - 1, dec + 1)),
            )
            self.widest[key[:2]] = widest
        return table(widest[1].take(widest[1].cone(ra, dec, rad / 3600)))
