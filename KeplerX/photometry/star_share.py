"""The star's expected Kepler-band flux, and its share of each quarter's SAP."""

import math
import re

import numpy as np

# Delivered level at Kp 12 on the Gaia G zero point (fit over 654 lane hosts with G < 16).
FLUX_AT_KP12 = 2.16e5
# Kp = G + slope * max(0, BP-RP - knot): a quantile fit over 25,689 lane curves.
KEPLER_COLOUR_KNOT = 1.75
KEPLER_COLOUR_SLOPE = 0.148
# Below this share a quarter holds too little of the star's light to correct.
SHARE_FLOOR = 0.01
QUARTER_KEYS = ("FLFRCSAP", "PSFFRAC")


def kepler_band_mag(g_mag, bp_rp):
    """Kepler-band magnitude on the Gaia G zero point."""
    if bp_rp is None or not np.isfinite(bp_rp):
        return g_mag
    return g_mag + KEPLER_COLOUR_SLOPE * max(0.0, bp_rp - KEPLER_COLOUR_KNOT)


def expected_flux(g_mag, bp_rp):
    """The star's Kepler-band flux in e-/s, or None without a G magnitude."""
    if g_mag is None or not np.isfinite(g_mag):
        return None
    return FLUX_AT_KP12 * 10 ** (-0.4 * (kepler_band_mag(g_mag, bp_rp) - 12))


def header_values(cards, keys):
    """The numeric values of `keys` among a FITS header's 80-character cards."""
    out = {}
    for card in re.findall(r".{80}", cards or ""):
        key = card[:8].strip()
        if key in keys and card[8:10] == "= ":
            try:
                out[key] = float(card[10:].split("/")[0].strip().strip("'"))
            except ValueError:
                pass
    return out


def colour(primary):
    """Gaia G and BP-RP from a KBonus primary header (a dict of values)."""
    g = primary.get("GMAG")
    bp, rp = primary.get("BPMAG"), primary.get("RPMAG")
    bp_rp = bp - rp if bp is not None and rp is not None else None
    finite = lambda x: x is not None and math.isfinite(x)  # noqa: E731
    return (g if finite(g) else None), (bp_rp if finite(bp_rp) else None)


def sap_shares(g_mag, bp_rp, quarters):
    """{quarter: share} from {quarter: {"level", "FLFRCSAP", "PSFFRAC"}}."""
    expected = expected_flux(g_mag, bp_rp)
    shares = {}
    for quarter, values in quarters.items():
        level = values.get("level")
        capture, inside = values.get("FLFRCSAP"), values.get("PSFFRAC")
        known = all(
            v is not None and math.isfinite(v) for v in (level, capture, inside)
        )
        if expected is None or not known or level <= 0:
            shares[int(quarter)] = 1.0
        else:
            shares[int(quarter)] = min(1.0, capture * inside * expected / level)
    return shares


def lc2_quarters(folder, source_id, primary_cards=None):
    """G, BP-RP and each quarter's SAP level and aperture keys from an LC2 shard."""
    from photometry.program import source_table

    if primary_cards is None:
        source = source_table(
            folder / "source.parquet", source_id, ["primary_header_cards"]
        )
        primary_cards = source["primary_header_cards"][0].as_py()
    primary = header_values(primary_cards, ("GMAG", "BPMAG", "RPMAG"))
    levels = source_table(
        folder / "quarter.parquet", source_id, ["quarter", "median_sap_flux"]
    )
    headers = source_table(
        folder / "header.parquet", source_id, ["extname", "header_cards"]
    )
    quarters = {int(q): {"level": v} for q, v in zip(
        levels["quarter"].to_pylist(), levels["median_sap_flux"].to_pylist(), strict=True)}  # fmt: skip
    for name, cards in zip(
        headers["extname"].to_pylist(), headers["header_cards"].to_pylist(), strict=True
    ):
        match = re.fullmatch(r"LIGHTCURVE_Q(\d+)", str(name))
        if match and int(match.group(1)) in quarters:
            quarters[int(match.group(1))] |= header_values(cards, QUARTER_KEYS)
    return (*colour(primary), quarters)


def fits_quarters(hdus):
    """The same from an open KBonus FITS file."""
    primary = {k: hdus[0].header.get(k) for k in ("GMAG", "BPMAG", "RPMAG")}
    primary = {k: float(v) for k, v in primary.items() if v is not None}
    quarters = {}
    for hdu in hdus[1:]:
        match = re.fullmatch(r"LIGHTCURVE_Q(\d+)", hdu.name)
        if not match:
            continue
        sap = np.asarray(hdu.data["SAP_FLUX"], float)
        sap = sap[np.isfinite(sap)]
        quarters[int(match.group(1))] = {
            "level": float(np.median(sap)) if len(sap) else None,
            **{k: float(hdu.header[k]) for k in QUARTER_KEYS if k in hdu.header},
        }
    return (*colour(primary), quarters)


def per_cadence(quarter, shares):
    """Each cadence's share from its quarter's; quarters without one keep 1."""
    return np.array([shares.get(int(q), 1.0) for q in quarter], dtype=float)
