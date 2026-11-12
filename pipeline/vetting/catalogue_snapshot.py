"""Prepare complete, immutable catalogue snapshots for exact-signal vetting."""

import argparse
import csv
import hashlib
import http.cookiejar
import io
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retrieve(raw, name, url, opener, data=None, headers=None):
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "ForgottenStars-catalogue-vetting/1.0",
            **(headers or {}),
        },
    )
    requested = datetime.now(timezone.utc).isoformat()
    with opener.open(request, timeout=120) as response:
        content = response.read()
        receipt = {
            "status": response.status,
            "url": url,
            "method": request.get_method(),
            "final_url": response.url,
            "requested_utc": requested,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    suffix = ".json" if "TAP/sync" in url else ".txt"
    (raw / (name + suffix)).write_bytes(content)
    (raw / (name + ".receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return content


def refresh(raw):
    raw.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    queries = {
        "koi-count": "select count(*) as n from cumulative",
        "tce-count": "select count(*) as n from q1_q17_dr25_tce",
        "ps-count": "select count(*) as n from ps",
        "keplerstellar-positions-count": "select count(distinct kepid) as n from keplerstellar",
        "cumulative-full": "select kepid,kepoi_name,kepler_name,koi_disposition,koi_pdisposition,koi_period,koi_period_err1,koi_period_err2,koi_time0bk,koi_time0bk_err1,koi_time0bk_err2,koi_duration,koi_depth,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,ra,dec from cumulative",
        "tce-full": "select kepid,tce_plnt_num,tce_rogue_flag,tce_delivname,tce_period,tce_period_err,tce_time0bk,tce_time0bk_err,tce_duration,tce_depth,tce_fwm_sra,tce_fwm_sdec from q1_q17_dr25_tce",
        "ps-full": "select pl_name,hostname,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_trandur,pl_tsystemref,ra,dec,gaia_dr2_id,gaia_dr3_id,tic_id,discoverymethod,disc_refname,pl_refname,default_flag from ps",
        "keplerstellar-positions": "select distinct kepid,ra,dec from keplerstellar",
    }
    for name, query in queries.items():
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
            + urllib.parse.urlencode(
                {
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "json",
                    "MAXREC": 1000000,
                    "QUERY": query,
                }
            )
        )
        retrieve(raw, name, url, opener)
    url = "https://keplerebs.villanova.edu/"
    content = retrieve(raw, "villanova-index", url, opener)
    token = re.search(
        rb"name=['\"]csrfmiddlewaretoken['\"] value=['\"]([^'\"]+)", content
    )
    if token is None:
        raise ValueError(
            "Cannot request the complete Villanova CSV: CSRF field unavailable"
        )
    data = urllib.parse.urlencode(
        {"format": "file_csv", "csrfmiddlewaretoken": token.group(1).decode()}
    ).encode()
    retrieve(raw, "villanova-full-csv", url, opener, data, {"Referer": url})


def identity(value):
    if value is None:
        return None
    match = re.search(r"(\d+)$", str(value))
    return match.group(1) if match else None


def catalogue_rows(raw, published):
    positions = {}
    for row in read(raw / "keplerstellar-positions.json"):
        if row["ra"] is not None and row["dec"] is not None:
            positions.setdefault(str(row["kepid"]), []).append((row["ra"], row["dec"]))
    signals = []
    for i, row in enumerate(read(raw / "cumulative-full.json")):
        signals.append(
            {
                "catalogue": "NASA_cumulative_KOI",
                "row_index": i,
                "name": row["kepoi_name"],
                "planet_name": row["kepler_name"],
                "kic": str(row["kepid"]),
                "gaia_dr3": None,
                "ra": row["ra"],
                "dec": row["dec"],
                "period": row["koi_period"],
                "epoch": row["koi_time0bk"],
                "time_reference": "BKJD (BJD minus 2454833)",
                "duration_hours": row["koi_duration"],
                "classification": row["koi_disposition"],
                "evidence_url": "https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html",
                "raw": row,
            }
        )
    missing_tce_positions = []
    for i, row in enumerate(read(raw / "tce-full.json")):
        pos = positions.get(str(row["kepid"]))
        source = "NASA keplerstellar KIC position"
        if not pos:
            missing_tce_positions.append(row["kepid"])
            if row["tce_fwm_sra"] is None or row["tce_fwm_sdec"] is None:
                raise ValueError(
                    f"No catalogue/centroid position for TCE {row['kepid']}"
                )
            pos = [(15 * row["tce_fwm_sra"], row["tce_fwm_sdec"])]
            source = "NASA TCE out-of-transit flux-weighted centroid; RA hours converted to degrees"
        signals.append(
            {
                "catalogue": "NASA_Kepler_DR25_TCE",
                "row_index": i,
                "name": f"KIC {row['kepid']} TCE {row['tce_plnt_num']}",
                "kic": str(row["kepid"]),
                "gaia_dr3": None,
                "ra": pos[0][0],
                "dec": pos[0][1],
                "period": row["tce_period"],
                "epoch": row["tce_time0bk"],
                "time_reference": "BKJD (BJD minus 2454833)",
                "duration_hours": row["tce_duration"],
                "classification": "TCE",
                "position_source": source,
                "position_variants": pos,
                "evidence_url": "https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_TCE_docs.html",
                "raw": row,
            }
        )
    for i, row in enumerate(read(raw / "ps-full.json")):
        epoch = row["pl_tranmid"]
        supported_time_reference = row["pl_tsystemref"] in ("BJD", "BJD-TDB", "BJD/TDB")
        signals.append(
            {
                "catalogue": "NASA_confirmed_planets_PS",
                "row_index": i,
                "name": row["pl_name"],
                "kic": None,
                "gaia_dr3": identity(row["gaia_dr3_id"]),
                "ra": row["ra"],
                "dec": row["dec"],
                "period": row["pl_orbper"],
                "epoch": epoch - 2454833
                if epoch is not None and supported_time_reference
                else None,
                "time_reference": f"{row['pl_tsystemref']}; pl_tranmid minus 2454833",
                "duration_hours": row["pl_trandur"],
                "classification": "CONFIRMED",
                "evidence_url": "https://exoplanetarchive.ipac.caltech.edu/overview/"
                + row["pl_name"].replace(" ", "%20"),
                "raw": row,
            }
        )
    lines = (raw / "villanova-full-csv.txt").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("#KIC,"))
    eb = list(
        csv.DictReader(io.StringIO("\n".join([lines[start][1:]] + lines[start + 1 :])))
    )
    missing_eb_positions = []
    for i, row in enumerate(eb):
        kic = str(int(row["KIC"]))
        pos = positions.get(kic)
        if pos:
            ra, dec = pos[0]
            source = "NASA keplerstellar KIC position"
        elif float(row["GLon"]) >= 0 and float(row["GLat"]) > -90:
            sky = SkyCoord(
                l=float(row["GLon"]) * u.deg,
                b=float(row["GLat"]) * u.deg,
                frame="galactic",
            ).icrs
            ra, dec = sky.ra.deg, sky.dec.deg
            source = "Villanova GLon/GLat converted to ICRS"
        else:
            missing_eb_positions.append(kic)
            continue
        signals.append(
            {
                "catalogue": "Villanova_Kepler_EB_live",
                "row_index": i,
                "name": f"KIC {kic}",
                "kic": kic,
                "gaia_dr3": None,
                "ra": ra,
                "dec": dec,
                "period": float(row["period"]),
                "epoch": float(row["bjd0"]) - 54833,
                "time_reference": "BJD0 minus 2400000; converted by subtracting 54833",
                "duration_hours": None,
                "classification": "EB_CATALOGUE",
                "position_source": source,
                "evidence_url": f"https://keplerebs.villanova.edu/overview/?k={kic}",
                "raw": row,
            }
        )
    for i, row in enumerate(read(published)["signals"]):
        signals.append(
            {
                "catalogue": "Bienias_Szabo_2025",
                "row_index": i,
                "name": row["name"],
                "kic": None,
                "gaia_dr3": row["gaia_dr3_source_id"],
                "ra": row["ra"],
                "dec": row["dec"],
                "period": row["period"],
                "epoch": row["epoch"],
                "time_reference": "BKJD, Appendix A Table 4 final fitted ephemeris",
                "duration_hours": None,
                "classification": "PUBLISHED_CANDIDATE",
                "evidence_url": "https://doi.org/10.5281/zenodo.14618891",
                "raw": row,
            }
        )
    stats = {
        "koi_rows": len(read(raw / "cumulative-full.json")),
        "tce_rows": len(read(raw / "tce-full.json")),
        "confirmed_planet_literature_rows": len(read(raw / "ps-full.json")),
        "villanova_eb_rows": len(eb),
        "keplerstellar_position_rows": len(read(raw / "keplerstellar-positions.json")),
        "missing_tce_position_kics": sorted(set(missing_tce_positions)),
        "missing_eb_position_kics": missing_eb_positions,
        "normalized_signal_rows": len(signals),
    }
    return signals, stats


def prepare(raw, published, output):
    raw, published, output = Path(raw), Path(published), Path(output)
    if output.exists():
        raise ValueError(
            "Snapshot destination must be new; frozen snapshots are immutable"
        )
    signals, counts = catalogue_rows(raw, published)
    if counts["missing_eb_position_kics"]:
        raise ValueError(
            f"Unsearchable EB positions: {counts['missing_eb_position_kics']}"
        )
    if (
        len({row["kepid"] for row in read(raw / "keplerstellar-positions.json")})
        != read(raw / "keplerstellar-positions-count.json")[0]["n"]
    ):
        raise ValueError("Incomplete Keplerstellar position snapshot")
    total = re.search(
        r"Showing \d+ of (\d+) results", (raw / "villanova-index.txt").read_text()
    )
    if total is None or counts["villanova_eb_rows"] != int(total.group(1)):
        raise ValueError(
            "EB download is not the full catalogue shown by the live index"
        )
    for name, table, count in (
        ("cumulative-full", "koi-count", counts["koi_rows"]),
        ("tce-full", "tce-count", counts["tce_rows"]),
        ("ps-full", "ps-count", counts["confirmed_planet_literature_rows"]),
    ):
        if count != read(raw / f"{table}.json")[0]["n"]:
            raise ValueError(f"Incomplete {name} snapshot: count query disagrees")
    names = (
        "cumulative-full.json",
        "tce-full.json",
        "ps-full.json",
        "keplerstellar-positions.json",
        "villanova-full-csv.txt",
        "villanova-index.txt",
        "koi-count.json",
        "tce-count.json",
        "ps-count.json",
        "keplerstellar-positions-count.json",
    )
    provenance = []
    for name in names:
        path = raw / name
        receipt = read(path.with_name(path.stem + ".receipt.json"))
        if receipt["status"] != 200 or receipt["sha256"] != digest(path):
            raise ValueError(f"Catalogue retrieval failed or bytes changed: {name}")
        provenance.append(
            {"path": str(path.resolve()), "sha256": digest(path), "request": receipt}
        )
    provenance.append(
        {
            "path": str(published.resolve()),
            "sha256": digest(published),
            "source": read(published)["source"],
            "tables": read(published)["tables"],
        }
    )
    output.mkdir(parents=True)
    rows = output / "signals.json"
    rows.write_text(json.dumps(signals, separators=(",", ":"), allow_nan=False) + "\n")
    manifest = {
        "schema": 1,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "signals": {"path": rows.name, "sha256": digest(rows)},
        "inputs": provenance,
        "normalizer_sha256": digest(__file__),
        "required_catalogues": [
            "NASA_cumulative_KOI",
            "NASA_Kepler_DR25_TCE",
            "NASA_confirmed_planets_PS",
            "Villanova_Kepler_EB_live",
            "Bienias_Szabo_2025",
        ],
        "position_qualification": "TCE rows without target-catalogue coordinates explicitly retain a flux-weighted centroid proxy; this is not a true-source exclusion.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output / "manifest.json"), "counts": counts}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch full catalogues and independent counts into a new --raw directory first",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh(args.raw)
    prepare(args.raw, args.published, args.output)
