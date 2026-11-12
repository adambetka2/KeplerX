"""Field-wide ephemeris matching against the Kepler DR25 TCE and cumulative KOI tables.

TRICERATOPS does not model optical ghosts, crosstalk or another star's eclipses reaching the
aperture (Coughlin et al. 2014). A catalogue object matches when its period times m/n (m, n <= 5,
coprime) drifts by less than one transit duration over the 1,460-day mission and one of its
events falls within one duration of a candidate event. The chance expectation uses the local
catalogue period density (a 1% window) and uniform epochs. A low-order (1:1, 2:1, 1:2) match on
another star stays unresolved until a pixel-level analysis explains it.

python -m post_boi.ephemeris CLOCKS_JSON OUTPUT_JSON. CLOCKS_JSON maps names to period_days,
epoch_bkjd, duration_days, ra and dec (degrees). Catalogue responses are saved beside the output.
"""

import json
import math
import subprocess
import sys
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from post_boi.records import read, receipt, require, write

BASELINE_DAYS = 1460.0
LOW_ORDER = ("1/1", "2/1", "1/2")
RATIOS = [(m, n) for m in range(1, 6) for n in range(1, 6) if math.gcd(m, n) == 1]
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=json&query="
CATALOGUES = {
    "dr25_tce": (
        "select kepid,tce_plnt_num,tce_period,tce_time0bk,ra,dec from q1_q17_dr25_tce",
        "tce_period",
        "tce_time0bk",
    ),
    "koi": (
        "select kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,ra,dec from cumulative",
        "koi_period",
        "koi_time0bk",
    ),
}


def separation_deg(ra1, dec1, ra2, dec2):
    a, b = math.radians(dec1), math.radians(dec2)
    return math.degrees(
        math.acos(
            min(
                1.0,
                math.sin(a) * math.sin(b)
                + math.cos(a) * math.cos(b) * math.cos(math.radians(ra1 - ra2)),
            )
        )
    )


def matches(clock, rows, period_key, epoch_key):
    """Coincident catalogue rows and the expected number of chance coincidences per ratio."""
    P, T, D = clock["period_days"], clock["epoch_bkjd"], clock["duration_days"]
    tolerance, window = D * P / BASELINE_DAYS, 0.01 * P
    hits, expected = [], {}
    for r in rows:
        p2, t2 = r[period_key], r[epoch_key]
        if not p2 or t2 is None:
            continue
        for m, n in RATIOS:
            offset = abs(p2 * m / n - P)
            if offset >= window:
                continue
            grid = min(P, p2)
            expected[f"{m}/{n}"] = (
                expected.get(f"{m}/{n}", 0.0)
                + min(1.0, 2 * D / grid) * tolerance / window
            )
            dt = (t2 - T + grid / 2) % grid - grid / 2
            if offset < tolerance and abs(dt) < D:
                sep = separation_deg(clock["ra"], clock["dec"], r["ra"], r["dec"])
                hits.append(
                    {
                        "row": r,
                        "ratio": f"{m}/{n}",
                        "epoch_offset_days": dt,
                        "separation_deg": sep,
                        "same_star": sep < 2 / 3600,
                    }
                )
    return hits, expected


def screen(clock, catalogues):
    hits, expected = [], {}
    for name, (_, period_key, epoch_key) in CATALOGUES.items():
        h, e = matches(clock, catalogues[name], period_key, epoch_key)
        hits += [dict(catalogue=name, **x) for x in h]
        expected = {k: expected.get(k, 0.0) + e.get(k, 0.0) for k in {*expected, *e}}
    low = [h for h in hits if h["ratio"] in LOW_ORDER and not h["same_star"]]
    return {
        "clock": clock,
        "matches": hits,
        "low_order_other_star": low,
        "expected_chance": sum(expected.values()),
        "expected_chance_low_order": sum(expected.get(k, 0.0) for k in LOW_ORDER),
    }


def fetch(name, directory):
    path = directory / f"ephemeris-catalogue-{name}.json"
    if not path.exists():
        url = TAP + urllib.parse.quote_plus(CATALOGUES[name][0])
        write(
            f"{path}.request.json", {"utc": datetime.now(UTC).isoformat(), "url": url}
        )
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--max-time",
                "600",
                "--output",
                str(path),
                url,
            ],
            check=True,
        )
    rows = json.loads(path.read_text())
    require(len(rows) > 1000, f"Catalogue {name} returned {len(rows)} rows")
    return rows


def main(clocks_path, out):
    catalogues = {name: fetch(name, out.parent) for name in CATALOGUES}
    results = {
        name: screen(clock, catalogues) for name, clock in read(clocks_path).items()
    }
    write(
        out,
        {
            "utc": datetime.now(UTC).isoformat(),
            "helper": receipt(__file__),
            "clocks": receipt(clocks_path),
            "catalogues": {
                n: receipt(out.parent / f"ephemeris-catalogue-{n}.json")
                for n in CATALOGUES
            },
            "results": results,
            "method": __doc__.strip(),
        },
    )
    return results


if __name__ == "__main__":
    for name, r in main(Path(sys.argv[1]), Path(sys.argv[2])).items():
        print(
            name,
            len(r["matches"]),
            "matches",
            f"(chance {r['expected_chance']:.2f});",
            len(r["low_order_other_star"]),
            f"low-order other-star (chance {r['expected_chance_low_order']:.2f})",
        )
