"""Localization requests: parent TPFs stored on disc, or streamed from MAST."""

import argparse
import gzip
import io
import json
from pathlib import Path

import pandas as pd
from astropy.io import fits

from calibration.lane_census import INVENTORY
from common import digest

MIN_QUARTERS = 1


def parent_kic(name):
    """KIC of a stored TPF: kplr<9 digits>-..._lpd-targ.fits.gz or tpf-<kic>-q.."""
    if name.startswith("kplr"):
        return int(name[4:13])
    if name.startswith("tpf-"):
        return int(name.split("-")[1])
    return None


def quarter_of(path):
    raw = path.read_bytes()
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    return int(fits.getheader(io.BytesIO(data), 0)["QUARTER"])


def index(folders):
    """One row per (KIC, quarter) stored under the folders, the first found."""
    rows, seen = [], set()
    for folder in map(Path, folders):
        for path in sorted(folder.iterdir()):
            kic = parent_kic(path.name)
            if kic is None:
                continue
            quarter = quarter_of(path)
            if (kic, quarter) in seen:
                continue
            seen.add((kic, quarter))
            rows.append({
                "kic": kic, "quarter": quarter, "path": str(path),
                "bytes": path.stat().st_size, "sha256": digest(path),
            })  # fmt: skip
    return pd.DataFrame(rows).sort_values(["kic", "quarter"], ignore_index=True)


def request(source_id, parent_kic, ra_deg, dec_deg, tpfs=None):
    """A source's localization request from its parent's indexed TPFs, or from MAST when tpfs is None."""
    pixels = {"archive": "mast"} if tpfs is None else {"tpfs": [
        {"path": r.path, "bytes": int(r.bytes), "sha256": r.sha256, "quarter": int(r.quarter)}
        for r in tpfs.itertuples()
    ]}  # fmt: skip
    return {
        "source_id": str(source_id), "parent_kic": int(parent_kic),
        "ra_deg": float(ra_deg), "dec_deg": float(dec_deg),
        "coordinate_epoch": "J2016", **pixels, "missing_quarters": [],
    }  # fmt: skip


def attach(index_path, manifest, out, min_quarters=MIN_QUARTERS, archive=None):
    tpfs = dict(tuple(pd.read_parquet(index_path).groupby("kic")))
    sources = pd.read_parquet(
        INVENTORY, columns=["source_id", "origin_tpf_kic", "ra_deg", "dec_deg"]
    ).set_index("source_id")
    attached = 0
    with Path(out).open("w") as stream:
        for line in Path(manifest).read_text().splitlines():
            row = json.loads(line)
            packet = json.loads((Path(row["prepared"]) / "packet.json").read_text())
            source_id = str(packet["source_id"])
            if source_id in sources.index and "localization" not in row:
                parent, ra, dec = sources.loc[source_id]
                stored = tpfs.get(int(parent))
                if stored is not None and len(stored) >= min_quarters:
                    row["localization"] = request(source_id, parent, ra, dec, stored)
                    attached += 1
                elif archive == "mast":
                    row["localization"] = request(source_id, parent, ra, dec)
                    attached += 1
            stream.write(json.dumps(row) + "\n")
    return attached


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("index")
    build.add_argument("out", type=Path)
    build.add_argument("folders", nargs="+", type=Path)
    join = commands.add_parser("attach")
    join.add_argument("index", type=Path)
    join.add_argument("manifest", type=Path)
    join.add_argument("out", type=Path)
    join.add_argument("--min-quarters", type=int, default=MIN_QUARTERS)
    join.add_argument("--archive", choices=["mast"])
    args = parser.parse_args(argv)
    if args.command == "index":
        table = index(args.folders)
        table.to_parquet(args.out)
        print(len(table), "TPFs of", table.kic.nunique(), "parents")
    else:
        count = attach(
            args.index, args.manifest, args.out, args.min_quarters, args.archive
        )
        print(count, "rows given a localization request")


if __name__ == "__main__":
    main()
