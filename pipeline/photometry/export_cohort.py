"""Freeze the existing 406,547-source LC2 cohort for fresh MAST retrieval."""

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq

TABLES = ("source", "cadence", "qdiag", "quarter", "header", "aperture")


def export_shard(root, out, shard):
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    folder = root / volume / f"data/kbonus/lc2/shard-{shard:04d}"
    parquet = pq.ParquetFile(folder / "source.parquet")
    columns = [
        name for name in parquet.schema_arrow.names if name != "primary_header_cards"
    ]
    rows = parquet.read(columns=columns, use_threads=False).to_pylist()
    for row in rows:
        row.update(shard=shard, volume=volume)
    with gzip.open(out / f"shard-{shard:04d}.json.gz", "wt") as stream:
        json.dump(rows, stream, allow_nan=False, separators=(",", ":"))
    return len(rows), sum(row["raw_fits_bytes"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in TABLES:
        path = args.root / f"discSSD/data/kbonus/lc2/shard-0000/{name}.parquet"
        schema = pq.ParquetFile(path).schema_arrow
        (args.out / f"{name}.arrow").write_bytes(schema.serialize().to_pybytes())
    with ThreadPoolExecutor(args.workers) as pool:
        totals = list(pool.map(partial(export_shard, args.root, args.out), range(814)))
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.out.glob("shard-*.json.gz"))
    }
    summary = dict(
        sources=sum(item[0] for item in totals),
        raw_bytes=sum(item[1] for item in totals),
        shards=814,
        files=files,
    )
    if summary["sources"] != 406547:
        raise ValueError("The existing corpus is not the 406,547-source KBonus cohort")
    (args.out / "cohort.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({key: value for key, value in summary.items() if key != "files"})


if __name__ == "__main__":
    main()
