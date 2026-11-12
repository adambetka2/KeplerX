"""Select the first server-queue sources with usable delivered PSF Flat records."""

import argparse
import base64
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = (
    "pipeline/photometry/lc2.py",
    "pipeline/deterministic/run.py",
    "pipeline/deterministic/signal_contracts.py",
)
COUNTS = (
    "finite_psf_flat_kbonus",
    "quality_zero_rows",
    "flatten_mask_zero_rows",
)
REMOTE = """import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path("/home/adam/spacelab")
sys.path.insert(0,str(root))
from pipeline.photometry.lc2 import curve_digest,photometry,read_source_rows,source_row
from pipeline.deterministic.run import InsufficientCadences,prepare_curve_records
request=json.load(sys.stdin)
for name,expected in request["bindings"].items():
 with (root/name).open("rb") as stream:
  if hashlib.file_digest(stream,"sha256").hexdigest()!=expected:
   raise ValueError("Server preparation source differs: "+name)
columns=["source_id","time_bkjd","cadenceno","quarter",
 "psf_flat_kbonus_flux","psf_flat_kbonus_flux_err","sap_quality","flatten_mask"]
items=sorted(request["items"],key=lambda row:(int(row["shard"]),row["source_id"]))
for item in items:
 sid=int(item["source_id"]); shard=int(item["shard"])
 volume="discSSD" if shard%2==0 else "discPCIe"
 folder=Path("/srv/kbonus/data")/volume/"data/kbonus/lc2"/f"shard-{shard:04d}"
 table=read_source_rows(folder/"cadence.parquet",sid,columns)
 curve=photometry({name:table[name].to_numpy() for name in columns},"psf_flat")
 metadata=source_row(folder,sid)
 result={"source_id":str(sid),"shard":shard,"view":"psf_flat",
  "raw_fits_sha256":metadata["raw_fits_sha256"],"curve_sha256":curve_digest(curve),
  "flatten_sha256":hashlib.sha256(curve["flatten"].tobytes()).hexdigest(),
  "stored_cadences":len(curve["time"]),"preparation_status":"INPUT_INELIGIBLE"}
 try:
  records=prepare_curve_records(curve,mask_flatten=True)
 except (InsufficientCadences,ValueError) as error:
  result["first_failure"]=f"{type(error).__name__}: {error}"
 else:
  result.update(preparation_status="PREPARED",prepared_cadences=len(records),
   prepared_quarters=len(np.unique(records["quarter"])),
   timespan_days=float(np.ptp(records["time"])),
   records_sha256=hashlib.sha256(records.tobytes()).hexdigest(),
   retained_cadence_sha256=hashlib.sha256(records["cadence"].tobytes()).hexdigest())
 print(json.dumps(result,allow_nan=False),flush=True)
"""


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(items, bindings, out, batch):
    encoded = base64.b64encode(REMOTE.encode()).decode()
    command = (
        "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
        "nice -n 19 /srv/kbonus/dwarfs/runtime/bin/python -B -c "
        f"'import base64;exec(base64.b64decode(\"{encoded}\"))'"
    )
    allowed = {row["source_id"] for row in items}
    seen = set()
    with (out / f"batch-{batch:03d}.stderr").open("xb") as errors:
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "kbonus", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        process.stdin.write(json.dumps({"items": items, "bindings": bindings}).encode())
        process.stdin.close()
        with (out / "validation.jsonl").open("a") as log:
            for line in process.stdout:
                row = json.loads(line)
                sid = row["source_id"]
                if sid not in allowed or sid in seen:
                    raise ValueError(f"Unexpected or duplicate server source: {sid}")
                seen.add(sid)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if len(seen) % 100 == 0:
                    print(
                        json.dumps({"batch": batch, "validated": len(seen)}), flush=True
                    )
    if process.wait() or seen != allowed:
        raise RuntimeError(f"Incomplete source validation; inspect batch {batch}")


def select(queue, receipts, count):
    selected, excluded = [], []
    for rank, item in enumerate(queue, 1):
        row = receipts.get(item["source_id"])
        if row is None:
            break
        entry = item | {"server_queue_rank": rank} | row
        if row["preparation_status"] == "PREPARED":
            selected.append(entry)
            if len(selected) == count:
                break
        else:
            excluded.append(entry)
    return selected, excluded


def load_receipts(queue, census_path, log):
    table = pq.read_table(census_path, columns=["source_id", *COUNTS])
    census = {str(row["source_id"]): row for row in table.to_pylist()}
    receipts = {}
    for item in queue:
        upper = min(census[item["source_id"]][name] for name in COUNTS)
        if upper < 100:
            receipts[item["source_id"]] = {
                "preparation_status": "METADATA_UPPER_BOUND_BELOW_100",
                "cadence_upper_bound": upper,
            }
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row["source_id"] in receipts:
                raise ValueError("Duplicate validation receipt")
            receipts[row["source_id"]] = row
    return receipts


def build(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "selection.json").exists():
        raise ValueError("A frozen real-source selection already exists")
    bindings = {name: digest(ROOT / name) for name in PREPARATION}
    identity = {
        "queue": str(args.queue),
        "queue_sha256": digest(args.queue),
        "census": str(args.census),
        "census_sha256": digest(args.census),
        "driver_sha256": digest(__file__),
        "preparation_bindings": bindings,
        "count": args.count,
        "view": "psf_flat",
        "mask_flatten": True,
        "minimum_prepared_cadences": 100,
        "ordering": "Original server queue; input availability only",
    }
    recipe = args.out / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != identity:
        raise ValueError("Unfinished selection inputs or preparation changed")
    save(recipe, identity)
    queue = json.loads(args.queue.read_text())["order"]
    if len({row["source_id"] for row in queue}) != len(queue):
        raise ValueError("Duplicate source IDs in the server queue")
    log = args.out / "validation.jsonl"
    receipts = load_receipts(queue, args.census, log)
    batch = len(list(args.out.glob("batch-*.json")))
    selected, excluded = select(queue, receipts, args.count)
    while len(selected) < args.count:
        remaining = args.count - sum(
            row["preparation_status"] == "PREPARED" for row in receipts.values()
        )
        items = [row for row in queue if row["source_id"] not in receipts][
            : max(remaining + 100, 100)
        ]
        if not items:
            raise ValueError("Server queue contains too few usable Flat sources")
        batch += 1
        save(args.out / f"batch-{batch:03d}.json", items)
        validate(items, bindings, args.out, batch)
        for line in log.read_text().splitlines():
            row = json.loads(line)
            receipts[row["source_id"]] = row
        selected, excluded = select(queue, receipts, args.count)
    write_csv(args.out / "real5000.csv", selected)
    write_csv(args.out / "excluded-prefix.csv", excluded)
    summary = {
        "status": "FROZEN_USABLE_PSF_FLAT_SELECTION",
        "prepared_at": datetime.now(UTC).isoformat(),
        "selected": len(selected),
        "population": dict(Counter(row["population"] for row in selected)),
        "last_server_queue_rank": selected[-1]["server_queue_rank"],
        "excluded_prefix": len(excluded),
        "excluded_statuses": dict(
            Counter(row["preparation_status"] for row in excluded)
        ),
        "prepared_cadences": sum(row["prepared_cadences"] for row in selected),
        "minimum_prepared_cadences": min(row["prepared_cadences"] for row in selected),
        "g_range": [
            min(float(row["phot_g_mean_mag"]) for row in selected),
            max(float(row["phot_g_mean_mag"]) for row in selected),
        ],
        "bindings": {
            str(path): digest(path)
            for path in (
                recipe,
                log,
                args.out / "real5000.csv",
                args.out / "excluded-prefix.csv",
            )
        },
        "real_detection": "DEFERRED_UNTIL_HOME",
    }
    save(args.out / "selection.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    build(args)


if __name__ == "__main__":
    main()
