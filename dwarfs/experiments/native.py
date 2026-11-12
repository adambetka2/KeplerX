"""Local paired checks of the unchanged production detector configurations."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1",
    NUMBA_NUM_THREADS="1",
)

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dwarfs/shallow"), str(Path(__file__).parent)]
import evolution
import preparation

DTYPE = [
    ("time", "f8"),
    ("flux", "f8"),
    ("error", "f8"),
    ("quarter", "i4"),
    ("cadence", "i8"),
]


DESIGNS_PATH = Path(__file__).with_name("native-designs.json")


def designs():
    return evolution.read(DESIGNS_PATH)["designs"]


def inputs(entry, folder):
    with np.load(entry["path"], allow_pickle=False) as data:
        curve = {name: data[name].copy() for name in data.files}
    use = np.isfinite(curve["time"])
    use &= curve["time"] <= np.nanmin(curve["time"]) + 45
    curve = {
        key: np.broadcast_to(value, use.shape)[use] for key, value in curve.items()
    }
    if "source_flux" not in curve:
        raise ValueError("Explicit physical source-flux response required")
    folder.mkdir()
    np.savez_compressed(folder / "raw.npz", **curve)
    start, stop = float(np.min(curve["time"])), float(np.max(curve["time"]))
    truth = {
        "period_days": 9.9731,
        "epoch_bkjd": start + 2.731,
        "duration_hours": 4.0,
        "start_bkjd": start,
        "stop_bkjd": stop,
        "exposure_days": preparation.DESIGN["exposure_days"],
    }
    cases = []
    for depth in (0, 1000, 3000):
        loss = (
            preparation.physical_loss(
                curve["time"], truth["period_days"], truth["epoch_bkjd"], 4, depth
            )
            if depth
            else np.zeros(len(curve["time"]))
        )
        injected = preparation.raw_injection(curve, loss)
        indices, arrays = preparation.relative_curve(injected)
        records = np.empty(len(indices), dtype=DTYPE)
        for name, values in zip(records.dtype.names, arrays):
            records[name] = values
        path = folder / f"depth-{depth}.npz"
        np.savez_compressed(
            path,
            records=records,
            indices=indices,
            injected_flux=injected["flux"],
            loss=loss,
        )
        support = loss[indices] > 0
        event = np.rint((records["time"] - truth["epoch_bkjd"]) / truth["period_days"])
        cases.append(
            {
                "key": f"{entry['name']}-{depth}",
                "path": str(path),
                "sha256": evolution.digest(path),
                "array_key": "records",
                "source_id": entry["source_id"],
                "depth_ppm": depth,
                "support": {
                    "nominal_area_ratio_ppm": depth,
                    "maximum_model_loss_ppm": float(loss.max() * 1e6),
                    "prepared_loss_cadences": int(support.sum()),
                    "prepared_events": len(np.unique(event[support])),
                },
            }
        )
    return cases, truth


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    entries = evolution.read(args.manifest)
    jobs, truths = [], {}
    for entry in entries:
        cases, truth = inputs(entry, args.out / entry["name"])
        jobs.extend(cases)
        truths.update({case["key"]: truth for case in cases})
    bound = [Path(__file__), Path(preparation.__file__), DESIGNS_PATH]
    bound.extend(
        ROOT / name for name in ("pipeline/runtime.py", "pipeline/photometry/lc2.py")
    )
    for folder in ("dwarfs/shallow", "pipeline/shallow", "pipeline/deterministic"):
        bound.extend(
            path for path in (ROOT / folder).iterdir() if path.suffix in (".py", ".c")
        )
    source_root = args.out / "source"
    source_bindings = {}
    for path in bound:
        target = source_root / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_bindings[str(target)] = evolution.digest(target)
    (source_root / "dump").symlink_to(
        (ROOT / "dump").resolve(), target_is_directory=True
    )
    source_bindings.update(
        evolution.bind_files(Path(entry["path"]) for entry in entries)
    )
    source_bindings.update(
        evolution.bind_files(
            [
                Path(__file__),
                Path(preparation.__file__),
                Path(evolution.__file__),
                DESIGNS_PATH,
            ]
        )
    )
    evolution.save(
        args.out / "identity.json",
        {
            "status": "PREPARED_UNSEARCHED",
            "jobs": jobs,
            "designs": designs(),
            "bindings": source_bindings,
            "truths_sha256": None,
            "source_root": str(source_root),
            "receipts": {},
            "input_manifest": entries,
            "runtime": evolution.runtime_identity(),
            "protocol": {
                "span_days": 45,
                "depth_ppm": [0, 1000, 3000],
                "duration_hours": 4,
                "period_days": 9.9731,
                "search": "unchanged full production period bands",
                "comparison": "factorial biweight window and physical spline spacing",
                "controls": "one original uninjected curve per source; not null-calibrated",
                "limits": "small conditional check, not a completeness or reliability estimate",
            },
        },
    )
    evolution.save(args.out / "truths.json", truths)
    identity = evolution.read(args.out / "identity.json")
    identity["truths_sha256"] = evolution.digest(args.out / "truths.json")
    evolution.save(args.out / "identity.json", identity)


def run(args):
    identity = evolution.read(args.out / "identity.json")
    if identity["status"] != "PREPARED_UNSEARCHED":
        raise ValueError(
            "Fresh prepared experiment required; preserve earlier failures"
        )
    evolution.check_bindings(identity["bindings"])
    if evolution.runtime_identity() != identity["runtime"]:
        raise ValueError("Runtime differs from prepared experiment")
    os.environ["NUMBA_CACHE_DIR"] = str(args.out / "numba-cache")
    identity["status"] = "BLIND_SEARCH_RUNNING"
    evolution.save(args.out / "identity.json", identity)
    for design in identity["designs"]:
        folder = args.out / design["name"]
        folder.mkdir()
        source = Path(identity["source_root"]) / "dwarfs/shallow"
        evolution.initialize(source, design["engine"], design["config"])
        for job in identity["jobs"]:
            target = folder / (job["key"] + ".json")
            result = evolution.scan_job((job, target))
            identity["receipts"][str(target)] = result["sha256"]
            evolution.save(args.out / "identity.json", identity)
            print(json.dumps(result), flush=True)
    evolution.check_bindings(identity["bindings"])
    if evolution.runtime_identity() != identity["runtime"]:
        raise ValueError("Runtime changed during experiment")
    identity["status"] = "BLIND_SEARCH_COMPLETE_UNSCORED"
    evolution.save(args.out / "identity.json", identity)


def score(args):
    identity = evolution.read(args.out / "identity.json")
    if identity["status"] != "BLIND_SEARCH_COMPLETE_UNSCORED":
        raise ValueError("Complete unscored blind search required")
    evolution.check_bindings(identity["bindings"])
    evolution.check_bindings(identity["receipts"])
    if evolution.runtime_identity() != identity["runtime"]:
        raise ValueError("Scoring runtime differs")
    if evolution.digest(args.out / "truths.json") != identity["truths_sha256"]:
        raise ValueError("Truth changed")
    truths, rows = evolution.read(args.out / "truths.json"), []
    for design in identity["designs"]:
        for job in identity["jobs"]:
            path = args.out / design["name"] / (job["key"] + ".json")
            if str(path) not in identity["receipts"]:
                raise ValueError("Unbound result")
            result = evolution.read(path)
            if result["input"] != job:
                raise ValueError("Result input differs from frozen case")
            evolution.check_bindings(
                {str(path.with_suffix(".npz")): result["arrays"]["sha256"]}
            )
            proposals = result["proposals"]
            matched = [p for p in proposals if evolution.matches(p, truths[job["key"]])]
            rows.append(
                {
                    "design": design["name"],
                    "source_id": job["source_id"],
                    "case": job["key"],
                    "depth_ppm": job["depth_ppm"],
                    "status": result["status"],
                    "first_failure": result["first_failure"],
                    "proposals": len(proposals),
                    "matched_proposals": matched,
                    "support": job["support"],
                    "recovered": bool(matched) if job["depth_ppm"] else None,
                    "search_seconds": result["search_seconds"],
                }
            )
    evolution.save(args.out / "scores.json", rows)
    identity["status"] = "SCORED"
    evolution.save(args.out / "identity.json", identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    {"prepare": prepare, "run": run, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
