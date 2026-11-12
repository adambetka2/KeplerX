"""Raw-pixel refits followed by the unchanged full production detector."""

import argparse
import json
import shutil
from pathlib import Path

import native
import numpy as np
import reprocess

METHODS = ("sap", "pixels", "mission_only")
PROTOCOL = {
    "period_days": 9.9731,
    "epoch_offset_days": 2.731,
    "duration_hours": 4.0,
    "depth_ppm": [0, 1000, 3000],
    "span": "All supplied common full-quarter native cadences",
    "methods": METHODS,
    "detector": "Unchanged selected production design; 0.5/1d biweight and3d physical knots",
    "error": "Same original aperture diagonal error for every arm; detector noise estimation unchanged",
    "injection": "Delivered PSF median times primary moving spatial profile; refit both adaptive arms after every raw-pixel injection",
    "controls": "Original curve and two positive controls per source and arm",
    "limits": "Positive-control endpoint check, not false-alarm calibration or programme gate",
}


def extract(data, loss):
    names = (
        "pixel_flux",
        "time",
        "pointing",
        "target_profiles",
        "aperture",
        "background",
        "pixel_error",
        "profile_derivatives",
        "cbv",
    )
    arguments = {key: data[key] for key in names}
    arguments["pixel_flux"] = (
        data["pixel_flux"]
        - float(data["source_flux"]) * loss[:, None] * data["target_profiles"]
    )
    adaptive = reprocess.fit_extract(**arguments)
    mission = reprocess.fit_extract(**arguments, reference_pixels=False)
    fluxes = (adaptive["baseline_flux"], adaptive["flux"], mission["flux"])
    return fluxes, adaptive, mission


def inputs(entry, folder):
    if reprocess.digest(entry["path"]) != entry["sha256"]:
        raise ValueError("Pixel bundle changed")
    with np.load(entry["path"], allow_pickle=False) as archive:
        data = dict(archive)
    folder.mkdir()
    time = data["time"]
    truth = {
        "period_days": PROTOCOL["period_days"],
        "epoch_bkjd": float(time[0] + PROTOCOL["epoch_offset_days"]),
        "duration_hours": PROTOCOL["duration_hours"],
        "start_bkjd": float(time[0]),
        "stop_bkjd": float(time[-1]),
        "exposure_days": native.preparation.DESIGN["exposure_days"],
    }
    jobs = []
    for depth in PROTOCOL["depth_ppm"]:
        loss = (
            native.preparation.physical_loss(
                time, truth["period_days"], truth["epoch_bkjd"], 4, depth
            )
            if depth
            else np.zeros(len(time))
        )
        fluxes, adaptive, mission = extract(data, loss)
        native.evolution.save(
            folder / f"fit-{depth}.json",
            {
                "pixels": adaptive["diagnostics"],
                "mission_only": mission["diagnostics"],
            },
        )
        for method, flux in zip(METHODS, fluxes):
            curve = {
                "time": time,
                "flux": flux,
                "error": adaptive["formal_error"],
                "cadence": data["cadence"],
                "quality": np.zeros(len(time), dtype=int),
                "quarter": np.full(len(time), int(entry["name"].rsplit("-q", 1)[1])),
            }
            indices, arrays = native.preparation.relative_curve(curve)
            records = np.empty(len(indices), dtype=native.DTYPE)
            for name, values in zip(records.dtype.names, arrays):
                records[name] = values
            path = folder / f"{method}-{depth}.npz"
            np.savez_compressed(
                path, records=records, indices=indices, flux=flux, loss=loss
            )
            support = loss[indices] > 0
            events = np.rint(
                (time[indices] - truth["epoch_bkjd"]) / truth["period_days"]
            )
            jobs.append(
                {
                    "key": f"{entry['name']}-{method}-{depth}",
                    "path": str(path),
                    "sha256": reprocess.digest(path),
                    "array_key": "records",
                    "source_id": entry["source_id"],
                    "depth_ppm": depth,
                    "method": method,
                    "support": {
                        "prepared_cadences": len(indices),
                        "prepared_loss_cadences": int(support.sum()),
                        "prepared_events": len(np.unique(events[support])),
                        "maximum_model_loss_ppm": float(loss.max() * 1e6),
                    },
                }
            )
    return jobs, truth


def prepare(args):
    selected = native.designs()[0]
    native.inputs = inputs
    native.designs = lambda: [selected]
    native.prepare(args)
    identity = native.evolution.read(args.out / "identity.json")
    identity["protocol"] = PROTOCOL
    sources = [Path(__file__), Path(reprocess.__file__)]
    for path in sources:
        target = Path(identity["source_root"]) / path.relative_to(native.ROOT)
        shutil.copy2(path, target)
        identity["bindings"][str(target)] = reprocess.digest(target)
    identity["bindings"].update(native.evolution.bind_files(sources))
    identity["bindings"].update(
        native.evolution.bind_files(args.out.glob("*/fit-*.json"))
    )
    native.evolution.save(args.out / "identity.json", identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    try:
        {"prepare": prepare, "run": native.run, "score": native.score}[args.action](
            args
        )
    except Exception as error:
        if args.out.exists():
            path = args.out / "endpoint-first-failure.json"
            if not path.exists():
                path.write_text(
                    json.dumps({"action": args.action, "error": repr(error)}, indent=2)
                    + "\n"
                )
        raise


if __name__ == "__main__":
    main()
