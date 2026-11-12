"""Bridge frozen dwarf-detector proposals into an unchanged private vetting snapshot."""

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ORIGIN = "dwarfs_physical_shallow_v1"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(bindings):
    for name, expected in bindings.items():
        require(digest(name) == expected, f"Frozen artifact changed: {name}")


def bind(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def utc():
    return datetime.now(UTC).isoformat()


def source_paths(plan):
    root = Path(plan["pipeline_root"])
    sys.path[:0] = [
        str(root),
        str(root / "pipeline"),
        str(root / "pipeline/photometry"),
        str(root / "pipeline/vetting"),
    ]


def install(plan_path):
    plan = read(plan_path)
    verify(plan["bindings"])
    source_paths(plan)
    os.environ["SPACELAB_VETTING_CONFIG"] = plan["config"]
    import vetting

    # The bridge adds provenance and complete scene inputs; every measurement and gate stays original.
    vetting.DETERMINISTIC_ORIGINS = vetting.DETERMINISTIC_ORIGINS | {ORIGIN}

    def discovery(queue_path):
        require(
            Path(queue_path).resolve() == Path(plan["queue"]), "Unexpected dwarf queue"
        )
        require(
            digest(queue_path) == plan["bindings"][str(Path(queue_path).resolve())],
            "Dwarf queue changed",
        )
        path = Path(plan["discovery"])
        manifest = read(path)
        require(
            manifest["origin"] == ORIGIN
            and manifest["detector_contract"]
            == read(plan["config"])["discovery_contract"],
            "Dwarf discovery contract differs",
        )
        return path, manifest

    def scene_rows(cohort, metadata):
        path = Path(plan["input_manifest"])
        specification = read(path)
        coverage = specification["scene_coverage"]
        require(
            coverage["original_index_sources"] == 406547
            and coverage["complete_neighbour_cones"],
            "Complete source-scene cones required",
        )
        shards = {source["shard"] for source in metadata.values()}
        return vetting.manifest_scene(
            path, specification["scene_metadata"], metadata, shards, complete_index=True
        )

    vetting.discovery_provenance = discovery
    vetting.scene_rows = scene_rows
    return vetting


def snapshot(args, preview):
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    require(
        not out.exists(), "Follow-up output already exists; never duplicate or restart"
    )
    out.mkdir(parents=True)
    source = out / "source"
    paths = [
        path
        for path in (root / "pipeline").rglob("*")
        if path.is_file()
        and (path.suffix in {".py", ".c", ".json"} or path.name == "requirements.txt")
    ]
    for path in paths:
        target = source / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(__file__, source / "adapter.py")
    (out / "work").mkdir()
    (source / "dump").symlink_to(out / "work", target_is_directory=True)
    data = source / "discSSD/data"
    data.mkdir(parents=True)
    resources = args.resources.resolve()
    (data / "index.sqlite").symlink_to(resources / "available-corpus-index.sqlite")
    for collection in ("kbonus", "pixels"):
        (data / collection).symlink_to(
            Path("/Volumes/discSSD/data") / collection, target_is_directory=True
        )
    (source / "discPCIe").symlink_to(
        Path("/Volumes/discPCIe"), target_is_directory=True
    )
    policy = read(args.detector_policy)
    plan = {
        "status": "AUTHORIZED_LOCAL_DETECTION_AND_VETTING",
        "created_at": utc(),
        "authorization": "Adam requested searching the available Flat subset and running every resulting proposal through our vetting pipeline.",
        "preflight_only": preview,
        "detector_policy": str(args.detector_policy.resolve()),
        "detector": policy["out"],
        "out": str(out),
        "pipeline_root": str(source),
        "python": str(args.python.absolute()),
        "resources": str(resources),
        "workers": policy["workers"],
        "bindings": bind(
            [source / path.relative_to(root) for path in paths]
            + [
                source / "adapter.py",
                args.detector_policy,
                resources / "receipts.json",
                resources / "scene-catalogue.json",
                resources / "available-corpus-index.sqlite",
            ]
        ),
    }
    references = read(resources / "receipts.json")
    manifest = Path(references["complete_catalogue_manifest"])
    catalogues = read(manifest)
    resource_bindings = references["legacy_catalogues"] | {
        references["parent_index"]: references["parent_index_sha256"],
        references["private_index"]: references["private_index_sha256"],
        str(manifest): references["complete_catalogue_manifest_sha256"],
        str(manifest.parent / catalogues["signals"]["path"]): catalogues["signals"][
            "sha256"
        ],
        str(resources / "scene-catalogue.npz"): read(
            resources / "scene-catalogue.json"
        )["scene_sha256"],
    }
    verify(resource_bindings)
    plan["bindings"].update(resource_bindings)
    require(
        digest(resources / "scene-catalogue.npz")
        == read(resources / "scene-catalogue.json")["scene_sha256"],
        "Complete scene catalogue changed",
    )
    save(out / "launch-plan.json", plan)
    return plan


def proposal_rows(plan):
    detector = Path(plan["detector"])
    cases = read(detector / "prepared-cases.json")
    identity = read(detector / "pilot/identity.json")
    verify(identity["source_bindings"] | identity["case_bindings"])
    design = identity["designs"][0]
    if plan["preflight_only"]:
        for case in cases:
            path = (
                detector
                / "pilot"
                / design["name"]
                / (
                    hashlib.sha256(
                        (case["input_sha256"] + ":" + case["array_key"]).encode()
                    ).hexdigest()[:24]
                    + ".json"
                )
            )
            if path.is_file() and read(path).get("proposals"):
                return [(case, path, read(path))]
        raise ValueError("Preflight needs a completed real proposal source")
    require(
        read(detector / "progress.json")
        == {
            "stage": "catalogue",
            "state": "COMPLETE",
            "workers": 10,
            "sources": len(cases),
        },
        "Completed detector catalogue required",
    )
    require(
        identity["status"] == "BLIND_SEARCH_COMPLETE_UNSCORED",
        "All detector searches required",
    )
    verify(identity["result_bindings"])
    catalogue = read(detector / "catalogue.json")
    require(
        digest(detector / "sources.json") == catalogue["sources_sha256"],
        "Detector catalogue changed",
    )
    rows = {row["source_id"]: row for row in read(detector / "sources.json")}
    require(
        len(rows)
        == len(cases)
        == catalogue["sources"]
        == read(plan["detector_policy"])["sources"],
        "Complete source accounting required",
    )
    tasks = []
    for case in cases:
        row = rows[case["source_id"]]
        path = Path(row["result_path"])
        result = read(path)
        require(
            row["proposals"] == result.get("proposals", [])
            and row["status"] == result["status"],
            "Catalogue and detector proposals differ",
        )
        if result.get("proposals"):
            tasks.append((case, path, result))
    return tasks


def export_source(task, plan):
    import numpy as np

    from pipeline.deterministic import run as deterministic
    from pipeline.photometry import lc2

    case, result_path, result = task
    sid = case["source_id"]
    require(
        digest(case["input_path"]) == case["input_sha256"],
        "Prepared input changed before vetting",
    )
    metadata, raw = lc2.load_source(Path(plan["pipeline_root"]), sid)
    metadata = lc2.finite_metadata(metadata)
    curve = lc2.photometry(raw, "psf_flat")
    records = deterministic.prepare_curve_records(curve, mask_flatten=True)
    require(
        hashlib.sha256(records.tobytes()).hexdigest() == case["records_sha256"],
        "Vetting and detector Flat records differ",
    )
    raw_path = Path(plan["out"]) / "work/inputs" / (sid + ".npz")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    require(not raw_path.exists(), "Vetting input already exists")
    np.savez_compressed(raw_path, **raw)
    alerts, hypotheses = [], []
    for ordinal, original in enumerate(result["proposals"], 1):
        row = original | {"source_id": sid, "search_rank": ordinal}
        events, hypothesis = deterministic.shallow_events(row, records, "psf_flat")
        prior = hypothesis["signal_id"]
        identity = f"dwarfs-physical-v1-{sid}-{ordinal}"
        hypothesis.update(
            signal_id=identity, hypothesis_id=identity, discovery_view="psf_flat"
        )
        hypothesis["detector_origin"] = ORIGIN
        hypothesis["shallow_record"] = original | {"source_id": sid}
        hypothesis["forced_photometry_views"] = ["psf_flat"]
        for event in events:
            event["event_id"] = event["event_id"].replace(prior, identity, 1)
            event["detector_signal_id"] = identity
            event["detector_origin"] = ORIGIN
            event["timing_origin"] = "SUPPLIED_DWARF_PHYSICAL_EPHEMERIS"
        hypothesis["alert_ids"] = [event["event_id"] for event in events]
        alerts.extend(events)
        hypotheses.append(hypothesis)
    proposal = {
        "source_id": sid,
        "views": deterministic.views(raw, {"psf_flat"}),
        "alerts": alerts,
        "supplied_hypotheses": hypotheses,
        "rust_rows": [],
        "shallow_rows": result["proposals"],
        "original_detector_result": str(result_path),
        "original_detector_result_sha256": digest(result_path),
    }
    item = {"arrays": str(raw_path), "sha256": digest(raw_path), "metadata": metadata}
    return proposal, item


def scene_inputs(plan, sources):
    import numpy as np
    from scipy.spatial import cKDTree

    path = Path(plan["resources"]) / "scene-catalogue.npz"
    with np.load(path, allow_pickle=False) as stored:
        scene = stored["scene"].copy()
    require(len(scene) == 406547, "Complete original source scene required")

    def xyz(ra, dec):
        ra, dec = np.radians(ra), np.radians(dec)
        return np.column_stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
        )

    points = xyz(scene["ra_deg"], scene["dec_deg"])
    tree = cKDTree(points)
    coordinates = xyz(
        [item["metadata"]["ra_deg"] for item in sources.values()],
        [item["metadata"]["dec_deg"] for item in sources.values()],
    )
    groups = tree.query_ball_point(
        coordinates, 2 * np.sin(np.radians(60 / 3600) / 2) + 1e-12
    )
    selected = sorted({int(index) for group in groups for index in group})
    rows = {
        str(scene[index]["source_id"]): {
            key: scene[index][key].item() for key in scene.dtype.names
        }
        for index in selected
    }
    fields = (
        "ra_deg",
        "dec_deg",
        "phot_g_mean_mag",
        "origin_tpf_kic",
        "shard",
        "raw_fits_sha256",
    )
    for sid, item in sources.items():
        rows[sid] = {"source_id": sid, **{key: item["metadata"][key] for key in fields}}
    for sid, row in rows.items():
        row["source_id"] = sid
        if row["phot_g_mean_mag"] is not None and not np.isfinite(
            row["phot_g_mean_mag"]
        ):
            row["phot_g_mean_mag"] = None
    output = Path(plan["out"]) / "work/scene.jsonl"
    output.write_text(
        "".join(json.dumps(rows[sid], allow_nan=False) + "\n" for sid in sorted(rows))
    )
    return {"path": str(output), "sha256": digest(output)}


def export_worker(task):
    proposal, plan = task
    source_paths(plan)
    return export_source(proposal, plan)


def configure(plan, queue, inputs, discovery):
    source_paths(plan)
    root = Path(plan["pipeline_root"])
    out = Path(plan["out"])
    resources = Path(plan["resources"])
    config = read(root / "pipeline/vetting/config.json")
    references = read(resources / "receipts.json")
    primary = [
        resources / "legacy-catalogues" / (name + "-rows.json")
        for name in ("cumulative", "q1_q17_dr25_koi", "ps", "toi")
    ]
    config.update(
        run_directory=str(out / "work/vetting"),
        queue=str(queue),
        input_manifest=str(inputs),
        disposition_policy=str(out / "work/policy.json"),
        pixel_cache_directory=str(out / "work/pixel-cache"),
        prf_cache_directory=str(out / "work/prf-cache"),
        parent_index=references["parent_index"],
        workers=plan["workers"],
        network_workers=plan["workers"],
        catalogues=[str(path) for path in primary],
        extra_catalogues=[
            str(resources / "legacy-catalogues" / name)
            for name in (
                "kepler-eb-2016-0694ada5b64a.csv",
                "exofop-ctoi-84a829ccae3d.txt",
            )
        ],
        discovery_contract=discovery["detector_contract"],
        shared_alert_scope_complete=True,
    )
    config["catalogue_search"]["manifest"] = references["complete_catalogue_manifest"]
    config["alias"]["workers"] = plan["workers"]
    config_path = out / "work/config.json"
    policy = read(root / "pipeline/vetting/disposition-policy.json")
    save(config["disposition_policy"], policy)
    save(config_path, config)
    os.environ["SPACELAB_VETTING_CONFIG"] = str(config_path)
    import vetting

    policy["pipeline_contract"] = vetting.pipeline_contract(config)
    policy["implementation"] = vetting.code_hashes()
    save(config["disposition_policy"], policy)
    return config_path


def prepare(plan):
    source_paths(plan)
    out = Path(plan["out"])
    tasks = proposal_rows(plan)
    require(tasks, "Detector produced no proposals requiring vetting")
    exports = {}
    with ProcessPoolExecutor(
        plan["workers"], mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(export_worker, (task, plan)) for task in tasks]
        for future in as_completed(futures):
            proposal, item = future.result()
            exports[proposal["source_id"]] = proposal, item
            save(
                out / "progress.json",
                {
                    "stage": "prepare_vetting",
                    "state": "RUNNING",
                    "workers": plan["workers"],
                    "exported": len(exports),
                    "sources": len(tasks),
                },
            )
    queue_rows = [exports[task[0]["source_id"]][0] for task in tasks]
    sources = {task[0]["source_id"]: exports[task[0]["source_id"]][1] for task in tasks}
    scene = scene_inputs(plan, sources)
    inputs = out / "work/input-manifest.json"
    save(
        inputs,
        {
            "schema": 1,
            "measurement_scope": "Delivered native Flat and diagnostic arrays for every reported proposal source, without injection or modification",
            "sources": sources,
            "scene_metadata": scene,
            "scene_coverage": {
                "original_index_sources": 406547,
                "complete_neighbour_cones": True,
                "radius_arcsec": 60,
            },
        },
    )
    identity = read(Path(plan["detector"]) / "pilot/identity.json")
    policy = read(plan["detector_policy"])
    discovery = {
        "origin": ORIGIN,
        "detector_contract": {
            "origin": ORIGIN,
            "discovery_view": "psf_flat",
            "design": policy["design"],
            "detector_source_bindings": identity["source_bindings"],
        },
        "binary": {
            "kind": "native_shared_libraries_and_python_engine",
            "detector_identity": str(Path(plan["detector"]) / "pilot/identity.json"),
        },
        "preflight_only": plan["preflight_only"],
        "sources_searched": policy["sources"],
        "proposal_sources": len(sources),
        "proposals": sum(len(row["shallow_rows"]) for row in queue_rows),
        "detector_runtime": identity["runtime"],
        "original_results": bind([task[1] for task in tasks]),
        "source_inputs": {
            task[0]["input_path"]: task[0]["input_sha256"] for task in tasks
        },
    }
    discovery_path = out / "work/discovery-manifest.json"
    queue_path = out / "work/queue.json"
    save(discovery_path, discovery)
    save(queue_path, {"discovery_manifest": str(discovery_path), "rows": queue_rows})
    config = configure(plan, queue_path, inputs, discovery)
    final = plan | {
        "config": str(config),
        "queue": str(queue_path),
        "discovery": str(discovery_path),
        "input_manifest": str(inputs),
        "bindings": plan["bindings"]
        | bind(
            [
                config,
                inputs,
                scene["path"],
                discovery_path,
                queue_path,
                out / "work/policy.json",
            ]
        ),
    }
    path = out / "runtime-plan.json"
    save(path, final)
    return path


def execute(args):
    os.environ["SPACELAB_DWARF_VETTING_PLAN"] = str(args.plan)
    vetting = install(args.plan)
    plan = read(args.plan)
    out = Path(plan["out"])
    save(
        out / "progress.json",
        {"stage": "vetting", "state": "RUNNING", "workers": plan["workers"]},
    )
    vetting.run_all()
    save(
        out / "progress.json",
        {
            "stage": "vetting",
            "state": "COMPLETE",
            "workers": plan["workers"],
            "results": str(out / "work/vetting/results.json"),
            "completed_at": utc(),
        },
    )


def run_vetting(plan, final):
    command = [
        "/opt/homebrew/bin/uv",
        "run",
        "--no-project",
        "--python",
        plan["python"],
        "python",
        "-B",
        str(Path(plan["out"]) / "source/adapter.py"),
        "execute",
        "--plan",
        str(final),
    ]
    result = subprocess.run(
        command,
        env=os.environ | {"SPACELAB_DWARF_VETTING_PLAN": str(final)},
        check=False,
    )
    require(
        result.returncode == 0,
        f"Vetting exited {result.returncode}; inspect execution.log",
    )


def detector_alive(plan):
    launch = read(Path(plan["detector"]) / "launch.json")
    result = subprocess.run(
        ["ps", "-p", str(launch["pid"]), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"TZ": "UTC", "LC_ALL": "C"},
    )
    line = result.stdout.strip()
    if result.returncode or not line:
        return False
    started = datetime.strptime(line[:24], "%a %b %d %H:%M:%S %Y").replace(tzinfo=UTC)
    return (
        abs((started - datetime.fromisoformat(launch["started_at"])).total_seconds())
        < 3
        and "flat_subset.py execute" in line
    )


def watch(args):
    plan = read(args.plan)
    out = Path(plan["out"])
    try:
        while read(Path(plan["detector"]) / "progress.json")["state"] != "COMPLETE":
            require(
                detector_alive(plan),
                "Detector process stopped before completed catalogue",
            )
            require(
                read(Path(plan["detector"]) / "progress.json")["state"] != "FAILED",
                "Detector failed; preserve its first failure",
            )
            save(
                out / "progress.json",
                {"stage": "wait_detector", "state": "WAITING", "updated_at": utc()},
            )
            time.sleep(60)
        final = prepare(plan)
        run_vetting(plan, final)
    except Exception:
        failure = {"at": utc(), "traceback": traceback.format_exc()}
        save(out / "first-failure.json", failure)
        save(out / "progress.json", {"state": "FAILED", "first_failure": failure})
        raise


def launch(args):
    if args.out.exists():
        return {"status": "ALREADY_STARTED", "out": str(args.out)}
    plan = snapshot(args, False)
    out = Path(plan["out"])
    command = [
        "/opt/homebrew/bin/uv",
        "run",
        "--no-project",
        "--python",
        plan["python"],
        "python",
        "-B",
        str(out / "source/adapter.py"),
        "watch",
        "--plan",
        str(out / "launch-plan.json"),
    ]
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "UV_CACHE_DIR": str(out.parent / "cache/uv"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    with (out / "execution.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wake = subprocess.Popen(
        ["caffeinate", "-i", "-w", str(process.pid)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    receipt = {
        "status": "WAITING_FOR_DETECTOR",
        "started_at": utc(),
        "pid": process.pid,
        "process_group": os.getpgid(process.pid),
        "caffeinate_pid": wake.pid,
        "out": str(out),
        "launch_plan_sha256": digest(out / "launch-plan.json"),
    }
    save(out / "launch.json", receipt)
    return receipt


def preflight(args):
    plan = snapshot(args, True)
    path = prepare(plan)
    run_vetting(plan, path)
    return {"status": "PREFLIGHT_COMPLETE", "out": plan["out"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("launch", "preflight"):
        command = commands.add_parser(name)
        command.add_argument("--detector-policy", type=Path, required=True)
        command.add_argument("--resources", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--python", type=Path, required=True)
    for name in ("watch", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    value = globals()[args.stage](args)
    if value is not None:
        print(json.dumps(value), flush=True)


if os.environ.get("SPACELAB_DWARF_VETTING_PLAN"):
    install(os.environ["SPACELAB_DWARF_VETTING_PLAN"])

if __name__ == "__main__":
    main()
