"""Freeze, retrieve and score a local paired PSF Flat pilot and matched real sample."""

import argparse
import base64
import csv
import hashlib
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "dwarfs/shallow"), str(ROOT / "dwarfs/cohort")]
import calibrate
import evolution as API
import evolve_controls as CONTROLS

from pipeline.deterministic.catalogue_controls import product_uri
from pipeline.deterministic.cloud_public import fits_arrays
from pipeline.deterministic.run import InsufficientCadences, prepare_curve_records
from pipeline.photometry.lc2 import curve_digest, photometry

SEED = "psf-flat-final10000-v7-pilot-20261014-v2"
FIELDS = ("time", "flux", "error", "quality", "flatten", "cadence", "quarter")
NUMERIC = (
    "phot_g_mean_mag",
    "depth_ppm",
    "period_days",
    "duration_hours",
    "quarter_count",
    "flat_cadences",
    "stored_cadences",
)
COUNTS = {
    "synthetic_planet": 600,
    "real_planet": 100,
    "catalogue_fp": 20,
    "native_eb": 25,
    "instrumental_artifact": 60,
    "original": 65,
    "inverted": 65,
    "year_scrambled": 65,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank(value):
    return hashlib.sha256((SEED + ":" + value).encode()).hexdigest()


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def recipe(case, receipt):
    origin = receipt.get("origin", receipt.get("trial", {}))
    truth = receipt.get("truth", origin.get("truth", {}))
    metadata = origin.get("metadata", {})
    return {
        "case": case,
        "origin": origin,
        "truth": truth,
        "photometry_view": receipt.get("photometry_view", truth.get("photometry_view")),
        "phot_g_mean_mag": number(
            truth.get(
                "phot_g_mean_mag",
                origin.get("phot_g_mean_mag", metadata.get("phot_g_mean_mag")),
            )
        ),
        "depth_ppm": number(case.get("depth_ppm")),
        "period_days": number(truth.get("period_days")),
        "duration_hours": number(truth.get("duration_hours")),
        "quarter_count": number(
            origin.get("quarter_count", receipt.get("trial", {}).get("quarter_count"))
        ),
        "flat_cadences": number(origin.get("finite_psf_flat_kbonus")),
        "stored_cadences": number(origin.get("stitched_rows")),
        "instrument_model": receipt.get("instrument_model"),
    }


def stratum(item):
    case = item["case"]
    depth = item["origin"].get("depth_bin", item["truth"].get("depth_bin", "UNKNOWN"))
    return case["population"], depth, case["preparation_status"]


def sample(items, count):
    groups = defaultdict(list)
    for item in items:
        groups[stratum(item)].append(item)
    exact = {key: len(values) * count / len(items) for key, values in groups.items()}
    quota = {key: math.floor(value) for key, value in exact.items()}
    remainder = sorted(
        groups, key=lambda key: (-(exact[key] - quota[key]), rank(str(key)))
    )
    for key in remainder[: count - sum(quota.values())]:
        quota[key] += 1
    selected = [
        item
        for key, values in groups.items()
        for item in sorted(values, key=lambda item: rank(item["case"]["case_id"]))[
            : quota[key]
        ]
    ]
    return balance_marginals(items, selected)


def marginal_features(items):
    columns = []
    for key in NUMERIC:
        values = np.array(
            [np.nan if item[key] is None else item[key] for item in items]
        )
        finite = np.isfinite(values)
        columns.append(finite[:, None])
        if finite.any():
            boundaries = np.unique(np.quantile(values[finite], np.linspace(0, 1, 81)))
            columns.append(finite[:, None] & (values[:, None] <= boundaries))
    return np.column_stack(columns).astype(np.int8)


def balance_marginals(items, selected):
    """Swap within fixed strata using metadata CDFs, without accessing detector outcomes."""
    features = marginal_features(items)
    indices = {item["case"]["case_id"]: index for index, item in enumerate(items)}
    chosen = {indices[item["case"]["case_id"]] for item in selected}
    groups = defaultdict(set)
    for index, item in enumerate(items):
        groups[stratum(item)].add(index)
    groups = [
        values for values in groups.values() if values & chosen and values - chosen
    ]
    target = features.mean(axis=0)
    current = features[sorted(chosen)].mean(axis=0)
    rng = np.random.default_rng(int(rank(items[0]["case"]["kind"])[:16], 16))

    def cost(value):
        delta = value - target
        return np.square(delta).sum(axis=-1) + 100 * np.max(np.abs(delta), axis=-1) ** 2

    for _ in range(100):
        if not groups:
            break
        pairs = []
        for _ in range(100):
            group = groups[int(rng.integers(len(groups)))]
            inside, outside = sorted(group & chosen), sorted(group - chosen)
            pairs.append((rng.choice(inside), rng.choice(outside)))
        removed, added = np.array(pairs).T
        candidates = current + (features[added] - features[removed]) / len(chosen)
        index = int(np.argmin(cost(candidates)))
        if cost(candidates[index]) < cost(current) - 1e-12:
            chosen.remove(int(removed[index]))
            chosen.add(int(added[index]))
            current = candidates[index]
    return [items[index] for index in sorted(chosen)]


def distributions(population, selected):
    report = {}
    for kind in COUNTS:
        full = [item for item in population if item["case"]["kind"] == kind]
        part = [item for item in selected if item["case"]["kind"] == kind]
        entry = {"population_cases": len(full), "selected": len(part), "numeric": {}}
        for key in NUMERIC:
            a = [item[key] for item in full if item[key] is not None]
            b = [item[key] for item in part if item[key] is not None]
            entry["numeric"][key] = {
                "population_observed": len(a),
                "selected_observed": len(b),
                "ks_distance": float(ks_2samp(a, b).statistic) if a and b else None,
                "population_range": [min(a), max(a)] if a else None,
                "selected_range": [min(b), max(b)] if b else None,
            }
        for key in ("population", "preparation_status"):
            a, b = (
                Counter(item["case"][key] for item in group) for group in (full, part)
            )
            entry[key] = {"population": dict(a), "selected": dict(b)}
        report[kind] = entry
    return report


def plan(args):
    import evolve_psf_flat as FLAT

    args.out.mkdir(parents=True, exist_ok=True)
    require(not (args.out / "plan.json").exists(), "A frozen plan already exists")
    cases = API.rows(args.cohort / "cases.csv")
    manifest = API.read(args.cohort / "manifest.json")
    API.check_bindings(
        {
            str(args.cohort / "cases.csv"): manifest["cases_csv"]["sha256"],
            str(args.cohort / "clocks.csv"): manifest["clocks_csv"]["sha256"],
        }
    )
    require(
        Counter(row["kind"] for row in cases)
        == Counter({k: v * 10 for k, v in COUNTS.items()}),
        "Expected the frozen final10000-v7 mix",
    )
    population = []
    for case in cases:
        path = args.cohort / case["receipt_path"]
        API.check_bindings({str(path): case["receipt_sha256"]})
        population.append(recipe(case, API.read(path)))
    clocks = defaultdict(list)
    for clock in API.rows(args.cohort / "clocks.csv"):
        clocks[clock["case_id"]].append(clock)
    for item in population:
        clock = clocks[item["case"]["case_id"]]
        if clock:
            for key in ("period_days", "duration_hours"):
                item[key] = number(clock[0][key])
    selected = []
    for kind, count in COUNTS.items():
        selected.extend(
            sample([item for item in population if item["case"]["kind"] == kind], count)
        )
    selected.sort(key=lambda item: rank(item["case"]["case_id"]))
    audit = distributions(population, selected)
    # With 20--25 rare anchors, a 0.20 marginal tolerance is descriptive, not equivalence.
    limits = {kind: 0.08 if count >= 60 else 0.20 for kind, count in COUNTS.items()}
    failures = [
        f"{kind}:{key}"
        for kind, entry in audit.items()
        for key, value in entry["numeric"].items()
        if value["ks_distance"] is not None and value["ks_distance"] > limits[kind]
    ]
    require(
        not failures,
        "Pilot marginal distance exceeds predeclared limits: " + ",".join(failures),
    )
    queue = API.read(args.server_queue)["order"]
    require(
        len(queue) >= 5000 and len({row["source_id"] for row in queue}) == len(queue),
        "Unique server queue with at least 5000 sources required",
    )
    designs = API.read(ROOT / "dwarfs/shallow/current-designs.json")
    (incumbent,) = designs
    candidate = json.loads(json.dumps(incumbent))
    candidate.update(
        name="delivered_psf_flat_segment_intercepts_v1", engine="evolve_psf_flat"
    )
    candidate["config"].update(
        input_view="psf_flat", psf_flat_preparation=FLAT.PREPARATION
    )
    candidate["config"]["detrending"] = {"method": "quarter_median"}
    candidate["config"]["long_period_search"]["detrending"] = {
        "method": "quarter_median"
    }
    API.save(args.out / "pilot-recipes.json", selected)
    write_csv(args.out / "selected-cases.csv", [item["case"] for item in selected])
    write_csv(args.out / "real5000.csv", queue[:5000])
    API.save(
        args.out / "representativeness.json",
        {
            "strata": "Exact kind quotas; largest-remainder M/K, depth-bin, original-eligibility quotas; deterministic hash followed by within-stratum metadata-CDF balancing, including delivered flat cadence coverage",
            "numeric_ks_limits": limits,
            "outcome_blind": True,
            "distributions": audit,
            "limits": "Marginal balance does not prove joint or noise balance; rare anchors have wider tolerance; missing metadata stays explicit",
        },
    )
    API.save(args.out / "designs.json", [incumbent, candidate])
    API.save(
        args.out / "plan.json",
        {
            "status": "FROZEN_LOCAL_PSF_FLAT_DEVELOPMENT_PILOT",
            "seed": SEED,
            "cohort": str(args.cohort.resolve()),
            "counts": COUNTS,
            "actual_v7_views": dict(
                Counter(item["photometry_view"] for item in population)
            ),
            "bindings": API.bind_files(
                [
                    args.cohort / "manifest.json",
                    args.cohort / "cases.csv",
                    args.cohort / "clocks.csv",
                    args.out / "pilot-recipes.json",
                    args.out / "selected-cases.csv",
                    args.out / "representativeness.json",
                    args.out / "designs.json",
                    args.out / "real5000.csv",
                    args.server_queue,
                ]
            ),
            "real5000_population": dict(
                Counter(row["population"] for row in queue[:5000])
            ),
            "real5000_g_range": [
                queue[0]["phot_g_mean_mag"],
                queue[4999]["phot_g_mean_mag"],
            ],
            "scoring": "Frozen period/epoch/duration clocks; zero retained injection support is a miss; every failure remains in denominators",
            "real_run_gate": {
                "candidate_planet_recall_min": 0.20,
                "unexpected_errors_max": 0,
                "control_active_increase_max": 0,
                "control_proposal_increase_max": 0,
            },
            "limits": "A 1000-case development pilot does not pass the 10000-case programme gate. Flux-level injections are conditional on delivered flattening; upstream attenuation is unmeasured. Real proposals alone do not measure recall or astrophysical false-positive rate.",
        },
    )
    print(
        json.dumps(
            {
                "pilot": 1000,
                "actual_v7_views": dict(
                    Counter(item["photometry_view"] for item in population)
                ),
                "real5000_population": dict(
                    Counter(row["population"] for row in queue[:5000])
                ),
            }
        ),
        flush=True,
    )


STREAM = """import io,json,sys,tarfile
from pathlib import Path
import numpy as np
sys.path.insert(0,"/home/adam/spacelab")
from pipeline.photometry.lc2 import photometry,read_source_rows,source_row,curve_digest
items=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as archive:
 for item in sorted(items,key=lambda x:(int(x["shard"]),x["source_id"])):
  sid=item["source_id"]; shard=int(item["shard"])
  volume="discSSD" if shard%2==0 else "discPCIe"
  folder=Path("/srv/kbonus/data")/volume/"data/kbonus/lc2"/f"shard-{shard:04d}"
  columns=["source_id","time_bkjd","cadenceno","quarter","psf_flat_kbonus_flux","psf_flat_kbonus_flux_err","sap_quality","flatten_mask"]
  table=read_source_rows(folder/"cadence.parquet",int(sid),columns)
  curve=photometry({key:table[key].to_numpy() for key in columns},"psf_flat")
  metadata=source_row(folder,int(sid)); metadata.update(source_id=sid,shard=shard,view="psf_flat",curve_sha256=curve_digest(curve),reader="server cadence.parquet source row groups")
  buffer=io.BytesIO();np.savez_compressed(buffer,**curve)
  for suffix,payload in (("npz",buffer.getvalue()),("json",json.dumps(metadata).encode())):
   entry=tarfile.TarInfo(f"{sid}.{suffix}");entry.size=len(payload);archive.addfile(entry,io.BytesIO(payload))
"""


def retrieve(items, out):
    out.mkdir(exist_ok=True)
    missing = []
    for item in items:
        sid = item["source_id"]
        if (out / f"{sid}.json").exists():
            receipt = API.read(out / f"{sid}.json")
            API.check_bindings({str(out / f"{sid}.npz"): receipt["sha256"]})
        else:
            require(not (out / f"{sid}.npz").exists(), "Unbound source archive exists")
            missing.append(item)
    if not missing:
        return
    encoded = base64.b64encode(STREAM.encode()).decode()
    command = f"PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 /srv/kbonus/dwarfs/runtime/bin/python -B -c 'import base64;exec(base64.b64decode(\"{encoded}\"))'"
    process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "kbonus", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    process.stdin.write(json.dumps(missing).encode())
    process.stdin.close()
    allowed = {item["source_id"] for item in missing}
    with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
        for member in stream:
            require(
                re.fullmatch(r"[0-9]+\.(npz|json)", member.name)
                and member.name.split(".")[0] in allowed
                and member.isfile()
                and member.size < 32 * 1024**2,
                "Unexpected source stream member",
            )
            target = out / member.name
            payload = stream.extractfile(member).read()
            if target.suffix == ".json":
                value = json.loads(payload)
                value["sha256"] = API.digest(target.with_suffix(".npz"))
                API.save(target, value)
                print(json.dumps({"retrieved": value["source_id"]}), flush=True)
            else:
                with target.open("xb") as handle:
                    handle.write(payload)
    require(process.wait() == 0, "Read-only server source retrieval failed")
    require(
        all((out / f"{sid}.json").exists() for sid in allowed),
        "Source stream is incomplete",
    )


def acquire(args):
    items = API.read(args.out / "pilot-recipes.json")
    sources = {}
    for item in items:
        if item["origin"].get("native_fits_path") or item["origin"].get("raw_path"):
            continue
        origin = item["origin"]
        sources[item["case"]["source_id"]] = {
            "source_id": item["case"]["source_id"],
            "shard": origin["shard"],
        }
    retrieve(list(sources.values()), args.out / "sources")
    catalogue = {
        item["case"]["source_id"]: item["origin"]
        for item in items
        if item["origin"].get("native_fits_path") or item["origin"].get("raw_path")
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        for path in pool.map(
            lambda origin: catalogue_fits(origin, args.out), catalogue.values()
        ):
            print(json.dumps({"catalogue_fits": str(path)}), flush=True)


def catalogue_fits(origin, folder):
    expected = origin.get("indexed_fits_sha256", origin.get("raw_fits_sha256"))
    name = f"hlsp_kbonus-bkg_kepler_kepler_kic-{int(origin['kic_id']):09d}_kepler_v1.0_lc.fits"
    original = (
        Path(origin["native_fits_path"]) if origin.get("native_fits_path") else None
    )
    path = (
        original
        if original is not None and original.exists()
        else folder / "fits" / name
    )
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        url = "https://mast.stsci.edu/api/v0.1/Download/file?" + urlencode(
            {"uri": product_uri(origin["kic_id"])}
        )
        temporary = path.with_suffix(".part")
        with urlopen(url, timeout=120) as response, temporary.open("xb") as handle:
            size = 0
            while chunk := response.read(1024**2):
                size += len(chunk)
                require(
                    size <= 256 * 1024**2,
                    "Catalogue FITS exceeds the acquisition budget",
                )
                handle.write(chunk)
        require(
            API.digest(temporary) == expected, "Downloaded frozen FITS hash differs"
        )
        temporary.rename(path)
    API.check_bindings({str(path): expected})
    return path


def checked_flat(item, folder):
    case, origin = item["case"], item["origin"]
    sid = case["source_id"]
    if origin.get("raw_path") and Path(origin["raw_path"]).exists():
        path = Path(origin["raw_path"])
        API.check_bindings({str(path): origin["raw_sha256"]})
        with np.load(path, allow_pickle=False) as data:
            raw = dict(data)
        return photometry(raw, "psf_flat"), {
            "path": str(path),
            "sha256": origin["raw_sha256"],
        }
    if origin.get("native_fits_path") or origin.get("raw_path"):
        path = catalogue_fits(origin, folder)
        expected = origin.get("indexed_fits_sha256", origin.get("raw_fits_sha256"))
        return photometry(
            fits_arrays(path, sid, {"product_kic": int(origin["kic_id"])}), "psf_flat"
        ), {"path": str(path), "sha256": expected}
    path = folder / "sources" / f"{sid}.npz"
    metadata = API.read(path.with_suffix(".json"))
    API.check_bindings({str(path): metadata["sha256"]})
    expected = origin.get("raw_fits_sha256", item["truth"].get("base_raw_fits_sha256"))
    require(
        metadata["raw_fits_sha256"] == expected,
        "Server base source differs from frozen v7",
    )
    with np.load(path, allow_pickle=False) as data:
        curve = {key: data[key].copy() for key in FIELDS}
    require(
        curve_digest(curve) == metadata["curve_sha256"],
        "Retrieved flat curve digest differs",
    )
    return curve, {
        "path": str(path),
        "sha256": metadata["sha256"],
        "metadata": metadata,
    }


def injected_flat(curve, truth):
    loss = calibrate.loss(curve["time"], truth)
    modified = {key: value.copy() for key, value in curve.items()}
    scale = np.zeros(len(loss))
    for quarter in np.unique(curve["quarter"]):
        use = curve["quarter"] == quarter
        good = (
            use
            & np.isfinite(curve["flux"])
            & (curve["quality"] == 0)
            & (curve["flatten"] == 0)
            & np.isfinite(curve["error"])
            & (curve["error"] > 0)
        )
        if np.any(good):
            scale[use] = np.median(curve["flux"][good])
    modified["flux"] = curve["flux"] - scale * loss
    return modified, {"physical_loss": loss, "flux_delta": -scale * loss}


def prepare_case(item, folder):
    case = item["case"].copy()
    case.update(
        photometry_view="psf_flat", photometry_input_class="DERIVED_DELIVERED_PSF_FLAT"
    )
    if item["phot_g_mean_mag"] is not None:
        case["phot_g_mean_mag"] = str(item["phot_g_mean_mag"])
    key = rank(case["case_id"])[:24]
    path = folder / "prepared" / f"{key}.npz"
    receipt = {
        "case_id": case["case_id"],
        "source_receipt_sha256": case["receipt_sha256"],
        "photometry_view": "psf_flat",
        "flatten_mask": True,
        "first_failure": None,
    }
    arrays = {}
    try:
        curve, provenance = checked_flat(item, folder)
        receipt["provenance"] = provenance
        arrays.update({"raw_" + key: value for key, value in curve.items()})
        original = prepare_curve_records(curve, mask_flatten=True)
        arrays["original"] = original
        arrays["original_retained_mask"] = np.isin(
            curve["cadence"], original["cadence"]
        )
        kind = case["kind"]
        if kind == "synthetic_planet":
            modified, injection = injected_flat(curve, item["truth"])
            arrays.update(injection)
            records = prepare_curve_records(modified, mask_flatten=True)
            support = calibrate.loss(records["time"], item["truth"])
            case.update(
                sampled_prepared_depth_ppm=float(support.max()) * 1e6,
                prepared_nonzero_loss_cadences=int(np.count_nonzero(support)),
                prepared_records_identical=str(
                    original.tobytes() == records.tobytes()
                ).lower(),
            )
        elif kind == "instrumental_artifact":
            modified, injection, model = CONTROLS.inject_instrument(
                curve, item["origin"]
            )
            arrays.update(injection)
            receipt["instrument_model"] = model
            records = prepare_curve_records(modified, mask_flatten=True)
        elif kind == "inverted":
            records = original.copy()
            records["flux"] *= -1
        elif kind == "year_scrambled":
            records, extra, transform = calibrate.year_scramble(original)
            arrays.update(extra)
            receipt["transform"] = transform
        else:
            records = original
        arrays["records"] = records
        if kind != "year_scrambled":
            arrays["retained_mask"] = np.isin(curve["cadence"], records["cadence"])
        case.update(
            preparation_status="PREPARED",
            first_failure="",
            input_path=str(path),
            array_key="records",
            photometry_input_class="DERIVED_DELIVERED_PSF_FLAT",
        )
        receipt.update(
            status="PREPARED",
            rows=len(records),
            records_sha256=hashlib.sha256(records.tobytes()).hexdigest(),
        )
    except (InsufficientCadences, calibrate.YearScrambleIneligible) as error:
        case.update(
            preparation_status="INPUT_INELIGIBLE",
            first_failure=f"{type(error).__name__}: {error}",
            input_path="",
            input_sha256="",
            array_key="",
        )
        receipt.update(status="INPUT_INELIGIBLE", first_failure=case["first_failure"])
    np.savez_compressed(path, **arrays)
    receipt["archive_sha256"] = API.digest(path)
    if case["preparation_status"] == "PREPARED":
        case["input_sha256"] = receipt["archive_sha256"]
    API.save(path.with_suffix(".json"), receipt)
    case.update(
        receipt_path=str(path.with_suffix(".json")),
        receipt_sha256=API.digest(path.with_suffix(".json")),
    )
    return case


def prepare(args):
    require(
        not (args.out / "prepared-cases.json").exists(), "Prepared pilot already exists"
    )
    (args.out / "prepared").mkdir()
    bindings = API.bind_files(
        [
            Path(__file__),
            Path(calibrate.__file__),
            Path(CONTROLS.__file__),
            ROOT / "pipeline/deterministic/injections.py",
            ROOT / "pipeline/deterministic/run.py",
            ROOT / "pipeline/deterministic/cloud_public.py",
            ROOT / "pipeline/deterministic/catalogue_controls.py",
            ROOT / "pipeline/photometry/lc2.py",
        ]
    )
    API.save(
        args.out / "preparation-identity.json",
        {
            "bindings": bindings,
            "runtime": API.runtime_identity(),
            "view": "psf_flat",
            "mask_flatten": True,
        },
    )
    cases = []
    for item in API.read(args.out / "pilot-recipes.json"):
        cases.append(prepare_case(item, args.out))
        print(
            json.dumps(
                {"prepared": len(cases), "status": cases[-1]["preparation_status"]}
            ),
            flush=True,
        )
    API.save(args.out / "prepared-cases.json", cases)
    API.check_bindings(bindings)


def initialize(out, index):
    identity = API.read(out / "identity.json")
    design = identity["designs"][index]
    API.check_bindings(identity["source_bindings"])
    require(
        identity["runtime"] == API.runtime_identity(),
        "Worker numerical runtime differs",
    )
    API.initialize(
        out / "source/dwarfs/shallow",
        design["engine"],
        design["config"],
        array_policy="proposals",
    )


def search(task):
    job, target = task
    API.capacity(Path(target).parent, 3 * 1024**3)
    try:
        API.scan_job((job, target))
    except RuntimeError:
        # scan_records has already preserved the original exception and partial evidence.
        pass
    result = API.read(target)
    return {
        "key": job["key"],
        "status": result["status"],
        "proposals": len(result.get("proposals", [])),
    }


@contextmanager
def worker_cache(out, engine):
    require(
        Path(engine).name == engine and engine not in (".", ".."),
        "Single declared engine cache namespace required",
    )
    cache = (out / "numba-cache" / engine).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    prior = os.environ.get("NUMBA_CACHE_DIR")
    # Spawned workers must inherit this before importing Numba or either engine.
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("NUMBA_CACHE_DIR")
        else:
            os.environ["NUMBA_CACHE_DIR"] = prior


def run(args):
    out = args.out / "pilot"
    plan_record = API.read(args.out / "plan.json")
    API.check_bindings(plan_record["bindings"])
    cases = API.read(args.out / "prepared-cases.json")
    preparation = API.read(args.out / "preparation-identity.json")
    API.check_bindings(preparation["bindings"])
    require(
        preparation["runtime"] == API.runtime_identity(),
        "Preparation numerical runtime differs",
    )
    designs = API.read(args.out / "designs.json")
    jobs = API.jobs_for(cases, args.out)
    if not out.exists():
        out.mkdir()
        bindings = API.snapshot(out, designs)
        API.save(
            out / "identity.json",
            {
                "status": "RUNNING",
                "source_bindings": bindings,
                "runtime": API.runtime_identity(),
                "designs": designs,
                "case_bindings": API.bind_files(
                    [
                        args.out / "prepared-cases.json",
                        args.out / "preparation-identity.json",
                        *[Path(case["receipt_path"]) for case in cases],
                        *[
                            Path(case["receipt_path"]).with_suffix(".npz")
                            for case in cases
                        ],
                    ]
                )
                | {job["path"]: job["sha256"] for job in jobs},
                "driver_sha256": API.digest(__file__),
            },
        )
    identity = API.read(out / "identity.json")
    API.check_bindings(identity["source_bindings"] | identity["case_bindings"])
    require(
        identity["runtime"] == API.runtime_identity()
        and identity["driver_sha256"] == API.digest(__file__),
        "Frozen runtime or driver differs",
    )
    for index, design in enumerate(designs):
        destination = out / design["name"]
        destination.mkdir(exist_ok=True)
        tasks = [
            (job, destination / f"{job['key']}.json")
            for job in jobs
            if not (destination / f"{job['key']}.json").exists()
        ]
        with (
            worker_cache(out, design["engine"]),
            ProcessPoolExecutor(
                args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(out, index),
            ) as pool,
        ):
            for entry in pool.map(search, tasks, chunksize=1):
                print(json.dumps({"design": design["name"], **entry}), flush=True)
    products = [
        out / design["name"] / f"{job['key']}.json"
        for design in designs
        for job in jobs
    ]
    arrays = [
        path.with_suffix(".npz") for path in products if API.read(path).get("arrays")
    ]
    API.save(
        out / "identity.json",
        identity
        | {
            "status": "BLIND_SEARCH_COMPLETE_UNSCORED",
            "result_bindings": API.bind_files([*products, *arrays]),
        },
    )


def scored(cases, clocks, folder):
    values = []
    for case in cases:
        if case["preparation_status"] != "PREPARED":
            result = {
                "proposals": [],
                "status": "INPUT_INELIGIBLE",
                "first_failure": case["first_failure"],
            }
        else:
            result = API.read(folder / f"{API.job_key(case)}.json")
            result.setdefault("proposals", [])
        values.append(API.score_case(case, result, clocks, True))
    return values


def metrics(rows):
    planets = [row for row in rows if row["positive"] == "true"]
    controls = [row for row in rows if row["positive"] == "false"]
    return {
        "planets": len(planets),
        "recovered": sum(row["recovered"] for row in planets),
        "recall": sum(row["recovered"] for row in planets) / len(planets),
        "controls": len(controls),
        "active_controls": sum(row["proposal_count"] > 0 for row in controls),
        "control_proposals": sum(row["proposal_count"] for row in controls),
        "statuses": dict(Counter(row["detector_status"] for row in rows)),
        "by_kind": {
            kind: {
                "cases": len(group),
                "recovered": sum(row["recovered"] for row in group),
                "proposals": sum(row["proposal_count"] for row in group),
            }
            for kind in COUNTS
            for group in [[row for row in rows if row["kind"] == kind]]
        },
    }


def report(args):
    identity = API.read(args.out / "pilot/identity.json")
    require(
        identity["status"] == "BLIND_SEARCH_COMPLETE_UNSCORED",
        "Both blind searches must finish before scoring",
    )
    API.check_bindings(
        identity["source_bindings"]
        | identity["case_bindings"]
        | identity["result_bindings"]
    )
    cases = API.read(args.out / "prepared-cases.json")
    plan_record = API.read(args.out / "plan.json")
    API.check_bindings(plan_record["bindings"])
    clocks = defaultdict(list)
    for row in API.rows(Path(plan_record["cohort"]) / "clocks.csv"):
        clocks[row["case_id"]].append(row)
    designs = identity["designs"]
    scores = [
        scored(cases, clocks, args.out / "pilot" / design["name"]) for design in designs
    ]
    summaries = [metrics(values) for values in scores]
    baseline, candidate = summaries
    allowed = (
        candidate["recall"] >= 0.20
        and candidate["active_controls"] <= baseline["active_controls"]
        and candidate["control_proposals"] <= baseline["control_proposals"]
        and all(
            summary["statuses"].get("UNEXPECTED_ERROR", 0) == 0 for summary in summaries
        )
    )
    for design, values in zip(designs, scores):
        API.save(args.out / (design["name"] + "-scored.json"), values)
    API.save(
        args.out / "retrieval-report.json",
        {
            "status": "SCORED_DEVELOPMENT_PILOT",
            "summaries": dict(zip((design["name"] for design in designs), summaries)),
            "paired": API.paired_statistics(*scores),
            "strata": {
                design["name"]: API.fractions(values)
                for design, values in zip(designs, scores)
            },
            "real5000_gate_passed": allowed,
            "programme_10000_gate_passed": False,
            "limits": plan_record["limits"],
        },
    )
    print(
        json.dumps({"summaries": summaries, "real5000_gate_passed": allowed}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("plan", "acquire", "prepare", "run", "report")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--server-queue", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.out = args.out.resolve()
    require(args.workers >= 1, "Positive worker count required")
    API.capacity(args.out.parent, 16 * 1024**3)
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
