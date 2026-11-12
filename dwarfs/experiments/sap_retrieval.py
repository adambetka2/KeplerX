"""Search only SAP on the frozen Flat cohort, then join the saved Flat outcomes."""

import argparse
import ast
import base64
import hashlib
import inspect
import json
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_retrieval as pilot

from dwarfs.harness.survey import PREPARATION, quarter_crowding, source_curve
from pipeline.photometry.lc2 import read_source_rows, source_row

API = pilot.API
require = pilot.require
FIELDS = pilot.FIELDS


def probe(curve, crowding):
    try:
        corrected, _ = source_curve(curve, crowding)
        records = pilot.prepare_curve_records(corrected)
    except (ValueError, pilot.InsufficientCadences) as error:
        return {
            "status": "INPUT_INELIGIBLE",
            "first_failure": f"{type(error).__name__}: {error}",
        }
    return {
        "status": "PREPARED",
        "rows": len(records),
        "records_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
    }


def native_source(root, item):
    sid, shard = item["source_id"], int(item["shard"])
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    folder = root / volume / f"data/kbonus/lc2/shard-{shard:04d}"
    columns = [
        "source_id",
        "time_bkjd",
        "cadenceno",
        "quarter",
        "sap_flux",
        "sap_flux_err",
        "sap_quality",
    ]
    table = read_source_rows(folder / "qdiag.parquet", int(sid), columns)
    raw = {
        "qdiag_" + name: table[name].to_numpy()
        for name in columns
        if name != "source_id"
    }
    table = read_source_rows(
        folder / "cadence.parquet", int(sid), ["source_id", "cadenceno", "flatten_mask"]
    )
    raw.update({name: table[name].to_numpy() for name in ("cadenceno", "flatten_mask")})
    curve = pilot.photometry(raw, "sap_qdiag")
    crowding = quarter_crowding(folder, int(sid))
    metadata = {
        "source_id": sid,
        "shard": shard,
        "raw_fits_sha256": source_row(folder, int(sid))["raw_fits_sha256"],
        "curve_sha256": pilot.curve_digest(curve),
        "flatten_sha256": hashlib.sha256(curve["flatten"].tobytes()).hexdigest(),
        "crowding": crowding,
        "unmodified_preparation": probe(curve, crowding),
        "reader": str(folder),
    }
    return curve, metadata


def store_source(out, curve, metadata):
    path = out / f"{metadata['source_id']}.npz"
    require(not path.exists(), "Source archive already exists")
    np.savez_compressed(path, **curve)
    API.save(path.with_suffix(".json"), metadata | {"sha256": API.digest(path)})


def server_stream(items, out):
    imports = """import io,json,hashlib,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/home/adam/spacelab')
import numpy as np
from types import SimpleNamespace
from pipeline.photometry.lc2 import photometry,curve_digest,read_source_rows,source_row
from pipeline.deterministic.run import prepare_curve_records,InsufficientCadences
from dwarfs.harness.survey import quarter_crowding,source_curve
pilot=SimpleNamespace(photometry=photometry,curve_digest=curve_digest,prepare_curve_records=prepare_curve_records,InsufficientCadences=InsufficientCadences)
"""
    stream = """
items=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for item in items:
  curve,meta=native_source(Path('/srv/kbonus/data'),item)
  buffer=io.BytesIO();np.savez_compressed(buffer,**curve)
  for suffix,payload in (('npz',buffer.getvalue()),('json',json.dumps(meta).encode())):
   entry=tarfile.TarInfo(f"{item['source_id']}.{suffix}");entry.size=len(payload)
   archive.addfile(entry,io.BytesIO(payload))
"""
    server = API.read(out.parent.parent / "server-identity.json")
    checks = {
        path: digest
        for path, digest in server["source_bindings"].items()
        if path.startswith("/home/adam/spacelab/")
    }
    verify = f"""
for name,expected in {checks!r}.items():
 with Path(name).open('rb') as handle:
  if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:
   raise ValueError('Server preparation source differs: '+name)
"""
    code = (
        imports
        + verify
        + inspect.getsource(probe)
        + inspect.getsource(native_source)
        + stream
    )
    encoded = base64.b64encode(code.encode()).decode()
    command = f"PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 /srv/kbonus/dwarfs/runtime/bin/python -B -c 'import base64;exec(base64.b64decode(\"{encoded}\"))'"
    process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "kbonus", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    process.stdin.write(json.dumps(items).encode())
    process.stdin.close()
    pending = {item["source_id"] for item in items}
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            sid, suffix = member.name.split(".")
            require(
                sid in pending
                and suffix in ("npz", "json")
                and member.isfile()
                and member.size < 32 * 1024**2,
                "Unexpected SAP stream member",
            )
            payload = archive.extractfile(member).read()
            path = out / member.name
            if suffix == "npz":
                with path.open("xb") as handle:
                    handle.write(payload)
            else:
                metadata = json.loads(payload)
                API.save(
                    path, metadata | {"sha256": API.digest(path.with_suffix(".npz"))}
                )
                pending.remove(sid)
                print(json.dumps({"acquired_server": sid}), flush=True)
    require(
        process.wait() == 0 and not pending, "Read-only SAP source retrieval failed"
    )


def verified_survey(path, expected):
    original = subprocess.check_output(
        [
            "git",
            "show",
            "f7e78fd5a6fc4508a744f189a55f1b5021122e04:dwarfs/harness/survey.py",
        ],
        cwd=pilot.ROOT,
    )
    require(
        hashlib.sha256(original).hexdigest() == expected,
        "Server preparation snapshot differs",
    )
    # The Mac added evolution.py to provenance, without changing preparation.
    trees = [ast.parse(text) for text in (original, path.read_bytes())]
    for tree in trees:
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "PREPARATION_FILES"
                    for target in node.targets
                )
            )
        ]
    require(
        ast.dump(trees[0]) == ast.dump(trees[1]),
        "Server SAP preparation implementation differs",
    )
    return API.digest(path)


def source_equivalence(flat, server):
    bindings = {}
    for old, digest in flat["source_bindings"].items():
        if "/source/" in old:
            bindings[str(pilot.ROOT / old.split("/source/", 1)[1])] = digest
    for old, digest in server["source_bindings"].items():
        if "/home/adam/spacelab/" in old:
            name = old.split("/home/adam/spacelab/", 1)[1]
            path = pilot.ROOT / name
            if name == "dwarfs/harness/survey.py":
                digest = verified_survey(path, digest)
            bindings[str(path)] = digest
    API.check_bindings(bindings)
    return bindings


def plan(args):
    require(not args.out.exists(), "Use a new SAP comparison folder")
    base = args.flat.parent
    flat = API.read(args.flat / "pilot/identity.json")
    require(
        flat["status"] == "BLIND_SEARCH_COMPLETE_UNSCORED",
        "Completed Flat search required",
    )
    API.check_bindings(
        flat["source_bindings"] | flat["case_bindings"] | flat["result_bindings"]
    )
    server = API.read(base / "server-identity.json")
    design = flat["designs"][0]
    require(
        design == server["design"], "Flat baseline must be the adopted server detector"
    )
    require(
        flat["runtime"] == API.runtime_identity(),
        "Saved Flat numerical runtime differs",
    )
    bindings = source_equivalence(flat, server)
    recipes = API.read(base / "usable1000-v1/pilot-recipes.json")
    cases = API.read(args.flat / "prepared-cases.json")
    require(
        [item["case"]["case_id"] for item in recipes]
        == [case["case_id"] for case in cases],
        "Exact ordered Flat cohort required",
    )
    old_plan = API.read(args.flat / "plan.json")
    API.check_bindings(old_plan["bindings"])
    args.out.mkdir()
    API.save(args.out / "designs.json", [design])
    API.save(args.out / "pilot-recipes.json", recipes)
    paths = [
        base / "usable1000-v1/pilot-recipes.json",
        args.flat / "prepared-cases.json",
        args.flat / "pilot/identity.json",
        args.flat / "retrieval-report.json",
        base / "server-identity.json",
        Path(old_plan["cohort"]) / "clocks.csv",
    ]
    API.save(
        args.out / "plan.json",
        {
            "status": "FROZEN_SAP_ONLY_SAME_COHORT",
            "flat": str(args.flat),
            "cohort": old_plan["cohort"],
            "cases": len(cases),
            "flat_design": design["name"],
            "preparation": PREPARATION,
            "bindings": bindings
            | API.bind_files(
                [*paths, args.out / "designs.json", args.out / "pilot-recipes.json"]
            ),
            "limits": [
                "Cohort conditions on usable Flat; SAP eligibility failures remain in denominators",
                "Same detector, parameters, physical injections and scorer clocks; extraction-specific masks and available cadences",
                "Synthetic injections are conditional on delivered products; upstream attenuation is unmeasured",
                "Real confirmed signals measure actual retained signal; controls measure proposal burden, not astrophysical false positives",
                "Exploratory 1000-case comparison does not pass the programme 10000-case gate",
            ],
        },
    )


def acquire(args):
    API.check_bindings(API.read(args.out / "plan.json")["bindings"])
    out = args.out / "sources"
    out.mkdir(exist_ok=True)
    native = {}
    for item in API.read(args.out / "pilot-recipes.json"):
        if not (
            item["origin"].get("native_fits_path") or item["origin"].get("raw_path")
        ):
            native[item["case"]["source_id"]] = {
                "source_id": item["case"]["source_id"],
                "shard": item["origin"]["shard"],
            }
    missing = []
    for sid, item in sorted(
        native.items(), key=lambda entry: (int(entry[1]["shard"]), entry[0])
    ):
        receipt = out / f"{sid}.json"
        if receipt.exists():
            API.check_bindings(
                {str(receipt.with_suffix(".npz")): API.read(receipt)["sha256"]}
            )
            continue
        volume = "discSSD" if int(item["shard"]) % 2 == 0 else "discPCIe"
        local = (
            pilot.ROOT
            / volume
            / f"data/kbonus/lc2/shard-{int(item['shard']):04d}/qdiag.parquet"
        )
        if local.exists():
            store_source(out, *native_source(pilot.ROOT, item))
            print(json.dumps({"acquired_local": sid}), flush=True)
        else:
            missing.append(item)
    if missing:
        server_stream(missing, out)


def checked_sap(item, folder):
    origin, sid = item["origin"], item["case"]["source_id"]
    if origin.get("native_fits_path") or origin.get("raw_path"):
        path = pilot.catalogue_fits(origin, folder.parent)
        raw = pilot.fits_arrays(path, sid, {"product_kic": int(origin["kic_id"])})
        with fits.open(path, memmap=False) as hdus:
            crowding = {
                int(hdu.header["QUARTER"]): float(hdu.header["CROWDSAP"])
                for hdu in hdus
                if hdu.name.startswith("LIGHTCURVE_Q") and "CROWDSAP" in hdu.header
            }
        curve = pilot.photometry(raw, "sap_qdiag")
        return (
            curve,
            crowding,
            {
                "path": str(path),
                "sha256": API.digest(path),
                "unmodified_preparation": probe(curve, crowding),
            },
        )
    path = folder / "sources" / f"{sid}.npz"
    metadata = API.read(path.with_suffix(".json"))
    API.check_bindings({str(path): metadata["sha256"]})
    expected = origin.get("raw_fits_sha256", item["truth"].get("base_raw_fits_sha256"))
    require(
        metadata["raw_fits_sha256"] == expected,
        "SAP source differs from frozen Flat base",
    )
    with np.load(path, allow_pickle=False) as stored:
        curve = {key: stored[key].copy() for key in FIELDS}
    require(
        pilot.curve_digest(curve) == metadata["curve_sha256"]
        and hashlib.sha256(curve["flatten"].tobytes()).hexdigest()
        == metadata["flatten_sha256"],
        "SAP curve or flatten mask differs",
    )
    return (
        curve,
        {int(key): value for key, value in metadata["crowding"].items()},
        {"path": str(path), "sha256": metadata["sha256"], **metadata},
    )


def inject(curve, truth):
    loss = pilot.calibrate.loss(curve["time"], truth)
    scale = np.zeros(len(loss))
    good = (
        np.isfinite(curve["time"])
        & np.isfinite(curve["flux"])
        & np.isfinite(curve["error"])
        & (curve["error"] > 0)
        & (curve["quality"] == 0)
    )
    for quarter in np.unique(curve["quarter"]):
        use = curve["quarter"] == quarter
        if np.any(use & good):
            scale[use] = np.median(curve["flux"][use & good])
    delta = -scale * loss
    return curve | {"flux": curve["flux"] + delta}, {
        "physical_loss": loss,
        "flux_delta": delta,
    }


def transform(case, item, curve, original):
    kind = case["kind"]
    if kind == "synthetic_planet":
        modified, arrays = inject(curve, item["truth"])
        records = pilot.prepare_curve_records(modified)
        support = pilot.calibrate.loss(records["time"], item["truth"])
        case.update(
            sampled_prepared_depth_ppm=float(support.max()) * 1e6,
            prepared_nonzero_loss_cadences=int(np.count_nonzero(support)),
            prepared_records_identical=str(
                original.tobytes() == records.tobytes()
            ).lower(),
        )
        return records, arrays, {"truth": item["truth"]}
    if kind == "instrumental_artifact":
        modified, arrays, model = pilot.CONTROLS.inject_instrument(
            curve, item["origin"]
        )
        return (
            pilot.prepare_curve_records(modified),
            arrays,
            {"instrument_model": model},
        )
    if kind == "inverted":
        records = original.copy()
        records["flux"] *= -1
        return records, {}, {"transform": "inverted prepared flux"}
    if kind == "year_scrambled":
        records, arrays, model = pilot.calibrate.year_scramble(original)
        return records, arrays, {"transform": model}
    return original, {}, {}


def prepare_case(item, folder):
    case = item["case"].copy()
    case.update(
        photometry_view="sap_qdiag",
        photometry_input_class="NATIVE_SAP_CONSTANT_NEIGHBOUR_SOURCE_FLUX_V1",
        phot_g_mean_mag=item["phot_g_mean_mag"],
    )
    path = (
        folder
        / "prepared"
        / (hashlib.sha256(case["case_id"].encode()).hexdigest()[:24] + ".npz")
    )
    require(not path.exists(), "Prepared SAP case already exists")
    receipt = {
        "case_id": case["case_id"],
        "source_receipt_sha256": case["receipt_sha256"],
        "flatten_mask": False,
        "first_failure": None,
    }
    delivered, crowding, provenance = checked_sap(item, folder)
    arrays = {"raw_" + key: value for key, value in delivered.items()}
    receipt["provenance"] = provenance
    try:
        curve, detail = source_curve(delivered, crowding)
        receipt.update(detail)
        arrays["source_flux"] = curve["flux"]
        original = pilot.prepare_curve_records(curve)
        if hashlib.sha256(original.tobytes()).hexdigest() != provenance[
            "unmodified_preparation"
        ].get("records_sha256"):
            raise RuntimeError(
                "Unmodified SAP preparation differs from acquisition probe"
            )
        arrays["original"] = original
        arrays["original_retained_mask"] = np.isin(
            curve["cadence"], original["cadence"]
        )
        records, extra, detail = transform(case, item, curve, original)
        arrays.update(extra, records=records)
        receipt.update(
            detail,
            rows=len(records),
            records_sha256=hashlib.sha256(records.tobytes()).hexdigest(),
            status="PREPARED",
        )
        if case["kind"] != "year_scrambled":
            arrays["retained_mask"] = np.isin(curve["cadence"], records["cadence"])
        case.update(
            preparation_status="PREPARED",
            first_failure="",
            input_path=str(path),
            array_key="records",
        )
    except (
        ValueError,
        pilot.InsufficientCadences,
        pilot.calibrate.YearScrambleIneligible,
    ) as error:
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
        not (args.out / "prepared-cases.json").exists(),
        "Prepared cohort already exists",
    )
    API.check_bindings(API.read(args.out / "plan.json")["bindings"])
    (args.out / "prepared").mkdir()
    bindings = API.bind_files(
        [
            Path(__file__),
            Path(pilot.__file__),
            Path(pilot.calibrate.__file__),
            Path(pilot.CONTROLS.__file__),
            pilot.ROOT / "dwarfs/harness/survey.py",
            pilot.ROOT / "pipeline/deterministic/injections.py",
            pilot.ROOT / "pipeline/deterministic/run.py",
            pilot.ROOT / "pipeline/deterministic/cloud_public.py",
            pilot.ROOT / "pipeline/photometry/lc2.py",
        ]
    )
    API.save(
        args.out / "preparation-identity.json",
        {
            "bindings": bindings,
            "runtime": API.runtime_identity(),
            "view": "sap_qdiag",
            "mask_flatten": False,
            "text": PREPARATION,
        },
    )
    cases = []
    for item in API.read(args.out / "pilot-recipes.json"):
        cases.append(prepare_case(item, args.out))
        print(
            json.dumps(
                {
                    "prepared": len(cases),
                    "status": cases[-1]["preparation_status"],
                    "first_failure": cases[-1]["first_failure"],
                }
            ),
            flush=True,
        )
    API.check_bindings(bindings)
    API.save(args.out / "prepared-cases.json", cases)
    API.save(
        args.out / "preparation-summary.json",
        {
            "cases": len(cases),
            "jobs": len(API.jobs_for(cases, args.out)),
            "statuses": dict(Counter(case["preparation_status"] for case in cases)),
            "failures": dict(
                Counter(
                    case["first_failure"] for case in cases if case["first_failure"]
                )
            ),
        },
    )


def overlap(sap, flat):
    require(
        [row["case_id"] for row in sap] == [row["case_id"] for row in flat],
        "Exact paired IDs and order required",
    )
    groups = {key: [] for key in ("both", "sap_only", "flat_only", "neither")}
    for a, b in zip(sap, flat, strict=True):
        if a["positive"] == "true":
            key = (
                ("both" if b["recovered"] else "sap_only")
                if a["recovered"]
                else ("flat_only" if b["recovered"] else "neither")
            )
            groups[key].append(a["case_id"])
    return {
        "counts": {key: len(value) for key, value in groups.items()},
        "case_ids": groups,
    }


def report(args):
    plan_record = API.read(args.out / "plan.json")
    API.check_bindings(plan_record["bindings"])
    identity = API.read(args.out / "pilot/identity.json")
    require(
        identity["status"] == "BLIND_SEARCH_COMPLETE_UNSCORED"
        and len(identity["designs"]) == 1,
        "The SAP-only search must finish before scoring",
    )
    flat_folder = Path(plan_record["flat"])
    old = API.read(flat_folder / "pilot/identity.json")
    require(
        identity["designs"][0] == old["designs"][0]
        and identity["runtime"] == old["runtime"],
        "Paired detector and runtime identities differ",
    )
    for entry in (identity, old):
        API.check_bindings(
            entry["source_bindings"] | entry["case_bindings"] | entry["result_bindings"]
        )
    clocks = defaultdict(list)
    for row in API.rows(Path(plan_record["cohort"]) / "clocks.csv"):
        clocks[row["case_id"]].append(row)
    design = plan_record["flat_design"]
    sap = pilot.scored(
        API.read(args.out / "prepared-cases.json"), clocks, args.out / "pilot" / design
    )
    flat = pilot.scored(
        API.read(flat_folder / "prepared-cases.json"),
        clocks,
        flat_folder / "pilot" / design,
    )
    saved = API.read(flat_folder / "retrieval-report.json")["summaries"][design]
    require(
        pilot.metrics(flat) == saved, "Reused Flat scores differ from the saved report"
    )
    API.save(args.out / "sap-scored.json", sap)
    API.save(args.out / "flat-reused-scored.json", flat)
    common = [
        (a, b)
        for a, b in zip(sap, flat, strict=True)
        if a["preparation_status"] == b["preparation_status"] == "PREPARED"
    ]
    result = {
        "status": "SCORED_SAP_VERSUS_SAVED_FLAT",
        "flat_searches_started": 0,
        "summaries": {"sap": pilot.metrics(sap), "flat_reused": pilot.metrics(flat)},
        "overlap": overlap(sap, flat),
        "overlap_by_kind": {
            kind: overlap(
                [a for a in sap if a["kind"] == kind],
                [b for b in flat if b["kind"] == kind],
            )
            for kind in ("synthetic_planet", "real_planet")
        },
        "both_prepared": {
            "cases": len(common),
            "overlap": overlap([a for a, _ in common], [b for _, b in common]),
            "summaries": {
                "sap": pilot.metrics([a for a, _ in common]),
                "flat_reused": pilot.metrics([b for _, b in common]),
            },
        },
        "errors": {
            label: [
                {
                    "case_id": row["case_id"],
                    "status": row["detector_status"],
                    "first_failure": row["detector_first_failure"],
                }
                for row in rows
                if row["detector_status"] not in ("SEARCHED", "DETECTOR_INELIGIBLE")
            ]
            for label, rows in (("sap", sap), ("flat_inherited", flat))
        },
        "strata": {"sap": API.fractions(sap), "flat_reused": API.fractions(flat)},
        "limits": plan_record["limits"],
    }
    API.save(args.out / "retrieval-report.json", result)
    print(
        json.dumps(
            {"summaries": result["summaries"], "overlap": result["overlap"]["counts"]}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("plan", "acquire", "prepare", "run", "report")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flat", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.flat is not None:
        args.flat = args.flat.resolve()
    API.capacity(args.out.parent, 16 * 1024**3)
    require(args.workers > 0, "Positive worker count required")
    if args.stage == "run":
        pilot.run(args)
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    main()
