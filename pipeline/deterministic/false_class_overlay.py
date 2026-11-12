"""Finish a frozen false-class cohort using a read-only partial materialization."""

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DUMP = (ROOT / "dump").resolve()
DOWNLOAD_WORKERS = 2


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local_output(path):
    path = Path(path).resolve()
    if LOCAL_DUMP not in path.parents:
        raise ValueError("Overlay output must be inside the local repository dump")
    return path


def require_volumes():
    for name in ("discSSD", "discPCIe"):
        volume = (ROOT / name).resolve()
        if not volume.is_mount() or not (volume / "data/index.json").is_file():
            raise FileNotFoundError(
                f"Required Kepler Bonus volume is not mounted: {volume}"
            )


def seed_cache(cache, source_cache):
    if source_cache.resolve() == cache.resolve():
        return 0
    linked = 0
    for source in source_cache.glob("kplr*_lpd-targ.fits.gz"):
        target = cache / source.name
        if target.exists():
            if target.is_symlink() and target.resolve() != source.resolve():
                raise ValueError(f"Target-pixel cache link changed: {target}")
            continue
        if target.is_symlink():
            raise ValueError(f"Broken target-pixel cache link: {target}")
        target.symlink_to(source.resolve())
        linked += 1
    return linked


def configure_cache(out):
    config = read(ROOT / "pipeline/vetting/config.json")
    config.update(
        pixel_cache_directory=str(out / "pixel-cache"),
        run_directory=str(out / "vetting"),
    )
    path = out / "vetting-config.json"
    if path.exists() and read(path) != config:
        raise ValueError("Overlay pixel-cache configuration changed")
    if not path.exists():
        write(path, config)
    cache = out / "pixel-cache"
    cache.mkdir(exist_ok=True)
    seed_cache(cache, ROOT / "dump/pixel-cache")
    os.environ["SPACELAB_VETTING_CONFIG"] = str(path)
    return path


def modules(out=None):
    sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]
    if out is not None:
        configure_cache(out)
    controls = importlib.import_module("deterministic.false_class_injections")
    pixels = importlib.import_module("deterministic.pixel_injections")
    if out is not None:
        pixels.TPF_DOWNLOAD_WORKERS = DOWNLOAD_WORKERS
        cache = importlib.import_module("vetting_pixels").PIXELS.resolve()
        if cache != (out / "pixel-cache").resolve():
            raise ValueError("False-class pixel cache is not local to the overlay")
    return controls, pixels


def verified_products(prior, pixels, row, record, check_headers=True):
    sid = row["source_id"]
    truth, item = record["truth"], record["manifest"]
    if (
        truth["source_id"] != sid
        or truth["category"] != row["category"]
        or str(truth["parent_tpf"]) != row["parent_tpf"]
        or truth["raw_fits_sha256"] != row["raw_fits_sha256"]
        or item["metadata"]["source_id"] != sid
        or item["base_raw_fits_sha256"] != row["raw_fits_sha256"]
        or any(
            str(pixel["base_parent_kic"]) != row["parent_tpf"]
            for pixel in item["pixels"].values()
        )
    ):
        raise ValueError(f"Stored false-class trial identity changed: {sid}")
    if check_headers:
        valid = pixels.validate_record(
            {"source_id": sid, "pixels": item["pixels"]},
            prior,
            sid,
            truth["injected_quarters"],
        )
    else:
        valid = set(item["pixels"]) == set(map(str, truth["injected_quarters"]))
        valid &= all(
            prior.resolve() in (prior / pixel["path"]).resolve().parents
            and digest(prior / pixel["path"]) == pixel["sha256"]
            for pixel in item["pixels"].values()
        )
    if not valid:
        raise ValueError(f"Stored false-class pixels changed: {sid}")
    array = prior / item["arrays"]
    if digest(array) != item["sha256"]:
        raise ValueError(f"Stored false-class arrays changed: {sid}")
    return array.stat().st_size + sum(
        (prior / pixel["path"]).stat().st_size for pixel in item["pixels"].values()
    )


def stored_rows(folder):
    records = {path.stem: read(path) for path in (folder / "records").glob("*.json")}
    unavailable = {
        path.stem: read(path)
        for path in (folder / "unavailable-records").glob("*.json")
    }
    if set(records) & set(unavailable):
        raise ValueError(
            "A false-class source has both a record and an unavailable result"
        )
    return records, unavailable


def verify_skip(row, skipped):
    sid = row["source_id"]
    if skipped["source_id"] != sid or skipped["category"] != row["category"]:
        raise ValueError(f"Unavailable false-class identity changed: {sid}")
    if "support_evidence" not in skipped:
        return False
    path = ROOT / skipped["support_evidence"]
    if not path.is_file() or digest(path) != skipped["support_evidence_sha256"]:
        raise ValueError(f"Unavailable false-class support changed: {sid}")
    evidence = read(path)
    if (
        evidence["source_id"] != sid
        or evidence["category"] != row["category"]
        or str(evidence["frozen_parent_tpf"]) != row["parent_tpf"]
        or evidence["raw_fits_sha256"] != row["raw_fits_sha256"]
        or evidence["rule_sha256"] != skipped["support_rule_sha256"]
        or any(
            evidence.get(key) != row.get(key)
            for key in ("contaminant_source_id", "contaminant_raw_fits_sha256")
            if key in evidence
        )
    ):
        raise ValueError(f"Unavailable false-class support identity changed: {sid}")
    reasons = {
        "Frozen parent TPF does not contain target and contaminant in quarters ": "off_stamp_quarters",
        "Frozen parent TPF lacks 2-pixel source support in quarters ": "unsupported_quarters",
        "Frozen parent TPF unavailable in quarters ": "unavailable_parent_quarters",
    }
    matching = [
        (prefix, key)
        for prefix, key in reasons.items()
        if skipped["reason"].startswith(prefix)
    ]
    if len(matching) != 1:
        raise ValueError(f"Unavailable false-class support reason changed: {sid}")
    prefix, key = matching[0]
    if json.loads(skipped["reason"][len(prefix) :]) != evidence[key]:
        raise ValueError(f"Unavailable false-class support reason changed: {sid}")
    rule = skipped.get("support_rule_path")
    candidates = [ROOT / rule] if rule else path.parent.glob("*rule*.json")
    if not any(
        candidate.is_file() and digest(candidate) == evidence["rule_sha256"]
        for candidate in candidates
    ):
        raise ValueError(f"Unavailable false-class support rule changed: {sid}")
    return True


def prior_prefix(rows, plan, seed, records, unavailable):
    counts, seen_records, seen_unavailable, next_candidates = {}, set(), set(), {}
    quota = plan["development_per_class"] + plan["evaluation_per_class"]
    for category in plan["classes"]:
        accepted, first_missing = [], None
        for row in (row for row in rows if row["category"] == category):
            if len(accepted) == quota:
                break
            sid = row["source_id"]
            if sid in unavailable:
                seen_unavailable.add(sid)
                continue
            if sid not in records:
                first_missing = sid
                break
            checked = copy.deepcopy(records[sid])
            checked["truth"]["partition"] = (
                "development"
                if len(accepted) < plan["development_per_class"]
                else "evaluation"
            )
            if len(accepted) < plan["development_per_class"] and (
                checked["truth"] != seed["accepted"][sid]
                or checked["manifest"] != seed["manifest"][sid]
            ):
                raise ValueError(f"Prior development seed changed: {sid}")
            accepted.append(sid)
            seen_records.add(sid)
        counts[category] = len(accepted)
        next_candidates[category] = first_missing
    if seen_records != set(records):
        raise ValueError("Prior false-class records are not a candidate-pool prefix")
    unvisited = set(unavailable) - seen_unavailable
    if any("support_evidence" not in unavailable[sid] for sid in unvisited):
        raise ValueError(
            "Prior false-class skips beyond the prefix lack frozen support"
        )
    return counts, next_candidates


def audit_prior(prior, controls, pixels, certificate=None):
    plan, pool = read(controls.PLAN), controls.POOL
    rows = controls.pool_rows(pool, plan, None)
    progress_sha256 = digest(prior / "progress.json")
    seed_path = Path(
        read(prior / "progress.json")["identity"]["development_seed"]["path"]
    )
    seed = controls.development_seed(seed_path.parent, pool, rows, plan)
    progress = read(prior / "progress.json")
    expected = {
        "pool_sha256": digest(pool / "freeze.json"),
        "plan_sha256": digest(controls.PLAN),
        "generator": controls.generator_identity(),
        "category": None,
        "limit": None,
        "development_seed": seed["binding"],
    }
    if progress["status"] != "RUNNING" or progress["identity"] != expected:
        raise ValueError(
            "Read-only prior false-class progress is not this frozen cohort"
        )
    records, unavailable = stored_rows(prior)
    if progress["complete"] != len(records):
        raise ValueError("Prior false-class progress count differs from stored records")
    by_id = {row["source_id"]: row for row in rows}
    if (set(records) | set(unavailable)) - set(by_id):
        raise ValueError("Prior false-class result is outside the candidate pool")
    record_digests_sha256 = hashlib.sha256(
        json.dumps(
            {sid: digest(prior / "records" / f"{sid}.json") for sid in sorted(records)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    reuse_header_audit = certificate is not None and all(
        certificate.get(key) == value
        for key, value in {
            "status": "VERIFIED_READ_ONLY_PREFIX",
            "prior": str(prior),
            "prior_progress_sha256": progress_sha256,
            "candidate_pool_sha256": expected["pool_sha256"],
            "generator": expected["generator"],
            "record_digests_sha256": record_digests_sha256,
        }.items()
    )
    bytes_verified = 0
    for sid, record in records.items():
        bytes_verified += verified_products(
            prior, pixels, by_id[sid], record, check_headers=not reuse_header_audit
        )
    for sid, skipped in unavailable.items():
        verify_skip(by_id[sid], skipped)
    counts, next_candidates = prior_prefix(rows, plan, seed, records, unavailable)
    if digest(prior / "progress.json") != progress_sha256:
        raise ValueError("Read-only prior false-class progress changed during audit")
    report = {
        "status": "VERIFIED_READ_ONLY_PREFIX",
        "prior": str(prior),
        "prior_progress_sha256": progress_sha256,
        "candidate_pool_sha256": expected["pool_sha256"],
        "generator": expected["generator"],
        "accepted": counts,
        "unavailable": dict(Counter(row["category"] for row in unavailable.values())),
        "next_candidates": next_candidates,
        "verified_input_bytes": bytes_verified,
        "record_digests_sha256": record_digests_sha256,
        "header_check": "BYTE_IDENTICAL_TO_PRIOR_FULL_AUDIT"
        if reuse_header_audit
        else "ALL_PIXEL_HEADERS_CHECKED",
    }
    return report, rows, plan, seed, records, unavailable


def rebase(record, folder):
    record = copy.deepcopy(record)
    item = record["manifest"]
    item["arrays"] = str((folder / item["arrays"]).resolve())
    for pixel in item["pixels"].values():
        pixel["path"] = str((folder / pixel["path"]).resolve())
    return record


def unselected_support_count(
    prior_records, local_records, prior_skips, local_skips, used_records, used_skips
):
    if used_records != set(prior_records) | set(local_records):
        raise ValueError("False-class overlay has unused accepted records")
    unused_skips = (set(prior_skips) | set(local_skips)) - used_skips
    if unused_skips - set(prior_skips) or any(
        "support_evidence" not in prior_skips[sid] for sid in unused_skips
    ):
        raise ValueError("False-class overlay has unused generated outcomes")
    return len(unused_skips)


def freeze(
    out, prior, controls, pixels, report, rows, plan, seed, prior_records, prior_skips
):
    local_records, local_skips = stored_rows(out)
    if (set(local_records) | set(local_skips)) & (
        set(prior_records) | set(prior_skips)
    ):
        raise ValueError("Overlay repeats a prior false-class outcome")
    by_id = {row["source_id"]: row for row in rows}
    if (set(local_records) | set(local_skips)) - set(by_id):
        raise ValueError("Overlay result is outside the candidate pool")
    for sid, record in local_records.items():
        verified_products(out, pixels, by_id[sid], record)
    for sid, skipped in local_skips.items():
        verify_skip(by_id[sid], skipped)
    identity = {
        "prior_progress_sha256": report["prior_progress_sha256"],
        "prior_record_digests_sha256": report["record_digests_sha256"],
        "generator": controls.generator_identity(),
        "overlay_sha256": digest(__file__),
        "pixel_cache_config_sha256": digest(out / "vetting-config.json"),
        "download_workers": DOWNLOAD_WORKERS,
    }
    progress_path = out / "progress.json"
    if progress_path.exists():
        progress = read(progress_path)
        if progress["status"] != "RUNNING" or progress["identity"] != identity:
            raise ValueError("False-class overlay resume identity changed")
        progress["complete"] = len(local_records)
        write(progress_path, progress)
    else:
        progress = {"status": "RUNNING", "identity": identity, "complete": 0}
        write(progress_path, progress)
    mapping, existing = controls.pixel_injections.indexed_pixels()
    accepted, unavailable = [], []
    used_records, used_skips = set(), set()
    quota = plan["development_per_class"] + plan["evaluation_per_class"]
    for category in plan["classes"]:
        complete = 0
        for row in (row for row in rows if row["category"] == category):
            if complete == quota:
                break
            sid = row["source_id"]
            if sid in prior_skips or sid in local_skips:
                skipped = prior_skips.get(sid, local_skips.get(sid))
                unavailable.append(skipped)
                used_skips.add(sid)
                continue
            if sid in prior_records or sid in local_records:
                folder = prior if sid in prior_records else out
                record = prior_records.get(sid, local_records.get(sid))
            else:
                folder = out
                try:
                    record = controls.materialize_trial(
                        out, row, plan, mapping, existing
                    )
                except (
                    controls.ControlUnavailable,
                    controls.pixel_injections.PixelUnavailable,
                ) as error:
                    skipped = {
                        "source_id": sid,
                        "category": category,
                        "reason": str(error),
                    }
                    write(out / "unavailable-records" / f"{sid}.json", skipped)
                    local_skips[sid] = skipped
                    unavailable.append(skipped)
                    used_skips.add(sid)
                    continue
                write(out / "records" / f"{sid}.json", record)
                local_records[sid] = record
            used_records.add(sid)
            checked = copy.deepcopy(record)
            checked["truth"]["partition"] = (
                "development"
                if complete < plan["development_per_class"]
                else "evaluation"
            )
            if complete < plan["development_per_class"] and (
                checked["truth"] != seed["accepted"][sid]
                or checked["manifest"] != seed["manifest"][sid]
            ):
                raise ValueError(f"False-class development seed changed: {sid}")
            accepted.append(rebase(checked, folder))
            complete += 1
            if folder == out:
                progress["complete"] = len(local_records)
                write(progress_path, progress)
        if complete != quota:
            raise ValueError(
                f"False-class {category} quota cannot be filled: {complete}/{quota}"
            )
    unselected_support = unselected_support_count(
        prior_records,
        local_records,
        prior_skips,
        local_skips,
        used_records,
        used_skips,
    )
    stage = out / "freeze-building"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    result = controls.finish_materialization(
        stage, controls.POOL, accepted, unavailable, plan["classes"], True, seed
    )
    for name in result["artifacts"]:
        (stage / name).replace(out / name)
    result["assembly"] = {
        "prior": str(prior),
        "prior_progress_sha256": report["prior_progress_sha256"],
        "prior_record_digests_sha256": report["record_digests_sha256"],
        "overlay_sha256": identity["overlay_sha256"],
        "local_new_records": len(local_records),
        "unselected_support_skips": unselected_support,
        "download_workers": DOWNLOAD_WORKERS,
    }
    write(out / "materialization.json", result)
    progress["status"] = "COMPLETE"
    write(progress_path, progress)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    require_volumes()
    prior, out = args.prior.resolve(), local_output(args.out)
    if prior == out or not prior.is_dir():
        raise ValueError("A separate read-only prior materialization is required")
    out.mkdir(parents=True, exist_ok=True)
    if args.materialize:
        for name in (
            "records",
            "unavailable-records",
            "pixels",
            "arrays",
            "spatial-tpf-queries",
        ):
            (out / name).mkdir(exist_ok=True)
    controls, pixels = modules(out if args.materialize else None)
    previous = (
        read(out / "prior-audit.json") if (out / "prior-audit.json").exists() else None
    )
    report, rows, plan, seed, records, skips = audit_prior(
        prior, controls, pixels, certificate=previous
    )
    write(out / "prior-audit.json", report)
    if args.materialize:
        result = freeze(
            out, prior, controls, pixels, report, rows, plan, seed, records, skips
        )
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
