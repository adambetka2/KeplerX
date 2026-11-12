"""Freeze verified-only exclusions; every unsupported or unknown proposal is retained."""

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

from pipeline.deterministic import cloud_null as pool
from pipeline.deterministic import run, signal_contracts as signals
from pipeline.photometry import lc2
from pipeline.runtime import ROOT, dump_path, implementation_hashes

QUOTA = 10000
DECLARATION_SHA = "c9b1636c3934b525d10bb745cfadbacf27cd1e06e3b1bd47a3fed8a834abe8d2"
MEMBERSHIP_SHA = "1a9a616f0a6d85d0e41f3e0d56d274302e95adbb2b2c850525dbadfac791b5fa"
ORDER_SHA = "5083362610cc928e7b1f5f86334c19f1317b61e032f28b8aaf79c70605eace8a"
SNAPSHOTS = ROOT / "pipeline/vetting/cloud-catalogues.json"
VERIFIED_INTERNAL = "VERIFIED_ASTROPHYSICAL_SIGNAL"


def reference(path):
    return {"path": str(path.resolve()), "sha256": pool.digest(path)}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    return reference(path)


def development_base(path, expected):
    pool.require(
        pool.valid_hash(expected), "Reviewed development base SHA256 is required"
    )
    base = signals.exclusion_document({"path": str(path.resolve()), "sha256": expected})
    pool.require(
        base.get("schema") == 3
        and base.get("protocol") == lc2.LC2_MANIFEST
        and base.get("role") == "development"
        and base.get("measurement_scope") == "OPENED_NULL_BASE"
        and not base.get("null_construction")
        and base.get("null_flux_constructed") is False
        and base.get("selection_seed") == base.get("transform_seed") == 20261002
        and base.get("declaration", {}).get("sha256") == DECLARATION_SHA
        and base.get("membership", {}).get("sha256") == MEMBERSHIP_SHA,
        "Only the frozen original development null bases may be read",
    )
    ids = list(base["sources"])
    pool.require(
        len(ids) == QUOTA
        and base["ordered_source_ids"] == ids
        and hashlib.sha256(pool.canonical(ids)).hexdigest() == ORDER_SHA,
        "Original development quota or source order changed",
    )
    rows = [base["sources"][sid]["original_pool_row"] for sid in ids]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=pool.FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    pool.require(
        hashlib.sha256(stream.getvalue().encode()).hexdigest() == MEMBERSHIP_SHA,
        "Original source/parent/raw/shard membership changed",
    )
    for sid, row in zip(ids, rows):
        pool.row_identity(row)
        metadata = base["sources"][sid]["metadata"]
        pool.require(
            row["source_id"] == str(metadata["source_id"]) == sid
            and row["origin_parent"] == str(metadata["origin_tpf_kic"])
            and row["raw_fits_sha256"] == metadata["raw_fits_sha256"]
            and int(row["shard"]) == int(metadata["shard"]),
            f"Original development identity changed: {sid}",
        )
    return base


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def ephemeris(name, row):
    if name.startswith(("cumulative", "q1_q17")):
        fields, offset = ("koi_period", "koi_time0bk", "koi_duration"), 0
    elif name.startswith("kepler-eb"):
        fields, offset = ("Per", "BJD0", "duration_hours"), 54833
    elif name.startswith("exofop-ctoi"):
        fields, offset = (
            ("Period (days)", "Transit Epoch (BJD)", "Duration (hrs)"),
            2454833,
        )
    else:
        fields = (
            "pl_orbper",
            "pl_tranmid",
            "pl_trandurh" if name.startswith("toi") else "pl_trandur",
        )
        offset = 2454833
    period, epoch, duration = [finite(row.get(field)) for field in fields]
    if period is None or period <= 0 or epoch is None:
        return None, "MISSING_OR_INVALID_EPHEMERIS"
    if duration is None or duration <= 0:
        return None, "MISSING_OR_INVALID_PHYSICAL_DURATION"
    return {
        "period_days": period,
        "epoch_bkjd": epoch - offset,
        "duration_hours": duration,
        "duration_field": fields[2],
    }, None


def coordinates(name, row):
    keys = (
        ("_RA", "_DE")
        if name.startswith("kepler-eb")
        else (("RA", "Dec") if name.startswith("exofop-ctoi") else ("ra", "dec"))
    )
    ra, dec = [finite(row.get(key)) for key in keys]
    return (
        (ra, dec) if ra is not None and dec is not None and -90 <= dec <= 90 else None
    )


def catalogue_lookups(base, directory, radius):
    ids = list(base["sources"])
    targets, located = [], []
    for sid in ids:
        metadata = base["sources"][sid]["metadata"]
        point = coordinates(
            "source", {"ra": metadata.get("ra_deg"), "dec": metadata.get("dec_deg")}
        )
        if point is not None:
            located.append(sid)
            targets.append(point)
    lookups, candidates, snapshots = (
        {sid: {} for sid in ids},
        {sid: [] for sid in ids},
        {},
    )
    located_ids = set(located)
    specification = json.loads(SNAPSHOTS.read_bytes())
    pool.require(
        len(specification["files"]) == 6,
        "All six frozen catalogue snapshots are required",
    )
    for name, binding in specification["files"].items():
        path = directory / name
        payload = path.read_bytes()
        pool.require(
            len(payload) == binding["bytes"]
            and hashlib.sha256(payload).hexdigest() == binding["sha256"],
            f"Frozen catalogue snapshot changed: {name}",
        )
        rows = (
            json.loads(payload)
            if name.endswith(".json")
            else list(csv.DictReader(io.StringIO(payload.decode())))
        )
        points = [(index, coordinates(name, row)) for index, row in enumerate(rows)]
        valid = [(index, point) for index, point in points if point is not None]
        snapshots[name] = {
            "reference": {"path": str(path.resolve()), "sha256": binding["sha256"]},
            "bytes": len(payload),
            "rows": len(rows),
            "unlocated_rows": [index for index, point in points if point is None],
        }
        for sid in ids:
            lookups[sid][name] = {
                "status": "SCOPED_SNAPSHOT_LOOKUP"
                if sid in located_ids
                else "SOURCE_COORDINATES_UNRESOLVED",
                "snapshot": snapshots[name]["reference"],
                "matched_rows": [],
                "unlocated_snapshot_rows": len(rows) - len(valid),
            }
        if not targets or not valid:
            continue
        target_sky = SkyCoord(
            np.asarray(targets)[:, 0] * u.deg, np.asarray(targets)[:, 1] * u.deg
        )
        sky = np.asarray([point for _, point in valid])
        a, b, separations, _ = search_around_sky(
            target_sky,
            SkyCoord(sky[:, 0] * u.deg, sky[:, 1] * u.deg),
            radius * u.arcsec,
        )
        for target, index, distance in sorted(zip(a, b, separations.arcsec)):
            sid, row_index = located[target], valid[index][0]
            clock, reason = ephemeris(name, rows[row_index])
            item = {
                "reference_id": f"catalogue:{name}:{row_index}",
                "catalogue": name,
                "row_index": row_index,
                "separation_arcsec": float(distance),
                "original_record": rows[row_index],
                "ephemeris": clock,
                "retained_reason": reason,
            }
            lookups[sid][name]["matched_rows"].append(row_index)
            candidates[sid].append(item)
    return snapshots, lookups, candidates


def internal_adjudications(path, expected, base_ref, ids):
    if path is None:
        pool.require(expected is None, "An adjudication SHA requires its index path")
        return {}, None
    binding = {"path": str(path.resolve()), "sha256": expected}
    index = signals.exclusion_document(binding)
    pool.require(
        index.get("role") == "development" and index.get("base_manifest") == base_ref,
        "Internal adjudications differ from the original development base",
    )
    grouped = {}
    for row in index["entries"]:
        pool.require(
            row["source_id"] in ids,
            "Internal adjudication is outside development membership",
        )
        grouped.setdefault(row["source_id"], []).append(row["adjudication"])
    return grouped, binding


def internal_candidate(binding, sid, metadata):
    try:
        review = signals.exclusion_document(binding)
    except FileNotFoundError:
        return {"adjudication": binding, "retained_reason": "ADJUDICATION_UNAVAILABLE"}
    pool.require(
        review["source_id"] == sid
        and review["raw_fits_sha256"] == metadata["raw_fits_sha256"],
        "Internal adjudication source identity changed",
    )
    item = {
        "reference_id": f"internal:{review['signal_id']}",
        "adjudication": binding,
        "retained_reason": "NO_VERIFIED_ADJUDICATION",
    }
    if review.get("status") != VERIFIED_INTERNAL:
        return item
    if not all(review.get(key) for key in ("reviewer", "rationale", "evidence")):
        item["retained_reason"] = "ADJUDICATION_ATTRIBUTION_INCOMPLETE"
        return item
    try:
        evidence = signals.exclusion_document(review["evidence"])
    except FileNotFoundError:
        item["retained_reason"] = "PINNED_INTERNAL_EVIDENCE_UNAVAILABLE"
        return item
    pool.require(
        evidence["source_id"] == sid
        and evidence["raw_fits_sha256"] == metadata["raw_fits_sha256"],
        "Internal evidence source identity changed",
    )
    signal = next(
        (row for row in evidence["signals"] if row["signal_id"] == review["signal_id"]),
        None,
    )
    if signal is None:
        item["retained_reason"] = "VERIFIED_SIGNAL_ABSENT_FROM_PINNED_EVIDENCE"
        return item
    from pipeline.deterministic.planet_control_report import (
        disposition,
        fitted_event_units,
    )

    try:
        units = fitted_event_units(
            evidence, signal, disposition(evidence, signal["signal_id"])
        )
    except (KeyError, ValueError):
        item["retained_reason"] = "VERIFIED_SIGNAL_PHYSICAL_TIMING_UNRESOLVED"
        return item
    centres = [finite(unit.get("centre_bkjd")) for unit in units or []]
    if any(centre is None for centre in centres) or np.any(np.diff(centres) <= 0):
        item["retained_reason"] = "VERIFIED_SIGNAL_PHYSICAL_TIMING_UNRESOLVED"
        return item
    duration = finite(review.get("physical_duration_hours"))
    if not units or duration is None or duration <= 0:
        item["retained_reason"] = (
            "VERIFIED_SIGNAL_PHYSICAL_TIMING_OR_DURATION_UNAVAILABLE"
        )
        return item
    item.update(
        events=[
            {
                "event_id": unit["unit_id"],
                "centre_bkjd": unit["centre_bkjd"],
                "quarter": unit["quarter"],
            }
            for unit in units
        ],
        duration_hours=duration,
        source_evidence=review["evidence"],
        retained_reason=None,
    )
    return item


def preparation_binding(path, expected, base_ref, base):
    if path is None:
        pool.require(expected is None, "A preparation SHA requires its path")
        return {}, None
    binding = {"path": str(path.resolve()), "sha256": expected}
    prepared = signals.exclusion_document(binding)
    ids = list(base["sources"])
    sources = {row["source_id"]: row for row in prepared["sources"]}
    selected = [sid for sid in ids if sid in sources]
    pool.require(
        prepared.get("status") in {"PREPARED", "PREPARATION_FAILED"}
        and prepared.get("input_manifest") == base_ref
        and prepared.get("evaluation_scope") is None
        and prepared["requested_sources"] == len(ids)
        and prepared["preparation_counts"]["attempted_sources"] == len(ids)
        and list(sources) == prepared["selection"]["source_ids"] == selected
        and len(sources) == len(prepared["sources"])
        and prepared["cohort_sha256"]
        == hashlib.sha256(pool.canonical(selected)).hexdigest()
        and prepared["detector_contract"] == run.detector_contract(),
        "Preparation changed the original base, source order, quota or discovery contract",
    )
    for sid, source in sources.items():
        metadata = base["sources"][sid]["metadata"]
        pool.require(
            source["parent_tpf"] == str(metadata["origin_tpf_kic"])
            and source["raw_fits_sha256"] == metadata["raw_fits_sha256"],
            f"Prepared source parent/raw identity changed: {sid}",
        )
    return sources, binding


def prepared_support(sid, prepared, binding):
    if binding is None:
        return None, {"status": "PREPARATION_UNSUPPLIED"}
    if sid not in prepared:
        return None, {
            "status": "INELIGIBLE_DISCOVERY_PREPARATION",
            "preparation": binding,
        }
    source = prepared[sid]
    item = source["inputs"].get("psf_flat")
    if item is None:
        return None, {"status": "DISCOVERY_VIEW_UNAVAILABLE", "preparation": binding}
    try:
        path = Path(item["fsc_path"])
        stamp = lc2.file_stamp(path)
        signals.verify_exclusion_file(
            {"path": str(path.resolve()), "sha256": item["fsc_sha256"]}
        )
        records = run.read_fsc(path, sid, "psf_flat")
        pool.require(
            stamp == lc2.file_stamp(path)
            and np.isfinite(records["time"]).all()
            and np.all(np.diff(records["time"]) >= 0),
            "Frozen prepared FSC changed or has invalid timing",
        )
    except OSError as error:
        return None, {"status": "ORIGINAL_INPUT_UNAVAILABLE", "detail": str(error)}
    if len(records) < 100:
        return None, {
            "status": "INELIGIBLE_DISCOVERY_PREPARATION",
            "retained_cadences": len(records),
        }
    return records, {
        "status": "PREPARED_DISCOVERY_SUPPORT",
        "retained_cadences": len(records),
        "quarters": list(map(int, np.unique(records["quarter"]))),
        "prepared_records_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
        "preparation": binding,
        "prepared_fsc": {"path": str(path.resolve()), "sha256": item["fsc_sha256"]},
    }


def observed_events(records, centres, duration, prefix, quarters=None):
    pool.require(
        all(math.isfinite(float(value)) for value in centres)
        and np.all(np.diff(centres) > 0),
        "Verified event clock is invalid or reordered",
    )
    events = []
    for index, centre in enumerate(centres):
        inside = abs(records["time"] - centre) <= duration / 48
        if quarters is not None:
            inside &= records["quarter"] == quarters[index]
        values, counts = np.unique(records["quarter"][inside], return_counts=True)
        if len(values):
            events.append(
                {
                    "event_id": f"{prefix}:{index}",
                    "centre_bkjd": float(centre),
                    "support": {
                        str(int(quarter)): {"psf_flat": int(count)}
                        for quarter, count in zip(values, counts)
                    },
                }
            )
    return events


def candidate_reference(item, records, snapshots, base_ref):
    if item.get("retained_reason") or records is None:
        return None
    if "catalogue" in item:
        clock = item["ephemeris"]
        period, epoch, duration = (
            clock["period_days"],
            clock["epoch_bkjd"],
            clock["duration_hours"],
        )
        first = math.ceil(
            (float(records["time"].min()) - duration / 48 - epoch) / period
        )
        last = math.floor(
            (float(records["time"].max()) + duration / 48 - epoch) / period
        )
        centres = [epoch + cycle * period for cycle in range(first, last + 1)]
        events = observed_events(records, centres, duration, item["reference_id"])
        verification = {
            "status": "FROZEN_CATALOGUE_EPHEMERIS",
            "catalogue": item["catalogue"],
            "snapshot": snapshots[item["catalogue"]]["reference"],
            "row_index": item["row_index"],
        }
        origin = "CATALOGUE_EPHEMERIS"
    else:
        duration = item["duration_hours"]
        events = observed_events(
            records,
            [row["centre_bkjd"] for row in item["events"]],
            duration,
            item["reference_id"],
            [row["quarter"] for row in item["events"]],
        )
        verification = {
            "status": "PINNED_VERIFIED_ADJUDICATION",
            "adjudication": item["adjudication"],
            "source_evidence": item["source_evidence"],
        }
        origin = "REAL_SIGNAL"
    if not events:
        return None
    verification["base_manifest_sha256"] = base_ref["sha256"]
    return {
        "reference_id": item["reference_id"],
        "origin": origin,
        "duration_hours": duration,
        "events": events,
        "verification": verification,
    }


def source_documents(
    sid,
    base,
    base_ref,
    lookups,
    candidates,
    internal,
    snapshots,
    out,
    prepared,
    preparation,
):
    metadata = base["sources"][sid]["metadata"]
    adjudications = [internal_candidate(binding, sid, metadata) for binding in internal]
    items = candidates + adjudications
    need_support = any(not item.get("retained_reason") for item in items)
    if need_support:
        records, support = prepared_support(sid, prepared, preparation)
    else:
        records, support = None, {"status": "NO_VERIFIED_REFERENCE_CANDIDATES"}
        if preparation is not None and sid not in prepared:
            support["status"] = "INELIGIBLE_DISCOVERY_PREPARATION"
    references, retained = [], []
    for item in items:
        found = candidate_reference(item, records, snapshots, base_ref)
        if found is None:
            retained.append(
                {
                    **item,
                    "retained_reason": item.get("retained_reason")
                    or (
                        support["status"]
                        if records is None
                        else "NO_PREPARED_QUARTER_SUPPORT"
                    ),
                }
            )
        else:
            references.append(found)
    pool.require(
        len({row["reference_id"] for row in references}) == len(references),
        "Duplicate verified exclusion identity",
    )
    support.update(
        source_id=sid,
        raw_fits_sha256=metadata["raw_fits_sha256"],
        base_manifest_sha256=base_ref["sha256"],
        original_preparation=preparation,
        supported_references={
            row["reference_id"]: {key: row[key] for key in ("duration_hours", "events")}
            for row in references
        },
    )
    support_ref = save(out / "support" / f"{sid}.json", support)
    for row in references:
        row["verification"]["prepared_support"] = support_ref
    common = {
        "source_id": sid,
        "base_manifest_sha256": base_ref["sha256"],
        "raw_fits_sha256": metadata["raw_fits_sha256"],
        "coverage_complete": True,
        "coverage_scope": signals.EXCLUSION_COVERAGE,
        "unknown_proposals": "RETAIN_IN_NULL_TRIAL_COUNT",
        "prepared_support": support_ref,
    }
    known = save(
        out / "catalogue" / f"{sid}.json",
        {
            **common,
            "lookups": lookups,
            "references": [
                row for row in references if row["origin"] == "CATALOGUE_EPHEMERIS"
            ],
            "retained": [row for row in retained if "catalogue" in row],
        },
    )
    real = save(
        out / "internal" / f"{sid}.json",
        {
            **common,
            "internal_accounting": "PINNED_ADJUDICATIONS_ONLY; absence gives no internal exclusion",
            "original_proposal_inventory": "NOT_REQUIRED_UNKNOWN_PROPOSALS_RETAINED",
            "references": [row for row in references if row["origin"] == "REAL_SIGNAL"],
            "retained": [row for row in retained if "catalogue" not in row],
        },
    )
    return {
        "coverage_complete": True,
        "real_signals": real,
        "catalogue_ephemerides": known,
    }, {
        "verified_references": len(references),
        "retained_catalogue_or_internal_cases": len(retained),
        "preparation_status": support["status"],
        "retained_reasons": Counter(row["retained_reason"] for row in retained),
    }


def produce(
    path,
    expected,
    directory,
    out,
    adjudications=None,
    adjudications_sha=None,
    preparation=None,
    preparation_sha=None,
):
    base = development_base(path, expected)
    base_ref = {"path": str(path.resolve()), "sha256": expected}
    prepared, preparation_ref = preparation_binding(
        preparation, preparation_sha, base_ref, base
    )
    config = ROOT / "pipeline/vetting/config.json"
    radius = json.loads(config.read_bytes())["neighbour_radius_arcsec"]
    snapshots, lookups, candidates = catalogue_lookups(base, directory, radius)
    internal, internal_ref = internal_adjudications(
        adjudications, adjudications_sha, base_ref, base["sources"]
    )
    out.mkdir(parents=True, exist_ok=False)
    sources, totals, statuses, reasons = {}, Counter(), Counter(), Counter()
    for sid in sorted(
        base["sources"], key=lambda sid: base["sources"][sid]["lc2_folder"]
    ):
        sources[sid], counts = source_documents(
            sid,
            base,
            base_ref,
            lookups[sid],
            candidates[sid],
            internal.get(sid, []),
            snapshots,
            out,
            prepared,
            preparation_ref,
        )
        totals.update(
            {
                key: counts[key]
                for key in (
                    "verified_references",
                    "retained_catalogue_or_internal_cases",
                )
            }
        )
        statuses[counts["preparation_status"]] += 1
        reasons.update(counts["retained_reasons"])
    index = {
        "contract": signals.NULL_CONTRACT,
        "exclusion_rule": signals.EXCLUSION_RULE,
        "coverage_scope": signals.EXCLUSION_COVERAGE,
        "unknown_proposals": "RETAIN_IN_NULL_TRIAL_COUNT",
        "role": "development",
        "base_manifest": base_ref,
        "membership_sha256": MEMBERSHIP_SHA,
        "ordered_source_ids_sha256": ORDER_SHA,
        "selection_seed": 20261002,
        "transform_seed": 20261002,
        "catalogue_snapshots": snapshots,
        "radius_arcsec": radius,
        "catalogue_specification": reference(SNAPSHOTS),
        "internal_adjudication_index": internal_ref,
        "original_preparation": preparation_ref,
        "sources": {sid: sources[sid] for sid in base["sources"]},
    }
    index_ref = save(out / "exclusions.json", index)
    receipt = {
        "status": "COMPLETE_VERIFIED_ONLY_EXCLUSION_ACCOUNTING",
        "exclusions": index_ref,
        "base_manifest": base_ref,
        "coverage_scope": signals.EXCLUSION_COVERAGE,
        "sources": len(sources),
        "source_lookups": len(sources) * len(snapshots),
        "counts": dict(totals),
        "preparation_statuses": dict(statuses),
        "retained_reasons": dict(reasons),
        "implementation": implementation_hashes(),
        "vetting_config": reference(config),
        "catalogue_specification": reference(SNAPSHOTS),
        "evaluation_arrays_read": False,
        "original_source_detection_or_vetting_run": False,
        "complete_real_signal_inventory_claim": False,
        "original_lc2_arrays_read": False,
        "original_preparation": preparation_ref,
        "membership_sha256": MEMBERSHIP_SHA,
        "ordered_source_ids_sha256": ORDER_SHA,
        "selection_seed": 20261002,
        "transform_seed": 20261002,
        "memberships_or_seeds_changed": False,
        "calibration_run": False,
    }
    save(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument(
        "--catalogue-directory",
        type=Path,
        default=ROOT / "discPCIe/data/reference-catalogues",
    )
    parser.add_argument("--internal-adjudications", type=Path)
    parser.add_argument("--internal-adjudications-sha256")
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--preparation-sha256")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = produce(
        args.base_manifest,
        args.base_sha256,
        args.catalogue_directory,
        dump_path(args.out),
        args.internal_adjudications,
        args.internal_adjudications_sha256,
        args.preparation,
        args.preparation_sha256,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "exclusions",
                    "sources",
                    "source_lookups",
                    "counts",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
