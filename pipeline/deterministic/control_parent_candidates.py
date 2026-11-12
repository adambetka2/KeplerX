"""Reconcile every LC stamp with public identity candidate headers."""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

from pipeline.deterministic.control_bindings import (
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_donor_headers import verify_targets
from pipeline.deterministic.control_eligibility import (
    BOUNDARY,
    CATALOGUE_SHA,
    digest,
    save,
    verify_plan,
)
from pipeline.deterministic.control_eligibility_archive import capacity
from pipeline.deterministic.control_parent_headers import (
    archive_prefix,
    parent_url,
    stamp_witness,
    verify_parent_headers,
)

RADIUS_DEG = 1 / 60


def candidate_projection(catalogue, sources):
    require(digest(catalogue) == CATALOGUE_SHA, "Public identity catalogue changed")
    result = {}
    with fits.open(catalogue, memmap=True) as hdus:
        table = hdus[1].data
        ra, dec = np.asarray(table["ra"]), np.asarray(table["dec"])
        for source in sources:
            target_ra, target_dec = source["product"]["ra"], source["product"]["dec"]
            dra = np.deg2rad((ra - target_ra + 180) % 360 - 180)
            ddec = np.deg2rad(dec - target_dec)
            haversine = (
                np.sin(ddec / 2) ** 2
                + np.cos(np.deg2rad(dec))
                * math.cos(math.radians(target_dec))
                * np.sin(dra / 2) ** 2
            )
            selected = np.flatnonzero(
                haversine <= math.sin(math.radians(RADIUS_DEG) / 2) ** 2
            )
            rows = [
                {
                    "original_row_index": int(i),
                    "kic": int(table["kic"][i]),
                    "global_origin_parent": int(table["tpforg"][i]),
                    "gaia_designation": str(table["gaia_designation"][i]).strip(),
                    "ra": float(ra[i]),
                    "dec": float(dec[i]),
                }
                for i in selected
            ]
            parents = sorted(
                {
                    v
                    for row in rows
                    for v in (row["kic"], row["global_origin_parent"])
                    if v > 0
                }
                | {int(source["parent_tpf"])}
            )
            result[source["source_id"]] = {
                "coordinates_deg": [target_ra, target_dec],
                "radius_deg": RADIUS_DEG,
                "public_identity_rows": rows,
                "candidate_parent_ids": parents,
                "complete_archive_neighbourhood": False,
            }
    return result


def candidate_record(session, parent, quarter, timestamp):
    record = {
        "parent_kic": str(parent),
        "quarter": quarter,
        "attempts": [],
        "status": "UNAVAILABLE",
    }
    try:
        headers, total = archive_prefix(
            session, parent_url(parent, timestamp), record["attempts"]
        )
        record.update(
            headers=headers,
            product_bytes=total,
            geometry=verify_parent_headers(headers, parent, quarter),
            status="VERIFIED_PARENT_HEADERS",
        )
    except (ValueError, OSError, KeyError, requests.RequestException) as error:
        record.update(error=type(error).__name__, message=str(error))
    return record


def exact_stamp_match(quarter, aperture, record):
    witness = stamp_witness(quarter, record)
    geometry = record["geometry"]
    dimensions = (
        None
        if aperture is None
        else (aperture["cards"]["NAXIS1"], aperture["cards"]["NAXIS2"])
    )
    witness["aperture_stamp_dimensions_match"] = (
        None
        if dimensions is None
        else (dimensions == (geometry["stamp_columns"], geometry["stamp_rows"]))
    )
    witness["exact_declared_geometry_match"] = (
        witness["stamp_origin_and_channel_match"]
        and witness["aperture_stamp_dimensions_match"]
    )
    witness["exact_declared_geometry_match"] = bool(
        witness["exact_declared_geometry_match"]
    )
    return witness


def reconcile_candidates(source, quarter, aperture, records):
    identities = [record["parent_kic"] for _, record in records]
    require(
        len(identities) == len(set(identities)), "Repeated candidate parent identity"
    )
    matches = []
    attempts = []
    for path, record in records:
        require(record["quarter"] == quarter["quarter"], "Candidate quarter changed")
        row = {
            "parent_kic": record["parent_kic"],
            "record": reference(path),
            "status": record["status"],
        }
        if record["status"] == "VERIFIED_PARENT_HEADERS":
            require(
                verify_parent_headers(
                    record["headers"], record["parent_kic"], record["quarter"]
                )
                == record["geometry"],
                "Candidate header geometry changed",
            )
            row["witness"] = exact_stamp_match(quarter, aperture, record)
            if row["witness"]["exact_declared_geometry_match"]:
                matches.append(record["parent_kic"])
        attempts.append(row)
    status = "NO_MATCHING_PUBLIC_CANDIDATE"
    if matches:
        status = (
            "ONE_MATCH_WITHIN_PUBLIC_CANDIDATE_SET"
            if len(matches) == 1
            else "AMBIGUOUS_MATCHING_PUBLIC_CANDIDATES"
        )
    if aperture is None:
        status = "LC_APERTURE_HEADER_UNAVAILABLE"
    return {
        "source_id": source["source_id"],
        "cohort": source["cohort"],
        "quarter": quarter["quarter"],
        "original_global_parent": source["parent_tpf"],
        "original_raw_fits_sha256": source["pool"]["raw_fits_sha256"],
        "lightcurve_header_sha256": quarter["archive_header"]["sha256"],
        "aperture_header_sha256": None if aperture is None else aperture["sha256"],
        "matching_candidate_parents": matches,
        "status": status,
        "candidates": attempts,
        "original_extraction_parent_uniqueness_verified": False,
        "complete_archive_neighbourhood": False,
        "physical_attribution_measured": False,
    }


def nominal_exact_match(source, quarter, aperture, row):
    require(row["cohort"] == source["cohort"], "Nominal quarter cohort changed")
    record = read_json(verified_reference(row["record"]))
    require(
        record["parent_kic"] == source["parent_tpf"]
        and record["quarter"] == quarter["quarter"],
        "Nominal quarter/parent identity changed",
    )
    if record["status"] == "UNAVAILABLE":
        return False
    require(
        record["status"] == "VERIFIED_PARENT_HEADERS"
        and record["geometry"]
        == verify_parent_headers(
            record["headers"], record["parent_kic"], record["quarter"]
        ),
        "Nominal cached geometry changed",
    )
    return exact_stamp_match(quarter, aperture, record)["exact_declared_geometry_match"]


def candidate_records(session, parents, quarter, timestamp, directory, cache):
    records = []
    for parent in parents:
        capacity(directory.parent)
        path = directory / "records" / f"{parent}-q{quarter}.json"
        cached = None if cache is None else cache / "records" / path.name
        if cached is not None and cached.is_file():
            path = cached
        if not path.exists():
            save(path, candidate_record(session, parent, quarter, timestamp))
        record = read_json(path)
        require(str(parent) == record["parent_kic"], "Cached candidate parent changed")
        records.append((path, record))
    return records


def collect(plan_path, target_report_path, parent_report_path, directory, cache=None):
    plan = read_json(plan_path)
    verify_plan(plan)
    report = read_json(target_report_path)
    verify_targets(report, plan, plan_path)
    nominal = read_json(parent_report_path)
    require(
        nominal["target_report"] == reference(target_report_path),
        "Nominal report identity differs",
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    projection = candidate_projection(
        plan["inputs"]["public_catalogue"]["path"], plan["sources"]
    )
    save(directory / "public-candidate-projection.json", projection)
    mapping = read_json(Path(parent_report_path).parent / "quarter-mapping.json")
    nominal_index = {(r["source_id"], r["quarter"]): r for r in nominal["quarters"]}
    require(len(nominal_index) == len(nominal["quarters"]), "Repeated nominal quarter")
    rows = []
    with requests.Session() as session:
        for source, target in zip(plan["sources"], report["sources"]):
            archive = read_json(target["archive_record"]["path"])
            quarters = target.get(
                "quarter_witnesses",
                target.get("witness", {}).get("quarter_witnesses", []),
            )
            for quarter in quarters:
                sid, q = source["source_id"], quarter["quarter"]
                aperture = next(
                    (h for h in archive["headers"] if h["name"] == f"APERTURE_Q{q}"),
                    None,
                )
                require((sid, q) in nominal_index, "Missing nominal quarter witness")
                nominal_exact_match(source, quarter, aperture, nominal_index[sid, q])
                records = candidate_records(
                    session,
                    projection[sid]["candidate_parent_ids"],
                    q,
                    mapping[str(q)],
                    directory,
                    cache,
                )
                rows.append(reconcile_candidates(source, quarter, aperture, records))
            print(
                f"Retained candidate reconciliation for {source['source_id']}",
                flush=True,
            )
    result = {
        "schema": 1,
        "plan": reference(plan_path),
        "target_report": reference(target_report_path),
        "nominal_parent_report": reference(parent_report_path),
        "public_candidates": reference(directory / "public-candidate-projection.json"),
        "implementation": reference(Path(__file__)),
        "nominal_match_rule": "ORIGIN_CHANNEL_AND_APERTURE_DIMENSIONS",
        "quarter_coverage": "ALL_ORIGINAL_OBSERVED_QUARTERS",
        "quarters": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "boundary": BOUNDARY,
    }
    save(directory / "candidate-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "target-report", "parent-report", "directory"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--record-cache", type=Path)
    args = parser.parse_args()
    collect(
        args.plan,
        args.target_report,
        args.parent_report,
        args.directory,
        args.record_cache,
    )


if __name__ == "__main__":
    main()
