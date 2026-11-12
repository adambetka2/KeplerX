"""Witness the unchanged prospective controls without freezing their roles."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from astropy.io import fits

from pipeline.deterministic.control_bindings import (
    digest,
    read_json,
    reference,
    require,
    verified_reference,
)

PACKED_SHA = "82f8bfd24ee2fa6c2c9929e1d1bddb6262a3a85fe7e842cc0b10f9949cd09575"
DECODED_SHA = "6651da608c044407f18b4d768ef1bb22e3195153780e343971a74f4f316110e6"
DECODED_BYTES = 3857660
CATALOGUE_SHA = "2e6d5a87cb90f941bdc207aef93d61847df228ae7d56ffd38d3ab9255dd30fcf"
POOL_SHA = "8321076d0bb47b81c6d4e359d5c6f7f0ced6730680d7809394fd0f6ae1d23923"
COHORTS = {"guide_star_family", "contamination_pair"}
FIELDS = (
    "gaia_designation",
    "ra",
    "dec",
    "tpforg",
    "kic",
    "kic_sep",
    "fname",
    "psf_avail",
    "sap_avail",
)
FAMILY_RULE = (
    "TRANSIT_LIKE, period within 1.2% of 1.6958 d times 1, 2 or 3, epoch phase 0.3 to 0.5"
)
BOUNDARY = {
    "scientific_acceptance": False,
    "joint_independence_complete": False,
    "cohorts_frozen": False,
    "contamination_attributed": False,
    "guide_mechanism_attributed": False,
    "sealed_science_read": False,
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    require(not path.is_symlink() and not temporary.is_symlink(), "Output symlink")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def opened_reference(folder):
    folder = Path(folder)
    packed = folder / "opened-metadata.json.gz"
    require(digest(packed) == PACKED_SHA, "Opened packet hash changed")
    with gzip.open(packed, "rb") as stream:
        decoded = stream.read(DECODED_BYTES + 1)
    require(
        len(decoded) == DECODED_BYTES
        and hashlib.sha256(decoded).hexdigest() == DECODED_SHA,
        "Opened packet decoded identity changed",
    )
    packet = json.loads(decoded)
    validate_packet(packet)
    return packet, reference(packed)


def validate_packet(packet):
    selections = packet["selections"]
    require(set(selections) == COHORTS, "Prospective cohort scope changed")
    all_ids = []
    for rows in selections.values():
        require(len(rows) == 100, "Prospective selection count changed")
        for row in rows:
            require(
                set(row) == {"source_id", "parent_tpf", "shard"}
                and all(re.fullmatch("[0-9]+", row[key]) for key in row)
                and int(row["source_id"]) > 0
                and int(row["parent_tpf"]) > 0,
                "Missing or invalid prospective identity",
            )
            all_ids.append(row["source_id"])
    require(len(set(all_ids)) == 200, "Repeated or overlapping prospective source")
    require(
        packet["original_family_rule"]["rule"] == FAMILY_RULE
        and len(packet["original_family_proposals"]) == 984
        and len(packet["original_family_source_ids"]) == 239
        and packet["scientific_acceptance"] is False
        and packet["frozen"] is False,
        "Original family recipe or boundary changed",
    )
    require(
        {row["source_id"] for row in selections["guide_star_family"]}
        <= set(packet["original_family_source_ids"]),
        "Guide selection absent from original family",
    )
    for row in packet["original_family_proposals"]:
        family_ephemeris(row)


def scalar(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def catalogue_projection(path, selections):
    require(digest(path) == CATALOGUE_SHA, "Public identity catalogue hash changed")
    ids = {row["source_id"] for rows in selections.values() for row in rows}
    parents = [int(row["parent_tpf"]) for row in selections["contamination_pair"]]
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        names = np.char.replace(
            np.char.strip(np.asarray(table["gaia_designation"]).astype(str)),
            "Gaia DR3 ",
            "",
        )
        selected = np.isin(names, list(ids)) | np.isin(table["kic"], parents)
        rows = [
            {"original_row_index": int(i), **{k: scalar(table[k][i]) for k in FIELDS}}
            for i in np.flatnonzero(selected)
        ]
    return rows


def unique_sources(rows, selected_ids):
    result = {}
    for row in rows:
        sid = row["gaia_designation"].split()[-1]
        if sid not in selected_ids:
            continue
        require(sid not in result, "Ambiguous source/product catalogue identity")
        result[sid] = row
    require(set(result) == selected_ids, "Missing source/product catalogue identity")
    return result


def product_identity(row):
    name = row["fname"]
    sid = row["gaia_designation"].split()[-1]
    if row["kic"] > 0:
        identifier = f"{row['kic']:09d}"
        require(name == f"kic-{identifier}", "KIC product identity differs")
    else:
        identifier = sid
        require(name == f"gaia-dr3-{sid}", "Gaia product identity differs")
    uri = (
        f"mast:HLSP/kbonus-bkg/lcs/{identifier[:4]}/{identifier}/"
        f"hlsp_kbonus-bkg_kepler_kepler_{name}_kepler_v1.0_lc.fits"
    )
    return {
        "uri": uri,
        "url": "https://mast.stsci.edu/api/v0.1/Download/file?"
        + urlencode({"uri": uri}),
    }


def pool_bindings(path, selections):
    require(digest(path) == POOL_SHA, "Original stopped-pool identity hash changed")
    selected = {row["source_id"]: row for rows in selections.values() for row in rows}
    found = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            sid = row["source_id"]
            if sid not in selected:
                continue
            require(sid not in found, "Repeated selected stopped-pool identity")
            require(
                row["origin_parent"] == selected[sid]["parent_tpf"]
                and row["shard"] == selected[sid]["shard"],
                "Original source/parent/shard binding differs",
            )
            found[sid] = row
    require(set(found) == set(selected), "Selected stopped-pool identity missing")
    return found


def catalogue_parent(row):
    value = row.get("kepid", row.get("KIC"))
    require(value is not None, "Catalogue donor parent missing")
    return str(int(value))


def reject_protected_sources(selections, exclusions):
    selected = {row["source_id"] for rows in selections.values() for row in rows}
    protected = {str(sid) for row in exclusions.values() for sid in row["source_ids"]}
    require(
        not selected & protected, "Prospective source overlaps sealed raw exclusions"
    )


def donor_bindings(packet, projection):
    results = []
    for selected in packet["selections"]["contamination_pair"]:
        parent = selected["parent_tpf"]
        catalogues = []
        for catalogue in packet["catalogues"]:
            rows = [
                r for r in catalogue["parent_rows"] if catalogue_parent(r) == parent
            ]
            catalogues.append(
                {
                    "original_snapshot": {
                        k: catalogue[k] for k in ("path", "bytes", "sha256")
                    },
                    "parent_rows": rows,
                }
            )
        results.append(
            {
                **selected,
                "catalogues": catalogues,
                "public_donor_product_identities": [
                    row for row in projection if str(row["kic"]) == parent
                ],
                "catalogue_parent_binding": any(c["parent_rows"] for c in catalogues),
                "target_donor_attribution": "NOT_MEASURED",
                "scene_acceptance": False,
            }
        )
    return results


def family_ephemeris(row):
    period, epoch = float(row["period_days"]), float(row["t0_bkjd"])
    matches = [m for m in (1, 2, 3) if abs(period / (1.6958 * m) - 1) <= 0.012]
    phase = (epoch % 1.6958) / 1.6958
    require(
        row["classification"] == "TRANSIT_LIKE"
        and len(matches) == 1
        and 0.3 <= phase <= 0.5
        and math.isfinite(epoch),
        "Original family ephemeris violates its declared recipe",
    )
    return {
        "period_multiple": matches[0],
        "base_clock_phase": phase,
        "period_days": period,
        "t0_bkjd": epoch,
        "duration_hours": float(row["duration_hours"]),
    }


def build_plan(reference_dir, catalogue, pool):
    packet, packed = opened_reference(reference_dir)
    metadata_path = Path(__file__).with_name("cloud-control-metadata.json")
    metadata = read_json(metadata_path)
    reject_protected_sources(packet["selections"], metadata["sealed_exclusions"])
    projection = catalogue_projection(catalogue, packet["selections"])
    ids = {r["source_id"] for rows in packet["selections"].values() for r in rows}
    products = unique_sources(projection, ids)
    identities = pool_bindings(pool, packet["selections"])
    sources = []
    for cohort, rows in packet["selections"].items():
        for selected in rows:
            sid = selected["source_id"]
            require(
                int(products[sid]["tpforg"]) == int(selected["parent_tpf"]),
                "Public catalogue extraction parent differs from original selection",
            )
            sources.append(
                {
                    **selected,
                    "cohort": cohort,
                    "pool": identities[sid],
                    "product": products[sid],
                    **product_identity(products[sid]),
                }
            )
    return {
        "schema": 1,
        "status": "UNCHANGED_PROSPECTIVE_ELIGIBILITY_PLAN",
        "inputs": {
            "opened_packet": packed,
            "public_catalogue": reference(catalogue),
            "stopped_identity_pool": reference(pool),
            "sealed_identity_exclusions": reference(metadata_path),
        },
        "sources": sources,
        "donor_bindings": donor_bindings(packet, projection),
        "original_family_rule": packet["original_family_rule"],
        "original_family_proposals": packet["original_family_proposals"],
        "selected_source_proposals": packet["selected_source_proposals"],
        "implementation": implementation(),
        "boundary": BOUNDARY,
    }


def implementation():
    root = Path(__file__).resolve().parents[2]
    files = (
        "pipeline/deterministic/control_eligibility.py",
        "pipeline/deterministic/control_eligibility_archive.py",
        "pipeline/deterministic/cloud_public.py",
        "pipeline/deterministic/catalogue_controls.py",
        "pipeline/deterministic/run.py",
        "pipeline/deterministic/signal_contracts.py",
        "pipeline/photometry/lc2.py",
    )
    return {name: digest(root / name) for name in files}


def verify_plan(plan):
    require(
        plan["schema"] == 1 and plan["boundary"] == BOUNDARY, "Plan boundary changed"
    )
    for row in plan["inputs"].values():
        verified_reference(row)
    require(
        plan["inputs"]["opened_packet"]["sha256"] == PACKED_SHA, "Packet pin changed"
    )
    packed = Path(plan["inputs"]["opened_packet"]["path"])
    rebuilt = build_plan(
        packed.parent,
        plan["inputs"]["public_catalogue"]["path"],
        plan["inputs"]["stopped_identity_pool"]["path"],
    )
    require(plan == rebuilt, "Eligibility plan recipe, identities or scope changed")


def header_identity(source, headers):
    primary = headers[0]["cards"]
    require(
        str(primary["GAIAID"]).split()[-1] == source["source_id"], "FITS Gaia mismatch"
    )
    expected_kic = source["product"]["kic"]
    observed = primary["KEPLERID"]
    require(
        (int(observed) == expected_kic) if expected_kic > 0 else observed == "",
        "FITS product KIC mismatch",
    )
    require(
        int(primary["TPFORG"]) == int(source["parent_tpf"]),
        "FITS origin parent mismatch",
    )
    quarters = []
    for header in headers:
        name, cards = header["name"], header["cards"]
        if not re.fullmatch("LIGHTCURVE_Q[0-9]+", name):
            continue
        quarter = int(name.split("_Q")[-1])
        require(cards["QUARTER"] == quarter, "Quarter extension/header mismatch")
        quarters.append(
            {
                "quarter": quarter,
                "archive_header": header,
                "global_origin_parent": source["parent_tpf"],
                "quarter_parent": str(cards["TPFORG"]) if cards.get("TPFORG") else None,
                "quarter_parent_status": "EXPLICIT_QUARTER_HEADER"
                if cards.get("TPFORG")
                else "NOT_DECLARED_IN_LIGHTCURVE_HEADER",
            }
        )
    require(
        len({q["quarter"] for q in quarters}) == len(quarters),
        "Duplicate quarter header",
    )
    return quarters


def mask_exposure(raw):
    from pipeline.deterministic.cloud_public import photometry
    from pipeline.deterministic.run import InsufficientCadences, prepare_records
    from pipeline.deterministic.signal_contracts import relative_curve

    curve = photometry(raw, "psf_flat")
    finite = (
        np.isfinite(curve["time"])
        & np.isfinite(curve["flux"])
        & np.isfinite(curve["error"])
        & (curve["error"] > 0)
    )
    quality = finite & (curve["quality"] == 0)
    flatten = quality & (curve["flatten"] == 0)
    indices, _ = relative_curve(curve, mask_flatten=True)
    kept = np.zeros(len(curve["time"]), dtype=bool)
    kept[indices] = True
    per_quarter = []
    for quarter in np.unique(curve["quarter"]):
        mask = curve["quarter"] == quarter
        per_quarter.append(
            {
                "quarter": int(quarter),
                "raw_cadences": int(mask.sum()),
                "finite_positive_error": int((mask & finite).sum()),
                "after_sap_quality_zero": int((mask & quality).sum()),
                "after_flatten_mask_zero": int((mask & flatten).sum()),
                "after_normalization_upper_clip_and_dedup": int((mask & kept).sum()),
            }
        )
    try:
        records = prepare_records(raw)
        status = "PREPARATION_ELIGIBLE"
        records_sha = hashlib.sha256(records.tobytes()).hexdigest()
    except InsufficientCadences:
        status, records_sha = "INSUFFICIENT_PREPARED_CADENCES", None
    result = {
        "status": status,
        "view": "psf_flat",
        "quarters": per_quarter,
        "prepared_cadences": len(indices),
        "prepared_records_sha256": records_sha,
        "kept_original_indices_sha256": hashlib.sha256(
            indices.astype("<i8").tobytes()
        ).hexdigest(),
        "sap_quality": "==0",
        "flatten_mask": "==0",
        "additional_preparation": (
            "production relative_curve normalization/upper clipping/dedup"
        ),
    }
    return result, curve, kept


def proposal_exposure(row, curve, kept):
    ephemeris = family_ephemeris(row)
    period, epoch = ephemeris["period_days"], ephemeris["t0_bkjd"]
    distance = np.abs((curve["time"] - epoch + period / 2) % period - period / 2)
    in_window = distance <= ephemeris["duration_hours"] / 48
    windows = []
    for quarter in (0, 1):
        raw = in_window & (curve["quarter"] == quarter)
        prepared = raw & kept
        event_numbers = np.rint((curve["time"][prepared] - epoch) / period).astype(
            np.int64
        )
        windows.append(
            {
                "quarter": quarter,
                "raw_window_cadences": int(raw.sum()),
                "prepared_window_cadences": int(prepared.sum()),
                "distinct_prepared_windows": len(np.unique(event_numbers)),
            }
        )
    return {
        "original_proposal": row,
        "family_association": ephemeris,
        "q0_q1_mask_exposure": windows,
        "physical_recovery_measured": False,
    }


def guide_witness(source, payload, headers, proposals):
    from pipeline.deterministic.cloud_public import fits_arrays

    require(
        digest(payload) == source["pool"]["raw_fits_sha256"],
        "Original raw FITS hash changed",
    )
    quarters = header_identity(source, headers)
    raw = fits_arrays(
        payload, source["source_id"], {"product_kic": source["product"]["kic"]}
    )
    exposure, curve, kept = mask_exposure(raw)
    family = [r for r in proposals if r["source_id"] == source["source_id"]]
    require(bool(family), "Guide source has no original family proposal")
    available = {r["quarter"] for r in quarters}
    return {
        "source_id": source["source_id"],
        "original_raw_input": reference(payload),
        "raw_hash_verified": True,
        "q0_available": 0 in available,
        "q1_available": 1 in available,
        "quarter_witnesses": quarters,
        "preparation": exposure,
        "original_family_proposals": [
            proposal_exposure(r, curve, kept) for r in family
        ],
        "eligibility_status": "PARTIAL_Q0_Q1_WITNESS_QUARTER_PARENT_UNRESOLVED",
        "boundary": BOUNDARY,
    }


def summarize(rows, donors):
    guides = [r for r in rows if r["cohort"] == "guide_star_family"]
    successful = [r["witness"] for r in guides if "witness" in r]
    return {
        "attempted_sources": len(rows),
        "source_statuses": dict(Counter(r["status"] for r in rows)),
        "guide_selected": len(guides),
        "guide_full_raw_hash_verified": len(successful),
        "guide_q0_available": sum(r["q0_available"] for r in successful),
        "guide_q1_available": sum(r["q1_available"] for r in successful),
        "guide_preparation_statuses": dict(
            Counter(r["preparation"]["status"] for r in successful)
        ),
        "guide_family_proposals_bound": sum(
            len(r["original_family_proposals"]) for r in successful
        ),
        "guide_actual_quarter_parents_verified": sum(
            q["quarter_parent"] is not None
            for r in successful
            for q in r["quarter_witnesses"]
        ),
        "contamination_selected": len(donors),
        "contamination_catalogue_parent_bound": sum(
            r["catalogue_parent_binding"] for r in donors
        ),
        "contamination_public_donor_product_bound": sum(
            bool(r["public_donor_product_identities"]) for r in donors
        ),
        "contamination_attributed": 0,
    }


def assemble(plan_path, directory):
    from pipeline.deterministic.control_eligibility_archive import (
        local_headers,
        verify_headers,
    )

    plan = read_json(plan_path)
    verify_plan(plan)
    directory = Path(directory)
    rows = []
    for source in plan["sources"]:
        record_path = directory / "records" / f"{source['source_id']}.json"
        record = read_json(record_path)
        require(record["source"] == source, "Archive record source binding changed")
        row = {
            **source,
            "archive_record": reference(record_path),
            "status": record["status"],
        }
        if record["status"] == "VERIFIED_FULL_PRODUCT":
            payload = verified_reference(record["payload"])
            require(
                local_headers(payload) == record["headers"],
                "Retained full-product header records changed",
            )
            row["witness"] = guide_witness(
                source, payload, record["headers"], plan["original_family_proposals"]
            )
        elif record["status"] == "VERIFIED_ARCHIVE_HEADERS":
            verify_headers(record["headers"], record["archive_product_bytes"])
            row["quarter_witnesses"] = header_identity(source, record["headers"])
            row["original_raw_hash_reverified"] = False
        rows.append(row)
    report = {
        "schema": 1,
        "plan": reference(plan_path),
        "inputs": plan["inputs"],
        "sources": rows,
        "donor_bindings": plan["donor_bindings"],
        "counts": summarize(rows, plan["donor_bindings"]),
        "implementation": plan["implementation"],
        "boundary": BOUNDARY,
    }
    save(directory / "eligibility-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    for key in ("reference-dir", "catalogue", "pool", "out"):
        plan.add_argument(f"--{key}", type=Path, required=True)
    for command in ("collect", "audit"):
        stage = sub.add_parser(command)
        stage.add_argument("--plan", type=Path, required=True)
        stage.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        save(args.out, build_plan(args.reference_dir, args.catalogue, args.pool))
    elif args.command == "collect":
        from pipeline.deterministic.control_eligibility_archive import collect

        collect(args.plan, args.directory)
    else:
        print(json.dumps(assemble(args.plan, args.directory)["counts"], indent=2))


if __name__ == "__main__":
    main()
