"""Bind catalogued counterpart product headers, without reading donor flux."""

import argparse
import time
from collections import Counter
from pathlib import Path

import requests

from pipeline.deterministic.control_bindings import (
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_eligibility import (
    BOUNDARY,
    header_identity,
    implementation,
    product_identity,
    save,
    verify_plan,
)
from pipeline.deterministic.control_eligibility_archive import (
    capacity,
    range_request,
    remote_headers,
    verify_headers,
)


def donor_product(binding):
    parent = int(binding["parent_tpf"])
    candidates = binding["public_donor_product_identities"]
    require(len(candidates) <= 1, "Ambiguous donor product identity")
    product = {
        "kic": parent,
        "fname": f"kic-{parent:09d}",
        "gaia_designation": candidates[0]["gaia_designation"]
        if candidates
        else "UNKNOWN",
    }
    return {"product": product, **product_identity(product)}


def bind_donor_header(binding, headers):
    cards = headers[0]["cards"]
    require(
        int(cards["KEPLERID"]) == int(binding["parent_tpf"]),
        "Donor KIC identity differs",
    )
    sid = str(cards["GAIAID"]).split()[-1]
    require(sid.isdecimal() and int(sid) > 0, "Donor Gaia identity missing")
    candidates = binding["public_donor_product_identities"]
    if candidates:
        require(
            sid == candidates[0]["gaia_designation"].split()[-1],
            "Donor Gaia catalogue/header differs",
        )
    require(sid != binding["source_id"], "Target and donor are the same source")
    quarters = [h for h in headers if h["name"].startswith("LIGHTCURVE_Q")]
    return {
        "catalogued_kic": binding["parent_tpf"],
        "archive_gaia_source_id": sid,
        "public_catalogue_gaia_match": bool(candidates),
        "global_extraction_parent": str(cards["TPFORG"]),
        "quarters": [
            {
                "quarter": h["cards"]["QUARTER"],
                "quarter_parent": h["cards"].get("TPFORG"),
                "archive_header_sha256": h["sha256"],
            }
            for h in quarters
        ],
        "donor_raw_hash_reverified": False,
        "donor_flux_read": False,
        "contamination_attributed": False,
    }


def collect_one(session, binding, root):
    result = {"binding": binding, "attempts": [], "status": "UNAVAILABLE"}
    started = time.monotonic()
    try:
        capacity(root)
        source = donor_product(binding)
        first, total = range_request(session, source, 0, 5759, result["attempts"])
        result["archive_product_bytes"] = total
        headers = remote_headers(session, source, first, total, result["attempts"])
        result["headers"] = headers
        verify_headers(headers, total)
        result["donor"] = bind_donor_header(binding, headers)
        result["status"] = "VERIFIED_DONOR_ARCHIVE_HEADERS"
    except (ValueError, OSError, KeyError, requests.RequestException) as error:
        result.update(error=type(error).__name__, message=str(error))
    result["wall_seconds"] = time.monotonic() - started
    return result


def quarter_association(target, donor):
    target_quarters = {q["quarter"]: q for q in target.get("quarter_witnesses", [])}
    rows = []
    for header in donor.get("headers", []):
        if not header["name"].startswith("LIGHTCURVE_Q"):
            continue
        cards = header["cards"]
        quarter = cards["QUARTER"]
        if quarter not in target_quarters:
            continue
        target_header = target_quarters[quarter]["archive_header"]
        rows.append(
            {
                "quarter": quarter,
                "target_header_sha256": target_header["sha256"],
                "donor_header_sha256": header["sha256"],
                "target_module_output_channel": [
                    target_header["cards"].get(k)
                    for k in ("MODULE", "OUTPUT", "CHANNEL")
                ],
                "donor_module_output_channel": [
                    cards.get(k) for k in ("MODULE", "OUTPUT", "CHANNEL")
                ],
                "target_quarter_parent": target_quarters[quarter]["quarter_parent"],
                "donor_quarter_parent": cards.get("TPFORG"),
                "attribution_measured": False,
            }
        )
    return rows


def verify_targets(report, plan, plan_path):
    require(
        report["plan"] == reference(plan_path), "Target report and donor plan differ"
    )
    require(report["boundary"] == BOUNDARY, "Target report acceptance boundary changed")
    sources = plan["sources"]
    require(
        [r["source_id"] for r in report["sources"]]
        == [r["source_id"] for r in sources],
        "Target source inventory or order changed",
    )
    for declared, row in zip(sources, report["sources"]):
        record = read_json(verified_reference(row["archive_record"]))
        require(record["source"] == declared, "Target archive source binding changed")
        if record["status"] == "UNAVAILABLE":
            continue
        quarters = header_identity(declared, record["headers"])
        observed = row.get(
            "quarter_witnesses", row.get("witness", {}).get("quarter_witnesses")
        )
        require(observed == quarters, "Target report quarter witnesses changed")
        if "payload" in record:
            verified_reference(record["payload"])


def collect(plan_path, target_report_path, directory):
    plan = read_json(plan_path)
    verify_plan(plan)
    target_report = read_json(target_report_path)
    verify_targets(target_report, plan, plan_path)
    targets = {r["source_id"]: r for r in target_report["sources"]}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with requests.Session() as session:
        for index, binding in enumerate(plan["donor_bindings"], 1):
            path = directory / "records" / f"{binding['source_id']}.json"
            if path.exists():
                record = read_json(path)
                require(record["binding"] == binding, "Retained donor identity changed")
            else:
                record = collect_one(session, binding, directory.parent)
                save(path, record)
            rows.append(
                {
                    "target_source_id": binding["source_id"],
                    "status": record["status"],
                    "record": reference(path),
                    "donor": record.get("donor"),
                    "target_donor_quarters": quarter_association(
                        targets[binding["source_id"]], record
                    ),
                }
            )
            if index % 10 == 0:
                print(
                    f"Retained {index}/100 fixed counterpart header attempts",
                    flush=True,
                )
    code = implementation()
    code["pipeline/deterministic/control_donor_headers.py"] = reference(Path(__file__))[
        "sha256"
    ]
    report = {
        "schema": 1,
        "plan": reference(plan_path),
        "target_report": reference(target_report_path),
        "donors": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "paired_archive_quarters": sum(len(r["target_donor_quarters"]) for r in rows),
        "implementation": code,
        "boundary": BOUNDARY,
    }
    save(directory / "donor-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "target-report", "directory"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    collect(args.plan, args.target_report, args.directory)


if __name__ == "__main__":
    main()
