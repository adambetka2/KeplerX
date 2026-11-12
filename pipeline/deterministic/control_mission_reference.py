"""Bind explicit mission-SAP donor references to original catalogue identities."""

import argparse
import math
import re
from pathlib import Path

import numpy as np

from pipeline.deterministic.control_bindings import (
    protected_sources,
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_eligibility import catalogue_parent, save
from pipeline.deterministic.control_eligibility_archive import (
    header_record,
    local_headers,
    next_header,
)

DOMAIN = "KEPLER_STANDARD_LC_SAP_REFERENCE_NOT_KBONUS_PSF"
READER = {
    "id": "kepler-mission-sap-reference-v1",
    "domain": DOMAIN,
    "view": "MISSION_SAP_REFERENCE",
    "use": "DONOR_SIGNAL_REFERENCE_ONLY",
    "time": "TIME + BJDREFI + BJDREFF; TDB; days",
    "quality": "SAP_QUALITY retained; usable reference cadences require zero",
    "flatten_mask": "ABSENT_NOT_SYNTHESIZED",
    "normalization": "NONE",
    "kbonus_view_substitution": False,
}
GAIA_ENDPOINT = "https://gea.esac.esa.int/tap-server/tap/sync"


def catalogue_identity(receipts, kic):
    require(
        set(receipts) == {"tic", "dr3_to_dr2", "dr2_to_dr3"},
        "Mission reference requires exact TIC and reciprocal Gaia receipts",
    )
    values = {key: read_json(verified_reference(row)) for key, row in receipts.items()}
    tic = values["tic"]
    response = read_json(verified_reference(tic["response"]))
    require(
        tic["canonical_url"] == "https://mast.stsci.edu/api/v0/invoke"
        and response["status"] == "COMPLETE"
        and response["data"] == tic["rows"]
        and response["paging"] == tic["paging"]
        and response["fields"] == tic["fields"]
        and tic["service_status"] == "COMPLETE"
        and tic["http_status"] == 200
        and tic["request"]["service"] == "Mast.Catalogs.Filtered.Tic"
        and tic["paging"]["pagesFiltered"] == 1
        and tic["paging"]["rowsFiltered"] == len(tic["rows"]),
        "Incomplete exact KIC/TIC identity receipt",
    )
    rows = [row for row in tic["rows"] if str(row["KIC"]) == str(kic)]
    require(
        len(rows) == 1
        and rows[0].get("duplicate_id") in {None, ""}
        and rows[0].get("disposition") not in {"DUPLICATE", "ARTIFACT"},
        "KIC/TIC identity is missing, duplicated or an artifact",
    )
    row = rows[0]
    dr2 = str(row["GAIA"])
    require(dr2.isdecimal() and int(dr2) > 0, "TIC Gaia DR2 identity missing")
    for key in ("dr3_to_dr2", "dr2_to_dr3"):
        require(
            values[key]["status"] == "COMPLETE"
            and values[key]["endpoint"] == GAIA_ENDPOINT,
            "Incomplete reciprocal Gaia catalogue receipt",
        )
    reverse = values["dr2_to_dr3"]["rows"]
    candidates = {
        str(r["dr3_source_id"]) for r in reverse if str(r["dr2_source_id"]) == dr2
    }
    require(len(candidates) == 1, "Gaia donor identity is not strictly reciprocal")
    sid = next(iter(candidates))
    forward = {
        str(r["dr2_source_id"])
        for r in values["dr3_to_dr2"]["rows"]
        if str(r["dr3_source_id"]) == sid
    }
    require(forward == {dr2}, "Gaia donor identity is not strictly reciprocal")
    return {
        "kic": str(kic),
        "tic": str(row["ID"]),
        "tic_version": row["version"],
        "gaia_dr2": dr2,
        "gaia_dr3": sid,
    }


def mission_header(record, kic):
    require(
        record["status"] == "CANONICAL_KEPLER_KIC_QUARTER_AND_SAP_HEADER_VERIFIED"
        and record["reference_domain"] == DOMAIN
        and record["catalogued_parent_kic"] == str(kic),
        "Mission reference header scope or KIC changed",
    )
    headers = record["headers"]
    require(len(headers) == 2, "Mission primary/table header inventory changed")
    for header in headers:
        require(
            header_record(header["raw_header_ascii"].encode("ascii"), header["offset"])
            == header,
            "Mission reference header content or hash changed",
        )
    primary, table = (h["cards"] for h in headers)
    total = record["archive_product_bytes"]
    require(
        headers[0]["offset"] == 0
        and next_header(headers[0], total) == headers[1]["offset"],
        "Mission header offsets changed",
    )
    next_header(headers[1], total)
    require(
        primary["MISSION"] == "Kepler"
        and primary["OBSMODE"] == "long cadence"
        and str(primary["KEPLERID"]) == str(kic)
        and "GAIAID" not in primary
        and table["EXTNAME"] == "LIGHTCURVE"
        and table["XTENSION"] == "BINTABLE",
        "Mission reference is not the declared native Kepler product",
    )
    columns = [table[f"TTYPE{i}"] for i in range(1, table["TFIELDS"] + 1)]
    require(
        len(columns) == len(set(columns))
        and {"TIME", "CADENCENO", "SAP_FLUX", "SAP_FLUX_ERR", "SAP_QUALITY"}
        <= set(columns)
        and "FLATTEN_MASK" not in columns,
        "Mission SAP columns or absent flatten-mask contract changed",
    )
    require(
        table["BJDREFI"] == 2454833
        and table["BJDREFF"] == 0.0
        and table["TIMESYS"] == "TDB"
        and table["TIMEREF"] == "SOLARSYSTEM"
        and table["TIMEUNIT"] == "d"
        and math.isfinite(table["TIMEDEL"])
        and table["TIMEDEL"] > 0,
        "Mission SAP time reference changed",
    )
    quarter = primary["QUARTER"]
    require(type(quarter) is int and 0 <= quarter <= 17, "Mission quarter changed")
    identity = f"{int(kic):09d}"
    pattern = (
        rf"https://archive\.stsci\.edu/pub/kepler/lightcurves/{identity[:4]}/"
        rf"{identity}/kplr{identity}-\d{{13}}_llc\.fits"
    )
    require(re.fullmatch(pattern, record["url"]), "Mission product URL/KIC changed")
    return {"url": record["url"], "quarter": quarter, "bytes": total}


def bind_reference(binding, original_donor, receipts, header, protected):
    prior = read_json(verified_reference(original_donor))
    require(
        prior["binding"] == binding and prior["status"] == "UNAVAILABLE",
        "Original unavailable KBonus donor binding changed",
    )
    kic = binding["parent_tpf"]
    require(
        any(
            catalogue_parent(row) == kic
            for c in binding["catalogues"]
            for row in c["parent_rows"]
        ),
        "Original catalogue parent identity is missing",
    )
    identity = catalogue_identity(receipts, kic)
    sid = identity["gaia_dr3"]
    require(
        sid != binding["source_id"]
        and sid not in protected
        and binding["source_id"] not in protected,
        "Mission donor/target identity is shared or protected",
    )
    record = read_json(verified_reference(header))
    require(
        record["source_target_id"] == binding["source_id"],
        "Mission target identity changed",
    )
    return {
        "schema": 1,
        "reader_contract": dict(READER),
        "identity": {"target_source_id": binding["source_id"], **identity},
        "catalogue_bindings": binding["catalogues"],
        "inputs": {
            "original_donor": original_donor,
            "identity_receipts": receipts,
            "header": header,
        },
        "mission_product": mission_header(record, kic),
        "original_kbonus_donor_status": "UNAVAILABLE",
        "full_product_hash_provisioned": False,
        "physical_attribution_measured": False,
        "final_scoped_role_authorization": False,
    }


def read_reference(contract, payload, protected, protected_hashes):
    from astropy.io import fits

    inputs = contract["inputs"]
    prior = read_json(verified_reference(inputs["original_donor"]))
    expected = bind_reference(
        prior["binding"],
        inputs["original_donor"],
        inputs["identity_receipts"],
        inputs["header"],
        protected,
    )
    require(contract == expected, "Mission SAP reference contract changed")
    require(payload["sha256"] not in protected_hashes, "Protected mission raw input")
    path = verified_reference(payload)
    require(
        path.stat().st_size == contract["mission_product"]["bytes"],
        "Mission payload size changed",
    )
    headers = local_headers(path)
    original = read_json(verified_reference(inputs["header"]))
    require(
        headers[:2] == original["headers"], "Mission payload/header identity changed"
    )
    with fits.open(path, memmap=False) as hdus:
        data = hdus[1].data
        arrays = {
            name: np.array(data[column], copy=True)
            for name, column in {
                "time_bkjd": "TIME",
                "cadenceno": "CADENCENO",
                "sap_flux": "SAP_FLUX",
                "sap_flux_err": "SAP_FLUX_ERR",
                "sap_quality": "SAP_QUALITY",
            }.items()
        }
    usable = (
        np.isfinite(arrays["time_bkjd"])
        & np.isfinite(arrays["sap_flux"])
        & np.isfinite(arrays["sap_flux_err"])
        & (arrays["sap_flux_err"] > 0)
        & (arrays["sap_quality"] == 0)
    )
    return {
        "domain": DOMAIN,
        "view": READER["view"],
        "arrays": arrays,
        "usable_reference": usable,
        "payload": reference(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "donors", "identity-report", "mission-headers", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Mission reference output exists; retain it")
    plan, donors, identities, headers = (
        read_json(path)
        for path in (args.plan, args.donors, args.identity_report, args.mission_headers)
    )
    require(
        donors["plan"] == reference(args.plan)
        and [r["target_source_id"] for r in donors["donors"]]
        == [r["source_id"] for r in plan["donor_bindings"]],
        "Original donor report plan, order or denominator changed",
    )
    protected = protected_sources(
        read_json(verified_reference(plan["inputs"]["sealed_identity_exclusions"]))
    )
    query_refs = {Path(r["path"]).name: r for r in identities["queries"]}
    receipts = {
        "tic": identities["inputs"]["tic"],
        "dr3_to_dr2": query_refs["stellar-dr3-1000.json"],
        "dr2_to_dr3": query_refs["stellar-dr2-1000.json"],
    }
    donor_index = {r["target_source_id"]: r for r in donors["donors"]}
    products = [r["record"] for r in headers["products"]]
    contracts, unavailable = [], []
    for binding in plan["donor_bindings"]:
        sid = binding["source_id"]
        previous = donor_index[sid]
        if previous["status"] != "UNAVAILABLE":
            continue
        try:
            catalogue_identity(receipts, binding["parent_tpf"])
        except ValueError as error:
            unavailable.append(
                {"source_id": sid, "kic": binding["parent_tpf"], "reason": str(error)}
            )
            continue
        for ref in products:
            record = read_json(verified_reference(ref))
            if record["source_target_id"] == sid:
                contracts.append(
                    bind_reference(
                        binding, previous["record"], receipts, ref, protected
                    )
                )
    save(
        args.out,
        {
            "schema": 1,
            "status": "EXPLICIT_MISSION_SAP_REFERENCE_CONTRACTS_ONLY",
            "inputs": {
                key: reference(getattr(args, key.replace("-", "_")))
                for key in ("plan", "donors", "identity-report", "mission-headers")
            },
            "contracts": contracts,
            "unavailable": unavailable,
            "implementation": reference(Path(__file__)),
            "flux_read": False,
            "scientific_acceptance": False,
            "cohort_frozen": False,
        },
    )


if __name__ == "__main__":
    main()
