"""Materialize frozen KOI candidate and false-positive KBonus controls."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic.catalogue_control_partition import parent_map_identities
from deterministic.run import InsufficientCadences, prepare_curve_records
from runtime import worker_count

POOLS = ROOT / "dump/release-controls-v2/catalogue-pools.csv"
CATALOGUE = ROOT / "discPCIe/data/reference-catalogues/q1_q17_dr25_koi-rows.json"
CONFIG = Path(__file__).with_name("config.json")
MINIMUM_FREE_BYTES = 20 * 1024**3
CATEGORIES = {
    "catalogue_candidate": "CANDIDATE",
    "catalogue_false_positive": "FALSE POSITIVE",
}
STITCHED_COLUMNS = {
    "time_bkjd": "TIME",
    "cadenceno": "CADENCENO",
    "quarter": "QUARTER",
    "psf_stitched_flux": "FLUX",
    "psf_stitched_flux_err": "FLUX_ERR",
    "sap_stitched_flux": "SAP_FLUX",
    "sap_stitched_flux_err": "SAP_FLUX_ERR",
    "psf_flat_kbonus_flux": "PSF_FLAT_FLUX",
    "psf_flat_kbonus_flux_err": "PSF_FLAT_FLUX_ERR",
    "sap_quality": "SAP_QUALITY",
    "flatten_mask": "FLATTEN_MASK",
}
QUARTER_COLUMNS = {
    "qdiag_time_bkjd": "TIME",
    "qdiag_cadenceno": "CADENCENO",
    "qdiag_psf_flux": "FLUX",
    "qdiag_psf_flux_err": "FLUX_ERR",
    "qdiag_sap_flux": "SAP_FLUX",
    "qdiag_sap_flux_err": "SAP_FLUX_ERR",
    "qdiag_psf_flux_nova": "PSF_FLUX_NOVA",
    "qdiag_psf_flux_err_nova": "PSF_FLUX_ERR_NOVA",
    "qdiag_sap_quality": "SAP_QUALITY",
}
SIGNAL_FIELDS = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_period",
    "koi_period_err1",
    "koi_period_err2",
    "koi_time0bk",
    "koi_time0bk_err1",
    "koi_time0bk_err2",
    "koi_duration",
    "koi_depth",
    "ra",
    "dec",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def product_uri(kic):
    identity = f"{int(kic):09d}"
    return (
        f"mast:HLSP/kbonus-bkg/lcs/{identity[:4]}/{identity}/"
        f"hlsp_kbonus-bkg_kepler_kepler_kic-{identity}_kepler_v1.0_lc.fits"
    )


def product_url(kic):
    uri = product_uri(kic)
    return "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + urllib.parse.quote(
        uri, safe=""
    )


def frozen_pools(path=POOLS):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for category, disposition in CATEGORIES.items():
        selected = [row for row in rows if row["category"] == category]
        selected.sort(key=lambda row: int(row["pool_rank"]))
        if (
            not selected
            or [int(row["pool_rank"]) for row in selected]
            != list(range(1, len(selected) + 1))
            or any(int(row["pool_count"]) != len(selected) for row in selected)
            or any(int(row["target_count"]) >= len(selected) for row in selected)
        ):
            raise ValueError(f"Malformed frozen catalogue pool: {category}")
        result[category] = {
            "disposition": disposition,
            "target": int(selected[0]["target_count"]),
            "rows": selected,
        }
    if sum(len(value["rows"]) for value in result.values()) != len(rows):
        raise ValueError("Frozen catalogue pool contains an unknown category")
    return result


def catalogue_signals(path=CATALOGUE):
    grouped = defaultdict(list)
    for row in json.loads(Path(path).read_text()):
        if row.get("kepid") is not None:
            grouped[int(row["kepid"])].append(
                {name: row.get(name) for name in SIGNAL_FIELDS}
            )
    return grouped


def download(path, kic, timeout, attempts=3):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    error = None
    for attempt in range(attempts):
        try:
            with requests.get(
                product_url(kic), timeout=timeout, stream=True
            ) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            temporary.replace(path)
            return path
        except (requests.RequestException, OSError) as exc:
            error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"KBonus download failed for KIC {kic}: {error}")


def download_pool_row(out, category, timeout, row):
    return download(
        out / "downloads" / category / f"kic-{int(row['kic_id']):09d}.fits",
        int(row["kic_id"]),
        timeout,
    )


def aligned_quarter_arrays(stitched_cadences, quarters):
    cadences = np.concatenate([quarter["CADENCENO"] for quarter in quarters])
    if len(cadences) != len(np.unique(cadences)):
        raise ValueError("Quarter light curves contain duplicate cadences")
    order = np.argsort(cadences)
    cadences = cadences[order]
    positions = np.searchsorted(cadences, stitched_cadences)
    if np.any(positions == len(cadences)) or not np.array_equal(
        cadences[positions], stitched_cadences
    ):
        raise ValueError("Stitched cadences are absent from quarter light curves")
    arrays = {
        name: np.concatenate([quarter[column] for quarter in quarters])[order][
            positions
        ]
        for name, column in QUARTER_COLUMNS.items()
    }
    quarter_numbers = np.concatenate(
        [np.full(len(quarter["CADENCENO"]), quarter["quarter"]) for quarter in quarters]
    )
    arrays["qdiag_quarter"] = quarter_numbers[order][positions]
    return arrays


def read_product(path, expected_kic):
    with fits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        if int(header["KEPLERID"]) != int(expected_kic):
            raise ValueError(f"KIC identity mismatch: {path}")
        source_id = str(header["GAIAID"]).split()[-1]
        if not source_id.isdigit() or int(source_id) <= 0:
            raise ValueError(f"Gaia identity unavailable: {path}")
        stitched = hdus["LIGHTCURVE_STITCHED"].data
        arrays = {
            name: np.array(stitched[column])
            for name, column in STITCHED_COLUMNS.items()
        }
        quarter_hdus = sorted(
            (hdu for hdu in hdus if hdu.name.startswith("LIGHTCURVE_Q")),
            key=lambda hdu: int(hdu.name.split("Q")[-1]),
        )
        quarters = [
            {
                **{
                    column: np.array(hdu.data[column])
                    for column in QUARTER_COLUMNS.values()
                },
                "quarter": int(hdu.name.split("Q")[-1]),
            }
            for hdu in quarter_hdus
        ]
        arrays.update(aligned_quarter_arrays(arrays["cadenceno"], quarters))
        metadata = {
            "source_id": source_id,
            "kic": int(expected_kic),
            "origin_tpf_kic": int(expected_kic),
            "ra_deg": float(header["RA_OBJ"]),
            "dec_deg": float(header["DEC_OBJ"]),
            "phot_g_mean_mag": (
                None if header.get("GMAG") is None else float(header["GMAG"])
            ),
            "raw_fits_sha256": digest(path),
            "shard": 0,
            "shard_path": product_uri(expected_kic),
        }
    return metadata, arrays


def discovery_eligible(arrays):
    curve = {
        "time": arrays["time_bkjd"],
        "cadence": arrays["cadenceno"],
        "quarter": arrays["quarter"],
        "flux": arrays["psf_flat_kbonus_flux"],
        "error": arrays["psf_flat_kbonus_flux_err"],
        "quality": arrays["sap_quality"],
        "flatten": arrays["flatten_mask"],
    }
    try:
        prepare_curve_records(curve, mask_flatten=True)
    except InsufficientCadences:
        return False
    return True


def write_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def materialize_one(out, category, pool_row, signals, download_path, exclusions=None):
    kic = int(pool_row["kic_id"])
    record = {
        "category": category,
        "pool_rank": int(pool_row["pool_rank"]),
        "kic_id": kic,
    }
    if download_path is None:
        return {**record, "status": "PRODUCT_UNAVAILABLE"}
    metadata, arrays = read_product(download_path, kic)
    exclusions = exclusions or {}
    identities = {
        "source_id": str(metadata["source_id"]),
        "kic_id": str(metadata["origin_tpf_kic"]),
        "fits_sha256": str(metadata["raw_fits_sha256"]),
    }
    if any(value in exclusions.get(key, set()) for key, value in identities.items()):
        return {**record, **identities, "status": "EXCLUDED_OPENED_OR_CONTROL_IDENTITY"}
    if not discovery_eligible(arrays):
        return {
            **record,
            "status": "INSUFFICIENT_DISCOVERY_CADENCES",
            "fits_sha256": metadata["raw_fits_sha256"],
        }
    path = out / "arrays" / f"{metadata['source_id']}.npz"
    write_arrays(path, arrays)
    return {
        **record,
        "status": "ELIGIBLE",
        "source_id": metadata["source_id"],
        "metadata": metadata,
        "arrays": str(path.relative_to(out)),
        "arrays_sha256": digest(path),
        "fits_sha256": metadata["raw_fits_sha256"],
        "catalogue_signals": signals[kic],
    }


def selected_categories(category):
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"Unknown catalogue category: {category}")
    return [category] if category else list(CATEGORIES)


def run_spec(limit_per_category, pools_path=POOLS, parent_maps=(), category=None):
    paths = [
        Path(pools_path),
        CATALOGUE,
        CONFIG,
        Path(__file__),
        Path(parent_map_identities.__code__.co_filename),
        Path(prepare_curve_records.__code__.co_filename),
        ROOT / "pipeline/photometry/lc2.py",
        *map(Path, parent_maps),
    ]
    inputs = {
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): digest(path)
        for path in paths
    }
    return {
        "schema": 1,
        "protocol": "fsc-kepler-catalogue-controls-v2",
        "mode": "PROBE" if limit_per_category is not None else "FROZEN",
        "limit_per_category": limit_per_category,
        "categories": selected_categories(category),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "inputs": inputs,
        "excluded_parent_maps": [str(path) for path in parent_maps],
        "selection": "First usable KBonus product in frozen pool-rank order for each category, excluding every declared source, parent and raw input identity",
        "claim_boundary": "Catalogue candidates are real-signal regression controls, not confirmed planets. Catalogue false positives retain their catalogue labels; denial still requires a reproduced mechanism.",
    }


def read_eligibility(path):
    if not path.exists():
        return []
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    identities = [(row["category"], row["pool_rank"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Eligibility ledger contains duplicate pool identities")
    return rows


def append_eligibility(path, rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def completed_by_category(rows):
    result = defaultdict(list)
    for row in rows:
        if row["status"] == "ELIGIBLE":
            result[row["category"]].append(row)
    for values in result.values():
        values.sort(key=lambda row: row["pool_rank"])
    return result


def validate_truth(pools, signals):
    for data in pools.values():
        for row in data["rows"]:
            kic = int(row["kic_id"])
            if not signals[kic] or any(
                signal["koi_disposition"] != data["disposition"]
                for signal in signals[kic]
            ):
                raise ValueError(f"Frozen catalogue truth changed for KIC {kic}")


def materialize_category(
    out,
    category,
    data,
    signals,
    target,
    ledger_path,
    ledger,
    workers,
    batch_size,
    timeout,
    exclusions=None,
):
    done = {(row["category"], row["pool_rank"]) for row in ledger}
    while len(completed_by_category(ledger)[category]) < target:
        pending = [
            row for row in data["rows"] if (category, int(row["pool_rank"])) not in done
        ]
        if len(completed_by_category(ledger)[category]) + len(pending) < target:
            raise ValueError(f"Frozen pool exhausted before {category} target")
        pending = pending[:batch_size]
        if shutil.disk_usage(out).free < MINIMUM_FREE_BYTES:
            raise RuntimeError(
                "Catalogue materialization stopped: less than 20 GiB free"
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(
                pool.map(
                    partial(download_pool_row, out, category, timeout),
                    pending,
                )
            )
        records = [
            materialize_one(out, category, row, signals, path, exclusions)
            for row, path in zip(pending, paths)
        ]
        append_eligibility(ledger_path, records)
        ledger.extend(records)
        done.update((category, row["pool_rank"]) for row in records)
        print(
            json.dumps(
                {
                    "category": category,
                    "eligible": len(completed_by_category(ledger)[category]),
                    "target": target,
                    "attempted": sum(row["category"] == category for row in ledger),
                }
            ),
            flush=True,
        )


def freeze_outputs(out, spec, rows, targets):
    if any(
        digest(ROOT / path) != expected for path, expected in spec["inputs"].items()
    ):
        raise ValueError("Catalogue materialization inputs changed before freeze")
    eligible = completed_by_category(rows)
    selected = {
        category: eligible[category][:target] for category, target in targets.items()
    }
    if any(len(selected[category]) != target for category, target in targets.items()):
        raise ValueError("Catalogue control targets are incomplete")
    source_rows = [row for values in selected.values() for row in values]
    source_ids = [row["source_id"] for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Catalogue control Gaia source identities overlap")
    manifest = {
        "schema": 1,
        "measurement_scope": spec["claim_boundary"],
        "sources": {
            row["source_id"]: {
                "arrays": row["arrays"],
                "sha256": row["arrays_sha256"],
                "metadata": row["metadata"],
            }
            for row in source_rows
        },
    }
    save(out / "input-manifest.json", manifest)
    save(out / "source-ids.json", source_ids)
    for category, values in selected.items():
        save(
            out / f"{category}-source-ids.json",
            [row["source_id"] for row in values],
        )
    truth_path = out / "truth.jsonl"
    with truth_path.open("x") as stream:
        for row in source_rows:
            stream.write(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "category",
                            "pool_rank",
                            "kic_id",
                            "source_id",
                            "fits_sha256",
                            "catalogue_signals",
                        )
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    outputs = [
        out / "input-manifest.json",
        out / "source-ids.json",
        out / "truth.jsonl",
        *(out / f"{category}-source-ids.json" for category in selected),
    ]
    status = (
        "FROZEN_INPUTS"
        if set(selected) == set(CATEGORIES)
        else "FROZEN_CATEGORY_INPUTS"
    )
    freeze = {
        **spec,
        "status": "PROBE_COMPLETE" if spec["mode"] == "PROBE" else status,
        "targets": targets,
        "selected": {category: len(values) for category, values in selected.items()},
        "eligibility_counts": dict(Counter(row["status"] for row in rows)),
        "artifacts": {str(path.relative_to(out)): digest(path) for path in outputs},
    }
    save(out / "freeze.json", freeze)
    return freeze


def materialize(
    out,
    resume,
    limit_per_category,
    workers,
    batch_size,
    timeout,
    pools_path=POOLS,
    parent_maps=(),
    requested_category=None,
):
    out = Path(out)
    spec = run_spec(limit_per_category, pools_path, parent_maps, requested_category)
    spec_path = out / "run-spec.json"
    if out.exists() and not resume:
        raise ValueError("Output exists; use --resume or a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("Catalogue-control run identity changed")
    else:
        save(spec_path, spec)
    if (out / "freeze.json").exists():
        return json.loads((out / "freeze.json").read_text())
    pools = {
        name: data
        for name, data in frozen_pools(pools_path).items()
        if name in spec["categories"]
    }
    exclusions = {"source_id": set(), "kic_id": set(), "fits_sha256": set()}
    for path in parent_maps:
        identities, _ = parent_map_identities(path)
        for key, values in identities.items():
            exclusions[key].update(values)
    signals = catalogue_signals()
    targets = {
        category: min(data["target"], limit_per_category or data["target"])
        for category, data in pools.items()
    }
    validate_truth(pools, signals)
    ledger_path = out / "eligibility.jsonl"
    ledger = read_eligibility(ledger_path)
    done = {(row["category"], row["pool_rank"]) for row in ledger}
    excluded = [
        {
            "category": category,
            "pool_rank": int(row["pool_rank"]),
            "kic_id": int(row["kic_id"]),
            "status": "EXCLUDED_PARENT_IDENTITY",
        }
        for category, data in pools.items()
        for row in data["rows"]
        if str(row["kic_id"]) in exclusions["kic_id"]
        and (category, int(row["pool_rank"])) not in done
    ]
    append_eligibility(ledger_path, excluded)
    ledger.extend(excluded)
    for category, data in pools.items():
        materialize_category(
            out,
            category,
            data,
            signals,
            targets[category],
            ledger_path,
            ledger,
            workers,
            batch_size,
            timeout,
            exclusions,
        )
    return freeze_outputs(out, spec, ledger, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", type=Path, default=POOLS)
    parser.add_argument("--exclude-parent-map", type=Path, action="append", default=[])
    parser.add_argument("--category", choices=list(CATEGORIES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--workers", type=int, default=worker_count())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.timeout) < 1 or (
        args.limit_per_category is not None and args.limit_per_category < 1
    ):
        raise ValueError("Limits, workers, batch size and timeout must be positive")
    report = materialize(
        ROOT / args.out,
        args.resume,
        args.limit_per_category,
        args.workers,
        args.batch_size,
        args.timeout,
        ROOT / args.pools,
        [ROOT / path for path in args.exclude_parent_map],
        args.category,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
