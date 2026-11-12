"""Materialize the sealed confirmed-planet holdout without detector or vetter outcomes."""

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from deterministic import catalogue_controls as inputs

PROTOCOL = "fsc-kepler-real-host-holdout-v1"
SIGNALS = ROOT / "dump/detector-benchmark-20260921/confirmed-signals.csv"
HOSTS = (
    ROOT / "discPCIe/data/kepler-planets/catalogue"
    / "kbonus-confirmed-kepler-hosts-20260904.json"
)
FITS = "kepler-planets/kbonus-fits"
INDEX = ROOT / "discSSD/data/index.sqlite"
GRAPH = ROOT / "dump/todo-20260926/joint-control-components-v11/trials.csv"
PUBLISHED = ROOT / "dump/rethink-20260929/published-candidates.csv"
HOLDOUT_SIGNALS = 687
EXPECTED_BYTES = 3 * 1024**3
SCOPE = (
    "Confirmed Kepler planets through the real KBonus extraction, sealed for vetting "
    "and disposition. Detector recovery of these hosts was opened during detector "
    "development and is not a sealed measurement."
)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def holdout_signals(rows):
    selected = [
        row for row in rows
        if row["split"] == "holdout"
        and row["input_eligible"] == "true"
        and row["in_domain"] == "true"
    ]
    if len(selected) != HOLDOUT_SIGNALS:
        raise ValueError(f"The holdout has {len(selected)} in-domain signals")
    return selected


def opened_identities(graph, signals, published, catalogue):
    """Source, parent-TPF and raw-file identities of every opened set."""
    opened = {"source_id": set(), "kic_id": set(), "fits_sha256": set()}
    for row in graph:
        if row["role"] != "development":
            continue
        opened["source_id"].update({row["source_id"], row["contaminant_source_id"]})
        opened["fits_sha256"].update(
            {row["raw_fits_sha256"], row["contaminant_raw_fits_sha256"]}
        )
        opened["kic_id"].add(row["origin_parent"])
        opened["kic_id"].update(row["resolved_parents"].split(";"))
        opened["kic_id"].update(
            item.split(":")[1]
            for item in row["spatial_pixel_parent_by_quarter"].split(";") if item
        )
    for row in signals:
        if row["split"] == "development":
            opened["source_id"].add(row["source_id"])
            opened["kic_id"].add(row["kic_id"])
            opened["kic_id"].add(catalogue[row["source_id"]]["originating_tpf"])
    for row in published:
        opened["source_id"].add(row["source_id"])
        opened["kic_id"].add(row["parent_tpf"])
    for values in opened.values():
        values.discard("")
    return opened


def conflicts(identities, opened):
    return sorted(name for name, values in identities.items() if values & opened[name])


def index_hashes():
    with sqlite3.connect(f"file:{INDEX.resolve()}?mode=ro", uri=True) as database:
        return dict(database.execute(
            "select path, sha256 from files "
            "where collection = 'kepler-planets' and path like ?",
            (f"{FITS}/%",),
        ))


def materialize_host(out, entry, source_id, expected_sha256):
    kic = int(entry["kic_id"])
    path = ROOT / "discPCIe/data" / FITS / entry["filename"]
    if str(fits.getval(path, "TPFORG")) != entry["originating_tpf"]:
        raise ValueError(f"The host extraction TPF differs from its catalogue: {kic}")
    metadata, arrays = inputs.read_product(path, kic)
    if (metadata["source_id"], metadata["raw_fits_sha256"]) != (source_id, expected_sha256):
        raise ValueError(f"The host product differs from the corpus index: {kic}")
    if not inputs.discovery_eligible(arrays):
        raise ValueError(f"The host product is no longer discovery eligible: {kic}")
    path = out / "arrays" / f"{source_id}.npz"
    inputs.write_arrays(path, arrays)
    quarters = np.unique(arrays["quarter"][arrays["sap_quality"] == 0])
    source = {
        "arrays": str(path.relative_to(out)),
        "sha256": inputs.digest(path),
        "metadata": metadata,
    }
    return source, [int(quarter) for quarter in quarters]


def seal_hosts(out, hosts, catalogue, opened):
    hashes = index_hashes()
    manifest, ledger = {}, []
    for source_id in hosts:
        entry = catalogue[source_id]
        record = {
            "source_id": source_id,
            "kic_id": str(entry["kic_id"]),
            "extraction_tpf_kic": entry["originating_tpf"],
            "fits_sha256": hashes[f"{FITS}/{entry['filename']}"],
        }
        identities = {
            "source_id": {source_id},
            "kic_id": {record["kic_id"], record["extraction_tpf_kic"]},
            "fits_sha256": {record["fits_sha256"]},
        }
        record["conflicts"] = conflicts(identities, opened)
        if record["conflicts"]:
            ledger.append({**record, "status": "EXCLUDED_OPENED_IDENTITY"})
            continue
        manifest[source_id], quarters = materialize_host(
            out, entry, source_id, record["fits_sha256"]
        )
        ledger.append({**record, "status": "SEALED", "required_pixel_quarters": quarters})
    return manifest, ledger


def write_outputs(out, manifest, truth, ledger):
    inputs.save(out / "input-manifest.json", {
        "schema": 1, "measurement_scope": SCOPE, "sources": manifest,
    })
    inputs.save(out / "source-ids.json", sorted(manifest, key=int))
    with (out / "truth.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    with (out / "eligibility.jsonl").open("x") as stream:
        stream.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in ledger)
    names = ("input-manifest.json", "source-ids.json", "truth.csv", "eligibility.jsonl")
    return {name: inputs.digest(out / name) for name in names}


def materialize(out):
    paths = [
        SIGNALS, HOSTS, INDEX, GRAPH, PUBLISHED, Path(__file__),
        Path(inputs.__file__), ROOT / "pipeline/deterministic/run.py",
    ]
    declared = {str(path.resolve()): inputs.digest(path) for path in paths}
    signal_rows = read_csv(SIGNALS)
    holdout = holdout_signals(signal_rows)
    catalogue = {
        str(entry["gaia_dr3_source_id"]): entry
        for entry in json.loads(HOSTS.read_text())
    }
    opened = opened_identities(
        read_csv(GRAPH), signal_rows, read_csv(PUBLISHED), catalogue
    )
    hosts = sorted({row["source_id"] for row in holdout}, key=int)
    out.mkdir()
    manifest, ledger = seal_hosts(out, hosts, catalogue, opened)
    truth = [row for row in holdout if row["source_id"] in manifest]
    artifacts = write_outputs(out, manifest, truth, ledger)
    for path, expected in declared.items():
        if inputs.digest(path) != expected:
            raise ValueError(f"A declared holdout input changed: {path}")
    freeze = {
        "schema": 1,
        "protocol": PROTOCOL,
        "status": "FROZEN_SEALED_INPUTS",
        "selection": (
            "Every split=holdout, input_eligible, in_domain signal of the 21 September "
            "detector benchmark. A host sharing a source, parent-TPF or raw-file "
            "identity with an opened set is excluded and not replaced. Parent "
            "identities are the host KIC and its KBonus extraction TPF (TPFORG)."
        ),
        "opened_sets": (
            "Development roles of the joint control graph with their contaminants and "
            "spatial parents; development confirmed-planet hosts; published background "
            "candidates."
        ),
        "claim_boundary": SCOPE,
        "inputs": declared,
        "hosts": len(hosts),
        "sealed_hosts": len(manifest),
        "sealed_signals": len(truth),
        "excluded_by_identity": dict(
            Counter(name for row in ledger for name in row["conflicts"])
        ),
        "required_pixel_quarters": sum(
            len(row.get("required_pixel_quarters", ())) for row in ledger
        ),
        "artifacts": artifacts,
        "scientific_outcomes_opened": False,
        "boi_catalogue_enabled": False,
    }
    inputs.save(out / "freeze.json", freeze)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"The holdout output already exists: {args.out}")
    free = shutil.disk_usage(args.out.parent).free
    if free < inputs.MINIMUM_FREE_BYTES + EXPECTED_BYTES:
        raise OSError("The holdout output volume lacks room above its 20 GiB reserve")
    os.nice(10)
    freeze = materialize(args.out)
    keys = ("status", "hosts", "sealed_hosts", "sealed_signals", "excluded_by_identity")
    print(json.dumps({key: freeze[key] for key in keys}))


if __name__ == "__main__":
    main()
