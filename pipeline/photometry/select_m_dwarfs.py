"""Index the published KBonus M-dwarf selection against the original LC2 corpus."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from astropy.io import fits

CATALOGUE_URL = (
    "https://archive.stsci.edu/hlsps/kbonus-bkg/catalogs/"
    "hlsp_kbonus-bkg_kepler_kepler_mstars-catalog_kepler_v1.0_cat.fits"
)
CATALOGUE_SHA256 = "406563951c367bb1ffe3c20f83c4684a4083a2fe2f3d7c36ad9a663b5f6977ef"
TABLES = ("source", "cadence", "qdiag", "quarter", "header", "aperture")
COLUMNS = [
    "source_id",
    "gaia_designation",
    "file_stem",
    "raw_fits_sha256",
    "raw_fits_bytes",
    "stitched_rows",
    "quarter_count",
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def storage(root, out):
    for name in ("discSSD", "discPCIe"):
        volume = Path("/Volumes") / name
        if not volume.is_mount() or (root / name).resolve() != volume:
            raise ValueError(f"Missing mounted volume or incorrect symlink: {name}")
        required = 2 * 1024**3 + 32 * 1024**2
        if shutil.disk_usage(volume).free < required:
            raise ValueError(f"Insufficient free space on {name}")
    dump = root / "dump"
    if not dump.is_symlink() or not out.resolve().is_relative_to(dump.resolve()):
        raise ValueError("Output must be under the repository's external dump symlink")


def published_sources(path):
    with fits.open(path, mode="readonly", memmap=True) as hdus:
        table = hdus[1].data
        if table.names != ["gaia_designation", "ra", "dec", "kic", "fname"]:
            raise ValueError("Unexpected published M-dwarf catalogue schema")
        rows = []
        for ordinal, row in enumerate(table):
            designation = str(row["gaia_designation"]).strip()
            prefix, sid = designation.rsplit(" ", 1)
            if prefix != "Gaia DR3" or not sid.isdecimal():
                raise ValueError(f"Invalid Gaia DR3 identity: {designation}")
            rows.append(
                (
                    sid,
                    ordinal,
                    designation,
                    float(row["ra"]),
                    float(row["dec"]),
                    int(row["kic"]),
                    str(row["fname"]).strip(),
                )
            )
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Duplicate published Gaia source IDs")
    return rows


def shard_metadata(root, volume, shard):
    folder = root / volume / f"data/kbonus/lc2/shard-{shard:04d}"
    for name in (*TABLES, "manifest"):
        suffix = "json" if name == "manifest" else "parquet"
        path = folder / f"{name}.{suffix}"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing LC2 input: {path}")
    metadata = pq.read_table(folder / "source.parquet", columns=COLUMNS).to_pylist()
    rows = {str(row["source_id"]): row for row in metadata}
    if len(rows) != len(metadata):
        raise ValueError(f"Duplicate LC2 metadata source IDs: {folder}")
    return folder, rows


def build(root, catalogue, out):
    storage(root, out)
    if digest(catalogue) != CATALOGUE_SHA256:
        raise ValueError("Catalogue differs from the frozen published v1.0 snapshot")
    rows = published_sources(catalogue)
    index = root / "discSSD/data/index.sqlite"
    targets = (
        root / "discSSD/data/reference-catalogues/kepler_pipeline_target_ids.csv.gz"
    )
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
        indexed = {
            sid: (volume, shard)
            for sid, volume, shard in database.execute(
                "SELECT gaia_source_id, volume, shard FROM sources"
            )
        }
    with gzip.open(targets, "rt") as stream:
        pipeline_targets = {int(row["kic"]) for row in csv.DictReader(stream)}
    assignments = {indexed[row[0]] for row in rows if row[0] in indexed}
    selected_ids = {row[0] for row in rows if row[0] in indexed}
    out.mkdir(parents=True, exist_ok=False)
    (out / "lc2").mkdir()
    path = out / "m-dwarfs.sqlite"
    with sqlite3.connect(path) as database:
        database.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE shards(
                shard INTEGER PRIMARY KEY, volume TEXT NOT NULL,
                lc2_path TEXT NOT NULL, manifest_sha256 TEXT NOT NULL
            );
            CREATE TABLE sources(
                source_id TEXT PRIMARY KEY, catalogue_row INTEGER UNIQUE NOT NULL,
                gaia_designation TEXT NOT NULL, ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL, catalogue_kic INTEGER NOT NULL,
                file_stem TEXT NOT NULL, in_kepler_pipeline INTEGER NOT NULL,
                shard INTEGER REFERENCES shards(shard), raw_fits_sha256 TEXT,
                raw_fits_bytes INTEGER, stitched_rows INTEGER, quarter_count INTEGER
            );
            CREATE INDEX source_shard ON sources(shard);
            CREATE VIEW selection AS
                SELECT s.*, h.volume, h.lc2_path FROM sources s
                JOIN shards h USING(shard);
            CREATE VIEW outside_background_cohort AS
                SELECT * FROM sources WHERE shard IS NULL;
        """)
        database.execute("PRAGMA foreign_keys=ON")
        local = {}
        for volume, shard in sorted(assignments):
            folder, metadata = shard_metadata(root, volume, shard)
            link = Path("lc2") / f"shard-{shard:04d}"
            (out / link).symlink_to(folder)
            database.execute(
                "INSERT INTO shards VALUES(?,?,?,?)",
                (shard, volume, str(link), digest(folder / "manifest.json")),
            )
            local.update(
                (sid, record) for sid, record in metadata.items() if sid in selected_ids
            )
        for row in rows:
            sid, _, designation, _, _, kic, stem = row
            original_target = kic in pipeline_targets
            record = None
            shard = None
            if sid in indexed:
                _, shard = indexed[sid]
                record = local[sid]
                if (
                    record["gaia_designation"] != designation
                    or record["file_stem"] != stem
                ):
                    raise ValueError(
                        f"Published and local source identities differ: {sid}"
                    )
            if original_target != (record is None):
                raise ValueError(f"Unresolved background-cohort membership: {sid}")
            details = [record[column] if record else None for column in COLUMNS[3:]]
            database.execute(
                "INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (*row, int(original_target), shard, *details),
            )
        selected = database.execute("SELECT count(*) FROM selection").fetchone()[0]
        counts = dict(Counter(indexed[row[0]][0] for row in rows if row[0] in indexed))
        manifest = {
            "schema": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": (
                "Published KBonus v1.0 M-dwarf membership intersected "
                "with the frozen LC2 background cohort"
            ),
            "catalogue_url": CATALOGUE_URL,
            "paper": "https://arxiv.org/abs/2310.17733",
            "dataset_doi": "10.17909/7jbr-w430",
            "catalogue_path": str(catalogue),
            "catalogue_sha256": digest(catalogue),
            "index_sha256": digest(index),
            "kepler_pipeline_target_ids_sha256": digest(targets),
            "builder_sha256": digest(Path(__file__)),
            "published_sources": len(rows),
            "selected_lc2_sources": selected,
            "outside_background_cohort": len(rows) - selected,
            "outside_background_cohort_reason": "Original Kepler pipeline targets",
            "sources_by_volume": counts,
            "linked_shards": len(assignments),
            "path_base": "Directory containing m-dwarfs.sqlite",
            "curve_locator": (
                "Read lc2_path/cadence.parquet rows for the exact source_id; "
                "source_id is stored as decimal text"
            ),
            "new_stellar_classification": False,
        }
        database.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            (
                (key, json.dumps(value, allow_nan=False))
                for key, value in manifest.items()
            ),
        )
    manifest["sqlite_sha256"] = digest(path)
    manifest["sqlite_bytes"] = path.stat().st_size
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.absolute(), args.catalogue.absolute(), args.out.absolute())


if __name__ == "__main__":
    main()
