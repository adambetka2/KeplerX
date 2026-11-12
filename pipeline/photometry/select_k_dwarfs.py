"""Build an auditable Gaia-selected K-dwarf catalogue linked to original LC2 shards."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io.votable import parse_single_table
from astropy.table import Table

POPULATION = "discSSD/data/kbonus/catalogue/fsc_lc2_background_population.parquet"
POPULATION_SHA256 = "c2974d021d0141c238bb09a0194b3767ed8cc60f0372b930a85ad1a5e4f58071"
INDEX_SHA256 = "394490f58fce4848df9e5dae670f5d1689904f6e83e61654ae05ec54fed40d24"
SEQUENCE_URL = (
    "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
)
SEQUENCE_SHA256 = "1de2edeec17bb3346e0e4e70b999de5ee29947df38474e64cddb7cfacc164b7f"
TAP = "https://gaia.ari.uni-heidelberg.de/tap"
GAIA_COLUMNS = [
    "parallax",
    "parallax_error",
    "ruwe",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "teff_gspphot",
    "teff_gspphot_lower",
    "teff_gspphot_upper",
    "logg_gspphot",
    "logg_gspphot_lower",
    "logg_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
    "ebpminrp_gspphot_lower",
    "ebpminrp_gspphot_upper",
    "phot_bp_rp_excess_factor",
    "classprob_dsc_combmod_star",
]
QUERY = (
    "SELECT g.source_id, "
    + ", ".join(f"g.{name}" for name in GAIA_COLUMNS)
    + "\nFROM TAP_UPLOAD.cohort AS c JOIN gaiadr3.gaia_source AS g\n"
    "ON g.source_id = c.source_id\n"
)
CRITERIA = {
    "teff_min_k": 3900.0,
    "teff_max_k_exclusive": 5300.0,
    "logg_min_dex": 4.2,
    "parallax_snr_min": 10.0,
    "ruwe_max": 1.4,
    "cmd_offset_min_mag": -1.0,
    "cmd_offset_max_mag": 1.5,
    "near_4300_min_k": 4100.0,
    "near_4300_max_k_exclusive": 4500.0,
    "late_k_max_k_exclusive": 4500.0,
    "bright_g_max_exclusive": 16.0,
}
BASE_COLUMNS = [
    "source_id",
    "shard",
    "source_ordinal",
    "file_stem",
    "kic",
    "ra_deg",
    "dec_deg",
    "psf_avail",
    "sap_avail",
    "published_m_dwarf",
    "gaia_matched",
]
DERIVED_COLUMNS = [
    "parallax_snr",
    "bp_rp_intrinsic",
    "absolute_g",
    "cmd_offset_mag",
    "temperature_interval_in_k",
    "gravity_interval_dwarf",
]
TABLES = ("source", "cadence", "qdiag", "quarter", "header", "aperture")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def storage(root, out):
    for name in ("discSSD", "discPCIe"):
        volume = Path("/Volumes") / name
        if not volume.is_mount() or (root / name).resolve() != volume:
            raise ValueError(f"Missing mounted volume or incorrect symlink: {name}")
        if shutil.disk_usage(volume).free < 2 * 1024**3 + 512 * 1024**2:
            raise ValueError(f"Insufficient free space on {name}")
    dump = root / "dump"
    if not dump.is_symlink() or not out.resolve().is_relative_to(dump.resolve()):
        raise ValueError("Output must be under the repository's external dump symlink")


def population(root):
    path = root / POPULATION
    if digest(path) != POPULATION_SHA256:
        raise ValueError("Population differs from the frozen LC2 background cohort")
    columns = [
        "source_id",
        "source_ordinal",
        "shard_id",
        "fname",
        "kic",
        "ra",
        "dec",
        "psf_avail",
        "sap_avail",
        "is_pipeline_target",
    ]
    data = pq.read_table(path, columns=columns).to_pydict()
    ids = data["source_id"]
    if len(ids) != 406547 or len(set(ids)) != len(ids):
        raise ValueError("Expected exactly 406,547 unique background sources")
    if any(data["is_pipeline_target"]):
        raise ValueError("Population contains original Kepler pipeline targets")
    return data


def prepare(root, out):
    storage(root, out)
    data = population(root)
    out.mkdir(parents=True, exist_ok=False)
    Table({"source_id": data["source_id"]}).write(out / "cohort.vot", format="votable")
    (out / "query.adql").write_text(QUERY)
    print(json.dumps({"upload_sources": len(data["source_id"]), "directory": str(out)}))


def gaia_parameters(path, ids):
    statuses = []
    for _, element in ET.iterparse(path, events=["end"]):
        if element.tag.rsplit("}", 1)[-1] == "INFO":
            if element.get("name") == "QUERY_STATUS":
                statuses.append(element.get("value"))
        element.clear()
    if not statuses or any(status != "OK" for status in statuses):
        raise ValueError(f"Unsuccessful or truncated Gaia response: {statuses}")
    table = parse_single_table(path).to_table()
    if table.colnames != ["source_id", *GAIA_COLUMNS]:
        raise ValueError("Unexpected Gaia response schema")
    returned = list(map(int, table["source_id"]))
    positions = {sid: i for i, sid in enumerate(ids)}
    if len(set(returned)) != len(returned) or set(returned) - positions.keys():
        raise ValueError("Duplicate or out-of-cohort Gaia source IDs")
    matched = np.zeros(len(ids), dtype=bool)
    order = np.array([positions[sid] for sid in returned], dtype=int)
    matched[order] = True
    result = {}
    for name in GAIA_COLUMNS:
        values = np.ma.asarray(table[name], dtype=float).filled(np.nan)
        aligned = np.full(len(ids), np.nan)
        aligned[order] = values
        result[name] = aligned
    return matched, result


def dwarf_sequence(path):
    if digest(path) != SEQUENCE_SHA256:
        raise ValueError("Dwarf sequence differs from the preserved reference snapshot")
    anchors = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith(("G", "K", "M")) and len(fields) == 32:
            teff = float(fields[1])
            if 3400 <= teff <= 6000:
                anchors.append((float(fields[11]), float(fields[13])))
    anchors.sort()
    colours, magnitudes = np.array(anchors).T
    if len(colours) < 20 or np.any(np.diff(colours) <= 0):
        raise ValueError("Invalid colour-magnitude dwarf sequence")
    return colours, magnitudes


def derived(gaia, sequence):
    c = CRITERIA
    parallax = gaia["parallax"]
    error = gaia["parallax_error"]
    colour = (
        gaia["phot_bp_mean_mag"] - gaia["phot_rp_mean_mag"] - gaia["ebpminrp_gspphot"]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(error > 0, parallax / error, np.nan)
        absolute_g = (
            gaia["phot_g_mean_mag"] + 5 * np.log10(parallax) - 10 - gaia["ag_gspphot"]
        )
    expected_g = np.interp(colour, *sequence, left=np.nan, right=np.nan)
    stable_temperature = (gaia["teff_gspphot_lower"] >= c["teff_min_k"]) & (
        gaia["teff_gspphot_upper"] < c["teff_max_k_exclusive"]
    )
    stable_gravity = gaia["logg_gspphot_lower"] >= c["logg_min_dex"]
    return dict(
        zip(
            DERIVED_COLUMNS,
            [
                snr,
                colour,
                absolute_g,
                absolute_g - expected_g,
                stable_temperature,
                stable_gravity,
            ],
            strict=True,
        )
    )


def schema(database):
    c = CRITERIA
    numeric = ",\n".join(f"{name} REAL" for name in GAIA_COLUMNS + DERIVED_COLUMNS)
    database.executescript(f"""
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE shards(
            shard INTEGER PRIMARY KEY, volume TEXT NOT NULL,
            lc2_path TEXT NOT NULL, manifest_sha256 TEXT NOT NULL
        );
        CREATE TABLE sources(
            source_id TEXT PRIMARY KEY, shard INTEGER NOT NULL REFERENCES shards(shard),
            source_ordinal INTEGER UNIQUE NOT NULL, file_stem TEXT NOT NULL,
            kic INTEGER, ra_deg REAL NOT NULL, dec_deg REAL NOT NULL,
            psf_avail INTEGER NOT NULL, sap_avail INTEGER NOT NULL,
            published_m_dwarf INTEGER NOT NULL, gaia_matched INTEGER NOT NULL,
            {numeric}
        );
        CREATE INDEX source_shard ON sources(shard);
        CREATE VIEW classified AS SELECT s.*, h.volume, h.lc2_path,
            CASE
                WHEN gaia_matched=0 THEN 'MISSING_GAIA'
                WHEN teff_gspphot IS NULL THEN 'MISSING_TEFF'
                WHEN teff_gspphot < {c["teff_min_k"]}
                    OR teff_gspphot >= {c["teff_max_k_exclusive"]}
                    THEN 'OUTSIDE_K_TEMPERATURE'
                WHEN logg_gspphot IS NULL THEN 'MISSING_LOGG'
                WHEN logg_gspphot < {c["logg_min_dex"]} THEN 'LOW_GRAVITY'
                WHEN published_m_dwarf=1 THEN 'PUBLISHED_M_CONFLICT'
                WHEN parallax_snr IS NULL OR parallax_snr < {c["parallax_snr_min"]}
                    OR ruwe IS NULL OR ruwe > {c["ruwe_max"]} THEN 'ASTROMETRY_REVIEW'
                WHEN cmd_offset_mag IS NULL THEN 'MISSING_CMD'
                WHEN cmd_offset_mag < {c["cmd_offset_min_mag"]}
                    OR cmd_offset_mag > {c["cmd_offset_max_mag"]} THEN 'CMD_REVIEW'
                ELSE 'SELECTED'
            END AS selection_status
            FROM sources s JOIN shards h USING(shard);
        CREATE VIEW candidates AS SELECT * FROM classified
            WHERE teff_gspphot >= {c["teff_min_k"]}
              AND teff_gspphot < {c["teff_max_k_exclusive"]}
              AND logg_gspphot >= {c["logg_min_dex"]};
        CREATE VIEW selection AS SELECT * FROM classified
            WHERE selection_status='SELECTED';
        CREATE VIEW near_4300 AS SELECT * FROM selection
            WHERE teff_gspphot >= {c["near_4300_min_k"]}
              AND teff_gspphot < {c["near_4300_max_k_exclusive"]};
        CREATE VIEW late_k AS SELECT * FROM selection
            WHERE teff_gspphot < {c["late_k_max_k_exclusive"]};
        CREATE VIEW bright AS SELECT * FROM selection
            WHERE phot_g_mean_mag < {c["bright_g_max_exclusive"]};
        CREATE VIEW priority AS SELECT * FROM near_4300
            WHERE phot_g_mean_mag < {c["bright_g_max_exclusive"]};
        CREATE VIEW needs_review AS SELECT * FROM candidates
            WHERE selection_status!='SELECTED';
    """)


def link_shards(root, out, database, data):
    index = root / "discSSD/data/index.sqlite"
    if digest(index) != INDEX_SHA256:
        raise ValueError("Index differs from the verified frozen LC2 index")
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as original:
        indexed = dict(
            (sid, (volume, shard))
            for sid, volume, shard in original.execute(
                "SELECT gaia_source_id, volume, shard FROM sources"
            )
        )
    if set(indexed) != set(map(str, data["source_id"])):
        raise ValueError("Population and original index source IDs differ")
    for sid, shard in zip(data["source_id"], data["shard_id"], strict=True):
        if indexed[str(sid)][1] != shard:
            raise ValueError(f"Population and index shard assignments differ: {sid}")
    (out / "lc2").mkdir()
    for volume, shard in sorted(set(indexed.values())):
        folder = root / volume / f"data/kbonus/lc2/shard-{shard:04d}"
        paths = [folder / f"{name}.parquet" for name in TABLES] + [
            folder / "manifest.json"
        ]
        if any(not path.is_file() or not path.stat().st_size for path in paths):
            raise ValueError(f"Incomplete LC2 shard: {folder}")
        link = Path("lc2") / folder.name
        (out / link).symlink_to(folder)
        database.execute(
            "INSERT INTO shards VALUES(?,?,?,?)",
            (
                shard,
                volume,
                str(link),
                digest(folder / "manifest.json"),
            ),
        )


def source_records(data, matched, gaia, extra, m_dwarfs):
    arrays = list(gaia.values()) + list(extra.values())
    for i, sid in enumerate(data["source_id"]):
        base = [
            str(sid),
            data["shard_id"][i],
            data["source_ordinal"][i],
            data["fname"][i],
            data["kic"][i],
            data["ra"][i],
            data["dec"][i],
            int(data["psf_avail"][i]),
            int(data["sap_avail"][i]),
            int(str(sid) in m_dwarfs),
            int(matched[i]),
        ]
        values = [
            float(array[i]) if np.isfinite(array[i]) else None for array in arrays
        ]
        yield base + values


def summarize(database):
    views = (
        "sources",
        "candidates",
        "selection",
        "near_4300",
        "late_k",
        "bright",
        "priority",
        "needs_review",
    )
    return {
        "counts": {
            view: database.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
            for view in views
        },
        "first_failure_counts": dict(
            database.execute(
                "SELECT selection_status,count(*) FROM classified "
                "GROUP BY selection_status"
            )
        ),
        "selected_by_volume": dict(
            database.execute("SELECT volume,count(*) FROM selection GROUP BY volume")
        ),
        "published_m_overlap": database.execute(
            "SELECT count(*) FROM selection WHERE published_m_dwarf=1"
        ).fetchone()[0],
        "candidate_published_m_overlap": database.execute(
            "SELECT count(*) FROM candidates WHERE published_m_dwarf=1"
        ).fetchone()[0],
        "linked_shards": database.execute("SELECT count(*) FROM shards").fetchone()[0],
    }


def build(root, inputs, m_catalogue, out):
    storage(root, out)
    data = population(root)
    matched, gaia = gaia_parameters(inputs / "gaia-dr3.vot", data["source_id"])
    extra = derived(gaia, dwarf_sequence(inputs / "dwarf-sequence.txt"))
    job = ET.parse(inputs / "job-status.xml").getroot()
    query = job.find("{*}parameters/{*}parameter[@id='query']").text
    if " ".join(query.split()) != " ".join(QUERY.split()):
        raise ValueError("Recorded TAP job does not match the declared query")
    if job.find("{*}phase").text != "COMPLETED":
        raise ValueError("Recorded TAP job is incomplete")
    with sqlite3.connect(
        f"file:{m_catalogue.resolve()}?mode=ro", uri=True
    ) as m_database:
        m_dwarfs = {
            sid for (sid,) in m_database.execute("SELECT source_id FROM selection")
        }
    out.mkdir(parents=True, exist_ok=False)
    path = out / "k-dwarfs.sqlite"
    with sqlite3.connect(path) as database:
        schema(database)
        link_shards(root, out, database, data)
        columns = BASE_COLUMNS + GAIA_COLUMNS + DERIVED_COLUMNS
        placeholders = ",".join("?" for _ in columns)
        database.executemany(
            f"INSERT INTO sources VALUES({placeholders})",
            source_records(
                data,
                matched,
                gaia,
                extra,
                m_dwarfs,
            ),
        )
        manifest = {
            "schema": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "classification": (
                "Photometric K-dwarf candidates; not confirmed spectral types"
            ),
            "criteria": CRITERIA,
            "selection_basis": (
                "Temperature, gravity, astrometry and colour-magnitude consistency; "
                "conflicts with published M-dwarf membership remain in review"
            ),
            "criteria_origin": (
                "Project preselection choices, "
                "not cuts prescribed by the motivation paper"
            ),
            "cmd_definition": (
                "G + 5*log10(parallax_mas) - 10 - A_G minus interpolated "
                "dwarf M_G at BP-RP-E(BP-RP)"
            ),
            "parallax_correction": "Raw DR3 parallax; no zero-point correction applied",
            "uncertainties": (
                "Gaia lower/upper values are internal 16th/84th percentiles, "
                "not calibrated errors"
            ),
            "limitations": [
                "Missing parameters and failed quality checks do not exclude K dwarfs",
                "GSP-Phot extinction degeneracy and faint-star gravity biases remain",
                "CMD shares Gaia extinction modeling; it is not independent validation",
                "Sequence uses DR2 colours; CMD tolerance is a project choice",
                "Temperature/gravity interval flags are retained, not used as cuts",
                "Binaries, metallicity and parallax zero-point affect classification",
            ],
            "motivation_paper": "https://arxiv.org/abs/2501.03214",
            "gaia_documentation": "https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html",
            "sequence_url": SEQUENCE_URL,
            "tap_endpoint": TAP,
            "adql": QUERY,
            "tap_job_id": job.find("{*}jobId").text,
            "gaia_matched_sources": int(matched.sum()),
            "input_directory": str(inputs),
            "population_sha256": POPULATION_SHA256,
            "index_sha256": INDEX_SHA256,
            "m_catalogue_path": str(m_catalogue),
            "m_catalogue_sha256": digest(m_catalogue),
            "builder_sha256": digest(Path(__file__)),
            "input_sha256": {
                name: digest(inputs / name)
                for name in (
                    "cohort.vot",
                    "gaia-dr3.vot",
                    "dwarf-sequence.txt",
                    "job-status.xml",
                )
            },
            "curve_locator": (
                "Resolve lc2_path relative to this SQLite directory; "
                "read cadence.parquet rows for the exact source_id"
            ),
            "source_id_storage": "Decimal text, never floating point",
            "views": {
                "candidates": "Temperature and gravity cuts only",
                "selection": "Candidates passing astrometry and CMD cuts",
                "near_4300": "Selection with 4100 <= Teff < 4500 K",
                "late_k": "Selection with 3900 <= Teff < 4500 K",
                "bright": "Selection with Gaia G < 16",
                "priority": "Near-4300 selection with Gaia G < 16",
                "needs_review": "Candidates with membership, astrometry or CMD issues",
                "classified": "All background sources and first failed criterion",
            },
            **summarize(database),
        }
        database.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            (
                (key, json.dumps(value, allow_nan=False))
                for key, value in manifest.items()
            ),
        )
        if database.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Catalogue contains broken shard references")
        database.row_factory = sqlite3.Row
        selected = pa.Table.from_pylist(
            [
                dict(row)
                for row in database.execute(
                    "SELECT * FROM selection ORDER BY source_ordinal"
                )
            ]
        )
        parquet = out / "k-dwarfs.parquet"
        pq.write_table(selected, parquet, compression="zstd")
    manifest.update(sqlite_sha256=digest(path), sqlite_bytes=path.stat().st_size)
    manifest.update(
        parquet_sha256=digest(parquet), parquet_bytes=parquet.stat().st_size
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest["counts"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("--out", type=Path, required=True)
    build_parser = actions.add_parser("build")
    build_parser.add_argument("--inputs", type=Path, required=True)
    build_parser.add_argument("--m-dwarfs", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.absolute(), args.out.absolute()
    if args.action == "prepare":
        prepare(root, out)
    else:
        build(root, args.inputs.absolute(), args.m_dwarfs.absolute(), out)


if __name__ == "__main__":
    main()
