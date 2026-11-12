"""A run's outputs: one SQLite database, OUT/results.sqlite."""

import hashlib
import json
import shutil
import sqlite3
import zlib
from pathlib import Path

DATABASE = "results.sqlite"
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(case_id TEXT PRIMARY KEY, ordinal INTEGER,
    state TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS files(case_id TEXT, path TEXT, bytes INTEGER,
    sha256 TEXT, data BLOB, PRIMARY KEY(case_id, path));
CREATE TABLE IF NOT EXISTS candidates(case_id TEXT, signal_id TEXT,
    period REAL, mes REAL, t0 REAL, tdur REAL, PRIMARY KEY(case_id, signal_id));
CREATE TABLE IF NOT EXISTS vetting(case_id TEXT, signal_id TEXT,
    state TEXT, metrics TEXT, metric_errors TEXT, PRIMARY KEY(case_id, signal_id));
CREATE TABLE IF NOT EXISTS bois(case_id TEXT, boi_id TEXT,
    signal_id TEXT, state TEXT, PRIMARY KEY(case_id, boi_id));
CREATE TABLE IF NOT EXISTS provenance(sha256 TEXT PRIMARY KEY, record TEXT);
"""
# Calibration cases share stars, so rows are keyed by case too.


def connect(out):
    database = sqlite3.connect(Path(out) / DATABASE)
    database.execute("PRAGMA journal_mode=WAL")
    database.executescript(SCHEMA)
    return database


def pack_case(destination):
    """Rows and compressed files of one completed case folder."""
    destination = Path(destination)
    files, shared = [], []
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        data = path.read_bytes()
        if path.parent.name == "provenance" and path.parent.parent == destination:
            shared.append((path.stem, data.decode()))
            continue
        files.append((str(path.relative_to(destination)), len(data),
                      hashlib.sha256(data).hexdigest(), zlib.compress(data, 6)))  # fmt: skip
    detections = _json(destination / "detections" / "detections.json")
    vetting = _json(destination / "vetting" / "vetting.json")
    bois = _json(destination / "bois.json")
    return {
        "files": files,
        "provenance": shared,
        "candidates": [
            (row["signal_id"], row["period"], row["mes"], row["t0"], row["tdur"])
            for row in (detections or {}).get("candidates", [])
        ],
        "vetting": [
            (
                row["signal_id"],
                row["disposition"]["state"],
                json.dumps(row["metrics"]),
                json.dumps(row.get("metric_errors", [])),
            )  # fmt: skip
            for row in (vetting or {}).get("results", [])
        ],
        "bois": [
            (row["boi_id"], row["signal_id"], row["state"])
            for row in (bois or {}).get("results", [])
        ],
    }


def store(database, case_id, pack, destination):
    """Write one case's pack, verify every file, commit, remove the folder."""
    for sha256, record in pack.get("provenance", []):
        if hashlib.sha256(record.encode()).hexdigest() != sha256:
            raise RuntimeError(
                f"Provenance record differs from its name: {case_id}/{sha256}"
            )
    database.executemany(
        "INSERT OR IGNORE INTO provenance VALUES(?,?)", pack.get("provenance", [])
    )
    database.executemany(
        "INSERT OR REPLACE INTO files VALUES(?,?,?,?,?)",
        [(case_id, *row) for row in pack["files"]],
    )
    for table, rows in (("candidates", pack["candidates"]), ("vetting", pack["vetting"]),
                        ("bois", pack["bois"])):  # fmt: skip
        if rows:
            marks = ",".join("?" * (len(rows[0]) + 1))
            database.executemany(
                f"INSERT OR REPLACE INTO {table} VALUES({marks})",
                [(case_id, *row) for row in rows],
            )
    database.commit()
    for path, _, sha256, _ in pack["files"]:
        (blob,) = database.execute(
            "SELECT data FROM files WHERE case_id=? AND path=?", (case_id, path)
        ).fetchone()
        if hashlib.sha256(zlib.decompress(blob)).hexdigest() != sha256:
            raise RuntimeError(f"Stored output differs: {case_id}/{path}")
    shutil.rmtree(destination)


class RunOutputs:
    """A run's per-case files and jobs, read from results.sqlite."""

    def __init__(self, run):
        self.database = sqlite3.connect(Path(run) / DATABASE)

    def read_bytes(self, case_id, path):
        row = self.database.execute(
            "SELECT data FROM files WHERE case_id=? AND path=?", (case_id, path)
        ).fetchone()
        return None if row is None else zlib.decompress(row[0])

    def read_json(self, case_id, path):
        data = self.read_bytes(case_id, path)
        return None if data is None else json.loads(data)

    def provenance(self, reference):
        """The shared record a run.json field refers to as {"sha256": ...}."""
        row = self.database.execute(
            "SELECT record FROM provenance WHERE sha256=?", (reference["sha256"],)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def case_ids(self):
        return {row[0] for row in self.database.execute("SELECT case_id FROM jobs")}


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None
