"""Local human review of frozen BOIs; selections form a separate catalogue."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import secrets
import sqlite3
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
CATALOGUE_HASH = "58292690ec7672ee6a3b706dc27d2f2c7bf95e9422a74a102ee4267f08c751bc"
CATEGORIES = ("transit", "shallow", "event", "legacy")


def utc():
    return datetime.now(UTC).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lane(proposal):
    if "TRANSIT_LIKE" in proposal["detector_classifications"]:
        return "transit"
    if proposal["signal_id"].startswith("shallow-v1-"):
        return "shallow"
    if proposal["signal_id"].startswith("signal-"):
        return "event"
    return "legacy"


def plot_key(proposal, view):
    value = [
        proposal["source_id"],
        proposal["signal_id"],
        proposal["discovery_view"],
        view,
    ]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def curve_key(source_id, view):
    return hashlib.sha256(json.dumps([source_id, view]).encode()).hexdigest()


@lru_cache(maxsize=1)
def read_source(path, expected, stamp):
    if sha(path) != expected:
        raise ValueError("Source evidence checksum differs")
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


@lru_cache(maxsize=2)
def read_plot_index(path, stamp):
    with gzip.open(path, "rt") as stream:
        receipt = json.load(stream)
    if receipt["catalogue_sha256"] != CATALOGUE_HASH:
        raise ValueError("Display extracts belong to another catalogue")
    return {r["key"]: r for r in receipt["files"]}


class Review:
    def __init__(self, archive, work):
        self.archive, self.work = archive.resolve(strict=True), work.resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.catalogue_path = (
            self.archive
            / "server/srv/kbonus/workspace/dump/kbonus-full-406547-20261005-transit-only-v1/boi-catalogue.json"
        )
        if sha(self.catalogue_path) != CATALOGUE_HASH:
            raise ValueError("Frozen BOI catalogue checksum differs")
        catalogue = json.loads(self.catalogue_path.read_text())
        self.objects = {row["boi_id"]: row for row in catalogue["objects"]}
        index = json.loads(
            (self.archive / "preservation/required-source-index.json").read_text()
        )
        self.source_hashes = {r["evidence"]: r["evidence_sha256"] for r in index}
        self.items = [self.item(row) for row in self.objects.values()]
        self.db_path = self.work / "reviews.sqlite"
        with self.database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT NOT NULL,created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(dataset TEXT,boi TEXT,choice TEXT NOT NULL,reason TEXT NOT NULL,note TEXT NOT NULL,updated TEXT NOT NULL,revision INTEGER NOT NULL,PRIMARY KEY(dataset,boi));
                CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY,dataset TEXT,boi TEXT,previous TEXT,current TEXT NOT NULL,created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS bindings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """)
            frozen = db.execute(
                "SELECT value FROM bindings WHERE key='catalogue'"
            ).fetchone()
            if frozen and frozen[0] != CATALOGUE_HASH:
                raise ValueError("This review database belongs to another catalogue")
            db.execute(
                "INSERT OR IGNORE INTO bindings VALUES('catalogue',?)",
                (CATALOGUE_HASH,),
            )
            if not db.execute("SELECT 1 FROM datasets LIMIT 1").fetchone():
                db.execute(
                    "INSERT INTO datasets VALUES(?,?,?)",
                    (uuid.uuid4().hex, "BOI manual review", utc()),
                )

    @contextmanager
    def database(self):
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def item(self, row):
        lanes = {lane(p) for p in row["proposals"]}
        category = next(c for c in CATEGORIES if c in lanes)
        primary = next(p for p in row["proposals"] if lane(p) == category)
        return {
            "id": row["boi_id"],
            "source_id": row["source_id"],
            "category": category,
            "lanes": sorted(lanes),
            "views": row["views"],
            "period": primary["period_days"],
            "duration_hours": 24 * primary["duration_days"]
            if primary["duration_days"] is not None
            else None,
            "units": len(primary["event_units"]),
            "proposals": len(row["proposals"]),
        }

    def datasets(self):
        with self.database() as db:
            return [
                dict(r)
                for r in db.execute("SELECT * FROM datasets ORDER BY created,id")
            ]

    def decisions(self, dataset):
        with self.database() as db:
            self.check_dataset(db, dataset)
            return {
                r["boi"]: dict(r)
                for r in db.execute(
                    "SELECT * FROM decisions WHERE dataset=?", (dataset,)
                )
            }

    @staticmethod
    def check_dataset(db, dataset):
        if not db.execute("SELECT 1 FROM datasets WHERE id=?", (dataset,)).fetchone():
            raise ValueError("Unknown dataset")

    def create(self, name):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Dataset name must contain 1–80 characters")
        row = {"id": uuid.uuid4().hex, "name": name.strip(), "created": utc()}
        with self.lock, self.database() as db:
            db.execute("INSERT INTO datasets VALUES(:id,:name,:created)", row)
        return row

    def decide(self, data):
        dataset, boi, choice = (data[k] for k in ("dataset", "boi", "choice"))
        if boi not in self.objects or choice not in {"YES", "NO", "UNREVIEWED"}:
            raise ValueError("Invalid BOI or decision")
        reason, note = data.get("reason", ""), data.get("note", "")
        if (
            not isinstance(note, str)
            or len(note) > 4000
            or reason not in {"", "EB", "VARIABLE", "ARTIFACT", "UNCLEAR", "OTHER"}
        ):
            raise ValueError("Invalid reason or note")
        with self.lock, self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            self.check_dataset(db, dataset)
            old = db.execute(
                "SELECT * FROM decisions WHERE dataset=? AND boi=?", (dataset, boi)
            ).fetchone()
            revision = old["revision"] if old else 0
            if data.get("revision") != revision:
                raise RuntimeError(
                    "Decision changed in another window. Reload before saving."
                )
            row = {
                "dataset": dataset,
                "boi": boi,
                "choice": choice,
                "reason": reason,
                "note": note,
                "updated": utc(),
                "revision": revision + 1,
            }
            db.execute(
                "INSERT OR REPLACE INTO decisions VALUES(:dataset,:boi,:choice,:reason,:note,:updated,:revision)",
                row,
            )
            db.execute(
                "INSERT INTO history(dataset,boi,previous,current,created) VALUES(?,?,?,?,?)",
                (
                    dataset,
                    boi,
                    json.dumps(dict(old)) if old else None,
                    json.dumps(row),
                    row["updated"],
                ),
            )
        return row

    def evidence_path(self, proposal):
        path = self.archive / "server" / proposal["evidence"].lstrip("/")
        if not path.resolve().is_relative_to(self.archive / "server"):
            raise ValueError("Evidence path escapes the archive")
        return path

    def proposal(self, boi, index):
        if boi not in self.objects:
            raise ValueError("Unknown BOI")
        proposals = self.objects[boi]["proposals"]
        if index < 0 or index >= len(proposals):
            raise ValueError("Unknown proposal")
        return proposals[index]

    def evidence(self, boi, index):
        p = self.proposal(boi, index)
        path = self.evidence_path(p)
        source = read_source(
            path, self.source_hashes[p["evidence"]], path.stat().st_mtime_ns
        )
        signal = next(s for s in source["signals"] if s["signal_id"] == p["signal_id"])
        orbital = (
            source.get("orbital", {}).get("hypotheses", {}).get(p["signal_id"], {})
        )
        return {
            "proposal": p,
            "proposals": self.objects[boi]["proposals"],
            "coordinates": source["coordinates"],
            "stellar": source["stellar"],
            "measurements": {
                v: {
                    k: m.get(k)
                    for k in ("primary_depth", "odd_even", "secondary_phase_half")
                }
                for v, m in orbital.get("views", {}).items()
            },
            "events": signal["events"],
            "source_bois": [
                {"boi_id": r["boi_id"], "events": r["proposals"][0]["event_units"]}
                for r in self.objects.values()
                if r["source_id"] == p["source_id"]
            ],
            "gates": [
                {k: g[k] for k in ("stage", "decision", "reason")}
                for g in signal["triage"]["history"]
            ],
            "catalogues": source["catalogues"],
            "evidence_path": str(path),
            "evidence_sha256": self.source_hashes[p["evidence"]],
        }

    def plot(self, boi, index, view, full=False):
        if view not in {"psf_flat", "psf_qdiag", "sap_qdiag"}:
            raise ValueError("Unknown photometry")
        proposal = self.proposal(boi, index)
        key = (
            curve_key(proposal["source_id"], view) if full else plot_key(proposal, view)
        )
        folder = "curves" if full else "plots"
        manifest = self.work / folder / "index.json.gz"
        if not manifest.exists():
            raise FileNotFoundError(
                "Light-curve display extracts are still being prepared"
            )
        row = read_plot_index(manifest, manifest.stat().st_mtime_ns)[key]
        path = self.work / folder / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("Display extract checksum differs")
        return path.read_bytes()

    def export(self, dataset, extension):
        decisions = self.decisions(dataset)
        info = next(r for r in self.datasets() if r["id"] == dataset)
        selection = []
        for boi, d in decisions.items():
            if d["choice"] == "UNREVIEWED":
                continue
            original = self.objects[boi]
            evidence = [
                {
                    "original_path": p["evidence"],
                    "local_path": str(self.evidence_path(p)),
                    "sha256": self.source_hashes[p["evidence"]],
                }
                for p in original["proposals"]
            ]
            selection.append(
                {
                    "review": d,
                    "original_boi": original,
                    "evidence": evidence,
                    "display": [
                        self.display_references(p) for p in original["proposals"]
                    ],
                }
            )
        if extension == "json":
            value = {
                "schema": 1,
                "dataset": info,
                "exported_utc": utc(),
                "catalogue_sha256": CATALOGUE_HASH,
                "counts": dict(
                    Counter(
                        decisions.get(boi, {}).get("choice", "UNREVIEWED")
                        for boi in self.objects
                    )
                ),
                "accepted": [r for r in selection if r["review"]["choice"] == "YES"],
                "denied": [r for r in selection if r["review"]["choice"] == "NO"],
                "unreviewed": len(self.objects)
                - sum(d["choice"] != "UNREVIEWED" for d in decisions.values()),
                "scope": "Human dataset inclusion/exclusion. Original survey dispositions are unchanged; these decisions are not physical confirmation or an automated false-positive adjudication.",
            }
            content = json.dumps(value, indent=2, allow_nan=False).encode()
        else:
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "boi_id",
                    "source_id",
                    "category",
                    "decision",
                    "reason",
                    "note",
                    "updated_utc",
                    "period_days",
                    "original_catalogue_sha256",
                ]
            )
            for item in self.items:
                d = decisions.get(item["id"], {})
                writer.writerow(
                    [
                        item["id"],
                        item["source_id"],
                        item["category"],
                        d.get("choice", "UNREVIEWED"),
                        d.get("reason", ""),
                        d.get("note", ""),
                        d.get("updated", ""),
                        item["period"],
                        CATALOGUE_HASH,
                    ]
                )
            content = stream.getvalue().encode()
        folder = self.work / "datasets" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"catalogue.{extension}"
        temporary = path.with_suffix(".tmp")
        with self.lock:
            temporary.write_bytes(content)
            temporary.replace(path)
        return content

    def display_references(self, proposal):
        references = {}
        for folder in ("plots", "curves"):
            manifest = self.work / folder / "index.json.gz"
            index = read_plot_index(manifest, manifest.stat().st_mtime_ns)
            rows = []
            for view in ("psf_flat", "psf_qdiag", "sap_qdiag"):
                key = (
                    plot_key(proposal, view)
                    if folder == "plots"
                    else curve_key(proposal["source_id"], view)
                )
                row = index[key]
                rows.append(
                    {
                        "view": view,
                        "path": str(self.work / folder / row["file"]),
                        "sha256": row["sha256"],
                    }
                )
            references[folder] = rows
        return references


class Handler(BaseHTTPRequestHandler):
    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        return host in hosts and (
            not origin or origin in {f"http://{h}" for h in hosts}
        )

    def respond(self, content, kind="application/json", status=200, extra=None):
        if not isinstance(content, bytes):
            content = json.dumps(content, allow_nan=False).encode()
        self.send_response(status)
        for key, value in dict(
            {
                "Content-Type": kind,
                "Content-Length": str(len(content)),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            },
            **(extra or {}),
        ).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        try:
            if not self.local_request():
                self.respond({"error": "Local access only"}, status=403)
                return
            self.get_route()
        except FileNotFoundError as exc:
            self.respond({"error": str(exc)}, status=503)
        except (ValueError, KeyError, IndexError) as exc:
            self.respond({"error": str(exc)}, status=400)
        except OSError as exc:
            self.respond({"error": str(exc)}, status=503)

    def get_route(self):
        app = self.server.review
        path = unquote(urlsplit(self.path).path)
        if path == "/api/boot":
            self.respond(
                {
                    "items": app.items,
                    "datasets": app.datasets(),
                    "token": app.token,
                    "archive": str(app.archive),
                    "work": str(app.work),
                    "plots_ready": (app.work / "plots/index.json.gz").exists(),
                }
            )
        elif path.startswith("/api/dataset/"):
            self.respond(app.decisions(path.rsplit("/", 1)[-1]))
        elif path.startswith("/api/evidence/"):
            _, _, _, boi, index = path.split("/")
            self.respond(app.evidence(boi, int(index)))
        elif path.startswith(("/api/plot/", "/api/curve/")):
            _, _, _, boi, index, view = path.split("/")
            self.respond(
                app.plot(boi, int(index), view, path.startswith("/api/curve/")),
                extra={"Content-Encoding": "gzip"},
            )
        elif path.startswith("/api/export/"):
            name = path.rsplit("/", 1)[-1]
            dataset, extension = name.rsplit(".", 1)
            if extension not in {"json", "csv"}:
                raise ValueError("Unsupported export")
            self.respond(
                app.export(dataset, extension),
                "application/json"
                if extension == "json"
                else "text/csv; charset=utf-8",
                extra={
                    "Content-Disposition": f'attachment; filename="boi-review.{extension}"'
                },
            )
        elif path.startswith("/api/original/"):
            _, _, _, boi, index = path.split("/")
            p = app.proposal(boi, int(index))
            f = app.evidence_path(p)
            if sha(f) != app.source_hashes[p["evidence"]]:
                raise ValueError("Source evidence checksum differs")
            self.respond(
                f.read_bytes(),
                "application/gzip",
                extra={
                    "Content-Disposition": f'attachment; filename="{p["source_id"]}.json.gz"'
                },
            )
        elif path in {"/", "/style.css", "/review.js"}:
            file = HERE / ("index.html" if path == "/" else path[1:])
            kind = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}[
                file.suffix
            ]
            self.respond(file.read_bytes(), kind + "; charset=utf-8")
        else:
            self.respond({"error": "Not found"}, status=404)

    def do_POST(self):
        try:
            app = self.server.review
            if (
                not self.local_request()
                or self.headers.get("X-Review-Token") != app.token
            ):
                self.respond({"error": "Invalid local review token"}, status=403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request length")
            data = json.loads(self.rfile.read(length))
            if self.path == "/api/decision":
                self.respond(app.decide(data))
            elif self.path == "/api/datasets":
                self.respond(app.create(data["name"]))
            else:
                self.respond({"error": "Not found"}, status=404)
        except RuntimeError as exc:
            self.respond({"error": str(exc)}, status=409)
        except (ValueError, KeyError, TypeError) as exc:
            self.respond({"error": str(exc)}, status=400)
        except (OSError, sqlite3.Error) as exc:
            self.respond({"error": str(exc)}, status=503)

    def log_message(self, fmt, *args):
        print(f"{utc()} {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("/Volumes/discPCIe/kbonus")
    )
    parser.add_argument(
        "--work", type=Path, default=HERE.parents[1] / "dump/kbonus-review-20261011"
    )
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.review = Review(args.archive, args.work)
    print(
        f"BOI review: http://127.0.0.1:{args.port} — {len(server.review.objects)} objects",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
