"""Build a verified parent-TPF index for the complete KBonus LC2 corpus."""

import argparse
import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def corpus_sources(index):
    with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
        return {
            str(source_id): shard
            for source_id, shard in database.execute(
                "SELECT gaia_source_id, shard FROM sources"
            )
        }


@lru_cache(maxsize=4)
def verified_index(path, expected_sha256):
    path = Path(path).resolve()
    if digest(path) != expected_sha256:
        raise ValueError("Frozen parent index hash differs from vetting configuration")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        metadata = dict(database.execute("SELECT key, value FROM metadata"))
        if metadata.get("sources") != "406547" or metadata.get("schema") != "1":
            raise ValueError("Frozen parent index does not cover the LC2 corpus")
    return path


def parent_members(path, expected_sha256, source_id, parent_tpf):
    path = verified_index(str(path), expected_sha256)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        source = database.execute(
            "SELECT parent_tpf FROM sources WHERE source_id = ?", (str(source_id),)
        ).fetchone()
        if source is None:
            return None
        if source[0] != int(parent_tpf):
            raise ValueError(f"Frozen parent index disagrees with source {source_id}")
        return tuple(
            sid
            for (sid,) in database.execute(
                "SELECT source_id FROM sources WHERE parent_tpf = ? AND source_id != ? "
                "ORDER BY source_id",
                (source[0], str(source_id)),
            )
        )


def build(output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.sqlite")
    report_path = output.with_suffix(".json")
    if output.exists() or partial.exists() or report_path.exists():
        raise FileExistsError("Parent index output already exists")

    specification = json.loads((ROOT / "discSSD/data/index.json").read_text())
    peer = json.loads((ROOT / "discPCIe/data/index.json").read_text())
    corpus = specification["primary_collections"]["kbonus"]
    index = ROOT / "discSSD/data/index.sqlite"
    if (
        peer["lookup_sha256"] != specification["lookup_sha256"]
        or digest(index) != specification["lookup_sha256"]
        or digest(ROOT / "discPCIe/data/index.sqlite") != peer["lookup_sha256"]
    ):
        raise ValueError("KBonus corpus index hash differs from its manifest")
    remaining = corpus_sources(index)
    if len(remaining) != corpus["sources"]:
        raise ValueError("KBonus corpus index source count differs from its manifest")

    manifest_hashes = hashlib.sha256()
    with sqlite3.connect(partial) as database:
        database.execute(
            "CREATE TABLE sources (source_id TEXT PRIMARY KEY, parent_tpf INTEGER NOT NULL, "
            "shard INTEGER NOT NULL)"
        )
        database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        for shard in range(corpus["shards"]):
            volume = "discSSD" if shard % 2 == 0 else "discPCIe"
            folder = ROOT / volume / f"data/kbonus/lc2/shard-{shard:04d}"
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            source_file = folder / manifest["files"]["source"]["path"]
            identity = manifest["files"]["source"]
            if manifest["shard"] != shard or source_file.stat().st_size != identity["bytes"]:
                raise ValueError(f"Source manifest identity differs for shard {shard}")
            if digest(source_file) != identity["sha256"]:
                raise ValueError(f"Source metadata hash differs for shard {shard}")
            table = pq.read_table(
                source_file, columns=["source_id", "origin_tpf_kic"], use_threads=False
            )
            if len(table) != manifest["sources"] or len(table) != identity["rows"]:
                raise ValueError(f"Source metadata count differs for shard {shard}")
            records = []
            for source_id, parent in zip(
                table["source_id"].to_pylist(),
                table["origin_tpf_kic"].to_pylist(),
            ):
                sid = str(source_id)
                if parent is None or remaining.pop(sid, None) != shard:
                    raise ValueError(f"Source or parent identity differs in shard {shard}: {sid}")
                records.append((sid, int(parent), shard))
            database.executemany("INSERT INTO sources VALUES (?, ?, ?)", records)
            manifest_hashes.update(digest(manifest_path).encode())
            if (shard + 1) % 100 == 0 or shard + 1 == corpus["shards"]:
                print(f"Verified {shard + 1}/{corpus['shards']} shards", flush=True)
        if remaining:
            raise ValueError(f"Parent index omitted {len(remaining)} corpus sources")
        database.execute("CREATE INDEX sources_parent ON sources(parent_tpf)")
        mapping = hashlib.sha256()
        for sid, parent, shard in database.execute(
            "SELECT source_id, parent_tpf, shard FROM sources ORDER BY source_id"
        ):
            mapping.update(f"{sid},{parent},{shard}\n".encode())
        report = {
            "schema": 1,
            "sources": corpus["sources"],
            "shards": corpus["shards"],
            "parents": database.execute(
                "SELECT COUNT(DISTINCT parent_tpf) FROM sources"
            ).fetchone()[0],
            "corpus_index_sha256": specification["lookup_sha256"],
            "shard_manifests_sha256": manifest_hashes.hexdigest(),
            "mapping_sha256": mapping.hexdigest(),
        }
        database.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            ((key, str(value)) for key, value in report.items()),
        )
    partial.replace(output)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
