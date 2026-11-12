"""Shared artifact contracts; scientific decisions live in their owning stage."""

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_directory(path, config):
    path = Path(path).resolve()
    require(
        not path.is_relative_to("/Volumes"),
        "Campaign outputs must use the Mac internal disk",
    )
    require(not path.exists(), f"Output already exists; use a new generation: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    reserve = config["execution"]["minimum_free_gib"] * 1024**3
    require(
        shutil.disk_usage(parent).free >= reserve, "Insufficient internal disk reserve"
    )
    path.mkdir()
    return path


def config_file(path=None):
    result = read(path or ROOT / "config.json")
    result["runtime_root"] = str(
        ROOT / os.environ.get("KEPLERX_RUNTIME_ROOT", result["runtime_root"])
    )
    detection = result["detection"]
    require(
        0 < detection["period_min_days"] < detection["period_max_days"],
        "Invalid period interval",
    )
    require(
        detection["min_observed_transits"] >= 3,
        "Periodic campaign requires at least three events",
    )
    return result


def save_packet(out, arrays, metadata):
    path = Path(out) / "photometry.npz"
    np.savez_compressed(path, **arrays)
    write(Path(out) / "packet.json", metadata | {"arrays": receipt(path)})


def load_packet(directory):
    directory = Path(directory)
    metadata = read(directory / "packet.json")
    path = directory / "photometry.npz"
    require(digest(path) == metadata["arrays"]["sha256"], "Photometry packet changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, metadata


def phase(time, period, epoch):
    return (np.asarray(time) - epoch + period / 2) % period - period / 2


def source_manifest():
    """Digests of the campaign's code and locks; runtime/, data/ and results/ are not code."""
    files = [ROOT / "uv.lock", ROOT / "upstream.json"]
    for directory, folders, names in os.walk(ROOT):
        folders[:] = [
            name
            for name in folders
            if name not in ("runtime", "data", "results", "__pycache__")
            and not name.startswith(".")
        ]
        files.extend(
            Path(directory) / name
            for name in names
            if Path(name).suffix
            in (".py", ".c", ".cpp", ".m", ".metal", ".h5", ".patch")
        )
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
