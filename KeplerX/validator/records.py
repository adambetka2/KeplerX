"""Receipts and never-overwritten JSON records (copied from post_boi.records)."""

import hashlib
import json
import math
import os
import uuid
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def candidate(path):
    path = Path(path).resolve()
    c = read(path)
    names = (
        "photometry",
        "prepared",
        "native_review",
        "readiness",
        "scene",
        "geometry",
        "background",
    )
    for key in names:
        if c[key] is not None:
            c[key] = str((path.parent / c[key]).resolve())
    c["native"]["priors"] = (
        str((path.parent / c["native"]["priors"]).resolve())
        if c["native"]["priors"] is not None
        else None
    )
    for key in ("contrast_curve", "molusc", "noise"):
        if c["probability"][key] is not None:
            c["probability"][key] = str((path.parent / c["probability"][key]).resolve())
    if c.get("protection_mask") is not None:
        c["protection_mask"]["path"] = str(
            (path.parent / c["protection_mask"]["path"]).resolve()
        )
    if c.get("psf_response") is not None:
        for key in ("data", "review", "input_review"):
            c["psf_response"][key] = str(
                (path.parent / c["psf_response"][key]).resolve()
            )
    return c


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def finite_record(value):
    if isinstance(value, dict):
        return {str(k): finite_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_record(v) for v in value]
    if hasattr(value, "tolist"):
        return finite_record(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def write(path, value):
    """Never replace an existing result; represent rejected support explicitly."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(finite_record(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
