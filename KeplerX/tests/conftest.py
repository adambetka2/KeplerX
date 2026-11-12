"""Fixtures shared by the golden and end-to-end tests."""

import os

# One BLAS thread before numpy's import: the golden arrays are exact only in that order.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration import population as pop
from calibration import sample_builder as builder
from calibration.lane_census import array_features
from common import (
    ROOT,
    config_file,
    digest,
    json_value,
    read,
    receipt,
    save_packet,
    write,
)
from photometry.program import prepare_arrays

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/cache/matplotlib"))
FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
LANE_HOST = "lane-k-dwarf-2103903753019155968-four-quarters"


@pytest.fixture
def config():
    return config_file()


@pytest.fixture
def fixtures():
    return FIXTURES


def lane_host_record():
    record = read(FIXTURES / f"{LANE_HOST}.json")
    assert digest(FIXTURES / f"{LANE_HOST}.npz") == record["sha256"]
    return record


@functools.lru_cache
def load_lane_host(quarters=None):
    """The quiet lane K dwarf under production preparation (only `quarters` if given)."""
    record = lane_host_record()
    with np.load(FIXTURES / f"{LANE_HOST}.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    if quarters is not None:
        keep = np.isin(arrays["quarter"], quarters)
        arrays = {name: values[keep] for name, values in arrays.items()}
    host = {
        "source_id": record["source_id"], "kic": record["kic"], "parent_kic": record["origin_tpf_kic"],
        "ra_deg": record["ra_deg"], "dec_deg": record["dec_deg"], "g_mag": record["phot_g_mean_mag"],
        "bp_rp": record["bp_rp"], "teff_gspphot": record["teff_gspphot"],
        "logg_gspphot": record["logg_gspphot"], "population": "K", "tpf_relation": "none",
        "family": str(record["origin_tpf_kic"]), "partition": "development", "host_weight": 1.0,
        "known_signals": [],
    }  # fmt: skip
    stellar = pop.catalogue_stellar(record["teff_gspphot"], record["logg_gspphot"])
    provenance = {"fixture": f"{LANE_HOST}.npz", "sha256": record["sha256"]}
    prepared, raw, metadata = builder.prepare(arrays, host, stellar, provenance)
    features = array_features(prepared, raw)
    light = builder.star_light(host["g_mag"], host["bp_rp"], prepared, metadata)
    host |= {"catalogue_stellar": stellar, "features": features, "star_light": light}
    return SimpleNamespace(
        host=host, stellar=stellar, prepared=prepared, raw=raw, metadata=metadata,
        features=features,
    )  # fmt: skip


def planet(lane, period, radius, epoch_offset, b=0.2):
    """A circular planet on the lane host's catalogue star, at a fixed epoch."""
    star = lane.stellar
    ld = pop.limb_darkening(star["teff"], star["logg"], builder.ld_interpolators())
    body = pop.planet(
        star, ld, None, period=period, radius=radius, e=0.0, omega=0.0, b=b
    )
    body["epoch_bkjd"] = lane.metadata["first_valid_bkjd"] + epoch_offset
    body["ttv"] = None
    return body | {"role": "primary"}


def inject(lane, bodies):
    """The lane host with `bodies` injected by the calibration builder."""
    arrays, metadata, signals, _ = builder.inject_system(
        lane.host, lane.prepared, lane.raw, lane.metadata, lane.features, bodies,
        lambda body: {"population_weight": 1.0},
    )  # fmt: skip
    return arrays, metadata, signals


def packet(folder, arrays, metadata):
    folder.mkdir(parents=True)
    save_packet(folder, arrays, metadata)
    return folder


def tres2_packet(folder, config):
    """TrES-2's quarter-0 KBonus PSF Flat under production preparation."""
    info = read(FIXTURES / "tres2-q0.json")
    assert digest(FIXTURES / "tres2-q0.npz") == info["fixture_sha256"]
    with np.load(FIXTURES / "tres2-q0.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays, scales = prepare_arrays(arrays, config["photometry"])
    metadata = {
        "source_id": info["source_id"],
        "exposure_days": info["exposure_days"],
        "stellar": {
            "teff": 5819.0, "logg": 4.4, "density_solar": 1.0,
            "basis": "Broad solar template for the known-object software control; no stellar inference or validation claim",
        },
        "normalization_by_quarter": scales,
    }  # fmt: skip
    return packet(folder, arrays, metadata), info


def kic12644822(folder, monkeypatch):
    """KIC 12644822, host of KOI K00791.01."""
    from localizator import gaia_cone_search

    record = read(FIXTURES / "kic12644822-q1-prepared.json")
    assert digest(FIXTURES / "kic12644822-q1-prepared.npz") == record["sha256"]
    with np.load(FIXTURES / "kic12644822-q1-prepared.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    prepared = packet(folder / "prepared", arrays, record["packet"])
    source = record["packet"]["source_id"]
    clock = record["clock"]
    detections = folder / "detections"
    write(detections / "detections.json", {
        "source_id": source, "prepared": receipt(prepared / "packet.json"),
        "candidates": [clock | {"mes": 20.0, "source_id": source, "star_id": 0}],
        "engine": "KOI catalogue clock",
    })  # fmt: skip
    pixels = FIXTURES / "kic12644822-q1-target-pixels.fits.gz"
    request = folder / "request.json"
    write(request, {
        "source_id": source, "parent_kic": 12644822, "ra_deg": record["packet"]["ra_deg"],
        "dec_deg": record["packet"]["dec_deg"], "coordinate_epoch": "J2016",
        "tpfs": [{"path": str(pixels), "bytes": pixels.stat().st_size, "sha256": digest(pixels), "quarter": 1}],
        "missing_quarters": [],
    })  # fmt: skip
    monkeypatch.setenv("KEPLERX_KERNEL_CACHE", str(folder / "cache"))
    stored = (FIXTURES / "gaia-dr3-12644822-esa.vot").read_bytes()
    monkeypatch.setattr(gaia_cone_search, "fetch", lambda *args, **kwargs: stored)
    return SimpleNamespace(
        record=record, prepared=prepared, detections=detections, request=request
    )


def array_digests(arrays):
    """SHA-256 of each array's bytes: a golden record of exact values."""
    return {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in sorted(arrays.items())
    }


def golden(name, record):
    """Compare `record` with tests/golden/NAME.json."""
    path = GOLDEN / f"{name}.json"
    record = json.loads(json.dumps(json_value(record)))
    if os.environ.get("KEPLERX_WRITE_GOLDEN") == "1":
        write(path, record)
    assert path.exists(), (
        f"No golden record {path.name}; write it with KEPLERX_WRITE_GOLDEN=1"
    )
    assert record == read(path)
