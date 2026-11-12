"""End-to-end: the SAP lane."""

from types import SimpleNamespace

import numpy as np
from conftest import FIXTURES, LANE_HOST, load_lane_host, packet, planet
from conftest import inject as inject_psf_flat

from calibration import sample_builder as builder
from calibration.lane_census import array_features
from calibration.lane_twin import inject, planet_model, sap_lane, star_light
from calibration.scoring import match_signal
from common import digest, read
from detector.program import run as detect
from photometry.program import star_shares
from vetter.policy import assess, features, score_lane
from vetter.program import run as vet

PERIOD = 70.0
# 2.5 Earth radii: S/N 12 in this SAP (2 Earth radii sit at the detection threshold).
RADIUS = 2.5


def sap_host():
    """The lane host's SAP twin."""
    record = read(FIXTURES / f"{LANE_HOST}-sap.json")
    assert digest(FIXTURES / f"{LANE_HOST}-sap.npz") == record["sha256"]
    with np.load(FIXTURES / f"{LANE_HOST}-sap.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    inputs = record["star_share_inputs"]
    quarters = {int(q): v for q, v in inputs["quarters"].items()}
    arrays["star_share"], shares = star_shares(
        arrays["quarter"], inputs["g_mag"], inputs["bp_rp"], quarters
    )
    lane = load_lane_host()
    provenance = {"fixture": record["sha256"], "star_share": shares}
    prepared, raw, metadata = sap_lane(lane.metadata, arrays, provenance)
    features = array_features(prepared, raw)
    host = lane.host | {"features": features}
    host["star_light"] = star_light(host, metadata, prepared)
    return lane, SimpleNamespace(
        host=host, stellar=lane.stellar, prepared=prepared, raw=raw, metadata=metadata,
        features=features, arrays=arrays, provenance=provenance,
    )  # fmt: skip


def test_sap_lane_finds_the_long_transit_planet(tmp_path, config):
    lane, sap = sap_host()
    assert sap.metadata["view"] == "sap_flat"
    assert sorted(sap.metadata["normalization_by_quarter"]) == ["2", "3", "4", "6"]
    good = sap.prepared["mask"]
    assert abs(np.median(sap.prepared["flux"][good]) - 1) < 1e-3
    shares = sap.metadata["star_share_by_quarter"]
    assert sorted(shares) == ["2", "3", "4", "6"]
    assert 0.8 < min(shares.values()) < max(shares.values()) < 0.96
    body = planet(lane, PERIOD, RADIUS, 15.0)
    arrays, _, metadata, (stats,) = inject(
        lane.metadata, sap.arrays, sap.provenance, sap.prepared, sap.features,
        [body], planet_model, "sap-lane-test",
    )  # fmt: skip
    signal = (
        body | stats | builder.detectability(body, stats, sap.host["catalogue_stellar"])
    )
    assert signal["detectable"] and signal["duration_days"] > 0.3
    # Decrowding gives back the depth; against the local baseline only the flattening's loss remains.
    assert 0.96 < signal["local_transfer"] < 1.02
    time = arrays["time"][arrays["mask"]]
    start, baseline = float(time.min()), float(np.ptp(time))
    prepared = packet(tmp_path / "prepared", arrays, metadata)
    out = tmp_path / "detections"
    out.mkdir()
    found = detect(SimpleNamespace(prepared=prepared, out=out), config)["candidates"]
    outcomes = [match_signal(c, signal, start, baseline)[0] for c in found]
    assert "exact" in outcomes, found
    # PSF Flat holds the same planet at each candidate's ephemeris (vetter/cross_lane.py).
    flat_arrays, flat_metadata, _ = inject_psf_flat(lane, [body])
    other = packet(tmp_path / "psf-flat", flat_arrays, flat_metadata)
    vetting = tmp_path / "vetting"
    vetting.mkdir()
    metrics_only = config | {"vetting": config["vetting"] | {"policy": None}}
    args = SimpleNamespace(
        prepared=prepared, detections=out, out=vetting, cross_lane=other
    )
    results = vet(args, metrics_only)["results"]
    metrics = results[outcomes.index("exact")]["metrics"]
    assert metrics["cross_lane_events"] >= 3
    assert metrics["cross_lane_mes"] > 0.5 * metrics["max_mes"]
    assert score_lane("sap_flat", features(metrics)) == "sap_flat_with_psf_flat"
    assert assess(metrics, config["vetting"]["policy"], "sap_flat")["state"] == "KEPT"
