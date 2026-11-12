"""Compare stored difference images with PRFs at Gaia source positions."""

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from kepler_prf import KeplerPRF
from vetting import count

# The calibrated Kepler PRF footprint spans at most 15 detector pixels per axis.
PIXEL_SUPPORT_MARGIN = 7.5


@lru_cache(maxsize=128)
def file_sha256(path, size, modified_ns):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_file(path, expected_sha256):
    path = Path(path)
    stat = path.stat()
    if file_sha256(str(path), stat.st_size, stat.st_mtime_ns) != expected_sha256:
        raise ValueError(f"Pixel or PRF input hash changed: {path}")
    return path


@lru_cache(maxsize=64)
def _pixel_headers(path, size, modified_ns, expected_sha256):
    path = verified_file(path, expected_sha256)
    count(pixel_files_opened=1, pixel_bytes_opened=size)
    with fits.open(path, memmap=False) as hdus:
        return tuple(hdus[index].header.copy() for index in range(3))


def pixel_headers(path, expected_sha256):
    path = Path(path)
    stat = path.stat()
    headers = _pixel_headers(str(path), stat.st_size, stat.st_mtime_ns, expected_sha256)
    return tuple(header.copy() for header in headers)


@lru_cache(maxsize=2)
def calibrated_prf(channel, filename, sha256, cache):
    verified_file(Path(cache) / filename, sha256)
    return KeplerPRF(channel=channel, cache_dir=str(cache) + "/")


def fixed_source_fit(profile, image, error):
    profile = np.asarray(profile, dtype=float)
    image = np.asarray(image, dtype=float)
    error = np.asarray(error, dtype=float)
    if (
        profile.shape != image.shape
        or image.shape != error.shape
        or not np.isfinite(profile).all()
        or not np.isfinite(image).all()
        or not np.isfinite(error).all()
        or np.any(error <= 0)
    ):
        raise ValueError("Difference image or fixed-source PRF is invalid")
    weight = 1 / error**2
    norm = float(np.sum(weight * profile**2))
    if norm <= 0 or not np.isfinite(norm):
        return None
    amplitude = max(0.0, float(np.sum(weight * profile * image) / norm))
    return {
        "amplitude": amplitude,
        "chi2": float(np.sum(weight * (image - amplitude * profile) ** 2)),
    }


def gaia_position(source, epoch_bkjd):
    if source["pmra"] is None or source["pmdec"] is None:
        return float(source["ra"]), float(source["dec"]), "CATALOGUE_EPOCH_ONLY"
    coordinate = SkyCoord(
        ra=float(source["ra"]) * u.deg,
        dec=float(source["dec"]) * u.deg,
        pm_ra_cosdec=float(source["pmra"]) * u.mas / u.yr,
        pm_dec=float(source["pmdec"]) * u.mas / u.yr,
        obstime=Time(2016.0, format="jyear"),
    )
    observed = coordinate.apply_space_motion(
        new_obstime=Time(epoch_bkjd + 2454833, format="jd")
    )
    return float(observed.ra.deg), float(observed.dec.deg), "PROPAGATED_GAIA_POSITION"


def supported_pixel(x, y, shape):
    return (
        np.isfinite(x)
        and np.isfinite(y)
        and -PIXEL_SUPPORT_MARGIN <= x <= shape[1] - 1 + PIXEL_SUPPORT_MARGIN
        and -PIXEL_SUPPORT_MARGIN <= y <= shape[0] - 1 + PIXEL_SUPPORT_MARGIN
    )


def scene_event(record, event_id, quarter, source_id, scene, prf_cache):
    evidence = record["pixel_evidence"]
    mask = np.asarray(evidence["good_pixel_mask"], dtype=bool)
    image = np.asarray(evidence["difference_amplitude"], dtype=float)
    error = np.asarray(evidence["difference_error"], dtype=float)
    if mask.ndim != 2 or len(image) != int(mask.sum()):
        raise ValueError("Difference-image mask disagrees with its retained pixels")
    primary, table, aperture = pixel_headers(record["tpf_path"], record["tpf_sha256"])
    if int(primary["QUARTER"]) != quarter:
        raise ValueError(f"Pixel quarter differs from selected event: {event_id}")
    channel = int(primary["CHANNEL"])
    origin = (int(table["2CRV4P"]), int(table["1CRV4P"]))
    wcs = WCS(aperture)
    calibration = record["prf_calibration"]
    prf = calibrated_prf(
        channel, calibration["filename"], calibration["sha256"], str(prf_cache)
    )
    sources = [(source_id, record["target_pixel"], "PROPAGATED_GAIA_POSITION")]
    for source in scene["sources"]:
        ra, dec, astrometry = gaia_position(source, record["observation_epoch_bkjd"])
        x, y = wcs.world_to_pixel_values(ra, dec)
        if supported_pixel(x, y, mask.shape):
            sources.append((source["source_id"], (x, y), astrometry))
    models = []
    for sid, (x, y), astrometry in sources:
        profile = prf.evaluate(
            targets=[(origin[0] + y, origin[1] + x)],
            origin=origin,
            shape=mask.shape,
        )[0]
        fit = fixed_source_fit(profile[mask], image, error)
        if fit is not None:
            models.append(
                {
                    "source_id": str(sid),
                    "astrometry": astrometry,
                    "pixel": [float(x), float(y)],
                    "captured_prf_fraction": float(profile.sum()),
                    **fit,
                }
            )
    target = next((model for model in models if model["source_id"] == source_id), None)
    if target is None:
        raise ValueError(f"Target PRF has no pixel support: {event_id}")
    alternatives = [model for model in models if model["source_id"] != source_id]
    best = min(alternatives, key=lambda model: model["chi2"]) if alternatives else None
    return {
        "event_id": event_id,
        "quarter": quarter,
        "status": "MEASURED" if best else "NO_MODELLED_ALTERNATIVE",
        "catalogue_epoch_only_source_ids": [
            model["source_id"]
            for model in alternatives
            if model["astrometry"] == "CATALOGUE_EPOCH_ONLY"
        ],
        "pixel_count": len(image),
        "tpf_sha256": record["tpf_sha256"],
        "prf_sha256": calibration["sha256"],
        "target_chi2": target["chi2"],
        "best_alternative_source_id": best["source_id"] if best else None,
        "best_alternative_chi2": best["chi2"] if best else None,
        "target_advantage_chi2": best["chi2"] - target["chi2"] if best else None,
        "models": models,
    }


def compare_signal(row, good_quarters, run, prf_cache, event_cache=None):
    scene = row.get("gaia_scene", {})
    if scene.get("status") != "GAIA_SCENE_QUERIED":
        return {"status": "UNAVAILABLE_GAIA_SCENE"}
    if not good_quarters:
        return {"status": "UNAVAILABLE_GOOD_PIXEL_QUARTERS"}
    if event_cache is None:
        event_cache = {}
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    events = []
    for quarter, records in sorted(
        good_quarters.items(), key=lambda item: int(item[0])
    ):
        selected = max(records.values(), key=lambda record: record["prf_correlation"])
        event_id = selected["event_id"]
        if event_id not in event_cache:
            path = (
                Path(run)
                / "localizations-final"
                / f"{row['source_id']}-{event_id}.json"
            )
            record = json.loads(path.read_text())
            if record.get("target_astrometry") != "PROPAGATED_GAIA_POSITION":
                return {
                    "status": "UNAVAILABLE_TARGET_ASTROMETRY",
                    "event_id": event_id,
                }
            event_cache[event_id] = scene_event(
                record, event_id, int(quarter), row["source_id"], scene, prf_cache
            )
        events.append(event_cache[event_id])
    margins = [
        event["target_advantage_chi2"]
        for event in events
        if event["target_advantage_chi2"] is not None
    ]
    return {
        "status": "MEASURED_DIAGNOSTIC" if margins else "NO_MODELLED_ALTERNATIVES",
        "selection": "One event with the highest free-PRF correlation per quality-qualified quarter",
        "model": "Fixed Gaia-source Kepler PRF with a nonnegative amplitude fitted to the retained difference-image pixels using formal errors. Uncalibrated Gaussian residual comparison, not a source probability or disposition.",
        "compared_quarters": len(margins),
        "target_preferred_quarters": sum(margin > 0 for margin in margins),
        "median_target_advantage_chi2": float(np.median(margins)) if margins else None,
        "catalogue_epoch_only_source_ids": sorted(
            {
                source_id
                for event in events
                for source_id in event["catalogue_epoch_only_source_ids"]
            }
        ),
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "events": events,
    }
