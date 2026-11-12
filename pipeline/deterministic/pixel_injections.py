"""Materialize frozen on-target physical transits in Kepler target pixels."""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

from deterministic import injections
from lc2 import load_source
from vetting_pixels import (
    PIXELS,
    archive_filenames,
    download_pixel,
    ensure_kepler_prf,
    indexed_pixels,
    load_kepler_prf,
)

PARTITIONS = ROOT / "dump/release-planet-control-partitions-v1"
BASE_INJECTIONS = ROOT / "dump/release-injections-v1"
TPF_DOWNLOAD_WORKERS = 8
PRF_CACHE = ROOT / "dump/prf-cache"
GAIA_ENDPOINT = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_COLUMNS = ("source_id", "ra", "dec", "pmra", "pmdec")


class MastHTTPAdapter(requests.adapters.HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, 60)
        return super().send(request, **kwargs)


class PixelUnavailable(ValueError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_inputs(partitions, base):
    freeze = read(partitions / "freeze.json")
    if freeze.get("status") != "FROZEN_PLANET_CONTROL_PARTITIONS":
        raise ValueError("Planet-control partitions are not frozen")
    for name, expected in freeze["artifacts"].items():
        if digest(partitions / name) != expected:
            raise ValueError(f"Frozen planet-control artifact changed: {name}")
    materialization = read(base / "materialization.json")
    if materialization.get("status") != "FROZEN":
        raise ValueError("Base flux injections are not frozen")
    for name in ("input-manifest.json", "truth.jsonl"):
        if digest(base / name) != materialization["artifacts"][name]:
            raise ValueError(f"Base flux-injection artifact changed: {name}")
    manifest = read(base / "input-manifest.json")
    if (
        manifest.get("schema") != 2
        or manifest.get("transform") != "additive_flux_delta"
    ):
        raise ValueError("Unsupported base injection manifest")
    return freeze, materialization, manifest


def partition_rows(partitions, partition, source_id=None, category=None, limit=None):
    path = partitions / f"{partition}-truth.jsonl"
    if not path.exists():
        raise ValueError(f"Unknown planet-control partition: {partition}")
    rows = read_jsonl(path)
    if source_id is not None:
        rows = [row for row in rows if row["source_id"] == str(source_id)]
        if len(rows) != 1:
            raise ValueError("Probe source is not unique in the selected partition")
    if category is not None:
        rows = [row for row in rows if row["category"] == category]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No planet controls selected")
    return rows


def gaia_query(source_ids):
    identifiers = ",".join(sorted(source_ids, key=int))
    return (
        "select source_id,ra,dec,pmra,pmdec from gaiadr3.gaia_source "
        f"where source_id in ({identifiers})"
    )


def query_gaia(query, attempts=3):
    last_error = None
    for _ in range(attempts):
        try:
            response = requests.get(
                GAIA_ENDPOINT,
                params={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "json",
                    "QUERY": query,
                },
                timeout=(10, 45),
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise OSError(
        f"Gaia astrometry query failed after {attempts} attempts: {last_error}"
    )


def gaia_astrometry(path, source_ids):
    query = gaia_query(source_ids)
    if path.exists():
        result = read(path)
        if result.get("endpoint") != GAIA_ENDPOINT or result.get("query") != query:
            raise ValueError("Cached Gaia query does not match this materialization")
    else:
        payload = query_gaia(query)
        names = [column["name"] for column in payload["metadata"]]
        result = {
            "endpoint": GAIA_ENDPOINT,
            "query": query,
            "status": "COMPLETE",
            "rows": sorted(
                (dict(zip(names, row)) for row in payload["data"]),
                key=lambda row: int(row["source_id"]),
            ),
        }
        write_json(path, result)
    if result.get("status") != "COMPLETE":
        raise ValueError("Gaia astrometry query is incomplete")
    rows = {str(row["source_id"]): row for row in result["rows"]}
    if (
        len(result["rows"]) != len(source_ids)
        or set(rows) != set(source_ids)
        or any(len(row) != len(GAIA_COLUMNS) for row in rows.values())
    ):
        raise ValueError("Gaia astrometry identities are incomplete or duplicated")
    return rows


def source_flux(raw, quarter):
    quarter_mask = np.asarray(raw["quarter"]) == quarter
    flux = np.asarray(raw[injections.PRIMARY_FLUX], dtype=np.float64)
    usable = quarter_mask & np.isfinite(flux)
    if not usable.any():
        raise ValueError(f"No finite source flux in quarter {quarter}")
    value = float(np.median(flux[usable]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Source flux is not positive in quarter {quarter}")
    return value


def propagated_position(astrometry, epoch_jd):
    if astrometry["pmra"] is None or astrometry["pmdec"] is None:
        return (
            float(astrometry["ra"]),
            float(astrometry["dec"]),
            "CATALOGUE_POSITION_UNPROPAGATED",
        )
    coordinate = SkyCoord(
        ra=float(astrometry["ra"]) * u.deg,
        dec=float(astrometry["dec"]) * u.deg,
        pm_ra_cosdec=float(astrometry["pmra"]) * u.mas / u.yr,
        pm_dec=float(astrometry["pmdec"]) * u.mas / u.yr,
        obstime=Time(2016.0, format="jyear"),
    )
    observed = coordinate.apply_space_motion(new_obstime=Time(epoch_jd, format="jd"))
    return float(observed.ra.deg), float(observed.dec.deg), "PROPAGATED_GAIA_POSITION"


def pixel_profile(hdus, astrometry, cache=PRF_CACHE):
    header = hdus[0].header
    table_header = hdus[1].header
    time = np.asarray(hdus[1].data["TIME"], dtype=np.float64)
    ra, dec, status = propagated_position(astrometry, np.nanmean(time) + 2454833.0)
    x, y = WCS(hdus[2].header).world_to_pixel_values(ra, dec)
    shape = hdus[1].data["FLUX"].shape[1:]
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError("Injected Gaia source position is nonfinite")
    inside = bool(0 <= x < shape[1] and 0 <= y < shape[0])
    channel = int(header["CHANNEL"])
    prf, calibration = load_kepler_prf(channel, cache)
    origin = (int(table_header["2CRV4P"]), int(table_header["1CRV4P"]))
    profile = prf.evaluate(
        targets=[(origin[0] + y, origin[1] + x)],
        origin=origin,
        shape=shape,
    )[0]
    if profile.shape != shape or not np.isfinite(profile).all() or profile.sum() <= 0:
        raise ValueError("Kepler PRF evaluation is invalid")
    return profile, calibration, [float(x), float(y)], [ra, dec], status, inside


def transit_loss(times, truth, signal):
    finite = np.asarray(times, dtype=np.float64)
    geometry = (
        truth["radius_ratio"],
        truth["scaled_semimajor_axis"],
        truth["inclination_degrees"],
        truth["duration_hours"] / 24.0,
    )
    centres = injections.truth_centres(
        truth, float(np.nanmin(finite)), float(np.nanmax(finite))
    )
    loss = np.zeros(len(finite), dtype=np.float64)
    valid = np.isfinite(finite)
    loss[valid] = injections.transit_loss(
        finite[valid],
        centres,
        truth["shape_period_days"],
        geometry,
        signal,
        truth["epoch_bkjd"] if truth["category"] == "periodic_injection" else None,
    )
    return loss


def inject_flux(flux, profile, loss, amplitude):
    flux = np.asarray(flux)
    delta = amplitude * loss[:, None, None] * profile[None, :, :]
    return (flux.astype(np.float64) - delta).astype(flux.dtype)


def deterministic_fits(path, hdus):
    checksum_comment = "Forgotten Stars deterministic checksum"
    for hdu in hdus:
        hdu.add_datasum(when=checksum_comment)
        hdu.add_checksum(when=checksum_comment, override_datasum=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream,
    ):
        hdus.writeto(stream)
    temporary.replace(path)


def archive_download_plan(filenames, quarters, mapping):
    selected = {}
    for filename in filenames:
        timestamp = filename.split("-")[1].split("_")[0]
        quarter = mapping.get(timestamp)
        if quarter in quarters:
            selected.setdefault(quarter, filename)
    missing = sorted(set(quarters) - set(selected))
    if missing:
        raise ValueError(f"Archive listing cannot resolve quarters {missing}")
    return sorted(selected.items())


def download_tpfs(base, parent, quarters, mapping, filenames=None):
    filenames = archive_filenames(base) if filenames is None else filenames
    jobs = archive_download_plan(filenames, quarters, mapping)

    def download(job):
        quarter, filename = job
        path = download_pixel(base, filename, parent)
        with fits.open(path, memmap=False) as hdus:
            if int(hdus[0].header["QUARTER"]) != quarter:
                raise ValueError("Downloaded target-pixel quarter changed")
        return quarter, path

    with ThreadPoolExecutor(max_workers=min(TPF_DOWNLOAD_WORKERS, len(jobs))) as pool:
        return dict(pool.map(download, jobs))


def mast_spatial_tpf_query(path, source, mapping, attempts=3):
    query = {
        "coordinates_deg": [source["ra_deg"], source["dec_deg"]],
        "radius_deg": 1 / 60,
        "obs_collection": "Kepler",
        "dataproduct_type": "timeseries",
    }
    if path.exists():
        result = read(path)
        if result.get("query") != query:
            raise ValueError("Cached MAST spatial query does not match this source")
        return result

    from astroquery.mast import Observations

    connection = Observations._portal_api_connection
    connection.TIMEOUT = 60
    endpoint = urlsplit(connection.MAST_REQUEST_URL)
    connection._session.mount(
        f"{endpoint.scheme}://{endpoint.netloc}/", MastHTTPAdapter()
    )
    last_error = None
    for _ in range(attempts):
        try:
            observations = Observations.query_criteria(
                coordinates=f"{source['ra_deg']} {source['dec_deg']}",
                radius=f"{query['radius_deg']} deg",
                obs_collection="Kepler",
                dataproduct_type="timeseries",
            )
            if len(observations) == 0:
                raise PixelUnavailable(
                    "MAST spatial query returned no Kepler long-cadence TPFs"
                )
            products = Observations.get_product_list(observations)
            break
        except (requests.RequestException, TimeoutError, ConnectionError) as exc:
            last_error = exc
    else:
        raise OSError(f"MAST spatial TPF query failed: {last_error}")

    distances = {str(row["obsid"]): float(row["distance"]) for row in observations}
    rows = []
    for row in products:
        filename = str(row["productFilename"])
        match = re.fullmatch(r"kplr(\d+)-(\d+)_lpd-targ\.fits\.gz", filename)
        if match is None or match[2] not in mapping:
            continue
        rows.append(
            {
                "filename": filename,
                "parent_kic": int(match[1]),
                "quarter": mapping[match[2]],
                "distance_arcsec": distances[str(row["obsID"])],
                "data_uri": str(row["dataURI"]),
            }
        )
    result = {
        "endpoint": "MAST Portal API",
        "query": query,
        "products": sorted(
            rows,
            key=lambda row: (
                row["quarter"],
                row["distance_arcsec"],
                row["filename"],
            ),
        ),
    }
    if not result["products"]:
        raise PixelUnavailable(
            "MAST spatial query returned no Kepler long-cadence TPFs"
        )
    write_json(path, result)
    return result


def stamp_distance(path, astrometry):
    with fits.open(path, memmap=False) as hdus:
        time = np.asarray(hdus[1].data["TIME"], dtype=np.float64)
        ra, dec, _ = propagated_position(astrometry, np.nanmean(time) + 2454833.0)
        x, y = WCS(hdus[2].header).world_to_pixel_values(ra, dec)
        height, width = hdus[1].data["FLUX"].shape[1:]
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.inf
    dx = max(-x, 0.0, x - (width - 1))
    dy = max(-y, 0.0, y - (height - 1))
    return float(np.hypot(dx, dy))


def spatial_tpfs(query_path, source, astrometry, quarters, mapping):
    result = mast_spatial_tpf_query(query_path, source, mapping)
    remaining = {
        quarter: [row for row in result["products"] if row["quarter"] == quarter]
        for quarter in sorted(quarters)
    }
    absent = [quarter for quarter, rows in remaining.items() if not rows]
    if absent:
        raise PixelUnavailable(f"MAST spatial query lacks quarters {absent}")
    selected = {}

    def download(job):
        quarter, candidate = job
        parent = candidate["parent_kic"]
        name = f"{parent:09d}"
        base = (
            "https://archive.stsci.edu/pub/kepler/target_pixel_files/"
            f"{name[:4]}/{name}/"
        )
        path = download_pixel(base, candidate["filename"], parent)
        with fits.open(path, memmap=False) as hdus:
            if int(hdus[0].header["QUARTER"]) != quarter:
                raise ValueError("Spatial TPF quarter changed")
        return quarter, path

    with ThreadPoolExecutor(
        max_workers=min(TPF_DOWNLOAD_WORKERS, len(remaining))
    ) as pool:
        while remaining:
            jobs = [(quarter, rows.pop(0)) for quarter, rows in remaining.items()]
            for quarter, path in pool.map(download, jobs):
                if stamp_distance(path, astrometry) <= 2.0:
                    selected[quarter] = path
            remaining = {
                quarter: rows
                for quarter, rows in remaining.items()
                if quarter not in selected
            }
            exhausted = [quarter for quarter, rows in remaining.items() if not rows]
            if exhausted:
                raise PixelUnavailable(
                    f"No spatial TPF supports source in quarters {exhausted}"
                )
    return selected


def available_tpfs(source, astrometry, quarters, mapping, existing, query_path):
    parent = int(source["origin_tpf_kic"])
    result = {"quarters": {}}
    for quarter in quarters:
        path = existing.get((parent, quarter))
        if path and stamp_distance(path, astrometry) <= 2.0:
            result["quarters"][str(quarter)] = {"path": path, "status": "AVAILABLE"}
    if len(result["quarters"]) != len(quarters):
        name = f"{parent:09d}"
        base = f"https://archive.stsci.edu/pub/kepler/target_pixel_files/{name[:4]}/{name}/"
        missing = set(quarters) - set(map(int, result["quarters"]))
        filenames = archive_filenames(base)
        listed = {
            mapping[filename.split("-")[1].split("_")[0]]
            for filename in filenames
            if filename.split("-")[1].split("_")[0] in mapping
        }
        origin_quarters = missing & listed
        if origin_quarters:
            for quarter, path in download_tpfs(
                base, parent, origin_quarters, mapping, filenames
            ).items():
                if stamp_distance(path, astrometry) <= 2.0:
                    result["quarters"][str(quarter)] = {
                        "path": str(path),
                        "status": "AVAILABLE",
                    }
    missing = set(quarters) - set(map(int, result["quarters"]))
    if missing:
        for quarter, path in spatial_tpfs(
            query_path, source, astrometry, missing, mapping
        ).items():
            result["quarters"][str(quarter)] = {
                "path": str(path),
                "status": "AVAILABLE_SPATIAL_PARENT",
            }
    missing = sorted(set(map(str, quarters)) - set(result["quarters"]))
    if missing:
        raise PixelUnavailable(
            f"Target pixels unavailable for parent {parent}, quarters {missing}"
        )
    return {
        int(quarter): Path(item["path"]) for quarter, item in result["quarters"].items()
    }


def prefetch_prfs(channels, cache=PRF_CACHE):
    channels = sorted(set(channels))
    if not channels:
        return

    def ensure(channel):
        ensure_kepler_prf(channel, cache)

    with ThreadPoolExecutor(
        max_workers=min(TPF_DOWNLOAD_WORKERS, len(channels))
    ) as pool:
        list(pool.map(ensure, channels))


def inject_tpf(base_path, target, source, truth, astrometry, signal, quarter):
    base_hash = digest(base_path)
    with fits.open(base_path, memmap=False) as hdus:
        hdus.verify("exception")
        parent = int(hdus[0].header["KEPLERID"])
        match = re.match(r"kplr(\d+)-", Path(base_path).name)
        if match is not None and int(match[1]) != parent:
            raise ValueError("Target-pixel parent identity changed")
        if int(hdus[0].header["QUARTER"]) != quarter:
            raise ValueError("Target-pixel quarter identity changed")
        profile, calibration, pixel, sky, astrometry_status, target_inside = (
            pixel_profile(hdus, astrometry)
        )
        times = np.asarray(hdus[1].data["TIME"], dtype=np.float64)
        loss = transit_loss(times, truth, signal)
        amplitude = source_flux(source["raw"], quarter)
        original_error = np.array(hdus[1].data["FLUX_ERR"], copy=True)
        hdus[1].data["FLUX"] = inject_flux(
            hdus[1].data["FLUX"], profile, loss, amplitude
        )
        if not np.array_equal(hdus[1].data["FLUX_ERR"], original_error, equal_nan=True):
            raise ValueError("Pixel uncertainties changed during injection")
        hdus[0].header["FSCSRC"] = (str(truth["source_id"]), "Injected Gaia DR3 source")
        hdus[0].header["FSCMODEL"] = ("physical-prf-v1", "Forgotten Stars control")
        deterministic_fits(target, hdus)
    with fits.open(target, memmap=False, checksum=True) as hdus:
        hdus.verify("exception")
    return {
        "path": str(target),
        "sha256": digest(target),
        "base_path": str(base_path),
        "base_parent_kic": parent,
        "base_sha256": base_hash,
        "prf_calibration": calibration,
        "target_pixel": pixel,
        "target_inside_stamp": target_inside,
        "target_sky_deg": sky,
        "target_astrometry": astrometry_status,
        "source_flux_electrons_per_second": amplitude,
        "maximum_relative_loss": float(loss.max()),
        "prf_stamp_fraction": float(profile.sum()),
    }


def generator_identity():
    paths = (
        Path(__file__),
        Path(injections.__file__),
        ROOT / "pipeline/vetting/vetting_pixels.py",
        ROOT / "pipeline/vetting/requirements.txt",
    )
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def materialization_identity(partitions, base, partition, rows):
    return {
        "partition": partition,
        "source_ids": [row["source_id"] for row in rows],
        "partition_freeze_sha256": digest(partitions / "freeze.json"),
        "base_materialization_sha256": digest(base / "materialization.json"),
        "generator": generator_identity(),
    }


def validate_record(record, out, source_id, quarters):
    if record.get("source_id") != source_id or set(record.get("pixels", {})) != set(
        map(str, quarters)
    ):
        return False
    for quarter, pixel in record.get("pixels", {}).items():
        path = (out / pixel["path"]).resolve()
        if out.resolve() not in path.parents or digest(path) != pixel["sha256"]:
            return False
        with fits.open(path, memmap=False, checksum=True) as hdus:
            if str(hdus[0].header.get("FSCSRC")) != source_id or int(
                hdus[0].header["QUARTER"]
            ) != int(quarter):
                return False
    query = record.get("spatial_tpf_query")
    if query is not None:
        path = (out / query["path"]).resolve()
        if out.resolve() not in path.parents or digest(path) != query["sha256"]:
            return False
    return True


def spatial_query_provenance(path, out):
    if not path.exists():
        return None
    return {"path": os.path.relpath(path, out), "sha256": digest(path)}


def materialize(
    out,
    partition="development",
    source_id=None,
    category=None,
    limit=None,
    resume=False,
    partitions=PARTITIONS,
    base=BASE_INJECTIONS,
):
    out = Path(out)
    partitions, base = Path(partitions), Path(base)
    freeze, base_report, base_manifest = validate_inputs(partitions, base)
    rows = partition_rows(partitions, partition, source_id, category, limit)
    identity = materialization_identity(partitions, base, partition, rows)
    identity["category"] = category
    if resume:
        progress = read(out / "progress.json")
        if (
            progress.get("identity") != identity
            or (out / "materialization.json").exists()
        ):
            raise ValueError(
                "Pixel materialization resume identity changed or run is complete"
            )
    else:
        out.mkdir(parents=True, exist_ok=False)
        PIXELS.mkdir(parents=True, exist_ok=True)
        (out / "pixels").mkdir()
        (out / "records").mkdir()
        (out / "spatial-tpf-queries").mkdir()
        progress = {
            "status": "RUNNING",
            "complete": 0,
            "total": len(rows),
            "identity": identity,
        }
        write_json(out / "progress.json", progress)
    astrometry_path = out / "gaia-astrometry.json"
    astrometry = gaia_astrometry(astrometry_path, [row["source_id"] for row in rows])
    plan = read(injections.PLAN)
    mapping, existing = indexed_pixels()
    truth_rows, manifest_sources = [], {}
    for row in rows:
        sid = row["source_id"]
        record_path = out / "records" / f"{sid}.json"
        if record_path.exists():
            record = read(record_path)
            if not validate_record(record, out, sid, row["observed_quarters"]):
                raise ValueError(f"Completed pixel record changed: {sid}")
        else:
            base_item = base_manifest["sources"][sid]
            metadata, raw = load_source(ROOT, sid)
            if metadata["raw_fits_sha256"] != base_item["base_raw_fits_sha256"]:
                raise ValueError(f"Base LC2 source changed: {sid}")
            source = {**metadata, "raw": raw}
            query_path = out / "spatial-tpf-queries" / f"{sid}.json"
            tpfs = available_tpfs(
                metadata,
                astrometry[sid],
                row["observed_quarters"],
                mapping,
                existing,
                query_path,
            )
            prefetch_prfs(fits.getheader(path, 0)["CHANNEL"] for path in tpfs.values())
            pixel_records = {}
            source_folder = out / "pixels" / sid
            source_folder.mkdir(exist_ok=True)
            for quarter in row["observed_quarters"]:
                target = source_folder / f"q{quarter:02d}.fits.gz"
                if target.exists():
                    target.unlink()
                result = inject_tpf(
                    tpfs[quarter],
                    target,
                    source,
                    row,
                    astrometry[sid],
                    plan["signal"],
                    quarter,
                )
                result["path"] = os.path.relpath(target, out)
                result["base_path"] = str(tpfs[quarter])
                pixel_records[str(quarter)] = result
            record = {
                "source_id": sid,
                "pixels": pixel_records,
                "spatial_tpf_query": spatial_query_provenance(query_path, out),
            }
            write_json(record_path, record)
        base_item = dict(base_manifest["sources"][sid])
        base_item["deltas"] = os.path.relpath(base / base_item["deltas"], out)
        base_item["truth_index"] = len(truth_rows)
        base_item["pixels"] = {
            quarter: {**pixel, "path": pixel["path"]}
            for quarter, pixel in record["pixels"].items()
        }
        base_item["spatial_tpf_query"] = record.get("spatial_tpf_query")
        manifest_sources[sid] = base_item
        truth_rows.append(row)
        progress["complete"] = len(truth_rows)
        write_json(out / "progress.json", progress)
        print(
            json.dumps(
                {"source_id": sid, "complete": len(truth_rows), "total": len(rows)}
            ),
            flush=True,
        )
    truth_path = out / "truth.jsonl"
    injections.write_jsonl(truth_path, truth_rows)
    source_path = out / "source-ids.json"
    write_json(source_path, [row["source_id"] for row in truth_rows])
    manifest = {
        "schema": 2,
        "measurement_scope": (
            "On-target physical transits injected coherently into frozen LC2 "
            "flux views and Kepler target pixels. Measures detector, flux-vetting, "
            "and source-localization recovery; does not validate planets or "
            "measure astrophysical false-positive rejection."
        ),
        "transform": "additive_flux_delta",
        "protocol": freeze["protocol"],
        "generator": identity["generator"],
        "truth": {"path": truth_path.name, "sha256": digest(truth_path)},
        "gaia_astrometry": {
            "path": astrometry_path.name,
            "sha256": digest(astrometry_path),
        },
        "sources": manifest_sources,
    }
    manifest_path = out / "input-manifest.json"
    write_json(manifest_path, manifest)
    artifacts = {
        path.name: digest(path)
        for path in out.iterdir()
        if path.is_file() and path.name not in {"materialization.json", "progress.json"}
    }
    report = {
        "schema": 1,
        "status": "FROZEN" if source_id is None and limit is None else "PROBE",
        "protocol": manifest["protocol"],
        **identity,
        "partition_status": freeze["partitions"][partition]["status"],
        "base_generator": base_report["generator"],
        "sources": len(rows),
        "pixel_files": sum(len(item["pixels"]) for item in manifest_sources.values()),
        "pixel_bytes": sum(
            (out / pixel["path"]).stat().st_size
            for item in manifest_sources.values()
            for pixel in item["pixels"].values()
        ),
        "artifacts": artifacts,
        "claim_boundary": manifest["measurement_scope"],
    }
    write_json(out / "materialization.json", report)
    progress.update(status="COMPLETE")
    write_json(out / "progress.json", progress)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--partition", choices=("development", "evaluation"), default="development"
    )
    parser.add_argument("--source-id")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partitions", default=PARTITIONS, type=Path)
    parser.add_argument("--base-injections", default=BASE_INJECTIONS, type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("Pixel-control limit must be positive")
    print(
        json.dumps(
            materialize(
                args.out,
                args.partition,
                args.source_id,
                args.category,
                args.limit,
                args.resume,
                args.partitions,
                args.base_injections,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
