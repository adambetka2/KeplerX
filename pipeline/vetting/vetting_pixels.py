"""Parent-TPF retrieval and difference-image localization of transit hypotheses."""

import fcntl
import json
import re
import shutil
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from kepler_prf import KeplerPRF
from lc2 import CADENCE, input_manifest
from lmfit import Parameters, minimize
from runtime import dump_path, execution_workers
from vetting import (
    CONFIG,
    ROOT,
    RUN,
    count,
    counter_deltas,
    counter_snapshot,
    digest,
    elapsed,
    log_worker,
    query_evidence,
    read,
    save,
    started,
)
from vetting_scene_prf import compare_signal, pixel_headers

PIXELS = dump_path(
    ROOT / read(CONFIG).get("pixel_cache_directory", str(RUN / "pixels"))
)
PIXEL_TRANSFERS = BoundedSemaphore(execution_workers(read(CONFIG))["network_workers"])
ARCHIVE_PIXEL_ROOT = "https://archive.stsci.edu/pub/kepler/target_pixel_files/"
CLOUD_PIXEL_ROOT = (
    "https://stpubdata.s3.us-east-1.amazonaws.com/kepler/public/target_pixel_files/"
)


def pixel_job_active(job, active_ids):
    return any(
        event_id in active_ids
        for event_id in job.get("represented_event_ids", [job["event_id"]])
    )


def indexed_pixels():
    database = sqlite3.connect(
        f"file:{ROOT}/discSSD/data/index.sqlite?mode=ro", uri=True
    )
    rows = database.execute(
        "select volume,path,kic from files where collection='pixels'"
    ).fetchall()
    database.close()
    mapping, existing = {}, {}
    for volume, path, kic in rows:
        match = re.search(r"kplr(\d+)-(\d+)_lpd-targ.fits", path)
        if not match:
            continue
        filename = ROOT / volume / "data" / path
        if match[2] not in mapping:
            with fits.open(filename, memmap=False) as hdus:
                mapping[match[2]] = int(hdus[0].header["QUARTER"])
        quarter = mapping[match[2]]
        existing[(int(match[1]), quarter)] = str(filename)
    return mapping, existing


def archive_filenames(base):
    cloud = cloud_pixel_url(base, "")
    bucket, prefix = cloud.split(".com/", 1)
    parameters = {"list-type": "2", "prefix": prefix}
    filenames, tokens = set(), set()
    while True:
        with PIXEL_TRANSFERS:
            listing = requests.get(bucket + ".com/", params=parameters, timeout=45)
            listing.raise_for_status()
        document = ET.fromstring(listing.content)
        for item in document.findall("{*}Contents"):
            key = item.findtext("{*}Key", default="")
            filename = key.removeprefix(prefix)
            if key.startswith(prefix) and re.fullmatch(
                r"kplr\d{9}-\d+_lpd-targ\.fits\.gz", filename
            ):
                filenames.add(filename)
        if document.findtext("{*}IsTruncated") == "false":
            return sorted(filenames)
        token = document.findtext("{*}NextContinuationToken")
        if not token or token in tokens:
            raise ValueError("Incomplete or repeated S3 pixel listing page")
        tokens.add(token)
        parameters["continuation-token"] = token


def cloud_pixel_url(base, filename):
    if not base.startswith(ARCHIVE_PIXEL_ROOT):
        raise ValueError(f"Unexpected Kepler TPF archive URL: {base}")
    return CLOUD_PIXEL_ROOT + base.removeprefix(ARCHIVE_PIXEL_ROOT) + filename


def download_pixel(base, filename, parent):
    path = PIXELS / filename
    with path.with_suffix(path.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return locked_download(base, filename, parent, path)


def locked_download(base, filename, parent, path, attempts=20, deadline_seconds=60):
    if path.exists():
        count(pixel_cache_hits=1)
        return path
    if shutil.disk_usage(PIXELS).free < read(CONFIG).get(
        "pixel_minimum_free_bytes", 20 * 1024**3
    ):
        raise OSError(
            "Insufficient free space for another TPF; pixel evidence remains unresolved"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    last_error = None
    for _ in range(attempts):
        try:
            with PIXEL_TRANSFERS:
                bounded_download(
                    cloud_pixel_url(base, filename), temporary, deadline_seconds
                )
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            continue
        try:
            with fits.open(temporary, memmap=False) as hdus:
                hdus.verify("exception")
                if int(hdus[0].header["KEPLERID"]) != parent:
                    raise ValueError("Downloaded TPF identity mismatch")
            temporary.replace(path)
            count(pixel_bytes_downloaded=path.stat().st_size)
            return path
        except (OSError, ValueError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
    raise OSError(f"Kepler TPF download failed after {attempts} attempts: {last_error}")


def archive_candidates(filenames, quarters, mapping, selected):
    listed = [
        (filename, mapping.get(filename.split("-")[1].split("_")[0]))
        for filename in filenames
    ]
    if any(quarter is None for _, quarter in listed):
        return filenames, False
    candidates = [
        (filename, quarter)
        for filename, quarter in listed
        if quarter in quarters and str(quarter) not in selected
    ]
    if len({quarter for _, quarter in candidates}) != len(candidates):
        return filenames, False
    return [filename for filename, _ in candidates], True


def add_archive_pixels(result, base, parent, quarters, mapping):
    filenames = archive_filenames(base)
    candidates, known = archive_candidates(
        filenames, quarters, mapping, result["quarters"]
    )
    if known and len(candidates) > 1:
        original = dict(result["quarters"])
        workers = min(
            execution_workers(read(CONFIG))["network_workers"], len(candidates)
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = pool.map(
                lambda filename: download_pixel(base, filename, parent), candidates
            )
            for filename, path in zip(candidates, paths):
                quarter = add_archive_pixel(result, base, filename, path, quarters)
                timestamp = filename.split("-")[1].split("_")[0]
                if quarter != mapping[timestamp]:
                    result["quarters"] = original
                    result["index_disagreement"] = {
                        "filename": filename,
                        "indexed_quarter": mapping[timestamp],
                        "fits_quarter": quarter,
                    }
                    break
                if len(result["quarters"]) == len(quarters):
                    return
            else:
                return
    for filename in filenames:
        if len(result["quarters"]) == len(quarters):
            break
        timestamp = filename.split("-")[1].split("_")[0]
        known_quarter = mapping.get(timestamp)
        if known_quarter is not None and (
            known_quarter not in quarters or str(known_quarter) in result["quarters"]
        ):
            continue
        path = download_pixel(base, filename, parent)
        add_archive_pixel(result, base, filename, path, quarters)


def add_archive_pixel(result, base, filename, path, quarters):
    with fits.open(path, memmap=False) as hdus:
        quarter = int(hdus[0].header["QUARTER"])
    if quarter in quarters:
        result["quarters"][str(quarter)] = {
            "status": "AVAILABLE",
            "path": str(path),
            "url": base + filename,
            "transport_uri": cloud_pixel_url(base, filename),
            "sha256": digest(path),
        }
    return quarter


def retrieve_parent(job, mapping, existing):
    parent, quarters = job
    name = f"{parent:09d}"
    base = f"https://archive.stsci.edu/pub/kepler/target_pixel_files/{name[:4]}/{name}/"
    result = {"parent_kic": parent, "archive_url": base, "quarters": {}}
    for quarter in quarters:
        if (parent, quarter) in existing:
            count(pixel_cache_hits=1)
            result["quarters"][str(quarter)] = {
                "status": "AVAILABLE",
                "path": existing[parent, quarter],
            }
    if len(result["quarters"]) == len(quarters):
        save(RUN / "pixel-manifests" / f"{parent}.json", result)
        return result
    if not read(CONFIG).get("network", True):
        for quarter in quarters:
            result["quarters"].setdefault(str(quarter), {"status": "NETWORK_DISABLED"})
        save(RUN / "pixel-manifests" / f"{parent}.json", result)
        return result
    try:
        add_archive_pixels(result, base, parent, quarters, mapping)
    except (requests.RequestException, OSError, ValueError, ET.ParseError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["retrieval_status"] = "TRANSPORT_FAILED"
        save(RUN / "pixel-manifests" / f"{parent}.json", result)
        raise
    for quarter in quarters:
        result["quarters"].setdefault(str(quarter), {"status": "TPF_UNAVAILABLE"})
    save(RUN / "pixel-manifests" / f"{parent}.json", result)
    return result


def retrieve():
    from vetting import retrieve_null_pixels
    from vetting_triage import measured_event_ids

    if retrieve_null_pixels():
        return
    PIXELS.mkdir(parents=True, exist_ok=True)
    metadata = read(RUN / "metadata.json")
    config = read(CONFIG)
    experimental = config.get("input_manifest")
    frozen = config.get("frozen_pixel_inputs")
    if frozen is not None:
        if experimental is not None:
            _, manifest = input_manifest(ROOT, experimental)
            if any(item.get("pixels") for item in manifest["sources"].values()):
                raise ValueError(
                    "Controlled pixels and a frozen public ledger are ambiguous"
                )
        return retrieve_frozen_pixels(frozen, metadata)
    if experimental is not None:
        _, manifest = input_manifest(ROOT, experimental)
        if (
            "native_control_binding" in manifest
            or manifest["schema"] == 2
            or (
                manifest["sources"]
                and all(source.get("pixels") for source in manifest["sources"].values())
            )
        ):
            return retrieve_experiment(experimental, metadata)
    jobs = {}
    for proposal in read(RUN / "cohort.json"):
        sid = proposal["source_id"]
        source = read(RUN / "sources" / f"{sid}.json")
        parent = metadata[sid]["origin_tpf_kic"]
        active = measured_event_ids(source)
        for event in source["pixel_jobs"]:
            if not pixel_job_active(event, active):
                continue
            jobs.setdefault(parent, set()).add(event["quarter"])
    mapping, existing = indexed_pixels() if jobs else ({}, {})
    with ThreadPoolExecutor(
        max_workers=execution_workers(config)["network_workers"]
    ) as pool:
        results = list(
            pool.map(lambda job: retrieve_parent(job, mapping, existing), jobs.items())
        )
    save(RUN / "pixel-retrieval.json", {"parents": len(jobs), "results": results})
    print("Pixel retrieval complete", flush=True)


def retrieve_frozen_pixels(binding, metadata):
    from deterministic.frozen_pixel_inputs import FrozenPixelInputs, retrieve_records
    from vetting_triage import measured_event_ids

    ledger = FrozenPixelInputs(Path(binding["path"]), binding["sha256"])
    config = read(CONFIG)
    scope = config.get("evaluation_scope")
    expected = {"development_replay": "development", "final_replay": "evaluation"}
    if scope and expected[scope["evaluation_role"]] != binding["role"]:
        raise ValueError("The frozen pixel role differs from the authorized evaluation")
    records, identities = [], []
    for sid, metadata_row in metadata.items():
        frozen_source = ledger.source(sid, metadata_row, binding["role"])
        source = read(RUN / "sources" / f"{sid}.json")
        active = measured_event_ids(source)
        quarters = sorted(
            {
                job["quarter"]
                for job in source["pixel_jobs"]
                if pixel_job_active(job, active)
            }
        )
        for quarter in quarters:
            records.append(ledger.record(frozen_source, quarter))
            identities.append(frozen_source["quarters"][str(quarter)])
    cache = PIXELS / f"frozen-{ledger.sha256}"
    results = retrieve_records(
        records,
        cache,
        network=config.get("network", True),
        reserve=config.get("pixel_minimum_free_bytes", 20 * 1024**3),
        maximum_bytes=binding["maximum_cache_bytes"],
    )
    parents = {}
    for record, result in zip(records, results):
        request = record["request"]
        quarters = parents.setdefault(request["origin_parent_kic"], {})
        merge_experiment_quarters(quarters, {str(request["quarter"]): result})
    for parent, quarters in parents.items():
        save(
            RUN / "pixel-manifests" / f"{parent}.json",
            {
                "quarters": quarters,
                "frozen_pixel_ledger_sha256": ledger.sha256,
            },
        )
    save(
        RUN / "pixel-retrieval.json",
        {
            "status": "FROZEN_PUBLIC_PIXEL_CONTENT_INPUTS",
            "parents": len(parents),
            "ledger": binding,
            "source_quarter_inputs": identities,
            "results": results,
        },
    )


def retrieve_experiment(path, metadata):
    path, manifest = input_manifest(ROOT, path)
    if "native_control_binding" in manifest:
        from pipeline.deterministic.native_prospective_controls import runtime_source

        for sid in metadata:
            runtime_source(manifest, sid)
            if not manifest["sources"][sid].get("pixels"):
                raise ValueError("Native replay has no bound target pixels")
    parents = {}
    for sid, source in metadata.items():
        item = manifest["sources"][sid]
        quarters = {}
        for quarter, pixel in item.get("pixels", {}).items():
            target = path.parent / pixel["path"]
            if digest(target) != pixel["sha256"]:
                raise ValueError("Experimental pixel input changed")
            quarters[quarter] = dict(pixel, path=str(target), status="AVAILABLE")
        row = read(RUN / "sources" / f"{sid}.json")
        for event in row["pixel_jobs"]:
            quarters.setdefault(
                str(event["quarter"]), {"status": "PIXELS_NOT_ALTERED_IN_THIS_CONTROL"}
            )
        shared = parents.setdefault(source["origin_tpf_kic"], {})
        merge_experiment_quarters(shared, quarters)
    for parent, quarters in parents.items():
        save(
            RUN / "pixel-manifests" / f"{parent}.json",
            {"quarters": quarters, "measurement_scope": manifest["measurement_scope"]},
        )
    save(
        RUN / "pixel-retrieval.json",
        {
            "status": "EXPLICIT_EXPERIMENT_INPUTS",
            "measurement_scope": manifest["measurement_scope"],
            "input_manifest": {"path": str(path), "sha256": digest(path)},
        },
    )


def merge_experiment_quarters(shared, quarters):
    for quarter, pixel in quarters.items():
        previous = shared.get(quarter)
        if previous and any(
            previous.get(k) != pixel.get(k) for k in ("status", "sha256")
        ):
            raise ValueError(
                "Conflicting controlled TPFs for one parent and quarter; use separate experiment manifests"
            )
        shared.setdefault(quarter, pixel)


def kepler_prf_path(channel, cache):
    from lkprf.utils import channel_to_module_output

    module, output = channel_to_module_output(channel)
    return cache / f"kplr{module:02}.{output}_2011265_prf.fits"


def verify_prf(path):
    import fitsio

    with fitsio.FITS(path) as hdus:
        for hdu in hdus:
            hdu.verify_checksum()


def bounded_download(url, target, deadline_seconds=60):
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "10",
        "--max-time",
        str(deadline_seconds),
        "--continue-at",
        "-",
        "--output",
        str(target),
        url,
    ]
    started_wall = time.monotonic()
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise OSError(f"Bounded archive transfer failed: {url}") from exc
    finally:
        count(network_calls=1, network_seconds=time.monotonic() - started_wall)


def download_prf(path, attempts=20, deadline_seconds=60):
    url = f"https://archive.stsci.edu/missions/kepler/fpc/prf/{path.name}"
    temporary = path.with_suffix(path.suffix + ".download")
    last_error = None
    for _ in range(attempts):
        try:
            bounded_download(url, temporary, deadline_seconds)
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            continue
        try:
            verify_prf(temporary)
            temporary.replace(path)
            return
        except OSError as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
    raise OSError(f"Kepler PRF download failed after {attempts} attempts: {last_error}")


def ensure_kepler_prf(channel, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = kepler_prf_path(channel, cache)
    with (cache / f"channel-{channel}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            download_prf(path)
        verify_prf(path)
    return path


def load_kepler_prf(channel, cache):
    path = ensure_kepler_prf(channel, cache)
    with (cache / f"channel-{channel}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prf = KeplerPRF(channel=channel, cache_dir=str(cache) + "/")
    stat = path.stat()
    return prf, {
        "filename": path.name,
        "sha256": pixel_digest(str(path), stat.st_size, stat.st_mtime_ns),
    }


def localize_event(path, source, event, expected_parent=None):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from incinerator.localize import Localize

    path = Path(path)
    count(pixel_files_opened=1, pixel_bytes_opened=path.stat().st_size)

    class IdentifiedLocalize(Localize):
        def _search_and_propagate(self):
            if astrometry_status != "PROPAGATED_GAIA_POSITION":
                self.ra_bonus, self.dec_bonus = source["ra_deg"], source["dec_deg"]
                return
            coord = SkyCoord(
                ra=identity["ra"] * u.deg,
                dec=identity["dec"] * u.deg,
                pm_ra_cosdec=identity["pmra"] * u.mas / u.yr,
                pm_dec=identity["pmdec"] * u.mas / u.yr,
                obstime=Time(2016.0, format="jyear"),
            )
            observed = coord.apply_space_motion(new_obstime=self.epoch)
            self.ra_bonus, self.dec_bonus = observed.ra.deg, observed.dec.deg

    identities = query_evidence(source["source_id"], "gaia")["matches"]
    identity = identities[0] if len(identities) == 1 else {}
    astrometry_status = (
        "PROPAGATED_GAIA_POSITION"
        if identity.get("pmra") is not None and identity.get("pmdec") is not None
        else "CATALOGUE_POSITION_UNPROPAGATED"
    )

    centre, duration = event["centre_bkjd"], event["duration_days"]
    with fits.open(path, memmap=False) as hdus:
        header, table = hdus[0].header, hdus[1].data
        parent = int(header["KEPLERID"])
        expected_parent = (
            source["origin_tpf_kic"] if expected_parent is None else expected_parent
        )
        if parent != expected_parent or int(header["QUARTER"]) != event["quarter"]:
            return {"status": "PIXEL_IDENTITY_OR_QUARTER_MISMATCH"}
        time = np.array(table["TIME"], float)
        use = np.isfinite(time) & (
            (np.array(table["QUALITY"]) & read(CONFIG)["quality_bitmask"]) == 0
        )
        phase = time - centre
        use &= abs(phase) <= max(1.0, 5 * duration)
        if use.sum() <= 3 or not np.any(abs(phase[use]) <= duration / 2):
            return {"status": "INSUFFICIENT_PIXEL_TIME_COVERAGE"}
        flux, error = (
            np.array(table["FLUX"][use], float),
            np.array(table["FLUX_ERR"][use], float),
        )
        wcs = WCS(hdus[2].header)
        channel = int(header["CHANNEL"])
        keywords = {
            "channel": channel,
            "quarter": int(header["QUARTER"]),
            "origin_row": hdus[1].header["2CRV4P"],
            "origin_col": hdus[1].header["1CRV4P"],
        }
        target_ra, target_dec = header["RA_OBJ"], header["DEC_OBJ"]
    period = 2 * np.ptp(time[use]) + 10 * duration
    tces = pd.DataFrame([{"period": period, "t0": centre + 2454833, "tdur": duration}])
    prf_cache = dump_path(
        ROOT / read(CONFIG).get("prf_cache_directory", str(RUN / "prf-cache"))
    )
    prf, calibration = load_kepler_prf(channel, prf_cache)
    local = IdentifiedLocalize(
        time[use] + 2454833,
        flux,
        error,
        tces,
        id=str(parent),
        ra_targ=target_ra,
        dec_targ=target_dec,
        ra_bonus=source["ra_deg"],
        dec_bonus=source["dec_deg"],
        wcs=wcs,
        epoch=Time(np.mean(time[use]) + 2454833, format="jd"),
        mission="kepler",
        prf=prf,
        **keywords,
    )
    result = solve_local(local, wcs, period)
    result["target_astrometry"] = astrometry_status
    result["period_days"] = None
    result["observation_epoch_bkjd"] = float(np.mean(time[use]))
    result["prf_calibration"] = calibration
    stat = path.stat()
    result["tpf_sha256"] = pixel_digest(str(path), stat.st_size, stat.st_mtime_ns)
    matrix = wcs.pixel_scale_matrix * 3600
    if result.get("metrics", {}).get("centerx") is not None:
        delta = (
            np.array([result["metrics"][k] for k in ("centerx", "centery")])
            - result["target_pixel"]
        )
        result["tangent_offset_arcsec"] = (matrix @ delta).tolist()
        covariance = result.get("centroid_uncertainty", {}).get("covariance_pixels")
        if covariance is not None:
            result["tangent_covariance_arcsec2"] = (
                matrix @ covariance @ matrix.T
            ).tolist()
    return result


@lru_cache(maxsize=64)
def pixel_digest(path, size, modified):
    return digest(Path(path))


def metric_values(metrics):
    return {
        key: float(value) if value is not None and np.isfinite(value) else None
        for key, value in metrics.items()
    }


def centroid_status(metrics):
    uncertainties = [metrics.get(key) for key in ("sigmax", "sigmay")]
    if all(
        value is not None and np.isfinite(value) and value > 0
        for value in uncertainties
    ):
        return "PRF_FIT_COMPLETE"
    return "CENTROID_UNCERTAINTY_UNRESOLVED"


def residualize_pixels(time, pix, error, pulses, cycle, phase):
    residual = np.empty_like(pix)
    projected = np.empty((*pix.shape, pulses.shape[1]))
    weight = 1 / error**2
    for event in np.unique(cycle):
        use = cycle == event
        baseline = np.column_stack([np.ones(use.sum()), phase[use]])
        precision = np.einsum("ti,tp,tj->pij", baseline, weight[use], baseline)
        rhs = np.einsum("ti,tp,tp->pi", baseline, weight[use], pix[use])
        coefficients = np.linalg.solve(precision, rhs[..., None])[..., 0]
        pulse_rhs = np.einsum("ti,tp,td->pid", baseline, weight[use], pulses[use])
        pulse_coefficients = np.linalg.solve(precision, pulse_rhs)
        residual[use] = pix[use] - baseline @ coefficients.T
        projected[use] = pulses[use, None, :] - np.einsum(
            "ti,pid->tpd", baseline, pulse_coefficients
        )
    return residual, projected, weight


def pixel_regression(time, pix, error, centre, period, duration, true_centre=None):
    if not np.isfinite(error).all() or np.any(error <= 0):
        raise ValueError("Positive finite pixel uncertainties required")
    cycle = np.rint((time - centre) / period)
    phase = time - centre - cycle * period
    pulse = -(abs(phase) <= duration / 2).astype(float)
    pulses = pulse[:, None]
    if true_centre is not None:
        true_phase = (time - true_centre + period / 2) % period - period / 2
        pulses = np.column_stack(
            [pulse, -(abs(true_phase) <= duration / 2).astype(float)]
        )
    residual, projected, weight = residualize_pixels(
        time, pix, error, pulses, cycle, phase
    )
    precision = np.einsum("tpd,tp,tpe->pde", projected, weight, projected)
    rhs = np.einsum("tpd,tp,tp->pd", projected, weight, residual)
    covariance = np.linalg.inv(precision)
    coefficients = np.linalg.solve(precision, rhs[..., None])[..., 0]
    residual -= np.einsum("tpd,pd->tp", projected, coefficients)
    influence = np.einsum("pd,tpd->tp", covariance[:, 0], projected) * weight
    return coefficients[:, 0], covariance[:, 0, 0], residual, influence


def fit_prf_heatmap(local, column, row):
    usable = local.good_pix_mask
    data = local.transit_weights[0][usable]
    error = local.transit_weights_err[0][usable]
    params = Parameters()
    params.add(
        "centerx",
        value=float(np.clip(column, -0.5, local.shape[1] - 0.5)),
        min=-0.5,
        max=local.shape[1] - 0.5,
    )
    params.add(
        "centery",
        value=float(np.clip(row, -0.5, local.shape[0] - 0.5)),
        min=-0.5,
        max=local.shape[0] - 0.5,
    )
    params.add("amplitude", value=max(float(np.nanmax(data)), 0.0), min=0)

    def residual(parameters):
        profile = prf_profile(
            local,
            parameters["centerx"].value,
            parameters["centery"].value,
        )
        values = (parameters["amplitude"].value * profile - data) / error
        return values[np.isfinite(values)]

    fit = minimize(residual, params)
    metrics = {
        "centerx": fit.params["centerx"].value,
        "sigmax": fit.params["centerx"].stderr,
        "centery": fit.params["centery"].value,
        "sigmay": fit.params["centery"].stderr,
    }
    return fit, metrics


def inside_pixel_footprint(column, row, shape):
    return bool(-0.5 <= column <= shape[1] - 0.5 and -0.5 <= row <= shape[0] - 0.5)


def spatial_null_bic_advantage(amplitude, error, fitted_prf, mask):
    values = np.asarray(amplitude, dtype=float)
    errors = np.asarray(error, dtype=float)
    fitted_prf = np.asarray(fitted_prf, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if (
        len(values) != mask.sum()
        or len(values) < 6
        or not np.isfinite(values).all()
        or not np.isfinite(errors).all()
        or not np.isfinite(fitted_prf).all()
        or np.any(errors <= 0)
    ):
        return None
    weights = 1 / errors**2
    prf_chi2 = np.sum(weights * (values - fitted_prf) ** 2)
    coordinates = np.indices(mask.shape)
    templates = [np.ones(len(values))]
    templates.extend(
        (coordinates[0][mask] == row).astype(float) for row in range(mask.shape[0])
    )
    templates.extend(
        (coordinates[1][mask] == column).astype(float)
        for column in range(mask.shape[1])
    )
    alternative_chi2 = []
    for template in templates:
        denominator = np.sum(weights * template**2)
        if denominator <= 0:
            continue
        coefficient = max(0.0, np.sum(weights * template * values) / denominator)
        alternative_chi2.append(
            np.sum(weights * (values - coefficient * template) ** 2)
        )
    if not alternative_chi2:
        return None
    trial_penalty = 2 * np.log(len(alternative_chi2))
    score = prf_chi2 - min(alternative_chi2) + 2 * np.log(len(values)) - trial_penalty
    return float(score) if np.isfinite(score) else None


def solve_local(local, wcs, period):
    centre, duration = local.tces.iloc[0]["t0"], local.tces.iloc[0]["tdur"]
    amplitude, variance, local.depth_residual, local.depth_influence = pixel_regression(
        local.time, local.pix, local.pix_err, centre, period, duration
    )
    local.valid_tces = np.array([0])
    local.tce_mapping = {0: 0}
    usable = local.good_pix_mask
    heatmap, errors = np.full(local.shape, np.nan), np.full(local.shape, np.nan)
    heatmap[usable], errors[usable] = amplitude, np.sqrt(variance)
    local.transit_weights, local.transit_weights_err = [heatmap], [errors]
    if not np.isfinite(local.transit_weights_err[0][usable]).all():
        return {"status": "INVALID_PIXEL_UNCERTAINTIES"}
    expected_x, expected_y = wcs.world_to_pixel_values(local.ra_bonus, local.dec_bonus)
    fit, metrics = fit_prf_heatmap(local, expected_x, expected_y)
    uncertainty = centroid_uncertainty(local, fit, metrics, duration)
    profile = prf_profile(local, metrics["centerx"], metrics["centery"])
    fitted_prf = profile * fit.params["amplitude"].value
    correlation = np.corrcoef(profile, local.transit_weights[0][usable])[0, 1]
    return {
        "status": centroid_status(metrics),
        "prf_chi2": float(fit.chisqr),
        "prf_reduced_chi2": float(fit.redchi),
        "spatial_null_bic_advantage": spatial_null_bic_advantage(
            local.transit_weights[0][usable],
            local.transit_weights_err[0][usable],
            fitted_prf,
            usable,
        ),
        "prf_fit_success": bool(fit.success),
        "metrics": metric_values(metrics),
        "centroid_uncertainty": uncertainty,
        "prf_correlation": float(correlation) if np.isfinite(correlation) else None,
        "pixel_scale_arcsec": float(
            np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
        ),
        "target_pixel": [float(expected_x), float(expected_y)],
        "target_inside_stamp": inside_pixel_footprint(
            expected_x, expected_y, local.shape
        ),
        "fixed_source_controls": pixel_controls(local, expected_x, expected_y),
        "pixel_shape": list(local.flux.shape[1:]),
        "pixel_evidence": {
            "good_pixel_mask": usable.tolist(),
            "difference_amplitude": local.transit_weights[0][usable].tolist(),
            "difference_error": local.transit_weights_err[0][usable].tolist(),
            "fitted_prf": fitted_prf.tolist(),
        },
        "period_days": None,
        "computational_period_days": period,
        "baseline_model": "Separate intercept and slope for each observed event, with a common transit depth map.",
        "scope": "A specified detector event in this quarter. Source-offset measurements and image quality are diagnostics for disposition; they do not establish planetary validation.",
    }


def prf_profile(local, column, row):
    cache = getattr(local, "_prf_profile_cache", None)
    if cache is None:
        cache = local._prf_profile_cache = {}
    position = (float(column), float(row))
    if position in cache:
        return cache[position]
    prf = local.prf.evaluate(
        targets=[(local.origin_row + row, local.origin_col + column)],
        origin=(local.origin_row, local.origin_col),
        shape=local.shape,
    )[0]
    cache[position] = prf[local.good_pix_mask]
    return cache[position]


def centroid_uncertainty(local, fit, metrics, duration):
    if centroid_status(metrics) != "PRF_FIT_COMPLETE":
        return {"status": "FORMAL_UNCERTAINTY_UNAVAILABLE"}
    influence, residual = local.depth_influence, local.depth_residual
    time_blocks = np.floor((local.time - local.time.min()) / max(duration, CADENCE))
    _, labels = np.unique(time_blocks, return_inverse=True)
    blocks = int(labels.max()) + 1
    if blocks < 2:
        return {"status": "INSUFFICIENT_RESIDUAL_BLOCKS"}
    contributions = np.zeros((blocks, local.pix.shape[1]))
    np.add.at(contributions, labels, influence * residual)
    x, y = metrics["centerx"], metrics["centery"]
    amplitude = fit.params["amplitude"].value
    step = 1e-3
    dx = (prf_profile(local, x + step, y) - prf_profile(local, x - step, y)) / (
        2 * step
    )
    dy = (prf_profile(local, x, y + step) - prf_profile(local, x, y - step)) / (
        2 * step
    )
    jacobian = np.column_stack(
        [amplitude * dx, amplitude * dy, prf_profile(local, x, y)]
    )
    errors = local.transit_weights_err[0][local.good_pix_mask]
    mapping = np.linalg.solve(
        jacobian.T @ (jacobian / errors[:, None] ** 2), jacobian.T / errors**2
    )
    offsets = contributions @ mapping[:2].T
    covariance = offsets.T @ offsets * blocks / (blocks - 1)
    formal = np.array([metrics["sigmax"], metrics["sigmay"]]) ** 2
    covariance += np.diag(np.maximum(formal - np.diag(covariance), 0))
    return {
        "status": "BLOCK_SANDWICH_ESTIMATED",
        "covariance_pixels": covariance.tolist(),
        "blocks": blocks,
        "block_days": max(duration, CADENCE),
        "method": "Transit-duration residual blocks propagated through the pixel regression and PRF fit Jacobians. Each marginal variance is at least the formal PRF variance. Instrumental systematics require separate calibration.",
    }


def control_amplitude(profile, amplitudes, variance):
    if not np.isfinite([profile, amplitudes, variance]).all() or np.any(variance <= 0):
        return {"status": "NONFINITE_CONTROL_OR_NOISE"}
    denominator = np.sum(profile**2 / variance)
    if not np.isfinite(denominator) or denominator <= 0:
        return {"status": "PRF_PROFILE_UNCONSTRAINED"}
    amplitude = np.sum(profile * amplitudes / variance) / denominator
    if not np.isfinite(amplitude):
        return {"status": "NONFINITE_CONTROL_AMPLITUDE"}
    return {
        "status": "MEASURED",
        "amplitude": float(amplitude),
        "formal_error": float(1 / np.sqrt(denominator)),
    }


def pixel_controls(local, column, row):
    profile = prf_profile(local, column, row)
    centre, duration = local.tces.iloc[0]["t0"], local.tces.iloc[0]["tdur"]
    period = local.tces.iloc[0]["period"]
    records = []
    for offset in [0, -4, -3, -2, 2, 3, 4]:
        try:
            amplitudes, variance, _, _ = pixel_regression(
                local.time,
                local.pix,
                local.pix_err,
                centre + offset * duration,
                period,
                duration,
                centre if offset else None,
            )
        except np.linalg.LinAlgError:
            records.append(
                {"offset_durations": offset, "status": "NO_CONTROL_COVERAGE"}
            )
            continue
        records.append(
            {
                "offset_durations": offset,
                **control_amplitude(profile, amplitudes, variance),
            }
        )
    return {
        "measurements": records,
        "interpretation": "Fixed proposed-source PRF amplitude at event and nearby nonoverlapping times. These few local controls are diagnostics, not calibrated false-alarm probabilities.",
    }


def pixel_quality(result):
    missing = []
    if not result.get("prf_fit_success"):
        missing.append("prf_fit_success")
    if (
        result.get("centroid_uncertainty", {}).get("status")
        != "BLOCK_SANDWICH_ESTIMATED"
    ):
        missing.append("CENTROID_NOISE_UNRESOLVED")
    if result.get("target_inside_stamp") is not True:
        missing.append("TARGET_OUTSIDE_STAMP")
    return missing


def compact_localization(result):
    keys = (
        "status",
        "prf_reduced_chi2",
        "spatial_null_bic_advantage",
        "prf_fit_success",
        "metrics",
        "centroid_uncertainty",
        "prf_correlation",
        "pixel_scale_arcsec",
        "target_pixel",
        "target_inside_stamp",
        "fixed_source_controls",
        "pixel_shape",
        "period_days",
        "computational_period_days",
        "target_astrometry",
        "observation_epoch_bkjd",
        "prf_calibration",
        "tpf_sha256",
        "tangent_offset_arcsec",
        "tangent_covariance_arcsec2",
        "attribution",
    )
    return {key: result[key] for key in keys if key in result}


def scene_offsets(result, scene, wcs, inverse):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    sources = [
        s for s in scene["sources"] if s["pmra"] is not None and s["pmdec"] is not None
    ]
    measured = {}
    if sources:
        position = SkyCoord(
            ra=np.array([s["ra"] for s in sources]) * u.deg,
            dec=np.array([s["dec"] for s in sources]) * u.deg,
            pm_ra_cosdec=np.array([s["pmra"] for s in sources]) * u.mas / u.yr,
            pm_dec=np.array([s["pmdec"] for s in sources]) * u.mas / u.yr,
            obstime=Time(2016, format="jyear"),
        )
        observed = position.apply_space_motion(
            new_obstime=Time(result["observation_epoch_bkjd"] + 2454833, format="jd")
        )
        centroid = np.array([result["metrics"][k] for k in ("centerx", "centery")])
        delta = (
            centroid
            - np.array(wcs.world_to_pixel_values(observed.ra.deg, observed.dec.deg)).T
        )
        sigma = np.sqrt(np.einsum("ni,ij,nj->n", delta, inverse, delta))
        if not np.isfinite(sigma).all():
            raise ValueError("Nonfinite propagated Gaia source offsets")
        measured = {s["source_id"]: float(v) for s, v in zip(sources, sigma)}
    missing_motion = [
        source
        for source in scene["sources"]
        if source["pmra"] is None or source["pmdec"] is None
    ]
    catalogue_epoch = {}
    if missing_motion:
        centroid = np.array([result["metrics"][k] for k in ("centerx", "centery")])
        pixels = np.array(
            wcs.world_to_pixel_values(
                [source["ra"] for source in missing_motion],
                [source["dec"] for source in missing_motion],
            )
        ).T
        delta = centroid - pixels
        sigma = np.sqrt(np.einsum("ni,ij,nj->n", delta, inverse, delta))
        if not np.isfinite(sigma).all():
            raise ValueError("Nonfinite Gaia catalogue-epoch source offsets")
        catalogue_epoch = {
            source["source_id"]: float(value)
            for source, value in zip(missing_motion, sigma)
        }
    return [
        {
            "source_id": source["source_id"],
            **(
                {"status": "MEASURED", "offset_sigma": measured[source["source_id"]]}
                if source["source_id"] in measured
                else {
                    "status": "CATALOGUE_EPOCH_ONLY",
                    "offset_sigma": catalogue_epoch[source["source_id"]],
                }
            ),
        }
        for source in scene["sources"]
    ]


def attribute_pixels(result, scene):
    missing = pixel_quality(result)
    if scene.get("status") != "GAIA_SCENE_QUERIED":
        missing.append("GAIA_SCENE_INCOMPLETE")
    if result.get("target_astrometry") != "PROPAGATED_GAIA_POSITION":
        missing.append("SOURCE_ASTROMETRY_UNRESOLVED")
    if missing:
        return {"status": "PIXEL_SOURCE_UNRESOLVED", "missing": missing}
    policy = read(CONFIG)["pixel_policy"]
    covariance = np.array(result["centroid_uncertainty"]["covariance_pixels"])
    systematic = policy["astrometric_systematic_arcsec"]
    if systematic is not None:
        covariance += np.eye(2) * (systematic / result["pixel_scale_arcsec"]) ** 2
    inverse = np.linalg.inv(covariance)
    centroid = np.array([result["metrics"][k] for k in ("centerx", "centery")])
    delta = centroid - result["target_pixel"]
    sigma = float(np.sqrt(delta @ inverse @ delta))
    if not np.isfinite(sigma):
        return {"status": "PIXEL_SOURCE_UNRESOLVED", "missing": ["NONFINITE_OFFSET"]}
    aperture = pixel_headers(result["tpf_path"], result["tpf_sha256"])[2]
    offsets = scene_offsets(result, scene, WCS(aperture), inverse)
    ambiguous = [
        r["source_id"]
        for r in offsets
        if r.get("offset_sigma") is not None
        and r["offset_sigma"] <= policy["offset_sigma"]
    ]
    status = (
        "OFFSET_DIAGNOSTIC_FLAG"
        if sigma > policy["offset_sigma"]
        else "NOMINAL_SOURCE_COMPATIBILITY"
    )
    if ambiguous:
        status = "MULTIPLE_POSSIBLE_SOURCES"
    return {
        "status": status,
        "target_offset_sigma": sigma,
        "astrometric_systematic_arcsec": systematic,
        "systematic_calibration": "UNAVAILABLE" if systematic is None else "CONFIGURED",
        "high_quality_difference_image": (result.get("prf_correlation") or 0)
        > policy["minimum_prf_correlation"],
        "covariance_pixels": covariance.tolist(),
        "neighbours": offsets,
        "compatible_neighbours": ambiguous,
        "unassessed_neighbour_motions": [
            r["source_id"] for r in offsets if r["status"] == "CATALOGUE_EPOCH_ONLY"
        ],
        "scope": "Per-event PRF correlation and covariance-normalized offsets are diagnostics, not automatic dispositions. The v2 source rule uses only inside-stamp, quality-qualified measurements in a robust multi-quarter aggregate; nominal compatibility is not unique host attribution.",
    }


def localized_event(sid, event, pixels, source, scene_row):
    cache = RUN / "localizations-final" / f"{sid}-{event['event_id']}.json"
    if cache.exists():
        return read(cache)
    item = pixels[str(event["quarter"])]
    if item["status"] != "AVAILABLE":
        result = {"status": item["status"]}
    else:
        try:
            result = localize_event(
                item["path"], source, event, item.get("base_parent_kic")
            )
            result["tpf_path"] = item["path"]
            result["attribution"] = attribute_pixels(result, scene_row["gaia_scene"])
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            requests.RequestException,
            np.linalg.LinAlgError,
        ) as exc:
            result = {
                "status": "LOCALIZATION_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
    save(cache, result)
    return result


def localize_source(proposal, metadata):
    from vetting_triage import apply_gate, measured_event_ids

    sid = proposal["source_id"]
    path = RUN / "sources" / f"{sid}.json"
    row = read(path)
    parent = metadata[sid]["origin_tpf_kic"]
    pixel_path = RUN / "pixel-manifests" / f"{parent}.json"
    active = measured_event_ids(row)
    jobs = [event for event in row["pixel_jobs"] if pixel_job_active(event, active)]
    pixels = read(pixel_path)["quarters"] if jobs else {}
    outcomes = []
    skipped = 0
    for event in row["pixel_jobs"]:
        if not pixel_job_active(event, active):
            skipped += 1
            continue
        result = localized_event(sid, event, pixels, metadata[sid], row)
        cache = RUN / "localizations-final" / f"{sid}-{event['event_id']}.json"
        outcomes.append(
            {
                **event,
                **compact_localization(result),
                "evidence": str(cache.relative_to(RUN)),
            }
        )
    row["localization"] = {
        "status": "UNRESOLVED",
        "events": outcomes,
        "skipped_nonpromoted_jobs": skipped,
    }
    aggregate_signals(row)
    row = apply_gate(row, "localization", read(CONFIG))
    from vetting_disposition import adjudicate

    row["disposition"] = adjudicate(row)
    save(path, row, compact=True)
    return {
        "source_id": sid,
        "events": len(outcomes),
        "triage": row.get("triage"),
    }


def combine_positions(records):
    if not records:
        return {"status": "POSITION_UNCONSTRAINED"}
    offsets = np.array([r["tangent_offset_arcsec"] for r in records])
    covariances = np.array([r["tangent_covariance_arcsec2"] for r in records])
    if not np.isfinite(offsets).all() or not np.isfinite(covariances).all():
        return {
            "status": "POSITION_UNCONSTRAINED",
            "reason": "NONFINITE_POSITION_MEASUREMENT",
            "measurement_count": len(records),
        }
    try:
        precision = np.linalg.inv(covariances)
        covariance = np.linalg.inv(precision.sum(axis=0))
    except np.linalg.LinAlgError:
        return {
            "status": "POSITION_UNCONSTRAINED",
            "reason": "SINGULAR_POSITION_COVARIANCE",
            "measurement_count": len(records),
        }
    mean = covariance @ np.einsum("nij,nj->i", precision, offsets)
    return {
        "status": "MEASURED",
        "offset_arcsec": mean.tolist(),
        "formal_independent_covariance_arcsec2": covariance.tolist(),
        "unknown_correlation_covariance_bound_arcsec2": (
            len(records) * covariance
        ).tolist(),
        "individual_offsets_arcsec": offsets.tolist(),
        "independent_measurements_assumed": False,
    }


def robust_coordinate(values):
    values = np.asarray(values, dtype=float)
    centre = float(np.median(values))
    scale = float(np.median(np.abs(values - centre)) / 0.6744897501960817)
    if not np.isfinite(scale) or scale <= 1e-12:
        return centre
    for _ in range(20):
        residual = (values - centre) / (4.685 * scale)
        weights = np.where(abs(residual) < 1, (1 - residual**2) ** 2, 0)
        if weights.sum() <= 0:
            break
        updated = float(np.sum(weights * values) / weights.sum())
        if abs(updated - centre) <= 1e-12:
            break
        centre = updated
    return centre


def robust_position(offsets):
    return np.array(
        [robust_coordinate(offsets[:, axis]) for axis in range(offsets.shape[1])]
    )


def unit_vector(position):
    distance = float(np.linalg.norm(position))
    return position / distance if distance else np.zeros(2)


def offset_test(position, variance, policy):
    distance = float(np.linalg.norm(position))
    uncertainty = float(np.sqrt(variance))
    significance = distance / uncertainty
    threshold = (
        policy["large_offset_sigma"]
        if distance > policy["large_offset_arcsec"]
        else policy["small_offset_sigma"]
    )
    conflict = (
        distance >= policy["minimum_material_offset_arcsec"]
        and significance > threshold
    )
    return "ROBUST_OFFSET_CONFLICT" if conflict else "TARGET_COMPATIBLE", {
        "offset_distance_arcsec": distance,
        "offset_uncertainty_arcsec": uncertainty,
        "offset_significance": significance,
        "required_significance": threshold,
    }


def target_snr(pixel):
    control = next(
        item
        for item in pixel["fixed_source_controls"]["measurements"]
        if item["offset_durations"] == 0
    )
    if control["status"] != "MEASURED":
        return None
    return control["amplitude"] / control["formal_error"]


def single_event_source_position(quarters, selected, policy):
    minimum = policy["single_event_minimum_quarter_snr"]
    if not quarters:
        return {
            "status": "INSUFFICIENT_GOOD_QUARTERS",
            "good_quarters": 0,
            "minimum_good_quarters": 1,
            "basis": "SINGLE_QUARTER",
        }
    ((quarter, record),) = quarters.items()
    (pixel,) = selected[quarter].values()
    snr = target_snr(pixel)
    if snr is None or snr < minimum:
        return {
            "status": "INSUFFICIENT_QUARTER_SNR",
            "good_quarters": 1,
            "target_snr": snr,
            "minimum_quarter_snr": minimum,
            "basis": "SINGLE_QUARTER",
        }
    position = np.array(record["offset_arcsec"])
    covariance = np.array(record["unknown_correlation_covariance_bound_arcsec2"])
    direction = unit_vector(position)
    formal = float(direction @ covariance @ direction)
    systematic = policy["single_quarter_systematic_arcsec"]
    status, test = offset_test(position, formal + systematic**2, policy)
    return {
        "status": status,
        "good_quarters": 1,
        "minimum_good_quarters": 1,
        "target_snr": snr,
        "minimum_quarter_snr": minimum,
        "offset_arcsec": position.tolist(),
        **test,
        "uncertainty_variance_arcsec2": {"formal": formal},
        "single_quarter_systematic_arcsec": systematic,
        "basis": "SINGLE_QUARTER",
        "method": "The quality-qualified quarter of a single event, accepted when the fixed-target PRF amplitude reaches the minimum S/N; its formal offset variance is combined with the single-quarter systematic floor.",
    }


def aggregate_source_position(quarters, policy):
    minimum = policy["minimum_good_quarters"]
    records = list(quarters.values())
    if len(records) < minimum:
        return {
            "status": "INSUFFICIENT_GOOD_QUARTERS",
            "good_quarters": len(records),
            "minimum_good_quarters": minimum,
        }
    offsets = np.array([record["offset_arcsec"] for record in records])
    covariances = np.array(
        [record["unknown_correlation_covariance_bound_arcsec2"] for record in records]
    )
    position = robust_position(offsets)
    direction = unit_vector(position)
    formal = covariances.sum(axis=0) / len(records) ** 2
    scatter = np.cov(offsets, rowvar=False, ddof=1) / len(records)
    leave_one_out = np.array(
        [
            robust_position(np.delete(offsets, index, axis=0))
            for index in range(len(records))
        ]
    )
    residual = leave_one_out - leave_one_out.mean(axis=0)
    jackknife = (len(records) - 1) / len(records) * residual.T @ residual
    components = {
        "formal": float(direction @ formal @ direction),
        "quarter_scatter": float(direction @ scatter @ direction),
        "jackknife": float(direction @ jackknife @ direction),
    }
    systematic = policy["aggregate_systematic_arcsec"]
    status, test = offset_test(
        position, max(0.0, *components.values()) + systematic**2, policy
    )
    return {
        "status": status,
        "good_quarters": len(records),
        "minimum_good_quarters": minimum,
        "robust_offset_arcsec": position.tolist(),
        **test,
        "uncertainty_variance_arcsec2": components,
        "aggregate_systematic_arcsec": systematic,
        "method": "Unweighted Tukey-biweight coordinates with formal, quarter-scatter, and leave-one-quarter-out jackknife uncertainty; the largest statistical variance is combined with the astrometric systematic floor.",
    }


def aggregate_signals(row):
    """Localize each signal from its pixel measurements once; later stages reuse it."""
    from vetting_triage import signal_measured

    config = read(CONFIG)
    policy = config["pixel_policy"]
    prf_cache = dump_path(
        ROOT / config.get("prf_cache_directory", str(RUN / "prf-cache"))
    )
    scene_event_cache = {}
    for signal in row.get("signals", []):
        if "localization" in signal:
            continue
        if not signal_measured(signal):
            signal.setdefault(
                "localization",
                {
                    "status": "SKIPPED_DISMISSED",
                    "reason": signal["triage"]["reason"],
                },
            )
            continue
        pixels = pixel_measurements_for_signal(row, signal["signal_id"])
        quarters, good_quarters, spatial_scores = {}, {}, []
        for event in signal["events"]:
            measurements = [pixels[key] for key in event["event_ids"] if key in pixels]
            available = [
                p
                for p in measurements
                if p.get("tangent_covariance_arcsec2") is not None
            ]
            if available:
                quarters.setdefault(str(event["quarter"]), {})[
                    available[0]["event_id"]
                ] = available[0]
                score = available[0].get("spatial_null_bic_advantage")
                if score is not None and np.isfinite(score):
                    spatial_scores.append(float(score))
            good = [
                measurement
                for measurement in available
                if measurement.get("target_inside_stamp") is True
                and measurement.get("prf_correlation") is not None
                and measurement["prf_correlation"] > policy["minimum_prf_correlation"]
            ]
            if good:
                selected = max(good, key=lambda item: item["prf_correlation"])
                good_quarters.setdefault(str(event["quarter"]), {})[
                    selected["event_id"]
                ] = selected
        combined_good = {
            quarter: combine_positions(list(records.values()))
            for quarter, records in good_quarters.items()
        }
        constrained = {
            quarter: record
            for quarter, record in combined_good.items()
            if record["status"] == "MEASURED"
        }
        signal["localization"] = {
            "status": "MEASURED" if quarters else "UNCONSTRAINED",
            "quarters": {
                quarter: combine_positions(list(records.values()))
                for quarter, records in quarters.items()
            },
            "good_quarters": combined_good,
            "source_attribution": (
                single_event_source_position(constrained, good_quarters, policy)
                if len(signal["events"]) == 1
                else aggregate_source_position(constrained, policy)
            ),
            "spatial_null": {
                "status": "MEASURED" if spatial_scores else "UNASSESSED",
                "event_units": len(spatial_scores),
                "median_bic_advantage": (
                    float(np.median(spatial_scores)) if spatial_scores else None
                ),
            },
            "scene_prf": compare_signal(
                row, good_quarters, RUN, prf_cache, scene_event_cache
            ),
            "scope": "One representative per overlapping event unit prevents PSF/SAP duplication. Within-quarter GLS retains a conservative unknown-correlation covariance bound. Automatic source attribution uses only inside-stamp, quality-qualified quarters and a robust cross-quarter aggregate.",
        }


def pixel_measurements_for_signal(row, signal_id):
    signal = next(
        (item for item in row.get("signals", []) if item["signal_id"] == signal_id),
        None,
    )
    hypotheses = row.get("orbital", {}).get("hypotheses", {})
    refined = signal is not None and any(
        hypotheses.get(item["hypothesis_id"], {})
        .get("timing_refinement", {})
        .get("status")
        == "SELECTED"
        for item in signal.get("timing_hypotheses", [])
        if item.get("timing_basis") == "ORIGINAL_PROPOSAL"
    )
    return {
        event_id: pixel
        for pixel in row.get("localization", {}).get("events", [])
        if pixel.get("timing_refined_signal_id") == (signal_id if refined else None)
        for event_id in pixel.get("represented_event_ids", [pixel["event_id"]])
    }


@lru_cache(maxsize=1)
def worker_metadata():
    return read(RUN / "metadata.json")


def localize_proposal(proposal):
    started_at, before = started(), counter_snapshot()
    result = localize_source(proposal, worker_metadata())
    return {
        **result,
        "program": "localize_source",
        "timing": elapsed(started_at),
        "counters": counter_deltas(before),
    }


def localization_tasks(proposals, events_per_task=16):
    from vetting_triage import measured_event_ids

    tasks = []
    for proposal in proposals:
        sid = proposal["source_id"]
        row = read(RUN / "sources" / f"{sid}.json")
        active = measured_event_ids(row)
        seen = set()
        events = []
        for event in row["pixel_jobs"]:
            event_id = event["event_id"]
            if pixel_job_active(event, active) and event_id not in seen:
                events.append(event)
                seen.add(event_id)
        scene_row = {"gaia_scene": row["gaia_scene"]} if "gaia_scene" in row else {}
        for start in range(0, len(events), events_per_task):
            tasks.append((sid, events[start : start + events_per_task], scene_row))
    return tasks


def localize_task(task):
    sid, events, scene_row = task
    started_at, before = started(), counter_snapshot()
    source = worker_metadata()[sid]
    pixels = read(RUN / "pixel-manifests" / f"{source['origin_tpf_kic']}.json")[
        "quarters"
    ]
    for event in events:
        localized_event(sid, event, pixels, source, scene_row)
    return {
        "source_id": sid,
        "events": len(events),
        "timing": elapsed(started_at),
        "counters": counter_deltas(before),
    }


def localize():
    import os

    from vetting_triage import gate_summary

    os.chdir(RUN)
    proposals = read(RUN / "cohort.json")
    tasks = localization_tasks(proposals)
    gate_states = []
    with ProcessPoolExecutor(max_workers=read(CONFIG)["workers"]) as pool:
        futures = [pool.submit(localize_task, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            log_worker(future.result())
            print(f"Localized pixel task {i}/{len(tasks)}", flush=True)
        for i, result in enumerate(pool.map(localize_proposal, proposals)):
            triage = result.pop("triage")
            if triage is not None:
                gate_states.append(triage)
            log_worker(result)
            print(f"Localized source {i + 1}/{len(proposals)}", flush=True)
    if gate_states:
        summary = gate_summary(gate_states)
        save(RUN / "triage" / "localization.json", summary)
        print(json.dumps({"program": "localization", "triage": summary}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["retrieve", "localize"])
    args = parser.parse_args()
    {"retrieve": retrieve, "localize": localize}[args.stage]()
