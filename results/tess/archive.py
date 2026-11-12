"""MAST TESScut access and explicit aperture photometry of native FFI pixels."""

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

MAST = "https://mast.stsci.edu/tesscut/api/v0.1/"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def query(endpoint, parameters):
    url = endpoint + "?" + urllib.parse.urlencode(parameters)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = json.load(response)
    return {"queried_utc": datetime.now(UTC).isoformat(), "url": url, "response": data}


def coverage(host):
    return query(MAST + "sector", {"ra": host["ra_deg"], "dec": host["dec_deg"]})


def cutouts(host, sector, size, directory):
    directory.mkdir(parents=True, exist_ok=True)
    parameters = {
        "ra": host["ra_deg"],
        "dec": host["dec_deg"],
        "sector": sector,
        "x": size,
        "y": size,
        "units": "px",
    }
    manifest = directory / "download.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record["parameters"] != parameters:
            raise ValueError("Cached cutout request differs")
        for file in record["files"]:
            if digest(directory / file["name"]) != file["sha256"]:
                raise ValueError("Cached FITS hash differs")
        return [directory / f["name"] for f in record["files"]]
    url = MAST + "astrocut?" + urllib.parse.urlencode(parameters)
    archive = directory / "download.zip.partial"
    with (
        urllib.request.urlopen(url, timeout=300) as response,
        archive.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    files = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith(".fits"):
                continue
            path = directory / Path(name).name
            with z.open(name) as src, path.open("wb") as dest:
                shutil.copyfileobj(src, dest)
            files.append({"name": path.name, "sha256": digest(path)})
    if not files:
        raise ValueError("MAST returned no FITS cutout for a covered sector")
    write(
        manifest,
        {
            "parameters": parameters,
            "url": url,
            "files": files,
            "retrieved_utc": datetime.now(UTC).isoformat(),
        },
    )
    archive.unlink()
    return [directory / f["name"] for f in files]


def position(host, bjd):
    coord = SkyCoord(host["ra_deg"], host["dec_deg"], unit="deg")
    motion = host.get("astrometry")
    if motion is None:
        return coord
    years = Time(bjd, format="jd", scale="tdb").jyear - motion["epoch_jyear"]
    return coord.spherical_offsets_by(
        motion["pmra_mas_yr"] * years * u.mas, motion["pmdec_mas_yr"] * years * u.mas
    )


def masks(shape, xy, settings):
    yy, xx = np.indices(shape)
    distance = np.hypot(xx - xy[0], yy - xy[1])
    aperture = distance <= settings["aperture_radius_pixels"]
    background = (distance >= settings["background_inner_pixels"]) & (
        distance <= settings["background_outer_pixels"]
    )
    if not aperture.any() or not background.any() or np.any(aperture & background):
        raise ValueError("Empty or overlapping photometry masks")
    if not (-0.5 < xy[0] < shape[1] - 0.5 and -0.5 < xy[1] < shape[0] - 0.5):
        raise ValueError("Source falls outside cutout")
    weights = aperture.astype(float) - background * aperture.sum() / background.sum()
    return aperture, background, weights


def read_pixels(path, host, settings, spacecraft):
    with fits.open(path, memmap=False) as h:
        header = h[1].header
        if (
            header["TIMESYS"] != "TDB"
            or header["TIMEUNIT"] != "d"
            or header["TIMEREF"] != "SOLARSYSTEM"
            or header["TIMEPIXR"] != 0.5
        ):
            raise ValueError("Expected barycentric TDB mid-exposure times in days")
        archive_time = (
            np.array(h[1].data["TIME"], float) + header["BJDREFI"] + header["BJDREFF"]
        )
        archive_timecorr = np.array(h[1].data["TIMECORR"], float)
        time = archive_time - archive_timecorr
        target = position(host, np.nanmedian(time))
        finite = np.isfinite(time)
        target_timecorr = np.full(len(time), np.nan)
        target_timecorr[finite] = (
            spacecraft.get_barycentric_time_correction(
                Time(time[finite], format="jd", scale="tdb"),
                target.ra.deg,
                target.dec.deg,
            )
            / 86400
        )
        time += target_timecorr
        exposure = np.full(len(time), header["TIMEDEL"], float)
        if not np.all(exposure > 0):
            raise ValueError("Missing/nonpositive exposure duration")
        pixels = np.array(h[1].data["FLUX"], float)
        errors = np.array(h[1].data["FLUX_ERR"], float)
        quality = np.array(h[1].data["QUALITY"], int)
        wcs = WCS(h[2].header)
        xy = np.array(wcs.world_to_pixel(position(host, np.nanmedian(time))))
        aperture, background, weights = masks(pixels.shape[1:], xy, settings)
        used = aperture | background
        flux = np.sum(pixels[:, used] * weights[used], axis=1)
        error = np.sqrt(np.sum((errors[:, used] * weights[used]) ** 2, axis=1))
        good = (
            np.isfinite(time)
            & np.isfinite(flux)
            & np.isfinite(error)
            & (error > 0)
            & np.all(np.isfinite(errors[:, used]) & (errors[:, used] > 0), axis=1)
            & ((quality & settings["quality_bitmask"]) == 0)
        )
        meta = {key: h[0].header[key] for key in ("SECTOR", "CAMERA", "CCD")}
        meta.update(
            file=str(path.resolve()),
            sha256=digest(path),
            exposure_days=float(exposure[0]),
            live_exposure_days=header["EXPOSURE"],
            target_minus_archive_time_seconds=[
                float(np.nanmin((target_timecorr - archive_timecorr) * 86400)),
                float(np.nanmax((target_timecorr - archive_timecorr) * 86400)),
            ],
            archive_background_subtracted=bool(header["BACKAPP"]),
            cadences=len(time),
            usable_cadences=int(good.sum()),
            target_pixel_xy=xy.tolist(),
            flux_unit=h[1].columns["FLUX"].unit,
            time_reference={
                "BJDREFI": header["BJDREFI"],
                "BJDREFF": header["BJDREFF"],
                "TIMESYS": header["TIMESYS"],
            },
            wcs_header=dict(h[2].header),
            astrometry_status="PROPAGATED"
            if host.get("astrometry")
            else "FIXED_INPUT_POSITION",
        )
    return {
        "time": time,
        "archive_time": archive_time,
        "archive_timecorr": archive_timecorr,
        "target_timecorr": target_timecorr,
        "exposure": exposure,
        "pixels": pixels,
        "pixel_error": errors,
        "quality": quality,
        "flux": flux,
        "error": error,
        "good": good,
        "meta": meta,
        "aperture": aperture,
        "background": background,
    }
