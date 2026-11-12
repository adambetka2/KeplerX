"""Read one KBonus lane of a source."""

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from scipy.ndimage import median_filter

from common import read, receipt, require, save_packet

COLUMNS = {
    "time": "time_bkjd",
    "cadence": "cadenceno",
    "quarter": "quarter",
    "flux": "psf_flat_kbonus_flux",
    "error": "psf_flat_kbonus_flux_err",
    "quality": "sap_quality",
    "flatten_mask": "flatten_mask",
}
# Each view's LC2 flux columns, and whether it is flattened here.
VIEWS = {
    "psf_flat": {"flux": "psf_flat_kbonus_flux", "error": "psf_flat_kbonus_flux_err",
                 "flattened_here": False, "decrowded_here": False},
    "sap_flat": {"flux": "sap_stitched_flux", "error": "sap_stitched_flux_err",
                 "flattened_here": True, "decrowded_here": True},
}  # fmt: skip
EXCLUSION_BITS = {
    "1": "nonfinite",
    "2": "nonpositive_error",
    "4": "quality",
    "8": "flatten_mask",
    "16": "invalid_quarter_normalization",
    "32": "retained_flag_outlier",
    "64": "field_artifact_epoch",
    "128": "star_share_below_floor",
}


def view_columns(view):
    """The LC2 columns of `view`, under COLUMNS' keys."""
    require(view in VIEWS, f"Unknown photometry view {view}")
    return COLUMNS | {"flux": VIEWS[view]["flux"], "error": VIEWS[view]["error"]}


def source_table(path, source_id, columns=None):
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.get_field_index("source_id")
    require(index >= 0, f"No source_id column: {path}")
    groups = []
    for i in range(parquet.num_row_groups):
        stat = parquet.metadata.row_group(i).column(index).statistics
        require(
            stat is not None and stat.min == stat.max,
            "LC2 row group is not source-specific",
        )
        if int(stat.min) == int(source_id):
            groups.append(i)
    require(groups, f"Source {source_id} absent from {path}")
    return parquet.read_row_groups(groups, columns=columns)


def locate_source(source_id, index):
    with sqlite3.connect(
        f"file:{Path(index).resolve()}?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT volume,shard FROM sources WHERE gaia_source_id=?", (str(source_id),)
        ).fetchall()
    require(len(set(rows)) == 1, f"Source lookup is missing or ambiguous: {source_id}")
    volume, shard = rows[0]
    folder = Path("/Volumes") / volume / "data/kbonus/lc2" / f"shard-{int(shard):04d}"
    require(folder.is_dir(), f"Source volume/shard unavailable: {folder}")
    return folder


def load_lc2(request):
    source_id = request["source_id"]
    require(
        isinstance(source_id, str) and source_id.isdecimal(),
        "Gaia source_id must be a decimal string",
    )
    folder = locate_source(source_id, request["index"])
    columns = view_columns(request.get("view", "psf_flat"))
    table = source_table(folder / "cadence.parquet", source_id, list(columns.values()))
    arrays = {key: table[name].to_numpy() for key, name in columns.items()}
    source = source_table(folder / "source.parquet", source_id).to_pylist()
    require(len(source) == 1, "Expected one source metadata row")
    provenance = {
        "shard": str(folder),
        "source_metadata": source[0],
        "index": receipt(request["index"]),
    }
    if VIEWS[request.get("view", "psf_flat")]["decrowded_here"]:
        from photometry.star_share import lc2_quarters

        arrays["star_share"], provenance["star_share"] = star_shares(
            arrays["quarter"],
            *lc2_quarters(folder, source_id, source[0]["primary_header_cards"]),
        )
    return arrays, provenance


def star_shares(quarter, g_mag, bp_rp, quarters):
    """Each cadence's share of its quarter's SAP."""
    from photometry.star_share import per_cadence, sap_shares

    shares = sap_shares(g_mag, bp_rp, quarters)
    record = {"g_mag": g_mag, "bp_rp": bp_rp, "quarters": {
        str(q): v | {"share": shares[q]} for q, v in sorted(quarters.items())}}  # fmt: skip
    return per_cadence(quarter, shares), record


def load_npz(request):
    require(request["input_kind"] == "native_arrays", "Unknown photometry input kind")
    require(request["view"] in VIEWS, f"Unknown photometry view {request['view']}")
    with np.load(request["path"], allow_pickle=False) as data:
        require(
            set(COLUMNS).issubset(data.files),
            "Native arrays are missing required columns",
        )
        arrays = {key: data[key] for key in COLUMNS}
        if VIEWS[request["view"]]["decrowded_here"]:
            require("star_share" in data.files, "Native SAP arrays need star_share")
            arrays["star_share"] = data["star_share"]
    return arrays, {
        "native_arrays": receipt(request["path"]),
        "basis": request["provenance"],
    }


def load_kbonus(path):
    """The delivered Flat and its mask live in LIGHTCURVE_STITCHED."""
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header
        table = hdus["LIGHTCURVE_STITCHED"]
        require(header["TIMESYS"] == "TDB", "Unexpected KBonus time scale")
        require(
            table.columns["TIME"].unit == "BJD - 2454833",
            "Unexpected KBonus time origin",
        )
        names = (
            "TIME",
            "CADENCENO",
            "QUARTER",
            "PSF_FLAT_FLUX",
            "PSF_FLAT_FLUX_ERR",
            "SAP_QUALITY",
            "FLATTEN_MASK",
        )
        arrays = {
            key: np.array(table.data[name])
            for key, name in zip(COLUMNS, names, strict=True)
        }
        info = {
            "source_id": header["GAIAID"].removeprefix("Gaia DR3 "),
            "parent_kic": int(header["TPFORG"]),
            "kic": int(header["KEPLERID"]),
            "ra_deg": float(header["RA_OBJ"]),
            "dec_deg": float(header["DEC_OBJ"]),
            "coordinate_epoch": f"J{header['EQUINOX']:.0f}",
            "exposure_days": float(header["XPOSURE"]) / 86400,
            "time_system": "BKJD_TDB",
            "view": "psf_flat",
        }
    return arrays, info


def field_artifacts(time, path):
    """Cadences inside the field-wide artifact epochs listed at `path`."""
    from common import ROOT

    windows = np.asarray(read(ROOT / path)["windows_bkjd"], dtype=float)
    inside = np.zeros(len(time), dtype=bool)
    for start, end in windows:
        inside |= (time >= start) & (time <= end)
    return inside


def retained_outliers(arrays, kept, sigma, window=3):
    """Retained flagged cadences that stand alone."""
    outliers = np.zeros(len(kept), dtype=bool)
    usable = arrays["mask"]
    for quarter in np.unique(arrays["quarter"][kept]):
        group = np.flatnonzero(usable & (arrays["quarter"] == quarter))
        if len(group) < window:
            continue
        flux = arrays["flux"][group]
        residual = flux - median_filter(flux, window, mode="nearest")
        clean = ~kept[group]
        # Point-to-point noise as an unbiased Gaussian sigma (a three-point median residual is often 0).
        basis = np.diff(flux[clean]) if clean.sum() >= 20 else np.diff(flux)
        scale = 1.4826 * np.median(np.abs(basis - np.median(basis))) / np.sqrt(2)
        if not np.isfinite(scale) or scale <= 0:
            continue
        outliers[group[kept[group] & (np.abs(residual) > sigma * scale)]] = True
    return outliers


def prepare_arrays(arrays, config, view="psf_flat"):
    lengths = {len(value) for value in arrays.values()}
    require(
        len(lengths) == 1 and next(iter(lengths)) > 0,
        "Empty or inconsistent native arrays",
    )
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    arrays["flux"] = arrays["flux"].astype(float)
    arrays["error"] = arrays["error"].astype(float)
    order = np.argsort(arrays["time"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    arrays["native_index"] = order
    reasons = np.zeros(len(order), dtype=np.uint8)
    finite = (
        np.isfinite(arrays["time"])
        & np.isfinite(arrays["flux"])
        & np.isfinite(arrays["error"])
    )
    reasons[~finite] |= 1
    reasons[arrays["error"] <= 0] |= 2
    # Flags whose cadences scatter like clean ones may be retained.
    retained = int(sum(config.get("retained_quality_bits", [])))
    quality = arrays["quality"].astype(np.int64)
    flatten = arrays["flatten_mask"].astype(np.int64)
    reasons[(quality & ~retained) != config["quality_equals"]] |= 4
    reasons[(flatten & ~retained) != config["flatten_mask_equals"]] |= 8
    if config.get("field_artifact_epochs"):
        reasons[field_artifacts(arrays["time"], config["field_artifact_epochs"])] |= 64
    arrays["exclusion_bits"] = reasons
    arrays["mask"] = reasons == 0
    if retained and config.get("retained_outlier_sigma"):
        kept = arrays["mask"] & (((quality | flatten) & retained) != 0)
        outliers = retained_outliers(arrays, kept, config["retained_outlier_sigma"])
        arrays["exclusion_bits"][outliers] |= 32
        arrays["mask"][outliers] = False
    good_time = arrays["time"][arrays["mask"]]
    # 155,624 of the 406,547 LC2 sources carry SAP flux only.
    require(
        np.isfinite(arrays["flux"]).any(),
        f"No {view} flux: every cadence is non-finite",
    )
    require(
        len(good_time) >= 3,
        f"Insufficient valid cadences: {len(good_time)} pass the masks",
    )
    require(np.all(np.diff(good_time) > 0), "Duplicate valid timestamps")
    arrays["native_flux"] = arrays["flux"].copy()
    arrays["native_error"] = arrays["error"].copy()
    normalizations = {}
    for quarter in np.unique(arrays["quarter"]):
        group = arrays["quarter"] == quarter
        usable = group & arrays["mask"]
        if not np.any(usable):
            continue
        scale = float(np.median(arrays["flux"][usable]))
        if not (np.isfinite(scale) and scale > 0):
            # A failed PSF fit can leave a quarter entirely negative; that quarter is dropped.
            arrays["exclusion_bits"][group] |= 16
            arrays["mask"][group] = False
            continue
        arrays["flux"][group] /= scale
        arrays["error"][group] /= scale
        normalizations[str(quarter)] = scale
    require(
        np.count_nonzero(arrays["mask"]) >= 3,
        "No quarter can be normalized: every usable median is non-positive",
    )
    if VIEWS[view]["flattened_here"]:
        for quarter in flatten_like_kbonus(arrays):
            normalizations.pop(str(quarter))
        require(
            np.count_nonzero(arrays["mask"]) >= 3,
            "No quarter can be flattened: every trend reaches zero",
        )
    if VIEWS[view]["decrowded_here"]:
        for quarter in decrowd(arrays, normalizations):
            normalizations.pop(str(quarter))
        require(
            np.count_nonzero(arrays["mask"]) >= 3,
            "No quarter holds enough of the star's light",
        )
    return arrays, normalizations


def flatten_like_kbonus(arrays):
    """Divide the usable cadences of each quarter by KBonus's emulated trend."""
    from photometry.flattening import trend

    dropped = []
    for quarter in np.unique(arrays["quarter"][arrays["mask"]]):
        usable = np.flatnonzero(arrays["mask"] & (arrays["quarter"] == quarter))
        fit = trend(
            arrays["time"][usable], arrays["flux"][usable],
            arrays["quarter"][usable], [quarter],
        )  # fmt: skip
        if not np.all(fit > 0):
            group = arrays["quarter"] == quarter
            arrays["exclusion_bits"][group] |= 16
            arrays["mask"][group] = False
            dropped.append(int(quarter))
            continue
        for name in ("flux", "error", "native_flux", "native_error"):
            arrays[name][usable] /= fit
    return dropped


def decrowd(arrays, normalizations):
    """Divide each quarter's deviations from its level by the star's share of the quarter's SAP."""
    from photometry.star_share import SHARE_FLOOR

    dropped = []
    for quarter in np.unique(arrays["quarter"][arrays["mask"]]):
        group = arrays["quarter"] == quarter
        usable = np.flatnonzero(arrays["mask"] & group)
        share = float(arrays["star_share"][usable[0]])
        if not share >= SHARE_FLOOR:
            arrays["exclusion_bits"][group] |= 128
            arrays["mask"][group] = False
            dropped.append(int(quarter))
            continue
        if share < 1:
            level = normalizations[str(quarter)]
            arrays["flux"][usable] = 1 + (arrays["flux"][usable] - 1) / share
            arrays["error"][usable] /= share
            arrays["native_flux"][usable] = level * arrays["flux"][usable]
            arrays["native_error"][usable] = level * arrays["error"][usable]
    return dropped


def run(args, config):
    request = read(args.request)
    require(request["time_system"] == "BKJD_TDB", "Declare native BKJD_TDB timestamps")
    require(0 < request["exposure_days"] < 1, "Actual exposure_days is required")
    require(isinstance(request["source_id"], str), "source_id must remain a string")
    loader = load_lc2 if request["input_kind"] == "lc2" else load_npz
    arrays, provenance = loader(request)
    view = request.get("view", "psf_flat")
    arrays, scales = prepare_arrays(arrays, config["photometry"], view)
    metadata = {
        "schema": 1,
        "source_id": request["source_id"],
        "view": view,
        "time_system": "BKJD_TDB",
        "exposure_days": request["exposure_days"],
        "flux_convention": request["flux_convention"],
        "stellar": request["stellar"],
        "request": receipt(args.request),
        "provenance": provenance,
        "normalization_by_quarter": scales,
        "additional_detrending": False,
        "stored_cadences": len(arrays["time"]),
        "usable_cadences": int(arrays["mask"].sum()),
        "exclusion_bit_definitions": EXCLUSION_BITS,
    }
    if VIEWS[view]["decrowded_here"]:
        metadata["star_share_by_quarter"] = share_by_quarter(arrays)
    save_packet(args.out, arrays, metadata)
    return metadata


def share_by_quarter(arrays):
    """{quarter: the star's share} over the quarters with usable cadences."""
    return {
        str(int(q)): float(arrays["star_share"][arrays["quarter"] == q][0])
        for q in np.unique(arrays["quarter"][arrays["mask"]])
    }


def arguments(parser):
    parser.add_argument("request", type=Path)
    parser.add_argument("out", type=Path)
