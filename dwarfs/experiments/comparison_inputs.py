"""Export metadata-selected native pixels and matched delivered photometry."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import reprocess
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
NATIVE = {
    "time": "TIME",
    "cadence": "CADENCENO",
    "quality": "QUALITY",
    "flux": "FLUX",
    "flux_err": "FLUX_ERR",
    "flux_bkg": "FLUX_BKG",
    "flux_bkg_err": "FLUX_BKG_ERR",
    "pos_corr1": "POS_CORR1",
    "pos_corr2": "POS_CORR2",
}
QDIAG = ("sap_flux", "sap_flux_err", "sap_background", "psf_flux", "psf_flux_err")
STITCHED = (
    "sap_stitched_flux",
    "sap_stitched_flux_err",
    "psf_stitched_flux",
    "psf_stitched_flux_err",
    "psf_flat_kbonus_flux",
    "psf_flat_kbonus_flux_err",
)


def table(folder, name, source, quarter=None):
    filters = [("source_id", "=", int(source))]
    if quarter is not None:
        filters.append(("quarter", "=", int(quarter)))
    return pq.read_table(folder / f"{name}.parquet", filters=filters)


def aligned(table_data, cadences):
    data = {key: np.asarray(value) for key, value in table_data.to_pydict().items()}
    _, native, delivered = np.intersect1d(
        cadences, data["cadenceno"], return_indices=True
    )
    present = np.zeros(len(cadences), dtype=bool)
    present[native] = True
    values = {}
    for key, value in data.items():
        if value.dtype.kind not in "iufb":
            continue
        result = np.full(len(cadences), np.nan)
        result[native] = value[delivered]
        values[key] = result
    return values, present


def aperture_mask(folder, entry, header, shape):
    row = table(folder, "aperture", entry["source_id"], entry["quarter"]).to_pylist()[0]
    headers = table(folder, "header", entry["source_id"]).to_pylist()
    cards = next(
        x["header_cards"]
        for x in headers
        if x["extname"] == f"LIGHTCURVE_Q{entry['quarter']}"
    )
    delivered = fits.Header.fromstring(cards, sep="\n")
    raw = np.frombuffer(row["mask_uint8_c_order"], dtype=np.uint8).reshape(
        row["height"], row["width"]
    )
    y, x = np.nonzero(raw)
    y = y + int(delivered["TPF_ROW"]) - int(header["2CRV5P"])
    x = x + int(delivered["TPF_COL"]) - int(header["1CRV5P"])
    if not len(y) or np.any((y < 0) | (x < 0) | (y >= shape[0]) | (x >= shape[1])):
        raise ValueError("Nonzero SAP aperture outside native stamp")
    result = np.zeros(shape, dtype=bool)
    result[y, x] = True
    return result, delivered


def common_support(arrays, qdiag, stitched, present):
    use = present & (arrays["quality"] == 0) & (qdiag["sap_quality"] == 0)
    use &= (stitched["sap_quality"] == 0) & (stitched["flatten_mask"] == 0)
    for name in ("time", "pos_corr1", "pos_corr2"):
        use &= np.isfinite(arrays[name])
    for values, keys in ((qdiag, QDIAG), (stitched, STITCHED)):
        for key in keys:
            use &= np.isfinite(values[key])
            if key.endswith("_err"):
                use &= values[key] > 0
    if use.sum() < 500 or np.ptp(arrays["time"][use]) < 30:
        raise ValueError("Insufficient all-view common quarter support")
    return use


def export(entry, destination):
    folder = ROOT / f"discSSD/data/kbonus/lc2/shard-{entry['shard']:04d}"
    with fits.open(entry["tpf_path"], memmap=False) as hdus:
        ph, header = hdus[0].header.copy(), hdus[1].header.copy()
        arrays = {
            key: np.asarray(hdus[1].data[column]).copy()
            for key, column in NATIVE.items()
        }
        shape = arrays["flux"].shape[1:]
        if np.prod(shape) > 100 or int(ph["QUARTER"]) != int(entry["quarter"]):
            raise ValueError("Wrong native quarter or stamp above100pixels")
        aperture, delivered_header = aperture_mask(folder, entry, header, shape)
        x, y = WCS(hdus[2].header).all_world2pix(entry["ra_deg"], entry["dec_deg"], 0)
    qdiag, qpresent = aligned(
        table(folder, "qdiag", entry["source_id"], entry["quarter"]), arrays["cadence"]
    )
    stitched, spresent = aligned(
        table(folder, "cadence", entry["source_id"], entry["quarter"]),
        arrays["cadence"],
    )
    use = common_support(arrays, qdiag, stitched, qpresent & spresent)
    background_replaced = bool(np.any(qdiag["sap_background"][use] != 0))
    replay = np.sum(arrays["flux"][:, aperture], axis=1, dtype=float)
    if background_replaced:
        replay += (
            np.sum(arrays["flux_bkg"][:, aperture], axis=1, dtype=float)
            - qdiag["sap_background"]
        )
    discrepancy = np.max(
        np.abs(replay[use] - qdiag["sap_flux"][use]) / np.abs(qdiag["sap_flux"][use])
    )
    if discrepancy > 1e-5:
        raise ValueError(f"Calibrated SAP replay differs by {discrepancy}")
    payload = {key: value[use] for key, value in arrays.items()}
    payload.update({"delivered_" + key: value[use] for key, value in qdiag.items()})
    payload.update({"delivered_" + key: stitched[key][use] for key in STITCHED})
    payload.update(
        aperture=aperture,
        delivered_present=np.ones(use.sum(), dtype=bool),
        original_native_indices=np.flatnonzero(use),
        flatten_mask=stitched["flatten_mask"][use],
    )
    name = f"{entry['source_id']}-q{entry['quarter']}"
    path = destination / (name + "-fullquarter.npz")
    np.savez_compressed(path, **payload)
    return {
        **entry,
        "case_id": name,
        "path": str(path.resolve()),
        "sha256": reprocess.digest(path),
        "tpf_sha256": reprocess.digest(entry["tpf_path"]),
        "background_replaced": background_replaced,
        "headers": {
            key: int(ph[key]) if key == "CHANNEL" else header[key]
            for key in ("CHANNEL", "1CRV5P", "2CRV5P")
        },
        "target": {
            "local_row": float(y),
            "local_column": float(x),
            "row": float(y + header["2CRV5P"]),
            "column": float(x + header["1CRV5P"]),
        },
        "target_header": {
            key: delivered_header.get(key)
            for key in ("PSFFRAC", "CROWDSAP", "FLFRCSAP")
        },
        "common_rows": int(use.sum()),
        "span_days": float(np.ptp(arrays["time"][use])),
        "sap_replay_max_relative_error": float(discrepancy),
        "source_class": entry["source_class"],
    }


def prepare_case(entry, cbv_entry, options):
    with np.load(entry["path"], allow_pickle=False) as exported:
        cadences = exported["cadence"]
    with np.load(cbv_entry["path"], allow_pickle=False) as stored:
        _, left, right = np.intersect1d(
            cadences, stored["cadenceno"], return_indices=True
        )
        if not np.array_equal(left, np.arange(len(cadences))):
            raise ValueError("CBV does not cover every exported native cadence")
        aligned_cbv = {key: stored[key][right] for key in stored.files}
    cbv_path = options.out / (entry["case_id"] + "-aligned-cbv.npz")
    np.savez_compressed(cbv_path, **aligned_cbv)
    result = reprocess.prepare_case(entry, {"path": str(cbv_path)}, options)
    result["original_cbv_sha256"] = reprocess.digest(cbv_entry["path"])
    with np.load(result["path"], allow_pickle=False) as stored:
        arrays = dict(stored)
    with np.load(entry["path"], allow_pickle=False) as exported:
        rows = arrays["native_row_indices"]
        arrays["delivered_sap_error"] = exported["delivered_sap_flux_err"][rows]
        arrays["delivered_qdiag_psf"] = arrays["delivered_psf"].copy()
        arrays["delivered_qdiag_psf_error"] = arrays["delivered_psf_error"].copy()
        arrays["delivered_psf_flat"] = exported["delivered_psf_flat_kbonus_flux"][rows]
        arrays["delivered_psf_flat_error"] = exported[
            "delivered_psf_flat_kbonus_flux_err"
        ][rows]
        arrays["flatten_mask"] = exported["flatten_mask"][rows]
        arrays["original_native_indices"] = exported["original_native_indices"][rows]
        for key in STITCHED[:4]:
            arrays["delivered_" + key] = exported["delivered_" + key][rows]
    if len(arrays["time"]) < 500 or np.ptp(arrays["time"]) < 30:
        raise ValueError("Insufficient final common support after CBV mask")
    np.savez_compressed(result["path"], **arrays)
    result.update(
        sha256=reprocess.digest(result["path"]),
        source_class=entry["source_class"],
        quarter=entry["quarter"],
        parent_tpf=entry["parent_tpf"],
        parent_id=str(entry["parent_tpf"]),
        selection_hash=entry["selection_hash"],
        photometry_units="electrons/second",
        delivered_psf_policy="Native quarter correctedPSF (app psf_qdiag); stitched separately preserved",
        delivered_sap_policy="Native qdiagSAP exact pixel replay; stitchedSAP separately preserved",
        delivered_psf_flat_policy="Actual delivered stitched PSF Flat; unavailable preprocessing operator",
    )
    return result


def prepare(options):
    options.out.mkdir(parents=True, exist_ok=False)
    entries = json.loads(options.manifest.read_text())["cases"]
    cbvs = {
        row["case_id"]: row["selected_channel"][0]
        for row in json.loads(options.cbv_manifest.read_text())
    }
    plan = {
        "source_sha256": reprocess.digest(__file__),
        "extractor_sha256": reprocess.digest(reprocess.__file__),
        "manifest_sha256": reprocess.digest(options.manifest),
        "cbv_manifest_sha256": reprocess.digest(options.cbv_manifest),
        "basis_sha256": reprocess.digest(options.basis_source),
        "mission_files": {
            str(p.resolve()): reprocess.digest(p)
            for p in sorted(options.prf_cache.glob("*.fits"))
        },
    }
    (options.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for entry in entries:
        try:
            results.append(prepare_case(entry, cbvs[entry["case_id"]], options))
        except Exception as error:
            (options.out / "first-failure.json").write_text(
                json.dumps(
                    {"case_id": entry["case_id"], "error": repr(error)}, indent=2
                )
                + "\n"
            )
            raise
        (options.out / "manifest.json").write_text(
            json.dumps(
                plan | {"cases": results, "complete": len(results) == len(entries)},
                indent=2,
            )
            + "\n"
        )
        print(entry["case_id"], "prepared", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "prepare"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cbv-manifest", type=Path)
    parser.add_argument("--prf-cache", type=Path)
    parser.add_argument("--basis-source", type=Path)
    parser.add_argument("--learned-dir", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    entries = json.loads(args.manifest.read_text())
    results = []
    for entry in entries:
        try:
            results.append(export(entry, args.out))
        except Exception as error:
            (args.out / "first-failure.json").write_text(
                json.dumps(
                    {"source_id": entry["source_id"], "error": repr(error)}, indent=2
                )
                + "\n"
            )
            raise
        (args.out / "fullquarter-manifest.json").write_text(
            json.dumps(
                {"cases": results, "complete": len(results) == len(entries)}, indent=2
            )
            + "\n"
        )
        print(results[-1]["case_id"], "exported", flush=True)


if __name__ == "__main__":
    main()
