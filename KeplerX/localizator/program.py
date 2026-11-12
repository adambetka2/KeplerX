"""Pass refined Vetter clocks and authenticated native pixels to Localizator."""

import contextlib
import hashlib
import io
import types
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from common import ROOT, digest, load_packet, phase, read, receipt, require, write
from localizator import accelerated_model, gaia_cone_search, gaia_field, mast_pixels

FETCH_CONNECTIONS = 8


def tce_table(candidates):
    frame = pd.DataFrame(candidates)[["period", "t0", "tdur"]].copy()
    require(np.isfinite(frame.to_numpy()).all(), "Nonfinite detected clock")
    require((frame[["period", "tdur"]] > 0).all().all(), "Invalid detected clock")
    frame["t0"] += 2454833.0
    return frame


def refined_candidates(candidates, vetted):
    """Join by signal ID; an unavailable fitted clock never becomes a detector clock."""
    rows = {row["signal_id"]: row for row in vetted["results"]}
    require(len(rows) == len(vetted["results"]), "Duplicate vetted signal ID")
    require(
        set(rows) == {row["signal_id"] for row in candidates}, "Vetted signals differ"
    )
    ready, unavailable = [], {}
    for candidate in candidates:
        disposition = rows[candidate["signal_id"]].get("disposition", {})
        if disposition.get("state") == "DISMISSED":
            unavailable[candidate["signal_id"]] = {
                "signal_id": candidate["signal_id"],
                "state": "NOT_RUN",
                "reason": f"Dismissed by vetting policy {vetted.get('policy')}",
                "quarters": [],
            }
            continue
        metrics = rows[candidate["signal_id"]]["metrics"]
        clock = {name: metrics.get(name) for name in ("period", "t0", "tdur")}
        if any(
            value is None or not np.isfinite(value) for value in clock.values()
        ) or any(clock[name] <= 0 for name in ("period", "tdur")):
            unavailable[candidate["signal_id"]] = {
                "signal_id": candidate["signal_id"],
                "state": "UNRESOLVED",
                "reason": "Vetter refined clock unavailable",
                "quarters": [],
            }
        else:
            ready.append(candidate | clock)
    return ready, unavailable


def transit_quarters(arrays, candidates):
    """Quarters holding a prepared cadence within a duration of some candidate's mid-transit time."""
    time = arrays["time"][arrays["mask"]]
    quarter = arrays["quarter"][arrays["mask"]]
    near = np.zeros(len(time), bool)
    for row in candidates:
        near |= np.abs(phase(time, row["period"], row["t0"])) < row["tdur"]
    return set(np.unique(quarter[near]).tolist())


def stored_pixels(items):
    """(item, bytes) of each TPF on disc, authenticated by its recorded sha256."""
    for item in items:
        raw = Path(item["path"]).read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == item["sha256"], "Pixel input changed"
        )
        yield item, raw


def open_tpf(raw, item, request, arrays):
    """A TPF's headers and the columns the localizer reads."""
    from astropy.time import Time

    with fits.open(io.BytesIO(accelerated_model.decompress(raw)), memmap=False) as hdus:
        headers = [hdu.header for hdu in hdus]
        table = hdus[1].data
        columns = {name: np.array(table[name]) for name in ("time", "flux", "flux_err")}
        cadence, quality = table["CADENCENO"], table["QUALITY"]
        require(headers[0]["KEPLERID"] == request["parent_kic"], "Wrong parent TPF")
        quarter = int(headers[0]["QUARTER"])
        require(quarter == item["quarter"], "Wrong TPF quarter")
        require(headers[1]["BJDREFI"] == 2454833, "Unexpected TPF time origin")
        require(headers[1]["TIMESYS"] == "TDB", "Unexpected TPF time scale")
        selected = arrays["mask"] & (arrays["quarter"] == quarter)
        allowed = np.isin(cadence, arrays["cadence"][selected]) & (quality == 0)
    epoch = Time(np.nanmin(columns["time"]) + 2454833, format="jd")
    return types.SimpleNamespace(
        headers=headers, quarter=quarter, allowed=allowed, epoch=epoch, **columns
    )


def kepler_localizer(tpf, tces, request):
    """Localize.from_tpf_info for an open Kepler TPF, with the channel PRF reused."""
    import lkprf
    from astropy.wcs import WCS

    from localizator.upstream.localize import Localize

    primary, table = tpf.headers[0], tpf.headers[1]
    return Localize(
        tpf.time + 2454833, tpf.flux, tpf.flux_err,
        tces, id=request["source_id"], mission="kepler", wcs=WCS(tpf.headers[2]),
        ra_bonus=request["ra_deg"], dec_bonus=request["dec_deg"],
        ra_targ=primary["ra_obj"], dec_targ=primary["dec_obj"],
        prf=accelerated_model.channel_prf(lkprf.KeplerPRF, primary["channel"]),
        epoch=tpf.epoch, headers=tpf.headers,
        channel=primary["channel"], origin_row=table["2CRV4P"],
        origin_col=table["1CRV4P"], quarter=primary["quarter"],
    )  # fmt: skip


def propagate_quarters(native, tpfs, request):
    """Each quarter's Gaia propagation as _search_and_propagate will ask it."""
    groups = {}
    for tpf in tpfs:
        primary = tpf.headers[0]
        shape = tpf.flux.shape[1:]
        key = (primary["ra_obj"], primary["dec_obj"], shape)
        groups.setdefault(key, []).append(tpf.epoch)
    # Widest aperture first: its cone serves the parent's narrower ones.
    for (ra, dec, shape), epochs in sorted(groups.items(), key=lambda g: -max(g[0][2])):
        table = native.module.query_vizier_background(ra, dec, shape, mission="kepler")
        if table is not None:
            points = [(ra, dec), (request["ra_deg"], request["dec_deg"])]
            native.propagation.prepare(table, epochs, points)


def observation(tpf, request, tces, out):
    loc = kepler_localizer(tpf, tces, request)
    quarter, allowed = tpf.quarter, tpf.allowed
    keep = allowed[loc.good_cad_mask]
    loc.time, loc.pix, loc.pix_err = (
        values[keep] for values in (loc.time, loc.pix, loc.pix_err)
    )
    loc.good_cad_mask &= allowed
    if len(loc.time) < 3:
        return None  # no usable pixels at the source's cadences this quarter
    require((loc.pix_err > 0).all(), "Nonpositive native pixel uncertainty")
    loc.build_design_matrix()
    if not len(loc.valid_tces):
        loc.transit_weights, loc.transit_weights_err = [], []
    else:
        accelerated_model.solve_transit_weights(loc)
    np.savez_compressed(
        out / f"quarter-{quarter:02d}.npz",
        time_bjd=loc.time,
        good_cadence_mask=loc.good_cad_mask,
        good_pixel_mask=loc.good_pix_mask,
        valid_tces=loc.valid_tces,
        depth=loc.transit_weights,
        depth_error=loc.transit_weights_err,
    )
    for index, error in zip(loc.valid_tces, loc.transit_weights_err, strict=True):
        usable = error[loc.good_pix_mask]
        require(
            np.isfinite(usable).all() and (usable > 0).all(),
            f"Native pixel fit has invalid uncertainties: Q{quarter}, TCE {index}",
        )
    return loc


def fit_signal(localizer, index, candidate, out):
    quarters = [
        loc.quarter for loc in localizer.loc_object_list if index in loc.valid_tces
    ]
    if not quarters:
        record = {
            "signal_id": candidate["signal_id"],
            "state": "NO_PIXEL_EVENTS",
            "clock": candidate,
            "quarters": [],
        }
        write(out / f"signal-{index + 1:02d}.json", record)
        return record
    result, metrics = localizer.fit_to_quarters(which_tce=index)
    errors = [metrics["sigmara"], metrics["sigmadec"]]
    has_errors = all(
        value is not None and np.isfinite(value) and value > 0 for value in errors
    )
    record = {
        "signal_id": candidate["signal_id"],
        "clock": candidate,
        "state": "MEASURED" if result.success and has_errors else "UNRESOLVED",
        "metrics": metrics,
        "fit_success": bool(result.success),
        "fit_message": result.message,
        "chi_square": result.chisqr,
        "degrees_of_freedom": result.nfree,
        "parameters": {
            name: {"value": p.value, "stderr": p.stderr}
            for name, p in result.params.items()
        },
        "quarters": quarters,
    }
    if has_errors:
        values = localizer.get_quarters_offset(metrics)
        record["offset"] = dict(
            zip(
                ("nsigma", "distance_degrees", "sigma_degrees", "mahalanobis"),
                values,
                strict=True,
            )
        )
    write(out / f"signal-{index + 1:02d}.json", record)
    return record


@contextlib.contextmanager
def native_localizer(out, field=None):
    """Native classes with Gaia DR3 from the field table or the ESA archive."""
    with contextlib.chdir(out):
        from localizator.upstream import localize as module

        original = module.query_vizier_background
        residual = module.quarters_prf_residual
        query, coord = module.propagate_query, module.propagate_coord
        splines = module.create_spline_design_matrix
        fit = module.minimize
        propagation = accelerated_model.Propagation(query, coord)
        module.query_vizier_background = gaia_cone_search.Cones(field)
        module.quarters_prf_residual = accelerated_model.PRFResidual(residual)
        module.minimize = accelerated_model.held_at_bounds(
            fit, module.quarters_prf_residual
        )
        module.propagate_query, module.propagate_coord = (
            propagation.query,
            propagation.coord,
        )
        module.create_spline_design_matrix = accelerated_model.spline_design_matrix
        try:
            yield types.SimpleNamespace(
                module=module,
                MultiLocalize=module.MultiLocalize,
                propagation=propagation,
            )
        finally:
            module.query_vizier_background = original
            module.quarters_prf_residual = residual
            module.propagate_query, module.propagate_coord = query, coord
            module.create_spline_design_matrix = splines
            module.minimize = fit


def pixel_inputs(request, arrays, candidates, connections):
    """The parent quarters that can hold the candidates' transits, opened from disc or streamed from MAST."""
    if "tpfs" in request:
        listed, client = request["tpfs"], None
    else:
        require(request.get("archive") == "mast", "No native pixel inputs supplied")
        client = mast_pixels.session(connections)
        listed = mast_pixels.listing(request["parent_kic"], client)
    require(listed, "No native pixel inputs supplied")
    quarters = [item["quarter"] for item in listed]
    require(len(quarters) == len(set(quarters)), "Duplicate TPF quarter")
    needed = transit_quarters(arrays, candidates)
    items = [item for item in listed if item["quarter"] in needed]
    covered = set(np.unique(arrays["quarter"][arrays["mask"]]).tolist())
    unread = sorted(set(quarters) - covered), sorted(covered & set(quarters) - needed)
    if client is None:
        source = stored_pixels(items)
    else:
        source = mast_pixels.fetch(items, client, connections)
    tpfs, receipts = [], []
    for item, raw in source:
        tpfs.append(open_tpf(raw, item, request, arrays))
        receipts.append(item)
    order = np.argsort([tpf.quarter for tpf in tpfs], kind="stable")
    return [tpfs[i] for i in order], [receipts[i] for i in order], unread


def localize(
    request, candidates, arrays, out, field=None, connections=FETCH_CONNECTIONS
):
    """Each candidate's localization, and the pixel files read."""
    with native_localizer(out, field) as native:
        require(request["coordinate_epoch"] == "J2016", "Use Gaia DR3 coordinates")
        frame = tce_table(candidates)
        try:
            tpfs, pixels, (uncovered, unread) = pixel_inputs(
                request, arrays, candidates, connections
            )
        except mast_pixels.PixelsUnreachable as error:
            return unresolved(candidates, error), []
        # A parent quarter without usable cadences of the source says nothing about it.
        uncovered += [tpf.quarter for tpf in tpfs if tpf.allowed.sum() < 3]
        tpfs = [tpf for tpf in tpfs if tpf.allowed.sum() >= 3]
        try:
            propagate_quarters(native, tpfs, request)
        except gaia_cone_search.ArchiveUnreachable as error:
            return unresolved(candidates, error), pixels
        observations, errors, inestimable = [], [], []
        for tpf in tpfs:
            try:
                loc = observation(tpf, request, frame, out)
            except accelerated_model.NotEstimable:
                # Trend columns that explain the transit say nothing about where its light comes from.
                inestimable.append(tpf.quarter)
                continue
            except np.linalg.LinAlgError as error:
                errors.append({"quarter": tpf.quarter, "error": str(error)})
                continue
            if loc is None:
                uncovered.append(tpf.quarter)
            else:
                observations.append(loc)
        left_out = {
            "quarters_without_transits": unread,
            "quarters_without_coverage": sorted(uncovered),
            "quarters_not_estimable": sorted(inestimable),
        }
        if not any(len(loc.valid_tces) for loc in observations):
            return [
                {
                    "signal_id": row["signal_id"],
                    "state": "UNRESOLVED"
                    if errors or inestimable
                    else "NO_PIXEL_EVENTS",
                    "clock": row,
                    "quarters": [],
                    "quarter_errors": errors,
                }
                | left_out
                for row in candidates
            ], pixels
        joint = native.MultiLocalize(observations)
        results = [
            fit_signal(joint, index, row, out) for index, row in enumerate(candidates)
        ]
        for row in results:
            row.update(left_out)
            if errors:
                row.update(state="UNRESOLVED", quarter_errors=errors)
        return results, pixels


def unresolved(candidates, error):
    """Every signal unresolved."""
    return [
        {"signal_id": row["signal_id"], "state": "UNRESOLVED", "clock": row,
         "quarters": [], "reason": str(error)}
        for row in candidates
    ]  # fmt: skip


def run(args, config):
    arrays, packet = load_packet(args.prepared)
    detections = read(args.detections / "detections.json")
    vetting_path = args.vetting / "vetting.json"
    vetted = read(vetting_path)
    request = read(args.request)
    require(
        packet["source_id"]
        == detections["source_id"]
        == request["source_id"]
        == vetted["source_id"],
        "Source identity differs",
    )
    require(packet["time_system"] == "BKJD_TDB", "Expected prepared BKJD_TDB")
    require(
        detections["prepared"]["sha256"] == digest(args.prepared / "packet.json"),
        "Detection and localization photons differ",
    )
    require(
        vetted["detections"]["sha256"] == digest(args.detections / "detections.json"),
        "Vetting and localization detections differ",
    )
    candidates = detections["candidates"]
    ready, unavailable = refined_candidates(candidates, vetted)
    settings = config.get("localization", {})
    folder = settings.get("gaia_field")
    # A field table returns the archive's cones bit for bit; otherwise the archive answers.
    field = None
    if folder and ready and (ROOT / folder / "field.json").exists():
        field = gaia_field.Field.load(ROOT / folder)
    connections = settings.get("fetch_connections", FETCH_CONNECTIONS)
    measured, pixels = (
        localize(request, ready, arrays, args.out.resolve(), field, connections)
        if ready
        else ([], [])
    )
    by_id = unavailable | {row["signal_id"]: row for row in measured}
    results = [by_id[row["signal_id"]] for row in candidates]
    result = {
        "source_id": packet["source_id"],
        "engine": "upstream.MultiLocalize",
        "state": "MEASURED" if candidates else "NO_DETECTIONS",
        "request": receipt(args.request),
        "detections": receipt(args.detections / "detections.json"),
        "vetting": receipt(vetting_path),
        "clock_basis": "Vetter refined period, epoch and duration",
        "prepared": receipt(args.prepared / "packet.json"),
        "results": results,
        "decision_policy_applied": False,
        "missing_quarters": request.get("missing_quarters", []),
        "pixels": pixels,
        "gaia": f"field table {folder}, archive beyond it" if field else "ESA archive",
        "mask": "Prepared cadences intersected with native finite pixels and QUALITY=0",
    }
    localized = [row for row in results if row["state"] != "NOT_RUN"]
    if candidates and not localized:
        result["state"] = "NOT_RUN"
    elif any(row["state"] != "MEASURED" for row in localized):
        result["state"] = "UNRESOLVED"
    write(args.out / "localization.json", result)
    return result


def arguments(parser):
    parser.add_argument("prepared", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("vetting", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("out", type=Path)
