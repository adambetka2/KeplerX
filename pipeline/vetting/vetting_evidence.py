"""Catalogue, shared-time and extraction measurements for every proposed event."""

import csv
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from lc2 import photometry, uses_flatten_mask, vetting_points
from runtime import execution_workers
from vetting import (
    CONFIG,
    EXTRA_CATALOGUES,
    ROOT,
    RUN,
    alert_intervals,
    check_run,
    counter_deltas,
    counter_snapshot,
    digest,
    elapsed,
    log_worker,
    observation,
    query_evidence,
    read,
    save,
    started,
    tap,
)

EVIDENCE_CONTEXT = ()


def promoted_cohort():
    from vetting_triage import measured_signals

    return [
        proposal
        for proposal in read(RUN / "cohort.json")
        if measured_signals(read(RUN / "sources" / f"{proposal['source_id']}.json"))
    ]


def exact_queries(cohort, metadata):
    jobs = []
    gaia_endpoint = read(CONFIG)["gaia_tap_endpoint"]
    ids = [row["source_id"] for row in cohort]
    for index in range(0, len(ids), 20):
        batch = ids[index : index + 20]
        query = (
            "select g.source_id,g.ra,g.dec,g.pmra,g.pmdec,g.phot_g_mean_mag,"
            "a.radius_flame,a.radius_flame_lower,a.radius_flame_upper,"
            "a.mass_flame,a.mass_flame_lower,a.mass_flame_upper,"
            "a.teff_gspphot,a.teff_gspphot_lower,a.teff_gspphot_upper "
            "from gaiadr3.gaia_source as g left outer join "
            "gaiadr3.astrophysical_parameters as a on g.source_id=a.source_id "
            "where g.source_id in (" + ",".join(batch) + ")"
        )
        jobs.append(
            (
                f"gaia-{index:03d}",
                gaia_endpoint,
                query,
            )
        )
        names = [f"'Gaia DR3 {sid}'" for sid in batch]
        names.extend(
            f"'KIC {metadata[sid]['kic']}'" for sid in batch if metadata[sid]["kic"]
        )
        query = (
            "select i.id,b.main_id,b.otype,b.nbref,b.oid from ident as i "
            "join basic as b on i.oidref=b.oid where i.id in (" + ",".join(names) + ")"
        )
        jobs.append(
            (
                f"simbad-{index:03d}",
                "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
                query,
            )
        )
    with ThreadPoolExecutor(max_workers=execution_workers(read(CONFIG))["network_workers"]) as pool:
        list(pool.map(lambda job: tap(*job), jobs))


def scenes(cohort, metadata):
    radius = read(CONFIG)["neighbour_radius_arcsec"]
    gaia_endpoint = read(CONFIG)["gaia_tap_endpoint"]
    jobs = []
    for i in range(0, len(cohort), 10):
        batch = [metadata[r["source_id"]] for r in cohort[i : i + 10]]
        cones = [
            f"1=contains(point('ICRS',ra,dec),circle('ICRS',"
            f"{r['ra_deg']},{r['dec_deg']},{radius / 3600}))"
            for r in batch
        ]
        query = (
            "select source_id,ra,dec,pmra,pmdec,phot_g_mean_mag "
            "from gaiadr3.gaia_source where "
            + " or ".join(cones)
        )
        jobs.append(
            (f"scene-{i:03d}", gaia_endpoint, query)
        )
    with ThreadPoolExecutor(max_workers=execution_workers(read(CONFIG))["network_workers"]) as pool:
        list(pool.map(lambda job: tap(*job), jobs))


def raw_curve(source):
    return observation(source["source_id"])[1]


def fixed_interval(data, centre, duration, view):
    curve = photometry(data, view)
    t, y, err = curve["time"], curve["flux"], curve["error"]
    config = read(CONFIG)
    use = np.isfinite(t + y + err) & vetting_points(
        curve, config["quality_bitmask"], uses_flatten_mask(view, config)
    )
    use &= abs(t - centre) <= max(4 * duration, 0.5)
    t, y, err = t[use] - centre, y[use], err[use]
    inside = abs(t) <= duration / 2
    if inside.sum() == 0 or (~inside).sum() < 3 or len(t) < 5:
        return {"status": "INSUFFICIENT_COVERAGE"}
    baseline = np.median(y[~inside])
    if baseline <= 0:
        return {"status": "NONPOSITIVE_BASELINE"}
    y, err = y / baseline - 1, err / baseline
    design = np.column_stack([np.ones(len(t)), t, -inside.astype(float)])
    if np.linalg.matrix_rank(design) < 3:
        return {"status": "SINGULAR_DESIGN"}
    coef = np.linalg.lstsq(design / err[:, None], y / err, rcond=None)[0]
    residual = y - design @ coef
    covariance = np.linalg.inv((design / err[:, None]).T @ (design / err[:, None]))
    scale = max(float(np.sum((residual / err) ** 2) / (len(t) - 3)), 1)
    return {
        "status": "MEASURED",
        "depth_fraction": float(coef[-1]),
        "formal_error": float(np.sqrt(covariance[-1, -1])),
        "residual_scaled_error": float(np.sqrt(covariance[-1, -1] * scale)),
        "in_event_points": int(inside.sum()),
        "out_event_points": int((~inside).sum()),
        "before_points": int((t < -duration / 2).sum()),
        "after_points": int((t > duration / 2).sum()),
    }


def shared_alerts():
    index = read(CONFIG).get("shared_alert_index")
    if index is not None:
        path = Path(index["path"])
        if digest(path) != index["sha256"]:
            raise ValueError("Shared-alert index changed after vetting configuration")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return connection, None, True
    records = []
    metadata = read(RUN / "metadata.json")
    for source in read(RUN / "cohort.json"):
        source_metadata = metadata[source["source_id"]]
        for alert in alert_intervals(source):
            records.append(
                (
                    float(alert["start_bkjd"]),
                    float(alert["end_bkjd"]),
                    source["source_id"],
                    source_metadata["origin_tpf_kic"],
                )
            )
    records.sort()
    return (
        records,
        alert_time_bounds(records),
        read(CONFIG).get("shared_alert_scope_complete") is True,
    )


def alert_time_bounds(records):
    windows = np.array([(r[0], r[1]) for r in records], dtype=float).reshape(-1, 2)
    starts = np.ascontiguousarray(windows[:, 0])
    if (
        not np.isfinite(windows).all()
        or np.any(windows[:, 1] < starts)
        or np.any(np.diff(starts) < 0)
    ):
        raise ValueError("Local coincidence windows must be finite, ordered intervals")
    return starts, np.maximum.accumulate(windows[:, 1])


def coincident_sources(shared, shared_time, start, end, sid, parent):
    if shared_time is None:
        margin = max(abs(start), abs(end), 1) * 1e-6
        rows = shared.execute(
            "SELECT a.source_id, a.parent_tpf FROM windows AS w "
            "JOIN alerts AS a ON a.id=w.id "
            "WHERE w.start_bkjd<=? AND w.end_bkjd>=? "
            "AND a.start_bkjd<=? AND a.end_bkjd>=? AND a.source_id<>?",
            (end + margin, start - margin, end, start, sid),
        )
        return {source: other_parent == parent for source, other_parent in rows}
    starts, maximum_ends = shared_time
    first = np.searchsorted(maximum_ends, start, side="left")
    limit = np.searchsorted(starts, end, side="right")
    return {
        source: other_parent == parent
        for begin, stop, source, other_parent in shared[first:limit]
        if stop >= start and source != sid
    }


def extra_catalogues():
    rows = []
    for path in EXTRA_CATALOGUES:
        with path.open() as stream:
            for r in csv.DictReader(stream):
                if "KIC" in r:
                    ra, dec, name = r["_RA"], r["_DE"], f"KIC {r['KIC']}"
                else:
                    ra, dec, name = r["RA"], r["Dec"], f"CTOI {r['CTOI']}"
                try:
                    coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
                except ValueError:
                    continue
                rows.append(
                    {
                        "ra": coord.ra.deg,
                        "dec": coord.dec.deg,
                        "name": name,
                        "catalogue": path.name,
                        "record": r,
                    }
                )
    return rows


def source_scene(source, queries):
    radius = read(CONFIG)["neighbour_radius_arcsec"]
    items = [row for query in queries for row in query["rows"]]
    rows = {}
    if items:
        target = SkyCoord(source["ra_deg"] * u.deg, source["dec_deg"] * u.deg)
        coordinates = SkyCoord(
            np.array([row["ra"] for row in items]) * u.deg,
            np.array([row["dec"] for row in items]) * u.deg,
        )
        distances = target.separation(coordinates).arcsec
        for row, separation in zip(items, distances):
            sid = str(row["source_id"])
            if sid != source["source_id"] and separation <= radius:
                rows[sid] = {
                    **row,
                    "source_id": sid,
                    "separation_arcsec": float(separation),
                }
    return {
        "status": "GAIA_SCENE_QUERIED"
        if queries and all(q["status"] == "COMPLETE" for q in queries)
        else "PARTIAL_QUERY_COVERAGE",
        "radius_arcsec": radius,
        "sources": [rows[sid] for sid in sorted(rows, key=int)],
        "dilution_constrained": False,
    }


def scene_queries_for_sources(cohort):
    queries = {}
    for index in range(0, len(cohort), 10):
        result = read(RUN / "queries" / f"scene-{index:03d}.json")
        for proposal in cohort[index : index + 10]:
            queries[proposal["source_id"]] = [result]
    return queries


def enrich_source(
    proposal, metadata, extra, queries, shared, shared_time, shared_complete
):
    sid = proposal["source_id"]
    source, path = metadata[sid], RUN / "sources" / f"{sid}.json"
    if shared_time is None:
        shared_scope = "All frozen survey follow-up alerts"
    elif shared_complete:
        shared_scope = "All alerts in the declared cohort"
    else:
        shared_scope = "Partial proposal queue"
    row = read(path)
    curve = raw_curve(source)
    row["gaia_scene"] = source_scene(source, queries)
    row["stellar"] = source_stellar(sid)
    complete_stellar(row["stellar"], sid)
    target = SkyCoord(source["ra_deg"] * u.deg, source["dec_deg"] * u.deg)
    matches = []
    radius = read(CONFIG)["neighbour_radius_arcsec"]
    for item in extra:
        if abs(item["dec"] - source["dec_deg"]) > radius / 3600:
            continue
        distance = target.separation(
            SkyCoord(item["ra"] * u.deg, item["dec"] * u.deg)
        ).arcsec
        if distance <= radius:
            exact = False
            matches.append(
                {
                    **item,
                    "separation_arcsec": float(distance),
                    "exact_identifier_match": exact,
                }
            )
    from catalogue_search import measure_row

    measure_row(row, source, read(CONFIG), ROOT)
    row["catalogues"]["eb_ctoi_matches"] = matches
    row["catalogues"]["status"] = "SCOPED_CATALOGUES_CHECKED"
    row["catalogues"]["coverage"] += (
        " Kepler EB 2016 and ExoFOP CTOI snapshots also checked "
        "in the configured scene."
    )
    from vetting_triage import measured_events

    for event in measured_events(row):
        centre, duration = event["centre_bkjd"], event["duration_days"]
        event["cross_view"] = {
            view: fixed_interval(curve, centre, duration, view)
            for view in ["psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova"]
        }
        event_start, event_end = centre - duration / 2, centre + duration / 2
        others = coincident_sources(
            shared,
            shared_time,
            event_start,
            event_end,
            sid,
            source["origin_tpf_kic"],
        )
        event["shared_time"] = {
            "other_alerted_sources": len(others),
            "same_parent_sources": sum(others.values()),
            "source_ids": sorted(others),
            "complete": shared_complete,
            "scope": f"{shared_scope}; coincidence is not contamination proof",
        }
    save(path, row, compact=True)


def init_evidence_worker(metadata, extra, queries, shared, times, complete):
    global EVIDENCE_CONTEXT
    if times is None:
        shared, times, complete = shared_alerts()
    EVIDENCE_CONTEXT = metadata, extra, queries, shared, times, complete


def measure_evidence_source(proposal):
    started_at, before = started(), counter_snapshot()
    metadata, extra, queries, shared, times, complete = EVIDENCE_CONTEXT
    enrich_source(
        proposal, metadata, extra, queries[proposal["source_id"]], shared, times, complete
    )
    return {
        "source_id": proposal["source_id"],
        "timing": elapsed(started_at),
        "counters": counter_deltas(before),
    }


def source_stellar(sid):
    gaia = query_evidence(sid, "gaia")
    return gaia["matches"][0] if len(gaia["matches"]) == 1 else {}


def needs_tic_record(stellar):
    required = ("radius_flame", "mass_flame", "teff_gspphot")
    return not all(
        stellar.get(key) is not None and stellar[key] > 0 for key in required
    )


def prefetch_stellar(cohort):
    """Resolve TIC records for the sources without Gaia FLAME parameters."""
    directory = RUN / "stellar"
    directory.mkdir(exist_ok=True)
    pending = [
        proposal["source_id"]
        for proposal in cohort
        if needs_tic_record(source_stellar(proposal["source_id"]))
    ]
    # Batches follow the cohort order so that a rerun issues identical named queries.
    batches = [
        (index, pending[index : index + 50]) for index in range(0, len(pending), 50)
    ]
    with ThreadPoolExecutor(max_workers=execution_workers(read(CONFIG))["network_workers"]) as pool:
        for records in pool.map(lambda batch: tic_stellar_records(*batch), batches):
            for sid, record in records.items():
                save(directory / f"{sid}.json", record)


def run():
    check_run(ROOT / read(CONFIG)["queue"])
    metadata, cohort = read(RUN / "metadata.json"), promoted_cohort()
    exact_queries(cohort, metadata)
    scenes(cohort, metadata)
    extra = extra_catalogues()
    queries = scene_queries_for_sources(cohort)
    shared, times, shared_complete = shared_alerts()
    try:
        prefetch_stellar(cohort)
        context = metadata, extra, queries, shared if times is not None else None, times, shared_complete
        with ProcessPoolExecutor(
            max_workers=execution_workers(read(CONFIG))["workers"],
            initializer=init_evidence_worker,
            initargs=context,
        ) as pool:
            for i, result in enumerate(pool.map(measure_evidence_source, cohort), 1):
                log_worker(result)
                if i % 20 == 0:
                    print(f"Evidence {i}/{len(cohort)}", flush=True)
    finally:
        if times is None:
            shared.close()


def reference_query(rows):
    oids = sorted({str(r["oid"]) for r in rows})
    if not oids:
        return {"status": "COMPLETE", "rows": []}
    query = (
        "select h.oidref,r.bibcode,r.title,r.doi from has_ref as h "
        "join ref as r on h.oidbibref=r.oidbib where h.oidref in ("
        + ",".join(oids)
        + ")"
    )
    return tap(
        "bibliography", "https://simbad.cds.unistra.fr/simbad/sim-tap/sync", query
    )


def source_aliases(sid, metadata):
    aliases = {f"Gaia DR3 {sid}"}
    if metadata[sid]["kic"]:
        aliases.add(f"KIC {metadata[sid]['kic']}")
    return aliases


def dr2_identities(index, sids):
    """The DR2 identity of each source with exactly one DR3 neighbourhood row."""
    forward = tap(
        f"stellar-dr3-{index:04d}",
        read(CONFIG)["gaia_tap_endpoint"],
        "select dr3_source_id,dr2_source_id,angular_distance,magnitude_difference "
        "from gaiadr3.dr2_neighbourhood where dr3_source_id in ("
        + ",".join(sids)
        + ")",
    )
    resolved = {}
    for sid in sids:
        rows = [row for row in forward["rows"] if str(row["dr3_source_id"]) == sid]
        if forward["status"] == "COMPLETE" and len(rows) == 1:
            resolved[sid] = int(rows[0]["dr2_source_id"])
    return resolved


def reciprocal_identities(index, resolved):
    """The DR2 identities whose only DR3 counterpart is the source itself."""
    if not resolved:
        return {}
    reverse = tap(
        f"stellar-dr2-{index:04d}",
        read(CONFIG)["gaia_tap_endpoint"],
        "select dr3_source_id,dr2_source_id from gaiadr3.dr2_neighbourhood "
        "where dr2_source_id in ("
        + ",".join(str(dr2) for dr2 in sorted(set(resolved.values())))
        + ")",
    )
    reciprocal = {}
    for sid, dr2 in resolved.items():
        identities = {
            str(row["dr3_source_id"])
            for row in reverse["rows"]
            if str(row["dr2_source_id"]) == str(dr2)
        }
        if reverse["status"] == "COMPLETE" and identities == {sid}:
            reciprocal[sid] = dr2
    return reciprocal


def tic_stellar_records(index, sids):
    """TIC records for one batch of sources with reciprocal Gaia DR3/DR2 identities."""
    records = {sid: {"status": "GAIA_RELEASE_IDENTITY_UNRESOLVED"} for sid in sids}
    reciprocal = reciprocal_identities(index, dr2_identities(index, sids))
    if not reciprocal:
        return records
    columns = (
        "TIC",
        "GAIA",
        "KIC",
        "Rad",
        "s_Rad",
        "e_Rad",
        "E_Rad",
        "Mass",
        "s_Mass",
        "e_Mass",
        "E_Mass",
        "Teff",
        "s_Teff",
        "e_Teff",
        "E_Teff",
        "Disp",
        "RadFl",
        "Flag",
        "r_Teff",
    )
    query = "select " + ",".join(f'"{c}"' for c in columns)
    query += ' from "IV/39/tic82" where "GAIA" in ('
    query += ",".join(str(dr2) for dr2 in sorted(set(reciprocal.values()))) + ")"
    response = tap(
        f"stellar-tic-{index:04d}",
        "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
        query,
    )
    for sid, dr2 in reciprocal.items():
        records[sid] = select_tic_stellar(response, dr2)
    return records


def select_tic_stellar(response, dr2):
    if response["status"] != "COMPLETE":
        return {"status": "TIC_QUERY_INCOMPLETE"}
    records = [
        r
        for r in response["rows"]
        if str(r.get("GAIA")) == str(dr2)
        and str(r.get("Disp") or "").strip() not in {"ARTIFACT", "DUPLICATE"}
    ]
    if len(records) != 1:
        return {"status": "TIC_IDENTITY_UNRESOLVED", "records": records}
    return {
        "status": "TIC_STELLAR_RECORD_IDENTIFIED",
        "record": records[0],
        "gaia_dr2_id": str(dr2),
        "catalogue": "TIC v8.2 (VizieR IV/39)",
        "identity": (
            "Unique reciprocal Gaia DR3/DR2 crossmatch and one non-duplicate, "
            "non-artifact TIC record"
        ),
    }


def complete_stellar(stellar, sid):
    if needs_tic_record(stellar):
        stellar["tic_parameters"] = read(RUN / "stellar" / f"{sid}.json")


def save_literature(sid, aliases, rows, references, complete):
    matches = [row for row in rows if row["id"] in aliases]
    objects = {row["oid"] for row in matches}
    save(
        RUN / "literature" / f"{sid}.json",
        {
            "status": (
                "SCOPED_IDENTIFIERS_AND_REFERENCES_QUERIED"
                if complete and references["status"] == "COMPLETE"
                else "PARTIAL_QUERY_COVERAGE"
            ),
            "aliases": sorted(aliases),
            "matches": matches,
            "references": [
                row for row in references["rows"] if row["oidref"] in objects
            ],
            "scope": (
                "Exact SIMBAD Gaia DR3 and KIC aliases plus indexed bibliography. "
                "Unindexed articles and identifier-free reports remain outside "
                "the search; no novelty claim."
            ),
        },
    )


def bibliography():
    check_run(ROOT / read(CONFIG)["queue"])
    queries = [read(path) for path in (RUN / "queries").glob("simbad-*.json")]
    complete = bool(queries) and all(query["status"] == "COMPLETE" for query in queries)
    rows = [row for query in queries for row in query["rows"]]
    references = reference_query(rows)
    metadata = read(RUN / "metadata.json")
    for source in promoted_cohort():
        sid = source["source_id"]
        save_literature(sid, source_aliases(sid, metadata), rows, references, complete)


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def ephemeris_values(record):
    if "koi_period" in record:
        return (
            record.get("koi_period"),
            record.get("koi_time0bk"),
            record.get("koi_period_err1"),
            record.get("koi_time0bk_err1"),
        )
    if "pl_orbper" in record:
        epoch = record.get("pl_tranmid")
        return (
            record.get("pl_orbper"),
            None if epoch is None else epoch - 2454833,
            record.get("pl_orbpererr1"),
            record.get("pl_tranmiderr1"),
        )
    if "BJD0" in record:
        epoch = number(record["BJD0"])
        return (
            number(record["Per"]),
            None if epoch is None else epoch - 54833,
            number(record["e_Per"]),
            number(record["e_BJD0"]),
        )
    if record.get("Period (days)") and record.get("Transit Epoch (BJD)"):
        return (
            float(record["Period (days)"]),
            float(record["Transit Epoch (BJD)"]) - 2454833,
            None,
            None,
        )
    return None


def catalogue_source(match):
    if match.get("name"):
        return str(match["name"])
    record = match["record"]
    for key in (
        "kepoi_name",
        "kepler_name",
        "pl_name",
        "toi",
        "ctoi_alias",
        "KIC",
        "TIC",
        "tid",
        "kepid",
    ):
        if record.get(key) not in (None, ""):
            return f"{key}:{record[key]}"
    return None


def catalogue_signal_class(match):
    catalogue = match["catalogue"].lower()
    record = match["record"]
    if "kepler-eb" in catalogue:
        return "ECLIPSING_BINARY"
    disposition = str(
        record.get("koi_pdisposition")
        or record.get("koi_disposition")
        or record.get("tfopwg_disp")
        or ""
    ).upper()
    if disposition in {"FALSE POSITIVE", "FP", "FA"}:
        return "FALSE_POSITIVE"
    if record.get("pl_name") or disposition in {
        "CONFIRMED",
        "CANDIDATE",
        "PC",
        "CP",
        "KP",
        "APC",
    }:
        return "PLANET_OR_CANDIDATE"
    return "UNCLASSIFIED"


def ephemeris_match(event, match):
    values = ephemeris_values(match["record"])
    if values is None:
        return None
    period, epoch, error, epoch_error = values
    if period is None or epoch is None or period <= 0:
        return None
    cycle = round((event["centre_bkjd"] - epoch) / period)
    nearest = epoch + cycle * period
    offset = event["centre_bkjd"] - nearest
    uncertainty = (
        None
        if error is None or epoch_error is None
        else float(np.hypot(cycle * error, epoch_error))
    )
    return {
        "catalogue": match["catalogue"],
        "source": catalogue_source(match),
        "signal_class": catalogue_signal_class(match),
        "exact_identifier_match": match.get("exact_identifier_match") is True,
        "separation_arcsec": match["separation_arcsec"],
        "known_period_days": period,
        "nearest_epoch_bkjd": nearest,
        "event_offset_days": offset,
        "propagated_epoch_error_days": uncertainty,
        "overlaps_original_interval_nominally": abs(offset)
        <= event["duration_days"] / 2,
        "interpretation": (
            "Timing/sky association and recorded catalogue class; "
            "no new physical mechanism inferred"
        ),
    }


def ephemeris_diagnostics(event, catalogues):
    matches = catalogues["matches"] + catalogues.get("eb_ctoi_matches", [])
    records = [ephemeris_match(event, match) for match in matches]
    return [record for record in records if record is not None]


def contamination(row, raw, config):
    from vetting_pixels import aggregate_signals
    from vetting_triage import signal_measured

    aggregate_signals(row)
    for signal in row["signals"]:
        if not signal_measured(signal):
            signal.setdefault(
                "contamination",
                {
                    "status": "SKIPPED_DISMISSED",
                    "reason": signal["triage"]["reason"],
                },
            )
            continue
        events = [e for e in row["events"] if e["event_id"] in signal["event_ids"]]
        signal["contamination"] = {
            "status": "EVIDENCE_ASSEMBLED",
            "catalogue_status": row.get("catalogues", {}).get("status"),
            "catalogue_timing": [
                {
                    "event_id": e["event_id"],
                    "matches": ephemeris_diagnostics(e, row["catalogues"]),
                }
                for e in events
            ],
            "neighbour_flux": [
                r
                for r in row.get("instrument", {})
                .get("comparisons", {})
                .get("records", [])
                if r["event_id"] in signal["event_ids"]
            ],
            "neighbour_comparisons": {
                key: value
                for key, value in row.get("instrument", {})
                .get("comparisons", {})
                .items()
                if key != "records"
            },
            "source_localization": signal.get(
                "localization", {"status": "UNAVAILABLE"}
            ),
            "scene": row.get("gaia_scene", {"status": "UNAVAILABLE"}),
            "mechanism_status": "NOT_ESTABLISHED",
            "scope": (
                "Catalogue timing, concurrent neighbour deficits and source "
                "offsets are compared. A transfer mechanism is not inferred from "
                "coincidence alone. Distant PRF-wing, ghost and readout "
                "contamination beyond the recorded scene remains untested."
            ),
        }
    return row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["collect", "contamination"], nargs="?", default="collect"
    )
    args = parser.parse_args()
    if args.stage == "contamination":
        from vetting import run_program

        run_program("contamination")
    else:
        run()
        bibliography()
