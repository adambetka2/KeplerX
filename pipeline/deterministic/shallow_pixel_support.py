"""Resolve shallow-control pixel inputs and select a parent-disjoint cohort."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from lc2 import source_catalogue

PLAN = Path(__file__).with_name("shallow-pixel-support-plan.json")
INJECTION_PLAN = Path(__file__).with_name("shallow-injection-plan.json")
INJECTION_CODE = Path(__file__).with_name("shallow_injections.py")
ENRICHMENT_PLAN = Path(__file__).with_name("shallow-cell-enrichment-plan.json")
ENRICHMENT_CODE = Path(__file__).with_name("shallow_cell_enrichment.py")
CELLS = [f"d{d}m{m}p{p}" for d in range(3) for m in range(4) for p in range(3)]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def truth_digest(truth):
    payload = json.dumps(truth, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def candidate_id(source_id, cell):
    return f"{source_id}-{cell}"


def rank(source_id, seed):
    return hashlib.sha256(f"{seed}:supported:{source_id}".encode()).digest()


def validate_screen(path, summary, exclusions):
    identity = summary["identity"]
    if summary["status"] != "COMPLETE_INPUT_ONLY_SHARD_SCREEN":
        raise ValueError(f"Incomplete screen: {path}")
    if identity["plan_sha256"] != digest(INJECTION_PLAN) or identity[
        "exclusions_sha256"
    ] != digest(exclusions["path"]):
        raise ValueError(f"Screen implementation or exclusion changed: {path}")
    for filename, expected in identity["implementation_sha256"].items():
        candidate = (
            ROOT / "pipeline/photometry/lc2.py"
            if filename == "lc2.py"
            else INJECTION_CODE.with_name(filename)
        )
        if digest(candidate) != expected:
            raise ValueError(f"Screen source implementation changed: {candidate}")
    shard = int(path.parent.name.split("-")[1])
    if identity["shard"] != shard:
        raise ValueError(f"Screen shard identity changed: {path}")
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    manifest = ROOT / volume / f"data/kbonus/lc2/shard-{shard:04d}/manifest.json"
    if digest(manifest) != identity["shard_manifest_sha256"]:
        raise ValueError(f"LC2 shard manifest changed: {manifest}")
    return shard


def checked_screen_rows(path, summary):
    for name, sha in sorted(summary["chunks"].items()):
        chunk = path.parent / name
        if digest(chunk) != sha:
            raise ValueError(f"Screen chunk changed: {chunk}")
        record = read(chunk)
        if record["identity"] != summary["identity"] or record["source_ids"] != [
            item["source_id"] for item in record["results"]
        ]:
            raise ValueError(f"Screen chunk identities changed: {chunk}")
        yield from record["results"]


def check_candidate(item, source_ids, exclusions, shard):
    truth = item["truth"]
    source_id = item["source_id"]
    if (
        source_id in source_ids
        or source_id in exclusions["source_ids_set"]
        or truth["source_id"] != source_id
        or truth["source_parent_tpf"] in exclusions["parents_set"]
        or truth["base_raw_fits_sha256"] in exclusions["raw_fits_sha256_set"]
        or truth["source_shard"] != shard
    ):
        raise ValueError(f"Screen candidate overlaps frozen inputs: {source_id}")
    source_ids.add(source_id)
    return truth


def screen_candidates(folder, exclusions, expected_shards=None):
    summaries = sorted(Path(folder).glob("shard-*/summary.json"))
    found = {int(path.parent.name.split("-")[1]) for path in summaries}
    if expected_shards is not None and found != set(expected_shards):
        raise ValueError(
            f"Screen shard coverage changed: found {len(found)}, expected {len(expected_shards)}"
        )
    if not summaries:
        raise ValueError("No completed input-only shard screens")
    exclusions = dict(
        exclusions,
        source_ids_set=set(exclusions["source_ids"]),
        parents_set=set(exclusions["parents"]),
        raw_fits_sha256_set=set(exclusions["raw_fits_sha256"]),
    )
    candidates, source_ids, summaries_sha = [], set(), {}
    for path in summaries:
        summary = read(path)
        shard = validate_screen(path, summary, exclusions)
        statuses, cells = Counter(), Counter()
        for item in checked_screen_rows(path, summary):
            statuses[item["status"]] += 1
            if item["status"] == "ELIGIBLE":
                truth = check_candidate(item, source_ids, exclusions, shard)
                cells[truth["cell"]] += 1
                candidates.append(truth)
        expected_cells = {
            cell: count
            for cell, count in summary["candidate_cell_counts"].items()
            if count
        }
        if (
            sum(statuses.values()) != summary["metadata_eligible_sources"]
            or dict(cells) != expected_cells
        ):
            raise ValueError(f"Screen counts changed: {path}")
        summaries_sha[str(shard)] = digest(path)
    return candidates, summaries_sha


def validate_enrichment_summary(path, base_root, plan):
    summary = read(path)
    shard = int(path.parent.name.split("-")[1])
    base_path = base_root / path.parent.name / "summary.json"
    identity = summary["identity"]
    if (
        summary["status"] != "COMPLETE_INPUT_ONLY_CELL_ENRICHMENT"
        or identity["shard"] != shard
        or identity["plan_sha256"] != digest(ENRICHMENT_PLAN)
        or identity["implementation_sha256"] != digest(ENRICHMENT_CODE)
        or identity["base_summary_sha256"] != digest(base_path)
        or plan["base_generator_sha256"] != digest(INJECTION_CODE)
        or plan["base_plan_sha256"] != digest(INJECTION_PLAN)
    ):
        raise ValueError(f"Enrichment identity changed: {path}")
    return summary, read(base_path), shard


def checked_extra_rows(path, summary, base_path, base_summary, plan):
    for name, expected in sorted(summary["chunks"].items()):
        extra_path = path.parent / name
        base_chunk_path = base_path.parent / name
        if (
            digest(extra_path) != expected
            or digest(base_chunk_path) != base_summary["chunks"][name]
        ):
            raise ValueError(f"Enrichment or base chunk changed: {name}")
        extra, base_chunk = read(extra_path), read(base_chunk_path)
        base_eligible = {
            row["source_id"]: row["truth"]
            for row in base_chunk["results"]
            if row["status"] == "ELIGIBLE"
        }
        identity = extra["identity"]
        if (
            extra["status"] != "COMPLETE_INPUT_ONLY_ENRICHMENT_CHUNK"
            or identity["plan_sha256"] != digest(ENRICHMENT_PLAN)
            or identity["implementation_sha256"] != digest(ENRICHMENT_CODE)
            or identity["base_chunk_sha256"] != base_summary["chunks"][name]
            or identity["base_summary_sha256"] != digest(base_path)
            or identity["shard"] != base_summary["identity"]["shard"]
            or extra["source_ids"] != list(base_eligible)
            or extra["source_ids"] != [row["source_id"] for row in extra["results"]]
        ):
            raise ValueError(f"Enrichment chunk identity changed: {extra_path}")
        for row in extra["results"]:
            baseline = base_eligible[row["source_id"]]
            cells = set()
            for truth in row["extra_truths"]:
                if (
                    truth["source_id"] != row["source_id"]
                    or truth["cell"] not in plan["target_cells"]
                    or truth["cell"] == baseline["cell"]
                    or truth["cell"] in cells
                    or truth["source_shard"] != baseline["source_shard"]
                    or truth["source_parent_tpf"] != baseline["source_parent_tpf"]
                    or truth["base_raw_fits_sha256"] != baseline["base_raw_fits_sha256"]
                ):
                    raise ValueError(
                        f"Enriched physical truth identity changed: {extra_path}"
                    )
                cells.add(truth["cell"])
                yield truth


def enrichment_candidates(folder, base_root, expected_shards=None):
    folder, base_root = Path(folder), Path(base_root)
    plan = read(ENRICHMENT_PLAN)
    if (
        plan["status"] != "INPUT_ONLY_DEVELOPMENT"
        or plan["screen_selection_sha256"] != digest(base_root / "selection.json")
        or plan["screen_audit_sha256"] != digest(base_root / "audit.json")
    ):
        raise ValueError("Frozen enrichment plan changed")
    summaries = sorted(folder.glob("shard-*/summary.json"))
    found = {int(path.parent.name.split("-")[1]) for path in summaries}
    if expected_shards is not None and found != set(expected_shards):
        raise ValueError(
            "Enrichment shard coverage changed: "
            f"found {len(found)}, expected {len(expected_shards)}"
        )
    truths, hashes = [], {}
    for path in summaries:
        summary, base_summary, shard = validate_enrichment_summary(
            path, base_root, plan
        )
        if set(summary["chunks"]) != set(base_summary["chunks"]):
            raise ValueError(f"Enrichment chunk coverage changed: {path}")
        base_path = base_root / path.parent.name / "summary.json"
        rows = list(checked_extra_rows(path, summary, base_path, base_summary, plan))
        counts = Counter(row["cell"] for row in rows)
        if (
            len(rows) != summary["extra_candidate_options"]
            or {cell: counts[cell] for cell in plan["target_cells"]}
            != summary["extra_candidate_cell_counts"]
            or summary["eligible_sources"]
            != base_summary["source_status"].get("ELIGIBLE", 0)
        ):
            raise ValueError(f"Enrichment counts changed: {path}")
        truths.extend(rows)
        hashes[str(shard)] = digest(path)
    return truths, hashes


def audited_extra_truths(folder, sparse, conditional):
    bright = []
    for shard, sha in conditional["shard_sha256"].items():
        shard_path = folder / f"rare-conditional-input-v1/shard-{int(shard):04d}.json"
        if digest(shard_path) != sha:
            raise ValueError(f"Conditional shallow shard changed: {shard}")
        for result in read(shard_path)["results"]:
            for item in result["cell_results"].values():
                if item["status"] == "ELIGIBLE":
                    bright.append(item["truth"])
    for shard, sha in sparse["shard_sha256"].items():
        shard_path = folder / f"sparse-cells-next-192-v1/shard-{int(shard):04d}.json"
        if digest(shard_path) != sha:
            raise ValueError(f"Sparse shallow shard changed: {shard}")
    return sparse["candidate_trials"] + bright


def extra_candidates(path, exclusions, base_summaries, enrichment_summaries):
    path = Path(path)
    record = read(path)
    folder = path.parent
    expected = {
        "exclusions_sha256": digest(exclusions["path"]),
        "generator_sha256": digest(INJECTION_CODE),
        "injection_plan_sha256": digest(INJECTION_PLAN),
        "support_plan_sha256": digest(PLAN),
        "collector_code_sha256": digest(folder / "combine_input_options.py"),
        "base_selection_sha256": digest(folder / "hash-screen-v1/selection.json"),
        "sparse_audit_sha256": digest(folder / "sparse-cells-audit.json"),
        "conditional_audit_sha256": digest(folder / "rare-conditional-audit.json"),
    }
    if (
        record["status"] != "COMPLETE_INPUT_ONLY_COMBINED_OPTIONS"
        or any(record[key] != value for key, value in expected.items())
        or record["base_summary_sha256"] != base_summaries
        or record["enrichment_summary_sha256"] != enrichment_summaries
    ):
        raise ValueError("Combined shallow option provenance changed")
    extras = record["extra_truths"]
    sparse = read(folder / "sparse-cells-audit.json")
    conditional = read(folder / "rare-conditional-audit.json")
    if (
        sparse["status"] != "COMPLETE_INPUT_ONLY_SPARSE_CELL_AUDIT"
        or conditional["status"] != "COMPLETE_INPUT_ONLY_CONDITIONAL_AUDIT"
        or len(extras) != record["extra_options"]
        or extras != audited_extra_truths(folder, sparse, conditional)
    ):
        raise ValueError("Combined shallow truth ledger changed")
    excluded_parents = set(exclusions["parents"])
    excluded_sources = set(exclusions["source_ids"])
    excluded_raw = set(exclusions["raw_fits_sha256"])
    metadata_cache = {}
    for truth in extras:
        if (
            truth["cell"] not in CELLS
            or truth["source_parent_tpf"] in excluded_parents
            or truth["source_id"] in excluded_sources
            or truth["base_raw_fits_sha256"] in excluded_raw
        ):
            raise ValueError(
                f"Extra shallow truth overlaps frozen inputs: {truth['source_id']}"
            )
        verified_lc2_metadata(truth, metadata_cache)
    return extras, record


def verified_lc2_metadata(truth, cache):
    shard = truth["source_shard"]
    if shard not in cache:
        volume = "discSSD" if shard % 2 == 0 else "discPCIe"
        folder = ROOT / volume / f"data/kbonus/lc2/shard-{shard:04d}"
        manifest = read(folder / "manifest.json")
        for key in ("source", "cadence"):
            item = manifest["files"][key]
            if digest(folder / item["path"]) != item["sha256"]:
                raise ValueError(f"LC2 {key} parquet changed: {folder}")
        cache[shard] = {
            row["source_id"]: row for row in source_catalogue(ROOT, [shard])
        }
    metadata = cache[shard][truth["source_id"]]
    if (
        metadata["raw_fits_sha256"] != truth["base_raw_fits_sha256"]
        or str(metadata["origin_tpf_kic"]) != truth["source_parent_tpf"]
    ):
        raise ValueError(f"LC2 raw FITS identity changed: {truth['source_id']}")
    return metadata


def inspect_tpf(option, astrometry, quarter, excluded_parents, prf_cache):
    from astropy.io import fits
    from deterministic import pixel_injections as pixels
    from vetting_pixels import kepler_prf_path

    path = Path(option["path"])
    if not path.is_file() or digest(path) != option["sha256"]:
        raise ValueError(f"Supplied TPF bytes changed: {path}")
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        parent = str(int(hdus[0].header["KEPLERID"]))
        actual_quarter = int(hdus[0].header["QUARTER"])
        channel = int(hdus[0].header["CHANNEL"])
    if actual_quarter != quarter:
        raise ValueError(f"Supplied TPF quarter changed: {path}")
    if parent in excluded_parents:
        return None, "RESOLVED_PARENT_EXCLUDED"
    distance = pixels.stamp_distance(path, astrometry)
    if distance > 0:
        return None, "TARGET_CENTER_OFF_STAMP"
    if not kepler_prf_path(channel, prf_cache).is_file():
        return None, "PENDING_PRF_CALIBRATION"
    with fits.open(path, memmap=False) as hdus:
        profile, calibration, pixel, sky, status, inside = pixels.pixel_profile(
            hdus, astrometry, cache=prf_cache
        )
    if not inside:
        raise ValueError(f"Resolved TPF/PRF identity changed: {path}")
    evidence = {
        "base_path": str(path),
        "base_sha256": option["sha256"],
        "base_parent_kic": int(parent),
        "prf_calibration": calibration,
        "target_pixel": pixel,
        "target_sky_deg": sky,
        "target_astrometry": status,
        "target_inside_stamp": inside,
        "prf_stamp_fraction": float(profile.sum()),
        "pixel_center_distance": distance,
    }
    return evidence, "SUPPORTED"


def assess_candidate(
    truth, astrometry, tpfs, excluded_parents, prf_cache, metadata_cache
):
    source_id = truth["source_id"]
    base = {
        "candidate_id": candidate_id(source_id, truth["cell"]),
        "source_id": source_id,
        "cell": truth["cell"],
        "truth_sha256": truth_digest(truth),
        "base_raw_fits_sha256": truth["base_raw_fits_sha256"],
        "observed_quarters": truth["observed_quarters"],
    }
    if astrometry is None:
        return dict(base, status="PENDING_ASTROMETRY")
    if astrometry["pmra"] is None or astrometry["pmdec"] is None:
        return dict(
            base,
            status="PENDING_ASTROMETRY",
            reason="GAIA_PROPER_MOTION_UNAVAILABLE",
        )
    if not tpfs:
        return dict(base, status="PENDING_TPF_INPUT")
    verified_lc2_metadata(truth, metadata_cache)
    options_by_quarter = tpfs.get("quarters", {})
    complete_quarters = set(map(str, tpfs.get("complete_quarters", [])))
    pixels, reasons = {}, {}
    pending_prf = False
    for quarter in truth["observed_quarters"]:
        options = options_by_quarter.get(str(quarter), [])
        if not options:
            reasons[str(quarter)] = (
                "NO_AVAILABLE_TPF"
                if str(quarter) in complete_quarters
                else "PENDING_TPF_SEARCH"
            )
            continue
        for option in sorted(options, key=lambda row: (row["path"], row["sha256"])):
            evidence, reason = inspect_tpf(
                option, astrometry, quarter, excluded_parents, prf_cache
            )
            if evidence is not None:
                pixels[str(quarter)] = evidence
                break
            reasons[str(quarter)] = reason
            pending_prf |= reason == "PENDING_PRF_CALIBRATION"
    if len(pixels) != len(truth["observed_quarters"]):
        terminal = (
            all(
                str(quarter) in complete_quarters
                for quarter in truth["observed_quarters"]
                if str(quarter) not in pixels
            )
            and not pending_prf
        )
        return dict(
            base,
            status="PIXEL_UNSUPPORTED" if terminal else "PENDING_PIXEL_SUPPORT",
            quarter_reasons=reasons,
        )
    return dict(
        base,
        status="PIXEL_SUPPORTED",
        pixels=pixels,
        resolved_parents=sorted(
            {str(item["base_parent_kic"]) for item in pixels.values()}, key=int
        ),
    )


def validate_tpf_map(mapping):
    for source_id, item in mapping["sources"].items():
        complete = set(map(str, item.get("complete_quarters", [])))
        if not complete:
            continue
        evidence = item.get("search_evidence", [])
        covered = {
            str(quarter) for record in evidence for quarter in record["quarters"]
        }
        if not complete <= covered:
            raise ValueError(f"Complete TPF search lacks provenance: {source_id}")
        for record in evidence:
            if digest(record["path"]) != record["sha256"]:
                raise ValueError(f"TPF search evidence changed: {record['path']}")


def candidate_pool(truths, plan, limit):
    truths.sort(key=lambda row: (rank(row["source_id"], plan["seed"]), row["cell"]))
    per_cell = Counter()
    pool = []
    for truth in truths:
        if truth["cell"] not in CELLS:
            raise ValueError(
                f"Screen candidate has an undeclared cell: {truth['cell']}"
            )
        if per_cell[truth["cell"]] < plan["candidate_pool_per_cell"]:
            pool.append(truth)
            per_cell[truth["cell"]] += 1
    return (pool[:limit] if limit else pool), {cell: per_cell[cell] for cell in CELLS}


def merged_screen_candidates(args, exclusions, selected):
    truths, summaries = screen_candidates(args.screens, exclusions, selected)
    enrichment_summaries = None
    if args.enrichment:
        if selected is None:
            raise ValueError("Enrichment requires a frozen base shard selection")
        if read(ENRICHMENT_PLAN)["exclusions_sha256"] != digest(args.exclusions):
            raise ValueError("Enrichment exclusion snapshot changed")
        extras, enrichment_summaries = enrichment_candidates(
            args.enrichment, args.screens, selected
        )
        truths.extend(extras)
    extra_record = None
    if args.extra_options:
        extras, extra_record = extra_candidates(
            args.extra_options, exclusions, summaries, enrichment_summaries
        )
        truths.extend(extras)
    keys = [candidate_id(row["source_id"], row["cell"]) for row in truths]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated source-cell option across shallow input screens")
    return truths, summaries, enrichment_summaries, extra_record


def resolve(args):
    exclusion_path = Path(args.exclusions)
    exclusions = read(exclusion_path)
    if exclusions["status"] != "FROZEN_INPUT_EXCLUSIONS" or exclusions[
        "plan_sha256"
    ] != digest(INJECTION_PLAN):
        raise ValueError("Frozen shallow exclusion identity changed")
    exclusions["path"] = str(exclusion_path)
    selection = read(args.selection) if args.selection else None
    if selection and selection["status"] != "INPUT_ONLY_SCREEN_SELECTION_FROZEN":
        raise ValueError("Screen shard selection is not frozen")
    selected = selection["selected_shards"] if selection else None
    truths, summaries, enrichment_summaries, extra_record = merged_screen_candidates(
        args, exclusions, selected
    )
    plan = read(PLAN)
    truths, per_cell = candidate_pool(truths, plan, args.limit)
    if extra_record and (
        args.limit != 0
        or len(truths) != extra_record["candidate_pool_options"]
        or len({row["source_id"] for row in truths})
        != extra_record["candidate_pool_distinct_sources"]
        or per_cell != extra_record["candidate_pool_cell_counts"]
        or sorted({row["source_id"] for row in truths}, key=int)
        != extra_record["candidate_pool_source_ids"]
    ):
        raise ValueError("Combined shallow candidate pool changed")
    gaia = read(args.gaia) if args.gaia else {"rows": []}
    astrometry = {str(row["source_id"]): row for row in gaia["rows"]}
    tpf_map = read(args.tpf_map) if args.tpf_map else {"schema": 1, "sources": {}}
    if gaia.get("rows") and gaia.get("status") != "COMPLETE":
        raise ValueError("Gaia astrometry input is incomplete")
    if (
        extra_record
        and args.gaia
        and (
            gaia.get("combined_options_sha256") != digest(args.extra_options)
            or set(astrometry) != set(extra_record["candidate_pool_source_ids"])
            or len(gaia["rows"]) != len(astrometry)
        )
    ):
        raise ValueError("Gaia astrometry does not cover the combined candidate pool")
    if tpf_map.get("schema") != 1:
        raise ValueError("Unsupported TPF input map")
    validate_tpf_map(tpf_map)
    identity = {
        "plan_sha256": digest(PLAN),
        "resolver_sha256": digest(__file__),
        "exclusions_sha256": digest(exclusion_path),
        "screen_summaries_sha256": summaries,
        "enrichment_summaries_sha256": enrichment_summaries,
        "extra_options_sha256": digest(args.extra_options)
        if args.extra_options
        else None,
        "selection_sha256": digest(args.selection) if args.selection else None,
        "gaia_sha256": digest(args.gaia) if args.gaia else None,
        "tpf_map_sha256": digest(args.tpf_map) if args.tpf_map else None,
        "limit": args.limit,
        "candidate_pool_per_cell": plan["candidate_pool_per_cell"],
        "candidate_ids_sha256": hashlib.sha256(
            "\n".join(
                candidate_id(row["source_id"], row["cell"]) for row in truths
            ).encode()
        ).hexdigest(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_folder = out / "records"
    records_folder.mkdir(exist_ok=True)
    metadata_cache = {}
    records = []
    for truth in truths:
        source_id = truth["source_id"]
        key = candidate_id(source_id, truth["cell"])
        path = records_folder / f"{key}.json"
        if path.exists():
            record = read(path)
            if (
                record.get("identity") != identity
                or record.get("candidate_id") != key
                or record.get("truth_sha256") != truth_digest(truth)
            ):
                raise ValueError(f"Pixel-support resume identity changed: {path}")
        else:
            record = assess_candidate(
                truth,
                astrometry.get(source_id),
                tpf_map["sources"].get(source_id),
                set(exclusions["parents"]),
                Path(args.prf_cache),
                metadata_cache,
            )
            record["identity"] = identity
            save(path, record)
        records.append(record)
    report = {
        "schema": 1,
        "status": "COMPLETE_INPUT_ONLY_PIXEL_SUPPORT_ASSESSMENT",
        "identity": identity,
        "sources": len(records),
        "distinct_sources": len({row["source_id"] for row in records}),
        "candidate_pool_cell_counts": dict(per_cell),
        "statuses": dict(Counter(row["status"] for row in records)),
        "cell_supported_counts": dict(
            Counter(
                row["cell"] for row in records if row["status"] == "PIXEL_SUPPORTED"
            )
        ),
        "records_sha256": {
            row["candidate_id"]: digest(records_folder / f"{row['candidate_id']}.json")
            for row in records
        },
        "claim_boundary": "Structural source and pixel input support only; no detector, vetter, localization, or sealed outcome was read.",
    }
    save(out / "support-report.json", report)
    return report


def selection_constraints(rows, cells, per_cell):
    constraints = defaultdict(list)
    for index, row in enumerate(rows):
        constraints[("cell", row["cell"])].append(index)
        constraints[("source", row["source_id"])].append(index)
        constraints[("raw", row["base_raw_fits_sha256"])].append(index)
        for parent in row["resolved_parents"]:
            constraints[("parent", str(parent))].append(index)
    for cell in cells:
        constraints[("cell", cell)]
    matrix = lil_matrix((len(constraints), len(rows)), dtype=np.float64)
    upper = np.ones(len(constraints))
    for row_index, (key, indexes) in enumerate(constraints.items()):
        matrix[row_index, indexes] = 1
        if key[0] == "cell":
            upper[row_index] = per_cell
    return LinearConstraint(matrix.tocsr(), np.zeros(len(upper)), upper)


def disjoint_selection(records, cells, per_cell, seed):
    rows = sorted(records, key=lambda row: (rank(row["source_id"], seed), row["cell"]))
    if not rows:
        return []
    if len({candidate_id(row["source_id"], row["cell"]) for row in rows}) != len(rows):
        raise ValueError("Repeated source-cell option in supported candidates")
    if any(row["cell"] not in cells for row in rows):
        raise ValueError("Supported candidate has an undeclared cell")
    linear = selection_constraints(rows, cells, per_cell)
    bounds = Bounds(np.zeros(len(rows)), np.ones(len(rows)))
    integers = np.ones(len(rows))
    maximum = milp(
        -np.ones(len(rows)),
        integrality=integers,
        bounds=bounds,
        constraints=linear,
        options={"mip_rel_gap": 0},
    )
    if maximum.status != 0:
        raise RuntimeError(
            f"Parent-disjoint maximum-cardinality optimization failed: {maximum.message}"
        )
    count = round(-maximum.fun)
    exact = LinearConstraint(np.ones((1, len(rows))), count, count)
    tie_break = np.array(
        [
            int.from_bytes(rank(row["source_id"], seed)[8:14], "big") / 2**48
            for row in rows
        ]
    )
    costs = np.arange(1, len(rows) + 1, dtype=float) + tie_break / (len(rows) + 1)
    ranked = milp(
        costs,
        integrality=integers,
        bounds=bounds,
        constraints=(linear, exact),
        options={"mip_rel_gap": 0},
    )
    if ranked.status != 0:
        raise RuntimeError(
            f"Deterministic parent-disjoint ranking failed: {ranked.message}"
        )
    chosen = [row for row, value in zip(rows, ranked.x, strict=True) if value > 0.5]
    if len(chosen) != count:
        raise ValueError("Parent-disjoint solution cardinality changed")
    return chosen


def verified_supported_records(support, folder, prf_cache):
    records = []
    for key, sha in support["records_sha256"].items():
        path = folder / "records" / f"{key}.json"
        if digest(path) != sha:
            raise ValueError(f"Pixel-support record changed: {path}")
        record = read(path)
        if record["identity"] != support["identity"] or record["candidate_id"] != key:
            raise ValueError(f"Pixel-support identity changed: {path}")
        if record["status"] != "PIXEL_SUPPORTED":
            continue
        if (
            set(record["pixels"]) != set(map(str, record["observed_quarters"]))
            or set(record["resolved_parents"])
            != {str(pixel["base_parent_kic"]) for pixel in record["pixels"].values()}
            or any(
                pixel["pixel_center_distance"] != 0
                for pixel in record["pixels"].values()
            )
            or not record["resolved_parents"]
        ):
            raise ValueError(f"Pixel support geometry changed: {path}")
        for pixel in record["pixels"].values():
            if digest(pixel["base_path"]) != pixel["base_sha256"]:
                raise ValueError(f"Resolved TPF changed: {pixel['base_path']}")
            calibration = prf_cache / pixel["prf_calibration"]["filename"]
            if digest(calibration) != pixel["prf_calibration"]["sha256"]:
                raise ValueError(f"Resolved PRF changed: {calibration}")
        records.append(record)
    return records


def partition_selected(chosen, cells, plan):
    by_cell = defaultdict(list)
    for row in chosen:
        by_cell[row["cell"]].append(row)
    development, evaluation = [], []
    for cell in cells:
        ranked = sorted(
            by_cell[cell], key=lambda row: rank(row["source_id"], plan["seed"])
        )
        development.extend(ranked[: plan["development_per_cell"]])
        evaluation.extend(ranked[plan["development_per_cell"] :])
    return development, evaluation


def partition_records(rows, hashes):
    return [
        {
            "source_id": row["source_id"],
            "cell": row["cell"],
            "truth_sha256": row["truth_sha256"],
            "resolved_parents": row["resolved_parents"],
            "support_record_sha256": hashes[row["candidate_id"]],
        }
        for row in rows
    ]


def select(args):
    folder = Path(args.support)
    support = read(folder / "support-report.json")
    if (
        support["status"] != "COMPLETE_INPUT_ONLY_PIXEL_SUPPORT_ASSESSMENT"
        or support["identity"]["plan_sha256"] != digest(PLAN)
        or support["identity"]["resolver_sha256"] != digest(__file__)
    ):
        raise ValueError("Pixel-support implementation identity changed")
    records = verified_supported_records(support, folder, Path(args.prf_cache))
    plan = read(PLAN)
    if (
        plan["development_per_cell"] + plan["evaluation_per_cell"]
        != plan["trials_per_cell"]
    ):
        raise ValueError("Shallow pixel partition counts disagree")
    cells = CELLS
    chosen = disjoint_selection(records, cells, plan["trials_per_cell"], plan["seed"])
    selected_parents = [parent for row in chosen for parent in row["resolved_parents"]]
    selected_raw = [row["base_raw_fits_sha256"] for row in chosen]
    selected_sources = [row["source_id"] for row in chosen]
    if (
        len(selected_parents) != len(set(selected_parents))
        or len(selected_raw) != len(set(selected_raw))
        or len(selected_sources) != len(set(selected_sources))
    ):
        raise ValueError("Global source, parent, or raw-input independence failed")
    counts = Counter(row["cell"] for row in chosen)
    complete = (
        len(chosen) == len(cells) * plan["trials_per_cell"]
        and all(counts[cell] == plan["trials_per_cell"] for cell in cells)
        and support["sources"]
        == len(records) + support["statuses"].get("PIXEL_UNSUPPORTED", 0)
        and support["identity"]["limit"] == 0
        and support["identity"]["selection_sha256"] is not None
        and support["identity"]["candidate_pool_per_cell"]
        == plan["candidate_pool_per_cell"]
    )
    report = {
        "schema": 1,
        "status": "BALANCED_INPUT_POOL_READY" if complete else "NO_BALANCED_INPUT_POOL",
        "support_report_sha256": digest(Path(args.support) / "support-report.json"),
        "plan_sha256": digest(PLAN),
        "selector_sha256": digest(__file__),
        "supported_candidates": len(records),
        "maximum_parent_disjoint_selected": len(chosen),
        "supported_cell_counts": {
            cell: sum(row["cell"] == cell for row in records) for cell in cells
        },
        "selected_cell_counts": {cell: counts[cell] for cell in cells},
        "claim_boundary": "Input-only feasibility. No detector, vetter, localization, or sealed outcomes were read.",
    }
    if complete:
        development, evaluation = partition_selected(chosen, cells, plan)
        report["partitions"] = {
            "development": partition_records(development, support["records_sha256"]),
            "evaluation": partition_records(evaluation, support["records_sha256"]),
        }
        report["status"] = "FROZEN_BALANCED_INPUT_IDENTITIES"
    save(args.out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    support = command.add_parser("resolve")
    support.add_argument("--screens", type=Path, required=True)
    support.add_argument("--selection", type=Path)
    support.add_argument("--enrichment", type=Path)
    support.add_argument("--extra-options", type=Path)
    support.add_argument("--exclusions", type=Path, required=True)
    support.add_argument("--gaia", type=Path)
    support.add_argument("--tpf-map", type=Path)
    support.add_argument("--prf-cache", type=Path, required=True)
    support.add_argument("--limit", type=int, default=0)
    support.add_argument("--out", type=Path, required=True)
    choice = command.add_parser("select")
    choice.add_argument("--support", type=Path, required=True)
    choice.add_argument("--prf-cache", type=Path, required=True)
    choice.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = resolve(args) if args.command == "resolve" else select(args)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "sources", "statuses")
                if key in report
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
