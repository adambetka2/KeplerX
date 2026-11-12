"""Run and review independent discovery views on one frozen source cohort."""

import argparse
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from deterministic import run as pipeline
from runtime import dump_path, worker_count

VIEWS = ("psf_flat", "psf_qdiag", "sap_qdiag")


def completed_view(run, view, source_ids_path):
    preparation = pipeline.read(run / "preparation.json")
    pipeline.validate_preparation_accounting(preparation)
    selection = preparation["selection"]
    if (
        selection["source_ids_file"]["sha256"] != pipeline.digest(source_ids_path)
        or preparation["detector_contract"]["discovery_view"] != view
    ):
        raise ValueError("View differs from the frozen comparison cohort")
    result = pipeline.read(run / "result.json")
    summary = pipeline.read(run / "vetting/results.json")
    timings = pipeline.read(run / "vetting/stage-timings.json")["stages"]
    survivors = pipeline.read(run / "vetting/surviving-signals.json")
    if (
        result["status"] != "EVIDENCE_STAGES_COMPLETE"
        or summary["processing_status"] != "EVIDENCE_STAGES_COMPLETE"
        or len(timings) != 16
        or any(stage["status"] != "COMPLETE" for stage in timings)
        or len(survivors) != summary["triage_surviving_signals"]
    ):
        raise ValueError("Independent view has incomplete evidence")
    ledger = pipeline.read(run / "vetting/signal-ledger.json")
    if len(ledger) != summary["signals"]:
        raise ValueError("Independent view signal ledger is incomplete")
    return {
        "signal_ledger": ledger,
        "discovery_view": view,
        "run": str(run),
        "result_sha256": pipeline.digest(run / "result.json"),
        "preparation_counts": preparation["preparation_counts"],
        "detection": result["detection"],
        "signals": summary["signals"],
        "active": summary["triage_surviving_signals"],
        "held": summary["triage_held_signals"],
        "dismissed": summary["triage_dismissed_signals"],
        "boi_catalogue_enabled": result["boi_catalogue_enabled"],
        "survivors": [{**row, "discovery_view": view} for row in survivors],
    }


def review_union(records):
    views = [record["discovery_view"] for record in records]
    if len(views) != len(set(views)):
        raise ValueError("Repeated discovery view in comparison")
    survivors = [row for record in records for row in record["survivors"]]
    identities = [
        (row["discovery_view"], row["source_id"], row["signal_id"]) for row in survivors
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated final signal identity in comparison")
    from pipeline.vetting.boi import catalogue

    ledger = [row for record in records for row in record.get("signal_ledger", [])]
    return {
        "catalogue": catalogue(ledger),
        "signal_ledger": ledger,
        "status": "INDEPENDENT_VIEW_REVIEW_COMPLETE",
        "boi_catalogue_enabled": any(r["boi_catalogue_enabled"] for r in records),
        "view_signal_counts": dict(Counter(row["discovery_view"] for row in survivors)),
        "sources_with_any_active_view": len({row["source_id"] for row in survivors}),
        "interpretation": "Every final ACTIVE view signal is retained independently. HELD or DISMISSED signals in another view do not replace it. Correlated view detections are not independent confidence or a count of unique astrophysical signals; all per-view source evidence and physical contradictions remain in the referenced runs.",
        "views": [{k: v for k, v in r.items() if k not in {"survivors", "signal_ledger"}} for r in records],
        "survivors": sorted(
            survivors,
            key=lambda row: (
                int(row["source_id"]),
                row["discovery_view"],
                row["signal_id"],
            ),
        ),
    }


def timed_stage(state, status_path, view, name, operation):
    state.update(stage=name, discovery_view=view)
    pipeline.save(status_path, state)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parent = time.process_time()
    value = operation()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    state["stages"].append(
        {
            "discovery_view": view,
            "stage": name,
            "wall_seconds": time.monotonic() - started,
            "parent_cpu_seconds": time.process_time() - parent,
            "child_cpu_seconds": after.ru_utime
            + after.ru_stime
            - before.ru_utime
            - before.ru_stime,
            "cumulative_child_peak_rss_kib": after.ru_maxrss,
        }
    )
    pipeline.save(status_path, state)
    return value


def run(args):
    ids = pipeline.read(args.source_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("A nonempty frozen source cohort is required")
    out = dump_path(args.run)
    out.mkdir(parents=True, exist_ok=False)
    status_path = out / "multiview.json"
    state = {
        "status": "RUNNING_INDEPENDENT_VIEWS",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_ids_sha256": pipeline.digest(args.source_ids),
        "requested_sources": len(ids),
        "views": list(args.views),
        "workers": args.workers,
        "stages": [],
        "completed_views": [],
        "timing_scope": "Sequential independent full runs: preparation, native compilation, both discovery searches, queue, all sixteen evidence stages, and reconciliation. Plotting excluded. Shared verified pixel cache and OS data cache retained; later views can reuse earlier retrievals.",
    }
    started = time.monotonic()
    pipeline.save(status_path, state)
    records = []
    try:
        for view in args.views:
            target = out / view
            options = argparse.Namespace(**vars(args))
            options.limit = len(ids)
            options.out = target
            options.discovery_view = view
            options.record_every_gate = False
            options.run = target
            timed_stage(
                state, status_path, view, "prepare", partial(pipeline.prepare, options)
            )
            options.binary = timed_stage(
                state,
                status_path,
                view,
                "build_binary",
                partial(pipeline.build_binary, target),
            )
            timed_stage(
                state, status_path, view, "scan", partial(pipeline.scan, options)
            )
            timed_stage(state, status_path, view, "vet", partial(pipeline.vet, options))
            records.append(completed_view(target, view, args.source_ids))
            state["completed_views"].append(view)
            pipeline.save(status_path, state)
        review = review_union(records)
        pipeline.save(out / "review.json", review)
        pipeline.save(out / "boi-catalogue.json", review["catalogue"])
        pipeline.save(out / "signal-ledger.json", review["signal_ledger"])
        state.update(
            status="COMPLETE_INDEPENDENT_VIEWS", wall_seconds=time.monotonic() - started
        )
    except BaseException as exc:
        state.update(
            status="FAILED_INDEPENDENT_VIEWS",
            wall_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        pipeline.save(status_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--shards", type=int, nargs="+", required=True)
    parser.add_argument("--views", choices=VIEWS, nargs="+", default=VIEWS)
    parser.add_argument("--workers", type=int, default=worker_count())
    parser.add_argument("--threads", type=int, default=worker_count())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument(
        "--pixel-cache-directory", type=Path, default=Path("dump/pixel-cache")
    )
    parser.add_argument("--pixel-minimum-free-bytes", type=int, default=20 * 1024**3)
    args = parser.parse_args()
    if len(args.views) != len(set(args.views)) or min(args.workers, args.threads) < 1:
        raise ValueError("Unique views and positive worker counts required")
    run(args)


if __name__ == "__main__":
    main()
