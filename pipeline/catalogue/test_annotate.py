"""Frozen example and end-to-end checks for derived catalogue annotations."""

import copy
import csv
import json
import shutil
from pathlib import Path

import pytest

from pipeline.catalogue.annotate import (
    REQUIRED_CATALOGUES,
    annotate,
    digest,
    period_flags,
    period_pair,
    save,
)


def example():
    # Rounded periods supplied by Adam for KIC 3975580; no Gaia identity inferred.
    return [{"source_id": "111", "signal_id": str(index), "label": "BOI",
             "period_days": period}
            for index, period in enumerate((4.405, 1.468, 0.629))]


def test_supplied_kic_example_flags_all_relations_without_changing_proposals():
    rows = example()
    before = copy.deepcopy(rows)
    pairs = period_flags(rows)
    assert {(p["numerator"], p["denominator"]) for p in pairs} == {(3, 1), (7, 1), (7, 3)}
    assert len(pairs) == 3
    assert rows == before


def test_different_stars_unknown_period_and_outside_tolerance_are_not_linked():
    left, right = example()[:2]
    assert period_pair(left, dict(right, source_id="different")) is None
    assert period_pair(left, dict(right, period_days=None)) is None
    assert period_pair(left, dict(right, period_days=1.46)) is None
    with pytest.raises(ValueError, match="restricted to BOIs"):
        period_flags([left, dict(right, label="HELD")])


def fixture_run(root):
    root.mkdir()
    matcher = root / "source/pipeline/vetting/catalogue_search.py"
    matcher.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "vetting/catalogue_search.py", matcher)
    frozen = root / "catalogues"
    frozen.mkdir()
    catalogues = sorted(REQUIRED_CATALOGUES | {"Bienias_Szabo_2025"})
    signals = [{"catalogue": name, "gaia_dr3": "222" if name == "NASA_confirmed_planets_PS" else None,
                "ra": 290.0 if name == "NASA_confirmed_planets_PS" else 1.0, "dec": 40.0,
                "name": "Known test planet", "planet_name": "Known test planet",
                "kic": "999", "period": 10., "epoch": 100., "duration_hours": 2.4,
                "classification": "CONFIRMED", "time_reference": "BKJD",
                "evidence_url": "https://example.test/frozen-catalogue"}
               for name in catalogues]
    save(frozen / "signals.json", signals)
    save(frozen / "manifest.json", {
        "schema": 1,
        "prepared_utc": "2026-10-12T21:54:47+00:00",
        "required_catalogues": catalogues,
        "counts": {"normalized_signal_rows": len(signals)},
        "position_qualification": "Frozen test catalogue positions.",
        "signals": {"path": "signals.json", "sha256": digest(frozen / "signals.json")},
    })
    spec = {"manifest": str(frozen / "manifest.json"),
            "manifest_sha256": digest(frozen / "manifest.json")}
    work = root / "work/batches/0001"
    sources = work / "vetting/sources"
    sources.mkdir(parents=True)
    save(work / "config.json", {"catalogue_search": spec})
    metadata = {source: {"kic": 3975580, "origin_tpf_kic": 3975580,
                         "ra_deg": 290.0, "dec_deg": 40.0}
                for source in ("111", "222")}
    save(work / "vetting/metadata.json", metadata)
    rows = example() + [{"source_id": "111", "signal_id": "held", "label": "HELD",
                         "period_days": 4.405 / 5},
                        {"source_id": "222", "signal_id": "dismissed",
                         "label": "DISMISSED", "period_days": 10.0}]
    for row in rows:
        row.update(discovery_view="sap_qdiag", epoch_bkjd=100., duration_days=0.1,
                   evidence=str(sources / f"{row['source_id']}.json"))
    for source in metadata:
        signals = []
        for row in [r for r in rows if r["source_id"] == source]:
            match = []
            if row["signal_id"] in {"0", "held", "dismissed"}:
                match = [{"catalogue": "NASA_cumulative_KOI", "name": "K00001.01",
                          "classification": "FALSE POSITIVE" if row["label"] == "HELD" else "CONFIRMED",
                          "host_relation": "NEARBY_CATALOGUED_SOURCE" if row["label"] == "BOI" else "SAME_GAIA_ID"}]
            signals.append({"signal_id": row["signal_id"], "period_days": row["period_days"],
                "events": [{"centre_bkjd": t} for t in (100., 110., 120.)],
                "timing_hypotheses": [{"hypothesis_id": row["signal_id"],
                    "timing_basis": "ORIGINAL_PROPOSAL", "period_days": row["period_days"],
                    "epoch_bkjd": 100., "duration_days": 0.1}],
                "catalogue_search": {
                "status": "COMPLETE", "manifest_sha256": spec["manifest_sha256"],
                "rule": {"id": "complete-catalogue-common-recurrence-v1"},
                "matches": match, "incomplete_pairs": [], "hypotheses_compared": 1,
            }})
        save(sources / f"{source}.json", {"source_id": source, "signals": signals})
    ledger = work / "vetting/signal-ledger.json"
    save(ledger, rows)
    result = work / "vetting/results.json"
    save(result, {"processing_status": "EVIDENCE_STAGES_COMPLETE", "signal_ledger": str(ledger)})
    save(root / "catalogue.json", {"status": "COMPLETE", "sources": 2,
         "proposals_supplied": 5, "batches": [{"result": str(result),
         "result_sha256": digest(result), "proposals_supplied": 5}]})
    save(root / "progress.json", {"state": "COMPLETE"})
    return rows


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def test_complete_run_keeps_all_outcomes_and_limits_period_checks_to_bois(tmp_path):
    run, out = tmp_path / "run", tmp_path / "derived"
    original = fixture_run(run)
    before = {str(p): digest(p) for p in run.rglob("*") if p.is_file()}
    report = annotate(run, out, workers=1)
    assert report["bois"]["possible_harmonic_or_alias_pairs"] == 3
    assert report["other_signals"]["possible_harmonic_or_alias_pairs"] == 0
    assert report["all_proposals"]["confirmed_planet_proposals_by_disposition"] == {"DISMISSED": 1}
    assert report["bois"]["host_compatible_confirmed_planet_proposals"] == 0
    bois = read_csv(out / "bois/signals.csv")
    other = read_csv(out / "other-signals/signals.csv")
    assert {row["signal_id"] for row in bois + other} == {r["signal_id"] for r in original}
    assert all(row["possible_harmonic_or_alias"] == "True" for row in bois)
    assert all(row["period_flag_scope"] == "NOT_RUN_AS_REQUESTED" for row in other)
    matches = read_csv(out / "other-signals/catalogue-matches.csv")
    assert {r["classification"] for r in matches} == {"CONFIRMED", "FALSE POSITIVE"}
    assert before == {str(p): digest(p) for p in run.rglob("*") if p.is_file()}
    assert len([p for p in report["inputs"] if "/sources/" in p]) == 2
    with pytest.raises(FileExistsError):
        annotate(run, out, workers=1)


def test_unfinished_vetting_and_changed_inputs_are_rejected(tmp_path):
    run, out = tmp_path / "run", tmp_path / "derived"
    fixture_run(run)
    save(run / "progress.json", {"state": "RUNNING"})
    with pytest.raises(ValueError, match="entire vetting pipeline"):
        annotate(run, out, workers=1)
    assert not out.exists()
    save(run / "progress.json", {"state": "COMPLETE"})
    (run / "work/batches/0001/vetting/results.json").write_text("{}")
    with pytest.raises(ValueError, match="Input hash differs"):
        annotate(run, out, workers=1)
    assert not out.exists()


def test_incomplete_catalogue_evidence_is_explicit_and_not_counted_as_recovery(tmp_path):
    run, out = tmp_path / "run", tmp_path / "derived"
    fixture_run(run)
    path = run / "work/batches/0001/vetting/sources/222.json"
    source = json.loads(path.read_text())
    source["signals"][0]["catalogue_search"].update(
        status="EPHEMERIS_COVERAGE_INCOMPLETE", matches=[],
        incomplete_pairs=[{"availability": "NO_COMPARABLE_EPOCH_OR_CENTRES"}],
    )
    source["signals"][0]["events"] = []
    save(path, source)
    report = annotate(run, out, workers=1)
    assert report["all_proposals"]["catalogue_statuses"] == {
        "COMPLETE": 4, "EPHEMERIS_COVERAGE_INCOMPLETE": 1,
    }
    assert report["all_proposals"]["host_compatible_confirmed_planet_proposals"] == 0


def test_missing_early_failure_report_runs_the_frozen_matcher(tmp_path):
    run, out = tmp_path / "run", tmp_path / "derived"
    fixture_run(run)
    path = run / "work/batches/0001/vetting/sources/222.json"
    source = json.loads(path.read_text())
    del source["signals"][0]["catalogue_search"]
    save(path, source)
    report = annotate(run, out, workers=1)
    assert report["all_proposals"]["host_compatible_confirmed_planet_proposals"] == 1
    assert report["all_proposals"]["comparison_origins"] == {
        "REUSED_COMPLETE_COMPARISON": 4, "COMPUTED_FROM_FROZEN_INPUTS": 1,
    }
    matches = read_csv(out / "other-signals/catalogue-matches.csv")
    assert any(row["catalogue"] == "NASA_confirmed_planets_PS"
               and row["host_relation"] == "SAME_GAIA_ID"
               and row["label"] == "DISMISSED" for row in matches)
