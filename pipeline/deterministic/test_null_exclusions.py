"""Verified-only accounting keeps unsupported signals in the null denominator."""

import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.deterministic import cloud_null as pool
from pipeline.deterministic import null_exclusions as producer
from pipeline.deterministic import run, signal_contracts as signals


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")
    return producer.reference(path)


def read(path):
    return json.loads(Path(path).read_bytes())


@pytest.fixture
def case(tmp_path, monkeypatch):
    ids = ["300", "100", "200"]
    metadata = {
        sid: {
            "source_id": sid,
            "origin_tpf_kic": int(sid),
            "shard": 0,
            "raw_fits_sha256": hashlib.sha256(sid.encode()).hexdigest(),
            "ra_deg": 90.0 if sid == "100" else 280.0,
            "dec_deg": 40.0,
        }
        for sid in ids
    }
    rows = [
        {
            "source_id": sid,
            "shard": "0",
            "origin_parent": sid,
            "raw_fits_sha256": metadata[sid]["raw_fits_sha256"],
            "metadata_dependency_component_id": hashlib.sha256(
                (sid + "component").encode()
            ).hexdigest(),
            "linked_existing_scopes": "",
            "linked_full_witness_component_ids": "",
            **pool.EXPOSURE,
        }
        for sid in ids
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=pool.FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    membership = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    monkeypatch.setattr(producer, "QUOTA", 3)
    monkeypatch.setattr(producer, "MEMBERSHIP_SHA", membership)
    monkeypatch.setattr(
        producer, "ORDER_SHA", hashlib.sha256(pool.canonical(ids)).hexdigest()
    )
    base = {
        "schema": 3,
        "protocol": "hash-bound-lc2-base-v1",
        "role": "development",
        "measurement_scope": "OPENED_NULL_BASE",
        "null_flux_constructed": False,
        "selection_seed": 20261002,
        "transform_seed": 20261002,
        "declaration": {"sha256": producer.DECLARATION_SHA},
        "membership": {"sha256": membership},
        "ordered_source_ids": ids,
        "sources": {
            row["source_id"]: {
                "metadata": metadata[row["source_id"]],
                "lc2_folder": "discSSD/data/kbonus/lc2/shard-0000",
                "original_pool_row": row,
            }
            for row in rows
        },
    }
    base_path = tmp_path / "base.json"
    base_ref = write(base_path, base)
    directory = tmp_path / "catalogues"
    specification = {"files": {}}
    original = read(producer.SNAPSHOTS)
    for name in original["files"]:
        path = directory / name
        row = (
            {
                "ra": 280.0,
                "dec": 40.0,
                "koi_period": 2.0,
                "koi_time0bk": 10.0,
                "koi_duration": 2.4,
            }
            if name.startswith("cumulative")
            else {}
        )
        if name.endswith(".json"):
            write(path, [row] if row else [])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            values = (
                {"_RA": "280", "_DE": "40", "Per": "2", "BJD0": "54843"}
                if name.startswith("kepler-eb")
                else {
                    "RA": "280",
                    "Dec": "40",
                    "Period (days)": "2",
                    "Transit Epoch (BJD)": "2454843",
                    "Duration (hrs)": "",
                }
            )
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(values))
                writer.writeheader()
                writer.writerow(values)
        specification["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": pool.digest(path),
        }
    spec_path = tmp_path / "snapshot-spec.json"
    write(spec_path, specification)
    monkeypatch.setattr(producer, "SNAPSHOTS", spec_path)
    prepared = []
    for sid in ids:
        records = np.zeros(200, dtype=run.RECORD)
        records["time"] = np.linspace(9.5, 14.5, 200)
        records["flux"], records["error"] = 0, 0.001
        records["quarter"] = np.repeat([1, 2, 3], [70, 70, 60])
        records["cadence"] = np.arange(200)
        path = tmp_path / "inputs" / f"{sid}.psf_flat.fsc"
        path.parent.mkdir(exist_ok=True)
        run.write_fsc(path, sid, "psf_flat", records)
        prepared.append(
            {
                "source_id": sid,
                "parent_tpf": sid,
                "raw_fits_sha256": metadata[sid]["raw_fits_sha256"],
                "inputs": {
                    "psf_flat": {"fsc_path": str(path), "fsc_sha256": pool.digest(path)}
                },
            }
        )
    prep = {
        "status": "PREPARED",
        "input_manifest": base_ref,
        "evaluation_scope": None,
        "requested_sources": 3,
        "preparation_counts": {"attempted_sources": 3},
        "selection": {"source_ids": ids},
        "sources": prepared,
        "cohort_sha256": producer.ORDER_SHA,
        "detector_contract": run.detector_contract(),
    }
    prep_path = tmp_path / "preparation.json"
    prep_ref = write(prep_path, prep)

    def forbidden(*args, **kwargs):
        pytest.fail("Exclusion producer attempted original or evaluation array access")

    monkeypatch.setattr(producer.lc2, "load_source", forbidden)
    return SimpleNamespace(
        root=tmp_path,
        ids=ids,
        base=base,
        base_path=base_path,
        base_ref=base_ref,
        metadata=metadata,
        prep=prep,
        prep_path=prep_path,
        prep_ref=prep_ref,
        directory=directory,
        specification=specification,
        spec_path=spec_path,
        out=tmp_path / "output",
    )


def produce(case, **kwargs):
    return producer.produce(
        case.base_path,
        case.base_ref["sha256"],
        case.directory,
        case.out,
        preparation=case.prep_path,
        preparation_sha=case.prep_ref["sha256"],
        **kwargs,
    )


def manifest(case):
    return {
        "role": "development",
        "null_construction": {"base_manifest": case.base_ref},
        "null_exclusions": producer.reference(case.out / "exclusions.json"),
    }


def alert(centre, quarter=1):
    return [
        {
            "event_id": "null-unknown",
            "quarter": quarter,
            "start_bkjd": centre - 0.01,
            "end_bkjd": centre + 0.01,
        }
    ]


def test_all_bases_get_six_lookups_and_unknown_nonproposals_are_retained(case):
    receipt = produce(case)
    index = read(case.out / "exclusions.json")
    assert list(index["sources"]) == case.ids
    assert receipt["sources"] == 3 and receipt["source_lookups"] == 18
    assert receipt["original_source_detection_or_vetting_run"] is False
    assert receipt["complete_real_signal_inventory_claim"] is False
    for sid in case.ids:
        known = read(index["sources"][sid]["catalogue_ephemerides"]["path"])
        real = read(index["sources"][sid]["real_signals"]["path"])
        assert len(known["lookups"]) == 6 and real["references"] == []
        assert known["coverage_scope"] == signals.EXCLUSION_COVERAGE
    assert signals.exclusion_references(manifest(case), "100") == []
    refs = signals.exclusion_references(manifest(case), "300")
    assert len(refs) == 1 and refs[0]["origin"] == "CATALOGUE_EPHEMERIS"
    assert signals.null_exclusion(alert(10), refs)["status"] == "EXCLUDED"
    assert signals.null_exclusion(alert(10, 3), refs)["status"] == "ELIGIBLE"
    result = signals.null_proposal_outcome(
        {
            "exclusion": signals.null_exclusion(alert(11.25), refs),
            "gate_path": [
                {"stage": name, "decision": "PASS"} for name in signals.FLUX_GATES
            ],
        }
    )
    assert result == {"vetted": True, "false_candidate": True, "excluded": False}
    assert receipt["retained_reasons"]["MISSING_OR_INVALID_PHYSICAL_DURATION"] == 4


def test_missing_preparation_keeps_all_catalogue_cases_and_full_source_accounting(case):
    receipt = producer.produce(
        case.base_path, case.base_ref["sha256"], case.directory, case.out
    )
    assert receipt["sources"] == 3 and receipt["counts"]["verified_references"] == 0
    assert receipt["retained_reasons"]["PREPARATION_UNSUPPLIED"] == 2
    assert signals.exclusion_references(manifest(case), "300") == []


@pytest.mark.parametrize(
    "change", ["evaluation", "hash", "order", "parent", "membership", "seed"]
)
def test_original_development_bindings_refuse_before_support_access(
    case, change, monkeypatch
):
    def forbidden(*args):
        pytest.fail("Invalid scope reached prepared FSC support")

    monkeypatch.setattr(producer, "prepared_support", forbidden)
    base, expected = case.base, case.base_ref["sha256"]
    if change == "evaluation":
        base["role"] = "evaluation"
    elif change == "order":
        base["ordered_source_ids"].reverse()
    elif change == "parent":
        base["sources"]["300"]["metadata"]["origin_tpf_kic"] = "999"
    elif change == "membership":
        base["sources"]["300"]["original_pool_row"]["raw_fits_sha256"] = "f" * 64
    elif change == "seed":
        base["transform_seed"] += 1
    if change != "hash":
        expected = write(case.base_path, base)["sha256"]
    else:
        expected = "0" * 64
    with pytest.raises(ValueError):
        producer.produce(case.base_path, expected, case.directory, case.out)
    assert not case.out.exists()


@pytest.mark.parametrize("change", ["base", "order", "fsc", "parent", "nonproposer"])
def test_pinned_preparation_refuses_stale_scope_and_input_bytes(case, change):
    if change == "base":
        case.prep["input_manifest"]["sha256"] = "0" * 64
    elif change == "order":
        case.prep["selection"]["source_ids"].reverse()
    elif change == "parent":
        case.prep["sources"][0]["parent_tpf"] = "999"
    elif change == "nonproposer":
        case.prep["sources"][1]["raw_fits_sha256"] = "0" * 64
    else:
        Path(case.prep["sources"][0]["inputs"]["psf_flat"]["fsc_path"]).write_bytes(
            b"stale"
        )
    case.prep_ref = write(case.prep_path, case.prep)
    with pytest.raises(ValueError):
        produce(case)


def test_ineligible_source_keeps_catalogue_matches_and_unknowns_in_denominator(case):
    case.prep["status"] = "PREPARATION_FAILED"
    case.prep["sources"].pop()
    case.prep["selection"]["source_ids"] = case.ids[:-1]
    case.prep["cohort_sha256"] = hashlib.sha256(
        pool.canonical(case.ids[:-1])
    ).hexdigest()
    case.prep_ref = write(case.prep_path, case.prep)
    receipt = produce(case)
    assert receipt["sources"] == 3
    assert receipt["preparation_statuses"]["INELIGIBLE_DISCOVERY_PREPARATION"] == 1
    assert signals.exclusion_references(manifest(case), "200") == []


@pytest.mark.parametrize(
    "change",
    [
        "legacy_scope",
        "missing_lookup",
        "unverified_internal",
        "event_clock",
        "stale_snapshot",
    ],
)
def test_consumers_refuse_unverified_or_stale_exclusions(case, change):
    produce(case)
    index_path = case.out / "exclusions.json"
    index = read(index_path)
    if change == "legacy_scope":
        del index["exclusion_rule"]
        write(index_path, index)
    elif change == "stale_snapshot":
        (case.directory / "cumulative-rows.json").write_text("[]\n")
    else:
        key = (
            "real_signals"
            if change == "unverified_internal"
            else "catalogue_ephemerides"
        )
        path = Path(index["sources"]["300"][key]["path"])
        data = read(path)
        if change == "missing_lookup":
            data["lookups"].pop("toi-rows.json")
        elif change == "event_clock":
            data["references"][0]["events"][0]["centre_bkjd"] += 1
        else:
            data["references"] = [
                {
                    "origin": "REAL_SIGNAL",
                    "reference_id": "unknown-proposal",
                    "duration_hours": 1,
                    "events": [],
                }
            ]
        index["sources"]["300"][key] = write(path, data)
        write(index_path, index)
    with pytest.raises(ValueError):
        signals.exclusion_references(manifest(case), "300")


def test_missing_fsc_is_retained_without_manufacturing_event_support(case):
    Path(case.prep["sources"][0]["inputs"]["psf_flat"]["fsc_path"]).unlink()
    receipt = produce(case)
    assert receipt["preparation_statuses"]["ORIGINAL_INPUT_UNAVAILABLE"] == 1
    assert signals.exclusion_references(manifest(case), "300") == []


def test_prepared_cadences_without_an_observed_catalogue_event_stay_retained(case):
    for source in case.prep["sources"]:
        path = Path(source["inputs"]["psf_flat"]["fsc_path"])
        records = run.read_fsc(path, source["source_id"], "psf_flat").copy()
        records["time"] = np.linspace(9.5, 9.55, len(records))
        path = path.with_name(f"{source['source_id']}.support-gap.fsc")
        run.write_fsc(path, source["source_id"], "psf_flat", records)
        source["inputs"]["psf_flat"]["fsc_path"] = str(path)
        source["inputs"]["psf_flat"]["fsc_sha256"] = pool.digest(path)
    case.prep_ref = write(case.prep_path, case.prep)
    receipt = produce(case)
    assert receipt["sources"] == 3
    assert receipt["retained_reasons"]["NO_PREPARED_QUARTER_SUPPORT"] == 2
    assert signals.exclusion_references(manifest(case), "300") == []


@pytest.mark.parametrize(
    "name,row,epoch,duration",
    [
        (
            "cumulative-rows.json",
            {"koi_period": 2, "koi_time0bk": 10, "koi_duration": 2.4},
            10,
            2.4,
        ),
        (
            "ps-rows.json",
            {"pl_orbper": 2, "pl_tranmid": 2454843, "pl_trandur": 2.4},
            10,
            2.4,
        ),
        (
            "toi-rows.json",
            {"pl_orbper": 2, "pl_tranmid": 2454843, "pl_trandurh": 2.4},
            10,
            2.4,
        ),
        (
            "exofop-ctoi-test.txt",
            {
                "Period (days)": "2",
                "Transit Epoch (BJD)": "2454843",
                "Duration (hrs)": "2.4",
            },
            10,
            2.4,
        ),
    ],
)
def test_catalogue_clocks_and_explicit_duration_units(name, row, epoch, duration):
    clock, reason = producer.ephemeris(name, row)
    assert (
        reason is None
        and clock["epoch_bkjd"] == epoch
        and clock["duration_hours"] == duration
    )


def test_eb_epoch_without_duration_is_retained():
    clock, reason = producer.ephemeris(
        "kepler-eb-2016.csv", {"Per": "2", "BJD0": "54843"}
    )
    assert clock is None and reason == "MISSING_OR_INVALID_PHYSICAL_DURATION"


def internal_index(case, status="VERIFIED_ASTROPHYSICAL_SIGNAL"):
    signal = {
        "signal_id": "verified-1",
        "event_ids": ["fit-event"],
        "events": [{"unit_id": "physical-unit", "quarter": 2, "centre_bkjd": 11.5}],
    }
    evidence = write(
        case.root / "source-evidence.json",
        {
            "source_id": "300",
            "raw_fits_sha256": case.metadata["300"]["raw_fits_sha256"],
            "signals": [signal],
            "events": [
                {
                    "event_id": "fit-event",
                    "fit": {
                        "status": "FIT_COMPLETE",
                        "centre_bkjd": 11.5,
                        "duration_days": 0.05,
                    },
                }
            ],
            "disposition": {
                "signals": [
                    {
                        "signal_id": "verified-1",
                        "signal_shape": {"records": [{"event_id": "fit-event"}]},
                    }
                ]
            },
        },
    )
    adjudication = write(
        case.root / "adjudication.json",
        {
            "source_id": "300",
            "raw_fits_sha256": case.metadata["300"]["raw_fits_sha256"],
            "signal_id": "verified-1",
            "status": status,
            "reviewer": "Fixture reviewer",
            "rationale": "Existing verified physical signal fixture",
            "evidence": evidence,
            "physical_duration_hours": 1.2,
        },
    )
    return write(
        case.root / "internal-index.json",
        {
            "role": "development",
            "base_manifest": case.base_ref,
            "entries": [{"source_id": "300", "adjudication": adjudication}],
        },
    )


@pytest.mark.parametrize(
    "status,expected",
    [("VERIFIED_ASTROPHYSICAL_SIGNAL", "EXCLUDED"), ("UNRESOLVED", "ELIGIBLE")],
)
def test_internal_exclusion_requires_existing_verified_adjudication(
    case, status, expected
):
    binding = internal_index(case, status)
    produce(
        case, adjudications=Path(binding["path"]), adjudications_sha=binding["sha256"]
    )
    refs = signals.exclusion_references(manifest(case), "300")
    assert signals.null_exclusion(alert(11.5, 2), refs)["status"] == expected
    assert sum(row["origin"] == "REAL_SIGNAL" for row in refs) == (
        expected == "EXCLUDED"
    )


def test_missing_internal_adjudication_is_explicit_and_retained(case):
    binding = internal_index(case)
    (case.root / "adjudication.json").unlink()
    receipt = produce(
        case, adjudications=Path(binding["path"]), adjudications_sha=binding["sha256"]
    )
    assert receipt["retained_reasons"]["ADJUDICATION_UNAVAILABLE"] == 1
    refs = signals.exclusion_references(manifest(case), "300")
    assert signals.null_exclusion(alert(11.5, 2), refs)["status"] == "ELIGIBLE"


def test_consumer_rechecks_the_pinned_adjudication_status(case):
    binding = internal_index(case)
    produce(
        case, adjudications=Path(binding["path"]), adjudications_sha=binding["sha256"]
    )
    path = case.root / "adjudication.json"
    review = read(path)
    review["status"] = "UNRESOLVED"
    write(path, review)
    with pytest.raises(ValueError, match="changed"):
        signals.exclusion_references(manifest(case), "300")


def test_verified_adjudication_without_physical_clock_is_explicit_and_retained(case):
    binding = internal_index(case)
    review_path = case.root / "adjudication.json"
    review = read(review_path)
    evidence_path = case.root / "source-evidence.json"
    evidence = read(evidence_path)
    evidence["events"][0]["fit"]["centre_bkjd"] = None
    review["evidence"] = write(evidence_path, evidence)
    index = read(binding["path"])
    index["entries"][0]["adjudication"] = write(review_path, review)
    binding = write(Path(binding["path"]), index)
    receipt = produce(
        case, adjudications=Path(binding["path"]), adjudications_sha=binding["sha256"]
    )
    assert (
        receipt["retained_reasons"]["VERIFIED_SIGNAL_PHYSICAL_TIMING_UNRESOLVED"] == 1
    )
    refs = signals.exclusion_references(manifest(case), "300")
    assert signals.null_exclusion(alert(11.5, 2), refs)["status"] == "ELIGIBLE"


def test_unlocated_snapshot_rows_and_source_coordinates_stay_explicit(case):
    name = "q1_q17_dr25_koi-rows.json"
    path = case.directory / name
    write(
        path,
        [
            {
                "ra": None,
                "dec": 40,
                "koi_period": 2,
                "koi_time0bk": 10,
                "koi_duration": 2.4,
            }
        ],
    )
    case.specification["files"][name] = {
        "bytes": path.stat().st_size,
        "sha256": pool.digest(path),
    }
    write(case.spec_path, case.specification)
    case.base["sources"]["100"]["metadata"]["ra_deg"] = None
    case.base_ref = write(case.base_path, case.base)
    case.prep["input_manifest"] = case.base_ref
    case.prep_ref = write(case.prep_path, case.prep)
    produce(case)
    index = read(case.out / "exclusions.json")
    assert index["catalogue_snapshots"][name]["unlocated_rows"] == [0]
    known = read(index["sources"]["100"]["catalogue_ephemerides"]["path"])
    assert all(
        row["status"] == "SOURCE_COORDINATES_UNRESOLVED"
        for row in known["lookups"].values()
    )
    assert signals.exclusion_references(manifest(case), "100") == []


@pytest.mark.parametrize("change", ["base", "role", "population", "legacy_scope"])
def test_construction_refuses_exclusion_scope_before_arrays_or_output(
    case, change, monkeypatch
):
    produce(case)
    index_path = case.out / "exclusions.json"
    index = read(index_path)
    if change == "base":
        index["base_manifest"]["sha256"] = "0" * 64
    elif change == "role":
        index["role"] = "evaluation"
    elif change == "population":
        del index["sources"]["100"]
    else:
        del index["exclusion_rule"]
    write(index_path, index)
    output = case.root / "null-output"
    monkeypatch.setattr(
        signals, "input_manifest", lambda *_: (case.base_path, case.base)
    )

    def forbidden(*args):
        pytest.fail("Invalid exclusions reached null array construction")

    monkeypatch.setattr(signals, "load_source", forbidden)
    args = SimpleNamespace(
        input_manifest=case.base_path,
        exclusions=index_path,
        out=output,
        variant="inverted",
        seed=20261002,
    )
    with pytest.raises(ValueError):
        signals.construct(args)
    assert not output.exists()


def test_consumer_requires_positive_actual_discovery_quarter_support(case):
    produce(case)
    index_path = case.out / "exclusions.json"
    index = read(index_path)
    path = Path(index["sources"]["300"]["catalogue_ephemerides"]["path"])
    catalogue = read(path)
    item = catalogue["references"][0]
    item["events"][0]["support"]["1"]["psf_flat"] = 0
    support_path = Path(item["verification"]["prepared_support"]["path"])
    support = read(support_path)
    support["supported_references"][item["reference_id"]]["events"] = item["events"]
    item["verification"]["prepared_support"] = write(support_path, support)
    index["sources"]["300"]["catalogue_ephemerides"] = write(path, catalogue)
    write(index_path, index)
    with pytest.raises(ValueError, match="prepared-quarter support"):
        signals.exclusion_references(manifest(case), "300")
