import csv
import importlib.util
import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.runtime import disposition_definition_sha256, disposition_execution

FALSE_CLASS_MODULE = Path(__file__).with_name("false_class_controls.py")
FALSE_CLASS_SPEC = importlib.util.spec_from_file_location(
    "deterministic_false_class_controls", FALSE_CLASS_MODULE
)
false_class_controls = importlib.util.module_from_spec(FALSE_CLASS_SPEC)
FALSE_CLASS_SPEC.loader.exec_module(false_class_controls)

MODULE = Path(__file__).with_name("controls.py")
SPEC = importlib.util.spec_from_file_location("deterministic_controls", MODULE)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)
PARTITION_MODULE = Path(__file__).with_name("control_partitions.py")
PARTITION_SPEC = importlib.util.spec_from_file_location(
    "deterministic_control_partitions", PARTITION_MODULE
)
control_partitions = importlib.util.module_from_spec(PARTITION_SPEC)
PARTITION_SPEC.loader.exec_module(control_partitions)
REPORT_MODULE = Path(__file__).with_name("control_report.py")
REPORT_SPEC = importlib.util.spec_from_file_location(
    "deterministic_control_report", REPORT_MODULE
)
control_report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(control_report)
CATALOGUE_MODULE = Path(__file__).with_name("catalogue_controls.py")
CATALOGUE_SPEC = importlib.util.spec_from_file_location(
    "deterministic_catalogue_controls", CATALOGUE_MODULE
)
catalogue_controls = importlib.util.module_from_spec(CATALOGUE_SPEC)
CATALOGUE_SPEC.loader.exec_module(catalogue_controls)
CATALOGUE_REPORT_MODULE = Path(__file__).with_name("catalogue_control_report.py")
CATALOGUE_REPORT_SPEC = importlib.util.spec_from_file_location(
    "deterministic_catalogue_control_report", CATALOGUE_REPORT_MODULE
)
catalogue_control_report = importlib.util.module_from_spec(CATALOGUE_REPORT_SPEC)
CATALOGUE_REPORT_SPEC.loader.exec_module(catalogue_control_report)
CATALOGUE_PARTITION_MODULE = Path(__file__).with_name("catalogue_control_partition.py")
CATALOGUE_PARTITION_SPEC = importlib.util.spec_from_file_location(
    "deterministic_catalogue_control_partition", CATALOGUE_PARTITION_MODULE
)
catalogue_control_partition = importlib.util.module_from_spec(CATALOGUE_PARTITION_SPEC)
CATALOGUE_PARTITION_SPEC.loader.exec_module(catalogue_control_partition)
EB_MODULE = Path(__file__).with_name("catalogue_eb_controls.py")
EB_SPEC = importlib.util.spec_from_file_location("catalogue_eb_controls", EB_MODULE)
catalogue_eb_controls = importlib.util.module_from_spec(EB_SPEC)
EB_SPEC.loader.exec_module(catalogue_eb_controls)
PLANET_PARTITION_MODULE = Path(__file__).with_name("planet_control_partitions.py")
PLANET_PARTITION_SPEC = importlib.util.spec_from_file_location(
    "deterministic_planet_control_partitions", PLANET_PARTITION_MODULE
)
planet_control_partitions = importlib.util.module_from_spec(PLANET_PARTITION_SPEC)
PLANET_PARTITION_SPEC.loader.exec_module(planet_control_partitions)
PLANET_REPORT_MODULE = Path(__file__).with_name("planet_control_report.py")
PLANET_REPORT_SPEC = importlib.util.spec_from_file_location(
    "deterministic_planet_control_report", PLANET_REPORT_MODULE
)
planet_control_report = importlib.util.module_from_spec(PLANET_REPORT_SPEC)
PLANET_REPORT_SPEC.loader.exec_module(planet_control_report)


def test_source_pools_are_disjoint_ordered_and_exclude_opened_sources():
    available = {
        str(value): {"source_id": str(value), "volume": "discSSD", "shard": 0}
        for value in range(1, 20)
    }
    cohorts = {
        "ordinary": {"target": 2, "pool": 4},
        "periodic": {"target": 2, "pool": 4},
    }
    rows = controls.select_source_pools(available, {"2", "7"}, cohorts, "seed")
    assert len(rows) == 8
    assert len({row["source_id"] for row in rows}) == 8
    assert not {"2", "7"} & {row["source_id"] for row in rows}
    assert [row["pool_rank"] for row in rows[:4]] == [1, 2, 3, 4]


def test_catalogue_pools_require_source_pure_dispositions():
    groups = {
        1: [{"koi_disposition": "CANDIDATE"}],
        2: [
            {"koi_disposition": "CANDIDATE"},
            {"koi_disposition": "FALSE POSITIVE"},
        ],
        3: [{"koi_disposition": "FALSE POSITIVE"}],
        4: [{"koi_disposition": "CANDIDATE"}],
    }
    cohorts = {
        "candidate": {"disposition": "CANDIDATE", "target": 1, "pool": 1},
        "false_positive": {
            "disposition": "FALSE POSITIVE",
            "target": 1,
            "pool": 1,
        },
    }
    rows = controls.select_catalogue_pools(groups, {4}, cohorts, "seed")
    assert {row["kic_id"] for row in rows} == {1, 3}


def test_parent_partitions_keep_opened_parents_out_of_evaluation():
    source_ids = [str(value) for value in range(1, 10)]
    parents = {
        "1": "a",
        "2": "b",
        "3": "a",
        "4": "c",
        "5": "b",
        "6": "d",
        "7": "d",
        "8": "e",
        "9": "f",
    }
    opened, development, evaluation, excluded = control_partitions.partition(
        source_ids, parents, 4
    )
    assert opened == ["1", "2", "3", "4"]
    assert development == ["1", "2", "4"]
    assert evaluation == ["6", "8", "9"]
    assert {(row["source_id"], row["reason"]) for row in excluded} == {
        ("3", "DEVELOPMENT_PARENT_DUPLICATE"),
        ("5", "DEVELOPMENT_PARENT_OVERLAP"),
        ("7", "EVALUATION_PARENT_DUPLICATE"),
    }
    assert {parents[source_id] for source_id in development}.isdisjoint(
        {parents[source_id] for source_id in evaluation}
    )


def test_parent_partitions_replenish_from_eligible_reserve():
    parents = {
        "1": "a",
        "2": "b",
        "3": "a",
        "4": "c",
        "5": "d",
        "6": "e",
        "7": "c",
        "8": "f",
        "9": "g",
    }
    opened, _, evaluation, excluded = control_partitions.partition(
        ["1", "2", "3", "4", "5", "6"],
        {key: parents[key] for key in map(str, range(1, 7))},
        3,
    )
    evaluation, selected, attempts = control_partitions.replenish_evaluation(
        evaluation,
        excluded,
        [
            {"source_id": "7", "pool_rank": 7},
            {"source_id": "8", "pool_rank": 8},
            {"source_id": "9", "pool_rank": 9},
        ],
        parents,
        {parents[source_id] for source_id in opened},
        4,
    )
    assert evaluation == ["4", "5", "6", "8"]
    assert selected == ["8"]
    assert attempts == 2
    assert excluded[-1]["reason"] == "RESERVE_EVALUATION_PARENT_DUPLICATE"


def test_ordinary_evaluation_replenishes_external_parent_exclusions():
    source_ids = list(map(str, range(1, 7)))
    parents = {str(value): f"parent{value}" for value in range(1, 9)}
    external = {"parent3", "parent7"}
    opened, development, evaluation, excluded = control_partitions.partition(
        source_ids, {key: parents[key] for key in source_ids}, 2, external
    )
    assert development == opened == ["1", "2"]
    assert evaluation == ["4", "5", "6"]
    evaluation, selected, attempts = control_partitions.replenish_evaluation(
        evaluation,
        excluded,
        [{"source_id": str(value), "pool_rank": value} for value in (7, 8)],
        parents,
        {parents[key] for key in opened},
        4,
        external,
    )
    assert evaluation == ["4", "5", "6", "8"]
    assert selected == ["8"]
    assert attempts == 2
    assert {parents[key] for key in evaluation}.isdisjoint(external)
    assert [row["reason"] for row in excluded] == [
        "EVALUATION_EXCLUDED_PARENT",
        "RESERVE_EXCLUDED_PARENT",
    ]


def test_ordinary_parent_exclusions_include_resolved_parents(tmp_path):
    path = tmp_path / "parents.csv"
    path.write_text("parent_tpf,resolved_parent_tpf\n1,2;3\n")
    parents, inputs = control_partitions.parent_exclusions([path])
    assert parents == {"1", "2", "3"}
    assert inputs == [{"path": str(path), "sha256": control_partitions.digest(path)}]
    path.write_text("source_id\n1\n")
    with pytest.raises(ValueError, match="lack parent_tpf"):
        control_partitions.parent_exclusions([path])


def test_ordinary_excluded_reserve_cannot_lower_trial_target():
    with pytest.raises(ValueError, match="cannot replenish"):
        control_partitions.replenish_evaluation(
            ["1"],
            [],
            [{"source_id": "2", "pool_rank": 2}],
            {"1": "a", "2": "b"},
            set(),
            2,
            {"b"},
        )


def test_ordinary_source_metadata_requires_its_indexed_content_hash(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "source_id": [1],
                "origin_tpf_kic": [2],
                "raw_fits_sha256": ["a" * 64],
            }
        ),
        path,
    )
    expected = control_partitions.digest(path)
    assert control_partitions.verified_source_metadata(path, expected, [1]) == [
        {"source_id": 1, "origin_tpf_kic": 2, "raw_fits_sha256": "a" * 64}
    ]
    pq.write_table(
        pa.table(
            {
                "source_id": [1],
                "origin_tpf_kic": [3],
                "raw_fits_sha256": ["b" * 64],
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="metadata is unavailable or changed"):
        control_partitions.verified_source_metadata(path, expected, [1])


@pytest.mark.parametrize("consumer", ["planet", "shallow"])
def test_injection_exclusions_use_verified_parent_source_contract(
    tmp_path, monkeypatch, consumer
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pipeline.deterministic import shallow_injections

    data = tmp_path / "discSSD/data"
    metadata = data / "kbonus/lc2/shard-0001/source.parquet"
    metadata.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "source_id": [1],
                "origin_tpf_kic": [10],
                "raw_fits_sha256": ["a" * 64],
            }
        ),
        metadata,
    )
    expected = control_partitions.digest(metadata)
    index = data / "index.sqlite"
    with sqlite3.connect(index) as database:
        database.execute(
            "create table sources(gaia_source_id integer, volume text, shard integer)"
        )
        database.execute("insert into sources values(1, 'discSSD', 1)")
        database.execute("create table files(volume, path, sha256, collection)")
        database.execute(
            "insert into files values('discSSD', ?, ?, 'kbonus')",
            (str(metadata.relative_to(data)), expected),
        )
    source_ids = tmp_path / "opened.csv"
    source_ids.write_text("source_id\n1\n")
    plan = {
        "excluded_parent_csv": [],
        "excluded_parent_jsonl": [],
        "opened_source_manifests": [{"path": str(source_ids), "column": "source_id"}],
    }
    monkeypatch.setattr(planet_control_partitions, "ROOT", tmp_path)

    def resolve():
        if consumer == "planet":
            parents, _, files = planet_control_partitions.excluded_parents(plan)
            assert parents == {"10"}
            assert files == [
                {
                    "volume": "discSSD",
                    "path": str(metadata.relative_to(data)),
                    "sha256": expected,
                }
            ]
        else:
            parents, inputs = set(), {}
            shallow_injections.resolve_source_parents(tmp_path, inputs, parents, {"1"})
            assert parents == {"10"}
            assert inputs[str(metadata)] == expected

    resolve()
    metadata.write_bytes(b"changed source metadata")
    with pytest.raises(ValueError, match="metadata is unavailable or changed"):
        resolve()


def test_ordinary_partitions_reject_shared_or_excluded_raw_fits(tmp_path):
    with pytest.raises(ValueError, match="share a raw FITS"):
        control_partitions.validate_raw_independence(
            ["1"], ["2"], {"1": "a", "2": "a"}, []
        )
    path = tmp_path / "parents.csv"
    path.write_text("source_id,parent_tpf,raw_fits_sha256\n3,3,b\n")
    with pytest.raises(ValueError, match="overlap excluded controls"):
        control_partitions.validate_raw_independence(
            ["1"], ["2"], {"1": "a", "2": "b"}, [path]
        )


def test_ordinary_partitions_preserve_exposed_development_inputs(tmp_path):
    path = tmp_path / "historical-development.csv"
    path.write_text("source_id,parent_tpf,raw_fits_sha256\n1,10,a\n")
    control_partitions.validate_raw_independence(
        ["1"], ["2"], {"1": "a", "2": "b"}, [path]
    )


def eb_input_fixture(tmp_path):
    records = [
        {"KIC": str(kic), "Per": "10.5", "BJD0": "55000.25"} for kic in range(1, 5)
    ]
    catalogue = tmp_path / "eb.csv"
    with catalogue.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    identity = tmp_path / "identities.json"
    identity.write_text(
        json.dumps(
            {
                "ordered_identity_records": [
                    {"kic_id": row["KIC"], "catalogue_records": [row]}
                    for row in records
                ]
            }
        )
    )
    remaining = tmp_path / "remaining.json"
    remaining.write_text(
        json.dumps({"remaining_catalogue_eb_kics": ["1", "2", "3", "4"]})
    )
    return identity, remaining, catalogue


def test_catalogue_eb_inputs_preserve_reserve_order_and_ephemerides(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(catalogue_eb_controls, "DEVELOPMENT", 1)
    monkeypatch.setattr(catalogue_eb_controls, "EVALUATION", 1)
    identity, remaining, catalogue = eb_input_fixture(tmp_path)
    data, signals = catalogue_eb_controls.control_pool(identity, remaining, catalogue)
    assert [row["kic_id"] for row in data["rows"]] == ["1", "2", "3", "4"]
    assert signals[1] == [{"KIC": "1", "Per": "10.5", "BJD0": "55000.25"}]
    remaining.write_text(json.dumps({"remaining_catalogue_eb_kics": ["2", "1"]}))
    with pytest.raises(ValueError, match="frozen order"):
        catalogue_eb_controls.control_pool(identity, remaining, catalogue)
    remaining.write_text(json.dumps({"remaining_catalogue_eb_kics": ["1", "2"]}))
    catalogue.write_text(catalogue.read_text().replace("55000.25", "55001.25"))
    with pytest.raises(ValueError, match="ephemeris records changed"):
        catalogue_eb_controls.control_pool(identity, remaining, catalogue)


def test_catalogue_eb_materialization_excludes_aliases_and_binds_roles(
    tmp_path, monkeypatch
):
    native = catalogue_eb_controls
    monkeypatch.setattr(
        native.inputs.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=native.inputs.MINIMUM_FREE_BYTES + 1),
    )
    monkeypatch.setattr(native, "DEVELOPMENT", 1)
    monkeypatch.setattr(native, "EVALUATION", 1)
    identity, remaining, catalogue = eb_input_fixture(tmp_path)
    exclusions = tmp_path / "excluded.csv"
    exclusions.write_text("source_id,parent_tpf,raw_fits_sha256\nsource-2,99,\n")

    def download(path, kic, timeout):
        if kic == 1:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(kic).encode())
        return path

    def product(path, kic):
        metadata = {
            "source_id": f"source-{kic}",
            "origin_tpf_kic": kic,
            "raw_fits_sha256": native.inputs.digest(path),
        }
        arrays = {
            "time_bkjd": np.arange(100, dtype=float),
            "cadenceno": np.arange(100),
            "quarter": np.ones(100, dtype=int),
            "psf_flat_kbonus_flux": np.ones(100),
            "psf_flat_kbonus_flux_err": np.full(100, 0.001),
            "sap_quality": np.zeros(100, dtype=int),
            "flatten_mask": np.zeros(100, dtype=int),
        }
        return metadata, arrays

    monkeypatch.setattr(native.inputs, "download", download)
    monkeypatch.setattr(native.inputs, "read_product", product)
    out = tmp_path / "inputs"
    freeze = native.materialize(
        out, identity, remaining, catalogue, [exclusions], False, 1
    )
    assert freeze["selected"] == {native.CATEGORY: 2}
    assert freeze["scientific_evaluation_run"] is False
    assert json.loads((out / "development-source-ids.json").read_text()) == ["source-3"]
    assert json.loads((out / "evaluation-source-ids.json").read_text()) == ["source-4"]
    ledger = native.inputs.read_eligibility(out / "eligibility.jsonl")
    assert [row["status"] for row in ledger] == [
        "PRODUCT_UNAVAILABLE",
        "EXCLUDED_OPENED_OR_CONTROL_IDENTITY",
        "ELIGIBLE",
        "ELIGIBLE",
    ]
    assert (
        native.materialize(out, identity, remaining, catalogue, [exclusions], True, 1)
        == freeze
    )
    (out / ledger[-1]["arrays"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="product content changed"):
        native.materialize(out, identity, remaining, catalogue, [exclusions], True, 1)


def test_planet_control_selection_is_parent_and_raw_input_disjoint():
    rows = [
        {"source_id": str(index), "stratum": stratum}
        for index, stratum in enumerate(
            ("core", "core", "long", "ultra", "core", "core"), 1
        )
    ]
    manifest = {
        "sources": {
            "1": {"metadata": {"origin_tpf_kic": 10, "raw_fits_sha256": "a"}},
            "2": {"metadata": {"origin_tpf_kic": 10, "raw_fits_sha256": "b"}},
            "3": {"metadata": {"origin_tpf_kic": 30, "raw_fits_sha256": "c"}},
            "4": {"metadata": {"origin_tpf_kic": 40, "raw_fits_sha256": "d"}},
            "5": {"metadata": {"origin_tpf_kic": 50, "raw_fits_sha256": "a"}},
            "6": {"metadata": {"origin_tpf_kic": 60, "raw_fits_sha256": "e"}},
        }
    }
    selected, parents, raw_hashes, rejected = planet_control_partitions.select(
        rows,
        {"core": 2, "long": 1, "ultra": 1},
        manifest,
        {"20"},
        set(),
    )
    assert [row["source_id"] for row in selected] == ["1", "3", "4", "6"]
    assert parents == {"20", "10", "30", "40", "60"}
    assert raw_hashes == {"a", "c", "d", "e"}
    assert rejected == {"PARENT_OVERLAP": 1, "RAW_INPUT_OVERLAP": 1}


def test_planet_control_plan_supports_distinct_morphologies():
    plan = {
        "schema": 2,
        "categories": ["sparse_injection", "ttv_injection"],
        "partitions": {
            "development": {
                "categories": {
                    "sparse_injection": {"strata": {"sparse": 4}},
                    "ttv_injection": {"strata": {"ttv": 5}},
                }
            }
        },
    }
    assert planet_control_partitions.plan_categories(plan) == (
        "sparse_injection",
        "ttv_injection",
    )
    assert planet_control_partitions.partition_quotas(
        plan, "development", "ttv_injection"
    ) == {"ttv": 5}


def test_planet_control_support_ledgers_bind_resolved_parents(tmp_path):
    support = tmp_path / "support.jsonl"
    rows = [
        {
            "source_id": "1",
            "category": "sparse_injection",
            "status": "PIXEL_SUPPORTED",
            "pixels": {"5": {"base_parent_kic": 11}},
        },
        {
            "source_id": "2",
            "category": "sparse_injection",
            "status": "PIXELS_UNAVAILABLE",
        },
    ]
    support.write_text("".join(json.dumps(row) + "\n" for row in rows))
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "COMPLETE_PIXEL_SUPPORT_PREFLIGHT",
                "category": "sparse_injection",
                "sources": 2,
                "support_statuses": {
                    "PIXEL_SUPPORTED": 1,
                    "PIXELS_UNAVAILABLE": 1,
                },
                "artifacts": {
                    "support.jsonl": planet_control_partitions.digest(support)
                },
            }
        )
    )
    plan = {
        "pixel_support": {
            "ledgers": [
                {
                    "category": "sparse_injection",
                    "preflight": str(preflight),
                    "support": str(support),
                }
            ]
        }
    }
    selected, parents, inputs, counts = (
        planet_control_partitions.validated_pixel_support(plan, ("sparse_injection",))
    )
    assert selected == {"1"}
    assert parents == {"1": {"11"}}
    assert inputs == [preflight, support]
    assert counts["sparse_injection"] == {
        "PIXEL_SUPPORTED": 1,
        "PIXELS_UNAVAILABLE": 1,
    }


def test_planet_control_selection_rejects_resolved_parent_overlap():
    rows = [{"source_id": str(index), "stratum": "sparse"} for index in range(1, 4)]
    manifest = {
        "sources": {
            str(index): {
                "metadata": {
                    "origin_tpf_kic": index,
                    "raw_fits_sha256": str(index),
                }
            }
            for index in range(1, 4)
        }
    }
    selected, parents, _, rejected = planet_control_partitions.select(
        rows,
        {"sparse": 2},
        manifest,
        set(),
        set(),
        {"1": {"90"}, "2": {"90"}, "3": {"91"}},
    )
    assert [row["source_id"] for row in selected] == ["1", "3"]
    assert parents == {"90", "91"}
    assert rejected == {"RESOLVED_PARENT_OVERLAP": 1}


def test_planet_control_report_separates_periodic_sparse_and_terminal_losses():
    cases = [
        {
            "match": {
                "match_type": "PERIODIC",
                "detector_origin": "rust_v6_deterministic",
                "accepted_harmonic": 1,
                "classification": "TRANSIT_LIKE",
                "downstream": {
                    "triage": {
                        "status": "DISMISSED",
                        "stage": "localization",
                        "reason": "SOURCE_LOCALIZATION_CONFLICT",
                    },
                    "localization": {"status": "MEASURED"},
                },
            }
        },
        {
            "match": {
                "match_type": "PERIODIC",
                "detector_origin": "rust_v6_deterministic",
                "accepted_harmonic": -2,
                "classification": "SYSTEMATIC_LIKE",
                "downstream": {
                    "triage": {
                        "status": "HELD",
                        "stage": "localization",
                        "reason": "EVIDENCE_INCOMPLETE",
                    },
                    "localization": {"status": "UNRESOLVED"},
                },
            }
        },
        {
            "match": {
                "match_type": "SPARSE",
                "detector_origin": "rust_v6_deterministic",
                "accepted_harmonic": None,
                "classification": "EVENT_LIKE",
                "downstream": {
                    "triage": {
                        "status": "DISMISSED",
                        "stage": "shape",
                        "reason": "NON_TRANSIT_MODEL_STRONGLY_PREFERRED",
                    },
                    "localization": {"status": "NOT_REACHED"},
                },
            }
        },
        {"match": None},
    ]
    for case in cases:
        case["matches"] = [case["match"]] if case["match"] is not None else []
    summary = planet_control_report.summarize(cases, False)
    assert summary["matched_detector_proposals"] == 3
    assert summary["periodic_matches"] == 2
    assert summary["exact_period_matches"] == 1
    assert summary["accepted_harmonic_matches"] == 1
    assert summary["sparse_matches"] == 1
    assert summary["matched_terminal_stages"] == {"localization": 2, "shape": 1}
    assert summary["retained_injections"] == 1
    assert summary["dismissed_only_injections"] == 2


def test_planet_control_report_retains_alternative_matching_path():
    dismissed = {
        "match_type": "PERIODIC",
        "detector_origin": "rust_v6_deterministic",
        "accepted_harmonic": 1,
        "classification": "TRANSIT_LIKE",
        "downstream": {
            "triage": {"status": "DISMISSED", "stage": "shape", "reason": "X"},
            "localization": {"status": "NOT_REACHED"},
        },
    }
    active = {
        **dismissed,
        "downstream": {
            "triage": {"status": "ACTIVE", "stage": "survived", "reason": "NONE"},
            "localization": {"status": "MEASURED"},
        },
    }
    summary = planet_control_report.summarize(
        [{"match": dismissed, "matches": [dismissed, active]}], False
    )
    assert summary["matched_detector_proposals"] == 1
    assert summary["matching_detector_proposals"] == 2
    assert summary["retained_injections"] == 1
    assert summary["active_injections"] == 1


def test_sparse_match_accepts_period_free_injection_truth():
    signal = {
        "koi_period": None,
        "koi_time0bk": 100.0,
        "koi_duration": 12.0,
    }
    proposal = {
        "supplied_hypotheses": [
            {
                "period_days": 10.0,
                "duration_days": 0.5,
                "epoch_bkjd": 100.0,
                "followup_eligible": True,
            }
        ],
        "alerts": [
            {
                "kind": "SPARSE_EVENT_WINDOW",
                "start_bkjd": 99.75,
                "end_bkjd": 100.25,
                "detector_signal_id": "event-1",
                "detector_origin": "rust_v6_deterministic",
            },
            {
                "kind": "SPARSE_EVENT_WINDOW",
                "start_bkjd": 109.75,
                "end_bkjd": 110.25,
                "detector_signal_id": "event-2",
                "detector_origin": "rust_v6_deterministic",
            },
        ],
    }
    matches = catalogue_control_report.match_options(signal, proposal)
    assert [match["detector_signal_id"] for match in matches] == ["event-1"]
    assert matches[0]["match_type"] == "SPARSE"


def test_ttv_match_uses_declared_timing_excursion():
    signal = {
        "koi_period": 10.0,
        "koi_time0bk": 100.0,
        "koi_duration": 2.4,
        "ttv_amplitude_days": 0.15,
    }
    proposal = {
        "supplied_hypotheses": [],
        "alerts": [
            {
                "kind": "SPARSE_EVENT_WINDOW",
                "start_bkjd": 109.95,
                "end_bkjd": 110.35,
                "detector_signal_id": "event-1",
                "detector_origin": "rust_v6_deterministic",
            }
        ],
    }
    matches = catalogue_control_report.match_options(signal, proposal)
    assert len(matches) == 1
    assert matches[0]["ttv_tolerance_days"] == 0.15


def test_ordinary_controls_measure_incidence_without_a_zero_output_gate():
    plan = json.loads(
        Path(__file__).with_name("control-evaluation-plan.json").read_text()
    )
    measurement = plan["ordinary_candidate_incidence"]
    assert measurement["trials"] == 4500
    assert measurement["confidence"] == 0.9875
    assert measurement["launch_blocker"] is False
    assert "ordinary_false_boi_gate" not in plan


@pytest.fixture
def ordinary_control_run(tmp_path, monkeypatch):
    root = tmp_path
    partition = root / "partition"
    partition.mkdir()
    source_ids = partition / "development-source-ids.json"
    source_ids.write_text('["1", "2"]\n')
    freeze = partition / "freeze.json"
    parent_map = partition / "parent-map.csv"
    parent_map.write_text(
        "source_id,parent_tpf,raw_fits_sha256\n1,10,"
        + "a" * 64
        + "\n2,20,"
        + "b" * 64
        + "\n"
    )
    freeze.write_text(
        json.dumps(
            {
                "artifacts": {
                    "development-source-ids.json": control_report.digest(source_ids),
                    "parent-map.csv": control_report.digest(parent_map),
                }
            }
        )
    )
    plan = {
        "schema": 3,
        "outcome_contract": control_report.OUTCOME_CONTRACT,
        "parent_partition_freeze": {
            "path": str(freeze),
            "sha256": control_report.digest(freeze),
        },
        "ordinary_candidate_incidence": {"confidence": 0.9875},
        "independence": {"exposure_inputs": []},
        "partitions": {
            "development": {
                "status": "OPENED_DEVELOPMENT",
                "source_ids": str(source_ids),
                "source_ids_sha256": control_report.digest(source_ids),
            }
        },
    }
    exposure_path = root / "exposed.csv"
    exposure_path.write_text(
        "source_id,parent_tpf,raw_fits_sha256\n99,100," + "c" * 64 + "\n"
    )
    plan["independence"]["exposure_inputs"].append(
        {"path": str(exposure_path), "sha256": control_report.digest(exposure_path)}
    )
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    run = root / "run"
    (run / "vetting/sources").mkdir(parents=True)
    (run / "proposals").mkdir()
    policy = root / "policy.json"
    policy.write_text('{"boi_catalogue_enabled": false}\n')
    execution = disposition_execution({"boi_catalogue_enabled": False})
    fixtures = {
        "preparation.json": {
            "sources": [
                {"source_id": "1", "parent_tpf": "10", "raw_fits_sha256": "a" * 64},
                {"source_id": "2", "parent_tpf": "20", "raw_fits_sha256": "b" * 64},
            ]
        },
        "detection.json": {
            "status": "DETECTION_COMPLETE",
            "sources": 2,
            "alerted_sources": 1,
            "raw_rows": 3,
            "alerts": 1,
            "classifications": {"EVENT_LIKE": 1},
            "binary": {"sha256": "frozen"},
        },
        "discovery-manifest.json": {},
        "vetting/results.json": {
            "disposition_execution": execution,
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "sources": 1,
            "boi_signals": 0,
            "unresolved_signals": 1,
        },
        "vetting/manifest.json": {},
        "vetting/stage-timings.json": {
            "stages": [{"stage": "prepare", "status": "COMPLETE", "wall_seconds": 2}]
        },
        "vetting.json": {"disposition_policy": str(policy)},
    }
    for name, value in fixtures.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    source = {
        "disposition_policy": {"boi_catalogue_enabled": False},
        "disposition_execution": execution,
        "events": [
            {
                "event_id": "event-1",
                "original_alert": {"kind": "SPARSE_EVENT_WINDOW"},
            }
        ],
        "signals": [
            {
                "signal_id": "signal-1",
                "event_ids": ["event-1"],
                "triage": {
                    "status": "HELD",
                    "held_stage": "shape",
                    "reason": "SHAPE_MODEL_COMPARISON_CLOSE",
                },
            }
        ],
        "localization": {"events": []},
    }
    (run / "vetting/sources/1.json").write_text(json.dumps(source))
    (run / "proposals/1.json").write_text(
        json.dumps(
            {
                "source_id": "1",
                "alerts": [{"kind": "SPARSE_EVENT_WINDOW"}],
                "rust_rows": [{"classification": "EVENT_LIKE"}],
            }
        )
    )
    (run / "proposals/2.json").write_text(
        json.dumps(
            {
                "source_id": "2",
                "alerts": [],
                "rust_rows": [{"classification": "LOW_SIGNIFICANCE"}],
            }
        )
    )
    monkeypatch.setattr(control_report, "ROOT", root)
    return run, plan_path, fixtures


@pytest.mark.parametrize("field,value", [("schema", 2), ("outcome_contract", "legacy")])
def test_ordinary_report_rejects_legacy_plan_contract(
    ordinary_control_run, field, value
):
    run, plan_path, _ = ordinary_control_run
    plan = json.loads(plan_path.read_text())
    plan[field] = value
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Unsupported control-evaluation plan"):
        control_report.build_report(run, plan_path)


def test_control_report_reconciles_parent_independent_replay(ordinary_control_run):
    run, plan_path, fixtures = ordinary_control_run
    report = control_report.build_report(run, plan_path)
    paths = sorted((run / "vetting/sources").glob("*.json"))
    assert report["source_evidence"] == {
        "files": len(paths),
        "digest_ledger_sha256": control_report.json_digest(
            {path.name: control_report.digest(path) for path in paths}
        ),
    }
    plan_identity = control_report.artifact_path(plan_path, control_report.ROOT)
    assert report["inputs"][plan_identity] == control_report.digest(plan_path)
    reordered = json.loads(json.dumps(fixtures["preparation.json"]))
    reordered["sources"].reverse()
    (run / "preparation.json").write_text(json.dumps(reordered))
    reordered_report = control_report.build_report(run, plan_path)
    assert reordered_report["trial_ledger"] == report["trial_ledger"]
    for field, changed in (("parent_tpf", "99"), ("raw_fits_sha256", "c" * 64)):
        invalid = json.loads(json.dumps(fixtures["preparation.json"]))
        invalid["sources"][0][field] = changed
        (run / "preparation.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="identities differ from their freeze"):
            control_report.build_report(run, plan_path)
    (run / "preparation.json").write_text(json.dumps(fixtures["preparation.json"]))
    assert report["population"]["queued_parent_fraction"] == 0.5
    assert report["vetting"]["signal_kinds"] == {"SPARSE": 1}
    assert (
        report["candidate_incidence"]["status"] == "NOT_MEASURED_BOI_CATALOGUE_DISABLED"
    )
    assert report["schema"] == 3
    assert [row["status"] for row in report["trial_ledger"]["trials"]] == [
        "VETTING_COMPLETE",
        "DETECTOR_NO_PROPOSAL",
    ]
    assert report["trial_ledger"]["trials"][0]["signals"][0] == {
        "signal_id": "signal-1",
        "kind": "SPARSE",
        "event_ids": ["event-1"],
        "triage_status": "HELD",
        "terminal_stage": "shape",
        "terminal_reason": "SHAPE_MODEL_COMPARISON_CLOSE",
        "automatic_disposition": "UNAVAILABLE",
        "automatic_reason_codes": [],
    }
    assert report["trial_ledger"]["trials"][1]["boi_catalogue"] is None


def enabled_ordinary_run(run, status, role):
    path = run.parent / "policy.json"
    policy = {"status": status, "boi_catalogue_enabled": True}
    policy["definition_sha256"] = disposition_definition_sha256(policy)
    path.write_text(json.dumps(policy))
    source_policy = {
        **policy,
        "_evaluation_role": role,
        "_evaluation_authorized": role is not None,
    }
    execution = disposition_execution(source_policy)
    source_path = run / "vetting/sources/1.json"
    source = json.loads(source_path.read_text())
    source.update(disposition_policy=source_policy, disposition_execution=execution)
    source["disposition"] = {
        "status": "UNRESOLVED",
        "boi_signal_count": 0,
        "signals": [{"signal_id": "signal-1", "disposition": "UNRESOLVED"}],
    }
    source["processing_status"] = "EVIDENCE_STAGES_COMPLETE"
    source_path.write_text(json.dumps(source))
    result_path = run / "vetting/results.json"
    result = json.loads(result_path.read_text())
    result["disposition_execution"] = execution
    result_path.write_text(json.dumps(result))


def ordinary_final_plan(plan_path):
    plan = json.loads(plan_path.read_text())
    plan["partitions"]["evaluation"] = {
        **plan["partitions"]["development"],
        "status": "DETECTOR_AND_VETTER_OUTCOMES_UNOPENED",
    }
    plan_path.write_text(json.dumps(plan))
    return plan


@pytest.mark.parametrize(
    "status,role", [("FROZEN_EVALUATION", "final_replay"), ("RELEASED", None)]
)
def test_ordinary_report_separates_scientific_abstentions(
    ordinary_control_run, status, role
):
    run, plan_path, _ = ordinary_control_run
    enabled_ordinary_run(run, status, role)
    ordinary_final_plan(plan_path)
    report = control_report.build_report(run, plan_path, "evaluation")
    rate = report["candidate_incidence"]
    assert rate["boi_catalogue_parents"] == 0
    assert rate["scientific_abstention_parents"] == 1
    assert rate["execution_failure_parents"] == 0
    assert rate["parent_trials"] == 2
    assert rate["completed_parent_fraction"] == 1
    assert rate["fraction"] == 0
    assert rate["upper_bound"] is not None


def test_ordinary_report_counts_any_emitted_alias_when_count_is_unknown(
    ordinary_control_run,
):
    run, plan_path, _ = ordinary_control_run
    enabled_ordinary_run(run, "FROZEN_EVALUATION", "development_replay")
    path = run / "vetting/sources/1.json"
    source = json.loads(path.read_text())
    alias = {**source["signals"][0], "signal_id": "alias-2"}
    source["signals"].append(alias)
    source["disposition"]["boi_signal_count"] = None
    source["disposition"]["signals"].append(
        {"signal_id": "alias-2", "disposition": "BOI"}
    )
    path.write_text(json.dumps(source))
    report = control_report.build_report(run, plan_path)
    assert report["trial_ledger"]["trials"][0]["boi_catalogue"] is True
    assert report["vetting"]["candidate_parents"] == 1
    incidence = report["candidate_incidence"]
    assert incidence["boi_catalogue_parents"] == 1
    assert incidence["execution_failure_parents"] == 1
    assert incidence["scientific_abstention_parents"] == 0
    assert incidence["status"] == "INCOMPLETE_EXECUTION"
    assert incidence["upper_bound"] is None


@pytest.mark.parametrize("role", [None, "development_replay"])
def test_ordinary_final_report_requires_final_role(ordinary_control_run, role):
    run, plan_path, _ = ordinary_control_run
    enabled_ordinary_run(run, "FROZEN_EVALUATION", role)
    ordinary_final_plan(plan_path)
    with pytest.raises(ValueError, match="final-evaluation execution"):
        control_report.build_report(run, plan_path, "evaluation")


def test_ordinary_final_report_rejects_development_before_outcomes(
    ordinary_control_run,
):
    run, plan_path, _ = ordinary_control_run
    ordinary_final_plan(plan_path)
    (run / "vetting/results.json").unlink()
    with pytest.raises(ValueError, match="frozen automatic disposition policy"):
        control_report.build_report(run, plan_path, "evaluation")


@pytest.mark.parametrize(
    "field,exposed",
    [("source_id", "1"), ("parent_tpf", "10"), ("raw_fits_sha256", "a" * 64)],
)
def test_ordinary_final_report_rejects_exposed_identities_before_outcomes(
    ordinary_control_run, field, exposed
):
    run, plan_path, _ = ordinary_control_run
    enabled_ordinary_run(run, "FROZEN_EVALUATION", "final_replay")
    plan = ordinary_final_plan(plan_path)
    reference = plan["independence"]["exposure_inputs"][0]
    path = Path(reference["path"])
    row = {"source_id": "99", "parent_tpf": "100", "raw_fits_sha256": "c" * 64}
    row[field] = exposed
    path.write_text(
        "source_id,parent_tpf,raw_fits_sha256\n" + ",".join(row.values()) + "\n"
    )
    reference["sha256"] = control_report.digest(path)
    plan_path.write_text(json.dumps(plan))
    (run / "vetting/results.json").unlink()
    with pytest.raises(ValueError, match="overlaps declared exposed identities"):
        control_report.build_report(run, plan_path, "evaluation")


@pytest.mark.parametrize(
    "problem", ["superseded", "missing_exposure", "changed_exposure"]
)
def test_ordinary_final_report_requires_current_exposure_freeze(
    ordinary_control_run, problem
):
    run, plan_path, _ = ordinary_control_run
    enabled_ordinary_run(run, "FROZEN_EVALUATION", "final_replay")
    plan = ordinary_final_plan(plan_path)
    if problem == "superseded":
        plan["partitions"]["evaluation"]["status"] = (
            "SUPERSEDED_EXPOSED_PARENT_IDENTITIES"
        )
    elif problem == "missing_exposure":
        plan["independence"]["exposure_inputs"] = []
    else:
        path = Path(plan["independence"]["exposure_inputs"][0]["path"])
        path.write_text(path.read_text() + "98,101," + "d" * 64 + "\n")
    plan_path.write_text(json.dumps(plan))
    (run / "vetting/results.json").unlink()
    with pytest.raises(ValueError):
        control_report.build_report(run, plan_path, "evaluation")


def test_catalogue_control_pool_and_quarter_alignment(tmp_path):
    pools = tmp_path / "catalogue-pools.csv"
    with pools.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "category",
                "pool_rank",
                "target_count",
                "pool_count",
                "kic_id",
                "signal_count",
            ),
        )
        writer.writeheader()
        for category in catalogue_controls.CATEGORIES:
            for rank in (1, 2):
                writer.writerow(
                    {
                        "category": category,
                        "pool_rank": rank,
                        "target_count": 1,
                        "pool_count": 2,
                        "kic_id": 100 * rank + len(category),
                        "signal_count": 1,
                    }
                )
    frozen = catalogue_controls.frozen_pools(pools)
    assert all(value["target"] == 1 for value in frozen.values())
    names = set(catalogue_controls.QUARTER_COLUMNS.values())
    quarters = []
    for quarter, cadences in ((1, [2, 4]), (2, [1, 3])):
        row = {
            name: np.asarray(cadences, dtype=float) + index
            for index, name in enumerate(sorted(names))
        }
        row["CADENCENO"] = np.asarray(cadences)
        row["quarter"] = quarter
        quarters.append(row)
    arrays = catalogue_controls.aligned_quarter_arrays(
        np.asarray([1, 2, 3, 4]), quarters
    )
    np.testing.assert_array_equal(arrays["qdiag_cadenceno"], [1, 2, 3, 4])
    np.testing.assert_array_equal(arrays["qdiag_quarter"], [2, 1, 2, 1])
    assert catalogue_controls.product_uri(123).startswith("mast:HLSP/kbonus-bkg/")
    assert "mast%3AHLSP" in catalogue_controls.product_url(123)


@pytest.fixture
def catalogue_report_inputs(tmp_path):
    folder = tmp_path / "controls"
    folder.mkdir()
    truth = [
        {
            "category": "catalogue_false_positive",
            "source_id": "123",
            "kic_id": 45,
            "fits_sha256": "a" * 64,
        }
    ]
    manifest = {
        "sources": {
            "123": {
                "metadata": {
                    "source_id": "123",
                    "origin_tpf_kic": 45,
                    "raw_fits_sha256": "a" * 64,
                }
            }
        }
    }
    (folder / "truth.jsonl").write_text(json.dumps(truth[0]) + "\n")
    (folder / "source-ids.json").write_text(json.dumps(["123"]))
    (folder / "input-manifest.json").write_text(json.dumps(manifest))
    freeze = {
        "protocol": "fsc-kepler-catalogue-controls-v2",
        "status": "FROZEN_CATEGORY_INPUTS",
        "categories": ["catalogue_false_positive"],
        "selected": {"catalogue_false_positive": 1},
        "artifacts": {
            path.name: catalogue_control_report.digest(path)
            for path in folder.iterdir()
        },
    }
    (folder / "freeze.json").write_text(json.dumps(freeze))
    return folder, truth, freeze


@pytest.mark.parametrize(
    "status", ["PROBE_COMPLETE", "FROZEN_INPUTS", "FROZEN_CATEGORY_INPUTS"]
)
def test_catalogue_report_accepts_complete_input_statuses(
    catalogue_report_inputs, status
):
    folder, truth, freeze = catalogue_report_inputs
    freeze["status"] = status
    (folder / "freeze.json").write_text(json.dumps(freeze))
    _, ids, actual = catalogue_control_report.validate_controls(folder)
    assert ids == ["123"]
    assert actual == truth


@pytest.mark.parametrize("mutation", ["counts", "category", "binding", "raw_identity"])
def test_catalogue_report_rejects_incomplete_category_inputs(
    catalogue_report_inputs, mutation
):
    folder, _, freeze = catalogue_report_inputs
    if mutation == "counts":
        freeze["selected"]["catalogue_false_positive"] = 2
    elif mutation == "category":
        freeze["categories"] = ["catalogue_eclipsing_binary"]
    elif mutation == "binding":
        del freeze["artifacts"]["input-manifest.json"]
    else:
        path = folder / "input-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["sources"]["123"]["metadata"]["raw_fits_sha256"] = "b" * 64
        path.write_text(json.dumps(manifest))
        freeze["artifacts"][path.name] = catalogue_control_report.digest(path)
    (folder / "freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="input|identity"):
        catalogue_control_report.validate_controls(folder)


@pytest.fixture
def catalogue_report_run(catalogue_report_inputs, tmp_path):
    folder, truth, _ = catalogue_report_inputs
    run = tmp_path / "run"
    run.mkdir()
    manifest = folder / "input-manifest.json"
    preparation = {
        "input_manifest": {
            "path": str(manifest),
            "sha256": catalogue_control_report.digest(manifest),
        },
        "sources": [
            {"source_id": "123", "parent_tpf": "45", "raw_fits_sha256": "a" * 64}
        ],
    }
    (run / "preparation.json").write_text(json.dumps(preparation))
    config = {"input_manifest": str(manifest)}
    (run / "vetting.json").write_text(json.dumps(config))
    return folder, run, truth, preparation, config


@pytest.fixture
def native_eb_report_inputs(catalogue_report_inputs):
    folder, _, freeze = catalogue_report_inputs
    truth = [
        {
            "category": "catalogue_eclipsing_binary",
            "pool_rank": rank,
            "source_id": sid,
            "kic_id": kic,
            "fits_sha256": raw,
            "catalogue_signals": [
                {
                    "KIC": str(kic),
                    "Per": "9.6332256",
                    "BJD0": "55008.819044",
                    "HB": heartbeat,
                }
            ],
        }
        for rank, (sid, kic, raw, heartbeat) in enumerate(
            [("123", 45, "a" * 64, "1"), ("456", 46, "b" * 64, "0")], 1
        )
    ]
    manifest = {
        "sources": {
            row["source_id"]: {
                "metadata": {
                    "source_id": row["source_id"],
                    "origin_tpf_kic": row["kic_id"],
                    "raw_fits_sha256": row["fits_sha256"],
                }
            }
            for row in truth
        }
    }
    (folder / "truth.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in truth)
    )
    (folder / "source-ids.json").write_text(json.dumps(["123", "456"]))
    (folder / "input-manifest.json").write_text(json.dumps(manifest))
    for role, sid in [("development", "123"), ("evaluation", "456")]:
        (folder / f"{role}-source-ids.json").write_text(json.dumps([sid]))
    freeze.update(
        protocol="fsc-kepler-catalogue-eb-controls-v1",
        categories=["catalogue_eclipsing_binary"],
        selected={"catalogue_eclipsing_binary": 2},
        partitions={"development": 1, "evaluation": 1},
        claim_boundary="Native catalogue membership is not a reproduced mechanism",
    )
    freeze["artifacts"] = {
        path.name: catalogue_control_report.digest(path)
        for path in folder.iterdir()
        if path.name != "freeze.json"
    }
    (folder / "freeze.json").write_text(json.dumps(freeze))
    return folder, truth, freeze


@pytest.mark.parametrize(
    "partition,sid", [("development", "123"), ("evaluation", "456")]
)
def test_native_eb_report_selects_exact_partition(
    native_eb_report_inputs, partition, sid
):
    folder, truth, _ = native_eb_report_inputs
    _, ids, selected = catalogue_control_report.validate_controls(folder, partition)
    assert ids == [sid]
    assert selected == [row for row in truth if row["source_id"] == sid]


@pytest.mark.parametrize(
    "mutation", ["no_partition", "binding", "order", "count", "raw", "foreign_protocol"]
)
def test_native_eb_report_rejects_unbound_scope(native_eb_report_inputs, mutation):
    folder, _, freeze = native_eb_report_inputs
    partition = "development"
    if mutation == "no_partition":
        partition = None
    elif mutation == "binding":
        del freeze["artifacts"]["development-source-ids.json"]
    elif mutation == "order":
        for role, sid in [("development", "456"), ("evaluation", "123")]:
            path = folder / f"{role}-source-ids.json"
            path.write_text(json.dumps([sid]))
            freeze["artifacts"][path.name] = catalogue_control_report.digest(path)
    elif mutation == "count":
        freeze["partitions"]["development"] = 2
    elif mutation == "raw":
        path = folder / "input-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["sources"]["123"]["metadata"]["raw_fits_sha256"] = "c" * 64
        path.write_text(json.dumps(manifest))
        freeze["artifacts"][path.name] = catalogue_control_report.digest(path)
    else:
        freeze["protocol"] = "fsc-kepler-catalogue-controls-v2"
    (folder / "freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError):
        catalogue_control_report.validate_controls(folder, partition)


@pytest.fixture
def native_eb_report_run(native_eb_report_inputs, tmp_path, monkeypatch):
    folder, truth, _ = native_eb_report_inputs
    policy = catalogue_control_report.read(
        catalogue_control_report.ROOT / "pipeline/vetting/disposition-policy.json"
    )
    policy.update(status="DEVELOPMENT", boi_catalogue_enabled=False)
    monkeypatch.setattr(catalogue_control_report, "ROOT", tmp_path)
    run = tmp_path / "run"
    (run / "proposals").mkdir(parents=True)
    (run / "vetting/sources").mkdir(parents=True)
    policy_path = run / "policy.json"
    policy_path.write_text(json.dumps(policy))
    manifest = folder / "input-manifest.json"
    files = {
        "preparation.json": {
            "input_manifest": {
                "path": str(manifest),
                "sha256": catalogue_control_report.digest(manifest),
            },
            "sources": [
                {"source_id": "123", "parent_tpf": "45", "raw_fits_sha256": "a" * 64}
            ],
        },
        "vetting.json": {
            "input_manifest": str(manifest),
            "disposition_policy": str(policy_path),
        },
        "detection.json": {
            "status": "DETECTION_COMPLETE",
            "sources": 1,
            "alerted_sources": 0,
            "alerts": 0,
            "raw_rows": 0,
            "classifications": {},
            "binary": {"sha256": "c" * 64},
        },
        "queue.json": {"rows": []},
        "discovery-manifest.json": {},
        "vetting/manifest.json": {},
        "vetting/results.json": {
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "sources": 0,
            "original_alerts_preserved": 0,
            "boi_signals": 0,
            "unresolved_signals": 0,
            "disposition_execution": catalogue_control_report.disposition_execution(
                policy
            ),
        },
        "vetting/stage-timings.json": {
            "stages": [{"stage": "finish", "wall_seconds": 0.1, "status": "COMPLETE"}]
        },
        "proposals/123.json": {
            "source_id": "123",
            "rust_rows": [],
            "shallow_rows": [],
            "alerts": [],
        },
    }
    for name, value in files.items():
        (run / name).write_text(json.dumps(value))
    return folder, run, truth


def test_native_eb_report_preserves_truth_without_fabricated_recovery(
    native_eb_report_run,
):
    folder, run, truth = native_eb_report_run
    report = catalogue_control_report.build_report(folder, run, "development")
    assert report["population"]["declared_sources"] == 1
    assert report["partition"] == "development"
    case = report["cases"][0]
    assert case["catalogue_record"] == truth[0]["catalogue_signals"][0]
    assert case["catalogue_record"]["HB"] == "1"
    assert case["ephemeris"]["supplied_epoch_days"] == "55008.819044"
    assert case["ephemeris"]["epoch_conversion_applied"] is False
    assert case["ephemeris"]["duration_hours"] is None
    assert case["match"] is None
    recovery = report["detector"]["catalogue_signal_recovery"]
    assert recovery["matched_detector_proposals"] is None
    assert recovery["matched_fraction"] is None
    assert report["false_boi_rate"]["status"] == "NOT_MEASURED_BOI_CATALOGUE_DISABLED"
    assert report["false_boi_rate"]["trials"] == 1
    trial = report["trial_ledger"]["trials"][0]
    assert trial["status"] == "DETECTOR_NO_PROPOSAL"
    assert trial["catalogue_signals"][0]["catalogue_record"] == case["catalogue_record"]
    assert trial["boi_catalogue"] is None


def test_native_eb_final_report_requires_scientific_freeze_before_outcomes(
    native_eb_report_run, monkeypatch
):
    folder, run, _ = native_eb_report_run
    path = run / "preparation.json"
    preparation = json.loads(path.read_text())
    preparation["sources"] = [
        {"source_id": "456", "parent_tpf": "46", "raw_fits_sha256": "b" * 64}
    ]
    path.write_text(json.dumps(preparation))
    monkeypatch.setattr(
        catalogue_control_report,
        "validate_run",
        lambda *args: pytest.fail("Final outcomes opened before freeze"),
    )
    with pytest.raises(ValueError, match="frozen scientific policy"):
        catalogue_control_report.build_report(folder, run, "evaluation")


def test_native_eb_frozen_marker_does_not_bypass_operational_policy(
    native_eb_report_run, monkeypatch
):
    folder, run, _ = native_eb_report_run
    path = run / "policy.json"
    policy = json.loads(path.read_text())
    policy["status"] = "FROZEN_EVALUATION"
    policy["definition_sha256"] = (
        catalogue_control_report.disposition_definition_sha256(policy)
    )
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(
        catalogue_control_report,
        "validate_run",
        lambda *args: pytest.fail("Outcomes opened before policy validation"),
    )
    with pytest.raises(ValueError):
        catalogue_control_report.build_report(folder, run, "development")


def test_native_eb_frozen_preflight_must_return_the_declared_role(
    native_eb_report_run, monkeypatch
):
    from pipeline.deterministic import run as runner

    folder, run, _ = native_eb_report_run
    path = run / "policy.json"
    policy = json.loads(path.read_text())
    policy["status"] = "FROZEN_EVALUATION"
    path.write_text(json.dumps(policy))
    visited = []

    def preflight(args, sources, manifest):
        visited.append((args.disposition_policy, sources, manifest))
        return {"evaluation_role": "final_replay"}

    monkeypatch.setattr(runner, "evaluation_preflight", preflight)
    monkeypatch.setattr(
        catalogue_control_report,
        "validate_run",
        lambda *args: pytest.fail("Foreign role outcomes were opened"),
    )
    with pytest.raises(ValueError, match="different partition role"):
        catalogue_control_report.build_report(folder, run, "development")
    assert len(visited) == 1
    assert visited[0] == (
        path,
        [{"source_id": "123", "parent_tpf": "45", "raw_fits_sha256": "a" * 64}],
        str(folder / "input-manifest.json"),
    )


def test_catalogue_report_rejects_queue_outside_declared_partition(
    native_eb_report_run,
):
    _, run, _ = native_eb_report_run
    (run / "queue.json").write_text(json.dumps({"rows": [{"source_id": "999"}]}))
    (run / "vetting/sources/999.json").write_text(json.dumps({"source_id": "999"}))
    for filename in ["detection.json", "vetting/results.json"]:
        path = run / filename
        value = json.loads(path.read_text())
        value["alerted_sources" if filename == "detection.json" else "sources"] = 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        catalogue_control_report.validate_run(run, ["123"])


def test_catalogue_report_accepts_exact_frozen_replay_inputs(catalogue_report_run):
    folder, run, truth, _, _ = catalogue_report_run
    catalogue_control_report.validate_control_inputs(folder, run, truth)


@pytest.mark.parametrize(
    "mutation",
    ["manifest_path", "manifest_hash", "vetting_manifest", "source", "parent", "raw"],
)
def test_catalogue_report_rejects_foreign_replay_before_outcomes(
    catalogue_report_run, monkeypatch, mutation
):
    folder, run, _, preparation, config = catalogue_report_run
    other = run / "foreign-manifest.json"
    other.write_bytes((folder / "input-manifest.json").read_bytes())
    if mutation == "manifest_path":
        preparation["input_manifest"]["path"] = str(other)
    elif mutation == "manifest_hash":
        preparation["input_manifest"]["sha256"] = "b" * 64
    elif mutation == "vetting_manifest":
        config["input_manifest"] = str(other)
    else:
        field = {
            "source": "source_id",
            "parent": "parent_tpf",
            "raw": "raw_fits_sha256",
        }[mutation]
        preparation["sources"][0][field] = "b" * 64 if mutation == "raw" else "999"
    (run / "preparation.json").write_text(json.dumps(preparation))
    (run / "vetting.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        catalogue_control_report,
        "validate_run",
        lambda *args: pytest.fail("Foreign control outcomes were opened"),
    )
    with pytest.raises(ValueError, match="manifest|identity"):
        catalogue_control_report.build_report(folder, run)


def test_candidate_only_report_does_not_create_a_false_positive_bound():
    result = catalogue_control_report.false_boi_report(
        {"boi_catalogue_enabled": True},
        [{"category": "catalogue_candidate", "source_id": "123"}],
        {},
    )
    assert result["status"] == "NOT_APPLICABLE_NO_FALSE_POSITIVE_TRIALS"
    assert result["trials"] == 0
    assert result["zero_error_design_upper_bound"] is None


def test_catalogue_report_matches_harmonics_and_sparse_events_once():
    signals = [
        {"koi_period": 10, "koi_time0bk": 100, "koi_duration": 4},
        {"koi_period": 17, "koi_time0bk": 104, "koi_duration": 4},
    ]
    proposals = {
        "supplied_hypotheses": [
            {
                "signal_id": "periodic-1",
                "period_days": 5,
                "epoch_bkjd": 100.02,
                "duration_days": 0.1,
                "followup_eligible": True,
                "detector_followup_lane": "PRIMARY_TRANSIT_LIKE",
                "detector_origin": "rust_v6_deterministic",
                "rust_record": {"classification": "TRANSIT_LIKE"},
            }
        ],
        "alerts": [
            {
                "event_id": "event-1",
                "detector_signal_id": "sparse-1",
                "detector_origin": "rust_v6_deterministic",
                "kind": "SPARSE_EVENT_WINDOW",
                "start_bkjd": 103.9,
                "end_bkjd": 104.1,
            }
        ],
    }
    matches = catalogue_control_report.assign_truth_matches(signals, proposals)
    assert matches[0]["accepted_harmonic"] == -2
    assert matches[1]["match_type"] == "SPARSE"
    assert len({row["key"] for row in matches.values()}) == 2
    assert math.isclose(
        catalogue_control_report.exact_upper_bound(0, 500),
        1 - (1 - catalogue_control_report.CONTROL_CONFIDENCE) ** (1 / 500),
    )


@pytest.mark.parametrize(
    "count,alias,candidates,unresolved",
    [
        (None, "BOI", 1, 1),
        (0, "UNAVAILABLE", 0, 1),
        (0, "FALSE_ALARM", 0, 0),
    ],
)
@pytest.mark.parametrize(
    "category", ["catalogue_false_positive", "catalogue_eclipsing_binary"]
)
def test_catalogue_false_rate_accounts_for_every_alias(
    count, alias, candidates, unresolved, category
):
    truth = [{"category": category, "source_id": "42"}]
    source = {
        "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "signals": [{"signal_id": name} for name in ("matched", "other-alias")],
        "disposition": {
            "status": "FALSE_ALARM",
            "boi_signal_count": count,
            "signals": [
                {"signal_id": "matched", "disposition": "FALSE_ALARM"},
                {"signal_id": "other-alias", "disposition": alias},
            ],
        },
    }
    trials = [
        {
            "source_id": "42",
            "boi_catalogue": catalogue_control_report.emitted_candidate(
                source["disposition"]
            ),
            "disposition_incomplete": catalogue_control_report.incomplete_disposition(
                source
            ),
            "scientific_abstention": catalogue_control_report.scientific_abstention(
                source
            ),
        }
    ]
    report = catalogue_control_report.false_boi_report(
        {"boi_catalogue_enabled": True}, truth, trials
    )
    assert report["boi_catalogue_parents"] == candidates
    assert report["execution_failure_parents"] == unresolved
    assert report["scientific_abstention_parents"] == 0
    assert (report["upper_bound"] is None) == bool(unresolved)


@pytest.mark.parametrize("label", ["UNRESOLVED", "FALSE_ALARM", "BOI"])
def test_completed_dispositions_separate_abstention_from_emission(label):
    source = {
        "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "signals": [{"signal_id": "alias"}],
        "disposition": {
            "status": label,
            "boi_signal_count": int(label == "BOI"),
            "signals": [{"signal_id": "alias", "disposition": label}],
        },
    }
    assert catalogue_control_report.incomplete_disposition(source) is False
    assert catalogue_control_report.scientific_abstention(source) is (
        label == "UNRESOLVED"
    )
    assert catalogue_control_report.emitted_candidate(source["disposition"]) is (
        label == "BOI"
    )


@pytest.mark.parametrize(
    "defect", ["lost_alias", "duplicate", "count", "label", "status", "job"]
)
def test_incomplete_disposition_cannot_be_reported_as_scientific_abstention(defect):
    source = {
        "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "signals": [{"signal_id": "alias"}],
        "disposition": {
            "status": "UNRESOLVED",
            "boi_signal_count": 0,
            "signals": [{"signal_id": "alias", "disposition": "UNRESOLVED"}],
        },
    }
    assigned = source["disposition"]
    if defect == "lost_alias":
        source["signals"].append({"signal_id": "missing"})
    elif defect == "duplicate":
        assigned["signals"] *= 2
    elif defect == "count":
        assigned["boi_signal_count"] = True
    elif defect == "label":
        assigned["signals"][0]["disposition"] = "UNAVAILABLE"
    elif defect == "status":
        assigned["status"] = "UNAVAILABLE"
    else:
        source["processing_status"] = "PARTIAL"
    assert catalogue_control_report.incomplete_disposition(source) is True
    assert catalogue_control_report.scientific_abstention(source) is False


def test_false_boi_bound_requires_accounting_for_every_declared_parent():
    truth = [{"category": "catalogue_false_positive", "source_id": "42"}]
    with pytest.raises(ValueError, match="accounting is incomplete"):
        catalogue_control_report.false_boi_report(
            {"boi_catalogue_enabled": True}, truth, []
        )


def test_planet_control_report_counts_shallow_and_rust_matches_by_origin():
    signal = {"koi_period": 10, "koi_time0bk": 100, "koi_duration": 2.4}
    proposal = {
        "supplied_hypotheses": [
            {
                "signal_id": "rust-1",
                "period_days": 10,
                "epoch_bkjd": 100,
                "duration_days": 0.1,
                "followup_eligible": True,
                "detector_followup_lane": "PRIMARY_TRANSIT_LIKE",
                "detector_origin": "rust_v6_deterministic",
                "rust_record": {"classification": "TRANSIT_LIKE"},
            },
            {
                "signal_id": "shallow-1",
                "period_days": 10,
                "epoch_bkjd": 100,
                "duration_days": 0.1,
                "followup_eligible": True,
                "detector_followup_lane": "SHALLOW_BLS",
                "detector_origin": "shallow_bls_v1",
                "shallow_record": {"search_rank": 1},
            },
        ],
        "alerts": [],
    }
    options = catalogue_control_report.match_options(signal, proposal)
    assert {option["classification"] for option in options} == {
        "TRANSIT_LIKE",
        "SHALLOW_BLS_PROPOSAL",
    }
    for option in options:
        status = "ACTIVE" if option["detector_origin"] == "shallow_bls_v1" else "HELD"
        option["downstream"] = {
            "triage": {"status": status, "stage": "localization", "reason": "NONE"},
            "localization": {"status": "MEASURED"},
        }
    summary = planet_control_report.summarize(
        [{"match": options[0], "matches": options}], False
    )
    assert summary["matched_detector_proposals"] == 1
    assert summary["matched_injections_by_detector_origin"] == {
        "rust_v6_deterministic": 1,
        "shallow_bls_v1": 1,
    }
    assert summary["active_injections_by_detector_origin"] == {"shallow_bls_v1": 1}


def physical_truth(category="periodic_injection"):
    return {
        "source_id": "42",
        "category": category,
        "shape_period_days": 10.0,
        "period_days": None if category == "sparse_injection" else 10.0,
        "epoch_bkjd": 100.0,
        "duration_hours": 2.4,
        "ttv": {"amplitude_days": 0.4, "superperiod_cycles": 4, "phase_radians": 0.8}
        if category == "ttv_injection"
        else None,
    }


def physical_events(centres):
    return [
        {
            "event_id": f"truth-{index}",
            "centre_bkjd": centre,
            "support": {"1": {"psf_flat": 3}},
        }
        for index, centre in enumerate(centres)
    ]


def physical_units(centres):
    return [
        {"unit_id": f"unit-{index}", "centre_bkjd": centre, "quarter": 1}
        for index, centre in enumerate(centres)
    ]


def test_exact_event_matching_rejects_long_baseline_period_drift():
    known = {"koi_period": 10, "koi_time0bk": 100, "koi_duration": 2.4}
    hypothesis = {
        "signal_id": "wrong",
        "period_days": 10.005,
        "epoch_bkjd": 100,
        "duration_days": 0.1,
        "followup_eligible": True,
        "detector_origin": "rust_v6_deterministic",
        "detector_followup_lane": "PRIMARY_TRANSIT_LIKE",
        "rust_record": {"classification": "TRANSIT_LIKE"},
    }
    assert catalogue_control_report.periodic_option(known, hypothesis) is not None
    result = planet_control_report.match_event_units(
        physical_events([100, 600, 1100]), physical_units([100, 600.25, 1100.5]), 2.4
    )
    assert result["status"] == "NOT_MATCHED"
    assert result["matched_units"] == 1


def test_harmonic_subset_and_single_event_recover_physical_signal():
    events = physical_events([100, 110, 120, 130])
    for times in ([100, 120], [110]):
        result = planet_control_report.match_event_units(
            events, physical_units(times), 2.4
        )
        assert result["status"] == "MATCHED"
        assert result["truth_event_coverage"] == len(times) / 4


def test_exact_event_matching_rejects_duplicate_truth_identities():
    events = physical_events([100, 110])
    events[1]["event_id"] = events[0]["event_id"]
    with pytest.raises(ValueError, match="Truth event identities"):
        planet_control_report.match_event_units(events, physical_units([100]), 2.4)


@pytest.mark.parametrize("defect", ["wrong_quarter", "duplicate_event", "wide_window"])
def test_unrelated_and_duplicated_event_units_do_not_recover_a_signal(defect):
    units = physical_units([100])
    if defect == "wrong_quarter":
        units[0]["quarter"] = 2
    elif defect == "duplicate_event":
        units.append({**units[0], "unit_id": "correlated-copy"})
    else:
        units[0].update(centre_bkjd=100.5, duration_days=10)
    result = planet_control_report.match_event_units(physical_events([100]), units, 2.4)
    assert result["status"] == "NOT_MATCHED"


def test_correlated_views_share_one_physical_truth_event():
    truth = physical_truth("sparse_injection")
    records = np.array(
        [(99.98, 1), (100, 1), (100.02, 1)], dtype=[("time", "f8"), ("quarter", "i2")]
    )
    events = planet_control_report.truth_event_ledger(
        truth, {"psf_flat": records, "psf_qdiag": records}
    )
    assert len(events) == 1
    assert events[0]["support"] == {"1": {"psf_flat": 3, "psf_qdiag": 3}}
    assert (
        planet_control_report.match_event_units(events, physical_units([100]), 2.4)[
            "status"
        ]
        == "MATCHED"
    )


def test_ttv_matching_uses_the_physical_clock_without_expanding_tolerance():
    truth = physical_truth("ttv_injection")
    centres = planet_control_report.injections.truth_centres(truth, 100, 140)
    visible = [centre for centre in centres if 100 <= centre <= 140]
    records = np.array(
        [(centre, 1) for centre in visible], dtype=[("time", "f8"), ("quarter", "i2")]
    )
    events = planet_control_report.truth_event_ledger(truth, {"psf_flat": records})
    actual = planet_control_report.match_event_units(
        events, physical_units(visible), 2.4
    )
    linear = [100 + round((centre - 100) / 10) * 10 for centre in visible]
    incorrect = planet_control_report.match_event_units(
        events, physical_units(linear), 2.4
    )
    assert actual["status"] == "MATCHED"
    assert incorrect["status"] == "NOT_MATCHED"
    assert actual["timing_tolerance_days"] < truth["ttv"]["amplitude_days"]


@pytest.mark.parametrize("selected_refinement", [False, True])
def test_fitted_event_matching_uses_the_selected_joint_model(selected_refinement):
    units = physical_units([100, 110])
    signal = {"events": units}
    selected = {
        "hypothesis_id": "h",
        "view": "psf_flat",
        "unit_ids": [unit["unit_id"] for unit in units],
    }
    timing = {"start_shift_days": 0.02, "slope": 0.001, "first_centre_bkjd": 100}
    fitted = {"status": "FIT_COMPLETE", "theta": [0.05]}
    result = {
        "views": {"psf_flat": {"joint_shape": fitted}},
        "timing_refinement": {
            "timing": timing,
            "views": {"psf_flat": {"joint_shape": fitted}},
        },
    }
    if selected_refinement:
        selected["timing_refinement"] = {"timing": timing}
    row = {"orbital": {"hypotheses": {"h": result}}}
    assigned = {"signal_shape": {"records": [selected]}}
    output = planet_control_report.fitted_event_units(row, signal, assigned)
    expected = [100.07, 110.08] if selected_refinement else [100.05, 110.05]
    assert np.allclose([unit["centre_bkjd"] for unit in output], expected)


def test_catalogue_trial_ledger_includes_detector_misses(tmp_path):
    (tmp_path / "proposals").mkdir()
    truth = [
        {
            "source_id": "1",
            "kic_id": 10,
            "category": "catalogue_candidate",
            "pool_rank": 1,
        },
        {
            "source_id": "2",
            "kic_id": 20,
            "category": "catalogue_false_positive",
            "pool_rank": 1,
        },
    ]
    for source_id, alerts, classification in (
        ("1", [{"kind": "PERIODIC_WINDOW"}], "TRANSIT_LIKE"),
        ("2", [], "LOW_SIGNIFICANCE"),
    ):
        (tmp_path / "proposals" / f"{source_id}.json").write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "alerts": alerts,
                    "rust_rows": [{"classification": classification}],
                }
            )
        )
    rows = {
        "1": {
            "events": [
                {
                    "event_id": "event-1",
                    "original_alert": {"kind": "PERIODIC_WINDOW"},
                }
            ],
            "signals": [
                {
                    "signal_id": "signal-1",
                    "event_ids": ["event-1"],
                    "triage": {
                        "status": "DISMISSED",
                        "dismissed_stage": "shape",
                        "reason": "NON_TRANSIT_MODEL_STRONGLY_PREFERRED",
                    },
                }
            ],
            "disposition": {"status": "UNRESOLVED", "signals": []},
        }
    }
    cases = [
        {
            "source_id": source_id,
            "kepoi_name": f"K{source_id}",
            "match": None,
        }
        for source_id in ("1", "2")
    ]

    ledger = catalogue_control_report.trial_ledger(tmp_path, truth, rows, cases, False)
    assert [row["status"] for row in ledger] == [
        "VETTING_COMPLETE",
        "DETECTOR_NO_PROPOSAL",
    ]
    assert ledger[0]["signals"][0]["terminal_reason"] == (
        "NON_TRANSIT_MODEL_STRONGLY_PREFERRED"
    )
    assert ledger[1]["boi_catalogue"] is None


def test_catalogue_evaluation_partition_excludes_opened_identities():
    rows = []
    for category in catalogue_control_partition.CATEGORIES:
        for rank in range(1, 4):
            rows.append(
                {
                    "category": category,
                    "status": "ELIGIBLE",
                    "pool_rank": rank,
                    "source_id": f"{category}-{rank}",
                    "kic_id": f"{category}-kic-{rank}",
                    "fits_sha256": f"{category}-fits-{rank}",
                }
            )
    opened = {
        "source_id": {"catalogue_candidate-1", "catalogue_false_positive-1"},
        "kic_id": set(),
        "fits_sha256": set(),
    }
    selected = catalogue_control_partition.select_evaluation(
        rows,
        {category: 2 for category in catalogue_control_partition.CATEGORIES},
        opened,
    )
    assert {row["pool_rank"] for values in selected.values() for row in values} == {
        2,
        3,
    }


def test_freeze_hash_binds_inputs_and_outputs(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    index = root / "index.sqlite"
    with sqlite3.connect(index) as database:
        database.execute(
            "create table sources(gaia_source_id text, volume text, shard integer)"
        )
        database.executemany(
            "insert into sources values (?, 'discSSD', 0)",
            [(str(value),) for value in range(1, 9)],
        )
    opened = root / "opened.csv"
    with opened.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "kic_id"])
        writer.writeheader()
        writer.writerow({"source_id": "1", "kic_id": "100"})
    catalogue = root / "koi.json"
    catalogue.write_text(
        json.dumps(
            [
                {"kepid": 200, "koi_disposition": "CANDIDATE"},
                {"kepid": 300, "koi_disposition": "FALSE POSITIVE"},
            ]
        )
    )
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": 1,
                "protocol": "test",
                "selection_seed": "seed",
                "source_index": "index.sqlite",
                "excluded_source_manifests": [
                    {"path": "opened.csv", "column": "source_id"}
                ],
                "source_cohorts": {"ordinary": {"target": 2, "pool": 4}},
                "koi_catalogue": "koi.json",
                "opened_confirmed_manifest": "opened.csv",
                "opened_false_positive_manifest": "opened.csv",
                "catalogue_cohorts": {
                    "candidate": {
                        "disposition": "CANDIDATE",
                        "target": 1,
                        "pool": 1,
                    },
                    "false_positive": {
                        "disposition": "FALSE POSITIVE",
                        "target": 1,
                        "pool": 1,
                    },
                },
            }
        )
    )
    monkeypatch.setattr(controls, "ROOT", root)

    report = controls.freeze(tmp_path / "out", plan)
    assert report["status"] == "FROZEN_SOURCE_POOLS"
    assert report["source_pool_counts"] == {"ordinary": 4}
    assert report["catalogue_pool_counts"] == {
        "candidate": 1,
        "false_positive": 1,
    }
    for name, sha256 in report["artifacts"].items():
        assert controls.digest(tmp_path / "out" / name) == sha256


def test_false_class_pool_is_deterministic_and_parent_disjoint():
    sources = {}
    for parent in range(1, 12):
        for member in range(2):
            source_id = str(100 * parent + member)
            sources[source_id] = {
                "source_id": source_id,
                "origin_tpf_kic": parent,
                "ra_deg": 300 + parent / 100 + member * 5 / 3600,
                "dec_deg": 40,
                "phot_g_mean_mag": 14,
                "quarter_count": 4,
                "finite_psf_flat_kbonus": 10000,
                "raw_fits_sha256": source_id,
                "volume": "discSSD",
            }
    plan = {
        "seed": "test",
        "classes": [
            "instrumental_artifact",
            "eclipsing_binary",
            "background_contaminant",
        ],
        "candidate_pool_per_class": 2,
        "minimum_quarters": 3,
        "minimum_finite_primary_cadences": 1000,
        "background_separation_arcsec": [3, 12],
        "background_max_magnitude_difference": 3.5,
    }
    rows = false_class_controls.select_pool(plan, sources, {"1"})
    assert rows == false_class_controls.select_pool(plan, sources, {"1"})
    assert len(rows) == 6
    assert len({row["parent_tpf"] for row in rows}) == 6
    assert all(row["parent_tpf"] != "1" for row in rows)
    assert all(
        row["contaminant_source_id"]
        for row in rows
        if row["category"] == "background_contaminant"
    )


def test_catalogue_evaluation_excludes_all_prepared_development_identities(tmp_path):
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": {
                    "41": {
                        "metadata": {
                            "source_id": "41",
                            "origin_tpf_kic": 51,
                            "raw_fits_sha256": "a" * 64,
                        }
                    },
                    "42": {
                        "metadata": {
                            "source_id": "42",
                            "origin_tpf_kic": 52,
                            "raw_fits_sha256": "b" * 64,
                        }
                    },
                }
            }
        )
    )
    run = tmp_path / "opened-run"
    run.mkdir()
    (run / "preparation.json").write_text(
        json.dumps(
            {
                "status": "PREPARED",
                "input_manifest": {
                    "path": str(manifest),
                    "sha256": catalogue_control_partition.digest(manifest),
                },
                "sources": [{"source_id": "41"}, {"source_id": "42"}],
            }
        )
    )
    identities, inputs = catalogue_control_partition.opened_identities([], [run])
    assert identities == {
        "source_id": {"41", "42"},
        "kic_id": {"51", "52"},
        "fits_sha256": {"a" * 64, "b" * 64},
    }
    assert str(manifest) in inputs
    candidates = [
        {
            "status": "ELIGIBLE",
            "category": "catalogue_false_positive",
            "source_id": "41",
            "kic_id": "51",
            "fits_sha256": "a" * 64,
        },
        {
            "status": "ELIGIBLE",
            "category": "catalogue_false_positive",
            "source_id": "43",
            "kic_id": "51",
            "fits_sha256": "c" * 64,
        },
        {
            "status": "ELIGIBLE",
            "category": "catalogue_false_positive",
            "source_id": "44",
            "kic_id": "54",
            "fits_sha256": "b" * 64,
        },
        {
            "status": "ELIGIBLE",
            "category": "catalogue_false_positive",
            "source_id": "45",
            "kic_id": "55",
            "fits_sha256": "d" * 64,
        },
    ]
    selected = catalogue_control_partition.select_evaluation(
        candidates, {"catalogue_false_positive": 1}, identities
    )
    assert [row["source_id"] for row in selected["catalogue_false_positive"]] == ["45"]
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="identities changed"):
        catalogue_control_partition.opened_run_identities(run)


def test_catalogue_evaluation_excludes_other_cohort_parent_maps(tmp_path):
    path = tmp_path / "parents.csv"
    path.write_text(
        "source_id,parent_tpf,resolved_parent_tpf,raw_fits_sha256\n41,51,52;53,"
        + "a" * 64
        + "\n"
    )
    identities, inputs = catalogue_control_partition.opened_identities(
        [], parent_maps=[path]
    )
    assert identities == {
        "source_id": {"41"},
        "kic_id": {"51", "52", "53"},
        "fits_sha256": {"a" * 64},
    }
    assert inputs[str(path)] == catalogue_control_partition.digest(path)


@pytest.mark.parametrize("requested_category", [None, "catalogue_false_positive"])
def test_catalogue_materialization_excludes_identities_before_acceptance(
    tmp_path, monkeypatch, requested_category
):
    monkeypatch.setattr(
        catalogue_controls.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=catalogue_controls.MINIMUM_FREE_BYTES + 1),
    )
    pools = tmp_path / "pools.csv"
    with pools.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "category",
                "pool_rank",
                "target_count",
                "pool_count",
                "kic_id",
                "signal_count",
            ],
        )
        writer.writeheader()
        for category, kics in (
            ("catalogue_candidate", [51, 52, 53, 54]),
            ("catalogue_false_positive", [61, 62]),
        ):
            for rank, kic in enumerate(kics, 1):
                writer.writerow(
                    {
                        "category": category,
                        "pool_rank": rank,
                        "target_count": 1,
                        "pool_count": len(kics),
                        "kic_id": kic,
                        "signal_count": 1,
                    }
                )
    parent_map = tmp_path / "parents.csv"
    parent_map.write_text(
        "source_id,parent_tpf,raw_fits_sha256\n41,51," + "a" * 64 + "\n"
    )
    calls = []

    def download(out, category, timeout, row):
        calls.append(int(row["kic_id"]))
        return tmp_path / str(row["kic_id"])

    def product(path, kic):
        return {
            "source_id": "41" if kic == 52 else str(kic + 100),
            "origin_tpf_kic": kic,
            "raw_fits_sha256": "a" * 64 if kic == 53 else str(kic).zfill(64),
        }, {}

    def arrays(path, values):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"arrays")

    signals = {
        kic: [
            {
                "kepid": kic,
                "koi_disposition": "CANDIDATE" if kic < 60 else "FALSE POSITIVE",
            }
        ]
        for kic in [51, 52, 53, 54, 61, 62]
    }
    monkeypatch.setattr(catalogue_controls, "download_pool_row", download)
    monkeypatch.setattr(catalogue_controls, "read_product", product)
    monkeypatch.setattr(catalogue_controls, "discovery_eligible", lambda _: True)
    monkeypatch.setattr(catalogue_controls, "write_arrays", arrays)
    monkeypatch.setattr(catalogue_controls, "catalogue_signals", lambda: signals)
    out = tmp_path / "controls"
    freeze = catalogue_controls.materialize(
        out, False, None, 1, 2, 1, pools, [parent_map], requested_category
    )
    assert 51 not in calls
    assert catalogue_controls.digest(parent_map) in freeze["inputs"].values()
    if requested_category:
        assert set(calls) <= {61, 62}
        assert freeze["status"] == "FROZEN_CATEGORY_INPUTS"
        assert freeze["categories"] == [requested_category]
        assert freeze["selected"] == {requested_category: 1}
        assert not (out / "catalogue_candidate-source-ids.json").exists()
        with pytest.raises(ValueError, match="run identity changed"):
            catalogue_controls.materialize(
                out, True, None, 1, 2, 1, pools, [parent_map]
            )
        return
    assert json.loads((out / "catalogue_candidate-source-ids.json").read_text()) == [
        "154"
    ]
    assert freeze["selected"] == {
        "catalogue_candidate": 1,
        "catalogue_false_positive": 1,
    }
    assert freeze["eligibility_counts"]["EXCLUDED_PARENT_IDENTITY"] == 1
    assert freeze["eligibility_counts"]["EXCLUDED_OPENED_OR_CONTROL_IDENTITY"] == 2
    parent_map.write_text(parent_map.read_text() + "99,99," + "c" * 64 + "\n")
    with pytest.raises(ValueError, match="run identity changed"):
        catalogue_controls.materialize(out, True, None, 1, 2, 1, pools, [parent_map])


def test_catalogue_insufficient_identity_capacity_stops_before_download(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        catalogue_controls,
        "download_pool_row",
        lambda *_: pytest.fail("Cannot reach the declared trial target"),
    )
    with pytest.raises(ValueError, match="pool exhausted"):
        catalogue_controls.materialize_category(
            tmp_path,
            "catalogue_candidate",
            {"rows": [{"pool_rank": 1}]},
            {},
            2,
            tmp_path / "ledger.jsonl",
            [],
            1,
            1,
            1,
        )


def test_catalogue_materialization_requires_capacity_before_download(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        catalogue_controls.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    monkeypatch.setattr(
        catalogue_controls,
        "download_pool_row",
        lambda *_: pytest.fail("Download started below capacity reserve"),
    )
    with pytest.raises(RuntimeError, match="less than 20 GiB"):
        catalogue_controls.materialize_category(
            tmp_path,
            "catalogue_candidate",
            {"rows": [{"pool_rank": 1}]},
            {},
            1,
            tmp_path / "ledger.jsonl",
            [],
            1,
            1,
            1,
        )


def test_catalogue_input_mutation_cannot_receive_a_freeze(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text("original")
    spec = {"inputs": {str(path): catalogue_controls.digest(path)}}
    path.write_text("changed")
    with pytest.raises(ValueError, match="inputs changed before freeze"):
        catalogue_controls.freeze_outputs(tmp_path, spec, [], {})
    assert not (tmp_path / "freeze.json").exists()


@pytest.mark.parametrize("through_alias", [False, True])
def test_ordinary_report_preserves_external_output_identities(
    ordinary_control_run, tmp_path, through_alias
):
    run, plan, _ = ordinary_control_run
    original = control_report.build_report(run, plan)
    external = tmp_path.parent / (tmp_path.name + "-external-ordinary")
    run.rename(external)
    if through_alias:
        run.symlink_to(external, target_is_directory=True)
        selected = run
    else:
        selected = external
    report = control_report.build_report(selected, plan)
    assert report["trial_ledger"] == original["trial_ledger"]
    expected = external / "vetting/results.json"
    assert report["inputs"][str(expected)] == control_report.digest(expected)
    assert all(not key.startswith("..") for key in report["inputs"])


@pytest.mark.parametrize("through_alias", [False, True])
def test_native_eb_report_preserves_external_output_identities(
    native_eb_report_run, tmp_path, through_alias
):
    folder, run, _ = native_eb_report_run
    original = catalogue_control_report.build_report(folder, run, "development")
    external = tmp_path.parent / (tmp_path.name + "-external-native")
    run.rename(external)
    configuration = json.loads((external / "vetting.json").read_text())
    configuration["disposition_policy"] = str(external / "policy.json")
    (external / "vetting.json").write_text(json.dumps(configuration))
    if through_alias:
        run.symlink_to(external, target_is_directory=True)
        selected = run
    else:
        selected = external
    report = catalogue_control_report.build_report(folder, selected, "development")
    assert report["trial_ledger"] == original["trial_ledger"]
    expected = external / "vetting/results.json"
    assert report["inputs"][str(expected)] == catalogue_control_report.digest(expected)
    assert all(not key.startswith("..") for key in report["inputs"])


@pytest.fixture
def opened_scope_rows():
    rows = [
        {
            "source_id": sid,
            "cohort": "ordinary",
            "origin_parent": parent,
            "raw_fits_sha256": checksum * 64,
            "full_witness_component_id": component,
            "role": "development",
            "scope": "CURRENT_OPENED_DEVELOPMENT",
            "trial_id": f"ordinary:development:{sid}",
            "input_status": status,
        }
        for sid, parent, checksum, component, status in (
            ("1", "10", "a", "opened", "ALL_ORIGINAL_ARRAY_CONTENT_VERIFIED"),
            ("2", "20", "b", "shared", "EXCLUDED_STOPPED_SHARED_COMPONENT"),
        )
    ]
    trials = {row["trial_id"]: dict(row) for row in rows}
    return {"metadata_ledger": rows}, trials, {"shared"}


def test_opened_scope_retains_protected_trials_without_fabricating_outcomes(
    opened_scope_rows,
):
    scope, trials, stopped = opened_scope_rows
    selected, metadata_only = catalogue_control_report.opened_scope_rows(
        scope, trials, stopped, ["1", "2"]
    )
    assert selected == ["1"]
    assert [row["source_id"] for row in metadata_only] == ["2"]
    assert metadata_only[0]["execution_complete"] is False
    assert metadata_only[0]["boi_catalogue"] is None


@pytest.mark.parametrize(
    "field",
    ["origin_parent", "raw_fits_sha256", "full_witness_component_id", "role", "cohort"],
)
def test_opened_scope_refuses_changed_frozen_identities(opened_scope_rows, field):
    scope, trials, stopped = opened_scope_rows
    scope["metadata_ledger"][0][field] = "changed"
    with pytest.raises(ValueError, match="frozen trial identity"):
        catalogue_control_report.opened_scope_rows(scope, trials, stopped, ["1", "2"])


@pytest.mark.parametrize(
    "index,status",
    [
        (0, "EXCLUDED_STOPPED_SHARED_COMPONENT"),
        (1, "ALL_ORIGINAL_ARRAY_CONTENT_VERIFIED"),
    ],
)
def test_opened_scope_cannot_drop_an_executable_source_or_open_a_stopped_component(
    opened_scope_rows, index, status
):
    scope, trials, stopped = opened_scope_rows
    scope["metadata_ledger"][index]["input_status"] = status
    with pytest.raises(ValueError, match="stopped component decision"):
        catalogue_control_report.opened_scope_rows(scope, trials, stopped, ["1", "2"])


@pytest.mark.parametrize("rows", [[], [0], [0, 0], [1, 0]])
def test_opened_scope_cannot_shrink_duplicate_or_reorder_the_original_ledger(
    opened_scope_rows, rows
):
    scope, trials, stopped = opened_scope_rows
    original = scope["metadata_ledger"]
    scope["metadata_ledger"] = [original[index] for index in rows]
    with pytest.raises(ValueError, match="original cohort ledger"):
        catalogue_control_report.opened_scope_rows(scope, trials, stopped, ["1", "2"])


def test_native_opened_subset_report_retains_the_denominator_and_withholds_a_bound(
    native_eb_report_run, monkeypatch
):
    folder, run, _ = native_eb_report_run
    scope = {
        "selected_source_ids": ["123"],
        "declared_sources": 2,
        "metadata_only_trials": [{"source_id": "protected", "boi_catalogue": None}],
        "inputs": {},
    }
    monkeypatch.setattr(
        catalogue_control_report, "opened_replay_scope", lambda *args: scope
    )
    result = catalogue_control_report.build_report(
        folder, run, "development", run / "execution.json"
    )
    assert result["status"] == "PARTIAL_OPENED_CATALOGUE_CONTROL_REPORT"
    assert result["population"]["declared_sources"] == 2
    assert result["population"]["prepared_sources"] == 1
    assert (
        result["opened_execution_scope"]["metadata_only_trials"]
        == scope["metadata_only_trials"]
    )
    assert result["false_boi_rate"]["upper_bound"] is None


def test_native_opened_subset_reporting_cannot_authorize_sealed_evaluation(
    native_eb_report_run, monkeypatch
):
    folder, run, _ = native_eb_report_run
    monkeypatch.setattr(
        catalogue_control_report,
        "validate_controls",
        lambda *args: (
            {"protocol": catalogue_control_report.NATIVE_EB_PROTOCOL},
            [],
            [],
        ),
    )
    with pytest.raises(ValueError, match="native EB development"):
        catalogue_control_report.build_report(
            folder, run, "evaluation", run / "execution.json"
        )
