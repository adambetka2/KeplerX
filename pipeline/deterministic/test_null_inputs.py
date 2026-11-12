import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/vetting"))

from pipeline.deterministic import run, signal_contracts
from pipeline.photometry import lc2
from pipeline.vetting import vetting


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def bound_lc2(tmp_path):
    relative = "discSSD/data/kbonus/lc2/shard-0000"
    folder = tmp_path / relative
    folder.mkdir(parents=True)
    count = 360
    data = {
        "source_id": np.full(count, 123, dtype=np.int64),
        "time_bkjd": 100 + np.arange(count) * 0.02,
        "quarter": np.repeat([1, 2, 3], 120).astype(np.int16),
        "cadenceno": np.arange(count, dtype=np.int32),
        "sap_quality": np.zeros(count, dtype=np.int32),
        "flatten_mask": np.zeros(count, dtype=np.int32),
    }
    flux = 1000 + np.sin(np.arange(count))
    flux[20] = 800
    for name in ("psf_flat_kbonus", "psf_stitched", "sap_stitched"):
        data[name + "_flux"] = flux.copy()
        data[name + "_flux_err"] = np.full(count, 0.3)
    diagnostic = {
        key: data[key]
        for key in ("source_id", "time_bkjd", "quarter", "cadenceno", "sap_quality")
    }
    for name in ("psf", "sap", "psf_flux_nova"):
        prefix = name if name.endswith("nova") else name + "_flux"
        diagnostic[prefix] = flux.copy()
        diagnostic[
            prefix.replace("flux_nova", "flux_err_nova")
            if name.endswith("nova")
            else name + "_flux_err"
        ] = np.full(count, 0.3)
    source = {"source_id": [123], "origin_tpf_kic": [45], "raw_fits_sha256": ["a" * 64]}
    for name, columns in (("source", source), ("cadence", data), ("qdiag", diagnostic)):
        pq.write_table(pa.table(columns), folder / f"{name}.parquet")
    metadata, raw = lc2.read_source(folder, "123")
    products = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name in ("source", "cadence", "qdiag")
        for path in [folder / f"{name}.parquet"]
    }
    manifest = {
        "schema": 3,
        "protocol": lc2.LC2_MANIFEST,
        "role": "development",
        "measurement_scope": "OPENED_NULL_BASE",
        "lc2_products": {relative: products},
        "sources": {
            "123": {"lc2_folder": relative, "metadata": lc2.finite_metadata(metadata)}
        },
    }
    path = tmp_path / "base.json"
    save(path, manifest)
    return tmp_path, folder, path, manifest, raw


def test_bound_lc2_reads_the_original_arrays_and_retains_source_identity(bound_lc2):
    root, _, path, manifest, original = bound_lc2
    metadata, raw = lc2.load_source(root, "123", path)
    assert metadata == {
        **manifest["sources"]["123"]["metadata"],
        "input_manifest": str(path),
    }
    assert set(raw) == set(original)
    for key in raw:
        assert raw[key].dtype == original[key].dtype
        assert raw[key].tobytes() == original[key].tobytes()


@pytest.mark.parametrize("name", ["source", "cadence", "qdiag"])
def test_bound_lc2_refuses_changed_products_after_a_warm_read(bound_lc2, name):
    root, folder, path, _, _ = bound_lc2
    lc2.load_source(root, "123", path)
    product = folder / f"{name}.parquet"
    data = bytearray(product.read_bytes())
    data[len(data) // 2] ^= 1
    product.write_bytes(data)
    with pytest.raises(ValueError, match="Bound LC2 product changed"):
        lc2.load_source(root, "123", path)


def test_bound_lc2_refuses_changed_parent_metadata(bound_lc2):
    root, _, path, manifest, _ = bound_lc2
    manifest["sources"]["123"]["metadata"]["origin_tpf_kic"] = 99
    save(path, manifest)
    with pytest.raises(ValueError, match="source metadata changed"):
        lc2.load_source(root, "123", path)


@pytest.mark.parametrize("variant", ["inverted", "quarter_scrambled"])
def test_bound_lc2_and_npz_nulls_have_identical_prepared_photometry(
    bound_lc2, monkeypatch, variant
):
    root, _, base, manifest, raw = bound_lc2
    monkeypatch.setattr(signal_contracts, "ROOT", root)
    monkeypatch.setattr(signal_contracts, "dump_path", Path)
    arrays = root / "base.npz"
    np.savez_compressed(arrays, **raw)
    encoded = root / "encoded.json"
    save(
        encoded,
        {
            "schema": 1,
            "role": "development",
            "measurement_scope": "OPENED_NULL_BASE",
            "sources": {
                "123": {
                    "arrays": arrays.name,
                    "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                    "metadata": manifest["sources"]["123"]["metadata"],
                }
            },
        },
    )
    prepared = []
    for name, original in (("lc2", base), ("npz", encoded)):
        out = root / name
        signal_contracts.construct(
            SimpleNamespace(
                input_manifest=original,
                variant=variant,
                seed=20261002,
                exclusions=None,
                disposition_policy=None,
                out=out,
            )
        )
        _, transformed = lc2.load_source(root, "123", out / "input-manifest.json")
        prepared.append(run.prepare_records(transformed, relative_input=True))
    assert prepared[0].tobytes() == prepared[1].tobytes()


def test_sealed_lc2_nulls_require_scope_before_source_arrays(bound_lc2, monkeypatch):
    root, _, path, manifest, _ = bound_lc2
    manifest["role"] = "evaluation"
    save(path, manifest)
    monkeypatch.setattr(signal_contracts, "ROOT", root)
    monkeypatch.setattr(
        signal_contracts,
        "load_source",
        lambda *args: pytest.fail("Sealed arrays were opened"),
    )
    with pytest.raises(ValueError, match="frozen scoped evaluation policy"):
        signal_contracts.construct(
            SimpleNamespace(
                input_manifest=path,
                variant="inverted",
                seed=20261002,
                disposition_policy=None,
                out=root / "sealed",
                exclusions=None,
            )
        )


def test_real_source_vetting_binds_all_original_lc2_products(bound_lc2, monkeypatch):
    root, folder, path, _, _ = bound_lc2
    config = root / "vetting.json"
    save(config, {"input_manifest": str(path)})
    monkeypatch.setattr(vetting, "ROOT", root)
    monkeypatch.setattr(vetting, "CONFIG", config)
    hashes = vetting.input_hashes(["123"])
    assert set(hashes) == {str(path), *(str(p) for p in folder.glob("*.parquet"))}
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in hashes.items()
    )


def test_manifest_is_read_once_and_changed_bytes_invalidate_the_cache(
    bound_lc2, monkeypatch
):
    root, _, path, manifest, _ = bound_lc2
    reads = []
    original = Path.read_text

    def counted(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    lc2.load_source(root, "123", path)
    lc2.load_source(root, "123", path)
    assert reads == [path]
    manifest["sources"]["123"]["metadata"]["origin_tpf_kic"] = 99
    save(path, manifest)
    with pytest.raises(ValueError, match="source metadata changed"):
        lc2.load_source(root, "123", path)
    assert reads == [path, path]


@pytest.mark.parametrize("variant", ["inverted", "quarter_scrambled"])
def test_shard_scheduling_preserves_frozen_order_and_every_null_array(
    bound_lc2, monkeypatch, variant
):
    root, folder, base, manifest, original = bound_lc2
    second_folder = folder.with_name("shard-0001")
    second_folder.mkdir()
    products = {}
    for name in ("source", "cadence", "qdiag"):
        table = pq.read_table(folder / f"{name}.parquet")
        column = table.schema.get_field_index("source_id")
        table = table.set_column(
            column, "source_id", pa.array([124] * len(table), type=pa.int64())
        )
        path = second_folder / f"{name}.parquet"
        pq.write_table(table, path)
        products[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    metadata, second = lc2.read_source(second_folder, "124")
    relative = str(second_folder.relative_to(root))
    manifest["lc2_products"][relative] = products
    manifest["sources"] = {
        "124": {"lc2_folder": relative, "metadata": lc2.finite_metadata(metadata)},
        **manifest["sources"],
    }
    save(base, manifest)
    encoded = {**manifest, "schema": 1, "sources": {}}
    for sid, raw in (("124", second), ("123", original)):
        arrays = root / f"{sid}.npz"
        np.savez_compressed(arrays, **raw)
        encoded["sources"][sid] = {
            "arrays": arrays.name,
            "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
            "metadata": manifest["sources"][sid]["metadata"],
        }
    npz_base = root / "npz-base.json"
    save(npz_base, encoded)
    monkeypatch.setattr(signal_contracts, "ROOT", root)
    monkeypatch.setattr(signal_contracts, "dump_path", Path)
    visits = []
    load = signal_contracts.load_source

    def tracked(*args):
        visits.append(args[1])
        return load(*args)

    monkeypatch.setattr(signal_contracts, "load_source", tracked)
    outputs = []
    for name, path in (("grouped", base), ("ordered", npz_base)):
        out = root / name
        signal_contracts.construct(
            SimpleNamespace(
                input_manifest=path,
                variant=variant,
                seed=20261002,
                exclusions=None,
                disposition_policy=None,
                out=out,
            )
        )
        output = json.loads((out / "input-manifest.json").read_text())
        assert list(output["sources"]) == ["124", "123"]
        assert [r["source_id"] for r in output["construction_ledger"]] == ["124", "123"]
        outputs.append(out / "input-manifest.json")
    assert visits == ["123", "124", "124", "123"]
    for sid in ("124", "123"):
        left = lc2.load_source(root, sid, outputs[0])[1]
        right = lc2.load_source(root, sid, outputs[1])[1]
        assert set(left) == set(right)
        for key in left:
            assert left[key].dtype == right[key].dtype
            assert left[key].tobytes() == right[key].tobytes()
