"""Native targets consume their bound pixels even with pixel-free siblings."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/photometry"),
    str(ROOT / "pipeline/vetting"),
]

import vetting
import vetting_pixels as pixels

from pipeline.deterministic import native_prospective_controls as native
from pipeline.deterministic.test_native_prospective_controls import (
    bind,
    enable,
)
from pipeline.deterministic.test_native_prospective_controls import (
    case as case,
)


@pytest.fixture
def retrieval(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    config.write_text(json.dumps({"input_manifest": str(manifest_path)}))
    manifest = {
        "schema": 1,
        "measurement_scope": "OPENED_NATIVE_MECHANISM_RECORD_ONLY",
        "native_control_binding": {"path": "fixture-binding.json"},
        "sources": {
            "target": {"pixels": {"1": {"path": "bound.fits", "sha256": "original"}}},
            "sibling": {"arrays": "original-sibling.npz"},
        },
    }
    metadata = {"target": {"origin_tpf_kic": "42"}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/target.json").write_text(
        json.dumps({"pixel_jobs": [{"quarter": 1}, {"quarter": 2}]})
    )
    monkeypatch.setattr(vetting, "CONFIG", config)
    monkeypatch.setattr(vetting, "retrieve_null_pixels", lambda: False)
    monkeypatch.setattr(pixels, "CONFIG", config)
    monkeypatch.setattr(pixels, "RUN", tmp_path)
    monkeypatch.setattr(pixels, "PIXELS", tmp_path / "cache")
    monkeypatch.setattr(pixels, "input_manifest", lambda *_: (manifest_path, manifest))
    return tmp_path, manifest, metadata


def forbidden(*_args, **_kwargs):
    pytest.fail("Native bound pixels accessed nominal inventory or transport")


def test_pixel_free_sibling_does_not_dispatch_native_target_to_archive(
    retrieval, monkeypatch
):
    folder, manifest, _ = retrieval
    calls = []
    monkeypatch.setattr(pixels, "indexed_pixels", forbidden)
    monkeypatch.setattr(pixels, "retrieve_parent", forbidden)
    monkeypatch.setattr(
        native, "runtime_source", lambda value, sid: calls.append((value, sid))
    )
    monkeypatch.setattr(pixels, "digest", lambda path: "original")
    pixels.retrieve()
    assert calls == [(manifest, "target")]
    result = pixels.read(folder / "pixel-manifests/42.json")["quarters"]
    assert result["1"]["path"] == str(folder / "bound.fits")
    assert result["1"]["sha256"] == "original"
    assert result["2"] == {"status": "PIXELS_NOT_ALTERED_IN_THIS_CONTROL"}
    assert (
        pixels.read(folder / "pixel-retrieval.json")["status"]
        == "EXPLICIT_EXPERIMENT_INPUTS"
    )


def test_all_native_target_guards_precede_first_pixel_payload(retrieval, monkeypatch):
    folder, manifest, metadata = retrieval
    metadata["protected"] = {"origin_tpf_kic": "43"}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    manifest["sources"]["protected"] = {"pixels": {"1": {"path": "sealed.fits"}}}

    def admit(_manifest, sid):
        if sid == "protected":
            raise ValueError("Protected full component")

    monkeypatch.setattr(native, "runtime_source", admit)
    monkeypatch.setattr(pixels, "digest", forbidden)
    monkeypatch.setattr(pixels, "indexed_pixels", forbidden)
    with pytest.raises(ValueError, match="Protected full component"):
        pixels.retrieve()
    assert not (folder / "pixel-retrieval.json").exists()


@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_missing_or_changed_native_pixels_never_fall_back(
    retrieval, monkeypatch, failure
):
    folder, manifest, _ = retrieval
    monkeypatch.setattr(native, "runtime_source", lambda *_: None)
    monkeypatch.setattr(pixels, "indexed_pixels", forbidden)
    if failure == "missing":
        manifest["sources"]["target"].pop("pixels")
        monkeypatch.setattr(pixels, "digest", forbidden)
        expected = "no bound target pixels"
    else:
        monkeypatch.setattr(pixels, "digest", lambda _: "changed")
        expected = "Experimental pixel input changed"
    with pytest.raises(ValueError, match=expected):
        pixels.retrieve()
    assert not (folder / "pixel-retrieval.json").exists()


def test_actual_native_runtime_contract_refuses_before_pixel_payload(case, monkeypatch):
    row = enable(case)
    bind(case)
    manifest = native.read_json(case.out / "input-manifest.json")
    Path(row["recipe"]["path"]).write_text("{}")
    monkeypatch.setattr(
        pixels,
        "input_manifest",
        lambda *_: (case.out / "input-manifest.json", manifest),
    )
    monkeypatch.setattr(pixels, "digest", forbidden)
    with pytest.raises(ValueError, match="changed|differs"):
        pixels.retrieve_experiment(
            case.out / "input-manifest.json",
            {row["source_id"]: {"origin_tpf_kic": "10000"}},
        )
