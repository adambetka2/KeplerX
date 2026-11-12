"""End-to-end: vetting, then pixel localization."""

import math
from types import SimpleNamespace

from conftest import kic12644822

from common import digest, read, write
from localizator import mast_pixels
from localizator.program import run as localize
from vetter.program import run as vet


def test_transit_light_is_centred_on_its_host(tmp_path, config, monkeypatch):
    target = kic12644822(tmp_path, monkeypatch)
    vetting = tmp_path / "vetting"
    vetting.mkdir()
    vetted = vet(
        SimpleNamespace(prepared=target.prepared, detections=target.detections, out=vetting),
        config,
    )  # fmt: skip
    (signal,) = vetted["results"]
    assert not signal["metric_errors"]
    assert abs(signal["metrics"]["period"] - target.record["clock"]["period"]) < 0.01
    out = tmp_path / "localization"
    out.mkdir()
    result = localize(
        SimpleNamespace(prepared=target.prepared, detections=target.detections,
                        vetting=vetting, request=target.request, out=out),
        config,
    )  # fmt: skip
    (row,) = result["results"]
    assert result["state"] == "MEASURED" and row["fit_success"]
    assert result["clock_basis"] == "Vetter refined period, epoch and duration"
    packet = target.record["packet"]
    dra = (row["metrics"]["centerra"] - packet["ra_deg"]) * math.cos(
        math.radians(packet["dec_deg"])
    )
    ddec = row["metrics"]["centerdec"] - packet["dec_deg"]
    assert math.hypot(dra, ddec) * 3600 < 1.0


def test_a_dismissed_signal_opens_no_pixels(tmp_path, config, monkeypatch):
    target = kic12644822(tmp_path, monkeypatch)
    vetting = tmp_path / "vetting"
    vetting.mkdir()
    vet(
        SimpleNamespace(prepared=target.prepared, detections=target.detections, out=vetting),
        config,
    )  # fmt: skip
    record = read(vetting / "vetting.json")
    for row in record["results"]:
        row["disposition"]["state"] = "DISMISSED"
    write(vetting / "vetting.json", record)
    out = tmp_path / "localization"
    out.mkdir()
    result = localize(
        SimpleNamespace(prepared=target.prepared, detections=target.detections,
                        vetting=vetting, request=target.request, out=out),
        config,
    )  # fmt: skip
    (row,) = result["results"]
    assert row["state"] == result["state"] == "NOT_RUN"
    assert not list(out.glob("quarter-*.npz"))


class Bucket:
    """MAST's public bucket for one parent, offline."""

    def __init__(self, files, refused):
        self.files, self.refused, self.fetched = files, refused, []

    def get(self, url, params=None, timeout=None):
        if params is not None:
            prefix = params["prefix"]
            listed = "".join(
                f"<Contents><Key>{key}</Key><Size>{len(data)}</Size></Contents>"
                for key, data in self.files.items()
                if key.startswith(prefix)
            )
            body = (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<IsTruncated>false</IsTruncated>{listed}</ListBucketResult>"
            ).encode()
        else:
            key = url.removeprefix(mast_pixels.BUCKET + "/")
            assert key not in self.refused, f"fetched {key}"
            self.fetched.append(key)
            body = self.files[key]
        return SimpleNamespace(content=body, raise_for_status=lambda: None)


def test_pixels_streamed_from_mast_localize_as_stored_ones(
    tmp_path, config, monkeypatch, fixtures
):
    """A request naming MAST lists the parent's folder."""
    target = kic12644822(tmp_path, monkeypatch)
    vetting = tmp_path / "vetting"
    vetting.mkdir()
    vet(
        SimpleNamespace(prepared=target.prepared, detections=target.detections, out=vetting),
        config,
    )  # fmt: skip
    stored = tmp_path / "stored"
    stored.mkdir()
    expected = localize(
        SimpleNamespace(prepared=target.prepared, detections=target.detections,
                        vetting=vetting, request=target.request, out=stored),
        config,
    )  # fmt: skip
    folder = "kepler/public/target_pixel_files/0126/012644822/"
    quarter_1 = folder + "kplr012644822-2009166043257_lpd-targ.fits.gz"
    refused = [
        folder + "kplr012644822-2009259160929_lpd-targ.fits.gz",
        folder + "kplr012644822-2009166043257_spd-targ.fits.gz",
        folder + "kplr012644822_tpf_lc_Q111111111111111111.tar",
    ]
    data = (fixtures / "kic12644822-q1-target-pixels.fits.gz").read_bytes()
    bucket = Bucket(
        {quarter_1: data} | {key: b"not needed" for key in refused}, refused
    )
    monkeypatch.setattr(mast_pixels, "session", lambda connections: bucket)
    request = read(target.request)
    del request["tpfs"]
    write(target.request, request | {"archive": "mast"})
    streamed = tmp_path / "streamed"
    streamed.mkdir()
    result = localize(
        SimpleNamespace(prepared=target.prepared, detections=target.detections,
                        vetting=vetting, request=target.request, out=streamed),
        config,
    )  # fmt: skip
    assert bucket.fetched == [quarter_1]
    assert result["pixels"] == [{
        "url": f"{mast_pixels.BUCKET}/{quarter_1}", "bytes": len(data), "quarter": 1,
        "sha256": digest(fixtures / "kic12644822-q1-target-pixels.fits.gz"),
    }]  # fmt: skip
    (row,) = result["results"]
    assert row["quarters_without_coverage"] == [2]
    row["quarters_without_coverage"] = []
    assert result["state"] == expected["state"] == "MEASURED"
    assert result["results"] == expected["results"]
