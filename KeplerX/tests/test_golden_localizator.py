"""Golden: the accelerated Localizator against the native one."""

from types import SimpleNamespace

import numpy as np
from astropy.table import Table
from conftest import kic12644822

from common import receipt, write
from localizator import gaia_cone_search, gaia_field
from localizator.program import run as localize


def test_accelerated_localizer_gives_the_native_centre(tmp_path, config, monkeypatch):
    target = kic12644822(tmp_path, monkeypatch)
    clock = target.record["clock"]
    vetting = tmp_path / "vetting"
    write(vetting / "vetting.json", {
        "source_id": target.record["packet"]["source_id"],
        "detections": receipt(target.detections / "detections.json"),
        "results": [{"signal_id": clock["signal_id"], "metrics": {k: clock[k] for k in ("period", "t0", "tdur")}}],
    })  # fmt: skip
    out = tmp_path / "localization"
    out.mkdir()
    result = localize(
        SimpleNamespace(prepared=target.prepared, detections=target.detections,
                        vetting=vetting, request=target.request, out=out),
        config,
    )  # fmt: skip
    (row,) = result["results"]
    assert result["state"] == "MEASURED" and row["fit_success"]
    native = target.record["native_localization_20261027"]
    got, expected = row["metrics"], native["metrics"]
    for centre, sigma in (("centerra", "sigmara"), ("centerdec", "sigmadec")):
        assert abs(got[centre] - expected[centre]) < 1e-4 * expected[sigma]
        assert abs(got[sigma] / expected[sigma] - 1) < 1e-4
    assert row["degrees_of_freedom"] == native["degrees_of_freedom"]
    assert abs(row["chi_square"] / native["chi_square"] - 1) < 1e-4


def test_archive_cone_is_vizier_table(fixtures):
    """KIC 12644822's TPF cone."""
    votable = (fixtures / "gaia-dr3-12644822-esa.vot").read_bytes()
    expected = Table.read(
        fixtures / "gaia-dr3-12644822-vizier.ecsv", format="ascii.ecsv"
    )
    table = gaia_cone_search.table(gaia_field.votable_columns(votable))
    assert table.colnames == expected.colnames
    assert list(table["DR3Name"]) == list(expected["DR3Name"])
    for name in expected.colnames[1:]:
        assert str(table[name].unit) == str(expected[name].unit)
        mask = np.ma.getmaskarray(table[name])
        assert np.array_equal(mask, np.ma.getmaskarray(expected[name]))
        got = np.ma.filled(table[name], 0.0).astype(float)
        want = np.ma.filled(expected[name], 0.0).astype(float)
        assert np.array_equal(got.view(np.int64), want.view(np.int64))


def test_field_table_cone_is_the_archive_cone(fixtures, tmp_path):
    """A field table holding the archive's sources returns the archive's cone bit for bit."""
    from astropy.io import fits

    votable = (fixtures / "gaia-dr3-12644822-esa.vot").read_bytes()
    with fits.open(fixtures / "kic12644822-q1-target-pixels.fits.gz") as hdus:
        ra, dec = hdus[0].header["RA_OBJ"], hdus[0].header["DEC_OBJ"]
        shape = hdus[1].data["FLUX"].shape[1:]
    columns = gaia_field.votable_columns(votable)
    gaia_field.save(
        tmp_path / "field", columns, (ra - 1, ra + 1), (dec - 1, dec + 1), []
    )
    field = gaia_field.Field.load(tmp_path / "field")
    expected = gaia_cone_search.table(columns)
    table = gaia_cone_search.Cones(field)(ra, dec, shape, "kepler")
    assert len(table) == len(expected) == len(columns["source_id"])
    assert list(table["DR3Name"]) == list(expected["DR3Name"])
    for name in expected.colnames[1:]:
        mask = np.ma.getmaskarray(table[name])
        assert np.array_equal(mask, np.ma.getmaskarray(expected[name]))
        got = np.ma.filled(table[name], 0.0).astype(float)
        want = np.ma.filled(expected[name], 0.0).astype(float)
        assert np.array_equal(got.view(np.int64), want.view(np.int64))
