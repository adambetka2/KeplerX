from post_boi.ephemeris import screen

CLOCK = {
    "period_days": 4.0,
    "epoch_bkjd": 200.0,
    "duration_days": 0.1,
    "ra": 290.0,
    "dec": 44.0,
}


def tce(period, epoch, ra=290.5, dec=44.0):
    return {
        "kepid": 1,
        "tce_plnt_num": 1,
        "tce_period": period,
        "tce_time0bk": epoch,
        "ra": ra,
        "dec": dec,
    }


def test_low_order_matches_include_a_shorter_catalogue_period():
    rows = [
        tce(4.0, 200.0 + 40 * 4.0),
        tce(2.0, 200.0 + 2.0),
        tce(2.0, 201.0),
        tce(8.0, 204.0),
        tce(4.0, 200.0, 290.0, 44.0),
    ]
    r = screen(CLOCK, {"dr25_tce": rows, "koi": []})
    assert [
        (h["ratio"], h["row"]["tce_time0bk"]) for h in r["low_order_other_star"]
    ] == [("1/1", 360.0), ("2/1", 202.0), ("1/2", 204.0)]
    assert [h["same_star"] for h in r["matches"]] == [False, False, False, True]


def test_drifting_period_does_not_match():
    drift = 0.1 * 4.0 / 1460 * 1.5
    r = screen(CLOCK, {"dr25_tce": [tce(4.0 + drift, 200.0)], "koi": []})
    assert r["matches"] == [] and r["expected_chance"] > 0
