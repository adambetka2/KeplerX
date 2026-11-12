"""Authenticate a KBonus array aperture against original Kepler pixel data."""

import numpy as np

from validator.records import require


def replay(source, mask, frame):
    """Return known background conventions reproducing all accepted SAP values."""
    lookup = {int(c): i for i, c in enumerate(frame["CADENCENO"])}
    require(
        all(int(c) in lookup for c in source["CADENCENO"]),
        "Source cadence absent from proposed parent",
    )
    index = np.array([lookup[int(c)] for c in source["CADENCENO"]])
    quality = (
        source["SAP_QUALITY"] if "SAP_QUALITY" in source.names else source["QUALITY"]
    )
    observed = np.asarray(source["SAP_FLUX"], float)
    valid = (quality == 0) & np.isfinite(observed)
    require(valid.any(), "No accepted source SAP values for pixel authentication")
    pixels = np.asarray(frame["FLUX"][index][:, mask], float)
    background = np.asarray(frame["FLUX_BKG"][index][:, mask], float)
    custom = np.asarray(source["SAP_BKG"], float)
    require(
        all(np.isfinite(a[valid]).all() for a in (pixels, background, custom)),
        "Accepted source pixels/background have missing values",
    )
    # Bound stored float32 additions/subtractions and final output rounding.
    operations = 2 * int(mask.sum()) + 4
    epsilon = np.finfo(np.float32).eps
    gamma = operations * epsilon / (1 - operations * epsilon)
    bound = gamma * (
        abs(pixels).sum(axis=1)
        + abs(background).sum(axis=1)
        + abs(custom)
        + abs(observed)
    )
    choices = {
        "FLUX": pixels.sum(axis=1),
        "FLUX_PLUS_FLUX_BKG_MINUS_SAP_BKG": (pixels + background).sum(axis=1) - custom,
    }
    checks = []
    for name, prediction in choices.items():
        error = abs(prediction - observed)[valid]
        checks.append(
            dict(
                convention=name,
                reproduces=bool(np.all(error <= bound[valid])),
                maximum_error_e_per_s=float(error.max()),
                maximum_float32_bound_e_per_s=float(bound[valid].max()),
            )
        )
    require(
        any(c["reproduces"] for c in checks),
        "Parent pixels do not reproduce source SAP",
    )
    return dict(accepted_source_cadences=int(valid.sum()), checks=checks)


def physical_aperture(light_curve, aperture, parent):
    """Resolve the physical mask through observed pixel replay, preserving headers."""
    header = light_curve.header
    require(
        header["QUARTER"] == parent[0].header["QUARTER"]
        and header["CHANNEL"] == parent[0].header["CHANNEL"],
        "Parent quarter/channel differs",
    )
    require(np.isin(aperture.data, [0, 1]).all(), "Expected binary source aperture")
    mask = aperture.data.astype(bool)
    require(
        mask.any() and mask.shape == parent[2].data.shape, "Parent/source stamp differs"
    )
    require(
        header["NPIXSAP"] == int(mask.sum()),
        "Saved mask does not contain the declared full aperture",
    )
    review = replay(light_curve.data, mask, parent[1].data)
    col, row = (parent[2].header[k] for k in ("CRVAL1P", "CRVAL2P"))
    declared = [header["TPF_COL"], header["TPF_ROW"]]
    y, x = np.nonzero(mask)
    review.update(
        source_header_origin=declared,
        replay_authenticated_origin=[col, row],
        metadata_conflict=declared != [col, row],
        scope="Physical mapping authenticated by cadencewise SAP replay; original metadata retained. No dilution or planet admission implied.",
    )
    return np.column_stack((x + col, y + row)), review
