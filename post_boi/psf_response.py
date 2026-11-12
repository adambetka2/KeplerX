"""Bind a reviewed signed PSF response to the original observed photons.

Coefficients map intrinsic source-relative flux into prepared observed Flat
flux, including source/baseline conversion once. They are not aperture shares.
The diagonal approximation and scalar secondary-screen noise require separate
method and candidate evidence; this module does not establish their adequacy.
"""

from pathlib import Path

import numpy as np

from post_boi.quarters import QuarterLikelihood
from post_boi.records import digest, read, require


def evidence_files(review):
    require(review["accepted"] is True, "PSF response review remains unresolved")
    require(bool(review["evidence"]), "PSF review needs scientific evidence")
    paths = []
    for item in review["evidence"]:
        require(digest(item["path"]) == item["sha256"], "PSF evidence changed")
        paths.append(Path(item["path"]))
    return paths


def reviewed_files(c, clock):
    """Validate inputs before freezing; retain all instrument review evidence."""
    declared = c["psf_response"]
    require(
        c["flux_convention"] == "psf_flat"
        and c["aperture_response"] == "signed_psf_flat",
        "Signed PSF response requires its explicit flux convention",
    )
    review = read(declared["review"])
    paths = evidence_files(review)
    require(
        (review["source_id"], review["signal_id"]) == (c["source_id"], c["signal_id"]),
        "PSF response review source/signal differs",
    )
    current = {k: clock[k] for k in ("period_days", "epoch_bkjd", "duration_days")}
    require(current in review["clocks"], "PSF response clock differs")
    inputs = {
        "data": declared["data"],
        "input_review": declared["input_review"],
        "photometry": c["photometry"],
        "prepared": Path(c["prepared"]) / "detrended.npz",
        "scene": c["scene"],
        "geometry": c["geometry"],
        "noise": c["probability"]["noise"],
    }
    for key, path in inputs.items():
        require(
            str(Path(path).resolve()) == review["inputs"][key]["path"]
            and digest(path) == review["inputs"][key]["sha256"],
            f"PSF response {key} binding differs",
        )
        paths.append(Path(path))
    for key in (
        "normalization_basis",
        "transfer_basis",
        "scene_basis",
        "secondary_noise_basis",
        "uncertainty_basis",
    ):
        require(bool(review[key]), f"Missing PSF {key}")
    method = review["method_review"]
    require(digest(method["path"]) == method["sha256"], "PSF method review changed")
    paths += [Path(method["path"]), *evidence_files(read(method["path"]))]
    ingestion = read(declared["input_review"])
    require(
        ingestion["view"] == "psf_flat"
        and ingestion["source_id"] == c["source_id"]
        and ingestion["dilution_correction_applied"] is False
        and ingestion["photons"]["sha256"] == digest(c["photometry"])
        and ingestion["exposure_days"] == c["exposure_days"],
        "Authenticate delivered Flat photons without a dilution correction",
    )
    for key in ("original", "reader"):
        item = ingestion[key]
        require(digest(item["path"]) == item["sha256"], "PSF input evidence changed")
        paths.append(Path(item["path"]))
    return [Path(declared["review"]), *paths]


def load_response(c, clock, packet):
    """Select exact archived rows; never infer response from a SAP aperture."""
    reviewed_files(c, clock)
    with (
        np.load(c["psf_response"]["data"], allow_pickle=False) as response,
        np.load(c["photometry"], allow_pickle=False) as native,
    ):
        index = response["original_index"]
        require(
            index.ndim == 1
            and len(index) > 0
            and index.dtype.kind in "iu"
            and len(np.unique(index)) == len(index)
            and (index >= 0).all()
            and (index < len(native["time_bkjd"])).all(),
            "Invalid PSF response photon census",
        )
        require(
            str(native["source_id"].item()) == c["source_id"]
            and np.array_equal(response["time_bkjd"], native["time_bkjd"][index])
            and np.array_equal(response["quarter"], native["quarter"][index]),
            "PSF response original times/quarters differ",
        )
        order = np.argsort(index)
        wanted = np.asarray(packet["original_index"])
        rows = np.searchsorted(index[order], wanted)
        require((rows < len(index)).all(), "Prepared photons absent from PSF response")
        rows = order[rows]
        require(
            np.array_equal(index[rows], wanted)
            and np.array_equal(response["quarter"][rows], packet["quarter"])
            and np.array_equal(response["time_bkjd"][rows], packet["time_bkjd"]),
            "Prepared photon indices, times or quarters differ from PSF response",
        )
        joint = PSFFlatLikelihood(
            packet, response["source_id"], wanted, response["coefficient"][rows]
        )
        joint.secondary_sigma = response["secondary_sigma"].astype(float)
    require(
        joint.secondary_sigma.shape == joint.source_ids.shape
        and (
            np.isnan(joint.secondary_sigma)
            | (np.isfinite(joint.secondary_sigma) & (joint.secondary_sigma > 0))
        ).all(),
        "Invalid intrinsic-source secondary-screen noise; missing values stay NaN",
    )
    return joint


class PSFFlatLikelihood(QuarterLikelihood):
    """Reuse the upstream source callback with finite signed coefficients."""

    def __init__(self, packet, source_ids, original_index, response):
        ids = np.asarray(source_ids)
        require(
            ids.ndim == 1
            and ids.dtype.kind in "US"
            and len(ids) > 0
            and len(np.unique(ids)) == len(ids),
            "Supply unique exact source strings",
        )
        self.source_ids = ids.astype(str)
        self.time = np.asarray(packet["time"], dtype=float)
        self.flux = np.asarray(packet["flux"], dtype=float)
        self.sigma = np.asarray(packet["sigma"], dtype=float)
        self.bins = np.asarray(packet["bin_index"])
        quarter = np.asarray(packet["quarter"])
        applied = np.asarray(packet["applied_fraction"], dtype=float)
        index = np.asarray(original_index)
        require(
            self.time.ndim == 1
            and 0 < len(self.time) <= 200
            and np.isfinite(self.time).all(),
            "Invalid common phase grid",
        )
        require(
            self.flux.ndim == 1
            and len(self.flux) > 0
            and all(
                a.shape == self.flux.shape
                for a in (self.sigma, self.bins, quarter, applied, index)
            ),
            "Photon arrays differ",
        )
        require(
            index.dtype.kind in "iu"
            and (index >= 0).all()
            and len(np.unique(index)) == len(index)
            and np.array_equal(index, packet["original_index"]),
            "Response is not bound to exact original photon indices",
        )
        require(
            self.bins.dtype.kind in "iu"
            and (self.bins >= 0).all()
            and (self.bins < len(self.time)).all(),
            "Invalid photon phase bins",
        )
        require(
            np.isfinite(self.flux).all()
            and np.isfinite(self.sigma).all()
            and (self.sigma > 0).all()
            and np.isfinite(quarter).all(),
            "Invalid observed flux or reviewed noise",
        )
        require(
            np.all(applied == 1),
            "PSF response already includes its conversion; no second dilution",
        )
        self.response = np.asarray(response, dtype=float).copy()
        require(
            self.response.shape == (len(self.flux), len(ids))
            and np.isfinite(self.response).all(),
            "Supply complete finite signed photon/source response",
        )
        self.quarters = np.unique(quarter)
        self.log_normalization = float(
            -0.5 * len(self.flux) * np.log(2 * np.pi) - np.log(self.sigma).sum()
        )
