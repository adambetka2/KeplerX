"""A complete physical scene for a signed PSF_FLAT extraction response."""

from pathlib import Path

import numpy as np
import pandas as pd
from triceratops.triceratops import target

from post_boi.records import require


def prepared_scene(c, clock, packet):
    from post_boi.psf_response import load_response

    joint = load_response(c, clock, packet)
    scene = PSFScene(c["scene"], c["geometry"], c["background"], c["source_id"])
    scene.prepare_depths(
        clock["observed_depth_fraction"], clock["probability_noise_sigma"], joint
    )
    return scene, joint


class PSFScene(target):
    """Never interpret SAP masks or normalized fractions as PSF response."""

    def __init__(self, scene_csv, geometry_npz, background, source_id):
        stars = pd.read_csv(scene_csv, dtype={"ID": str})
        require(
            len(stars) > 0 and str(stars.iloc[0]["ID"]) == source_id,
            "First PSF scene row must be the authenticated source",
        )
        require(
            stars["ID"].is_unique and stars["ID"].str.fullmatch(r"[0-9]+").all(),
            "Supply unique exact physical source identities",
        )
        require(np.isfinite(stars["Tmag"]).all(), "Missing upstream-band magnitudes")
        with np.load(geometry_npz, allow_pickle=False) as geometry:
            require(
                geometry["source_id"].astype(str).tolist() == stars["ID"].tolist(),
                "Physical PSF geometry source order differs",
            )
            self.sectors = geometry["quarter"].copy()
            require(
                self.sectors.ndim == 1
                and len(self.sectors) > 0
                and len(np.unique(self.sectors)) == len(self.sectors),
                "Missing or duplicate physical quarters",
            )
            self.pix_coords = [geometry[f"xy_q{q}"].copy() for q in self.sectors]
            self.footprints = [geometry[f"footprint_q{q}"].copy() for q in self.sectors]
        for xy, footprint in zip(self.pix_coords, self.footprints, strict=True):
            require(
                xy.shape == (len(stars), 2) and np.isfinite(xy).all(),
                "Invalid physical source-quarter coordinates",
            )
            require(
                footprint.ndim == 2
                and footprint.shape[1] == 2
                and len(footprint) > 0
                and np.isfinite(footprint).all()
                and len(np.unique(footprint, axis=0)) == len(footprint),
                "Actual complete PSF extraction footprint required",
            )
        require(
            Path(background).is_file() and Path(background).stat().st_size > 0,
            "Missing upstream field population",
        )
        self.ID, self.mission, self.stars = int(source_id), "Kepler", stars
        self.trilegal_fname, self.trilegal_url = str(Path(background).resolve()), None

    def prepare_depths(self, depth, observed_sigma, joint):
        require(joint is not None, "PSF probability requires its signed likelihood")
        require(
            joint.source_ids.tolist() == self.stars["ID"].tolist()
            and np.isin(joint.quarters, self.sectors).all(),
            "PSF response scene/quarter census differs",
        )
        self.joint_likelihood = joint
        maximum = joint.response.max(axis=0)
        intrinsic = np.divide(
            depth, maximum, out=np.full_like(maximum, np.inf), where=maximum > 0
        )
        self.stars["tdepth"] = np.where(intrinsic <= 1, intrinsic, 0)
        capable = self.stars["tdepth"].to_numpy() > 0
        require(
            np.isfinite(joint.secondary_sigma[capable]).all()
            and (joint.secondary_sigma[capable] > 0).all(),
            "Missing reviewed secondary-screen noise for a capable physical source",
        )
        # Convert scalar screen noise here. The primary likelihood uses the signed response.
        self.stars["fluxratio"] = np.divide(
            observed_sigma,
            joint.secondary_sigma,
            out=np.zeros(len(self.stars)),
            where=np.isfinite(joint.secondary_sigma) & (joint.secondary_sigma > 0),
        )
        possible = self.stars[self.stars["tdepth"] > 0]
        require(
            len(possible) > 0 and str(possible.iloc[0]["ID"]) == str(self.ID),
            "Target cannot produce supplied depth under reviewed PSF response",
        )
        for name in ("mass", "rad", "Teff"):
            require(
                np.isfinite(possible[name]).all() and (possible[name] > 0).all(),
                f"Missing {name} for a potentially signal-producing source",
            )
        require(
            np.isfinite(possible.iloc[0]["plx"]) and possible.iloc[0]["plx"] > 0,
            "Declare target parallax for upstream companion prior",
        )

    def plot_fits(self, *args, **kwargs):
        raise ValueError("Inspect signed-response model curves in observed Flat units")
