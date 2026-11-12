"""Bind an authenticated KBonus scene to upstream TRICERATOPS public arrays."""

from pathlib import Path

import numpy as np
import pandas as pd
from triceratops.triceratops import target

from validator.records import require


class KeplerScene(target):
    """Skip archive target resolution, which can select a different TPF owner."""

    def __init__(self, scene_csv, geometry_npz, background, source_id, response):
        require(
            response in ("upstream_gaussian", "measured_kepler", "calibrated_kepler"),
            "Declare aperture response and review its applicability",
        )
        self.response = response
        stars = pd.read_csv(scene_csv, dtype={"ID": str})
        require(len(stars) > 0, "Empty scene")
        require(
            str(stars.iloc[0]["ID"]) == source_id,
            "First scene row must be the authenticated source",
        )
        require(stars["ID"].is_unique, "Duplicate scene source identity")
        require(
            stars["ID"].str.fullmatch(r"[0-9]+").all(),
            "Use exact numeric Gaia/TIC identities",
        )
        require(
            np.isfinite(stars["Tmag"]).all(),
            "Provide observed/justified upstream-band magnitudes for all sources",
        )
        with np.load(geometry_npz, allow_pickle=False) as geometry:
            require(
                geometry["source_id"].astype(str).tolist() == stars["ID"].tolist(),
                "Geometry source order differs",
            )
            self.sectors = geometry["quarter"].copy()
            require(
                len(self.sectors) > 0
                and len(np.unique(self.sectors)) == len(self.sectors),
                "Missing or duplicate quarters",
            )
            self.pix_coords = [geometry[f"xy_q{q}"].copy() for q in self.sectors]
            self.apertures = [geometry[f"aperture_q{q}"].copy() for q in self.sectors]
            self.fractions = None
            if response != "upstream_gaussian":
                self.fractions = np.stack(
                    [geometry[f"flux_fraction_q{q}"].copy() for q in self.sectors]
                )
                require(
                    self.fractions.shape == (len(self.sectors), len(stars)),
                    "Kepler fraction source/quarter census differs",
                )
                require(
                    np.isfinite(self.fractions).all() and np.all(self.fractions >= 0),
                    "Invalid Kepler aperture fractions",
                )
                require(
                    np.allclose(self.fractions.sum(axis=1), 1, rtol=0, atol=1e-10),
                    "Supply normalized complete source fractions",
                )
        for xy, ap in zip(self.pix_coords, self.apertures, strict=True):
            require(
                xy.shape == (len(stars), 2) and np.isfinite(xy).all(),
                "Invalid source-quarter coordinates",
            )
            require(
                ap.ndim == 2
                and ap.shape[1] == 2
                and len(ap) > 0
                and np.isfinite(ap).all(),
                "Actual quarter aperture required",
            )
        require(
            Path(background).is_file() and Path(background).stat().st_size > 0,
            "Missing field population; never skip background families",
        )
        self.ID, self.mission, self.stars = int(source_id), "Kepler", stars
        self.trilegal_fname, self.trilegal_url = str(Path(background).resolve()), None

    def prepare_depths(self, depth, dilution_corrected, joint=None):
        self.joint_likelihood = joint
        if joint is None:
            self.calc_depths(
                tdepth=depth,
                all_ap_pixels=self.apertures,
                dilution_corrected=dilution_corrected,
            )
        else:
            require(
                self.fractions is not None,
                "Joint likelihood requires quarter fractions",
            )
        if self.fractions is not None:
            fractions = self.fractions.mean(axis=0)
            self.stars["fluxratio"] = fractions
            aperture_depth = depth * fractions[0] if dilution_corrected else depth
            if joint is not None:
                # Conservative capability bound; the likelihood uses every response.
                fractions = joint.response.max(axis=0)
                aperture_depth = depth
            intrinsic = np.divide(
                aperture_depth,
                fractions,
                out=np.full_like(fractions, np.inf),
                where=fractions > 0,
            )
            self.stars["tdepth"] = np.where(intrinsic <= 1, intrinsic, 0)
        possible = self.stars[self.stars["tdepth"] > 0]
        require(
            len(possible) > 0 and str(possible.iloc[0]["ID"]) == str(self.ID),
            "Target cannot produce supplied depth",
        )
        for name in ("mass", "rad", "Teff"):
            require(
                np.isfinite(possible[name]).all() and np.all(possible[name] > 0),
                f"Missing {name} for a potentially signal-producing source",
            )
        require(
            np.isfinite(possible.iloc[0]["plx"]) and possible.iloc[0]["plx"] > 0,
            "Declare target parallax for companion prior",
        )

    def plot_fits(self, *args, **kwargs):
        require(
            self.joint_likelihood is None,
            "Upstream plot_fits uses mean dilution; inspect quarter-resolved model curves",
        )
        return super().plot_fits(*args, **kwargs)
