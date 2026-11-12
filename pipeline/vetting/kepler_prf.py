"""Kepler PRFs with exact-grid reuse; adapted from lkprf."""

from collections import OrderedDict

import numpy as np
from lkprf import KeplerPRF as ArchiveKeplerPRF
from scipy.interpolate import RectBivariateSpline


class KeplerPRF(ArchiveKeplerPRF):
    def __init__(self, channel, cache_dir):
        self._spline_cache = OrderedDict()
        super().__init__(channel=channel, cache_dir=cache_dir)

    def _prepare_supersamp_prf(self, targets, shape):
        row, column = self._unpack_targets(targets)
        row, column = np.atleast_1d(row), np.atleast_1d(column)
        row = np.min([np.max([row, row**0 * 20], axis=0), row**0 * 1044], axis=0).mean()
        column = np.min(
            [np.max([column, column**0 * 12], axis=0), column**0 * 1112], axis=0
        ).mean()
        ref_column = column + 0.5 * shape[1]
        ref_row = row + 0.5 * shape[0]
        # Keep the pinned lkprf 1.1.1 grid arithmetic and float32 rounding.
        grid = np.zeros(self.PRFdata.shape[1:], dtype="float32")
        weights = [
            np.sqrt(
                (ref_column - self.crval1p[i]) ** 2 + (ref_row - self.crval2p[i]) ** 2
            )
            for i in range(self.PRFdata.shape[0])
        ]
        for i in np.argpartition(weights, 3)[:3]:
            if weights[i] < 1e-6:
                weights[i] = 1e-6
            grid += self.PRFdata[i] / weights[i]
        grid /= np.nansum(grid) * self.cdelt1p[0] * self.cdelt2p[0]
        key = grid.tobytes()
        if key in self._spline_cache:
            self.interpolate = self._spline_cache.pop(key)
        else:
            self.interpolate = RectBivariateSpline(self.PRFrow, self.PRFcol, grid)
        self._spline_cache[key] = self.interpolate
        if len(self._spline_cache) > 4:
            self._spline_cache.popitem(last=False)
