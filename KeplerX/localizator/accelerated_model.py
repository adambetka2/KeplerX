"""Localizator's model with faster mathematics: the same estimates, not the same bits."""

import copy
import gzip
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lapack


class NotEstimable(np.linalg.LinAlgError):
    """A quarter's transit column is a combination of its nuisance columns."""


# Relative Schur pivot below which a column is a combination of the others.
PIVOT_TOLERANCE = 1e-10
# Widest band (columns) the band columns' normal matrix may have.
MAX_BAND = 64
MIN_PRF_WEIGHT = 1e-6  # lkprf's
CHANNELS = 8  # channel bases kept in memory; a Kepler star visits four
_BASES = OrderedDict()
_CHANNEL_PRFS = {}


def independent_columns(matrix, tol=PIVOT_TOLERANCE):
    """Sorted indices of columns spanning the same space as all of them."""
    gram = sparse.csr_matrix(matrix)
    gram = (gram.T @ gram).toarray()
    scale = np.sqrt(np.diag(gram))
    live = np.flatnonzero(scale > 0)
    unit = gram[np.ix_(live, live)] / np.outer(scale[live], scale[live])
    _, pivots, rank, info = lapack.dpstrf(unit, tol=tol)
    if info < 0:
        raise np.linalg.LinAlgError(f"Pivoted Cholesky failed: {info}")
    return np.sort(live[pivots[:rank] - 1])


def normal_matrices(design, weights):
    """X^T diag(w_p) X for every pixel p, from the rows of a sparse X."""
    design = sparse.csr_matrix(design)
    n, m = design.shape
    counts = np.diff(design.indptr)
    width = int(counts.max())
    rows = np.repeat(np.arange(n), counts)
    slot = np.arange(design.nnz) - np.repeat(design.indptr[:-1], counts)
    index = np.zeros((n, width), dtype=np.int64)
    value = np.zeros((n, width))
    index[rows, slot] = design.indices
    value[rows, slot] = design.data
    pairs = (index[:, :, None] * m + index[:, None, :]).reshape(n, -1)
    products = (value[:, :, None] * value[:, None, :]).reshape(n, -1)
    outer = sparse.csr_matrix(
        (products.ravel(), (np.repeat(np.arange(n), width * width), pairs.ravel())),
        shape=(n, m * m),
    )
    return np.asarray(outer.T @ weights).T.reshape(-1, m, m)


def nuisance_fit(columns, values):
    """Unweighted least-squares fit of independent columns to each column of `values`."""
    from scipy.linalg import cho_factor, cho_solve

    gram = (columns.T @ columns).toarray()
    scale = np.sqrt(np.diag(gram))
    unit = cho_factor(gram / np.outer(scale, scale))
    coefficients = cho_solve(unit, np.asarray(columns.T @ values) / scale[:, None])
    return np.asarray(columns @ (coefficients / scale[:, None]))


def transit_depths(design, transits, pix, pix_err, tol=PIVOT_TOLERANCE):
    """Per-pixel transit weights and 1-sigma errors of the native regression."""
    full = sparse.csc_matrix(design)
    nuisance = full[:, transits:]
    keep = independent_columns(nuisance, tol)
    kept = nuisance[:, keep]
    transit = full[:, :transits].toarray()
    w = 1.0 / pix_err**2
    # Compare Schur pivots with the original columns (they are invariant to the reparametrization).
    original = w.T @ transit**2
    transit -= nuisance_fit(kept, transit)
    y = pix - nuisance_fit(kept, pix)
    band, border, width = band_split(kept)
    if band is None:
        factor, t, last = dense_factor(kept, transit, y, w)
    else:
        factor, t, last = banded_factor(
            kept[:, band], kept[:, border], width, transit, y, w
        )
    lower = factor[:, t, t]
    pivot = np.diagonal(lower, axis1=1, axis2=2) ** 2
    explained = ~(pivot > tol * original)
    if explained.any():
        raise NotEstimable("Transit depth is not estimable")
    inverse = np.linalg.inv(lower)
    depth = np.einsum("pji,pj->pi", inverse, factor[:, last, t])
    return depth, np.sqrt((inverse**2).sum(axis=1))


def band_split(columns, max_band=MAX_BAND):
    """(band, border, width): the short-support columns, whose normal matrix is banded, and the others."""
    columns = sparse.csc_matrix(columns)
    rows = columns.indices
    first = np.minimum.reduceat(rows, columns.indptr[:-1])
    last = np.maximum.reduceat(rows, columns.indptr[:-1])
    support = last - first + 1
    narrow = support <= max(64, 8 * np.median(support))
    band = np.flatnonzero(narrow)
    band = band[np.lexsort((last[band], first[band]))]
    if not len(band):
        return None, None, None
    pattern = sparse.csc_matrix(columns[:, band] != 0, dtype=np.int8)
    gram = (pattern.T @ pattern).tocoo()
    width = int(np.abs(gram.row - gram.col).max())
    if width > max_band:
        return None, None, None
    return band, np.flatnonzero(~narrow), width


def dense_factor(kept, transit, y, w):
    """Cholesky factors of every pixel's augmented normal equations [nuisance, transit, pixel]."""
    reduced = sparse.hstack([kept, sparse.csc_matrix(transit)], format="csr")
    m = reduced.shape[1]
    augmented = np.empty((y.shape[1], m + 1, m + 1))
    augmented[:, :m, :m] = normal_matrices(reduced, w)
    rhs = np.asarray(reduced.T @ (w * y)).T
    augmented[:, m, :m] = rhs
    augmented[:, :m, m] = rhs
    # Any value above the fitted part keeps the matrix definite; it is unused.
    augmented[:, m, m] = 2.0 * (w * y * y).sum(axis=0) + 1.0
    try:
        factor = np.linalg.cholesky(augmented)
    except np.linalg.LinAlgError as error:
        raise NotEstimable("Transit depth is not estimable") from error
    return factor, slice(m - transit.shape[1], m), m


def banded_matrices(band, w, width):
    """B^T diag(w_p) B for every pixel p in LAPACK lower band storage."""
    band = sparse.csr_matrix(band)
    n, s = band.shape
    counts = np.diff(band.indptr)
    most = int(counts.max())
    rows = np.repeat(np.arange(n), counts)
    slot = np.arange(band.nnz) - np.repeat(band.indptr[:-1], counts)
    index = np.full((n, most), -1, dtype=np.int64)
    value = np.zeros((n, most))
    index[rows, slot] = band.indices
    value[rows, slot] = band.data
    i, j = index[:, :, None], index[:, None, :]
    lower = (i >= j) & (j >= 0)
    position = ((i - j) * s + j)[lower]
    product = (value[:, :, None] * value[:, None, :])[lower]
    row = np.broadcast_to(np.arange(n)[:, None, None], lower.shape)[lower]
    spread = sparse.csr_matrix((product, (position, row)), shape=((width + 1) * s, n))
    return np.asarray(spread @ w).T.reshape(-1, width + 1, s)


def banded_factor(band, border, width, transit, y, w):
    """dense_factor's trailing [border, transit, pixel] block, from the band columns eliminated first."""
    from scipy.linalg import cho_solve_banded, cholesky_banded

    shared = np.hstack([sparse.csc_matrix(border).toarray(), transit])
    k = shared.shape[1]
    pixels = y.shape[1]
    schur = np.empty((pixels, k + 1, k + 1))
    schur[:, :k, :k] = np.einsum("np,ni,nj->pij", w, shared, shared)
    schur[:, k, :k] = np.einsum("np,ni,np->pi", w, shared, y)
    schur[:, :k, k] = schur[:, k, :k]
    # Any value above the fitted part keeps the matrix definite; it is unused.
    schur[:, k, k] = 2.0 * (w * y * y).sum(axis=0) + 1.0
    normal = banded_matrices(band, w, width)
    band_t = sparse.csr_matrix(band.T)
    coupling_y = np.asarray(band_t @ (w * y))
    try:
        for p in range(pixels):
            coupling = np.column_stack(
                [np.asarray(band_t @ (w[:, p, None] * shared)), coupling_y[:, p]]
            )
            factor = cholesky_banded(normal[p], lower=True)
            schur[p] -= coupling.T @ cho_solve_banded((factor, True), coupling)
        factor = np.linalg.cholesky(schur)
    except np.linalg.LinAlgError as error:
        raise NotEstimable("Transit depth is not estimable") from error
    border_count = k - transit.shape[1]
    return factor, slice(border_count, k), k


def spline_design_matrix(time, tdur, order=3, spacing_mult=3.0):
    """utils.create_spline_design_matrix with scipy's B-spline evaluation."""
    from scipy.interpolate import BSpline

    from localizator.upstream import utils

    t = (time - time.mean()) / (time.max() - time.min())
    dt = np.diff(t) / np.median(np.diff(t))
    knot_spacing = (spacing_mult * tdur) / (time.max() - time.min())
    breakpoints = np.where(dt > 4)[0] + 1
    columns = []
    for seg in np.array_split(np.arange(len(t)), breakpoints):
        t_seg = t[seg]
        knots = np.arange(
            np.min(t_seg) - knot_spacing * (order - 1),
            np.max(t_seg) + (knot_spacing * (order - 1)) + knot_spacing,
            knot_spacing,
        )
        try:
            # lamatrix order is degree + 1; a point rounded below the base interval extrapolates the same piece.
            block = BSpline.design_matrix(t_seg, knots, order - 1, extrapolate=True)
            block = block.toarray()
        except ValueError:
            block = utils.Spline("x", knots=knots, order=order).design_matrix(x=t_seg)
        full = np.zeros((len(t), block.shape[1]))
        full[seg, :] = block
        columns.append(full)
    return np.hstack(columns)


def solve_transit_weights(localizer):
    """Localize.solve_transit_weights with the stable regression above."""
    shape = localizer.flux.shape[1:]
    good = localizer.good_pix_mask
    depth, error = transit_depths(
        localizer.design_matrix,
        len(localizer.valid_tces),
        localizer.pix,
        localizer.pix_err,
    )
    localizer.transit_weights, localizer.transit_weights_err = [], []
    for i in localizer.valid_tces:
        column = localizer.tce_mapping[i]
        for values, maps in (
            (depth, localizer.transit_weights),
            (error, localizer.transit_weights_err),
        ):
            image = np.full(shape, np.nan)
            image[good] = values[:, column]
            maps.append(image)


def decompress(raw):
    """TPF bytes as astropy reads them from a .fits or .fits.gz path."""
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def basis_cache():
    default = Path(__file__).resolve().parents[1] / "runtime" / "cache"
    return Path(os.environ.get("KEPLERX_KERNEL_CACHE", default)) / "prf-basis"


def _fit_basis(prf, i):
    key = hashlib.sha256()
    for part in (prf.PRFrow, prf.PRFcol, prf.PRFdata[i]):
        key.update(np.ascontiguousarray(part).tobytes())
    key.update(scipy.__version__.encode())
    path = basis_cache() / f"{key.hexdigest()[:24]}.npz"
    if path.exists():
        with np.load(path) as stored:
            tck = (stored["tx"], stored["ty"], stored["c"])
        spline = RectBivariateSpline.__new__(RectBivariateSpline)
        spline.tck, spline.degrees, spline.fp = tck, (3, 3), 0.0
        return spline
    spline = RectBivariateSpline(prf.PRFrow, prf.PRFcol, prf.PRFdata[i])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    tx, ty, c = spline.tck
    np.savez(temporary, tx=tx, ty=ty, c=c)
    os.replace(temporary, path)
    return spline


def channel_bases(prf):
    """The calibrated images' splines, fitted once per channel and process."""
    key = (prf.mission, prf.channel, prf.date)
    bases = _BASES.pop(key, None)
    if bases is None:
        bases = [None] * len(prf.PRFdata)
    _BASES[key] = bases
    while len(_BASES) > CHANNELS:
        _BASES.popitem(last=False)
    return bases


class Combination:
    """sum_i scale_i spline_i, called like the RectBivariateSpline it replaces."""

    def __init__(self, splines, scales):
        self.terms = list(zip(scales, splines, strict=True))

    def __call__(self, x, y, dx=0, dy=0, grid=True):
        total = None
        for scale, spline in self.terms:
            value = scale * spline(x, y, dx=dx, dy=dy, grid=grid)
            total = value if total is None else total + value
        return total


def linear_preparation(prf, totals):
    """lkprf's per-position preparation as a sum of fixed splines."""
    bases = channel_bases(prf)

    def prepare(targets, shape):
        row, column = prf._unpack_targets(targets)
        row, column = np.atleast_1d(row), np.atleast_1d(column)
        row = np.min([np.max([row, row**0 * 20], axis=0), row**0 * 1044], axis=0)
        column = np.min(
            [np.max([column, column**0 * 12], axis=0), column**0 * 1112], axis=0
        )
        ref_column = column.mean() + 0.5 * shape[1]
        ref_row = row.mean() + 0.5 * shape[0]
        distance = [
            np.sqrt(
                (ref_column - prf.crval1p[i]) ** 2 + (ref_row - prf.crval2p[i]) ** 2
            )
            for i in range(prf.PRFdata.shape[0])
        ]
        nearest = np.argpartition(distance, 3)[:3]
        weights = [max(distance[i], MIN_PRF_WEIGHT) for i in nearest]
        norm = sum(totals[i] / w for i, w in zip(nearest, weights, strict=True))
        norm *= prf.cdelt1p[0] * prf.cdelt2p[0]
        for i in nearest:
            if bases[i] is None:
                bases[i] = _fit_basis(prf, i)
        prf.interpolate = Combination(
            [bases[i] for i in nearest], [1.0 / (w * norm) for w in weights]
        )

    return prepare


def channel_prf(factory, channel):
    """A KeplerPRF whose calibrated images are read once per process."""
    if channel not in _CHANNEL_PRFS:
        base = factory(channel=channel)
        # lkprf's nansum differs from the sum of its parts only with NaNs.
        finite = np.isfinite(base.PRFdata).all()
        totals = base.PRFdata.astype(np.float64).sum(axis=(1, 2)) if finite else None
        _CHANNEL_PRFS[channel] = base, totals
    base, totals = _CHANNEL_PRFS[channel]
    prf = copy.copy(base)
    if totals is not None:
        prf._prepare_supersamp_prf = linear_preparation(prf, totals)
    return prf


def direct_wcs(wcs):
    """Whether all_world2pix(ra, dec) equals upstream's SkyCoord conversion."""
    return wcs.wcs.radesys == "ICRS" and [c[:4] for c in wcs.wcs.ctype] == [
        "RA--",
        "DEC-",
    ]


class PRFResidual:
    """utils.quarters_prf_residual with models cached by quarter and position."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.models = {}

    def model(self, i, prf, wcs, origin, shape, ra, dec):
        key = (i, ra, dec)
        if key not in self.models:
            col_pix, row_pix = wcs.all_world2pix(ra, dec, 0)
            target = (origin[0] + row_pix, origin[1] + col_pix)
            self.models[key] = prf.evaluate(
                targets=[target], origin=origin, shape=shape
            )[0, :, :]
        return self.models[key]

    def __call__(self, params, prf_list, data_list, data_err_list, wcs_list,
                 origin_list, shape_list, tce_mapping_list, valid_idxs, which_tce):  # fmt: skip
        if not all(map(direct_wcs, wcs_list)):
            return self.fallback(
                params, prf_list, data_list, data_err_list, wcs_list, origin_list,
                shape_list, tce_mapping_list, valid_idxs, which_tce,
            )  # fmt: skip
        ra, dec = params["centerra"].value, params["centerdec"].value
        res = []
        for i in valid_idxs:
            which_col = tce_mapping_list[i].get(which_tce)
            if which_col is None:
                continue
            model = self.model(
                i, prf_list[i], wcs_list[i], origin_list[i], shape_list[i], ra, dec
            )
            amp = params[f"amplitude_{i}"].value
            res_i = (
                (amp * model - data_list[i][which_col]) / data_err_list[i][which_col]
            ).ravel()
            res_i = res_i[np.isfinite(res_i)]
            if res_i.size:
                res.append(res_i)
        if not res:
            raise ValueError(
                "Multiple Quarter PRF residual failed: no valid pixels in any quarter"
            )
        return np.concatenate(res)


class Propagation:
    """utils.propagate_query and utils.propagate_coord for every quarter, one astropy call each."""

    def __init__(self, query, coord):
        self.native_query, self.native_coord = query, coord
        self.tables, self.moved, self.points = [], {}, {}

    @staticmethod
    def _key(table, epoch):
        return (id(table), epoch.scale, float(epoch.jd1), float(epoch.jd2))

    def prepare(self, table, epochs, points):
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time

        self.tables.append(table)  # keeps id(table) unique while keyed
        batch = Time(epochs)
        gaia = SkyCoord(
            ra=table["RA_ICRS"], dec=table["DE_ICRS"], pm_ra_cosdec=table["pmRA"],
            pm_dec=table["pmDE"], frame="icrs", obstime=Time("J2016"),
        )  # fmt: skip
        moved = gaia[None, :].apply_space_motion(new_obstime=batch[:, None])
        for i, epoch in enumerate(epochs):
            self.moved[self._key(table, epoch)] = moved[i]
        for ra, dec in points:
            target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            idx, sep, _ = target.match_to_catalog_sky(gaia)
            if sep > 2.0 * u.arcsec:
                continue  # upstream raises when it is asked
            row = table[idx]
            coord = SkyCoord(
                ra=row["RA_ICRS"] * u.deg, dec=row["DE_ICRS"] * u.deg,
                pm_ra_cosdec=row["pmRA"] * u.mas / u.yr,
                pm_dec=row["pmDE"] * u.mas / u.yr, frame="icrs", obstime=Time("J2016"),
            )  # fmt: skip
            new = coord.apply_space_motion(new_obstime=batch)
            for i, epoch in enumerate(epochs):
                key = (ra, dec) + self._key(table, epoch)
                self.points[key] = new.ra.deg[i], new.dec.deg[i]

    def query(self, query_result, wcs, epoch):
        moved = self.moved.get(self._key(query_result, epoch))
        if moved is None:
            return self.native_query(query_result, wcs, epoch)
        return moved, wcs.all_world2pix(moved.ra, moved.dec, 0)

    def coord(self, ra_targ, dec_targ, query_result, epoch, *args, **kwargs):
        point = None
        if not args and not kwargs:
            point = self.points.get(
                (ra_targ, dec_targ) + self._key(query_result, epoch)
            )
        if point is None:
            return self.native_coord(
                ra_targ, dec_targ, query_result, epoch, *args, **kwargs
            )
        return point


# An amplitude this close to its bound, relative to the largest, rests on it.
AT_BOUND = 1e-5


def held_at_bounds(minimize, residual):
    """lmfit.minimize, refitted with amplitudes held at their bounds when the centre lost its errors."""

    def fit(fcn, params, **kwargs):
        result = minimize(fcn, params, **kwargs)
        if fcn is not residual or not result.success or centred(result):
            return result
        held = result.params.copy()
        amplitudes = {
            n: p for n, p in held.items() if n.startswith("amplitude_") and p.vary
        }
        top = max((p.value for p in amplitudes.values()), default=0.0)
        stuck = [n for n, p in amplitudes.items() if p.value - p.min <= AT_BOUND * top]
        if not stuck or top <= 0:
            return result
        for name in stuck:
            held[name].vary = False
        again = minimize(fcn, held, **kwargs)
        return again if again.success and centred(again) else result

    return fit


def centred(result):
    """Whether the joint fit's centre has finite, positive uncertainties."""
    errors = [result.params[name].stderr for name in ("centerra", "centerdec")]
    return all(e is not None and np.isfinite(e) and e > 0 for e in errors)
