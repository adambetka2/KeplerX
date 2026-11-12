"""The iterative search's repeat rule: one signal, one candidate."""

import numpy as np
import pandas as pd

from detector.gap_padding import MIN_SEARCH_CADENCES, DegenerateCurve


def remove_repeats(test_TCEs, known_TCEs, tolerance=0.0001):
    """upstream.remove_TCE_harmonics(test_TCEs, known_TCEs) without its single-detection exemption."""
    sorted_mes_index = np.argsort(-test_TCEs[:, 2])
    test_TCEs_sorted = test_TCEs[sorted_mes_index]
    is_harmonic = np.zeros(len(test_TCEs), dtype=bool)
    known_periods = known_TCEs[:, 1]
    known_t0s = known_TCEs[:, 3]
    tce_t0s, tce_durs = test_TCEs_sorted[:, 3], test_TCEs_sorted[:, 4]
    for i, p in enumerate(test_TCEs_sorted[:, 1]):
        harm_test = np.concatenate(
            [(known_periods / p) % 1.0, (p / known_periods) % 1.0]
        )
        if any(harm_test < tolerance) or any((1.0 - harm_test) < tolerance):
            same_period = np.abs(known_TCEs[:, 1] / p - 1.0) < tolerance
            same_epoch = np.abs(known_t0s - tce_t0s[i]) < 2.0 * tce_durs[i]
            if same_period.any() or same_epoch.any():
                is_harmonic[i] = True
        else:
            known_periods = np.append(known_periods, p)
    test_TCEs_noharm = test_TCEs_sorted[~is_harmonic]
    same_t0 = np.zeros(len(test_TCEs_noharm), dtype=bool)
    for i, tce in enumerate(test_TCEs_noharm[1:]):
        if any(np.abs(test_TCEs_noharm[: i + 1, 3] - tce[3]) < 2 * tce[4]):
            same_t0[i + 1] = True
    return test_TCEs_noharm[~same_t0]


ALIAS_DENOMINATOR = 4
# Fractional aliases are exact to the refined periods (about 1e-5).
FRACTION_TOLERANCE = 5e-5


def _offset(a, b, spacing):
    return abs((a - b + spacing / 2) % spacing - spacing / 2)


def alias(tce, known, same_pass, tolerance):
    """Whether detection tce (star, P, mes, t0, tdur) is an alias of `known` under epoch cleaning."""
    P, other = tce[1], known[1]
    duration = max(tce[4], known[4])
    ratio = P / other
    for x in range(1, 11):
        if abs(ratio / x - 1.0) < tolerance:
            return True
        if x > 1 and abs(ratio * x - 1.0) < tolerance:
            shared = _offset(tce[3], known[3], P) < duration
            return not shared or same_pass
    short, long = min(P, other), max(P, other)
    for q in range(2, ALIAS_DENOMINATOR + 1):
        p = round(q * long / short)
        # Up to ten periods: beyond, the windows of p/q crowd together and catch planets by chance.
        if q < p <= 10 * q and abs(q * long / short - p) < FRACTION_TOLERANCE * p:
            if _offset(tce[3], known[3], short / q) < duration:
                return True
    return False


class Repeats:
    """Mixin before a stage's upstream Dump: the iterative search's repeat rule."""

    repeat_passes = 1
    # Refined periods of one signal differ between passes by up to two trial steps.
    repeat_tolerance = 5e-4

    def _later_pass(self, search, check_vetoes, fill_mode):
        """One masked pass, or None when too little is left to search."""
        if np.count_nonzero(self.lc.mask & self.tce_mask) < MIN_SEARCH_CADENCES:
            return None
        try:
            return self.Search_for_TCEs(
                calc_ses=True, remove_hises=False, return_df=False,
                use_tce_mask=True, check_vetoes=check_vetoes, fill_mode=fill_mode,
                **search,
            )  # fmt: skip
        except DegenerateCurve:
            return None

    def _record(self, TCEs, n, searched, candidates):
        """Upstream's harmonic removal and masks for a pass, so the passes follow upstream's."""
        if n > 1:
            TCEs = self._upstream.remove_TCE_harmonics(TCEs, searched.reshape(-1, 5))
        for tce in TCEs:
            self.tce_mask &= self._upstream.make_transit_mask(
                self.lc.time, P=tce[1], t0=tce[3], dur=tce[4]
            )
        new = (
            remove_repeats(TCEs, candidates.reshape(-1, 5), self.repeat_tolerance)
            if n > 1
            else TCEs
        )
        flags = np.array(
            [any(np.array_equal(row, x) for x in new) for row in TCEs], dtype=bool
        )
        if self.reference is None:
            flags &= ~self._alias_repeats(TCEs, searched.reshape(-1, 5))
        return TCEs, flags

    def _alias_repeats(self, TCEs, searched):
        """Detections of this pass that are another's alias (epoch cleaning keeps them in the pass)."""
        repeat = np.zeros(len(TCEs), dtype=bool)
        order = np.argsort(-TCEs[:, 2], kind="stable")
        for rank, i in enumerate(order):
            stronger = TCEs[order[:rank]]
            repeat[i] = any(
                alias(TCEs[i], k, True, self.repeat_tolerance) for k in stronger
            )
            repeat[i] |= any(
                alias(TCEs[i], k, False, self.repeat_tolerance) for k in searched
            )
        return repeat

    def Iterative_TCE_Search(self, niter=3, ses_cut=999999.0, threshold=7.0,
                             check_vetoes=False, remove_hises=False, mad_window=7.0,
                             fill_mode="reflect", **tce_search_kw):  # fmt: skip
        """upstream's Iterative_TCE_Search with the repeat rule."""
        n, ntces, idle = 0, 1, 0
        candidates, searched, is_candidate = np.array([]), np.array([]), []
        self.tce_mask = self.lc.mask.copy()
        search = dict(threshold=threshold, ses_cut=ses_cut, **tce_search_kw)
        while n < niter and ntces > 0:
            n += 1
            if n == 1:
                TCEs = self.Search_for_TCEs(
                    calc_ses=True, remove_hises=remove_hises, return_df=False,
                    use_tce_mask=False, check_vetoes=False, **search,
                )  # fmt: skip
            else:
                TCEs = self._later_pass(search, check_vetoes, fill_mode)
                if TCEs is None:
                    break
            # Folds that missed the transit requirement mask their events as detections do.
            events = getattr(self, "masked_events", None) or []
            for centre, width in events:
                self.tce_mask &= np.abs(self.lc.time - centre) > width
            if np.sum(np.isnan(TCEs[:, 1])) > 0:
                idle += 1
                ntces = 1 if events and idle <= self.repeat_passes else 0
            else:
                TCEs, flags = self._record(TCEs, n, searched, candidates)
                ntces = len(TCEs)
                searched = np.append(searched, TCEs)
                candidates = np.append(candidates, TCEs[flags] if len(TCEs) else [])
                is_candidate.extend(bool(f) for f in flags)
                idle = 0 if flags.any() else idle + 1
                if idle > self.repeat_passes:
                    ntces = 0
            if len(searched) > 10 * 5:
                ntces = 0
        self.TCEs = candidates.reshape(-1, 5)
        # Masked detections in upstream's order, as upstream's vetting masks them.
        self.searched_TCEs = searched.reshape(-1, 5)
        self.searched_is_candidate = np.array(is_candidate, dtype=bool)
        return pd.DataFrame(
            candidates.reshape(-1, 5),
            columns=["star_id", "period", "mes", "t0", "tdur"],
        )
