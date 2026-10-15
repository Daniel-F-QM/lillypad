"""Folding a trace about its overlap: where the fold goes and what the score is.

These two functions used to be methods on the Qt window, which meant the only
way to check "does a perfectly symmetric trace score zero" was to build a
QApplication. They are numpy in and numpy out, so they now live in
lillypad.scan.core with the rest of the maths and can be tested for what they
actually claim.

The claims worth pinning, in the order they bite:

  * the fold STARTS at the AUTOCORRELATION PEAK, not the middle column, or a
    scan range that was not centred on the overlap would report its own offset
    as asymmetry — and then moves, in half columns, to the most symmetric axis
    near it;
  * the difference is ABSOLUTE, because every colormap here is sequential with
    clim (0, peak) and signed values would flatten onto the floor;
  * columns with no counterpart across the fold are NaN, not zero — zero would
    claim perfect symmetry where there is nothing to compare;
  * the score collapses the WAVELENGTH axis before taking the modulus. That one
    is not cosmetic and is the last test here.
"""

from __future__ import annotations

import numpy as np
import pytest

from lillypad.scan import optimal_fold, symmetry_fold, symmetry_view

N_WL, N_DELAY = 24, 21
DELAYS = np.linspace(-100.0, 100.0, N_DELAY)          # 10 fs per column


def gaussian_trace(center_fs=0.0, width_fs=30.0, peak=1000.0):
    """A trace that is symmetric about `center_fs` by construction."""
    d = DELAYS[None, :] - center_fs
    wl = np.linspace(-1.0, 1.0, N_WL)[:, None]
    return peak * np.exp(-(d / width_fs) ** 2) * np.exp(-(wl / 0.6) ** 2)


def ac_of(trace):
    return np.asarray(trace, float).sum(axis=0)


def fold_fs(k):
    """Delay of half-index k: on column k/2, or halfway between two."""
    return 0.5 * (DELAYS[k // 2] + DELAYS[(k + 1) // 2])


class TestWhereTheFoldGoes:
    def test_a_trace_symmetric_about_a_column_folds_on_that_column(self):
        trace = gaussian_trace(center_fs=0.0)
        assert symmetry_fold(ac_of(trace), DELAYS) == 2 * (N_DELAY // 2)

    def test_an_off_centre_scan_folds_about_its_overlap_not_its_middle(self):
        """The whole reason it is not the middle column: a range that was not
        centred on the overlap would otherwise report its own offset as
        asymmetry."""
        trace = gaussian_trace(center_fs=40.0)
        k = symmetry_fold(ac_of(trace), DELAYS)
        assert fold_fs(k) == pytest.approx(40.0)
        assert k != 2 * (N_DELAY // 2)

    def test_a_hand_placed_fold_overrides_the_optimum(self):
        trace = gaussian_trace(center_fs=0.0)
        k = symmetry_fold(ac_of(trace), DELAYS, fold_fs=-50.0)
        assert fold_fs(k) == pytest.approx(-50.0)

    def test_a_hand_placed_fold_snaps_to_the_nearest_half_column(self):
        """On a measured column or exactly halfway between two — never
        anywhere that would need a resampled column to compare against."""
        ac = ac_of(gaussian_trace())
        assert fold_fs(symmetry_fold(ac, DELAYS, fold_fs=23.0)) == pytest.approx(25.0)
        assert fold_fs(symmetry_fold(ac, DELAYS, fold_fs=21.0)) == pytest.approx(20.0)

    def test_a_fold_outside_the_range_lands_on_the_nearest_edge(self):
        trace = gaussian_trace()
        assert symmetry_fold(ac_of(trace), DELAYS, fold_fs=-1e6) == 0
        assert symmetry_fold(ac_of(trace), DELAYS,
                             fold_fs=1e6) == 2 * (N_DELAY - 1)


class TestTheFoldIsOptimised:
    """It starts at the AC peak and moves to the most symmetric axis near it.
    The peak is quantised to whole columns, so a trace centred between two
    would otherwise score asymmetric for no physical reason."""

    def test_a_trace_centred_between_columns_folds_between_them(self):
        trace = gaussian_trace(center_fs=5.0)       # halfway between 0 and 10
        ac = ac_of(trace)
        assert DELAYS[np.argmax(ac)] in (0.0, 10.0)  # the peak can't say 5
        view, _p, score, fold = symmetry_view(trace, ac, DELAYS)
        assert fold == pytest.approx(5.0)
        assert score == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(view[:, 11:], 0.0, atol=1e-9)

    def test_it_never_scores_worse_than_the_ac_peak(self):
        rng = np.random.default_rng(7)
        for center in (-13.0, 2.0, 5.0, 17.0, 31.0):
            trace = gaussian_trace(center_fs=center)
            trace = np.clip(trace + rng.normal(0, 20, trace.shape), 0, None)
            ac = ac_of(trace)
            at_peak = symmetry_view(trace, ac, DELAYS,
                                    fold_fs=DELAYS[np.argmax(ac)])[2]
            assert symmetry_view(trace, ac, DELAYS)[2] <= at_peak + 1e-12

    def test_it_stays_near_the_peak(self):
        """Towards the edges fewer pairs are compared and the score shrinks
        for that reason alone; the search must not wander off to exploit it."""
        trace = gaussian_trace(center_fs=0.0)
        trace[:, 13:] *= 2.0                        # lopsided: no good fold
        ac = ac_of(trace)
        k = optimal_fold(ac, DELAYS)
        width = 30.0 * np.sqrt(2.0) * 2.0 * np.sqrt(np.log(2.0))   # AC FWHM
        peak = DELAYS[np.argmax(ac)]
        assert abs(fold_fs(k) - peak) <= max(width / 2, 20.0) + 1e-9

    def test_a_flat_or_empty_autocorrelation_does_not_raise(self):
        assert optimal_fold(np.zeros(N_DELAY), DELAYS) == 0
        assert optimal_fold(np.full(N_DELAY, np.nan), DELAYS) == 0


class TestWhatTheViewContains:
    def test_the_left_half_is_the_measured_trace_untouched(self):
        """Kept as measured so the eye has something to judge the other half
        against, on the same colour scale."""
        trace = gaussian_trace()
        view, peak, _score, _fold = symmetry_view(trace, ac_of(trace), DELAYS)
        c = N_DELAY // 2
        assert np.allclose(view[:, :c], trace[:, :c])
        assert peak == pytest.approx(trace.max())

    def test_a_symmetric_trace_gives_a_black_right_half_and_a_zero_score(self):
        trace = gaussian_trace(center_fs=0.0)
        view, _peak, score, fold = symmetry_view(trace, ac_of(trace), DELAYS)
        c = N_DELAY // 2
        assert np.allclose(view[:, c:], 0.0, atol=1e-9)
        assert score == pytest.approx(0.0, abs=1e-12)
        assert fold == pytest.approx(0.0)

    def test_an_asymmetric_trace_scores_above_zero(self):
        trace = gaussian_trace(center_fs=0.0)
        trace[:, 14:] *= 3.0                      # one wing lifted
        _view, _peak, score, _fold = symmetry_view(trace, ac_of(trace), DELAYS)
        assert score > 0.05

    def test_the_difference_is_absolute(self):
        """Signed values would land below zero, and every colormap here is
        sequential with clim (0, peak) — they would flatten onto the floor and
        hide exactly what the mode exists to show."""
        trace = gaussian_trace(center_fs=0.0)
        trace[:, 14:] *= 3.0
        view, _peak, _score, _fold = symmetry_view(trace, ac_of(trace), DELAYS)
        right = view[:, N_DELAY // 2:]
        assert np.nanmin(right) >= 0.0
        assert np.nanmax(right) > 0.0

    def test_columns_with_no_counterpart_are_nan_not_zero(self):
        """Zero would claim perfect symmetry where there is simply nothing to
        compare against. NaN takes the plot background, so unmeasured reads as
        unmeasured."""
        trace = gaussian_trace(center_fs=0.0)
        view, _p, _s, _f = symmetry_view(trace, ac_of(trace), DELAYS,
                                         fold_fs=-60.0)
        c = symmetry_fold(ac_of(trace), DELAYS, fold_fs=-60.0) // 2
        h = min(c, N_DELAY - 1 - c)
        assert np.all(np.isnan(view[:, c + h + 1:]))
        assert not np.any(np.isnan(view[:, :c]))

    def test_a_fold_on_the_edge_degrades_rather_than_raising(self):
        trace = gaussian_trace()
        view, peak, score, fold = symmetry_view(trace, ac_of(trace), DELAYS,
                                                fold_fs=DELAYS[0])
        assert np.all(np.isnan(view[:, 1:]))
        assert score == 0.0
        assert peak > 0
        assert fold == pytest.approx(DELAYS[0])

    def test_the_input_trace_is_not_modified(self):
        """It is the finished scan's display view — every other panel reads the
        same array."""
        trace = gaussian_trace()
        before = trace.copy()
        symmetry_view(trace, ac_of(trace), DELAYS)
        assert np.array_equal(trace, before)

    def test_a_half_column_fold_compares_the_columns_either_side(self):
        """Fold at -55 fs, between columns 4 and 5: column 5 is compared with
        4, 6 with 3 … 9 with 0, and everything after that has no partner."""
        trace = gaussian_trace(center_fs=0.0)
        view, _p, _s, fold = symmetry_view(trace, ac_of(trace), DELAYS,
                                           fold_fs=-55.0)
        assert fold == pytest.approx(-55.0)
        assert np.allclose(view[:, :5], trace[:, :5])
        for j in range(5, 10):
            assert np.allclose(view[:, j], np.abs(trace[:, j] - trace[:, 9 - j]))
        assert np.all(np.isnan(view[:, 10:]))


class TestTheScoreCollapsesWavelengthFirst:
    """The subtle one, and the reason the score is not a per-pixel mean.

    Per pixel, |noise − noise| never cancels: the modulus turns every one of
    ~30k near-zero background pixels into a positive contribution and their sum
    swamps the signal. Measured on the simulator, a PERFECTLY symmetric beam
    scored 13.8% at 460 peak counts, 2.7% at 4000 and 0.9% at 20000 — a floor
    that moves with exposure and, at realistic levels, is not far off the 26.7%
    a badly misaligned trace scored.
    """

    @staticmethod
    def _noisy(peak, seed=4):
        rng = np.random.default_rng(seed)
        trace = gaussian_trace(center_fs=0.0, peak=peak)
        return np.clip(trace + rng.normal(0.0, 0.02 * peak, trace.shape), 0, None)

    def test_a_symmetric_beam_scores_near_zero_at_every_exposure(self):
        """Near enough exposure-independent, which is what makes the number
        worth showing at all."""
        scores = [symmetry_view(self._noisy(p), ac_of(self._noisy(p)),
                                DELAYS)[2] for p in (460.0, 4000.0, 20000.0)]
        assert max(scores) < 0.05, scores
        assert max(scores) - min(scores) < 0.03, scores

    def test_a_misaligned_trace_scores_well_clear_of_the_symmetric_floor(self):
        """The contrast is the useful property, not the absolute number. One
        wing lifted by 2.5x scores ~0.14 here against ~0.01 for the same beam
        symmetric — an order of magnitude, which is what makes the readout
        worth looking at."""
        clean = self._noisy(4000.0)
        floor = symmetry_view(clean, ac_of(clean), DELAYS)[2]
        trace = self._noisy(4000.0)
        trace[:, 11:] *= 2.5
        score = symmetry_view(trace, ac_of(trace), DELAYS)[2]
        assert score > 0.10, score
        assert score > 8 * max(floor, 1e-3), (score, floor)

    def test_the_per_pixel_score_would_have_a_floor_that_moves(self):
        """Characterising the alternative, so the choice is not re-litigated.
        This is what the score would do if the modulus came first."""
        def per_pixel(trace):
            c = N_DELAY // 2
            h = min(c, N_DELAY - 1 - c)
            d = np.abs(trace[:, c:c + h + 1] - trace[:, c - h:c + 1][:, ::-1])
            return float(d.sum() / trace.sum())

        floors = [per_pixel(self._noisy(p)) for p in (460.0, 20000.0)]
        collapsed = [symmetry_view(self._noisy(p), ac_of(self._noisy(p)),
                                   DELAYS)[2] for p in (460.0, 20000.0)]
        assert max(floors) > max(collapsed), (floors, collapsed)


class TestItIsReachableWithoutQt:
    def test_the_symmetry_maths_lives_in_the_core(self):
        """It was on the Qt window, where testing "does a symmetric trace score
        zero" needed a QApplication."""
        from lillypad.scan import core
        assert hasattr(core, "symmetry_fold")
        assert hasattr(core, "optimal_fold")
        assert hasattr(core, "symmetry_view")
