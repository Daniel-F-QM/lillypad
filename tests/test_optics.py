"""Delay/position conversions, the delay grid, and the two analysis reductions.

Ported from scan.py's self-test (lines 728-739). This is the layer whose
constants are load-bearing for every recorded trace: C_NM_PER_FS sets the delay
axis, and pass_factor sets how far the stage moves per femtosecond.
"""

from __future__ import annotations

import numpy as np
import pytest

from lillypad.scan import (FrogScanConfig, autocorrelation, build_delay_grid,
                           delay_to_position_um, fwhm, position_to_delay_fs,
                           stage_to_um, um_to_stage)


def test_delay_position_round_trip_is_exact():
    cfg = FrogScanConfig(delay_start_fs=-300, delay_stop_fs=300,
                         delay_step_fs=13.3426, zero_pos_um=150000.0)
    d = cfg.delays_fs()
    p = cfg.positions_um()
    back = position_to_delay_fs(p, cfg.zero_pos_um, cfg.pass_factor)
    # The self-test printed this error; it is ~1e-12 fs, i.e. float noise.
    assert np.max(np.abs(back - d)) < 1e-9


def test_positions_bracket_the_marked_zero():
    cfg = FrogScanConfig(delay_start_fs=-300, delay_stop_fs=300,
                         delay_step_fs=13.3426, zero_pos_um=150000.0)
    p = cfg.positions_um()
    assert p.min() < cfg.zero_pos_um < p.max()
    # Double pass: 300 fs of delay is ~45 um of travel, not ~90.
    assert 40.0 < (p.max() - p.min()) / 2 < 50.0


def test_pass_factor_halves_the_travel():
    single = delay_to_position_um(100.0, 0.0, pass_factor=1)
    double = delay_to_position_um(100.0, 0.0, pass_factor=2)
    assert single == pytest.approx(2.0 * double)


def test_mm_um_conversions_are_inverses():
    assert um_to_stage(150000.0) == pytest.approx(150.0)
    assert stage_to_um(150.0) == pytest.approx(150000.0)
    assert stage_to_um(um_to_stage(1234.5)) == pytest.approx(1234.5)


class TestDelayGrid:
    def test_ascending(self):
        g = build_delay_grid(-100.0, 100.0, 10.0)
        assert g[0] == -100.0 and g[-1] <= 100.0
        assert np.all(np.diff(g) > 0)

    def test_descending_keeps_direction(self):
        """A high-to-low scan must stay high-to-low: the sweep direction is what
        makes the backlash approach single-sided."""
        g = build_delay_grid(100.0, -100.0, 10.0)
        assert g[0] == 100.0
        assert np.all(np.diff(g) < 0)

    @pytest.mark.parametrize("args", [
        (0.0, 0.0, 10.0),      # zero span
        (-100.0, 100.0, 0.0),  # zero step
    ])
    def test_rejects_degenerate_input(self, args):
        with pytest.raises(Exception):
            build_delay_grid(*args)


def test_fwhm_of_a_known_gaussian():
    """sigma=80 fs -> FWHM = 2*sqrt(2 ln2)*80 = 188.4 fs. The self-test printed
    this and expected ~188."""
    d = np.linspace(-500.0, 500.0, 2001)
    ac = np.exp(-(d ** 2) / (2 * 80.0 ** 2))
    assert fwhm(d, ac) == pytest.approx(188.4, abs=0.5)


@pytest.mark.parametrize("y", [
    pytest.param(np.zeros(101), id="all-zero"),
    pytest.param(-np.ones(101), id="all-negative"),
    pytest.param(np.ones(2), id="too-few-points"),
])
def test_fwhm_returns_nan_when_it_cannot_bracket(y):
    x = np.linspace(-10.0, 10.0, y.size)
    assert np.isnan(fwhm(x, y))


def test_fwhm_of_a_flat_positive_curve_is_the_full_span():
    """Characterization, not an aspiration. The half level is halfway between
    min and max, so on a constant curve every sample qualifies and the width
    comes back as the whole x range. Worth pinning: it is the value the FWHM
    readout shows for a dark, flat autocorrelation."""
    d = np.linspace(-10.0, 10.0, 101)
    assert fwhm(d, np.ones_like(d)) == pytest.approx(20.0)


def test_autocorrelation_is_the_wavelength_marginal():
    n_wl, n_delay = 16, 9
    trace = np.zeros((n_wl, n_delay))
    trace[:, 4] = 10.0            # one bright column at the centre
    ac = autocorrelation(trace, baseline_subtract=False)
    assert ac.shape == (n_delay,)
    assert int(np.argmax(ac)) == 4
    assert ac[4] == pytest.approx(n_wl * 10.0)


def test_autocorrelation_baseline_subtract_removes_a_pedestal():
    trace = np.full((16, 9), 5.0)
    trace[:, 4] += 10.0
    plain = autocorrelation(trace, baseline_subtract=False)
    subbed = autocorrelation(trace, baseline_subtract=True)
    assert subbed.min() == pytest.approx(0.0)
    assert plain.min() > 0.0
    # The peak-to-floor contrast is preserved, only the offset goes.
    assert (subbed.max() - subbed.min()) == pytest.approx(plain.max() - plain.min())
