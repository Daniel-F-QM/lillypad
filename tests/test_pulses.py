"""The pulse-simulation physics: envelopes, spectral phase, the Fourier shift,
the split-step propagator, and the "bird" measured continuum.

New tests — the self-tests only exercised this layer indirectly, by acquiring a
column from each preset and printing the peak. These pin the maths itself, which
is what makes the `simulation/` split verifiable: every function here is pure
numpy over an array, with no device and no Qt in sight.
"""

from __future__ import annotations

import numpy as np
import pytest

from hardware import (DEFAULT_PULSE, PULSE_SHAPES, SIM_BEAM_DEFAULTS,
                      SIM_SPECTROMETERS, SimulatedSpectrometer, SimulatedStage,
                      sim_spectrometer_pair, _bird_envelope, _energy_window,
                      _fourier_shift, _gauss, _misaligned_efficiency, _sech,
                      _spectral_phase, _split_step)


@pytest.fixture
def grid():
    """The simulator's own default time grid: +/-1024 fs over 4096 samples."""
    n = int(SIM_BEAM_DEFAULTS["n_time"])
    t_max = float(SIM_BEAM_DEFAULTS["t_max_fs"])
    t = np.linspace(-t_max, t_max, n, endpoint=False)
    return t, t[1] - t[0]


def _intensity_fwhm(t, A):
    """FWHM of |A|^2 by linear interpolation of the half-max crossings."""
    I = np.abs(A) ** 2
    half = 0.5 * I.max()
    idx = np.flatnonzero(I >= half)
    lo, hi = idx[0], idx[-1]
    left = np.interp(half, [I[lo - 1], I[lo]], [t[lo - 1], t[lo]])
    right = np.interp(half, [I[hi + 1], I[hi]], [t[hi + 1], t[hi]])
    return right - left


class TestEnvelopes:
    def test_sech_intensity_fwhm_is_1_7627_tau0(self, grid):
        t, _ = grid
        tau0 = 30.0
        assert _intensity_fwhm(t, _sech(t, tau0)) == pytest.approx(
            1.7627 * tau0, rel=1e-3)

    def test_gauss_matches_the_sech_fwhm_by_construction(self, grid):
        """The docstring's whole claim: the Gaussian is built to have the same
        intensity FWHM as sech(t/tau0), so the presets are comparable."""
        t, _ = grid
        tau0 = 30.0
        assert (_intensity_fwhm(t, _gauss(t, tau0))
                == pytest.approx(_intensity_fwhm(t, _sech(t, tau0)), rel=1e-3))

    def test_both_peak_at_zero_delay_with_unit_amplitude(self, grid):
        t, _ = grid
        for build in (_sech, _gauss):
            A = build(t, 30.0)
            assert np.abs(A).max() == pytest.approx(1.0, rel=1e-6)
            assert abs(t[int(np.argmax(np.abs(A)))]) <= t[1] - t[0]


class TestSpectralPhase:
    def test_zero_phase_is_the_identity(self, grid):
        t, dt = grid
        A = _sech(t, 30.0)
        assert np.allclose(_spectral_phase(A, dt, 0.0, 0.0), A, atol=1e-12)

    def test_gdd_broadens_the_pulse(self, grid):
        t, dt = grid
        A = _sech(t, 30.0)
        chirped = _spectral_phase(A, dt, phi2=2000.0)
        assert _intensity_fwhm(t, chirped) > 2.0 * _intensity_fwhm(t, A)

    def test_gdd_conserves_energy(self, grid):
        """Pure phase in the spectral domain cannot change the energy."""
        t, dt = grid
        A = _sech(t, 30.0)
        chirped = _spectral_phase(A, dt, phi2=2000.0, phi3=5e4)
        assert (np.sum(np.abs(chirped) ** 2)
                == pytest.approx(np.sum(np.abs(A) ** 2), rel=1e-9))

    def test_gdd_sign_is_symmetric_in_duration(self, grid):
        """+phi2 and -phi2 give the same intensity envelope. An SHG trace cannot
        tell them apart either, which is why the sign convention only matters
        for the PG gate."""
        t, dt = grid
        A = _sech(t, 30.0)
        up = np.abs(_spectral_phase(A, dt, phi2=1500.0)) ** 2
        down = np.abs(_spectral_phase(A, dt, phi2=-1500.0)) ** 2
        assert np.allclose(up, down, atol=1e-9 * up.max())

    def test_tod_makes_the_envelope_asymmetric(self, grid):
        t, dt = grid
        A = _spectral_phase(_sech(t, 30.0), dt, phi3=2e5)
        I = np.abs(A) ** 2
        centre = np.sum(t * I) / np.sum(I)
        # Third moment about the centroid: nonzero means a one-sided tail.
        skew = np.sum(((t - centre) ** 3) * I) / np.sum(I)
        assert abs(skew) > 1.0


class TestFourierShift:
    @pytest.mark.parametrize("tau", [0.0, 50.0, -50.0])
    def test_shifts_the_peak_by_tau(self, grid, tau):
        t, _ = grid
        A = _sech(t, 30.0)
        out = _fourier_shift(A, t, tau)
        peak_t = t[int(np.argmax(np.abs(out)))]
        assert peak_t == pytest.approx(tau, abs=2.0 * (t[1] - t[0]))

    def test_is_exact_for_a_chirped_field(self, grid):
        """The reason it is a Fourier shift and not an interpolation: an
        interpolated shift smears a strongly chirped field."""
        t, dt = grid
        A = _spectral_phase(_sech(t, 30.0), dt, phi2=3000.0)
        there = _fourier_shift(A, t, 40.0)
        back = _fourier_shift(there, t, -40.0)
        # Compare only where neither pass zeroed a wrapped tail.
        core = np.abs(t) < 400.0
        assert np.allclose(back[core], A[core], atol=1e-8)

    def test_wrapped_tail_is_zeroed_not_recirculated(self, grid):
        """Cyclic shifting would make a pulse pushed off one edge reappear on
        the other and fake a signal at large delay."""
        t, _ = grid
        A = _sech(t, 30.0)
        tau = 900.0
        out = _fourier_shift(A, t, tau)
        assert np.all(out[t < t[0] + tau] == 0.0)


class TestEnergyWindow:
    def test_brackets_a_narrow_peak_tightly(self):
        x = np.linspace(-100.0, 100.0, 2001)
        w = np.exp(-0.5 * (x / 5.0) ** 2)
        lo, hi = _energy_window(x, w, frac=0.995)
        assert -25.0 < lo < -10.0
        assert 10.0 < hi < 25.0

    def test_wider_frac_gives_a_wider_window(self):
        x = np.linspace(-100.0, 100.0, 2001)
        w = np.exp(-0.5 * (x / 5.0) ** 2)
        tight = _energy_window(x, w, frac=0.90)
        loose = _energy_window(x, w, frac=0.999)
        assert loose[0] < tight[0] and loose[1] > tight[1]

    def test_zero_weight_falls_back_to_the_full_range(self):
        x = np.linspace(-10.0, 10.0, 101)
        assert _energy_window(x, np.zeros_like(x)) == (-10.0, 10.0)


class TestSplitStep:
    def test_conserves_energy(self, grid):
        """Self-phase modulation is a pure phase in time and the dispersion a
        pure phase in frequency, so the propagator is lossless by construction.
        If a refactor breaks the FFT normalisation, this is what catches it."""
        t, _ = grid
        A = _sech(t, 30.0)
        out = _split_step(A, t, phi2=500.0, b_int=2.0, n_steps=50)
        assert (np.sum(np.abs(out) ** 2)
                == pytest.approx(np.sum(np.abs(A) ** 2), rel=1e-9))

    def test_zero_dispersion_and_zero_nonlinearity_is_the_identity(self, grid):
        t, _ = grid
        A = _sech(t, 30.0)
        out = _split_step(A, t, phi2=0.0, b_int=0.0, n_steps=10)
        assert np.allclose(out, A, atol=1e-12)

    def test_spm_broadens_the_spectrum(self, grid):
        t, _ = grid
        A = _sech(t, 30.0)
        out = _split_step(A, t, phi2=0.0, b_int=4.0, n_steps=100)

        def rms_bandwidth(field):
            S = np.abs(np.fft.fft(field)) ** 2
            f = np.fft.fftfreq(field.size, d=t[1] - t[0])
            centre = np.sum(f * S) / np.sum(S)
            return np.sqrt(np.sum(((f - centre) ** 2) * S) / np.sum(S))

        assert rms_bandwidth(out) > 1.5 * rms_bandwidth(A)


class TestMisalignedEfficiency:
    def test_is_monotonically_falling_in_delay(self):
        tau = np.linspace(-200.0, 200.0, 401)
        eff = _misaligned_efficiency(tau, 30.0)
        assert np.all(np.diff(eff) < 0)

    def test_is_one_sided(self):
        """One wing is vignetted and the other is not — that asymmetry is the
        whole point of the preset, and what symmetry mode exists to catch."""
        tau0 = 30.0
        assert _misaligned_efficiency(-5 * tau0, tau0) > 0.99
        assert _misaligned_efficiency(+5 * tau0, tau0) < 0.01

    def test_scales_with_the_pulse_duration(self):
        """Placed in units of tau0, so the same misalignment reads the same on
        any pulse length."""
        for tau0 in (10.0, 30.0, 100.0):
            assert _misaligned_efficiency(0.4 * tau0, tau0) == pytest.approx(0.5)


class TestBird:
    def test_envelope_spans_the_measured_band(self):
        um = np.linspace(1.0, 3.2, 400)
        env = _bird_envelope(um)
        assert env.shape == um.shape
        assert np.all(np.isfinite(env))
        assert np.all(env >= 0.0)
        assert env.max() > 0.0

    def test_has_the_water_absorption_notches(self):
        """The measured continuum dips at the 1.38 and 1.93 um H2O bands."""
        env = _bird_envelope(np.linspace(1.0, 3.2, 2201))
        um = np.linspace(1.0, 3.2, 2201)
        for notch in (1.38, 1.93):
            at_notch = env[np.argmin(np.abs(um - notch))]
            shoulder = max(env[np.argmin(np.abs(um - (notch - 0.12)))],
                           env[np.argmin(np.abs(um - (notch + 0.12)))])
            assert at_notch < shoulder, f"no dip at {notch} um"


class TestPresetRegistry:
    def test_default_pulse_is_registered(self):
        assert DEFAULT_PULSE in PULSE_SHAPES

    @pytest.mark.parametrize("key", sorted(PULSE_SHAPES))
    def test_every_preset_is_well_formed(self, key):
        preset = PULSE_SHAPES[key]
        assert preset["label"] and isinstance(preset["label"], str)
        assert preset["desc"] and isinstance(preset["desc"], str)
        assert callable(preset["build"])

    @pytest.mark.parametrize("key", sorted(PULSE_SHAPES))
    def test_every_preset_builds_a_finite_field_and_acquires(self, key):
        """The self-test's loop, turned into assertions: every beam in the
        picker must produce a finite field and a non-empty column."""
        stage = SimulatedStage()
        spec = SimulatedSpectrometer(stage, gate="shg", pulse=key)
        stage.move_to(0.0)
        col = spec.acquire()
        assert col.size == spec.n_pixels
        assert np.all(np.isfinite(col))
        assert col.max() > 0.0, f"{key} produced no signal at zero delay"
        assert spec.wavelengths[0] < spec.wavelengths[-1]
        assert spec.suggested_delay_fs > 0.0

    @pytest.mark.parametrize("key", sorted(PULSE_SHAPES))
    def test_named_spectrometer_pairs_exist(self, key):
        """A preset that names its own pair of instruments must name real
        entries in SIM_SPECTROMETERS, or the Simulation window hands the
        operator two dead devices."""
        for model in PULSE_SHAPES[key].get("spectrometers", ()):
            assert model in SIM_SPECTROMETERS


class TestSpectrometerPairSelection:
    def test_bird_is_measured_on_the_ocean_pair(self):
        assert sim_spectrometer_pair("bird") == ("ocean_sr", "ocean_nirquest")

    def test_pg_gate_never_claims_a_named_pair(self):
        """Under PG the signal sits at the fundamental — for bird that is
        1.0-3.2 um, past the red end of any silicon or InGaAs device."""
        assert sim_spectrometer_pair("bird", "pg") is None

    def test_a_preset_without_a_pair_returns_none(self):
        assert sim_spectrometer_pair("tl_sech") is None
