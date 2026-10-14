"""StitchedSpectrometer: geometric overlap, the crossfade band, and the
dark-aware gain fit.

Ported from hardware.py's self-test (lines 2743-2883). The `_Half` double is
carried over deliberately: two windows onto ONE analytic spectrum with a KNOWN
gain between them give the fit an exact answer to be checked against, which a
pair of SimulatedSpectrometers — whose ratio falls out of the beam and two
detector responses — could only be compared with approximately.

That double is also the proof that StitchedSpectrometer is vendor-neutral AND
simulator-neutral, which is why it belongs with the interfaces rather than under
either devices/ or sim/.
"""

from __future__ import annotations

import numpy as np
import pytest

from lillypad.hardware import (OVERLAP_BAND_FRAC, SpectrometerBase,
                               StitchedSpectrometer)
from lillypad.sim import (SIM_SPECTROMETERS, SimulatedSpectrometer,
                          SimulatedStage, sim_spectrometer_pair)


def _true_spectrum(wl):
    return 4000.0 * np.exp(-0.5 * ((np.asarray(wl, float) - 950.0) / 120.0) ** 2)


class _Half(SpectrometerBase):
    """One half of the band, with a settable gain and dark pedestal."""

    def __init__(self, lo, hi, gain=1.0, dark=0.0, n=512):
        self._w = np.linspace(lo, hi, n)
        self._gain, self._dark = gain, dark
        self.name = f"half {lo:.0f}-{hi:.0f}"
        self.max_counts = 65535.0

    @property
    def wavelengths(self):
        return self._w

    def set_integration_time(self, ms):
        self.integration_ms = float(ms)

    def acquire(self):
        return _true_spectrum(self._w) * self._gain + self._dark


@pytest.fixture
def pair():
    return StitchedSpectrometer(_Half(700.0, 1000.0, gain=1.0),
                                _Half(900.0, 1200.0, gain=4.0))


class TestOverlapGeometry:
    def test_geometric_overlap_is_the_shared_span(self, pair):
        glo, ghi = pair.geometric_overlap
        assert glo == pytest.approx(900.0, abs=1.0)
        assert ghi == pytest.approx(1000.0, abs=1.0)

    def test_default_band_is_the_central_fraction_of_it(self, pair):
        glo, ghi = pair.geometric_overlap
        blo, bhi = pair.overlap_band
        assert (bhi - blo) == pytest.approx((ghi - glo) * OVERLAP_BAND_FRAC,
                                            abs=1e-6)

    def test_default_band_is_centred(self, pair):
        """The outer 5% at each end, where both detectors sit at their noise
        floor, is excluded until the operator says otherwise."""
        glo, ghi = pair.geometric_overlap
        blo, bhi = pair.overlap_band
        assert (blo - glo) == pytest.approx(ghi - bhi, abs=1e-6)


class TestSetBand:
    def test_refuses_a_band_it_cannot_honour(self, pair):
        """Refused rather than silently clamped: a band quietly moved somewhere
        else is a fit the operator cannot reason about."""
        glo, ghi = pair.geometric_overlap
        blo, bhi = pair.overlap_band
        for bad in ((glo - 50.0, bhi), (blo, ghi + 50.0), (bhi, blo)):
            with pytest.raises(RuntimeError):
                pair.set_band(*bad)

    def test_a_refused_band_changes_nothing(self, pair):
        before = pair.overlap_band
        glo, _ = pair.geometric_overlap
        with pytest.raises(RuntimeError):
            pair.set_band(glo - 50.0, before[1])
        assert pair.overlap_band == before

    def test_reset_band_restores_the_default(self, pair):
        default = pair.overlap_band
        glo, ghi = pair.geometric_overlap
        pair.set_band(glo + 10.0, ghi - 10.0)
        assert pair.overlap_band != default
        pair.reset_band()
        assert pair.overlap_band == pytest.approx(default)


class TestGainFit:
    def test_recovers_the_known_gain_ratio(self, pair):
        assert pair.fit_stitch_factor() == pytest.approx(4.0, abs=0.4)

    def test_reports_a_small_residual(self, pair):
        pair.fit_stitch_factor()
        assert pair.stitch_residual is not None
        assert pair.stitch_residual < 0.05

    def test_crossfade_leaves_no_seam(self, pair):
        """The biggest jump between neighbouring pixels must stay in scale with
        the curve's own gradient. The old hard 50/50 average failed this at both
        seams."""
        pair.fit_stitch_factor()
        merged = pair.acquire()
        step = np.max(np.abs(np.diff(merged)))
        assert step < 0.05 * merged.max(), \
            f"seam step {step:.3g} vs peak {merged.max():.3g}"

    def test_only_one_member_contributes_outside_the_band(self, pair):
        """Below the band only spec1, above it only spec2 — otherwise the
        crossfade is leaking a detector into a region it cannot see."""
        assert pair._w1[pair._wl < pair.band_lo_nm].min() == 1.0
        assert pair._w1[pair._wl > pair.band_hi_nm].max() == 0.0


class TestDarkAwareFit:
    """A pedestal on the long-exposure member wrecks an undarkened fit and is
    harmless once the dark is passed — the reason fit_stitch_factor takes one."""

    @pytest.fixture
    def pedestalled(self):
        return StitchedSpectrometer(_Half(700.0, 1000.0, gain=1.0),
                                    _Half(900.0, 1200.0, gain=4.0, dark=500.0))

    def test_dark_subtraction_improves_the_fit(self, pedestalled):
        darks = (np.zeros(512), np.full(512, 500.0))
        naive = pedestalled.fit_stitch_factor()
        darked = pedestalled.fit_stitch_factor(darks)
        assert abs(darked - 4.0) < abs(naive - 4.0)
        assert darked == pytest.approx(4.0, abs=0.4)

    def test_a_mismatched_dark_is_refused_not_broadcast(self, pedestalled):
        with pytest.raises(RuntimeError):
            pedestalled.fit_stitch_factor((np.zeros(8), np.zeros(8)))


class TestBirdOnTheOceanPair:
    """A beam measured on its own pair of real instruments.

    Uncalibrated, one scalar cannot reconcile a silicon roll-off with an InGaAs
    cut-on across the same 150 nm — which is the whole reason the devices carry
    calibration files.
    """

    @pytest.fixture
    def bench(self):
        models = sim_spectrometer_pair("bird")
        devices = []
        for key in models:
            model = SIM_SPECTROMETERS[key]
            dev = SimulatedSpectrometer(SimulatedStage(), gate="shg",
                                        pulse="bird", **model["device"])
            dev.name = model["name"]
            devices.append(dev)
        return models, devices

    def test_the_pair_is_a_silicon_and_an_ingaas_device(self, bench):
        _, (sr, nq) = bench
        assert sr.n_pixels == 2048
        assert nq.n_pixels == 512

    def test_detector_response_really_costs_signal(self):
        """Not normalised away: at 1050 nm the silicon unit is ~17x less
        sensitive than at its peak, so its reddest pixels sit near the dark
        whatever the light is doing there."""
        common = dict(gate="shg", pulse="bird", wl_start=1040.0, wl_end=1050.0,
                      n_pixels=32, peak_counts=25000.0, read_noise=0.0)
        ideal = SimulatedSpectrometer(SimulatedStage(), **common)
        deaf = SimulatedSpectrometer(
            SimulatedStage(),
            response=SIM_SPECTROMETERS["ocean_sr"]["device"]["response"],
            **common)
        assert deaf.acquire().max() < 0.2 * ideal.acquire().max()

    def test_overlap_spans_the_silicon_ingaas_crossover(self, bench):
        _, (sr, nq) = bench
        glo, ghi = StitchedSpectrometer(sr, nq).geometric_overlap
        assert glo == pytest.approx(900.0, abs=1.0)
        assert ghi == pytest.approx(1050.0, abs=1.0)

    def test_uncalibrated_fit_leaves_a_large_residual(self, bench):
        _, (sr, nq) = bench
        pair = StitchedSpectrometer(sr, nq)
        pair.fit_stitch_factor()
        assert pair.stitch_residual > 0.2

    def test_calibration_at_least_halves_the_residual(self, bench, calibration_dir):
        models, devices = bench
        files = {k: calibration_dir / f"{SIM_SPECTROMETERS[k]['calibration']}.txt"
                 for k in models}
        missing = [f.name for f in files.values() if not f.is_file()]
        if missing:
            pytest.skip(f"calibration files not on disk: {missing}")

        pair = StitchedSpectrometer(*devices)
        pair.fit_stitch_factor()
        raw_residual = pair.stitch_residual

        for dev, key in zip(devices, models):
            dev.set_calibration(files[key])
        pair.fit_stitch_factor()
        assert pair.stitch_residual < 0.5 * raw_residual
