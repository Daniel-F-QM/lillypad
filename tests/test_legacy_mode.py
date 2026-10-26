"""Legacy mode: the same numbers the legacy acquisition program produces.

The oracle is the legacy program's own code. `_LegacyDevice` and
`_LegacyStitched` below are its SeabreezeSpectrometer and StitchedSpectrometer,
copied verbatim apart from the print() calls and the USB device, which is
swapped for the same deterministic double Lillypad is driven with. Most checks
are therefore exact (np.array_equal, ==), not approximate: legacy mode exists
so that any difference left between the two programs is a real one.

The GUI half at the bottom covers what only the window can do: the switch
reaching every device, the dark following the legacy rules, and the choice
being remembered.
"""

from __future__ import annotations

import numpy as np
import pytest

from lillypad.hardware import (SpectrometerBase, StitchedSpectrometer,
                               load_calibration_file,
                               load_calibration_file_legacy, legacy_trim_mask)
from lillypad.scan import FrogScanConfig
from lillypad.sim import SimulatedStage


# ── The double ───────────────────────────────────────────────────────────────
def _true_spectrum(wl):
    return 4000.0 * np.exp(-0.5 * ((np.asarray(wl, float) - 950.0) / 120.0) ** 2)


class _Dev(SpectrometerBase):
    """One analytic spectrum seen through a gain, on a pedestal, with
    `light` to block the beam for a dark. Deterministic, and a FRESH array on
    every acquire(): the legacy program averages by adding into the frame it
    is handed.

    The axis is slightly quadratic, like a real spectrometer's, so the legacy
    first-two-pixel grid step and Lillypad's median step really differ."""

    def __init__(self, lo, hi, n, gain=1.0, pedestal=0.0, name="dev"):
        i = np.arange(n, dtype=float)
        lin = (hi - lo) / (n - 1)
        self._w = lo + lin * i * 0.97 + 0.03 * lin * i ** 2 / (n - 1)
        self._gain, self._pedestal = gain, pedestal
        self.light = True
        self.name = name
        self.max_counts = 65535.0

    @property
    def wavelengths(self):
        return self._w

    def set_integration_time(self, ms):
        self.integration_ms = float(ms)

    def acquire(self):
        sig = _true_spectrum(self._w) * self._gain if self.light else 0.0
        return np.asarray(sig + self._pedestal + np.zeros_like(self._w), float)


# ── The oracle: the legacy program, verbatim ─────────────────────────────────
class _LegacyDevice:
    """The legacy program's SeabreezeSpectrometer, from open() down, with the
    seabreeze device replaced by a _Dev's raw frames."""

    def __init__(self, source):
        self._src = source
        self.wavelength_vec = np.asarray(source.wavelengths, float).copy()
        self.filter = np.ones_like(self.wavelength_vec,dtype=bool)
        self.calibration = np.ones_like(self.wavelength_vec)
        self.dark = np.zeros_like(self.wavelength_vec)
        self.navg = 1
        self.subtract_dark = False

    def _intensities(self):
        return np.array(self._src.acquire(), float)

    def wavelengths(self):
        return self.wavelength_vec[self.filter]

    def intensities(self):
        I = self._intensities()
        for i in range(self.navg-1):
            I+=self._intensities()
        I/=self.navg
        if self.subtract_dark:
            return ((I-self.dark)*self.calibration)[self.filter]
        else:
            return (I*self.calibration)[self.filter]

    def set_averages(self,navg):
        self.navg = navg

    def set_subtract_dark(self,subtract_dark):
        self.subtract_dark = subtract_dark

    def record_dark(self):
        self.dark = self._intensities()
        for i in range(self.navg-1):
            self.dark += self._intensities()
        self.dark/= self.navg

    def update_calibration(self,fname):
        wl,calfac = np.loadtxt(fname,unpack=True)
        isort = np.argsort(wl)
        wl = wl[isort]
        calfac= calfac[isort]

        self.calibration = np.interp(self.wavelength_vec,wl,calfac,right=0,left=0)
        calfac = self.calibration
        self.filter = _legacy_filter(calfac)


def _legacy_filter(calfac):
    """The trimming loops of SeabreezeSpectrometer.update_calibration."""
    for i,_ in enumerate(calfac):
        if calfac[i] == 0:
            continue
        else:
            break
    pad_left = i
    for i,_ in enumerate(calfac[::-1]):

        if calfac[-(i+1)] == 0:
            continue
        else:
            break
    pad_right = i+1
    filt = np.zeros_like(calfac,dtype=bool)
    filt[pad_left:-pad_right] = True
    return filt


class _LegacyStitched:
    """The legacy program's StitchedSpectrometer."""

    def __init__(self,spec1,spec2):
        self.spec1 = spec1
        self.spec2 = spec2
        self.wl1 = spec1.wavelengths()
        self.wl2 = spec2.wavelengths()
        if np.max(self.wl1)>np.max(self.wl2):
            self.wl2,self.wl1 = self.wl1,self.wl2
            self.spec1,self.spec2 = self.spec2,self.spec1

        if np.min(self.wl2)>np.max(self.wl1):
            raise Exception('Error: There is no overlap in the wl ranges of the selected spectrometers')

        wl1 = self.wl1
        wl2 = self.wl2
        dwl1 = wl1[1]-wl1[0]
        dwl2 = wl2[1]-wl2[0]

        dwl = dwl1 if dwl1<dwl2 else dwl2

        self.wl = np.arange(np.min(wl1),np.max(wl2),dwl)

        self.wlo1 = self.wl[ np.argmin( abs(self.wl-np.min(wl2)) ) ]
        self.wlo2 = self.wl[ np.argmin( abs(self.wl-np.max(wl1)) ) ]

        self.ovl_filt = (self.wl>self.wlo1) & (self.wl<self.wlo2) #filter for the spectral overlap
        self.wl1_filt = (self.wl<=self.wlo1)
        self.wl2_filt = (self.wl>=self.wlo2)
        self.stitch_factor = 1

    def intensities(self):
        I = np.zeros_like(self.wl)
        I1 = self.spec1.intensities()*self.stitch_factor
        I2 = self.spec2.intensities()
        I[self.wl1_filt] = np.interp(self.wl[self.wl1_filt],self.wl1,I1)
        I[self.wl2_filt] = np.interp(self.wl[self.wl2_filt],self.wl2,I2)
        o = self.ovl_filt
        I[o] = 0.5*( np.interp(self.wl[o],self.wl1,I1) + np.interp(self.wl[o],self.wl2,I2) )

        return I

    def fit_stitch_factor(self):
        from scipy.optimize import minimize_scalar
        I1 = self.spec1.intensities()
        I2 = self.spec2.intensities()
        I1 = np.interp(self.wl[self.ovl_filt],self.wl1,I1)
        I2 = np.interp(self.wl[self.ovl_filt],self.wl2,I2)
        minfunc = lambda s: np.sum(np.abs(I1*s-I2)**2)
        res = minimize_scalar(minfunc)
        if res.success:
            self.stitch_factor = res.x
        return self.stitch_factor


# ── Calibration files ────────────────────────────────────────────────────────
def _write(path, wl, fac):
    np.savetxt(path, np.column_stack([wl, fac]))
    return path


@pytest.fixture
def cals(tmp_path):
    """Calibration files shaped like the bench's: zero-padded where a detector
    dies, ramps in between. `b` has no zero pad at its red end, so only
    the legacy off-by-one trims there."""
    wl_a = np.arange(650.0, 1050.0, 0.7)
    fac_a = 1.0 + 0.002 * (wl_a - 850.0)
    fac_a[(wl_a < 720.0) | (wl_a > 985.0)] = 0.0
    wl_b = np.arange(880.0, 1250.0, 0.7)
    fac_b = 0.25 * (1.0 - 0.001 * (wl_b - 1000.0))
    fac_b[wl_b < 915.0] = 0.0
    wl_n = np.arange(750.0, 950.0, 0.7)
    a2 = fac_a.copy()
    a2[wl_a > 960.0] = 0.0
    return {
        "a": _write(tmp_path / "cal_a.txt", wl_a, fac_a),
        "a2": _write(tmp_path / "cal_a2.txt", wl_a, a2),
        "b": _write(tmp_path / "cal_b.txt", wl_b, fac_b),
        "narrow": _write(tmp_path / "cal_narrow.txt", wl_n,
                         np.ones_like(wl_n)),
    }


def _blue(**kw):
    return _Dev(700.0, 1000.0, 512, gain=1.0, name="blue", **kw)


def _red(**kw):
    return _Dev(900.0, 1200.0, 256, gain=4.0, name="red", **kw)


# ── The strict reader ────────────────────────────────────────────────────────
class TestStrictReader:
    def test_reads_what_np_loadtxt_reads(self, cals):
        wl, fac = load_calibration_file_legacy(cals["b"])
        ref_wl, ref_fac = np.loadtxt(cals["b"], unpack=True)
        order = np.argsort(ref_wl)
        assert np.array_equal(wl, ref_wl[order])
        assert np.array_equal(fac, ref_fac[order])

    def test_one_ragged_line_rejects_the_whole_file(self, tmp_path):
        """The exact shape the legacy program's Niquest_New.txt has. Lillypad
        skips the line; the legacy program — and legacy mode — refuse the
        file."""
        p = tmp_path / "ragged.txt"
        p.write_text("400 0.5\n500 1.0\n600 0.25\n 0")
        assert load_calibration_file(p)[2] == 1
        with pytest.raises(RuntimeError):
            load_calibration_file_legacy(p)

    def test_a_third_column_is_refused(self, tmp_path):
        p = tmp_path / "three.txt"
        p.write_text("400 0.5 1\n500 1.0 1\n600 0.25 1\n")
        with pytest.raises(RuntimeError):
            load_calibration_file_legacy(p)

    def test_shipped_files_read_as_the_legacy_program_reads_them(
            self, calibration_dir):
        files = sorted(calibration_dir.glob("*.txt"))
        if not files:
            pytest.skip(f"no calibration files in {calibration_dir}")
        for f in files:
            try:
                ref = np.loadtxt(f, unpack=True)
            except Exception:
                with pytest.raises(RuntimeError):
                    load_calibration_file_legacy(f)
                continue
            wl, fac = load_calibration_file_legacy(f)
            order = np.argsort(ref[0])
            assert np.array_equal(wl, ref[0][order]), f.name
            assert np.array_equal(fac, ref[1][order]), f.name


# ── Legacy trimming ──────────────────────────────────────────────────────────
class TestTrimMask:
    CASES = {
        "zero pads both ends": [0, 0, 1, 2, 3, 4, 0, 0],
        "no zero pad": [1, 2, 3, 4, 5],
        "zero pad left only": [0, 0, 0, 1, 2, 3],
        "zero pad right only": [1, 2, 3, 0, 0],
        "interior zeros kept": [0, 1, 0, 0, 2, 3, 0],
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_matches_legacy(self, name):
        cal = np.asarray(self.CASES[name], float)
        assert np.array_equal(legacy_trim_mask(cal), _legacy_filter(cal))

    def test_the_red_end_loses_one_non_zero_pixel_too(self):
        """The legacy off-by-one, kept on purpose: the 3 is dropped with the 0."""
        assert legacy_trim_mask([0, 0, 1, 2, 3, 0]).tolist() == \
            [False, False, True, True, False, False]

    def test_without_a_zero_pad_the_last_pixel_still_goes(self):
        assert legacy_trim_mask([1, 2, 3, 4]).tolist() == [True, True, True,
                                                          False]

    def test_an_all_zero_calibration_is_refused(self):
        with pytest.raises(RuntimeError):
            legacy_trim_mask([0, 0, 0])


# ── One spectrometer ─────────────────────────────────────────────────────────
class TestSingleDevice:
    @pytest.fixture
    def both(self, cals):
        dev = _blue(pedestal=300.0)
        dev.set_processing("legacy")
        dev.set_calibration(cals["a"])
        ref = _LegacyDevice(dev)
        ref.update_calibration(cals["a"])
        return dev, ref

    def test_the_axis_matches_legacy(self, both):
        dev, ref = both
        assert np.array_equal(dev.output_wavelengths, ref.wavelengths())
        assert dev.output_wavelengths.size < dev.wavelengths.size

    def test_a_calibrated_frame_matches_legacy_bit_for_bit(self, both):
        dev, ref = both
        assert np.array_equal(dev.calibrate(dev.acquire()), ref.intensities())

    def test_a_dark_subtracted_average_matches_legacy(self, both):
        dev, ref = both
        n = 4
        ref.set_averages(n)
        dev.light = False
        ref.record_dark()
        dark = dev.acquire()
        for _ in range(n - 1):
            dark += dev.acquire()
        dark /= n
        dev.light = True
        ref.set_subtract_dark(True)
        acc = dev.acquire()             # the scan worker's arithmetic
        for _ in range(n - 1):
            acc = acc + dev.acquire()
        assert np.array_equal(dev.calibrate(acc / n - dark), ref.intensities())

    def test_pixels_outside_the_file_are_zeroed_and_dropped(self, cals):
        dev = _blue()
        dev.set_processing("legacy")
        dev.set_calibration(cals["narrow"])
        wl = dev.output_wavelengths
        assert wl.min() >= 750.0 and wl.max() <= 950.0

    def test_lillypad_mode_is_unchanged(self, cals):
        dev = _blue()
        dev.set_calibration(cals["a"])
        assert dev.processing == "lillypad"
        assert dev.output_wavelengths is dev.wavelengths
        wl, fac, _ = load_calibration_file(cals["a"])
        raw = dev.acquire()
        assert np.array_equal(dev.calibrate(raw),
                              raw * np.interp(dev.wavelengths, wl, fac))

    def test_switching_method_rereads_the_file(self, cals):
        dev = _blue()
        dev.set_calibration(cals["a"])
        lilly = dev.calibrate(dev.acquire())
        dev.set_processing("legacy")
        fresh = _blue()
        fresh.set_processing("legacy")
        fresh.set_calibration(cals["a"])
        assert np.array_equal(dev.calibrate(dev.acquire()),
                              fresh.calibrate(fresh.acquire()))
        dev.set_processing("lillypad")
        assert np.array_equal(dev.calibrate(dev.acquire()), lilly)

    def test_a_file_legacy_mode_cannot_read_refuses_the_switch(self, tmp_path):
        p = tmp_path / "ragged.txt"
        p.write_text("690 1.0\n800 2.0\n1010 0.5\n 0")
        dev = _blue()
        dev.set_calibration(p)
        before = dev.calibrate(dev.acquire())
        with pytest.raises(RuntimeError):
            dev.set_processing("legacy")
        assert dev.processing == "lillypad"
        assert np.array_equal(dev.calibrate(dev.acquire()), before)

    def test_a_scan_records_the_trimmed_axis(self, both):
        """The scan worker sizes its trace from the CALIBRATED axis."""
        from lillypad.scan.worker import FrogScanWorker
        dev, _ = both
        stage = SimulatedStage(travel_mm=300.0)
        cfg = FrogScanConfig(delay_start_fs=-12.0, delay_stop_fs=12.0,
                             delay_step_fs=6.0, zero_pos_um=150000.0,
                             processing="legacy")
        out = {}
        worker = FrogScanWorker(stage, dev, cfg)
        worker.finished_scan.connect(lambda r: out.__setitem__("r", r))
        worker.error.connect(lambda m: out.__setitem__("err", m))
        worker.run()
        assert "err" not in out, out.get("err")
        r = out["r"]
        assert np.array_equal(r.wavelengths_nm, dev.output_wavelengths)
        assert np.array_equal(r.trace[:, 0], dev.calibrate(dev.acquire()))
        assert r.metadata()["processing"] == "legacy"


# ── A stitched pair ──────────────────────────────────────────────────────────
class TestStitchedPair:
    @pytest.fixture
    def both(self, cals):
        a, b = _blue(pedestal=300.0), _red(pedestal=1200.0)
        a.set_calibration(cals["a"])          # read in Lillypad mode first:
        b.set_calibration(cals["b"])          # the pair re-reads them
        pair = StitchedSpectrometer(a, b, method="legacy")
        pa, pb = _LegacyDevice(a), _LegacyDevice(b)
        pa.update_calibration(cals["a"])
        pb.update_calibration(cals["b"])
        return pair, _LegacyStitched(pa, pb)

    def test_the_members_follow_the_pair(self, both):
        pair, _ = both
        assert all(m.processing == "legacy" for m in pair.members)

    def test_the_grid_matches_legacy(self, both):
        pair, ref = both
        assert np.array_equal(pair.wavelengths, ref.wl)

    def test_the_overlap_matches_legacy(self, both):
        pair, ref = both
        assert pair.overlap_band == (ref.wlo1, ref.wlo2)
        assert np.array_equal(pair._ovl, ref.ovl_filt)
        assert np.array_equal(pair._m1, ref.wl1_filt)
        assert np.array_equal(pair._m2, ref.wl2_filt)

    def test_a_merged_frame_matches_legacy_bit_for_bit(self, both):
        pair, ref = both
        pair.stitch_factor = ref.stitch_factor = 2.5
        assert np.array_equal(pair.acquire(), ref.intensities())

    def test_the_fit_matches_legacy(self, both):
        pair, ref = both
        ref.spec1.set_averages(3)
        ref.spec2.set_averages(3)
        assert pair.fit_stitch_factor(n_frames=3) == ref.fit_stitch_factor()

    def test_the_fit_with_a_dark_matches_legacy(self, both):
        pair, ref = both
        for m in pair.members:
            m.light = False
        darks = tuple(m.acquire() for m in pair.members)
        for p in (ref.spec1, ref.spec2):
            p.record_dark()
            p.set_subtract_dark(True)
        for m in pair.members:
            m.light = True
        assert pair.fit_stitch_factor(darks) == ref.fit_stitch_factor()

    def test_the_merged_dark_corrected_frame_matches_legacy(self, both):
        """What the scan exports: merged frame minus the merged dark."""
        pair, ref = both
        for m in pair.members:
            m.light = False
        darks = tuple(m.acquire() for m in pair.members)
        for p in (ref.spec1, ref.spec2):
            p.record_dark()
            p.set_subtract_dark(True)
        for m in pair.members:
            m.light = True
        pair.stitch_factor = ref.stitch_factor = 1.3
        merged = pair.acquire() - pair.combined_dark(darks)
        assert np.allclose(merged, ref.intensities(), rtol=0, atol=1e-9)

    def test_there_is_no_band_to_set(self, both):
        pair, _ = both
        lo, hi = pair.geometric_overlap
        with pytest.raises(RuntimeError):
            pair.set_band(lo + 1.0, hi - 1.0)
        assert pair.reset_band() == pair.overlap_band

    def test_negative_values_are_kept(self, both):
        pair, _ = both
        raw = tuple(m.acquire() for m in pair.members)
        too_much = tuple(r + 1000.0 for r in raw)
        assert min(x.min() for x in pair.prepare_pair(*raw, too_much)) < 0
        pair.set_processing("lillypad")
        assert min(x.min() for x in pair.prepare_pair(*raw, too_much)) >= 0

    def test_a_round_trip_restores_lillypad(self, cals):
        a, b = _blue(), _red()
        a.set_calibration(cals["a"])
        b.set_calibration(cals["b"])
        pair = StitchedSpectrometer(a, b)
        wl, band, frame = pair.wavelengths, pair.overlap_band, pair.acquire()
        pair.set_processing("legacy")
        assert pair.wavelengths.size != wl.size
        pair.set_processing("lillypad")
        assert np.array_equal(pair.wavelengths, wl)
        assert pair.overlap_band == band
        assert np.array_equal(pair.acquire(), frame)

    def test_a_switch_that_cannot_stitch_changes_nothing(self, tmp_path):
        """Trimmed to their files, these two no longer overlap; with edges
        held (Lillypad) they still do."""
        wl = np.arange(600.0, 1300.0, 0.7)
        lo = np.where(wl < 850.0, 1.0, 0.0)
        hi = np.where(wl > 950.0, 1.0, 0.0)
        a, b = _blue(), _red()
        a.set_calibration(_write(tmp_path / "lo.txt", wl, lo))
        b.set_calibration(_write(tmp_path / "hi.txt", wl, hi))
        pair = StitchedSpectrometer(a, b)
        grid = pair.wavelengths
        with pytest.raises(RuntimeError):
            pair.set_processing("legacy")
        assert pair.processing == "lillypad"
        assert all(m.processing == "lillypad" for m in (a, b))
        assert np.array_equal(pair.wavelengths, grid)
        assert a.output_wavelengths is a.wavelengths

    def test_a_new_file_moves_the_grid(self, both, cals):
        pair, _ = both
        a = next(m for m in pair.members if m.name == "blue")
        a.set_calibration(cals["a2"])
        assert pair.relayout()
        pa, pb = _LegacyDevice(a), _LegacyDevice(pair.spec2)
        pa.update_calibration(cals["a2"])
        pb.update_calibration(cals["b"])
        ref = _LegacyStitched(pa, pb)
        assert np.array_equal(pair.wavelengths, ref.wl)
        assert pair.overlap_band == (ref.wlo1, ref.wlo2)

    def test_lillypad_relayout_is_a_no_op(self, cals):
        a, b = _blue(), _red()
        pair = StitchedSpectrometer(a, b)
        lo, hi = pair.geometric_overlap
        pair.set_band(lo + 5.0, hi - 5.0)
        a.set_calibration(cals["a"])
        assert not pair.relayout()
        assert pair.overlap_band == (lo + 5.0, hi - 5.0)


# ── The window ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def window():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from lillypad.design.theme import install_theme
    app = QApplication.instance() or QApplication([])
    install_theme(app, "dark")
    from lillypad.gui.window import FrogWindow
    win = FrogWindow()
    avg = win.dlg_settings.spin_avg.value()
    yield win
    # Put back what these tests touch before closing. The window saves
    # settings.json on the way out — into conftest's throwaway file, but a
    # later window in the same run reads that file too.
    win.devices.set_processing("lillypad")
    win.devices.disconnect_spectrometer()
    win.dlg_settings.spin_avg.setValue(avg)
    win.close()


@pytest.fixture
def win(window):
    yield window
    window.devices.set_processing("lillypad")
    window.devices.disconnect_spectrometer()


class TestTheWindow:
    def test_the_switch_reaches_a_connected_pair(self, win):
        assert win.devices.use_sim_pair()[0]
        ok, msg = win.devices.set_processing("legacy")
        assert ok, msg
        assert win.processing == "legacy"
        assert win.spec.processing == "legacy"
        assert all(m.processing == "legacy" for m in win.spec.members)

    def test_new_devices_open_on_the_method_in_force(self, win):
        assert win.devices.set_processing("legacy")[0]
        assert win.devices.use_sim_pair()[0]
        assert all(m.processing == "legacy" for m in win.spec.members)
        assert win.devices.use_sim_spectrometer()[0]
        assert win.spec.processing == "legacy"

    def test_an_unknown_method_is_refused(self, win):
        ok, _ = win.devices.set_processing("nonsense")
        assert not ok
        assert win.processing == "lillypad"

    def test_the_dark_takes_one_scan_points_frames(self, win):
        win.dlg_settings.spin_avg.setValue(3)
        assert win.devices.set_processing("legacy")[0]
        assert win.feed._dark_frame_count() == 3

    def test_the_dark_survives_an_exposure_change_only_in_legacy_mode(self, win):
        assert win.devices.use_sim_spectrometer()[0]
        win.dlg_settings.spin_avg.setValue(2)
        for mode, kept in (("legacy", True), ("lillypad", False)):
            assert win.devices.set_processing(mode)[0]
            win.feed._capture_dark()
            assert win.background is not None
            win.spin_integration.setValue(win.spin_integration.value() + 5)
            win.feed._apply_integration_time()
            assert (win.background is not None) is kept, mode

    def test_back_to_lillypad_drops_a_dark_from_another_exposure(self, win):
        assert win.devices.use_sim_spectrometer()[0]
        win.dlg_settings.spin_avg.setValue(2)
        assert win.devices.set_processing("legacy")[0]
        win.feed._capture_dark()
        win.spin_integration.setValue(win.spin_integration.value() + 5)
        win.feed._apply_integration_time()
        assert win.background is not None
        assert win.devices.set_processing("lillypad")[0]
        assert win.background is None

    def test_a_raw_dark_survives_the_switch_itself(self, win):
        assert win.devices.use_sim_pair()[0]
        win.dlg_settings.spin_avg.setValue(2)
        win.feed._capture_dark()
        assert win.background_members is not None
        assert win.devices.set_processing("legacy")[0]
        assert win.background_members is not None

    def test_the_choice_is_remembered(self, win):
        assert win.devices.set_processing("legacy")[0]
        assert win.settings_ctl._settings_snapshot()["processing"] == "legacy"
        win.settings_ctl._restore_processing("lillypad")
        assert win.processing == "lillypad"
        with pytest.raises(ValueError):
            win.settings_ctl._restore_processing("nonsense")

    def test_the_dialog_follows_the_method(self, win):
        assert win.devices.use_sim_pair()[0]
        assert win.devices.set_processing("legacy")[0]
        win.dlg_spec.refresh()
        assert win.dlg_spec.cmb_method.currentData() == "legacy"
        assert not win.dlg_spec.btn_band.isEnabled()
        assert win.dlg_spec.btn_fit.isEnabled()
        assert win.devices.set_processing("lillypad")[0]
        win.dlg_spec.refresh()
        assert win.dlg_spec.cmb_method.currentData() == "lillypad"
        assert win.dlg_spec.btn_band.isEnabled()

    def test_auto_stitch_runs_in_legacy_mode(self, win):
        assert win.devices.use_sim_pair()[0]
        assert win.devices.set_processing("legacy")[0]
        win.dlg_settings.spin_avg.setValue(2)
        ok, msg = win.devices.fit_stitch_factor()
        assert ok, msg
        assert "average of 2 frames" in msg
