"""Dark correction: what gets subtracted, from what, and what never does.

The instrument this runs on shows a ~2245-count pedestal against ~489 counts of
signal, so "is this dark-subtracted" is not a detail — an uncorrected trace is
mostly pedestal. These tests pin the three answers:

  * .dwc and .csv are corrected with the RECORDED DARK, the same frame the
    Spectrum panel uses
  * the RAW alignment view is dark-corrected at MEMBER level, before either
    calibration or the stitch, because that is the only place the pedestal can
    come off without touching what the view exists to show
  * the delay-extreme reference is archived and NEVER subtracted
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from lillypad.scan import FrogScanConfig, write_dwc, write_npz
from lillypad.scan.model import FrogResult
from lillypad.scan.worker import FrogScanWorker
from lillypad.sim import SimulatedSpectrometer, SimulatedStage

N_PX = 128


def run(dark=None, take_reference=False, seed=11):
    np.random.seed(seed)
    stage = SimulatedStage(travel_mm=300.0)
    spec = SimulatedSpectrometer(stage, gate="shg", n_pixels=N_PX)
    cfg = FrogScanConfig(delay_start_fs=-20, delay_stop_fs=20, delay_step_fs=10.0,
                         zero_pos_um=150000.0,
                         capture_delay_reference=take_reference)
    out = {}
    w = FrogScanWorker(stage, spec, cfg, dark=dark)
    w.finished_scan.connect(lambda r: out.__setitem__("r", r))
    w.error.connect(lambda m: out.__setitem__("err", m))
    w.run()
    assert "err" not in out, out.get("err")
    return out["r"]


class TestTheScanCarriesTheDark:
    def test_the_dark_passed_in_is_what_the_result_reports(self):
        d = np.linspace(10.0, 90.0, N_PX)
        r = run(dark=d)
        assert r.dark_subtracted
        assert np.array_equal(r.dark, d)

    def test_no_dark_means_no_correction_and_it_says_so(self):
        r = run(dark=None)
        assert not r.dark_subtracted
        assert np.array_equal(r.corrected_trace(), r.trace)

    def test_the_worker_never_measures_its_own_dark(self):
        """The dark is passed IN. A scan that recorded its own would be a second,
        silently different notion of the same word — which is exactly what the
        old bracketed 'background' was."""
        r = run(dark=None, take_reference=True)
        assert r.dark is None
        assert r.delay_reference() is not None


class TestExportsUseTheDark:
    def test_dwc_is_the_trace_minus_the_dark(self, tmp_path):
        d = np.full(N_PX, 120.0)
        r = run(dark=d)
        write_dwc(tmp_path / "t.dwc", r)
        txt = (tmp_path / "t.dwc").read_text(encoding="ascii").splitlines()
        i = txt.index("[Data array]")
        dwc = np.array([[float(v) for v in x.split("\t")]
                        for x in txt[i + 1:] if x.strip()])
        assert np.allclose(dwc, r.corrected_trace(), rtol=1e-5)
        assert dwc.min() < r.trace.min() - 50, "the pedestal is still there"

    def test_dwc_is_raw_when_no_dark_was_subtracted(self, tmp_path):
        r = run(dark=None)
        write_dwc(tmp_path / "t.dwc", r)
        txt = (tmp_path / "t.dwc").read_text(encoding="ascii").splitlines()
        i = txt.index("[Data array]")
        dwc = np.array([[float(v) for v in x.split("\t")]
                        for x in txt[i + 1:] if x.strip()])
        assert np.allclose(dwc, r.trace, rtol=1e-5)

    def test_the_npz_can_reproduce_the_dwc(self, tmp_path):
        """trace - dark is exactly what the text formats contain, so an archive
        is enough to recreate either form."""
        r = run(dark=np.full(N_PX, 120.0))
        write_npz(tmp_path / "t.npz", r)
        with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
            assert np.allclose(z["trace"] - z["dark"][:, None],
                               r.corrected_trace())

    def test_the_metadata_records_which_it_was(self, tmp_path):
        for dark, expected in ((np.full(N_PX, 5.0), True), (None, False)):
            r = run(dark=dark)
            write_npz(tmp_path / "t.npz", r)
            with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
                meta = ast.literal_eval(str(z["metadata"]))
            assert meta["dark_subtracted"] is expected


class TestTheDelayReferenceIsNeverSubtracted:
    def test_it_is_off_by_default(self):
        assert FrogScanConfig.capture_delay_reference is False
        assert run().delay_reference() is None

    def test_taking_one_does_not_change_the_exported_trace(self):
        """The reference is provenance. Two scans differing only in whether it
        was taken must export the same correction."""
        d = np.full(N_PX, 30.0)
        a = run(dark=d, take_reference=False, seed=3)
        b = run(dark=d, take_reference=True, seed=3)
        assert np.array_equal(a.dark, b.dark)
        assert np.array_equal(a.corrected_trace() - a.trace,
                              b.corrected_trace() - b.trace)

    def test_it_is_archived_under_its_own_name(self, tmp_path):
        r = run(dark=np.full(N_PX, 5.0), take_reference=True)
        write_npz(tmp_path / "t.npz", r)
        with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
            assert "delay_reference_before" in z
            assert "delay_reference_after" in z
            assert "dark" in z
            # The old name meant something else and must not reappear.
            assert "background_before" not in z


class TestRawViewDarkCorrection:
    """The RAW alignment view, which the operator uses to judge whether an arm
    is really seeing the beam. It must lose the pedestal without gaining either
    calibration or the stitch."""

    @pytest.fixture
    def win(self):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from lillypad.design.theme import install_theme
        app = QApplication.instance() or QApplication([])
        install_theme(app, "dark")
        import frog_gui_fast
        w = frog_gui_fast.FrogWindow()
        yield w
        w.close()

    @staticmethod
    def _arm(win, member_dark):
        """Put the window into the state a running stitched scan leaves it in."""
        wl = np.linspace(500.0, 700.0, 64)
        win._scan_wl = wl
        win._align_wl = [wl, wl]
        win._align_cut = 32
        win._align_trace = np.zeros((wl.size, 2), np.float32)
        win._align_peak = [0.0, 0.0]
        win._align_view = None
        win._scan_member_dark = member_dark

    def test_the_member_dark_comes_off_each_member(self, win):
        frames = (np.full(64, 500.0), np.full(64, 900.0))
        darks = (np.full(64, 200.0), np.full(64, 800.0))
        self._arm(win, darks)
        win.alignment._store_align_column(0, frames)
        stored = np.asarray(win._align_trace[:, 0], float)
        # blue half: 500-200 = 300; red half: 900-800 = 100
        assert np.allclose(stored[:32], 300.0)
        assert np.allclose(stored[32:], 100.0)

    def test_without_a_dark_the_frames_are_stored_as_measured(self, win):
        frames = (np.full(64, 500.0), np.full(64, 900.0))
        self._arm(win, None)
        win.alignment._store_align_column(0, frames)
        stored = np.asarray(win._align_trace[:, 0], float)
        assert np.allclose(stored[:32], 500.0)
        assert np.allclose(stored[32:], 900.0)

    def test_a_mismatched_dark_is_ignored_rather_than_broadcast(self, win):
        """Better raw than wrong — the same rule the display correction uses."""
        frames = (np.full(64, 500.0), np.full(64, 900.0))
        self._arm(win, (np.full(8, 200.0), np.full(8, 800.0)))
        win.alignment._store_align_column(0, frames)
        stored = np.asarray(win._align_trace[:, 0], float)
        assert np.allclose(stored[:32], 500.0)

    def test_each_half_is_still_normalised_to_its_own_max(self, win):
        """The other half of what RAW is: no stitch factor, no calibration, two
        independent scales and a hard cut."""
        self._arm(win, (np.zeros(64), np.zeros(64)))
        win.alignment._store_align_column(0, (np.full(64, 400.0), np.full(64, 50.0)))
        view = win.alignment._align_trace_view()
        assert np.allclose(view[:32, 0], 1.0)
        assert np.allclose(view[32:, 0], 1.0)
