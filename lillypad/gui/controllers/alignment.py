"""The three diagnostic views, none of which reach a saved file.

Worth saying first, because it is the invariant that makes this controller safe
to read: **nothing here touches a FrogResult or any export.** These are three
ways of looking at what is on the bench right now.

  * **The four-point symmetry sweep** (the Δ button). Measures at −2x, −x, +x
    and +2x from where the stage is, plots S(+x)−S(−x) and S(+2x)−S(−2x) on
    their own panel — a symmetric pulse gives two flat curves on zero — and
    puts the stage back. It has no abort, deliberately: it is four points long
    and its last act is the one that must not be cut short.
  * **The RAW stitched trace.** The uncalibrated, unstitched counterpart of the
    scan trace for a pair: the two members interpolated onto the common grid and
    hard-cut, each half normalized to its own running maximum. Dark-corrected at
    MEMBER level, because that is the only place the pedestal can come off
    without bringing the calibration or the crossfade with it.
  * **Symmetry mode.** The finished trace folded about the most symmetric axis
    near its autocorrelation peak (in half-column steps), or about a fold line
    the operator drags.

The last two are mutually exclusive, and the reason is mechanical rather than a
policy: both rewrite the same image.
"""

import numpy as np

from lillypad.gui.text import FEED_BUSY_MSG
from lillypad.gui.workers import AlignmentWorker
from lillypad.hardware import StitchedSpectrometer
from lillypad.scan import (delay_to_position_um, stage_to_um, symmetry_fold,
                           symmetry_view)


class AlignmentController:
    """The alignment sweep, the RAW stitched view, and symmetry mode."""

    def __init__(self, win):
        self.win = win

    # ── Alignment mode: spectrum symmetry ────────────────────────────────────
    def _align_running(self):
        return (self.win._align_worker is not None
                and self.win._align_worker.isRunning())

    def _uncheck_align_spec(self):
        """Pop the Δ button back out without re-entering its own handler."""
        self.win.btn_align_spec.blockSignals(True)
        self.win.btn_align_spec.setChecked(False)
        self.win.btn_align_spec.blockSignals(False)

    def _close_align_mode(self):
        """Leave alignment mode: panel closed, curves dropped, button out.

        The one way out, so every caller that has to abandon the mode — the
        toggle, a device swap, the start of a scan — leaves the same state
        behind. Silent about signals: _uncheck_align_spec blocks them, so this
        never re-enters the toggle handler that may have called it.
        """
        self._uncheck_align_spec()
        self.win.btn_align_refresh.hide()
        self.win.canvas.clear_diff()
        self.win.canvas.set_dev_visible(False)
        self.win.appearance._position_panel_buttons()   # the header row just lost a button

    def _on_align_spec_toggled(self, on):
        """Open or close alignment mode.

        Opening it puts the deviation panel on screen and runs the first sweep;
        the ⟳ button then re-runs it in place, so judging an adjustment no
        longer means toggling the mode off and on (which threw away the
        measurement you wanted to compare against).
        """
        if not on:
            self._close_align_mode()
            self.win.status.showMessage("Alignment mode off.", 2500)
            return
        if not self._start_align_sweep():
            self._uncheck_align_spec()
            return
        self.win.canvas.set_dev_visible(True)
        self.win.btn_align_refresh.show()
        self.win.appearance._position_panel_buttons()

    def _start_align_sweep(self):
        """Run one alignment sweep. False if it could not be started — the
        caller that was also opening the mode then does not open it.

        Every refusal below leaves the hardware exactly as it found it; the
        feed, which has to be handed over before the worker can drive the
        devices, is the only thing with any unwinding to do.
        """
        if self._align_running():
            return False
        if self.win.scanning.running():
            self.win.status.showMessage("A scan is running — the stage is busy.", 3000)
            return False
        if self.win.spec is None or self.win.stage is None:
            self.win.status.showMessage("Connect a stage and a spectrometer first.", 4000)
            return False
        # An unhomed axis sweeps just as well — the sweep is relative to
        # wherever the stage is, and running into an end stops the move
        # (StageLimitError). Only a locked axis (switches not confirmed) is
        # held back until the operator homes it or waives homing.
        if not self.win.stage_ctl.check_unlocked():
            return False

        # Same handover contract as _start_scan: the worker drives the stage
        # and the spectrometer for the whole sweep, so the feed has to let go
        # of them first, and refusing beats running against a device the feed
        # thread is still inside.
        self.win._feed_was_on = self.win.btn_feed.isChecked()
        if not self.win._feed.pause():
            if self.win._feed_was_on:
                self.win._feed.resume()
            self.win.status.showMessage(f"Alignment not started — {FEED_BUSY_MSG}", 6000)
            return False

        # Only now is it safe to read the stage from this thread — the feed is
        # parked, so nothing else is talking to it. Checked BEFORE moving:
        # clamping a target would quietly destroy the +/- symmetry the whole
        # measurement is about, so an out-of-range sweep is refused instead.
        step = self.win.dlg_align.step_fs
        pf   = self.win.scan_cfg.pass_factor
        try:
            start_um = stage_to_um(self.win.stage.get_position())
        except Exception as e:
            if self.win._feed_was_on:
                self.win._feed.resume()
            self.win.status.showMessage(f"Alignment failed — stage: {e}", 5000)
            return False
        span_um = abs(float(delay_to_position_um(2.0 * step, 0.0, pf)))
        lo, hi = self.win.stage_ctl._travel_range_um()
        if start_um - span_um < lo or start_um + span_um > hi:
            if self.win._feed_was_on:
                self.win._feed.resume()
            self.win.status.showMessage(
                f"±2×{step:.0f} fs (±{span_um:.1f} um) from here leaves the "
                f"travel range [{lo:.1f}, {hi:.1f}] um — move away from the "
                f"limit or reduce the Alignment Step.", 7000)
            return False

        self.win.appearance._set_hardware_buttons_enabled(False)
        self.win.btn_scan.setEnabled(False)
        self.win.stage_ctl._set_stage_controls_enabled(False)
        self.win.btn_align_spec.setEnabled(False)
        self.win.btn_align_refresh.setEnabled(False)
        self.win.stage_ctl._moving(True)

        self.win._align_worker = AlignmentWorker(self.win.stage, self.win.spec,
                                             self.win.scan_cfg, step)
        self.win._align_worker.progress.connect(self._on_align_progress)
        self.win._align_worker.done.connect(self._on_align_done)
        self.win._align_worker.error.connect(self._on_align_error)
        self.win._align_worker.start()
        self.win.status.showMessage(
            f"Alignment sweep: −{2 * step:.0f}, −{step:.0f}, "
            f"+{step:.0f}, +{2 * step:.0f} fs…", 0)
        return True

    def _on_align_progress(self, done, total):
        self.win.status.showMessage(f"Alignment sweep: point {done}/{total}…", 0)

    def _on_align_done(self, results):
        """Slot for AlignmentWorker.done — results in offset order."""
        self._reset_align_ui()
        try:
            # Corrected through exactly the path the spectrum panel uses, each
            # frame with ITS OWN member frames, so the differences sit on the
            # displayed baseline rather than a second derivation of it.
            s = [np.asarray(self.win.feed._corrected_frame(raw, frames), float)
                 for raw, frames in results]
            wl = np.asarray(self.win.spec.output_wavelengths, float)
            d1 = s[2] - s[1]        # S(+x)  - S(-x)
            d2 = s[3] - s[0]        # S(+2x) - S(-2x)
        except Exception as e:
            # The MODE stays open: the panel is still the right place to look,
            # the previous sweep (if any) is still on it, and ⟳ is how you try
            # again. Closing it here would throw away the comparison the
            # operator was in the middle of making.
            self.win.status.showMessage(f"Alignment differences failed: {e}", 5000)
            return
        self.win.canvas.show_diff(wl, d1, d2)
        step = self.win.dlg_align.step_fs
        # Peak absolute imbalance as a fraction of the largest measured signal:
        # the one number that says "symmetric" or "not" without reading the
        # curves off the panel.
        scale = max(float(max(np.max(np.abs(x)) for x in s)), 1e-12)
        r1 = float(np.max(np.abs(d1))) / scale
        r2 = float(np.max(np.abs(d2))) / scale
        self.win.status.showMessage(
            f"Alignment: peak asymmetry {100 * r1:.1f}% at ±{step:.0f} fs, "
            f"{100 * r2:.1f}% at ±{2 * step:.0f} fs.", 8000)

    def _on_align_error(self, msg):
        # Mode left open, for the reason in _on_align_done: a sweep that failed
        # is a reason to fix something and press ⟳, not to lose the panel.
        self._reset_align_ui()
        self.win.status.showMessage(f"Alignment sweep failed: {msg}", 6000)

    def _reset_align_ui(self):
        """Hand the devices back and re-enable what the sweep locked out."""
        self.win.stage_ctl._moving(False)
        self.win.appearance._set_hardware_buttons_enabled(True)
        self.win.btn_scan.setEnabled(True)
        self.win.stage_ctl._set_stage_controls_enabled(True)
        self.win.btn_align_spec.setEnabled(True)
        self.win.btn_align_refresh.setEnabled(True)
        self.win.stage_ctl._refresh_positions()
        if self.win._feed_was_on:
            self.win._feed.resume()

    def _on_align_trace_toggled(self, on):
        if on and self.win._align_trace is None:
            self.win.btn_align_trace.blockSignals(True)
            self.win.btn_align_trace.setChecked(False)
            self.win.btn_align_trace.blockSignals(False)
            self.win.status.showMessage(
                "No raw trace yet — run a scan in multi-spectrometer mode.", 4000)
            return
        self.win._align_trace_on = bool(on)
        if self.win._align_trace_on and self.win.btn_symmetry.isChecked():
            self._clear_symmetry()      # one trace-view mode at a time
        # Before the re-render: the raw view is each half scaled to its own
        # peak, and the colour bar has to stop calling that counts.
        self.win.canvas.set_trace_normalised(self.win._align_trace_on)
        # Re-render at once rather than waiting for the next column, so the
        # toggle also works on a finished trace.
        if self.win._align_trace_on:
            self.win.canvas.update_trace(self._align_trace_view(), 1.0)
        elif self.win._scan_trace is not None:
            self.win.canvas.update_trace(self.win._scan_trace, self.win._scan_peak)
        self.win.status.showMessage(
            "FROG trace: raw counts, per-spectrometer normalised." if on else
            "FROG trace: calibrated, stitched.", 4000)

    def _current_symmetry_view(self):
        """The symmetry view of the finished scan, or None if there is none."""
        if self.win.result is None:
            return None
        return symmetry_view(self.win._result_view, self.win._result_ac,
                             self.win.result.delays_fs, self.win._fold_fs)

    def _show_symmetry(self):
        """Push the symmetry view, its fold line and its score to the canvas."""
        view = self._current_symmetry_view()
        if view is None:
            return
        data, peak, score, fold = view
        if self.win._fold_fs is not None:
            where = "   (manual)"
        else:
            # How far the optimiser moved the fold off the autocorrelation
            # peak it started from — itself a reading of where zero delay is.
            ac_peak = float(self.win.result.delays_fs[
                int(np.nanargmax(self.win._result_ac))])
            where = ("" if abs(fold - ac_peak) < 1e-9
                     else f"   (AC peak {ac_peak:+.1f} fs)")
        with self.win.canvas.batch():
            self.win.canvas.update_trace(data, peak)
            self.win.canvas.set_fold_line(fold)
            self.win.canvas.set_symmetry_score(
                f"Asymmetry  {100 * score:.2f} %   @ {fold:+.1f} fs{where}")

    def _on_fold_dragged(self, x_fs):
        """The fold line was dragged. Re-render only when it has actually
        changed HALF COLUMN: a drag reports every pixel it crosses, and
        recomputing a trace-sized difference for a move that lands on the same
        fold would be work with nothing to show for it. Leaving the optimised
        fold for a hand-placed one always re-renders, to relabel the readout."""
        if self.win.result is None or not self.win._symmetry_on:
            return
        was_manual = self.win._fold_fs is not None
        before = symmetry_fold(self.win._result_ac, self.win.result.delays_fs,
                               self.win._fold_fs)
        self.win._fold_fs = float(x_fs)
        after = symmetry_fold(self.win._result_ac, self.win.result.delays_fs,
                              self.win._fold_fs)
        if after != before or not was_manual:
            self._show_symmetry()

    def _on_fold_reset(self):
        """Double-click on the line: back to the optimised fold."""
        if self.win._fold_fs is None or not self.win._symmetry_on:
            return
        self.win._fold_fs = None
        view = self._current_symmetry_view()
        self._show_symmetry()
        self.win.status.showMessage(
            f"Fold back on the most symmetric axis ({view[3]:+.1f} fs).", 3000)

    def _on_symmetry_toggled(self, on):
        # Switching the mode re-centres the fold, in BOTH directions. A
        # hand-dragged position belongs to the look the operator was taking at
        # the time; turning the mode off ends that look, so coming back to it
        # should start from the optimised fold rather than resurrect a fold
        # placed for a question already answered. Before the view is computed
        # on the next line, so the `on` path folds about the optimum.
        self.win._fold_fs = None
        view = self._current_symmetry_view() if on else None
        if on and view is None:
            self.win.btn_symmetry.blockSignals(True)
            self.win.btn_symmetry.setChecked(False)
            self.win.btn_symmetry.blockSignals(False)
            self.win.status.showMessage("No finished scan to fold yet.", 4000)
            return
        self.win._symmetry_on = bool(on)
        if self.win._symmetry_on and self.win.btn_align_trace.isChecked():
            # One trace-view mode at a time: both write the same image, so the
            # last one switched on would silently win on the next re-render.
            self.win.btn_align_trace.blockSignals(True)
            self.win.btn_align_trace.setChecked(False)
            self.win.btn_align_trace.blockSignals(False)
            self.win._align_trace_on = False
            self.win.canvas.set_trace_normalised(False)
            self.win.appearance._refresh_align_trace_button()
        if self.win._symmetry_on:
            self._show_symmetry()
        else:
            self.win.canvas.set_fold_line(None)
            self.win.canvas.set_symmetry_score("")
            self._restore_trace_view()
        self.win.status.showMessage(
            "Symmetry — folded at the most symmetric axis near the "
            "autocorrelation peak: measured trace left of the fold, "
            "|T(+d) − T(−d)| right of it. Drag the line to move it, "
            "double-click to return to the optimum."
            if on else "FROG trace: measured counts.", 6000)

    def _restore_trace_view(self):
        """Put the ordinary trace back after a display mode is switched off."""
        if self.win.result is not None:
            self.win.canvas.update_trace(self.win._result_view)
        elif self.win._scan_trace is not None:
            self.win.canvas.update_trace(self.win._scan_trace, self.win._scan_peak)

    def _clear_symmetry(self, running=None):
        """Drop out of symmetry mode. Called when a scan starts: the fold line
        belongs to the result it was computed from, and the panel is about to
        start showing a different trace entirely."""
        if self.win.btn_symmetry.isChecked():
            self.win.btn_symmetry.blockSignals(True)
            self.win.btn_symmetry.setChecked(False)
            self.win.btn_symmetry.blockSignals(False)
        self.win._symmetry_on = False
        # A hand-placed fold belongs to the trace it was placed on, so a new
        # scan starts back on the optimised fold.
        self.win._fold_fs = None
        self.win.canvas.set_fold_line(None)
        self.win.canvas.set_symmetry_score("")
        self.win.appearance._refresh_symmetry_button(running)

    def _init_align_trace(self, wl, n_delays):
        """Allocate the raw trace for a scan about to start, or drop it.

        The cut is fixed HERE, for the whole scan: every column is
        interpolated and sliced on arrival, so moving the overlap band
        mid-scan cannot retroactively re-cut what is already stored. The next
        scan picks up wherever the band has been left.
        """
        spec = self.win.spec
        if not isinstance(spec, StitchedSpectrometer):
            self._clear_align_trace()
            return
        w = np.asarray(wl, float)
        cut = 0.5 * (spec.band_lo_nm + spec.band_hi_nm)
        # Below the cut comes from spec1 (the bluer member), above from spec2.
        # Neither half extrapolates: the common grid starts at spec1's first
        # pixel, and the cut sits inside the overlap, so spec2 covers w[k:].
        self.win._align_cut   = int(np.searchsorted(w, cut))
        self.win._align_trace = np.zeros((w.size, n_delays), np.float32)
        self.win._align_view  = None
        self.win._align_peak  = [0.0, 0.0]
        self.win._align_wl    = [np.asarray(m.wavelengths, float) for m in spec.members]
        self.win.appearance._refresh_align_trace_button()

    def _store_align_column(self, i, members_raw):
        """Fold one column's member frames into the raw trace. O(n_wl).

        The recorded dark comes off each member BEFORE interpolation, at member
        level, because that is the only place it can be taken off without
        involving either calibration or the crossfade — which is the whole point
        of this view. Everything else stays raw: no calibration, no stitch
        factor, a hard cut rather than a blend, each half on its own scale.
        """
        if self.win._align_trace is None or members_raw is None:
            return
        w = self.win._scan_wl
        k = self.win._align_cut
        wl1, wl2 = self.win._align_wl
        m1, m2 = (np.asarray(f, float) for f in members_raw)
        darks = self.win._scan_member_dark
        if darks is not None and len(darks) == 2:
            d1, d2 = (np.asarray(d, float) for d in darks)
            if d1.shape == m1.shape and d2.shape == m2.shape:
                m1 = m1 - d1
                m2 = m2 - d2
        lo = np.interp(w[:k], wl1, m1)
        hi = np.interp(w[k:], wl2, m2)
        self.win._align_trace[:k, i] = lo
        self.win._align_trace[k:, i] = hi
        if lo.size:
            self.win._align_peak[0] = max(self.win._align_peak[0], float(lo.max()))
        if hi.size:
            self.win._align_peak[1] = max(self.win._align_peak[1], float(hi.max()))

    def _align_trace_view(self):
        """The raw trace with each member's half normalised to its own max.

        Normalisation happens HERE, not when a column is stored: the two
        maxima only grow as a scan runs, so scaling on arrival would leave
        every earlier column on a stale factor. Written into a reused buffer —
        this runs once per column during a live scan.
        """
        raw = self.win._align_trace
        k   = self.win._align_cut
        if self.win._align_view is None or self.win._align_view.shape != raw.shape:
            self.win._align_view = np.empty_like(raw)
        buf = self.win._align_view
        np.divide(raw[:k], max(self.win._align_peak[0], 1.0), out=buf[:k])
        np.divide(raw[k:], max(self.win._align_peak[1], 1.0), out=buf[k:])
        return buf

    def _clear_align_trace(self):
        """Forget the raw trace and drop out of the raw view.

        Called on a device swap: a raw trace belongs to the pair that measured
        it — its cut, its grid and its two normalisations all describe those
        two spectrometers.
        """
        self.win._align_trace = None
        self.win._align_view  = None
        self.win._align_peak  = [0.0, 0.0]
        if self.win.btn_align_trace.isChecked():
            self.win.btn_align_trace.blockSignals(True)
            self.win.btn_align_trace.setChecked(False)
            self.win.btn_align_trace.blockSignals(False)
        self.win._align_trace_on = False
        self.win.canvas.set_trace_normalised(False)
        self.win.appearance._refresh_align_trace_button()
