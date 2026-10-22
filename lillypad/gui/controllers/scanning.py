"""Running a FROG scan and drawing it as it arrives.

`start` reads every setting live, checks the range fits the travel, asks about a
missing dark, and hands a FrogScanConfig to a worker thread. Everything after
that is a slot on one of the worker's signals.

The build-up is O(1) in the column index, which is the difference between a scan
that stays responsive and one that slows down as it runs. The autocorrelation
and the trace peak are accumulated per column rather than re-reduced over the
whole trace, which used to cost O(n_wl · N²) over a scan, and the render is
coalesced into the same 60 ms display tick the live feed uses.

Two policies are deliberately different, and both are settings:

  * a **stage fault** means the stage is not where the program thinks it is, so
    every later column carries a delay that never happened — the default is to
    stop at the first one;
  * **saturation** is an alert by default, because a clipped spectrum is still
    labelled with the right delay, and the clipped columns are marked in the
    .npz.

The dark is the one recorded in the Spectrum panel, snapshotted when the scan
starts. The delay-extreme reference, if it was asked for, is archived beside it
and never subtracted.
"""

import numpy as np

from PySide6.QtWidgets import QMessageBox
from lillypad.gui.text import FEED_BUSY_MSG
from lillypad.scan import autocorrelation, fwhm
from lillypad.scan.worker import FrogScanWorker


class ScanController:
    """Starting a scan, following it, and finishing it."""

    def __init__(self, win):
        self.win = win

    # ── Shared-device access ─────────────────────────────────────────────────
    def running(self):
        return self.win._worker is not None and self.win._worker.isRunning()

    # ── FROG scan ─────────────────────────────────────────────────────────────
    def _start_scan(self):
        if self.win._worker is not None and self.win._worker.isRunning():
            self.win._worker.abort()
            self.win.btn_scan.setEnabled(False); self.win.btn_scan.setText("Aborting…")
            return
        if not self.win._have_spec():
            self.win.status.showMessage("Connect a spectrometer first.", 4000); return
        if not self.win._have_stage():
            self.win.status.showMessage("Connect a stage first — a FROG scan has "
                                    "to sweep the delay.", 4000); return
        if self.win.alignment._align_running():
            self.win.status.showMessage(
                "An alignment sweep is running — the stage is busy.", 3000); return
        # An unhomed axis may still scan: every delay is measured from the zero
        # the operator marked, which is just as valid in the power-up frame.
        # Only homing moves that frame, and _home_stage drops the zero when it
        # does. What is lost is the pre-flight check against the real ends —
        # the worker stops the scan instead if it runs into one.
        if not self.win.stage_ctl.check_unlocked():
            return
        unhomed = bool(getattr(self.win.stage, "needs_homing", False))

        c = self.win.scan_cfg
        c.delay_start_fs = self.win.spin_start.value()
        c.delay_stop_fs  = self.win.spin_stop.value()
        c.delay_step_fs  = self.win.spin_step_fs.value()
        c.n_average      = self.win.dlg_settings.n_average
        c.idle_shots     = self.win.dlg_settings.idle_shots
        c.wait_after_move_s = self.win.dlg_settings.settle_ms / 1000.0
        c.capture_delay_reference = self.win.dlg_settings.take_delay_reference
        c.saturation_fraction  = self.win.dlg_settings.sat_percent / 100.0
        c.abort_on_saturation  = self.win.dlg_settings.abort_on_saturation
        c.abort_on_stage_fault = self.win.dlg_settings.abort_on_stage_fault
        # c.saturation_counts is owned by the Hardware dialog's override field.

        try:
            delays = c.delays_fs()
        except ValueError as e:
            self.win.status.showMessage(f"Bad scan range: {e}", 4000); return
        wl = np.asarray(self.win.spec.wavelengths, float)

        # No dark means the exported trace carries the full pedestal, which on
        # a typical bench is several times the signal — so say so ONCE, before
        # anything moves, and let the operator decide. Asked here rather than
        # refused outright because a deliberately uncorrected scan is a
        # legitimate thing to want; the .npz records which it was either way.
        if self.win.feed.display_dark() is None and not self._confirm_no_dark():
            return

        # Hand the shared stage/spectrometer over to the scan worker: pause the
        # live feed and wait for any in-flight acquire to finish first. If it
        # will not let go, refuse to start rather than run the scan against a
        # spectrometer the feed thread is still inside.
        self.win._feed_was_on = self.win.btn_feed.isChecked()
        if not self.win._feed.pause():
            if self.win._feed_was_on:
                self.win._feed.resume()
            self.win.status.showMessage(f"Scan not started — {FEED_BUSY_MSG}", 6000)
            return

        # The worker captures these devices for the whole scan, so lock out
        # everything that could swap or drive them underneath it.
        self.win.appearance._set_hardware_buttons_enabled(False)
        self.win.stage_ctl._set_stage_controls_enabled(False)
        # The fold line belongs to the result it came from, and the panel is
        # about to start showing a different trace. running=True is asserted:
        # the worker for THIS scan does not exist yet, so _scan_running() would
        # still be answering for the previous one.
        self.win.alignment._clear_symmetry(running=True)

        self.win._scan_trace  = np.zeros((wl.size, delays.size))
        self.win._scan_delays = delays
        self.win._scan_wl     = wl
        # The pedestal every column of THIS scan is drawn against, sampled once
        # here rather than per column or per render.
        #
        # Once, because a trace assembled from columns corrected two different
        # ways is not a picture of anything — and because the alternative,
        # subtracting at render time, is a pass over the whole trace on every
        # display tick (measured: 12 ms on a 4000 x 1000 trace, against the
        # 4 ms the row pooling already costs), where doing it as each column
        # lands is one 4000-element subtract.
        #
        # Nothing it depends on can move mid-scan in any case: the worker owns
        # the device, so no dark can be recorded and Auto-stitch cannot re-fit,
        # and the Spectrometer window (the band) is locked out for the
        # duration. The one thing an operator CAN still do is untick Subtract
        # Dark, and that now takes effect on the next scan rather than striping
        # this one.
        self.win._scan_dark   = self.win.feed.display_dark()
        # The same dark at MEMBER level, for the RAW view. RAW shows the two
        # detectors uncalibrated and unstitched, so the pedestal has to come off
        # each member's own raw frame — the combined dark above has already been
        # through both calibrations and the crossfade.
        self.win._scan_member_dark = (
            self.win.background_members
            if (self.win.chk_dark.isChecked() and self.win._pair_live()) else None)
        # Running reductions _on_column keeps current, so the per-column render
        # never has to walk the trace. Reallocated with the trace: an aborted
        # scan leaves these sized to the OLD delay axis.
        self.win._scan_ac     = np.zeros(delays.size)
        self.win._scan_peak   = 0.0
        # A pending render from a previous (aborted) scan must not fire
        # against the fresh, differently-sized arrays.
        self.win._scan_dirty  = False
        self.win._scan_last_i = -1
        self.win._scan_members_raw = None   # …and the last scan's member frames
        self.win._live_frame  = None     # park any leftover live-feed frame too
        self.win.alignment._init_align_trace(wl, delays.size)
        self.win.canvas.init_trace(delays, wl)
        self.win.feed.reset_saturation(); self.win.progress.setValue(0)
        # A difference measured at the old position says nothing about the
        # scan that is starting, and the scan owns the stage the sweep needs.
        if self.win.btn_align_spec.isChecked() or self.win.canvas.dev_visible():
            self.win.alignment._close_align_mode()

        self.win.btn_scan.setObjectName("danger"); self.win.btn_scan.setText("Abort Scan")
        self.win.btn_scan.style().unpolish(self.win.btn_scan); self.win.btn_scan.style().polish(self.win.btn_scan)
        self.win.export.set_save_enabled(False)

        self.win._scan_stage_faulted = False      # see _on_error
        self.win._worker = FrogScanWorker(self.win.stage, self.win.spec, c,
                                      dark=self.win._scan_dark)
        self.win._worker.progress.connect(self._on_progress)
        self.win._worker.column_ready.connect(self._on_column)
        self.win._worker.delay_reference_ready.connect(self._on_delay_reference)
        self.win._worker.saturation_warning.connect(self._on_saturation)
        self.win._worker.stage_fault.connect(self._on_stage_fault)
        self.win._worker.finished_scan.connect(self._on_finished)
        self.win._worker.error.connect(self._on_error)
        # QThread.finished, not finished_scan: the worker parks the stage on
        # zero delay in a finally, which runs AFTER finished_scan is emitted.
        # Refreshing the readout from _on_finished would print the last delay's
        # position and leave it there.
        self.win._worker.finished.connect(self.win.stage_ctl._refresh_positions)
        self.win._worker.start()
        # Auto-stitch drives the pair the worker now owns, so it greys out for
        # the duration. The per-spectrometer view does not — the scan carries
        # its own member frames, and _render_scan_frame draws them.
        self.win.appearance._refresh_overlay_button()
        # Explicit, not inferred: this happens to run after _worker.start(), so
        # _scan_running() would answer True — but that is an ordering accident,
        # and the same call one line earlier would have greyed nothing.
        self.win.appearance._refresh_autostitch_button(running=True)
        self.win.status.showMessage(
            f"FROG scan: {delays.size} points…" + (
                " Stage not homed — scanning in its power-up frame; the scan "
                "stops if it reaches either end." if unhomed else ""), 0)

    def _on_progress(self, done, total):
        self.win.progress.setValue(int(100 * done / total))

    def _on_column(self, i, delay_fs, pos_um, col, members_raw=None):
        """Slot for FrogScanWorker.column_ready — O(1), like _on_spectrum.

        Every column is RECORDED here (the data path must never drop), but
        rendering is deferred to _display_tick so a slow machine skips
        intermediate redraws instead of queuing them up.

        `col` is corrected on the way in. _scan_trace is a DISPLAY buffer — the
        FrogResult is assembled by the worker from its own array, and nothing
        here reaches a file — so this is the one place the correction has to
        happen for the trace, the running peak, the autocorrelation sum and the
        spectrum panel to agree about what a column contains.
        """
        col = self._corrected_column(col)
        self.win._scan_trace[:, i] = col
        # The raw member frames are interpolated and cut ONCE, here, rather
        # than kept per column and re-derived at render time: that keeps the
        # memory to one trace-sized array and the work to O(n_wl) per column.
        self.win.alignment._store_align_column(i, members_raw)
        # …but the NEWEST pair is kept as-is, so the per-spectrometer spectrum
        # view has something to draw while a scan owns the device. One tuple
        # reference, overwritten per column — the scan's cut and normalisation
        # above are for the trace and are the wrong thing for this panel.
        self.win._scan_members_raw = members_raw
        # Fold the new column into the running reductions the render needs, so
        # _render_scan_frame stays O(1) in the column index. Recomputing either
        # of these from the whole trace per column made the scan cost O(N^2).
        self.win._scan_ac[i]  = col.sum()
        self.win._scan_peak   = max(self.win._scan_peak, float(col.max()))
        self.win._scan_last_i = i
        self.win._scan_col = col
        self.win._scan_pos_um = pos_um
        self.win._scan_dirty = True

    def _render_scan_frame(self):
        """Draw the in-progress scan from the newest recorded column.

        Everything here is O(1) in the column index on purpose. The obvious
        version — autocorrelation() over self.win._scan_trace[:, :i+1], and a
        raw.max() over the whole preallocated trace inside _render_trace — is
        O(n_wl * i) per column, i.e. O(n_wl * N^2) over the scan, and was what
        made a long scan get slower the further it ran. _on_column keeps both
        reductions up to date incrementally instead.
        """
        self.win._scan_dirty = False
        i = self.win._scan_last_i
        self.win.lbl_pos.setText(f"{self.win._scan_pos_um:.2f} um")
        # Same value autocorrelation() would return for this slice: a per-column
        # sum over the wavelength axis, baseline-shifted to its own minimum
        # (scan.py:87-89). Only the baseline couples the columns, and a min over
        # the accumulated sums is 1-D and cheap.
        acc = self.win._scan_ac[:i + 1]
        ac  = acc - acc.min()
        # One blit for all three panels instead of three.
        with self.win.canvas.batch():
            # Keep the spectrum panel alive through the scan. The live feed is
            # parked (the worker owns the device), so the column the scan just
            # measured is the only spectrum there is.
            #
            # Trace only, never the headroom lamp: `col` is an average of
            # n_average frames, so its peak sits below any single frame's and
            # would under-report clipping. Saturation during a scan is reported
            # per-frame by the worker via saturation_warning -> _on_saturation.
            #
            # The per-spectrometer view is honoured here exactly as the live
            # feed honours it: update_spectrum owns the panel mode, so calling
            # it unconditionally was what dropped a stitched pair back to the
            # combined curve the moment a measurement started. A column whose
            # member frames did not arrive falls through to the combined curve
            # rather than freezing on the last one.
            if self.win._scan_wl is not None:
                if not (self.win._overlay_on
                        and self.win.feed._render_overlay(self.win._scan_members_raw)):
                    # Already corrected, in _on_column, along with the column
                    # that went into the trace beside this panel.
                    self.win.canvas.update_spectrum(self.win._scan_wl, self.win._scan_col)
            if self.win._align_trace_on and self.win._align_trace is not None:
                self.win.canvas.update_trace(self.win.alignment._align_trace_view(), 1.0)
            else:
                self.win.canvas.update_trace(self.win._scan_trace, self.win._scan_peak)
            self.win.canvas.update_ac(self.win._scan_delays[:i + 1], ac)
            # Inside the batch: the readout lives in the AC panel now, so its
            # repaint is the same blit the three updates above share.
            self.win.canvas.set_fwhm(fwhm(self.win._scan_delays[:i + 1], ac))

    def _corrected_column(self, col):
        """One scan column with this scan's pedestal removed.

        A new array, never a write into `col`: the worker built that one and
        the raw member frames beside it are what the RAW alignment view is
        assembled from.

        Clipped at zero like every other correction in the program
        (prepare_pair, _corrected_spectrum): below the pedestal there is no
        signal to show, and the trace panel has no way to draw a negative
        count. The autocorrelation is unaffected by the floor — it subtracts
        its own baseline (scan.autocorrelation).
        """
        bg = self.win._scan_dark
        if bg is None or np.shape(bg) != np.shape(col):
            return col
        return np.clip(col - bg, 0, None)

    def _confirm_no_dark(self):
        """Ask before running a scan whose export will not be dark-corrected.

        Two distinct reasons land here and the message names both, because the
        fix differs: no dark has been recorded at all, or one has but Subtract
        Dark is off. Answering No leaves everything untouched.
        """
        if self.win.background is None and self.win.background_members is None:
            why = ("No dark has been recorded.")
            fix = ("Record Dark in the Spectrum panel first, or continue and "
                   "the .dwc and .csv will carry the raw pedestal.")
        else:
            why = ("A dark is recorded, but Subtract Dark is switched off.")
            fix = ("Tick Subtract Dark to correct this scan, or continue and "
                   "the .dwc and .csv will carry the raw pedestal.")
        return QMessageBox.question(
            self.win, "Scan without a dark?",
            f"{why}\n\n{fix}\n\nThe .npz records that no dark was "
            f"subtracted either way.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No) == QMessageBox.Yes

    def _on_delay_reference(self, which, spectrum):
        self.win.status.showMessage(
            f"Delay-extreme reference ({which}) captured — archived, not "
            f"subtracted.", 2500)

    def _on_stage_fault(self, i, delay_fs, reason):
        """Slot for FrogScanWorker.stage_fault.

        With the abort enabled the worker's error() follows immediately and
        supersedes this; the message matters in the override case, where the
        scan carries on and this is the only live sign that it has stopped
        tracking. Timeout 0 — it must not quietly disappear.
        """
        self.win._scan_stage_faulted = True
        where = "the reference frame" if i < 0 else f"{delay_fs:+.0f} fs"
        self.win.status.showMessage(f"⚠ STAGE FAULT @ {where} — {reason}", 0)

    def _on_saturation(self, i, delay_fs, npx, peak):
        """Slot for FrogScanWorker.saturation_warning — one signal per frame.

        Latches: it reports that THIS scan's data is clipped, which stays true
        for the rest of the scan even if later columns come back clean. The
        worst frame is what gets named, since simply overwriting on every
        signal would leave the last (often marginal) column on display and hide
        how much of the trace is actually ruined.
        """
        self.win._sat_latched = True
        self.win._sat_frames += 1
        if npx > self.win._sat_worst[0]:
            self.win._sat_worst = (npx, delay_fs, i)
        npx_w, delay_w, i_w = self.win._sat_worst
        where = "background" if i_w < 0 else f"{delay_w:+.0f} fs"
        self.win.appearance._set_lamp("sat",
                       f"⚠ SATURATED — {self.win._sat_frames} frame"
                       f"{'' if self.win._sat_frames == 1 else 's'}, "
                       f"worst @ {where} ({npx_w} px)",
                       "sat")
        # The worker's signal aggregates over stitched members; latch both
        # lamps so neither device shows a green light over clipped data.
        if self.win._pair_live():
            self.win.lamp2.set_state("sat")

    def _finished_trace_view(self, result):
        """The finished scan as it should be DRAWN — pedestal removed.

        Prefers the scan's OWN bracketed background over the recorded dark, and
        never uses both: the background frames were taken through this very
        pipeline with the beam blocked, so they already carry the dark along
        with whatever stray light was on the bench, and subtracting the dark as
        well would take the pedestal off twice. This is also exactly what
        .dwc and .csv are written from (FrogResult.corrected_trace), so the
        panel and the file finally agree.

        Falls back to the dark when the scan ran with Background unticked, and
        to the raw trace when there is neither — a trace with no pedestal
        removed, which is what every finished scan used to show.

        Display only. result.trace is untouched, and the .npz still archives
        the raw counts with both background frames kept separately.
        """
        if result.dark is None:
            return result.trace
        return np.clip(result.corrected_trace(), 0, None)

    def _on_finished(self, result):
        self.win.result = result
        # Cached, not recomputed per view: the symmetry fold and the RAW toggle
        # both re-render from it, and it is a trace-sized subtraction.
        self.win._result_view = self._finished_trace_view(result)
        # From the SAME array the panel shows, so the curve and the width
        # readout describe the picture above them. They used to come off
        # result.trace, i.e. off the pedestal-bearing counts.
        self.win._result_ac = autocorrelation(self.win._result_view)
        # The final render below supersedes any pending intermediate tick.
        self.win._scan_dirty = False
        if self.win._align_trace_on and self.win._align_trace is not None:
            # The raw view is a display mode, not a stage of the scan: finishing
            # must not silently drop the operator back to the calibrated trace.
            self.win.canvas.update_trace(self.win.alignment._align_trace_view(), 1.0)
        else:
            self.win.canvas.update_trace(self.win._result_view)
        ac = self.win._result_ac
        self.win.canvas.update_ac(result.delays_fs, ac)
        self.win.canvas.set_fwhm(fwhm(result.delays_fs, ac))
        self.win.progress.setValue(100)
        n_bad = int(result.faulted_columns().size)
        if n_bad:
            # Only reachable with the abort override off — the scan ran to the
            # end over a stage that had stopped tracking. Say so permanently
            # rather than for five seconds; the delay axis is compromised.
            self.win.status.showMessage(
                f"Scan complete — {result.trace.shape[1]} columns, but the "
                f"stage reported a fault on {n_bad} of them. Their delays are "
                f"not trustworthy; the .npz marks which.", 0)
        else:
            self.win.status.showMessage(
                f"Scan complete — {result.trace.shape[1]} columns.", 5000)
        self.win.export.set_save_enabled(True)
        self._reset_scan_ui()

    def _on_error(self, msg):
        # A scan stopped by the stage (a fault, or an end of travel on an
        # unhomed axis) keeps its message up: the operator has to act on it
        # before rescanning, and six seconds is easy to miss on a long scan.
        faulted = self.win._scan_stage_faulted
        self.win.status.showMessage(f"{'⚠ ' if faulted else ''}Scan: {msg}",
                                    0 if faulted else 6000)
        self._reset_scan_ui()

    def _reset_scan_ui(self):
        self.win.btn_scan.setEnabled(True)
        self.win.btn_scan.setObjectName("accent"); self.win.btn_scan.setText("Measure FROG")
        self.win.btn_scan.style().unpolish(self.win.btn_scan); self.win.btn_scan.style().polish(self.win.btn_scan)
        self.win.appearance._set_hardware_buttons_enabled(True)
        self.win.stage_ctl._set_stage_controls_enabled(True)
        self.win.appearance._refresh_overlay_button()
        # running=False for the same reason as the symmetry button below: the
        # worker's thread may not have left run() yet, so neither may infer it.
        self.win.appearance._refresh_autostitch_button(running=False)
        self.win.appearance._refresh_align_trace_button()
        self.win.appearance._refresh_symmetry_button(running=False)   # a finished scan folds
        if self.win._feed_was_on and self.win.btn_feed.isChecked():
            self.win._feed.resume()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def join_timeout_ms(self):
        """How long to allow the scan worker to finish the step it is on.

        abort() only takes effect between delay points, and a point is a stage
        move plus (idle_shots + n_average) exposures — far more than the live
        feed's single-cycle estimate covers, so it gets its own figure derived
        from the settings the running scan was started with.
        """
        c = self.win.scan_cfg
        shots = max(1, int(c.idle_shots) + int(c.n_average))
        exposure = float(getattr(self.win.spec, "integration_ms", 10.0))
        return max(5000.0, 2.0 * shots * exposure
                   + 2000.0 * float(c.wait_after_move_s) + 5000.0)
