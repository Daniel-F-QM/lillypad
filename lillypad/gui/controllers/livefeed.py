"""The live spectrum: acquisition, display, the dark, and saturation.

The hot path. A worker thread acquires as fast as the spectrometer produces
frames and only ever *stores* its payload; a fixed 60 ms timer renders whatever
is newest. On a machine where a render costs more than the acquisition interval
that drops stale frames instead of piling them up in the Qt event queue, so the
display cannot fall progressively behind real time. `_display_tick` is that
timer, and it is the one method here where an extra array copy is measurable.

The **dark** lives here too, and it is the same dark the scan uses — there is
one recorded dark in this program and not two. It is kept per member for a
stitched pair, because the pedestal has to come off BEFORE each member's own
calibration; the merged version is derived at whatever stitch factor is current
rather than stored. A dark is offset plus dark-current x t, so it is only valid
at its own exposure, and `invalidate_dark` drops it when the exposure changes
rather than rescaling something that is not a pure ratio.

**Saturation** is judged here rather than in the scan because it has to be live:
the threshold is applied to every displayed frame, so the lamp reflects the
setting you are in the middle of tuning. Dropped frames cannot hide a clipped
one — the highest-peak frame since the last tick is the one that gets judged.
"""

import numpy as np

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QApplication
from lillypad.gui.perf import perf_tick
from lillypad.gui.text import FEED_BUSY_MSG
from lillypad.gui.widgets import SAT_WARN_FRACTION
from lillypad.gui.workers import _exposure_bounds, _exposure_ms


class LiveFeedController:
    """The live feed, the displayed spectrum, the dark, and the lamp."""

    def __init__(self, win):
        self.win = win

    def _on_sat_fraction(self, percent):
        self.win.scan_cfg.saturation_fraction = percent / 100.0
        self.reset_saturation()    # re-judge; the old latch used a stale rule

    @staticmethod
    def _seed_integration_spin(spin, dev):
        """Point one exposure box at `dev`: range, resolution, step and value.

        The range is the device's, so sub-millisecond exposures appear only on
        hardware that actually has them and the box cannot be used to ask for
        something the SDK will refuse. Resolution follows: microsecond steps
        would be noise on a device with a 1 ms floor, and adaptive stepping
        keeps the arrows useful across a range that now spans four decades.

        Blocked throughout: seeding must not restart the debounce, or 250 ms
        later we take a feed handover to write back a value the device already
        holds.
        """
        lo, hi = _exposure_bounds(dev)
        sub_ms = lo < 1.0
        with QSignalBlocker(spin):
            spin.setDecimals(3 if sub_ms else 0)   # before setRange: it rounds
            spin.setRange(lo, hi)
            spin.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType if sub_ms
                             else QAbstractSpinBox.DefaultStepType)
            spin.setSingleStep(1.0)
            spin.setValue(min(hi, max(lo, _exposure_ms(dev))))

    def sync_integration_ui(self):
        """Reseed the integration spinbox(es) from the live device(s).

        Needed whenever the device behind a box changes, and whenever
        something OTHER than the box changes the exposure it should be
        showing. A live pair takes one box per member, from that member's own
        limits — the two halves of a mixed-vendor pair do not share a range.
        """
        if not self.win._have_spec():
            return          # the boxes are hidden with the rest of the panel
        slots = self.win.devices._slot_members() if self.win._pair_live() else []
        if len(slots) == 2:
            for mem, spin in zip(slots, (self.win.spin_integration,
                                         self.win.spin_integration2)):
                self._seed_integration_spin(spin, mem)
        else:
            self._seed_integration_spin(self.win.spin_integration, self.win.spec)

    def _on_overlay_toggled(self, on):
        if on and not (self.win._pair_live()
                       and len(self.win.devices._slot_members()) == 2):
            with QSignalBlocker(self.win.btn_overlay):
                self.win.btn_overlay.setChecked(False)
            return
        self.win._overlay_on = bool(on)
        if not on:
            # Don't wait for the next frame to put the combined curve back.
            self.win.canvas.clear_members()
        self.win.appearance._refresh_overlay_button()
        self.win.appearance._refresh_autostitch_button()
        self.win.status.showMessage(
            "Spectrum panel: one curve per spectrometer." if on else
            "Spectrum panel: combined stitched spectrum.", 4000)

    def reset_saturation(self):
        """Clear a latched warning and re-read the effective full scale(s)."""
        self.win._sat_latched = False
        self.win._sat_frames  = 0            # saturated frames so far this scan
        self.win._sat_worst   = (0, 0.0, 0)  # (n_pixels, delay_fs, column index)
        slots = self.win.devices._slot_members() if self.win._pair_live() else []
        if slots:
            # Live pair: one lamp per member, each judged against its own full
            # scale.
            for i, mem in enumerate(slots):
                full = self.win.devices._member_full_scale(mem)
                lamp = self.win.lamp if i == 0 else self.win.lamp2
                if full:
                    src = ("override" if self.win.scan_cfg.saturation_counts
                           else "device")
                    lamp.setToolTip(
                        f"{mem.name} headroom — full scale {full:.0f} counts "
                        f"({src}), saturated at "
                        f"{100 * self.win.scan_cfg.saturation_fraction:.0f}%")
                    self.win.appearance._set_lamp("ok", f"S{i + 1}", "satok", i)
                else:
                    lamp.setToolTip(
                        f"{mem.name} does not report a full-scale value, so "
                        f"saturation cannot be checked. Set one in "
                        f"Spectrometer → Full scale.")
                    self.win.appearance._set_lamp("unknown", f"S{i + 1} — % FS", "satok", i)
            return
        if not self.win._have_spec():
            self.win.lamp.setToolTip("No spectrometer connected.")
            self.win.appearance._set_lamp("unknown", "", "satok")
            return
        full = self.win.devices._full_scale()
        if not full:
            self.win.lamp.setToolTip(
                "This spectrometer does not report a full-scale value, so "
                "saturation cannot be checked. Set one in Spectrometer → "
                "Full scale.")
            self.win.appearance._set_lamp("unknown", "— % FS", "satok")
        else:
            source = ("override" if self.win.scan_cfg.saturation_counts else "device")
            self.win.lamp.setToolTip(
                f"Detector headroom — full scale {full:.0f} counts ({source}), "
                f"saturated at {100 * self.win.scan_cfg.saturation_fraction:.0f}%")
            self.win.appearance._set_lamp("ok", "", "satok")

    def _judge_lamp(self, which, raw, full, prefix=""):
        """Drive one lamp from one RAW frame and its full scale.

        Same threshold the scan worker uses, so the live lamp and the scan
        warning can never disagree about what counts as saturated.
        """
        if not full:
            self.win.appearance._set_lamp("unknown", f"{prefix}— % FS", "satok", which)
            return
        peak = float(np.max(raw)) if raw.size else 0.0
        frac = peak / full
        sat_frac = self.win.scan_cfg.saturation_fraction
        if frac >= sat_frac:
            n = int(np.count_nonzero(raw >= sat_frac * full))
            self.win.appearance._set_lamp("sat", f"{prefix}⚠ SATURATED  ({n} px)", "sat", which)
        elif frac >= SAT_WARN_FRACTION:
            self.win.appearance._set_lamp("warn", f"{prefix}⚠ {100 * frac:.0f}% FS",
                           "satwarn", which)
        else:
            self.win.appearance._set_lamp("ok", f"{prefix}{100 * frac:.0f}% FS", "satok", which)

    def _update_saturation(self, raw):
        """Live headroom readout, driven by the feed's RAW (un-subtracted) frame.

        Skipped while a scan's warning is latched: that one records that the
        measurement is already compromised, and must not be scrolled away by
        whatever the feed sees once the scan hands the hardware back.
        """
        if self.win._sat_latched:
            return
        slots = self.win.devices._slot_members() if self.win._pair_live() else []
        frames = getattr(self.win.spec, "last_member_raw", None)
        if slots and frames is not None:
            # Two separate alarms: each member's own raw frame against its own
            # full scale. `raw` (the combined stitched frame) has no single
            # ADC scale and is deliberately not judged.
            members = self.win.spec.members
            raw_by_id = {id(m): f for m, f in zip(members, frames)}
            for i, mem in enumerate(slots):
                frame = raw_by_id.get(id(mem))
                if frame is not None:
                    self._judge_lamp(i, frame, self.win.devices._member_full_scale(mem),
                                     prefix=f"S{i + 1} ")
            return
        full = self.win.devices._full_scale()
        if not full:
            return
        self._judge_lamp(0, raw, full)

    # ── Live feed ─────────────────────────────────────────────────────────────
    def _on_feed_error(self, msg):
        """Slot for LiveFeedWorker.acquire_failed — runs on the GUI thread.

        The live feed cannot stop on a failed frame (a device is allowed the
        occasional hiccup, and the loop must keep trying), but it must not
        swallow one either: a plot frozen on its last good frame reads as "the
        setting I just changed did nothing" rather than as an error. The worker
        throttles the repeats, so this only ever shows a live problem.
        """
        self.win.status.showMessage(f"Live feed: {msg}", 6000)

    def _on_spectrum(self, wl, raw):
        """Slot for LiveFeedWorker.spectrum_ready — runs on the GUI thread.

        Deliberately O(1): it only records the newest frame for _display_tick
        to render. Doing the rendering here let the event queue grow without
        bound whenever a render outlasted the acquisition interval (weak
        machines), and the display lagged further behind every second.
        """
        self.win._live_frame = (wl, raw)
        # Keep the worst (highest-peak) frame between ticks so a transiently
        # clipped frame that never gets displayed still trips the lamp.
        p = float(raw.max()) if raw.size else 0.0
        if p > self.win._sat_peak:
            self.win._sat_peak = p
            self.win._sat_frame = raw

        # NB: no stage read here. Polling get_position() at feed rate raced the
        # feed thread's own acquire() (which, for the simulator, itself reads
        # the stage). The readout is instead refreshed after every move we make
        # and from the scan worker's per-column read-back.

    def _display_tick(self):
        """Render the newest pending data — scan column or live frame."""
        if self.win._scan_dirty:
            with perf_tick("scan_render"):
                self.win.scanning._render_scan_frame()
            return
        if self.win._live_frame is None:
            return
        if not self.win._have_spec():
            # The feed emits from its own thread, so a spectrum_ready queued
            # before the disconnect is still delivered after it. Drop it: the
            # frame describes a device that is gone, and every correction
            # below it would go through self.win.spec.
            self.win._live_frame = None
            return
        with perf_tick("live_render"):
            wl, raw = self.win._live_frame
            self.win._live_frame = None
            self.win.last_spectrum = raw
            self._update_saturation(self.win._sat_frame if self.win._sat_frame is not None
                                    else raw)
            self.win._sat_frame = None
            self.win._sat_peak = -1.0
            if self.win._overlay_on and self._render_overlay():
                self._flush_pending_fit()
                return
            corrected = self._dark_corrected(raw)
            if np.shape(corrected)[-1:] != np.shape(wl):
                # Emitted before the calibrated axis last moved (a change of
                # method, or a calibration in legacy mode, which trims pixels):
                # its axis and today's correction disagree. The next frame is
                # right, so this one is simply dropped.
                return
            self.win.canvas.update_spectrum(wl, corrected)
            self._flush_pending_fit()

    def _dark_corrected(self, raw):
        """The merged live frame with the dark removed, ready to plot."""
        return self._corrected_frame(
            raw, getattr(self.win.spec, "last_member_raw", None))

    def _corrected_frame(self, raw, frames):
        """`raw` with the dark removed and the calibration applied.

        `frames` are the RAW member frames `raw` was merged from (None for a
        single device). Passed in rather than read from the device here, so a
        caller holding a frame from some time ago — the alignment sweep — gets
        the correction for ITS OWN member frames instead of whatever the feed
        has acquired since.

        A stitched pair is rebuilt from the member frames rather than having a
        stored merged dark subtracted: the pedestal comes off each member
        before its own calibration, and the crossfade then happens at the
        CURRENT stitch factor and band. That is what makes this curve equal to
        the crossfade of the two per-spectrometer curves by construction — a
        pre-merged dark carried the factor it was recorded at and drifted the
        moment Auto-stitch ran.
        """
        if not self.win.chk_dark.isChecked():
            return raw if self.win._pair_live() \
                else self.win.spec.calibrate(raw)
        if self.win._pair_live():
            darks = self.win.devices._usable_member_darks()
            if darks is None or frames is None:
                self._warn_dark_mismatch()
                return raw          # already calibrated and merged by acquire()
            return self.win.spec.combine(
                *self.win.spec.prepare_pair(*frames, darks))
        spectrum = raw
        if self.win.background is not None:
            spectrum = raw - self.win.background
            if self.win.processing != "legacy":      # legacy mode keeps negatives
                spectrum = np.clip(spectrum, 0, None)
        # Calibration LAST, and never on what the lamp judged: saturation is a
        # raw-ADC property, the calibration is display/data physics.
        return self.win.spec.calibrate(spectrum)

    def display_dark(self):
        """The pedestal to take off a MERGED, CALIBRATED frame before drawing
        it — or None when there is nothing to take off.

        The trace panel's counterpart to _corrected_spectrum, which cannot use
        that function: a scan column arrives already merged and calibrated and
        there are no per-column member frames to rebuild it from, so the dark
        has to be pushed through the same merge instead of being taken off each
        member first. StitchedSpectrometer.combined_dark is that push, at the
        CURRENT stitch factor and band, which is what lands this on the same
        baseline the live spectrum was drawn against. A single device's dark is
        raw counts, so it takes the calibration the column already carries.

        Returning None rather than raising on a mismatch is deliberate: an
        uncorrected trace is honest, a trace with somebody else's pedestal
        subtracted is not.
        """
        if not self.win.chk_dark.isChecked():
            return None
        if self.win._pair_live():
            darks = self.win.background_members
            if darks is None:
                return None
            try:
                return self.win.spec.combined_dark(darks)
            except Exception:
                return None        # mismatched shapes: better raw than wrong
        if self.win.background is None:
            return None
        return self.win.spec.calibrate(self.win.background)

    def _warn_dark_mismatch(self):
        """Say once that a recorded dark does not fit the live pair.

        One shot only — the callers run every 60 ms and would bury the status
        bar. Re-armed whenever a dark is recorded or dropped.
        """
        if self.win._dark_member_warned:
            return
        self.win._dark_member_warned = True
        self.win.status.showMessage(
            "The recorded dark does not match this pair — not subtracted. "
            "Re-record it.", 6000)

    def _flush_pending_fit(self):
        """Run the one-shot fit a device swap queued, once its first frame is
        actually on the axes.

        Driven from the render, not from frame arrival: fit_xy() fits to what
        the curves currently hold, and arrival leads the render by up to one
        display tick. Fitting on arrival therefore had to guess a delay long
        enough to outlast the tick, and on a starved event loop it guessed
        wrong and fitted the OUTGOING device's wavelength range.
        """
        if self.win._pending_fit:
            self.win._pending_fit = False
            self._autofit_spectrum()
        elif self.win._pending_rescale:
            # elif: a device swap's full fit is the stronger action and already
            # covers the rescale, so it must not be undone by one queued behind it.
            self.win._pending_rescale = False
            self._rescale_spectrum()

    def _render_overlay(self, frames=None):
        """Draw one curve per stitched member. False = nothing to draw yet, so
        the caller falls back to the combined frame.

        Runs the SAME prepare_pair() the merged frame does, then applies each
        member's scale, so the two curves and the merged one are two views of
        one computation rather than two derivations that can drift. Each member
        keeps its NATIVE pixel grid; no interpolation happens here.

        `frames` are the RAW member frames to draw. Passed in by the scan, which
        holds the pair its worker measured the current column with; the live
        feed omits it and gets the device's newest pair instead. Same
        arrangement, and for the same reason, as _corrected_frame's.

        Reads self.win.spec.last_member_raw from the GUI thread without the device
        lock, exactly as _update_saturation does: the tuple assignment is
        atomic, and calibrate() plus the wavelength arrays are pure.
        """
        spec = self.win.spec
        if frames is None:
            frames = getattr(spec, "last_member_raw", None)
        slots = self.win.devices._slot_members()
        if frames is None or len(slots) != 2:
            return False
        darks = None
        if self.win.chk_dark.isChecked():
            darks = self.win.devices._usable_member_darks()
            if darks is None:
                self._warn_dark_mismatch()
        prepared = spec.prepare_pair(*frames, darks)
        by_id = {id(m): y for m, y in zip(spec.members, prepared)}
        curves = []
        for mem in slots:
            y = by_id.get(id(mem))
            if y is None:
                return False
            idx = spec.member_index(mem)
            # output_wavelengths: y is CALIBRATED, and legacy mode trims it.
            curves.append((np.asarray(mem.output_wavelengths, float),
                           y * spec.member_scale(idx)))
        # Cheap and idempotent — set_overlap_band returns immediately unless
        # the band actually moved.
        self.win.canvas.set_overlap_band(*spec.overlap_band)
        self.win.canvas.update_member_spectra(curves[0][0], curves[0][1],
                                          curves[1][0], curves[1][1])
        return True

    def _autofit_spectrum(self):
        """One-shot fit of spectrum X + Y; syncs Graphics Settings spinboxes."""
        self.win.canvas.fit_xy()
        if hasattr(self.win, 'dlg_graphics'):
            self.win.dlg_graphics.sync_limits()

    def _rescale_spectrum(self):
        """Refit the spectrum to the current frame, LEAVING auto-scale as it was.

        fit_xy() freezes both flags — right for a device swap, where the new
        spectrometer's range is a fresh start, but wrong for an exposure change:
        turning the operator's auto-scale off as a side effect of typing in a
        new integration time is not something they asked for. So the flags are
        put back afterwards.

        Auto-scale Y goes back on through set_autoscale_y, which also drops the
        damping filter's state — otherwise the filter would still be converging
        on the old exposure's levels and would spend its decay constant crawling
        off the fit that was just made.
        """
        c = self.win.canvas
        was_x, was_y = c.autoscale_x, c.autoscale_y
        c.fit_xy()                     # fits both axes, freezes both flags
        c.set_autoscale_x(was_x)
        if was_y:
            c.set_autoscale_y(True)
        if hasattr(self.win, 'dlg_graphics'):
            self.win.dlg_graphics.sync_limits()

    def _freeze_reference(self):
        """Capture the spectrum panel as it stands.

        Replaces any reference already held rather than refusing: this button is
        reached for after every adjustment, and making that "drop, then freeze"
        would be two presses for the one thing anyone ever wants.
        """
        if not self.win.canvas.freeze_reference():
            self.win.status.showMessage(
                "Nothing on the spectrum panel to freeze yet.", 4000)
            return
        self.win.status.showMessage(
            "Spectrum frozen as a grey reference — press + again to replace "
            "it, the eye to hide it, or right-click + to drop it.", 6000)

    def _toggle_reference(self):
        self.win.canvas.set_reference_visible(not self.win.canvas.reference_shown())

    def _clear_reference(self):
        """Right-click on +. Silent when there is nothing held: a right-click
        that reports having done nothing is just noise."""
        if self.win.canvas.has_reference():
            self.win.canvas.clear_reference()
            self.win.status.showMessage("Reference spectrum dropped.", 3000)

    def _on_autoscale_clicked(self):
        """Toggle the spectrum panel's auto-scale.

        Routed through the canvas' own reset_axes rather than set_autoscale_y:
        that is exactly what a right-click on the panel does, so the button and
        the right-click cannot end up meaning different things. It emits
        limits_changed on the way out, which is what puts the button's own
        checked state back in agreement with the canvas — including when the
        canvas refuses to move.
        """
        self.win.canvas.reset_axes(self.win.canvas.ax_spec)

    def _toggle_feed(self, on):
        if on:
            # During a scan the worker owns the hardware; arm the feed and let
            # _reset_scan_ui start it once the scan hands the devices back.
            if not self.win.scanning.running():
                self.win._feed.resume()
        else:
            self.win._feed.pause()
        self.win.appearance._refresh_feed_button()

    def _apply_integration_time(self):
        """Push the (debounced) integration time(s) to the spectrometer.

        A stitched pair takes one value per member, from the S1/S2 spinboxes
        in slot order; anything else takes the single spinbox. Both boxes
        share this one debounce, so editing them together costs a single feed
        handover.
        """
        if self.win.scanning.running():
            self.win.status.showMessage(
                "A scan is running — integration time applies to the next scan.", 4000)
            return
        if not self.win._have_spec():
            return
        slots = self.win.devices._slot_members() if self.win._pair_live() else []
        changed = []
        with self.win.devices.device_lock() as ok:
            if not ok:
                self.win.status.showMessage(FEED_BUSY_MSG, 4000)
                return
            try:
                if len(slots) == 2:
                    for i, (mem, spin) in enumerate(
                            zip(slots, (self.win.spin_integration,
                                        self.win.spin_integration2))):
                        ms = float(spin.value())
                        # Compared at the box's own resolution, not exactly:
                        # the Avantes stores the exposure as a C float, so
                        # 0.02 ms reads back as 0.019999999552965164 and an
                        # equality test would rewrite both devices — and take
                        # a feed handover — every time the debounce fired.
                        dp = spin.decimals()
                        if round(_exposure_ms(mem), dp) == round(ms, dp):
                            continue   # already there — don't disturb the device
                        self.win.spec.set_member_integration_time(
                            self.win.spec.member_index(mem), ms)
                        changed.append(f"S{i + 1} {ms:g} ms")
                else:
                    ms = float(self.win.spin_integration.value())
                    # Same resolution-limited compare as the pair branch above,
                    # and for the same reason — but here it also decides whether
                    # the dark is thrown away, so a debounce that fires without
                    # the value having moved must not cost the operator a dark.
                    dp = self.win.spin_integration.decimals()
                    if round(_exposure_ms(self.win.spec), dp) != round(ms, dp):
                        self.win.spec.set_integration_time(ms)
                        changed.append(f"{ms:g} ms")
            except Exception as e:
                self.win.status.showMessage(f"Integration time failed: {e}", 5000)
                return
        if not changed:
            return
        # Signal scales with exposure, so the counts axis is now the wrong size
        # — dropping the exposure 10x leaves the trace in the bottom tenth of
        # the panel. Refit once, on the next frame rather than now: the frame on
        # screen was taken at the OLD exposure and fitting to it would just
        # reproduce the range being escaped.
        self.win._pending_rescale = True
        # And drop the frame already in flight for the same reason — it was
        # acquired before the device was rewritten, so the fit must not land on
        # it. Latest-frame-wins means nothing else is waiting behind it.
        self.win._live_frame = None
        # A dark holds the pedestal of the exposure it was taken at, so it is
        # now wrong — on a single device as much as on a pair, which used to go
        # unmentioned entirely.
        had_dark = (self.win.background is not None
                    or self.win.background_members is not None)
        msg = None
        if len(slots) == 2:
            # Frames stay in raw counts, so stitch_factor now carries the wrong
            # exposure ratio and the seam will show. Say so rather than let the
            # user discover it in the data.
            self.win._stitch_stale = True
            msg = (f"{', '.join(changed)} — the two spectrometers no longer "
                   f"share a scale; re-run Auto-stitch (⇌ above the spectrum, "
                   f"or the Spectrometer window).")
            self.win.status.showMessage(msg, 8000)
        if had_dark and self.win.processing == "legacy":
            # The legacy program keeps the dark it has whatever the exposure
            # now is — and so does legacy mode, but it says so.
            rec = ", ".join(f"{e:g} ms"
                            for e in (self.win.background_exposures or ()))
            note = (f"Dark kept (legacy mode) — it was recorded at {rec}."
                    if rec else "Dark kept (legacy mode).")
            self.win.status.showMessage(
                f"{msg or ', '.join(changed) + '.'}  {note}", 10000)
        elif had_dark:
            # Last, so its persistent message is the one left standing.
            self.invalidate_dark(f"integration time changed to "
                                  f"{', '.join(changed)}")

    # Dark averaging: at least this many frames, and DARK_FRAMES_PER_AVG times
    # the scan's points-per-average, so the dark is always much quieter than
    # one scan point — but never more than DARK_BUDGET_MS of exposure, because
    # the capture blocks the GUI thread for its whole length.
    DARK_MIN_FRAMES = 16
    DARK_FRAMES_PER_AVG = 4
    DARK_MAX_FRAMES = 64
    DARK_BUDGET_MS = 5000.0

    def _dark_frame_count(self):
        """How many frames to average into a dark (see DARK_* above).

        Legacy mode takes exactly the scan's frames per point, as the legacy
        program's record_dark does (it averages navg) — no floor, cap or time
        budget."""
        if self.win.processing == "legacy":
            return max(1, int(self.win.dlg_settings.n_average))
        n = max(self.DARK_MIN_FRAMES,
                self.DARK_FRAMES_PER_AVG * int(self.win.dlg_settings.n_average))
        members = getattr(self.win.spec, "members", None)
        frame_ms = (sum(_exposure_ms(m) for m in members) if members
                    else _exposure_ms(self.win.spec))
        if frame_ms > 0:
            n = min(n, int(self.DARK_BUDGET_MS // frame_ms))
        return max(1, min(n, self.DARK_MAX_FRAMES))

    def _capture_dark(self):
        """Record the dark as the AVERAGE of several fresh frames.

        Averaged because the same dark is subtracted from every column of a
        scan: its noise is not averaged away along the delay axis the way each
        point's own noise is, so a single-frame dark prints its noise into the
        trace as stripes along delay. N frames cut that by sqrt(N).

        Always acquires rather than reusing self.win.last_spectrum: that field only
        moves while the display ticks, so with the feed stopped it holds the
        last LIT frame and "Record Dark" recorded the light. Acquiring also
        means the merged frame and the member frames come from the same
        exposures instead of two different ones.
        """
        if self.win.scanning.running():
            self.win.status.showMessage("A scan is running — spectrometer is busy.", 3000)
            return
        if not self.win._have_spec():
            self.win.status.showMessage("Connect a spectrometer first.", 4000)
            return
        with self.win.devices.device_lock() as ok:
            if not ok:
                self.win.status.showMessage(FEED_BUSY_MSG, 4000)
                return
            n = self._dark_frame_count()
            self.win.status.showMessage(f"Recording dark — {n} frames…", 0)
            self.win.status.repaint()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            acc = m_acc = None
            try:
                for _ in range(n):
                    f = np.asarray(self.win.spec.acquire(), float)
                    acc = f.copy() if acc is None else acc + f
                    # Inside the lock: these frames belong to the acquire just
                    # made, and the feed cannot overwrite last_member_raw.
                    mf = getattr(self.win.spec, "last_member_raw", None)
                    if mf is not None:
                        if m_acc is None:
                            m_acc = [np.asarray(x, float).copy() for x in mf]
                        else:
                            for a, x in zip(m_acc, mf):
                                a += x
            except Exception as e:
                self.win.status.showMessage(f"Dark capture failed: {e}", 5000)
                return
            finally:
                QApplication.restoreOverrideCursor()
            frame = acc / n
            members = self.win.devices._slot_members()
            frames = None if m_acc is None else tuple(a / n for a in m_acc)
            if len(members) == 2 and frames is not None:
                # Stored positionally in self.win.spec.members order — NOT keyed by
                # id(), which is recycled across reconnects; _apply_spectrometer
                # clears the field on every swap, so the order cannot go stale.
                self.win.background = None
                self.win.background_members = tuple(np.asarray(f, float).copy()
                                                for f in frames)
                exposures = tuple(_exposure_ms(m) for m in self.win.spec.members)
                where = ", ".join(
                    f"S{i + 1} {_exposure_ms(m):g} ms"
                    for i, m in enumerate(members))
            else:
                self.win.background = frame
                self.win.background_members = None
                exposures = (_exposure_ms(self.win.spec),)
                where = f"{_exposure_ms(self.win.spec):g} ms"
        self.win.background_exposures = exposures
        self.win.last_spectrum = frame
        self.win._dark_member_warned = False
        self.win.chk_dark.setEnabled(True); self.win.chk_dark.setChecked(True)
        self.win.status.showMessage(
            f"Dark recorded — {where}, average of {n} frames.", 4000)

    def dark_matches_exposure(self):
        """Whether the recorded dark was taken at the exposure(s) the device
        has now. Only legacy mode can leave them apart (it keeps the dark across
        an exposure change). True when there is no dark to mismatch."""
        rec = self.win.background_exposures
        if rec is None or not self.win._have_spec():
            return True
        members = getattr(self.win.spec, "members", None)
        now = (tuple(_exposure_ms(m) for m in members) if members
               else (_exposure_ms(self.win.spec),))
        return len(now) == len(rec) and all(
            abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(now, rec))

    def invalidate_dark(self, reason):
        """Throw the recorded dark away and say why.

        Used when something changes what a frame CONTAINS — an exposure, or an
        Avantes on-board correction. A dark is offset + dark-current x t, so
        rescaling it by an exposure ratio is wrong in general and would leave a
        plausible-looking but incorrect baseline; better to have none. No-op
        when there is nothing recorded, so callers need not check.

        Not needed for a stitch-factor or overlap-band change: the merged dark
        is derived from the member frames, so it follows both on its own.
        """
        if (self.win.background is None and self.win.background_members is None):
            return
        self.win.background = None
        self.win.background_members = None
        self.win.background_exposures = None
        self.win._dark_member_warned = False
        self.win.chk_dark.setChecked(False); self.win.chk_dark.setEnabled(False)
        # Timeout 0 — a silently dropped dark is exactly the kind of thing that
        # should not scroll away before the operator looks up.
        self.win.status.showMessage(f"Dark discarded — {reason}. Re-record it.", 0)
