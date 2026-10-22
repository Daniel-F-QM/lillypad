"""The three panels' own content: what each one draws and how it is updated.

One mixin per panel, in the order they appear on screen:

  * **SpectrumPanelMixin** — the live curve, the per-spectrometer overlay with
    its crossfade band, the frozen reference, and the deviation strip the
    alignment sweep writes into. The busiest of the three, because it is the one
    with modes.
  * **TracePanelMixin** — the FROG trace image, its colormap and threshold, and
    the symmetry fold line.
  * **AcPanelMixin** — the autocorrelation, its half-maximum line and the width
    readout.

They are separate classes rather than three sections of one so that "what does
the trace panel do" is answerable by opening one class, and so that adding a
fourth panel is adding a file rather than finding the right 200 lines.

Two performance details in here are measured and must not be tidied away. The
trace image is rasterized at DISPLAY resolution — rows max-pooled to about one
per screen pixel, max rather than mean so a narrow spectral line cannot vanish —
and `interpolation_stage="data"` colours the resampled result instead of
colouring a million values and resampling the RGBA. Together those took a column
from ~93 ms to ~17 ms.
"""
import numpy as np

from lillypad.design.mpl_style import _AC_YLIM
from lillypad.design.tokens import DIFF_COLORS


class SpectrumPanelMixin:
    """The spectrum panel: live curve, members, reference, deviation strip."""

    @property
    def linewidth(self):
        """Curve width in points, as last set. Read back by the Graphics window
        to seed its spinbox and by the settings table to save it."""
        return self._lw

    def set_log_scale(self, on):
        self.ax_spec.set_yscale('log' if on else 'linear')
        if on:
            lo, hi = self.ax_spec.get_ylim()
            if hi <= 0:
                # Entirely nonpositive view (possible after a zoom): a bottom
                # clamp alone would leave an inverted log axis.
                self.ax_spec.set_ylim(1.0, 10.0)
            elif lo <= 0:
                self.ax_spec.set_ylim(bottom=max(1.0, hi * 0.001))
        # Pinned, not just cached: the filter's state is in the OLD scale's
        # units, and easing a linear-space limit toward a log-space target
        # would walk the view somewhere neither of them asked for.
        self._pin_ylim(self.ax_spec.get_ylim())
        self.draw_idle()

    def set_linewidth(self, lw):
        self._lw = lw
        self.line_spec.set_linewidth(lw)
        self.line_ac.set_linewidth(lw)
        self.line_m1.set_linewidth(lw)
        self.line_m2.set_linewidth(lw)
        # The reference is a spectrum like any other — a frozen curve drawn at
        # a width the live ones are not would read as a different kind of thing.
        for ln in (self.ref_spec, self.ref_m1, self.ref_m2):
            ln.set_linewidth(lw)
        self.draw_idle()

    # ── Spectrum panel: combined curve vs one curve per spectrometer ──────
    def _set_spec_mode(self, overlay):
        """Switch the spectrum panel between the combined curve and the two
        per-spectrometer curves. Idempotent — every update_* path calls it.

        Whichever lines go dark have their data CLEARED, not just hidden:
        matplotlib's relim() defaults to visible_only=False, so a hidden line
        still holding a frame would go on driving the y auto-scale.
        """
        if overlay == self._overlay:
            return
        self._overlay = overlay
        self.line_spec.set_visible(not overlay)
        self.line_m1.set_visible(overlay)
        self.line_m2.set_visible(overlay)
        # The band only means anything next to the two curves it describes: on
        # the combined trace the crossfade has already happened and shading it
        # would suggest there is still something to see there.
        self.band_span.set_visible(overlay and self._band is not None)
        for ln in ((self.line_spec,) if overlay else (self.line_m1, self.line_m2)):
            ln.set_data([], [])
        # The frozen reference is deliberately NOT touched here: it stays on the
        # panel across the toggle, in the shape it was frozen in.
        #
        # Switching views is rare and changes which curves exist on screen; a
        # full draw re-caches both blit backgrounds, so nothing from the
        # previous view can survive in them. _request_full (not draw_idle)
        # keeps it batch-safe — a scan column reaches here inside batch().
        self._request_full()

    def set_overlap_band(self, lo_nm, hi_nm):
        """Shade the stitched pair's overlap band, or hide it when passed None.

        No-op when the band has not moved: this is called from every overlay
        frame (60 ms), and a full redraw per frame to re-place an unchanged
        patch would cost far more than the curves themselves.
        """
        band = None if lo_nm is None or hi_nm is None else (float(lo_nm),
                                                            float(hi_nm))
        if band == self._band:
            return
        self._band = band
        if band is not None:
            lo, hi = band
            # axvspan gives a Rectangle drawn in data-x / AXES-y, so setting
            # the bounds to full height keeps it spanning the panel whatever
            # the y limits autoscale to.
            self.band_span.set_bounds(lo, 0.0, hi - lo, 1.0)
        self.band_span.set_visible(self._overlay and band is not None)
        self._request_full()      # static artist: it lives in the blit cache

    def clear_members(self):
        """Drop back to the combined curve and forget the member frames —
        used on a device swap, so a dead pair's curves cannot linger."""
        self._set_spec_mode(False)
        self.line_m1.set_data([], [])
        self.line_m2.set_data([], [])
        self.set_overlap_band(None, None)

    def clear_spectrum(self):
        """Empty the spectrum panel entirely — no device is connected.

        Data cleared, not just hidden, for the reason in _set_spec_mode: a line
        still holding a frame keeps driving relim(), so the axes would go on
        describing a spectrometer that is no longer there. A full redraw, not a
        blit: the "no spectrometer" overlay sits on top of this, and a stale
        blit background underneath it would show the old curve again the moment
        anything else animated.
        """
        self.clear_members()
        self.line_spec.set_data([], [])
        self._request_full()

    # ── Spectrum panel: the frozen reference ──────────────────────────────
    # A grey copy of the spectrum as it was at some moment, kept under the live
    # curves so an adjustment can be judged against what preceded it. Nothing
    # here touches the recorded data or any export: it is a second set of
    # artists on the panel and nothing else.
    def freeze_reference(self):
        """Snapshot the curves now on the spectrum panel. False if it is empty.

        Takes what is PLOTTED rather than a frame from upstream. That is the one
        description of the panel already correct in both views and under every
        correction applied to get there (dark, calibration, stitch factor), so
        a reference can never disagree with the curve it was taken from.

        Copied, not referenced: set_data keeps whatever array it was handed, and
        the next frame is free to reuse that buffer — a reference quietly
        tracking the live signal is the one failure this must not have.
        """
        src = (self.line_m1, self.line_m2) if self._overlay else (self.line_spec,)
        curves = [(np.array(ln.get_xdata(), dtype=float),
                   np.array(ln.get_ydata(), dtype=float)) for ln in src]
        if not all(x.size for x, _ in curves):
            return False
        self._ref = curves
        self._ref_shown = True     # freezing something you cannot see is a trap
        self._sync_reference()
        return True

    def clear_reference(self):
        """Drop the snapshot entirely — the panel keeps no memory of it."""
        self._ref = None
        self._sync_reference()

    def set_reference_visible(self, on):
        self._ref_shown = bool(on)
        self._sync_reference()

    def has_reference(self):
        return self._ref is not None

    def reference_shown(self):
        return self._ref is not None and self._ref_shown

    def _sync_reference(self):
        """Put the reference artists in step with the snapshot and the eye
        toggle, then re-cache the background.

        Which artists carry it follows the shape of the SNAPSHOT, not the view
        on the panel: a reference frozen apart stays two curves, and a combined
        one stays a single curve, whichever way the live spectrum is being
        drawn over them.

        Data is CLEARED off whichever artists go dark rather than only hidden,
        for the reason in _set_spec_mode: relim() is called with
        visible_only=True, so an array left on a hidden line is invisible to the
        auto-scale right up until it is shown again and is not.
        """
        curves = self._ref if self._ref_shown and self._ref is not None else ()
        targets = (self.ref_m1, self.ref_m2) if len(curves) == 2 else (self.ref_spec,)
        live = set()
        for ln, (x, y) in zip(targets, curves):
            ln.set_data(x, y)
            ln.set_visible(True)
            live.add(id(ln))
        for ln in (self.ref_spec, self.ref_m1, self.ref_m2):
            if id(ln) not in live:
                ln.set_data([], [])
                ln.set_visible(False)
        self._request_full()      # static artist: it lives in the blit cache
        self.reference_changed.emit()

    # ── Alignment mode: the deviation panel ───────────────────────────────
    def set_dev_visible(self, on):
        """Open or close the deviation panel under the spectrum.

        Opening it takes room from the spectrum, so the whole figure is laid
        out again — and the spectrum hands over its wavelength ticks, which is
        a decoration change, hence both calls.
        """
        on = bool(on)
        if on == self._dev_on:
            return
        self._dev_on = on
        self.ax_dev.set_visible(on)
        if on:
            # Adopt the spectrum's wavelength range NOW. The panel is stacked
            # flush under it sharing that axis, and nothing else would sync it
            # until the next thing to move the spectrum's x — which, with
            # auto-scale x frozen, might be never.
            self._sync_dev_x()
        self._apply_mode_decorations()
        self._layout_axes()          # ends in draw_idle()

    def dev_visible(self):
        return self._dev_on

    def _clear_ghosts(self):
        for g in self._ghosts:
            g.remove()
        self._ghosts = []
        self._ghost_y = ()      # …and what the auto-scale knew about them

    def show_diff(self, wl, d1, d2):
        """Draw one alignment sweep's two symmetry differences.

        Whatever was on the panel becomes a semi-transparent area under the new
        curves first, so a refresh reads as "this against the last one" — which
        is the whole question a sweep is run to answer. Exactly ONE generation
        is kept: the comparison that matters is with the state before the last
        adjustment, and a pile of older fills would just darken the panel.

        Fills are removed and rebuilt rather than updated, because fill_between
        returns a collection with no set_data. A sweep is four stage moves, so
        the allocation is free next to what produced the data.
        """
        self._clear_ghosts()
        if self._prev_diff is not None:
            pwl, pd1, pd2 = self._prev_diff
            for prev, color in ((pd1, DIFF_COLORS[0]), (pd2, DIFF_COLORS[1])):
                self._ghosts.append(self.ax_dev.fill_between(
                    pwl, 0.0, prev, color=color, alpha=0.18, linewidth=0,
                    zorder=1))
            # Recorded separately from _prev_diff, which is about to become the
            # sweep being drawn NOW: the auto-scale needs what is on the panel,
            # and after the line below _prev_diff no longer describes the ghost.
            self._ghost_y = (pd1, pd2)
        self._prev_diff = (np.asarray(wl, float).copy(),
                           np.asarray(d1, float).copy(),
                           np.asarray(d2, float).copy())
        self.line_d1.set_data(wl, d1)
        self.line_d2.set_data(wl, d2)
        self.line_d1.set_visible(True)
        self.line_d2.set_visible(True)
        self.diff_zero.set_visible(True)
        self._sync_dev_x()
        self._autoscale_dev()
        self._request_full()

    def clear_diff(self):
        """Empty the deviation panel — both curves and the remembered sweep.

        Data cleared as well as hidden, for the reason in _set_spec_mode: a
        hidden line still holding a frame keeps driving relim().
        """
        self._clear_ghosts()
        self._prev_diff = None
        self.line_d1.set_data([], [])
        self.line_d2.set_data([], [])
        self.line_d1.set_visible(False)
        self.line_d2.set_visible(False)
        self.diff_zero.set_visible(False)
        self._request_full()

    def diff_visible(self):
        return bool(self.line_d1.get_visible())

    def update_spectrum(self, wl, spectrum):
        # Last writer owns the panel: whoever has a combined frame to show
        # (live feed, scan column) takes it back from the overlay without any
        # coordination, and the overlay takes it back on its next frame.
        self._set_spec_mode(False)
        self.line_spec.set_data(wl, spectrum)
        changed = False
        if self.autoscale_x:
            xl = (float(wl[0]), float(wl[-1]))
            if xl != self._xlim_cache:
                self.ax_spec.set_xlim(*xl); self._pin_xlim(xl); changed = True
        if self.autoscale_y:
            changed |= self._autoscale_y()
        if changed:
            self._request_full()      # limits moved: full redraw + re-cache bg
        else:
            self._request_blit(spec_only=True)

    def update_member_spectra(self, wl1, s1, wl2, s2):
        """Draw one curve per spectrometer instead of the combined one.

        Each member sits on its OWN pixel grid, so the two x arrays differ and
        the panel spans their union — which is exactly the stitched grid's
        span, so toggling the view never jumps the x-axis.
        """
        if not (len(wl1) and len(wl2)):
            return
        self._set_spec_mode(True)
        self.line_m1.set_data(wl1, s1)
        self.line_m2.set_data(wl2, s2)
        changed = False
        if self.autoscale_x:
            xl = (float(min(wl1[0], wl2[0])), float(max(wl1[-1], wl2[-1])))
            if xl != self._xlim_cache:
                self.ax_spec.set_xlim(*xl); self._pin_xlim(xl); changed = True
        if self.autoscale_y:
            changed |= self._autoscale_y()
        if changed:
            self._request_full()
        else:
            self._request_blit(spec_only=True)


class TracePanelMixin:
    """The FROG trace panel: the image, its colour mapping and the fold line."""

    @property
    def cmap_name(self):
        """The colormap as named in TRACE_COLORMAPS, without the "_r" that
        reversing adds — the two are stored and saved separately."""
        return self._cmap_name

    @property
    def cmap_reversed(self):
        return bool(self._cmap_rev)

    # ── Symmetry mode: fold line + score readout ──────────────────────────
    def set_fold_line(self, x_fs):
        """Put the symmetry fold line at `x_fs` (delay, data coords), or hide
        it with None. Drawing it is what makes it draggable — _hit_fold reads
        the position back off the artist rather than keeping a second copy."""
        visible = x_fs is not None
        changed = visible != self.line_fold.get_visible()
        if visible:
            if float(self.line_fold.get_xdata()[0]) != float(x_fs):
                changed = True
            self.line_fold.set_xdata([x_fs, x_fs])
        self.line_fold.set_visible(visible)
        if changed:
            self._bg_static = None     # composited trace panel is now stale
            self._request_blit()

    def set_symmetry_score(self, text=""):
        """Asymmetry readout in the trace panel's corner. '' clears it."""
        if text == self.txt_sym.get_text():
            return                     # a drag re-reports the same number often
        self.txt_sym.set_text(text)
        self._bg_static = None
        self._request_blit()

    def set_cmap(self, name):
        self._cmap_name = name
        self._apply_cmap()
        self.draw_idle()

    def set_trace_reversed(self, on):
        self._cmap_rev = bool(on)
        self._apply_cmap()
        self.draw_idle()

    def set_trace_normalised(self, on):
        """Say whether the image now holds a normalised view (alignment's raw
        trace, each spectrometer scaled to its own peak) rather than counts.

        Only the colour bar's label depends on it — and the colour range: a
        manual range typed in counts means nothing on a 0…1 image (it would
        paint the whole trace in the bottom colour), so changing what the image
        holds hands the range back to auto.
        """
        on = bool(on)
        if on == self._trace_normalised:
            return
        self._trace_normalised = on
        self.cbar.set_label("Intensity (norm.)" if on else "Intensity (counts)")
        if not self.autoscale_clim:
            self.set_autoscale_clim(True)
        else:
            self._bg_static = None
            self._request_blit()

    def set_trace_threshold(self, percent):
        """Hide trace pixels below `percent` of the trace peak. Display only —
        the stored trace keeps every count."""
        self._trace_thresh = max(0.0, min(100.0, float(percent))) / 100.0
        self._render_trace()          # takes effect without waiting for a column

    def set_trace_smooth(self, on):
        """Turn the display-only delay-axis interpolation on or off."""
        self.trace_smooth = bool(on)
        self._render_trace()

    def refresh_trace_resolution(self):
        """Re-render after a relayout if the panel's width changed, so the
        delay interpolation keeps matching the pixels it is drawn on."""
        if not (self.trace_smooth and self._trace_raw is not None):
            return
        try:
            width = int(min(self.ax_trace.get_window_extent().width, 4096))
        except Exception:
            return
        if width != getattr(self, "_smooth_width", None):
            self._render_trace()

    @property
    def has_trace(self):
        """True once a scan has given the trace panel something to show."""
        return self._trace_raw is not None

    def _smooth_columns(self, shown):
        """Linearly interpolate `shown` along the delay axis to about one
        column per screen pixel. Display only — the stored trace is untouched.

        A scan has far fewer delay columns than the panel has pixels, so each
        column is stretched into a wide flat block. Interpolating between
        neighbouring columns removes the blocks without inventing anything a
        column did not measure. Each original column stays centred where it is
        drawn without smoothing: imshow lays n columns out as n equal cells
        across the extent, and the new grid samples those cell centres.
        The wavelength axis is left alone; its max-reduction above is what
        keeps narrow lines visible.
        """
        n = shown.shape[1]
        if n < 2:
            return shown
        try:
            width = self.ax_trace.get_window_extent().width
        except Exception:
            return shown
        if self._exporting:
            width *= 3          # the saved picture has ~3x the screen's pixels
        m = int(min(width, 4096))
        if not self._exporting:
            self._smooth_width = m      # what refresh_trace_resolution checks
        if m <= n:
            return shown        # already a column per pixel or finer
        x = (np.arange(m) + 0.5) * (n / m) - 0.5
        np.clip(x, 0.0, n - 1.0, out=x)
        i0 = np.floor(x).astype(np.intp)
        i1 = np.minimum(i0 + 1, n - 1)
        t = x - i0
        return shown[:, i0] * (1.0 - t) + shown[:, i1] * t

    def init_trace(self, delays, wl):
        # A new scan can span a different delay range; resume following it,
        # mirroring the trace-view reset below.
        self.autoscale_ac_x = True
        self.autoscale_ac_y = True
        self._trace_raw = np.zeros((wl.size, delays.size))
        self._trace_peak = 0.0       # empty trace; the scan's running max follows
        self._clim_peak = None
        self._bg_static = None
        self.im.set_data(self._trace_raw)
        # The extent is the data -> axes mapping and must always track the new
        # scan, even under manual bounds — otherwise the columns would land in
        # the wrong place. Only the VIEW is left alone when the user has dialled
        # bounds in by hand.
        self.im.set_extent([float(delays[0]), float(delays[-1]),
                            float(wl[0]), float(wl[-1])])
        if self.autoscale_trace:
            self.ax_trace.set_xlim(delays[0], delays[-1])
            self.ax_trace.set_ylim(wl[0], wl[-1])
            self._trace_xlim_cache = self.ax_trace.get_xlim()
            self._trace_ylim_cache = self.ax_trace.get_ylim()
        self._sync_ac_x()      # the AC follows the new delay range
        self.draw_idle()

    def update_trace(self, trace, peak=None):
        """Show `trace`. `peak` is its maximum when the caller already knows it.

        A scan in progress does: the trace is filled column by column, so it
        tracks a running max for free, while raw.max() here would walk the whole
        preallocated array (measured columns AND the zeros ahead of them) once
        per column. Callers that re-render a finished trace — the threshold,
        colormap and theme controls — pass None and it is derived as before.
        """
        self._trace_raw  = trace
        self._trace_peak = peak
        self._render_trace()

    def _display_decimation(self, n_rows):
        """Row-blocking factor that brings `n_rows` to about one row per pixel
        of the trace panel — 1 when the trace already fits.

        Always an exact divisor of n_rows, so the blocks tile the wavelength
        axis completely and the image extent set by init_trace stays correct
        with no remainder row to account for.

        Pools down to AT MOST one row per pixel, never merely close to it: the
        resampler then has a row for every pixel it draws and discards none of
        them, so the max-reduction in _render_trace is what fully decides which
        value each pixel shows. Stopping at 1024 rows for a 552 px panel would
        leave `nearest` free to drop half the blocks — and with them any narrow
        line that happened to fall in one.
        """
        # Printing: the page has three times the window's pixels (PICTURE_DPI),
        # so the panel's screen height is the wrong budget entirely — hand the
        # printer every row there is and let the resampler pick.
        if self._exporting:
            return 1
        try:
            h = self.ax_trace.get_window_extent().height
        except Exception:
            return 1                  # before the first draw: leave it alone
        f = 1
        while n_rows // f > h and n_rows % (f * 2) == 0:
            f *= 2
        return f

    def _render_trace(self):
        """Push the stored trace to the image, applying the display threshold.

        The colour scale is computed from the RAW trace, so moving the
        threshold only removes pixels — it never restretches the colours of
        the ones that survive.
        """
        raw = self._trace_raw
        if raw is None:
            return
        peak = max(float(raw.max() if self._trace_peak is None
                         else self._trace_peak), 1.0)
        # Reduce to display resolution FIRST. set_data invalidates the image's
        # cached RGBA, so the next draw re-normalizes, re-colormaps and
        # resamples everything handed over — and a scan hands it over once per
        # column. A 2048-row trace in a ~550 px panel therefore paid ~90 ms per
        # column to produce detail the panel cannot show. Blocks are reduced by
        # MAX, not by slicing: a narrow spectral line is exactly what must not
        # disappear, and `raw[::4]` drops three rows in four.
        f = self._display_decimation(raw.shape[0])
        shown = (raw.reshape(raw.shape[0] // f, f, raw.shape[1]).max(axis=1)
                 if f > 1 else raw)
        if self.trace_smooth:
            shown = self._smooth_columns(shown)
        data = (np.ma.masked_less(shown, self._trace_thresh * peak)
                if self._trace_thresh > 0 else shown)
        self.im.set_data(data)
        # Once a row is a screen pixel the antialiasing filter has nothing left
        # to average, and the max-reduction above already did the honest part.
        self.im.set_interpolation("nearest" if f > 1 else "auto")
        # set_clim invalidates the image's cached RGBA (full re-normalize +
        # re-colormap), so only touch it when the peak moved visibly (>0.5%).
        # Never under a manual colour range: that is the operator's, and the
        # next column must not snap it back to 0…peak.
        if self.autoscale_clim and (
                self._clim_peak is None
                or abs(peak - self._clim_peak) > 0.005 * self._clim_peak):
            self.im.set_clim(0, peak)
            self._clim_peak = peak
        self._bg_static = None        # trace pixels changed
        # Extent/limits are fixed by init_trace, so this is always a fast blit.
        self._request_blit()


class AcPanelMixin:
    """The autocorrelation panel: the curve, its half-maximum line and FWHM."""
    def update_ac(self, delays, ac):
        # Normalized to its own peak for DISPLAY only: the panel always reads
        # 0…1, so its tick labels stay short whatever the count level (they used
        # to grow a digit per decade and steal width from the plot beside them)
        # and a scan's partial curve fills the panel from the
        # first column. Nothing reads this line back — the stored trace, the
        # exports and the autocorrelation the scan computed keep raw counts.
        peak = float(ac.max()) if ac.size else 0.0
        shown = ac / peak if peak > 0 else ac
        self.line_ac.set_data(delays, shown)
        self._set_half_line(shown)
        self._bg_static = None        # AC line changed
        # The trace dictates the delay axis, so the AC's
        # own x autoscale never runs: _sync_ac_x is the single writer, and
        # init_trace has already pointed the trace at this scan's range.
        changed = self._sync_ac_x()
        if self.autoscale_ac_y and self._ac_ylim_cache != _AC_YLIM:
            # The normalized curve needs one fixed view, so this fires at most
            # once (after a zoom was reset). The old stepped autoscale existed
            # only because the raw AC peak grew with nearly every column, and
            # each ylim move costs a full redraw.
            self.ax_ac.set_ylim(*_AC_YLIM)
            self._ac_ylim_cache = _AC_YLIM; changed = True
        if changed:
            self._request_full()
        else:
            self._request_blit()

    def _set_half_line(self, shown):
        """Put the half-maximum rule at the level scan.fwhm() measured `shown`
        at, or hide it when that curve cannot be bracketed.

        The level is derived from the PLOTTED array with fwhm()'s own formula
        rather than pinned at 0.5. The two agree for a finished scan — the
        stored autocorrelation is already shifted to its minimum, and update_ac
        divides by its max — but not for everything that reaches this panel: a
        scan's first columns have barely any contrast, and a zoom or a symmetry
        re-render pushes a different slice through here. Recomputing costs one
        pass over a 1-D array and removes the whole class of "the line is not
        where the number says" by construction.
        """
        y = np.asarray(shown, float)
        # fwhm() gives up on these too, and it is the authority on the level:
        # a line drawn where no width could be measured is a claim we cannot
        # back with a number in the corner.
        if y.size < 3 or not np.any(y > 0):
            if self.line_half.get_visible():
                self.line_half.set_visible(False)
            return
        half = 0.5 * (float(np.nanmax(y)) + float(np.nanmin(y)))
        if not np.isfinite(half):
            self.line_half.set_visible(False)
            return
        self.line_half.set_ydata([half, half])
        self.line_half.set_visible(True)

    def set_fwhm(self, fs=None):
        """Autocorrelation width shown in the AC panel's top-right corner.
        None (or a non-finite value) clears it — there is no width to report."""
        text = ("" if fs is None or not np.isfinite(fs)
                else f"AC FWHM  {fs:.1f} fs")
        if text == self.txt_fwhm.get_text():
            return                   # a scan re-reports the same width often
        self.txt_fwhm.set_text(text)
        self._bg_static = None       # it is baked into that cache
        self._request_blit()
