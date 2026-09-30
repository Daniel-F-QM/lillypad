"""Where the axes point: the set_*/fit_*/reset surface, and auto-scaling.

Two kinds of thing, deliberately in one file, because they are two halves of one
question and every bug here has been one half disagreeing with the other:

  * the **explicit** setters (`set_xlim`, `set_trace_ylim`, `reset_axes`,
    `fit_xy`), which the Graphics window and the typed-bound editor drive, and
  * the **automatic** ones (`_autoscale_y`, `_autoscale_dev`), which run per
    frame off live data.

Turning auto-scale off has to pin the current view, and typing a bound has to
turn auto-scale off; those pairings are what `_pin_xlim` / `_pin_ylim` and the
`_set_*_autoscale` methods exist to keep true.

The spectrum's auto-Y is the subtle one. A live spectrum is noisy, and a limit
that tracked every frame's peak would shimmer, so the target is eased toward
with a time constant, asymmetrically — fast up so a real rise is visible at
once, slow down so a dropout does not rescale the panel — and only escalates a
frame to a full redraw once the limits have moved by more than a threshold. That
threshold is measured, not guessed: it took the full redraws during a live feed
from about 300 per 300 frames to zero.
"""
import math
import time

import numpy as np

from lillypad.design.mpl_style import _AC_YLIM


def _ease(current, target, dt, tau):
    """One-pole step from `current` toward `target` over `dt` seconds, with time
    constant `tau`.

    Solved rather than iterated, so the result depends only on elapsed TIME and
    not on how many times it was called: at a 1 ms exposure this runs ~16 times
    a second and at a 10 s exposure once every ten, and a per-call fraction
    would have made the response time a function of the exposure.
    """
    return target + (current - target) * math.exp(-dt / tau)


class AxesApiMixin:
    """Axis limits, auto-scaling and the fit/reset commands."""
    def reset_axes(self, ax):
        """Right-click: TOGGLE this panel's auto-scale.

        Off -> on fits the full data range and then follows it live. On -> off
        freezes the view exactly where it is, so a right-click is also how you
        stop the axes moving under you without opening Graphics Settings. It
        used to only ever switch autoscale on, which left the mouse with no way
        back off it.

        Freezing writes the current limits into the caches: they are what the
        Graphics Settings spinboxes and the blit paths read, and a stale cache
        would snap the view back on the next frame.
        """
        if ax is self.ax_spec:
            # autoscale_y alone decides which way this goes, NOT either flag.
            # The two are not equals: y is the continuous follow the operator
            # means by "auto-scale" (and what the header toggle displays),
            # while x defaults to on and only ever tracks the spectrometer's
            # wavelength span. Reading `x or y` made the very first right-click
            # from a fresh start — y off, x on — freeze the panel instead of
            # turning the follow on, so it took two to do what one says.
            self._set_spec_autoscale(not self.autoscale_y)
        elif ax is self.ax_trace:
            self._set_trace_autoscale(not self.autoscale_trace)
        elif ax is self.ax_ac:
            on = not (self.autoscale_ac_x or self.autoscale_ac_y)
            self._set_ac_autoscale(on)
            # Shared delay axis in both modes: the trace owns it, so it has to
            # go the SAME way rather than toggle off its own flag. (_sync_ac_x
            # in there then pulls the AC's x back onto the scan range.)
            self._set_trace_autoscale(on)
        elif ax is self.ax_dev:
            # Only ever fits: the panel holds one sweep's worth of data, so
            # there is nothing continuous here to freeze.
            self._autoscale_dev()
            self._sync_dev_x()
        self.draw_idle()
        self.limits_changed.emit()

    def _set_spec_autoscale(self, on):
        """Spectrum panel auto-scale, both axes together. On fits first."""
        if not on:
            self.autoscale_x = False           # freeze wherever it sits now
            self.autoscale_y = False
            self._pin_xlim(self.ax_spec.get_xlim())
            self._pin_ylim(self.ax_spec.get_ylim())
            return
        self.fit_xy()                          # fits, but freezes autoscale…
        self.autoscale_x = True                # …so re-enable live follow
        self.set_autoscale_y(True)

    def _set_trace_autoscale(self, on):
        """FROG-trace auto-scale. On restores the scan's full extent."""
        self.autoscale_trace = bool(on)
        if on:
            x0, x1, y0, y1 = self.im.get_extent()
            self.ax_trace.set_xlim(x0, x1)
            self.ax_trace.set_ylim(y0, y1)
        self._trace_xlim_cache = self.ax_trace.get_xlim()
        self._trace_ylim_cache = self.ax_trace.get_ylim()
        self._sync_ac_x()

    def _set_ac_autoscale(self, on):
        """Autocorrelation auto-scale, both axes together."""
        self.autoscale_ac_x = bool(on)
        self.autoscale_ac_y = bool(on)
        if not on:
            self._ac_xlim_cache = self.ax_ac.get_xlim()
            self._ac_ylim_cache = self.ax_ac.get_ylim()
            return
        # The curve is normalized, so y has one canonical view to go back to.
        self.ax_ac.set_ylim(*_AC_YLIM)
        self._ac_ylim_cache = _AC_YLIM
        if len(self.line_ac.get_xdata()) > 1:
            self.ax_ac.set_autoscalex_on(True)
            self.ax_ac.relim()
            self.ax_ac.autoscale_view(scaley=False)
            self._ac_xlim_cache = self.ax_ac.get_xlim()

    def fit_y(self):
        """One-shot Y auto-fit; returns new (ymin, ymax)."""
        self.ax_spec.set_autoscaley_on(True)
        self.ax_spec.relim(visible_only=True)
        self.ax_spec.autoscale_view(scalex=False)
        self._pin_ylim(self.ax_spec.get_ylim())
        self.draw_idle()
        return self.ax_spec.get_ylim()

    def fit_xy(self):
        """One-shot fit of both spectrum axes, then freezes auto-scale."""
        # Any earlier set_xlim/set_ylim (spinboxes, rubber-band zoom) turned
        # matplotlib's internal autoscale off, making autoscale_view a no-op.
        self.ax_spec.set_autoscalex_on(True)
        self.ax_spec.set_autoscaley_on(True)
        self.ax_spec.relim(visible_only=True)
        self.ax_spec.autoscale_view()
        self.autoscale_x = False
        self.autoscale_y = False
        self._pin_xlim(self.ax_spec.get_xlim())
        self._pin_ylim(self.ax_spec.get_ylim())
        self.draw_idle()

    def set_ylim(self, ymin, ymax):
        if ymin < ymax:
            self.ax_spec.set_ylim(ymin, ymax)
            self._pin_ylim((ymin, ymax))
            self.draw_idle()

    def set_xlim(self, xmin, xmax):
        if xmin < xmax:
            self.ax_spec.set_xlim(xmin, xmax)
            self._pin_xlim((xmin, xmax))
            self.draw_idle()

    # ── FROG trace: manual bounds and colour ──────────────────────────────
    def set_trace_xlim(self, xmin, xmax):
        """Delay axis (fs) of the FROG trace."""
        if xmin < xmax:
            self.ax_trace.set_xlim(xmin, xmax)
            self._trace_xlim_cache = (xmin, xmax)
            self._sync_ac_x()
            self.draw_idle()

    def set_trace_ylim(self, ymin, ymax):
        """Wavelength axis (nm) of the FROG trace."""
        if ymin < ymax:
            self.ax_trace.set_ylim(ymin, ymax)
            self._trace_ylim_cache = (ymin, ymax)
            self.draw_idle()

    def _autoscale_dev(self):
        """Fit the deviation panel to its curves AND the ghost under them.

        Derived from the DATA on every call, never from the view it is about to
        replace. set_ylim() switches matplotlib's own y auto-scale off, which
        makes a later autoscale_view() a silent no-op — so reading get_ylim()
        back as a starting point fed each refresh the PREVIOUS refresh's
        already-padded limits, and the panel ratcheted wider every sweep
        however small the new deviation was.

        The ghost has to be folded in by hand either way: relim() only looks at
        lines, and the previous sweep is a filled collection.
        """
        series = [np.asarray(ln.get_ydata(), float)
                  for ln in (self.line_d1, self.line_d2) if ln.get_visible()]
        series.extend(self._ghost_y)       # the shaded previous sweep, if any
        vals = [s for s in series if s.size and np.any(np.isfinite(s))]
        if not vals:
            return
        lo = min(float(np.nanmin(s)) for s in vals)
        hi = max(float(np.nanmax(s)) for s in vals)
        # Zero is the reference the whole panel is read against, so it is always
        # in view even when both sweeps sit to one side of it.
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        if hi <= lo:                     # an all-zero difference: show a band
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        # This panel's title is drawn INSIDE its top-left corner — the spectrum
        # sits flush above it, so there is no header band to put it in — and a
        # symmetry difference is usually one-sided, which parks the zero line
        # and the flat wings of both curves in exactly that spot. So the fit
        # above is given only the part of the panel BELOW the title band, and
        # the view stretched back out around it.
        frac = self._title_band_frac(self.ax_dev)
        if frac > 0.0:
            hi = lo + (hi - lo) / (1.0 - frac)
        self.ax_dev.set_ylim(lo, hi)
        self._dev_ylim_auto = (lo, hi)

    # Continuous y auto-scale is a FILTER, not a follower.
    #
    # A deadband alone was not enough. It rejects small moves, but every move it
    # accepts lands on the RAW instantaneous autoscale result — so a live
    # spectrum whose peak wanders a few percent frame to frame (shot noise, and
    # several percent is ordinary) cleared the old 2% guard constantly and the
    # view chased the noise both ways. Each accepted nudge also turns a ~1 ms
    # blit into a draw_idle(), which once a scan has put a real trace in the
    # image re-rasterizes that too, so the jitter came with a stutter.
    #
    # Instead the raw result is a TARGET fed through a one-pole filter with
    # asymmetric time constants: expand fast, contract slowly. The view settles
    # on the ceiling of the noise envelope and stays there — a real change still
    # shows within a fraction of a second, but a spike that decays again never
    # pulls the view back down. The deadband survives on top, now deciding only
    # how coarsely the filtered value is committed (and so how often the
    # expensive redraw is paid for).
    #
    # Contraction therefore runs in three regimes, because "slow to shrink" is
    # only right while the discrepancy is noise-sized:
    #
    #   far   (gap > _YLIM_STEP_FRAC of the view)  the signal really did drop;
    #         contract at the FAST constant, and LATCH that until it converges.
    #   near  (gap between the two thresholds)     still closing a latched move,
    #         or ordinary wander; whichever the latch says.
    #   home  (gap < _YLIM_HYST of the view)       converged; drop the latch and
    #         go back to rejecting noise, which the deadband then absorbs.
    #
    # The latch is what makes a real drop feel instant. Without it the fast
    # constant would hand back to the slow one the moment the gap fell under
    # _YLIM_STEP_FRAC, leaving the last fifth of the move to crawl — a smaller
    # version of exactly the rubber-band this is meant to remove.
    _YLIM_TAU_UP    = 0.15  # s — view growing to meet new data

    _YLIM_TAU_DOWN  = 2.0   # s — view shrinking after the signal drops

    _YLIM_HYST      = 0.04  # commit deadband, fraction of the applied span

    _YLIM_STEP_FRAC = 0.20  # gap this big is a real change, not noise

    def _autoscale_y(self):
        """Autoscale the spectrum's y axis; True if the view actually moved.

        A rejected move RESTORES the previous limits rather than merely leaving
        the cache alone: autoscale_view() has already written the new ones to
        the axes, and keeping them while taking the blit path would draw the
        curve over a cached background whose ticks were rendered for the old
        ones.

        autoscale_view() is kept as the source of the target rather than reading
        dataLim directly: it already accounts for the 5% margins, the log
        transform set_log_scale installs and visible_only's overlay-line
        bookkeeping, none of which is worth reimplementing here.
        """
        self.ax_spec.set_autoscaley_on(True)
        self.ax_spec.relim(visible_only=True)
        self.ax_spec.autoscale_view(scalex=False)
        target = self.ax_spec.get_ylim()          # raw, instantaneous
        old = self._ylim_cache
        if old is None or self._ylim_smooth is None:
            self._pin_ylim(target)                # nothing to filter from yet
            return True
        now = time.monotonic()
        dt  = max(now - self._ylim_t, 0.0)
        self._ylim_t = now
        slo, shi = self._ylim_smooth
        # How far the target sits from where the filter currently is, relative
        # to the view it would be moving. Both ends share one decision: they
        # describe one axis, and letting them latch separately would contract
        # the top fast while the bottom crawled.
        span_s = max(abs(shi - slo), 1e-12)
        gap    = max(abs(target[0] - slo), abs(target[1] - shi))
        if gap > self._YLIM_STEP_FRAC * span_s:
            self._ylim_fast = True     # a real change: stop easing, go there
        elif gap < self._YLIM_HYST * span_s:
            self._ylim_fast = False    # arrived: back to rejecting noise
        tau_down = self._YLIM_TAU_UP if self._ylim_fast else self._YLIM_TAU_DOWN
        # Asymmetric about the CURRENT filter state, not the applied limits: the
        # bottom expands by going lower, the top by going higher, so each end
        # gets the fast constant only in the direction that grows the view —
        # unless the latch above has already said this is not noise.
        lo = _ease(slo, target[0], dt,
                   self._YLIM_TAU_UP if target[0] < slo else tau_down)
        hi = _ease(shi, target[1], dt,
                   self._YLIM_TAU_UP if target[1] > shi else tau_down)
        self._ylim_smooth = (lo, hi)
        span = max(abs(old[1] - old[0]), 1e-12)
        if (abs(lo - old[0]) > self._YLIM_HYST * span
                or abs(hi - old[1]) > self._YLIM_HYST * span):
            # Commit the FILTERED value, never the target — committing the
            # target here would put the noise straight back on screen and make
            # the filter nothing but a delay on the same jitter.
            self.ax_spec.set_ylim(lo, hi)
            self._ylim_cache = (lo, hi)
            return True
        self.ax_spec.set_ylim(*old)
        return False

    def set_autoscale_x(self, on):
        """Turn the spectrum's x auto-scale on or off.

        A method rather than a bare attribute write from outside, to match
        set_autoscale_y: the two are set together by every caller, and one of
        them having to re-seed a filter is an implementation detail the caller
        should not have to know about.
        """
        self.autoscale_x = bool(on)

    def set_autoscale_trace(self, on):
        """Turn the FROG trace's auto-scale on or off.

        Only the FLAG: turning it off leaves the view exactly where it is, and
        the caller pins the four manual boxes to it. Turning it back on is what
        restores the scan's full extent — that is _set_trace_autoscale.
        """
        self.autoscale_trace = bool(on)

    def set_autoscale_y(self, on):
        """Turn continuous y auto-scale on or off.

        Switching it ON drops the filter state, so the next frame SNAPS onto the
        data instead of easing toward it. Without that, ticking the checkbox
        from a manual view — or restoring auto-scale at startup, where the
        spinbox defaults have already put 0…5000 counts on the axes — would
        spend ten seconds crawling down from limits that were never about the
        live signal. The filter exists to reject noise, not to animate a
        deliberate change of mode.
        """
        on = bool(on)
        if on and not self.autoscale_y:
            self._ylim_smooth = None
            self._ylim_fast   = False    # no move in flight to stay fast for
        self.autoscale_y = on

    def _pin_xlim(self, xl):
        """Record `xl` as the spectrum's applied x limits and mirror them onto
        the deviation panel.

        Every path that moves the spectrum's wavelength view goes through here,
        for the same reason _pin_ylim exists below: the two panels are stacked
        flush on one wavelength axis, and a bound written straight to
        _xlim_cache would leave the deviations describing a different stretch of
        spectrum than the curve directly above them.
        """
        self._xlim_cache = tuple(xl)
        self._sync_dev_x()

    def _sync_dev_x(self):
        """Point the deviation panel at the spectrum's wavelength range.
        Returns True if it moved. The spectrum owns the axis; this mirrors it,
        exactly as _sync_ac_x mirrors the trace's delay axis."""
        xl = self.ax_spec.get_xlim()
        if xl == self._dev_xlim_cache:
            return False
        self.ax_dev.set_xlim(*xl)
        self._dev_xlim_cache = xl
        return True

    def _pin_ylim(self, yl):
        """Record `yl` as both the applied limits AND the auto-scale filter's
        state.

        Every path that moves the y view by hand goes through here. Setting only
        _ylim_cache would leave the filter still converging on where the view
        USED to be, and the next few live frames would walk it straight back off
        the zoom, fit or manual limits the operator just asked for.
        """
        self._ylim_cache  = tuple(yl)
        self._ylim_smooth = tuple(yl)
        self._ylim_t      = time.monotonic()
        # The view IS where the filter wants to be, so any contraction that was
        # in flight is over — leaving the latch set would spend the next frames
        # racing away from limits that were just chosen deliberately.
        self._ylim_fast   = False
