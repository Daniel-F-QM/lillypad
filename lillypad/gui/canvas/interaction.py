"""Mouse work on the canvas: zoom, axis-bound editing and the fold line.

Three gestures share one press/motion/release chain, so they live together:

  * **rubber-band zoom** on any panel, with the band drawn as a Qt widget over
    the canvas rather than as an artist, so dragging it costs no redraw;
  * **click an end tick label** to type a bound (`_hit_axis_end` →
    `_open_axis_editor` → `apply_axis_bound`);
  * **drag the symmetry fold line**, which only reports — the window decides
    what a new fold means.

The trace's colour bar takes part in the first two like any panel, with the
colour range standing in for its view: drag up or down it to pick the range,
click an end label to type one, right-click to follow the peak again. Its one
gesture of its own is the wheel, which stretches the top of the range — a drag
can only ever narrow it.

Nothing here decides anything about the data. A gesture either changes the axes
through the same setters the dialog uses, or emits a signal.
"""
import numpy as np

from PySide6.QtCore import QPoint, QRect


class CanvasInteractionMixin:
    """Press/motion/release, hit tests, and the zoom rubber band."""
    # ── Mouse interaction ─────────────────────────────────────────────────
    def _disp_to_qt(self, x, y):
        """mpl display coords (physical px, origin bottom-left) →
        Qt widget coords (logical px, origin top-left)."""
        dpr = self.devicePixelRatioF()
        return QPoint(round(x / dpr), round(self.height() - y / dpr))

    def _hit_spec_yaxis(self, event):
        """Click landed in the spectrum's y-axis tick-label strip? ax_spec is
        the leftmost column, so everything left of its live bbox within its
        vertical span belongs to its y-axis."""
        bb = self.ax_spec.get_window_extent()
        return event.x < bb.x0 and bb.y0 <= event.y <= bb.y1

    # ── Click-to-edit axis bounds ─────────────────────────────────────────
    # Which axis ends can be typed into, and what each one writes through to.
    # The spectrum's Y is deliberately absent: its tick-label strip is the log
    # toggle (see _hit_spec_yaxis), and one strip cannot mean two things.
    # The AC's delay ends ARE editable and land on the TRACE, which owns that
    # axis — in horizontal mode the trace's own delay ticks are hidden, so the
    # AC's are the only ones there are to click.
    _AXIS_EDIT_PAD = 3      # px of slack around a tick label's ink

    def _editable_axes(self):
        """(axes, which) pairs whose end tick labels open an editor.

        The deviation panel's wavelength ends are in here for the same reason
        the AC's delay ends are: while it is open the spectrum's own ticks are
        hidden, so these are the only wavelength numbers on screen.
        """
        return ((self.ax_spec, "x"), (self.ax_dev, "x"),
                (self.ax_trace, "x"), (self.ax_trace, "y"),
                (self.ax_ac, "x"), (self.ax_cbar, "y"))

    def _end_tick_labels(self, ax, which):
        """The lowest and highest tick labels actually on `ax`'s `which` axis.

        Filtered to ticks inside the view: matplotlib keeps locator-generated
        labels for positions just outside the limits, and those draw nothing
        but would still take the click.
        """
        axis = ax.xaxis if which == "x" else ax.yaxis
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        lo, hi = min(lo, hi), max(lo, hi)
        inside = [t for t in axis.get_ticklabels()
                  if t.get_visible() and t.get_text()
                  and lo <= (t.get_position()[0] if which == "x"
                             else t.get_position()[1]) <= hi]
        return (inside[0], inside[-1]) if len(inside) >= 2 else ()

    def _hit_axis_end(self, event):
        """(ax, which, end, QRect) for the end tick label under the cursor, or
        None. Needs a renderer, so it declines until the first draw."""
        if self._renderer is None:
            return None
        pad = self._AXIS_EDIT_PAD
        for ax, which in self._editable_axes():
            if not ax.get_visible():
                continue
            labels = self._end_tick_labels(ax, which)
            for end, label in zip(("lo", "hi"), labels):
                try:
                    bb = label.get_window_extent(self._renderer)
                except (ValueError, RuntimeError):
                    continue            # nothing drawn for this tick
                if (bb.x0 - pad <= event.x <= bb.x1 + pad
                        and bb.y0 - pad <= event.y <= bb.y1 + pad):
                    tl = self._disp_to_qt(bb.x0, bb.y1)
                    br = self._disp_to_qt(bb.x1, bb.y0)
                    return ax, which, end, QRect(tl, br).normalized()
        return None

    def _open_axis_editor(self, hit):
        ax, which, end, rect = hit
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        self._axis_editor.open_on(ax, which, end,
                                  lo if end == "lo" else hi, rect)

    def apply_axis_bound(self, ax, which, end, value):
        """Write one typed axis bound through the ordinary limit setters.

        Goes through set_xlim/set_trace_xlim/set_trace_ylim rather than touching
        the axes directly, so the limit caches the blit paths read stay in step.
        A typed bound is a MANUAL bound, so the matching auto-scale is switched
        off the same way a rubber-band zoom switches it off — otherwise the next
        frame would snap the view straight back off the number just entered.
        """
        if ax is self.ax_ac and which == "x":
            ax, which = self.ax_trace, "x"      # the trace owns the delay axis
        elif ax is self.ax_dev and which == "x":
            ax, which = self.ax_spec, "x"       # …and the spectrum the wavelength
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        lo, hi = (value, hi) if end == "lo" else (lo, value)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            self.axis_edit_failed.emit(
                f"{value:g} would leave the axis inverted or empty — "
                f"axis unchanged.")
            return
        if ax is self.ax_spec:
            self.autoscale_x = False
            self.set_xlim(lo, hi)
        elif ax is self.ax_cbar:
            self.set_trace_clim(lo, hi)        # switches the colour auto off
        elif which == "x":
            self.autoscale_trace = False
            self.set_trace_xlim(lo, hi)
        else:
            self.autoscale_trace = False
            self.set_trace_ylim(lo, hi)
        self.limits_changed.emit()

    def _cancel_drag(self):
        self._rubber.hide()
        self._drag_ax = None
        self._drag_start = None
        self._fold_drag = False

    def _clamped_point(self, event):
        """Current cursor position clamped to the drag axes' live bbox, so a
        drag never extends into a neighboring axes or outside the figure."""
        bb = self._drag_ax.get_window_extent()
        return (min(max(event.x, bb.x0), bb.x1),
                min(max(event.y, bb.y0), bb.y1))

    def _on_press(self, event):
        axes = (self.ax_spec, self.ax_trace, self.ax_ac, self.ax_dev,
                self.ax_cbar)
        if event.button == 3:
            if self._drag_ax is not None:      # right-click aborts a drag
                self._cancel_drag()
            elif event.inaxes in axes:
                self.reset_axes(event.inaxes)  # …otherwise toggles auto-scale
            return
        if event.button != 1:
            return
        # The fold line gets first refusal on a press in the trace panel, but
        # only within FOLD_GRAB_PX of itself — anywhere else the panel keeps
        # the rubber-band zoom it has always had.
        if event.inaxes is self.ax_trace and self._hit_fold(event):
            if event.dblclick:
                self.fold_reset_requested.emit()
            else:
                self._fold_drag = True
            return
        if event.inaxes in axes:
            self._drag_ax = event.inaxes
            self._drag_start = (event.x, event.y)
        elif event.inaxes is None:
            # Outside every panel: the margins. An end tick label gets first
            # refusal — it is a specific few pixels, while the log toggle is
            # the whole y strip, so testing the strip first would swallow the
            # spectrum's own end labels where the two overlap.
            hit = self._hit_axis_end(event)
            if hit is not None:
                self._open_axis_editor(hit)
            elif self._hit_spec_yaxis(event):
                self.log_toggle_requested.emit()

    def _on_motion(self, event):
        if self._fold_drag:
            self._emit_fold(event)
            return
        if self._drag_ax is None:
            return
        x0, y0 = self._drag_start
        cx, cy = self._clamped_point(event)
        moved = max(abs(cx - x0), abs(cy - y0))
        if self._drag_ax is self.ax_cbar:
            # Only a height means anything on the colour bar, and it is barely
            # wider than a cursor: the band spans the bar's full width, and only
            # vertical travel counts towards the click threshold.
            bb = self.ax_cbar.get_window_extent()
            x0, cx, moved = bb.x0, bb.x1, abs(cy - y0)
        # Show the rectangle only past the click threshold so a plain
        # click never flashes it.
        if moved < 5 * self.devicePixelRatioF():
            self._rubber.hide()
            return
        self._rubber.setGeometry(
            QRect(self._disp_to_qt(x0, y0), self._disp_to_qt(cx, cy)).normalized())
        self._rubber.show()

    def _on_release(self, event):
        if self._fold_drag:
            self._fold_drag = False
            return
        if event.button != 1 or self._drag_ax is None:
            return
        ax = self._drag_ax
        x0, y0 = self._drag_start
        cx, cy = self._clamped_point(event)
        self._cancel_drag()
        thr = 5 * self.devicePixelRatioF()
        # The colour bar needs only height: a straight vertical drag down a
        # 14 px bar is the gesture, not a degenerate one.
        if abs(cy - y0) < thr or (ax is not self.ax_cbar and abs(cx - x0) < thr):
            return                             # plain click / degenerate drag
        inv = ax.transData.inverted()          # includes any log transform
        (dx0, dy0), (dx1, dy1) = inv.transform([(x0, y0), (cx, cy)])
        xlo, xhi = sorted((dx0, dx1))
        ylo, yhi = sorted((dy0, dy1))
        self._apply_zoom(ax, xlo, xhi, ylo, yhi)

    def _apply_zoom(self, ax, xlo, xhi, ylo, yhi):
        if ax is self.ax_cbar:
            # The bar's y axis IS the colour range, in the image's units; its x
            # means nothing. Through the setter, so the auto switches off too.
            self.set_trace_clim(ylo, yhi)
            self.limits_changed.emit()
            return
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        # Freeze the matching autoscale and sync the limit caches, or the
        # next data frame would snap the view straight back.
        if ax is self.ax_spec:
            self.autoscale_x = False
            self.autoscale_y = False
            self._pin_xlim((xlo, xhi))
            self._pin_ylim((ylo, yhi))
        elif ax is self.ax_ac:
            self.autoscale_ac_x = False
            self.autoscale_ac_y = False
            self._ac_xlim_cache = (xlo, xhi)
            self._ac_ylim_cache = (ylo, yhi)
            # The trace owns the delay axis in BOTH modes now, so hand the zoom
            # to it either way: both stay registered, the dialog's trace bounds
            # stay truthful, and the two delay axes cannot come apart.
            self.ax_trace.set_xlim(xlo, xhi)
            self.autoscale_trace = False
            self._trace_xlim_cache = (xlo, xhi)
        elif ax is self.ax_trace:
            # update_trace never touches limits, so nothing to freeze for the
            # rest of this scan — but the NEXT scan's init_trace would snap the
            # view back, and the dialog must show the zoom as manual bounds.
            self.autoscale_trace = False
            self._trace_xlim_cache = (xlo, xhi)
            self._trace_ylim_cache = (ylo, yhi)
            self._sync_ac_x()
        elif ax is self.ax_dev:
            # Shared wavelength axis: hand the x half to its owner, the
            # spectrum, exactly as an AC zoom is handed to the trace. The y half
            # stays here — it is counts of difference, not counts.
            self.autoscale_x = False
            self.ax_spec.set_xlim(xlo, xhi)
            self._pin_xlim((xlo, xhi))
        self.draw_idle()
        self.limits_changed.emit()

    # Per wheel notch over the colour bar: the top of the range moves by this
    # factor of the span. Ten notches double it, near enough, which takes a
    # saturated-looking trace to a readable one in a flick of the wheel.
    CBAR_WHEEL_STEP = 1.07

    def _on_scroll(self, event):
        """Wheel over the colour bar: up brightens (top of the range comes
        down), down darkens. The bottom stays put — it is the floor the trace
        is read against, and moving both ends at once would be a pan."""
        if event.inaxes is not self.ax_cbar or not event.step:
            return
        lo, hi = self.im.get_clim()
        span = (hi - lo) * self.CBAR_WHEEL_STEP ** (-event.step)
        self.set_trace_clim(lo, lo + span)
        self.limits_changed.emit()

    def _hit_fold(self, event):
        """Did this press land on the fold line? Within FOLD_GRAB_PX of it, so
        the line can be caught without pixel-perfect aim — and so every press
        further away still starts a rubber-band zoom as it always did."""
        if not self.line_fold.get_visible():
            return False
        x = float(self.line_fold.get_xdata()[0])
        xd, _ = self.ax_trace.transData.transform((x, 0.0))
        return abs(event.x - xd) <= self.FOLD_GRAB_PX * self.devicePixelRatioF()

    FOLD_GRAB_PX = 6

    def _emit_fold(self, event):
        """Report the cursor's delay, clamped to the trace's own x range so a
        drag off the edge parks the line at the edge instead of somewhere with
        no data behind it."""
        x, _ = self.ax_trace.transData.inverted().transform((event.x, event.y))
        lo, hi = sorted(self.ax_trace.get_xlim())
        self.fold_dragged.emit(float(min(max(x, lo), hi)))
