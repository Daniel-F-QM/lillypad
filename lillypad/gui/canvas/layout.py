"""Where the three panels sit in the figure, and how they move.

One layout solver for both arrangements. Everything is expressed in FIGURE
FRACTIONS computed from the current pixel size, not in a fixed grid, because
three of the quantities involved are fixed PIXEL widths — the left margin that
has to fit the widest y tick label, the gap between columns, the title band —
and a fraction that was right at 1600 px is wrong at 900 px. That is also why
resizeEvent re-solves rather than just repositioning.

The split handle is a Qt widget laid over the canvas, so its position is solved
here too: it has to track a figure-fraction boundary in Qt pixel coordinates.
"""
# FIRST, above every other matplotlib import: importing this selects the QtAgg
# backend and installs the global rcParams, and the backend has to be chosen
# before the figure classes are imported. Module import is idempotent, so every
# canvas module can open with it and the order stops being a rule to remember.
from lillypad.design import mpl_style  # noqa: F401  (backend side effect)
from lillypad.design.mpl_style import (_GEO_DEV_RATIO, _GEO_GAP_PX, _GEO_H_BOT,
                                       _GEO_H_COLGAP, _GEO_H_ROWGAP, _GEO_L_MAX,
                                       _GEO_L_PX, _GEO_R, _GEO_ROW_RATIO,
                                       _GEO_TOP_MAX, _GEO_TOP_PX, _GEO_V_BOT,
                                       _GEO_V_COLGAP, _GEO_V_HSPACE,
                                       _TITLE_ABOVE_IN, _TITLE_BAND_IN,
                                       _TITLE_INSET_IN)
from matplotlib.transforms import ScaledTranslation

from PySide6.QtCore import QRect

from lillypad.design.tokens import PALETTE
from lillypad.gui.widgets import PlotSplitHandle


class FigureLayoutMixin:
    """Axes geometry, panel rectangles and the split handle.

    Mixed into FrogCanvas — see figure.py for why these are mixins and not
    collaborators.
    """
    # ── Layout: plot arrangement and proportions ──────────────────────────
    def _place_title(self, ax, text, inside=False):
        """Anchor an axes title an inch-offset from its top-left corner, either
        just above the axes or (horizontal mode's autocorrelation, which has the
        trace's x-axis immediately above it) just inside the plot area.

        Passing y to set_title switches matplotlib's per-draw title
        auto-positioning off, so our own placement survives every redraw, and
        offsetting in inches rather than axes fractions keeps the gap constant
        as the window is resized.
        """
        # set_title returns the artist that actually carries the text, which for
        # loc="left" is ax._left_title and NOT ax.title — style the return value.
        t = ax.set_title(text, color=PALETTE["accent"], fontsize=12,
                         fontweight="bold", loc="left", y=1.0)
        t.set_va("top" if inside else "bottom")
        dx, dy = (_TITLE_INSET_IN[0], -_TITLE_INSET_IN[1]) if inside \
            else (0.0, _TITLE_ABOVE_IN)
        t.set_transform(
            ax.transAxes + ScaledTranslation(dx, dy, self.fig.dpi_scale_trans))
        # Kept so title_right_px can measure it; the artist is the same object
        # on every re-call (set_title writes into ax._left_title), so this just
        # re-records the same handle when the mode decorations re-title a panel.
        self._title_art[ax] = t

    def _apply_mode_decorations(self):
        """Labelling that differs between the two layout modes.

        Horizontal mode stacks the FROG trace over the autocorrelation on one
        shared delay axis, so only the bottom plot carries the delay ticks and
        label. That leaves no room for the AC's title above it (the trace's
        x-axis is right there), so it moves inside the plot area.
        """
        horiz = self._layout_mode == "horizontal"
        self.ax_trace.set_xlabel("" if horiz else "Delay (fs)",
                                 color=PALETTE["text_dim"], fontsize=9.5,
                                 labelpad=2)
        # Inward ticks in horizontal mode: with the two plots flush, outward
        # ticks on the trace's bottom spine would poke down into the AC's panel.
        self.ax_trace.tick_params(axis="x", labelbottom=not horiz,
                                  direction="in" if horiz else "out")
        self._place_title(self.ax_ac, "Autocorrelation", inside=horiz)
        # The deviation panel does to the spectrum exactly what the AC does to
        # the trace above: same wavelength axis, stacked flush, so the bottom
        # panel carries the one set of ticks and the top one gives them up. The
        # `dev` flag stands in for `horiz` here — this pair is stacked in BOTH
        # layout modes, or not present at all.
        dev = self._dev_on
        self.ax_spec.set_xlabel("" if dev else "Wavelength (nm)",
                                color=PALETTE["text_dim"], fontsize=9.5,
                                labelpad=2)
        self.ax_spec.tick_params(axis="x", labelbottom=not dev,
                                 direction="in" if dev else "out")
        self._place_title(self.ax_dev, "Alignment", inside=True)

    def _split_spec_cell(self, x, y, w, h):
        """Place the spectrum in the cell (x, y, w, h) it has been given.

        With alignment mode off it takes the whole cell — which is what every
        layout did before the deviation panel existed. With it on, the cell is
        split into spectrum over deviations, flush, in the same
        tall-row : short-row proportion the trace and AC use.
        """
        if not self._dev_on:
            self.ax_spec.set_position([x, y, w, h])
            return
        h_dev = h / (1.0 + _GEO_DEV_RATIO)
        self.ax_spec.set_position([x, y + h_dev, w, h - h_dev])
        self.ax_dev.set_position([x, y, w, h_dev])

    def _layout_axes(self):
        """Position every axes for the current mode and spectrum width.

        Single source of truth for the plot geometry: the axes are placed
        explicitly instead of being left where the gridspec put them, so the
        proportion slider works in either mode (and before the first draw).

        The spectrum's cell is handed to _split_spec_cell rather than assigned
        directly, so alignment mode's extra panel is one decision made in one
        place instead of a branch in each layout mode.
        """
        L, R, TOP = self._left_margin(), _GEO_R, self._top_margin()
        horiz   = self._layout_mode == "horizontal"
        cgap    = self._col_gap()
        # Clamped, not rejected: the slider must never be a silent no-op.
        frac    = min(max(self._spec_frac, 0.05), 0.95)
        spec_w  = frac * (R - L - cgap)
        right_x = L + spec_w + cgap
        right_w = R - right_x
        if horiz:
            BOT, rgap = _GEO_H_BOT, _GEO_H_ROWGAP
            h_ac = (TOP - BOT - rgap) / (1.0 + _GEO_ROW_RATIO)
            h_tr = TOP - BOT - rgap - h_ac
            self._split_spec_cell(L, BOT, spec_w, TOP - BOT)
            self.ax_trace.set_position([right_x, BOT + h_ac + rgap, right_w, h_tr])
            self.ax_ac.set_position([right_x, BOT, right_w, h_ac])
            # The whole column gap is shared by both sides here.
            split_bot = BOT
        else:
            BOT = _GEO_V_BOT
            # Reproduce the original gridspec spacing exactly: hspace is a
            # fraction of the MEAN row height, so rows + gap = TOP - BOT gives
            # rows = (TOP - BOT) / (1 + hspace/2).
            rows  = (TOP - BOT) / (1.0 + _GEO_V_HSPACE / 2.0)
            rgap  = (TOP - BOT) - rows
            h_ac  = rows / (1.0 + _GEO_ROW_RATIO)
            h_top = rows - h_ac
            self._split_spec_cell(L, BOT + h_ac + rgap, spec_w, h_top)
            self.ax_trace.set_position([right_x, BOT + h_ac + rgap, right_w, h_top])
            self.ax_ac.set_position([L, BOT, R - L, h_ac])
            # Only the top row is split — the autocorrelation spans both columns.
            split_bot = BOT + h_ac + rgap
        # The deviation panel reserves a strip of its y range for the title it
        # carries inside itself, and that strip is a fixed height in inches —
        # so a panel that just changed height needs the fit done again. Only
        # when the view is still the one the fit produced: a hand-drawn zoom is
        # the user's, and a resize is no reason to throw it away.
        if self._dev_on and self._dev_ylim_auto is not None \
                and self.ax_dev.get_ylim() == self._dev_ylim_auto:
            self._autoscale_dev()
        # The gap's centre line, over the rows the two sides actually share.
        self._split_band = (right_x - cgap / 2.0, split_bot, TOP)
        self._position_split_handle()
        # An open bound editor is pinned to a tick label that has just moved,
        # and there is nothing sensible to re-pin it to — the ticks themselves
        # are about to be regenerated. Drop it rather than leave it floating.
        if getattr(self, "_axis_editor", None) is not None:
            self._axis_editor.cancel()
        # Everything moved, so both cached backgrounds describe the old geometry.
        self._bg = None
        self._bg_static = None
        # Anything parented to the canvas and pinned to an axes (the trace's
        # overlay button) has to follow. Emitted last, so it reads the geometry
        # this call just set.
        self.axes_relaid.emit()
        self.draw_idle()

    def _left_margin(self):
        """Left margin as a figure fraction — a fixed pixel width (see
        _GEO_L_PX), so a wide window spends it on plot instead of blank paper."""
        return min(_GEO_L_PX / max(self.width(), 1), _GEO_L_MAX)

    def _top_margin(self):
        """Top of the axes as a figure fraction: the header band reserved for
        each panel's title and overlay buttons, in pixels (see _GEO_TOP_PX) so
        the fixed-size buttons always clear the axes."""
        return 1.0 - min(_GEO_TOP_PX / max(self.height(), 1), _GEO_TOP_MAX)

    def _col_gap(self):
        """Column gap as a figure fraction, floored at the pixels the right
        column's labels and the split handle need side by side."""
        base = (_GEO_H_COLGAP if self._layout_mode == "horizontal"
                else _GEO_V_COLGAP)
        return max(base, _GEO_GAP_PX / max(self.width(), 1))

    def panel_edges_px(self, ax):
        """(left, top, right) of `ax` in canvas widget pixels.

        Read from ax.get_position() — the figure fractions _layout_axes just
        wrote — rather than get_window_extent(), which needs a renderer and so
        is unusable before the first draw. Same y flip as the split handle:
        figure fractions count up from the bottom, Qt from the top.
        """
        pos = ax.get_position()
        w, h = float(self.width()), float(self.height())
        return pos.x0 * w, (1.0 - pos.y1) * h, pos.x1 * w

    def title_right_px(self, ax):
        """Right edge of `ax`'s title in canvas widget pixels — where a header
        button placed AFTER the title has to start.

        Measured, not estimated: the titles differ in length ("Spectrum" vs
        "Autocorrelation") and the font is a theme/DPI property, so the only
        honest answer comes from the rendered artist. Falls back to the panel's
        left edge before the first draw provides a renderer, which packs the
        buttons over the title for exactly one frame — _on_draw re-runs the
        layout as soon as there is something to measure with.
        """
        t = self._title_art.get(ax)
        left = self.panel_edges_px(ax)[0]
        if t is None or self._renderer is None or not t.get_text():
            return left
        try:
            bb = t.get_window_extent(self._renderer)
        except (ValueError, RuntimeError):
            return left
        return max(bb.x1 / self.devicePixelRatioF(), left)

    def _title_band_frac(self, ax):
        """Fraction of `ax`'s height taken by a title drawn INSIDE it.

        In inches like the title offsets themselves, so the band stays the same
        strip of screen however the window is resized, and clamped so a panel
        squeezed very short hands the title a slice rather than its whole y
        range.
        """
        h_in = ax.get_position().height * float(self.fig.get_figheight())
        if h_in <= 0.0:
            return 0.0
        return min(_TITLE_BAND_IN / h_in, 0.4)

    def panel_rect_px(self, ax):
        """`ax` as a QRect in canvas widget pixels — what an overlay widget
        covering that whole panel is resized to. Same fractions, same y flip,
        and the same "works before the first draw" property as panel_edges_px.

        All four EDGES are rounded and the size derived from them, rather than
        the offset and the size rounded separately: round(top) + round(height)
        need not equal round(bottom), so the latter can put the right and
        bottom edges a pixel off the axes they are supposed to trace.
        """
        pos = ax.get_position()
        w, h = float(self.width()), float(self.height())
        left, right = round(pos.x0 * w), round(pos.x1 * w)
        top, bottom = round((1.0 - pos.y1) * h), round((1.0 - pos.y0) * h)
        return QRect(left, top, right - left, bottom - top)

    def _position_split_handle(self, renderer=None):
        """Put the separator over the band the split governs, in the clear strip
        between the two panels. Figure fractions have their origin at the bottom
        left, Qt widget coords at the top left — hence the flip."""
        # getattr, not the attribute: the base class can resize the widget while
        # its own __init__ runs, before the handle exists.
        if getattr(self, "_split_band", None) is None:
            return
        x, bot, top = self._split_band
        w, h = float(self.width()), float(self.height())
        x_px = self._split_x_px(renderer, x * w)
        self._split.setGeometry(
            QRect(round(x_px - PlotSplitHandle.W / 2.0), round((1.0 - top) * h),
                  PlotSplitHandle.W, max(1, round((top - bot) * h))))

    def _split_x_px(self, renderer, fallback):
        """Centre of the clear strip between the spectrum's right spine and the
        right column's y labels, in logical px.

        Measured rather than derived from the gap: the gap also holds the right
        column's y label and tick labels, whose width depends on the font and on
        the data, so the midpoint of what they leave over is the one position
        guaranteed to sit inside neither panel. Falls back to the gap's centre
        until the first draw provides a renderer.

        Between draws the last renderer is reused: it serves only as a source of
        font metrics here, while the positions come from the live axes — so a
        drag re-measures against the geometry it just set instead of snapping
        back to the gap centre until the next redraw lands.
        """
        renderer = renderer or self._renderer
        if renderer is None:
            return fallback
        right_axes = ((self.ax_trace, self.ax_ac)
                      if self._layout_mode == "horizontal" else (self.ax_trace,))
        dpr = self.devicePixelRatioF()
        left = self.ax_spec.get_window_extent().x1 / dpr
        inks = [bb.x0 / dpr for bb in
                (ax.yaxis.get_tightbbox(renderer) for ax in right_axes)
                if bb is not None]
        if not inks:
            return fallback
        # Clamped to the panels themselves, so a label that overhangs its own
        # column (or a tightbbox we could not trust) can never push the handle
        # onto a plot.
        half = PlotSplitHandle.W / 2.0
        spine = self.ax_trace.get_window_extent().x0 / dpr
        return min(max((left + min(inks)) / 2.0, left + half), spine - half)

    def split_travel_px(self):
        """Widget pixels the split can travel over the full 0…1 fraction range —
        the drag handle's px → fraction conversion."""
        return (_GEO_R - self._left_margin() - self._col_gap()) * self.width()

    def spec_fraction(self):
        return self._spec_frac

    @property
    def layout_mode(self):
        """"horizontal" or "vertical". Read-only on purpose — changing it has to
        go through set_layout_mode, which re-solves the geometry and moves the
        panel decorations with it. Assigning the attribute would leave the two
        disagreeing until the next resize."""
        return self._layout_mode

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Not just a repositioning: the left margin and the column gap are fixed
        # pixel widths, so their figure fractions change with the window.
        if getattr(self, "_split", None) is not None:
            self._layout_axes()

    def set_layout_mode(self, mode):
        """"horizontal": spectrum at full height on the left, FROG trace over
        the autocorrelation on a shared delay axis on the right. "vertical":
        spectrum beside the trace with the autocorrelation full-width below."""
        if mode == self._layout_mode:
            return
        self._layout_mode = mode
        self._apply_mode_decorations()
        self._sync_ac_x()        # horizontal: adopt the trace's delay range now
        self._layout_axes()      # ends in draw_idle()

    def _sync_ac_x(self):
        """The FROG trace owns the delay axis; the autocorrelation mirrors it.
        Returns True if the AC view moved.

        In BOTH layout modes, not just the one where the two panels are
        stacked. They plot the same quantity over the same scan either way, so
        a delay that means one thing on the trace and another on the AC below
        it is never what is wanted — it just happened to be invisible while
        both were independently autoscaling to the same scan range. Making it
        explicit is what lets a delay bound be typed into either panel's axis
        and land on both.

        Deliberately does NOT touch autoscale_ac_x: the flag keeps whatever the
        user left it on, and _set_ac_autoscale still consults it.
        """
        xl = self.ax_trace.get_xlim()
        if xl == self._ac_xlim_cache:
            return False
        self.ax_ac.set_xlim(*xl)
        self._ac_xlim_cache = xl
        return True

    def set_proportions(self, spec_frac):
        """Width of the spectrum column as a fraction of the plot area. Applies
        in both layout modes.

        Clamped to the dialog slider's 0.20–0.80 range rather than to the wider
        one _layout_axes tolerates: the drag handle and the slider drive the same
        value, so a drag must never reach a split the slider cannot represent.
        """
        frac = min(max(float(spec_frac), 0.20), 0.80)
        if frac == self._spec_frac:
            return                      # dragging past the clamp: nothing moved
        self._spec_frac = frac
        self._layout_axes()
        self.proportions_changed.emit(round(frac * 100))
