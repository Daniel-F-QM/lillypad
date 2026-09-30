"""FrogCanvas — the one concrete canvas object the window talks to.

The class is assembled from the mixins beside this file. That is a deliberate
choice over composition: this is genuinely ONE object's state. There is one
Figure, one cached blit background shared by three axes, one palette and one
layout solver, and the panels read each other — the autocorrelation mirrors the
trace's delay axis, the deviation strip shares the spectrum's wavelength axis.
Collaborator objects would each need a back-reference to the canvas, and the
window's ~80 `self.canvas.*` calls would have had to learn which collaborator
to ask. The split is for reading, not for decoupling, and mixins say so.

What stays HERE is what cannot be split:

  * the **eight Signals**. Qt builds a class's metaobject from a QObject
    subclass, so a Signal declared on a plain mixin is never connected — it
    fails silently, at runtime, in one feature.
  * **__init__**, which creates every artist and is the only honest place to see
    what the figure is made of.
  * the **blitting core**. The three live artists (spectrum line, trace image,
    autocorrelation line) are `animated`, so an ordinary draw skips them and the
    background can be cached without them; a frame then restores that background
    and re-rasterizes only those three. A full redraw happens only when
    something STATIC changes.

    Do not simplify any of it. The animated-artist selection, the threshold
    before auto-Y escalates a frame to a full redraw, and the coalescing of
    render requests are measured optimizations — a plausible-looking
    simplification costs about 80 ms per scan column.
"""
# FIRST, above every other matplotlib import: importing this selects the QtAgg
# backend and installs the global rcParams, and the backend has to be chosen
# before the figure classes are imported. Module import is idempotent, so every
# canvas module can open with it and the order stops being a rule to remember.
from lillypad.design import mpl_style  # noqa: F401  (backend side effect)
from lillypad.design.mpl_style import PICTURE_DPI, PICTURE_MAX_PX, _AC_YLIM
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from contextlib import contextmanager

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QRubberBand, QSizePolicy

from lillypad.design.tokens import (DIFF_COLORS, MEMBER_COLORS,
                                    OVERLAP_BAND_ALPHA, OVERLAP_BAND_COLOR,
                                    PALETTE)
from lillypad.gui.canvas.axes_api import AxesApiMixin
from lillypad.gui.canvas.interaction import CanvasInteractionMixin
from lillypad.gui.canvas.layout import FigureLayoutMixin
from lillypad.gui.canvas.panels import (AcPanelMixin, SpectrumPanelMixin,
                                        TracePanelMixin)
from lillypad.gui.perf import perf_tick
from lillypad.gui.widgets import AxisBoundEditor, PlotSplitHandle


# MRO, left to right: the panels first (they are what the window calls most),
# then the axis surface they drive, then layout and interaction, then Qt. Every
# method is defined exactly ONCE across the six — a duplicate would resolve
# silently to whichever class came first, which is the one failure mode a mixin
# split has that a single class does not. tests/test_canvas.py checks it.
class FrogCanvas(AxesApiMixin, SpectrumPanelMixin, TracePanelMixin,
                 AcPanelMixin, FigureLayoutMixin, CanvasInteractionMixin,
                 FigureCanvasQTAgg):
    """Spectrum (top) / FROG trace (middle) / autocorrelation (bottom)."""
    limits_changed = Signal()        # zoom/reset happened → window syncs dialog

    log_toggle_requested = Signal()  # click on spectrum y-axis strip

    proportions_changed = Signal(int)  # split handle dragged → dialog follows

    axes_relaid = Signal()           # axes repositioned → overlay buttons follow

    fold_dragged = Signal(float)     # symmetry fold line moved to this delay

    fold_reset_requested = Signal()  # double-click on it → back to the AC peak

    axis_edit_failed = Signal(str)   # typed axis bound rejected → status bar

    reference_changed = Signal()     # reference frozen, dropped, hidden, shown

    def __init__(self):
        self.fig = Figure(facecolor=PALETTE["plot_bg"])
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Layout state, read by _style()/_apply_mode_decorations() below.
        self._layout_mode = "horizontal"
        self._spec_frac   = 0.50
        # The gridspec only creates the axes and seeds the vertical arrangement;
        # _layout_axes() positions all three explicitly at the end of __init__
        # and on every mode/proportion change.
        gs = self.fig.add_gridspec(2, 2,
                                   height_ratios=[1.8, 1.0],
                                   width_ratios=[1.0, 1.0],
                                   hspace=0.31, wspace=0.14,
                                   left=0.10, right=0.99,
                                   top=0.95, bottom=0.095)
        self.ax_spec  = self.fig.add_subplot(gs[0, 0])
        self.ax_trace = self.fig.add_subplot(gs[0, 1])
        self.ax_ac    = self.fig.add_subplot(gs[1, :])
        # Alignment mode's deviation panel, under the spectrum and sharing its
        # wavelength axis — the same relationship the autocorrelation has to the
        # FROG trace. It exists from the start but is hidden and unpositioned
        # until set_dev_visible(True); _layout_axes gives it room only then, so
        # a session that never runs an alignment sweep has the geometry it
        # always had.
        self.ax_dev = self.fig.add_subplot(gs[1, 0])
        self.ax_dev.set_visible(False)
        self._dev_on = False

        (self.line_spec,) = self.ax_spec.plot([], [], color=PALETTE["accent"], lw=1)
        (self.line_ac,)   = self.ax_ac.plot([], [], color=PALETTE["accent2"], lw=1)
        # AC width readout, boxed inside the panel it describes rather than in
        # the side panel. Top-RIGHT: horizontal mode puts the AC's own title
        # inside the panel's top-left, and the curve's peak sits at the middle.
        self.txt_fwhm = self.ax_ac.text(
            0.985, 0.94, "", transform=self.ax_ac.transAxes,
            ha="right", va="top", fontsize=9, color=PALETTE["accent2"],
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["surface"],
                      edgecolor=PALETTE["border"], alpha=0.85))
        # The level that width is measured AT. The number alone says how wide
        # the pulse is; the line says where the curve was cut to get it, which
        # is what shows a pedestal or a shoulder distorting the figure. Its
        # height is recomputed from the plotted curve on every update rather
        # than pinned at 0.5, so it cannot disagree with scan.fwhm().
        self.line_half = self.ax_ac.axhline(
            0.5, color=PALETTE["accent2"], lw=0.9, ls="--", alpha=0.7, zorder=4)
        self.line_half.set_visible(False)
        # Symmetry mode: the fold line and the asymmetry readout, both on the
        # trace panel and both hidden until the mode is on. The readout mirrors
        # txt_fwhm above — same corner, same box, in the panel it describes.
        self.line_fold = self.ax_trace.axvline(
            0.0, color=PALETTE["accent2"], lw=1.2, ls="--", zorder=6)
        self.line_fold.set_visible(False)
        self.txt_sym = self.ax_trace.text(
            0.985, 0.94, "", transform=self.ax_trace.transAxes,
            ha="right", va="top", fontsize=9, color=PALETTE["accent2"],
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["surface"],
                      edgecolor=PALETTE["border"], alpha=0.85))
        # Multi-spectrometer overlay: one curve per member, shown INSTEAD of
        # line_spec. Separate artists rather than a re-coloured line_spec, so
        # the single-device path stays exactly as it was. line_m1 is SLOT 1
        # and line_m2 slot 2, matching the S1/S2 saturation lamps and
        # integration spinboxes — not the internal blue/red member order.
        (self.line_m1,) = self.ax_spec.plot([], [], color=MEMBER_COLORS[0], lw=1)
        (self.line_m2,) = self.ax_spec.plot([], [], color=MEMBER_COLORS[1], lw=1)
        self.line_m1.set_visible(False)
        self.line_m2.set_visible(False)
        # Overlap band marker. A static patch, NOT one of the blitted artists:
        # it changes only when the band or the view does, and both of those
        # already go through _request_full(), which re-caches the blit
        # backgrounds it has to be baked into.
        self.band_span = self.ax_spec.axvspan(
            0.0, 0.0, facecolor=OVERLAP_BAND_COLOR, alpha=OVERLAP_BAND_ALPHA,
            edgecolor="none", zorder=0)
        self.band_span.set_visible(False)
        self._band = None
        self._overlay = False
        # ── The frozen reference ──────────────────────────────────────────
        # One artist per curve the panel is able to show, since a snapshot taken
        # in the pair view is two curves and one taken in the combined view is
        # a single curve. Which of the two it is depends only on the view it was
        # FROZEN in: a reference stays on screen across a split/unsplit toggle,
        # because grey counts against wavelength say the same thing whichever
        # live curves are drawn over them, and a reference that disappeared
        # every time the view moved would be no reference at all.
        #
        # Grey — the theme's dim text colour, the one thing on the panel that is
        # neither accent nor a member colour — so "then" is separable from "now"
        # at a glance in EITHER view, and re-themed in apply_palette because a
        # grey that carries on one background does not on the other. Slightly
        # transparent on top of that: at full strength the light theme's grey is
        # dark enough to read as the MORE important curve of the two, which is
        # the one thing a reference must never do.
        #
        # Under the live curves, and deliberately NOT animated: a reference
        # moves only when it is frozen, hidden, shown or dropped, and every one
        # of those ends in the _request_full() that bakes it into the blit
        # background. Animating it would cost each live frame three draw_artist
        # calls to put back curves that never changed.
        ref_kw = dict(color=PALETTE["text_dim"], lw=1, alpha=0.75, zorder=1.5)
        (self.ref_spec,) = self.ax_spec.plot([], [], **ref_kw)
        (self.ref_m1,)   = self.ax_spec.plot([], [], **ref_kw)
        (self.ref_m2,)   = self.ax_spec.plot([], [], **ref_kw)
        for ln in (self.ref_spec, self.ref_m1, self.ref_m2):
            ln.set_visible(False)
        self._ref = None          # [(x, y), …]: one entry, or one per member
        self._ref_shown = True    # the eye toggle
        # Alignment mode: the two symmetry-difference curves, on their OWN panel
        # rather than over the spectrum. They used to share ax_spec, which put
        # two quantities that happen to share an axis and a unit on one set of
        # limits: a difference swings either side of zero while a spectrum sits
        # on a pedestal, so whichever had the larger excursion flattened the
        # other. Their own panel gives each its own y range, and leaves the
        # spectrum readable while a sweep is being judged.
        #
        # Still in the SAME count units as the spectrum, hence the zero
        # reference line — a difference is only readable against the level it is
        # a difference from.
        (self.line_d1,) = self.ax_dev.plot([], [], color=DIFF_COLORS[0], lw=1.2)
        (self.line_d2,) = self.ax_dev.plot([], [], color=DIFF_COLORS[1], lw=1.2)
        self.line_d1.set_visible(False)
        self.line_d2.set_visible(False)
        self.diff_zero = self.ax_dev.axhline(
            0.0, color=PALETTE["text_dim"], lw=0.8, ls="--", zorder=0)
        self.diff_zero.set_visible(False)
        # The PREVIOUS sweep, kept as a filled area under the new curves so a
        # tweak can be judged against what it replaced. PolyCollections, not
        # lines: fill_between has no set_data, so each refresh removes these and
        # makes new ones — a sweep is four stage moves, so the cost is nothing.
        self._ghosts = []
        self._ghost_y = ()           # the ghost's y arrays, for the auto-scale
        self._prev_diff = None       # (wl, d1, d2) of the last sweep shown
        # NaN, not zeros: with no scan yet there is no data, and _apply_cmap
        # paints 'bad' pixels in the plot background — so an empty panel reads
        # as empty. Zeros would paint the colormap's bottom colour over the
        # whole panel, which is invisible under magma but a solid blue field
        # under jet. The clim is pinned because an all-NaN array gives the
        # autoscale nothing to work with.
        self.im = self.ax_trace.imshow(np.full((2, 2), np.nan), origin="lower",
                                       aspect="auto", cmap="jet",
                                       extent=[-1, 1, 0, 1], vmin=0.0, vmax=1.0)
        # Resample the DATA to screen resolution and colour the result, rather
        # than colouring every one of the trace's ~10^6 values and resampling
        # the RGBA. Both give the same picture under the linear norm this image
        # uses, but "auto" picks the second and it dominated the cost of every
        # scan column (measured: 44 ms vs 15 ms on a 1024x400 trace).
        self.im.set_interpolation_stage("data")
        # Empty-state ranges. Autoscaling an empty line gives -0.05…0.05, whose
        # delay range contradicts the trace it shares that axis with; matching
        # the image extent makes the two plots agree before the first scan. The
        # y range is the normalized one update_ac keeps, set here too so the
        # first frame of a scan needs no relimit at all.
        self.ax_ac.set_xlim(-1.0, 1.0)
        self.ax_ac.set_ylim(*_AC_YLIM)
        # Title artist per axes, recorded by _place_title. The header buttons
        # that sit AFTER a title have to know how wide that title actually
        # rendered, and set_title's return value is the only handle on it.
        self._title_art = {}
        self._style()

        self.autoscale_y = False
        self.autoscale_x = True
        self.autoscale_ac_x = True
        self.autoscale_ac_y = True
        # Unlike the spectrum's flags (consulted every frame), this one only
        # bites when a new scan calls init_trace: manual trace bounds are meant
        # to outlive the scan they were dialled in on.
        self.autoscale_trace = True
        self._lw = 1

        # ── FROG-trace display settings ───────────────────────────────────
        # The threshold hides weak pixels from the PLOT only: the raw trace is
        # kept here and re-rendered on every settings change, and what the scan
        # worker recorded (and what gets exported / autocorrelated) is never
        # touched.
        self._trace_raw    = None    # last full trace, unmasked
        self._trace_peak   = None    # its max when the caller supplied one
        self._trace_thresh = 0.0     # display floor, fraction of peak (0 = off)
        self._cmap_name    = "jet"
        self._cmap_rev     = False
        self._bg_static    = None    # defined before _apply_cmap touches it
        self._apply_cmap()           # installs the masked-pixel colour

        # ── Mouse interaction: rubber-band zoom + axis-click log toggle ──
        # The rectangle is a Qt child widget composited above the canvas, so
        # the blit background cache never contains it.
        self._drag_ax = None
        self._drag_start = None      # (x, y) in mpl display coords
        self._fold_drag = False      # dragging the symmetry fold line
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        # Click an end tick label to type a bound. Created here so _on_press can
        # reach it from the first click.
        self._axis_editor = AxisBoundEditor(self)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

        # ── Blitting state ────────────────────────────────────────────────
        # These are drawn by hand every frame; exclude them from draw().
        # The member lines must be in here even while hidden, or they would
        # appear on a blit and then vanish on the next full redraw (resize,
        # theme switch) — _on_draw only repaints what this list names.
        # Line2D.draw() returns immediately when invisible, so the two extra
        # entries cost nothing outside overlay mode.
        self.line_spec.set_animated(True)
        self.line_ac.set_animated(True)
        self.line_m1.set_animated(True)
        self.line_m2.set_animated(True)
        # line_d1/line_d2 are NOT animated: they change once per alignment
        # sweep, which is four stage moves apart, and both show_diff and
        # clear_diff end in _request_full() — the redraw that re-caches the
        # background they now live in. Leaving them animated cost every live
        # frame two extra draw_artist calls to put back curves that had not
        # moved since the last stage move.
        self.im.set_animated(True)
        self.txt_fwhm.set_animated(True)
        self.line_half.set_animated(True)
        # Animated even while hidden, like the member lines above: draw()
        # returns immediately on an invisible artist, and leaving them out
        # would make them appear on a blit and vanish on the next full redraw.
        self.line_fold.set_animated(True)
        self.txt_sym.set_animated(True)
        self._animated = [(self.ax_spec, self.line_spec),
                          (self.ax_spec, self.line_m1),
                          (self.ax_spec, self.line_m2),
                          (self.ax_trace, self.im),
                          (self.ax_trace, self.line_fold),
                          (self.ax_trace, self.txt_sym),
                          (self.ax_ac, self.line_ac),
                          (self.ax_ac, self.line_half),
                          (self.ax_ac, self.txt_fwhm)]
        self._bg = None            # cached full-figure background (no animated)
        # Background with the trace image + AC line already composited, so a
        # live spectrum frame only has to draw the spectrum line on top.
        # Invalidated whenever either of those artists (or the figure) changes.
        self._bg_static = None
        # batch(): collapse several update_* calls into one blit/draw.
        self._batch = False
        self._want_blit = False
        self._want_full = False
        # Set only for the duration of save_picture, which prints this very
        # figure at a different size and dpi. Everything that caches something
        # measured in SCREEN pixels — the blit backgrounds, the trace's row
        # pooling — consults it and stands aside.
        self._exporting = False
        self._clim_peak = None     # last clim top actually applied to the image
        # Cache last-applied limits so we only force a full redraw when they
        # actually move (autoscale otherwise re-sets identical limits each frame).
        self._xlim_cache = None
        self._ylim_cache = None
        # Continuous y auto-scale filter state — see _autoscale_y / _pin_ylim.
        self._ylim_smooth = None   # where the view is heading, unquantized
        self._ylim_t      = 0.0    # monotonic stamp of the last filter step
        self._ylim_fast   = False  # latched: contracting onto a real change
        self._dev_xlim_cache = None           # mirrors _xlim_cache when open
        # The last y range _autoscale_dev fitted, so a resize can tell its own
        # view from one the user has since zoomed by hand.
        self._dev_ylim_auto = None
        self._ac_xlim_cache = (-1.0, 1.0)     # the empty-state ranges set above
        self._ac_ylim_cache = _AC_YLIM
        self._trace_xlim_cache = None
        self._trace_ylim_cache = None
        # Re-captures the background after every real draw (init, resize, or any
        # of our draw_idle() calls) and re-renders the animated artists on top.
        self.mpl_connect('draw_event', self._on_draw)
        # Divider between the spectrum and the column beside it. Created before
        # the first _layout_axes(), which is what positions it.
        self._split_band = None      # (x, y_bottom, y_top) in figure fractions
        self._renderer = None        # last draw's renderer, for label metrics
        self._split = PlotSplitHandle(self)
        # Place the axes for the startup layout mode (must come after the blit
        # state above: it invalidates both caches and requests a draw).
        self._layout_axes()

    def _style(self):
        specs = [(self.ax_spec,  "Wavelength (nm)", "Counts",          "Spectrum"),
                 (self.ax_trace, "Delay (fs)",      "Wavelength (nm)", "FROG Trace"),
                 (self.ax_ac,    "Delay (fs)",      "AC (a.u.)",       "Autocorrelation"),
                 (self.ax_dev,   "Wavelength (nm)", "Δ counts",        "Alignment")]
        for ax, xl, yl, title in specs:
            ax.set_facecolor(PALETTE["plot_bg"])
            ax.tick_params(colors=PALETTE["text_dim"], labelsize=8.5)
            for s in ax.spines.values():
                s.set_edgecolor(PALETTE["border"])
            ax.set_xlabel(xl, color=PALETTE["text_dim"], fontsize=9.5, labelpad=2)
            ax.set_ylabel(yl, color=PALETTE["text_dim"], fontsize=9.5)
            self._place_title(ax, title)
        self.ax_spec.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)
        self.ax_ac.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)
        self.ax_dev.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)
        # Last: this re-does the two decorations the loop just reset to their
        # vertical-mode defaults, so a theme switch cannot undo the layout mode.
        self._apply_mode_decorations()

    def _apply_cmap(self):
        """Install the selected colormap, with sub-threshold ('bad') pixels
        painted in the plot background so they read as absent rather than as
        the map's darkest colour. with_extremes returns a copy, so the shared
        matplotlib registry is never mutated."""
        name = self._cmap_name + ("_r" if self._cmap_rev else "")
        self.im.set_cmap(
            matplotlib.colormaps[name].with_extremes(bad=PALETTE["plot_bg"]))
        self._bg_static = None   # composited trace pixels are now stale

    def apply_palette(self, pal):
        """Recolor the figure for a theme switch (light/dark)."""
        self.fig.set_facecolor(pal["plot_bg"])
        self.line_spec.set_color(pal["accent"])
        self.line_ac.set_color(pal["accent2"])
        self.line_half.set_color(pal["accent2"])
        self.txt_fwhm.set_color(pal["accent2"])
        self.txt_fwhm.get_bbox_patch().set_facecolor(pal["surface"])
        self.txt_fwhm.get_bbox_patch().set_edgecolor(pal["border"])
        self.line_fold.set_color(pal["accent2"])
        self.txt_sym.set_color(pal["accent2"])
        self.txt_sym.get_bbox_patch().set_facecolor(pal["surface"])
        self.txt_sym.get_bbox_patch().set_edgecolor(pal["border"])
        # line_m1/line_m2 keep MEMBER_COLORS in both themes — they are chosen
        # to work on either background, and re-theming them would cost the
        # colourblind separation that is the whole point. Same for the two
        # DIFF_COLORS curves; their zero reference is a plain rule, so it does
        # follow the theme. The ghost fills are drawn in DIFF_COLORS too, so
        # they need no re-colouring either.
        self.diff_zero.set_color(pal["text_dim"])
        # The reference curves DO follow the theme, unlike the member curves
        # they can be drawn beneath: their whole job is to read as "not live",
        # which takes a grey that recedes into the plot background it is on —
        # and the greys that do that on white and on near-black are different.
        for ln in (self.ref_spec, self.ref_m1, self.ref_m2):
            ln.set_color(pal["text_dim"])
        self._style()   # re-applies axes/tick/label/grid colors from PALETTE
        self._apply_cmap()   # masked pixels must follow the new background
        self._split.update()  # a Qt child: draw_idle would not repaint it
        self.draw_idle()

    # ── Blitting core ─────────────────────────────────────────────────────
    def _on_draw(self, event):
        """After any real (full) draw, cache the background and paint the
        animated artists on top so they survive resizes and forced redraws."""
        if event is not None and event.canvas is not self:
            return
        # A print draw (save_picture) runs at another dpi through another
        # renderer, so its buffer is neither the size nor the content of the
        # one on screen. Caching it would hand every later blit a background
        # that does not fit the canvas it is restored into; save_picture ends
        # in a draw_idle() that re-takes both caches honestly.
        if self._exporting:
            return
        self._bg = self.copy_from_bbox(self.fig.bbox)
        self._bg_static = None       # figure changed; recomposite lazily
        for ax, art in self._animated:
            ax.draw_artist(art)
        # Only a completed draw knows how much of the gap the tick labels took,
        # and only then can the separator be placed clear of both panels. Moving
        # a Qt child cannot recurse back into the figure draw.
        had_renderer = self._renderer is not None
        self._renderer = getattr(event, "renderer", None) or self._renderer
        self._position_split_handle()
        # The first renderer is also the first chance to measure a title, which
        # is what the buttons packed after one are placed against. Re-run that
        # layout once, here — not on every draw: after this the title's width
        # only changes with the theme or the DPI, and both already relayout.
        if not had_renderer and self._renderer is not None:
            self.axes_relaid.emit()

    def _blit(self):
        """Fast path: repaint the animated artists over the cached background.
        All three must be redrawn, not just the one that changed: the restore
        wipes every animated artist out of the render buffer, and a buffer
        missing some of them shows those plots vanished on the next full
        widget repaint. (Restoring only one axes' patch via restore_region's
        bbox argument is not an option — its sub-region path disagrees with
        display coords on the y origin and lands the rect in the wrong place.)
        """
        if self._bg is None:
            self.draw_idle()          # nothing cached yet — force a full draw
            return
        with perf_tick("blit"):
            self.restore_region(self._bg)
            for ax, art in self._animated:
                ax.draw_artist(art)
            self.blit(self.fig.bbox)

    def _blit_spec(self):
        """Cheaper fast path for live-feed frames, where only the spectrum
        line moves: composite the (static) trace image and AC line into a
        second cached background once, then each frame draw just the spectrum
        line over it. Avoids re-rasterizing the AxesImage — by far the most
        expensive artist — every frame. Uses only full-figure copy/restore,
        never restore_region's sub-region path (see _blit's docstring).
        """
        if self._bg is None:
            self.draw_idle()
            return
        with perf_tick("blit_spec"):
            if self._bg_static is None:
                self.restore_region(self._bg)
                self.ax_trace.draw_artist(self.im)
                # Static between live frames exactly like the image they sit
                # on: symmetry mode only runs on a finished trace, while the
                # feed underneath goes on driving this path.
                self.ax_trace.draw_artist(self.line_fold)
                self.ax_trace.draw_artist(self.txt_sym)
                self.ax_ac.draw_artist(self.line_ac)
                # Static between scan columns, exactly like the AC line — so
                # these belong in the cached background, not the per-frame draw.
                self.ax_ac.draw_artist(self.line_half)
                self.ax_ac.draw_artist(self.txt_fwhm)
                self._bg_static = self.copy_from_bbox(self.fig.bbox)
            self.restore_region(self._bg_static)
            self.ax_spec.draw_artist(self.line_spec)
            self.ax_spec.draw_artist(self.line_m1)
            self.ax_spec.draw_artist(self.line_m2)
            # The alignment differences are NOT drawn here — they live on their
            # own panel now, are not animated, and are already in the cached
            # background this path restored.
            # Only the spectrum panel changed, so that is all Qt has to repaint:
            # restore_region above left the trace/AC regions of the buffer byte
            # for byte identical to what is already on screen. matplotlib's Qt
            # blit() is a synchronous repaint(), so shrinking the rect saves a
            # full-canvas copy_from_bbox + eraseRect + drawImage every frame.
            self.blit(self.ax_spec.get_window_extent())

    # ── Render-request coalescing ─────────────────────────────────────────
    # update_spectrum/update_trace/update_ac end in one of these instead of
    # calling _blit()/draw_idle() directly. Outside a batch the behavior is
    # identical; inside canvas.batch() the requests are merged so a scan tick
    # that updates all three plots costs one blit (or one full draw) total.
    def _request_blit(self, spec_only=False):
        if self._batch:
            self._want_blit = True
        elif spec_only:
            self._blit_spec()
        else:
            self._blit()

    def _request_full(self):
        if self._batch:
            self._want_full = True
        else:
            self.draw_idle()

    @contextmanager
    def batch(self):
        self._batch = True
        self._want_blit = self._want_full = False
        try:
            yield
        finally:
            self._batch = False
            if self._want_full:
                self.draw_idle()   # repaints everything + recaches background
            elif self._want_blit:
                self._blit()

    def _on_first_draw(self, _event):   # retained for API parity; unused
        pass

    # ── Printing the figure ───────────────────────────────────────────────
    def save_picture(self, path, dpi=PICTURE_DPI):
        """Write the plots to `path` as a print-resolution PNG.

        The LIVE figure is printed, not a second one assembled from the same
        data: every curve, limit, colour, colormap and readout the window is
        showing is already on it, and a parallel figure would be a second place
        for all of that to be got right — and to drift out of step the next time
        one of them moves. Three things are therefore borrowed for the duration
        and handed straight back in the finally:

          * the blitted artists' `animated` flag. Figure.draw SKIPS animated
            artists — that is what makes blitting possible — so printed as it
            stands the page would come out with no spectrum, no trace and no
            autocorrelation, which is to say without the three things it is a
            picture of.
          * the layout mode, and the alignment panel if it is open. The page is
            always the horizontal arrangement (full-height spectrum on the
            left, trace over the autocorrelation on the right) whichever mode
            the window happens to be in, and never the alignment sweep — that
            panel is an aid for adjusting the beam, not part of a result.
          * the trace image's row pooling — see _display_decimation.

        The figure keeps its on-screen SIZE in inches and only gains dpi, so the
        page is the window's own layout at more detail rather than a
        re-proportioned one: the margins and gaps are fractions of the figure
        and the fonts are in points, so both land exactly where they look like
        they should. Returns the (width, height) it wrote, in pixels.

        Runs on the GUI thread, unlike the data exports: this is the figure the
        live feed draws into, and handing it to a worker would put two threads
        in matplotlib at once. A full-window page costs a few hundred ms.
        """
        # A big window is already a big page — see PICTURE_MAX_PX.
        w_in, h_in = self.fig.get_size_inches()
        dpi = min(dpi, PICTURE_MAX_PX / max(w_in, h_in))
        prev_mode = self._layout_mode
        prev_dev  = self._dev_on
        prev_anim = [art.get_animated() for _ax, art in self._animated]
        self._exporting = True
        try:
            for _ax, art in self._animated:
                art.set_animated(False)
            if prev_dev:
                self.set_dev_visible(False)
            self.set_layout_mode("horizontal")
            self._render_trace()         # unpooled, at the page's resolution
            # facecolor explicitly rather than by rcParam: the plot background
            # is a theme colour held in PALETTE, and a page saved in dark mode
            # on a white sheet would be unreadable.
            self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        finally:
            self._exporting = False
            for (_ax, art), was in zip(self._animated, prev_anim):
                art.set_animated(was)
            self.set_layout_mode(prev_mode)
            if prev_dev:
                self.set_dev_visible(True)
            self._render_trace()         # …and back to the panel's resolution
            # Both caches were taken against a geometry (and a renderer) the
            # lines above have been moving; the draw re-takes them.
            self._bg = self._bg_static = None
            self.draw_idle()
        # int(), not round(): this has to be the size actually written, and Agg
        # truncates the figure's pixel bbox when it sizes the renderer.
        return int(w_in * dpi), int(h_in * dpi)
