"""
frog_gui_fast.py — performance-optimized variant of frog_gui.py (PySide6)
========================================================================
Functionally identical to frog_gui.py, but with the four live-loop
inefficiencies from the performance report fixed so the two can be run
side-by-side and compared:

  (1) Blitting for the live spectrum — only the changed line is re-rasterized
      each frame instead of a full-figure redraw (draw_idle). See FrogCanvas.
  (2) Acquisition runs on a LiveFeedWorker(QThread), not the GUI thread, so a
      long spectrometer integration no longer freezes the UI.
  (3) The feed is paced to the integration time (never faster than the hardware
      actually produces frames), instead of a fixed 80 ms QTimer.
  (4) The scan build-up (FROG trace + autocorrelation) is blitted too, so each
      column costs at most one small redraw instead of two full-figure redraws.
  (5) Latest-frame-wins display: worker signals only store their payload; a
      fixed-rate GUI timer renders whatever is newest. On machines where a
      render costs more than the acquisition interval, stale frames are
      dropped instead of piling up in the event queue, so the display can no
      longer fall progressively behind real time. Live-feed frames also use a
      cheaper blit that skips re-rasterizing the (static) FROG trace image.

A second round, aimed at the scan (a column used to cost ~106 ms of frozen GUI
on a 2048 x 400 trace; it now costs ~28 ms, and no longer grows as the scan
runs):

  (6) The scan render is O(1) in the column index. The autocorrelation and the
      trace peak are accumulated by _on_column instead of being re-reduced over
      the whole trace on every column — that was O(n_wl * N^2) over a scan.
  (7) The FROG trace image is rasterized at display resolution: rows are
      max-pooled to about one per screen pixel (max, so a narrow spectral line
      cannot vanish), and interpolation_stage="data" colours the resampled
      result instead of colouring ~10^6 values and resampling the RGBA.
      Together: ~93 ms -> ~17 ms per column.
  (8) A live-feed frame repaints only the spectrum panel's rectangle rather
      than the whole canvas, and Auto-Y needs a 2% move before it escalates a
      frame to a full redraw (it used to escalate ~every frame on a noisy
      signal — measured 300 full redraws per 300 frames, now 0).
  (9) Session-lifetime leaks closed: the export worker is no longer parented to
      the window (it retained a whole FrogResult per save), repopulating a menu
      no longer leaves its submenus and action groups behind, and gc.freeze()
      keeps the static startup graph out of every later full collection.

Run with LILLYPAD_PERF=1 for a 10 s stderr report of render times and object
counts (see _PerfProbe).

Everything else (layout, controls, hardware layer, scan engine) is unchanged
and shared with frog_gui.py via hardware.py / scan.py.

    python frog_gui_fast.py
"""

import gc
import sys
import math
import time
import shutil
import numpy as np
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QCheckBox, QPushButton,
    QProgressBar, QScrollArea, QSizePolicy, QStatusBar, QFileDialog,
    QToolBar, QMenu, QRubberBand,
    QMessageBox, QInputDialog, QAbstractSpinBox
)
from PySide6.QtCore import (Qt, QTimer, Signal, QSize, QRect, QPoint,
                            QSignalBlocker)
from PySide6.QtGui import QFont, QIcon, QAction, QActionGroup
# FIRST, above every other matplotlib import: importing this selects the QtAgg
# backend and installs the global rcParams. See lillypad.design.mpl_style.
# `import matplotlib` on its own would NOT be enough — the backend has to be
# chosen before the figure classes below are imported.
import matplotlib
from lillypad.design.mpl_style import (PICTURE_DPI, PICTURE_MAX_PX,
                                       TRACE_COLORMAPS, _AC_YLIM,
                                       _GEO_DEV_RATIO, _GEO_GAP_PX, _GEO_H_BOT,
                                       _GEO_H_COLGAP, _GEO_H_ROWGAP, _GEO_L_MAX,
                                       _GEO_L_PX, _GEO_R, _GEO_ROW_RATIO,
                                       _GEO_TOP_MAX, _GEO_TOP_PX, _GEO_V_BOT,
                                       _GEO_V_COLGAP, _GEO_V_HSPACE,
                                       _TITLE_ABOVE_IN, _TITLE_BAND_IN,
                                       _TITLE_INSET_IN)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation

from lillypad.paths import CALIBRATION_DIR, seed_calibration_dir
from lillypad.settings import (SETTINGS_VERSION, as_bool, load_settings,
                               save_settings)
# The design layer. No colour, font size or icon path is defined in this file any
# more — see lillypad/design/. PALETTE is mutated in place by install_theme, so
# importing the name here follows a runtime theme switch.
from lillypad.design.geometry import (HDR_BTN, HDR_GAP, HDR_GROUP_GAP, HDR_ICON,
                                      HDR_PAD, HDR_TITLE_GAP)
from lillypad.design.icons import (ALIGN_ICON, HIDE_ICON,
                                   HORIZ_ICON,
                                   LOCKED_ICON, MERGE_ICON, MOON_ICON,
                                   PICTURE_ICON, REFRESH_ICON, RESCALE_ICON,
                                   SHOW_ICON, SPLIT_ICON, STITCH_ICON, SUN_ICON,
                                   SYMMETRY_ICON, UNLOCKED_ICON, VERT_ICON,
                                   app_icon)
from lillypad.design.icons_runtime import glyph_icon as _glyph_icon
from lillypad.design.theme import install_theme, startup_theme
from lillypad.gui.dialogs.acquisition import AcquisitionSettingsDialog
from lillypad.gui.dialogs.alignment import AlignmentDialog
from lillypad.gui.dialogs.avantes import AvantesSettingsDialog
from lillypad.gui.dialogs.base import DevicePickerDialog
from lillypad.gui.dialogs.export import EXPORT_FORMATS, ExportDialog
from lillypad.gui.dialogs.graphics import GraphicsSettingsDialog
from lillypad.gui.dialogs.simulation import SimulationDialog
from lillypad.gui.dialogs.spectrometer import SpectrometerDialog
from lillypad.gui.dialogs.stage import StageDialog
from lillypad.gui.perf import PERF, install_probe, perf_tick
from lillypad.gui.state import InstrumentState
from lillypad.gui.widgets import (SAT_WARN_FRACTION, AxisBoundEditor,
                                  DoubleSpinBox, NoDeviceOverlay, PlotSplitHandle,
                                  Slider, SpinBox, StatusLamp, _hline)
from lillypad.gui.text import FEED_BUSY_MSG, RIGHT_CLICK_HINT
from lillypad.gui.workers import (MAX_UI_EXPOSURE_MS,
                                  _ORPHANED_THREADS, AlignmentWorker,
                                  ExportWorker, LiveFeedWorker, _exposure_bounds,
                                  _exposure_ms)
from lillypad.design.tokens import (DIFF_COLORS, MEMBER_COLORS,
                                    OVERLAP_BAND_ALPHA, OVERLAP_BAND_COLOR,
                                    PALETTE)
from hardware import (SimulatedStage, SimulatedSpectrometer,
                      KinesisStage, ZaberStage, PiezoJenaStage,
                      SeabreezeSpectrometer, AvantesSpectrometer,
                      StitchedSpectrometer,
                      list_kinesis_stages, list_spectrometers,
                      list_avantes_spectrometers, list_seabreeze_spectrometers,
                      open_spectrometer,
                      load_calibration_file,
                      PULSE_SHAPES,
                      SIM_SPECTROMETERS, sim_spectrometer_pair)
from scan import (FrogScanWorker, fwhm, autocorrelation,
                  position_to_delay_fs, delay_to_position_um,
                  _um_to_stage, _stage_to_um)


# ─────────────────────────────────────────────────────────────────────────────

# Simulated members offered alongside real devices in the multi-spectrometer
# slots, so the mode can be exercised without two spectrometers on the bench.
# These are sentinels, not device ids — _open_slot_device matches them BEFORE
# it hands anything to hardware.open_spectrometer, so they deliberately do not
# carry a "vendor:serial" tag. Each covers one overlapping half of the
# simulated signal band; see _make_sim_member for why halves and not two
# full-band copies.
SIM_SLOT_DEVICES = (("__sim_blue__", "Simulated — blue half"),
                    ("__sim_red__",  "Simulated — red half"))


# ─────────────────────────────────────────────────────────────────────────────


# resource_path(), app_dir() and the settings reader/writer now live in
# lillypad.paths and lillypad.settings — see the top import block. Read
# lillypad.paths' docstring before using either path helper: the difference
# between a bundled, read-only resource and a writable folder beside the program
# is the thing they exist to keep straight.


def _on_a_screen(rect):
    """True if `rect` overlaps some connected screen enough to be grabbed.

    A saved position only means anything on the monitor layout it was saved on:
    unplug the second display and a restored window sits in dead space with no
    title bar to drag it back by.
    """
    for s in QApplication.screens():
        i = s.availableGeometry().intersected(rect)
        if i.width() >= 120 and i.height() >= 40:
            return True
    return False


def _ease(current, target, dt, tau):
    """One-pole step from `current` toward `target` over `dt` seconds, with time
    constant `tau`.

    Solved rather than iterated, so the result depends only on elapsed TIME and
    not on how many times it was called: at a 1 ms exposure this runs ~16 times
    a second and at a 10 s exposure once every ten, and a per-call fraction
    would have made the response time a function of the exposure.
    """
    return target + (current - target) * math.exp(-dt / tau)




# ─────────────────────────────────────────────────────────────────────────────
# Performance probe (opt-in: LILLYPAD_PERF=1)
# ─────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────
# Live-feed acquisition worker (FIX 2 + 3)
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Exposure-box plumbing
# ─────────────────────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────────────────────
# Saturation indicator
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Export worker — writing a full trace is far too slow for the GUI thread
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Alignment worker — the four-point symmetry check
# ─────────────────────────────────────────────────────────────────────────────


















# ─────────────────────────────────────────────────────────────────────────────
# "Nothing connected" plot overlay
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Right-hand canvas: spectrum (top) / FROG trace (middle) / autocorrelation
#
# FIX 1 + 4 — blitting. The three live artists (spectrum line, FROG-trace image,
# autocorrelation line) are marked `animated`, so a normal draw() skips them and
# a cached background can be captured without them. Per-frame updates then just
# restore that background and re-rasterize the three animated artists instead of
# redrawing every tick/spine/grid/label. A full draw_idle() (which re-captures
# the background via the draw_event handler) is only issued when something static
# actually changes — axis limits, theme, log scale, line width, proportions.
# ─────────────────────────────────────────────────────────────────────────────

# The rest of the panel header band (HDR_BTN and friends) is defined up with
# the other Qt chrome constants — _glyph_icon needs HDR_ICON at import time.






class FrogCanvas(FigureCanvasQTAgg):
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
                (self.ax_ac, "x"))

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
        axes = (self.ax_spec, self.ax_trace, self.ax_ac, self.ax_dev)
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
        # Show the rectangle only past the click threshold so a plain
        # click never flashes it.
        if max(abs(cx - x0), abs(cy - y0)) < 5 * self.devicePixelRatioF():
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
        if abs(cx - x0) < thr or abs(cy - y0) < thr:
            return                             # plain click / degenerate drag
        inv = ax.transData.inverted()          # includes any log transform
        (dx0, dy0), (dx1, dy1) = inv.transform([(x0, y0), (cx, cy)])
        xlo, xhi = sorted((dx0, dx1))
        ylo, yhi = sorted((dy0, dy1))
        self._apply_zoom(ax, xlo, xhi, ylo, yhi)

    def _apply_zoom(self, ax, xlo, xhi, ylo, yhi):
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

    def set_cmap(self, name):
        self._cmap_name = name
        self._apply_cmap()
        self.draw_idle()

    def set_trace_reversed(self, on):
        self._cmap_rev = bool(on)
        self._apply_cmap()
        self.draw_idle()

    def set_trace_threshold(self, percent):
        """Hide trace pixels below `percent` of the trace peak. Display only —
        the stored trace keeps every count."""
        self._trace_thresh = max(0.0, min(100.0, float(percent))) / 100.0
        self._render_trace()          # takes effect without waiting for a column

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
        self._sync_ac_x()      # horizontal: the AC follows the new delay range
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
        data = (np.ma.masked_less(shown, self._trace_thresh * peak)
                if self._trace_thresh > 0 else shown)
        self.im.set_data(data)
        # Once a row is a screen pixel the antialiasing filter has nothing left
        # to average, and the max-reduction above already did the honest part.
        self.im.set_interpolation("nearest" if f > 1 else "auto")
        # set_clim invalidates the image's cached RGBA (full re-normalize +
        # re-colormap), so only touch it when the peak moved visibly (>0.5%).
        if self._clim_peak is None or abs(peak - self._clim_peak) > 0.005 * self._clim_peak:
            self.im.set_clim(0, peak)
            self._clim_peak = peak
        self._bg_static = None        # trace pixels changed
        # Extent/limits are fixed by init_trace, so this is always a fast blit.
        self._request_blit()

    def update_ac(self, delays, ac):
        # Normalized to its own peak for DISPLAY only: the panel always reads
        # 0…1, so its tick labels stay short whatever the count level (they used
        # to grow a digit per decade and steal width from the plot beside them in
        # horizontal mode) and a scan's partial curve fills the panel from the
        # first column. Nothing reads this line back — the stored trace, the
        # exports and the autocorrelation the scan computed keep raw counts.
        peak = float(ac.max()) if ac.size else 0.0
        shown = ac / peak if peak > 0 else ac
        self.line_ac.set_data(delays, shown)
        self._set_half_line(shown)
        self._bg_static = None        # AC line changed
        # The trace dictates the delay axis in either layout mode, so the AC's
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


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class FrogWindow(QMainWindow):
    # Raised by the stage layer (StageBase.warn_cb) when it has to do something
    # the operator should know about — currently only a long move being split.
    # A signal, not a direct status-bar call: the scan worker drives the same
    # stage from its own thread, and Qt queues cross-thread emissions for us.
    stage_warning = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lillypad")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1180, 760)
        # load_settings() is cached, so this is the same dict main() already
        # read the theme from on its way to styling the QApplication — the two
        # cannot disagree. The rest is applied at the END of __init__, once the
        # widgets it drives exist.
        self._settings = load_settings()
        self._settings_saved = {}
        self._theme = startup_theme(self._settings)

        # Everything about the instrument itself — the two devices, the scan
        # configuration, the recorded dark, the spectrometer slots, what the
        # simulator stands in for. See lillypad/gui/state.py for why these are
        # gathered and the display scratch below is not. The @property
        # forwarders further down keep `self.spec`, `self.background` and the
        # rest reading exactly as they always did.
        #
        # NOTHING is connected at startup: state.spec and state.stage stay None
        # until the operator connects a device (or asks for a simulator by name,
        # in the Simulation window), and _sync_connection_ui puts a "no
        # spectrometer connected" panel up in the meantime.
        self.state = InstrumentState()
        self._stage_units_fs = True     # jog/move fields default to fs
        self._dark_member_warned = False
        self._overlay_on   = False   # spectrum panel showing the members apart
        self._pair_seen    = False   # a live pair has already had its default
        self.last_spectrum = None
        self.result        = None
        self._worker       = None
        self._export_worker = None
        self._scan_trace   = None
        self._scan_delays  = None
        self._scan_ac      = None    # running per-column sums (see _on_column)
        self._scan_peak    = 0.0     # running trace max, for the image clim
        # Cached at scan start so _on_column can redraw the spectrum panel
        # without touching self.spec from the GUI thread while the worker is
        # inside acquire().
        self._scan_wl      = None
        # The pedestal the running scan's columns are drawn against, sampled
        # once when it starts (see _start_scan). None = draw them as measured.
        self._scan_dark    = None
        self._scan_member_dark = None
        # The finished scan as the panels show it — pedestal removed — and its
        # autocorrelation. Every view of a finished trace (the trace itself,
        # the symmetry fold, the width readout) is built from these two rather
        # than from result.trace, so none of them can disagree with the others
        # or with the exported file. Both are None until a scan finishes;
        # result.trace itself is never modified.
        self._result_view  = None
        self._result_ac    = None
        # ── Alignment mode ────────────────────────────────────────────────
        # Diagnostic views only: none of this reaches the FrogResult or any
        # export. _align_worker runs the four-point symmetry measurement;
        # _align_trace holds the RAW (uncalibrated, un-darked) counterpart of
        # _scan_trace for a stitched pair — the two members interpolated onto
        # the common grid and hard-cut at _align_cut, with _align_peak the
        # running max of each half so the halves can be normalized at render
        # time without any stored column going stale.
        self._align_worker   = None
        self._align_trace    = None
        self._align_wl       = None   # the two members' native grids
        self._align_cut      = 0
        self._align_peak     = [0.0, 0.0]
        self._align_view     = None   # scratch buffer for the normalized view
        self._align_trace_on = False
        # Symmetry mode: a display-only view of the finished trace folded about
        # its autocorrelation peak. Mutually exclusive with the raw view above —
        # both rewrite the same image.
        self._symmetry_on    = False
        self._fold_fs        = None   # hand-placed fold; None = the AC peak
        self._feed_was_on  = True
        self._pending_fit  = True
        # Weaker cousin of _pending_fit: refit the spectrum to the next frame
        # but leave the auto-scale flags exactly as the operator set them.
        self._pending_rescale = False
        self._export_fmt   = "dwc"

        self.dlg_settings = AcquisitionSettingsDialog(self)
        self.dlg_align    = AlignmentDialog(self)
        # Built here with the other pop-ups, but note it reaches back into the
        # window on construction (the format radios call _set_export_fmt): it
        # must come after _export_fmt above and before _build_ui, which is what
        # puts the toolbar button that opens it on screen.
        self.dlg_export   = ExportDialog(self)
        self.dlg_spec     = SpectrometerDialog(self, self)
        self.dlg_stage    = StageDialog(self, self)
        self.dlg_sim      = SimulationDialog(self, self)
        # Vendor-specific controls, opened from a toolbar button that stays
        # hidden until an Avantes is actually connected.
        self.dlg_avantes  = AvantesSettingsDialog(self, self)
        # The Avantes window announces a full-scale change; the Spectrometer
        # window shows full scale as its override placeholder. Neither knows
        # about the other — this is the one place that knows both exist.
        self.dlg_avantes.full_scale_changed.connect(
            lambda: self.dlg_spec.isVisible() and self.dlg_spec.refresh())
        # Two WIDGETS that moved into a sub-window but are still driven from
        # here, aliased because the widget itself is what is needed —
        # _sync_backlash_ui blocks its signals and writes a value into it, and
        # _set_stage_controls_enabled puts it in a tuple with six others. Every
        # other cross-window read is a named value property on the dialog
        # (dlg_settings.n_average, dlg_align.step_fs, dlg_graphics.log_scale …),
        # which is what a caller actually wants and what keeps this window from
        # knowing which spinbox holds what. Both dialogs are built above
        # _build_ui(), which is what needs them.
        self.spin_backlash   = self.dlg_stage.spin_backlash
        self.lbl_backlash_fs = self.dlg_stage.lbl_backlash_fs
        # The threshold is applied live, not just at scan start, so the lamp
        # reflects the setting you are in the middle of tuning.
        self.dlg_settings.spin_sat.valueChanged.connect(self._on_sat_fraction)

        self._build_ui()

        # No parking move and no integration push here any more: there is no
        # device to make them to. Both happen on connect instead —
        # _apply_spectrometer clamps the exposure into the new device's range,
        # and _apply_stage parks or adopts the position.
        self.stage_warning.connect(self._on_stage_warning)
        self._sync_connection_ui()

        # FIX 2/3 — acquisition runs on a worker thread, paced to integration.
        self._feed = LiveFeedWorker(lambda: self.spec)
        self._feed.spectrum_ready.connect(self._on_spectrum)
        self._feed.acquire_failed.connect(self._on_feed_error)
        self._feed.start()
        if self.btn_feed.isChecked():
            self._feed.resume()

        # FIX 5 — latest-frame-wins display. The worker signals above only
        # *store* their payload (O(1)); all rendering happens here, at a fixed
        # cadence, on whatever data is newest. On a machine where a render
        # costs more than the acquisition interval, frames are overwritten
        # (dropped) instead of piling up in the Qt event queue — the display
        # can no longer fall progressively behind real time.
        self._live_frame  = None     # newest (wl, raw) from the live feed
        self._sat_frame   = None     # highest-peak raw frame since last tick
        self._sat_peak    = -1.0     # ...so dropped frames can't hide clipping
        self._scan_dirty  = False    # a scan column arrived since last tick
        self._scan_last_i = -1
        self._scan_col    = None
        self._scan_pos_um = 0.0
        self._scan_members_raw = None  # newest column's raw member frames
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(60)      # ~16 fps display cadence
        self._display_timer.timeout.connect(self._display_tick)
        self._display_timer.start()

        # Opt-in instrumentation (LILLYPAD_PERF=1). Reports to stderr every 10 s.
        self._perf_timer = None
        if PERF:
            probe = install_probe(self)
            self._perf_timer = QTimer(self)
            self._perf_timer.setInterval(10_000)
            self._perf_timer.timeout.connect(probe.report)
            self._perf_timer.start()

        # Last, because it drives widgets from every group above as well as the
        # canvas _build_ui created. Nothing has painted yet — the event loop
        # only starts in main() — so the draw_idle()s this sets off all collapse
        # into the first frame.
        self._restore_settings()
        # Closing is not the only way this program ends: a vendor driver can
        # take the process with it (see the warning in closeEvent), and losing a
        # session of preferences to a fault in someone else's DLL is exactly the
        # annoyance persistence is here to remove. Ten seconds is the most that
        # can be lost, and the write only happens when something changed.
        self._settings_timer = QTimer(self)
        self._settings_timer.setInterval(10_000)
        self._settings_timer.timeout.connect(self._autosave_settings)
        self._settings_timer.start()

    # ── Instrument state ─────────────────────────────────────────────────────
    # One forwarder per InstrumentState field. Read lillypad/gui/state.py for
    # what is in there and why; these exist so that the window's own ~200 uses
    # of `self.spec`, `self.background`, `self.scan_cfg` and the rest read as
    # they always did while a dialog or a controller can be handed the state
    # object alone and be unable to reach the widgets.
    #
    # Both directions on purpose. Several of these are assigned in ordinary
    # control flow (`self.spec = new_spec`, `self.background = None`), and a
    # read-only property would turn each of those into a silent shadowing bug
    # the first time one was missed.
    @property
    def spec(self):
        return self.state.spec

    @spec.setter
    def spec(self, value):
        self.state.spec = value

    @property
    def stage(self):
        return self.state.stage

    @stage.setter
    def stage(self, value):
        self.state.stage = value

    @property
    def scan_cfg(self):
        return self.state.scan_cfg

    @scan_cfg.setter
    def scan_cfg(self, value):
        self.state.scan_cfg = value

    @property
    def background(self):
        return self.state.background

    @background.setter
    def background(self, value):
        self.state.background = value

    @property
    def background_members(self):
        return self.state.background_members

    @background_members.setter
    def background_members(self, value):
        self.state.background_members = value

    @property
    def background_exposures(self):
        return self.state.background_exposures

    @background_exposures.setter
    def background_exposures(self, value):
        self.state.background_exposures = value

    @property
    def _multi(self):
        return self.state.multi

    @_multi.setter
    def _multi(self, value):
        self.state.multi = value

    @property
    def _multi_members(self):
        return self.state.multi_members

    @_multi_members.setter
    def _multi_members(self, value):
        self.state.multi_members = value

    @property
    def _stitch_stale(self):
        return self.state.stitch_stale

    @_stitch_stale.setter
    def _stitch_stale(self, value):
        self.state.stitch_stale = value

    @property
    def sim_pulse(self):
        return self.state.sim_pulse

    @sim_pulse.setter
    def sim_pulse(self, value):
        self.state.sim_pulse = value

    @property
    def sim_gate(self):
        return self.state.sim_gate

    @sim_gate.setter
    def sim_gate(self, value):
        self.state.sim_gate = value

    @property
    def seabreeze_backend(self):
        return self.state.seabreeze_backend

    @seabreeze_backend.setter
    def seabreeze_backend(self, value):
        self.state.seabreeze_backend = value

    # ── Connection state ─────────────────────────────────────────────────────
    # spec and stage are None when nothing is connected — a real absence rather
    # than a stand-in device, so nothing can quietly return plausible-looking
    # counts or positions for hardware that is not there. Everything that
    # reaches a device goes through one of these three.
    def _have_spec(self):
        return self.state.have_spec()

    def _have_stage(self):
        return self.state.have_stage()

    def _pair_live(self):
        """True while two spectrometers are connected as one stitched device.
        This is the ONLY definition of multi-spectrometer mode."""
        return self.state.pair_live()

    # ── Shared-device access ─────────────────────────────────────────────────
    def _scan_running(self):
        return self._worker is not None and self._worker.isRunning()

    @contextmanager
    def _device_lock(self):
        """Take the stage/spectrometer for a device call made ON the GUI thread.

        The live feed drives the same two devices from its own thread, so any
        direct `self.stage.*` / `self.spec.*` call from a slot has to park the
        feed first — otherwise two threads are inside one vendor driver at once.
        (Harmless for the simulators; a real Kinesis/Zaber/seabreeze can
        interleave transactions and return garbage.)

        Yields True when the feed actually parked. On False the caller MUST NOT
        touch the device. The feed is restored on the way out either way, unless
        a scan now owns the hardware.
        """
        ok = self._feed.pause()
        try:
            yield ok
        finally:
            if self.btn_feed.isChecked() and not self._scan_running():
                self._feed.resume()

    # ── Hardware ─────────────────────────────────────────────────────────────
    def _make_sim_spectrometer(self):
        """A sim spectrometer that derives its delay from the LIVE stage (so it
        works with a real stage too — the Stage.py test setup)."""
        return SimulatedSpectrometer(
            self.stage, gate=self.sim_gate, pulse=self.sim_pulse,
            position_to_delay=lambda pos_mm: position_to_delay_fs(
                _stage_to_um(pos_mm), self.scan_cfg.zero_pos_um, self.scan_cfg.pass_factor))

    def _make_sim_device(self, half):
        """One member of the simulated pair, modelled on a REAL spectrometer.

        Used for a beam that names a pair in PULSE_SHAPES (see "bird", whose
        SHG covers 500–1400 nm and needs the Si + InGaAs bench to be measured
        at all). Unlike the two halves below, these devices have their own
        pixel grids, detector responses, darks and noise, so the two curves
        differ for the reasons two real spectrometers differ — and the fit that
        matches them has something real to absorb.

        The matching calibration file is loaded when it is on disk: it is the
        reciprocal of the device's simulated response, exactly as the shipped
        files are for the hardware they were measured on. Without it the pair
        still connects, and Auto-stitch then reports the large residual an
        uncalibrated Si/InGaAs overlap really does have.
        """
        model = SIM_SPECTROMETERS[
            sim_spectrometer_pair(self.sim_pulse, self.sim_gate)[half]]
        sim = SimulatedSpectrometer(
            self.stage, gate=self.sim_gate, pulse=self.sim_pulse,
            position_to_delay=lambda pos_mm: position_to_delay_fs(
                _stage_to_um(pos_mm), self.scan_cfg.zero_pos_um,
                self.scan_cfg.pass_factor),
            **model["device"])
        sim.name = model["name"]
        cal = CALIBRATION_DIR / f"{model['calibration']}.txt"
        if cal.is_file():
            try:
                sim.set_calibration(cal)
            except Exception:
                pass        # raw counts is a fine fallback; nothing is wrong yet
        return sim

    def _make_sim_member(self, half):
        """A simulated spectrometer covering one half of the simulated signal
        band, for use as a stitched-pair member.

        Halves, not two full-band copies: identical grids would make the whole
        spectrum "overlap", so the stitch geometry — a blue-only region, a
        shared middle, a red-only region — would never be exercised. The 30%
        shared middle is what Auto-stitch fits over. The band itself is only
        known to a constructed simulator (it is sized from the signal), hence
        the throwaway probe; construction touches no hardware and costs a
        handful of FFTs.

        A beam that names its own pair of instruments gets those instead.
        """
        if sim_spectrometer_pair(self.sim_pulse, self.sim_gate) is not None:
            return self._make_sim_device(half)
        probe = self._make_sim_spectrometer()
        wl = np.asarray(probe.wavelengths, float)
        lo, hi = float(wl[0]), float(wl[-1])
        span = hi - lo
        start, end = ((lo, lo + 0.65 * span) if half == 0
                      else (lo + 0.35 * span, hi))
        sim = SimulatedSpectrometer(
            self.stage, gate=self.sim_gate, pulse=self.sim_pulse,
            wl_start=start, wl_end=end, n_pixels=512,
            # DIFFERENT pedestals, and deliberately so: the halves used to
            # default to background_counts=0, which is why every dark bug in
            # the stitched path could only be found on the bench. Two unequal,
            # exposure-scaled pedestals (see SimulatedSpectrometer._raw_column)
            # are what a real pair looks like, and they make a mis-applied dark
            # show up as a step at the seam.
            background_counts=300.0 if half == 0 else 1200.0,
            position_to_delay=lambda pos_mm: position_to_delay_fs(
                _stage_to_um(pos_mm), self.scan_cfg.zero_pos_um,
                self.scan_cfg.pass_factor))
        # Distinct names: the two members share a class, so the saturation-lamp
        # tooltips and the Auto-stitch message would otherwise name both
        # identically and give no way to tell which is which.
        sim.name = (f"simulated {'blue' if half == 0 else 'red'} half "
                    f"[{start:.0f}–{end:.0f} nm]")
        return sim

    def _open_slot_device(self, ident):
        """Open whatever a multi-spectrometer slot points at — a simulated
        half, or a real device by its tagged id. The sentinels are matched
        first: they carry no vendor tag, so open_spectrometer would reject
        them."""
        for half, (sim_ident, _label) in enumerate(SIM_SLOT_DEVICES):
            if ident == sim_ident:
                return self._make_sim_member(half)
        return open_spectrometer(ident, self.seabreeze_backend)

    def _attach_stage_warnings(self, stage):
        """Route a stage's warnings into the status bar. Safe from any thread —
        emit() is what crosses back to the GUI one."""
        stage.warn_cb = self.stage_warning.emit

    def _on_stage_warning(self, message):
        self.status.showMessage(f"Stage: {message}", 5000)

    def _apply_stage(self, new_stage):
        """Swap in a new stage. Returns (ok, error) — on failure NOTHING has
        changed and `new_stage` is still the caller's to dispose of.

        `new_stage` may be None: that is the disconnect path, and it takes the
        same lock and the same teardown as any other swap.

        Refused outright while a scan is running: FrogScanWorker captured the
        old stage at construction, so disconnecting it here would pull the
        device out from under the thread currently driving it.
        """
        if self._scan_running():
            return False, "A scan is running — stop it before changing hardware."
        # Pause the feed (and wait for any in-flight acquire) before swapping,
        # so the worker never reads a half-swapped device.
        with self._device_lock() as ok:
            if not ok:
                return False, FEED_BUSY_MSG
            old = getattr(self, "stage", None)
            if old is not None and old is not new_stage:
                try:
                    old.disconnect()
                except Exception:
                    pass
            self.stage = new_stage
            if new_stage is not None:
                self._attach_stage_warnings(new_stage)
            # A simulated spectrometer reads the live stage, so re-point it.
            # None is allowed: SimulatedSpectrometer.acquire falls back to zero
            # delay, so the simulator keeps working with no stage attached.
            if isinstance(self.spec, SimulatedSpectrometer):
                self.spec.stage = self.stage
            if isinstance(new_stage, SimulatedStage):
                new_stage.move_to(_um_to_stage(self.scan_cfg.zero_pos_um))
            elif new_stage is not None:
                # Real stage: adopt its ACTUAL position as zero-delay so
                # "Move to 0 fs" can never slam it into a travel limit.
                # Except when the axis has no reference — that readback is an
                # arbitrary number, and homing later would move the frame under
                # it. Adopt it anyway so the travel maths has something sane to
                # work with, but _start_scan refuses to run until it is homed.
                self.scan_cfg.zero_pos_um = _stage_to_um(new_stage.get_position())
            self._update_stage_unit_ranges()  # new travel + possibly new zero
            self._refresh_positions()   # inside the lock — we still own the stage
        self._sync_backlash_ui()        # the new stage brings its own default
        self._refresh_scan_um()         # zero may have moved: fs→um previews
        self._sync_connection_ui()      # panel swaps between controls/Connect
        return True, ""

    def _apply_spectrometer(self, new_spec):
        """Swap in a new spectrometer, or None to disconnect. (ok, error);
        see _apply_stage."""
        if self._scan_running():
            return False, "A scan is running — stop it before changing hardware."
        with self._device_lock() as ok:
            if not ok:
                return False, FEED_BUSY_MSG
            old = getattr(self, "spec", None)
            if old is not None and old is not new_spec:
                try:
                    old.disconnect()
                except Exception:
                    pass
            self.spec = new_spec
            if not isinstance(new_spec, StitchedSpectrometer):
                # Any single device replaces a stitched pair wholesale — the
                # slot->member mapping is only meaningful while the pair lives.
                self._multi_members = [None, None]
            if new_spec is not None:
                # Clamped to the INCOMING device's range: the box still carries
                # the outgoing device's value, and 0.05 ms from an Avantes is a
                # hard error on an Ocean unit with a 1 ms floor. _sync_multi_ui
                # below reseeds the box from whatever the device ended up with.
                lo, hi = _exposure_bounds(new_spec)
                new_spec.set_integration_time(
                    min(hi, max(lo, float(self.spin_integration.value()))))
            # The old device's frames and dark are meaningless for the new one
            # — and a different pixel count (certain with a stitched grid)
            # would crash the dark subtraction outright.
            self.background = None
            self.background_members = None
            self.background_exposures = None
            self._dark_member_warned = False
            if new_spec is None:
                # Disconnecting: take the curve off the panel too, or the last
                # frame would sit frozen under the "no spectrometer" message.
                self.canvas.clear_spectrum()
            else:
                self.canvas.clear_members()  # a dead pair's curves can't linger
            # And the reference with them, in both cases: it is a spectrum from
            # the outgoing device, on that device's pixel grid, and comparing
            # the incoming one against it would be meaningless where it was not
            # simply misleading.
            self.canvas.clear_reference()
            # Same for both halves of alignment mode: a difference and a raw
            # trace describe the device that measured them, right down to the
            # pixel grid they sit on.
            self._close_align_mode()
            self._clear_align_trace()
            self.chk_dark.setChecked(False); self.chk_dark.setEnabled(False)
            self.last_spectrum = None
            self._live_frame = None
            self._sat_frame = None
            self._sat_peak = -1.0
            # New device, new full scale — a latched warning about the old one
            # would be meaningless (and its threshold plain wrong).
            self._reset_saturation()
            # Re-centre spectrum when a new spectrometer comes online
            self._pending_fit = True
            self.canvas.autoscale_x = True
            if hasattr(self, 'dlg_graphics'):
                self.dlg_graphics.chk_auto_x.setChecked(True)
        # Outside the lock: pure widget state, no device I/O.
        self._sync_multi_ui()
        return True, ""

    def _use_sim_stage(self):
        ok, err = self._apply_stage(SimulatedStage(travel_mm=300.0))
        if not ok:
            return False, err
        self.status.showMessage("Stage: simulated.", 4000)
        return True, "Stage set to simulated."

    def _disconnect_stage(self):
        ok, err = self._apply_stage(None)
        if not ok:
            return False, err
        self.status.showMessage("Stage disconnected.", 4000)
        return True, "Stage disconnected."

    def _connect_real_stage(self, conn=None):
        """Connect a Kinesis stage. `conn` names one the caller already chose
        (the Stage dialog's scan list); without it this enumerates and picks or
        asks, which is what any other caller needs."""
        if conn is None:
            try:
                devices = list_kinesis_stages()   # brief per-device model query
            except Exception as e:
                return False, f"Stage connect failed: {e}"
            if not devices:
                return False, "Stage connect failed: No Kinesis devices found."
            if len(devices) > 1:
                conn = DevicePickerDialog.pick(
                    self.dlg_stage, devices, "Select Kinesis Stage",
                    f"{len(devices)} Kinesis devices found — choose one:")
                if conn is None:
                    return False, "Cancelled — stage unchanged."
            else:
                conn = devices[0][1]
        try:
            # Identifies the stage and refuses anything it can't calibrate in mm.
            stage = KinesisStage(serial=conn)
        except Exception as e:
            return False, f"Stage connect failed: {e}"
        ok, err = self._apply_stage(stage)
        if not ok:
            self._drop(stage)      # never adopted — don't leak the connection
            return False, err
        self.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        travel = (f"Travel {stage.travel_mm:.0f} mm." if stage.travel_mm else
                  "Travel unknown — no soft range limit, set one in the scan "
                  "config if you need it.")
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.scan_cfg.zero_pos_um:.1f} um). {travel}")

    def _connect_zaber_stage(self, port=None):
        try:
            stage = ZaberStage(port=port)     # blank port -> auto-scan
        except Exception as e:
            return False, f"Zaber connect failed: {e}"
        ok, err = self._apply_stage(stage)
        if not ok:
            self._drop(stage)      # never adopted — release the serial port
            return False, err
        if stage.needs_homing:
            self.status.showMessage(
                f"Stage: {stage.name} — NOT HOMED, home it before scanning.", 0)
            return True, (
                f"Stage connected: {stage.name}, but the axis is NOT HOMED. Its "
                f"position readback has no physical meaning until you press "
                f"Home, and homing will move the coordinate frame — so home "
                f"first, then set zero. Scans are blocked until then.")
        self.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        backlash_um = _stage_to_um(stage.backlash_mm)
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.scan_cfg.zero_pos_um:.1f} um). "
                      f"Backlash approach {backlash_um:.0f} um — Zaber does no "
                      f"backlash correction of its own, and without this the "
                      f"marked zero and a scan sweep sit in frames that differ "
                      f"by the screw slack.")

    def _connect_piezo_jena_stage(self, port=None):
        try:
            stage = PiezoJenaStage(port=port)     # blank port -> auto-scan
        except Exception as e:
            return False, f"Piezo Jena connect failed: {e}"
        ok, err = self._apply_stage(stage)
        if not ok:
            self._drop(stage)      # never adopted — release the serial port
            return False, err
        self.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.scan_cfg.zero_pos_um:.1f} um).")

    def _use_sim_spectrometer(self):
        ok, err = self._apply_spectrometer(self._make_sim_spectrometer())
        if not ok:
            return False, err
        # The full-band simulator is not reachable by id, so no slot can name
        # it — empty them rather than leave them pointing at whatever was
        # connected before.
        self._multi["serials"] = [None, None]
        self._multi["labels"]  = [None, None]
        self.status.showMessage(f"Spectrometer: {self.spec.name}.", 4000)
        return True, f"Simulated: {self.spec.pulse_label}."

    def _use_sim_pair(self):
        """Both slots on simulated halves, connected as a stitched pair.

        The only way to exercise multi-spectrometer mode without two devices on
        the bench — _open_slot_device matches the SIM_SLOT_DEVICES sentinels to
        the half-band simulators, so this is the ordinary pair path with the
        slots filled in for you.

        The slot labels follow what the sentinels will actually open: the two
        named instruments when the beam brings a pair of its own, the two
        halves of its band otherwise.
        """
        models = sim_spectrometer_pair(self.sim_pulse, self.sim_gate)
        for slot, (ident, label) in enumerate(SIM_SLOT_DEVICES):
            self._multi["serials"][slot] = ident
            self._multi["labels"][slot]  = (
                label if models is None
                else SIM_SPECTROMETERS[models[slot]]["name"])
        return self._connect_multi_pair()

    def _disconnect_spectrometer(self):
        """Release whatever is connected and empty both slots. (ok, msg)."""
        ok, err = self._apply_spectrometer(None)
        if not ok:
            return False, err
        self._multi["serials"] = [None, None]
        self._multi["labels"]  = [None, None]
        self.status.showMessage("Spectrometer disconnected.", 4000)
        return True, "Spectrometer disconnected."

    def _live_seabreeze_backend(self):
        """Backend name of the currently connected seabreeze device, or None
        when no live device is a seabreeze one.

        Every member is checked, not just spec1: StitchedSpectrometer assigns
        spec1 by wavelength, so in a mixed-vendor pair the seabreeze device can
        land in either slot. Missing it would skip the release step below and
        let select_seabreeze_backend shut the API down under a LIVE device.
        calibration_targets() is the existing "physical devices behind the
        facade" accessor — [self] for a single device, both members for a pair.
        """
        if not self._have_spec():
            return None
        for spec in self.spec.calibration_targets():
            if isinstance(spec, SeabreezeSpectrometer):
                return getattr(spec, "backend", None)
        return None

    def _list_spectrometers(self):
        """Enumerate every vendor's attached spectrometers as tagged ids.
        Selecting a seabreeze backend tears the previous backend's API down,
        which would sever a device still open through it — so any such device
        is released first. Raises RuntimeError when that release is refused
        (scan running / feed busy)."""
        live = self._live_seabreeze_backend()
        if live is not None and live != self.seabreeze_backend:
            ok, err = self._apply_spectrometer(None)
            if not ok:
                raise RuntimeError(err)
        return list_spectrometers(self.seabreeze_backend)

    def _spectrometer_scan_notes(self):
        """Why a vendor contributed nothing to the last enumeration.

        list_spectrometers deliberately swallows a vendor whose SDK is missing,
        so one absent driver cannot hide the other vendor's devices. That is
        right for the list and wrong for the operator staring at an empty one,
        so the same two calls are made again here purely for their errors —
        this is what the old per-vendor connect buttons used to surface.
        """
        notes = []
        try:
            list_seabreeze_spectrometers(self.seabreeze_backend)
        except Exception as e:
            notes.append(f"seabreeze ({self.seabreeze_backend}): {e}")
        try:
            list_avantes_spectrometers()
        except Exception as e:
            notes.append(f"Avantes: {e}")
        return notes

    def _avantes_devices(self):
        """Every live Avantes device — [], [one], or both members of a mixed
        stitched pair. calibration_targets() is the existing accessor for
        "the physical devices behind whatever facade is connected", so this
        keeps working for any future composite."""
        if not self._have_spec():
            return []
        return [s for s in self.spec.calibration_targets()
                if isinstance(s, AvantesSpectrometer)]

    @staticmethod
    def _drop(device):
        """Close a device we opened but did not end up adopting."""
        try:
            device.disconnect()
        except Exception:
            pass

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        tb = QToolBar("Main"); tb.setMovable(False)
        self.addToolBar(tb)
        b_set = QPushButton("Acquisition Settings")
        b_set.clicked.connect(self.dlg_settings.toggle)
        tb.addWidget(b_set)
        # One button per device, replacing the old combined Hardware button.
        # They are the same dialogs the panel Connect buttons open — this row is
        # the way back to them once something IS connected and the panels have
        # swapped over to their controls.
        self.btn_spec_dlg = QPushButton("Spectrometer")
        self.btn_spec_dlg.setToolTip(
            "Connect one or two spectrometers, assign calibrations, and manage "
            "the stitch between a pair")
        self.btn_spec_dlg.clicked.connect(self.dlg_spec.toggle)
        tb.addWidget(self.btn_spec_dlg)
        self.btn_stage_dlg = QPushButton("Stage")
        self.btn_stage_dlg.setToolTip("Connect a delay stage and set its "
                                      "backlash approach margin")
        self.btn_stage_dlg.clicked.connect(self.dlg_stage.toggle)
        tb.addWidget(self.btn_stage_dlg)
        # Its own button, deliberately apart from the two above: simulated
        # hardware is no longer a default anything, and nothing that reads as a
        # hardware control should be able to hand back synthetic data.
        self.btn_sim_dlg = QPushButton("Simulation")
        self.btn_sim_dlg.setToolTip(
            "Run against simulated hardware — a synthetic beam, spectrometer, "
            "stage or stitched pair, with no instrument attached")
        self.btn_sim_dlg.clicked.connect(self.dlg_sim.toggle)
        tb.addWidget(self.btn_sim_dlg)
        b_gfx = QPushButton("Graphics Settings")
        b_gfx.clicked.connect(lambda: self.dlg_graphics.toggle())
        tb.addWidget(b_gfx)
        b_align = QPushButton("Alignment")
        b_align.setToolTip("Alignment-mode sweep settings — the step the Δ "
                           "button over the spectrum measures at")
        b_align.clicked.connect(self.dlg_align.toggle)
        tb.addWidget(b_align)
        # A window, like its neighbours — not the drop-down of formats it used
        # to be. See ExportDialog for why.
        self.btn_export = QPushButton("Export")
        self.btn_export.setToolTip(
            "Save the last scan — pick the format and the file name")
        self.btn_export.clicked.connect(self.dlg_export.toggle)
        tb.addWidget(self.btn_export)
        tb.addWidget(self._build_calibration_button())
        # Vendor-specific, so it is HIDDEN rather than disabled when there is
        # no Avantes attached — the toolbar already carries six buttons, and a
        # permanently greyed one is noise on a bench that has none. Visibility
        # is owned by _refresh_avantes_button.
        self.btn_avantes = QPushButton("Avantes")
        self.btn_avantes.setToolTip(
            "Avantes-specific settings: averaging, ADC resolution, "
            "corrections, triggering, detector temperature")
        self.btn_avantes.clicked.connect(self.dlg_avantes.toggle)
        self.btn_avantes.setVisible(False)
        tb.addWidget(self.btn_avantes)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.btn_theme = QPushButton()
        # Derived from self._theme rather than hard-coded to the dark-mode pair:
        # the theme is restored from settings.json before _build_ui runs, so a
        # session that ended in light mode would otherwise come back showing the
        # sun and offering to switch to the mode it is already in. Same
        # expressions as _apply_theme, which owns the runtime switch — the
        # button always advertises what a CLICK produces.
        self.btn_theme.setIcon(QIcon(str(MOON_ICON if self._theme == "light"
                                         else SUN_ICON)))
        self.btn_theme.setIconSize(QSize(18, 18))
        self.btn_theme.setFixedWidth(42)
        self.btn_theme.setToolTip("Switch to dark mode" if self._theme == "light"
                                  else "Switch to light mode")
        self.btn_theme.clicked.connect(self._toggle_theme)
        tb.addWidget(self.btn_theme)
        # Save the plots as a picture, between the two controls that decide what
        # those plots LOOK like — the theme and the layout. It belongs with them
        # rather than beside Export: Export writes the measurement, in formats
        # another program reads, while this writes the figure, for a slide or a
        # page. Icon from _refresh_picture_icon (one file per theme).
        self.btn_picture = QPushButton()
        self.btn_picture.setIconSize(QSize(18, 18))
        self.btn_picture.setFixedWidth(42)
        self._refresh_picture_icon()
        self.btn_picture.clicked.connect(self._save_picture)
        tb.addWidget(self.btn_picture)
        # Plot layout toggle; icon and tooltip come from _refresh_layout_button.
        self.btn_layout = QPushButton()
        self.btn_layout.setIconSize(QSize(18, 18))
        self.btn_layout.setFixedWidth(42)
        self.btn_layout.clicked.connect(self._toggle_layout)
        tb.addWidget(self.btn_layout)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        ctrl = QWidget()
        ctrl.setMinimumWidth(280); ctrl.setMaximumWidth(320)
        cl = QVBoxLayout(ctrl); cl.setSpacing(6); cl.setContentsMargins(0, 0, 4, 0)
        cl.addWidget(self._build_spectrum_group())
        cl.addWidget(self._build_stage_group())
        cl.addWidget(self._build_scan_group())
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(312)
        root.addWidget(scroll)

        self.canvas = FrogCanvas()
        root.addWidget(self.canvas, stretch=1)

        # ── Panel header buttons ─────────────────────────────────────────────
        # All children of the canvas so they float over the figure margin, and
        # all positioned by _position_panel_buttons: each one belongs to the
        # panel it acts on, and every panel edge moves with the split fraction,
        # the window size and the layout mode.
        self.btn_autofit = QPushButton(self.canvas)
        self.btn_autofit.setObjectName("overlay")
        self.btn_autofit.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_autofit_icon()
        self.btn_autofit.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_autofit.setToolTip(
            "Auto-fit spectrum X and Y axes to current data.\n"
            + RIGHT_CLICK_HINT)
        self.btn_autofit.show()
        self.btn_autofit.clicked.connect(self._autofit_spectrum)

        # Continuous auto-scale, next to the one-shot fit it complements: that
        # button fits ONCE, this one says whether the panel goes on following
        # the data. It is the third way in to the same state — Graphics
        # Settings' checkbox and a right-click on the panel are the other two —
        # so it drives reset_axes(), the very function the right-click calls,
        # rather than a parallel path that could drift out of step with it.
        # Its own checked state is never set here: _refresh_autoscale_button
        # reads it back off the canvas, whichever of the three moved it.
        self.btn_autoscale = QPushButton(self.canvas)
        self.btn_autoscale.setObjectName("overlay")
        self.btn_autoscale.setCheckable(True)
        self.btn_autoscale.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_autoscale.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_autoscale_icon()
        self.btn_autoscale.show()
        self.btn_autoscale.clicked.connect(self._on_autoscale_clicked)

        # Log/linear, at the top of the y axis it rescales — beside the padlock
        # that freezes the same axis, so the two controls that decide what the
        # y axis does sit together on it. The scale was previously reachable
        # only by knowing to click the tick-label strip (still supported) or by
        # opening Graphics Settings; neither announces itself.
        #
        # The text is the CURRENT scale, matching the padlock beside it rather
        # than the "icon shows what a click gives you" rule the pair toggles
        # use. Both of these read as a readout of the axis' state.
        #
        # Square like every other header button, not widened for its label: at
        # #overlay's 12px/600 "log" measures 18 px in Segoe UI and 21 on the
        # widest face in FONT_STACK, against the 22 px this button's content box
        # gives it — so the word fits as it is, and a lone oblong in the row was
        # the only thing that had ever made it look like a different kind of
        # control.
        self.btn_logscale = QPushButton(self.canvas)
        self.btn_logscale.setObjectName("overlay")
        self.btn_logscale.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_logscale.show()
        # Through the checkbox, not canvas.set_log_scale: chk_log is the single
        # source of truth for the scale (it is what gets persisted, and what the
        # y-strip click already routes through), so driving the canvas directly
        # would leave the dialog showing the wrong thing.
        # Lambda, not a direct connect: dlg_graphics is built further down.
        self.btn_logscale.clicked.connect(
            lambda: self.dlg_graphics.chk_log.toggle())

        # Auto-stitch, in the spectrum header rather than only in the
        # Spectrometer window: it is fitted against what is on the panel right
        # now (it needs light across the overlap), so it belongs next to the
        # curve you are judging it by. Pair-only, like the overlay toggle —
        # _refresh_autostitch_button owns its visibility, and its icon comes
        # from _refresh_autostitch_icon (per-theme, like the alignment mark).
        self.btn_autostitch = QPushButton(self.canvas)
        self.btn_autostitch.setObjectName("overlay")
        self.btn_autostitch.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_autostitch.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_autostitch_icon()
        self.btn_autostitch.hide()
        self.btn_autostitch.clicked.connect(
            lambda: self._menu_result(self._fit_stitch_factor))

        # Live-feed start/stop. In the spectrum header rather than the side
        # panel: it is the control most often reached for while watching that
        # panel, and it was costing the panel a full-width button. Icon and
        # colour come from _refresh_feed_button.
        self.btn_feed = QPushButton(self.canvas)
        self.btn_feed.setCheckable(True); self.btn_feed.setChecked(True)
        self.btn_feed.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_feed.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.btn_feed.show()
        self.btn_feed.toggled.connect(self._toggle_feed)
        self._refresh_feed_button()

        # Freeze the spectrum as a grey reference, next to the feed whose newest
        # frame it captures. A generated glyph rather than a pair of PNGs, like
        # that neighbour: a plus is two bars, and drawing it from the live
        # PALETTE saves two files that would be wrong the moment an accent moved.
        self.btn_freeze = QPushButton(self.canvas)
        self.btn_freeze.setObjectName("overlay")
        self.btn_freeze.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_freeze.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_freeze_icon()
        self.btn_freeze.setToolTip(
            "Freeze the spectrum as a grey reference under the live curve — "
            "make an adjustment, and you can see exactly what it changed.\n"
            "Press again to replace the reference with the current spectrum, "
            "or right-click to drop it.")
        self.btn_freeze.show()
        self.btn_freeze.clicked.connect(self._freeze_reference)
        # Dropping a reference has no button of its own: it is the rare half of
        # a pair whose common half (hide) is beside it already, and a third
        # header button for it would cost every session the room to say so.
        self.btn_freeze.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_freeze.customContextMenuRequested.connect(
            lambda _pos: self._clear_reference())

        # …and the reference's own hide/show. Pair-only in the same sense as the
        # alignment sweep's re-run button: it exists exactly as long as there is
        # something for it to act on, so _refresh_reference_button owns both its
        # visibility and its icon.
        self.btn_refshow = QPushButton(self.canvas)
        self.btn_refshow.setObjectName("overlay")
        self.btn_refshow.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_refshow.setIconSize(QSize(HDR_ICON, HDR_ICON))
        # Not checkable: the eye already shows the state by showing the action,
        # and a checked frame around it would be a second readout of the same
        # bit — one that says "on" while the eye says "hide".
        self.btn_refshow.hide()
        self.btn_refshow.clicked.connect(self._toggle_reference)

        # Per-spectrometer view toggle. Only meaningful for a stitched pair, so
        # it stays hidden otherwise; its icon and tooltip are set by
        # _refresh_overlay_button.
        self.btn_overlay = QPushButton(self.canvas)
        self.btn_overlay.setObjectName("overlay")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_overlay.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.btn_overlay.hide()
        self.btn_overlay.toggled.connect(self._on_overlay_toggled)

        # Alignment mode, spectrum side: opens the deviation panel under the
        # spectrum, steps to +/-x and +/-2x, and plots the two differences there.
        self.btn_align_spec = QPushButton(self.canvas)
        self.btn_align_spec.setObjectName("overlay")
        self.btn_align_spec.setCheckable(True)
        self.btn_align_spec.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_align_spec.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_align_button()
        self.btn_align_spec.setToolTip(
            "Alignment mode — opens a deviation panel under the spectrum, then "
            "measures at −2x, −x, +x, +2x (Alignment → Alignment step) and "
            "plots S(+x)−S(−x) and S(+2x)−S(−2x) in it.\nA symmetric pulse "
            "gives two flat curves on zero. Use ⟳ to re-measure after an "
            "adjustment — the previous sweep stays as a shaded area to compare "
            "against.\nPress again to close the panel.")
        self.btn_align_spec.show()
        self.btn_align_spec.toggled.connect(self._on_align_spec_toggled)

        # Re-run the sweep without leaving the mode. Only meaningful while the
        # panel is open, so it lives and dies with it — and the comparison it
        # makes possible (this sweep against the last) is the reason the panel
        # keeps a ghost at all.
        self.btn_align_refresh = QPushButton(self.canvas)
        self.btn_align_refresh.setObjectName("overlay")
        self.btn_align_refresh.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_align_refresh.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_align_refresh_icon()
        self.btn_align_refresh.setToolTip(
            "Re-run the alignment sweep. The sweep now on the panel is kept as "
            "a shaded area underneath, so you can see whether the last "
            "adjustment made the deviation better or worse.")
        self.btn_align_refresh.hide()
        self.btn_align_refresh.clicked.connect(self._start_align_sweep)

        # Alignment mode, trace side — the trace panel's own header.
        # Wider than the rest: it is the one button still carrying a word.
        self.btn_align_trace = QPushButton("RAW", self.canvas)
        self.btn_align_trace.setObjectName("overlay")
        self.btn_align_trace.setCheckable(True)
        self.btn_align_trace.setFixedSize(40, HDR_BTN)
        self.btn_align_trace.hide()
        self.btn_align_trace.toggled.connect(self._on_align_trace_toggled)

        # Symmetry mode, trace panel. Always present — unlike RAW it does not
        # depend on the pair — but only enabled once a finished scan gives it an
        # autocorrelation to find the fold line in.
        self.btn_symmetry = QPushButton(self.canvas)
        self.btn_symmetry.setObjectName("overlay")
        self.btn_symmetry.setCheckable(True)
        if SYMMETRY_ICON.exists():
            self.btn_symmetry.setIcon(QIcon(str(SYMMETRY_ICON)))
            self.btn_symmetry.setIconSize(QSize(HDR_ICON, HDR_ICON))
        else:
            self.btn_symmetry.setText("|")
        self.btn_symmetry.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_symmetry.show()
        self.btn_symmetry.toggled.connect(self._on_symmetry_toggled)
        self._refresh_symmetry_button()

        # ── "No spectrometer connected" ──────────────────────────────────────
        # A Qt child of the canvas, not a matplotlib artist: the canvas blits,
        # so a text artist would have to force a full redraw on every show and
        # hide, and this one has to carry a clickable button anyway. Stretched
        # over the whole spectrum panel by _position_panel_buttons, which
        # already runs on every relayout, resize and layout-mode change.
        self.pnl_no_spec = NoDeviceOverlay(
            self.canvas, "No spectrometer connected", "Connect Spectrometer",
            self.dlg_spec.open_fresh)
        self.pnl_no_spec.hide()

        self.canvas.axes_relaid.connect(self._position_panel_buttons)
        self._position_panel_buttons()

        # The reference's toggle is an INDICATOR as much as a control, and the
        # canvas is what owns the state — a freeze from the button, a drop from
        # a device swap. One connection covers every way in, so the eye cannot
        # end up describing a reference that is not there.
        self.canvas.reference_changed.connect(self._refresh_reference_button)
        self._refresh_reference_button()

        self._refresh_layout_button()    # needs the canvas for the current mode

        self.dlg_graphics = GraphicsSettingsDialog(self.canvas, self)
        # Mouse zoom/reset keeps the dialog's spinboxes and auto-scale
        # checkboxes truthful; the y-axis click routes through chk_log so the
        # checkbox stays the single source of truth for the log scale.
        self.canvas.limits_changed.connect(self.dlg_graphics.sync_limits)
        self.canvas.log_toggle_requested.connect(self.dlg_graphics.chk_log.toggle)
        self.canvas.proportions_changed.connect(self.dlg_graphics.sync_proportions)
        self.canvas.fold_dragged.connect(self._on_fold_dragged)
        self.canvas.fold_reset_requested.connect(self._on_fold_reset)
        self.canvas.axis_edit_failed.connect(
            lambda msg: self.status.showMessage(msg, 4000))
        # The auto-scale toggle is an INDICATOR as much as a control, so it has
        # to follow the state however it was reached: limits_changed covers the
        # right-click and every zoom, the checkbox covers Graphics Settings.
        self.canvas.limits_changed.connect(self._refresh_autoscale_button)
        self.dlg_graphics.chk_auto_y.toggled.connect(self._refresh_autoscale_button)
        self._refresh_autoscale_button()
        # Same arrangement for the scale: the checkbox owns it, and everything
        # that can change it (this button, the y-strip click, the dialog, a
        # restored setting) goes through chk_log, so one connection keeps the
        # label honest.
        self.dlg_graphics.chk_log.toggled.connect(self._refresh_logscale_button)
        self._refresh_logscale_button()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_sat = QLabel(""); self.lbl_sat.setObjectName("satok")
        self.lamp = StatusLamp()
        # Second lamp pair for multi-spectrometer mode — one alarm per device.
        # Hidden until the mode is enabled.
        self.lbl_sat2 = QLabel(""); self.lbl_sat2.setObjectName("satok")
        self.lamp2 = StatusLamp()
        self.status.addPermanentWidget(self.lbl_sat)
        self.status.addPermanentWidget(self.lamp)
        self.status.addPermanentWidget(self.lbl_sat2)
        self.status.addPermanentWidget(self.lamp2)
        self.lbl_sat2.hide(); self.lamp2.hide()
        self._reset_saturation()
        self.status.showMessage(
            "No hardware connected — connect a spectrometer and a stage, or "
            "open Simulation to run without either.", 0)

    # ── Calibration menu ──────────────────────────────────────────────────────
    def _build_calibration_button(self):
        """Header Calibration control — assigns an intensity-calibration file
        to each connected physical spectrometer. Rebuilt on every open: both
        the calibration_files folder and the connected devices change at
        runtime."""
        menu = QMenu(self)
        menu.aboutToShow.connect(lambda: self._populate_calibration_menu(menu))
        self.btn_cal = QPushButton("Calibration")
        self.btn_cal.setToolTip(
            "Apply an intensity calibration (wavelength/factor text file) to "
            "the spectrometer — files live in calibration_files/ next to the "
            "program")
        self.btn_cal.setMenu(menu)
        return self.btn_cal

    @staticmethod
    def _reset_menu(menu):
        """Empty `menu` for repopulation, submenus and action groups included.

        QMenu.clear() only deletes the actions the menu itself owns. A submenu
        built by addMenu() is a child WIDGET whose menuAction() belongs to the
        submenu, and a QActionGroup is not an action at all — so neither is
        touched, and both stay children of a menu that lives for the whole
        session. These menus repopulate on every aboutToShow, so without this
        each open permanently added another set (along with the actions and
        lambdas parented to them, some of which capture a spectrometer that has
        since been disconnected).
        """
        # One call per type: PySide6's findChildren takes a single type, not the
        # tuple isinstance() would accept.
        for cls in (QMenu, QActionGroup):
            for child in menu.findChildren(cls,
                                           options=Qt.FindDirectChildrenOnly):
                child.setParent(None)
                child.deleteLater()
        menu.clear()

    @staticmethod
    def _calibration_files():
        try:
            return sorted(CALIBRATION_DIR.glob("*.txt"),
                          key=lambda p: p.name.lower())
        except OSError:
            return []

    def _populate_calibration_menu(self, menu):
        self._reset_menu(menu)
        if not self._have_spec():
            act = QAction("(no spectrometer connected)", menu)
            act.setEnabled(False)
            menu.addAction(act)
            return
        files = self._calibration_files()
        targets = self.spec.calibration_targets()
        for spec in targets:
            # Flat list for a single device; one submenu per member of a
            # stitched pair, so each takes its own file.
            dest = menu.addMenu(spec.name) if len(targets) > 1 else menu
            group = QActionGroup(dest)
            group.setExclusive(True)
            act = QAction("None (raw counts)", dest)
            act.setCheckable(True)
            act.setChecked(spec.calibration_name is None)
            act.triggered.connect(
                lambda _=False, s=spec: self._apply_calibration(s, None))
            group.addAction(act); dest.addAction(act)
            for f in files:
                act = QAction(f.stem, dest)
                act.setCheckable(True)
                act.setChecked(spec.calibration_name == f.stem)
                act.triggered.connect(
                    lambda _=False, s=spec, p=f: self._apply_calibration(s, p))
                group.addAction(act); dest.addAction(act)
            if not files:
                act = QAction("(no .txt files in calibration_files)", dest)
                act.setEnabled(False)
                dest.addAction(act)
        menu.addSeparator()
        act_add = QAction("Add new calibration…", menu)
        act_add.triggered.connect(self._add_calibration_file)
        menu.addAction(act_add)

    def _apply_calibration(self, spec, path) -> bool:
        """Assign `path` (or None = raw counts) to one physical spectrometer.

        Returns True only if the device really carries it now. A failure is
        raised to a modal dialog rather than a status message: an unloaded
        calibration leaves the device on raw counts while every label still
        names the file, and that mismatch silently corrupts a whole session's
        data — it must not be allowed to scroll past.
        """
        if self._scan_running():
            self.status.showMessage(
                "A scan is running — stop it before changing calibration.", 4000)
            return False
        if spec not in self.spec.calibration_targets():
            self.status.showMessage(
                "That spectrometer is no longer connected.", 4000)
            return False
        skipped = 0
        with self._device_lock() as ok:
            if not ok:
                self.status.showMessage(FEED_BUSY_MSG, 4000)
                return False
            try:
                if path is None:
                    spec.clear_calibration()
                else:
                    skipped = spec.set_calibration(path)
            except Exception as e:
                QMessageBox.warning(
                    self, "Calibration not applied",
                    f"{spec.name} could not load '{getattr(path, 'stem', path)}':"
                    f"\n\n{e}\n\nThe spectrometer is still on "
                    f"{spec.calibration_name or 'raw counts'}.")
                return False
        msg = (f"{spec.name}: calibration "
               + (f"'{path.stem}' applied." if path is not None
                  else "removed (raw counts)."))
        if skipped:
            msg += (f"  NOTE: {skipped} malformed line(s) in the file were "
                    f"skipped.")
        self.status.showMessage(msg, 8000 if skipped else 4000)
        return True

    def _add_calibration_file(self):
        """Copy a calibration file into calibration_files/ so it shows up in
        the menu — and, for the frozen build, survives next to the .exe."""
        src, _ = QFileDialog.getOpenFileName(
            self, "Add calibration file", "",
            "Calibration files (*.txt);;All files (*)")
        if not src:
            return
        src = Path(src)
        try:
            load_calibration_file(src)   # reject unparseable files up front
        except Exception as e:
            QMessageBox.warning(self, "Add calibration",
                                f"Not a usable calibration file:\n{e}")
            return
        dest = CALIBRATION_DIR / src.name
        try:
            CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not src.samefile(dest):
                if QMessageBox.question(
                        self, "Add calibration",
                        f"{dest.name} already exists in calibration_files — "
                        f"overwrite it?") != QMessageBox.Yes:
                    return
            if not (dest.exists() and src.samefile(dest)):
                shutil.copyfile(src, dest)
        except OSError as e:
            QMessageBox.warning(self, "Add calibration",
                                f"Could not copy into {CALIBRATION_DIR}:\n{e}")
            return
        self.status.showMessage(
            f"Calibration '{src.stem}' added — assign it from the "
            f"Calibration menu.", 5000)

    # ── Spectrometer slots ───────────────────────────────────────────────────
    # Two slots, always. Slot 1 alone is a single spectrometer; both filled is a
    # stitched pair. The Spectrometer dialog is the only caller — it drives
    # these three, and everything else about the pair (stitch factor, overlap
    # band, per-slot calibration) is unchanged from when this was a menu.
    @contextmanager
    def _slots_rolled_back_on_failure(self):
        """Restore the whole slot table if the connect it wraps fails.

        A snapshot of all three lists, not of the one entry being edited:
        clearing slot 1 promotes slot 2 into it, so an edit can touch every
        field, and a per-field undo would leave the table half-shuffled. Yields
        a one-element list the body puts its (ok, msg) into.
        """
        before = {k: list(v) for k, v in self._multi.items()}
        box = []
        yield box
        if box and not box[0][0]:
            self._multi = before

    def _set_slot(self, slot, serial, label):
        """Point one slot at a device and connect whatever the slots now
        describe. (ok, msg)."""
        if self._scan_running():
            return False, ("A scan is running — stop it before changing "
                           "spectrometers.")
        with self._slots_rolled_back_on_failure() as out:
            self._multi["serials"][slot] = serial
            self._multi["labels"][slot]  = label
            if all(self._multi["serials"]):
                out.append(self._connect_multi_pair())
            elif slot == 0:
                out.append(self._connect_single_slot())
            else:
                # Slot 2 filled with slot 1 still empty. Nothing to connect
                # against yet, so remember it and say so rather than opening
                # this device on its own — slot 1 is the single-device slot,
                # by definition.
                out.append((True, f"Slot 2: {label}. Fill slot 1 to connect "
                                  f"the pair."))
        return out[0]

    def _clear_slot(self, slot):
        """Empty one slot and connect whatever is left. (ok, msg)."""
        if self._scan_running():
            return False, ("A scan is running — stop it before changing "
                           "spectrometers.")
        if slot == 0 and self._multi["serials"][1] is None:
            return self._disconnect_spectrometer()
        with self._slots_rolled_back_on_failure() as out:
            for key in ("serials", "labels", "cals"):
                self._multi[key][slot] = None
            if slot == 0:
                # Slot 1 is the single-device slot, so slot 2 alone is not a
                # configuration. Promote it rather than leave a device named in
                # a slot with nothing driving it.
                for key in ("serials", "labels", "cals"):
                    self._multi[key][0] = self._multi[key][1]
                    self._multi[key][1] = None
                out.append(self._connect_single_slot())
            elif not self._pair_live():
                out.append((True, "Slot 2 cleared."))
            else:
                # A live pair loses a member: release both, then reopen slot 1
                # as the single device. Filling slot 2, in reverse.
                out.append(self._connect_single_slot())
        return out[0]

    def _connect_single_slot(self):
        """Open slot 1 alone as the connected spectrometer. (ok, msg).

        The single-device counterpart of _connect_multi_pair, and it releases
        the current handle for the same reason: a vendor SDK cannot open the
        same spectrometer twice, and slot 1 may well be a member of the pair
        being torn down.
        """
        ident = self._multi["serials"][0]
        if ident is None:
            return self._disconnect_spectrometer()
        if self._have_spec():
            ok, err = self._apply_spectrometer(None)
            if not ok:
                return False, err
        cal = self._multi["cals"][0]
        ragged = 0
        try:
            spec = self._open_slot_device(ident)
            if cal is not None:
                ragged = spec.set_calibration(cal)
        except Exception as e:
            return False, f"Spectrometer connect failed: {e}"
        ok, err = self._apply_spectrometer(spec)
        if not ok:
            self._drop(spec)
            return False, err
        self.status.showMessage(f"Spectrometer: {spec.name}", 5000)
        msg = f"Spectrometer connected: {spec.name}."
        if ragged:
            msg += (f"\n\nNOTE: {ragged} malformed line(s) were skipped in "
                    f"'{cal.stem}' — check the file.")
        return True, msg

    def _select_slot_calibration(self, slot, path):
        member = self._multi_members[slot]
        if member is None and slot == 0 and not self._pair_live():
            # Single-device mode: slot 1 IS the connected spectrometer, so its
            # calibration can be applied to the live device straight away.
            member = self.spec
        if member is not None:
            # Commit the slot ONLY once the device really carries the file.
            # The dialog's combo is drawn from _multi["cals"], so recording it
            # first made the combo claim a calibration that had failed to load
            # and left that device silently on raw counts.
            if not self._apply_calibration(member, path):
                return
            self._multi["cals"][slot] = path
            return
        # Not live yet — nothing to verify against, so the file is checked on
        # its own and only then remembered for the connect.
        if path is not None:
            try:
                _wl, _fac, skipped = load_calibration_file(path)
            except Exception as e:
                QMessageBox.warning(
                    self, "Calibration not applied",
                    f"'{path.stem}' could not be read:\n\n{e}\n\n"
                    f"Slot {slot + 1} is unchanged.")
                return
        else:
            skipped = 0
        self._multi["cals"][slot] = path
        msg = (f"Slot {slot + 1} calibration: "
               f"{path.stem if path else 'none'} — applied when the device "
               f"connects.")
        if skipped:
            msg += f"  NOTE: {skipped} malformed line(s) skipped."
        self.status.showMessage(msg, 8000 if skipped else 4000)

    def _connect_multi_pair(self):
        """Open both slot devices and swap in the stitched pair. (ok, msg)."""
        if self._scan_running():
            return False, "A scan is running — stop it before changing hardware."
        s1, s2 = self._multi["serials"]
        if s1 == s2:
            return False, ("Both slots point at the same spectrometer — pick "
                           "two different devices.")
        # Release any handle we already hold on one of these devices first: a
        # vendor SDK cannot open the same spectrometer twice.
        if self._have_spec():
            ok, err = self._apply_spectrometer(None)
            if not ok:
                return False, err
        opened = []
        ragged = []
        try:
            for serial in (s1, s2):
                opened.append(self._open_slot_device(serial))
            for slot, (spec, cal) in enumerate(zip(opened, self._multi["cals"])):
                if cal is not None:
                    if spec.set_calibration(cal):
                        ragged.append(f"slot {slot + 1} ('{cal.stem}')")
            stitched = StitchedSpectrometer(*opened)
        except Exception as e:
            for s in opened:
                self._drop(s)
            return False, f"Stitched connect failed: {e}"
        ok, err = self._apply_spectrometer(stitched)
        if not ok:
            self._drop(stitched)   # disconnects both members
            return False, err
        self._multi_members = list(opened)      # slot order, not blue/red
        self._stitch_stale = False              # fresh pair, equal exposures
        self._reset_saturation()   # slot-ordered lamp tooltips need the mapping
        # Again, now that the slot mapping exists: _apply_spectrometer ran
        # before _multi_members was assigned, so its _sync_multi_ui saw
        # _slot_members() fall back to (blue, red) and would have seeded the
        # S1/S2 spinboxes from the wrong members.
        self._sync_multi_ui()
        wl = stitched.wavelengths
        lo, hi = stitched.overlap_band
        self.status.showMessage(f"Spectrometer: {stitched.name}", 5000)
        msg = (f"Stitched pair connected: {stitched.name}, "
               f"{wl[0]:.0f}–{wl[-1]:.0f} nm, overlap band {lo:.1f}–{hi:.1f} "
               f"nm. Use Auto-stitch with light across the overlap to match "
               f"the two devices.")
        if ragged:
            msg += (f"\n\nNOTE: malformed lines were skipped in the "
                    f"calibration for {', '.join(ragged)} — check the file.")
        return True, msg

    def _menu_result(self, fn):
        """Run an (ok, msg) action from a header button; failures pop a message
        box — the canvas header has no inline status label like the dialogs."""
        ok, msg = fn()
        if ok:
            if msg:
                self.status.showMessage(msg, 8000)
        else:
            QMessageBox.warning(self, "Spectrometer", msg)

    def _usable_member_darks(self):
        """The recorded member darks, or None when they cannot be applied.

        All-or-nothing on purpose: subtracting from one member and not the
        other would put the two on different baselines, which is precisely what
        the overlap comparison — and the stitch fit — exist to measure. The one
        gate every consumer goes through, so the per-spectrometer view, the
        merged view and Auto-stitch can never disagree about whether a dark is
        good.
        """
        darks = self.background_members
        frames = getattr(self.spec, "last_member_raw", None)   # atomic read
        if darks is None or frames is None or len(darks) != len(frames):
            return None
        if not all(np.shape(d) == np.shape(f) for d, f in zip(darks, frames)):
            return None
        return darks

    def _fit_darks(self):
        """The member darks Auto-stitch fits against, or None.

        Deliberately NOT gated on the Subtract Dark checkbox: that is a display
        preference, whereas the fit is physics. Two members at different
        exposures carry different pedestals, and a fit over signal-plus-pedestal
        returns a factor biased by the difference.
        """
        return self._usable_member_darks()

    def _fit_stitch_factor(self):
        if not self._pair_live():
            return False, "No stitched pair is connected."
        if self._scan_running():
            return False, "A scan is running — stop it before fitting."
        darks = self._fit_darks()
        with self._device_lock() as ok:
            if not ok:
                return False, FEED_BUSY_MSG
            try:
                factor = self.spec.fit_stitch_factor(darks)
            except Exception as e:
                return False, f"Stitch-factor fit failed: {e}"
        self._stitch_stale = False
        lo, hi = self.spec.overlap_band
        res = self.spec.stitch_residual
        quality = ("" if res is None else
                   f" Residual mismatch {res * 100:.1f}% — "
                   + ("the two spectra agree across the band."
                      if res < 0.05 else
                      "one scalar does not reconcile them here; check the "
                      "calibrations or narrow the band."))
        return True, (f"Stitch factor fitted: {factor:.4g} "
                      f"(applied to {self.spec.spec1.name}) over "
                      f"{lo:.1f}–{hi:.1f} nm"
                      + ("" if darks is not None else ", no dark subtracted")
                      + f".{quality}")

    def _set_stitch_factor(self):
        if not self._pair_live():
            return
        val, ok = QInputDialog.getDouble(
            self, "Set stitch factor",
            f"Multiplier applied to {self.spec.spec1.name} before stitching:",
            self.spec.stitch_factor, 1e-6, 1e6, 6)
        if ok:
            self.spec.stitch_factor = float(val)   # atomic — no lock needed
            self.spec.stitch_residual = None   # hand-set: nothing was measured
            self._stitch_stale = False   # the user just said what they want
            # No dark warning: the merged background is derived from the member
            # frames at the current factor, so it follows this change by itself.
            self.status.showMessage(f"Stitch factor set to {val:.4g}.", 4000)

    def _set_overlap_band(self):
        """Pick the sub-range of the overlap that the fit and blend use."""
        if not self._pair_live():
            return
        lo, hi = self.spec.overlap_band
        glo, ghi = self.spec.geometric_overlap
        new_lo, ok = QInputDialog.getDouble(
            self, "Overlap band",
            f"Band START (nm) — the two spectrometers share "
            f"{glo:.1f}–{ghi:.1f} nm.\nEnter {glo:.1f} to reset to the full "
            f"shared range:", lo, glo, ghi, 1)
        if not ok:
            return
        new_hi, ok = QInputDialog.getDouble(
            self, "Overlap band", "Band END (nm):", hi, glo, ghi, 1)
        if not ok:
            return
        try:
            lo, hi = self.spec.set_band(new_lo, new_hi)
        except Exception as e:
            QMessageBox.warning(self, "Overlap band", str(e))
            return
        # Push the band to the canvas now rather than waiting for the next
        # overlay frame, so the shading moves as soon as the dialog closes. The
        # shading only renders in the per-spectrometer view (set_overlap_band
        # gates it on that); the combined curve has already been crossfaded, so
        # there is nothing there for it to mark.
        self.canvas.set_overlap_band(lo, hi)
        self.status.showMessage(
            f"Overlap band set to {lo:.1f}–{hi:.1f} nm — re-run Auto-stitch "
            f"to fit over it.", 6000)

    @staticmethod
    def _connect_page(grp, body, button):
        """Put a panel's controls behind a Connect button.

        Exactly one of the two is visible; _sync_connection_ui picks which.
        Plain show/hide rather than a QStackedWidget, which sizes every page to
        the tallest and would leave a lone Connect button floating in the
        middle of a panel-sized empty box.
        """
        lay = QVBoxLayout(grp); lay.setSpacing(4)
        lay.addWidget(button)
        lay.addWidget(body)
        button.hide()

    def _build_spectrum_group(self):
        grp = QGroupBox("Spectrum")
        # The controls go in `body`, which is hidden wholesale while there is
        # no spectrometer — every one of them drives a device that isn't there.
        body = QWidget()
        lay = QVBoxLayout(body); lay.setSpacing(4)
        lay.setContentsMargins(0, 0, 0, 0)
        # The feed toggle lives in the spectrum panel's header (see _build_ui).
        self.lbl_integration = QLabel("Integration Time")
        lay.addWidget(self.lbl_integration)
        # Floating point, because the range is the DEVICE's: an Avantes goes
        # down to 2 us, so a whole-millisecond box would put most of its usable
        # exposure range out of reach. Range, decimals and step are all seeded
        # from the live device by _seed_integration_spin — an Ocean box that
        # cannot do sub-ms keeps its whole-millisecond look.
        self.spin_integration = DoubleSpinBox()
        self.spin_integration.setDecimals(0)
        self.spin_integration.setRange(1.0, MAX_UI_EXPOSURE_MS)
        self.spin_integration.setValue(10.0)
        self.spin_integration.setSuffix("  ms")
        # Debounced: set_integration_time is a device call, so it has to take
        # the hardware off the feed thread first. Doing that on every spinbox
        # tick would park and restart the feed on each keystroke, so coalesce
        # the edits and apply once the user stops typing.
        self._integration_timer = QTimer(self)
        self._integration_timer.setSingleShot(True)
        self._integration_timer.setInterval(250)
        self._integration_timer.timeout.connect(self._apply_integration_time)
        self.spin_integration.valueChanged.connect(
            lambda _v: self._integration_timer.start())
        # Second exposure, shown only for a stitched pair. The two devices see
        # very different signal levels, so one shared value always leaves one
        # of them either buried in read noise or clipped. S1/S2 are SLOT order
        # — the same numbering as the saturation lamps and the Spectrometer
        # window's two slots.
        #
        # Side by side under ONE caption rather than a second labelled row: a
        # stitched pair is the tallest the panel ever gets, and a caption that
        # reads "S1 / S2" over two boxes says the same thing in half the space.
        self.spin_integration2 = DoubleSpinBox()
        self.spin_integration2.setDecimals(0)
        self.spin_integration2.setRange(1.0, MAX_UI_EXPOSURE_MS)
        self.spin_integration2.setValue(10.0)
        self.spin_integration2.setSuffix("  ms")
        # Same debounce timer as S1 on purpose: editing both boxes then costs
        # ONE feed handover instead of two, and a handover blocks until the
        # in-flight acquire returns.
        self.spin_integration2.valueChanged.connect(
            lambda _v: self._integration_timer.start())
        irow = QHBoxLayout(); irow.setSpacing(6)
        irow.addWidget(self.spin_integration, 1)
        irow.addWidget(self.spin_integration2, 1)
        lay.addLayout(irow)
        self.spin_integration2.setVisible(False)

        lay.addWidget(_hline())
        # Record and use on one line: the checkbox is what you reach for right
        # after the button, and neither needs the panel's full width.
        drow = QHBoxLayout(); drow.setSpacing(8)
        self.btn_dark = QPushButton("Record Dark")
        self.btn_dark.setObjectName("accentcompact")
        self.btn_dark.clicked.connect(self._capture_dark)
        self.chk_dark = QCheckBox("Subtract Dark")
        self.chk_dark.setEnabled(False)
        drow.addWidget(self.btn_dark)
        drow.addWidget(self.chk_dark, 1)
        lay.addLayout(drow)
        # The alignment step lives in the Alignment dialog (toolbar).

        self.btn_connect_spec = QPushButton("Connect Spectrometer")
        self.btn_connect_spec.setObjectName("accent")
        self.btn_connect_spec.setToolTip(
            "Scan for attached spectrometers and connect one — or two, as a "
            "stitched pair")
        self.btn_connect_spec.clicked.connect(self.dlg_spec.open_fresh)
        self._connect_page(grp, body, self.btn_connect_spec)
        self._body_spectrum = body
        return grp

    def _build_stage_group(self):
        grp = QGroupBox("Stage")
        body = QWidget()
        lay = QVBoxLayout(body); lay.setSpacing(4)
        lay.setContentsMargins(0, 0, 0, 0)

        jog = QHBoxLayout()
        self.btn_minus = QPushButton("−"); self.btn_plus = QPushButton("+")
        self.btn_minus.clicked.connect(lambda: self._jog(-1))
        self.btn_plus.clicked.connect(lambda: self._jog(+1))
        self.spin_step = DoubleSpinBox()
        self.spin_step.setDecimals(0)
        self.spin_step.setSuffix(" fs")     # value set after ranges, below
        self.btn_units = QPushButton("fs")
        self.btn_units.setObjectName("unit")   # trimmed padding — see stylesheet
        self.btn_units.setFixedWidth(42)
        self.btn_units.setToolTip("Toggle jog/move units between optical delay (fs) "
                                  "and stage position (um)")
        self.btn_units.clicked.connect(self._toggle_stage_units)
        jog.addWidget(self.btn_minus); jog.addWidget(self.spin_step); jog.addWidget(self.btn_plus)
        jog.addWidget(self.btn_units)
        lay.addLayout(jog)

        mv = QHBoxLayout()
        self.spin_moveto = DoubleSpinBox()
        self.spin_moveto.setDecimals(0)
        self.spin_moveto.setSuffix(" fs")
        self.btn_moveto = QPushButton("Move")
        self.btn_moveto.clicked.connect(self._move_absolute)
        mv.addWidget(self.spin_moveto); mv.addWidget(self.btn_moveto)
        lay.addLayout(mv)

        self.btn_home = QPushButton("Home")
        self.btn_home.clicked.connect(self._home_stage)
        self.btn_goto_zero = QPushButton("Move to 0 fs")
        self.btn_goto_zero.clicked.connect(self._move_to_zero)
        zrow = QHBoxLayout()
        zrow.addWidget(self.btn_home); zrow.addWidget(self.btn_goto_zero)
        lay.addLayout(zrow)

        lay.addWidget(_hline())
        # Positions and stage state on one line, caption over value. They are
        # read together (a delay is the difference between the first two, and
        # neither means anything while the stage is still moving), and three
        # stacked full-width rows cost the panel three lines for six short
        # strings.
        def _readout(caption, value, stretch, align=Qt.AlignLeft):
            col = QVBoxLayout(); col.setSpacing(1)
            t = QLabel(caption); t.setObjectName("dim")
            t.setAlignment(align | Qt.AlignVCenter)
            value.setAlignment(align | Qt.AlignVCenter)
            col.addWidget(t); col.addWidget(value)
            prow.addLayout(col, stretch)

        prow = QHBoxLayout(); prow.setSpacing(8)
        self.lbl_pos  = QLabel("— um"); self.lbl_pos.setObjectName("readout_sm")
        self.lbl_zero = QLabel("— um"); self.lbl_zero.setObjectName("readout_sm")
        self.lbl_moving = QLabel("IDLE"); self.lbl_moving.setObjectName("idle")
        _readout("Current", self.lbl_pos, 1)
        _readout("Zero delay", self.lbl_zero, 1)
        # Centred, and bold in both states (see the #idle/#moving rules): it is
        # a status lamp in text, not a number to read off. No stretch either —
        # the two positions need every pixel at six significant figures.
        #
        # Which is exactly why it needs a floor: with stretch 0 it is the column
        # the row squeezes first, and the two readouts beside it reach eleven
        # monospace characters. Below the width of "MOVING" a centred QLabel
        # clips at BOTH edges, so the word lost its leading M and read "OVING".
        # _size_moving_label reserves that width; the layout takes the pixels
        # off the stretchy columns, which have them to give.
        _readout("Stage", self.lbl_moving, 0, Qt.AlignHCenter)
        self._size_moving_label()
        lay.addLayout(prow)

        self.btn_set_zero = QPushButton("Set Position as 0 fs")
        self.btn_set_zero.setObjectName("accentcompact")
        self.btn_set_zero.clicked.connect(self._mark_zero)
        lay.addWidget(self.btn_set_zero)

        # Backlash lives in the Stage dialog — it is a property of the
        # connected stage, and _sync_backlash_ui below still seeds it from here.

        # Ranges from whatever stage is loaded, which at build time is none:
        # _travel_range_um falls back to 300 mm of travel, and _apply_stage
        # calls this again with the real numbers on every connect.
        self._update_stage_unit_ranges()
        self.spin_step.setValue(10.0)      # after ranges: default 10 fs jog
        self._sync_backlash_ui()

        self.btn_connect_stage = QPushButton("Connect Stage")
        self.btn_connect_stage.setObjectName("accent")
        self.btn_connect_stage.setToolTip(
            "Scan for attached delay stages and connect one")
        self.btn_connect_stage.clicked.connect(self.dlg_stage.open_fresh)
        self._connect_page(grp, body, self.btn_connect_stage)
        self._body_stage = body
        return grp

    def _build_scan_group(self):
        grp = QGroupBox("FROG Scan")
        # Roomier than the two groups above it: this one is three short rows
        # rather than a stack, so the space bought by pairing them is better
        # spent making the pairs legible than left at the bottom of the panel.
        lay = QVBoxLayout(grp); lay.setSpacing(8)
        # Start beside Stop, then Step beside the background checkbox: the two
        # ends of the sweep are read as a pair, and three label+spin+equivalent
        # rows spent three lines on what fits in two.
        #
        # Four columns, twice over: [label][control]. Everything in column 3
        # therefore lines up under the Stop box — including the checkbox, which
        # used to start back at the label column and sat under nothing.
        g = QGridLayout()
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        g.setColumnStretch(1, 1); g.setColumnStretch(3, 1)

        def cell(r, c, label, spin, suffix, dec, lo, hi, val):
            g.addWidget(QLabel(label), r, c)
            spin.setRange(lo, hi); spin.setDecimals(dec)
            spin.setValue(val); spin.setSuffix(suffix)
            g.addWidget(spin, r, c + 1)

        self.spin_start = DoubleSpinBox()
        self.spin_stop  = DoubleSpinBox()
        self.spin_step_fs = DoubleSpinBox()
        cell(0, 0, "Start", self.spin_start, " fs", 1, -1e6, 1e6, -250.0)
        cell(0, 2, "Stop",  self.spin_stop,  " fs", 1, -1e6, 1e6,  250.0)
        # The um equivalents move under their own boxes — as a full-width pair
        # they would push the group back out to three rows.
        self.eq_start = QLabel(""); self.eq_start.setObjectName("dim")
        self.eq_stop  = QLabel(""); self.eq_stop.setObjectName("dim")
        g.addWidget(self.eq_start, 1, 1)
        g.addWidget(self.eq_stop,  1, 3)
        cell(2, 0, "Step", self.spin_step_fs, " fs", 1, 0.1, 1e5, 1.0)
        lay.addLayout(g)
        for s in (self.spin_start, self.spin_stop):
            s.valueChanged.connect(self._refresh_scan_um)

        # The scan and its export on one line, in the order they are used.
        # btn_scan keeps the wider share: it also carries "Abort Scan" while a
        # scan runs, and it is the button the whole panel exists for.
        arow = QHBoxLayout(); arow.setSpacing(8)
        self.btn_scan = QPushButton("Measure FROG")
        self.btn_scan.setObjectName("accent")
        self.btn_scan.setMinimumHeight(30)
        self.btn_scan.clicked.connect(self._start_scan)
        arow.addWidget(self.btn_scan, 3)

        self.btn_save = QPushButton(f"Save ({EXPORT_FORMATS[self._export_fmt][1]})")
        self.btn_save.setEnabled(False)
        self.btn_save.setMinimumHeight(30)
        # Straight to the write, in the format and under the name the Export
        # window is showing — this button is the shortcut past that window, not
        # a second set of choices.
        self.btn_save.clicked.connect(self._export_now)
        arow.addWidget(self.btn_save, 2)
        lay.addLayout(arow)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setFixedHeight(18)
        lay.addWidget(self.progress)

        # The AC FWHM is reported inside the autocorrelation panel itself
        # (FrogCanvas.set_fwhm), next to the curve it measures.

        self._refresh_scan_um()
        return grp

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._apply_theme("light" if self._theme == "dark" else "dark")

    def _apply_theme(self, name):
        self._theme = name
        install_theme(QApplication.instance(), name)   # shared with startup
        self.canvas.apply_palette(PALETTE)
        self.lamp.update()          # repaints from the new PALETTE
        self.btn_theme.setIcon(QIcon(str(MOON_ICON if name == "light" else SUN_ICON)))
        self.btn_theme.setToolTip(
            "Switch to dark mode" if name == "light" else "Switch to light mode")
        self._refresh_overlay_button()   # its icons are per-theme too
        self._refresh_layout_button()    # …and so are this one's
        self._refresh_feed_button()      # …and its glyph is drawn from PALETTE
        self._refresh_align_button()     # …and it has one file per theme
        self._refresh_align_refresh_icon()  # …as does its re-run button
        self._refresh_autofit_icon()     # …and the auto-fit mark
        self._refresh_autostitch_icon()  # …as does this one
        self._refresh_autoscale_icon()   # …and this one
        self._refresh_freeze_icon()      # …the + is drawn from PALETTE too
        self._refresh_reference_button() # …and the eye has one file per theme
        self._refresh_picture_icon()     # …as does the camera
        self._size_moving_label()        # its font came from the stylesheet
        self.pnl_no_spec.refresh_theme()  # …and its watermark is tinted live
        self.status.showMessage(f"{name.capitalize()} mode.", 2000)

    # ── Plot layout ──────────────────────────────────────────────────────────
    def _toggle_layout(self):
        mode = ("vertical" if self.canvas._layout_mode == "horizontal"
                else "horizontal")
        self._set_layout(mode)
        self.status.showMessage(
            "Horizontal layout — trace and autocorrelation share the delay axis."
            if mode == "horizontal" else
            "Vertical layout — autocorrelation full width below.", 4000)

    def _set_layout(self, mode):
        """Apply a layout mode and keep its button honest.

        Split out of _toggle_layout so a restored mode takes exactly the same
        path as a click — minus the status message, which at startup would push
        the 'no hardware connected' banner off the bar before it was read.
        """
        if mode not in ("horizontal", "vertical"):
            raise ValueError(mode)          # a bad settings key, caught upstream
        self.canvas.set_layout_mode(mode)
        self._refresh_layout_button()

    def _refresh_layout_button(self):
        """Icon and tooltip of the plot-layout toggle. Like the theme and
        overlay buttons, it advertises the layout a CLICK produces, so the
        vertical icon is shown while the horizontal layout is active."""
        horiz = self.canvas._layout_mode == "horizontal"
        icon = (VERT_ICON if horiz else HORIZ_ICON)[self._theme]
        if icon.exists():
            self.btn_layout.setIcon(QIcon(str(icon)))
            self.btn_layout.setText("")
        else:
            self.btn_layout.setIcon(QIcon())
            self.btn_layout.setText("▤" if horiz else "▥")
        self.btn_layout.setToolTip(
            "Vertical layout: autocorrelation full width below the spectrum "
            "and FROG trace" if horiz else
            "Horizontal layout: FROG trace above the autocorrelation, sharing "
            "one delay axis, beside a full-height spectrum")

    # ── Picture of the figure ────────────────────────────────────────────────
    def _refresh_picture_icon(self):
        """Per-theme camera on the picture button, with the same text fallback
        every other icon button carries. Only the icon is re-read here — unlike
        its two neighbours this button has one state, so its tooltip is fixed
        and set once."""
        icon = PICTURE_ICON[self._theme]
        if icon.exists():
            self.btn_picture.setIcon(QIcon(str(icon)))
            self.btn_picture.setText("")
        else:
            self.btn_picture.setIcon(QIcon())
            self.btn_picture.setText("PNG")
        self.btn_picture.setToolTip(
            "Save the plots as a high-resolution PNG — the FROG trace with the "
            "autocorrelation below it and the spectrum beside them, in the "
            "current theme. Opens on the Export window's file name.")

    def _save_picture(self):
        """Ask for a file and print the plots into it.

        Opens on the Export window's file name — same stem, same date prefix,
        only the suffix differs — so a scan and the picture of it land side by
        side in a folder listing under one name instead of needing the two to be
        matched up by timestamp afterwards.

        Deliberately NOT gated on self.result the way the data exports are: a
        picture of the live spectrum, or of a trace still filling in, is a
        perfectly good thing to want — there is always something on the panels
        worth a picture, which is not true of a scan file.
        """
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure as PNG", self.dlg_export.file_name(".png"),
            "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.status.showMessage(f"Saving figure → {path} …", 0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # repaint(), not the usual deferred update: the save below runs on this
        # thread and holds the event loop for as long as it takes, so a message
        # left for the next paint would first appear once it was already stale.
        self.status.repaint()
        try:
            w, h = self.canvas.save_picture(path)
        except Exception as e:
            # Broad: matplotlib raises OSError for a path it cannot write and
            # ValueError for a format it will not, and neither is worth a
            # traceback to a console this build does not have.
            self.status.showMessage(f"Figure save failed: {e}", 8000)
        else:
            # The size is reported because it is not simply PICTURE_DPI times a
            # fixed page: it follows the window, and above PICTURE_MAX_PX the
            # dpi gives way instead of the page growing.
            self.status.showMessage(f"Saved → {path}  ({w} × {h} px)", 5000)
        finally:
            QApplication.restoreOverrideCursor()

    # ── Saturation indicator ─────────────────────────────────────────────────
    def _set_lamp(self, state, text, obj, which=0):
        lamp = self.lamp if which == 0 else self.lamp2
        lbl  = self.lbl_sat if which == 0 else self.lbl_sat2
        lamp.set_state(state)
        lbl.setText(text)
        # Colour via objectName + repolish (not setStyleSheet) so a later theme
        # switch restyles the label along with everything else.
        if lbl.objectName() != obj:
            lbl.setObjectName(obj)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def _full_scale(self):
        """Effective detector full scale, in counts, or None if unknown.

        Mirrors FrogScanWorker.run exactly — a typed override beats the
        device's own report — so the lamp and the scan can never be judging
        against different thresholds.
        """
        full = self.scan_cfg.saturation_counts
        if full is None:
            full = getattr(self.spec, "max_counts", None)
        return full if full else None

    def _on_sat_fraction(self, percent):
        self.scan_cfg.saturation_fraction = percent / 100.0
        self._reset_saturation()    # re-judge; the old latch used a stale rule

    def _slot_members(self):
        """Live stitched members in SLOT order (slot 1 first). Falls back to
        the stitched device's own (blue, red) order when the pair was not
        connected through the multi-spectrometer slots."""
        members = getattr(self.spec, "members", None)
        if not members:
            return []
        if (all(m is not None for m in self._multi_members)
                and set(map(id, self._multi_members)) == set(map(id, members))):
            return list(self._multi_members)
        return list(members)

    def _sync_connection_ui(self):
        """Show the Connect buttons or the controls, per what is connected.

        The side panels and the "no spectrometer connected" message all say the
        same thing, so one function owns all three. Called from _sync_multi_ui
        (every spectrometer swap ends there) and from _apply_stage.
        """
        have_spec, have_stage = self._have_spec(), self._have_stage()
        self._body_spectrum.setVisible(have_spec)
        self.btn_connect_spec.setVisible(not have_spec)
        self._body_stage.setVisible(have_stage)
        self.btn_connect_stage.setVisible(not have_stage)
        self.pnl_no_spec.setVisible(not have_spec)
        if not have_spec:
            self.pnl_no_spec.raise_()
        # The header buttons act on a spectrum that may not exist; and the
        # message just claimed the panel, so the row has to be re-packed
        # around whatever is left showing.
        for b in (self.btn_autofit, self.btn_feed, self.btn_align_spec):
            b.setEnabled(have_spec)
        self._position_panel_buttons()
        if self.dlg_spec.isVisible():
            self.dlg_spec.refresh()
        if self.dlg_stage.isVisible():
            self.dlg_stage.refresh()
        if self.dlg_sim.isVisible():
            self.dlg_sim.refresh()

    def _sync_multi_ui(self):
        """Bring every stitched-vs-single widget into line with self.spec.

        One owner for the second integration spinbox and the overlay button,
        so no caller has to remember the set. Pure widget state — no device
        I/O, so it is safe to call outside _device_lock.
        """
        stitched = self._pair_live()
        slots = self._slot_members() if stitched else []
        # One caption over both boxes — S1 on the left, S2 on the right, each
        # in ITS OWN curve's colour. Two boxes under one caption is the compact
        # arrangement (see _build_spectrum_group), but it leaves the reader to
        # infer which box is which device from position alone; matching the
        # names to the curves on the panel says it outright. Rich text rather
        # than two labels, so the caption stays one widget on one line.
        #
        # MEMBER_COLORS is theme-invariant by design, so this needs no
        # _apply_theme hook — see the note beside line_m1/line_m2 in
        # FrogCanvas.apply_palette.
        self.lbl_integration.setText(
            (f"Integration Time — "
             f"<span style='color:{MEMBER_COLORS[0]}'><b>S1</b></span> / "
             f"<span style='color:{MEMBER_COLORS[1]}'><b>S2</b></span>")
            if stitched else "Integration Time")
        self.spin_integration2.setVisible(stitched)
        # Second saturation lamp: one alarm per device, so it exists exactly as
        # long as the pair does.
        self.lbl_sat2.setVisible(stitched)
        self.lamp2.setVisible(stitched)
        if len(slots) != 2:
            # No live pair: the per-spectrometer view has nothing to show.
            if self.btn_overlay.isChecked():
                with QSignalBlocker(self.btn_overlay):
                    self.btn_overlay.setChecked(False)
            self._overlay_on = False
            self.canvas.clear_members()
        elif not self._pair_seen:
            # A pair that has just come up starts UNSTITCHED — one curve per
            # spectrometer. The combined curve hides the one thing a fresh pair
            # most needs checking: whether the two halves actually agree across
            # the overlap. Seeing them apart is what makes a bad stitch factor
            # or a dark half obvious, and Auto-stitch is judged against it.
            #
            # Only on the way IN, hence the latch: this runs on every device
            # sync, and re-asserting the default there would override the
            # operator every time anything else about the pair changed.
            with QSignalBlocker(self.btn_overlay):
                self.btn_overlay.setChecked(True)
            self._overlay_on = True
        # Latch AFTER the branch above, and cleared whenever the pair goes, so
        # the next one to come up gets the default again.
        self._pair_seen = len(slots) == 2
        self._sync_integration_ui()
        self._refresh_overlay_button()
        self._refresh_autostitch_button()
        self._refresh_align_trace_button()
        self._refresh_avantes_button()
        self._sync_connection_ui()

    def _refresh_autostitch_icon(self):
        """Per-theme mark on the Auto-stitch button, with the same text
        fallback the other icon buttons keep for a bundle missing its icons."""
        icon = STITCH_ICON[self._theme]
        if icon.exists():
            self.btn_autostitch.setIcon(QIcon(str(icon)))
            self.btn_autostitch.setText("")
        else:
            self.btn_autostitch.setIcon(QIcon())
            self.btn_autostitch.setText("⇌")

    def _refresh_autostitch_button(self, running=None):
        """Show the Auto-stitch header button only while a pair is live, and
        enable it only while no scan owns the hardware it would drive.

        `running` overrides the inferred state for the same reason
        _refresh_symmetry_button takes it: _reset_scan_ui runs from the slot for
        finished_scan, which the worker emits INSIDE run() — before the finally
        that parks the stage. Whether the QThread has left run() by the time the
        GUI thread picks that signal up is a race, and on real hardware (where
        parking is a stage move measured in seconds) it reliably has not, so
        _scan_running() reads True and this re-greyed the button immediately
        after the scan that was over. Nothing refreshed it again, so it stayed
        dead until a device swap or a theme switch happened along.
        """
        if running is None:
            running = self._scan_running()
        self.btn_autostitch.setVisible(self._pair_live()
                                       and len(self._slot_members()) == 2)
        self.btn_autostitch.setEnabled(not running)
        self.btn_autostitch.setToolTip(
            "Auto-stitch — fit the factor that matches the two spectrometers "
            "across the overlap. Needs light across the overlap region; the "
            "fitted factor and its mismatch are shown in the Spectrometer "
            "window.")
        self._position_panel_buttons()   # its visibility drives the header row

    def _refresh_avantes_button(self):
        """Show the Avantes toolbar button only while an Avantes is live.

        Called from _sync_multi_ui because that is the one function guaranteed
        to run after every device swap (_apply_spectrometer, _connect_multi_pair
        and the slot handlers all end in it).
        """
        devices = self._avantes_devices()
        self.btn_avantes.setVisible(bool(devices))
        if not devices and self.dlg_avantes.isVisible():
            self.dlg_avantes.hide()      # its target just went away
        elif devices and self.dlg_avantes.isVisible():
            self.dlg_avantes.refresh()  # a swap may have changed which device

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

    def _sync_integration_ui(self):
        """Reseed the integration spinbox(es) from the live device(s).

        Needed whenever the device behind a box changes, and whenever
        something OTHER than the box changes the exposure it should be
        showing. A live pair takes one box per member, from that member's own
        limits — the two halves of a mixed-vendor pair do not share a range.
        """
        if not self._have_spec():
            return          # the boxes are hidden with the rest of the panel
        slots = self._slot_members() if self._pair_live() else []
        if len(slots) == 2:
            for mem, spin in zip(slots, (self.spin_integration,
                                         self.spin_integration2)):
                self._seed_integration_spin(spin, mem)
        else:
            self._seed_integration_spin(self.spin_integration, self.spec)

    def _refresh_overlay_button(self):
        """Icon, tooltip and availability of the per-spectrometer view toggle.

        The icon shows the view a CLICK produces, not the current one: broken
        spectrum while the combined curve is up, continuous once the members
        are drawn apart.
        """
        stitched = self._pair_live()
        self.btn_overlay.setVisible(stitched and len(self._slot_members()) == 2)
        # Live through a scan, unlike Auto-stitch beside it: this only chooses
        # which curves the panel draws, and _render_scan_frame now honours the
        # choice from the scan's own member frames. Auto-stitch has to stay
        # disabled — it drives the hardware the worker owns.
        on = self.btn_overlay.isChecked()
        icon = (MERGE_ICON if on else SPLIT_ICON)[self._theme]
        if icon.exists():
            self.btn_overlay.setIcon(QIcon(str(icon)))
            self.btn_overlay.setText("")
        else:
            self.btn_overlay.setIcon(QIcon())
            self.btn_overlay.setText("∿" if on else "⌇")
        self.btn_overlay.setToolTip(
            "Show the combined stitched spectrum as one curve" if on else
            "Show each spectrometer as its own curve — compare the two across "
            "the overlap to judge the stitch factor")
        # Its visibility just changed, and the header row is packed
        # right-to-left around whatever is showing.
        self._position_panel_buttons()

    def _on_overlay_toggled(self, on):
        if on and not (self._pair_live()
                       and len(self._slot_members()) == 2):
            with QSignalBlocker(self.btn_overlay):
                self.btn_overlay.setChecked(False)
            return
        self._overlay_on = bool(on)
        if not on:
            # Don't wait for the next frame to put the combined curve back.
            self.canvas.clear_members()
        self._refresh_overlay_button()
        self._refresh_autostitch_button()
        self.status.showMessage(
            "Spectrum panel: one curve per spectrometer." if on else
            "Spectrum panel: combined stitched spectrum.", 4000)

    def _member_full_scale(self, member):
        """A member's effective full scale: the Hardware override, when set,
        applies to every member; otherwise the member's own report."""
        full = self.scan_cfg.saturation_counts
        if full is None:
            full = getattr(member, "max_counts", None)
        return full if full else None

    def _reset_saturation(self):
        """Clear a latched warning and re-read the effective full scale(s)."""
        self._sat_latched = False
        self._sat_frames  = 0            # saturated frames so far this scan
        self._sat_worst   = (0, 0.0, 0)  # (n_pixels, delay_fs, column index)
        slots = self._slot_members() if self._pair_live() else []
        if slots:
            # Live pair: one lamp per member, each judged against its own full
            # scale.
            for i, mem in enumerate(slots):
                full = self._member_full_scale(mem)
                lamp = self.lamp if i == 0 else self.lamp2
                if full:
                    src = ("override" if self.scan_cfg.saturation_counts
                           else "device")
                    lamp.setToolTip(
                        f"{mem.name} headroom — full scale {full:.0f} counts "
                        f"({src}), saturated at "
                        f"{100 * self.scan_cfg.saturation_fraction:.0f}%")
                    self._set_lamp("ok", f"S{i + 1}", "satok", i)
                else:
                    lamp.setToolTip(
                        f"{mem.name} does not report a full-scale value, so "
                        f"saturation cannot be checked. Set one in "
                        f"Spectrometer → Full scale.")
                    self._set_lamp("unknown", f"S{i + 1} — % FS", "satok", i)
            return
        if not self._have_spec():
            self.lamp.setToolTip("No spectrometer connected.")
            self._set_lamp("unknown", "", "satok")
            return
        full = self._full_scale()
        if not full:
            self.lamp.setToolTip(
                "This spectrometer does not report a full-scale value, so "
                "saturation cannot be checked. Set one in Spectrometer → "
                "Full scale.")
            self._set_lamp("unknown", "— % FS", "satok")
        else:
            source = ("override" if self.scan_cfg.saturation_counts else "device")
            self.lamp.setToolTip(
                f"Detector headroom — full scale {full:.0f} counts ({source}), "
                f"saturated at {100 * self.scan_cfg.saturation_fraction:.0f}%")
            self._set_lamp("ok", "", "satok")

    def _judge_lamp(self, which, raw, full, prefix=""):
        """Drive one lamp from one RAW frame and its full scale.

        Same threshold the scan worker uses, so the live lamp and the scan
        warning can never disagree about what counts as saturated.
        """
        if not full:
            self._set_lamp("unknown", f"{prefix}— % FS", "satok", which)
            return
        peak = float(np.max(raw)) if raw.size else 0.0
        frac = peak / full
        sat_frac = self.scan_cfg.saturation_fraction
        if frac >= sat_frac:
            n = int(np.count_nonzero(raw >= sat_frac * full))
            self._set_lamp("sat", f"{prefix}⚠ SATURATED  ({n} px)", "sat", which)
        elif frac >= SAT_WARN_FRACTION:
            self._set_lamp("warn", f"{prefix}⚠ {100 * frac:.0f}% FS",
                           "satwarn", which)
        else:
            self._set_lamp("ok", f"{prefix}{100 * frac:.0f}% FS", "satok", which)

    def _update_saturation(self, raw):
        """Live headroom readout, driven by the feed's RAW (un-subtracted) frame.

        Skipped while a scan's warning is latched: that one records that the
        measurement is already compromised, and must not be scrolled away by
        whatever the feed sees once the scan hands the hardware back.
        """
        if self._sat_latched:
            return
        slots = self._slot_members() if self._pair_live() else []
        frames = getattr(self.spec, "last_member_raw", None)
        if slots and frames is not None:
            # Two separate alarms: each member's own raw frame against its own
            # full scale. `raw` (the combined stitched frame) has no single
            # ADC scale and is deliberately not judged.
            members = self.spec.members
            raw_by_id = {id(m): f for m, f in zip(members, frames)}
            for i, mem in enumerate(slots):
                frame = raw_by_id.get(id(mem))
                if frame is not None:
                    self._judge_lamp(i, frame, self._member_full_scale(mem),
                                     prefix=f"S{i + 1} ")
            return
        full = self._full_scale()
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
        self.status.showMessage(f"Live feed: {msg}", 6000)

    def _on_spectrum(self, wl, raw):
        """Slot for LiveFeedWorker.spectrum_ready — runs on the GUI thread.

        Deliberately O(1): it only records the newest frame for _display_tick
        to render. Doing the rendering here let the event queue grow without
        bound whenever a render outlasted the acquisition interval (weak
        machines), and the display lagged further behind every second.
        """
        self._live_frame = (wl, raw)
        # Keep the worst (highest-peak) frame between ticks so a transiently
        # clipped frame that never gets displayed still trips the lamp.
        p = float(raw.max()) if raw.size else 0.0
        if p > self._sat_peak:
            self._sat_peak = p
            self._sat_frame = raw
        # NB: no stage read here. Polling get_position() at feed rate raced the
        # feed thread's own acquire() (which, for the simulator, itself reads
        # the stage). The readout is instead refreshed after every move we make
        # and from the scan worker's per-column read-back.

    def _display_tick(self):
        """Render the newest pending data — scan column or live frame."""
        if self._scan_dirty:
            with perf_tick("scan_render"):
                self._render_scan_frame()
            return
        if self._live_frame is None:
            return
        if not self._have_spec():
            # The feed emits from its own thread, so a spectrum_ready queued
            # before the disconnect is still delivered after it. Drop it: the
            # frame describes a device that is gone, and every correction
            # below it would go through self.spec.
            self._live_frame = None
            return
        with perf_tick("live_render"):
            wl, raw = self._live_frame
            self._live_frame = None
            self.last_spectrum = raw
            self._update_saturation(self._sat_frame if self._sat_frame is not None
                                    else raw)
            self._sat_frame = None
            self._sat_peak = -1.0
            if self._overlay_on and self._render_overlay():
                self._flush_pending_fit()
                return
            self.canvas.update_spectrum(wl, self._dark_corrected(raw))
            self._flush_pending_fit()

    def _dark_corrected(self, raw):
        """The merged live frame with the dark removed, ready to plot."""
        return self._corrected_frame(
            raw, getattr(self.spec, "last_member_raw", None))

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
        if not self.chk_dark.isChecked():
            return raw if self._pair_live() \
                else self.spec.calibrate(raw)
        if self._pair_live():
            darks = self._usable_member_darks()
            if darks is None or frames is None:
                self._warn_dark_mismatch()
                return raw          # already calibrated and merged by acquire()
            return self.spec.combine(
                *self.spec.prepare_pair(*frames, darks))
        spectrum = raw
        if self.background is not None:
            spectrum = np.clip(raw - self.background, 0, None)
        # Calibration LAST, and never on what the lamp judged: saturation is a
        # raw-ADC property, the calibration is display/data physics.
        return self.spec.calibrate(spectrum)

    def _display_dark(self):
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
        if not self.chk_dark.isChecked():
            return None
        if self._pair_live():
            darks = self.background_members
            if darks is None:
                return None
            try:
                return self.spec.combined_dark(darks)
            except Exception:
                return None        # mismatched shapes: better raw than wrong
        if self.background is None:
            return None
        return self.spec.calibrate(self.background)

    def _warn_dark_mismatch(self):
        """Say once that a recorded dark does not fit the live pair.

        One shot only — the callers run every 60 ms and would bury the status
        bar. Re-armed whenever a dark is recorded or dropped.
        """
        if self._dark_member_warned:
            return
        self._dark_member_warned = True
        self.status.showMessage(
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
        if self._pending_fit:
            self._pending_fit = False
            self._autofit_spectrum()
        elif self._pending_rescale:
            # elif: a device swap's full fit is the stronger action and already
            # covers the rescale, so it must not be undone by one queued behind it.
            self._pending_rescale = False
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

        Reads self.spec.last_member_raw from the GUI thread without the device
        lock, exactly as _update_saturation does: the tuple assignment is
        atomic, and calibrate() plus the wavelength arrays are pure.
        """
        spec = self.spec
        if frames is None:
            frames = getattr(spec, "last_member_raw", None)
        slots = self._slot_members()
        if frames is None or len(slots) != 2:
            return False
        darks = None
        if self.chk_dark.isChecked():
            darks = self._usable_member_darks()
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
            curves.append((np.asarray(mem.wavelengths, float),
                           y * spec.member_scale(idx)))
        # Cheap and idempotent — set_overlap_band returns immediately unless
        # the band actually moved.
        self.canvas.set_overlap_band(*spec.overlap_band)
        self.canvas.update_member_spectra(curves[0][0], curves[0][1],
                                          curves[1][0], curves[1][1])
        return True

    def _autofit_spectrum(self):
        """One-shot fit of spectrum X + Y; syncs Graphics Settings spinboxes."""
        self.canvas.fit_xy()
        if hasattr(self, 'dlg_graphics'):
            self.dlg_graphics.sync_limits()

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
        c = self.canvas
        was_x, was_y = c.autoscale_x, c.autoscale_y
        c.fit_xy()                     # fits both axes, freezes both flags
        c.autoscale_x = was_x
        if was_y:
            c.set_autoscale_y(True)
        if hasattr(self, 'dlg_graphics'):
            self.dlg_graphics.sync_limits()

    # ── Alignment mode: spectrum symmetry ────────────────────────────────────
    def _align_running(self):
        return (self._align_worker is not None
                and self._align_worker.isRunning())

    def _uncheck_align_spec(self):
        """Pop the Δ button back out without re-entering its own handler."""
        self.btn_align_spec.blockSignals(True)
        self.btn_align_spec.setChecked(False)
        self.btn_align_spec.blockSignals(False)

    def _close_align_mode(self):
        """Leave alignment mode: panel closed, curves dropped, button out.

        The one way out, so every caller that has to abandon the mode — the
        toggle, a device swap, the start of a scan — leaves the same state
        behind. Silent about signals: _uncheck_align_spec blocks them, so this
        never re-enters the toggle handler that may have called it.
        """
        self._uncheck_align_spec()
        self.btn_align_refresh.hide()
        self.canvas.clear_diff()
        self.canvas.set_dev_visible(False)
        self._position_panel_buttons()   # the header row just lost a button

    def _on_align_spec_toggled(self, on):
        """Open or close alignment mode.

        Opening it puts the deviation panel on screen and runs the first sweep;
        the ⟳ button then re-runs it in place, so judging an adjustment no
        longer means toggling the mode off and on (which threw away the
        measurement you wanted to compare against).
        """
        if not on:
            self._close_align_mode()
            self.status.showMessage("Alignment mode off.", 2500)
            return
        if not self._start_align_sweep():
            self._uncheck_align_spec()
            return
        self.canvas.set_dev_visible(True)
        self.btn_align_refresh.show()
        self._position_panel_buttons()

    def _start_align_sweep(self):
        """Run one alignment sweep. False if it could not be started — the
        caller that was also opening the mode then does not open it.

        Every refusal below leaves the hardware exactly as it found it; the
        feed, which has to be handed over before the worker can drive the
        devices, is the only thing with any unwinding to do.
        """
        if self._align_running():
            return False
        if self._scan_running():
            self.status.showMessage("A scan is running — the stage is busy.", 3000)
            return False
        if self.spec is None or self.stage is None:
            self.status.showMessage("Connect a stage and a spectrometer first.", 4000)
            return False
        if getattr(self.stage, "needs_homing", False):
            self.status.showMessage(
                "Stage is not homed — press Home before an alignment sweep.", 6000)
            return False

        # Same handover contract as _start_scan: the worker drives the stage
        # and the spectrometer for the whole sweep, so the feed has to let go
        # of them first, and refusing beats running against a device the feed
        # thread is still inside.
        self._feed_was_on = self.btn_feed.isChecked()
        if not self._feed.pause():
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(f"Alignment not started — {FEED_BUSY_MSG}", 6000)
            return False

        # Only now is it safe to read the stage from this thread — the feed is
        # parked, so nothing else is talking to it. Checked BEFORE moving:
        # clamping a target would quietly destroy the +/- symmetry the whole
        # measurement is about, so an out-of-range sweep is refused instead.
        step = self.dlg_align.step_fs
        pf   = self.scan_cfg.pass_factor
        try:
            start_um = _stage_to_um(self.stage.get_position())
        except Exception as e:
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(f"Alignment failed — stage: {e}", 5000)
            return False
        span_um = abs(float(delay_to_position_um(2.0 * step, 0.0, pf)))
        lo, hi = self._travel_range_um()
        if start_um - span_um < lo or start_um + span_um > hi:
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(
                f"±2×{step:.0f} fs (±{span_um:.1f} um) from here leaves the "
                f"travel range [{lo:.1f}, {hi:.1f}] um — move away from the "
                f"limit or reduce the Alignment Step.", 7000)
            return False

        self._set_hardware_buttons_enabled(False)
        self.btn_scan.setEnabled(False)
        self._set_stage_controls_enabled(False)
        self.btn_align_spec.setEnabled(False)
        self.btn_align_refresh.setEnabled(False)
        self._moving(True)

        self._align_worker = AlignmentWorker(self.stage, self.spec,
                                             self.scan_cfg, step)
        self._align_worker.progress.connect(self._on_align_progress)
        self._align_worker.done.connect(self._on_align_done)
        self._align_worker.error.connect(self._on_align_error)
        self._align_worker.start()
        self.status.showMessage(
            f"Alignment sweep: −{2 * step:.0f}, −{step:.0f}, "
            f"+{step:.0f}, +{2 * step:.0f} fs…", 0)
        return True

    def _on_align_progress(self, done, total):
        self.status.showMessage(f"Alignment sweep: point {done}/{total}…", 0)

    def _on_align_done(self, results):
        """Slot for AlignmentWorker.done — results in offset order."""
        self._reset_align_ui()
        try:
            # Corrected through exactly the path the spectrum panel uses, each
            # frame with ITS OWN member frames, so the differences sit on the
            # displayed baseline rather than a second derivation of it.
            s = [np.asarray(self._corrected_frame(raw, frames), float)
                 for raw, frames in results]
            wl = np.asarray(self.spec.wavelengths, float)
            d1 = s[2] - s[1]        # S(+x)  - S(-x)
            d2 = s[3] - s[0]        # S(+2x) - S(-2x)
        except Exception as e:
            # The MODE stays open: the panel is still the right place to look,
            # the previous sweep (if any) is still on it, and ⟳ is how you try
            # again. Closing it here would throw away the comparison the
            # operator was in the middle of making.
            self.status.showMessage(f"Alignment differences failed: {e}", 5000)
            return
        self.canvas.show_diff(wl, d1, d2)
        step = self.dlg_align.step_fs
        # Peak absolute imbalance as a fraction of the largest measured signal:
        # the one number that says "symmetric" or "not" without reading the
        # curves off the panel.
        scale = max(float(max(np.max(np.abs(x)) for x in s)), 1e-12)
        r1 = float(np.max(np.abs(d1))) / scale
        r2 = float(np.max(np.abs(d2))) / scale
        self.status.showMessage(
            f"Alignment: peak asymmetry {100 * r1:.1f}% at ±{step:.0f} fs, "
            f"{100 * r2:.1f}% at ±{2 * step:.0f} fs.", 8000)

    def _on_align_error(self, msg):
        # Mode left open, for the reason in _on_align_done: a sweep that failed
        # is a reason to fix something and press ⟳, not to lose the panel.
        self._reset_align_ui()
        self.status.showMessage(f"Alignment sweep failed: {msg}", 6000)

    def _reset_align_ui(self):
        """Hand the devices back and re-enable what the sweep locked out."""
        self._moving(False)
        self._set_hardware_buttons_enabled(True)
        self.btn_scan.setEnabled(True)
        self._set_stage_controls_enabled(True)
        self.btn_align_spec.setEnabled(True)
        self.btn_align_refresh.setEnabled(True)
        self._refresh_positions()
        if self._feed_was_on:
            self._feed.resume()

    # ── Alignment mode: raw stitched trace ───────────────────────────────────
    def _position_panel_buttons(self):
        """Lay out each panel's header row: the buttons that act on that panel,
        on the line the panel title occupies.

        Two groups per panel. Most buttons are right-aligned to the panel's
        right edge. The two that describe the Y AXIS — the auto-scale padlock
        and the log/linear toggle — are packed immediately AFTER the title
        instead, so they read as part of that panel's name rather than as two
        more entries in a row of six; the clear span between the two groups is
        what separates "what this panel's y axis is doing" from "things you can
        do to this panel".

        Every edge here moves with the split fraction, the window size and the
        layout mode, so this runs off canvas.axes_relaid — and off the refresh
        methods that show or hide a button, because each row is packed from its
        own edge and its width depends on which buttons are visible.
        """
        # getattr, not the attribute: the refresh methods that call this also
        # run from _apply_spectrometer, which the connect paths can reach
        # before _build_ui has created any of these buttons.
        if getattr(self, "pnl_no_spec", None) is None:
            return
        c = self.canvas

        def _row_y(top):
            # The band sits ABOVE the axes: HDR_PAD of clear space between the
            # buttons' bottom edge and the top spine.
            return round(top) - HDR_PAD - HDR_BTN

        def pack(ax, buttons):
            """Right-align `buttons` to the panel's right edge. Returns the x of
            the leftmost one placed, which is where the clear span before it
            begins."""
            _left, top, right = c.panel_edges_px(ax)
            y = _row_y(top)
            x = round(right) - 6
            for b in reversed(buttons):
                # isHidden, not isVisible: isVisible() is False for every child
                # until the top-level window itself is shown, so packing on it
                # skipped the whole row on the layout passes that run during
                # __init__ and left the buttons piled at (0, 0) until something
                # else happened to trigger a relayout.
                if b.isHidden():
                    continue      # hidden buttons take no room in the row
                x -= b.width()
                b.move(x, y)
                x -= HDR_GAP
            return x + HDR_GAP        # undo the trailing gap

        def pack_after_title(ax, buttons, limit):
            """Lay `buttons` out left to right starting just past the title.

            `limit` is the x the group must not reach — the leftmost button of
            the right-hand group. On a narrow panel the whole group slides left
            to respect it, over the title if it has to: a control half off the
            panel edge, or overlapping a row it does not belong to, is worse
            than one sitting on its own title for the few pixels it takes.
            """
            _left, top, _right = c.panel_edges_px(ax)
            y = _row_y(top)
            start = round(c.title_right_px(ax)) + HDR_TITLE_GAP
            shown = [b for b in buttons if not b.isHidden()]
            if not shown:
                return
            width = sum(b.width() for b in shown) + HDR_GAP * (len(shown) - 1)
            start = min(start, round(limit) - HDR_GROUP_GAP - width)
            x = max(start, 0)
            for b in shown:
                b.move(x, y)
                x += b.width() + HDR_GAP

        spec_right = pack(
            c.ax_spec, [self.btn_autofit, self.btn_feed, self.btn_freeze,
                        self.btn_refshow, self.btn_autostitch,
                        self.btn_overlay, self.btn_align_spec,
                        self.btn_align_refresh])
        pack(c.ax_trace, [self.btn_align_trace, self.btn_symmetry])
        # The y-axis pair, tucked against the title it qualifies. Padlock first:
        # it is the one that changes what the axis DOES, the scale toggle only
        # how it is drawn.
        pack_after_title(c.ax_spec, [self.btn_autoscale, self.btn_logscale],
                         spec_right)

        # The "no spectrometer connected" panel, stretched over the whole
        # spectrum axes so it reads as the empty plot itself rather than a card
        # floating in it. It is laid out only here — it has no parent layout to
        # do that for it.
        self.pnl_no_spec.setGeometry(c.panel_rect_px(c.ax_spec))

    def _refresh_autofit_icon(self):
        """Per-theme mark on the one-shot auto-fit button, with the text
        fallback it used to carry if the icons are missing from the bundle."""
        icon = RESCALE_ICON[self._theme]
        if icon.exists():
            self.btn_autofit.setIcon(QIcon(str(icon)))
            self.btn_autofit.setText("")
        else:
            self.btn_autofit.setIcon(QIcon())
            self.btn_autofit.setText("↔↕")

    def _refresh_autoscale_icon(self):
        """Padlock matching the axis' CURRENT state — open while it follows the
        data, shut while it is frozen. Per theme, and re-read on every state
        change, so this is called from _refresh_autoscale_button rather than
        only from the theme switch."""
        on = bool(self.canvas.autoscale_y)
        icon = (UNLOCKED_ICON if on else LOCKED_ICON)[self._theme]
        if icon.exists():
            self.btn_autoscale.setIcon(QIcon(str(icon)))
            self.btn_autoscale.setText("")
        else:
            self.btn_autoscale.setIcon(QIcon())
            self.btn_autoscale.setText("🔓" if on else "🔒")

    def _refresh_freeze_icon(self):
        """The + mark, drawn from the live PALETTE so a theme switch reruns it
        — same arrangement as the feed button's play/stop glyph."""
        self.btn_freeze.setIcon(_glyph_icon("plus", PALETTE["accent"]))

    def _freeze_reference(self):
        """Capture the spectrum panel as it stands.

        Replaces any reference already held rather than refusing: this button is
        reached for after every adjustment, and making that "drop, then freeze"
        would be two presses for the one thing anyone ever wants.
        """
        if not self.canvas.freeze_reference():
            self.status.showMessage(
                "Nothing on the spectrum panel to freeze yet.", 4000)
            return
        self.status.showMessage(
            "Spectrum frozen as a grey reference — press + again to replace "
            "it, the eye to hide it, or right-click + to drop it.", 6000)

    def _toggle_reference(self):
        self.canvas.set_reference_visible(not self.canvas.reference_shown())

    def _clear_reference(self):
        """Right-click on +. Silent when there is nothing held: a right-click
        that reports having done nothing is just noise."""
        if self.canvas.has_reference():
            self.canvas.clear_reference()
            self.status.showMessage("Reference spectrum dropped.", 3000)

    def _refresh_reference_button(self):
        """Visibility, icon and tooltip of the reference's hide/show toggle.

        The icon shows the ACTION its two files were drawn for (see SHOW_ICON):
        a reference on screen offers the struck-through eye, a hidden one the
        open eye.
        """
        c = self.canvas
        have = c.has_reference()
        self.btn_refshow.setVisible(have)
        if have:
            shown = c.reference_shown()
            icon = (HIDE_ICON if shown else SHOW_ICON)[self._theme]
            if icon.exists():
                self.btn_refshow.setIcon(QIcon(str(icon)))
                self.btn_refshow.setText("")
            else:
                self.btn_refshow.setIcon(QIcon())
                self.btn_refshow.setText("◉" if shown else "○")
            self.btn_refshow.setToolTip(
                "Hide the frozen reference spectrum" if shown else
                "Show the frozen reference spectrum")
        # Its visibility just changed, and the header row is packed
        # right-to-left around whatever is showing.
        self._position_panel_buttons()

    def _refresh_logscale_button(self):
        """Label and tooltip of the log/linear toggle. Reads the checkbox, not
        the axes: chk_log is the single source of truth for the scale."""
        on = self.dlg_graphics.log_scale
        self.btn_logscale.setText("log" if on else "lin")
        self.btn_logscale.setToolTip(
            "Spectrum Y axis is LOGARITHMIC — click for linear." if on else
            "Spectrum Y axis is LINEAR — click for logarithmic.")

    def _on_autoscale_clicked(self):
        """Toggle the spectrum panel's auto-scale.

        Routed through the canvas' own reset_axes rather than set_autoscale_y:
        that is exactly what a right-click on the panel does, so the button and
        the right-click cannot end up meaning different things. It emits
        limits_changed on the way out, which is what puts the button's own
        checked state back in agreement with the canvas — including when the
        canvas refuses to move.
        """
        self.canvas.reset_axes(self.canvas.ax_spec)

    def _refresh_autoscale_button(self):
        """Show the canvas' auto-scale state on the header toggle.

        Signals blocked: this is called FROM limits_changed, which reset_axes
        emits, so letting the setChecked re-enter the click handler would
        toggle the mode straight back off again.
        """
        on = bool(self.canvas.autoscale_y)
        with QSignalBlocker(self.btn_autoscale):
            self.btn_autoscale.setChecked(on)
        self._refresh_autoscale_icon()   # the padlock IS the state readout
        self.btn_autoscale.setToolTip(
            ("Y axis UNLOCKED — the spectrum's Y axis follows the live signal.\n"
             "Click to lock it where it is."
             if on else
             "Y axis LOCKED — the spectrum's Y axis stays where it is.\n"
             "Click to unlock it: fit the data, then follow it.")
            + "\n" + RIGHT_CLICK_HINT)

    def _refresh_align_refresh_icon(self):
        """Per-theme mark on the re-run button, with the usual text fallback."""
        icon = REFRESH_ICON[self._theme]
        if icon.exists():
            self.btn_align_refresh.setIcon(QIcon(str(icon)))
            self.btn_align_refresh.setText("")
        else:
            self.btn_align_refresh.setIcon(QIcon())
            self.btn_align_refresh.setText("⟳")

    def _refresh_align_button(self):
        """Per-theme crosshair on the alignment-sweep toggle.

        Unlike the overlay and layout buttons this one depicts a MODE, so the
        mark does not change with the button's state — only with the theme,
        each file carrying that theme's accent. Falls back to the Δ it used to
        show if the icons are missing from the bundle.
        """
        icon = ALIGN_ICON[self._theme]
        if icon.exists():
            self.btn_align_spec.setIcon(QIcon(str(icon)))
            self.btn_align_spec.setText("")
        else:
            self.btn_align_spec.setIcon(QIcon())
            self.btn_align_spec.setText("Δ")

    def _refresh_feed_button(self):
        """Icon, colour and tooltip of the live-feed toggle.

        Like the theme and layout buttons it advertises the CURRENT state, not
        the state a click produces: a running feed is the red stop square, a
        stopped one the accent play triangle. Both glyphs are drawn from the
        live PALETTE, so a theme switch re-generates them.
        """
        running = self.btn_feed.isChecked()
        self.btn_feed.setIcon(_glyph_icon(
            "stop" if running else "play",
            PALETTE["danger"] if running else PALETTE["accent"]))
        self.btn_feed.setToolTip("Stop the live spectrum feed" if running
                                 else "Start the live spectrum feed")
        # Colour via objectName + repolish, not setStyleSheet, so a later theme
        # switch restyles it along with everything else.
        name = "overlaydanger" if running else "overlay"
        if self.btn_feed.objectName() != name:
            self.btn_feed.setObjectName(name)
            self.btn_feed.style().unpolish(self.btn_feed)
            self.btn_feed.style().polish(self.btn_feed)

    def _refresh_align_trace_button(self):
        """Show the RAW toggle only for a stitched pair, and only enable it
        once there is a scan whose raw member columns were recorded."""
        stitched = self._pair_live()
        self.btn_align_trace.setVisible(stitched)
        self._position_panel_buttons()   # visibility drives the header row
        if not stitched:
            return
        ready = self._align_trace is not None
        self.btn_align_trace.setEnabled(ready)
        self.btn_align_trace.setToolTip(
            "Raw counts — no calibration, no dark subtraction. Each "
            "spectrometer is normalised to its own maximum and the two are "
            "cut (not blended) at the middle of the overlap, so features "
            "buried in the background show up.\nDisplay only: the recorded "
            "trace and every export are unaffected."
            if ready else
            "Run a scan first — the raw per-spectrometer columns are recorded "
            "as it goes.")

    def _on_align_trace_toggled(self, on):
        if on and self._align_trace is None:
            self.btn_align_trace.blockSignals(True)
            self.btn_align_trace.setChecked(False)
            self.btn_align_trace.blockSignals(False)
            self.status.showMessage(
                "No raw trace yet — run a scan in multi-spectrometer mode.", 4000)
            return
        self._align_trace_on = bool(on)
        if self._align_trace_on and self.btn_symmetry.isChecked():
            self._clear_symmetry()      # one trace-view mode at a time
        # Re-render at once rather than waiting for the next column, so the
        # toggle also works on a finished trace.
        if self._align_trace_on:
            self.canvas.update_trace(self._align_trace_view(), 1.0)
        elif self._scan_trace is not None:
            self.canvas.update_trace(self._scan_trace, self._scan_peak)
        self.status.showMessage(
            "FROG trace: raw counts, per-spectrometer normalised." if on else
            "FROG trace: calibrated, stitched.", 4000)

    # ── Symmetry mode ────────────────────────────────────────────────────────
    def _symmetry_center(self, ac, delays):
        """Column index the trace is folded about.

        The autocorrelation maximum by default: the AC is the trace summed over
        wavelength, so its peak is where the two pulses overlap best — the delay
        the trace SHOULD be symmetric about. Not the middle column, which would
        make a scan range that was not centred on the overlap report its own
        offset as asymmetry.

        A hand-placed line (dragged on the plot) overrides it, snapped to the
        nearest measured column: the fold is an index into the trace, so there
        is no sub-column position it could honour without resampling.
        """
        if self._fold_fs is None:
            return int(np.nanargmax(np.asarray(ac, float)))
        d = np.asarray(delays, float)
        return int(np.argmin(np.abs(d - float(self._fold_fs))))

    def _symmetry_view(self, trace, ac, delays):
        """Measured trace on the left of the fold, |T(+d) - T(-d)| on the right.

        Returns (image, peak, score, fold_delay_fs).

        Half and half rather than the difference everywhere: keeping one side as
        it was measured gives the eye something to judge the other against, and
        the two are directly comparable because they share one colour scale —
        taken from the MEASURED peak, so the left half looks exactly as it does
        in the ordinary view and "nearly black on the right" genuinely means
        "nearly symmetric" rather than "rescaled until it looks that way".

        ABSOLUTE difference. The signed one is antisymmetric, so folding it onto
        a single side would put half the pixels below zero, and every colormap
        here is sequential with clim (0, peak) — those would flatten onto the
        floor and hide exactly what the mode exists to show.

        Columns past the fold with no counterpart behind it — the tail of the
        longer half, whenever the fold sits off-centre — are left NaN rather
        than zeroed. Zero would claim perfect symmetry where there is simply
        nothing to compare against; NaN takes _apply_cmap's 'bad' colour, the
        plot background, so unmeasured reads as unmeasured.
        """
        t = np.asarray(trace, float)
        n = t.shape[1]
        c = self._symmetry_center(ac, delays)
        fold = float(np.asarray(delays, float)[c])
        out = t.copy()                   # left of the fold: measured, untouched
        h = min(c, n - 1 - c)            # columns both sides can cover
        peak = float(np.nanmax(t)) if t.size else 0.0
        if h < 1:
            out[:, c + 1:] = np.nan      # fold on an edge: nothing to compare
            return out, peak, 0.0, fold
        left  = t[:, c - h:c + 1]        # columns c-h … c
        right = t[:, c:c + h + 1]        # columns c   … c+h
        d = np.abs(right - left[:, ::-1])
        out[:, c:c + h + 1] = d
        out[:, c + h + 1:]  = np.nan     # beyond the overlap: no counterpart
        # Score: sum of the absolute difference over the sum of the whole
        # trace — but with the sum over WAVELENGTH taken first, i.e. on the
        # autocorrelation rather than pixel by pixel.
        #
        # This is not cosmetic. Per pixel, |noise - noise| never cancels: the
        # modulus turns every one of the ~30k near-zero background pixels into
        # a positive contribution, and their sum swamps the signal. Measured on
        # the simulator, a PERFECTLY symmetric beam scored 13.8% at 460 peak
        # counts, 2.7% at 4000 and 0.9% at 20000 — a floor that moves with
        # exposure and, at realistic levels, is not far off the 26.7% a badly
        # misaligned trace scored. Collapsing the wavelength axis first averages
        # that noise down instead of rectifying it: the same two traces score
        # 0.6% and 20.6%, near enough exposure-independent.
        #
        # The blind spot is an asymmetry that cancels in the wavelength sum. The
        # IMAGE is per-pixel and still shows it — the picture says where, the
        # number says how much.
        a = np.asarray(ac, float)
        total = float(np.nansum(a))
        score = (float(np.nansum(np.abs(a[c:c + h + 1] - a[c - h:c + 1][::-1])))
                 / total) if total > 0 else 0.0
        return out, peak, score, fold

    def _current_symmetry_view(self):
        """The symmetry view of the finished scan, or None if there is none."""
        if self.result is None:
            return None
        return self._symmetry_view(self._result_view, self._result_ac,
                                   self.result.delays_fs)

    def _show_symmetry(self):
        """Push the symmetry view, its fold line and its score to the canvas."""
        view = self._current_symmetry_view()
        if view is None:
            return
        data, peak, score, fold = view
        with self.canvas.batch():
            self.canvas.update_trace(data, peak)
            self.canvas.set_fold_line(fold)
            self.canvas.set_symmetry_score(
                f"Asymmetry  {100 * score:.2f} %   @ {fold:+.0f} fs")

    def _on_fold_dragged(self, x_fs):
        """The fold line was dragged. Re-render only when it has actually
        changed COLUMN: a drag reports every pixel it crosses, and recomputing
        a trace-sized difference for a move that lands on the same column would
        be work with nothing to show for it."""
        if self.result is None or not self._symmetry_on:
            return
        before = self._symmetry_center(self._result_ac, self.result.delays_fs)
        self._fold_fs = float(x_fs)
        after = self._symmetry_center(self._result_ac, self.result.delays_fs)
        if after != before:
            self._show_symmetry()

    def _on_fold_reset(self):
        """Double-click on the line: back to the autocorrelation peak."""
        if self._fold_fs is None or not self._symmetry_on:
            return
        self._fold_fs = None
        self._show_symmetry()
        self.status.showMessage("Fold line back on the autocorrelation peak.", 3000)

    def _refresh_symmetry_button(self, running=None):
        """Enable the symmetry toggle only when there is a finished scan to fold.

        A scan in progress is excluded even though self.result may still hold the
        PREVIOUS one: the panel is showing the new trace being filled in, and a
        symmetry view of the old result over the top of it would be a picture of
        something the operator is no longer looking at.

        `running` overrides the inferred state at the two moments _scan_running()
        cannot answer for it. A scan about to start has not created its worker
        yet, so the flag still describes the PREVIOUS scan and reads False; and
        _reset_scan_ui runs from the worker's own finished signal, by which point
        the QThread has not necessarily left run() and it reads True. Inferring
        at either would leave the button in the wrong state until some unrelated
        refresh happened along.
        """
        if running is None:
            running = self._scan_running()
        ready = self.result is not None and not running
        self.btn_symmetry.setEnabled(ready)
        self.btn_symmetry.setToolTip(
            "Symmetry mode — the measured trace left of the fold line, and "
            "|T(+d) − T(−d)| right of it: how far the two temporal halves are "
            "from being each other's mirror image.\nBoth halves share one "
            "colour scale, so a dark right half genuinely means a symmetric "
            "trace. The corner readout is the asymmetry as a fraction of the "
            "whole trace.\nThe line starts on the autocorrelation peak — drag "
            "it to fold somewhere else, double-click it to put it back.\n"
            "Display only: the recorded trace and every export are unaffected."
            if ready else
            "Run a scan first — symmetry mode folds a finished trace about its "
            "autocorrelation peak.")

    def _on_symmetry_toggled(self, on):
        # Switching the mode re-centres the fold, in BOTH directions. A
        # hand-dragged position belongs to the look the operator was taking at
        # the time; turning the mode off ends that look, so coming back to it
        # should start from the autocorrelation peak rather than resurrect a
        # fold placed for a question already answered. Before the view is
        # computed on the next line, so the `on` path folds about the peak.
        self._fold_fs = None
        view = self._current_symmetry_view() if on else None
        if on and view is None:
            self.btn_symmetry.blockSignals(True)
            self.btn_symmetry.setChecked(False)
            self.btn_symmetry.blockSignals(False)
            self.status.showMessage("No finished scan to fold yet.", 4000)
            return
        self._symmetry_on = bool(on)
        if self._symmetry_on and self.btn_align_trace.isChecked():
            # One trace-view mode at a time: both write the same image, so the
            # last one switched on would silently win on the next re-render.
            self.btn_align_trace.blockSignals(True)
            self.btn_align_trace.setChecked(False)
            self.btn_align_trace.blockSignals(False)
            self._align_trace_on = False
            self._refresh_align_trace_button()
        if self._symmetry_on:
            self._show_symmetry()
        else:
            self.canvas.set_fold_line(None)
            self.canvas.set_symmetry_score("")
            self._restore_trace_view()
        self.status.showMessage(
            "Symmetry — measured trace left of the fold, |T(+d) − T(−d)| "
            "right of it. Drag the line to move it, double-click to re-centre."
            if on else "FROG trace: measured counts.", 6000)

    def _restore_trace_view(self):
        """Put the ordinary trace back after a display mode is switched off."""
        if self.result is not None:
            self.canvas.update_trace(self._result_view)
        elif self._scan_trace is not None:
            self.canvas.update_trace(self._scan_trace, self._scan_peak)

    def _clear_symmetry(self, running=None):
        """Drop out of symmetry mode. Called when a scan starts: the fold line
        belongs to the result it was computed from, and the panel is about to
        start showing a different trace entirely."""
        if self.btn_symmetry.isChecked():
            self.btn_symmetry.blockSignals(True)
            self.btn_symmetry.setChecked(False)
            self.btn_symmetry.blockSignals(False)
        self._symmetry_on = False
        # A hand-placed fold belongs to the trace it was placed on, so a new
        # scan starts back on the autocorrelation peak.
        self._fold_fs = None
        self.canvas.set_fold_line(None)
        self.canvas.set_symmetry_score("")
        self._refresh_symmetry_button(running)

    def _init_align_trace(self, wl, n_delays):
        """Allocate the raw trace for a scan about to start, or drop it.

        The cut is fixed HERE, for the whole scan: every column is
        interpolated and sliced on arrival, so moving the overlap band
        mid-scan cannot retroactively re-cut what is already stored. The next
        scan picks up wherever the band has been left.
        """
        spec = self.spec
        if not isinstance(spec, StitchedSpectrometer):
            self._clear_align_trace()
            return
        w = np.asarray(wl, float)
        cut = 0.5 * (spec.band_lo_nm + spec.band_hi_nm)
        # Below the cut comes from spec1 (the bluer member), above from spec2.
        # Neither half extrapolates: the common grid starts at spec1's first
        # pixel, and the cut sits inside the overlap, so spec2 covers w[k:].
        self._align_cut   = int(np.searchsorted(w, cut))
        self._align_trace = np.zeros((w.size, n_delays), np.float32)
        self._align_view  = None
        self._align_peak  = [0.0, 0.0]
        self._align_wl    = [np.asarray(m.wavelengths, float) for m in spec.members]
        self._refresh_align_trace_button()

    def _store_align_column(self, i, members_raw):
        """Fold one column's member frames into the raw trace. O(n_wl).

        The recorded dark comes off each member BEFORE interpolation, at member
        level, because that is the only place it can be taken off without
        involving either calibration or the crossfade — which is the whole point
        of this view. Everything else stays raw: no calibration, no stitch
        factor, a hard cut rather than a blend, each half on its own scale.
        """
        if self._align_trace is None or members_raw is None:
            return
        w = self._scan_wl
        k = self._align_cut
        wl1, wl2 = self._align_wl
        m1, m2 = (np.asarray(f, float) for f in members_raw)
        darks = self._scan_member_dark
        if darks is not None and len(darks) == 2:
            d1, d2 = (np.asarray(d, float) for d in darks)
            if d1.shape == m1.shape and d2.shape == m2.shape:
                m1 = m1 - d1
                m2 = m2 - d2
        lo = np.interp(w[:k], wl1, m1)
        hi = np.interp(w[k:], wl2, m2)
        self._align_trace[:k, i] = lo
        self._align_trace[k:, i] = hi
        if lo.size:
            self._align_peak[0] = max(self._align_peak[0], float(lo.max()))
        if hi.size:
            self._align_peak[1] = max(self._align_peak[1], float(hi.max()))

    def _align_trace_view(self):
        """The raw trace with each member's half normalised to its own max.

        Normalisation happens HERE, not when a column is stored: the two
        maxima only grow as a scan runs, so scaling on arrival would leave
        every earlier column on a stale factor. Written into a reused buffer —
        this runs once per column during a live scan.
        """
        raw = self._align_trace
        k   = self._align_cut
        if self._align_view is None or self._align_view.shape != raw.shape:
            self._align_view = np.empty_like(raw)
        buf = self._align_view
        np.divide(raw[:k], max(self._align_peak[0], 1.0), out=buf[:k])
        np.divide(raw[k:], max(self._align_peak[1], 1.0), out=buf[k:])
        return buf

    def _clear_align_trace(self):
        """Forget the raw trace and drop out of the raw view.

        Called on a device swap: a raw trace belongs to the pair that measured
        it — its cut, its grid and its two normalisations all describe those
        two spectrometers.
        """
        self._align_trace = None
        self._align_view  = None
        self._align_peak  = [0.0, 0.0]
        if self.btn_align_trace.isChecked():
            self.btn_align_trace.blockSignals(True)
            self.btn_align_trace.setChecked(False)
            self.btn_align_trace.blockSignals(False)
        self._align_trace_on = False
        self._refresh_align_trace_button()

    # Whether the live feed is meant to be running. The toolbar button is the
    # one piece of state that answers that, but callers outside this file (the
    # Avantes window has to stop the feed before arming a hardware trigger)
    # should ask in these terms rather than reaching for the widget.
    @property
    def feed_enabled(self):
        return self.btn_feed.isChecked()

    def set_feed_enabled(self, on):
        """Turn the live feed on or off as if the button had been clicked —
        setChecked runs _toggle_feed through the toggled signal, so the feed,
        the button's icon and its tooltip all follow."""
        self.btn_feed.setChecked(bool(on))

    def _toggle_feed(self, on):
        if on:
            # During a scan the worker owns the hardware; arm the feed and let
            # _reset_scan_ui start it once the scan hands the devices back.
            if not self._scan_running():
                self._feed.resume()
        else:
            self._feed.pause()
        self._refresh_feed_button()

    def _apply_integration_time(self):
        """Push the (debounced) integration time(s) to the spectrometer.

        A stitched pair takes one value per member, from the S1/S2 spinboxes
        in slot order; anything else takes the single spinbox. Both boxes
        share this one debounce, so editing them together costs a single feed
        handover.
        """
        if self._scan_running():
            self.status.showMessage(
                "A scan is running — integration time applies to the next scan.", 4000)
            return
        if not self._have_spec():
            return
        slots = self._slot_members() if self._pair_live() else []
        changed = []
        with self._device_lock() as ok:
            if not ok:
                self.status.showMessage(FEED_BUSY_MSG, 4000)
                return
            try:
                if len(slots) == 2:
                    for i, (mem, spin) in enumerate(
                            zip(slots, (self.spin_integration,
                                        self.spin_integration2))):
                        ms = float(spin.value())
                        # Compared at the box's own resolution, not exactly:
                        # the Avantes stores the exposure as a C float, so
                        # 0.02 ms reads back as 0.019999999552965164 and an
                        # equality test would rewrite both devices — and take
                        # a feed handover — every time the debounce fired.
                        dp = spin.decimals()
                        if round(_exposure_ms(mem), dp) == round(ms, dp):
                            continue   # already there — don't disturb the device
                        self.spec.set_member_integration_time(
                            self.spec.member_index(mem), ms)
                        changed.append(f"S{i + 1} {ms:g} ms")
                else:
                    ms = float(self.spin_integration.value())
                    # Same resolution-limited compare as the pair branch above,
                    # and for the same reason — but here it also decides whether
                    # the dark is thrown away, so a debounce that fires without
                    # the value having moved must not cost the operator a dark.
                    dp = self.spin_integration.decimals()
                    if round(_exposure_ms(self.spec), dp) != round(ms, dp):
                        self.spec.set_integration_time(ms)
                        changed.append(f"{ms:g} ms")
            except Exception as e:
                self.status.showMessage(f"Integration time failed: {e}", 5000)
                return
        if not changed:
            return
        # Signal scales with exposure, so the counts axis is now the wrong size
        # — dropping the exposure 10x leaves the trace in the bottom tenth of
        # the panel. Refit once, on the next frame rather than now: the frame on
        # screen was taken at the OLD exposure and fitting to it would just
        # reproduce the range being escaped.
        self._pending_rescale = True
        # And drop the frame already in flight for the same reason — it was
        # acquired before the device was rewritten, so the fit must not land on
        # it. Latest-frame-wins means nothing else is waiting behind it.
        self._live_frame = None
        # A dark holds the pedestal of the exposure it was taken at, so it is
        # now wrong — on a single device as much as on a pair, which used to go
        # unmentioned entirely.
        had_dark = (self.background is not None
                    or self.background_members is not None)
        if len(slots) == 2:
            # Frames stay in raw counts, so stitch_factor now carries the wrong
            # exposure ratio and the seam will show. Say so rather than let the
            # user discover it in the data.
            self._stitch_stale = True
            self.status.showMessage(
                f"{', '.join(changed)} — the two spectrometers no longer "
                f"share a scale; re-run Auto-stitch (⇌ above the spectrum, or "
                f"the Spectrometer window).", 8000)
        if had_dark:
            # Last, so its persistent message is the one left standing.
            self._invalidate_dark(f"integration time changed to "
                                  f"{', '.join(changed)}")

    def _capture_dark(self):
        """Record the dark from a FRESH frame.

        Always acquires rather than reusing self.last_spectrum: that field only
        moves while the display ticks, so with the feed stopped it holds the
        last LIT frame and "Record Dark" recorded the light. Acquiring also
        means the merged frame and the member frames come from one exposure
        instead of two different ones.
        """
        if self._scan_running():
            self.status.showMessage("A scan is running — spectrometer is busy.", 3000)
            return
        if not self._have_spec():
            self.status.showMessage("Connect a spectrometer first.", 4000)
            return
        with self._device_lock() as ok:
            if not ok:
                self.status.showMessage(FEED_BUSY_MSG, 4000)
                return
            try:
                frame = np.asarray(self.spec.acquire(), float)
            except Exception as e:
                self.status.showMessage(f"Dark capture failed: {e}", 5000)
                return
            # Inside the lock: the frames belong to the acquire above, and the
            # feed must not overwrite last_member_raw before we have copied it.
            members = self._slot_members()
            frames = getattr(self.spec, "last_member_raw", None)
            if len(members) == 2 and frames is not None:
                # Stored positionally in self.spec.members order — NOT keyed by
                # id(), which is recycled across reconnects; _apply_spectrometer
                # clears the field on every swap, so the order cannot go stale.
                self.background = None
                self.background_members = tuple(np.asarray(f, float).copy()
                                                for f in frames)
                exposures = tuple(_exposure_ms(m) for m in self.spec.members)
                where = ", ".join(
                    f"S{i + 1} {_exposure_ms(m):g} ms"
                    for i, m in enumerate(members))
            else:
                self.background = frame
                self.background_members = None
                exposures = (_exposure_ms(self.spec),)
                where = f"{_exposure_ms(self.spec):g} ms"
        self.background_exposures = exposures
        self.last_spectrum = frame
        self._dark_member_warned = False
        self.chk_dark.setEnabled(True); self.chk_dark.setChecked(True)
        self.status.showMessage(f"Dark recorded — {where}.", 4000)

    def _invalidate_dark(self, reason):
        """Throw the recorded dark away and say why.

        Used when something changes what a frame CONTAINS — an exposure, or an
        Avantes on-board correction. A dark is offset + dark-current x t, so
        rescaling it by an exposure ratio is wrong in general and would leave a
        plausible-looking but incorrect baseline; better to have none. No-op
        when there is nothing recorded, so callers need not check.

        Not needed for a stitch-factor or overlap-band change: the merged dark
        is derived from the member frames, so it follows both on its own.
        """
        if (self.background is None and self.background_members is None):
            return
        self.background = None
        self.background_members = None
        self.background_exposures = None
        self._dark_member_warned = False
        self.chk_dark.setChecked(False); self.chk_dark.setEnabled(False)
        # Timeout 0 — a silently dropped dark is exactly the kind of thing that
        # should not scroll away before the operator looks up.
        self.status.showMessage(f"Dark discarded — {reason}. Re-record it.", 0)

    def _set_hardware_buttons_enabled(self, on):
        """Every route to a device swap, locked together.

        A scan and an alignment sweep both capture the stage and spectrometer
        for their duration, so nothing may connect, disconnect or simulate
        underneath them — and that is now five buttons across the toolbar and
        the two side panels rather than the one it used to be.
        """
        for w in (self.btn_spec_dlg, self.btn_stage_dlg, self.btn_sim_dlg,
                  self.btn_connect_spec, self.btn_connect_stage):
            w.setEnabled(on)

    # ── Manual stage ──────────────────────────────────────────────────────────
    def _set_stage_controls_enabled(self, on):
        for w in (self.btn_minus, self.btn_plus, self.btn_moveto,
                  self.btn_home, self.btn_goto_zero, self.btn_set_zero,
                  self.btn_units, self.spin_backlash):
            w.setEnabled(on)

    def _size_moving_label(self):
        """Reserve the width of the WIDER of the stage label's two states.

        The label shows "IDLE" at rest, so its own size hint asks for four
        characters — and the layout, which has no reason to expect a fifth,
        hands it exactly that when the row is tight. "MOVING" then does not fit,
        and because the label is centred it is clipped at both ends, which is
        what turned it into "OVING".

        Measured rather than hard-coded: the #idle/#moving rules set the size
        and weight in the stylesheet, so the number changes with the theme —
        hence the call from _apply_theme as well as from the build.
        """
        fm = self.lbl_moving.fontMetrics()
        # +4: QLabel's own text margin, which fontMetrics does not include.
        self.lbl_moving.setMinimumWidth(fm.horizontalAdvance("MOVING") + 4)

    def _moving(self, on):
        self.lbl_moving.setText("MOVING" if on else "IDLE")
        self.lbl_moving.setObjectName("moving" if on else "idle")
        self.lbl_moving.style().unpolish(self.lbl_moving); self.lbl_moving.style().polish(self.lbl_moving)
        # repaint() paints synchronously WITHOUT spinning the event loop.
        # processEvents() here used to dispatch queued work — a spectrum_ready
        # delivery, or another click on the jog button — in the middle of a
        # blocking move, re-entering this code path.
        self.lbl_moving.repaint()

    def _stage_action(self, fn, done_msg=None):
        """Run a blocking stage operation from a slot, safely.

        Parks the live feed (which drives the same stage from thread 2),
        disables the stage controls so the operation cannot be re-entered, and
        restores both however `fn` exits.
        """
        if self._scan_running():
            self.status.showMessage("A scan is running — the stage is busy.", 3000)
            return
        if not self._have_stage():
            self.status.showMessage("Connect a stage first.", 4000)
            return
        self._set_stage_controls_enabled(False)
        self._moving(True)
        try:
            with self._device_lock() as ok:
                if not ok:
                    self.status.showMessage(FEED_BUSY_MSG, 5000)
                    return
                fn()
                self._refresh_positions()   # still inside the lock
                # An adapter that reports a failed move through position_fault()
                # instead of raising (the piezo does, so a scan can record the
                # column and carry on) would otherwise leave a manual move
                # looking successful while the stage sits somewhere else.
                fault = None
                try:
                    fault = self.stage.position_fault()
                except Exception:
                    pass
                if fault:
                    self.status.showMessage(f"Stage: {fault}", 8000)
                elif done_msg:
                    self.status.showMessage(done_msg, 3000)
        except Exception as e:
            self.status.showMessage(f"Stage error: {e}", 5000)
        finally:
            self._moving(False)
            self._set_stage_controls_enabled(True)

    def _travel_range_um(self):
        """Reachable position range (lo, hi) in um.

        Not [0, travel]: limit.min is not always 0 (a soft limit protecting the
        optics is common), and the low end has to leave room for the backlash
        pre-move to undershoot into. Falls back to 300 mm of travel when the
        adapter reports none.
        """
        t = getattr(self.stage, "travel_mm", None)
        hi = _stage_to_um(t if t else 300.0)
        lo = _stage_to_um(float(getattr(self.stage, "travel_min_mm", 0.0) or 0.0)
                          + float(getattr(self.stage, "backlash_mm", 0.0) or 0.0))
        return lo, hi

    def _update_stage_unit_ranges(self):
        """Set spin_step/spin_moveto ranges for the active unit, the connected
        stage's travel, and the current zero. Call after: unit toggle, stage
        swap, zero change, backlash change."""
        lo_um, hi_um = self._travel_range_um()
        pf, z = self.scan_cfg.pass_factor, self.scan_cfg.zero_pos_um
        if self._stage_units_fs:
            # Inward rounding (ceil min, floor max) so every integer fs in
            # range maps to a position strictly inside the travel range.
            self.spin_step.setRange(
                1, max(1, math.floor(float(
                    position_to_delay_fs(hi_um - lo_um, 0.0, pf)))))
            self.spin_moveto.setRange(
                math.ceil(float(position_to_delay_fs(lo_um, z, pf))),
                math.floor(float(position_to_delay_fs(hi_um, z, pf))))
        else:
            self.spin_step.setRange(0.01, hi_um - lo_um)
            self.spin_moveto.setRange(lo_um, hi_um)

    def _sync_backlash_ui(self):
        """Push the active stage's backlash into the spin box and its fs hint.

        Blocks signals: this reflects the stage, it must not write back to it.
        """
        b_um = _stage_to_um(float(getattr(self.stage, "backlash_mm", 0.0) or 0.0))
        self.spin_backlash.blockSignals(True)
        self.spin_backlash.setValue(b_um)
        self.spin_backlash.blockSignals(False)
        self._refresh_backlash_hint(b_um)

    def _refresh_backlash_hint(self, b_um):
        if b_um <= 0.0:
            self.lbl_backlash_fs.setText(
                "Off — moves arrive from whichever side they came from.")
            return
        fs = float(position_to_delay_fs(b_um, 0.0, self.scan_cfg.pass_factor))
        self.lbl_backlash_fs.setText(
            f"Approach from below, undershooting {b_um:.1f} um ({fs:.0f} fs).")

    def _on_backlash_changed(self, value_um):
        if self._scan_running():
            self.status.showMessage("A scan is running — the stage is busy.", 3000)
            self._sync_backlash_ui()      # snap back to what the stage has
            return
        if not self._have_stage():
            return
        # Plain attribute write, no device I/O — no lock needed. Ranges shift
        # because the low end reserves the margin.
        self.stage.backlash_mm = _um_to_stage(float(value_um))
        self._refresh_backlash_hint(float(value_um))
        self._update_stage_unit_ranges()

    def _toggle_stage_units(self):
        pf, z = self.scan_cfg.pass_factor, self.scan_cfg.zero_pos_um
        step, move = self.spin_step.value(), self.spin_moveto.value()
        self._stage_units_fs = not self._stage_units_fs
        if self._stage_units_fs:
            new_step = round(float(position_to_delay_fs(step, 0.0, pf)))
            new_move = round(float(position_to_delay_fs(move, z, pf)))
            suffix, dec = " fs", 0
        else:
            new_step = round(float(delay_to_position_um(step, 0.0, pf)))
            new_move = round(float(delay_to_position_um(move, z, pf)))
            suffix, dec = " um", 2
        for s in (self.spin_step, self.spin_moveto):
            s.setSuffix(suffix); s.setDecimals(dec)
        self._update_stage_unit_ranges()   # ranges before values: no bad clamp
        self.spin_step.setValue(new_step)
        self.spin_moveto.setValue(new_move)
        self.btn_units.setText("fs" if self._stage_units_fs else "um")

    def _jog(self, sign):
        v = self.spin_step.value()
        step_um = (float(delay_to_position_um(v, 0.0, self.scan_cfg.pass_factor))
                   if self._stage_units_fs else v)
        lo, hi = self._travel_range_um()
        def do():
            # Clamped move_to instead of move_by: the target can never leave
            # the travel range. get_position() runs inside the device lock.
            cur = _stage_to_um(self.stage.get_position())
            target = min(max(cur + sign * step_um, lo), hi)
            self.stage.move_to(_um_to_stage(target))
        self._stage_action(do)

    def _move_absolute(self):
        v = self.spin_moveto.value()
        pf, z = self.scan_cfg.pass_factor, self.scan_cfg.zero_pos_um
        target_um = float(delay_to_position_um(v, z, pf)) if self._stage_units_fs else v
        lo, hi = self._travel_range_um()
        target_um = min(max(target_um, lo), hi)
        self._stage_action(lambda: self.stage.move_to(_um_to_stage(target_um)))

    def _move_to_zero(self):
        self._stage_action(
            lambda: self.stage.move_to(_um_to_stage(self.scan_cfg.zero_pos_um)))

    def _home_stage(self):
        """Home the axis, and drop a zero that homing has invalidated.

        Homing an axis that had no reference re-establishes the coordinate
        frame, which moves any zero marked in the old one. Silently keeping it
        would shift the whole delay axis by the home offset, so it goes.
        """
        was_unreferenced = bool(getattr(self.stage, "needs_homing", False))
        landed = []          # stays empty if home() raised or the feed was busy

        def do():
            self.stage.home()
            # Read it here, inside the lock — get_position() is device I/O and
            # the live feed drives the same stage from its own thread.
            landed.append(_stage_to_um(self.stage.get_position()))

        self._stage_action(do)
        if not landed:
            return           # _stage_action already reported why
        pos_um = landed[0]
        if was_unreferenced:
            self.scan_cfg.zero_pos_um = pos_um
            self._update_stage_unit_ranges()
            self._refresh_positions()
            self._refresh_scan_um()
            self.status.showMessage(
                f"Homed to {pos_um:.2f} um. The axis had no reference before, "
                f"so the old zero-delay was in a different frame — set zero "
                f"again before scanning.", 0)
        else:
            self.status.showMessage(f"Stage homed to {pos_um:.2f} um.", 3000)

    def _mark_zero(self):
        def mark():
            # get_position() is a device read like any other — take the lock.
            self.scan_cfg.zero_pos_um = _stage_to_um(self.stage.get_position())
        self._stage_action(mark)
        # fs-mode values are delays and stay as typed — the zero moved, so the
        # same delay now maps to the new (correct) absolute position. Only the
        # ranges need recomputing; Qt clamps anything now out of range.
        self._update_stage_unit_ranges()
        self._refresh_scan_um()
        self.status.showMessage(f"Zero-delay set to {self.scan_cfg.zero_pos_um:.2f} um.", 3000)

    def _refresh_positions(self):
        self.lbl_pos.setText(
            f"{_stage_to_um(self.stage.get_position()):.2f} um"
            if self._have_stage() else "— um")
        self.lbl_zero.setText(f"{self.scan_cfg.zero_pos_um:.2f} um")

    def _refresh_scan_um(self):
        z = self.scan_cfg.zero_pos_um
        p0 = delay_to_position_um(self.spin_start.value(), z, self.scan_cfg.pass_factor)
        p1 = delay_to_position_um(self.spin_stop.value(),  z, self.scan_cfg.pass_factor)
        self.eq_start.setText(f"{p0:,.1f} um")
        self.eq_stop.setText(f"{p1:,.1f} um")

    # ── FROG scan ─────────────────────────────────────────────────────────────
    def _start_scan(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self.btn_scan.setEnabled(False); self.btn_scan.setText("Aborting…")
            return
        if not self._have_spec():
            self.status.showMessage("Connect a spectrometer first.", 4000); return
        if not self._have_stage():
            self.status.showMessage("Connect a stage first — a FROG scan has "
                                    "to sweep the delay.", 4000); return
        if self._align_running():
            self.status.showMessage(
                "An alignment sweep is running — the stage is busy.", 3000); return
        # An unreferenced axis has no usable coordinate frame: every delay the
        # scan would record is measured from a zero that means nothing, and
        # homing afterwards moves the frame again. Refuse rather than save it.
        if getattr(self.stage, "needs_homing", False):
            self.status.showMessage(
                "Stage is not homed — press Home, then set zero, then scan.",
                6000)
            return

        c = self.scan_cfg
        c.delay_start_fs = self.spin_start.value()
        c.delay_stop_fs  = self.spin_stop.value()
        c.delay_step_fs  = self.spin_step_fs.value()
        c.n_average      = self.dlg_settings.n_average
        c.idle_shots     = self.dlg_settings.idle_shots
        c.wait_after_move_s = self.dlg_settings.settle_ms / 1000.0
        c.capture_delay_reference = self.dlg_settings.take_delay_reference
        c.saturation_fraction  = self.dlg_settings.sat_percent / 100.0
        c.abort_on_saturation  = self.dlg_settings.abort_on_saturation
        c.abort_on_stage_fault = self.dlg_settings.abort_on_stage_fault
        # c.saturation_counts is owned by the Hardware dialog's override field.

        try:
            delays = c.delays_fs()
        except ValueError as e:
            self.status.showMessage(f"Bad scan range: {e}", 4000); return
        wl = np.asarray(self.spec.wavelengths, float)

        # No dark means the exported trace carries the full pedestal, which on
        # a typical bench is several times the signal — so say so ONCE, before
        # anything moves, and let the operator decide. Asked here rather than
        # refused outright because a deliberately uncorrected scan is a
        # legitimate thing to want; the .npz records which it was either way.
        if self._display_dark() is None and not self._confirm_no_dark():
            return

        # Hand the shared stage/spectrometer over to the scan worker: pause the
        # live feed and wait for any in-flight acquire to finish first. If it
        # will not let go, refuse to start rather than run the scan against a
        # spectrometer the feed thread is still inside.
        self._feed_was_on = self.btn_feed.isChecked()
        if not self._feed.pause():
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(f"Scan not started — {FEED_BUSY_MSG}", 6000)
            return

        # The worker captures these devices for the whole scan, so lock out
        # everything that could swap or drive them underneath it.
        self._set_hardware_buttons_enabled(False)
        self._set_stage_controls_enabled(False)
        # The fold line belongs to the result it came from, and the panel is
        # about to start showing a different trace. running=True is asserted:
        # the worker for THIS scan does not exist yet, so _scan_running() would
        # still be answering for the previous one.
        self._clear_symmetry(running=True)

        self._scan_trace  = np.zeros((wl.size, delays.size))
        self._scan_delays = delays
        self._scan_wl     = wl
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
        self._scan_dark   = self._display_dark()
        # The same dark at MEMBER level, for the RAW view. RAW shows the two
        # detectors uncalibrated and unstitched, so the pedestal has to come off
        # each member's own raw frame — the combined dark above has already been
        # through both calibrations and the crossfade.
        self._scan_member_dark = (
            self.background_members
            if (self.chk_dark.isChecked() and self._pair_live()) else None)
        # Running reductions _on_column keeps current, so the per-column render
        # never has to walk the trace. Reallocated with the trace: an aborted
        # scan leaves these sized to the OLD delay axis.
        self._scan_ac     = np.zeros(delays.size)
        self._scan_peak   = 0.0
        # A pending render from a previous (aborted) scan must not fire
        # against the fresh, differently-sized arrays.
        self._scan_dirty  = False
        self._scan_last_i = -1
        self._scan_members_raw = None   # …and the last scan's member frames
        self._live_frame  = None     # park any leftover live-feed frame too
        self._init_align_trace(wl, delays.size)
        self.canvas.init_trace(delays, wl)
        self._reset_saturation(); self.progress.setValue(0)
        # A difference measured at the old position says nothing about the
        # scan that is starting, and the scan owns the stage the sweep needs.
        if self.btn_align_spec.isChecked() or self.canvas.dev_visible():
            self._close_align_mode()

        self.btn_scan.setObjectName("danger"); self.btn_scan.setText("Abort Scan")
        self.btn_scan.style().unpolish(self.btn_scan); self.btn_scan.style().polish(self.btn_scan)
        self._set_save_enabled(False)

        self._worker = FrogScanWorker(self.stage, self.spec, c,
                                      dark=self._scan_dark)
        self._worker.progress.connect(self._on_progress)
        self._worker.column_ready.connect(self._on_column)
        self._worker.delay_reference_ready.connect(self._on_delay_reference)
        self._worker.saturation_warning.connect(self._on_saturation)
        self._worker.stage_fault.connect(self._on_stage_fault)
        self._worker.finished_scan.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # QThread.finished, not finished_scan: the worker parks the stage on
        # zero delay in a finally, which runs AFTER finished_scan is emitted.
        # Refreshing the readout from _on_finished would print the last delay's
        # position and leave it there.
        self._worker.finished.connect(self._refresh_positions)
        self._worker.start()
        # Auto-stitch drives the pair the worker now owns, so it greys out for
        # the duration. The per-spectrometer view does not — the scan carries
        # its own member frames, and _render_scan_frame draws them.
        self._refresh_overlay_button()
        # Explicit, not inferred: this happens to run after _worker.start(), so
        # _scan_running() would answer True — but that is an ordering accident,
        # and the same call one line earlier would have greyed nothing.
        self._refresh_autostitch_button(running=True)
        self.status.showMessage(f"FROG scan: {delays.size} points…", 0)

    def _on_progress(self, done, total):
        self.progress.setValue(int(100 * done / total))

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
        self._scan_trace[:, i] = col
        # The raw member frames are interpolated and cut ONCE, here, rather
        # than kept per column and re-derived at render time: that keeps the
        # memory to one trace-sized array and the work to O(n_wl) per column.
        self._store_align_column(i, members_raw)
        # …but the NEWEST pair is kept as-is, so the per-spectrometer spectrum
        # view has something to draw while a scan owns the device. One tuple
        # reference, overwritten per column — the scan's cut and normalisation
        # above are for the trace and are the wrong thing for this panel.
        self._scan_members_raw = members_raw
        # Fold the new column into the running reductions the render needs, so
        # _render_scan_frame stays O(1) in the column index. Recomputing either
        # of these from the whole trace per column made the scan cost O(N^2).
        self._scan_ac[i]  = col.sum()
        self._scan_peak   = max(self._scan_peak, float(col.max()))
        self._scan_last_i = i
        self._scan_col = col
        self._scan_pos_um = pos_um
        self._scan_dirty = True

    def _render_scan_frame(self):
        """Draw the in-progress scan from the newest recorded column.

        Everything here is O(1) in the column index on purpose. The obvious
        version — autocorrelation() over self._scan_trace[:, :i+1], and a
        raw.max() over the whole preallocated trace inside _render_trace — is
        O(n_wl * i) per column, i.e. O(n_wl * N^2) over the scan, and was what
        made a long scan get slower the further it ran. _on_column keeps both
        reductions up to date incrementally instead.
        """
        self._scan_dirty = False
        i = self._scan_last_i
        self.lbl_pos.setText(f"{self._scan_pos_um:.2f} um")
        # Same value autocorrelation() would return for this slice: a per-column
        # sum over the wavelength axis, baseline-shifted to its own minimum
        # (scan.py:87-89). Only the baseline couples the columns, and a min over
        # the accumulated sums is 1-D and cheap.
        acc = self._scan_ac[:i + 1]
        ac  = acc - acc.min()
        # One blit for all three panels instead of three.
        with self.canvas.batch():
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
            if self._scan_wl is not None:
                if not (self._overlay_on
                        and self._render_overlay(self._scan_members_raw)):
                    # Already corrected, in _on_column, along with the column
                    # that went into the trace beside this panel.
                    self.canvas.update_spectrum(self._scan_wl, self._scan_col)
            if self._align_trace_on and self._align_trace is not None:
                self.canvas.update_trace(self._align_trace_view(), 1.0)
            else:
                self.canvas.update_trace(self._scan_trace, self._scan_peak)
            self.canvas.update_ac(self._scan_delays[:i + 1], ac)
            # Inside the batch: the readout lives in the AC panel now, so its
            # repaint is the same blit the three updates above share.
            self.canvas.set_fwhm(fwhm(self._scan_delays[:i + 1], ac))

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
        bg = self._scan_dark
        if bg is None or np.shape(bg) != np.shape(col):
            return col
        return np.clip(col - bg, 0, None)

    def _confirm_no_dark(self):
        """Ask before running a scan whose export will not be dark-corrected.

        Two distinct reasons land here and the message names both, because the
        fix differs: no dark has been recorded at all, or one has but Subtract
        Dark is off. Answering No leaves everything untouched.
        """
        if self.background is None and self.background_members is None:
            why = ("No dark has been recorded.")
            fix = ("Record Dark in the Spectrum panel first, or continue and "
                   "the .dwc and .csv will carry the raw pedestal.")
        else:
            why = ("A dark is recorded, but Subtract Dark is switched off.")
            fix = ("Tick Subtract Dark to correct this scan, or continue and "
                   "the .dwc and .csv will carry the raw pedestal.")
        return QMessageBox.question(
            self, "Scan without a dark?",
            f"{why}\n\n{fix}\n\nThe .npz records that no dark was "
            f"subtracted either way.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No) == QMessageBox.Yes

    def _on_delay_reference(self, which, spectrum):
        self.status.showMessage(
            f"Delay-extreme reference ({which}) captured — archived, not "
            f"subtracted.", 2500)

    def _on_stage_fault(self, i, delay_fs, reason):
        """Slot for FrogScanWorker.stage_fault.

        With the abort enabled the worker's error() follows immediately and
        supersedes this; the message matters in the override case, where the
        scan carries on and this is the only live sign that it has stopped
        tracking. Timeout 0 — it must not quietly disappear.
        """
        where = "the reference frame" if i < 0 else f"{delay_fs:+.0f} fs"
        self.status.showMessage(f"⚠ STAGE FAULT @ {where} — {reason}", 0)

    def _on_saturation(self, i, delay_fs, npx, peak):
        """Slot for FrogScanWorker.saturation_warning — one signal per frame.

        Latches: it reports that THIS scan's data is clipped, which stays true
        for the rest of the scan even if later columns come back clean. The
        worst frame is what gets named, since simply overwriting on every
        signal would leave the last (often marginal) column on display and hide
        how much of the trace is actually ruined.
        """
        self._sat_latched = True
        self._sat_frames += 1
        if npx > self._sat_worst[0]:
            self._sat_worst = (npx, delay_fs, i)
        npx_w, delay_w, i_w = self._sat_worst
        where = "background" if i_w < 0 else f"{delay_w:+.0f} fs"
        self._set_lamp("sat",
                       f"⚠ SATURATED — {self._sat_frames} frame"
                       f"{'' if self._sat_frames == 1 else 's'}, "
                       f"worst @ {where} ({npx_w} px)",
                       "sat")
        # The worker's signal aggregates over stitched members; latch both
        # lamps so neither device shows a green light over clipped data.
        if self._pair_live():
            self.lamp2.set_state("sat")

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
        self.result = result
        # Cached, not recomputed per view: the symmetry fold and the RAW toggle
        # both re-render from it, and it is a trace-sized subtraction.
        self._result_view = self._finished_trace_view(result)
        # From the SAME array the panel shows, so the curve and the width
        # readout describe the picture above them. They used to come off
        # result.trace, i.e. off the pedestal-bearing counts.
        self._result_ac = autocorrelation(self._result_view)
        # The final render below supersedes any pending intermediate tick.
        self._scan_dirty = False
        if self._align_trace_on and self._align_trace is not None:
            # The raw view is a display mode, not a stage of the scan: finishing
            # must not silently drop the operator back to the calibrated trace.
            self.canvas.update_trace(self._align_trace_view(), 1.0)
        else:
            self.canvas.update_trace(self._result_view)
        ac = self._result_ac
        self.canvas.update_ac(result.delays_fs, ac)
        self.canvas.set_fwhm(fwhm(result.delays_fs, ac))
        self.progress.setValue(100)
        n_bad = int(result.faulted_columns().size)
        if n_bad:
            # Only reachable with the abort override off — the scan ran to the
            # end over a stage that had stopped tracking. Say so permanently
            # rather than for five seconds; the delay axis is compromised.
            self.status.showMessage(
                f"Scan complete — {result.trace.shape[1]} columns, but the "
                f"stage reported a fault on {n_bad} of them. Their delays are "
                f"not trustworthy; the .npz marks which.", 0)
        else:
            self.status.showMessage(
                f"Scan complete — {result.trace.shape[1]} columns.", 5000)
        self._set_save_enabled(True)
        self._reset_scan_ui()

    def _on_error(self, msg):
        self.status.showMessage(f"Scan: {msg}", 6000)
        self._reset_scan_ui()

    def _reset_scan_ui(self):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setObjectName("accent"); self.btn_scan.setText("Measure FROG")
        self.btn_scan.style().unpolish(self.btn_scan); self.btn_scan.style().polish(self.btn_scan)
        self._set_hardware_buttons_enabled(True)
        self._set_stage_controls_enabled(True)
        self._refresh_overlay_button()
        # running=False for the same reason as the symmetry button below: the
        # worker's thread may not have left run() yet, so neither may infer it.
        self._refresh_autostitch_button(running=False)
        self._refresh_align_trace_button()
        self._refresh_symmetry_button(running=False)   # a finished scan folds
        if self._feed_was_on and self.btn_feed.isChecked():
            self._feed.resume()

    # ── Save ──────────────────────────────────────────────────────────────────
    def _set_save_enabled(self, on):
        """Enable or disable both ways in to an export at once — the side
        panel's Save button and the Export window's. They write the same file
        from the same data, so a state that applies to one applies to both."""
        self.btn_save.setEnabled(on)
        self.dlg_export.btn_save.setEnabled(on)

    def _set_export_fmt(self, key):
        """Make `key` the active export format, without writing anything.

        The single writer of _export_fmt: the Export window's radios come here,
        and so does a format restored from settings at startup. It writes back
        to the radios as well, so the two can never disagree — that path is
        blocked at the widget (see ExportDialog.show_format) rather than guarded
        here, which keeps this function one-directional.
        """
        _label, suffix, _filt, _writer = EXPORT_FORMATS[key]   # bad key: KeyError
        self._export_fmt = key
        self.dlg_export.show_format(key)
        self.dlg_export.refresh_preview()      # the suffix in it just changed
        self.btn_save.setText(f"Save ({suffix})")

    def _export_now(self):
        """Write the last scan in the active format.

        Both Save buttons land here, and neither can be pressed without a scan
        (_set_save_enabled), so there is no 'nothing to save yet' branch — that
        message existed because picking a format from the old menu also tried
        to save. The guard below is for the one case the buttons cannot cover:
        a second press while the first export is still running.
        """
        if self.result is None:
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            self.status.showMessage("An export is already in progress.", 3000)
            return
        _label, suffix, filt, writer = EXPORT_FORMATS[self._export_fmt]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FROG scan", self.dlg_export.file_name(suffix), filt)
        if not path:
            return
        if not path.lower().endswith(suffix):
            path += suffix
        # Formatting a full trace takes seconds — write it on a worker thread so
        # the window keeps painting (and the live feed keeps running).
        self._set_save_enabled(False)
        self.status.showMessage(f"Saving → {path} …", 0)
        # Deliberately NOT parented to the window: a QThread child of a window
        # that lives for the session is never destroyed, so rebinding
        # self._export_worker would leak one thread — and the whole FrogResult
        # it holds — per save. The Python reference below is what keeps it alive
        # for the duration; finished() (queued onto the GUI thread) drops both.
        w = ExportWorker(writer, path, self.result)
        w.done.connect(self._on_export_done)
        w.error.connect(self._on_export_error)
        w.finished.connect(w.deleteLater)
        w.finished.connect(self._release_export_worker)
        self._export_worker = w
        w.start()

    def _release_export_worker(self):
        self._export_worker = None

    def _on_export_done(self, path):
        self._end_export()
        self.status.showMessage(f"Saved → {path}", 5000)

    def _on_export_error(self, msg):
        self._end_export()
        self.status.showMessage(f"Save failed: {msg}", 8000)

    def _end_export(self):
        # btn_export is left alone: it opens the Export window now rather than
        # writing anything, so there is nothing to lock out while a write runs.
        self._set_save_enabled(self.result is not None)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def _scan_join_timeout_ms(self):
        """How long to allow the scan worker to finish the step it is on.

        abort() only takes effect between delay points, and a point is a stage
        move plus (idle_shots + n_average) exposures — far more than the live
        feed's single-cycle estimate covers, so it gets its own figure derived
        from the settings the running scan was started with.
        """
        c = self.scan_cfg
        shots = max(1, int(c.idle_shots) + int(c.n_average))
        exposure = float(getattr(self.spec, "integration_ms", 10.0))
        return max(5000.0, 2.0 * shots * exposure
                   + 2000.0 * float(c.wait_after_move_s) + 5000.0)

    # ── Settings persistence ─────────────────────────────────────────────────
    # One table, read in both directions: (key, getter, setter). Adding a
    # setting is one line here and nothing else — the save, the autosave diff
    # and the restore all walk this list.
    #
    # A setter of None means "restored somewhere else, for a reason given at
    # that site": the theme has to exist before any widget does, and the
    # geometry has to be applied in the same breath as the decision to show
    # maximized. Both are still SAVED from here, so this stays the one place
    # that knows the full set.
    #
    # ORDER IS LOAD-BEARING on restore, twice over:
    #   • the auto-scale checkboxes come before the limit spinboxes, because
    #     _on_autoscale_trace(False) pins those four boxes to whatever is on
    #     screen and would overwrite the values being restored;
    #   • the layout mode comes before the split fraction, so the axes are laid
    #     out once, in the mode they are going to stay in.
    #
    # Getters read the CANVAS for the auto-scale flags and the WIDGETS for the
    # manual limits. That split is not arbitrary: _apply_zoom and reset_axes
    # mutate canvas.autoscale_* directly, so the canvas is the only thing that
    # always knows; while the spinboxes are the memory of the last bounds the
    # operator dialled in by hand, which is what should come back — not a
    # snapshot of an auto-scaled view that happened to be live at the time.
    def _setting_specs(self):
        c = self.canvas
        g = self.dlg_graphics
        d = self.dlg_settings
        return [
            ("theme",   lambda: self._theme,    None),
            ("window",  self._window_geometry,  None),

            ("layout_mode", lambda: c._layout_mode, self._set_layout),
            ("spec_frac",   lambda: float(c._spec_frac),
                            lambda v: g.sld_prop.setValue(int(round(float(v) * 100)))),

            # NO axis state here — not the auto-scale flags, not the manual
            # limits, not the autocorrelation's view. Where the axes are
            # pointing is a property of the session, not a preference: a
            # rubber-band zoom and a hand-typed range land in exactly the same
            # place (auto-scale off, limits pinned), so persisting either meant
            # a zoom taken to read one feature survived the restart and the
            # program came back still zoomed into it, on data that had gone.
            # Every launch starts on the normal view and fits to what is there.
            ("log_scale",  lambda: g.log_scale,
                           lambda v: g.chk_log.setChecked(as_bool(v))),
            # setCurrentText is a silent no-op on a non-editable combo for a name
            # that is not in the list, so a colormap dropped from
            # TRACE_COLORMAPS needs no guard of its own.
            ("cmap",          lambda: c._cmap_name, g.cmb_cmap.setCurrentText),
            ("cmap_reversed", lambda: bool(c._cmap_rev),
                              lambda v: g.chk_cmap_rev.setChecked(as_bool(v))),
            # Percent, as typed — the canvas keeps the fraction.
            ("trace_threshold_pct", g.spin_thresh.value,
                                    lambda v: g.spin_thresh.setValue(float(v))),
            ("line_width",    lambda: float(c._lw),
                              lambda v: g.spin_lw.setValue(float(v))),
            ("export_format", lambda: self._export_fmt, self._set_export_fmt),
            # The name as TYPED, not the composed file name: the date prefix is
            # the checkbox's doing and is re-made from today's date on every
            # save, so storing it would bring back the day this was written.
            ("export_name", self.dlg_export.edit_name.text,
                            lambda v: self.dlg_export.edit_name.setText(str(v))),
            ("export_date_prefix", lambda: self.dlg_export.date_stamped,
                                   lambda v: self.dlg_export.chk_date.setChecked(
                                       as_bool(v))),

            ("avg_per_point",      lambda: d.n_average,
                                   lambda v: d.spin_avg.setValue(int(v))),
            ("idle_shots",         lambda: d.idle_shots,
                                   lambda v: d.spin_idle.setValue(int(v))),
            ("wait_after_move_ms", lambda: d.settle_ms,
                                   lambda v: d.spin_wait.setValue(int(v))),
            ("saturation_pct",     lambda: d.sat_percent,
                                   lambda v: d.spin_sat.setValue(float(v))),
            ("abort_on_saturation",  lambda: d.abort_on_saturation,
                                     lambda v: d.chk_abort_sat.setChecked(as_bool(v))),
            ("abort_on_stage_fault", lambda: d.abort_on_stage_fault,
                                     lambda v: d.chk_abort_stage_fault.setChecked(as_bool(v))),

            ("scan_start_fs",   self.spin_start.value,
                                lambda v: self.spin_start.setValue(float(v))),
            ("scan_stop_fs",    self.spin_stop.value,
                                lambda v: self.spin_stop.setValue(float(v))),
            ("scan_step_fs",    self.spin_step_fs.value,
                                lambda v: self.spin_step_fs.setValue(float(v))),
            ("scan_delay_reference", lambda: d.take_delay_reference,
                                lambda v: self.dlg_settings.chk_ref.setChecked(
                                    as_bool(v))),
            ("align_step_fs",   lambda: self.dlg_align.step_fs,
                                lambda v: self.dlg_align.spin_align_step.setValue(float(v))),
        ]

    def _window_geometry(self):
        """Position and size for the next launch.

        normalGeometry when maximized, not geometry: a maximized window reports
        the whole screen, and saving that would make the next un-maximize
        restore to a 'normal' size the operator never chose.
        """
        r = self.normalGeometry() if self.isMaximized() else self.geometry()
        return {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height(),
                "maximized": self.isMaximized()}

    def show_restored(self):
        """Show the window where the last session left it, maximized by default.

        Neither in __init__ nor as two calls in main(): setGeometry has to
        happen before the first show to be honoured without a visible jump, and
        the maximized decision has to be made in the same place or one of the
        two silently wins.

        Explicit x/y/w/h rather than a base64 saveGeometry() blob, because this
        file is meant to be readable and fixable by hand — an opaque blob cannot
        be corrected when a window comes back on a monitor that is no longer
        there, and restoreGeometry rejects blobs across Qt versions without
        saying so.
        """
        geo = self._settings.get("window")
        maximized = True
        if isinstance(geo, dict):
            maximized = geo.get("maximized", True) is not False
            try:
                r = QRect(int(geo["x"]), int(geo["y"]),
                          int(geo["w"]), int(geo["h"]))
            except (KeyError, TypeError, ValueError):
                r = QRect()
            if r.isValid() and _on_a_screen(r):
                self.setGeometry(r)        # Qt clamps up to setMinimumSize itself
        if maximized:
            self.showMaximized()
        else:
            self.show()

    def _restore_settings(self):
        """Apply the saved settings to widgets that already exist.

        Every key is applied on its own and every failure is swallowed on its
        own: a file half-written by an older build, a hand-edit with a typo in
        it, or a colormap that no longer exists must cost the operator that ONE
        setting and never the rest of them.

        Deliberately NOT blockSignals. The dialog's signal wiring IS the only
        path from a widget to the canvas — blocking it would restore the
        dialog's appearance while leaving the canvas on defaults — and the
        handlers carry side effects nothing else does: _on_autoscale_* enable
        and disable the boxes they own, _on_prop repaints its percent label.
        Restoring through the same route a click takes is also the version that
        cannot drift from it. push_to_canvas() then covers the values that
        happened to equal a widget's default and so emitted nothing.
        """
        for key, _get, setter in self._setting_specs():
            if setter is None or key not in self._settings:
                continue
            try:
                setter(self._settings[key])
            except Exception:
                continue          # one bad key costs one setting
        try:
            self.dlg_graphics.push_to_canvas()
        except Exception:
            pass
        self._settings_saved = self._settings_snapshot()

    def _settings_snapshot(self):
        """Everything worth saving, as JSON-native values."""
        data = {"version": SETTINGS_VERSION}
        for key, getter, _set in self._setting_specs():
            try:
                data[key] = getter()
            except Exception:
                # Carry the last good value forward rather than dropping the
                # key: a getter that throws should not also erase what the
                # operator had.
                if key in self._settings_saved:
                    data[key] = self._settings_saved[key]
        return data

    def _save_settings(self, data=None):
        if data is None:
            data = self._settings_snapshot()
        if save_settings(data):
            self._settings_saved = data

    def _autosave_settings(self):
        """Periodic write, but only when something actually changed.

        Polled off the table rather than wired to thirty valueChanged signals,
        for two reasons: adding a setting stays one line, and a slider drag
        emits fifty changes a second, not one of which is worth a disk write.
        Comparing a ~30-entry dict every ten seconds is free next to the 60 ms
        render tick.
        """
        snap = self._settings_snapshot()
        if snap != self._settings_saved:
            self._save_settings(snap)

    def closeEvent(self, event):
        # First, before anything else: everything below this waits on threads
        # that may not stop and calls into vendor code that may not return, and
        # the settings must not be the casualty of a device that will not let go.
        self._settings_timer.stop()
        self._save_settings()
        self._display_timer.stop()
        if self._perf_timer is not None:
            self._perf_timer.stop()   # report() walks the widget tree we tear down
        # Both of these drive the shared stage/spectrometer, and their timeouts
        # have to allow for the work actually in flight — a flat 2 s expired
        # during any exposure longer than that, leaving a QThread to be
        # destroyed with run() still on its stack.
        self._feed.stop()
        released = self._feed.wait(int(self._feed.join_timeout_ms()))
        if not released:
            _ORPHANED_THREADS.append(self._feed)
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            if not self._worker.wait(int(self._scan_join_timeout_ms())):
                _ORPHANED_THREADS.append(self._worker)
                released = False
        # The alignment sweep has no abort — it is four points long and its
        # last act is putting the stage back, which is exactly what must not be
        # cut short. Waited out on the same per-point budget as a scan, with
        # room for the four moves plus the return.
        if self._align_worker is not None and self._align_worker.isRunning():
            if not self._align_worker.wait(int(5 * self._scan_join_timeout_ms())):
                _ORPHANED_THREADS.append(self._align_worker)
                released = False
        # A half-written export would be a corrupt file, and the worker holds a
        # live reference to the result — let it finish before tearing down.
        if self._export_worker is not None and self._export_worker.isRunning():
            self.status.showMessage("Finishing export…", 0)
            if not self._export_worker.wait(30000):
                _ORPHANED_THREADS.append(self._export_worker)
        # Only once nothing is inside the drivers any more: closing a device
        # out from under a thread still in acquire() faults in vendor code.
        if released:
            for dev in (self.stage, self.spec):
                if dev is None:
                    continue          # nothing was ever connected here
                try:
                    dev.disconnect()
                except Exception:
                    pass
        else:
            print("warning: a device thread did not stop in time; leaving the "
                  "hardware to be released by process exit rather than closing "
                  "it underneath a running acquire.", file=sys.stderr)
        self.dlg_settings.close(); self.dlg_spec.close()
        self.dlg_stage.close(); self.dlg_sim.close()
        self.dlg_graphics.close(); self.dlg_avantes.close()
        self.dlg_export.close()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Without this the taskbar groups us under the Python launcher and shows its
    # icon when run from source. Microsoft specifies the form
    # Company.Product.SubProduct.Version — a bare "Lillypad" (what this used to
    # be) is not a valid AppUserModelID. The shell caches a taskbar icon per
    # AUMID, so changing the string also abandons any blank icon cached against
    # the old one. Must run before the first window exists.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "AlcortaFleischmann.Lillypad")
    except (AttributeError, OSError):
        pass  # not Windows — silently skip
    app = QApplication(sys.argv)
    app.setApplicationName("Lillypad")
    app.setWindowIcon(app_icon())
    app.setFont(QFont("Segoe UI", 10, QFont.Normal))
    # Theme before the first widget exists: PALETTE is sampled at construction
    # time by the figure and by the per-theme icon files _build_ui picks, so a
    # theme restored after the window was built would mean restyling the whole
    # application at startup and trusting every widget that read a colour on the
    # way up to have a refresh hook. _apply_theme still owns the runtime switch,
    # and shares install_theme with this. load_settings() is cached, so
    # FrogWindow reads the same dict.
    install_theme(app, startup_theme(load_settings()))
    seed_calibration_dir()

    win = FrogWindow()
    # Maximized, not fullscreen: the plots are the point and they scale with
    # whatever room there is, but the operator still needs the title bar and
    # the taskbar to get at the acquisition software beside this one. Still the
    # default — but a session that deliberately left the window a particular
    # size on a particular monitor gets it back. See show_restored.
    win.show_restored()
    # The startup graph — every widget, artist, stylesheet rule and Qt binding —
    # is permanent, but the live loop allocates numpy arrays continuously, so
    # gen-2 collections run regularly and would otherwise walk all of it every
    # time. freeze() moves it out of reach of the collector for good, which is
    # what keeps GC pauses from growing with session length.
    gc.freeze()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
