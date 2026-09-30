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

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QGridLayout,
                               QGroupBox, QLabel, QCheckBox, QPushButton,
                               QProgressBar, QScrollArea, QSizePolicy,
                               QStatusBar, QToolBar)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QRect
from PySide6.QtGui import QFont, QIcon

from lillypad.paths import seed_calibration_dir
from lillypad.settings import load_settings
# The design layer. No colour, font size or icon path is defined in this file any
# more — see lillypad/design/. PALETTE is mutated in place by install_theme, so
# importing the name here follows a runtime theme switch.
from lillypad.design.geometry import HDR_BTN, HDR_ICON
from lillypad.design.icons import MOON_ICON, SUN_ICON, SYMMETRY_ICON, app_icon
from lillypad.design.theme import install_theme, startup_theme
# The plotting canvas. Importing it is also what selects matplotlib's QtAgg
# backend and installs the rcParams, so nothing in this file imports matplotlib
# at all any more — see lillypad/gui/canvas/figure.py.
from lillypad.gui.canvas import FrogCanvas
from lillypad.gui.controllers import (AlignmentController, AppearanceController,
                                      DeviceController, ExportController,
                                      LiveFeedController, ScanController,
                                      SettingsController, StageController)
from lillypad.gui.dialogs.acquisition import AcquisitionSettingsDialog
from lillypad.gui.dialogs.alignment import AlignmentDialog
from lillypad.gui.dialogs.avantes import AvantesSettingsDialog
from lillypad.gui.dialogs.export import EXPORT_FORMATS, ExportDialog
from lillypad.gui.dialogs.graphics import GraphicsSettingsDialog
from lillypad.gui.dialogs.simulation import SimulationDialog
from lillypad.gui.dialogs.spectrometer import SpectrometerDialog
from lillypad.gui.dialogs.stage import StageDialog
from lillypad.gui.perf import PERF, install_probe
from lillypad.gui.state import InstrumentState
from lillypad.gui.widgets import (DoubleSpinBox, NoDeviceOverlay, StatusLamp,
                                  _hline)
from lillypad.gui.text import RIGHT_CLICK_HINT
from lillypad.gui.workers import (MAX_UI_EXPOSURE_MS, _ORPHANED_THREADS,
                                  LiveFeedWorker)


# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Everything that used to sit between here and FrogWindow has moved into the
# package, and each banner it left behind was a heading with nothing under it:
#
#   the path and settings helpers  → lillypad/paths.py, lillypad/settings.py
#   colours, fonts, icons, the QSS → lillypad/design/
#   the small widgets             → lillypad/gui/widgets.py
#   the performance probe         → lillypad/gui/perf.py
#   the three worker threads      → lillypad/gui/workers.py
#   the nine dialogs              → lillypad/gui/dialogs/
#   FrogCanvas                    → lillypad/gui/canvas/
#
# What is left below is the window itself: the state it owns, the controls it
# builds, and the slots that connect the two.
# ─────────────────────────────────────────────────────────────────────────────


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

        # ── The controllers ───────────────────────────────────────────────
        # One per concern, each holding a reference to this window. They are a
        # grouping of BEHAVIOUR, not an encapsulation boundary: the widgets stay
        # on the window and the instrument stays in InstrumentState, so a
        # controller reads `self.btn_feed` and `self.spec` exactly where
        # the flat method read `self.btn_feed` and `self.spec`.
        #
        # That is deliberate. This was 4,000 lines relocated out of one class,
        # and giving each controller private state would have meant redesigning
        # every coupling at the same time as moving it. The couplings here are
        # not accidental — a device swap has to re-arm the saturation lamp,
        # refit the spectrum, reseed the exposure boxes and repaint eight
        # buttons, and those genuinely are four concerns having to agree. They
        # are now visible as `self.win.<other>.<call>`, which is also the list
        # of things to think about before changing one.
        #
        # Constructed BEFORE _build_ui, whose lambdas connect straight to
        # controller methods. None of them touches a widget on construction.
        self.devices      = DeviceController(self)
        self.appearance   = AppearanceController(self)
        self.feed         = LiveFeedController(self)
        self.alignment    = AlignmentController(self)
        self.stage_ctl    = StageController(self)
        self.scanning     = ScanController(self)
        self.export       = ExportController(self)
        self.settings_ctl = SettingsController(self)

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
        self.dlg_settings.spin_sat.valueChanged.connect(self.feed._on_sat_fraction)

        self._build_ui()

        # No parking move and no integration push here any more: there is no
        # device to make them to. Both happen on connect instead —
        # _apply_spectrometer clamps the exposure into the new device's range,
        # and _apply_stage parks or adopts the position.
        self.stage_warning.connect(self.devices._on_stage_warning)
        self.appearance._sync_connection_ui()

        # FIX 2/3 — acquisition runs on a worker thread, paced to integration.
        self._feed = LiveFeedWorker(lambda: self.spec)
        self._feed.spectrum_ready.connect(self.feed._on_spectrum)
        self._feed.acquire_failed.connect(self.feed._on_feed_error)
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
        self._display_timer.timeout.connect(self.feed._display_tick)
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
        self.settings_ctl._restore_settings()
        # Closing is not the only way this program ends: a vendor driver can
        # take the process with it (see the warning in closeEvent), and losing a
        # session of preferences to a fault in someone else's DLL is exactly the
        # annoyance persistence is here to remove. Ten seconds is the most that
        # can be lost, and the write only happens when something changed.
        self._settings_timer = QTimer(self)
        self._settings_timer.setInterval(10_000)
        self._settings_timer.timeout.connect(self.settings_ctl._autosave_settings)
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
        tb.addWidget(self.devices._build_calibration_button())
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
        self.btn_theme.clicked.connect(self.appearance._toggle_theme)
        tb.addWidget(self.btn_theme)
        # Save the plots as a picture, between the two controls that decide what
        # those plots LOOK like — the theme and the layout. It belongs with them
        # rather than beside Export: Export writes the measurement, in formats
        # another program reads, while this writes the figure, for a slide or a
        # page. Icon from _refresh_picture_icon (one file per theme).
        self.btn_picture = QPushButton()
        self.btn_picture.setIconSize(QSize(18, 18))
        self.btn_picture.setFixedWidth(42)
        self.appearance._refresh_picture_icon()
        self.btn_picture.clicked.connect(self.appearance._save_picture)
        tb.addWidget(self.btn_picture)
        # Plot layout toggle; icon and tooltip come from _refresh_layout_button.
        self.btn_layout = QPushButton()
        self.btn_layout.setIconSize(QSize(18, 18))
        self.btn_layout.setFixedWidth(42)
        self.btn_layout.clicked.connect(self.appearance._toggle_layout)
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
        self.appearance._refresh_autofit_icon()
        self.btn_autofit.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_autofit.setToolTip(
            "Auto-fit spectrum X and Y axes to current data.\n"
            + RIGHT_CLICK_HINT)
        self.btn_autofit.show()
        self.btn_autofit.clicked.connect(self.feed._autofit_spectrum)

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
        self.appearance._refresh_autoscale_icon()
        self.btn_autoscale.show()
        self.btn_autoscale.clicked.connect(self.feed._on_autoscale_clicked)

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
        self.appearance._refresh_autostitch_icon()
        self.btn_autostitch.hide()
        self.btn_autostitch.clicked.connect(
            lambda: self.devices._menu_result(self.devices.fit_stitch_factor))

        # Live-feed start/stop. In the spectrum header rather than the side
        # panel: it is the control most often reached for while watching that
        # panel, and it was costing the panel a full-width button. Icon and
        # colour come from _refresh_feed_button.
        self.btn_feed = QPushButton(self.canvas)
        self.btn_feed.setCheckable(True); self.btn_feed.setChecked(True)
        self.btn_feed.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_feed.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.btn_feed.show()
        self.btn_feed.toggled.connect(self.feed._toggle_feed)
        self.appearance._refresh_feed_button()

        # Freeze the spectrum as a grey reference, next to the feed whose newest
        # frame it captures. A generated glyph rather than a pair of PNGs, like
        # that neighbour: a plus is two bars, and drawing it from the live
        # PALETTE saves two files that would be wrong the moment an accent moved.
        self.btn_freeze = QPushButton(self.canvas)
        self.btn_freeze.setObjectName("overlay")
        self.btn_freeze.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_freeze.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.appearance._refresh_freeze_icon()
        self.btn_freeze.setToolTip(
            "Freeze the spectrum as a grey reference under the live curve — "
            "make an adjustment, and you can see exactly what it changed.\n"
            "Press again to replace the reference with the current spectrum, "
            "or right-click to drop it.")
        self.btn_freeze.show()
        self.btn_freeze.clicked.connect(self.feed._freeze_reference)
        # Dropping a reference has no button of its own: it is the rare half of
        # a pair whose common half (hide) is beside it already, and a third
        # header button for it would cost every session the room to say so.
        self.btn_freeze.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_freeze.customContextMenuRequested.connect(
            lambda _pos: self.feed._clear_reference())

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
        self.btn_refshow.clicked.connect(self.feed._toggle_reference)

        # Per-spectrometer view toggle. Only meaningful for a stitched pair, so
        # it stays hidden otherwise; its icon and tooltip are set by
        # _refresh_overlay_button.
        self.btn_overlay = QPushButton(self.canvas)
        self.btn_overlay.setObjectName("overlay")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_overlay.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.btn_overlay.hide()
        self.btn_overlay.toggled.connect(self.feed._on_overlay_toggled)

        # Alignment mode, spectrum side: opens the deviation panel under the
        # spectrum, steps to +/-x and +/-2x, and plots the two differences there.
        self.btn_align_spec = QPushButton(self.canvas)
        self.btn_align_spec.setObjectName("overlay")
        self.btn_align_spec.setCheckable(True)
        self.btn_align_spec.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_align_spec.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.appearance._refresh_align_button()
        self.btn_align_spec.setToolTip(
            "Alignment mode — opens a deviation panel under the spectrum, then "
            "measures at −2x, −x, +x, +2x (Alignment → Alignment step) and "
            "plots S(+x)−S(−x) and S(+2x)−S(−2x) in it.\nA symmetric pulse "
            "gives two flat curves on zero. Use ⟳ to re-measure after an "
            "adjustment — the previous sweep stays as a shaded area to compare "
            "against.\nPress again to close the panel.")
        self.btn_align_spec.show()
        self.btn_align_spec.toggled.connect(self.alignment._on_align_spec_toggled)

        # Re-run the sweep without leaving the mode. Only meaningful while the
        # panel is open, so it lives and dies with it — and the comparison it
        # makes possible (this sweep against the last) is the reason the panel
        # keeps a ghost at all.
        self.btn_align_refresh = QPushButton(self.canvas)
        self.btn_align_refresh.setObjectName("overlay")
        self.btn_align_refresh.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_align_refresh.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self.appearance._refresh_align_refresh_icon()
        self.btn_align_refresh.setToolTip(
            "Re-run the alignment sweep. The sweep now on the panel is kept as "
            "a shaded area underneath, so you can see whether the last "
            "adjustment made the deviation better or worse.")
        self.btn_align_refresh.hide()
        self.btn_align_refresh.clicked.connect(self.alignment._start_align_sweep)

        # Alignment mode, trace side — the trace panel's own header.
        # Wider than the rest: it is the one button still carrying a word.
        self.btn_align_trace = QPushButton("RAW", self.canvas)
        self.btn_align_trace.setObjectName("overlay")
        self.btn_align_trace.setCheckable(True)
        self.btn_align_trace.setFixedSize(40, HDR_BTN)
        self.btn_align_trace.hide()
        self.btn_align_trace.toggled.connect(self.alignment._on_align_trace_toggled)

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
        self.btn_symmetry.toggled.connect(self.alignment._on_symmetry_toggled)
        self.appearance._refresh_symmetry_button()

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

        self.canvas.axes_relaid.connect(self.appearance._position_panel_buttons)
        self.appearance._position_panel_buttons()

        # The reference's toggle is an INDICATOR as much as a control, and the
        # canvas is what owns the state — a freeze from the button, a drop from
        # a device swap. One connection covers every way in, so the eye cannot
        # end up describing a reference that is not there.
        self.canvas.reference_changed.connect(self.appearance._refresh_reference_button)
        self.appearance._refresh_reference_button()

        self.appearance._refresh_layout_button()    # needs the canvas for the current mode

        self.dlg_graphics = GraphicsSettingsDialog(self.canvas, self)
        # Mouse zoom/reset keeps the dialog's spinboxes and auto-scale
        # checkboxes truthful; the y-axis click routes through chk_log so the
        # checkbox stays the single source of truth for the log scale.
        self.canvas.limits_changed.connect(self.dlg_graphics.sync_limits)
        self.canvas.log_toggle_requested.connect(self.dlg_graphics.chk_log.toggle)
        self.canvas.proportions_changed.connect(self.dlg_graphics.sync_proportions)
        self.canvas.fold_dragged.connect(self.alignment._on_fold_dragged)
        self.canvas.fold_reset_requested.connect(self.alignment._on_fold_reset)
        self.canvas.axis_edit_failed.connect(
            lambda msg: self.status.showMessage(msg, 4000))
        # The auto-scale toggle is an INDICATOR as much as a control, so it has
        # to follow the state however it was reached: limits_changed covers the
        # right-click and every zoom, the checkbox covers Graphics Settings.
        self.canvas.limits_changed.connect(self.appearance._refresh_autoscale_button)
        self.dlg_graphics.chk_auto_y.toggled.connect(self.appearance._refresh_autoscale_button)
        self.appearance._refresh_autoscale_button()
        # Same arrangement for the scale: the checkbox owns it, and everything
        # that can change it (this button, the y-strip click, the dialog, a
        # restored setting) goes through chk_log, so one connection keeps the
        # label honest.
        self.dlg_graphics.chk_log.toggled.connect(self.appearance._refresh_logscale_button)
        self.appearance._refresh_logscale_button()

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
        self.feed.reset_saturation()
        self.status.showMessage(
            "No hardware connected — connect a spectrometer and a stage, or "
            "open Simulation to run without either.", 0)



















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
        self._integration_timer.timeout.connect(self.feed._apply_integration_time)
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
        self.btn_dark.clicked.connect(self.feed._capture_dark)
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
        self.btn_minus.clicked.connect(lambda: self.stage_ctl._jog(-1))
        self.btn_plus.clicked.connect(lambda: self.stage_ctl._jog(+1))
        self.spin_step = DoubleSpinBox()
        self.spin_step.setDecimals(0)
        self.spin_step.setSuffix(" fs")     # value set after ranges, below
        self.btn_units = QPushButton("fs")
        self.btn_units.setObjectName("unit")   # trimmed padding — see stylesheet
        self.btn_units.setFixedWidth(42)
        self.btn_units.setToolTip("Toggle jog/move units between optical delay (fs) "
                                  "and stage position (um)")
        self.btn_units.clicked.connect(self.stage_ctl._toggle_stage_units)
        jog.addWidget(self.btn_minus); jog.addWidget(self.spin_step); jog.addWidget(self.btn_plus)
        jog.addWidget(self.btn_units)
        lay.addLayout(jog)

        mv = QHBoxLayout()
        self.spin_moveto = DoubleSpinBox()
        self.spin_moveto.setDecimals(0)
        self.spin_moveto.setSuffix(" fs")
        self.btn_moveto = QPushButton("Move")
        self.btn_moveto.clicked.connect(self.stage_ctl._move_absolute)
        mv.addWidget(self.spin_moveto); mv.addWidget(self.btn_moveto)
        lay.addLayout(mv)

        self.btn_home = QPushButton("Home")
        self.btn_home.clicked.connect(self.stage_ctl._home_stage)
        self.btn_goto_zero = QPushButton("Move to 0 fs")
        self.btn_goto_zero.clicked.connect(self.stage_ctl._move_to_zero)
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
        self.appearance._size_moving_label()
        lay.addLayout(prow)

        self.btn_set_zero = QPushButton("Set Position as 0 fs")
        self.btn_set_zero.setObjectName("accentcompact")
        self.btn_set_zero.clicked.connect(self.stage_ctl._mark_zero)
        lay.addWidget(self.btn_set_zero)

        # Backlash lives in the Stage dialog — it is a property of the
        # connected stage, and _sync_backlash_ui below still seeds it from here.

        # Ranges from whatever stage is loaded, which at build time is none:
        # _travel_range_um falls back to 300 mm of travel, and _apply_stage
        # calls this again with the real numbers on every connect.
        self.stage_ctl._update_stage_unit_ranges()
        self.spin_step.setValue(10.0)      # after ranges: default 10 fs jog
        self.stage_ctl._sync_backlash_ui()

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
            s.valueChanged.connect(self.stage_ctl._refresh_scan_um)

        # The scan and its export on one line, in the order they are used.
        # btn_scan keeps the wider share: it also carries "Abort Scan" while a
        # scan runs, and it is the button the whole panel exists for.
        arow = QHBoxLayout(); arow.setSpacing(8)
        self.btn_scan = QPushButton("Measure FROG")
        self.btn_scan.setObjectName("accent")
        self.btn_scan.setMinimumHeight(30)
        self.btn_scan.clicked.connect(self.scanning._start_scan)
        arow.addWidget(self.btn_scan, 3)

        self.btn_save = QPushButton(f"Save ({EXPORT_FORMATS[self._export_fmt][1]})")
        self.btn_save.setEnabled(False)
        self.btn_save.setMinimumHeight(30)
        # Straight to the write, in the format and under the name the Export
        # window is showing — this button is the shortcut past that window, not
        # a second set of choices.
        self.btn_save.clicked.connect(self.export.save_now)
        arow.addWidget(self.btn_save, 2)
        lay.addLayout(arow)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setFixedHeight(18)
        lay.addWidget(self.progress)

        # The AC FWHM is reported inside the autocorrelation panel itself
        # (FrogCanvas.set_fwhm), next to the curve it measures.

        self.stage_ctl._refresh_scan_um()
        return grp




























































    # ── Symmetry mode ────────────────────────────────────────────────────────
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





    def closeEvent(self, event):
        # First, before anything else: everything below this waits on threads
        # that may not stop and calls into vendor code that may not return, and
        # the settings must not be the casualty of a device that will not let go.
        self._settings_timer.stop()
        self.settings_ctl._save_settings()
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
            if not self._worker.wait(int(self.scanning.join_timeout_ms())):
                _ORPHANED_THREADS.append(self._worker)
                released = False
        # The alignment sweep has no abort — it is four points long and its
        # last act is putting the stage back, which is exactly what must not be
        # cut short. Waited out on the same per-point budget as a scan, with
        # room for the four moves plus the return.
        if self._align_worker is not None and self._align_worker.isRunning():
            if not self._align_worker.wait(int(5 * self.scanning.join_timeout_ms())):
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
