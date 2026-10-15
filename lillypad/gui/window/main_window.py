"""FrogWindow — the instrument's one window.

What it is, after the split: the object that OWNS things and wires them
together. It holds

  * the instrument, in `self.state` (see lillypad/gui/state.py),
  * the display scratch — render buffers, pending-fit flags, the alignment view
    — which belongs to one panel each and is not shared,
  * the ~200 widgets build_ui makes,
  * the eight controllers, which are where the behaviour went,
  * the nine dialogs,
  * and the three timers: the 60 ms display tick, the 10 s settings autosave,
    and the optional performance probe.

Its own methods are only the ones that are genuinely the window's: the property
forwarders onto InstrumentState, the three connection predicates, the live-feed
switch, showing itself where the last session left it, and shutting down.

__init__ IS the wiring diagram, and its order is load-bearing throughout —
state, then the controllers, then the dialogs, then build_ui, then the feed and
the timers, then _restore_settings last of all. Each step is commented with why
it sits where it does. Read it once and the rest of lillypad/gui/ has a shape.
"""

import sys

from PySide6.QtCore import QRect, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMainWindow

from lillypad.design.icons import app_icon
from lillypad.design.theme import startup_theme
from lillypad.gui.controllers import (AlignmentController, AppearanceController,
                                      DeviceController, ExportController,
                                      LiveFeedController, ScanController,
                                      SettingsController, StageController)
from lillypad.gui.dialogs.acquisition import AcquisitionSettingsDialog
from lillypad.gui.dialogs.alignment import AlignmentDialog
from lillypad.gui.dialogs.avantes import AvantesSettingsDialog
from lillypad.gui.dialogs.export import ExportDialog
from lillypad.gui.dialogs.simulation import SimulationDialog
from lillypad.gui.dialogs.spectrometer import SpectrometerDialog
from lillypad.gui.dialogs.stage import StageDialog
from lillypad.gui.perf import PERF, install_probe
from lillypad.gui.state import InstrumentState
from lillypad.gui.window.build_ui import build_ui
from lillypad.gui.workers import _ORPHANED_THREADS, LiveFeedWorker
from lillypad.settings import load_settings


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

        build_ui(self)

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
        """Show the window maximized, over where the last session left it.

        Always maximized, whatever the last session closed as — the saved
        rectangle is still restored first, so un-maximizing lands on the size
        the operator last chose rather than on some size nobody did.

        Neither in __init__ nor as two calls in main(): setGeometry has to
        happen before the first show to be honoured without a visible jump.

        Explicit x/y/w/h rather than a base64 saveGeometry() blob, because this
        file is meant to be readable and fixable by hand — an opaque blob cannot
        be corrected when a window comes back on a monitor that is no longer
        there, and restoreGeometry rejects blobs across Qt versions without
        saying so.
        """
        geo = self._settings.get("window")
        if isinstance(geo, dict):
            try:
                r = QRect(int(geo["x"]), int(geo["y"]),
                          int(geo["w"]), int(geo["h"]))
            except (KeyError, TypeError, ValueError):
                r = QRect()
            if r.isValid() and _on_a_screen(r):
                self.setGeometry(r)        # Qt clamps up to setMinimumSize itself
        self.showMaximized()





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
