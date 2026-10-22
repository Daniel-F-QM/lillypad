"""Building the window: the toolbar, the three side-panel groups, the canvas.

Free functions taking `win`, not a class. `build_ui` is 375 lines of pure
construction that assigns some two hundred attributes onto the window; wrapping
that in an object would have added `self.win.` to every one of those lines and
bought nothing, because there is no state here to hold — the window IS the
state, and this file is the one-time act of filling it in.

Read it top to bottom and you have the whole interface: the toolbar, then the
panel header buttons over each plot, then the three groups down the left side
(Spectrum, Stage, FROG Scan), then the status bar. Every signal is connected
straight to the controller that owns the behaviour, so a control whose name you
can see here tells you which file to open next.

Three things not to tidy:

  * **The order is load-bearing.** The canvas exists before the panel buttons
    that sit over it, the Graphics window is built after the canvas it drives,
    and _restore_settings runs after all of it (in __init__, not here).
  * **`hasattr(win, "dlg_graphics")` is an init-order guard, not defensiveness.**
    Methods called during construction touch a dialog that does not exist yet.
    Turning one into a plain attribute access is an AttributeError at startup.
  * **`_connect_page` swaps a group between its controls and a single Connect
    button.** Both pages are built once and the group shows one of them, which
    is why nothing here is created lazily.
"""

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QStatusBar, QToolBar, QVBoxLayout,
                               QWidget)

from lillypad.design.geometry import HDR_BTN, HDR_ICON
from lillypad.design.icons import MOON_ICON, SUN_ICON
from lillypad.gui.canvas import FrogCanvas
from lillypad.gui.dialogs.export import EXPORT_FORMATS
from lillypad.gui.dialogs.graphics import GraphicsSettingsDialog
from lillypad.gui.text import RIGHT_CLICK_HINT
from lillypad.gui.widgets import (DoubleSpinBox, NoDeviceOverlay, StatusLamp,
                                  TraceWatermark,
                                  _hline)
from lillypad.gui.workers import MAX_UI_EXPOSURE_MS


# ── UI ───────────────────────────────────────────────────────────────────
def build_ui(win):
    tb = QToolBar("Main"); tb.setMovable(False)
    win.addToolBar(tb)
    b_set = QPushButton("Acquisition Settings")
    b_set.clicked.connect(win.dlg_settings.toggle)
    tb.addWidget(b_set)
    # One button per device, replacing the old combined Hardware button.
    # They are the same dialogs the panel Connect buttons open — this row is
    # the way back to them once something IS connected and the panels have
    # swapped over to their controls.
    win.btn_spec_dlg = QPushButton("Spectrometer")
    win.btn_spec_dlg.setToolTip(
        "Connect one or two spectrometers, assign calibrations, and manage "
        "the stitch between a pair")
    win.btn_spec_dlg.clicked.connect(win.dlg_spec.toggle)
    tb.addWidget(win.btn_spec_dlg)
    win.btn_stage_dlg = QPushButton("Stage")
    win.btn_stage_dlg.setToolTip("Connect a delay stage and set its "
                                  "backlash approach margin")
    win.btn_stage_dlg.clicked.connect(win.dlg_stage.toggle)
    tb.addWidget(win.btn_stage_dlg)
    # Its own button, deliberately apart from the two above: simulated
    # hardware is no longer a default anything, and nothing that reads as a
    # hardware control should be able to hand back synthetic data.
    win.btn_sim_dlg = QPushButton("Simulation")
    win.btn_sim_dlg.setToolTip(
        "Run against simulated hardware — a synthetic beam, spectrometer, "
        "stage or stitched pair, with no instrument attached")
    win.btn_sim_dlg.clicked.connect(win.dlg_sim.toggle)
    tb.addWidget(win.btn_sim_dlg)
    b_gfx = QPushButton("Graphics Settings")
    b_gfx.clicked.connect(lambda: win.dlg_graphics.toggle())
    tb.addWidget(b_gfx)
    b_align = QPushButton("Alignment")
    b_align.setToolTip("Alignment-mode sweep settings — the step the Δ "
                       "button over the spectrum measures at")
    b_align.clicked.connect(win.dlg_align.toggle)
    tb.addWidget(b_align)
    # A window, like its neighbours — not the drop-down of formats it used
    # to be. See ExportDialog for why.
    win.btn_export = QPushButton("Export")
    win.btn_export.setToolTip(
        "Save the last scan — pick the format and the file name")
    win.btn_export.clicked.connect(win.dlg_export.toggle)
    tb.addWidget(win.btn_export)
    tb.addWidget(win.devices._build_calibration_button())
    # Vendor-specific, so it is HIDDEN rather than disabled when there is
    # no Avantes attached — the toolbar already carries six buttons, and a
    # permanently greyed one is noise on a bench that has none. Visibility
    # is owned by _refresh_avantes_button.
    win.btn_avantes = QPushButton("Avantes")
    win.btn_avantes.setToolTip(
        "Avantes-specific settings: averaging, ADC resolution, "
        "corrections, triggering, detector temperature")
    win.btn_avantes.clicked.connect(win.dlg_avantes.toggle)
    win.btn_avantes.setVisible(False)
    tb.addWidget(win.btn_avantes)
    spacer = QWidget()
    spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    tb.addWidget(spacer)
    win.btn_theme = QPushButton()
    # Derived from win._theme rather than hard-coded to the dark-mode pair:
    # the theme is restored from settings.json before _build_ui runs, so a
    # session that ended in light mode would otherwise come back showing the
    # sun and offering to switch to the mode it is already in. Same
    # expressions as _apply_theme, which owns the runtime switch — the
    # button always advertises what a CLICK produces.
    win.btn_theme.setIcon(QIcon(str(MOON_ICON if win._theme == "light"
                                     else SUN_ICON)))
    win.btn_theme.setIconSize(QSize(18, 18))
    win.btn_theme.setFixedWidth(42)
    win.btn_theme.setToolTip("Switch to dark mode" if win._theme == "light"
                              else "Switch to light mode")
    win.btn_theme.clicked.connect(win.appearance._toggle_theme)
    tb.addWidget(win.btn_theme)
    # Save the plots as a picture, beside the theme — the control that
    # decides what those plots LOOK like. It belongs with it rather than
    # beside Export: Export writes the measurement, in formats another
    # program reads, while this writes the figure, for a slide or a page.
    # Icon from _refresh_picture_icon (one file per theme).
    win.btn_picture = QPushButton()
    win.btn_picture.setIconSize(QSize(18, 18))
    win.btn_picture.setFixedWidth(42)
    win.appearance._refresh_picture_icon()
    win.btn_picture.clicked.connect(win.appearance._save_picture)
    tb.addWidget(win.btn_picture)

    central = QWidget()
    win.setCentralWidget(central)
    root = QHBoxLayout(central)
    root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

    ctrl = QWidget()
    ctrl.setMinimumWidth(280); ctrl.setMaximumWidth(320)
    cl = QVBoxLayout(ctrl); cl.setSpacing(6); cl.setContentsMargins(0, 0, 4, 0)
    cl.addWidget(build_spectrum_group(win))
    cl.addWidget(build_stage_group(win))
    cl.addWidget(build_scan_group(win))
    cl.addStretch()

    scroll = QScrollArea()
    scroll.setWidget(ctrl); scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFixedWidth(312)
    root.addWidget(scroll)

    win.canvas = FrogCanvas()
    root.addWidget(win.canvas, stretch=1)

    # ── Panel header buttons ─────────────────────────────────────────────
    # All children of the canvas so they float over the figure margin, and
    # all positioned by _position_panel_buttons: each one belongs to the
    # panel it acts on, and every panel edge moves with the split fraction
    # and the window size.
    win.btn_autofit = QPushButton(win.canvas)
    win.btn_autofit.setObjectName("overlay")
    win.btn_autofit.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.appearance._refresh_autofit_icon()
    win.btn_autofit.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_autofit.setToolTip(
        "Auto-fit spectrum X and Y axes to current data.\n"
        + RIGHT_CLICK_HINT)
    win.btn_autofit.show()
    win.btn_autofit.clicked.connect(win.feed._autofit_spectrum)

    # Continuous auto-scale, next to the one-shot fit it complements: that
    # button fits ONCE, this one says whether the panel goes on following
    # the data. It is the third way in to the same state — Graphics
    # Settings' checkbox and a right-click on the panel are the other two —
    # so it drives reset_axes(), the very function the right-click calls,
    # rather than a parallel path that could drift out of step with it.
    # Its own checked state is never set here: _refresh_autoscale_button
    # reads it back off the canvas, whichever of the three moved it.
    win.btn_autoscale = QPushButton(win.canvas)
    win.btn_autoscale.setObjectName("overlay")
    win.btn_autoscale.setCheckable(True)
    win.btn_autoscale.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_autoscale.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.appearance._refresh_autoscale_icon()
    win.btn_autoscale.show()
    win.btn_autoscale.clicked.connect(win.feed._on_autoscale_clicked)

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
    win.btn_logscale = QPushButton(win.canvas)
    win.btn_logscale.setObjectName("overlay")
    win.btn_logscale.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_logscale.show()
    # Through the checkbox, not canvas.set_log_scale: chk_log is the single
    # source of truth for the scale (it is what gets persisted, and what the
    # y-strip click already routes through), so driving the canvas directly
    # would leave the dialog showing the wrong thing.
    # Lambda, not a direct connect: dlg_graphics is built further down.
    win.btn_logscale.clicked.connect(
        lambda: win.dlg_graphics.chk_log.toggle())

    # Auto-stitch, in the spectrum header rather than only in the
    # Spectrometer window: it is fitted against what is on the panel right
    # now (it needs light across the overlap), so it belongs next to the
    # curve you are judging it by. Pair-only, like the overlay toggle —
    # _refresh_autostitch_button owns its visibility, and its icon comes
    # from _refresh_autostitch_icon (per-theme, like the alignment mark).
    win.btn_autostitch = QPushButton(win.canvas)
    win.btn_autostitch.setObjectName("overlay")
    win.btn_autostitch.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_autostitch.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.appearance._refresh_autostitch_icon()
    win.btn_autostitch.hide()
    win.btn_autostitch.clicked.connect(
        lambda: win.devices._menu_result(win.devices.fit_stitch_factor))

    # Live-feed start/stop. In the spectrum header rather than the side
    # panel: it is the control most often reached for while watching that
    # panel, and it was costing the panel a full-width button. Icon and
    # colour come from _refresh_feed_button.
    win.btn_feed = QPushButton(win.canvas)
    win.btn_feed.setCheckable(True); win.btn_feed.setChecked(True)
    win.btn_feed.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_feed.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.btn_feed.show()
    win.btn_feed.toggled.connect(win.feed._toggle_feed)
    win.appearance._refresh_feed_button()

    # Freeze the spectrum as a grey reference, next to the feed whose newest
    # frame it captures. A generated glyph rather than a pair of PNGs, like
    # that neighbour: a plus is two bars, and drawing it from the live
    # PALETTE saves two files that would be wrong the moment an accent moved.
    win.btn_freeze = QPushButton(win.canvas)
    win.btn_freeze.setObjectName("overlay")
    win.btn_freeze.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_freeze.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.appearance._refresh_freeze_icon()
    win.btn_freeze.setToolTip(
        "Freeze the spectrum as a grey reference under the live curve — "
        "make an adjustment, and you can see exactly what it changed.\n"
        "Press again to replace the reference with the current spectrum, "
        "or right-click to drop it.")
    win.btn_freeze.show()
    win.btn_freeze.clicked.connect(win.feed._freeze_reference)
    # Dropping a reference has no button of its own: it is the rare half of
    # a pair whose common half (hide) is beside it already, and a third
    # header button for it would cost every session the room to say so.
    win.btn_freeze.setContextMenuPolicy(Qt.CustomContextMenu)
    win.btn_freeze.customContextMenuRequested.connect(
        lambda _pos: win.feed._clear_reference())

    # …and the reference's own hide/show. Pair-only in the same sense as the
    # alignment sweep's re-run button: it exists exactly as long as there is
    # something for it to act on, so _refresh_reference_button owns both its
    # visibility and its icon.
    win.btn_refshow = QPushButton(win.canvas)
    win.btn_refshow.setObjectName("overlay")
    win.btn_refshow.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_refshow.setIconSize(QSize(HDR_ICON, HDR_ICON))
    # Not checkable: the eye already shows the state by showing the action,
    # and a checked frame around it would be a second readout of the same
    # bit — one that says "on" while the eye says "hide".
    win.btn_refshow.hide()
    win.btn_refshow.clicked.connect(win.feed._toggle_reference)

    # Per-spectrometer view toggle. Only meaningful for a stitched pair, so
    # it stays hidden otherwise; its icon and tooltip are set by
    # _refresh_overlay_button.
    win.btn_overlay = QPushButton(win.canvas)
    win.btn_overlay.setObjectName("overlay")
    win.btn_overlay.setCheckable(True)
    win.btn_overlay.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_overlay.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.btn_overlay.hide()
    win.btn_overlay.toggled.connect(win.feed._on_overlay_toggled)

    # Alignment mode, spectrum side: opens the deviation panel under the
    # spectrum, steps to +/-x and +/-2x, and plots the two differences there.
    win.btn_align_spec = QPushButton(win.canvas)
    win.btn_align_spec.setObjectName("overlaytoggle")
    win.btn_align_spec.setCheckable(True)
    win.btn_align_spec.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_align_spec.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.appearance._refresh_align_button()
    win.btn_align_spec.setToolTip(
        "Alignment mode — opens a deviation panel under the spectrum, then "
        "measures at −2x, −x, +x, +2x (Alignment → Alignment step) and "
        "plots S(+x)−S(−x) and S(+2x)−S(−2x) in it.\nA symmetric pulse "
        "gives two flat curves on zero. Use ⟳ to re-measure after an "
        "adjustment — the previous sweep stays as a shaded area to compare "
        "against.\nPress again to close the panel.")
    win.btn_align_spec.show()
    win.btn_align_spec.toggled.connect(win.alignment._on_align_spec_toggled)

    # Re-run the sweep without leaving the mode. Only meaningful while the
    # panel is open, so it lives and dies with it — and the comparison it
    # makes possible (this sweep against the last) is the reason the panel
    # keeps a ghost at all.
    win.btn_align_refresh = QPushButton(win.canvas)
    win.btn_align_refresh.setObjectName("overlay")
    win.btn_align_refresh.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_align_refresh.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.appearance._refresh_align_refresh_icon()
    win.btn_align_refresh.setToolTip(
        "Re-run the alignment sweep. The sweep now on the panel is kept as "
        "a shaded area underneath, so you can see whether the last "
        "adjustment made the deviation better or worse.")
    win.btn_align_refresh.hide()
    win.btn_align_refresh.clicked.connect(win.alignment._start_align_sweep)

    # Alignment mode, trace side — the trace panel's own header.
    # Wider than the rest: it is the one button still carrying a word.
    win.btn_align_trace = QPushButton("RAW", win.canvas)
    win.btn_align_trace.setObjectName("overlaytoggle")
    win.btn_align_trace.setCheckable(True)
    win.btn_align_trace.setFixedSize(40, HDR_BTN)
    win.btn_align_trace.hide()
    win.btn_align_trace.toggled.connect(win.alignment._on_align_trace_toggled)

    # Symmetry mode, trace panel. Always present — unlike RAW it does not
    # depend on the pair — but only enabled once a finished scan gives it an
    # autocorrelation to find the fold line in.
    win.btn_symmetry = QPushButton(win.canvas)
    win.btn_symmetry.setObjectName("overlaytoggle")
    win.btn_symmetry.setCheckable(True)
    win.appearance._refresh_symmetry_icon()
    win.btn_symmetry.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_symmetry.show()
    win.btn_symmetry.toggled.connect(win.alignment._on_symmetry_toggled)
    win.appearance._refresh_symmetry_button()

    # Smoothing toggle: linear interpolation along the delay axis, display
    # only (FrogCanvas._smooth_columns). On by default.
    win.btn_smooth = QPushButton(win.canvas)
    win.btn_smooth.setObjectName("overlaytoggle")
    win.btn_smooth.setCheckable(True)
    win.btn_smooth.setChecked(win.canvas.trace_smooth)
    win.btn_smooth.setFixedSize(HDR_BTN, HDR_BTN)
    win.btn_smooth.setIconSize(QSize(HDR_ICON, HDR_ICON))
    win.btn_smooth.setToolTip(
        "Smooth the trace along the delay axis (display only — saved data "
        "is never smoothed)")
    win.btn_smooth.show()
    win.btn_smooth.toggled.connect(win.canvas.set_trace_smooth)
    win.appearance._refresh_smooth_icon()

    # The Lilypad mark on the empty trace panel. Lowered beneath the header
    # buttons; placed and shown/hidden by _position_panel_buttons.
    win.trace_mark = TraceWatermark(win.canvas)
    win.trace_mark.lower()

    # ── "No spectrometer connected" ──────────────────────────────────────
    # A Qt child of the canvas, not a matplotlib artist: the canvas blits,
    # so a text artist would have to force a full redraw on every show and
    # hide, and this one has to carry a clickable button anyway. Stretched
    # over the whole spectrum panel by _position_panel_buttons, which
    # already runs on every relayout, resize and layout-mode change.
    win.pnl_no_spec = NoDeviceOverlay(
        win.canvas, "No spectrometer connected", "Connect Spectrometer",
        win.dlg_spec.open_fresh)
    win.pnl_no_spec.hide()

    win.canvas.axes_relaid.connect(win.appearance._position_panel_buttons)
    win.canvas.axes_relaid.connect(win.canvas.refresh_trace_resolution)
    win.appearance._position_panel_buttons()

    # The reference's toggle is an INDICATOR as much as a control, and the
    # canvas is what owns the state — a freeze from the button, a drop from
    # a device swap. One connection covers every way in, so the eye cannot
    # end up describing a reference that is not there.
    win.canvas.reference_changed.connect(win.appearance._refresh_reference_button)
    win.appearance._refresh_reference_button()

    win.dlg_graphics = GraphicsSettingsDialog(win.canvas, win)
    # Mouse zoom/reset keeps the dialog's spinboxes and auto-scale
    # checkboxes truthful; the y-axis click routes through chk_log so the
    # checkbox stays the single source of truth for the log scale.
    win.canvas.limits_changed.connect(win.dlg_graphics.sync_limits)
    win.canvas.log_toggle_requested.connect(win.dlg_graphics.chk_log.toggle)
    win.canvas.proportions_changed.connect(win.dlg_graphics.sync_proportions)
    win.canvas.fold_dragged.connect(win.alignment._on_fold_dragged)
    win.canvas.fold_reset_requested.connect(win.alignment._on_fold_reset)
    win.canvas.axis_edit_failed.connect(
        lambda msg: win.status.showMessage(msg, 4000))
    # The auto-scale toggle is an INDICATOR as much as a control, so it has
    # to follow the state however it was reached: limits_changed covers the
    # right-click and every zoom, the checkbox covers Graphics Settings.
    win.canvas.limits_changed.connect(win.appearance._refresh_autoscale_button)
    win.dlg_graphics.chk_auto_y.toggled.connect(win.appearance._refresh_autoscale_button)
    win.appearance._refresh_autoscale_button()
    # Same arrangement for the scale: the checkbox owns it, and everything
    # that can change it (this button, the y-strip click, the dialog, a
    # restored setting) goes through chk_log, so one connection keeps the
    # label honest.
    win.dlg_graphics.chk_log.toggled.connect(win.appearance._refresh_logscale_button)
    win.appearance._refresh_logscale_button()

    win.status = QStatusBar()
    win.setStatusBar(win.status)
    win.lbl_sat = QLabel(""); win.lbl_sat.setObjectName("satok")
    win.lamp = StatusLamp()
    # Second lamp pair for multi-spectrometer mode — one alarm per device.
    # Hidden until the mode is enabled.
    win.lbl_sat2 = QLabel(""); win.lbl_sat2.setObjectName("satok")
    win.lamp2 = StatusLamp()
    win.status.addPermanentWidget(win.lbl_sat)
    win.status.addPermanentWidget(win.lamp)
    win.status.addPermanentWidget(win.lbl_sat2)
    win.status.addPermanentWidget(win.lamp2)
    win.lbl_sat2.hide(); win.lamp2.hide()
    win.feed.reset_saturation()
    win.status.showMessage(
        "No hardware connected — connect a spectrometer and a stage, or "
        "open Simulation to run without either.", 0)

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

def build_spectrum_group(win):
    grp = QGroupBox("Spectrum")
    # The controls go in `body`, which is hidden wholesale while there is
    # no spectrometer — every one of them drives a device that isn't there.
    body = QWidget()
    lay = QVBoxLayout(body); lay.setSpacing(4)
    lay.setContentsMargins(0, 0, 0, 0)
    # The feed toggle lives in the spectrum panel's header (see _build_ui).
    win.lbl_integration = QLabel("Integration Time")
    lay.addWidget(win.lbl_integration)
    # Floating point, because the range is the DEVICE's: an Avantes goes
    # down to 2 us, so a whole-millisecond box would put most of its usable
    # exposure range out of reach. Range, decimals and step are all seeded
    # from the live device by _seed_integration_spin — an Ocean box that
    # cannot do sub-ms keeps its whole-millisecond look.
    win.spin_integration = DoubleSpinBox()
    win.spin_integration.setDecimals(0)
    win.spin_integration.setRange(1.0, MAX_UI_EXPOSURE_MS)
    win.spin_integration.setValue(10.0)
    win.spin_integration.setSuffix("  ms")
    # Debounced: set_integration_time is a device call, so it has to take
    # the hardware off the feed thread first. Doing that on every spinbox
    # tick would park and restart the feed on each keystroke, so coalesce
    # the edits and apply once the user stops typing.
    win._integration_timer = QTimer(win)
    win._integration_timer.setSingleShot(True)
    win._integration_timer.setInterval(250)
    win._integration_timer.timeout.connect(win.feed._apply_integration_time)
    win.spin_integration.valueChanged.connect(
        lambda _v: win._integration_timer.start())
    # Second exposure, shown only for a stitched pair. The two devices see
    # very different signal levels, so one shared value always leaves one
    # of them either buried in read noise or clipped. S1/S2 are SLOT order
    # — the same numbering as the saturation lamps and the Spectrometer
    # window's two slots.
    #
    # Side by side under ONE caption rather than a second labelled row: a
    # stitched pair is the tallest the panel ever gets, and a caption that
    # reads "S1 / S2" over two boxes says the same thing in half the space.
    win.spin_integration2 = DoubleSpinBox()
    win.spin_integration2.setDecimals(0)
    win.spin_integration2.setRange(1.0, MAX_UI_EXPOSURE_MS)
    win.spin_integration2.setValue(10.0)
    win.spin_integration2.setSuffix("  ms")
    # Same debounce timer as S1 on purpose: editing both boxes then costs
    # ONE feed handover instead of two, and a handover blocks until the
    # in-flight acquire returns.
    win.spin_integration2.valueChanged.connect(
        lambda _v: win._integration_timer.start())
    irow = QHBoxLayout(); irow.setSpacing(6)
    irow.addWidget(win.spin_integration, 1)
    irow.addWidget(win.spin_integration2, 1)
    lay.addLayout(irow)
    win.spin_integration2.setVisible(False)

    lay.addWidget(_hline())
    # Record and use on one line: the checkbox is what you reach for right
    # after the button, and neither needs the panel's full width.
    drow = QHBoxLayout(); drow.setSpacing(8)
    win.btn_dark = QPushButton("Record Dark")
    win.btn_dark.setObjectName("accentcompact")
    win.btn_dark.clicked.connect(win.feed._capture_dark)
    win.chk_dark = QCheckBox("Subtract Dark")
    win.chk_dark.setEnabled(False)
    drow.addWidget(win.btn_dark)
    drow.addWidget(win.chk_dark, 1)
    lay.addLayout(drow)
    # The alignment step lives in the Alignment dialog (toolbar).

    win.btn_connect_spec = QPushButton("Connect Spectrometer")
    win.btn_connect_spec.setObjectName("accent")
    win.btn_connect_spec.setToolTip(
        "Scan for attached spectrometers and connect one — or two, as a "
        "stitched pair")
    win.btn_connect_spec.clicked.connect(win.dlg_spec.open_fresh)
    _connect_page(grp, body, win.btn_connect_spec)
    win._body_spectrum = body
    return grp

def build_stage_group(win):
    grp = QGroupBox("Stage")
    body = QWidget()
    lay = QVBoxLayout(body); lay.setSpacing(4)
    lay.setContentsMargins(0, 0, 0, 0)

    jog = QHBoxLayout()
    win.btn_minus = QPushButton("−"); win.btn_plus = QPushButton("+")
    win.btn_minus.clicked.connect(lambda: win.stage_ctl._jog(-1))
    win.btn_plus.clicked.connect(lambda: win.stage_ctl._jog(+1))
    win.spin_step = DoubleSpinBox()
    win.spin_step.setDecimals(0)
    win.spin_step.setSuffix(" fs")     # value set after ranges, below
    win.btn_units = QPushButton("fs")
    win.btn_units.setObjectName("unit")   # trimmed padding — see stylesheet
    win.btn_units.setFixedWidth(42)
    win.btn_units.setToolTip("Toggle jog/move units between optical delay (fs) "
                              "and stage position (um)")
    win.btn_units.clicked.connect(win.stage_ctl._toggle_stage_units)
    jog.addWidget(win.btn_minus); jog.addWidget(win.spin_step); jog.addWidget(win.btn_plus)
    jog.addWidget(win.btn_units)
    lay.addLayout(jog)

    mv = QHBoxLayout()
    win.spin_moveto = DoubleSpinBox()
    win.spin_moveto.setDecimals(0)
    win.spin_moveto.setSuffix(" fs")
    win.btn_moveto = QPushButton("Move")
    win.btn_moveto.clicked.connect(win.stage_ctl._move_absolute)
    mv.addWidget(win.spin_moveto); mv.addWidget(win.btn_moveto)
    lay.addLayout(mv)

    win.btn_home = QPushButton("Home")
    win.btn_home.clicked.connect(win.stage_ctl._home_stage)
    win.btn_goto_zero = QPushButton("Move to 0 fs")
    win.btn_goto_zero.clicked.connect(win.stage_ctl._move_to_zero)
    zrow = QHBoxLayout()
    zrow.addWidget(win.btn_home); zrow.addWidget(win.btn_goto_zero)
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
    win.lbl_pos  = QLabel("— um"); win.lbl_pos.setObjectName("readout_sm")
    win.lbl_zero = QLabel("— um"); win.lbl_zero.setObjectName("readout_sm")
    win.lbl_moving = QLabel("IDLE"); win.lbl_moving.setObjectName("idle")
    _readout("Current", win.lbl_pos, 1)
    _readout("Zero delay", win.lbl_zero, 1)
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
    _readout("Stage", win.lbl_moving, 0, Qt.AlignHCenter)
    win.appearance._size_moving_label()
    lay.addLayout(prow)

    win.btn_set_zero = QPushButton("Set Position as 0 fs")
    win.btn_set_zero.setObjectName("accentcompact")
    win.btn_set_zero.clicked.connect(win.stage_ctl._mark_zero)
    lay.addWidget(win.btn_set_zero)

    # Backlash lives in the Stage dialog — it is a property of the
    # connected stage, and _sync_backlash_ui below still seeds it from here.

    # Ranges from whatever stage is loaded, which at build time is none:
    # _travel_range_um falls back to 300 mm of travel, and _apply_stage
    # calls this again with the real numbers on every connect.
    win.stage_ctl._update_stage_unit_ranges()
    win.spin_step.setValue(10.0)      # after ranges: default 10 fs jog
    win.stage_ctl._sync_backlash_ui()

    win.btn_connect_stage = QPushButton("Connect Stage")
    win.btn_connect_stage.setObjectName("accent")
    win.btn_connect_stage.setToolTip(
        "Scan for attached delay stages and connect one")
    win.btn_connect_stage.clicked.connect(win.dlg_stage.open_fresh)
    _connect_page(grp, body, win.btn_connect_stage)
    win._body_stage = body
    return grp

def build_scan_group(win):
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

    win.spin_start = DoubleSpinBox()
    win.spin_stop  = DoubleSpinBox()
    win.spin_step_fs = DoubleSpinBox()
    cell(0, 0, "Start", win.spin_start, " fs", 1, -1e6, 1e6, -250.0)
    cell(0, 2, "Stop",  win.spin_stop,  " fs", 1, -1e6, 1e6,  250.0)
    # The um equivalents move under their own boxes — as a full-width pair
    # they would push the group back out to three rows.
    win.eq_start = QLabel(""); win.eq_start.setObjectName("dim")
    win.eq_stop  = QLabel(""); win.eq_stop.setObjectName("dim")
    g.addWidget(win.eq_start, 1, 1)
    g.addWidget(win.eq_stop,  1, 3)
    cell(2, 0, "Step", win.spin_step_fs, " fs", 1, 0.1, 1e5, 1.0)
    lay.addLayout(g)
    for s in (win.spin_start, win.spin_stop):
        s.valueChanged.connect(win.stage_ctl._refresh_scan_um)

    # The scan and its export on one line, in the order they are used.
    # btn_scan keeps the wider share: it also carries "Abort Scan" while a
    # scan runs, and it is the button the whole panel exists for.
    arow = QHBoxLayout(); arow.setSpacing(8)
    win.btn_scan = QPushButton("Measure FROG")
    win.btn_scan.setObjectName("accent")
    win.btn_scan.setMinimumHeight(30)
    win.btn_scan.clicked.connect(win.scanning._start_scan)
    arow.addWidget(win.btn_scan, 3)

    win.btn_save = QPushButton(f"Save ({EXPORT_FORMATS[win._export_fmt][1]})")
    win.btn_save.setEnabled(False)
    win.btn_save.setMinimumHeight(30)
    # Straight to the write, in the format and under the name the Export
    # window is showing — this button is the shortcut past that window, not
    # a second set of choices.
    win.btn_save.clicked.connect(win.export.save_now)
    arow.addWidget(win.btn_save, 2)
    lay.addLayout(arow)

    win.progress = QProgressBar()
    win.progress.setRange(0, 100); win.progress.setValue(0)
    win.progress.setFixedHeight(18)
    lay.addWidget(win.progress)

    # The AC FWHM is reported inside the autocorrelation panel itself
    # (FrogCanvas.set_fwhm), next to the curve it measures.

    win.stage_ctl._refresh_scan_um()
    return grp
