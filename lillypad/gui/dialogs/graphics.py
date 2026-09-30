"""Plot settings: axis bounds, autoscale, log scale, colormap, proportions.

The one dialog that takes the CANVAS as its collaborator rather than the window
— the author's existing decision about which object it actually drives.
"""

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QHBoxLayout,
                               QLabel, QPushButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from lillypad.design.mpl_style import TRACE_COLORMAPS
from lillypad.gui.text import RIGHT_CLICK_HINT
from lillypad.gui.widgets import ComboBox, DoubleSpinBox, Slider, _hline


# ─────────────────────────────────────────────────────────────────────────────
# Graphics settings dialog
# ─────────────────────────────────────────────────────────────────────────────
class GraphicsSettingsDialog(QDialog):
    def __init__(self, canvas, parent=None):
        super().__init__(parent, Qt.Tool)
        self.canvas = canvas
        self.setWindowTitle("Graphics Settings")
        # Width is set from the content once it is built (see below) — the
        # rows are wider than they look, and a hard-coded width clips them at
        # font sizes or DPI scalings other than the one it was picked on.
        #
        # Scrolled body + pinned Close button: the settings list is taller than
        # a short laptop screen, and a dialog that runs off the bottom takes
        # its Close button with it. Same pattern (and stylesheet) as the main
        # window's control column.
        outer = QVBoxLayout(self); outer.setSpacing(8); outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        lay = QVBoxLayout(body); lay.setSpacing(10); lay.setContentsMargins(14, 14, 14, 14)

        # ── Spectrum Y-axis ──────────────────────────────────────────────────
        lay.addWidget(self._hdr("Spectrum Y-axis"))
        self.chk_auto_y = QCheckBox("Auto-scale Y (follows live data)")
        self.chk_auto_y.setToolTip(RIGHT_CLICK_HINT)
        self.chk_auto_y.setChecked(canvas.autoscale_y)
        self.chk_auto_y.toggled.connect(self._on_autoscale_y)
        lay.addWidget(self.chk_auto_y)

        ylim_row = QHBoxLayout(); ylim_row.setSpacing(6)
        ylim_row.addWidget(QLabel("Min"))
        self.spin_ymin = DoubleSpinBox()
        self.spin_ymin.setRange(-1e6, 1e6); self.spin_ymin.setDecimals(0)
        self.spin_ymin.setSingleStep(100); self.spin_ymin.setValue(0)
        ylim_row.addWidget(self.spin_ymin)
        ylim_row.addWidget(QLabel("Max"))
        self.spin_ymax = DoubleSpinBox()
        self.spin_ymax.setRange(-1e6, 1e6); self.spin_ymax.setDecimals(0)
        self.spin_ymax.setSingleStep(100); self.spin_ymax.setValue(5000)
        ylim_row.addWidget(self.spin_ymax)
        lay.addLayout(ylim_row)
        self.spin_ymin.valueChanged.connect(
            lambda v: canvas.set_ylim(v, self.spin_ymax.value()))
        self.spin_ymax.valueChanged.connect(
            lambda v: canvas.set_ylim(self.spin_ymin.value(), v))

        self.chk_log = QCheckBox("Log scale")
        self.chk_log.toggled.connect(canvas.set_log_scale)
        lay.addWidget(self.chk_log)

        lay.addWidget(_hline())

        # ── Spectrum X-axis ──────────────────────────────────────────────────
        lay.addWidget(self._hdr("Spectrum X-axis"))
        self.chk_auto_x = QCheckBox("Auto-scale X (follows spectrometer range)")
        self.chk_auto_x.setToolTip(RIGHT_CLICK_HINT)
        self.chk_auto_x.setChecked(canvas.autoscale_x)
        self.chk_auto_x.toggled.connect(self._on_autoscale_x)
        lay.addWidget(self.chk_auto_x)

        xlim_row = QHBoxLayout(); xlim_row.setSpacing(6)
        xlim_row.addWidget(QLabel("Min"))
        self.spin_xmin = DoubleSpinBox()
        self.spin_xmin.setRange(0, 4000); self.spin_xmin.setDecimals(1)
        self.spin_xmin.setSingleStep(0.5); self.spin_xmin.setValue(500)
        self.spin_xmin.setSuffix(" nm")
        xlim_row.addWidget(self.spin_xmin)
        xlim_row.addWidget(QLabel("Max"))
        self.spin_xmax = DoubleSpinBox()
        self.spin_xmax.setRange(0, 4000); self.spin_xmax.setDecimals(1)
        self.spin_xmax.setSingleStep(0.5); self.spin_xmax.setValue(600)
        self.spin_xmax.setSuffix(" nm")
        xlim_row.addWidget(self.spin_xmax)
        lay.addLayout(xlim_row)
        self.spin_xmin.valueChanged.connect(
            lambda v: canvas.set_xlim(v, self.spin_xmax.value()))
        self.spin_xmax.valueChanged.connect(
            lambda v: canvas.set_xlim(self.spin_xmin.value(), v))

        lay.addWidget(_hline())

        # ── FROG trace axes ──────────────────────────────────────────────────
        lay.addWidget(self._hdr("FROG Trace Axes"))
        self.chk_auto_trace = QCheckBox("Auto-scale to scan range")
        self.chk_auto_trace.setToolTip(RIGHT_CLICK_HINT)
        self.chk_auto_trace.setChecked(canvas.autoscale_trace)
        self.chk_auto_trace.toggled.connect(self._on_autoscale_trace)
        lay.addWidget(self.chk_auto_trace)

        tx_row = QHBoxLayout(); tx_row.setSpacing(6)
        tx_row.addWidget(QLabel("Delay"))
        # ±100 ps of delay axis is already far beyond any FROG scan this stage
        # can produce; the tighter range keeps the spinbox from sizing itself
        # for a seven-figure number it will never show.
        self.spin_tmin = DoubleSpinBox()
        self.spin_tmin.setRange(-1e5, 1e5); self.spin_tmin.setDecimals(0)
        self.spin_tmin.setSingleStep(10); self.spin_tmin.setValue(-500)
        self.spin_tmin.setSuffix(" fs")
        tx_row.addWidget(self.spin_tmin)
        self.spin_tmax = DoubleSpinBox()
        self.spin_tmax.setRange(-1e5, 1e5); self.spin_tmax.setDecimals(0)
        self.spin_tmax.setSingleStep(10); self.spin_tmax.setValue(500)
        self.spin_tmax.setSuffix(" fs")
        tx_row.addWidget(self.spin_tmax)
        lay.addLayout(tx_row)
        self.spin_tmin.valueChanged.connect(
            lambda v: canvas.set_trace_xlim(v, self.spin_tmax.value()))
        self.spin_tmax.valueChanged.connect(
            lambda v: canvas.set_trace_xlim(self.spin_tmin.value(), v))

        ty_row = QHBoxLayout(); ty_row.setSpacing(6)
        ty_row.addWidget(QLabel("Wavel."))
        self.spin_twmin = DoubleSpinBox()
        self.spin_twmin.setRange(0, 4000); self.spin_twmin.setDecimals(1)
        self.spin_twmin.setSingleStep(0.5); self.spin_twmin.setValue(380)
        self.spin_twmin.setSuffix(" nm")
        ty_row.addWidget(self.spin_twmin)
        self.spin_twmax = DoubleSpinBox()
        self.spin_twmax.setRange(0, 4000); self.spin_twmax.setDecimals(1)
        self.spin_twmax.setSingleStep(0.5); self.spin_twmax.setValue(620)
        self.spin_twmax.setSuffix(" nm")
        ty_row.addWidget(self.spin_twmax)
        lay.addLayout(ty_row)
        self.spin_twmin.valueChanged.connect(
            lambda v: canvas.set_trace_ylim(v, self.spin_twmax.value()))
        self.spin_twmax.valueChanged.connect(
            lambda v: canvas.set_trace_ylim(self.spin_twmin.value(), v))

        lay.addWidget(_hline())

        # ── FROG trace colour ────────────────────────────────────────────────
        lay.addWidget(self._hdr("FROG Trace Colour"))
        cm_row = QHBoxLayout(); cm_row.setSpacing(6)
        cm_row.addWidget(QLabel("Colormap"))
        self.cmb_cmap = ComboBox()
        self.cmb_cmap.addItems(TRACE_COLORMAPS)
        self.cmb_cmap.setCurrentText(canvas.cmap_name)
        self.cmb_cmap.currentTextChanged.connect(canvas.set_cmap)
        cm_row.addWidget(self.cmb_cmap, 1)
        lay.addLayout(cm_row)
        self.chk_cmap_rev = QCheckBox("Reversed")
        self.chk_cmap_rev.toggled.connect(canvas.set_trace_reversed)
        lay.addWidget(self.chk_cmap_rev)

        thr_row = QHBoxLayout(); thr_row.setSpacing(6)
        thr_row.addWidget(QLabel("Hide below"))
        self.spin_thresh = DoubleSpinBox()
        self.spin_thresh.setRange(0.0, 100.0); self.spin_thresh.setDecimals(1)
        self.spin_thresh.setSingleStep(0.5); self.spin_thresh.setValue(0.0)
        self.spin_thresh.setSuffix(" %")
        self.spin_thresh.valueChanged.connect(canvas.set_trace_threshold)
        thr_row.addWidget(self.spin_thresh)
        thr_row.addWidget(QLabel("of peak")); thr_row.addStretch()
        lay.addLayout(thr_row)
        thr_hint = QLabel("Display only — saved data and the autocorrelation "
                          "always use the full trace.")
        thr_hint.setObjectName("dim"); thr_hint.setWordWrap(True)
        lay.addWidget(thr_hint)

        lay.addWidget(_hline())

        # ── Line width ───────────────────────────────────────────────────────
        lay.addWidget(self._hdr("Line Width"))
        lw_row = QHBoxLayout(); lw_row.setSpacing(6)
        lw_row.addWidget(QLabel("Width"))
        self.spin_lw = DoubleSpinBox()
        self.spin_lw.setRange(0.3, 6.0); self.spin_lw.setDecimals(1)
        self.spin_lw.setSingleStep(0.1); self.spin_lw.setValue(canvas.linewidth)
        self.spin_lw.setSuffix(" px")
        self.spin_lw.valueChanged.connect(canvas.set_linewidth)
        lw_row.addWidget(self.spin_lw); lw_row.addStretch()
        lay.addLayout(lw_row)

        lay.addWidget(_hline())

        # ── Plot proportions ─────────────────────────────────────────────────
        lay.addWidget(self._hdr("Plot Proportions"))
        prop_hint = QLabel("Spectrum column width  (% of total plot area) — or "
                           "drag the handle between the plots")
        prop_hint.setObjectName("dim"); prop_hint.setWordWrap(True)
        lay.addWidget(prop_hint)
        self.sld_prop = Slider(Qt.Horizontal)
        self.sld_prop.setRange(20, 80); self.sld_prop.setValue(50)
        self.sld_prop.setTickInterval(5); self.sld_prop.setTickPosition(QSlider.TicksBelow)
        lay.addWidget(self.sld_prop)
        self.lbl_prop = QLabel("50%")
        self.lbl_prop.setObjectName("value"); self.lbl_prop.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_prop)
        self.sld_prop.valueChanged.connect(self._on_prop)

        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(body); scroll.setWidgetResizable(True)
        # As-needed, not off: if a screen really cannot take the natural width,
        # the content must stay reachable rather than being silently clipped.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)

        close_row = QHBoxLayout(); close_row.setContentsMargins(14, 0, 14, 14)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        close_row.addWidget(btn_close)
        outer.addLayout(close_row)

        self._scroll = scroll
        self._body   = body
        self._sized  = False
        self.setFixedWidth(390)        # provisional; _fit_to_content resizes

        self._on_autoscale_y(canvas.autoscale_y)
        self._on_autoscale_x(canvas.autoscale_x)
        self._on_autoscale_trace(canvas.autoscale_trace)

    def _hdr(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("hdr")
        return lbl

    @property
    def log_scale(self):
        """Whether the spectrum panel is on a logarithmic y axis."""
        return self.chk_log.isChecked()

    def _fit_to_content(self):
        """Size to the content: wide enough that no row is cut off (hence no
        horizontal scrolling), tall enough to show what the screen allows.

        Both are clamped to the screen, so a small display degrades to
        scrolling rather than to a dialog with unreachable controls. Deferred
        to the first show: the row widths are only final once the stylesheet
        has polished the widgets, and a width measured before that under-sizes
        the dialog and clips the very rows it was meant to fit.
        """
        avail = QApplication.primaryScreen().availableGeometry()
        vbar  = self._scroll.verticalScrollBar().sizeHint().width()
        need  = (self._body.minimumSizeHint().width() + vbar
                 + 2 * self._scroll.frameWidth())
        self.setFixedWidth(max(390, min(need, int(avail.width() * 0.9))))
        self.resize(self.width(),
                    min(self._body.sizeHint().height() + 60,
                        int(avail.height() * 0.9)))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            self._fit_to_content()

    def _on_autoscale_y(self, on):
        self.canvas.set_autoscale_y(on)   # re-seeds the filter on the way on
        self.spin_ymin.setEnabled(not on)
        self.spin_ymax.setEnabled(not on)

    def _on_autoscale_x(self, on):
        self.canvas.set_autoscale_x(on)
        self.spin_xmin.setEnabled(not on)
        self.spin_xmax.setEnabled(not on)

    def _on_autoscale_trace(self, on):
        self.canvas.set_autoscale_trace(on)
        for sb in (self.spin_tmin, self.spin_tmax,
                   self.spin_twmin, self.spin_twmax):
            sb.setEnabled(not on)
        if not on:
            # Start manual from whatever is on screen, and PIN it: with the
            # axes still on matplotlib's own autoscale, the next scan's
            # im.set_extent() would drag the view along with it.
            tlo, thi = self.canvas.ax_trace.get_xlim()
            wlo, whi = self.canvas.ax_trace.get_ylim()
            for sb, v in ((self.spin_tmin, tlo), (self.spin_tmax, thi),
                          (self.spin_twmin, wlo), (self.spin_twmax, whi)):
                sb.blockSignals(True); sb.setValue(v); sb.blockSignals(False)
            self.canvas.set_trace_xlim(tlo, thi)
            self.canvas.set_trace_ylim(wlo, whi)

    def sync_limits(self):
        """Pull current axis limits and auto-scale flags into the dialog."""
        xlo, xhi = self.canvas.ax_spec.get_xlim()
        ylo, yhi = self.canvas.ax_spec.get_ylim()
        tlo, thi = self.canvas.ax_trace.get_xlim()
        wlo, whi = self.canvas.ax_trace.get_ylim()
        for sb, v in ((self.spin_xmin, xlo), (self.spin_xmax, xhi),
                      (self.spin_ymin, ylo), (self.spin_ymax, yhi),
                      (self.spin_tmin, tlo), (self.spin_tmax, thi),
                      (self.spin_twmin, wlo), (self.spin_twmax, whi)):
            sb.blockSignals(True); sb.setValue(v); sb.blockSignals(False)
        # Sync checkboxes to canvas state without re-triggering callbacks
        for chk, flag in ((self.chk_auto_x, self.canvas.autoscale_x),
                          (self.chk_auto_y, self.canvas.autoscale_y),
                          (self.chk_auto_trace, self.canvas.autoscale_trace)):
            chk.blockSignals(True); chk.setChecked(flag); chk.blockSignals(False)
        self.spin_xmin.setEnabled(not self.canvas.autoscale_x)
        self.spin_xmax.setEnabled(not self.canvas.autoscale_x)
        self.spin_ymin.setEnabled(not self.canvas.autoscale_y)
        self.spin_ymax.setEnabled(not self.canvas.autoscale_y)
        for sb in (self.spin_tmin, self.spin_tmax,
                   self.spin_twmin, self.spin_twmax):
            sb.setEnabled(not self.canvas.autoscale_trace)

    def _on_prop(self, val):
        self.lbl_prop.setText(f"{val}%")
        self.canvas.set_proportions(val / 100.0)

    def sync_proportions(self, pct):
        """Canvas-side change (the split handle was dragged) — keep the slider
        truthful without re-driving the canvas from it."""
        with QSignalBlocker(self.sld_prop):
            self.sld_prop.setValue(int(pct))
        self.lbl_prop.setText(f"{int(pct)}%")

    def push_to_canvas(self):
        """Drive the canvas from every widget in here, changed or not.

        The counterpart of sync_limits, and what lets a settings restore leave
        the normal signal wiring to do the work: setValue/setChecked emit only
        on a CHANGE, so a restored value that happens to equal a widget's
        construction default would reach the widget and never reach the plot.

        Every setter here is idempotent and their draw_idle()s coalesce into one
        frame, so re-asserting the lot is cheaper to reason about than working
        out which ones stayed silent.

        Appearance only — no axis limits. Nothing restores those any more (see
        _setting_specs), so the axes are on their own defaults at this point and
        pushing the spinboxes at them would pin the panel to a range nobody
        chose, purely because those boxes have to hold some number at startup.
        """
        c = self.canvas
        c.set_cmap(self.cmb_cmap.currentText())
        c.set_trace_reversed(self.chk_cmap_rev.isChecked())
        c.set_trace_threshold(self.spin_thresh.value())
        c.set_linewidth(self.spin_lw.value())
        c.set_log_scale(self.chk_log.isChecked())
        c.set_proportions(self.sld_prop.value() / 100.0)

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_()
