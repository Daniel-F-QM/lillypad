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

Everything else (layout, controls, hardware layer, scan engine) is unchanged
and shared with frog_gui.py via hardware.py / scan.py.

    python frog_gui_fast.py
"""

import sys
import math
import time
import tempfile
import threading
import numpy as np
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
    QProgressBar, QFrame, QScrollArea, QSizePolicy, QStatusBar, QFileDialog,
    QDialog, QToolBar, QSlider, QLineEdit, QMenu, QComboBox, QRubberBand
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPointF, QSize, QRect, QPoint
from PySide6.QtGui import (QPalette, QColor, QFont, QIcon, QPixmap, QPainter,
                           QPen, QPolygonF, QAction, QActionGroup)
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.0, "axes.titleweight": "bold", "axes.titlesize": 9.5,
    "axes.labelsize": 9,
})
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from hardware import (SimulatedStage, SimulatedSpectrometer,
                      KinesisStage, ZaberStage, PiezoJenaStage,
                      SeabreezeSpectrometer,
                      list_kinesis_stages, list_seabreeze_spectrometers,
                      PULSE_SHAPES, DEFAULT_PULSE)
from scan import (FrogScanConfig, FrogScanWorker, autocorrelation, fwhm,
                  position_to_delay_fs, delay_to_position_um,
                  write_dwc, write_npz, write_csv,
                  _um_to_stage, _stage_to_um)


# ─────────────────────────────────────────────────────────────────────────────
# Layering: bg (window) < plot_bg (plot panel) < surface (controls/dialogs),
# each a distinct step so panels and controls read as raised. Dark text is
# #e6edf3, not near-white — ~13:1 on bg avoids halation from max-contrast text.
DARK_PALETTE = {
    "bg": "#0d1117", "surface": "#1c2128", "border": "#363d47",
    "border_hover": "#444c56",
    "accent": "#58a6ff", "accent2": "#79c0ff", "text": "#e6edf3",
    "text_dim": "#b0bcc9", "text_disabled": "#6e7681",
    "danger": "#ff7b72", "warn": "#f0883e", "good": "#3fb950",
    "plot_bg": "#161b22", "grid": "#2a3140",
}

# Neutral gray ramp (no warm cast) so the cool blue accents don't clash.
LIGHT_PALETTE = {
    "bg": "#f2f3f5", "surface": "#ffffff", "border": "#d0d7de",
    "border_hover": "#afb8c1",
    "accent": "#0969da", "accent2": "#0550ae", "text": "#1f2328",
    "text_dim": "#57606a", "text_disabled": "#8c959f",
    "danger": "#cf222e", "warn": "#bc4c00", "good": "#1a7f37",
    "plot_bg": "#ffffff", "grid": "#d8dee4",
}

# Mutated in place on theme switch so every runtime PALETTE[...] lookup follows.
PALETTE = dict(DARK_PALETTE)

FONT_STACK = "'Segoe UI','DejaVu Sans',Arial,sans-serif"


def _make_arrow_icons(color, tag):
    """Render spinbox up/down arrows to PNG files (QSS url() cannot load data
    URIs, which is why the arrows were missing). Needs a running QApplication.
    Returns (up_path, down_path) with forward slashes for use in QSS."""
    icon_dir = Path(tempfile.gettempdir()) / "lillypad_icons"
    icon_dir.mkdir(exist_ok=True)
    shapes = {"up": [(4.0, 1.5), (7.5, 6.0), (0.5, 6.0)],
              "dn": [(0.5, 1.0), (7.5, 1.0), (4.0, 5.5)]}
    paths = {}
    for name, pts in shapes.items():
        pm = QPixmap(8, 7)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
        p.end()
        fp = icon_dir / f"arrow_{name}_{tag}.png"
        pm.save(str(fp))
        paths[name] = fp.as_posix()
    return paths["up"], paths["dn"]


def _make_check_icon(color, tag):
    """Render a checkmark PNG for the checked checkbox indicator (same QSS
    url() limitation as the arrows). Needs a running QApplication."""
    icon_dir = Path(tempfile.gettempdir()) / "lillypad_icons"
    icon_dir.mkdir(exist_ok=True)
    pm = QPixmap(10, 10)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 2.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.drawPolyline(QPolygonF([QPointF(1.5, 5.5), QPointF(4.0, 8.0),
                              QPointF(8.5, 2.5)]))
    p.end()
    fp = icon_dir / f"check_{tag}.png"
    pm.save(str(fp))
    return fp.as_posix()


def build_stylesheet(pal, tag):
    uri_up, uri_dn = _make_arrow_icons(pal["text"], tag)
    uri_chk = _make_check_icon(pal["bg"], tag)
    return f"""
QMainWindow, QWidget {{ background-color:{pal['bg']}; color:{pal['text']};
    font-family:{FONT_STACK}; font-size:13px; font-weight:400; }}
QGroupBox {{ border:1px solid {pal['border']}; border-radius:6px; margin-top:12px;
    padding:10px 6px 8px 6px; font-size:11px; font-weight:700; color:{pal['text_dim']};
    letter-spacing:1.5px; text-transform:uppercase; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
QPushButton {{ background-color:{pal['surface']}; border:1px solid {pal['border']};
    border-radius:5px; padding:7px 14px; color:{pal['text']}; font-size:13px;
    font-weight:600; }}
QPushButton:focus {{ border-color:{pal['accent']}; }}
QPushButton:hover {{ border-color:{pal['accent']}; color:{pal['accent']};
    background-color:{pal['border_hover']}; }}
QPushButton:pressed {{ background-color:{pal['accent']}; color:{pal['bg']}; }}
QPushButton:disabled {{ color:{pal['text_disabled']}; }}
QPushButton#accent {{ border-color:{pal['accent']}; color:{pal['accent']}; }}
QPushButton#accent:hover {{ background-color:{pal['accent']}; color:{pal['bg']}; }}
QPushButton#danger {{ border-color:{pal['danger']}; color:{pal['danger']}; }}
QPushButton#danger:hover {{ background-color:{pal['danger']}; color:{pal['bg']}; }}
QPushButton#overlay {{ border-color:{pal['accent']}; color:{pal['accent']};
    padding:0px; font-size:12px; border-radius:4px; }}
QPushButton#overlay:hover {{ background-color:{pal['accent']}; color:{pal['bg']}; }}
QPushButton#overlay:pressed {{ background-color:{pal['accent']}; color:{pal['bg']}; }}
QPushButton::menu-indicator {{ image: url("{uri_dn}"); width:8px; height:7px;
    subcontrol-origin:padding; subcontrol-position:center right; right:6px; }}
QMenu {{ background-color:{pal['surface']}; border:1px solid {pal['border']};
    border-radius:5px; padding:4px; color:{pal['text']}; font-size:13px;
    font-weight:400; }}
QMenu::item {{ padding:6px 14px 6px 26px; border-radius:4px; }}
QMenu::item:selected {{ background-color:{pal['border_hover']};
    color:{pal['accent']}; }}
QMenu::indicator {{ width:12px; height:12px; left:8px; }}
QDoubleSpinBox, QSpinBox {{ background-color:{pal['surface']};
    border:1px solid {pal['border']}; border-radius:4px;
    padding:4px 24px 4px 8px; color:{pal['text']}; font-weight:400;
    min-height:26px; selection-background-color:{pal['accent']};
    selection-color:{pal['bg']}; }}
QDoubleSpinBox:focus, QSpinBox:focus {{ border-color:{pal['accent']}; }}
QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin:border; subcontrol-position:top right;
    width:22px; border-left:1px solid {pal['border']};
    border-bottom:1px solid {pal['border']};
    border-top-right-radius:4px; background:{pal['surface']}; }}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {{
    background:{pal['border_hover']}; }}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed {{
    background:{pal['accent']}; }}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin:border; subcontrol-position:bottom right;
    width:22px; border-left:1px solid {pal['border']};
    border-bottom-right-radius:4px; background:{pal['surface']}; }}
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background:{pal['border_hover']}; }}
QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
    background:{pal['accent']}; }}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    image: url("{uri_up}"); width:8px; height:7px; }}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    image: url("{uri_dn}"); width:8px; height:7px; }}
QCheckBox {{ spacing:8px; font-weight:400; background:transparent; }}
QCheckBox::indicator {{ width:15px; height:15px; border:2px solid {pal['border']};
    border-radius:3px; background:{pal['surface']}; }}
QCheckBox::indicator:checked {{ background-color:{pal['accent']};
    border-color:{pal['accent']}; image: url("{uri_chk}"); }}
QLabel {{ background:transparent; }}
QLabel#dim {{ color:{pal['text_dim']}; font-size:12px; font-weight:400; }}
QLabel#value {{ color:{pal['accent']}; font-weight:600; font-size:15px; }}
QLabel#readout {{ color:{pal['accent']}; font-weight:600; font-size:15px;
    font-family:'Cascadia Mono','Consolas',monospace; }}
QLabel#moving {{ color:{pal['warn']}; font-weight:600; }}
QLabel#hdr {{ color:{pal['text_dim']}; font-weight:700; }}
QLabel#sat {{ color:{pal['danger']}; font-size:12px; font-weight:600; }}
QLabel#satok {{ color:{pal['text_dim']}; font-size:12px; font-weight:400; }}
QLabel#satwarn {{ color:{pal['warn']}; font-size:12px; font-weight:600; }}
QLabel#ok {{ color:{pal['accent']}; }}
QLabel#err {{ color:{pal['danger']}; }}
QProgressBar {{ border:1px solid {pal['border']}; border-radius:4px;
    text-align:center; background:{pal['surface']}; color:{pal['text']};
    font-weight:500; }}
QProgressBar::chunk {{ background-color:{pal['accent']}; border-radius:3px; }}
QStatusBar {{ background-color:{pal['surface']}; color:{pal['text_dim']};
    border-top:1px solid {pal['border']}; font-size:12px; font-weight:400; }}
QToolBar {{ background:{pal['surface']}; border-bottom:1px solid {pal['border']};
    spacing:6px; padding:4px 8px; }}
QDialog {{ background:{pal['bg']}; }}
QScrollArea {{ border:none; background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:18px; margin:2px;
    border:none; border-radius:7px; }}
QScrollBar::handle:vertical {{ background:{pal['border']}; min-height:32px;
    border-radius:7px; }}
QScrollBar::handle:vertical:hover {{ background:{pal['border_hover']}; }}
QScrollBar::handle:vertical:pressed {{ background:{pal['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px;
    width:0px; background:transparent; border:none; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
QScrollBar:horizontal {{ background:transparent; height:18px; margin:2px;
    border:none; border-radius:7px; }}
QScrollBar::handle:horizontal {{ background:{pal['border']}; min-width:32px;
    border-radius:7px; }}
QScrollBar::handle:horizontal:hover {{ background:{pal['border_hover']}; }}
QScrollBar::handle:horizontal:pressed {{ background:{pal['accent']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0px;
    height:0px; background:transparent; border:none; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background:transparent; }}
QFrame#sep {{ background-color:{pal['border']}; }}
"""


def apply_app_palette(app, pal):
    qpal = QPalette()
    qpal.setColor(QPalette.Window, QColor(pal["bg"]))
    qpal.setColor(QPalette.WindowText, QColor(pal["text"]))
    qpal.setColor(QPalette.Base, QColor(pal["surface"]))
    qpal.setColor(QPalette.Text, QColor(pal["text"]))
    qpal.setColor(QPalette.Button, QColor(pal["surface"]))
    qpal.setColor(QPalette.ButtonText, QColor(pal["text"]))
    app.setPalette(qpal)
def resource_path(*parts):
    """Resolve a path to a bundled resource, working both when run from source
    and when frozen by PyInstaller (assets live under sys._MEIPASS)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


ICON_PATH    = resource_path("icons", "Lilypad.png")
SUN_ICON     = resource_path("icons", "sun.png")
MOON_ICON    = resource_path("icons", "moon.png")
RESCALE_ICON = resource_path("icons", "rescale.png")




def _hline():
    f = QFrame(); f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Live-feed acquisition worker (FIX 2 + 3)
# ─────────────────────────────────────────────────────────────────────────────
class LiveFeedWorker(QThread):
    """Continuously acquires spectra off the GUI thread and emits them.

    FIX 2 — spectrometer.acquire() blocks for the integration time; running it
    here keeps the UI responsive even at long integrations.
    FIX 3 — each loop is paced to the integration time (with a small floor) so
    we never spin faster than the hardware actually produces frames.

    `pause()` blocks until any in-flight acquire has finished, so the scan can
    safely take over the shared stage/spectrometer.
    """
    spectrum_ready = Signal(object, object)   # (wavelengths, raw counts)

    def __init__(self, get_spec, min_interval_ms=30, parent=None):
        super().__init__(parent)
        self._get_spec = get_spec
        self._min = float(min_interval_ms)
        self._run = True
        self._paused = True
        self._idle = threading.Event()   # set whenever not mid-acquire
        self._idle.set()
        # Measured worst-case time for one loop iteration. pause() sizes its
        # timeout from this instead of a flat 2 s, which a long integration
        # would otherwise blow through before the loop could possibly park.
        self._last_cycle_ms = float(min_interval_ms)

    def run(self):
        while self._run:
            if self._paused:
                self._idle.set()
                self.msleep(15)
                continue
            self._idle.clear()
            spec = self._get_spec()
            if spec is None:
                self._idle.set()
                self.msleep(30)
                continue
            try:
                t0 = time.monotonic()
                raw = np.asarray(spec.acquire(), float)
                wl  = np.asarray(spec.wavelengths, float)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
            except Exception:
                self.msleep(50)
                continue
            if self._run and not self._paused:
                self.spectrum_ready.emit(wl, raw)
            # Pace to the integration time; acquire already consumed `elapsed_ms`
            # of it on real hardware, so only sleep the remainder.
            target = max(self._min, float(getattr(spec, "integration_ms", self._min)))
            # Whichever of the two actually dominates the cycle is what a
            # pause() may have to sit through before we come round the top.
            self._last_cycle_ms = max(target, elapsed_ms)
            self.msleep(int(max(0.0, target - elapsed_ms)))

    def resume(self):
        self._paused = False

    def pause(self, wait_ms=None):
        """Stop acquiring; block until the current acquire (if any) returns.

        Returns True once the loop has parked in the idle branch — only then is
        it safe to hand the stage/spectrometer to another thread. False means
        the timeout expired with an acquire STILL in flight; callers must treat
        that as "device not available" rather than ignoring it.

        The default timeout tracks the measured loop cycle (one acquire plus one
        pacing sleep, with margin) so a long integration cannot time out early.
        """
        self._paused = True
        if wait_ms is None:
            wait_ms = max(2000.0, 3.0 * self._last_cycle_ms + 500.0)
        return self._idle.wait(wait_ms / 1000.0)

    def stop(self):
        self._run = False
        self._paused = True


# Shown whenever a device operation has to be refused because the live feed did
# not release the hardware in time.
FEED_BUSY_MSG = ("Live feed is still mid-acquisition — try again in a moment "
                 "(or stop the feed first).")


# ─────────────────────────────────────────────────────────────────────────────
# Saturation indicator
# ─────────────────────────────────────────────────────────────────────────────
# Fraction of full scale at which the lamp goes amber. Deliberately well below
# the saturation threshold itself (FrogScanConfig.saturation_fraction, which
# both the scan worker and the live lamp use): most detectors go nonlinear
# before they actually clip, so by the time pixels are pinned at full scale the
# spectrum around them is already wrong.
SAT_WARN_FRACTION = 0.90

# Colormaps offered for the FROG trace. All perceptually uniform, so equal
# steps in count map to equal steps in apparent brightness — a trace read off
# a non-uniform map (jet and friends) shows structure the data does not have.
# First entry is the default.
TRACE_COLORMAPS = ["magma", "viridis", "inferno", "plasma", "cividis"]


class StatusLamp(QWidget):
    """Round go/no-go indicator for detector headroom.

    Painted rather than styled: a QLabel with a border-radius cannot draw the
    soft halo, and the halo is what makes a 13 px dot readable at a glance in
    the corner of the status bar.

    States — unknown (grey, no full-scale reported), ok (green), warn (amber,
    close to clipping), sat (red, pixels pinned at full scale).
    """
    COLORS = {"unknown": "text_disabled", "ok": "good",
              "warn": "warn", "sat": "danger"}

    def __init__(self, diameter=12, parent=None):
        super().__init__(parent)
        self._d = diameter
        self._state = "unknown"
        self.setFixedSize(diameter + 8, diameter + 8)

    def set_state(self, state):
        if state not in self.COLORS:
            raise ValueError(f"unknown lamp state {state!r}")
        if state != self._state:
            self._state = state
            self.update()

    def state(self):
        return self._state

    def paintEvent(self, _event):
        # PALETTE is read at paint time, so a theme switch only needs update().
        color = QColor(PALETTE[self.COLORS[self._state]])
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = self.width() / 2.0, self.height() / 2.0, self._d / 2.0
        if self._state != "unknown":
            halo = QColor(color)
            halo.setAlpha(70)
            p.setPen(Qt.NoPen); p.setBrush(halo)
            p.drawEllipse(QPointF(cx, cy), r + 3.0, r + 3.0)
        p.setPen(QPen(color.darker(150), 1.0))
        p.setBrush(color)
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Export worker — writing a full trace is far too slow for the GUI thread
# ─────────────────────────────────────────────────────────────────────────────
class ExportWorker(QThread):
    """Writes a FrogResult to disk off the GUI thread.

    write_dwc formats on the order of 1.7M values for a full trace, which froze
    the window for the whole save. A finished FrogResult is never mutated again,
    so the worker can read it without any locking — and because a later scan
    rebinds `self.result` rather than modifying it, an in-flight export keeps
    writing the snapshot it was handed.
    """
    done  = Signal(str)
    error = Signal(str)

    def __init__(self, writer, path, result, parent=None):
        super().__init__(parent)
        self._writer = writer
        self._path   = path
        self._result = result

    def run(self):
        try:
            self._writer(self._path, self._result)
        except Exception as e:
            self.error.emit(str(e))
        else:
            self.done.emit(self._path)


# ─────────────────────────────────────────────────────────────────────────────
# Pop-up dialogs (Qt.Tool = thin frame, floats above, non-modal)
# ─────────────────────────────────────────────────────────────────────────────
class AcquisitionSettingsDialog(QDialog):
    """Per-point acquisition settings. Values are read live at scan start."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Acquisition Settings")
        self.setFixedWidth(300)
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14, 14, 14, 14)

        grid = QGridLayout(); grid.setSpacing(8)
        grid.addWidget(QLabel("Averages / point"), 0, 0)
        self.spin_avg = QSpinBox(); self.spin_avg.setRange(1, 1000); self.spin_avg.setValue(1)
        grid.addWidget(self.spin_avg, 0, 1)
        grid.addWidget(QLabel("Idle shots"), 1, 0)
        self.spin_idle = QSpinBox(); self.spin_idle.setRange(0, 1000); self.spin_idle.setValue(0)
        grid.addWidget(self.spin_idle, 1, 1)
        grid.addWidget(QLabel("Wait after move"), 2, 0)
        self.spin_wait = QSpinBox(); self.spin_wait.setRange(0, 10000)
        self.spin_wait.setValue(0); self.spin_wait.setSuffix(" ms")
        grid.addWidget(self.spin_wait, 2, 1)
        lay.addLayout(grid)

        hint = QLabel("After each move the stage discards the idle shots, then "
                      "averages the next N frames into one column.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Saturation ───────────────────────────────────────────────────────
        lay.addWidget(self._hdr("Saturation"))
        srow = QGridLayout(); srow.setSpacing(8)
        srow.addWidget(QLabel("Threshold"), 0, 0)
        self.spin_sat = QDoubleSpinBox()
        self.spin_sat.setRange(50.0, 100.0); self.spin_sat.setDecimals(1)
        self.spin_sat.setSingleStep(1.0); self.spin_sat.setSuffix(" % FS")
        self.spin_sat.setValue(100.0 * FrogScanConfig.saturation_fraction)
        srow.addWidget(self.spin_sat, 0, 1)
        lay.addLayout(srow)
        self.chk_abort_sat = QCheckBox("Abort scan on saturation")
        self.chk_abort_sat.setChecked(FrogScanConfig.abort_on_saturation)
        lay.addWidget(self.chk_abort_sat)

        sat_hint = QLabel("Detectors go nonlinear before they hard-clip, so a "
                          "threshold below 100% is the honest setting. Left "
                          "unchecked, a saturated scan finishes and is saved "
                          "with the clipped columns marked in the .npz.")
        sat_hint.setObjectName("dim"); sat_hint.setWordWrap(True)
        lay.addWidget(sat_hint)

        btn = QPushButton("Close"); btn.clicked.connect(self.hide)
        lay.addWidget(btn)

    def _hdr(self, text):
        l = QLabel(text); l.setObjectName("hdr")
        return l

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_()


class HardwareDialog(QDialog):
    """Switch the stage and spectrometer INDEPENDENTLY (e.g. real stage +
    simulated spectrometer), with inline status."""
    def __init__(self, main, parent=None):
        super().__init__(parent, Qt.Tool)
        self.main = main
        self.setWindowTitle("Hardware")
        self.setFixedWidth(360)
        lay = QVBoxLayout(self); lay.setSpacing(8); lay.setContentsMargins(14, 14, 14, 14)

        lay.addWidget(self._header("Stage"))
        self.lbl_stage = QLabel(); self.lbl_stage.setObjectName("value")
        self.lbl_stage.setWordWrap(True)
        lay.addWidget(self.lbl_stage)
        # Two equal-width columns shared by both stage rows, so "Real (Zaber)"
        # lines up with (and matches the width of) "Real (Kinesis)" and the
        # port field shrinks to the "Simulated" column.
        sgrid = QGridLayout(); sgrid.setSpacing(6)
        sgrid.setColumnStretch(0, 1); sgrid.setColumnStretch(1, 1)
        b_ss = QPushButton("Simulated")
        b_ss.clicked.connect(lambda: self._do(self.main._use_sim_stage))
        b_sr = QPushButton("Real (Kinesis)"); b_sr.setObjectName("accent")
        b_sr.clicked.connect(lambda: self._do(self.main._connect_real_stage))
        sgrid.addWidget(b_ss, 0, 0); sgrid.addWidget(b_sr, 0, 1)

        # Zaber: optional serial port (blank = auto-scan the machine's ports).
        self.edit_zaber_port = QLineEdit()
        self.edit_zaber_port.setPlaceholderText("COM (auto)")
        b_zr = QPushButton("Real (Zaber)"); b_zr.setObjectName("accent")
        b_zr.clicked.connect(lambda: self._do(self._connect_zaber))
        sgrid.addWidget(self.edit_zaber_port, 1, 0); sgrid.addWidget(b_zr, 1, 1)

        # Piezo Jena: optional serial port (blank = auto-scan), like Zaber.
        self.edit_piezo_port = QLineEdit()
        self.edit_piezo_port.setPlaceholderText("COM (auto)")
        b_pj = QPushButton("Real (Piezo Jena)"); b_pj.setObjectName("accent")
        b_pj.clicked.connect(lambda: self._do(self._connect_piezo))
        sgrid.addWidget(self.edit_piezo_port, 2, 0); sgrid.addWidget(b_pj, 2, 1)
        lay.addLayout(sgrid)

        lay.addWidget(_hline())

        lay.addWidget(self._header("Spectrometer"))
        self.lbl_spec = QLabel(); self.lbl_spec.setObjectName("value")
        self.lbl_spec.setWordWrap(True)
        lay.addWidget(self.lbl_spec)
        prow = QHBoxLayout()
        b_ps = QPushButton("Simulated")
        b_ps.clicked.connect(lambda: self._do(self.main._use_sim_spectrometer))
        b_pr = QPushButton("Real (seabreeze)"); b_pr.setObjectName("accent")
        b_pr.clicked.connect(lambda: self._do(self.main._connect_real_spectrometer))
        prow.addWidget(b_ps); prow.addWidget(b_pr)
        lay.addLayout(prow)

        # Full-scale override. Without this, a spectrometer that does not
        # report `max_intensity` leaves saturation unchecked with no way out
        # from the UI. Blank = trust the device.
        fsrow = QGridLayout(); fsrow.setSpacing(6)
        fsrow.setColumnStretch(0, 0); fsrow.setColumnStretch(1, 1)
        fsrow.addWidget(QLabel("Full scale"), 0, 0)
        self.edit_full_scale = QLineEdit()
        self.edit_full_scale.setPlaceholderText("auto (from device)")
        self.edit_full_scale.editingFinished.connect(self._on_full_scale)
        fsrow.addWidget(self.edit_full_scale, 0, 1)
        lay.addLayout(fsrow)

        lay.addWidget(_hline())

        # ── Simulated beam ───────────────────────────────────────────────────
        # Which pulse the simulated spectrometer is measuring. Changing either
        # box re-makes the simulator on the spot (its wavelength window is
        # derived from the beam, so it cannot just be mutated in place).
        lay.addWidget(self._header("Simulated beam"))
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(0, 1); brow.setColumnStretch(1, 0)
        self.cmb_pulse = QComboBox()
        for key, shape in PULSE_SHAPES.items():
            self.cmb_pulse.addItem(shape["label"], key)
        self.cmb_pulse.setCurrentIndex(
            max(0, self.cmb_pulse.findData(self.main.sim_pulse)))
        self.cmb_pulse.currentIndexChanged.connect(self._on_beam)
        self.cmb_gate = QComboBox()
        self.cmb_gate.addItem("SHG", "shg")
        self.cmb_gate.addItem("PG", "pg")
        self.cmb_gate.setCurrentIndex(
            max(0, self.cmb_gate.findData(self.main.sim_gate)))
        self.cmb_gate.currentIndexChanged.connect(self._on_beam)
        brow.addWidget(self.cmb_pulse, 0, 0); brow.addWidget(self.cmb_gate, 0, 1)
        lay.addLayout(brow)
        self.lbl_beam = QLabel(); self.lbl_beam.setObjectName("dim")
        self.lbl_beam.setWordWrap(True)
        lay.addWidget(self.lbl_beam)

        self.lbl_msg = QLabel(""); self.lbl_msg.setObjectName("dim"); self.lbl_msg.setWordWrap(True)
        lay.addWidget(self.lbl_msg)
        btn = QPushButton("Close"); btn.clicked.connect(self.hide)
        lay.addWidget(btn)

    def _connect_zaber(self):
        port = self.edit_zaber_port.text().strip() or None
        return self.main._connect_zaber_stage(port)

    def _connect_piezo(self):
        port = self.edit_piezo_port.text().strip() or None
        return self.main._connect_piezo_jena_stage(port)

    def _on_full_scale(self):
        """Apply the typed full-scale override (blank clears it back to auto)."""
        text = self.edit_full_scale.text().strip()
        if not text:
            self.main.scan_cfg.saturation_counts = None
        else:
            try:
                value = float(text)
            except ValueError:
                value = -1.0
            if value <= 0:
                # Refuse rather than silently reverting: a bad value here
                # disables saturation checking, which must not happen quietly.
                self._do(lambda: (False, f"Full scale must be a positive "
                                         f"number of counts (got {text!r})."))
                return
            self.main.scan_cfg.saturation_counts = value
        self.main._reset_saturation()
        self._do(lambda: (True, "Full scale: " + (
            f"{self.main.scan_cfg.saturation_counts:.0f} counts (override)."
            if self.main.scan_cfg.saturation_counts else "auto (from device).")))

    def _on_beam(self):
        self.main.sim_pulse = self.cmb_pulse.currentData()
        self.main.sim_gate  = self.cmb_gate.currentData()
        if isinstance(self.main.spec, SimulatedSpectrometer):
            self._do(self.main._use_sim_spectrometer)
        else:
            # A real spectrometer is connected — remember the choice for the
            # next time "Simulated" is pressed rather than swapping it out.
            self._refresh()
            self.lbl_msg.setText("Beam saved — press Simulated to use it.")

    def _header(self, text):
        l = QLabel(text)
        l.setObjectName("hdr")
        return l

    def _do(self, fn):
        ok, msg = fn()
        # Color via objectName + repolish (not setStyleSheet) so a later theme
        # switch restyles the label along with everything else.
        self.lbl_msg.setObjectName("ok" if ok else "err")
        self.lbl_msg.style().unpolish(self.lbl_msg)
        self.lbl_msg.style().polish(self.lbl_msg)
        self.lbl_msg.setText(msg)
        self._refresh()

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self._refresh(); self.lbl_msg.setText(""); self.show(); self.raise_()

    def _refresh(self):
        self.lbl_stage.setText(self.main.stage.name)
        self.lbl_spec.setText(self.main.spec.name)
        reported = getattr(self.main.spec, "max_counts", None)
        self.edit_full_scale.setPlaceholderText(
            f"auto — device reports {reported:.0f}" if reported
            else "device reports none — saturation unchecked")
        desc = PULSE_SHAPES[self.main.sim_pulse]["desc"]
        spec = self.main.spec
        if isinstance(spec, SimulatedSpectrometer):
            desc += (f"\nSuggested scan: ±{spec.suggested_delay_fs:.0f} fs   ·   "
                     f"{spec.wavelengths[0]:.0f}–{spec.wavelengths[-1]:.0f} nm")
        self.lbl_beam.setText(desc)


# ─────────────────────────────────────────────────────────────────────────────
# Spectrometer picker (modal — shown only when seabreeze finds 2+ devices)
# ─────────────────────────────────────────────────────────────────────────────
class DevicePickerDialog(QDialog):
    """Pick one device from an enumeration. `devices` is [(label, id), ...] —
    the shape both hardware.list_*() helpers return — and the chosen id comes
    back from pick()."""
    def __init__(self, devices, parent=None, title="Select Device", prompt=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(360)
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14, 14, 14, 14)
        lbl = QLabel(prompt or f"{len(devices)} devices found — choose one:")
        lbl.setObjectName("dim")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self.cmb = QComboBox()
        for label, ident in devices:
            self.cmb.addItem(f"{label}  [{ident}]", ident)
        lay.addWidget(self.cmb)
        row = QHBoxLayout()
        b_cancel = QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        b_ok = QPushButton("Connect"); b_ok.setObjectName("accent")
        b_ok.setDefault(True); b_ok.clicked.connect(self.accept)
        row.addWidget(b_cancel); row.addWidget(b_ok)
        lay.addLayout(row)

    @staticmethod
    def pick(parent, devices, title="Select Device", prompt=""):
        """Returns the chosen id, or None on cancel."""
        dlg = DevicePickerDialog(devices, parent, title, prompt)
        return dlg.cmb.currentData() if dlg.exec() == QDialog.Accepted else None


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
        self.chk_auto_y.setChecked(canvas.autoscale_y)
        self.chk_auto_y.toggled.connect(self._on_autoscale_y)
        lay.addWidget(self.chk_auto_y)

        ylim_row = QHBoxLayout(); ylim_row.setSpacing(6)
        ylim_row.addWidget(QLabel("Min"))
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1e6, 1e6); self.spin_ymin.setDecimals(0)
        self.spin_ymin.setSingleStep(100); self.spin_ymin.setValue(0)
        ylim_row.addWidget(self.spin_ymin)
        ylim_row.addWidget(QLabel("Max"))
        self.spin_ymax = QDoubleSpinBox()
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
        self.chk_auto_x.setChecked(canvas.autoscale_x)
        self.chk_auto_x.toggled.connect(self._on_autoscale_x)
        lay.addWidget(self.chk_auto_x)

        xlim_row = QHBoxLayout(); xlim_row.setSpacing(6)
        xlim_row.addWidget(QLabel("Min"))
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(0, 4000); self.spin_xmin.setDecimals(1)
        self.spin_xmin.setSingleStep(0.5); self.spin_xmin.setValue(500)
        self.spin_xmin.setSuffix(" nm")
        xlim_row.addWidget(self.spin_xmin)
        xlim_row.addWidget(QLabel("Max"))
        self.spin_xmax = QDoubleSpinBox()
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
        self.chk_auto_trace.setChecked(canvas.autoscale_trace)
        self.chk_auto_trace.toggled.connect(self._on_autoscale_trace)
        lay.addWidget(self.chk_auto_trace)

        tx_row = QHBoxLayout(); tx_row.setSpacing(6)
        tx_row.addWidget(QLabel("Delay"))
        # ±100 ps of delay axis is already far beyond any FROG scan this stage
        # can produce; the tighter range keeps the spinbox from sizing itself
        # for a seven-figure number it will never show.
        self.spin_tmin = QDoubleSpinBox()
        self.spin_tmin.setRange(-1e5, 1e5); self.spin_tmin.setDecimals(1)
        self.spin_tmin.setSingleStep(10); self.spin_tmin.setValue(-500)
        self.spin_tmin.setSuffix(" fs")
        tx_row.addWidget(self.spin_tmin)
        self.spin_tmax = QDoubleSpinBox()
        self.spin_tmax.setRange(-1e5, 1e5); self.spin_tmax.setDecimals(1)
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
        self.spin_twmin = QDoubleSpinBox()
        self.spin_twmin.setRange(0, 4000); self.spin_twmin.setDecimals(1)
        self.spin_twmin.setSingleStep(0.5); self.spin_twmin.setValue(380)
        self.spin_twmin.setSuffix(" nm")
        ty_row.addWidget(self.spin_twmin)
        self.spin_twmax = QDoubleSpinBox()
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
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(TRACE_COLORMAPS)
        self.cmb_cmap.setCurrentText(canvas._cmap_name)
        self.cmb_cmap.currentTextChanged.connect(canvas.set_cmap)
        cm_row.addWidget(self.cmb_cmap, 1)
        lay.addLayout(cm_row)
        self.chk_cmap_rev = QCheckBox("Reversed")
        self.chk_cmap_rev.toggled.connect(canvas.set_trace_reversed)
        lay.addWidget(self.chk_cmap_rev)

        thr_row = QHBoxLayout(); thr_row.setSpacing(6)
        thr_row.addWidget(QLabel("Hide below"))
        self.spin_thresh = QDoubleSpinBox()
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
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.3, 6.0); self.spin_lw.setDecimals(1)
        self.spin_lw.setSingleStep(0.1); self.spin_lw.setValue(canvas._lw)
        self.spin_lw.setSuffix(" px")
        self.spin_lw.valueChanged.connect(canvas.set_linewidth)
        lw_row.addWidget(self.spin_lw); lw_row.addStretch()
        lay.addLayout(lw_row)

        lay.addWidget(_hline())

        # ── Plot proportions ─────────────────────────────────────────────────
        lay.addWidget(self._hdr("Plot Proportions"))
        prop_hint = QLabel("Spectrum column width  (% of total plot area)")
        prop_hint.setObjectName("dim"); prop_hint.setWordWrap(True)
        lay.addWidget(prop_hint)
        self.sld_prop = QSlider(Qt.Horizontal)
        self.sld_prop.setRange(15, 55); self.sld_prop.setValue(50)
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
        self.canvas.autoscale_y = on
        self.spin_ymin.setEnabled(not on)
        self.spin_ymax.setEnabled(not on)

    def _on_autoscale_x(self, on):
        self.canvas.autoscale_x = on
        self.spin_xmin.setEnabled(not on)
        self.spin_xmax.setEnabled(not on)

    def _on_autoscale_trace(self, on):
        self.canvas.autoscale_trace = on
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

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_()


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
class FrogCanvas(FigureCanvasQTAgg):
    limits_changed = Signal()        # zoom/reset happened → window syncs dialog
    log_toggle_requested = Signal()  # click on spectrum y-axis strip

    def __init__(self):
        self.fig = Figure(facecolor=PALETTE["plot_bg"])
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # hspace must leave room for the x-labels of the top row plus the
        # title of the bottom row, or they collide. Margins sized for the
        # 9-pt plot fonts; keep left in sync with set_proportions.
        gs = self.fig.add_gridspec(2, 2,
                                   height_ratios=[1.8, 1.0],
                                   width_ratios=[1.0, 1.0],
                                   hspace=0.31, wspace=0.14,
                                   left=0.10, right=0.99,
                                   top=0.95, bottom=0.095)
        self.ax_spec  = self.fig.add_subplot(gs[0, 0])
        self.ax_trace = self.fig.add_subplot(gs[0, 1])
        self.ax_ac    = self.fig.add_subplot(gs[1, :])

        (self.line_spec,) = self.ax_spec.plot([], [], color=PALETTE["accent"], lw=1.6)
        (self.line_ac,)   = self.ax_ac.plot([], [], color=PALETTE["accent2"], lw=1.6)
        self.im = self.ax_trace.imshow(np.zeros((2, 2)), origin="lower",
                                       aspect="auto", cmap="magma",
                                       extent=[-1, 1, 0, 1])
        self._style()

        self.autoscale_y = False
        self.autoscale_x = True
        self.autoscale_ac_x = True
        self.autoscale_ac_y = True
        # Unlike the spectrum's flags (consulted every frame), this one only
        # bites when a new scan calls init_trace: manual trace bounds are meant
        # to outlive the scan they were dialled in on.
        self.autoscale_trace = True
        self._lw = 1.6

        # ── FROG-trace display settings ───────────────────────────────────
        # The threshold hides weak pixels from the PLOT only: the raw trace is
        # kept here and re-rendered on every settings change, and what the scan
        # worker recorded (and what gets exported / autocorrelated) is never
        # touched.
        self._trace_raw    = None    # last full trace, unmasked
        self._trace_thresh = 0.0     # display floor, fraction of peak (0 = off)
        self._cmap_name    = "magma"
        self._cmap_rev     = False
        self._bg_static    = None    # defined before _apply_cmap touches it
        self._apply_cmap()           # installs the masked-pixel colour

        # ── Mouse interaction: rubber-band zoom + axis-click log toggle ──
        # The rectangle is a Qt child widget composited above the canvas, so
        # the blit background cache never contains it.
        self._drag_ax = None
        self._drag_start = None      # (x, y) in mpl display coords
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

        # ── Blitting state ────────────────────────────────────────────────
        # These three are drawn by hand every frame; exclude them from draw().
        self.line_spec.set_animated(True)
        self.line_ac.set_animated(True)
        self.im.set_animated(True)
        self._animated = [(self.ax_spec, self.line_spec),
                          (self.ax_trace, self.im),
                          (self.ax_ac, self.line_ac)]
        self._bg = None            # cached full-figure background (no animated)
        # Background with the trace image + AC line already composited, so a
        # live spectrum frame only has to draw the spectrum line on top.
        # Invalidated whenever either of those artists (or the figure) changes.
        self._bg_static = None
        # batch(): collapse several update_* calls into one blit/draw.
        self._batch = False
        self._want_blit = False
        self._want_full = False
        self._clim_peak = None     # last clim top actually applied to the image
        # Cache last-applied limits so we only force a full redraw when they
        # actually move (autoscale otherwise re-sets identical limits each frame).
        self._xlim_cache = None
        self._ylim_cache = None
        self._ac_xlim_cache = None
        self._ac_ylim_cache = None
        self._trace_xlim_cache = None
        self._trace_ylim_cache = None
        # Axis y-positions are cached after first draw so set_proportions can
        # reposition axes without touching the row heights.
        self._ay_spec = self._ay_trace = self._ay_ac = None
        # Re-captures the background after every real draw (init, resize, or any
        # of our draw_idle() calls) and re-renders the animated artists on top.
        self.mpl_connect('draw_event', self._on_draw)

    def _style(self):
        specs = [(self.ax_spec,  "Wavelength (nm)", "Counts",          "Spectrum"),
                 (self.ax_trace, "Delay (fs)",      "Wavelength (nm)", "FROG Trace"),
                 (self.ax_ac,    "Delay (fs)",      "AC (a.u.)",       "Autocorrelation")]
        for ax, xl, yl, title in specs:
            ax.set_facecolor(PALETTE["plot_bg"])
            ax.tick_params(colors=PALETTE["text_dim"], labelsize=8.5)
            for s in ax.spines.values():
                s.set_edgecolor(PALETTE["border"])
            ax.set_xlabel(xl, color=PALETTE["text_dim"], fontsize=9.5, labelpad=2)
            ax.set_ylabel(yl, color=PALETTE["text_dim"], fontsize=9.5)
            ax.set_title(title, color=PALETTE["accent"], fontsize=10.5,
                         fontweight="bold", loc="left", pad=5)
        self.ax_spec.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)
        self.ax_ac.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)

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
        self._style()   # re-applies axes/tick/label/grid colors from PALETTE
        self._apply_cmap()   # masked pixels must follow the new background
        self.draw_idle()

    # ── Blitting core ─────────────────────────────────────────────────────
    def _on_draw(self, event):
        """After any real (full) draw, cache the background and paint the
        animated artists on top so they survive resizes and forced redraws."""
        if event is not None and event.canvas is not self:
            return
        if self._ay_spec is None:
            p = self.ax_spec.get_position();  self._ay_spec  = (p.y0, p.height)
            p = self.ax_trace.get_position(); self._ay_trace = (p.y0, p.height)
            p = self.ax_ac.get_position();    self._ay_ac    = (p.y0, p.height)
        self._bg = self.copy_from_bbox(self.fig.bbox)
        self._bg_static = None       # figure changed; recomposite lazily
        for ax, art in self._animated:
            ax.draw_artist(art)

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
        if self._bg_static is None:
            self.restore_region(self._bg)
            self.ax_trace.draw_artist(self.im)
            self.ax_ac.draw_artist(self.line_ac)
            self._bg_static = self.copy_from_bbox(self.fig.bbox)
        self.restore_region(self._bg_static)
        self.ax_spec.draw_artist(self.line_spec)
        self.blit(self.fig.bbox)

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

    def _cancel_drag(self):
        self._rubber.hide()
        self._drag_ax = None
        self._drag_start = None

    def _clamped_point(self, event):
        """Current cursor position clamped to the drag axes' live bbox, so a
        drag never extends into a neighboring axes or outside the figure."""
        bb = self._drag_ax.get_window_extent()
        return (min(max(event.x, bb.x0), bb.x1),
                min(max(event.y, bb.y0), bb.y1))

    def _on_press(self, event):
        axes = (self.ax_spec, self.ax_trace, self.ax_ac)
        if event.button == 3:
            if self._drag_ax is not None:      # right-click aborts a drag
                self._cancel_drag()
            elif event.inaxes in axes:
                self.reset_axes(event.inaxes)
            return
        if event.button != 1:
            return
        if event.inaxes in axes:
            self._drag_ax = event.inaxes
            self._drag_start = (event.x, event.y)
        elif event.inaxes is None and self._hit_spec_yaxis(event):
            self.log_toggle_requested.emit()

    def _on_motion(self, event):
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
            self._xlim_cache = (xlo, xhi)
            self._ylim_cache = (ylo, yhi)
        elif ax is self.ax_ac:
            self.autoscale_ac_x = False
            self.autoscale_ac_y = False
            self._ac_xlim_cache = (xlo, xhi)
            self._ac_ylim_cache = (ylo, yhi)
        elif ax is self.ax_trace:
            # update_trace never touches limits, so nothing to freeze for the
            # rest of this scan — but the NEXT scan's init_trace would snap the
            # view back, and the dialog must show the zoom as manual bounds.
            self.autoscale_trace = False
            self._trace_xlim_cache = (xlo, xhi)
            self._trace_ylim_cache = (ylo, yhi)
        self.draw_idle()
        self.limits_changed.emit()

    def reset_axes(self, ax):
        """Right-click: restore the full data range and resume autoscaling."""
        if ax is self.ax_spec:
            self.fit_xy()                      # fits, but freezes autoscale…
            self.autoscale_x = True            # …so re-enable live follow
            self.autoscale_y = True
        elif ax is self.ax_trace:
            self.autoscale_trace = True        # resume following the scan range
            x0, x1, y0, y1 = self.im.get_extent()
            self.ax_trace.set_xlim(x0, x1)
            self.ax_trace.set_ylim(y0, y1)
            self._trace_xlim_cache = self.ax_trace.get_xlim()
            self._trace_ylim_cache = self.ax_trace.get_ylim()
        elif ax is self.ax_ac:
            self.autoscale_ac_x = True
            self.autoscale_ac_y = True
            if len(self.line_ac.get_xdata()) > 1:
                self.ax_ac.set_autoscalex_on(True)
                self.ax_ac.set_autoscaley_on(True)
                self.ax_ac.relim()
                self.ax_ac.autoscale_view()
                self._ac_xlim_cache = self.ax_ac.get_xlim()
                self._ac_ylim_cache = self.ax_ac.get_ylim()
        self.draw_idle()
        self.limits_changed.emit()

    def fit_y(self):
        """One-shot Y auto-fit; returns new (ymin, ymax)."""
        self.ax_spec.set_autoscaley_on(True)
        self.ax_spec.relim()
        self.ax_spec.autoscale_view(scalex=False)
        self._ylim_cache = self.ax_spec.get_ylim()
        self.draw_idle()
        return self.ax_spec.get_ylim()

    def fit_xy(self):
        """One-shot fit of both spectrum axes, then freezes auto-scale."""
        # Any earlier set_xlim/set_ylim (spinboxes, rubber-band zoom) turned
        # matplotlib's internal autoscale off, making autoscale_view a no-op.
        self.ax_spec.set_autoscalex_on(True)
        self.ax_spec.set_autoscaley_on(True)
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.autoscale_x = False
        self.autoscale_y = False
        self._xlim_cache = self.ax_spec.get_xlim()
        self._ylim_cache = self.ax_spec.get_ylim()
        self.draw_idle()

    def set_ylim(self, ymin, ymax):
        if ymin < ymax:
            self.ax_spec.set_ylim(ymin, ymax)
            self._ylim_cache = (ymin, ymax)
            self.draw_idle()

    def set_xlim(self, xmin, xmax):
        if xmin < xmax:
            self.ax_spec.set_xlim(xmin, xmax)
            self._xlim_cache = (xmin, xmax)
            self.draw_idle()

    # ── FROG trace: manual bounds and colour ──────────────────────────────
    def set_trace_xlim(self, xmin, xmax):
        """Delay axis (fs) of the FROG trace."""
        if xmin < xmax:
            self.ax_trace.set_xlim(xmin, xmax)
            self._trace_xlim_cache = (xmin, xmax)
            self.draw_idle()

    def set_trace_ylim(self, ymin, ymax):
        """Wavelength axis (nm) of the FROG trace."""
        if ymin < ymax:
            self.ax_trace.set_ylim(ymin, ymax)
            self._trace_ylim_cache = (ymin, ymax)
            self.draw_idle()

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
        self._ylim_cache = self.ax_spec.get_ylim()
        self.draw_idle()

    def set_linewidth(self, lw):
        self._lw = lw
        self.line_spec.set_linewidth(lw)
        self.line_ac.set_linewidth(lw)
        self.draw_idle()

    def set_proportions(self, spec_frac):
        """Resize spectrum vs FROG trace columns; spec_frac = 0.15–0.55."""
        if self._ay_spec is None:
            return
        L, R = 0.10, 0.99
        total_w = R - L
        # Gap matches the initial gridspec spacing so the default 50% split
        # reproduces the startup layout (equal columns) with no jump, and the
        # FROG trace's y-axis label never collides with the spectrum plot.
        gap = 0.055
        avail = total_w - gap
        spec_w = spec_frac * avail
        trace_w = avail - spec_w
        if trace_w < 0.12:
            return
        sy0, sh = self._ay_spec
        ty0, th = self._ay_trace
        ay0, ah = self._ay_ac
        self.ax_spec.set_position([L, sy0, spec_w, sh])
        self.ax_trace.set_position([L + spec_w + gap, ty0, trace_w, th])
        self.ax_ac.set_position([L, ay0, total_w, ah])
        self.draw_idle()

    def update_spectrum(self, wl, spectrum):
        self.line_spec.set_data(wl, spectrum)
        changed = False
        if self.autoscale_x:
            xl = (float(wl[0]), float(wl[-1]))
            if xl != self._xlim_cache:
                self.ax_spec.set_xlim(*xl); self._xlim_cache = xl; changed = True
        if self.autoscale_y:
            self.ax_spec.set_autoscaley_on(True)
            self.ax_spec.relim()
            self.ax_spec.autoscale_view(scalex=False)
            yl = self.ax_spec.get_ylim()
            if yl != self._ylim_cache:
                self._ylim_cache = yl; changed = True
        if changed:
            self._request_full()      # limits moved: full redraw + re-cache bg
        else:
            self._request_blit(spec_only=True)

    def init_trace(self, delays, wl):
        # A new scan can span a different delay range; resume following it,
        # mirroring the trace-view reset below.
        self.autoscale_ac_x = True
        self.autoscale_ac_y = True
        self._trace_raw = np.zeros((wl.size, delays.size))
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
        self.draw_idle()

    def update_trace(self, trace):
        self._trace_raw = trace
        self._render_trace()

    def _render_trace(self):
        """Push the stored trace to the image, applying the display threshold.

        The colour scale is computed from the RAW trace, so moving the
        threshold only removes pixels — it never restretches the colours of
        the ones that survive.
        """
        raw = self._trace_raw
        if raw is None:
            return
        peak = max(float(raw.max()), 1.0)
        data = (np.ma.masked_less(raw, self._trace_thresh * peak)
                if self._trace_thresh > 0 else raw)
        self.im.set_data(data)
        # set_clim invalidates the image's cached RGBA (full re-normalize +
        # re-colormap), so only touch it when the peak moved visibly (>0.5%).
        if self._clim_peak is None or abs(peak - self._clim_peak) > 0.005 * self._clim_peak:
            self.im.set_clim(0, peak)
            self._clim_peak = peak
        self._bg_static = None        # trace pixels changed
        # Extent/limits are fixed by init_trace, so this is always a fast blit.
        self._request_blit()

    def update_ac(self, delays, ac):
        self.line_ac.set_data(delays, ac)
        self._bg_static = None        # AC line changed
        changed = False
        if self.autoscale_ac_x and delays.size > 1:
            xl = (float(delays[0]), float(delays[-1]))
            if xl != self._ac_xlim_cache:
                self.ax_ac.set_xlim(*xl); self._ac_xlim_cache = xl; changed = True
        if self.autoscale_ac_y and ac.size:
            # Stepped autoscale: during a scan the AC peak grows with nearly
            # every column, and each ylim move forces a full redraw. Grow with
            # 25% headroom so limits only step a handful of times per scan,
            # and shrink only once the data has fallen well below the view.
            dmax = float(ac.max())
            dmin = min(0.0, float(ac.min()))
            lo, hi = self._ac_ylim_cache if self._ac_ylim_cache else (0.0, 0.0)
            if dmax > hi or dmax < 0.55 * hi or dmin < lo:
                yl = (dmin, dmax * 1.25 if dmax > 0 else 1.0)
                self.ax_ac.set_ylim(*yl)
                self._ac_ylim_cache = yl; changed = True
        if changed:
            self._request_full()
        else:
            self._request_blit()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
# Export formats offered by the header menu: key -> (menu label, suffix,
# file-dialog filter, writer(path, result)). .dwc is the default; only the
# npz carries the raw counts, background frames and metadata.
EXPORT_FORMATS = {
    "dwc": ("FROG trace  (.dwc)",    ".dwc", "FROG trace (*.dwc)",    write_dwc),
    "npz": ("NumPy archive  (.npz)", ".npz", "NumPy archive (*.npz)", write_npz),
    "csv": ("CSV table  (.csv)",     ".csv", "CSV table (*.csv)",     write_csv),
}


class FrogWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lillypad — Fast")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumSize(1180, 760)
        self._theme = "dark"

        self.scan_cfg      = FrogScanConfig(delay_start_fs=-500, delay_stop_fs=500,
                                            delay_step_fs=1.0, zero_pos_um=150000.0)
        self._stage_units_fs = True     # jog/move fields default to fs
        self.background    = None
        self.last_spectrum = None
        self.result        = None
        self._worker       = None
        self._export_worker = None
        self._scan_trace   = None
        self._scan_delays  = None
        # Cached at scan start so _on_column can redraw the spectrum panel
        # without touching self.spec from the GUI thread while the worker is
        # inside acquire().
        self._scan_wl      = None
        self._feed_was_on  = True
        self._pending_fit  = True
        self._export_fmt   = "dwc"
        # Which beam the simulated spectrometer measures (see PULSE_SHAPES).
        # Kept on the window, not the simulator, so the choice survives a swap
        # to real hardware and back.
        self.sim_pulse     = DEFAULT_PULSE
        self.sim_gate      = "shg"

        self._build_hardware_sim()

        self.dlg_settings = AcquisitionSettingsDialog(self)
        self.dlg_hardware = HardwareDialog(self, self)
        self.spin_avg  = self.dlg_settings.spin_avg
        self.spin_idle = self.dlg_settings.spin_idle
        self.spin_wait = self.dlg_settings.spin_wait
        # The threshold is applied live, not just at scan start, so the lamp
        # reflects the setting you are in the middle of tuning.
        self.dlg_settings.spin_sat.valueChanged.connect(self._on_sat_fraction)

        self._build_ui()

        # The device default (100 ms) and the spinbox default differ; push the
        # spinbox value so the display matches reality. Safe without the
        # device lock — the feed thread doesn't exist yet.
        self.spec.set_integration_time(self.spin_integration.value())

        self.stage.move_to(_um_to_stage(self.scan_cfg.zero_pos_um))
        self._refresh_positions()

        # FIX 2/3 — acquisition runs on a worker thread, paced to integration.
        self._feed = LiveFeedWorker(lambda: self.spec)
        self._feed.spectrum_ready.connect(self._on_spectrum)
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
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(60)      # ~16 fps display cadence
        self._display_timer.timeout.connect(self._display_tick)
        self._display_timer.start()

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

    def _build_hardware_sim(self):
        self.stage = SimulatedStage(travel_mm=300.0)
        self.spec  = self._make_sim_spectrometer()

    def _apply_stage(self, new_stage):
        """Swap in a new stage. Returns (ok, error) — on failure NOTHING has
        changed and `new_stage` is still the caller's to dispose of.

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
            # A simulated spectrometer reads the live stage, so re-point it.
            if isinstance(self.spec, SimulatedSpectrometer):
                self.spec.stage = self.stage
            if isinstance(new_stage, SimulatedStage):
                new_stage.move_to(_um_to_stage(self.scan_cfg.zero_pos_um))
            else:
                # Real stage: adopt its ACTUAL position as zero-delay so
                # "Move to 0 fs" can never slam it into a travel limit.
                self.scan_cfg.zero_pos_um = _stage_to_um(new_stage.get_position())
            self._update_stage_unit_ranges()  # new travel + possibly new zero
            self._refresh_positions()   # inside the lock — we still own the stage
        self._refresh_scan_um()         # zero may have moved: fs→um previews
        return True, ""

    def _apply_spectrometer(self, new_spec):
        """Swap in a new spectrometer. Returns (ok, error); see _apply_stage."""
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
            self.spec.set_integration_time(self.spin_integration.value())
            # New device, new full scale — a latched warning about the old one
            # would be meaningless (and its threshold plain wrong).
            self._reset_saturation()
            # Re-centre spectrum when a new spectrometer comes online
            self._pending_fit = True
            self.canvas.autoscale_x = True
            if hasattr(self, 'dlg_graphics'):
                self.dlg_graphics.chk_auto_x.setChecked(True)
        return True, ""

    def _use_sim_stage(self):
        ok, err = self._apply_stage(SimulatedStage(travel_mm=300.0))
        if not ok:
            return False, err
        self.status.showMessage("Stage: simulated.", 4000)
        return True, "Stage set to simulated."

    def _connect_real_stage(self):
        try:
            devices = list_kinesis_stages()   # brief per-device model query
        except Exception as e:
            return False, f"Stage connect failed: {e}"
        if not devices:
            return False, "Stage connect failed: No Kinesis devices found."
        if len(devices) > 1:
            conn = DevicePickerDialog.pick(
                self.dlg_hardware, devices, "Select Kinesis Stage",
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
        self.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.scan_cfg.zero_pos_um:.1f} um).")

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
        self.status.showMessage(f"Spectrometer: {self.spec.name}.", 4000)
        return True, f"Simulated: {self.spec.pulse_label}."

    def _connect_real_spectrometer(self):
        try:
            devices = list_seabreeze_spectrometers()  # quick USB descriptor query
        except Exception as e:
            return False, f"Spectrometer connect failed: {e}"
        if not devices:
            return False, "Spectrometer connect failed: No spectrometers found."
        if len(devices) > 1:
            serial = DevicePickerDialog.pick(
                self.dlg_hardware, devices, "Select Spectrometer",
                f"{len(devices)} spectrometers found — choose one:")
            if serial is None:
                return False, "Cancelled — spectrometer unchanged."
        else:
            serial = devices[0][1]
        try:
            # "?" = serial unreadable; fall back to first-device auto-connect.
            spec = (SeabreezeSpectrometer(serial=serial)
                    if serial and serial != "?" else SeabreezeSpectrometer())
        except Exception as e:
            return False, f"Spectrometer connect failed: {e}"
        ok, err = self._apply_spectrometer(spec)
        if not ok:
            self._drop(spec)
            return False, err
        self.status.showMessage(f"Spectrometer: {spec.name}", 5000)
        return True, f"Spectrometer connected: {spec.name}"

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
        self.btn_hw = QPushButton("Hardware")
        self.btn_hw.clicked.connect(self.dlg_hardware.toggle)
        tb.addWidget(self.btn_hw)
        b_gfx = QPushButton("Graphics Settings")
        b_gfx.clicked.connect(lambda: self.dlg_graphics.toggle())
        tb.addWidget(b_gfx)
        tb.addWidget(self._build_export_button())
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(QIcon(str(SUN_ICON)))
        self.btn_theme.setIconSize(QSize(18, 18))
        self.btn_theme.setFixedWidth(42)
        self.btn_theme.setToolTip("Switch to light mode")
        self.btn_theme.clicked.connect(self._toggle_theme)
        tb.addWidget(self.btn_theme)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        ctrl = QWidget()
        ctrl.setMinimumWidth(280); ctrl.setMaximumWidth(320)
        cl = QVBoxLayout(ctrl); cl.setSpacing(8); cl.setContentsMargins(0, 0, 4, 0)
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

        # Overlay auto-fit button: child of canvas so it floats in the margin area
        self.btn_autofit = QPushButton(self.canvas)
        self.btn_autofit.setObjectName("overlay")
        if RESCALE_ICON.exists():
            self.btn_autofit.setIcon(QIcon(str(RESCALE_ICON)))
            self.btn_autofit.setIconSize(QSize(18, 18))
        else:
            self.btn_autofit.setText("↔↕")
        self.btn_autofit.setFixedSize(30, 30)
        self.btn_autofit.setToolTip("Auto-fit spectrum X and Y axes to current data")
        self.btn_autofit.move(6, 6)
        self.btn_autofit.show()
        self.btn_autofit.clicked.connect(self._autofit_spectrum)

        self.dlg_graphics = GraphicsSettingsDialog(self.canvas, self)
        # Mouse zoom/reset keeps the dialog's spinboxes and auto-scale
        # checkboxes truthful; the y-axis click routes through chk_log so the
        # checkbox stays the single source of truth for the log scale.
        self.canvas.limits_changed.connect(self.dlg_graphics.sync_limits)
        self.canvas.log_toggle_requested.connect(self.dlg_graphics.chk_log.toggle)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_sat = QLabel(""); self.lbl_sat.setObjectName("satok")
        self.lamp = StatusLamp()
        self.status.addPermanentWidget(self.lbl_sat)
        self.status.addPermanentWidget(self.lamp)
        self._reset_saturation()
        self.status.showMessage("Simulated hardware — ready (fast build).", 4000)

    def _build_export_button(self):
        """Header export control — a drop-down of the output formats. Picking
        one makes it the active format and saves straight away; the sidebar
        Save button follows the same choice."""
        menu = QMenu(self)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._export_actions = {}
        for key, (label, _suffix, _filt, _writer) in EXPORT_FORMATS.items():
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(key == self._export_fmt)
            act.triggered.connect(lambda _checked=False, k=key: self._export_as(k))
            group.addAction(act)
            menu.addAction(act)
            self._export_actions[key] = act
        self.btn_export = QPushButton("Export")
        self.btn_export.setToolTip("Save the last scan — pick the output format")
        self.btn_export.setMenu(menu)
        return self.btn_export

    def _build_spectrum_group(self):
        grp = QGroupBox("Spectrum")
        lay = QVBoxLayout(grp); lay.setSpacing(6)
        self.btn_feed = QPushButton("STOP FEED")
        self.btn_feed.setObjectName("danger")
        self.btn_feed.setCheckable(True); self.btn_feed.setChecked(True)
        self.btn_feed.toggled.connect(self._toggle_feed)
        lay.addWidget(self.btn_feed)
        lay.addWidget(QLabel("Integration Time"))
        self.spin_integration = QSpinBox()
        self.spin_integration.setRange(1, 10000); self.spin_integration.setValue(10)
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
        lay.addWidget(self.spin_integration)
        lay.addWidget(_hline())
        self.btn_dark = QPushButton("Record Dark")
        self.btn_dark.setObjectName("accent")
        self.btn_dark.clicked.connect(self._capture_dark)
        lay.addWidget(self.btn_dark)
        self.chk_dark = QCheckBox("Subtract Dark")
        self.chk_dark.setEnabled(False)
        lay.addWidget(self.chk_dark)
        return grp

    def _build_stage_group(self):
        grp = QGroupBox("Stage")
        lay = QVBoxLayout(grp); lay.setSpacing(6)

        jog = QHBoxLayout()
        self.btn_minus = QPushButton("−"); self.btn_plus = QPushButton("+")
        self.btn_minus.clicked.connect(lambda: self._jog(-1))
        self.btn_plus.clicked.connect(lambda: self._jog(+1))
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setDecimals(0)
        self.spin_step.setSuffix(" fs")     # value set after ranges, below
        self.btn_units = QPushButton("fs")
        self.btn_units.setFixedWidth(38)
        self.btn_units.setToolTip("Toggle jog/move units between optical delay (fs) "
                                  "and stage position (um)")
        self.btn_units.clicked.connect(self._toggle_stage_units)
        jog.addWidget(self.btn_minus); jog.addWidget(self.spin_step); jog.addWidget(self.btn_plus)
        jog.addWidget(self.btn_units)
        lay.addLayout(jog)

        mv = QHBoxLayout()
        self.spin_moveto = QDoubleSpinBox()
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

        row = QHBoxLayout()
        row.addWidget(QLabel("Stage"))
        self.lbl_moving = QLabel("idle"); self.lbl_moving.setObjectName("dim")
        self.lbl_moving.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.lbl_moving)
        lay.addLayout(row)

        lay.addWidget(_hline())
        for label, attr in [("Current position", "lbl_pos"), ("Zero-delay pos", "lbl_zero")]:
            r = QHBoxLayout()
            t = QLabel(label); t.setObjectName("dim")
            v = QLabel("— um"); v.setObjectName("readout")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            setattr(self, attr, v)
            r.addWidget(t); r.addWidget(v)
            lay.addLayout(r)

        self.btn_set_zero = QPushButton("Set Current Position as 0 fs")
        self.btn_set_zero.setObjectName("accent")
        self.btn_set_zero.clicked.connect(self._mark_zero)
        lay.addWidget(self.btn_set_zero)
        self._update_stage_unit_ranges()   # stage + scan_cfg exist by now
        self.spin_step.setValue(100.0)     # after ranges: default 100 fs jog
        return grp

    def _build_scan_group(self):
        grp = QGroupBox("FROG Scan")
        lay = QVBoxLayout(grp); lay.setSpacing(6)
        g = QGridLayout(); g.setSpacing(4)

        def row(r, label, spin, suffix, dec, lo, hi, val):
            g.addWidget(QLabel(label), r, 0)
            spin.setRange(lo, hi); spin.setDecimals(dec)
            spin.setValue(val); spin.setSuffix(suffix)
            g.addWidget(spin, r, 1)
            eq = QLabel(""); eq.setObjectName("dim")
            g.addWidget(eq, r, 2)
            return eq

        self.spin_start = QDoubleSpinBox()
        self.spin_stop  = QDoubleSpinBox()
        self.spin_step_fs = QDoubleSpinBox()
        self.eq_start = row(0, "Start", self.spin_start, " fs", 1, -1e6, 1e6, -500.0)
        self.eq_stop  = row(1, "Stop",  self.spin_stop,  " fs", 1, -1e6, 1e6,  500.0)
        row(2, "Step", self.spin_step_fs, " fs", 4, 0.0001, 1e5, 1.0)
        lay.addLayout(g)
        for s in (self.spin_start, self.spin_stop):
            s.valueChanged.connect(self._refresh_scan_um)

        self.chk_bg = QCheckBox("Capture background (before/after)")
        self.chk_bg.setChecked(True)
        lay.addWidget(self.chk_bg)

        self.btn_scan = QPushButton("Measure FROG")
        self.btn_scan.setObjectName("accent")
        self.btn_scan.clicked.connect(self._start_scan)
        lay.addWidget(self.btn_scan)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        lay.addWidget(self.progress)

        self.lbl_fwhm = QLabel("AC FWHM:  — fs")
        self.lbl_fwhm.setObjectName("readout")
        self.lbl_fwhm.setToolTip("Autocorrelation width, not pulse width "
                                 "(differ by a shape-dependent factor).")
        lay.addWidget(self.lbl_fwhm)

        self.btn_save = QPushButton(f"Save Scan ({EXPORT_FORMATS[self._export_fmt][1]})")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(lambda: self._export_as(self._export_fmt))
        lay.addWidget(self.btn_save)

        self._refresh_scan_um()
        return grp

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._apply_theme("light" if self._theme == "dark" else "dark")

    def _apply_theme(self, name):
        self._theme = name
        PALETTE.clear()
        PALETTE.update(LIGHT_PALETTE if name == "light" else DARK_PALETTE)
        app = QApplication.instance()
        apply_app_palette(app, PALETTE)
        app.setStyleSheet(build_stylesheet(PALETTE, name))
        self.canvas.apply_palette(PALETTE)
        self.lamp.update()          # repaints from the new PALETTE
        self.btn_theme.setIcon(QIcon(str(MOON_ICON if name == "light" else SUN_ICON)))
        self.btn_theme.setToolTip(
            "Switch to dark mode" if name == "light" else "Switch to light mode")
        self.status.showMessage(f"{name.capitalize()} mode.", 2000)

    # ── Saturation indicator ─────────────────────────────────────────────────
    def _set_lamp(self, state, text, obj):
        self.lamp.set_state(state)
        self.lbl_sat.setText(text)
        # Colour via objectName + repolish (not setStyleSheet) so a later theme
        # switch restyles the label along with everything else.
        if self.lbl_sat.objectName() != obj:
            self.lbl_sat.setObjectName(obj)
            self.lbl_sat.style().unpolish(self.lbl_sat)
            self.lbl_sat.style().polish(self.lbl_sat)

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

    def _reset_saturation(self):
        """Clear a latched warning and re-read the effective full scale."""
        self._sat_latched = False
        self._sat_frames  = 0            # saturated frames so far this scan
        self._sat_worst   = (0, 0.0, 0)  # (n_pixels, delay_fs, column index)
        full = self._full_scale()
        if not full:
            self.lamp.setToolTip(
                "This spectrometer does not report a full-scale value, so "
                "saturation cannot be checked. Set one in Hardware → Full scale.")
            self._set_lamp("unknown", "— % FS", "satok")
        else:
            source = ("override" if self.scan_cfg.saturation_counts else "device")
            self.lamp.setToolTip(
                f"Detector headroom — full scale {full:.0f} counts ({source}), "
                f"saturated at {100 * self.scan_cfg.saturation_fraction:.0f}%")
            self._set_lamp("ok", "", "satok")

    def _update_saturation(self, raw):
        """Live headroom readout, driven by the feed's RAW (un-subtracted) frame.

        Skipped while a scan's warning is latched: that one records that the
        measurement is already compromised, and must not be scrolled away by
        whatever the feed sees once the scan hands the hardware back.
        """
        if self._sat_latched:
            return
        full = self._full_scale()
        if not full:
            return
        peak = float(np.max(raw)) if raw.size else 0.0
        frac = peak / full
        # Same threshold the scan worker uses, so the live lamp and the scan
        # warning can never disagree about what counts as saturated.
        sat_frac = self.scan_cfg.saturation_fraction
        if frac >= sat_frac:
            n = int(np.count_nonzero(raw >= sat_frac * full))
            self._set_lamp("sat", f"⚠ SATURATED  ({n} px)", "sat")
        elif frac >= SAT_WARN_FRACTION:
            self._set_lamp("warn", f"⚠ {100 * frac:.0f}% FS", "satwarn")
        else:
            self._set_lamp("ok", f"{100 * frac:.0f}% FS", "satok")

    # ── Live feed ─────────────────────────────────────────────────────────────
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
        if self._pending_fit:
            self._pending_fit = False
            QTimer.singleShot(150, self._autofit_spectrum)
        # NB: no stage read here. Polling get_position() at feed rate raced the
        # feed thread's own acquire() (which, for the simulator, itself reads
        # the stage). The readout is instead refreshed after every move we make
        # and from the scan worker's per-column read-back.

    def _display_tick(self):
        """Render the newest pending data — scan column or live frame."""
        if self._scan_dirty:
            self._render_scan_frame()
            return
        if self._live_frame is None:
            return
        wl, raw = self._live_frame
        self._live_frame = None
        self.last_spectrum = raw
        self._update_saturation(self._sat_frame if self._sat_frame is not None
                                else raw)
        self._sat_frame = None
        self._sat_peak = -1.0
        spectrum = raw
        if self.chk_dark.isChecked() and self.background is not None:
            spectrum = np.clip(raw - self.background, 0, None)
        self.canvas.update_spectrum(wl, spectrum)

    def _autofit_spectrum(self):
        """One-shot fit of spectrum X + Y; syncs Graphics Settings spinboxes."""
        self.canvas.fit_xy()
        if hasattr(self, 'dlg_graphics'):
            self.dlg_graphics.sync_limits()

    def _toggle_feed(self, on):
        if on:
            # During a scan the worker owns the hardware; arm the feed and let
            # _reset_scan_ui start it once the scan hands the devices back.
            if not self._scan_running():
                self._feed.resume()
            self.btn_feed.setText("STOP FEED"); self.btn_feed.setObjectName("danger")
        else:
            self._feed.pause()
            self.btn_feed.setText("START FEED"); self.btn_feed.setObjectName("accent")
        self.btn_feed.style().unpolish(self.btn_feed); self.btn_feed.style().polish(self.btn_feed)

    def _apply_integration_time(self):
        """Push the (debounced) integration time to the spectrometer."""
        ms = self.spin_integration.value()
        if self._scan_running():
            self.status.showMessage(
                "A scan is running — integration time applies to the next scan.", 4000)
            return
        with self._device_lock() as ok:
            if not ok:
                self.status.showMessage(FEED_BUSY_MSG, 4000)
                return
            try:
                self.spec.set_integration_time(ms)
            except Exception as e:
                self.status.showMessage(f"Integration time failed: {e}", 5000)

    def _capture_dark(self):
        if self.last_spectrum is None:
            if self._scan_running():
                self.status.showMessage("A scan is running — spectrometer is busy.", 3000)
                return
            with self._device_lock() as ok:
                if not ok:
                    self.status.showMessage(FEED_BUSY_MSG, 4000)
                    return
                self.last_spectrum = np.asarray(self.spec.acquire(), float)
        self.background = self.last_spectrum.copy()
        self.chk_dark.setEnabled(True); self.chk_dark.setChecked(True)
        self.status.showMessage("Dark recorded.", 3000)

    # ── Manual stage ──────────────────────────────────────────────────────────
    def _set_stage_controls_enabled(self, on):
        for w in (self.btn_minus, self.btn_plus, self.btn_moveto,
                  self.btn_home, self.btn_goto_zero, self.btn_set_zero,
                  self.btn_units):
            w.setEnabled(on)

    def _moving(self, on):
        self.lbl_moving.setText("MOVING" if on else "idle")
        self.lbl_moving.setObjectName("moving" if on else "dim")
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
        self._set_stage_controls_enabled(False)
        self._moving(True)
        try:
            with self._device_lock() as ok:
                if not ok:
                    self.status.showMessage(FEED_BUSY_MSG, 5000)
                    return
                fn()
                self._refresh_positions()   # still inside the lock
                if done_msg:
                    self.status.showMessage(done_msg, 3000)
        except Exception as e:
            self.status.showMessage(f"Stage error: {e}", 5000)
        finally:
            self._moving(False)
            self._set_stage_controls_enabled(True)

    def _travel_um(self):
        """Stage travel in um; 300 mm fallback when the adapter reports none."""
        t = getattr(self.stage, "travel_mm", None)
        return _stage_to_um(t if t else 300.0)

    def _update_stage_unit_ranges(self):
        """Set spin_step/spin_moveto ranges for the active unit, the connected
        stage's travel, and the current zero. Call after: unit toggle, stage
        swap, zero change."""
        travel_um = self._travel_um()
        pf, z = self.scan_cfg.pass_factor, self.scan_cfg.zero_pos_um
        if self._stage_units_fs:
            # Inward rounding (ceil min, floor max) so every integer fs in
            # range maps to a position strictly inside [0, travel].
            self.spin_step.setRange(
                1, math.floor(float(position_to_delay_fs(travel_um, 0.0, pf))))
            self.spin_moveto.setRange(
                math.ceil(float(position_to_delay_fs(0.0, z, pf))),
                math.floor(float(position_to_delay_fs(travel_um, z, pf))))
        else:
            self.spin_step.setRange(0.01, travel_um)
            self.spin_moveto.setRange(0.0, travel_um)

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
        def do():
            # Clamped move_to instead of move_by: the target can never leave
            # the travel range. get_position() runs inside the device lock.
            cur = _stage_to_um(self.stage.get_position())
            target = min(max(cur + sign * step_um, 0.0), self._travel_um())
            self.stage.move_to(_um_to_stage(target))
        self._stage_action(do)

    def _move_absolute(self):
        v = self.spin_moveto.value()
        pf, z = self.scan_cfg.pass_factor, self.scan_cfg.zero_pos_um
        target_um = float(delay_to_position_um(v, z, pf)) if self._stage_units_fs else v
        target_um = min(max(target_um, 0.0), self._travel_um())
        self._stage_action(lambda: self.stage.move_to(_um_to_stage(target_um)))

    def _move_to_zero(self):
        self._stage_action(
            lambda: self.stage.move_to(_um_to_stage(self.scan_cfg.zero_pos_um)))

    def _home_stage(self):
        self._stage_action(lambda: self.stage.home(), "Stage homed to 0 mm.")

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
        self.lbl_pos.setText(f"{_stage_to_um(self.stage.get_position()):.2f} um")
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
        if self.spec is None:
            self.status.showMessage("Connect a spectrometer first.", 4000); return

        c = self.scan_cfg
        c.delay_start_fs = self.spin_start.value()
        c.delay_stop_fs  = self.spin_stop.value()
        c.delay_step_fs  = self.spin_step_fs.value()
        c.n_average      = self.spin_avg.value()
        c.idle_shots     = self.spin_idle.value()
        c.wait_after_move_s = self.spin_wait.value() / 1000.0
        c.capture_background = self.chk_bg.isChecked()
        c.saturation_fraction  = self.dlg_settings.spin_sat.value() / 100.0
        c.abort_on_saturation  = self.dlg_settings.chk_abort_sat.isChecked()
        # c.saturation_counts is owned by the Hardware dialog's override field.

        try:
            delays = c.delays_fs()
        except ValueError as e:
            self.status.showMessage(f"Bad scan range: {e}", 4000); return
        wl = np.asarray(self.spec.wavelengths, float)

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
        self.btn_hw.setEnabled(False)
        self._set_stage_controls_enabled(False)

        self._scan_trace  = np.zeros((wl.size, delays.size))
        self._scan_delays = delays
        self._scan_wl     = wl
        # A pending render from a previous (aborted) scan must not fire
        # against the fresh, differently-sized arrays.
        self._scan_dirty  = False
        self._scan_last_i = -1
        self._live_frame  = None     # park any leftover live-feed frame too
        self.canvas.init_trace(delays, wl)
        self._reset_saturation(); self.progress.setValue(0)

        self.btn_scan.setObjectName("danger"); self.btn_scan.setText("Abort Scan")
        self.btn_scan.style().unpolish(self.btn_scan); self.btn_scan.style().polish(self.btn_scan)
        self.btn_save.setEnabled(False)

        self._worker = FrogScanWorker(self.stage, self.spec, c)
        self._worker.progress.connect(self._on_progress)
        self._worker.column_ready.connect(self._on_column)
        self._worker.background_ready.connect(self._on_background)
        self._worker.saturation_warning.connect(self._on_saturation)
        self._worker.finished_scan.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self.status.showMessage(f"FROG scan: {delays.size} points…", 0)

    def _on_progress(self, done, total):
        self.progress.setValue(int(100 * done / total))

    def _on_column(self, i, delay_fs, pos_um, col):
        """Slot for FrogScanWorker.column_ready — O(1), like _on_spectrum.

        Every column is RECORDED here (the data path must never drop), but
        rendering is deferred to _display_tick so a slow machine skips
        intermediate redraws instead of queuing them up.
        """
        self._scan_trace[:, i] = col
        self._scan_last_i = i
        self._scan_col = col
        self._scan_pos_um = pos_um
        self._scan_dirty = True

    def _render_scan_frame(self):
        """Draw the in-progress scan from the newest recorded column."""
        self._scan_dirty = False
        i = self._scan_last_i
        self.lbl_pos.setText(f"{self._scan_pos_um:.2f} um")
        ac = autocorrelation(self._scan_trace[:, :i + 1])
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
            if self._scan_wl is not None:
                spectrum = self._scan_col
                if self.chk_dark.isChecked() and self.background is not None:
                    spectrum = np.clip(spectrum - self.background, 0, None)
                self.canvas.update_spectrum(self._scan_wl, spectrum)
            self.canvas.update_trace(self._scan_trace)
            self.canvas.update_ac(self._scan_delays[:i + 1], ac)
        f = fwhm(self._scan_delays[:i + 1], ac)
        self.lbl_fwhm.setText(f"AC FWHM:  {f:.1f} fs" if np.isfinite(f) else "AC FWHM:  — fs")

    def _on_background(self, which, spectrum):
        self.status.showMessage(f"Background ({which}) captured.", 2500)

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

    def _on_finished(self, result):
        self.result = result
        # The final render below supersedes any pending intermediate tick.
        self._scan_dirty = False
        self.canvas.update_trace(result.trace)
        ac = result.autocorrelation()
        self.canvas.update_ac(result.delays_fs, ac)
        f = result.fwhm_fs()
        self.lbl_fwhm.setText(f"AC FWHM:  {f:.1f} fs" if np.isfinite(f) else "AC FWHM:  — fs")
        self.progress.setValue(100)
        self.status.showMessage(f"Scan complete — {result.trace.shape[1]} columns.", 5000)
        self.btn_save.setEnabled(True)
        self._reset_scan_ui()

    def _on_error(self, msg):
        self.status.showMessage(f"Scan: {msg}", 6000)
        self._reset_scan_ui()

    def _reset_scan_ui(self):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setObjectName("accent"); self.btn_scan.setText("Measure FROG")
        self.btn_scan.style().unpolish(self.btn_scan); self.btn_scan.style().polish(self.btn_scan)
        self.btn_hw.setEnabled(True)
        self._set_stage_controls_enabled(True)
        if self._feed_was_on and self.btn_feed.isChecked():
            self._feed.resume()

    # ── Save ──────────────────────────────────────────────────────────────────
    def _export_as(self, key):
        """Make `key` the active export format and write the last scan in it.
        With no scan yet this only switches the format."""
        self._export_fmt = key
        _label, suffix, filt, writer = EXPORT_FORMATS[key]
        self._export_actions[key].setChecked(True)
        self.btn_save.setText(f"Save Scan ({suffix})")
        if self.result is None:
            self.status.showMessage(f"Export format: {suffix} — no scan to save yet.", 4000)
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            self.status.showMessage("An export is already in progress.", 3000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save FROG scan",
                                              f"frog_scan{suffix}", filt)
        if not path:
            return
        if not path.lower().endswith(suffix):
            path += suffix
        # Formatting a full trace takes seconds — write it on a worker thread so
        # the window keeps painting (and the live feed keeps running).
        self.btn_save.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.status.showMessage(f"Saving → {path} …", 0)
        self._export_worker = ExportWorker(writer, path, self.result, self)
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_done(self, path):
        self._end_export()
        self.status.showMessage(f"Saved → {path}", 5000)

    def _on_export_error(self, msg):
        self._end_export()
        self.status.showMessage(f"Save failed: {msg}", 8000)

    def _end_export(self):
        self.btn_export.setEnabled(True)
        self.btn_save.setEnabled(self.result is not None)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._display_timer.stop()
        self._feed.stop()
        self._feed.wait(2000)
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort(); self._worker.wait(2000)
        # A half-written export would be a corrupt file, and the worker holds a
        # live reference to the result — let it finish before tearing down.
        if self._export_worker is not None and self._export_worker.isRunning():
            self.status.showMessage("Finishing export…", 0)
            self._export_worker.wait(30000)
        for dev in (self.stage, self.spec):
            try:
                dev.disconnect()
            except Exception:
                pass
        self.dlg_settings.close(); self.dlg_hardware.close(); self.dlg_graphics.close()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Lillypad")
    except (AttributeError, OSError):
        pass  # not Windows — silently skip
    app = QApplication(sys.argv)
    app.setApplicationName("Lillypad")
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    app.setFont(QFont("Segoe UI", 10, QFont.Normal))
    apply_app_palette(app, PALETTE)
    app.setStyleSheet(build_stylesheet(PALETTE, "dark"))

    win = FrogWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
