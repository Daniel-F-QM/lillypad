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

import os
import gc
import sys
import json
import math
import time
import shutil
import tempfile
import threading
import tracemalloc
import numpy as np
from contextlib import contextmanager, nullcontext
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
    QProgressBar, QFrame, QScrollArea, QSizePolicy, QStatusBar, QFileDialog,
    QDialog, QToolBar, QSlider, QLineEdit, QMenu, QComboBox, QRubberBand,
    QMessageBox, QInputDialog, QAbstractSpinBox
)
from PySide6.QtCore import (Qt, QTimer, QThread, Signal, QPointF, QSize, QRect,
                            QRectF, QPoint, QSignalBlocker, QObject)
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
from matplotlib.transforms import ScaledTranslation

from hardware import (SimulatedStage, SimulatedSpectrometer,
                      KinesisStage, ZaberStage, PiezoJenaStage,
                      SeabreezeSpectrometer, AvantesSpectrometer,
                      StitchedSpectrometer,
                      list_kinesis_stages, list_spectrometers,
                      list_avantes_spectrometers, list_seabreeze_spectrometers,
                      open_spectrometer,
                      avantes_trigger_options,
                      load_calibration_file, SEABREEZE_BACKENDS,
                      PULSE_SHAPES, DEFAULT_PULSE)
from scan import (FrogScanConfig, FrogScanWorker, fwhm,
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

# Per-spectrometer curves in multi-spectrometer mode. Okabe-Ito sky blue and
# orange — the highest-contrast pair in that set. Deliberately NOT palette
# keys: both read clearly on the dark and the light plot background, and they
# stay separable under deuteranopia, protanopia and tritanopia (and in
# greyscale), which the theme accents do not.
MEMBER_COLORS = ("#56B4E9", "#E69F00")      # SLOT order: (S1, S2)

# Shading for the overlap band in the per-spectrometer view: the stretch the
# stitch factor is fitted over and the two spectra are crossfaded across. Green
# because it has to sit UNDER both member curves without being confused for
# either, and Okabe-Ito's bluish green is the remaining unused hue in that set.
OVERLAP_BAND_COLOR = "#009E73"
OVERLAP_BAND_ALPHA = 0.16

# The two symmetry-difference curves of alignment mode, S(+x)-S(-x) and
# S(+2x)-S(-2x). Okabe-Ito again, and deliberately the two hues MEMBER_COLORS
# and the overlap band do NOT use: a difference curve can be on screen at the
# same time as either of those and must never be mistaken for one.
DIFF_COLORS = ("#D55E00", "#CC79A7")        # (+/-x, +/-2x)

# Simulated members offered alongside real devices in the multi-spectrometer
# slots, so the mode can be exercised without two spectrometers on the bench.
# These are sentinels, not device ids — _open_slot_device matches them BEFORE
# it hands anything to hardware.open_spectrometer, so they deliberately do not
# carry a "vendor:serial" tag. Each covers one overlapping half of the
# simulated signal band; see _make_sim_member for why halves and not two
# full-band copies.
SIM_SLOT_DEVICES = (("__sim_blue__", "Simulated — blue half"),
                    ("__sim_red__",  "Simulated — red half"))

FONT_STACK = "'Segoe UI','DejaVu Sans',Arial,sans-serif"


# ─────────────────────────────────────────────────────────────────────────────
# Value widgets that ignore the mouse wheel. Scrolling a panel must never edit
# a setting: ev.ignore() propagates the wheel event to the enclosing
# QScrollArea, so the panel under the cursor still scrolls. Keyboard, arrows
# and typing are untouched. Qt stylesheet type selectors match subclasses, so
# the QDoubleSpinBox/QSpinBox rules in build_stylesheet still apply.
class _NoWheel:
    def wheelEvent(self, ev):
        ev.ignore()


class SpinBox(_NoWheel, QSpinBox):             pass
class DoubleSpinBox(_NoWheel, QDoubleSpinBox): pass
class ComboBox(_NoWheel, QComboBox):           pass
class Slider(_NoWheel, QSlider):               pass


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


# ── Panel header band, in logical pixels ─────────────────────────────────────
# The overlay buttons above each plot panel (see FrogWindow._position_panel_
# buttons). Square and small enough that a row of them reads as one line with
# the panel title beside it; HDR_PAD is the gap between the row's bottom edge
# and the axes' top spine.
HDR_BTN = 24
HDR_GAP = 4
HDR_PAD = 6
# Icon side inside a header button. EVEN, and even after the 1 px border is
# taken off both sides of HDR_BTN: (24 - 2 - 14) / 2 = 4 exactly, so Qt centres
# the icon on a whole pixel instead of rounding half a pixel one way.
HDR_ICON = 14


def _glyph_icon(kind, color, px=HDR_ICON):
    """A play/stop glyph as a QIcon, drawn in `color`.

    Generated rather than loaded from icons/ because this one button has to
    re-colour itself twice over: once for its state (accent while stopped,
    danger while running) and once for the theme. A PNG can do neither, and the
    four files it would otherwise take would still be wrong the moment either
    palette changes. Same QPainter approach as the spinbox arrows above.

    Rendered at the screen's device pixel ratio: at 150% Windows scaling a
    logical-size pixmap is upscaled by the compositor, which both softens the
    edges and shifts them by a fraction of a pixel.
    """
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app is not None else 1.0
    n = max(1, round(px * dpr))
    pm = QPixmap(n, n)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    c = n / 2.0
    if kind == "play":
        # Centred on the triangle's CENTROID, not its bounding box: a
        # right-pointing triangle whose box is centred reads as sitting too far
        # right, because most of its area is on the left. The centroid of
        # (x0,·) (x0,·) (x0+w,·) is at x0 + w/3, so x0 = c - w/3 puts the mass
        # in the middle of the button.
        w, h = 0.52 * n, 0.62 * n
        x0 = c - w / 3.0
        p.drawPolygon(QPolygonF([QPointF(x0, c - h / 2), QPointF(x0, c + h / 2),
                                 QPointF(x0 + w, c)]))
    else:
        s = 0.56 * n
        p.drawRoundedRect(QRectF(c - s / 2, c - s / 2, s, s), 0.10 * n, 0.10 * n)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return QIcon(pm)


def build_stylesheet(pal, tag):
    uri_up, uri_dn = _make_arrow_icons(pal["text"], tag)
    uri_chk = _make_check_icon(pal["bg"], tag)
    return f"""
QMainWindow, QWidget {{ background-color:{pal['bg']}; color:{pal['text']};
    font-family:{FONT_STACK}; font-size:13px; font-weight:400; }}
QGroupBox {{ border:1px solid {pal['border']}; border-radius:6px; margin-top:11px;
    padding:6px 6px 6px 6px; font-size:11px; font-weight:700; color:{pal['text_dim']};
    letter-spacing:1.5px; text-transform:uppercase; }}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
QPushButton {{ background-color:{pal['surface']}; border:1px solid {pal['border']};
    border-radius:5px; padding:5px 12px; color:{pal['text']}; font-size:13px;
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
/* Narrow unit toggle: the default 14px side padding leaves too little text box
   inside its fixed width, and clipped "um". */
QPushButton#unit {{ padding:5px 4px; }}
/* Secondary actions in the side panel — same colours as the default/accent
   buttons, but on one text line's worth of padding so the panel fits without
   scrolling. */
QPushButton#compact {{ padding:4px 10px; font-size:12px; }}
QPushButton#accentcompact {{ padding:4px 10px; font-size:12px;
    border-color:{pal['accent']}; color:{pal['accent']}; }}
QPushButton#accentcompact:hover {{ background-color:{pal['accent']};
    color:{pal['bg']}; }}
QPushButton#overlay {{ border-color:{pal['accent']}; color:{pal['accent']};
    padding:0px; font-size:12px; border-radius:4px; }}
QPushButton#overlay:hover {{ background-color:{pal['accent']}; color:{pal['bg']}; }}
QPushButton#overlay:pressed {{ background-color:{pal['accent']}; color:{pal['bg']}; }}
/* The "no spectrometer connected" message centred on an empty spectrum panel.
   Accent, not danger: nothing is wrong, there is simply nothing connected yet
   — and the accent is what carries against the plot background it sits on,
   which the dim text colour was never meant to be read over. Transparent, so
   NoDeviceOverlay's own painted background (and its watermark) shows through. */
QLabel#nodev {{ color:{pal['accent']}; background:transparent; font-size:22px;
    font-weight:600; letter-spacing:0.5px; }}
/* Same header button, in the stop colour — the live feed's running state. */
QPushButton#overlaydanger {{ border-color:{pal['danger']}; color:{pal['danger']};
    padding:0px; font-size:12px; border-radius:4px; }}
QPushButton#overlaydanger:hover {{ background-color:{pal['danger']};
    color:{pal['bg']}; }}
QPushButton#overlaydanger:pressed {{ background-color:{pal['danger']};
    color:{pal['bg']}; }}
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
    padding:1px 16px 1px 6px; color:{pal['text']}; font-weight:400;
    min-height:18px; selection-background-color:{pal['accent']};
    selection-color:{pal['bg']}; }}
QDoubleSpinBox:focus, QSpinBox:focus {{ border-color:{pal['accent']}; }}
QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin:border; subcontrol-position:top right;
    width:14px; border-left:1px solid {pal['border']};
    border-bottom:1px solid {pal['border']};
    border-top-right-radius:4px; background:{pal['surface']}; }}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {{
    background:{pal['border_hover']}; }}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed {{
    background:{pal['accent']}; }}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin:border; subcontrol-position:bottom right;
    width:14px; border-left:1px solid {pal['border']};
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
QLabel#readout_sm {{ color:{pal['accent']}; font-weight:600; font-size:13px;
    font-family:'Cascadia Mono','Consolas',monospace; }}
/* The stage state, under its caption in the readout row. Bold in both states
   so the column keeps one width and does not twitch as it flips; _moving()
   swaps between the two object names. */
QLabel#idle {{ color:{pal['text_dim']}; font-size:12px; font-weight:700; }}
QLabel#moving {{ color:{pal['warn']}; font-size:12px; font-weight:700; }}
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


# Window/taskbar icon: the multi-frame .ico (16→256 px), NOT the 2481² PNG.
# Windows asks for 16/20/24/32/40/48/64 px depending on the surface (title bar,
# taskbar, Alt-Tab) and the monitor DPI. A single huge bitmap makes Qt decode
# ~25 MB and smooth-downscale it on the GUI thread for every one of those, which
# is slow and lossy; worse, with an explicit AppUserModelID (see main()) the
# shell no longer falls back to the .exe's own icon, so a window icon that is
# not ready in time gets cached as a blank taskbar button. Qt's ICO handler
# reads every frame, so QIcon can hand back the exact size instead.
ICON_ICO_PATH = resource_path("icons", "Lilypad.ico")
ICON_PATH    = resource_path("icons", "Lilypad.png")   # README / large uses
SUN_ICON     = resource_path("icons", "sun.png")
MOON_ICON    = resource_path("icons", "moon.png")
RESCALE_ICON = resource_path("icons", "rescale.png")
# Spectrum-panel view toggle, keyed by theme (the suffix names the theme each
# icon was drawn for, not what it depicts). The icons advertise the ACTION:
# the broken one means "split the pair apart", the continuous one "put it back
# together" — so the button shows the view you get by clicking it.
SPLIT_ICON = {"dark":  resource_path("icons", "broken_spectrum_dark.png"),
              "light": resource_path("icons", "broken_spectrum_light.png")}
MERGE_ICON = {"dark":  resource_path("icons", "continuous_spectrum_dark.png"),
              "light": resource_path("icons", "continuous_spectrum_light.png")}
# Plot-layout toggle, same convention as above: each icon DEPICTS the layout you
# get by clicking it. horizontal_* shows the tall spectrum beside the stacked
# trace/AC pair, vertical_* the two top panels over a full-width bottom one.
HORIZ_ICON = {"dark":  resource_path("icons", "horizontal_dark.png"),
              "light": resource_path("icons", "horizontal_light.png")}
VERT_ICON  = {"dark":  resource_path("icons", "vertical_dark.png"),
              "light": resource_path("icons", "vertical_light.png")}
# Alignment-mode sweep, on the spectrum panel's header. A crosshair: unlike the
# two pairs above this one depicts a MODE rather than an action, so both files
# show the same mark and differ only in the accent each theme draws it in.
ALIGN_ICON = {"dark":  resource_path("icons", "alignment_dark.png"),
              "light": resource_path("icons", "alignment_light.png")}
# Auto-stitch, on the spectrum panel's header. Depicts the ACTION — matching
# the two halves of a pair — so like ALIGN_ICON both files carry the same mark
# in their own theme's accent.
STITCH_ICON = {"dark":  resource_path("icons", "autostitch_dark.png"),
               "light": resource_path("icons", "autostitch_light.png")}
# Watermark behind the "nothing connected" message. ONE file for both themes,
# unlike the pairs above: it is drawn as an alpha mask and tinted from the live
# PALETTE (see NoDeviceOverlay), so it follows a theme switch by itself.
DISCONNECTED_ICON = resource_path("icons", "disconnected.png")


_APP_ICON = None


def app_icon():
    """The application icon, built once from the multi-resolution .ico.

    Cached lazily rather than at import: a QIcon may not be constructed before
    the QApplication exists. The pixmap loop realises the sizes Windows asks
    for while we are still in main(), so the HICON is ready by the time the
    native window is created and the shell samples the taskbar button."""
    global _APP_ICON
    if _APP_ICON is None:
        src = ICON_ICO_PATH if ICON_ICO_PATH.exists() else ICON_PATH
        _APP_ICON = QIcon(str(src)) if src.exists() else QIcon()
        for px in (16, 20, 24, 32, 40, 48, 64, 256):
            _APP_ICON.pixmap(px, px)
    return _APP_ICON


def app_dir():
    """Folder the program lives in — next to the .exe when frozen, else next
    to this file. User-editable data (calibration files) belongs HERE, not in
    resource_path()'s _MEIPASS bundle, which is read-only and re-extracted on
    every launch — files added there would silently vanish."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


CALIBRATION_DIR = app_dir() / "calibration_files"

# Preferences live beside the program for the same reason the calibration files
# do: app_dir() is writable and permanent, while resource_path()'s _MEIPASS
# bundle is re-extracted every launch. Never seeded, unlike the calibration
# folder — the absence of this file IS the default state, so deleting it is a
# full factory reset.
SETTINGS_PATH = app_dir() / "settings.json"
SETTINGS_VERSION = 1


def seed_calibration_dir():
    """First-run seeding for the frozen build: the bundle ships the repo's
    calibration files under _internal (sys._MEIPASS), but the user-editable
    folder lives next to the .exe. Copy the bundled files over ONLY if the
    folder does not exist yet — never touch a folder the user already owns."""
    if CALIBRATION_DIR.exists():
        return
    bundled = resource_path("calibration_files")
    if not bundled.is_dir() or bundled.resolve() == CALIBRATION_DIR.resolve():
        return
    try:
        CALIBRATION_DIR.mkdir(parents=True)
        for f in bundled.glob("*.txt"):
            shutil.copyfile(f, CALIBRATION_DIR / f.name)
    except OSError:
        pass   # e.g. read-only install dir — the menu just shows no files


# ─────────────────────────────────────────────────────────────────────────────
# Settings persistence
# ─────────────────────────────────────────────────────────────────────────────
# Display preferences and acquisition policy only — never anything bound to a
# particular piece of hardware. A serial number, an integration time or a stage
# zero describes the rig that happened to be plugged in last time, and restoring
# one silently would put a number the operator never entered into the next
# measurement. What is kept here is only how the program LOOKS and how it has
# been told to behave.
#
# A plain JSON file rather than QSettings and the registry, for the same reason
# the calibration files are a folder: the operator can open it, read it, correct
# it by hand and delete it.

_SETTINGS = None


def load_settings():
    """Read settings.json once per process; {} for anything wrong with it.

    Cached because two moments need the same answer and must not disagree:
    main() needs the theme before the QApplication is styled, and the window
    needs everything else after its widgets exist.

    A missing file is the ordinary first-run case, and a corrupt one is treated
    exactly the same way — defaults are always a working program, and a startup
    that refused to start because a preferences file lost a brace would be a far
    worse failure than a forgotten colormap.
    """
    global _SETTINGS
    if _SETTINGS is None:
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _SETTINGS = data if isinstance(data, dict) else {}
        except (OSError, ValueError, UnicodeDecodeError):
            _SETTINGS = {}
    return _SETTINGS


def save_settings(data):
    """Write settings.json atomically. True if it landed.

    Temp file plus os.replace, never an in-place rewrite: a crash partway
    through a rewrite leaves a TRUNCATED file, which is the one outcome worse
    than not saving at all — it loses the previous session's settings too.
    os.replace is atomic within a volume, hence the temp file in the same
    folder.

    No fsync deliberately: the rename already keeps the file from being torn,
    and flushing the disk on every autosave to defend a colormap choice against
    a power cut is not a trade worth making.

    Failure is silent, like seed_calibration_dir — a read-only install directory
    should cost the operator nothing but the memory of their preferences.
    """
    tmp = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            # default=float: matplotlib hands back numpy scalars from get_ylim()
            # and friends, and one of them must not abort the whole write.
            json.dump(data, f, indent=2, default=float)
            f.write("\n")
        os.replace(tmp, SETTINGS_PATH)
        return True
    except (OSError, TypeError, ValueError):
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def startup_theme(data):
    """The theme to build the whole program in, validated.

    Read straight from the settings dict instead of being restored through
    _apply_theme like every other setting: PALETTE is sampled at CONSTRUCTION
    time by the figure and by _build_ui's per-theme icon files, so the palette
    has to be right before the first widget exists. Restoring it afterwards
    would mean restyling the entire application at startup and trusting that
    every widget which read a colour on the way up has a refresh hook.
    """
    name = data.get("theme")
    return name if name in ("dark", "light") else "dark"


def install_theme(app, name):
    """Point the module-level PALETTE at `name` and restyle the application.

    Shared by main()'s startup and _apply_theme's runtime switch, which
    otherwise carried the same four lines twice — and a startup that styled
    itself even slightly differently from a toggle is exactly the drift that
    surfaces later as one stale widget colour.
    """
    PALETTE.clear()
    PALETTE.update(LIGHT_PALETTE if name == "light" else DARK_PALETTE)
    apply_app_palette(app, PALETTE)
    app.setStyleSheet(build_stylesheet(PALETTE, name))


def _as_bool(v):
    """Strict bool for restore.

    bool("false") is True, so a hand-edited file would otherwise mean the exact
    opposite of what it says. Anything that is not a real JSON boolean raises,
    and that one key falls back to its default.
    """
    if isinstance(v, bool):
        return v
    raise TypeError(v)


def _restore_pair(lo_sb, hi_sb, value):
    """Restore a (min, max) spinbox pair, low box first.

    Order matters: each box's valueChanged handler pushes BOTH boxes at the
    canvas, and the canvas setters ignore an inverted pair. Setting the low box
    first guarantees the second call — the always-valid one — is what lands.
    """
    lo, hi = value
    lo_sb.setValue(float(lo))
    hi_sb.setValue(float(hi))


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


# Said in three places (the auto-scale checkboxes) about one gesture, so it is
# written once — FrogCanvas.reset_axes is what implements it.
RIGHT_CLICK_HINT = ("Right-click the plot to toggle this: off fits the full "
                    "range and follows it live, on freezes the view where it "
                    "is.")


def _ease(current, target, dt, tau):
    """One-pole step from `current` toward `target` over `dt` seconds, with time
    constant `tau`.

    Solved rather than iterated, so the result depends only on elapsed TIME and
    not on how many times it was called: at a 1 ms exposure this runs ~16 times
    a second and at a 10 s exposure once every ten, and a per-call fraction
    would have made the response time a function of the exposure.
    """
    return target + (current - target) * math.exp(-dt / tau)


def _hline():
    f = QFrame(); f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Performance probe (opt-in: LILLYPAD_PERF=1)
# ─────────────────────────────────────────────────────────────────────────────
# Exists to answer "which render path is slow, and is anything growing?" with
# numbers instead of guesses. Off by default and reduced to a nullcontext() when
# off, so the instrumented paths cost nothing in normal use.
PERF = bool(os.environ.get("LILLYPAD_PERF"))


class _PerfProbe:
    """Collects per-path frame times and session-growth counters.

    Frame times are kept as raw samples and reduced to percentiles at report
    time: a mean would hide exactly what we are looking for, since the symptom
    is occasional long frames (a full redraw escaping the blit path, or a GC
    pause), not a uniformly slower loop.
    """
    def __init__(self, window):
        self._w = window
        self._samples = {}          # name -> list[ms], cleared each report
        tracemalloc.start()
        self._snap = tracemalloc.take_snapshot()

    @contextmanager
    def tick(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self._samples.setdefault(name, []).append(
                (time.perf_counter() - t) * 1000.0)

    def report(self):
        for name, xs in sorted(self._samples.items()):
            xs.sort()
            print(f"[perf] {name:12s} n={len(xs):4d} "
                  f"p50={xs[len(xs) // 2]:7.2f}ms "
                  f"p95={xs[min(int(len(xs) * 0.95), len(xs) - 1)]:7.2f}ms "
                  f"max={xs[-1]:7.2f}ms", file=sys.stderr)
        self._samples.clear()
        snap = tracemalloc.take_snapshot()
        grown = sum(s.size_diff for s in snap.compare_to(self._snap, "filename"))
        self._snap = snap
        # objects/qt_children are the leak detectors: both must plateau. A
        # rising qt_children with a flat objects count is a Qt-side leak (a
        # C++ child never destroyed), which is exactly the menu/worker case.
        print(f"[perf] objects={len(gc.get_objects()):8d} "
              f"qt_children={len(self._w.findChildren(QObject)):6d} "
              f"tracemalloc_delta={grown / 1e6:+.1f}MB", file=sys.stderr)


# Module-level singleton: FrogCanvas and FrogWindow both instrument themselves
# and neither should have to carry a probe reference through its constructor.
_perf = None


def perf_tick(name):
    """Time a block into the probe, or do nothing when the probe is off."""
    return _perf.tick(name) if _perf is not None else nullcontext()


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
    # A device that has started refusing to acquire. Emitted on the first
    # failure and then at most every FAIL_REPORT_S while it persists, because
    # the alternative — the loop quietly retrying forever — is indistinguishable
    # from "the setting I just changed did nothing": the plot simply stops
    # updating and holds the last good frame.
    acquire_failed = Signal(str)
    FAIL_REPORT_S = 3.0

    def __init__(self, get_spec, min_interval_ms=30, parent=None):
        super().__init__(parent)
        self._get_spec = get_spec
        self._min = float(min_interval_ms)
        self._last_fail_report = 0.0
        self._run = True
        self._paused = True
        self._idle = threading.Event()   # set whenever not mid-acquire
        self._idle.set()
        # Guards `_paused` together with the matching set/clear of `_idle`.
        # The pair has to move as one: pause() decides it is safe to hand the
        # device over by reading `_idle`, and run() decides to acquire by
        # reading `_paused`, so the two decisions must not interleave. Held
        # only for those few statements — never across an acquire or a wait.
        self._lock = threading.Lock()
        # Measured worst-case time for one loop iteration. pause() sizes its
        # timeout from this instead of a flat 2 s, which a long integration
        # would otherwise blow through before the loop could possibly park.
        self._last_cycle_ms = float(min_interval_ms)

    def run(self):
        while self._run:
            # Reading `_paused` and publishing the answer in `_idle` is one
            # atomic step. Clearing `_idle` after an unlocked check of
            # `_paused` left a window where pause() could set `_paused`, see
            # the still-set `_idle` from the previous park, and report the
            # device free — while this thread was already on its way into
            # acquire(). Two threads in one vendor driver is exactly what
            # _device_lock exists to prevent.
            with self._lock:
                parked = self._paused
                if parked:
                    self._idle.set()
                else:
                    self._idle.clear()
            if parked:
                self.msleep(15)
                continue
            spec = self._get_spec()
            if spec is None:
                # No device to hold: safe to report idle without the lock,
                # since the loop goes back to the top (and re-checks `_paused`
                # under it) before it could touch anything.
                self._idle.set()
                self.msleep(30)
                continue
            try:
                t0 = time.monotonic()
                raw = np.asarray(spec.acquire(), float)
                wl  = np.asarray(spec.wavelengths, float)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
            except Exception as e:
                now = time.monotonic()
                if now - self._last_fail_report >= self.FAIL_REPORT_S:
                    self._last_fail_report = now
                    if self._run and not self._paused:
                        self.acquire_failed.emit(str(e))
                self.msleep(50)
                continue
            self._last_fail_report = 0.0     # recovered — report the next one
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
        with self._lock:
            self._paused = False

    def join_timeout_ms(self):
        """How long to allow the loop to reach its next park.

        One acquire plus one pacing sleep, with margin, measured rather than
        assumed: acquire() blocks for the whole exposure, which the UI allows
        up to MAX_UI_EXPOSURE_MS, so a flat 2 s would expire while the device
        was still perfectly healthy.

        The measured cycle only exists once an acquire has COMPLETED, so the
        device's own integration time is taken into account as well — the very
        first frame, and the first frame after the exposure is raised, would
        otherwise be judged against a stale (or default) figure.
        """
        cycle = self._last_cycle_ms
        try:
            spec = self._get_spec()
        except Exception:
            spec = None
        if spec is not None:
            cycle = max(cycle, float(getattr(spec, "integration_ms", 0.0)))
        return max(2000.0, 3.0 * cycle + 500.0)

    def pause(self, wait_ms=None):
        """Stop acquiring; block until the current acquire (if any) returns.

        Returns True once the loop has parked in the idle branch — only then is
        it safe to hand the stage/spectrometer to another thread. False means
        the timeout expired with an acquire STILL in flight; callers must treat
        that as "device not available" rather than ignoring it.

        `_paused` is set under the lock, so from here on run() cannot clear
        `_idle` — it only ever does that after reading `_paused` as False under
        the same lock. A set `_idle` therefore means the device is free and
        will stay free until resume(), which is what makes the True honest.
        """
        with self._lock:
            self._paused = True
        if wait_ms is None:
            wait_ms = self.join_timeout_ms()
        return self._idle.wait(wait_ms / 1000.0)

    def stop(self):
        self._run = False
        with self._lock:
            self._paused = True


# Shown whenever a device operation has to be refused because the live feed did
# not release the hardware in time.
FEED_BUSY_MSG = ("Live feed is still mid-acquisition — try again in a moment "
                 "(or stop the feed first).")

# Workers that would not stop before the window closed. Parked here so their
# QThread objects outlive the window: destroying one while its run() is still
# inside a vendor call is a hard crash, not a warning, and at this point the
# process is on its way out anyway. Deliberately never emptied.
_ORPHANED_THREADS = []


# ─────────────────────────────────────────────────────────────────────────────
# Exposure-box plumbing
# ─────────────────────────────────────────────────────────────────────────────
# Longest exposure the integration box will offer, whatever the device claims.
# A GUI policy, not a hardware limit: acquire() blocks for the whole exposure,
# so a device that will happily take its documented 600 s would leave the app
# looking hung, with the feed unable to park and every dialog answering
# FEED_BUSY_MSG. Type-in of a longer value is not something the app needs.
MAX_UI_EXPOSURE_MS = 10_000.0


def _exposure_ms(spec) -> float:
    """What the integration box should SHOW for `spec`.

    The exposure, never the frame time: the box writes set_integration_time(),
    so seeding it from a frame cost that folds in on-board averaging would
    make every later edit multiply the exposure by the average count.
    """
    return float(getattr(spec, "exposure_ms",
                         getattr(spec, "integration_ms", 10.0)))


def _exposure_bounds(spec) -> tuple[float, float]:
    """(min, max) exposure in ms the integration box may offer for `spec`."""
    lo = float(getattr(spec, "min_integration_ms", 1.0))
    hi = float(getattr(spec, "max_integration_ms", MAX_UI_EXPOSURE_MS))
    return lo, min(hi, MAX_UI_EXPOSURE_MS)


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
TRACE_COLORMAPS = ["jet", "magma", "viridis", "inferno", "plasma", "cividis"]


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
# Alignment worker — the four-point symmetry check
# ─────────────────────────────────────────────────────────────────────────────
class AlignmentWorker(QThread):
    """Measures one spectrum at each of -2x, -x, +x, +2x fs and comes back.

    A thread rather than the blocking _stage_action path: four moves plus
    n_average frames each takes seconds, and the window must stay alive.

    The offsets are RELATIVE to wherever the stage is standing when the worker
    starts — the operator parks at the delay whose symmetry they want to judge
    (usually zero) and presses the button, so no marked zero is required.

    The four points are visited in increasing order, exactly like a scan, so
    every one of them is approached from the same side and the mechanical
    backlash cancels out of the +/- comparison. Whatever happens, the stage is
    put back where it started.

    Frames are returned RAW: the window applies the same dark/calibration path
    the live spectrum uses, so the differences sit on the displayed baseline
    instead of a second, drifting derivation of it.

    Signals:
      progress(done, total)
      done([(raw, member_frames_or_None)] x 4)   # in offset order
      error(message)
    """
    progress = Signal(int, int)
    done     = Signal(object)
    error    = Signal(str)

    def __init__(self, stage, spectrometer, config, step_fs, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.spec  = spectrometer
        self.cfg   = config
        self.offsets_fs = (-2.0 * step_fs, -step_fs, step_fs, 2.0 * step_fs)

    def _measure(self):
        """One averaged raw frame here. Mirrors FrogScanWorker._measure minus
        the saturation bookkeeping and the calibration."""
        cfg = self.cfg
        if cfg.wait_after_move_s:
            time.sleep(cfg.wait_after_move_s)
        for _ in range(max(0, cfg.idle_shots)):
            self.spec.acquire()                    # discard settling frames
        n     = max(1, cfg.n_average)
        acc   = None
        m_acc = None
        for _ in range(n):
            s = np.asarray(self.spec.acquire(), float)
            acc = s if acc is None else acc + s
            frames = getattr(self.spec, "last_member_raw", None)
            if frames is not None:
                if m_acc is None:
                    m_acc = [np.asarray(f, float).copy() for f in frames]
                else:
                    for a, f in zip(m_acc, frames):
                        a += f
        members = None if m_acc is None else tuple(a / n for a in m_acc)
        return acc / n, members

    def run(self):
        start_um = None
        out, err = [], None
        try:
            start_um = _stage_to_um(self.stage.get_position())
            pf = self.cfg.pass_factor
            for k, off in enumerate(self.offsets_fs):
                # zero = 0.0 makes this a RELATIVE conversion, as _jog does.
                d_um = float(delay_to_position_um(off, 0.0, pf))
                self.stage.move_to(_um_to_stage(start_um + d_um))
                out.append(self._measure())
                self.progress.emit(k + 1, len(self.offsets_fs))
        except Exception as e:
            err = str(e)
        # Always come home, including after a failure part-way through: leaving
        # the stage parked at +2x would silently shift every later jog, move
        # and scan the operator makes from here.
        if start_um is not None:
            try:
                self.stage.move_to(_um_to_stage(start_um))
            except Exception as e:
                if err is None:
                    err = f"could not return to the start position: {e}"
        # Reported only once the stage is back: the slot restarts the live feed
        # and reads the position, and neither may happen while this thread is
        # still driving the hardware.
        if err is None:
            self.done.emit(out)
        else:
            self.error.emit(err)


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
        self.spin_avg = SpinBox(); self.spin_avg.setRange(1, 1000); self.spin_avg.setValue(1)
        grid.addWidget(self.spin_avg, 0, 1)
        grid.addWidget(QLabel("Idle shots"), 1, 0)
        self.spin_idle = SpinBox(); self.spin_idle.setRange(0, 1000); self.spin_idle.setValue(0)
        grid.addWidget(self.spin_idle, 1, 1)
        grid.addWidget(QLabel("Wait after move"), 2, 0)
        self.spin_wait = SpinBox(); self.spin_wait.setRange(0, 10000)
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
        self.spin_sat = DoubleSpinBox()
        self.spin_sat.setRange(50.0, 100.0); self.spin_sat.setDecimals(0)
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

        lay.addWidget(_hline())

        # ── Stage health ─────────────────────────────────────────────────────
        lay.addWidget(self._hdr("Stage"))
        self.chk_abort_stage_fault = QCheckBox(
            "Abort scan on stage fault")
        self.chk_abort_stage_fault.setChecked(FrogScanConfig.abort_on_stage_fault)
        lay.addWidget(self.chk_abort_stage_fault)

        stage_hint = QLabel(
            "A stall, a knob nudge or an unhomed axis means the stage is not "
            "where the program thinks it is, so every later column carries a "
            "delay that never happened. Checked (the default), the scan stops "
            "at the first one. Unchecked, it runs to the end and the affected "
            "columns are marked in the .npz.")
        stage_hint.setObjectName("dim"); stage_hint.setWordWrap(True)
        lay.addWidget(stage_hint)

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


class AlignmentDialog(QDialog):
    """Settings for alignment mode — the Δ button over the spectrum panel.

    Its own window rather than a side-panel row: the step is dialled in once
    when the geometry changes and then left alone, so it was costing the panel
    a permanent block for a control nobody touches during a measurement.
    """
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Alignment")
        self.setFixedWidth(300)
        lay = QVBoxLayout(self); lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        lay.addWidget(self._hdr("Sweep"))
        row = QGridLayout(); row.setSpacing(8)
        row.addWidget(QLabel("Alignment step"), 0, 0)
        # Half-width of the four-point sweep. Its own box rather than the stage
        # jog step: that one switches to um, and an alignment offset is only
        # ever a delay.
        self.spin_align_step = DoubleSpinBox()
        self.spin_align_step.setDecimals(0)
        self.spin_align_step.setRange(1.0, 100000.0)
        self.spin_align_step.setValue(100.0)
        self.spin_align_step.setSuffix(" fs")
        self.spin_align_step.setToolTip(
            "Half-width x of the alignment sweep: spectra are taken at "
            "−2x, −x, +x and +2x from the current position.")
        row.addWidget(self.spin_align_step, 0, 1)
        lay.addLayout(row)

        hint = QLabel(
            "Press Δ above the spectrum to run the sweep. It measures at −2x, "
            "−x, +x and +2x from where the stage is now and overlays "
            "S(+x)−S(−x) and S(+2x)−S(−2x): a symmetric pulse gives two flat "
            "curves on zero. The stage is left where it started.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

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


# ─────────────────────────────────────────────────────────────────────────────
# Connection dialogs
# ─────────────────────────────────────────────────────────────────────────────
class _ConnectDialog(QDialog):
    """Shared plumbing for the three connection windows.

    They all do the same three things: run an (ok, msg) action and colour the
    result inline, draw a small section header, and refresh themselves whenever
    they are shown. Only `_refresh` differs, so only `_refresh` is overridden.
    """
    def __init__(self, main, parent=None, title="", width=380):
        super().__init__(parent, Qt.Tool)
        self.main = main
        self.setWindowTitle(title)
        self.setFixedWidth(width)
        # Created here rather than in each subclass: _do() writes to it, and
        # every subclass has to place it somewhere in its own layout.
        self.lbl_msg = QLabel("")
        self.lbl_msg.setObjectName("dim")
        self.lbl_msg.setWordWrap(True)

    @staticmethod
    def _header(text):
        l = QLabel(text)
        l.setObjectName("hdr")
        return l

    def _do(self, fn):
        """Run an (ok, msg) action and report it in lbl_msg."""
        ok, msg = fn()
        # Color via objectName + repolish (not setStyleSheet) so a later theme
        # switch restyles the label along with everything else.
        self.lbl_msg.setObjectName("ok" if ok else "err")
        self.lbl_msg.style().unpolish(self.lbl_msg)
        self.lbl_msg.style().polish(self.lbl_msg)
        self.lbl_msg.setText(msg)
        self._refresh()
        self._refit()

    def _refit(self):
        """Grow the window to whatever the contents now need.

        These dialogs are full of word-wrapped labels — a device name, a status
        line, a stitch summary — that go from one line to four as devices come
        and go, and Qt will not re-grow an already-shown window for a wrapped
        label on its own. The width is fixed, so this only ever changes height.
        """
        self.layout().activate()
        self.resize(self.width(), self.sizeHint().height())

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.open_fresh()

    def open_fresh(self):
        """Show the dialog with a clean status line and current contents.

        This — not toggle() — is what the panel Connect buttons call: pressing
        "Connect Spectrometer" must always open the window, never close one
        that happens to be up already.
        """
        self.lbl_msg.setText("")
        self._refresh()
        self._refit()
        self.show()
        self.raise_()
        self.activateWindow()

    def _refresh(self):
        pass


class SpectrometerDialog(_ConnectDialog):
    """Connect one or two spectrometers, and manage the stitch between them.

    Two slots are offered from the start, because a pair is not a separate mode
    to be enabled — it is just the second slot being filled. Slot 1 alone is
    single-spectrometer mode; filling slot 2 reopens both as one stitched
    device; emptying it drops back to slot 1. The old Multi-Spec menu's entire
    contents (per-slot device and calibration, auto/manual stitch, overlap
    band) live here, next to the connection they describe.

    Simulated devices are deliberately absent: they belong to the Simulation
    window, so nothing on a bench-hardware dialog can quietly hand back fake
    data.
    """
    NONE_LABEL = "(none)"

    def __init__(self, main, parent=None):
        super().__init__(main, parent, "Spectrometer", width=430)
        lay = QVBoxLayout(self); lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        hrow = QHBoxLayout(); hrow.setSpacing(6)
        hrow.addWidget(self._header("Spectrometer"), 1)
        self.btn_scan = QPushButton("Rescan")
        self.btn_scan.setObjectName("compact")
        self.btn_scan.setToolTip("Re-enumerate every attached spectrometer")
        self.btn_scan.clicked.connect(self._rescan)
        hrow.addWidget(self.btn_scan, 0)
        lay.addLayout(hrow)

        self.lbl_spec = QLabel(); self.lbl_spec.setObjectName("value")
        self.lbl_spec.setWordWrap(True)
        lay.addWidget(self.lbl_spec)

        # Slot rows. Device on the left, that slot's calibration on the right:
        # a calibration file belongs to ONE physical spectrometer, so putting
        # it on any other row would invite assigning it to the pair.
        grid = QGridLayout(); grid.setSpacing(6)
        grid.setColumnStretch(1, 3); grid.setColumnStretch(2, 2)
        self.cmb_dev = []
        self.cmb_cal = []
        for slot in (0, 1):
            grid.addWidget(QLabel(f"Slot {slot + 1}"), slot, 0)
            dev = ComboBox()
            dev.setToolTip("Which spectrometer this slot holds")
            dev.activated.connect(
                lambda _i, s=slot: self._on_device_picked(s))
            grid.addWidget(dev, slot, 1)
            self.cmb_dev.append(dev)
            cal = ComboBox()
            cal.setToolTip("Intensity calibration applied to this slot's "
                           "device — raw counts when none")
            cal.activated.connect(
                lambda _i, s=slot: self._on_cal_picked(s))
            grid.addWidget(cal, slot, 2)
            self.cmb_cal.append(cal)
        lay.addLayout(grid)
        hint = QLabel("Leave slot 2 empty for single-spectrometer mode. "
                      "Filling it reopens both devices as one stitched "
                      "spectrometer.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Stitching ────────────────────────────────────────────────────────
        # Only meaningful while a pair is live, so the whole block greys out
        # rather than disappearing: it is where you look to find out WHY the
        # seam is wrong, and a block that vanishes gives nowhere to look.
        lay.addWidget(self._header("Stitching"))
        self.lbl_stitch = QLabel("—")
        self.lbl_stitch.setObjectName("dim"); self.lbl_stitch.setWordWrap(True)
        lay.addWidget(self.lbl_stitch)
        srow = QHBoxLayout(); srow.setSpacing(6)
        self.btn_fit = QPushButton("Auto-stitch")
        self.btn_fit.setObjectName("accentcompact")
        self.btn_fit.setToolTip("Fit the stitch factor from one frame — needs "
                                "light across the overlap region")
        self.btn_fit.clicked.connect(
            lambda: self._do(self.main._fit_stitch_factor))
        self.btn_manual = QPushButton("Manual…")
        self.btn_manual.setObjectName("compact")
        self.btn_manual.clicked.connect(self._manual_stitch)
        self.btn_band = QPushButton("Overlap band…")
        self.btn_band.setObjectName("compact")
        self.btn_band.setToolTip(
            "The part of the overlap the stitch factor is fitted over and the "
            "two spectra are crossfaded across — shown shaded green in the "
            "per-spectrometer view")
        self.btn_band.clicked.connect(self._set_band)
        srow.addWidget(self.btn_fit); srow.addWidget(self.btn_manual)
        srow.addWidget(self.btn_band)
        lay.addLayout(srow)

        lay.addWidget(_hline())

        # seabreeze backend switch. pyseabreeze is the default because it is
        # the only backend that knows the newer Ocean Insight models;
        # cseabreeze stays available for devices that only enumerate through
        # the vendor C library. Applied on the next connect.
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(1, 1)
        brow.addWidget(QLabel("Backend"), 0, 0)
        self.cmb_backend = ComboBox()
        for name in SEABREEZE_BACKENDS:
            self.cmb_backend.addItem(name, name)
        self.cmb_backend.setCurrentIndex(
            max(0, self.cmb_backend.findData(self.main.seabreeze_backend)))
        self.cmb_backend.currentIndexChanged.connect(self._on_backend)
        brow.addWidget(self.cmb_backend, 0, 1)
        # Full-scale override. Without this, a spectrometer that does not
        # report `max_intensity` leaves saturation unchecked with no way out
        # from the UI. Blank = trust the device.
        brow.addWidget(QLabel("Full scale"), 1, 0)
        self.edit_full_scale = QLineEdit()
        self.edit_full_scale.setPlaceholderText("auto (from device)")
        self.edit_full_scale.editingFinished.connect(self._on_full_scale)
        brow.addWidget(self.edit_full_scale, 1, 1)
        lay.addLayout(brow)

        lay.addWidget(self.lbl_msg)
        frow = QHBoxLayout(); frow.setSpacing(6)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setObjectName("danger")
        self.btn_disconnect.clicked.connect(self._disconnect)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        frow.addWidget(self.btn_disconnect); frow.addWidget(btn_close)
        lay.addLayout(frow)

        self._devices = []      # last enumeration: [(label, ident), ...]

    # ── Enumeration ──────────────────────────────────────────────────────────
    def open_fresh(self):
        """Opening the window IS the request to scan — that is what the panel's
        Connect button means. Scanned AFTER the base clears the status line, so
        the result of this scan is what stays on it."""
        super().open_fresh()
        self._rescan()

    def _rescan(self):
        self._do(self._scan)

    def _scan(self):
        """Re-enumerate every vendor. (ok, msg), for _do.

        list_spectrometers already merges seabreeze and Avantes into one
        tagged-id list, so one call covers the bench.
        """
        try:
            self._devices = list(self.main._list_spectrometers())
        except Exception as e:
            self._devices = []
            return False, f"Scan failed: {e}"
        n = len(self._devices)
        if n:
            return True, (f"{n} spectrometer{'' if n == 1 else 's'} found — "
                          f"pick one for slot 1.")
        # Nothing found. list_spectrometers hides a vendor whose SDK failed, so
        # say WHICH one failed and why — otherwise a missing AvaSpec driver and
        # an unplugged cable look identical from here.
        msg = ("No spectrometers found. Check the USB cables and the vendor "
               "drivers, then press Rescan.")
        notes = self.main._spectrometer_scan_notes()
        if notes:
            msg += "\n\n" + "\n".join(notes)
        return False, msg

    # ── Slot handlers ────────────────────────────────────────────────────────
    def _on_device_picked(self, slot):
        ident = self.cmb_dev[slot].currentData()
        label = self.cmb_dev[slot].currentText()
        if ident == self.main._multi["serials"][slot]:
            return                       # re-picked what is already there
        if ident is None:
            self._do(lambda: self.main._clear_slot(slot))
        else:
            self._do(lambda: self.main._set_slot(slot, ident, label))

    def _on_cal_picked(self, slot):
        path = self.cmb_cal[slot].currentData()
        self.main._select_slot_calibration(slot, path)
        self._refresh()

    def _disconnect(self):
        self._do(self.main._disconnect_spectrometer)

    # ── Stitching handlers ───────────────────────────────────────────────────
    def _manual_stitch(self):
        self.main._set_stitch_factor()
        self._refresh()

    def _set_band(self):
        self.main._set_overlap_band()
        self._refresh()

    # ── Device-wide settings ─────────────────────────────────────────────────
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

    def _on_backend(self):
        backend = self.cmb_backend.currentData()
        if backend == self.main.seabreeze_backend:
            return
        # A device opened through the other backend cannot survive the switch
        # (selecting a backend shuts the previous backend's API down), so
        # release it now rather than severing it mid-use on the next connect.
        live = self.main._live_seabreeze_backend()
        if live is not None and live != backend:
            ok, err = self.main._disconnect_spectrometer()
            if not ok:
                # Refused (scan running / feed busy) — keep the old choice.
                self.cmb_backend.setCurrentIndex(
                    max(0, self.cmb_backend.findData(
                        self.main.seabreeze_backend)))
                self._do(lambda: (False, err))
                return
            self.main.seabreeze_backend = backend
            self._scan()      # the new backend enumerates its own devices
            self._do(lambda: (True, f"Backend: {backend}. The spectrometer "
                                    f"was released — pick it again in slot 1 "
                                    f"to reconnect through it."))
        else:
            self.main.seabreeze_backend = backend
            self._scan()
            self._do(lambda: (True, f"Backend: {backend} — used on the next "
                                    f"connect."))

    # ── Display ──────────────────────────────────────────────────────────────
    def _refresh(self):
        spec = self.main.spec
        self.lbl_spec.setText(spec.name if spec is not None
                              else "Not connected.")
        self.btn_disconnect.setEnabled(spec is not None)
        for slot in (0, 1):
            self._fill_device_combo(slot)
            self._fill_cal_combo(slot)
        reported = getattr(spec, "max_counts", None)
        self.edit_full_scale.setPlaceholderText(
            f"auto — device reports {reported:.0f}" if reported
            else "device reports none — saturation unchecked")
        self._refresh_stitch()

    def _fill_device_combo(self, slot):
        """Rebuild one slot's device list: (none), whatever the slot currently
        holds, and every enumerated device not claimed by the other slot.

        The current choice is re-added even when the last scan missed it — a
        simulated half assigned from the Simulation window never appears in an
        enumeration, and dropping it here would make the combo claim the slot
        was empty while the device was live.
        """
        cmb = self.cmb_dev[slot]
        ident = self.main._multi["serials"][slot]
        label = self.main._multi["labels"][slot]
        other = self.main._multi["serials"][1 - slot]
        with QSignalBlocker(cmb):
            cmb.clear()
            cmb.addItem(self.NONE_LABEL, None)
            known = set()
            if ident is not None:
                cmb.addItem(label or str(ident), ident)
                known.add(ident)
            for model, dev_id in self._devices:
                if dev_id == other or dev_id in known:
                    continue
                # Bare serial in the label: the model name already says which
                # vendor it is, so repeating the tag would only make the combo
                # wider. The full tagged id is what gets stored in the slot.
                cmb.addItem(f"{model} [{dev_id.split(':', 1)[-1]}]", dev_id)
            cmb.setCurrentIndex(max(0, cmb.findData(ident)))

    def _fill_cal_combo(self, slot):
        cmb = self.cmb_cal[slot]
        cal = self.main._multi["cals"][slot]
        with QSignalBlocker(cmb):
            cmb.clear()
            cmb.addItem("No calibration", None)
            current = 0
            for i, f in enumerate(self.main._calibration_files(), start=1):
                cmb.addItem(f.stem, f)
                if cal is not None and f.stem == cal.stem:
                    current = i
            cmb.setCurrentIndex(current)

    def _refresh_stitch(self):
        """The stitch factor, its fit quality and whether it has gone stale.

        The residual is the honest answer to "is one scalar enough for this
        pair?", so it belongs next to the factor rather than in a status
        message that has already scrolled away.
        """
        spec = self.main.spec
        live = isinstance(spec, StitchedSpectrometer)
        for b in (self.btn_fit, self.btn_manual, self.btn_band):
            b.setEnabled(live)
        if not live:
            self.lbl_stitch.setText(
                "Fill both slots to stitch two spectrometers into one.")
            return
        res = spec.stitch_residual
        quality = ("not fitted yet" if res is None
                   else f"mismatch {res * 100:.1f}%")
        stale = ("  ·  STALE — integration times changed"
                 if self.main._stitch_stale else "")
        lo, hi = spec.overlap_band
        glo, ghi = spec.geometric_overlap
        self.lbl_stitch.setText(
            f"Factor {spec.stitch_factor:.4g}  ({quality}){stale}\n"
            f"Overlap band {lo:.1f}–{hi:.1f} nm of {glo:.1f}–{ghi:.1f} nm.")


class StageDialog(_ConnectDialog):
    """Connect a delay stage, and hold the one setting that belongs to it.

    Kinesis controllers enumerate, so they get a scanned list. Zaber and Piezo
    Jena do not — their adapters auto-scan the serial ports themselves on
    connect — so they keep a Connect button each, with an optional port
    override for a bench where the auto-scan picks the wrong one.
    """
    def __init__(self, main, parent=None):
        super().__init__(main, parent, "Stage", width=400)
        lay = QVBoxLayout(self); lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        hrow = QHBoxLayout(); hrow.setSpacing(6)
        hrow.addWidget(self._header("Stage"), 1)
        self.btn_scan = QPushButton("Rescan")
        self.btn_scan.setObjectName("compact")
        self.btn_scan.setToolTip("Re-enumerate the attached Kinesis controllers")
        self.btn_scan.clicked.connect(self._rescan)
        hrow.addWidget(self.btn_scan, 0)
        lay.addLayout(hrow)

        self.lbl_stage = QLabel(); self.lbl_stage.setObjectName("value")
        self.lbl_stage.setWordWrap(True)
        lay.addWidget(self.lbl_stage)

        # Two equal columns: the parameter of a row on the left, that row's
        # connect button on the right. Putting each port field on its OWN
        # vendor's row is what stops it reading as a global setting.
        grid = QGridLayout(); grid.setSpacing(6)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        self.cmb_kinesis = ComboBox()
        self.cmb_kinesis.setToolTip("Kinesis controllers found by the last scan")
        b_kin = QPushButton("Connect Kinesis"); b_kin.setObjectName("accent")
        b_kin.clicked.connect(self._connect_kinesis)
        grid.addWidget(self.cmb_kinesis, 0, 0); grid.addWidget(b_kin, 0, 1)

        # Zaber: optional serial port (blank = auto-scan the machine's ports).
        self.edit_zaber_port = QLineEdit()
        self.edit_zaber_port.setPlaceholderText("COM (auto)")
        b_zab = QPushButton("Connect Zaber"); b_zab.setObjectName("accent")
        b_zab.clicked.connect(lambda: self._do(self._connect_zaber))
        grid.addWidget(self.edit_zaber_port, 1, 0); grid.addWidget(b_zab, 1, 1)

        # Piezo Jena: optional serial port (blank = auto-scan), like Zaber.
        self.edit_piezo_port = QLineEdit()
        self.edit_piezo_port.setPlaceholderText("COM (auto)")
        b_pj = QPushButton("Connect Piezo Jena"); b_pj.setObjectName("accent")
        b_pj.clicked.connect(lambda: self._do(self._connect_piezo))
        grid.addWidget(self.edit_piezo_port, 2, 0); grid.addWidget(b_pj, 2, 1)
        lay.addLayout(grid)

        lay.addWidget(_hline())

        # Backlash approach margin. Every move undershoots by this much when it
        # would otherwise arrive from above, so the zero you mark by jogging and
        # the positions a scan sweeps through sit in the same frame. It lives
        # here because it is a property OF THE CONNECTED STAGE — the connect
        # buttons above seed it, and _sync_backlash_ui pushes the new stage's
        # default into this box on every swap.
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(1, 1)
        brow.addWidget(QLabel("Backlash"), 0, 0)
        self.spin_backlash = DoubleSpinBox()
        self.spin_backlash.setRange(0.0, 1000.0); self.spin_backlash.setDecimals(1)
        self.spin_backlash.setSingleStep(10.0); self.spin_backlash.setSuffix(" um")
        self.spin_backlash.setToolTip(
            "Approach margin. Lead-screw stages land in a different place "
            "depending on which way they arrived; undershooting by more than "
            "the slack and coming back up makes every move repeatable.\n\n"
            "0 disables it — correct for a Thorlabs controller (its firmware "
            "already does this) or a piezo, wrong for a Zaber.")
        self.spin_backlash.valueChanged.connect(self.main._on_backlash_changed)
        brow.addWidget(self.spin_backlash, 0, 1)
        lay.addLayout(brow)
        self.lbl_backlash_fs = QLabel("—"); self.lbl_backlash_fs.setObjectName("dim")
        self.lbl_backlash_fs.setWordWrap(True)
        lay.addWidget(self.lbl_backlash_fs)

        lay.addWidget(self.lbl_msg)
        frow = QHBoxLayout(); frow.setSpacing(6)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setObjectName("danger")
        self.btn_disconnect.clicked.connect(
            lambda: self._do(self.main._disconnect_stage))
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        frow.addWidget(self.btn_disconnect); frow.addWidget(btn_close)
        lay.addLayout(frow)

        self._devices = []      # last Kinesis enumeration: [(model, conn), ...]

    def open_fresh(self):
        """Scanned AFTER the base clears the status line, so what the scan
        found is what the window opens showing."""
        super().open_fresh()
        self._rescan()

    def _rescan(self):
        self._do(self._scan)

    def _scan(self):
        """Enumerate the Kinesis controllers. (ok, msg), for _do.

        Each controller is briefly opened to read its model number, so this is
        a real device query, not a list of ports — hence a button rather than
        something done continuously.
        """
        try:
            self._devices = list(list_kinesis_stages())
        except Exception as e:
            self._devices = []
            self._fill_kinesis_combo()
            return False, f"Kinesis scan failed: {e}"
        self._fill_kinesis_combo()
        n = len(self._devices)
        return bool(n), (
            f"{n} Kinesis controller{'' if n == 1 else 's'} found." if n else
            "No Kinesis controllers found. Zaber and Piezo Jena stages cannot "
            "be enumerated — use their own Connect buttons, which scan the "
            "serial ports themselves.")

    def _fill_kinesis_combo(self):
        with QSignalBlocker(self.cmb_kinesis):
            self.cmb_kinesis.clear()
            for model, conn in self._devices:
                self.cmb_kinesis.addItem(f"{model} [{conn}]", conn)
            if not self._devices:
                self.cmb_kinesis.addItem("(none found)", None)
        self.cmb_kinesis.setEnabled(bool(self._devices))

    def _connect_kinesis(self):
        conn = self.cmb_kinesis.currentData()
        if conn is None:
            self._do(lambda: (False, "No Kinesis controller selected — press "
                                     "Rescan."))
            return
        self._do(lambda: self.main._connect_real_stage(conn))

    def _connect_zaber(self):
        port = self.edit_zaber_port.text().strip() or None
        return self.main._connect_zaber_stage(port)

    def _connect_piezo(self):
        port = self.edit_piezo_port.text().strip() or None
        return self.main._connect_piezo_jena_stage(port)

    def _refresh(self):
        stage = self.main.stage
        self.lbl_stage.setText(stage.name if stage is not None
                               else "Not connected.")
        self.btn_disconnect.setEnabled(stage is not None)
        self.spin_backlash.setEnabled(stage is not None)


class SimulationDialog(_ConnectDialog):
    """Run the app against simulated hardware, on purpose.

    Simulation used to be the startup default, which meant a fake spectrum
    could be mistaken for a measurement. It is now only ever reached from here,
    and the choices are the same ones the Hardware dialog used to make
    silently: which beam the simulator is measuring, and whether to stand in
    for the spectrometer, the stage, or a stitched pair.
    """
    def __init__(self, main, parent=None):
        super().__init__(main, parent, "Simulation", width=380)
        lay = QVBoxLayout(self); lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        # ── Simulated beam ───────────────────────────────────────────────────
        # Which pulse the simulated spectrometer is measuring. Changing either
        # box re-makes the simulator on the spot (its wavelength window is
        # derived from the beam, so it cannot just be mutated in place).
        lay.addWidget(self._header("Simulated beam"))
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(0, 1); brow.setColumnStretch(1, 0)
        self.cmb_pulse = ComboBox()
        for key, shape in PULSE_SHAPES.items():
            self.cmb_pulse.addItem(shape["label"], key)
        self.cmb_pulse.setCurrentIndex(
            max(0, self.cmb_pulse.findData(self.main.sim_pulse)))
        self.cmb_pulse.currentIndexChanged.connect(self._on_beam)
        self.cmb_gate = ComboBox()
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

        lay.addWidget(_hline())

        lay.addWidget(self._header("Stand in for"))
        grid = QGridLayout(); grid.setSpacing(6)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        b_spec = QPushButton("Spectrometer"); b_spec.setObjectName("accent")
        b_spec.setToolTip("One simulated spectrometer covering the whole "
                          "simulated signal band")
        b_spec.clicked.connect(lambda: self._do(self.main._use_sim_spectrometer))
        b_stage = QPushButton("Stage"); b_stage.setObjectName("accent")
        b_stage.setToolTip("A 300 mm simulated delay stage")
        b_stage.clicked.connect(lambda: self._do(self.main._use_sim_stage))
        grid.addWidget(b_spec, 0, 0); grid.addWidget(b_stage, 0, 1)
        b_pair = QPushButton("Stitched pair"); b_pair.setObjectName("accent")
        b_pair.setToolTip(
            "Two simulated spectrometers covering overlapping halves of the "
            "band, connected as one stitched device — the only way to "
            "exercise multi-spectrometer mode without two on the bench")
        b_pair.clicked.connect(lambda: self._do(self.main._use_sim_pair))
        grid.addWidget(b_pair, 1, 0, 1, 2)
        lay.addLayout(grid)

        lay.addWidget(self.lbl_msg)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        lay.addWidget(btn_close)

    def _on_beam(self):
        self.main.sim_pulse = self.cmb_pulse.currentData()
        self.main.sim_gate  = self.cmb_gate.currentData()
        spec = self.main.spec
        if isinstance(spec, SimulatedSpectrometer):
            self._do(self.main._use_sim_spectrometer)
        elif (isinstance(spec, StitchedSpectrometer)
              and all(isinstance(m, SimulatedSpectrometer)
                      for m in spec.members)):
            self._do(self.main._use_sim_pair)
        else:
            # Real hardware (or nothing) is connected — remember the choice for
            # the next time a simulator is asked for, rather than swapping the
            # device out from under the operator.
            self._refresh()
            self.lbl_msg.setText("Beam saved — press Spectrometer to use it.")

    def _refresh(self):
        desc = PULSE_SHAPES[self.main.sim_pulse]["desc"]
        spec = self.main.spec
        if isinstance(spec, SimulatedSpectrometer):
            desc += (f"\nSuggested scan: ±{spec.suggested_delay_fs:.0f} fs   ·   "
                     f"{spec.wavelengths[0]:.0f}–{spec.wavelengths[-1]:.0f} nm")
        self.lbl_beam.setText(desc)


# ─────────────────────────────────────────────────────────────────────────────
# Avantes settings dialog
# ─────────────────────────────────────────────────────────────────────────────
class AvantesSettingsDialog(QDialog):
    """Every Avantes-only control, in its own window.

    The Avantes DLL exposes far more than the Ocean adapter does — on-board
    averaging, ADC resolution, dark and non-linearity correction, smoothing,
    external triggering, sync, prescan, board temperature. None of it belongs
    in the Hardware dialog, which has to stay legible on a bench with no
    Avantes attached, so it lives here behind a toolbar button that is HIDDEN
    unless an Avantes is connected (see FrogWindow._refresh_avantes_button).

    Nothing persists between launches — there is no settings file — so
    `_refresh` reads the device's current state INTO the widgets. Pushing
    widget defaults the other way would mean that merely opening this dialog
    silently reconfigured the spectrometer.
    """

    # Scrubbable controls are debounced through one shared timer: a
    # valueChanged per spinbox tick would park and restart the live feed on
    # every keystroke, the same reason the main integration spinbox debounces.
    WRITE_DEBOUNCE_MS = 250
    POLL_MS = 2000

    def __init__(self, main, parent=None):
        super().__init__(parent, Qt.Tool)
        self.main = main
        self.setWindowTitle("Avantes Settings")

        outer = QVBoxLayout(self); outer.setSpacing(8)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        lay = QVBoxLayout(body); lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        # Which device these controls act on. Hidden with only one Avantes
        # connected, which is the normal case; a two-Avantes stitched pair is
        # the reason it exists at all.
        self.cmb_target = ComboBox()
        self.cmb_target.currentIndexChanged.connect(lambda _=0: self._refresh())
        lay.addWidget(self.cmb_target)

        # ── Acquisition ─────────────────────────────────────────────────────
        lay.addWidget(self._header("Acquisition"))
        arow = QGridLayout(); arow.setSpacing(6)
        arow.setColumnStretch(0, 0); arow.setColumnStretch(1, 1)
        arow.addWidget(QLabel("On-board averages"), 0, 0)
        self.spin_onboard_avg = SpinBox()
        self.spin_onboard_avg.setRange(1, 10000)
        self.spin_onboard_avg.valueChanged.connect(self._queue_write)
        arow.addWidget(self.spin_onboard_avg, 0, 1)
        arow.addWidget(QLabel("Smoothing (± pixels)"), 1, 0)
        self.spin_smooth = SpinBox()
        self.spin_smooth.setRange(0, 128)
        self.spin_smooth.valueChanged.connect(self._queue_write)
        arow.addWidget(self.spin_smooth, 1, 1)
        lay.addLayout(arow)
        self.lbl_avg_hint = QLabel(); self.lbl_avg_hint.setObjectName("dim")
        self.lbl_avg_hint.setWordWrap(True)
        lay.addWidget(self.lbl_avg_hint)

        self.chk_hires = QCheckBox("16-bit ADC (high resolution)")
        self.chk_hires.toggled.connect(self._on_hires)
        lay.addWidget(self.chk_hires)
        self.lbl_hires = QLabel(); self.lbl_hires.setObjectName("dim")
        self.lbl_hires.setWordWrap(True)
        lay.addWidget(self.lbl_hires)

        lay.addWidget(_hline())

        # ── Corrections ─────────────────────────────────────────────────────
        lay.addWidget(self._header("Corrections"))
        self.chk_dark = QCheckBox("Dynamic dark correction")
        self.chk_dark.toggled.connect(
            lambda on: self._write("Dark correction",
                                   lambda d: d.set_dark_correction(on),
                                   voids_dark=True))
        lay.addWidget(self.chk_dark)
        self.chk_prescan = QCheckBox("Prescan (discard the first scan)")
        self.chk_prescan.toggled.connect(
            lambda on: self._write("Prescan", lambda d: d.set_prescan(on),
                                   voids_dark=True))
        lay.addWidget(self.chk_prescan)
        hint = QLabel("These change what a frame contains: a recorded dark is "
                      "discarded and any fitted stitch factor goes stale — "
                      "re-record and re-fit after changing them.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Trigger & sync ──────────────────────────────────────────────────
        lay.addWidget(self._header("Trigger & sync"))
        trow = QGridLayout(); trow.setSpacing(6)
        trow.setColumnStretch(0, 0); trow.setColumnStretch(1, 1)
        self.cmb_trig_mode = ComboBox()
        self.cmb_trig_src = ComboBox()
        self.cmb_trig_type = ComboBox()
        for r, (label, cmb) in enumerate((("Mode", self.cmb_trig_mode),
                                          ("Source", self.cmb_trig_src),
                                          ("Edge/level", self.cmb_trig_type))):
            trow.addWidget(QLabel(label), r, 0)
            trow.addWidget(cmb, r, 1)
            cmb.currentIndexChanged.connect(self._on_trigger)
        lay.addLayout(trow)
        self.chk_sync = QCheckBox("Sync mode")
        self.chk_sync.toggled.connect(
            lambda on: self._write("Sync mode",
                                   lambda d: d.set_sync_mode(on)))
        lay.addWidget(self.chk_sync)
        self.lbl_trig = QLabel(); self.lbl_trig.setObjectName("dim")
        self.lbl_trig.setWordWrap(True)
        lay.addWidget(self.lbl_trig)

        lay.addWidget(_hline())

        # ── Detector ────────────────────────────────────────────────────────
        lay.addWidget(self._header("Detector"))
        self.lbl_temp = QLabel("—"); self.lbl_temp.setObjectName("value")
        lay.addWidget(self.lbl_temp)
        trow2 = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_readback)
        self.chk_poll = QCheckBox(f"Auto-refresh ({self.POLL_MS // 1000} s)")
        self.chk_poll.toggled.connect(self._on_poll_toggle)
        trow2.addWidget(btn_refresh); trow2.addWidget(self.chk_poll)
        lay.addLayout(trow2)
        hint = QLabel("Each refresh borrows the spectrometer from the live "
                      "feed for one frame — leave auto-refresh off at long "
                      "integration times.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Device info ─────────────────────────────────────────────────────
        lay.addWidget(self._header("Device"))
        self.lbl_info = QLabel(); self.lbl_info.setObjectName("dim")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)

        self.lbl_msg = QLabel(""); self.lbl_msg.setObjectName("dim")
        self.lbl_msg.setWordWrap(True)
        lay.addWidget(self.lbl_msg)
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(body); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)
        close_row = QHBoxLayout(); close_row.setContentsMargins(14, 0, 14, 14)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        close_row.addWidget(btn_close)
        outer.addLayout(close_row)
        self.setFixedWidth(360)

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.WRITE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._flush_writes)
        self._loading = False        # True while _refresh seeds the widgets
        self._target_ids = None      # device list the target combo was built from

    # ── plumbing ────────────────────────────────────────────────────────────
    def _header(self, text):
        l = QLabel(text); l.setObjectName("hdr")
        return l

    def _target(self):
        """The device the controls act on, re-resolved every time.

        Never cached: _apply_spectrometer can disconnect it out from under an
        open dialog, and a stale handle would be a call into a closed device.
        The combo carries the tagged id rather than the device object for the
        same reason — widget data must not keep a closed device alive.
        """
        devices = self.main._avantes_devices()
        if not devices:
            return None
        want = self.cmb_target.currentData()
        for d in devices:
            if getattr(d, "spec_id", None) == want:
                return d
        return devices[0]

    def _sync_target_combo(self, devices):
        """Rebuild the target list only when the devices actually changed.

        An unconditional clear()/addItem() would reset the selection to the
        first device on every _refresh — and _refresh runs after every write,
        so a setting applied while member 2 was selected would land on member 2
        and then silently snap the UI back to member 1, with the next write
        going to the wrong device.
        """
        ids = [getattr(d, "spec_id", None) or d.name for d in devices]
        if ids == self._target_ids:
            return
        keep = self.cmb_target.currentData()
        self.cmb_target.blockSignals(True)
        self.cmb_target.clear()
        for d, ident in zip(devices, ids):
            self.cmb_target.addItem(d.name, ident)
        if keep is not None:
            self.cmb_target.setCurrentIndex(max(0, self.cmb_target.findData(keep)))
        self.cmb_target.blockSignals(False)
        # Only meaningful with a two-Avantes pair; hidden in the normal case.
        self.cmb_target.setVisible(len(devices) > 1)
        self._target_ids = ids

    def _do(self, ok, msg):
        # objectName + repolish rather than setStyleSheet, so a theme switch
        # restyles this label with everything else.
        self.lbl_msg.setObjectName("ok" if ok else "err")
        self.lbl_msg.style().unpolish(self.lbl_msg)
        self.lbl_msg.style().polish(self.lbl_msg)
        self.lbl_msg.setText(msg)

    def _write(self, label, fn, refresh=True, voids_dark=False):
        """One device write: refused during a scan, taken under the feed
        handover, reported inline.

        The _scan_running() check is not belt-and-braces — _device_lock only
        parks the LIVE FEED, while the scan worker holds the devices
        independently, which is why _apply_spectrometer and _apply_calibration
        both check separately too.

        `voids_dark` marks a setting that changes what a frame CONTAINS — the
        on-board corrections, the averaging, the ADC range. A dark recorded
        before it no longer describes the frames coming out, so it is dropped
        rather than left to be subtracted from data it does not match.
        """
        if self._loading:
            return                      # _refresh is seeding widgets, not a user edit
        dev = self._target()
        if dev is None:
            self._do(False, "No Avantes spectrometer is connected.")
            return
        if self.main._scan_running():
            self._do(False, "A scan is running — stop it before changing "
                            "spectrometer settings.")
            return
        with self.main._device_lock() as ok:
            if not ok:
                self._do(False, FEED_BUSY_MSG)
                return
            try:
                fn(dev)
            except Exception as e:
                self._do(False, f"{label} failed: {e}")
                return
        self._do(True, f"{label}: applied.")
        if voids_dark:
            self.main._invalidate_dark(f"{label.lower()} changed")
        if refresh:
            self._refresh()

    def _queue_write(self, _value=None):
        """Debounce a scrubbable control."""
        if not self._loading:
            self._debounce.start()

    def _flush_writes(self):
        """Apply the debounced spinboxes in one device handover."""
        avg, smooth = self.spin_onboard_avg.value(), self.spin_smooth.value()

        def apply(dev):
            dev.set_averages(avg)
            dev.set_smoothing(smooth)

        # voids_dark: on-board averaging changes the pedestal a frame carries
        # along with the signal, and smoothing redistributes it across pixels.
        self._write("Acquisition settings", apply, voids_dark=True)
        # On-board averaging multiplies the frame time, and the main
        # integration spinbox shows what one acquire costs — reseed it, or it
        # disagrees with what the feed is actually pacing to.
        self.main._sync_integration_ui()

    # ── control handlers ────────────────────────────────────────────────────
    def _on_hires(self, on):
        """ADC resolution moves full scale (16383 <-> 65535), so the saturation
        alarm has to be re-armed against the new ceiling — the same pairing
        SpectrometerDialog._on_full_scale does. Without it the lamp and the scan's
        clip test keep judging against the old one."""
        if self._loading:
            return
        result = {}

        def apply(dev):
            result["on"] = dev.set_high_res_adc(on)

        self._write("ADC resolution", apply, refresh=False, voids_dark=True)
        self.main._reset_saturation()
        if self.main.dlg_spec.isVisible():
            self.main.dlg_spec._refresh()   # its full-scale placeholder moved
        if result.get("on") is False and on:
            self._do(False, "This device has no 16-bit ADC — staying at 14-bit.")
        self._refresh()

    def _on_trigger(self, _idx=None):
        if self._loading:
            return
        mode = self.cmb_trig_mode.currentData()
        src = self.cmb_trig_src.currentData()
        typ = self.cmb_trig_type.currentData()
        # An armed hardware trigger makes acquire() block until a pulse
        # arrives. The live feed would then never return a frame, pause() would
        # time out, and every later action would fail with FEED_BUSY_MSG and no
        # visible cause. Stop the feed for the operator and say so.
        stopped = False
        opts = avantes_trigger_options()
        hw = dict(opts["mode"]).get("Hardware trigger")
        if mode == hw and self.main.btn_feed.isChecked():
            self.main.btn_feed.setChecked(False)     # runs _toggle_feed
            stopped = True
        self._write("Trigger",
                    lambda d: d.set_trigger(mode, src, typ))
        if stopped:
            self._do(True, "Trigger: applied. The live feed was stopped — "
                           "under a hardware trigger a frame only arrives when "
                           "the experiment fires.")

    def _on_poll_toggle(self, on):
        if on:
            self._timer.start()
        else:
            self._timer.stop()

    def _poll(self):
        """Timed readback. Silent on failure: a busy feed must not turn the
        dialog into a blinking error, and the scan worker owns the device
        outright while a scan runs."""
        if not self.isVisible() or self.main._scan_running():
            return
        self._refresh_readback(quiet=True)

    def _refresh_readback(self, quiet=False):
        """Read the board temperature. A device call, so it takes the feed
        handover like any other."""
        dev = self._target()
        if dev is None or self.main._scan_running():
            return
        with self.main._device_lock() as ok:
            if not ok:
                if not quiet:
                    self._do(False, FEED_BUSY_MSG)
                return
            try:
                temp = dev.temperature_c()
            except Exception as e:
                if not quiet:
                    self._do(False, f"Temperature read failed: {e}")
                return
        self.lbl_temp.setText("Board temperature: " +
                              ("not reported by this device" if temp is None
                               else f"{temp:.1f} °C"))

    # ── state ───────────────────────────────────────────────────────────────
    def _refresh(self):
        """Seed every widget from the device. `_loading` suppresses the write
        handlers throughout, so reading the state cannot write it back."""
        devices = self.main._avantes_devices()
        self._loading = True
        try:
            self._sync_target_combo(devices)
            dev = self._target()
            for w in (self.spin_onboard_avg, self.spin_smooth, self.chk_hires,
                      self.chk_dark, self.chk_prescan, self.chk_sync,
                      self.cmb_trig_mode, self.cmb_trig_src,
                      self.cmb_trig_type):
                w.setEnabled(dev is not None)
            if dev is None:
                self.lbl_info.setText("No Avantes spectrometer is connected.")
                return

            self.spin_onboard_avg.setValue(dev.n_averages)
            self.spin_smooth.setValue(dev.smoothing_pixels)
            self.chk_hires.setChecked(dev.high_res_adc)
            self.chk_dark.setChecked(dev.dark_correction)

            opts = avantes_trigger_options()
            mode, src, typ = dev.trigger
            for cmb, key, value in ((self.cmb_trig_mode, "mode", mode),
                                    (self.cmb_trig_src, "source", src),
                                    (self.cmb_trig_type, "source_type", typ)):
                if cmb.count() == 0:
                    for label, val in opts[key]:
                        cmb.addItem(label, val)
                cmb.setCurrentIndex(max(0, cmb.findData(value)))

            n_avg = dev.n_averages
            self.lbl_avg_hint.setText(
                f"One frame costs {dev.integration_ms:g} ms "
                f"({dev.exposure_ms:g} ms exposure × {n_avg} "
                f"{'average' if n_avg == 1 else 'averages'}). The scan also "
                f"averages in software — leave this at 1 unless you want both.")
            self.lbl_hires.setText(
                f"Full scale {dev.max_counts:.0f} counts. Switching resolution "
                f"rescales the counts and re-arms the saturation alarm.")
            self.lbl_trig.setText(
                "Hardware trigger: a frame arrives only when the trigger "
                "input fires, so the live feed is stopped while it is armed."
                if dev.hardware_triggered else
                "Free running — the device clocks its own scans.")

            info = dev.device_info()
            wl_lo, wl_hi = info["wavelength_range_nm"]
            lines = [f"{info['model']}  [{info['serial']}]",
                     f"{info['board']} board · {info['n_pixels']} pixels · "
                     f"{wl_lo:.1f}–{wl_hi:.1f} nm",
                     f"firmware {info.get('firmware') or '?'} · "
                     f"FPGA {info.get('fpga') or '?'} · "
                     f"DLL {info.get('dll') or '?'}"]
            if info.get("detector"):
                lines.append(f"detector {info['detector']}")
            if not info.get("config_verified"):
                # Honest rather than silent: the EEPROM block did not match the
                # layout this build expects, so anything read from it is
                # suspect. Nothing above depends on it, which is why this is a
                # note and not a failure.
                lines.append("note: the device configuration block did not "
                             "match the expected layout, so EEPROM-derived "
                             "details are omitted.")
            self.lbl_info.setText("\n".join(lines))
        finally:
            self._loading = False

    # ── lifecycle ───────────────────────────────────────────────────────────
    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self._refresh(); self.lbl_msg.setText(""); self.show(); self.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh()
        self._refresh_readback(quiet=True)
        if self.chk_poll.isChecked():
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()          # never poll a hidden dialog
        super().hideEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Spectrometer picker (modal — shown only when enumeration finds 2+ devices)
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
        self.cmb = ComboBox()
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
# "Nothing connected" plot overlay
# ─────────────────────────────────────────────────────────────────────────────
class NoDeviceOverlay(QWidget):
    """Covers an empty plot panel with a message and a Connect button.

    Sized to the whole panel and painted in the plot's own background colour,
    so it reads as an empty plot carrying a note rather than a card dropped on
    top of one — the axes frame and labels still show around it.

    The plug is painted as a WATERMARK rather than stacked in the layout: a
    third row would push the message and button off the panel's centre, and the
    mark is meant to be read as background, not as content. It is drawn from
    the icon's alpha channel and tinted from the live PALETTE, so one file
    serves both themes and a theme switch needs no new asset — the same trick
    _glyph_icon uses for the feed button.
    """
    ICON_FRAC  = 0.45     # of the panel's shorter side
    ICON_ALPHA = 0.14     # faint enough to read the message over

    def __init__(self, canvas, message, button_text, on_click):
        super().__init__(canvas)
        self._src = (QPixmap(str(DISCONNECTED_ICON))
                     if DISCONNECTED_ICON.exists() else QPixmap())
        self._cache = None          # (side, tint) -> tinted pixmap
        lay = QVBoxLayout(self)
        lay.setSpacing(14)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addStretch()
        self.lbl = QLabel(message)
        self.lbl.setObjectName("nodev")
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.btn = QPushButton(button_text)
        self.btn.setObjectName("accent")
        self.btn.clicked.connect(on_click)
        lay.addWidget(self.btn, 0, Qt.AlignCenter)
        lay.addStretch()

    def _watermark(self, side):
        """The plug scaled to `side` px and tinted for the current theme."""
        tint = PALETTE["text_dim"]
        if self._cache is not None and self._cache[0] == (side, tint):
            return self._cache[1]
        scaled = self._src.scaled(side, side, Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation)
        out = QPixmap(scaled.size())
        out.fill(Qt.transparent)
        p = QPainter(out)
        p.drawPixmap(0, 0, scaled)
        # SourceIn keeps the destination's alpha and replaces its colour, so
        # this repaints the glyph in the palette colour and leaves the
        # transparent surround alone.
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(out.rect(), QColor(tint))
        p.end()
        self._cache = ((side, tint), out)
        return out

    def paintEvent(self, _event):
        p = QPainter(self)
        # The plot's own background, not the window's: blending in is the point.
        p.fillRect(self.rect(), QColor(PALETTE["plot_bg"]))
        # Redraw the axes frame this widget is standing on. Covering the panel
        # rect exactly still eats the spine whenever the screen's device pixel
        # ratio is above 1: matplotlib centres the spine ON the axes boundary,
        # so half its width falls on physical rows INSIDE the logical rect and
        # the border came out thinned at the sides and missing along the
        # bottom. Drawing it here in the same colour puts a continuous frame
        # back, at any ratio, without insetting the overlay (which would leave
        # grid stubs poking out around the edges).
        pen = QPen(QColor(PALETTE["border"]))
        pen.setWidth(1)
        p.setPen(pen)
        # adjusted: a 1px pen straddles the path, so an un-inset rect would
        # draw half of each edge outside the widget and get clipped away.
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._src.isNull():
            return
        side = int(min(self.width(), self.height()) * self.ICON_FRAC)
        if side < 16:
            return          # a panel this small has no room for a watermark
        pm = self._watermark(side)
        p.setOpacity(self.ICON_ALPHA)
        p.drawPixmap((self.width() - pm.width()) // 2,
                     (self.height() - pm.height()) // 2, pm)

    def refresh_theme(self):
        """Repaint against the new PALETTE — the tint is baked into the cache."""
        self._cache = None
        self.update()


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
        self.cmb_cmap.setCurrentText(canvas._cmap_name)
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
        self.spin_lw.setSingleStep(0.1); self.spin_lw.setValue(canvas._lw)
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
        on a CHANGE, and the widget construction defaults are not the canvas
        defaults — an empty spectrum panel sits nowhere near 0…5000 counts, so a
        restored value that happens to match a spinbox's default would reach the
        widget and never reach the plot.

        Every setter below is idempotent and their draw_idle()s coalesce into
        one frame, so re-asserting the lot is cheaper to reason about than
        working out which ones stayed silent.
        """
        c = self.canvas
        c.set_cmap(self.cmb_cmap.currentText())
        c.set_trace_reversed(self.chk_cmap_rev.isChecked())
        c.set_trace_threshold(self.spin_thresh.value())
        c.set_linewidth(self.spin_lw.value())
        c.set_log_scale(self.chk_log.isChecked())
        c.set_proportions(self.sld_prop.value() / 100.0)
        # Manual bounds only mean anything with their auto-scale off; applying
        # them anyway would fight the very next live frame.
        if not c.autoscale_x:
            c.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
        if not c.autoscale_y:
            c.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
        if not c.autoscale_trace:
            c.set_trace_xlim(self.spin_tmin.value(), self.spin_tmax.value())
            c.set_trace_ylim(self.spin_twmin.value(), self.spin_twmax.value())

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
# ── Plot geometry, in figure fractions (see FrogCanvas._layout_axes) ─────────
# Margins are sized for the 9-pt plot fonts: LEFT holds the spectrum's y label
# plus its tick labels, COLGAP the right column's y label, and BOT the bottom
# row's tick labels and x label.
_GEO_R = 0.99
# TOP is the header band: the panel title and that panel's overlay buttons share
# one line above the axes. Sized in pixels for the same reason as LEFT below —
# it holds a fixed-size Qt button, not a fraction of the figure.
_GEO_TOP_PX  = 34    # HDR_BTN (24) + HDR_PAD (6) + 4 px above the button
_GEO_TOP_MAX = 0.12  # …but never eat this much of a short window
# The left margin and the column gap hold text at a FIXED point size, so they
# are sized in logical pixels and only converted to a fraction at layout time:
# one fraction that fits a small window wastes half of a large one (the old
# 0.10 left margin was 90 px at 900 px wide but 192 px at 1920, for content
# that never needs more than the measured worst case below).
_GEO_L_PX   = 68     # y label + 5-digit tick labels at 8.5 pt measure ~58 px
_GEO_L_MAX  = 0.14   # …but never eat this much of a narrow window
_GEO_GAP_PX = 70     # column-gap floor: the right column's y label and tick
                     # labels, plus clearance for the split handle beside them
_GEO_ROW_RATIO = 1.8        # tall row : short row, both modes
_GEO_V_BOT     = 0.095
_GEO_V_COLGAP  = 0.055
_GEO_V_HSPACE  = 0.31       # the gridspec hspace the vertical layout was tuned with
_GEO_H_BOT     = 0.095
_GEO_H_COLGAP  = 0.065      # wider: two stacked y-axes share this gap, and the
                            # autocorrelation's tick labels run longer than the
                            # trace's, so 0.055 leaves its y label touching.
_GEO_H_ROWGAP  = 0.0        # flush: the stacked pair shares one delay axis, so
                            # the trace's bottom spine IS the AC's top spine

# The autocorrelation panel is drawn normalized to its own peak, so its view is
# fixed: 0…1 plus a little headroom to keep the peak off the top spine.
_AC_YLIM = (0.0, 1.05)

# Title offsets in inches, so the gaps are DPI- and resize-independent.
_TITLE_ABOVE_IN  = 6 / 72   # centres the 12 pt title in the header band
_TITLE_INSET_IN  = (0.08, 0.06)   # (right, down) from the axes' top-left corner

# The rest of the panel header band (HDR_BTN and friends) is defined up with
# the other Qt chrome constants — _glyph_icon needs HDR_ICON at import time.


class PlotSplitHandle(QWidget):
    """The divider between the spectrum and the panel(s) beside it: a separator
    line down the shared height, with a draggable grip at its middle — the
    Graphics dialog's proportion slider, in reach.

    A Qt child of the canvas rather than a matplotlib artist: hover feedback and
    drag tracking cost one small widget repaint instead of a figure redraw, and
    the divider composites above the canvas without ever landing in the cached
    blit background (the same reason the rubber band is a Qt widget).
    """
    W = 13          # logical px: wide enough to grab, narrow enough for the gap
    GRIP_H = 46     # grip length; the separator line runs the full height

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._hover = False
        self._press_x = None      # global x at press
        self._press_frac = None   # spectrum fraction at press
        self.setCursor(Qt.SplitHCursor)
        self.setToolTip("Drag to resize the spectrum panel  ·  "
                        "double-click for 50 / 50")

    # ── Painting ──────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        # PALETTE is read at paint time, so a theme switch only needs update().
        active = self._hover or self._press_x is not None
        h, cx = float(self.height()), self.width() / 2.0
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        # Separator: a hairline the full height of the two panels, so they read
        # as separate panels whether or not anyone ever drags it.
        p.setBrush(QColor(PALETTE["border"]))
        p.drawRect(QRectF(cx - 0.5, 0.0, 1.0, h))
        # Grip: the same line thickened over a short run at the middle. Accent
        # on hover so it is discoverable without shouting at rest.
        gw, gh = 5.0, min(self.GRIP_H, h)
        p.setBrush(QColor(PALETTE["accent"] if active
                          else PALETTE["border_hover"]))
        p.drawRoundedRect(QRectF(cx - gw / 2.0, (h - gh) / 2.0, gw, gh),
                          gw / 2.0, gw / 2.0)
        # Grip dots punched in the panel colour: they read as a handle at any
        # size, where a plain bar could pass for a plot decoration.
        p.setBrush(QColor(PALETTE["plot_bg"]))
        for dy in (-5.0, 0.0, 5.0):
            p.drawEllipse(QPointF(cx, h / 2.0 + dy), 0.9, 0.9)

    def enterEvent(self, event):
        self._hover = True; self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False; self.update()
        super().leaveEvent(event)

    # ── Dragging ──────────────────────────────────────────────────────────
    # Anchored to the press position rather than to the cursor's offset within
    # the grip, so the split never jumps on the first move.
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._press_x = event.globalPosition().x()
        self._press_frac = self._canvas.spec_fraction()
        self.update()

    def mouseMoveEvent(self, event):
        if self._press_x is None:
            return
        travel = self._canvas.split_travel_px()
        if travel <= 0:
            return
        self._canvas.set_proportions(
            self._press_frac + (event.globalPosition().x() - self._press_x) / travel)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._press_x = self._press_frac = None
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._canvas.set_proportions(0.50)


class FrogCanvas(FigureCanvasQTAgg):
    limits_changed = Signal()        # zoom/reset happened → window syncs dialog
    log_toggle_requested = Signal()  # click on spectrum y-axis strip
    proportions_changed = Signal(int)  # split handle dragged → dialog follows
    axes_relaid = Signal()           # axes repositioned → overlay buttons follow

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
        # Alignment mode: the two symmetry-difference curves. Independent of
        # the combined/member switch above, so they stay on screen over a live
        # spectrum, a scan column or the per-member view without any of those
        # having to know about them. Drawn in the SAME count units as the
        # spectrum, hence the zero reference line — a difference is only
        # readable against the level it is a difference from.
        (self.line_d1,) = self.ax_spec.plot([], [], color=DIFF_COLORS[0], lw=1.2)
        (self.line_d2,) = self.ax_spec.plot([], [], color=DIFF_COLORS[1], lw=1.2)
        self.line_d1.set_visible(False)
        self.line_d2.set_visible(False)
        # Static, like band_span: it moves only when the view does.
        self.diff_zero = self.ax_spec.axhline(
            0.0, color=PALETTE["text_dim"], lw=0.8, ls="--", zorder=0)
        self.diff_zero.set_visible(False)
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
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
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
        self.line_d1.set_animated(True)
        self.line_d2.set_animated(True)
        self.im.set_animated(True)
        self.txt_fwhm.set_animated(True)
        self._animated = [(self.ax_spec, self.line_spec),
                          (self.ax_spec, self.line_m1),
                          (self.ax_spec, self.line_m2),
                          (self.ax_spec, self.line_d1),
                          (self.ax_spec, self.line_d2),
                          (self.ax_trace, self.im),
                          (self.ax_ac, self.line_ac),
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
        self._clim_peak = None     # last clim top actually applied to the image
        # Cache last-applied limits so we only force a full redraw when they
        # actually move (autoscale otherwise re-sets identical limits each frame).
        self._xlim_cache = None
        self._ylim_cache = None
        # Continuous y auto-scale filter state — see _autoscale_y / _pin_ylim.
        self._ylim_smooth = None   # where the view is heading, unquantized
        self._ylim_t      = 0.0    # monotonic stamp of the last filter step
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
                 (self.ax_ac,    "Delay (fs)",      "AC (a.u.)",       "Autocorrelation")]
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

    def _layout_axes(self):
        """Position all three axes for the current mode and spectrum width.

        Single source of truth for the plot geometry: the axes are placed
        explicitly instead of being left where the gridspec put them, so the
        proportion slider works in either mode (and before the first draw).
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
            self.ax_spec.set_position([L, BOT, spec_w, TOP - BOT])
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
            self.ax_spec.set_position([L, BOT + h_ac + rgap, spec_w, h_top])
            self.ax_trace.set_position([right_x, BOT + h_ac + rgap, right_w, h_top])
            self.ax_ac.set_position([L, BOT, R - L, h_ac])
            # Only the top row is split — the autocorrelation spans both columns.
            split_bot = BOT + h_ac + rgap
        # The gap's centre line, over the rows the two sides actually share.
        self._split_band = (right_x - cgap / 2.0, split_bot, TOP)
        self._position_split_handle()
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
        """Horizontal mode: the FROG trace owns the shared delay axis and the
        autocorrelation mirrors it. Returns True if the AC view moved.

        Deliberately does NOT touch autoscale_ac_x: the flag keeps whatever the
        user left it on, so switching back to vertical resumes independent AC
        autoscaling with no bookkeeping. No-op in vertical mode.
        """
        if self._layout_mode != "horizontal":
            return False
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
        self.txt_fwhm.set_color(pal["accent2"])
        self.txt_fwhm.get_bbox_patch().set_facecolor(pal["surface"])
        self.txt_fwhm.get_bbox_patch().set_edgecolor(pal["border"])
        # line_m1/line_m2 keep MEMBER_COLORS in both themes — they are chosen
        # to work on either background, and re-theming them would cost the
        # colourblind separation that is the whole point. Same for the two
        # DIFF_COLORS curves; their zero reference is a plain rule, so it does
        # follow the theme.
        self.diff_zero.set_color(pal["text_dim"])
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
        self._bg = self.copy_from_bbox(self.fig.bbox)
        self._bg_static = None       # figure changed; recomposite lazily
        for ax, art in self._animated:
            ax.draw_artist(art)
        # Only a completed draw knows how much of the gap the tick labels took,
        # and only then can the separator be placed clear of both panels. Moving
        # a Qt child cannot recurse back into the figure draw.
        self._renderer = getattr(event, "renderer", None) or self._renderer
        self._position_split_handle()

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
                self.ax_ac.draw_artist(self.line_ac)
                # Static between scan columns, exactly like the AC line — so it
                # belongs in the cached background, not in the per-frame draw.
                self.ax_ac.draw_artist(self.txt_fwhm)
                self._bg_static = self.copy_from_bbox(self.fig.bbox)
            self.restore_region(self._bg_static)
            self.ax_spec.draw_artist(self.line_spec)
            self.ax_spec.draw_artist(self.line_m1)
            self.ax_spec.draw_artist(self.line_m2)
            # The alignment differences are animated too, so the restore above
            # wiped them out of the buffer — every live frame has to put them
            # back or they would flicker away under the feed.
            self.ax_spec.draw_artist(self.line_d1)
            self.ax_spec.draw_artist(self.line_d2)
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
                self.reset_axes(event.inaxes)  # …otherwise toggles auto-scale
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
            self._pin_ylim((ylo, yhi))
        elif ax is self.ax_ac:
            self.autoscale_ac_x = False
            self.autoscale_ac_y = False
            self._ac_xlim_cache = (xlo, xhi)
            self._ac_ylim_cache = (ylo, yhi)
            if self._layout_mode == "horizontal":
                # Shared delay axis: hand the zoom to its owner, the trace, so
                # both stay registered (and the dialog's trace bounds truthful).
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
            self._set_spec_autoscale(not (self.autoscale_x or self.autoscale_y))
        elif ax is self.ax_trace:
            self._set_trace_autoscale(not self.autoscale_trace)
        elif ax is self.ax_ac:
            on = not (self.autoscale_ac_x or self.autoscale_ac_y)
            self._set_ac_autoscale(on)
            if self._layout_mode == "horizontal":
                # Shared delay axis: the trace owns it, so it has to go the
                # SAME way rather than toggle off its own flag. (_sync_ac_x in
                # there then pulls the AC's x back onto the scan range.)
                self._set_trace_autoscale(on)
        self.draw_idle()
        self.limits_changed.emit()

    def _set_spec_autoscale(self, on):
        """Spectrum panel auto-scale, both axes together. On fits first."""
        if not on:
            self.autoscale_x = False           # freeze wherever it sits now
            self.autoscale_y = False
            self._xlim_cache = self.ax_spec.get_xlim()
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
        self._xlim_cache = self.ax_spec.get_xlim()
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
            self._xlim_cache = (xmin, xmax)
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

    def set_ac_xlim(self, xmin, xmax):
        """Autocorrelation delay axis. No dialog row drives this — it exists so
        a frozen AC view (right-click, or a rubber-band zoom) can be restored
        between sessions together with autoscale_ac_x, which on its own would
        pin the panel to a range it was never pinned to."""
        if xmin < xmax:
            self.ax_ac.set_xlim(xmin, xmax)
            self._ac_xlim_cache = (xmin, xmax)
            self.draw_idle()

    def set_ac_ylim(self, ymin, ymax):
        """Autocorrelation amplitude axis. See set_ac_xlim."""
        if ymin < ymax:
            self.ax_ac.set_ylim(ymin, ymax)
            self._ac_ylim_cache = (ymin, ymax)
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

    def show_diff(self, wl, d1, d2):
        """Overlay the two alignment symmetry differences on the spectrum.

        Untouched by every other spectrum path, so the live feed keeps drawing
        underneath: the curves stay until clear_diff().
        """
        self.line_d1.set_data(wl, d1)
        self.line_d2.set_data(wl, d2)
        self.line_d1.set_visible(True)
        self.line_d2.set_visible(True)
        self.diff_zero.set_visible(True)
        # A difference dips below zero, so the y range almost always has to
        # move; and diff_zero is a static artist that has to be baked into the
        # blit backgrounds either way. Autoscale first so one full draw covers
        # both — _autoscale_y reads the visible lines, which now include these.
        if self.autoscale_y:
            self._autoscale_y()
        self._request_full()

    def clear_diff(self):
        """Take the alignment differences off the spectrum panel.

        Data cleared as well as hidden, for the reason in _set_spec_mode: a
        hidden line still holding a frame keeps driving relim().
        """
        self.line_d1.set_data([], [])
        self.line_d2.set_data([], [])
        self.line_d1.set_visible(False)
        self.line_d2.set_visible(False)
        self.diff_zero.set_visible(False)
        if self.autoscale_y:
            self._autoscale_y()
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
                self.ax_spec.set_xlim(*xl); self._xlim_cache = xl; changed = True
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
    _YLIM_TAU_UP   = 0.15   # s — view growing to meet new data
    _YLIM_TAU_DOWN = 2.0    # s — view shrinking after the signal drops
    _YLIM_HYST     = 0.04   # commit deadband, fraction of the applied span

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
        # Asymmetric about the CURRENT filter state, not the applied limits: the
        # bottom expands by going lower, the top by going higher, so each end
        # gets the fast constant only in the direction that grows the view.
        lo = _ease(slo, target[0], dt,
                   self._YLIM_TAU_UP if target[0] < slo else self._YLIM_TAU_DOWN)
        hi = _ease(shi, target[1], dt,
                   self._YLIM_TAU_UP if target[1] > shi else self._YLIM_TAU_DOWN)
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
        self.autoscale_y = on

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
                self.ax_spec.set_xlim(*xl); self._xlim_cache = xl; changed = True
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
        self.line_ac.set_data(delays, ac / peak if peak > 0 else ac)
        self._bg_static = None        # AC line changed
        changed = False
        if self._layout_mode == "horizontal":
            # Shared delay axis: the trace dictates it, so the AC's own x
            # autoscale sits this one out (its flag is left untouched, and takes
            # over again the moment the layout goes back to vertical).
            changed = self._sync_ac_x()
        elif self.autoscale_ac_x and delays.size > 1:
            xl = (float(delays[0]), float(delays[-1]))
            if xl != self._ac_xlim_cache:
                self.ax_ac.set_xlim(*xl); self._ac_xlim_cache = xl; changed = True
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
    # Raised by the stage layer (StageBase.warn_cb) when it has to do something
    # the operator should know about — currently only a long move being split.
    # A signal, not a direct status-bar call: the scan worker drives the same
    # stage from its own thread, and Qt queues cross-thread emissions for us.
    stage_warning = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lillypad — Fast")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1180, 760)
        # load_settings() is cached, so this is the same dict main() already
        # read the theme from on its way to styling the QApplication — the two
        # cannot disagree. The rest is applied at the END of __init__, once the
        # widgets it drives exist.
        self._settings = load_settings()
        self._settings_saved = {}
        self._theme = startup_theme(self._settings)

        self.scan_cfg      = FrogScanConfig(delay_start_fs=-500, delay_stop_fs=500,
                                            delay_step_fs=1.0, zero_pos_um=150000.0)
        self._stage_units_fs = True     # jog/move fields default to fs
        # The recorded dark, in exactly one of two forms — never both.
        #   single device:  self.background, one RAW frame.
        #   stitched pair:  self.background_members, the two RAW member frames,
        #                   and self.background stays None.
        # A pair's dark is kept per member because the pedestal has to come off
        # BEFORE each member's own calibration, and because the merged
        # background is then derived (StitchedSpectrometer.combined_dark) at
        # whatever stitch factor and band are current. Storing the merged frame
        # instead baked both into its overlap region, where the two devices are
        # already mixed, and it went silently wrong on the next re-fit.
        self.background    = None
        self.background_members = None
        # Exposure(s) the dark was recorded at — (exp,) or (exp1, exp2) in
        # self.spec.members order. A dark is offset + dark-current x t, so it
        # is only valid at its own exposure; _invalidate_dark drops it when
        # that changes rather than rescaling something that is not a pure ratio.
        self.background_exposures = None
        self._dark_member_warned = False
        self._overlay_on   = False   # spectrum panel showing the members apart
        # Set when an integration time changes under a stitched pair: frames
        # are raw counts, so stitch_factor carries the exposure ratio and goes
        # stale. Surfaced in the Multi-Spec menu rather than silently re-fitted.
        self._stitch_stale = False
        self.last_spectrum = None
        self.result        = None
        self._worker       = None
        self._export_worker = None
        # The two spectrometer slots, each holding a device choice and a
        # calibration choice; _multi_members are the live spectrometer objects
        # in SLOT order once a stitched pair is connected.
        # "serials" holds tagged ids ("vendor:serial", see hardware.spec_ident)
        # or one of the SIM_SLOT_DEVICES sentinels — the two slots may hold
        # devices from different vendors.
        #
        # There is no "multi-spectrometer mode" flag: both slots exist from the
        # start, slot 1 alone IS single-spectrometer mode, and "is a pair live"
        # is answered by _pair_live() off the connected device rather than by a
        # second piece of state that could disagree with it.
        self._multi = {"serials": [None, None],
                       "labels": [None, None], "cals": [None, None]}
        self._multi_members = [None, None]
        self._scan_trace   = None
        self._scan_delays  = None
        self._scan_ac      = None    # running per-column sums (see _on_column)
        self._scan_peak    = 0.0     # running trace max, for the image clim
        # Cached at scan start so _on_column can redraw the spectrum panel
        # without touching self.spec from the GUI thread while the worker is
        # inside acquire().
        self._scan_wl      = None
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
        self._feed_was_on  = True
        self._pending_fit  = True
        self._export_fmt   = "dwc"
        # Which beam the simulated spectrometer measures (see PULSE_SHAPES).
        # Kept on the window, not the simulator, so the choice survives a swap
        # to real hardware and back.
        self.sim_pulse     = DEFAULT_PULSE
        self.sim_gate      = "shg"
        # seabreeze backend used for every enumeration/connect. pyseabreeze
        # (the default, SEABREEZE_BACKENDS[0]) also covers the newer Ocean
        # Insight models that cseabreeze does not know about.
        self.seabreeze_backend = SEABREEZE_BACKENDS[0]

        # NOTHING is connected at startup. The app used to open on a simulated
        # stage and spectrometer, which meant a fake spectrum was on screen
        # before anyone had chosen anything and looked exactly like a
        # measurement. Both stay None until the operator connects a device (or
        # asks for a simulator by name, in the Simulation window), and
        # _sync_connection_ui puts a "no spectrometer connected" panel up in
        # the meantime.
        self.spec  = None
        self.stage = None

        self.dlg_settings = AcquisitionSettingsDialog(self)
        self.dlg_align    = AlignmentDialog(self)
        self.dlg_spec     = SpectrometerDialog(self, self)
        self.dlg_stage    = StageDialog(self, self)
        self.dlg_sim      = SimulationDialog(self, self)
        # Vendor-specific controls, opened from a toolbar button that stays
        # hidden until an Avantes is actually connected.
        self.dlg_avantes  = AvantesSettingsDialog(self, self)
        self.spin_avg  = self.dlg_settings.spin_avg
        self.spin_idle = self.dlg_settings.spin_idle
        self.spin_wait = self.dlg_settings.spin_wait
        # Controls that moved into sub-windows but are still driven from here.
        # Aliased rather than reached through the dialog at every call site, so
        # _sync_backlash_ui, _on_backlash_changed, _set_stage_controls_enabled
        # and the alignment sweep read exactly what they always did. Both
        # dialogs are built above _build_ui(), which is what needs them.
        self.spin_align_step = self.dlg_align.spin_align_step
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
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(60)      # ~16 fps display cadence
        self._display_timer.timeout.connect(self._display_tick)
        self._display_timer.start()

        # Opt-in instrumentation (LILLYPAD_PERF=1). Reports to stderr every 10 s.
        self._perf_timer = None
        if PERF:
            global _perf
            _perf = _PerfProbe(self)
            self._perf_timer = QTimer(self)
            self._perf_timer.setInterval(10_000)
            self._perf_timer.timeout.connect(_perf.report)
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

    # ── Connection state ─────────────────────────────────────────────────────
    # self.spec and self.stage are None when nothing is connected — a real
    # absence rather than a stand-in device, so nothing can quietly return
    # plausible-looking counts or positions for hardware that is not there.
    # Everything that reaches a device goes through one of these three.
    def _have_spec(self):
        return self.spec is not None

    def _have_stage(self):
        return self.stage is not None

    def _pair_live(self):
        """True while two spectrometers are connected as one stitched device.
        This is the ONLY definition of multi-spectrometer mode."""
        return isinstance(self.spec, StitchedSpectrometer)

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
        """
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
            # Same for both halves of alignment mode: a difference and a raw
            # trace describe the device that measured them, right down to the
            # pixel grid they sit on.
            self._uncheck_align_spec()
            self.canvas.clear_diff()
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
        """
        for slot, (ident, label) in enumerate(SIM_SLOT_DEVICES):
            self._multi["serials"][slot] = ident
            self._multi["labels"][slot]  = label
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
        tb.addWidget(self._build_export_button())
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
        if RESCALE_ICON.exists():
            self.btn_autofit.setIcon(QIcon(str(RESCALE_ICON)))
            self.btn_autofit.setIconSize(QSize(HDR_ICON, HDR_ICON))
        else:
            self.btn_autofit.setText("↔↕")
        self.btn_autofit.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_autofit.setToolTip(
            "Auto-fit spectrum X and Y axes to current data.\n"
            + RIGHT_CLICK_HINT)
        self.btn_autofit.show()
        self.btn_autofit.clicked.connect(self._autofit_spectrum)

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

        # Alignment mode, spectrum side: step to +/-x and +/-2x and overlay the
        # two differences.
        self.btn_align_spec = QPushButton(self.canvas)
        self.btn_align_spec.setObjectName("overlay")
        self.btn_align_spec.setCheckable(True)
        self.btn_align_spec.setFixedSize(HDR_BTN, HDR_BTN)
        self.btn_align_spec.setIconSize(QSize(HDR_ICON, HDR_ICON))
        self._refresh_align_button()
        self.btn_align_spec.setToolTip(
            "Alignment mode — measure at −2x, −x, +x, +2x (Alignment → "
            "Alignment step) and overlay S(+x)−S(−x) and S(+2x)−S(−2x).\nA "
            "symmetric pulse gives two flat curves on zero. Press again to "
            "clear.")
        self.btn_align_spec.show()
        self.btn_align_spec.toggled.connect(self._on_align_spec_toggled)

        # Alignment mode, trace side — the trace panel's own header.
        # Wider than the rest: it is the one button still carrying a word.
        self.btn_align_trace = QPushButton("RAW", self.canvas)
        self.btn_align_trace.setObjectName("overlay")
        self.btn_align_trace.setCheckable(True)
        self.btn_align_trace.setFixedSize(40, HDR_BTN)
        self.btn_align_trace.hide()
        self.btn_align_trace.toggled.connect(self._on_align_trace_toggled)

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

        self._refresh_layout_button()    # needs the canvas for the current mode

        self.dlg_graphics = GraphicsSettingsDialog(self.canvas, self)
        # Mouse zoom/reset keeps the dialog's spinboxes and auto-scale
        # checkboxes truthful; the y-axis click routes through chk_log so the
        # checkbox stays the single source of truth for the log scale.
        self.canvas.limits_changed.connect(self.dlg_graphics.sync_limits)
        self.canvas.log_toggle_requested.connect(self.dlg_graphics.chk_log.toggle)
        self.canvas.proportions_changed.connect(self.dlg_graphics.sync_proportions)

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
        # The shading and the merged curve both change; push the band to the
        # canvas now rather than waiting for the next overlay frame, so the
        # combined view updates too.
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
        # — the same numbering as the saturation lamps and the Multi-Spec menu.
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
        _readout("Stage", self.lbl_moving, 0, Qt.AlignHCenter)
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
        self.spin_step.setValue(100.0)     # after ranges: default 100 fs jog
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
        cell(0, 0, "Start", self.spin_start, " fs", 1, -1e6, 1e6, -500.0)
        cell(0, 2, "Stop",  self.spin_stop,  " fs", 1, -1e6, 1e6,  500.0)
        # The um equivalents move under their own boxes — as a full-width pair
        # they would push the group back out to three rows.
        self.eq_start = QLabel(""); self.eq_start.setObjectName("dim")
        self.eq_stop  = QLabel(""); self.eq_stop.setObjectName("dim")
        g.addWidget(self.eq_start, 1, 1)
        g.addWidget(self.eq_stop,  1, 3)
        cell(2, 0, "Step", self.spin_step_fs, " fs", 1, 0.1, 1e5, 1.0)
        self.chk_bg = QCheckBox("Background")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip("Record a background frame before and after the "
                               "scan, with the beam blocked when prompted.")
        g.addWidget(self.chk_bg, 2, 3, Qt.AlignLeft | Qt.AlignVCenter)
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
        self.btn_save.clicked.connect(lambda: self._export_as(self._export_fmt))
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
        self._refresh_autostitch_icon()  # …as does this one
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
            self.dlg_spec._refresh()
        if self.dlg_stage.isVisible():
            self.dlg_stage._refresh()
        if self.dlg_sim.isVisible():
            self.dlg_sim._refresh()

    def _sync_multi_ui(self):
        """Bring every stitched-vs-single widget into line with self.spec.

        One owner for the second integration spinbox and the overlay button,
        so no caller has to remember the set. Pure widget state — no device
        I/O, so it is safe to call outside _device_lock.
        """
        stitched = self._pair_live()
        slots = self._slot_members() if stitched else []
        # One caption over both boxes — S1 on the left, S2 on the right.
        self.lbl_integration.setText("Integration Time — S1 / S2" if stitched
                                     else "Integration Time")
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

    def _refresh_autostitch_button(self):
        """Show the Auto-stitch header button only while a pair is live.

        Same rule as the overlay toggle beside it: the fit needs two members to
        match against, and it drives the same hardware a scan owns.
        """
        self.btn_autostitch.setVisible(self._pair_live()
                                       and len(self._slot_members()) == 2)
        self.btn_autostitch.setEnabled(not self._scan_running())
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
            self.dlg_avantes._refresh()  # a swap may have changed which device

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
        self.btn_overlay.setEnabled(not self._scan_running())
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

    def _render_overlay(self):
        """Draw one curve per stitched member. False = nothing to draw yet, so
        the caller falls back to the combined frame.

        Runs the SAME prepare_pair() the merged frame does, then applies each
        member's scale, so the two curves and the merged one are two views of
        one computation rather than two derivations that can drift. Each member
        keeps its NATIVE pixel grid; no interpolation happens here.

        Reads self.spec.last_member_raw from the GUI thread without the device
        lock, exactly as _update_saturation does: the tuple assignment is
        atomic, and calibrate() plus the wavelength arrays are pure.
        """
        spec = self.spec
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

    # ── Alignment mode: spectrum symmetry ────────────────────────────────────
    def _align_running(self):
        return (self._align_worker is not None
                and self._align_worker.isRunning())

    def _uncheck_align_spec(self):
        """Pop the Δ button back out without re-entering its own handler."""
        self.btn_align_spec.blockSignals(True)
        self.btn_align_spec.setChecked(False)
        self.btn_align_spec.blockSignals(False)

    def _on_align_spec_toggled(self, on):
        if not on:
            self.canvas.clear_diff()
            self.status.showMessage("Alignment differences cleared.", 2500)
            return
        if self._align_running():
            self._uncheck_align_spec(); return
        if self._scan_running():
            self.status.showMessage("A scan is running — the stage is busy.", 3000)
            self._uncheck_align_spec(); return
        if self.spec is None or self.stage is None:
            self.status.showMessage("Connect a stage and a spectrometer first.", 4000)
            self._uncheck_align_spec(); return
        if getattr(self.stage, "needs_homing", False):
            self.status.showMessage(
                "Stage is not homed — press Home before an alignment sweep.", 6000)
            self._uncheck_align_spec(); return

        # Same handover contract as _start_scan: the worker drives the stage
        # and the spectrometer for the whole sweep, so the feed has to let go
        # of them first, and refusing beats running against a device the feed
        # thread is still inside.
        self._feed_was_on = self.btn_feed.isChecked()
        if not self._feed.pause():
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(f"Alignment not started — {FEED_BUSY_MSG}", 6000)
            self._uncheck_align_spec(); return

        # Only now is it safe to read the stage from this thread — the feed is
        # parked, so nothing else is talking to it. Checked BEFORE moving:
        # clamping a target would quietly destroy the +/- symmetry the whole
        # measurement is about, so an out-of-range sweep is refused instead.
        step = float(self.spin_align_step.value())
        pf   = self.scan_cfg.pass_factor
        try:
            start_um = _stage_to_um(self.stage.get_position())
        except Exception as e:
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(f"Alignment failed — stage: {e}", 5000)
            self._uncheck_align_spec(); return
        span_um = abs(float(delay_to_position_um(2.0 * step, 0.0, pf)))
        lo, hi = self._travel_range_um()
        if start_um - span_um < lo or start_um + span_um > hi:
            if self._feed_was_on:
                self._feed.resume()
            self.status.showMessage(
                f"±2×{step:.0f} fs (±{span_um:.1f} um) from here leaves the "
                f"travel range [{lo:.1f}, {hi:.1f}] um — move away from the "
                f"limit or reduce the Alignment Step.", 7000)
            self._uncheck_align_spec(); return

        self._set_hardware_buttons_enabled(False)
        self.btn_scan.setEnabled(False)
        self._set_stage_controls_enabled(False)
        self.btn_align_spec.setEnabled(False)
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
            self.status.showMessage(f"Alignment differences failed: {e}", 5000)
            self._uncheck_align_spec()
            return
        self.canvas.show_diff(wl, d1, d2)
        step = float(self.spin_align_step.value())
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
        self._reset_align_ui()
        self._uncheck_align_spec()
        self.status.showMessage(f"Alignment sweep failed: {msg}", 6000)

    def _reset_align_ui(self):
        """Hand the devices back and re-enable what the sweep locked out."""
        self._moving(False)
        self._set_hardware_buttons_enabled(True)
        self.btn_scan.setEnabled(True)
        self._set_stage_controls_enabled(True)
        self.btn_align_spec.setEnabled(True)
        self._refresh_positions()
        if self._feed_was_on:
            self._feed.resume()

    # ── Alignment mode: raw stitched trace ───────────────────────────────────
    def _position_panel_buttons(self):
        """Lay out each panel's header row: the buttons that act on that panel,
        right-aligned to its right edge, on the line the panel title occupies.

        Every edge here moves with the split fraction, the window size and the
        layout mode, so this runs off canvas.axes_relaid — and off the two
        refresh methods that show or hide a button, because the row is packed
        right-to-left and its width depends on which buttons are visible.
        """
        # getattr, not the attribute: the refresh methods that call this also
        # run from _apply_spectrometer, which the connect paths can reach
        # before _build_ui has created any of these buttons.
        if getattr(self, "pnl_no_spec", None) is None:
            return
        c = self.canvas

        def pack(ax, buttons):
            _left, top, right = c.panel_edges_px(ax)
            # The band sits ABOVE the axes: HDR_PAD of clear space between the
            # buttons' bottom edge and the top spine.
            y = round(top) - HDR_PAD - HDR_BTN
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

        pack(c.ax_spec, [self.btn_autofit, self.btn_feed, self.btn_autostitch,
                         self.btn_overlay, self.btn_align_spec])
        pack(c.ax_trace, [self.btn_align_trace])

        # The "no spectrometer connected" panel, stretched over the whole
        # spectrum axes so it reads as the empty plot itself rather than a card
        # floating in it. It is laid out only here — it has no parent layout to
        # do that for it.
        self.pnl_no_spec.setGeometry(c.panel_rect_px(c.ax_spec))

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
        # Re-render at once rather than waiting for the next column, so the
        # toggle also works on a finished trace.
        if self._align_trace_on:
            self.canvas.update_trace(self._align_trace_view(), 1.0)
        elif self._scan_trace is not None:
            self.canvas.update_trace(self._scan_trace, self._scan_peak)
        self.status.showMessage(
            "FROG trace: raw counts, per-spectrometer normalised." if on else
            "FROG trace: calibrated, stitched.", 4000)

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
        """Fold one column's RAW member frames into the raw trace. O(n_wl)."""
        if self._align_trace is None or members_raw is None:
            return
        w = self._scan_wl
        k = self._align_cut
        wl1, wl2 = self._align_wl
        m1, m2 = members_raw
        lo = np.interp(w[:k], wl1, np.asarray(m1, float))
        hi = np.interp(w[k:], wl2, np.asarray(m2, float))
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
                f"share a scale; re-run Multi-Spec → Auto-stitch.", 8000)
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
        c.n_average      = self.spin_avg.value()
        c.idle_shots     = self.spin_idle.value()
        c.wait_after_move_s = self.spin_wait.value() / 1000.0
        c.capture_background = self.chk_bg.isChecked()
        c.saturation_fraction  = self.dlg_settings.spin_sat.value() / 100.0
        c.abort_on_saturation  = self.dlg_settings.chk_abort_sat.isChecked()
        c.abort_on_stage_fault = self.dlg_settings.chk_abort_stage_fault.isChecked()
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
        self._set_hardware_buttons_enabled(False)
        self._set_stage_controls_enabled(False)

        self._scan_trace  = np.zeros((wl.size, delays.size))
        self._scan_delays = delays
        self._scan_wl     = wl
        # Running reductions _on_column keeps current, so the per-column render
        # never has to walk the trace. Reallocated with the trace: an aborted
        # scan leaves these sized to the OLD delay axis.
        self._scan_ac     = np.zeros(delays.size)
        self._scan_peak   = 0.0
        # A pending render from a previous (aborted) scan must not fire
        # against the fresh, differently-sized arrays.
        self._scan_dirty  = False
        self._scan_last_i = -1
        self._live_frame  = None     # park any leftover live-feed frame too
        self._init_align_trace(wl, delays.size)
        self.canvas.init_trace(delays, wl)
        self._reset_saturation(); self.progress.setValue(0)
        # A difference measured at the old position says nothing about the
        # scan that is starting, and the scan owns the spectrum panel now.
        if self.canvas.diff_visible():
            self._uncheck_align_spec()
            self.canvas.clear_diff()

        self.btn_scan.setObjectName("danger"); self.btn_scan.setText("Abort Scan")
        self.btn_scan.style().unpolish(self.btn_scan); self.btn_scan.style().polish(self.btn_scan)
        self.btn_save.setEnabled(False)

        self._worker = FrogScanWorker(self.stage, self.spec, c)
        self._worker.progress.connect(self._on_progress)
        self._worker.column_ready.connect(self._on_column)
        self._worker.background_ready.connect(self._on_background)
        self._worker.saturation_warning.connect(self._on_saturation)
        self._worker.stage_fault.connect(self._on_stage_fault)
        self._worker.finished_scan.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        # The worker only hands back combined columns, so the panel falls back
        # to the stitched curve on its own (update_spectrum owns the mode);
        # grey the toggle out rather than let it look broken for the duration.
        self._refresh_overlay_button()
        self._refresh_autostitch_button()
        self.status.showMessage(f"FROG scan: {delays.size} points…", 0)

    def _on_progress(self, done, total):
        self.progress.setValue(int(100 * done / total))

    def _on_column(self, i, delay_fs, pos_um, col, members_raw=None):
        """Slot for FrogScanWorker.column_ready — O(1), like _on_spectrum.

        Every column is RECORDED here (the data path must never drop), but
        rendering is deferred to _display_tick so a slow machine skips
        intermediate redraws instead of queuing them up.
        """
        self._scan_trace[:, i] = col
        # The raw member frames are interpolated and cut ONCE, here, rather
        # than kept per column and re-derived at render time: that keeps the
        # memory to one trace-sized array and the work to O(n_wl) per column.
        self._store_align_column(i, members_raw)
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
            if self._scan_wl is not None:
                self.canvas.update_spectrum(self._scan_wl,
                                            self._scan_col_corrected())
            if self._align_trace_on and self._align_trace is not None:
                self.canvas.update_trace(self._align_trace_view(), 1.0)
            else:
                self.canvas.update_trace(self._scan_trace, self._scan_peak)
            self.canvas.update_ac(self._scan_delays[:i + 1], ac)
            # Inside the batch: the readout lives in the AC panel now, so its
            # repaint is the same blit the three updates above share.
            self.canvas.set_fwhm(fwhm(self._scan_delays[:i + 1], ac))

    def _scan_col_corrected(self):
        """The scan column to plot, with the dark removed.

        A scan column arrives already merged and calibrated, and there are no
        per-column member frames to rebuild it from, so a stitched pair takes
        the DERIVED merged dark — the member darks pushed through the current
        factor and band. That lands on the same baseline the live view used.
        A single device's dark is raw counts and needs the calibration applied
        to it first, since the worker's column already carries one.
        """
        col = self._scan_col
        if not self.chk_dark.isChecked():
            return col
        if self._pair_live():
            darks = self.background_members
            if darks is None:
                return col
            try:
                bg = self.spec.combined_dark(darks)
            except Exception:
                return col          # mismatched shapes: better raw than wrong
            if bg.shape != np.shape(col):
                return col
            return np.clip(col - bg, 0, None)
        if self.background is None:
            return col
        return np.clip(col - self.spec.calibrate(self.background), 0, None)

    def _on_background(self, which, spectrum):
        self.status.showMessage(f"Background ({which}) captured.", 2500)

    def _on_stage_fault(self, i, delay_fs, reason):
        """Slot for FrogScanWorker.stage_fault.

        With the abort enabled the worker's error() follows immediately and
        supersedes this; the message matters in the override case, where the
        scan carries on and this is the only live sign that it has stopped
        tracking. Timeout 0 — it must not quietly disappear.
        """
        where = "the background frame" if i < 0 else f"{delay_fs:+.0f} fs"
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

    def _on_finished(self, result):
        self.result = result
        # The final render below supersedes any pending intermediate tick.
        self._scan_dirty = False
        if self._align_trace_on and self._align_trace is not None:
            # The raw view is a display mode, not a stage of the scan: finishing
            # must not silently drop the operator back to the calibrated trace.
            self.canvas.update_trace(self._align_trace_view(), 1.0)
        else:
            self.canvas.update_trace(result.trace)
        ac = result.autocorrelation()
        self.canvas.update_ac(result.delays_fs, ac)
        self.canvas.set_fwhm(result.fwhm_fs())
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
        self.btn_save.setEnabled(True)
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
        self._refresh_autostitch_button()
        self._refresh_align_trace_button()
        if self._feed_was_on and self.btn_feed.isChecked():
            self._feed.resume()

    # ── Save ──────────────────────────────────────────────────────────────────
    def _set_export_fmt(self, key):
        """Make `key` the active export format, without writing anything.

        Split out of _export_as so a restored format can be applied at startup:
        _export_as goes on to announce 'no scan to save yet', which would land
        on the status bar over the 'no hardware connected' banner.
        """
        _label, suffix, _filt, _writer = EXPORT_FORMATS[key]   # bad key: KeyError
        self._export_fmt = key
        self._export_actions[key].setChecked(True)
        self.btn_save.setText(f"Save ({suffix})")

    def _export_as(self, key):
        """Make `key` the active export format and write the last scan in it.
        With no scan yet this only switches the format."""
        self._set_export_fmt(key)
        _label, suffix, filt, writer = EXPORT_FORMATS[key]
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
        self.btn_export.setEnabled(True)
        self.btn_save.setEnabled(self.result is not None)

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

            ("autoscale_x",     lambda: bool(c.autoscale_x),
                                lambda v: g.chk_auto_x.setChecked(_as_bool(v))),
            ("autoscale_y",     lambda: bool(c.autoscale_y),
                                lambda v: g.chk_auto_y.setChecked(_as_bool(v))),
            ("autoscale_trace", lambda: bool(c.autoscale_trace),
                                lambda v: g.chk_auto_trace.setChecked(_as_bool(v))),
            ("autoscale_ac_x",  lambda: bool(c.autoscale_ac_x),
                                lambda v: setattr(c, "autoscale_ac_x", _as_bool(v))),
            ("autoscale_ac_y",  lambda: bool(c.autoscale_ac_y),
                                lambda v: setattr(c, "autoscale_ac_y", _as_bool(v))),

            ("spec_xlim",  lambda: [g.spin_xmin.value(), g.spin_xmax.value()],
                           lambda v: _restore_pair(g.spin_xmin, g.spin_xmax, v)),
            ("spec_ylim",  lambda: [g.spin_ymin.value(), g.spin_ymax.value()],
                           lambda v: _restore_pair(g.spin_ymin, g.spin_ymax, v)),
            ("trace_xlim", lambda: [g.spin_tmin.value(), g.spin_tmax.value()],
                           lambda v: _restore_pair(g.spin_tmin, g.spin_tmax, v)),
            ("trace_ylim", lambda: [g.spin_twmin.value(), g.spin_twmax.value()],
                           lambda v: _restore_pair(g.spin_twmin, g.spin_twmax, v)),
            # The autocorrelation has no dialog row, so a frozen view has to be
            # carried by the limits themselves — see set_ac_xlim.
            ("ac_xlim",    lambda: [float(x) for x in c.ax_ac.get_xlim()],
                           lambda v: c.set_ac_xlim(float(v[0]), float(v[1]))),
            ("ac_ylim",    lambda: [float(y) for y in c.ax_ac.get_ylim()],
                           lambda v: c.set_ac_ylim(float(v[0]), float(v[1]))),

            ("log_scale",  g.chk_log.isChecked,
                           lambda v: g.chk_log.setChecked(_as_bool(v))),
            # setCurrentText is a silent no-op on a non-editable combo for a name
            # that is not in the list, so a colormap dropped from
            # TRACE_COLORMAPS needs no guard of its own.
            ("cmap",          lambda: c._cmap_name, g.cmb_cmap.setCurrentText),
            ("cmap_reversed", lambda: bool(c._cmap_rev),
                              lambda v: g.chk_cmap_rev.setChecked(_as_bool(v))),
            # Percent, as typed — the canvas keeps the fraction.
            ("trace_threshold_pct", g.spin_thresh.value,
                                    lambda v: g.spin_thresh.setValue(float(v))),
            ("line_width",    lambda: float(c._lw),
                              lambda v: g.spin_lw.setValue(float(v))),
            ("export_format", lambda: self._export_fmt, self._set_export_fmt),

            ("avg_per_point",      d.spin_avg.value,
                                   lambda v: d.spin_avg.setValue(int(v))),
            ("idle_shots",         d.spin_idle.value,
                                   lambda v: d.spin_idle.setValue(int(v))),
            ("wait_after_move_ms", d.spin_wait.value,
                                   lambda v: d.spin_wait.setValue(int(v))),
            ("saturation_pct",     d.spin_sat.value,
                                   lambda v: d.spin_sat.setValue(float(v))),
            ("abort_on_saturation",  d.chk_abort_sat.isChecked,
                                     lambda v: d.chk_abort_sat.setChecked(_as_bool(v))),
            ("abort_on_stage_fault", d.chk_abort_stage_fault.isChecked,
                                     lambda v: d.chk_abort_stage_fault.setChecked(_as_bool(v))),

            ("scan_start_fs",   self.spin_start.value,
                                lambda v: self.spin_start.setValue(float(v))),
            ("scan_stop_fs",    self.spin_stop.value,
                                lambda v: self.spin_stop.setValue(float(v))),
            ("scan_step_fs",    self.spin_step_fs.value,
                                lambda v: self.spin_step_fs.setValue(float(v))),
            ("scan_background", self.chk_bg.isChecked,
                                lambda v: self.chk_bg.setChecked(_as_bool(v))),
            ("align_step_fs",   self.spin_align_step.value,
                                lambda v: self.spin_align_step.setValue(float(v))),
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
