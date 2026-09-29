"""Small, reusable widgets.

Nothing here knows what Lillypad measures. These are the pieces the panels and
dialogs are built from: value widgets that refuse the mouse wheel, a separator,
a status lamp, and the three custom-painted widgets that live on top of the
figure.

Colour comes from design.tokens.PALETTE, read at PAINT time so a theme switch
reaches widgets that already exist. None of them carries its own stylesheet —
the chrome ones are styled by object name from design/qss/app.qss, and the
painted ones sample the palette directly.
"""

from __future__ import annotations


from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame,
                               QLabel, QLineEdit, QPushButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from lillypad.design.icons import DISCONNECTED_ICON
from lillypad.design.tokens import PALETTE


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

def _hline():
    f = QFrame(); f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
    return f

# Fraction of full scale at which the lamp goes amber. Deliberately well below
# the saturation threshold itself (FrogScanConfig.saturation_fraction, which
# both the scan worker and the live lamp use): most detectors go nonlinear
# before they actually clip, so by the time pixels are pinned at full scale the
# spectrum around them is already wrong.
SAT_WARN_FRACTION = 0.90

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

class AxisBoundEditor(QLineEdit):
    """The box that appears over an axis' end tick label when it is clicked.

    A Qt child of the canvas rather than a dialog: the value being changed is
    the number under the cursor, and a modal box for one float would put a
    title bar and two buttons in front of the plot it is about. Same reasoning
    as the rubber band and the split handle — it composites above the canvas
    and never lands in the cached blit background.

    Commits on Return and cancels on Escape or on losing focus. Cancelling on
    focus-out (rather than committing) is deliberate: clicking away from a
    half-typed number is how you abandon it, and the axis limits are not
    somewhere a stray keystroke should be able to land unnoticed.
    """
    PAD = 10          # logical px of slack around the tick label's own width
    MIN_W = 56

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._target = None       # (ax, "x"|"y", "lo"|"hi") while open
        self.setAlignment(Qt.AlignCenter)
        self.hide()
        self.returnPressed.connect(self._commit)
        self.editingFinished.connect(self.cancel)

    def open_on(self, ax, which, end, value, rect):
        """Start editing `end` of `ax`'s `which` axis, over the tick label at
        `rect` (canvas-widget pixels)."""
        self._target = (ax, which, end)
        # %g, not the tick's own label: the label is rounded for display, and
        # seeding the box with a rounded number makes "just nudge the max"
        # silently move the bound the operator did not mean to touch.
        self.setText(f"{value:g}")
        w = max(rect.width() + self.PAD, self.MIN_W)
        self.setGeometry(rect.center().x() - w // 2, rect.top() - 2,
                         w, rect.height() + 6)
        self.show()
        self.raise_()
        self.setFocus(Qt.OtherFocusReason)
        self.selectAll()

    def cancel(self):
        self._target = None
        self.hide()

    def _commit(self):
        if self._target is None:
            return                      # Return after a cancel; nothing to do
        ax, which, end = self._target
        text = self.text().strip()
        # Clear the target BEFORE applying: apply_axis_bound redraws, which can
        # pull focus and re-enter through editingFinished -> cancel().
        self._target = None
        self.hide()
        try:
            value = float(text)
        except ValueError:
            self._canvas.axis_edit_failed.emit(
                f"{text!r} is not a number — axis unchanged.")
            return
        self._canvas.apply_axis_bound(ax, which, end, value)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancel()
            return
        super().keyPressEvent(event)

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
