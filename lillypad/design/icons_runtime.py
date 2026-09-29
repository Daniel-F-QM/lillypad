"""Icons painted at runtime rather than shipped as files.

Three of the interface's marks cannot be PNGs on disk:

  * the spinbox arrows and the checkbox tick, because QSS `url()` will not load a
    data URI — so they are written to real files in %TEMP% and referenced from
    there, one pair per theme tag;
  * the play/stop/plus glyphs, because those buttons re-colour themselves twice
    over, once for their state (the feed's accent while stopped, danger while
    running) and once for the theme. A file can do neither, and the files it
    would take would still be wrong the moment either palette changed.

All three need a running QApplication.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication

from lillypad.design.geometry import HDR_ICON


def _icon_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "lillypad_icons"
    d.mkdir(exist_ok=True)
    return d


def make_arrow_icons(color, tag) -> tuple[str, str]:
    """Render spinbox up/down arrows to PNG files (QSS url() cannot load data
    URIs, which is why the arrows were missing). Needs a running QApplication.
    Returns (up_path, down_path) with forward slashes for use in QSS."""
    icon_dir = _icon_dir()
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


def make_check_icon(color, tag) -> str:
    """Render a checkmark PNG for the checked checkbox indicator (same QSS
    url() limitation as the arrows). Needs a running QApplication."""
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
    fp = _icon_dir() / f"check_{tag}.png"
    pm.save(str(fp))
    return fp.as_posix()


def glyph_icon(kind, color, px=HDR_ICON) -> QIcon:
    """A play / stop / plus glyph as a QIcon, drawn in `color`.

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
    elif kind == "plus":
        # Two filled bars, not a stroked cross: the same NoPen/brush path as the
        # other two marks, so all three land on the pixel grid the same way at
        # any device ratio. Arms nearly the full icon, like the glyphs in the
        # PNG buttons beside it — a small plus in a 24 px button reads as a
        # decoration rather than as the control.
        arm, th = 0.78 * n, 0.16 * n
        r = th / 2.0
        p.drawRoundedRect(QRectF(c - arm / 2, c - th / 2, arm, th), r, r)
        p.drawRoundedRect(QRectF(c - th / 2, c - arm / 2, th, arm), r, r)
    else:
        s = 0.56 * n
        p.drawRoundedRect(QRectF(c - s / 2, c - s / 2, s, s), 0.10 * n, 0.10 * n)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return QIcon(pm)
