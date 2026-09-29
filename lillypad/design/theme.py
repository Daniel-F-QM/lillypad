"""Installing a theme: the palette, the Qt palette, and the stylesheet.

One entry point, `install_theme`, shared by startup and the runtime toggle. They
used to carry the same four lines twice, and a startup that styled itself even
slightly differently from a toggle is exactly the drift that surfaces later as
one stale widget colour.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

from lillypad.design.stylesheet import build_stylesheet
from lillypad.design.tokens import DARK_PALETTE, LIGHT_PALETTE, PALETTE

THEMES = ("dark", "light")


def startup_theme(data) -> str:
    """The theme to build the whole program in, validated.

    Read straight from the settings dict instead of being restored through
    _apply_theme like every other setting: PALETTE is sampled at CONSTRUCTION
    time by the figure and by _build_ui's per-theme icon files, so the palette
    has to be right before the first widget exists. Restoring it afterwards
    would mean restyling the entire application at startup and trusting that
    every widget which read a colour on the way up has a refresh hook.
    """
    name = data.get("theme")
    return name if name in THEMES else "dark"


def apply_app_palette(app, pal) -> None:
    qpal = QPalette()
    qpal.setColor(QPalette.Window, QColor(pal["bg"]))
    qpal.setColor(QPalette.WindowText, QColor(pal["text"]))
    qpal.setColor(QPalette.Base, QColor(pal["surface"]))
    qpal.setColor(QPalette.Text, QColor(pal["text"]))
    qpal.setColor(QPalette.Button, QColor(pal["surface"]))
    qpal.setColor(QPalette.ButtonText, QColor(pal["text"]))
    app.setPalette(qpal)


def install_theme(app, name) -> None:
    """Point the module-level PALETTE at `name` and restyle the application.

    PALETTE is CLEARED AND UPDATED, never rebound: every runtime `PALETTE[...]`
    lookup in the program — including the ones inside matplotlib artists that
    were constructed under the old theme — shares this one dict object, and
    rebinding the name here would leave all of them on the old colours.
    """
    PALETTE.clear()
    PALETTE.update(LIGHT_PALETTE if name == "light" else DARK_PALETTE)
    apply_app_palette(app, PALETTE)
    app.setStyleSheet(build_stylesheet(PALETTE, name))
