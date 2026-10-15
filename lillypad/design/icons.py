"""Icon files on disk, and the application icon.

Every path here is resolved through lillypad.paths.resource_path, so the same
constants work from source and from the frozen bundle.

The long comments are the iconographic conventions, and they are not decoration:
three different rules are in play about what an icon depicts, and mixing them up
produces a button that says the opposite of what it does.

  * A two-state PAIR shows the view a click GIVES you (SPLIT/MERGE, HORIZ/VERT,
    SHOW/HIDE).
  * A padlock shows the CURRENT state (UNLOCKED/LOCKED), because a padlock reads
    as a condition rather than as an instruction.
  * A mode or an action with no second state has ONE mark in two files, differing
    only in the theme accent it is drawn in (ALIGN, STITCH, RESCALE, REFRESH,
    PICTURE).
"""

from __future__ import annotations

from PySide6.QtGui import QIcon

from lillypad.paths import resource_path


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
# Save the plots as a PNG, on the main toolbar between the theme and layout
# toggles. An ACTION with no second state to depict, so it follows the
# ALIGN_ICON/RESCALE_ICON convention: one mark, one file per theme's accent.
PICTURE_ICON = {"dark":  resource_path("icons", "save_picture_dark.png"),
                "light": resource_path("icons", "save_picture_light.png")}
# One-shot auto-fit, on the spectrum panel's header. An ACTION, and the mark is
# the same in both files — only the accent differs — so it follows the
# ALIGN_ICON/STITCH_ICON convention rather than the two-state pairs'.
RESCALE_ICON = {"dark":  resource_path("icons", "rescale_dark.png"),
                "light": resource_path("icons", "rescale_light.png")}
# Spectrum-panel view toggle, keyed by theme (the suffix names the theme each
# icon was drawn for, not what it depicts). The icons advertise the ACTION:
# the broken one means "split the pair apart", the continuous one "put it back
# together" — so the button shows the view you get by clicking it.
SPLIT_ICON = {"dark":  resource_path("icons", "broken_spectrum_dark.png"),
              "light": resource_path("icons", "broken_spectrum_light.png")}
MERGE_ICON = {"dark":  resource_path("icons", "continuous_spectrum_dark.png"),
              "light": resource_path("icons", "continuous_spectrum_light.png")}
# Alignment-mode sweep, on the spectrum panel's header. A crosshair: unlike the
# pair above this one depicts a MODE rather than an action, so both files
# show the same mark and differ only in the accent each theme draws it in.
ALIGN_ICON = {"dark":  resource_path("icons", "alignment_dark.png"),
              "light": resource_path("icons", "alignment_light.png")}
# Auto-stitch, on the spectrum panel's header. Depicts the ACTION — matching
# the two halves of a pair — so like ALIGN_ICON both files carry the same mark
# in their own theme's accent.
STITCH_ICON = {"dark":  resource_path("icons", "autostitch_dark.png"),
               "light": resource_path("icons", "autostitch_light.png")}
# Continuous y auto-scale, beside the spectrum's title. A padlock showing the
# CURRENT state, not the action a click performs: an axis free to follow the
# data is UNLOCKED, one frozen where it sits is LOCKED. That is the opposite
# convention to SPLIT_ICON/MERGE_ICON above, and deliberately so — a padlock is
# read as a condition ("this is locked"), not as an instruction, so showing the
# lock you would GET by clicking would say exactly the wrong thing.
UNLOCKED_ICON = {"dark":  resource_path("icons", "unlocked_dark.png"),
                 "light": resource_path("icons", "unlocked_light.png")}
LOCKED_ICON   = {"dark":  resource_path("icons", "locked_dark.png"),
                 "light": resource_path("icons", "locked_light.png")}
# Re-run the alignment sweep, beside the mode toggle that opened the panel.
# An ACTION, but one with no second state to depict, so it follows the
# mode-icon convention: one mark, one file per theme's accent.
REFRESH_ICON = {"dark":  resource_path("icons", "refresh_dark.png"),
                "light": resource_path("icons", "refresh_light.png")}
# Frozen-reference visibility, beside the + that created it. A two-state PAIR,
# so it follows SPLIT_ICON/MERGE_ICON rather than the padlock: each icon shows
# the view a click GIVES you — the open eye puts the reference back, the struck
# -through one takes it away. (The + itself is a generated glyph, not a file;
# see _glyph_icon.)
SHOW_ICON = {"dark":  resource_path("icons", "show_dark.png"),
             "light": resource_path("icons", "show_light.png")}
HIDE_ICON = {"dark":  resource_path("icons", "hide_dark.png"),
             "light": resource_path("icons", "hide_light.png")}
# Symmetry mode, on the FROG trace panel's header. ONE file for both themes —
# the mark is a mirror line, which reads the same either way.
SYMMETRY_ICON = resource_path("icons", "symmetry.png")
# Watermark behind the "nothing connected" message. ONE file for both themes,
# unlike the pairs above: it is drawn as an alpha mask and tinted from the live
# PALETTE (see NoDeviceOverlay), so it follows a theme switch by itself.
DISCONNECTED_ICON = resource_path("icons", "disconnected.png")

_APP_ICON = None


def app_icon() -> QIcon:
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
