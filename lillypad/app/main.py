"""Starting the program.

Six steps, and the ORDER is the whole content of this file:

  1. the AppUserModelID, before any window exists, or the taskbar groups us
     under the Python launcher and shows its icon;
  2. the QApplication, its name, icon and base font;
  3. **the theme, before the first widget** — PALETTE is sampled at
     construction time by the figure and by the per-theme icon files build_ui
     picks, so a theme applied afterwards would mean restyling the whole
     application at startup and trusting every widget that read a colour on the
     way up to have a refresh hook;
  4. seeding calibration_files/ beside the program, so the first run has the
     shipped calibrations to offer;
  5. the window, shown where the last session left it;
  6. gc.freeze(), which moves the permanent startup graph out of the
     collector's reach for good.

`load_settings()` is cached, so the theme read here and the one FrogWindow reads
come from the same dict and cannot disagree.
"""

import gc
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from lillypad.design.icons import app_icon
from lillypad.design.theme import install_theme, startup_theme
from lillypad.gui.window.main_window import FrogWindow
from lillypad.paths import seed_calibration_dir
from lillypad.settings import load_settings


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
    # time by the figure and by the per-theme icon files build_ui picks, so a
    # theme restored after the window was built would mean restyling the whole
    # application at startup and trusting every widget that read a colour on the
    # way up to have a refresh hook. AppearanceController still owns the runtime
    # switch, and shares install_theme with this. load_settings() is cached, so
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
