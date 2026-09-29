"""Where Lillypad's files live — the one module that resolves a path.

This is deliberately the ONLY place in the program that derives a location from
`__file__`. There used to be four independent anchors (the GUI's resource_path
and app_dir, the Avantes wrapper's DLL search, and hardware.py's self-test), each
counting its own directory levels, so every one of them shifted by one the moment
a file moved into a package. Collapsing them here is what makes the rest of the
refactor safe to do at all.

Two kinds of location, and the difference matters:

  resource_path()  BUNDLED, read-only. Icons, stylesheets, the seed calibration
                   files. Frozen, these live in PyInstaller's _MEIPASS, which is
                   re-extracted on every launch — anything written there
                   silently vanishes.

  app_dir()        BESIDE THE PROGRAM, writable and permanent. The operator's
                   calibration files and settings.json. Frozen, this is the
                   folder holding the .exe, NOT the bundle.

No Qt, no numpy, no vendor SDK: this module must import in a bare interpreter,
because main() needs settings.json before a QApplication exists.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# The repository root when running from source. `paths.py` sits one level down,
# in lillypad/, so the root is parents[1].
#
# Private and used ONLY as the fallback inside the two functions below, both of
# which test for a frozen process first. That ordering is load-bearing: in a
# frozen bundle the package modules live inside the PYZ archive and `__file__`
# is a synthetic path under _MEIPASS, so this would evaluate to the bundle
# directory. Neither function ever reaches it there.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resource_path(*parts) -> Path:
    """Resolve a path to a bundled resource, from source or frozen.

    Frozen, assets live under sys._MEIPASS; from source they sit at the
    repository root. Called with no arguments it returns the base directory
    itself, which is what the Avantes DLL search wants.
    """
    base = Path(getattr(sys, "_MEIPASS", _PROJECT_ROOT))
    return base.joinpath(*parts)


def app_dir() -> Path:
    """Folder the program lives in — next to the .exe when frozen, else the
    repository root.

    User-editable data belongs HERE, not in resource_path()'s _MEIPASS bundle,
    which is read-only and re-extracted on every launch — files added there
    would silently vanish.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return _PROJECT_ROOT


CALIBRATION_DIR = app_dir() / "calibration_files"

# Preferences live beside the program for the same reason the calibration files
# do: app_dir() is writable and permanent, while resource_path()'s _MEIPASS
# bundle is re-extracted every launch. Never seeded, unlike the calibration
# folder — the absence of this file IS the default state, so deleting it is a
# full factory reset.
SETTINGS_PATH = app_dir() / "settings.json"


def seed_calibration_dir() -> None:
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
