"""Reading and writing settings.json.

Display preferences and acquisition policy only — never anything bound to a
particular piece of hardware. A serial number, an integration time or a stage
zero describes the rig that happened to be plugged in last time, and restoring
one silently would put a number the operator never entered into the next
measurement. What is kept here is only how the program LOOKS and how it has been
told to behave. (The full-scale override counts as policy: it is what the
saturation threshold is a percentage of.)

A plain JSON file rather than QSettings and the registry, for the same reason the
calibration files are a folder: the operator can open it, read it, correct it by
hand and delete it.

No Qt here — main() needs the theme out of this file before the QApplication is
styled. (The one settings-adjacent helper that IS Qt-dependent, the
monitor-layout check behind window-geometry restore, stays with the window.)
"""

from __future__ import annotations

import copy
import json
import os

from lillypad.paths import SETTINGS_PATH

SETTINGS_VERSION = 1

# The program's defaults: what a fresh install, or a deleted settings.json,
# starts with. They go through exactly the same restore path as a saved value,
# so they reach each widget — and everything that widget drives, such as the
# live saturation threshold — the way a remembered setting does. That is why
# they live here rather than in the widget constructors: a constructor value
# never fires the change signal that keeps the rest of the program in step.
# A key missing here falls back to the value its widget is built with.
DEFAULT_SETTINGS = {
    "theme": "dark",
    "window": {"x": 0, "y": 0, "w": 1180, "h": 760, "maximized": False},
    "spec_frac": 0.44,
    "log_scale": False,
    "cmap": "jet",
    "trace_smooth": True,
    "cmap_reversed": False,
    "trace_threshold_pct": 0.0,
    "line_width": 1.0,
    "export_format": "dwc",
    "export_name": "FROG_trace",
    "export_date_prefix": False,
    "export_date_delimiter": "-",
    "export_trim_empty_rows": False,
    "avg_per_point": 1,
    "idle_shots": 0,
    "wait_after_move_ms": 250,
    "saturation_pct": 85.0,
    "abort_on_saturation": False,
    "abort_on_stage_fault": False,
    "scan_start_fs": -100.0,
    "scan_stop_fs": 100.0,
    "scan_step_fs": 1.0,
    "scan_delay_reference": False,
    "align_step_fs": 10.0,
    "processing": "lillypad",
}

_SETTINGS = None


def load_settings() -> dict:
    """The settings in force: DEFAULT_SETTINGS, overlaid with whatever
    settings.json holds. Read once per process.

    Cached because two moments need the same answer and must not disagree:
    main() needs the theme before the QApplication is styled, and the window
    needs everything else after its widgets exist.

    A missing file is the ordinary first-run case, and a corrupt one is treated
    exactly the same way — the defaults alone, because defaults are always a
    working program, and a startup that refused to start because a preferences
    file lost a brace would be a far worse failure than a forgotten colormap.
    """
    global _SETTINGS
    if _SETTINGS is None:
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, UnicodeDecodeError):
            data = {}
        # A deep copy: "window" is a nested dict, and the defaults must come
        # out of every load exactly as written above.
        merged = copy.deepcopy(DEFAULT_SETTINGS)
        merged.update(data if isinstance(data, dict) else {})
        _SETTINGS = merged
    return _SETTINGS


def save_settings(data) -> bool:
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


def as_bool(v) -> bool:
    """Strict bool for restore.

    bool("false") is True, so a hand-edited file would otherwise mean the exact
    opposite of what it says. Anything that is not a real JSON boolean raises,
    and that one key falls back to its default.
    """
    if isinstance(v, bool):
        return v
    raise TypeError(v)


def _reset_cache() -> None:
    """Drop the process-lifetime cache. For tests only — the application reads
    settings.json exactly once, and that is the point of the cache."""
    global _SETTINGS
    _SETTINGS = None
