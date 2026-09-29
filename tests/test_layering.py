"""The dependency rules, asserted rather than documented.

Layer boundaries decay quietly: one convenient import at the top of an __init__
and a package that used to be reachable on its own now drags a GUI toolkit in
behind it. Nothing fails, nothing looks different, and the property is gone.

That is not hypothetical — it happened while writing step 4. lillypad/scan/
__init__.py re-exported FrogScanWorker, and because importing any submodule runs
the package __init__ first, `import lillypad.scan.core` pulled in Qt and undid
the whole reason for the split. These tests are what caught it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# A meta-path finder that refuses one top-level package, so the import runs the
# way it would on a machine that does not have it installed at all.
_BLOCK = """
import sys, importlib.abc

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == {pkg!r} or name.startswith({pkg!r} + "."):
            raise ImportError("blocked for this check: " + name)
        return None

sys.meta_path.insert(0, _Block())
{body}
leaked = sorted(m for m in sys.modules if m.split(".")[0] == {pkg!r})
assert not leaked, "leaked: " + ", ".join(leaked)
print("ok")
"""


def _run_isolated(pkg, body):
    """Import `body` in a fresh interpreter with `pkg` blocked."""
    code = _BLOCK.format(pkg=pkg, body=body)
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True)
    return proc


class TestScanCoreNeedsNoQt:
    """The whole point of splitting scan.py. Its own docstring named this as the
    boundary worth making: the conversions and the writers should be usable by
    anything — a retrieval step, a batch reprocessing script, a test — without a
    GUI toolkit installed."""

    @pytest.mark.parametrize("module", [
        "lillypad.scan.core",
        "lillypad.scan.model",
        "lillypad.scan.writers",
        "lillypad.scan",          # the package __init__ itself must stay clean
    ])
    def test_imports_with_pyside6_blocked(self, module):
        proc = _run_isolated("PySide6", f"import {module}")
        assert proc.returncode == 0, (
            f"{module} needs Qt:\n{proc.stderr[-1500:]}")

    def test_and_still_computes(self):
        proc = _run_isolated("PySide6", """
import numpy as np
from lillypad.scan.core import build_delay_grid, fwhm
d = build_delay_grid(-100, 100, 10)
assert d.size == 21
assert abs(fwhm(d, np.exp(-(d**2) / (2 * 40.0**2))) - 94.1) < 5
""")
        assert proc.returncode == 0, proc.stderr[-1500:]

    def test_the_worker_is_the_one_module_that_does_need_qt(self):
        """Stated as an expectation rather than left implicit: if this ever
        passes, the worker stopped being a QThread and the split has a different
        shape than this file claims."""
        proc = _run_isolated("PySide6", "import lillypad.scan.worker")
        assert proc.returncode != 0


class TestDeviceLayerNeedsNoVendorSdk:
    """`import lillypad.hardware` must work with only numpy, or a missing DLL
    stops being one device's problem and becomes the program's."""

    @pytest.mark.parametrize("sdk", ["pylablib", "zaber_motion", "seabreeze",
                                     "serial", "usb"])
    def test_hardware_imports_without(self, sdk):
        proc = _run_isolated(sdk, "import lillypad.hardware")
        assert proc.returncode == 0, (
            f"lillypad.hardware needs {sdk} at import time:\n"
            f"{proc.stderr[-1500:]}")

    @pytest.mark.parametrize("sdk", ["seabreeze", "zaber_motion", "pylablib"])
    def test_spectrometer_enumeration_degrades_quietly_without(self, sdk):
        """list_spectrometers is the one enumeration that swallows a per-vendor
        failure: it feeds the device picker and every multi-spectrometer
        submenu, so one missing driver must not take all of them down."""
        proc = _run_isolated(sdk, """
from lillypad.hardware import list_spectrometers
assert isinstance(list_spectrometers(), list)
""")
        assert proc.returncode == 0, proc.stderr[-1500:]

    def test_stage_enumeration_raises_instead_of_degrading(self):
        """The deliberate asymmetry, pinned so nobody "fixes" it into silence.

        list_kinesis_stages RAISES when pylablib is absent, and should: unlike
        the spectrometer list it is not a background sweep across vendors but an
        explicit "scan for Kinesis stages" the operator asked for, and both call
        sites catch it and put the reason in the dialog. Degrading to an empty
        list would report "no devices found" to someone whose problem is a
        missing driver.
        """
        proc = _run_isolated("pylablib", """
from lillypad.hardware import list_kinesis_stages
try:
    list_kinesis_stages()
except Exception:
    pass
else:
    raise AssertionError("expected an error with pylablib absent")
""")
        assert proc.returncode == 0, proc.stderr[-1500:]

    def test_the_simulator_needs_no_sdk_either(self):
        proc = _run_isolated("pylablib", """
from lillypad.sim import SimulatedSpectrometer, SimulatedStage
s = SimulatedStage()
spec = SimulatedSpectrometer(s, gate="shg")
assert spec.acquire().size == spec.n_pixels
""")
        assert proc.returncode == 0, proc.stderr[-1500:]


class TestDesignLayerNeedsNoHardware:
    """design/ describes how the program looks. It should not be able to reach a
    spectrometer even by accident."""

    def test_tokens_import_without_numpy_or_qt(self):
        proc = _run_isolated("PySide6", "import lillypad.design.tokens")
        assert proc.returncode == 0, proc.stderr[-1500:]

    def test_paths_and_settings_need_no_qt(self):
        proc = _run_isolated("PySide6",
                             "import lillypad.paths, lillypad.settings")
        assert proc.returncode == 0, proc.stderr[-1500:]
