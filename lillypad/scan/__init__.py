"""The scan engine: optics conversions, the data model, the writers, the worker.

    core     delay<->position, the delay grid, autocorrelation, FWHM. numpy only.
    model    FrogScanConfig and FrogResult
    writers  .dwc / .npz / .csv
    worker   FrogScanWorker(QThread) — the only Qt in this package

The split exists so the first three are reachable WITHOUT Qt installed. That was
the one boundary the old scan.py deliberately got wrong, and said so: its
`from PySide6.QtCore import QThread` sat at module scope, so importing a unit
conversion meant importing a GUI toolkit.

Which is why FrogScanWorker is deliberately NOT re-exported here. Importing any
submodule runs this file first, so one `from .worker import ...` on the line
below would pull Qt into `import lillypad.scan.core` and undo the whole point of
the split. The worker is imported from lillypad.scan.worker by name, which also
makes every Qt-dependent caller visible to a grep. A test asserts the core stays
clean.
"""

from lillypad.scan.core import (autocorrelation, build_delay_grid,
                                delay_to_position_um, fwhm,
                                position_to_delay_fs, stage_to_um, um_to_stage)
from lillypad.scan.model import FrogResult, FrogScanConfig
from lillypad.scan.writers import write_csv, write_dwc, write_npz

__all__ = [
    "um_to_stage", "stage_to_um",
    "delay_to_position_um", "position_to_delay_fs", "build_delay_grid",
    "autocorrelation", "fwhm",
    "FrogScanConfig", "FrogResult",
    "write_dwc", "write_npz", "write_csv",
]
