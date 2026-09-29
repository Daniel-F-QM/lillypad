"""TEMPORARY compatibility shim — scan.py moved into lillypad/scan/.

Kept only so frog_gui_fast.py and zaber_diagnostics.py keep importing as they
did while the refactor continues. Deleted in the step that repoints them.

Re-exports only, never redefinitions — see hardware.py's shim for why that
matters.

`_um_to_stage` and `_stage_to_um` are now public as `um_to_stage` / `stage_to_um`
(they cross a module boundary, so the underscore was claiming something untrue).
The old private names are aliased here because the GUI still imports them by
name; both spellings resolve to the same function object, which matters because
the worker and the GUI must agree on the conversion.

New code should import from lillypad.scan.
"""

from lillypad.scan import (FrogResult, FrogScanConfig,
                           autocorrelation, build_delay_grid,
                           delay_to_position_um, fwhm, position_to_delay_fs,
                           stage_to_um, um_to_stage, write_csv, write_dwc,
                           write_npz)
from lillypad.scan.worker import FrogScanWorker

# Old spellings, same objects.
_um_to_stage = um_to_stage
_stage_to_um = stage_to_um
