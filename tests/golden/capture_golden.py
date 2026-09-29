"""Capture the golden simulator baseline.

Run BEFORE a refactor that touches the simulation physics, the scan orchestration
or the file writers; `tests/test_golden_sim.py` regenerates the same artifacts
afterwards and asserts they are identical.

    python tests/golden/capture_golden.py

Determinism comes from seeding numpy's GLOBAL random state: SimulatedSpectrometer
draws its shot and read noise with np.random.poisson / np.random.normal, so a
fixed seed pins not only the values but the ORDER AND NUMBER of draws. That makes
the comparison stricter than a numerical one — a refactor that acquires the same
spectra in a different sequence, or acquires one extra frame somewhere, shows up
here too. That is a feature, not a nuisance: in a scan loop, an extra frame is a
real behaviour change.

Regenerating these files is therefore a deliberate act. If a change makes them
differ, the question to answer first is "should it have?", not "how do I update
the baseline".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware import PULSE_SHAPES, SimulatedSpectrometer, SimulatedStage
from scan import FrogScanConfig, FrogScanWorker, write_npz

HERE = Path(__file__).resolve().parent

SEED = 20260929

# Small but not degenerate: enough delay points for the trace to have structure,
# few enough pixels to keep the baseline around 100 kB in git.
N_PIXELS = 512
SCAN = dict(delay_start_fs=-60.0, delay_stop_fs=60.0, delay_step_fs=6.0,
            zero_pos_um=150000.0, capture_background=True)

# One acquired column per beam, all at the same delay, so the whole beam library
# is pinned in one small file rather than one scan per beam.
COLUMN_DELAY_MM = 150.0


def capture_scan(path: Path) -> None:
    """A full scan through the real worker, written by the real writer — so this
    one file pins the physics, the scan orchestration and the .npz format."""
    np.random.seed(SEED)
    stage = SimulatedStage(travel_mm=300.0)
    spec = SimulatedSpectrometer(stage, gate="shg", pulse="tl_sech",
                                 n_pixels=N_PIXELS)
    cfg = FrogScanConfig(**SCAN)
    worker = FrogScanWorker(stage, spec, cfg)
    out = {}
    worker.finished_scan.connect(lambda r: out.__setitem__("r", r))
    worker.error.connect(lambda m: out.__setitem__("err", m))
    worker.run()
    if "err" in out:
        raise SystemExit(f"scan failed: {out['err']}")
    write_npz(path, out["r"])
    print(f"  {path.name}: trace {out['r'].trace.shape}, "
          f"{path.stat().st_size / 1024:.0f} kB")


def capture_columns(path: Path) -> None:
    """One column per beam. Broad coverage of the pulse library for very little
    disk: every builder, the bird continuum, the misalignment efficiency and the
    saturation clip all run."""
    data = {}
    for key in PULSE_SHAPES:
        np.random.seed(SEED)
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg", pulse=key,
                                     n_pixels=N_PIXELS)
        stage.move_to(COLUMN_DELAY_MM)
        data[f"{key}__counts"] = spec.acquire()
        data[f"{key}__wl"] = np.asarray(spec.wavelengths, float)
    np.savez_compressed(path, **data)
    print(f"  {path.name}: {len(PULSE_SHAPES)} beams, "
          f"{path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    print(f"capturing golden baseline (seed {SEED})")
    capture_scan(HERE / "sim_scan.npz")
    capture_columns(HERE / "pulse_columns.npz")
    print("done")
