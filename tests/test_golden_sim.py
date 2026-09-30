"""The simulator must produce bit-identical output across the refactor.

Regenerates the artifacts in tests/golden/ and compares them to the versions
captured before the package split. Every other test in this suite asserts a
property — that energy is conserved, that a fit recovers a known gain. This one
asserts SAMENESS, which is the only thing that actually proves a pure relocation
was pure.

Determinism comes from seeding numpy's global random state, which the simulator
draws its shot and read noise from. That pins the order and number of draws as
well as the values, so this also catches a refactor that acquires the same
spectra in a different sequence, or slips in one extra frame somewhere.

If these fail, the first question is "should the output have changed?" — not
"how do I regenerate the baseline". Regenerating is
`python tests/golden/capture_golden.py`, and is a deliberate act.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hardware import PULSE_SHAPES, SimulatedSpectrometer, SimulatedStage
from scan import FrogScanConfig, FrogScanWorker, write_npz

from .golden.capture_golden import (COLUMN_DELAY_MM, N_PIXELS, SCAN, SEED,
                                    scan_dark)

GOLDEN = Path(__file__).resolve().parent / "golden"


def _require(name):
    path = GOLDEN / name
    if not path.is_file():
        pytest.skip(f"no golden baseline at {path} — "
                    f"run python tests/golden/capture_golden.py")
    return path


@pytest.fixture(scope="module")
def fresh_scan(tmp_path_factory):
    """Re-run the captured scan exactly as capture_golden.py did."""
    np.random.seed(SEED)
    stage = SimulatedStage(travel_mm=300.0)
    spec = SimulatedSpectrometer(stage, gate="shg", pulse="tl_sech",
                                 n_pixels=N_PIXELS)
    worker = FrogScanWorker(stage, spec, FrogScanConfig(**SCAN),
                            dark=scan_dark())
    out = {}
    worker.finished_scan.connect(lambda r: out.__setitem__("r", r))
    worker.error.connect(lambda m: out.__setitem__("err", m))
    worker.run()
    assert "err" not in out, out.get("err")
    path = tmp_path_factory.mktemp("golden") / "sim_scan.npz"
    write_npz(path, out["r"])
    return path


class TestScanBaseline:
    """One full scan through the real worker and the real writer: this pins the
    physics, the scan orchestration and the .npz format together."""

    @pytest.fixture(scope="class")
    def pair(self, fresh_scan):
        with np.load(_require("sim_scan.npz"), allow_pickle=False) as want, \
                np.load(fresh_scan, allow_pickle=False) as got:
            yield ({k: want[k] for k in want.files},
                   {k: got[k] for k in got.files})

    def test_the_archive_holds_the_same_arrays(self, pair):
        want, got = pair
        assert sorted(want) == sorted(got)

    @pytest.mark.parametrize("key", [
        "trace", "delays_fs", "wavelengths_nm",
        "positions_cmd_um", "positions_readback_um",
        "dark", "delay_reference_before", "delay_reference_after",
    ])
    def test_array_is_bit_identical(self, pair, key):
        want, got = pair
        if key not in want:
            pytest.skip(f"{key} not in the baseline")
        a, b = want[key], got[key]
        assert a.shape == b.shape, f"{key}: {a.shape} != {b.shape}"
        if not np.array_equal(a, b):
            bad = int(np.count_nonzero(a != b))
            worst = float(np.max(np.abs(a - b)))
            pytest.fail(f"{key}: {bad}/{a.size} values differ, "
                        f"largest difference {worst:.6g}")

    def test_metadata_matches_apart_from_the_timestamp(self, pair):
        """The timestamp is the one field that MUST differ — it records when the
        scan ran."""
        import ast
        want, got = pair
        w = ast.literal_eval(str(want["metadata"]))
        g = ast.literal_eval(str(got["metadata"]))
        assert w.pop("timestamp") != g.pop("timestamp")
        assert w == g


class TestPulseLibraryBaseline:
    """One acquired column per beam. Broad coverage of the builders, the bird
    continuum, the misalignment efficiency and the saturation clip."""

    @pytest.fixture(scope="class")
    def baseline(self):
        with np.load(_require("pulse_columns.npz"), allow_pickle=False) as z:
            yield {k: z[k] for k in z.files}

    def test_every_beam_in_the_registry_has_a_baseline(self, baseline):
        """A new beam must be captured, or it is silently untested."""
        missing = [k for k in PULSE_SHAPES if f"{k}__counts" not in baseline]
        assert not missing, (f"no golden column for: {missing} — "
                             f"re-run tests/golden/capture_golden.py")

    @pytest.mark.parametrize("key", sorted(PULSE_SHAPES))
    def test_column_is_bit_identical(self, baseline, key):
        if f"{key}__counts" not in baseline:
            pytest.skip(f"{key} not in the baseline")
        np.random.seed(SEED)
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg", pulse=key,
                                     n_pixels=N_PIXELS)
        stage.move_to(COLUMN_DELAY_MM)
        counts = spec.acquire()
        wl = np.asarray(spec.wavelengths, float)

        assert np.array_equal(wl, baseline[f"{key}__wl"]), \
            f"{key}: the wavelength grid moved"
        want = baseline[f"{key}__counts"]
        if not np.array_equal(counts, want):
            bad = int(np.count_nonzero(counts != want))
            worst = float(np.max(np.abs(counts - want)))
            pytest.fail(f"{key}: {bad}/{want.size} pixels differ, "
                        f"largest difference {worst:.6g}")
