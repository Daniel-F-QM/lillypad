"""A full FROG scan driven end to end by simulated hardware, with no window and
no event loop.

Ported from scan.py's self-test (lines 789-850), extended with a clean-scan case
the self-test only exercised implicitly.

`FrogScanWorker` is a QThread, but calling `.run()` DIRECTLY (never `.start()`)
executes the scan on the calling thread, and Signal.connect/emit between objects
on one thread are direct connections that fire synchronously. So the whole scan
orchestration — travel guard, background brackets, the sweep, saturation policy,
stage-fault policy, park-at-zero — is testable without a QApplication. The
existing self-test already relied on this.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lillypad.hardware.base import StageLimitError
from lillypad.sim import SimulatedSpectrometer, SimulatedStage
from lillypad.scan import FrogScanConfig, stage_to_um
from lillypad.scan.worker import FrogScanWorker


@pytest.fixture
def small_cfg():
    """Seven columns, no background frames — enough to exercise every branch
    without paying for a real trace."""
    return FrogScanConfig(delay_start_fs=-30, delay_stop_fs=30,
                          delay_step_fs=10.0, zero_pos_um=150000.0,
                          capture_delay_reference=False)


def run_scan(stage, spec, cfg, dark=None):
    """Run a scan synchronously and collect everything it emitted."""
    worker = FrogScanWorker(stage, spec, cfg, dark=dark)
    out = {"result": None, "error": None, "faults": [], "columns": [],
           "references": [], "saturations": [], "progress": []}
    worker.finished_scan.connect(lambda r: out.__setitem__("result", r))
    worker.error.connect(lambda m: out.__setitem__("error", m))
    worker.stage_fault.connect(
        lambda i, delay, why: out["faults"].append((i, why)))
    worker.column_ready.connect(
        lambda i, *rest: out["columns"].append(i))
    worker.delay_reference_ready.connect(
        lambda which, frame: out["references"].append(which))
    worker.saturation_warning.connect(
        lambda i, *rest: out["saturations"].append(i))
    worker.progress.connect(lambda i, n: out["progress"].append((i, n)))
    worker.run()
    return out


class _FaultingStage(SimulatedStage):
    """Misses its target on one column and says so, without raising."""

    def __init__(self, fault_at_um, **kw):
        super().__init__(**kw)
        self._fault_at = fault_at_um
        self._fault = None

    def _move_to_raw(self, position_mm):
        self._fault = None
        if abs(stage_to_um(position_mm) - self._fault_at) < 1e-6:
            super()._move_to_raw(position_mm - 0.002)     # 2 um short
            self._fault = "stage did not reach the commanded position"
            return
        super()._move_to_raw(position_mm)

    def position_fault(self):
        return self._fault


class TestCleanScan:
    def test_produces_a_trace_of_the_expected_shape(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        out = run_scan(stage, spec, small_cfg)
        assert out["error"] is None
        r = out["result"]
        assert r is not None
        assert r.trace.shape == (spec.n_pixels, small_cfg.delays_fs().size)
        assert np.all(np.isfinite(r.trace))

    def test_emits_one_column_per_delay_point(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        out = run_scan(stage, spec, small_cfg)
        n_pts = small_cfg.delays_fs().size
        assert out["columns"] == list(range(n_pts))
        assert out["progress"][-1] == (n_pts, n_pts)

    def test_axes_are_consistent_with_the_config(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        r = run_scan(stage, spec, small_cfg)["result"]
        assert np.allclose(r.delays_fs, small_cfg.delays_fs())
        assert np.allclose(r.positions_cmd_um, small_cfg.positions_um())
        # Readback tracks the commands closely on a stage with no slack.
        assert np.allclose(r.positions_readback_um, r.positions_cmd_um, atol=1.0)

    def test_records_no_faults(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        r = run_scan(stage, spec, small_cfg)["result"]
        assert not r.has_stage_fault()

    def test_parks_back_at_zero_delay(self, small_cfg):
        """The `finally` park: the operator should find the stage where they
        marked zero, not at the red end of the last sweep."""
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        run_scan(stage, spec, small_cfg)
        assert stage_to_um(stage.get_position()) == pytest.approx(
            small_cfg.zero_pos_um, abs=1.0)

    def test_captures_both_delay_references_when_asked(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        cfg = replace(small_cfg, capture_delay_reference=True)
        out = run_scan(stage, spec, cfg)
        r = out["result"]
        assert r.delay_reference_before is not None
        assert r.delay_reference_after is not None
        assert len(out["references"]) == 2
        assert r.delay_reference().size == spec.n_pixels

    def test_no_delay_reference_by_default(self, small_cfg):
        """It costs two acquisitions and is diagnostic only, so it is opt-in."""
        assert FrogScanConfig.capture_delay_reference is False
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        out = run_scan(stage, spec, small_cfg)
        assert out["references"] == []
        assert out["result"].delay_reference() is None

    def test_abort_stops_the_scan(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        worker = FrogScanWorker(stage, spec, small_cfg)
        got = {"result": None}
        worker.finished_scan.connect(lambda r: got.__setitem__("result", r))
        # Abort after the second column, from the signal itself.
        worker.column_ready.connect(
            lambda i, *rest: worker.abort() if i >= 1 else None)
        worker.run()
        assert got["result"] is None


class TestStageFaultPolicy:
    """A stage that reports a bad move through position_fault() instead of
    raising must not cost the operator the whole scan. This is the path a Piezo
    Jena takes after its retries are exhausted."""

    @pytest.fixture
    def faulting(self, small_cfg):
        targets = small_cfg.positions_um()

        def build(abort_on_fault):
            stage = _FaultingStage(targets[2], travel_mm=300.0)
            spec = SimulatedSpectrometer(stage, gate="shg")
            cfg = replace(small_cfg, abort_on_stage_fault=abort_on_fault)
            return run_scan(stage, spec, cfg), targets
        return build

    def test_abort_off_keeps_the_whole_trace(self, faulting, small_cfg):
        out, _ = faulting(False)
        assert out["result"] is not None, "the scan must finish and keep its data"
        assert out["result"].trace.shape[1] == small_cfg.delays_fs().size

    def test_abort_off_flags_the_offending_column(self, faulting):
        out, _ = faulting(False)
        assert len(out["faults"]) == 1
        assert out["faults"][0][0] == 2
        assert out["result"].stage_faults[2]
        assert list(out["result"].faulted_columns()) == [2]

    def test_abort_off_labels_the_column_with_where_the_stage_really_was(
            self, faulting):
        """Not with the command. An exception, by contrast, unwinds past run()
        and every measured column is lost."""
        out, targets = faulting(False)
        gap = abs(out["result"].positions_readback_um[2] - targets[2])
        assert gap > 1.0

    def test_abort_on_stops_the_scan(self, faulting):
        out, _ = faulting(True)
        assert out["result"] is None
        assert out["error"] and "Stage fault" in out["error"]
        assert len(out["faults"]) == 1


class TestSaturationPolicy:
    """Saturation is an alert by default, unlike a stage fault: a clipped
    spectrum is still labelled with the right delay."""

    @pytest.fixture
    def saturating(self, small_cfg):
        def build(abort_on_saturation):
            stage = SimulatedStage(travel_mm=300.0)
            spec = SimulatedSpectrometer(stage, gate="shg")
            # A full scale just above the dark floor makes every column clip.
            cfg = replace(small_cfg, saturation_counts=1.0,
                          saturation_fraction=0.01,
                          abort_on_saturation=abort_on_saturation)
            return run_scan(stage, spec, cfg)
        return build

    def test_alert_only_by_default(self, saturating, small_cfg):
        out = saturating(False)
        assert out["result"] is not None
        assert out["result"].trace.shape[1] == small_cfg.delays_fs().size
        assert out["saturations"], "a clipped scan must warn"
        assert out["result"].is_saturated()
        assert out["result"].saturation_checked

    def test_abort_on_saturation_stops_the_scan(self, saturating):
        out = saturating(True)
        assert out["result"] is None
        assert out["error"]

    def test_an_unclipped_scan_reports_checked_but_clean(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        cfg = replace(small_cfg, saturation_counts=1e9)
        r = run_scan(stage, spec, cfg)["result"]
        assert r.saturation_checked
        assert not r.is_saturated()


class TestTravelGuard:
    def test_a_scan_past_the_travel_limits_is_refused_before_moving(self):
        """Pre-flight, not mid-sweep: the operator gets told the range does not
        fit rather than losing half a scan to a limit stop."""
        cfg = FrogScanConfig(delay_start_fs=-1e7, delay_stop_fs=1e7,
                             delay_step_fs=1e6, zero_pos_um=150000.0,
                             capture_delay_reference=False)
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        start = stage.get_position()
        out = run_scan(stage, spec, cfg)
        assert out["result"] is None
        assert out["error"]
        assert out["columns"] == [], "nothing should have been measured"
        assert stage.get_position() == start, "the stage should not have moved"


class _EndStopStage(SimulatedStage):
    """An unhomed stage whose forward switch sits at `switch_mm` in the
    power-up frame: a move past it stops there and raises, as
    KinesisStage._move_to_raw does."""

    def __init__(self, switch_mm, **kw):
        super().__init__(**kw)
        self.switch_mm = switch_mm
        self.needs_homing = True
        self.hw_limits_stop = True

    def _move_to_raw(self, position_mm):
        if position_mm > self.switch_mm:
            super()._move_to_raw(self.switch_mm)
            raise StageLimitError(
                f"{self.name}: stopped by the forward limit switch at "
                f"{self.switch_mm:.4f} mm.")
        super()._move_to_raw(position_mm)


class TestUnhomedScan:
    """An unhomed axis scans in its power-up frame, and a scan that runs into
    an end stops there with a stage fault instead of carrying on."""

    @staticmethod
    def _cfg_at_power_up_zero(small_cfg):
        # Zero marked where the stage powered up: half the sweep sits below 0.
        return replace(small_cfg, zero_pos_um=0.0)

    def test_a_scan_below_the_power_up_zero_runs(self, small_cfg):
        stage = SimulatedStage(travel_mm=300.0)
        stage.needs_homing = stage.hw_limits_stop = True
        spec = SimulatedSpectrometer(stage, gate="shg")
        out = run_scan(stage, spec, self._cfg_at_power_up_zero(small_cfg))
        assert out["error"] is None
        assert out["result"] is not None
        assert out["result"].positions_cmd_um.min() < 0.0

    def test_the_same_scan_homed_is_refused_before_moving(self, small_cfg):
        """Homed, the frame is real and 0 is the bottom of travel."""
        stage = SimulatedStage(travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        out = run_scan(stage, spec, self._cfg_at_power_up_zero(small_cfg))
        assert out["result"] is None
        assert out["columns"] == []

    def test_reaching_an_end_stops_the_scan_with_a_fault(self, small_cfg):
        cfg = self._cfg_at_power_up_zero(small_cfg)
        targets = cfg.positions_um()
        # Between columns 4 and 5, so five columns land and the sixth is cut.
        switch_um = 0.5 * (targets[4] + targets[5])
        stage = _EndStopStage(switch_um / 1000.0, travel_mm=300.0)
        spec = SimulatedSpectrometer(stage, gate="shg")
        out = run_scan(stage, spec, cfg)
        assert out["result"] is None
        assert out["columns"] == [0, 1, 2, 3, 4]
        assert out["error"] and "end of its travel" in out["error"]
        assert [i for i, _ in out["faults"]] == [5]
        # Not parked: it stays on the switch where it stopped.
        assert stage_to_um(stage.get_position()) == pytest.approx(switch_um)
