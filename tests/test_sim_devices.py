"""SimulatedStage: the long-move splitter, the backlash approach, and the
zero-shift bug the approach machinery exists to prevent.

Ported from hardware.py's self-test (lines 2644-2661, 2695-2717) and scan.py's
(lines 741-787). The zero-shift check is the highest-value test in the suite: at
double pass, 8 um of lead-screw slack is 53 fs of delay error on every column,
and nothing in the recorded data reveals it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lillypad.hardware import StitchedSpectrometer
from lillypad.sim import SimulatedSpectrometer, SimulatedStage
from lillypad.scan import (FrogScanConfig, delay_to_position_um,
                           position_to_delay_fs, stage_to_um, um_to_stage)

# 8 um — Zaber X-LSM class spec, as used by scan.py's self-test.
SLACK_MM = 0.008


class TestLongMoveSplit:
    """A move longer than max_step_mm (5 mm default) is issued in pieces, so a
    single command can never ask the stage for a huge unmonitored traverse."""

    def test_47mm_becomes_ten_submoves_and_still_lands(self, move_counter):
        stage = SimulatedStage(travel_mm=300.0)
        n = move_counter(stage)
        stage.move_to(47.0)
        assert n[0] == 10, f"expected 10 sub-moves for 47 mm, got {n[0]}"
        assert stage.get_position() == 47.0

    def test_short_move_stays_a_single_move(self, move_counter):
        stage = SimulatedStage(travel_mm=300.0)
        stage.move_to(47.0)
        n = move_counter(stage)
        stage.move_to(45.0)
        assert n[0] == 1, f"expected 1 move for 2 mm, got {n[0]}"


class TestBacklashZeroShift:
    """Marking zero after arriving from ABOVE, then sweeping upward, puts the
    marked zero and the scan in frames that differ by the screw slack. On a
    Zaber nothing in the controller compensates and `pos` reads back the
    commanded value either way, so the offset is completely silent."""

    @staticmethod
    def _zero_shift_fs(correction_mm, delays, pass_factor):
        stage = SimulatedStage(travel_mm=300.0, physical_backlash_mm=SLACK_MM,
                               backlash_mm=correction_mm)
        stage.move_to(160.0)                 # park above
        stage.move_to(150.0)                 # jog DOWN onto zero delay
        zero_um = stage_to_um(stage.get_position())   # "set current position as 0 fs"
        landed = []
        for t in delay_to_position_um(delays, zero_um, pass_factor):
            stage.move_to(um_to_stage(t))   # the scan: monotonic, upward
            landed.append(stage_to_um(stage.get_position()))
        # True zero delay is at 150 mm. Compare the delay each column REALLY
        # sampled against the delay it is labelled with. Column 0 is skipped: it
        # is the one point the sweep arrives at from above, so it carries an
        # extra slack error of its own on top of the systematic shift.
        real = position_to_delay_fs(np.array(landed), 150000.0, pass_factor)
        return float(np.mean(real[1:] - delays[1:]))

    @pytest.fixture
    def delays(self):
        cfg = FrogScanConfig(delay_start_fs=-300, delay_stop_fs=300,
                             delay_step_fs=13.3426, zero_pos_um=150000.0)
        return cfg.delays_fs(), cfg.pass_factor

    def test_uncompensated_shifts_the_whole_axis(self, delays):
        d, pf = delays
        shift = self._zero_shift_fs(0.0, d, pf)
        # The self-test expected ~-53 fs: 8 um of slack at double pass.
        assert shift == pytest.approx(-53.0, abs=2.0)

    def test_compensation_removes_it(self, delays):
        d, pf = delays
        assert self._zero_shift_fs(0.05, d, pf) == pytest.approx(0.0, abs=0.5)

    def test_correction_costs_exactly_one_extra_move(self, delays, move_counter):
        """An ascending sweep only ever reverses once — on the way into the scan
        range — so the approach should cost one extra move, not one per point."""
        d, pf = delays
        stage = SimulatedStage(travel_mm=300.0, backlash_mm=0.05)
        stage.move_to(150.0)
        n = move_counter(stage)
        for t in delay_to_position_um(d, 150000.0, pf):
            stage.move_to(um_to_stage(t))
        assert n[0] == d.size + 1, (
            f"{n[0]} moves for {d.size} points; expected one pre-move "
            f"entering the range")


class TestSimulatedSpectrometerRanges:
    """The GUI builds its integration box from these, so a device that reports
    nothing must still land on a safe whole-millisecond range rather than an
    empty or inverted one."""

    def test_default_exposure_range(self):
        sim = SimulatedSpectrometer(SimulatedStage(), gate="shg")
        assert sim.min_integration_ms == 1.0
        assert sim.max_integration_ms == 10000.0

    def test_exposure_equals_integration_without_onboard_averaging(self):
        sim = SimulatedSpectrometer(SimulatedStage(), gate="shg")
        sim.set_integration_time(25.0)
        assert sim.exposure_ms == sim.integration_ms == 25.0

    def test_stitched_pair_offers_only_what_both_members_accept(self):
        """The facade writes BOTH devices, so its range is the intersection —
        a sub-millisecond Avantes paired with a 1 ms floor gives 1 ms."""
        sim = SimulatedSpectrometer(SimulatedStage(), gate="shg")

        class _Fast(SimulatedSpectrometer):      # stands in for an Avantes
            min_integration_ms = 0.002
            max_integration_ms = 600_000.0

        fast = _Fast(SimulatedStage(), gate="shg")
        pair = (StitchedSpectrometer(fast, sim)
                if fast.wavelengths[0] <= sim.wavelengths[0]
                else StitchedSpectrometer(sim, fast))
        assert pair.min_integration_ms == 1.0
        assert pair.max_integration_ms == 10000.0
