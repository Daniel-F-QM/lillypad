"""KinesisStage: limit switches on an unhomed axis.

After a power cycle the controller counter reads 0 wherever the carriage sits,
so the app lets an unhomed stage move over ±travel — but only when both limit
switches are configured to stop a move, and a move that ends on a switch has to
say so instead of reporting success short of its target.

The fake motor stands in for pylablib's KinesisMotor. `_fake_kinesis` bypasses
__init__ because there is no controller to identify without hardware.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lillypad.hardware.base import StageLimitError
from lillypad.hardware.stages.kinesis import KinesisStage


class FakeMotor:
    """A KinesisMotor whose carriage stops at `switch_lo`/`switch_hi` (in the
    counter frame) with the matching limit bit set, as the firmware does."""

    def __init__(self, switch_lo=-30.0, switch_hi=120.0,
                 kinds=("make", "make")):
        self.pos = 0.0
        self.switch_lo, self.switch_hi = switch_lo, switch_hi
        self.kinds = kinds

    def move_to(self, p):
        self.pos = min(max(p, self.switch_lo), self.switch_hi)

    def wait_move(self):
        pass

    def get_position(self):
        return self.pos

    def get_status(self):
        status = ["enabled"]
        if self.pos <= self.switch_lo:
            status.append("hw_bk_lim")
        if self.pos >= self.switch_hi:
            status.append("hw_fw_lim")
        return status

    def get_limit_switch_parameters(self):
        if self.kinds is None:
            raise RuntimeError("no reply")
        return SimpleNamespace(hw_kind_cw=self.kinds[0],
                               hw_kind_ccw=self.kinds[1])


def _fake_kinesis(motor, travel_mm=150.0):
    stage = object.__new__(KinesisStage)
    stage._motor = motor
    stage._mm_per_unit = 1.0
    stage.name = "LTS150C/M [fake]"
    stage.travel_mm = travel_mm
    stage.backlash_mm = 0.0
    stage.needs_homing = True
    stage.hw_limits_stop = KinesisStage._limits_stop(motor)
    return stage


class TestLimitSwitchConfig:
    @pytest.mark.parametrize("kinds", [("make", "make"), ("break", "make")])
    def test_switches_that_stop_moves(self, kinds):
        assert KinesisStage._limits_stop(FakeMotor(kinds=kinds))

    @pytest.mark.parametrize("kinds", [("ignore", "make"),
                                       ("make", "make_home"), None])
    def test_absent_homing_only_or_unreadable_do_not(self, kinds):
        assert not KinesisStage._limits_stop(FakeMotor(kinds=kinds))


class TestMovingUnhomed:
    def test_carriage_can_move_below_its_power_up_zero(self):
        stage = _fake_kinesis(FakeMotor())
        assert stage.reachable_range_mm() == (-150.0, 150.0)
        stage.move_to(-12.0)
        assert stage.get_position() == -12.0

    def test_running_into_the_reverse_switch_raises_once(self):
        motor = FakeMotor(switch_lo=-30.0)
        stage = _fake_kinesis(motor)
        calls = []
        orig = motor.move_to
        motor.move_to = lambda p: (calls.append(p), orig(p))
        with pytest.raises(StageLimitError, match="reverse limit switch"):
            stage.move_to(-100.0)
        # Six 5 mm sub-moves reach the switch at -30; the seventh, which the
        # switch cuts short, raises — nothing keeps pushing after it.
        assert calls == [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0]
        assert stage.get_position() == -30.0

    def test_arriving_exactly_at_a_switch_is_not_an_error(self):
        stage = _fake_kinesis(FakeMotor(switch_hi=4.0))
        stage.move_to(4.0)                 # on the switch, but on target
        assert stage.get_position() == 4.0
