"""ZaberStage.position_fault on an axis that connected unhomed.

WH/WR only say "not homed". An axis that connected that way may scan in its
power-up frame, so they must not stop the scan at its first point — but once
the axis has been homed they mean the reference was lost, and count again.

`_fake_zaber` bypasses __init__ because there is no device to open without
hardware; only the warning-flag surface position_fault reads is faked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lillypad.hardware import ZaberStage


def _fake_zaber(flags, needs_homing):
    stage = object.__new__(ZaberStage)
    stage._axis = SimpleNamespace(
        warnings=SimpleNamespace(get_flags=lambda: set(flags)))
    stage._has_encoder = False
    stage.needs_homing = needs_homing
    return stage


@pytest.mark.parametrize("flag", ["WH", "WR"])
def test_not_homed_flags_are_quiet_while_working_unhomed(flag):
    assert _fake_zaber({flag}, needs_homing=True).position_fault() is None


@pytest.mark.parametrize("flag", ["WH", "WR"])
def test_they_are_faults_again_once_homed(flag):
    assert _fake_zaber({flag}, needs_homing=False).position_fault()


def test_other_faults_still_count_while_unhomed():
    reason = _fake_zaber({"WR", "FS"}, needs_homing=True).position_fault()
    assert reason == "motor stalled and stopped"
