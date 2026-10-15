"""Driving the delay stage by hand: jog, absolute move, home, mark zero.

Every move goes through `_stage_action`, which is the whole point of the file: it
takes the feed handover, disables the controls, runs the move, catches whatever
the vendor driver throws, reports it, and re-enables. A move written any other
way is a move that can leave the buttons dead or two threads inside one driver.

Two conversions live here because they are UI concerns rather than physics. The
jog and move boxes switch between femtoseconds and micrometres, and which one
they are in changes their ranges, their step sizes and their suffixes — the
optics conversion itself is in lillypad.scan.core.

The **backlash margin** is here rather than with the scan because it is a
property of the connected stage. Lead-screw stages land in a different place
depending on which way they arrived, so undershooting by more than the slack and
coming back up is what makes the zero you mark by jogging and the positions a
scan sweeps through sit in the same frame. 0 is correct for a Thorlabs
controller (its firmware already does this) or a piezo, and wrong for a Zaber.
"""

import math

from lillypad.gui.text import FEED_BUSY_MSG
from lillypad.scan import (delay_to_position_um, position_to_delay_fs,
                           stage_to_um, um_to_stage)


class StageController:
    """Manual stage control, unit switching, and the backlash margin."""

    def __init__(self, win):
        self.win = win

    # ── Manual stage ──────────────────────────────────────────────────────────
    def _set_stage_controls_enabled(self, on):
        for w in (self.win.btn_minus, self.win.btn_plus, self.win.btn_moveto,
                  self.win.btn_home, self.win.btn_goto_zero, self.win.btn_set_zero,
                  self.win.btn_units, self.win.spin_backlash):
            w.setEnabled(on)
        self._refresh_home_button()

    def _refresh_home_button(self):
        """Grey Home out on a stage with no homing routine (the piezo), rather
        than let it fail on click. Otherwise it follows the rest of the stage
        controls — with no stage it stays live, like they do, and says
        "Connect a stage first". Call after a stage swap."""
        stage = self.win.stage
        can = stage is None or stage.can_home
        self.win.btn_home.setEnabled(can and self.win.btn_minus.isEnabled())
        self.win.btn_home.setToolTip(
            "" if can else f"{stage.name} has no homing routine.")

    def _moving(self, on):
        self.win.lbl_moving.setText("MOVING" if on else "IDLE")
        self.win.lbl_moving.setObjectName("moving" if on else "idle")
        self.win.lbl_moving.style().unpolish(self.win.lbl_moving); self.win.lbl_moving.style().polish(self.win.lbl_moving)
        # repaint() paints synchronously WITHOUT spinning the event loop.
        # processEvents() here used to dispatch queued work — a spectrum_ready
        # delivery, or another click on the jog button — in the middle of a
        # blocking move, re-entering this code path.
        self.win.lbl_moving.repaint()

    def _stage_action(self, fn, done_msg=None):
        """Run a blocking stage operation from a slot, safely.

        Parks the live feed (which drives the same stage from thread 2),
        disables the stage controls so the operation cannot be re-entered, and
        restores both however `fn` exits.
        """
        if self.win.scanning.running():
            self.win.status.showMessage("A scan is running — the stage is busy.", 3000)
            return
        if not self.win._have_stage():
            self.win.status.showMessage("Connect a stage first.", 4000)
            return
        self._set_stage_controls_enabled(False)
        self._moving(True)
        try:
            with self.win.devices.device_lock() as ok:
                if not ok:
                    self.win.status.showMessage(FEED_BUSY_MSG, 5000)
                    return
                fn()
                self._refresh_positions()   # still inside the lock
                # An adapter that reports a failed move through position_fault()
                # instead of raising (the piezo does, so a scan can record the
                # column and carry on) would otherwise leave a manual move
                # looking successful while the stage sits somewhere else.
                fault = None
                try:
                    fault = self.win.stage.position_fault()
                except Exception:
                    pass
                if fault:
                    self.win.status.showMessage(f"Stage: {fault}", 8000)
                elif done_msg:
                    self.win.status.showMessage(done_msg, 3000)
        except Exception as e:
            self.win.status.showMessage(f"Stage error: {e}", 5000)
        finally:
            self._moving(False)
            self._set_stage_controls_enabled(True)

    def _travel_range_um(self):
        """Reachable position range (lo, hi) in um.

        Not [0, travel]: limit.min is not always 0 (a soft limit protecting the
        optics is common), an unhomed axis whose limit switches stop it may go
        negative (StageBase.reachable_range_mm), and the low end has to leave
        room for the backlash pre-move to undershoot into. Falls back to 300 mm
        of travel when the adapter reports none.
        """
        stage = self.win.stage          # None at build time and after disconnect
        lo_mm, hi_mm = (stage.reachable_range_mm() if stage is not None
                        else (0.0, None))
        hi = stage_to_um(hi_mm if hi_mm else 300.0)
        lo = stage_to_um(float(lo_mm or 0.0)
                          + float(getattr(stage, "backlash_mm", 0.0) or 0.0))
        return lo, hi

    def _update_stage_unit_ranges(self):
        """Set spin_step/spin_moveto ranges for the active unit, the connected
        stage's travel, and the current zero. Call after: unit toggle, stage
        swap, zero change, backlash change."""
        lo_um, hi_um = self._travel_range_um()
        pf, z = self.win.scan_cfg.pass_factor, self.win.scan_cfg.zero_pos_um
        if self.win._stage_units_fs:
            # Inward rounding (ceil min, floor max) so every integer fs in
            # range maps to a position strictly inside the travel range.
            self.win.spin_step.setRange(
                1, max(1, math.floor(float(
                    position_to_delay_fs(hi_um - lo_um, 0.0, pf)))))
            self.win.spin_moveto.setRange(
                math.ceil(float(position_to_delay_fs(lo_um, z, pf))),
                math.floor(float(position_to_delay_fs(hi_um, z, pf))))
        else:
            self.win.spin_step.setRange(0.01, hi_um - lo_um)
            self.win.spin_moveto.setRange(lo_um, hi_um)

    def _sync_backlash_ui(self):
        """Push the active stage's backlash into the spin box and its fs hint.

        Blocks signals: this reflects the stage, it must not write back to it.
        """
        b_um = stage_to_um(float(getattr(self.win.stage, "backlash_mm", 0.0) or 0.0))
        self.win.spin_backlash.blockSignals(True)
        self.win.spin_backlash.setValue(b_um)
        self.win.spin_backlash.blockSignals(False)
        self._refresh_backlash_hint(b_um)

    def _refresh_backlash_hint(self, b_um):
        if b_um <= 0.0:
            self.win.lbl_backlash_fs.setText(
                "Off — moves arrive from whichever side they came from.")
            return
        fs = float(position_to_delay_fs(b_um, 0.0, self.win.scan_cfg.pass_factor))
        self.win.lbl_backlash_fs.setText(
            f"Approach from below, undershooting {b_um:.1f} um ({fs:.0f} fs).")

    def on_backlash_changed(self, value_um):
        if self.win.scanning.running():
            self.win.status.showMessage("A scan is running — the stage is busy.", 3000)
            self._sync_backlash_ui()      # snap back to what the stage has
            return
        if not self.win._have_stage():
            return
        # Plain attribute write, no device I/O — no lock needed. Ranges shift
        # because the low end reserves the margin.
        self.win.stage.backlash_mm = um_to_stage(float(value_um))
        self._refresh_backlash_hint(float(value_um))
        self._update_stage_unit_ranges()

    def _toggle_stage_units(self):
        pf, z = self.win.scan_cfg.pass_factor, self.win.scan_cfg.zero_pos_um
        step, move = self.win.spin_step.value(), self.win.spin_moveto.value()
        self.win._stage_units_fs = not self.win._stage_units_fs
        if self.win._stage_units_fs:
            new_step = round(float(position_to_delay_fs(step, 0.0, pf)))
            new_move = round(float(position_to_delay_fs(move, z, pf)))
            suffix, dec = " fs", 0
        else:
            new_step = round(float(delay_to_position_um(step, 0.0, pf)))
            new_move = round(float(delay_to_position_um(move, z, pf)))
            suffix, dec = " um", 2
        for s in (self.win.spin_step, self.win.spin_moveto):
            s.setSuffix(suffix); s.setDecimals(dec)
        self._update_stage_unit_ranges()   # ranges before values: no bad clamp
        self.win.spin_step.setValue(new_step)
        self.win.spin_moveto.setValue(new_move)
        self.win.btn_units.setText("fs" if self.win._stage_units_fs else "um")

    def _jog(self, sign):
        v = self.win.spin_step.value()
        step_um = (float(delay_to_position_um(v, 0.0, self.win.scan_cfg.pass_factor))
                   if self.win._stage_units_fs else v)
        lo, hi = self._travel_range_um()
        def do():
            # Clamped move_to instead of move_by: the target can never leave
            # the travel range. get_position() runs inside the device lock.
            cur = stage_to_um(self.win.stage.get_position())
            target = min(max(cur + sign * step_um, lo), hi)
            self.win.stage.move_to(um_to_stage(target))
        self._stage_action(do)

    def _move_absolute(self):
        v = self.win.spin_moveto.value()
        pf, z = self.win.scan_cfg.pass_factor, self.win.scan_cfg.zero_pos_um
        target_um = float(delay_to_position_um(v, z, pf)) if self.win._stage_units_fs else v
        lo, hi = self._travel_range_um()
        target_um = min(max(target_um, lo), hi)
        self._stage_action(lambda: self.win.stage.move_to(um_to_stage(target_um)))

    def _move_to_zero(self):
        self._stage_action(
            lambda: self.win.stage.move_to(um_to_stage(self.win.scan_cfg.zero_pos_um)))

    def _home_stage(self):
        """Home the axis, and drop a zero that homing has invalidated.

        Homing an axis that had no reference re-establishes the coordinate
        frame, which moves any zero marked in the old one. Silently keeping it
        would shift the whole delay axis by the home offset, so it goes.
        """
        was_unreferenced = bool(getattr(self.win.stage, "needs_homing", False))
        landed = []          # stays empty if home() raised or the feed was busy

        def do():
            self.win.stage.home()
            # Read it here, inside the lock — get_position() is device I/O and
            # the live feed drives the same stage from its own thread.
            landed.append(stage_to_um(self.win.stage.get_position()))

        self._stage_action(do)
        if not landed:
            return           # _stage_action already reported why
        pos_um = landed[0]
        if was_unreferenced:
            self.win.scan_cfg.zero_pos_um = pos_um
            self._update_stage_unit_ranges()
            self._refresh_positions()
            self._refresh_scan_um()
            self.win.status.showMessage(
                f"Homed to {pos_um:.2f} um. The axis had no reference before, "
                f"so the old zero-delay was in a different frame — set zero "
                f"again before scanning.", 0)
        else:
            self.win.status.showMessage(f"Stage homed to {pos_um:.2f} um.", 3000)

    def _mark_zero(self):
        def mark():
            # get_position() is a device read like any other — take the lock.
            self.win.scan_cfg.zero_pos_um = stage_to_um(self.win.stage.get_position())
        self._stage_action(mark)
        # fs-mode values are delays and stay as typed — the zero moved, so the
        # same delay now maps to the new (correct) absolute position. Only the
        # ranges need recomputing; Qt clamps anything now out of range.
        self._update_stage_unit_ranges()
        self._refresh_scan_um()
        self.win.status.showMessage(f"Zero-delay set to {self.win.scan_cfg.zero_pos_um:.2f} um.", 3000)

    def _refresh_positions(self):
        self.win.lbl_pos.setText(
            f"{stage_to_um(self.win.stage.get_position()):.2f} um"
            if self.win._have_stage() else "— um")
        self.win.lbl_zero.setText(f"{self.win.scan_cfg.zero_pos_um:.2f} um")

    def _refresh_scan_um(self):
        z = self.win.scan_cfg.zero_pos_um
        p0 = delay_to_position_um(self.win.spin_start.value(), z, self.win.scan_cfg.pass_factor)
        p1 = delay_to_position_um(self.win.spin_stop.value(),  z, self.win.scan_cfg.pass_factor)
        self.win.eq_start.setText(f"{p0:,.1f} um")
        self.win.eq_stop.setText(f"{p1:,.1f} um")
