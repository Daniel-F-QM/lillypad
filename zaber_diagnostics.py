"""
zaber_diagnostics.py -- bench diagnostics for a Zaber delay stage
===============================================================
Answers one question: does the stage go where the program thinks it does?

Why this exists
---------------
Zaber does NO backlash correction of its own (Thorlabs controllers do it in
firmware, which is why the same rig behaves differently on a Kinesis stage). So
a position reached travelling down and the same position reached travelling up
are physically different places. Mark zero-delay after jogging down, then sweep
a scan upward, and the whole delay axis is offset by the screw slack -- tens of
femtoseconds at double pass.

And the offset is invisible from software: zaber-motion's get_position() reads
the `pos` setting, which on a stepper is the TRAJECTORY COUNTER. After a
completed move it returns what you asked for whether or not the carriage got
there. Only the encoder (on devices that have one) can see the difference.

Usage
-----
    python zaber_diagnostics.py                    # report only, no motion
    python zaber_diagnostics.py --port COM5
    python zaber_diagnostics.py --measure          # MOVES: measure backlash
    python zaber_diagnostics.py --check-compensation   # MOVES: verify the fix

--measure and --check-compensation drive the stage and ask before doing it.
Clear the beam path first. Add --yes to skip the prompt in a script.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware import ZaberStage
from scan import position_to_delay_fs

PASS_FACTOR = 2          # double pass, matching the app's default


def _fs(um, pass_factor=PASS_FACTOR):
    """Optical delay equivalent (fs) of a position error of `um`."""
    return float(position_to_delay_fs(um, 0.0, pass_factor))


def _fmt(value, unit="", nd=4):
    return "unsupported" if value is None else f"{value:.{nd}f}{unit}"


# ===========================================================================
# Report -- no motion
# ===========================================================================
def report(stage: ZaberStage) -> None:
    axis, device = stage._axis, stage._device

    print("-- Device ----------------------------------------------")
    print(f"  adapter name       {stage.name}")
    for label, attr in (("device name", "name"), ("serial", "serial_number"),
                        ("axis count", "axis_count")):
        try:
            print(f"  {label:18} {getattr(device, attr)}")
        except Exception:
            pass
    try:
        fw = device.firmware_version
        print(f"  firmware           {fw.major}.{fw.minor}.{fw.build}")
    except Exception:
        pass
    # peripheral_name / peripheral_id are properties on Axis itself; they are
    # empty on an integrated device (one where the stage is not a swappable
    # peripheral), which is not an error.
    for label, attr in (("peripheral", "peripheral_name"),
                        ("peripheral id", "peripheral_id"),
                        ("axis type", "axis_type")):
        try:
            print(f"  {label:18} {getattr(axis, attr) or '--'}")
        except Exception:
            pass

    print("\n-- Range and resolution --------------------------------")
    lo = stage.travel_min_mm
    hi = stage.travel_mm
    print(f"  limit.min          {_fmt(lo, ' mm')}")
    print(f"  limit.max          {_fmt(hi, ' mm')}")
    if hi is not None:
        print(f"  usable travel      {hi - (lo or 0.0):.4f} mm")
    if lo:
        print("    NOTE limit.min is not 0 -- a coordinate range starting at 0 "
              "would command moves the firmware rejects.")

    # Microstep size: ask for the same setting in mm and in native units; the
    # ratio is mm per microstep. Cleaner than guessing from the motor spec.
    micro_um = None
    try:
        from zaber_motion import Units
        native = float(axis.settings.get("limit.max", Units.NATIVE))
        if native:
            micro_um = abs(hi / native) * 1000.0
    except Exception:
        pass
    if micro_um:
        print(f"  microstep          {micro_um:.6f} um  =  {_fs(micro_um):.2f} fs")
        print("    Quantisation floor: no commanded position can be finer.")
    for name, unit in (("maxspeed", " mm/s"), ("accel", " mm/s^2")):
        print(f"  {name:18} {_fmt(stage._setting(name), unit, 3)}")

    print("\n-- Position reference ----------------------------------")
    print(f"  is_homed()         {not stage.needs_homing}")
    if stage.needs_homing:
        print("    NOTE unhomed. `pos` is an arbitrary number until you home, "
              "and homing will shift the coordinate frame under any zero-delay "
              "marked before it.")
    print(f"  pos                {stage.get_position():.6f} mm   "
          f"(trajectory counter -- NOT a measurement)")
    enc = stage.encoder_position()
    if enc is None:
        print("  encoder.pos        none on this device")
        print("    Without an encoder nothing can detect lost steps, and "
              "--measure cannot read the backlash directly.")
    else:
        gap_um = abs(enc - stage.get_position()) * 1000.0
        print(f"  encoder.pos        {enc:.6f} mm   (measured)")
        print(f"  divergence         {gap_um:.2f} um  =  {_fs(gap_um):.1f} fs")
    print(f"  cloop.mode         {_fmt(stage._setting('cloop.mode'), '', 0)}")

    print("\n-- Warning flags ---------------------------------------")
    try:
        flags = sorted(stage._axis.warnings.get_flags())
    except Exception as e:
        print(f"  could not read flags: {e}")
        flags = []
    if not flags:
        print("  none -- the axis reports no complaints.")
    for f in flags:
        note = ZaberStage._FAULT_FLAGS.get(f)
        print(f"  {f}   {note or 'informational'}"
              + ("   <-- INVALIDATES THE POSITION" if note else ""))

    print("\n-- Backlash correction ---------------------------------")
    b_um = stage.backlash_mm * 1000.0
    print(f"  backlash_mm        {stage.backlash_mm:.4f} mm ({b_um:.1f} um)")
    if b_um > 0:
        print(f"    Every move undershoots by up to {b_um:.0f} um and comes "
              f"back up, so all positions share one approach direction.")
    else:
        print("    OFF -- the marked zero and a scan sweep can sit in frames "
              "that differ by the screw slack. Run --measure to see by how "
              "much.")


# ===========================================================================
# Motion tests
# ===========================================================================
def _confirm(what: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    print(f"\n{what}")
    try:
        return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _target_and_excursion(stage: ZaberStage, excursion_mm: float):
    """A mid-travel target with room to approach from both sides."""
    lo = stage.travel_min_mm or 0.0
    hi = stage.travel_mm
    if hi is None:
        return stage.get_position(), excursion_mm
    target = 0.5 * (lo + hi)
    # Never ask for more excursion than the travel can give on both sides.
    room = min(target - lo, hi - target)
    return target, max(min(excursion_mm, room * 0.9), 0.0)


def _approach(stage, target_mm, from_mm):
    """Park at from_mm, then go to target_mm. Returns (pos, encoder)."""
    stage._move_to_raw(from_mm)      # raw: this test IS the approach direction
    stage._move_to_raw(target_mm)
    return stage.get_position(), stage.encoder_position()


def measure_backlash(stage: ZaberStage, repeats: int, excursion_mm: float,
                     assume_yes: bool) -> None:
    target, excursion = _target_and_excursion(stage, excursion_mm)
    if excursion <= 0:
        print("Not enough travel around mid-range to run the test.")
        return
    if not _confirm(
            f"This MOVES the stage: {repeats} round trips of +/-{excursion:.2f} mm "
            f"around {target:.3f} mm. Clear the beam path.", assume_yes):
        print("Cancelled.")
        return

    print(f"\nApproaching {target:.4f} mm from both sides, {repeats} times.\n")
    print(f"  {'#':>3}  {'from below':>26}  {'from above':>26}")
    print(f"  {'':>3}  {'pos (mm)':>12} {'enc (mm)':>13}"
          f"  {'pos (mm)':>12} {'enc (mm)':>13}")

    up_enc, down_enc, up_pos, down_pos = [], [], [], []
    for i in range(repeats):
        pu, eu = _approach(stage, target, target - excursion)   # arrive upward
        pd, ed = _approach(stage, target, target + excursion)   # arrive downward
        up_pos.append(pu); down_pos.append(pd)
        if eu is not None:
            up_enc.append(eu); down_enc.append(ed)
        print(f"  {i + 1:>3}  {pu:>12.6f} {('--' if eu is None else f'{eu:.6f}'):>13}"
              f"  {pd:>12.6f} {('--' if ed is None else f'{ed:.6f}'):>13}")

    pos_gap_um = abs(np.mean(up_pos) - np.mean(down_pos)) * 1000.0
    print(f"\n  `pos` differs between the two directions by {pos_gap_um:.3f} um "
          f"({_fs(pos_gap_um):.2f} fs)")
    print("    Expected to be ~0: `pos` is the commanded value, which is why "
          "the app cannot see this on its own.")

    if not up_enc:
        print("\n  No encoder on this device, so the backlash cannot be read "
              "electrically. Measure it optically instead: the two approaches "
              "above land in different places, so a FROG trace taken after "
              "each will sit at a different delay. That difference IS the "
              "backlash.")
        return

    diffs_um = (np.array(up_enc) - np.array(down_enc)) * 1000.0
    mean_um, spread_um = float(np.mean(diffs_um)), float(np.std(diffs_um))
    print(f"\n  BACKLASH (encoder) {mean_um:+.3f} um  +/- {spread_um:.3f} um")
    print(f"                     {_fs(mean_um):+.2f} fs  +/- {_fs(spread_um):.2f} fs"
          f"   at pass_factor={PASS_FACTOR}")
    print(f"\n  This is the zero-delay shift to expect when the position you "
          f"mark as zero\n  is approached from one side and the scan sweeps "
          f"from the other.")
    need_um = abs(mean_um) * 2.0
    print(f"  Set the stage's backlash margin to at least {need_um:.0f} um "
          f"(currently {stage.backlash_mm * 1000.0:.0f} um).")


def check_compensation(stage: ZaberStage, repeats: int, excursion_mm: float,
                       assume_yes: bool) -> None:
    if stage.backlash_mm <= 0:
        print("backlash_mm is 0 on this stage -- nothing to verify. Pass "
              "--backlash-um to set a margin for this test.")
        return
    target, excursion = _target_and_excursion(stage, excursion_mm)
    if excursion <= 0:
        print("Not enough travel around mid-range to run the test.")
        return
    if not _confirm(
            f"This MOVES the stage: {repeats} round trips of +/-{excursion:.2f} mm "
            f"around {target:.3f} mm, through the corrected move_to(). "
            f"Clear the beam path.", assume_yes):
        print("Cancelled.")
        return

    print(f"\nSame two-sided test, but arriving through move_to() with a "
          f"{stage.backlash_mm * 1000.0:.0f} um approach margin.\n")
    landed = []
    for i in range(repeats):
        for label, park in (("below", target - excursion),
                            ("above", target + excursion)):
            stage._move_to_raw(park)
            stage.move_to(target)          # corrected: always arrives upward
            enc = stage.encoder_position()
            landed.append(enc if enc is not None else stage.get_position())
            print(f"  {i + 1:>3}  parked {label:<5}  ->  "
                  f"{landed[-1]:.6f} mm")

    spread_um = float(np.max(landed) - np.min(landed)) * 1000.0
    print(f"\n  Spread across both approach directions: {spread_um:.3f} um "
          f"({_fs(spread_um):.2f} fs)")
    if stage.encoder_position() is None:
        print("    No encoder, so this only shows the commanded positions "
              "agree. Confirm optically.")
    elif spread_um < 1.0:
        print("    Compensation is working -- the approach direction no longer "
              "changes where the stage lands.")
    else:
        print("    Still spread out. Increase the backlash margin (it must "
              "exceed the mechanical slack) and re-measure.")


# ===========================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None,
                   help="serial port (e.g. COM5); default auto-scan")
    p.add_argument("--device-index", type=int, default=0,
                   help="which device on a daisy chain (default 0)")
    p.add_argument("--axis", type=int, default=1,
                   help="axis number (default 1)")
    p.add_argument("--backlash-um", type=float, default=None,
                   help="override the adapter's approach margin for this run")
    p.add_argument("--measure", action="store_true",
                   help="MOVES THE STAGE: measure the backlash")
    p.add_argument("--check-compensation", action="store_true",
                   help="MOVES THE STAGE: verify the correction removes it")
    p.add_argument("--repeats", type=int, default=5,
                   help="round trips per motion test (default 5)")
    p.add_argument("--excursion-mm", type=float, default=1.0,
                   help="how far to back off before each approach (default 1)")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt before moving")
    args = p.parse_args(argv)

    try:
        stage = ZaberStage(port=args.port, device_index=args.device_index,
                           axis_number=args.axis)
    except Exception as e:
        print(f"Could not open a Zaber stage: {e}", file=sys.stderr)
        return 1

    if args.backlash_um is not None:
        stage.backlash_mm = args.backlash_um / 1000.0

    try:
        report(stage)
        if args.measure:
            measure_backlash(stage, args.repeats, args.excursion_mm, args.yes)
        if args.check_compensation:
            check_compensation(stage, args.repeats, args.excursion_mm, args.yes)
        if not (args.measure or args.check_compensation):
            print("\n(Report only -- nothing was moved. Add --measure to "
                  "measure the backlash.)")
    finally:
        stage.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
