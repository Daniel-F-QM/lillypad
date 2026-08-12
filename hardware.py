"""
hardware.py — hardware abstraction layer for the FROG acquisition app
=====================================================================
Everything device-specific lives here so the GUI never imports a vendor SDK
directly. To support new hardware, implement StageBase / SpectrometerBase in
this file (or a sibling module) — the UI only ever sees the abstract methods.

Unit conventions
----------------
  * Stage positions are ALWAYS in millimetres. The stage knows nothing about
    femtoseconds; the FROG/optics layer (next module) owns the
    delay(fs) <-> position(mm) conversion and the zero-delay offset.
  * Spectrometer wavelengths are in nm; spectra are raw counts vs pixel.

Adapters are imported lazily, so this module imports cleanly on a dev machine
with only numpy installed (no pylablib / seabreeze needed to run the sims).

    python hardware.py     # runs a quick simulator self-test, no hardware
"""

from __future__ import annotations

import abc
import math
import os
import re
import sys
import time
import numpy as np
from collections.abc import Callable
from pathlib import Path

C_NM_PER_FS = 299.792458   # speed of light, nm/fs (used only by the simulator)


def load_calibration_file(path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an intensity-calibration text file: two whitespace-separated
    columns, wavelength (nm) and multiplicative factor, '#' comments allowed.
    Returns (wavelengths, factors) sorted by wavelength."""
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"{Path(path).name}: expected two columns (wavelength_nm factor) "
            f"with at least two rows, got data of shape {data.shape}.")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


# ===========================================================================
# Abstract interfaces — implement these to add new hardware
# ===========================================================================
class StageBase(abc.ABC):
    """A 1-D motorised delay stage. All positions are in MILLIMETRES.

    Absolute/relative moves MUST block until the stage has settled, so the
    scan loop can read back a trustworthy position immediately afterwards.
    Subclasses implement the raw vendor move as _move_to_raw; move_to on this
    base wraps it with the unidirectional approach described below.

    Backlash / approach direction
    -----------------------------
    On a lead-screw stage the position you land on depends on which way you
    were travelling when you got there. A FROG scan sweeps monotonically, so
    every scan point is approached from one side — but the zero-delay position
    is marked after jogging, which arrives from whichever side the operator
    happened to be on. The two frames then differ by the mechanical backlash,
    and the whole trace sits at the wrong delay. At double pass, 8 um of
    backlash is 53 fs.

    Setting backlash_mm > 0 makes EVERY move on this stage arrive travelling in
    the + direction, so the marked zero and the scan share one frame. The
    pre-move only happens when the move would otherwise arrive from above, so a
    monotonic ascending sweep pays nothing for it.

    Leave backlash_mm at 0 when the controller already does this in firmware
    (Thorlabs Kinesis does) or when there is no backlash to correct (piezo).

    Step limit
    ----------
    A move is blocking and cannot be interrupted — stop() has no call site — so a
    mistyped target, a stale zero or a fs/um mix-up becomes one long traverse at
    full speed with nothing to halt it. max_step_mm caps how far a single raw
    move may travel: anything longer is carried out as a run of sub-moves of at
    most that far, each one settling before the next is issued, and the operator
    is warned. The landing point is unchanged — the last sub-move is the exact
    target — so this costs nothing but time, and only on moves that were already
    long. Scan points are microns apart and never trip it.
    """
    units = "mm"
    name  = "stage"
    travel_mm:     float | None = None
    travel_min_mm: float = 0.0
    backlash_mm:   float = 0.0
    # Longest single raw move, in mm. 0 disables the split entirely.
    max_step_mm:   float = 5.0
    # Where long-move warnings go. The GUI points this at a status-bar signal;
    # left None (headless, self-tests) they go to stderr.
    warn_cb: Callable[[str], None] | None = None
    # True when the axis has no valid reference and its readback is meaningless
    # until homed. Adapters that can tell set this; the GUI refuses to adopt a
    # zero-delay position from a stage that reports it.
    needs_homing:  bool = False

    def move_to(self, position_mm: float) -> None:
        """Absolute move to position_mm; blocks until settled.

        With backlash_mm > 0 the stage always arrives travelling in the +
        direction: if we would otherwise come down onto the target, undershoot
        by backlash_mm first and take the slack up on the way back.

        Both legs go through _move_stepped, so neither can travel further than
        max_step_mm in one go. The position is read once here and then carried
        through, so a move costs the same device I/O it always did.
        """
        target = float(position_mm)
        b = self.backlash_mm
        if b <= 0.0 and (self.max_step_mm or 0.0) <= 0.0:
            self._move_to_raw(target)      # nothing to arrange — no readback
            return
        cur = self.get_position()
        if b > 0.0:
            pre = max(target - b, self.travel_min_mm)
            # pre == target means the target sits inside the backlash margin at
            # the bottom of travel and there is no room to undershoot. Move
            # directly rather than refuse — the range checks in scan.py and the
            # GUI reserve this margin so it should not normally happen.
            if pre < target and cur > target - 1e-9:
                self._move_stepped(cur, pre)
                cur = pre
        # The run-up from pre to target is backlash_mm (tens of um), so it stays
        # one move and the arrival is still unidirectional.
        self._move_stepped(cur, target)

    def _move_stepped(self, cur_mm: float, target_mm: float) -> None:
        """Absolute move to target_mm from a known cur_mm, in sub-moves of at
        most max_step_mm. No backlash logic — move_to owns that."""
        step = float(self.max_step_mm or 0.0)
        dist = abs(target_mm - cur_mm)
        if step <= 0.0 or dist <= step + 1e-9:
            self._move_to_raw(target_mm)
            return

        n = int(math.ceil(dist / step - 1e-9))
        self._warn(f"{self.name}: {dist:.3f} mm move ({cur_mm:.3f} -> "
                   f"{target_mm:.3f} mm) split into {n} steps of at most "
                   f"{step:g} mm")
        sign = 1.0 if target_mm > cur_mm else -1.0
        lo = self.travel_min_mm
        hi = self.travel_mm
        for k in range(1, n):
            # Stepped off cur_mm rather than off a fresh readback: no extra I/O,
            # and no room for readback error to accumulate along the way. The
            # clamp guards the case where cur_mm was never trustworthy in the
            # first place (needs_homing) and the arithmetic walks out of travel.
            mid = cur_mm + sign * step * k
            mid = max(mid, lo)
            if hi is not None:
                mid = min(mid, hi)
            self._move_to_raw(mid)
        self._move_to_raw(target_mm)       # exact target, always

    def _warn(self, message: str) -> None:
        """Report something the operator should see. Goes to warn_cb when the
        UI has set one, otherwise to stderr."""
        cb = self.warn_cb
        if cb is None:
            print(f"warning: {message}", file=sys.stderr)
            return
        try:
            cb(message)
        except Exception:
            print(f"warning: {message}", file=sys.stderr)

    def move_by(self, delta_mm: float) -> None:
        """Relative move by delta_mm; blocks until settled."""
        self.move_to(self.get_position() + float(delta_mm))

    @abc.abstractmethod
    def _move_to_raw(self, position_mm: float) -> None:
        """Vendor absolute move to position_mm; block until settled.

        No backlash handling here — move_to owns that.
        """

    @abc.abstractmethod
    def get_position(self) -> float:
        """Read back the *actual* current position in mm."""

    def position_fault(self) -> str | None:
        """Human-readable reason the readback cannot be trusted, or None.

        Checked by the scan loop after every point, so a stall, a knob nudge or
        an unhomed axis stops the scan instead of silently mis-labelling the
        trace. Adapters that cannot report health leave this at None.
        """
        return None

    def clear_position_faults(self) -> None:
        """Clear latched controller warnings, so a scan only ever sees faults
        that happened during that scan."""

    def home(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def is_moving(self) -> bool:
        return False

    def disconnect(self) -> None:
        pass


class SpectrometerBase(abc.ABC):
    """A spectrometer returning one spectrum (counts vs pixel) per acquire()."""
    name = "spectrometer"
    # Integration time currently set, in ms. Part of the interface because the
    # live feed paces itself off it (and derives its handover timeout from it),
    # so EVERY adapter must keep this current in set_integration_time().
    integration_ms: float = 100.0
    # Full-scale reading of the detector, in counts — the level at which the
    # ADC clips and the measurement stops being a measurement. Adapters SHOULD
    # set this; the scan worker and the live feed both use it to flag
    # saturation, and neither can warn about anything while it is None.
    max_counts: float | None = None
    # The tagged id ("vendor:serial") this device was opened by — see
    # list_spectrometers / open_spectrometer. None on simulated devices, which
    # are not reachable by id. Real adapters MUST set it: it is how the GUI
    # names a live device without reaching into a vendor object's privates,
    # and how a multi-spectrometer slot remembers what it points at.
    spec_id: str | None = None

    @property
    @abc.abstractmethod
    def wavelengths(self) -> np.ndarray:
        """Fixed wavelength axis (nm), one entry per pixel."""

    @property
    def n_pixels(self) -> int:
        return int(self.wavelengths.size)

    @abc.abstractmethod
    def set_integration_time(self, ms: float) -> None:
        ...

    @abc.abstractmethod
    def acquire(self) -> np.ndarray:
        """Return a single spectrum (counts vs pixel)."""

    # ── Intensity calibration ────────────────────────────────────────────────
    # Per-pixel factors the raw counts are multiplied by, or None for raw
    # output. Deliberately NOT folded into acquire(): saturation is a property
    # of raw ADC counts, so consumers must check saturation on acquire()'s
    # output and only then run it through calibrate().
    calibration_name: str | None = None
    _cal_pixels: np.ndarray | None = None

    def set_calibration(self, path) -> None:
        """Load a calibration file (see load_calibration_file) and hold it
        interpolated onto this device's pixel grid. Pixels outside the file's
        wavelength range hold the edge factor rather than extrapolating."""
        wl, fac = load_calibration_file(path)
        self._cal_pixels = np.interp(np.asarray(self.wavelengths, float),
                                     wl, fac)
        self.calibration_name = Path(path).stem

    def clear_calibration(self) -> None:
        self._cal_pixels = None
        self.calibration_name = None

    def calibrate(self, counts: np.ndarray) -> np.ndarray:
        """Apply the loaded intensity calibration (identity when none)."""
        cal = self._cal_pixels
        return counts if cal is None else counts * cal

    def calibration_targets(self) -> list["SpectrometerBase"]:
        """The physical spectrometers a calibration file can be assigned to:
        [self], except for composites like StitchedSpectrometer, which expose
        their members so each can take its own file."""
        return [self]

    def disconnect(self) -> None:
        pass


# ===========================================================================
# Stage calibration registry
# ===========================================================================
# A Kinesis stage can get its millimetre calibration two ways, tried in order:
#
#   1. An entry below, selected by matching the model number the CONTROLLER
#     reports against that entry's `models` patterns. This is for stages
#     pylablib cannot calibrate itself — the LTS150/LTS300 are two: their
#     controllers report no stage ID, so pylablib would silently fall back to
#     raw steps.
#   2. pylablib's own calibration (KinesisMotor(scale="stage")), which covers
#     the stages it can identify — the Z6xx/Z7xx/Z8xx and MTS families on a
#     KDC101/TDC001, K10CR1, MPC…
#
# A stage matching NEITHER is refused rather than driven with somebody else's
# steps-per-mm, which would quietly distort the delay axis instead of failing.
#
# Entry fields:
#   models    — regexes matched (case-insensitively, anchored at the start)
#               against the reported model number. One entry can cover a family.
#   scale     — passed straight to pylablib's KinesisMotor:
#                 * a (position, velocity, acceleration) tuple in steps per
#                   MILLIMETRE — pylablib calls these units "user" and applies
#                   no conversion of its own, so the stage reads out in mm; or
#                 * "stage" / a pylablib stage name ("MTS50-Z8", …) — pylablib
#                   then works in METRES, which KinesisStage converts.
#   travel_mm — full travel, used as the scan-range soft limit. Optional.
#
# Either way the resulting scale units are VERIFIED after opening (see
# KinesisStage), so a rotational or uncalibrated stage cannot slip through as
# a millimetre one. Adding an entry is all it takes; nothing in the UI changes.
STAGE_CONFIGS: dict[str, dict] = {
    "LTS300C/M": {
        # Matches LTS300/M, LTS300C/M, … — every LTS300 variant shares the
        # controller and therefore the scale.
        "models": (r"LTS300",),
        # pylablib KinesisMotor scale = (position, velocity, acceleration) in
        # steps per physical unit. These are the LTS300C/M's APT values
        # (409600 steps/mm), so get_position()/move_to() come out directly in mm.
        "scale": (409600, 21990232, 4506),
        "travel_mm": 300.0,
    },
    "LTS150C/M": {
        # Matches LTS150/M, LTS150C/M, … The LTS150 shares the LTS300's
        # controller and step scale and differs only in travel. Verified
        # against a connected LTS150 (fw 3.0.8), whose factory defaults read
        # back exactly right through this scale: 50 um backlash distance,
        # 0.5 mm home offset, 20 mm/s max velocity, 2 mm/s homing velocity,
        # 20 mm/s^2 acceleration.
        "models": (r"LTS150",),
        "scale": (409600, 21990232, 4506),
        "travel_mm": 150.0,
    },
    # To add another stage pylablib cannot calibrate, copy an entry above:
    # match its reported model number and give the APT steps per mm/mm/s/mm/s^2.
}

# Travel of the stages pylablib identifies by itself, so an auto-calibrated
# stage still gets a soft scan-range limit. This covers every translational
# stage in pylablib's autodetect table; a stage missing from it connects with
# travel_mm = None, i.e. no soft limit beyond the stage's own limit switches.
KINESIS_TRAVEL_MM: dict[str, float] = {
    "Z806": 6.0, "Z812": 12.0, "Z825": 25.0,
    "Z706": 6.0, "Z712": 12.0, "Z725": 25.0,
    "MTS25-Z8": 25.0, "MTS50-Z8": 50.0,
}


def _match_stage_config(model_no: str | None) -> str | None:
    """The STAGE_CONFIGS key whose `models` patterns match `model_no`, else None."""
    if not model_no:
        return None
    for key, cfg in STAGE_CONFIGS.items():
        for pattern in cfg.get("models", ()):
            if re.match(pattern, model_no, re.IGNORECASE):
                return key
    return None


def _kinesis_model(conn: str, timeout: float = 1.0) -> str | None:
    """The model number a Kinesis controller reports, or None if unreadable.

    Opens the device and closes it again straight away — it only asks for the
    identification message, so nothing moves and no device state changes. The
    short timeout keeps enumeration snappy when a device is busy or wedged.
    """
    from pylablib.devices import Thorlabs
    dev = None
    try:
        dev = Thorlabs.BasicKinesisDevice(conn, timeout=timeout)
        return str(dev.get_device_info().model_no).strip() or None
    except Exception:
        return None
    finally:
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass


def list_kinesis_stages() -> list[tuple[str, str]]:
    """Enumerate attached Thorlabs Kinesis devices without adopting any.

    Returns [(model, conn), ...] — deliberately the same shape as
    list_seabreeze_spectrometers(), so the GUI can drive one picker dialog for
    both. `conn` is the connection string (normally the 8-digit serial) that
    KinesisStage(serial=...) takes. Each device is briefly opened to read its
    model number; one whose model cannot be read (already in use, say) falls
    back to the USB description and then to "?" rather than being dropped.
    """
    from pylablib.devices import Thorlabs
    out = []
    for entry in Thorlabs.list_kinesis_devices():
        conn = str(entry[0])
        description = str(entry[1]).strip() if len(entry) > 1 else ""
        out.append((_kinesis_model(conn) or description or "?", conn))
    return out


# ===========================================================================
# Real hardware adapters (vendor SDKs imported lazily)
# ===========================================================================
class KinesisStage(StageBase):
    """Thorlabs Kinesis stage via pylablib. Positions in mm.

    The stage is IDENTIFIED before it is driven: the model number its
    controller reports selects a STAGE_CONFIGS entry, and failing that
    pylablib is asked to calibrate the stage itself. Whichever supplied the
    scale, the units pylablib ends up working in are checked, and anything
    that is not a linear millimetre-convertible unit is refused — a rotational
    or uncalibrated stage driven with somebody else's steps-per-mm would
    quietly distort the delay axis instead of failing.

    Arguments:
      serial — connection string / serial of the device to open (substring
               match against the enumerated devices); default: first device.
      index  — which enumerated device to open when `serial` is None.
      model  — force a STAGE_CONFIGS entry instead of identifying the stage.
               An escape hatch for a controller whose reported model number no
               entry matches; you are then asserting the scale is right.
      backlash_mm — software approach margin (see StageBase). Defaults to 0
               because Kinesis controllers already do this in FIRMWARE: the APT
               "general move parameters" carry a backlash distance which the
               controller applies to every move that would arrive from the
               wrong side (pylablib: get_gen_move_parameters / setup_gen_move).
               Doing it twice would only add travel. Read the firmware value
               with firmware_backlash_mm(); if a controller comes back with 0
               there, set this instead.
    """
    def __init__(self, serial: str | None = None, index: int = 0,
                 model: str | None = None, backlash_mm: float = 0.0):
        from pylablib.devices import Thorlabs

        devices = Thorlabs.list_kinesis_devices()
        if not devices:
            raise RuntimeError("No Kinesis devices found.")
        if serial is not None:
            conn = next((d[0] for d in devices if serial in d[0]), None)
            if conn is None:
                raise RuntimeError(f"Kinesis device with serial {serial!r} not found.")
        else:
            if index >= len(devices):
                raise RuntimeError(
                    f"index {index} out of range ({len(devices)} Kinesis device(s)).")
            conn = devices[index][0]

        reported = _kinesis_model(str(conn))
        if model is not None:
            if model not in STAGE_CONFIGS:
                raise KeyError(
                    f"No calibration for stage model {model!r}. Add an entry to "
                    f"STAGE_CONFIGS in hardware.py. Known: {sorted(STAGE_CONFIGS)}"
                )
            key = model
        else:
            key = _match_stage_config(reported)
        cfg = STAGE_CONFIGS.get(key, {}) if key else {}

        # No entry for this stage -> ask pylablib to calibrate it. That is a
        # request, not an assumption: the units check below is what decides
        # whether the answer is usable.
        motor = Thorlabs.KinesisMotor(conn, scale=cfg.get("scale", "stage"))
        try:
            units = motor.get_scale_units()
            if units == "user":
                mm_per_unit = 1.0        # STAGE_CONFIGS tuple: steps per mm
            elif units == "m":
                mm_per_unit = 1000.0     # pylablib calibration: metres
            else:
                raise RuntimeError(
                    self._no_calibration_msg(conn, reported, units))
            stage_name = motor.get_stage()
            travel_mm = cfg.get("travel_mm")
            if travel_mm is None and stage_name:
                travel_mm = KINESIS_TRAVEL_MM.get(str(stage_name).strip().upper())
        except Exception:
            try:
                motor.close()
            except Exception:
                pass
            raise

        self.model        = key or reported
        self.stage_name   = str(stage_name) if stage_name else None
        self.scale_units  = units
        self.travel_mm    = travel_mm
        self.name         = f"{self._label(key, reported, stage_name)} [{conn}]"
        self.backlash_mm  = float(backlash_mm)
        self._mm_per_unit = mm_per_unit
        self._motor       = motor

    @staticmethod
    def _label(key, reported, stage_name) -> str:
        """Human name for the stage: what it is, and what is driving it."""
        if key:
            return key
        if stage_name and reported and str(stage_name) != reported:
            return f"{stage_name} on {reported}"
        return str(stage_name or reported or "Kinesis stage")

    @staticmethod
    def _no_calibration_msg(conn, reported, units) -> str:
        who = f"{reported} [{conn}]" if reported else str(conn)
        if units == "deg":
            why = ("it is a rotational stage (pylablib reports degrees) — this "
                   "app drives linear delay stages only")
        else:
            why = (f"no STAGE_CONFIGS entry matches it and pylablib cannot "
                   f"calibrate it either (scale units: {units!r})")
        return (f"Stage {who} has no millimetre calibration: {why}. "
                f"Add an entry for it to STAGE_CONFIGS in hardware.py.")

    def firmware_backlash_mm(self) -> float | None:
        """The controller's own backlash-correction distance, or None if it
        cannot be read. Reported only — see backlash_mm in __init__."""
        try:
            p = self._motor.get_gen_move_parameters()
            return float(p.backlash_distance) * self._mm_per_unit
        except Exception:
            return None

    def _move_to_raw(self, position_mm: float) -> None:
        self._motor.move_to(float(position_mm) / self._mm_per_unit)
        self._motor.wait_move()

    def get_position(self) -> float:
        return float(self._motor.get_position()) * self._mm_per_unit

    def home(self) -> None:
        self._motor.home()
        self._motor.wait_for_home()

    def stop(self) -> None:
        self._motor.stop()

    def is_moving(self) -> bool:
        try:
            return bool(self._motor.is_moving())
        except Exception:
            return False

    def disconnect(self) -> None:
        try:
            self._motor.close()
        except Exception:
            pass


class ZaberStage(StageBase):
    """Zaber motorised stage via the zaber-motion library. Positions in mm.

    Zaber devices speak over a serial port. `port` may be:
      * an explicit port name ("COM5", "/dev/ttyUSB0"), or
      * None (the default) to auto-scan the machine's serial ports and use the
        first one that answers with a Zaber device.

    A single serial connection can host a daisy-chain of devices, each with one
    or more axes; `device_index` and `axis_number` pick which axis to drive
    (both default to the first). The connection is kept open for the life of the
    object and closed by disconnect().

    Moves block (wait_until_idle is passed explicitly) so the scan loop can read
    back a trustworthy position immediately afterwards. The travel range is read
    from the axis's own limit.min / limit.max settings when the caller does not
    override it.

    Two things to know about Zaber that this class exists to handle:

    * Unlike a Thorlabs controller, Zaber does NO automatic backlash correction
      — the firmware drives straight to the target from whichever side it is on.
      backlash_mm therefore defaults to 50 um here, so every move arrives from
      the same direction and the marked zero-delay position stays in the same
      frame as a scan sweep. See StageBase for why that matters.

    * get_position() reads the `pos` setting, which on a stepper is the
      TRAJECTORY COUNTER, not a measurement — after a completed move it returns
      the commanded value whether or not the carriage got there. So a readback
      alone can never reveal a stall, a hand-nudge or a lost reference. That is
      what position_fault() is for: it reads the device's warning flags (and the
      encoder, on devices that have one) and reports anything that means the
      position is a lie.
    """
    # Zaber warning/fault flags that invalidate the position readback, mapped to
    # what to tell the user. See zaber_motion.ascii.WarningFlags.
    _FAULT_FLAGS = {
        "FS": "motor stalled and stopped",
        "WS": "motor stalled and recovered",
        "WM": "stage displaced while stationary",
        "WL": "unexpected limit switch trigger",
        "WH": "axis is not homed",
        "WR": "no reference position (axis needs homing)",
        "NC": "stage was moved by manual control (knob/joystick)",
        "FQ": "encoder error",
        "FI": "index error",
        "FE": "limit error",
        "FR": "overdrive limit exceeded",
    }
    # pos vs encoder.pos divergence beyond this means lost steps, not rounding.
    _ENCODER_TOL_MM = 0.010

    def __init__(self, port: str | None = None, device_index: int = 0,
                 axis_number: int = 1, travel_mm: float | None = None,
                 probe_timeout_ms: int = 500, backlash_mm: float = 0.05):
        from zaber_motion import Units
        from zaber_motion.ascii import Connection

        self._Units = Units
        self._conn  = None
        self.backlash_mm = float(backlash_mm)

        conn, device, used_port = self._open(
            port, device_index, probe_timeout_ms, Connection)
        try:
            self._conn   = conn
            self._device = device
            self._axis   = device.get_axis(axis_number)

            if travel_mm is None:
                travel_mm = self._setting("limit.max")
            self.travel_mm = travel_mm
            # limit.min is NOT always 0 — a user-set soft limit protecting the
            # optics is common. Clamping to [0, travel] instead would command
            # moves the firmware rejects.
            self.travel_min_mm = self._setting("limit.min") or 0.0

            # An unhomed Zaber reports a `pos` with no physical meaning, and
            # homing later shifts the whole coordinate frame under any zero
            # marked in the meantime.
            try:
                self.needs_homing = not bool(self._axis.is_homed())
            except Exception:
                self.needs_homing = False

            # Devices with an encoder can be cross-checked against `pos`; those
            # without leave position_fault() relying on the warning flags alone.
            self._has_encoder = self._setting("encoder.pos") is not None

            label = getattr(device, "name", None) or "Zaber"
            axis_tag = f" ax{axis_number}" if device.axis_count > 1 else ""
            self.name = f"{label} [{used_port}{axis_tag}]"
        except Exception:
            self.disconnect()
            raise

    @staticmethod
    def _candidate_ports() -> list[str]:
        try:
            from serial.tools import list_ports
            # Bluetooth-link COM ports block for seconds on open (long enough
            # to look like a hang when auto-scanning) and are never a Zaber.
            return [p.device for p in list_ports.comports()
                    if not p.description.startswith(
                        "Standard Serial over Bluetooth link")]
        except Exception:
            return []

    def _setting(self, name: str, unit=None) -> float | None:
        """Read one axis setting in mm (or `unit`), or None if unsupported.

        Not every setting exists on every Zaber device — encoder.pos only on
        devices with an encoder, limit.min only on some firmware — and asking
        for a missing one raises. A missing setting is information, not an
        error, so it comes back as None.
        """
        try:
            return float(self._axis.settings.get(
                name, self._Units.LENGTH_MILLIMETRES if unit is None else unit))
        except Exception:
            return None

    def _open(self, port, device_index, probe_timeout_ms, Connection):
        """Open the serial connection and return (connection, device, port).

        Each port is probed with a short request timeout so a port with no Zaber
        on it (or the field left blank to auto-scan) fails fast instead of
        blocking the caller for the full 1 s default per port.
        """
        ports = [port] if port else self._candidate_ports()
        if not ports:
            raise RuntimeError(
                "No serial ports found. Pass an explicit port (e.g. 'COM5').")

        last_err = None
        for p in ports:
            conn = None
            try:
                conn = Connection.open_serial_port(p)
                conn.default_request_timeout = int(probe_timeout_ms)
                devices = conn.detect_devices()
                if devices:
                    if device_index >= len(devices):
                        conn.close()
                        raise RuntimeError(
                            f"device_index {device_index} out of range "
                            f"({len(devices)} device(s) on {p}).")
                    # Restore a comfortable timeout now that we have a real device.
                    conn.default_request_timeout = 1000
                    return conn, devices[device_index], p
                conn.close()
            except Exception as e:            # busy / not a Zaber / no reply
                last_err = e
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        if port:
            raise RuntimeError(f"No Zaber device on {port!r}: {last_err}")
        raise RuntimeError(
            f"No Zaber device found on any serial port ({', '.join(ports)}). "
            f"Last error: {last_err}")

    def _move_to_raw(self, position_mm: float) -> None:
        # wait_until_idle is the library default, but StageBase makes blocking a
        # hard contract — say so rather than inherit it.
        self._axis.move_absolute(float(position_mm),
                                 self._Units.LENGTH_MILLIMETRES,
                                 wait_until_idle=True)

    def get_position(self) -> float:
        """Position in mm from the `pos` setting.

        This is the app's coordinate frame; position_fault() is what tells you
        whether to believe it. Deliberately NOT encoder.pos — that has its own
        origin, and switching would silently re-reference every saved zero.
        """
        return float(self._axis.get_position(self._Units.LENGTH_MILLIMETRES))

    def encoder_position(self) -> float | None:
        """Measured position in mm from the encoder, or None if there is none.

        Unlike get_position() this is a measurement, so the two disagreeing
        means the axis lost steps.
        """
        return self._setting("encoder.pos") if self._has_encoder else None

    def position_fault(self) -> str | None:
        try:
            flags = set(self._axis.warnings.get_flags())
        except Exception:
            return None
        reasons = [text for flag, text in self._FAULT_FLAGS.items()
                   if flag in flags]

        enc = self.encoder_position()
        if enc is not None:
            try:
                gap = abs(enc - self.get_position())
            except Exception:
                gap = 0.0
            if gap > self._ENCODER_TOL_MM:
                reasons.append(
                    f"encoder disagrees with the step counter by "
                    f"{gap * 1000.0:.1f} um (lost steps)")
        return "; ".join(reasons) or None

    def clear_position_faults(self) -> None:
        try:
            self._axis.warnings.clear_flags()
        except Exception:
            pass

    def home(self) -> None:
        self._axis.home(wait_until_idle=True)
        self.needs_homing = False

    def stop(self) -> None:
        self._axis.stop()

    def is_moving(self) -> bool:
        try:
            return bool(self._axis.is_busy())
        except Exception:
            return False

    def disconnect(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._conn = None


class PiezoJenaStage(StageBase):
    """piezosystems Jena closed-loop piezo stage over a plain serial port.

    The controller speaks a bare ASCII protocol at 9600 baud ("i1" remote
    mode, "cl" closed loop, "rd" position readback, "wr,<um>" absolute move).
    It works in MICROMETRES over 0..320 um with 0.1 um resolution; this
    adapter converts to the mm contract of StageBase.

    `port` may be an explicit port name ("COM5") or None (the default) to
    probe the machine's serial ports and use the first that answers like a
    Piezo Jena controller: an unknown command is answered with "err,2", which
    no other device on a bench is likely to say.

    "wr" only sets the target — the controller settles on its own — so
    _move_to_raw() polls "rd" until the readback is within tolerance, with a
    hard deadline so a wedged controller raises instead of hanging the GUI.

    backlash_mm stays 0: a closed-loop flexure piezo has no screw and therefore
    no backlash, and "rd" is a real measurement rather than a step counter, so
    every move verifies itself against the stage's own 0.1 um resolution.
    """
    _TOL_UM = 0.1   # settle tolerance = the stage's own resolution

    def __init__(self, port: str | None = None, travel_um: float = 320.0,
                 probe_timeout_s: float = 0.5):
        import serial

        self._ser = None
        self._ser, used_port = self._open(port, probe_timeout_s, serial)
        try:
            self._ser.timeout = 2.0        # working timeout, per readline
            self._command(b"i1")           # remote-control mode
            time.sleep(0.05)
            self._command(b"cl")           # closed-loop mode
            time.sleep(0.05)
            self.travel_mm = travel_um / 1000.0
            self.name = f"Piezo Jena [{used_port}]"
        except Exception:
            self.disconnect()
            raise

    @staticmethod
    def _candidate_ports() -> list[str]:
        try:
            from serial.tools import list_ports
            # Bluetooth-link COM ports block for seconds on open — never a
            # piezo controller, so don't even probe them.
            return [p.device for p in list_ports.comports()
                    if not p.description.startswith(
                        "Standard Serial over Bluetooth link")]
        except Exception:
            return []

    def _open(self, port, probe_timeout_s, serial):
        """Open and identify the controller; return (Serial, port name).

        Each candidate is probed with a deliberately unknown command — a
        Piezo Jena controller answers "err,2", anything else is not ours.
        """
        ports = [port] if port else self._candidate_ports()
        if not ports:
            raise RuntimeError(
                "No serial ports found. Pass an explicit port (e.g. 'COM5').")

        last_err = None
        for p in ports:
            ser = None
            try:
                ser = serial.Serial(p, baudrate=9600,
                                    timeout=probe_timeout_s)
                ser.reset_input_buffer()
                ser.write(b"hello\r")
                if ser.readline().startswith(b"err,2"):
                    return ser, p
                ser.close()
            except Exception as e:            # busy / no reply / not ours
                last_err = e
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
        if port:
            raise RuntimeError(f"No Piezo Jena stage on {port!r}: {last_err}")
        raise RuntimeError(
            f"No Piezo Jena stage found on any serial port "
            f"({', '.join(ports)}). Last error: {last_err}")

    def _command(self, cmd: bytes) -> None:
        self._ser.write(cmd + b"\r\n")

    def _move_to_raw(self, position_mm: float) -> None:
        # Clamp instead of raising: the scan worker pre-checks its targets
        # against travel_mm, so anything out of range here is a manual move.
        target_um = min(max(float(position_mm) * 1000.0, 0.0),
                        self.travel_mm * 1000.0)
        self._command(b"wr,%.2f" % target_um)
        deadline = time.monotonic() + 5.0
        while abs(self.get_position() * 1000.0 - target_um) > self._TOL_UM:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Stage did not settle at {target_um:.2f} um within 5 s "
                    f"(readback {self.get_position() * 1000.0:.2f} um).")
            time.sleep(0.05)

    def get_position(self) -> float:
        # Reply looks like b"rd,123.4\r\n" — the value follows the last comma.
        self._ser.reset_input_buffer()
        self._command(b"rd")
        reply = self._ser.readline()
        try:
            return float(reply[reply.rfind(b",") + 1:]) / 1000.0
        except ValueError:
            raise RuntimeError(
                f"Piezo Jena stage gave no usable position readback "
                f"(reply: {reply!r}).") from None

    def home(self) -> None:
        # No homing routine exists for this closed-loop piezo; raising keeps
        # the GUI's "homed to 0 mm" message from ever lying about it.
        raise RuntimeError("home not defined for this stage")

    def disconnect(self) -> None:
        ser = getattr(self, "_ser", None)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            self._ser = None


# Both python-seabreeze backends, in the order the GUI offers them. pyseabreeze
# first: it is the only backend that supports the newer Ocean Insight models
# (SR/ST/HDX series), while the devices cseabreeze covers also work through it.
SEABREEZE_BACKENDS = ("pyseabreeze", "cseabreeze")


def _ensure_libusb_dll() -> None:
    """pyseabreeze drives USB through pyusb, which on Windows locates
    libusb-1.0.dll by searching PATH — it knows nothing about the pip-installed
    `libusb-package` wheel that actually carries the DLL. Bridge the two by
    prepending the wheel's DLL folder to PATH (idempotent, no-op if the wheel
    is missing or a system-wide libusb already exists)."""
    if sys.platform != "win32":
        return
    try:
        import libusb_package
        dll_dir = str(Path(libusb_package.get_library_path()).parent)
    except Exception:
        return
    if dll_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]


def select_seabreeze_backend(backend: str) -> None:
    """Make `backend` ("pyseabreeze" or "cseabreeze") the active seabreeze
    backend. seabreeze binds its backend lazily on first use and then caches
    it in three places, so switching at runtime means clearing all three;
    otherwise seabreeze.use() only warns and the old backend stays live.
    Any spectrometer opened through the previous backend must already be
    closed — its API is shut down here."""
    if backend not in SEABREEZE_BACKENDS:
        raise ValueError(f"backend must be one of {SEABREEZE_BACKENDS}, "
                         f"got {backend!r}")
    if backend == "pyseabreeze":
        _ensure_libusb_dll()
    import seabreeze
    import seabreeze.backends
    mod = sys.modules.get("seabreeze.spectrometers")
    cached = None
    if mod is not None:
        # The lazy binding lands in either place first depending on whether
        # list_devices() or Spectrometer() ran first — check both.
        descriptor = mod.Spectrometer.__dict__["_backend"]
        cached = mod.__dict__.get("_lib") or descriptor._backend
    if cached is not None and cached.__name__.endswith(backend):
        return                      # already live — nothing to reset
    if cached is not None:
        api = getattr(mod.list_devices, "_api", None)
        if api is not None:
            try:
                api.shutdown()
            except Exception:
                pass
            del mod.list_devices._api
        mod.__dict__.pop("_lib", None)
        mod.__dict__.pop("SeaBreezeDevice", None)
        descriptor._backend = None
    if (seabreeze.backends.BackendConfig.requested != backend
            or cached is not None):
        seabreeze.use(backend)


def list_seabreeze_spectrometers(backend: str = "pyseabreeze") -> list[tuple[str, str]]:
    """Enumerate attached Ocean Optics spectrometers without adopting any.
    Returns [(model, serial), ...]; entries whose metadata cannot be read come
    back as "?" placeholders rather than being dropped. Lives here so the GUI
    never touches the vendor SDK, and so backend selection precedes the
    spectrometers import."""
    select_seabreeze_backend(backend)
    from seabreeze.spectrometers import list_devices
    out = []
    for dev in list_devices():
        try:
            model = str(getattr(dev, "model", "?"))
            serial = str(getattr(dev, "serial_number", "?"))
        except Exception:
            model, serial = "?", "?"
        out.append((model, serial))
    return out


# ===========================================================================
# Vendor-agnostic spectrometer enumeration
# ===========================================================================
# More than one vendor can be on the bench at once, so a bare serial number no
# longer identifies a device: two vendors may legitimately use the same string,
# and nothing in it says which adapter class to open it with. Devices are
# therefore named by a TAGGED id, "vendor:serial".
#
# The tag is deliberately a plain string rather than a dataclass. The id is
# carried through QComboBox userData, stored in the multi-spectrometer slots,
# compared with ==, and printed in the device picker and in status messages —
# a string keeps all of that working and renders as something a human can read.
# The simulated multi-spec halves already use the same idiom (the
# "__sim_blue__" / "__sim_red__" sentinels in the GUI), and those are matched
# before any tag parsing, so they are left exactly as they are.
SPEC_VENDORS = ("seabreeze", "avantes")


def spec_ident(vendor: str, serial) -> str:
    """Build the tagged id for a device. '?' when the serial is unreadable —
    the GUI treats that as "cannot reopen by id, auto-connect instead" rather
    than dropping the device from the list."""
    return f"{vendor}:{serial if serial else '?'}"


def split_spec_ident(ident: str) -> tuple[str, str]:
    """Inverse of spec_ident. Raises on anything that is not a tagged id."""
    vendor, _, serial = str(ident).partition(":")
    if vendor not in SPEC_VENDORS:
        raise RuntimeError(
            f"Unknown spectrometer id {ident!r}: expected "
            f"'<vendor>:<serial>' with vendor one of "
            f"{', '.join(SPEC_VENDORS)}.")
    return vendor, serial


def list_spectrometers(seabreeze_backend: str = "pyseabreeze") -> list[tuple[str, str]]:
    """Every attached spectrometer, from every vendor, as
    [(label, "vendor:serial"), ...] — the same shape list_kinesis_stages uses,
    so the GUI's device picker drives all of them.

    A vendor whose SDK is missing or whose enumeration fails contributes
    NOTHING and raises nothing. This is background enumeration — it fills the
    picker and the multi-spectrometer submenus — and one absent driver must not
    hide the other vendor's devices or turn every submenu into an error. Code
    that wants the failure reported (because the operator explicitly pressed
    that vendor's connect button) calls that vendor's list_* helper directly.
    """
    out: list[tuple[str, str]] = []
    try:
        for model, serial in list_seabreeze_spectrometers(seabreeze_backend):
            out.append((model, spec_ident("seabreeze", serial)))
    except Exception:
        pass
    try:
        for model, serial in list_avantes_spectrometers():
            out.append((model, spec_ident("avantes", serial)))
    except Exception:
        pass
    return out


def open_spectrometer(ident: str,
                      seabreeze_backend: str = "pyseabreeze") -> SpectrometerBase:
    """Open the device a tagged id names. The ONLY place a vendor tag is mapped
    to an adapter class, so a new vendor is one entry here rather than one per
    call site in the GUI."""
    vendor, serial = split_spec_ident(ident)
    if vendor == "seabreeze":
        # "?" = the serial could not be read at enumeration time; fall back to
        # first-device auto-connect rather than refusing outright.
        return (SeabreezeSpectrometer(serial=serial, backend=seabreeze_backend)
                if serial and serial != "?"
                else SeabreezeSpectrometer(backend=seabreeze_backend))
    if vendor == "avantes":
        return AvantesSpectrometer(serial=serial if serial != "?" else None)
    raise RuntimeError(f"No adapter for spectrometer vendor {vendor!r}.")


class SeabreezeSpectrometer(SpectrometerBase):
    """Ocean Optics / Ocean Insight spectrometer via python-seabreeze."""
    def __init__(self, device=None, serial: str | None = None,
                 backend: str = "pyseabreeze"):
        select_seabreeze_backend(backend)
        self.backend = backend      # the GUI checks this before backend swaps
        from seabreeze.spectrometers import Spectrometer, list_devices

        if device is not None:
            self._spec = device if hasattr(device, "intensities") \
                         else Spectrometer(device)
        elif serial is not None:
            self._spec = Spectrometer.from_serial_number(serial)
        else:
            devs = list_devices()
            if not devs:
                raise RuntimeError("No spectrometers found.")
            self._spec = Spectrometer(devs[0])

        self.name = (f"{getattr(self._spec, 'model', 'spectrometer')} "
                     f"[{getattr(self._spec, 'serial_number', '?')}]")
        # Taken from the OPENED device, not from the `serial` argument: the
        # device= and auto-connect paths never see one, and this is the id a
        # multi-spectrometer slot has to be able to reopen later.
        self.spec_id = spec_ident(
            "seabreeze", getattr(self._spec, "serial_number", None))
        self._wl = np.asarray(self._spec.wavelengths(), dtype=float)

        # Detector full scale, as the device itself reports it (seabreeze's
        # `max_intensity`). Guarded: not every backend/model exposes it, and a
        # missing value must degrade to "cannot check" rather than to a wrong
        # threshold that would either cry wolf or stay silent through clipping.
        try:
            self.max_counts = float(self._spec.max_intensity)
        except Exception:
            self.max_counts = None

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def set_integration_time(self, ms: float) -> None:
        self._spec.integration_time_micros(int(ms * 1000))
        self.integration_ms = float(ms)   # keeps the live feed's pacing honest

    def acquire(self) -> np.ndarray:
        return np.asarray(self._spec.intensities(), dtype=float)

    def disconnect(self) -> None:
        try:
            self._spec.close()
        except Exception:
            pass


def list_avantes_spectrometers() -> list[tuple[str, str]]:
    """Enumerate attached Avantes spectrometers without adopting any.
    Returns [(model, serial), ...].

    Raises when the AvaSpec DLL is missing or the enumeration fails — callers
    that want a quiet "nothing found" (background enumeration) go through
    list_spectrometers, which swallows it. The message is worth surfacing when
    the operator explicitly asked for an Avantes: it names the package to
    install.
    """
    import avantes
    return [(name, serial) for name, serial, _status in avantes.list_devices()]


def avantes_trigger_options() -> dict[str, list[tuple[str, int]]]:
    """Labelled choices for the Avantes trigger controls, as
    {"mode"/"source"/"source_type": [(label, value), ...]}.

    Read from the vendor module rather than restated here so the numbers have
    exactly one definition. The GUI calls this only while an Avantes is
    connected, so the lazy import costs nothing on a machine without the DLL —
    and it is what keeps frog_gui_fast from importing avantes directly.
    """
    import avantes
    return {
        "mode": [("Free running", avantes.SW_TRIGGER_MODE),
                 ("Hardware trigger", avantes.HW_TRIGGER_MODE),
                 ("Single scan", avantes.SS_TRIGGER_MODE)],
        "source": [("External", avantes.EXTERNAL_TRIGGER),
                   ("Sync", avantes.SYNC_TRIGGER)],
        "source_type": [("Edge", avantes.EDGE_TRIGGER_SOURCE),
                        ("Level", avantes.LEVEL_TRIGGER_SOURCE)],
    }


class AvantesSpectrometer(SpectrometerBase):
    """Avantes AvaSpec spectrometer via the vendor DLL (see avantes.py).

    The vendor module is imported lazily, inside __init__, so `import hardware`
    and the simulator self-test still work on a machine with no AvaSpec DLL —
    the same arrangement ZaberStage and PiezoJenaStage use for their SDKs.

    Two device settings are entangled with things the rest of the app assumes,
    and this adapter owns both couplings so no consumer has to know about them:

      * `integration_ms` reports the cost of ONE acquire(), which is the
        exposure times the on-board average count — not the exposure. The live
        feed paces itself off this value and derives its handover timeout from
        it, so reporting the bare exposure while a frame silently costs N times
        that makes the feed spin hot and the device-handover time out.
      * `max_counts` follows the ADC resolution. Switching to the 16-bit ADC
        moves full scale from 16383 to 65535, and anything that cached the old
        ceiling would then either cry wolf or stay silent through clipping.
        `set_high_res_adc` returns whether the mode actually changed so the UI
        can re-arm its saturation state only when it did.
    """

    def __init__(self, serial: str | None = None, high_res_adc: bool = True):
        import avantes
        self._spec = avantes.AvaSpec(serial=serial, high_res_adc=high_res_adc)
        try:
            self.serial = self._spec.serial
            self.spec_id = spec_ident("avantes", self.serial)
            self.name = f"{self._spec.model} [{self.serial}]"
            self._wl = np.asarray(self._spec.wavelengths, dtype=float)
            self._sync_derived()
        except Exception:
            self.disconnect()
            raise

    def _sync_derived(self) -> None:
        """Re-read everything the base class contract exposes as plain
        attributes. Called after any change that can move them, so the two
        couplings in the class docstring stay true by construction."""
        self.integration_ms = float(self._spec.frame_time_ms)
        self.max_counts = float(self._spec.max_counts)

    # ── SpectrometerBase ────────────────────────────────────────────────────
    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def set_integration_time(self, ms: float) -> None:
        self._spec.set_integration_time(float(ms))
        self._sync_derived()

    def acquire(self) -> np.ndarray:
        return self._spec.acquire()

    def disconnect(self) -> None:
        spec = getattr(self, "_spec", None)
        if spec is not None:
            try:
                spec.disconnect()
            except Exception:
                pass
            self._spec = None

    # ── Avantes-specific settings ───────────────────────────────────────────
    # Everything below is reached only from the Avantes settings dialog. It is
    # exposed here rather than by handing the GUI the raw avantes.AvaSpec so
    # the one-way frog_gui_fast -> scan -> hardware dependency holds, and so
    # _sync_derived() cannot be forgotten at a call site.
    @property
    def exposure_ms(self) -> float:
        """The exposure alone, without the averaging factor that
        `integration_ms` folds in."""
        return float(self._spec.integration_ms)

    @property
    def n_averages(self) -> int:
        return int(self._spec.n_averages)

    def set_averages(self, n: int) -> None:
        self._spec.set_averages(int(n))
        self._sync_derived()          # frame time just changed

    @property
    def high_res_adc(self) -> bool:
        return bool(self._spec.high_res_adc)

    def set_high_res_adc(self, enable: bool) -> bool:
        """Returns the mode actually in force — False when the hardware has no
        16-bit ADC, which is a fact about the model rather than an error."""
        on = self._spec.set_high_res_adc(bool(enable))
        self._sync_derived()          # full scale just changed
        return on

    def set_dark_correction(self, enable: bool) -> None:
        self._spec.set_dark_correction(bool(enable))

    def set_smoothing(self, pixels: int) -> None:
        self._spec.set_smoothing(int(pixels))

    @property
    def smoothing_pixels(self) -> int:
        return int(self._spec.config.m_Smoothing.m_SmoothPix)

    @property
    def dark_correction(self) -> bool:
        return bool(self._spec.config.m_CorDynDark.m_Enable)

    def set_prescan(self, enable: bool) -> None:
        self._spec.set_prescan_mode(bool(enable))

    def set_sync_mode(self, enable: bool) -> None:
        self._spec.set_sync_mode(bool(enable))

    def set_trigger(self, mode: int, source: int, source_type: int) -> None:
        self._spec.set_trigger(mode, source, source_type)

    @property
    def trigger(self) -> tuple[int, int, int]:
        t = self._spec.config.m_Trigger
        return int(t.m_Mode), int(t.m_Source), int(t.m_SourceType)

    @property
    def hardware_triggered(self) -> bool:
        """True when acquire() waits on an external pulse and may therefore
        block indefinitely. The GUI checks this before leaving a live feed
        running, since a feed that never returns a frame wedges the device
        handover with no visible cause."""
        return bool(self._spec.hardware_triggered)

    def temperature_c(self) -> float | None:
        """Board temperature, or None on a device with no such sensor."""
        return self._spec.temperature_c()

    def saturated_pixels(self) -> np.ndarray:
        """The device's own clipped-pixel mask for the last frame. Independent
        of the count-vs-max_counts test the scan does, and useful as a
        cross-check rather than a replacement."""
        return self._spec.saturated_pixels()

    def device_info(self) -> dict:
        return self._spec.device_info()


class StitchedSpectrometer(SpectrometerBase):
    """Two spectrometers with overlapping wavelength ranges presented as ONE
    device on a common grid.

    spec1 is whichever device reaches the lower maximum wavelength (the
    "bluer" one). Each member's own intensity calibration is applied to its
    raw frame BEFORE interpolation and stitching — that is what makes the two
    halves comparable — so calibrate() on the stitched device itself stays the
    identity and consumers cannot double-apply anything. spec1 is additionally
    scaled by `stitch_factor` (fit or set by hand) to absorb any residual
    sensitivity mismatch; in the overlap the two contributions are averaged.

    The two members hold INDEPENDENT integration times (see
    set_member_integration_time). Frames stay in raw counts, so the exposure
    ratio lands in `stitch_factor` along with the sensitivity ratio: change
    either member's integration time and the factor must be re-fitted, or the
    seam reappears. The GUI flags this rather than compensating silently.

    max_counts is None on purpose: counts on the common grid mix two detectors
    and two calibrations, so no single ADC full scale applies. Saturation goes
    unchecked unless the user sets a Full scale override.
    """
    def __init__(self, spec_a: SpectrometerBase, spec_b: SpectrometerBase):
        wl_a = np.asarray(spec_a.wavelengths, float)
        wl_b = np.asarray(spec_b.wavelengths, float)
        if wl_a.max() <= wl_b.max():
            self.spec1, self.spec2 = spec_a, spec_b
            wl1, wl2 = wl_a, wl_b
        else:
            self.spec1, self.spec2 = spec_b, spec_a
            wl1, wl2 = wl_b, wl_a
        if wl2.min() >= wl1.max():
            raise RuntimeError(
                f"No spectral overlap between {self.spec1.name} "
                f"({wl1.min():.0f}–{wl1.max():.0f} nm) and {self.spec2.name} "
                f"({wl2.min():.0f}–{wl2.max():.0f} nm) — stitching needs "
                f"overlapping wavelength ranges.")
        self._wl1, self._wl2 = wl1, wl2
        # Common grid at the finer of the two pixel spacings (median: real
        # spectrometer grids are not perfectly uniform).
        dwl = min(float(np.median(np.diff(wl1))), float(np.median(np.diff(wl2))))
        self._wl = np.arange(wl1.min(), wl2.max(), dwl)
        #  |-- spec1 only --|== overlap: average ==|-- spec2 only --|
        self._m1  = self._wl <= wl2.min()
        self._m2  = self._wl >= wl1.max()
        self._ovl = ~(self._m1 | self._m2)
        self.stitch_factor = 1.0
        self.name = f"stitched: {self.spec1.name} + {self.spec2.name}"
        self.max_counts = None
        self._sync_integration()
        # RAW (pre-calibration) frames of the most recent acquire, one per
        # member. This is what per-device saturation alarms judge: each frame
        # against its own member's max_counts. Tuple assignment is atomic, so
        # the GUI thread may read this while the feed thread acquires.
        self.last_member_raw: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def members(self) -> tuple[SpectrometerBase, SpectrometerBase]:
        return (self.spec1, self.spec2)

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def member_index(self, member: SpectrometerBase) -> int:
        """0 for spec1 (bluer), 1 for spec2. Identity, not equality — the two
        members are distinct live handles and nothing else can match them."""
        return 0 if member is self.spec1 else 1

    def member_scale(self, index: int) -> float:
        """What acquire() multiplies member `index`'s calibrated frame by.

        The single definition of that factor, so anything drawing the members
        separately lands on the same scale as the combined frame instead of
        re-deriving it and drifting.
        """
        return float(self.stitch_factor) if index == 0 else 1.0

    def _sync_integration(self) -> None:
        """Recompute `integration_ms` from the members' current exposures.

        On a stitched device this is the cost of ONE acquire(), not an
        exposure: acquire() drives the two members sequentially, so a frame
        takes t1+t2. Its only consumers — the live feed's pacing sleep and the
        timeout pause() waits out before handing the hardware over — both want
        exactly that cycle time, and a max() here would under-estimate the
        handover by up to 2x at long integrations.
        """
        self.integration_ms = (float(getattr(self.spec1, "integration_ms", 100.0))
                               + float(getattr(self.spec2, "integration_ms", 100.0)))

    def set_integration_time(self, ms: float) -> None:
        """Give BOTH members the same exposure. The single-value path, used
        when a pair is first connected; see set_member_integration_time for
        the per-device one."""
        self.spec1.set_integration_time(ms)
        self.spec2.set_integration_time(ms)
        self._sync_integration()

    def set_member_integration_time(self, index: int, ms: float) -> None:
        """Expose one member independently of the other.

        The two devices usually see very different signal levels, so a single
        shared exposure means one of them is always either buried in read
        noise or clipped. NOTE that this invalidates `stitch_factor` — frames
        stay in raw counts, so the factor carries the exposure ratio too.
        """
        self.members[index].set_integration_time(ms)
        self._sync_integration()

    def _acquire_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """One frame from each member, calibrated, on their native grids."""
        raw1 = np.asarray(self.spec1.acquire(), float)
        raw2 = np.asarray(self.spec2.acquire(), float)
        self.last_member_raw = (raw1, raw2)
        return self.spec1.calibrate(raw1), self.spec2.calibrate(raw2)

    def acquire(self) -> np.ndarray:
        i1, i2 = self._acquire_pair()
        i1 = i1 * self.member_scale(0)
        out = np.empty_like(self._wl)
        out[self._m1] = np.interp(self._wl[self._m1], self._wl1, i1)
        out[self._m2] = np.interp(self._wl[self._m2], self._wl2, i2)
        o = self._ovl
        out[o] = 0.5 * (np.interp(self._wl[o], self._wl1, i1)
                        + np.interp(self._wl[o], self._wl2, i2))
        return out

    def fit_stitch_factor(self) -> float:
        """Take one frame from each member and choose the factor that makes
        spec1 match spec2 over the overlap, least-squares. The residual is
        quadratic in the factor, so the minimum is the closed form
        s = sum(I1*I2) / sum(I1^2) — no iterative optimiser needed.

        The fit is over raw counts, so the result is the sensitivity ratio
        TIMES the exposure ratio t2/t1. That makes it exposure-dependent: any
        change to either member's integration time leaves it stale and it has
        to be re-fitted.
        """
        i1, i2 = self._acquire_pair()
        o = self._ovl
        a = np.interp(self._wl[o], self._wl1, i1)
        b = np.interp(self._wl[o], self._wl2, i2)
        denom = float(np.dot(a, a))
        if not np.isfinite(denom) or denom <= 0.0:
            raise RuntimeError(
                "No signal in the overlap region — cannot fit a stitch factor.")
        self.stitch_factor = float(np.dot(a, b)) / denom
        return self.stitch_factor

    def calibration_targets(self) -> list[SpectrometerBase]:
        return [self.spec1, self.spec2]

    def disconnect(self) -> None:
        for s in (self.spec1, self.spec2):
            try:
                s.disconnect()
            except Exception:
                pass


# ===========================================================================
# Simulated hardware (drop-in test doubles, no SDK required)
# ===========================================================================
class SimulatedStage(StageBase):
    """In-memory delay stage in mm, with optional settle time.

    physical_backlash_mm models real lead-screw slack: the carriage only
    follows the nut after the slack has been taken up, so a target reached
    travelling downward lands physical_backlash_mm short of the same target
    reached travelling upward. Zero (the default) is an ideal stage.

    This is what makes the zero-shift bug reproducible without hardware: mark
    zero after arriving from one side, sweep a scan from the other, and the
    trace comes back offset. Setting backlash_mm (the *correction*) to anything
    larger than physical_backlash_mm makes the offset go away.
    """
    def __init__(self, settle_s: float = 0.0, travel_mm: float = 300.0,
                 physical_backlash_mm: float = 0.0, backlash_mm: float = 0.0):
        self._pos      = 0.0     # where the nut is  (= commanded)
        self._true_pos = 0.0     # where the carriage actually is
        self.settle_s  = settle_s
        self.travel_mm = travel_mm
        self.backlash_mm = float(backlash_mm)
        self.physical_backlash_mm = float(physical_backlash_mm)
        self.name      = "simulated stage"

    def _move_to_raw(self, position_mm: float) -> None:
        target = float(position_mm)
        if target != self._pos:      # a no-op command moves nothing, slack included
            # Arriving upward, the carriage is pushed flush with the nut;
            # arriving downward it trails by the slack.
            self._true_pos = (target if target > self._pos
                              else target - self.physical_backlash_mm)
            self._pos = target
        if self.settle_s:
            time.sleep(self.settle_s)

    def get_position(self) -> float:
        return self._true_pos

    def home(self) -> None:
        self.move_to(0.0)


# ---------------------------------------------------------------------------
# Pulse-shape library ("beams") for the simulated spectrometer
# ---------------------------------------------------------------------------
# Each builder takes the time grid (fs) and the transform-limited sech
# parameter tau0 (fs) and returns the COMPLEX baseband envelope A(t). The
# carrier and the nonlinear gate are applied by SimulatedSpectrometer, so a
# builder only ever describes the pulse itself.
#
# Every knob below is expressed as a multiple of tau0, so a preset keeps its
# character (stretch factor, number of lobes, ...) at any pulse duration.
#
# Fourier convention: A(t) = sum_Omega Atilde(Omega)·exp(+i·Omega·t), matching
# numpy's ifft. Spectral phase is therefore applied as exp(+i·phi(Omega)).
# An SHG trace is symmetric in delay, so it cannot distinguish +phi2 from
# -phi2 — the sign convention only matters for the PG gate.

def _sech(t: np.ndarray, tau0: float) -> np.ndarray:
    return (1.0 / np.cosh(t / tau0)).astype(complex)


def _gauss(t: np.ndarray, tau0: float) -> np.ndarray:
    """Gaussian with the same intensity FWHM as sech(t/tau0) — 1.7627·tau0."""
    fwhm = 1.7627 * tau0
    return np.exp(-2.0 * np.log(2.0) * (t / fwhm) ** 2).astype(complex)


def _spectral_phase(A: np.ndarray, dt: float,
                    phi2: float = 0.0, phi3: float = 0.0) -> np.ndarray:
    """Apply GDD (fs^2) and TOD (fs^3) to a field."""
    Om = 2 * np.pi * np.fft.fftfreq(A.size, d=dt)
    phase = 0.5 * phi2 * Om ** 2 + phi3 * Om ** 3 / 6.0
    return np.fft.ifft(np.fft.fft(A) * np.exp(1j * phase))


def _fourier_shift(A: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """A(t - tau), by Fourier shift, with the wrapped tail zeroed.

    The shift theorem is exact for arbitrarily structured complex fields (an
    interpolated shift would smear a strongly chirped one), but it is cyclic:
    without the mask a pulse shifted past the edge of the grid would reappear
    on the other side and fake a signal at large delay.
    """
    Om = 2 * np.pi * np.fft.fftfreq(A.size, d=t[1] - t[0])
    out = np.fft.ifft(np.fft.fft(A) * np.exp(-1j * Om * tau))
    if tau > 0:
        out[t < t[0] + tau] = 0.0
    elif tau < 0:
        out[t > t[-1] + tau] = 0.0
    return out


def _energy_window(x: np.ndarray, weight: np.ndarray,
                   frac: float = 0.995) -> tuple[float, float]:
    """The x-range holding `frac` of the weight, split evenly between tails.

    Used instead of a fixed "above 1e-3 of peak" cut because the presets differ
    by orders of magnitude in how fast their wings fall off: an absolute floor
    reads a sech's exponential tail as a real feature and blows the window up.
    """
    cum = np.cumsum(weight)
    if cum[-1] <= 0:
        return float(x[0]), float(x[-1])
    cum = cum / cum[-1]
    tail = 0.5 * (1.0 - frac)
    return float(np.interp(tail, cum, x)), float(np.interp(1.0 - tail, cum, x))


def _split_step(A: np.ndarray, t: np.ndarray, phi2: float, b_int: float,
                n_steps: int = 200) -> np.ndarray:
    """Propagate A through a nonlinear, dispersive medium (symmetric split-step).

    Parametrised by what actually shapes the output rather than by fibre
    constants: `phi2` is the GDD accumulated over the whole length (fs^2) and
    `b_int` the peak nonlinear phase, the B-integral (rad).
    """
    Om   = 2 * np.pi * np.fft.fftfreq(A.size, d=t[1] - t[0])
    half = np.exp(1j * 0.25 * (phi2 / n_steps) * Om ** 2)   # half a linear step
    peak = np.abs(A).max() ** 2 or 1.0
    for _ in range(n_steps):
        A = np.fft.ifft(np.fft.fft(A) * half)
        A = A * np.exp(1j * (b_int / n_steps) * np.abs(A) ** 2 / peak)
        A = np.fft.ifft(np.fft.fft(A) * half)
    return A


def _build_tl_sech(t, tau0):
    return _sech(t, tau0)


def _build_tl_gauss(t, tau0):
    return _gauss(t, tau0)


def _build_chirp(t, tau0):
    # +3·tau0^2 of GDD stretches the pulse ~6x: the trace grows along delay
    # while each spectral slice stays narrow.
    return _spectral_phase(_sech(t, tau0), t[1] - t[0], phi2=3.0 * tau0 ** 2)


def _build_tod(t, tau0):
    # Pure cubic phase — a sharp leading edge trailed by a decaying oscillatory
    # ripple, which prints as a fan of satellites in the trace.
    #
    # Pure cubic phase gives an Airy-like profile whose satellite-to-peak ratio
    # is universal — raising phi3 only stretches the train in time, it never
    # makes the satellites brighter. 3·tau0^3 therefore keeps them compact
    # enough to sit inside a sane delay scan; they are ~1% of the peak, so the
    # trace needs a compressed colour scale to show them off.
    #
    # Cubic phase is odd in Omega, so it preserves the field's Hermitian
    # symmetry: A(t) comes out real. The pulse is still badly distorted — the
    # "chirp" lives in the sign flips between satellites, not in a frequency
    # sweep — which is exactly why the trace is the interesting part.
    return _spectral_phase(_sech(t, tau0), t[1] - t[0], phi3=3.0 * tau0 ** 3)


def _build_spm(t, tau0):
    # Instantaneous Kerr phase, no dispersion: the envelope is untouched but
    # the spectrum splits into the textbook multi-lobed SPM comb.
    A = _sech(t, tau0)
    return A * np.exp(1j * 4.5 * np.abs(A) ** 2)


def _build_fiber(t, tau0):
    # SPM plus normal GVD -> optical wave breaking: a near-rectangular pulse
    # with ripples on both edges and a hard-edged, fringed trace.
    #
    # What matters is the soliton number, N^2 = LD/LNL = tau0^2·B/phi2 ~ 25
    # here. At the earlier N^2 ~ 2 the pulse merely chirped and the trace was
    # indistinguishable from the plain GDD preset.
    return _split_step(_sech(t, tau0), t, phi2=0.4 * tau0 ** 2, b_int=10.0)


def _build_double(t, tau0):
    # Two near-equal replicas 6·tau0 apart with a quadrature phase offset:
    # three lobes along delay, all carrying spectral interference fringes.
    A   = _sech(t, tau0)
    sep = 6.0 * tau0
    return (_fourier_shift(A, t, -0.5 * sep)
            + 0.9 * np.exp(0.5j * np.pi) * _fourier_shift(A, t, +0.5 * sep))


def _build_two_color(t, tau0):
    # Two sub-pulses at different centre frequencies, separated in time. SHG
    # mixes them into three bands (2w1, w1+w2, 2w2) at three delays — the
    # off-diagonal peaks that only a 2-D measurement can show.
    A   = _sech(t, 1.3 * tau0)
    sep = 3.0 * tau0
    dw  = 1.2 / tau0                      # rad/fs, about one TL bandwidth
    return (_fourier_shift(A, t, -0.5 * sep) * np.exp(+1j * dw * t)
            + _fourier_shift(A, t, +0.5 * sep) * np.exp(-1j * dw * t))


# Ordered simple -> structured; the GUI builds its picker straight from this.
PULSE_SHAPES: dict[str, dict] = {
    "tl_sech": {
        "label": "Transform-limited sech²",
        "build": _build_tl_sech,
        "desc":  "Clean soliton pulse, flat phase. The reference trace: a "
                 "symmetric blob with no tilt or structure.",
    },
    "tl_gauss": {
        "label": "Transform-limited Gaussian",
        "build": _build_tl_gauss,
        "desc":  "Same FWHM as the sech, but with Gaussian wings — a slightly "
                 "tighter trace with no pedestal.",
    },
    "chirp": {
        "label": "Linearly chirped (GDD)",
        "build": _build_chirp,
        "desc":  "+3·τ₀² of GDD (~6x stretch). Long in delay, unchanged in "
                 "bandwidth — the classic 'this pulse is not compressed' trace.",
    },
    "tod": {
        "label": "Third-order dispersion",
        "build": _build_tod,
        "desc":  "Pure cubic phase: a steep front edge trailed by an Airy "
                 "satellite train. The satellites sit ~1% below the peak, so "
                 "they need a compressed colour scale to see.",
    },
    "spm": {
        "label": "Self-phase modulation",
        "build": _build_spm,
        "desc":  "B-integral of 4.5 rad, no dispersion. Envelope unchanged, "
                 "spectrum split into the multi-lobed SPM comb.",
    },
    "fiber": {
        "label": "Fibre output (SPM + normal GVD)",
        "build": _build_fiber,
        "desc":  "Split-step propagation past optical wave breaking (N²≈25). "
                 "Hard-edged diamond with interference fringes inside — the "
                 "busiest trace in the set.",
    },
    "double": {
        "label": "Double pulse",
        "build": _build_double,
        "desc":  "Two replicas 6·τ₀ apart in quadrature: three lobes along "
                 "delay, all striped with spectral interference fringes.",
    },
    "two_color": {
        "label": "Two-colour pair",
        "build": _build_two_color,
        "desc":  "Two sub-pulses at different colours and different times. SHG "
                 "mixes them into three off-diagonal bands.",
    },
    "saturated": {
        "label": "Over-exposed sech² (clips)",
        "build": _build_tl_sech,
        # `exposure` is a property of the MEASUREMENT, not of the field: it
        # scales peak_counts, so the same clean sech pulse is simply driven
        # 25x too hard and clips against max_counts. Here to exercise the
        # saturation warning — flat-topped spectrum, ruined trace.
        "exposure": 25.0,
        "desc":  "A clean sech pulse deliberately over-exposed 25x. The core "
                 "clips flat against full scale and the saturation lamp goes "
                 "red — turn the integration time down to recover it.",
    },
}

DEFAULT_PULSE = "tl_sech"


class SimulatedSpectrometer(SpectrometerBase):
    """Synthetic FROG spectrometer for offline testing.

    Each acquire() reads the stage's *actual* mm position, converts it to a
    delay via `position_to_delay` (injected by the app so the simulator sees
    the same zero-delay as the real optics), and returns the corresponding
    FROG spectrum. Defaults to a transform-limited sech pulse, SHG gate.

    `pulse` selects the beam under test: a key of PULSE_SHAPES, or any
    callable(t_fs, tau0_fs) -> complex envelope for a one-off field. Because
    the envelope is complex, the spectral phase is what shapes the trace —
    that is the whole point of the chirp / TOD / SPM presets.

    Gate-agnostic by design:
        shg : E_sig = E(t)·E(t-tau)        -> signal at 2*omega0
        pg  : E_sig = E(t)·|E(t-tau)|^2    -> signal at  omega0
    """
    def __init__(self, stage: StageBase, position_to_delay=None,
                 gate: str = "shg", lambda0_nm: float = 1030.0,
                 tau0_fs: float = 30.0, pulse=DEFAULT_PULSE,
                 wl_start: float | None = None, wl_end: float | None = None,
                 n_pixels: int = 1024, n_time: int = 4096,
                 t_max_fs: float = 1024.0,
                 peak_counts: float = 4000.0, read_noise: float = 15.0,
                 background_counts: float = 0.0,
                 max_counts: float | None = 65535.0):
        self.stage             = stage
        self.gate              = gate
        self.tau0              = tau0_fs
        self.read_noise        = read_noise
        # 16-bit full scale, like the Ocean Optics units. Pass None for the old
        # unbounded behaviour (nothing ever clips, nothing ever warns).
        self.max_counts        = max_counts
        # Delay-independent pedestal hook. The *realistic* (delay-dependent,
        # single-arm) background term depends on the background strategy we
        # settle on, so this stays a simple constant for now.
        self.background_counts = background_counts
        self.integration_ms    = 100.0

        if callable(pulse):
            self.pulse, build   = "custom", pulse
            self.pulse_label    = getattr(pulse, "__name__", "custom pulse")
            self.pulse_desc     = ""
            self.exposure       = 1.0
        else:
            if pulse not in PULSE_SHAPES:
                raise KeyError(f"Unknown pulse {pulse!r}. "
                               f"Known: {sorted(PULSE_SHAPES)}")
            shape = PULSE_SHAPES[pulse]
            self.pulse, build   = pulse, shape["build"]
            self.pulse_label    = shape["label"]
            self.pulse_desc     = shape["desc"]
            self.exposure       = float(shape.get("exposure", 1.0))
        # Presets may ask to be driven harder than full scale (see "saturated").
        self.peak_counts = peak_counts * self.exposure
        self.name = f"simulated {gate}-FROG — {self.pulse_label}"

        if position_to_delay is None:
            # Default double-pass conversion, zero at 0 mm. The app overrides
            # this with the user's marked zero-delay offset.
            position_to_delay = lambda x_mm: 2.0 * x_mm * 1e6 / C_NM_PER_FS
        self._pos_to_delay = position_to_delay

        omega0       = 2 * np.pi * C_NM_PER_FS / lambda0_nm
        self.carrier = 2 * omega0 if gate == "shg" else omega0

        # Time grid + the beam under test, as a complex envelope A(t).
        self.t  = np.linspace(-t_max_fs, t_max_fs, n_time)
        self.dt = self.t[1] - self.t[0]
        self.A  = np.asarray(build(self.t, tau0_fs), dtype=complex)

        self._Omega = 2 * np.pi * np.fft.fftfreq(n_time, d=self.dt)
        self._Afft  = np.fft.fft(self.A)
        omega = self.carrier + self._Omega
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(omega > 0, 2 * np.pi * C_NM_PER_FS / omega, np.inf)
        self._order      = np.argsort(lam)
        self._lam_sorted = lam[self._order]

        # How far the pulse actually reaches in time (99% of its energy).
        lo, hi = _energy_window(self.t, np.abs(self.A) ** 2, frac=0.99)
        self.pulse_span_fs = max(abs(lo), abs(hi), 2.0 * tau0_fs)
        # A delay scan has to cover the correlation, which is wider than the
        # pulse itself. Offered to the UI as a hint; nothing here enforces it.
        self.suggested_delay_fs = float(np.ceil(1.5 * self.pulse_span_fs / 25) * 25)

        # Structured beams are far broader than a TL pulse (and PG sits at the
        # fundamental, not the second harmonic), so the window is sized from
        # the signal itself unless the caller pinned it.
        auto_lo, auto_hi = self._spectral_extent()
        self._wl = np.linspace(auto_lo if wl_start is None else wl_start,
                               auto_hi if wl_end is None else wl_end, n_pixels)

        # peak_counts is the peak of the whole TRACE — for a double pulse the
        # brightest column is not the one at zero delay.
        norm = max(self._raw_column(tau).max() for tau in self._delay_samples())
        self._norm = norm if norm > 0 else 1.0

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def set_integration_time(self, ms: float) -> None:
        self.integration_ms = float(ms)

    def _delay_samples(self, n: int = 21) -> np.ndarray:
        """Delays spanning the pulse — enough to characterise the whole trace."""
        return np.linspace(-self.pulse_span_fs, self.pulse_span_fs, n)

    def _signal_spectrum(self, tau: float) -> np.ndarray:
        """|E_sig(omega, tau)|^2 on the FFT's own (wavelength-sorted) grid."""
        A_shift = _fourier_shift(self.A, self.t, tau)
        sig = (self.A * A_shift if self.gate == "shg"
               else self.A * np.abs(A_shift) ** 2)
        return (np.abs(np.fft.fft(sig)) ** 2)[self._order]

    def _spectral_extent(self, frac: float = 0.995,
                         pad: float = 0.35) -> tuple[float, float]:
        """Wavelength window holding the delay-integrated signal, plus margin."""
        marg = sum(self._signal_spectrum(tau) for tau in self._delay_samples())
        finite = np.isfinite(self._lam_sorted)
        lo, hi = _energy_window(self._lam_sorted[finite], marg[finite], frac)
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        half = max(half * (1.0 + pad), 2.0)                 # never absurdly tight
        return mid - half, mid + half

    def _raw_column(self, tau: float) -> np.ndarray:
        return np.interp(self._wl, self._lam_sorted, self._signal_spectrum(tau),
                         left=0.0, right=0.0)

    def acquire(self) -> np.ndarray:
        tau    = self._pos_to_delay(self.stage.get_position())   # read-back!
        scale  = self.integration_ms / 100.0                     # signal ∝ integ. time
        counts = self._raw_column(tau) / self._norm * self.peak_counts * scale
        counts = counts + self.background_counts * scale
        counts = np.random.poisson(np.clip(counts, 0, None)).astype(float)
        counts += np.random.normal(0.0, self.read_noise, size=counts.shape)  # read noise: fixed
        # The ADC ceiling is applied LAST, after noise, so a clipped pixel sits
        # at exactly full scale with no noise on top — which is what makes
        # saturation detectable (and what makes the data unrecoverable).
        return np.clip(counts, 0, self.max_counts)


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == "__main__":
    stage = SimulatedStage()
    print("Self-test — simulated SHG-FROG (no hardware)")
    for key in PULSE_SHAPES:
        spec = SimulatedSpectrometer(stage, gate="shg", pulse=key)
        stage.move_to(0.0)
        col = spec.acquire()
        print(f"  {spec.pulse_label:<32s} "
              f"{spec.wavelengths[0]:6.1f}–{spec.wavelengths[-1]:6.1f} nm   "
              f"scan ±{spec.suggested_delay_fs:4.0f} fs   "
              f"peak {col.max():7.1f} counts")

    # Long-move split: a 47 mm move at the 5 mm default is 10 sub-moves, and the
    # stage still lands exactly on target. A short one stays a single move.
    stage = SimulatedStage(travel_mm=300.0)
    raw = [0]
    stage._move_to_raw = (lambda f: lambda mm: (raw.__setitem__(0, raw[0] + 1), f(mm))[1]
                          )(stage._move_to_raw)
    stage.move_to(47.0)
    assert raw[0] == 10, f"expected 10 sub-moves for 47 mm, got {raw[0]}"
    assert stage.get_position() == 47.0, stage.get_position()
    raw[0] = 0
    stage.move_to(45.0)
    assert raw[0] == 1, f"expected 1 move for 2 mm, got {raw[0]}"
    print(f"  long-move split: 47 mm -> {10} steps of <=5 mm, 2 mm -> 1 step")

    # Tagged spectrometer ids. All of this runs with no vendor SDK installed
    # and no hardware attached — which is the point: enumeration must degrade
    # to "found nothing" rather than raising, or a missing driver takes the
    # device picker and every multi-spectrometer submenu down with it.
    assert spec_ident("seabreeze", "USB2+H12345") == "seabreeze:USB2+H12345"
    assert spec_ident("seabreeze", None) == "seabreeze:?"   # unreadable serial
    assert split_spec_ident("seabreeze:USB2+H12345") == ("seabreeze", "USB2+H12345")
    assert split_spec_ident("avantes:2001234") == ("avantes", "2001234")
    for bad in ("USB2+H12345", "ocean:123", "", "__sim_blue__"):
        try:
            split_spec_ident(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not parse as a tagged id")
    devices = list_spectrometers()          # must not raise without an SDK
    assert isinstance(devices, list)
    assert all(len(d) == 2 and split_spec_ident(d[1]) for d in devices)
    print(f"  tagged spectrometer ids: ok ({len(devices)} device(s) enumerated)")

    # A vendor whose SDK is absent must fail with the message that names what
    # to install, not with an ImportError or a bare OSError — that string is
    # what the operator reads in the Hardware dialog.
    try:
        open_spectrometer("avantes:2001234")
    except Exception as e:
        # Printed whole, not truncated: the point of the check is that this
        # sentence tells an operator what to install.
        print(f"  avantes without its DLL -> {type(e).__name__}: {e}")
    else:
        print("  avantes connected (DLL present)")