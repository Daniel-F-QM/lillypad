"""Thorlabs Kinesis / APT motion stages, via pylablib.

The vendor SDK is imported INSIDE the functions that need it, never at module
scope, so importing this module costs nothing on a machine with no Thorlabs
software installed. The import is a literal statement in a function body, which
is also what lets PyInstaller's analysis find it with no hiddenimports entry.
"""

from __future__ import annotations

import re
import time

from lillypad.hardware.base import StageBase

# APT message the controller sends when a home completes (MGMSG_MOT_MOVE_HOMED).
_MOT_MOVE_HOMED = 0x0444

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
            # The controller keeps its "homed" status bit for as long as it
            # stays powered, so this is only False after a power cycle — not
            # on every connect. Until then the position reads 0 wherever the
            # carriage sits, and homing later shifts the whole frame under any
            # zero marked in the meantime.
            try:
                needs_homing = not bool(motor.is_homed())
            except Exception:
                needs_homing = False
        except Exception:
            try:
                motor.close()
            except Exception:
                pass
            raise

        self.conn         = str(conn)
        self.model        = key or reported
        self.stage_name   = str(stage_name) if stage_name else None
        self.scale_units  = units
        self.travel_mm    = travel_mm
        self.name         = f"{self._label(key, reported, stage_name)} [{conn}]"
        self.backlash_mm  = float(backlash_mm)
        self.needs_homing = needs_homing
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

    def home(self, timeout_s: float = 300.0) -> None:
        """Home the axis and block until the carriage has finished.

        force=True: pylablib's default skips the move entirely when the
        controller already reports homed, which made Home a silent no-op while
        the status bar announced a fresh home. But forcing it breaks pylablib's
        own wait, which polls for the "homed" bit — on an axis that was already
        homed that bit is still set from last time, so the wait returned with
        the carriage still travelling. Completion is therefore taken from the
        controller's MOT_MOVE_HOMED message (pylablib counts it as a background
        message on every read), with the status bits as a fallback for a
        controller that never sends it.

        The timeout is generous because homing is slow: an LTS300 seeking its
        switch at the 2 mm/s factory homing velocity takes 150 s end to end.
        """
        motor = self._motor
        sent_before, _ = motor.check_background_comm(_MOT_MOVE_HOMED)
        motor.home(sync=False, force=True)
        deadline = time.monotonic() + timeout_s
        seen_homing = False
        while True:
            status = motor.get_status()         # drains queued messages too
            sent_now, _ = motor.check_background_comm(_MOT_MOVE_HOMED)
            homing = "homing" in status
            seen_homing = seen_homing or homing
            if not homing and (sent_now > sent_before
                               or (seen_homing and "homed" in status)):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"{self.name}: homing did not finish within {timeout_s:.0f} s.")
            time.sleep(0.05)
        self.needs_homing = False

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
