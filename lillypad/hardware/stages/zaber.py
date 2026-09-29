"""Zaber ASCII motion controllers, over a serial daisy-chain.

zaber_motion and pyserial are imported inside the methods that need them — see
kinesis.py for why that matters to the frozen build.
"""

from __future__ import annotations

from lillypad.hardware.base import StageBase

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
        """Position in mm, as the library reports it (Zaber's `pos`).

        On a stepper that is the trajectory counter, not a measurement: after a
        completed move it returns what was commanded whether or not the carriage
        got there. This is the app's coordinate frame; position_fault() is what
        tells you whether to believe it. Deliberately NOT encoder.pos — that has
        its own origin, and switching would silently re-reference every saved
        zero.
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
