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
import re
import time
import numpy as np

C_NM_PER_FS = 299.792458   # speed of light, nm/fs (used only by the simulator)


# ===========================================================================
# Abstract interfaces — implement these to add new hardware
# ===========================================================================
class StageBase(abc.ABC):
    """A 1-D motorised delay stage. All positions are in MILLIMETRES.

    Absolute/relative moves MUST block until the stage has settled, so the
    scan loop can read back a trustworthy position immediately afterwards.
    """
    units = "mm"
    name  = "stage"
    travel_mm: float | None = None

    @abc.abstractmethod
    def move_to(self, position_mm: float) -> None:
        """Absolute move to position_mm; block until settled."""

    @abc.abstractmethod
    def move_by(self, delta_mm: float) -> None:
        """Relative move by delta_mm; block until settled."""

    @abc.abstractmethod
    def get_position(self) -> float:
        """Read back the *actual* current position in mm."""

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

    def disconnect(self) -> None:
        pass


# ===========================================================================
# Stage calibration registry
# ===========================================================================
# A Kinesis stage can get its millimetre calibration two ways, tried in order:
#
#   1. An entry below, selected by matching the model number the CONTROLLER
#     reports against that entry's `models` patterns. This is for stages
#     pylablib cannot calibrate itself — the LTS300C/M is one: its controller
#     reports no stage ID, so pylablib would silently fall back to raw steps.
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
    # Example — the LTS150 shares the LTS300's controller and step scale and
    # differs only in travel (adapt / uncomment):
    # "LTS150C/M": {
    #     "models": (r"LTS150",),
    #     "scale": (409600, 21990232, 4506),
    #     "travel_mm": 150.0,
    # },
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
    """
    def __init__(self, serial: str | None = None, index: int = 0,
                 model: str | None = None):
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

    def move_to(self, position_mm: float) -> None:
        self._motor.move_to(float(position_mm) / self._mm_per_unit)
        self._motor.wait_move()

    def move_by(self, delta_mm: float) -> None:
        self._motor.move_by(float(delta_mm) / self._mm_per_unit)
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

    Moves block (zaber-motion's wait_until_idle) so the scan loop can read back a
    trustworthy position immediately afterwards. Travel is read from the axis's
    own limit.max setting when the caller does not override it.
    """
    def __init__(self, port: str | None = None, device_index: int = 0,
                 axis_number: int = 1, travel_mm: float | None = None,
                 probe_timeout_ms: int = 500):
        from zaber_motion import Units
        from zaber_motion.ascii import Connection

        self._Units = Units
        self._conn  = None

        conn, device, used_port = self._open(
            port, device_index, probe_timeout_ms, Connection)
        try:
            self._conn   = conn
            self._device = device
            self._axis   = device.get_axis(axis_number)

            if travel_mm is None:
                try:
                    travel_mm = float(self._axis.settings.get(
                        "limit.max", Units.LENGTH_MILLIMETRES))
                except Exception:
                    travel_mm = None
            self.travel_mm = travel_mm

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
            return [p.device for p in list_ports.comports()]
        except Exception:
            return []

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

    def move_to(self, position_mm: float) -> None:
        self._axis.move_absolute(float(position_mm),
                                 self._Units.LENGTH_MILLIMETRES)

    def move_by(self, delta_mm: float) -> None:
        self._axis.move_relative(float(delta_mm),
                                 self._Units.LENGTH_MILLIMETRES)

    def get_position(self) -> float:
        return float(self._axis.get_position(self._Units.LENGTH_MILLIMETRES))

    def home(self) -> None:
        self._axis.home()

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


def list_seabreeze_spectrometers(backend: str = "cseabreeze") -> list[tuple[str, str]]:
    """Enumerate attached Ocean Optics spectrometers without adopting any.
    Returns [(model, serial), ...]; entries whose metadata cannot be read come
    back as "?" placeholders rather than being dropped. Lives here so the GUI
    never touches the vendor SDK, and so seabreeze.use() precedes the
    spectrometers import."""
    import seabreeze
    seabreeze.use(backend)
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


class SeabreezeSpectrometer(SpectrometerBase):
    """Ocean Optics / Ocean Insight spectrometer via python-seabreeze."""
    def __init__(self, device=None, serial: str | None = None,
                 backend: str = "cseabreeze"):
        import seabreeze
        seabreeze.use(backend)
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


# ===========================================================================
# Simulated hardware (drop-in test doubles, no SDK required)
# ===========================================================================
class SimulatedStage(StageBase):
    """In-memory delay stage in mm, with optional settle time."""
    def __init__(self, settle_s: float = 0.0, travel_mm: float = 300.0):
        self._pos      = 0.0
        self.settle_s  = settle_s
        self.travel_mm = travel_mm
        self.name      = "simulated stage"

    def move_to(self, position_mm: float) -> None:
        self._pos = float(position_mm)
        if self.settle_s:
            time.sleep(self.settle_s)

    def move_by(self, delta_mm: float) -> None:
        self.move_to(self._pos + float(delta_mm))

    def get_position(self) -> float:
        return self._pos

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