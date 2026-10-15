"""Abstract hardware interfaces — implement these to add a new device.

Everything in this module is vendor-free and SDK-free. It imports numpy and the
standard library and nothing else, so it can be read, tested and reasoned about
without a spectrometer on the bench.

Unit conventions
----------------
  * Stage positions are ALWAYS in millimetres. The stage knows nothing about
    femtoseconds; the optics layer (lillypad.scan) owns the
    delay(fs) <-> position(mm) conversion and the zero-delay offset.
  * Spectrometer wavelengths are in nm; spectra are raw counts vs pixel.

To add a device, subclass StageBase or SpectrometerBase in
lillypad/hardware/stages/ or lillypad/hardware/spectrometers/, and register it
in registry.open_spectrometer. Moves must block until the stage has settled, so
the scan loop can read back a position it can trust on the next line.
"""

from __future__ import annotations

import abc
import math
import sys

import numpy as np
from collections.abc import Callable
from pathlib import Path

# Speed of light, nm/fs. Used by the simulator here AND by scan.py's
# delay <-> position conversions, which are the load-bearing ones: change this
# and every delay axis the app records moves with it.
C_NM_PER_FS = 299.792458

# Fraction of the geometric overlap a stitched pair fits and crossfades over by
# default, centred — so the outer 5% at each end, where both detectors are at
# their noise floor, is excluded until the operator says otherwise. See
# StitchedSpectrometer.
OVERLAP_BAND_FRAC = 0.90


def load_calibration_file(path) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse an intensity-calibration text file: two whitespace-separated
    columns, wavelength (nm) and multiplicative factor, '#' comments allowed.
    Returns (wavelengths, factors, n_skipped) sorted by wavelength.

    Rows that do not carry two parseable numbers are SKIPPED rather than
    fatal, and counted in n_skipped so the caller can say so. A single ragged
    line — an editor's stray trailing field, a truncated export — used to take
    np.loadtxt down with "the number of columns changed", which the GUI then
    turned into a transient message while the device silently stayed on raw
    counts. A file is only rejected when fewer than two usable rows survive.
    """
    try:
        data = np.atleast_2d(np.loadtxt(path, comments="#", usecols=(0, 1)))
        skipped = 0
    except ValueError:
        rows, skipped = [], 0
        with open(path, "r") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = line.split()
                try:
                    rows.append((float(fields[0]), float(fields[1])))
                except (IndexError, ValueError):
                    skipped += 1
        data = np.atleast_2d(np.asarray(rows, float).reshape(-1, 2))
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"{Path(path).name}: expected two columns (wavelength_nm factor) "
            f"with at least two rows, got data of shape {data.shape}.")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1], skipped


# ===========================================================================
# Abstract interfaces — implement these to add new hardware
# ===========================================================================
class StageLimitError(RuntimeError):
    """A move was stopped by an end-of-travel switch short of its target.

    Raised by adapters that can tell (KinesisStage reads the limit bits). Its
    own type so the scan can report "reached the end of its travel" instead of
    a generic failure — on an unhomed axis that is the expected way to find
    out where the ends are.
    """


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

    Setting backlash_mm > 0 makes moves on this stage arrive travelling in the
    + direction, so the marked zero and the scan share one frame. The pre-move
    only happens when the move would otherwise arrive from above, so a monotonic
    ascending sweep pays nothing for it. The one case it cannot cover is a target
    within backlash_mm of the bottom of travel: there is no room to undershoot,
    so the pre-move is skipped and that move may still arrive from above.

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
    # True when hardware limit switches at BOTH ends will stop any move. Only
    # then may an axis that needs_homing be driven below its travel_min_mm —
    # see reachable_range_mm. Adapters that cannot confirm it leave it False.
    hw_limits_stop: bool = False
    # Smallest move the axis can resolve, in mm, or None when the adapter does
    # not know. This is what defines "arrived": a settle window narrower than
    # the resolution can never close, and one wider than it throws away
    # accuracy the stage actually has.
    resolution_mm: float | None = None

    def reachable_range_mm(self) -> tuple[float, float | None]:
        """Position range (lo, hi) a move may target, in mm; hi None = unknown.

        Normally [travel_min_mm, travel_mm]. On an axis that needs_homing the
        readback is a counter zeroed wherever the carriage sat at power-up, so
        the carriage can be anywhere in [0, travel] and every reachable
        counter value lies within ±travel. That wider range is only offered
        when hw_limits_stop says the switches will halt the carriage at the
        real ends — which is how Thorlabs' own Kinesis software drives an
        unhomed stage.
        """
        t = self.travel_mm
        if self.needs_homing and self.hw_limits_stop and t:
            return -float(t), float(t)
        return self.travel_min_mm, t

    def move_to(self, position_mm: float) -> None:
        """Absolute move to position_mm; blocks until settled.

        With backlash_mm > 0 the stage always arrives travelling in the +
        direction: if we would otherwise come down onto the target, undershoot
        by backlash_mm first and take the slack up on the way back.

        Both legs go through _move_stepped, so neither can travel further than
        max_step_mm in one go. The current position is read once here and
        carried through both legs rather than re-read per leg — but it IS an
        extra get_position() per move, which is why an adapter that needs
        neither backlash nor a step limit (PiezoJenaStage) sets both to 0 and
        takes the no-readback fast path below.
        """
        target = float(position_mm)
        b = self.backlash_mm
        if b <= 0.0 and (self.max_step_mm or 0.0) <= 0.0:
            self._move_to_raw(target)      # nothing to arrange — no readback
            return
        cur = self.get_position()
        if b > 0.0:
            pre = max(target - b, self.reachable_range_mm()[0])
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
        lo, hi = self.reachable_range_mm()
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
        """Drive to the reference position and re-establish the frame.

        Raises unless an adapter overrides it: a silent no-op here once let the
        GUI announce "Stage homed" on a stage that had not moved.
        """
        raise NotImplementedError(f"{self.name} has no homing routine")

    @property
    def can_home(self) -> bool:
        """Whether home() does anything — i.e. the adapter overrides it. The
        GUI greys the Home button out when this is False."""
        return type(self).home is not StageBase.home

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
    # Exposure range this device accepts, in ms — what the GUI's integration
    # box may offer. The defaults are the conservative whole-millisecond range
    # the app used before any sub-ms device existed; an adapter that knows
    # better MUST narrow or widen them, because the input is built from these.
    # The device stays the authority: these shape the input and catch the
    # obviously-wrong, they do not replace the SDK's own range check.
    min_integration_ms: float = 1.0
    max_integration_ms: float = 10000.0
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

    @property
    def exposure_ms(self) -> float:
        """The exposure alone, without any on-board averaging that
        `integration_ms` folds in — i.e. exactly what set_integration_time()
        writes. They differ only on a device that averages internally
        (Avantes); everywhere else this IS the integration time. A UI that
        both shows and sets the exposure must read THIS, or a round trip
        through it would multiply the frame cost by the average count."""
        return float(self.integration_ms)

    @abc.abstractmethod
    def acquire(self) -> np.ndarray:
        """Return a single spectrum (counts vs pixel)."""

    # ── Intensity calibration ────────────────────────────────────────────────
    # Per-pixel factors the raw counts are multiplied by, or None for raw
    # output. Deliberately NOT folded into acquire(): saturation is a property
    # of raw ADC counts, so consumers must check saturation on acquire()'s
    # output and only then run it through calibrate().
    calibration_name: str | None = None
    # How many malformed lines the loaded file had. Non-zero means the file
    # parsed but was not clean, which the GUI reports — a calibration is data
    # the operator has to trust, so a partial read must never pass silently.
    calibration_skipped: int = 0
    _cal_pixels: np.ndarray | None = None

    def set_calibration(self, path) -> int:
        """Load a calibration file (see load_calibration_file) and hold it
        interpolated onto this device's pixel grid. Pixels outside the file's
        wavelength range hold the edge factor rather than extrapolating.

        Returns the number of malformed lines skipped. Nothing on the device is
        touched until the file has parsed, so a failed load leaves the previous
        calibration (or raw counts) in place rather than a half-applied one.
        """
        wl, fac, skipped = load_calibration_file(path)
        self._cal_pixels = np.interp(np.asarray(self.wavelengths, float),
                                     wl, fac)
        self.calibration_name = Path(path).stem
        self.calibration_skipped = skipped
        return skipped

    def clear_calibration(self) -> None:
        self._cal_pixels = None
        self.calibration_name = None
        self.calibration_skipped = 0

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
