"""
scan.py — FROG delay-scan engine
================================
Two layers in one file:

  1. A pure, Qt-free core — delay<->position conversion, the delay grid,
     autocorrelation/FWHM, and the FrogScanConfig / FrogResult dataclasses.
     This is the part you unit-test and that a future retrieval step consumes.
  2. A thin FrogScanWorker(QThread) that drives a StageBase + SpectrometerBase
     through a scan, emitting columns live for the build-up plot.

Unit conventions: delays in fs, stage positions in MICROMETRES (um). The
master unit for the FROG axis is fs; positions are um everywhere user-facing.
pylablib reports the LTS300 in mm, so that one boundary is isolated in the
_um_to_stage / _stage_to_um helpers below — flip them to identity if the
stage adapter is ever changed to report um directly.

    python scan.py     # self-test of the pure core (no Qt, no hardware)
"""

from __future__ import annotations

import time
import datetime
from dataclasses import dataclass, field, asdict

import numpy as np

from hardware import C_NM_PER_FS, StageBase, SpectrometerBase


# The stage adapter speaks mm (pylablib native); the engine works in um.
# These two functions are the ONLY place mm appears — change them to identity
# if the adapter is ever switched to report um.
def _um_to_stage(um):
    return um / 1000.0          # um -> mm, for stage.move_to


def _stage_to_um(mm):
    return mm * 1000.0          # mm -> um, from stage.get_position / travel_mm


# ===========================================================================
# Pure optics — delay <-> stage position (um)
# ===========================================================================
# Double pass (retroreflector): an arm move of x changes the path by 2x, so
# tau = 2 x / c  ->  x = c tau / 2. pass_factor generalises this (2 = double
# pass, 1 = single pass) to leave room for other geometries later.
def delay_to_position_um(tau_fs, zero_pos_um, pass_factor=2):
    """Optical delay (fs) -> absolute stage position (um)."""
    x_um = (np.asarray(tau_fs, float) * C_NM_PER_FS) / pass_factor / 1000.0
    return zero_pos_um + x_um


def position_to_delay_fs(position_um, zero_pos_um, pass_factor=2):
    """Absolute stage position (um) -> optical delay (fs)."""
    x_um = np.asarray(position_um, float) - zero_pos_um
    return pass_factor * x_um * 1000.0 / C_NM_PER_FS


def build_delay_grid(start_fs, stop_fs, step_fs):
    """Delay axis from start to stop in steps of step_fs (direction-aware,
    so asymmetric ranges and high->low scans both work). The final point is
    snapped onto the step grid, so the realised stop may differ slightly."""
    if step_fs <= 0:
        raise ValueError("step_fs must be > 0")
    span = stop_fs - start_fs
    if span == 0:
        raise ValueError("start and stop delays are equal")
    direction = 1.0 if span > 0 else -1.0
    n = int(round(abs(span) / step_fs)) + 1
    if n < 2:
        raise ValueError("delay range is smaller than one step")
    return start_fs + direction * np.arange(n) * step_fs


# ===========================================================================
# Autocorrelation (wavelength-integrated marginal) + FWHM
# ---------------------------------------------------------------------------
# These feed the reference layout's autocorrelation panel + FWHM readout.
# The FWHM here is the AUTOCORRELATION width, not the pulse width (they differ
# by a shape-dependent deconvolution factor); a true duration needs retrieval.
# ===========================================================================
def autocorrelation(trace, baseline_subtract=True):
    """Collapse a (n_wl, n_delays) trace to an autocorrelation vs delay by
    integrating over the wavelength axis."""
    ac = np.asarray(trace, float).sum(axis=0)
    if baseline_subtract and ac.size:
        ac = ac - ac.min()
    return ac


def fwhm(x, y):
    """FWHM of a single-peaked curve y(x), linearly interpolated at half-max.
    Returns NaN if it can't be bracketed."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if y.size < 3 or not np.any(y > 0):
        return float("nan")
    half = 0.5 * (y.max() + y.min())
    idx = np.where(y >= half)[0]
    if idx.size < 1:
        return float("nan")
    i0, i1 = idx[0], idx[-1]

    def cross(ia, ib):
        ya, yb = y[ia], y[ib]
        if yb == ya:
            return x[ia]
        return x[ia] + (half - ya) / (yb - ya) * (x[ib] - x[ia])

    left  = cross(i0 - 1, i0) if i0 > 0 else x[i0]
    right = cross(i1, i1 + 1) if i1 < x.size - 1 else x[i1]
    return abs(right - left)


# ===========================================================================
# Config + result (self-contained; ready for a future retrieval step)
# ===========================================================================
@dataclass
class FrogScanConfig:
    # Delay axis — master unit is fs
    delay_start_fs: float = -3000.0
    delay_stop_fs:  float =  3000.0
    delay_step_fs:  float =  13.3426
    # Geometry / calibration
    zero_pos_um:    float = 0.0      # marked zero-delay stage position (um)
    pass_factor:    int   = 2        # 2 = double pass (retroreflector)
    # Per-point acquisition
    n_average:         int   = 1     # spectra averaged per delay point
    idle_shots:        int   = 0     # frames discarded after each move
    wait_after_move_s: float = 0.0   # extra settle after the blocking move
    # Background — bracketed before/after, stored separately, not subtracted
    capture_background: bool = True
    # Saturation alert
    saturation_fraction: float = 0.99       # fraction of full-scale = saturated
    saturation_counts: float | None = None  # explicit full-scale; None -> use
                                             # spectrometer.max_counts
    abort_on_saturation: bool = False        # alert only by default
    # Optional soft travel limits (um); None -> fall back to stage.travel_mm
    pos_min_um: float | None = None
    pos_max_um: float | None = None
    # Metadata only (does not affect acquisition)
    gate: str = "shg"

    def delays_fs(self):
        return build_delay_grid(self.delay_start_fs, self.delay_stop_fs,
                                self.delay_step_fs)

    def positions_um(self):
        return delay_to_position_um(self.delays_fs(), self.zero_pos_um,
                                    self.pass_factor)


@dataclass
class FrogResult:
    delays_fs:             np.ndarray
    positions_cmd_um:      np.ndarray   # commanded targets
    positions_readback_um: np.ndarray   # actual, read back per column
    wavelengths_nm:        np.ndarray
    trace:                 np.ndarray   # (n_wl, n_delays), raw counts
    background_before:     np.ndarray | None
    background_after:      np.ndarray | None
    config:                FrogScanConfig
    # Saturation record. `saturation_threshold` is None when the scan could not
    # check at all (no full scale known) — which is NOT the same as a clean
    # scan, and the two must stay distinguishable once the data is on disk.
    saturation_threshold:  float | None = None      # counts
    n_saturated:           np.ndarray | None = None # clipped px per column
    n_saturated_bg_before: int = 0
    n_saturated_bg_after:  int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds"))

    @property
    def saturation_checked(self) -> bool:
        return self.saturation_threshold is not None

    def saturated_columns(self) -> np.ndarray:
        """Indices of the trace columns that contain clipped pixels."""
        if self.n_saturated is None:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(np.asarray(self.n_saturated) > 0)

    def is_saturated(self) -> bool:
        return bool(self.saturated_columns().size
                    or self.n_saturated_bg_before or self.n_saturated_bg_after)

    def autocorrelation(self, baseline_subtract=True):
        return autocorrelation(self.trace, baseline_subtract)

    def fwhm_fs(self):
        return fwhm(self.delays_fs, self.autocorrelation())

    def metadata(self):
        """Light, text-friendly metadata. Excludes the raw background arrays,
        which are archived separately and not exported by default."""
        d = asdict(self.config)
        d.update(
            timestamp=self.timestamp,
            n_delays=int(self.delays_fs.size),
            n_pixels=int(self.wavelengths_nm.size),
            wl_start_nm=float(self.wavelengths_nm[0]),
            wl_end_nm=float(self.wavelengths_nm[-1]),
            fwhm_ac_fs=float(self.fwhm_fs()),
            has_background=bool(self.background_before is not None),
            # Saturation summary. Carried here (and in the npz arrays) so a
            # clipped scan is self-identifying on disk instead of only in a
            # status-bar label that vanishes with the session.
            saturation_checked=self.saturation_checked,
            saturation_threshold_counts=(None if self.saturation_threshold is None
                                         else float(self.saturation_threshold)),
            saturated=self.is_saturated(),
            n_saturated_columns=int(self.saturated_columns().size),
            n_saturated_pixels_max=(0 if self.n_saturated is None
                                    else int(np.max(self.n_saturated, initial=0))),
            n_saturated_bg_before=int(self.n_saturated_bg_before),
            n_saturated_bg_after=int(self.n_saturated_bg_after),
        )
        return d

    def background(self):
        """Mean of the bracketing background spectra (whichever were captured),
        or None if the scan ran without background frames."""
        bgs = [b for b in (self.background_before, self.background_after)
               if b is not None]
        if not bgs:
            return None
        return np.mean(np.asarray(bgs, float), axis=0)

    def corrected_trace(self):
        """Trace with the bracketed background removed; raw trace if there is
        none. This is what the file exporters write by default."""
        bg = self.background()
        if bg is None:
            return np.asarray(self.trace, float)
        return np.asarray(self.trace, float) - np.asarray(bg, float)[:, None]


# ===========================================================================
# File export — .dwc (Femtosoft-style), .csv
# ---------------------------------------------------------------------------
# The .dwc layout, as decoded from a reference file:
#
#   Number of delay points = 800          <- trace columns
#   Number of wavelength points = 2068    <- trace rows
#   Delay increment = 1.0000000           <- fs, uniform; no absolute axis
#   <blank>
#   [Wavelength vector]
#   <n_wl tab-separated nm values>
#   <blank>
#   [Data array]
#   <n_wl rows of n_delays tab-separated counts>
#
# Tab-separated, CRLF, LabVIEW "%.6E" numbers whose exponent is NOT zero
# padded (1.843471E+2, not 1.843471E+02) — hence _lv_row below.
# ===========================================================================
def _row_fmt(n, spec="%.6E", sep="\t"):
    """Template for one line of n numbers — formatting a whole row through a
    single % is a few times faster than per-value f-strings, which matters at
    ~1.7M values for a full trace."""
    return sep.join([spec] * n)


def _lv_row(values, fmt=None):
    """One tab-separated line of LabVIEW-style %.6E numbers. A line holds
    nothing but numbers, so dropping the exponent's padding zero is a plain
    string replace ("E+02" -> "E+2", "E+00" -> "E+0"). Pass a prebuilt `fmt`
    from _row_fmt() when writing many rows of the same length."""
    line = (fmt or _row_fmt(len(values))) % tuple(values)
    return line.replace("E+0", "E+").replace("E-0", "E-")


def _ascending(result):
    """(delays, trace) ordered by increasing delay. A high->low scan is stored
    in acquisition order, but .dwc only records a (positive) increment, so the
    columns have to be flipped to stay physically correct."""
    delays = np.asarray(result.delays_fs, float)
    trace  = result.corrected_trace()
    if delays.size > 1 and delays[-1] < delays[0]:
        return delays[::-1], trace[:, ::-1]
    return delays, trace


def _delay_increment(result, delays):
    """Uniform delay step (fs). Taken from the realised axis, falling back to
    the configured step for a single-column scan."""
    if delays.size > 1:
        return float(np.median(np.abs(np.diff(delays))))
    return float(result.config.delay_step_fs)


def write_dwc(path, result):
    """Write a FrogResult as a .dwc trace (background-subtracted)."""
    delays, trace = _ascending(result)
    wl = np.asarray(result.wavelengths_nm, float)
    n_wl, n_delays = trace.shape
    if wl.size != n_wl:
        raise ValueError(f"wavelength vector ({wl.size}) does not match the "
                         f"trace ({n_wl} rows)")

    with open(path, "w", encoding="ascii", newline="\r\n") as f:
        f.write(f"Number of delay points = {n_delays}\n"
                f"Number of wavelength points = {n_wl}\n"
                f"Delay increment = {_delay_increment(result, delays):.7f}\n"
                f"\n[Wavelength vector]\n")
        f.write(_lv_row(wl))
        f.write("\n\n[Data array]\n")
        fmt = _row_fmt(n_delays)
        f.write("\n".join(_lv_row(row, fmt) for row in trace))
        f.write("\n")


def write_npz(path, result):
    """Write the full FrogResult as a NumPy archive — raw counts plus the
    background frames, the per-column saturation record and metadata the text
    formats cannot carry. (.dwc and .csv have fixed layouts and take none of
    this; the archive is the only format that can carry it.)"""
    data = dict(
        trace=result.trace, delays_fs=result.delays_fs,
        wavelengths_nm=result.wavelengths_nm,
        positions_cmd_um=result.positions_cmd_um,
        positions_readback_um=result.positions_readback_um,
        metadata=np.array(str(result.metadata())),
    )
    if result.n_saturated is not None:
        # Per column, aligned with delays_fs — enough to mask out the ruined
        # columns downstream rather than just knowing that some exist.
        data["n_saturated"] = np.asarray(result.n_saturated, dtype=int)
    if result.background_before is not None:
        data["background_before"] = result.background_before
    if result.background_after is not None:
        data["background_after"] = result.background_after
    np.savez(path, **data)


def write_csv(path, result):
    """Write a FrogResult as a spreadsheet-friendly CSV: a delay header row
    (fs) over one row per wavelength (nm), background-subtracted."""
    delays, trace = _ascending(result)
    wl  = np.asarray(result.wavelengths_nm, float)
    fmt = _row_fmt(trace.shape[1], "%.6g", ",")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("wavelength_nm\\delay_fs," + _row_fmt(delays.size, "%.6g", ",") % tuple(delays))
        f.write("\n")
        for w, row in zip(wl, trace):
            f.write(f"{w:.6g}," + fmt % tuple(row) + "\n")


# ===========================================================================
# Scan worker (Qt) — must run off the GUI thread
# ===========================================================================
from PySide6.QtCore import QThread, Signal


class FrogScanWorker(QThread):
    """Drives a stage + spectrometer through a delay scan.

    Per delay point: absolute move (blocks until settled) -> optional extra
    wait -> discard `idle_shots` frames -> average `n_average` frames -> read
    back the actual position -> emit the column. The trace is assembled here
    and returned whole as a FrogResult, but columns are also emitted live so
    the GUI can build the trace up as it goes.

    Signals:
      progress(done, total)
      column_ready(i, delay_fs, readback_um, column)        # for the live plot
      background_ready("before"|"after", spectrum)
      saturation_warning(i, delay_fs, n_pixels, peak)       # i = -1 for a bg frame
      finished_scan(FrogResult)
      error(message)
    """
    progress           = Signal(int, int)
    column_ready       = Signal(int, float, float, object)
    background_ready   = Signal(str, object)
    saturation_warning = Signal(int, float, int, float)
    finished_scan      = Signal(object)
    error              = Signal(str)

    def __init__(self, stage: StageBase, spectrometer: SpectrometerBase,
                 config: FrogScanConfig, parent=None):
        super().__init__(parent)
        self.stage    = stage
        self.spec     = spectrometer
        self.cfg      = config
        self._abort   = False
        self._sat_thr = None      # full-scale threshold, set in run()
        self._member_thr = None   # per-member thresholds (stitched), set in run()

    def abort(self):
        """Request a clean stop after the current step."""
        self._abort = True

    def _measure(self):
        """One averaged spectrum at the current position. Returns
        (column, peak_count, n_saturated_pixels)."""
        cfg = self.cfg
        if cfg.wait_after_move_s:
            time.sleep(cfg.wait_after_move_s)
        for _ in range(max(0, cfg.idle_shots)):
            self.spec.acquire()                         # discard settling frames

        thr   = self._sat_thr
        n     = max(1, cfg.n_average)
        acc   = None
        peak  = 0.0
        n_sat = 0
        for _ in range(n):
            s = np.asarray(self.spec.acquire(), float)
            if self._member_thr is not None:
                # Stitched device: per-member raw frames, per-member scales.
                for raw, t in zip(self.spec.last_member_raw, self._member_thr):
                    m = float(raw.max())
                    if m > peak:
                        peak = m
                    if t is not None and m >= t:
                        n_sat += int(np.count_nonzero(raw >= t))
            else:
                m = float(s.max())
                if m > peak:
                    peak = m
                if thr is not None and m >= thr:
                    n_sat += int(np.count_nonzero(s >= thr))
            acc = s if acc is None else acc + s
        # Saturation was judged above on RAW counts (the threshold is an ADC
        # property); only the returned column is intensity-calibrated.
        return self.spec.calibrate(acc / n), peak, n_sat

    def run(self):
        try:
            cfg     = self.cfg
            delays  = cfg.delays_fs()
            targets = cfg.positions_um()
            total   = delays.size

            # Saturation threshold: config override beats the device's report.
            max_counts = cfg.saturation_counts
            if max_counts is None:
                max_counts = getattr(self.spec, "max_counts", None)
            self._sat_thr = (cfg.saturation_fraction * max_counts) if max_counts else None

            # Composite (stitched) spectrometer: the combined column has no
            # single full scale, so instead each member's RAW frame is judged
            # against that member's own full scale (the override, if set,
            # applies to every member).
            members = getattr(self.spec, "members", None)
            if members:
                thr = []
                for mem in members:
                    full = cfg.saturation_counts
                    if full is None:
                        full = getattr(mem, "max_counts", None)
                    thr.append(cfg.saturation_fraction * full if full else None)
                self._member_thr = thr if any(t is not None for t in thr) else None

            # Travel-limit guard — catches a bad zero or range BEFORE moving.
            lo, hi = cfg.pos_min_um, cfg.pos_max_um
            if lo is None and getattr(self.stage, "travel_mm", None):
                lo, hi = 0.0, _stage_to_um(float(self.stage.travel_mm))
            if lo is not None and hi is not None:
                bad = (targets < lo) | (targets > hi)
                if bad.any():
                    raise RuntimeError(
                        f"{int(bad.sum())} commanded positions fall outside "
                        f"[{lo:.1f}, {hi:.1f}] um — check zero-delay / range.")

            n_wl     = self.spec.wavelengths.size
            trace    = np.zeros((n_wl, total))
            readback = np.zeros(total)
            bg_before = bg_after = None
            # Recorded for EVERY frame, not just the clipped ones, so the saved
            # archive carries a per-column map of what is trustworthy.
            n_sat_col = np.zeros(total, dtype=int)
            bg_sat    = {"before": 0, "after": 0}

            def check_sat(idx, delay_fs, peak, n_sat, bg=None):
                if bg is not None:
                    bg_sat[bg] = int(n_sat)
                elif 0 <= idx < total:
                    n_sat_col[idx] = int(n_sat)
                if n_sat:
                    self.saturation_warning.emit(idx, float(delay_fs),
                                                 int(n_sat), float(peak))
                    if cfg.abort_on_saturation:
                        where = ("the background frame" if bg
                                 else f"delay {delay_fs:+.1f} fs")
                        # In stitched mode there is no single threshold — each
                        # member was judged against its own full scale.
                        rule = (f">= {self._sat_thr:.0f} counts"
                                if self._sat_thr is not None
                                else "at their detector's full scale")
                        self.error.emit(
                            f"Spectrometer saturated at {where} "
                            f"({n_sat} px {rule}). Scan stopped.")
                        return True
                return False

            # Background BEFORE — captured at the start position.
            self.stage.move_to(_um_to_stage(targets[0]))
            if cfg.capture_background:
                bg_before, peak, n_sat = self._measure()
                self.background_ready.emit("before", bg_before)
                if check_sat(-1, delays[0], peak, n_sat, bg="before"):
                    return

            for i in range(total):
                if self._abort:
                    self.error.emit("Scan aborted by user.")
                    return
                self.stage.move_to(_um_to_stage(targets[i]))   # absolute, blocking
                col, peak, n_sat = self._measure()
                pos_um = _stage_to_um(self.stage.get_position())   # read-back tag
                trace[:, i] = col
                readback[i] = pos_um
                self.column_ready.emit(i, float(delays[i]), pos_um, col)
                self.progress.emit(i + 1, total)
                if check_sat(i, delays[i], peak, n_sat):
                    return

            # Background AFTER — captured at the end position.
            if cfg.capture_background and not self._abort:
                bg_after, peak, n_sat = self._measure()
                self.background_ready.emit("after", bg_after)
                check_sat(-1, delays[-1], peak, n_sat, bg="after")

            self.finished_scan.emit(FrogResult(
                delays_fs=delays,
                positions_cmd_um=targets,
                positions_readback_um=readback,
                wavelengths_nm=np.asarray(self.spec.wavelengths, float),
                trace=trace,
                background_before=bg_before,
                background_after=bg_after,
                config=cfg,
                saturation_threshold=self._sat_thr,
                n_saturated=n_sat_col,
                n_saturated_bg_before=bg_sat["before"],
                n_saturated_bg_after=bg_sat["after"],
            ))
        except Exception as e:
            self.error.emit(str(e))


# ===========================================================================
# Self-test — pure core only
# ===========================================================================
if __name__ == "__main__":
    cfg = FrogScanConfig(delay_start_fs=-300, delay_stop_fs=300,
                         delay_step_fs=13.3426, zero_pos_um=150000.0)
    d = cfg.delays_fs()
    p = cfg.positions_um()
    print("scan.py self-test (pure core)")
    print(f"  {d.size} points, {d[0]:+.1f}..{d[-1]:+.1f} fs, step {cfg.delay_step_fs} fs")
    print(f"  positions {p.min():.3f}..{p.max():.3f} um (zero at {cfg.zero_pos_um:.0f} um)")
    back = position_to_delay_fs(p, cfg.zero_pos_um, cfg.pass_factor)
    print(f"  delay round-trip max error: {np.max(np.abs(back - d)):.2e} fs")
    ac = np.exp(-(d ** 2) / (2 * 80.0 ** 2))   # gaussian, sigma=80 -> FWHM~188
    print(f"  FWHM of test gaussian: {fwhm(d, ac):.1f} fs  (expect ~188)")
