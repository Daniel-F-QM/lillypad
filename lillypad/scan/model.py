"""What a scan is asked for, and what it produced.

FrogScanConfig and FrogResult are plain dataclasses with no Qt and no device:
the worker fills a result in, the writers read one out, and a future retrieval
step would consume one without either.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field

import numpy as np

from lillypad.scan.core import (autocorrelation, build_delay_grid,
                                delay_to_position_um, fwhm)

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
    # Take one frame at each end of the delay sweep and archive it in the .npz.
    # Provenance only: nothing subtracts it, and the exports are corrected with
    # the recorded dark instead. Off by default because it costs two extra
    # acquisitions and no longer affects the data.
    capture_delay_reference: bool = False
    # Saturation alert
    saturation_fraction: float = 0.99       # fraction of full-scale = saturated
    saturation_counts: float | None = None  # explicit full-scale; None -> use
                                             # spectrometer.max_counts
    abort_on_saturation: bool = False        # alert only by default
    # Stage health. Unlike saturation, this defaults to ABORTING: a clipped
    # spectrum is still labelled with the right delay, but a stalled or
    # hand-nudged stage means every column after it carries a delay that never
    # happened, and nothing downstream can tell.
    abort_on_stage_fault: bool = True
    # Readback-vs-commanded tolerance (um); None disables the check. Off by
    # default because it is a no-op on Zaber (its readback is the trajectory
    # counter, so it always agrees) and would false-trip on adapters whose
    # readback resolution is coarser than the tolerance.
    position_tolerance_um: float | None = None
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
    # The dark ACTUALLY SUBTRACTED from the exports, or None when none was.
    # Trace-row shaped, already combined and calibrated for a stitched pair —
    # i.e. exactly the frame the Spectrum panel is subtracting, snapshotted when
    # the scan started so every column is corrected the same way.
    dark:                  np.ndarray | None
    # OPTIONAL provenance, never subtracted from anything: one frame at each end
    # of the delay sweep, taken with the beam as it was. Useful for telling a
    # real pedestal from delay-independent scatter after the fact, which is why
    # it is archived in the .npz beside the dark rather than folded in.
    delay_reference_before: np.ndarray | None
    delay_reference_after:  np.ndarray | None
    config:                FrogScanConfig
    # Saturation record. `saturation_threshold` is None when the scan could not
    # check at all (no full scale known) — which is NOT the same as a clean
    # scan, and the two must stay distinguishable once the data is on disk.
    saturation_threshold:  float | None = None      # counts
    n_saturated:           np.ndarray | None = None # clipped px per column
    n_saturated_ref_before: int = 0
    n_saturated_ref_after:  int = 0
    # Per-column stage-health report, "" where the column was clean. Non-empty
    # entries mean the delay label on that column is not trustworthy.
    stage_faults:          np.ndarray | None = None
    stage_name:            str = ""
    stage_backlash_um:     float = 0.0
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
                    or self.n_saturated_ref_before or self.n_saturated_ref_after)

    def faulted_columns(self) -> np.ndarray:
        """Indices of the columns the stage reported a position fault at."""
        if self.stage_faults is None:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(np.asarray(self.stage_faults, dtype=object) != "")

    def has_stage_fault(self) -> bool:
        return bool(self.faulted_columns().size)

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
            # How the exported trace was corrected. False means .dwc and .csv
            # carry raw counts, pedestal and all — the one thing a reader of
            # those files cannot otherwise tell.
            dark_subtracted=self.dark_subtracted,
            has_delay_reference=bool(self.delay_reference_before is not None),
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
            n_saturated_ref_before=int(self.n_saturated_ref_before),
            n_saturated_ref_after=int(self.n_saturated_ref_after),
            # Stage provenance. backlash_um is what makes a trace's delay axis
            # comparable to another one's: a scan taken without the approach
            # correction carries an unknown zero offset of that order.
            stage_name=self.stage_name,
            stage_backlash_um=float(self.stage_backlash_um),
            stage_faulted=self.has_stage_fault(),
            n_faulted_columns=int(self.faulted_columns().size),
        )
        return d

    @property
    def dark_subtracted(self) -> bool:
        """Whether the exports carry a dark subtraction.

        Distinct from "the scan recorded a dark": Subtract Dark can be off, or
        no dark may have been recorded at all. A file that was NOT corrected has
        to say so, because at these levels an uncorrected trace is not slightly
        wrong but mostly pedestal.
        """
        return self.dark is not None

    def delay_reference(self):
        """Mean of the delay-extreme reference frames, or None if none were
        taken. Provenance only — nothing subtracts this."""
        refs = [r for r in (self.delay_reference_before, self.delay_reference_after)
                if r is not None]
        if not refs:
            return None
        return np.mean(np.asarray(refs, float), axis=0)

    def corrected_trace(self):
        """Trace with the recorded dark removed; the raw trace if none was
        subtracted. This is what .dwc and .csv are written from.

        The dark is the same frame the Spectrum panel uses, so the picture on
        screen and the file on disk are corrected identically.
        """
        if self.dark is None:
            return np.asarray(self.trace, float)
        return np.asarray(self.trace, float) - np.asarray(self.dark, float)[:, None]
