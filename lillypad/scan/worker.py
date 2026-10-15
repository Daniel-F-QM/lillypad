"""The scan worker: drive a stage and a spectrometer through a delay sweep.

The only Qt under lillypad/scan/. Everything it orchestrates — the travel
pre-flight, the background brackets, the saturation and stage-fault policies,
the park at the end — is here rather than spread across the GUI, so the sequence
a measurement actually follows can be read in one place.

run() may be called DIRECTLY, on the calling thread, instead of through start().
Signals then fire synchronously to same-thread receivers with no event loop,
which is how the test suite drives a whole scan headlessly.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from lillypad.hardware.base import (SpectrometerBase, StageBase,
                                    StageLimitError)
from lillypad.scan.core import delay_to_position_um, stage_to_um, um_to_stage
from lillypad.scan.model import FrogResult, FrogScanConfig

class FrogScanWorker(QThread):
    """Drives a stage + spectrometer through a delay scan.

    Per delay point: absolute move (blocks until settled) -> optional extra
    wait -> discard `idle_shots` frames -> average `n_average` frames -> read
    back the actual position -> check the stage still knows where it is -> emit
    the column. The trace is assembled here and returned whole as a FrogResult,
    but columns are also emitted live so the GUI can build the trace up as it
    goes.

    The scan sweeps monotonically, so every point is approached from the same
    side. Keeping that true of the marked zero-delay position as well is the
    stage adapter's job (StageBase.backlash_mm) — without it the two sit in
    frames that differ by the mechanical backlash and the whole delay axis is
    offset.

    Signals:
      progress(done, total)
      column_ready(i, delay_fs, readback_um, column, members)  # for the live plot
        members: the two RAW member frames (stitched pair) or None
      delay_reference_ready("before"|"after", spectrum)
      saturation_warning(i, delay_fs, n_pixels, peak)       # i = -1 for a ref frame
      stage_fault(i, delay_fs, reason)                      # i = -1 outside the sweep
      finished_scan(FrogResult)
      error(message)
    """
    progress           = Signal(int, int)
    column_ready       = Signal(int, float, float, object, object)
    delay_reference_ready = Signal(str, object)
    saturation_warning = Signal(int, float, int, float)
    stage_fault        = Signal(int, float, str)
    finished_scan      = Signal(object)
    error              = Signal(str)

    def __init__(self, stage: StageBase, spectrometer: SpectrometerBase,
                 config: FrogScanConfig, dark=None, parent=None):
        super().__init__(parent)
        self.stage    = stage
        self.spec     = spectrometer
        self.cfg      = config
        # The frame to subtract from the exported trace, or None to export raw
        # counts. Passed IN rather than measured here: it is the same dark the
        # Spectrum panel is using, snapshotted by the caller when the scan
        # started, so the picture on screen and the file on disk agree. A scan
        # never records its own — that is what made the old bracketed
        # "background" a second, silently different notion of the same word.
        self.dark     = None if dark is None else np.asarray(dark, float)
        self._abort   = False
        self._sat_thr = None      # full-scale threshold, set in run()
        self._member_thr = None   # per-member thresholds (stitched), set in run()
        # The members themselves, or None for a single device. Separate from
        # _member_thr, which is None whenever no member reports a full scale —
        # the raw frames are still there and still wanted.
        self._members = None

    def abort(self):
        """Request a clean stop after the current step."""
        self._abort = True

    def _measure(self):
        """One averaged spectrum at the current position. Returns
        (column, peak_count, n_saturated_pixels, member_frames).

        member_frames is None for a single device, and otherwise the two RAW
        (pre-dark, pre-calibration) member frames averaged over the same shots
        as the column. They are carried out for the GUI's raw-trace alignment
        view only — the column, the FrogResult and every export stay exactly
        as they were.
        """
        cfg = self.cfg
        if cfg.wait_after_move_s:
            time.sleep(cfg.wait_after_move_s)
        for _ in range(max(0, cfg.idle_shots)):
            self.spec.acquire()                         # discard settling frames

        thr   = self._sat_thr
        n     = max(1, cfg.n_average)
        acc   = None
        m_acc = None
        peak  = 0.0
        n_sat = 0
        for _ in range(n):
            s = np.asarray(self.spec.acquire(), float)
            if self._members is not None:
                # Stitched device: per-member raw frames, per-member scales.
                # One read of the tuple, used for both the saturation check and
                # the raw accumulation — re-reading it would mix two shots.
                frames = self.spec.last_member_raw
                thrs = self._member_thr or (None, None)
                for raw, t in zip(frames, thrs):
                    m = float(raw.max())
                    if m > peak:
                        peak = m
                    if t is not None and m >= t:
                        n_sat += int(np.count_nonzero(raw >= t))
                if m_acc is None:
                    m_acc = [np.asarray(f, float).copy() for f in frames]
                else:
                    for a, f in zip(m_acc, frames):
                        a += f
            else:
                m = float(s.max())
                if m > peak:
                    peak = m
                if thr is not None and m >= thr:
                    n_sat += int(np.count_nonzero(s >= thr))
            acc = s if acc is None else acc + s
        members = None if m_acc is None else tuple(a / n for a in m_acc)
        # Saturation was judged above on RAW counts (the threshold is an ADC
        # property); only the returned column is intensity-calibrated.
        return self.spec.calibrate(acc / n), peak, n_sat, members

    def run(self):
        # Cleared by anything that makes the position frame untrustworthy, so
        # the park below is skipped rather than commanding a move to a "zero"
        # that is no longer where zero is. See _park_at_zero.
        park = True
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
            self._members = tuple(members) if members else None
            if members:
                thr = []
                for mem in members:
                    full = cfg.saturation_counts
                    if full is None:
                        full = getattr(mem, "max_counts", None)
                    thr.append(cfg.saturation_fraction * full if full else None)
                self._member_thr = thr if any(t is not None for t in thr) else None

            # Travel-limit guard — catches a bad zero or range BEFORE moving.
            # The low end is limit.min (not necessarily 0) plus the backlash
            # margin, since the stage undershoots by that much to approach a
            # target from below and must still have room to do it. On an
            # unhomed axis the range is ±travel (StageBase.reachable_range_mm):
            # nothing here can know where the real ends are, so reaching one
            # is caught during the sweep instead — see move_or_stop.
            lo, hi = cfg.pos_min_um, cfg.pos_max_um
            backlash_um = stage_to_um(
                float(getattr(self.stage, "backlash_mm", 0.0) or 0.0))
            if lo is None:
                r_lo, r_hi = self.stage.reachable_range_mm()
                if r_hi:
                    lo, hi = stage_to_um(float(r_lo)), stage_to_um(float(r_hi))
            if lo is not None:
                lo += backlash_um
            if lo is not None and hi is not None:
                bad = (targets < lo) | (targets > hi)
                if bad.any():
                    raise RuntimeError(
                        f"{int(bad.sum())} commanded positions fall outside "
                        f"[{lo:.1f}, {hi:.1f}] um — check zero-delay / range.")

            n_wl     = self.spec.wavelengths.size
            trace    = np.zeros((n_wl, total))
            readback = np.zeros(total)
            ref_before = ref_after = None
            # Recorded for EVERY frame, not just the clipped ones, so the saved
            # archive carries a per-column map of what is trustworthy.
            n_sat_col = np.zeros(total, dtype=int)
            bg_sat    = {"before": 0, "after": 0}
            faults    = np.array([""] * total, dtype=object)

            # Warning flags latch on the controller, so anything left over from
            # a previous session would be blamed on this scan. Start clean.
            self.stage.clear_position_faults()

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

            def check_fault(idx, delay_fs, target_um=None, pos_um=None):
                """Ask the stage whether its position is still trustworthy.

                Returns True when the scan must stop. Unlike saturation this
                aborts by DEFAULT: a clipped column is still labelled with the
                delay it was taken at, but a stalled or hand-nudged stage
                labels every later column with a delay that never happened.
                """
                reason = self.stage.position_fault()
                # A readback that disagrees with the command is the same class
                # of problem. Off unless a tolerance is configured — see
                # FrogScanConfig.position_tolerance_um for why.
                tol = cfg.position_tolerance_um
                if tol is not None and target_um is not None and pos_um is not None:
                    gap = abs(pos_um - target_um)
                    if gap > tol:
                        drift = (f"stage stopped {gap:.2f} um from the commanded "
                                 f"position (tolerance {tol:.2f} um)")
                        reason = f"{reason}; {drift}" if reason else drift
                if not reason:
                    return False
                if 0 <= idx < total:
                    faults[idx] = reason
                self.stage_fault.emit(idx, float(delay_fs), reason)
                if cfg.abort_on_stage_fault:
                    nonlocal park
                    park = False       # this zero no longer means anything
                    where = ("the background frame" if idx < 0
                             else f"delay {delay_fs:+.1f} fs")
                    self.error.emit(
                        f"Stage fault at {where}: {reason}. Scan stopped — the "
                        f"delay axis from here on cannot be trusted. Re-home "
                        f"the stage and set zero again before rescanning.")
                    return True
                return False

            def move_or_stop(idx, delay_fs, target_um):
                """Move to target_um. Returns True when the scan must stop
                because the stage hit an end of travel getting there.

                Always stops, whatever abort_on_stage_fault says: the stage is
                not at the commanded delay and cannot get there, so carrying
                on would only record more columns at the same wrong position.
                Not parked either — same caution as the other stage faults.
                """
                nonlocal park
                try:
                    self.stage.move_to(um_to_stage(target_um))
                except StageLimitError as e:
                    reason = str(e)
                    if 0 <= idx < total:
                        faults[idx] = reason
                    self.stage_fault.emit(idx, float(delay_fs), reason)
                    park = False
                    self.error.emit(
                        f"Stage reached the end of its travel at "
                        f"{delay_fs:+.1f} fs — {reason} Scan stopped. Move the "
                        f"scan range or the zero away from that end.")
                    return True
                return False

            # Delay-extreme reference BEFORE — captured at the start position.
            # Archived, never subtracted (see FrogScanConfig.capture_delay_reference).
            if move_or_stop(-1, delays[0], targets[0]):
                return
            if check_fault(-1, delays[0]):
                return
            if cfg.capture_delay_reference:
                ref_before, peak, n_sat, _ = self._measure()
                self.delay_reference_ready.emit("before", ref_before)
                if check_sat(-1, delays[0], peak, n_sat, bg="before"):
                    return

            for i in range(total):
                if self._abort:
                    self.error.emit("Scan aborted by user.")
                    return
                if move_or_stop(i, delays[i], targets[i]):    # absolute, blocking
                    return
                col, peak, n_sat, members_raw = self._measure()
                pos_um = stage_to_um(self.stage.get_position())   # read-back tag
                trace[:, i] = col
                readback[i] = pos_um
                self.column_ready.emit(i, float(delays[i]), pos_um, col,
                                       members_raw)
                self.progress.emit(i + 1, total)
                if check_sat(i, delays[i], peak, n_sat):
                    return
                if check_fault(i, delays[i], targets[i], pos_um):
                    return

            # Delay-extreme reference AFTER — captured at the end position.
            if cfg.capture_delay_reference and not self._abort:
                ref_after, peak, n_sat, _ = self._measure()
                self.delay_reference_ready.emit("after", ref_after)
                check_sat(-1, delays[-1], peak, n_sat, bg="after")

            self.finished_scan.emit(FrogResult(
                delays_fs=delays,
                positions_cmd_um=targets,
                positions_readback_um=readback,
                wavelengths_nm=np.asarray(self.spec.wavelengths, float),
                trace=trace,
                dark=self.dark,
                delay_reference_before=ref_before,
                delay_reference_after=ref_after,
                config=cfg,
                saturation_threshold=self._sat_thr,
                n_saturated=n_sat_col,
                n_saturated_ref_before=bg_sat["before"],
                n_saturated_ref_after=bg_sat["after"],
                stage_faults=faults,
                stage_name=str(getattr(self.stage, "name", "")),
                stage_backlash_um=backlash_um,
            ))
        except Exception as e:
            park = False       # unknown failure: do not drive the stage on it
            self.error.emit(str(e))
        finally:
            if park:
                self._park_at_zero()

    def _park_at_zero(self):
        """Leave the stage on zero delay once the scan is over.

        A scan ends wherever its last delay happened to be, which for a
        one-sided range is a long way from the overlap — so the next thing the
        operator does (a live spectrum, an alignment tweak) starts on a dark
        frame. Parking on zero delay puts the pulses back on top of each other,
        which is the only position that means anything between scans.

        In a finally, so it happens on every ordinary way out: a completed
        scan, a user abort, a saturation abort. On the WORKER's thread, because
        that is the one that owns the stage for the whole scan — parking from
        the GUI's finished handler would command a move while run() may still
        be on its way out.

        Skipped after a stage fault or an unexpected exception (see `park`):
        both mean the position frame is no longer trustworthy, and "go to zero"
        would then drive somewhere that is not zero.

        Failures are swallowed. The measurement is already complete and emitted
        by this point, and losing the park is not a reason to turn a good scan
        into an error the operator has to interpret.
        """
        try:
            self.stage.move_to(um_to_stage(
                delay_to_position_um(0.0, self.cfg.zero_pos_um,
                                     self.cfg.pass_factor)))
        except Exception:
            pass
