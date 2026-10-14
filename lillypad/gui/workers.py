"""The GUI's worker threads, and the exposure plumbing they pace against.

Three QThreads, all of the same shape: they own no widgets, they touch devices
off the GUI thread, and they report back only through signals.

  LiveFeedWorker   continuous acquisition, paced to the exposure
  ExportWorker     writing a trace, which is far too slow for the GUI thread
  AlignmentWorker  the four-point symmetry sweep

They are together rather than one per controller so that "where do the threads
live" has an answer, and so the rules they share — park the feed before touching
a device from a slot, never let two threads into one vendor driver — are
reviewable in one place.

The scan worker is NOT here: it belongs to the scan engine and lives in
lillypad.scan.worker, which has no GUI dependency at all.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from lillypad.scan.core import (delay_to_position_um, stage_to_um,
                                um_to_stage)


class LiveFeedWorker(QThread):
    """Continuously acquires spectra off the GUI thread and emits them.

    FIX 2 — spectrometer.acquire() blocks for the integration time; running it
    here keeps the UI responsive even at long integrations.
    FIX 3 — each loop is paced to the integration time (with a small floor) so
    we never spin faster than the hardware actually produces frames.

    `pause()` blocks until any in-flight acquire has finished, so the scan can
    safely take over the shared stage/spectrometer.
    """
    spectrum_ready = Signal(object, object)   # (wavelengths, raw counts)
    # A device that has started refusing to acquire. Emitted on the first
    # failure and then at most every FAIL_REPORT_S while it persists, because
    # the alternative — the loop quietly retrying forever — is indistinguishable
    # from "the setting I just changed did nothing": the plot simply stops
    # updating and holds the last good frame.
    acquire_failed = Signal(str)
    FAIL_REPORT_S = 3.0

    def __init__(self, get_spec, min_interval_ms=30, parent=None):
        super().__init__(parent)
        self._get_spec = get_spec
        self._min = float(min_interval_ms)
        self._last_fail_report = 0.0
        self._run = True
        self._paused = True
        self._idle = threading.Event()   # set whenever not mid-acquire
        self._idle.set()
        # Guards `_paused` together with the matching set/clear of `_idle`.
        # The pair has to move as one: pause() decides it is safe to hand the
        # device over by reading `_idle`, and run() decides to acquire by
        # reading `_paused`, so the two decisions must not interleave. Held
        # only for those few statements — never across an acquire or a wait.
        self._lock = threading.Lock()
        # Measured worst-case time for one loop iteration. pause() sizes its
        # timeout from this instead of a flat 2 s, which a long integration
        # would otherwise blow through before the loop could possibly park.
        self._last_cycle_ms = float(min_interval_ms)

    def run(self):
        while self._run:
            # Reading `_paused` and publishing the answer in `_idle` is one
            # atomic step. Clearing `_idle` after an unlocked check of
            # `_paused` left a window where pause() could set `_paused`, see
            # the still-set `_idle` from the previous park, and report the
            # device free — while this thread was already on its way into
            # acquire(). Two threads in one vendor driver is exactly what
            # _device_lock exists to prevent.
            with self._lock:
                parked = self._paused
                if parked:
                    self._idle.set()
                else:
                    self._idle.clear()
            if parked:
                self.msleep(15)
                continue
            spec = self._get_spec()
            if spec is None:
                # No device to hold: safe to report idle without the lock,
                # since the loop goes back to the top (and re-checks `_paused`
                # under it) before it could touch anything.
                self._idle.set()
                self.msleep(30)
                continue
            try:
                t0 = time.monotonic()
                raw = np.asarray(spec.acquire(), float)
                wl  = np.asarray(spec.wavelengths, float)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
            except Exception as e:
                now = time.monotonic()
                if now - self._last_fail_report >= self.FAIL_REPORT_S:
                    self._last_fail_report = now
                    if self._run and not self._paused:
                        self.acquire_failed.emit(str(e))
                self.msleep(50)
                continue
            self._last_fail_report = 0.0     # recovered — report the next one
            if self._run and not self._paused:
                self.spectrum_ready.emit(wl, raw)
            # Pace to the integration time; acquire already consumed `elapsed_ms`
            # of it on real hardware, so only sleep the remainder.
            target = max(self._min, float(getattr(spec, "integration_ms", self._min)))
            # Whichever of the two actually dominates the cycle is what a
            # pause() may have to sit through before we come round the top.
            self._last_cycle_ms = max(target, elapsed_ms)
            self.msleep(int(max(0.0, target - elapsed_ms)))

    def resume(self):
        with self._lock:
            self._paused = False

    def join_timeout_ms(self):
        """How long to allow the loop to reach its next park.

        One acquire plus one pacing sleep, with margin, measured rather than
        assumed: acquire() blocks for the whole exposure, which the UI allows
        up to MAX_UI_EXPOSURE_MS, so a flat 2 s would expire while the device
        was still perfectly healthy.

        The measured cycle only exists once an acquire has COMPLETED, so the
        device's own integration time is taken into account as well — the very
        first frame, and the first frame after the exposure is raised, would
        otherwise be judged against a stale (or default) figure.
        """
        cycle = self._last_cycle_ms
        try:
            spec = self._get_spec()
        except Exception:
            spec = None
        if spec is not None:
            cycle = max(cycle, float(getattr(spec, "integration_ms", 0.0)))
        return max(2000.0, 3.0 * cycle + 500.0)

    def pause(self, wait_ms=None):
        """Stop acquiring; block until the current acquire (if any) returns.

        Returns True once the loop has parked in the idle branch — only then is
        it safe to hand the stage/spectrometer to another thread. False means
        the timeout expired with an acquire STILL in flight; callers must treat
        that as "device not available" rather than ignoring it.

        `_paused` is set under the lock, so from here on run() cannot clear
        `_idle` — it only ever does that after reading `_paused` as False under
        the same lock. A set `_idle` therefore means the device is free and
        will stay free until resume(), which is what makes the True honest.
        """
        with self._lock:
            self._paused = True
        if wait_ms is None:
            wait_ms = self.join_timeout_ms()
        return self._idle.wait(wait_ms / 1000.0)

    def stop(self):
        self._run = False
        with self._lock:
            self._paused = True

# Workers that would not stop before the window closed. Parked here so their
# QThread objects outlive the window: destroying one while its run() is still
# inside a vendor call is a hard crash, not a warning, and at this point the
# process is on its way out anyway. Deliberately never emptied.
_ORPHANED_THREADS = []

# Longest exposure the integration box will offer, whatever the device claims.
# A GUI policy, not a hardware limit: acquire() blocks for the whole exposure,
# so a device that will happily take its documented 600 s would leave the app
# looking hung, with the feed unable to park and every dialog answering
# FEED_BUSY_MSG. Type-in of a longer value is not something the app needs.
MAX_UI_EXPOSURE_MS = 10_000.0

def _exposure_ms(spec) -> float:
    """What the integration box should SHOW for `spec`.

    The exposure, never the frame time: the box writes set_integration_time(),
    so seeding it from a frame cost that folds in on-board averaging would
    make every later edit multiply the exposure by the average count.
    """
    return float(getattr(spec, "exposure_ms",
                         getattr(spec, "integration_ms", 10.0)))

def _exposure_bounds(spec) -> tuple[float, float]:
    """(min, max) exposure in ms the integration box may offer for `spec`."""
    lo = float(getattr(spec, "min_integration_ms", 1.0))
    hi = float(getattr(spec, "max_integration_ms", MAX_UI_EXPOSURE_MS))
    return lo, min(hi, MAX_UI_EXPOSURE_MS)

class ExportWorker(QThread):
    """Writes a FrogResult to disk off the GUI thread.

    write_dwc formats on the order of 1.7M values for a full trace, which froze
    the window for the whole save. A finished FrogResult is never mutated again,
    so the worker can read it without any locking — and because a later scan
    rebinds `self.result` rather than modifying it, an in-flight export keeps
    writing the snapshot it was handed.
    """
    done  = Signal(str)
    error = Signal(str)

    def __init__(self, writer, path, result, parent=None):
        super().__init__(parent)
        self._writer = writer
        self._path   = path
        self._result = result

    def run(self):
        try:
            self._writer(self._path, self._result)
        except Exception as e:
            self.error.emit(str(e))
        else:
            self.done.emit(self._path)

class AlignmentWorker(QThread):
    """Measures one spectrum at each of -2x, -x, +x, +2x fs and comes back.

    A thread rather than the blocking _stage_action path: four moves plus
    n_average frames each takes seconds, and the window must stay alive.

    The offsets are RELATIVE to wherever the stage is standing when the worker
    starts — the operator parks at the delay whose symmetry they want to judge
    (usually zero) and presses the button, so no marked zero is required.

    The four points are visited in increasing order, exactly like a scan, so
    every one of them is approached from the same side and the mechanical
    backlash cancels out of the +/- comparison. Whatever happens, the stage is
    put back where it started.

    Frames are returned RAW: the window applies the same dark/calibration path
    the live spectrum uses, so the differences sit on the displayed baseline
    instead of a second, drifting derivation of it.

    Signals:
      progress(done, total)
      done([(raw, member_frames_or_None)] x 4)   # in offset order
      error(message)
    """
    progress = Signal(int, int)
    done     = Signal(object)
    error    = Signal(str)

    def __init__(self, stage, spectrometer, config, step_fs, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.spec  = spectrometer
        self.cfg   = config
        self.offsets_fs = (-2.0 * step_fs, -step_fs, step_fs, 2.0 * step_fs)

    def _measure(self):
        """One averaged raw frame here. Mirrors FrogScanWorker._measure minus
        the saturation bookkeeping and the calibration."""
        cfg = self.cfg
        if cfg.wait_after_move_s:
            time.sleep(cfg.wait_after_move_s)
        for _ in range(max(0, cfg.idle_shots)):
            self.spec.acquire()                    # discard settling frames
        n     = max(1, cfg.n_average)
        acc   = None
        m_acc = None
        for _ in range(n):
            s = np.asarray(self.spec.acquire(), float)
            acc = s if acc is None else acc + s
            frames = getattr(self.spec, "last_member_raw", None)
            if frames is not None:
                if m_acc is None:
                    m_acc = [np.asarray(f, float).copy() for f in frames]
                else:
                    for a, f in zip(m_acc, frames):
                        a += f
        members = None if m_acc is None else tuple(a / n for a in m_acc)
        return acc / n, members

    def run(self):
        start_um = None
        out, err = [], None
        try:
            start_um = stage_to_um(self.stage.get_position())
            pf = self.cfg.pass_factor
            for k, off in enumerate(self.offsets_fs):
                # zero = 0.0 makes this a RELATIVE conversion, as _jog does.
                d_um = float(delay_to_position_um(off, 0.0, pf))
                self.stage.move_to(um_to_stage(start_um + d_um))
                out.append(self._measure())
                self.progress.emit(k + 1, len(self.offsets_fs))
        except Exception as e:
            err = str(e)
        # Always come home, including after a failure part-way through: leaving
        # the stage parked at +2x would silently shift every later jog, move
        # and scan the operator makes from here.
        if start_um is not None:
            try:
                self.stage.move_to(um_to_stage(start_um))
            except Exception as e:
                if err is None:
                    err = f"could not return to the start position: {e}"
        # Reported only once the stage is back: the slot restarts the live feed
        # and reads the position, and neither may happen while this thread is
        # still driving the hardware.
        if err is None:
            self.done.emit(out)
        else:
            self.error.emit(err)
