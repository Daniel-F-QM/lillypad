"""Opt-in performance probe (LILLYPAD_PERF=1).

Exists to answer "which render path is slow, and is anything growing?" with
numbers instead of guesses. Off by default and reduced to a nullcontext() when
off, so the instrumented paths cost nothing in normal use.

Worth keeping through the refactor: it is the only tool that can say whether a
step that moved rendering code also slowed it down.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager, nullcontext

from PySide6.QtCore import QObject


# Exists to answer "which render path is slow, and is anything growing?" with
# numbers instead of guesses. Off by default and reduced to a nullcontext() when
# off, so the instrumented paths cost nothing in normal use.
PERF = bool(os.environ.get("LILLYPAD_PERF"))

class _PerfProbe:
    """Collects per-path frame times and session-growth counters.

    Frame times are kept as raw samples and reduced to percentiles at report
    time: a mean would hide exactly what we are looking for, since the symptom
    is occasional long frames (a full redraw escaping the blit path, or a GC
    pause), not a uniformly slower loop.
    """
    def __init__(self, window):
        self._w = window
        self._samples = {}          # name -> list[ms], cleared each report
        tracemalloc.start()
        self._snap = tracemalloc.take_snapshot()

    @contextmanager
    def tick(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self._samples.setdefault(name, []).append(
                (time.perf_counter() - t) * 1000.0)

    def report(self):
        for name, xs in sorted(self._samples.items()):
            xs.sort()
            print(f"[perf] {name:12s} n={len(xs):4d} "
                  f"p50={xs[len(xs) // 2]:7.2f}ms "
                  f"p95={xs[min(int(len(xs) * 0.95), len(xs) - 1)]:7.2f}ms "
                  f"max={xs[-1]:7.2f}ms", file=sys.stderr)
        self._samples.clear()
        snap = tracemalloc.take_snapshot()
        grown = sum(s.size_diff for s in snap.compare_to(self._snap, "filename"))
        self._snap = snap
        # objects/qt_children are the leak detectors: both must plateau. A
        # rising qt_children with a flat objects count is a Qt-side leak (a
        # C++ child never destroyed), which is exactly the menu/worker case.
        print(f"[perf] objects={len(gc.get_objects()):8d} "
              f"qt_children={len(self._w.findChildren(QObject)):6d} "
              f"tracemalloc_delta={grown / 1e6:+.1f}MB", file=sys.stderr)

# Module-level singleton: FrogCanvas and FrogWindow both instrument themselves
# and neither should have to carry a probe reference through its constructor.
_perf = None


def install_probe(window):
    """Point perf_tick at a probe on `window`, and hand the probe back.

    An explicit installer rather than `global _perf` at the call site: that
    worked while the probe and its user were the same module, and silently would
    not once they were not. The caller still owns the report timer, because the
    reporting interval is a UI decision.
    """
    global _perf
    _perf = _PerfProbe(window)
    return _perf


def perf_tick(name):
    """Time a block into the probe, or do nothing when the probe is off."""
    return _perf.tick(name) if _perf is not None else nullcontext()
