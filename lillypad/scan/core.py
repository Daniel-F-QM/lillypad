"""Delay/position conversions, the delay grid, and the two analysis reductions.

The device-free, Qt-free kernel of the scan engine. Importing this costs numpy
and nothing else — which is the point of it being its own module: scan.py used
to put `from PySide6.QtCore import QThread` at module scope, so reaching these
five functions meant having Qt installed. Its own docstring called that out as
the split worth making.

Unit conventions: delays in fs, stage positions in MICROMETRES. The master unit
for the FROG axis is fs; positions are um everywhere user-facing. The stage
adapter speaks mm, and um_to_stage / stage_to_um below are the only place that
conversion lives.
"""

from __future__ import annotations

import numpy as np

from lillypad.hardware.base import C_NM_PER_FS

# The stage adapter speaks mm (pylablib native); the engine works in um.
# These two functions are the only place the CONVERSION lives — change them to
# identity if the adapter is ever switched to report um. (mm still appears as
# attribute names read straight off the adapter — backlash_mm, travel_mm,
# travel_min_mm in run() below — and those would need converting too.)
def um_to_stage(um):
    return um / 1000.0          # um -> mm, for stage.move_to


def stage_to_um(mm):
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
    """Full width of a single-peaked curve y(x) at its half level, linearly
    interpolated. The level is halfway between y.min() and y.max(), which is
    half-MAX only when the baseline is already at zero — as it is for the
    autocorrelation above, which baseline-subtracts by default. Returns NaN if
    it can't be bracketed."""
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


# ── Symmetry folding ────────────────────────────────────────────────────────
# Display maths, but maths: these take arrays and return arrays, so they belong
# with the rest of the numpy in this module rather than on a Qt window where
# they cannot be tested without one. `fold_fs` is the hand-placed fold line, or
# None for the optimised fold.
#
# A fold is a HALF-INDEX k: it sits at index k/2, so k = 2c is on column c and
# k = 2c+1 is halfway between columns c and c+1. Half columns, not anything
# finer, because then every comparison is still between two columns that were
# actually measured — no resampling. Interpolating a mirrored column would
# average its noise down, and an optimiser would then prefer positions
# between columns for that reason alone.


def _fold_delay(k, delays):
    """Delay of fold k: the mean of the column(s) it sits on or between."""
    d = np.asarray(delays, float)
    return 0.5 * (d[k // 2] + d[(k + 1) // 2])


def _fold_pairs(k, n):
    """Column pairs (i, j), i = k - j, mirrored across fold k: j runs from
    the fold outward for as long as its partner i is still measured. On an
    even k the first pair is the fold column with itself."""
    j = np.arange((k + 1) // 2, min(n - 1, k) + 1)
    return k - j, j


def _asymmetry(ac, k):
    """The asymmetry score of fold k — see the comment in symmetry_view."""
    a = np.asarray(ac, float)
    total = float(np.nansum(a))
    i, j = _fold_pairs(k, a.size)
    if total <= 0 or not np.any(j > i):
        return 0.0
    return float(np.nansum(np.abs(a[j] - a[i]))) / total


def optimal_fold(ac, delays):
    """The fold (a half-index) that minimises the asymmetry score, searched
    outward from the autocorrelation peak.

    The AC peak is where the pulses overlap best, so it is where the search
    STARTS — but as an estimate of the symmetry axis it is quantised to whole
    columns and pushed about by noise, and a trace whose real centre falls
    between two columns would score asymmetric for no physical reason.

    Every half-column within ±max(AC FWHM/2, two scan steps) of the peak is
    scored, rather than descending from it, so a noise wiggle cannot trap the
    search; ties go to the fold nearest the peak. The window is kept local on
    purpose: towards the trace edges fewer pairs are compared and the score
    shrinks for that reason alone, and within it the compared span barely
    changes, so the scores are comparable. The score minimised is the one the
    readout shows, so dragging the line cannot beat it inside the window.
    """
    a = np.asarray(ac, float)
    d = np.asarray(delays, float)
    if a.size == 0 or not np.any(np.isfinite(a)):
        return 0
    k0 = 2 * int(np.nanargmax(a))
    if a.size < 2:
        return k0
    step = float(np.median(np.abs(np.diff(d))))
    width = fwhm(d, np.nan_to_num(a))
    reach = max(0.5 * width if np.isfinite(width) else 0.0, 2.0 * step)
    peak_fs = _fold_delay(k0, d)
    best_k, best_s = k0, _asymmetry(a, k0)
    # Nearest first, and only a STRICTLY lower score replaces the best, so a
    # tie stays with the fold closest to the peak.
    for k in sorted(range(2 * a.size - 1), key=lambda k: abs(k - k0)):
        if abs(_fold_delay(k, d) - peak_fs) > reach + 1e-9 * step:
            continue
        i, j = _fold_pairs(k, a.size)
        if not np.any(j > i):
            continue                  # on an edge: nothing to compare
        s = _asymmetry(a, k)
        if s < best_s - 1e-12:
            best_k, best_s = k, s
    return best_k


def symmetry_fold(ac, delays, fold_fs=None):
    """Half-index the trace is folded about.

    By default the optimised fold (optimal_fold): it starts at the
    autocorrelation maximum, where the two pulses overlap best, and moves to
    the most symmetric axis near it. Not the middle column, which would make a
    scan range that was not centred on the overlap report its own offset as
    asymmetry.

    A hand-placed line (dragged on the plot) overrides it, snapped to the
    nearest half column — on a measured column or halfway between two.
    """
    if fold_fs is None:
        return optimal_fold(ac, delays)
    d = np.asarray(delays, float)
    ks = np.arange(2 * d.size - 1)
    fds = 0.5 * (d[ks // 2] + d[(ks + 1) // 2])
    return int(ks[np.argmin(np.abs(fds - float(fold_fs)))])

def symmetry_view(trace, ac, delays, fold_fs=None):
    """Measured trace on the left of the fold, |T(+d) - T(-d)| on the right.

    Returns (image, peak, score, fold_delay_fs).

    Half and half rather than the difference everywhere: keeping one side as
    it was measured gives the eye something to judge the other against, and
    the two are directly comparable because they share one colour scale —
    taken from the MEASURED peak, so the left half looks exactly as it does
    in the ordinary view and "nearly black on the right" genuinely means
    "nearly symmetric" rather than "rescaled until it looks that way".

    ABSOLUTE difference. The signed one is antisymmetric, so folding it onto
    a single side would put half the pixels below zero, and every colormap
    here is sequential with clim (0, peak) — those would flatten onto the
    floor and hide exactly what the mode exists to show.

    Columns past the fold with no counterpart behind it — the tail of the
    longer half, whenever the fold sits off-centre — are left NaN rather
    than zeroed. Zero would claim perfect symmetry where there is simply
    nothing to compare against; NaN takes _apply_cmap's 'bad' colour, the
    plot background, so unmeasured reads as unmeasured.
    """
    t = np.asarray(trace, float)
    n = t.shape[1]
    k = symmetry_fold(ac, delays, fold_fs)
    fold = float(_fold_delay(k, delays))
    out = t.copy()                   # left of the fold: measured, untouched
    peak = float(np.nanmax(t)) if t.size else 0.0
    i, j = _fold_pairs(k, n)
    if not np.any(j > i):
        out[:, k // 2 + 1:] = np.nan     # fold on an edge: nothing to compare
        return out, peak, 0.0, fold
    out[:, j] = np.abs(t[:, j] - t[:, i])
    out[:, j[-1] + 1:] = np.nan      # beyond the overlap: no counterpart
    # Score: sum of the absolute difference over the sum of the whole
    # trace — but with the sum over WAVELENGTH taken first, i.e. on the
    # autocorrelation rather than pixel by pixel.
    #
    # This is not cosmetic. Per pixel, |noise - noise| never cancels: the
    # modulus turns every one of the ~30k near-zero background pixels into
    # a positive contribution, and their sum swamps the signal. Measured on
    # the simulator, a PERFECTLY symmetric beam scored 13.8% at 460 peak
    # counts, 2.7% at 4000 and 0.9% at 20000 — a floor that moves with
    # exposure and, at realistic levels, is not far off the 26.7% a badly
    # misaligned trace scored. Collapsing the wavelength axis first averages
    # that noise down instead of rectifying it: the same two traces score
    # 0.6% and 20.6%, near enough exposure-independent.
    #
    # The blind spot is an asymmetry that cancels in the wavelength sum. The
    # IMAGE is per-pixel and still shows it — the picture says where, the
    # number says how much.
    return out, peak, _asymmetry(ac, k), fold
