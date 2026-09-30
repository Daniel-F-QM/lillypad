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
# None for the autocorrelation peak.


def symmetry_center(ac, delays, fold_fs=None):
    """Column index the trace is folded about.

    The autocorrelation maximum by default: the AC is the trace summed over
    wavelength, so its peak is where the two pulses overlap best — the delay
    the trace SHOULD be symmetric about. Not the middle column, which would
    make a scan range that was not centred on the overlap report its own
    offset as asymmetry.

    A hand-placed line (dragged on the plot) overrides it, snapped to the
    nearest measured column: the fold is an index into the trace, so there
    is no sub-column position it could honour without resampling.
    """
    if fold_fs is None:
        return int(np.nanargmax(np.asarray(ac, float)))
    d = np.asarray(delays, float)
    return int(np.argmin(np.abs(d - float(fold_fs))))

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
    c = symmetry_center(ac, delays, fold_fs)
    fold = float(np.asarray(delays, float)[c])
    out = t.copy()                   # left of the fold: measured, untouched
    h = min(c, n - 1 - c)            # columns both sides can cover
    peak = float(np.nanmax(t)) if t.size else 0.0
    if h < 1:
        out[:, c + 1:] = np.nan      # fold on an edge: nothing to compare
        return out, peak, 0.0, fold
    left  = t[:, c - h:c + 1]        # columns c-h … c
    right = t[:, c:c + h + 1]        # columns c   … c+h
    d = np.abs(right - left[:, ::-1])
    out[:, c:c + h + 1] = d
    out[:, c + h + 1:]  = np.nan     # beyond the overlap: no counterpart
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
    a = np.asarray(ac, float)
    total = float(np.nansum(a))
    score = (float(np.nansum(np.abs(a[c:c + h + 1] - a[c - h:c + 1][::-1])))
             / total) if total > 0 else 0.0
    return out, peak, score, fold
