"""Saving a FrogResult: .dwc, .npz and .csv.

The whole file-format surface, in one place, so a format regression is
reviewable on its own. .dwc in particular is a fixed third-party layout
(Femtosoft), which means a formatting slip produces a file that loads into
someone else's software and is quietly wrong rather than one that fails.

Takes a FrogResult but does not import it — these functions only ever read
attributes off what they are handed.
"""

from __future__ import annotations

import numpy as np

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


def _trim_empty_rows(wl, trace):
    """Drop the wavelength rows at either END that are exactly zero at every
    delay. Rows inside the range are kept, zero or not.

    Those rows come from an intensity calibration whose factor is 0 at its
    ends (a detector cut off where its response dies). Every pixel there is
    multiplied by 0, the dark is too, so the row is exactly 0 everywhere and
    carries no data. A trace with no non-zero row at all is returned whole,
    rather than as a file with no rows.
    """
    live = np.flatnonzero(np.any(trace != 0, axis=1))
    if live.size == 0:
        return wl, trace
    keep = slice(live[0], live[-1] + 1)
    return wl[keep], trace[keep]


def _export_rows(result, trim_empty_rows):
    """(delays, wavelengths, trace) as the two text formats write them:
    ascending delay, dark-corrected, optionally edge-trimmed."""
    delays, trace = _ascending(result)
    wl = np.asarray(result.wavelengths_nm, float)
    if wl.size != trace.shape[0]:
        raise ValueError(f"wavelength vector ({wl.size}) does not match the "
                         f"trace ({trace.shape[0]} rows)")
    if trim_empty_rows:
        wl, trace = _trim_empty_rows(wl, trace)
    return delays, wl, trace


def _delay_increment(result, delays):
    """Uniform delay step (fs). Taken from the realised axis, falling back to
    the configured step for a single-column scan."""
    if delays.size > 1:
        return float(np.median(np.abs(np.diff(delays))))
    return float(result.config.delay_step_fs)


def write_dwc(path, result, trim_empty_rows=False):
    """Write a FrogResult as a .dwc trace (background-subtracted).

    `trim_empty_rows` drops all-zero wavelength rows at either end (see
    _trim_empty_rows); the header's point count follows the trimmed data."""
    delays, wl, trace = _export_rows(result, trim_empty_rows)
    n_wl, n_delays = trace.shape

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
    background frames, the per-column saturation and stage-fault records, and
    metadata the text formats cannot carry. (.dwc and .csv have fixed layouts
    and take none of this; the archive is the only format that can carry it.)

    Everything here is in ACQUISITION order, so a high->low scan is stored
    descending. That is self-consistent — trace, delays_fs and both position
    vectors share the ordering — but it is NOT the order .dwc and .csv use,
    since those flip to ascending (see _ascending). Line up an archive against
    a .dwc by delay value, not by column index.
    """
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
    if result.stage_faults is not None:
        # Also per column: "" is clean, anything else names what the stage
        # reported and means that column's delay label cannot be trusted.
        data["stage_faults"] = np.asarray(result.stage_faults, dtype=str)
    # The dark that WAS subtracted from .dwc/.csv, so an archive can reproduce
    # either form: trace - dark is what the other two files contain.
    if result.dark is not None:
        data["dark"] = np.asarray(result.dark, float)
    # The delay-extreme reference frames, which nothing subtracted. Named for
    # what they are: they used to be called background_before/after, back when
    # they were what the exports were corrected with.
    if result.delay_reference_before is not None:
        data["delay_reference_before"] = result.delay_reference_before
    if result.delay_reference_after is not None:
        data["delay_reference_after"] = result.delay_reference_after
    np.savez(path, **data)


def write_csv(path, result, trim_empty_rows=False):
    """Write a FrogResult as a spreadsheet-friendly CSV: a delay header row
    (fs) over one row per wavelength (nm), background-subtracted.
    `trim_empty_rows` as for write_dwc."""
    delays, wl, trace = _export_rows(result, trim_empty_rows)
    fmt = _row_fmt(trace.shape[1], "%.6g", ",")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("wavelength_nm\\delay_fs," + _row_fmt(delays.size, "%.6g", ",") % tuple(delays))
        f.write("\n")
        for w, row in zip(wl, trace):
            f.write(f"{w:.6g}," + fmt % tuple(row) + "\n")
