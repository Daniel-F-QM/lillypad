"""The intensity-calibration file loader.

Ported from hardware.py's self-test (lines 2719-2741). The ragged-line case is
not hypothetical: it is the exact shape Niquest_New.txt had, a trailing
single-field line, which used to take np.loadtxt down with "the number of
columns changed" and leave the device silently on raw counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from lillypad.hardware import load_calibration_file


def test_clean_file_loads_sorted(tmp_path):
    p = tmp_path / "good.txt"
    p.write_text("400 0.5\n500 1.0\n600 0.25\n")
    wl, fac, skipped = load_calibration_file(p)
    assert skipped == 0
    assert len(wl) == 3
    assert fac[1] == 1.0


def test_rows_come_back_sorted_by_wavelength(tmp_path):
    p = tmp_path / "unsorted.txt"
    p.write_text("600 0.25\n400 0.5\n500 1.0\n")
    wl, fac, _ = load_calibration_file(p)
    assert list(wl) == [400.0, 500.0, 600.0]
    assert list(fac) == [0.5, 1.0, 0.25]


def test_comments_are_ignored(tmp_path):
    p = tmp_path / "commented.txt"
    p.write_text("# wavelength factor\n400 0.5\n500 1.0  # mid\n600 0.25\n")
    wl, _, skipped = load_calibration_file(p)
    assert len(wl) == 3 and skipped == 0


def test_one_ragged_line_is_skipped_and_counted(tmp_path):
    """A single bad row must cost that row, not the file."""
    p = tmp_path / "ragged.txt"
    p.write_text("400 0.5\n500 1.0\n600 0.25\n 0")
    wl, fac, skipped = load_calibration_file(p)
    assert skipped == 1
    assert len(wl) == 3


def test_unusable_file_is_refused(tmp_path):
    """A silently empty calibration would be worse than an error."""
    p = tmp_path / "junk.txt"
    p.write_text("# only a comment\nnot numbers\n")
    with pytest.raises(RuntimeError):
        load_calibration_file(p)


def test_single_row_is_refused(tmp_path):
    """Two rows are the minimum: one point cannot be interpolated."""
    p = tmp_path / "one.txt"
    p.write_text("400 0.5\n")
    with pytest.raises(RuntimeError):
        load_calibration_file(p)


def test_shipped_calibration_files_load(calibration_dir):
    files = sorted(calibration_dir.glob("*.txt"))
    if not files:
        pytest.skip(f"no calibration files in {calibration_dir}")
    for f in files:
        wl, fac, _ = load_calibration_file(f)
        assert wl.size >= 2, f
        assert np.all(np.diff(wl) >= 0), f"{f.name} came back unsorted"
        assert np.all(np.isfinite(fac)), f
