"""Shared fixtures for the Lillypad test suite.

These tests are the characterization net for the package refactor: they assert
what the code does TODAY, so that a move which changes behaviour fails here
instead of on the bench. The assertions are carried over from the three
`if __name__ == "__main__"` self-test blocks in hardware.py, scan.py and
avantes.py, with their measured numbers kept verbatim rather than paraphrased.

Nothing here touches hardware or opens a window. Every vendor SDK import in the
application is lazy and function-local, so the whole suite runs on a machine
with nothing but numpy and PySide6 installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Offscreen Qt, set before anything can import PySide6. Only the scan-worker
# test needs Qt at all (QThread), and it never constructs a widget — but a
# future test that accidentally does should not try to reach a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]

# pyproject.toml's `pythonpath = ["."]` covers a normal `pytest` run from the
# repo root. This keeps `pytest tests/` working from anywhere, and keeps the
# suite runnable once the flat modules move under lillypad/.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def calibration_dir() -> Path:
    """The repo's shipped calibration files.

    Replaces hardware.py's hardcoded `Path(__file__).resolve().parent /
    "calibration_files"` (line 2867 of the self-test). Tests that need these
    files skip when they are absent rather than failing, exactly as the
    self-test degraded to a shorter message — the files are data, and a
    checkout without them is not a broken program.
    """
    return REPO_ROOT / "calibration_files"


@pytest.fixture
def move_counter():
    """Wrap a stage's `_move_to_raw` and count the raw moves it issues.

    Both the hardware.py and scan.py self-tests hand-rolled this same closure to
    prove that the long-move splitter and the backlash approach cost the moves
    they should and no more. It is the only way to see that from outside, since
    `move_to` is deliberately the only entry point.
    """
    def attach(stage):
        n = [0]
        original = stage._move_to_raw

        def counted(position_mm, _f=original, _n=n):
            _n[0] += 1
            return _f(position_mm)

        stage._move_to_raw = counted
        return n
    return attach
