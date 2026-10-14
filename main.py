#!/usr/bin/env python
"""Lillypad — FROG ultrafast-pulse diagnostic.

    python main.py

The launcher, and nothing else: the startup sequence is in
lillypad/app/main.py, where the order it has to happen in can be explained
next to it.

Named main.py rather than lillypad.py because a `lillypad.py` module and the
`lillypad/` package beside it shadow each other — and Windows' filesystem is
case-insensitive, so `Lillypad.py` does not escape it either. The built
executable is still Lillypad.exe; PyInstaller takes its name from the spec, not
from this file.
"""

from lillypad.app.main import main

if __name__ == "__main__":
    main()
