"""Lillypad — FROG ultrafast-pulse diagnostic acquisition software.

Deliberately empty apart from the version. Nothing is re-exported here, so
importing any part of Lillypad never drags in Qt, matplotlib or a vendor SDK it
did not ask for — `lillypad.paths` and `lillypad.settings` in particular must
stay importable in a bare interpreter, because the frozen build's startup path
and the test suite both rely on it.
"""

__version__ = "1.0.0"
