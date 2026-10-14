"""The window itself, and the one-time act of building it.

    main_window.py  FrogWindow: what it owns and how it is wired together
    build_ui.py     free functions that fill it with widgets

Split because they are two different kinds of reading. main_window.py is the
wiring diagram — 500 lines you read once to understand the program's shape.
build_ui.py is 680 lines of construction you read when you need to find a
particular control.
"""

from lillypad.gui.window.main_window import FrogWindow

__all__ = ["FrogWindow"]
