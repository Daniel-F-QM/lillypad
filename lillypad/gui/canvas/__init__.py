"""The plotting canvas.

    figure.py       FrogCanvas itself: the Signals, __init__, the blit core
    layout.py       where the panels sit and how they move on a resize
    panels.py       one mixin per panel — spectrum, trace, autocorrelation
    axes_api.py     axis limits, fit/reset, and the auto-scalers
    interaction.py  zoom, typed axis bounds, the fold line

The window imports FrogCanvas from here and nothing else.
"""

from lillypad.gui.canvas.figure import FrogCanvas

__all__ = ["FrogCanvas"]
