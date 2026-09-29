"""Panel header band geometry, in logical pixels.

The overlay buttons above each plot panel (positioned by the window's
_position_panel_buttons). Square and small enough that a row of them reads as one
line with the panel title beside it.
"""

from __future__ import annotations

HDR_BTN = 24
HDR_GAP = 4
# Gap between the button row's bottom edge and the axes' top spine.
HDR_PAD = 6

# Space after a panel title before the buttons that qualify it, and the clear
# span that separates that group from the right-aligned row. The second is much
# the larger of the two: it is the only thing telling the eye these are two
# groups and not one row of eight.
HDR_TITLE_GAP = 10
HDR_GROUP_GAP = 28

# Icon side inside a header button. EVEN, and even after the 1 px border is
# taken off both sides of HDR_BTN: (24 - 2 - 14) / 2 = 4 exactly, so Qt centres
# the icon on a whole pixel instead of rounding half a pixel one way.
HDR_ICON = 14
