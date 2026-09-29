"""Plot styling: the matplotlib backend, the global rcParams, and the figure
geometry the canvas lays itself out with.

Importing this module SELECTS THE BACKEND. That is an import-time side effect, and
a deliberate one: every module that builds any part of the figure imports this
first, above `from matplotlib.figure import Figure`, so the backend is chosen
before matplotlib can pick one for itself. Module import is idempotent, so there
is no "the bootstrap must run first" rule to remember.

The colours the canvas draws with are NOT here — they come from
design.tokens.PALETTE, read at paint time so a theme switch reaches artists that
already exist.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
matplotlib.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.0, "axes.titleweight": "bold", "axes.titlesize": 9.5,
    "axes.labelsize": 9,
})

# Colormaps offered for the FROG trace; the FIRST ENTRY IS THE DEFAULT, so the
# order is load-bearing.
#
# "jet" leads because it is what the group uses most, NOT because it is the best
# choice — it is the one entry here that is not perceptually uniform. The other
# five map equal steps in count to equal steps in apparent brightness; jet does
# not, so a trace read off it shows structure the data does not have. It is the
# default because matching what colleagues are used to reading matters more here
# than the colour science, and that is a deliberate trade, not an oversight.
TRACE_COLORMAPS = ["jet", "magma", "viridis", "inferno", "plasma", "cividis"]


# ── Plot geometry, in figure fractions (see FrogCanvas._layout_axes) ─────────
# Margins are sized for the 9-pt plot fonts: LEFT holds the spectrum's y label
# plus its tick labels, COLGAP the right column's y label, and BOT the bottom
# row's tick labels and x label.
_GEO_R = 0.99
# TOP is the header band: the panel title and that panel's overlay buttons share
# one line above the axes. Sized in pixels for the same reason as LEFT below —
# it holds a fixed-size Qt button, not a fraction of the figure.
_GEO_TOP_PX  = 34    # HDR_BTN (24) + HDR_PAD (6) + 4 px above the button
_GEO_TOP_MAX = 0.12  # …but never eat this much of a short window
# The left margin and the column gap hold text at a FIXED point size, so they
# are sized in logical pixels and only converted to a fraction at layout time:
# one fraction that fits a small window wastes half of a large one (the old
# 0.10 left margin was 90 px at 900 px wide but 192 px at 1920, for content
# that never needs more than the measured worst case below).
_GEO_L_PX   = 68     # y label + 5-digit tick labels at 8.5 pt measure ~58 px
_GEO_L_MAX  = 0.14   # …but never eat this much of a narrow window
_GEO_GAP_PX = 70     # column-gap floor: the right column's y label and tick
                     # labels, plus clearance for the split handle beside them
_GEO_ROW_RATIO = 1.8        # tall row : short row, both modes
# Spectrum : deviation panel, when alignment mode splits the spectrum's cell.
# Taller than _GEO_ROW_RATIO: the deviations are a strip either side of zero,
# not a curve that has to fill a panel, and the spectrum it takes room from is
# the one being aligned.
_GEO_DEV_RATIO = 2.4
_GEO_V_BOT     = 0.095
_GEO_V_COLGAP  = 0.055
_GEO_V_HSPACE  = 0.31       # the gridspec hspace the vertical layout was tuned with
_GEO_H_BOT     = 0.095
_GEO_H_COLGAP  = 0.065      # wider: two stacked y-axes share this gap, and the
                            # autocorrelation's tick labels run longer than the
                            # trace's, so 0.055 leaves its y label touching.
_GEO_H_ROWGAP  = 0.0        # flush: the stacked pair shares one delay axis, so
                            # the trace's bottom spine IS the AC's top spine

# The autocorrelation panel is drawn normalized to its own peak, so its view is
# fixed: 0…1 plus a little headroom to keep the peak off the top spine.
_AC_YLIM = (0.0, 1.05)

# Title offsets in inches, so the gaps are DPI- and resize-independent.
_TITLE_ABOVE_IN  = 6 / 72   # centres the 12 pt title in the header band
_TITLE_INSET_IN  = (0.08, 0.06)   # (right, down) from the axes' top-left corner
# Strip an INSIDE title occupies, measured down from the axes' top spine: the
# inset above the text, the 12 pt line itself, and a little air under it. A
# panel titled this way keeps its curves below the band — see _title_band_frac.
_TITLE_BAND_IN   = _TITLE_INSET_IN[1] + 12 / 72 + 4 / 72

# Resolution the toolbar's camera prints at (FrogCanvas.save_picture). The
# figure is printed at the size it is ON SCREEN, so this is purely a detail
# multiplier over the dpi the canvas is drawn at (100, times the display's
# device-pixel ratio): every line, font and gap keeps exactly the proportion it
# has in the window, and only the pixel count goes up. 300 dpi is what a
# journal asks for.
PICTURE_DPI = 300
# …but the page's pixel size is the WINDOW's size times that ratio, and a
# maximized window on a 4K panel would ask for ~10000 px of trace image —
# tens of megabytes, and seconds of it on the GUI thread, for detail no
# screen or printer will ever resolve. Past this the dpi is reduced to fit;
# a page that big is already a long side of 20 inches at 300 dpi.
PICTURE_MAX_PX = 6000
