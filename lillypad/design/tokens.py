"""The visual vocabulary: colours, type scale, radii.

Every colour and every font size in the application is defined here and nowhere
else. `qss/app.qss` refers to these by `$name`, and the plot styling in
`mpl_style.py` reads the same palette dict, so the two halves of the interface
cannot drift apart.

No Qt import. A QApplication is not needed to know what colour something is, and
keeping it out means the token set can be inspected and tested headlessly.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Chrome
# ─────────────────────────────────────────────────────────────────────────────
# Layering: bg (window) < plot_bg (plot panel) < surface (controls/dialogs),
# each a distinct step so panels and controls read as raised. Dark text is
# #e6edf3, not near-white — ~13:1 on bg avoids halation from max-contrast text.
DARK_PALETTE = {
    "bg": "#0d1117", "surface": "#1c2128", "border": "#363d47",
    "border_hover": "#444c56",
    "accent": "#58a6ff", "accent2": "#79c0ff", "text": "#e6edf3",
    "text_dim": "#b0bcc9", "text_disabled": "#6e7681",
    "danger": "#ff7b72", "warn": "#f0883e", "good": "#3fb950",
    "plot_bg": "#161b22", "grid": "#2a3140",
}

# Neutral gray ramp (no warm cast) so the cool blue accents don't clash.
LIGHT_PALETTE = {
    "bg": "#f2f3f5", "surface": "#ffffff", "border": "#d0d7de",
    "border_hover": "#afb8c1",
    "accent": "#0969da", "accent2": "#0550ae", "text": "#1f2328",
    "text_dim": "#57606a", "text_disabled": "#8c959f",
    "danger": "#cf222e", "warn": "#bc4c00", "good": "#1a7f37",
    "plot_bg": "#ffffff", "grid": "#d8dee4",
}

# The live palette. MUTATED IN PLACE on a theme switch (see theme.install_theme)
# and never rebound, so that every `PALETTE[...]` read anywhere in the program —
# including the ones inside already-constructed matplotlib artists — follows the
# switch. Rebinding this name would break theme switching silently, in one panel,
# at some later date. A test asserts its identity is stable across a switch.
PALETTE = dict(DARK_PALETTE)


# ─────────────────────────────────────────────────────────────────────────────
# Data colours — deliberately NOT palette keys
# ─────────────────────────────────────────────────────────────────────────────
# These identify measurements, not chrome, so they are chosen for discriminability
# rather than to match a theme, and they stay the same in dark and light.

# Per-spectrometer curves in multi-spectrometer mode. Okabe-Ito sky blue and
# orange — the highest-contrast pair in that set. Both read clearly on the dark
# and the light plot background, and they stay separable under deuteranopia,
# protanopia and tritanopia (and in greyscale), which the theme accents do not.
MEMBER_COLORS = ("#56B4E9", "#E69F00")      # SLOT order: (S1, S2)

# Shading for the overlap band in the per-spectrometer view: the stretch the
# stitch factor is fitted over and the two spectra are crossfaded across. Green
# because it has to sit UNDER both member curves without being confused for
# either, and Okabe-Ito's bluish green is the remaining unused hue in that set.
OVERLAP_BAND_COLOR = "#009E73"
OVERLAP_BAND_ALPHA = 0.16

# The two symmetry-difference curves of alignment mode, S(+x)-S(-x) and
# S(+2x)-S(-2x). Okabe-Ito again, and deliberately the two hues MEMBER_COLORS
# and the overlap band do NOT use: a difference curve can be on screen at the
# same time as either of those and must never be mistaken for one.
DIFF_COLORS = ("#D55E00", "#CC79A7")        # (+/-x, +/-2x)


# ─────────────────────────────────────────────────────────────────────────────
# Type
# ─────────────────────────────────────────────────────────────────────────────
FONT_STACK = "'Segoe UI','DejaVu Sans',Arial,sans-serif"
MONO_STACK = "'Cascadia Mono','Consolas',monospace"

# The type scale, named by role. Steps rather than free-form sizes, so "make the
# secondary text a little larger" is one edit instead of eleven.
#
#   label    uppercase group-box titles, tracked out
#   small    secondary and dim text: hints, status bar, the compact buttons
#   body     the default — controls, buttons, menus
#   value    numeric readouts, which have to be legible across the room
#   display  the "nothing connected" message on an empty plot
#
# A rule that wants the neighbouring step uses that step's name; nothing here is
# sized by accident. Weights follow the usual 400/500/600/700 ladder.
TYPE = {
    "font_stack": FONT_STACK,
    "font_mono": MONO_STACK,
    "fs_label": "11px",
    "fs_small": "12px",
    "fs_body": "13px",
    "fs_value": "15px",
    "fs_display": "22px",
    "fw_normal": "400",
    "fw_medium": "500",
    "fw_semibold": "600",
    "fw_bold": "700",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
# Corner radii, smallest to largest: an indicator, an input, a button, a panel,
# and the fully rounded scrollbar track.
#
# Corner-specific radii in the stylesheet (border-top-right-radius on the spinbox
# buttons) keep their literal values on purpose: they have to agree with the
# adjacent border-radius, and a separate token would let the two drift apart
# without anything noticing.
METRICS = {
    "radius_xs": "3px",
    "radius_sm": "4px",
    "radius": "5px",
    "radius_lg": "6px",
    "radius_pill": "7px",
}
