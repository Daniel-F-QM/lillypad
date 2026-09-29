"""The visual design layer: colours, type, icons, the stylesheet, plot styling.

This package never imports anything from the GUI. That is the whole point of it
being separate — the dependency runs one way, and "does the design layer know
about widgets?" is answerable by one grep rather than by reading 9,000 lines. A
test enforces it.

Nothing is re-exported here on purpose: `from lillypad.design.tokens import
PALETTE` says which layer a colour came from, and importing the theme should not
drag matplotlib in behind it.

  tokens          colours, the type scale, radii — no Qt
  qss/app.qss     the stylesheet, with $tokens
  stylesheet      renders app.qss through string.Template
  icons           icon files on disk + the application icon
  icons_runtime   the three marks that must be painted at runtime
  theme           install_theme: palette + Qt palette + stylesheet
  mpl_style       matplotlib backend, rcParams, figure geometry
  geometry        panel header band metrics, in logical pixels
"""
