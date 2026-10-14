"""The Qt application layer.

    state        InstrumentState — what the whole program has to agree about
    widgets      small reusable widgets, built from the design layer
    workers      the three GUI worker threads, together
    perf         the opt-in render/allocation probe (LILLYPAD_PERF=1)
    text         two long user-facing strings
    canvas/      FrogCanvas: one concrete class plus five mixins
    dialogs/     the nine pop-up windows
    controllers/ the window's behaviour, one module per concern
    window/      FrogWindow, and the one-time act of building it

Read `window/main_window.py` first — `__init__` is the wiring diagram, and its
order is load-bearing throughout. Then whichever controller owns the thing you
came for; `controllers/__init__.py` says what each one is.

Nothing is re-exported here. Importing a package __init__ that pulls Qt in is
how lillypad.scan nearly lost its Qt-free core (see tests/test_layering.py), and
naming the module you want says which layer you are reaching into.
"""
