"""The Qt application layer.

    perf     the opt-in render/allocation probe
    widgets  small reusable widgets, built from the design layer
    workers  the three GUI worker threads

Nothing is re-exported here. Importing a package __init__ that pulls Qt in is
how lillypad.scan nearly lost its Qt-free core (see tests/test_layering.py), and
naming the module you want says which layer you are reaching into.
"""
