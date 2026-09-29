"""Spectrometers, one module per vendor.

Each implements SpectrometerBase and imports its SDK lazily. registry.py maps a
"vendor:serial" id to the right one; nothing else should need to choose.
"""
