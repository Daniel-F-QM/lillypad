"""Vendor-agnostic spectrometer enumeration and opening.

open_spectrometer is the ONE place a vendor tag is mapped to an adapter class.
The mapping is a literal if/elif over imported classes, never a dict of module
names: PyInstaller finds imports by reading bytecode, so a string-keyed lookup
would drop the drivers from the frozen build and fail only on the bench.

The tagged-id helpers this builds on (spec_ident, split_spec_ident, SPEC_VENDORS)
live in base.py, because the id format is part of what SpectrometerBase promises.
"""

from __future__ import annotations

from lillypad.hardware.base import SpectrometerBase, spec_ident, split_spec_ident
from lillypad.hardware.spectrometers.avantes import (AvantesSpectrometer,
                                                     list_avantes_spectrometers)
from lillypad.hardware.spectrometers.seabreeze import (
    SeabreezeSpectrometer, list_seabreeze_spectrometers)

def list_spectrometers(seabreeze_backend: str = "pyseabreeze") -> list[tuple[str, str]]:
    """Every attached spectrometer, from every vendor, as
    [(label, "vendor:serial"), ...] — the same shape list_kinesis_stages uses,
    so the GUI's device picker drives all of them.

    A vendor whose SDK is missing or whose enumeration fails contributes
    NOTHING and raises nothing. This is background enumeration — it fills the
    picker and the multi-spectrometer submenus — and one absent driver must not
    hide the other vendor's devices or turn every submenu into an error. Code
    that wants the failure reported (because the operator explicitly pressed
    that vendor's connect button) calls that vendor's list_* helper directly.
    """
    out: list[tuple[str, str]] = []
    try:
        for model, serial in list_seabreeze_spectrometers(seabreeze_backend):
            out.append((model, spec_ident("seabreeze", serial)))
    except Exception:
        pass
    try:
        for model, serial in list_avantes_spectrometers():
            out.append((model, spec_ident("avantes", serial)))
    except Exception:
        pass
    return out


def open_spectrometer(ident: str,
                      seabreeze_backend: str = "pyseabreeze") -> SpectrometerBase:
    """Open the device a tagged id names. The ONLY place a vendor tag is mapped
    to an adapter class, so a new vendor is one entry here rather than one per
    call site in the GUI."""
    vendor, serial = split_spec_ident(ident)
    if vendor == "seabreeze":
        # "?" = the serial could not be read at enumeration time; fall back to
        # first-device auto-connect rather than refusing outright.
        return (SeabreezeSpectrometer(serial=serial, backend=seabreeze_backend)
                if serial and serial != "?"
                else SeabreezeSpectrometer(backend=seabreeze_backend))
    if vendor == "avantes":
        return AvantesSpectrometer(serial=serial if serial != "?" else None)
    raise RuntimeError(f"No adapter for spectrometer vendor {vendor!r}.")
