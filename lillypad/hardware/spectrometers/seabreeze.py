"""Ocean Optics / Ocean Insight spectrometers, via python-seabreeze.

Two backends: pyseabreeze (pyusb + libusb, the default, and the only one that
supports the newer Ocean Insight models) and cseabreeze (the vendor C library).
Both are imported lazily.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from lillypad.hardware.base import SpectrometerBase, spec_ident

# (SR/ST/HDX series), while the devices cseabreeze covers also work through it.
SEABREEZE_BACKENDS = ("pyseabreeze", "cseabreeze")


def _ensure_libusb_dll() -> None:
    """pyseabreeze drives USB through pyusb, which on Windows locates
    libusb-1.0.dll by searching PATH — it knows nothing about the pip-installed
    `libusb-package` wheel that actually carries the DLL. Bridge the two by
    prepending the wheel's DLL folder to PATH (idempotent, no-op if the wheel
    is missing or a system-wide libusb already exists)."""
    if sys.platform != "win32":
        return
    try:
        import libusb_package
        dll_dir = str(Path(libusb_package.get_library_path()).parent)
    except Exception:
        return
    if dll_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]


def select_seabreeze_backend(backend: str) -> None:
    """Make `backend` ("pyseabreeze" or "cseabreeze") the active seabreeze
    backend. seabreeze binds its backend lazily on first use and then caches
    it in three places, so switching at runtime means clearing all three;
    otherwise seabreeze.use() only warns and the old backend stays live.
    Any spectrometer opened through the previous backend must already be
    closed — its API is shut down here."""
    if backend not in SEABREEZE_BACKENDS:
        raise ValueError(f"backend must be one of {SEABREEZE_BACKENDS}, "
                         f"got {backend!r}")
    if backend == "pyseabreeze":
        _ensure_libusb_dll()
    import seabreeze
    import seabreeze.backends
    mod = sys.modules.get("seabreeze.spectrometers")
    cached = None
    if mod is not None:
        # The lazy binding lands in either place first depending on whether
        # list_devices() or Spectrometer() ran first — check both.
        descriptor = mod.Spectrometer.__dict__["_backend"]
        cached = mod.__dict__.get("_lib") or descriptor._backend
    if cached is not None and cached.__name__.endswith(backend):
        return                      # already live — nothing to reset
    if cached is not None:
        api = getattr(mod.list_devices, "_api", None)
        if api is not None:
            try:
                api.shutdown()
            except Exception:
                pass
            del mod.list_devices._api
        mod.__dict__.pop("_lib", None)
        mod.__dict__.pop("SeaBreezeDevice", None)
        descriptor._backend = None
    if (seabreeze.backends.BackendConfig.requested != backend
            or cached is not None):
        seabreeze.use(backend)


def list_seabreeze_spectrometers(backend: str = "pyseabreeze") -> list[tuple[str, str]]:
    """Enumerate attached Ocean Optics spectrometers without adopting any.
    Returns [(model, serial), ...]; entries whose metadata cannot be read come
    back as "?" placeholders rather than being dropped. Lives here so the GUI
    never touches the vendor SDK, and so backend selection precedes the
    spectrometers import."""
    select_seabreeze_backend(backend)
    from seabreeze.spectrometers import list_devices
    out = []
    for dev in list_devices():
        try:
            model = str(getattr(dev, "model", "?"))
            serial = str(getattr(dev, "serial_number", "?"))
        except Exception:
            model, serial = "?", "?"
        out.append((model, serial))
    return out


class SeabreezeSpectrometer(SpectrometerBase):
    """Ocean Optics / Ocean Insight spectrometer via python-seabreeze."""
    def __init__(self, device=None, serial: str | None = None,
                 backend: str = "pyseabreeze"):
        select_seabreeze_backend(backend)
        self.backend = backend      # the GUI checks this before backend swaps
        from seabreeze.spectrometers import Spectrometer, list_devices

        if device is not None:
            self._spec = device if hasattr(device, "intensities") \
                         else Spectrometer(device)
        elif serial is not None:
            self._spec = Spectrometer.from_serial_number(serial)
        else:
            devs = list_devices()
            if not devs:
                raise RuntimeError("No spectrometers found.")
            self._spec = Spectrometer(devs[0])

        self.name = (f"{getattr(self._spec, 'model', 'spectrometer')} "
                     f"[{getattr(self._spec, 'serial_number', '?')}]")
        # Taken from the OPENED device, not from the `serial` argument: the
        # device= and auto-connect paths never see one, and this is the id a
        # multi-spectrometer slot has to be able to reopen later.
        self.spec_id = spec_ident(
            "seabreeze", getattr(self._spec, "serial_number", None))
        self._wl = np.asarray(self._spec.wavelengths(), dtype=float)

        # Detector full scale, as the device itself reports it (seabreeze's
        # `max_intensity`). Guarded: not every backend/model exposes it, and a
        # missing value must degrade to "cannot check" rather than to a wrong
        # threshold that would either cry wolf or stay silent through clipping.
        try:
            self.max_counts = float(self._spec.max_intensity)
        except Exception:
            self.max_counts = None

        # Exposure range, again as the device reports it. Ocean models differ
        # widely (1 ms on a USB2000+, 3.8 ms on an HR4000, 10 us on some), so
        # asking beats assuming. Guarded: not every backend exposes the limits,
        # and missing ones must fall back to the safe default range rather than
        # to a wrong one.
        try:
            lo_us, hi_us = self._spec.integration_time_micros_limits
            self.min_integration_ms = float(lo_us) / 1000.0
            self.max_integration_ms = float(hi_us) / 1000.0
        except Exception:
            pass

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def set_integration_time(self, ms: float) -> None:
        # Rounded, not truncated: the API takes whole microseconds, and
        # truncating a fractional-millisecond exposure biases every one of
        # them low (3.999 ms -> 3998 us).
        self._spec.integration_time_micros(int(round(float(ms) * 1000.0)))
        self.integration_ms = float(ms)   # keeps the live feed's pacing honest

    def acquire(self) -> np.ndarray:
        return np.asarray(self._spec.intensities(), dtype=float)

    def disconnect(self) -> None:
        try:
            self._spec.close()
        except Exception:
            pass
