"""Avantes AvaSpec spectrometers.

This is the ADAPTER. The ctypes wrapper around the vendor DLL is
lillypad.hardware.vendor.avaspec, which knows nothing about Lillypad; everything
that translates between it and SpectrometerBase lives here.
"""

from __future__ import annotations

import numpy as np

from lillypad.hardware.base import SpectrometerBase, spec_ident

def _avantes_module():
    """The Avantes vendor module, with our program folder registered as a DLL
    search location.

    Every Avantes entry point goes through here for two reasons. One place means
    the registration cannot be forgotten at a new call site. And the import stays
    a literal statement inside a function body, which is what lets PyInstaller's
    analysis find the module with no hiddenimports entry — an
    importlib.import_module here would break the frozen build silently.

    The vendor module deliberately knows nothing about Lillypad, so it cannot
    work out where we are installed; app_dir() is us telling it. Frozen, that is
    the folder holding the .exe, which is what the build has always documented as
    a place to drop a specific DLL — the module's own fallback only ever looked
    inside the _internal bundle.

    Bound to the name `avantes` by the callers below because that is what the
    vendor's own documentation calls it; the file is avaspec.py to match the DLL.
    """
    from lillypad.hardware.vendor import avaspec
    from lillypad.paths import app_dir
    avaspec.add_search_dir(app_dir())
    return avaspec


def list_avantes_spectrometers() -> list[tuple[str, str]]:
    """Enumerate attached Avantes spectrometers without adopting any.
    Returns [(model, serial), ...].

    Raises when the AvaSpec DLL is missing or the enumeration fails — callers
    that want a quiet "nothing found" (background enumeration) go through
    list_spectrometers, which swallows it. The message is worth surfacing when
    the operator explicitly asked for an Avantes: it names the package to
    install.
    """
    avantes = _avantes_module()
    return [(name, serial) for name, serial, _status in avantes.list_devices()]


def avantes_trigger_options() -> dict[str, list[tuple[str, int]]]:
    """Labelled choices for the Avantes trigger controls, as
    {"mode"/"source"/"source_type": [(label, value), ...]}.

    Read from the vendor module rather than restated here so the numbers have
    exactly one definition. The GUI calls this only while an Avantes is
    connected, so the lazy import costs nothing on a machine without the DLL —
    and it is what keeps frog_gui_fast from importing avantes directly.
    """
    avantes = _avantes_module()
    return {
        "mode": [("Free running", avantes.SW_TRIGGER_MODE),
                 ("Hardware trigger", avantes.HW_TRIGGER_MODE),
                 ("Single scan", avantes.SS_TRIGGER_MODE)],
        "source": [("External", avantes.EXTERNAL_TRIGGER),
                   ("Sync", avantes.SYNC_TRIGGER)],
        "source_type": [("Edge", avantes.EDGE_TRIGGER_SOURCE),
                        ("Level", avantes.LEVEL_TRIGGER_SOURCE)],
    }


class AvantesSpectrometer(SpectrometerBase):
    """Avantes AvaSpec spectrometer via the vendor DLL (see avantes.py).

    The vendor module is imported lazily, inside __init__, so `import hardware`
    and the simulator self-test still work on a machine with no AvaSpec DLL —
    the same arrangement ZaberStage and PiezoJenaStage use for their SDKs.

    Two device settings are entangled with things the rest of the app assumes,
    and this adapter owns both couplings so no consumer has to know about them:

      * `integration_ms` reports the cost of ONE acquire(), which is the
        exposure times the on-board average count — not the exposure. The live
        feed paces itself off this value and derives its handover timeout from
        it, so reporting the bare exposure while a frame silently costs N times
        that makes the feed spin hot and the device-handover time out.
      * `max_counts` follows the ADC resolution. Switching to the 16-bit ADC
        moves full scale from 16383 to 65535, and anything that cached the old
        ceiling would then either cry wolf or stay silent through clipping.
        `set_high_res_adc` returns whether the mode actually changed so the UI
        can re-arm its saturation state only when it did.
    """

    def __init__(self, serial: str | None = None, high_res_adc: bool = True):
        avantes = _avantes_module()
        self._spec = avantes.AvaSpec(serial=serial, high_res_adc=high_res_adc)
        try:
            self.serial = self._spec.serial
            self.spec_id = spec_ident("avantes", self.serial)
            self.name = f"{self._spec.model} [{self.serial}]"
            self._wl = np.asarray(self._spec.wavelengths, dtype=float)
            # THIS sensor's range, probed from the device at connect — not the
            # library's much wider one. The gap is two orders of magnitude (a
            # CMOS unit takes 9 us where the library claims 2 us and an ILX
            # CCD needs 1.1 ms), and offering an exposure the device will
            # refuse is worse than not offering it: the refusal surfaces one
            # acquire() later, on the live feed's thread.
            self.min_integration_ms = float(self._spec.min_integration_ms)
            self.max_integration_ms = float(self._spec.max_integration_ms)
            self._sync_derived()
        except Exception:
            self.disconnect()
            raise

    def _sync_derived(self) -> None:
        """Re-read everything the base class contract exposes as plain
        attributes. Called after any change that can move them, so the two
        couplings in the class docstring stay true by construction."""
        self.integration_ms = float(self._spec.frame_time_ms)
        self.max_counts = float(self._spec.max_counts)

    # ── SpectrometerBase ────────────────────────────────────────────────────
    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def set_integration_time(self, ms: float) -> None:
        self._spec.set_integration_time(float(ms))
        self._sync_derived()

    def acquire(self) -> np.ndarray:
        return self._spec.acquire()

    def disconnect(self) -> None:
        spec = getattr(self, "_spec", None)
        if spec is not None:
            try:
                spec.disconnect()
            except Exception:
                pass
            self._spec = None

    # ── Avantes-specific settings ───────────────────────────────────────────
    # Everything below is reached only from the Avantes settings dialog. It is
    # exposed here rather than by handing the GUI the raw avantes.AvaSpec so
    # the one-way frog_gui_fast -> scan -> hardware dependency holds, and so
    # _sync_derived() cannot be forgotten at a call site.
    @property
    def exposure_ms(self) -> float:
        """The exposure alone, without the averaging factor that
        `integration_ms` folds in."""
        return float(self._spec.integration_ms)

    @property
    def n_averages(self) -> int:
        return int(self._spec.n_averages)

    def set_averages(self, n: int) -> None:
        self._spec.set_averages(int(n))
        self._sync_derived()          # frame time just changed

    @property
    def high_res_adc(self) -> bool:
        return bool(self._spec.high_res_adc)

    def set_high_res_adc(self, enable: bool) -> bool:
        """Returns the mode actually in force — False when the hardware has no
        16-bit ADC, which is a fact about the model rather than an error."""
        on = self._spec.set_high_res_adc(bool(enable))
        self._sync_derived()          # full scale just changed
        return on

    def set_dark_correction(self, enable: bool) -> None:
        self._spec.set_dark_correction(bool(enable))

    def set_smoothing(self, pixels: int) -> None:
        self._spec.set_smoothing(int(pixels))

    @property
    def smoothing_pixels(self) -> int:
        return int(self._spec.config.m_Smoothing.m_SmoothPix)

    @property
    def dark_correction(self) -> bool:
        return bool(self._spec.config.m_CorDynDark.m_Enable)

    def set_prescan(self, enable: bool) -> None:
        self._spec.set_prescan_mode(bool(enable))

    def set_sync_mode(self, enable: bool) -> None:
        self._spec.set_sync_mode(bool(enable))

    def set_trigger(self, mode: int, source: int, source_type: int) -> None:
        self._spec.set_trigger(mode, source, source_type)

    @property
    def trigger(self) -> tuple[int, int, int]:
        t = self._spec.config.m_Trigger
        return int(t.m_Mode), int(t.m_Source), int(t.m_SourceType)

    @property
    def hardware_triggered(self) -> bool:
        """True when acquire() waits on an external pulse and may therefore
        block indefinitely. The GUI checks this before leaving a live feed
        running, since a feed that never returns a frame wedges the device
        handover with no visible cause."""
        return bool(self._spec.hardware_triggered)

    def temperature_c(self) -> float | None:
        """Board temperature, or None on a device with no such sensor."""
        return self._spec.temperature_c()

    def saturated_pixels(self) -> np.ndarray:
        """The device's own clipped-pixel mask for the last frame. Independent
        of the count-vs-max_counts test the scan does, and useful as a
        cross-check rather than a replacement."""
        return self._spec.saturated_pixels()

    def device_info(self) -> dict:
        return self._spec.device_info()
