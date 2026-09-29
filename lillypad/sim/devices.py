"""The two simulated devices.

Drop-in SpectrometerBase / StageBase implementations that need no SDK and no
hardware. They are simultaneously the production simulators the Simulation
window offers and the test doubles the suite runs against, which is why they sit
here rather than under hardware/.

SimulatedStage models lead-screw slack, so the backlash zero-shift that used to
be findable only on the bench reproduces offline.
"""

from __future__ import annotations

import time

import numpy as np

from lillypad.hardware.base import C_NM_PER_FS, SpectrometerBase, StageBase
from lillypad.sim.presets import DEFAULT_PULSE, PULSE_SHAPES, SIM_BEAM_DEFAULTS
from lillypad.sim.pulses import _energy_window, _fourier_shift

# ===========================================================================
class SimulatedStage(StageBase):
    """In-memory delay stage in mm, with optional settle time.

    physical_backlash_mm models real lead-screw slack: the carriage only
    follows the nut after the slack has been taken up, so a target reached
    travelling downward lands physical_backlash_mm short of the same target
    reached travelling upward. Zero (the default) is an ideal stage.

    This is what makes the zero-shift bug reproducible without hardware: mark
    zero after arriving from one side, sweep a scan from the other, and the
    trace comes back offset. Setting backlash_mm (the *correction*) to anything
    larger than physical_backlash_mm makes the offset go away.
    """
    def __init__(self, settle_s: float = 0.0, travel_mm: float = 300.0,
                 physical_backlash_mm: float = 0.0, backlash_mm: float = 0.0):
        self._pos      = 0.0     # where the nut is  (= commanded)
        self._true_pos = 0.0     # where the carriage actually is
        self.settle_s  = settle_s
        self.travel_mm = travel_mm
        self.backlash_mm = float(backlash_mm)
        self.physical_backlash_mm = float(physical_backlash_mm)
        self.name      = "simulated stage"

    def _move_to_raw(self, position_mm: float) -> None:
        target = float(position_mm)
        if target != self._pos:      # a no-op command moves nothing, slack included
            # Arriving upward, the carriage is pushed flush with the nut;
            # arriving downward it trails by the slack.
            self._true_pos = (target if target > self._pos
                              else target - self.physical_backlash_mm)
            self._pos = target
        if self.settle_s:
            time.sleep(self.settle_s)

    def get_position(self) -> float:
        return self._true_pos

    def home(self) -> None:
        self.move_to(0.0)


# ---------------------------------------------------------------------------
# Pulse-shape library ("beams") for the simulated spectrometer
# ---------------------------------------------------------------------------
# Each builder takes the time grid (fs) and the transform-limited sech
# parameter tau0 (fs) and returns the COMPLEX baseband envelope A(t). The
# carrier and the nonlinear gate are applied by SimulatedSpectrometer, so a
# builder only ever describes the pulse itself.
#
# Every knob below is expressed as a multiple of tau0, so a preset keeps its
# character (stretch factor, number of lobes, ...) at any pulse duration.
#
# Fourier convention: A(t) = sum_Omega Atilde(Omega)·exp(+i·Omega·t), matching
# numpy's ifft. Spectral phase is therefore applied as exp(+i·phi(Omega)).
# An SHG trace is symmetric in delay, so it cannot distinguish +phi2 from
# -phi2 — the sign convention only matters for the PG gate.

class SimulatedSpectrometer(SpectrometerBase):
    """Synthetic FROG spectrometer for offline testing.

    Each acquire() reads the stage's *actual* mm position, converts it to a
    delay via `position_to_delay` (injected by the app so the simulator sees
    the same zero-delay as the real optics), and returns the corresponding
    FROG spectrum. Defaults to a transform-limited sech pulse, SHG gate.

    `pulse` selects the beam under test: a key of PULSE_SHAPES, or any
    callable(t_fs, tau0_fs) -> complex envelope for a one-off field. Because
    the envelope is complex, the spectral phase is what shapes the trace —
    that is the whole point of the chirp / TOD / SPM presets.

    `response` is the DEVICE's relative sensitivity, callable(wl_nm) -> factor.
    It shapes the counts after the beam has been normalised, so unlike
    everything else here it describes the detector rather than the light — see
    SIM_SPECTROMETERS, and acquire() for where it lands.

    Gate-agnostic by design:
        shg : E_sig = E(t)·E(t-tau)        -> signal at 2*omega0
        pg  : E_sig = E(t)·|E(t-tau)|^2    -> signal at  omega0
    """
    def __init__(self, stage: StageBase, position_to_delay=None,
                 gate: str = "shg", lambda0_nm: float | None = None,
                 tau0_fs: float | None = None, pulse=DEFAULT_PULSE,
                 wl_start: float | None = None, wl_end: float | None = None,
                 n_pixels: int = 1024, n_time: int | None = None,
                 t_max_fs: float | None = None,
                 peak_counts: float = 4000.0, read_noise: float = 15.0,
                 background_counts: float = 0.0,
                 max_counts: float | None = 65535.0,
                 response=None):
        self.stage             = stage
        self.gate              = gate
        self.read_noise        = read_noise
        # 16-bit full scale, like the Ocean Optics units. Pass None for the old
        # unbounded behaviour (nothing ever clips, nothing ever warns).
        self.max_counts        = max_counts
        # Delay-independent pedestal hook. The *realistic* (delay-dependent,
        # single-arm) background term depends on the background strategy we
        # settle on, so this stays a simple constant for now.
        self.background_counts = background_counts
        self.integration_ms    = 100.0

        if callable(pulse):
            self.pulse, build   = "custom", pulse
            self.pulse_label    = getattr(pulse, "__name__", "custom pulse")
            self.pulse_desc     = ""
            self.exposure       = 1.0
            self._efficiency    = None
            beam: dict          = {}
        else:
            if pulse not in PULSE_SHAPES:
                raise KeyError(f"Unknown pulse {pulse!r}. "
                               f"Known: {sorted(PULSE_SHAPES)}")
            shape = PULSE_SHAPES[pulse]
            self.pulse, build   = pulse, shape["build"]
            self.pulse_label    = shape["label"]
            self.pulse_desc     = shape["desc"]
            self.exposure       = float(shape.get("exposure", 1.0))
            # Optional delay-dependent efficiency (see _misaligned_efficiency).
            # Read before _norm is computed below, so the normalisation is of
            # the trace as actually measured rather than of an ideal one.
            self._efficiency    = shape.get("efficiency")
            # Carrier, duration and time grid the BEAM asks for ("bird" is a
            # specific 2.4 um pulse, not a shape that scales).
            beam                = dict(shape.get("sim", {}))

        def _beam_arg(name, given):
            """Explicit argument, else what the preset asks for, else the
            module default. Three sources, one order, no silent surprises."""
            if given is not None:
                return given
            return beam.get(name, SIM_BEAM_DEFAULTS[name])

        lambda0_nm = float(_beam_arg("lambda0_nm", lambda0_nm))
        tau0_fs    = float(_beam_arg("tau0_fs",    tau0_fs))
        n_time     = int(_beam_arg("n_time",       n_time))
        t_max_fs   = float(_beam_arg("t_max_fs",   t_max_fs))
        # Kept because the measurement-side hooks below are scaled in units of
        # the pulse's own duration.
        self.tau0 = self.tau0_fs = tau0_fs
        # Presets may ask to be driven harder than full scale (see "saturated").
        self.peak_counts = peak_counts * self.exposure
        self.name = f"simulated {gate}-FROG — {self.pulse_label}"

        if position_to_delay is None:
            # Default double-pass conversion, zero at 0 mm. The app overrides
            # this with the user's marked zero-delay offset.
            position_to_delay = lambda x_mm: 2.0 * x_mm * 1e6 / C_NM_PER_FS
        self._pos_to_delay = position_to_delay

        omega0       = 2 * np.pi * C_NM_PER_FS / lambda0_nm
        self.carrier = 2 * omega0 if gate == "shg" else omega0

        # Time grid + the beam under test, as a complex envelope A(t).
        self.t  = np.linspace(-t_max_fs, t_max_fs, n_time)
        self.dt = self.t[1] - self.t[0]
        self.A  = np.asarray(build(self.t, tau0_fs), dtype=complex)

        self._Omega = 2 * np.pi * np.fft.fftfreq(n_time, d=self.dt)
        self._Afft  = np.fft.fft(self.A)
        omega = self.carrier + self._Omega
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(omega > 0, 2 * np.pi * C_NM_PER_FS / omega, np.inf)
        self._order      = np.argsort(lam)
        self._lam_sorted = lam[self._order]

        # How far the pulse actually reaches in time (99% of its energy).
        lo, hi = _energy_window(self.t, np.abs(self.A) ** 2, frac=0.99)
        self.pulse_span_fs = max(abs(lo), abs(hi), 2.0 * tau0_fs)
        # A delay scan has to cover the correlation, which is wider than the
        # pulse itself. Offered to the UI as a hint; nothing here enforces it.
        self.suggested_delay_fs = float(np.ceil(1.5 * self.pulse_span_fs / 25) * 25)

        # Structured beams are far broader than a TL pulse (and PG sits at the
        # fundamental, not the second harmonic), so the window is sized from
        # the signal itself unless the caller pinned it.
        auto_lo, auto_hi = self._spectral_extent()
        self._wl = np.linspace(auto_lo if wl_start is None else wl_start,
                               auto_hi if wl_end is None else wl_end, n_pixels)
        # Per-pixel detector sensitivity, 1 everywhere on an ideal device.
        self._pixel_response = (
            None if response is None
            else np.clip(np.asarray(response(self._wl), float), 0.0, None))

        # peak_counts is the peak of the whole TRACE — for a double pulse the
        # brightest column is not the one at zero delay.
        #
        # Measured over the WHOLE signal band, not over this device's window:
        # otherwise a device covering a dim corner of the spectrum would
        # renormalise that corner up to full brightness, and two members of a
        # stitched pair would arrive with an unknowable factor between them
        # instead of the honest ratio of what each one can actually see.
        norm = max(self._raw_column(tau, np.linspace(auto_lo, auto_hi, n_pixels)
                                    ).max() for tau in self._delay_samples())
        self._norm = norm if norm > 0 else 1.0

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def set_integration_time(self, ms: float) -> None:
        self.integration_ms = float(ms)

    def _delay_samples(self, n: int = 21) -> np.ndarray:
        """Delays spanning the pulse — enough to characterise the whole trace."""
        return np.linspace(-self.pulse_span_fs, self.pulse_span_fs, n)

    def _signal_spectrum(self, tau: float) -> np.ndarray:
        """|E_sig(omega, tau)|^2 on the FFT's own (wavelength-sorted) grid."""
        A_shift = _fourier_shift(self.A, self.t, tau)
        sig = (self.A * A_shift if self.gate == "shg"
               else self.A * np.abs(A_shift) ** 2)
        return (np.abs(np.fft.fft(sig)) ** 2)[self._order]

    def _spectral_extent(self, frac: float = 0.995,
                         pad: float = 0.35) -> tuple[float, float]:
        """Wavelength window holding the delay-integrated signal, plus margin."""
        marg = sum(self._signal_spectrum(tau) for tau in self._delay_samples())
        finite = np.isfinite(self._lam_sorted)
        lo, hi = _energy_window(self._lam_sorted[finite], marg[finite], frac)
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        half = max(half * (1.0 + pad), 2.0)                 # never absurdly tight
        # Kept inside a couple of octaves of the carrier. Wavelength is 1/omega,
        # so a beam broad enough to carry weight near zero frequency — an
        # octave-spanning pulse under the PG gate, where the gate |E|^2 pushes
        # the signal down towards DC — otherwise produces a window stretching
        # to tens of microns and out the far side into NEGATIVE wavelengths.
        # Every ordinary preset sits well inside these bounds and is untouched.
        lam_c = 2 * np.pi * C_NM_PER_FS / self.carrier
        return (max(mid - half, 0.3 * lam_c), min(mid + half, 3.0 * lam_c))

    def _raw_column(self, tau: float, wl: np.ndarray | None = None
                    ) -> np.ndarray:
        """The signal on `wl` (this device's pixels by default), before the
        detector's own response and before any scaling to counts."""
        col = np.interp(self._wl if wl is None else wl,
                        self._lam_sorted, self._signal_spectrum(tau),
                        left=0.0, right=0.0)
        if self._efficiency is not None:
            # Scales the whole column, never its shape: a geometric overlap
            # loss costs signal at every wavelength equally.
            col = col * float(self._efficiency(tau, self.tau0_fs))
        return col

    def acquire(self) -> np.ndarray:
        # `stage` may be None — the app lets a simulated spectrometer run with
        # no stage connected at all. Zero delay is the honest answer then: it
        # is the one delay that needs no axis to define it, and it gives the
        # brightest column, so an unattended simulator still shows a spectrum.
        tau    = (self._pos_to_delay(self.stage.get_position())   # read-back!
                  if self.stage is not None else 0.0)
        scale  = self.integration_ms / 100.0                     # signal ∝ integ. time
        counts = self._raw_column(tau) / self._norm * self.peak_counts * scale
        if self._pixel_response is not None:
            # The detector's own sensitivity, applied to the light and NOT to
            # the dark below it — a pixel where the detector has gone deaf
            # still carries its full pedestal and read noise, which is exactly
            # what makes the dead edge of an overlap band worthless to a fit.
            counts = counts * self._pixel_response
        counts = counts + self.background_counts * scale
        counts = np.random.poisson(np.clip(counts, 0, None)).astype(float)
        counts += np.random.normal(0.0, self.read_noise, size=counts.shape)  # read noise: fixed
        # The ADC ceiling is applied LAST, after noise, so a clipped pixel sits
        # at exactly full scale with no noise on top — which is what makes
        # saturation detectable (and what makes the data unrecoverable).
        return np.clip(counts, 0, self.max_counts)
