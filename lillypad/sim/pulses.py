"""Pulse physics: envelopes, spectral phase, propagation, and the beam builders.

Pure functions over numpy arrays. Nothing here knows about a device, a Qt object
or a file — which is what makes it the part of the simulator worth testing
directly, and what lets the beam library grow without touching device code.

Fourier convention: A(t) = sum_Omega Atilde(Omega)*exp(+i*Omega*t), matching
numpy's ifft. Spectral phase is applied as exp(+i*phi(Omega)). An SHG trace is
symmetric in delay, so it cannot distinguish +phi2 from -phi2 — the sign
convention only matters for the PG gate.
"""

from __future__ import annotations

import numpy as np

def _sech(t: np.ndarray, tau0: float) -> np.ndarray:
    return (1.0 / np.cosh(t / tau0)).astype(complex)


def _gauss(t: np.ndarray, tau0: float) -> np.ndarray:
    """Gaussian with the same intensity FWHM as sech(t/tau0) — 1.7627·tau0."""
    fwhm = 1.7627 * tau0
    return np.exp(-2.0 * np.log(2.0) * (t / fwhm) ** 2).astype(complex)


def _spectral_phase(A: np.ndarray, dt: float,
                    phi2: float = 0.0, phi3: float = 0.0) -> np.ndarray:
    """Apply GDD (fs^2) and TOD (fs^3) to a field."""
    Om = 2 * np.pi * np.fft.fftfreq(A.size, d=dt)
    phase = 0.5 * phi2 * Om ** 2 + phi3 * Om ** 3 / 6.0
    return np.fft.ifft(np.fft.fft(A) * np.exp(1j * phase))


def _fourier_shift(A: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """A(t - tau), by Fourier shift, with the wrapped tail zeroed.

    The shift theorem is exact for arbitrarily structured complex fields (an
    interpolated shift would smear a strongly chirped one), but it is cyclic:
    without the mask a pulse shifted past the edge of the grid would reappear
    on the other side and fake a signal at large delay.
    """
    Om = 2 * np.pi * np.fft.fftfreq(A.size, d=t[1] - t[0])
    out = np.fft.ifft(np.fft.fft(A) * np.exp(-1j * Om * tau))
    if tau > 0:
        out[t < t[0] + tau] = 0.0
    elif tau < 0:
        out[t > t[-1] + tau] = 0.0
    return out


def _energy_window(x: np.ndarray, weight: np.ndarray,
                   frac: float = 0.995) -> tuple[float, float]:
    """The x-range holding `frac` of the weight, split evenly between tails.

    Used instead of a fixed "above 1e-3 of peak" cut because the presets differ
    by orders of magnitude in how fast their wings fall off: an absolute floor
    reads a sech's exponential tail as a real feature and blows the window up.
    """
    cum = np.cumsum(weight)
    if cum[-1] <= 0:
        return float(x[0]), float(x[-1])
    cum = cum / cum[-1]
    tail = 0.5 * (1.0 - frac)
    return float(np.interp(tail, cum, x)), float(np.interp(1.0 - tail, cum, x))


def _split_step(A: np.ndarray, t: np.ndarray, phi2: float, b_int: float,
                n_steps: int = 200) -> np.ndarray:
    """Propagate A through a nonlinear, dispersive medium (symmetric split-step).

    Parametrised by what actually shapes the output rather than by fibre
    constants: `phi2` is the GDD accumulated over the whole length (fs^2) and
    `b_int` the peak nonlinear phase, the B-integral (rad).
    """
    Om   = 2 * np.pi * np.fft.fftfreq(A.size, d=t[1] - t[0])
    half = np.exp(1j * 0.25 * (phi2 / n_steps) * Om ** 2)   # half a linear step
    peak = np.abs(A).max() ** 2 or 1.0
    for _ in range(n_steps):
        A = np.fft.ifft(np.fft.fft(A) * half)
        A = A * np.exp(1j * (b_int / n_steps) * np.abs(A) ** 2 / peak)
        A = np.fft.ifft(np.fft.fft(A) * half)
    return A


def _build_tl_sech(t, tau0):
    return _sech(t, tau0)


def _build_tl_gauss(t, tau0):
    return _gauss(t, tau0)


def _build_chirp(t, tau0):
    # +3·tau0^2 of GDD stretches the pulse ~6x: the trace grows along delay
    # while each spectral slice stays narrow.
    return _spectral_phase(_sech(t, tau0), t[1] - t[0], phi2=3.0 * tau0 ** 2)


def _build_tod(t, tau0):
    # Pure cubic phase — a sharp leading edge trailed by a decaying oscillatory
    # ripple, which prints as a fan of satellites in the trace.
    #
    # Pure cubic phase gives an Airy-like profile whose satellite-to-peak ratio
    # is universal — raising phi3 only stretches the train in time, it never
    # makes the satellites brighter. 3·tau0^3 therefore keeps them compact
    # enough to sit inside a sane delay scan; they are ~1% of the peak, so the
    # trace needs a compressed colour scale to show them off.
    #
    # Cubic phase is odd in Omega, so it preserves the field's Hermitian
    # symmetry: A(t) comes out real. The pulse is still badly distorted — the
    # "chirp" lives in the sign flips between satellites, not in a frequency
    # sweep — which is exactly why the trace is the interesting part.
    return _spectral_phase(_sech(t, tau0), t[1] - t[0], phi3=3.0 * tau0 ** 3)


def _build_spm(t, tau0):
    # Instantaneous Kerr phase, no dispersion: the envelope is untouched but
    # the spectrum splits into the textbook multi-lobed SPM comb.
    A = _sech(t, tau0)
    return A * np.exp(1j * 4.5 * np.abs(A) ** 2)


def _build_fiber(t, tau0):
    # SPM plus normal GVD -> optical wave breaking: a near-rectangular pulse
    # with ripples on both edges and a hard-edged, fringed trace.
    #
    # What matters is the soliton number, N^2 = LD/LNL = tau0^2·B/phi2 ~ 25
    # here. At the earlier N^2 ~ 2 the pulse merely chirped and the trace was
    # indistinguishable from the plain GDD preset.
    return _split_step(_sech(t, tau0), t, phi2=0.4 * tau0 ** 2, b_int=10.0)


def _build_double(t, tau0):
    # Two near-equal replicas 6·tau0 apart with a quadrature phase offset:
    # three lobes along delay, all carrying spectral interference fringes.
    A   = _sech(t, tau0)
    sep = 6.0 * tau0
    return (_fourier_shift(A, t, -0.5 * sep)
            + 0.9 * np.exp(0.5j * np.pi) * _fourier_shift(A, t, +0.5 * sep))


def _build_two_color(t, tau0):
    # Two sub-pulses at different centre frequencies, separated in time. SHG
    # mixes them into three bands (2w1, w1+w2, 2w2) at three delays — the
    # off-diagonal peaks that only a 2-D measurement can show.
    A   = _sech(t, 1.3 * tau0)
    sep = 3.0 * tau0
    dw  = 1.2 / tau0                      # rad/fs, about one TL bandwidth
    return (_fourier_shift(A, t, -0.5 * sep) * np.exp(+1j * dw * t)
            + _fourier_shift(A, t, +0.5 * sep) * np.exp(-1j * dw * t))


# ── "Bird": a measured broadband MIR continuum ──────────────────────────────
# The presets above are analytic shapes that keep their character at any pulse
# duration. This one does not: it is a specific beam — a near-single-cycle
# 2.4 um continuum out of a MIR OPA — so its landmarks (the water-vapour
# notches, the band edges) sit at absolute wavelengths and it declares its own
# carrier and duration in PULSE_SHAPES["bird"]["sim"]. Only the spectral PHASE
# is written in units of tau0, so the beam still behaves sensibly if someone

def _misaligned_efficiency(tau, tau0):
    """Delay-dependent conversion efficiency of a MISALIGNED correlator.

    Nothing is wrong with the pulse here — the field stays a clean sech. What
    is wrong is the geometry, so this is a property of the MEASUREMENT, like
    "saturated"'s exposure rather than like a build function. With the two arms
    not properly overlapped, moving the delay stage walks one beam across the
    other at the crystal, so the signal is progressively vignetted as the stage
    travels one way and the trace loses that wing while the other stays put.
    That one-sided fall-off is what a FROG in need of realignment looks like,
    and it is what symmetry mode exists to catch.

    A sigmoid rather than a hard edge, because a beam clips off gradually as
    the overlap shrinks; placed and scaled in units of the pulse's own duration
    so the same misalignment reads the same on any tau0.

    The edge sits just inside the correlation rather than out in the wings.
    Further out it only nibbles at tails that are already dark and the trace
    still looks symmetric (7% by the score below); here it takes about a fifth
    of the signal off one side — bad enough to see at a glance, mild enough to
    still be a usable trace, which is the point of a preset for exercising
    symmetry mode.
    """
    return 1.0 / (1.0 + np.exp((tau - 0.4 * tau0) / (0.5 * tau0)))


# Ordered simple -> structured; the GUI builds its picker straight from this.
#
# Optional keys, all of them describing something OTHER than the field itself:
#   exposure        — drive the detector this many times harder ("saturated")
#   efficiency      — callable(tau, tau0), a delay-dependent conversion
#                     efficiency of the correlator ("misaligned")
#   sim             — SimulatedSpectrometer constructor defaults the BEAM owns
#                     (carrier, duration, time grid). An explicit argument
#                     still wins; everything else falls back to the module
#                     defaults in SIM_BEAM_DEFAULTS.
#   spectrometers   — keys into SIM_SPECTROMETERS: the pair of devices this
