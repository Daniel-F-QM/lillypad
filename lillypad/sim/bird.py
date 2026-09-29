"""The "bird" beam: a measured broadband MIR continuum.

A real spectrum, digitised from a measurement, rather than an analytic envelope —
~8 fs over 1.0-3.2 um, an octave and a half, whose second harmonic lands across
500-1400 nm and therefore needs a silicon AND an InGaAs spectrometer to be seen
at all. It is the reason the simulated stitched pair exists.
"""

from __future__ import annotations

import numpy as np

from lillypad.hardware.base import C_NM_PER_FS

# builds it at another duration.
_BIRD_LAMBDA0_NM = 2400.0       # carrier: the peak of the measured spectrum
_BIRD_TAU0_FS    = 4.5          # sech tau0 of the ~8 fs retrieved pulse

# Smooth envelope of the measured spectrum as (wavelength um, relative dB)
# control points — a measurement is a table, not a formula. The ripples, the
# water notches and the spectral phase are added analytically on top.
_BIRD_SPECTRUM_DB = np.array([
    (0.90, -48.0), (0.95, -40.0), (1.00, -32.0), (1.05, -24.0), (1.10, -17.5),
    (1.15, -14.5), (1.20, -13.5), (1.30, -11.5), (1.40, -10.4), (1.50, -10.0),
    (1.60,  -9.6), (1.70,  -9.3), (1.80,  -9.0), (1.90,  -9.4), (2.00, -11.0),
    (2.10,  -7.5), (2.20,  -3.5), (2.30,  -1.2), (2.42,   0.0), (2.50,  -0.5),
    (2.60,  -2.0), (2.70,  -5.0), (2.80,  -9.5), (2.90, -15.0), (3.00, -21.0),
    (3.10, -27.0), (3.20, -33.0), (3.35, -45.0), (3.50, -60.0),
])


def _bird_envelope(um: np.ndarray) -> np.ndarray:
    """Relative spectral INTENSITY at wavelengths `um` (micrometres).

    The control points are interpolated in dB (the axis they were read off)
    and then smoothed, because straight-line segments in dB leave visible
    kinks exactly where the log plot is most read.
    """
    grid = np.arange(0.85, 3.60, 0.005)
    db   = np.interp(grid, _BIRD_SPECTRUM_DB[:, 0], _BIRD_SPECTRUM_DB[:, 1])
    kern = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    db   = np.convolve(np.pad(db, 12, mode="edge"), kern / kern.sum(),
                       mode="valid")
    # Hard zero outside the table: extrapolating a dB slope would put spurious
    # signal on the detectors at wavelengths the beam never reaches.
    return 10.0 ** (np.interp(um, grid, db, left=-90.0, right=-90.0) / 10.0)


def _build_bird(t, tau0):
    """A ~1.0-3.2 um continuum compressed to ~8 fs — the busiest beam here.

    Structure at every scale, which is the point of it: an octave and a half of
    bandwidth, a shoulder ~10 dB below the main 2.4 um band carrying a
    modulation ripple, water-vapour notches at 1.38 and 1.93 um, and enough
    residual GDD/TOD to put a satellite ahead of the main peak.

    SHG of a spectrum this wide lands anywhere from ~500 to ~1400 nm — more
    than any one detector covers, which is why this beam ships with a
    spectrometer PAIR (see SIM_SPECTROMETERS). The cross-terms between the
    2.4 um band and the shoulder are the bright part at 800-1100 nm; they exist
    only where the two spectral regions also overlap in time, so the blue
    spectrometer is genuinely measuring something the red one cannot see.
    """
    dt    = t[1] - t[0]
    Om    = 2 * np.pi * np.fft.fftfreq(t.size, d=dt)
    omega = 2 * np.pi * C_NM_PER_FS / _BIRD_LAMBDA0_NM + Om
    ok    = omega > 0
    # Wavelength of every grid point, in um. Non-positive frequencies get a
    # placeholder far outside the table (and are masked out below) rather than
    # inf, which would poison the sin() ripple with NaNs.
    um = np.where(ok, 2 * np.pi * C_NM_PER_FS / np.where(ok, omega, 1.0) * 1e-3,
                  1e3)

    S = _bird_envelope(um)
    # Modulation ripple across the shoulder — the fingerprint of a continuum
    # generated from two overlapping processes. Its 0.185 um period is what
    # puts the echo at ~60 fs in the trace, so it sets the scan range.
    win = np.exp(-0.5 * ((um - 1.75) / 0.60) ** 4)
    S   = S * (1.0 + 0.20 * np.sin(2 * np.pi * (um - 1.02) / 0.185) * win)
    # Water vapour in the beam path (1.38 and 1.93 um) and a CO2/OH edge near
    # 2.72 um: (centre um, depth, width um).
    for centre, depth, width in ((1.93, 0.55, 0.030),
                                 (1.38, 0.30, 0.022),
                                 (2.72, 0.30, 0.045)):
        S = S * (1.0 - depth * np.exp(-0.5 * ((um - centre) / width) ** 2))
    amp = np.sqrt(np.clip(S, 0.0, None) * ok)

    # Uncompensated residual: a little GDD, a little TOD, plus a shallow
    # sinusoidal ripple of the kind a chirped mirror pair leaves behind. Takes
    # the ~5.1 fs transform limit to ~8.1 fs and hangs a satellite off the
    # leading edge.
    phi = (0.5 * (0.45 * tau0 ** 2) * Om ** 2
           + (0.25 * tau0 ** 3) * Om ** 3 / 6.0
           + 0.20 * np.sin(3.4 * tau0 * Om + 0.6))
    # Built in the frequency domain, so the pulse lands at index 0 — fftshift
    # moves it to the middle of the time grid where every other builder puts it.
    A = np.fft.fftshift(np.fft.ifft(amp * np.exp(1j * phi)))
    return A / np.abs(A).max()
