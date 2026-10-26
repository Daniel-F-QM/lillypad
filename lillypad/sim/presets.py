"""The beam picker and the simulated instruments it is measured on.

PULSE_SHAPES is the registry the Simulation window builds itself from, ordered
simple -> structured. SIM_SPECTROMETERS models two REAL devices closely enough
that the calibration files shipped for them apply unchanged.

The "calibration" values are bare filename stems and are deliberately NOT
resolved here: keeping the path lookup in the caller is what stops the physics
layer from depending on where the program is installed.
"""

from __future__ import annotations

import numpy as np

from lillypad.sim.bird import _BIRD_LAMBDA0_NM, _BIRD_TAU0_FS, _build_bird
from lillypad.sim.pulses import (_build_chirp, _build_double, _build_fiber,
                                 _build_spm, _build_tl_gauss, _build_tl_sech,
                                 _build_tod, _build_two_color,
                                 _misaligned_efficiency)

#                     beam is measured with, for the simulated stitched pair.
PULSE_SHAPES: dict[str, dict] = {
    "tl_sech": {
        "label": "Transform-limited sech²",
        "build": _build_tl_sech,
        "desc":  "Clean soliton pulse, flat phase. The reference trace: a "
                 "symmetric blob with no tilt or structure.",
    },
    "tl_gauss": {
        "label": "Transform-limited Gaussian",
        "build": _build_tl_gauss,
        "sim":   {"tau0_fs": 5.5},     # SHG over 550–900 nm
        "desc":  "Same FWHM as the sech, but with Gaussian wings — a slightly "
                 "tighter trace with no pedestal.",
    },
    "chirp": {
        "label": "Linearly chirped (GDD)",
        "build": _build_chirp,
        "sim":   {"tau0_fs": 5.0},     # SHG over 550–900 nm
        "desc":  "+3·τ₀² of GDD (~6x stretch). Long in delay, unchanged in "
                 "bandwidth — the classic 'this pulse is not compressed' trace.",
    },
    "tod": {
        "label": "Third-order dispersion",
        "build": _build_tod,
        "sim":   {"tau0_fs": 4.9},     # SHG over 550–900 nm
        "desc":  "Pure cubic phase: a steep front edge trailed by an Airy "
                 "satellite train. The satellites sit ~1% below the peak, so "
                 "they need a compressed colour scale to see.",
    },
    "spm": {
        "label": "Self-phase modulation",
        "build": _build_spm,
        "sim":   {"tau0_fs": 19.2},     # SHG over 550–900 nm
        "desc":  "B-integral of 4.5 rad, no dispersion. Envelope unchanged, "
                 "spectrum split into the multi-lobed SPM comb.",
    },
    "fiber": {
        "label": "Fibre output (SPM + normal GVD)",
        "build": _build_fiber,
        "sim":   {"tau0_fs": 23.5},     # SHG over 550–900 nm
        "desc":  "Split-step propagation past optical wave breaking (N²≈25). "
                 "Hard-edged diamond with interference fringes inside — the "
                 "busiest trace in the set.",
    },
    "double": {
        "label": "Double pulse",
        "build": _build_double,
        "sim":   {"tau0_fs": 5.3},     # SHG over 550–900 nm
        "desc":  "Two replicas 6·τ₀ apart in quadrature: three lobes along "
                 "delay, all striped with spectral interference fringes.",
    },
    "two_color": {
        "label": "Two-colour pair",
        "build": _build_two_color,
        "sim":   {"tau0_fs": 8.2},     # SHG over 550–900 nm
        "desc":  "Two sub-pulses at different colours and different times. SHG "
                 "mixes them into three off-diagonal bands.",
    },
    "bird": {
        "label": "Bird (broadband MIR continuum)",
        "build": _build_bird,
        # The beam's own carrier and duration, and a time grid to match: a
        # near-single-cycle pulse on an octave and a half of bandwidth needs
        # the finer spectral sampling (2048 fs of window) to put more than one
        # simulated sample on a NIRQuest pixel.
        "sim": {"lambda0_nm": _BIRD_LAMBDA0_NM, "tau0_fs": _BIRD_TAU0_FS,
                "n_time": 8192, "t_max_fs": 2048.0},
        "spectrometers": ("ocean_sr", "ocean_nirquest"),
        "desc":  "A measured 1.0–3.2 µm OPA continuum compressed to ~8 fs "
                 "(5.1 fs transform limit): a 2.4 µm main band over a rippled "
                 "shoulder 10 dB down, water notches at 1.38 and 1.93 µm, and "
                 "a satellite ahead of the main peak. Its SHG covers 500–"
                 "1400 nm, so Stitched pair gives you the real Ocean SR + "
                 "NIRQuest bench rather than two halves of one band.",
    },
    "misaligned": {
        "label": "Misaligned correlator (asymmetric)",
        "build": _build_tl_sech,
        # Like "saturated" below, this describes the MEASUREMENT and not the
        # field: the same clean sech pulse, seen through a correlator whose
        # beams walk apart as the stage moves.
        "efficiency": _misaligned_efficiency,
        "desc":  "A clean sech pulse measured through a misaligned "
                 "correlator: the beams walk apart as the delay stage travels, "
                 "so one wing of the trace is vignetted away while the other "
                 "is untouched. The pulse is fine — the trace is lopsided "
                 "because the setup is. Exercises symmetry mode.",
    },
    "saturated": {
        "label": "Over-exposed sech² (clips)",
        "build": _build_tl_sech,
        # `exposure` is a property of the MEASUREMENT, not of the field: it
        # scales peak_counts, so the same clean sech pulse is simply driven
        # 25x too hard and clips against max_counts. Here to exercise the
        # saturation warning — flat-topped spectrum, ruined trace.
        "exposure": 25.0,
        "desc":  "A clean sech pulse deliberately over-exposed 25x. The core "
                 "clips flat against full scale and the saturation lamp goes "
                 "red — turn the integration time down to recover it.",
    },
    "low_snr": {
        "label": "Weak signal (low SNR)",
        "build": _build_tl_sech,
        # The other end of "saturated": the same clean sech pulse with 1/20 of
        # the light. The read noise and the pedestal's shot noise do not scale
        # with it, so the trace sinks into them. Measured on the app's single
        # simulator: peak SNR ~2.5 per frame at 10 ms (the clean pulse: ~26),
        # ~13 at 100 ms.
        "exposure": 0.05,
        "desc":  "A clean sech pulse with 1/20 of the light. At 10 ms the "
                 "peak sits only 2–3x above the noise and the wings vanish "
                 "into it. For testing averaging, the dark and the fits on "
                 "noisy data — raise the integration time or Averages / point "
                 "to dig it out.",
    },
}

DEFAULT_PULSE = "tl_sech"

# SimulatedSpectrometer arguments a BEAM may own, and what they are when
# neither the caller nor the preset says. A 1400 nm, 5.2 fs pulse on a +/-1024
# fs grid: its SHG fills 550–900 nm. The structured beams spread their spectrum
# further than a transform-limited pulse does, so each carries its own tau0_fs
# (see "sim" above), fitted so that its SHG window is that same 550–900 nm.
SIM_BEAM_DEFAULTS: dict[str, float | int] = {
    "lambda0_nm": 1400.0, "tau0_fs": 5.2, "n_time": 4096, "t_max_fs": 1024.0,
}


# ---------------------------------------------------------------------------
# Simulated spectrometers modelled on real devices
# ---------------------------------------------------------------------------
# Enough of a real spectrometer to make a simulated stitched pair behave like
# the bench: where its pixels are, how many, what the detector's response does
# across them, and how much dark and read noise it carries. The two here are
# the pair this app was written for, and their `response` curves are the
# reciprocal of the calibration files shipped in calibration_files/ — so
# loading SR_New / Niquest_New onto the simulated devices flattens them the
# same way it flattens the real ones.
#
# `response` shapes the counts and is applied AFTER normalisation (see
# SimulatedSpectrometer.acquire), so it really does cost signal: a beam sitting
# where a detector has gone deaf reads at the noise floor, which is what makes
# the dead edges of an overlap dead.

def _response_curve(wl_nm, relative):
    """A detector response as a lookup on a coarse table, held flat outside it
    (never extrapolated to zero or to nonsense at the band edges)."""
    wl, rel = np.asarray(wl_nm, float), np.asarray(relative, float)
    return lambda w: np.interp(np.asarray(w, float), wl, rel)


SIM_SPECTROMETERS: dict[str, dict] = {
    "ocean_sr": {
        "name": "simulated Ocean SR (Si, 350–1050 nm)",
        # Stem of the calibration file in calibration_files/ that inverts this
        # device's response, loaded by the GUI when the file is there.
        "calibration": "SR_New",
        "device": dict(
            wl_start=350.0, wl_end=1050.0, n_pixels=2048,
            # Silicon: broad and flat through the visible, then off a cliff
            # past 950 nm. Nothing but noise by 1050, which is exactly where
            # this pair's overlap sits — hence the 25 nm sampling through the
            # roll-off, where a coarser table would stop being the reciprocal
            # of the calibration file.
            response=_response_curve(
                (350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875,
                 900, 925, 950, 975, 1000, 1025, 1050),
                (0.253, 0.315, 0.465, 0.682, 0.837, 0.949, 0.982, 0.994,
                 0.926, 0.801, 0.750, 0.688, 0.727, 0.660, 0.556, 0.424,
                 0.296, 0.174, 0.059)),
            peak_counts=25000.0, read_noise=8.0, background_counts=250.0),
    },
    "ocean_nirquest": {
        "name": "simulated Ocean NIRQuest512 (InGaAs, 900–1700 nm)",
        "calibration": "Niquest_New",
        "device": dict(
            wl_start=900.0, wl_end=1700.0, n_pixels=512,
            # InGaAs: cutting on hard at the blue end of its range and flat
            # from ~1.3 um. Four times the pixel pitch of the Si unit and a
            # much larger dark — the reason the two want separate exposures.
            response=_response_curve(
                (900, 925, 950, 975, 1000, 1025, 1050, 1100, 1150, 1200, 1250,
                 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700),
                (0.225, 0.351, 0.411, 0.468, 0.499, 0.516, 0.530, 0.618,
                 0.687, 0.716, 0.766, 0.852, 0.912, 0.898, 0.930, 0.979,
                 0.992, 0.968, 0.968, 0.968)),
            peak_counts=25000.0, read_noise=30.0, background_counts=1400.0),
    },
}


def sim_spectrometer_pair(pulse: str, gate: str = "shg"
                          ) -> tuple[str, str] | None:
    """The (blue, red) SIM_SPECTROMETERS keys a beam is measured with, or None
    when the simulated halves should just split the beam's own band.

    A named pair is a pair of real detectors, so it only applies where the
    signal really lands on them: on the SECOND HARMONIC. Under a PG gate the
    signal sits at the fundamental — for "bird" that is 1.0-3.2 um, past the
    red end of any silicon or InGaAs spectrometer — so the generic halves take
    over rather than handing the operator two dead devices.
    """
    if gate != "shg":
        return None
    models = PULSE_SHAPES.get(pulse, {}).get("spectrometers")
    return tuple(models) if models else None
