"""The simulator: pulse physics, beam presets, and two drop-in devices.

    pulses    envelopes, spectral phase, split-step propagation, the builders
    bird      the measured broadband MIR continuum
    presets   PULSE_SHAPES and the simulated instruments
    devices   SimulatedStage and SimulatedSpectrometer

Top-level rather than under hardware/ on purpose. The ~700 lines of physics here
touch no device I/O and are the simulator's real substance; the two device
classes are simultaneously the production simulators and the suite's test
doubles. The only edge into the device layer is hardware.base, for the abstract
interfaces and the speed of light.

Nothing here reads a file or knows where the program is installed — the
calibration stems in presets are resolved by the caller.
"""

from lillypad.sim.devices import SimulatedSpectrometer, SimulatedStage
from lillypad.sim.presets import (DEFAULT_PULSE, PULSE_SHAPES,
                                  SIM_BEAM_DEFAULTS, SIM_SPECTROMETERS,
                                  sim_spectrometer_pair)

__all__ = [
    "SimulatedStage", "SimulatedSpectrometer",
    "PULSE_SHAPES", "DEFAULT_PULSE", "SIM_BEAM_DEFAULTS",
    "SIM_SPECTROMETERS", "sim_spectrometer_pair",
]
