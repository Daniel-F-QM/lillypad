"""TEMPORARY compatibility shim — hardware.py moved into lillypad/.

The device layer now lives in lillypad/hardware/ and the simulator in
lillypad/sim/. This file exists only so frog_gui_fast.py, scan.py and
zaber_diagnostics.py keep importing as they did while the refactor continues; it
is deleted in the step that repoints them.

Every name below is RE-EXPORTED, never redefined. That is not a style
preference: a dozen `isinstance` checks in the GUI test against these classes,
and a shim that declared its own would make every one of them quietly false — a
spectrum that looks right and a "no device" overlay that never lifts. Explicit
`from X import Y` also fails loudly at startup if a name went missing, instead of
surfacing as an AttributeError deep inside a dialog.

New code should import from lillypad.hardware or lillypad.sim directly.
"""

from lillypad.hardware import (SEABREEZE_BACKENDS, SPEC_VENDORS,
                               AvantesSpectrometer, C_NM_PER_FS, KinesisStage,
                               OVERLAP_BAND_FRAC, PiezoJenaStage,
                               SeabreezeSpectrometer, SpectrometerBase,
                               StageBase, StitchedSpectrometer, ZaberStage,
                               avantes_trigger_options, list_avantes_spectrometers,
                               list_kinesis_stages, list_seabreeze_spectrometers,
                               list_spectrometers, load_calibration_file,
                               open_spectrometer, select_seabreeze_backend,
                               spec_ident, split_spec_ident)
# Private helpers the test suite reaches for by name. They are private to the
# device layer, not to the process, and the alternative is duplicating the
# fixtures they support.
from lillypad.hardware.spectrometers.avantes import _avantes_module
from lillypad.sim import (DEFAULT_PULSE, PULSE_SHAPES, SIM_BEAM_DEFAULTS,
                          SIM_SPECTROMETERS, SimulatedSpectrometer,
                          SimulatedStage, sim_spectrometer_pair)
from lillypad.sim.bird import _bird_envelope
from lillypad.sim.pulses import (_energy_window, _fourier_shift, _gauss,
                                 _misaligned_efficiency, _sech, _spectral_phase,
                                 _split_step)
