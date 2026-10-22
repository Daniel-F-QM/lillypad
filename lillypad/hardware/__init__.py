"""Device layer: abstract interfaces, vendor drivers, and the registry.

    base            StageBase / SpectrometerBase, the tagged-id format, the
                    calibration loader. No vendor SDK, no Qt.
    stages/         one module per vendor: Thorlabs, Zaber, piezosystems Jena
    spectrometers/  one module per vendor: Ocean (seabreeze), Avantes
    vendor/         raw SDK wrappers that know nothing about Lillypad
    stitched        two spectrometers presented as one — vendor-neutral
    registry        enumeration, and the one vendor-tag -> class mapping

No vendor SDK is imported at module scope anywhere below this line. Every one of
them loads inside the function that needs it, so `import lillypad.hardware`
succeeds with only numpy installed and a missing DLL costs you exactly one
device's connect call — not the program.

The names re-exported here are the layer's public surface; everything else is an
implementation detail of the module it lives in.
"""

from lillypad.hardware.base import (C_NM_PER_FS, OVERLAP_BAND_FRAC,
                                    SPEC_VENDORS, SpectrometerBase, StageBase,
                                    StageLimitError, StageLockedError,
                                    load_calibration_file,
                                    spec_ident,
                                    split_spec_ident)
from lillypad.hardware.registry import list_spectrometers, open_spectrometer
from lillypad.hardware.spectrometers.avantes import (AvantesSpectrometer,
                                                     avantes_trigger_options,
                                                     list_avantes_spectrometers)
from lillypad.hardware.spectrometers.seabreeze import (
    SEABREEZE_BACKENDS, SeabreezeSpectrometer, list_seabreeze_spectrometers,
    select_seabreeze_backend)
from lillypad.hardware.stages.kinesis import KinesisStage, list_kinesis_stages
from lillypad.hardware.stages.piezo_jena import PiezoJenaStage
from lillypad.hardware.stages.zaber import ZaberStage
from lillypad.hardware.stitched import StitchedSpectrometer

__all__ = [
    "C_NM_PER_FS", "OVERLAP_BAND_FRAC", "SPEC_VENDORS",
    "StageBase", "StageLimitError", "StageLockedError", "SpectrometerBase",
    "load_calibration_file",
    "spec_ident", "split_spec_ident",
    "KinesisStage", "ZaberStage", "PiezoJenaStage", "list_kinesis_stages",
    "SeabreezeSpectrometer", "list_seabreeze_spectrometers",
    "select_seabreeze_backend", "SEABREEZE_BACKENDS",
    "AvantesSpectrometer", "list_avantes_spectrometers",
    "avantes_trigger_options",
    "StitchedSpectrometer",
    "list_spectrometers", "open_spectrometer",
]
