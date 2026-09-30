"""The pop-up windows: connection, settings, and export.

    base          _ConnectDialog (the shared base) + DevicePickerDialog
    spectrometer  stage  simulation     connecting devices
    acquisition   alignment  graphics   settings
    avantes                             vendor-specific device settings
    export                              format, file name, Save

Most of these are Qt.Tool windows: a thin frame that floats above the main
window without being modal, so a setting can be changed while watching the live
feed react to it. DevicePickerDialog is the exception and is modal, because it
asks a question the caller needs answered.

Nothing is re-exported here — each dialog is imported from its own module, which
keeps this package's import graph flat and says at every call site which dialog
is meant.
"""
