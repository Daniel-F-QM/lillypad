"""
avantes.py — ctypes wrapper for the Avantes AvaSpec-DLL
=======================================================
A self-contained Python API for Avantes AvaSpec spectrometers, written against
the vendor's C DLL. Nothing in here knows about Lillypad; `hardware.py` wraps
the `AvaSpec` class below in a `SpectrometerBase` adapter.

    python avantes.py          # self-test: struct layout + whatever is attached

Getting the DLL
---------------
This module needs **AvaSpecX64.dll** (64-bit) or **avaspec.dll** (32-bit),
whichever matches the running interpreter. Those ship in the Avantes
**AvaSpec-DLL** package, which is a SEPARATE download from AvaSoft: installing
AvaSoft alone leaves you with a 32-bit `avaspec.dll` inside the AvaSoft folder
and nothing a 64-bit Python can load. `_find_dll` says so explicitly when it
hits that case, because the failure mode otherwise is a bare WinError 193.

Set LILLYPAD_AVASPEC_DLL to a full path to override the search.

Units and conventions
---------------------
  * Integration time is in MILLISECONDS everywhere (the DLL's own unit).
  * Wavelengths are in nm; spectra are raw ADC counts as doubles.
  * Every DLL function returns int: 0 (or a count) on success, negative on
    error. `_check` turns a negative into an AvantesError naming the code.

Provenance
----------
Struct layouts, sizes, value ranges and error codes are taken from the Avantes
*AvaSpec Library* user manual, ID 020381, version 9.11.0.0 (May 2022), Table 6
and section 2.6.1. Function signatures come from Avantes' `avaspec.h`. Where
the manual is internally inconsistent — it is, once, about the total size of
DeviceConfigType — the code asks the DLL instead of picking a side; see
DeviceConfigHeadType.
"""

from __future__ import annotations

import atexit
import ctypes
import glob
import math
import os
import struct
import sys
import threading
import time
from ctypes import (POINTER, Structure, byref, c_bool, c_char, c_double,
                    c_float, c_int, c_int16, c_int32, c_short, c_ubyte,
                    c_uint, c_uint16, c_uint32, c_ushort, c_void_p)
from pathlib import Path

import numpy as np

# ===========================================================================
# Constants (avaspec.h)
# ===========================================================================
USER_ID_LEN         = 64
AVS_SERIAL_LEN      = 10
MAX_TEMP_SENSORS    = 3
VERSION_LEN         = 16
DETECTOR_NAME_LEN   = 20
OEM_DATA_LEN        = 4096
NR_WAVELEN_POL_COEF = 5
NR_NONLIN_POL_COEF  = 8
MAX_VIDEO_CHANNELS  = 2
NR_DEFECTIVE_PIXELS = 30
MAX_NR_PIXELS       = 4096
NR_TEMP_POL_COEF    = 5
NR_DAC_POL_COEF     = 2
CLIENT_ID_SIZE      = 32
ETHSET_RES_SIZE     = 79
# DeviceConfigType is padded to exactly 62 KiB minus the 4-byte CRC the
# firmware appends, and avaspec.h derives the reserved block's length from
# that rather than stating it:
#
#   SETTINGS_RESERVED_LEN = 62*1024 - sizeof(uint32) - sizeof(every other member)
#
# So the total is fixed at 63484 and the reserved length follows from the rest
# of the layout. Both are computed here the same way, from the structs below,
# so they cannot drift out of step with the field definitions.
DEVICE_CONFIG_SIZE = 62 * 1024 - 4
# m_ConfigVersion is NOT used to validate the layout. The manual gives its
# value as 3; a ULS4096CL-EVO on firmware 1.11 reports 257 (0x0101), which
# looks like a packed major.minor rather than the documented scalar. Since the
# encoding is not reliably documented it is reported as-is and nothing is
# gated on it — _read_config checks things that genuinely prove the offsets.

# Detector chips, from Table 6 of the library manual. Used to name the detector
# and to decide whether a sensor-specific call is worth making at all.
SENSOR_TYPES = {
    0x00: "reserved",        0x01: "HAMS8378-256",   0x02: "HAMS8378-1024",
    0x03: "ILX554",          0x04: "HAMS9201",       0x05: "Toshiba TCD1304",
    0x06: "TSL1301",         0x07: "TSL1401",        0x08: "HAMS8378-512",
    0x09: "HAMS9840",        0x0A: "ILX511",         0x0B: "HAMS10420-11850",
    0x0C: "HAMS11071-2048x64", 0x0D: "HAMS7031-11501",
    0x0E: "HAMS7031-1024x58", 0x0F: "HAMS11071-2048x16",
    0x10: "HAMS11155",       0x11: "SU256LSB",       0x12: "SU512LDB",
    0x13: "reserved",        0x14: "reserved",       0x15: "HAMS11638",
    0x16: "HAMS11639",       0x17: "HAMS12443",      0x18: "HAMG9208-512",
    0x19: "HAMG13913",       0x1A: "HAMS13496",
}
# Detectors AVS_SetSensitivityMode is documented as working on. Everything else
# answers ERR_NOT_SUPPORTED_BY_SENSOR_TYPE (-120), so the call is skipped
# rather than made and swallowed.
SENSITIVITY_MODE_SENSORS = frozenset({0x04, 0x11, 0x12, 0x18})
# Saturation-detection level 2 additionally corrects inverted pixels, and is
# ILX554-only. It also cannot be combined with averaging (-110) or
# store-to-RAM (-114).
SAT_DETECT_LEVEL2_SENSORS = frozenset({0x03})

INVALID_AVS_HANDLE_VALUE = 1000

# AVS_Init() port selector.
INIT_USB  = 0
INIT_ETH  = 256
INIT_BOTH = -1

# TriggerType.m_Mode
SW_TRIGGER_MODE = 0     # free-running: the DLL starts each scan
HW_TRIGGER_MODE = 1     # wait for an edge/level on the trigger input
SS_TRIGGER_MODE = 2     # single-scan
# TriggerType.m_Source
EXTERNAL_TRIGGER = 0
SYNC_TRIGGER     = 1
# TriggerType.m_SourceType
EDGE_TRIGGER_SOURCE  = 0
LEVEL_TRIGGER_SOURCE = 1

# AVS_GetAnalogIn() input ids on the AS7010 board. Id 6 is the one worth
# having: it is a digital sensor and returns DEGREES CELSIUS directly, where
# id 0 returns thermistor VOLTS that have to go through the device's own
# m_aTemperature polynomial to become a temperature.
ANALOG_IN_THERMISTOR   = 0
ANALOG_IN_AI2          = 4      # pin 18 on the 26-pin connector
ANALOG_IN_AI1          = 5      # pin 9
ANALOG_IN_BOARD_TEMP_C = 6      # digital sensor, already in °C

NR_DIGITAL_OUTPUTS = 13
NR_DIGITAL_INPUTS  = 13
NR_ANALOG_OUTPUTS  = 2

# DEVICE_STATUS values reported in AvsIdentityType.Status.
DEVICE_STATUS = {
    0: "unknown",
    1: "USB available",
    2: "USB in use by this application",
    3: "USB in use by another application",
    4: "ethernet available",
    5: "ethernet in use by this application",
    6: "ethernet in use by another application",
    7: "already in use over USB",
}

# AvsDeviceType (AVS_GetDeviceType).
DEVICE_TYPES = {0: "unknown", 1: "AS5216", 2: "Mini", 3: "AS7010", 4: "AS7007"}

# Full scale of one pixel, in counts. The AS7010 has a 16-bit ADC but the DLL
# reports 14-bit values unless AVS_UseHighResAdc is enabled, at which point the
# same signal comes back scaled by ADC_HIGH_RES_FACTOR. That factor is why the
# EEPROM's non-linearity and irradiance coefficients — which are stored in the
# 14-bit domain — need it applied when working in high-resolution mode.
ADC_FULL_SCALE_14BIT  = 16383.0
ADC_FULL_SCALE_16BIT  = 65535.0
ADC_HIGH_RES_FACTOR   = 4.0

# MeasConfigType limits, from Table 6 of the library manual. These are the
# LIBRARY's bounds; a given sensor may be narrower still and answers
# ERR_INVALID_INT_TIME (-11) for anything it cannot do, so these only catch the
# obviously-wrong before it reaches the device.
MIN_INTEGRATION_MS = 0.002
MAX_INTEGRATION_MS = 600_000.0
MAX_SMOOTH_PIX     = 2048
# A long exposure combined with heavy averaging is refused as
# ERR_INVALID_COMBINATION (-12); the manual's example is 600000 ms with
# Navg > 5000.
MAX_AVERAGES_AT_MAX_INTEGRATION = 5000


# ===========================================================================
# Error codes (avaspec.h)
# ===========================================================================
# Kept complete and verbatim: these strings are what the operator sees when a
# connect or a measurement fails, and a bare "-24" in a status bar is useless.
ERROR_CODES: dict[int, tuple[str, str]] = {
    0:    ("ERR_SUCCESS", "success"),
    -1:   ("ERR_INVALID_PARAMETER", "Function called with an invalid parameter value."),
    -2:   ("ERR_OPERATION_NOT_SUPPORTED", "This device does not support that operation (e.g. 16-bit ADC mode on 14-bit hardware)."),
    -3:   ("ERR_DEVICE_NOT_FOUND", "Opening communication failed, or a time-out occurred."),
    -4:   ("ERR_INVALID_DEVICE_ID", "The device handle is unknown to the DLL."),
    -5:   ("ERR_OPERATION_PENDING", "A previous measurement has not finished yet."),
    -6:   ("ERR_TIMEOUT", "No answer received from the device."),
    -7:   ("ERR_RESERVED", "Reserved error code."),
    -8:   ("ERR_INVALID_MEAS_DATA", "No measurement data available yet."),
    -9:   ("ERR_INVALID_SIZE", "The allocated buffer is too small."),
    -10:  ("ERR_INVALID_PIXEL_RANGE", "Invalid pixel range."),
    -11:  ("ERR_INVALID_INT_TIME", "Integration time is out of range for this sensor."),
    -12:  ("ERR_INVALID_COMBINATION", "Invalid combination of measurement settings — e.g. a 600000 ms integration time with more than 5000 averages."),
    -13:  ("ERR_RESERVED", "Reserved error code."),
    -14:  ("ERR_NO_MEAS_BUFFER_AVAIL", "No measurement buffers available."),
    -15:  ("ERR_UNKNOWN", "Unknown error reported by the spectrometer."),
    -16:  ("ERR_COMMUNICATION", "Communication error, or the ethernet connection failed."),
    -17:  ("ERR_NO_SPECTRA_IN_RAM", "No spectra in RAM — all read already, or the measurement was never started."),
    -18:  ("ERR_INVALID_DLL_VERSION", "DLL version information could not be retrieved."),
    -19:  ("ERR_NO_MEMORY", "Memory allocation error inside the DLL."),
    -20:  ("ERR_DLL_INITIALISATION", "Function called before AVS_Init()."),
    -21:  ("ERR_INVALID_STATE", "Invalid state (e.g. AVS_Measure without AVS_PrepareMeasure first)."),
    -22:  ("ERR_INVALID_REPLY", "The reply is not a recognised protocol message."),
    -23:  ("ERR_RESERVED", "Reserved error code."),
    -24:  ("ERR_ACCESS", "Access denied — e.g. insufficient user rights for the USB device."),
    -25:  ("ERR_INTERNAL_READ", "The spectrometer failed to read its own flash memory or on-board temperature sensor."),
    -26:  ("ERR_INTERNAL_WRITE", "The spectrometer failed to write its configuration to flash memory."),
    -27:  ("ERR_ETHCONN_REUSE", "Ethernet initialisation failed — another instance of the library is running on this machine, or AVS_Init() was called too soon after AVS_Done(). Retrying AVS_Init() usually clears it."),
    -28:  ("ERR_INVALID_DEVICE_TYPE", "The device type stored in the spectrometer is not one the library recognises."),
    -29:  ("ERR_SECURE_CFG_NOT_READ", "The secure configuration has not been read yet — the device is probably not initialised correctly."),
    -30:  ("ERR_UNEXPECTED_MEAS_RESPONSE", "Unexpected response while fetching measurement data; the measurement was stopped."),
    # Not in the v9.11 manual; kept so a newer library's code still gets a name.
    -31:  ("ERR_MEAS_STOPPED", "The measurement was stopped."),
    -100: ("ERR_INVALID_PARAMETER_NR_PIXELS", "The pixel count in the device data is incorrect."),
    -101: ("ERR_INVALID_PARAMETER_ADC_GAIN", "ADC gain setting out of range."),
    -102: ("ERR_INVALID_PARAMETER_ADC_OFFSET", "ADC offset setting out of range."),
    -110: ("ERR_INVALID_MEASPARAM_AVG_SAT2", "Saturation detection level 2 cannot be combined with averaging."),
    -111: ("ERR_INVALID_MEASPARAM_AVG_RAM", "Averaging cannot be combined with store-to-RAM."),
    -112: ("ERR_INVALID_MEASPARAM_SYNC_RAM", "Synchronise cannot be combined with store-to-RAM."),
    -113: ("ERR_INVALID_MEASPARAM_LEVEL_RAM", "Level triggering cannot be combined with store-to-RAM."),
    -114: ("ERR_INVALID_MEASPARAM_SAT2_RAM", "Saturation detection level 2 cannot be combined with store-to-RAM."),
    -115: ("ERR_INVALID_MEASPARAM_FWVER_RAM", "Store-to-RAM needs firmware 0.20.0.0 or newer."),
    -116: ("ERR_INVALID_MEASPARAM_DYNDARK", "This device does not support dynamic dark correction."),
    -120: ("ERR_NOT_SUPPORTED_BY_SENSOR_TYPE", "Not supported by this detector type."),
    -121: ("ERR_NOT_SUPPORTED_BY_FW_VER", "Not supported by this firmware version."),
    -122: ("ERR_NOT_SUPPORTED_BY_FPGA_VER", "Not supported by this FPGA version."),
    -140: ("ERR_SL_CALIBRATION_NOT_AVAILABLE", "This device is not calibrated for stray-light correction."),
    -141: ("ERR_SL_STARTPIXEL_NOT_IN_RANGE", "Incorrect stray-light start pixel in the device EEPROM."),
    -142: ("ERR_SL_ENDPIXEL_NOT_IN_RANGE", "Incorrect stray-light end pixel in the device EEPROM."),
    -143: ("ERR_SL_STARTPIX_GT_ENDPIX", "Incorrect stray-light pixel range in the device EEPROM."),
    -144: ("ERR_SL_MFACTOR_OUT_OF_RANGE", "Stray-light factor must be between 0.0 and 4.0."),
}

ERR_INVALID_SIZE = -9


class AvantesError(RuntimeError):
    """A DLL call returned a negative status. Carries the numeric `code` and
    the symbolic `name` so callers can branch on a specific failure (notably
    ERR_OPERATION_NOT_SUPPORTED / ERR_NOT_SUPPORTED_BY_SENSOR_TYPE, which are
    "this model does not have that feature" rather than real errors)."""

    def __init__(self, code: int, call: str):
        self.code = int(code)
        self.name, text = ERROR_CODES.get(
            self.code, ("ERR_UNRECOGNISED", "Unrecognised error code."))
        self.call = call
        super().__init__(f"{call} failed: {text} ({self.name}, {self.code})")


def _check(rc: int, call: str) -> int:
    """Raise on a negative return, otherwise pass the value through — several
    calls return a meaningful count (AVS_Init, AVS_GetNrOfDevices) or a state
    (AVS_PollScan), so success is 'not negative', not 'zero'."""
    if rc < 0:
        raise AvantesError(rc, call)
    return rc


# ===========================================================================
# Structures — every one packed to 1 byte
# ===========================================================================
# _pack_ = 1 is mandatory, not stylistic: the DLL reads these as packed C
# structs, and with default alignment MeasConfigType grows from 41 to 48 bytes
# and AVS_PrepareMeasure silently receives garbage instead of failing. The
# sizes are asserted at import time (see _assert_struct_sizes).
class AvsIdentityType(Structure):
    _pack_ = 1
    _fields_ = [("SerialNumber", c_char * AVS_SERIAL_LEN),
                ("UserFriendlyName", c_char * USER_ID_LEN),
                ("Status", c_ubyte)]


class DarkCorrectionType(Structure):
    _pack_ = 1
    _fields_ = [("m_Enable", c_ubyte),
                ("m_ForgetPercentage", c_ubyte)]


class SmoothingType(Structure):
    _pack_ = 1
    _fields_ = [("m_SmoothPix", c_uint16),
                ("m_SmoothModel", c_ubyte)]


class TriggerType(Structure):
    _pack_ = 1
    _fields_ = [("m_Mode", c_ubyte),
                ("m_Source", c_ubyte),
                ("m_SourceType", c_ubyte)]


class ControlSettingsType(Structure):
    _pack_ = 1
    _fields_ = [("m_StrobeControl", c_uint16),
                ("m_LaserDelay", c_uint32),
                ("m_LaserWidth", c_uint32),
                ("m_LaserWaveLength", c_float),
                ("m_StoreToRam", c_uint16)]


class MeasConfigType(Structure):
    _pack_ = 1
    _fields_ = [("m_StartPixel", c_uint16),
                ("m_StopPixel", c_uint16),
                ("m_IntegrationTime", c_float),      # milliseconds
                ("m_IntegrationDelay", c_uint32),
                ("m_NrAverages", c_uint32),
                ("m_CorDynDark", DarkCorrectionType),
                ("m_Smoothing", SmoothingType),
                ("m_SaturationDetection", c_ubyte),
                ("m_Trigger", TriggerType),
                ("m_Control", ControlSettingsType)]


class DetectorType(Structure):
    _pack_ = 1
    _fields_ = [("m_SensorType", c_ubyte),
                ("m_NrPixels", c_uint16),
                ("m_aFit", c_float * NR_WAVELEN_POL_COEF),
                ("m_NLEnable", c_bool),
                ("m_aNLCorrect", c_double * NR_NONLIN_POL_COEF),
                ("m_aLowNLCounts", c_double),
                ("m_aHighNLCounts", c_double),
                ("m_Gain", c_float * MAX_VIDEO_CHANNELS),
                ("m_Reserved", c_float),
                ("m_Offset", c_float * MAX_VIDEO_CHANNELS),
                ("m_ExtOffset", c_float),
                ("m_DefectivePixels", c_uint16 * NR_DEFECTIVE_PIXELS)]


class SpectrumCalibrationType(Structure):
    _pack_ = 1
    _fields_ = [("m_Smoothing", SmoothingType),
                ("m_CalInttime", c_float),
                ("m_aCalibConvers", c_float * MAX_NR_PIXELS)]


class IrradianceType(Structure):
    _pack_ = 1
    _fields_ = [("m_IntensityCalib", SpectrumCalibrationType),
                ("m_CalibrationType", c_ubyte),
                ("m_FiberDiameter", c_uint32)]


class SpectrumCorrectionType(Structure):
    _pack_ = 1
    _fields_ = [("m_aSpectrumCorrect", c_float * MAX_NR_PIXELS)]


class StandAloneType(Structure):
    _pack_ = 1
    _fields_ = [("m_Enable", c_bool),
                ("m_Meas", MeasConfigType),
                ("m_Nmsr", c_int16)]


class DynamicStorageType(Structure):
    _pack_ = 1
    _fields_ = [("m_Nmsr", c_int32),
                ("m_Reserved", c_ubyte * 8)]


class TempSensorType(Structure):
    _pack_ = 1
    _fields_ = [("m_aFit", c_float * NR_TEMP_POL_COEF)]


class TecControlType(Structure):
    _pack_ = 1
    _fields_ = [("m_Enable", c_bool),
                ("m_Setpoint", c_float),
                ("m_aFit", c_float * NR_DAC_POL_COEF)]


class ProcessControlType(Structure):
    _pack_ = 1
    _fields_ = [("m_AnalogLow", c_float * 2),
                ("m_AnalogHigh", c_float * 2),
                ("m_DigitalLow", c_float * 10),
                ("m_DigitalHigh", c_float * 10)]


class EthernetSettingsType(Structure):
    """Straight from avaspec.h.

    The v9.11 LIBRARY MANUAL documents two extra fields here
    (m_MeasurementDataPortKey / m_MeasurementDataPort, carved out of the
    reserved tail, which it shortens to 75). The 9.14 header has neither and
    keeps m_Reserved[79]. The header wins — it is what the DLL was compiled
    against. The total is 128 bytes either way, so nothing after this struct
    shifts; only the field offsets inside it would have been wrong.
    """
    _pack_ = 1
    _fields_ = [("m_IpAddr", c_uint32),
                ("m_NetMask", c_uint32),
                ("m_Gateway", c_uint32),
                ("m_DhcpEnabled", c_ubyte),
                ("m_TcpPort", c_uint16),
                ("m_LinkStatus", c_ubyte),
                ("m_ClientIdType", c_ubyte),
                ("m_ClientIdCustom", c_char * CLIENT_ID_SIZE),
                ("m_Reserved", c_ubyte * ETHSET_RES_SIZE)]


class DeviceConfigHeadType(Structure):
    """The LEADING, well-determined part of DeviceConfigType.

    The full struct ends with `uint8 m_aReserved[SETTINGS_RESERVED_LEN]` and a
    4096-byte OEM block, and the sources disagree about the reserved length —
    MSL-Equipment uses 9720 and omits the two bools below, while Avantes' own
    Python sample just passes a flat 63484-byte blob. Rather than guess, this
    module never declares the whole struct: it asks the DLL how many bytes
    DeviceConfigType is (AVS_GetParameter's two-call sizing), keeps the answer
    as an opaque buffer, and casts only THIS head over the front of it. Reads
    and writes therefore round-trip the tail untouched, and nothing depends on
    a constant no source agrees on.

    The v9.11 manual is internally inconsistent here, which is the whole
    justification for not trusting a constant. It states every sub-struct's
    size AND m_aReserved[9718] AND "Size = 63484" for the whole thing — but
    those sizes sum to 63596, 112 bytes more. Every individual number below is
    quoted verbatim from Table 6 and each one checks out; only the stated total
    does not, and it matches what an older wrapper hardcoded, so 63484 is most
    likely stale rather than the parts being wrong. Either way the DLL is the
    authority: ask it, then cast this head over what it returns.

    `AvaSpec.config_verified` reports whether the head then passed its sanity
    check (see _read_config), and anything reading past the detector block goes
    through that flag.
    """
    _pack_ = 1
    _fields_ = [("m_Len", c_uint16),
                ("m_ConfigVersion", c_uint16),
                ("m_aUserFriendlyId", c_char * USER_ID_LEN),
                ("m_Detector", DetectorType),
                ("m_Irradiance", IrradianceType),
                ("m_Reflectance", SpectrumCalibrationType),
                ("m_SpectrumCorrect", SpectrumCorrectionType),
                ("m_StandAlone", StandAloneType),
                ("m_DynamicStorage", DynamicStorageType),
                ("m_aTemperature", TempSensorType * MAX_TEMP_SENSORS),
                ("m_TecControl", TecControlType),
                ("m_ProcessControl", ProcessControlType),
                ("m_EthernetSettings", EthernetSettingsType),
                ("m_MessageAckDisable", c_bool),
                ("m_IncludeCRC", c_bool)]


class BroadcastAnswerType(Structure):
    _pack_ = 1
    _fields_ = [("InterfaceType", c_ubyte),
                ("serial", c_ubyte * AVS_SERIAL_LEN),
                ("port", c_ushort),
                ("status", c_ubyte),
                ("RemoteHostIp", c_uint32),
                ("LocalIp", c_uint32),
                ("reserved", c_ubyte * 4)]


# Expected sizes, checked at import time so a packing mistake surfaces as a
# clear assertion on a dev machine with no hardware instead of as a corrupt
# measurement on the bench.
#
# These are split by how much they are worth. DOCUMENTED sizes are stated
# outright in the AvaSpec-DLL sources, so matching one is independent evidence
# that the field list AND the packing are right — MeasConfigType at 41 bytes is
# the important one, since it is the struct AVS_PrepareMeasure reads. DERIVED
# sizes are just the sum of the fields as declared, so they only prove that
# _pack_ = 1 took effect; they cannot catch a wrong field list. Keeping the two
# apart stops a derived number from being mistaken for a verified one.
_DOCUMENTED_SIZES = {
    AvsIdentityType: 75,
    BroadcastAnswerType: 26,
    ControlSettingsType: 16,
    DarkCorrectionType: 2,
    DetectorType: 188,
    DynamicStorageType: 12,
    IrradianceType: 16396,
    MeasConfigType: 41,
    ProcessControlType: 96,
    SmoothingType: 3,
    SpectrumCalibrationType: 16391,
    SpectrumCorrectionType: 16384,
    StandAloneType: 44,
    TecControlType: 13,
    TempSensorType: 20,
    TriggerType: 3,
}
_DERIVED_SIZES = {
    # avaspec.h gives this struct's fields; neither it nor the manual states a
    # size, so 128 is the sum of the fields as declared.
    EthernetSettingsType: 128,
}
_EXPECTED_SIZES = {**_DOCUMENTED_SIZES, **_DERIVED_SIZES}

# Length of DeviceConfigType.m_aReserved, derived exactly as avaspec.h derives
# it. Computed rather than hardcoded so it tracks the struct definitions above;
# the v9.11 manual's stated 9718 disagrees with the header by 112 bytes and is
# not used. Nothing reads the reserved block — this exists so the layout can be
# proved self-consistent at import.
SETTINGS_RESERVED_LEN = DEVICE_CONFIG_SIZE - 4096 - (
    2 + 2 + USER_ID_LEN
    + ctypes.sizeof(DetectorType) + ctypes.sizeof(IrradianceType)
    + ctypes.sizeof(SpectrumCalibrationType)
    + ctypes.sizeof(SpectrumCorrectionType)
    + ctypes.sizeof(StandAloneType) + ctypes.sizeof(DynamicStorageType)
    + ctypes.sizeof(TempSensorType) * MAX_TEMP_SENSORS
    + ctypes.sizeof(TecControlType) + ctypes.sizeof(ProcessControlType)
    + ctypes.sizeof(EthernetSettingsType) + 1 + 1)


def _assert_struct_sizes() -> None:
    bad = [(t.__name__, ctypes.sizeof(t), n)
           for t, n in _EXPECTED_SIZES.items() if ctypes.sizeof(t) != n]
    # The head plus the reserved block plus the OEM block must come to exactly
    # 62 KiB minus the CRC. That is the header's own invariant, and it ties
    # every struct above together: get any one of them wrong and this fails.
    total = ctypes.sizeof(DeviceConfigHeadType) + SETTINGS_RESERVED_LEN + 4096
    if total != DEVICE_CONFIG_SIZE:
        bad.append(("DeviceConfigType (head+reserved+OEM)", total,
                    DEVICE_CONFIG_SIZE))
    if SETTINGS_RESERVED_LEN < 0:
        bad.append(("SETTINGS_RESERVED_LEN", SETTINGS_RESERVED_LEN, 0))
    if bad:
        raise AssertionError(
            "Avantes struct packing is wrong — the DLL would read garbage. "
            + "; ".join(f"{name}: {got} bytes, expected {want}"
                        for name, got, want in bad))


_assert_struct_sizes()


# ===========================================================================
# DLL loading
# ===========================================================================
# AvsHandle is `long`, which is 32-bit on Windows even in an x64 build and
# 64-bit on Linux. Getting this wrong truncates handles rather than failing.
AvsHandle = c_int32 if sys.platform == "win32" else ctypes.c_long

_ENV_VAR = "LILLYPAD_AVASPEC_DLL"
_DLL_NAMES = ("AvaSpecX64.dll",) if struct.calcsize("P") == 8 else ("avaspec.dll",)

_lib = None                      # the loaded CDLL/WinDLL, or None
_lib_path: str | None = None
# The DLL is not reentrant, and callers drive it from more than one thread (a
# live-feed thread acquiring while the UI thread reads a temperature). Every
# call below takes this lock. It is deliberately module-global rather than
# per-device: the library's own state — AVS_Init/AVS_Done, the device list — is
# process-wide. The lock is taken per CALL, never held across the poll loop, so
# a long integration on one device does not block a status read on another.
_LOCK = threading.RLock()

# AVS_Init/AVS_Done are process-global and AVS_Done tears down EVERY open
# device, so the library is refcounted: the first device opened initialises it
# and only the last one closed shuts it down. Without this, disconnecting one
# member of a stitched pair would kill the other.
_init_refs = 0
_init_port: int | None = None

# Serial numbers currently activated in this process, mapped to the AvaSpec
# that owns each. AVS_Activate on an already-active device does NOT fail and
# does NOT open a second session — it returns the SAME handle. Two AvaSpec
# objects would then alias one device, and whichever was closed first would
# deactivate it for both, leaving the other raising ERR_INVALID_STATE (-21)
# from its next call with nothing to suggest why. Verified on an
# AvaSpec-ULS4096CL-EVO: both objects got handle 1. So a second open of the
# same serial is refused here rather than handed out.
_open_serials: dict[str, "AvaSpec"] = {}


@atexit.register
def _close_open_devices() -> None:
    """Deactivate anything still open when the process ends.

    Without this, a script that dies on an unhandled exception leaves a device
    activated and the DLL loaded, and the interpreter then dies during unload
    with STATUS_STACK_BUFFER_OVERRUN (0xC0000409) — a fail-fast crash whose
    exit code tells you nothing about the real error, which scrolled past
    earlier. Observed exactly that during bring-up. Best-effort and silent: by
    this point there is nothing useful to report and nowhere to report it.
    """
    for spec in list(_open_serials.values()):
        try:
            spec.disconnect()
        except Exception:
            pass


def _candidate_paths() -> list[str]:
    """Where to look for the DLL, in order of decreasing confidence."""
    out: list[str] = []
    env = os.environ.get(_ENV_VAR)
    if env:
        out.append(env)
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for name in _DLL_NAMES:
        out.append(str(here / name))
    if sys.platform == "win32":
        # The AvaSpec-DLL installer defaults to a versioned folder at the ROOT
        # of the system drive (C:\AvaSpecX64-DLL_9.14.0.0\), not Program Files.
        # Newest version first, so a machine with several installs picks the
        # latest rather than whichever sorts first.
        sysdrive = os.environ.get("SystemDrive", "C:") + os.sep
        for pattern in ("AvaSpec*DLL*", os.path.join("Avantes", "*"), "Avantes"):
            for name in _DLL_NAMES:
                out.extend(sorted(glob.glob(os.path.join(sysdrive, pattern, name)),
                                  reverse=True))
        for base in (r"C:\Program Files\Avantes", r"C:\Program Files (x86)\Avantes"):
            for name in _DLL_NAMES:
                out.extend(sorted(glob.glob(os.path.join(base, "*", name)),
                                  reverse=True))
                out.extend(sorted(glob.glob(os.path.join(base, name))))
        # AvaSoft carries a 32-bit avaspec.dll. It is listed LAST and only so
        # that a 64-bit interpreter finds it and can say why it is useless,
        # rather than reporting "no DLL found" while one sits on the disk.
        out.append(r"C:\Program Files (x86)\AvaSoft8\avaspec.dll")
    else:
        out.extend(["/usr/local/lib/libavs.so", "libavs.so.0", "libavs.so"])
    # Last resort: let the OS search PATH.
    out.extend(_DLL_NAMES)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _bitness_of(path: str) -> str | None:
    """'x86' / 'x64' / 'arm64' for a PE file, or None if it cannot be read.
    Used only to explain a WinError 193 in terms an operator can act on."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0x3C)
            pe = int.from_bytes(fh.read(4), "little")
            fh.seek(pe + 4)
            machine = int.from_bytes(fh.read(2), "little")
    except Exception:
        return None
    return {0x14C: "x86", 0x8664: "x64", 0xAA64: "arm64"}.get(machine)


def _load_dll():
    """Load the AvaSpec DLL, or raise a RuntimeError that says what to install.

    The message matters more than usual here: the overwhelmingly common failure
    is having installed AvaSoft (which ships only a 32-bit avaspec.dll) instead
    of the AvaSpec-DLL package, and the raw OSError for that is
    "[WinError 193] %1 is not a valid Win32 application", which tells an
    operator nothing.
    """
    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    tried: list[str] = []
    wrong_bits: list[str] = []
    for path in _candidate_paths():
        if os.path.isabs(path) and not os.path.exists(path):
            continue
        tried.append(path)
        try:
            return loader(path), path
        except OSError as e:
            # WinError 193 == "not a valid Win32 application" == wrong bitness.
            if getattr(e, "winerror", None) == 193 or "not a valid Win32" in str(e):
                bits = _bitness_of(path) or "the wrong architecture"
                wrong_bits.append(f"{path} ({bits})")
        except Exception:
            pass

    want = "64-bit" if struct.calcsize("P") == 8 else "32-bit"
    msg = [f"Avantes {' / '.join(_DLL_NAMES)} not found or not loadable."]
    if wrong_bits:
        msg.append(
            f"Found {', '.join(wrong_bits)}, but this is {want} Python and the "
            f"architectures must match. AvaSoft ships ONLY the 32-bit "
            f"avaspec.dll; AvaSpecX64.dll comes in the separate Avantes "
            f"AvaSpec-DLL package.")
    else:
        msg.append(
            "Install the Avantes AvaSpec-DLL package (a separate download from "
            "AvaSoft) — Lillypad does not ship it.")
    msg.append(f"Set {_ENV_VAR} to the full path of the DLL if it is installed "
               f"somewhere unusual.")
    if tried:
        msg.append("Looked in: " + ", ".join(tried) + ".")
    raise RuntimeError(" ".join(msg))


def _prototypes(lib) -> None:
    """Declare argtypes/restypes. Without these ctypes guesses, and on x64 a
    guessed `double*` argument is passed in the wrong register class."""
    P_d, P_i, P_u8 = POINTER(c_double), POINTER(c_int), POINTER(c_ubyte)
    P_u16, P_u32, P_f = POINTER(c_uint16), POINTER(c_uint32), POINTER(c_float)

    def sig(name, restype, *argtypes):
        fn = getattr(lib, name, None)
        if fn is None:                # older DLLs lack the newer entry points
            return
        fn.restype = restype
        fn.argtypes = list(argtypes)

    sig("AVS_Init", c_int, c_short)
    sig("AVS_Done", c_int)
    sig("AVS_GetNrOfDevices", c_int)
    sig("AVS_UpdateUSBDevices", c_int)
    sig("AVS_UpdateETHDevices", c_int, c_uint32, P_u32, POINTER(BroadcastAnswerType))
    sig("AVS_GetList", c_int, c_uint32, P_u32, POINTER(AvsIdentityType))
    sig("AVS_Activate", AvsHandle, POINTER(AvsIdentityType))
    sig("AVS_Deactivate", c_bool, AvsHandle)
    sig("AVS_GetHandleFromSerial", AvsHandle, ctypes.c_char_p)

    sig("AVS_PrepareMeasure", c_int, AvsHandle, POINTER(MeasConfigType))
    # The callback argument is declared void* and always passed NULL: this
    # module polls with AVS_PollScan instead. The callback typedef carries no
    # calling-convention decoration in avaspec.h, Avantes' own sample declares
    # it __cdecl while MSL-Equipment declares it __stdcall, and getting that
    # wrong corrupts the stack rather than raising. Polling sidesteps it.
    sig("AVS_Measure", c_int, AvsHandle, c_void_p, c_short)
    sig("AVS_StopMeasure", c_int, AvsHandle)
    # int, NOT bool: 0 = no data, 1 = data ready, negative = error. Declared as
    # c_bool (as one published wrapper does) an error reads as "data ready".
    sig("AVS_PollScan", c_int, AvsHandle)
    sig("AVS_GetScopeData", c_int, AvsHandle, P_u32, P_d)
    sig("AVS_GetSaturatedPixels", c_int, AvsHandle, P_u8)
    sig("AVS_GetDarkPixelData", c_int, AvsHandle, P_d)

    sig("AVS_GetLambda", c_int, AvsHandle, P_d)
    sig("AVS_GetNumPixels", c_int, AvsHandle, P_u16)
    sig("AVS_GetParameter", c_int, AvsHandle, c_uint32, P_u32, c_void_p)
    sig("AVS_SetParameter", c_int, AvsHandle, c_void_p)

    sig("AVS_GetVersionInfo", c_int, AvsHandle,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
    sig("AVS_GetDLLVersion", c_int, ctypes.c_char_p)
    sig("AVS_GetDeviceType", c_int, AvsHandle, P_u8)
    sig("AVS_GetDetectorName", c_int, AvsHandle, c_ubyte, ctypes.c_char_p)

    sig("AVS_UseHighResAdc", c_int, AvsHandle, c_bool)
    sig("AVS_SetSyncMode", c_int, AvsHandle, c_ubyte)
    sig("AVS_SetPrescanMode", c_int, AvsHandle, c_bool)
    sig("AVS_SetSensitivityMode", c_int, AvsHandle, c_uint32)

    sig("AVS_GetAnalogIn", c_int, AvsHandle, c_ubyte, P_f)
    sig("AVS_SetAnalogOut", c_int, AvsHandle, c_ubyte, c_float)
    sig("AVS_GetDigIn", c_int, AvsHandle, c_ubyte, P_u8)
    sig("AVS_SetDigOut", c_int, AvsHandle, c_ubyte, c_ubyte)
    sig("AVS_ResetDevice", c_int, AvsHandle)


def load(path: str | None = None):
    """Load and cache the DLL. Idempotent; `path` forces a specific file."""
    global _lib, _lib_path
    with _LOCK:
        if _lib is not None and path is None:
            return _lib
        if path is not None:
            loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
            _lib, _lib_path = loader(path), path
        else:
            _lib, _lib_path = _load_dll()
        _prototypes(_lib)
        return _lib


def dll_path() -> str | None:
    """Path of the loaded DLL, or None if it has not been loaded yet."""
    return _lib_path


def dll_version() -> str:
    lib = load()
    buf = ctypes.create_string_buffer(VERSION_LEN + 1)
    with _LOCK:
        _check(lib.AVS_GetDLLVersion(buf), "AVS_GetDLLVersion")
    return buf.value.decode("latin-1", "replace")


def _acquire_library(port: int) -> None:
    """Refcounted AVS_Init. See _init_refs."""
    global _init_refs, _init_port
    lib = load()
    with _LOCK:
        if _init_refs == 0:
            _check(lib.AVS_Init(c_short(port)), "AVS_Init")
            _init_port = port
        _init_refs += 1


def _release_library() -> None:
    """Refcounted AVS_Done. Never raises — it runs from disconnect()."""
    global _init_refs, _init_port
    with _LOCK:
        if _init_refs <= 0:
            return
        _init_refs -= 1
        if _init_refs == 0:
            try:
                _lib.AVS_Done()
            except Exception:
                pass
            _init_port = None


# ===========================================================================
# Enumeration
# ===========================================================================
def _update_devices(lib, port: int) -> int:
    if port == INIT_ETH:
        required = c_uint32(0)
        # Two-call sizing: ask with a zero-length buffer, then allocate.
        lib.AVS_UpdateETHDevices(0, byref(required), None)
        n = max(1, required.value // ctypes.sizeof(BroadcastAnswerType))
        buf = (BroadcastAnswerType * n)()
        return _check(lib.AVS_UpdateETHDevices(
            ctypes.sizeof(buf), byref(required), buf), "AVS_UpdateETHDevices")
    return _check(lib.AVS_UpdateUSBDevices(), "AVS_UpdateUSBDevices")


def _identities(port: int = INIT_USB) -> list[AvsIdentityType]:
    """Every attached device, as AvsIdentityType records."""
    lib = load()
    _acquire_library(port)
    try:
        with _LOCK:
            n = _update_devices(lib, port)
            if n <= 0:
                return []
            required = c_uint32(0)
            size = n * ctypes.sizeof(AvsIdentityType)
            buf = (AvsIdentityType * n)()
            rc = lib.AVS_GetList(size, byref(required), buf)
            if rc == ERR_INVALID_SIZE and required.value > size:
                # The DLL knows about more devices than AVS_UpdateUSBDevices
                # reported; take it at its word rather than truncating.
                n = required.value // ctypes.sizeof(AvsIdentityType)
                buf = (AvsIdentityType * n)()
                rc = lib.AVS_GetList(required.value, byref(required), buf)
            _check(rc, "AVS_GetList")
            return list(buf)
    finally:
        _release_library()


def display_name(user_name: str | None, serial: str) -> str:
    """Best human label for a device, given its EEPROM name and serial.

    The AvaSpec API has no model field. The nearest thing is the user-friendly
    name, which is free text: it may hold the model, but on a factory-fresh
    unit it is just the serial number again, and pairing that with the serial
    gives the useless "2006058U1 [2006058U1]". So it is used only when it says
    something the serial does not.

    Shared by list_devices and AvaSpec.model so an enumerated device and a
    connected one are never labelled differently.
    """
    name = (user_name or "").strip()
    return name if name and name != serial else "AvaSpec"


def list_devices(port: int = INIT_USB) -> list[tuple[str, str, str]]:
    """Enumerate attached spectrometers without opening any.

    Returns [(name, serial, status), ...]. `name` is the user-friendly name
    stored in the device (usually the model), `status` is a human-readable
    DEVICE_STATUS — "USB in use by another application" is the one to watch
    for: AvaSoft holds the device open and this API will not get it.
    """
    out = []
    for ident in _identities(port):
        serial = ident.SerialNumber.decode("latin-1", "replace").strip("\x00 ")
        name = ident.UserFriendlyName.decode("latin-1", "replace").strip("\x00 ")
        out.append((display_name(name, serial), serial,
                    DEVICE_STATUS.get(int(ident.Status), "unknown")))
    return out


# ===========================================================================
# The spectrometer
# ===========================================================================
class AvaSpec:
    """One Avantes spectrometer.

    The constructor connects; `disconnect()` is idempotent and never raises.
    A failed constructor cleans up after itself before re-raising, so a caller
    that only ever sees the exception leaks nothing.

        with AvaSpec() as spec:
            spec.integration_ms = 5.0
            counts = spec.acquire()

    Configuration is held in `self.config` (a MeasConfigType) and pushed to the
    device by AVS_PrepareMeasure. Changing anything sets `_prepared = False`,
    and the next acquire() re-prepares — so a caller can set several fields
    without paying for a device round-trip per field. The two settings the
    device can REFUSE (exposure and averaging) are the exception: they push
    immediately, so the refusal reaches the caller instead of the next
    acquire() on some other thread.
    """

    # Replaced in __init__ with what this sensor actually accepts. Present as
    # class attributes only so a half-built instance still has an answer.
    min_integration_ms: float = MIN_INTEGRATION_MS
    max_integration_ms: float = MAX_INTEGRATION_MS

    def __init__(self, serial: str | None = None, port: int = INIT_USB,
                 high_res_adc: bool = True):
        self._h: int | None = None
        # Serialises one whole measurement. The module lock is taken per DLL
        # CALL, which is not enough here: a measurement spans
        # PrepareMeasure -> Measure -> PollScan… -> GetScopeData, and two
        # threads interleaving those on one device make the second AVS_Measure
        # fail with ERR_OPERATION_PENDING (-5) — or, worse, hand one thread the
        # other's frame. Held across the whole sequence so concurrent acquires
        # queue instead of corrupting each other.
        self._measure_lock = threading.RLock()
        self._lib = load()
        self._holds_library = False
        self.port = int(port)
        self._prepared = False
        self._high_res = False
        self.config_verified = False
        self._config_raw: ctypes.Array | None = None

        _acquire_library(self.port)
        self._holds_library = True
        try:
            ident = self._find(serial)
            want = ident.SerialNumber.decode("latin-1", "replace").strip("\x00 ")
            with _LOCK:
                # Refuse before activating — see _open_serials. Doing it after
                # would be too late: the alias already exists by then.
                if want in _open_serials:
                    raise RuntimeError(
                        f"Avantes {want} is already open in this program. The "
                        f"AvaSpec library hands out the same handle for a "
                        f"second open, so the two would share one device and "
                        f"closing either would disconnect both. Release the "
                        f"existing connection first.")
                handle = self._lib.AVS_Activate(byref(ident))
            if handle == INVALID_AVS_HANDLE_VALUE or handle < 0:
                raise RuntimeError(
                    f"Could not open the Avantes spectrometer "
                    f"({ident.SerialNumber.decode('latin-1', 'replace').strip()}). "
                    f"Its status is "
                    f"'{DEVICE_STATUS.get(int(ident.Status), 'unknown')}' — if "
                    f"another program (AvaSoft) has it open, close that first.")
            self._h = int(handle)
            self.serial = want
            _open_serials[want] = self
            self.user_name = ident.UserFriendlyName.decode("latin-1", "replace").strip("\x00 ")

            n = c_uint16(0)
            self._call("AVS_GetNumPixels", byref(n))
            self.n_pixels = int(n.value)
            if not 1 <= self.n_pixels <= MAX_NR_PIXELS:
                raise RuntimeError(
                    f"Device reported an implausible pixel count "
                    f"({self.n_pixels}); expected 1..{MAX_NR_PIXELS}.")

            # Buffers are allocated at MAX_NR_PIXELS regardless of n_pixels:
            # Avantes' own sample does, and a DLL that writes a full 4096
            # doubles into a 2048-double buffer would corrupt the heap.
            self._lambda_buf = (c_double * MAX_NR_PIXELS)()
            self._scope_buf = (c_double * MAX_NR_PIXELS)()
            self._sat_buf = (c_ubyte * MAX_NR_PIXELS)()

            self._call("AVS_GetLambda", self._lambda_buf)
            self._wavelengths = np.frombuffer(
                self._lambda_buf, dtype=np.float64, count=self.n_pixels).copy()

            self.config = self._default_config()
            self.set_high_res_adc(high_res_adc)
            self._read_config()
            # Asked once, here, so every later range check and every UI built
            # from this range talks about THIS sensor rather than the library.
            self.min_integration_ms = self._probe_min_integration()
            self.max_integration_ms = MAX_INTEGRATION_MS
        except Exception:
            self.disconnect()
            raise

    # ── plumbing ────────────────────────────────────────────────────────────
    def _find(self, serial: str | None) -> AvsIdentityType:
        with _LOCK:
            n = _update_devices(self._lib, self.port)
            if n <= 0:
                raise RuntimeError(
                    "No Avantes spectrometers found. Check the USB cable and "
                    "that the AvaSpec USB driver is installed (it ships with "
                    "AvaSoft and with the AvaSpec-DLL package).")
            required = c_uint32(0)
            buf = (AvsIdentityType * n)()
            _check(self._lib.AVS_GetList(ctypes.sizeof(buf), byref(required), buf),
                   "AVS_GetList")
        if serial is None:
            return buf[0]
        want = str(serial).strip()
        for ident in buf:
            if ident.SerialNumber.decode("latin-1", "replace").strip("\x00 ") == want:
                return ident
        found = ", ".join(
            i.SerialNumber.decode("latin-1", "replace").strip("\x00 ") or "?"
            for i in buf)
        raise RuntimeError(
            f"No Avantes spectrometer with serial {serial!r}. Found: {found}.")

    def _call(self, name: str, *args) -> int:
        """One DLL call on this device's handle, under the library lock."""
        if self._h is None:
            raise RuntimeError("Spectrometer is disconnected.")
        fn = getattr(self._lib, name, None)
        if fn is None:
            raise RuntimeError(
                f"{name} is not exported by this AvaSpec DLL "
                f"({_lib_path}) — it is probably older than this module "
                f"expects.")
        with _LOCK:
            return _check(fn(AvsHandle(self._h), *args), name)

    def _default_config(self) -> MeasConfigType:
        """A plain free-running single scan over the full detector.

        Averaging is left at 1 deliberately: on-board averaging multiplies how
        long acquire() blocks, and consumers that already average in software
        would otherwise silently do it twice.
        """
        cfg = MeasConfigType()
        cfg.m_StartPixel = 0
        cfg.m_StopPixel = self.n_pixels - 1
        cfg.m_IntegrationTime = 10.0
        cfg.m_IntegrationDelay = 0
        cfg.m_NrAverages = 1
        cfg.m_CorDynDark.m_Enable = 0
        cfg.m_CorDynDark.m_ForgetPercentage = 100
        cfg.m_Smoothing.m_SmoothPix = 0
        cfg.m_Smoothing.m_SmoothModel = 0
        # Level 1: flags clipped pixels without the level-2 restrictions
        # (level 2 cannot be combined with averaging — ERR_INVALID_MEASPARAM_AVG_SAT2).
        cfg.m_SaturationDetection = 1
        cfg.m_Trigger.m_Mode = SW_TRIGGER_MODE
        cfg.m_Trigger.m_Source = EXTERNAL_TRIGGER
        cfg.m_Trigger.m_SourceType = EDGE_TRIGGER_SOURCE
        cfg.m_Control.m_StrobeControl = 0
        cfg.m_Control.m_LaserDelay = 0
        cfg.m_Control.m_LaserWidth = 0
        cfg.m_Control.m_LaserWaveLength = 0.0
        cfg.m_Control.m_StoreToRam = 0
        return cfg

    def _invalidate(self) -> None:
        """Mark the on-device configuration stale; the next acquire re-pushes."""
        self._prepared = False

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._call("AVS_PrepareMeasure", byref(self.config))
        self._prepared = True

    def _push_config(self, undo) -> None:
        """Send the current config to the device NOW, undoing it if refused.

        AVS_PrepareMeasure is the only thing that validates a measurement
        config, and acquire() would otherwise be the first call to make it —
        on whatever thread happens to be running the live feed, where the
        error is far away from the setting that caused it. Pushing here means
        an unacceptable value fails at the call site, and the device is left
        holding the last one that worked rather than a config it rejects.
        """
        self._invalidate()
        try:
            with self._measure_lock:
                self._prepare()
        except AvantesError:
            undo()
            self._invalidate()
            raise

    def _probe_min_integration(self) -> float:
        """The shortest exposure THIS sensor accepts, found by asking it.

        MIN_INTEGRATION_MS is the LIBRARY's bound and is nowhere near the
        sensor's: the CMOS unit here takes 9 us, while the header's ILX CCD
        constant is 1.1 ms — two orders of magnitude apart, and nothing in the
        device config reports which applies. AVS_PrepareMeasure is the only
        oracle, so bisect against it.

        Costs a dozen PrepareMeasure calls once per connect, and buys an
        exposure range that cannot ask for something the device will refuse —
        which matters because that refusal otherwise surfaces one acquire()
        later, on the caller's acquisition thread.
        """
        keep = float(self.config.m_IntegrationTime)

        def accepts(ms: float) -> bool:
            self.config.m_IntegrationTime = float(ms)
            self._prepared = False
            try:
                self._call("AVS_PrepareMeasure", byref(self.config))
            except AvantesError as e:
                if e.name == "ERR_INVALID_INT_TIME":
                    return False
                raise                       # not a range question — give up
            return True

        try:
            lo, hi = MIN_INTEGRATION_MS, 2.0   # 2 ms: above every documented min
            if accepts(lo):
                return lo
            if not accepts(hi):
                return MIN_INTEGRATION_MS      # unexpected; the write-time
            while hi - lo > 1e-4:              # check still guards the device
                mid = (lo + hi) / 2
                if accepts(mid):
                    hi = mid
                else:
                    lo = mid
            # Snap to whole microseconds. Bisection converges from above, so
            # the raw answer is a hair over the real limit (9.012 us for a 9 us
            # sensor) — an ugly number to show, and one that needlessly refuses
            # the round value the datasheet quotes. Tested, not assumed: if the
            # rounded-down value is refused, keep the one known to work.
            snapped = math.floor(hi * 1000.0) / 1000.0
            return snapped if snapped >= MIN_INTEGRATION_MS and accepts(snapped) \
                   else hi
        except (AvantesError, RuntimeError):
            # A probe is a nicety; never let it fail a connect.
            return MIN_INTEGRATION_MS
        finally:
            self.config.m_IntegrationTime = keep
            self._prepared = False

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def wavelengths(self) -> np.ndarray:
        """Fixed wavelength axis in nm, one entry per pixel."""
        return self._wavelengths

    @property
    def model(self) -> str:
        """Best available human label. See display_name; the detector type in
        device_info() carries the detail this cannot."""
        return display_name(self.user_name, self.serial)

    def versions(self) -> dict[str, str]:
        """FPGA / firmware / DLL version strings."""
        bufs = [ctypes.create_string_buffer(VERSION_LEN + 1) for _ in range(3)]
        self._call("AVS_GetVersionInfo", *bufs)
        keys = ("fpga", "firmware", "dll")
        return {k: b.value.decode("latin-1", "replace").strip()
                for k, b in zip(keys, bufs)}

    def device_type(self) -> str:
        """'AS7010', 'AS5216', … — the electronics board, which is what gates
        several features (high-resolution ADC, heartbeat, ethernet)."""
        t = c_ubyte(0)
        try:
            self._call("AVS_GetDeviceType", byref(t))
        except (AvantesError, RuntimeError):
            return "unknown"
        return DEVICE_TYPES.get(int(t.value), f"unknown ({t.value})")

    @property
    def sensor_type(self) -> int | None:
        """The detector's SensorType code, or None when the configuration
        block could not be read or did not pass its layout check. Several calls
        are documented as sensor-specific, and this is what gates them."""
        if not self.config_verified:
            return None
        return int(self._config_head().m_Detector.m_SensorType)

    def detector_name(self) -> str:
        """Name of the detector chip. Asks the DLL first (it knows names this
        module's table may predate) and falls back to the table."""
        sensor = self.sensor_type
        if sensor is None:
            return ""
        buf = ctypes.create_string_buffer(DETECTOR_NAME_LEN + 1)
        try:
            self._call("AVS_GetDetectorName", c_ubyte(sensor), buf)
            name = buf.value.decode("latin-1", "replace").strip()
        except (AvantesError, RuntimeError):
            name = ""
        return name or SENSOR_TYPES.get(sensor, f"sensor 0x{sensor:02X}")

    # ── acquisition settings ────────────────────────────────────────────────
    @property
    def integration_ms(self) -> float:
        return float(self.config.m_IntegrationTime)

    @integration_ms.setter
    def integration_ms(self, ms: float) -> None:
        self.set_integration_time(ms)

    def set_integration_time(self, ms: float) -> None:
        """Exposure per scan, in milliseconds.

        Checked twice, because a wrong exposure is otherwise invisible until
        it breaks something far away. First against `min_integration_ms` —
        probed from this sensor at connect, not the library's much wider bound
        — so an obvious slip (0, a negative, a seconds-for-milliseconds
        mix-up) is named here. Then by pushing it to the device: the sensor
        may refuse a value the probe suggested was fine (heavy averaging
        narrows the range), and it must refuse it NOW rather than at the next
        acquire() on somebody else's thread. A refused value leaves the
        device on the exposure it already had.
        """
        ms = float(ms)
        if not self.min_integration_ms <= ms <= self.max_integration_ms:
            raise ValueError(
                f"integration time must be between {self.min_integration_ms:g} "
                f"and {self.max_integration_ms:g} ms for this sensor, got "
                f"{ms:g}.")
        prev = float(self.config.m_IntegrationTime)
        self.config.m_IntegrationTime = ms
        try:
            self._push_config(
                lambda: setattr(self.config, "m_IntegrationTime", prev))
        except AvantesError as e:
            if e.name == "ERR_INVALID_INT_TIME":
                # A value problem, not a device fault, so say so as one — the
                # bare -11 is what this whole path exists to avoid.
                raise ValueError(
                    f"The sensor refused an exposure of {ms:g} ms. It accepts "
                    f"{self.min_integration_ms:g} to "
                    f"{self.max_integration_ms:g} ms, and can be narrower at "
                    f"{self.n_averages} on-board averages. The exposure is "
                    f"unchanged at {prev:g} ms.") from None
            raise

    @property
    def n_averages(self) -> int:
        return int(self.config.m_NrAverages)

    @n_averages.setter
    def n_averages(self, n: int) -> None:
        self.set_averages(n)

    def set_averages(self, n: int) -> None:
        """On-board averaging. acquire() then blocks for roughly n × the
        integration time — see `frame_time_ms`."""
        n = int(n)
        if n < 1:
            raise ValueError(f"averages must be >= 1, got {n}")
        prev = int(self.config.m_NrAverages)
        self.config.m_NrAverages = n
        try:
            self._push_config(
                lambda: setattr(self.config, "m_NrAverages", prev))
        except AvantesError as e:
            # The device refuses some exposure/averaging combinations outright
            # (the manual's example is 600 s at more than 5000 averages), and
            # it is the pairing that is wrong, not either number alone.
            if e.name in ("ERR_INVALID_COMBINATION", "ERR_INVALID_INT_TIME",
                          "ERR_INVALID_MEASPARAM_AVG_SAT2"):
                raise ValueError(
                    f"The sensor refused {n} on-board averages at an exposure "
                    f"of {self.integration_ms:g} ms. Averaging is unchanged at "
                    f"{prev}.") from None
            raise

    @property
    def frame_time_ms(self) -> float:
        """How long one acquire() actually takes, near enough for pacing: the
        exposure times the on-board average count. Callers that pace a live
        feed off an exposure time need THIS, not integration_ms, or they spin
        faster than the device produces frames."""
        return self.integration_ms * max(1, self.n_averages)

    def set_pixel_range(self, start: int, stop: int) -> None:
        """Restrict readout to pixels [start, stop]. Shortens the transfer;
        the wavelength axis is unchanged, so callers must slice to match."""
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop < self.n_pixels:
            raise ValueError(
                f"pixel range must satisfy 0 <= start <= stop < {self.n_pixels}, "
                f"got {start}..{stop}")
        self.config.m_StartPixel = start
        self.config.m_StopPixel = stop
        self._invalidate()

    def set_dark_correction(self, enable: bool, forget_percentage: int = 100) -> None:
        """Dynamic dark correction: the device subtracts its own masked-pixel
        reading from every scan. Not supported on every sensor — the DLL
        answers ERR_INVALID_MEASPARAM_DYNDARK (-116) when it is not."""
        self.config.m_CorDynDark.m_Enable = 1 if enable else 0
        self.config.m_CorDynDark.m_ForgetPercentage = int(forget_percentage)
        self._invalidate()

    def set_smoothing(self, pixels: int, model: int = 0) -> None:
        """Boxcar smoothing over ±`pixels` neighbours, done on the device.
        0 disables it. Smoothing is a filter over real data — it does not make
        a noisy spectrum more accurate, and it widens narrow lines."""
        pixels = int(pixels)
        if not 0 <= pixels <= MAX_SMOOTH_PIX:
            raise ValueError(
                f"smoothing width must be 0..{MAX_SMOOTH_PIX}, got {pixels}")
        self.config.m_Smoothing.m_SmoothPix = pixels
        self.config.m_Smoothing.m_SmoothModel = int(model)
        self._invalidate()

    def set_saturation_detection(self, level: int) -> None:
        """0 off, 1 detect, 2 detect + correct inverted pixels.

        Level 2 is ILX554-only and additionally cannot be combined with
        averaging (-110) or store-to-RAM (-114). It is refused here rather than
        at the next PrepareMeasure, so the reason is legible.
        """
        level = int(level)
        if level not in (0, 1, 2):
            raise ValueError(f"saturation detection level must be 0, 1 or 2, got {level}")
        if level == 2:
            sensor = self.sensor_type
            if sensor is not None and sensor not in SAT_DETECT_LEVEL2_SENSORS:
                raise ValueError(
                    f"saturation detection level 2 is only supported on the "
                    f"ILX554; this device has a "
                    f"{SENSOR_TYPES.get(sensor, hex(sensor))} detector. Use "
                    f"level 1.")
            if self.n_averages > 1:
                raise ValueError(
                    "saturation detection level 2 cannot be combined with "
                    "averaging — set averages to 1 first.")
        self.config.m_SaturationDetection = level
        self._invalidate()

    def set_trigger(self, mode: int = SW_TRIGGER_MODE,
                    source: int = EXTERNAL_TRIGGER,
                    source_type: int = EDGE_TRIGGER_SOURCE) -> None:
        """Configure triggering.

        WARNING: with mode = HW_TRIGGER_MODE the device does not start a scan
        until a pulse arrives, so acquire() blocks for as long as that takes.
        Give acquire() a timeout, or stop any free-running feed first.
        """
        if mode not in (SW_TRIGGER_MODE, HW_TRIGGER_MODE, SS_TRIGGER_MODE):
            raise ValueError(f"unknown trigger mode {mode}")
        self.config.m_Trigger.m_Mode = int(mode)
        self.config.m_Trigger.m_Source = int(source)
        self.config.m_Trigger.m_SourceType = int(source_type)
        self._invalidate()

    @property
    def hardware_triggered(self) -> bool:
        """True when acquire() is waiting on an external event rather than on
        the device's own clock — i.e. when it may block indefinitely."""
        return int(self.config.m_Trigger.m_Mode) == HW_TRIGGER_MODE

    def set_sync_mode(self, enable: bool) -> None:
        """Synchronise multiple channels/spectrometers off one trigger."""
        self._call("AVS_SetSyncMode", c_ubyte(1 if enable else 0))

    def set_prescan_mode(self, enable: bool) -> None:
        """Discard the first (charge-accumulating) scan of a sequence rather
        than reporting it. Costs one frame time per measurement."""
        self._call("AVS_SetPrescanMode", c_bool(bool(enable)))

    # ── ADC resolution ──────────────────────────────────────────────────────
    @property
    def high_res_adc(self) -> bool:
        return self._high_res

    @property
    def max_counts(self) -> float:
        """Full scale in counts for the ADC mode currently in force. This is
        the number saturation must be judged against; it CHANGES when the
        resolution is switched, so anything caching it has to be told."""
        return ADC_FULL_SCALE_16BIT if self._high_res else ADC_FULL_SCALE_14BIT

    def set_high_res_adc(self, enable: bool) -> bool:
        """Switch the 16-bit ADC mode on or off. Returns whether it is on.

        On hardware with only a 14-bit ADC the DLL answers
        ERR_OPERATION_NOT_SUPPORTED (-2); that is a fact about the model, not a
        failure, so it is reported by the return value rather than raised.
        Note the counts scale by ADC_HIGH_RES_FACTOR between the two modes.
        """
        try:
            self._call("AVS_UseHighResAdc", c_bool(bool(enable)))
        except AvantesError as e:
            if e.name in ("ERR_OPERATION_NOT_SUPPORTED",
                          "ERR_NOT_SUPPORTED_BY_SENSOR_TYPE",
                          "ERR_NOT_SUPPORTED_BY_FW_VER",
                          "ERR_NOT_SUPPORTED_BY_FPGA_VER"):
                self._high_res = False
                return False
            raise
        self._high_res = bool(enable)
        return self._high_res

    def set_sensitivity_mode(self, mode: int) -> bool:
        """High-sensitivity mode. Returns whether it was applied.

        Documented as supported only by the HAMS9201, HAMG9208-512, SU256LSB
        and SU512LDB detectors. On anything else this is a fact about the
        model, not a failure, so it returns False — and when the detector type
        is known the call is skipped entirely rather than made and swallowed.
        """
        sensor = self.sensor_type
        if sensor is not None and sensor not in SENSITIVITY_MODE_SENSORS:
            return False
        try:
            self._call("AVS_SetSensitivityMode", c_uint32(int(mode)))
        except AvantesError as e:
            if e.name in ("ERR_NOT_SUPPORTED_BY_SENSOR_TYPE",
                          "ERR_OPERATION_NOT_SUPPORTED",
                          "ERR_NOT_SUPPORTED_BY_FW_VER"):
                return False
            raise
        return True

    # ── measurement ─────────────────────────────────────────────────────────
    def acquire(self, timeout_s: float | None = None) -> np.ndarray:
        """One spectrum, in raw counts. Blocks until the scan is complete.

        Polling, not the DLL's completion callback: `avaspec.h` declares the
        callback with no calling convention, the published wrappers disagree
        about whether it is __cdecl or __stdcall, and a mismatch corrupts the
        stack instead of raising. Polling costs a few hundred microseconds of
        latency and cannot go wrong.

        `timeout_s` defaults to a generous multiple of the frame time. Under a
        hardware trigger there is no meaningful default — the scan starts when
        the experiment says so — so pass one explicitly if the trigger might
        not fire, or acquire() waits forever.

        Atomic per device: concurrent callers queue rather than interleave (see
        _measure_lock). A caller blocked here is waiting out at most one frame,
        plus whatever the caller ahead of it was waiting for.
        """
        with self._measure_lock:
            self._prepare()
            # nmsr = 1: one scan, then the measurement ends by itself. The
            # callback argument is always NULL (see the note above).
            self._call("AVS_Measure", None, c_short(1))
            self._poll(timeout_s)
            tick = c_uint32(0)
            self._call("AVS_GetScopeData", byref(tick), self._scope_buf)
            self.last_tick = int(tick.value)
            return np.frombuffer(self._scope_buf, dtype=np.float64,
                                 count=self.n_pixels).copy()

    def _poll(self, timeout_s: float | None) -> None:
        if timeout_s is None and not self.hardware_triggered:
            # Three frame times plus two seconds of slack: enough that a slow
            # USB round-trip or a prescan never trips it, short enough that a
            # wedged device is reported rather than hanging the caller.
            timeout_s = 3.0 * self.frame_time_ms / 1000.0 + 2.0
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        # Poll fast enough not to add latency to a short exposure, slow enough
        # not to spin a core through a 40 s one.
        interval = min(max(self.frame_time_ms / 50_000.0, 0.0005), 0.02)
        while True:
            if self._call("AVS_PollScan") == 1:
                return
            if deadline is not None and time.monotonic() > deadline:
                self.stop()
                n = self.n_averages
                raise TimeoutError(
                    f"Avantes {self.serial}: no scan after {timeout_s:.1f} s "
                    f"(integration {self.integration_ms:g} ms × {n} "
                    f"{'average' if n == 1 else 'averages'}"
                    + (", waiting on a hardware trigger"
                       if self.hardware_triggered else "") + ").")
            time.sleep(interval)

    def saturated_pixels(self) -> np.ndarray:
        """Boolean mask of pixels the device flagged as clipped in the last
        scan. Only populated when saturation detection is enabled.

        Under the measurement lock: this reads the LAST measurement's result,
        so it has to be paired with the acquire it belongs to, not with
        whatever frame another thread started in between.
        """
        with self._measure_lock:
            self._call("AVS_GetSaturatedPixels", self._sat_buf)
            return np.frombuffer(self._sat_buf, dtype=np.uint8,
                                 count=self.n_pixels).astype(bool)

    def stop(self) -> None:
        """Abort any measurement in progress. Safe to call when none is."""
        try:
            self._call("AVS_StopMeasure")
        except (AvantesError, RuntimeError):
            pass
        self._invalidate()

    # ── device configuration (EEPROM) ───────────────────────────────────────
    def _read_config(self) -> None:
        """Fetch the DeviceConfigType blob and sanity-check our view of it.

        The size is asked of the DLL rather than assumed (see
        DeviceConfigHeadType), then the result is cross-checked against values
        obtained independently. Everything reading EEPROM-derived fields is
        gated on `config_verified`.
        """
        self.config_verified = False
        self._config_raw = None
        required = c_uint32(0)
        dummy = (c_ubyte * 1)()
        with _LOCK:
            self._lib.AVS_GetParameter(AvsHandle(self._h), 0, byref(required),
                                       ctypes.cast(dummy, c_void_p))
        size = int(required.value)
        if size <= 0 or size > 1 << 20:
            return                              # nothing trustworthy to read
        buf = (c_ubyte * size)()
        try:
            self._call("AVS_GetParameter", c_uint32(size), byref(required),
                       ctypes.cast(buf, c_void_p))
        except (AvantesError, RuntimeError):
            return
        self._config_raw = buf
        if size >= ctypes.sizeof(DeviceConfigHeadType):
            head = self._config_head()
            det = head.m_Detector
            # Three cross-checks against values obtained by other means. The
            # last is the strongest: m_aFit is the wavelength polynomial, so
            # its constant term IS the wavelength of pixel 0, which
            # AVS_GetLambda reported independently. Agreement there means every
            # field offset from the start of the struct through the middle of
            # DetectorType is right — a coincidence is not plausible.
            self.config_verified = (
                int(head.m_Len) == size
                and int(det.m_NrPixels) == self.n_pixels
                and abs(float(det.m_aFit[0]) - float(self._wavelengths[0])) < 1.0)

    def _config_head(self) -> DeviceConfigHeadType:
        if self._config_raw is None:
            raise RuntimeError("Device configuration has not been read.")
        return ctypes.cast(self._config_raw,
                           POINTER(DeviceConfigHeadType)).contents

    def device_info(self) -> dict:
        """Everything worth showing about the device, best-effort.

        Keys are always present; values fall back to None/"" when the device or
        the DLL will not answer, so a caller can render this without guarding
        every field.
        """
        info = {
            "serial": self.serial,
            "name": self.user_name,
            "model": self.model,
            "board": self.device_type(),
            "n_pixels": self.n_pixels,
            "wavelength_range_nm": (float(self._wavelengths[0]),
                                    float(self._wavelengths[-1])),
            "high_res_adc": self._high_res,
            "max_counts": self.max_counts,
            "detector": None,
            "config_version": None,
            "config_verified": self.config_verified,
        }
        info.update(self.versions() if self._h is not None else {})
        if self._config_raw is not None:
            info["config_version"] = int(self._config_head().m_ConfigVersion)
        if self.config_verified:
            info["detector"] = self.detector_name()
        return info

    def temperature_c(self) -> float | None:
        """Board temperature in °C, or None when the device has no such sensor.

        Uses the AS7010's digital sensor (analog input 6), which reports
        degrees directly. The alternative — thermistor volts on input 0 — would
        need the device's own conversion polynomial, which lives in the part of
        DeviceConfigType this module does not fully trust.
        """
        val = c_float(0.0)
        try:
            self._call("AVS_GetAnalogIn", c_ubyte(ANALOG_IN_BOARD_TEMP_C),
                       byref(val))
        except (AvantesError, RuntimeError):
            return None
        return float(val.value)

    def analog_in(self, channel: int) -> float:
        """Read an analog input in volts. See the ANALOG_IN_* ids."""
        val = c_float(0.0)
        self._call("AVS_GetAnalogIn", c_ubyte(int(channel)), byref(val))
        return float(val.value)

    def analog_out(self, port: int, value: float) -> None:
        if not 0 <= int(port) < NR_ANALOG_OUTPUTS:
            raise ValueError(f"analog output must be 0..{NR_ANALOG_OUTPUTS - 1}")
        self._call("AVS_SetAnalogOut", c_ubyte(int(port)), c_float(float(value)))

    def digital_in(self, port: int) -> bool:
        if not 0 <= int(port) < NR_DIGITAL_INPUTS:
            raise ValueError(f"digital input must be 0..{NR_DIGITAL_INPUTS - 1}")
        val = c_ubyte(0)
        self._call("AVS_GetDigIn", c_ubyte(int(port)), byref(val))
        return bool(val.value)

    def digital_out(self, port: int, state: bool) -> None:
        if not 0 <= int(port) < NR_DIGITAL_OUTPUTS:
            raise ValueError(f"digital output must be 0..{NR_DIGITAL_OUTPUTS - 1}")
        self._call("AVS_SetDigOut", c_ubyte(int(port)), c_ubyte(1 if state else 0))

    # ── lifecycle ───────────────────────────────────────────────────────────
    def disconnect(self) -> None:
        """Close the device. Idempotent, and never raises — it runs from
        __init__'s failure path and from consumers' cleanup, neither of which
        can do anything useful with an error here."""
        handle, self._h = self._h, None
        # Only drop the registry entry if it is still ours: a failed __init__
        # that never got as far as registering must not evict a live sibling.
        with _LOCK:
            if _open_serials.get(getattr(self, "serial", None)) is self:
                del _open_serials[self.serial]
        if handle is not None:
            try:
                with _LOCK:
                    self._lib.AVS_StopMeasure(AvsHandle(handle))
            except Exception:
                pass
            try:
                with _LOCK:
                    self._lib.AVS_Deactivate(AvsHandle(handle))
            except Exception:
                pass
        if self._holds_library:
            self._holds_library = False
            _release_library()

    def __enter__(self) -> "AvaSpec":
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        state = "disconnected" if self._h is None else f"handle {self._h}"
        return f"<AvaSpec {self.model} [{getattr(self, 'serial', '?')}] {state}>"


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == "__main__":
    print("avantes.py self-test")

    # 1. Struct layout. This is the part that MUST hold, runs without a DLL,
    #    and is the only thing standing between a packing slip and a silently
    #    corrupt measurement config.
    _assert_struct_sizes()
    for group, sizes in (("documented", _DOCUMENTED_SIZES),
                         ("derived", _DERIVED_SIZES)):
        for t, n in sorted(sizes.items(), key=lambda kv: kv[0].__name__):
            print(f"  {t.__name__:<28s} {ctypes.sizeof(t):6d} bytes  "
                  f"({group}: {n})")
    print(f"  {'DeviceConfigHeadType':<28s} {ctypes.sizeof(DeviceConfigHeadType):6d} "
          f"bytes  (full struct size asked of the DLL at runtime)")

    # 2. Error table: every code must map to a name and a sentence.
    assert ERROR_CODES[0][0] == "ERR_SUCCESS"
    for code, (name, text) in ERROR_CODES.items():
        assert name.startswith("ERR_") and text.endswith((".", "success")), (code, name)
    assert (code := AvantesError(-24, "AVS_Activate")).code == -24
    assert code.name == "ERR_ACCESS" and "rights" in str(code)
    assert _check(3, "AVS_Init") == 3          # positive returns pass through
    try:
        _check(-11, "AVS_PrepareMeasure")
    except AvantesError as e:
        assert e.name == "ERR_INVALID_INT_TIME", e.name
    else:
        raise AssertionError("_check must raise on a negative return")
    print(f"  error table: {len(ERROR_CODES)} codes, all named")

    # 3. The DLL. Absent, this must produce an actionable sentence rather than
    #    a bare OSError — that message is what an operator gets in the UI.
    try:
        load()
    except RuntimeError as e:
        print(f"\n  no usable DLL:\n    {e}")
        print("\n  Struct and error checks passed; connect checks need the DLL.")
        raise SystemExit(0)

    print(f"\n  DLL: {dll_path()}  (version {dll_version()})")
    devices = list_devices()
    if not devices:
        print("  No spectrometers attached — nothing further to test.")
        raise SystemExit(0)
    for name, serial, status in devices:
        print(f"  found: {name} [{serial}] — {status}")

    with AvaSpec() as spec:
        info = spec.device_info()
        print(f"\n  {info['model']} [{info['serial']}] on {info['board']}")
        print(f"  {info['n_pixels']} pixels, "
              f"{info['wavelength_range_nm'][0]:.1f}–"
              f"{info['wavelength_range_nm'][1]:.1f} nm")
        print(f"  firmware {info.get('firmware', '?')}, FPGA {info.get('fpga', '?')}")
        print(f"  high-res ADC: {info['high_res_adc']} -> full scale "
              f"{info['max_counts']:.0f} counts")
        print(f"  device config layout verified: {info['config_verified']}")
        t = spec.temperature_c()
        print(f"  board temperature: {'n/a' if t is None else f'{t:.1f} °C'}")

        spec.set_integration_time(10.0)
        t0 = time.monotonic()
        counts = spec.acquire()
        dt = (time.monotonic() - t0) * 1000.0
        print(f"\n  acquire: {counts.size} pixels in {dt:.1f} ms "
              f"(exposure {spec.integration_ms:g} ms), "
              f"min {counts.min():.0f} / peak {counts.max():.0f} counts")
        n_sat = int(spec.saturated_pixels().sum())
        print(f"  saturated pixels: {n_sat}")
        assert counts.size == spec.n_pixels
        assert spec.wavelengths.size == spec.n_pixels

    print("\n  OK")
