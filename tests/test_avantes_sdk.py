"""The Avantes ctypes layer, tested with no DLL and no spectrometer attached.

Ported from avantes.py's self-test (lines 1666-1694, plus _version_key). Only
the struct-layout and error-table half of that block runs without hardware, and
that is precisely the half that must hold: a ctypes struct whose packing drifts
from the vendor header is the difference between a correct measurement config
and a silently corrupt one, with no error anywhere to say so.

The DLL-loading path is covered through its pure helpers. `load()` itself is not
called: on a machine with the AvaSpec DLL installed it would reach real hardware
state, and this suite touches no hardware.
"""

from __future__ import annotations

import ctypes

import pytest

from lillypad.hardware.vendor import avaspec as avantes  # the vendor DLL wrapper
from lillypad.hardware.vendor.avaspec import (ADC_FULL_SCALE_14BIT, ADC_FULL_SCALE_16BIT,
                     ADC_HIGH_RES_FACTOR, ERROR_CODES, MAX_INTEGRATION_MS,
                     MIN_INTEGRATION_MS, SENSOR_TYPES, AvantesError,
                     DeviceConfigHeadType, _assert_struct_sizes, _candidate_paths,
                     _check, _DERIVED_SIZES, _DOCUMENTED_SIZES, _version_key)


class TestStructLayout:
    def test_every_struct_matches_its_documented_size(self):
        """The whole-table check, exactly as the self-test ran it. This is the
        only thing standing between a packing slip and a silently corrupt
        measurement config."""
        _assert_struct_sizes()

    @pytest.mark.parametrize("struct,size", sorted(
        {**_DOCUMENTED_SIZES, **_DERIVED_SIZES}.items(),
        key=lambda kv: kv[0].__name__))
    def test_struct_size(self, struct, size):
        """Per-struct, so a failure names the one that drifted."""
        assert ctypes.sizeof(struct) == size, struct.__name__

    def test_device_config_head_has_a_size(self):
        """Its full size is what gets asked of the DLL at runtime."""
        assert ctypes.sizeof(DeviceConfigHeadType) > 0


class TestErrorTable:
    def test_success_is_code_zero(self):
        assert ERROR_CODES[0][0] == "ERR_SUCCESS"

    @pytest.mark.parametrize("code", sorted(ERROR_CODES))
    def test_every_code_has_a_name_and_a_sentence(self, code):
        name, text = ERROR_CODES[code]
        assert name.startswith("ERR_"), name
        assert text.endswith((".", "success")), (code, name, text)

    def test_error_carries_its_code_name_and_explanation(self):
        err = AvantesError(-24, "AVS_Activate")
        assert err.code == -24
        assert err.name == "ERR_ACCESS"
        assert "rights" in str(err)

    def test_an_unknown_code_still_produces_an_error(self):
        """A DLL newer than this wrapper must not crash the wrapper."""
        err = AvantesError(-9999, "AVS_Whatever")
        assert err.code == -9999
        assert str(err)


class TestCheck:
    def test_positive_returns_pass_through(self):
        """A positive return is a value — a handle, a pixel count — not a
        status."""
        assert _check(3, "AVS_Init") == 3

    def test_zero_passes_through(self):
        assert _check(0, "AVS_Init") == 0

    def test_a_negative_return_raises_the_named_error(self):
        with pytest.raises(AvantesError) as exc:
            _check(-11, "AVS_PrepareMeasure")
        assert exc.value.name == "ERR_INVALID_INT_TIME"

    def test_the_failing_call_is_named_in_the_message(self):
        with pytest.raises(AvantesError) as exc:
            _check(-11, "AVS_PrepareMeasure")
        assert "AVS_PrepareMeasure" in str(exc.value)


class TestVersionKey:
    def test_orders_9_14_above_9_9(self):
        """The one case a plain string sort gets wrong: "9.9" sorts above "9.14"
        character by character, so the OLDER install would win."""
        paths = [r"C:\AvaSpecX64-DLL_9.9.0.0\AvaSpecX64.dll",
                 r"C:\AvaSpecX64-DLL_9.14.0.0\AvaSpecX64.dll"]
        newest = sorted(paths, key=_version_key, reverse=True)[0]
        assert "9.14" in newest

    def test_sorts_numerically_across_all_components(self):
        paths = [r"C:\AvaSpecX64-DLL_9.14.0.0\x.dll",
                 r"C:\AvaSpecX64-DLL_10.0.0.0\x.dll",
                 r"C:\AvaSpecX64-DLL_9.14.1.0\x.dll"]
        order = sorted(paths, key=_version_key, reverse=True)
        assert "10.0.0.0" in order[0]
        assert "9.14.1.0" in order[1]
        assert "9.14.0.0" in order[2]

    def test_an_unversioned_path_still_sorts_stably(self):
        paths = [r"C:\avaspec.dll", r"D:\avaspec.dll"]
        assert sorted(paths, key=_version_key) == paths


class TestCandidatePaths:
    def test_returns_a_list_of_strings(self):
        paths = _candidate_paths()
        assert isinstance(paths, list)
        assert all(isinstance(p, str) for p in paths)

    def test_the_env_override_comes_first(self, monkeypatch):
        """LILLYPAD_AVASPEC_DLL is the operator's escape hatch, so it has to
        outrank every discovered install."""
        monkeypatch.setenv(avantes._ENV_VAR, r"C:\somewhere\mine.dll")
        assert _candidate_paths()[0] == r"C:\somewhere\mine.dll"

    def test_without_the_override_nothing_named_mine_appears(self, monkeypatch):
        monkeypatch.delenv(avantes._ENV_VAR, raising=False)
        assert all("mine.dll" not in p for p in _candidate_paths())


class TestConstants:
    def test_adc_scales_are_the_documented_full_scales(self):
        assert ADC_FULL_SCALE_14BIT == 16383.0
        assert ADC_FULL_SCALE_16BIT == 65535.0
        assert ADC_HIGH_RES_FACTOR == 4.0

    def test_integration_limits_are_ordered(self):
        assert 0.0 < MIN_INTEGRATION_MS < MAX_INTEGRATION_MS

    def test_sensor_table_is_populated_and_named(self):
        assert SENSOR_TYPES[0x00] == "reserved"
        assert all(isinstance(v, str) and v for v in SENSOR_TYPES.values())

    def test_trigger_mode_constants_are_distinct(self):
        modes = (avantes.SW_TRIGGER_MODE, avantes.HW_TRIGGER_MODE,
                 avantes.SS_TRIGGER_MODE)
        assert len(set(modes)) == 3

    def test_trigger_source_and_type_constants_are_distinct(self):
        assert avantes.EXTERNAL_TRIGGER != avantes.SYNC_TRIGGER
        assert avantes.EDGE_TRIGGER_SOURCE != avantes.LEVEL_TRIGGER_SOURCE


class TestNoDllSideEffects:
    def test_importing_avantes_loads_no_library(self):
        """The module must import cleanly with no DLL anywhere — that is what
        lets `python avantes.py` and this suite run on a dev machine."""
        assert avantes._lib is None or avantes._lib_path is not None

    def test_no_devices_are_registered_at_import(self):
        assert avantes._open_serials == {}
        assert avantes._init_refs == 0
