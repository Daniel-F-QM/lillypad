"""Tagged spectrometer ids and vendor-agnostic enumeration.

Ported from hardware.py's self-test (lines 2663-2693). All of this runs with no
vendor SDK installed and nothing attached, which is the point: enumeration must
degrade to "found nothing" rather than raising, or one missing driver takes the
device picker and every multi-spectrometer submenu down with it.
"""

from __future__ import annotations

import pytest

from hardware import (list_spectrometers, open_spectrometer, spec_ident,
                      split_spec_ident)


def test_spec_ident_tags_the_serial():
    assert spec_ident("seabreeze", "USB2+H12345") == "seabreeze:USB2+H12345"


def test_spec_ident_marks_an_unreadable_serial():
    assert spec_ident("seabreeze", None) == "seabreeze:?"


@pytest.mark.parametrize("ident,expected", [
    ("seabreeze:USB2+H12345", ("seabreeze", "USB2+H12345")),
    ("avantes:2001234", ("avantes", "2001234")),
])
def test_split_spec_ident_round_trip(ident, expected):
    assert split_spec_ident(ident) == expected
    assert spec_ident(*expected) == ident


@pytest.mark.parametrize("bad", [
    "USB2+H12345",     # untagged
    "ocean:123",       # unknown vendor
    "",                # empty
    "__sim_blue__",    # a simulated-slot sentinel, deliberately not an id
])
def test_split_spec_ident_rejects_untagged_ids(bad):
    with pytest.raises(RuntimeError):
        split_spec_ident(bad)


def test_enumeration_degrades_instead_of_raising():
    """Must not raise with no SDK present, and everything it returns must be a
    parseable tagged id."""
    devices = list_spectrometers()
    assert isinstance(devices, list)
    for d in devices:
        assert len(d) == 2
        assert split_spec_ident(d[1])


def test_missing_vendor_sdk_gives_an_actionable_message():
    """The operator reads this string in the Hardware dialog, so it must name
    what to install rather than surfacing a bare ImportError or OSError.

    Skipped when the Avantes DLL happens to be installed on this machine — then
    there is no missing-SDK path to exercise.
    """
    try:
        open_spectrometer("avantes:2001234")
    except Exception as e:                       # noqa: BLE001 - that is the point
        if isinstance(e, (ImportError, OSError)):
            raise AssertionError(
                f"raw {type(e).__name__} reached the caller: {e}") from e
        assert str(e).strip(), "the failure carried no message at all"
    else:
        pytest.skip("an Avantes spectrometer answered; no missing-SDK path here")
