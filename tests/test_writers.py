"""The three export formats: .dwc, .npz and .csv.

New tests — the self-tests never exercised the writers at all, which is a gap
worth closing before the files move: .dwc is a fixed third-party layout
(Femtosoft), so a formatting regression produces a file that loads into someone
else's software and is quietly wrong rather than one that fails to load.

The ordering asymmetry is the subtle part, and it is deliberate in the source:
.dwc and .csv flip a high-to-low scan to ascending (because .dwc records only a
positive increment), while .npz keeps acquisition order. Both are pinned here.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from lillypad.scan import (FrogResult, FrogScanConfig, write_csv, write_dwc,
                           write_npz)

N_WL, N_DELAY = 6, 5


def make_result(descending=False, dark=True, reference=False, **kw):
    """A small, fully-populated FrogResult with recognisable values.

    `descending` models the SAME physical measurement taken high-to-low, so the
    delay axis, the trace columns and both position vectors all reverse
    together — that is what acquisition order means. The exported .dwc and .csv
    must then come out identical to the ascending scan's.
    """
    delays = np.linspace(-20.0, 20.0, N_DELAY)
    wl = np.linspace(500.0, 550.0, N_WL)
    # Distinct per-cell values so a transpose or a flip cannot pass unnoticed.
    trace = (np.arange(N_WL)[:, None] * 100.0
             + np.arange(N_DELAY)[None, :] * 10.0 + 1000.0)
    cmd = 150000.0 + delays          # varies with delay, so a flip is visible
    readback = cmd + 0.5
    if descending:
        delays, cmd, readback = delays[::-1], cmd[::-1], readback[::-1]
        trace = trace[:, ::-1]
    cfg = FrogScanConfig(delay_start_fs=float(delays[0]),
                         delay_stop_fs=float(delays[-1]),
                         delay_step_fs=10.0, zero_pos_um=150000.0)
    the_dark = np.full(N_WL, 50.0) if dark else None
    ref = np.full(N_WL, 70.0) if reference else None
    defaults = dict(
        delays_fs=delays,
        positions_cmd_um=cmd,
        positions_readback_um=readback,
        wavelengths_nm=wl,
        trace=trace,
        dark=the_dark,
        delay_reference_before=ref,
        delay_reference_after=ref,
        config=cfg,
        stage_name="SimulatedStage",
        stage_backlash_um=8.0,
    )
    defaults.update(kw)
    return FrogResult(**defaults)


class TestResultModel:
    def test_corrected_trace_subtracts_the_dark_per_wavelength(self):
        r = make_result()
        assert np.allclose(r.corrected_trace(), np.asarray(r.trace) - 50.0)

    def test_corrected_trace_is_the_raw_trace_without_a_dark(self):
        r = make_result(dark=False)
        assert np.allclose(r.corrected_trace(), r.trace)

    def test_dark_subtracted_reports_which_it_was(self):
        """The one thing a reader of a .dwc cannot otherwise tell."""
        assert make_result(dark=True).dark_subtracted is True
        assert make_result(dark=False).dark_subtracted is False

    def test_the_delay_reference_is_never_subtracted(self):
        """Archived for provenance only. A result carrying a reference and no
        dark must still export raw counts."""
        r = make_result(dark=False, reference=True)
        assert r.delay_reference() is not None
        assert np.allclose(r.corrected_trace(), r.trace)

    def test_delay_reference_is_the_mean_of_the_two_frames(self):
        assert np.allclose(make_result(reference=True).delay_reference(), 70.0)

    def test_delay_reference_is_none_when_not_taken(self):
        assert make_result().delay_reference() is None

    def test_saturation_unchecked_is_distinct_from_clean(self):
        """A scan that could not check at all is NOT the same as a clean scan,
        and the two must stay distinguishable once the data is on disk."""
        assert make_result().saturation_checked is False
        assert make_result(saturation_threshold=64000.0).saturation_checked is True

    def test_faulted_columns_finds_the_flagged_ones(self):
        faults = np.array(["", "", "stalled", "", ""], dtype=object)
        r = make_result(stage_faults=faults)
        assert list(r.faulted_columns()) == [2]
        assert r.has_stage_fault()

    def test_a_clean_scan_reports_no_faults(self):
        r = make_result(stage_faults=np.array([""] * N_DELAY, dtype=object))
        assert not r.has_stage_fault()


class TestDwc:
    def test_header_reports_the_trace_shape(self, tmp_path):
        p = tmp_path / "t.dwc"
        write_dwc(p, make_result())
        head = p.read_text(encoding="ascii").splitlines()
        assert head[0] == f"Number of delay points = {N_DELAY}"
        assert head[1] == f"Number of wavelength points = {N_WL}"
        assert head[2].startswith("Delay increment = 10.0000000")

    def test_sections_and_row_count(self, tmp_path):
        p = tmp_path / "t.dwc"
        write_dwc(p, make_result())
        lines = p.read_text(encoding="ascii").splitlines()
        assert "[Wavelength vector]" in lines
        assert "[Data array]" in lines
        data_at = lines.index("[Data array]")
        rows = [ln for ln in lines[data_at + 1:] if ln.strip()]
        assert len(rows) == N_WL
        assert all(len(r.split("\t")) == N_DELAY for r in rows)

    def test_wavelength_vector_round_trips(self, tmp_path):
        p = tmp_path / "t.dwc"
        r = make_result()
        write_dwc(p, r)
        lines = p.read_text(encoding="ascii").splitlines()
        wl = [float(v) for v in lines[lines.index("[Wavelength vector]") + 1].split("\t")]
        assert np.allclose(wl, r.wavelengths_nm, rtol=1e-6)

    def test_data_is_background_subtracted(self, tmp_path):
        p = tmp_path / "t.dwc"
        r = make_result()
        write_dwc(p, r)
        lines = p.read_text(encoding="ascii").splitlines()
        data_at = lines.index("[Data array]")
        rows = [[float(v) for v in ln.split("\t")]
                for ln in lines[data_at + 1:] if ln.strip()]
        assert np.allclose(np.array(rows), r.corrected_trace(), rtol=1e-6)

    def test_labview_exponents_are_not_zero_padded(self, tmp_path):
        """Femtosoft-style "%.6E" with the exponent's padding zero dropped:
        1.843471E+2, not 1.843471E+02."""
        p = tmp_path / "t.dwc"
        write_dwc(p, make_result())
        text = p.read_text(encoding="ascii")
        assert "E+0" not in text
        assert "E-0" not in text

    def test_crlf_line_endings(self, tmp_path):
        p = tmp_path / "t.dwc"
        write_dwc(p, make_result())
        raw = p.read_bytes()
        assert b"\r\n" in raw
        assert b"\n" not in raw.replace(b"\r\n", b"")

    def test_a_descending_scan_is_flipped_to_ascending(self, tmp_path):
        """.dwc records only a positive increment, so the columns have to be
        flipped to stay physically correct."""
        asc, desc = tmp_path / "a.dwc", tmp_path / "d.dwc"
        write_dwc(asc, make_result(descending=False))
        write_dwc(desc, make_result(descending=True))
        assert asc.read_text(encoding="ascii") == desc.read_text(encoding="ascii")

    def test_mismatched_wavelength_vector_is_refused(self, tmp_path):
        r = make_result(wavelengths_nm=np.linspace(500.0, 550.0, N_WL + 3))
        with pytest.raises(ValueError):
            write_dwc(tmp_path / "t.dwc", r)


class TestNpz:
    def test_round_trips_the_raw_trace_and_axes(self, tmp_path):
        p = tmp_path / "t.npz"
        r = make_result()
        write_npz(p, r)
        with np.load(p, allow_pickle=False) as z:
            assert np.array_equal(z["trace"], r.trace)
            assert np.array_equal(z["delays_fs"], r.delays_fs)
            assert np.array_equal(z["wavelengths_nm"], r.wavelengths_nm)
            assert np.array_equal(z["positions_readback_um"],
                                  r.positions_readback_um)

    def test_stores_the_raw_trace_not_the_corrected_one(self, tmp_path):
        """The archive is the only format that can carry the background frames,
        so it keeps the raw counts and the brackets separately."""
        p = tmp_path / "t.npz"
        r = make_result()
        write_npz(p, r)
        with np.load(p, allow_pickle=False) as z:
            assert np.array_equal(z["trace"], r.trace)
            assert not np.allclose(z["trace"], r.corrected_trace())
            assert np.array_equal(z["dark"], r.dark)

    def test_keeps_acquisition_order_for_a_descending_scan(self, tmp_path):
        """Unlike .dwc and .csv. Line up an archive against a .dwc by delay
        value, not by column index."""
        p = tmp_path / "t.npz"
        r = make_result(descending=True)
        write_npz(p, r)
        with np.load(p, allow_pickle=False) as z:
            assert z["delays_fs"][0] > z["delays_fs"][-1]
            assert np.array_equal(z["delays_fs"], r.delays_fs)

    def test_omits_optional_arrays_that_were_never_recorded(self, tmp_path):
        p = tmp_path / "t.npz"
        write_npz(p, make_result(dark=False))
        with np.load(p, allow_pickle=False) as z:
            assert "dark" not in z
            assert "delay_reference_before" not in z
            assert "n_saturated" not in z
            assert "stage_faults" not in z

    def test_carries_the_per_column_records_when_present(self, tmp_path):
        p = tmp_path / "t.npz"
        r = make_result(n_saturated=np.array([0, 0, 7, 0, 0]),
                        stage_faults=np.array(["", "", "stalled", "", ""],
                                              dtype=object))
        write_npz(p, r)
        with np.load(p, allow_pickle=False) as z:
            assert list(z["n_saturated"]) == [0, 0, 7, 0, 0]
            assert z["stage_faults"][2] == "stalled"
            assert z["stage_faults"][0] == ""

    def test_metadata_is_a_readable_mapping(self, tmp_path):
        p = tmp_path / "t.npz"
        r = make_result()
        write_npz(p, r)
        with np.load(p, allow_pickle=False) as z:
            meta = ast.literal_eval(str(z["metadata"]))
        assert meta["n_delays"] == N_DELAY
        assert meta["n_pixels"] == N_WL
        assert meta["stage_name"] == "SimulatedStage"
        assert meta["stage_backlash_um"] == 8.0
        assert meta["dark_subtracted"] is True
        assert meta["has_delay_reference"] is False
        assert meta["saturation_checked"] is False


class TestCsv:
    def test_header_is_the_delay_axis(self, tmp_path):
        p = tmp_path / "t.csv"
        r = make_result()
        write_csv(p, r)
        first = p.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith(r"wavelength_nm\delay_fs,")
        delays = [float(v) for v in first.split(",")[1:]]
        assert np.allclose(delays, r.delays_fs)

    def test_one_row_per_wavelength_led_by_the_wavelength(self, tmp_path):
        p = tmp_path / "t.csv"
        r = make_result()
        write_csv(p, r)
        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines()[1:]
                if ln.strip()]
        assert len(rows) == N_WL
        for wl, row in zip(r.wavelengths_nm, rows):
            fields = row.split(",")
            assert float(fields[0]) == pytest.approx(wl)
            assert len(fields) == N_DELAY + 1

    def test_data_is_background_subtracted(self, tmp_path):
        p = tmp_path / "t.csv"
        r = make_result()
        write_csv(p, r)
        rows = [[float(v) for v in ln.split(",")[1:]]
                for ln in p.read_text(encoding="utf-8").splitlines()[1:]
                if ln.strip()]
        assert np.allclose(np.array(rows), r.corrected_trace(), rtol=1e-5)

    def test_a_descending_scan_is_flipped_to_ascending(self, tmp_path):
        asc, desc = tmp_path / "a.csv", tmp_path / "d.csv"
        write_csv(asc, make_result(descending=False))
        write_csv(desc, make_result(descending=True))
        assert asc.read_text(encoding="utf-8") == desc.read_text(encoding="utf-8")
