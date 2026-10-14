"""PiezoJenaStage: recovering from a controller that lost the plot.

Ported from hardware.py's self-test (lines 2885-3009). This is the only driver
Lillypad speaks to over a bare ASCII protocol with no vendor SDK in between, so
the parsing is ours to get wrong — and the "err,2" case below is a real silent
corruption: the reply happens to contain a comma, so it used to parse as a
position of 0.002 mm.

The fake serial port stands in for pyserial. `_fake_piezo` bypasses __init__'s
port probe because there is no port to probe without hardware; it is the one
place in the suite that reaches into private attributes, and it does so because
the alternative is not testing this driver at all.
"""

from __future__ import annotations

import pytest

from lillypad.hardware import PiezoJenaStage


class FakePiezoSerial:
    """A Piezo Jena controller that emits its reset banner for the first
    `n_banner` position queries, then behaves. `stuck` makes it accept "wr"
    without ever moving, the way one does after rebooting out of remote mode.
    """

    def __init__(self, n_banner=0, stuck=False, start_um=0.0):
        self.n_banner, self.stuck = n_banner, stuck
        self.pos_um = start_um
        self.timeout = 2.0
        self.writes = []
        self._out = b""

    # -- the pyserial surface the driver actually uses --
    def write(self, data):
        self.writes.append(data)
        cmd = data.strip().lower()
        if cmd.startswith(b"wr,"):
            if not self.stuck:
                self.pos_um = float(cmd[3:])
        elif cmd == b"rd":
            if self.n_banner > 0:
                self.n_banner -= 1
                self._out = b"NV1CL V1.236>"       # no newline, on purpose
            else:
                self._out = b"rd,%.2f\r\n" % self.pos_um
        return len(data)

    def read_until(self, sep):
        out, self._out = self._out, b""
        return out

    def read(self, n):
        out, self._out = self._out[:n], self._out[n:]
        return out

    @property
    def in_waiting(self):
        return len(self._out)

    def reset_input_buffer(self):
        self._out = b""

    def reset_output_buffer(self):
        pass

    def close(self):
        pass


class ErrSerial(FakePiezoSerial):
    """A controller that answers a position query with an error code."""

    def write(self, data):
        self.writes.append(data)
        if data.strip().lower() == b"rd":
            self._out = b"err,2\r\n"
        return len(data)


def fake_piezo(ser, travel_um=320.0):
    """A PiezoJenaStage wrapped around `ser`, skipping __init__'s port probe."""
    st = PiezoJenaStage.__new__(PiezoJenaStage)
    st._ser, st._fault = ser, None
    st.travel_mm = travel_um / 1000.0
    st.name = "Piezo Jena [fake]"
    return st


class TestReadbackRecovery:
    @pytest.mark.parametrize("n_banner", [1, 2])
    def test_recovers_from_a_short_run_of_reset_banners(self, n_banner):
        """One or two banners still fit inside NTRIES (3)."""
        pz = fake_piezo(FakePiezoSerial(n_banner=n_banner, start_um=72.5))
        assert pz.get_position() == pytest.approx(0.0725, abs=1e-9)

    def test_an_unbroken_run_of_banners_raises(self):
        pz = fake_piezo(FakePiezoSerial(n_banner=4, start_um=72.5))
        with pytest.raises(RuntimeError) as exc:
            pz.get_position()
        assert "NV1CL" in str(exc.value)
        assert "3 attempts" in str(exc.value)

    def test_err_2_is_rejected_not_read_as_a_position(self):
        """THE silent-corruption case: "err,2" contains a comma, so it used to
        parse as 0.002 mm — a plausible position, silently wrong."""
        pz = fake_piezo(ErrSerial())
        with pytest.raises(RuntimeError):
            got = pz.get_position()
            pytest.fail(f"err,2 parsed as a position: {got}")


class TestMoves:
    def test_a_move_that_lands_leaves_no_fault(self):
        ser = FakePiezoSerial(n_banner=1)
        pz = fake_piezo(ser)
        pz.move_to(0.1)
        assert pz.get_position() == pytest.approx(0.1, abs=1e-9)
        assert pz.position_fault() is None

    def test_a_single_move_issues_exactly_one_write(self):
        """max_step_mm = 0 means move_to takes the no-readback fast path."""
        ser = FakePiezoSerial(n_banner=1)
        pz = fake_piezo(ser)
        pz.move_to(0.1)
        assert sum(w.startswith(b"wr,") for w in ser.writes) == 1

    def test_a_move_that_never_lands_latches_a_fault_instead_of_raising(self):
        """A latched fault, not an exception: a scan can then record the real
        position and carry on, instead of unwinding past run() and losing every
        column measured so far."""
        ser = FakePiezoSerial(stuck=True, start_um=40.0)
        pz = fake_piezo(ser)
        pz.SETTLE_TIMEOUT_S = 0.1      # the wait is the point elsewhere, not here
        pz.move_to(0.1)                # returns instead of raising

        fault = pz.position_fault()
        assert fault and "did not reach" in fault
        assert sum(w.startswith(b"wr,") for w in ser.writes) == PiezoJenaStage.NTRIES
        assert pz.get_position() == pytest.approx(0.04, abs=1e-9), \
            "the real position must stay readable through a fault"

    def test_modes_are_reasserted_on_each_retry(self):
        """The actual fix for a rebooted box: it comes up out of remote mode, so
        re-sending i1/cl is what makes the retry worth anything."""
        ser = FakePiezoSerial(stuck=True, start_um=40.0)
        pz = fake_piezo(ser)
        pz.SETTLE_TIMEOUT_S = 0.1
        pz.move_to(0.1)
        assert ser.writes.count(b"i1\r\n") >= PiezoJenaStage.NTRIES - 1

    def test_a_fault_can_be_cleared(self):
        ser = FakePiezoSerial(stuck=True, start_um=40.0)
        pz = fake_piezo(ser)
        pz.SETTLE_TIMEOUT_S = 0.1
        pz.move_to(0.1)
        assert pz.position_fault() is not None
        pz.clear_position_faults()
        assert pz.position_fault() is None

    def test_a_fault_never_outlives_its_move(self):
        """The next successful move clears it on its own."""
        ser = FakePiezoSerial(stuck=True, start_um=40.0)
        pz = fake_piezo(ser)
        pz.SETTLE_TIMEOUT_S = 0.1
        pz.move_to(0.1)
        assert pz.position_fault() is not None
        ser.stuck = False
        pz.move_to(0.1)
        assert pz.position_fault() is None
