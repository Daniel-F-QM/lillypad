"""piezosystems Jena closed-loop piezo stages.

No vendor SDK at all: the controller speaks a bare ASCII protocol over a 9600
baud serial line (i1 / cl / rd / wr,<um>), so the parsing is ours to get right.
See the reply handling for the two failures that matter — a controller that
comes back from a reset mid-conversation, and an "err,2" that happens to look
like a position because it contains a comma.
"""

from __future__ import annotations

import time

from lillypad.hardware.base import StageBase

class PiezoJenaStage(StageBase):
    """piezosystems Jena closed-loop piezo stage over a plain serial port.

    The controller speaks a bare ASCII protocol at 9600 baud ("i1" remote
    mode, "cl" closed loop, "rd" position readback, "wr,<um>" absolute move).
    It works in MICROMETRES over 0..320 um with 0.1 um resolution; this
    adapter converts to the mm contract of StageBase.

    `port` may be an explicit port name ("COM5") or None (the default) to
    probe the machine's serial ports and use the first that answers like a
    Piezo Jena controller: an unknown command is answered with "err,2", which
    no other device on a bench is likely to say.

    "wr" only sets the target — the controller settles on its own — so
    _move_to_raw() polls "rd" until the readback is within tolerance, with a
    hard deadline (SETTLE_TIMEOUT_S per attempt) so a wedged controller is
    reported rather than hanging the GUI. How it is reported is the point of
    "Recovering from a confused controller" below: a move that never lands
    LATCHES the reason in position_fault() instead of raising. It raises only
    when no readback succeeded at all, i.e. when there is no position to report.

    backlash_mm stays 0: a closed-loop flexure piezo has no screw and therefore
    no backlash, and "rd" is a real measurement rather than a step counter, so
    every move verifies itself against the stage's own 0.1 um resolution.

    Recovering from a confused controller
    -------------------------------------
    This controller emits unsolicited output — a power-up/reset banner such as
    b"NV1CL V1.236>" — that lands in the middle of a conversation, has no
    trailing newline, and means the box has just rebooted OUT of remote and
    closed-loop mode. Two consequences drive the design here:

      * Every reply is validated against the prefix it must start with. Testing
        only "does float() parse the tail after the last comma" accepted
        b"err,2" as a position of 0.002 mm — a wrong position that reads as a
        perfectly good one is far worse than an exception.
      * Both the READ and the MOVE get NTRIES attempts, and each retry
        re-asserts "i1"/"cl" first, because a rebooted controller has silently
        dropped the modes that make "wr" do anything at all.

    When a move still has not landed after all of that, _move_to_raw latches
    the reason in position_fault() and returns instead of raising. That routes
    the failure into the scan's own stage-health path, which records it against
    the column, reports the position the stage really is at, and stops the scan
    only when abort_on_stage_fault is set — instead of an exception unwinding
    past the whole scan and discarding every column already measured.
    """
    NTRIES = 3          # attempts per readback, and per move
    SETTLE_TIMEOUT_S = 5.0     # how long one move attempt gets to land
    QUERY_TIMEOUT_S = 0.5      # per reply — see _read_reply
    _EOL = b"\r\n"      # command terminator, probe included
    resolution_mm = 0.0001                   # 0.1 um, the controller's own step
    _TOL_UM = resolution_mm * 1000.0         # settle window = that step
    # Travel is 0.32 mm against StageBase's 5 mm default, so the long-move
    # split can never fire — but a non-zero value makes move_to() take an extra
    # get_position() before EVERY move. Zero takes the no-readback fast path:
    # half the serial traffic per move, and one less chance to catch a banner.
    max_step_mm = 0.0

    def __init__(self, port: str | None = None, travel_um: float = 320.0,
                 probe_timeout_s: float = 0.5):
        import serial

        self._ser = None
        # Latched reason the readback cannot be trusted, published through
        # position_fault(). Set before _open so a failure in there still leaves
        # a well-formed object for disconnect() to clean up.
        self._fault: str | None = None
        self._ser, used_port = self._open(port, probe_timeout_s, serial)
        try:
            self._ser.timeout = self.QUERY_TIMEOUT_S   # working, per read
            self._assert_modes()
            self.travel_mm = travel_um / 1000.0
            self.name = f"Piezo Jena [{used_port}]"
        except Exception:
            self.disconnect()
            raise

    @staticmethod
    def _candidate_ports() -> list[str]:
        try:
            from serial.tools import list_ports
            # Bluetooth-link COM ports block for seconds on open — never a
            # piezo controller, so don't even probe them.
            return [p.device for p in list_ports.comports()
                    if not p.description.startswith(
                        "Standard Serial over Bluetooth link")]
        except Exception:
            return []

    def _open(self, port, probe_timeout_s, serial):
        """Open and identify the controller; return (Serial, port name).

        Each candidate is probed with a deliberately unknown command — a
        Piezo Jena controller answers "err,2", anything else is not ours.
        """
        ports = [port] if port else self._candidate_ports()
        if not ports:
            raise RuntimeError(
                "No serial ports found. Pass an explicit port (e.g. 'COM5').")

        last_err = None
        for p in ports:
            ser = None
            try:
                ser = serial.Serial(p, baudrate=9600,
                                    timeout=probe_timeout_s)
                ser.reset_input_buffer()
                ser.write(b"hello" + self._EOL)   # same terminator as _command
                if ser.readline().startswith(b"err,2"):
                    return ser, p
                ser.close()
            except Exception as e:            # busy / no reply / not ours
                last_err = e
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
        if port:
            raise RuntimeError(f"No Piezo Jena stage on {port!r}: {last_err}")
        raise RuntimeError(
            f"No Piezo Jena stage found on any serial port "
            f"({', '.join(ports)}). Last error: {last_err}")

    def _command(self, cmd: bytes) -> None:
        self._ser.write(cmd + self._EOL)

    def _assert_modes(self) -> None:
        """Put the controller into remote + closed-loop mode.

        Neither command answers, so there is nothing to verify — but it is
        cheap and idempotent, which is what makes it usable as the first step
        of a retry: a controller that emitted a reset banner has dropped both
        modes, and until they are back "wr" is quietly ignored.
        """
        self._command(b"i1")           # remote-control mode
        time.sleep(0.05)
        self._command(b"cl")           # closed-loop mode
        time.sleep(0.05)

    def _read_reply(self) -> bytes:
        """One reply, terminator or timeout.

        Unterminated output (the banner has no newline) costs the full read
        timeout — there is no way to know a line has ended early. That is why
        the port runs at QUERY_TIMEOUT_S rather than seconds: a "rd" answer is
        ~10 bytes, about 10 ms at 9600 baud, so the timeout only ever elapses
        when something is already wrong, and NTRIES of them stay well inside
        one settle deadline. Anything that arrived late is swept up too, so it
        cannot masquerade as the next command's answer.
        """
        reply = self._ser.read_until(b"\n")
        if not reply.endswith(b"\n"):
            extra = getattr(self._ser, "in_waiting", 0) or 0
            if extra:
                reply += self._ser.read(extra)
        return reply

    def _query(self, cmd: bytes, prefix: bytes) -> bytes:
        """Send `cmd` and return the first reply that starts with `prefix`.

        Up to NTRIES attempts. A reply that does not match is discarded, the
        input drained, and the controller's modes re-asserted before asking
        again — that sequence is what recovers from a reset banner instead of
        giving up on it.

        Matching on the prefix rather than on "parses as a number" is the point
        of the helper: b"err,2" would otherwise be read as a position.
        """
        seen = []
        for attempt in range(self.NTRIES):
            if attempt:
                # Something is out of step: clear both directions and put the
                # controller back into the mode it answers in.
                try:
                    self._ser.reset_input_buffer()
                    self._ser.reset_output_buffer()
                except Exception:
                    pass
                self._assert_modes()
            self._ser.reset_input_buffer()
            self._command(cmd)
            reply = self._read_reply()
            if reply.startswith(prefix):
                return reply
            seen.append(reply)
        raise RuntimeError(
            f"Piezo Jena stage gave no usable {cmd.decode()} reply in "
            f"{self.NTRIES} attempts (expected {prefix.decode()!r}, got "
            f"{', '.join(repr(r) for r in seen)}).")

    def get_position(self) -> float:
        # Reply looks like b"rd,123.4\r\n" — the value follows the comma.
        reply = self._query(b"rd", b"rd,")
        try:
            return float(reply[reply.find(b",") + 1:]) / 1000.0
        except ValueError:
            raise RuntimeError(
                f"Piezo Jena stage gave no usable position readback "
                f"(reply: {reply!r}).") from None

    def _move_to_raw(self, position_mm: float) -> None:
        """Command an absolute move and wait for it to land.

        Each attempt re-asserts the modes and re-issues "wr" — a controller
        that reset simply never acted on the previous one. After NTRIES the
        fault is LATCHED rather than raised: the stage is wherever it is, that
        position is readable and honest, and the scan is entitled to record it
        and carry on if abort_on_stage_fault is off.
        """
        # Clamp instead of raising: the scan worker pre-checks its targets
        # against travel_mm, so anything out of range here is a manual move.
        target_um = min(max(float(position_mm) * 1000.0, 0.0),
                        self.travel_mm * 1000.0)
        # Cleared per move, so position_fault() stays a plain read and each
        # scan column only ever sees the fault from its own move.
        self._fault = None
        last_um = None
        read_err = None
        for attempt in range(self.NTRIES):
            if attempt:
                self._assert_modes()
            self._command(b"wr,%.2f" % target_um)
            deadline = time.monotonic() + self.SETTLE_TIMEOUT_S
            while True:
                try:
                    last_um = self.get_position() * 1000.0
                except RuntimeError as e:
                    # The whole point of retrying: a controller that started
                    # talking nonsense mid-settle gets its modes re-asserted
                    # and the move re-issued, rather than taking the scan down.
                    read_err = e
                    break
                if abs(last_um - target_um) <= self._TOL_UM:
                    return
                if time.monotonic() > deadline:
                    break
                time.sleep(0.05)
        if last_um is None:
            # Never got a single reading — there is no position to accept, so
            # this one really is fatal.
            raise RuntimeError(
                f"Piezo Jena stage is not answering position queries, so the "
                f"move to {target_um:.2f} um cannot be verified: {read_err}")
        self._fault = (
            f"stage did not reach {target_um:.2f} um in {self.NTRIES} attempts "
            f"(readback {last_um:.2f} um, window ±{self._TOL_UM:g} um)")
        if read_err is not None:
            self._fault += f"; last readback error: {read_err}"

    def position_fault(self) -> str | None:
        """The last move's failure to land, or None.

        Set by _move_to_raw and cleared at the start of the next move, so this
        reports the CURRENT move rather than latching forever — the scan calls
        it after every column.
        """
        return self._fault

    def clear_position_faults(self) -> None:
        self._fault = None

    def disconnect(self) -> None:
        ser = getattr(self, "_ser", None)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            self._ser = None


# Both python-seabreeze backends, in the order the GUI offers them. pyseabreeze
# first: it is the only backend that supports the newer Ocean Insight models
