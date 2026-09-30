"""What the instrument knows: the state shared between the window, its
controllers and its dialogs.

This is deliberately NOT everything `FrogWindow` holds. The window keeps its own
display scratch — the render buffers, the pending-fit flags, the alignment view,
the running scan's accumulators — because those belong to one panel each and
nothing else reads them. What lives here is the state that three or more parts of
the program have to agree about:

  * which devices are connected,
  * what the scan is configured to do,
  * what the recorded dark is and at what exposure,
  * what the two spectrometer slots hold, and
  * what the simulator is standing in for.

Gathering them answers a question the flat window could not: *what is the state
of the instrument right now?* A newcomer can read this file and know, without
grepping 200 attribute assignments.

`FrogWindow` keeps a `@property` forwarder for every field, so `self.spec` still
works everywhere it did and none of its ~200 existing uses churn. The
forwarders are not a transitional measure — reading `self.spec` inside the window
is the right level of ceremony for the object that owns the instrument. What the
indirection buys is that a *dialog* or a *controller* can be handed
`InstrumentState` alone and be unable to reach the widgets.

Two invariants this file is the place to state:

  * **There is no "multi-spectrometer mode" flag.** Both slots exist from the
    start; slot 1 alone IS single-spectrometer mode. "Is a pair live" is
    answered by `pair_live()`, off the connected device, never by a second piece
    of state that could disagree with it.
  * **`None` means nothing is connected** — a real absence, not a stand-in
    device, so nothing can quietly return plausible-looking counts or positions
    for hardware that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lillypad.hardware import SEABREEZE_BACKENDS, StitchedSpectrometer
from lillypad.scan import FrogScanConfig
from lillypad.sim import DEFAULT_PULSE


def _empty_slots():
    """The two spectrometer slots: a device choice and a calibration choice each.

    "serials" holds tagged ids ("vendor:serial", see hardware.spec_ident) or one
    of the SIM_SLOT_DEVICES sentinels — the two slots may hold devices from
    different vendors.
    """
    return {"serials": [None, None], "labels": [None, None],
            "cals": [None, None]}


@dataclass
class InstrumentState:
    """The connected instrument, as the whole program sees it."""

    # ── Devices ──────────────────────────────────────────────────────────────
    # NOTHING is connected at startup. The app used to open on a simulated stage
    # and spectrometer, which meant a fake spectrum was on screen before anyone
    # had chosen anything and looked exactly like a measurement.
    spec: object | None = None
    stage: object | None = None

    # ── What a scan will do ──────────────────────────────────────────────────
    scan_cfg: FrogScanConfig = field(
        default_factory=lambda: FrogScanConfig(
            delay_start_fs=-500, delay_stop_fs=500,
            delay_step_fs=1.0, zero_pos_um=150000.0))

    # ── The recorded dark ────────────────────────────────────────────────────
    # In exactly one of two forms — never both.
    #   single device:  `background`, one RAW frame.
    #   stitched pair:  `background_members`, the two RAW member frames, and
    #                   `background` stays None.
    # A pair's dark is kept per member because the pedestal has to come off
    # BEFORE each member's own calibration, and because the merged background is
    # then derived (StitchedSpectrometer.combined_dark) at whatever stitch
    # factor and band are current. Storing the merged frame instead baked both
    # into its overlap region, where the two devices are already mixed, and it
    # went silently wrong on the next re-fit.
    background: object | None = None
    background_members: object | None = None
    # Exposure(s) the dark was recorded at — (exp,) or (exp1, exp2) in
    # `spec.members` order. A dark is offset + dark-current x t, so it is only
    # valid at its own exposure; _invalidate_dark drops it when that changes
    # rather than rescaling something that is not a pure ratio.
    background_exposures: object | None = None

    # ── The two spectrometer slots ───────────────────────────────────────────
    multi: dict = field(default_factory=_empty_slots)
    # The live spectrometer objects in SLOT order once a pair is connected.
    multi_members: list = field(default_factory=lambda: [None, None])
    # Set when an integration time changes under a stitched pair: frames are raw
    # counts, so stitch_factor carries the exposure ratio and goes stale.
    # Surfaced in the Spectrometer window's Stitching block rather than silently
    # re-fitted.
    stitch_stale: bool = False

    # ── What the simulator stands in for ─────────────────────────────────────
    # Kept here, not on the simulator, so the choice survives a swap to real
    # hardware and back.
    sim_pulse: str = DEFAULT_PULSE
    sim_gate: str = "shg"

    # seabreeze backend used for every enumeration/connect. pyseabreeze (the
    # default, SEABREEZE_BACKENDS[0]) also covers the newer Ocean Insight models
    # that cseabreeze does not know about.
    seabreeze_backend: str = SEABREEZE_BACKENDS[0]

    # ── Derived answers ──────────────────────────────────────────────────────
    # Everything that reaches a device goes through one of these three, so that
    # "is there hardware" is asked in one way and cannot be asked in two.
    def have_spec(self) -> bool:
        return self.spec is not None

    def have_stage(self) -> bool:
        return self.stage is not None

    def pair_live(self) -> bool:
        """True while two spectrometers are connected as one stitched device.
        This is the ONLY definition of multi-spectrometer mode."""
        return isinstance(self.spec, StitchedSpectrometer)
