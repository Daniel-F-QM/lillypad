"""Connecting, disconnecting and configuring the two devices.

The biggest of the controllers, because "what is connected" is the question the
rest of the program keeps asking. It owns:

  * the **feed handover** (`device_lock`), which every GUI-thread device call
    has to take, because the live feed drives the same two devices from its own
    thread and a real Kinesis/Zaber/seabreeze will interleave transactions and
    return garbage;
  * **connect and disconnect** for each stage vendor and each spectrometer
    vendor, plus the simulators;
  * the **two spectrometer slots** and the stitched pair they open as one
    device;
  * **calibration** — the menu, the files, and applying one to a slot;
  * the **stitch**: fitting the factor, setting it by hand, and the overlap
    band it is fitted over.

Nothing here is a simulator by accident. The Simulation window is the only route
to one, and every vendor SDK still loads inside the function that needs it, so a
missing DLL costs exactly one device's connect call.

The methods the dialogs call have no underscore. That is the honest signal: they
are this controller's public API, and a dialog reaching a private would be
reaching into an implementation.
"""

from contextlib import contextmanager
from pathlib import Path
import numpy as np
import shutil

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (QFileDialog, QInputDialog, QMenu, QMessageBox,
                               QPushButton)
from lillypad.paths import CALIBRATION_DIR
from lillypad.gui.dialogs.base import DevicePickerDialog
from lillypad.gui.text import FEED_BUSY_MSG
from lillypad.gui.workers import _exposure_bounds
from lillypad.hardware import (AvantesSpectrometer, KinesisStage,
                               PiezoJenaStage, SeabreezeSpectrometer,
                               StitchedSpectrometer, ZaberStage,
                               list_avantes_spectrometers,
                               list_kinesis_stages,
                               list_seabreeze_spectrometers,
                               list_spectrometers, load_calibration_file,
                               open_spectrometer)
from lillypad.sim import (SIM_SPECTROMETERS, SimulatedSpectrometer,
                          SimulatedStage, sim_spectrometer_pair)
from lillypad.scan import position_to_delay_fs, stage_to_um, um_to_stage


# Simulated members offered alongside real devices in the multi-spectrometer
# slots, so the mode can be exercised without two spectrometers on the bench.
# These are sentinels, not device ids — _open_slot_device matches them BEFORE
# it hands anything to hardware.open_spectrometer, so they deliberately do not
# carry a "vendor:serial" tag. Each covers one overlapping half of the
# simulated signal band; see _make_sim_member for why halves and not two
# full-band copies.
SIM_SLOT_DEVICES = (("__sim_blue__", "Simulated — blue half"),
                    ("__sim_red__",  "Simulated — red half"))


class DeviceController:
    """Everything about the devices: connect, disconnect, slots, stitch."""

    def __init__(self, win):
        self.win = win

    @contextmanager
    def device_lock(self):
        """Take the stage/spectrometer for a device call made ON the GUI thread.

        The live feed drives the same two devices from its own thread, so any
        direct `self.win.stage.*` / `self.win.spec.*` call from a slot has to park the
        feed first — otherwise two threads are inside one vendor driver at once.
        (Harmless for the simulators; a real Kinesis/Zaber/seabreeze can
        interleave transactions and return garbage.)

        Yields True when the feed actually parked. On False the caller MUST NOT
        touch the device. The feed is restored on the way out either way, unless
        a scan now owns the hardware.
        """
        ok = self.win._feed.pause()
        try:
            yield ok
        finally:
            if self.win.btn_feed.isChecked() and not self.win.scanning.running():
                self.win._feed.resume()

    # ── Hardware ─────────────────────────────────────────────────────────────
    def _make_sim_spectrometer(self):
        """A sim spectrometer that derives its delay from the LIVE stage (so it
        works with a real stage too — the Stage.py test setup)."""
        return SimulatedSpectrometer(
            self.win.stage, gate=self.win.sim_gate, pulse=self.win.sim_pulse,
            position_to_delay=lambda pos_mm: position_to_delay_fs(
                stage_to_um(pos_mm), self.win.scan_cfg.zero_pos_um, self.win.scan_cfg.pass_factor))

    def _make_sim_device(self, half):
        """One member of the simulated pair, modelled on a REAL spectrometer.

        Used for a beam that names a pair in PULSE_SHAPES (see "bird", whose
        SHG covers 500–1400 nm and needs the Si + InGaAs bench to be measured
        at all). Unlike the two halves below, these devices have their own
        pixel grids, detector responses, darks and noise, so the two curves
        differ for the reasons two real spectrometers differ — and the fit that
        matches them has something real to absorb.

        The matching calibration file is loaded when it is on disk: it is the
        reciprocal of the device's simulated response, exactly as the shipped
        files are for the hardware they were measured on. Without it the pair
        still connects, and Auto-stitch then reports the large residual an
        uncalibrated Si/InGaAs overlap really does have.
        """
        model = SIM_SPECTROMETERS[
            sim_spectrometer_pair(self.win.sim_pulse, self.win.sim_gate)[half]]
        sim = SimulatedSpectrometer(
            self.win.stage, gate=self.win.sim_gate, pulse=self.win.sim_pulse,
            position_to_delay=lambda pos_mm: position_to_delay_fs(
                stage_to_um(pos_mm), self.win.scan_cfg.zero_pos_um,
                self.win.scan_cfg.pass_factor),
            **model["device"])
        sim.name = model["name"]
        cal = CALIBRATION_DIR / f"{model['calibration']}.txt"
        if cal.is_file():
            try:
                sim.set_calibration(cal)
            except Exception:
                pass        # raw counts is a fine fallback; nothing is wrong yet
        return sim

    def _make_sim_member(self, half):
        """A simulated spectrometer covering one half of the simulated signal
        band, for use as a stitched-pair member.

        Halves, not two full-band copies: identical grids would make the whole
        spectrum "overlap", so the stitch geometry — a blue-only region, a
        shared middle, a red-only region — would never be exercised. The 30%
        shared middle is what Auto-stitch fits over. The band itself is only
        known to a constructed simulator (it is sized from the signal), hence
        the throwaway probe; construction touches no hardware and costs a
        handful of FFTs.

        A beam that names its own pair of instruments gets those instead.
        """
        if sim_spectrometer_pair(self.win.sim_pulse, self.win.sim_gate) is not None:
            return self._make_sim_device(half)
        probe = self._make_sim_spectrometer()
        wl = np.asarray(probe.wavelengths, float)
        lo, hi = float(wl[0]), float(wl[-1])
        span = hi - lo
        start, end = ((lo, lo + 0.65 * span) if half == 0
                      else (lo + 0.35 * span, hi))
        sim = SimulatedSpectrometer(
            self.win.stage, gate=self.win.sim_gate, pulse=self.win.sim_pulse,
            wl_start=start, wl_end=end, n_pixels=512,
            # DIFFERENT pedestals, and deliberately so: the halves used to
            # default to background_counts=0, which is why every dark bug in
            # the stitched path could only be found on the bench. Two unequal,
            # exposure-scaled pedestals (see SimulatedSpectrometer._raw_column)
            # are what a real pair looks like, and they make a mis-applied dark
            # show up as a step at the seam.
            background_counts=300.0 if half == 0 else 1200.0,
            position_to_delay=lambda pos_mm: position_to_delay_fs(
                stage_to_um(pos_mm), self.win.scan_cfg.zero_pos_um,
                self.win.scan_cfg.pass_factor))
        # Distinct names: the two members share a class, so the saturation-lamp
        # tooltips and the Auto-stitch message would otherwise name both
        # identically and give no way to tell which is which.
        sim.name = (f"simulated {'blue' if half == 0 else 'red'} half "
                    f"[{start:.0f}–{end:.0f} nm]")
        return sim

    def _open_slot_device(self, ident):
        """Open whatever a multi-spectrometer slot points at — a simulated
        half, or a real device by its tagged id. The sentinels are matched
        first: they carry no vendor tag, so open_spectrometer would reject
        them."""
        for half, (sim_ident, _label) in enumerate(SIM_SLOT_DEVICES):
            if ident == sim_ident:
                return self._make_sim_member(half)
        return open_spectrometer(ident, self.win.seabreeze_backend)

    def _sim_devices(self):
        """Every SimulatedSpectrometer the connected device is made of — itself,
        or the members of a stitched pair. Empty for real hardware."""
        spec = self.win.spec
        devs = spec.members if isinstance(spec, StitchedSpectrometer) else (spec,)
        return [d for d in devs if isinstance(d, SimulatedSpectrometer)]

    def _attach_stage_warnings(self, stage):
        """Route a stage's warnings into the status bar. Safe from any thread —
        emit() is what crosses back to the GUI one."""
        stage.warn_cb = self.win.stage_warning.emit

    def _on_stage_warning(self, message):
        self.win.status.showMessage(f"Stage: {message}", 5000)

    def _apply_stage(self, new_stage):
        """Swap in a new stage. Returns (ok, error) — on failure NOTHING has
        changed and `new_stage` is still the caller's to dispose of.

        `new_stage` may be None: that is the disconnect path, and it takes the
        same lock and the same teardown as any other swap.

        Refused outright while a scan is running: FrogScanWorker captured the
        old stage at construction, so disconnecting it here would pull the
        device out from under the thread currently driving it.
        """
        if self.win.scanning.running():
            return False, "A scan is running — stop it before changing hardware."
        # Pause the feed (and wait for any in-flight acquire) before swapping,
        # so the worker never reads a half-swapped device.
        with self.device_lock() as ok:
            if not ok:
                return False, FEED_BUSY_MSG
            old = getattr(self.win, "stage", None)
            if old is not None and old is not new_stage:
                try:
                    old.disconnect()
                except Exception:
                    pass
            self.win.stage = new_stage
            if new_stage is not None:
                self._attach_stage_warnings(new_stage)
            # A simulated spectrometer reads the live stage, so re-point it.
            # None is allowed: SimulatedSpectrometer.acquire falls back to zero
            # delay, so the simulator keeps working with no stage attached.
            #
            # Every simulator INSIDE a stitched pair too, not just a top-level
            # one. The pair is a StitchedSpectrometer, so checking only
            # win.spec skipped both members: a pair connected before the stage
            # kept stage=None, and every frame came from zero delay — a trace
            # with the same column at every delay.
            for dev in self._sim_devices():
                dev.stage = self.win.stage
            if isinstance(new_stage, SimulatedStage):
                new_stage.move_to(um_to_stage(self.win.scan_cfg.zero_pos_um))
            elif new_stage is not None:
                # Real stage: adopt its ACTUAL position as zero-delay so
                # "Move to 0 fs" can never slam it into a travel limit.
                # An axis with no reference reads an arbitrary number here, but
                # the zero only has to be consistent with the scan, which runs
                # in the same power-up frame. Homing later moves that frame, and
                # _home_stage drops the zero when it does.
                self.win.scan_cfg.zero_pos_um = stage_to_um(new_stage.get_position())
            self.win.stage_ctl._update_stage_unit_ranges()  # new travel + possibly new zero
            self.win.stage_ctl._refresh_positions()   # inside the lock — we still own the stage
        self.win.stage_ctl._sync_backlash_ui()        # the new stage brings its own default
        self.win.stage_ctl._refresh_home_button()     # …and may not be able to home
        self.win.stage_ctl._refresh_scan_um()         # zero may have moved: fs→um previews
        self.win.appearance._sync_connection_ui()      # panel swaps between controls/Connect
        return True, ""

    def _apply_spectrometer(self, new_spec):
        """Swap in a new spectrometer, or None to disconnect. (ok, error);
        see _apply_stage."""
        if self.win.scanning.running():
            return False, "A scan is running — stop it before changing hardware."
        with self.device_lock() as ok:
            if not ok:
                return False, FEED_BUSY_MSG
            old = getattr(self.win, "spec", None)
            if old is not None and old is not new_spec:
                try:
                    old.disconnect()
                except Exception:
                    pass
            self.win.spec = new_spec
            if not isinstance(new_spec, StitchedSpectrometer):
                # Any single device replaces a stitched pair wholesale — the
                # slot->member mapping is only meaningful while the pair lives.
                self.win._multi_members = [None, None]
            if new_spec is not None:
                # Clamped to the INCOMING device's range: the box still carries
                # the outgoing device's value, and 0.05 ms from an Avantes is a
                # hard error on an Ocean unit with a 1 ms floor. _sync_multi_ui
                # below reseeds the box from whatever the device ended up with.
                lo, hi = _exposure_bounds(new_spec)
                new_spec.set_integration_time(
                    min(hi, max(lo, float(self.win.spin_integration.value()))))
            # The old device's frames and dark are meaningless for the new one
            # — and a different pixel count (certain with a stitched grid)
            # would crash the dark subtraction outright.
            self.win.background = None
            self.win.background_members = None
            self.win.background_exposures = None
            self.win._dark_member_warned = False
            if new_spec is None:
                # Disconnecting: take the curve off the panel too, or the last
                # frame would sit frozen under the "no spectrometer" message.
                self.win.canvas.clear_spectrum()
            else:
                self.win.canvas.clear_members()  # a dead pair's curves can't linger
            # And the reference with them, in both cases: it is a spectrum from
            # the outgoing device, on that device's pixel grid, and comparing
            # the incoming one against it would be meaningless where it was not
            # simply misleading.
            self.win.canvas.clear_reference()
            # Same for both halves of alignment mode: a difference and a raw
            # trace describe the device that measured them, right down to the
            # pixel grid they sit on.
            self.win.alignment._close_align_mode()
            self.win.alignment._clear_align_trace()
            self.win.chk_dark.setChecked(False); self.win.chk_dark.setEnabled(False)
            self.win.last_spectrum = None
            self.win._live_frame = None
            self.win._sat_frame = None
            self.win._sat_peak = -1.0
            # New device, new full scale — a latched warning about the old one
            # would be meaningless (and its threshold plain wrong).
            self.win.feed.reset_saturation()
            # Re-centre spectrum when a new spectrometer comes online
            self.win._pending_fit = True
            self.win.canvas.set_autoscale_x(True)
            if hasattr(self.win, 'dlg_graphics'):
                self.win.dlg_graphics.chk_auto_x.setChecked(True)
        # Outside the lock: pure widget state, no device I/O.
        self.win.appearance._sync_multi_ui()
        return True, ""

    def use_sim_stage(self):
        ok, err = self._apply_stage(SimulatedStage(travel_mm=300.0))
        if not ok:
            return False, err
        self.win.status.showMessage("Stage: simulated.", 4000)
        return True, "Stage set to simulated."

    def disconnect_stage(self):
        ok, err = self._apply_stage(None)
        if not ok:
            return False, err
        self.win.status.showMessage("Stage disconnected.", 4000)
        return True, "Stage disconnected."

    def connect_kinesis_stage(self, conn=None):
        """Connect a Kinesis stage. `conn` names one the caller already chose
        (the Stage dialog's scan list); without it this enumerates and picks or
        asks, which is what any other caller needs."""
        if conn is None:
            try:
                devices = list_kinesis_stages()   # brief per-device model query
            except Exception as e:
                return False, f"Stage connect failed: {e}"
            if not devices:
                return False, "Stage connect failed: No Kinesis devices found."
            if len(devices) > 1:
                conn = DevicePickerDialog.pick(
                    self.win.dlg_stage, devices, "Select Kinesis Stage",
                    f"{len(devices)} Kinesis devices found — choose one:")
                if conn is None:
                    return False, "Cancelled — stage unchanged."
            else:
                conn = devices[0][1]
        # Release our own handle on this controller first: the USB link opens
        # once, so reconnecting the stage already in use would otherwise fail
        # with an open error while the old handle sat there.
        old = getattr(self.win, "stage", None)
        if isinstance(old, KinesisStage) and old.conn == str(conn):
            ok, err = self._apply_stage(None)
            if not ok:
                return False, err
        try:
            # Identifies the stage and refuses anything it can't calibrate in mm.
            stage = KinesisStage(serial=conn)
        except Exception as e:
            return False, f"Stage connect failed: {e}"
        ok, err = self._apply_stage(stage)
        if not ok:
            self._drop(stage)      # never adopted — don't leak the connection
            return False, err
        if stage.needs_homing:
            self.win.status.showMessage(
                f"Stage: {stage.name} — NOT HOMED, working in its power-up "
                f"frame.", 0)
            t = stage.travel_mm
            if stage.moves_locked:
                self._prompt_homing()
            if not t and not stage.moves_locked:
                reach = ""
            elif stage.hw_limits_stop:
                reach = (f" Manual moves are allowed ±{t:.0f} mm from here; "
                         f"the limit switches stop the carriage at either end.")
            else:
                reach = (" Its limit switches could not be confirmed to stop "
                         "a move, so the stage is LOCKED until you home it or "
                         "choose to proceed without homing.")
            return True, (
                f"Stage connected: {stage.name}, but the controller reports the "
                f"axis NOT HOMED (it has been power-cycled). Positions count from "
                f"wherever it powered up. You can set zero and scan in that "
                f"frame — a scan that runs into an end stops there — but "
                f"homing later moves the frame and drops the zero.{reach}")
        self.win.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        travel = (f"Travel {stage.travel_mm:.0f} mm." if stage.travel_mm else
                  "Travel unknown — no soft range limit, set one in the scan "
                  "config if you need it.")
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.win.scan_cfg.zero_pos_um:.1f} um). {travel}")

    def _prompt_homing(self):
        """Show the red homing warning for a stage that just connected locked.
        Deferred to the event loop so the connect message lands first and the
        modal is not opened from inside the connect call."""
        QTimer.singleShot(0, self.win.stage_ctl.check_unlocked)

    def connect_zaber_stage(self, port=None):
        try:
            stage = ZaberStage(port=port)     # blank port -> auto-scan
        except Exception as e:
            return False, f"Zaber connect failed: {e}"
        ok, err = self._apply_stage(stage)
        if not ok:
            self._drop(stage)      # never adopted — release the serial port
            return False, err
        if stage.needs_homing:
            self.win.status.showMessage(
                f"Stage: {stage.name} — NOT HOMED, working in its power-up "
                f"frame.", 0)
            locked = ""
            if stage.moves_locked:
                self._prompt_homing()
                locked = (" Its limit switches could not be confirmed to stop "
                          "a move, so the stage is LOCKED until you home it or "
                          "choose to proceed without homing.")
            return True, (
                f"Stage connected: {stage.name}, but the axis is NOT HOMED. "
                f"Positions count from wherever it powered up. You can set zero "
                f"and scan in that frame, but homing later moves the frame and "
                f"drops the zero.{locked}")
        self.win.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        backlash_um = stage_to_um(stage.backlash_mm)
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.win.scan_cfg.zero_pos_um:.1f} um). "
                      f"Backlash approach {backlash_um:.0f} um — Zaber does no "
                      f"backlash correction of its own, and without this the "
                      f"marked zero and a scan sweep sit in frames that differ "
                      f"by the screw slack.")

    def connect_piezo_jena_stage(self, port=None):
        try:
            stage = PiezoJenaStage(port=port)     # blank port -> auto-scan
        except Exception as e:
            return False, f"Piezo Jena connect failed: {e}"
        ok, err = self._apply_stage(stage)
        if not ok:
            self._drop(stage)      # never adopted — release the serial port
            return False, err
        self.win.status.showMessage(f"Stage: {stage.name} — zero at current position.", 5000)
        return True, (f"Stage connected: {stage.name}. Zero-delay set to current "
                      f"position ({self.win.scan_cfg.zero_pos_um:.1f} um).")

    def use_sim_spectrometer(self):
        ok, err = self._apply_spectrometer(self._make_sim_spectrometer())
        if not ok:
            return False, err
        # The full-band simulator is not reachable by id, so no slot can name
        # it — empty them rather than leave them pointing at whatever was
        # connected before.
        self.win._multi["serials"] = [None, None]
        self.win._multi["labels"]  = [None, None]
        self.win.status.showMessage(f"Spectrometer: {self.win.spec.name}.", 4000)
        return True, f"Simulated: {self.win.spec.pulse_label}."

    def use_sim_pair(self):
        """Both slots on simulated halves, connected as a stitched pair.

        The only way to exercise multi-spectrometer mode without two devices on
        the bench — _open_slot_device matches the SIM_SLOT_DEVICES sentinels to
        the half-band simulators, so this is the ordinary pair path with the
        slots filled in for you.

        The slot labels follow what the sentinels will actually open: the two
        named instruments when the beam brings a pair of its own, the two
        halves of its band otherwise.
        """
        models = sim_spectrometer_pair(self.win.sim_pulse, self.win.sim_gate)
        for slot, (ident, label) in enumerate(SIM_SLOT_DEVICES):
            self.win._multi["serials"][slot] = ident
            self.win._multi["labels"][slot]  = (
                label if models is None
                else SIM_SPECTROMETERS[models[slot]]["name"])
        return self._connect_multi_pair()

    def disconnect_spectrometer(self):
        """Release whatever is connected and empty both slots. (ok, msg)."""
        ok, err = self._apply_spectrometer(None)
        if not ok:
            return False, err
        self.win._multi["serials"] = [None, None]
        self.win._multi["labels"]  = [None, None]
        self.win.status.showMessage("Spectrometer disconnected.", 4000)
        return True, "Spectrometer disconnected."

    def live_seabreeze_backend(self):
        """Backend name of the currently connected seabreeze device, or None
        when no live device is a seabreeze one.

        Every member is checked, not just spec1: StitchedSpectrometer assigns
        spec1 by wavelength, so in a mixed-vendor pair the seabreeze device can
        land in either slot. Missing it would skip the release step below and
        let select_seabreeze_backend shut the API down under a LIVE device.
        calibration_targets() is the existing "physical devices behind the
        facade" accessor — [self] for a single device, both members for a pair.
        """
        if not self.win._have_spec():
            return None
        for spec in self.win.spec.calibration_targets():
            if isinstance(spec, SeabreezeSpectrometer):
                return getattr(spec, "backend", None)
        return None

    def list_spectrometers(self):
        """Enumerate every vendor's attached spectrometers as tagged ids.
        Selecting a seabreeze backend tears the previous backend's API down,
        which would sever a device still open through it — so any such device
        is released first. Raises RuntimeError when that release is refused
        (scan running / feed busy)."""
        live = self.live_seabreeze_backend()
        if live is not None and live != self.win.seabreeze_backend:
            ok, err = self._apply_spectrometer(None)
            if not ok:
                raise RuntimeError(err)
        return list_spectrometers(self.win.seabreeze_backend)

    def scan_notes(self):
        """Why a vendor contributed nothing to the last enumeration.

        list_spectrometers deliberately swallows a vendor whose SDK is missing,
        so one absent driver cannot hide the other vendor's devices. That is
        right for the list and wrong for the operator staring at an empty one,
        so the same two calls are made again here purely for their errors —
        this is what the old per-vendor connect buttons used to surface.
        """
        notes = []
        try:
            list_seabreeze_spectrometers(self.win.seabreeze_backend)
        except Exception as e:
            notes.append(f"seabreeze ({self.win.seabreeze_backend}): {e}")
        try:
            list_avantes_spectrometers()
        except Exception as e:
            notes.append(f"Avantes: {e}")
        return notes

    def avantes_devices(self):
        """Every live Avantes device — [], [one], or both members of a mixed
        stitched pair. calibration_targets() is the existing accessor for
        "the physical devices behind whatever facade is connected", so this
        keeps working for any future composite."""
        if not self.win._have_spec():
            return []
        return [s for s in self.win.spec.calibration_targets()
                if isinstance(s, AvantesSpectrometer)]

    @staticmethod
    def _drop(device):
        """Close a device we opened but did not end up adopting."""
        try:
            device.disconnect()
        except Exception:
            pass

    # ── Calibration menu ──────────────────────────────────────────────────────
    def _build_calibration_button(self):
        """Header Calibration control — assigns an intensity-calibration file
        to each connected physical spectrometer. Rebuilt on every open: both
        the calibration_files folder and the connected devices change at
        runtime."""
        menu = QMenu(self.win)
        menu.aboutToShow.connect(lambda: self._populate_calibration_menu(menu))
        self.win.btn_cal = QPushButton("Calibration")
        self.win.btn_cal.setToolTip(
            "Apply an intensity calibration (wavelength/factor text file) to "
            "the spectrometer — files live in calibration_files/ next to the "
            "program")
        self.win.btn_cal.setMenu(menu)
        return self.win.btn_cal

    @staticmethod
    def _reset_menu(menu):
        """Empty `menu` for repopulation, submenus and action groups included.

        QMenu.clear() only deletes the actions the menu itself owns. A submenu
        built by addMenu() is a child WIDGET whose menuAction() belongs to the
        submenu, and a QActionGroup is not an action at all — so neither is
        touched, and both stay children of a menu that lives for the whole
        session. These menus repopulate on every aboutToShow, so without this
        each open permanently added another set (along with the actions and
        lambdas parented to them, some of which capture a spectrometer that has
        since been disconnected).
        """
        # One call per type: PySide6's findChildren takes a single type, not the
        # tuple isinstance() would accept.
        for cls in (QMenu, QActionGroup):
            for child in menu.findChildren(cls,
                                           options=Qt.FindDirectChildrenOnly):
                child.setParent(None)
                child.deleteLater()
        menu.clear()

    @staticmethod
    def calibration_files():
        try:
            return sorted(CALIBRATION_DIR.glob("*.txt"),
                          key=lambda p: p.name.lower())
        except OSError:
            return []

    def _populate_calibration_menu(self, menu):
        self._reset_menu(menu)
        if not self.win._have_spec():
            act = QAction("(no spectrometer connected)", menu)
            act.setEnabled(False)
            menu.addAction(act)
            return
        files = self.calibration_files()
        targets = self.win.spec.calibration_targets()
        for spec in targets:
            # Flat list for a single device; one submenu per member of a
            # stitched pair, so each takes its own file.
            dest = menu.addMenu(spec.name) if len(targets) > 1 else menu
            group = QActionGroup(dest)
            group.setExclusive(True)
            act = QAction("None (raw counts)", dest)
            act.setCheckable(True)
            act.setChecked(spec.calibration_name is None)
            act.triggered.connect(
                lambda _=False, s=spec: self._apply_calibration(s, None))
            group.addAction(act); dest.addAction(act)
            for f in files:
                act = QAction(f.stem, dest)
                act.setCheckable(True)
                act.setChecked(spec.calibration_name == f.stem)
                act.triggered.connect(
                    lambda _=False, s=spec, p=f: self._apply_calibration(s, p))
                group.addAction(act); dest.addAction(act)
            if not files:
                act = QAction("(no .txt files in calibration_files)", dest)
                act.setEnabled(False)
                dest.addAction(act)
        menu.addSeparator()
        act_add = QAction("Add new calibration…", menu)
        act_add.triggered.connect(self._add_calibration_file)
        menu.addAction(act_add)

    def _apply_calibration(self, spec, path) -> bool:
        """Assign `path` (or None = raw counts) to one physical spectrometer.

        Returns True only if the device really carries it now. A failure is
        raised to a modal dialog rather than a status message: an unloaded
        calibration leaves the device on raw counts while every label still
        names the file, and that mismatch silently corrupts a whole session's
        data — it must not be allowed to scroll past.
        """
        if self.win.scanning.running():
            self.win.status.showMessage(
                "A scan is running — stop it before changing calibration.", 4000)
            return False
        if spec not in self.win.spec.calibration_targets():
            self.win.status.showMessage(
                "That spectrometer is no longer connected.", 4000)
            return False
        skipped = 0
        with self.device_lock() as ok:
            if not ok:
                self.win.status.showMessage(FEED_BUSY_MSG, 4000)
                return False
            try:
                if path is None:
                    spec.clear_calibration()
                else:
                    skipped = spec.set_calibration(path)
            except Exception as e:
                QMessageBox.warning(
                    self.win, "Calibration not applied",
                    f"{spec.name} could not load '{getattr(path, 'stem', path)}':"
                    f"\n\n{e}\n\nThe spectrometer is still on "
                    f"{spec.calibration_name or 'raw counts'}.")
                return False
        msg = (f"{spec.name}: calibration "
               + (f"'{path.stem}' applied." if path is not None
                  else "removed (raw counts)."))
        if skipped:
            msg += (f"  NOTE: {skipped} malformed line(s) in the file were "
                    f"skipped.")
        self.win.status.showMessage(msg, 8000 if skipped else 4000)
        return True

    def _add_calibration_file(self):
        """Copy a calibration file into calibration_files/ so it shows up in
        the menu — and, for the frozen build, survives next to the .exe."""
        src, _ = QFileDialog.getOpenFileName(
            self.win, "Add calibration file", "",
            "Calibration files (*.txt);;All files (*)")
        if not src:
            return
        src = Path(src)
        try:
            load_calibration_file(src)   # reject unparseable files up front
        except Exception as e:
            QMessageBox.warning(self.win, "Add calibration",
                                f"Not a usable calibration file:\n{e}")
            return
        dest = CALIBRATION_DIR / src.name
        try:
            CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not src.samefile(dest):
                if QMessageBox.question(
                        self.win, "Add calibration",
                        f"{dest.name} already exists in calibration_files — "
                        f"overwrite it?") != QMessageBox.Yes:
                    return
            if not (dest.exists() and src.samefile(dest)):
                shutil.copyfile(src, dest)
        except OSError as e:
            QMessageBox.warning(self.win, "Add calibration",
                                f"Could not copy into {CALIBRATION_DIR}:\n{e}")
            return
        self.win.status.showMessage(
            f"Calibration '{src.stem}' added — assign it from the "
            f"Calibration menu.", 5000)

    # ── Spectrometer slots ───────────────────────────────────────────────────
    # Two slots, always. Slot 1 alone is a single spectrometer; both filled is a
    # stitched pair. The Spectrometer dialog is the only caller — it drives
    # these three, and everything else about the pair (stitch factor, overlap
    # band, per-slot calibration) is unchanged from when this was a menu.
    @contextmanager
    def _slots_rolled_back_on_failure(self):
        """Restore the whole slot table if the connect it wraps fails.

        A snapshot of all three lists, not of the one entry being edited:
        clearing slot 1 promotes slot 2 into it, so an edit can touch every
        field, and a per-field undo would leave the table half-shuffled. Yields
        a one-element list the body puts its (ok, msg) into.
        """
        before = {k: list(v) for k, v in self.win._multi.items()}
        box = []
        yield box
        if box and not box[0][0]:
            self.win._multi = before

    def set_slot(self, slot, serial, label):
        """Point one slot at a device and connect whatever the slots now
        describe. (ok, msg)."""
        if self.win.scanning.running():
            return False, ("A scan is running — stop it before changing "
                           "spectrometers.")
        with self._slots_rolled_back_on_failure() as out:
            self.win._multi["serials"][slot] = serial
            self.win._multi["labels"][slot]  = label
            if all(self.win._multi["serials"]):
                out.append(self._connect_multi_pair())
            elif slot == 0:
                out.append(self._connect_single_slot())
            else:
                # Slot 2 filled with slot 1 still empty. Nothing to connect
                # against yet, so remember it and say so rather than opening
                # this device on its own — slot 1 is the single-device slot,
                # by definition.
                out.append((True, f"Slot 2: {label}. Fill slot 1 to connect "
                                  f"the pair."))
        return out[0]

    def clear_slot(self, slot):
        """Empty one slot and connect whatever is left. (ok, msg)."""
        if self.win.scanning.running():
            return False, ("A scan is running — stop it before changing "
                           "spectrometers.")
        if slot == 0 and self.win._multi["serials"][1] is None:
            return self.disconnect_spectrometer()
        with self._slots_rolled_back_on_failure() as out:
            for key in ("serials", "labels", "cals"):
                self.win._multi[key][slot] = None
            if slot == 0:
                # Slot 1 is the single-device slot, so slot 2 alone is not a
                # configuration. Promote it rather than leave a device named in
                # a slot with nothing driving it.
                for key in ("serials", "labels", "cals"):
                    self.win._multi[key][0] = self.win._multi[key][1]
                    self.win._multi[key][1] = None
                out.append(self._connect_single_slot())
            elif not self.win._pair_live():
                out.append((True, "Slot 2 cleared."))
            else:
                # A live pair loses a member: release both, then reopen slot 1
                # as the single device. Filling slot 2, in reverse.
                out.append(self._connect_single_slot())
        return out[0]

    def _connect_single_slot(self):
        """Open slot 1 alone as the connected spectrometer. (ok, msg).

        The single-device counterpart of _connect_multi_pair, and it releases
        the current handle for the same reason: a vendor SDK cannot open the
        same spectrometer twice, and slot 1 may well be a member of the pair
        being torn down.
        """
        ident = self.win._multi["serials"][0]
        if ident is None:
            return self.disconnect_spectrometer()
        if self.win._have_spec():
            ok, err = self._apply_spectrometer(None)
            if not ok:
                return False, err
        cal = self.win._multi["cals"][0]
        ragged = 0
        try:
            spec = self._open_slot_device(ident)
            if cal is not None:
                ragged = spec.set_calibration(cal)
        except Exception as e:
            return False, f"Spectrometer connect failed: {e}"
        ok, err = self._apply_spectrometer(spec)
        if not ok:
            self._drop(spec)
            return False, err
        self.win.status.showMessage(f"Spectrometer: {spec.name}", 5000)
        msg = f"Spectrometer connected: {spec.name}."
        if ragged:
            msg += (f"\n\nNOTE: {ragged} malformed line(s) were skipped in "
                    f"'{cal.stem}' — check the file.")
        return True, msg

    def select_slot_calibration(self, slot, path):
        member = self.win._multi_members[slot]
        if member is None and slot == 0 and not self.win._pair_live():
            # Single-device mode: slot 1 IS the connected spectrometer, so its
            # calibration can be applied to the live device straight away.
            member = self.win.spec
        if member is not None:
            # Commit the slot ONLY once the device really carries the file.
            # The dialog's combo is drawn from _multi["cals"], so recording it
            # first made the combo claim a calibration that had failed to load
            # and left that device silently on raw counts.
            if not self._apply_calibration(member, path):
                return
            self.win._multi["cals"][slot] = path
            return
        # Not live yet — nothing to verify against, so the file is checked on
        # its own and only then remembered for the connect.
        if path is not None:
            try:
                _wl, _fac, skipped = load_calibration_file(path)
            except Exception as e:
                QMessageBox.warning(
                    self.win, "Calibration not applied",
                    f"'{path.stem}' could not be read:\n\n{e}\n\n"
                    f"Slot {slot + 1} is unchanged.")
                return
        else:
            skipped = 0
        self.win._multi["cals"][slot] = path
        msg = (f"Slot {slot + 1} calibration: "
               f"{path.stem if path else 'none'} — applied when the device "
               f"connects.")
        if skipped:
            msg += f"  NOTE: {skipped} malformed line(s) skipped."
        self.win.status.showMessage(msg, 8000 if skipped else 4000)

    def _connect_multi_pair(self):
        """Open both slot devices and swap in the stitched pair. (ok, msg)."""
        if self.win.scanning.running():
            return False, "A scan is running — stop it before changing hardware."
        s1, s2 = self.win._multi["serials"]
        if s1 == s2:
            return False, ("Both slots point at the same spectrometer — pick "
                           "two different devices.")
        # Release any handle we already hold on one of these devices first: a
        # vendor SDK cannot open the same spectrometer twice.
        if self.win._have_spec():
            ok, err = self._apply_spectrometer(None)
            if not ok:
                return False, err
        opened = []
        ragged = []
        try:
            for serial in (s1, s2):
                opened.append(self._open_slot_device(serial))
            for slot, (spec, cal) in enumerate(zip(opened, self.win._multi["cals"])):
                if cal is not None:
                    if spec.set_calibration(cal):
                        ragged.append(f"slot {slot + 1} ('{cal.stem}')")
            stitched = StitchedSpectrometer(*opened)
        except Exception as e:
            for s in opened:
                self._drop(s)
            return False, f"Stitched connect failed: {e}"
        ok, err = self._apply_spectrometer(stitched)
        if not ok:
            self._drop(stitched)   # disconnects both members
            return False, err
        self.win._multi_members = list(opened)      # slot order, not blue/red
        self.win._stitch_stale = False              # fresh pair, equal exposures
        self.win.feed.reset_saturation()   # slot-ordered lamp tooltips need the mapping
        # Again, now that the slot mapping exists: _apply_spectrometer ran
        # before _multi_members was assigned, so its _sync_multi_ui saw
        # _slot_members() fall back to (blue, red) and would have seeded the
        # S1/S2 spinboxes from the wrong members.
        self.win.appearance._sync_multi_ui()
        wl = stitched.wavelengths
        lo, hi = stitched.overlap_band
        self.win.status.showMessage(f"Spectrometer: {stitched.name}", 5000)
        msg = (f"Stitched pair connected: {stitched.name}, "
               f"{wl[0]:.0f}–{wl[-1]:.0f} nm, overlap band {lo:.1f}–{hi:.1f} "
               f"nm. Use Auto-stitch with light across the overlap to match "
               f"the two devices.")
        if ragged:
            msg += (f"\n\nNOTE: malformed lines were skipped in the "
                    f"calibration for {', '.join(ragged)} — check the file.")
        return True, msg

    def _menu_result(self, fn):
        """Run an (ok, msg) action from a header button; failures pop a message
        box — the canvas header has no inline status label like the dialogs."""
        ok, msg = fn()
        if ok:
            if msg:
                self.win.status.showMessage(msg, 8000)
        else:
            QMessageBox.warning(self.win, "Spectrometer", msg)

    def _usable_member_darks(self):
        """The recorded member darks, or None when they cannot be applied.

        All-or-nothing on purpose: subtracting from one member and not the
        other would put the two on different baselines, which is precisely what
        the overlap comparison — and the stitch fit — exist to measure. The one
        gate every consumer goes through, so the per-spectrometer view, the
        merged view and Auto-stitch can never disagree about whether a dark is
        good.
        """
        darks = self.win.background_members
        frames = getattr(self.win.spec, "last_member_raw", None)   # atomic read
        if darks is None or frames is None or len(darks) != len(frames):
            return None
        if not all(np.shape(d) == np.shape(f) for d, f in zip(darks, frames)):
            return None
        return darks

    def _fit_darks(self):
        """The member darks Auto-stitch fits against, or None.

        Deliberately NOT gated on the Subtract Dark checkbox: that is a display
        preference, whereas the fit is physics. Two members at different
        exposures carry different pedestals, and a fit over signal-plus-pedestal
        returns a factor biased by the difference.
        """
        return self._usable_member_darks()

    def fit_stitch_factor(self):
        if not self.win._pair_live():
            return False, "No stitched pair is connected."
        if self.win.scanning.running():
            return False, "A scan is running — stop it before fitting."
        darks = self._fit_darks()
        with self.device_lock() as ok:
            if not ok:
                return False, FEED_BUSY_MSG
            try:
                factor = self.win.spec.fit_stitch_factor(darks)
            except Exception as e:
                return False, f"Stitch-factor fit failed: {e}"
        self.win._stitch_stale = False
        lo, hi = self.win.spec.overlap_band
        res = self.win.spec.stitch_residual
        quality = ("" if res is None else
                   f" Residual mismatch {res * 100:.1f}% — "
                   + ("the two spectra agree across the band."
                      if res < 0.05 else
                      "one scalar does not reconcile them here; check the "
                      "calibrations or narrow the band."))
        return True, (f"Stitch factor fitted: {factor:.4g} "
                      f"(applied to {self.win.spec.spec1.name}) over "
                      f"{lo:.1f}–{hi:.1f} nm"
                      + ("" if darks is not None else ", no dark subtracted")
                      + f".{quality}")

    def set_stitch_factor(self):
        if not self.win._pair_live():
            return
        val, ok = QInputDialog.getDouble(
            self.win, "Set stitch factor",
            f"Multiplier applied to {self.win.spec.spec1.name} before stitching:",
            self.win.spec.stitch_factor, 1e-6, 1e6, 6)
        if ok:
            self.win.spec.stitch_factor = float(val)   # atomic — no lock needed
            self.win.spec.stitch_residual = None   # hand-set: nothing was measured
            self.win._stitch_stale = False   # the user just said what they want
            # No dark warning: the merged background is derived from the member
            # frames at the current factor, so it follows this change by itself.
            self.win.status.showMessage(f"Stitch factor set to {val:.4g}.", 4000)

    def set_overlap_band(self):
        """Pick the sub-range of the overlap that the fit and blend use."""
        if not self.win._pair_live():
            return
        lo, hi = self.win.spec.overlap_band
        glo, ghi = self.win.spec.geometric_overlap
        new_lo, ok = QInputDialog.getDouble(
            self.win, "Overlap band",
            f"Band START (nm) — the two spectrometers share "
            f"{glo:.1f}–{ghi:.1f} nm.\nEnter {glo:.1f} to reset to the full "
            f"shared range:", lo, glo, ghi, 1)
        if not ok:
            return
        new_hi, ok = QInputDialog.getDouble(
            self.win, "Overlap band", "Band END (nm):", hi, glo, ghi, 1)
        if not ok:
            return
        try:
            lo, hi = self.win.spec.set_band(new_lo, new_hi)
        except Exception as e:
            QMessageBox.warning(self.win, "Overlap band", str(e))
            return
        # Push the band to the canvas now rather than waiting for the next
        # overlay frame, so the shading moves as soon as the dialog closes. The
        # shading only renders in the per-spectrometer view (set_overlap_band
        # gates it on that); the combined curve has already been crossfaded, so
        # there is nothing there for it to mark.
        self.win.canvas.set_overlap_band(lo, hi)
        self.win.status.showMessage(
            f"Overlap band set to {lo:.1f}–{hi:.1f} nm — re-run Auto-stitch "
            f"to fit over it.", 6000)

    def _full_scale(self):
        """Effective detector full scale, in counts, or None if unknown.

        Mirrors FrogScanWorker.run exactly — a typed override beats the
        device's own report — so the lamp and the scan can never be judging
        against different thresholds.
        """
        full = self.win.scan_cfg.saturation_counts
        if full is None:
            full = getattr(self.win.spec, "max_counts", None)
        return full if full else None

    def _slot_members(self):
        """Live stitched members in SLOT order (slot 1 first). Falls back to
        the stitched device's own (blue, red) order when the pair was not
        connected through the multi-spectrometer slots."""
        members = getattr(self.win.spec, "members", None)
        if not members:
            return []
        if (all(m is not None for m in self.win._multi_members)
                and set(map(id, self.win._multi_members)) == set(map(id, members))):
            return list(self.win._multi_members)
        return list(members)

    def _member_full_scale(self, member):
        """A member's effective full scale: the Hardware override, when set,
        applies to every member; otherwise the member's own report."""
        full = self.win.scan_cfg.saturation_counts
        if full is None:
            full = getattr(member, "max_counts", None)
        return full if full else None
