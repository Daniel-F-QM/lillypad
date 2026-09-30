"""Avantes device settings: averaging, ADC resolution, corrections, trigger.

Vendor-specific, and the only dialog that is: these are settings no other
spectrometer has. Writes are debounced, because each one is a USB round trip to
a device the live feed may also be driving.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from lillypad.hardware import avantes_trigger_options
from lillypad.gui.text import FEED_BUSY_MSG
from lillypad.gui.widgets import ComboBox, SpinBox, _hline


# ─────────────────────────────────────────────────────────────────────────────
# Avantes settings dialog
# ─────────────────────────────────────────────────────────────────────────────
class AvantesSettingsDialog(QDialog):
    """Every Avantes-only control, in its own window.

    The Avantes DLL exposes far more than the Ocean adapter does — on-board
    averaging, ADC resolution, dark and prescan correction, smoothing,
    external triggering, sync, board temperature. None of it belongs in the
    Hardware dialog, which has to stay legible on a bench with no Avantes
    attached, so it lives here behind a toolbar button that is HIDDEN unless an
    Avantes is connected (see FrogWindow._refresh_avantes_button).

    None of these device settings are written to settings.json — that file
    holds the app's own preferences, not the spectrometer's state — so
    `refresh` reads the device's current state INTO the widgets. Pushing
    widget defaults the other way would mean that merely opening this dialog
    silently reconfigured the spectrometer.
    """

    # The device's full scale moved (a 14-bit <-> 16-bit ADC switch). Anything
    # that displays or judges against the ceiling has to re-read it.
    full_scale_changed = Signal()

    # Scrubbable controls are debounced through one shared timer: a
    # valueChanged per spinbox tick would park and restart the live feed on
    # every keystroke, the same reason the main integration spinbox debounces.
    WRITE_DEBOUNCE_MS = 250
    POLL_MS = 2000

    def __init__(self, main, parent=None):
        super().__init__(parent, Qt.Tool)
        self.main = main
        self.setWindowTitle("Avantes Settings")

        outer = QVBoxLayout(self); outer.setSpacing(8)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        lay = QVBoxLayout(body); lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        # Which device these controls act on. Hidden with only one Avantes
        # connected, which is the normal case; a two-Avantes stitched pair is
        # the reason it exists at all.
        self.cmb_target = ComboBox()
        self.cmb_target.currentIndexChanged.connect(lambda _=0: self.refresh())
        lay.addWidget(self.cmb_target)

        # ── Acquisition ─────────────────────────────────────────────────────
        lay.addWidget(self._header("Acquisition"))
        arow = QGridLayout(); arow.setSpacing(6)
        arow.setColumnStretch(0, 0); arow.setColumnStretch(1, 1)
        arow.addWidget(QLabel("On-board averages"), 0, 0)
        self.spin_onboard_avg = SpinBox()
        self.spin_onboard_avg.setRange(1, 10000)
        self.spin_onboard_avg.valueChanged.connect(self._queue_write)
        arow.addWidget(self.spin_onboard_avg, 0, 1)
        arow.addWidget(QLabel("Smoothing (± pixels)"), 1, 0)
        self.spin_smooth = SpinBox()
        self.spin_smooth.setRange(0, 128)
        self.spin_smooth.valueChanged.connect(self._queue_write)
        arow.addWidget(self.spin_smooth, 1, 1)
        lay.addLayout(arow)
        self.lbl_avg_hint = QLabel(); self.lbl_avg_hint.setObjectName("dim")
        self.lbl_avg_hint.setWordWrap(True)
        lay.addWidget(self.lbl_avg_hint)

        self.chk_hires = QCheckBox("16-bit ADC (high resolution)")
        self.chk_hires.toggled.connect(self._on_hires)
        lay.addWidget(self.chk_hires)
        self.lbl_hires = QLabel(); self.lbl_hires.setObjectName("dim")
        self.lbl_hires.setWordWrap(True)
        lay.addWidget(self.lbl_hires)

        lay.addWidget(_hline())

        # ── Corrections ─────────────────────────────────────────────────────
        lay.addWidget(self._header("Corrections"))
        self.chk_dark = QCheckBox("Dynamic dark correction")
        self.chk_dark.toggled.connect(
            lambda on: self._write("Dark correction",
                                   lambda d: d.set_dark_correction(on),
                                   voids_dark=True))
        lay.addWidget(self.chk_dark)
        self.chk_prescan = QCheckBox("Prescan (discard the first scan)")
        self.chk_prescan.toggled.connect(
            lambda on: self._write("Prescan", lambda d: d.set_prescan(on),
                                   voids_dark=True))
        lay.addWidget(self.chk_prescan)
        hint = QLabel("These change what a frame contains: a recorded dark is "
                      "discarded and any fitted stitch factor goes stale — "
                      "re-record and re-fit after changing them.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Trigger & sync ──────────────────────────────────────────────────
        lay.addWidget(self._header("Trigger & sync"))
        trow = QGridLayout(); trow.setSpacing(6)
        trow.setColumnStretch(0, 0); trow.setColumnStretch(1, 1)
        self.cmb_trig_mode = ComboBox()
        self.cmb_trig_src = ComboBox()
        self.cmb_trig_type = ComboBox()
        for r, (label, cmb) in enumerate((("Mode", self.cmb_trig_mode),
                                          ("Source", self.cmb_trig_src),
                                          ("Edge/level", self.cmb_trig_type))):
            trow.addWidget(QLabel(label), r, 0)
            trow.addWidget(cmb, r, 1)
            cmb.currentIndexChanged.connect(self._on_trigger)
        lay.addLayout(trow)
        self.chk_sync = QCheckBox("Sync mode")
        self.chk_sync.toggled.connect(
            lambda on: self._write("Sync mode",
                                   lambda d: d.set_sync_mode(on)))
        lay.addWidget(self.chk_sync)
        self.lbl_trig = QLabel(); self.lbl_trig.setObjectName("dim")
        self.lbl_trig.setWordWrap(True)
        lay.addWidget(self.lbl_trig)

        lay.addWidget(_hline())

        # ── Detector ────────────────────────────────────────────────────────
        lay.addWidget(self._header("Detector"))
        self.lbl_temp = QLabel("—"); self.lbl_temp.setObjectName("value")
        lay.addWidget(self.lbl_temp)
        trow2 = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_readback)
        self.chk_poll = QCheckBox(f"Auto-refresh ({self.POLL_MS // 1000} s)")
        self.chk_poll.toggled.connect(self._on_poll_toggle)
        trow2.addWidget(btn_refresh); trow2.addWidget(self.chk_poll)
        lay.addLayout(trow2)
        hint = QLabel("Each refresh borrows the spectrometer from the live "
                      "feed for one frame — leave auto-refresh off at long "
                      "integration times.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Device info ─────────────────────────────────────────────────────
        lay.addWidget(self._header("Device"))
        self.lbl_info = QLabel(); self.lbl_info.setObjectName("dim")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)

        self.lbl_msg = QLabel(""); self.lbl_msg.setObjectName("dim")
        self.lbl_msg.setWordWrap(True)
        lay.addWidget(self.lbl_msg)
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(body); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)
        close_row = QHBoxLayout(); close_row.setContentsMargins(14, 0, 14, 14)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        close_row.addWidget(btn_close)
        outer.addLayout(close_row)
        self.setFixedWidth(360)

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.WRITE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._flush_writes)
        self._loading = False        # True while refresh seeds the widgets
        self._target_ids = None      # device list the target combo was built from

    # ── plumbing ────────────────────────────────────────────────────────────
    def _header(self, text):
        l = QLabel(text); l.setObjectName("hdr")
        return l

    def _target(self):
        """The device the controls act on, re-resolved every time.

        Never cached: _apply_spectrometer can disconnect it out from under an
        open dialog, and a stale handle would be a call into a closed device.
        The combo carries the tagged id rather than the device object for the
        same reason — widget data must not keep a closed device alive.
        """
        devices = self.main._avantes_devices()
        if not devices:
            return None
        want = self.cmb_target.currentData()
        for d in devices:
            if getattr(d, "spec_id", None) == want:
                return d
        return devices[0]

    def _sync_target_combo(self, devices):
        """Rebuild the target list only when the devices actually changed.

        An unconditional clear()/addItem() would reset the selection to the
        first device on every refresh — and refresh runs after every write,
        so a setting applied while member 2 was selected would land on member 2
        and then silently snap the UI back to member 1, with the next write
        going to the wrong device.
        """
        ids = [getattr(d, "spec_id", None) or d.name for d in devices]
        if ids == self._target_ids:
            return
        keep = self.cmb_target.currentData()
        self.cmb_target.blockSignals(True)
        self.cmb_target.clear()
        for d, ident in zip(devices, ids):
            self.cmb_target.addItem(d.name, ident)
        if keep is not None:
            self.cmb_target.setCurrentIndex(max(0, self.cmb_target.findData(keep)))
        self.cmb_target.blockSignals(False)
        # Only meaningful with a two-Avantes pair; hidden in the normal case.
        self.cmb_target.setVisible(len(devices) > 1)
        self._target_ids = ids

    def _do(self, ok, msg):
        # objectName + repolish rather than setStyleSheet, so a theme switch
        # restyles this label with everything else.
        self.lbl_msg.setObjectName("ok" if ok else "err")
        self.lbl_msg.style().unpolish(self.lbl_msg)
        self.lbl_msg.style().polish(self.lbl_msg)
        self.lbl_msg.setText(msg)

    def _write(self, label, fn, refresh=True, voids_dark=False):
        """One device write: refused during a scan, taken under the feed
        handover, reported inline.

        The _scan_running() check is not belt-and-braces — _device_lock only
        parks the LIVE FEED, while the scan worker holds the devices
        independently, which is why _apply_spectrometer and _apply_calibration
        both check separately too.

        `voids_dark` marks a setting that changes what a frame CONTAINS — the
        on-board corrections, the averaging, the ADC range. A dark recorded
        before it no longer describes the frames coming out, so it is dropped
        rather than left to be subtracted from data it does not match.
        """
        if self._loading:
            return                      # refresh is seeding widgets, not a user edit
        dev = self._target()
        if dev is None:
            self._do(False, "No Avantes spectrometer is connected.")
            return
        if self.main._scan_running():
            self._do(False, "A scan is running — stop it before changing "
                            "spectrometer settings.")
            return
        with self.main._device_lock() as ok:
            if not ok:
                self._do(False, FEED_BUSY_MSG)
                return
            try:
                fn(dev)
            except Exception as e:
                self._do(False, f"{label} failed: {e}")
                return
        self._do(True, f"{label}: applied.")
        if voids_dark:
            self.main._invalidate_dark(f"{label.lower()} changed")
        if refresh:
            self.refresh()

    def _queue_write(self, _value=None):
        """Debounce a scrubbable control."""
        if not self._loading:
            self._debounce.start()

    def _flush_writes(self):
        """Apply the debounced spinboxes in one device handover."""
        avg, smooth = self.spin_onboard_avg.value(), self.spin_smooth.value()

        def apply(dev):
            dev.set_averages(avg)
            dev.set_smoothing(smooth)

        # voids_dark: on-board averaging changes the pedestal a frame carries
        # along with the signal, and smoothing redistributes it across pixels.
        self._write("Acquisition settings", apply, voids_dark=True)
        # On-board averaging multiplies the frame time, and the main
        # integration spinbox shows what one acquire costs — reseed it, or it
        # disagrees with what the feed is actually pacing to.
        self.main._sync_integration_ui()

    # ── control handlers ────────────────────────────────────────────────────
    def _on_hires(self, on):
        """ADC resolution moves full scale (16383 <-> 65535), so the saturation
        alarm has to be re-armed against the new ceiling — the same pairing
        SpectrometerDialog._on_full_scale does. Without it the lamp and the scan's
        clip test keep judging against the old one."""
        if self._loading:
            return
        result = {}

        def apply(dev):
            result["on"] = dev.set_high_res_adc(on)

        self._write("ADC resolution", apply, refresh=False, voids_dark=True)
        self.main._reset_saturation()
        # Announced rather than reached for. The Spectrometer window shows the
        # device's full scale as its override placeholder, so it has to be told
        # the ceiling moved — but this dialog has no business knowing that
        # window exists, let alone whether it is open. The window connects them.
        self.full_scale_changed.emit()
        if result.get("on") is False and on:
            self._do(False, "This device has no 16-bit ADC — staying at 14-bit.")
        self.refresh()

    def _on_trigger(self, _idx=None):
        if self._loading:
            return
        mode = self.cmb_trig_mode.currentData()
        src = self.cmb_trig_src.currentData()
        typ = self.cmb_trig_type.currentData()
        # An armed hardware trigger makes acquire() block until a pulse
        # arrives. The live feed would then never return a frame, pause() would
        # time out, and every later action would fail with FEED_BUSY_MSG and no
        # visible cause. Stop the feed for the operator and say so.
        stopped = False
        opts = avantes_trigger_options()
        hw = dict(opts["mode"]).get("Hardware trigger")
        if mode == hw and self.main.feed_enabled:
            self.main.set_feed_enabled(False)
            stopped = True
        self._write("Trigger",
                    lambda d: d.set_trigger(mode, src, typ))
        if stopped:
            self._do(True, "Trigger: applied. The live feed was stopped — "
                           "under a hardware trigger a frame only arrives when "
                           "the experiment fires.")

    def _on_poll_toggle(self, on):
        if on:
            self._timer.start()
        else:
            self._timer.stop()

    def _poll(self):
        """Timed readback. Silent on failure: a busy feed must not turn the
        dialog into a blinking error, and the scan worker owns the device
        outright while a scan runs."""
        if not self.isVisible() or self.main._scan_running():
            return
        self._refresh_readback(quiet=True)

    def _refresh_readback(self, quiet=False):
        """Read the board temperature. A device call, so it takes the feed
        handover like any other."""
        dev = self._target()
        if dev is None or self.main._scan_running():
            return
        with self.main._device_lock() as ok:
            if not ok:
                if not quiet:
                    self._do(False, FEED_BUSY_MSG)
                return
            try:
                temp = dev.temperature_c()
            except Exception as e:
                if not quiet:
                    self._do(False, f"Temperature read failed: {e}")
                return
        self.lbl_temp.setText("Board temperature: " +
                              ("not reported by this device" if temp is None
                               else f"{temp:.1f} °C"))

    # ── state ───────────────────────────────────────────────────────────────
    def refresh(self):
        """Seed every widget from the device. `_loading` suppresses the write
        handlers throughout, so reading the state cannot write it back."""
        devices = self.main._avantes_devices()
        self._loading = True
        try:
            self._sync_target_combo(devices)
            dev = self._target()
            for w in (self.spin_onboard_avg, self.spin_smooth, self.chk_hires,
                      self.chk_dark, self.chk_prescan, self.chk_sync,
                      self.cmb_trig_mode, self.cmb_trig_src,
                      self.cmb_trig_type):
                w.setEnabled(dev is not None)
            if dev is None:
                self.lbl_info.setText("No Avantes spectrometer is connected.")
                return

            self.spin_onboard_avg.setValue(dev.n_averages)
            self.spin_smooth.setValue(dev.smoothing_pixels)
            self.chk_hires.setChecked(dev.high_res_adc)
            self.chk_dark.setChecked(dev.dark_correction)

            opts = avantes_trigger_options()
            mode, src, typ = dev.trigger
            for cmb, key, value in ((self.cmb_trig_mode, "mode", mode),
                                    (self.cmb_trig_src, "source", src),
                                    (self.cmb_trig_type, "source_type", typ)):
                if cmb.count() == 0:
                    for label, val in opts[key]:
                        cmb.addItem(label, val)
                cmb.setCurrentIndex(max(0, cmb.findData(value)))

            n_avg = dev.n_averages
            self.lbl_avg_hint.setText(
                f"One frame costs {dev.integration_ms:g} ms "
                f"({dev.exposure_ms:g} ms exposure × {n_avg} "
                f"{'average' if n_avg == 1 else 'averages'}). The scan also "
                f"averages in software — leave this at 1 unless you want both.")
            self.lbl_hires.setText(
                f"Full scale {dev.max_counts:.0f} counts. Switching resolution "
                f"rescales the counts and re-arms the saturation alarm.")
            self.lbl_trig.setText(
                "Hardware trigger: a frame arrives only when the trigger "
                "input fires, so the live feed is stopped while it is armed."
                if dev.hardware_triggered else
                "Free running — the device clocks its own scans.")

            info = dev.device_info()
            wl_lo, wl_hi = info["wavelength_range_nm"]
            lines = [f"{info['model']}  [{info['serial']}]",
                     f"{info['board']} board · {info['n_pixels']} pixels · "
                     f"{wl_lo:.1f}–{wl_hi:.1f} nm",
                     f"firmware {info.get('firmware') or '?'} · "
                     f"FPGA {info.get('fpga') or '?'} · "
                     f"DLL {info.get('dll') or '?'}"]
            if info.get("detector"):
                lines.append(f"detector {info['detector']}")
            if not info.get("config_verified"):
                # Honest rather than silent: the EEPROM block did not match the
                # layout this build expects, so anything read from it is
                # suspect. Nothing above depends on it, which is why this is a
                # note and not a failure.
                lines.append("note: the device configuration block did not "
                             "match the expected layout, so EEPROM-derived "
                             "details are omitted.")
            self.lbl_info.setText("\n".join(lines))
        finally:
            self._loading = False

    # ── lifecycle ───────────────────────────────────────────────────────────
    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.refresh(); self.lbl_msg.setText(""); self.show(); self.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self._refresh_readback(quiet=True)
        if self.chk_poll.isChecked():
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()          # never poll a hidden dialog
        super().hideEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
