"""Connecting spectrometers: enumeration, slots, calibration, stitch band.

Simulated devices are deliberately absent here — they belong to the Simulation
window, so nothing on a bench-hardware dialog can quietly hand back fake counts.
"""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout)

from lillypad.hardware import SEABREEZE_BACKENDS, StitchedSpectrometer
from lillypad.gui.dialogs.base import _ConnectDialog
from lillypad.gui.widgets import ComboBox, _hline


class SpectrometerDialog(_ConnectDialog):
    """Connect one or two spectrometers, and manage the stitch between them.

    Two slots are offered from the start, because a pair is not a separate mode
    to be enabled — it is just the second slot being filled. Slot 1 alone is
    single-spectrometer mode; filling slot 2 reopens both as one stitched
    device; emptying it drops back to slot 1. The old Multi-Spec menu's entire
    contents (per-slot device and calibration, auto/manual stitch, overlap
    band) live here, next to the connection they describe.

    Simulated devices are deliberately absent: they belong to the Simulation
    window, so nothing on a bench-hardware dialog can quietly hand back fake
    data.
    """
    NONE_LABEL = "(none)"

    def __init__(self, main, parent=None):
        super().__init__(main, parent, "Spectrometer", width=430)
        lay = QVBoxLayout(self); lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        hrow = QHBoxLayout(); hrow.setSpacing(6)
        hrow.addWidget(self._header("Spectrometer"), 1)
        self.btn_scan = QPushButton("Rescan")
        self.btn_scan.setObjectName("compact")
        self.btn_scan.setToolTip("Re-enumerate every attached spectrometer")
        self.btn_scan.clicked.connect(self._rescan)
        hrow.addWidget(self.btn_scan, 0)
        lay.addLayout(hrow)

        self.lbl_spec = QLabel(); self.lbl_spec.setObjectName("value")
        self.lbl_spec.setWordWrap(True)
        lay.addWidget(self.lbl_spec)

        # Slot rows. Device on the left, that slot's calibration on the right:
        # a calibration file belongs to ONE physical spectrometer, so putting
        # it on any other row would invite assigning it to the pair.
        grid = QGridLayout(); grid.setSpacing(6)
        grid.setColumnStretch(1, 3); grid.setColumnStretch(2, 2)
        self.cmb_dev = []
        self.cmb_cal = []
        for slot in (0, 1):
            grid.addWidget(QLabel(f"Slot {slot + 1}"), slot, 0)
            dev = ComboBox()
            dev.setToolTip("Which spectrometer this slot holds")
            dev.activated.connect(
                lambda _i, s=slot: self._on_device_picked(s))
            grid.addWidget(dev, slot, 1)
            self.cmb_dev.append(dev)
            cal = ComboBox()
            cal.setToolTip("Intensity calibration applied to this slot's "
                           "device — raw counts when none")
            cal.activated.connect(
                lambda _i, s=slot: self._on_cal_picked(s))
            grid.addWidget(cal, slot, 2)
            self.cmb_cal.append(cal)
        lay.addLayout(grid)
        hint = QLabel("Leave slot 2 empty for single-spectrometer mode. "
                      "Filling it reopens both devices as one stitched "
                      "spectrometer.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Stitching ────────────────────────────────────────────────────────
        # Only meaningful while a pair is live, so the whole block greys out
        # rather than disappearing: it is where you look to find out WHY the
        # seam is wrong, and a block that vanishes gives nowhere to look.
        lay.addWidget(self._header("Stitching"))
        self.lbl_stitch = QLabel("—")
        self.lbl_stitch.setObjectName("dim"); self.lbl_stitch.setWordWrap(True)
        lay.addWidget(self.lbl_stitch)
        srow = QHBoxLayout(); srow.setSpacing(6)
        self.btn_fit = QPushButton("Auto-stitch")
        self.btn_fit.setObjectName("accentcompact")
        self.btn_fit.setToolTip("Fit the stitch factor from one frame — needs "
                                "light across the overlap region")
        self.btn_fit.clicked.connect(
            lambda: self._do(self.main._fit_stitch_factor))
        self.btn_manual = QPushButton("Manual…")
        self.btn_manual.setObjectName("compact")
        self.btn_manual.clicked.connect(self._manual_stitch)
        self.btn_band = QPushButton("Overlap band…")
        self.btn_band.setObjectName("compact")
        self.btn_band.setToolTip(
            "The part of the overlap the stitch factor is fitted over and the "
            "two spectra are crossfaded across — shown shaded green in the "
            "per-spectrometer view")
        self.btn_band.clicked.connect(self._set_band)
        srow.addWidget(self.btn_fit); srow.addWidget(self.btn_manual)
        srow.addWidget(self.btn_band)
        lay.addLayout(srow)

        lay.addWidget(_hline())

        # seabreeze backend switch. pyseabreeze is the default because it is
        # the only backend that knows the newer Ocean Insight models;
        # cseabreeze stays available for devices that only enumerate through
        # the vendor C library. Applied on the next connect.
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(1, 1)
        brow.addWidget(QLabel("Backend"), 0, 0)
        self.cmb_backend = ComboBox()
        for name in SEABREEZE_BACKENDS:
            self.cmb_backend.addItem(name, name)
        self.cmb_backend.setCurrentIndex(
            max(0, self.cmb_backend.findData(self.main.seabreeze_backend)))
        self.cmb_backend.currentIndexChanged.connect(self._on_backend)
        brow.addWidget(self.cmb_backend, 0, 1)
        # Full-scale override. Without this, a spectrometer that does not
        # report `max_intensity` leaves saturation unchecked with no way out
        # from the UI. Blank = trust the device.
        brow.addWidget(QLabel("Full scale"), 1, 0)
        self.edit_full_scale = QLineEdit()
        self.edit_full_scale.setPlaceholderText("auto (from device)")
        self.edit_full_scale.editingFinished.connect(self._on_full_scale)
        brow.addWidget(self.edit_full_scale, 1, 1)
        lay.addLayout(brow)

        lay.addWidget(self.lbl_msg)
        frow = QHBoxLayout(); frow.setSpacing(6)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setObjectName("danger")
        self.btn_disconnect.clicked.connect(self._disconnect)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        frow.addWidget(self.btn_disconnect); frow.addWidget(btn_close)
        lay.addLayout(frow)

        self._devices = []      # last enumeration: [(label, ident), ...]

    # ── Enumeration ──────────────────────────────────────────────────────────
    def open_fresh(self):
        """Opening the window IS the request to scan — that is what the panel's
        Connect button means. Scanned AFTER the base clears the status line, so
        the result of this scan is what stays on it."""
        super().open_fresh()
        self._rescan()

    def _rescan(self):
        self._do(self._scan)

    def _scan(self):
        """Re-enumerate every vendor. (ok, msg), for _do.

        list_spectrometers already merges seabreeze and Avantes into one
        tagged-id list, so one call covers the bench.
        """
        try:
            self._devices = list(self.main._list_spectrometers())
        except Exception as e:
            self._devices = []
            return False, f"Scan failed: {e}"
        n = len(self._devices)
        if n:
            return True, (f"{n} spectrometer{'' if n == 1 else 's'} found — "
                          f"pick one for slot 1.")
        # Nothing found. list_spectrometers hides a vendor whose SDK failed, so
        # say WHICH one failed and why — otherwise a missing AvaSpec driver and
        # an unplugged cable look identical from here.
        msg = ("No spectrometers found. Check the USB cables and the vendor "
               "drivers, then press Rescan.")
        notes = self.main._spectrometer_scan_notes()
        if notes:
            msg += "\n\n" + "\n".join(notes)
        return False, msg

    # ── Slot handlers ────────────────────────────────────────────────────────
    def _on_device_picked(self, slot):
        ident = self.cmb_dev[slot].currentData()
        label = self.cmb_dev[slot].currentText()
        if ident == self.main._multi["serials"][slot]:
            return                       # re-picked what is already there
        if ident is None:
            self._do(lambda: self.main._clear_slot(slot))
        else:
            self._do(lambda: self.main._set_slot(slot, ident, label))

    def _on_cal_picked(self, slot):
        path = self.cmb_cal[slot].currentData()
        self.main._select_slot_calibration(slot, path)
        self._refresh()

    def _disconnect(self):
        self._do(self.main._disconnect_spectrometer)

    # ── Stitching handlers ───────────────────────────────────────────────────
    def _manual_stitch(self):
        self.main._set_stitch_factor()
        self._refresh()

    def _set_band(self):
        self.main._set_overlap_band()
        self._refresh()

    # ── Device-wide settings ─────────────────────────────────────────────────
    def _on_full_scale(self):
        """Apply the typed full-scale override (blank clears it back to auto)."""
        text = self.edit_full_scale.text().strip()
        if not text:
            self.main.scan_cfg.saturation_counts = None
        else:
            try:
                value = float(text)
            except ValueError:
                value = -1.0
            if value <= 0:
                # Refuse rather than silently reverting: a bad value here
                # disables saturation checking, which must not happen quietly.
                self._do(lambda: (False, f"Full scale must be a positive "
                                         f"number of counts (got {text!r})."))
                return
            self.main.scan_cfg.saturation_counts = value
        self.main._reset_saturation()
        self._do(lambda: (True, "Full scale: " + (
            f"{self.main.scan_cfg.saturation_counts:.0f} counts (override)."
            if self.main.scan_cfg.saturation_counts else "auto (from device).")))

    def _on_backend(self):
        backend = self.cmb_backend.currentData()
        if backend == self.main.seabreeze_backend:
            return
        # A device opened through the other backend cannot survive the switch
        # (selecting a backend shuts the previous backend's API down), so
        # release it now rather than severing it mid-use on the next connect.
        live = self.main._live_seabreeze_backend()
        if live is not None and live != backend:
            ok, err = self.main._disconnect_spectrometer()
            if not ok:
                # Refused (scan running / feed busy) — keep the old choice.
                self.cmb_backend.setCurrentIndex(
                    max(0, self.cmb_backend.findData(
                        self.main.seabreeze_backend)))
                self._do(lambda: (False, err))
                return
            self.main.seabreeze_backend = backend
            self._scan()      # the new backend enumerates its own devices
            self._do(lambda: (True, f"Backend: {backend}. The spectrometer "
                                    f"was released — pick it again in slot 1 "
                                    f"to reconnect through it."))
        else:
            self.main.seabreeze_backend = backend
            self._scan()
            self._do(lambda: (True, f"Backend: {backend} — used on the next "
                                    f"connect."))

    # ── Display ──────────────────────────────────────────────────────────────
    def _refresh(self):
        spec = self.main.spec
        self.lbl_spec.setText(spec.name if spec is not None
                              else "Not connected.")
        self.btn_disconnect.setEnabled(spec is not None)
        for slot in (0, 1):
            self._fill_device_combo(slot)
            self._fill_cal_combo(slot)
        reported = getattr(spec, "max_counts", None)
        self.edit_full_scale.setPlaceholderText(
            f"auto — device reports {reported:.0f}" if reported
            else "device reports none — saturation unchecked")
        self._refresh_stitch()

    def _fill_device_combo(self, slot):
        """Rebuild one slot's device list: (none), whatever the slot currently
        holds, and every enumerated device not claimed by the other slot.

        The current choice is re-added even when the last scan missed it — a
        simulated half assigned from the Simulation window never appears in an
        enumeration, and dropping it here would make the combo claim the slot
        was empty while the device was live.
        """
        cmb = self.cmb_dev[slot]
        ident = self.main._multi["serials"][slot]
        label = self.main._multi["labels"][slot]
        other = self.main._multi["serials"][1 - slot]
        with QSignalBlocker(cmb):
            cmb.clear()
            cmb.addItem(self.NONE_LABEL, None)
            known = set()
            if ident is not None:
                cmb.addItem(label or str(ident), ident)
                known.add(ident)
            for model, dev_id in self._devices:
                if dev_id == other or dev_id in known:
                    continue
                # Bare serial in the label: the model name already says which
                # vendor it is, so repeating the tag would only make the combo
                # wider. The full tagged id is what gets stored in the slot.
                cmb.addItem(f"{model} [{dev_id.split(':', 1)[-1]}]", dev_id)
            cmb.setCurrentIndex(max(0, cmb.findData(ident)))

    def _fill_cal_combo(self, slot):
        cmb = self.cmb_cal[slot]
        cal = self.main._multi["cals"][slot]
        with QSignalBlocker(cmb):
            cmb.clear()
            cmb.addItem("No calibration", None)
            current = 0
            for i, f in enumerate(self.main._calibration_files(), start=1):
                cmb.addItem(f.stem, f)
                if cal is not None and f.stem == cal.stem:
                    current = i
            cmb.setCurrentIndex(current)

    def _refresh_stitch(self):
        """The stitch factor, its fit quality and whether it has gone stale.

        The residual is the honest answer to "is one scalar enough for this
        pair?", so it belongs next to the factor rather than in a status
        message that has already scrolled away.
        """
        spec = self.main.spec
        live = isinstance(spec, StitchedSpectrometer)
        for b in (self.btn_fit, self.btn_manual, self.btn_band):
            b.setEnabled(live)
        if not live:
            self.lbl_stitch.setText(
                "Fill both slots to stitch two spectrometers into one.")
            return
        res = spec.stitch_residual
        quality = ("not fitted yet" if res is None
                   else f"mismatch {res * 100:.1f}%")
        stale = ("  ·  STALE — integration times changed"
                 if self.main._stitch_stale else "")
        lo, hi = spec.overlap_band
        glo, ghi = spec.geometric_overlap
        self.lbl_stitch.setText(
            f"Factor {spec.stitch_factor:.4g}  ({quality}){stale}\n"
            f"Overlap band {lo:.1f}–{hi:.1f} nm of {glo:.1f}–{ghi:.1f} nm.")
