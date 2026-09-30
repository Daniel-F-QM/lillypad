"""Acquisition settings: averaging, idle shots, settle time, saturation policy.

Holds no state of its own beyond its widgets — the window reads the values back
off them. That is what step 7 turns into properties.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QGridLayout, QLabel,
                               QPushButton, QVBoxLayout)

from lillypad.gui.widgets import DoubleSpinBox, SpinBox, _hline
from lillypad.scan.model import FrogScanConfig


# ─────────────────────────────────────────────────────────────────────────────
# Pop-up dialogs (Qt.Tool = thin frame, floats above, non-modal)
# ─────────────────────────────────────────────────────────────────────────────
class AcquisitionSettingsDialog(QDialog):
    """Per-point acquisition settings. Values are read live at scan start."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Acquisition Settings")
        self.setFixedWidth(300)
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14, 14, 14, 14)

        grid = QGridLayout(); grid.setSpacing(8)
        grid.addWidget(QLabel("Averages / point"), 0, 0)
        self.spin_avg = SpinBox(); self.spin_avg.setRange(1, 1000); self.spin_avg.setValue(1)
        grid.addWidget(self.spin_avg, 0, 1)
        grid.addWidget(QLabel("Idle shots"), 1, 0)
        self.spin_idle = SpinBox(); self.spin_idle.setRange(0, 1000); self.spin_idle.setValue(0)
        grid.addWidget(self.spin_idle, 1, 1)
        grid.addWidget(QLabel("Wait after move"), 2, 0)
        self.spin_wait = SpinBox(); self.spin_wait.setRange(0, 10000)
        self.spin_wait.setValue(0); self.spin_wait.setSuffix(" ms")
        grid.addWidget(self.spin_wait, 2, 1)
        lay.addLayout(grid)

        hint = QLabel("After each move the stage discards the idle shots, then "
                      "averages the next N frames into one column.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addWidget(_hline())

        # ── Saturation ───────────────────────────────────────────────────────
        lay.addWidget(self._hdr("Saturation"))
        srow = QGridLayout(); srow.setSpacing(8)
        srow.addWidget(QLabel("Threshold"), 0, 0)
        self.spin_sat = DoubleSpinBox()
        self.spin_sat.setRange(50.0, 100.0); self.spin_sat.setDecimals(0)
        self.spin_sat.setSingleStep(1.0); self.spin_sat.setSuffix(" % FS")
        self.spin_sat.setValue(100.0 * FrogScanConfig.saturation_fraction)
        srow.addWidget(self.spin_sat, 0, 1)
        lay.addLayout(srow)
        self.chk_abort_sat = QCheckBox("Abort scan on saturation")
        self.chk_abort_sat.setChecked(FrogScanConfig.abort_on_saturation)
        lay.addWidget(self.chk_abort_sat)

        sat_hint = QLabel("Detectors go nonlinear before they hard-clip, so a "
                          "threshold below 100% is the honest setting. Left "
                          "unchecked, a saturated scan finishes and is saved "
                          "with the clipped columns marked in the .npz.")
        sat_hint.setObjectName("dim"); sat_hint.setWordWrap(True)
        lay.addWidget(sat_hint)

        lay.addWidget(_hline())

        # ── Stage health ─────────────────────────────────────────────────────
        lay.addWidget(self._hdr("Stage"))
        self.chk_abort_stage_fault = QCheckBox(
            "Abort scan on stage fault")
        self.chk_abort_stage_fault.setChecked(FrogScanConfig.abort_on_stage_fault)
        lay.addWidget(self.chk_abort_stage_fault)

        stage_hint = QLabel(
            "A stall, a knob nudge or an unhomed axis means the stage is not "
            "where the program thinks it is, so every later column carries a "
            "delay that never happened. Checked (the default), the scan stops "
            "at the first one. Unchecked, it runs to the end and the affected "
            "columns are marked in the .npz.")
        stage_hint.setObjectName("dim"); stage_hint.setWordWrap(True)
        lay.addWidget(stage_hint)

        lay.addWidget(_hline())

        # ── Delay-extreme reference ──────────────────────────────────────────
        lay.addWidget(self._hdr("Delay-extreme reference"))
        self.chk_ref = QCheckBox("Record a reference frame at each end of the sweep")
        self.chk_ref.setChecked(FrogScanConfig.capture_delay_reference)
        lay.addWidget(self.chk_ref)

        ref_hint = QLabel(
            "Diagnostic only — NOTHING is subtracted from it. The exported "
            "trace is corrected with the dark recorded in the Spectrum panel, "
            "and these two frames are archived in the .npz beside that dark so "
            "a real pedestal can be told from delay-independent scatter after "
            "the fact.\n\n"
            "This used to be called “Background”, sat on the scan "
            "panel, and was what the .dwc and .csv were corrected with — "
            "even though it is measured with the beam exactly as it is, at the "
            "first and last delay. Subtracting it also removed any real signal "
            "still present at the ends of the sweep. Costs two extra "
            "acquisitions.")
        ref_hint.setObjectName("dim"); ref_hint.setWordWrap(True)
        lay.addWidget(ref_hint)

        btn = QPushButton("Close"); btn.clicked.connect(self.hide)
        lay.addWidget(btn)

    def _hdr(self, text):
        l = QLabel(text); l.setObjectName("hdr")
        return l

    # ── What the window reads back ───────────────────────────────────────────
    # Named values rather than widgets. The caller wants "how many averages",
    # not "the spinbox that happens to hold it", and a reader of _start_scan
    # should not have to know which of the seven boxes in this window is which.
    # Read live at scan start, so there is nothing to keep in sync.
    @property
    def n_average(self):
        return int(self.spin_avg.value())

    @property
    def idle_shots(self):
        return int(self.spin_idle.value())

    @property
    def settle_ms(self):
        return int(self.spin_wait.value())

    @property
    def sat_percent(self):
        """Saturation threshold as typed, in percent of full scale."""
        return float(self.spin_sat.value())

    @property
    def abort_on_saturation(self):
        return self.chk_abort_sat.isChecked()

    @property
    def abort_on_stage_fault(self):
        return self.chk_abort_stage_fault.isChecked()

    @property
    def take_delay_reference(self):
        """Whether to record a frame at each end of the sweep. Diagnostic only
        — nothing is ever subtracted from it (see the hint in this window)."""
        return self.chk_ref.isChecked()

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_()
