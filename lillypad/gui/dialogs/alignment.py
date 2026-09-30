"""Alignment sweep settings: the step size the four-point symmetry check uses."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QGridLayout, QLabel, QPushButton,
                               QVBoxLayout)

from lillypad.gui.widgets import DoubleSpinBox


class AlignmentDialog(QDialog):
    """Settings for alignment mode — the Δ button over the spectrum panel.

    Its own window rather than a side-panel row: the step is dialled in once
    when the geometry changes and then left alone, so it was costing the panel
    a permanent block for a control nobody touches during a measurement.
    """
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Alignment")
        self.setFixedWidth(300)
        lay = QVBoxLayout(self); lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        lay.addWidget(self._hdr("Sweep"))
        row = QGridLayout(); row.setSpacing(8)
        row.addWidget(QLabel("Alignment step"), 0, 0)
        # Half-width of the four-point sweep. Its own box rather than the stage
        # jog step: that one switches to um, and an alignment offset is only
        # ever a delay.
        self.spin_align_step = DoubleSpinBox()
        self.spin_align_step.setDecimals(0)
        self.spin_align_step.setRange(1.0, 100000.0)
        self.spin_align_step.setValue(100.0)
        self.spin_align_step.setSuffix(" fs")
        self.spin_align_step.setToolTip(
            "Half-width x of the alignment sweep: spectra are taken at "
            "−2x, −x, +x and +2x from the current position.")
        row.addWidget(self.spin_align_step, 0, 1)
        lay.addLayout(row)

        hint = QLabel(
            "Press Δ above the spectrum to run the sweep. It measures at −2x, "
            "−x, +x and +2x from where the stage is now and plots "
            "S(+x)−S(−x) and S(+2x)−S(−2x) on their own panel under the "
            "spectrum: a symmetric pulse gives two flat curves on zero. The "
            "stage is left where it started.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)

        btn = QPushButton("Close"); btn.clicked.connect(self.hide)
        lay.addWidget(btn)

    def _hdr(self, text):
        l = QLabel(text); l.setObjectName("hdr")
        return l

    @property
    def step_fs(self):
        """Half-width x of the four-point sweep, in fs."""
        return float(self.spin_align_step.value())

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_()
