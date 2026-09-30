"""The Simulation window — the ONLY route to a simulated device.

Deliberately separate from the hardware dialogs: nothing that connects bench
hardware can hand back a simulator, and nothing starts connected.
"""

from PySide6.QtWidgets import (QGridLayout, QLabel, QPushButton, QVBoxLayout)

from lillypad.hardware import StitchedSpectrometer
from lillypad.sim import (PULSE_SHAPES, SIM_SPECTROMETERS,
                          SimulatedSpectrometer, sim_spectrometer_pair)
from lillypad.gui.dialogs.base import ConnectDialog
from lillypad.gui.widgets import ComboBox, _hline


class SimulationDialog(ConnectDialog):
    """Run the app against simulated hardware, on purpose.

    Simulation used to be the startup default, which meant a fake spectrum
    could be mistaken for a measurement. It is now only ever reached from here,
    and the choices are the same ones the Hardware dialog used to make
    silently: which beam the simulator is measuring, and whether to stand in
    for the spectrometer, the stage, or a stitched pair.
    """
    def __init__(self, main, parent=None):
        super().__init__(main, parent, "Simulation", width=380)
        lay = QVBoxLayout(self); lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        # ── Simulated beam ───────────────────────────────────────────────────
        # Which pulse the simulated spectrometer is measuring. Changing either
        # box re-makes the simulator on the spot (its wavelength window is
        # derived from the beam, so it cannot just be mutated in place).
        lay.addWidget(self._header("Simulated beam"))
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(0, 1); brow.setColumnStretch(1, 0)
        self.cmb_pulse = ComboBox()
        for key, shape in PULSE_SHAPES.items():
            self.cmb_pulse.addItem(shape["label"], key)
        self.cmb_pulse.setCurrentIndex(
            max(0, self.cmb_pulse.findData(self.state.sim_pulse)))
        self.cmb_pulse.currentIndexChanged.connect(self._on_beam)
        self.cmb_gate = ComboBox()
        self.cmb_gate.addItem("SHG", "shg")
        self.cmb_gate.addItem("PG", "pg")
        self.cmb_gate.setCurrentIndex(
            max(0, self.cmb_gate.findData(self.state.sim_gate)))
        self.cmb_gate.currentIndexChanged.connect(self._on_beam)
        brow.addWidget(self.cmb_pulse, 0, 0); brow.addWidget(self.cmb_gate, 0, 1)
        lay.addLayout(brow)
        self.lbl_beam = QLabel(); self.lbl_beam.setObjectName("dim")
        self.lbl_beam.setWordWrap(True)
        lay.addWidget(self.lbl_beam)

        lay.addWidget(_hline())

        lay.addWidget(self._header("Stand in for"))
        grid = QGridLayout(); grid.setSpacing(6)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        b_spec = QPushButton("Spectrometer"); b_spec.setObjectName("accent")
        b_spec.setToolTip("One simulated spectrometer covering the whole "
                          "simulated signal band")
        b_spec.clicked.connect(lambda: self._do(self.main._use_sim_spectrometer))
        b_stage = QPushButton("Stage"); b_stage.setObjectName("accent")
        b_stage.setToolTip("A 300 mm simulated delay stage")
        b_stage.clicked.connect(lambda: self._do(self.main._use_sim_stage))
        grid.addWidget(b_spec, 0, 0); grid.addWidget(b_stage, 0, 1)
        b_pair = QPushButton("Stitched pair"); b_pair.setObjectName("accent")
        b_pair.setToolTip(
            "Two simulated spectrometers connected as one stitched device — "
            "the only way to exercise multi-spectrometer mode without two on "
            "the bench. Overlapping halves of the band, or the pair of real "
            "instruments the beam names (see the description above)")
        b_pair.clicked.connect(lambda: self._do(self.main._use_sim_pair))
        grid.addWidget(b_pair, 1, 0, 1, 2)
        lay.addLayout(grid)

        lay.addWidget(self.lbl_msg)
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        lay.addWidget(btn_close)

    def _on_beam(self):
        self.state.sim_pulse = self.cmb_pulse.currentData()
        self.state.sim_gate  = self.cmb_gate.currentData()
        spec = self.state.spec
        if isinstance(spec, SimulatedSpectrometer):
            self._do(self.main._use_sim_spectrometer)
        elif (isinstance(spec, StitchedSpectrometer)
              and all(isinstance(m, SimulatedSpectrometer)
                      for m in spec.members)):
            self._do(self.main._use_sim_pair)
        else:
            # Real hardware (or nothing) is connected — remember the choice for
            # the next time a simulator is asked for, rather than swapping the
            # device out from under the operator.
            self.refresh()
            self.lbl_msg.setText("Beam saved — press Spectrometer to use it.")

    def refresh(self):
        desc = PULSE_SHAPES[self.state.sim_pulse]["desc"]
        spec = self.state.spec
        if isinstance(spec, SimulatedSpectrometer):
            desc += (f"\nSuggested scan: ±{spec.suggested_delay_fs:.0f} fs   ·   "
                     f"{spec.wavelengths[0]:.0f}–{spec.wavelengths[-1]:.0f} nm")
        models = sim_spectrometer_pair(self.state.sim_pulse, self.state.sim_gate)
        if models is not None:
            # Named here rather than only in the tooltip: which two instruments
            # Stitched pair is about to stand in for is part of the beam.
            desc += ("\nStitched pair: "
                     + " + ".join(SIM_SPECTROMETERS[m]["name"]
                                  .replace("simulated ", "") for m in models))
        self.lbl_beam.setText(desc)
