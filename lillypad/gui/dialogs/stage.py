"""Connecting a delay stage: Kinesis, Zaber or piezosystems Jena."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout)

from lillypad.hardware import list_kinesis_stages
from lillypad.gui.dialogs.base import ConnectDialog
from lillypad.gui.widgets import ComboBox, DoubleSpinBox, _hline


class StageDialog(ConnectDialog):
    """Connect a delay stage, and hold the one setting that belongs to it.

    Kinesis controllers enumerate, so they get a scanned list. Zaber and Piezo
    Jena do not — their adapters auto-scan the serial ports themselves on
    connect — so they keep a Connect button each, with an optional port
    override for a bench where the auto-scan picks the wrong one.
    """
    def __init__(self, main, parent=None):
        super().__init__(main, parent, "Stage", width=400)
        lay = QVBoxLayout(self); lay.setSpacing(8)
        lay.setContentsMargins(14, 14, 14, 14)

        hrow = QHBoxLayout(); hrow.setSpacing(6)
        hrow.addWidget(self._header("Stage"), 1)
        self.btn_scan = QPushButton("Rescan")
        self.btn_scan.setObjectName("compact")
        self.btn_scan.setToolTip("Re-enumerate the attached Kinesis controllers")
        self.btn_scan.clicked.connect(self._rescan)
        hrow.addWidget(self.btn_scan, 0)
        lay.addLayout(hrow)

        self.lbl_stage = QLabel(); self.lbl_stage.setObjectName("value")
        self.lbl_stage.setWordWrap(True)
        lay.addWidget(self.lbl_stage)

        # Two equal columns: the parameter of a row on the left, that row's
        # connect button on the right. Putting each port field on its OWN
        # vendor's row is what stops it reading as a global setting.
        grid = QGridLayout(); grid.setSpacing(6)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        self.cmb_kinesis = ComboBox()
        self.cmb_kinesis.setToolTip("Kinesis controllers found by the last scan")
        b_kin = QPushButton("Connect Kinesis"); b_kin.setObjectName("accent")
        b_kin.clicked.connect(self._connect_kinesis)
        grid.addWidget(self.cmb_kinesis, 0, 0); grid.addWidget(b_kin, 0, 1)

        # Zaber: optional serial port (blank = auto-scan the machine's ports).
        self.edit_zaber_port = QLineEdit()
        self.edit_zaber_port.setPlaceholderText("COM (auto)")
        b_zab = QPushButton("Connect Zaber"); b_zab.setObjectName("accent")
        b_zab.clicked.connect(lambda: self._do(self._connect_zaber))
        grid.addWidget(self.edit_zaber_port, 1, 0); grid.addWidget(b_zab, 1, 1)

        # Piezo Jena: optional serial port (blank = auto-scan), like Zaber.
        self.edit_piezo_port = QLineEdit()
        self.edit_piezo_port.setPlaceholderText("COM (auto)")
        b_pj = QPushButton("Connect Piezo Jena"); b_pj.setObjectName("accent")
        b_pj.clicked.connect(lambda: self._do(self._connect_piezo))
        grid.addWidget(self.edit_piezo_port, 2, 0); grid.addWidget(b_pj, 2, 1)
        lay.addLayout(grid)

        lay.addWidget(_hline())

        # Backlash approach margin. Every move undershoots by this much when it
        # would otherwise arrive from above, so the zero you mark by jogging and
        # the positions a scan sweeps through sit in the same frame. It lives
        # here because it is a property OF THE CONNECTED STAGE — the connect
        # buttons above seed it, and _sync_backlash_ui pushes the new stage's
        # default into this box on every swap.
        brow = QGridLayout(); brow.setSpacing(6)
        brow.setColumnStretch(1, 1)
        brow.addWidget(QLabel("Backlash"), 0, 0)
        self.spin_backlash = DoubleSpinBox()
        self.spin_backlash.setRange(0.0, 1000.0); self.spin_backlash.setDecimals(1)
        self.spin_backlash.setSingleStep(10.0); self.spin_backlash.setSuffix(" um")
        self.spin_backlash.setToolTip(
            "Approach margin. Lead-screw stages land in a different place "
            "depending on which way they arrived; undershooting by more than "
            "the slack and coming back up makes every move repeatable.\n\n"
            "0 disables it — correct for a Thorlabs controller (its firmware "
            "already does this) or a piezo, wrong for a Zaber.")
        self.spin_backlash.valueChanged.connect(self.main._on_backlash_changed)
        brow.addWidget(self.spin_backlash, 0, 1)
        lay.addLayout(brow)
        self.lbl_backlash_fs = QLabel("—"); self.lbl_backlash_fs.setObjectName("dim")
        self.lbl_backlash_fs.setWordWrap(True)
        lay.addWidget(self.lbl_backlash_fs)

        lay.addWidget(self.lbl_msg)
        frow = QHBoxLayout(); frow.setSpacing(6)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setObjectName("danger")
        self.btn_disconnect.clicked.connect(
            lambda: self._do(self.main.devices.disconnect_stage))
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.hide)
        frow.addWidget(self.btn_disconnect); frow.addWidget(btn_close)
        lay.addLayout(frow)

        self._devices = []      # last Kinesis enumeration: [(model, conn), ...]

    def open_fresh(self):
        """Scanned AFTER the base clears the status line, so what the scan
        found is what the window opens showing."""
        super().open_fresh()
        self._rescan()

    def _rescan(self):
        self._do(self._scan)

    def _scan(self):
        """Enumerate the Kinesis controllers. (ok, msg), for _do.

        Each controller is briefly opened to read its model number, so this is
        a real device query, not a list of ports — hence a button rather than
        something done continuously.
        """
        try:
            self._devices = list(list_kinesis_stages())
        except Exception as e:
            self._devices = []
            self._fill_kinesis_combo()
            return False, f"Kinesis scan failed: {e}"
        self._fill_kinesis_combo()
        n = len(self._devices)
        return bool(n), (
            f"{n} Kinesis controller{'' if n == 1 else 's'} found." if n else
            "No Kinesis controllers found. Zaber and Piezo Jena stages cannot "
            "be enumerated — use their own Connect buttons, which scan the "
            "serial ports themselves.")

    def _fill_kinesis_combo(self):
        with QSignalBlocker(self.cmb_kinesis):
            self.cmb_kinesis.clear()
            for model, conn in self._devices:
                self.cmb_kinesis.addItem(f"{model} [{conn}]", conn)
            if not self._devices:
                self.cmb_kinesis.addItem("(none found)", None)
        self.cmb_kinesis.setEnabled(bool(self._devices))

    def _connect_kinesis(self):
        conn = self.cmb_kinesis.currentData()
        if conn is None:
            self._do(lambda: (False, "No Kinesis controller selected — press "
                                     "Rescan."))
            return
        self._do(lambda: self.main.devices.connect_kinesis_stage(conn))

    def _connect_zaber(self):
        port = self.edit_zaber_port.text().strip() or None
        return self.main.devices.connect_zaber_stage(port)

    def _connect_piezo(self):
        port = self.edit_piezo_port.text().strip() or None
        return self.main.devices.connect_piezo_jena_stage(port)

    @property
    def backlash_um(self):
        """The approach margin the connected stage is being driven with."""
        return float(self.spin_backlash.value())

    def refresh(self):
        stage = self.state.stage
        self.lbl_stage.setText(stage.name if stage is not None
                               else "Not connected.")
        self.btn_disconnect.setEnabled(stage is not None)
        self.spin_backlash.setEnabled(stage is not None)
