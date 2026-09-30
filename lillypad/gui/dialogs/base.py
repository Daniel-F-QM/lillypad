"""The shared dialog base, and the modal device picker.

_ConnectDialog is what the three connection dialogs are built on: a header, an
error-reporting wrapper, and the open/toggle behaviour they share. It takes
`main` and `parent` separately — `main` is the window it drives, `parent` is the
Qt parent — which is the author's existing (partial) answer to which way the
dependency runs.

DevicePickerDialog is modal and shown only when enumeration finds more than one
device, so it is the one dialog here that asks a question and returns.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout)

from lillypad.gui.widgets import ComboBox


# ─────────────────────────────────────────────────────────────────────────────
# Connection dialogs
# ─────────────────────────────────────────────────────────────────────────────
class _ConnectDialog(QDialog):
    """Shared plumbing for the three connection windows.

    They all do the same three things: run an (ok, msg) action and colour the
    result inline, draw a small section header, and refresh themselves whenever
    they are shown. Only `_refresh` differs, so only `_refresh` is overridden.
    """
    def __init__(self, main, parent=None, title="", width=380):
        super().__init__(parent, Qt.Tool)
        self.main = main
        self.setWindowTitle(title)
        self.setFixedWidth(width)
        # Created here rather than in each subclass: _do() writes to it, and
        # every subclass has to place it somewhere in its own layout.
        self.lbl_msg = QLabel("")
        self.lbl_msg.setObjectName("dim")
        self.lbl_msg.setWordWrap(True)

    @staticmethod
    def _header(text):
        l = QLabel(text)
        l.setObjectName("hdr")
        return l

    def _do(self, fn):
        """Run an (ok, msg) action and report it in lbl_msg."""
        ok, msg = fn()
        # Color via objectName + repolish (not setStyleSheet) so a later theme
        # switch restyles the label along with everything else.
        self.lbl_msg.setObjectName("ok" if ok else "err")
        self.lbl_msg.style().unpolish(self.lbl_msg)
        self.lbl_msg.style().polish(self.lbl_msg)
        self.lbl_msg.setText(msg)
        self._refresh()
        self._refit()

    def _refit(self):
        """Grow the window to whatever the contents now need.

        These dialogs are full of word-wrapped labels — a device name, a status
        line, a stitch summary — that go from one line to four as devices come
        and go, and Qt will not re-grow an already-shown window for a wrapped
        label on its own. The width is fixed, so this only ever changes height.
        """
        self.layout().activate()
        self.resize(self.width(), self.sizeHint().height())

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.open_fresh()

    def open_fresh(self):
        """Show the dialog with a clean status line and current contents.

        This — not toggle() — is what the panel Connect buttons call: pressing
        "Connect Spectrometer" must always open the window, never close one
        that happens to be up already.
        """
        self.lbl_msg.setText("")
        self._refresh()
        self._refit()
        self.show()
        self.raise_()
        self.activateWindow()

    def _refresh(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Spectrometer picker (modal — shown only when enumeration finds 2+ devices)
# ─────────────────────────────────────────────────────────────────────────────
class DevicePickerDialog(QDialog):
    """Pick one device from an enumeration. `devices` is [(label, id), ...] —
    the shape both hardware.list_*() helpers return — and the chosen id comes
    back from pick()."""
    def __init__(self, devices, parent=None, title="Select Device", prompt=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(360)
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(14, 14, 14, 14)
        lbl = QLabel(prompt or f"{len(devices)} devices found — choose one:")
        lbl.setObjectName("dim")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self.cmb = ComboBox()
        for label, ident in devices:
            self.cmb.addItem(f"{label}  [{ident}]", ident)
        lay.addWidget(self.cmb)
        row = QHBoxLayout()
        b_cancel = QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        b_ok = QPushButton("Connect"); b_ok.setObjectName("accent")
        b_ok.setDefault(True); b_ok.clicked.connect(self.accept)
        row.addWidget(b_cancel); row.addWidget(b_ok)
        lay.addLayout(row)

    @staticmethod
    def pick(parent, devices, title="Select Device", prompt=""):
        """Returns the chosen id, or None on cancel."""
        dlg = DevicePickerDialog(devices, parent, title, prompt)
        return dlg.cmb.currentData() if dlg.exec() == QDialog.Accepted else None
