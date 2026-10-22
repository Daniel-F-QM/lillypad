"""The Export window: format, file name, and one Save button.

EXPORT_FORMATS lives here because the table IS what this dialog presents. The
window imports it back for the actual save; it moves to the export controller in
step 9.
"""

import time

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QCheckBox, QDialog, QLabel, QLineEdit,
                               QPushButton, QRadioButton, QVBoxLayout)

from lillypad.gui.widgets import _hline
from lillypad.scan import write_csv, write_dwc, write_npz


# Export formats offered by the Export window: key -> (label, suffix,
# file-dialog filter, writer(path, result)). .dwc is the default; only the
# npz carries the raw counts, background frames and metadata.
EXPORT_FORMATS = {
    "dwc": ("FROG trace  (.dwc)",    ".dwc", "FROG trace (*.dwc)",    write_dwc),
    "npz": ("NumPy archive  (.npz)", ".npz", "NumPy archive (*.npz)", write_npz),
    "csv": ("CSV table  (.csv)",     ".csv", "CSV table (*.csv)",     write_csv),
}

# The formats the "trim empty rows" option applies to. Not the .npz: the
# archive keeps every pixel, so it always lines up with the raw device.
TRIMMABLE_FORMATS = {"dwc", "csv"}

# Stem the Save dialog opens on until someone types their own. Deliberately not
# a date and not a counter: the date is the checkbox's job below, and a counter
# would have to be kept per folder to mean anything.
DEFAULT_EXPORT_NAME = "frog_scan"


class ExportDialog(QDialog):
    """Format, file name and the one button that writes the last scan.

    A window rather than the drop-down menu it replaces. That menu could offer
    exactly one thing — a format — and it both SET the format and wrote the file
    on the same click, so there was nowhere to put a name, and no way to change
    your mind about the format without writing a file to find out. Here the
    format is a choice you leave sitting, the name is editable and shown as it
    will land, and exactly one button writes anything.

    Non-modal (Qt.Tool) like the other pop-ups: a file name is usually typed
    against something on screen — the scan it is for.
    """

    # Characters Windows will not put in a file name. Replaced rather than
    # rejected: this field feeds a Save dialog's DEFAULT, and a dialog opened on
    # a name it cannot use is worse than one opened on a lightly-corrected one.
    # A separator is in the list on purpose — "scan/2" naming a subfolder that
    # may not exist is not what anyone means by it.
    _ILLEGAL = '<>:"/\\|?*'

    def __init__(self, win):
        super().__init__(win, Qt.Tool)
        self.win = win
        self.setWindowTitle("Export")
        self.setFixedWidth(330)
        lay = QVBoxLayout(self); lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        lay.addWidget(self._hdr("Format"))
        # Auto-exclusive by sharing this dialog as their parent, so no
        # QButtonGroup is needed; the dict is how _set_export_fmt finds the one
        # to tick when the format is set from somewhere else (startup, mainly).
        self.radios = {}
        for key, (label, _suffix, _filt, _writer) in EXPORT_FORMATS.items():
            rb = QRadioButton(label)
            # `on and …`: a click also un-toggles the radio that HELD the
            # choice, and that second signal must not report the old format as
            # a new one.
            rb.toggled.connect(lambda on, k=key: on and win.export.set_format(k))
            lay.addWidget(rb)
            self.radios[key] = rb
        hint = QLabel(
            "Only the .npz keeps the raw counts, both background frames and the "
            "metadata. The two text formats have fixed layouts that cannot "
            "carry the extras, so they get the background-subtracted trace.")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        lay.addWidget(hint)
        self.chk_trim = QCheckBox("Trim empty wavelength rows")
        self.chk_trim.setChecked(True)
        self.chk_trim.setToolTip(
            "Leave out the wavelength rows at either end of the trace that are "
            "zero at every delay. They come from a calibration file whose "
            "factor is 0 there, and carry no data. Rows in between are always "
            "kept. Applies to .dwc and .csv; the .npz always keeps every pixel.")
        lay.addWidget(self.chk_trim)

        lay.addWidget(_hline())

        lay.addWidget(self._hdr("File name"))
        self.edit_name = QLineEdit(DEFAULT_EXPORT_NAME)
        self.edit_name.setPlaceholderText(DEFAULT_EXPORT_NAME)
        self.edit_name.setToolTip(
            "The name the Save dialog opens on — for this window, for the side "
            "panel's Save, and for the toolbar's camera, which puts .png on the "
            "same stem. The suffix comes from the format above, and the Save "
            "dialog can still be typed over: this is the starting point, not a "
            "lock.")
        lay.addWidget(self.edit_name)
        self.chk_date = QCheckBox("Date prefix?")
        self.chk_date.setChecked(True)
        self.chk_date.setToolTip(
            "Put today's date in front of the name as YYYY-MM-DD_ . ISO order, "
            "so a folder of exports sorts into chronological order by name.")
        lay.addWidget(self.chk_date)
        # What the two controls above actually produce, spelled out: the date
        # is not in the field being typed into, and the suffix is not in it
        # either, so neither is guessable from the field alone.
        self.lbl_preview = QLabel()
        self.lbl_preview.setObjectName("dim")
        self.lbl_preview.setWordWrap(True)
        lay.addWidget(self.lbl_preview)
        self.edit_name.textChanged.connect(self.refresh_preview)
        self.chk_date.toggled.connect(self.refresh_preview)

        lay.addWidget(_hline())

        # Disabled until there is a scan. The window owns that state for both
        # this button and the side panel's — see FrogWindow._set_save_enabled.
        self.btn_save = QPushButton("Save…")
        self.btn_save.setMinimumHeight(30)
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("Choose where to write the scan")
        self.btn_save.clicked.connect(win.export.save_now)
        lay.addWidget(self.btn_save)
        btn = QPushButton("Close"); btn.clicked.connect(self.hide)
        lay.addWidget(btn)

        # Signals blocked: the window's _export_fmt already says "dwc", and a
        # toggled() reporting it back would reach _set_export_fmt before the
        # side panel's Save button it writes to exists.
        first = next(iter(EXPORT_FORMATS))
        with QSignalBlocker(self.radios[first]):
            self.radios[first].setChecked(True)
        self.refresh_preview()

    def _hdr(self, text):
        l = QLabel(text); l.setObjectName("hdr")
        return l

    @property
    def trim_empty_rows(self):
        """Whether .dwc/.csv exports drop the all-zero rows at either end."""
        return self.chk_trim.isChecked()

    # ── The name ──────────────────────────────────────────────────────────
    @property
    def date_stamped(self):
        """Whether the composed name carries today's date as a prefix."""
        return self.chk_date.isChecked()

    def active_key(self):
        """The format key whose radio is ticked."""
        for key, rb in self.radios.items():
            if rb.isChecked():
                return key
        return next(iter(EXPORT_FORMATS))

    def stem(self):
        """The file name to open the Save dialog on, without its suffix.

        An empty (or all-illegal) field falls back to DEFAULT_EXPORT_NAME
        rather than producing a file called ".dwc" — a name the operator cannot
        see in a folder listing and Windows treats as an extension.
        """
        text = "".join("-" if c in self._ILLEGAL else c
                       for c in self.edit_name.text()).strip(" .")
        if not text:
            text = DEFAULT_EXPORT_NAME
        if self.chk_date.isChecked():
            text = f"{time.strftime('%Y-%m-%d')}_{text}"
        return text

    def file_name(self, suffix=None):
        """The stem with a suffix on it — the active format's unless told."""
        if suffix is None:
            suffix = EXPORT_FORMATS[self.active_key()][1]
        return self.stem() + suffix

    def refresh_preview(self):
        self.lbl_preview.setText(f"Saves as   {self.file_name()}")

    def show_format(self, key):
        """Tick `key`'s radio without reporting it back as a fresh choice —
        this is the path FROM the window, so the window already knows."""
        with QSignalBlocker(self.radios[key]):
            self.radios[key].setChecked(True)
        # Greyed out, not unticked, for the .npz: the choice is kept for when
        # a text format is picked again.
        self.chk_trim.setEnabled(key in TRIMMABLE_FORMATS)

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            # The preview carries a date, and a session can outlive one.
            self.refresh_preview()
            self.show(); self.raise_()
