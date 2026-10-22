"""Writing the finished scan out.

Small, and separate because writing is genuinely its own step: it happens on a
worker thread (a full trace is far too slow for the GUI thread), and the worker
is deliberately NOT parented to the window — it held a whole FrogResult per save
for the life of the session.

The three formats differ in more than their syntax, which is the thing to know
before touching this:

  * **.dwc** and **.csv** carry the dark-corrected trace, and both flip a
    high-to-low scan to ascending delay, because .dwc records only a positive
    increment.
  * **.npz** carries the RAW counts, the dark that would be subtracted, and the
    delay-extreme reference frames if they were taken — in acquisition order.
    Line an archive up against a .dwc by delay VALUE, not by column index.
"""

from functools import partial

from PySide6.QtWidgets import QFileDialog
from lillypad.gui.dialogs.export import EXPORT_FORMATS, TRIMMABLE_FORMATS
from lillypad.gui.workers import ExportWorker


class ExportController:
    """Saving the finished trace, on a thread, in one of three formats."""

    def __init__(self, win):
        self.win = win

    # ── Save ──────────────────────────────────────────────────────────────────
    def set_save_enabled(self, on):
        """Enable or disable both ways in to an export at once — the side
        panel's Save button and the Export window's. They write the same file
        from the same data, so a state that applies to one applies to both."""
        self.win.btn_save.setEnabled(on)
        self.win.dlg_export.btn_save.setEnabled(on)

    def set_format(self, key):
        """Make `key` the active export format, without writing anything.

        The single writer of _export_fmt: the Export window's radios come here,
        and so does a format restored from settings at startup. It writes back
        to the radios as well, so the two can never disagree — that path is
        blocked at the widget (see ExportDialog.show_format) rather than guarded
        here, which keeps this function one-directional.
        """
        _label, suffix, _filt, _writer = EXPORT_FORMATS[key]   # bad key: KeyError
        self.win._export_fmt = key
        self.win.dlg_export.show_format(key)
        self.win.dlg_export.refresh_preview()      # the suffix in it just changed
        self.win.btn_save.setText(f"Save ({suffix})")

    def save_now(self):
        """Write the last scan in the active format.

        Both Save buttons land here, and neither can be pressed without a scan
        (_set_save_enabled), so there is no 'nothing to save yet' branch — that
        message existed because picking a format from the old menu also tried
        to save. The guard below is for the one case the buttons cannot cover:
        a second press while the first export is still running.
        """
        if self.win.result is None:
            return
        if self.win._export_worker is not None and self.win._export_worker.isRunning():
            self.win.status.showMessage("An export is already in progress.", 3000)
            return
        _label, suffix, filt, writer = EXPORT_FORMATS[self.win._export_fmt]
        if (self.win._export_fmt in TRIMMABLE_FORMATS
                and self.win.dlg_export.trim_empty_rows):
            writer = partial(writer, trim_empty_rows=True)
        path, _ = QFileDialog.getSaveFileName(
            self.win, "Save FROG scan", self.win.dlg_export.file_name(suffix), filt)
        if not path:
            return
        if not path.lower().endswith(suffix):
            path += suffix
        # Formatting a full trace takes seconds — write it on a worker thread so
        # the window keeps painting (and the live feed keeps running).
        self.set_save_enabled(False)
        self.win.status.showMessage(f"Saving → {path} …", 0)
        # Deliberately NOT parented to the window: a QThread child of a window
        # that lives for the session is never destroyed, so rebinding
        # self.win._export_worker would leak one thread — and the whole FrogResult
        # it holds — per save. The Python reference below is what keeps it alive
        # for the duration; finished() (queued onto the GUI thread) drops both.
        w = ExportWorker(writer, path, self.win.result)
        w.done.connect(self._on_export_done)
        w.error.connect(self._on_export_error)
        w.finished.connect(w.deleteLater)
        w.finished.connect(self._release_export_worker)
        self.win._export_worker = w
        w.start()

    def _release_export_worker(self):
        self.win._export_worker = None

    def _on_export_done(self, path):
        self._end_export()
        self.win.status.showMessage(f"Saved → {path}", 5000)

    def _on_export_error(self, msg):
        self._end_export()
        self.win.status.showMessage(f"Save failed: {msg}", 8000)

    def _end_export(self):
        # btn_export is left alone: it opens the Export window now rather than
        # writing anything, so there is nothing to lock out while a write runs.
        self.set_save_enabled(self.win.result is not None)
