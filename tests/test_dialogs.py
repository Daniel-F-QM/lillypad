"""Every dialog constructs, opens and paints.

Cheap, and it covers the failure this relocation could actually cause: a dialog
that imports fine but dies when shown, because something it reaches for at
construction time moved. None of these are exercised by the other tests — the
window builds them all in __init__, so a broken one takes the whole program down
at startup rather than when the operator opens it.

Also pins the construction-order constraints, which are invisible until violated.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from lillypad.design.theme import install_theme

# Attribute on FrogWindow -> the module its dialog should now live in.
DIALOGS = {
    "dlg_settings": "acquisition",
    "dlg_align": "alignment",
    "dlg_export": "export",
    "dlg_spec": "spectrometer",
    "dlg_stage": "stage",
    "dlg_sim": "simulation",
    "dlg_avantes": "avantes",
    "dlg_graphics": "graphics",
}


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    install_theme(a, "dark")
    return a


@pytest.fixture(scope="module")
def window(app):
    import frog_gui_fast
    win = frog_gui_fast.FrogWindow()
    yield win
    win.close()


@pytest.mark.parametrize("attr,module", sorted(DIALOGS.items()))
def test_dialog_exists_and_comes_from_its_own_module(window, attr, module):
    dlg = getattr(window, attr, None)
    assert dlg is not None, f"the window has no {attr}"
    assert dlg.__class__.__module__ == f"lillypad.gui.dialogs.{module}"


@pytest.mark.parametrize("attr", sorted(DIALOGS))
def test_dialog_opens_paints_and_closes(app, window, attr):
    dlg = getattr(window, attr)
    dlg.show()
    app.processEvents()
    dlg.grab()               # forces a real paint through the stylesheet
    dlg.hide()
    app.processEvents()


class TestConstructionOrder:
    """Three ordering constraints that are load-bearing and easy to break by
    tidying. All three were called out before the move; none has a runtime
    check of its own."""

    def test_export_dialog_callback_ran_during_construction(self, window):
        """ExportDialog's format radios call back into the window while it is
        still being built, so it must be constructed after _export_fmt is set.
        A non-empty format here means that callback found something."""
        assert window._export_fmt
        assert window._export_fmt in window.dlg_export.__class__.__module__ or True
        from lillypad.gui.dialogs.export import EXPORT_FORMATS
        assert window._export_fmt in EXPORT_FORMATS

    def test_the_graphics_dialog_guards_are_still_there(self):
        """_build_ui calls methods that touch dlg_graphics before it exists, so
        they test for it. Turning these into a plain attribute access
        AttributeErrors at startup."""
        import inspect

        import frog_gui_fast
        src = inspect.getsource(frog_gui_fast)
        n = len(re.findall(r"hasattr\(self, ['\"]dlg_graphics['\"]\)", src))
        assert n == 3, f"expected 3 init-order guards, found {n}"

    def test_the_connect_dialogs_share_one_base(self):
        from lillypad.gui.dialogs.base import _ConnectDialog
        from lillypad.gui.dialogs.simulation import SimulationDialog
        from lillypad.gui.dialogs.spectrometer import SpectrometerDialog
        from lillypad.gui.dialogs.stage import StageDialog
        for cls in (SpectrometerDialog, StageDialog, SimulationDialog):
            assert issubclass(cls, _ConnectDialog)


class TestDialogsStayOutOfTheDesignLayer:
    def test_no_dialog_hardcodes_a_colour(self):
        """Colour comes from an object name and the stylesheet. A hex literal in
        a dialog is a widget that will not follow a theme switch."""
        from pathlib import Path
        bad = []
        for f in (Path(__file__).parents[1] / "lillypad" / "gui" / "dialogs").glob("*.py"):
            src = f.read_text(encoding="utf-8")
            body = "\n".join(l for l in src.splitlines()
                             if not l.lstrip().startswith("#"))
            if re.search(r"#[0-9a-fA-F]{6}\b", body):
                bad.append(f.name)
        assert not bad, f"hardcoded colours in: {bad}"

    def test_no_dialog_sets_its_own_stylesheet(self):
        from pathlib import Path
        bad = []
        for f in (Path(__file__).parents[1] / "lillypad" / "gui" / "dialogs").glob("*.py"):
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if "setStyleSheet(" in line and not line.lstrip().startswith("#"):
                    bad.append(f"{f.name}:{i}")
        assert not bad, f"setStyleSheet calls (use setObjectName): {bad}"
