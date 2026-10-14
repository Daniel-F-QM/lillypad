"""The controllers: that they exist, that they are reachable, and that the
couplings between them are the ones that were there before.

A relocation of 4,000 lines has two failure modes that no other test in this
suite would notice.

The first is a **reference that did not get repointed** — `self._apply_stage()`
left on the window after the method moved to DeviceController. Python raises that
as an AttributeError only when the line actually runs, which for a device path
means on someone's bench. So the second class here walks every
`.<controller>.<method>` call site in the repository and checks it against the
real class.

The second is a **controller quietly growing state**. These objects are a
grouping of behaviour: the widgets live on the window and the instrument lives in
InstrumentState. A controller that started keeping its own copy of `spec` or its
own dark would be a second source of truth for something the whole program has
to agree about, and it would disagree on the next device swap. So `__init__` is
pinned to exactly one attribute.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

pytest.importorskip("PySide6")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTROLLERS = ROOT / "lillypad" / "gui" / "controllers"

# window attribute -> class name
EXPECTED = {
    "devices": "DeviceController",
    "appearance": "AppearanceController",
    "feed": "LiveFeedController",
    "alignment": "AlignmentController",
    "stage_ctl": "StageController",
    "scanning": "ScanController",
    "export": "ExportController",
    "settings_ctl": "SettingsController",
}
# Two of them are `_ctl`, because `win.stage` is the delay stage and
# `win.settings` is the loaded preferences dict. Shadowing either would be a
# silent bug in the code that reads them, so the controller yields the name.


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from lillypad.design.theme import install_theme
    app = QApplication.instance() or QApplication([])
    install_theme(app, "dark")
    from lillypad.gui.window import FrogWindow
    win = FrogWindow()
    yield win
    win.close()


class TestTheyAreWiredUp:
    @pytest.mark.parametrize("attr,cls", sorted(EXPECTED.items()))
    def test_the_window_holds_one_of_each(self, window, attr, cls):
        obj = getattr(window, attr, None)
        assert obj is not None, f"the window has no .{attr}"
        assert type(obj).__name__ == cls

    @pytest.mark.parametrize("attr", sorted(EXPECTED))
    def test_each_one_points_back_at_the_window(self, window, attr):
        assert getattr(window, attr).win is window

    @pytest.mark.parametrize("attr", sorted(EXPECTED))
    def test_a_controller_holds_no_state_of_its_own(self, window, attr):
        """One attribute, `win`. A controller keeping its own copy of anything
        the window or InstrumentState owns would be a second source of truth for
        it, and would disagree on the next device swap."""
        held = sorted(vars(getattr(window, attr)))
        assert held == ["win"], f"{attr} also holds {held}"

    def test_they_are_constructed_before_the_ui_is_built(self):
        """build_ui connects lambdas straight to controller methods, so the
        controllers have to exist by then. The order is load-bearing and the
        failure is an AttributeError at startup."""
        from lillypad.gui.window import main_window
        src = inspect.getsource(main_window.FrogWindow.__init__)
        built_at = src.index("build_ui(self)")
        for attr, cls in EXPECTED.items():
            made = re.search(rf"(?m)^\s*self\.{attr}\s*=\s*{cls}\(self\)\s*$", src)
            assert made, f"the window never constructs .{attr}"
            assert made.start() < built_at, f"{attr} is built after the UI"

    def test_none_of_them_is_a_qobject(self):
        """Deliberate: no controller declares a signal, so none of them needs a
        metaobject, and one can be built before any widget exists."""
        from PySide6.QtCore import QObject

        from lillypad.gui import controllers
        for cls in EXPECTED.values():
            obj = getattr(controllers, cls)
            assert not issubclass(obj, QObject), cls

    def test_no_controller_declares_a_signal(self):
        offenders = [p.name for p in CONTROLLERS.glob("*.py")
                     if re.search(r"^\s*\w+ = Signal\(", p.read_text(encoding="utf-8"),
                                  re.M)]
        assert not offenders, offenders


def _controller_classes():
    from lillypad.gui import controllers
    return {attr: getattr(controllers, cls) for attr, cls in EXPECTED.items()}


class TestEveryCallSiteResolves:
    """The failure mode a relocation has: a reference left pointing at a method
    that is no longer there. Python only notices when the line runs, which for
    a device path means on the bench."""

    def test_every_controller_call_in_the_repo_exists(self):
        classes = _controller_classes()
        pattern = re.compile(r"\.(" + "|".join(sorted(classes)) +
                             r")\.([A-Za-z_][A-Za-z0-9_]*)")
        files = (sorted((ROOT / "lillypad" / "gui" / "window").glob("*.py"))
                 + sorted((ROOT / "lillypad").rglob("*.py"))
                 + sorted((ROOT / "tests").glob("*.py")))
        missing, checked = [], 0
        for path in files:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                for attr, name in pattern.findall(line):
                    checked += 1
                    if not hasattr(classes[attr], name):
                        missing.append(f"{path.name}:{i}: {attr}.{name}")
        assert not missing, missing
        assert checked > 100, f"only found {checked} call sites — did the regex break?"


class TestTheDialogsCallControllersNotTheWindow:
    """The dialogs drive devices. After the split they should be asking the
    controller that owns them, and the methods they call should have no
    underscore — that is the honest signal that they are a public API."""

    DEVICE_CALLS = ["use_sim_spectrometer", "use_sim_stage", "use_sim_pair",
                    "connect_zaber_stage", "connect_piezo_jena_stage",
                    "disconnect_stage", "disconnect_spectrometer",
                    "list_spectrometers", "set_slot", "clear_slot",
                    "fit_stitch_factor", "set_overlap_band", "device_lock"]

    @pytest.mark.parametrize("name", DEVICE_CALLS)
    def test_the_public_device_api_exists_under_that_name(self, name):
        from lillypad.gui.controllers import DeviceController
        assert hasattr(DeviceController, name)

    def test_no_dialog_calls_a_window_private_that_moved(self):
        """`self.main._foo` for anything that is now a controller's."""
        classes = _controller_classes()
        owned = {n for cls in classes.values() for n in dir(cls)}
        offenders = []
        for path in (ROOT / "lillypad" / "gui" / "dialogs").glob("*.py"):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for name in re.findall(r"self\.main\.(_[A-Za-z0-9_]+)", line):
                    if name in owned:
                        offenders.append(f"{path.name}:{i}: {name}")
        assert not offenders, offenders


class TestTheCouplingsAreVisible:
    """A controller reaching another one is not a smell here — a device swap
    genuinely has to re-arm the lamp and repaint the buttons. But it should be
    READABLE, which is what `self.win.<other>.<call>` gives. This records which
    ones exist, so that a new one is a deliberate act rather than a surprise."""

    # window attribute -> module file that defines it
    MODULE_OF = {"devices": "devices", "appearance": "appearance",
                 "feed": "livefeed", "alignment": "alignment",
                 "stage_ctl": "stage", "scanning": "scanning",
                 "export": "export", "settings_ctl": "persistence"}

    def test_no_controller_reaches_itself_through_the_window(self):
        """Going out to the window and back into the controller you are already
        in would work, and would mean a method was left on the wrong side of
        the split."""
        offenders = []
        for path in CONTROLLERS.glob("*.py"):
            if path.name == "__init__.py":
                continue
            src = path.read_text(encoding="utf-8")
            for attr, mod in self.MODULE_OF.items():
                if mod == path.stem and re.search(rf"self\.win\.{attr}\.", src):
                    offenders.append(f"{path.name} reaches .{attr}")
        assert not offenders, offenders

    def test_every_cross_controller_call_names_a_real_controller(self):
        """`self.win.<attr>.` where <attr> is not one of the eight is a typo
        that only fails when the line runs."""
        classes = _controller_classes()
        found = set()
        for path in CONTROLLERS.glob("*.py"):
            src = path.read_text(encoding="utf-8")
            found |= set(re.findall(r"self\.win\.([a-z_]+)\.[a-z_]", src))
        # Only the names that look like controllers are checked — the rest are
        # window attributes (canvas, status, btn_feed, _feed, dlg_spec, …).
        suspects = {n for n in found if n.endswith("_ctl") or n in classes}
        assert suspects <= set(classes), sorted(suspects - set(classes))


class TestNoUnreachableImports:
    """A controller that imports something it does not use is a leftover from
    the move, and the one that USES something it did not import is an
    AttributeError the moment that branch runs."""

    @pytest.mark.parametrize("name", sorted(p.name for p in CONTROLLERS.glob("*.py")))
    def test_every_name_used_is_bound(self, name):
        src = (CONTROLLERS / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        bound = set(dir(__builtins__) if isinstance(__builtins__, dict)
                    else dir(__builtins__))
        import builtins
        bound = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    bound.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
                args = node.args
                for a in (args.posonlyargs + args.args + args.kwonlyargs):
                    bound.add(a.arg)
                if args.vararg:
                    bound.add(args.vararg.arg)
                if args.kwarg:
                    bound.add(args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                bound.add(node.name)
            elif isinstance(node, ast.Lambda):
                for a in node.args.args:
                    bound.add(a.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                bound.add(node.id)
        free = sorted({n.id for n in ast.walk(tree)
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                      - bound)
        assert not free, free
