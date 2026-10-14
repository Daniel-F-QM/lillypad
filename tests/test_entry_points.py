"""The ways in, and the layout they depend on.

Three of these are the kind of thing that only fails for someone else: a
launcher that imports but does not run, a package that works from a checkout
and not from `-m`, a spec pointing at a file that no longer exists. They are
cheap to check and expensive to discover.

The last class is the trap that decided the entry point's name. A `lillypad.py`
module and a `lillypad/` package as siblings shadow each other, and Windows'
filesystem is case-insensitive, so `Lillypad.py` does not escape it either.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args, **kw):
    import os
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=180, **kw)


class TestTheLauncher:
    def test_main_py_exists_and_is_tiny(self):
        """It is a launcher. The startup sequence is in lillypad/app/main.py,
        where the order it has to happen in can be explained next to it."""
        src = (ROOT / "main.py").read_text(encoding="utf-8")
        code = [l for l in src.splitlines()
                if l.strip() and not l.lstrip().startswith("#")]
        # docstring + import + the __main__ guard
        assert len(code) < 20, f"main.py has grown to {len(code)} lines"

    def test_both_entry_points_call_the_same_function(self):
        """`main.py` and `python -m lillypad` must not diverge."""
        from lillypad.app.main import main
        for name in ("main.py", "lillypad/__main__.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            assert "from lillypad.app.main import main" in src, name
            assert "main()" in src, name
        assert callable(main)

    def test_the_app_package_does_not_shadow_its_own_module(self):
        """A `main` re-export in lillypad/app/__init__.py would bind the
        FUNCTION onto the package as the attribute `main`, so
        `import lillypad.app.main` and `from lillypad.app import main` would
        hand back two different objects."""
        import importlib
        import lillypad.app
        mod = importlib.import_module("lillypad.app.main")
        assert lillypad.app.main is mod

    @pytest.mark.parametrize("args", [("main.py",), ("-m", "lillypad")])
    def test_it_starts_far_enough_to_build_a_window(self, args):
        """Driven as a subprocess up to the event loop and no further: the
        QApplication, the theme, the calibration seeding and the whole window
        all happen before app.exec(), which is where everything that can go
        wrong at startup goes wrong."""
        probe = (
            "import sys, runpy\n"
            "from PySide6.QtCore import QTimer\n"
            "from PySide6.QtWidgets import QApplication\n"
            "_exec = QApplication.exec\n"
            "def _stop(self=None):\n"
            "    print('WINDOW UP', file=sys.stderr)\n"
            "    return 0\n"
            "QApplication.exec = _stop\n"
            "sys.exit = lambda *a: None\n"
        )
        if args[0] == "-m":
            probe += "runpy.run_module('lillypad', run_name='__main__')\n"
        else:
            probe += "runpy.run_path('main.py', run_name='__main__')\n"
        r = run("-c", probe)
        assert "WINDOW UP" in r.stderr, r.stderr[-3000:]


class TestTheShimsAreGone:
    @pytest.mark.parametrize("name", ["frog_gui_fast.py", "hardware.py",
                                      "scan.py", "avantes.py"])
    def test_the_flat_module_is_gone(self, name):
        assert not (ROOT / name).exists(), f"{name} came back"

    @pytest.mark.parametrize("name", ["frog_gui_fast", "hardware", "scan"])
    def test_importing_it_fails(self, name):
        """A leftover import anywhere would resolve to a stale copy on someone's
        machine and quietly use it."""
        r = run("-c", f"import {name}")
        assert r.returncode != 0
        assert "ModuleNotFoundError" in r.stderr

    def test_nothing_in_the_repo_still_imports_them(self):
        import re
        pattern = re.compile(r"^\s*(?:from (frog_gui_fast|hardware|scan|avantes)"
                             r" import|import (frog_gui_fast|hardware|scan|avantes)\b)",
                             re.M)
        offenders = []
        for path in list(ROOT.glob("*.py")) + \
                sorted((ROOT / "lillypad").rglob("*.py")) + \
                sorted((ROOT / "tests").rglob("*.py")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.match(line):
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
        assert not offenders, offenders


class TestTheEntryPointCannotShadowThePackage:
    def test_there_is_no_module_named_like_the_package(self):
        """A `lillypad.py` beside `lillypad/` shadow each other, and the
        filesystem here is case-insensitive, so `Lillypad.py` is the same
        collision. This is why the launcher is main.py."""
        clashes = [p.name for p in ROOT.glob("*.py")
                   if p.stem.lower() == "lillypad"]
        assert not clashes, clashes

    def test_the_spec_builds_from_the_launcher(self):
        spec = (ROOT / "Lillypad.spec").read_text(encoding="utf-8")
        assert "Analysis(\n    ['main.py']" in spec
        assert "frog_gui_fast" not in spec

    def test_the_spec_still_bundles_what_the_app_resolves_at_runtime(self):
        """Each of these is asked for by one `resource_path()` call, and a
        missing one degrades rather than crashing — an unstyled window, an
        empty calibration menu — which a headless check will not notice."""
        spec = (ROOT / "Lillypad.spec").read_text(encoding="utf-8")
        assert "('icons', 'icons')" in spec
        assert "('lillypad/design/qss', 'lillypad/design/qss')" in spec
        assert "calibration_files/*.txt" in spec

    def test_the_diagnostics_tool_still_runs_from_the_root(self):
        """It stays at the repo root rather than moving to tools/, because that
        is what puts `lillypad` on sys.path for it."""
        r = run("zaber_diagnostics.py", "--help")
        assert r.returncode == 0, r.stderr[-2000:]
