"""Does the application actually come up, and survive a theme switch?

Every other test in this suite exercises a module. None of them had ever
CONSTRUCTED the window, which is how the design extraction managed to remove
`import matplotlib` from frog_gui_fast while a colormap call still needed it: 354
unit tests passed and the program died on the first FrogCanvas.

So this builds the real FrogWindow offscreen and drives the paths that read the
design layer at construction and paint time. It is slower than the rest of the
suite and worth it — it is the only test that would have caught that.

Nothing here touches hardware: the window opens with nothing connected, which is
the application's actual startup state.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from lillypad.design.theme import install_theme, startup_theme
from lillypad.design.tokens import DARK_PALETTE, LIGHT_PALETTE, PALETTE
from lillypad.settings import load_settings


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(app):
    """One window for the module: constructing it is the expensive part, and the
    startup ORDER is what is under test — theme before the first widget, exactly
    as main() does it."""
    install_theme(app, startup_theme(load_settings()))
    import frog_gui_fast
    win = frog_gui_fast.FrogWindow()
    yield win
    install_theme(app, "dark")
    win.close()


class TestStartup:
    def test_the_theme_installs_before_any_widget_exists(self, app):
        """main() does this first because PALETTE is sampled at CONSTRUCTION time
        by the figure and by _build_ui's per-theme icon files."""
        install_theme(app, "dark")
        assert app.styleSheet().strip()

    def test_the_window_constructs(self, window):
        assert window.canvas is not None

    def test_the_figure_has_its_four_panels(self, window):
        """spectrum, trace, autocorrelation, deviation."""
        assert len(window.canvas.fig.axes) == 4

    def test_nothing_is_connected_at_startup(self, window):
        """The application's real initial state: a real absence, not a stand-in
        device that could quietly return plausible counts."""
        assert window.spec is None
        assert window.stage is None


class TestThemeSwitchOnALiveWindow:
    def test_switching_does_not_rebind_the_palette(self, window, app):
        """The property the whole design layer rests on, checked against a window
        whose artists were built under the other theme."""
        before = id(PALETTE)
        for name in ("light", "dark", "light", "dark"):
            window._apply_theme(name)
            assert id(PALETTE) == before

    @pytest.mark.parametrize("name,expected", [("light", LIGHT_PALETTE),
                                              ("dark", DARK_PALETTE)])
    def test_switching_reaches_the_palette_and_the_stylesheet(self, window, app,
                                                             name, expected):
        window._apply_theme(name)
        assert PALETTE["bg"] == expected["bg"]
        assert expected["bg"] in app.styleSheet()

    def test_the_canvas_redraws_after_a_switch(self, window):
        """A draw is what runs every artist that reads PALETTE at paint time, so
        a colour left dangling by the extraction surfaces here."""
        for name in ("light", "dark"):
            window._apply_theme(name)
            window.canvas.draw()


class TestLayoutModes:
    @pytest.mark.parametrize("mode", ["vertical", "horizontal"])
    def test_both_layouts_draw(self, window, mode):
        window.canvas.set_layout_mode(mode)
        window.canvas.draw()

    def test_the_figure_geometry_constants_are_reachable(self):
        """They moved to design.mpl_style; a missing one is an AttributeError
        deep inside a resize, which is an awkward place to find out."""
        from lillypad.design import mpl_style
        for name in ("_GEO_R", "_GEO_TOP_PX", "_GEO_L_PX", "_GEO_ROW_RATIO",
                     "_GEO_DEV_RATIO", "_AC_YLIM", "_TITLE_BAND_IN",
                     "PICTURE_DPI", "PICTURE_MAX_PX", "TRACE_COLORMAPS"):
            assert hasattr(mpl_style, name), name


class TestGeneratedIcons:
    def test_the_header_glyphs_render(self, app):
        """play/stop/plus are painted, not loaded, because they re-colour for
        both their state and the theme."""
        from lillypad.design.icons_runtime import glyph_icon
        for kind in ("play", "stop", "plus"):
            icon = glyph_icon(kind, PALETTE["accent"])
            assert not icon.isNull(), kind

    def test_the_application_icon_loads(self, app):
        from lillypad.design.icons import app_icon
        assert not app_icon().isNull()

    def test_every_icon_file_the_ui_asks_for_exists(self):
        """A missing PNG is a silently blank button, so check the whole set
        rather than waiting to notice one."""
        from pathlib import Path

        from lillypad.design import icons
        missing, checked = [], 0
        for name in dir(icons):
            value = getattr(icons, name)
            # Path constants and {theme: Path} pairs only — _APP_ICON is a cached
            # QIcon and also ends in _ICON.
            if isinstance(value, dict):
                paths = [p for p in value.values() if isinstance(p, Path)]
            elif isinstance(value, Path):
                paths = [value]
            else:
                continue
            checked += len(paths)
            missing += [f"{name}: {p}" for p in paths if not p.is_file()]
        assert not missing, missing
        assert checked >= 25, f"only found {checked} icon paths to check"
