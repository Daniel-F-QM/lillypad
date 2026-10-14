"""The design layer: tokens, the stylesheet, the theme switch.

These are the tests that keep the layer honest once the oracle in
test_stylesheet_oracle.py is gone. They enforce three properties that are easy to
break by accident and invisible when broken:

  * every placeholder in app.qss has a token behind it (otherwise startup dies);
  * no colour or font size is defined anywhere but design.tokens (otherwise the
    theme stops being switchable in one place);
  * PALETTE keeps its identity across a theme switch (otherwise artists built
    under the old theme silently keep the old colours).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from lillypad.design import stylesheet, theme, tokens

# The setObjectName vocabulary the stylesheet styles. Widgets never carry their
# own stylesheet — colour comes from an object name plus a repolish — so every id
# the code sets has to have a rule here or the widget silently goes unstyled.
QSS_OBJECT_NAMES = (
    "accent", "accentcompact", "compact", "danger", "dim", "err", "hdr", "idle",
    "moving", "nodev", "ok", "overlay", "overlaydanger", "readout", "readout_sm",
    "sat", "satok", "satwarn", "sep", "unit", "value",
)


@pytest.fixture(scope="module")
def qapp():
    """The arrow and tick PNGs are painted with QPainter, which needs a
    QApplication. Offscreen, set in conftest."""
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def qss_source():
    return stylesheet._load_template().template


class TestPalettes:
    def test_both_themes_define_the_same_keys(self):
        """A key present in one theme and missing from the other renders fine
        until someone switches, then raises mid-session."""
        assert set(tokens.DARK_PALETTE) == set(tokens.LIGHT_PALETTE)

    @pytest.mark.parametrize("name", ["DARK_PALETTE", "LIGHT_PALETTE"])
    def test_every_value_is_a_hex_colour(self, name):
        for key, value in getattr(tokens, name).items():
            assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), (name, key, value)

    def test_the_live_palette_starts_dark(self):
        assert tokens.PALETTE == tokens.DARK_PALETTE

    def test_data_colours_are_not_palette_keys(self):
        """MEMBER_COLORS, DIFF_COLORS and the overlap band identify measurements,
        not chrome: they must stay the same in both themes, so they must not be
        drawn from either palette."""
        chrome = set(tokens.DARK_PALETTE.values()) | set(tokens.LIGHT_PALETTE.values())
        data = set(tokens.MEMBER_COLORS) | set(tokens.DIFF_COLORS) | {
            tokens.OVERLAP_BAND_COLOR}
        assert not (chrome & data)

    def test_the_data_colours_are_all_distinct(self):
        """A difference curve can be on screen at the same time as a member
        curve or the overlap band and must never be mistaken for one."""
        data = list(tokens.MEMBER_COLORS) + list(tokens.DIFF_COLORS) + [
            tokens.OVERLAP_BAND_COLOR]
        assert len(set(data)) == len(data)


class TestStylesheetTokens:
    def test_every_placeholder_has_a_token(self, qss_source):
        """`substitute` raises on an unknown name, so a typo here is a startup
        crash rather than a styling glitch — but the crash happens in the field,
        and this catches it in the suite."""
        supplied = (set(tokens.DARK_PALETTE) | set(tokens.TYPE)
                    | set(tokens.METRICS) | {"arrow_up", "arrow_dn", "check"})
        used = stylesheet.stylesheet_tokens()
        assert used <= supplied, f"no token for: {sorted(used - supplied)}"

    def test_no_token_is_defined_but_unused(self, qss_source):
        """Dead tokens are how a type scale rots: someone adds a step, never
        wires it up, and the next person trusts it."""
        used = stylesheet.stylesheet_tokens()
        for group in ("TYPE", "METRICS"):
            unused = set(getattr(tokens, group)) - used
            assert not unused, f"{group} defines unused tokens: {sorted(unused)}"

    def test_the_stylesheet_holds_no_raw_hex_colour(self, qss_source):
        """Every colour goes through a token, or a theme switch misses it."""
        body = re.sub(r"/\*.*?\*/", "", qss_source, flags=re.S)
        assert not re.findall(r"#[0-9a-fA-F]{6}\b", body)

    def test_the_stylesheet_holds_no_bare_font_size(self, qss_source):
        body = re.sub(r"/\*.*?\*/", "", qss_source, flags=re.S)
        assert not re.findall(r"font-size:\s*\d", body)

    def test_prose_dollars_in_comments_are_escaped(self, qss_source):
        """An unescaped dollar in a comment makes Template try to substitute the
        following word, and the whole render dies at startup. This is the trap
        that actually bit during the conversion."""
        for comment in re.findall(r"/\*.*?\*/", qss_source, flags=re.S):
            for m in re.finditer(r"\$+", comment):
                assert len(m.group(0)) % 2 == 0, (
                    f"unescaped dollar in a comment: "
                    f"...{comment[max(0, m.start()-40):m.start()+20]}...")


class TestRendering:
    @pytest.mark.parametrize("pal,tag", [(tokens.DARK_PALETTE, "dark"),
                                         (tokens.LIGHT_PALETTE, "light")])
    def test_renders_with_nothing_left_unsubstituted(self, qapp, pal, tag):
        out = stylesheet.build_stylesheet(pal, tag)
        body = re.sub(r"/\*.*?\*/", "", out, flags=re.S)
        assert "$" not in body

    def test_both_themes_render_their_own_colours(self, qapp):
        dark = stylesheet.build_stylesheet(tokens.DARK_PALETTE, "dark")
        light = stylesheet.build_stylesheet(tokens.LIGHT_PALETTE, "light")
        assert tokens.DARK_PALETTE["bg"] in dark
        assert tokens.DARK_PALETTE["bg"] not in light
        assert tokens.LIGHT_PALETTE["bg"] in light

    @pytest.mark.parametrize("obj_name", QSS_OBJECT_NAMES)
    def test_every_object_name_is_styled(self, qapp, obj_name):
        out = stylesheet.build_stylesheet(tokens.DARK_PALETTE, "dark")
        assert f"#{obj_name} " in out or f"#{obj_name}:" in out, obj_name

    def test_the_two_themes_get_separate_icon_files(self, qapp):
        """Per-tag filenames, or the light arrows overwrite the dark ones in
        %TEMP% and one theme comes up with the wrong contrast."""
        dark = stylesheet.build_stylesheet(tokens.DARK_PALETTE, "dark")
        light = stylesheet.build_stylesheet(tokens.LIGHT_PALETTE, "light")
        assert "arrow_up_dark.png" in dark and "arrow_up_dark.png" not in light
        assert "arrow_up_light.png" in light


class TestThemeInstall:
    def test_palette_is_mutated_in_place_not_rebound(self, qapp):
        """THE load-bearing property of the whole design layer. 25-odd
        `PALETTE[...]` reads live inside already-constructed matplotlib artists;
        they share this dict object, so rebinding the name would leave every one
        of them on the old colours — silently, and only in some panels."""
        before = id(tokens.PALETTE)
        try:
            theme.install_theme(qapp, "light")
            assert id(tokens.PALETTE) == before
            assert tokens.PALETTE["bg"] == tokens.LIGHT_PALETTE["bg"]
            theme.install_theme(qapp, "dark")
            assert id(tokens.PALETTE) == before
            assert tokens.PALETTE["bg"] == tokens.DARK_PALETTE["bg"]
        finally:
            theme.install_theme(qapp, "dark")

    def test_installing_a_theme_sets_an_application_stylesheet(self, qapp):
        try:
            theme.install_theme(qapp, "dark")
            assert qapp.styleSheet().strip()
        finally:
            theme.install_theme(qapp, "dark")

    @pytest.mark.parametrize("stored,expected", [
        ({"theme": "light"}, "light"),
        ({"theme": "dark"}, "dark"),
        ({}, "dark"),
        ({"theme": "solarized"}, "dark"),
        ({"theme": None}, "dark"),
    ])
    def test_startup_theme_validates(self, stored, expected):
        """A hand-edited settings.json must not be able to leave the program
        unstyled — the palette has to be right before the first widget exists."""
        assert theme.startup_theme(stored) == expected


class TestLayerBoundary:
    """design/ must never import the GUI. The dependency runs one way, and this
    is the grep that says so."""

    @pytest.mark.parametrize("module", sorted(
        p.name for p in (Path(__file__).parents[1] / "lillypad" / "design").glob("*.py")))
    def test_design_never_imports_the_gui(self, module):
        src = (Path(__file__).parents[1] / "lillypad" / "design" / module).read_text(
            encoding="utf-8")
        body = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        for forbidden in ("lillypad.gui", "lillypad.hardware", "lillypad.scan",
                          "lillypad.sim"):
            assert forbidden not in body, f"{module} imports {forbidden}"

    def test_tokens_needs_no_qt(self):
        """Knowing what colour something is must not require a QApplication —
        that is what lets the token set be tested headlessly."""
        src = (Path(__file__).parents[1] / "lillypad" / "design"
               / "tokens.py").read_text(encoding="utf-8")
        assert "PySide6" not in src
