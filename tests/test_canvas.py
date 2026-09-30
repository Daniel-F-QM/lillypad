"""FrogCanvas after the mixin split: the structure, and the parts that are fast
on purpose.

A mixin split has exactly one failure mode a single class does not have — the
same method defined in two mixins, resolving silently to whichever comes first
in the MRO. There is no compile-time signal for it, so the first test here is a
mechanical check across the six classes. The second is the Qt one: a `Signal`
declared on a plain mixin is never wired up at all, because Qt builds a class's
metaobject from a QObject subclass, and it fails silently in one feature.

The rest pins the measured optimizations. The blitting arrangement, the
row-pooling factor and the auto-Y escalation threshold all have documented
histories and cost real milliseconds per scan column; a plausible-looking
simplification of any of them is invisible to every other test in this suite.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

CANVAS = pathlib.Path(__file__).resolve().parents[1] / "lillypad" / "gui" / "canvas"
MIXIN_FILES = ["layout.py", "interaction.py", "axes_api.py", "panels.py"]


def _members(path):
    """{member name: class it is defined in} for one module."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            name = getattr(item, "name", None)
            if name is None and isinstance(item, ast.Assign):
                target = item.targets[0]
                name = target.id if isinstance(target, ast.Name) else None
            if name:
                out.setdefault(name, []).append(f"{path.stem}.{node.name}")
    return out


class TestTheSplitIsWellFormed:
    def test_no_member_is_defined_twice_across_the_mixins(self):
        """The one failure mode this split introduces. A duplicate resolves to
        whichever class comes first in the MRO, with no error anywhere."""
        seen = {}
        clashes = []
        for name in MIXIN_FILES + ["figure.py"]:
            for member, where in _members(CANVAS / name).items():
                if len(where) > 1:
                    clashes.append((member, where))
                if member in seen:
                    clashes.append((member, seen[member] + where))
                seen[member] = where
        assert not clashes, clashes

    def test_every_signal_is_declared_on_the_concrete_class(self):
        """Qt builds the metaobject from a QObject subclass, so a Signal on a
        plain mixin is never connected — silently, at runtime."""
        offenders = []
        for name in MIXIN_FILES:
            src = (CANVAS / name).read_text(encoding="utf-8")
            if "Signal(" in src:
                offenders.append(name)
        assert not offenders, offenders

    def test_all_eight_signals_survived_the_move(self):
        from lillypad.gui.canvas import FrogCanvas
        for sig in ("limits_changed", "log_toggle_requested",
                    "proportions_changed", "axes_relaid", "fold_dragged",
                    "fold_reset_requested", "axis_edit_failed",
                    "reference_changed"):
            assert hasattr(FrogCanvas, sig), sig

    def test_the_mro_puts_the_mixins_before_qt(self):
        """resizeEvent is overridden in FigureLayoutMixin and calls super(), so
        the mixin has to come first or the re-solve never runs."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from lillypad.gui.canvas import FrogCanvas
        from lillypad.gui.canvas.layout import FigureLayoutMixin
        mro = FrogCanvas.__mro__
        assert mro.index(FigureLayoutMixin) < mro.index(FigureCanvasQTAgg)

    def test_nothing_outside_the_package_reaches_a_canvas_private(self):
        """The canvas has a public surface — set_*/fit_*/reset_axes, the eight
        signals, layout_mode, cmap_name, linewidth. Reaching past it is how a
        caller ends up depending on a blit detail."""
        root = pathlib.Path(__file__).resolve().parents[1]
        import re
        pattern = re.compile(r"canvas\._[A-Za-z]")
        offenders = []
        for path in list(root.glob("*.py")) + list((root / "lillypad").rglob("*.py")):
            if CANVAS in path.parents:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line) and not line.strip().startswith("#"):
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
        assert not offenders, offenders


@pytest.fixture(scope="module")
def canvas():
    from PySide6.QtWidgets import QApplication

    from lillypad.design.theme import install_theme
    from lillypad.gui.canvas import FrogCanvas
    app = QApplication.instance() or QApplication([])
    install_theme(app, "dark")
    c = FrogCanvas()
    c.resize(1200, 800)
    c.draw()
    yield c
    c.close()


class TestBlittingIsIntact:
    def test_the_three_live_artists_are_animated(self, canvas):
        """`animated` is what makes an ordinary draw skip them, which is what
        lets the background be cached without them. Un-animate one and the fast
        path draws it twice."""
        for _ax, art in canvas._animated:
            assert art.get_animated(), art

    def test_the_spectrum_line_is_among_them(self, canvas):
        arts = [art for _ax, art in canvas._animated]
        assert canvas.line_spec in arts
        assert canvas.im in arts
        assert canvas.line_ac in arts

    def test_a_batch_records_intent_instead_of_painting(self, canvas):
        """A scan tick updates all three panels; inside a batch those collapse
        to flags and one paint happens on the way out."""
        with canvas.batch():
            canvas._request_blit()
            canvas._request_blit()
            assert canvas._want_blit and not canvas._want_full

    def test_a_full_request_wins_over_a_blit_in_the_same_batch(self, canvas):
        """A blit over a stale background would show the old axes."""
        with canvas.batch():
            canvas._request_blit()
            canvas._request_full()
            assert canvas._want_full

    def test_a_static_change_invalidates_the_cheap_background(self, canvas):
        """_blit_spec composites the trace image once and reuses it. Anything
        static that moves has to drop that cache or the panel freezes."""
        canvas._bg_static = object()
        canvas.set_symmetry_score("asymmetry 3.2%")
        assert canvas._bg_static is None


class TestTraceDecimation:
    """Rows are max-pooled to about one per screen pixel. Max, not mean, so a
    narrow spectral line cannot vanish; an exact divisor, so the blocks tile the
    wavelength axis and the image extent stays correct."""

    def test_a_trace_that_already_fits_is_untouched(self, canvas):
        assert canvas._display_decimation(64) == 1

    def test_the_factor_is_always_a_power_of_two_divisor(self, canvas):
        for n in (2048, 1024, 4096):
            f = canvas._display_decimation(n)
            assert n % f == 0, (n, f)
            assert f & (f - 1) == 0, (n, f)

    def test_it_pools_to_at_most_one_row_per_pixel(self, canvas):
        """Not merely close to it: leaving 1024 rows for a ~550 px panel lets
        `nearest` drop half the blocks, and any narrow line inside them."""
        h = canvas.ax_trace.get_window_extent().height
        n = 2048
        f = canvas._display_decimation(n)
        if f > 1:
            assert n // f <= h

    def test_printing_hands_the_resampler_every_row(self, canvas):
        """The page has three times the window's pixels, so the panel's screen
        height is the wrong budget entirely."""
        canvas._exporting = True
        try:
            assert canvas._display_decimation(4096) == 1
        finally:
            canvas._exporting = False


class TestThePublicAxisSurface:
    def test_layout_mode_is_read_only_and_set_through_the_method(self, canvas):
        canvas.set_layout_mode("vertical")
        assert canvas.layout_mode == "vertical"
        canvas.set_layout_mode("horizontal")
        assert canvas.layout_mode == "horizontal"
        with pytest.raises(AttributeError):
            canvas.layout_mode = "vertical"

    def test_the_autoscale_switches_round_trip(self, canvas):
        for on in (False, True):
            canvas.set_autoscale_x(on)
            canvas.set_autoscale_y(on)
            canvas.set_autoscale_trace(on)
            assert canvas.autoscale_x is on
            assert canvas.autoscale_y is on
            assert canvas.autoscale_trace is on

    def test_turning_auto_y_on_drops_the_filter_state(self, canvas):
        """Otherwise the next frame eases down from limits that were never
        about the live signal, over the filter's whole decay constant."""
        canvas.set_autoscale_y(False)
        canvas._ylim_smooth = (0.0, 5000.0)
        canvas.set_autoscale_y(True)
        assert canvas._ylim_smooth is None

    def test_the_colormap_reads_back_without_its_reverse_suffix(self, canvas):
        canvas.set_cmap("viridis")
        canvas.set_trace_reversed(True)
        assert canvas.cmap_name == "viridis"
        assert canvas.cmap_reversed is True
        canvas.set_trace_reversed(False)
        canvas.set_cmap("jet")

    def test_linewidth_reads_back(self, canvas):
        canvas.set_linewidth(2.5)
        assert canvas.linewidth == pytest.approx(2.5)
        canvas.set_linewidth(1)


class TestThePanelsStillDraw:
    """One pass through each panel's update path in both layouts — the cheapest
    check that the split did not leave a method reaching for state that ended up
    on a class it no longer shares."""

    @pytest.mark.parametrize("mode", ["horizontal", "vertical"])
    def test_a_full_cycle_draws(self, canvas, mode):
        canvas.set_layout_mode(mode)
        wl = np.linspace(400.0, 700.0, 512)
        canvas.update_spectrum(wl, np.exp(-((wl - 550.0) / 20.0) ** 2) * 1e4)
        canvas.update_member_spectra(wl[:300], np.ones(300),
                                     wl[212:], np.ones(300))
        delays = np.linspace(-100.0, 100.0, 41)
        canvas.init_trace(delays, wl)
        trace = np.random.default_rng(0).random((wl.size, delays.size))
        canvas.update_trace(trace, peak=float(trace.max()))
        canvas.update_ac(delays, trace.sum(axis=0))
        canvas.set_fwhm(42.0)
        canvas.draw()

    def test_the_reference_freezes_and_clears(self, canvas):
        wl = np.linspace(400.0, 700.0, 256)
        canvas.update_spectrum(wl, np.ones(256))
        canvas.freeze_reference()
        assert canvas.has_reference()
        canvas.set_reference_visible(False)
        assert not canvas.reference_shown()
        canvas.clear_reference()
        assert not canvas.has_reference()

    def test_the_deviation_panel_shows_and_clears(self, canvas):
        wl = np.linspace(400.0, 700.0, 128)
        canvas.set_dev_visible(True)
        canvas.show_diff(wl, np.zeros(128), np.ones(128) * 0.1)
        assert canvas.diff_visible()
        canvas.clear_diff()
        canvas.set_dev_visible(False)
        assert not canvas.dev_visible()
