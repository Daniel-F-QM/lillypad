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
    """One pass through each panel's update path — the cheapest check that the
    split did not leave a method reaching for state that ended up on a class it
    no longer shares."""

    def test_a_full_cycle_draws(self, canvas):
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

    def test_the_colour_bar_has_its_own_cached_layer(self, canvas):
        """Its range moves with the trace's peak on most scan columns: as a
        static artist each move would cost a full redraw, and drawn on every
        blit its tick labels cost more than the trace image. So it is out of
        the background AND out of the per-frame list, and its layer is reused
        until what it shows changes."""
        assert canvas.ax_cbar.get_animated()
        assert all(art is not canvas.ax_cbar for _ax, art in canvas._animated)
        canvas.draw()
        layer = canvas._cbar_background()
        assert canvas._cbar_background() is layer        # nothing changed
        lo, hi = canvas.im.get_clim()
        canvas.set_trace_clim(lo, hi + 1.0)
        assert canvas._cbar_background() is not layer    # the range did
        canvas.set_autoscale_clim(True)

    def test_the_deviation_panel_shows_and_clears(self, canvas):
        wl = np.linspace(400.0, 700.0, 128)
        canvas.set_dev_visible(True)
        canvas.show_diff(wl, np.zeros(128), np.ones(128) * 0.1)
        assert canvas.diff_visible()
        canvas.clear_diff()
        canvas.set_dev_visible(False)
        assert not canvas.dev_visible()


class TestTheColourBar:
    """The FROG trace's colour bar: where it sits, what it says, and the colour
    range it lets the operator take over from the follow-the-peak default."""

    @pytest.fixture(autouse=True)
    def _a_trace(self, canvas):
        wl = np.linspace(400.0, 700.0, 64)
        delays = np.linspace(-100.0, 100.0, 21)
        canvas.init_trace(delays, wl)
        trace = np.random.default_rng(1).random((wl.size, delays.size)) * 4000.0
        canvas.update_trace(trace, peak=4000.0)
        canvas.draw()
        yield
        canvas.set_autoscale_clim(True)
        canvas.set_trace_normalised(False)

    def test_it_stands_beside_the_trace_over_its_height(self, canvas):
        tr, cb = canvas.ax_trace.get_position(), canvas.ax_cbar.get_position()
        assert cb.x0 > tr.x1                      # clear of the trace
        assert cb.x1 < 1.0                        # and inside the figure
        assert cb.y0 == pytest.approx(tr.y0)
        assert cb.y1 == pytest.approx(tr.y1)

    def test_the_stacked_panels_keep_one_delay_axis(self, canvas):
        """The AC gives up the same strip as the trace, or a delay would sit
        at two different x positions in the two panels."""
        tr, ac = canvas.ax_trace.get_position(), canvas.ax_ac.get_position()
        assert (ac.x0, ac.x1) == pytest.approx((tr.x0, tr.x1))

    def test_it_is_labelled_in_the_units_on_screen(self, canvas):
        assert canvas.ax_cbar.get_ylabel() == "Intensity (counts)"
        canvas.set_trace_normalised(True)
        assert canvas.ax_cbar.get_ylabel() == "Intensity (norm.)"

    def test_it_follows_the_peak_by_default(self, canvas):
        assert canvas.autoscale_clim
        assert canvas.im.get_clim() == pytest.approx((0.0, 4000.0))
        assert canvas.ax_cbar.get_ylim() == pytest.approx((0.0, 4000.0))

    def test_a_manual_range_survives_the_next_column(self, canvas):
        canvas.set_trace_clim(500.0, 2500.0)
        assert not canvas.autoscale_clim
        canvas.update_trace(canvas._trace_raw * 2.0, peak=8000.0)
        assert canvas.im.get_clim() == pytest.approx((500.0, 2500.0))

    def test_an_inverted_range_is_refused(self, canvas):
        canvas.set_trace_clim(3000.0, 1000.0)
        assert canvas.autoscale_clim
        assert canvas.im.get_clim() == pytest.approx((0.0, 4000.0))

    def test_auto_snaps_straight_back_to_the_peak(self, canvas):
        canvas.set_trace_clim(500.0, 2500.0)
        canvas.set_autoscale_clim(True)
        assert canvas.im.get_clim() == pytest.approx((0.0, 4000.0))

    def test_right_click_toggles_the_follow(self, canvas):
        canvas.reset_axes(canvas.ax_cbar)
        assert not canvas.autoscale_clim
        assert canvas.im.get_clim() == pytest.approx((0.0, 4000.0))  # frozen
        canvas.reset_axes(canvas.ax_cbar)
        assert canvas.autoscale_clim

    def test_a_drag_on_the_bar_sets_the_range(self, canvas):
        canvas._apply_zoom(canvas.ax_cbar, 0.0, 1.0, 1000.0, 3000.0)
        assert canvas.im.get_clim() == pytest.approx((1000.0, 3000.0))
        assert not canvas.autoscale_clim

    def test_a_typed_end_label_sets_one_end(self, canvas):
        canvas.apply_axis_bound(canvas.ax_cbar, "y", "hi", 1234.0)
        assert canvas.im.get_clim() == pytest.approx((0.0, 1234.0))

    def test_its_end_labels_are_clickable(self, canvas):
        assert (canvas.ax_cbar, "y") in canvas._editable_axes()
        assert canvas._end_tick_labels(canvas.ax_cbar, "y")

    def test_the_wheel_stretches_only_the_top(self, canvas):
        from types import SimpleNamespace
        canvas.set_trace_clim(1000.0, 3000.0)
        canvas._on_scroll(SimpleNamespace(inaxes=canvas.ax_cbar, step=1))
        lo, hi = canvas.im.get_clim()
        assert lo == pytest.approx(1000.0)
        assert hi < 3000.0                       # wheel up brightens
        canvas._on_scroll(SimpleNamespace(inaxes=canvas.ax_trace, step=1))
        assert canvas.im.get_clim() == pytest.approx((lo, hi))  # bar only

    def test_switching_to_a_normalised_view_drops_a_counts_range(self, canvas):
        """A range in counts on a 0…1 image paints the whole trace one colour."""
        canvas.set_trace_clim(500.0, 2500.0)
        canvas.set_trace_normalised(True)
        assert canvas.autoscale_clim

    def test_printing_hands_the_bar_back_animated(self, canvas, tmp_path):
        """save_picture un-animates everything in _animated for the print (an
        animated axes is skipped by Figure.draw) and must hand the flag back,
        or every later blit would draw the bar twice."""
        canvas.save_picture(str(tmp_path / "p.png"), dpi=50)
        assert canvas.ax_cbar.get_animated()
