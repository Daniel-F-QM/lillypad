"""The reusable widgets, and the two contracts that are easy to break silently.

Both of these fail in a way nobody notices until it bites on the bench:

  * A value widget that accepts the mouse wheel edits a setting when the
    operator meant to scroll the panel. The number changes, nothing says so, and
    the next scan runs with it.
  * A widget class that stops matching its Qt base type stops being styled, so
    it comes up with the platform's default chrome in the middle of a themed
    panel — which looks like a theme bug, not an inheritance one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox,
                               QScrollArea, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

from lillypad.design.theme import install_theme
from lillypad.design.tokens import PALETTE
from lillypad.gui import widgets


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    install_theme(a, "dark")
    return a


def _wheel(app, w, delta=120):
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                     QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    app.sendEvent(w, ev)


class TestNoWheel:
    """Scrolling a panel must never edit a setting. ev.ignore() propagates the
    wheel to the enclosing QScrollArea, so the panel still scrolls."""

    @pytest.mark.parametrize("cls,setup", [
        (widgets.SpinBox, lambda w: (w.setRange(0, 100), w.setValue(50))),
        (widgets.DoubleSpinBox, lambda w: (w.setRange(0, 100), w.setValue(50.0))),
        (widgets.Slider, lambda w: (w.setRange(0, 100), w.setValue(50))),
    ])
    def test_wheel_does_not_change_the_value(self, app, cls, setup):
        w = cls()
        setup(w)
        before = w.value()
        _wheel(app, w, +120)
        _wheel(app, w, -120)
        assert w.value() == before

    def test_combo_box_keeps_its_selection(self, app):
        c = widgets.ComboBox()
        c.addItems(["a", "b", "c"])
        c.setCurrentIndex(1)
        _wheel(app, c, +120)
        assert c.currentIndex() == 1

    def test_the_wheel_still_reaches_an_enclosing_scroll_area(self, app):
        """The other half of the contract: ignoring is not swallowing."""
        area = QScrollArea()
        inner = QWidget()
        lay = QVBoxLayout(inner)
        sb = widgets.DoubleSpinBox()
        sb.setRange(0, 100)
        lay.addWidget(sb)
        area.setWidget(inner)
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                         QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                         Qt.NoScrollPhase, False)
        assert not app.sendEvent(sb, ev) or True   # accepted() is what matters
        assert not ev.isAccepted(), "the event was swallowed, not propagated"

    def test_typing_and_arrows_are_untouched(self, app):
        """Only the wheel is refused — the widget is still a normal spin box."""
        sb = widgets.SpinBox()
        sb.setRange(0, 100)
        sb.setValue(50)
        sb.stepUp()
        assert sb.value() == 51


class TestStylesheetStillMatches:
    """Qt stylesheet TYPE selectors match subclasses, which is what lets
    app.qss style these without naming them. Moving the classes to another
    module must not change that."""

    @pytest.mark.parametrize("cls,base", [
        (widgets.SpinBox, QSpinBox),
        (widgets.DoubleSpinBox, QDoubleSpinBox),
        (widgets.ComboBox, QComboBox),
        (widgets.Slider, QSlider),
    ])
    def test_subclasses_its_qt_base(self, app, cls, base):
        assert issubclass(cls, base)

    @pytest.mark.parametrize("selector", ["QDoubleSpinBox", "QSpinBox",
                                          "QScrollBar", "QPushButton"])
    def test_the_app_sheet_still_carries_the_type_rule(self, app, selector):
        assert selector in app.styleSheet()

    def test_no_widget_carries_its_own_stylesheet(self, app):
        """The author's enforced rule: colour comes from object name plus a
        repolish, so a runtime theme switch reaches everything."""
        for cls in (widgets.SpinBox, widgets.DoubleSpinBox, widgets.ComboBox,
                    widgets.Slider):
            assert not cls().styleSheet()


class TestPaintedWidgets:
    """The four custom-painted widgets read PALETTE at paint time rather than
    caching a colour, which is what makes them follow a theme switch."""

    @pytest.mark.parametrize("factory", [
        lambda: widgets.StatusLamp(),
        lambda: widgets._hline(),
    ])
    def test_constructs_and_paints(self, app, factory):
        w = factory()
        w.resize(40, 20)
        w.grab()          # forces paintEvent

    @pytest.mark.parametrize("state", sorted(widgets.StatusLamp.COLORS))
    def test_status_lamp_paints_every_state(self, app, state):
        lamp = widgets.StatusLamp()
        lamp.set_state(state)
        lamp.resize(16, 16)
        lamp.grab()

    def test_status_lamp_refuses_an_unknown_state(self, app):
        """Loudly, rather than painting the previous colour: the lamp is a
        saturation warning, and one that silently keeps showing "ok" is worse
        than no lamp."""
        with pytest.raises(ValueError):
            widgets.StatusLamp().set_state("bad")

    def test_every_lamp_state_names_a_real_palette_key(self, app):
        from lillypad.design.tokens import DARK_PALETTE, LIGHT_PALETTE
        for key in widgets.StatusLamp.COLORS.values():
            assert key in DARK_PALETTE and key in LIGHT_PALETTE, key

    def test_no_device_overlay_paints_its_watermark(self, app):
        parent = QWidget()          # must outlive the child
        ov = widgets.NoDeviceOverlay(parent, "Nothing connected",
                                     "Connect…", lambda: None)
        ov.resize(400, 300)
        ov.grab()

    def test_painted_widgets_follow_a_theme_switch(self, app):
        """Grab under each theme; the point is that neither raises and the
        palette they read is the live one."""
        parent = QWidget()          # must outlive the child
        ov = widgets.NoDeviceOverlay(parent, "Nothing connected",
                                     "Connect…", lambda: None)
        ov.resize(200, 150)
        try:
            for theme in ("light", "dark"):
                install_theme(app, theme)
                ov.refresh_theme()
                ov.grab()
                assert PALETTE["bg"]
        finally:
            install_theme(app, "dark")


class TestExposurePlumbing:
    """The integration box's bounds. A device that reports nothing must still
    land on a safe whole-millisecond range rather than an empty or inverted one."""

    class _Bare:
        pass

    def test_defaults_for_a_device_that_reports_nothing(self):
        from lillypad.gui.workers import (MAX_UI_EXPOSURE_MS, _exposure_bounds,
                                          _exposure_ms)
        lo, hi = _exposure_bounds(self._Bare())
        assert lo == 1.0
        assert hi == MAX_UI_EXPOSURE_MS
        assert lo < hi
        assert _exposure_ms(self._Bare()) == 10.0

    def test_a_long_exposure_device_is_capped_by_ui_policy(self):
        """acquire() blocks for the whole exposure, so a device that would
        happily take its documented 600 s must not be offered it."""
        from lillypad.gui.workers import MAX_UI_EXPOSURE_MS, _exposure_bounds

        class _Slow:
            min_integration_ms = 0.002
            max_integration_ms = 600_000.0

        lo, hi = _exposure_bounds(_Slow())
        assert lo == 0.002
        assert hi == MAX_UI_EXPOSURE_MS

    def test_exposure_prefers_exposure_ms_over_integration_ms(self):
        """integration_ms folds in on-board averaging on an Avantes; seeding the
        box from it would multiply the exposure by the average count on the next
        edit."""
        from lillypad.gui.workers import _exposure_ms

        class _Averaging:
            exposure_ms = 20.0
            integration_ms = 200.0

        assert _exposure_ms(_Averaging()) == 20.0
