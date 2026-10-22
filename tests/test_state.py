"""InstrumentState, the window's forwarders onto it, and the dialog value API.

Two things here are worth a test rather than a convention.

The **forwarders** are where a refactor can go silently wrong. `self.spec` used
to be a plain attribute assigned in a dozen places; it is now a property over
`state.spec`. A property whose setter was forgotten does not fail — Python
happily shadows it on the instance if the class attribute is a plain value, and
where it is a property the assignment raises somewhere deep in a connect path.
So every field is round-tripped through the window AND checked to be the same
object the state holds, which is the part a shadowing bug breaks.

The **dialog value properties** are checked against their own widgets, because
their whole purpose is that a caller never touches the widget: if the two drift,
the window reads a stale value and nothing says so.
"""

from __future__ import annotations

import pytest

from lillypad.gui.state import InstrumentState
from lillypad.hardware import StitchedSpectrometer
from lillypad.sim import DEFAULT_PULSE


class TestTheStateItself:
    def test_nothing_is_connected_by_default(self):
        """The application's real startup state — a real absence, not a
        stand-in device that could return plausible-looking counts."""
        s = InstrumentState()
        assert s.spec is None and s.stage is None
        assert not s.have_spec() and not s.have_stage()
        assert not s.pair_live()

    def test_both_slots_exist_from_the_start_and_are_empty(self):
        """There is no "multi-spectrometer mode" flag to get out of step: slot 1
        alone IS single-spectrometer mode."""
        s = InstrumentState()
        assert s.multi["serials"] == [None, None]
        assert s.multi["labels"] == [None, None]
        assert s.multi["cals"] == [None, None]
        assert s.multi_members == [None, None]

    def test_two_states_do_not_share_their_slot_tables(self):
        """A mutable default on a dataclass is the classic way for two
        instruments to quietly become one."""
        a, b = InstrumentState(), InstrumentState()
        a.multi["serials"][0] = "seabreeze:X"
        a.multi_members[0] = object()
        assert b.multi["serials"][0] is None
        assert b.multi_members[0] is None

    def test_pair_live_is_answered_off_the_device(self, monkeypatch):
        """Not off a flag. `pair_live()` is the only definition of the mode."""
        s = InstrumentState()
        s.spec = object()
        assert not s.pair_live()
        s.spec = StitchedSpectrometer.__new__(StitchedSpectrometer)
        assert s.pair_live()

    def test_the_simulator_choice_has_a_default(self):
        s = InstrumentState()
        assert s.sim_pulse == DEFAULT_PULSE
        assert s.sim_gate == "shg"

    def test_no_dark_is_recorded_to_begin_with(self):
        s = InstrumentState()
        assert s.background is None
        assert s.background_members is None
        assert s.background_exposures is None


@pytest.fixture(scope="module")
def window():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from lillypad.design.theme import install_theme
    app = QApplication.instance() or QApplication([])
    install_theme(app, "dark")
    from lillypad.gui.window import FrogWindow
    win = FrogWindow()
    yield win
    win.close()


# Every forwarder, and a value distinguishable from the default.
FORWARDED = [
    ("spec", object()),
    ("stage", object()),
    ("background", [1.0, 2.0]),
    ("background_members", ([1.0], [2.0])),
    ("background_exposures", (5.0, 7.0)),
    ("sim_pulse", "tl_sech"),
    ("sim_gate", "pg"),
    ("seabreeze_backend", "cseabreeze"),
    ("_multi_members", ["a", "b"]),
    ("_stitch_stale", True),
]

# window attribute -> InstrumentState field, where the two names differ.
RENAMED = {"_multi": "multi", "_multi_members": "multi_members",
           "_stitch_stale": "stitch_stale"}


class TestTheWindowForwardsOntoTheState:
    @pytest.mark.parametrize("name,value", FORWARDED, ids=[n for n, _ in FORWARDED])
    def test_a_write_through_the_window_lands_in_the_state(self, window, name,
                                                           value):
        field = RENAMED.get(name, name)
        before = getattr(window.state, field)
        try:
            setattr(window, name, value)
            assert getattr(window, name) is value
            assert getattr(window.state, field) is value
        finally:
            setattr(window, name, before)

    def test_the_slot_table_is_the_same_object_both_ways(self, window):
        """Read-mutate-in-place is how the slots are edited
        (`self._multi["serials"][slot] = ident`), so a forwarder that copied
        would drop every edit on the floor."""
        assert window._multi is window.state.multi

    def test_the_slot_table_can_also_be_replaced_wholesale(self, window):
        """_slots_rolled_back_on_failure restores a snapshot by assignment."""
        before = {k: list(v) for k, v in window._multi.items()}
        try:
            window._multi = {"serials": ["x", None], "labels": [None, None],
                             "cals": [None, None]}
            assert window.state.multi["serials"][0] == "x"
        finally:
            window._multi = before

    def test_scan_cfg_is_shared_not_copied(self, window):
        """It is mutated field by field at scan start (`c.n_average = …`)."""
        assert window.scan_cfg is window.state.scan_cfg

    def test_the_connection_predicates_read_the_state(self, window):
        assert window._have_spec() is False
        assert window._have_stage() is False
        try:
            window.state.spec = object()
            assert window._have_spec() is True
            assert window._pair_live() is False
        finally:
            window.state.spec = None


class TestDialogsReadTheStateNotTheWindow:
    def test_every_connect_dialog_holds_the_same_state_object(self, window):
        """Handed the state rather than left to find it: a dialog asking what is
        connected must not be able to reach a widget on the way."""
        for dlg in (window.dlg_spec, window.dlg_stage, window.dlg_sim):
            assert dlg.state is window.state


class TestDialogValueProperties:
    """Each one against its own widget. Their purpose is that no caller outside
    the dialog touches the widget, so drift between the two is invisible."""

    def test_acquisition_values_track_their_widgets(self, window):
        d = window.dlg_settings
        d.spin_avg.setValue(7); d.spin_idle.setValue(3); d.spin_wait.setValue(250)
        d.spin_sat.setValue(85.0)
        assert (d.n_average, d.idle_shots, d.settle_ms) == (7, 3, 250)
        assert d.sat_percent == pytest.approx(85.0)

    @pytest.mark.parametrize("flag", [True, False])
    def test_acquisition_checkboxes_track_their_widgets(self, window, flag):
        d = window.dlg_settings
        d.chk_abort_sat.setChecked(flag)
        d.chk_abort_stage_fault.setChecked(flag)
        d.chk_ref.setChecked(flag)
        assert d.abort_on_saturation is flag
        assert d.abort_on_stage_fault is flag
        assert d.take_delay_reference is flag

    def test_the_delay_reference_is_off_by_default(self, window):
        """It costs two acquisitions and is diagnostic only."""
        from lillypad.scan import FrogScanConfig
        assert FrogScanConfig.capture_delay_reference is False

    def test_alignment_step_tracks_its_widget(self, window):
        window.dlg_align.spin_align_step.setValue(250.0)
        assert window.dlg_align.step_fs == pytest.approx(250.0)

    def test_log_scale_tracks_its_widget(self, window):
        for flag in (True, False):
            window.dlg_graphics.chk_log.setChecked(flag)
            assert window.dlg_graphics.log_scale is flag

    def test_backlash_tracks_its_widget(self, window):
        window.dlg_stage.spin_backlash.setValue(40.0)
        assert window.dlg_stage.backlash_um == pytest.approx(40.0)

    def test_date_stamping_tracks_its_widget(self, window):
        for flag in (True, False):
            window.dlg_export.chk_date.setChecked(flag)
            assert window.dlg_export.date_stamped is flag

    def test_trim_empty_rows_tracks_its_widget(self, window):
        for flag in (True, False):
            window.dlg_export.chk_trim.setChecked(flag)
            assert window.dlg_export.trim_empty_rows is flag

    def test_trim_is_greyed_out_for_the_npz_only(self, window):
        was = window._export_fmt
        try:
            for key, enabled in (("npz", False), ("csv", True), ("dwc", True)):
                window.export.set_format(key)
                assert window.dlg_export.chk_trim.isEnabled() is enabled
        finally:
            window.export.set_format(was)


class TestTheLiveFeedSwitchIsNamed:
    """The Avantes window has to stop the feed before arming a hardware
    trigger. It asks in these terms rather than reaching for the toolbar."""

    def test_the_switch_round_trips(self, window):
        was = window.feed_enabled
        try:
            window.set_feed_enabled(False)
            assert window.feed_enabled is False
            window.set_feed_enabled(True)
            assert window.feed_enabled is True
        finally:
            window.set_feed_enabled(was)

    def test_no_dialog_reaches_for_the_feed_button(self):
        """The reach-through this replaced. A grep, because the point is that it
        cannot come back somewhere else."""
        import pathlib
        dialogs = (pathlib.Path(__file__).resolve().parents[1]
                   / "lillypad" / "gui" / "dialogs")
        offenders = [p.name for p in dialogs.glob("*.py")
                     if "btn_feed" in p.read_text(encoding="utf-8")]
        assert not offenders, offenders
