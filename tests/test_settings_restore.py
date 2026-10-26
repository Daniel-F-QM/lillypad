"""settings.json through the real window: the defaults a fresh install starts
on, and every acquisition and alignment setting coming back after a restart.

The window here is pointed at a throwaway settings.json, so nothing these tests
change can reach the real one — the window saves on close.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox

from lillypad import settings


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """A window on a fresh install: no settings.json at all."""
    from PySide6.QtWidgets import QApplication

    from lillypad.design.theme import install_theme
    app = QApplication.instance() or QApplication([])
    install_theme(app, "dark")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "SETTINGS_PATH",
               tmp_path_factory.mktemp("settings") / "settings.json")
    settings._reset_cache()
    from lillypad.gui.window import FrogWindow
    win = FrogWindow()
    yield win
    win.close()              # saves — into the throwaway file
    mp.undo()
    settings._reset_cache()


def _apply(win, data):
    """Restore `data` exactly as startup does: through every setter."""
    for key, _get, setter in win.settings_ctl._setting_specs():
        if setter is not None and key in data:
            setter(data[key])


def _controls(dialog):
    """Every number box and tick box in a dialog."""
    return (dialog.findChildren(QAbstractSpinBox)
            + dialog.findChildren(QCheckBox))


def _nudge(w):
    """Change a control's value, and return how to put it back."""
    if isinstance(w, QCheckBox):
        w.setChecked(not w.isChecked())
        return lambda: w.setChecked(not w.isChecked())
    before = w.value()
    w.setValue(w.minimum() if before != w.minimum() else w.maximum())
    return lambda: w.setValue(before)


class TestAFreshInstall:
    def test_starts_on_the_defaults(self, fresh):
        snap = fresh.settings_ctl._settings_snapshot()
        for key, want in settings.DEFAULT_SETTINGS.items():
            if key == "window":
                continue          # the live geometry, not a value to compare
            assert snap[key] == pytest.approx(want), key

    def test_the_defaults_reach_what_the_widgets_drive(self, fresh):
        """Restored through the widgets' own signals, so the live saturation
        threshold follows the 85% the box shows rather than the 99% it was
        built with."""
        want = settings.DEFAULT_SETTINGS["saturation_pct"] / 100.0
        assert fresh.scan_cfg.saturation_fraction == pytest.approx(want)


class TestDateDelimiter:
    """The Export window's date prefix, with whatever goes between year,
    month and day."""

    @pytest.fixture
    def export(self, fresh):
        d = fresh.dlg_export
        d.chk_date.setChecked(True)
        d.edit_name.setText("FROG_trace")
        yield d
        _apply(fresh, settings.DEFAULT_SETTINGS)

    @staticmethod
    def _today(sep):
        import time
        t = time.localtime()
        return f"{t.tm_year:04d}{sep}{t.tm_mon:02d}{sep}{t.tm_mday:02d}"

    def test_the_default_is_a_dash(self, export):
        assert export.date_delimiter == "-"
        assert export.stem() == f"{self._today('-')}_FROG_trace"

    @pytest.mark.parametrize("sep", [".", "", "__", "%"])
    def test_any_file_name_safe_delimiter_is_used(self, export, sep):
        export.set_date_delimiter(sep)
        assert export.stem() == f"{self._today(sep)}_FROG_trace"
        assert export.lbl_preview.text().endswith(
            f"{self._today(sep)}_FROG_trace.dwc")

    @pytest.mark.parametrize("bad", ["/", ":", "a*b", "----"])
    def test_what_typing_could_not_produce_is_refused(self, export, bad):
        with pytest.raises(ValueError):
            export.set_date_delimiter(bad)
        assert export.date_delimiter == "-"

    def test_illegal_characters_cannot_be_typed(self, export):
        from PySide6.QtGui import QValidator
        v = export.edit_date_delim.validator()
        assert v.validate("/", 0)[0] == QValidator.State.Invalid
        assert v.validate(".", 0)[0] == QValidator.State.Acceptable

    def test_it_greys_out_without_a_date(self, export):
        export.chk_date.setChecked(False)
        assert not export.edit_date_delim.isEnabled()
        assert export.stem() == "FROG_trace"
        export.chk_date.setChecked(True)
        assert export.edit_date_delim.isEnabled()


class TestEverySettingIsKept:
    @pytest.mark.parametrize("dialog", ["dlg_settings", "dlg_align"])
    def test_every_control_in_the_window_is_saved(self, fresh, dialog):
        """Acquisition Settings and Alignment: a control left out of
        settings.json would quietly reset on every launch."""
        controls = _controls(getattr(fresh, dialog))
        assert controls
        for w in controls:
            before = fresh.settings_ctl._settings_snapshot()
            undo = _nudge(w)
            try:
                after = fresh.settings_ctl._settings_snapshot()
                assert after != before, (
                    f"{dialog}: {type(w).__name__} {w.objectName() or w} "
                    f"is not saved")
            finally:
                undo()

    def test_saved_values_come_back(self, fresh):
        d = fresh.dlg_settings
        d.spin_avg.setValue(5); d.spin_idle.setValue(2)
        d.spin_wait.setValue(300); d.spin_sat.setValue(90.0)
        d.chk_abort_sat.setChecked(True)
        d.chk_abort_stage_fault.setChecked(True)
        d.chk_ref.setChecked(True)
        fresh.spin_start.setValue(-40.0); fresh.spin_stop.setValue(60.0)
        fresh.spin_step_fs.setValue(2.0)
        fresh.dlg_align.spin_align_step.setValue(33.0)
        fresh.dlg_spec.set_full_scale(4000.0)
        saved = fresh.settings_ctl._settings_snapshot()

        _apply(fresh, settings.DEFAULT_SETTINGS)     # as if restarted clean…
        fresh.dlg_spec.set_full_scale(None)
        assert fresh.settings_ctl._settings_snapshot() != saved
        _apply(fresh, saved)                         # …then restored

        assert fresh.settings_ctl._settings_snapshot() == saved
        assert fresh.scan_cfg.saturation_counts == 4000.0
        assert fresh.scan_cfg.saturation_fraction == pytest.approx(0.90)
        assert fresh.dlg_spec.edit_full_scale.text() == "4000"
        _apply(fresh, settings.DEFAULT_SETTINGS)
        fresh.dlg_spec.set_full_scale(None)

    def test_full_scale_auto_is_saved_as_null(self, fresh):
        fresh.dlg_spec.set_full_scale(None)
        assert fresh.settings_ctl._settings_snapshot()["full_scale_counts"] \
            is None

    @pytest.mark.parametrize("bad", [-5, 0, "lots"])
    def test_a_bad_full_scale_is_refused(self, fresh, bad):
        """Raising costs that one key on restore, never the whole file."""
        with pytest.raises((ValueError, TypeError)):
            fresh.settings_ctl._restore_full_scale(bad)
        assert fresh.scan_cfg.saturation_counts is None

    def test_the_date_delimiter_comes_back(self, fresh):
        d = fresh.dlg_export
        d.set_date_delimiter(".")
        saved = fresh.settings_ctl._settings_snapshot()
        assert saved["export_date_delimiter"] == "."
        _apply(fresh, settings.DEFAULT_SETTINGS)
        assert d.date_delimiter == "-"
        _apply(fresh, saved)
        assert d.date_delimiter == "."
        _apply(fresh, settings.DEFAULT_SETTINGS)

    def test_integration_time_and_jog_step_are_not_kept(self, fresh):
        keys = {k for k, _g, _s in fresh.settings_ctl._setting_specs()}
        assert not any("integration" in k or "jog" in k for k in keys)
