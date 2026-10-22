"""settings.json: what is remembered between sessions, and what is not.

One table, `setting_specs()`, read in both directions as (key, getter, setter).
Adding a setting is one line there and nothing else — the save, the ten-second
autosave diff and the restore all walk that list. Polled off the table rather
than wired to thirty valueChanged signals, because a slider drag emits fifty
changes a second and not one of them is worth a disk write.

**ORDER IS LOAD-BEARING on restore.** The auto-scale checkboxes come before the
limit spinboxes, because turning trace auto-scale off pins those four boxes to
whatever is on screen and would overwrite the values being restored.

What is deliberately NOT saved is as much of the design as what is: **no axis
state**. Not the auto-scale flags, not the manual limits. Where the axes are
pointing is a property of the session, not a preference — a rubber-band zoom and
a hand-typed range land in exactly the same place, so persisting either meant a
zoom taken to read one feature survived the restart and the program came back
still zoomed into it, on data that had gone.
"""

from lillypad.settings import SETTINGS_VERSION, as_bool, save_settings


class SettingsController:
    """Saving and restoring the app's own preferences."""

    def __init__(self, win):
        self.win = win

    # ── Settings persistence ─────────────────────────────────────────────────
    # One table, read in both directions: (key, getter, setter). Adding a
    # setting is one line here and nothing else — the save, the autosave diff
    # and the restore all walk this list.
    #
    # A setter of None means "restored somewhere else, for a reason given at
    # that site": the theme has to exist before any widget does, and the
    # geometry has to be applied in the same breath as the decision to show
    # maximized. Both are still SAVED from here, so this stays the one place
    # that knows the full set.
    #
    # ORDER IS LOAD-BEARING on restore: the auto-scale checkboxes come before
    # the limit spinboxes, because _on_autoscale_trace(False) pins those four
    # boxes to whatever is on screen and would overwrite the values being
    # restored.
    #
    # Getters read the CANVAS for the auto-scale flags and the WIDGETS for the
    # manual limits. That split is not arbitrary: _apply_zoom and reset_axes
    # mutate canvas.autoscale_* directly, so the canvas is the only thing that
    # always knows; while the spinboxes are the memory of the last bounds the
    # operator dialled in by hand, which is what should come back — not a
    # snapshot of an auto-scaled view that happened to be live at the time.
    def _setting_specs(self):
        c = self.win.canvas
        g = self.win.dlg_graphics
        d = self.win.dlg_settings
        return [
            ("theme",   lambda: self.win._theme,    None),
            ("window",  self._window_geometry,  None),

            ("spec_frac", lambda: float(c.spec_fraction()),
                          lambda v: g.sld_prop.setValue(int(round(float(v) * 100)))),

            # NO axis state here — not the auto-scale flags, not the manual
            # limits, not the autocorrelation's view. Where the axes are
            # pointing is a property of the session, not a preference: a
            # rubber-band zoom and a hand-typed range land in exactly the same
            # place (auto-scale off, limits pinned), so persisting either meant
            # a zoom taken to read one feature survived the restart and the
            # program came back still zoomed into it, on data that had gone.
            # Every launch starts on the normal view and fits to what is there.
            ("log_scale",  lambda: g.log_scale,
                           lambda v: g.chk_log.setChecked(as_bool(v))),
            # setCurrentText is a silent no-op on a non-editable combo for a name
            # that is not in the list, so a colormap dropped from
            # TRACE_COLORMAPS needs no guard of its own.
            ("cmap",          lambda: c.cmap_name, g.cmb_cmap.setCurrentText),
            # Display-only delay smoothing — a preference, like the scale.
            ("trace_smooth",  self.win.btn_smooth.isChecked,
                              lambda v: self.win.btn_smooth.setChecked(as_bool(v))),
            ("cmap_reversed", lambda: c.cmap_reversed,
                              lambda v: g.chk_cmap_rev.setChecked(as_bool(v))),
            # Percent, as typed — the canvas keeps the fraction.
            ("trace_threshold_pct", g.spin_thresh.value,
                                    lambda v: g.spin_thresh.setValue(float(v))),
            ("line_width",    lambda: float(c.linewidth),
                              lambda v: g.spin_lw.setValue(float(v))),
            ("export_format", lambda: self.win._export_fmt, self.win.export.set_format),
            # The name as TYPED, not the composed file name: the date prefix is
            # the checkbox's doing and is re-made from today's date on every
            # save, so storing it would bring back the day this was written.
            ("export_name", self.win.dlg_export.edit_name.text,
                            lambda v: self.win.dlg_export.edit_name.setText(str(v))),
            ("export_date_prefix", lambda: self.win.dlg_export.date_stamped,
                                   lambda v: self.win.dlg_export.chk_date.setChecked(
                                       as_bool(v))),
            ("export_trim_empty_rows", lambda: self.win.dlg_export.trim_empty_rows,
                                       lambda v: self.win.dlg_export.chk_trim.setChecked(
                                           as_bool(v))),

            ("avg_per_point",      lambda: d.n_average,
                                   lambda v: d.spin_avg.setValue(int(v))),
            ("idle_shots",         lambda: d.idle_shots,
                                   lambda v: d.spin_idle.setValue(int(v))),
            ("wait_after_move_ms", lambda: d.settle_ms,
                                   lambda v: d.spin_wait.setValue(int(v))),
            ("saturation_pct",     lambda: d.sat_percent,
                                   lambda v: d.spin_sat.setValue(float(v))),
            ("abort_on_saturation",  lambda: d.abort_on_saturation,
                                     lambda v: d.chk_abort_sat.setChecked(as_bool(v))),
            ("abort_on_stage_fault", lambda: d.abort_on_stage_fault,
                                     lambda v: d.chk_abort_stage_fault.setChecked(as_bool(v))),

            ("scan_start_fs",   self.win.spin_start.value,
                                lambda v: self.win.spin_start.setValue(float(v))),
            ("scan_stop_fs",    self.win.spin_stop.value,
                                lambda v: self.win.spin_stop.setValue(float(v))),
            ("scan_step_fs",    self.win.spin_step_fs.value,
                                lambda v: self.win.spin_step_fs.setValue(float(v))),
            ("scan_delay_reference", lambda: d.take_delay_reference,
                                lambda v: self.win.dlg_settings.chk_ref.setChecked(
                                    as_bool(v))),
            ("align_step_fs",   lambda: self.win.dlg_align.step_fs,
                                lambda v: self.win.dlg_align.spin_align_step.setValue(float(v))),
        ]

    def _window_geometry(self):
        """Position and size for the next launch.

        normalGeometry when maximized, not geometry: a maximized window reports
        the whole screen, and saving that would make the next un-maximize
        restore to a 'normal' size the operator never chose.
        """
        r = self.win.normalGeometry() if self.win.isMaximized() else self.win.geometry()
        return {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height(),
                "maximized": self.win.isMaximized()}

    def _restore_settings(self):
        """Apply the saved settings to widgets that already exist.

        Every key is applied on its own and every failure is swallowed on its
        own: a file half-written by an older build, a hand-edit with a typo in
        it, or a colormap that no longer exists must cost the operator that ONE
        setting and never the rest of them.

        Deliberately NOT blockSignals. The dialog's signal wiring IS the only
        path from a widget to the canvas — blocking it would restore the
        dialog's appearance while leaving the canvas on defaults — and the
        handlers carry side effects nothing else does: _on_autoscale_* enable
        and disable the boxes they own, _on_prop repaints its percent label.
        Restoring through the same route a click takes is also the version that
        cannot drift from it. push_to_canvas() then covers the values that
        happened to equal a widget's default and so emitted nothing.
        """
        for key, _get, setter in self._setting_specs():
            if setter is None or key not in self.win._settings:
                continue
            try:
                setter(self.win._settings[key])
            except Exception:
                continue          # one bad key costs one setting
        try:
            self.win.dlg_graphics.push_to_canvas()
        except Exception:
            pass
        self.win._settings_saved = self._settings_snapshot()

    def _settings_snapshot(self):
        """Everything worth saving, as JSON-native values."""
        data = {"version": SETTINGS_VERSION}
        for key, getter, _set in self._setting_specs():
            try:
                data[key] = getter()
            except Exception:
                # Carry the last good value forward rather than dropping the
                # key: a getter that throws should not also erase what the
                # operator had.
                if key in self.win._settings_saved:
                    data[key] = self.win._settings_saved[key]
        return data

    def _save_settings(self, data=None):
        if data is None:
            data = self._settings_snapshot()
        if save_settings(data):
            self.win._settings_saved = data

    def _autosave_settings(self):
        """Periodic write, but only when something actually changed.

        Polled off the table rather than wired to thirty valueChanged signals,
        for two reasons: adding a setting stays one line, and a slider drag
        emits fifty changes a second, not one of which is worth a disk write.
        Comparing a ~30-entry dict every ten seconds is free next to the 60 ms
        render tick.
        """
        snap = self._settings_snapshot()
        if snap != self.win._settings_saved:
            self._save_settings(snap)
