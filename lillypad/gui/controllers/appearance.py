"""How the window looks: theme, layout, and the state of every button.

Two kinds of work, and they belong together because the second is entirely a
consequence of the first. The theme and the plot layout are the two global
appearance choices; everything else here is a `_refresh_*` that reads the
current state and puts the right icon, tooltip, enabled-ness and object name on
one button. There are two dozen of them and they were scattered through the
window between the logic they decorate.

They are all called the same thing on purpose. `_refresh_<thing>_button` is the
one place that decides what that button looks like, so a control that shows the
wrong state has exactly one place to look — and every one of them is safe to
call at any time, which is what lets a device swap end in a handful of refresh
calls rather than in a list of individual widget writes.

Two subtleties worth knowing before editing:

  * **Icons are per-theme files, sampled at paint time.** A theme switch has to
    re-ask for every one of them, which is what `_apply_theme` walks the
    refreshers for.
  * **`_position_panel_buttons` is geometry, not styling.** It reads the
    canvas's panel rectangles in device pixels, so it can only run after a draw
    — it hangs off the canvas's `axes_relaid` signal. Calling it earlier puts
    every panel header button in the wrong place.
"""

from lillypad.design.icons_runtime import glyph_icon as _glyph_icon

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFileDialog
from lillypad.design.geometry import (HDR_BTN, HDR_GAP, HDR_GROUP_GAP,
                                      HDR_PAD, HDR_TITLE_GAP)
from lillypad.design.icons import (ALIGN_ICON, HIDE_ICON, HORIZ_ICON,
                                   LOCKED_ICON, MERGE_ICON, MOON_ICON,
                                   PICTURE_ICON, REFRESH_ICON, RESCALE_ICON,
                                   SHOW_ICON, SPLIT_ICON, STITCH_ICON,
                                   SUN_ICON, UNLOCKED_ICON, VERT_ICON)
from lillypad.design.theme import install_theme
from lillypad.design.tokens import MEMBER_COLORS, PALETTE
from lillypad.gui.text import RIGHT_CLICK_HINT


class AppearanceController:
    """Theme, plot layout, and the state of every button in the window."""

    def __init__(self, win):
        self.win = win

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._apply_theme("light" if self.win._theme == "dark" else "dark")

    def _apply_theme(self, name):
        self.win._theme = name
        install_theme(QApplication.instance(), name)   # shared with startup
        self.win.canvas.apply_palette(PALETTE)
        self.win.lamp.update()          # repaints from the new PALETTE
        self.win.btn_theme.setIcon(QIcon(str(MOON_ICON if name == "light" else SUN_ICON)))
        self.win.btn_theme.setToolTip(
            "Switch to dark mode" if name == "light" else "Switch to light mode")
        self._refresh_overlay_button()   # its icons are per-theme too
        self._refresh_layout_button()    # …and so are this one's
        self._refresh_feed_button()      # …and its glyph is drawn from PALETTE
        self._refresh_align_button()     # …and it has one file per theme
        self._refresh_align_refresh_icon()  # …as does its re-run button
        self._refresh_autofit_icon()     # …and the auto-fit mark
        self._refresh_autostitch_icon()  # …as does this one
        self._refresh_autoscale_icon()   # …and this one
        self._refresh_freeze_icon()      # …the + is drawn from PALETTE too
        self._refresh_reference_button() # …and the eye has one file per theme
        self._refresh_picture_icon()     # …as does the camera
        self._size_moving_label()        # its font came from the stylesheet
        self.win.pnl_no_spec.refresh_theme()  # …and its watermark is tinted live
        self.win.status.showMessage(f"{name.capitalize()} mode.", 2000)

    # ── Plot layout ──────────────────────────────────────────────────────────
    def _toggle_layout(self):
        mode = ("vertical" if self.win.canvas.layout_mode == "horizontal"
                else "horizontal")
        self._set_layout(mode)
        self.win.status.showMessage(
            "Horizontal layout — trace and autocorrelation share the delay axis."
            if mode == "horizontal" else
            "Vertical layout — autocorrelation full width below.", 4000)

    def _set_layout(self, mode):
        """Apply a layout mode and keep its button honest.

        Split out of _toggle_layout so a restored mode takes exactly the same
        path as a click — minus the status message, which at startup would push
        the 'no hardware connected' banner off the bar before it was read.
        """
        if mode not in ("horizontal", "vertical"):
            raise ValueError(mode)          # a bad settings key, caught upstream
        self.win.canvas.set_layout_mode(mode)
        self._refresh_layout_button()

    def _refresh_layout_button(self):
        """Icon and tooltip of the plot-layout toggle. Like the theme and
        overlay buttons, it advertises the layout a CLICK produces, so the
        vertical icon is shown while the horizontal layout is active."""
        horiz = self.win.canvas.layout_mode == "horizontal"
        icon = (VERT_ICON if horiz else HORIZ_ICON)[self.win._theme]
        if icon.exists():
            self.win.btn_layout.setIcon(QIcon(str(icon)))
            self.win.btn_layout.setText("")
        else:
            self.win.btn_layout.setIcon(QIcon())
            self.win.btn_layout.setText("▤" if horiz else "▥")
        self.win.btn_layout.setToolTip(
            "Vertical layout: autocorrelation full width below the spectrum "
            "and FROG trace" if horiz else
            "Horizontal layout: FROG trace above the autocorrelation, sharing "
            "one delay axis, beside a full-height spectrum")

    # ── Picture of the figure ────────────────────────────────────────────────
    def _refresh_picture_icon(self):
        """Per-theme camera on the picture button, with the same text fallback
        every other icon button carries. Only the icon is re-read here — unlike
        its two neighbours this button has one state, so its tooltip is fixed
        and set once."""
        icon = PICTURE_ICON[self.win._theme]
        if icon.exists():
            self.win.btn_picture.setIcon(QIcon(str(icon)))
            self.win.btn_picture.setText("")
        else:
            self.win.btn_picture.setIcon(QIcon())
            self.win.btn_picture.setText("PNG")
        self.win.btn_picture.setToolTip(
            "Save the plots as a high-resolution PNG — the FROG trace with the "
            "autocorrelation below it and the spectrum beside them, in the "
            "current theme. Opens on the Export window's file name.")

    def _save_picture(self):
        """Ask for a file and print the plots into it.

        Opens on the Export window's file name — same stem, same date prefix,
        only the suffix differs — so a scan and the picture of it land side by
        side in a folder listing under one name instead of needing the two to be
        matched up by timestamp afterwards.

        Deliberately NOT gated on self.win.result the way the data exports are: a
        picture of the live spectrum, or of a trace still filling in, is a
        perfectly good thing to want — there is always something on the panels
        worth a picture, which is not true of a scan file.
        """
        path, _ = QFileDialog.getSaveFileName(
            self.win, "Save figure as PNG", self.win.dlg_export.file_name(".png"),
            "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.win.status.showMessage(f"Saving figure → {path} …", 0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # repaint(), not the usual deferred update: the save below runs on this
        # thread and holds the event loop for as long as it takes, so a message
        # left for the next paint would first appear once it was already stale.
        self.win.status.repaint()
        try:
            w, h = self.win.canvas.save_picture(path)
        except Exception as e:
            # Broad: matplotlib raises OSError for a path it cannot write and
            # ValueError for a format it will not, and neither is worth a
            # traceback to a console this build does not have.
            self.win.status.showMessage(f"Figure save failed: {e}", 8000)
        else:
            # The size is reported because it is not simply PICTURE_DPI times a
            # fixed page: it follows the window, and above PICTURE_MAX_PX the
            # dpi gives way instead of the page growing.
            self.win.status.showMessage(f"Saved → {path}  ({w} × {h} px)", 5000)
        finally:
            QApplication.restoreOverrideCursor()

    # ── Saturation indicator ─────────────────────────────────────────────────
    def _set_lamp(self, state, text, obj, which=0):
        lamp = self.win.lamp if which == 0 else self.win.lamp2
        lbl  = self.win.lbl_sat if which == 0 else self.win.lbl_sat2
        lamp.set_state(state)
        lbl.setText(text)
        # Colour via objectName + repolish (not setStyleSheet) so a later theme
        # switch restyles the label along with everything else.
        if lbl.objectName() != obj:
            lbl.setObjectName(obj)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def _sync_connection_ui(self):
        """Show the Connect buttons or the controls, per what is connected.

        The side panels and the "no spectrometer connected" message all say the
        same thing, so one function owns all three. Called from _sync_multi_ui
        (every spectrometer swap ends there) and from _apply_stage.
        """
        have_spec, have_stage = self.win._have_spec(), self.win._have_stage()
        self.win._body_spectrum.setVisible(have_spec)
        self.win.btn_connect_spec.setVisible(not have_spec)
        self.win._body_stage.setVisible(have_stage)
        self.win.btn_connect_stage.setVisible(not have_stage)
        self.win.pnl_no_spec.setVisible(not have_spec)
        if not have_spec:
            self.win.pnl_no_spec.raise_()
        # The header buttons act on a spectrum that may not exist; and the
        # message just claimed the panel, so the row has to be re-packed
        # around whatever is left showing.
        for b in (self.win.btn_autofit, self.win.btn_feed, self.win.btn_align_spec):
            b.setEnabled(have_spec)
        self._position_panel_buttons()
        if self.win.dlg_spec.isVisible():
            self.win.dlg_spec.refresh()
        if self.win.dlg_stage.isVisible():
            self.win.dlg_stage.refresh()
        if self.win.dlg_sim.isVisible():
            self.win.dlg_sim.refresh()

    def _sync_multi_ui(self):
        """Bring every stitched-vs-single widget into line with self.win.spec.

        One owner for the second integration spinbox and the overlay button,
        so no caller has to remember the set. Pure widget state — no device
        I/O, so it is safe to call outside _device_lock.
        """
        stitched = self.win._pair_live()
        slots = self.win.devices._slot_members() if stitched else []
        # One caption over both boxes — S1 on the left, S2 on the right, each
        # in ITS OWN curve's colour. Two boxes under one caption is the compact
        # arrangement (see _build_spectrum_group), but it leaves the reader to
        # infer which box is which device from position alone; matching the
        # names to the curves on the panel says it outright. Rich text rather
        # than two labels, so the caption stays one widget on one line.
        #
        # MEMBER_COLORS is theme-invariant by design, so this needs no
        # _apply_theme hook — see the note beside line_m1/line_m2 in
        # FrogCanvas.apply_palette.
        self.win.lbl_integration.setText(
            (f"Integration Time — "
             f"<span style='color:{MEMBER_COLORS[0]}'><b>S1</b></span> / "
             f"<span style='color:{MEMBER_COLORS[1]}'><b>S2</b></span>")
            if stitched else "Integration Time")
        self.win.spin_integration2.setVisible(stitched)
        # Second saturation lamp: one alarm per device, so it exists exactly as
        # long as the pair does.
        self.win.lbl_sat2.setVisible(stitched)
        self.win.lamp2.setVisible(stitched)
        if len(slots) != 2:
            # No live pair: the per-spectrometer view has nothing to show.
            if self.win.btn_overlay.isChecked():
                with QSignalBlocker(self.win.btn_overlay):
                    self.win.btn_overlay.setChecked(False)
            self.win._overlay_on = False
            self.win.canvas.clear_members()
        elif not self.win._pair_seen:
            # A pair that has just come up starts UNSTITCHED — one curve per
            # spectrometer. The combined curve hides the one thing a fresh pair
            # most needs checking: whether the two halves actually agree across
            # the overlap. Seeing them apart is what makes a bad stitch factor
            # or a dark half obvious, and Auto-stitch is judged against it.
            #
            # Only on the way IN, hence the latch: this runs on every device
            # sync, and re-asserting the default there would override the
            # operator every time anything else about the pair changed.
            with QSignalBlocker(self.win.btn_overlay):
                self.win.btn_overlay.setChecked(True)
            self.win._overlay_on = True
        # Latch AFTER the branch above, and cleared whenever the pair goes, so
        # the next one to come up gets the default again.
        self.win._pair_seen = len(slots) == 2
        self.win.feed.sync_integration_ui()
        self._refresh_overlay_button()
        self._refresh_autostitch_button()
        self._refresh_align_trace_button()
        self._refresh_avantes_button()
        self._sync_connection_ui()

    def _refresh_autostitch_icon(self):
        """Per-theme mark on the Auto-stitch button, with the same text
        fallback the other icon buttons keep for a bundle missing its icons."""
        icon = STITCH_ICON[self.win._theme]
        if icon.exists():
            self.win.btn_autostitch.setIcon(QIcon(str(icon)))
            self.win.btn_autostitch.setText("")
        else:
            self.win.btn_autostitch.setIcon(QIcon())
            self.win.btn_autostitch.setText("⇌")

    def _refresh_autostitch_button(self, running=None):
        """Show the Auto-stitch header button only while a pair is live, and
        enable it only while no scan owns the hardware it would drive.

        `running` overrides the inferred state for the same reason
        _refresh_symmetry_button takes it: _reset_scan_ui runs from the slot for
        finished_scan, which the worker emits INSIDE run() — before the finally
        that parks the stage. Whether the QThread has left run() by the time the
        GUI thread picks that signal up is a race, and on real hardware (where
        parking is a stage move measured in seconds) it reliably has not, so
        _scan_running() reads True and this re-greyed the button immediately
        after the scan that was over. Nothing refreshed it again, so it stayed
        dead until a device swap or a theme switch happened along.
        """
        if running is None:
            running = self.win._scan_running()
        self.win.btn_autostitch.setVisible(self.win._pair_live()
                                       and len(self.win.devices._slot_members()) == 2)
        self.win.btn_autostitch.setEnabled(not running)
        self.win.btn_autostitch.setToolTip(
            "Auto-stitch — fit the factor that matches the two spectrometers "
            "across the overlap. Needs light across the overlap region; the "
            "fitted factor and its mismatch are shown in the Spectrometer "
            "window.")
        self._position_panel_buttons()   # its visibility drives the header row

    def _refresh_avantes_button(self):
        """Show the Avantes toolbar button only while an Avantes is live.

        Called from _sync_multi_ui because that is the one function guaranteed
        to run after every device swap (_apply_spectrometer, _connect_multi_pair
        and the slot handlers all end in it).
        """
        devices = self.win.devices.avantes_devices()
        self.win.btn_avantes.setVisible(bool(devices))
        if not devices and self.win.dlg_avantes.isVisible():
            self.win.dlg_avantes.hide()      # its target just went away
        elif devices and self.win.dlg_avantes.isVisible():
            self.win.dlg_avantes.refresh()  # a swap may have changed which device

    def _refresh_overlay_button(self):
        """Icon, tooltip and availability of the per-spectrometer view toggle.

        The icon shows the view a CLICK produces, not the current one: broken
        spectrum while the combined curve is up, continuous once the members
        are drawn apart.
        """
        stitched = self.win._pair_live()
        self.win.btn_overlay.setVisible(stitched and len(self.win.devices._slot_members()) == 2)
        # Live through a scan, unlike Auto-stitch beside it: this only chooses
        # which curves the panel draws, and _render_scan_frame now honours the
        # choice from the scan's own member frames. Auto-stitch has to stay
        # disabled — it drives the hardware the worker owns.
        on = self.win.btn_overlay.isChecked()
        icon = (MERGE_ICON if on else SPLIT_ICON)[self.win._theme]
        if icon.exists():
            self.win.btn_overlay.setIcon(QIcon(str(icon)))
            self.win.btn_overlay.setText("")
        else:
            self.win.btn_overlay.setIcon(QIcon())
            self.win.btn_overlay.setText("∿" if on else "⌇")
        self.win.btn_overlay.setToolTip(
            "Show the combined stitched spectrum as one curve" if on else
            "Show each spectrometer as its own curve — compare the two across "
            "the overlap to judge the stitch factor")
        # Its visibility just changed, and the header row is packed
        # right-to-left around whatever is showing.
        self._position_panel_buttons()

    # ── Alignment mode: raw stitched trace ───────────────────────────────────
    def _position_panel_buttons(self):
        """Lay out each panel's header row: the buttons that act on that panel,
        on the line the panel title occupies.

        Two groups per panel. Most buttons are right-aligned to the panel's
        right edge. The two that describe the Y AXIS — the auto-scale padlock
        and the log/linear toggle — are packed immediately AFTER the title
        instead, so they read as part of that panel's name rather than as two
        more entries in a row of six; the clear span between the two groups is
        what separates "what this panel's y axis is doing" from "things you can
        do to this panel".

        Every edge here moves with the split fraction, the window size and the
        layout mode, so this runs off canvas.axes_relaid — and off the refresh
        methods that show or hide a button, because each row is packed from its
        own edge and its width depends on which buttons are visible.
        """
        # getattr, not the attribute: the refresh methods that call this also
        # run from _apply_spectrometer, which the connect paths can reach
        # before _build_ui has created any of these buttons.
        if getattr(self.win, "pnl_no_spec", None) is None:
            return
        c = self.win.canvas

        def _row_y(top):
            # The band sits ABOVE the axes: HDR_PAD of clear space between the
            # buttons' bottom edge and the top spine.
            return round(top) - HDR_PAD - HDR_BTN

        def pack(ax, buttons):
            """Right-align `buttons` to the panel's right edge. Returns the x of
            the leftmost one placed, which is where the clear span before it
            begins."""
            _left, top, right = c.panel_edges_px(ax)
            y = _row_y(top)
            x = round(right) - 6
            for b in reversed(buttons):
                # isHidden, not isVisible: isVisible() is False for every child
                # until the top-level window itself is shown, so packing on it
                # skipped the whole row on the layout passes that run during
                # __init__ and left the buttons piled at (0, 0) until something
                # else happened to trigger a relayout.
                if b.isHidden():
                    continue      # hidden buttons take no room in the row
                x -= b.width()
                b.move(x, y)
                x -= HDR_GAP
            return x + HDR_GAP        # undo the trailing gap

        def pack_after_title(ax, buttons, limit):
            """Lay `buttons` out left to right starting just past the title.

            `limit` is the x the group must not reach — the leftmost button of
            the right-hand group. On a narrow panel the whole group slides left
            to respect it, over the title if it has to: a control half off the
            panel edge, or overlapping a row it does not belong to, is worse
            than one sitting on its own title for the few pixels it takes.
            """
            _left, top, _right = c.panel_edges_px(ax)
            y = _row_y(top)
            start = round(c.title_right_px(ax)) + HDR_TITLE_GAP
            shown = [b for b in buttons if not b.isHidden()]
            if not shown:
                return
            width = sum(b.width() for b in shown) + HDR_GAP * (len(shown) - 1)
            start = min(start, round(limit) - HDR_GROUP_GAP - width)
            x = max(start, 0)
            for b in shown:
                b.move(x, y)
                x += b.width() + HDR_GAP

        spec_right = pack(
            c.ax_spec, [self.win.btn_autofit, self.win.btn_feed, self.win.btn_freeze,
                        self.win.btn_refshow, self.win.btn_autostitch,
                        self.win.btn_overlay, self.win.btn_align_spec,
                        self.win.btn_align_refresh])
        pack(c.ax_trace, [self.win.btn_align_trace, self.win.btn_symmetry])
        # The y-axis pair, tucked against the title it qualifies. Padlock first:
        # it is the one that changes what the axis DOES, the scale toggle only
        # how it is drawn.
        pack_after_title(c.ax_spec, [self.win.btn_autoscale, self.win.btn_logscale],
                         spec_right)

        # The "no spectrometer connected" panel, stretched over the whole
        # spectrum axes so it reads as the empty plot itself rather than a card
        # floating in it. It is laid out only here — it has no parent layout to
        # do that for it.
        self.win.pnl_no_spec.setGeometry(c.panel_rect_px(c.ax_spec))

    def _refresh_autofit_icon(self):
        """Per-theme mark on the one-shot auto-fit button, with the text
        fallback it used to carry if the icons are missing from the bundle."""
        icon = RESCALE_ICON[self.win._theme]
        if icon.exists():
            self.win.btn_autofit.setIcon(QIcon(str(icon)))
            self.win.btn_autofit.setText("")
        else:
            self.win.btn_autofit.setIcon(QIcon())
            self.win.btn_autofit.setText("↔↕")

    def _refresh_autoscale_icon(self):
        """Padlock matching the axis' CURRENT state — open while it follows the
        data, shut while it is frozen. Per theme, and re-read on every state
        change, so this is called from _refresh_autoscale_button rather than
        only from the theme switch."""
        on = bool(self.win.canvas.autoscale_y)
        icon = (UNLOCKED_ICON if on else LOCKED_ICON)[self.win._theme]
        if icon.exists():
            self.win.btn_autoscale.setIcon(QIcon(str(icon)))
            self.win.btn_autoscale.setText("")
        else:
            self.win.btn_autoscale.setIcon(QIcon())
            self.win.btn_autoscale.setText("🔓" if on else "🔒")

    def _refresh_freeze_icon(self):
        """The + mark, drawn from the live PALETTE so a theme switch reruns it
        — same arrangement as the feed button's play/stop glyph."""
        self.win.btn_freeze.setIcon(_glyph_icon("plus", PALETTE["accent"]))

    def _refresh_reference_button(self):
        """Visibility, icon and tooltip of the reference's hide/show toggle.

        The icon shows the ACTION its two files were drawn for (see SHOW_ICON):
        a reference on screen offers the struck-through eye, a hidden one the
        open eye.
        """
        c = self.win.canvas
        have = c.has_reference()
        self.win.btn_refshow.setVisible(have)
        if have:
            shown = c.reference_shown()
            icon = (HIDE_ICON if shown else SHOW_ICON)[self.win._theme]
            if icon.exists():
                self.win.btn_refshow.setIcon(QIcon(str(icon)))
                self.win.btn_refshow.setText("")
            else:
                self.win.btn_refshow.setIcon(QIcon())
                self.win.btn_refshow.setText("◉" if shown else "○")
            self.win.btn_refshow.setToolTip(
                "Hide the frozen reference spectrum" if shown else
                "Show the frozen reference spectrum")
        # Its visibility just changed, and the header row is packed
        # right-to-left around whatever is showing.
        self._position_panel_buttons()

    def _refresh_logscale_button(self):
        """Label and tooltip of the log/linear toggle. Reads the checkbox, not
        the axes: chk_log is the single source of truth for the scale."""
        on = self.win.dlg_graphics.log_scale
        self.win.btn_logscale.setText("log" if on else "lin")
        self.win.btn_logscale.setToolTip(
            "Spectrum Y axis is LOGARITHMIC — click for linear." if on else
            "Spectrum Y axis is LINEAR — click for logarithmic.")

    def _refresh_autoscale_button(self):
        """Show the canvas' auto-scale state on the header toggle.

        Signals blocked: this is called FROM limits_changed, which reset_axes
        emits, so letting the setChecked re-enter the click handler would
        toggle the mode straight back off again.
        """
        on = bool(self.win.canvas.autoscale_y)
        with QSignalBlocker(self.win.btn_autoscale):
            self.win.btn_autoscale.setChecked(on)
        self._refresh_autoscale_icon()   # the padlock IS the state readout
        self.win.btn_autoscale.setToolTip(
            ("Y axis UNLOCKED — the spectrum's Y axis follows the live signal.\n"
             "Click to lock it where it is."
             if on else
             "Y axis LOCKED — the spectrum's Y axis stays where it is.\n"
             "Click to unlock it: fit the data, then follow it.")
            + "\n" + RIGHT_CLICK_HINT)

    def _refresh_align_refresh_icon(self):
        """Per-theme mark on the re-run button, with the usual text fallback."""
        icon = REFRESH_ICON[self.win._theme]
        if icon.exists():
            self.win.btn_align_refresh.setIcon(QIcon(str(icon)))
            self.win.btn_align_refresh.setText("")
        else:
            self.win.btn_align_refresh.setIcon(QIcon())
            self.win.btn_align_refresh.setText("⟳")

    def _refresh_align_button(self):
        """Per-theme crosshair on the alignment-sweep toggle.

        Unlike the overlay and layout buttons this one depicts a MODE, so the
        mark does not change with the button's state — only with the theme,
        each file carrying that theme's accent. Falls back to the Δ it used to
        show if the icons are missing from the bundle.
        """
        icon = ALIGN_ICON[self.win._theme]
        if icon.exists():
            self.win.btn_align_spec.setIcon(QIcon(str(icon)))
            self.win.btn_align_spec.setText("")
        else:
            self.win.btn_align_spec.setIcon(QIcon())
            self.win.btn_align_spec.setText("Δ")

    def _refresh_feed_button(self):
        """Icon, colour and tooltip of the live-feed toggle.

        Like the theme and layout buttons it advertises the CURRENT state, not
        the state a click produces: a running feed is the red stop square, a
        stopped one the accent play triangle. Both glyphs are drawn from the
        live PALETTE, so a theme switch re-generates them.
        """
        running = self.win.btn_feed.isChecked()
        self.win.btn_feed.setIcon(_glyph_icon(
            "stop" if running else "play",
            PALETTE["danger"] if running else PALETTE["accent"]))
        self.win.btn_feed.setToolTip("Stop the live spectrum feed" if running
                                 else "Start the live spectrum feed")
        # Colour via objectName + repolish, not setStyleSheet, so a later theme
        # switch restyles it along with everything else.
        name = "overlaydanger" if running else "overlay"
        if self.win.btn_feed.objectName() != name:
            self.win.btn_feed.setObjectName(name)
            self.win.btn_feed.style().unpolish(self.win.btn_feed)
            self.win.btn_feed.style().polish(self.win.btn_feed)

    def _refresh_align_trace_button(self):
        """Show the RAW toggle only for a stitched pair, and only enable it
        once there is a scan whose raw member columns were recorded."""
        stitched = self.win._pair_live()
        self.win.btn_align_trace.setVisible(stitched)
        self._position_panel_buttons()   # visibility drives the header row
        if not stitched:
            return
        ready = self.win._align_trace is not None
        self.win.btn_align_trace.setEnabled(ready)
        self.win.btn_align_trace.setToolTip(
            "Raw counts — no calibration, no dark subtraction. Each "
            "spectrometer is normalised to its own maximum and the two are "
            "cut (not blended) at the middle of the overlap, so features "
            "buried in the background show up.\nDisplay only: the recorded "
            "trace and every export are unaffected."
            if ready else
            "Run a scan first — the raw per-spectrometer columns are recorded "
            "as it goes.")

    def _refresh_symmetry_button(self, running=None):
        """Enable the symmetry toggle only when there is a finished scan to fold.

        A scan in progress is excluded even though self.win.result may still hold the
        PREVIOUS one: the panel is showing the new trace being filled in, and a
        symmetry view of the old result over the top of it would be a picture of
        something the operator is no longer looking at.

        `running` overrides the inferred state at the two moments _scan_running()
        cannot answer for it. A scan about to start has not created its worker
        yet, so the flag still describes the PREVIOUS scan and reads False; and
        _reset_scan_ui runs from the worker's own finished signal, by which point
        the QThread has not necessarily left run() and it reads True. Inferring
        at either would leave the button in the wrong state until some unrelated
        refresh happened along.
        """
        if running is None:
            running = self.win._scan_running()
        ready = self.win.result is not None and not running
        self.win.btn_symmetry.setEnabled(ready)
        self.win.btn_symmetry.setToolTip(
            "Symmetry mode — the measured trace left of the fold line, and "
            "|T(+d) − T(−d)| right of it: how far the two temporal halves are "
            "from being each other's mirror image.\nBoth halves share one "
            "colour scale, so a dark right half genuinely means a symmetric "
            "trace. The corner readout is the asymmetry as a fraction of the "
            "whole trace.\nThe line starts on the autocorrelation peak — drag "
            "it to fold somewhere else, double-click it to put it back.\n"
            "Display only: the recorded trace and every export are unaffected."
            if ready else
            "Run a scan first — symmetry mode folds a finished trace about its "
            "autocorrelation peak.")

    def _set_hardware_buttons_enabled(self, on):
        """Every route to a device swap, locked together.

        A scan and an alignment sweep both capture the stage and spectrometer
        for their duration, so nothing may connect, disconnect or simulate
        underneath them — and that is now five buttons across the toolbar and
        the two side panels rather than the one it used to be.
        """
        for w in (self.win.btn_spec_dlg, self.win.btn_stage_dlg, self.win.btn_sim_dlg,
                  self.win.btn_connect_spec, self.win.btn_connect_stage):
            w.setEnabled(on)

    def _size_moving_label(self):
        """Reserve the width of the WIDER of the stage label's two states.

        The label shows "IDLE" at rest, so its own size hint asks for four
        characters — and the layout, which has no reason to expect a fifth,
        hands it exactly that when the row is tight. "MOVING" then does not fit,
        and because the label is centred it is clipped at both ends, which is
        what turned it into "OVING".

        Measured rather than hard-coded: the #idle/#moving rules set the size
        and weight in the stylesheet, so the number changes with the theme —
        hence the call from _apply_theme as well as from the build.
        """
        fm = self.win.lbl_moving.fontMetrics()
        # +4: QLabel's own text margin, which fontMetrics does not include.
        self.win.lbl_moving.setMinimumWidth(fm.horizontalAdvance("MOVING") + 4)
