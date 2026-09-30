"""The window's behaviour, one module per concern.

    devices.py      connect / disconnect, the two slots, calibration, stitch
    appearance.py   theme, plot layout, and the state of every button
    livefeed.py     acquisition, the displayed spectrum, the dark, saturation
    alignment.py    the sweep, the RAW stitched view, symmetry mode
    stage.py        manual stage control, units, backlash
    scanning.py     running a FROG scan and drawing it as it arrives
    export.py       writing the finished trace out
    persistence.py  settings.json

Each is a plain object holding one reference, `self.win`. They are a grouping of
behaviour, not an encapsulation boundary: the widgets live on the window and the
instrument lives in InstrumentState, so a controller reads `self.win.btn_feed`
and `self.win.spec` where the flat window method read `self.btn_feed` and
`self.spec`.

That is the honest shape for what this was — 4,000 lines relocated out of one
class. Giving each controller its own private state would have meant redesigning
every coupling at the same time as moving it, and the couplings in this program
are not accidental: a device swap has to re-arm the saturation lamp, refit the
spectrum, reseed the exposure boxes and repaint eight buttons, and those really
are four different concerns having to agree.

What the split buys is that each question has one place to look, and that each
file opens on a docstring about its concern instead of on line 2,300 of a
window. The couplings that remain are visible as `self.win.<other>.<call>`,
which is also the list of things to think about before changing one.

None of them is a QObject. Nothing here declares a signal — the two that exist
belong to the window and to the Avantes dialog — so there is no metaobject to
get wrong, and a controller can be constructed before any widget exists.
"""

from lillypad.gui.controllers.alignment import AlignmentController
from lillypad.gui.controllers.appearance import AppearanceController
from lillypad.gui.controllers.devices import DeviceController
from lillypad.gui.controllers.export import ExportController
from lillypad.gui.controllers.livefeed import LiveFeedController
from lillypad.gui.controllers.persistence import SettingsController
from lillypad.gui.controllers.scanning import ScanController
from lillypad.gui.controllers.stage import StageController

__all__ = ["DeviceController", "AppearanceController", "LiveFeedController",
           "AlignmentController", "StageController", "ScanController",
           "ExportController", "SettingsController"]
