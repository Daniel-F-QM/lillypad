# Lillypad — architecture

For someone about to change this code. The [README](README.md) covers what the
program does and how to use it; this covers how it is put together, which
boundaries are real, and where to put a new thing.

---

## The dependency rule

```
                      main.py
                         │
                 lillypad/app/main.py
                         │
                 lillypad/gui/window/
                         │
          ┌──────────────┼──────────────┐
          │              │              │
  lillypad/gui/      controllers/   dialogs/ canvas/ widgets
   (leaves)               │
                          │
          ┌───────────────┼───────────────┐
          │               │               │
  lillypad/scan/   lillypad/hardware/  lillypad/sim/
          │               │               │
          └───────────────┴───────┬───────┘
                                  │
                      lillypad/hardware/base.py
                                  │
                 lillypad/paths.py · settings.py

      lillypad/design/  ──→  (nothing in this diagram)
```

**Imports run downward only.** Four parts of that are load-bearing rather than
stylistic, and each has a test:

| Rule | Why | Test |
| --- | --- | --- |
| `design/` imports nothing from `gui/` | `tokens.py` and `mpl_style.py` must be readable without a `QApplication`, and `FrogCanvas` reads `PALETTE` | `test_design.py::TestLayerBoundary` |
| `hardware/` has no Qt and no vendor SDK at module scope | `import lillypad.hardware` must succeed with only numpy, and a missing DLL must cost one device's connect call, not the program | `test_layering.py` |
| `scan/core.py` has no Qt | the physics is testable in milliseconds, with PySide6 not installed | `test_layering.py` |
| `sim/` imports only `hardware/base.py` | the simulator is both the production simulator and the suite's test doubles, and must not pull a device layer in | `test_layering.py` |

`test_layering.py` checks these in **subprocesses** with a `sys.meta_path`
finder that raises `ImportError` for the blocked package. That is the only way
to check it honestly: inside one interpreter the module is already in
`sys.modules` from some earlier test.

---

## The layers, bottom up

### `paths.py` — where everything is

The **only** `__file__` anchor in the repository. Two functions that answer two
different questions, and conflating them is the bug this module exists to
prevent:

- `resource_path(*parts)` — a **bundled, read-only** resource. Checks
  `sys._MEIPASS` first.
- `app_dir()` — the **writable** folder beside the program. Checks `sys.frozen`
  first.

Both orderings are load-bearing: in a frozen bundle the package modules live in
the PYZ and `__file__` is synthetic, so the repo-root fallback would evaluate to
`_MEIPASS` itself. Neither function ever reaches it when frozen.

### `design/` — every visual decision

No colour, font size, radius or icon path appears anywhere else in the program.

- `tokens.py` — `DARK_PALETTE`, `LIGHT_PALETTE`, the live `PALETTE`, the type
  scale, the metrics, the plot colours.
- `qss/app.qss` — the stylesheet, as text, with `$token` placeholders.
- `stylesheet.py` — loads it and substitutes.
- `theme.py` — installs a theme onto the `QApplication`.
- `mpl_style.py` — selects matplotlib's QtAgg backend, sets the rcParams, and
  holds the figure layout and typography constants.
- `icons.py` / `icons_runtime.py` — the per-theme icon files, and the three
  glyphs that are painted rather than loaded.

Two things here will bite:

**`PALETTE` is mutated in place and never rebound.** `install_theme` does
`PALETTE.clear(); PALETTE.update(...)`. Twenty-five call sites read the dict at
call time and share the object, so a `PALETTE = ...` anywhere breaks theme
switching silently, in one panel, later.

**The stylesheet is `string.Template`, not `str.format`.** QSS is made of
braces — the old f-string version had 75 `{{`/`}}` escape sites, and one miss
unstyles the whole app. `Template` ignores `{}` entirely and QSS has no `$`. A
literal dollar in a comment must be doubled (`$$`), and substitution uses
`substitute`, never `safe_substitute`, so a typo'd token raises at startup
instead of shipping an unstyled widget.

**`mpl_style` must be imported before any matplotlib figure class.** Importing it
is what selects the backend. Module import is idempotent, so every canvas module
simply opens with it and the order stops being a rule to remember.

### `hardware/` — the devices

```
base.py          StageBase, SpectrometerBase, the tagged-id format,
                 the calibration loader. No SDK, no Qt.
registry.py      enumeration, and the one vendor-tag → class mapping
stitched.py      two spectrometers presented as one
stages/          kinesis.py, zaber.py, piezo_jena.py
spectrometers/   seabreeze.py, avantes.py  (the adapters)
vendor/avaspec.py  the raw ctypes SDK wrapper — knows nothing about Lillypad
```

Devices are identified by a tagged id, `"vendor:serial"` — see `spec_ident` and
`split_spec_ident`. Two slots may hold devices from different vendors, so the
tag is what says which adapter opens which.

`StitchedSpectrometer` sits with the interfaces rather than under a vendor or
under `sim/`, because it composes two `SpectrometerBase` and is vendor-neutral
*and* simulator-neutral. `test_stitched.py` proves it by driving it with a
hand-rolled double.

`vendor/avaspec.py` is one 1,700-line module on purpose. Its DLL handle, lock,
init refcount, open-serial registry and `atexit` hook are only correct if
exactly one module object exists; a split risks two locks serialising nothing.

### `sim/` — the simulator

Top-level rather than under `hardware/`, because the ~700 lines of pulse physics
touch no device I/O and are the primary test target, and because
`SimulatedStage` / `SimulatedSpectrometer` are simultaneously the production
simulators *and* the suite's test doubles. `tests/` importing `lillypad.sim`
rather than `lillypad.hardware` states that.

`presets.py` keeps `"calibration": "SR_New"` as a bare filename stem and never
resolves it. Resolution stays in the caller, which keeps `sim/` free of
`paths.py` and keeps the physics tests independent of the repo's data files.

### `scan/` — the measurement

```
core.py     delay↔position, the delay grid, autocorrelation, FWHM,
            symmetry folding. numpy only.
model.py    FrogScanConfig and FrogResult
writers.py  .dwc / .npz / .csv
worker.py   FrogScanWorker(QThread) — the only Qt in this package
```

`FrogScanWorker` is deliberately **not** re-exported from `__init__.py`.
Importing any submodule runs that file first, so one `from .worker import ...`
would pull Qt into `import lillypad.scan.core` and undo the whole point of the
split. The worker is imported from `lillypad.scan.worker` by name, which also
makes every Qt-dependent caller visible to a grep.

The three export formats differ in more than syntax:

- **`.dwc`** and **`.csv`** carry the dark-corrected trace, and both flip a
  high-to-low scan to ascending delay, because `.dwc` records only a positive
  increment. With the Export window's trim option (off by default) they also
  drop the wavelength rows at either end that are zero at every delay.
- **`.npz`** carries the RAW counts, the dark that would be subtracted, and the
  delay-extreme reference frames if they were taken — **in acquisition order**.
  Line an archive up against a `.dwc` by delay *value*, not by column index.

### `gui/` — the interface

```
state.py        InstrumentState: what the whole program has to agree about
widgets.py      the small custom widgets
workers.py      all three QThreads, together
perf.py         the opt-in performance probe (LILLYPAD_PERF=1)
text.py         two long user-facing strings
canvas/         FrogCanvas, as one concrete class plus five mixins
dialogs/        the nine pop-up windows
controllers/    the window's behaviour, one module per concern
window/         FrogWindow, and the one-time act of building it
```

#### `InstrumentState`

Which devices are connected, what a scan is configured to do, what the recorded
dark is and at what exposure, what the two spectrometer slots hold, and what the
simulator stands in for. `FrogWindow` keeps a property forwarder for every
field, so `self.spec` still works everywhere; a dialog or a controller can be
handed the state object alone and be unable to reach a widget.

Deliberately **not** everything the window has. The render buffers, the
pending-fit flags and the alignment view belong to one panel each; gathering
those too would have made the file a second copy of `__init__`.

Two invariants it is the place to state:

- **There is no "multi-spectrometer mode" flag.** Both slots exist from the
  start, slot 1 alone *is* single-spectrometer mode, and "is a pair live" is
  answered by `pair_live()` off the connected device.
- **`None` means nothing is connected** — a real absence, not a stand-in device
  that could quietly return plausible-looking counts.

#### `canvas/`

One concrete `FrogCanvas` assembled from five mixins. Mixins rather than
collaborators because this is genuinely one object's state: one `Figure`, one
cached blit background shared by three axes, one palette, one layout solver, and
the panels read each other.

**All eight `Signal`s stay on the concrete class.** Qt builds a class's
metaobject from a `QObject` subclass, so a `Signal` declared on a plain mixin is
never connected — silently, at runtime, in one feature.

**Do not simplify the blitting.** The animated-artist selection, the threshold
before auto-Y escalates a frame to a full redraw, and the coalescing of render
requests are measured optimizations worth about 80 ms per scan column.

#### `controllers/`

Eight plain objects, each holding one reference, `self.win`:

| Controller | `win` attribute | Owns |
| --- | --- | --- |
| `DeviceController` | `devices` | connect / disconnect, the two slots, calibration, the stitch |
| `AppearanceController` | `appearance` | theme, plot layout, the state of every button |
| `LiveFeedController` | `feed` | acquisition, the displayed spectrum, the dark, saturation |
| `AlignmentController` | `alignment` | the sweep, the RAW stitched view, symmetry mode |
| `StageController` | `stage_ctl` | manual stage control, units, backlash |
| `ScanController` | `scanning` | running a scan and drawing it as it arrives |
| `ExportController` | `export` | writing the finished trace out |
| `SettingsController` | `settings_ctl` | `settings.json` |

They are a grouping of **behaviour, not an encapsulation boundary.** The widgets
live on the window and the instrument lives in `InstrumentState`, so a
controller reads `self.win.btn_feed` and `self.win.spec`.

That is deliberate, and the alternative looks tidier and is not. Giving each
controller private state would have meant redesigning every coupling at the same
time as moving it, and the couplings here are not accidental: a device swap has
to re-arm the saturation lamp, refit the spectrum, reseed the exposure boxes and
repaint eight buttons, and those genuinely are four concerns having to agree.
They are now visible as `self.win.<other>.<call>`, which is also the list of
things to think about before changing one.

`stage_ctl` and `settings_ctl` carry the suffix because `win.stage` is the delay
stage and `win.settings` is the loaded preferences dict.

None of them is a `QObject` and none declares a signal, so there is no
metaobject to get wrong and one can be constructed before any widget exists.

#### `window/`

`main_window.py` is the wiring diagram: `__init__` in order — state, controllers,
dialogs, `build_ui`, the feed and the timers, `_restore_settings` last. Each step
is commented with why it sits where it does.

`build_ui.py` is free functions taking `win`, not a class. It is 375 lines of
pure construction assigning some two hundred attributes onto the window; a class
would have added `self.win.` to every line and held no state, because the window
*is* the state.

Three things in there not to tidy:

- **`hasattr(win, "dlg_graphics")` is an init-order guard**, not defensiveness.
  Three methods called during construction touch a dialog that does not exist
  yet. Turning one into a plain attribute access is an `AttributeError` at
  startup.
- **`ExportDialog` calls back into the window during its own `__init__`** (the
  format radios), so its construction position is load-bearing.
- **`position_panel_buttons` reads the canvas's panel rectangles in device
  pixels**, so it can only run after a draw — it hangs off the canvas's
  `axes_relaid` signal.

---

## Three recipes

### Add a device

1. Write the adapter under `lillypad/hardware/stages/` or
   `lillypad/hardware/spectrometers/`, subclassing `StageBase` or
   `SpectrometerBase` from `base.py`.
2. **Import the vendor SDK inside the function that needs it**, never at module
   scope. A literal `import` in a function body — PyInstaller's analysis scans
   function bodies, so this is what keeps the driver in the frozen graph.
3. For a spectrometer, add the vendor tag to `SPEC_VENDORS` and a branch to
   `open_spectrometer` in `registry.py`. **A literal `if`/`elif` over imported
   classes** — a dict of module names or an `importlib.import_module` would drop
   every driver from the frozen build and fail only on the bench.
4. Moves must **block until the stage has settled**, so the scan loop can read
   back a position it can trust on the next line.
5. Add a case to `tests/test_registry.py`. Enumeration must degrade to `[]` when
   the SDK is missing — except `list_kinesis_stages()`, which raises on purpose,
   because it is an explicit operator-requested scan and `[]` would tell someone
   with a missing driver that they have no hardware.

### Add a theme token

1. Add the key to **both** `DARK_PALETTE` and `LIGHT_PALETTE` in
   `design/tokens.py` (or to `METRICS` / `TYPE` if it is a size rather than a
   colour).
2. Use `$yourtoken` in `design/qss/app.qss`, or read `PALETTE["yourtoken"]` at
   call time from Python. **Never** cache the value at import time and never
   rebind `PALETTE`.
3. `tests/test_design.py` will fail if the two palettes disagree about their
   keys, if a `$token` has no value, or if a raw hex colour appears in the
   `.qss`.

### Add a panel to the figure

1. Create the axes in `canvas/figure.py`'s `__init__`, with the other artists.
2. Give it a mixin in `canvas/panels.py` — `update_*`, `clear_*`, and whatever
   it needs. Every method must be defined **exactly once** across the six
   classes; `tests/test_canvas.py` checks that, because a duplicate resolves
   silently to whichever comes first in the MRO.
3. Add its rectangle to `canvas/layout.py`'s solver. Everything there is in
   *figure fractions computed from the current pixel size* — three of the
   quantities involved are fixed pixel widths, so a fraction that was right at
   1600 px is wrong at 900 px.
4. If it needs a live artist, add it to `self._animated` and read `_blit` and
   `_blit_spec` first.
5. If it needs limits, put the setters in `canvas/axes_api.py` beside the
   others, and remember the pairing: turning auto-scale off must pin the current
   view, and typing a bound must turn auto-scale off.

---

## Things that have gone wrong before

- **A shim that redefined a class instead of re-exporting it.** There are eleven
  `isinstance` probes on the device classes; a second class object makes every
  one silently false — a spectrum that looks right and a "no device" overlay
  that never lifts.
- **`importlib.import_module` or a string→module table in the device path.** It
  works from source and drops the driver from the frozen build, where the only
  signal is `build/Lillypad/xref-Lillypad.html`.
- **Removing an import the unit tests never reach.** 354 tests passed with
  `import matplotlib` gone and the program died on the first `FrogCanvas`. That
  is what `tests/test_app_smoke.py` is for.
- **Treating a bracketed frame as a dark.** The scan used to record a
  "background" at the first and last delay with the beam exactly as it was, and
  subtract it from `.dwc` and `.csv`. On the instrument this runs on that is a
  ~2245-count pedestal against ~489 counts of signal, and it removed any real
  signal still present at the ends of the sweep. There is now one recorded dark,
  shared by the Spectrum panel and the scan; the delay-extreme frames are
  archived and never subtracted.
