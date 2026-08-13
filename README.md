<p align="center">
  <img src="icons/Lilypad.png" width="120" alt="Lillypad">
</p>

<h1 align="center">Lillypad</h1>

<p align="center">
  A FROG acquisition app for ultrashort-pulse characterisation —
  motorised delay stage control + spectrometer live feed in a Qt GUI.
</p>

---
## Features

- **Live spectrum feed** on a background thread, paced to the spectrometer's
  integration time, with blitted redraws so the UI stays responsive during long
  integrations.
- **Live FROG build-up** — the trace and its autocorrelation are drawn column by
  column as the scan runs, each costing one small blit rather than a full redraw.
- **Hardware abstraction** — the GUI never imports a vendor SDK. Thorlabs
  Kinesis, Zaber, Ocean Optics and Avantes adapters all sit behind two abstract
  base classes, and vendor libraries are imported lazily.
- **Simulator with nine pulse shapes** — transform-limited sech²/Gaussian, GDD
  chirp, third-order dispersion, self-phase modulation, split-step fibre
  output, double pulse, two-colour pair, and a deliberately over-exposed pulse
  for exercising the saturation alarm.
- **Saturation detection** — per-column clipped-pixel counts, a status lamp, and
  an optional abort. Whether a scan *could* be checked at all is recorded
  separately from whether it *was* clean.
- **Bracketed background** — dark frames captured before and after the scan,
  stored alongside the trace rather than silently subtracted.
- **Three export formats** — `.dwc` FROG trace, `.npz` archive (raw counts,
  backgrounds, saturation record, full metadata), and a spreadsheet `.csv`.
- **Light and dark themes**, log/linear spectrum, manual or auto axis limits,
  and an interactive colour scale for the trace.

## Layout

| File (Core)| Role |
| --- | --- |
| [frog_gui_fast.py](frog_gui_fast.py) | PySide6 GUI — application entry point |
| [hardware.py](hardware.py) | Device abstraction: stage + spectrometer adapters, pulse simulator |
| [scan.py](scan.py) | Delay-scan engine: optics conversions, scan worker, file writers |
| [icons/](icons/) | Application and toolbar icons |

| **File (Helpers)**| **Role** |
| --- | --- |
| [Lillypad.spec](Lillypad.spec) | PyInstaller recipe for a standalone Windows build |
| [requirements.txt](requirements.txt) | Pinned lockfile (Python 3.13) — install with `--no-deps` |
| [avantes.py](avantes.py) | Standalone ctypes wrapper around the Avantes `AvaSpecX64.dll` |
| [zaber_diagnostics.py](zaber_diagnostics.py)| Support program to help diagnose issues with backlash compensation in zaber stages| 



The dependency direction is strictly one-way — `frog_gui_fast` → `scan` →
`hardware` — and `hardware.py` has no Qt dependency at all. `scan.py` is split
into a Qt-free core (conversions, dataclasses, autocorrelation, writers) and a
thin `FrogScanWorker(QThread)` on top of it, so the analysis half is unit-testable
and reusable without a GUI.

## Getting started

Requires Python 3.13.  

Requires Visual C++ - e.g. x64 "Visual C++ Redistributable" (vc_redist.x64.exe)  

Requires Avantes DLL and drivers to use their spectrometers - distributed by them separately

(likely) Requires Kinesis to be installed for Thorlabs stages to get the drivers  


!!! Important: if Python is installed through Anaconda it will give you an error "DLL load failed..." because it carries its own DLLs / search behaviour. If python is not installed otherwise, install the latests version and make sure you deactivate conda in your terminal if it starts up with it.

```bash
git clone https://github.com/Daniel-F-QM/lillypad.git

###
Without anaconda installed
###

python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on POSIX
pip install --no-deps -r requirements.txt
seabreeze_os_setup              # Ocean Optics only; Avantes has its own driver

python frog_gui_fast.py

###
With anaconda installed
###

conda deactivate                # in case conda is running in your terminal

py -3.14  -m venv .venv         # e.g. for python 3.14, if you don't have non-conda python installed you need to install it
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on POSIX
pip install --no-deps -r requirements.txt
seabreeze_os_setup              # Ocean Optics only; Avantes has its own driver

python frog_gui_fast.py

```

> **`--no-deps` is required.** `requirements.txt` is a complete lockfile, and
> the flag is what keeps **PyQt5** out. `pylablib` hard-declares `pyqt5` even
> though Lillypad is PySide6-only and only uses `pylablib.devices` — a plain
> `pip install -r` puts two Qt bindings in the environment, which has broken
> PyInstaller builds here before. `pip check` will warn that pylablib is
> missing pyqt5; that warning is expected and safe to ignore.

The app starts on simulated hardware, so this works with nothing plugged in.
Press **START FEED** for a live spectrum, then **Measure FROG** to run a scan.
Pick a pulse shape under **Hardware** to change what the simulator produces.

### Self-tests

All three non-GUI modules are runnable and check themselves — no hardware, no Qt:

```bash
python hardware.py    # simulator self-test
python scan.py        # pure-core self-test (delay grid, conversions, autocorrelation)
python avantes.py     # ctypes struct layout + error table; DLL and device optional
```

## Real hardware

Open the **Hardware** dialog and pick a device. Vendor libraries load only when
you actually select their device, so a missing SDK costs you that one adapter
rather than the whole app.

| Device | Adapter | Backend |
| --- | --- | --- |
| Thorlabs Kinesis stage (LTS150C/M, LTS300C/M + autodetected stages) | `KinesisStage` | `pylablib` |
| Zaber stage (serial / daisy-chain) | `ZaberStage` | `zaber-motion` |
| piezosystems Jena piezo stage (serial, 320 um closed loop) | `PiezoJenaStage` | `pyserial` |
| Ocean Optics / Ocean Insight spectrometer | `SeabreezeSpectrometer` | `seabreeze` |
| Avantes AvaSpec spectrometer (USB) | `AvantesSpectrometer` | ctypes → `AvaSpecX64.dll` |

Kinesis, Seabreeze and Avantes all enumerate first and pop a picker when more
than one device is attached. Zaber and Piezo Jena auto-scan serial ports when
no port is given.

**Spectrometers are named `vendor:serial`.** Two vendors can be on the bench at
once, so a bare serial no longer identifies a device — `hardware.list_spectrometers()`
returns tagged ids and `hardware.open_spectrometer()` is the single place a tag
is mapped to an adapter class. Both multi-spectrometer slots accept either
vendor, so an Avantes and an Ocean device can be stitched into one span. A
vendor whose SDK is missing simply contributes nothing to the enumeration
rather than taking the other vendor's devices down with it.

**Spectrometer backends.** python-seabreeze has two backends and the box beside
*Real (seabreeze)* in the Hardware dialog switches between them (applied on the
next connect). It sits on the seabreeze row for the same reason the port fields
sit on the Zaber and Piezo Jena rows — it applies to that adapter only, and
nothing else. The default is **pyseabreeze**, the pure-Python backend — it is
the only one that supports the newer Ocean Insight models (SR/ST/HDX series).
**cseabreeze**, the vendor C library, remains selectable for older devices
that misbehave through pyseabreeze. pyseabreeze talks USB through `pyusb`,
which needs a libusb driver: the `libusb-package` wheel in
`requirements.txt` provides the DLL on Windows (Lillypad puts it on `PATH`
automatically), and the device itself must be bound to a WinUSB/libusb
driver — `seabreeze_os_setup` (above) installs Ocean's drivers; if the device
still does not enumerate, bind its USB interface to **WinUSB** with
[Zadig](https://zadig.akeo.ie/). None of this applies to Avantes, which has its
own driver and its own DLL.

**Avantes spectrometers.** Two separate Avantes downloads matter, and installing
the wrong one is the usual first stumble:

| You need | What it gives you |
| --- | --- |
| **AvaSpec-DLL package** (`AvaspecX64Dll_*.Setup_64bit.exe`) | `AvaSpecX64.dll` (64-bit), `avaspec.h`, the library manual, examples |
| AvaSoft | the GUI app, the USB driver, and a **32-bit only** `avaspec.dll` |

Lillypad runs on 64-bit Python, and a 32-bit DLL cannot be loaded into it —
AvaSoft alone leaves you with a DLL Python refuses with `WinError 193`.
[avantes.py](avantes.py) detects exactly that case and names it, rather than
passing the bare OSError on. The USB driver ships with either package, so
installing AvaSoft first is harmless.

The DLL installer drops everything into a **versioned folder at the root of the
system drive** — `C:\AvaSpecX64-DLL_9.14.0.0\` — not into Program Files.
[avantes.py](avantes.py) searches, in order: `LILLYPAD_AVASPEC_DLL` (a full
path), next to the program (or the `.exe`), `C:\AvaSpec*DLL*\`,
`C:\Program Files\Avantes\…`, then `PATH`. With several versions installed the
newest wins.

Press **Real (Avantes)** in the Hardware dialog to connect. Everything
Avantes-specific then lives behind an **Avantes** button that appears in the
toolbar only while such a device is connected, so the Hardware panel stays the
same size it always was. That dialog covers on-board averaging, the ADC
resolution, dark and prescan correction, smoothing, triggering and sync, the
board temperature, and a device-info block. Three couplings are worth knowing:

- **ADC resolution moves full scale.** 14-bit is 16383 counts, 16-bit is 65535,
  and the counts scale by 4 between them. Switching re-arms the saturation
  alarm against the new ceiling.
- **On-board averaging multiplies the frame time.** The Integration Time box
  shows what one frame *costs* (exposure × averages), because that is what the
  live feed paces itself to. The scan already averages in software
  (*Acquisition Settings → averages*), so leave the on-board count at 1 unless
  you want both.
- **An armed hardware trigger stops the live feed.** Under an external trigger
  a frame only arrives when the experiment fires, so a free-running feed would
  simply block; Lillypad stops it for you and says so. `acquire()` still
  enforces a timeout rather than waiting forever.

Two things worth not being surprised by. The device reports its **own** pixel
count and wavelength axis, and they need not match the datasheet: a
ULS4096CL-EVO here reports **4094** pixels spanning 183.6–1338.6 nm, against a
nominal 4096 and a 200–1100 nm *usable* range. Nothing is hardcoded, so this is
fine — but the axis extends past where the grating is specified. And the DLL
hands out **the same handle** if you open one spectrometer twice, so closing
either copy disconnects both; `avantes.py` refuses the second open instead.

`python avantes.py` is a self-test. Without the DLL it still checks the ctypes
struct layout and the error table and prints what to install; with the DLL and a
spectrometer attached it enumerates, connects, acquires and reports the device.

**Kinesis stages.** `KinesisStage` identifies the stage from the model number
its controller reports before driving it, and calibrates it either from a
`STAGE_CONFIGS` entry in [hardware.py](hardware.py) (needed for the LTS150C/M
and LTS300C/M, which pylablib cannot calibrate itself) or from pylablib's own stage
autodetection (the Z6xx/Z7xx/Z8xx and MTS families, K10CR1, …). The resulting
scale units are then verified, so a rotational or uncalibrated stage is
*refused* rather than driven with somebody else's steps-per-mm — a wrong scale
would silently distort the delay axis instead of failing. To support a stage
that matches neither, add a `STAGE_CONFIGS` entry.

**Backlash and approach direction.** A lead-screw stage lands in a different
place depending on which way it arrived. A scan sweeps monotonically, so every
scan point is approached from one side — but you mark zero-delay after jogging,
which arrives from whichever side you happened to be on. Left uncorrected, the
two sit in frames that differ by the mechanical backlash and the whole delay
axis is offset: at double pass, 8 um of slack is **53 fs**.

The **Backlash** box in the Stage panel is the fix. With it set, every move —
jog, *Move*, *Move to 0 fs*, *Set Current Position as 0 fs* and each scan point
— undershoots by that margin and comes back up when it would otherwise arrive
from above, so everything shares one approach direction. A monotonic sweep only
reverses once (on the way into the range), so it costs one extra move per scan,
not one per point.

Defaults are per adapter, and you can override them:

| Adapter | Default | Why |
| --- | --- | --- |
| `ZaberStage` | **50 um** | Zaber does no backlash correction of its own |
| `KinesisStage` | 0 | Thorlabs controllers already do it in firmware (`get_gen_move_parameters`) |
| `PiezoJenaStage` | 0 | Closed-loop flexure — no screw, and every move verifies itself |

This is why the same rig can show a zero shift on a Zaber and none on a
Thorlabs stage.

**Long moves are split up.** A move blocks until the stage has settled and
nothing in the app can interrupt one, so a mistyped target or a fs/um mix-up
would otherwise become a single full-speed traverse across the travel. Any move
longer than `StageBase.max_step_mm` (**5 mm**, the same for every adapter; set it
to 0 to switch this off) is instead carried out as a run of sub-moves of at most
that far, each settling before the next is issued, and the status bar reports the
split. The stage still lands on exactly the position you asked for — the last
sub-move is the target itself — and the backlash approach above is unaffected.
Scan points are microns apart, so a scan never trips it.

**Zaber stages.** Two Zaber-specific things are worth knowing.

`get_position()` reads Zaber's `pos` setting, which on a stepper is the
*trajectory counter*, not a measurement: after a completed move it returns what
was commanded whether or not the carriage got there. A readback alone therefore
cannot reveal a stall, a knob nudge or a lost reference. `ZaberStage` instead
reads the device's warning flags after every scan point (`FS` stalled, `WM`
displaced while stationary, `NC` moved by manual control, `WH`/`WR` unhomed, …)
and, on devices with an encoder, cross-checks `pos` against `encoder.pos`.
Anything it finds stops the scan — *Abort scan on stage fault* in **Acquisition
Settings**, on by default. Unchecked, the scan finishes and the affected
columns are marked in the `.npz` (`stage_faults`, one entry per column).

An **unhomed** axis is refused outright: its `pos` has no physical meaning, and
homing afterwards moves the coordinate frame under any zero marked before it.
Home first, then set zero. (Homing an axis that *was* unhomed also clears the
stored zero-delay, for the same reason.)

To measure the backlash on your own stage:

```bash
python zaber_diagnostics.py                     # report only, moves nothing
python zaber_diagnostics.py --measure           # MOVES: measures the backlash
python zaber_diagnostics.py --check-compensation    # MOVES: verifies the fix
```

The report covers firmware, peripheral, microstep size (in um *and* fs),
`limit.min`/`limit.max`, homed state and active warning flags. `--measure`
approaches one target from both directions and reads the difference off the
encoder, printing it in um and fs — that number is the zero shift to expect
without correction, and it is what the **Backlash** box needs to exceed.

**Spectrometer calibration.** The toolbar's **Calibration** menu assigns an
intensity calibration to each connected spectrometer: a two-column text file
(wavelength in nm, multiplicative factor; `#` comments allowed) from the
`calibration_files/` folder next to the program (next to the `.exe` for the
frozen build — the folder is user-editable, drop new files in or use *Add new
calibration…* in the menu). The factors are interpolated onto the device's
pixel grid and multiply every displayed and recorded spectrum. Saturation is
always judged on **raw** ADC counts, before calibration.

**Multi-spectrometer mode.** Either slot takes a device from either vendor, so
an Avantes and an Ocean spectrometer stitch together like two of a kind.
*Enable multi-spectrometer mode* in the
toolbar's **Multi-Spec** menu opens two slots, each with its own spectrometer
and calibration submenu; once both slots are filled the pair connects
automatically as one stitched device (`StitchedSpectrometer`): spectra are
interpolated onto a common grid, each device's own calibration file is
applied first, and the two are averaged across the overlap (the ranges must
overlap). *Auto-stitch* least-squares-matches the bluer spectrometer to the
redder one over the overlap (do this with light spanning the overlap);
*Manual stitch…* enters the factor by hand. Entering the mode adds a second
saturation lamp to the status bar: each device's RAW frames are judged
against that device's own full scale, both live and during scans, so either
detector clipping trips its own alarm. *Disable multi-spectrometer mode*
keeps the slot-1 spectrometer connected as a normal single device.

Each slot also offers a **simulated** member — *Simulated — blue half* and
*Simulated — red half*, listed above the real devices and available even with
no spectrometer attached — so the whole mode can be exercised offline. The two
cover overlapping halves of the simulated signal band (roughly 65% each, so
there is a genuine blue-only / shared / red-only geometry for *Auto-stitch* to
work on) rather than two identical full-band copies, which would make the
entire spectrum "overlap" and test nothing. Entering the mode while the
simulator is connected pre-fills slot 1, so picking the red half in slot 2 is
all it takes. Leaving the mode restores the normal full-band simulator.

While a pair is connected the Spectrum panel offers **one integration time per
spectrometer** (*S1* / *S2*, in slot order — the same numbering as the lamps),
so an arm that sees little light can be exposed longer than a bright one.
Frames stay in raw counts, so the exposure ratio ends up inside the stitch
factor: after changing either time, re-run *Auto-stitch*, or the seam
reappears. The Multi-Spec menu marks the factor *stale* until you do, and the
dark is exposure-specific too, so re-record it.

The button beside the auto-fit control on the plot switches the Spectrum panel
between the combined stitched curve and **one curve per spectrometer** (S1 sky
blue, S2 orange — a colourblind-safe pair), each on its own pixel grid with its
own dark and calibration applied. That is the view for judging the stitch: with
a good factor the two curves lie on top of each other across the overlap. The
icon shows the view you get by clicking. The combined curve returns
automatically during a scan, since the scan records stitched columns.

**Adding a device class.** Implement `StageBase` or `SpectrometerBase` in
[hardware.py](hardware.py). Moves must block until settled, so the scan loop can
read back a trustworthy position on the next line.

## Unit conventions

These are load-bearing — mixing them up produces a plausible-looking trace with a
wrong time axis.

- **Delay: femtoseconds.** The master unit for the FROG axis.
- **Positions: micrometres** everywhere user-facing and throughout `scan.py`.
- **Stage adapters: millimetres.** `pylablib` reports the LTS150/LTS300 in mm, so that
  boundary is isolated in `_um_to_stage` / `_stage_to_um` — the only two places
  mm appears. Make them identity if an adapter is ever changed to report µm.
- **Wavelength: nanometres**; spectra are **raw counts**.

Delay and position assume a double-pass geometry, otherwise adjust `pass_factor`.

## Building a standalone executable

```bash
pyinstaller Lillypad.spec
```

Produces `dist/Lillypad/Lillypad.exe` with the icons bundled. `resource_path()`
in the GUI resolves assets through `sys._MEIPASS`, so the same code path works
frozen and from source.

The spec excludes `PyQt5`, `PyQt6` and `PySide2`, so a stray Qt binding in the
environment cannot end up in the bundle even if something reinstalls one.
