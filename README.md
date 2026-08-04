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
  Kinesis, Zaber and Ocean Optics adapters all sit behind two abstract base
  classes, and vendor libraries are imported lazily.
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

| File | Role |
| --- | --- |
| [frog_gui_fast.py](frog_gui_fast.py) | PySide6 GUI — application entry point |
| [hardware.py](hardware.py) | Device abstraction: stage + spectrometer adapters, pulse simulator |
| [scan.py](scan.py) | Delay-scan engine: optics conversions, scan worker, file writers |
| [Lillypad.spec](Lillypad.spec) | PyInstaller recipe for a standalone Windows build |
| [requirements.txt](requirements.txt) | Pinned lockfile (Python 3.13) — install with `--no-deps` |
| [icons/](icons/) | Application and toolbar icons |

The dependency direction is strictly one-way — `frog_gui_fast` → `scan` →
`hardware` — and `hardware.py` has no Qt dependency at all. `scan.py` is split
into a Qt-free core (conversions, dataclasses, autocorrelation, writers) and a
thin `FrogScanWorker(QThread)` on top of it, so the analysis half is unit-testable
and reusable without a GUI.

## Getting started

Requires Python 3.13.

```bash
git clone https://github.com/Daniel-F-QM/lillypad.git
cd lillypad

python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on POSIX
pip install --no-deps -r requirements.txt

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

Both non-GUI modules are runnable and check themselves — no hardware, no Qt:

```bash
python hardware.py    # simulator self-test
python scan.py        # pure-core self-test (delay grid, conversions, autocorrelation)
```

## Real hardware

Open the **Hardware** dialog and pick a device. Vendor libraries load only when
you actually select their device, so a missing SDK costs you that one adapter
rather than the whole app.

| Device | Adapter | Backend |
| --- | --- | --- |
| Thorlabs Kinesis stage (LTS300C/M) | `KinesisStage` | `pylablib` |
| Zaber stage (serial / daisy-chain) | `ZaberStage` | `zaber-motion` |
| Ocean Optics / Ocean Insight spectrometer | `SeabreezeSpectrometer` | `seabreeze` |

Zaber auto-scans serial ports when no port is given. Seabreeze picks the first
device it finds unless you pass a serial number.

**Adding a stage model.** `KinesisStage` looks its scale up in `STAGE_CONFIGS`
and *raises* on an unknown model rather than guessing — a wrong steps-per-mm
scale would silently distort the delay axis instead of failing. Add an entry to
[hardware.py](hardware.py) to support a new one.

**Adding a device class.** Implement `StageBase` or `SpectrometerBase` in
[hardware.py](hardware.py). Moves must block until settled, so the scan loop can
read back a trustworthy position on the next line.

## Unit conventions

These are load-bearing — mixing them up produces a plausible-looking trace with a
wrong time axis.

- **Delay: femtoseconds.** The master unit for the FROG axis.
- **Positions: micrometres** everywhere user-facing and throughout `scan.py`.
- **Stage adapters: millimetres.** `pylablib` reports the LTS300 in mm, so that
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
