<p align="center">
  <img src="icons/Lilypad.png" width="120" alt="Lillypad">
</p>

<h1 align="center">Lillypad</h1>

<p align="center">
  Measure ultrashort laser pulses with FROG — delay stage and spectrometer,
  driven from one window.
</p>

---

Lillypad runs a **FROG** measurement: it walks a motorised delay stage through a
range of positions and records a spectrum at each one. Stack those spectra side
by side and you get the FROG trace — the picture that tells you how long your
pulse is and how it is chirped.

It drives Thorlabs, Zaber and piezosystem Jena stages, and Ocean Optics and
Avantes spectrometers. It also has a full simulator, so you can click through
everything with nothing plugged in.

## What it does

- **Live spectrum and live trace** — the trace builds up column by column while
  the scan runs.
- **Eleven simulated pulse shapes**, from a clean transform-limited pulse to a
  chirped one, a double pulse, fibre output, a deliberately over-exposed one, a
  deliberately misaligned one, and *Bird* — a measured 1.0–3.2 µm continuum
  that comes with its own pair of simulated Ocean spectrometers.
- **Saturation warnings** — clipped pixels counted per column, shown on a lamp,
  with an optional automatic stop.
- **Three alignment views** for finding a misaligned or chirped beam without
  running a full measurement.
- **Two spectrometers at once**, stitched into one wide spectrum.
- **Three export formats**, including an archive that keeps everything raw.
- **Light and dark themes**, log or linear spectrum, manual or automatic axis
  limits, and a choice of colour maps.

---

# Getting started

## What you need

- **Python 3.13** from [python.org](https://www.python.org/downloads/)
- **Visual C++ Redistributable (x64)** — `vc_redist.x64.exe`, from Microsoft
- For **Avantes**: their DLL package and driver ([details below](#avantes-spectrometers))
- For **Thorlabs**: the Kinesis software, which carries the drivers

> [!WARNING]
> **If you have Anaconda installed**, expect `DLL load failed…` — Anaconda
> brings its own DLLs and its own way of finding them. Run `conda deactivate`
> first and make sure you are using a python.org install. If you have both,
> name the right one when creating the environment: `py -3.13 -m venv .venv`.

## Install

```bash
git clone https://github.com/Daniel-F-QM/lillypad.git
cd lillypad

python -m venv .venv                    # an isolated environment
.venv\Scripts\activate                  # Windows; source .venv/bin/activate elsewhere

pip install --no-deps -r requirements.txt
seabreeze_os_setup                      # Ocean Optics only — Avantes has its own driver

python frog_gui_fast.py
```

> [!IMPORTANT]
> **Keep the `--no-deps` flag.** `requirements.txt` already pins every package,
> so nothing needs resolving — and the flag is what keeps **PyQt5** out.
> `pylablib` insists on it even though Lillypad does not use it, and two Qt
> toolkits in one environment has broken builds here before. Afterwards
> `pip check` will warn that pylablib is missing pyqt5; that is expected.

## Try it with no hardware

Lillypad starts with **nothing connected**, and says so. To take it for a spin,
open **Simulation** in the toolbar and press **Spectrometer** and **Stage** (or
**Stitched pair** for two). Then press **Measure FROG**. The same window is
where you choose the pulse shape.

For the busiest version of that, pick the **Bird** beam and press **Stitched
pair**: a 1.0–3.2 µm continuum measured on a simulated Ocean SR + NIRQuest
bench, which is the one combination that exercises nearly every diagnostic in
the app at once (see [The *Bird* beam](#two-spectrometers-at-once)).

Simulated devices appear *only* there — never in the Spectrometer or Stage
windows — so nothing that looks like a hardware control can hand you fake data.

---

# Using it

## The window

**Toolbar:** Acquisition Settings · Spectrometer · Stage · Simulation ·
Graphics Settings · Alignment · Export · Calibration · *Avantes* (only when one
is connected) · light/dark toggle · camera (save the plots as a picture) ·
layout toggle.

**Panels:** the live **Spectrum** on one side; the **FROG Trace** and its
**autocorrelation** on the other. Small buttons in the top corners of each plot
are the view controls — log scale, axis lock, and the alignment views.

## A measurement, start to finish

1. **Connect.** Press **Connect Spectrometer** and **Connect Stage** on the side
   panels, or use the toolbar buttons — same windows either way. Each scans for
   what is attached and lists it.
2. **Home the stage** if it asks you to.
3. **Find zero delay.** Jog until the signal peaks, then press
   **Set Position as 0 fs**.
4. **Set the range** — start, stop and step in femtoseconds — and press
   **Measure FROG**.
5. **Export** from the toolbar, or **Save** on the side panel — same file
   either way.

When a scan finishes Lillypad drives the stage back to zero delay, so the next
thing you do starts on signal rather than on a dark frame. It skips that if the
stage reported a fault, since "zero" would no longer mean anything.

## Saving your data

| Format | What it contains |
| --- | --- |
| `.dwc` | The FROG trace, Femtosoft-style. **Background subtracted.** |
| `.csv` | The same trace as a spreadsheet table. **Background subtracted.** |
| `.npz` | Everything: **raw counts**, both background frames kept separately, the per-column saturation and stage-fault record, and full metadata. |

If you want untouched numbers, use `.npz`. The two text formats have fixed
layouts that cannot carry the extras, so they get the subtracted trace.

**Export** opens a window holding the format, the file name and the Save
button. The name is a starting point — the Save dialog can still be typed over
— and with **Date prefix?** ticked (it is by default) today's date goes in front
of it as `YYYY-MM-DD_`, so a folder of exports sorts chronologically by name.
It is the name every save in the program starts from, the toolbar's camera
included. The line under the field shows exactly what the Save dialog will open
on:

> Saves as   2026-09-15_frog_scan.dwc

The format and the name are both remembered between sessions; the date is not
— it is made fresh from today's date on every save. **Save** on the side panel
skips the window and writes with whatever it is set to.

## Saving the picture

The **camera** in the toolbar writes the plots themselves to a PNG at 300 dpi —
the FROG trace with the autocorrelation under it and the spectrum full height
beside them, in whichever theme and colour map you are looking at. It is always
that arrangement, whichever layout the window is in, and it leaves the alignment
panel out; it is a figure for a slide or a page, not a record of the
measurement. Export is what writes the numbers.

The page takes the window's own proportions, so a wider window gives a wider
picture — and a very large one prints at a little under 300 dpi, to keep the
file sane. The status bar reports the size it wrote.

It opens on the same file name as **Export**, date prefix and all, with `.png`
on the end — so a scan and the picture of it sit together in a folder listing
under one name.

Everything on the panels goes into it as it stands — the axis limits you have
dialled in, the log scale, the AC width, a frozen reference curve, the symmetry
fold. There is no need to wait for a scan: the live spectrum prints just as
well.

## Two kinds of "background"

Easy to mix up:

- **Record Dark** (Spectrum panel) takes a dark frame *now* and, with
  **Subtract Dark** ticked, removes it from what you see and from the stitching
  fit. A display and fitting aid.
- **Background** (Acquisition Settings) records a frame *before and after* each
  scan. These are stored with the measurement, not silently folded into it.

A dark is only valid for the exposure it was taken at, so changing the
integration time throws it away and says why. Record it again.

**What the trace panel shows.** Every panel is drawn with the pedestal taken
off. While a scan runs that is the recorded dark, sampled once when the scan
starts so the whole trace is corrected the same way — unticking **Subtract
Dark** mid-scan therefore applies to the *next* one. When the scan finishes the
panel switches to the scan's own bracketed background, which was measured with
the beam blocked and already contains the dark, so the two are never subtracted
together. With **Background** unticked it stays on the dark; with neither, the
trace is drawn as measured.

This matters most on a **stitched pair**, where it is the difference between a
readable trace and a mystery. The two detectors have very different dark
levels, the blue one's is multiplied by the stitch factor, and the calibration
multiplies both — hardest right where a detector's response dies. An
uncorrected merged frame therefore carries a *step* across the crossfade band,
at the same wavelengths in every column, which draws as a hard horizontal line
across the whole FROG trace.

All of this is display only. `result.trace`, the `.npz` and the exports are
untouched: the `.dwc` and `.csv` have always been written background-subtracted
(the panel now agrees with them), and the `.npz` still archives raw counts with
both background frames kept separately. One thing has *not* followed: the
`fwhm_ac_fs` field inside the `.npz` metadata is still computed from the raw
trace, so it can differ slightly from the **AC FWHM** shown on the panel.

---

# Alignment tools

Four diagnostic views. **None of them change your data** — the saved result and
every export are exactly what they would have been without them.

### + — before and after

*Top-right of the Spectrum panel.* Freezes the spectrum currently on the panel
and keeps it as a **grey curve underneath the live one**, so you can make an
adjustment and see exactly what it changed rather than trying to remember the
shape.

An **eye** button appears beside the **+** to hide and show the reference.
Pressing **+** again replaces it with the current spectrum; right-clicking **+**
drops it altogether.

With two spectrometers the shape of the reference follows the view it was taken
in: freeze the combined curve and you get one grey curve, freeze the split view
and you get one per spectrometer. It then **stays on the panel whichever view
you switch to** — grey counts against wavelength read the same either way.

### Δ — is the pulse symmetric in time?

*Top-right of the Spectrum panel.* A misaligned or chirped beam looks different
at `+x` fs than at `−x` fs, and normally you would only find that out by running
a whole scan. Pressing **Δ**:

- pauses the live feed,
- steps the stage to `−2x`, `−x`, `+x`, `+2x` — in that order, so every point is
  approached from the same side and mechanical slack cancels out,
- measures a spectrum at each using your current averaging settings,
- returns the stage to where it started,
- and plots `S(+x) − S(−x)` and `S(+2x) − S(−2x)` on **their own panel below the
  spectrum**, against a dashed zero line.

Two flat curves on zero means a symmetric pulse; the status bar gives the worst
imbalance as a percentage. Press **Δ** again to clear. Set `x` in the toolbar
under **Alignment**.

The four positions are measured from **wherever the stage is standing**, not
from your marked zero — park where you want to test (usually *Move to 0 fs*) and
press. A sweep that would run off the end of the travel is **refused, not
clamped**, since a clamped target would destroy the symmetry you are measuring.
It is also refused during a scan, or before the stage has been homed.

### Symmetry — is the *trace* symmetric?

*Top-right of the FROG Trace panel.* Folds a finished trace in half: measured
data on one side, the difference `|T(+d) − T(−d)|` on the other, sharing one
colour scale. Nearly black on the right means nearly symmetric, and a score in
the corner puts a number on it.

It folds about the autocorrelation peak — the delay the trace *should* be
symmetric about — rather than the middle column, so an off-centre scan range
does not report its own offset as asymmetry. Drag the fold line to move it.

### RAW — what the detectors actually saw

*Top-right of the FROG Trace panel, two spectrometers only.* The normal trace is
dark-subtracted, calibrated and blended at the stitch factor. All three tidy away
the low-level background — exactly where alignment problems hide: satellite
pulses, scatter, an arm that is not really seeing the beam.

**RAW** shows the unprocessed version: no dark subtraction, no calibration, each
half scaled to its own maximum, and the two hard-**cut** at the middle of the
overlap instead of blended. The visible seam and raised background are
deliberate.

Hidden with a single spectrometer, and greyed out until a scan has recorded
something. The cut is fixed when the scan starts — moving the overlap band
afterwards affects the *next* scan, not the one on screen.

---

# Connecting real hardware

Press **Connect Spectrometer** / **Connect Stage** on the side panels, or the
toolbar buttons. **Disconnect** in either window releases the device. Vendor
software loads only when you pick that vendor's device, so a missing SDK costs
you one adapter rather than the whole app.

| Device | Needs |
| --- | --- |
| Thorlabs Kinesis stage (LTS150C/M, LTS300C/M, autodetected models) | Kinesis installed |
| Zaber stage (serial or daisy-chain) | — |
| piezosystem Jena piezo stage (320 µm, closed loop) | — |
| Ocean Optics / Ocean Insight spectrometer | `seabreeze_os_setup`, sometimes Zadig |
| Avantes AvaSpec spectrometer (USB) | AvaSpec-DLL package |

Thorlabs, Ocean and Avantes devices are listed for you to pick from; Zaber and
Piezo Jena scan the serial ports if you do not name one. **Spectrometers are
identified as `vendor:serial`**, since two vendors can share a bench and a bare
serial number no longer tells them apart.

## Integration time

The box takes its limits **from the device you connected**, so sub-millisecond
exposures appear only where the hardware really has them. The maximum is capped
at **10 s** whatever the device claims, because acquiring blocks for the whole
exposure and anything longer just looks like a hang.

Changing the value writes it to the device straight away (after a short pause
for you to stop typing), so a refusal names itself there and then instead of
freezing the live plot at the next frame. Connect a coarser device while the box
holds a fine value and it is raised to that device's minimum rather than
failing.

> **Below a millisecond, frames do not get proportionally faster.** Every frame
> also pays sensor readout, USB transfer and the DLL round trip. Measured on an
> Avantes ULS4096CL-EVO: 10 ms exposure → 12.4 ms per frame, 1 ms → 2.9 ms,
> 0.05 ms → 1.8 ms, 9 µs → 2.5 ms. Below ~1 ms you are buying dynamic range
> against a bright source, not speed.

The live feed asks for at most ~33 frames a second and the plot redraws about 16
times a second. A scan is not capped and takes frames as fast as the device
yields them.

<details>
<summary><b>Ocean Optics / Ocean Insight details</b></summary>

python-seabreeze has two backends, and the **Backend** box in the Spectrometer
window chooses between them (applied on the next connect). It sits on the
seabreeze row because it affects that adapter only.

- **pyseabreeze** (default) is pure Python and the only one supporting the newer
  SR/ST/HDX models.
- **cseabreeze** is Ocean's C library, still useful for older devices that
  misbehave through pyseabreeze.

pyseabreeze reaches USB through `pyusb`, which needs a libusb driver. The
`libusb-package` wheel supplies the DLL and Lillypad puts it on `PATH` for you.
The device must also be bound to a WinUSB/libusb driver: `seabreeze_os_setup`
installs Ocean's, and if it still does not appear, bind its USB interface to
**WinUSB** with [Zadig](https://zadig.akeo.ie/). None of this applies to
Avantes, which has its own driver and DLL.
</details>

<details id="avantes-spectrometers">
<summary><b>Avantes details — including which download you need</b></summary>

**Two different Avantes downloads exist, and picking the wrong one is the usual
first stumble:**

| Download | What it gives you |
| --- | --- |
| **AvaSpec-DLL package** (`AvaspecX64Dll_*.Setup_64bit.exe`) — **this one** | the 64-bit `AvaSpecX64.dll`, headers, manual, examples |
| AvaSoft | the GUI app, the USB driver, and a **32-bit only** DLL |

Lillypad runs on 64-bit Python and cannot load a 32-bit DLL. Install only
AvaSoft and Python refuses with `WinError 193` — Lillypad detects exactly that
and says so, rather than passing on a cryptic error. The USB driver ships with
both packages, so installing AvaSoft first does no harm.

**Where the DLL goes.** The installer drops it in a versioned folder at the root
of your system drive — `C:\AvaSpecX64-DLL_9.14.0.0\` — not in Program Files.
Lillypad looks in this order: `LILLYPAD_AVASPEC_DLL` if you set it to a path;
next to the program (or `Lillypad.exe`); `C:\AvaSpec*DLL*\` and `C:\Avantes\`;
`C:\Program Files\Avantes\…` and the `(x86)` equivalents; then `PATH`. With
several versions installed, the highest version number wins. AvaSoft's 32-bit
DLL is checked last, and only so Lillypad can tell you *why* it is unusable
rather than reporting "no DLL found" while one sits on the disk.

**The Avantes button** appears in the toolbar only while an Avantes is
connected. It covers on-board averaging, ADC resolution, dark and prescan
correction, smoothing, triggering and sync, board temperature, and device info.
Three things are worth knowing:

- **ADC resolution changes full scale.** 14-bit tops out at 16383 counts, 16-bit
  at 65535 — a factor of 4. Switching re-arms the saturation alarm against the
  new ceiling.
- **On-board averaging multiplies the frame time.** The integration-time box
  sets the *exposure*; the Avantes window shows what one frame actually costs
  (exposure × averages). Lillypad already averages in software under
  *Acquisition Settings*, so leave the on-board count at 1 unless you want both.
- **An armed hardware trigger stops the live feed**, since a frame only arrives
  when the experiment fires and a free-running feed would just block. In this
  one case acquiring waits indefinitely by design — there is no sensible timeout
  when you are waiting on the experiment.

**Two things not to be surprised by.** The device reports its own pixel count
and wavelength axis, and they need not match the datasheet — a ULS4096CL-EVO
here reports 4094 pixels over 183.6–1338.6 nm, against a nominal 4096 and a
200–1100 nm *usable* range. Nothing is hardcoded, so this is fine, but the axis
extends past where the grating is specified. And the DLL hands out the *same
handle* if you open one spectrometer twice, so closing either copy would
disconnect both — Lillypad refuses the second open instead.

**The minimum exposure is the sensor's, not the library's**, and the two can
differ by orders of magnitude with nothing in the device config to say which
applies. Lillypad finds the real floor by asking the device at connect (~18
`AVS_PrepareMeasure` calls, bisected). The ULS4096CL-EVO here answers **9 µs**
against the library's 2 µs. This matters: an exposure the sensor refuses is
accepted by the library and then fails at the *next* acquisition, which on a
running feed means the plot quietly freezes on its last good frame.
</details>

<details>
<summary><b>Thorlabs Kinesis details</b></summary>

Lillypad reads the model number from the controller before driving anything and
calibrates the stage either from a built-in `STAGE_CONFIGS` entry (needed for
the LTS150C/M and LTS300C/M, which pylablib cannot work out itself) or from
pylablib's own detection (Z6xx/Z7xx/Z8xx, MTS, K10CR1, …).

It then **verifies** the resulting units. A rotational or uncalibrated stage is
*refused* rather than driven with someone else's steps-per-millimetre — a wrong
scale would silently stretch your delay axis instead of failing visibly. For a
stage that matches neither route, add a `STAGE_CONFIGS` entry in
[hardware.py](hardware.py).
</details>

<details>
<summary><b>Zaber details</b></summary>

**The position readback is not a measurement.** Zaber's `pos` is the trajectory
counter: after a move it reports what was commanded, whether or not the carriage
got there. Reading it back cannot reveal a stall, a knob nudge or a lost
reference.

So Lillypad reads the controller's warning flags after every scan point (`FS`
stalled, `WM` displaced while stationary, `NC` moved by hand, `WH`/`WR` unhomed,
and others) and, where there is an encoder, cross-checks `pos` against it.
Anything it finds stops the scan — *Abort scan on stage fault* in **Acquisition
Settings**, on by default. Switch it off and the scan finishes with the affected
columns marked in the `.npz`.

**An unhomed axis connects, but will not scan.** Its position has no physical
meaning yet, and homing afterwards moves the coordinate frame underneath any
zero marked before it. Lillypad connects, says clearly that the axis is not
homed, and blocks scans and alignment sweeps until you press Home. Homing an
unreferenced axis also resets the stored zero — home first, then set zero.

**To measure the backlash on your own stage:**

```bash
python zaber_diagnostics.py                      # report only — moves nothing
python zaber_diagnostics.py --measure            # MOVES: measures the backlash
python zaber_diagnostics.py --check-compensation # MOVES: verifies the fix
```

The report covers firmware, peripheral, microstep size (in µm *and* fs), travel
limits, homed state and active warning flags. `--measure` approaches one target
from both directions and reads the difference off the encoder. That number is
the zero shift you would get without correction, and it is what the **Backlash**
box needs to exceed.
</details>

<details>
<summary><b>piezosystem Jena details</b></summary>

This controller sends things you did not ask for — a power-up banner like
`NV1CL V1.236>` — which arrive mid-conversation and mean the box has just
rebooted *out of* remote and closed-loop mode, where move commands do nothing.

So every reply is checked against the prefix it must start with (`rd,` for a
position) rather than just scanned for a number. The error reply `err,2` happens
to contain a comma and used to be read as a position of 0.002 mm — a wrong
position that looks perfectly good, which is far worse than an error.

Readbacks and moves each get **3** attempts, and every retry re-asserts remote
and closed-loop mode first; that is what recovers a rebooted controller instead
of giving up on it. A move counts as landed within the stage's own 0.1 µm
resolution.

If a move still has not landed, the adapter **records the reason rather than
raising an error**, which routes it into the normal stage-health path: the
column is flagged, tagged with where the stage really was, and the scan stops
only if *Abort scan on stage fault* is set. A single bad reply used to throw away
the entire scan, including every column already measured.
</details>

---

# Two spectrometers at once

No mode to switch on: the **Spectrometer** window has two slots from the start.
Slot 1 alone is an ordinary single spectrometer. Fill slot 2 and the pair
connects immediately as one stitched device — both spectra go on a common
wavelength grid, each with its own calibration applied, and are crossfaded where
they overlap. Set slot 2 back to *(none)* and slot 1 carries on alone. Either
slot takes either vendor, so an Avantes and an Ocean device stitch together
happily, as long as their ranges overlap.

**Matching the two.** *Auto-stitch* scales the bluer spectrometer onto the redder
one across the overlap — do it with light that spans the band. It is in the
Spectrometer window and also as a **⇌** button above the Spectrum panel, next to
the curve you are judging it by. *Manual…* lets you type the factor instead.
Auto-stitch subtracts the recorded dark from both devices first, so a
long-exposure arm's pedestal cannot get matched instead of the light.

**The overlap band.** The range the two devices geometrically share always
includes both detectors' dead edges, where one has stopped responding and the
other has not started. Those samples are pure noise and contribute nothing but
error to the fit. *Overlap band…* sets the narrower range actually used — for
the fit *and* for the crossfade, which uses a raised cosine so there is no
visible seam. It defaults to the middle 90% of the shared range and is shaded
**green** in the per-spectrometer view.

Auto-stitch also reports a **residual mismatch**, answering "is a single scale
factor enough for this pair?" — a few percent means the two calibrated curves
genuinely agree; a large one means a calibration is wrong or the band is too
wide. Factor, mismatch and band all appear in the *Stitching* block.

**Separate exposures.** Each spectrometer gets its own integration time (*S1* and
*S2*, numbered by slot, same as the lamps), so a dim arm can be exposed longer
than a bright one. Frames stay in raw counts, so that exposure ratio ends up
buried in the stitch factor — **after changing either time, re-run Auto-stitch**
or the seam comes back. The window marks the factor *stale* until you do.
Re-record the dark too.

**Seeing the two separately.** The button beside the auto-fit control switches
the Spectrum panel between the combined curve and **one curve per spectrometer**
(S1 sky blue, S2 orange — a colourblind-safe pair), each with its own dark and
calibration. That is the view for judging a stitch: with a good factor the two
curves lie on top of each other across the overlap. The icon shows the view you
get by clicking, and the view stays up during a scan.

A live pair also puts a second saturation lamp in the status bar — each device
is judged against its own full scale, so either one clipping trips its own
alarm.

**Simulated pair.** *Stitched pair* in the Simulation window fills both slots
with simulated devices covering overlapping two-thirds of the band, so there is
a genuine blue-only / shared / red-only geometry for Auto-stitch to work on.

**The *Bird* beam.** One simulated beam brings its own instruments instead. Its
SHG covers 500–1400 nm — more than any one detector — so *Stitched pair* gives
you a simulated **Ocean SR** (silicon, 350–1050 nm, 2048 pixels) and a simulated
**NIRQuest512** (InGaAs, 900–1700 nm, 512 pixels), each with that detector's own
response curve, dark level and read noise. They share 900–1050 nm, where the
silicon unit is falling off its red end and the InGaAs one has barely cut on —
the awkward overlap a real bench has, rather than two halves of one band.

It is the beam to pick for exercising the stitching tools, because every step
visibly moves the residual:

| | Auto-stitch residual |
| --- | --- |
| Raw counts, no calibration | ~40% — one number cannot reconcile two responses |
| With `SR_New` / `Niquest_New` loaded (done for you, if they are in `calibration_files`) | ~9% |
| After **Record Dark** | ~5% |
| After narrowing the *Overlap band* to 915–1000 nm | ~4% |

Switch to the per-spectrometer view to watch it happen, and to **RAW** after a
scan to see what the two detectors actually recorded underneath.

---

# Things worth understanding

## Units

Mixing these up gives a plausible-looking trace with a wrong time axis:

- **Delay: femtoseconds** — the master unit for the FROG axis.
- **Positions: micrometres**, everywhere you see them.
- **Wavelength: nanometres.** Spectra are raw counts.
- Stage adapters talk **millimetres** internally, because that is what the vendor
  libraries use. The conversion lives in one pair of functions in
  [scan.py](scan.py).

Delays assume a **double-pass** geometry — the beam goes to a retroreflector and
back, so moving the stage by *x* changes the path by *2x*. For another geometry,
change `pass_factor`.

## Backlash, and why your zero can be wrong

A lead-screw stage lands in a slightly different place depending on which
direction it arrived from. A scan sweeps one way, so every scan point is
approached the same way — but you mark zero after *jogging*, which arrives from
whichever side you happened to be on. The two then sit in frames that differ by
the mechanical slack. At double pass, 8 µm of slack is **53 fs**.

The **Backlash** box in the Stage panel fixes it: every move — jogging, *Move*,
*Move to 0 fs*, and each scan point — deliberately undershoots and comes back up
whenever it would otherwise arrive from above, so everything shares one approach
direction, including the jog before you marked zero. (*Set Position as 0 fs* does
not move the stage; it reads where the stage is. It is the jog before it that
gets corrected.) A scan sweep reverses only once on its way into the range, so
this costs one extra move per scan, not one per point.

| Stage | Default | Why |
| --- | --- | --- |
| Zaber | **50 µm** | Zaber does no backlash correction of its own |
| Thorlabs Kinesis | 0 | the controller already does it in firmware |
| piezosystem Jena | 0 | closed-loop flexure — no screw, and every move verifies itself |

This is why the same setup can show a zero shift on a Zaber and none on a
Thorlabs stage.

## Long moves are split up

A move blocks until the stage has settled and nothing can interrupt one, so a
mistyped target or a fs/µm mix-up would otherwise become a single full-speed
traverse across the whole travel. Any move longer than 5 mm is carried out as a
series of shorter ones, each settling before the next, and the status bar says
so. You still land exactly where you asked. Scan points are microns apart, so a
scan never triggers this.

## Spectrometer calibration

The toolbar's **Calibration** menu assigns an intensity calibration to each
connected spectrometer. A calibration is a plain two-column text file —
wavelength in nm, then a multiplying factor, `#` for comments — kept in
`calibration_files/` next to the program (next to `Lillypad.exe` in a built
version). Drop new files in there or use *Add new calibration…*.

The factors are interpolated onto the device's own pixel grid and multiply every
spectrum shown and recorded. **Saturation is always judged on raw counts, before
calibration** — clipping is a property of the detector, not of the physics
applied afterwards.

A file with the odd malformed line still loads; the bad lines are skipped and
counted in the status message. A file that cannot be read at all pops a warning
and the device keeps what it had, so a menu label can never claim a calibration
the spectrometer is not carrying.

## Your settings are remembered

Lillypad writes a `settings.json` next to the program and reads it back on the
next launch. It holds the theme, window size and position, colour map, line
width, the export format and file name (date prefix included), and your
acquisition settings — scan range and step, averaging, saturation threshold and
the two abort switches.

So the defaults described here are what you get on a *fresh* install, and
**deleting `settings.json` is a clean factory reset**. It is plain text and safe
to edit or delete while the app is closed. Device settings, including everything
in the Avantes window, are *not* stored there — those are read from the
instrument each time.

---

# For developers

## Layout

| File | Role |
| --- | --- |
| [frog_gui_fast.py](frog_gui_fast.py) | The PySide6 GUI — the entry point |
| [hardware.py](hardware.py) | Device adapters and the pulse simulator |
| [scan.py](scan.py) | Scan engine: optics conversions, scan worker, file writers |
| [avantes.py](avantes.py) | Standalone ctypes wrapper around `AvaSpecX64.dll` |
| [zaber_diagnostics.py](zaber_diagnostics.py) | Backlash diagnostics for Zaber stages |
| [Lillypad.spec](Lillypad.spec) | PyInstaller recipe for a Windows build |
| [requirements.txt](requirements.txt) | Pinned lockfile — install with `--no-deps` |

Imports run one way: `frog_gui_fast` → `scan` → `hardware`, with the GUI also
using `hardware` directly for the adapter classes, and `zaber_diagnostics.py` a
separate command-line consumer of both. `hardware.py` has **no Qt dependency at
all**, and no vendor SDK is ever imported by the GUI — every adapter sits behind
one of two abstract base classes, and vendor libraries load lazily.

`scan.py` holds a device-free, GUI-free core — conversions, the delay grid,
autocorrelation, the dataclasses and the file writers — with a thin
`FrogScanWorker(QThread)` on top. The core needs no hardware and opens no
window, but importing the module needs PySide6 installed, because the worker's
import sits at module level.

**To add a device:** implement `StageBase` or `SpectrometerBase` in
[hardware.py](hardware.py). Moves must block until the stage has settled, so the
scan loop can read back a position it can trust on the next line.

## Self-tests

All three non-GUI modules check themselves — no hardware, no window:

```bash
python hardware.py    # simulator, stage logic, stitching, calibration parsing
python scan.py        # delay grid, conversions, backlash, fault handling
python avantes.py     # ctypes struct layout and error table; DLL optional
```

`hardware.py` and `avantes.py` run with only numpy installed. `scan.py` also
needs PySide6, since it exercises the scan worker. `avantes.py` reports what to
install if the DLL is missing, and enumerates, connects and acquires if a
spectrometer is attached.

## Building a standalone executable

```bash
pyinstaller Lillypad.spec
```

Run this **from the repository root** — the spec picks up `calibration_files/`
by a relative path. It produces `dist/Lillypad/Lillypad.exe` with the icons and
calibration files bundled; the calibration files are copied out next to the
`.exe` on first run so they stay editable.

`resource_path()` resolves bundled assets through `sys._MEIPASS`, so the same
code works frozen and from source. User-editable things (`calibration_files/`,
`settings.json`) deliberately use a different path beside the executable. The
spec excludes `PyQt5`, `PyQt6` and `PySide2`, so a stray Qt binding cannot end
up in the bundle even if something reinstalls one.
