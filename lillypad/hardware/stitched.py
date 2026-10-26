"""Two overlapping spectrometers presented as one device.

Vendor-neutral AND simulator-neutral: it composes two SpectrometerBase objects
and never asks what they are. That is why it sits beside the interfaces rather
than under stages/, spectrometers/ or sim/ — and the stitching tests prove it by
driving it with a hand-rolled double.
"""

from __future__ import annotations

import numpy as np

from lillypad.hardware.base import (OVERLAP_BAND_FRAC, PROCESSING_MODES,
                                    SpectrometerBase)

class StitchedSpectrometer(SpectrometerBase):
    """Two spectrometers with overlapping wavelength ranges presented as ONE
    device on a common grid.

    spec1 is whichever device reaches the lower maximum wavelength (the
    "bluer" one). Each member's own intensity calibration is applied to its
    raw frame BEFORE interpolation and stitching — that is what makes the two
    halves comparable — so calibrate() on the stitched device itself stays the
    identity and consumers cannot double-apply anything. spec1 is additionally
    scaled by `stitch_factor` (fit or set by hand) to absorb any residual
    sensitivity mismatch.

    Two methods
    -----------
    `processing` is "lillypad" (everything below) or "legacy", which
    reproduces the legacy program's StitchedSpectrometer exactly: a grid at
    the first-pixel spacing of the finer member, the WHOLE overlap snapped to
    grid points, a flat 50/50 average across it, and a minimize_scalar fit
    over it on averaged frames. The members follow the same switch for their calibration
    (SpectrometerBase.set_processing), and set_processing() here moves all
    three together, all or nothing.

    The overlap band
    ----------------
    The GEOMETRIC overlap runs from spec2's first pixel to spec1's last, and it
    always includes both detectors' dead edges — the region where spec1's
    sensitivity has fallen off the red end and spec2's has not yet risen at the
    blue end. There the counts sit at the noise floor and each member's
    calibration factor is at its steepest, so those samples carry no
    information while contributing maximum noise to a least-squares fit.

    `band_lo_nm`/`band_hi_nm` name the sub-range that is actually used: the fit
    is restricted to it, and the blend crossfades across it (spec1 alone below,
    spec2 alone above, raised-cosine in between). Default is the central
    OVERLAP_BAND_FRAC of the geometric overlap; set_band() moves it. A taper
    rather than the old flat 50/50 average, which stepped at both seams and
    dragged the dead edges into the merged curve. (Legacy mode is that old
    average, on purpose; there the band is the whole overlap and cannot move.)

    The two members hold INDEPENDENT integration times (see
    set_member_integration_time). Frames stay in raw counts, so the exposure
    ratio lands in `stitch_factor` along with the sensitivity ratio: change
    either member's integration time and the factor must be re-fitted, or the
    seam reappears. The GUI flags this rather than compensating silently.

    Darks
    -----
    A dark is held as the two RAW member frames and subtracted per member
    BEFORE calibration (prepare_pair), because a calibration multiplies the
    pedestal along with the signal and each member has its own. The merged
    background is DERIVED from those on demand (combined_dark), never stored:
    a dark captured through acquire() carries the stitch factor and the band it
    was taken at inside its overlap region, where the two devices are already
    mixed, so it cannot be un-mixed and goes silently wrong as soon as either
    moves. Deriving it keeps the merged curve equal to the crossfade of the two
    member curves by construction. Exposure is the one thing this cannot
    absorb — a dark is offset + dark-current x t — so a dark stays valid only
    for the exposures it was recorded at, and the GUI discards it otherwise.

    max_counts is None on purpose: counts on the common grid mix two detectors
    and two calibrations, so no single ADC full scale applies to the merged
    curve. Saturation is still checked — just per member rather than here. Each
    device's RAW frame (last_member_raw, below) is judged against that device's
    own max_counts, live and during a scan, so either detector clipping trips
    its own alarm. See scan.py's FrogScanWorker.run.
    """
    def __init__(self, spec_a: SpectrometerBase, spec_b: SpectrometerBase,
                 method: str = "lillypad"):
        if method not in PROCESSING_MODES:
            raise ValueError(f"method must be one of {PROCESSING_MODES}, "
                             f"got {method!r}")
        # Construction order, kept so the layout can be rebuilt (relayout)
        # when a member's trimmed range or the method changes. spec1/spec2
        # are derived from these by wavelength, every time.
        self._spec_a, self._spec_b = spec_a, spec_b
        # The members calibrate under the same method as the stitch. Brought
        # into line here so a pair can never mix the two.
        for m in (spec_a, spec_b):
            m.set_processing(method)
        self.stitch_factor = 1.0
        # RMS relative mismatch left over after the last fit, or None if never
        # fitted. This is what says whether ONE scalar is enough for this pair:
        # small means the two calibrated curves really do agree across the
        # band, large means a calibration is wrong or the band is too wide.
        self.stitch_residual: float | None = None
        self._apply_layout(self._compute_layout(method))
        self.name = f"stitched: {self.spec1.name} + {self.spec2.name}"
        self.max_counts = None
        # The intersection of the members' ranges: set_integration_time() on
        # the facade writes BOTH devices, so it can only offer what both
        # accept. The per-member boxes in the GUI are seeded from the members
        # themselves and keep their own (possibly wider) ranges.
        self.min_integration_ms = max(m.min_integration_ms for m in self.members)
        self.max_integration_ms = min(m.max_integration_ms for m in self.members)
        self._sync_integration()
        # RAW (pre-calibration) frames of the most recent acquire, one per
        # member. This is what per-device saturation alarms judge: each frame
        # against its own member's max_counts. Tuple assignment is atomic, so
        # the GUI thread may read this while the feed thread acquires.
        self.last_member_raw: tuple[np.ndarray, np.ndarray] | None = None

    # ── Layout: member order, common grid, overlap ───────────────────────────
    def _compute_layout(self, method: str) -> dict:
        """Everything the grid and the overlap depend on, for `method`, from
        the members' CALIBRATED axes (output_wavelengths — legacy mode trims
        them). Pure: raises without touching anything when it cannot be done.
        """
        a, b = self._spec_a, self._spec_b
        wl_a = np.asarray(a.output_wavelengths, float)
        wl_b = np.asarray(b.output_wavelengths, float)
        if wl_a.max() <= wl_b.max():
            spec1, spec2, wl1, wl2 = a, b, wl_a, wl_b
        else:
            spec1, spec2, wl1, wl2 = b, a, wl_b, wl_a
        # The legacy program only refuses a gap (min > max); a pair that just
        # touches goes on with an empty overlap, as it does there.
        gap = (wl2.min() > wl1.max()) if method == "legacy" \
            else (wl2.min() >= wl1.max())
        if gap:
            raise RuntimeError(
                f"No spectral overlap between {spec1.name} "
                f"({wl1.min():.0f}–{wl1.max():.0f} nm) and {spec2.name} "
                f"({wl2.min():.0f}–{wl2.max():.0f} nm) — stitching needs "
                f"overlapping wavelength ranges.")
        layout = dict(method=method, spec1=spec1, spec2=spec2, wl1=wl1, wl2=wl2,
                      # Geometric overlap bounds, the hard limits set_band()
                      # validates against. Taken from the members' ranges,
                      # not from the grid, so a grid that stops a fraction of
                      # a pixel short cannot narrow them.
                      overlap_lo_nm=float(wl2.min()),
                      overlap_hi_nm=float(wl1.max()))
        if method == "legacy":
            layout.update(self._legacy_grid(spec1, spec2, wl1, wl2))
            return layout
        # Common grid at the finer of the two pixel spacings (median: real
        # spectrometer grids are not perfectly uniform).
        dwl = min(float(np.median(np.diff(wl1))), float(np.median(np.diff(wl2))))
        wl = np.arange(wl1.min(), wl2.max(), dwl)
        #  |-- spec1 only --|== geometric overlap ==|-- spec2 only --|
        m1 = wl <= wl2.min()
        m2 = wl >= wl1.max()
        layout.update(wl=wl, m1=m1, m2=m2, ovl=~(m1 | m2))
        return layout

    @staticmethod
    def _legacy_grid(spec1, spec2, wl1, wl2) -> dict:
        """The legacy program's StitchedSpectrometer.__init__, line for line:
        the grid step is the spacing of the FIRST TWO pixels of the finer
        member, and the overlap is snapped to the grid points nearest spec2's
        first and spec1's last pixel (wlo1, wlo2)."""
        for spec, w in ((spec1, wl1), (spec2, wl2)):
            if w.size < 2:
                raise RuntimeError(f"{spec.name} has fewer than two pixels "
                                   f"left — nothing to stitch.")
        dwl1 = wl1[1] - wl1[0]
        dwl2 = wl2[1] - wl2[0]
        dwl = dwl1 if dwl1 < dwl2 else dwl2
        if not dwl > 0:
            raise RuntimeError("Legacy stitching needs ascending wavelength "
                               "axes on both spectrometers.")
        wl = np.arange(np.min(wl1), np.max(wl2), dwl)
        # |--------|-----|--------|
        # wlmin    wlo1   wlo2      wlmax
        wlo1 = wl[np.argmin(abs(wl - np.min(wl2)))]
        wlo2 = wl[np.argmin(abs(wl - np.max(wl1)))]
        ovl = (wl > wlo1) & (wl < wlo2)      # filter for the spectral overlap
        m1 = (wl <= wlo1)
        m2 = (wl >= wlo2)
        return dict(wl=wl, m1=m1, m2=m2, ovl=ovl,
                    wlo1=float(wlo1), wlo2=float(wlo2))

    def _apply_layout(self, layout: dict) -> None:
        self.processing = layout["method"]
        self.spec1, self.spec2 = layout["spec1"], layout["spec2"]
        self._wl1, self._wl2 = layout["wl1"], layout["wl2"]
        self._wl = layout["wl"]
        self._m1, self._m2, self._ovl = layout["m1"], layout["m2"], layout["ovl"]
        self.overlap_lo_nm = layout["overlap_lo_nm"]
        self.overlap_hi_nm = layout["overlap_hi_nm"]
        if self.processing == "legacy":
            # No band to choose: the 50/50 region IS the band, so the overlap
            # shading and the RAW view's cut both land on the legacy overlap.
            self.band_lo_nm, self.band_hi_nm = layout["wlo1"], layout["wlo2"]
            self._fit_mask = self._ovl
            # Informational only (the legacy combine fills the grid region by
            # region), in the same order, so a point in both m1 and m2 goes
            # to spec2 as it does there.
            w1 = np.zeros_like(self._wl)
            w1[self._m1] = 1.0
            w1[self._m2] = 0.0
            w1[self._ovl] = 0.5
            self._w1 = w1
            self.stitch_residual = None
        else:
            self._set_default_band()

    def relayout(self) -> bool:
        """Rebuild the grid after a member's calibrated range changed (a new
        calibration file in legacy mode trims a different set of pixels).

        Returns True when anything changed. A no-op — band, residual and all
        left alone — when the members' axes are what the grid was built from,
        which in Lillypad mode is always. Raises, changing nothing, when the
        new ranges cannot be stitched; the caller restores the member.

        spec1 and spec2 can swap here. The stitch factor then belongs to the
        other device, so it goes back to 1.
        """
        layout = self._compute_layout(self.processing)
        if (layout["spec1"] is self.spec1
                and np.array_equal(layout["wl1"], self._wl1)
                and np.array_equal(layout["wl2"], self._wl2)):
            return False
        swapped = layout["spec1"] is not self.spec1
        self._apply_layout(layout)
        if swapped:
            self.stitch_factor = 1.0
        self.stitch_residual = None
        return True

    def set_processing(self, mode: str) -> int:
        """Move both members' calibration AND the stitch to `mode`.

        All or nothing: if either member's file will not load under `mode`, or
        the trimmed ranges no longer overlap, this raises and the pair stays
        exactly as it was. Returns the members' total of skipped lines. The
        stitch factor is kept, unless spec1 and spec2 swap (see relayout).
        """
        if mode not in PROCESSING_MODES:
            raise ValueError(f"processing must be one of {PROCESSING_MODES}, "
                             f"got {mode!r}")
        members = (self._spec_a, self._spec_b)
        saved = [m.calibration_state() for m in members]
        try:
            skipped = sum(m.set_processing(mode) for m in members)
            layout = self._compute_layout(mode)
        except Exception:
            for m, state in zip(members, saved):
                m.restore_calibration_state(state)
            raise
        if mode == self.processing and layout["spec1"] is self.spec1 \
                and np.array_equal(layout["wl1"], self._wl1) \
                and np.array_equal(layout["wl2"], self._wl2):
            return skipped
        swapped = layout["spec1"] is not self.spec1
        self._apply_layout(layout)
        if swapped:
            self.stitch_factor = 1.0
        self.stitch_residual = None
        return skipped

    @property
    def members(self) -> tuple[SpectrometerBase, SpectrometerBase]:
        return (self.spec1, self.spec2)

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wl

    def member_index(self, member: SpectrometerBase) -> int:
        """0 for spec1 (bluer), 1 for spec2. Identity, not equality — the two
        members are distinct live handles and nothing else can match them."""
        return 0 if member is self.spec1 else 1

    def member_scale(self, index: int) -> float:
        """What acquire() multiplies member `index`'s calibrated frame by.

        The single definition of that factor, so anything drawing the members
        separately lands on the same scale as the combined frame instead of
        re-deriving it and drifting.
        """
        return float(self.stitch_factor) if index == 0 else 1.0

    # ── Overlap band ─────────────────────────────────────────────────────────
    @property
    def overlap_band(self) -> tuple[float, float]:
        """(lo_nm, hi_nm) actually used for the fit and the crossfade."""
        return (self.band_lo_nm, self.band_hi_nm)

    @property
    def geometric_overlap(self) -> tuple[float, float]:
        """(lo_nm, hi_nm) where the two members' ranges intersect at all — the
        hard bounds any band has to sit inside."""
        return (self.overlap_lo_nm, self.overlap_hi_nm)

    def _set_default_band(self) -> None:
        """Centre a band covering OVERLAP_BAND_FRAC of the geometric overlap."""
        lo, hi = self.overlap_lo_nm, self.overlap_hi_nm
        margin = 0.5 * (hi - lo) * (1.0 - OVERLAP_BAND_FRAC)
        self.set_band(lo + margin, hi - margin)

    def reset_band(self) -> tuple[float, float]:
        """Back to the default band. Returns the new (lo, hi). In legacy mode
        the band is the whole overlap already, so nothing moves."""
        if self.processing != "legacy":
            self._set_default_band()
        return self.overlap_band

    def set_band(self, lo_nm: float, hi_nm: float) -> tuple[float, float]:
        """Choose the sub-range of the overlap to fit and crossfade over.

        Rejected rather than clamped when it does not fit inside the geometric
        overlap: a band the operator typed is a statement about where the two
        detectors agree, and silently moving it would leave the green marker in
        the GUI showing something other than what the fit used.

        Recomputes only the masks and the blend weights — the common grid and
        the member masks are untouched, so a band change is free.
        """
        if self.processing == "legacy":
            raise RuntimeError(
                "Legacy mode averages over the whole overlap — there is no band "
                "to set. Switch the method to Lillypad to choose one.")
        lo, hi = float(lo_nm), float(hi_nm)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise RuntimeError("Overlap band bounds must be finite numbers.")
        if lo >= hi:
            raise RuntimeError(
                f"Overlap band low bound ({lo:.2f} nm) must be below the high "
                f"bound ({hi:.2f} nm).")
        glo, ghi = self.overlap_lo_nm, self.overlap_hi_nm
        if lo < glo - 1e-9 or hi > ghi + 1e-9:
            raise RuntimeError(
                f"Overlap band {lo:.2f}–{hi:.2f} nm lies outside the range the "
                f"two spectrometers share ({glo:.2f}–{ghi:.2f} nm).")
        self.band_lo_nm, self.band_hi_nm = lo, hi
        # Fit mask: the band, on the common grid. Restricted to self._ovl too,
        # so a band touching the geometric bounds can never pick up a sample
        # where only one member has real pixels.
        self._fit_mask = self._ovl & (self._wl >= lo) & (self._wl <= hi)
        # Crossfade weight for spec1: 1 below the band, 0 above it, raised
        # cosine across. Defined over the WHOLE grid so acquire() is one
        # vectorised blend with no seams to line up by hand.
        t = np.clip((self._wl - lo) / (hi - lo), 0.0, 1.0)
        self._w1 = 0.5 * (1.0 + np.cos(np.pi * t))
        # Outside each member's own pixel range np.interp holds the edge value,
        # which is not data. Force the weight hard to the member that really
        # covers that wavelength, whatever the ramp says.
        self._w1[self._m1] = 1.0
        self._w1[self._m2] = 0.0
        # A stale fit is worse than none: the factor was chosen over a
        # different set of samples, so its residual no longer describes it.
        self.stitch_residual = None
        return self.overlap_band

    def _sync_integration(self) -> None:
        """Recompute `integration_ms` from the members' current exposures.

        On a stitched device this is the cost of ONE acquire(), not an
        exposure: acquire() drives the two members sequentially, so a frame
        takes t1+t2. Its only consumers — the live feed's pacing sleep and the
        timeout pause() waits out before handing the hardware over — both want
        exactly that cycle time, and a max() here would under-estimate the
        handover by up to 2x at long integrations.
        """
        self.integration_ms = (float(getattr(self.spec1, "integration_ms", 100.0))
                               + float(getattr(self.spec2, "integration_ms", 100.0)))

    def set_integration_time(self, ms: float) -> None:
        """Give BOTH members the same exposure. The single-value path, used
        when a pair is first connected; see set_member_integration_time for
        the per-device one."""
        self.spec1.set_integration_time(ms)
        self.spec2.set_integration_time(ms)
        self._sync_integration()

    def set_member_integration_time(self, index: int, ms: float) -> None:
        """Expose one member independently of the other.

        The two devices usually see very different signal levels, so a single
        shared exposure means one of them is always either buried in read
        noise or clipped. NOTE that this invalidates `stitch_factor` — frames
        stay in raw counts, so the factor carries the exposure ratio too.
        """
        self.members[index].set_integration_time(ms)
        self._sync_integration()

    def prepare_pair(self, raw1, raw2, darks=None) -> tuple[np.ndarray,
                                                            np.ndarray]:
        """Dark-subtract and calibrate two RAW member frames, on their native
        grids. Pure — pass frames already in hand and it touches no hardware.

        `darks` is an optional (dark1, dark2) pair of RAW member frames in
        self.members order, subtracted BEFORE calibrate() — the only order that
        is correct, since a calibration multiplies the dark pedestal along with
        the signal. Lillypad clips the difference at 0; legacy mode keeps
        negatives.

        `member_scale` is deliberately NOT applied here: the per-spectrometer
        view needs it per curve, and combine() applies it for the merged frame.
        """
        raw1 = np.asarray(raw1, float)
        raw2 = np.asarray(raw2, float)
        sig1, sig2 = raw1, raw2
        if darks is not None:
            d1, d2 = (np.asarray(d, float) for d in darks)
            if d1.shape != raw1.shape or d2.shape != raw2.shape:
                raise RuntimeError(
                    "The recorded dark does not match this pair's frames — "
                    "re-record it before fitting.")
            if self.processing == "legacy":
                sig1, sig2 = raw1 - d1, raw2 - d2
            else:
                sig1 = np.clip(raw1 - d1, 0, None)
                sig2 = np.clip(raw2 - d2, 0, None)
        return self.spec1.calibrate(sig1), self.spec2.calibrate(sig2)

    def combine(self, i1, i2) -> np.ndarray:
        """Crossfade two CALIBRATED member frames onto the common grid.

        The single definition of what a merged frame IS, so the live view, the
        stored dark and acquire() cannot drift apart: whatever goes through
        here lands on the current stitch factor and the current band.
        """
        if self.processing == "legacy":
            return self._combine_legacy(i1, i2)
        i1 = np.asarray(i1, float) * self.member_scale(0)
        # One crossfade over the whole grid. _w1 is already 1 where only spec1
        # has pixels and 0 where only spec2 does, so the edge-held values
        # np.interp produces outside a member's range are multiplied away.
        w1 = self._w1
        return (w1 * np.interp(self._wl, self._wl1, i1)
                + (1.0 - w1) * np.interp(self._wl, self._wl2, np.asarray(i2,
                                                                        float)))

    def _combine_legacy(self, i1, i2) -> np.ndarray:
        """The legacy program's StitchedSpectrometer.intensities, line for
        line:
        spec1 alone (scaled) up to wlo1, spec2 alone from wlo2, and the plain
        average of the two in between."""
        wl = self._wl
        I = np.zeros_like(wl)
        I1 = np.asarray(i1, float) * self.member_scale(0)
        I2 = np.asarray(i2, float)
        I[self._m1] = np.interp(wl[self._m1], self._wl1, I1)
        I[self._m2] = np.interp(wl[self._m2], self._wl2, I2)
        o = self._ovl
        I[o] = 0.5 * (np.interp(wl[o], self._wl1, I1)
                      + np.interp(wl[o], self._wl2, I2))
        return I

    def combined_dark(self, darks) -> np.ndarray:
        """The two RAW member darks pushed through the merge pipeline, i.e. the
        background of a frame acquire() would return right now.

        Derived on demand rather than stored: a dark captured through acquire()
        would have the stitch factor and the band of the moment baked into its
        overlap region and could never be un-mixed, so it went silently wrong
        the moment either changed. Costs two interps.
        """
        d1, d2 = darks
        return self.combine(*self.prepare_pair(d1, d2))

    def _acquire_pair(self, darks=None) -> tuple[np.ndarray, np.ndarray]:
        """One frame from each member, calibrated, on their native grids."""
        raw1 = np.asarray(self.spec1.acquire(), float)
        raw2 = np.asarray(self.spec2.acquire(), float)
        self.last_member_raw = (raw1, raw2)
        return self.prepare_pair(raw1, raw2, darks)

    def acquire(self) -> np.ndarray:
        return self.combine(*self._acquire_pair())

    def fit_stitch_factor(self, darks=None, n_frames: int = 1) -> float:
        """Take one frame from each member and choose the factor that makes
        spec1 match spec2 over the OVERLAP BAND, least-squares. The residual is
        quadratic in the factor, so the minimum is the closed form
        s = sum(I1*I2) / sum(I1^2) — no iterative optimiser needed.

        Fitted over the band rather than the whole geometric overlap: the dead
        edges are pure noise and would pull the fit without carrying signal.

        `darks` (raw member frames, self.members order) is subtracted first.
        Pass it whenever a dark has been recorded — a long-exposure member
        carries a pedestal that the fit would otherwise partly match instead of
        matching the light.

        The fit is over raw counts, so the result is the sensitivity ratio
        TIMES the exposure ratio t2/t1. That makes it exposure-dependent: any
        change to either member's integration time leaves it stale and it has
        to be re-fitted.

        Also records `stitch_residual`, the RMS of (s*a - b) over the band
        relative to that band's peak — how well one scalar actually did.

        In legacy mode see _fit_legacy instead; `n_frames` is used only there.
        """
        if self.processing == "legacy":
            return self._fit_legacy(darks, n_frames)
        i1, i2 = self._acquire_pair(darks)
        o = self._fit_mask
        if not o.any():
            raise RuntimeError(
                f"The overlap band {self.band_lo_nm:.1f}–{self.band_hi_nm:.1f} "
                f"nm contains no grid points — widen it.")
        a = np.interp(self._wl[o], self._wl1, i1)
        b = np.interp(self._wl[o], self._wl2, i2)
        denom = float(np.dot(a, a))
        if not np.isfinite(denom) or denom <= 0.0:
            raise RuntimeError(
                f"No signal from {self.spec1.name} in the overlap band "
                f"{self.band_lo_nm:.1f}–{self.band_hi_nm:.1f} nm — cannot fit "
                f"a stitch factor.")
        self.stitch_factor = float(np.dot(a, b)) / denom
        self._record_residual(a, b)
        return self.stitch_factor

    def _fit_legacy(self, darks, n_frames) -> float:
        """The legacy program's fit_stitch_factor, line for line.

        Each member averages `n_frames` raw frames (the legacy navg, the frames
        per scan point), has its dark subtracted when one is passed (there:
        when Subtract Dark is ticked), is calibrated and trimmed, and is
        interpolated onto the whole overlap; the factor is whatever
        scipy.optimize.minimize_scalar makes of sum(|s*I1 - I2|^2) there.

        Unlike the legacy program, an empty overlap, a spec1 with no signal
        there or a minimisation that fails raises instead of leaving an
        arbitrary or the old factor in place without a word.
        """
        n = max(1, int(n_frames))
        acc1 = acc2 = None
        for _ in range(n):
            raw1 = np.asarray(self.spec1.acquire(), float)
            raw2 = np.asarray(self.spec2.acquire(), float)
            self.last_member_raw = (raw1, raw2)
            acc1 = raw1.copy() if acc1 is None else acc1 + raw1
            acc2 = raw2.copy() if acc2 is None else acc2 + raw2
        i1, i2 = self.prepare_pair(acc1 / n, acc2 / n, darks)
        o = self._ovl
        if not o.any():
            raise RuntimeError(
                f"The overlap {self.band_lo_nm:.1f}–{self.band_hi_nm:.1f} nm "
                f"contains no grid points — cannot fit a stitch factor.")
        I1 = np.interp(self._wl[o], self._wl1, i1)
        I2 = np.interp(self._wl[o], self._wl2, i2)
        if not np.any(I1):
            raise RuntimeError(
                f"No signal from {self.spec1.name} in the overlap "
                f"{self.band_lo_nm:.1f}–{self.band_hi_nm:.1f} nm — cannot fit "
                f"a stitch factor.")
        # Imported here, not at module scope: lillypad.hardware must import
        # with numpy alone (test_layering.py). A literal import inside the
        # function is still what PyInstaller's analysis finds.
        from scipy.optimize import minimize_scalar
        minfunc = lambda s: np.sum(np.abs(I1 * s - I2) ** 2)  # noqa: E731
        res = minimize_scalar(minfunc)
        if not res.success:
            raise RuntimeError(f"Stitch-factor minimisation did not converge: "
                               f"{res.message}")
        self.stitch_factor = float(res.x)
        self._record_residual(I1, I2)
        return self.stitch_factor

    def _record_residual(self, a, b) -> None:
        """stitch_residual for the factor just fitted over samples a, b."""
        scale = float(np.max(np.abs(b)))
        self.stitch_residual = (
            float(np.sqrt(np.mean((self.stitch_factor * a - b) ** 2)) / scale)
            if scale > 0.0 else None)

    def calibration_targets(self) -> list[SpectrometerBase]:
        return [self.spec1, self.spec2]

    def disconnect(self) -> None:
        for s in (self.spec1, self.spec2):
            try:
                s.disconnect()
            except Exception:
                pass
