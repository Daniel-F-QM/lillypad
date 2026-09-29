"""Two overlapping spectrometers presented as one device.

Vendor-neutral AND simulator-neutral: it composes two SpectrometerBase objects
and never asks what they are. That is why it sits beside the interfaces rather
than under stages/, spectrometers/ or sim/ — and the stitching tests prove it by
driving it with a hand-rolled double.
"""

from __future__ import annotations

import numpy as np

from lillypad.hardware.base import OVERLAP_BAND_FRAC, SpectrometerBase

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
    dragged the dead edges into the merged curve.

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
    def __init__(self, spec_a: SpectrometerBase, spec_b: SpectrometerBase):
        wl_a = np.asarray(spec_a.wavelengths, float)
        wl_b = np.asarray(spec_b.wavelengths, float)
        if wl_a.max() <= wl_b.max():
            self.spec1, self.spec2 = spec_a, spec_b
            wl1, wl2 = wl_a, wl_b
        else:
            self.spec1, self.spec2 = spec_b, spec_a
            wl1, wl2 = wl_b, wl_a
        if wl2.min() >= wl1.max():
            raise RuntimeError(
                f"No spectral overlap between {self.spec1.name} "
                f"({wl1.min():.0f}–{wl1.max():.0f} nm) and {self.spec2.name} "
                f"({wl2.min():.0f}–{wl2.max():.0f} nm) — stitching needs "
                f"overlapping wavelength ranges.")
        self._wl1, self._wl2 = wl1, wl2
        # Common grid at the finer of the two pixel spacings (median: real
        # spectrometer grids are not perfectly uniform).
        dwl = min(float(np.median(np.diff(wl1))), float(np.median(np.diff(wl2))))
        self._wl = np.arange(wl1.min(), wl2.max(), dwl)
        #  |-- spec1 only --|== geometric overlap ==|-- spec2 only --|
        self._m1  = self._wl <= wl2.min()
        self._m2  = self._wl >= wl1.max()
        self._ovl = ~(self._m1 | self._m2)
        # Geometric overlap bounds, the hard limits set_band() validates
        # against. Taken from the members' ranges, not from self._wl, so a grid
        # that stops a fraction of a pixel short cannot narrow them.
        self.overlap_lo_nm = float(wl2.min())
        self.overlap_hi_nm = float(wl1.max())
        self.stitch_factor = 1.0
        # RMS relative mismatch left over after the last fit, or None if never
        # fitted. This is what says whether ONE scalar is enough for this pair:
        # small means the two calibrated curves really do agree across the
        # band, large means a calibration is wrong or the band is too wide.
        self.stitch_residual: float | None = None
        self._set_default_band()
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
        """Back to the default band. Returns the new (lo, hi)."""
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
        the signal.

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
            sig1 = np.clip(raw1 - d1, 0, None)
            sig2 = np.clip(raw2 - d2, 0, None)
        return self.spec1.calibrate(sig1), self.spec2.calibrate(sig2)

    def combine(self, i1, i2) -> np.ndarray:
        """Crossfade two CALIBRATED member frames onto the common grid.

        The single definition of what a merged frame IS, so the live view, the
        stored dark and acquire() cannot drift apart: whatever goes through
        here lands on the current stitch factor and the current band.
        """
        i1 = np.asarray(i1, float) * self.member_scale(0)
        # One crossfade over the whole grid. _w1 is already 1 where only spec1
        # has pixels and 0 where only spec2 does, so the edge-held values
        # np.interp produces outside a member's range are multiplied away.
        w1 = self._w1
        return (w1 * np.interp(self._wl, self._wl1, i1)
                + (1.0 - w1) * np.interp(self._wl, self._wl2, np.asarray(i2,
                                                                        float)))

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

    def fit_stitch_factor(self, darks=None) -> float:
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
        """
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
        scale = float(np.max(np.abs(b)))
        self.stitch_residual = (
            float(np.sqrt(np.mean((self.stitch_factor * a - b) ** 2)) / scale)
            if scale > 0.0 else None)
        return self.stitch_factor

    def calibration_targets(self) -> list[SpectrometerBase]:
        return [self.spec1, self.spec2]

    def disconnect(self) -> None:
        for s in (self.spec1, self.spec2):
            try:
                s.disconnect()
            except Exception:
                pass
