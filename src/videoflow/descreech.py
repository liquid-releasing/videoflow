"""Source-audio de-screech: tame clipped/overdriven spikes before analysis.

A brief loud/clipped transient in the source master (a "screech") inflates
the local onset strength and per-beat RMS, which downstream becomes a
spurious max-intensity beat — and, once the e-stim channels are derived,
a painful flash (see
``funscriptforge/internal/screech_safety_architecture.md``).

This is a feed-forward peak limiter applied to the loaded mono buffer
*for analysis only*. It is deliberately gentle: the threshold sits high
so normal dynamics pass untouched and only the clipped overshoots are
pulled down. Empirically (VictoriaOaks clip 14) classic declipping is a
no-op here — the defect is codec overshoot, not flat-top clipping — but a
limiter/compressor cuts the offending energy spike, which is what stops
the analyzer over-reacting.

Pure numpy, no extra dependencies, so it runs on both the video-extracted
WAV path and direct-audio loads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d


# Threshold (fraction of full scale) above which gain reduction kicks in.
# High on purpose — only clipped/overdriven peaks exceed it, so ordinary
# loud-but-clean passages are left alone.
THRESHOLD = 0.85
# Compression ratio above the threshold (8:1 ≈ limiting).
RATIO = 8.0
# Envelope follower times. Fast attack catches transients; moderate release
# avoids pumping. In seconds.
ATTACK_S = 0.002
RELEASE_S = 0.050
# A region is reported (for the sidecar / UI markers) when sustained gain
# reduction exceeds this, for at least MIN_REGION_S.
REPORT_GR_DB = 3.0
MIN_REGION_S = 0.020
# Merge reported regions separated by less than this.
REGION_MERGE_GAP_S = 0.150


@dataclass(frozen=True)
class ScreechRegion:
    """A span the limiter pulled down, for the sidecar report / UI markers."""

    start_s: float
    end_s: float
    peak_amp: float  # max |sample| in the span before limiting (may exceed 1.0)
    max_gr_db: float  # peak gain reduction applied in the span

    def as_dict(self) -> dict:
        return {
            "start_s": round(float(self.start_s), 3),
            "end_s": round(float(self.end_s), 3),
            "peak_amp": round(float(self.peak_amp), 3),
            "max_gr_db": round(float(self.max_gr_db), 2),
        }


def descreech(
    y: np.ndarray, sr: int, *, threshold: float = THRESHOLD, ratio: float = RATIO
) -> tuple[np.ndarray, list[ScreechRegion]]:
    """Limit clipped peaks in mono `y`; return (cleaned, reported regions).

    The returned buffer is the same shape/dtype as `y`. Regions list the
    spans where meaningful gain reduction was applied (clipped/screech
    moments) so callers can write a sidecar and draw timeline markers.
    """
    if y.size == 0:
        return y, []

    x = np.asarray(y, dtype=np.float64)
    env = _envelope(np.abs(x), sr)

    # Static gain curve: unity below threshold, (thr/env)^(1-1/ratio) above.
    over = env > threshold
    gain = np.ones_like(env)
    e_over = env[over]
    gain[over] = (threshold / e_over) ** (1.0 - 1.0 / ratio)

    # Smooth the gain so we don't introduce our own clicks (release-shaped).
    gain = _smooth_gain(gain, sr)
    cleaned = (x * gain).astype(y.dtype, copy=False)

    gr_db = -20.0 * np.log10(np.maximum(gain, 1e-9))
    regions = _regions(np.abs(x), gr_db, sr)
    return cleaned, regions


# ── internals ─────────────────────────────────────────────────────────────────

def _envelope(absx: np.ndarray, sr: int) -> np.ndarray:
    """Peak-hold envelope over |x| (vectorized).

    A symmetric max filter spanning roughly attack+release acts as a small
    look-ahead peak detector: gain reduction begins just before a peak and
    releases just after, so the limiter never overshoots. O(n) via
    ``scipy.ndimage`` — essential on multi-million-sample files.
    """
    win = max(1, int((ATTACK_S + RELEASE_S) * sr))
    return maximum_filter1d(absx, size=win, mode="nearest")


def _smooth_gain(gain: np.ndarray, sr: int) -> np.ndarray:
    """Moving-average smoothing of the gain curve to avoid introducing clicks."""
    win = max(1, int(RELEASE_S * sr))
    return uniform_filter1d(gain, size=win, mode="nearest")


def _regions(absx: np.ndarray, gr_db: np.ndarray, sr: int) -> list[ScreechRegion]:
    mask = gr_db >= REPORT_GR_DB
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    gap = int(REGION_MERGE_GAP_S * sr)
    min_len = int(MIN_REGION_S * sr)
    regions: list[ScreechRegion] = []
    run_start = idx[0]
    prev = idx[0]
    for j in idx[1:]:
        if j - prev > gap:
            _emit(regions, absx, gr_db, run_start, prev, sr, min_len)
            run_start = j
        prev = j
    _emit(regions, absx, gr_db, run_start, prev, sr, min_len)
    return regions


def _emit(regions, absx, gr_db, lo, hi, sr, min_len):
    if hi - lo + 1 < min_len:
        return
    regions.append(
        ScreechRegion(
            start_s=lo / sr,
            end_s=hi / sr,
            peak_amp=float(absx[lo : hi + 1].max()),
            max_gr_db=float(gr_db[lo : hi + 1].max()),
        )
    )
