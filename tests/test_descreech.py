"""Tests for the source-audio de-screech limiter."""

from __future__ import annotations

import numpy as np

from videoflow.descreech import THRESHOLD, ScreechRegion, descreech


SR = 22050


def _tone(freq, dur, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_clean_audio_passes_through():
    y = _tone(220, 1.0, amp=0.4)  # well below threshold
    out, regions = descreech(y, SR)
    assert regions == []
    np.testing.assert_allclose(out, y, atol=1e-6)


def test_clipped_spike_is_limited_and_reported():
    y = _tone(220, 1.0, amp=0.3)
    # inject a loud clipped screech in the middle
    lo, hi = int(0.45 * SR), int(0.55 * SR)
    y[lo:hi] = _tone(6000, (hi - lo) / SR, amp=1.6).astype(np.float32)
    out, regions = descreech(y, SR)
    # peak is pulled toward the threshold...
    assert np.abs(out).max() < np.abs(y).max()
    assert np.abs(out[lo:hi]).max() <= THRESHOLD * 1.25
    # ...and the region is reported around the spike
    assert regions
    r = max(regions, key=lambda r: r.max_gr_db)
    assert 0.4 < r.start_s < 0.6
    assert r.peak_amp > 1.0
    assert r.max_gr_db > 3.0


def test_empty_input():
    out, regions = descreech(np.array([], dtype=np.float32), SR)
    assert out.size == 0
    assert regions == []


def test_region_dict_is_json_friendly():
    r = ScreechRegion(start_s=1.234567, end_s=1.3, peak_amp=1.55, max_gr_db=4.21)
    d = r.as_dict()
    assert d == {"start_s": 1.235, "end_s": 1.3, "peak_amp": 1.55, "max_gr_db": 4.21}


def test_dtype_preserved():
    y = _tone(220, 0.2).astype(np.float32)
    out, _ = descreech(y, SR)
    assert out.dtype == np.float32
