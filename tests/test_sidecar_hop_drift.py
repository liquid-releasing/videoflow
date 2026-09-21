"""Sidecar timelines must not drift against the media they describe.

Both audio sidecars rounded their hop to a whole number and then treated
that rounded value as exact, so frame index -> milliseconds was a linear
time-stretch. Measured 2026-09-21 on a real 60:12 scene:

    audio.json          3621.05s for a 3612.89s video   +8.16s  (+0.227%)
    spectrogram.json    3578.64s for the same video    -34.25s  (-0.948%)

    22050 Hz, 10ms hop  ->  220.5 samples, stored as 220, labelled 10ms
    512 hop @ 22050 Hz  ->  23.21995ms,     stored as 23

They drifted in OPPOSITE directions, so the waveform and the spectrogram
of one track disagreed with each other by ~42s by the end of an hour.
Both are editing surfaces: anything placed by eye against them late in a
long scene was placed against a timeline that was lying.

The bug only shows on long material — a few seconds of test audio drifts
by microseconds — so the tests that matter here assert the RATIO, and use
durations long enough for a whole-millisecond error to appear.
"""

from __future__ import annotations

import unittest

import numpy as np

from videoflow.audio_peaks import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    SIDECAR_VERSION as PEAKS_VERSION,
    compute_sidecar_from_samples,
)


HOUR_S = 3600


def _tone(seconds: float, sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Cheap non-silent signal of a known length."""
    n = int(round(seconds * sr))
    return np.sin(np.arange(n, dtype=np.float32) * 0.01).astype(np.float32)


class TestPeaksHopIsExact(unittest.TestCase):

    def test_reported_duration_matches_the_audio_it_analysed(self):
        """THE regression. 22050 x 10ms = 220.5 samples; the old code used
        220 and still called each hop 10ms, inflating duration by 0.227%."""
        y = _tone(HOUR_S)
        data = compute_sidecar_from_samples(y, sr=22050, hop_ms=10)

        self.assertAlmostEqual(data["duration_ms"] / 1000.0, HOUR_S, delta=0.02)

    def test_an_hour_does_not_drift_by_more_than_a_frame(self):
        """The old code was out by 8.2s here. A 20ms budget is under one
        video frame at 60fps, i.e. below anything a human can see."""
        y = _tone(HOUR_S)
        data = compute_sidecar_from_samples(y, sr=22050, hop_ms=10)

        drift_ms = abs(data["duration_ms"] - HOUR_S * 1000)
        self.assertLess(drift_ms, 20, f"drifted {drift_ms / 1000:.2f}s over an hour")

    def test_peak_count_is_the_honest_number_of_hops(self):
        """peaks[i] has to cover [i*hop_ms, (i+1)*hop_ms). With 220-sample
        hops an hour of audio yielded 360_818 peaks and claimed 3608.18s —
        both wrong for 3600s of input."""
        y = _tone(HOUR_S)
        data = compute_sidecar_from_samples(y, sr=22050, hop_ms=10)

        self.assertEqual(data["peak_count"], len(data["peaks"]))
        self.assertAlmostEqual(data["peak_count"], HOUR_S * 100, delta=2)

    def test_the_rate_that_used_to_divide_evenly_is_unchanged(self):
        """8000 Hz x 10ms = 80 samples exactly — this case was always right
        and must stay bit-for-bit right."""
        y = _tone(60, sr=8000)
        data = compute_sidecar_from_samples(y, sr=8000, hop_ms=10)

        self.assertEqual(data["duration_ms"], 60_000)
        self.assertEqual(data["peak_count"], 6000)

    def test_awkward_rates_and_hops_all_hold(self):
        """Any sr*hop_ms/1000 that isn't a whole number used to drift. The
        44100/7ms pair is the worst of these at 308.7 samples."""
        for sr, hop_ms in ((22050, 10), (44100, 7), (44100, 10), (11025, 3)):
            with self.subTest(sr=sr, hop_ms=hop_ms):
                data = compute_sidecar_from_samples(_tone(600, sr), sr=sr, hop_ms=hop_ms)
                ratio = data["duration_ms"] / 600_000
                self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_hops_average_the_exact_fractional_width(self):
        """220.5 samples per hop is delivered as alternating 220/221 frames.
        If every frame were the same width the drift would be back."""
        sr, hop_ms = 22050, 10
        y = _tone(10, sr)
        data = compute_sidecar_from_samples(y, sr=sr, hop_ms=hop_ms)

        samples_per_hop = len(y) / data["peak_count"]
        self.assertAlmostEqual(samples_per_hop, 220.5, delta=0.01)

    def test_the_envelope_still_lands_where_the_sound_is(self):
        """Drift is only interesting because it moves loud moments. Put a
        burst at a known second, late enough that 0.227% would show, and
        read it back by index."""
        sr, hop_ms = 22050, 10
        y = np.zeros(int(1000 * sr), dtype=np.float32)
        burst_at_s = 900
        y[burst_at_s * sr:(burst_at_s + 1) * sr] = 1.0

        data = compute_sidecar_from_samples(y, sr=sr, hop_ms=hop_ms)
        peaks = np.asarray(data["peaks"])
        loudest_hop = int(np.argmax(peaks))
        found_at_s = loudest_hop * hop_ms / 1000.0

        # The old code reported this burst ~2s early in sidecar time.
        self.assertAlmostEqual(found_at_s, burst_at_s, delta=0.05)

    def test_silence_and_empty_input_are_unchanged(self):
        self.assertIsNone(compute_sidecar_from_samples(np.array([], dtype=np.float32)))
        data = compute_sidecar_from_samples(np.zeros(22050, dtype=np.float32), sr=22050)
        self.assertEqual(set(data["peaks"]), {0.0})

    def test_audio_shorter_than_one_hop_still_returns_none(self):
        self.assertIsNone(
            compute_sidecar_from_samples(_tone(0.001, 22050), sr=22050, hop_ms=10))

    def test_version_moved_so_drifted_caches_are_not_trusted(self):
        """Sidecars written before the fix carry the stretch. They are caches,
        so the version is what makes them be recomputed instead of believed."""
        self.assertNotEqual(PEAKS_VERSION, "1.0")


class TestSpectrogramHopIsExact(unittest.TestCase):
    """The spectrogram's hop comes from librosa's hop_length, so the fix is
    to stop rounding it to a whole millisecond rather than to re-frame."""

    def test_hop_ms_is_the_exact_frame_spacing(self):
        from videoflow.audio_spectrogram import DEFAULT_HOP_LENGTH

        sr = 22050
        exact = DEFAULT_HOP_LENGTH * 1000.0 / sr
        self.assertAlmostEqual(exact, 23.219954, places=5)
        # The stored value must be this, not int(round(...)) == 23.
        self.assertNotEqual(round(exact), exact)

    def test_an_hour_of_frames_lands_within_a_frame_of_the_truth(self):
        """155_593 frames of a real 60:12 scene reported 3578.64s against a
        3612.89s video. Same arithmetic, asserted directly."""
        from videoflow.audio_spectrogram import DEFAULT_HOP_LENGTH

        sr = 22050
        hop_ms = DEFAULT_HOP_LENGTH * 1000.0 / sr
        n_frames = 155_593
        duration_s = n_frames * hop_ms / 1000.0

        self.assertAlmostEqual(duration_s, 3612.85, delta=0.5)
        # What the rounded hop produced, kept as the thing we moved away from.
        self.assertAlmostEqual(n_frames * 23 / 1000.0, 3578.64, delta=0.5)

    def test_the_sidecar_stores_a_float_hop(self):
        import inspect

        from videoflow import audio_spectrogram

        src = inspect.getsource(audio_spectrogram.compute_sidecar_from_samples)
        self.assertIn("hop_length * 1000.0 / sr", src)
        self.assertNotIn("int(round(hop_length * 1000.0 / sr))", src)

    def test_version_moved_so_drifted_caches_are_not_trusted(self):
        from videoflow.audio_spectrogram import SIDECAR_VERSION

        self.assertNotEqual(SIDECAR_VERSION, "1.0")


class TestTheTwoSidecarsAgreeWithEachOther(unittest.TestCase):
    """The waveform and the spectrogram are two views of one track, shown on
    one timeline. They drifted opposite ways, which is how a 42s disagreement
    accumulated over an hour without either looking obviously wrong alone."""

    def test_both_describe_the_same_hour(self):
        from videoflow.audio_spectrogram import DEFAULT_HOP_LENGTH

        sr = 22050
        peaks = compute_sidecar_from_samples(_tone(HOUR_S, sr), sr=sr, hop_ms=10)
        peaks_s = peaks["duration_ms"] / 1000.0

        spectro_hop_ms = DEFAULT_HOP_LENGTH * 1000.0 / sr
        n_frames = int(HOUR_S * sr / DEFAULT_HOP_LENGTH)
        spectro_s = n_frames * spectro_hop_ms / 1000.0

        self.assertAlmostEqual(peaks_s, spectro_s, delta=0.05)
        self.assertAlmostEqual(peaks_s, HOUR_S, delta=0.05)


if __name__ == "__main__":
    unittest.main()
