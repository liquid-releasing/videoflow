"""Tests for videoflow.audio — analyze_beats() and AudioBeatMap."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_librosa_mock(
    *,
    bpm: float = 120.0,
    beat_count: int = 32,
    duration_s: float = 16.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> MagicMock:
    """Return a MagicMock that behaves like librosa for a synthetic track."""
    import numpy as np

    mock = MagicMock()

    y = np.zeros(int(sr * duration_s), dtype=np.float32)
    mock.load.return_value = (y, sr)

    # beat_frames evenly spaced at bpm
    beat_interval_frames = int(sr / (bpm / 60) / hop_length)
    beat_frames = np.array(
        [i * beat_interval_frames for i in range(beat_count)], dtype=np.int32
    )
    mock.beat.beat_track.return_value = (np.array([bpm]), beat_frames)

    # frames_to_time: frame * hop / sr
    mock.frames_to_time.return_value = beat_frames * hop_length / sr

    # rms: uniform 0.1 across all frames
    n_frames = int(len(y) / hop_length) + 1
    mock.feature.rms.return_value = np.full((1, n_frames), 0.1, dtype=np.float32)

    mock.get_duration.return_value = duration_s

    return mock


class TestAnalyzeBeats(unittest.TestCase):

    # ------------------------------------------------------------------
    # Happy path — shape and types
    # ------------------------------------------------------------------

    def test_returns_audio_beat_map(self):
        from videoflow.audio import AudioBeatMap
        mock_lib = _make_librosa_mock()
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertIsInstance(result, AudioBeatMap)

    def test_bpm_correct(self):
        mock_lib = _make_librosa_mock(bpm=128.0)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertAlmostEqual(result.bpm, 128.0, places=1)

    def test_beat_count(self):
        mock_lib = _make_librosa_mock(beat_count=32)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(len(result.beats), 32)

    def test_beats_are_milliseconds(self):
        """Beat timestamps must be integers and in ms range."""
        mock_lib = _make_librosa_mock(bpm=120.0, beat_count=8, duration_s=4.0)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        for b in result.beats:
            self.assertIsInstance(b, int)
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, 4000)

    def test_duration_ms(self):
        mock_lib = _make_librosa_mock(duration_s=16.0)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(result.duration_ms, 16000)

    # ------------------------------------------------------------------
    # Downbeats — every 4th beat
    # ------------------------------------------------------------------

    def test_downbeat_count(self):
        """32 beats → 8 downbeats (every 4th)."""
        mock_lib = _make_librosa_mock(beat_count=32)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(len(result.downbeats), 8)

    def test_downbeats_are_subset_of_beats(self):
        mock_lib = _make_librosa_mock(beat_count=32)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        beat_set = set(result.beats)
        for db in result.downbeats:
            self.assertIn(db, beat_set)

    def test_downbeats_are_every_4th_beat(self):
        mock_lib = _make_librosa_mock(beat_count=16)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(result.downbeats, result.beats[::4])

    # ------------------------------------------------------------------
    # Phrases — every 16 beats
    # ------------------------------------------------------------------

    def test_phrase_count_32_beats(self):
        """32 beats → 2 phrases of 16 beats each."""
        mock_lib = _make_librosa_mock(beat_count=32)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(len(result.phrases), 2)

    def test_phrases_are_tuples(self):
        mock_lib = _make_librosa_mock(beat_count=16)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        for phrase in result.phrases:
            self.assertIsInstance(phrase, tuple)
            self.assertEqual(len(phrase), 2)

    def test_phrase_start_lte_end(self):
        mock_lib = _make_librosa_mock(beat_count=20)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        for start, end in result.phrases:
            self.assertLessEqual(start, end)

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def test_energy_length_matches_beats(self):
        mock_lib = _make_librosa_mock(beat_count=32)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(len(result.energy), len(result.beats))

    def test_energy_normalised(self):
        """All energy values must be in [0.0, 1.0]."""
        mock_lib = _make_librosa_mock(beat_count=16)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        for e in result.energy:
            self.assertGreaterEqual(e, 0.0)
            self.assertLessEqual(e, 1.0)

    def test_energy_max_is_one(self):
        """When there is any signal, the peak energy must be 1.0."""
        mock_lib = _make_librosa_mock(beat_count=16)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertAlmostEqual(max(result.energy), 1.0, places=5)

    # ------------------------------------------------------------------
    # beat_interval_ms property
    # ------------------------------------------------------------------

    def test_beat_interval_ms(self):
        """At 120 BPM interval = 500 ms."""
        mock_lib = _make_librosa_mock(bpm=120.0)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertAlmostEqual(result.beat_interval_ms, 500.0, places=3)

    # ------------------------------------------------------------------
    # beats_in_range
    # ------------------------------------------------------------------

    def test_beats_in_range_basic(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000, 1500, 2000],
            downbeats=[0], phrases=[(0, 2000)], energy=[1.0] * 5, duration_ms=2000,
        )
        self.assertEqual(bm.beats_in_range(0, 1000), [0, 500])

    def test_beats_in_range_empty(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], phrases=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.beats_in_range(100, 400), [])

    # ------------------------------------------------------------------
    # nearest_beat
    # ------------------------------------------------------------------

    def test_nearest_beat_exact(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], phrases=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(500), 500)

    def test_nearest_beat_between(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], phrases=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(300), 500)

    def test_nearest_beat_before(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], phrases=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(900, direction="before"), 500)

    def test_nearest_beat_after(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], phrases=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(600, direction="after"), 1000)

    def test_nearest_beat_invalid_direction(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500],
            downbeats=[0], phrases=[(0, 500)], energy=[1.0] * 2, duration_ms=500,
        )
        with self.assertRaises(ValueError):
            bm.nearest_beat(250, direction="sideways")

    def test_nearest_beat_no_beats(self):
        from videoflow.audio import AudioBeatMap, BeatError
        bm = AudioBeatMap(
            bpm=120.0, beats=[], downbeats=[], phrases=[], energy=[], duration_ms=0,
        )
        with self.assertRaises(BeatError):
            bm.nearest_beat(500)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_file_not_found(self):
        from videoflow.audio import analyze_beats
        with self.assertRaises(FileNotFoundError):
            analyze_beats("nonexistent_track.mp3")

    def test_librosa_not_installed(self):
        from videoflow.audio import BeatError, analyze_beats
        with patch("videoflow.audio._librosa", None), \
             patch.object(Path, "exists", return_value=True):
            with self.assertRaises(BeatError) as ctx:
                analyze_beats("track.mp3")
        self.assertIn("librosa", str(ctx.exception).lower())

    def test_librosa_exception_wrapped(self):
        from videoflow.audio import BeatError, analyze_beats
        mock_lib = MagicMock()
        mock_lib.load.side_effect = RuntimeError("codec not found")
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            with self.assertRaises(BeatError) as ctx:
                analyze_beats("track.mp3")
        self.assertIn("codec not found", str(ctx.exception))

    # ------------------------------------------------------------------
    # accepts Path object
    # ------------------------------------------------------------------

    def test_accepts_path_object(self):
        mock_lib = _make_librosa_mock()
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats(Path("track.mp3"))
        self.assertIsNotNone(result)


    # ------------------------------------------------------------------
    # source parameter — HPSS
    # ------------------------------------------------------------------

    def test_source_full_does_not_call_hpss(self):
        mock_lib = _make_librosa_mock()
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", source="full")
        mock_lib.effects.hpss.assert_not_called()

    def test_source_percussive_calls_hpss(self):
        import numpy as np
        mock_lib = _make_librosa_mock()
        # hpss must return (harmonic, percussive) tuple of same shape as y
        y_dummy = np.zeros(22050, dtype=np.float32)
        mock_lib.effects.hpss.return_value = (y_dummy, y_dummy)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", source="percussive")
        mock_lib.effects.hpss.assert_called_once()

    def test_source_percussive_passes_percussive_to_beat_track(self):
        """beat_track must receive the percussive component, not the full mix."""
        import numpy as np
        mock_lib = _make_librosa_mock()
        y_harmonic = np.ones(22050, dtype=np.float32)
        y_percussive = np.zeros(22050, dtype=np.float32)
        mock_lib.effects.hpss.return_value = (y_harmonic, y_percussive)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", source="percussive")
        _, call_kwargs = mock_lib.beat.beat_track.call_args
        passed_y = call_kwargs.get("y", mock_lib.beat.beat_track.call_args[0][0]
                                   if mock_lib.beat.beat_track.call_args[0] else None)
        np.testing.assert_array_equal(passed_y, y_percussive)

    def test_source_percussive_returns_valid_beat_map(self):
        import numpy as np
        mock_lib = _make_librosa_mock()
        y_dummy = np.zeros(22050, dtype=np.float32)
        mock_lib.effects.hpss.return_value = (y_dummy, y_dummy)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import AudioBeatMap, analyze_beats
            result = analyze_beats("track.mp3", source="percussive")
        self.assertIsInstance(result, AudioBeatMap)
        self.assertGreater(result.bpm, 0)

    def test_source_invalid_raises_value_error(self):
        with patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            with self.assertRaises(ValueError) as ctx:
                analyze_beats("track.mp3", source="drums")
        self.assertIn("drums", str(ctx.exception))

    def test_source_default_is_full(self):
        mock_lib = _make_librosa_mock()
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3")   # no source argument
        mock_lib.effects.hpss.assert_not_called()


if __name__ == "__main__":
    unittest.main()
