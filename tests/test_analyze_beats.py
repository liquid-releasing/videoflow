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
    # Stanzas — every 16 beats
    # ------------------------------------------------------------------

    def test_stanza_count_32_beats(self):
        """32 beats → 2 stanzas of 16 beats each."""
        mock_lib = _make_librosa_mock(beat_count=32)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        self.assertEqual(len(result.stanzas), 2)

    def test_stanzas_are_tuples(self):
        mock_lib = _make_librosa_mock(beat_count=16)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        for stanza in result.stanzas:
            self.assertIsInstance(stanza, tuple)
            self.assertEqual(len(stanza), 2)

    def test_stanza_start_lte_end(self):
        mock_lib = _make_librosa_mock(beat_count=20)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats("track.mp3")
        for start, end in result.stanzas:
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
            downbeats=[0], stanzas=[(0, 2000)], energy=[1.0] * 5, duration_ms=2000,
        )
        self.assertEqual(bm.beats_in_range(0, 1000), [0, 500])

    def test_beats_in_range_empty(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], stanzas=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.beats_in_range(100, 400), [])

    # ------------------------------------------------------------------
    # damaged_after_ms (corrupt-audio salvage marker)
    # ------------------------------------------------------------------

    def test_damaged_after_ms_round_trips(self):
        import json, tempfile, os
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500], downbeats=[0], stanzas=[(0, 500)],
            energy=[1.0, 1.0], duration_ms=500, damaged_after_ms=1_285_000,
        )
        self.assertEqual(bm.to_dict()["damaged_after_ms"], 1_285_000)
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        f.close()
        try:
            bm.save(f.name)
            self.assertEqual(AudioBeatMap.load(f.name).damaged_after_ms, 1_285_000)
        finally:
            os.unlink(f.name)

    def test_damaged_after_ms_defaults_none_and_omitted(self):
        # Healthy sources carry no marker and stay compact on disk.
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0], downbeats=[0], stanzas=[(0, 1)],
            energy=[1.0], duration_ms=1,
        )
        self.assertIsNone(bm.damaged_after_ms)
        self.assertNotIn("damaged_after_ms", bm.to_dict())

    # ------------------------------------------------------------------
    # nearest_beat
    # ------------------------------------------------------------------

    def test_nearest_beat_exact(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], stanzas=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(500), 500)

    def test_nearest_beat_between(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], stanzas=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(300), 500)

    def test_nearest_beat_before(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], stanzas=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(900, direction="before"), 500)

    def test_nearest_beat_after(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500, 1000],
            downbeats=[0], stanzas=[(0, 1000)], energy=[1.0] * 3, duration_ms=1000,
        )
        self.assertEqual(bm.nearest_beat(600, direction="after"), 1000)

    def test_nearest_beat_invalid_direction(self):
        from videoflow.audio import AudioBeatMap
        bm = AudioBeatMap(
            bpm=120.0, beats=[0, 500],
            downbeats=[0], stanzas=[(0, 500)], energy=[1.0] * 2, duration_ms=500,
        )
        with self.assertRaises(ValueError):
            bm.nearest_beat(250, direction="sideways")

    def test_nearest_beat_no_beats(self):
        from videoflow.audio import AudioBeatMap, BeatError
        bm = AudioBeatMap(
            bpm=120.0, beats=[], downbeats=[], stanzas=[], energy=[], duration_ms=0,
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

    # ------------------------------------------------------------------
    # tracker parameter — auto / beat_track / plp
    # ------------------------------------------------------------------

    def test_tracker_invalid_raises_value_error(self):
        with patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            with self.assertRaises(ValueError) as ctx:
                analyze_beats("track.mp3", tracker="madmom")
        self.assertIn("madmom", str(ctx.exception))

    def test_tracker_auto_short_track_uses_beat_track(self):
        """Track <= 10 min → auto picks beat_track, not plp."""
        mock_lib = _make_librosa_mock(duration_s=60.0, beat_count=120)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3")  # default tracker="auto"
        mock_lib.beat.beat_track.assert_called_once()
        mock_lib.beat.plp.assert_not_called()

    def test_tracker_auto_long_track_uses_plp(self):
        """Track > 10 min → auto picks plp."""
        import numpy as np
        mock_lib = _make_librosa_mock(duration_s=900.0, beat_count=64)  # 15 min
        # PLP returns a pulse curve
        mock_lib.beat.plp.return_value = np.zeros(2000, dtype=np.float32)
        # localmax marks every 50th frame as a beat → ~40 beats
        localmax_arr = np.zeros(2000, dtype=bool)
        localmax_arr[::50] = True
        mock_lib.util.localmax.return_value = localmax_arr
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3")  # default tracker="auto"
        mock_lib.beat.plp.assert_called_once()
        mock_lib.beat.beat_track.assert_not_called()

    def test_tracker_beat_track_explicit(self):
        """Explicit tracker='beat_track' overrides auto on long tracks."""
        mock_lib = _make_librosa_mock(duration_s=1800.0, beat_count=120)  # 30 min
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", tracker="beat_track")
        mock_lib.beat.beat_track.assert_called_once()
        mock_lib.beat.plp.assert_not_called()

    def test_tracker_plp_explicit_short_track(self):
        """Explicit tracker='plp' uses PLP even on short tracks."""
        import numpy as np
        mock_lib = _make_librosa_mock(duration_s=10.0, beat_count=20)
        mock_lib.beat.plp.return_value = np.zeros(500, dtype=np.float32)
        lm = np.zeros(500, dtype=bool)
        lm[::25] = True
        mock_lib.util.localmax.return_value = lm
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", tracker="plp")
        mock_lib.beat.plp.assert_called_once()
        mock_lib.beat.beat_track.assert_not_called()

    def test_tracker_plp_returns_valid_beat_map(self):
        import numpy as np
        mock_lib = _make_librosa_mock(duration_s=60.0)
        # Pulse with localmax every 50 frames @ 22050 Hz / 512 hop ≈ 1.16s
        # → BPM ≈ 60 / 1.16 ≈ 51.7
        n_frames = 2000
        pulse = np.zeros(n_frames, dtype=np.float32)
        mock_lib.beat.plp.return_value = pulse
        lm = np.zeros(n_frames, dtype=bool)
        lm[::50] = True  # every 50 frames
        mock_lib.util.localmax.return_value = lm
        # frames_to_time: frame * 512 / 22050
        mock_lib.frames_to_time.side_effect = lambda frames, sr: (
            np.asarray(frames) * 512 / sr
        )
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import AudioBeatMap, analyze_beats
            result = analyze_beats("track.mp3", tracker="plp")
        self.assertIsInstance(result, AudioBeatMap)
        self.assertGreater(len(result.beats), 0)
        self.assertGreater(result.bpm, 0)

    # ------------------------------------------------------------------
    # locked_bpm
    # ------------------------------------------------------------------

    def test_locked_bpm_invalid_raises_value_error(self):
        with patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            with self.assertRaises(ValueError):
                analyze_beats("track.mp3", locked_bpm=0)
            with self.assertRaises(ValueError):
                analyze_beats("track.mp3", locked_bpm=-10)

    def test_locked_bpm_overrides_detected_for_beat_track(self):
        """locked_bpm pins the reported BPM regardless of librosa's detection."""
        mock_lib = _make_librosa_mock(bpm=137.0)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats(
                "track.mp3", tracker="beat_track", locked_bpm=120.0
            )
        self.assertAlmostEqual(result.bpm, 120.0, places=3)

    def test_locked_bpm_passes_start_bpm_to_beat_track(self):
        mock_lib = _make_librosa_mock()
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", __import__("numpy")), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", tracker="beat_track", locked_bpm=128.0)
        _, call_kwargs = mock_lib.beat.beat_track.call_args
        self.assertAlmostEqual(call_kwargs["start_bpm"], 128.0, places=3)
        self.assertGreaterEqual(call_kwargs["tightness"], 100)

    def test_locked_bpm_narrows_plp_tempo_window(self):
        import numpy as np
        mock_lib = _make_librosa_mock(duration_s=60.0)
        mock_lib.beat.plp.return_value = np.zeros(500, dtype=np.float32)
        lm = np.zeros(500, dtype=bool)
        lm[::25] = True
        mock_lib.util.localmax.return_value = lm
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", tracker="plp", locked_bpm=140.0)
        _, call_kwargs = mock_lib.beat.plp.call_args
        self.assertAlmostEqual(call_kwargs["tempo_min"], 138.0, places=3)
        self.assertAlmostEqual(call_kwargs["tempo_max"], 142.0, places=3)

    def test_locked_bpm_overrides_detected_for_plp(self):
        import numpy as np
        mock_lib = _make_librosa_mock(duration_s=60.0)
        mock_lib.beat.plp.return_value = np.zeros(500, dtype=np.float32)
        lm = np.zeros(500, dtype=bool)
        lm[::25] = True
        mock_lib.util.localmax.return_value = lm
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            result = analyze_beats(
                "track.mp3", tracker="plp", locked_bpm=140.0
            )
        self.assertAlmostEqual(result.bpm, 140.0, places=3)


class TestAnalyzeBeatsWithChapters(unittest.TestCase):
    """Chapter-aware analyze_beats — per-chunk analysis + stitched output.

    Uses real synthetic audio rather than mocking, since the chapter path
    slices the buffer and runs librosa on each slice — mocking those calls
    would obscure whether the slicing/stitching actually does the right
    thing.
    """

    @staticmethod
    def _click_audio(duration_s: float, bpm: float, sr: int = 22050):
        """Sharp impulses at *bpm* — a clean signal for beat-tracker tests."""
        import numpy as np
        y = np.zeros(int(duration_s * sr), dtype=np.float32)
        interval = int(sr * 60.0 / bpm)
        for i in range(0, len(y), interval):
            burst = min(int(sr * 0.005), len(y) - i)
            y[i:i + burst] = np.linspace(1.0, 0.0, burst)
        return y

    def _write_wav(self, path, y, sr=22050):
        import soundfile as sf
        sf.write(str(path), y, sr, subtype="FLOAT")
        return path

    def test_chapters_path_returns_stitched_timeline(self):
        """beats / stanzas / energy concat in chapter order; timestamps ascending."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats
        from videoflow.chapters import Chapter

        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(60.0, 120.0),
            )
            chapters = [
                Chapter(at_ms=0, end_ms=30_000),
                Chapter(at_ms=30_000, end_ms=60_000),
            ]
            result = analyze_beats(path, chapters=chapters)

        self.assertGreater(len(result.beats), 0)
        self.assertEqual(result.beats, sorted(result.beats))
        # Each chapter spans 30s — beats should land in both halves
        first_half = [b for b in result.beats if b < 30_000]
        second_half = [b for b in result.beats if b >= 30_000]
        self.assertGreater(len(first_half), 5)
        self.assertGreater(len(second_half), 5)

    def test_chapters_path_per_chunk_energy_normalization(self):
        """Each chunk's max energy = 1.0 — the key fix for ambient-vs-music drift."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats
        from videoflow.chapters import Chapter

        with TemporaryDirectory() as td:
            # Quiet first half (amp 0.05) + loud second half (amp 0.6)
            quiet = self._click_audio(30.0, 120.0) * 0.05
            loud = self._click_audio(30.0, 120.0) * 0.6
            import numpy as np
            y = np.concatenate([quiet, loud])
            path = self._write_wav(Path(td) / "mixed.wav", y)
            chapters = [
                Chapter(at_ms=0, end_ms=30_000),
                Chapter(at_ms=30_000, end_ms=60_000),
            ]
            result = analyze_beats(path, chapters=chapters)

        first_energies = [
            e for b, e in zip(result.beats, result.energy) if b < 30_000
        ]
        second_energies = [
            e for b, e in zip(result.beats, result.energy) if b >= 30_000
        ]
        # Both chunks should have a beat that hits 1.0 in their normalized energy.
        # Without per-chunk normalization, the quiet half would peak ~0.08.
        self.assertGreater(max(first_energies), 0.95,
                           "first chunk should self-normalize to 1.0")
        self.assertGreater(max(second_energies), 0.95,
                           "second chunk should self-normalize to 1.0")

    def test_chapters_none_matches_default_path(self):
        """Passing chapters=None is equivalent to the existing whole-file path."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats

        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(20.0, 120.0),
            )
            a = analyze_beats(path)
            b = analyze_beats(path, chapters=None)
        self.assertEqual(a.beats, b.beats)
        self.assertEqual(a.energy, b.energy)
        self.assertEqual(a.duration_ms, b.duration_ms)

    def test_chapters_progress_callback_per_chunk(self):
        """progress_callback fires a per-chunk label like 'Analyzing chapter X/Y…'."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats
        from videoflow.chapters import Chapter

        labels: list[str] = []
        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(40.0, 120.0),
            )
            chapters = [
                Chapter(at_ms=0, end_ms=20_000),
                Chapter(at_ms=20_000, end_ms=40_000),
            ]
            analyze_beats(path, chapters=chapters, progress_callback=labels.append)

        # The per-chunk announcement fires once per chapter span. (The chunk
        # label now also PREFIXES each sub-step message — HPSS/beats/stanzas —
        # so the footer keeps showing progress on long files; filter to the
        # announcement line to count spans.)
        announce = [lb for lb in labels if lb.startswith("Analyzing chapter")]
        self.assertEqual(len(announce), 2)
        self.assertIn("1/2", announce[0])
        self.assertIn("2/2", announce[1])

    def test_empty_chapters_falls_back_to_whole_file(self):
        """Empty list is treated as 'no chapters' — whole-file analysis runs."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats

        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(15.0, 120.0),
            )
            result = analyze_beats(path, chapters=[])
        self.assertGreater(len(result.beats), 0)

    def test_sub_second_chunk_skipped(self):
        """A chapter under 1 second is skipped (beat tracker can't say much)."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats
        from videoflow.chapters import Chapter

        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(20.0, 120.0),
            )
            # First chapter is sub-second, should be skipped silently
            chapters = [
                Chapter(at_ms=0, end_ms=500),
                Chapter(at_ms=500, end_ms=20_000),
            ]
            result = analyze_beats(path, chapters=chapters)
        # All beats should land in the second chunk (>= 500ms)
        self.assertTrue(all(b >= 500 for b in result.beats))


class TestMakeTimeChunks(unittest.TestCase):
    """The synthetic time-window splitter — pure function, no librosa."""

    def test_short_file_is_one_chunk(self):
        from videoflow.audio import _make_time_chunks
        self.assertEqual(_make_time_chunks(100_000, 180), [(0, 100_000)])

    def test_long_file_splits_into_equal_windows(self):
        from videoflow.audio import _make_time_chunks
        # 9 min / 3-min windows = 3 clean chunks
        self.assertEqual(
            _make_time_chunks(540_000, 180),
            [(0, 180_000), (180_000, 360_000), (360_000, 540_000)],
        )

    def test_short_trailing_remainder_merges_into_last(self):
        from videoflow.audio import _make_time_chunks
        # 9m40s: the 40s tail (< half a 3-min chunk) folds into chunk 3
        self.assertEqual(
            _make_time_chunks(580_000, 180),
            [(0, 180_000), (180_000, 360_000), (360_000, 580_000)],
        )

    def test_substantial_trailing_remainder_kept_separate(self):
        from videoflow.audio import _make_time_chunks
        # 11 min: the 2-min tail (> half a 3-min chunk) stays its own window
        self.assertEqual(
            _make_time_chunks(660_000, 180),
            [(0, 180_000), (180_000, 360_000), (360_000, 540_000), (540_000, 660_000)],
        )

    def test_chunks_cover_the_whole_timeline(self):
        from videoflow.audio import _make_time_chunks
        spans = _make_time_chunks(1_234_567, 180)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 1_234_567)
        for (a, b), (c, _d) in zip(spans, spans[1:]):
            self.assertEqual(b, c)  # contiguous, no gaps/overlaps


class TestAnalyzeBeatsChunked(unittest.TestCase):
    """Time-windowed chunking (chunk_secs) — the chapters machinery applied
    to material with no semantic chapter list.
    """

    _click_audio = staticmethod(TestAnalyzeBeatsWithChapters._click_audio)
    _write_wav = TestAnalyzeBeatsWithChapters._write_wav

    def test_chunk_secs_splits_and_stitches(self):
        """A long file with chunk_secs is split into windows; beats stitch ascending."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats

        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(30.0, 120.0),
            )
            result = analyze_beats(path, chunk_secs=10)
        self.assertGreater(len(result.beats), 0)
        self.assertEqual(result.beats, sorted(result.beats))
        # beats should land across all three 10s windows
        self.assertTrue(any(b < 10_000 for b in result.beats))
        self.assertTrue(any(10_000 <= b < 20_000 for b in result.beats))
        self.assertTrue(any(b >= 20_000 for b in result.beats))

    def test_chunk_secs_progress_labels_say_chunk(self):
        """progress_callback fires 'Analyzing chunk X/Y…' per window."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats

        labels: list[str] = []
        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(30.0, 120.0),
            )
            analyze_beats(path, chunk_secs=10, progress_callback=labels.append)
        # Filter to the per-window announcement line (the chunk label also
        # prefixes each sub-step message now, so count announcements).
        announce = [lb for lb in labels if lb.startswith("Analyzing chunk")]
        self.assertEqual(len(announce), 3)
        self.assertIn("1/3", announce[0])
        self.assertIn("3/3", announce[2])

    def test_chunk_secs_per_chunk_energy_normalization(self):
        """Each window self-normalizes its energy to 1.0 (quiet intro not crushed)."""
        from tempfile import TemporaryDirectory
        import numpy as np
        from videoflow.audio import analyze_beats

        with TemporaryDirectory() as td:
            quiet = self._click_audio(15.0, 120.0) * 0.05
            loud = self._click_audio(15.0, 120.0) * 0.6
            y = np.concatenate([quiet, loud])
            path = self._write_wav(Path(td) / "mixed.wav", y)
            result = analyze_beats(path, chunk_secs=15)
        first = [e for b, e in zip(result.beats, result.energy) if b < 15_000]
        second = [e for b, e in zip(result.beats, result.energy) if b >= 15_000]
        self.assertGreater(max(first), 0.95, "quiet window self-normalizes")
        self.assertGreater(max(second), 0.95, "loud window self-normalizes")

    def test_chunk_secs_short_file_no_split(self):
        """A file shorter than one window is analysed whole — no 'chunk' labels."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats

        labels: list[str] = []
        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(20.0, 120.0),
            )
            result = analyze_beats(path, chunk_secs=180, progress_callback=labels.append)
        self.assertGreater(len(result.beats), 0)
        self.assertEqual([lb for lb in labels if "chunk" in lb.lower()], [])

    def test_chunk_secs_true_uses_default_width(self):
        """chunk_secs=True works and (on a short file) keeps a single window."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats

        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(20.0, 120.0),
            )
            result = analyze_beats(path, chunk_secs=True)
        self.assertGreater(len(result.beats), 0)

    def test_chunk_secs_runs_hpss_per_chunk(self):
        """source='percussive' + chunk_secs → HPSS runs once PER window, not once
        over the whole file (the fix for a monolithic multi-hour HPSS hang)."""
        import numpy as np
        mock_lib = _make_librosa_mock(duration_s=30.0, beat_count=40)
        mock_lib.effects.hpss.side_effect = lambda y: (y, y)
        with patch("videoflow.audio._librosa", mock_lib), \
             patch("videoflow.audio._np", np), \
             patch.object(Path, "exists", return_value=True):
            from videoflow.audio import analyze_beats
            analyze_beats("track.mp3", source="percussive", chunk_secs=10)
        # 30s / 10s = 3 windows → 3 HPSS calls
        self.assertEqual(mock_lib.effects.hpss.call_count, 3)

    def test_chapters_win_over_chunk_secs(self):
        """When both are given, chapters take precedence (they carry meaning)."""
        from tempfile import TemporaryDirectory
        from videoflow.audio import analyze_beats
        from videoflow.chapters import Chapter

        labels: list[str] = []
        with TemporaryDirectory() as td:
            path = self._write_wav(
                Path(td) / "click.wav", self._click_audio(40.0, 120.0),
            )
            analyze_beats(
                path,
                chapters=[Chapter(at_ms=0, end_ms=20_000),
                          Chapter(at_ms=20_000, end_ms=40_000)],
                chunk_secs=5,
                progress_callback=labels.append,
            )
        # chapter spans drive analysis, not chunk time-windows: the per-span
        # announcement says "chapter", and no "chunk" split label appears
        # anywhere (including the now-prefixed sub-step messages).
        announce = [lb for lb in labels if lb.startswith("Analyzing chapter")]
        self.assertEqual(len(announce), 2)
        self.assertEqual([lb for lb in labels if "chunk" in lb.lower()], [])


class TestTempoOctaveCorrection(unittest.TestCase):
    """The tempo-octave guard — folds a doubled tempo back to the felt pulse
    without touching genuinely fast tracks. Pure function, no librosa.

    Validated on Rhythms of Desire: free-run 235 BPM generated 7.7 actions/s
    vs the gold's 4.2; folding to 117 BPM landed at 3.9.
    """

    def setUp(self):
        from videoflow.audio import _correct_tempo_octave
        self.fold = _correct_tempo_octave

    def _beats(self, n, interval=250):
        return [i * interval for i in range(n)]

    def test_doubled_tempo_is_halved(self):
        beats = self._beats(8, 250)
        energy = [1.0] * 8
        bpm, b, e = self.fold(235.0, beats, energy)
        self.assertAlmostEqual(bpm, 117.5)
        self.assertEqual(len(b), 4)
        self.assertEqual(len(e), len(b))       # stays index-aligned

    def test_normal_tempo_untouched(self):
        beats = self._beats(8, 500)
        energy = [1.0] * 8
        bpm, b, e = self.fold(120.0, beats, energy)
        self.assertEqual(bpm, 120.0)
        self.assertEqual(b, beats)
        self.assertEqual(e, energy)

    def test_at_ceiling_untouched(self):
        # 160 <= 165 ceiling → left alone (a genuinely fast track).
        bpm, _b, _e = self.fold(160.0, self._beats(8, 375), [1.0] * 8)
        self.assertEqual(bpm, 160.0)

    def test_keeps_higher_energy_phase(self):
        # Even-indexed beats carry the energy → odd phase must drop.
        beats = self._beats(8, 250)
        energy = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        _bpm, b, _e = self.fold(235.0, beats, energy)
        self.assertEqual(b, beats[0::2])       # kept the strong on-beats

    def test_keeps_odd_phase_when_stronger(self):
        beats = self._beats(8, 250)
        energy = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        _bpm, b, _e = self.fold(235.0, beats, energy)
        self.assertEqual(b, beats[1::2])

    def test_quadrupled_tempo_folds_twice(self):
        beats = self._beats(16, 125)
        energy = [1.0] * 16
        bpm, b, _e = self.fold(360.0, beats, energy)
        self.assertAlmostEqual(bpm, 90.0)      # 360 → 180 → 90
        self.assertEqual(len(b), 4)

    def test_empty_energy_defaults_to_even_phase(self):
        beats = self._beats(8, 250)
        bpm, b, e = self.fold(235.0, beats, [])
        self.assertEqual(b, beats[0::2])
        self.assertEqual(e, [])

    def test_too_few_beats_not_folded(self):
        # Even a high BPM is left alone if there's nothing to fold.
        bpm, b, _e = self.fold(235.0, [0, 250], [1.0, 1.0])
        self.assertEqual(bpm, 235.0)
        self.assertEqual(b, [0, 250])


if __name__ == "__main__":
    unittest.main()
