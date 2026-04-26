"""Tests for videoflow.generate — funscript generation from beat maps."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from videoflow.audio import AudioBeatMap
from videoflow.generate import (
    GenerateError,
    beats_to_curve,
    classify_modes,
    export_funscript,
    generate_from_beats,
    shape_curve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beat_map(
    bpm: float = 120.0,
    beats: list[int] | None = None,
    energy: list[float] | None = None,
    phrases: list[tuple[int, int]] | None = None,
    duration_ms: int = 10_000,
) -> AudioBeatMap:
    """Build a minimal AudioBeatMap for testing."""
    if beats is None:
        beats = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    if energy is None:
        energy = [0.8] * len(beats)
    if phrases is None:
        phrases = [(0, duration_ms)]
    return AudioBeatMap(
        bpm=bpm,
        beats=beats,
        downbeats=beats[::4],
        phrases=phrases,
        energy=energy,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# beats_to_curve
# ---------------------------------------------------------------------------

class TestBeatsToCurve(unittest.TestCase):

    def test_empty_beat_map_returns_empty(self):
        bm = _beat_map(beats=[], energy=[])
        self.assertEqual(beats_to_curve(bm), [])

    def test_returns_one_point_per_beat(self):
        bm = _beat_map()
        curve = beats_to_curve(bm)
        self.assertEqual(len(curve), len(bm.beats))

    def test_even_beats_are_peaks(self):
        bm = _beat_map(beats=[0, 500, 1000], energy=[1.0, 1.0, 1.0])
        curve = beats_to_curve(bm, low=10, high=90)
        self.assertGreater(curve[0][1], 10)  # beat 0 = peak
        self.assertGreater(curve[2][1], 10)  # beat 2 = peak

    def test_odd_beats_are_troughs(self):
        bm = _beat_map(beats=[0, 500, 1000], energy=[1.0, 1.0, 1.0])
        curve = beats_to_curve(bm, low=10, high=90)
        self.assertEqual(curve[1][1], 10)  # beat 1 = trough

    def test_alternating_high_low(self):
        bm = _beat_map(beats=list(range(0, 4000, 500)), energy=[1.0] * 8)
        curve = beats_to_curve(bm, low=10, high=90)
        for i, (_, pos) in enumerate(curve):
            if i % 2 == 0:
                self.assertGreater(pos, 10)
            else:
                self.assertEqual(pos, 10)

    def test_high_energy_produces_high_peak(self):
        bm = _beat_map(beats=[0, 500], energy=[1.0, 0.0])
        curve = beats_to_curve(bm, low=10, high=90)
        self.assertEqual(curve[0][1], 90)  # max energy = max peak

    def test_low_energy_still_produces_min_stroke(self):
        bm = _beat_map(beats=[0, 500], energy=[0.0, 0.0])
        curve = beats_to_curve(bm, low=10, high=90, min_stroke=20)
        self.assertEqual(curve[0][1], 30)  # low + min_stroke

    def test_positions_clamped_to_100(self):
        bm = _beat_map(beats=[0], energy=[1.0])
        curve = beats_to_curve(bm, low=10, high=100)
        self.assertLessEqual(curve[0][1], 100)

    def test_timestamps_match_beats(self):
        beats = [0, 484, 968, 1452]
        bm = _beat_map(beats=beats, energy=[0.8] * 4)
        curve = beats_to_curve(bm)
        self.assertEqual([t for t, _ in curve], beats)


# ---------------------------------------------------------------------------
# classify_modes
# ---------------------------------------------------------------------------

class TestClassifyModes(unittest.TestCase):

    def test_returns_one_mode_per_phrase(self):
        phrases = [(0, 4000), (4000, 8000)]
        bm = _beat_map(phrases=phrases)
        modes = classify_modes(bm)
        self.assertEqual(len(modes), 2)

    def test_break_on_silent_phrase(self):
        beats = [0, 500, 1000, 1500]
        energy = [0.05, 0.05, 0.08, 0.06]
        bm = _beat_map(beats=beats, energy=energy, phrases=[(0, 2000)])
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "break")

    def test_tease_on_quiet_phrase(self):
        beats = [0, 500, 1000, 1500]
        energy = [0.20, 0.22, 0.18, 0.25]
        bm = _beat_map(beats=beats, energy=energy, phrases=[(0, 2000)])
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "tease")

    def test_edging_on_rising_energy(self):
        # Second half energy much higher than first half
        beats = list(range(0, 4000, 500))
        energy = [0.3, 0.3, 0.3, 0.3, 0.6, 0.65, 0.7, 0.75]
        bm = _beat_map(beats=beats, energy=energy, phrases=[(0, 4000)])
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "edging")

    def test_fast_on_high_bpm(self):
        bm = _beat_map(bpm=150.0)
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "fast")

    def test_slow_on_low_bpm(self):
        bm = _beat_map(bpm=60.0)
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "slow")

    def test_steady_on_normal_energy_and_bpm(self):
        beats = list(range(0, 4000, 500))
        energy = [0.7] * 8
        bm = _beat_map(bpm=120.0, beats=beats, energy=energy, phrases=[(0, 4000)])
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "steady")

    def test_empty_phrase_returns_break(self):
        # Phrase range has no beats in it
        bm = _beat_map(beats=[5000, 5500], energy=[0.8, 0.8], phrases=[(0, 1000)])
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "break")

    def test_mode_timestamps_match_phrases(self):
        phrases = [(0, 3000), (3000, 6000)]
        beats = list(range(0, 6000, 500))
        energy = [0.7] * len(beats)
        bm = _beat_map(beats=beats, energy=energy, phrases=phrases)
        modes = classify_modes(bm)
        self.assertEqual(modes[0][0], 0)
        self.assertEqual(modes[0][1], 3000)
        self.assertEqual(modes[1][0], 3000)
        self.assertEqual(modes[1][1], 6000)


# ---------------------------------------------------------------------------
# shape_curve
# ---------------------------------------------------------------------------

class TestShapeCurve(unittest.TestCase):

    def _simple_curve(self):
        # 4 beats: peak, trough, peak, trough
        return [(0, 80), (500, 10), (1000, 80), (1500, 10)]

    def test_troughs_are_preserved(self):
        curve = self._simple_curve()
        modes = [(0, 2000, "steady")]
        shaped = shape_curve(curve, modes, low=10)
        troughs = [pos for t, pos in shaped if t in (500, 1500)]
        self.assertTrue(all(p == 10 for p in troughs))

    def test_break_compresses_peaks(self):
        curve = [(0, 80), (500, 10)]
        modes = [(0, 1000, "break")]
        shaped = shape_curve(curve, modes, low=10)
        peak = shaped[0][1]
        self.assertLess(peak, 30)  # highly compressed

    def test_tease_compresses_peaks_moderately(self):
        curve = [(0, 80), (500, 10)]
        modes = [(0, 1000, "tease")]
        shaped = shape_curve(curve, modes, low=10)
        # tease should be between break and steady
        break_shaped = shape_curve(curve, [(0, 1000, "break")], low=10)
        steady_shaped = shape_curve(curve, [(0, 1000, "steady")], low=10)
        self.assertGreater(shaped[0][1], break_shaped[0][1])
        self.assertLess(shaped[0][1], steady_shaped[0][1])

    def test_slow_produces_near_full_amplitude(self):
        curve = [(0, 90), (500, 10)]
        modes = [(0, 1000, "slow")]
        shaped = shape_curve(curve, modes, low=10)
        self.assertGreaterEqual(shaped[0][1], 80)

    def test_edging_builds_amplitude_over_phrase(self):
        # Four peaks spread across phrase — last should be bigger than first
        curve = [(0, 80), (250, 10), (500, 80), (750, 10),
                 (1000, 80), (1250, 10), (1500, 80), (1750, 10)]
        modes = [(0, 2000, "edging")]
        shaped = shape_curve(curve, modes, low=10)
        peaks = [pos for t, pos in shaped if pos > 10]
        self.assertGreater(peaks[-1], peaks[0])

    def test_positions_clamped_to_100(self):
        curve = [(0, 100)]
        modes = [(0, 1000, "slow")]
        shaped = shape_curve(curve, modes, low=10)
        self.assertLessEqual(shaped[0][1], 100)

    def test_empty_curve_returns_empty(self):
        self.assertEqual(shape_curve([], [(0, 1000, "steady")]), [])

    def test_empty_modes_returns_curve_unchanged(self):
        curve = [(0, 80), (500, 10)]
        shaped = shape_curve(curve, [], low=10)
        self.assertEqual(shaped, curve)


# ---------------------------------------------------------------------------
# export_funscript
# ---------------------------------------------------------------------------

class TestExportFunscript(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_valid_json(self):
        curve = [(0, 90), (500, 10), (1000, 85)]
        path = export_funscript(curve, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        self.assertEqual(data["version"], "1.0")
        self.assertIn("actions", data)

    def test_actions_sorted_by_time(self):
        curve = [(1000, 10), (0, 90), (500, 50)]
        path = export_funscript(curve, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        times = [a["at"] for a in data["actions"]]
        self.assertEqual(times, sorted(times))

    def test_duplicate_timestamps_deduplicated(self):
        curve = [(0, 90), (0, 50), (500, 10)]
        path = export_funscript(curve, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        times = [a["at"] for a in data["actions"]]
        self.assertEqual(len(times), len(set(times)))

    def test_positions_clamped(self):
        curve = [(0, 150), (500, -10)]
        path = export_funscript(curve, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        for action in data["actions"]:
            self.assertGreaterEqual(action["pos"], 0)
            self.assertLessEqual(action["pos"], 100)

    def test_title_stored_in_metadata(self):
        curve = [(0, 90), (500, 10)]
        path = export_funscript(curve, self.tmp / "out.funscript", title="Test Track")
        data = json.loads(path.read_text())
        self.assertEqual(data["metadata"]["title"], "Test Track")

    def test_no_title_no_metadata_key(self):
        curve = [(0, 90), (500, 10)]
        path = export_funscript(curve, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        self.assertNotIn("metadata", data)

    def test_empty_curve_raises(self):
        with self.assertRaises(GenerateError):
            export_funscript([], self.tmp / "out.funscript")

    def test_creates_parent_dirs(self):
        path = self.tmp / "sub" / "deep" / "out.funscript"
        export_funscript([(0, 90), (500, 10)], path)
        self.assertTrue(path.exists())

    def test_inverted_false(self):
        curve = [(0, 90), (500, 10)]
        path = export_funscript(curve, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        self.assertFalse(data["inverted"])

    def test_range_field(self):
        curve = [(0, 90), (500, 10)]
        path = export_funscript(curve, self.tmp / "out.funscript", range_=80)
        data = json.loads(path.read_text())
        self.assertEqual(data["range"], 80)


# ---------------------------------------------------------------------------
# generate_from_beats (end-to-end)
# ---------------------------------------------------------------------------

class TestGenerateFromBeats(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_produces_funscript_file(self):
        bm = _beat_map()
        path = generate_from_beats(bm, self.tmp / "out.funscript")
        self.assertTrue(path.exists())

    def test_output_is_valid_funscript(self):
        bm = _beat_map()
        path = generate_from_beats(bm, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        self.assertIn("actions", data)
        self.assertIn("version", data)
        self.assertTrue(len(data["actions"]) > 0)

    def test_action_count_matches_beats(self):
        beats = list(range(0, 8000, 500))
        bm = _beat_map(beats=beats, energy=[0.8] * len(beats))
        path = generate_from_beats(bm, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        self.assertEqual(len(data["actions"]), len(beats))

    def test_title_passed_through(self):
        bm = _beat_map()
        path = generate_from_beats(bm, self.tmp / "out.funscript", title="My Track")
        data = json.loads(path.read_text())
        self.assertEqual(data["metadata"]["title"], "My Track")

    def test_all_positions_in_range(self):
        bm = _beat_map()
        path = generate_from_beats(bm, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        for action in data["actions"]:
            self.assertGreaterEqual(action["pos"], 0)
            self.assertLessEqual(action["pos"], 100)

    def test_actions_sorted(self):
        bm = _beat_map()
        path = generate_from_beats(bm, self.tmp / "out.funscript")
        data = json.loads(path.read_text())
        times = [a["at"] for a in data["actions"]]
        self.assertEqual(times, sorted(times))

    def test_high_bpm_mode_is_fast(self):
        beats = list(range(0, 4000, 400))
        bm = _beat_map(bpm=150.0, beats=beats, energy=[0.8] * len(beats))
        modes = classify_modes(bm)
        self.assertTrue(any(m == "fast" for _, _, m in modes))

    def test_silent_track_produces_break_mode(self):
        beats = [0, 500, 1000, 1500]
        bm = _beat_map(beats=beats, energy=[0.05] * 4, phrases=[(0, 2000)])
        modes = classify_modes(bm)
        self.assertEqual(modes[0][2], "break")


if __name__ == "__main__":
    unittest.main()
