"""Tests for videoflow.structural — audio-structure auto-chapter detection."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from videoflow.chapters import Chapter
from videoflow.structural import (
    AutoChapterError,
    _build_energy,
    _classify_content,
    _merge_micro_chapters,
    _percentile,
    _segment_boundaries,
    _silence_breakpoints,
    _snap_to_silence,
    auto_chapter,
)


def _save_wav(path: Path, y: np.ndarray, sr: int) -> Path:
    """Persist a mono float32 buffer as a WAV using soundfile."""
    sf.write(str(path), y.astype(np.float32), sr, subtype="FLOAT")
    return path


def _sine(duration_s: float, freq: float, sr: int, amp: float = 0.6) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return amp * np.sin(2.0 * np.pi * freq * t)


def _click_track(duration_s: float, sr: int, bpm: float = 120.0) -> np.ndarray:
    """Sharp impulses at the given BPM — strong percussive signature."""
    y = np.zeros(int(duration_s * sr), dtype=np.float32)
    interval = int(sr * 60.0 / bpm)
    for i in range(0, len(y), interval):
        # 5ms percussive burst
        burst_len = min(int(sr * 0.005), len(y) - i)
        y[i:i + burst_len] = np.linspace(1.0, 0.0, burst_len)
    return y


def _pink_noise(duration_s: float, sr: int, amp: float = 0.05) -> np.ndarray:
    """Low-amplitude broadband noise — ambient-like."""
    rng = np.random.default_rng(seed=0)
    return amp * rng.standard_normal(int(duration_s * sr))


# ---------------------------------------------------------------------------
# auto_chapter — error paths and short-file behaviour
# ---------------------------------------------------------------------------

class TestAutoChapterErrors(unittest.TestCase):

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            auto_chapter("/nope/not/a/real/file.mp4", write_sidecar=False)

    def test_librosa_missing_raises_autochaptererror(self):
        with TemporaryDirectory() as td:
            path = _save_wav(Path(td) / "x.wav", _sine(1.0, 440, 22050), 22050)
            with patch("videoflow.structural._librosa", None):
                with self.assertRaises(AutoChapterError):
                    auto_chapter(path, write_sidecar=False)


class TestAutoChapterShortFile(unittest.TestCase):
    """Files below the chunk threshold return a single whole-file chapter."""

    def test_under_threshold_returns_single_chapter(self):
        sr = 22050
        with TemporaryDirectory() as td:
            # 30 seconds — well under the 8-min minimum chunk threshold
            path = _save_wav(Path(td) / "short.wav", _sine(30.0, 440, sr), sr)
            chapters = auto_chapter(path, write_sidecar=False)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].at_ms, 0)
            self.assertGreater(chapters[0].end_ms, 25_000)
            self.assertLess(chapters[0].end_ms, 35_000)

    def test_progress_callback_invoked_on_short_file(self):
        sr = 22050
        labels: list[str] = []
        with TemporaryDirectory() as td:
            path = _save_wav(Path(td) / "short.wav", _sine(15.0, 440, sr), sr)
            auto_chapter(path, write_sidecar=False, progress_callback=labels.append)
        self.assertTrue(any("Loading audio" in lb for lb in labels))
        self.assertTrue(any("too short" in lb.lower() for lb in labels))

    def test_progress_callback_errors_swallowed(self):
        sr = 22050
        with TemporaryDirectory() as td:
            path = _save_wav(Path(td) / "short.wav", _sine(10.0, 440, sr), sr)

            def boom(_label: str) -> None:
                raise RuntimeError("kaboom")

            chapters = auto_chapter(
                path, write_sidecar=False, progress_callback=boom,
            )
            self.assertEqual(len(chapters), 1)


# ---------------------------------------------------------------------------
# Algorithm helpers — synthetic-input unit tests
# ---------------------------------------------------------------------------

class TestSilenceBreakpoints(unittest.TestCase):

    def test_finds_known_silence_region(self):
        sr = 22050
        # 5s tone, 3s silence, 5s tone
        y = np.concatenate([
            _sine(5.0, 440, sr),
            np.zeros(int(3.0 * sr), dtype=np.float32),
            _sine(5.0, 440, sr),
        ])
        centers = _silence_breakpoints(y, sr)
        self.assertEqual(len(centers), 1)
        # Silence center ~6.5s (5 + 1.5)
        self.assertAlmostEqual(centers[0], 6.5, delta=0.3)

    def test_short_silence_below_min_duration_ignored(self):
        sr = 22050
        # Only 0.5s silence — below the 1.5s minimum
        y = np.concatenate([
            _sine(2.0, 440, sr),
            np.zeros(int(0.5 * sr), dtype=np.float32),
            _sine(2.0, 440, sr),
        ])
        centers = _silence_breakpoints(y, sr)
        self.assertEqual(centers, [])

    def test_no_silence_returns_empty(self):
        sr = 22050
        y = _sine(10.0, 440, sr)
        self.assertEqual(_silence_breakpoints(y, sr), [])


class TestSegmentBoundaries(unittest.TestCase):

    def test_returns_interior_only_excludes_zero(self):
        sr = 22050
        y = np.concatenate([_click_track(15.0, sr), _pink_noise(15.0, sr)])
        # 30s file targeting 10s chunks → ~3 segments → 2 interior boundaries
        boundaries = _segment_boundaries(
            y, sr, target_seconds=10.0, duration_s=30.0,
        )
        self.assertGreater(len(boundaries), 0)
        for b in boundaries:
            self.assertGreater(b, 0.0)


class TestSnapToSilence(unittest.TestCase):

    def test_snaps_to_nearest_silence_within_radius(self):
        result = _snap_to_silence([10.5], [10.0, 30.0], radius_s=2.0)
        self.assertEqual(result, [10.0])

    def test_no_snap_outside_radius(self):
        result = _snap_to_silence([10.5], [50.0], radius_s=2.0)
        self.assertEqual(result, [10.5])

    def test_no_silence_returns_boundaries_unchanged(self):
        result = _snap_to_silence([10.0, 20.0, 30.0], [], radius_s=2.0)
        self.assertEqual(result, [10.0, 20.0, 30.0])


class TestMergeMicroChapters(unittest.TestCase):

    def test_drops_interior_cut_that_makes_micro_chapter(self):
        # Cuts at 0, 100, 110, 300 — the 100-110 chapter is 10s (a sliver).
        # Algorithm drops cut 110 so the sliver merges into the next
        # chapter (100-300), preserving the full-sized previous chapter
        # (0-100).
        result = _merge_micro_chapters([0.0, 100.0, 110.0, 300.0], min_duration_s=30.0)
        self.assertEqual(result, [0.0, 100.0, 300.0])

    def test_drops_short_tail_into_previous_chapter(self):
        # Final 295-300 is only 5s
        result = _merge_micro_chapters([0.0, 100.0, 295.0, 300.0], min_duration_s=30.0)
        self.assertEqual(result, [0.0, 100.0, 300.0])

    def test_preserves_endpoints(self):
        result = _merge_micro_chapters([0.0, 50.0, 100.0], min_duration_s=200.0)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[-1], 100.0)

    def test_no_merge_when_all_chapters_meet_minimum(self):
        result = _merge_micro_chapters([0.0, 100.0, 200.0, 300.0], min_duration_s=50.0)
        self.assertEqual(result, [0.0, 100.0, 200.0, 300.0])


class TestClassifyContent(unittest.TestCase):

    def test_click_track_classifies_with_evidence(self):
        sr = 22050
        y = _click_track(20.0, sr, bpm=120.0)
        content_type, confidence, evidence = _classify_content(y, sr)
        # Strong percussive signature should land in music or mixed
        self.assertIn(content_type, {"music", "mixed"})
        self.assertGreater(confidence, 0.5)
        self.assertTrue(any("percussive" in e or "rms" in e or "flux" in e for e in evidence))

    def test_low_amplitude_noise_classifies_ambient_or_mixed(self):
        sr = 22050
        y = _pink_noise(20.0, sr, amp=0.02)
        content_type, _conf, _evidence = _classify_content(y, sr)
        self.assertIn(content_type, {"ambient", "mixed"})

    def test_sub_second_returns_unclassified(self):
        sr = 22050
        y = _sine(0.5, 440, sr)
        content_type, confidence, _ = _classify_content(y, sr)
        self.assertEqual(content_type, "")
        self.assertEqual(confidence, 0.0)


# Sidecar persistence is now videoflow.sidecar.write_sidecar's responsibility;
# see tests/test_sidecar.py for read / write / merge coverage.


# ---------------------------------------------------------------------------
# Energy payload builder (phrase classification + chapter-index lookup
# moved to videoflow.phrases — see tests/test_phrases.py)
# ---------------------------------------------------------------------------

class TestBuildEnergy(unittest.TestCase):

    class _FakeBeatMap:
        def __init__(self, beats, energy, duration_ms, downbeats=None):
            self.beats = beats
            self.energy = energy
            self.duration_ms = duration_ms
            self.downbeats = list(downbeats) if downbeats is not None else []

    def test_emits_percentiles_and_beat_map(self):
        bm = self._FakeBeatMap(
            beats=[0, 500, 1000, 1500, 2000],
            energy=[0.1, 0.2, 0.5, 0.8, 0.9],
            duration_ms=2500,
        )
        chapters = [Chapter(at_ms=0, end_ms=2500, content_type="music")]
        energy = _build_energy(bm, chapters)
        self.assertIn("percentiles", energy)
        for key in ("p5", "p25", "p50", "p75", "p95"):
            self.assertIn(key, energy["percentiles"])
        self.assertEqual(energy["beat_map"]["times_ms"], [0, 500, 1000, 1500, 2000])
        self.assertEqual(len(energy["beat_map"]["strengths"]), 5)
        self.assertEqual(energy["beat_map"]["is_downbeat"], [False] * 5)

    def test_beat_map_marks_downbeats(self):
        bm = self._FakeBeatMap(
            beats=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
            energy=[0.5] * 8,
            duration_ms=4000,
            downbeats=[0, 2000],  # every 4th beat in 4/4
        )
        chapters = [Chapter(at_ms=0, end_ms=4000, content_type="music")]
        energy = _build_energy(bm, chapters)
        self.assertEqual(
            energy["beat_map"]["is_downbeat"],
            [True, False, False, False, True, False, False, False],
        )

    def test_per_chapter_block_has_bpm_and_content_type(self):
        bm = self._FakeBeatMap(
            beats=[0, 500, 1000, 1500],  # 4 beats over 2 seconds = 120 BPM
            energy=[0.1, 0.2, 0.3, 0.4],
            duration_ms=2000,
        )
        chapters = [Chapter(at_ms=0, end_ms=2000, content_type="music")]
        energy = _build_energy(bm, chapters)
        self.assertIn("0", energy["per_chapter"])
        per = energy["per_chapter"]["0"]
        self.assertIn("percentiles", per)
        self.assertIsNotNone(per["bpm"])
        self.assertEqual(per["content_type"], "music")

    def test_per_chapter_skips_empty_chapters(self):
        bm = self._FakeBeatMap(
            beats=[100, 200],
            energy=[0.5, 0.5],
            duration_ms=2000,
        )
        chapters = [
            Chapter(at_ms=0, end_ms=300),       # has beats
            Chapter(at_ms=1500, end_ms=2000),   # empty range
        ]
        energy = _build_energy(bm, chapters)
        self.assertIn("0", energy["per_chapter"])
        self.assertNotIn("1", energy["per_chapter"])


class TestPercentile(unittest.TestCase):

    def test_empty_returns_zero(self):
        self.assertEqual(_percentile([], 0.5), 0.0)

    def test_p50_of_uniform_is_middle(self):
        self.assertAlmostEqual(_percentile([0.0, 0.5, 1.0], 0.5), 0.5)

    def test_p5_picks_lowest_quintile(self):
        sorted_vals = [float(i) for i in range(100)]
        self.assertLess(_percentile(sorted_vals, 0.05), 10)


# ---------------------------------------------------------------------------
# Integration — auto_chapter writes phrases + energy into the sidecar
# ---------------------------------------------------------------------------

class TestAutoChapterEmitsExpandedSidecar(unittest.TestCase):

    def test_short_file_sidecar_carries_phrases_and_energy_blocks(self):
        """auto_chapter on a real audio file lands chapters + phrases + energy.

        Uses a click track so analyze_beats / classify_modes have real
        rhythmic content to work on (unlike a uniform sine wave).
        """
        sr = 22050
        with TemporaryDirectory() as td:
            path = _save_wav(
                Path(td) / "track.wav",
                _click_track(30.0, sr, bpm=120.0),
                sr,
            )
            auto_chapter(path, write_sidecar=True)

            sidecar = path.with_name("track.chapters.json")
            self.assertTrue(sidecar.exists())
            doc = json.loads(sidecar.read_text(encoding="utf-8"))

            # Schema upgraded to v2 by the new write path
            self.assertEqual(doc["version"], "2.0")
            self.assertEqual(doc["schema"], "audio-structure")
            self.assertGreaterEqual(len(doc["chapters"]), 1)

            # phrases is always present (may be empty if classify_modes
            # found nothing), energy is present when beats were analyzed.
            self.assertIn("phrases", doc)
            self.assertIn("energy", doc)
            self.assertIn("percentiles", doc["energy"])
            self.assertIn("beat_map", doc["energy"])

            # provenance entry recorded
            self.assertGreaterEqual(len(doc["provenance"]), 1)
            self.assertEqual(
                doc["provenance"][-1]["writer"], "videoflow.structural",
            )


if __name__ == "__main__":
    unittest.main()
