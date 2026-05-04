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
    _classify_content,
    _maybe_write_sidecar,
    _merge_micro_chapters,
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


# ---------------------------------------------------------------------------
# Sidecar persistence
# ---------------------------------------------------------------------------

class TestSidecarPersistence(unittest.TestCase):

    def _make_chapters(self) -> list[Chapter]:
        return [
            Chapter(
                at_ms=0, end_ms=300_000,
                content_type="music", confidence=0.85,
                evidence=["rms_variance", "percussive_ratio"],
            ),
            Chapter(
                at_ms=300_000, end_ms=600_000,
                content_type="ambient", confidence=0.72,
                evidence=["rms_variance"],
            ),
        ]

    def test_writes_sidecar_when_absent(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.touch()
            path = _maybe_write_sidecar(
                media, self._make_chapters(), target_minutes=5.5, force=False,
            )
            self.assertIsNotNone(path)
            self.assertEqual(path.name, "track.chapters.json")
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(doc["auto_generated"])
            self.assertEqual(doc["version"], "1.0")
            self.assertEqual(len(doc["chapters"]), 2)
            self.assertEqual(doc["chapters"][0]["content_type"], "music")
            self.assertEqual(doc["generated_by"]["target_minutes"], 5.5)

    def test_skips_when_sidecar_exists_and_not_forced(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.touch()
            sidecar = media.with_name("track.chapters.json")
            sidecar.write_text('{"version":"1.0","chapters":[]}', encoding="utf-8")
            result = _maybe_write_sidecar(
                media, self._make_chapters(), target_minutes=5.5, force=False,
            )
            self.assertIsNone(result)
            # Original content preserved
            self.assertEqual(
                json.loads(sidecar.read_text())["chapters"], [],
            )

    def test_force_overwrites_auto_generated_sidecar(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.touch()
            sidecar = media.with_name("track.chapters.json")
            sidecar.write_text(
                json.dumps({"version": "1.0", "auto_generated": True, "chapters": []}),
                encoding="utf-8",
            )
            result = _maybe_write_sidecar(
                media, self._make_chapters(), target_minutes=5.5, force=True,
            )
            self.assertIsNotNone(result)
            doc = json.loads(sidecar.read_text())
            self.assertEqual(len(doc["chapters"]), 2)

    def test_force_refuses_to_clobber_user_edited_sidecar(self):
        """auto_generated:false marks user-edited; force=True still respects it."""
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.touch()
            sidecar = media.with_name("track.chapters.json")
            user_edited = {
                "version": "1.0",
                "auto_generated": False,
                "chapters": [{"at_ms": 0, "name": "user-named"}],
            }
            sidecar.write_text(json.dumps(user_edited), encoding="utf-8")
            result = _maybe_write_sidecar(
                media, self._make_chapters(), target_minutes=5.5, force=True,
            )
            self.assertIsNone(result)
            # User edit preserved
            self.assertEqual(
                json.loads(sidecar.read_text())["chapters"][0]["name"],
                "user-named",
            )


if __name__ == "__main__":
    unittest.main()
