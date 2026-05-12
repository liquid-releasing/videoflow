"""Tests for per-chapter funscript generation — _slice_beat_map,
generate_from_beats_per_chapter, and the export_funscript metadata block."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from videoflow.audio import AudioBeatMap
from videoflow.generate import (
    GenerateError,
    _slice_beat_map,
    export_funscript,
    generate_from_beats_per_chapter,
)


def _synth_beat_map(duration_ms: int = 60_000) -> AudioBeatMap:
    """60-second synthetic beat-map: beat every 500ms (120 BPM), downbeat
    every 4 beats, four 15s phrases. Enough structure for the per-chapter
    generator to produce real curves."""
    beats = list(range(0, duration_ms, 500))
    return AudioBeatMap(
        bpm=120.0,
        beats=beats,
        downbeats=list(range(0, duration_ms, 2000)),
        phrases=[
            (0, 15000),
            (15000, 30000),
            (30000, 45000),
            (45000, 60000),
        ],
        energy=[0.5] * len(beats),
        duration_ms=duration_ms,
    )


class TestSliceBeatMap(unittest.TestCase):

    def setUp(self):
        self.bm = _synth_beat_map()

    def test_first_chapter_window(self):
        sliced = _slice_beat_map(self.bm, 0, 20_000)
        self.assertEqual(sliced.beats[0], 0)
        self.assertLess(sliced.beats[-1], 20_000)
        self.assertEqual(sliced.duration_ms, 20_000)
        self.assertEqual(len(sliced.beats), 40)  # 20s / 500ms

    def test_middle_chapter_keeps_absolute_timestamps(self):
        sliced = _slice_beat_map(self.bm, 20_000, 40_000)
        self.assertEqual(sliced.beats[0], 20_000)
        self.assertLess(sliced.beats[-1], 40_000)

    def test_last_chapter_to_end(self):
        sliced = _slice_beat_map(self.bm, 40_000, 60_000)
        self.assertEqual(sliced.beats[0], 40_000)
        # 60_000 is exclusive; last beat is at 59_500
        self.assertEqual(sliced.beats[-1], 59_500)

    def test_empty_window_outside_track(self):
        sliced = _slice_beat_map(self.bm, 100_000, 110_000)
        self.assertEqual(sliced.beats, [])
        self.assertEqual(sliced.duration_ms, 10_000)

    def test_energy_aligns_with_beats(self):
        sliced = _slice_beat_map(self.bm, 0, 10_000)
        self.assertEqual(len(sliced.energy), len(sliced.beats))

    def test_phrase_overlapping_window_is_kept(self):
        # Phrase 0 spans 0..15000; window 10000..20000 partially overlaps.
        sliced = _slice_beat_map(self.bm, 10_000, 20_000)
        self.assertIn((0, 15000), sliced.phrases)


class TestGenerateFromBeatsPerChapter(unittest.TestCase):

    def setUp(self):
        self.bm = _synth_beat_map()
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _basic_3_chapter_call(self, **overrides):
        chapters = [
            {"at_ms": 0, "end_ms": 20000},
            {"at_ms": 20000, "end_ms": 40000},
            {"at_ms": 40000, "end_ms": 60000},
        ]
        recipes = [
            {"source": "percussive", "stroke_density": "half", "tone": "flat", "emphasize_beats": False},
            {"source": "percussive", "stroke_density": "1", "tone": "rise", "emphasize_beats": True},
            {"source": "percussive", "stroke_density": "2", "tone": "fall", "emphasize_beats": False},
        ]
        out = self.tmp / "out.funscript"
        return generate_from_beats_per_chapter(
            self.bm, chapters, recipes, out,
            **overrides,
        )

    def test_produces_funscript_with_actions_across_all_chapters(self):
        out = self._basic_3_chapter_call()
        data = json.loads(out.read_text())
        actions = data["actions"]
        self.assertGreater(len(actions), 0)
        # At least one action lands in each chapter window.
        per_chapter = [
            sum(1 for a in actions if 0 <= a["at"] < 20_000),
            sum(1 for a in actions if 20_000 <= a["at"] < 40_000),
            sum(1 for a in actions if 40_000 <= a["at"] < 60_000),
        ]
        for n in per_chapter:
            self.assertGreater(n, 0, f"chapter empty: per-chapter counts = {per_chapter}")

    def test_strict_length_mismatch_raises(self):
        chapters = [{"at_ms": 0, "end_ms": 30000}, {"at_ms": 30000, "end_ms": 60000}]
        recipes = [{"source": "percussive", "stroke_density": "half", "tone": "flat"}]
        out = self.tmp / "out.funscript"
        with self.assertRaises(GenerateError) as ctx:
            generate_from_beats_per_chapter(self.bm, chapters, recipes, out)
        self.assertIn("1 recipes", str(ctx.exception))
        self.assertIn("2 chapters", str(ctx.exception))

    def test_actions_sorted_and_unique_at_times(self):
        out = self._basic_3_chapter_call()
        data = json.loads(out.read_text())
        ats = [a["at"] for a in data["actions"]]
        self.assertEqual(ats, sorted(ats), "actions must be time-sorted")
        self.assertEqual(len(ats), len(set(ats)), "no duplicate timestamps")

    def test_generated_from_metadata_embedded(self):
        provenance = {
            "tool": "videoflow",
            "tool_version": "test-1.0",
            "source": {"path": "fake.mp4", "duration_ms": 60000, "partial_md5": "abc"},
            "chapters": [{"at_ms": 0, "end_ms": 60000}],
            "recipes": [{"source": "percussive"}],
        }
        out = self._basic_3_chapter_call(generated_from=provenance)
        data = json.loads(out.read_text())
        self.assertEqual(data["metadata"]["generated_from"], provenance)

    def test_progress_callback_invoked_per_chapter(self):
        labels = []
        self._basic_3_chapter_call(progress_callback=labels.append)
        chapter_labels = [l for l in labels if l.startswith("Generating chapter")]
        self.assertEqual(len(chapter_labels), 3)
        self.assertIn("1/3", chapter_labels[0])
        self.assertIn("3/3", chapter_labels[2])

    def test_end_ms_none_extends_to_duration(self):
        chapters = [{"at_ms": 0, "end_ms": None}]
        recipes = [{"source": "percussive", "stroke_density": "half", "tone": "flat"}]
        out = self.tmp / "out.funscript"
        generate_from_beats_per_chapter(self.bm, chapters, recipes, out)
        data = json.loads(out.read_text())
        # Should cover the full track
        self.assertGreater(max(a["at"] for a in data["actions"]), 50_000)


class TestExportFunscriptMetadata(unittest.TestCase):

    def test_generated_from_field_lands_under_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "x.funscript"
            export_funscript(
                [(0, 50), (1000, 80), (2000, 20)],
                out,
                generated_from={"tool": "videoflow", "source": {"path": "x.mp4"}},
            )
            data = json.loads(out.read_text())
            self.assertEqual(data["metadata"]["generated_from"]["tool"], "videoflow")
            self.assertNotIn("title", data["metadata"])

    def test_title_and_generated_from_coexist(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "x.funscript"
            export_funscript(
                [(0, 50), (1000, 80)],
                out,
                title="my track",
                generated_from={"tool": "videoflow"},
            )
            data = json.loads(out.read_text())
            self.assertEqual(data["metadata"]["title"], "my track")
            self.assertEqual(data["metadata"]["generated_from"]["tool"], "videoflow")

    def test_no_metadata_block_when_neither_provided(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "x.funscript"
            export_funscript([(0, 50), (1000, 80)], out)
            data = json.loads(out.read_text())
            self.assertNotIn("metadata", data)


if __name__ == "__main__":
    unittest.main()
