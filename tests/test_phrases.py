"""Tests for videoflow.phrases — Phrase record + audio/funscript classifiers."""

from __future__ import annotations

import unittest

from videoflow.audio import AudioBeatMap
from videoflow.chapters import Chapter
from videoflow.phrases import (
    Phrase,
    _chapter_index_at,
    _segment_actions_into_phrases,
    classify_modes,
    classify_phrases,
    classify_phrases_from_funscript,
)


# ---------------------------------------------------------------------------
# Phrase round-trip
# ---------------------------------------------------------------------------

class TestPhraseRecord(unittest.TestCase):

    def test_to_dict_omits_defaults(self):
        ph = Phrase(chapter_idx=0, at_ms=0, end_ms=1000)
        d = ph.to_dict()
        self.assertEqual(d["chapter_idx"], 0)
        self.assertEqual(d["at_ms"], 0)
        self.assertEqual(d["end_ms"], 1000)
        self.assertNotIn("mode", d)
        self.assertNotIn("confidence", d)
        self.assertNotIn("source", d)
        self.assertTrue(d["auto_generated"])

    def test_to_dict_includes_populated_optional_fields(self):
        ph = Phrase(
            chapter_idx=2, at_ms=1000, end_ms=2000,
            mode="tease", source="audio",
            evidence=["beat_density"],
        )
        d = ph.to_dict()
        self.assertEqual(d["mode"], "tease")
        self.assertEqual(d["source"], "audio")
        self.assertEqual(d["evidence"], ["beat_density"])

    def test_round_trip_via_dict(self):
        original = Phrase(
            chapter_idx=1, at_ms=100, end_ms=900,
            mode="steady", confidence=0.7, source="audio",
            intent="build", tone="build",
            evidence=["mfcc", "beat_density"],
            auto_generated=False,
        )
        round_tripped = Phrase.from_dict(original.to_dict())
        self.assertEqual(round_tripped, original)

    def test_from_dict_raises_on_missing_required(self):
        with self.assertRaises(ValueError):
            Phrase.from_dict({"chapter_idx": 0, "at_ms": 100})  # no end_ms


# ---------------------------------------------------------------------------
# _chapter_index_at
# ---------------------------------------------------------------------------

class TestChapterIndexAt(unittest.TestCase):

    def setUp(self):
        self.chapters = [
            Chapter(at_ms=0, end_ms=1000),
            Chapter(at_ms=1000, end_ms=2000),
            Chapter(at_ms=2000, end_ms=3000),
        ]

    def test_finds_chapter_for_time_in_range(self):
        self.assertEqual(_chapter_index_at(500, self.chapters), 0)
        self.assertEqual(_chapter_index_at(1500, self.chapters), 1)
        self.assertEqual(_chapter_index_at(2500, self.chapters), 2)

    def test_boundary_inclusive_on_start(self):
        self.assertEqual(_chapter_index_at(1000, self.chapters), 1)

    def test_beyond_last_falls_to_last(self):
        self.assertEqual(_chapter_index_at(99_999, self.chapters), 2)

    def test_handles_none_end_ms(self):
        self.assertEqual(_chapter_index_at(99_999, [Chapter(at_ms=0)]), 0)


# ---------------------------------------------------------------------------
# Audio-driven classifier — preserves classify_modes back-compat
# ---------------------------------------------------------------------------

def _build_beat_map(
    *,
    bpm: float,
    phrases: list[tuple[int, int]],
    beats: list[int],
    energy: list[float],
    duration_ms: int,
) -> AudioBeatMap:
    return AudioBeatMap(
        bpm=bpm,
        beats=beats,
        downbeats=beats[::4] if beats else [],
        phrases=phrases,
        energy=energy,
        duration_ms=duration_ms,
    )


class TestClassifyPhrases(unittest.TestCase):

    def test_returns_phrase_records_with_source_audio(self):
        bm = _build_beat_map(
            bpm=120.0,
            phrases=[(0, 4_000)],
            beats=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
            energy=[0.8] * 8,
            duration_ms=4_000,
        )
        phrases = classify_phrases(bm)
        self.assertEqual(len(phrases), 1)
        self.assertIsInstance(phrases[0], Phrase)
        self.assertEqual(phrases[0].source, "audio")
        self.assertTrue(phrases[0].auto_generated)
        # Steady is the default for high-energy/medium-bpm content.
        self.assertEqual(phrases[0].mode, "steady")

    def test_chapter_idx_attached_when_chapters_supplied(self):
        bm = _build_beat_map(
            bpm=120.0,
            phrases=[(0, 1000), (1000, 2000)],
            beats=[100, 500, 1100, 1500],
            energy=[0.7, 0.7, 0.7, 0.7],
            duration_ms=2000,
        )
        chapters = [Chapter(at_ms=0, end_ms=1000), Chapter(at_ms=1000, end_ms=2000)]
        phrases = classify_phrases(bm, chapters=chapters)
        self.assertEqual(phrases[0].chapter_idx, 0)
        self.assertEqual(phrases[1].chapter_idx, 1)

    def test_low_energy_classified_break(self):
        bm = _build_beat_map(
            bpm=120.0,
            phrases=[(0, 4_000)],
            beats=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
            energy=[0.05] * 8,
            duration_ms=4_000,
        )
        phrases = classify_phrases(bm)
        self.assertEqual(phrases[0].mode, "break")


class TestClassifyModesBackCompat(unittest.TestCase):

    def test_classify_modes_returns_tuples(self):
        """The legacy tuple API survives the lift."""
        bm = _build_beat_map(
            bpm=120.0,
            phrases=[(0, 4_000)],
            beats=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
            energy=[0.7] * 8,
            duration_ms=4_000,
        )
        result = classify_modes(bm)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 3)
        start_ms, end_ms, mode = result[0]
        self.assertEqual(start_ms, 0)
        self.assertEqual(end_ms, 4_000)
        self.assertIn(mode, ["steady", "fast", "slow", "tease", "edging", "break"])

    def test_re_export_from_videoflow_generate(self):
        from videoflow.generate import classify_modes as cm_via_generate
        from videoflow.phrases import classify_modes as cm_direct
        self.assertIs(cm_via_generate, cm_direct)


# ---------------------------------------------------------------------------
# Funscript-driven classifier
# ---------------------------------------------------------------------------

class TestSegmentActionsIntoPhrases(unittest.TestCase):

    def test_groups_sixteen_actions_per_phrase(self):
        actions = [{"at": i * 100, "pos": 50} for i in range(40)]
        phrases = _segment_actions_into_phrases(actions)
        self.assertEqual(len(phrases), 3)  # 16 + 16 + 8
        self.assertEqual(phrases[0][0], 0)
        self.assertEqual(phrases[1][0], 1600)
        self.assertEqual(phrases[2][0], 3200)

    def test_empty_actions_produce_no_phrases(self):
        self.assertEqual(_segment_actions_into_phrases([]), [])

    def test_last_phrase_runs_to_final_action(self):
        actions = [{"at": i * 100, "pos": 50} for i in range(20)]
        phrases = _segment_actions_into_phrases(actions)
        # Final phrase ends at last action timestamp + 1
        self.assertEqual(phrases[-1][1], 19 * 100 + 1)


class TestClassifyPhrasesFromFunscript(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        self.assertEqual(classify_phrases_from_funscript([]), [])

    def test_static_strokes_classify_break(self):
        """Pos held near a single value → very low amplitude → break."""
        actions = [{"at": i * 100, "pos": 50 + (i % 2)} for i in range(40)]
        phrases = classify_phrases_from_funscript(
            actions,
            phrase_boundaries=[(0, 4_000)],
        )
        self.assertEqual(phrases[0].mode, "break")
        self.assertEqual(phrases[0].source, "funscript")

    def test_full_range_strokes_classify_steady_or_fast(self):
        # 20 actions over 4 seconds (5 actions/sec) alternating 0 and 100.
        actions = [
            {"at": i * 200, "pos": 0 if i % 2 else 100}
            for i in range(20)
        ]
        phrases = classify_phrases_from_funscript(
            actions,
            phrase_boundaries=[(0, 4_000)],
        )
        self.assertEqual(phrases[0].source, "funscript")
        # 5 actions/sec ≥ 4 → fast
        self.assertEqual(phrases[0].mode, "fast")

    def test_low_density_classifies_slow(self):
        # 4 full-range strokes over 8 seconds → 0.5 actions/sec → slow
        actions = [
            {"at": 0, "pos": 0},
            {"at": 2_000, "pos": 100},
            {"at": 4_000, "pos": 0},
            {"at": 6_000, "pos": 100},
        ]
        phrases = classify_phrases_from_funscript(
            actions,
            phrase_boundaries=[(0, 8_000)],
        )
        self.assertEqual(phrases[0].mode, "slow")

    def test_rising_amplitude_classifies_edging(self):
        # First half: small moves around centre. Second half: full range.
        first_half = [
            {"at": i * 100, "pos": 50 + (5 if i % 2 else -5)} for i in range(16)
        ]
        second_half = [
            {"at": 1_600 + i * 100, "pos": 0 if i % 2 else 100}
            for i in range(16)
        ]
        actions = first_half + second_half
        phrases = classify_phrases_from_funscript(
            actions,
            phrase_boundaries=[(0, 3_200)],
        )
        self.assertEqual(phrases[0].mode, "edging")

    def test_chapter_idx_attached_when_chapters_supplied(self):
        actions = [{"at": i * 100, "pos": 50} for i in range(40)]
        chapters = [Chapter(at_ms=0, end_ms=2_000), Chapter(at_ms=2_000, end_ms=4_000)]
        phrases = classify_phrases_from_funscript(
            actions, chapters=chapters,
            phrase_boundaries=[(0, 1_000), (2_500, 3_500)],
        )
        self.assertEqual(phrases[0].chapter_idx, 0)
        self.assertEqual(phrases[1].chapter_idx, 1)

    def test_default_phrase_boundaries_segment_into_groups(self):
        actions = [{"at": i * 100, "pos": 50} for i in range(48)]
        phrases = classify_phrases_from_funscript(actions)
        # 48 actions / 16 per phrase = 3 phrases
        self.assertEqual(len(phrases), 3)


if __name__ == "__main__":
    unittest.main()
