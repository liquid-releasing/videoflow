"""Tests for videoflow.chapters — chapter resolver across sources."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from videoflow.chapters import (
    Chapter,
    ChapterError,
    load_chapters,
    read_analysis_chapters,
    read_mp4_chapters,
    read_sidecar_chapters,
)


class TestChapterDataclass(unittest.TestCase):

    def test_default_fields(self):
        c = Chapter(at_ms=1000)
        self.assertEqual(c.at_ms, 1000)
        self.assertIsNone(c.end_ms)
        self.assertEqual(c.name, "")
        self.assertEqual(c.intent, "")

    def test_to_dict_omits_none_end_ms(self):
        c = Chapter(at_ms=0, name="intro", intent="intro")
        self.assertNotIn("end_ms", c.to_dict())

    def test_to_dict_includes_end_ms_when_set(self):
        c = Chapter(at_ms=0, end_ms=90_000, name="intro", intent="intro")
        self.assertEqual(c.to_dict()["end_ms"], 90_000)

    def test_from_dict_authored_shape(self):
        c = Chapter.from_dict({"at_ms": 90_000, "name": "build", "intent": "build"})
        self.assertEqual(c.at_ms, 90_000)
        self.assertEqual(c.intent, "build")

    def test_from_dict_proposal_shape(self):
        """analysis-schema chapter_proposals[] uses start_ms / intent_proposal."""
        c = Chapter.from_dict({
            "start_ms": 90_000,
            "end_ms": 480_000,
            "intent_proposal": "build",
            "confidence": 0.82,
        })
        self.assertEqual(c.at_ms, 90_000)
        self.assertEqual(c.end_ms, 480_000)
        self.assertEqual(c.intent, "build")

    def test_from_dict_legacy_at_field(self):
        """funscript metadata.chapters historically uses 'at' not 'at_ms'."""
        c = Chapter.from_dict({"at": 0, "name": "intro", "intent": "intro"})
        self.assertEqual(c.at_ms, 0)

    def test_from_dict_missing_start_raises(self):
        with self.assertRaises(ChapterError):
            Chapter.from_dict({"name": "no time"})

    def test_round_trip(self):
        c = Chapter(at_ms=42_000, end_ms=120_000, name="edge", intent="edge")
        clone = Chapter.from_dict(c.to_dict())
        self.assertEqual(c, clone)


# ---------------------------------------------------------------------------
# Sidecar reader
# ---------------------------------------------------------------------------

class TestSidecarReader(unittest.TestCase):

    def test_no_sidecar_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            self.assertIsNone(read_sidecar_chapters(media))

    def test_sidecar_dot_chapters_json_alongside_stem(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            sidecar = Path(td) / "track.chapters.json"
            sidecar.write_text(json.dumps({
                "version": "1.0",
                "chapters": [
                    {"at_ms": 0, "name": "intro", "intent": "intro"},
                    {"at_ms": 90_000, "intent": "build"},
                ],
            }))
            chapters = read_sidecar_chapters(media)
            self.assertEqual(len(chapters), 2)
            self.assertEqual(chapters[0].intent, "intro")
            self.assertEqual(chapters[1].at_ms, 90_000)

    def test_sidecar_bare_list_accepted(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            sidecar = Path(td) / "track.chapters.json"
            sidecar.write_text(json.dumps([
                {"at_ms": 0, "intent": "intro"},
            ]))
            chapters = read_sidecar_chapters(media)
            self.assertEqual(len(chapters), 1)

    def test_sidecar_invalid_json_raises(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            (Path(td) / "track.chapters.json").write_text("not json")
            with self.assertRaises(ChapterError):
                read_sidecar_chapters(media)

    def test_sidecar_chapters_field_not_a_list_raises(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            (Path(td) / "track.chapters.json").write_text(
                json.dumps({"chapters": "nope"})
            )
            with self.assertRaises(ChapterError):
                read_sidecar_chapters(media)


# ---------------------------------------------------------------------------
# Analysis-JSON reader
# ---------------------------------------------------------------------------

class TestAnalysisReader(unittest.TestCase):

    def test_no_analysis_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            self.assertIsNone(read_analysis_chapters(media))

    def test_analysis_metadata_chapters_takes_priority(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            (Path(td) / "track.analysis.json").write_text(json.dumps({
                "metadata": {
                    "chapters": [
                        {"at_ms": 0, "intent": "intro"},
                    ],
                },
                "chapter_proposals": [
                    {"start_ms": 90_000, "intent_proposal": "build"},
                ],
            }))
            chapters = read_analysis_chapters(media)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].intent, "intro")

    def test_analysis_chapter_proposals_used_when_no_metadata_chapters(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            (Path(td) / "track.analysis.json").write_text(json.dumps({
                "chapter_proposals": [
                    {"start_ms": 0, "end_ms": 90_000, "intent_proposal": "intro"},
                    {"start_ms": 90_000, "end_ms": 480_000, "intent_proposal": "build"},
                ],
            }))
            chapters = read_analysis_chapters(media)
            self.assertEqual(len(chapters), 2)
            self.assertEqual(chapters[0].intent, "intro")
            self.assertEqual(chapters[1].at_ms, 90_000)
            self.assertEqual(chapters[1].end_ms, 480_000)

    def test_analysis_with_no_chapters_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            (Path(td) / "track.analysis.json").write_text(json.dumps({
                "metadata": {"title": "no chapters here"}
            }))
            self.assertIsNone(read_analysis_chapters(media))

    def test_analysis_invalid_json_raises(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            (Path(td) / "track.analysis.json").write_text("not json")
            with self.assertRaises(ChapterError):
                read_analysis_chapters(media)


# ---------------------------------------------------------------------------
# mp4 reader (mocked ffprobe)
# ---------------------------------------------------------------------------

class TestMp4Reader(unittest.TestCase):

    def test_non_video_extension_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp3"
            media.write_bytes(b"")
            self.assertIsNone(read_mp4_chapters(media))

    def test_missing_file_returns_none(self):
        self.assertIsNone(read_mp4_chapters("/no/such/path.mp4"))

    def test_ffprobe_unavailable_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            with patch(
                "videoflow.chapters.subprocess.run",
                side_effect=FileNotFoundError(),
            ):
                self.assertIsNone(read_mp4_chapters(media))

    def test_ffprobe_no_chapters_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")

            class _R:
                stdout = json.dumps({"chapters": []})

            with patch(
                "videoflow.chapters.subprocess.run", return_value=_R()
            ):
                self.assertIsNone(read_mp4_chapters(media))

    def test_ffprobe_chapters_parsed(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")

            class _R:
                stdout = json.dumps({
                    "chapters": [
                        {
                            "start_time": "0.000000",
                            "end_time": "90.000000",
                            "tags": {"title": "intro"},
                        },
                        {
                            "start_time": "90.000000",
                            "end_time": "480.000000",
                            "tags": {"title": "build 1"},
                        },
                    ]
                })

            with patch(
                "videoflow.chapters.subprocess.run", return_value=_R()
            ):
                chapters = read_mp4_chapters(media)
            self.assertEqual(len(chapters), 2)
            self.assertEqual(chapters[0].at_ms, 0)
            self.assertEqual(chapters[0].end_ms, 90_000)
            self.assertEqual(chapters[0].name, "intro")
            self.assertEqual(chapters[1].at_ms, 90_000)


# ---------------------------------------------------------------------------
# load_chapters resolver — priority order
# ---------------------------------------------------------------------------

class TestLoadChaptersResolver(unittest.TestCase):

    def test_missing_media_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_chapters("/definitely/nope.mp4")

    def test_no_sources_returns_none(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp3"
            media.write_bytes(b"")
            # mp3 → no mp4 chapters; no sidecar; no analysis
            self.assertIsNone(load_chapters(media))

    def test_mp4_chapters_take_priority_over_sidecar(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            # sidecar would say "from sidecar"
            (Path(td) / "track.chapters.json").write_text(json.dumps([
                {"at_ms": 0, "intent": "from_sidecar"},
            ]))

            class _R:
                stdout = json.dumps({
                    "chapters": [
                        {
                            "start_time": "0.000000",
                            "end_time": "90.000000",
                            "tags": {"title": "from mp4"},
                        }
                    ]
                })

            with patch(
                "videoflow.chapters.subprocess.run", return_value=_R()
            ):
                chapters = load_chapters(media)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].name, "from mp4")

    def test_sidecar_used_when_no_mp4_chapters(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp3"  # not video → mp4 reader skips
            media.write_bytes(b"")
            (Path(td) / "track.chapters.json").write_text(json.dumps([
                {"at_ms": 0, "intent": "from_sidecar"},
            ]))
            chapters = load_chapters(media)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].intent, "from_sidecar")

    def test_analysis_used_when_no_mp4_or_sidecar(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp3"
            media.write_bytes(b"")
            (Path(td) / "track.analysis.json").write_text(json.dumps({
                "chapter_proposals": [
                    {"start_ms": 0, "end_ms": 90_000, "intent_proposal": "intro"}
                ],
            }))
            chapters = load_chapters(media)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].intent, "intro")

    def test_sidecar_takes_priority_over_analysis(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp3"
            media.write_bytes(b"")
            (Path(td) / "track.chapters.json").write_text(json.dumps([
                {"at_ms": 0, "intent": "from_sidecar"},
            ]))
            (Path(td) / "track.analysis.json").write_text(json.dumps({
                "chapter_proposals": [
                    {"start_ms": 0, "intent_proposal": "from_analysis"}
                ],
            }))
            chapters = load_chapters(media)
            self.assertEqual(chapters[0].intent, "from_sidecar")


if __name__ == "__main__":
    unittest.main()
