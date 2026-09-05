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
    _format_ffmetadata1,
    _format_youtube_timestamp,
    embed_in_mp4,
    format_youtube_description,
    load_chapters,
    read_analysis_chapters,
    read_mp4_chapters,
    read_sidecar_chapters,
    write_chapters_sidecar,
)


def _sidecar(td) -> Path:
    """Where track.mp4's chapters sidecar actually lives.

    Sidecars moved into the project's hidden `.forge` folder rather than
    sitting next to the media. These tests constructed the old flat path
    inline in nine places, so all of them broke on a documented, deliberate
    change. Ask the module for the path (and make the folder, since this is
    used as a write target) instead of freezing a copy of the rule.
    """
    from videoflow.sidecar import sidecar_path_for
    path = sidecar_path_for(Path(td) / "track.mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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

    def test_tone_round_trips(self):
        # User-chosen tone must survive to_dict/from_dict so the Chapters tab
        # rehydrates the accepted tone on reopen instead of resetting to the
        # analyzer suggestion.
        c = Chapter(at_ms=0, end_ms=1_000, name="x", intent="build", tone="edge")
        d = c.to_dict()
        self.assertEqual(d["tone"], "edge")
        self.assertEqual(Chapter.from_dict(d).tone, "edge")

    def test_to_dict_omits_empty_tone(self):
        # Authored / un-toned chapters stay compact — no empty "tone" key.
        c = Chapter(at_ms=0, end_ms=1_000, name="x")
        self.assertNotIn("tone", c.to_dict())

    def test_from_dict_defaults_tone_when_absent(self):
        # Legacy sidecars (pre-tone) parse with an empty tone, not an error.
        self.assertEqual(Chapter.from_dict({"at_ms": 0, "name": "y"}).tone, "")


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
            sidecar = _sidecar(td)
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
            sidecar = _sidecar(td)
            sidecar.write_text(json.dumps([
                {"at_ms": 0, "intent": "intro"},
            ]))
            chapters = read_sidecar_chapters(media)
            self.assertEqual(len(chapters), 1)

    def test_sidecar_invalid_json_raises(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            _sidecar(td).write_text("not json")
            with self.assertRaises(ChapterError):
                read_sidecar_chapters(media)

    def test_sidecar_chapters_field_not_a_list_raises(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            _sidecar(td).write_text(
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

    def test_sidecar_takes_priority_over_mp4(self):
        """Priority flipped 2026-05-14 — hand-built sidecar wins over embedded.

        Pre-flip behaviour was the reverse (embedded won); see the
        docstring of load_chapters for the rationale.
        """
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            # sidecar says "from sidecar" — should win
            _sidecar(td).write_text(json.dumps([
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

            # ffprobe shouldn't even be called when the sidecar resolves first,
            # but stub it anyway so an accidental call doesn't make the test
            # fall back to the system binary.
            with patch(
                "videoflow.chapters.subprocess.run", return_value=_R()
            ):
                chapters = load_chapters(media)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].intent, "from_sidecar")

    def test_mp4_used_when_no_sidecar(self):
        """With no sidecar, embedded chapters are the next fallback."""
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")

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
            _sidecar(td).write_text(json.dumps([
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
            _sidecar(td).write_text(json.dumps([
                {"at_ms": 0, "intent": "from_sidecar"},
            ]))
            (Path(td) / "track.analysis.json").write_text(json.dumps({
                "chapter_proposals": [
                    {"start_ms": 0, "intent_proposal": "from_analysis"}
                ],
            }))
            chapters = load_chapters(media)
            self.assertEqual(chapters[0].intent, "from_sidecar")


# ---------------------------------------------------------------------------
# write_chapters_sidecar
# ---------------------------------------------------------------------------

class TestWriteChaptersSidecar(unittest.TestCase):

    def test_writes_sidecar_with_latch(self):
        """Hand-authored chapters land with auto_generated: false (LATCH)."""
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            chapters = [
                Chapter(at_ms=0, end_ms=90_000, name="intro", intent="intro"),
                Chapter(at_ms=90_000, end_ms=480_000, name="build", intent="build"),
            ]
            written = write_chapters_sidecar(media, chapters, writer="test")
            self.assertTrue(written.exists())
            doc = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(len(doc["chapters"]), 2)
            for ch in doc["chapters"]:
                self.assertFalse(ch["auto_generated"])

    def test_round_trip_via_load_chapters(self):
        """write → load returns the same chapters (sidecar-first priority)."""
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp3"  # mp3 so the mp4 reader skips
            media.write_bytes(b"")
            chapters = [
                Chapter(at_ms=0, name="intro", intent="intro"),
                Chapter(at_ms=90_000, name="build", intent="build"),
            ]
            write_chapters_sidecar(media, chapters, writer="test")
            roundtrip = load_chapters(media)
            self.assertEqual(len(roundtrip), 2)
            self.assertEqual(roundtrip[0].name, "intro")
            self.assertEqual(roundtrip[1].at_ms, 90_000)

    def test_provenance_records_writer(self):
        """The sidecar's provenance trail names the writer arg."""
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            write_chapters_sidecar(
                media,
                [Chapter(at_ms=0, name="intro", intent="intro")],
                writer="videoflow.cli",
                writer_version="0.0.7-test",
            )
            doc = json.loads(_sidecar(td).read_text())
            prov = doc.get("provenance") or []
            self.assertTrue(any(p.get("writer") == "videoflow.cli" for p in prov))


# ---------------------------------------------------------------------------
# embed_in_mp4 / _format_ffmetadata1
# ---------------------------------------------------------------------------

class TestFFMetadata1Formatter(unittest.TestCase):

    def test_empty_chapters_produces_header_only(self):
        self.assertEqual(_format_ffmetadata1([]), ";FFMETADATA1\n")

    def test_chapters_render_with_timebase_and_title(self):
        text = _format_ffmetadata1([
            Chapter(at_ms=0, end_ms=90_000, name="intro", intent="intro"),
            Chapter(at_ms=90_000, end_ms=480_000, name="build", intent="build"),
        ])
        self.assertIn(";FFMETADATA1", text)
        self.assertIn("TIMEBASE=1/1000", text)
        self.assertIn("START=0", text)
        self.assertIn("END=90000", text)
        self.assertIn("title=intro", text)
        self.assertIn("START=90000", text)
        self.assertIn("END=480000", text)
        self.assertIn("title=build", text)

    def test_missing_end_ms_filled_from_next_start(self):
        text = _format_ffmetadata1([
            Chapter(at_ms=0, name="intro"),
            Chapter(at_ms=90_000, name="build"),
        ])
        # First chapter has no end_ms; should be filled from the second's start.
        self.assertIn("START=0", text)
        self.assertIn("END=90000", text)

    def test_missing_end_ms_on_final_chapter_uses_sentinel(self):
        text = _format_ffmetadata1([Chapter(at_ms=0, name="only")])
        self.assertIn("START=0", text)
        # 1ms-after-start sentinel keeps ffmpeg happy.
        self.assertIn("END=1", text)

    def test_title_falls_back_to_intent(self):
        text = _format_ffmetadata1([Chapter(at_ms=0, end_ms=1000, intent="build")])
        self.assertIn("title=build", text)

    def test_title_escapes_special_chars(self):
        """FFMETADATA1 spec requires \\, =, ;, #, \\n escaped with backslash."""
        text = _format_ffmetadata1([
            Chapter(at_ms=0, end_ms=1000, name="A=B;C#D"),
        ])
        self.assertIn(r"title=A\=B\;C\#D", text)


class TestEmbedInMp4(unittest.TestCase):

    def test_rejects_in_place_overwrite(self):
        with TemporaryDirectory() as td:
            media = Path(td) / "track.mp4"
            media.write_bytes(b"")
            with self.assertRaises(ChapterError):
                embed_in_mp4(media, media, [])

    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            embed_in_mp4("/no/such/track.mp4", "/tmp/out.mp4", [])

    def test_invokes_ffmpeg_with_expected_args(self):
        """The ffmpeg invocation passes -map_chapters 1 and -codec copy."""
        with TemporaryDirectory() as td:
            src = Path(td) / "in.mp4"
            src.write_bytes(b"")
            dst = Path(td) / "out.mp4"

            class _R:
                returncode = 0
                stderr = ""

            with patch(
                "videoflow.chapters.subprocess.run", return_value=_R()
            ) as run:
                embed_in_mp4(src, dst, [Chapter(at_ms=0, end_ms=1000, name="x")])
            args = run.call_args[0][0]
            self.assertIn("-map_chapters", args)
            self.assertIn("-codec", args)
            # ensure stream-copy was requested
            codec_idx = args.index("-codec")
            self.assertEqual(args[codec_idx + 1], "copy")

    def test_ffmpeg_nonzero_exit_raises(self):
        with TemporaryDirectory() as td:
            src = Path(td) / "in.mp4"
            src.write_bytes(b"")
            dst = Path(td) / "out.mp4"

            class _R:
                returncode = 1
                stderr = "synthetic ffmpeg failure"

            with patch(
                "videoflow.chapters.subprocess.run", return_value=_R()
            ):
                with self.assertRaises(ChapterError) as ctx:
                    embed_in_mp4(src, dst, [])
            self.assertIn("synthetic ffmpeg failure", str(ctx.exception))


# ---------------------------------------------------------------------------
# format_youtube_description / _format_youtube_timestamp
# ---------------------------------------------------------------------------

class TestYoutubeTimestamp(unittest.TestCase):

    def test_under_one_hour_uses_m_ss(self):
        self.assertEqual(_format_youtube_timestamp(0, force_hours=False), "0:00")
        self.assertEqual(_format_youtube_timestamp(83_000, force_hours=False), "1:23")
        self.assertEqual(_format_youtube_timestamp(3_599_000, force_hours=False), "59:59")

    def test_at_or_above_one_hour_uses_h_mm_ss(self):
        self.assertEqual(_format_youtube_timestamp(3_600_000, force_hours=False), "1:00:00")
        self.assertEqual(_format_youtube_timestamp(5_025_000, force_hours=False), "1:23:45")

    def test_force_hours_pads_minutes(self):
        """When the block contains any 1h+ chapter, all rows render H:MM:SS."""
        self.assertEqual(_format_youtube_timestamp(0, force_hours=True), "0:00:00")
        self.assertEqual(_format_youtube_timestamp(83_000, force_hours=True), "0:01:23")


class TestFormatYoutubeDescription(unittest.TestCase):

    def test_valid_short_block(self):
        text = format_youtube_description([
            Chapter(at_ms=0, name="Intro"),
            Chapter(at_ms=83_000, name="Topic 1"),
            Chapter(at_ms=300_000, name="Topic 2"),
        ])
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "0:00 Intro")
        self.assertEqual(lines[1], "1:23 Topic 1")
        self.assertEqual(lines[2], "5:00 Topic 2")
        # trailing newline included for safe concat
        self.assertTrue(text.endswith("\n"))

    def test_validates_first_chapter_at_zero(self):
        with self.assertRaises(ChapterError) as ctx:
            format_youtube_description([
                Chapter(at_ms=1000, name="Late start"),
                Chapter(at_ms=20_000, name="Two"),
                Chapter(at_ms=40_000, name="Three"),
            ])
        self.assertIn("0:00", str(ctx.exception))

    def test_validates_minimum_three_chapters(self):
        with self.assertRaises(ChapterError) as ctx:
            format_youtube_description([
                Chapter(at_ms=0, name="A"),
                Chapter(at_ms=20_000, name="B"),
            ])
        self.assertIn("at least 3", str(ctx.exception))

    def test_validates_ten_second_minimum(self):
        with self.assertRaises(ChapterError) as ctx:
            format_youtube_description([
                Chapter(at_ms=0, name="Intro"),
                Chapter(at_ms=5_000, name="Too soon"),  # only 5s gap
                Chapter(at_ms=20_000, name="Three"),
            ])
        self.assertIn("10 seconds", str(ctx.exception))

    def test_hour_chapter_forces_h_mm_ss_throughout(self):
        text = format_youtube_description([
            Chapter(at_ms=0, name="Intro"),
            Chapter(at_ms=1_800_000, name="Middle"),
            Chapter(at_ms=3_700_000, name="Bonus"),
        ])
        lines = text.strip().splitlines()
        # All three should be H:MM:SS because the last one is past 1h.
        self.assertEqual(lines[0], "0:00:00 Intro")
        self.assertEqual(lines[1], "0:30:00 Middle")
        self.assertEqual(lines[2], "1:01:40 Bonus")

    def test_title_falls_back_to_intent(self):
        text = format_youtube_description([
            Chapter(at_ms=0, intent="intro"),
            Chapter(at_ms=20_000, intent="build"),
            Chapter(at_ms=40_000, intent="climax"),
        ])
        lines = text.strip().splitlines()
        self.assertIn("intro", lines[0])
        self.assertIn("build", lines[1])
        self.assertIn("climax", lines[2])

    def test_title_falls_back_to_chapter_n(self):
        text = format_youtube_description([
            Chapter(at_ms=0),
            Chapter(at_ms=20_000),
            Chapter(at_ms=40_000),
        ])
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "0:00 Chapter 1")
        self.assertEqual(lines[1], "0:20 Chapter 2")
        self.assertEqual(lines[2], "0:40 Chapter 3")

    def test_handles_unsorted_input(self):
        """Caller may pass chapters in any order; formatter sorts."""
        text = format_youtube_description([
            Chapter(at_ms=40_000, name="Last"),
            Chapter(at_ms=0, name="First"),
            Chapter(at_ms=20_000, name="Middle"),
        ])
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "0:00 First")
        self.assertEqual(lines[1], "0:20 Middle")
        self.assertEqual(lines[2], "0:40 Last")


if __name__ == "__main__":
    unittest.main()
