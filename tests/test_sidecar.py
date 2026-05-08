"""Tests for videoflow.sidecar — read / write / field-level merge."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from videoflow.chapters import Chapter
from videoflow.sidecar import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_NAME,
    SidecarError,
    chapters_from_sidecar,
    read_sidecar,
    sidecar_path_for,
    write_sidecar,
)


def _media(td: str, name: str = "track.mp4") -> Path:
    p = Path(td) / name
    p.touch()
    return p


def _chapters(td: str) -> Path:
    return Path(td) / "track.chapters.json"


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

class TestSidecarPath(unittest.TestCase):

    def test_path_uses_stem_chapters_json(self):
        self.assertEqual(
            sidecar_path_for("/tmp/foo.mp4").name, "foo.chapters.json",
        )

    def test_path_strips_one_suffix(self):
        # Multi-dot stems keep all but the final suffix.
        self.assertEqual(
            sidecar_path_for("/tmp/clip.v2.wav").name, "clip.v2.chapters.json",
        )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class TestRead(unittest.TestCase):

    def test_returns_none_when_absent(self):
        with TemporaryDirectory() as td:
            self.assertIsNone(read_sidecar(_media(td)))

    def test_reads_v1_chapters_only(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "version": "1.0",
                "auto_generated": True,
                "chapters": [{"at_ms": 0, "name": "intro"}],
            }), encoding="utf-8")
            doc = read_sidecar(media)
            self.assertIsNotNone(doc)
            self.assertEqual(doc["chapters"], [{"at_ms": 0, "name": "intro"}])
            self.assertEqual(doc["phrases"], [])
            self.assertEqual(doc["provenance"], [])

    def test_reads_bare_list_as_chapters(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(
                json.dumps([{"at_ms": 0}, {"at_ms": 1000}]),
                encoding="utf-8",
            )
            doc = read_sidecar(media)
            self.assertEqual(len(doc["chapters"]), 2)

    def test_raises_on_invalid_json(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text("not json {", encoding="utf-8")
            with self.assertRaises(SidecarError):
                read_sidecar(media)

    def test_raises_on_chapter_missing_at_ms(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(
                json.dumps({"chapters": [{"name": "no anchor"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(SidecarError):
                read_sidecar(media)

    def test_raises_on_invalid_content_type(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(
                json.dumps({"chapters": [
                    {"at_ms": 0, "content_type": "rubbish"},
                ]}),
                encoding="utf-8",
            )
            with self.assertRaises(SidecarError):
                read_sidecar(media)

    def test_raises_on_invalid_phrase_mode(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(
                json.dumps({
                    "chapters": [{"at_ms": 0}],
                    "phrases": [{
                        "chapter_idx": 0, "at_ms": 0, "end_ms": 1000,
                        "mode": "not-a-mode",
                    }],
                }),
                encoding="utf-8",
            )
            with self.assertRaises(SidecarError):
                read_sidecar(media)

    def test_accepts_fast_and_slow_modes(self):
        """classify_modes returns fast/slow; schema must accept them."""
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0}],
                "phrases": [
                    {"chapter_idx": 0, "at_ms": 0, "end_ms": 1000, "mode": "fast"},
                    {"chapter_idx": 0, "at_ms": 1000, "end_ms": 2000, "mode": "slow"},
                ],
            }), encoding="utf-8")
            doc = read_sidecar(media)
            self.assertEqual(doc["phrases"][0]["mode"], "fast")
            self.assertEqual(doc["phrases"][1]["mode"], "slow")

    def test_accepts_locked_tone_vocabulary(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0, "tone": "build"}],
            }), encoding="utf-8")
            doc = read_sidecar(media)
            self.assertEqual(doc["chapters"][0]["tone"], "build")

    def test_raises_on_invalid_tone_label(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0, "tone": "soft"}],  # not in enum
            }), encoding="utf-8")
            with self.assertRaises(SidecarError):
                read_sidecar(media)

    def test_accepts_locked_shape_vocabulary(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0, "shape": "rise"}],
            }), encoding="utf-8")
            doc = read_sidecar(media)
            self.assertEqual(doc["chapters"][0]["shape"], "rise")

    def test_raises_on_invalid_shape(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0, "shape": "wobble"}],  # not in enum
            }), encoding="utf-8")
            with self.assertRaises(SidecarError):
                read_sidecar(media)

    def test_v1_top_level_user_edited_propagates_to_records(self):
        """v1 sidecars with top-level auto_generated=false: each chapter is latched."""
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "version": "1.0",
                "auto_generated": False,  # legacy whole-file user-edit marker
                "chapters": [
                    {"at_ms": 0, "name": "user named"},
                    {"at_ms": 1000, "name": "also user", "auto_generated": True},
                ],
            }), encoding="utf-8")
            doc = read_sidecar(media)
            self.assertIs(doc["chapters"][0]["auto_generated"], False)
            # explicit per-record value wins over propagation
            self.assertIs(doc["chapters"][1]["auto_generated"], True)


# ---------------------------------------------------------------------------
# Fresh write
# ---------------------------------------------------------------------------

class TestFreshWrite(unittest.TestCase):

    def test_writes_v2_doc_with_provenance(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            payload = {
                "chapters": [{"at_ms": 0, "content_type": "music"}],
                "generated_by": {"tool": "videoflow.structural", "tool_version": "x"},
            }
            path = write_sidecar(
                media, payload,
                writer="videoflow.structural", writer_version="0.0.5",
            )
            self.assertEqual(path.name, "track.chapters.json")
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(doc["schema"], SCHEMA_NAME)
            self.assertEqual(doc["version"], CURRENT_SCHEMA_VERSION)
            self.assertTrue(doc["auto_generated"])
            self.assertEqual(len(doc["provenance"]), 1)
            self.assertEqual(doc["provenance"][0]["writer"], "videoflow.structural")
            self.assertEqual(doc["provenance"][0]["writer_version"], "0.0.5")
            self.assertIn("chapters", doc["provenance"][0]["fields"])

    def test_returns_target_path(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            path = write_sidecar(
                media, {"chapters": []},
                writer="x",
            )
            self.assertEqual(path, sidecar_path_for(media))


# ---------------------------------------------------------------------------
# Merge — analyze mode
# ---------------------------------------------------------------------------

class TestMergeAnalyze(unittest.TestCase):

    def _seed(self, td: str, doc: dict) -> Path:
        media = _media(td)
        _chapters(td).write_text(json.dumps(doc), encoding="utf-8")
        return media

    def test_overwrites_analytical_when_latch_open(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "version": "2.0",
                "chapters": [{
                    "at_ms": 0, "end_ms": 1000, "name": "intro",
                    "content_type": "ambient", "confidence": 0.5,
                    "auto_generated": True,
                }],
            })
            write_sidecar(media, {
                "chapters": [{
                    "at_ms": 0,
                    "content_type": "music", "confidence": 0.9,
                    "evidence": ["mfcc"],
                }],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            ch = doc["chapters"][0]
            self.assertEqual(ch["content_type"], "music")
            self.assertEqual(ch["confidence"], 0.9)
            self.assertEqual(ch["evidence"], ["mfcc"])

    def test_preserves_authored_fields(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [{
                    "at_ms": 0, "end_ms": 1000,
                    "name": "user-named", "intent": "intro",
                    "include": False,
                    "auto_generated": True,
                }],
            })
            write_sidecar(media, {
                "chapters": [{
                    "at_ms": 0, "end_ms": 1000,
                    "name": "auto-detected",  # writer should NOT clobber
                    "intent": "build",
                    "include": True,
                    "content_type": "music",
                }],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            ch = doc["chapters"][0]
            self.assertEqual(ch["name"], "user-named")
            self.assertEqual(ch["intent"], "intro")
            self.assertIs(ch["include"], False)
            self.assertEqual(ch["content_type"], "music")  # ANALYTICAL applied

    def test_preserves_structural_fields(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [{
                    "at_ms": 0, "end_ms": 1000,
                    "auto_generated": True,
                }],
            })
            write_sidecar(media, {
                "chapters": [{
                    "at_ms": 0, "end_ms": 9999,  # boundary drift
                    "content_type": "music",
                }],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["chapters"][0]["end_ms"], 1000)

    def test_latch_freezes_analytical(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [{
                    "at_ms": 0, "end_ms": 1000,
                    "name": "user", "content_type": "ambient",
                    "auto_generated": False,
                }],
            })
            write_sidecar(media, {
                "chapters": [{
                    "at_ms": 0,
                    "content_type": "music", "confidence": 0.99,
                }],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            ch = doc["chapters"][0]
            self.assertEqual(ch["content_type"], "ambient")
            self.assertNotIn("confidence", ch)
            self.assertEqual(ch["name"], "user")

    def test_appends_new_chapters(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [{"at_ms": 0, "auto_generated": True}],
            })
            write_sidecar(media, {
                "chapters": [
                    {"at_ms": 0, "content_type": "music"},
                    {"at_ms": 5000, "content_type": "ambient"},
                ],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(len(doc["chapters"]), 2)
            self.assertEqual(doc["chapters"][1]["at_ms"], 5000)

    def test_keeps_existing_chapters_not_in_incoming(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [
                    {"at_ms": 0, "name": "intro", "auto_generated": False},
                    {"at_ms": 1000, "name": "outro", "auto_generated": False},
                ],
            })
            # Re-detection only finds the first chapter.
            write_sidecar(media, {
                "chapters": [{"at_ms": 0, "content_type": "music"}],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(len(doc["chapters"]), 2)
            self.assertEqual(doc["chapters"][1]["name"], "outro")

    def test_mixed_field_tone_follows_latch(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [
                    {"at_ms": 0, "tone": "tender", "auto_generated": True},
                    {"at_ms": 1000, "tone": "tender", "auto_generated": False},
                ],
            })
            write_sidecar(media, {
                "chapters": [
                    {"at_ms": 0, "tone": "build"},
                    {"at_ms": 1000, "tone": "build"},
                ],
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["chapters"][0]["tone"], "build")    # latch open
            self.assertEqual(doc["chapters"][1]["tone"], "tender")   # latched

    def test_energy_block_replaced_wholesale(self):
        with TemporaryDirectory() as td:
            media = self._seed(td, {
                "chapters": [{"at_ms": 0, "auto_generated": True}],
                "energy": {"percentiles": {"p5": 0.0, "p25": 0.1, "p50": 0.2,
                                           "p75": 0.3, "p95": 0.4}},
            })
            write_sidecar(media, {
                "chapters": [{"at_ms": 0}],
                "energy": {"percentiles": {"p5": 0.5, "p25": 0.6, "p50": 0.7,
                                           "p75": 0.8, "p95": 0.9}},
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["energy"]["percentiles"]["p5"], 0.5)


# ---------------------------------------------------------------------------
# Merge — edit mode
# ---------------------------------------------------------------------------

class TestMergeEdit(unittest.TestCase):

    def test_edit_writes_authored_fields(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{
                    "at_ms": 0, "end_ms": 1000,
                    "name": "", "auto_generated": True,
                }],
            }), encoding="utf-8")
            write_sidecar(media, {
                "chapters": [{"at_ms": 0, "name": "user-named"}],
            }, writer="FunscriptForge", mode="edit")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["chapters"][0]["name"], "user-named")

    def test_edit_flips_per_record_auto_generated_false(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0, "auto_generated": True}],
            }), encoding="utf-8")
            write_sidecar(media, {
                "chapters": [{"at_ms": 0, "name": "edited"}],
            }, writer="FunscriptForge", mode="edit")
            doc = json.loads(_chapters(td).read_text())
            self.assertIs(doc["chapters"][0]["auto_generated"], False)

    def test_edit_respects_latch_freezes_record(self):
        """Even edit mode honours an explicit per-record latch.

        Use case: a user has marked a record as deliberately authored
        and a different writer should not be able to overwrite it
        without an explicit un-latch step.
        """
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{
                    "at_ms": 0, "name": "first user",
                    "auto_generated": False,
                }],
            }), encoding="utf-8")
            write_sidecar(media, {
                "chapters": [{"at_ms": 0, "name": "second writer"}],
            }, writer="forgeassembler", mode="edit")
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["chapters"][0]["name"], "first user")

    def test_edit_flips_top_level_auto_generated_false(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "auto_generated": True,
                "chapters": [{"at_ms": 0, "auto_generated": True}],
            }), encoding="utf-8")
            write_sidecar(media, {
                "chapters": [{"at_ms": 0, "name": "x"}],
            }, writer="FunscriptForge", mode="edit")
            doc = json.loads(_chapters(td).read_text())
            self.assertIs(doc["auto_generated"], False)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance(unittest.TestCase):

    def test_each_write_appends_entry(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            write_sidecar(
                media, {"chapters": [{"at_ms": 0}]},
                writer="videoflow.structural", writer_version="0.0.5",
            )
            write_sidecar(
                media, {"chapters": [{"at_ms": 0, "content_type": "music"}]},
                writer="videoflow.structural", writer_version="0.0.6",
            )
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(len(doc["provenance"]), 2)
            self.assertEqual(doc["provenance"][0]["writer_version"], "0.0.5")
            self.assertEqual(doc["provenance"][1]["writer_version"], "0.0.6")

    def test_writer_version_optional(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            write_sidecar(
                media, {"chapters": [{"at_ms": 0}]},
                writer="x",  # no writer_version
            )
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["provenance"][0]["writer"], "x")
            self.assertNotIn("writer_version", doc["provenance"][0])

    def test_records_fields_touched(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            write_sidecar(media, {
                "chapters": [{"at_ms": 0}],
                "phrases": [{"chapter_idx": 0, "at_ms": 0, "end_ms": 100}],
                "energy": {"percentiles": {"p5": 0, "p25": 0, "p50": 0,
                                           "p75": 0, "p95": 0}},
            }, writer="videoflow.structural")
            doc = json.loads(_chapters(td).read_text())
            fields = doc["provenance"][0]["fields"]
            self.assertEqual(set(fields), {"chapters", "phrases", "energy"})


# ---------------------------------------------------------------------------
# Round-trip / unknown fields
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):

    def test_unknown_top_level_fields_round_trip(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{"at_ms": 0, "auto_generated": True}],
                "experimental_block": {"vendor": "future"},
            }), encoding="utf-8")
            write_sidecar(
                media, {"chapters": [{"at_ms": 0, "content_type": "music"}]},
                writer="videoflow.structural",
            )
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["experimental_block"], {"vendor": "future"})

    def test_unknown_chapter_fields_round_trip_through_merge(self):
        with TemporaryDirectory() as td:
            media = _media(td)
            _chapters(td).write_text(json.dumps({
                "chapters": [{
                    "at_ms": 0,
                    "vendor_score": 0.42,  # unknown field
                    "auto_generated": True,
                }],
            }), encoding="utf-8")
            write_sidecar(
                media, {"chapters": [{"at_ms": 0, "content_type": "music"}]},
                writer="videoflow.structural",
            )
            doc = json.loads(_chapters(td).read_text())
            self.assertEqual(doc["chapters"][0]["vendor_score"], 0.42)
            self.assertEqual(doc["chapters"][0]["content_type"], "music")


# ---------------------------------------------------------------------------
# Typed view
# ---------------------------------------------------------------------------

class TestTypedView(unittest.TestCase):

    def test_chapters_from_sidecar_returns_chapter_records(self):
        doc = {
            "chapters": [
                {"at_ms": 0, "name": "intro", "content_type": "music",
                 "confidence": 0.8, "evidence": ["mfcc"]},
            ],
        }
        chapters = chapters_from_sidecar(doc)
        self.assertEqual(len(chapters), 1)
        self.assertIsInstance(chapters[0], Chapter)
        self.assertEqual(chapters[0].name, "intro")
        self.assertEqual(chapters[0].content_type, "music")


if __name__ == "__main__":
    unittest.main()
