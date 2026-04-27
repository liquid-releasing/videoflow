"""Tests for videoflow.events — funscript metadata events read/write."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from videoflow.events import (
    EventError,
    FunscriptEvent,
    events_from_dicts,
    events_to_dicts,
    read_events,
    write_events,
)


def _empty_funscript_at(path: Path) -> Path:
    """Create a minimal valid .funscript at *path*."""
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "actions": [{"at": 0, "pos": 50}, {"at": 1000, "pos": 50}],
                "metadata": {"title": "test"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class TestFunscriptEventDataclass(unittest.TestCase):

    def test_default_fields(self):
        e = FunscriptEvent(type="accent", at_ms=12345)
        self.assertEqual(e.type, "accent")
        self.assertEqual(e.at_ms, 12345)
        self.assertIsNone(e.duration_ms)
        self.assertEqual(e.confidence, 1.0)
        self.assertEqual(e.source, [])
        self.assertEqual(e.params, {})

    def test_to_dict_omits_none_duration(self):
        e = FunscriptEvent(type="accent", at_ms=100)
        d = e.to_dict()
        self.assertNotIn("duration_ms", d)

    def test_to_dict_includes_duration_when_set(self):
        e = FunscriptEvent(type="edge_hold", at_ms=100, duration_ms=8000)
        d = e.to_dict()
        self.assertEqual(d["duration_ms"], 8000)

    def test_to_dict_returns_independent_collections(self):
        """Mutating the dict's source/params must not mutate the event."""
        e = FunscriptEvent(
            type="accent", at_ms=0, source=["audio_peak"], params={"k": 1}
        )
        d = e.to_dict()
        d["source"].append("video_peak")
        d["params"]["k"] = 99
        self.assertEqual(e.source, ["audio_peak"])
        self.assertEqual(e.params, {"k": 1})

    def test_from_dict_minimal(self):
        e = FunscriptEvent.from_dict({"type": "accent", "at_ms": 500})
        self.assertEqual(e.type, "accent")
        self.assertEqual(e.at_ms, 500)
        self.assertIsNone(e.duration_ms)
        self.assertEqual(e.confidence, 1.0)

    def test_from_dict_full(self):
        d = {
            "type": "edge_hold",
            "at_ms": 42000,
            "duration_ms": 8000,
            "confidence": 0.78,
            "source": ["audio_peak"],
            "params": {"holds_under_climax": True},
        }
        e = FunscriptEvent.from_dict(d)
        self.assertEqual(e.type, "edge_hold")
        self.assertEqual(e.at_ms, 42000)
        self.assertEqual(e.duration_ms, 8000)
        self.assertAlmostEqual(e.confidence, 0.78)
        self.assertEqual(e.source, ["audio_peak"])
        self.assertEqual(e.params, {"holds_under_climax": True})

    def test_round_trip(self):
        original = FunscriptEvent(
            type="climax_candidate",
            at_ms=780_000,
            duration_ms=None,
            confidence=0.91,
            source=["audio_peak", "video_peak"],
            params={"narrative": "peak"},
        )
        clone = FunscriptEvent.from_dict(original.to_dict())
        self.assertEqual(original, clone)

    # --- malformed ----------------------------------------------------

    def test_from_dict_missing_type_raises(self):
        with self.assertRaises(EventError):
            FunscriptEvent.from_dict({"at_ms": 100})

    def test_from_dict_missing_at_ms_raises(self):
        with self.assertRaises(EventError):
            FunscriptEvent.from_dict({"type": "accent"})

    def test_from_dict_invalid_at_ms_raises(self):
        with self.assertRaises(EventError):
            FunscriptEvent.from_dict({"type": "accent", "at_ms": "soon"})

    def test_from_dict_invalid_duration_raises(self):
        with self.assertRaises(EventError):
            FunscriptEvent.from_dict(
                {"type": "edge_hold", "at_ms": 0, "duration_ms": "long"}
            )

    def test_from_dict_invalid_confidence_raises(self):
        with self.assertRaises(EventError):
            FunscriptEvent.from_dict(
                {"type": "accent", "at_ms": 0, "confidence": "high"}
            )


class TestEventListSerialisation(unittest.TestCase):

    def test_events_to_dicts_preserves_order(self):
        events = [
            FunscriptEvent(type="b", at_ms=200),
            FunscriptEvent(type="a", at_ms=100),
        ]
        dicts = events_to_dicts(events)
        self.assertEqual([d["type"] for d in dicts], ["b", "a"])

    def test_events_from_dicts_returns_events(self):
        dicts = [{"type": "accent", "at_ms": 0}, {"type": "edge_hold", "at_ms": 1000}]
        events = events_from_dicts(dicts)
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], FunscriptEvent)

    def test_events_from_dicts_non_list_raises(self):
        with self.assertRaises(EventError):
            events_from_dicts({"type": "accent", "at_ms": 0})

    def test_events_from_dicts_malformed_item_raises(self):
        with self.assertRaises(EventError):
            events_from_dicts([{"type": "accent"}])  # missing at_ms


class TestReadWriteEvents(unittest.TestCase):

    def test_read_events_missing_file_raises(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                read_events(Path(td) / "nope.funscript")

    def test_read_events_no_metadata_returns_empty(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "track.funscript"
            p.write_text(json.dumps({"version": "1.0", "actions": []}))
            self.assertEqual(read_events(p), [])

    def test_read_events_no_events_field_returns_empty(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            self.assertEqual(read_events(p), [])

    def test_read_events_invalid_json_raises_event_error(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "track.funscript"
            p.write_text("not json")
            with self.assertRaises(EventError):
                read_events(p)

    def test_read_events_returns_parsed_events(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "track.funscript"
            p.write_text(json.dumps({
                "version": "1.0",
                "actions": [],
                "metadata": {
                    "events": [
                        {"type": "accent", "at_ms": 100, "confidence": 0.9},
                        {"type": "edge_hold", "at_ms": 5000,
                         "duration_ms": 3000, "source": ["audio_peak"]},
                    ]
                }
            }))
            events = read_events(p)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].type, "accent")
            self.assertEqual(events[1].duration_ms, 3000)
            self.assertEqual(events[1].source, ["audio_peak"])

    def test_write_events_creates_metadata_events(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            write_events(p, [FunscriptEvent(type="accent", at_ms=500)])
            doc = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(doc["metadata"]["events"][0]["type"], "accent")

    def test_write_events_preserves_other_metadata(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            write_events(p, [FunscriptEvent(type="accent", at_ms=0)])
            doc = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(doc["metadata"]["title"], "test")

    def test_write_events_preserves_actions(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            original = json.loads(p.read_text(encoding="utf-8"))["actions"]
            write_events(p, [FunscriptEvent(type="accent", at_ms=0)])
            after = json.loads(p.read_text(encoding="utf-8"))["actions"]
            self.assertEqual(original, after)

    def test_write_events_replaces_existing(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            write_events(p, [FunscriptEvent(type="accent", at_ms=100)])
            write_events(p, [FunscriptEvent(type="edge_hold", at_ms=2000)])
            doc = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(len(doc["metadata"]["events"]), 1)
            self.assertEqual(doc["metadata"]["events"][0]["type"], "edge_hold")

    def test_write_events_sorts_by_at_ms(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            write_events(p, [
                FunscriptEvent(type="late", at_ms=2000),
                FunscriptEvent(type="early", at_ms=100),
                FunscriptEvent(type="middle", at_ms=1000),
            ])
            doc = json.loads(p.read_text(encoding="utf-8"))
            ordered = [e["type"] for e in doc["metadata"]["events"]]
            self.assertEqual(ordered, ["early", "middle", "late"])

    def test_write_events_missing_file_raises(self):
        with TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                write_events(Path(td) / "nope.funscript", [])

    def test_write_events_invalid_json_raises_event_error(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "track.funscript"
            p.write_text("not json")
            with self.assertRaises(EventError):
                write_events(p, [FunscriptEvent(type="accent", at_ms=0)])

    def test_round_trip_through_disk(self):
        with TemporaryDirectory() as td:
            p = _empty_funscript_at(Path(td) / "track.funscript")
            originals = [
                FunscriptEvent(type="accent", at_ms=0),
                FunscriptEvent(
                    type="edge_hold", at_ms=10_000, duration_ms=4000,
                    confidence=0.8, source=["audio_peak"],
                    params={"narrative": True},
                ),
            ]
            write_events(p, originals)
            loaded = read_events(p)
            self.assertEqual(loaded, originals)


if __name__ == "__main__":
    unittest.main()
