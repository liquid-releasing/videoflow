"""Tests for videoflow.patterns — catalog data + JSON export contract."""

from __future__ import annotations

import json
import unittest

from videoflow.patterns import (
    CATALOG,
    CONSUMERS,
    NONE_ID,
    VALID_PATTERN_IDS,
    Pattern,
    PatternConsumer,
    PatternParam,
    get,
    is_valid,
    to_json,
)


class TestCatalogShape(unittest.TestCase):

    def test_seven_patterns(self):
        self.assertEqual(len(CATALOG), 7)

    def test_pattern_ids_are_unique(self):
        ids = [p.id for p in CATALOG]
        self.assertEqual(len(ids), len(set(ids)))

    def test_pattern_ids_are_h_prefixed(self):
        for p in CATALOG:
            self.assertTrue(p.id.startswith("h_"), f"{p.id} should be h_-prefixed")

    def test_each_pattern_has_at_least_one_param(self):
        for p in CATALOG:
            self.assertGreater(len(p.params), 0, f"{p.id} has no params")

    def test_each_pattern_declares_consumers(self):
        for p in CATALOG:
            self.assertGreater(len(p.consumers), 0, f"{p.id} has no consumers")

    def test_consumer_ids_match_known_set(self):
        known = {c.id for c in CONSUMERS}
        for p in CATALOG:
            for c in p.consumers:
                self.assertIn(c, known, f"{p.id} references unknown consumer {c!r}")


class TestLookup(unittest.TestCase):

    def test_get_returns_pattern_by_id(self):
        p = get("h_pulse")
        self.assertIsNotNone(p)
        self.assertEqual(p.label, "Pulse")

    def test_get_returns_none_for_unknown(self):
        self.assertIsNone(get("h_unknown"))

    def test_is_valid_includes_catalog(self):
        for p in CATALOG:
            self.assertTrue(is_valid(p.id))

    def test_is_valid_includes_none_sentinel(self):
        self.assertTrue(is_valid(NONE_ID))

    def test_is_valid_rejects_unknown(self):
        self.assertFalse(is_valid("h_nope"))


class TestSerialisation(unittest.TestCase):

    def test_to_json_parses(self):
        parsed = json.loads(to_json())
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(len(parsed["patterns"]), 7)
        self.assertEqual(len(parsed["consumers"]), 4)
        self.assertEqual(parsed["none_sentinel"], NONE_ID)

    def test_to_json_pattern_round_trip(self):
        parsed = json.loads(to_json())
        pulse = next(p for p in parsed["patterns"] if p["id"] == "h_pulse")
        self.assertEqual(pulse["label"], "Pulse")
        self.assertEqual(pulse["color"], "#4cc3ff")
        period = next(p for p in pulse["params"] if p["id"] == "period_ms")
        self.assertEqual(period["unit"], "ms")
        self.assertEqual(period["default"], 600)

    def test_to_json_enum_param_emits_enumValues(self):
        parsed = json.loads(to_json())
        rolling = next(p for p in parsed["patterns"] if p["id"] == "h_rolling")
        direction = next(p for p in rolling["params"] if p["id"] == "direction")
        self.assertEqual(
            direction["enumValues"],
            ["clockwise", "counter", "outward", "inward"],
        )

    def test_to_json_hint_param_emits_hint(self):
        parsed = json.loads(to_json())
        reactive = next(p for p in parsed["patterns"] if p["id"] == "h_reactive")
        downbeat_w = next(p for p in reactive["params"] if p["id"] == "downbeat_weight")
        self.assertIn("downbeat", downbeat_w["hint"])

    def test_to_json_excludes_none_default(self):
        # PatternParam with default=None should not emit a 'default' key.
        # Verify by checking that all emitted defaults are non-null.
        parsed = json.loads(to_json())
        for pat in parsed["patterns"]:
            for param in pat["params"]:
                if "default" in param:
                    self.assertIsNotNone(param["default"])


class TestValidIDs(unittest.TestCase):

    def test_valid_set_size(self):
        self.assertEqual(len(VALID_PATTERN_IDS), 8)  # 7 patterns + _none


if __name__ == "__main__":
    unittest.main()
