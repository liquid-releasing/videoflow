"""Tests for videoflow.funscript_stats — the shared quality-metrics core."""

from __future__ import annotations

import unittest

from videoflow.funscript_stats import (
    decile_histogram,
    dynamics_index,
    motion_stats,
    position_stats,
    summarize,
    to_pairs,
    velocities,
    windowed_profile,
)


def _bimodal(n=100, period_ms=500):
    """Rail-to-rail strokes: alternating 0 / 100."""
    return [{"at": i * period_ms, "pos": 0 if i % 2 else 100} for i in range(n)]


def _bell(n=100, period_ms=500):
    """Centered strokes that never reach the rails: alternating 40 / 60."""
    return [{"at": i * period_ms, "pos": 40 if i % 2 else 60} for i in range(n)]


class TestToPairs(unittest.TestCase):

    def test_accepts_dicts_and_tuples(self):
        self.assertEqual(to_pairs([{"at": 0, "pos": 50}]), [(0, 50)])
        self.assertEqual(to_pairs([(0, 50)]), [(0, 50)])

    def test_sorts_by_time(self):
        pairs = to_pairs([{"at": 100, "pos": 0}, {"at": 0, "pos": 100}])
        self.assertEqual([t for t, _ in pairs], [0, 100])

    def test_clamps_positions(self):
        pairs = to_pairs([(0, 150), (1, -20)])
        self.assertEqual([p for _, p in pairs], [100, 0])


class TestPositionStats(unittest.TestCase):

    def test_decile_histogram_sums_to_n(self):
        pairs = to_pairs(_bimodal())
        self.assertEqual(sum(decile_histogram(pairs)), len(pairs))

    def test_bimodal_is_flagged_bimodal(self):
        stats = position_stats(to_pairs(_bimodal()))
        self.assertTrue(stats["bimodal"])
        self.assertGreater(stats["rails_pct"], 90)
        self.assertEqual(stats["mid_pct"], 0)

    def test_bell_is_not_bimodal(self):
        stats = position_stats(to_pairs(_bell()))
        self.assertFalse(stats["bimodal"])
        self.assertEqual(stats["rails_pct"], 0)
        self.assertGreater(stats["mid_pct"], 90)

    def test_empty_is_safe(self):
        self.assertEqual(position_stats([]), {"n": 0})


class TestMotion(unittest.TestCase):

    def test_velocities_units_per_second(self):
        # 100-unit move in 500ms = 200 units/sec
        pairs = to_pairs([(0, 0), (500, 100)])
        self.assertAlmostEqual(velocities(pairs)[0][1], 200.0)

    def test_rate_is_actions_per_second(self):
        pairs = to_pairs([{"at": i * 500, "pos": i % 2 * 100} for i in range(5)])
        # 5 actions over 2.0s span = 2.5/s
        self.assertAlmostEqual(motion_stats(pairs)["rate"], 2.5)

    def test_avg_stroke_full_for_bimodal(self):
        self.assertAlmostEqual(motion_stats(to_pairs(_bimodal()))["avg_stroke"], 100.0)


class TestWindowedProfileAndDynamics(unittest.TestCase):

    def test_profile_has_one_entry_per_window(self):
        profile = windowed_profile(to_pairs(_bimodal(n=120)), n_windows=6)
        self.assertEqual(len(profile), 6)

    def test_dynamic_track_scores_higher_cov_than_monotone(self):
        # Monotone: uniform 2/s throughout.
        monotone = [{"at": i * 500, "pos": i % 2 * 100} for i in range(120)]
        # Dynamic: first half sparse (1/s), second half dense (4/s).
        dynamic = [{"at": i * 1000, "pos": i % 2 * 100} for i in range(30)]
        t = dynamic[-1]["at"]
        dynamic += [{"at": t + i * 250, "pos": i % 2 * 100} for i in range(1, 120)]
        mono_cov = dynamics_index(windowed_profile(to_pairs(monotone)))["rate_cov"]
        dyn_cov = dynamics_index(windowed_profile(to_pairs(dynamic)))["rate_cov"]
        self.assertGreater(dyn_cov, mono_cov)


class TestSummarize(unittest.TestCase):

    def test_summary_shape(self):
        s = summarize(_bimodal())
        self.assertIn("position", s)
        self.assertIn("motion", s)
        self.assertIn("profile", s)
        self.assertIn("dynamics", s)
        self.assertTrue(s["position"]["bimodal"])


if __name__ == "__main__":
    unittest.main()
