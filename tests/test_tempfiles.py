"""Tests for the temp-leak release gateway:

- a dedicated temp dir for audio extractions,
- a sweep that reclaims orphans left by killed processes (a kill bypasses
  the finally that normally unlinks them),
- the coverage guard that surfaces silent decode truncation.

These lock the behavior so the leak / silent-truncation can't regress
unnoticed in any forge release that consumes videoflow.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from videoflow.audio import _coverage_shortfall
from videoflow.tempfiles import audio_temp_dir, sweep_audio_temp


class TestAudioTempDir(unittest.TestCase):

    def test_is_dedicated_subdir_under_system_temp(self):
        d = audio_temp_dir()
        self.assertTrue(d.is_dir())
        self.assertEqual(d.name, "forge-audio")
        self.assertTrue(str(d).startswith(tempfile.gettempdir()))


class TestSweep(unittest.TestCase):

    def test_removes_stale_keeps_fresh(self):
        d = audio_temp_dir()
        stale = d / "tmp_gateway_stale.wav"
        fresh = d / "tmp_gateway_fresh.wav"
        stale.write_bytes(b"x" * 256)
        fresh.write_bytes(b"y" * 256)
        self.addCleanup(lambda: fresh.unlink(missing_ok=True))
        self.addCleanup(lambda: stale.unlink(missing_ok=True))
        # Backdate the stale file by two hours.
        past = time.time() - 7200
        os.utime(stale, (past, past))

        removed, freed = sweep_audio_temp(max_age_s=3600)

        self.assertFalse(stale.exists(), "stale orphan should be swept")
        self.assertTrue(fresh.exists(), "fresh (in-progress) file must survive")
        self.assertGreaterEqual(removed, 1)
        self.assertGreaterEqual(freed, 256)

    def test_never_raises_and_returns_counts(self):
        result = sweep_audio_temp(max_age_s=999_999)  # huge age → no-op
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestCoverageGuard(unittest.TestCase):

    def test_full_coverage_no_warning(self):
        self.assertIsNone(_coverage_shortfall(57_000 * 60, 57_800 * 60))

    def test_exact_match_no_warning(self):
        self.assertIsNone(_coverage_shortfall(60_000, 60_000))

    def test_truncated_warns(self):
        warn = _coverage_shortfall(44 * 60_000, 58 * 60_000)  # the Sinful case
        self.assertIsNotNone(warn)
        self.assertIn("truncated", warn)

    def test_small_gap_ignored(self):
        # 3s short of a 60s track — under the 5s floor, not worth alarming.
        self.assertIsNone(_coverage_shortfall(57_000, 60_000))

    def test_zero_source_is_safe(self):
        self.assertIsNone(_coverage_shortfall(1000, 0))


if __name__ == "__main__":
    unittest.main()
