"""Tests for videoflow.cli._emit_progress — the side-channel progress emit
that forgegen's bridge polls. Ensures stderr always receives the line and
that VIDEOFLOW_PROGRESS_FILE is written per-call without keeping a long-
lived handle (which on Windows hides metadata updates from other procs)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from videoflow.cli import _emit_progress


class TestEmitProgress(unittest.TestCase):

    def setUp(self):
        self._prev_env = os.environ.pop("VIDEOFLOW_PROGRESS_FILE", None)

    def tearDown(self):
        if self._prev_env is not None:
            os.environ["VIDEOFLOW_PROGRESS_FILE"] = self._prev_env
        else:
            os.environ.pop("VIDEOFLOW_PROGRESS_FILE", None)

    def test_stderr_receives_progress_prefix(self):
        with patch("sys.stderr", new=StringIO()) as buf:
            _emit_progress("Loading audio (librosa)…")
        self.assertIn("progress: Loading audio (librosa)…", buf.getvalue())

    def test_file_unset_no_write(self):
        # No env var, no file should be created. Should not raise.
        with patch("sys.stderr", new=StringIO()):
            _emit_progress("any label")

    def test_file_set_appends_each_call(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "progress.log"
            os.environ["VIDEOFLOW_PROGRESS_FILE"] = str(p)
            with patch("sys.stderr", new=StringIO()):
                _emit_progress("first")
                _emit_progress("second")
                _emit_progress("third")
            text = p.read_text(encoding="utf-8")
            self.assertEqual(
                text,
                "progress: first\nprogress: second\nprogress: third\n",
            )

    def test_file_handle_does_not_stay_open(self):
        # Critical: on Windows, a long-lived writer handle hides file-size
        # updates from other processes (forgegen's polling). Per-emit
        # open+close means each call commits a fresh file size.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "progress.log"
            os.environ["VIDEOFLOW_PROGRESS_FILE"] = str(p)
            with patch("sys.stderr", new=StringIO()):
                _emit_progress("one")
                size_after_first = p.stat().st_size
                _emit_progress("two")
                size_after_second = p.stat().st_size
            self.assertGreater(size_after_second, size_after_first)

    def test_file_unwritable_does_not_raise(self):
        # Pointing at a directory (not a file) makes open() raise OSError;
        # the helper must swallow it so progress reporting never breaks
        # the underlying analysis.
        with tempfile.TemporaryDirectory() as td:
            os.environ["VIDEOFLOW_PROGRESS_FILE"] = td  # a directory
            with patch("sys.stderr", new=StringIO()):
                _emit_progress("should not raise")


if __name__ == "__main__":
    unittest.main()
