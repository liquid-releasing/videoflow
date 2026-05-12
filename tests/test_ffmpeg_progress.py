"""Tests for the ffmpeg sub-stage progress helpers in videoflow.structural —
``_format_extract_timestamp`` and ``_parse_ffmpeg_progress_line``. These
turn ffmpeg's structured ``-progress`` output into Stepper detail labels."""

from __future__ import annotations

import unittest

from videoflow.structural import (
    _format_extract_timestamp,
    _parse_ffmpeg_progress_line,
)


class TestFormatExtractTimestamp(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(_format_extract_timestamp(0), "0:00")

    def test_seconds_only(self):
        self.assertEqual(_format_extract_timestamp(23_500_000), "0:23")

    def test_minutes_seconds(self):
        # 1m 23s
        self.assertEqual(_format_extract_timestamp(83_000_000), "1:23")

    def test_promotes_to_hours_when_needed(self):
        # 1h 02m 03s — hours format kicks in
        us = (3600 + 2 * 60 + 3) * 1_000_000
        self.assertEqual(_format_extract_timestamp(us), "1:02:03")

    def test_negative_clamps_to_zero(self):
        self.assertEqual(_format_extract_timestamp(-500), "0:00")

    def test_truncates_microseconds(self):
        # 0.999 seconds → still 0:00 (we only show whole seconds)
        self.assertEqual(_format_extract_timestamp(999_999), "0:00")


class TestParseFfmpegProgressLine(unittest.TestCase):

    def test_out_time_us_returns_label(self):
        label = _parse_ffmpeg_progress_line("out_time_us=12345000")
        self.assertEqual(label, "Extracting audio… 0:12 done")

    def test_strips_trailing_newline(self):
        label = _parse_ffmpeg_progress_line("out_time_us=60000000\n")
        self.assertEqual(label, "Extracting audio… 1:00 done")

    def test_other_progress_keys_ignored(self):
        for line in (
            "frame=1234",
            "fps=58.5",
            "bitrate=128.0kbits/s",
            "speed=1.2x",
            "progress=continue",
            "progress=end",
            "",
        ):
            self.assertIsNone(
                _parse_ffmpeg_progress_line(line),
                msg=f"line {line!r} should not produce a label",
            )

    def test_malformed_value_returns_none(self):
        self.assertIsNone(_parse_ffmpeg_progress_line("out_time_us=NaN"))
        self.assertIsNone(_parse_ffmpeg_progress_line("out_time_us="))


if __name__ == "__main__":
    unittest.main()
