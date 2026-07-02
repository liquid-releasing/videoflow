"""Tests for videoflow.chapter_clips — extraction args + resolution probe."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from videoflow.chapter_clips import (
    CACHE_VERSION,
    DOWNSCALE_WIDTH_THRESHOLD,
    FFMPEG_CLIP_ARGS,
    FFMPEG_CLIP_ARGS_4K_DOWNSCALE,
    _PROBE_RESOLUTION_RE,
    _is_cfr,
    _probe_video_dimensions,
)


# Real-ish ffmpeg -i stderr capture from a 4K HEVC source. ffmpeg's
# Video stream line carries the resolution after the pixel format;
# downstream WxH (like display aspect ratio, e.g. "DAR 16:9") arrives
# bracketed, so it doesn't fight our regex.
_FFMPEG_STDERR_4K = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/path/source.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:62:32.06, start: 0.000000, bitrate: 11532 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main 10) (hev1 / 0x31766568), yuv420p10le(tv, bt2020nc/bt2020/smpte2084), 3840x2160 [SAR 1:1 DAR 16:9], 11261 kb/s, 23.98 fps, 23.98 tbr, 24k tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 256 kb/s (default)
At least one output file must be specified
"""

_FFMPEG_STDERR_1080P = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/path/source.mp4':
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 5023 kb/s, 30 fps, 30 tbr, 30k tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC), 48000 Hz, stereo, fltp, 192 kb/s
At least one output file must be specified
"""

_FFMPEG_STDERR_AUDIO_ONLY = """\
Input #0, mp3, from '/path/song.mp3':
  Duration: 00:03:42.18, start: 0.025057, bitrate: 320 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s
At least one output file must be specified
"""


class TestProbeResolutionRegex(unittest.TestCase):

    def test_matches_4k_stream(self):
        m = _PROBE_RESOLUTION_RE.search(_FFMPEG_STDERR_4K)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 3840)
        self.assertEqual(int(m.group(2)), 2160)

    def test_matches_1080p_stream(self):
        m = _PROBE_RESOLUTION_RE.search(_FFMPEG_STDERR_1080P)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 1920)
        self.assertEqual(int(m.group(2)), 1080)

    def test_no_match_on_audio_only(self):
        m = _PROBE_RESOLUTION_RE.search(_FFMPEG_STDERR_AUDIO_ONLY)
        self.assertIsNone(m)


class TestProbeVideoDimensions(unittest.TestCase):

    def _fake_proc(self, stderr_text: str):
        class _P:
            stderr = stderr_text
            returncode = 1
        return _P()

    def test_returns_4k_dimensions(self):
        with patch(
            "videoflow.chapter_clips.subprocess.run",
            return_value=self._fake_proc(_FFMPEG_STDERR_4K),
        ):
            dims = _probe_video_dimensions("/x/y.mp4", "ffmpeg")
        self.assertEqual(dims, (3840, 2160))

    def test_returns_1080p_dimensions(self):
        with patch(
            "videoflow.chapter_clips.subprocess.run",
            return_value=self._fake_proc(_FFMPEG_STDERR_1080P),
        ):
            dims = _probe_video_dimensions("/x/y.mp4", "ffmpeg")
        self.assertEqual(dims, (1920, 1080))

    def test_returns_none_for_audio_only(self):
        with patch(
            "videoflow.chapter_clips.subprocess.run",
            return_value=self._fake_proc(_FFMPEG_STDERR_AUDIO_ONLY),
        ):
            dims = _probe_video_dimensions("/x/y.mp3", "ffmpeg")
        self.assertIsNone(dims)

    def test_returns_none_when_ffmpeg_missing(self):
        with patch(
            "videoflow.chapter_clips.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            dims = _probe_video_dimensions("/x/y.mp4", "ffmpeg-not-here")
        self.assertIsNone(dims)


class TestDownscaleArgs(unittest.TestCase):

    def test_threshold_is_1920(self):
        # Documented as "downscale anything > 1080p HD". 1920 is the
        # exclusive lower bound — 1080p sources keep the SDR args.
        self.assertEqual(DOWNSCALE_WIDTH_THRESHOLD, 1920)

    def test_downscale_args_include_scale_filter(self):
        args = FFMPEG_CLIP_ARGS_4K_DOWNSCALE
        self.assertIn("-vf", args)
        vf_idx = args.index("-vf")
        self.assertEqual(args[vf_idx + 1], "scale=1280:720:flags=lanczos")

    def test_downscale_args_preserve_color_metadata(self):
        # Without these flags ffmpeg drops color tags on re-encode and
        # iris coloring on graded sources reads desaturated. Pinned.
        for flag in ("-color_primaries", "-color_trc", "-colorspace", "-color_range"):
            self.assertIn(flag, FFMPEG_CLIP_ARGS_4K_DOWNSCALE)

    def test_downscale_args_use_tighter_crf(self):
        args = FFMPEG_CLIP_ARGS_4K_DOWNSCALE
        self.assertIn("-crf", args)
        crf_idx = args.index("-crf")
        self.assertEqual(args[crf_idx + 1], "20")

    def test_sdr_args_unchanged_at_crf_23(self):
        # Don't accidentally tighten the SDR path's CRF — 1080p sources
        # don't need it and the bigger file/encode-cost tradeoff isn't
        # justified at native resolution.
        args = FFMPEG_CLIP_ARGS
        crf_idx = args.index("-crf")
        self.assertEqual(args[crf_idx + 1], "23")

    def test_cache_version_bumped_to_v12(self):
        self.assertEqual(CACHE_VERSION, "v12")


class TestCfrTolerance(unittest.TestCase):
    """Near-CFR sources (HandBrake reports avg fps a hair below nominal) count
    as CFR so they skip the clip pipeline; true VFR still triggers clips."""

    def test_handbrake_near_cfr_counts_as_cfr(self):
        # Real ddt470.720: r=30, avg=2080658801/69461181 ≈ 29.954.
        self.assertTrue(_is_cfr("2080658801/69461181", "30/1"))

    def test_exact_rates_are_cfr(self):
        self.assertTrue(_is_cfr("30/1", "30/1"))
        self.assertTrue(_is_cfr("24000/1001", "24000/1001"))  # 23.976

    def test_true_vfr_is_not_cfr(self):
        self.assertFalse(_is_cfr("30/1", "15/1"))   # far apart
        self.assertFalse(_is_cfr("60/1", "30/1"))

    def test_unknown_or_empty_is_not_cfr(self):
        self.assertFalse(_is_cfr("", ""))
        self.assertFalse(_is_cfr("30/0", "30/1"))   # zero denominator


if __name__ == "__main__":
    unittest.main()
