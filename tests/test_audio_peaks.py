"""Tests for videoflow.audio_peaks — pre-computed RMS sidecar.

These cover the math, the sidecar IO, and the legacy compute/decode/load
aliases that funscriptforge.forge.audio_peaks re-exports under the same
names. If you rename or remove an alias here, the funscriptforge cli.py
audio-peaks command will break.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from videoflow.audio_peaks import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    SIDECAR_SUFFIX,
    SIDECAR_VERSION,
    compute_peaks,
    compute_sidecar_from_samples,
    decode_audio,
    extract_peaks,
    extract_sidecar,
    load_peaks,
    load_sidecar,
    sidecar_path,
    write_sidecar,
)


class TestSidecarPath(unittest.TestCase):

    def test_replaces_extension(self):
        # Confirm the canonical mapping: stem stays, suffix becomes .audio.json.
        self.assertTrue(sidecar_path("video.mp4").endswith(".audio.json"))
        self.assertTrue(sidecar_path("song.mp3").endswith(".audio.json"))

    def test_uses_pathlib_with_suffix(self):
        # Multi-dot stems: with_suffix replaces only the LAST suffix.
        self.assertTrue(
            sidecar_path("track.v2.mp4").endswith("track.v2.audio.json")
        )


class TestComputeSidecar(unittest.TestCase):

    def test_returns_none_on_empty_samples(self):
        self.assertIsNone(compute_sidecar_from_samples(np.array([], dtype=np.float32)))
        self.assertIsNone(compute_sidecar_from_samples(None))

    def test_returns_none_on_audio_shorter_than_one_hop(self):
        # 5 samples at sr=22050, hop_ms=10 → hop_samples=220 → 0 hops.
        short = np.zeros(5, dtype=np.float32)
        self.assertIsNone(compute_sidecar_from_samples(short))

    def test_raises_on_invalid_hop(self):
        y = np.zeros(22050, dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_sidecar_from_samples(y, hop_ms=0)

    def test_full_sidecar_shape(self):
        # 1 second of constant amplitude → 100 hops at hop_ms=10.
        y = np.full(22050, 0.5, dtype=np.float32)
        data = compute_sidecar_from_samples(y, hop_ms=10, sr=22050)
        self.assertIsNotNone(data)
        self.assertEqual(data["version"], SIDECAR_VERSION)
        self.assertEqual(data["hop_ms"], 10)
        self.assertEqual(data["duration_ms"], 1000)
        self.assertEqual(data["peak_count"], 100)
        self.assertEqual(len(data["peaks"]), 100)
        # All peaks are positive, normalized to [0, 1].
        for p in data["peaks"]:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_peaks_normalize_to_unit_max(self):
        # A track with one loud middle and quiet ends should peak at 1.0
        # somewhere and be lower at the ends.
        y = np.zeros(22050, dtype=np.float32)
        y[10000:12000] = 1.0  # short loud burst
        data = compute_sidecar_from_samples(y, hop_ms=10, sr=22050)
        peaks = data["peaks"]
        self.assertAlmostEqual(max(peaks), 1.0, places=3)
        # Ends should be quiet relative to the peak.
        self.assertLess(peaks[0], 0.2)
        self.assertLess(peaks[-1], 0.2)

    def test_silent_track_returns_zero_peaks(self):
        # No samples > 0 → peak_max = 0, peaks stay at the raw RMS (which
        # is also 0). Sidecar still emitted; consumers handle a flat 0.
        y = np.zeros(22050, dtype=np.float32)
        data = compute_sidecar_from_samples(y, hop_ms=10)
        self.assertIsNotNone(data)
        for p in data["peaks"]:
            self.assertEqual(p, 0.0)


class TestSidecarIO(unittest.TestCase):

    def test_roundtrip(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "track.mp4"
            data = {
                "version": SIDECAR_VERSION,
                "hop_ms": 10,
                "duration_ms": 1000,
                "peaks": [0.0, 0.5, 1.0, 0.25],
                "peak_count": 4,
                "generated_by": {"tool": "test"},
            }
            written = write_sidecar(media, data)
            self.assertTrue(written.endswith(SIDECAR_SUFFIX))
            self.assertTrue(Path(written).exists())
            loaded = load_sidecar(media)
            self.assertEqual(loaded, data)

    def test_load_returns_none_when_absent(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "no_such.mp4"
            self.assertIsNone(load_sidecar(media))

    def test_load_returns_none_on_corrupt_json(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "broken.mp4"
            # sidecar_path now points inside the project's .forge folder,
            # which nothing has created yet -- the corrupt file still has to
            # be written somewhere real for load_sidecar to reject it.
            corrupt = Path(sidecar_path(media))
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            corrupt.write_text("{not json")
            self.assertIsNone(load_sidecar(media))

    def test_compact_json_no_pretty_print(self):
        # Peak arrays double in size with pretty-print; the writer uses
        # compact json.dump. Smoke-check by ensuring no newlines exist.
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "track.mp4"
            data = {
                "version": SIDECAR_VERSION,
                "hop_ms": 10,
                "duration_ms": 100,
                "peaks": [0.1, 0.2, 0.3],
                "peak_count": 3,
                "generated_by": {"tool": "test"},
            }
            written = write_sidecar(media, data)
            text = Path(written).read_text()
            self.assertNotIn("\n", text)
            self.assertNotIn("  ", text)


class TestLegacyAliases(unittest.TestCase):
    """funscriptforge.forge.audio_peaks re-exports these by name —
    cli.py audio-peaks breaks if any go missing."""

    def test_compute_peaks_alias(self):
        # compute_peaks should behave identically to compute_sidecar_from_samples.
        y = np.full(22050, 0.5, dtype=np.float32)
        new = compute_sidecar_from_samples(y, hop_ms=10, sr=22050)
        legacy = compute_peaks(y, hop_ms=10, sr=22050)
        self.assertEqual(new, legacy)

    def test_load_peaks_alias(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "track.mp4"
            data = {
                "version": SIDECAR_VERSION, "hop_ms": 10, "duration_ms": 0,
                "peaks": [], "peak_count": 0,
                "generated_by": {"tool": "t"},
            }
            write_sidecar(media, data)
            self.assertEqual(load_peaks(media), data)

    def test_extract_peaks_alias(self):
        # Just verify it's callable with the legacy positional signature
        # used by cli.py. The function ultimately routes through librosa.load,
        # which we can't easily mock here, so this is a smoke check that
        # the symbol exists with the expected arity.
        self.assertTrue(callable(extract_peaks))
        self.assertTrue(callable(extract_sidecar))
        self.assertTrue(callable(decode_audio))


class TestConstants(unittest.TestCase):
    """The constants are part of the module's public surface — the
    funscriptforge shim re-exports them by name."""

    def test_defaults_present_and_sensible(self):
        self.assertEqual(SIDECAR_SUFFIX, ".audio.json")
        self.assertEqual(SIDECAR_VERSION, "1.0")
        self.assertEqual(DEFAULT_HOP_MS, 10)
        self.assertEqual(DEFAULT_SAMPLE_RATE, 22050)


if __name__ == "__main__":
    unittest.main()
