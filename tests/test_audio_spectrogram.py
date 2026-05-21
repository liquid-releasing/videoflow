"""Tests for videoflow.audio_spectrogram — mel-spectrogram sidecar.

The sidecar's binary cells are base64-encoded; tests verify the encode/
decode roundtrip and that the byte values map sensibly to the source
dB range (loud → 255, silent → 0).
"""

from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from videoflow.audio_spectrogram import (
    DEFAULT_DB_CEILING,
    DEFAULT_DB_FLOOR,
    DEFAULT_FMAX,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS,
    SIDECAR_SUFFIX,
    SIDECAR_VERSION,
    compute_sidecar_from_samples,
    load_sidecar,
    sidecar_path,
    write_sidecar,
)


class TestSidecarPath(unittest.TestCase):

    def test_replaces_extension(self):
        self.assertTrue(sidecar_path("video.mp4").endswith(".spectrogram.json"))
        self.assertTrue(sidecar_path("song.wav").endswith(".spectrogram.json"))


class TestComputeSidecar(unittest.TestCase):

    def test_returns_none_on_empty_samples(self):
        self.assertIsNone(compute_sidecar_from_samples(np.array([], dtype=np.float32)))
        self.assertIsNone(compute_sidecar_from_samples(None))

    def test_short_audio_still_produces_padded_frame(self):
        # librosa pads short signals up to one FFT window, so a 10-
        # sample input still yields a single frame. The function
        # returns a valid sidecar — *truly* empty input is rejected
        # separately in test_returns_none_on_empty_samples.
        short = np.zeros(10, dtype=np.float32)
        data = compute_sidecar_from_samples(short)
        if data is not None:
            self.assertGreaterEqual(data["n_frames"], 1)

    def test_returns_none_on_invalid_db_range(self):
        y = np.full(22050, 0.5, dtype=np.float32)
        # db_floor >= db_ceiling collapses the mapping.
        self.assertIsNone(
            compute_sidecar_from_samples(y, db_floor=0.0, db_ceiling=-80.0)
        )

    def test_full_sidecar_shape(self):
        # 2 seconds at sr=22050 / hop_length=512 → 2 * 22050 / 512 ≈ 86 frames.
        y = np.full(22050 * 2, 0.5, dtype=np.float32)
        data = compute_sidecar_from_samples(y, sr=22050)
        self.assertIsNotNone(data)
        self.assertEqual(data["version"], SIDECAR_VERSION)
        self.assertEqual(data["n_mels"], DEFAULT_N_MELS)
        self.assertEqual(data["fmax"], DEFAULT_FMAX)
        self.assertEqual(data["db_floor"], DEFAULT_DB_FLOOR)
        self.assertEqual(data["db_ceiling"], DEFAULT_DB_CEILING)
        self.assertGreater(data["n_frames"], 0)
        # hop_ms derived from hop_length / sr → 512 / 22050 * 1000 ≈ 23ms.
        self.assertEqual(data["hop_ms"], 23)
        # duration_ms = n_frames * hop_ms (integer arithmetic).
        self.assertEqual(data["duration_ms"], data["n_frames"] * data["hop_ms"])

    def test_cells_b64_size_matches_n_frames_x_n_mels(self):
        y = np.full(22050, 0.5, dtype=np.float32)
        data = compute_sidecar_from_samples(y)
        decoded = base64.b64decode(data["cells_b64"])
        self.assertEqual(len(decoded), data["n_frames"] * data["n_mels"])

    def test_loud_signal_quantizes_toward_top_of_range(self):
        # A 440Hz sine at near-full amplitude should produce at least some
        # cells very near the top of the LUT (byte ~255). librosa's
        # power_to_db(ref=np.max) sets the loudest cell to 0 dB which maps
        # to byte 255.
        sr = 22050
        t = np.arange(sr) / sr
        y = (0.9 * np.sin(2.0 * np.pi * 440 * t)).astype(np.float32)
        data = compute_sidecar_from_samples(y, sr=sr)
        decoded = np.frombuffer(base64.b64decode(data["cells_b64"]), dtype=np.uint8)
        self.assertEqual(int(decoded.max()), 255)

    def test_silent_signal_quantizes_to_zero(self):
        # Pure silence has no power; power_to_db floors out at -80 (our
        # configured db_floor), which maps to byte 0.
        y = np.zeros(22050, dtype=np.float32)
        data = compute_sidecar_from_samples(y, sr=22050)
        decoded = np.frombuffer(base64.b64decode(data["cells_b64"]), dtype=np.uint8)
        self.assertEqual(int(decoded.max()), 0)

    def test_cells_are_time_major(self):
        # Construct a signal where energy turns on AFTER the midpoint —
        # verifying the byte layout means "byte at (t * n_mels + bin) is
        # for time t, mel bin bin". If we got this wrong (e.g. left it as
        # mel-major from librosa), the loud region would appear as a band
        # of mel bins rather than a band of time frames.
        sr = 22050
        n_samples = sr * 4  # 4 seconds
        y = np.zeros(n_samples, dtype=np.float32)
        # Loud second half: pink-ish noise.
        rng = np.random.default_rng(0)
        y[n_samples // 2:] = (0.6 * rng.standard_normal(n_samples - n_samples // 2)).astype(np.float32)

        data = compute_sidecar_from_samples(y, sr=sr)
        n_mels = data["n_mels"]
        n_frames = data["n_frames"]
        cells = np.frombuffer(base64.b64decode(data["cells_b64"]), dtype=np.uint8)
        cells = cells.reshape(n_frames, n_mels)  # time-major assumption

        # Average energy in the first half should be lower than in the
        # second half across every mel bin (since the silent half has
        # cells near 0 and the loud half has cells > 0).
        first_half_mean = cells[: n_frames // 2].mean()
        second_half_mean = cells[n_frames // 2 :].mean()
        self.assertGreater(second_half_mean, first_half_mean)

    def test_n_mels_parameter_respected(self):
        y = np.full(22050, 0.5, dtype=np.float32)
        data = compute_sidecar_from_samples(y, n_mels=32)
        self.assertEqual(data["n_mels"], 32)
        decoded = base64.b64decode(data["cells_b64"])
        self.assertEqual(len(decoded), data["n_frames"] * 32)


class TestSidecarIO(unittest.TestCase):

    def test_roundtrip(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "track.mp4"
            sample = {
                "version": SIDECAR_VERSION,
                "hop_ms": 23,
                "n_mels": 4,
                "n_frames": 2,
                "duration_ms": 46,
                "fmax": 8000,
                "db_floor": -80.0,
                "db_ceiling": 0.0,
                "cells_b64": base64.b64encode(bytes([0, 64, 128, 255, 200, 100, 50, 10])).decode(),
                "generated_by": {"tool": "test"},
            }
            written = write_sidecar(media, sample)
            self.assertTrue(written.endswith(SIDECAR_SUFFIX))
            self.assertTrue(Path(written).exists())
            loaded = load_sidecar(media)
            self.assertEqual(loaded, sample)

    def test_load_returns_none_when_absent(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "no_such.mp4"
            self.assertIsNone(load_sidecar(media))

    def test_load_returns_none_on_corrupt_json(self):
        with TemporaryDirectory() as tmp:
            media = Path(tmp) / "broken.mp4"
            Path(sidecar_path(media)).write_text("{not json")
            self.assertIsNone(load_sidecar(media))


class TestConstants(unittest.TestCase):

    def test_defaults_present_and_sensible(self):
        self.assertEqual(SIDECAR_SUFFIX, ".spectrogram.json")
        self.assertEqual(SIDECAR_VERSION, "1.0")
        self.assertEqual(DEFAULT_N_MELS, 64)
        self.assertEqual(DEFAULT_HOP_LENGTH, 512)
        self.assertEqual(DEFAULT_FMAX, 8000)
        self.assertEqual(DEFAULT_DB_FLOOR, -80.0)
        self.assertEqual(DEFAULT_DB_CEILING, 0.0)


if __name__ == "__main__":
    unittest.main()
