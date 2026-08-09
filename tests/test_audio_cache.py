"""Tests for the shared extracted-audio cache (bug D22).

Generation and chapter analysis used to each run their own ffmpeg extraction
of the same source, so building a funscript and landing on Analysis paid for
the decode twice. These lock the cache that makes it once:

- identity: same source + same sample rate hits; a different rate, an edited
  source, or a bumped cache version all miss,
- atomicity: a killed extraction cannot publish a truncated WAV that a later
  run would trust,
- metadata: the corrupt-audio ``damaged_after_ms`` marker survives a hit, so
  the "audio is damaged after MM:SS" warning doesn't vanish on the cached path,
- ownership: a published slot is recognisable as cache-owned so the callers'
  cleanup ``finally`` leaves it alone (deleting it would defeat the cache).
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from videoflow import audio_cache
from videoflow.tempfiles import audio_temp_dir


def _make_source(tmpdir: Path, name: str = "clip.mp4", data: bytes = b"video-bytes") -> Path:
    p = tmpdir / name
    p.write_bytes(data)
    return p


class _CacheTestBase(unittest.TestCase):
    """Cleans up any cache slots a test publishes.

    The cache lives in the real shared ``forge-audio`` dir (that is the point
    — it must be reachable across processes), so tests track what they create
    and remove it rather than relying on the age-based sweep.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self._created: list[Path] = []

    def tearDown(self):
        for p in self._created:
            Path(p).unlink(missing_ok=True)
            Path(p).with_suffix(".json").unlink(missing_ok=True)
        self._tmp.cleanup()

    def _publish(self, src: Path, sr: int, payload: bytes = b"RIFFwav", **kw) -> str:
        staging = audio_cache.new_staging_path()
        Path(staging).write_bytes(payload)
        out = audio_cache.publish(staging, src, sr, **kw)
        self._created.append(Path(out))
        return out


class TestCacheKey(_CacheTestBase):

    def test_same_source_and_rate_is_stable(self):
        src = _make_source(self.tmpdir)
        self.assertEqual(
            audio_cache.cache_key(src, 22050),
            audio_cache.cache_key(src, 22050),
        )

    def test_sample_rate_is_part_of_the_key(self):
        src = _make_source(self.tmpdir)
        self.assertNotEqual(
            audio_cache.cache_key(src, 22050),
            audio_cache.cache_key(src, 44100),
        )

    def test_edited_source_invalidates(self):
        """A re-encode changes size/mtime, so the old extraction is not reused."""
        src = _make_source(self.tmpdir)
        before = audio_cache.cache_key(src, 22050)
        src.write_bytes(b"different-video-bytes-entirely")
        os.utime(src, (0, 0))  # force a distinct mtime even on coarse clocks
        self.assertNotEqual(before, audio_cache.cache_key(src, 22050))

    def test_missing_source_is_unkeyable_not_an_error(self):
        missing = self.tmpdir / "nope.mp4"
        self.assertEqual(audio_cache.cache_key(missing, 22050), "")
        self.assertIsNone(audio_cache.cached_wav_path(missing, 22050))
        self.assertIsNone(audio_cache.lookup(missing, 22050))


class TestPublishAndLookup(_CacheTestBase):

    def test_miss_before_publish_hit_after(self):
        src = _make_source(self.tmpdir)
        self.assertIsNone(audio_cache.lookup(src, 22050))
        out = self._publish(src, 22050)
        hit = audio_cache.lookup(src, 22050)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], out)

    def test_publish_is_atomic_move_staging_is_gone(self):
        src = _make_source(self.tmpdir)
        staging = audio_cache.new_staging_path()
        Path(staging).write_bytes(b"RIFFwav")
        out = audio_cache.publish(staging, src, 22050)
        self._created.append(Path(out))
        self.assertFalse(Path(staging).exists(), "staging file should be moved, not copied")
        self.assertEqual(Path(out).read_bytes(), b"RIFFwav")

    def test_unpublished_staging_is_not_a_cache_hit(self):
        """A killed extraction leaves scratch behind; it must never be trusted."""
        src = _make_source(self.tmpdir)
        staging = audio_cache.new_staging_path()
        self._created.append(Path(staging))
        Path(staging).write_bytes(b"truncated")
        self.assertIsNone(audio_cache.lookup(src, 22050))

    def test_empty_slot_is_a_miss(self):
        src = _make_source(self.tmpdir)
        slot = audio_cache.cached_wav_path(src, 22050)
        self._created.append(slot)
        slot.write_bytes(b"")
        self.assertIsNone(audio_cache.lookup(src, 22050))

    def test_different_rate_does_not_hit_the_other_slot(self):
        src = _make_source(self.tmpdir)
        self._publish(src, 22050)
        self.assertIsNotNone(audio_cache.lookup(src, 22050))
        self.assertIsNone(audio_cache.lookup(src, 44100))


class TestDamagedMetadata(_CacheTestBase):

    def test_damaged_after_ms_survives_a_cache_hit(self):
        src = _make_source(self.tmpdir)
        self._publish(src, 22050, damaged_after_ms=1_294_000)
        hit = audio_cache.lookup(src, 22050)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], 1_294_000)

    def test_clean_source_reports_none(self):
        src = _make_source(self.tmpdir)
        self._publish(src, 22050)
        self.assertIsNone(audio_cache.lookup(src, 22050)[1])

    def test_corrupt_meta_still_serves_the_wav(self):
        """The WAV is the artifact; a bad meta file must not force a re-decode."""
        src = _make_source(self.tmpdir)
        out = self._publish(src, 22050, damaged_after_ms=5000)
        Path(out).with_suffix(".json").write_text("{not json", encoding="utf-8")
        hit = audio_cache.lookup(src, 22050)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], out)
        self.assertIsNone(hit[1])

    def test_meta_records_provenance(self):
        src = _make_source(self.tmpdir)
        out = self._publish(src, 22050, damaged_after_ms=None)
        meta = json.loads(Path(out).with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(meta["sr"], 22050)
        self.assertEqual(meta["cache_version"], audio_cache.CACHE_VERSION)


class TestOwnership(_CacheTestBase):

    def test_published_slot_is_cache_owned(self):
        """Callers key their cleanup off this — a false negative deletes the cache."""
        src = _make_source(self.tmpdir)
        out = self._publish(src, 22050)
        self.assertTrue(audio_cache.is_cached_path(out))

    def test_scratch_and_foreign_paths_are_not_cache_owned(self):
        staging = audio_cache.new_staging_path()
        self._created.append(Path(staging))
        self.assertFalse(audio_cache.is_cached_path(staging))
        self.assertFalse(audio_cache.is_cached_path(self.tmpdir / "elsewhere.wav"))
        self.assertFalse(audio_cache.is_cached_path(None))

    def test_slot_lives_in_the_swept_temp_dir(self):
        """Lifetime is delegated to the existing forge-audio orphan sweep."""
        src = _make_source(self.tmpdir)
        out = self._publish(src, 22050)
        self.assertEqual(Path(out).parent, audio_temp_dir())


class TestCrossStageReuse(_CacheTestBase):
    """The actual D22 regression: two stages, one decode.

    Counts real ffmpeg invocations across the generation path
    (``audio.analyze_beats``' extractor) and the analysis path
    (``structural._prepare_audio``) to prove the second stage reuses the
    first's work instead of re-extracting.
    """

    def _patched_ffmpeg(self, calls):
        def fake(_ffmpeg, _input, out_path, *, sr, progress, **kw):
            calls.append(out_path)
            Path(out_path).write_bytes(b"RIFF" + b"\0" * 64)
            return (0, None)
        return fake

    def test_second_stage_reuses_the_first_extraction(self):
        import videoflow.structural as structural
        src = _make_source(self.tmpdir, "scene.mp4")
        calls: list[str] = []
        orig = structural._run_ffmpeg_with_progress
        structural._run_ffmpeg_with_progress = self._patched_ffmpeg(calls)
        try:
            first, tmp1 = structural._prepare_audio(src, sr=22050, progress=lambda _m: None)
            self._created.append(Path(first))
            second, tmp2 = structural._prepare_audio(src, sr=22050, progress=lambda _m: None)
        finally:
            structural._run_ffmpeg_with_progress = orig

        self.assertEqual(len(calls), 1, "the second pass must not re-run ffmpeg")
        self.assertEqual(first, second, "both passes should read the same cached wav")
        self.assertIsNone(tmp1, "a cached extraction is not the caller's to unlink")
        self.assertIsNone(tmp2)
        self.assertTrue(Path(second).is_file(), "cache slot must survive the first run")

    def test_cache_slot_survives_a_callers_cleanup_contract(self):
        """``_prepare_audio`` returning ``None`` is what stops the finally deleting it."""
        import videoflow.structural as structural
        src = _make_source(self.tmpdir, "scene2.mp4")
        calls: list[str] = []
        orig = structural._run_ffmpeg_with_progress
        structural._run_ffmpeg_with_progress = self._patched_ffmpeg(calls)
        try:
            path, tmp = structural._prepare_audio(src, sr=22050, progress=lambda _m: None)
            self._created.append(Path(path))
        finally:
            structural._run_ffmpeg_with_progress = orig

        # Emulate auto_chapter's cleanup, which unlinks only a non-None tmp.
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)
        self.assertTrue(Path(path).is_file())
        self.assertIsNotNone(audio_cache.lookup(src, 22050))

    def test_a_different_source_still_extracts(self):
        import videoflow.structural as structural
        a = _make_source(self.tmpdir, "a.mp4", b"aaaa")
        b = _make_source(self.tmpdir, "b.mp4", b"bbbbbb")
        calls: list[str] = []
        orig = structural._run_ffmpeg_with_progress
        structural._run_ffmpeg_with_progress = self._patched_ffmpeg(calls)
        try:
            pa, _ = structural._prepare_audio(a, sr=22050, progress=lambda _m: None)
            pb, _ = structural._prepare_audio(b, sr=22050, progress=lambda _m: None)
            self._created.extend([Path(pa), Path(pb)])
        finally:
            structural._run_ffmpeg_with_progress = orig

        self.assertEqual(len(calls), 2)
        self.assertNotEqual(pa, pb)

    def test_audio_file_input_is_untouched_by_the_cache(self):
        """Non-video sources are read in place — no extraction, no cache slot."""
        import videoflow.structural as structural
        wav = _make_source(self.tmpdir, "already.wav", b"RIFF")
        path, tmp = structural._prepare_audio(wav, sr=22050, progress=lambda _m: None)
        self.assertEqual(path, str(wav))
        self.assertIsNone(tmp)


if __name__ == "__main__":
    unittest.main()
