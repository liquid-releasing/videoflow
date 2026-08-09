"""Shared extracted-audio cache, so one source video is decoded to WAV once.

Why this exists (bug D22): generation and chapter analysis each extracted the
*same* audio track from the *same* video with their own ffmpeg run. Build a
funscript on the Generate tab and land on Analysis, and the app spent minutes
re-extracting audio it had just finished extracting ("Extracting audio…
40:38 done" twice in a row). The two stages persist different artifacts —
generation writes the full ``<stem>.beatmap.json``, ``auto_chapter`` writes its
own reduced peaks/spectrogram/beats sidecars — so neither one's freshness check
could ever see the other's work.

The fix is one level below both of them: cache the *extraction* itself, which
is the expensive part they genuinely share, and leave every downstream
detection semantic untouched. Chapter analysis still runs its own chapter-
windowed beat pass; it just no longer pays for a second decode to do it.

Design notes:

* **Keyed by content identity, not by name.** The key hashes the resolved
  source path + mtime_ns + size + target sample rate, so re-encoding or
  swapping a file invalidates it, and two sample rates never collide.
* **Lives in the existing ``forge-audio`` temp dir**, not in the project's
  ``.forge/`` folder. That deliberately reuses :func:`sweep_audio_temp` as the
  reclamation policy — the cache inherits the same one-hour orphan sweep that
  already protects the user's disk, so this adds no permanent storage and no
  new cleanup rules. A generate→analyze handoff happens in seconds; an
  analysis an hour later just pays for a fresh extract, exactly as today.
* **Atomic publish.** ffmpeg writes to a random temp name and the result is
  ``os.replace``\\ d into the deterministic slot. A killed run therefore
  strands a *partial* file under the random name (which the sweep reclaims)
  and can never publish a truncated WAV that a later run would trust.
* **Metadata rides along.** ``damaged_after_ms`` from the corrupt-audio
  salvage is recorded in a sidecar JSON next to the WAV, so a cache hit still
  surfaces the "this source's audio is damaged after MM:SS" warning instead of
  silently dropping it.

See ``internal/beta_test_bugs.md`` D22 in the funscriptforge repo for the
original dogfood report and the three fix options considered.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from videoflow.tempfiles import audio_temp_dir

# Bump when the extraction's *output* format changes (sample format, channel
# layout, filters applied during extraction) so stale WAVs from an older
# videoflow can't be reused under the new semantics. The sample RATE is part
# of the key itself and does not need a bump.
CACHE_VERSION = 1

_PREFIX = "fa"


def cache_key(media_path: str | Path, sr: int) -> str:
    """Stable short key for *media_path* decoded at *sr*.

    Hashes the resolved path, mtime_ns and size alongside the sample rate and
    :data:`CACHE_VERSION`. Any edit/re-encode of the source changes mtime+size
    and therefore the key, so a stale extraction is never reused.

    Returns a marker key when the source cannot be stat'd; callers treat that
    as a cache miss rather than raising, since an unreadable source will fail
    louder in ffmpeg a moment later.
    """
    p = Path(media_path)
    try:
        st = p.stat()
        ident = f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}|{int(sr)}|{CACHE_VERSION}"
    except OSError:
        return ""
    return hashlib.sha256(ident.encode("utf-8", "replace")).hexdigest()[:16]


def cached_wav_path(media_path: str | Path, sr: int) -> Path | None:
    """Deterministic slot for the cached WAV, or ``None`` if unkeyable."""
    key = cache_key(media_path, sr)
    if not key:
        return None
    return audio_temp_dir() / f"{_PREFIX}-{key}.wav"


def _meta_path(wav: Path) -> Path:
    return wav.with_suffix(".json")


def lookup(media_path: str | Path, sr: int) -> tuple[str, int | None] | None:
    """Return ``(wav_path, damaged_after_ms)`` on a cache hit, else ``None``.

    A hit also *touches* the WAV so that an actively-reused extraction keeps
    outrunning the temp sweep's age threshold instead of being reclaimed out
    from under a working session.
    """
    wav = cached_wav_path(media_path, sr)
    if wav is None:
        return None
    try:
        if not wav.is_file() or wav.stat().st_size == 0:
            return None
    except OSError:
        return None

    damaged_after_ms = None
    try:
        meta = json.loads(_meta_path(wav).read_text(encoding="utf-8"))
        raw = meta.get("damaged_after_ms")
        damaged_after_ms = int(raw) if raw is not None else None
    except (OSError, ValueError, TypeError):
        # A missing/corrupt meta file is not a reason to re-extract 40 minutes
        # of audio — the WAV is the artifact. Worst case we lose the damaged
        # warning for this run.
        damaged_after_ms = None

    try:
        os.utime(wav, None)
        _meta_path(wav).touch(exist_ok=True)
    except OSError:
        pass

    return (str(wav), damaged_after_ms)


def new_staging_path() -> str:
    """Path for ffmpeg to write into before the atomic publish.

    Lands in the same dedicated temp dir as the cache slot (so ``os.replace``
    stays a same-filesystem rename, and so the sweep reclaims it if the run is
    killed mid-extraction).
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=str(audio_temp_dir()),
    )
    tmp.close()
    return tmp.name


def publish(
    staging_path: str | Path,
    media_path: str | Path,
    sr: int,
    *,
    damaged_after_ms: int | None = None,
) -> str:
    """Atomically move a finished extraction into its cache slot.

    Returns the path callers should read from: the cache slot on success, or
    the original *staging_path* if publishing was not possible (unkeyable
    source, cross-device rename, permissions). Never raises — a cache that
    cannot be written must degrade to "works, just slower", never to a failed
    analysis.
    """
    staging = Path(staging_path)
    wav = cached_wav_path(media_path, sr)
    if wav is None:
        return str(staging)
    try:
        os.replace(staging, wav)
    except OSError:
        return str(staging)

    try:
        _meta_path(wav).write_text(
            json.dumps({
                "damaged_after_ms": damaged_after_ms,
                "sr": int(sr),
                "cache_version": CACHE_VERSION,
                "source": str(Path(media_path)),
            }),
            encoding="utf-8",
        )
    except OSError:
        pass  # the WAV is what matters; meta is best-effort
    return str(wav)


def is_cached_path(path: str | Path | None) -> bool:
    """True when *path* is a published cache slot rather than a scratch temp.

    Call sites use this to decide whether their cleanup ``finally`` should
    unlink the file they were handed. Cache slots are owned by the temp sweep
    and must survive the run that created them — that survival is the entire
    point of the cache.
    """
    if not path:
        return False
    p = Path(path)
    try:
        return p.parent == audio_temp_dir() and p.name.startswith(f"{_PREFIX}-")
    except OSError:
        return False
