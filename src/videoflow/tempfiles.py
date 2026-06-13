"""Dedicated temp directory for forge audio extractions, with an orphan sweep.

Why this exists: video analysis extracts the full audio track to a temp WAV
(100 MB–1 GB), loads it, and unlinks it in a ``finally``. That works on the
normal path — but a ``finally`` never runs when the process is **killed**
(cancel, window-close, crash), so killed runs strand multi-hundred-MB WAVs
that silently fill the user's disk over time. No amount of cleanup code
survives a kill.

The robust fix is to defend at the boundary, not the creation site:

1. Route every extraction into one known subdir (:func:`audio_temp_dir`).
2. On each analysis entry point, :func:`sweep_audio_temp` reclaims stale
   files there — orphans from prior killed runs, however they were stranded.

Because the sweep only ever touches our own dedicated subdir and only files
older than ``max_age_s`` (default 1 h, so it can't race a concurrent
in-progress extraction), it is safe to call liberally.

This is the durable half of the temp-leak release gateway; the other half is
a regression test asserting the normal path leaves no orphan.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

_SUBDIR = "forge-audio"


def audio_temp_dir() -> Path:
    """Return (creating if needed) the dedicated temp dir for audio
    extractions: ``<system-temp>/forge-audio``. Pass as ``dir=`` to
    ``NamedTemporaryFile`` so every extraction lands somewhere the sweep
    can safely reclaim."""
    d = Path(tempfile.gettempdir()) / _SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def sweep_audio_temp(max_age_s: float = 3600.0) -> tuple[int, int]:
    """Delete files in the forge-audio temp dir older than ``max_age_s``.

    Reclaims orphans left by killed processes. Only touches our own
    subdir, only files past the age threshold (so an in-progress
    extraction from a concurrent run — seconds old — is never removed).
    Never raises; returns ``(files_removed, bytes_freed)``.
    """
    removed = 0
    freed = 0
    base = Path(tempfile.gettempdir()) / _SUBDIR
    try:
        if not base.is_dir():
            return (0, 0)
        now = time.time()
        for p in base.iterdir():
            try:
                st = p.stat()
            except OSError:
                continue
            if not p.is_file():
                continue
            if now - st.st_mtime >= max_age_s:
                try:
                    p.unlink()
                    removed += 1
                    freed += st.st_size
                except OSError:
                    continue
    except OSError:
        pass
    return (removed, freed)
