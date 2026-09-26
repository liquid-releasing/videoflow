"""Atomic writes for files another process may be reading.

Why this exists — reported from funscriptforge dogfood, 2026-09-26:

    could not parse beats sidecar at …beats.json:
    EOF while parsing a value at line 1 column 0

Column 0 of line 1 means the file was ZERO BYTES. It was not corrupt: checked
on disk seconds later it was a valid 5487-byte sidecar. It was read mid-write.

``open(path, "w")`` and ``Path.write_text`` both **truncate the destination
before writing a single byte**. Between those two moments the file on disk is
empty, and it stays incomplete until the last flush. Any other process reading
it in that window sees garbage — and in this system another process is always
reading: the desktop app polls these sidecars while analysis produces them. The
window scales with the payload, so the 1.1 MB spectrogram is far more exposed
than the 5 KB beats file that happened to get caught.

The same truncation is why a **killed** run can leave a permanently broken
file. ``tempfiles.py`` makes this argument already: a ``finally`` never runs
when the process is killed, so the defence has to be structural rather than
cleanup code.

So: write the new content to a temporary file in the SAME directory, then
``os.replace`` it onto the destination. ``os.replace`` is atomic on both POSIX
and Windows, so a reader sees either the entire old file or the entire new one
and never a partial. Same directory matters — a cross-filesystem replace is a
copy, which is not atomic.

This repo already does exactly this for the two big binary artifacts:
``audio_cache`` stages the extracted WAV and replaces it (see its module
docstring), and ``chapter_clips`` does the same for each clip. The JSON
sidecars were simply never given the same treatment.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# How long to keep retrying a replace that Windows refuses because a reader
# currently has the destination open. Readers hold these files for
# milliseconds, so this is generous.
_REPLACE_TIMEOUT_S = 2.0


def _replace_with_retry(tmp: str, dest: Path, timeout_s: float) -> None:
    """``os.replace`` with a bounded retry, for Windows.

    ★ Found by this module's own concurrency test, not by reasoning.

    POSIX replaces a file happily while other processes have it open -- their
    handles keep pointing at the old inode. Windows does not: ``MoveFileEx``
    fails with ``PermissionError`` (WinError 5) if the DESTINATION is open
    without FILE_SHARE_DELETE, which is exactly how Python's ``open()`` opens
    it for reading.

    That matters more than it sounds. The whole point of this module is that
    another process is reading these files while we write them, so on the
    platform this app actually ships on, the contended case is the NORMAL case.
    Without a retry, making writes atomic would have traded an occasional bad
    read for an occasional hard write failure -- a worse bug, and one that
    would have shipped, because it cannot happen on the Linux box CI runs on.

    A reader opens, reads and closes in milliseconds, so retrying across a
    couple of seconds wins the gap. If it genuinely cannot, the error is
    raised: the destination still holds its previous, complete contents, and a
    silent fallback to a truncating write would resurrect the exact bug this
    module exists to prevent.
    """
    delay = 0.005
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    replace_timeout_s: float = _REPLACE_TIMEOUT_S,
) -> Path:
    """Write *text* to *path* so no reader ever observes a partial file.

    Creates the parent directory, writes to a sibling temporary file, flushes
    it to disk, then atomically replaces the destination.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A sibling of the destination, so `os.replace` stays on one filesystem and
    # therefore stays atomic. The leading dot keeps it out of the way of any
    # directory listing that filters hidden files mid-write.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".part")
    try:
        # Newline translation is left at the platform default so the bytes on
        # disk match what `open(p, "w")` / `Path.write_text` produced before.
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Best effort. Some filesystems refuse fsync; the replace below
                # is still atomic, we just lose the power-loss guarantee.
                pass
        _replace_with_retry(tmp, p, replace_timeout_s)
    except BaseException:
        # Includes KeyboardInterrupt and SystemExit on purpose: a cancelled run
        # must not leave `.part` litter beside the real file.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def write_json_atomic(
    path: str | Path, obj: Any, *, replace_timeout_s: float = _REPLACE_TIMEOUT_S,
    **dumps_kwargs: Any,
) -> Path:
    """``json.dumps`` *obj* and write it atomically. Keyword arguments are
    passed straight through to ``json.dumps``, so callers keep control of
    ``indent``, ``separators`` and ``ensure_ascii``."""
    return write_text_atomic(
        path, json.dumps(obj, **dumps_kwargs), replace_timeout_s=replace_timeout_s,
    )
