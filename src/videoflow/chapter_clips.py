"""Chapter clip extraction — stream-quality slice of a chapter to a small file.

Pre-builds the per-chapter clips that funscriptforge's MediaViewer plays
during editing. Doing this once during the (already-long) ``auto_chapter``
analyze pass means chapter selection in the editor feels instant —
the clip is already on disk when the user clicks the band.

The funscriptforge Tauri command ``extract_chapter_clip`` mirrors this
function. Both write to ``%TEMP%/funscriptforge_clips/`` with the same
deterministic naming so either side's output satisfies the other side's
cache check. The Rust command is the fallback path for projects that
were analyzed before this stage existed.

ffmpeg args are pinned in :data:`FFMPEG_CLIP_ARGS` so the two paths stay
in sync. Update both when the args change, and bump
:data:`CACHE_VERSION` so older clips age out.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Cache version. Mirrors the Rust constant in
# ``funscriptforge/ui/web/src-tauri/src/commands.rs::extract_chapter_clip``.
# Bump when the ffmpeg args below change so cached clips from the prior
# version don't get served.
#
# v11: AAC restored + `-avoid_negative_ts make_zero` added. The
# PIPELINE_ERROR_DECODE failure under both v9 (AAC) and v10 (MP3) was
# not codec-related — it was negative timestamps on the first audio
# packet caused by `-ss` before `-i` fast-seeking to the nearest video
# keyframe. Chromium rejects negative-TS packets; `make_zero` shifts
# the whole stream so the smallest TS lands at 0.
CACHE_VERSION = "v11"

# ffmpeg encode args. Match the Rust command verbatim — frontend's Rust
# fallback must produce identical output for a given (media, start, end).
#
# Video: baseline H.264 profile + level 3.1 + constant 30fps. Chose this
# combo after WebView2 stutter testing on long high-bitrate sources
# (Victoria Oaks 90min, Angel Anjelica 18GB). The source's H.264 profile
# / B-frame structure / variable-frame-rate was the WebView2 stutter
# trigger; this re-encode normalises every clip to a profile the
# embedded Chromium plays cleanly.
#
# Audio: 48 kHz AAC stereo. Chromium's audio output runs at 48 kHz
# internally; matching the source rate avoids the per-chunk resampler.
#
# `-avoid_negative_ts make_zero` forces output timestamps to start at
# zero. With `-ss` before `-i` ffmpeg fast-seeks to the nearest video
# keyframe, which can leave the audio stream's first PTS negative.
# Chromium rejects negative-TS packets with PIPELINE_ERROR_DECODE
# (2026-05-22 dogfood: same failure under both AAC and MP3, ruling
# out the codec). `make_zero` shifts every stream so the smallest TS
# becomes 0 — audio and video stay aligned, packets stay positive.
FFMPEG_CLIP_ARGS: tuple[str, ...] = (
    "-c:v", "libx264",
    "-profile:v", "baseline",
    "-level", "3.1",
    "-preset", "ultrafast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-r", "30",
    "-c:a", "aac",
    "-b:a", "192k",
    "-ar", "48000",
    "-ac", "2",
    "-avoid_negative_ts", "make_zero",
    "-movflags", "+faststart",
    # Explicit output container; mirrors the Rust args. Defense against
    # any future tmp-name change confusing ffmpeg's extension guesser.
    "-f", "mp4",
    "-y",
)

# Default extension when the source doesn't carry one. mp4 is what
# Chromium plays best and what every editor input is converted to.
_DEFAULT_EXT = "mp4"

# Filename character allow-list. Anything outside this gets replaced
# with an underscore so paths from arbitrary user filenames produce a
# safe filesystem name across Windows / macOS / Linux.
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _sanitize_stem(stem: str) -> str:
    """Reduce a filename stem to ``[A-Za-z0-9_.-]``-only characters."""
    return _SAFE_STEM_RE.sub("_", stem).strip("._") or "media"


def chapter_clips_dir(media_path: str | Path) -> Path:
    """Return the chapter-clip cache directory for *media_path*.

    Lives inside the per-project hidden forge dir as
    ``<dir>/.<stem>.forge/clips/``. Caller is responsible for mkdir'ing
    before writing (:func:`extract_chapter_clip` does this).
    """
    from videoflow.sidecar import forge_dir
    return forge_dir(media_path) / "clips"


def chapter_clip_path(media_path: str | Path, start_ms: int, end_ms: int) -> Path:
    """Compute the deterministic cache path for one chapter clip.

    Naming scheme: ``<sanitized_stem>_<CACHE_VERSION>_<start_ms>_<end_ms>.<ext>``

    Hash-free so the funscriptforge Rust command and this Python helper
    produce the same path without sharing a hash implementation. Stem
    sanitization collapses unusual characters; collision on the same
    sanitized stem + same chapter bounds means the same clip content
    (cache hit either way).
    """
    src = Path(media_path)
    ext = src.suffix.lstrip(".").lower() or _DEFAULT_EXT
    stem = _sanitize_stem(src.stem)
    name = f"{stem}_{CACHE_VERSION}_{int(start_ms)}_{int(end_ms)}.{ext}"
    return chapter_clips_dir(media_path) / name


def extract_chapter_clip(
    media_path: str | Path,
    start_ms: int,
    end_ms: int,
    output_path: str | Path | None = None,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Stream-quality re-encode of a chapter slice.

    Args:
        media_path: Source media file.
        start_ms: Slice start in milliseconds.
        end_ms: Slice end in milliseconds (must exceed ``start_ms``).
        output_path: Where to write the clip. Defaults to the cache
            path :func:`chapter_clip_path` computes.
        ffmpeg: Override the ffmpeg binary path. Defaults to the
            :mod:`videoflow.chapters` lookup (PATH + adjacent binary).

    Returns:
        The output path. If a clip already exists there, returns the
        path without re-running ffmpeg.

    Raises:
        ValueError: If ``end_ms <= start_ms``.
        RuntimeError: If ffmpeg exits non-zero (stderr surfaced).
        FileNotFoundError: If ffmpeg can't be found.
    """
    if end_ms <= start_ms:
        raise ValueError(
            f"extract_chapter_clip: end_ms ({end_ms}) must exceed start_ms ({start_ms})"
        )

    import os

    out = Path(output_path) if output_path is not None else chapter_clip_path(
        media_path, start_ms, end_ms,
    )
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Local import to avoid a hard cycle in videoflow's package init.
    # ``chapters._find_ffmpeg`` already encapsulates the PATH-or-adjacent
    # lookup pattern used everywhere else in videoflow.
    if ffmpeg is None:
        from videoflow.chapters import _find_ffmpeg
        ffmpeg = _find_ffmpeg()

    # Write to a process-scoped tmp filename, then atomic-rename to the
    # final path on success. Without this, this Python pipeline and the
    # funscriptforge Rust command (which targets the same final path)
    # can race when the user clicks a chapter while Analyze is running
    # — two ffmpeg processes interleave their writes and produce a
    # structurally-corrupt MP4 (duplicated MOOV atom, garbled H.264
    # NAL units; 2026-05-22 dogfood). Atomic rename gives a
    # winner-takes-all semantic: if another writer finishes first our
    # tmp is dropped and we keep theirs.
    #
    # Slot the pid between the stem and the real extension so the file
    # still ends in `.mp4` — ffmpeg auto-detects the output format from
    # the trailing extension, and `<final>.mp4.tmp.<pid>` made it bail
    # with "Unable to choose an output format" (2026-05-22 dogfood).
    tmp = out.parent / f"{out.stem}.tmp.{os.getpid()}{out.suffix}"

    args: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_ms / 1000:.3f}",
        "-to", f"{end_ms / 1000:.3f}",
        "-i", str(media_path),
        *FFMPEG_CLIP_ARGS,
        str(tmp),
    ]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0:
        # Don't leave a partial / zero-byte file behind to poison the cache.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg extract failed (exit {proc.returncode}): {stderr}"
        )

    # Atomic publish. If another process beat us to the final path,
    # drop our tmp and serve theirs. On Windows `os.replace` will
    # overwrite, but a race where two processes both call replace can
    # interleave the rename target; the existence pre-check + post-check
    # bounds the damage.
    if out.exists():
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return out
    try:
        os.replace(tmp, out)
    except OSError:
        if out.exists():
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return out
        raise
    return out


def extract_chapter_clips(
    media_path: str | Path,
    chapters: Iterable[tuple[int, int]],
    *,
    ffmpeg: str | None = None,
    on_clip: "Callable[[int, int, Path, bool], None] | None" = None,  # noqa: F821 (forward type)
) -> list[Path]:
    """Extract clips for every chapter in turn. Returns the list of paths.

    Args:
        media_path: Source media file.
        chapters: Iterable of ``(start_ms, end_ms)`` tuples.
        ffmpeg: Override the ffmpeg binary path.
        on_clip: Optional callback invoked per chapter as
            ``on_clip(start_ms, end_ms, path, cached)``. Used by the
            ``auto_chapter`` pipeline to report per-chapter progress.
    """
    paths: list[Path] = []
    for start_ms, end_ms in chapters:
        out = chapter_clip_path(media_path, start_ms, end_ms)
        cached = out.exists()
        if not cached:
            extract_chapter_clip(
                media_path, start_ms, end_ms, out, ffmpeg=ffmpeg,
            )
        paths.append(out)
        if on_clip is not None:
            try:
                on_clip(start_ms, end_ms, out, cached)
            except Exception:
                pass
    return paths
