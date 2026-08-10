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

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable


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
#
# v12: Conditional 720p downscale for sources wider than 1920px.
# Extraction time on 4K sources drops ~10x (multi-GB clip → ~100MB at
# 720p) and the WebView2 OOM cliff at multi-GB blobs goes away. CRF
# tightened to 20 (vs 23) because lanczos downsampling cleans up the
# codec input; explicit BT.709 color flags preserve graded-source
# iris coloring through the re-encode. HDR sources (BT.2020) get
# tonemap-less conversion to BT.709 — acceptable for editor preview
# since the funscript output is colour-agnostic. SDR (≤1920px) keeps
# the v11 args verbatim.
CACHE_VERSION = "v12"

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


# Encode args for sources wider than 1920px (4K and above). Lanczos
# downscale to 1280x720 + explicit BT.709 color flags. Chapter clips
# are editor previews — 720p is plenty for "click chapter → watch it
# play to decide what to edit," and the size/decode win pays off
# everywhere: extraction time, WebView2 memory, disk space.
#
# CRF 20 (vs the SDR path's 23) because lanczos downsampling cleans
# noise out of the codec input, so we can afford a tighter quantizer
# without paying file-size cost. Color flags are critical for graded
# sources — without them ffmpeg drops color metadata on re-encode and
# iris coloring reads desaturated next to the source.
#
# HDR / BT.2020 sources: tonemap-less conversion to SDR BT.709.
# Acceptable for editor preview (the funscript output ignores pixel
# colour); proper HDR tonemapping would require zscale and a heavier
# pipeline. See [[project-funscriptforge-pending]] 4K downsize block.
FFMPEG_CLIP_ARGS_4K_DOWNSCALE: tuple[str, ...] = (
    "-c:v", "libx264",
    "-profile:v", "baseline",
    "-level", "3.1",
    "-preset", "ultrafast",
    "-crf", "20",
    "-pix_fmt", "yuv420p",
    "-r", "30",
    "-vf", "scale=1280:720:flags=lanczos",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-color_range", "tv",
    "-c:a", "aac",
    "-b:a", "192k",
    "-ar", "48000",
    "-ac", "2",
    "-avoid_negative_ts", "make_zero",
    "-movflags", "+faststart",
    "-f", "mp4",
    "-y",
)

# Sources wider than this trigger the 4K downscale path. 1920 catches
# 4K (3840), 5K (5120), 8K (7680) and any odd resolution between
# 1080p and 4K (e.g. 2560-wide 1440p). 1080p sources keep the v11
# args verbatim — they don't need the downscale.
DOWNSCALE_WIDTH_THRESHOLD: int = 1920

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


# Match ffmpeg's stream-info line. ffmpeg prints "Stream #N:M(...) :
# Video: codec ..., pix_fmt, WIDTHxHEIGHT [bracketed metadata]" to
# stderr for each video stream. The first numeric WxH on a Video
# line is the source resolution; later WxH on the same line are SAR
# / display dimensions that we don't care about.
_PROBE_RESOLUTION_RE = re.compile(
    r"Stream #\d+:\d+[^\n]*?Video:[^\n]*?\b(\d{3,5})x(\d{3,5})\b"
)


def _probe_video_dimensions(
    media_path: str | Path, ffmpeg: str,
) -> tuple[int, int] | None:
    """Probe ``(width, height)`` of the source's first video stream.

    Runs ``ffmpeg -i <path>`` with no output specified. ffmpeg prints
    stream info to stderr and exits 1 because there's no output file —
    that's expected. Returns ``None`` if the probe couldn't find a
    Video stream line (audio-only file, unreadable format, ffmpeg not
    found, etc.); callers fall back to the safe default encode args
    on ``None``.
    """
    args = [ffmpeg, "-hide_banner", "-loglevel", "info", "-i", str(media_path)]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, creationflags=_NO_WINDOW,
        )
    except (FileNotFoundError, OSError):
        return None
    match = _PROBE_RESOLUTION_RE.search(proc.stderr or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _fps_to_float(fr: str) -> float:
    """Parse an ffprobe frame-rate ('30/1', '2080658801/69461181', '29.97') to
    float; 0.0 on anything unparseable or a zero denominator."""
    try:
        if "/" in fr:
            n, d = fr.split("/", 1)
            d = float(d)
            return float(n) / d if d else 0.0
        return float(fr)
    except (ValueError, ZeroDivisionError):
        return 0.0


# Near-CFR tolerance: HandBrake / x264 report avg_frame_rate a hair below the
# nominal r_frame_rate (avg = frames ÷ duration), e.g. 29.955 vs 30 — that's
# effectively CFR and plays fine. True VFR diverges far more. 1% of nominal.
# MUST stay in sync with funscriptforge cli.py::_is_cfr.
_CFR_FPS_TOLERANCE = 0.01


def _is_cfr(avg: str, rrate: str) -> bool:
    a, r = _fps_to_float(avg), _fps_to_float(rrate)
    return a > 0 and r > 0 and abs(a - r) <= _CFR_FPS_TOLERANCE * r


def is_direct_playable(media_path: str | Path) -> bool:
    """True when the source streams cleanly into an embedded Chromium <video>
    via asset:// WITHOUT a normalizing re-encode — so the per-chapter clip
    pre-build can be SKIPPED entirely (single-file streaming fast path).

    Mirrors FunscriptForge's ``cli.py::_verdict_direct_playable`` (the same
    criteria drive the editor's playback-time decision in useChapterClip.js);
    keep the two in sync. The clip pipeline exists for sources that STUTTER /
    OOM played whole — long high-bitrate / 4K / VFR / HDR / >8-bit / exotic
    profile — so we only skip clips when confident WebView2 handles the source
    cleanly. CONSERVATIVE: any probe failure or any disqualifier returns False
    (→ extract clips as before), because a false "playable" costs the user a
    stuttering preview while a false "not playable" only costs a clip.

    Audio-only files are not direct-*video*-playable in this sense; callers
    already skip clip extraction for non-video suffixes, so this is only
    consulted for video sources.
    """
    import json as _json
    import shutil

    # Locate ffprobe: alongside the bundled ffmpeg first, else PATH.
    ffprobe = "ffprobe"
    try:
        from videoflow.chapters import _find_ffmpeg
        ff = _find_ffmpeg()
        cand = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if cand.exists():
            ffprobe = str(cand)
        elif shutil.which("ffprobe"):
            ffprobe = shutil.which("ffprobe")
    except Exception:
        if shutil.which("ffprobe"):
            ffprobe = shutil.which("ffprobe")

    try:
        proc = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_streams", str(media_path)],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW,
        )
        data = _json.loads(proc.stdout or "{}")
    except Exception:
        return False  # can't tell → extract clips (safe default)

    vstream = next(
        (s for s in (data.get("streams") or []) if s.get("codec_type") == "video"),
        None,
    )
    if vstream is None:
        return False

    codec = (vstream.get("codec_name") or "").lower()
    pix_fmt = (vstream.get("pix_fmt") or "").lower()
    profile = (vstream.get("profile") or "").lower()
    width = vstream.get("width") or 0
    transfer = (vstream.get("color_transfer") or "").lower()
    primaries = (vstream.get("color_primaries") or "").lower()
    avg = vstream.get("avg_frame_rate") or ""
    rrate = vstream.get("r_frame_rate") or ""

    if codec != "h264":
        return False
    if pix_fmt and pix_fmt != "yuv420p":
        return False
    if any(tok in profile for tok in ("10", "422", "444")):
        return False
    if width and width > 1920:
        return False
    if transfer in ("smpte2084", "arib-std-b67") or primaries == "bt2020":
        return False
    if not _is_cfr(avg, rrate):  # true VFR (or unknown) → not safe; near-CFR OK
        return False
    return True


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


def _run_ffmpeg_clip(
    args: list[str],
    *,
    duration_ms: int,
    on_progress: "Callable[[float], None] | None",
) -> subprocess.CompletedProcess:
    """Run a clip encode, reporting fractional progress while it works.

    A chapter clip is a full re-encode, so on a 2.5K source a 7-minute chapter
    is minutes of ffmpeg. With a plain blocking ``subprocess.run`` the caller
    can only announce "extracting clip 3/3…" once and then go silent for the
    whole encode, which reads as a frozen app (bug D32).

    Progress goes to a temp FILE rather than ``pipe:1`` deliberately: ffmpeg
    block-flushes stdout pipes and delivers nothing until exit, whereas the
    file path flushes incrementally. Same technique as
    ``structural._run_ffmpeg_with_progress``, which documents the finding.

    ``on_progress`` receives a 0.0–1.0 fraction, throttled to whole-percent
    changes so a long encode doesn't flood the consumer's event stream.
    Progress reporting is strictly best-effort — any failure inside it leaves
    the encode itself untouched.
    """
    if on_progress is None or duration_ms <= 0:
        return subprocess.run(
            args, capture_output=True, text=True, creationflags=_NO_WINDOW,
        )

    fd, progress_path = tempfile.mkstemp(suffix=".log", prefix="vfclip_")
    os.close(fd)
    # `-progress <file> -nostats` goes before the output path.
    cmd = args[:-1] + ["-progress", progress_path, "-nostats", args[-1]]

    done = threading.Event()

    def _poll():
        last_pct = -1
        pos = 0
        while not done.is_set():
            try:
                with open(progress_path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                for line in chunk.splitlines():
                    if not line.startswith("out_time_us="):
                        continue
                    raw = line.split("=", 1)[1].strip()
                    if not raw or raw == "N/A":
                        continue
                    frac = (int(raw) / 1000.0) / duration_ms
                    frac = max(0.0, min(1.0, frac))
                    pct = int(frac * 100)
                    if pct != last_pct:
                        last_pct = pct
                        try:
                            on_progress(frac)
                        except Exception:
                            # A consumer's reporting bug must not silently kill
                            # the poll loop (and with it all further progress).
                            pass
            except (OSError, ValueError):
                pass
            done.wait(0.5)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=_NO_WINDOW,
        )
    finally:
        done.set()
        poller.join(timeout=2.0)
        try:
            os.unlink(progress_path)
        except OSError:
            pass
    return proc


def extract_chapter_clip(
    media_path: str | Path,
    start_ms: int,
    end_ms: int,
    output_path: str | Path | None = None,
    *,
    ffmpeg: str | None = None,
    on_progress: "Callable[[float], None] | None" = None,
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
        on_progress: Optional callback receiving a 0.0–1.0 fraction as the
            encode advances. A chapter clip can take minutes on a high-res
            source, so callers use this to keep the UI honest instead of
            sitting silent for the whole encode (D32). Best-effort: it never
            affects the encode's success.

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

    # Pick encode args based on source resolution. Probe is cheap
    # (ffmpeg reads the header only) and runs once per clip. A failed
    # probe (None) falls through to the SDR args — safer to keep the
    # original 4K bytes than to corrupt a clip from an unreadable
    # source.
    dims = _probe_video_dimensions(media_path, ffmpeg)
    if dims is not None and dims[0] > DOWNSCALE_WIDTH_THRESHOLD:
        encode_args: tuple[str, ...] = FFMPEG_CLIP_ARGS_4K_DOWNSCALE
    else:
        encode_args = FFMPEG_CLIP_ARGS

    args: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_ms / 1000:.3f}",
        "-to", f"{end_ms / 1000:.3f}",
        "-i", str(media_path),
        *encode_args,
        str(tmp),
    ]
    proc = _run_ffmpeg_clip(
        args, duration_ms=end_ms - start_ms, on_progress=on_progress,
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
