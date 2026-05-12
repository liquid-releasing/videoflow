"""Audio-structure auto-detection — find natural chapter boundaries.

The audio-structure primitive that lets every lqr tool work against the
natural shape of long-form material instead of treating audio as one
undifferentiated stream. Reads silence, recurrence patterns, and energy
transitions to propose chapter boundaries; downstream tools consume the
same `Chapter` records produced here.

See ``videoflow/docs/architecture/audio-structure-primitive.md`` for
the architectural rationale and the consumer matrix.

Example::

    from videoflow.structural import auto_chapter

    chapters = auto_chapter(
        "track.mp4",
        target_minutes=5.5,
        progress_callback=print,
    )
    for ch in chapters:
        print(f"{ch.at_ms}-{ch.end_ms}ms  {ch.content_type}  conf={ch.confidence}")
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from videoflow.audio import analyze_beats as _analyze_beats
from videoflow.chapters import Chapter
from videoflow.phrases import Phrase, classify_phrases as _classify_phrases
from videoflow.progress import OnProgress, ProgressReporter
from videoflow.sidecar import write_sidecar as _write_sidecar

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

try:
    import librosa as _librosa  # type: ignore[import]
    import numpy as _np  # type: ignore[import]
except ImportError:
    _librosa = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]


# Default target chapter length in minutes. Picked from the bench: long
# enough that per-chapter analysis converges (≥ ~3 min for stable
# classify_modes), short enough that progress events feel responsive
# (chapter completion every few seconds during analysis even on long
# files). Tunable per-call.
DEFAULT_TARGET_MINUTES = 5.5

# Below this fraction of the target, a chapter is considered too small
# to be useful and is merged into its smaller neighbor. Boundary-snap
# can occasionally pull two cuts close together when adjacent silence
# regions exist; this cleans that up.
_MIN_CHAPTER_FRACTION = 0.4

# Minimum duration to bother chunking. Below this, return one whole-file
# chapter — the segmentation cost isn't worth it and the per-segment
# benefits of `analyze_beats(chapters=...)` only show up on multi-chunk
# files.
_MIN_CHUNK_DURATION_MIN = 8.0

# Silence-detection parameters. RMS threshold relative to the file's
# 95th-percentile RMS. Long enough to skip foley/breath gaps but short
# enough to catch real scene-change pauses.
_SILENCE_RMS_RATIO = 0.05
_SILENCE_MIN_DURATION_S = 1.5

# Snap each agglomerative-clustering boundary to the nearest silence
# region within this radius. Keeps boundaries on natural transitions.
_BOUNDARY_SNAP_RADIUS_S = 8.0


class AutoChapterError(RuntimeError):
    """Raised when auto-chapter detection fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_chapter(
    media: str | Path,
    *,
    target_minutes: float = DEFAULT_TARGET_MINUTES,
    progress_callback: Callable[[str], None] | None = None,
    on_progress: OnProgress | None = None,
    write_sidecar: bool = True,
    force: bool = False,
    sr: int = 22050,
) -> list[Chapter]:
    """Detect natural chapter boundaries in a media file.

    Audio-only path — extracts audio from video files via ffmpeg, then
    runs silence detection + self-similarity clustering to find chunk
    boundaries that respect the content's natural structure. Each
    chapter carries a content-type heuristic (``"music"``, ``"ambient"``,
    ``"mixed"``) and a confidence score for downstream consumers.

    Files shorter than 1.5× the target duration return a single
    whole-file chapter. Above that, the function aims for chunks of
    *target_minutes* on average; exact lengths vary because boundaries
    snap to silence and recurrence-cluster edges rather than wall-clock
    intervals.

    Args:
        media: Path to the source media (video or audio).
        target_minutes: Average target chapter length in minutes.
            Default 5.5; tuneable per-call.
        progress_callback: Called with stage labels as the analysis
            progresses (e.g. ``"Loading audio (librosa)…"``). Errors
            in the callback are swallowed — progress UI never breaks
            the analysis.
        write_sidecar: When ``True`` (default), merges the result into
            ``<stem>.chapters.json`` next to *media* via
            :func:`videoflow.sidecar.write_sidecar` in ``analyze`` mode.
            User edits are preserved per-record through the field-level
            merge contract — see
            ``docs/architecture/audio-structure-primitive.md``.
        force: Deprecated. Retained for back-compat with v0.0.4 callers.
            The merge model in :mod:`videoflow.sidecar` preserves user
            data automatically, so this flag is a no-op as of v0.0.5.
        sr: Sample rate for analysis. 22050 is the librosa standard
            for beat / structure work; lower values speed analysis on
            multi-hour files at the cost of fine-grain resolution.

    Returns:
        Ordered list of :class:`videoflow.chapters.Chapter`. The list
        is non-empty even for very short files (single whole-file
        chapter) unless analysis itself fails.

    Raises:
        FileNotFoundError: If *media* does not exist.
        AutoChapterError: If librosa is not installed or analysis
            fails.
    """
    media_path = Path(media)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    if _librosa is None:
        raise AutoChapterError(
            "librosa is required for audio-structure analysis. "
            'Install it with: pip install "videoflow[audio]"'
        )

    reporter = ProgressReporter(on_progress)

    def _progress(label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass
        reporter.message(label)

    chapters: list[Chapter] = []
    beat_map = None
    phrases: list[Phrase] = []

    with reporter.stage("structural.auto_chapter"):
        with reporter.stage("extract"):
            audio_path, _tmp = _prepare_audio(media_path, sr=sr, progress=_progress)
            reporter.complete(
                summary="audio extracted to wav" if _tmp else "no extraction needed",
            )
        try:
            with reporter.stage("load"):
                _progress("Loading audio (librosa)…")
                y, sr_ = _librosa.load(audio_path, sr=sr, mono=True)
                duration_ms = int(round(_librosa.get_duration(y=y, sr=sr_) * 1000))
                reporter.complete(
                    summary=f"{duration_ms / 1000:.1f}s @ {sr_} Hz mono",
                )

            if duration_ms < int(_MIN_CHUNK_DURATION_MIN * 60_000):
                with reporter.stage("whole_file"):
                    _progress("Track too short to chunk — returning single chapter")
                    chapters = [
                        _whole_file_chapter(y, sr_, duration_ms),
                    ]
                    reporter.complete(summary="1 chapter (whole file)")
            else:
                with reporter.stage("detect"):
                    chapters = _detect_chapters(
                        y, sr_, duration_ms,
                        target_minutes=target_minutes,
                        progress=_progress,
                    )
                    reporter.complete(
                        summary=f"{len(chapters)} chapters detected",
                    )

            with reporter.stage("beats"):
                _progress("Analysing beats and energy…")
                beat_map = _analyze_beats(
                    audio_path, sr=sr_,
                    chapters=chapters,
                    on_progress=on_progress,
                )
                reporter.complete(
                    summary=(
                        f"{len(beat_map.beats)} beats, "
                        f"{len(beat_map.phrases)} phrases @ "
                        f"{beat_map.bpm:.1f} BPM"
                    )
                )

            with reporter.stage("classify"):
                _progress("Classifying phrases…")
                phrases = _classify_phrases(beat_map, chapters=chapters)
                reporter.complete(
                    summary=f"{len(phrases)} phrases classified",
                )
        finally:
            if _tmp is not None:
                Path(_tmp).unlink(missing_ok=True)

        if write_sidecar:
            with reporter.stage("sidecar"):
                _progress("Writing sidecar…")
                payload: dict = {
                    "chapters": [c.to_dict() for c in chapters],
                    "phrases": [p.to_dict() for p in phrases],
                    "generated_by": {
                        "tool": "videoflow.structural",
                        "tool_version": _videoflow_version(),
                        "target_minutes": target_minutes,
                    },
                }
                if beat_map is not None:
                    payload["energy"] = _build_energy(beat_map, chapters)
                _write_sidecar(
                    media_path, payload,
                    writer="videoflow.structural",
                    writer_version=_videoflow_version(),
                    mode="analyze",
                )
                reporter.complete(
                    summary=(
                        f"sidecar merged: {len(chapters)} chapters, "
                        f"{len(payload['phrases'])} phrases"
                    ),
                )

        reporter.complete(summary=f"{len(chapters)} chapters")

    return chapters


# ---------------------------------------------------------------------------
# Audio extraction (mirrors videoflow.audio so behaviour is consistent)
# ---------------------------------------------------------------------------

def _prepare_audio(
    media_path: Path,
    *,
    sr: int,
    progress: Callable[[str], None],
) -> tuple[str, str | None]:
    """Return ``(load_path, tmp_path_or_None)``.

    For audio files, returns the source path directly. For video files,
    extracts a mono WAV via ffmpeg and returns the temp path; caller is
    responsible for unlinking it.
    """
    if media_path.suffix.lower() not in _VIDEO_SUFFIXES:
        return str(media_path), None

    progress("Extracting audio from video (ffmpeg)…")
    ffmpeg = "ffmpeg"
    for candidate in (
        Path(__file__).parent / "ffmpeg.exe",
        Path(__file__).parent / "ffmpeg",
        media_path.parent / "ffmpeg.exe",
        media_path.parent / "ffmpeg",
    ):
        if candidate.is_file():
            ffmpeg = str(candidate)
            break

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        rc = _run_ffmpeg_with_progress(
            ffmpeg, media_path, tmp.name, sr=sr, progress=progress,
        )
        if rc != 0:
            Path(tmp.name).unlink(missing_ok=True)
            raise AutoChapterError(f"FFmpeg audio extraction failed (exit {rc})")
    except FileNotFoundError:
        Path(tmp.name).unlink(missing_ok=True)
        raise AutoChapterError(
            "FFmpeg is required to extract audio from video files. "
            "Install it from https://ffmpeg.org/download.html — "
            "or place ffmpeg.exe alongside videoflow."
        )
    return tmp.name, tmp.name


def _format_extract_timestamp(out_time_us: int) -> str:
    """Format ffmpeg's ``out_time_us`` (microseconds of decoded audio) as
    ``m:ss`` or ``h:mm:ss``. Negative or invalid values clamp to zero."""
    seconds = max(0, out_time_us // 1_000_000)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _parse_ffmpeg_progress_line(line: str) -> str | None:
    """Return a Stepper-friendly progress label for one ffmpeg ``-progress``
    line, or None if the line carries no decode-position info we can show."""
    line = line.strip()
    if not line.startswith("out_time_us="):
        return None
    try:
        us = int(line.split("=", 1)[1])
    except ValueError:
        return None
    return f"Extracting audio… {_format_extract_timestamp(us)} done"


def _run_ffmpeg_with_progress(
    ffmpeg: str,
    media_path: Path,
    out_path: str,
    *,
    sr: int,
    progress: Callable[[str], None],
) -> int:
    """Run ffmpeg → mono WAV, emitting sub-stage progress lines like
    ``Extracting audio… 0:23 done`` while it works. Returns the exit code.

    Uses ``-progress <file>`` rather than ``-progress pipe:1`` because
    ffmpeg only block-flushes stdout pipes on exit (verified — pipe-mode
    delivers nothing until the process closes), whereas the file path
    flushes incrementally as expected. A background thread polls the
    file every 500 ms and streams new lines through ``progress``."""
    import os
    import threading
    import time

    fd, progress_path = tempfile.mkstemp(suffix=".log", prefix="vfprog_")
    os.close(fd)

    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-i", str(media_path),
            "-vn", "-ar", str(sr), "-ac", "1",
            "-f", "wav",
            "-progress", progress_path, "-nostats",
            out_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
    )

    stop = threading.Event()
    last_label: list[str | None] = [None]

    def _drain_once() -> None:
        try:
            text = Path(progress_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        for raw in text.splitlines():
            label = _parse_ffmpeg_progress_line(raw)
            if label and label != last_label[0]:
                last_label[0] = label
                progress(label)

    def _poll() -> None:
        while not stop.wait(0.5):
            _drain_once()

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    try:
        rc = proc.wait()
    finally:
        stop.set()
        poll_thread.join(timeout=2.0)
        # Final drain catches anything written between the last poll and exit.
        _drain_once()
        try:
            Path(progress_path).unlink()
        except OSError:
            pass
    return rc


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def _whole_file_chapter(y, sr, duration_ms: int) -> Chapter:
    """Return a single chapter spanning the entire file."""
    content_type, confidence, evidence = _classify_content(y, sr)
    return Chapter(
        at_ms=0,
        end_ms=duration_ms,
        content_type=content_type,
        confidence=confidence,
        evidence=evidence,
    )


def _detect_chapters(
    y,
    sr: int,
    duration_ms: int,
    *,
    target_minutes: float,
    progress: Callable[[str], None],
) -> list[Chapter]:
    """Run the chapter-detection pipeline on a loaded audio buffer."""
    target_seconds = target_minutes * 60.0

    progress("Computing silence map…")
    silence_centers_s = _silence_breakpoints(y, sr)

    progress("Computing recurrence matrix…")
    boundaries_s = _segment_boundaries(
        y, sr, target_seconds=target_seconds, duration_s=duration_ms / 1000.0,
    )

    progress("Snapping boundaries to silence…")
    snapped = _snap_to_silence(
        boundaries_s, silence_centers_s, radius_s=_BOUNDARY_SNAP_RADIUS_S,
    )

    # Always start at 0 and end at duration; ensure ordered + unique.
    duration_s = duration_ms / 1000.0
    cuts_s = sorted({0.0, duration_s, *snapped})
    if cuts_s[-1] > duration_s:
        cuts_s[-1] = duration_s

    # Merge micro-chapters into neighbors so boundary-snap can't produce
    # 30-second slivers when adjacent silence regions are close together.
    cuts_s = _merge_micro_chapters(
        cuts_s,
        min_duration_s=target_seconds * _MIN_CHAPTER_FRACTION,
    )

    chapters: list[Chapter] = []
    for i in range(len(cuts_s) - 1):
        start_s = cuts_s[i]
        end_s = cuts_s[i + 1]
        progress(f"Classifying chapter {i + 1}/{len(cuts_s) - 1}…")
        content_type, confidence, evidence = _classify_content(
            _slice_audio(y, sr, start_s, end_s), sr,
        )
        chapters.append(Chapter(
            at_ms=int(round(start_s * 1000)),
            end_ms=int(round(end_s * 1000)),
            content_type=content_type,
            confidence=confidence,
            evidence=evidence,
        ))
    return chapters


def _silence_breakpoints(y, sr: int) -> list[float]:
    """Return centers (seconds) of silence regions ≥ ``_SILENCE_MIN_DURATION_S``.

    Uses RMS thresholded against the file's 95th-percentile RMS to handle
    quiet ambient material as well as loud-music tracks.
    """
    hop = 512
    rms = _librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size == 0:
        return []
    ref = _np.percentile(rms, 95.0)
    if ref <= 0:
        return []
    threshold = ref * _SILENCE_RMS_RATIO
    silent = rms < threshold

    frame_duration = hop / sr
    min_frames = max(1, int(_SILENCE_MIN_DURATION_S / frame_duration))

    centers: list[float] = []
    run_start: int | None = None
    for i, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            run_len = i - run_start
            if run_len >= min_frames:
                centers.append(((run_start + i) / 2.0) * frame_duration)
            run_start = None
    if run_start is not None:
        run_len = len(silent) - run_start
        if run_len >= min_frames:
            centers.append(((run_start + len(silent)) / 2.0) * frame_duration)
    return centers


def _segment_boundaries(
    y,
    sr: int,
    *,
    target_seconds: float,
    duration_s: float,
) -> list[float]:
    """Return interior chapter boundaries (seconds) via librosa segmentation.

    Uses MFCC features + agglomerative clustering. ``num_segments`` is
    chosen so segments are roughly *target_seconds* on average; librosa
    picks the actual cut locations from the recurrence structure.
    """
    num_segments = max(2, int(round(duration_s / target_seconds)))

    # MFCCs are a strong content-similarity signal — same instruments
    # and timbre cluster, scene changes break the cluster.
    mfcc = _librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Agglomerative clustering returns frame indices of segment starts.
    boundary_frames = _librosa.segment.agglomerative(mfcc, num_segments)
    times = _librosa.frames_to_time(boundary_frames, sr=sr)
    # Drop frame 0 (always returned as first boundary) — caller adds 0
    # explicitly; we return interior cuts only.
    return [float(t) for t in times if t > 0.0]


def _snap_to_silence(
    boundaries_s: list[float],
    silence_centers_s: list[float],
    *,
    radius_s: float,
) -> list[float]:
    """For each boundary, snap to the nearest silence within ``radius_s``."""
    if not silence_centers_s:
        return list(boundaries_s)
    sil = sorted(silence_centers_s)
    snapped: list[float] = []
    for b in boundaries_s:
        # Linear search is fine — silence regions are sparse on long-form.
        nearest = min(sil, key=lambda s: abs(s - b))
        snapped.append(nearest if abs(nearest - b) <= radius_s else b)
    return snapped


def _merge_micro_chapters(
    cuts_s: list[float],
    *,
    min_duration_s: float,
) -> list[float]:
    """Drop interior cuts that produce sub-``min_duration_s`` chapters.

    Walks the cut list left-to-right; when a chapter is too short, drops
    the cut that *ends* it (merging into the next chapter). The first
    and last cuts (0 and duration) are always preserved.
    """
    if len(cuts_s) <= 2:
        return list(cuts_s)
    kept = [cuts_s[0]]
    for i in range(1, len(cuts_s) - 1):
        if cuts_s[i] - kept[-1] >= min_duration_s:
            kept.append(cuts_s[i])
    # Tail check: if the final chapter ended up too short, drop the
    # second-to-last cut (merge tail into the previous chapter).
    if len(kept) >= 2 and cuts_s[-1] - kept[-1] < min_duration_s:
        kept.pop()
    kept.append(cuts_s[-1])
    return kept


# ---------------------------------------------------------------------------
# Content-type heuristic
# ---------------------------------------------------------------------------

def _slice_audio(y, sr: int, start_s: float, end_s: float):
    """Return ``y[start_s:end_s]`` in samples."""
    a = max(0, int(round(start_s * sr)))
    b = min(len(y), int(round(end_s * sr)))
    return y[a:b]


def _classify_content(y, sr: int) -> tuple[str, float, list[str]]:
    """Crude content-type heuristic: music / ambient / mixed.

    Looks at:
    - **rms_variance** — high in music (kick + snare swings), low in
      ambient steady-state material
    - **percussive_ratio** — fraction of energy in the percussive HPSS
      component; high in music, low in ambient
    - **spectral_flux** — frame-to-frame spectral change; high in music
      with clear transients

    Returns ``(content_type, confidence, evidence_list)``. Confidence
    is a rough scalar in ``[0.5, 1.0]`` reflecting how strongly the
    features cluster on one type vs. the others — meant for sorting,
    not for hard thresholding.
    """
    if len(y) < sr:  # < 1 second — can't classify
        return "", 0.0, []

    rms = _librosa.feature.rms(y=y)[0]
    rms_variance = float(_np.var(rms))
    rms_mean = float(_np.mean(rms))
    # Coefficient of variation — scale-invariant, so quiet ambient and
    # loud ambient both register as "low variation."
    rms_cv = rms_variance**0.5 / rms_mean if rms_mean > 0 else 0.0

    # Percussive energy ratio via HPSS.
    try:
        _, perc = _librosa.effects.hpss(y, margin=2.0)
        perc_energy = float(_np.mean(_librosa.feature.rms(y=perc)[0]))
        full_energy = max(rms_mean, 1e-9)
        perc_ratio = float(min(1.0, perc_energy / full_energy))
    except Exception:
        perc_ratio = 0.0

    # Spectral flux: sum of positive frame-to-frame magnitude changes.
    spec = _np.abs(_librosa.stft(y))
    flux = _np.diff(spec, axis=1)
    flux_pos = _np.maximum(flux, 0.0).sum(axis=0)
    flux_score = float(_np.mean(flux_pos))
    flux_score_norm = float(min(1.0, flux_score / (rms_mean * 100.0 + 1e-9)))

    evidence: list[str] = []
    music_score = 0.0
    ambient_score = 0.0

    # Music indicators: high CV + high percussive ratio + high flux.
    if rms_cv > 0.6:
        music_score += 0.35
        evidence.append("rms_variance")
    if perc_ratio > 0.45:
        music_score += 0.35
        evidence.append("percussive_ratio")
    if flux_score_norm > 0.35:
        music_score += 0.30
        evidence.append("spectral_flux")

    # Ambient indicators: low CV + low percussive ratio + low flux.
    if rms_cv < 0.3:
        ambient_score += 0.35
    if perc_ratio < 0.25:
        ambient_score += 0.35
    if flux_score_norm < 0.15:
        ambient_score += 0.30

    if music_score >= 0.6 and music_score > ambient_score + 0.2:
        return "music", round(min(1.0, 0.5 + music_score / 2.0), 3), evidence
    if ambient_score >= 0.6 and ambient_score > music_score + 0.2:
        if not evidence:
            evidence = ["rms_variance", "percussive_ratio"]
        return "ambient", round(min(1.0, 0.5 + ambient_score / 2.0), 3), evidence
    return "mixed", round(0.5 + abs(music_score - ambient_score) / 4.0, 3), (
        evidence or ["rms_variance"]
    )


# ---------------------------------------------------------------------------
# Energy payload builder (consumed by write_sidecar in auto_chapter).
# Phrase building lives in videoflow.phrases as part of classify_phrases.
# ---------------------------------------------------------------------------

def _build_energy(beat_map, chapters: list[Chapter]) -> dict:
    """Build the sidecar ``energy`` block from a chapter-aware AudioBeatMap.

    Emits whole-file ``percentiles``, the inline ``beat_map`` (times,
    strengths, and per-beat ``is_downbeat`` flags so downstream tools
    can phase-lock without re-analysing audio), and a ``per_chapter``
    map keyed by chapter index with its own percentile statistics, BPM
    derived from beat density in the chapter window, and content_type
    when present.

    The ``envelope`` field (sampled RMS over time) is not produced here
    — videoflow.audio doesn't expose it. Schema permits absence.
    """
    energies = list(beat_map.energy)
    beats_ms = list(beat_map.beats)
    downbeats_ms = list(getattr(beat_map, "downbeats", None) or [])
    duration_ms = int(beat_map.duration_ms)

    out: dict = {}

    if energies:
        sorted_e = sorted(energies)
        out["percentiles"] = {
            f"p{p}": round(_percentile(sorted_e, p / 100.0), 6)
            for p in (5, 25, 50, 75, 95)
        }

    if beats_ms:
        downbeats_set = {int(t) for t in downbeats_ms}
        out["beat_map"] = {
            "times_ms": [int(t) for t in beats_ms],
            "strengths": [round(float(e), 6) for e in energies],
            "is_downbeat": [int(t) in downbeats_set for t in beats_ms],
        }

    per_chapter: dict[str, dict] = {}
    for i, ch in enumerate(chapters):
        ch_end = ch.end_ms if ch.end_ms is not None else duration_ms
        in_range = [
            (b, e) for b, e in zip(beats_ms, energies)
            if ch.at_ms <= b < ch_end
        ]
        if not in_range:
            continue
        ch_sorted = sorted(e for _, e in in_range)
        ch_dur_s = max(0, min(ch_end, duration_ms) - ch.at_ms) / 1000.0
        bpm = (60.0 * len(in_range) / ch_dur_s) if ch_dur_s > 0 else None
        entry: dict = {
            "percentiles": {
                f"p{p}": round(_percentile(ch_sorted, p / 100.0), 6)
                for p in (5, 25, 50, 75, 95)
            },
            "bpm": round(bpm, 2) if bpm is not None else None,
        }
        if ch.content_type:
            entry["content_type"] = ch.content_type
        per_chapter[str(i)] = entry

    if per_chapter:
        out["per_chapter"] = per_chapter

    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list. Empty → 0.0."""
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


# ---------------------------------------------------------------------------
# Version helper (sidecar IO migrated to videoflow.sidecar in 0.0.5)
# ---------------------------------------------------------------------------

def _videoflow_version() -> str:
    try:
        from importlib.metadata import version as _v
        return _v("videoflow")
    except Exception:
        return "unknown"
