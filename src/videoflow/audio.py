"""Audio beat analysis — librosa wrapper returning a rich AudioBeatMap."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from videoflow.progress import OnProgress, ProgressReporter

_nullcontext = contextlib.nullcontext

if TYPE_CHECKING:
    from videoflow.chapters import Chapter

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# Optional dependency — imported at module level so tests can patch it.
try:
    import librosa as _librosa  # type: ignore[import]
    import numpy as _np  # type: ignore[import]
except ImportError:
    _librosa = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]


class BeatError(RuntimeError):
    """Raised when beat analysis fails."""


@dataclasses.dataclass
class AudioBeatMap:
    """Rich beat-analysis result — one pass, many consumers.

    All timestamps are in milliseconds.
    """

    bpm: float
    """Detected tempo in beats per minute."""

    beats: list[int]
    """Timestamp (ms) of every detected beat."""

    downbeats: list[int]
    """Timestamp (ms) of every downbeat (first beat of each measure).

    V1 assumes 4/4 time — every 4th beat.
    """

    stanzas: list[tuple[int, int]]
    """(start_ms, end_ms) of each musical stanza.

    V1 groups every 16 beats (4 bars of 4/4).
    """

    energy: list[float]
    """Normalised RMS energy (0.0–1.0) at each beat timestamp.

    Useful for ranking which beats have the most drive — peaks tend to fall
    on kick drums and snare hits.
    """

    duration_ms: int
    """Total audio duration in milliseconds."""

    @property
    def beat_interval_ms(self) -> float:
        """Average interval between beats in milliseconds."""
        return 60_000.0 / self.bpm

    def beats_in_range(self, start_ms: int, end_ms: int) -> list[int]:
        """Return beat timestamps that fall within [start_ms, end_ms)."""
        return [b for b in self.beats if start_ms <= b < end_ms]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return {
            "bpm": self.bpm,
            "duration_ms": self.duration_ms,
            "beats": self.beats,
            "downbeats": self.downbeats,
            "stanzas": [{"start_ms": s, "end_ms": e} for s, e in self.stanzas],
            "energy": [round(e, 6) for e in self.energy],
        }

    def save(self, path: str | Path) -> Path:
        """Save the beat map to a JSON file.

        The saved file can be reloaded with :meth:`load` — no need to
        re-run librosa on subsequent renders.

        Args:
            path: Destination ``.json`` file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AudioBeatMap":
        """Load a beat map previously saved with :meth:`save`.

        Args:
            path: Path to the ``.json`` file.

        Returns:
            Reconstructed :class:`AudioBeatMap`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            BeatError: If the file is missing required fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Beat map file not found: {path}")
        try:
            data = json.loads(path.read_text())
            return cls(
                bpm=float(data["bpm"]),
                duration_ms=int(data["duration_ms"]),
                beats=[int(b) for b in data["beats"]],
                downbeats=[int(b) for b in data["downbeats"]],
                stanzas=[(int(p["start_ms"]), int(p["end_ms"])) for p in data["stanzas"]],
                energy=[float(e) for e in data["energy"]],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BeatError(f"Invalid beat map file {path}: {exc}") from exc

    def nearest_beat(self, ms: int, *, direction: str = "nearest") -> int:
        """Return the beat timestamp closest to *ms*.

        Args:
            ms: Target time in milliseconds.
            direction: ``"nearest"`` (default), ``"before"``, or ``"after"``.

        Raises:
            ValueError: If *direction* is not a recognised value.
            BeatError: If the beat map contains no beats.
        """
        if not self.beats:
            raise BeatError("Beat map contains no beats.")
        if direction not in ("nearest", "before", "after"):
            raise ValueError(
                f"direction must be 'nearest', 'before', or 'after'; got {direction!r}"
            )

        before = [b for b in self.beats if b <= ms]
        after = [b for b in self.beats if b > ms]

        if direction == "before":
            return before[-1] if before else self.beats[0]
        if direction == "after":
            return after[0] if after else self.beats[-1]

        # nearest
        candidates = []
        if before:
            candidates.append(before[-1])
        if after:
            candidates.append(after[0])
        return min(candidates, key=lambda b: abs(b - ms))


_SOURCES = ("full", "percussive")
_TRACKERS = ("auto", "beat_track", "plp")

# Above this duration, "auto" tracker switches from librosa.beat.beat_track
# (single global tempo, dynamic-programming, drifts on long-form material)
# to librosa.beat.plp (locally-stable tempo, robust over multi-hour tracks).
PLP_AUTO_DURATION_MS = 10 * 60 * 1000  # 10 minutes

# Default width of a synthetic time chunk when `chunk_secs` is requested
# without an explicit value. ~3-minute windows give a 2-hour file ~40
# progress steps ("chunk 4/40") — enough to prove the analysis is moving,
# few enough that beat tracking still has a meaningful span to work with.
DEFAULT_CHUNK_SECS = 180.0
# Trailing remainder shorter than this fraction of a full chunk is merged
# into the final chunk rather than analysed as a runt (a 10s tail tracks
# badly and adds a useless progress step).
_CHUNK_MERGE_TAIL_FRACTION = 0.5


def _make_time_chunks(duration_ms: int, chunk_secs: float) -> list[tuple[int, int]]:
    """Split ``[0, duration_ms]`` into ~``chunk_secs``-wide ``(at_ms, end_ms)``
    spans for chunked analysis.

    Used when a long file has no semantic chapters: equal-width time windows
    are still enough to (a) emit per-chunk progress so a multi-minute analysis
    visibly advances ("chunk 4/12") instead of looking hung, and (b) normalise
    energy per window so a quiet intro isn't crushed by a loud climax's RMS.
    The trailing remainder is merged into the last chunk when it is shorter
    than half a chunk, so we never analyse a useless runt at the end.
    """
    chunk_ms = max(1000, int(round(chunk_secs * 1000)))
    if duration_ms <= chunk_ms:
        return [(0, duration_ms)]
    spans: list[tuple[int, int]] = []
    at = 0
    while at < duration_ms:
        end = min(at + chunk_ms, duration_ms)
        spans.append((at, end))
        at = end
    # Merge a short trailing remainder into the chunk before it.
    if len(spans) >= 2:
        last_start, last_end = spans[-1]
        if (last_end - last_start) < chunk_ms * _CHUNK_MERGE_TAIL_FRACTION:
            prev_start, _ = spans[-2]
            spans[-2] = (prev_start, last_end)
            spans.pop()
    return spans


def _coverage_shortfall(
    decoded_ms: int, source_ms: int, *, tol: float = 0.02, min_gap_ms: int = 5000,
) -> str | None:
    """Return a warning string if the decoded audio is materially shorter
    than the source's reported duration, else ``None``.

    Guards against silent truncation: a corrupt audio stream can make the
    decoder stop partway yet still "succeed" (ffmpeg exits 0), yielding a
    short track with no error. Pure + side-effect-free so it's unit-testable.
    """
    if source_ms <= 0:
        return None
    if decoded_ms >= source_ms * (1 - tol):
        return None
    if (source_ms - decoded_ms) < min_gap_ms:
        return None
    return (
        f"⚠ audio decode truncated: {decoded_ms / 60000:.1f} of "
        f"{source_ms / 60000:.1f} min ({decoded_ms * 100 // source_ms}%) — "
        f"source audio may be corrupt; output will be incomplete."
    )


def analyze_beats(
    input: str | Path,
    *,
    sr: int = 22050,
    source: str = "full",
    tracker: str = "auto",
    locked_bpm: float | None = None,
    progress_callback=None,
    on_progress: OnProgress | None = None,
    chapters: list["Chapter"] | None = None,
    chunk_secs: float | None = None,
    _reporter: "ProgressReporter | None" = None,
) -> AudioBeatMap:
    """Analyse the beat structure of an audio or video file.

    Wraps librosa for onset detection, BPM estimation, beat grid, downbeat
    tracking, and per-beat energy. Accepts audio files (.mp3, .wav, .flac,
    .m4a, …) and video files with an audio track.

    One call, one pass — the returned :class:`AudioBeatMap` is the single
    source of truth for all downstream consumers (beat-snap, beat-grid
    assembly, multi-panel canvas sync).

    Install librosa with: ``pip install "videoflow[audio]"``

    Args:
        input:  Path to the audio or video file.
        sr:     Sample rate to use when loading (default 22050 Hz). Lower
                values are faster; 22050 is librosa's standard for beat
                tracking.
        source: Which component of the audio to use for beat tracking.

                ``"full"`` (default) — use the full mix as-is.

                ``"percussive"`` — apply harmonic-percussive source
                separation (HPSS) first and track beats on the percussive
                component only.  Voice and melody are harmonic, so they are
                effectively invisible to the beat tracker.  Use this when
                the recording contains speech or prominent vocals over music
                and you want beats to follow the drums rather than vocal
                onsets.  Energy values also reflect percussive energy.
                No extra dependencies — HPSS is built into librosa.
        tracker: Beat-tracking algorithm.

                ``"auto"`` (default) — picks ``"plp"`` for tracks longer
                than 10 minutes, ``"beat_track"`` otherwise. Long-form
                material drifts under a global-tempo DP tracker; PLP's
                locally-stable pulse handles multi-hour tracks better.

                ``"beat_track"`` — librosa's classic dynamic-programming
                beat tracker. Assumes a single global tempo. Best for short
                tracks (< 10 min) with steady tempo.

                ``"plp"`` — predominant local pulse. Estimates a locally
                stable tempo per frame; beats are local maxima of the
                pulse curve. More robust to tempo drift, gradual rallentando,
                and long-form structure where a global tempo doesn't fit.
        locked_bpm: If set, pin the reported BPM to this value and bias
                the tracker toward it. For ``beat_track``, sets
                ``start_bpm`` and tightens the prior. For ``plp``, narrows
                the tempo search window to ±2 BPM around the lock. Use
                when you know the tempo and want the tracker to stop
                hunting (especially helpful for tracks where
                auto-detection falls onto a half/double tempo octave).
        chapters: Optional chapter list (typically from
                :func:`videoflow.structural.auto_chapter`). When provided,
                analysis runs per-chapter and results are stitched into a
                single timeline — the key win is **per-chunk energy
                normalization**, so quiet ambient sections aren't crushed
                by a loud climax's RMS distribution. Each chapter also
                gets its own tracker resolution (a 5-minute climax chunk
                resolves ``tracker="auto"`` differently than a 5-minute
                quiet intro). Beats / stanzas / energy are concatenated
                in chapter order; the reported ``bpm`` is the
                duration-weighted mean of per-chapter BPMs. Pass ``None``
                (default) for whole-file analysis.
        chunk_secs: When set (and ``chapters`` is not given), split a long
                file into equal-width time windows of roughly this many
                seconds and analyse each window in isolation — same
                stitching and per-chunk energy normalisation as the
                ``chapters`` path, but for material with no semantic
                chapter list. The point is responsiveness on long files:
                each window emits its own progress message ("chunk 4/12")
                and runs its own (cheaper) tracker + HPSS pass, so a
                multi-hour analysis visibly advances instead of looking
                hung inside one monolithic librosa call. Pass ``True`` to
                use :data:`DEFAULT_CHUNK_SECS`. Ignored when ``chapters``
                is provided (chapters win — they carry meaning). ``None``
                (default) keeps whole-file analysis.
        progress_callback: **Deprecated** legacy progress hook —
                ``Callable[[str], None]`` invoked with stage labels.
                Errors are swallowed. Prefer ``on_progress`` for new
                callers; both are honoured during transition.
        on_progress: Modern progress hook —
                :data:`videoflow.progress.OnProgress` callback that
                receives :class:`~videoflow.progress.StageEvent` records
                (start / progress / complete) so UIs can render a tree
                of work, ETAs, and per-stage summaries. Errors in the
                callback are swallowed.

    Returns:
        :class:`AudioBeatMap` with bpm, beats, downbeats, stanzas, energy,
        and duration_ms.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If *source*, *tracker*, or *locked_bpm* is invalid.
        BeatError: If librosa is not installed or analysis fails.
    """
    if source not in _SOURCES:
        raise ValueError(
            f"source must be one of {list(_SOURCES)!r}, got {source!r}"
        )
    if tracker not in _TRACKERS:
        raise ValueError(
            f"tracker must be one of {list(_TRACKERS)!r}, got {tracker!r}"
        )
    if locked_bpm is not None and locked_bpm <= 0:
        raise ValueError(
            f"locked_bpm must be positive, got {locked_bpm!r}"
        )

    # Reporter inheritance: callers that already opened a stage stack
    # (e.g. structural.auto_chapter's `beats` stage) can pass their own
    # reporter so this function's sub-stages emit at depths nested under
    # theirs. Without inheritance the inner stages emitted at "their"
    # depth 2, colliding with structural's depth-2 leaf names in the
    # consumer's progress footer (2026-05-25 footer-collision bug).
    reporter = _reporter or ProgressReporter(on_progress)

    def _progress(label: str) -> None:
        # Thin shim so callers don't have to null-check. We swallow any
        # exception in the callback — UI feedback should never break
        # analysis. Each label marks the START of a new stage; the
        # reporter forwards it as an informational message inside the
        # currently-open stage so the modern on_progress consumer also
        # sees it.
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass
        reporter.message(label)

    input = Path(input)
    if not input.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    if _librosa is None:
        raise BeatError(
            "librosa is required for beat analysis. "
            'Install it with: pip install "videoflow[audio]"'
        )

    # Reclaim temp WAVs orphaned by previously-killed analyses — a kill
    # bypasses the finally that normally unlinks them, so they pile up and
    # silently fill the user's disk. Cheap (one dir scan); runs each call.
    from videoflow.tempfiles import audio_temp_dir, sweep_audio_temp
    sweep_audio_temp()

    with reporter.stage("audio.analyze"):
        # For video files, extract audio to a temp WAV via FFmpeg first.
        _tmp_audio = None
        if input.suffix.lower() in _VIDEO_SUFFIXES:
            with reporter.stage("extract"):
                _progress("Extracting audio from video (ffmpeg)…")
                # Look for ffmpeg on PATH, then alongside this file, then alongside the input file.
                _ffmpeg = "ffmpeg"
                for _candidate in [
                    Path(__file__).parent / "ffmpeg.exe",
                    Path(__file__).parent / "ffmpeg",
                    input.parent / "ffmpeg.exe",
                    input.parent / "ffmpeg",
                ]:
                    if _candidate.is_file():
                        _ffmpeg = str(_candidate)
                        break

                _tmp_audio = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir=str(audio_temp_dir()),
                )
                _tmp_audio.close()
                try:
                    # Lazy import to avoid hoisting structural's heavy deps
                    # (numpy, librosa transitively) into audio.py's import
                    # graph for the non-extracting code paths.
                    from videoflow.structural import _run_ffmpeg_with_progress
                    _rc = _run_ffmpeg_with_progress(
                        _ffmpeg, input, _tmp_audio.name,
                        sr=sr, progress=_progress,
                    )
                    if _rc != 0:
                        Path(_tmp_audio.name).unlink(missing_ok=True)
                        raise BeatError(f"FFmpeg audio extraction failed (exit {_rc})")
                except FileNotFoundError:
                    Path(_tmp_audio.name).unlink(missing_ok=True)
                    raise BeatError(
                        "FFmpeg is required to extract audio from video files. "
                        "Install it from https://ffmpeg.org/download.html — "
                        "or place ffmpeg.exe in the forgegen folder."
                    )
                load_path = _tmp_audio.name
                reporter.complete(summary="audio extracted to wav")
        else:
            load_path = str(input)

        try:
            with reporter.stage("load"):
                _progress("Loading audio (librosa)…")
                y, sr_ = _librosa.load(load_path, sr=sr, mono=True)
                duration_ms = round(_librosa.get_duration(y=y, sr=sr_) * 1000)
                reporter.complete(
                    summary=f"{duration_ms / 1000:.1f}s @ {sr_} Hz mono",
                )

            # Coverage guard — surface silent truncation (corrupt stream that
            # the decoder bails on yet "succeeds"). Compare decoded length to
            # the source's reported duration; warn loudly, don't fail.
            try:
                _src_ms = round(_librosa.get_duration(path=str(input)) * 1000)
            except Exception:
                _src_ms = 0
            _cov_warn = _coverage_shortfall(duration_ms, _src_ms)
            if _cov_warn:
                _progress(_cov_warn)

            # Decide whether to analyse in chunks. Chapters (semantic spans)
            # win — they carry meaning. Otherwise a long file with chunk_secs
            # requested is split into equal-width time windows. Both go through
            # the same per-chunk loop, which runs HPSS (if any) PER CHUNK so a
            # long analysis never blocks inside one monolithic librosa call and
            # energy is normalised per window.
            chunk_spans: list[tuple[int, int]] | None = None
            label_kind = "chunk"
            if chapters:
                chunk_spans = [
                    (ch.at_ms, ch.end_ms if ch.end_ms is not None else duration_ms)
                    for ch in chapters
                ]
                label_kind = "chapter"
            elif chunk_secs:
                width = DEFAULT_CHUNK_SECS if chunk_secs is True else float(chunk_secs)
                spans = _make_time_chunks(duration_ms, width)
                if len(spans) > 1:
                    chunk_spans = spans
                    label_kind = "chunk"

            if chunk_spans is not None:
                stage_name = "chapters" if label_kind == "chapter" else "chunks"
                with reporter.stage(stage_name):
                    bpm, beats_ms, downbeats_ms, stanzas, energy = _analyze_chunks(
                        y, sr_, duration_ms,
                        spans=chunk_spans,
                        source=source,
                        tracker=tracker,
                        locked_bpm=locked_bpm,
                        progress=_progress,
                        reporter=reporter,
                        label_kind=label_kind,
                    )
                    reporter.complete(
                        summary=(
                            f"{len(chunk_spans)} {label_kind}s · "
                            f"BPM {bpm:.0f} · {len(beats_ms)} beats"
                        ),
                    )
            else:
                # Whole-file path. Select the tracking signal once: HPSS
                # separates harmonic (voice, melody) from percussive (drums).
                if source == "percussive":
                    with reporter.stage("hpss"):
                        _progress("Separating percussive component (HPSS)…")
                        _, y_track = _librosa.effects.hpss(y)
                        reporter.complete(summary="percussive component isolated")
                else:
                    y_track = y
                with reporter.stage("track"):
                    bpm, beats_ms, downbeats_ms, stanzas, energy = _analyze_buffer(
                        y_track, sr_, duration_ms,
                        tracker=tracker,
                        locked_bpm=locked_bpm,
                        progress=_progress,
                        time_offset_ms=0,
                    )
                    reporter.complete(
                        summary=f"BPM {bpm:.0f} · {len(beats_ms)} beats",
                    )

        except (BeatError, ValueError):
            raise
        except Exception as exc:
            raise BeatError(f"Beat analysis failed: {exc}") from exc
        finally:
            if _tmp_audio is not None:
                Path(_tmp_audio.name).unlink(missing_ok=True)

        reporter.complete(
            summary=(
                f"BPM {bpm:.0f} · {len(beats_ms)} beats · "
                f"{len(stanzas)} stanzas"
            ),
        )

    return AudioBeatMap(
        bpm=bpm,
        beats=beats_ms,
        downbeats=downbeats_ms,
        stanzas=stanzas,
        energy=energy,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Tempo-octave correction
# ---------------------------------------------------------------------------
#
# Audio beat trackers can lock onto a metrical harmonic and report double
# (or quadruple) the *felt* tempo — e.g. tracking eighth-notes as beats. That
# inflates downstream stroke density ~2x everywhere. Validated on Rhythms of
# Desire: free-run reports 235 BPM and generates 7.7 actions/s, while the gold
# hand-script sits at 4.2 actions/s; folding the octave to 117 BPM lands the
# generator at 3.9 actions/s — essentially on the gold. (Video optical-flow
# timing wouldn't need this — each reversal is a literal stroke — but audio
# infers periodicity and is prone to the octave error.)

_TEMPO_OCTAVE_CEILING = 165.0  # BPM above which the felt pulse is likely half
_TEMPO_OCTAVE_FLOOR = 70.0     # never fold below this (would undershoot)


def _correct_tempo_octave(
    bpm: float,
    beats_ms: list[int],
    energy: list[float],
    *,
    ceiling: float = _TEMPO_OCTAVE_CEILING,
    floor: float = _TEMPO_OCTAVE_FLOOR,
) -> tuple[float, list[int], list[float]]:
    """Fold a doubled tempo octave back toward the felt pulse.

    Returns ``(bpm, beats, energy)``. A no-op unless ``bpm`` is above
    ``ceiling`` *and* halving stays at or above ``floor`` — so a genuinely
    fast track is left alone. When it folds, it keeps whichever phase
    (even- vs odd-indexed beats) carries more total energy, so the strong
    on-beats (kicks / snares) survive and the off-beats drop. Folds
    repeatedly to handle a quadrupled octave (rare). ``energy`` stays
    index-aligned with ``beats`` throughout.
    """
    while (
        bpm > ceiling
        and bpm / 2.0 >= floor
        and len(beats_ms) >= 4
    ):
        if energy and len(energy) == len(beats_ms):
            keep = 0 if sum(energy[0::2]) >= sum(energy[1::2]) else 1
        else:
            keep = 0
        beats_ms = beats_ms[keep::2]
        energy = energy[keep::2] if energy else energy
        bpm = bpm / 2.0
    return bpm, beats_ms, energy


# ---------------------------------------------------------------------------
# Internal — per-buffer analysis core
# ---------------------------------------------------------------------------

def _analyze_buffer(
    y_track,
    sr: int,
    duration_ms: int,
    *,
    tracker: str,
    locked_bpm: float | None,
    progress,
    time_offset_ms: int,
) -> tuple[float, list[int], list[int], list[tuple[int, int]], list[float]]:
    """Run the beat / stanza / energy pipeline on one buffer.

    Returns ``(bpm, beats_ms, downbeats_ms, stanzas, energy)``. All
    timestamps are shifted by ``time_offset_ms`` so the result can be
    placed at any position in a longer timeline. Energy is normalised
    *within this buffer only* — that's the per-chunk normalisation lever
    that fixes ambient/hentai content from being crushed by a loud
    climax's RMS distribution.
    """
    # Resolve "auto" → concrete tracker by buffer length.
    resolved_tracker = tracker
    if resolved_tracker == "auto":
        resolved_tracker = (
            "plp" if duration_ms > PLP_AUTO_DURATION_MS else "beat_track"
        )

    if resolved_tracker == "plp":
        progress("Detecting beats (PLP — long-form stable)…")
        plp_kwargs: dict = {"y": y_track, "sr": sr}
        if locked_bpm is not None:
            plp_kwargs["tempo_min"] = max(1.0, locked_bpm - 2.0)
            plp_kwargs["tempo_max"] = locked_bpm + 2.0
        pulse = _librosa.beat.plp(**plp_kwargs)
        beat_frames = _np.flatnonzero(_librosa.util.localmax(pulse))
        beat_times = _librosa.frames_to_time(beat_frames, sr=sr)

        beats_ms = [round(float(t) * 1000) + time_offset_ms for t in beat_times]
        if locked_bpm is not None:
            bpm = float(locked_bpm)
        elif len(beats_ms) >= 2:
            deltas = sorted(
                beats_ms[i + 1] - beats_ms[i]
                for i in range(len(beats_ms) - 1)
            )
            median_delta = deltas[len(deltas) // 2]
            bpm = 60_000.0 / median_delta if median_delta > 0 else 0.0
        else:
            bpm = 0.0
    else:
        progress("Detecting beats (beat_track)…")
        bt_kwargs: dict = {"y": y_track, "sr": sr}
        if locked_bpm is not None:
            bt_kwargs["start_bpm"] = float(locked_bpm)
            bt_kwargs["tightness"] = 200
        tempo, beat_frames = _librosa.beat.beat_track(**bt_kwargs)
        beat_times = _librosa.frames_to_time(beat_frames, sr=sr)

        bpm = (
            float(locked_bpm)
            if locked_bpm is not None
            else float(_np.atleast_1d(tempo)[0])
        )
        beats_ms = [round(float(t) * 1000) + time_offset_ms for t in beat_times]

    progress("Computing stanzas + per-beat energy…")

    rms = _librosa.feature.rms(y=y_track)[0]
    energy_raw = [
        float(rms[min(int(f), len(rms) - 1)]) for f in beat_frames
    ]
    max_e = max(energy_raw) if energy_raw else 1.0
    energy = [e / max_e if max_e > 0 else 0.0 for e in energy_raw]

    # Correct a doubled tempo octave before deriving downbeats / stanzas, so
    # everything downstream is built on the felt pulse. Skipped when the BPM
    # is pinned (the caller asserted the tempo) — see _correct_tempo_octave.
    if locked_bpm is None:
        bpm, beats_ms, energy = _correct_tempo_octave(bpm, beats_ms, energy)

    downbeats_ms = beats_ms[::4]

    stanzas: list[tuple[int, int]] = []
    for i in range(0, len(beats_ms), 16):
        start = beats_ms[i]
        end = beats_ms[min(i + 16, len(beats_ms) - 1)]
        stanzas.append((start, end))

    return bpm, beats_ms, downbeats_ms, stanzas, energy


def _analyze_chunks(
    y,
    sr: int,
    duration_ms: int,
    *,
    spans: list[tuple[int, int]],
    source: str,
    tracker: str,
    locked_bpm: float | None,
    progress,
    reporter: ProgressReporter | None = None,
    label_kind: str = "chunk",
) -> tuple[float, list[int], list[int], list[tuple[int, int]], list[float]]:
    """Run :func:`_analyze_buffer` per time-span and stitch the results.

    *spans* is a list of ``(at_ms, end_ms)`` windows — either semantic
    chapters or synthetic equal-width time chunks. Each span is analysed
    in isolation: HPSS (when ``source == 'percussive'``), beat tracking,
    stanza grouping, AND energy normalisation all happen against that
    window's own audio. Then beats / downbeats / stanzas / energies are
    concatenated in order; the reported BPM is the duration-weighted mean
    of per-span BPMs.

    HPSS runs **per span**, not once over the whole file — that is what
    keeps a multi-hour percussive analysis from blocking inside one giant
    ``librosa.effects.hpss`` call with no feedback. *y* is therefore the
    RAW signal; this function selects the tracking signal per chunk.

    If a *reporter* is supplied, each span opens its own nested sub-stage
    so UIs can render the per-chunk progress as a tree. *label_kind*
    ("chunk" / "chapter") only changes the progress wording.
    """
    if not spans:
        return _select_and_analyze_buffer(
            y, sr, duration_ms,
            source=source, tracker=tracker, locked_bpm=locked_bpm,
            progress=progress, time_offset_ms=0,
        )

    all_beats: list[int] = []
    all_downbeats: list[int] = []
    all_stanzas: list[tuple[int, int]] = []
    all_energy: list[float] = []
    weighted_bpm = 0.0
    total_weight = 0

    n = len(spans)
    for i, (at_ms, end_ms) in enumerate(spans):
        end_ms = end_ms if end_ms is not None else duration_ms
        chunk_duration_ms = max(0, end_ms - at_ms)
        if chunk_duration_ms < 1000:
            continue  # skip sub-second chunks — beat tracker can't say much

        progress(
            f"Analyzing {label_kind} {i + 1}/{n} "
            f"({at_ms // 1000}s-{end_ms // 1000}s)…"
        )

        span_label = f"{label_kind} {i + 1}/{n}"
        span_ctx = (
            reporter.stage(span_label) if reporter is not None else _nullcontext()
        )

        start_sample = max(0, int(round(at_ms / 1000.0 * sr)))
        end_sample = min(len(y), int(round(end_ms / 1000.0 * sr)))
        y_chunk = y[start_sample:end_sample]
        if len(y_chunk) < sr:
            continue

        with span_ctx:
            chunk_bpm, chunk_beats, chunk_db, chunk_stanzas, chunk_energy = (
                _select_and_analyze_buffer(
                    y_chunk, sr, chunk_duration_ms,
                    source=source, tracker=tracker, locked_bpm=locked_bpm,
                    progress=progress, time_offset_ms=at_ms,
                )
            )
            if reporter is not None:
                reporter.complete(
                    summary=(
                        f"BPM {chunk_bpm:.0f} · {len(chunk_beats)} beats"
                    ),
                )
        all_beats.extend(chunk_beats)
        all_downbeats.extend(chunk_db)
        all_stanzas.extend(chunk_stanzas)
        all_energy.extend(chunk_energy)
        weighted_bpm += chunk_bpm * chunk_duration_ms
        total_weight += chunk_duration_ms

    bpm = weighted_bpm / total_weight if total_weight > 0 else 0.0
    return bpm, all_beats, all_downbeats, all_stanzas, all_energy


def _select_and_analyze_buffer(
    y,
    sr: int,
    duration_ms: int,
    *,
    source: str,
    tracker: str,
    locked_bpm: float | None,
    progress,
    time_offset_ms: int,
) -> tuple[float, list[int], list[int], list[tuple[int, int]], list[float]]:
    """HPSS-if-percussive then :func:`_analyze_buffer`, for one buffer.

    Pulls the per-chunk HPSS into a small helper so each chunk separates
    its own percussive component (and emits its own progress) rather than
    relying on one whole-file pass.
    """
    if source == "percussive":
        progress("Separating percussive component (HPSS)…")
        _, y_track = _librosa.effects.hpss(y)
    else:
        y_track = y
    return _analyze_buffer(
        y_track, sr, duration_ms,
        tracker=tracker, locked_bpm=locked_bpm,
        progress=progress, time_offset_ms=time_offset_ms,
    )
