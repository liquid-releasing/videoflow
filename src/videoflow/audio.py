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

            # Select the signal used for beat tracking and energy.
            # HPSS separates harmonic (voice, melody) from percussive (drums).
            if source == "percussive":
                with reporter.stage("hpss"):
                    _progress("Separating percussive component (HPSS)…")
                    _, y_track = _librosa.effects.hpss(y)
                    reporter.complete(summary="percussive component isolated")
            else:
                y_track = y

            if chapters:
                with reporter.stage("chapters"):
                    bpm, beats_ms, downbeats_ms, stanzas, energy = _analyze_per_chapter(
                        y_track, sr_, duration_ms,
                        chapters=chapters,
                        tracker=tracker,
                        locked_bpm=locked_bpm,
                        progress=_progress,
                        reporter=reporter,
                    )
                    reporter.complete(
                        summary=(
                            f"{len(chapters)} chapters · "
                            f"BPM {bpm:.0f} · {len(beats_ms)} beats"
                        ),
                    )
            else:
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
    downbeats_ms = beats_ms[::4]

    stanzas: list[tuple[int, int]] = []
    for i in range(0, len(beats_ms), 16):
        start = beats_ms[i]
        end = beats_ms[min(i + 16, len(beats_ms) - 1)]
        stanzas.append((start, end))

    rms = _librosa.feature.rms(y=y_track)[0]
    energy_raw = [
        float(rms[min(int(f), len(rms) - 1)]) for f in beat_frames
    ]
    max_e = max(energy_raw) if energy_raw else 1.0
    energy = [e / max_e if max_e > 0 else 0.0 for e in energy_raw]

    return bpm, beats_ms, downbeats_ms, stanzas, energy


def _analyze_per_chapter(
    y_track,
    sr: int,
    duration_ms: int,
    *,
    chapters: list["Chapter"],
    tracker: str,
    locked_bpm: float | None,
    progress,
    reporter: ProgressReporter | None = None,
) -> tuple[float, list[int], list[int], list[tuple[int, int]], list[float]]:
    """Run :func:`_analyze_buffer` per chapter and stitch the results.

    Each chapter is analysed in isolation — beat tracking, stanza
    grouping, AND energy normalisation all happen against the chunk's
    own audio. Then beats / downbeats / stanzas / energies are
    concatenated in chapter order; the reported BPM is the
    duration-weighted mean of per-chapter BPMs.

    If a *reporter* is supplied, each chapter opens its own nested
    sub-stage so UIs can render the per-chapter progress as a tree.
    """
    if not chapters:
        return _analyze_buffer(
            y_track, sr, duration_ms,
            tracker=tracker, locked_bpm=locked_bpm,
            progress=progress, time_offset_ms=0,
        )

    all_beats: list[int] = []
    all_downbeats: list[int] = []
    all_stanzas: list[tuple[int, int]] = []
    all_energy: list[float] = []
    weighted_bpm = 0.0
    total_weight = 0

    for i, ch in enumerate(chapters):
        end_ms = ch.end_ms if ch.end_ms is not None else duration_ms
        chunk_duration_ms = max(0, end_ms - ch.at_ms)
        if chunk_duration_ms < 1000:
            continue  # skip sub-second chunks — beat tracker can't say much

        progress(
            f"Analyzing chapter {i + 1}/{len(chapters)} "
            f"({ch.at_ms // 1000}s-{end_ms // 1000}s)…"
        )

        chapter_label = f"chapter {i + 1}/{len(chapters)}"
        chapter_ctx = (
            reporter.stage(chapter_label) if reporter is not None else _nullcontext()
        )

        start_sample = max(0, int(round(ch.at_ms / 1000.0 * sr)))
        end_sample = min(len(y_track), int(round(end_ms / 1000.0 * sr)))
        y_chunk = y_track[start_sample:end_sample]
        if len(y_chunk) < sr:
            continue

        with chapter_ctx:
            chunk_bpm, chunk_beats, chunk_db, chunk_stanzas, chunk_energy = _analyze_buffer(
                y_chunk, sr, chunk_duration_ms,
                tracker=tracker, locked_bpm=locked_bpm,
                progress=progress, time_offset_ms=ch.at_ms,
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
