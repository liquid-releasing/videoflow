"""Audio beat analysis — librosa wrapper returning a rich AudioBeatMap."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import tempfile
from pathlib import Path

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

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

    phrases: list[tuple[int, int]]
    """(start_ms, end_ms) of each musical phrase.

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
            "phrases": [{"start_ms": s, "end_ms": e} for s, e in self.phrases],
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
                phrases=[(int(p["start_ms"]), int(p["end_ms"])) for p in data["phrases"]],
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


def analyze_beats(
    input: str | Path,
    *,
    sr: int = 22050,
    source: str = "full",
    tracker: str = "auto",
    locked_bpm: float | None = None,
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

    Returns:
        :class:`AudioBeatMap` with bpm, beats, downbeats, phrases, energy,
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

    input = Path(input)
    if not input.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    if _librosa is None:
        raise BeatError(
            "librosa is required for beat analysis. "
            'Install it with: pip install "videoflow[audio]"'
        )

    # For video files, extract audio to a temp WAV via FFmpeg first.
    _tmp_audio = None
    if input.suffix.lower() in _VIDEO_SUFFIXES:
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

        _tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _tmp_audio.close()
        try:
            subprocess.run(
                [
                    _ffmpeg, "-y", "-i", str(input),
                    "-vn", "-ar", str(sr), "-ac", "1",
                    "-f", "wav", _tmp_audio.name,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            Path(_tmp_audio.name).unlink(missing_ok=True)
            raise BeatError(
                "FFmpeg is required to extract audio from video files. "
                "Install it from https://ffmpeg.org/download.html — "
                "or place ffmpeg.exe in the forgegen folder."
            )
        except subprocess.CalledProcessError as exc:
            Path(_tmp_audio.name).unlink(missing_ok=True)
            raise BeatError(f"FFmpeg audio extraction failed: {exc}") from exc
        load_path = _tmp_audio.name
    else:
        load_path = str(input)

    try:
        y, sr_ = _librosa.load(load_path, sr=sr, mono=True)
        duration_ms = round(_librosa.get_duration(y=y, sr=sr_) * 1000)

        # Select the signal used for beat tracking and energy.
        # HPSS separates harmonic (voice, melody) from percussive (drums).
        if source == "percussive":
            _, y_track = _librosa.effects.hpss(y)
        else:
            y_track = y

        # Resolve "auto" → concrete tracker by track length.
        resolved_tracker = tracker
        if resolved_tracker == "auto":
            resolved_tracker = (
                "plp" if duration_ms > PLP_AUTO_DURATION_MS else "beat_track"
            )

        if resolved_tracker == "plp":
            # PLP — predominant local pulse. Robust on long-form material
            # where the global tempo drifts. Beats are local maxima of the
            # pulse envelope.
            plp_kwargs: dict = {"y": y_track, "sr": sr_}
            if locked_bpm is not None:
                # Narrow the tempo search around the lock so PLP stops
                # hunting across tempo octaves.
                plp_kwargs["tempo_min"] = max(1.0, locked_bpm - 2.0)
                plp_kwargs["tempo_max"] = locked_bpm + 2.0
            pulse = _librosa.beat.plp(**plp_kwargs)
            beat_frames = _np.flatnonzero(_librosa.util.localmax(pulse))
            beat_times = _librosa.frames_to_time(beat_frames, sr=sr_)

            beats_ms = [round(float(t) * 1000) for t in beat_times]
            if locked_bpm is not None:
                bpm = float(locked_bpm)
            elif len(beats_ms) >= 2:
                # Median inter-beat interval → BPM.
                deltas = [
                    beats_ms[i + 1] - beats_ms[i]
                    for i in range(len(beats_ms) - 1)
                ]
                deltas.sort()
                median_delta = deltas[len(deltas) // 2]
                bpm = 60_000.0 / median_delta if median_delta > 0 else 0.0
            else:
                bpm = 0.0
        else:
            # Classic beat_track — global-tempo dynamic programming.
            bt_kwargs: dict = {"y": y_track, "sr": sr_}
            if locked_bpm is not None:
                bt_kwargs["start_bpm"] = float(locked_bpm)
                bt_kwargs["tightness"] = 200  # lock harder to start_bpm
            tempo, beat_frames = _librosa.beat.beat_track(**bt_kwargs)
            beat_times = _librosa.frames_to_time(beat_frames, sr=sr_)

            bpm = (
                float(locked_bpm)
                if locked_bpm is not None
                else float(_np.atleast_1d(tempo)[0])
            )
            beats_ms = [round(float(t) * 1000) for t in beat_times]

        # Downbeats: every 4th beat (assume 4/4 time, V1)
        downbeats_ms = beats_ms[::4]

        # Phrases: every 16 beats = 4 bars
        phrases: list[tuple[int, int]] = []
        for i in range(0, len(beats_ms), 16):
            start = beats_ms[i]
            end = beats_ms[min(i + 16, len(beats_ms) - 1)]
            phrases.append((start, end))

        # Per-beat energy: RMS of the tracked signal, normalised to 0.0–1.0.
        # Using y_track keeps energy consistent with the beat source —
        # percussive mode reports drum energy, not vocal energy.
        rms = _librosa.feature.rms(y=y_track)[0]  # shape: (n_frames,)
        energy_raw = [
            float(rms[min(int(f), len(rms) - 1)]) for f in beat_frames
        ]
        max_e = max(energy_raw) if energy_raw else 1.0
        energy = [e / max_e if max_e > 0 else 0.0 for e in energy_raw]

    except (BeatError, ValueError):
        raise
    except Exception as exc:
        raise BeatError(f"Beat analysis failed: {exc}") from exc
    finally:
        if _tmp_audio is not None:
            Path(_tmp_audio.name).unlink(missing_ok=True)

    return AudioBeatMap(
        bpm=bpm,
        beats=beats_ms,
        downbeats=downbeats_ms,
        phrases=phrases,
        energy=energy,
        duration_ms=duration_ms,
    )
