"""Funscript generation from audio beat maps.

Pipeline:
    AudioBeatMap
        → beats_to_curve()    — beat-locked alternating stroke signal
        → classify_modes()    — phrase-level mode labels (slow/fast/tease/edging/break)
        → shape_curve()       — mode-aware amplitude adjustment
        → export_funscript()  — validated .funscript JSON

Or use the convenience wrapper::

    from videoflow.generate import generate_from_beats
    from videoflow.audio import analyze_beats

    beat_map = analyze_beats("track.mp3", source="percussive")
    output = generate_from_beats(beat_map, "track.funscript")

Output is a standard .funscript file ready for any device or for
refinement in FunScriptForge.
"""

from __future__ import annotations

import json
from pathlib import Path

from videoflow.audio import AudioBeatMap


class GenerateError(RuntimeError):
    """Raised when funscript generation fails."""


# ---------------------------------------------------------------------------
# Mode shaping parameters
# ---------------------------------------------------------------------------

#: Amplitude scale factors per mode.
#: high_scale: fraction of the raw (high - low) amplitude to keep.
#: edging is handled separately (builds within the phrase).
_MODE_HIGH_SCALE: dict[str, float] = {
    "break":  0.12,  # near-still — tiny movement, device stays alive
    "tease":  0.38,  # subtle — noticeable but restrained
    "slow":   0.95,  # full range — wide, unhurried strokes
    "steady": 0.78,  # normal — default feel for most music
    "fast":   0.70,  # slightly compressed — snappy, not overextended
    "edging": 1.00,  # max — but built progressively (see shape_curve)
}

_VALID_MODES = frozenset(_MODE_HIGH_SCALE)


# ---------------------------------------------------------------------------
# Step 1 — Raw motion curve
# ---------------------------------------------------------------------------

def beats_to_curve(
    beat_map: AudioBeatMap,
    *,
    low: int = 10,
    high: int = 90,
    min_stroke: int = 20,
) -> list[tuple[int, int]]:
    """Convert an :class:`~videoflow.audio.AudioBeatMap` into a raw motion curve.

    Each beat produces one point in the curve. Positions alternate between a
    high peak and a low trough so the device strokes on every beat. The peak
    height is energy-scaled — loud beats produce tall strokes, quiet beats
    produce smaller ones.

    Args:
        beat_map: Result of :func:`~videoflow.audio.analyze_beats`.
        low: Trough position (0–100). Default 10.
        high: Maximum peak position (0–100). Default 90.
        min_stroke: Minimum stroke amplitude above *low*, even on silent
            beats. Keeps the device moving at all times. Default 20.

    Returns:
        List of ``(timestamp_ms, position)`` pairs, one per beat, sorted by
        time. Positions are in ``[0, 100]``.

    Example::

        curve = beats_to_curve(beat_map)
        # [(0, 85), (484, 10), (968, 72), (1452, 10), ...]
    """
    if not beat_map.beats:
        return []

    curve: list[tuple[int, int]] = []
    for i, (beat_ms, energy) in enumerate(zip(beat_map.beats, beat_map.energy)):
        if i % 2 == 0:
            # Peak — scale amplitude by energy, enforce minimum stroke
            amplitude = max(min_stroke, round((high - low) * energy))
            pos = min(100, low + amplitude)
        else:
            pos = low
        curve.append((beat_ms, pos))

    return curve


# ---------------------------------------------------------------------------
# Step 2 — Mode classification
# ---------------------------------------------------------------------------

def classify_modes(
    beat_map: AudioBeatMap,
) -> list[tuple[int, int, str]]:
    """Label each musical phrase with a behavioural mode.

    Modes are determined by phrase-level energy statistics and the overall
    BPM. No ML required — purely rule-based.

    Mode rules (evaluated in priority order):

    ======== ================================================================
    Mode     Rule
    ======== ================================================================
    break    Average energy < 0.15 — near-silence, minimal movement
    tease    Average energy < 0.30 — quiet, restrained
    edging   Energy trend rising (second half > first half by ≥ 0.15) and
             average energy ≥ 0.35 — building tension
    fast     BPM ≥ 140 — rapid tempo
    slow     BPM ≤ 75 — slow tempo
    steady   Everything else — the normal case
    ======== ================================================================

    Args:
        beat_map: Result of :func:`~videoflow.audio.analyze_beats`.

    Returns:
        List of ``(start_ms, end_ms, mode)`` tuples, one per phrase.
        Covers the entire duration of the beat map.
    """
    modes: list[tuple[int, int, str]] = []

    for start_ms, end_ms in beat_map.phrases:
        indices = [
            i for i, b in enumerate(beat_map.beats)
            if start_ms <= b < end_ms
        ]

        if not indices:
            modes.append((start_ms, end_ms, "break"))
            continue

        phrase_energy = [beat_map.energy[i] for i in indices]
        avg = sum(phrase_energy) / len(phrase_energy)

        # Energy trend: compare second half vs first half average
        mid = len(phrase_energy) // 2
        if mid > 0:
            first_avg = sum(phrase_energy[:mid]) / mid
            second_avg = sum(phrase_energy[mid:]) / max(1, len(phrase_energy) - mid)
            trend = second_avg - first_avg
        else:
            trend = 0.0

        if avg < 0.15:
            mode = "break"
        elif avg < 0.30:
            mode = "tease"
        elif trend >= 0.15 and avg >= 0.35:
            mode = "edging"
        elif beat_map.bpm >= 140:
            mode = "fast"
        elif beat_map.bpm <= 75:
            mode = "slow"
        else:
            mode = "steady"

        modes.append((start_ms, end_ms, mode))

    return modes


# ---------------------------------------------------------------------------
# Step 3 — Mode-aware shaping
# ---------------------------------------------------------------------------

def shape_curve(
    curve: list[tuple[int, int]],
    modes: list[tuple[int, int, str]],
    *,
    low: int = 10,
) -> list[tuple[int, int]]:
    """Apply mode-aware amplitude shaping to a raw motion curve.

    Each point in *curve* is adjusted based on the mode of the phrase it
    falls in:

    - **break** — amplitude compressed to ~12 % (device barely moves)
    - **tease** — amplitude at ~38 % (restrained, subtle)
    - **slow** — full amplitude (~95 %)
    - **steady** — normal amplitude (~78 %)
    - **fast** — slightly compressed (~70 %)
    - **edging** — amplitude builds linearly from 50 % → 100 % over
      the phrase, creating a rising tension arc

    Trough points (those at *low*) are preserved as-is. Only peaks are scaled.

    Args:
        curve: Raw ``(timestamp_ms, position)`` pairs from
            :func:`beats_to_curve`.
        modes: Mode timeline from :func:`classify_modes`.
        low: Trough position used in the original curve. Default 10.

    Returns:
        Shaped ``(timestamp_ms, position)`` pairs clamped to ``[0, 100]``.
    """
    if not curve or not modes:
        return list(curve)

    # Build a fast lookup: timestamp → (section_start, section_end, mode)
    def _get_section(t: int) -> tuple[int, int, str]:
        for start, end, mode in modes:
            if start <= t < end:
                return start, end, mode
        return modes[-1]

    shaped: list[tuple[int, int]] = []

    for t, pos in curve:
        if pos <= low:
            # Trough — preserve
            shaped.append((t, pos))
            continue

        sec_start, sec_end, mode = _get_section(t)
        raw_amplitude = pos - low

        if mode == "edging":
            sec_dur = max(1, sec_end - sec_start)
            progress = min(1.0, (t - sec_start) / sec_dur)
            scale = 0.50 + 0.50 * progress  # 50% → 100% over phrase
        else:
            scale = _MODE_HIGH_SCALE.get(mode, _MODE_HIGH_SCALE["steady"])

        new_pos = low + round(raw_amplitude * scale)
        shaped.append((t, max(0, min(100, new_pos))))

    return shaped


# ---------------------------------------------------------------------------
# Step 4 — Funscript export
# ---------------------------------------------------------------------------

def export_funscript(
    curve: list[tuple[int, int]],
    output: str | Path,
    *,
    title: str = "",
    range_: int = 90,
) -> Path:
    """Write a motion curve to a ``.funscript`` file.

    The output is a standard funscript JSON file compatible with The Handy,
    OSR2, and any player that reads the format.

    Args:
        curve: ``(timestamp_ms, position)`` pairs. Will be sorted by time
            and deduplicated (first occurrence at each timestamp wins).
        output: Destination ``.funscript`` file path.
        title: Optional title stored in the file metadata.
        range_: ``range`` field in the funscript header. Default 90.

    Returns:
        Path to the written file.

    Raises:
        GenerateError: If the curve is empty after deduplication.
    """
    output = Path(output)

    # Sort, deduplicate, clamp
    seen: set[int] = set()
    actions: list[dict] = []
    for t, pos in sorted(curve):
        if t in seen:
            continue
        seen.add(t)
        actions.append({"at": t, "pos": max(0, min(100, pos))})

    if not actions:
        raise GenerateError("curve is empty — nothing to export")

    data: dict = {
        "version": "1.0",
        "inverted": False,
        "range": range_,
        "actions": actions,
    }
    if title:
        data["metadata"] = {"title": title}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_from_beats(
    beat_map: AudioBeatMap,
    output: str | Path,
    *,
    low: int = 10,
    high: int = 90,
    title: str = "",
) -> Path:
    """Full pipeline: :class:`~videoflow.audio.AudioBeatMap` → ``.funscript``.

    Convenience wrapper that runs all four steps in sequence::

        beats_to_curve → classify_modes → shape_curve → export_funscript

    Args:
        beat_map: Result of :func:`~videoflow.audio.analyze_beats`.
        output: Destination ``.funscript`` file path.
        low: Trough position. Default 10.
        high: Maximum peak position. Default 90.
        title: Optional title stored in the file metadata.

    Returns:
        Path to the written ``.funscript`` file.

    Example::

        from videoflow.audio import analyze_beats
        from videoflow.generate import generate_from_beats

        beat_map = analyze_beats("track.mp3", source="percussive")
        path = generate_from_beats(beat_map, "track.funscript")
        print(f"generated: {path}")
    """
    curve = beats_to_curve(beat_map, low=low, high=high)
    modes = classify_modes(beat_map)
    shaped = shape_curve(curve, modes, low=low)
    return export_funscript(shaped, output, title=title)
