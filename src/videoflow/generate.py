"""Funscript generation from audio beat maps.

Pipeline:
    AudioBeatMap
        → beats_to_curve()    — beat-locked alternating stroke signal
        → classify_modes()    — stanza-level mode labels (slow/fast/tease/edging/break)
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
from typing import Callable

from videoflow.audio import AudioBeatMap


class GenerateError(RuntimeError):
    """Raised when funscript generation fails."""


# Map every accepted stroke_density form to its integer "actions per beat".
# - "half" / 1 — one action per beat, alternates peak/trough across beats.
#   One full stroke spans two beats. Sensual.
# - "full" / 2 — peak at beat, trough at mid-beat. Canonical PD-style.
# - 4 — four actions per beat (16th-note resolution at moderate BPM).
#   Two strokes per beat. Dense.
# - 8 — eight actions per beat (32nd-note resolution). Very dense; reserved
#   for short climactic chapters where saturation is intentional.
_DENSITY_ALIASES: dict = {
    "half": 1, "1": 1, 1: 1,
    "full": 2, "2": 2, 2: 2,
    "4": 4, 4: 4,
    "8": 8, 8: 8,
}


def _resolve_density(value) -> int:
    """Normalise a stroke_density argument to an integer actions-per-beat.

    Accepts: "half"|"full" (legacy aliases), "1"|"2"|"4"|"8", or
    1|2|4|8 as int.

    Raises:
        ValueError: If *value* is not one of the accepted forms.
    """
    try:
        return _DENSITY_ALIASES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"stroke_density must be one of "
            f"{sorted({k for k in _DENSITY_ALIASES if isinstance(k, str)})!r} "
            f"or 1|2|4|8, got {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Mode shaping parameters
# ---------------------------------------------------------------------------

#: Amplitude scale factors per mode.
#: high_scale: fraction of the raw (high - low) amplitude to keep.
#: edging is handled separately (builds within the stanza).
_MODE_HIGH_SCALE: dict[str, float] = {
    "break":  0.12,  # near-still — tiny movement, device stays alive
    "tease":  0.38,  # subtle — noticeable but restrained
    "slow":   0.95,  # full range — wide, unhurried strokes
    "steady": 0.78,  # normal — default feel for most music
    "fast":   0.70,  # slightly compressed — snappy, not overextended
    "edging": 1.00,  # max — but built progressively (see shape_curve)
}

_VALID_MODES = frozenset(_MODE_HIGH_SCALE)


#: Fixed-depth model: fraction of a FULL half-stroke (50 → rail) per mode.
#: 1.0 = rail-to-rail (the bimodal backbone). Partials add the deliberate
#: middle texture the hand-made gold keeps (~20-25 % of points), WITHOUT
#: collapsing the whole distribution to a centered bell. Proven independently
#: via audio energy (bench) and video optical flow (FF-Flow timing + snap):
#: depth must be FIXED, not signal-scaled — the signal drives timing/density,
#: never amplitude. break/tease are the only deliberately-shallow modes.
_MODE_DEPTH: dict[str, float] = {
    "break":  0.55,  # softened but still moving (not a near-still bell)
    "tease":  0.72,  # restrained, partial — quiet-section middle texture
    "slow":   0.95,  # big, unhurried strokes — low rate, near-full depth
    "steady": 1.00,  # full rails
    "fast":   0.50,  # DEPTH<->RATE COUPLING: busy sections = smaller strokes.
                     # Measured on Sinful gold (2026-06-13): at ~9 strokes/s
                     # avg stroke ~52/100, not rail-to-rail — you physically
                     # can't slam full strokes at speed. NOT softness (tease);
                     # it's the rate coupling. Lands in the mid deciles (the
                     # gold's middle texture), distinct from tease's limbo.
    "edging": 1.00,  # full rails — the climax
}


# ---------------------------------------------------------------------------
# Step 1 — Raw motion curve
# ---------------------------------------------------------------------------

def beats_to_curve(
    beat_map: AudioBeatMap,
    *,
    low: int = 10,
    high: int = 90,
    min_stroke: int = 20,
    center: int | None = None,
    center_trajectory: tuple[int, int] | None = None,
    tone_per_stanza: list[tuple[int, int, int, int]] | None = None,
    energy_normalize: bool = False,
    stroke_density: object = "half",
    depth_model: str = "fixed",
    gate: float = 0.10,
    modes: list[tuple[int, int, str]] | None = None,
) -> list[tuple[int, int]]:
    """Convert an :class:`~videoflow.audio.AudioBeatMap` into a raw motion curve.

    **Depth models** (``depth_model``):

    - **fixed** (default): full-depth backbone — every surviving beat strokes
        rail-to-rail (around ``center``, default 50). Energy is a *gate*
        (beats below ``gate`` on the 0-1 normalised scale are dropped →
        density modulation), NOT an amplitude scale. ``modes`` apply
        per-beat *partial* depth via :data:`_MODE_DEPTH` for the gold's
        middle texture. This is the bimodal model — strokes reach the rails.
        ``shape_curve`` must be skipped (shaping is folded in here).
    - **energy** (legacy): peak height scales with energy — loud beats tall,
        quiet beats small. Produces a centered bell; kept for back-compat
        and the pre-0.0.8 calling convention. Honours ``low``/``high``/
        ``min_stroke`` and is refined afterwards by :func:`shape_curve`.

    Each beat produces one point in the curve. Positions alternate between a
    high peak and a low trough so the device strokes on every beat. The peak
    height is energy-scaled — loud beats produce tall strokes, quiet beats
    produce smaller ones.

    Two stroke models are supported:

    - **Legacy** (``center=None``, default): troughs sit at ``low``; peaks
        rise to ``low + amplitude`` where amplitude is energy-scaled.
        Asymmetric — the curve oscillates upward from a fixed floor.
    - **Centered** (``center=int``): troughs and peaks sit symmetrically
        below and above ``center``, both energy-scaled. Matches PythonDancer
        / haptic-rest convention where the device idles at mid-stroke.

    Args:
        beat_map: Result of :func:`~videoflow.audio.analyze_beats`.
        low: Trough position (0–100). Default 10. In centered mode, used
            with ``high`` to derive the maximum half-stroke ``(high-low)/2``.
        high: Maximum peak position (0–100). Default 90.
        min_stroke: Minimum stroke amplitude even on silent beats. In legacy
            mode this is the floor amplitude above ``low``. In centered mode
            half of this value is the floor half-stroke either side of
            ``center``. Default 20.
        center: If set, use the centered stroke model with this midpoint.
            Default ``None`` (legacy model).

    Returns:
        List of ``(timestamp_ms, position)`` pairs, one per beat, sorted by
        time. Positions are in ``[0, 100]``.

    Example::

        curve = beats_to_curve(beat_map)                  # legacy
        curve = beats_to_curve(beat_map, center=50)       # centered
    """
    if not beat_map.beats:
        return []

    actions_per_beat = _resolve_density(stroke_density)

    # Optionally normalise energy so the 95th-percentile-loudest beat hits
    # 1.0 — guarantees the loudest beats actually reach `high` / `low`
    # instead of compressing into the middle when the track's raw RMS
    # values never approach 1.0. Capped at 1.0 so a single outlier beat
    # doesn't suppress everything else.
    energies = list(beat_map.energy)
    # Always compute a 0-1 normalised copy (95th-pct = 1.0) for the gate
    # decision, independent of the legacy energy_normalize amplitude flag.
    beats_count = len(beat_map.beats)
    if energies:
        sorted_e = sorted(energies)
        ref = sorted_e[int(len(sorted_e) * 0.95)] or sorted_e[-1]
        norm_energies = (
            [min(1.0, e / ref) for e in energies] if ref > 0 else list(energies)
        )
    else:
        # No energy track — disable gating (every beat fires at full depth).
        norm_energies = [1.0] * beats_count
    if energy_normalize and energies:
        energies = list(norm_energies)

    # Pre-compute half-beat durations for "full" density mode
    beats_list = list(beat_map.beats)
    if len(beats_list) >= 2:
        deltas = sorted(beats_list[i + 1] - beats_list[i] for i in range(len(beats_list) - 1))
        median_beat_dur = deltas[len(deltas) // 2]
    else:
        median_beat_dur = 480  # 125 BPM fallback

    def _next_dur(i: int) -> int:
        if i + 1 < len(beats_list):
            return beats_list[i + 1] - beats_list[i]
        return median_beat_dur

    # Resolve a per-beat center function. center_trajectory takes priority;
    # it's the first "tone" primitive — a linearly-interpolated center
    # position from start to end of the track. (50, 50) = flat (current
    # default). (30, 70) = rising. (70, 30) = falling. Future tones layer
    # amplitude/density/mode trajectories on top of this same shape.
    track_start = beats_list[0] if beats_list else 0
    track_end = beats_list[-1] if beats_list else 1
    track_dur = max(track_end - track_start, 1)

    def center_at(t: int) -> int:
        # Per-stanza tone takes priority — picks the segment containing t
        # and interpolates within it. Falls back to track-wide trajectory,
        # then to constant center.
        if tone_per_stanza:
            for ps, pe, sc, ec in tone_per_stanza:
                if ps <= t < pe:
                    span = max(1, pe - ps)
                    progress = max(0.0, min(1.0, (t - ps) / span))
                    return int(round(sc + (ec - sc) * progress))
            # past end of last segment — clamp to last segment's end
            ps, pe, sc, ec = tone_per_stanza[-1]
            return ec
        if center_trajectory is not None:
            start_c, end_c = center_trajectory
            progress = (t - track_start) / track_dur
            progress = max(0.0, min(1.0, progress))
            return int(round(start_c + (end_c - start_c) * progress))
        return center if center is not None else 0  # 0 = unused in legacy path

    curve: list[tuple[int, int]] = []

    # ---- Fixed-depth model (default) — the bimodal backbone --------------
    # Each surviving (gated) beat emits a self-contained full stroke toward
    # the rails. Energy gates *which* beats fire (density); modes set partial
    # depth (texture). Depth never scales with the signal — that's the law.
    if depth_model == "fixed":
        def _mode_at(t: int) -> str:
            if not modes:
                return "steady"
            for s, e, m in modes:
                if s <= t < e:
                    return m
            return modes[-1][2]

        # Default center 50 unless a trajectory/tone/explicit center is set.
        moving_center = (
            center is not None
            or center_trajectory is not None
            or tone_per_stanza is not None
        )
        # At least 2 points per surviving beat so each beat is a complete
        # stroke (peak→trough) — gating can drop neighbours without breaking
        # cross-beat alternation.
        n_pts = max(2, actions_per_beat)
        for i, beat_ms in enumerate(beats_list):
            if norm_energies[i] < gate:
                continue
            depth = _MODE_DEPTH.get(_mode_at(beat_ms), 1.0)
            half = 50.0 * depth
            beat_dur = _next_dur(i)
            for k in range(n_pts):
                t = beat_ms + (k * beat_dur) // n_pts
                c = center_at(t) if moving_center else 50
                pos = c + half if k % 2 == 0 else c - half
                curve.append((t, max(0, min(100, round(pos)))))
        return curve

    use_centered = (
        center is not None
        or center_trajectory is not None
        or tone_per_stanza is not None
    )
    if use_centered:
        max_half = (high - low) / 2
        min_half = min_stroke / 2

        if actions_per_beat == 1:
            # 1 action per beat: alternate peak/trough across beats.
            # One stroke spans two beats — calmer, longer-cycle motion.
            for i, (beat_ms, energy) in enumerate(zip(beats_list, energies)):
                half = max(min_half, max_half * energy)
                c = center_at(beat_ms)
                if i % 2 == 0:
                    pos = min(100, round(c + half))
                else:
                    pos = max(0, round(c - half))
                curve.append((beat_ms, pos))
            return curve

        # N actions per beat (N >= 2, even): N evenly-spaced points per
        # beat, alternating peak/trough/peak/trough… within each beat.
        # Sub-beat slots inherit the parent beat's energy in v0.0.4;
        # sub-beat onset weighting is queued.
        for i, (beat_ms, energy) in enumerate(zip(beats_list, energies)):
            half = max(min_half, max_half * energy)
            beat_dur = _next_dur(i)
            for k in range(actions_per_beat):
                t = beat_ms + (k * beat_dur) // actions_per_beat
                c = center_at(t)
                if k % 2 == 0:
                    pos = min(100, round(c + half))
                else:
                    pos = max(0, round(c - half))
                curve.append((t, pos))
        return curve

    # Legacy (uncentered) path — peaks rise from a fixed `low` floor.
    if actions_per_beat == 1:
        for i, (beat_ms, energy) in enumerate(zip(beats_list, energies)):
            if i % 2 == 0:
                amplitude = max(min_stroke, round((high - low) * energy))
                pos = min(100, low + amplitude)
            else:
                pos = low
            curve.append((beat_ms, pos))
        return curve

    for i, (beat_ms, energy) in enumerate(zip(beats_list, energies)):
        amplitude = max(min_stroke, round((high - low) * energy))
        peak = min(100, low + amplitude)
        beat_dur = _next_dur(i)
        for k in range(actions_per_beat):
            t = beat_ms + (k * beat_dur) // actions_per_beat
            pos = peak if k % 2 == 0 else low
            curve.append((t, pos))
    return curve


# ---------------------------------------------------------------------------
# Auto-tone — per-stanza center trajectory from energy slope
# ---------------------------------------------------------------------------

def compute_auto_tone(
    beat_map: AudioBeatMap,
    *,
    baseline: int = 50,
    swing: int = 20,
) -> list[tuple[int, int, int, int]]:
    """Per-stanza center trajectory derived from each stanza's energy slope.

    For each stanza, fit a line to its beat-level energies. A rising slope
    means the stanza is building (more intensity / density / drone weight
    over the stanza) — its center starts lower and ends higher. A falling
    slope means the stanza relaxes — center starts higher and ends lower.
    A flat slope keeps center near baseline.

    Works on **musical movement** broadly, not just melody — drones and
    rhythmic-only stanzas produce a non-zero slope too because their energy
    builds and decays through layered onsets, even with static pitch.

    Args:
        beat_map: Result of :func:`~videoflow.audio.analyze_beats`.
        baseline: Center value the trajectory swings around (0–100).
            Default 50 (mid-stroke).
        swing: Maximum total swing of one stanza, in position units. With
            swing=20 a fully-rising stanza goes from baseline-10 to
            baseline+10. Default 20.

    Returns:
        List of ``(start_ms, end_ms, start_center, end_center)`` tuples,
        one per stanza. Pass directly to ``beats_to_curve(tone_per_stanza=…)``.
    """
    segments: list[tuple[int, int, int, int]] = []
    energies = list(beat_map.energy)
    for start_ms, end_ms in beat_map.stanzas:
        idx = [
            i for i, b in enumerate(beat_map.beats)
            if start_ms <= b < end_ms
        ]
        if len(idx) < 2:
            segments.append((start_ms, end_ms, baseline, baseline))
            continue

        stanza_e = [energies[i] for i in idx]
        n = len(stanza_e)
        x_mean = (n - 1) / 2
        y_mean = sum(stanza_e) / n
        num = sum((i - x_mean) * (e - y_mean) for i, e in enumerate(stanza_e))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        # Normalise: a slope that climbs 1.0 energy unit across the stanza
        # = full swing in one direction. Clip to ±1 for stability.
        rise = max(-1.0, min(1.0, slope * (n - 1)))
        delta = int(round(swing * rise / 2))
        segments.append((start_ms, end_ms, baseline - delta, baseline + delta))
    return segments


# ---------------------------------------------------------------------------
# Step 2 — Mode classification (lifted to videoflow.stanzas in 0.0.5;
# re-exported here for back-compat with v0.0.4 callers)
# ---------------------------------------------------------------------------

from videoflow.stanzas import classify_modes  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Step 3 — Mode-aware shaping
# ---------------------------------------------------------------------------

def shape_curve(
    curve: list[tuple[int, int]],
    modes: list[tuple[int, int, str]],
    *,
    low: int = 10,
    center: int | None = None,
    center_trajectory: tuple[int, int] | None = None,
    tone_per_stanza: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Apply mode-aware amplitude shaping to a raw motion curve.

    Each point in *curve* is adjusted based on the mode of the stanza it
    falls in:

    - **break** — amplitude compressed to ~12 % (device barely moves)
    - **tease** — amplitude at ~38 % (restrained, subtle)
    - **slow** — full amplitude (~95 %)
    - **steady** — normal amplitude (~78 %)
    - **fast** — slightly compressed (~70 %)
    - **edging** — amplitude builds linearly from 50 % → 100 % over
      the stanza, creating a rising tension arc

    Two compression models match the two :func:`beats_to_curve` modes:

    - **Legacy** (``center=None``, default): troughs (positions ``<= low``)
        are preserved; only peaks are scaled toward ``low``.
    - **Centered** (``center=int``): both peaks and troughs are scaled
        toward ``center``, preserving symmetry.

    Args:
        curve: Raw ``(timestamp_ms, position)`` pairs from
            :func:`beats_to_curve`.
        modes: Mode timeline from :func:`classify_modes`.
        low: Trough position used in the original curve (legacy model).
            Default 10.
        center: Curve midpoint (centered model). If set, deviations from
            ``center`` are scaled instead of deviations from ``low``.

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

    # Resolve a per-time center value (mirrors beats_to_curve). When the
    # trajectory rises or falls across the track, the compression in
    # shape_curve has to scale toward the *current* center, not a constant.
    track_start = curve[0][0] if curve else 0
    track_end = curve[-1][0] if curve else 1
    track_dur = max(track_end - track_start, 1)

    def center_at(t: int) -> int | None:
        if tone_per_stanza:
            for ps, pe, sc, ec in tone_per_stanza:
                if ps <= t < pe:
                    span = max(1, pe - ps)
                    progress = max(0.0, min(1.0, (t - ps) / span))
                    return int(round(sc + (ec - sc) * progress))
            return tone_per_stanza[-1][3]  # past end → last segment's end
        if center_trajectory is not None:
            start_c, end_c = center_trajectory
            progress = (t - track_start) / track_dur
            progress = max(0.0, min(1.0, progress))
            return int(round(start_c + (end_c - start_c) * progress))
        return center

    shaped: list[tuple[int, int]] = []

    for t, pos in curve:
        sec_start, sec_end, mode = _get_section(t)

        if mode == "edging":
            sec_dur = max(1, sec_end - sec_start)
            progress = min(1.0, (t - sec_start) / sec_dur)
            scale = 0.50 + 0.50 * progress  # 50% → 100% over stanza
        else:
            scale = _MODE_HIGH_SCALE.get(mode, _MODE_HIGH_SCALE["steady"])

        c = center_at(t)
        if c is not None:
            new_pos = round(c + (pos - c) * scale)
            shaped.append((t, max(0, min(100, new_pos))))
            continue

        if pos <= low:
            # Trough — preserve (legacy model)
            shaped.append((t, pos))
            continue

        raw_amplitude = pos - low
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
    generated_from: dict | None = None,
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
        generated_from: Optional provenance block embedded in the
            ``metadata.generated_from`` field. Lets downstream tools
            (FunscriptForge, ForgePlayer) detect drift between the
            funscript and a re-edited source video, render per-chapter
            context without needing to re-load the chapters sidecar, and
            audit which recipe was used per chapter. Conventional shape::

                {
                  "tool": "videoflow",
                  "tool_version": "0.0.7-alpha",
                  "source": {
                      "path": "C:/.../track.mp4",
                      "duration_ms": 845000,
                      "partial_md5": "abc123…",
                      "size_bytes": 12345678,
                  },
                  "chapters": [{"at_ms": 0, "end_ms": 60000}, …],
                  "recipes": [{"source": "percussive",
                               "stroke_density": "half",
                               "tone": "flat",
                               "emphasize_beats": false}, …],
                }

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
    metadata: dict = {}
    if title:
        metadata["title"] = title
    if generated_from is not None:
        metadata["generated_from"] = generated_from
    if metadata:
        data["metadata"] = metadata

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
    center: int | None = None,
    center_trajectory: tuple[int, int] | None = None,
    tone_per_stanza: list[tuple[int, int, int, int]] | None = None,
    energy_normalize: bool = False,
    stroke_density: object = "half",
    depth_model: str = "fixed",
    gate: float = 0.10,
    title: str = "",
    on_progress: "OnProgress | None" = None,
    progress_callback: "Callable[[str], None] | None" = None,
) -> Path:
    """Full pipeline: :class:`~videoflow.audio.AudioBeatMap` → ``.funscript``.

    Convenience wrapper that runs all four steps in sequence::

        beats_to_curve → classify_modes → shape_curve → export_funscript

    Args:
        beat_map: Result of :func:`~videoflow.audio.analyze_beats`.
        output: Destination ``.funscript`` file path.
        low: Trough position. Default 10.
        high: Maximum peak position. Default 90.
        center: If set, use the centered stroke model (peaks above /
            troughs below ``center``, both energy-scaled). See
            :func:`beats_to_curve` for details. Default ``None``.
        title: Optional title stored in the file metadata.
        on_progress: Optional :data:`videoflow.progress.OnProgress`
            callback. Receives :class:`~videoflow.progress.StageEvent`
            records (start / progress / complete) for each of the four
            pipeline stages so UIs can render a progress tree.

    Returns:
        Path to the written ``.funscript`` file.

    Example::

        beat_map = analyze_beats("track.mp3", source="percussive")
        path = generate_from_beats(beat_map, "track.funscript", center=50)
    """
    from videoflow.progress import ProgressReporter

    def _progress(label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass

    reporter = ProgressReporter(on_progress)
    with reporter.stage("generate"):
        # Classify first — the fixed-depth model folds mode-aware partials
        # into the curve, so modes must exist before beats_to_curve runs.
        with reporter.stage("classify"):
            _progress("Classifying stanza modes…")
            modes = classify_modes(beat_map)
            reporter.complete(summary=f"{len(modes)} mode segments")

        with reporter.stage("curve"):
            _progress("Generating motion curve…")
            curve = beats_to_curve(
                beat_map,
                low=low, high=high, center=center,
                center_trajectory=center_trajectory,
                tone_per_stanza=tone_per_stanza,
                energy_normalize=energy_normalize,
                stroke_density=stroke_density,
                depth_model=depth_model,
                gate=gate,
                modes=modes,
            )
            reporter.complete(summary=f"{len(curve)} curve points")

        with reporter.stage("shape"):
            if depth_model == "fixed":
                # Shaping is folded into the fixed-depth curve — skip the
                # legacy compression pass (it would collapse the rails).
                _progress("Fixed-depth curve — shaping folded in…")
                shaped = curve
                reporter.complete(summary=f"{len(shaped)} actions (fixed)")
            else:
                _progress("Shaping curve per mode…")
                shaped = shape_curve(
                    curve, modes,
                    low=low, center=center,
                    center_trajectory=center_trajectory,
                    tone_per_stanza=tone_per_stanza,
                )
                reporter.complete(summary=f"{len(shaped)} actions shaped")

        with reporter.stage("export"):
            _progress("Writing funscript…")
            path = export_funscript(shaped, output, title=title)
            reporter.complete(summary=f"wrote {path.name}")

        reporter.complete(
            summary=f"{len(shaped)} actions → {Path(output).name}",
        )
    return path


# ---------------------------------------------------------------------------
# Per-chapter generation — one recipe per chapter
# ---------------------------------------------------------------------------

def _slice_beat_map(bm: AudioBeatMap, start_ms: int, end_ms: int) -> AudioBeatMap:
    """Return a new :class:`AudioBeatMap` restricted to ``[start_ms, end_ms)``.

    Beats and energy keep their absolute timestamps so generated actions
    line up with the source media when concatenated. Stanzas that overlap
    the window are kept intact (a stanza straddling a chapter boundary is
    classified by whichever chapter contains its centre — that's a
    follow-up; for now both chapters see it)."""
    beat_indices = [i for i, b in enumerate(bm.beats) if start_ms <= b < end_ms]
    sliced_beats = [bm.beats[i] for i in beat_indices]
    sliced_energy = (
        [bm.energy[i] for i in beat_indices] if bm.energy else []
    )
    sliced_downbeats = [d for d in bm.downbeats if start_ms <= d < end_ms]
    sliced_stanzas = [(s, e) for (s, e) in bm.stanzas if e > start_ms and s < end_ms]
    return AudioBeatMap(
        bpm=bm.bpm,
        beats=sliced_beats,
        downbeats=sliced_downbeats,
        stanzas=sliced_stanzas,
        energy=sliced_energy,
        duration_ms=max(0, end_ms - start_ms),
    )


def generate_from_beats_per_chapter(
    beat_map: AudioBeatMap,
    chapters: list[dict],
    recipes: list[dict],
    output: str | Path,
    *,
    low: int = 10,
    high: int = 90,
    depth_model: str = "fixed",
    gate: float = 0.10,
    title: str = "",
    generated_from: dict | None = None,
    progress_callback: "Callable[[str], None] | None" = None,
) -> Path:
    """Per-chapter funscript generation.

    Loops chapters and runs the curve / classify / shape pipeline on a
    sliced beat-map for each chapter with that chapter's recipe. The
    resulting actions are concatenated (sorted by absolute time, dedup'd)
    and written as a single funscript via :func:`export_funscript`.

    ``chapters`` is the list of ``{"at_ms": int, "end_ms": int|None}``
    entries; ``recipes`` is the parallel list of
    ``{"source", "stroke_density", "tone", "emphasize_beats"}`` dicts.
    Lengths must match exactly — strict by design (forgegen always sends
    one recipe per chapter; a mismatch is a bug worth surfacing).

    The ``source`` field on each recipe is informational at this layer —
    re-running HPSS per chapter would require re-loading the audio. The
    bridge sends the same source on every recipe (which was used at
    analyze time); we record it in the funscript metadata regardless so
    downstream tools see what mix the run was authored against.
    """
    if len(recipes) != len(chapters):
        raise GenerateError(
            f"recipe-bundle has {len(recipes)} recipes but {len(chapters)} "
            f"chapters — counts must match exactly."
        )

    def _progress(label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(label)
            except Exception:
                pass

    all_actions: list[tuple[int, int]] = []
    for idx, (chapter, recipe) in enumerate(zip(chapters, recipes)):
        _progress(f"Generating chapter {idx + 1}/{len(chapters)}…")
        start_ms = int(chapter.get("at_ms", 0))
        end_ms = chapter.get("end_ms")
        if end_ms is None:
            end_ms = beat_map.duration_ms
        end_ms = int(end_ms)

        sliced = _slice_beat_map(beat_map, start_ms, end_ms)
        if not sliced.beats:
            continue  # no beats in this chapter window — silent skip

        # Resolve tone for this chapter's recipe.
        traj = None
        tone_per_stanza = None
        tone = recipe.get("tone", "flat")
        if tone == "rise":
            traj = (30, 70)
        elif tone == "fall":
            traj = (70, 30)
        elif tone == "auto":
            tone_per_stanza = compute_auto_tone(sliced)

        density = recipe.get("stroke_density", "half")

        modes = classify_modes(sliced)
        curve = beats_to_curve(
            sliced,
            low=low, high=high,
            center_trajectory=traj,
            tone_per_stanza=tone_per_stanza,
            stroke_density=density,
            depth_model=depth_model,
            gate=gate,
            modes=modes,
        )
        if depth_model == "fixed":
            shaped = curve  # fixed-depth folds shaping in
        else:
            shaped = shape_curve(
                curve, modes,
                low=low,
                center_trajectory=traj,
                tone_per_stanza=tone_per_stanza,
            )
        all_actions.extend(shaped)

    if not all_actions:
        raise GenerateError("no actions produced across any chapter")

    return export_funscript(
        all_actions, output,
        title=title,
        generated_from=generated_from,
    )
