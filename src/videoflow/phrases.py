"""Phrase classification — audio-driven and funscript-driven.

A phrase is a within-chapter intent unit. Two classifiers, one record:

- :func:`classify_phrases` reads :class:`videoflow.audio.AudioBeatMap`
  and labels each musical phrase with a behavioural mode (``tease``,
  ``steady``, ``edging``, etc.) using audio-energy and tempo signals.
- :func:`classify_phrases_from_funscript` reads a funscript's actions
  and applies the same mode vocabulary using stroke-amplitude and
  stroke-rate signals instead of audio energy. Useful when a user has
  hand-edited the funscript and wants phrase classifications that
  match the edited curve rather than the source audio.

Both produce the same :class:`Phrase` shape — only the ``source`` field
differs (``"audio"`` vs ``"funscript"``). Records round-trip through the
sidecar via :meth:`Phrase.to_dict` / :meth:`Phrase.from_dict` and feed
the merge contract in :mod:`videoflow.sidecar`.

The legacy tuple-shaped :func:`classify_modes` API is preserved here as
a thin wrapper for back-compat with v0.0.4 callers (forgegen, the CLI,
and existing tests). New code should prefer :func:`classify_phrases`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from videoflow.audio import AudioBeatMap
    from videoflow.chapters import Chapter


# ---------------------------------------------------------------------------
# Phrase record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Phrase:
    """A within-chapter intent unit with classified mode + provenance.

    See ``videoflow/docs/architecture/audio-structure-primitive.md``
    for the field categories (STRUCTURAL / AUTHORED / ANALYTICAL / MIXED
    / LATCH) that drive the sidecar's field-level merge contract.

    Attributes:
        chapter_idx: 0-based index into the sidecar's chapters list.
            STRUCTURAL.
        at_ms: Phrase start in milliseconds. STRUCTURAL.
        end_ms: Phrase end in milliseconds. STRUCTURAL.
        mode: Behavioural mode label — one of ``tease`` / ``steady`` /
            ``edging`` / ``break`` / ``fast`` / ``slow`` (or empty
            string for unclassified). ANALYTICAL.
        confidence: Classification confidence in ``[0.0, 1.0]`` or
            ``None`` when the classifier is rule-based and binary.
            ANALYTICAL.
        source: Where the classification came from. ``"audio"`` =
            :func:`classify_phrases` from an AudioBeatMap; ``"funscript"``
            = :func:`classify_phrases_from_funscript`; ``"user"`` = a UI
            edit; ``""`` = unknown / from a sidecar that didn't record
            it. ANALYTICAL.
        intent: Phrase-intent vocabulary (``intro`` / ``build`` /
            ``sustain`` / ``edge`` / ``climax`` / ``recover`` /
            ``outro``). AUTHORED.
        tone: Optional FunscriptForge tone override for this phrase.
            MIXED.
        evidence: Identifiers of the features that fired. ANALYTICAL.
        auto_generated: Per-record latch. ``False`` freezes the record
            on regen. LATCH.
    """

    chapter_idx: int
    at_ms: int
    end_ms: int
    mode: str = ""
    confidence: float | None = None
    source: str = ""
    intent: str = ""
    tone: str = ""
    evidence: list[str] = dataclasses.field(default_factory=list)
    auto_generated: bool = True

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Omits optional fields at their defaults so authored phrases
        stay compact. The shape matches the schema documented in
        ``audio-structure-primitive.md``.
        """
        out: dict = {
            "chapter_idx": int(self.chapter_idx),
            "at_ms": int(self.at_ms),
            "end_ms": int(self.end_ms),
        }
        if self.mode:
            out["mode"] = self.mode
        if self.confidence is not None:
            out["confidence"] = float(self.confidence)
        if self.source:
            out["source"] = self.source
        if self.intent:
            out["intent"] = self.intent
        if self.tone:
            out["tone"] = self.tone
        if self.evidence:
            out["evidence"] = list(self.evidence)
        out["auto_generated"] = bool(self.auto_generated)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Phrase":
        """Parse a phrase dict from a sidecar.

        Raises:
            ValueError: If ``chapter_idx`` / ``at_ms`` / ``end_ms`` are
                missing.
        """
        for required in ("chapter_idx", "at_ms", "end_ms"):
            if required not in data:
                raise ValueError(f"phrase missing '{required}': {data!r}")
        confidence = data.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
        evidence_raw = data.get("evidence") or []
        evidence = (
            [str(e) for e in evidence_raw]
            if isinstance(evidence_raw, list) else []
        )
        return cls(
            chapter_idx=int(data["chapter_idx"]),
            at_ms=int(data["at_ms"]),
            end_ms=int(data["end_ms"]),
            mode=str(data.get("mode") or ""),
            confidence=confidence,
            source=str(data.get("source") or ""),
            intent=str(data.get("intent") or ""),
            tone=str(data.get("tone") or ""),
            evidence=evidence,
            auto_generated=bool(data.get("auto_generated", True)),
        )


# ---------------------------------------------------------------------------
# Audio-driven classification
# ---------------------------------------------------------------------------

def classify_modes(
    beat_map: "AudioBeatMap",
    *,
    chapters: list | None = None,
) -> list[tuple[int, int, str]]:
    """Label each musical phrase with a behavioural mode (tuple API).

    Modes are rule-based — no ML. Two regimes:

    **Whole-file** (default, ``chapters=None``): phrase-level energy
    compared against absolute thresholds.

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

    **Chapter-aware** (``chapters`` is a list of
    :class:`videoflow.chapters.Chapter`): each chapter computes its own
    energy reference (median + 75th percentile) and BPM (from beat
    density within the chapter). Phrases are classified relative to
    their chapter's stats so a quiet phrase in a quiet ambient chunk
    isn't auto-mapped to ``break`` just because its absolute energy is
    low — it can still be ``steady`` or ``slow`` if it's average for
    its chapter. This is the lever that fixes the ambient flat-output
    problem the whole-file regime suffers from.

    Args:
        beat_map: Result of :func:`videoflow.audio.analyze_beats`.
        chapters: Optional chapter list (typically from
            :func:`videoflow.structural.auto_chapter`).

    Returns:
        List of ``(start_ms, end_ms, mode)`` tuples, one per phrase.
        Tuple-shaped for back-compat with v0.0.4 callers; new code
        should prefer :func:`classify_phrases`, which returns rich
        :class:`Phrase` records.
    """
    if chapters:
        return _classify_modes_per_chapter(beat_map, chapters)
    return _classify_modes_whole(beat_map)


def classify_phrases(
    beat_map: "AudioBeatMap",
    *,
    chapters: list | None = None,
) -> list[Phrase]:
    """Audio-driven phrase classification with rich :class:`Phrase` records.

    Same rules as :func:`classify_modes` (lifted here unchanged); the
    difference is the return type — :class:`Phrase` records carry
    ``chapter_idx``, ``source="audio"``, and round-trip cleanly through
    the sidecar.

    Args:
        beat_map: Result of :func:`videoflow.audio.analyze_beats`.
        chapters: Optional chapter list. Used both as a classification
            regime selector (per-chapter relative thresholds) and to
            attach ``chapter_idx`` to each phrase record.

    Returns:
        list[Phrase] with ``source="audio"`` on every record.
    """
    chapter_list = list(chapters) if chapters else []
    tuples = classify_modes(beat_map, chapters=chapter_list or None)
    return [
        Phrase(
            chapter_idx=_chapter_index_at(
                (start_ms + end_ms) // 2, chapter_list,
            ) if chapter_list else 0,
            at_ms=int(start_ms),
            end_ms=int(end_ms),
            mode=mode or "",
            source="audio",
            auto_generated=True,
        )
        for start_ms, end_ms, mode in tuples
    ]


def _classify_modes_whole(
    beat_map: "AudioBeatMap",
) -> list[tuple[int, int, str]]:
    """Original whole-file classification — kept for back-compat with the
    v0.0.4 API and used for short-file analysis where chapter awareness
    adds no value."""
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
        trend, _ = _energy_trend(phrase_energy)

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


def _classify_modes_per_chapter(
    beat_map: "AudioBeatMap",
    chapters: list,
) -> list[tuple[int, int, str]]:
    """Chunk-relative classification.

    For each chapter, computes per-chunk median + 75th-percentile beat
    energies and per-chunk BPM (from beat density inside the chunk),
    then classifies each phrase against THAT chunk's stats. Result:
    ambient chapters land mostly in ``steady`` instead of being crushed
    to ``break`` / ``tease`` by a whole-file 0.15 / 0.30 cut.
    """
    out: list[tuple[int, int, str]] = []
    for ch in chapters:
        ch_start = ch.at_ms
        ch_end = ch.end_ms if ch.end_ms is not None else beat_map.duration_ms

        chunk_phrases = [
            (s, e) for s, e in beat_map.phrases if ch_start <= s < ch_end
        ]
        if not chunk_phrases:
            continue

        chunk_beat_indices = [
            i for i, b in enumerate(beat_map.beats) if ch_start <= b < ch_end
        ]
        chunk_energies = [beat_map.energy[i] for i in chunk_beat_indices]

        if not chunk_energies:
            for s, e in chunk_phrases:
                out.append((s, e, "break"))
            continue

        sorted_e = sorted(chunk_energies)
        n = len(sorted_e)
        if n % 2 == 1:
            chunk_median = sorted_e[n // 2]
        else:
            chunk_median = (sorted_e[n // 2 - 1] + sorted_e[n // 2]) / 2.0

        chunk_duration_min = max(1e-6, (ch_end - ch_start) / 60_000.0)
        chunk_bpm = len(chunk_beat_indices) / chunk_duration_min

        # break threshold floors at 0.10 absolute so true silence is
        # still detected even in chunks whose median is itself near
        # silent. Without the floor, a fully-silent chunk would get
        # `chunk_median ≈ 0` and nothing would qualify as break.
        break_threshold = max(chunk_median * 0.4, 0.10)
        tease_threshold = chunk_median * 0.85

        for start_ms, end_ms in chunk_phrases:
            ph_indices = [
                i for i, b in enumerate(beat_map.beats)
                if start_ms <= b < end_ms
            ]
            if not ph_indices:
                out.append((start_ms, end_ms, "break"))
                continue

            phrase_energy = [beat_map.energy[i] for i in ph_indices]
            avg = sum(phrase_energy) / len(phrase_energy)
            trend, second_avg = _energy_trend(phrase_energy)

            if avg < break_threshold:
                mode = "break"
            elif trend >= 0.15 and second_avg >= chunk_median:
                mode = "edging"
            elif chunk_bpm >= 140:
                mode = "fast"
            elif chunk_bpm <= 75:
                mode = "slow"
            elif avg < tease_threshold:
                mode = "tease"
            else:
                mode = "steady"

            out.append((start_ms, end_ms, mode))
    return out


def _energy_trend(phrase_energy: list[float]) -> tuple[float, float]:
    """Per-phrase energy slope: returns ``(trend, second_avg)`` where trend
    is second-half-average minus first-half-average. Returning second_avg
    too lets callers test "is the phrase peaking?" without recomputing.
    """
    mid = len(phrase_energy) // 2
    if mid <= 0:
        avg = phrase_energy[0] if phrase_energy else 0.0
        return 0.0, avg
    first_avg = sum(phrase_energy[:mid]) / mid
    second_avg = sum(phrase_energy[mid:]) / max(1, len(phrase_energy) - mid)
    return second_avg - first_avg, second_avg


# ---------------------------------------------------------------------------
# Funscript-driven classification
# ---------------------------------------------------------------------------

# Default phrase boundary derivation: group every N actions into one phrase.
# 16 mirrors the audio path's 16-beat grouping (4 bars of 4/4).
_DEFAULT_FUNSCRIPT_PHRASE_ACTIONS = 16


def classify_phrases_from_funscript(
    actions: Iterable[dict],
    *,
    chapters: list | None = None,
    phrase_boundaries: list[tuple[int, int]] | None = None,
) -> list[Phrase]:
    """Classify phrases from funscript actions instead of audio.

    Funscript actions carry timing (``at`` ms) and position (``pos``,
    0–100). Stroke amplitude (peak-to-trough range), stroke density
    (actions per second), and amplitude trend serve as the analogues
    of audio energy, BPM, and energy trend used by :func:`classify_modes`.

    The same six modes are produced — ``break`` / ``tease`` / ``edging``
    / ``fast`` / ``slow`` / ``steady`` — using the rules below. Records
    carry ``source="funscript"`` so consumers can distinguish them from
    audio-derived classifications.

    ======== ================================================================
    Mode     Rule
    ======== ================================================================
    break    Stroke amplitude < 15/100 — near-static, almost no movement
    tease    Amplitude < 30/100 — restrained
    edging   Amplitude rising (second-half range > first-half by ≥ 15)
             and second-half amplitude ≥ chapter median (or 35 in
             whole-file mode)
    fast     Density ≥ 4 actions/sec — rapid stroking
    slow     Density ≤ 0.8 actions/sec — sparse strokes
    steady   Everything else
    ======== ================================================================

    Args:
        actions: Funscript actions, each ``{"at": ms_int, "pos": 0..100}``.
            Order is sorted internally.
        chapters: Optional chapter list. When provided, classification
            is chapter-relative (chunk amplitude median sets the
            reference) and each phrase's ``chapter_idx`` is set
            accordingly. When absent, classification uses absolute
            thresholds.
        phrase_boundaries: Optional explicit ``(start_ms, end_ms)``
            phrase segmentation. When ``None``, actions are grouped
            into 16-action phrases (mirrors AudioBeatMap's 16-beat
            grouping).

    Returns:
        list[Phrase] with ``source="funscript"`` on every record.
    """
    sorted_actions = sorted(
        ({"at": int(a["at"]), "pos": int(a["pos"])} for a in actions),
        key=lambda a: a["at"],
    )
    if not sorted_actions:
        return []

    chapter_list = list(chapters) if chapters else []

    if phrase_boundaries is None:
        phrase_boundaries = _segment_actions_into_phrases(sorted_actions)

    # Build per-chapter reference stats when chapters are provided.
    chapter_stats: dict[int, _ChapterFunscriptStats] = {}
    if chapter_list:
        for i, ch in enumerate(chapter_list):
            ch_end = (
                ch.end_ms if ch.end_ms is not None
                else sorted_actions[-1]["at"]
            )
            in_chunk = [
                a for a in sorted_actions
                if ch.at_ms <= a["at"] < ch_end
            ]
            chapter_stats[i] = _ChapterFunscriptStats.from_actions(
                in_chunk, duration_ms=ch_end - ch.at_ms,
            )

    out: list[Phrase] = []
    for start_ms, end_ms in phrase_boundaries:
        ph_actions = [
            a for a in sorted_actions if start_ms <= a["at"] < end_ms
        ]
        ch_idx = (
            _chapter_index_at((start_ms + end_ms) // 2, chapter_list)
            if chapter_list else 0
        )
        chapter_ref = chapter_stats.get(ch_idx) if chapter_list else None

        mode = _classify_funscript_phrase(
            ph_actions, start_ms, end_ms, chapter_ref=chapter_ref,
        )
        out.append(Phrase(
            chapter_idx=ch_idx,
            at_ms=int(start_ms),
            end_ms=int(end_ms),
            mode=mode,
            source="funscript",
            auto_generated=True,
        ))
    return out


@dataclasses.dataclass
class _ChapterFunscriptStats:
    """Per-chapter reference stats for the funscript-driven classifier."""

    median_amplitude: float
    density: float

    @classmethod
    def from_actions(
        cls, actions: list[dict], *, duration_ms: int,
    ) -> "_ChapterFunscriptStats":
        if not actions:
            return cls(median_amplitude=0.0, density=0.0)
        positions = [a["pos"] for a in actions]
        amplitude = max(positions) - min(positions)
        # Single-amplitude per chunk (peak-to-trough); finer per-cycle
        # ranges are a future refinement.
        return cls(
            median_amplitude=float(amplitude),
            density=(
                len(actions) * 1000.0 / duration_ms
                if duration_ms > 0 else 0.0
            ),
        )


def _classify_funscript_phrase(
    ph_actions: list[dict],
    start_ms: int,
    end_ms: int,
    *,
    chapter_ref: _ChapterFunscriptStats | None,
) -> str:
    """Pick a mode for one phrase's worth of funscript actions."""
    if len(ph_actions) < 2:
        return "break"

    positions = [a["pos"] for a in ph_actions]
    amplitude = max(positions) - min(positions)
    duration_ms = max(1, end_ms - start_ms)
    density = len(ph_actions) * 1000.0 / duration_ms

    mid = len(ph_actions) // 2
    first_half = positions[:mid]
    second_half = positions[mid:]
    first_amp = (max(first_half) - min(first_half)) if first_half else 0
    second_amp = (max(second_half) - min(second_half)) if second_half else 0
    amp_trend = second_amp - first_amp

    if chapter_ref is not None:
        # Chapter-relative thresholds. Floor break at 10 so an entirely-
        # static chunk still has a meaningful break threshold.
        break_threshold = max(chapter_ref.median_amplitude * 0.4, 10.0)
        tease_threshold = chapter_ref.median_amplitude * 0.85
        edging_floor = chapter_ref.median_amplitude
    else:
        break_threshold = 15.0
        tease_threshold = 30.0
        edging_floor = 35.0

    if amplitude < break_threshold:
        return "break"
    if amp_trend >= 15 and second_amp >= edging_floor:
        return "edging"
    if density >= 4.0:
        return "fast"
    if density <= 0.8:
        return "slow"
    if amplitude < tease_threshold:
        return "tease"
    return "steady"


def _segment_actions_into_phrases(
    actions: list[dict],
    *,
    actions_per_phrase: int = _DEFAULT_FUNSCRIPT_PHRASE_ACTIONS,
) -> list[tuple[int, int]]:
    """Group sorted actions into ``actions_per_phrase``-sized phrases.

    Returns ``[(start_ms, end_ms), ...]`` covering every action. The last
    phrase may be smaller than ``actions_per_phrase`` and runs to the
    final action's timestamp.
    """
    if not actions:
        return []
    out: list[tuple[int, int]] = []
    n = len(actions)
    for i in range(0, n, actions_per_phrase):
        chunk = actions[i:i + actions_per_phrase]
        start_ms = chunk[0]["at"]
        # End-of-phrase = start of next phrase, or last-action timestamp + 1.
        end_idx = i + actions_per_phrase
        if end_idx < n:
            end_ms = actions[end_idx]["at"]
        else:
            end_ms = chunk[-1]["at"] + 1
        out.append((int(start_ms), int(end_ms)))
    return out


# ---------------------------------------------------------------------------
# Internal helper (mirrors structural._chapter_index_at to avoid the
# structural→phrases import that would create a cycle)
# ---------------------------------------------------------------------------

def _chapter_index_at(time_ms: int, chapters: list) -> int:
    """Return the index of the chapter containing *time_ms*."""
    for i, ch in enumerate(chapters):
        ch_end = ch.end_ms if ch.end_ms is not None else 1 << 62
        if ch.at_ms <= time_ms < ch_end:
            return i
    return max(0, len(chapters) - 1)
