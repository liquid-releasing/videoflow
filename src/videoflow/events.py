"""Funscript metadata events — read / write scaffolding.

The events layer overlays modulation, accents, edge holds, vocal cues,
scene accents, and similar nuance markers on top of a funscript's
position curve. forgegen emits canonical curves; forgevents (and other
consumers) layer events on top. restim and ForgePlayer consume the
combined `actions + metadata.events` at playback.

This module is the **format only** in v0.0.4 — read / write of events as
part of a funscript file's `metadata.events` array. Event auto-detection
and the type-vocabulary classifier are v0.0.5+ territory (see
forgegen's `docs/architecture/analysis-schema.md` and
`auto-detection.md`).

Event shape (per analysis-schema.md):

- ``type`` (str) — event-type identifier; vocabulary is open
- ``at_ms`` (int) — start time in milliseconds
- ``duration_ms`` (int | None) — for durational events (edge_hold etc.)
- ``confidence`` (float, 0–1) — auto-finder confidence; 1.0 = human
- ``source`` (list[str]) — which signals contributed
- ``params`` (dict) — type-specific parameters

Example::

    from videoflow.events import (
        FunscriptEvent, read_events, write_events
    )

    events = [
        FunscriptEvent(type="accent", at_ms=12345, confidence=1.0),
        FunscriptEvent(
            type="edge_hold", at_ms=42000, duration_ms=8000,
            confidence=0.78, source=["audio_peak"],
        ),
    ]
    write_events("track.funscript", events)

    same = read_events("track.funscript")
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


class EventError(RuntimeError):
    """Raised when reading or writing events fails."""


@dataclasses.dataclass
class FunscriptEvent:
    """One event marker on a funscript's timeline.

    Attributes:
        type: Event-type identifier (e.g. ``"accent"``, ``"edge_hold"``,
            ``"climax_candidate"``). The vocabulary is intentionally open
            in v0.0.4 so the module is not coupled to any particular
            classifier; consumers and producers are responsible for
            agreeing on the strings they emit and read.
        at_ms: Start time in milliseconds.
        duration_ms: Duration in milliseconds for durational events
            (edge holds, tight-cut zones). ``None`` for point events.
        confidence: Auto-finder confidence in ``[0.0, 1.0]``. ``1.0``
            indicates a human-authored or human-confirmed event.
        source: Which signals contributed (e.g. ``["audio_peak",
            "video_peak"]``). Empty list = no provenance recorded.
        params: Type-specific parameters. Free-form dict; serialised
            as JSON without further validation.
    """

    type: str
    at_ms: int
    duration_ms: int | None = None
    confidence: float = 1.0
    source: list[str] = dataclasses.field(default_factory=list)
    params: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Omits ``duration_ms`` when ``None`` so point events stay compact.
        """
        out: dict = {
            "type": self.type,
            "at_ms": self.at_ms,
            "confidence": self.confidence,
            "source": list(self.source),
            "params": dict(self.params),
        }
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "FunscriptEvent":
        """Parse one event dict.

        Raises:
            EventError: If required fields are missing or the wrong type.
        """
        try:
            type_ = str(data["type"])
            at_ms = int(data["at_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EventError(
                f"event missing required field 'type' or 'at_ms': {data!r}"
            ) from exc

        duration_ms = data.get("duration_ms")
        if duration_ms is not None:
            try:
                duration_ms = int(duration_ms)
            except (TypeError, ValueError) as exc:
                raise EventError(
                    f"event 'duration_ms' must be int or null: {data!r}"
                ) from exc

        try:
            confidence = float(data.get("confidence", 1.0))
        except (TypeError, ValueError) as exc:
            raise EventError(
                f"event 'confidence' must be a number: {data!r}"
            ) from exc

        source = list(data.get("source") or [])
        params = dict(data.get("params") or {})

        return cls(
            type=type_,
            at_ms=at_ms,
            duration_ms=duration_ms,
            confidence=confidence,
            source=source,
            params=params,
        )


def events_to_dicts(events: list[FunscriptEvent]) -> list[dict]:
    """Serialise a list of events for JSON storage."""
    return [e.to_dict() for e in events]


def events_from_dicts(data: list) -> list[FunscriptEvent]:
    """Parse a list of event dicts.

    Raises:
        EventError: If *data* is not a list or contains malformed events.
    """
    if not isinstance(data, list):
        raise EventError(
            f"events payload must be a list, got {type(data).__name__}"
        )
    return [FunscriptEvent.from_dict(d) for d in data]


def read_events(funscript_path: str | Path) -> list[FunscriptEvent]:
    """Read events from a funscript file's ``metadata.events`` array.

    Returns an empty list when the file has no events (so callers can
    treat "no events" and "events == []" as the same thing).

    Args:
        funscript_path: Path to a ``.funscript`` JSON file.

    Returns:
        List of :class:`FunscriptEvent` ordered as stored on disk.

    Raises:
        FileNotFoundError: If the file does not exist.
        EventError: If the file is not valid JSON or events are malformed.
    """
    path = Path(funscript_path)
    if not path.exists():
        raise FileNotFoundError(f"Funscript not found: {path}")

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventError(f"Funscript {path} is not valid JSON: {exc}") from exc

    metadata = doc.get("metadata") or {}
    raw = metadata.get("events", [])
    return events_from_dicts(raw)


def write_events(
    funscript_path: str | Path,
    events: list[FunscriptEvent],
) -> Path:
    """Embed *events* into a funscript file's ``metadata.events`` array.

    Replaces any existing events in the file. Other metadata fields and
    the ``actions`` array are preserved unchanged. Events are sorted by
    ``at_ms`` before write so on-disk order is stable.

    Args:
        funscript_path: Path to an existing ``.funscript`` JSON file.
        events: Events to write.

    Returns:
        The path that was written.

    Raises:
        FileNotFoundError: If the file does not exist.
        EventError: If the file is not valid JSON.
    """
    path = Path(funscript_path)
    if not path.exists():
        raise FileNotFoundError(f"Funscript not found: {path}")

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EventError(f"Funscript {path} is not valid JSON: {exc}") from exc

    sorted_events = sorted(events, key=lambda e: e.at_ms)
    metadata = doc.setdefault("metadata", {})
    metadata["events"] = events_to_dicts(sorted_events)

    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
