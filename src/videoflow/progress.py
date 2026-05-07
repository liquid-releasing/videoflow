"""Progress reporting for long-running videoflow operations.

Engine functions emit :class:`StageEvent` records to an ``on_progress``
callback. UI layers (PySide6, CLI, tests) interpret the stream to render
trees of work, ETAs, and post-hoc summaries.

Three event kinds:

  * ``"start"``    — a stage begins. UI opens a node in its tree.
  * ``"progress"`` — fractional update within an open stage. UI updates
                     the progress bar / ETA on that node.
  * ``"complete"`` — a stage finishes. UI marks the node done with
                     elapsed time and an optional summary line.

Stage paths are tuples (e.g. ``("analyze", "beats", "chapter 3/12")``) so
the UI can build a hierarchy. A child is implicit by path: a new event
whose path extends an open stage's path nests beneath it.

Typical callsite::

    from videoflow.progress import ProgressReporter

    reporter = ProgressReporter(on_progress)
    with reporter.stage("analyze"):
        with reporter.stage("decode"):
            decode_audio(...)
            reporter.complete(summary="22 050 Hz mono")
        with reporter.stage("beats"):
            for i, chunk in enumerate(chunks):
                with reporter.stage(f"chapter {i+1}/{len(chunks)}"):
                    detect_beats(chunk)
                    reporter.complete(summary=f"{n} beats")

Compatibility shim: legacy ``progress_callback: Callable[[str], None]``
keeps working via :func:`adapt_string_callback` — the adapter forwards
only the leaf stage name on ``"start"`` events and drops the rest.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

ProgressEventKind = Literal["start", "progress", "complete"]


@dataclass(frozen=True)
class StageEvent:
    """A single progress event emitted during a long-running operation."""

    kind: ProgressEventKind
    stage_path: tuple[str, ...]
    progress: float | None = None
    """Fractional progress within the current stage, 0.0–1.0. Set on
    ``"progress"`` events; ``None`` on ``"start"`` and ``"complete"``."""

    eta_seconds: float | None = None
    """Estimated seconds remaining for the current stage. May be ``None``
    on the very first run before the estimator has data."""

    elapsed_seconds: float | None = None
    """Wall-clock seconds the stage took, set on ``"complete"`` events."""

    summary: str | None = None
    """Optional one-line description of what the stage produced
    (e.g. ``"8 431 beats, BPM 124"``). Set on ``"complete"`` events."""

    message: str | None = None
    """Optional human-readable detail for ``"progress"`` events
    (e.g. ``"loading 22 050 Hz mono"``)."""

    @property
    def stage(self) -> str:
        """Convenience accessor for the leaf stage name."""
        return self.stage_path[-1] if self.stage_path else ""


OnProgress = Callable[[StageEvent], None]
"""Callback signature for receiving progress events."""


# ---------------------------------------------------------------------------
# ETAEstimator — learns observed stage durations across runs
# ---------------------------------------------------------------------------

DEFAULT_TIMINGS_PATH = Path.home() / ".lqr" / "videoflow-timings.json"
"""Default location for the persisted timings cache."""

_EMA_ALPHA = 0.3


class ETAEstimator:
    """Learns observed stage durations and predicts ETA for in-flight stages.

    Each stage path keys an exponential moving average of observed
    durations; predictions for an in-progress stage scale linearly with
    reported progress. Persists to a small JSON file so the second run
    of the same operation has a calibrated ETA from the first event.

    First-time stages return ``None`` from :meth:`predict` until they
    complete once.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_TIMINGS_PATH
        self._timings: dict[str, float] = {}
        self._load()

    @staticmethod
    def _key(stage_path: tuple[str, ...]) -> str:
        return ":".join(stage_path)

    def observe(self, stage_path: tuple[str, ...], elapsed: float) -> None:
        """Record an observed stage duration. Updates the EMA in memory."""
        key = self._key(stage_path)
        prior = self._timings.get(key)
        if prior is None:
            self._timings[key] = elapsed
        else:
            self._timings[key] = (1 - _EMA_ALPHA) * prior + _EMA_ALPHA * elapsed

    def predict(
        self,
        stage_path: tuple[str, ...],
        progress: float | None = None,
    ) -> float | None:
        """Predict remaining seconds for a stage at given progress.

        Returns ``None`` if no prior observation exists for this stage.
        """
        key = self._key(stage_path)
        total = self._timings.get(key)
        if total is None:
            return None
        if progress is None:
            return total
        return max(0.0, total * (1.0 - progress))

    def save(self) -> None:
        """Persist current timings to disk. Failures are swallowed —
        timing telemetry never breaks an analysis."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._timings, indent=2))
        except OSError:
            pass

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                self._timings = {
                    str(k): float(v)
                    for k, v in data.items()
                    if isinstance(v, (int, float))
                }
        except (json.JSONDecodeError, OSError, ValueError):
            self._timings = {}


# ---------------------------------------------------------------------------
# ProgressReporter — stage-stack manager that emits StageEvents
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Stack-based stage tracker that emits paired start / complete events.

    Use as a context manager via :meth:`stage`. The stack-based API
    guarantees a ``"complete"`` event fires even on exceptions —
    ``__exit__`` always emits with the elapsed time.
    """

    def __init__(
        self,
        on_progress: OnProgress | None = None,
        estimator: ETAEstimator | None = None,
    ) -> None:
        self.on_progress = on_progress
        self.estimator = estimator
        self._stack: list[tuple[tuple[str, ...], float]] = []
        self._summary_for_active: str | None = None

    def _emit(self, event: StageEvent) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(event)
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str) -> Iterator["ProgressReporter"]:
        """Open a stage as a context. Emits ``start`` on enter,
        ``complete`` on exit."""
        parent_path = self._stack[-1][0] if self._stack else ()
        path = parent_path + (name,)
        start = time.monotonic()
        prior_summary = self._summary_for_active
        self._summary_for_active = None
        self._stack.append((path, start))

        eta = self.estimator.predict(path) if self.estimator else None
        self._emit(StageEvent(kind="start", stage_path=path, eta_seconds=eta))

        try:
            yield self
        finally:
            popped_path, popped_start = self._stack.pop()
            elapsed = time.monotonic() - popped_start
            summary = self._summary_for_active
            self._summary_for_active = prior_summary
            if self.estimator is not None:
                self.estimator.observe(popped_path, elapsed)
            self._emit(StageEvent(
                kind="complete",
                stage_path=popped_path,
                elapsed_seconds=elapsed,
                summary=summary,
            ))

    def progress(
        self,
        fraction: float,
        *,
        message: str | None = None,
    ) -> None:
        """Report fractional progress within the currently open stage."""
        if not self._stack:
            return
        path, start = self._stack[-1]
        elapsed = time.monotonic() - start
        clamped = max(0.0, min(1.0, fraction))
        eta: float | None = None
        if self.estimator is not None:
            eta = self.estimator.predict(path, progress=clamped)
        if eta is None and clamped > 0:
            eta = elapsed * (1.0 - clamped) / clamped
        self._emit(StageEvent(
            kind="progress",
            stage_path=path,
            progress=clamped,
            eta_seconds=eta,
            message=message,
        ))

    def message(self, text: str) -> None:
        """Emit an informational update inside the current stage.

        Use this when there is a status to share but no meaningful
        fractional progress yet (e.g. ``"Detecting beats…"``). Emits a
        ``progress`` event with ``progress=None`` and ``message=text``;
        UI consumers typically render it as a status line beneath the
        active stage's spinner.
        """
        if not self._stack:
            return
        path, _ = self._stack[-1]
        self._emit(StageEvent(
            kind="progress",
            stage_path=path,
            progress=None,
            message=text,
        ))

    def complete(self, *, summary: str | None = None) -> None:
        """Set the summary line for the currently-open stage. The
        summary is included in the ``complete`` event when the stage
        context exits."""
        self._summary_for_active = summary

    def save_timings(self) -> None:
        """Persist accumulated stage timings via the estimator, if any."""
        if self.estimator is not None:
            self.estimator.save()


# ---------------------------------------------------------------------------
# Compatibility shim for legacy progress_callback callers
# ---------------------------------------------------------------------------

def adapt_string_callback(
    legacy: Callable[[str], None] | None,
) -> OnProgress | None:
    """Wrap a legacy ``Callable[[str], None]`` into the new event signature.

    The adapter forwards only the leaf stage name on ``"start"`` events
    so existing UIs that show "currently doing X" keep working without
    modification.
    """
    if legacy is None:
        return None

    def _adapter(event: StageEvent) -> None:
        if event.kind == "start" and event.stage_path:
            try:
                legacy(event.stage_path[-1])
            except Exception:
                pass

    return _adapter
