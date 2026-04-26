"""Scene detection and video analysis functions."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

# Optional dependency — imported at module level so tests can patch it.
try:
    from scenedetect import detect, ContentDetector, ThresholdDetector, AdaptiveDetector  # type: ignore[import]

    _SCENEDETECT_AVAILABLE = True
except ImportError:
    detect = None  # type: ignore[assignment]
    ContentDetector = None  # type: ignore[assignment]
    ThresholdDetector = None  # type: ignore[assignment]
    AdaptiveDetector = None  # type: ignore[assignment]
    _SCENEDETECT_AVAILABLE = False


class SceneError(RuntimeError):
    """Raised when scene detection fails."""


@dataclasses.dataclass
class Scene:
    """A detected scene boundary."""

    index: int
    """1-based scene number."""

    start_ms: int
    """Scene start time in milliseconds."""

    end_ms: int
    """Scene end time in milliseconds."""

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclasses.dataclass
class DetectorInfo:
    """User-facing description of a detector — for UI sliders and help text."""

    name: str
    """Short display name."""

    description: str
    """What the detector looks for, in plain language."""

    threshold_label: str
    """Label for the threshold slider."""

    threshold_default: float
    """Recommended starting value."""

    threshold_min: float
    """Minimum useful value."""

    threshold_max: float
    """Maximum useful value."""

    threshold_low_note: str
    """What happens when the user moves the slider lower."""

    threshold_high_note: str
    """What happens when the user moves the slider higher."""

    best_for: list[str]
    """Content types this detector handles well."""

    not_great_for: list[str]
    """Content types or transitions it tends to miss or over-detect."""

    novice_tip: str
    """One-sentence plain-language advice for a first-time user."""


# ---------------------------------------------------------------------------
# Detector metadata — consumed by UI to render sliders and advisory text
# ---------------------------------------------------------------------------

DETECTOR_INFO: dict[str, DetectorInfo] = {
    "content": DetectorInfo(
        name="Content (hard cuts)",
        description=(
            "Compares the color, saturation, and brightness of consecutive frames. "
            "When two frames look significantly different, that's a cut."
        ),
        threshold_label="Cut sensitivity",
        threshold_default=27.0,
        threshold_min=5.0,
        threshold_max=100.0,
        threshold_low_note="More scenes detected — catches subtle cuts and fast camera moves. May over-split on shaky footage.",
        threshold_high_note="Fewer scenes — only major cuts like a new location or subject.",
        best_for=["live action", "edited video", "music videos", "interviews", "sports"],
        not_great_for=["slow dissolves", "fades to black", "very shaky handheld footage"],
        novice_tip=(
            "Start at 27. If you're getting too many splits on action shots, raise it toward 40. "
            "If cuts are being missed, lower it toward 15."
        ),
    ),
    "threshold": DetectorInfo(
        name="Threshold (fade to black)",
        description=(
            "Watches the average brightness of each frame. "
            "When the screen goes dark enough, that's a scene boundary."
        ),
        threshold_label="Brightness cutoff (0–255)",
        threshold_default=12.0,
        threshold_min=1.0,
        threshold_max=255.0,
        threshold_low_note="Only triggers on very dark frames — near-black transitions only.",
        threshold_high_note="Triggers on any dark scene, not just intentional fades. May split night scenes.",
        best_for=["documentary", "film", "broadcast TV", "content with fade-to-black transitions"],
        not_great_for=["hard cuts", "dissolves", "wipes", "content without fades"],
        novice_tip=(
            "Use this when your video fades to black between scenes. "
            "Leave the brightness cutoff at 12 unless your blacks aren't quite black (e.g. dark grey) — then raise it slightly."
        ),
    ),
    "adaptive": DetectorInfo(
        name="Adaptive (mixed content)",
        description=(
            "Measures how different each frame is from its neighbours, then adjusts the threshold "
            "automatically based on the local average. Works well when some sections are fast-cut "
            "and others are slow — it recalibrates instead of using a fixed bar."
        ),
        threshold_label="Adaptive sensitivity",
        threshold_default=3.0,
        threshold_min=1.0,
        threshold_max=10.0,
        threshold_low_note="More scenes — picks up subtle transitions and moderate camera moves.",
        threshold_high_note="Fewer scenes — only catches the most obvious cuts.",
        best_for=[
            "long-form video",
            "documentaries",
            "vlogs",
            "mixed content",
            "content where cut frequency varies",
        ],
        not_great_for=["fades to black (use Threshold detector instead)"],
        novice_tip=(
            "This is the best default for videos you know nothing about. "
            "If you're getting too many false splits, raise the sensitivity toward 5–6. "
            "For very subtly edited content, try lowering it to 2."
        ),
    ),
}

_DETECTORS = tuple(DETECTOR_INFO.keys())


def detect_scenes(
    input: str | Path,
    *,
    threshold: float | None = None,
    detector: str = "adaptive",
) -> list[Scene]:
    """Detect scene boundaries in a video file.

    Wraps PySceneDetect. Returns :class:`Scene` objects with start/end timestamps
    in milliseconds. Install PySceneDetect with: ``pip install scenedetect[opencv]``

    The threshold meaning and scale differ per detector — see :data:`DETECTOR_INFO`
    for per-detector defaults, ranges, and UI guidance.

    Args:
        input: Path to the video file.
        threshold: Detection sensitivity. If ``None``, uses the detector's default
            (27.0 for ``content``, 12.0 for ``threshold``, 3.0 for ``adaptive``).
            See :data:`DETECTOR_INFO` for what the value means per detector.
        detector: Which algorithm to use:

            - ``"adaptive"`` *(default)* — adjusts sensitivity to local activity;
              best for mixed or unknown content.
            - ``"content"`` — fixed frame-difference threshold; best for hard cuts
              in consistently edited video.
            - ``"threshold"`` — brightness-based; best for fade-to-black transitions.

    Returns:
        List of :class:`Scene` objects ordered by start time.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If *detector* is not a known value.
        SceneError: If PySceneDetect is not installed or detection fails.
    """
    input = Path(input)
    if not input.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    if detector not in _DETECTORS:
        raise ValueError(
            f"detector must be one of {list(_DETECTORS)!r}, got {detector!r}"
        )

    if detect is None:
        raise SceneError(
            "PySceneDetect is required for scene detection. "
            "Install it with: pip install scenedetect[opencv]"
        )

    info = DETECTOR_INFO[detector]
    t = threshold if threshold is not None else info.threshold_default

    try:
        if detector == "content":
            det = ContentDetector(threshold=t)
        elif detector == "threshold":
            det = ThresholdDetector(threshold=t)
        else:  # adaptive
            det = AdaptiveDetector(adaptive_threshold=t)

        raw = detect(str(input), det)
    except Exception as exc:
        raise SceneError(f"Scene detection failed: {exc}") from exc

    return [
        Scene(
            index=i,
            start_ms=round(start_tc.get_seconds() * 1000),
            end_ms=round(end_tc.get_seconds() * 1000),
        )
        for i, (start_tc, end_tc) in enumerate(raw, start=1)
    ]
