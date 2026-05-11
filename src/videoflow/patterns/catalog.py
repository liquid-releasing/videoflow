"""Pattern catalog data — mirrors `patterns-data.js` in forge-ui-design.

Single Python source of truth for the seven haptic patterns plus the
`_none` absence sentinel. React UIs read the same data via
`videoflow patterns-list` (JSON to stdout) so the catalog can never
drift between the backend and the studios.

When evolving the catalog, also update:
- `forge-ui-design/iterations/09-.../patterns-data.js` (or migrate the
  React side to fetch from this CLI and delete the JS literal)
- `videoflow/SIDECAR_GAP_VS_ITER09.md` if pattern_id semantics change
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternParam:
    """One parameter slot on a pattern. Drives UI sliders / dropdowns."""

    id: str
    label: str
    default: float | int | str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    enum_values: tuple[str, ...] | None = None
    hint: str | None = None

    def to_dict(self) -> dict:
        out: dict = {"id": self.id, "label": self.label}
        if self.default is not None:
            out["default"] = self.default
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.step is not None:
            out["step"] = self.step
        if self.unit:
            out["unit"] = self.unit
        if self.enum_values:
            out["enumValues"] = list(self.enum_values)
        if self.hint:
            out["hint"] = self.hint
        return out


@dataclass(frozen=True)
class Pattern:
    """A modulation pattern in the catalog. Consumed by haptics / stim /
    multi-axis / edit per the `consumers` field."""

    id: str
    label: str
    color: str
    category: str
    consumers: tuple[str, ...]
    summary: str
    description: str
    topology: tuple[str, ...]
    params: tuple[PatternParam, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "color": self.color,
            "category": self.category,
            "consumers": list(self.consumers),
            "summary": self.summary,
            "description": self.description,
            "topology": list(self.topology),
            "params": [p.to_dict() for p in self.params],
        }


@dataclass(frozen=True)
class PatternConsumer:
    """Consumer family — describes what a pattern_id means in each domain."""

    id: str
    label: str
    icon: str
    color: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "icon": self.icon,
            "color": self.color,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Catalog — seven haptic patterns
# ---------------------------------------------------------------------------

PULSE = Pattern(
    id="h_pulse",
    label="Pulse",
    color="#4cc3ff",
    category="haptic",
    consumers=("haptic", "edit", "stim", "multiaxis"),
    summary="Steady on/off oscillation. The metronome.",
    description=(
        "Hard-edged on/off cycle at a fixed period. Reads as rhythm. "
        "Pairs well with Climax tone and high-BPM phrases."
    ),
    topology=("full_body", "torso", "cushion", "wrists"),
    params=(
        PatternParam("period_ms", "Period", default=600, min=100, max=2000, step=50, unit="ms"),
        PatternParam("depth", "Depth", default=0.6, min=0, max=1, step=0.05),
        PatternParam("duty", "Duty", default=0.5, min=0.1, max=0.9, step=0.05),
    ),
)

WAVE = Pattern(
    id="h_wave",
    label="Wave",
    color="#3ed598",
    category="haptic",
    consumers=("haptic", "edit", "stim", "multiaxis"),
    summary="Smooth sine envelope across active regions.",
    description=(
        "Continuous sine modulation. Reads as breathing. "
        "Pairs well with Build and Tease tones."
    ),
    topology=("full_body", "torso", "cushion"),
    params=(
        PatternParam("period_ms", "Period", default=1200, min=200, max=4000, step=100, unit="ms"),
        PatternParam("depth", "Depth", default=0.7, min=0, max=1, step=0.05),
    ),
)

ROLLING = Pattern(
    id="h_rolling",
    label="Rolling",
    color="#ffb547",
    category="haptic",
    consumers=("haptic", "edit", "stim", "multiaxis"),
    summary="Sensation chases around the body.",
    description=(
        "Phased activation — neighbouring regions fire in sequence so "
        "the sensation appears to travel."
    ),
    topology=("full_body", "torso"),
    params=(
        PatternParam("period_ms", "Cycle", default=1800, min=400, max=6000, step=100, unit="ms"),
        PatternParam("depth", "Depth", default=0.6, min=0, max=1, step=0.05),
        PatternParam(
            "direction",
            "Direction",
            default="clockwise",
            enum_values=("clockwise", "counter", "outward", "inward"),
        ),
    ),
)

TREMOR = Pattern(
    id="h_tremor",
    label="Tremor",
    color="#c77dff",
    category="haptic",
    consumers=("haptic", "edit", "stim", "multiaxis"),
    summary="Random flutter. No fixed period.",
    description=(
        "Seeded noise envelope per region. No two regions in phase. "
        "Reads as anticipation."
    ),
    topology=("full_body", "torso", "cushion", "wrists"),
    params=(
        PatternParam("rate", "Rate", default=0.5, min=0, max=1, step=0.01),
        PatternParam("depth", "Depth", default=0.4, min=0, max=1, step=0.01),
        PatternParam("seed", "Seed", default=42, min=0, max=999, step=1),
    ),
)

SUSTAIN = Pattern(
    id="h_sustain",
    label="Sustain",
    color="#8b9bff",
    category="haptic",
    consumers=("haptic", "stim", "edit", "multiaxis"),
    summary="Hold a steady pressure. The drone.",
    description="Soft warm-up to a held intensity. No oscillation.",
    topology=("full_body", "torso", "cushion", "wrists"),
    params=(
        PatternParam("depth", "Depth", default=0.5, min=0, max=1, step=0.01),
        PatternParam("warmup_ms", "Warm-up", default=800, min=0, max=5000, step=100, unit="ms"),
    ),
)

IMPACT = Pattern(
    id="h_impact",
    label="Impact",
    color="#ff4b4b",
    category="haptic",
    consumers=("haptic", "edit", "stim", "multiaxis"),
    summary="Sharp punctuation. Rare, high-intensity hits.",
    description=(
        "Single-decay hits at long intervals. The accent layer — "
        "composes well over Sustain or Wave."
    ),
    topology=("full_body", "torso", "cushion", "wrists"),
    params=(
        PatternParam("every_ms", "Every", default=4000, min=500, max=30000, step=100, unit="ms"),
        PatternParam("depth", "Depth", default=0.9, min=0, max=1, step=0.01),
        PatternParam("decay_ms", "Decay", default=250, min=50, max=1000, step=10, unit="ms"),
    ),
)

REACTIVE = Pattern(
    id="h_reactive",
    label="Reactive",
    color="#56e0a0",
    category="haptic",
    consumers=("haptic", "edit", "stim", "multiaxis"),
    summary="Audio-following. No inherent shape — envelope = audio energy.",
    description=(
        "For ambient and score-driven chapters. Listens to the soundtrack "
        "via an envelope follower; bands + thresholds come from the Bias "
        "panel."
    ),
    topology=("full_body", "torso", "cushion", "wrists"),
    params=(
        PatternParam("gain", "Gain", default=1.0, min=0, max=2, step=0.05),
        PatternParam("floor", "Floor", default=0.0, min=0, max=1, step=0.05),
        PatternParam("attack_ms", "Attack", default=30, min=5, max=500, step=5, unit="ms"),
        PatternParam("release_ms", "Release", default=250, min=50, max=2000, step=10, unit="ms"),
        PatternParam(
            "downbeat_weight",
            "Weight downbeats",
            default=0,
            min=0,
            max=2,
            step=0.1,
            hint="0 = follow audio only · 1 = +50% on downbeats · 2 = double",
        ),
    ),
)

CATALOG: tuple[Pattern, ...] = (
    PULSE,
    WAVE,
    ROLLING,
    TREMOR,
    SUSTAIN,
    IMPACT,
    REACTIVE,
)


# ---------------------------------------------------------------------------
# Consumer families
# ---------------------------------------------------------------------------

CONSUMERS: tuple[PatternConsumer, ...] = (
    PatternConsumer(
        id="haptic",
        label="Haptics",
        icon="vibrate",
        color="#4cc3ff",
        summary="Body-region motor envelope",
    ),
    PatternConsumer(
        id="stim",
        label="Stim",
        icon="zap",
        color="#ffb547",
        summary="Carrier amplitude train",
    ),
    PatternConsumer(
        id="multiaxis",
        label="Multi-axis",
        icon="move-3d",
        color="#c77dff",
        summary="Axis displacement around rest",
    ),
    PatternConsumer(
        id="edit",
        label="Edit",
        icon="edit-3",
        color="#3ed598",
        summary="Funscript depth transform",
    ),
)


# ---------------------------------------------------------------------------
# Sentinel + lookup
# ---------------------------------------------------------------------------

NONE_ID: str = "_none"
"""Recipe-slot sentinel meaning 'no pattern on this axis'. See iter 09 §3."""

VALID_PATTERN_IDS: frozenset[str] = frozenset({p.id for p in CATALOG} | {NONE_ID})


def get(pattern_id: str) -> Pattern | None:
    """Return the Pattern with matching id, or None if not in the catalog."""
    for p in CATALOG:
        if p.id == pattern_id:
            return p
    return None


def is_valid(pattern_id: str) -> bool:
    """True if pattern_id is in the catalog or is the absence sentinel."""
    return pattern_id in VALID_PATTERN_IDS


def to_json(indent: int | None = None) -> str:
    """Serialise the full catalog (patterns + consumers) for CLI export."""
    return json.dumps(
        {
            "version": 1,
            "patterns": [p.to_dict() for p in CATALOG],
            "consumers": [c.to_dict() for c in CONSUMERS],
            "none_sentinel": NONE_ID,
        },
        indent=indent,
        ensure_ascii=False,
    )
