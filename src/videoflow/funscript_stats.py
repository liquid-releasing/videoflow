"""Funscript quality metrics — the shared measurement core.

One module, three consumers:

1. **Generation benchmarking** — compare generated output against a gold
   reference (decile shape, density dynamics, velocity arc).
2. **A user diagnostic lens** — score a hand-made script and flag weak
   sections (the "raise the state of the art" feature: the editor gets
   opinionated, not just a viewer).
3. **Phrase / mode segmentation** — the windowed profile is the motion-
   character envelope that phrase boundaries and stanza modes are derived
   from (segment the envelope, then label each run).

The metrics here are the ones identified in
``forgegen/architecture/funscript-quality-characteristics.md`` (Dimensions
1–2) and proven discriminating in
``forgegen/architecture/GENERATION_DEPTH_LAW.md``:

- **Distribution shape** (decile histogram, rails% / mid%) — bimodal good,
  centered-bell bad.
- **Velocity** (units/sec) — the intensity envelope; its build over time is
  the narrative arc.
- **Density** (actions/sec) — what the gold modulates through a piece.
- **Dynamics index** — variance of the per-window profile; high = dynamic
  (great), low = monotone (mediocre).

Stdlib only — no numpy — so it drops cleanly into any consumer (videoflow
engine, forgegen bench, a FunscriptForge panel).
"""

from __future__ import annotations

from typing import Iterable


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def to_pairs(actions: Iterable) -> list[tuple[int, int]]:
    """Normalise funscript actions to a sorted list of ``(at_ms, pos)``.

    Accepts either dicts (``{"at": ms, "pos": 0..100}``) or ``(at, pos)``
    tuples. Sorts by time and clamps positions to ``[0, 100]``.
    """
    pairs: list[tuple[int, int]] = []
    for a in actions:
        if isinstance(a, dict):
            at, pos = int(a["at"]), int(a["pos"])
        else:
            at, pos = int(a[0]), int(a[1])
        pairs.append((at, max(0, min(100, pos))))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[idx]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Distribution shape — the decile oracle
# ---------------------------------------------------------------------------

def decile_histogram(pairs: list[tuple[int, int]]) -> list[int]:
    """Return ten counts: how many positions fall in each 0-9 / 10-19 / …
    / 90-100 band. The shape oracle — bimodal (rails-loaded) vs bell
    (mid-piled)."""
    bins = [0] * 10
    for _, pos in pairs:
        bins[min(9, pos // 10)] += 1
    return bins


def position_stats(pairs: list[tuple[int, int]]) -> dict:
    """Distribution-shape stats: decile histogram, rails% / mid% / limbo%,
    and position percentiles.

    - ``rails_pct``  = % at the extremes (deciles 0 + 9) — full strokes.
    - ``mid_pct``    = % in the centre (deciles 2–7) — partial strokes.
    - ``limbo_pct``  = % in deciles 1 + 8 (just-inside-rails).
    - ``bimodal`` is True when rails dominate the centre (rails > mid).
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    deciles = decile_histogram(pairs)
    pct = [c * 100 // n for c in deciles]
    positions = sorted(p for _, p in pairs)
    return {
        "n": n,
        "deciles": deciles,
        "deciles_pct": pct,
        "rails_pct": pct[0] + pct[9],
        "limbo_pct": pct[1] + pct[8],
        "mid_pct": sum(pct[2:8]),
        "p5": _percentile(positions, 0.05),
        "p25": _percentile(positions, 0.25),
        "p50": _percentile(positions, 0.50),
        "p75": _percentile(positions, 0.75),
        "p95": _percentile(positions, 0.95),
        "iqr": _percentile(positions, 0.75) - _percentile(positions, 0.25),
        "bimodal": (pct[0] + pct[9]) > sum(pct[2:8]),
    }


# ---------------------------------------------------------------------------
# Velocity + stroke + density
# ---------------------------------------------------------------------------

def velocities(pairs: list[tuple[int, int]]) -> list[tuple[int, float]]:
    """Per-adjacent-pair velocity in position-units/sec, tagged with the
    end timestamp. Velocity = density × stroke-size — the intensity signal
    whose envelope over time is the narrative arc."""
    out: list[tuple[int, float]] = []
    for i in range(1, len(pairs)):
        dt = pairs[i][0] - pairs[i - 1][0]
        if dt > 0:
            out.append((pairs[i][0], abs(pairs[i][1] - pairs[i - 1][1]) / dt * 1000.0))
    return out


def motion_stats(pairs: list[tuple[int, int]]) -> dict:
    """Velocity, stroke-amplitude, and density aggregates over the whole
    file (Dimension 2 in the quality doc)."""
    n = len(pairs)
    if n < 2:
        return {"n": n, "rate": 0.0, "avg_stroke": 0.0,
                "avg_velocity": 0.0, "p90_velocity": 0.0}
    span_s = (pairs[-1][0] - pairs[0][0]) / 1000.0 or 1.0
    strokes = [abs(pairs[i][1] - pairs[i - 1][1]) for i in range(1, n)]
    vels = sorted(v for _, v in velocities(pairs))
    return {
        "n": n,
        "rate": n / span_s,                       # actions per second
        "avg_stroke": _mean(strokes),             # avg |Δpos| per action
        "avg_velocity": _mean(vels),              # units/sec
        "p50_velocity": _percentile(vels, 0.50),
        "p90_velocity": _percentile(vels, 0.90),  # peak-intensity bursts
        "duration_ms": pairs[-1][0] - pairs[0][0],
    }


# ---------------------------------------------------------------------------
# Speed distribution — the movement-speed histogram (F.A.P.S-style)
# ---------------------------------------------------------------------------

#: Movement-speed bins in position-units/sec (|Δpos| per second between
#: adjacent actions — the same unit :func:`velocities` returns). Seven named
#: bands from "barely moving" to "flash", matching the speed-distribution
#: readout users already know from F.A.P.S / heatmap tools. Six edges → seven
#: bins. Tunable; the names travel with the values so any UI can label them.
SPEED_BINS: tuple[str, ...] = (
    "very_slow", "slow", "medium", "fast", "very_fast", "ultra_fast", "flash",
)
SPEED_EDGES: tuple[int, ...] = (50, 150, 250, 350, 450, 600)


def speed_distribution(pairs: list[tuple[int, int]]) -> dict:
    """Histogram of per-action movement speed into the :data:`SPEED_BINS`.

    The speed counterpart to :func:`position_stats`' decile histogram: where
    decile shape answers "how far do strokes reach", this answers "how fast do
    they move". A great script spreads across the bands; a monotone one piles
    into one. Returns counts + percentages per named band so a UI can render
    the familiar Very-Slow…Flash bar without re-deriving the bins."""
    vels = [v for _, v in velocities(pairs)]
    n = len(vels)
    counts = [0] * len(SPEED_BINS)
    for v in vels:
        b = 0
        while b < len(SPEED_EDGES) and v >= SPEED_EDGES[b]:
            b += 1
        counts[b] += 1
    pct = [round(c * 100.0 / n, 1) if n else 0.0 for c in counts]
    return {
        "bins": list(SPEED_BINS),
        "edges": list(SPEED_EDGES),
        "counts": counts,
        "pct": pct,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Windowed profile — the temporal envelope (modes / phrases / arc live here)
# ---------------------------------------------------------------------------

def windowed_profile(
    pairs: list[tuple[int, int]], *, n_windows: int = 12,
) -> list[dict]:
    """Split the file into ``n_windows`` equal-time windows and report each
    window's rate, rails%/mid%, avg stroke, mean position, and velocity.

    This is the motion-character envelope: its *shape over time* is the
    arc, its *variance* is the dynamics index, and its *change-points* are
    candidate phrase / mode boundaries.
    """
    if len(pairs) < 2 or n_windows < 1:
        return []
    t0, t1 = pairs[0][0], pairs[-1][0]
    width = max(1, (t1 - t0) // n_windows)
    vel = velocities(pairs)
    out: list[dict] = []
    for w in range(n_windows):
        ws, we = t0 + w * width, t0 + (w + 1) * width
        seg = [(t, p) for t, p in pairs if ws <= t < we]
        vseg = [v for t, v in vel if ws <= t < we]
        if len(seg) < 2:
            out.append({"window": w, "at_ms": ws, "n": len(seg),
                        "rate": 0.0, "rails_pct": 0, "mid_pct": 0,
                        "avg_stroke": 0.0, "mean_pos": 0.0,
                        "avg_velocity": 0.0})
            continue
        ps = position_stats(seg)
        span_s = (seg[-1][0] - seg[0][0]) / 1000.0 or 1.0
        strokes = [abs(seg[i][1] - seg[i - 1][1]) for i in range(1, len(seg))]
        out.append({
            "window": w,
            "at_ms": ws,
            "n": len(seg),
            "rate": len(seg) / span_s,
            "rails_pct": ps["rails_pct"],
            "mid_pct": ps["mid_pct"],
            "avg_stroke": _mean(strokes),
            "mean_pos": _mean([p for _, p in seg]),
            "avg_velocity": _mean(vseg),
        })
    return out


def _cov(xs: list[float]) -> float:
    """Coefficient of variation (std / mean). Scale-free spread."""
    m = _mean(xs)
    if m == 0:
        return 0.0
    var = _mean([(x - m) ** 2 for x in xs])
    return (var ** 0.5) / m


def dynamics_index(profile: list[dict]) -> dict:
    """How much the profile VARIES across windows — the great-vs-monotone
    discriminator. Great scripts swing density/velocity widely (high CoV);
    mediocre ones are monotone (low CoV)."""
    if not profile:
        return {"velocity_cov": 0.0, "rate_cov": 0.0}
    return {
        "velocity_cov": _cov([w["avg_velocity"] for w in profile]),
        "rate_cov": _cov([w["rate"] for w in profile]),
    }


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------

def summarize(actions: Iterable, *, n_windows: int = 12) -> dict:
    """One call → the full quality fingerprint of a funscript.

    Returns ``position`` (shape), ``motion`` (velocity/stroke/density),
    ``profile`` (per-window envelope), and ``dynamics`` (variance index).
    Used by the generation benchmark, the user diagnostic lens, and as the
    feature source for phrase/mode segmentation.
    """
    pairs = to_pairs(actions)
    profile = windowed_profile(pairs, n_windows=n_windows)
    return {
        "position": position_stats(pairs),
        "motion": motion_stats(pairs),
        "speed": speed_distribution(pairs),
        "profile": profile,
        "dynamics": dynamics_index(profile),
    }
