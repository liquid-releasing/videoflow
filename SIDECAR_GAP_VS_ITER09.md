---
title: videoflow sidecar gap analysis vs iter 09 spec
date: 2026-05-11
status: Identifies what videoflow's sidecar implementation needs to add for §0 (sidecars-are-the-database) compliance and iter 09 multi-axis support
---

# videoflow sidecar — gap vs iter 09 spec

videoflow is the Python analysis backbone for the lqr toolchain. Its sidecar (`videoflow.sidecar`) is the §0 database that downstream tools (forgegen, FunscriptForge Pro, ForgeStream) read.

This doc compares what videoflow's sidecar emits today against what iter 09's `implementation_handoff.md` and `architecture_feel.md` call for. **Headline: ~40% of the way there, with one critical §0 violation that blocks iter 09 multi-axis.**

---

## What videoflow has today

### Sidecar module

`videoflow/src/videoflow/sidecar.py` — schema v2.0.

**Top-level fields:** `schema`, `version`, `chapters`, `phrases`, `energy`, `provenance`, `auto_generated`.

### Chapters

Produced by `videoflow.structural.auto_chapter()`. Each `Chapter` carries: `at_ms`, `end_ms`, `name`, `intent`, `content_type`, `confidence`, `evidence`, `tone`, `shape`, `include`, `auto_generated`.

✅ Matches iter 09 chapter shape. Production-ready.

### Phrases

Produced by `videoflow.phrases.classify_phrases()`. Each `Phrase`: `chapter_idx`, `at_ms`, `end_ms`, `mode`, `confidence`, `source`, `intent`, `tone`, `evidence`, `auto_generated`.

Two source flavours: `source="audio"` (from energy) and `source="funscript"` (from stroke amplitude/density).

✅ Matches iter 09 phrase shape with bonus lineage tracking.

### Energy

`videoflow.audio.analyze_beats()` produces `AudioBeatMap` with `bpm`, `beats`, `downbeats`, `phrases`, `energy`, `duration_ms`. Energy gets exported to the sidecar via `auto_chapter()` stage `"energy"` (`structural.py:200-202`).

✅ Energy block lands in sidecar.  
✅ **Beats land in sidecar** as `energy.beat_map.times_ms` + `strengths` (correction to earlier draft of this doc — beats persistence already existed).  
✅ **Downbeats land in sidecar** as `energy.beat_map.is_downbeat` (parallel bool array — added 2026-05-11; was the one gap blocking iter 09 §3.5 beat-snap).

### Tone & shape

Both at chapter level (validated `_VALID_TONE_LABELS = {tender, build, tease, edge, climax, dominant}`, `_VALID_SHAPES = {flat, rise, fall, auto}`). Phrase-level `tone` override supported.

✅ Matches iter 09.

### Field-level merge with provenance

Implemented in `sidecar.py`:

- `mode="analyze"` vs `mode="edit"` write paths
- ANALYTICAL / AUTHORED / MIXED / STRUCTURAL / LATCH categorization (lines 40-52)
- Per-record `auto_generated` flag protects user-authored data (LATCH semantics)
- Append-only provenance log records writer + version + timestamp + fields touched (lines 542-553)

✅ Strong implementation. Matches iter 09 §0 + §6 contract.

---

## What iter 09 calls for

### From `implementation_handoff.md` §0

> *"a feature only exists if it lands in JSON."*

Implication: anything computed and dropped — even temporarily — violates §0 if downstream tools need it.

### From `implementation_handoff.md` §3 (Multi-axis recipes)

A separate sidecar `_funscript_multiaxis.json` keyed by phrase_id:

```json
{
  "version": 2,
  "phrase_recipes": {
    "P3": {
      "preset": "cowgirl" | "missionary" | "doggy" | "riding" | "random" | "none" | "custom",
      "axes": {
        "roll":  { "pattern_id": "h_wave",   "amount": 0.6 },
        "pitch": { "pattern_id": "h_pulse",  "amount": 0.4 },
        "twist": { "pattern_id": "_none",    "amount": 0.0 },
        "surge": { "pattern_id": "h_wave",   "amount": 0.3 },
        "sway":  { "pattern_id": "_none",    "amount": 0.0 }
      },
      "beat_snap": "bar" | "off" | "1" | "1/2" | "1/4"
    }
  }
}
```

### From `implementation_handoff.md` §3.5 (Beat-snap)

Phase-locker, NOT quantizer. Places emphasis on a specific grid point within the bar. Storage: `phrase_recipes[phrase_id].beat_snap`. **Input dependency: `beats[].downbeat: bool` from the analysis sidecar.**

### From `architecture_feel.md` §4.5 (`pattern_id` as universal field)

Long-term, every sidecar that today references `algorithm` or `carrier` or `character` should normalise to `pattern_id`. Migration target, not v1 scope, but establishes pattern_id as the convergence vocabulary.

### From `architecture_feel.md` §7 (.vibe envelope)

A `.vibe` zip-of-tracks that bundles all output formats and routes to devices. Assembly artifact — emitted at FFP/forgegen export time, not by videoflow analysis. **Out of scope for videoflow itself.**

---

## The gap — feature-by-feature

| Iter 09 requirement | videoflow status | Classification | Effort to close |
|---|---|---|---|
| Phrases with mode/tone/intent | Present in `phrases[]` | ✅ Has it | — |
| Chapters with content_type/confidence/evidence | Present in `chapters[]` | ✅ Has it | — |
| Field-level merge with auto_generated latch | Implemented in `sidecar.py` | ✅ Has it | — |
| Provenance audit log | Implemented in `sidecar.py` | ✅ Has it | — |
| Beats persisted to sidecar | `energy.beat_map.times_ms` + `strengths` | ✅ Has it | — |
| **Downbeats persisted to sidecar** | `energy.beat_map.is_downbeat` parallel bool array | ✅ **Done 2026-05-11** | — |
| `pattern_id` field (catalog vocabulary) | `videoflow.patterns.CATALOG` (7 patterns + `_none`); CLI: `videoflow patterns-list` | ✅ **Done 2026-05-11** | — |
| `phrase_recipes` block | Absent | ❌ Missing | ~4 h (schema + merge rules) |
| `beat_snap` per phrase | Absent (downbeats now available — unblocked) | ❌ Missing | trivial once recipes land |
| `_funscript_multiaxis.json` separate sidecar | Absent | ❌ Missing | ~2 h (schema doc + writer) |
| `.vibe` envelope (zip-of-tracks) | Absent — but **out of scope for videoflow** (export-time concern) | ✅ deferred | n/a here |
| Stim character as pattern_id | Absent (Stim sidecar not designed) | ❌ Missing | future scope |
| Edit transform `apply_pattern` with params | Absent (`_funscript_edit.json` not built) | ❌ Missing | belongs in FFP, not videoflow |

---

## The §0 violation — closed 2026-05-11

`AudioBeatMap.downbeats` (`videoflow/src/videoflow/audio.py:50`) was computed during analysis but never flowed into the sidecar — a textbook §0 violation (*the feature can't land in JSON because the data isn't in JSON*).

**Fix landed:** `videoflow.structural._build_energy()` now emits `is_downbeat` as a parallel bool array alongside `energy.beat_map.times_ms`:

```json
"energy": {
  "beat_map": {
    "times_ms":    [0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
    "strengths":   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "is_downbeat": [true, false, false, false, true, false, false, false]
  }
}
```

Tests: `test_structural.py::TestBuildEnergy::test_beat_map_marks_downbeats` (added). All 112 structural+sidecar+beats tests pass.

This unblocks iter 09 §3.5 beat-snap downstream — phrase recipes can now reference `is_downbeat[]` from the sidecar without re-analysing audio.

---

## Recommended next moves

### ✅ Tier 1 — Done (2026-05-11)

- **Downbeats persisted** as `energy.beat_map.is_downbeat` parallel bool array
- **Pattern catalog v1** built in `videoflow.patterns` (data + `videoflow patterns-list` JSON CLI export)

### Tier 2 — Recipe sidecar shape (next)

**Extend `sidecar.py` to support a `phrase_recipes` block.** Add as ANALYTICAL field (recomputed per-phrase, user overrides via auto_generated latch). ~4 hours.

**Define `_funscript_multiaxis.json` schema.** Create `videoflow/docs/architecture/multiaxis-recipes-sidecar.md` mirroring the `audio-structure-primitive.md` precedent. ~2 hours.

videoflow doesn't *generate* recipes (that's FFP/forgegen), but the sidecar shape needs to be defined here so the round-trip works. With pattern_id now in `videoflow.patterns.VALID_PATTERN_IDS`, recipe validation has a vocabulary to check against.

### Tier 3 — "Emphasize beats" wiring (when forgegen Generate lands)

User decision 2026-05-11: **"Emphasize beats" lives in the Generate tab as a per-chapter toggle** (next to Style/Density/Shape).

When forgegen Generate lands:
- Add `emphasize_beats: bool` to per-chapter generation parameters in the sidecar
- `videoflow.generate` reads this flag during funscript synthesis and applies stronger accent at downbeat moments (using `is_downbeat[]` from the energy block)
- Field category: AUTHORED (user-set, preserved by analyze writers)

### Tier 4 — Polish

**Benchmark downbeat detection accuracy across diverse content.** Run librosa downbeat against the 25-file harness, measure false-positive rate vs ground truth. ~4 hours.

Beat-snap is an emphasis phase-locker — accuracy of the downbeat anchor matters for user experience.

---

## Critical-path summary

| Step | Status |
|---|---|
| 1. Persist beats + downbeats | ✅ Done 2026-05-11 |
| 2. Lock pattern_id catalog | ✅ Done 2026-05-11 (`videoflow.patterns`) |
| 3. Add `phrase_recipes` schema support | Pending — ~4h, blocks multi-axis recipes round-trip |
| 4. `_funscript_multiaxis.json` schema doc | Pending — ~2h, blocks FFP/forgegen recipe writers |
| **Critical-path remaining** | **~6 h** |

After this, FFP can begin authoring recipes; forgegen can pass them through to renderers; ForgeStream can phase-lock secondary axes against beats. Until then, the multi-axis branch of iter 09 is on hold.

---

## Out of scope for videoflow

- `.vibe` envelope assembly — export-time concern, belongs in FFP/forgegen
- `_funscript_edit.json` — FFP-specific, transforms aren't applied in videoflow
- Stim sidecar schema — depends on iter 09 phases 2-4 which haven't started
- Pattern catalog *implementation* (the `apply_pattern` Python class) — also FFP-specific per `implementation_handoff.md` §4

---

## Cross-references

- `forge-ui-design/ARCHITECTURE_ADDENDUM_2026_05.md` — stack-unification + master-clock context
- `forge-ui-design/iterations/09-.../docs/implementation_handoff.md` — §0, §3, §3.5, §4 (the spec)
- `forge-ui-design/iterations/09-.../docs/architecture_feel.md` — §4.5, §7
- `videoflow/docs/architecture/audio-structure-primitive.md` — current sidecar architecture
- `videoflow/docs/architecture/sidecar-fragment.md` — block-composition model
