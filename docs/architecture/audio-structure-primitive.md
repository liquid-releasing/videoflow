# Audio structure as a first-class primitive

> **Status:** architectural decision record. Implementation lives at `videoflow.structural` (planned for v0.5+; see [LONG_FORM_SCALING.md](https://github.com/liquid-releasing/forgegen/blob/main/architecture/LONG_FORM_SCALING.md) in forgegen for the implementation roadmap and benchmark data).

> **Audience:** engineers and operators across the lqr family touching long-form audio + funscript pipelines. Lead with this doc when explaining why videoflow contains structural-analysis code that doesn't directly produce a funscript.

---

## The core insight

> **A tool that reads the structure of the audio and generates against that structure — instead of treating audio as one undifferentiated stream.**

This is the differentiator vs. PythonDancer and every other audio→funscript generator we've evaluated. Whole-file analysis flattens authorial intent: a 90-minute scene that opens with ambient pacing and ends with a music-driven climax should *feel different* in those two sections. Whole-file normalization, whole-file mode classification, and whole-file rendering treat that contrast as noise to be averaged out. Structure-aware analysis preserves it.

This document records why `videoflow.structural` exists as a primitive, what it does and does not own, and how multiple lqr applications consume it.

---

## Problem: whole-file analysis loses content shape

forgegen v0.1's audio path runs a single `analyze_beats()` call against the entire media file, then a single `classify_modes()` pass over the whole beat map, then a single `beats_to_curve()` normalization. Every stage operates on the *file* as the unit.

Empirical result, measured across a 25-file benchmark spanning music, ambient, hentai, hypnotic, and cock-hero compilations from 5 minutes to 8 hours:

| Symptom | Root cause |
|---|---|
| Ambient/hentai content produces flat funscripts (75 % of strokes within 47–53 of center 50) | Whole-file `classify_modes()` puts 90%+ of beats into low-amplitude modes (`break: 0.12×`, `tease: 0.38×`); `shape_curve()` then crushes amplitude across the file |
| Music-driven content clusters into the 40–60 mid-band (28% of strokes) | Whole-file `energy_normalize=True` (95th-percentile reference) is dominated by the loudest 5% of beats; the rest are crushed into the middle |
| Long files (90 min+) provide one opaque progress wait | Single librosa pass; no checkpoints to surface to the UI |

These three symptoms have one shared cause: the analysis stages run at the wrong scope. Each was designed to be content-aware within its scope, but the scope is the entire file rather than each chapter's local context.

`librosa` itself does not crash on long files; we measured a clean 8-hour run at 16.8 GB peak RSS. The cliff is *quality*, not stability.

---

## Audio structure ≠ funscript structure

Two different kinds of structural analysis live in the lqr family. They are complementary, not redundant.

| | Lives in | Signal | Answers |
|---|---|---|---|
| **Audio structure** | `videoflow.structural` (this doc) | silence, recurrence, energy transitions, scene cuts | *"Where does the **content** change?"* |
| **Funscript structure** | FunscriptForge | velocity, peak/trough patterns, stroke clustering, phrase analysis | *"Where does the **script** change?"* |

On well-scripted content these usually align — a scene cut in the audio is also a phrase boundary in the funscript. But they will diverge in interesting ways:

- A funscript that intentionally lags or leads the audio beat for tension.
- A creator who pulls back during a loud climax.
- An ambient section with subtle audio-energy shifts that don't show in stroke positions.

These divergences are exactly the moments a creator wants to *see and decide about*. Fusing the two analyses into one signal would erase them. We build two primitives, expose both layers to consumers, and let the user reason about both at once.

This is why `videoflow.structural` belongs in videoflow, not in FunscriptForge: it is pure audio analysis with no funscript dependency. FunscriptForge's existing phrase analysis remains untouched, and gains audio-structure as a *companion* signal.

---

## Two roles, one primitive — the "chapter" overload

The word "chapter" carries two distinct meanings in the lqr toolchain, and
they share boundaries but serve different ends. Calling out the
distinction explicitly so the doc doesn't conflate them:

| Role | Use case | Who cares about it | What the data needs |
|---|---|---|---|
| **Generation segmentation** | Bitesize chunks for per-segment analysis. Each chunk classifies modes, normalizes energy, and (later) biases by content type independently. The "theme" the user gets emerges from the audio's actual progression instead of a single whole-file averaging. | forgegen | `at_ms`, `end_ms`, `content_type` heuristic, `confidence`, `evidence` |
| **Navigation waypoints** | Named, navigable sections. Find favorite parts. Cut points for assembly. Ruler-track overlay against funscript phrases. | forgeplayer, forgeassembler, FunscriptForge | `at_ms`, `end_ms`, `name`, `intent`, `include` flag |

**The same `<stem>.chapters.json` carries both.** The boundaries are
shared; the fields are additive. A chapter record produced by
`auto_chapter()` has generation-flavored fields populated; downstream
editors enrich it with navigation-flavored fields (names, intents,
exclusion flags) over time. Forgegen ignores fields it doesn't care
about; navigation consumers ignore the analysis fields.

This is why we resist the pull to create two parallel data structures
("analysis_segments.json" + "chapters.json"). They would drift over
time, force consumers to merge, and lose the round-trip property where
a user's edit in FunscriptForge informs forgegen's next regeneration.
One primitive, additive fields, role-specific consumption.

---

## The primitive: `videoflow.structural`

### `auto_chapter(media, target_minutes=5.5) -> list[Chapter]`

Synthesize a chapter list from the media's audio (or video, when present and useful). Return `Chapter` objects compatible with the existing `videoflow.chapters` resolver — the same downstream code consumes either source.

**Audio path** (the v1 driver):

Four stages, each addressing a specific failure mode of the previous:

1. **Silence detection.** Find windows of low audio energy ≥ N seconds long — natural pauses between sections, songs, scenes, or dialogue and action transitions. These become *candidate boundary anchors*. Real content tends to pause naturally between content units, and humans expect chapter breaks to land in those pauses.

2. **MFCC agglomerative clustering.** Mel-frequency cepstral coefficients capture the *spectral character* of audio — music, dialogue, foley, and silence sit in clearly separable MFCC regions. Compute MFCC vectors over short windows (`librosa.feature.mfcc(y, sr, n_mfcc=13)`), then run `librosa.segment.agglomerative` over the MFCC matrix. Raw chapter boundaries fall at the frame indices where consecutive MFCC frames belong to different clusters — i.e., where the audio's *timbre* changes meaningfully.

3. **Silence-snap.** Push each raw cluster boundary to the nearest silence anchor from step 1. Without this step, boundaries can land mid-word, mid-beat, or mid-phrase. Silence-snap aligns chapter starts and ends with the natural pauses a human listener already perceives as breaks.

4. **Micro-chapter merge.** If two consecutive chapters are closer than ~N seconds, merge them. Cleans up cluster jitter where MFCC briefly switches and switches back (e.g., a brief sound effect in a dialogue scene, a pickup beat at the end of a song). Also enforces the `target_minutes` floor — the algorithm coalesces adjacent similar segments until each chapter reaches the target duration. Never splits a recurring segment in half.

**Video path** (deferred to v2; not blocking forgegen):
- `videoflow.analysis.detect_scenes` (PySceneDetect) emits frame-accurate scene cuts.
- Coalesce adjacent scenes until target duration reached.
- Useful for forgeplayer chapter-nav and forgeassembler suggested-cut consumption.

**Output:** ordered list of `Chapter` records with start/end timestamps, a content-type heuristic (music / ambient / mixed), and metadata for downstream stages to bias on.

### Sidecar caching

`<stem>.chapters.json` next to the source file. Lets every consumer (forgegen, forgeplayer, forgeassembler, FunscriptForge) reuse the same chapter analysis without re-running detection. Fits the existing chapter-resolver priority order: embedded MP4 chapters → sidecar → auto-chapter. The file is also the **structural database** for the media — see [The sidecar schema](#the-sidecar-schema) below for the full shape (chapters + phrases + energy + tone + provenance) and the field-level merge semantics that let multiple writers cooperate.

### Chapter-aware analysis (downstream)

`analyze_beats()` and `beats_to_curve()` accept an optional `chapters` argument. When provided:
- The audio is analyzed one chunk at a time, holding only one chunk's tempogram in memory.
- `classify_modes()` runs per chunk, so a quiet intro classifies as `break` and a climactic chunk classifies as `edging` — both legitimately, within their own context.
- Per-chunk normalization (whichever mode is configured) operates on the chunk's local energy distribution, not the whole-file 95th percentile.

Stitching: per-chunk `AudioBeatMap`s merge into a single timeline. v1 strategy is concatenate-with-overlap-drop. Crossfade-at-seam is a polish item gated on observed seam quality.

### `progress_callback` extension

The existing `progress_callback` hook (added in videoflow 0.0.4 for staged feedback) extends to emit per-chapter events. This is the substrate for chapter-aware UX in consumer apps.

---

## The sidecar schema

`<stem>.chapters.json` is the **structural database** for one media file: the persistence layer for chapters, phrases, energy analysis, per-chapter tone, and any other carry-forward analysis the toolchain wants to share. The file is bidirectional public API. Multiple writers cooperate through field-level merge semantics rather than a lock.

> **Schema version: 2.x.** Sidecars produced by videoflow ≤ 0.0.4 carry only top-level `chapters` and read as v2 with empty `phrases` / `energy` / `provenance`. Consumers ignore unknown fields; new analysis dimensions ship as additive fields, not refactors. Same posture [sidecar-fragment.md](sidecar-fragment.md) takes for the reusable-fragment artifact.

### Annotated example

```jsonc
{
  "schema": "audio-structure",         // identifies this file shape
  "version": "2.0",                    // schema version (2.x = additive)
  "auto_generated": true,              // top-level: file untouched by user
  "generated_by": {
    "tool": "videoflow.structural",
    "tool_version": "0.0.5",
    "generated_at": "2026-05-07T18:42:11Z",
    "target_minutes": 5.5
  },

  "chapters": [
    {
      "at_ms": 0,
      "end_ms": 412000,
      "name": "intro",                 // AUTHORED — user/editor-supplied label
      "intent": "intro",               // AUTHORED — chapter-intent vocabulary
      "content_type": "ambient",       // ANALYTICAL — from auto-detection
      "confidence": 0.81,              // ANALYTICAL — detection confidence
      "evidence": ["mfcc", "silence"], // ANALYTICAL — feature ids that fired
      "tone": "soft",                  // MIXED — content tonality / stim character
      "include": true,                 // AUTHORED — false = consumer skips
      "auto_generated": true           // per-record latch (see merge rules)
    },
    {
      "at_ms": 412000,
      "end_ms": 1180000,
      "name": "build",
      "intent": "build",
      "content_type": "music",
      "confidence": 0.94,
      "evidence": ["mfcc", "rms_variance"],
      "tone": "driving",
      "include": true,
      "auto_generated": false          // user has touched this — frozen on regen
    }
  ],

  "phrases": [
    {
      "chapter_idx": 1,                // 0-based index into chapters[]
      "at_ms": 412000,
      "end_ms": 488000,
      "intent": "build",               // AUTHORED — phrase-intent vocabulary
      "mode": "tease",                 // ANALYTICAL — from classify_modes
      "confidence": 0.72,
      "evidence": ["beat_density"],
      "tone": "soft",                  // optional override of chapter tone
      "auto_generated": true
    },
    {
      "chapter_idx": 1,
      "at_ms": 488000,
      "end_ms": 720000,
      "intent": "sustain",
      "mode": "steady",
      "confidence": 0.85,
      "evidence": ["beat_density", "energy_envelope"],
      "auto_generated": true
    }
  ],

  "energy": {                          // optional; absent = not analyzed
    "percentiles": {                   // whole-file summary (cheap, always inline)
      "p5": 0.04, "p25": 0.12, "p50": 0.31,
      "p75": 0.58, "p95": 0.86
    },
    "beat_map": {                      // optional; can be large, inline OK
      "times_ms":  [412, 873, 1334, 1795 /* ... */],
      "strengths": [0.31, 0.42, 0.38, 0.45 /* ... */]
    },
    "envelope": {                      // optional; sampled at low rate to keep size bounded
      "sample_rate_hz": 10,
      "times_ms": [0, 100, 200 /* ... */],
      "values":   [0.04, 0.05, 0.07 /* ... */]
    },
    "per_chapter": {                   // optional; chapter idx → per-chapter analysis
      "0": {
        "percentiles": { "p5": 0.02, "p25": 0.06, "p50": 0.11, "p75": 0.18, "p95": 0.31 },
        "bpm": null
      },
      "1": {
        "percentiles": { "p5": 0.18, "p25": 0.32, "p50": 0.51, "p75": 0.74, "p95": 0.91 },
        "bpm": 122.0
      }
    }
  },

  "provenance": [                      // append-only audit log, oldest first
    {
      "writer": "videoflow.structural",
      "writer_version": "0.0.5",
      "written_at": "2026-05-07T18:42:11Z",
      "fields": ["chapters", "phrases", "energy"]
    },
    {
      "writer": "FunscriptForge",
      "writer_version": "0.4.2",
      "written_at": "2026-05-07T19:03:55Z",
      "fields": ["chapters[1].name", "chapters[1].intent", "chapters[1].auto_generated"]
    }
  ]
}
```

When optional fields are absent, consumers that care about them treat the data as unanalyzed (forgegen detects `energy` is missing and runs analysis; FunscriptForge detects `phrases` is empty and renders chapters without phrase overlays). Absent ≠ broken.

### Per-chapter fields

| Field | Type | Category | Notes |
|---|---|---|---|
| `at_ms` | int | STRUCTURAL | required; chapter start in milliseconds |
| `end_ms` | int? | STRUCTURAL | start of next chapter or end of media; absent = "to next" |
| `name` | str | AUTHORED | human label; default `""` |
| `intent` | str | AUTHORED | chapter-intent vocabulary; default `""` |
| `content_type` | str | ANALYTICAL | `music` / `ambient` / `mixed` / `""` |
| `confidence` | float? | ANALYTICAL | detection confidence in `[0, 1]` |
| `evidence` | str[] | ANALYTICAL | feature ids the detector fired on |
| `tone` | str | MIXED | content tonality / stim character |
| `include` | bool | AUTHORED | `false` = consumer skips this chapter; default `true` |
| `auto_generated` | bool | LATCH | `false` = freeze ALL fields on regen (see merge rules) |

STRUCTURAL fields are immutable in spirit — moving a chapter boundary creates a new analytical context. Edits to `at_ms` / `end_ms` are still allowed (forgeassembler does this) but should set `auto_generated: false` to mark the record as authored.

### Per-phrase fields

| Field | Type | Category | Notes |
|---|---|---|---|
| `chapter_idx` | int | STRUCTURAL | 0-based index into `chapters[]` |
| `at_ms` | int | STRUCTURAL | required; phrase start in milliseconds |
| `end_ms` | int | STRUCTURAL | required; phrase end in milliseconds |
| `intent` | str | AUTHORED | phrase-intent vocabulary; default `""` |
| `mode` | str | ANALYTICAL | from `classify_modes`: `tease` / `steady` / `pump` / `edging` / `break` |
| `confidence` | float? | ANALYTICAL | mode-classification confidence in `[0, 1]` |
| `evidence` | str[] | ANALYTICAL | feature ids; default `[]` |
| `tone` | str | MIXED | optional override of chapter tone |
| `auto_generated` | bool | LATCH | `false` = freeze on regen |

Phrases are not required to be contiguous or to cover the whole chapter — they mark recognised intent units only.

### Energy block

| Field | Required | Notes |
|---|---|---|
| `percentiles` | yes (when block present) | whole-file `{p5, p25, p50, p75, p95}` of normalised RMS |
| `beat_map.times_ms` | optional | int millis; can run to ~10⁵ entries on long files |
| `beat_map.strengths` | optional | float per beat in `[0, 1]`; same length as `times_ms` |
| `envelope.sample_rate_hz` | optional | sampling rate of the envelope; 10 Hz is the default |
| `envelope.times_ms` | optional | sampled times; same length as `values` |
| `envelope.values` | optional | RMS envelope value at each sample |
| `per_chapter.<idx>` | optional | per-chapter `{percentiles, bpm?, content_type?, evidence?}` |

All energy fields are ANALYTICAL — only the writer that computed them should overwrite them. The `envelope` block can grow large for long media; externalising it to a sibling `<stem>.energy.json` is permitted, in which case `energy.envelope` carries `{ "external": true }`. v1 keeps everything inline; externalisation lands when bench data shows file-size pain.

### Provenance block

`provenance` is an append-only audit log. Each entry records one write:

```jsonc
{
  "writer": "videoflow.structural",        // module / app id
  "writer_version": "0.0.5",               // tool version
  "written_at": "2026-05-07T18:42:11Z",    // ISO-8601 UTC
  "fields": ["chapters[1].name", "phrases"] // dotted paths or block names
}
```

Writers append, never edit prior entries. Readers ignore unknown writer ids. Useful for debugging round-trip behaviour and for "regenerate everything I last touched" power-user operations.

### Field-level merge semantics

forgegen and FunscriptForge are both writers. When a writer wants to update an existing sidecar:

1. **Field categorization.** Each field is ANALYTICAL, AUTHORED, MIXED, STRUCTURAL, or LATCH (see tables above).
2. **Existing records keep AUTHORED fields verbatim.** A re-run overwrites only ANALYTICAL fields of records it recomputes. AUTHORED fields (`name`, `intent`, `include`) survive untouched.
3. **MIXED fields follow the per-record `auto_generated` latch.** When `auto_generated: true`, the writer overwrites; when `false`, freeze.
4. **STRUCTURAL fields are not overwritten on existing records.** A re-run that re-detects the same chapter at a slightly different boundary keeps the old boundary unless `auto_generated: true`. Major boundary changes produce new records alongside.
5. **LATCH protection.** When `auto_generated: false` on a record, ALL fields including ANALYTICAL freeze. The user has explicitly authored that record; the writer respects it as a unit.
6. **New records append.** If the writer detects a chapter or phrase that wasn't in the prior sidecar, append it.
7. **Missing detections don't delete user-authored records.** If a re-run no longer detects a chapter that has any AUTHORED fields populated or `auto_generated: false`, the record stays. ANALYTICAL fields can be cleared to defaults to indicate "no longer detected," but the record itself is not dropped. forgegen never deletes user data.
8. **Append a provenance entry.** Every write logs writer id, version, timestamp, and the fields touched.

The contract: a user who edits one chapter name in FunscriptForge can re-run forgegen on the source and lose nothing. Consumers cooperate through field categories; the file is never locked.

### Schema versioning posture

- **v1.x:** chapters-only sidecars produced by videoflow ≤ 0.0.4. Read as v2 with empty `phrases` / `energy` / `provenance`.
- **v2.x:** chapters + phrases + energy + tone + provenance. Additive growth (new fields, new evidence ids, new content-type values) does not bump the major version.
- **v3.x:** reserved for non-additive changes. None planned. Triggers would be a category-table renaming, removing a previously-required field, or changing field semantics.

Consumers MUST tolerate forward-additive fields they don't recognise. Round-trip preservation (read → write) of unknown fields is **recommended for editors** (FunscriptForge, forgeassembler) and **optional for read-only consumers** (forgeplayer).

### Formal schema

Draft 2020-12 JSON Schema lives at [`sidecar.schema.json`](sidecar.schema.json) next to this doc. The `videoflow.sidecar` write API (item #3 of the backend gap-fix workplan, in flight) validates against it. Hand-editors should not need to read the JSON Schema — the field tables above are normative.

---

## Consumer matrix

Every consumer imports the same `videoflow.structural` primitive. None of them is special.

| Consumer | Role | What it consumes | What it produces / writes back |
|---|---|---|---|
| **forgegen** | The "easy button": *just get me a funscript.* | Auto-runs `auto_chapter()` on the source if no `<stem>.chapters.json` exists, then drives `analyze_beats(chapters=...)` per-chapter. **No chapter editor UI** — chapters are a means to the funscript, not a thing the user touches. Per-chapter progress events are still rendered (engagement during the long load). | Writes the freshly-detected chapter list to `<stem>.chapters.json` so downstream tools have something to refine. Writes `<stem>.analysis.json` with `structural.chapter_proposals` provenance. The funscript itself is the user-facing artifact. |
| **forgeplayer** | Playback only. | Sidecar chapters via the existing resolver. | Read-only. Renders chapter-nav. Never edits. |
| **forgeassembler** | *Clip assembly editor.* The user is already cutting; chapter editing rides along. | Sidecar chapters as suggested cut points. | Sidecar I/O — split, merge, drag boundaries, **mark chapters excluded from the assembled output** (drop commercials, ambient sections, bad takes). Plus user-preference export: per-chapter file split, mux into output MP4, etc. |
| **FunscriptForge** | *Funscript editor.* The user is already editing a timeline; chapter editing rides along. | Sidecar chapters as a second ruler-track overlay alongside funscript phrases. `<stem>.analysis.json` for forgegen-time provenance. | Sidecar I/O — split, merge, drag boundaries, **mark chapters excluded from the funscript** (skip a slow intro or section the creator doesn't want scripted). Plus user-preference export: MP4 metadata, embed into funscript metadata block. **Cross-signal feature audio-only and funscript-only tools cannot deliver.** |

### Where chapter editing lives — and where it doesn't

**Auto-detection + sidecar write** are videoflow's responsibility (via the
primitive). Every consumer reads through the same resolver. forgegen, the
"easy button," writes the auto-detected list once so the rest of the
toolchain has something concrete to start from.

**Chapter editing — split / merge / drag / exclude — and durable export
into other containers — muxing into MP4 metadata, embedding into funscript
metadata blocks, splitting per-chapter files — live in FunscriptForge and
forgeassembler, never in forgegen.** Reasons:

1. **forgegen is the easy button.** Its single user promise is *give me
   a funscript with no fuss.* A chapter editor in the Generate panel
   would mean every user has to confront chapter UI before they get a
   funscript — that's the opposite of "easy button." If the auto-detected
   chapters are wrong, the recourse isn't *edit them in forgegen* — it's
   *open the file in FunscriptForge or forgeassembler to refine.* The
   sidecar makes that recourse durable: edits persist, re-running
   forgegen picks up the refined chapters.
2. **FunscriptForge and forgeassembler are already in editing mode.**
   Both apps have timeline UIs where chapter authoring belongs natively.
   Putting editing + export there matches where the user already is.
3. **User preferences for export are app-scoped.** A creator using
   FunscriptForge to polish a hand-edited script has different export
   preferences (embed in funscript? mux into MP4?) than a creator using
   forgeassembler to slice a 5-hour source into clips (split per-chapter?
   suggest cuts?). One global setting in forgegen would conflate these.

The split:
- **forgegen** — *easy button.* Auto-detect, save sidecar, generate. No
  chapter UI beyond progress feedback during the long load.
- **FunscriptForge** — *funscript editor.* Full chapter editing
  (split / merge / drag / exclude) plus user-preference export to MP4
  metadata or funscript metadata block.
- **forgeassembler** — *clip assembler.* Full chapter editing plus
  user-preference export — per-chapter file splits, muxing into output
  MP4, etc.

The `<stem>.chapters.json` sidecar carries the full chapter list always;
"excluded" is a per-chapter flag (`include: false` or similar) that each
consumer interprets in its own context. Reflecting exclusion in the data
model rather than deletion lets the user un-exclude later, lets a
different consumer make a different inclusion decision, and preserves the
audio-structure analysis intact for future re-use.

---

## Reusable patterns: sidecar fragments

A chunk of a chapter sidecar — a single chapter with its phrases, or a multi-chapter sequence, or a phrase-only pattern — can be bottled, named, and applied to other media. This is how cookbook recipes, the DPL phrase library, user-saved title sequences, and forgegen's recipe-per-chapter selection all unify into one primitive.

See [sidecar-fragment.md](sidecar-fragment.md) for the schema, CLI surface, library/catalog architecture, and authorship matrix across the four products.

---

## Cross-signal opportunities (future)

Once both audio and funscript structural analyses are stable on their own, layered combinations become possible.

### Seeded phrase detection
An audio-chapter boundary can SEED FunscriptForge's phrase detection — a warmer initial guess that often matches. FF's analysis still runs independently so it can confirm the seed or detect a divergence the audio missed.

### Divergence highlighting
Visual cue in FunscriptForge when audio-chapter boundary and funscript-phrase boundary disagree by more than a small threshold. Surface as "*this section drifts from the audio — intentional?*" with a one-click jump to the offset. Surfaces creator decisions that today are invisible.

### Chapter-intent biasing
Per-chapter content-type heuristic (music / ambient / mixed) feeds forward into `classify_modes()` to bias the per-chapter classification toward content-appropriate distributions. A chapter that audio-side classifies as "ambient" can prefer `tease` over `break`; a chapter classified as "music" can prefer `steady` over `slow`. Don't build until we have data showing the unbiased classifications drift in measurable ways.

### Content-adaptive normalization mode
Different content types respond best to different `energy_normalize` modes (measured: log-compression for music, percentile-based for ambient, fixed for hypnotic). Once chapter content type is detected, the normalization mode can be selected per-chapter rather than per-call.

These layers do *not* belong in v1. They are the design space the audio-structure primitive opens up.

---

## v1 scope (build on what we understand to be the best product)

Things that look like cost-saving deferrals but actually belong in v1:

- **User-editable chapter boundaries — but in FunscriptForge and forgeassembler, not forgegen.** Auto-detection will be wrong on some files. *Wrong chapters → bad output → no recourse* is the dead-end UX that makes a tool feel cheap. The recourse here isn't a chapter editor in forgegen (which would compromise the easy-button promise); it's the chapter editor that ships as part of v1 in FunscriptForge and forgeassembler. The primitive returns chapters that are easy to mutate; the sidecar makes those edits durable across tools and sessions; re-running forgegen picks up the refined chapters automatically.
- **Sidecar persistence is the default, not an optimization.** Auto-chapter results write to `<stem>.chapters.json` on first run. Subsequent consumers find chapters via the existing resolver order: embedded MP4 chapters → sidecar → re-run auto-chapter. User edits in FunscriptForge / forgeassembler go to the sidecar, never lost.
- **Per-chapter progress events** through the existing `progress_callback`. Long-form analysis without per-chapter visibility is the opaque-wait UX the structural primitive is supposed to fix; missing it would defeat the purpose. This is the *only* chapter-aware UX in forgegen — feedback during the long load, not editing.

## Decisions held open until data is in

These are genuine "measure first" decisions, not deferrals.

1. **Audio-only scene-detection algorithm parameters.** `librosa.segment.recurrence_matrix` is the obvious starting point. Similarity threshold tuning for music vs. spoken vs. ambient material is real work. Bench against the 25-file harness; pick parameters from data, not intuition.
2. **Stitching seam quality.** v1 ships concatenate-with-overlap-drop. Crossfade-at-seam is *not* polish if seams audibly degrade on the bench — it's required and ships in v1 too. Decision is gated on measurement on the bench, not on user reports.
3. **Video-path scene detection.** Deferred to v2 not because video doesn't matter (forgeplayer chapter-nav and forgeassembler cut suggestions both want it) but because a single audio path is enough to validate the primitive's design and unblock forgegen. v2 is "second consumer wave," not "next year."
4. **Cross-signal seeding (audio-chapter ↔ funscript-phrase).** Build only after both detectors are stable independently, so divergences can be observed cleanly. The cross-signal feature is a layered enhancement, not a shortcut around solving each side properly.
5. **Chapter-intent biasing.** Per-chapter content-type heuristic feeding `classify_modes()` is plausible but unproven. Don't build until benchmark data shows unbiased classifications drift in measurable ways.

---

## Cross-references

- [`sidecar.schema.json`](sidecar.schema.json) — formal JSON Schema (Draft 2020-12) for the expanded sidecar; consumed by the `videoflow.sidecar` write API.
- [`sidecar-fragment.md`](sidecar-fragment.md) — sister doc; reusable-fragment artifact composed of the same blocks at sub-media scope.
- `videoflow/src/videoflow/chapters.py` — existing chapter resolver. Docstring already reserves `videoflow.structural` for this work.
- `videoflow/src/videoflow/analysis.py` — existing PySceneDetect wrapper; the building block for the video path.
- [`forgegen/architecture/LONG_FORM_SCALING.md`](https://github.com/liquid-releasing/forgegen/blob/main/architecture/LONG_FORM_SCALING.md) — implementation roadmap and benchmark data; the immediate driver of this primitive.
- `forgegen/testcases/long_form_benchmark/` — 25-file diverse-content harness (music / ambient / hentai / hypnotic / cock-hero / 8h stress) for validating analysis output across content types.
- forgegen ROADMAP v0.5 entry — short summary + link here.

---

## When this doc updates

This doc captures architectural intent, not progress. Update only when:
- The audio-structure primitive's API changes substantively (new modes, new return types, new consumer relationships).
- The consumer matrix grows or shrinks.
- A cross-signal layer ships and changes how consumers compose the primitive.
- The sidecar schema grows or shrinks. Additive growth (new fields, new evidence ids, new content-type or mode values) is expected and does not require a major-version bump; non-additive changes do.
- The field-level merge semantics or category assignments change.

Implementation status, ship dates, and bug history live in each project's CHANGELOG and roadmap docs.
