# Sidecar fragments — the reusable pattern primitive

> **Status:** architectural decision record. Sister doc to [audio-structure-primitive.md](audio-structure-primitive.md). Defines the schema for `<name>.fragment.json` — a portable, named chunk of a chapter sidecar that can be saved, shared, and applied across media files and across the four lqr products.

> **Audience:** engineers working on any consumer app (forgegen, forgeplayer, forgeassembler, FunscriptForge), CLI tooling in videoflow, or the Pro pattern catalog. Lead with this doc when explaining what a "recipe", "pattern", "phrase library entry", or "template" actually is — they're all fragments at different scopes.

---

## Why fragments exist

Once `<stem>.chapters.json` is the public API across products, an obvious extension follows: the user wants to **bottle a great decision and apply it elsewhere**. Build a strong title sequence in forgeassembler? Save that chunk; apply it on the next project. Discover a phrase pattern that captures the DPL hypnotic style? Save it; apply it inside any chapter that wants that voice. Compose a four-pattern story (intro fragment + middle + climax + outro) in forgegen? That's just dropping fragments into chapter slots.

Fragments are the unification we've been circling. The cookbook recipes, the curated phrase libraries (Pro tier), the per-content-type defaults, the user's own saved templates — they're all the same artifact at different scopes:

| Today's concept | What it actually is |
|---|---|
| Cookbook recipe (PMV / Composed-CH / Hypnotic / Mixed) | Default `chapter-sequence` fragment per content type |
| DPL phrase library (Pro tier) | Library of `phrase` and `phrase-sequence` fragments |
| forgegen recipe-per-chapter selection | User composing fragments into a sidecar |
| User-saved title sequence | A user-authored `chapter-sequence` fragment |
| FunscriptForge phrase swap | Applying a `phrase` fragment at a position |
| The catalog of intents | Library of fragments, indexed by tags |

One schema, one library mechanism, one CLI verb set, four products consuming.

---

## Schema (`fragment.json` v1.0)

A fragment is a JSON document with the following shape. All fields not marked optional are required.

```json
{
  "schema": "fragment",
  "version": "1.0",
  "name": "PMV title sequence",
  "tags": ["title", "intro", "PMV"],
  "author": "username (optional)",
  "created_at": "2026-05-05T14:00:00Z",
  "description": "30-second beat-locked title intro (optional)",

  "scope": "chapter-sequence",
  "duration_mode": "scaled",

  "chapters": [
    {
      "rel_at_ms": 0,
      "rel_end_ms": 8000,
      "name": "intro",
      "intent": "intro",
      "content_type": "music",
      "mode_recipe": "tease",
      "include": true
    }
  ],

  "phrases": [
    {
      "rel_at_ms": 1500,
      "rel_end_ms": 4500,
      "intent": "build",
      "mode": "steady",
      "amplitude_hint": 0.6,
      "beat_lock": true
    }
  ],

  "parameters": {
    "total_duration_ms": "scaled",
    "beat_locked": true,
    "audio_required_bpm_min": 90,
    "audio_required_bpm_max": null
  },

  "provenance": {
    "extracted_from": "<stem>.chapters.json (optional)",
    "media_hint": "PMV / 4-on-the-floor (optional)"
  }
}
```

### Field reference

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | Always `"fragment"`. Distinguishes a fragment from a full sidecar. |
| `version` | yes | Schema version. v1.0 is additive-only; breaking changes need v2.0. |
| `name` | yes | Human-readable label. Unique within a catalog. |
| `tags` | yes | Free-form string list. The catalog browser indexes by tags. |
| `author` | optional | String. Catalog provenance. |
| `created_at` | optional | ISO-8601 timestamp. |
| `description` | optional | One-paragraph explanation of what the fragment achieves. |
| `scope` | yes | One of `single-chapter`, `chapter-sequence`, `phrase`, `phrase-sequence`. Determines what fields below are populated. |
| `duration_mode` | yes | One of `scaled`, `fixed`, `anchored` (see below). |
| `chapters` | scope-dependent | Present when scope is `single-chapter` or `chapter-sequence`. |
| `phrases` | scope-dependent | Present when scope is `phrase` or `phrase-sequence`. May also be present nested under chapter scopes when the fragment carries phrase detail. |
| `parameters` | yes | Constraints on the target media + behavior knobs at apply time. |
| `provenance` | optional | Where the fragment came from. Free-form metadata. |

### Scope levels

| Scope | Contains | Typical authoring app | Typical use |
|---|---|---|---|
| `single-chapter` | one chapter record + optional phrases | forgeassembler, FunscriptForge | "this one chapter is so good — save it" |
| `chapter-sequence` | ordered list of chapters | forgeassembler, forgegen story builder | Title sequences, transitions, full-story templates |
| `phrase` | one phrase record | FunscriptForge | DPL-style drone, climax phrase |
| `phrase-sequence` | ordered list of phrases (within one chapter span) | FunscriptForge | A reusable build pattern |

### Relative timing

All time fields use `rel_at_ms` / `rel_end_ms` — milliseconds **from the fragment's own start (zero)**, not absolute media time. The application layer maps relative time onto the target span using the fragment's `duration_mode`.

### Duration modes

| Mode | Semantics | Use |
|---|---|---|
| `scaled` | Stretch / squeeze the fragment proportionally to fit the target span. | Title sequences, story templates — pattern shape matters more than absolute timing. |
| `fixed` | Hold absolute timing. If target span is shorter than fragment, fail. If longer, fragment occupies the head, rest is empty. | Beat-locked phrases at a specific BPM, time-critical loops. |
| `anchored` | Anchor key points to features in the target audio (downbeat, silence, energy peak). | Beat-aligned climaxes; phrases that must hit a specific cue. |

`fixed` and `anchored` are v1.0 schema slots; full implementation can land later. `scaled` is the v1 mover.

### Parameters block

The `parameters` block declares constraints and apply-time behavior:

- `total_duration_ms` — `"scaled"` (use target span), or an integer (use this duration regardless of target).
- `beat_locked` — boolean; whether application should re-snap stroke timings to beats in the target audio.
- `audio_required_bpm_min` / `audio_required_bpm_max` — optional BPM gates. Application can warn (or refuse) if the target audio falls outside the range.

v1.0 keeps `parameters` deliberately small. Future versions add slots additively.

---

## Authorship + application matrix

| App | Authors fragments at scope | Applies fragments at scope |
|---|---|---|
| **forgegen** | n/a (consumes only) | `chapter-sequence` (recipe defaults) + user-composed sidecar at generation time |
| **forgeassembler** | `single-chapter`, `chapter-sequence` | `chapter-sequence` to drive assembly + chapter ordering |
| **FunscriptForge** | `phrase`, `phrase-sequence`, `single-chapter`, `chapter-sequence` | All scopes — phrase swaps, chapter overlays, full-story applies |
| **forgeplayer** | n/a | n/a (read-only consumer) |

forgeplayer is read-only across the board — no fragment authoring or application. It only renders the composed sidecar.

---

## Library / catalog

Fragments live as plain JSON files on disk. Three catalog tiers, all the same format, all discovered by the same resolver:

| Tier | Location | Source | Editable |
|---|---|---|---|
| **Built-in** | bundled with each app | Cookbook recipes ship as fragments | No (read-only) |
| **User** | `~/.lqr/patterns/` (global) and `<project>/.lqr/patterns/` (project-local) | User-saved | Yes |
| **Pro** | `~/.lqr/patterns/pro/` | Curated catalog (DPL library, edger phrases) | Read-only; updated via package or remote pull |

Resolver order: project-local → user-global → pro → built-in. First match wins. Same priority pattern as the chapter sidecar resolver.

The library is a flat folder of `<name>.fragment.json` files. No subdirectories required (tags handle organization). No service backend in v1.

---

## CLI surface (videoflow `chapters` and `patterns`)

```
# Extract a fragment from an existing sidecar
videoflow chapters extract video.chapters.json --range 0:30 -o intro.fragment.json --name "PMV title sequence" --tags title,intro

# List available fragments matching tags
videoflow patterns list --tag title

# Apply a fragment into a sidecar at a position
videoflow chapters apply intro.fragment.json --into video.chapters.json --at 0 --duration scaled

# Save a fragment into the user catalog
videoflow patterns add intro.fragment.json

# Validate a fragment against the schema
videoflow patterns validate intro.fragment.json
```

The CLI is the automation seam. UIs in the four apps wrap the same operations.

---

## Provenance tracking

When a fragment is applied to a sidecar, each chapter / phrase the fragment produced records its origin:

```json
{
  "at_ms": 0,
  "end_ms": 8000,
  "intent": "intro",
  "from_fragment": "PMV title sequence@1.0"
}
```

This unlocks two operations that matter for power users:

1. **Bulk regeneration** — "regenerate all chapters that came from fragment X." Useful when the fragment improves and you want to propagate.
2. **Drift detection** — flag chapters where the user has edited away from the fragment's source pattern. Surfaces "this chapter has diverged from its template."

`from_fragment` is optional in the chapter sidecar schema — its absence means user-authored or auto-detected, no fragment origin.

---

## Versioning discipline

- **v1.0 is additive-only.** New optional fields can land in v1.0.x without breaking consumers. Required-field additions or rename / removal forces v2.0.
- **Consumers must ignore unknown fields.** Forward compatibility within a major version.
- **Breaking changes** require a written migration note + a one-release transition window where both v1 and v2 are accepted.

The schema version (`version` field in the JSON) is the contract. Codebase changes outside the schema (e.g., new mode names) are not schema versions — those are mode-vocabulary changes, separately governed.

---

## What's in v1, what's deferred

### v1 (ships with first fragment-aware product release)

- Schema lock at v1.0 (this doc)
- `scope`: `chapter-sequence` and `phrase` (most-used)
- `duration_mode`: `scaled` only
- Built-in + user catalog tiers
- CLI: `extract`, `apply`, `list`, `add`, `validate`
- Provenance tracking (`from_fragment` on applied chapters/phrases)

### v2+ (deferred but schema slots reserved)

- `single-chapter` and `phrase-sequence` scopes (schema accepts; full UI later)
- `duration_mode`: `fixed` and `anchored`
- Pro catalog (remote pull, package distribution mechanics)
- Composition (fragments referencing sub-fragments)
- Conditional application logic (apply this fragment only if BPM matches)
- Drift detection UI

The v1.0 schema accepts all of the above as additive — v2 won't break v1 consumers.

---

## Cross-references

- [audio-structure-primitive.md](audio-structure-primitive.md) — chapter sidecar primitive that fragments compose into.
- `videoflow/src/videoflow/chapters.py` — chapter resolver + sidecar IO.
- `videoflow/src/videoflow/structural.py` — auto-detection that produces the initial sidecar.
- forgegen `architecture/funscript-quality-characteristics.md` — quality dimensions fragments target.
- forgegen `architecture/chapter-composition.md` — gold-standard reference patterns the cookbook fragments encode.

---

## When this doc updates

This doc captures the schema contract. Update only when:

- The fragment schema changes (new fields, new scopes, new duration modes).
- The catalog resolver order changes.
- The CLI verb set changes.
- A new consumer app joins the matrix.

Implementation status, ship dates, and library contents (which fragments exist, who uses them) live in each app's CHANGELOG and roadmap docs, not here.
