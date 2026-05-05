# Sidecar fragments — the reusable pattern primitive

> **Status:** **draft v0.1 — schema deliberately unlocked.** Sister doc to [audio-structure-primitive.md](audio-structure-primitive.md). Defines the **block-composition model** for `<name>.fragment.json` — a portable, named chunk of a chapter sidecar that can be saved, shared, and applied across media files and across the four lqr products. The block taxonomy will iterate freely until usage patterns stabilize; we have no installed user base, so backward-compat constraints don't apply yet.

> **Audience:** engineers working on any consumer app (forgegen, forgeplayer, forgeassembler, FunscriptForge), CLI tooling in videoflow, or the Pro pattern catalog. Lead with this doc when explaining what a "recipe", "pattern", "phrase library entry", or "template" actually is — they're all fragments composed of the same blocks at different combinations.

---

## Why fragments exist

Once `<stem>.chapters.json` is the public API across products, an obvious extension follows: the user wants to **bottle a great decision and apply it elsewhere**. Build a strong title sequence in forgeassembler? Save that chunk; apply it on the next project. Discover a phrase pattern that captures the DPL hypnotic style? Save it; apply it inside any chapter that wants that voice. Compose a four-pattern story (intro fragment + middle + climax + outro) in forgegen? That's just dropping fragments into chapter slots.

Fragments are the unification we've been circling. The cookbook recipes, the curated phrase libraries (Pro tier), the per-content-type defaults, the user's own saved templates — they're all the same artifact at different scopes:

| Today's concept | What it actually is |
|---|---|
| Cookbook recipe (PMV / Composed-CH / Hypnotic / Mixed) | A fragment composed of chapter-blocks + mode-recipe-blocks |
| DPL phrase library (Pro tier) | Fragments composed of phrase-blocks + beat-lock-blocks |
| forgegen recipe-per-chapter selection | User composing fragments into a sidecar |
| User-saved title sequence | A fragment composed of chapter-blocks + phrase-blocks + tag-blocks |
| FunscriptForge phrase swap | Applying a fragment containing phrase-blocks at a position |
| The catalog of intents | Library of fragments, indexed by tags |

One composition model, one library mechanism, one CLI verb set, four products consuming.

**Cookbook == library**, not doc. The cookbook ships *as* the built-in fragment library bundled with the products. There is no separate "cookbook document" anymore — the recipes are runnable artifacts.

---

## Design pivot: blocks compose; fragments are thin envelopes

Instead of one monolithic fragment schema, fragments are **compositions of typed blocks**. Each block has its own small schema; the fragment is a list of blocks that plug together.

Why this matters:

- **Independent evolution.** A `mode-recipe-block` can change without touching `chapter-block`. New block types can appear without versioning the world.
- **DRY by composition.** A great `mode-recipe-block` ("tease at 0.45 amplitude with smoothing") can be referenced by name from many fragments.
- **Forward compatibility for free.** Consumers ignore unknown block types. New blocks land without breaking old apps.
- **No premature lock.** We don't define a high-level schema we'd have to version. We define each block as it earns its place.

The fragment file is a thin envelope:

```json
{
  "schema": "fragment",
  "version": "0.1",
  "name": "PMV title sequence",
  "blocks": [
    { "type": "chapter-block", "rel_at_ms": 0, "rel_end_ms": 8000, "intent": "intro", "content_type": "music" },
    { "type": "phrase-block",  "in_chapter_idx": 0, "rel_at_ms": 1500, "rel_end_ms": 4500, "intent": "build", "mode_ref": "tease" },
    { "type": "mode-recipe-block", "name": "tease", "amplitude": 0.38, "smoothing": 0.6 },
    { "type": "bpm-constraint-block", "min": 90, "max": null },
    { "type": "duration-block", "mode": "scaled" },
    { "type": "tag-block", "tags": ["title", "intro", "PMV"] },
    { "type": "provenance-block", "extracted_from": "video.chapters.json", "author": "username", "created_at": "2026-05-05T14:00:00Z" }
  ]
}
```

The envelope carries `schema`, `version`, `name`, and `blocks`. Everything semantic is a block. This is closer to Notion's block model or composable design tokens than to a rigid IDL.

---

## Block taxonomy (current draft)

Block types fall into rough groups. **This list will grow and shift as we learn what authors actually save.** No schema lock until usage stabilizes.

### Structural blocks (where things live in time)

| Block type | Carries | Notes |
|---|---|---|
| `chapter-block` | `rel_at_ms`, `rel_end_ms`, `name?`, `intent?`, `content_type?` | One movement on the score |
| `phrase-block` | `in_chapter_idx`, `rel_at_ms`, `rel_end_ms`, `intent?`, `mode_ref?` | One measure within a movement |
| `transition-block` | `at_ms`, `kind` (snap / fade / crossfade), `duration_ms?` | Connector between two structural blocks |

### Behavioral blocks (how strokes are shaped)

| Block type | Carries | Notes |
|---|---|---|
| `mode-recipe-block` | `name`, `amplitude`, `density?`, `normalization?`, `smoothing?` | Reusable mode parameters; referenced by `mode_ref` |
| `beat-lock-block` | `subdivision`, `offset_ms?`, `strict?` | How strokes align to beats |
| `amplitude-curve-block` | array of `{rel_at_ms, value}` points | Explicit shape over a span |

### Constraint blocks (when the fragment is applicable)

| Block type | Carries | Notes |
|---|---|---|
| `bpm-constraint-block` | `min?`, `max?` | Warns / refuses if target audio is outside range |
| `duration-block` | `mode` (scaled / fixed / anchored), `target_ms?` | How relative timing maps onto target span |
| `content-type-constraint-block` | `allow` (list of content types) | "Apply only to chapters classified as music" |

### Metadata blocks

| Block type | Carries | Notes |
|---|---|---|
| `tag-block` | `tags` (string list) | Catalog indexing |
| `description-block` | `text` | Human-readable explanation |
| `provenance-block` | `extracted_from?`, `author?`, `created_at?`, `media_hint?` | Where the fragment came from |

### Plug rules (informal until they need to be formal)

- **Plurality**: most block types can appear multiple times in a fragment (`chapter-block`, `phrase-block`, `mode-recipe-block`). Some appear at most once (`duration-block`, `description-block`).
- **References**: a `phrase-block`'s `mode_ref` resolves to a `mode-recipe-block` with that `name` in the same fragment, or to a built-in mode if no local match.
- **Indexing**: `phrase-block.in_chapter_idx` is a 0-based index into the fragment's `chapter-block`s in order. Defaults to 0 if absent.
- **Time**: all `rel_at_ms` / `rel_end_ms` are measured from the fragment's own zero. `duration-block` controls how that maps to absolute time at apply.

### Adding a new block type

1. Pick a name. Document its schema as a markdown table in this doc under the right group.
2. Update consumers to handle it — or don't; consumers ignore unknown blocks.
3. No version bump required during the unlocked period.

---

## Authorship + application matrix

| App | Authors fragments at scope | Applies fragments at scope |
|---|---|---|
| **forgegen** | n/a (consumes only) | Compositions of chapter-blocks (recipe defaults) + user-composed sidecar at generation time |
| **forgeassembler** | Single chapter, multi-chapter sequences | chapter-block compositions to drive assembly + ordering |
| **FunscriptForge** | Phrase-only, phrase sequences, single-chapter, multi-chapter | All — phrase swaps, chapter overlays, full-story applies |
| **forgeplayer** | n/a | n/a (read-only consumer) |

forgeplayer is read-only across the board.

---

## Library / catalog

Fragments live as plain JSON files on disk. Three catalog tiers, all the same format, all discovered by the same resolver:

| Tier | Location | Source | Editable |
|---|---|---|---|
| **Built-in** | bundled with each app | The cookbook ships *as* this catalog | No |
| **User** | `~/.lqr/patterns/` (global) and `<project>/.lqr/patterns/` (project-local) | User-saved | Yes |
| **Pro** | `~/.lqr/patterns/pro/` | Curated catalog (DPL library, edger phrases) | Read-only; updated via package or remote pull |

Resolver order: project-local → user-global → pro → built-in. First match wins. Same priority pattern as the chapter sidecar resolver.

The library is a flat folder of `<name>.fragment.json` files. No subdirectories required (tags handle organization). No service backend.

---

## CLI surface (videoflow)

```
# Extract a fragment from an existing sidecar
videoflow chapters extract video.chapters.json --range 0:30 -o intro.fragment.json --name "PMV title sequence" --tags title,intro

# List available fragments matching tags
videoflow patterns list --tag title

# Apply a fragment into a sidecar at a position
videoflow chapters apply intro.fragment.json --into video.chapters.json --at 0

# Save a fragment into the user catalog
videoflow patterns add intro.fragment.json

# Validate a fragment against the (current) block taxonomy
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
  "from_fragment": "PMV title sequence@0.1"
}
```

Cheap to add, useful for two power-user operations:

1. **Bulk regeneration** — "regenerate all chapters that came from fragment X."
2. **Drift detection** — flag chapters where the user has edited away from the fragment's source pattern.

`from_fragment` is optional; absence means user-authored or auto-detected.

The provenance shape itself may be reworked once we see how it's used; treat it as a starting position, not a contract.

---

## Versioning posture (deliberately loose)

While the schema is unlocked:

- The envelope `version` field tracks the *block taxonomy snapshot*, not a stability promise. Bump it when block-shape changes are not just additive.
- **Consumers ignore unknown block types.** This is the only durable rule.
- No installed user base ⇒ no migration discipline yet. Break things freely; fix them in the same PR.

When usage patterns stabilize and a real user base exists:

- Lock the block taxonomy. Each block schema becomes its own contract.
- Switch to additive-only rules within a block major version.
- Document a migration policy.

The trigger to lock is **observed authoring stability** — when authors stop wanting new block types and start wanting old ones to behave reliably. Not a calendar date.

---

## Phase 4 gating

**Schema (current block taxonomy) + library plumbing must be in place to call Phase 4 complete.** That means:

- The block taxonomy in this doc is implemented as parsers + validators in `videoflow.structural`.
- The catalog resolver (built-in / user / Pro tiers) is implemented.
- CLI verbs (`chapters extract`, `chapters apply`, `patterns list / add / validate`) work.
- At least one consumer app authors and applies a fragment end-to-end (forgeassembler is the natural first — title sequence is the obvious first use case).

UI for **recipe-selection / story-builder** in forgegen (compose four patterns into a story arc) is **deferred to v2 or very late v1.** We need to learn which decisions users actually want to make at compose time before designing that UI.

---

## What's in v1, what's deferred

### v1 (ships with first fragment-aware product release)

- Block taxonomy as documented above (subject to iteration)
- `duration-block` mode `scaled` (the everyday case)
- Built-in + user catalog tiers
- CLI: `extract`, `apply`, `list`, `add`, `validate`
- Provenance tracking on applied chapters/phrases
- Cookbook ships *as* the built-in library

### v2+ (deferred)

- Story-builder UI in forgegen
- `duration-block` modes `fixed` and `anchored`
- Pro catalog (remote pull, package distribution)
- Fragment composition (fragments referencing sub-fragments)
- Conditional application (apply only if BPM matches, etc.)
- Drift detection UI

---

## Cross-references

- [audio-structure-primitive.md](audio-structure-primitive.md) — chapter sidecar primitive that fragments compose into.
- `videoflow/src/videoflow/chapters.py` — chapter resolver + sidecar IO.
- `videoflow/src/videoflow/structural.py` — auto-detection that produces the initial sidecar; will host the fragment block taxonomy.
- forgegen `architecture/funscript-quality-characteristics.md` — quality dimensions fragments target.
- forgegen `architecture/chapter-composition.md` — gold-standard reference patterns the cookbook fragments will encode.

---

## When this doc updates

This doc captures the block taxonomy and composition model. Update whenever:

- A new block type appears.
- A block's schema changes.
- The catalog resolver order changes.
- The CLI verb set changes.
- The Phase 4 gate criteria change.

Implementation status, ship dates, and library contents (which fragments exist, who uses them) live in each app's CHANGELOG and roadmap docs, not here.
