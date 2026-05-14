# API reference

Public Python API of videoflow v0.0.5.

Every public name is re-exported from the top-level `videoflow` package, so the imports below all resolve. The module-qualified imports (e.g. `videoflow.sidecar`) are equivalent and useful when you want to highlight which subsystem a name belongs to.

## Audio

```python
from videoflow.audio import (
    AudioBeatMap,
    BeatError,
    analyze_beats,
)
```

### `analyze_beats(input, *, sr=22050, source="full", tracker="auto", locked_bpm=None, chapters=None, on_progress=None, progress_callback=None) -> AudioBeatMap`

Beat / BPM / phrase / energy detection for one media file. Accepts audio or video; ffmpeg-extracts audio from video.

- `source`: `"full"` uses the mix as-is. `"percussive"` HPSS-separates the percussive component first so beats follow drums rather than vocals.
- `tracker`: `"auto"` (default) picks `"plp"` for tracks > 10 min and `"beat_track"` otherwise. `"beat_track"` is the classic global-tempo DP tracker; `"plp"` (predominant local pulse) handles long-form tempo drift.
- `locked_bpm`: pin the tracker to a specific tempo when auto-detection lands on a half/double octave.
- `chapters`: when supplied (typically from `auto_chapter`), analysis runs per chapter with chunk-relative energy normalisation. Beats / phrases / energy are concatenated in chapter order and the reported `bpm` is the duration-weighted mean of per-chapter BPMs.
- `on_progress`: structured progress callback receiving `StageEvent` objects (see [`videoflow.progress`](#progress)). Hierarchical stages — `audio.analyze → extract / load / hpss / chapters → chapter N/M / track`.

Returns `AudioBeatMap` with `bpm`, `beats: list[int]` (ms), `downbeats`, `phrases: list[(start_ms, end_ms)]` (16-beat groupings), `energy: list[float]` (per-beat normalised RMS), `duration_ms`. Raises `BeatError` on decode failure or no detected rhythm.

### `AudioBeatMap`

```python
beat_map.save("track_beat.json")            # serialise to JSON
loaded = AudioBeatMap.load("track_beat.json")
nearest = beat_map.beats_in_range(start_ms, end_ms)
```

## Structural — chapter detection

```python
from videoflow.structural import (
    AutoChapterError,
    auto_chapter,
)
```

### `auto_chapter(media, *, target_minutes=5.5, write_sidecar=True, on_progress=None, sr=22050) -> list[Chapter]`

The audio-structure primitive. Detects natural chapter boundaries from silence + MFCC clustering + silence-snap, runs `analyze_beats(chapters=…)` and `classify_phrases(beat_map, chapters=…)` inline, and merges the full payload — chapters + phrases + energy — into `<stem>.chapters.json` via [`videoflow.sidecar.write_sidecar`](#sidecar) in `analyze` mode. User edits are preserved per-record through the field-level merge contract.

Files shorter than ~8 min return a single whole-file chapter; longer files aim for `target_minutes` per chunk with boundaries snapped to natural pauses.

See [audio-structure-primitive.md](architecture/audio-structure-primitive.md) for the design rationale, consumer matrix, and merge semantics.

## Sidecar — read / write the structural database

```python
from videoflow.sidecar import (
    SidecarError,
    chapters_from_sidecar,
    read_sidecar,
    sidecar_path_for,
    write_sidecar,
)
```

`<stem>.chapters.json` is the persistence layer for chapters, phrases, energy analysis, per-chapter tone, and any other carry-forward analysis. Multiple writers (forgegen, FunscriptForge, forgeassembler) cooperate through field-level merge rather than a lock.

### `read_sidecar(media_path) -> dict | None`

Read and validate `<stem>.chapters.json` next to *media_path*. Returns the parsed JSON document with v1 sidecars normalised to v2 shape (empty `phrases` / `provenance`, no `energy`). Returns `None` when no sidecar exists. Unknown fields round-trip verbatim. Raises `SidecarError` on malformed input.

### `write_sidecar(media_path, payload, *, writer, writer_version="", mode="analyze") -> Path`

Write *payload* to `<stem>.chapters.json` with field-level merge against any existing file.

- `mode="analyze"` (default, for recompute writers like `auto_chapter`): on matched records with the per-record `auto_generated` latch open, ANALYTICAL and MIXED fields are overwritten; AUTHORED and STRUCTURAL fields are preserved verbatim. New records append.
- `mode="edit"` (for user-edit writers like FunscriptForge): all categories except STRUCTURAL flow through; per-record `auto_generated` flips to `false` to mark the user's authorship.

LATCH (`auto_generated=false`) freezes the entire record in both modes. Each write appends a `provenance` entry recording writer + version + fields touched.

### `chapters_from_sidecar(doc) -> list[Chapter]`

Convenience: parse a sidecar dict into typed `Chapter` records. Unknown chapter fields stay in the dict; they are not surfaced on `Chapter`.

### `sidecar_path_for(media_path) -> Path`

Return the canonical sidecar path next to *media_path* (e.g. `track.mp4` → `track.chapters.json`).

## Phrases — mode classification

```python
from videoflow.phrases import (
    Phrase,
    classify_phrases,
    classify_phrases_from_funscript,
)
from videoflow.generate import classify_modes  # legacy tuple API
```

A phrase is a within-chapter intent unit. Two classifiers, one record:

### `classify_phrases(beat_map, *, chapters=None) -> list[Phrase]`

Audio-driven classification. Each phrase gets one of `tease` / `steady` / `edging` / `break` / `fast` / `slow` based on phrase energy and tempo. Chapter-aware mode (chunk-relative thresholds) when `chapters` is supplied — fixes the "ambient flat output" problem of whole-file thresholds. Records carry `source="audio"`.

### `classify_phrases_from_funscript(actions, *, chapters=None, phrase_boundaries=None) -> list[Phrase]`

Parallel classifier that uses stroke amplitude and stroke density instead of audio energy. Same six modes. Useful when a user has hand-edited the funscript and wants phrase classifications that match the edited curve rather than the source audio. Records carry `source="funscript"`. When `phrase_boundaries` is `None`, actions are grouped into 16-action phrases (mirrors `AudioBeatMap`'s 16-beat grouping).

### `Phrase`

```python
Phrase(
    chapter_idx=0, at_ms=0, end_ms=4_000,
    mode="tease",
    confidence=None,
    source="audio",         # "audio" | "funscript" | "user" | ""
    intent="",              # phrase-intent vocabulary, AUTHORED
    tone="",                # FunscriptForge tone override, MIXED
    evidence=[],
    auto_generated=True,    # LATCH: false freezes the record on regen
)
```

Round-trips through the sidecar via `Phrase.to_dict()` / `Phrase.from_dict()`.

### `classify_modes(beat_map, *, chapters=None) -> list[(start_ms, end_ms, mode)]`

Legacy tuple-shaped API preserved at `videoflow.generate.classify_modes` and `videoflow.phrases.classify_modes`. New code should prefer `classify_phrases` for the rich `Phrase` records.

## Generate — funscript synthesis

```python
from videoflow.generate import (
    GenerateError,
    beats_to_curve,
    compute_auto_tone,
    export_funscript,
    generate_from_beats,
    shape_curve,
)
```

### `generate_from_beats(beat_map, output, *, low=10, high=90, center=None, center_trajectory=None, tone_per_phrase=None, energy_normalize=False, stroke_density="half", title="") -> Path`

End-to-end convenience: beat map → shaped funscript file. Write a `.funscript` JSON to *output*.

### Lower-level building blocks

```python
curve = beats_to_curve(beat_map, low=10, high=90)
phrases = classify_phrases(beat_map, chapters=chapters)
shaped = shape_curve(curve, [(p.at_ms, p.end_ms, p.mode) for p in phrases])
funscript = export_funscript(shaped, beat_map.duration_ms)
```

`shape_curve` applies per-mode easing, velocity profiles, and semantic shaping (tease oscillations, edging ramps, break minimisation). `compute_auto_tone` derives the per-phrase center trajectory from each phrase's energy slope — pass directly to `shape_curve(tone_per_phrase=…)`.

## Progress — staged events with ETA

```python
from videoflow.progress import (
    ETAEstimator,
    OnProgress,
    ProgressReporter,
    StageEvent,
    adapt_string_callback,
)
```

`StageEvent` is a hierarchical progress event (`kind: "start" | "progress" | "complete"`, `stage_path: tuple[str, ...]`, `progress: float | None`, `eta_seconds: float | None`, `summary: str | None`). `analyze_beats`, `auto_chapter`, and `generate_from_beats` all accept `on_progress: Callable[[StageEvent], None]` and emit nested stages so consumers can render a tree:

```
◐  audio.analyze
   ✓  load                    (1.4s · 60.0s @ 22050 Hz mono)
   ✓  hpss                    (3.2s · percussive component isolated)
   ◐  chapters
      ✓  chapter 1/2          (8.1s · BPM 122 · 412 beats)
      ◐  chapter 2/2          ETA 7s
```

`ETAEstimator` persists rolling-EMA stage durations to `~/.lqr/videoflow-timings.json` so the second run shows accurate ETAs from the first event. `adapt_string_callback` wraps a legacy `Callable[[str], None]` callback into the new event signature.

## Analysis — scene detection

```python
from videoflow.analysis import (
    DETECTOR_INFO,
    DetectorInfo,
    Scene,
    SceneError,
    detect_scenes,
)

scenes = detect_scenes("video.mp4", detector="adaptive")
```

`DETECTOR_INFO` carries human-readable guidance per detector (best-for / not-great-for / threshold ranges).

## Chapters — typed record + resolver + writers

```python
from videoflow.chapters import (
    Chapter,
    ChapterError,
    load_chapters,
    read_mp4_chapters,
    read_sidecar_chapters,
    read_analysis_chapters,
    write_chapters_sidecar,
    embed_in_mp4,
)
```

`Chapter` is the typed dataclass for one chapter record (`at_ms` / `end_ms` / `name` / `intent` / `content_type` / `confidence` / `evidence`; `to_dict` / `from_dict` for serialisation).

`load_chapters(media_path)` resolves chapters from the highest-priority source. Priority (revised 2026-05-14):

1. `<stem>.chapters.json` sidecar — the editable, authoritative store. Hand-built chapters live here.
2. Embedded MP4 chapters (`ffprobe -show_chapters`) — honoured when no sidecar exists.
3. `<stem>.analysis.json` (forgegen-produced) — fallback.

Returns `None` when no source has chapters. **Principle**: hand-built chapters are authoritative; detection is the fallback, not the default.

### Writers

```python
write_chapters_sidecar(media_path, chapters, *, writer="external")
embed_in_mp4(input_path, output_path, chapters)
```

`write_chapters_sidecar` writes a chapter list to `<stem>.chapters.json` via the merge layer in `videoflow.sidecar`. Records land with `auto_generated: false` so subsequent analyze writers (`videoflow.structural.auto_chapter`) leave them alone. Use this for hand-authored / external-recovery workflows.

`embed_in_mp4` muxes chapters into a copy of *input_path* at *output_path* using FFMETADATA1 + `ffmpeg -codec copy`. No re-encode. Round-trips hand-authored chapters back into the file itself so external tools that only read embedded markers (mpv, QuickTime, Plex) see them.

CLI equivalents: `videoflow chapters-write-sidecar <video> <chapters.json>` and `videoflow chapters-embed <input> <output>`.

## CLI

```python
from videoflow.cli import main
main()  # same entry point as the `videoflow` command
```

See [CLI reference](cli.md) for the command surface.
