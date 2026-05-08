# CLI reference

`videoflow` provides a thin CLI on top of the Python API. Every command
prints JSON by default; add `--human` for a readable summary.

## `auto-chapter`

Detect natural chapter boundaries in long-form audio or video. Writes the structural `<stem>.chapters.json` sidecar — chapters + phrases + energy — that every other lqr tool reads to work against the natural shape of the content instead of treating it as one undifferentiated stream.

```bash
videoflow auto-chapter MEDIA [--target-minutes N] [--no-sidecar] [--human]
```

Flags:

- `--target-minutes N` — average target chapter length in minutes (default 5.5). Boundaries snap to silence and recurrence-cluster edges, so actual lengths vary.
- `--no-sidecar` — skip writing the sidecar; print detection results only.
- `--human` — readable output instead of JSON.

The default writes the sidecar in **merge mode**: a re-run preserves user edits per-record through the field-level merge contract. Records flagged `auto_generated: false` (the safety latch) are frozen entirely; analytical fields on un-latched records are recomputed. See [audio-structure-primitive.md](architecture/audio-structure-primitive.md) for the merge rules.

Example:

```bash
# Detect chapters + write track.chapters.json next to the source
videoflow auto-chapter track.mp4

# Print detection results without persisting
videoflow auto-chapter track.mp4 --no-sidecar --human

# Aim for shorter chunks
videoflow auto-chapter long-set.mp3 --target-minutes 3.0
```

Long-form material (≥ 8 min) is chunked; shorter files return one whole-file chapter. Either way, the sidecar grows the full v2 schema (chapters + phrases from `classify_modes` + energy block from `analyze_beats(chapters=…)`).

## `analyze-beats`

Run beat detection on an audio file.

```bash
videoflow analyze-beats AUDIO [--save FILE] [--human]
```

Flags:

- `--save FILE` — save the full beat map (BPM, beats, onsets, phrases, energy) as JSON for reuse without re-analysing.
- `--human` — print a readable summary instead of raw JSON.

Example:

```bash
videoflow analyze-beats track.mp3 --human
videoflow analyze-beats track.mp3 --save track_beats.json
```

## `generate-funscript`

Generate a `.funscript` from an audio file (or a saved beat map).

```bash
videoflow generate-funscript AUDIO [--source full|percussive] [--low N] [--high N] [-o OUT]
```

Flags:

- `--source full|percussive` — analyse the full mix (melodic / sensual styles) or the HPSS-separated percussive layer (rhythmic / EDM-style).
- `--low N`, `--high N` — clamp the funscript curve to this range (0–100). Useful for "tease" or "intense" presets.
- `-o OUT` — output filename (default: `<audio_stem>.funscript`).

Example:

```bash
videoflow generate-funscript track.mp3
videoflow generate-funscript track.mp3 --source percussive --low 10 --high 90
videoflow generate-funscript track.mp3 --source full --low 20 --high 75
```

## `detect-scenes`

Run scene boundary detection on a video file (PySceneDetect-backed).

```bash
videoflow detect-scenes VIDEO [--detector adaptive|content|threshold] [--threshold N] [--human]
```

Flags:

- `--detector adaptive|content|threshold` — pick the underlying detector. Default `adaptive`.
- `--threshold N` — sensitivity (lower = more boundaries). Detector-specific defaults if omitted.
- `--human` — readable output.

## `detectors`

Print guidance on choosing a scene detector and threshold.

```bash
videoflow detectors --human
```
