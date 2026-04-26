# CLI reference

`videoflow` provides a thin CLI on top of the Python API. Every command
prints JSON by default; add `--human` for a readable summary.

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
