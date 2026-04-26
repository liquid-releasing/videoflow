# API reference

Public Python API of videoflow v0.0.1.

## Audio

```python
from videoflow.audio import (
    AudioBeatMap,
    BeatError,
    analyze_beats,
)
```

### `analyze_beats(path, source="percussive") -> AudioBeatMap`

Analyse an audio file. Returns an `AudioBeatMap` with:

- `bpm: float` — detected tempo.
- `beats: list[float]` — beat times in seconds.
- `phrases: list[Phrase]` — phrase boundaries with energy classification.
- `duration_ms: int` — track length in milliseconds.

`source` is `"percussive"` (HPSS-separated drums — beat-locked, EDM-style) or `"full"` (full mix — better for melodic / slow content). Raises `BeatError` if the file can't be decoded or no rhythmic content is found.

### `AudioBeatMap`

```python
beat_map.save("track_beats.json")            # serialise
loaded = AudioBeatMap.load("track_beats.json")
nearest = beat_map.nearest_beat(seconds)     # snap a time to the closest beat
in_range = beat_map.beats_in_range(start_s, end_s)  # slice the beat grid
```

## Generate

```python
from videoflow.generate import (
    GenerateError,
    beats_to_curve,
    classify_modes,
    export_funscript,
    generate_from_beats,
    shape_curve,
)
```

### `generate_from_beats(beat_map, style="rhythmic", low=10, high=90) -> dict`

End-to-end convenience: analysed beats → shaped funscript dict ready to write
as JSON. `style` is `"rhythmic"`, `"sensual"`, `"intense"`, or `"chaotic"`.

### Lower-level building blocks

If you want to compose the pipeline yourself:

```python
curve = beats_to_curve(beat_map, low=10, high=90)
modes = classify_modes(beat_map)
shaped = shape_curve(curve, modes)
funscript = export_funscript(shaped, beat_map.duration_ms)
```

Each step is independently testable. `shape_curve` applies per-mode easing,
velocity profiles, and semantic shaping (tease oscillations, edging ramps,
break minimisation).

## Analysis (scene detection)

```python
from videoflow.analysis import (
    DETECTOR_INFO,
    DetectorInfo,
    Scene,
    SceneError,
    detect_scenes,
)

scenes = detect_scenes("video.mp4", detector="adaptive")
for s in scenes:
    print(f"{s.start_s:.1f}s → {s.end_s:.1f}s")
```

`DETECTOR_INFO` returns guidance on picking a detector and tuning the threshold.

## CLI

```python
from videoflow.cli import main
main()  # same entry point as the `videoflow` command
```
