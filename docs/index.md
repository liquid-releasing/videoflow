# videoflow

Audio analysis + funscript generation engine. Powers
[forgegen](https://github.com/liquid-releasing/forgegen)'s audio-to-funscript
pipeline.

## What it does

Drop in a music track. Get a `.funscript` matched to the rhythm, phrase
structure, and energy envelope of the music — in seconds, not hours.

```python
from videoflow.audio import analyze_beats
from videoflow.generate import generate_from_beats

beat_map = analyze_beats("track.mp3")
funscript = generate_from_beats(beat_map, style="rhythmic")
```

## Modules

- **`videoflow.audio`** — beat & BPM detection, onset / energy / spectral analysis, HPSS percussive separation.
- **`videoflow.generate`** — beats → motion curve, phrase classification (`break` / `tease` / `slow` / `steady` / `fast` / `edging`), curve shaping, validated funscript JSON export.
- **`videoflow.analysis`** — scene boundary detection (adaptive / content / threshold).
- **`videoflow.cli`** — `videoflow` command-line entry point.

## Install

```bash
pip install -e ".[audio]"            # audio path (librosa)
pip install -e ".[scenes]"           # scene detection (PySceneDetect)
pip install -e ".[audio,scenes,dev]" # everything + tests
```

FFmpeg required at runtime for audio decode and scene detection.

## Next

- **[CLI reference](cli.md)** — every command and flag.
- **[API reference](api.md)** — public Python API.

## License

MIT. © 2026 Liquid Releasing.
