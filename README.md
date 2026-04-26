# videoflow

Audio analysis + funscript generation engine. The library that powers
[forgegen](https://github.com/liquid-releasing/forgegen)'s audio-to-funscript
pipeline.

Drop in a music track. Get a quality `.funscript` matched to the rhythm,
phrase structure, and energy envelope of the music — in seconds, not hours.

---

## What's in the box

| Module | What it does |
| --- | --- |
| `videoflow.audio` | Beat & BPM detection, onsets, energy, HPSS percussive separation |
| `videoflow.generate` | Beats → motion curve, phrase classification, curve shaping, funscript export |
| `videoflow.analysis` | Scene boundary detection (adaptive / content / threshold) |
| `videoflow.cli` | One-shot CLI for the above |

---

## Install

```bash
pip install -e ".[audio]"            # core + beat analysis (librosa)
pip install -e ".[scenes]"           # core + scene detection (PySceneDetect)
pip install -e ".[audio,scenes,dev]" # everything + tests
```

FFmpeg is needed at runtime. Install from [ffmpeg.org](https://ffmpeg.org)
or via `imageio-ffmpeg` (pulled by downstream products).

---

## Quick start — analyse a track

```python
from videoflow.audio import analyze_beats

beat_map = analyze_beats("track.mp3")
print(f"{beat_map.bpm:.1f} BPM — {len(beat_map.beats)} beats")
beat_map.save("track_beats.json")   # reuse without re-analysing
```

CLI:

```bash
videoflow analyze-beats track.mp3 --human
videoflow analyze-beats track.mp3 --save track_beats.json
```

---

## Quick start — generate a funscript

```python
from videoflow.audio import analyze_beats
from videoflow.generate import generate_from_beats

beat_map = analyze_beats("track.mp3")
funscript = generate_from_beats(beat_map, style="rhythmic")
# funscript is a dict ready to be written as JSON
```

CLI:

```bash
videoflow generate-funscript track.mp3
videoflow generate-funscript track.mp3 --source full --low 20 --high 75
```

Style presets: `rhythmic` (percussive, EDM-style beat lock), `sensual`
(full-mix, slow + melodic), `intense` (max range), `chaotic` (full mix,
unpredictable peaks).

---

## CLI reference

```bash
videoflow analyze-beats AUDIO  [--save FILE] [--human]
videoflow generate-funscript AUDIO [--source full|percussive] [--low N] [--high N] [-o OUT]
videoflow detect-scenes VIDEO  [--detector adaptive|content|threshold] [--threshold N] [--human]
videoflow detectors            # guidance on choosing a detector
```

Every command outputs JSON by default. Add `--human` for readable output.

---

## Tests

```bash
pytest tests/
```

---

## License

MIT. See [LICENSE](LICENSE).
