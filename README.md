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
| `videoflow.audio` | Beat & BPM detection (PLP for long-form, beat_track for short), onsets, energy, HPSS percussive separation, optional locked-BPM mode, progress callbacks |
| `videoflow.generate` | Beats → motion curve, phrase classification, mode-aware shaping, centered model + tone primitives + sub-beat density (1/2/4/8 actions per beat), funscript export |
| `videoflow.events` | Funscript metadata events read/write (edge_hold / accent / climax_candidate / vocal_cue / scene_accent…) — format scaffolding for forgevents + restim |
| `videoflow.chapters` | Chapter resolver — mp4 markers (ffprobe) → sidecar JSON → analysis JSON |
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

# Long-form tracks → PLP tracker auto-picks (drift-resistant on >10 min)
beat_map = analyze_beats("90min_set.mp3")  # tracker="auto"

# Lock to a known tempo when half/double-octave detection misfires
beat_map = analyze_beats("track.mp3", locked_bpm=128.0)

# Live progress feedback for long pipelines
def on_stage(label):
    print(label)
beat_map = analyze_beats("video.mp4", progress_callback=on_stage)
# → "Extracting audio from video (ffmpeg)…"
# → "Loading audio (librosa)…"
# → "Detecting beats (PLP — long-form stable)…"
# → "Computing phrases + per-beat energy…"
```

CLI:

```bash
videoflow analyze-beats track.mp3 --human
videoflow analyze-beats track.mp3 --save track_beats.json
videoflow analyze-beats track.mp3 --tracker plp --locked-bpm 128
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
videoflow generate-funscript track.mp3 --stroke-density 4 --tone auto --center 50
videoflow generate-funscript long_set.mp3 --tracker plp --locked-bpm 128
```

Style presets: `rhythmic` (percussive, EDM-style beat lock), `sensual`
(full-mix, slow + melodic), `intense` (max range), `chaotic` (full mix,
unpredictable peaks).

Stroke density: `1`/`half` (one stroke per two beats — sensual),
`2`/`full` (canonical PD-style — one stroke per beat), `4` (two strokes
per beat — dense), `8` (four strokes per beat — saturated; reserved
for short climactic chapters).

Tone (whole-shape macro arc): `flat` (constant centre 50), `rise`
(centre drifts 30→70 over track), `fall` (70→30), `auto` (per-phrase
energy-slope shaping — adapts to whatever audio is loaded).

Centered stroke model: pass `--center 50` (or any 0–100) to make the
curve oscillate symmetrically around that midpoint instead of rising
from a fixed `low` floor. Matches PythonDancer / haptic-rest convention
where the device idles at mid-stroke.

---

## CLI reference

```bash
videoflow analyze-beats AUDIO  [--save FILE] [--source full|percussive]
                               [--tracker auto|beat_track|plp]
                               [--locked-bpm BPM] [--beats] [--human]

videoflow generate-funscript AUDIO|BEATMAP_JSON OUTPUT
                               [--source full|percussive]
                               [--tracker auto|beat_track|plp]
                               [--locked-bpm BPM]
                               [--low N] [--high N] [--center N]
                               [--energy-normalize]
                               [--stroke-density 1|2|4|8|half|full]
                               [--tone flat|rise|fall|auto]
                               [--center-trajectory START,END]
                               [--title TEXT]

videoflow detect-scenes VIDEO  [--detector adaptive|content|threshold]
                               [--threshold N] [--human]

videoflow detectors            # guidance on choosing a detector
```

Every command outputs JSON by default. Add `--human` for readable output.

---

## Tests

```bash
pytest tests/
```

---

## Acknowledgements

- **[Funscript-Flow](https://github.com/Funscript-Flow/Funscript-Flow)**
  (Apache-2.0) — the computer-vision video→funscript generator whose
  optical-flow output independently **confirmed the depth law** this engine
  implements: mapping any signal (audio energy *or* video flow magnitude) to
  stroke *depth* yields a centered bell, while a fixed full-depth backbone with
  the signal driving only timing/density yields the bimodal, rail-to-rail shape
  of a great script. That two-modality agreement is why we treat it as a law,
  not a heuristic. Funscript-Flow is also the intended external video timing
  source for the downstream [forgegen](https://github.com/liquid-releasing/forgegen)
  pipeline. We also adopted its rolling local min/max normalisation idea for
  forcing rails on signals of unreliable scale. Thanks to its authors.

---

## License

MIT. See [LICENSE](LICENSE). The acknowledged project above is independent and
distributed under its own license.
