# videoflow

Audio analysis + funscript generation engine. Powers
[forgegen](https://github.com/liquid-releasing/forgegen)'s audio-to-funscript
pipeline.

## What it does

Drop in a music track. Get a `.funscript` matched to the rhythm, phrase structure, and energy envelope of the music — in seconds, not hours. For long-form material, the same pipeline runs **per chapter** so an ambient intro and a music-driven climax don't average each other out.

```python
from videoflow.audio import analyze_beats
from videoflow.generate import generate_from_beats
from videoflow.structural import auto_chapter

# Short / single-chunk content
beat_map = analyze_beats("track.mp3")
generate_from_beats(beat_map, "track.funscript")

# Long-form: detect chapter structure, then analyse + generate per chapter
chapters = auto_chapter("set.mp3")  # also writes set.chapters.json
beat_map = analyze_beats("set.mp3", chapters=chapters)
generate_from_beats(beat_map, "set.funscript")
```

## Modules

- **`videoflow.audio`** — beat & BPM detection, onset / energy / spectral analysis, HPSS percussive separation. Chapter-aware mode for long-form content.
- **`videoflow.structural`** — auto-detect natural chapter boundaries (silence + MFCC clustering + silence-snap); writes the structural `<stem>.chapters.json` sidecar.
- **`videoflow.sidecar`** — read / write the structural database with field-level merge so multiple writers (forgegen, FunscriptForge, forgeassembler) cooperate without locking.
- **`videoflow.phrases`** — phrase classification with rich `Phrase` records. Audio-driven (`classify_phrases`) and funscript-driven (`classify_phrases_from_funscript`) classifiers share the same six-mode vocabulary.
- **`videoflow.generate`** — beats → motion curve, curve shaping, validated funscript JSON export. `classify_modes` re-exports from `videoflow.phrases` for back-compat.
- **`videoflow.analysis`** — scene boundary detection (adaptive / content / threshold).
- **`videoflow.progress`** — staged progress events (`StageEvent`) + persisted ETA estimator so multi-minute analyses surface a live tree, not an opaque spinner.
- **`videoflow.cli`** — `videoflow` command-line entry point. See [CLI reference](cli.md).

## Architecture

- **[Audio structure as a first-class primitive](architecture/audio-structure-primitive.md)** — why long-form work runs against detected chapters, the v2 sidecar schema (chapters + phrases + energy + tone + provenance), the field categories, and the field-level merge contract.
- **[Sidecar fragments](architecture/sidecar-fragment.md)** — the reusable-pattern primitive for cookbook recipes, phrase libraries, and saved templates.
- **[`sidecar.schema.json`](architecture/sidecar.schema.json)** — Draft 2020-12 JSON Schema; the formal contract for non-Python consumers.

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
