# Backlog

Features and improvements in priority order.
Items at the top are closest to ready. Items at the bottom are future/exploratory.

---

## Ready to build

### Audio beat analysis — librosa (V1)

`analyze_beats(input)` → `AudioBeatMap`. Wraps librosa for onset detection,
BPM estimation, beat grid, and downbeat tracking. Accepts audio or video files
(extracts audio automatically). Foundation for beat-snap and beat-grid assembly.

```python
from videoflow.audio import analyze_beats

beat_map = analyze_beats("track.mp3")
# beat_map.bpm, beat_map.beats, beat_map.downbeats, beat_map.phrases
```

---

### Simple split (media-tools)

`split(input, at_ms)` → two files. `split(input, [t1, t2, t3])` → N+1 files.
Stream copy — no re-encode. Lives in media-tools, same pattern as `clip()`.

---

### Beat-snap

Adjust a manual cut point to the nearest beat. Direction: `nearest`, `before`,
`after`. Works across tempo changes — the beat map knows every beat timestamp
regardless of BPM variation. Foundation for the UI snap-to-beat interaction.

---

### Beat-grid assembly

Assemble clips from multiple source videos, each cut to exactly N beats.
The beat map is the master timeline; clips breathe with the music.
See spec for full design.

---

## Near-term

### Speech-aware editing — filler/stammer detection (V1)

`analyze_speech(input)` → `SpeechAnalysis`. Uses Whisper (open-source) for
word-level timestamped transcription. Detects filler words ("uh", "um", "right?",
"you know"), stammers (repeated tokens), and long pauses (silence threshold).

```python
from videoflow.speech import analyze_speech, remove_fillers

speech = analyze_speech("interview.mp4")
clean_segments = remove_fillers("interview.mp4", speech, remove=["uh", "um", "right?"])
# → list of (start_ms, end_ms) segments — feed into concat_videos()
```

One audio pass → filler markers + stammer markers + pause markers + full transcript.
Filler cuts can be beat-snapped via `snap_to_beat()` so music doesn't stumble at the cut.
Configurable filler word list; `pad_ms` to soften cuts.

**Why it's near-term:** Every speaker recording benefits. Powers the customer lifecycle
agent (pre-sales recordings, onboarding tutorials, support videos). Descript solves this
as a product; videoflow makes it a composable pipeline step.

---

### Production board — pipeline stage tracker

At 2+ videos/week across multiple machines, you need to see where every project
is in the pipeline at a glance. Not just job progress — stage status per project.

```text
PROJECT              RAW    TOPAZ   EDIT   RENDER  DELIVER
─────────────────────────────────────────────────────────
dc_metro             ✓      ✓       ►      ·       ·
customer_acme        ✓      ►       ·      ·       ·        71%
music_video_01       ✓      ✓       ✓      ✓       ✓
music_video_02       ►      ·       ·      ·       ·
```

Each project is a folder containing a `project.json` tracking stage status
**and file locations**. Knowing the stage isn't enough — you need to know
where the actual files are across machines and drives.

```json
{
  "project": "dc_metro",
  "stages": {
    "raw":    {"status": "done", "machine": "nas",      "path": "//NAS/raw/dc_metro/"},
    "topaz":  {"status": "done", "machine": "machine2", "path": "D:/topaz/dc_metro/"},
    "edit":   {"status": "in_progress", "machine": "machine1", "path": "C:/edit/dc_metro/"},
    "render": {"status": "pending"},
    "deliver":{"status": "pending"}
  }
}
```

Any machine can update it. No server — coordination via shared folder
(NAS, Dropbox, OneDrive). Terminal table refreshes every 30s.

Stages: `raw` → `topaz` → `edit` → `render` → `deliver`.
Each stage: `·` (pending) / `►` (in progress, with %) / `✓` (done, with path) / `✗` (failed).

The board shows status AND location — no hunting across drives:

```text
PROJECT     RAW        TOPAZ           EDIT        RENDER  DELIVER
──────────────────────────────────────────────────────────────────
dc_metro    NAS ✓      machine2/D: ✓   machine1 ►  ·       ·
acme_video  NAS ✓      machine3/D: ►   ·           ·       ·   62%
```

Stage transitions are tracked — the "did I already copy this to the edit machine?"
question is answered by the project.json, not by memory.

```python
from videoflow.monitor import ProductionBoard

board = ProductionBoard(projects_dir="~/Videos/projects/")
board.watch()   # live terminal table, Ctrl+C to exit
```

CLI:

```bash
videoflow board                    # live table
videoflow board --once             # print once and exit
videoflow board --project dc_metro # single project status
```

**Why this matters at scale:** Two videos a week means 8–10 projects in flight
at any time across 4 machines. Without the board you're SSHing into machines
or opening folders manually 3–4× a day. The board makes it a 2-second glance.

---

### Multi-machine job monitor

Live terminal table of all jobs/ folder entries. Refreshes every 30s.
No server — coordination via shared folder (NAS, Dropbox, OneDrive).
Granular job-level progress within a pipeline stage (e.g. Topaz % per file).

---

### Topaz Video AI script generator

Generate batch scripts for Topaz Video AI from human-friendly inputs
(model, target resolution, GPU device index).
Nice-to-have for power users running 4+ machines; manual Topaz is fine for V1.
RTX 4070 / `hevc_nvenc` primary target.

---

### Smart crop — energy histogram

4K → ultra-wide QHD (3440×1440, 21:9). Auto-crop mode uses vertical spatial
energy to find the region of interest. No ML required for V1.

---

### Multi-panel canvas layout

Compose multiple video streams onto a single wider-than-4K canvas.
Panels run at independent speeds; inner panels use smart crop to show the
region of highest energy from the outer panel's source.

Standard layout: 4 panels — two outer portrait + two inner square — on a
4860×2160 canvas (fits four 9:16 portrait streams side by side).

```python
from videoflow.layout import MultiPanelCanvas, Panel

canvas = MultiPanelCanvas(
    panels=[
        Panel(input="tunnel.mp4",   speed=2.0, position="outer_left",  crop="full"),
        Panel(input="tunnel.mp4",   speed=2.0, position="inner_left",  crop="smart"),
        Panel(input="platform.mp4", speed=0.5, position="inner_right", crop="smart"),
        Panel(input="platform.mp4", speed=0.5, position="outer_right", crop="full"),
    ],
    canvas_size=(4860, 2160),
    beat_map=beat_map,
)
canvas.set_finale(clip=capitol_clip, beats=8, mode="full_width")
canvas.render(output="dc_metro.mp4")
```

Implementation: FFmpeg `filter_complex` — `setpts` for speed, smart crop for
inner panels, `hstack` or `overlay` to compose the canvas. The finale
`mode="full_width"` expands a single clip across all four panel positions.

The dance principle: panels are dancers, beat is the conductor.
Fast outer vs. slow outer = tension. Panels syncing on the reveal = release.

---

## Future / exploratory

### madmom — higher-accuracy beat tracking

**Why it's here:** librosa is good for most music, but its beat tracker can
drift on complex rhythms — live recordings with tempo fluctuation, jazz,
polyrhythm, or music where the downbeat is ambiguous. madmom uses a recurrent
neural network (RNN) trained specifically on beat tracking, and consistently
outperforms librosa on these edge cases. Its downbeat detection is also more
reliable, which matters for beat-grid assembly where a wrong downbeat shifts
every cut in a measure.

**Why it's not V1:** madmom requires compiled C extensions and is harder to
install than librosa. It also processes slower. For a first release and for
most pop/electronic music (the primary use case), librosa is sufficient.

**How to add it:** expose as an optional backend parameter on `analyze_beats()`:

```python
beat_map = analyze_beats("track.mp3", backend="madmom")
# Falls back to librosa if madmom not installed, with a warning
```

Install: `pip install "videoflow[madmom]"` — separate optional dep group.

---

### Demucs — full stem separation (vocals, drums, bass, other)

[Demucs](https://github.com/facebookresearch/demucs) (Meta Research) separates audio into four clean stems: **drums**, **bass**, **vocals**, **other**. Open source, state of the art quality.

**Use cases in videoflow:**

1. **Drums-only beat tracking** — most accurate possible BPM/grid for complex music (jazz, live recordings, anything where HPSS is not clean enough):

   ```python
   beat_map = analyze_beats("track.mp3", source="drums")  # requires demucs
   ```

2. **Speech rhythm analysis** — isolate the vocal stem, detect speech onsets and emphasis patterns. Feeds the speech-editing feature: find where a speaker stresses words or pauses, use those as cut points:

   ```python
   speech_map = analyze_beats("interview.mp4", source="vocal")  # requires demucs
   ```

3. **Music-only export** — strip vocals from a track before attaching it to a canvas render. Useful when you want the instrumental bed without a lyric clash.

**How to add it:** `source` parameter on `analyze_beats()` already exists.
Add `"drums"` and `"vocal"` as valid values. Before analysis, run demucs to
produce the stem, then pass the stem file to the existing librosa pipeline.
Demucs writes stems to disk — keep them in a temp dir unless the caller asks
to save them.

```python
beat_map = analyze_beats("track.mp3", source="drums")   # drums stem → librosa
beat_map = analyze_beats("track.mp3", source="vocal")   # vocal stem → onset detection
```

**Install:** `pip install "videoflow[stems]"` — demucs as a separate optional dep group.
GPU recommended (CUDA) for speed; CPU works but is slow on long files.

**Why it's not V1:** HPSS (`source="percussive"`) covers 90% of the use case with zero
extra deps. Demucs is worth adding once the core pipeline is in regular use and the
drums/vocal distinction becomes a real need.

---

### Real-ESRGAN integration

Free AI upscaler. Frame extract → process → reassemble.
Good 2× and 4× quality. Compare against Topaz on a test clip before committing.

---

### Drone / geospatial analysis

Anomaly detection, historical comparison, object detection (YOLO), temporal
progression. See spec for full design. Separate track — build after core
video production features are stable.

---

### DaVinci Resolve scripting

Python API wrapper for import, timeline, color grade, Smart Reframe, export.
Requires Resolve to be running — investigate headless options.

---

### 3D rendering integration

Headless render script generation for Blender, Cinema 4D, and Unreal Engine.
The rendered frames or video clips feed directly into the videoflow pipeline —
3D output is just another clip source for `Reel`, `MultiPanelCanvas`, or beat-grid assembly.

**Target renderers (V1 priority order):**

1. **Blender** — open source, scriptable via `blender --background --python render.py`.
   Cycles + EEVEE. GPU selection via `bpy.context.preferences.addons['cycles'].preferences`.
   Output: image sequence → `concat_videos()` reassembles to mp4.

2. **Unreal Engine (Movie Render Queue)** — command-line render via
   `UnrealEditor-Cmd.exe project.uproject -MoviePipelineLocalExecutorClass=...`.
   More complex; useful for real-time-quality cinematic output.

3. **Cinema 4D** — `c4dpy` scripting or command-line token-based rendering.
   Lower priority; less common in indie video production.

**API concept:**

```python
from videoflow.render3d import BlenderRenderJob

job = BlenderRenderJob(
    blend_file="scene.blend",
    output_dir="frames/",
    frame_range=(1, 240),  # 10s at 24fps
    device="GPU",          # "GPU" or "CPU"
    engine="CYCLES",       # "CYCLES" or "EEVEE"
    samples=128,
)
job.run()  # blocks; streams progress

# Then feed into pipeline
reel = Reel([ReelClip("frames/output.mp4"), ReelClip("live_footage.mp4")])
reel.render("final.mp4")
```

**Integration with production board:**
The `render3d` stage sits between `topaz` and `edit` in the pipeline.
`project.json` tracks which machine is running the Blender job, progress %, and output path.
GPU-aware: pick the machine with the fastest card (RTX 4070 for most jobs).
Progress: parse Blender's stdout (`Fra:240 Mem:...`) for `%` updates in the production board.

**Why it's deep in the backlog:**
Blender rendering is a standalone step — you set it up, it runs overnight.
The real value is (a) integrating its output seamlessly into the pipeline and
(b) tracking it in the production board alongside other stages.
Neither is urgent until the rest of the pipeline is running smoothly.
