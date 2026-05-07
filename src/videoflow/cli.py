"""videoflow CLI — composable video workflow pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _out(data: dict, human: bool) -> None:
    if human:
        for k, v in data.items():
            print(f"{k}: {v}")
    else:
        print(json.dumps(data, indent=2))


def _err(message: str, human: bool) -> None:
    if human:
        print(f"error: {message}", file=sys.stderr)
    else:
        print(json.dumps({"error": message}), file=sys.stderr)


# ---------------------------------------------------------------------------
# detect-scenes
# ---------------------------------------------------------------------------

def cmd_detect_scenes(args: argparse.Namespace) -> int:
    from videoflow.analysis import DETECTOR_INFO, SceneError, detect_scenes

    try:
        scenes = detect_scenes(
            args.input,
            threshold=args.threshold,
            detector=args.detector,
        )
    except FileNotFoundError as exc:
        _err(str(exc), args.human)
        return 1
    except ValueError as exc:
        _err(str(exc), args.human)
        return 1
    except SceneError as exc:
        _err(str(exc), args.human)
        return 1

    info = DETECTOR_INFO[args.detector]
    t = args.threshold if args.threshold is not None else info.threshold_default

    data = {
        "input": str(args.input),
        "detector": args.detector,
        "threshold": t,
        "scene_count": len(scenes),
        "scenes": [
            {
                "index": s.index,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "duration_ms": s.duration_ms,
            }
            for s in scenes
        ],
    }

    if args.human:
        print(f"input:       {args.input}")
        print(f"detector:    {args.detector} — {info.name}")
        print(f"threshold:   {t}")
        print(f"scene_count: {len(scenes)}")
        for s in scenes:
            print(
                f"  scene {s.index:>3}: "
                f"{s.start_ms / 1000:.2f}s — {s.end_ms / 1000:.2f}s "
                f"({s.duration_ms / 1000:.2f}s)"
            )
    else:
        print(json.dumps(data, indent=2))

    return 0


def cmd_detectors(args: argparse.Namespace) -> int:
    """List available detectors with guidance."""
    from videoflow.analysis import DETECTOR_INFO

    if args.human:
        for key, info in DETECTOR_INFO.items():
            print(f"\n{key}  —  {info.name}")
            print(f"  {info.description}")
            print(f"  Threshold ({info.threshold_label}): "
                  f"{info.threshold_min} – {info.threshold_max}, "
                  f"default {info.threshold_default}")
            print(f"  Lower:   {info.threshold_low_note}")
            print(f"  Higher:  {info.threshold_high_note}")
            print(f"  Tip:     {info.novice_tip}")
            print(f"  Best for:    {', '.join(info.best_for)}")
            print(f"  Not great for: {', '.join(info.not_great_for)}")
    else:
        data = {
            key: {
                "name": info.name,
                "description": info.description,
                "threshold_label": info.threshold_label,
                "threshold_default": info.threshold_default,
                "threshold_min": info.threshold_min,
                "threshold_max": info.threshold_max,
                "threshold_low_note": info.threshold_low_note,
                "threshold_high_note": info.threshold_high_note,
                "best_for": info.best_for,
                "not_great_for": info.not_great_for,
                "novice_tip": info.novice_tip,
            }
            for key, info in DETECTOR_INFO.items()
        }
        print(json.dumps(data, indent=2))

    return 0


# ---------------------------------------------------------------------------
# analyze-beats
# ---------------------------------------------------------------------------

def cmd_analyze_beats(args: argparse.Namespace) -> int:
    from videoflow.audio import BeatError, analyze_beats

    try:
        beat_map = analyze_beats(
            args.input,
            source=args.source,
            tracker=args.tracker,
            locked_bpm=args.locked_bpm,
        )
    except FileNotFoundError as exc:
        _err(str(exc), args.human)
        return 1
    except ValueError as exc:
        _err(str(exc), args.human)
        return 1
    except BeatError as exc:
        _err(str(exc), args.human)
        return 1

    data = {
        "input": str(args.input),
        "bpm": round(beat_map.bpm, 2),
        "duration_ms": beat_map.duration_ms,
        "beat_count": len(beat_map.beats),
        "downbeat_count": len(beat_map.downbeats),
        "phrase_count": len(beat_map.phrases),
        "beats": beat_map.beats,
        "downbeats": beat_map.downbeats,
        "phrases": [{"start_ms": s, "end_ms": e} for s, e in beat_map.phrases],
        "energy": [round(e, 4) for e in beat_map.energy],
    }

    if args.save:
        save_path = beat_map.save(args.save)
        if args.human:
            print(f"saved:        {save_path}")

    if args.human:
        print(f"input:        {args.input}")
        print(f"bpm:          {beat_map.bpm:.1f}")
        print(f"duration:     {beat_map.duration_ms / 1000:.1f}s")
        print(f"beats:        {len(beat_map.beats)}")
        print(f"downbeats:    {len(beat_map.downbeats)}")
        print(f"phrases:      {len(beat_map.phrases)}")
        print(f"beat interval: {beat_map.beat_interval_ms:.0f}ms")
        if args.beats:
            for i, b in enumerate(beat_map.beats):
                bar = (i // 4) + 1
                beat_in_bar = (i % 4) + 1
                marker = " ← downbeat" if beat_in_bar == 1 else ""
                print(f"  beat {i + 1:>4}  bar {bar:>3}.{beat_in_bar}  {b / 1000:>7.3f}s{marker}")
    else:
        if not args.beats:
            data.pop("beats")
            data.pop("energy")
        print(json.dumps(data, indent=2))

    return 0


# ---------------------------------------------------------------------------
# generate-funscript — audio/beat-map → .funscript
# ---------------------------------------------------------------------------

def cmd_generate_funscript(args: argparse.Namespace) -> int:
    from videoflow.generate import GenerateError, generate_from_beats

    input_path = Path(args.input)

    # Accept either a saved beat-map JSON or an audio/video file
    if input_path.suffix.lower() == ".json":
        from videoflow.audio import AudioBeatMap
        try:
            beat_map = AudioBeatMap.load(input_path)
        except FileNotFoundError as exc:
            _err(str(exc), args.human)
            return 1
        except (KeyError, ValueError) as exc:
            _err(f"invalid beat map JSON: {exc}", args.human)
            return 1
    else:
        from videoflow.audio import BeatError, analyze_beats
        try:
            beat_map = analyze_beats(
                input_path,
                source=args.source,
                tracker=args.tracker,
                locked_bpm=args.locked_bpm,
            )
        except FileNotFoundError as exc:
            _err(str(exc), args.human)
            return 1
        except (BeatError, ValueError) as exc:
            _err(str(exc), args.human)
            return 1

    traj = None
    auto_tone = None
    if args.tone == "rise":
        traj = (30, 70)
    elif args.tone == "fall":
        traj = (70, 30)
    elif args.tone == "auto":
        from videoflow.generate import compute_auto_tone
        auto_tone = compute_auto_tone(beat_map)
    elif args.center_trajectory:
        a, b = args.center_trajectory.split(",")
        traj = (int(a), int(b))

    try:
        output = generate_from_beats(
            beat_map,
            args.output,
            low=args.low,
            high=args.high,
            center=args.center,
            center_trajectory=traj,
            tone_per_phrase=auto_tone,
            energy_normalize=args.energy_normalize,
            stroke_density=args.stroke_density,
            title=args.title or "",
        )
    except GenerateError as exc:
        _err(str(exc), args.human)
        return 1

    data = {
        "output": str(output),
        "bpm": round(beat_map.bpm, 2),
        "beats": len(beat_map.beats),
        "phrases": len(beat_map.phrases),
        "duration_ms": beat_map.duration_ms,
    }

    if args.human:
        print(f"output:   {output}")
        print(f"bpm:      {beat_map.bpm:.1f}")
        print(f"beats:    {len(beat_map.beats)}")
        print(f"phrases:  {len(beat_map.phrases)}")
        print(f"duration: {beat_map.duration_ms / 1000:.1f}s")
    else:
        print(json.dumps(data, indent=2))

    return 0


# ---------------------------------------------------------------------------
# render / concat — placeholder commands. The implementations referenced
# `videoflow.layout` and `videoflow.reel` modules that never landed; the
# subcommands stay registered (so `--help` still mentions them) but
# resolve to a clear "not implemented" error instead of import-time crashes.
# Track the work in BACKLOG.md when these come back.
# ---------------------------------------------------------------------------

def cmd_render(args: argparse.Namespace) -> int:
    _err(
        "videoflow render is not implemented in this build "
        "(canvas / multi-panel layout module not yet landed).",
        args.human,
    )
    return 2


def cmd_concat(args: argparse.Namespace) -> int:
    _err(
        "videoflow concat is not implemented in this build "
        "(reel / clip-assembly module not yet landed).",
        args.human,
    )
    return 2


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videoflow",
        description="Composable video workflow pipeline.",
    )
    parser.add_argument(
        "--human", action="store_true", help="Human-readable output instead of JSON."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # detect-scenes
    p_scenes = sub.add_parser(
        "detect-scenes",
        help="Detect scene boundaries in a video file.",
    )
    p_scenes.add_argument("input", type=Path, help="Input video file.")
    p_scenes.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Detection sensitivity. Uses the detector's default if not set. "
            "adaptive default: 3.0 (scale 1–10). "
            "content default: 27.0 (scale 5–100). "
            "threshold default: 12.0 (brightness 1–255). "
            "Lower = more scenes. Run 'videoflow detectors' for full guidance."
        ),
    )
    p_scenes.add_argument(
        "--detector",
        choices=["adaptive", "content", "threshold"],
        default="adaptive",
        help=(
            "adaptive (default, handles mixed content), "
            "content (hard cuts), "
            "threshold (fade-to-black). "
            "Run 'videoflow detectors --human' for details."
        ),
    )
    p_scenes.set_defaults(func=cmd_detect_scenes)

    # detectors — list available detectors with guidance
    p_det = sub.add_parser(
        "detectors",
        help="List available detectors with threshold guidance.",
    )
    p_det.set_defaults(func=cmd_detectors)

    # analyze-beats
    p_beats = sub.add_parser(
        "analyze-beats",
        help="Analyse the beat structure of an audio or video file.",
    )
    p_beats.add_argument("input", type=Path, help="Input audio or video file.")
    p_beats.add_argument(
        "--beats",
        action="store_true",
        help="Include full beat timestamp list in output.",
    )
    p_beats.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save beat map to a JSON file for reuse (e.g. track_beat.json).",
    )
    p_beats.add_argument(
        "--source",
        choices=["full", "percussive"],
        default="full",
        help=(
            "full (default): use the full mix. "
            "percussive: apply HPSS first — voice and melody are removed "
            "before beat tracking, so beats follow the drums."
        ),
    )
    p_beats.add_argument(
        "--tracker",
        choices=["auto", "beat_track", "plp"],
        default="auto",
        help=(
            "auto (default): plp for tracks > 10 min, beat_track otherwise. "
            "beat_track: classic global-tempo DP tracker (best for short, "
            "steady-tempo tracks). "
            "plp: predominant local pulse — robust on long-form material "
            "where tempo drifts."
        ),
    )
    p_beats.add_argument(
        "--locked-bpm",
        type=float,
        default=None,
        metavar="BPM",
        dest="locked_bpm",
        help=(
            "Pin reported BPM and bias the tracker toward this tempo. "
            "Useful when auto-detection lands on a half/double octave."
        ),
    )
    p_beats.set_defaults(func=cmd_analyze_beats)

    # generate-funscript
    p_gen = sub.add_parser(
        "generate-funscript",
        help="Generate a .funscript from an audio/video file or saved beat map.",
    )
    p_gen.add_argument(
        "input", type=Path,
        help="Audio/video file (MP3, WAV, MP4…) or a saved beat map .json.",
    )
    p_gen.add_argument(
        "output", type=Path,
        help="Output .funscript file path.",
    )
    p_gen.add_argument(
        "--source",
        choices=["full", "percussive"],
        default="percussive",
        help=(
            "percussive (default): strip voice/melody before beat tracking — "
            "best for music with vocals. "
            "full: use the raw mix."
        ),
    )
    p_gen.add_argument(
        "--tracker",
        choices=["auto", "beat_track", "plp"],
        default="auto",
        help=(
            "auto (default): plp for tracks > 10 min, beat_track otherwise. "
            "beat_track: classic global-tempo DP tracker. "
            "plp: predominant local pulse — robust on long-form material "
            "where tempo drifts."
        ),
    )
    p_gen.add_argument(
        "--locked-bpm",
        type=float,
        default=None,
        metavar="BPM",
        dest="locked_bpm",
        help="Pin BPM and bias the tracker toward this tempo.",
    )
    p_gen.add_argument(
        "--low", type=int, default=10, metavar="POS",
        help="Trough floor 0–100 (default 10). In centered mode, sets the "
             "lower bound of the stroke at full energy.",
    )
    p_gen.add_argument(
        "--high", type=int, default=90, metavar="POS",
        help="Peak ceiling 0–100 (default 90). In centered mode, sets the "
             "upper bound of the stroke at full energy.",
    )
    p_gen.add_argument(
        "--center", type=int, default=None, metavar="POS",
        help="Curve midpoint 0–100. When set, the curve oscillates "
             "symmetrically around this value (PD-style). When omitted, "
             "uses the legacy trough-at-low model.",
    )
    p_gen.add_argument(
        "--energy-normalize", action="store_true",
        help="Normalise energies to the 95th-percentile beat = 1.0. Ensures "
             "loud beats reach the configured high/low instead of compressing "
             "into the middle.",
    )
    p_gen.add_argument(
        "--stroke-density",
        choices=["half", "full", "1", "2", "4", "8"],
        default="half",
        help="half|1 (default): one action per beat — alternating peak/trough "
             "across beats (one stroke spans two beats, sensual). "
             "full|2: two actions per beat — peak + trough each beat "
             "(canonical PD-style). "
             "4: four actions per beat (two strokes per beat, dense). "
             "8: eight actions per beat (very dense; reserved for "
             "short climactic chapters).",
    )
    p_gen.add_argument(
        "--tone",
        choices=["flat", "rise", "fall", "auto"],
        default="flat",
        help="Whole-shape gestalt for the curve. flat (default): constant "
             "center 50. rise: center drifts 30→70 over the track. fall: "
             "70→30. auto: per-phrase center swing derived from each "
             "phrase's energy slope — rising phrases climb, falling ones "
             "relax. Works on drones too (energy moves even without melody).",
    )
    p_gen.add_argument(
        "--center-trajectory", default=None, metavar="START,END",
        help="Custom center trajectory as 'start,end' (each 0–100). E.g. "
             "'40,80' rises from 40 to 80. Overrides --tone if both given.",
    )
    p_gen.add_argument(
        "--title", default="", metavar="TEXT",
        help="Optional title stored in funscript metadata.",
    )
    p_gen.set_defaults(func=cmd_generate_funscript)

    # render — execute a canvas.json edit file
    p_render = sub.add_parser(
        "render",
        help="Render a canvas edit JSON file to video.",
    )
    p_render.add_argument("edit", type=Path, help="Path to canvas.json edit file.")
    p_render.add_argument(
        "--output", type=Path, default=None,
        help="Output video path (overrides 'output' field in JSON).",
    )
    p_render.add_argument(
        "--crf", type=int, default=18,
        help="H.264 quality factor (lower = better, default 18).",
    )
    p_render.add_argument(
        "--preset", default="fast",
        help="ffmpeg encoding preset (default: fast).",
    )
    p_render.set_defaults(func=cmd_render)

    # concat — assemble clips from a folder or JSON reel file
    p_concat = sub.add_parser(
        "concat",
        help="Concat video clips with gaps and chapter markers.",
    )
    p_concat_src = p_concat.add_mutually_exclusive_group(required=True)
    p_concat_src.add_argument(
        "--from-folder",
        type=Path,
        metavar="DIR",
        dest="from_folder",
        help="Scan a folder for .mp4 files (sorted by name).",
    )
    p_concat_src.add_argument(
        "reel",
        type=Path,
        nargs="?",
        help="Path to a reel.json file.",
    )
    p_concat.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output video file (required when using --from-folder).",
    )
    p_concat.add_argument(
        "--gap",
        type=int,
        default=2000,
        metavar="MS",
        help="Gap duration in milliseconds between clips (default 2000).",
    )
    p_concat.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern when using --from-folder (default '*.mp4').",
    )
    p_concat.add_argument(
        "--sort",
        choices=["name", "date"],
        default="name",
        help="Sort order for --from-folder: name (default, alphabetical) or date (modification time, oldest first).",
    )
    p_concat.add_argument(
        "--canvas",
        default="1920x1080",
        metavar="WxH",
        help="Canvas size for scaling (default '1920x1080').",
    )
    p_concat.add_argument(
        "--crf", type=int, default=18,
        help="H.264 quality factor (lower = better, default 18).",
    )
    p_concat.add_argument(
        "--preset", default="fast",
        help="ffmpeg encoding preset (default: fast).",
    )
    p_concat.set_defaults(func=cmd_concat)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))
