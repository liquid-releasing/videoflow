"""videoflow CLI — composable video workflow pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Progress emission — writes to stderr AND optional side-channel file
# ---------------------------------------------------------------------------
#
# Direct CLI use: stderr captures the progress lines as `progress: <label>`.
# Forgegen's Tauri bridge can't read stderr reliably (Tokio's stdio piping
# on Windows doesn't forward Python's stderr), so when VIDEOFLOW_PROGRESS_FILE
# is set the same lines also append to that file. The bridge polls the file
# for live UI updates.

def _emit_progress(label: str) -> None:
    """Emit a stage label to stderr (parseable `progress: ` prefix) and,
    when VIDEOFLOW_PROGRESS_FILE env var is set, append the same line to
    that file. Errors are swallowed — progress reporting never breaks
    the underlying analysis.

    The side-channel file is opened+written+closed per call rather than
    held open across emits. On Windows, GetFileSize for another process
    (forgegen's Rust polling task) lags until the writing handle closes,
    so a long-lived handle hides updates from the poller. Open-per-emit
    is microseconds; pipeline stages emit a handful of labels per run."""
    try:
        print(f"progress: {label}", file=sys.stderr, flush=True)
    except OSError:
        # stderr pipe broken (Tokio closes it on Windows after spawn).
        # Without this catch, BrokenPipeError propagates and the side-
        # channel write below never executes — only the FIRST emit landed
        # because subsequent ones tripped on the closed pipe.
        pass
    path = os.environ.get("VIDEOFLOW_PROGRESS_FILE")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"progress: {label}\n")
    except OSError:
        pass


def _videoflow_version_for_metadata() -> str:
    """Best-effort videoflow version for funscript metadata.generated_from.
    Mirrors structural._videoflow_version() — kept local so cli.py doesn't
    take a top-level import on structural just for this."""
    try:
        from importlib.metadata import version as _v
        return _v("videoflow")
    except Exception:
        return "unknown"


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
    # Also surface to the side-channel so forgegen's bridge sees the
    # failure cause — Tokio can't read python.exe's stderr on Windows.
    path = os.environ.get("VIDEOFLOW_PROGRESS_FILE")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"error: {message}\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# patterns-list
# ---------------------------------------------------------------------------

def cmd_patterns_list(args: argparse.Namespace) -> int:
    from videoflow.patterns import CATALOG, to_json

    if args.human:
        for p in CATALOG:
            print(f"{p.id:14}  {p.label:10}  {p.summary}")
    else:
        print(to_json(indent=2 if args.pretty else None))
    return 0


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
        "stanza_count": len(beat_map.stanzas),
        "beats": beat_map.beats,
        "downbeats": beat_map.downbeats,
        "stanzas": [{"start_ms": s, "end_ms": e} for s, e in beat_map.stanzas],
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
        print(f"stanzas:      {len(beat_map.stanzas)}")
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
# auto-chapter — detect chapter boundaries + emit structural sidecar
# ---------------------------------------------------------------------------

def cmd_auto_chapter(args: argparse.Namespace) -> int:
    from videoflow.structural import AutoChapterError, auto_chapter

    try:
        chapters = auto_chapter(
            args.input,
            target_minutes=args.target_minutes,
            write_sidecar=not args.no_sidecar,
            progress_callback=_emit_progress,
        )
    except FileNotFoundError as exc:
        _err(str(exc), args.human)
        return 1
    except AutoChapterError as exc:
        _err(str(exc), args.human)
        return 1

    sidecar_path = (
        Path(args.input).with_name(Path(args.input).stem + ".chapters.json")
        if not args.no_sidecar else None
    )

    data = {
        "input": str(args.input),
        "chapter_count": len(chapters),
        "chapters": [c.to_dict() for c in chapters],
    }
    if sidecar_path is not None:
        data["sidecar"] = str(sidecar_path)

    if args.human:
        print(f"input:         {args.input}")
        print(f"chapter_count: {len(chapters)}")
        if sidecar_path is not None:
            print(f"sidecar:       {sidecar_path}")
        for i, ch in enumerate(chapters):
            end = ch.end_ms if ch.end_ms is not None else "—"
            print(
                f"  chapter {i + 1:>3}: {ch.at_ms / 1000:>7.1f}s — "
                f"{(end / 1000) if isinstance(end, int) else end:>7}s  "
                f"{ch.content_type or '?':>7}  conf={ch.confidence}"
            )
    else:
        print(json.dumps(data, indent=2))

    return 0


# ---------------------------------------------------------------------------
# generate-funscript — audio/beat-map → .funscript
# ---------------------------------------------------------------------------

def cmd_generate_funscript(args: argparse.Namespace) -> int:
    from videoflow.generate import (
        GenerateError,
        generate_from_beats,
        generate_from_beats_per_chapter,
    )

    input_path = Path(args.input)

    # Determine mode early so we use the right `source` when analysing.
    bundle = None
    if args.recipe_bundle:
        try:
            bundle_text = Path(args.recipe_bundle).read_text(encoding="utf-8")
            bundle = json.loads(bundle_text)
        except (OSError, json.JSONDecodeError) as exc:
            _err(f"invalid recipe-bundle {args.recipe_bundle}: {exc}", args.human)
            return 1
        # The first recipe's `source` defines the audio mix used at
        # analyze time. Re-running HPSS per chapter would require
        # re-loading audio per chapter; v1 of per-chapter recipes uses
        # one source globally and varies stroke-density/tone per chapter.
        recipes = bundle.get("recipes") or []
        if recipes and isinstance(recipes[0], dict):
            args.source = recipes[0].get("source", args.source)

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
        # When a recipe-bundle is supplied, hand its chapter list to
        # analyze_beats so beat tracking runs per-chapter (with progress
        # per chunk) instead of one silent whole-file pass. For long
        # files (multi-hour) the difference is "looks frozen for 30s"
        # vs. "Analyzing chapter 7/15 (412s-619s)…" ticking by.
        chapters_for_beats = None
        if bundle is not None and bundle.get("chapters"):
            from videoflow.chapters import Chapter
            chapters_for_beats = [
                Chapter(at_ms=int(c.get("at_ms", 0)), end_ms=c.get("end_ms"))
                for c in bundle["chapters"]
            ]
        try:
            beat_map = analyze_beats(
                input_path,
                source=args.source,
                tracker=args.tracker,
                locked_bpm=args.locked_bpm,
                chapters=chapters_for_beats,
                progress_callback=_emit_progress,
            )
        except FileNotFoundError as exc:
            _err(str(exc), args.human)
            return 1
        except (BeatError, ValueError) as exc:
            _err(str(exc), args.human)
            return 1

    # Per-chapter generation path: validates strict length, loops chapters
    # with per-recipe stroke-density/tone, embeds full provenance in the
    # funscript metadata so downstream tools (FFP, ForgePlayer) can detect
    # video-edit drift and render per-chapter context without the sidecar.
    if bundle is not None:
        chapters = bundle.get("chapters") or []
        recipes = bundle.get("recipes") or []
        source_meta = bundle.get("source") or {}
        generated_from = {
            "tool": "videoflow",
            "tool_version": _videoflow_version_for_metadata(),
            "source": source_meta,
            "chapters": chapters,
            "recipes": recipes,
        }
        try:
            output = generate_from_beats_per_chapter(
                beat_map,
                chapters=chapters,
                recipes=recipes,
                output=args.output,
                low=args.low,
                high=args.high,
                title=args.title or "",
                generated_from=generated_from,
                progress_callback=_emit_progress,
            )
        except GenerateError as exc:
            _err(str(exc), args.human)
            return 1
    else:
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
                tone_per_stanza=auto_tone,
                energy_normalize=args.energy_normalize,
                stroke_density=args.stroke_density,
                title=args.title or "",
                progress_callback=_emit_progress,
            )
        except GenerateError as exc:
            _err(str(exc), args.human)
            return 1

    data = {
        "output": str(output),
        "bpm": round(beat_map.bpm, 2),
        "beats": len(beat_map.beats),
        "stanzas": len(beat_map.stanzas),
        "duration_ms": beat_map.duration_ms,
    }

    if args.human:
        print(f"output:   {output}")
        print(f"bpm:      {beat_map.bpm:.1f}")
        print(f"beats:    {len(beat_map.beats)}")
        print(f"stanzas:  {len(beat_map.stanzas)}")
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
# chapters-write-sidecar — author a sidecar from an external chapter list
# ---------------------------------------------------------------------------

def cmd_chapters_write_sidecar(args: argparse.Namespace) -> int:
    """Write a hand-authored chapter list to ``<stem>.chapters.json``.

    Source format is either a bare list of chapter dicts or the
    ``{"chapters": [...]}`` envelope. Each chapter must have ``at_ms``
    (or legacy ``at`` / proposal ``start_ms``); ``name`` and ``intent``
    are optional. The records land with ``auto_generated: false`` so
    subsequent ``mode="analyze"`` writers never overwrite them.

    Primary use: Movavi chapter recovery — user reads timestamps off
    Movavi's project file, drops them into a JSON list, runs this
    command to materialise the sidecar next to the source video.
    """
    from videoflow.chapters import Chapter, ChapterError, write_chapters_sidecar
    from videoflow.sidecar import SidecarError

    try:
        raw = json.loads(Path(args.chapters_json).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        _err(str(exc), args.human)
        return 1
    except json.JSONDecodeError as exc:
        _err(f"{args.chapters_json}: invalid JSON: {exc}", args.human)
        return 1

    chapter_list = raw if isinstance(raw, list) else raw.get("chapters", [])
    if not isinstance(chapter_list, list):
        _err(
            f"{args.chapters_json}: expected a list or an object with a "
            f"'chapters' field; got {type(chapter_list).__name__}",
            args.human,
        )
        return 1

    try:
        chapters = [Chapter.from_dict(c) for c in chapter_list]
    except ChapterError as exc:
        _err(str(exc), args.human)
        return 1

    try:
        written = write_chapters_sidecar(
            args.input,
            chapters,
            writer="videoflow.cli",
            writer_version=_videoflow_version_for_metadata(),
        )
    except (FileNotFoundError, SidecarError) as exc:
        _err(str(exc), args.human)
        return 1

    data = {
        "input": str(args.input),
        "source": str(args.chapters_json),
        "sidecar": str(written),
        "chapter_count": len(chapters),
    }
    if args.human:
        print(f"input:         {args.input}")
        print(f"source:        {args.chapters_json}")
        print(f"sidecar:       {written}")
        print(f"chapter_count: {len(chapters)}")
    else:
        print(json.dumps(data, indent=2))
    return 0


# ---------------------------------------------------------------------------
# chapters-to-youtube-description — render the YouTube-pasteable chapter block
# ---------------------------------------------------------------------------

def cmd_chapters_to_youtube_description(args: argparse.Namespace) -> int:
    """Print a YouTube-description chapter block to stdout.

    Loads chapters via :func:`load_chapters` (sidecar wins), validates
    them against YouTube's rules (first at 0:00, ≥3 chapters, each
    ≥10s), and prints the pasteable block. For uploads, embedding the
    chapters into the MP4 via ``chapters-embed`` is also valid —
    YouTube reads embedded markers on upload — but the description-
    text path lets the user re-edit titles after upload without re-
    encoding.
    """
    from videoflow.chapters import (
        ChapterError, format_youtube_description, load_chapters,
    )

    try:
        chapters = load_chapters(args.input)
    except (FileNotFoundError, ChapterError) as exc:
        _err(str(exc), args.human)
        return 1

    if chapters is None:
        _err(
            f"no chapters found for {args.input} (looked at sidecar, "
            f"embedded mp4, and analysis.json)",
            args.human,
        )
        return 1

    try:
        text = format_youtube_description(chapters)
    except ChapterError as exc:
        _err(str(exc), args.human)
        return 1

    # The description block goes to stdout regardless of --human so the
    # caller can pipe it straight into the clipboard / a file. JSON mode
    # wraps it for programmatic consumers.
    if args.human:
        print(text, end="")
    else:
        print(json.dumps({
            "input": str(args.input),
            "chapter_count": len(chapters),
            "description_block": text,
        }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# chapters-embed — mux a chapter set into a video via FFMETADATA1
# ---------------------------------------------------------------------------

def cmd_chapters_embed(args: argparse.Namespace) -> int:
    """Mux the sidecar's chapters into ``<output>`` via ffmpeg ``-codec copy``.

    Reads chapters from the highest-priority source for ``<input>`` (see
    :func:`videoflow.chapters.load_chapters`). With the sidecar-first
    priority, this is the sidecar when present — exactly the
    "I just authored chapters and want them inside the file" workflow.

    No re-encode. Stream-copy is near-instant on multi-GB media. Useful
    when downstream consumers only read embedded markers (some players,
    QuickTime, Plex) and won't pick up the sidecar.
    """
    from videoflow.chapters import (
        ChapterError, embed_in_mp4, load_chapters,
    )

    try:
        chapters = load_chapters(args.input)
    except (FileNotFoundError, ChapterError) as exc:
        _err(str(exc), args.human)
        return 1

    if chapters is None:
        _err(
            f"no chapters found for {args.input} (looked at sidecar, "
            f"embedded mp4, and analysis.json)",
            args.human,
        )
        return 1

    try:
        embed_in_mp4(args.input, args.output, chapters)
    except FileNotFoundError as exc:
        _err(str(exc), args.human)
        return 1
    except ChapterError as exc:
        _err(str(exc), args.human)
        return 1

    data = {
        "input": str(args.input),
        "output": str(args.output),
        "chapter_count": len(chapters),
    }
    if args.human:
        print(f"input:         {args.input}")
        print(f"output:        {args.output}")
        print(f"chapter_count: {len(chapters)}")
    else:
        print(json.dumps(data, indent=2))
    return 0


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

    # patterns-list — modulation library catalog (haptics / stim / multi-axis / edit)
    p_patterns = sub.add_parser(
        "patterns-list",
        help="List the pattern catalog (modulation vocabulary across consumers).",
    )
    p_patterns.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )
    p_patterns.set_defaults(func=cmd_patterns_list)

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

    # auto-chapter — detect chapters + emit structural sidecar
    p_auto = sub.add_parser(
        "auto-chapter",
        help=(
            "Detect natural chapter boundaries and write the structural "
            "<stem>.chapters.json sidecar (chapters + stanzas + energy)."
        ),
    )
    p_auto.add_argument("input", type=Path, help="Input audio or video file.")
    p_auto.add_argument(
        "--target-minutes",
        type=float,
        default=5.5,
        dest="target_minutes",
        help=(
            "Average target chapter length in minutes (default 5.5). "
            "Boundaries snap to silence and recurrence-cluster edges, so "
            "actual lengths vary."
        ),
    )
    p_auto.add_argument(
        "--no-sidecar",
        action="store_true",
        dest="no_sidecar",
        help=(
            "Skip writing the sidecar; print detection results only. "
            "By default the analysis is merged into <stem>.chapters.json "
            "via the field-level merge contract — user edits are preserved "
            "per-record."
        ),
    )
    p_auto.set_defaults(func=cmd_auto_chapter)

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
             "70→30. auto: per-stanza center swing derived from each "
             "stanza's energy slope — rising stanzas climb, falling ones "
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
    p_gen.add_argument(
        "--recipe-bundle",
        type=Path,
        default=None,
        metavar="PATH",
        dest="recipe_bundle",
        help=(
            "Path to a JSON file with per-chapter recipes. Schema: "
            "{version: 1, chapters: [{at_ms, end_ms}, ...], "
            "recipes: [{source, stroke_density, tone, emphasize_beats}, ...], "
            "source: {path, duration_ms, partial_md5, size_bytes}}. "
            "Recipes count must equal chapters count (strict). When set, "
            "single-recipe args (--source, --stroke-density, --tone, "
            "--center-trajectory) are ignored except --source which is "
            "taken from the first recipe for the analyze pass. Source mix "
            "(percussive vs full) is still global per run; stroke-density "
            "and tone vary per chapter."
        ),
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

    # chapters-write-sidecar — author <stem>.chapters.json from an external chapter list
    p_chwrite = sub.add_parser(
        "chapters-write-sidecar",
        help=(
            "Write a hand-authored chapter list to <stem>.chapters.json next to "
            "the source video. Records land with auto_generated: false (LATCH)."
        ),
    )
    p_chwrite.add_argument("input", type=Path, help="Source media file.")
    p_chwrite.add_argument(
        "chapters_json",
        type=Path,
        help=(
            "Path to a JSON file containing either a bare list of chapter "
            "dicts or an object with a 'chapters' field. Each chapter "
            "must have at_ms (or legacy 'at' / proposal 'start_ms')."
        ),
    )
    p_chwrite.set_defaults(func=cmd_chapters_write_sidecar)

    # chapters-to-youtube-description — render the YouTube-pasteable block
    p_chyt = sub.add_parser(
        "chapters-to-youtube-description",
        help=(
            "Print a YouTube-description chapter block to stdout. Validates "
            "against YouTube's rules (first at 0:00, >=3 chapters, each >=10s)."
        ),
    )
    p_chyt.add_argument("input", type=Path, help="Source media file.")
    p_chyt.set_defaults(func=cmd_chapters_to_youtube_description)

    # chapters-embed — mux chapters into an MP4 via ffmpeg -codec copy
    p_chembed = sub.add_parser(
        "chapters-embed",
        help=(
            "Mux chapters into a copy of <input> at <output>. Reads chapters "
            "from the highest-priority source (sidecar first); no re-encode."
        ),
    )
    p_chembed.add_argument("input", type=Path, help="Source media file.")
    p_chembed.add_argument(
        "output",
        type=Path,
        help="Output video path. Must not equal input.",
    )
    p_chembed.set_defaults(func=cmd_chapters_embed)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rc = args.func(args)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — we re-raise after logging
        # Surface uncaught exceptions to the side-channel so forgegen sees
        # them. Python's stderr is invisible to Tokio's piped reader on
        # Windows, so without this the only signal is a non-zero exit code.
        import traceback
        tb = traceback.format_exc()
        path = os.environ.get("VIDEOFLOW_PROGRESS_FILE")
        if path:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"error: uncaught {type(exc).__name__}: {exc}\n")
                    for line in tb.splitlines():
                        f.write(f"error: {line}\n")
            except OSError:
                pass
        raise
    sys.exit(rc)
