"""videoflow — composable audio + video pipeline for haptic content generation.

Audio path is the v0.1 product surface: analyze a track for beats and energy,
classify stanzas, shape a motion curve, export a funscript. forgegen is the
downstream Streamlit UI on top of this.

Public names are re-exported LAZILY (PEP 562). Importing this package used to
pull in every submodule eagerly, which meant `from videoflow.sidecar import
forge_dir` — a pure path helper — dragged in videoflow.audio, then descreech,
then scipy.ndimage: 460ms measured, on a call that touches no audio at all.

FunscriptForge pays that on every CLI invocation, and the Events tab writes
its sidecar through the CLI on each edit, so the cost landed on every event a
user entered (dogfood 2026-09-05: "the lag in being able to edit after
entering an event persists").

`from videoflow import analyze_beats` still works exactly as before — Python
falls back to __getattr__ for a name the module does not already define — and
the submodule itself is imported the first time one of its names is touched.
"""

import importlib

# Public name -> the submodule that defines it. Keep in step with __all__;
# the test suite asserts the two agree.
_EXPORTS = {
    "DETECTOR_INFO": "videoflow.analysis",
    "DetectorInfo": "videoflow.analysis",
    "Scene": "videoflow.analysis",
    "SceneError": "videoflow.analysis",
    "detect_scenes": "videoflow.analysis",
    "AudioBeatMap": "videoflow.audio",
    "BeatError": "videoflow.audio",
    "analyze_beats": "videoflow.audio",
    "Chapter": "videoflow.chapters",
    "ChapterError": "videoflow.chapters",
    "load_chapters": "videoflow.chapters",
    "GenerateError": "videoflow.generate",
    "beats_to_curve": "videoflow.generate",
    "classify_modes": "videoflow.generate",
    "export_funscript": "videoflow.generate",
    "generate_from_beats": "videoflow.generate",
    "shape_curve": "videoflow.generate",
    "Stanza": "videoflow.stanzas",
    "classify_stanzas": "videoflow.stanzas",
    "classify_stanzas_from_funscript": "videoflow.stanzas",
    "ETAEstimator": "videoflow.progress",
    "OnProgress": "videoflow.progress",
    "ProgressReporter": "videoflow.progress",
    "StageEvent": "videoflow.progress",
    "adapt_string_callback": "videoflow.progress",
    "SidecarError": "videoflow.sidecar",
    "chapters_from_sidecar": "videoflow.sidecar",
    "read_sidecar": "videoflow.sidecar",
    "sidecar_path_for": "videoflow.sidecar",
    "write_sidecar": "videoflow.sidecar",
    "AutoChapterError": "videoflow.structural",
    "auto_chapter": "videoflow.structural",
}


def __getattr__(name):
    """Resolve a public name, or a submodule, on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value          # cache: __getattr__ runs once per name
        return value
    # `import videoflow; videoflow.audio...` used to work because __init__
    # imported the submodule as a side effect. Keep that working rather than
    # turning a lazy import into a breaking change.
    try:
        module = importlib.import_module(f"videoflow.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module 'videoflow' has no attribute {name!r}") from None
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(__all__) | set(globals()))


__all__ = [
    "DETECTOR_INFO",
    "DetectorInfo",
    "Scene",
    "SceneError",
    "detect_scenes",
    "AudioBeatMap",
    "BeatError",
    "analyze_beats",
    "Chapter",
    "ChapterError",
    "load_chapters",
    "AutoChapterError",
    "auto_chapter",
    "GenerateError",
    "beats_to_curve",
    "classify_modes",
    "export_funscript",
    "generate_from_beats",
    "shape_curve",
    "Stanza",
    "classify_stanzas",
    "classify_stanzas_from_funscript",
    "ETAEstimator",
    "OnProgress",
    "ProgressReporter",
    "StageEvent",
    "adapt_string_callback",
    "SidecarError",
    "chapters_from_sidecar",
    "read_sidecar",
    "sidecar_path_for",
    "write_sidecar",
]
