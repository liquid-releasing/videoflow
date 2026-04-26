"""videoflow — composable audio + video pipeline for haptic content generation.

Audio path is the v0.1 product surface: analyze a track for beats and energy,
classify phrases, shape a motion curve, export a funscript. forgegen is the
downstream Streamlit UI on top of this.
"""

from videoflow.analysis import (
    DETECTOR_INFO,
    DetectorInfo,
    Scene,
    SceneError,
    detect_scenes,
)
from videoflow.audio import AudioBeatMap, BeatError, analyze_beats
from videoflow.generate import (
    GenerateError,
    beats_to_curve,
    classify_modes,
    export_funscript,
    generate_from_beats,
    shape_curve,
)

__all__ = [
    "DETECTOR_INFO",
    "DetectorInfo",
    "Scene",
    "SceneError",
    "detect_scenes",
    "AudioBeatMap",
    "BeatError",
    "analyze_beats",
    "GenerateError",
    "beats_to_curve",
    "classify_modes",
    "export_funscript",
    "generate_from_beats",
    "shape_curve",
]
