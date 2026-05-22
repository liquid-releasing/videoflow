"""Audio beats sidecar — discrete beat times for MediaViewer overlays.

A lightweight companion to the peaks + spectrogram sidecars. Where peaks
gives you "loud or quiet" and spectrogram gives you "what frequencies
are present", beats gives you "exactly when the rhythm hits". Used as a
tick overlay on the Audio waveform (and later as a snap target for
action placement in the editor).

The beat data already exists in :class:`videoflow.audio.AudioBeatMap`,
which is computed during the ``beats`` stage of
:func:`videoflow.structural.auto_chapter`. This module just writes that
data out as a standalone sidecar so the MediaViewer's React side can
read it without parsing the (much larger) chapters.json structural
payload.

Sidecar shape (``<stem>.beats.json``):
    {
      "version": "1.0",
      "duration_ms": int,
      "bpm": float,
      "beats_ms": [int, ...],
      "downbeats_ms": [int, ...],
      "generated_by": {...}
    }

Pipeline integration: ``structural.auto_chapter`` calls
:func:`write_sidecar_from_beat_map` with the AudioBeatMap it already
built — no extra analysis. If the chapter analysis pass didn't run yet,
the sidecar isn't present and the MediaViewer simply renders no ticks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SIDECAR_SUFFIX = ".beats.json"
SIDECAR_VERSION = "1.0"


def sidecar_path(media_path: str | Path) -> str:
    """Canonical sidecar path for the given media file.

    Lives inside the per-project ``.<stem>.forge/`` directory (see
    :func:`videoflow.sidecar.forge_dir`).
    """
    from videoflow.sidecar import forge_dir
    p = Path(media_path)
    return str(forge_dir(p) / f"{p.stem}{SIDECAR_SUFFIX}")


def load_sidecar(media_path: str | Path) -> dict[str, Any] | None:
    """Return the cached sidecar dict, or None when absent / unparseable."""
    sp = Path(sidecar_path(media_path))
    if not sp.exists():
        return None
    try:
        with open(sp) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_sidecar(media_path: str | Path, data: dict[str, Any]) -> str:
    """Write *data* to ``<stem>.beats.json`` next to the media file."""
    sp = sidecar_path(media_path)
    Path(sp).parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return sp


def write_sidecar_from_beat_map(media_path: str | Path, beat_map) -> str:
    """Build the sidecar dict from a :class:`videoflow.audio.AudioBeatMap`
    and write it. The beat map carries everything we need — bpm, beats
    (in ms), and downbeats (in ms). Duration is the last beat's time;
    consumers fall back to the project's total duration when ms beyond
    the last beat are scrubbed.
    """
    beats_ms = list(beat_map.beats) if beat_map.beats else []
    downbeats_ms = list(beat_map.downbeats) if beat_map.downbeats else []
    duration_ms = (
        max(beats_ms[-1] if beats_ms else 0, downbeats_ms[-1] if downbeats_ms else 0)
    )
    data = {
        "version": SIDECAR_VERSION,
        "duration_ms": int(duration_ms),
        "bpm": float(beat_map.bpm),
        "beats_ms": [int(b) for b in beats_ms],
        "downbeats_ms": [int(b) for b in downbeats_ms],
        "generated_by": {
            "tool": "videoflow.audio_beats",
            "method": "librosa.beat",
        },
    }
    return write_sidecar(media_path, data)
