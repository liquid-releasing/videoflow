"""Mel-spectrogram sidecar — the frequency-over-time view for MediaViewer.

A second audio visualization alongside `audio.analyze_beats` peaks. Where
peaks compress a hop down to one RMS magnitude, the spectrogram preserves
frequency texture: bass band energy reveals song-section boundaries,
vocal-band turn-on marks a new chapter start, percussive vs sustained
content reads at a glance. Particularly useful for Chapters-tab split-
finding workflows in FunscriptForge; future Beatflo / forgeassembler
consumers get it for free since the renderer lives in forgemoment.

This module is the **analysis** half of the feature. The **render** half
lives in forgemoment's `MediaViewer.jsx::SpectrogramCanvas`. They meet at
the sidecar shape documented below.

Sidecar shape (``<stem>.spectrogram.json``):
    {
      "version": "1.0",
      "hop_ms": int,            # frame hop in ms (derived from sr / hop_length)
      "n_mels": int,            # number of mel frequency bins
      "n_frames": int,          # number of time frames in the spectrogram
      "duration_ms": int,       # n_frames * hop_ms
      "fmax": int,              # max mel frequency in Hz
      "db_floor": float,        # cells=0 maps to this dB
      "db_ceiling": float,      # cells=255 maps to this dB (typically 0)
      "cells_b64": "<base64>",  # uint8 Uint8Array[n_frames * n_mels], time-major:
                                # cells[t * n_mels + bin]. Each byte 0..255
                                # is the quantized dB intensity that indexes
                                # directly into the magma colormap LUT.
      "generated_by": {...}
    }

Storage rationale: a 30-min track at hop_ms≈23 and n_mels=64 yields ~5M
cells. JSON int array would be ~15MB ASCII and a slow parse; base64
encoded uint8 bytes are ~6.7MB ASCII and avoid the per-cell number-parse
cost. The viewer holds the bytes as a Uint8Array, so we also avoid the
8-byte-per-number heap inflation that a Vec<u8> → JSON array → JS array
of Numbers would cause.

Pipeline integration: `videoflow.structural.auto_chapter` calls
:func:`compute_sidecar_from_samples` directly with the already-loaded
audio array, so chapter analysis and spectrogram building share one
decode. Don't call :func:`extract_sidecar` from inside auto_chapter —
that would re-decode and defeat the point.
"""

from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path
from typing import Any


SIDECAR_SUFFIX = ".spectrogram.json"
SIDECAR_VERSION = "1.0"

# dB floor for the quantization mapping. librosa's power_to_db(ref=np.max)
# returns 0 dB at the loudest cell and trends negative for quieter cells.
# -80 dB is the librosa convention for "effectively silent" — anything
# quieter maps to byte 0 (darkest magma) and contributes no visible
# signal. Going further down (e.g. -120) wastes the colormap on noise.
DEFAULT_DB_FLOOR = -80.0
DEFAULT_DB_CEILING = 0.0

# Default mel parameters. n_mels=64 reads cleanly at typical viewer
# heights (one bin per ~2 pixels at 120px tall). hop_length=512 at sr=22050
# ≈ 23ms hop — fine enough to resolve song-section transitions and beats,
# coarse enough to keep the matrix small. fmax=8000 covers vocal and
# percussion content; higher rates rarely add signal for music sources.
DEFAULT_N_MELS = 64
DEFAULT_HOP_LENGTH = 512
DEFAULT_FMAX = 8000


def sidecar_path(media_path: str | Path) -> str:
    """Canonical sidecar path for the given media file. Mirrors the
    audio_peaks sidecar convention — replace the suffix, write next to
    the media file."""
    return str(Path(media_path).with_suffix(SIDECAR_SUFFIX))


def load_sidecar(media_path: str | Path) -> dict[str, Any] | None:
    """Return the cached sidecar dict, or None when absent / unparseable.

    Treats parse errors as absence so the viewer's empty-state path
    handles both cases identically ("sidecar not built yet — run chapter
    analysis to enable spectrogram view").
    """
    sp = Path(sidecar_path(media_path))
    if not sp.exists():
        return None
    try:
        with open(sp) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_sidecar(media_path: str | Path, data: dict[str, Any]) -> str:
    """Write *data* to ``<stem>.spectrogram.json`` next to the media file.

    Compact JSON (no indent) — the cells_b64 string dominates the file
    size and isn't helped by indentation; the small surrounding metadata
    is fine without it. Returns the written sidecar path.
    """
    sp = sidecar_path(media_path)
    Path(sp).parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return sp


def compute_sidecar_from_samples(
    samples,
    *,
    sr: int = 22050,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmax: int = DEFAULT_FMAX,
    db_floor: float = DEFAULT_DB_FLOOR,
    db_ceiling: float = DEFAULT_DB_CEILING,
) -> dict[str, Any] | None:
    """Compute the mel spectrogram from already-loaded audio samples and
    return the sidecar dict ready for :func:`write_sidecar`.

    This is the integration entry point for ``structural.auto_chapter``:
    audio is decoded once in the ``load`` stage, then the same samples
    flow into beat / chapter / spectrogram analysis. Returns None when
    the audio is too short to yield a single frame (degenerate input)
    or when librosa / numpy aren't installed.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        warnings.warn(
            "audio-spectrogram requires librosa and numpy. "
            'Install with: pip install "videoflow[audio]"'
        )
        return None

    if samples is None or len(samples) == 0:
        warnings.warn("audio-spectrogram: no samples to analyze")
        return None

    db_span = db_ceiling - db_floor
    if db_span <= 0:
        warnings.warn(
            f"audio-spectrogram: invalid db range "
            f"[{db_floor}, {db_ceiling}]"
        )
        return None

    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32, copy=False),
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        fmax=fmax,
    )
    if mel.shape[1] == 0:
        warnings.warn("audio-spectrogram: audio too short for one frame")
        return None

    # Silent-input guard. `power_to_db(ref=np.max)` is a *relative*
    # normalization — every cell is mapped relative to the loudest
    # cell in the track. When the track is fully silent the loudest
    # cell is essentially zero, and the resulting "every cell is 0 dB"
    # gets clipped+quantized to byte 255 (visually: a white spectrogram
    # for a silent track, which is misleading). Short-circuit to all-
    # zero cells so silence reads as black.
    max_power = float(np.max(mel))
    if max_power < 1e-12:
        quantized = np.zeros_like(mel, dtype=np.uint8)
    else:
        # shape (n_mels, n_frames)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Clip to display range and linearly map [db_floor, db_ceiling] →
        # [0, 255]. The viewer's magma LUT is indexed directly by this byte,
        # so the quantization happens here once and the render path stays
        # cheap (one LUT lookup per cell, no float math at paint time).
        clipped = np.clip(mel_db, db_floor, db_ceiling)
        normalized = (clipped - db_floor) / db_span
        quantized = (normalized * 255.0 + 0.5).astype(np.uint8)

    # Reorient to time-major so a contiguous slice [t0..t1] is one chunk
    # of the byte stream. The canvas reads visible windows as
    # cells[t * n_mels + bin], and a Uint8Array.subarray over that
    # range is zero-copy.
    time_major = quantized.T.copy(order="C")  # (n_frames, n_mels), contiguous
    n_frames, n_mels_out = time_major.shape

    hop_ms = int(round(hop_length * 1000.0 / sr))
    duration_ms = int(n_frames * hop_ms)

    cells_b64 = base64.b64encode(time_major.tobytes()).decode("ascii")

    return {
        "version": SIDECAR_VERSION,
        "hop_ms": hop_ms,
        "n_mels": int(n_mels_out),
        "n_frames": int(n_frames),
        "duration_ms": duration_ms,
        "fmax": int(fmax),
        "db_floor": float(db_floor),
        "db_ceiling": float(db_ceiling),
        "cells_b64": cells_b64,
        "generated_by": {
            "tool": "videoflow.audio_spectrogram",
            "method": "mel_power_to_db",
            "sample_rate": sr,
            "hop_length": hop_length,
        },
    }


def extract_sidecar(
    media_path: str | Path,
    *,
    sr: int = 22050,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmax: int = DEFAULT_FMAX,
) -> dict[str, Any] | None:
    """Standalone path: decode + compute in one call. Intended for direct
    CLI use or ad-hoc analysis. **Don't call this from inside auto_chapter**
    — that would re-decode the audio. Use :func:`compute_sidecar_from_samples`
    with the already-loaded ``y`` instead.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        warnings.warn(
            "audio-spectrogram requires librosa and numpy. "
            'Install with: pip install "videoflow[audio]"'
        )
        return None

    try:
        samples, _ = librosa.load(str(media_path), sr=sr, mono=True)
    except Exception as exc:
        warnings.warn(f"audio-spectrogram: could not load audio: {exc}")
        return None
    if samples is None or samples.size == 0:
        warnings.warn(f"audio-spectrogram: no audio in {media_path!r}")
        return None

    return compute_sidecar_from_samples(
        samples.astype(np.float32, copy=False),
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        fmax=fmax,
    )
