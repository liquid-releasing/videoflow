"""Audio peaks sidecar — the per-hop RMS envelope for MediaViewer Audio mode.

A pre-computed waveform: one RMS magnitude per hop, normalised to [0, 1].
Cheap to render (canvas2D bar chart) and cheap to keep in memory
(~4MB JSON for a 30-min track at 10ms hop). Pairs with the heavier
spectrogram sidecar (:mod:`videoflow.audio_spectrogram`) which carries
frequency texture.

This module is the **analysis** half of the feature. The **render** half
lives in forgemoment's `MediaViewer.jsx::WaveformCanvas`. They meet at
the sidecar shape documented below.

Sidecar shape (``<stem>.audio.json``):
    {
      "version": "1.0",
      "hop_ms": int,            # window size in ms (default 10)
      "duration_ms": int,
      "peaks": [float, ...],    # 0..1, length = duration_ms / hop_ms
      "peak_count": int,
      "generated_by": {"tool": "...", "method": "rms", "sample_rate": int}
    }

Pipeline integration: :func:`videoflow.structural.auto_chapter` calls
:func:`compute_sidecar_from_samples` directly with the already-loaded
audio array, so chapter / peaks / spectrogram analysis share one decode.
Don't call :func:`extract_sidecar` from inside auto_chapter — that would
re-decode and defeat the point.

History: this code was originally `funscriptforge.forge.audio_peaks`.
Moved upstream to videoflow 2026-05-21 so all forge apps (Beatflo,
forgeassembler, etc.) get the same sidecar pipeline, and so peaks are
built alongside chapters/spectrogram in one user-triggered pass instead
of via the lazy-load that burped video playback. funscriptforge's
module is now a thin re-export shim of this one.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from .atomic_write import write_json_atomic


SIDECAR_SUFFIX = ".audio.json"
# hop boundaries are exact from 1.1 — a 1.0 sidecar
# carries the +0.227% stretch and must be recomputed, not trusted.
SIDECAR_VERSION = "1.1"

# Default hop. 10ms ≈ 100 peaks/sec → 30min track ≈ 180000 peaks ≈ 4MB
# JSON. Fine enough to read individual beats once the canvas renderer
# lands; coarse enough to keep memory reasonable on long material.
DEFAULT_HOP_MS = 10
DEFAULT_SAMPLE_RATE = 22050


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
    """Write *data* to ``<stem>.audio.json`` next to the media file.

    Compact JSON (no indent) — peak arrays double in size when pretty-
    printed and the file is machine-consumed anyway.
    """
    sp = sidecar_path(media_path)
    write_json_atomic(sp, data, separators=(",", ":"))
    return sp


def compute_sidecar_from_samples(
    samples,
    *,
    sr: int = DEFAULT_SAMPLE_RATE,
    hop_ms: int = DEFAULT_HOP_MS,
) -> dict[str, Any] | None:
    """Compute per-hop RMS from already-loaded audio samples and return
    the sidecar dict ready for :func:`write_sidecar`.

    Peaks are RMS magnitudes normalised into [0, 1] against the global
    max. Normalising per-track (rather than per-hop) preserves dynamic
    range — quiet passages read as quiet, not "loudest local sound."

    Returns None when audio is too short for the requested hop.
    """
    if hop_ms < 1:
        raise ValueError(f"hop_ms must be >= 1, got {hop_ms}")
    try:
        import numpy as np
    except ImportError:
        warnings.warn("audio-peaks requires numpy. Install with: pip install numpy")
        return None

    if samples is None or len(samples) == 0:
        warnings.warn("audio-peaks: no samples to analyze")
        return None

    # Hop boundaries come from the EXACT fractional hop length, not a
    # rounded one. At the default 22050 Hz a 10ms hop is 220.5 samples;
    # rounding that to 220 and still calling each hop "10ms" stretched the
    # whole timeline by 220.5/220 = +0.227%. On an hour of audio that is
    # +8.2s, so the waveform under the playhead drifted steadily further
    # ahead of the picture — measured on a real 60:12 scene whose sidecar
    # claimed 3621.05s (2026-09-21).
    #
    # Alternating 220/221-sample frames average exactly 220.5, so peaks[i]
    # really does cover [i*hop_ms, (i+1)*hop_ms) and duration_ms is true.
    hop_exact = sr * hop_ms / 1000.0
    if hop_exact < 1.0:
        hop_exact = 1.0  # sub-sample hop: one sample per hop is the floor
    n_hops = int(len(samples) / hop_exact)
    if n_hops == 0:
        warnings.warn(f"audio-peaks: audio too short for hop_ms={hop_ms}")
        return None
    edges = np.rint(np.arange(n_hops + 1) * hop_exact).astype(np.int64)
    trimmed = samples[: int(edges[-1])].astype(np.float32, copy=False)

    # reduceat rather than reshape: frames are no longer a constant width.
    # It also avoids materialising an (n_hops, hop) view of a multi-GB track.
    squared = trimmed * trimmed
    sums = np.add.reduceat(squared, edges[:-1])
    counts = np.diff(edges).astype(np.float32)
    rms = np.sqrt(sums / counts)
    peak_max = float(np.max(rms))
    if peak_max > 0:
        norm = (rms / peak_max).astype(np.float32, copy=False)
    else:
        norm = rms
    peaks = [round(float(v), 4) for v in norm]
    duration_ms = int(round(n_hops * hop_ms))

    return {
        "version": SIDECAR_VERSION,
        "hop_ms": int(hop_ms),
        "duration_ms": duration_ms,
        "peaks": peaks,
        "peak_count": len(peaks),
        "generated_by": {
            "tool": "videoflow.audio_peaks",
            "method": "rms",
            "sample_rate": sr,
        },
    }


def extract_sidecar(
    media_path: str | Path,
    *,
    sr: int = DEFAULT_SAMPLE_RATE,
    hop_ms: int = DEFAULT_HOP_MS,
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
            "audio-peaks requires librosa and numpy. "
            'Install with: pip install "videoflow[audio]"'
        )
        return None

    try:
        samples, _ = librosa.load(str(media_path), sr=sr, mono=True)
    except Exception as exc:
        warnings.warn(f"audio-peaks: could not load audio from {media_path!r}: {exc}")
        return None
    if samples is None or samples.size == 0:
        warnings.warn(f"audio-peaks: no audio in {media_path!r}")
        return None

    return compute_sidecar_from_samples(
        samples.astype(np.float32, copy=False),
        sr=sr,
        hop_ms=hop_ms,
    )


# ─── Legacy decode/compute split (kept for funscriptforge cli.py compat) ──
# The original funscriptforge cli.py audio-peaks command splits decode
# and RMS into two stages so it can emit per-stage progress events
# between them. The re-export shim in funscriptforge.forge.audio_peaks
# uses these names; keeping them at the same module path here makes the
# shim a one-line re-export.


def decode_audio(media_path: str | Path, sr: int = DEFAULT_SAMPLE_RATE):
    """Decode mono float32 samples at *sr* Hz. Returns None on failure."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        warnings.warn(
            "audio-peaks requires librosa and numpy. "
            'Install with: pip install "videoflow[audio]"'
        )
        return None

    try:
        samples, _ = librosa.load(str(media_path), sr=sr, mono=True)
    except Exception as exc:
        warnings.warn(f"audio-peaks: could not load audio from {media_path!r}: {exc}")
        return None
    if samples is None or samples.size == 0:
        warnings.warn(f"audio-peaks: no audio in {media_path!r}")
        return None
    return samples.astype(np.float32, copy=False)


def compute_peaks(samples, hop_ms: int = DEFAULT_HOP_MS, sr: int = DEFAULT_SAMPLE_RATE):
    """Alias for :func:`compute_sidecar_from_samples`. Kept under the
    legacy name for the funscriptforge cli.py audio-peaks command."""
    return compute_sidecar_from_samples(samples, sr=sr, hop_ms=hop_ms)


def extract_peaks(
    media_path: str | Path,
    hop_ms: int = DEFAULT_HOP_MS,
    sr: int = DEFAULT_SAMPLE_RATE,
):
    """Alias for :func:`extract_sidecar`. Kept under the legacy name."""
    return extract_sidecar(media_path, sr=sr, hop_ms=hop_ms)


def load_peaks(media_path: str | Path):
    """Alias for :func:`load_sidecar`. Kept under the legacy name."""
    return load_sidecar(media_path)
