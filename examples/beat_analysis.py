"""beat_analysis.py — analyse beats, save, reload, snap to beat.

Demonstrates:
  - analyze_beats() — one call returns the full AudioBeatMap
  - save() / load() — skip re-analysis on repeat runs
  - Exploring bpm, beats, downbeats, phrases, energy
  - nearest_beat() — snap a time to the nearest beat
  - beats_in_range() — find beats in a window

Usage:
    python examples/beat_analysis.py <audio_or_video_file>

Example:
    python examples/beat_analysis.py music.mp3
    python examples/beat_analysis.py footage/with_audio.mp4

Requires librosa:
    pip install "videoflow[audio]"
"""

import sys
from pathlib import Path

from videoflow.audio import AudioBeatMap, analyze_beats


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    json_path  = input_path.with_suffix(".beats.json")

    # Re-use a saved beat map if one exists; otherwise analyse and save
    if json_path.exists():
        print(f"Loading saved beat map from {json_path}")
        beat_map = AudioBeatMap.load(json_path)
    else:
        print(f"Analysing {input_path} …")
        beat_map = analyze_beats(input_path)
        beat_map.save(json_path)
        print(f"Saved → {json_path}")

    # Basic stats
    print(f"\n{'─' * 40}")
    print(f"BPM:           {beat_map.bpm:.2f}")
    print(f"Duration:      {beat_map.duration_ms / 1000:.1f}s")
    print(f"Beat interval: {beat_map.beat_interval_ms:.0f}ms")
    print(f"Beats:         {len(beat_map.beats)}")
    print(f"Downbeats:     {len(beat_map.downbeats)}")
    print(f"Phrases:       {len(beat_map.phrases)}")

    # First 8 beats
    print(f"\nFirst 8 beat timestamps (ms):")
    for i, b in enumerate(beat_map.beats[:8]):
        marker = " ← downbeat" if b in beat_map.downbeats else ""
        print(f"  {i+1:>2}  {b:>7}ms{marker}")

    # Phrases
    print(f"\nPhrases (first 5):")
    for i, (start, end) in enumerate(beat_map.phrases[:5]):
        print(f"  {i+1:>2}  {start/1000:>6.2f}s → {end/1000:>6.2f}s")

    # High-energy beats
    high_energy = [(b, e) for b, e in zip(beat_map.beats, beat_map.energy) if e > 0.85]
    print(f"\nHigh-energy beats (energy > 0.85): {len(high_energy)}")
    for b, e in high_energy[:5]:
        print(f"  {b:>7}ms  energy={e:.2f}")
    if len(high_energy) > 5:
        print(f"  … and {len(high_energy) - 5} more")

    # Beat snapping examples
    print(f"\nBeat snapping:")
    test_ms = beat_map.beats[4] + 120   # 120ms after beat 5
    print(f"  Input time:         {test_ms}ms")
    print(f"  nearest (default):  {beat_map.nearest_beat(test_ms)}ms")
    print(f"  nearest before:     {beat_map.nearest_beat(test_ms, direction='before')}ms")
    print(f"  nearest after:      {beat_map.nearest_beat(test_ms, direction='after')}ms")

    # Beats in the first 10 seconds
    window_beats = beat_map.beats_in_range(0, 10_000)
    print(f"\nBeats in first 10s: {len(window_beats)}")

    print(f"\n{'─' * 40}")
    print(f"Beat map saved at: {json_path}")
    print("Pass this to AudioBeatMap.load() to skip re-analysis on future runs.")


if __name__ == "__main__":
    main()
