"""Chapter resolution — find chapters for a media file from any available source.

Every forge tool that loads a track wants to ask the same question —
*"are there chapters here, and where do I find them?"* This module
centralises that lookup so forgegen / forgevents / FunscriptForge Pro /
ForgeAssembler / ForgePlayer all see the same answer.

Priority order (locked 2026-04-27):

1. **Embedded mp4 chapters** — ``ffprobe -show_chapters`` on the source
   media. Honours chapter markers authored by external tools.
2. **Sidecar JSON** — ``<stem>.chapters.json`` next to the source.
3. **Analysis JSON** — ``<stem>.analysis.json`` produced by forgegen,
   reading either ``metadata.chapters`` (authored) or ``chapter_proposals``
   (auto-detected). The proposals shape is mapped to :class:`Chapter`
   by treating ``intent_proposal`` as the intent.

If none of the sources has chapters, :func:`load_chapters` returns
``None`` so the caller can decide what to do (v0.0.5+ runs auto-detection;
v0.0.4 generates without chapter biasing).

This module does **not** auto-detect. Auto-detection is v0.0.5+ scope
and lives in ``videoflow.structural``.

Example::

    from videoflow.chapters import load_chapters

    chapters = load_chapters("track.mp4")
    if chapters is None:
        # No chapters anywhere — generate without biasing or run
        # auto-detection (v0.0.5+).
        ...
    else:
        for ch in chapters:
            print(f"{ch.at_ms}ms  {ch.intent}  {ch.name}")
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class ChapterError(RuntimeError):
    """Raised when chapter resolution fails."""


@dataclasses.dataclass
class Chapter:
    """One chapter on a track's timeline.

    Attributes:
        at_ms: Start time in milliseconds.
        end_ms: End time in milliseconds. ``None`` means "until the next
            chapter or end of track" — common for sources that only
            store start times (mp4 chapters, funscript ``metadata.chapters``).
        name: Optional human label (e.g. ``"build 1"``). Defaults to
            empty string.
        intent: Optional intent label from the chapter intent vocabulary
            (e.g. ``"intro"``, ``"build"``, ``"sustain"``, ``"edge"``,
            ``"climax"``, ``"recover"``, ``"outro"``). Open string in
            v0.0.4; v0.0.5 locks the 7-element vocabulary alongside
            chapter intent biasing.
    """

    at_ms: int
    end_ms: int | None = None
    name: str = ""
    intent: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Omits ``end_ms`` when ``None`` so author-shape chapters stay
        compact.
        """
        out: dict = {
            "at_ms": self.at_ms,
            "name": self.name,
            "intent": self.intent,
        }
        if self.end_ms is not None:
            out["end_ms"] = self.end_ms
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Chapter":
        """Parse a chapter dict.

        Accepts both the *authored* shape (``at_ms`` or legacy ``at``)
        and the analysis-schema *proposal* shape (``start_ms`` /
        ``end_ms`` / ``intent_proposal``). The first present field wins.

        Raises:
            ChapterError: If neither ``at_ms``, ``at``, nor ``start_ms``
                is present.
        """
        if "at_ms" in data:
            at_ms = int(data["at_ms"])
        elif "start_ms" in data:
            at_ms = int(data["start_ms"])
        elif "at" in data:
            at_ms = int(data["at"])
        else:
            raise ChapterError(
                f"chapter missing 'at_ms' / 'at' / 'start_ms': {data!r}"
            )

        end_ms = data.get("end_ms")
        if end_ms is not None:
            end_ms = int(end_ms)

        intent = str(data.get("intent") or data.get("intent_proposal") or "")
        name = str(data.get("name") or "")
        return cls(at_ms=at_ms, end_ms=end_ms, name=name, intent=intent)


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def _find_ffprobe() -> str:
    """Locate ffprobe — PATH first, then alongside videoflow's package."""
    pkg_dir = Path(__file__).parent
    for candidate in (pkg_dir / "ffprobe.exe", pkg_dir / "ffprobe"):
        if candidate.is_file():
            return str(candidate)
    return "ffprobe"


def read_mp4_chapters(media_path: str | Path) -> list[Chapter] | None:
    """Read embedded chapter markers from a video file via ``ffprobe``.

    Returns ``None`` if the file is not a recognised video format, if
    ffprobe is not available, or if the file has no chapter markers.
    Returns a list (possibly empty in pathological cases) when ffprobe
    successfully ran.

    Args:
        media_path: Path to the source media.

    Returns:
        List of :class:`Chapter`, or ``None`` if no chapters / ffprobe
        unavailable.
    """
    path = Path(media_path)
    if not path.exists():
        return None
    if path.suffix.lower() not in _VIDEO_SUFFIXES:
        return None

    ffprobe = _find_ffprobe()
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_chapters", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            creationflags=_NO_WINDOW,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    raw_chapters = data.get("chapters") or []
    if not raw_chapters:
        return None

    chapters: list[Chapter] = []
    for ch in raw_chapters:
        # ffprobe emits start in seconds (float string) and start_time / end_time
        # plus tags{title}.
        try:
            start_s = float(ch.get("start_time", 0))
            end_s = float(ch["end_time"]) if "end_time" in ch else None
        except (TypeError, ValueError):
            continue
        tags = ch.get("tags") or {}
        chapters.append(Chapter(
            at_ms=int(round(start_s * 1000)),
            end_ms=int(round(end_s * 1000)) if end_s is not None else None,
            name=str(tags.get("title") or ""),
            intent="",  # mp4 doesn't carry our intent vocabulary
        ))
    return chapters


def read_sidecar_chapters(media_path: str | Path) -> list[Chapter] | None:
    """Read chapters from a ``<stem>.chapters.json`` sidecar.

    The sidecar shape is::

        {
          "version": "1.0",
          "chapters": [
            {"at_ms": 0, "name": "intro", "intent": "intro"},
            {"at_ms": 90000, "name": "build", "intent": "build"}
          ]
        }

    A bare list is also accepted. Returns ``None`` when the sidecar does
    not exist; an empty list when the sidecar exists but lists no
    chapters (caller treats both the same in practice but the distinction
    is preserved).

    Raises:
        ChapterError: If the sidecar exists but is unreadable JSON or
            contains malformed chapter records.
    """
    path = Path(media_path)
    sidecar = path.with_suffix(path.suffix + ".chapters.json")
    if not sidecar.exists():
        # Also check the more common "<stem>.chapters.json" form
        # (e.g. track.mp4 → track.chapters.json) in case the suffix-
        # appending convention isn't used.
        alt = path.with_name(path.stem + ".chapters.json")
        if alt.exists():
            sidecar = alt
        else:
            return None

    try:
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChapterError(f"sidecar {sidecar} is not valid JSON: {exc}") from exc

    if isinstance(doc, list):
        raw = doc
    elif isinstance(doc, dict):
        raw = doc.get("chapters", [])
    else:
        raise ChapterError(
            f"sidecar {sidecar} must be a list or object, got {type(doc).__name__}"
        )

    if not isinstance(raw, list):
        raise ChapterError(
            f"sidecar {sidecar} 'chapters' field must be a list"
        )

    return [Chapter.from_dict(c) for c in raw]


def read_analysis_chapters(media_path: str | Path) -> list[Chapter] | None:
    """Read chapters from a ``<stem>.analysis.json`` produced by forgegen.

    Looks for chapters in two places, in order:

    1. ``metadata.chapters`` — confirmed / authored chapters
    2. ``chapter_proposals`` — auto-detected proposals (mapped to
       :class:`Chapter` with ``intent_proposal`` becoming ``intent``)

    Returns ``None`` if no analysis JSON exists or it has no chapters
    in either location.

    Raises:
        ChapterError: If the file exists but is unreadable JSON or
            contains malformed chapters.
    """
    path = Path(media_path)
    analysis = path.with_suffix(path.suffix + ".analysis.json")
    if not analysis.exists():
        alt = path.with_name(path.stem + ".analysis.json")
        if alt.exists():
            analysis = alt
        else:
            return None

    try:
        doc = json.loads(analysis.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChapterError(
            f"analysis {analysis} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(doc, dict):
        raise ChapterError(
            f"analysis {analysis} must be an object, got {type(doc).__name__}"
        )

    # Authored / confirmed chapters take priority over proposals.
    metadata = doc.get("metadata") or {}
    confirmed = metadata.get("chapters")
    if confirmed:
        return [Chapter.from_dict(c) for c in confirmed]

    proposals = doc.get("chapter_proposals")
    if proposals:
        return [Chapter.from_dict(c) for c in proposals]

    return None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def load_chapters(media_path: str | Path) -> list[Chapter] | None:
    """Resolve chapters for *media_path* from the highest-priority source.

    Priority:

    1. Embedded mp4 chapters (``ffprobe -show_chapters``)
    2. ``<stem>.chapters.json`` sidecar
    3. ``<stem>.analysis.json`` (``metadata.chapters`` or ``chapter_proposals``)

    Returns ``None`` if no source had chapters. Returns the chapter
    list from the first source that did, even if that list is empty.

    The resolver does not run auto-detection — that's
    :mod:`videoflow.structural` (v0.0.5+). Callers can decide what to
    do when this returns ``None``: skip chapter biasing, or run
    auto-detection.

    Raises:
        FileNotFoundError: If *media_path* itself does not exist.
        ChapterError: If a sidecar / analysis JSON file exists but is
            malformed.
    """
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    mp4 = read_mp4_chapters(path)
    if mp4 is not None:
        return mp4

    sidecar = read_sidecar_chapters(path)
    if sidecar is not None:
        return sidecar

    analysis = read_analysis_chapters(path)
    if analysis is not None:
        return analysis

    return None
