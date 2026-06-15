"""Chapter resolution — find chapters for a media file from any available source.

Every forge tool that loads a track wants to ask the same question —
*"are there chapters here, and where do I find them?"* This module
centralises that lookup so forgegen / forgevents / FunscriptForge Pro /
ForgeAssembler / ForgePlayer all see the same answer.

Priority order (revised 2026-05-14; was the inverse pre-revision):

1. **Sidecar JSON** — ``<stem>.chapters.json`` next to the source. The
   sidecar is the editable, authoritative store; if the user has
   touched chapters anywhere in the toolchain, they live here. Wins
   over embedded markers because the sidecar reflects later edits.
2. **Embedded mp4 chapters** — ``ffprobe -show_chapters`` on the source
   media. Honoured when no sidecar exists; common path for the
   "imported from Movavi / external authoring tool" workflow when the
   recovery hasn't yet written a sidecar.
3. **Analysis JSON** — ``<stem>.analysis.json`` produced by forgegen,
   reading either ``metadata.chapters`` (authored) or ``chapter_proposals``
   (auto-detected). The proposals shape is mapped to :class:`Chapter`
   by treating ``intent_proposal`` as the intent.

The principle (articulated by user 2026-05-07): **hand-built chapters
are authoritative. Detection is the fallback, not the default.** The
sidecar is where hand-built data accumulates as it flows through the
toolchain — so the sidecar wins.

If none of the sources has chapters, :func:`load_chapters` returns
``None`` so the caller can decide what to do (v0.0.5+ runs auto-detection;
v0.0.4 generates without chapter biasing).

This module does **not** auto-detect. Auto-detection is v0.0.5+ scope
and lives in ``videoflow.structural``.

Writers:

- :func:`write_chapters_sidecar` — author the sidecar from a chapter
  list (the "Movavi recovery" workflow: user has timestamps from an
  external tool, wants them written to a forgegen-readable sidecar).
  Marks records ``auto_generated: false`` so the merge layer in
  :mod:`videoflow.sidecar` treats them as LATCHED.
- :func:`embed_in_mp4` — mux chapters into an MP4 via FFMETADATA1 +
  ``ffmpeg -codec copy`` (no re-encode). Used to round-trip hand-
  authored chapters back into the file itself so other tools that
  only read embedded markers (e.g. some players) see them.

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

# Legacy → new texture-label mapping for chapter sidecars written by
# pre-rename builds (before 2026-05-24). Applied on read in
# :meth:`Chapter.from_dict` so old chapters.json files render under
# the new vocabulary without rewriting the file. structural.py also
# exports this for consumers that want the same mapping.
_LEGACY_CONTENT_TYPE_MAP = {
    "music":   "driving",
    "ambient": "calm",
    "mixed":   "varied",
}


class ChapterError(RuntimeError):
    """Raised when chapter resolution fails."""


@dataclasses.dataclass
class Chapter:
    """One chapter on a track's timeline.

    A chapter serves two roles in the lqr toolchain — generation
    segmentation (forgegen uses it as an analysis chunk boundary) and
    navigation waypoint (forgeplayer / forgeassembler / FunscriptForge
    use it as a section the user navigates / cuts / overlays). The
    same record carries fields for both; consumers read what they need.
    See ``videoflow/docs/architecture/audio-structure-primitive.md``.

    Attributes:
        at_ms: Start time in milliseconds.
        end_ms: End time in milliseconds. ``None`` means "until the next
            chapter or end of track" — common for sources that only
            store start times (mp4 chapters, funscript ``metadata.chapters``).
        name: Optional human label (e.g. ``"build 1"``). Defaults to
            empty string. Navigation-flavored.
        intent: Optional intent label from the chapter intent vocabulary
            (e.g. ``"intro"``, ``"build"``, ``"sustain"``, ``"edge"``,
            ``"climax"``, ``"recover"``, ``"outro"``). Open string in
            v0.0.4; v0.0.5 locks the 7-element vocabulary alongside
            chapter intent biasing. Navigation-flavored.
        content_type: Audio *texture* label from
            :mod:`videoflow.structural` auto-detection — one of
            ``"driving"``, ``"calm"``, ``"varied"``, or ``""`` when not
            classified. Renamed 2026-05-24 (was ``"music"`` /
            ``"ambient"`` / ``"mixed"``); legacy values are mapped
            on read by :meth:`from_dict`. Generation-flavored.
        voice_label: Voice-presence label from the Silero VAD pass —
            ``"talk"`` when the chapter is dominated by speech, ``""``
            otherwise. Round 2 will subdivide into ``"sing"`` /
            ``"react"`` via pitch-stability heuristics. Independent
            axis from ``content_type``: a chapter can be (talk · calm)
            or (talk · driving) or just (calm). Generation-flavored.
        confidence: Detection confidence in ``[0.0, 1.0]`` from
            auto-detection, or ``None`` when authored / unknown.
            Generation-flavored.
        evidence: Identifiers of the analysis features that produced
            this chapter — e.g. ``["recurrence_matrix", "silence",
            "rms_variance"]``. ``"voice_vad"`` is added when Silero
            detected speech. Empty list for authored chapters.
            Generation-flavored.
    """

    at_ms: int
    end_ms: int | None = None
    name: str = ""
    intent: str = ""
    content_type: str = ""
    voice_label: str = ""
    confidence: float | None = None
    evidence: list[str] = dataclasses.field(default_factory=list)
    # User-chosen tone id (FunscriptForge Chapters tab: tender/build/tease/
    # edge/climax/dominant/tame, or "none" for Untoned). Empty until the
    # user accepts a tone; persisted so the choice survives reopen rather
    # than resetting to the analyzer suggestion.
    tone: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Omits ``end_ms`` when ``None`` and the analytical fields
        (``content_type`` / ``voice_label`` / ``confidence`` /
        ``evidence``) when at their defaults — authored chapters
        stay compact.
        """
        out: dict = {
            "at_ms": self.at_ms,
            "name": self.name,
            "intent": self.intent,
        }
        if self.end_ms is not None:
            out["end_ms"] = self.end_ms
        if self.content_type:
            out["content_type"] = self.content_type
        if self.voice_label:
            out["voice_label"] = self.voice_label
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.evidence:
            out["evidence"] = list(self.evidence)
        if self.tone:
            out["tone"] = self.tone
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Chapter":
        """Parse a chapter dict.

        Accepts both the *authored* shape (``at_ms`` or legacy ``at``)
        and the analysis-schema *proposal* shape (``start_ms`` /
        ``end_ms`` / ``intent_proposal``). The first present field wins.
        Analytical fields (``content_type``, ``confidence``, ``evidence``)
        are read when present, otherwise default.

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
        # Legacy migration: sidecars from before 2026-05-24 carry
        # "music" / "ambient" / "mixed". Map to the new texture vocab
        # on read so old chapters.json files render correctly under
        # the new labels; the file gets rewritten with new values on
        # the next analyze pass.
        content_type = str(data.get("content_type") or "")
        if content_type in _LEGACY_CONTENT_TYPE_MAP:
            content_type = _LEGACY_CONTENT_TYPE_MAP[content_type]
        voice_label = str(data.get("voice_label") or "")
        confidence = data.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
        evidence_raw = data.get("evidence") or []
        evidence = [str(e) for e in evidence_raw] if isinstance(evidence_raw, list) else []
        tone = str(data.get("tone") or "")
        return cls(
            at_ms=at_ms,
            end_ms=end_ms,
            name=name,
            intent=intent,
            content_type=content_type,
            voice_label=voice_label,
            confidence=confidence,
            evidence=evidence,
            tone=tone,
        )


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
    # Lazy import — sidecar.py imports Chapter from this module, so a
    # top-level import would close the cycle.
    from videoflow.sidecar import forge_dir

    path = Path(media_path)
    # Sidecars live inside the per-project hidden forge dir
    # (``<dir>/.<stem>.forge/<stem>.chapters.json``). Pre-forge-dir
    # sidecars in the source directory are ignored — re-Analyze
    # rebuilds them in the new location.
    sidecar = forge_dir(path) / f"{path.stem}.chapters.json"
    if not sidecar.exists():
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

    Priority (revised 2026-05-14):

    1. ``<stem>.chapters.json`` sidecar — the editable, authoritative
       store. Hand-built chapters live here.
    2. Embedded mp4 chapters (``ffprobe -show_chapters``) — honoured
       when no sidecar exists.
    3. ``<stem>.analysis.json`` (``metadata.chapters`` or ``chapter_proposals``)
       — forgegen-produced fallback.

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

    sidecar = read_sidecar_chapters(path)
    if sidecar is not None:
        return sidecar

    mp4 = read_mp4_chapters(path)
    if mp4 is not None:
        return mp4

    analysis = read_analysis_chapters(path)
    if analysis is not None:
        return analysis

    return None


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_chapters_sidecar(
    media_path: str | Path,
    chapters: list[Chapter],
    *,
    writer: str = "external",
    writer_version: str = "",
) -> Path:
    """Write *chapters* to ``<stem>.chapters.json`` next to *media_path*.

    Thin convenience over :func:`videoflow.sidecar.write_sidecar` for
    the "I have a chapter list and want it on disk" use case (Movavi
    timestamp recovery, hand-author from CLI, etc.). Each chapter is
    written with ``auto_generated: false`` so the sidecar merge layer
    treats it as LATCHED — subsequent ``mode="analyze"`` writers
    (e.g. ``videoflow.structural.auto_chapter``) leave it alone.

    Uses ``mode="edit"`` so the call counts as a user edit in the
    sidecar's provenance trail. If a sidecar already exists, the
    field-level merger in :mod:`videoflow.sidecar` preserves any
    non-chapter data (phrases / energy / etc.) and merges chapters
    by primary key.

    Args:
        media_path: Path to the source media.
        chapters: List of :class:`Chapter` records to persist.
        writer: Module / app id recorded in the sidecar's provenance
            block. Defaults to ``"external"`` for hand-authored
            workflows; CLI callers should pass ``"videoflow.cli"``.
        writer_version: Optional version string for provenance.

    Returns:
        The sidecar path that was written.

    Raises:
        videoflow.sidecar.SidecarError: If the existing sidecar is
            malformed or merge rules fail.
    """
    # Lazy import — sidecar.py imports Chapter from this module, so a
    # top-level import here would close the cycle. The lazy form lets
    # ``from videoflow.chapters import *`` work even when sidecar.py
    # is partially loaded.
    from videoflow.sidecar import write_sidecar as _write_sidecar

    payload: dict = {
        "chapters": [
            {**c.to_dict(), "auto_generated": False}
            for c in chapters
        ],
    }
    return _write_sidecar(
        media_path,
        payload,
        writer=writer,
        writer_version=writer_version,
        mode="edit",
    )


def format_youtube_description(chapters: list[Chapter]) -> str:
    """Render *chapters* as a YouTube-description chapter block.

    YouTube auto-extracts chapter markers from the video description if
    the text follows a specific shape (rules YouTube has documented but
    not stabilised in writing — current behaviour as of mid-2025):

    1. **First chapter must start at 0:00.** Anything else and YouTube
       refuses to treat the block as chapters at all.
    2. **At least 3 chapter lines.** Fewer than 3 and chapters don't
       show in the UI.
    3. **Each chapter at least 10 seconds long.** Sub-10s segments
       cause YouTube to drop the whole block.
    4. **Timestamps at line start**, formatted ``M:SS``, ``MM:SS``,
       ``H:MM:SS``, or ``HH:MM:SS``. We emit the shortest valid form
       per line (``M:SS`` under 1h, ``H:MM:SS`` at/above 1h) — both
       are accepted, but the shorter form reads more naturally.
    5. **Title follows after a space.** We prefer ``name`` and fall
       back to ``intent`` so chapters with neither still get a
       meaningful label (``"Chapter N"``).

    The returned string can be pasted directly into the YouTube
    description box, or appended programmatically to an existing
    description. Trailing newline included so concatenation with
    other text doesn't collapse the final line.

    Args:
        chapters: List of :class:`Chapter` records. Must satisfy the
            YouTube rules above.

    Returns:
        Multi-line string, one chapter per line, trailing newline.

    Raises:
        ChapterError: If the chapter list would be rejected by
            YouTube — first chapter not at 0, fewer than 3 chapters,
            or any chapter shorter than 10 seconds. The error message
            names the specific rule violated so the caller can fix.
    """
    if len(chapters) < 3:
        raise ChapterError(
            f"YouTube requires at least 3 chapters to render a chapter "
            f"block; got {len(chapters)}"
        )

    sorted_ch = sorted(chapters, key=lambda c: c.at_ms)
    if sorted_ch[0].at_ms != 0:
        raise ChapterError(
            f"YouTube requires the first chapter to start at 0:00; "
            f"got at_ms={sorted_ch[0].at_ms}. Add an intro chapter or "
            f"shift the first chapter back to 0."
        )

    # Validate ≥10s spacing. For chapters with open-ended end_ms we
    # compare against the next chapter's start; for the last chapter
    # we can't verify length here (no track duration) — YouTube does
    # that check on its end. Caller is responsible for ensuring the
    # last chapter is also ≥10s.
    for i in range(len(sorted_ch) - 1):
        gap = sorted_ch[i + 1].at_ms - sorted_ch[i].at_ms
        if gap < 10_000:
            raise ChapterError(
                f"YouTube requires each chapter to be at least 10 seconds "
                f"long; chapter {i} → {i + 1} is only {gap}ms "
                f"(at {sorted_ch[i].at_ms}ms → {sorted_ch[i + 1].at_ms}ms). "
                f"Merge or extend the short chapter."
            )

    lines: list[str] = []
    has_hour_chapter = sorted_ch[-1].at_ms >= 3_600_000
    for i, ch in enumerate(sorted_ch):
        timestamp = _format_youtube_timestamp(
            ch.at_ms, force_hours=has_hour_chapter,
        )
        title = ch.name or ch.intent or f"Chapter {i + 1}"
        lines.append(f"{timestamp} {title}")
    return "\n".join(lines) + "\n"


def _format_youtube_timestamp(at_ms: int, *, force_hours: bool) -> str:
    """Render *at_ms* as a YouTube-recognised timestamp.

    Without *force_hours*, returns ``M:SS`` for times under an hour
    and ``H:MM:SS`` otherwise. With *force_hours*, always returns
    ``H:MM:SS`` — used when the same chapter block has at least one
    chapter past the 1h mark, so the column reads consistently.
    """
    total_seconds = at_ms // 1000
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0 or force_hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_ffmetadata1(chapters: list[Chapter]) -> str:
    """Render *chapters* as an FFMETADATA1 document.

    FFMETADATA1 is the text format ffmpeg accepts via ``-i <file>`` +
    ``-map_metadata <stream>``. Each chapter block carries a
    ``TIMEBASE=1/1000`` declaration so START/END can be written in
    milliseconds directly (matches :class:`Chapter`'s ``at_ms`` /
    ``end_ms`` fields without a units conversion).

    For chapters with ``end_ms is None`` (open-ended "until next
    chapter"), END is set to the START of the next chapter; the last
    chapter's END is left as START + 1 (a placeholder that ffmpeg
    rejects if it equals START exactly). Callers wanting a clean
    final-chapter END should set it explicitly.
    """
    if not chapters:
        return ";FFMETADATA1\n"

    # Sort by start so the file matches reading order. Embedded
    # chapters are typically already sorted, but be defensive.
    sorted_ch = sorted(chapters, key=lambda c: c.at_ms)
    out: list[str] = [";FFMETADATA1\n"]
    for i, ch in enumerate(sorted_ch):
        end_ms = ch.end_ms
        if end_ms is None:
            # Fill in from the next chapter's start, or fall back to
            # a 1ms-after-start sentinel for the trailing chapter.
            if i + 1 < len(sorted_ch):
                end_ms = sorted_ch[i + 1].at_ms
            else:
                end_ms = ch.at_ms + 1
        # ffmpeg rejects START >= END; guard against degenerate input.
        if end_ms <= ch.at_ms:
            end_ms = ch.at_ms + 1

        out.append("[CHAPTER]\n")
        out.append("TIMEBASE=1/1000\n")
        out.append(f"START={ch.at_ms}\n")
        out.append(f"END={end_ms}\n")
        # Prefer the human label; fall back to intent so the chapter
        # at least has SOMETHING in players that show titles.
        title = ch.name or ch.intent
        if title:
            # FFMETADATA1 escapes \, =, ;, #, and \n with a backslash.
            escaped = (
                title.replace("\\", "\\\\")
                .replace("=", "\\=")
                .replace(";", "\\;")
                .replace("#", "\\#")
                .replace("\n", "\\\n")
            )
            out.append(f"title={escaped}\n")
    return "".join(out)


def _find_ffmpeg() -> str:
    """Locate ffmpeg — PATH first, then alongside videoflow's package.

    Mirrors :func:`_find_ffprobe` and the lookup pattern in
    :mod:`videoflow.audio` so all videoflow callers find the same
    binary regardless of entry point.
    """
    pkg_dir = Path(__file__).parent
    for candidate in (pkg_dir / "ffmpeg.exe", pkg_dir / "ffmpeg"):
        if candidate.is_file():
            return str(candidate)
    return "ffmpeg"


def embed_in_mp4(
    input_path: str | Path,
    output_path: str | Path,
    chapters: list[Chapter],
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Mux *chapters* into a copy of *input_path* at *output_path*.

    Builds an FFMETADATA1 document from *chapters* and runs
    ``ffmpeg -i <input> -i <ffmetadata> -map_metadata 1 -codec copy
    <output>``. No re-encode: stream copy is near-instant even on
    multi-GB media.

    Use this to round-trip hand-authored chapters back into the file
    itself, so external tools that only read embedded markers see them
    (mpv's ``chapter_list``, QuickTime's chapter track, players that
    don't speak sidecar JSON, etc.).

    Args:
        input_path: Source media. Must exist.
        output_path: Destination path. Overwritten if it exists.
            Must not be the same as *input_path* (ffmpeg refuses to
            read and write the same file simultaneously).
        chapters: Chapters to embed. Empty list is allowed and
            produces a metadata-only copy without any chapter atoms.
        ffmpeg: Override the ffmpeg binary. Defaults to
            :func:`_find_ffmpeg`.

    Returns:
        The output path.

    Raises:
        FileNotFoundError: If *input_path* does not exist or ffmpeg
            isn't on PATH and no bundled binary is available.
        ChapterError: If ffmpeg exits non-zero (stderr surfaced).
    """
    import tempfile

    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    if src.resolve() == dst.resolve():
        raise ChapterError(
            "embed_in_mp4: input_path and output_path resolve to the "
            "same file; ffmpeg cannot stream-copy in place"
        )

    ffmpeg_bin = ffmpeg or _find_ffmpeg()
    metadata_text = _format_ffmetadata1(chapters)

    # Write the FFMETADATA1 document to a temp file so ffmpeg can read
    # it as a second input. Use NamedTemporaryFile with delete=False so
    # the path is stable across the subprocess call; clean up in
    # `finally` regardless of outcome.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".ffmetadata", delete=False,
    )
    try:
        tmp.write(metadata_text)
        tmp.close()
        try:
            result = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",                     # overwrite output without prompt
                    "-i", str(src),
                    "-i", tmp.name,
                    "-map_metadata", "1",     # take metadata from the FFMETADATA1 input
                    "-map_chapters", "1",     # take chapters from the FFMETADATA1 input
                    "-codec", "copy",         # no re-encode
                    str(dst),
                ],
                capture_output=True,
                text=True,
                creationflags=_NO_WINDOW,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"ffmpeg not found ({ffmpeg_bin}). Install ffmpeg or "
                f"place ffmpeg.exe alongside videoflow."
            ) from exc

        if result.returncode != 0:
            raise ChapterError(
                f"ffmpeg exited {result.returncode} while embedding "
                f"chapters into {dst}: {result.stderr.strip()}"
            )
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    return dst
