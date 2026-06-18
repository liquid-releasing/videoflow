"""Sidecar IO for the audio-structure database (``<stem>.chapters.json``).

The sidecar is the structural database for one media file: chapters,
stanzas, energy analysis, per-chapter tone, and any other carry-forward
analysis the toolchain wants to share. Multiple writers (forgegen,
FunscriptForge, forgeassembler) cooperate through field-level merge
semantics rather than a lock.

See ``videoflow/docs/architecture/audio-structure-primitive.md`` for
the architectural rationale, field categories, and the 8-rule merge
contract. ``sidecar.schema.json`` next to that doc is the formal
JSON Schema (Draft 2020-12).

Two write modes:

- ``mode="analyze"`` (default) — the writer is recomputing analysis
  (forgegen, ``structural.auto_chapter``, etc.). On records that match
  by primary key, only ANALYTICAL and MIXED fields are taken from the
  incoming payload; AUTHORED and STRUCTURAL fields on existing records
  are preserved verbatim. New records (no match) are appended as-is.
- ``mode="edit"`` — the writer is committing user edits (FunscriptForge,
  forgeassembler). On matched records, all fields except STRUCTURAL are
  taken from incoming. New records appended.

Both modes respect the LATCH: records with ``auto_generated: false``
are frozen — no field changes regardless of mode.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from videoflow.chapters import Chapter

CURRENT_SCHEMA_VERSION = "3.0"
SCHEMA_NAME = "audio-structure"

# Analyzer (algorithm) version — DISTINCT from the schema version and the
# package version. Bump this ONLY when the chapter/beat/energy detection
# ALGORITHM changes such that an existing analysis should be regenerated
# (e.g. a new chapter heuristic, a different beat tracker). Stamped into the
# chapters.json on every analyze-mode write; consumers (FunscriptForge's
# load_project + auto-analyze gate) compare it against their own mirror
# constant and force a re-analyze on mismatch. Decoupled from the package
# version so unrelated releases don't needlessly re-grind every project.
# Mirror constant: funscriptforge ui/web/src/api/forge.js ANALYZER_VERSION.
ANALYZER_VERSION = "1"

WriteMode = Literal["analyze", "edit"]

# Field categories — drives merge. See audio-structure-primitive.md.
_CHAPTER_ANALYTICAL: tuple[str, ...] = ("content_type", "voice_label", "confidence", "evidence")
_CHAPTER_AUTHORED: tuple[str, ...] = ("name", "intent", "include")
_CHAPTER_MIXED: tuple[str, ...] = ("tone", "shape")
_CHAPTER_STRUCTURAL: tuple[str, ...] = ("at_ms", "end_ms")

_STANZA_ANALYTICAL: tuple[str, ...] = ("mode", "confidence", "evidence")
_STANZA_AUTHORED: tuple[str, ...] = ("intent",)
_STANZA_MIXED: tuple[str, ...] = ("tone",)
_STANZA_STRUCTURAL: tuple[str, ...] = ("chapter_idx", "at_ms", "end_ms")

# Texture vocabulary. New names (driving / calm / varied) replaced
# music / ambient / mixed on 2026-05-24 — see
# videoflow.structural._TEXTURE_DRIVING for the rename rationale.
# Both vocabularies are accepted on write so legacy sidecars round-trip;
# Chapter.from_dict maps legacy values to new ones on read.
_VALID_CONTENT_TYPES = frozenset({
    "",
    "driving", "calm", "varied",      # current (2026-05-24+)
    "music", "ambient", "mixed",      # legacy — accepted on write, mapped on read
})
# Voice axis (Silero VAD). Round 1 is binary (talk vs. not). Round 2
# will add 'sing' / 'react' via pitch-stability + segment-length heuristics
# on the VAD output.
_VALID_VOICE_LABELS = frozenset({"", "talk"})
_VALID_STANZA_MODES = frozenset({"", "tease", "steady", "edging", "break", "fast", "slow"})

# FunscriptForge mood vocabulary (closed enum, cross-product semantic).
# See videoflow/docs/architecture/audio-structure-primitive.md#tone-vocabulary
# and funscriptforge/docs/guide/tone.md.
_VALID_TONE_LABELS = frozenset({
    "", "tender", "build", "tease", "edge", "climax", "dominant",
})

# forgegen curve-direction primitive (closed enum). Drives the per-stanza
# center trajectory used by beats_to_curve. Distinct from `tone` above.
# Sidecar field name is `shape`; forgegen's UI labels this control "Tone".
_VALID_SHAPES = frozenset({"", "flat", "rise", "fall", "auto"})


class SidecarError(RuntimeError):
    """Raised when the sidecar is malformed or merge cannot proceed."""


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

def forge_dir(media_path: str | Path) -> Path:
    """Return the per-project hidden cache directory for *media_path*.

    All videoflow sidecars (chapters, peaks, spectrogram, beats,
    stanzas) and the funscriptforge chapter-clip MP4 cache live inside
    this folder rather than scattering next to the source file:

        <dir>/.<stem>.forge/
            <stem>.chapters.json
            <stem>.audio.json
            <stem>.spectrogram.json
            <stem>.beats.json
            clips/
                <stem>_v11_<start>_<end>.mp4

    Leading dot keeps it tidy in directory listings. Callers are
    responsible for ``mkdir(parents=True, exist_ok=True)`` before
    writing.
    """
    path = Path(media_path)
    return path.parent / f".{path.stem}.forge"


def sidecar_path_for(media_path: str | Path) -> Path:
    """Return the canonical chapters sidecar path inside the forge dir.

    e.g. ``track.mp4`` → ``.track.forge/track.chapters.json``.
    """
    path = Path(media_path)
    return forge_dir(path) / f"{path.stem}.chapters.json"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_sidecar(media_path: str | Path) -> dict[str, Any] | None:
    """Read ``<stem>.chapters.json`` next to *media_path*.

    Returns the parsed JSON document with v1 sidecars normalised to v2
    shape (empty ``stanzas`` / ``provenance`` lists, no ``energy``).
    Returns ``None`` when no sidecar exists. Unknown fields round-trip
    verbatim — readers must not strip them on writeback.

    Raises:
        SidecarError: If the file is unreadable JSON, the top-level
            shape is not an object or list, or a required field is
            missing on a chapter / stanza record.
    """
    path = sidecar_path_for(media_path)
    if not path.exists():
        return None

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SidecarError(f"sidecar {path} is not valid JSON: {exc}") from exc

    if isinstance(doc, list):
        # v0: bare list of chapters (the form chapters.read_sidecar_chapters
        # accepts). Treat as a tiny v1 doc so the merger doesn't fork.
        doc = {"chapters": doc}

    if not isinstance(doc, dict):
        raise SidecarError(
            f"sidecar {path} must be a JSON object or list, got "
            f"{type(doc).__name__}"
        )

    return _normalise(doc, source=str(path))


def chapters_from_sidecar(doc: dict[str, Any]) -> list[Chapter]:
    """Return :class:`Chapter` records parsed from a sidecar dict.

    Convenience for callers that want the typed dataclass instead of
    raw dicts. Unknown chapter fields stay in the dict — they are not
    surfaced on :class:`Chapter`.
    """
    return [Chapter.from_dict(c) for c in doc.get("chapters", [])]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_sidecar(
    media_path: str | Path,
    payload: dict[str, Any],
    *,
    writer: str,
    writer_version: str = "",
    mode: WriteMode = "analyze",
) -> Path:
    """Write *payload* to ``<stem>.chapters.json`` with field-level merge.

    Behaviour:

    - **Fresh write** (no sidecar exists): *payload* is written verbatim
      with v2 defaults filled in and a fresh ``provenance`` entry stamped.
    - **Merge write** (sidecar exists): the existing document is loaded,
      and each record in *payload* is merged into it according to *mode*
      and the field-level rules in the architecture doc. ANALYTICAL
      fields on records flagged ``auto_generated: true`` are overwritten
      by analyze-mode writers; AUTHORED fields are preserved by analyze
      writers and overwritten by edit writers. Records flagged
      ``auto_generated: false`` (LATCH) are left fully untouched in
      both modes. New records are appended. Existing records that
      *payload* does not mention are kept as-is — writers never delete
      user data.

    The top-level ``energy`` block is ANALYTICAL: it is overwritten
    wholesale when present in *payload* (both modes).

    Args:
        media_path: Path to the source media. The sidecar is written
            next to it as ``<stem>.chapters.json``.
        payload: The data to merge in. Recognised top-level fields:
            ``chapters``, ``stanzas``, ``energy``, plus any unknown
            top-level fields the caller wants persisted.
        writer: Module / app id (e.g. ``"videoflow.structural"``,
            ``"FunscriptForge"``). Recorded in provenance.
        writer_version: Caller-supplied version string. Optional.
        mode: ``"analyze"`` (default) for recompute writers,
            ``"edit"`` for user-edit writers.

    Returns:
        The path that was written.

    Raises:
        SidecarError: If the existing sidecar is malformed or *payload*
            contains malformed records.
    """
    target = sidecar_path_for(media_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    incoming = _normalise(dict(payload), source="payload")

    existing = read_sidecar(media_path)
    if existing is None:
        merged = _fresh_payload(
            incoming, writer=writer, writer_version=writer_version,
        )
    else:
        merged = _merge_payload(
            existing, incoming,
            writer=writer, writer_version=writer_version, mode=mode,
        )

    # Stamp the analyzer (algorithm) version on every analyze-mode write so
    # consumers can detect a version bump and re-analyze. Edit-mode writes
    # preserve whatever the analyzer last stamped (merged already carries it
    # from the existing doc) — a user editing chapter names must not look like
    # a fresh analysis.
    if mode == "analyze":
        merged["analyzer_version"] = ANALYZER_VERSION

    target.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Normalisation / validation
# ---------------------------------------------------------------------------

def _normalise(doc: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Validate *doc* and fill in v2 defaults. Returns a new dict.

    v1 sidecars with top-level ``auto_generated: false`` (the legacy
    "user-edited file" marker) are migrated forward by setting
    per-record ``auto_generated: false`` on every chapter and stanza
    that doesn't already carry its own value. This preserves the v1
    user-edit-protection semantics inside the new per-record latch
    model the merger consults.
    """
    out: dict[str, Any] = dict(doc)

    out["schema"] = doc.get("schema", SCHEMA_NAME)
    out["version"] = doc.get("version", "1.0")

    legacy_user_edited = doc.get("auto_generated") is False

    chapters = doc.get("chapters", [])
    if not isinstance(chapters, list):
        raise SidecarError(f"{source}: 'chapters' must be a list")
    for i, ch in enumerate(chapters):
        _coerce_legacy_tone(ch)
        _validate_chapter(ch, source=source, idx=i)
        if legacy_user_edited and isinstance(ch, dict) and "auto_generated" not in ch:
            ch["auto_generated"] = False
    out["chapters"] = chapters

    stanzas = doc.get("stanzas", [])
    if not isinstance(stanzas, list):
        raise SidecarError(f"{source}: 'stanzas' must be a list")
    for i, ph in enumerate(stanzas):
        _coerce_legacy_tone(ph)
        _validate_stanza(ph, source=source, idx=i)
        if legacy_user_edited and isinstance(ph, dict) and "auto_generated" not in ph:
            ph["auto_generated"] = False
    out["stanzas"] = stanzas

    energy = doc.get("energy")
    if energy is not None:
        if not isinstance(energy, dict):
            raise SidecarError(
                f"{source}: 'energy' must be an object or absent"
            )
        out["energy"] = energy

    provenance = doc.get("provenance", [])
    if not isinstance(provenance, list):
        raise SidecarError(f"{source}: 'provenance' must be a list")
    out["provenance"] = provenance

    return out


def _coerce_legacy_tone(rec: Any) -> None:
    """FunscriptForge persists a UI-only ``tone: "none"`` (Untoned) sentinel
    that has no videoflow counterpart — its canonical untoned value is ``""``.
    Map ``"none"`` → ``""`` in place so an FF-authored sidecar survives a
    round-trip through an analyze write (which re-reads + re-validates the
    existing doc). Both read_sidecar and write_sidecar route through
    _normalise, so this covers the on-disk doc AND the incoming payload.
    Mirrors the legacy ``content_type`` "accepted on write, mapped on read"
    policy above."""
    if isinstance(rec, dict) and rec.get("tone") == "none":
        rec["tone"] = ""


def _validate_chapter(ch: Any, *, source: str, idx: int) -> None:
    if not isinstance(ch, dict):
        raise SidecarError(
            f"{source}: chapters[{idx}] must be an object, got "
            f"{type(ch).__name__}"
        )
    if "at_ms" not in ch and "at" not in ch and "start_ms" not in ch:
        raise SidecarError(
            f"{source}: chapters[{idx}] missing 'at_ms'"
        )
    ct = ch.get("content_type")
    if ct is not None and ct not in _VALID_CONTENT_TYPES:
        raise SidecarError(
            f"{source}: chapters[{idx}].content_type must be one of "
            f"{sorted(_VALID_CONTENT_TYPES)}, got {ct!r}"
        )
    vl = ch.get("voice_label")
    if vl is not None and vl not in _VALID_VOICE_LABELS:
        raise SidecarError(
            f"{source}: chapters[{idx}].voice_label must be one of "
            f"{sorted(_VALID_VOICE_LABELS)}, got {vl!r}"
        )
    tone = ch.get("tone")
    if tone is not None and tone not in _VALID_TONE_LABELS:
        raise SidecarError(
            f"{source}: chapters[{idx}].tone must be one of "
            f"{sorted(_VALID_TONE_LABELS)}, got {tone!r}"
        )
    shape = ch.get("shape")
    if shape is not None and shape not in _VALID_SHAPES:
        raise SidecarError(
            f"{source}: chapters[{idx}].shape must be one of "
            f"{sorted(_VALID_SHAPES)}, got {shape!r}"
        )


def _validate_stanza(ph: Any, *, source: str, idx: int) -> None:
    if not isinstance(ph, dict):
        raise SidecarError(
            f"{source}: stanzas[{idx}] must be an object, got "
            f"{type(ph).__name__}"
        )
    for required in ("chapter_idx", "at_ms", "end_ms"):
        if required not in ph:
            raise SidecarError(
                f"{source}: stanzas[{idx}] missing '{required}'"
            )
    mode = ph.get("mode")
    if mode is not None and mode not in _VALID_STANZA_MODES:
        raise SidecarError(
            f"{source}: stanzas[{idx}].mode must be one of "
            f"{sorted(_VALID_STANZA_MODES)}, got {mode!r}"
        )
    tone = ph.get("tone")
    if tone is not None and tone not in _VALID_TONE_LABELS:
        raise SidecarError(
            f"{source}: stanzas[{idx}].tone must be one of "
            f"{sorted(_VALID_TONE_LABELS)}, got {tone!r}"
        )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _fresh_payload(
    incoming: dict[str, Any],
    *,
    writer: str,
    writer_version: str,
) -> dict[str, Any]:
    """Build the on-disk payload for a fresh sidecar (no existing file)."""
    out = dict(incoming)
    out["schema"] = SCHEMA_NAME
    out["version"] = CURRENT_SCHEMA_VERSION
    out.setdefault("auto_generated", True)
    fields = _fields_touched(incoming)
    prior = list(incoming.get("provenance") or [])
    out["provenance"] = prior + [
        _provenance_entry(writer, writer_version, fields),
    ]
    return out


_KNOWN_TOP_LEVEL = frozenset({
    "schema", "version", "analyzer_version", "chapters", "stanzas", "energy",
    "provenance", "auto_generated", "generated_by",
})


def _merge_payload(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    writer: str,
    writer_version: str,
    mode: WriteMode,
) -> dict[str, Any]:
    """Merge *incoming* into *existing* with field-level rules."""
    merged = dict(existing)

    # Once any v2-aware writer touches the file, it's a v2 doc.
    merged["schema"] = SCHEMA_NAME
    merged["version"] = CURRENT_SCHEMA_VERSION

    # Top-level auto_generated is informational once a non-detector writer
    # has touched the file; per-record latch is what the merge actually
    # consults. Edit writers always flip this off.
    if mode == "edit":
        merged["auto_generated"] = False

    # First-write info from incoming wins only when existing has none.
    if "generated_by" in incoming and "generated_by" not in existing:
        merged["generated_by"] = incoming["generated_by"]

    # Chapters
    if incoming.get("chapters"):
        merged["chapters"] = _merge_record_list(
            existing.get("chapters") or [],
            incoming["chapters"],
            analytical=_CHAPTER_ANALYTICAL,
            authored=_CHAPTER_AUTHORED,
            mixed=_CHAPTER_MIXED,
            structural=_CHAPTER_STRUCTURAL,
            key_fn=_chapter_key,
            mode=mode,
        )

    # Stanzas — same rules, composite key.
    if incoming.get("stanzas"):
        merged["stanzas"] = _merge_record_list(
            existing.get("stanzas") or [],
            incoming["stanzas"],
            analytical=_STANZA_ANALYTICAL,
            authored=_STANZA_AUTHORED,
            mixed=_STANZA_MIXED,
            structural=_STANZA_STRUCTURAL,
            key_fn=_stanza_key,
            mode=mode,
        )

    # Energy — ANALYTICAL block; replace wholesale when incoming present.
    if "energy" in incoming and incoming["energy"] is not None:
        merged["energy"] = incoming["energy"]

    # Round-trip unknown top-level fields from incoming.
    for k, v in incoming.items():
        if k in _KNOWN_TOP_LEVEL:
            continue
        merged[k] = v

    fields = _fields_touched(incoming)
    merged["provenance"] = list(existing.get("provenance") or []) + [
        _provenance_entry(writer, writer_version, fields),
    ]
    return merged


def _chapter_key(rec: dict) -> tuple:
    return (rec.get("at_ms"),)


def _stanza_key(rec: dict) -> tuple:
    return (rec.get("chapter_idx"), rec.get("at_ms"))


def _merge_record_list(
    existing: list[dict],
    incoming: list[dict],
    *,
    analytical: tuple[str, ...],
    authored: tuple[str, ...],
    mixed: tuple[str, ...],
    structural: tuple[str, ...],
    key_fn,
    mode: WriteMode,
) -> list[dict]:
    """Merge *incoming* records into *existing* by primary key.

    Existing records retain their order; new records (no key match)
    append in incoming order.
    """
    out: list[dict] = []
    seen: set[tuple] = set()

    # First: walk existing in order, applying merge if incoming has a match.
    for old in existing:
        k = key_fn(old)
        if k in seen:
            # Drop duplicates inside existing — keep first.
            continue
        seen.add(k)
        match = next((n for n in incoming if key_fn(n) == k), None)
        if match is None:
            out.append(old)
        else:
            out.append(_merge_one(
                old, match,
                analytical=analytical,
                authored=authored,
                mixed=mixed,
                structural=structural,
                mode=mode,
            ))

    # Second: append incoming records that didn't match any existing.
    existing_keys = {key_fn(r) for r in existing}
    for new in incoming:
        if key_fn(new) in existing_keys:
            continue
        out.append(dict(new))

    return out


def _merge_one(
    old: dict,
    new: dict,
    *,
    analytical: tuple[str, ...],
    authored: tuple[str, ...],
    mixed: tuple[str, ...],
    structural: tuple[str, ...],
    mode: WriteMode,
) -> dict:
    """Field-level merge of one matched record."""
    # LATCH: user has authored this record — freeze all fields.
    if old.get("auto_generated") is False:
        return dict(old)

    out = dict(old)

    if mode == "analyze":
        # Replace ANALYTICAL with whatever incoming has (including
        # explicit clears). MIXED follows the same rule (latch is open).
        for f in (*analytical, *mixed):
            if f in new:
                out[f] = new[f]
        # AUTHORED + STRUCTURAL are not touched by analyze writers.
        # auto_generated: only honour if incoming explicitly sets it.
        if "auto_generated" in new:
            out["auto_generated"] = bool(new["auto_generated"])
        # Round-trip unknown fields incoming carries.
        _carry_unknown(
            out, new,
            known=set(analytical) | set(authored) | set(mixed)
                  | set(structural) | {"auto_generated"},
        )
    else:  # mode == "edit"
        # User edit: apply ANALYTICAL + AUTHORED + MIXED. STRUCTURAL stays.
        for f in (*analytical, *authored, *mixed):
            if f in new:
                out[f] = new[f]
        # Edit flips per-record auto_generated to false unless caller
        # explicitly sets it (e.g. an "un-author" action).
        out["auto_generated"] = bool(new.get("auto_generated", False))
        _carry_unknown(
            out, new,
            known=set(analytical) | set(authored) | set(mixed)
                  | set(structural) | {"auto_generated"},
        )

    return out


def _carry_unknown(out: dict, new: dict, *, known: set[str]) -> None:
    """Round-trip unknown fields from *new* that aren't already in *out*."""
    for k, v in new.items():
        if k not in known and k not in out:
            out[k] = v


def _fields_touched(payload: dict[str, Any]) -> list[str]:
    """Best-effort summary of which top-level blocks the writer touched."""
    fields: list[str] = []
    for k in ("chapters", "stanzas", "energy"):
        v = payload.get(k)
        if v:
            fields.append(k)
    return fields


def _provenance_entry(
    writer: str, writer_version: str, fields: list[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "writer": writer,
        "written_at": _utc_now_iso(),
    }
    if writer_version:
        entry["writer_version"] = writer_version
    if fields:
        entry["fields"] = fields
    return entry


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
