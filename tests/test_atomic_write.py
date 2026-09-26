"""A reader must never observe a partially-written sidecar.

Reported from funscriptforge dogfood 2026-09-26:

    could not parse beats sidecar at ...beats.json:
    EOF while parsing a value at line 1 column 0

Column 0 of line 1 means ZERO BYTES. The file was valid (5487 bytes) seconds
later -- it was read while ``open(path, "w")`` had it truncated.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from videoflow.atomic_write import write_json_atomic, write_text_atomic


def test_writes_content_and_creates_parents(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "x.json"
    write_json_atomic(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}


def test_overwrites_an_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    write_json_atomic(p, {"v": 1})
    write_json_atomic(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}


def test_leaves_no_part_files_behind(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    write_json_atomic(p, {"a": 1})
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_dumps_kwargs_reach_json(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    write_json_atomic(p, {"a": [1, 2]}, separators=(",", ":"))
    assert p.read_text() == '{"a":[1,2]}'
    write_json_atomic(p, {"a": 1}, indent=2)
    assert "\n" in p.read_text()


class TestTheBugItself:
    def test_a_failed_write_leaves_the_ORIGINAL_file_intact(self, tmp_path: Path) -> None:
        # ★ This is what `open(path, "w")` could not do. It truncates first, so
        # a serialisation error (or a kill) left an EMPTY file where a valid
        # one had been. Here the destination is untouched until the content is
        # complete on disk.
        p = tmp_path / "beats.json"
        write_json_atomic(p, {"bpm": 128})
        with pytest.raises(TypeError):
            write_json_atomic(p, {"bpm": object()})
        assert json.loads(p.read_text()) == {"bpm": 128}

    def test_a_failed_write_leaves_no_litter(self, tmp_path: Path) -> None:
        p = tmp_path / "beats.json"
        write_json_atomic(p, {"bpm": 128})
        with pytest.raises(TypeError):
            write_json_atomic(p, {"bpm": object()})
        assert [f.name for f in tmp_path.iterdir()] == ["beats.json"]

    def test_a_concurrent_reader_never_sees_an_empty_or_partial_file(
        self, tmp_path: Path
    ) -> None:
        # The reported failure, reproduced as a race. A reader hammers the path
        # while a writer rewrites it repeatedly with a payload big enough that a
        # truncate-then-write would leave a wide window. Every successful read
        # must parse; "file missing" is allowed, "file empty" is not.
        p = tmp_path / "spectrogram.json"
        big = {"cells_b64": "x" * 400_000, "n": 0}
        write_json_atomic(p, big)

        seen_bad: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    raw = p.read_text()
                except (FileNotFoundError, PermissionError):
                    continue
                if raw == "":
                    seen_bad.append("EMPTY")
                    continue
                try:
                    json.loads(raw)
                except json.JSONDecodeError:
                    seen_bad.append("PARTIAL")

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for i in range(40):
                big["n"] = i
                write_json_atomic(p, big)
        finally:
            stop.set()
            t.join(timeout=5)

        assert seen_bad == [], f"reader saw {len(seen_bad)} bad reads: {set(seen_bad)}"
        assert json.loads(p.read_text())["n"] == 39


def test_write_text_atomic_round_trips_unicode(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    write_text_atomic(p, "café ★")
    assert p.read_text(encoding="utf-8") == "café ★"


def test_returns_the_destination_path(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    assert write_json_atomic(p, {}) == Path(p)
    assert write_text_atomic(tmp_path / "y.txt", "hi") == Path(tmp_path / "y.txt")


class TestWindowsReplaceRetry:
    """Windows refuses to replace a destination another handle has open.

    POSIX does not, so these are Windows-only -- and that asymmetry is the
    point: CI runs on Linux, so without these the retry could rot unnoticed
    and only fail on the platform the app ships on.
    """

    nt_only = pytest.mark.skipif(os.name != "nt", reason="POSIX replaces open files")

    @nt_only
    def test_it_waits_for_a_reader_to_let_go(self, tmp_path: Path) -> None:
        p = tmp_path / "held.json"
        write_json_atomic(p, {"v": 1})

        released = threading.Event()

        def holder() -> None:
            with open(p, "r"):
                released.wait(timeout=5)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        threading.Event().wait(0.05)          # let the holder actually open it

        def free_it() -> None:
            threading.Event().wait(0.3)
            released.set()

        threading.Thread(target=free_it, daemon=True).start()

        write_json_atomic(p, {"v": 2}, replace_timeout_s=5.0)
        assert json.loads(p.read_text()) == {"v": 2}
        released.set()
        t.join(timeout=5)

    @nt_only
    def test_it_raises_rather_than_silently_truncating(self, tmp_path: Path) -> None:
        # ★ The destination must keep its previous, COMPLETE contents. Falling
        # back to a truncating write here would resurrect the very bug this
        # module exists to prevent, and would do it silently.
        p = tmp_path / "held.json"
        write_json_atomic(p, {"v": 1})

        release = threading.Event()

        def holder() -> None:
            with open(p, "r"):
                release.wait(timeout=10)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        threading.Event().wait(0.05)

        with pytest.raises(PermissionError):
            write_json_atomic(p, {"v": 2}, replace_timeout_s=0.2)

        assert json.loads(p.read_text()) == {"v": 1}     # intact, not empty
        assert [f.name for f in tmp_path.iterdir()] == ["held.json"]  # no litter
        release.set()
        t.join(timeout=5)
