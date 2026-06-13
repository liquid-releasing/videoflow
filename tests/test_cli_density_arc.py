"""Tests for the generate-funscript --density-arc wiring and Windows-safe help.

The narrative density arc is the dynamic-density signature of a great script
(see GENERATION_DENSITY_ARC.md). It must reach users through the CLI — on by
default so the box delivers the arc with zero tuning — and the help must not
crash on a cp1252 (Windows) console.
"""

from __future__ import annotations

import io
import unittest

from videoflow.cli import build_parser


class TestDensityArcArg(unittest.TestCase):

    def setUp(self):
        self.parser = build_parser()

    def test_default_is_arc_on(self):
        args = self.parser.parse_args(
            ["generate-funscript", "in.mp3", "out.funscript"]
        )
        self.assertEqual(args.density_arc, "default")

    def test_can_disable(self):
        args = self.parser.parse_args(
            ["generate-funscript", "in.mp3", "out.funscript",
             "--density-arc", "none"]
        )
        self.assertEqual(args.density_arc, "none")

    def test_rejects_unknown_value(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["generate-funscript", "in.mp3", "out.funscript",
                 "--density-arc", "wild"]
            )


class TestHelpIsCp1252Safe(unittest.TestCase):
    """generate-funscript --help crashed on Windows because a help string held
    a '->' arrow as U+2192, which cp1252 can't encode. Lock the help text to
    characters a Windows console can print."""

    def test_generate_help_encodes_in_cp1252(self):
        parser = build_parser()
        # Reach the generate-funscript subparser and render its help.
        sub = parser._subparsers._group_actions[0].choices["generate-funscript"]
        help_text = sub.format_help()
        try:
            help_text.encode("cp1252")
        except UnicodeEncodeError as exc:  # pragma: no cover - failure detail
            self.fail(f"generate-funscript help not cp1252-safe: {exc}")

    def test_all_subcommand_help_encodes_in_cp1252(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices
        for name, sub in choices.items():
            with self.subTest(command=name):
                try:
                    sub.format_help().encode("cp1252")
                except UnicodeEncodeError as exc:
                    self.fail(f"'{name}' help not cp1252-safe: {exc}")


if __name__ == "__main__":
    unittest.main()
