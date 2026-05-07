"""Tests for videoflow.progress — StageEvent, ETAEstimator, ProgressReporter."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from videoflow.progress import (
    ETAEstimator,
    OnProgress,
    ProgressReporter,
    StageEvent,
    adapt_string_callback,
)


class TestStageEvent(unittest.TestCase):

    def test_immutable_dataclass(self):
        ev = StageEvent(kind="start", stage_path=("a", "b"))
        with self.assertRaises(Exception):
            ev.kind = "complete"  # type: ignore[misc]

    def test_default_fields(self):
        ev = StageEvent(kind="start", stage_path=("a",))
        self.assertEqual(ev.kind, "start")
        self.assertEqual(ev.stage_path, ("a",))
        self.assertIsNone(ev.progress)
        self.assertIsNone(ev.eta_seconds)
        self.assertIsNone(ev.elapsed_seconds)
        self.assertIsNone(ev.summary)
        self.assertIsNone(ev.message)

    def test_stage_property_returns_leaf(self):
        ev = StageEvent(kind="start", stage_path=("analyze", "beats", "chapter 3/12"))
        self.assertEqual(ev.stage, "chapter 3/12")

    def test_stage_property_empty_path(self):
        ev = StageEvent(kind="start", stage_path=())
        self.assertEqual(ev.stage, "")

    def test_complete_event_carries_elapsed_and_summary(self):
        ev = StageEvent(
            kind="complete",
            stage_path=("decode",),
            elapsed_seconds=2.4,
            summary="22 050 Hz mono",
        )
        self.assertEqual(ev.elapsed_seconds, 2.4)
        self.assertEqual(ev.summary, "22 050 Hz mono")


class TestETAEstimator(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.timings_path = Path(self._tmp.name) / "timings.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_predict_unknown_stage_returns_none(self):
        est = ETAEstimator(path=self.timings_path)
        self.assertIsNone(est.predict(("never", "seen")))

    def test_first_observation_sets_total(self):
        est = ETAEstimator(path=self.timings_path)
        est.observe(("a", "b"), 10.0)
        self.assertEqual(est.predict(("a", "b")), 10.0)

    def test_observation_uses_ema_after_first(self):
        # _EMA_ALPHA is 0.3 — second observation pulls 30% toward new value.
        est = ETAEstimator(path=self.timings_path)
        est.observe(("a",), 10.0)
        est.observe(("a",), 20.0)
        # 0.7 * 10 + 0.3 * 20 = 13.0
        self.assertAlmostEqual(est.predict(("a",)), 13.0, places=4)

    def test_predict_with_progress_scales_remaining(self):
        est = ETAEstimator(path=self.timings_path)
        est.observe(("a",), 10.0)
        # half done → 5s left; 80% done → 2s left
        self.assertAlmostEqual(est.predict(("a",), progress=0.5), 5.0)
        self.assertAlmostEqual(est.predict(("a",), progress=0.8), 2.0)

    def test_predict_with_progress_one_returns_zero(self):
        est = ETAEstimator(path=self.timings_path)
        est.observe(("a",), 10.0)
        self.assertEqual(est.predict(("a",), progress=1.0), 0.0)

    def test_save_and_load_round_trip(self):
        est = ETAEstimator(path=self.timings_path)
        est.observe(("decode",), 3.5)
        est.observe(("track", "beats"), 12.0)
        est.save()

        reloaded = ETAEstimator(path=self.timings_path)
        self.assertAlmostEqual(reloaded.predict(("decode",)), 3.5)
        self.assertAlmostEqual(reloaded.predict(("track", "beats")), 12.0)

    def test_load_tolerates_missing_file(self):
        # File doesn't exist; should not raise.
        est = ETAEstimator(path=self.timings_path)
        self.assertIsNone(est.predict(("anything",)))

    def test_load_tolerates_corrupt_file(self):
        self.timings_path.parent.mkdir(parents=True, exist_ok=True)
        self.timings_path.write_text("not valid json {{{")
        est = ETAEstimator(path=self.timings_path)
        self.assertIsNone(est.predict(("anything",)))

    def test_load_ignores_non_numeric_values(self):
        self.timings_path.parent.mkdir(parents=True, exist_ok=True)
        self.timings_path.write_text(json.dumps({"a": 1.5, "b": "not a number"}))
        est = ETAEstimator(path=self.timings_path)
        self.assertAlmostEqual(est.predict(("a",)), 1.5)
        self.assertIsNone(est.predict(("b",)))

    def test_save_failure_is_swallowed(self):
        # Point at a path under a non-writable parent on POSIX or just
        # use a file whose parent we delete; here we mock the write_text
        # to raise, ensuring save() doesn't propagate.
        est = ETAEstimator(path=self.timings_path)
        est.observe(("a",), 1.0)
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            est.save()  # Should not raise.


class _Recorder:
    """Simple on_progress that stores every event in a list."""

    def __init__(self) -> None:
        self.events: list[StageEvent] = []

    def __call__(self, event: StageEvent) -> None:
        self.events.append(event)


class TestProgressReporter(unittest.TestCase):

    def test_emits_start_and_complete_for_single_stage(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("decode"):
            pass
        kinds = [e.kind for e in rec.events]
        self.assertEqual(kinds, ["start", "complete"])
        self.assertEqual(rec.events[0].stage_path, ("decode",))
        self.assertEqual(rec.events[1].stage_path, ("decode",))

    def test_complete_carries_elapsed_seconds(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("decode"):
            time.sleep(0.01)
        complete = rec.events[-1]
        self.assertEqual(complete.kind, "complete")
        self.assertIsNotNone(complete.elapsed_seconds)
        self.assertGreaterEqual(complete.elapsed_seconds, 0.005)

    def test_nested_stages_carry_extended_path(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("analyze"):
            with reporter.stage("beats"):
                with reporter.stage("chapter 1/2"):
                    pass

        paths = [e.stage_path for e in rec.events]
        self.assertEqual(paths, [
            ("analyze",),
            ("analyze", "beats"),
            ("analyze", "beats", "chapter 1/2"),
            ("analyze", "beats", "chapter 1/2"),  # complete
            ("analyze", "beats"),                  # complete
            ("analyze",),                          # complete
        ])

    def test_complete_emitted_even_if_stage_raises(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with self.assertRaises(RuntimeError):
            with reporter.stage("decode"):
                raise RuntimeError("boom")
        kinds = [e.kind for e in rec.events]
        self.assertEqual(kinds, ["start", "complete"])

    def test_summary_set_via_complete_method(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("decode"):
            reporter.complete(summary="22 050 Hz mono")
        complete_ev = rec.events[-1]
        self.assertEqual(complete_ev.summary, "22 050 Hz mono")

    def test_summary_isolated_per_stage(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("a"):
            reporter.complete(summary="A done")
            with reporter.stage("b"):
                reporter.complete(summary="B done")
        b_complete = next(e for e in rec.events if e.stage_path == ("a", "b") and e.kind == "complete")
        a_complete = next(e for e in rec.events if e.stage_path == ("a",) and e.kind == "complete")
        self.assertEqual(b_complete.summary, "B done")
        self.assertEqual(a_complete.summary, "A done")

    def test_progress_emits_progress_event(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("encode"):
            reporter.progress(0.4, message="frame 400/1000")

        progress_events = [e for e in rec.events if e.kind == "progress"]
        self.assertEqual(len(progress_events), 1)
        ev = progress_events[0]
        self.assertEqual(ev.stage_path, ("encode",))
        self.assertAlmostEqual(ev.progress, 0.4)
        self.assertEqual(ev.message, "frame 400/1000")

    def test_progress_clamps_out_of_range(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("encode"):
            reporter.progress(-0.5)
            reporter.progress(1.5)
        progress_events = [e for e in rec.events if e.kind == "progress"]
        self.assertEqual(progress_events[0].progress, 0.0)
        self.assertEqual(progress_events[1].progress, 1.0)

    def test_progress_outside_stage_is_silent(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        reporter.progress(0.5)  # No active stage — should silently no-op.
        self.assertEqual(rec.events, [])

    def test_message_emits_progress_event_with_no_fraction(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        with reporter.stage("track"):
            reporter.message("Detecting beats…")
        progress_events = [e for e in rec.events if e.kind == "progress"]
        self.assertEqual(len(progress_events), 1)
        ev = progress_events[0]
        self.assertIsNone(ev.progress)
        self.assertEqual(ev.message, "Detecting beats…")
        self.assertEqual(ev.stage_path, ("track",))

    def test_message_outside_stage_is_silent(self):
        rec = _Recorder()
        reporter = ProgressReporter(rec)
        reporter.message("nope")
        self.assertEqual(rec.events, [])

    def test_no_callback_does_not_crash(self):
        reporter = ProgressReporter(on_progress=None)
        with reporter.stage("a"):
            reporter.progress(0.5)
            reporter.complete(summary="ok")
        # Reaching here without exception is the assertion.

    def test_callback_exception_is_swallowed(self):
        def boom(_event):
            raise RuntimeError("UI thread crashed")

        reporter = ProgressReporter(on_progress=boom)
        # Engine should keep running even if the UI throws.
        with reporter.stage("a"):
            reporter.progress(0.5)

    def test_estimator_gets_observations(self):
        with TemporaryDirectory() as tmp:
            est = ETAEstimator(path=Path(tmp) / "t.json")
            reporter = ProgressReporter(_Recorder(), estimator=est)
            with reporter.stage("decode"):
                time.sleep(0.01)
            # The stage was observed in the estimator.
            predicted = est.predict(("decode",))
            self.assertIsNotNone(predicted)
            self.assertGreaterEqual(predicted, 0.005)

    def test_save_timings_no_estimator_is_silent(self):
        reporter = ProgressReporter(_Recorder(), estimator=None)
        reporter.save_timings()  # Should not raise.

    def test_estimator_provides_eta_on_second_run(self):
        with TemporaryDirectory() as tmp:
            est_path = Path(tmp) / "t.json"
            # First run: warm the estimator.
            est = ETAEstimator(path=est_path)
            reporter = ProgressReporter(_Recorder(), estimator=est)
            with reporter.stage("track"):
                time.sleep(0.01)
            est.save()

            # Second run: start event should now carry an ETA.
            est2 = ETAEstimator(path=est_path)
            rec = _Recorder()
            reporter2 = ProgressReporter(rec, estimator=est2)
            with reporter2.stage("track"):
                pass
            start_ev = rec.events[0]
            self.assertEqual(start_ev.kind, "start")
            self.assertIsNotNone(start_ev.eta_seconds)


class TestAdaptStringCallback(unittest.TestCase):

    def test_returns_none_for_none_input(self):
        self.assertIsNone(adapt_string_callback(None))

    def test_forwards_leaf_name_on_start(self):
        labels: list[str] = []
        adapter = adapt_string_callback(labels.append)
        assert adapter is not None
        adapter(StageEvent(kind="start", stage_path=("analyze", "beats")))
        self.assertEqual(labels, ["beats"])

    def test_drops_progress_and_complete_events(self):
        labels: list[str] = []
        adapter = adapt_string_callback(labels.append)
        assert adapter is not None
        adapter(StageEvent(kind="progress", stage_path=("a",), progress=0.5))
        adapter(StageEvent(kind="complete", stage_path=("a",), elapsed_seconds=1.0))
        self.assertEqual(labels, [])

    def test_swallows_legacy_callback_exception(self):
        def boom(_label: str) -> None:
            raise RuntimeError("UI crashed")

        adapter = adapt_string_callback(boom)
        assert adapter is not None
        # Should not raise.
        adapter(StageEvent(kind="start", stage_path=("a",)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
