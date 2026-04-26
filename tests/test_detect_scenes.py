"""Tests for videoflow.analysis.detect_scenes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from videoflow.analysis import DETECTOR_INFO, Scene, SceneError, detect_scenes


def _make_timecode(seconds: float) -> MagicMock:
    tc = MagicMock()
    tc.get_seconds.return_value = seconds
    return tc


def _make_raw_scenes(*pairs: tuple[float, float]) -> list:
    """Build a fake PySceneDetect scene list from (start_s, end_s) pairs."""
    return [(_make_timecode(start), _make_timecode(end)) for start, end in pairs]


class TestDetectScenes(unittest.TestCase):

    def _run(self, raw_scenes, *, threshold=None, detector="adaptive"):
        """Run detect_scenes with a mocked PySceneDetect."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            with patch("videoflow.analysis.detect", return_value=raw_scenes), \
                 patch("videoflow.analysis.ContentDetector"), \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector"):
                return detect_scenes(src, threshold=threshold, detector=detector)

    def test_returns_list_of_scenes(self):
        result = self._run(_make_raw_scenes((0.0, 4.2), (4.2, 9.8)))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for s in result:
            self.assertIsInstance(s, Scene)

    def test_timestamps_converted_to_ms(self):
        result = self._run(_make_raw_scenes((0.0, 4.2), (4.2, 9.8)))
        self.assertEqual(result[0].start_ms, 0)
        self.assertEqual(result[0].end_ms, 4200)
        self.assertEqual(result[1].start_ms, 4200)
        self.assertEqual(result[1].end_ms, 9800)

    def test_indices_are_one_based(self):
        result = self._run(_make_raw_scenes((0.0, 3.0), (3.0, 6.0), (6.0, 9.0)))
        self.assertEqual([s.index for s in result], [1, 2, 3])

    def test_duration_ms_property(self):
        result = self._run(_make_raw_scenes((0.0, 5.0)))
        self.assertEqual(result[0].duration_ms, 5000)

    def test_empty_result(self):
        result = self._run([])
        self.assertEqual(result, [])

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            detect_scenes("/nonexistent/clip.mp4")

    def test_invalid_detector(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with self.assertRaises(ValueError) as ctx:
                detect_scenes(f.name, detector="optical_flow")
            self.assertIn("detector", str(ctx.exception))

    def test_scenedetect_not_installed(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch("videoflow.analysis.detect", None):
                with self.assertRaises(SceneError) as ctx:
                    detect_scenes(f.name)
                self.assertIn("pip install scenedetect", str(ctx.exception))

    def test_detection_error_wrapped(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            with patch("videoflow.analysis.detect", side_effect=RuntimeError("codec error")), \
                 patch("videoflow.analysis.ContentDetector"), \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector"):
                with self.assertRaises(SceneError) as ctx:
                    detect_scenes(src)
                self.assertIn("codec error", str(ctx.exception))

    def test_default_detector_is_adaptive(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            mock_adaptive = MagicMock()
            with patch("videoflow.analysis.detect", return_value=[]) as mock_detect, \
                 patch("videoflow.analysis.ContentDetector"), \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector", return_value=mock_adaptive):
                detect_scenes(src)
                mock_detect.assert_called_once_with(str(src), mock_adaptive)

    def test_uses_content_detector(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            mock_content = MagicMock()
            with patch("videoflow.analysis.detect", return_value=[]) as mock_detect, \
                 patch("videoflow.analysis.ContentDetector", return_value=mock_content), \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector"):
                detect_scenes(src, detector="content")
                mock_detect.assert_called_once_with(str(src), mock_content)

    def test_uses_threshold_detector(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            mock_threshold = MagicMock()
            with patch("videoflow.analysis.detect", return_value=[]) as mock_detect, \
                 patch("videoflow.analysis.ContentDetector"), \
                 patch("videoflow.analysis.ThresholdDetector", return_value=mock_threshold), \
                 patch("videoflow.analysis.AdaptiveDetector"):
                detect_scenes(src, detector="threshold")
                mock_detect.assert_called_once_with(str(src), mock_threshold)

    def test_uses_adaptive_detector(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            mock_adaptive = MagicMock()
            with patch("videoflow.analysis.detect", return_value=[]) as mock_detect, \
                 patch("videoflow.analysis.ContentDetector"), \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector", return_value=mock_adaptive):
                detect_scenes(src, detector="adaptive")
                mock_detect.assert_called_once_with(str(src), mock_adaptive)

    def test_explicit_threshold_passed_to_content_detector(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            with patch("videoflow.analysis.detect", return_value=[]), \
                 patch("videoflow.analysis.ContentDetector") as mock_cls, \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector"):
                detect_scenes(src, threshold=15.0, detector="content")
                mock_cls.assert_called_once_with(threshold=15.0)

    def test_explicit_threshold_passed_to_adaptive_detector(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            src = Path(f.name)
            with patch("videoflow.analysis.detect", return_value=[]), \
                 patch("videoflow.analysis.ContentDetector"), \
                 patch("videoflow.analysis.ThresholdDetector"), \
                 patch("videoflow.analysis.AdaptiveDetector") as mock_cls:
                detect_scenes(src, threshold=5.0, detector="adaptive")
                mock_cls.assert_called_once_with(adaptive_threshold=5.0)

    def test_default_threshold_from_detector_info(self):
        """When threshold=None, each detector uses its own default from DETECTOR_INFO."""
        for detector_name, info in DETECTOR_INFO.items():
            with self.subTest(detector=detector_name):
                with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                    src = Path(f.name)
                    with patch("videoflow.analysis.detect", return_value=[]), \
                         patch("videoflow.analysis.ContentDetector") as mock_content, \
                         patch("videoflow.analysis.ThresholdDetector") as mock_thresh, \
                         patch("videoflow.analysis.AdaptiveDetector") as mock_adaptive:
                        detect_scenes(src, detector=detector_name)
                        if detector_name == "content":
                            mock_content.assert_called_once_with(threshold=info.threshold_default)
                        elif detector_name == "threshold":
                            mock_thresh.assert_called_once_with(threshold=info.threshold_default)
                        else:
                            mock_adaptive.assert_called_once_with(
                                adaptive_threshold=info.threshold_default
                            )

    def test_rounding_ms(self):
        """Fractional milliseconds are rounded to the nearest int."""
        result = self._run(_make_raw_scenes((0.0, 4.2009)))
        self.assertIsInstance(result[0].end_ms, int)
        self.assertEqual(result[0].end_ms, 4201)


class TestDetectorInfo(unittest.TestCase):

    def test_all_detectors_have_info(self):
        for key in ("content", "threshold", "adaptive"):
            self.assertIn(key, DETECTOR_INFO)

    def test_info_fields_populated(self):
        for key, info in DETECTOR_INFO.items():
            with self.subTest(detector=key):
                self.assertTrue(info.name)
                self.assertTrue(info.description)
                self.assertTrue(info.threshold_label)
                self.assertTrue(info.novice_tip)
                self.assertGreater(info.threshold_max, info.threshold_min)
                self.assertGreaterEqual(info.threshold_default, info.threshold_min)
                self.assertLessEqual(info.threshold_default, info.threshold_max)
                self.assertIsInstance(info.best_for, list)
                self.assertIsInstance(info.not_great_for, list)

    def test_adaptive_default_lower_than_content(self):
        """Adaptive threshold scale (1–10) is different from content scale (1–100)."""
        self.assertLess(
            DETECTOR_INFO["adaptive"].threshold_default,
            DETECTOR_INFO["content"].threshold_default,
        )


if __name__ == "__main__":
    unittest.main()
