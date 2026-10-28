"""
NutriVision AI — Unit Tests for Depth Anything V2 Monocular Depth Model
Phase 5: Food Depth Estimation
Verifies real model loading, preprocessing, resolution preservation, metadata,
and strict relative-vs-metric calibration contracts.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.depth.depth_anything import DepthAnythingV2Estimator
from src.depth.depth_estimator import DepthResult, DepthType
from src.detection.detector import ModelStatus


class TestDepthAnythingV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = Path("models/depth/depth_anything_v2_vits.pth")
        cls.estimator = DepthAnythingV2Estimator(model_path=cls.ckpt_path, device="cpu")

    def test_depth_model_loading_and_readiness(self):
        """Verify real Depth Anything V2 checkpoint loads and marks model ready."""
        self.assertTrue(self.ckpt_path.is_file(), f"Checkpoint missing: {self.ckpt_path}")
        self.assertTrue(self.estimator.is_ready())
        self.assertEqual(self.estimator.status, ModelStatus.READY)

    def test_missing_checkpoint_reports_not_loaded(self):
        """Unconfigured or missing checkpoint strictly reports MODEL_NOT_LOADED."""
        missing_est = DepthAnythingV2Estimator(model_path="non_existent_depth_model.pth")
        self.assertFalse(missing_est.is_ready())
        self.assertEqual(missing_est.status, ModelStatus.MODEL_NOT_LOADED)

        dummy = np.zeros((120, 120, 3), dtype=np.uint8)
        res = missing_est.estimate_depth(dummy)
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertIsNone(res.depth_map)
        self.assertFalse(res.is_success)

    def test_inference_shape_and_resolution_preservation(self):
        """Inference output depth map must strictly match input image dimensions (H, W)."""
        test_h, test_w = 180, 240
        dummy_rgb = np.full((test_h, test_w, 3), 100, dtype=np.uint8)
        result = self.estimator.estimate_depth(dummy_rgb)

        self.assertTrue(result.is_success)
        self.assertIsNotNone(result.depth_map)
        self.assertEqual(result.depth_map.shape, (test_h, test_w))
        self.assertEqual(result.depth_map.dtype, np.float32)

    def test_depth_statistics_and_metadata(self):
        """Verify depth metadata contains min, max, mean, median, percentiles, and device."""
        dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.estimator.estimate_depth(dummy_rgb)

        self.assertGreater(result.inference_time_ms, 0.0)
        self.assertGreaterEqual(result.max_depth, result.min_depth)
        self.assertIsNotNone(result.percentiles)
        self.assertIn("p50", result.percentiles)
        self.assertEqual(result.device, "cpu")

        meta = result.to_dict()
        self.assertEqual(meta["status"], "READY")
        self.assertFalse(meta["is_metric_calibrated"])
        self.assertEqual(meta["calibration_status"], "DEPTH_SCALE_NOT_CALIBRATED")

    def test_render_colormap_rgb_output(self):
        """Colormap rendering produces valid (H, W, 3) RGB uint8 visualization."""
        dummy_rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        result = self.estimator.estimate_depth(dummy_rgb)

        colormap = result.render_colormap()
        self.assertIsNotNone(colormap)
        self.assertEqual(colormap.shape, (80, 100, 3))
        self.assertEqual(colormap.dtype, np.uint8)

    def test_uncalibrated_depth_is_strictly_relative(self):
        """Without external calibration, output depth is relative and not metric."""
        dummy_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        result = self.estimator.estimate_depth(dummy_rgb, calibration=None)

        self.assertEqual(result.depth_type, DepthType.RELATIVE)
        self.assertFalse(result.is_metric_calibrated)
        self.assertEqual(result.calibration_status, "DEPTH_SCALE_NOT_CALIBRATED")

    def test_calibrated_depth_status(self):
        """When calibration information is supplied, depth is marked calibrated."""
        dummy_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        calib = {"cm_per_pixel": 0.05, "reference_object": "test_coin"}
        result = self.estimator.estimate_depth(dummy_rgb, calibration=calib)

        self.assertEqual(result.depth_type, DepthType.METRIC)
        self.assertTrue(result.is_metric_calibrated)
        self.assertEqual(result.calibration_status, "CALIBRATED")
        self.assertEqual(result.reference_object_used, "test_coin")


if __name__ == "__main__":
    unittest.main()
