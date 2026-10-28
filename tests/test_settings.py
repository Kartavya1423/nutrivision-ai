"""
Unit tests for centralized configuration system (src.config.settings).
"""

import os
import unittest
from pathlib import Path

from src.config.settings import Settings, get_settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.settings = get_settings()

    def test_project_defaults(self):
        self.assertEqual(self.settings.project_name, "NutriVision AI")
        self.assertTrue(self.settings.model_paths.root.is_dir())

    def test_model_paths_configured(self):
        paths = self.settings.model_paths
        self.assertTrue(str(paths.detection_model_path).endswith("best.pt") or str(paths.detection_model_path).endswith("yolo11n.pt"))
        self.assertTrue(str(paths.segmentation_model_path).endswith("sam2.1_t.pt") or str(paths.segmentation_model_path).endswith("sam2_hiera_base_plus.pt"))
        self.assertTrue(str(paths.classification_model_path).endswith("efficientnet_b4_food.pth"))
        self.assertTrue(str(paths.depth_model_path).endswith("depth_anything_v2_vits.pth"))
        self.assertTrue(str(paths.freshness_model_path).endswith("freshness_classifier.pth"))

    def test_threshold_ranges(self):
        self.assertGreater(self.settings.detection.confidence_threshold, 0.0)
        self.assertLess(self.settings.detection.confidence_threshold, 1.0)
        self.assertGreater(self.settings.detection.iou_threshold, 0.0)
        self.assertLess(self.settings.detection.iou_threshold, 1.0)

    def test_output_directories_exist(self):
        outputs = self.settings.outputs
        self.assertTrue(outputs.detections.is_dir())
        self.assertTrue(outputs.segmentation.is_dir())
        self.assertTrue(outputs.crops.is_dir())
        self.assertTrue(outputs.depth.is_dir())
        self.assertTrue(outputs.results.is_dir())
        self.assertTrue(outputs.visualizations.is_dir())
        self.assertTrue(outputs.logs.is_dir())


if __name__ == "__main__":
    unittest.main()
