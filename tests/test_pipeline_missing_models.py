"""
Unit tests for pipeline behavior when models are absent or untrained (src.pipeline).
Ensures pipeline returns structured status (e.g. DETECTION_MODEL_NOT_AVAILABLE) without crashing.
"""

import unittest
import numpy as np
from src.pipeline.nutrivision_pipeline import NutriVisionPipeline


from src.detection.mock_detector import MockFoodDetector
from src.detection.detector import ModelStatus


class TestPipelineMissingModels(unittest.TestCase):
    def setUp(self):
        self.dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_pipeline_with_unloaded_yolo_returns_detection_model_not_available(self):
        unloaded_det = MockFoodDetector(force_status=ModelStatus.MODEL_NOT_LOADED)
        pipeline = NutriVisionPipeline(detector=unloaded_det)
        result = pipeline.process_image(self.dummy_img)

        # Must not crash; returns structured status
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "DETECTION_MODEL_NOT_AVAILABLE")
        self.assertIn("message", result)
        self.assertIn("foods", result)
        self.assertEqual(len(result["foods"]), 0)


if __name__ == "__main__":
    unittest.main()
