"""
Unit tests for end-to-end pipeline orchestration (src.pipeline.pipeline).
Verifies multi-task execution, graceful status propagation, and density/nutrition scaling.
"""

import unittest
import numpy as np

from src.detection.detector import FoodDetection, ModelStatus
from src.detection.mock_detector import MockFoodDetector
from src.pipeline.pipeline import NutriVisionPipeline
from src.segmentation.mock_segmenter import MockFoodSegmenter
from src.segmentation.segmenter import FoodMask


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_pipeline_with_unloaded_models_reports_honest_status(self):
        unloaded_det = MockFoodDetector(status=ModelStatus.MODEL_NOT_LOADED)
        pipeline = NutriVisionPipeline(detector=unloaded_det)
        result = pipeline.analyze_image(self.img)

        # Unloaded models must report honest MODEL_NOT_LOADED status
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertEqual(result.total_items, 0)
        self.assertIn("MODEL_NOT_LOADED", result.stage_statuses["detection"])

    def test_pipeline_with_mock_fixtures(self):
        # Create test fixture detection and mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 20:60] = True

        fixture_det = [
            FoodDetection(
                class_id=1,
                class_name="rice",
                confidence=0.92,
                bbox=[20.0, 20.0, 60.0, 60.0],
            )
        ]
        fixture_mask = [
            FoodMask(
                mask=mask,
                class_name="rice",
                class_id=1,
                confidence=0.95,
                bbox=[20.0, 20.0, 60.0, 60.0],
            )
        ]

        mock_det = MockFoodDetector(fixture_detections=fixture_det)
        mock_seg = MockFoodSegmenter(fixture_masks=fixture_mask)

        pipeline = NutriVisionPipeline(
            detector=mock_det,
            segmenter=mock_seg,
        )

        result = pipeline.analyze_image(self.img, save_artifacts=False)
        self.assertEqual(result.status, ModelStatus.READY)
        self.assertEqual(result.total_items, 1)

        item = result.items[0]
        self.assertEqual(item.class_name, "rice")
        self.assertTrue(item.has_mask)
        self.assertIsNotNone(item.crop_image)
        self.assertGreater(item.weight_grams, 0.0)
        self.assertIsNotNone(item.nutrition)

        # Check meal nutrition report
        self.assertIsNotNone(result.meal_nutrition)
        self.assertGreater(result.meal_nutrition["total_calories"], 0.0)


if __name__ == "__main__":
    unittest.main()
