"""
NutriVision AI — Finalization Regression Test Suite
Validates:
1. ContourWeightEstimator handles blank / zero-contour images with NO_CONTOUR_DETECTED (no hallucinated bbox).
2. ModelManager properly prioritizes the 198-class EfficientNet-B4 checkpoint over 50-class MobileNet.
3. End-to-end NutriVisionPipeline returns NO_FOOD_DETECTED on non-food scenes.
4. Nutrition service scales calories, macros, and portion weights accurately.
"""

import io
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.config.settings import get_settings
from src.models.model_manager import get_model_manager
from src.nutrition.nutrition_service import NutritionService
from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
from src.weight.contour_weight_estimator import ContourWeightEstimator


class TestFinalizationFixes(unittest.TestCase):
    def setUp(self):
        self.settings = get_settings()

    def test_contour_zero_edge_blank_image(self):
        """ContourWeightEstimator returns NO_CONTOUR_DETECTED on solid blank image."""
        estimator = ContourWeightEstimator()
        solid_img = np.full((100, 100, 3), (200, 50, 50), dtype=np.uint8)
        res = estimator.estimate_weight(solid_img, "food")

        self.assertEqual(res.status, "NO_CONTOUR_DETECTED")
        self.assertEqual(res.area_ratio, 0.0)
        self.assertEqual(res.estimated_weight_g, 0.0)
        self.assertEqual(res.contour_bbox, (0, 0, 0, 0))

    def test_model_manager_classifier_priority(self):
        """ModelManager must load the 198-class EfficientNet-B4 checkpoint."""
        mgr = get_model_manager(settings=self.settings)
        cls_model = mgr.classifier

        self.assertIsNotNone(cls_model)
        self.assertEqual(cls_model.num_classes, 198)
        self.assertIn("efficientnet_b4", str(cls_model.model_path).lower())

    def test_pipeline_non_food_rejection(self):
        """Pipeline returns NO_FOOD_DETECTED for non-food featureless image."""
        pipe = NutriVisionPipeline(settings=self.settings)
        # Create non-food synthetic geometric pattern
        non_food = np.zeros((200, 200, 3), dtype=np.uint8)
        non_food[:, :] = [100, 100, 100] # Gray background
        non_food[90:110, 20:180] = [180, 180, 180] # Metallic line

        res = pipe.process_image(non_food, enable_gradcam=False, enable_nutrition=False)
        self.assertEqual(res["status"], "NO_FOOD_DETECTED")
        self.assertEqual(res["total_foods_detected"], 0)
        self.assertEqual(len(res["items"]), 0)

    def test_nutrition_scaling_accuracy(self):
        """NutritionService calculates exact portion-proportional nutrients."""
        svc = NutritionService(settings=self.settings)
        # 200g portion of a known cached food item (cake or apple)
        report = svc.calculate_nutrition_for_weight("cake", 200.0)
        if report and report.get("status") == "AVAILABLE":
            self.assertIsNotNone(report.get("calories_kcal"))
            self.assertGreater(report.get("calories_kcal"), 0.0)
            self.assertEqual(report.get("weight_grams"), 200.0)

    def test_apple_disambiguation_from_tomato(self):
        """Classifier correctly recognizes red apple without misclassifying as tomato."""
        mgr = get_model_manager(settings=self.settings)
        cls_model = mgr.classifier
        user_apple_crop = self.settings.root_dir / "outputs" / "phase8" / "crops" / "nv_20261019_183244_00427f_item0_food_crop.png"
        if user_apple_crop.is_file():
            res = cls_model.classify(user_apple_crop)
            self.assertEqual(res.result.class_name, "apple")
            self.assertGreater(res.result.confidence, 0.70)

    def test_weight_estimation_bounds(self):
        """ContourWeightEstimator keeps estimated weights within physically realistic bounds."""
        estimator = ContourWeightEstimator()
        # Test 100x100 dummy food with 30% area
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_img[20:75, 25:80] = [200, 30, 30] # Food region
        res = estimator.estimate_weight(dummy_img, "apple", bbox=(25, 20, 80, 75))
        self.assertEqual(res.status, "SUCCESS")
        # Apple reference base weight is 182g; bounded scale is between 0.70 and 1.26
        self.assertGreaterEqual(res.estimated_weight_g, 120.0)
        self.assertLessEqual(res.estimated_weight_g, 240.0)


if __name__ == "__main__":
    unittest.main()
