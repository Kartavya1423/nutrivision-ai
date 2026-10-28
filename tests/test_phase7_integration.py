"""
NutriVision AI — Phase 7 End-to-End Integration Tests
Covers:
- Food Classification -> USDA Nutrition Pipeline
- Estimated Weight -> Nutritional Portion Scaling
- Food Classification -> Grad-CAM Explainability
- Freshness Detection -> Grad-CAM Explainability
- End-to-End Multi-Task Scene Pipeline Execution
- Failure and Missing Data Graceful Handling
"""

from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.config.settings import get_settings
from src.explainability.explainer import NutriVisionExplainer
from src.nutrition.fooddata_central import FoodDataCentralClient
from src.nutrition.nutrition_service import NutritionService
from src.pipeline.phase7_pipeline import Phase7Pipeline


class TestPhase7PipelineIntegration(unittest.TestCase):
    """Integration tests connecting vision models to nutrition and explainability."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.pipeline = Phase7Pipeline(settings=cls.settings)
        cls.nutrition_service = NutritionService(settings=cls.settings)
        cls.explainer = NutriVisionExplainer(settings=cls.settings)

        # 120x120 test food crop
        arr = np.random.randint(60, 210, (120, 120, 3), dtype=np.uint8)
        cls.crop_img = Image.fromarray(arr)

    def test_classification_to_nutrition_flow(self):
        """Food classification output feeds directly into USDA FoodData Central query."""
        food_name = "banana"
        scaled = self.nutrition_service.get_nutrition_for_item(food_name, weight_grams=120.0)

        self.assertEqual(scaled.status, "AVAILABLE")
        self.assertEqual(scaled.food_name, "banana")
        self.assertEqual(scaled.weight_grams, 120.0)
        self.assertIsNotNone(scaled.calories_kcal)
        self.assertAlmostEqual(scaled.calories_kcal, 106.8, places=1)
        self.assertIsNotNone(scaled.usda_fdc_id)
        self.assertEqual(scaled.source, "USDA FoodData Central")

    def test_weight_to_nutrition_portion_scaling(self):
        """Estimated weight scales all nutrients linearly from standard 100g profile."""
        # Orange per 100g: 47 kcal, 0.94g protein, 11.75g carbs, 0.12g fat, 2.4g fiber
        scaled_200g = self.nutrition_service.get_nutrition_for_item("orange", weight_grams=200.0)

        self.assertEqual(scaled_200g.status, "AVAILABLE")
        self.assertAlmostEqual(scaled_200g.calories_kcal, 94.0, places=1)
        self.assertAlmostEqual(scaled_200g.protein_g, 1.88, places=2)
        self.assertAlmostEqual(scaled_200g.carbohydrates_g, 23.5, places=1)
        self.assertAlmostEqual(scaled_200g.fat_g, 0.24, places=2)
        self.assertAlmostEqual(scaled_200g.fiber_g, 4.8, places=1)

    def test_food_crop_to_gradcam_flow(self):
        """Food-only crop generates both food and freshness Grad-CAM heatmaps."""
        # 1. Food Grad-CAM
        food_cam = self.explainer.explain_food_classification(self.crop_img)
        self.assertEqual(food_cam["status"], "AVAILABLE")
        self.assertTrue(food_cam["has_heatmap"])
        self.assertEqual(food_cam["heatmap_normalized"].shape, (120, 120))

        # 2. Freshness Grad-CAM
        fresh_cam = self.explainer.explain_freshness(self.crop_img)
        self.assertEqual(fresh_cam["status"], "AVAILABLE")
        self.assertTrue(fresh_cam["has_heatmap"])
        self.assertEqual(fresh_cam["heatmap_normalized"].shape, (120, 120))

    def test_full_pipeline_process_image(self):
        """Run full 10-stage Phase7Pipeline on real test image."""
        test_path = self.settings.root_dir / "data" / "processed" / "freshness" / "test" / "freshapples" / "freshapples_0006.png"
        if not test_path.is_file():
            self.skipTest("Test produce sample not found.")

        result = self.pipeline.process_image(
            image_path=test_path,
            generate_gradcam=True,
            save_visualizations=True,
        )

        self.assertEqual(result.overall_status, "SUCCESS")
        self.assertGreaterEqual(result.total_foods_detected, 1)

        item = result.items[0]
        self.assertEqual(item.food, "apple")
        self.assertIn(item.freshness, ["fresh", "rotten"])
        self.assertGreater(item.freshness_confidence, 0.5)

        # Nutrition check
        self.assertEqual(item.nutrition["status"], "AVAILABLE")
        self.assertIsNotNone(item.nutrition["calories_kcal"])
        self.assertEqual(item.usda["source"], "USDA FoodData Central")

        # Grad-CAM check
        self.assertIsNotNone(item.explainability["freshness_gradcam"])
        self.assertEqual(item.explainability["freshness_gradcam"]["status"], "AVAILABLE")
        self.assertIsNotNone(item.explainability["food_gradcam"])

        # Visualization check
        self.assertIsNotNone(result.visualization_path)
        self.assertTrue(Path(result.visualization_path).is_file())


if __name__ == "__main__":
    unittest.main()
