"""
NutriVision AI — Dataset-Driven Architecture, Food Registry & Nutrition Scaling Tests (Task 7)
Verifies:
- Dataset class discovery and taxonomy synchronization (198 unified classes)
- Dynamic Quick-Select catalog generation
- Authentic USDA FoodData Central nutrition lookup and strict missing-record handling
- Linear weight scaling formulas: nutrient = nutrient_100g * weight_g / 100
- Error handling for unlisted / unsupported classes
- Resilience against missing dataset directories and model checkpoints
"""

import json
from pathlib import Path
import unittest
import yaml

from src.nutrition.food_registry import FoodRegistry, get_food_registry, SupportedFood
from src.nutrition.fooddata_central import FoodDataCentralClient, FOOD_NAME_NORMALIZATION
from src.nutrition.nutrition_service import NutritionService


class TestDatasetDrivenFeatures(unittest.TestCase):
    def setUp(self):
        self.registry = get_food_registry()
        self.nutrition_service = NutritionService()
        self.fdc_client = FoodDataCentralClient()

    def test_taxonomy_class_discovery(self):
        """Verify dynamic discovery of 198 unified classes from data.yaml."""
        classes = self.registry.get_all_taxonomy_classes()
        self.assertGreaterEqual(len(classes), 198)
        self.assertIn("apple", classes)
        self.assertIn("banana", classes)
        self.assertIn("pizza", classes)
        self.assertIn("carrot", classes)

    def test_yolo_and_classification_label_synchronization(self):
        """Verify 100% synchronization between YOLO detection taxonomy and classification ImageFolder classes."""
        yolo_path = Path("data/processed/yolo_detection/data.yaml")
        cls_path = Path("data/processed/classification/train")

        if yolo_path.is_file() and cls_path.is_dir():
            with open(yolo_path, "r", encoding="utf-8") as f:
                yolo_cfg = yaml.safe_load(f)
            yolo_names = set(yolo_cfg.get("names", {}).values())
            cls_folders = set(d.name for d in cls_path.iterdir() if d.is_dir())

            mismatches = yolo_names.symmetric_difference(cls_folders)
            self.assertEqual(len(mismatches), 0, f"Label mismatches found between YOLO and classification: {mismatches}")

    def test_dynamic_quick_select_generation(self):
        """Verify dynamic quick-select generation produces verified foods with unique labels."""
        supported = self.registry.get_supported_foods()
        self.assertGreaterEqual(len(supported), 24, "Expected at least 24 verified supported foods in catalog")

        labels = [f.label for f in supported]
        # Guarantee no duplicate labels
        self.assertEqual(len(labels), len(set(labels)), "Duplicate food labels detected in supported foods")

        # Mandatory staple foods must be present
        keys = [f.key for f in supported]
        for staple in ["pizza", "burger", "salad", "sandwich", "steak", "cake", "ice_cream", "samosa"]:
            self.assertIn(staple, keys, f"Staple food '{staple}' must be available in dynamic catalog")

    def test_search_supported_foods(self):
        """Search functionality filters foods accurately by keyword and prefix."""
        res_berry = self.registry.search_foods("berry")
        self.assertTrue(any("berry" in f.key.lower() for f in res_berry))

        res_piz = self.registry.search_foods("pizza")
        self.assertEqual(len(res_piz), 1)
        self.assertEqual(res_piz[0].key, "pizza")

        # Empty search returns full catalog
        all_foods = self.registry.search_foods("")
        self.assertEqual(len(all_foods), len(self.registry.get_supported_foods()))

    def test_nutrition_lookup_authentic_cache(self):
        """Verify USDA nutrition lookup retrieves authentic records with valid FDC IDs."""
        for food_key in ["pizza", "burger", "salad", "sandwich", "steak", "cake", "ice_cream", "samosa", "apple", "carrot"]:
            nutr = self.nutrition_service.get_nutrition_for_item(food_key, weight_grams=100.0)
            self.assertEqual(nutr.status, "AVAILABLE", f"Nutrition lookup failed for {food_key}")
            self.assertIsNotNone(nutr.usda_fdc_id, f"Missing authentic FDC ID for {food_key}")
            self.assertGreater(nutr.calories_kcal, 0.0, f"Calories must be positive for {food_key}")
            self.assertIsNotNone(nutr.protein_g)
            self.assertIsNotNone(nutr.carbohydrates_g)
            self.assertIsNotNone(nutr.fat_g)

    def test_nutrition_linear_scaling_formula(self):
        """Verify exact nutrient_amount = nutrient_100g * food_weight_g / 100."""
        food = "pizza"
        rep_100 = self.nutrition_service.get_nutrition_for_item(food, weight_grams=100.0)
        base_cals = rep_100.calories_kcal
        base_prot = rep_100.protein_g

        for test_weight in [50.0, 150.0, 200.0, 250.0, 300.0]:
            rep = self.nutrition_service.get_nutrition_for_item(food, weight_grams=test_weight)
            expected_cals = base_cals * (test_weight / 100.0)
            expected_prot = base_prot * (test_weight / 100.0)
            self.assertAlmostEqual(rep.calories_kcal, expected_cals, places=2)
            self.assertAlmostEqual(rep.protein_g, expected_prot, places=2)

    def test_missing_record_handling_strictly_non_hallucinating(self):
        """Unlisted foods return NUTRITION_DATA_NOT_AVAILABLE with zero fake values."""
        bogus_query = "completely_fictional_alien_food_9999"
        nutr = self.nutrition_service.get_nutrition_for_item(bogus_query, weight_grams=100.0)
        self.assertEqual(nutr.status, "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertIsNone(nutr.calories_kcal)
        self.assertIsNone(nutr.protein_g)
        self.assertIsNone(nutr.carbohydrates_g)
        self.assertIsNone(nutr.fat_g)

    def test_missing_dataset_directory_resilience(self):
        """FoodRegistry initializes gracefully even if non-existent root path is provided."""
        empty_reg = FoodRegistry(root_dir=Path("d:/non_existent_empty_path_xyz"))
        # Should not crash; returns empty or fallback gracefully
        self.assertEqual(len(empty_reg.get_all_taxonomy_classes()), 0)
        self.assertEqual(len(empty_reg.get_supported_foods()), 0)
        self.assertIsNone(empty_reg.get_food("pizza"))


if __name__ == "__main__":
    unittest.main()
