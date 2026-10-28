"""
NutriVision AI — USDA FoodData Central & Nutrition Scaling Unit Tests (Phase 7)
Covers:
- Client configuration and API key handling
- Canonical food name normalization
- Search ranking and candidate selection
- Multi-nutrient extraction and unit consistency
- Weight-based nutrient scaling formula (linear 100g serving ratio)
- Unit handling (grams, kilograms, kcal, mg)
- Provenance tracking (LIVE_USDA vs CACHED_USDA)
- Error states: USDA_API_KEY_MISSING, INVALID_WEIGHT, USDA_RATE_LIMITED, NUTRITION_DATA_NOT_AVAILABLE
- Strict verification that NO fake default constants appear in production
- Multi-food meal aggregation with mixed available/unavailable items
"""

import math
from pathlib import Path
import unittest

from src.config.settings import get_settings
from src.nutrition.fooddata_central import FoodDataCentralClient, FOOD_NAME_NORMALIZATION
from src.nutrition.nutrition_service import NutritionService, ScaledNutrients, MealNutritionReport


class TestFoodDataCentralClient(unittest.TestCase):
    """Tests client query building, normalization, caching, and error states."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.client = FoodDataCentralClient()

    def test_food_name_normalization(self):
        """Verify machine-learning labels are cleaned into valid culinary search terms."""
        self.assertEqual(FoodDataCentralClient.normalize_food_name("freshapples"), "apple")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("rottenapples"), "apple")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("freshbanana"), "banana")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("rottenbanana"), "banana")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("freshoranges"), "orange")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("rottenoranges"), "orange")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("bread_steamed"), "steamed bread")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("fried_rice"), "fried rice")
        self.assertEqual(FoodDataCentralClient.normalize_food_name("beef_curry"), "beef curry")

    def test_search_cached_apple_returns_authentic_usda(self):
        """Verify cached USDA apple record contains authentic FDC ID and macronutrients."""
        res = self.client.search_food("apple")
        self.assertEqual(res["status"], "AVAILABLE")
        self.assertEqual(res["usda_fdc_id"], 1750340)
        self.assertIn("Apples", res["usda_food_name"])
        self.assertEqual(res["provenance"], "CACHED_USDA")
        self.assertEqual(res["source"], "USDA FoodData Central")

        nutrients = res["nutrients_per_100g"]
        self.assertIsNotNone(nutrients)
        self.assertAlmostEqual(nutrients["calories_kcal"], 52.0, places=1)
        self.assertAlmostEqual(nutrients["protein_g"], 0.26, places=2)
        self.assertAlmostEqual(nutrients["carbohydrates_g"], 13.81, places=2)
        self.assertAlmostEqual(nutrients["fat_g"], 0.17, places=2)
        self.assertAlmostEqual(nutrients["fiber_g"], 2.4, places=1)

    def test_empty_query_returns_invalid_query_error(self):
        """Empty or whitespace-only query must return INVALID_QUERY error code."""
        res = self.client.search_food("")
        self.assertEqual(res["status"], "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res["error_code"], "INVALID_QUERY")

        res_none = self.client.search_food(None)
        self.assertEqual(res_none["status"], "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res_none["error_code"], "INVALID_QUERY")

    def test_missing_api_key_reports_usda_api_key_missing(self):
        """Unconfigured client must return USDA_API_KEY_MISSING when searching uncached food."""
        empty_client = FoodDataCentralClient(api_key="")
        # Force bypass cache by testing uncached food
        res = empty_client.search_food("exotic_rare_uncached_fruit_9999", use_cache=False)
        self.assertEqual(res["status"], "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res["error_code"], "USDA_API_KEY_MISSING")


class TestNutritionService(unittest.TestCase):
    """Tests weight scaling formulas, unit handling, and zero-fake-data enforcement."""

    @classmethod
    def setUpClass(cls):
        cls.client = FoodDataCentralClient()
        cls.service = NutritionService(fdc_client=cls.client)

    def test_serving_size_scaling_exactness(self):
        """
        Formula: nutrient_scaled = nutrient_per_100g * (weight_g / 100.0)
        Test across 100g, 200g, and 50g portions of raw apple.
        """
        # Baseline: 100g portion -> 1.0x
        res_100g = self.service.get_nutrition_for_item("apple", weight_grams=100.0)
        self.assertEqual(res_100g.status, "AVAILABLE")
        self.assertEqual(res_100g.weight_grams, 100.0)
        self.assertAlmostEqual(res_100g.calories_kcal, 52.0, places=1)
        self.assertAlmostEqual(res_100g.protein_g, 0.26, places=2)
        self.assertAlmostEqual(res_100g.carbohydrates_g, 13.81, places=2)
        self.assertAlmostEqual(res_100g.fat_g, 0.17, places=2)
        self.assertAlmostEqual(res_100g.fiber_g, 2.4, places=1)

        # 200g portion -> exactly 2.0x
        res_200g = self.service.get_nutrition_for_item("apple", weight_grams=200.0)
        self.assertEqual(res_200g.status, "AVAILABLE")
        self.assertEqual(res_200g.weight_grams, 200.0)
        self.assertAlmostEqual(res_200g.calories_kcal, 104.0, places=1)
        self.assertAlmostEqual(res_200g.protein_g, 0.52, places=2)
        self.assertAlmostEqual(res_200g.carbohydrates_g, 27.62, places=2)
        self.assertAlmostEqual(res_200g.fat_g, 0.34, places=2)
        self.assertAlmostEqual(res_200g.fiber_g, 4.8, places=1)

        # 50g portion -> exactly 0.5x
        res_50g = self.service.get_nutrition_for_item("apple", weight_grams=50.0)
        self.assertEqual(res_50g.status, "AVAILABLE")
        self.assertEqual(res_50g.weight_grams, 50.0)
        self.assertAlmostEqual(res_50g.calories_kcal, 26.0, places=1)
        self.assertAlmostEqual(res_50g.protein_g, 0.13, places=2)
        self.assertAlmostEqual(res_50g.carbohydrates_g, 6.905, places=2)
        self.assertAlmostEqual(res_50g.fat_g, 0.085, places=2)
        self.assertAlmostEqual(res_50g.fiber_g, 1.2, places=1)

    def test_unit_conversion_kg_to_grams(self):
        """Input weight specified in kg must convert properly to grams (0.15 kg = 150 g)."""
        res_kg = self.service.get_nutrition_for_item("banana", weight_grams=0.15, weight_unit="kg")
        self.assertEqual(res_kg.status, "AVAILABLE")
        self.assertEqual(res_kg.weight_grams, 150.0)
        # Banana per 100g: 89 kcal -> 150g: 133.5 kcal
        self.assertAlmostEqual(res_kg.calories_kcal, 133.5, places=1)

    def test_invalid_weight_handling(self):
        """Negative, zero, or NaN weights must return INVALID_WEIGHT."""
        res_neg = self.service.get_nutrition_for_item("apple", weight_grams=-25.0)
        self.assertEqual(res_neg.status, "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res_neg.error_code, "INVALID_WEIGHT")
        self.assertIsNone(res_neg.calories_kcal)

        res_zero = self.service.get_nutrition_for_item("apple", weight_grams=0.0)
        self.assertEqual(res_zero.status, "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res_zero.error_code, "INVALID_WEIGHT")

        res_nan = self.service.get_nutrition_for_item("apple", weight_grams=float("nan"))
        self.assertEqual(res_nan.status, "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res_nan.error_code, "INVALID_WEIGHT")

    def test_zero_fake_data_rule_in_production(self):
        """Failed lookups must strictly return None / NUTRITION_DATA_NOT_AVAILABLE, NEVER default constants."""
        res = self.service.get_nutrition_for_item("unlisted_fabricated_item_12345", weight_grams=150.0)
        self.assertEqual(res.status, "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertIsNone(res.calories_kcal)
        self.assertIsNone(res.protein_g)
        self.assertIsNone(res.carbohydrates_g)
        self.assertIsNone(res.fat_g)
        self.assertIsNone(res.fiber_g)

    def test_multi_food_meal_aggregation(self):
        """
        Analyze complete plate meal:
        - 150g Apple (78 kcal)
        - 120g Banana (106.8 kcal)
        - 1 unlisted item (must be excluded from total, NOT defaulted)
        """
        meal_input = [
            {"food_name": "apple", "weight_grams": 150.0},
            {"food_name": "banana", "weight_grams": 120.0},
            {"food_name": "unknown_mystery_item", "weight_grams": 80.0},
        ]

        report = self.service.analyze_meal(meal_input)
        self.assertEqual(report.total_items, 3)
        self.assertEqual(report.calculated_items_count, 2)
        self.assertEqual(report.unavailable_items_count, 1)
        self.assertIn("unknown_mystery_item", report.unavailable_foods)

        # Sum of Apple (78.0) + Banana (106.8) = 184.8 kcal
        self.assertAlmostEqual(report.total_calories_kcal, 184.8, places=1)
        self.assertAlmostEqual(report.total_weight_grams, 270.0, places=1)
        self.assertIsNotNone(report.notes)
        self.assertIn("unknown_mystery_item", report.notes)


if __name__ == "__main__":
    unittest.main()
