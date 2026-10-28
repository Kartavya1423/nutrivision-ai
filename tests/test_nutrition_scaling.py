"""
Unit tests for nutrition service and weight-based scaling (src.nutrition).
"""

import unittest
from src.nutrition.fooddata_central import FoodDataCentralClient
from src.nutrition.nutrition_service import NutritionService


class TestNutritionScaling(unittest.TestCase):
    def test_unconfigured_api_reports_status(self):
        # Empty API key without cache reports USDA_API_KEY_MISSING
        client = FoodDataCentralClient(api_key="")
        res = client.search_food("uncached_item_9999", use_cache=False)
        self.assertEqual(res["status"], "NUTRITION_DATA_NOT_AVAILABLE")
        self.assertEqual(res["error_code"], "USDA_API_KEY_MISSING")
        self.assertIsNone(res["nutrients_per_100g"])

    def test_weight_based_scaling_logic(self):
        # Create a mock FDC client that returns known profile
        class MockFDC:
            is_configured = True
            def search_food(self, query):
                return {
                    "status": "AVAILABLE",
                    "food_name": query,
                    "nutrients_per_100g": {
                        "calories": 130.0,
                        "protein_g": 2.7,
                        "carbs_g": 28.2,
                        "fat_g": 0.3,
                        "fiber_g": 0.4,
                    },
                    "source": "Mock USDA",
                }

        svc = NutritionService(fdc_client=MockFDC())
        # 200g of rice = 2.0x nutrients
        nutrients = svc.get_nutrition_for_item("rice", weight_grams=200.0)
        d = nutrients.to_dict()

        self.assertEqual(d["status"], "AVAILABLE")
        self.assertEqual(d["weight_grams"], 200.0)
        self.assertEqual(d["calories"], 260.0)
        self.assertAlmostEqual(d["protein_grams"], 5.4, places=1)
        self.assertAlmostEqual(d["carbohydrates_grams"], 56.4, places=1)


if __name__ == "__main__":
    unittest.main()
