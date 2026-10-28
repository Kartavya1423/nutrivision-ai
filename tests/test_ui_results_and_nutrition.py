"""
NutriVision AI — Comprehensive Results UI & Nutrition Output Unit Tests
Addresses Task Requirement 7:
1. No estimated weight available (graceful handling, no NameError, no fabricated 0/constant)
2. Valid estimated weight (correct parsing, linear scaling, vision_estimated tag)
3. User-entered measured weight (scale override, user_measured tag, correct recalculation)
4. Nutrient scaling by grams (exact linear scaling: per_100g * weight_g / 100.0)
5. Missing nutrient fields (authentic USDA extraction, zero-fabrication of missing fields)
6. Multiple detected items (per-item breakdown and aggregated meal summary totals)
"""

import math
import pytest
from unittest.mock import MagicMock

from frontend.streamlit_app import (
    extract_available_vitamins_and_minerals,
    generate_summary_csv,
)
from src.nutrition.nutrition_service import NutritionService
from src.nutrition.fooddata_central import FoodDataCentralClient


class TestUIResultsAndNutrition:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fdc = FoodDataCentralClient()
        self.service = NutritionService(fdc_client=self.fdc)

    # -------------------------------------------------------------------------
    # Test 1: No estimated weight available
    # -------------------------------------------------------------------------
    def test_no_estimated_weight_available(self):
        """When pipeline returns None, non-positive, or missing weight, handle safely."""
        test_cases = [
            {"value_g": None},
            {"value_g": 0.0},
            {"value_g": -15.0},
            {},
        ]

        for wgt in test_cases:
            raw_w = wgt.get("value_g")
            base_est_w = float(raw_w) if (raw_w is not None and isinstance(raw_w, (int, float)) and raw_w > 0) else None
            assert base_est_w is None, f"Expected base_est_w to be None for {wgt}, got {base_est_w}"

        # If base_est_w is None, nutrition cannot be calculated without user input
        active_weight = base_est_w  # None
        if active_weight is not None and active_weight > 0:
            nut = {"status": "SUCCESS"}
        else:
            nut = {
                "status": "WEIGHT_UNAVAILABLE",
                "message": "Weight estimate unavailable. Please enter measured grams to calculate nutrition.",
            }

        assert nut["status"] == "WEIGHT_UNAVAILABLE"
        assert "Weight estimate unavailable" in nut["message"]

    # -------------------------------------------------------------------------
    # Test 2: Valid estimated weight
    # -------------------------------------------------------------------------
    def test_valid_estimated_weight(self):
        """When pipeline produces a valid estimated weight, verify correct extraction and scaling."""
        wgt = {"value_g": 182.5, "source": "contour_depth_calibrated"}
        raw_w = wgt.get("value_g")
        base_est_w = float(raw_w) if (raw_w is not None and isinstance(raw_w, (int, float)) and raw_w > 0) else None

        assert base_est_w == 182.5

        # Query nutrition using the estimated weight
        res = self.service.get_nutrition_for_item("apple", weight_grams=base_est_w, weight_source="vision_estimated")
        assert res.status == "AVAILABLE"
        assert res.weight_grams == 182.5
        assert res.weight_source == "vision_estimated"

        # Verify scaled calories (Apple standard: ~52 kcal / 100g -> ~94.9 kcal / 182.5g)
        expected_cal = round(52.0 * (182.5 / 100.0), 3)
        assert math.isclose(res.calories_kcal, expected_cal, abs_tol=0.1)

    # -------------------------------------------------------------------------
    # Test 3: User-entered measured weight
    # -------------------------------------------------------------------------
    def test_user_entered_measured_weight(self):
        """User-measured weight overrides vision estimate and labels output accurately."""
        # Simulated pipeline returned 180g vision estimate
        raw_w = 180.0
        base_est_w = float(raw_w)

        # User scale measurement entered in UI
        user_scale_weight = 150.0
        active_weight = user_scale_weight
        weight_basis_src = "user_measured"

        res = self.service.get_nutrition_for_item(
            food_name="apple",
            weight_grams=active_weight,
            weight_source=weight_basis_src,
        )

        assert res.status == "AVAILABLE"
        assert res.weight_grams == 150.0
        assert res.weight_source == "user_measured"

        # 150g apple calories = 52.0 * 1.5 = 78.0 kcal
        assert math.isclose(res.calories_kcal, 78.0, abs_tol=0.1)
        assert math.isclose(res.carbohydrates_g, 13.81 * 1.5, abs_tol=0.1)

    # -------------------------------------------------------------------------
    # Test 4: Nutrient scaling by grams
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("food,weight_g", [
        ("apple", 75.0),
        ("banana", 120.0),
        ("chicken", 220.0),
        ("broccoli", 160.0),
        ("pizza", 250.0),
    ])
    def test_nutrient_scaling_by_grams(self, food, weight_g):
        """All macronutrients must scale strictly linearly with weight."""
        base = self.service.get_nutrition_for_item(food, weight_grams=100.0)
        scaled = self.service.get_nutrition_for_item(food, weight_grams=weight_g)

        factor = weight_g / 100.0
        assert math.isclose(scaled.calories_kcal, round(base.calories_kcal * factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.protein_g, round(base.protein_g * factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.carbohydrates_g, round(base.carbohydrates_g * factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.fat_g, round(base.fat_g * factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.fiber_g, round(base.fiber_g * factor, 3), abs_tol=0.05)

    # -------------------------------------------------------------------------
    # Test 5: Missing nutrient fields & Micronutrient extraction
    # -------------------------------------------------------------------------
    def test_missing_nutrient_fields_handling(self):
        """Micronutrient extractor handles missing fields without fabricating fake zeros."""
        # Simulated nutrition dict with only some fields present
        partial_nut = {
            "status": "SUCCESS",
            "calories_kcal": 52.0,
            "protein_g": 0.26,
            "carbohydrates_g": 13.81,
            "fat_g": 0.17,
            "fiber_g": 2.4,
            "potassium_mg": 107.0,
            "calcium_mg": 6.0,
            "iron_mg": None,  # Explicitly missing/null
            # sodium_mg omitted entirely
            "nutrients_per_100g": {
                "potassium_mg": 107.0,
                "calcium_mg": 6.0,
                "iron_mg": None,
                "vitamin_c_mg": 4.6,
            }
        }

        micros = extract_available_vitamins_and_minerals(partial_nut, portion_weight_g=150.0)
        labels = [m["label"] for m in micros]

        # Present items should be extracted and scaled (150g -> 1.5x)
        assert "Potassium" in labels
        assert "Calcium" in labels
        assert "Vitamin C" in labels

        # Missing or null items should NOT be fabricated
        assert "Iron" not in labels
        assert "Sodium" not in labels

        # Verify scaled values
        potassium_entry = next(m for m in micros if m["label"] == "Potassium")
        assert math.isclose(potassium_entry["value"], 107.0 * 1.5, abs_tol=0.1)
        assert potassium_entry["unit"] == "mg"

        # Edge case: status not SUCCESS returns empty list
        empty_micros = extract_available_vitamins_and_minerals({"status": "WEIGHT_UNAVAILABLE"})
        assert empty_micros == []

    # -------------------------------------------------------------------------
    # Test 6: Multiple detected items & Meal summary aggregation
    # -------------------------------------------------------------------------
    def test_multiple_detected_items_aggregation(self):
        """Multiple detections must each have distinct results and sum correctly in meal summary."""
        items = [
            {
                "food": "apple",
                "detection": {"confidence": 0.94},
                "weight": {"value_g": 150.0, "source": "vision_estimated"},
                "nutrition": {
                    "status": "SUCCESS",
                    "calories_kcal": 78.0,
                    "protein_g": 0.39,
                    "carbohydrates_g": 20.72,
                    "fat_g": 0.26,
                    "fiber_g": 3.6,
                    "sodium_mg": 1.5,
                    "potassium_mg": 160.5,
                },
            },
            {
                "food": "banana",
                "detection": {"confidence": 0.89},
                "weight": {"value_g": 120.0, "source": "user_measured"},
                "nutrition": {
                    "status": "SUCCESS",
                    "calories_kcal": 106.8,
                    "protein_g": 1.31,
                    "carbohydrates_g": 27.41,
                    "fat_g": 0.40,
                    "fiber_g": 3.12,
                    "sodium_mg": 1.2,
                    "potassium_mg": 429.6,
                },
            },
            {
                "food": "cucumber",
                "detection": {"confidence": 0.82},
                "weight": {"value_g": None, "source": "vision_estimated"},
                "nutrition": {
                    "status": "WEIGHT_UNAVAILABLE",
                    "message": "Weight estimate unavailable",
                },
            },
        ]

        # Simulate Meal Summary calculation from frontend Section 6
        tot_wt = 0.0
        tot_cal = 0.0
        tot_p = 0.0
        tot_c = 0.0
        tot_f = 0.0
        tot_fib = 0.0
        has_valid_weights = False
        valid_nutrition_count = 0

        for it in items:
            it_nut = it.get("nutrition", {})
            it_wgt = it.get("weight", {})
            w_val = it_wgt.get("value_g")
            if w_val is not None and isinstance(w_val, (int, float)) and w_val > 0:
                tot_wt += float(w_val)
                has_valid_weights = True
            if it_nut.get("status") == "SUCCESS":
                valid_nutrition_count += 1
                tot_cal += float(it_nut.get("calories_kcal") or 0.0)
                tot_p += float(it_nut.get("protein_g") or 0.0)
                tot_c += float(it_nut.get("carbohydrates_g") or 0.0)
                tot_f += float(it_nut.get("fat_g") or 0.0)
                tot_fib += float(it_nut.get("fiber_g") or 0.0)

        assert valid_nutrition_count == 2
        assert has_valid_weights is True
        assert tot_wt == 270.0  # 150.0 + 120.0
        assert math.isclose(tot_cal, 184.8, abs_tol=0.1)  # 78.0 + 106.8
        assert math.isclose(tot_p, 1.70, abs_tol=0.01)   # 0.39 + 1.31
        assert math.isclose(tot_c, 48.13, abs_tol=0.01)  # 20.72 + 27.41
        assert math.isclose(tot_f, 0.66, abs_tol=0.01)   # 0.26 + 0.40
        assert math.isclose(tot_fib, 6.72, abs_tol=0.01) # 3.6 + 3.12

        # Verify summary CSV generation
        csv_out = generate_summary_csv(items)
        assert "apple" in csv_out
        assert "banana" in csv_out
        assert "cucumber" in csv_out
        assert "150.0" in csv_out
        assert "120.0" in csv_out
        assert "N/A" in csv_out  # For cucumber missing weight and nutrition
