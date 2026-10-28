"""
NutriVision AI — Phase 8 Nutrition & Portion Reporting Test Suite
Validates:
1. USDA FoodData Central integration for all 24 target food classes.
2. Linear nutrient scaling per 100g: scaled = (per_100g * weight_g) / 100.0.
3. Five primary nutrients: Calories (kcal), Protein (g), Carbohydrates (g), Fat (g), Fiber (g).
4. Authentic recipe/preparation variant handling for preparation-sensitive foods.
5. Raw vs cooked produce distinctions (spinach, broccoli, tomato, egg, etc.).
6. Portion basis: user-measured vs vision-estimated portion weights.
7. Monocular depth topography caveats (no claim of exact weight from monocular depth alone).
8. Strict zero-fabrication / NUTRITION_DATA_NOT_AVAILABLE enforcement.
"""

import math
import pytest
from src.nutrition.fooddata_central import (
    FoodDataCentralClient,
    FOOD_PREPARATION_VARIANTS,
)
from src.nutrition.nutrition_service import NutritionService, ScaledNutrients
from src.nutrition.food_registry import get_food_registry, SupportedFood


# The 24 target foods
TARGET_FOODS_24 = [
    "tomato", "mango", "apple", "banana", "orange", "strawberry",
    "grape", "watermelon", "pineapple", "broccoli", "cucumber", "spinach",
    "lettuce", "pizza", "hamburger", "bread", "french_fries", "egg",
    "chicken", "paneer", "butter", "cake", "ice_cream", "chocolate"
]


class TestPhase8TargetFoodsNutrition:
    """Verifies that all 24 target foods have authentic USDA records and scale correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fdc = FoodDataCentralClient()
        self.service = NutritionService(fdc_client=self.fdc)

    @pytest.mark.parametrize("food", TARGET_FOODS_24)
    def test_target_food_nutrition_available(self, food):
        """Every target food must resolve to an authentic USDA record with non-null nutrients."""
        res = self.service.get_nutrition_for_item(food, weight_grams=100.0)
        assert res.status == "AVAILABLE", f"Food {food} failed lookup: {res.message}"
        assert res.usda_fdc_id is not None, f"Food {food} missing USDA FDC ID"
        assert res.usda_food_name is not None, f"Food {food} missing USDA Food Name"
        assert res.source == "USDA FoodData Central"
        
        # Five primary metrics must be available
        assert res.calories_kcal is not None and res.calories_kcal >= 0, f"{food}: missing calories"
        assert res.protein_g is not None and res.protein_g >= 0, f"{food}: missing protein"
        assert res.carbohydrates_g is not None and res.carbohydrates_g >= 0, f"{food}: missing carbs"
        assert res.fat_g is not None and res.fat_g >= 0, f"{food}: missing fat"
        assert res.fiber_g is not None and res.fiber_g >= 0, f"{food}: missing fiber"

    @pytest.mark.parametrize("weight_g", [50.0, 100.0, 150.0, 225.5, 350.0])
    def test_linear_scaling_formula(self, weight_g):
        """Test linear scaling formula: scaled = per_100g * (weight_g / 100.0)."""
        food = "apple"
        base = self.service.get_nutrition_for_item(food, weight_grams=100.0)
        scaled = self.service.get_nutrition_for_item(food, weight_grams=weight_g)

        scale_factor = weight_g / 100.0
        assert math.isclose(scaled.calories_kcal, round(base.calories_kcal * scale_factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.protein_g, round(base.protein_g * scale_factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.carbohydrates_g, round(base.carbohydrates_g * scale_factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.fat_g, round(base.fat_g * scale_factor, 3), abs_tol=0.05)
        assert math.isclose(scaled.fiber_g, round(base.fiber_g * scale_factor, 3), abs_tol=0.05)


class TestPhase8PreparationVariants:
    """Verifies recipe and preparation variants for preparation-sensitive foods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fdc = FoodDataCentralClient()
        self.service = NutritionService(fdc_client=self.fdc)

    def test_pizza_variants(self):
        """Test pizza generic vs pepperoni vs meat lovers vs veggie."""
        cheese = self.service.get_nutrition_for_item("pizza", weight_grams=100.0, variant="generic")
        pepperoni = self.service.get_nutrition_for_item("pizza", weight_grams=100.0, variant="pepperoni")
        meat_lovers = self.service.get_nutrition_for_item("pizza", weight_grams=100.0, variant="meat_lovers")
        veggie = self.service.get_nutrition_for_item("pizza", weight_grams=100.0, variant="veggie")

        assert cheese.calories_kcal == 267.0
        assert pepperoni.calories_kcal == 298.0
        assert meat_lovers.calories_kcal == 310.0
        assert veggie.calories_kcal == 220.0
        assert cheese.usda_fdc_id != pepperoni.usda_fdc_id
        assert cheese.is_generic_match is True
        assert pepperoni.is_generic_match is False

    def test_burger_variants(self):
        """Test hamburger generic vs cheeseburger vs veggie burger."""
        generic = self.service.get_nutrition_for_item("burger", weight_grams=100.0, variant="generic")
        cheese = self.service.get_nutrition_for_item("burger", weight_grams=100.0, variant="cheeseburger")
        veggie = self.service.get_nutrition_for_item("burger", weight_grams=100.0, variant="veggie_burger")

        assert generic.calories_kcal == 254.0
        assert cheese.calories_kcal == 297.0
        assert veggie.calories_kcal == 177.0

    def test_french_fries_variants(self):
        """Test french fries fried vs oven baked vs sweet potato."""
        fast_food = self.service.get_nutrition_for_item("french_fries", weight_grams=100.0, variant="generic")
        baked = self.service.get_nutrition_for_item("french_fries", weight_grams=100.0, variant="oven_baked")
        sweet = self.service.get_nutrition_for_item("french_fries", weight_grams=100.0, variant="sweet_potato")

        assert fast_food.calories_kcal == 312.0
        assert baked.calories_kcal == 160.0
        assert sweet.calories_kcal == 280.0

    def test_paneer_dishes_variants(self):
        """Test paneer fresh raw vs palak paneer vs paneer butter masala."""
        raw = self.service.get_nutrition_for_item("paneer", weight_grams=100.0, variant="generic")
        palak = self.service.get_nutrition_for_item("paneer", weight_grams=100.0, variant="palak_paneer")
        masala = self.service.get_nutrition_for_item("paneer", weight_grams=100.0, variant="paneer_butter_masala")

        assert raw.calories_kcal == 289.0
        assert palak.calories_kcal == 101.0
        assert masala.calories_kcal == 235.0

    def test_chicken_variants(self):
        """Test roasted breast vs fried chicken vs chicken curry."""
        roasted = self.service.get_nutrition_for_item("chicken", weight_grams=100.0, variant="generic")
        fried = self.service.get_nutrition_for_item("chicken", weight_grams=100.0, variant="fried")
        curry = self.service.get_nutrition_for_item("chicken", weight_grams=100.0, variant="curry")

        assert roasted.calories_kcal == 165.0
        assert fried.calories_kcal == 260.0
        assert curry.calories_kcal == 135.0

    def test_cake_variants(self):
        """Test yellow cake vs chocolate cake vs cheesecake."""
        yellow = self.service.get_nutrition_for_item("cake", weight_grams=100.0, variant="generic")
        choc = self.service.get_nutrition_for_item("cake", weight_grams=100.0, variant="chocolate_cake")
        cheese = self.service.get_nutrition_for_item("cake", weight_grams=100.0, variant="cheesecake")

        assert yellow.calories_kcal == 389.0
        assert choc.calories_kcal == 371.0
        assert cheese.calories_kcal == 321.0

    def test_chocolate_variants(self):
        """Test milk chocolate vs dark 70-85% vs white chocolate."""
        milk = self.service.get_nutrition_for_item("chocolate", weight_grams=100.0, variant="generic")
        dark = self.service.get_nutrition_for_item("chocolate", weight_grams=100.0, variant="dark_70_85")
        white = self.service.get_nutrition_for_item("chocolate", weight_grams=100.0, variant="white")

        assert milk.calories_kcal == 535.0
        assert dark.calories_kcal == 598.0
        assert white.calories_kcal == 539.0

    def test_ice_cream_variants(self):
        """Test vanilla vs chocolate vs soft serve vs strawberry."""
        vanilla = self.service.get_nutrition_for_item("ice_cream", weight_grams=100.0, variant="generic")
        choc = self.service.get_nutrition_for_item("ice_cream", weight_grams=100.0, variant="chocolate")
        soft = self.service.get_nutrition_for_item("ice_cream", weight_grams=100.0, variant="soft_serve")

        assert vanilla.calories_kcal == 207.0
        assert choc.calories_kcal == 216.0
        assert soft.calories_kcal == 222.0

    def test_butter_vs_ghee_variants(self):
        """Test salted butter vs clarified ghee."""
        butter = self.service.get_nutrition_for_item("butter", weight_grams=100.0, variant="generic")
        ghee = self.service.get_nutrition_for_item("butter", weight_grams=100.0, variant="ghee")

        assert butter.calories_kcal == 717.0
        assert ghee.calories_kcal == 900.0

    def test_raw_vs_cooked_distinctions(self):
        """Test raw vs cooked records for produce (spinach, broccoli, tomato, egg, bread)."""
        # Spinach
        raw_sp = self.service.get_nutrition_for_item("spinach", weight_grams=100.0, variant="generic")
        cooked_sp = self.service.get_nutrition_for_item("spinach", weight_grams=100.0, variant="cooked_boiled")
        assert raw_sp.usda_fdc_id == 168462
        assert cooked_sp.usda_fdc_id == 168463

        # Broccoli
        raw_br = self.service.get_nutrition_for_item("broccoli", weight_grams=100.0, variant="generic")
        cooked_br = self.service.get_nutrition_for_item("broccoli", weight_grams=100.0, variant="cooked_steamed")
        assert raw_br.usda_fdc_id == 170379
        assert cooked_br.usda_fdc_id == 170380

        # Tomato
        raw_tom = self.service.get_nutrition_for_item("tomato", weight_grams=100.0, variant="generic")
        cooked_tom = self.service.get_nutrition_for_item("tomato", weight_grams=100.0, variant="cooked_stewed")
        cherry_tom = self.service.get_nutrition_for_item("tomato", weight_grams=100.0, variant="cherry")
        assert raw_tom.usda_fdc_id == 170457
        assert cooked_tom.usda_fdc_id == 170458
        assert cherry_tom.usda_fdc_id == 170459

        # Egg
        boiled_egg = self.service.get_nutrition_for_item("egg", weight_grams=100.0, variant="generic")
        raw_egg = self.service.get_nutrition_for_item("egg", weight_grams=100.0, variant="raw")
        fried_egg = self.service.get_nutrition_for_item("egg", weight_grams=100.0, variant="fried")
        scrambled_egg = self.service.get_nutrition_for_item("egg", weight_grams=100.0, variant="scrambled")
        assert boiled_egg.calories_kcal == 155.0
        assert raw_egg.calories_kcal == 143.0
        assert fried_egg.calories_kcal == 196.0
        assert scrambled_egg.calories_kcal == 149.0

        # Bread
        white_b = self.service.get_nutrition_for_item("bread", weight_grams=100.0, variant="generic")
        wheat_b = self.service.get_nutrition_for_item("bread", weight_grams=100.0, variant="whole_wheat")
        multi_b = self.service.get_nutrition_for_item("bread", weight_grams=100.0, variant="multigrain")
        assert white_b.calories_kcal == 265.0
        assert wheat_b.calories_kcal == 247.0
        assert multi_b.calories_kcal == 265.0
        assert wheat_b.fiber_g == 6.8
        assert white_b.fiber_g == 2.7


class TestPhase8PortionBasisAndCaveats:
    """Verifies portion weight source labeling and monocular depth disclaimers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fdc = FoodDataCentralClient()
        self.service = NutritionService(fdc_client=self.fdc)

    def test_user_measured_weight_label(self):
        """User-measured weight must be tagged as User-Entered Measured Weight."""
        res = self.service.get_nutrition_for_item(
            "banana",
            weight_grams=118.0,
            weight_source="user_measured",
        )
        assert res.weight_source == "user_measured"
        assert res.weight_label == "User-Entered Measured Weight"
        assert "kitchen scale" in res.weight_caveat.lower() or "user measurement" in res.weight_caveat.lower()

    def test_vision_estimated_weight_label_and_disclaimer(self):
        """Vision-estimated weight must be clearly labeled as an estimate with depth caveat."""
        res = self.service.get_nutrition_for_item(
            "banana",
            weight_grams=118.0,
            weight_source="vision_estimated",
        )
        assert res.weight_source == "vision_estimated"
        assert res.weight_label == "Vision Estimated Portion"
        assert "monocular depth" in res.weight_caveat.lower()
        assert "surface relief" in res.weight_caveat.lower()
        assert "density" in res.weight_caveat.lower()


class TestPhase8StrictZeroFabrication:
    """Verifies that missing or invalid data strictly returns NUTRITION_DATA_NOT_AVAILABLE."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fdc = FoodDataCentralClient()
        self.service = NutritionService(fdc_client=self.fdc)

    def test_nonexistent_food_returns_unavailable(self):
        """Unknown or unmapped food must report NUTRITION_DATA_NOT_AVAILABLE without fabricating numbers."""
        res = self.service.get_nutrition_for_item("unobtainium_snack_xyz", weight_grams=100.0)
        assert res.status == "NUTRITION_DATA_NOT_AVAILABLE"
        assert res.calories_kcal is None
        assert res.protein_g is None
        assert res.carbohydrates_g is None
        assert res.fat_g is None
        assert res.fiber_g is None

    def test_invalid_negative_weight(self):
        """Negative weight must return INVALID_WEIGHT."""
        res = self.service.get_nutrition_for_item("apple", weight_grams=-50.0)
        assert res.status == "NUTRITION_DATA_NOT_AVAILABLE"
        assert res.error_code == "INVALID_WEIGHT"
        assert res.calories_kcal is None

    def test_invalid_zero_weight(self):
        """Zero weight must return INVALID_WEIGHT."""
        res = self.service.get_nutrition_for_item("apple", weight_grams=0.0)
        assert res.status == "NUTRITION_DATA_NOT_AVAILABLE"
        assert res.error_code == "INVALID_WEIGHT"
        assert res.calories_kcal is None
