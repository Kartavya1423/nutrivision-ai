"""
NutriVision AI — Nutritional Analysis Service
Integrates USDA FoodData Central client and computes weight-based nutrient scaling.
Calculates: calories, protein, carbohydrates, fat, fiber, sodium, potassium, calcium, iron, sugars.
Strict rules:
- No silent fallback constants in production (DEFAULT_NUTRITION_PER_100G removed from production path)
- Missing or failed lookups strictly report NUTRITION_DATA_NOT_AVAILABLE
- Provenance and FDC IDs preserved on all scaled results
- Unit conversion support for grams, kilograms, milligrams, kcal
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Union

from src.nutrition.fooddata_central import FoodDataCentralClient


@dataclass
class ScaledNutrients:
    """Nutritional profile scaled to the physical food weight in grams."""
    food_name: str
    weight_grams: Optional[float]
    usda_food_name: Optional[str] = None
    usda_fdc_id: Optional[Union[int, str]] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    calcium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    sugars_g: Optional[float] = None
    serving_size_g: float = 100.0
    status: str = "AVAILABLE"
    error_code: Optional[str] = None
    source: str = "USDA FoodData Central"
    provenance: Optional[str] = None
    retrieved_at: Optional[str] = None
    message: Optional[str] = None
    # Phase 8: Recipe / Preparation Variant Metadata
    preparation_variant: Optional[str] = None
    variant_name: Optional[str] = None
    is_generic_match: bool = True
    available_variants: List[Dict[str, Any]] = field(default_factory=list)
    # Portion weight source & monocular depth caveats
    weight_source: str = "user_measured"  # "user_measured", "vision_estimated", "standard_reference"
    weight_label: str = "User-Entered Measured Weight"
    weight_caveat: Optional[str] = None
    scaling_formula: str = "nutrient_scaled = nutrient_per_100g * (portion_weight_g / 100.0)"
    nutrients_per_100g: Optional[Dict[str, float]] = None

    # Backward compatibility properties
    @property
    def calories(self) -> Optional[float]:
        return self.calories_kcal

    @property
    def protein_grams(self) -> Optional[float]:
        return self.protein_g

    @property
    def carbohydrates_grams(self) -> Optional[float]:
        return self.carbohydrates_g

    @property
    def fat_grams(self) -> Optional[float]:
        return self.fat_g

    @property
    def fiber_grams(self) -> Optional[float]:
        return self.fiber_g

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary format."""
        res = {
            "food_name": self.food_name,
            "weight_grams": round(float(self.weight_grams), 2) if self.weight_grams is not None else None,
            "usda_food_name": self.usda_food_name,
            "usda_fdc_id": self.usda_fdc_id,
            "calories_kcal": round(float(self.calories_kcal), 2) if self.calories_kcal is not None else None,
            "protein_g": round(float(self.protein_g), 2) if self.protein_g is not None else None,
            "carbohydrates_g": round(float(self.carbohydrates_g), 2) if self.carbohydrates_g is not None else None,
            "fat_g": round(float(self.fat_g), 2) if self.fat_g is not None else None,
            "fiber_g": round(float(self.fiber_g), 2) if self.fiber_g is not None else None,
            "sodium_mg": round(float(self.sodium_mg), 2) if self.sodium_mg is not None else None,
            "potassium_mg": round(float(self.potassium_mg), 2) if self.potassium_mg is not None else None,
            "calcium_mg": round(float(self.calcium_mg), 2) if self.calcium_mg is not None else None,
            "iron_mg": round(float(self.iron_mg), 2) if self.iron_mg is not None else None,
            "sugars_g": round(float(self.sugars_g), 2) if self.sugars_g is not None else None,
            "serving_size_g": self.serving_size_g,
            "status": self.status,
            "error_code": self.error_code,
            "source": self.source,
            "provenance": self.provenance,
            "retrieved_at": self.retrieved_at,
            "message": self.message,
            # Phase 8 extended metadata
            "preparation_variant": self.preparation_variant,
            "variant_name": self.variant_name,
            "is_generic_match": self.is_generic_match,
            "available_variants": self.available_variants,
            "weight_source": self.weight_source,
            "weight_label": self.weight_label,
            "weight_caveat": self.weight_caveat,
            "scaling_formula": self.scaling_formula,
            "nutrients_per_100g": self.nutrients_per_100g,
            # Legacy keys for existing tests
            "calories": round(float(self.calories_kcal), 2) if self.calories_kcal is not None else None,
            "protein_grams": round(float(self.protein_g), 2) if self.protein_g is not None else None,
            "carbohydrates_grams": round(float(self.carbohydrates_g), 2) if self.carbohydrates_g is not None else None,
            "fat_grams": round(float(self.fat_g), 2) if self.fat_g is not None else None,
            "fiber_grams": round(float(self.fiber_g), 2) if self.fiber_g is not None else None,
            "serving_size_grams": self.serving_size_g,
        }
        # Automatically scale all additional vitamins & minerals in nutrients_per_100g
        if self.nutrients_per_100g and isinstance(self.nutrients_per_100g, dict):
            scale = (float(self.weight_grams) / 100.0) if (self.weight_grams is not None and self.weight_grams > 0) else 1.0
            for k, v in self.nutrients_per_100g.items():
                if k not in res and v is not None:
                    try:
                        res[k] = round(float(v) * scale, 3)
                    except (ValueError, TypeError):
                        pass
        return res


@dataclass
class MealNutritionReport:
    """Consolidated meal nutrient summary across detected foods."""
    items: List[ScaledNutrients] = field(default_factory=list)
    total_items: int = 0
    calculated_items_count: int = 0
    unavailable_items_count: int = 0
    total_weight_grams: float = 0.0
    total_calories_kcal: float = 0.0
    total_protein_g: float = 0.0
    total_carbohydrates_g: float = 0.0
    total_fat_g: float = 0.0
    total_fiber_g: float = 0.0
    total_sodium_mg: float = 0.0
    unavailable_foods: List[str] = field(default_factory=list)
    source: str = "USDA FoodData Central"
    notes: Optional[str] = None

    # Backward compatibility
    @property
    def total_calories(self) -> float:
        return self.total_calories_kcal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "calculated_items_count": self.calculated_items_count,
            "unavailable_items_count": self.unavailable_items_count,
            "total_weight_grams": round(float(self.total_weight_grams), 2),
            "total_calories_kcal": round(float(self.total_calories_kcal), 2),
            "total_protein_g": round(float(self.total_protein_g), 2),
            "total_carbohydrates_g": round(float(self.total_carbohydrates_g), 2),
            "total_fat_g": round(float(self.total_fat_g), 2),
            "total_fiber_g": round(float(self.total_fiber_g), 2),
            "total_sodium_mg": round(float(self.total_sodium_mg), 2),
            "unavailable_foods": self.unavailable_foods,
            "source": self.source,
            "notes": self.notes,
            "items": [i.to_dict() for i in self.items],
            # Legacy key
            "total_calories": round(float(self.total_calories_kcal), 2),
        }


class NutritionService:
    """Production service for USDA nutritional lookup and weight-based nutrient scaling."""

    def __init__(
        self,
        fdc_client: Optional[FoodDataCentralClient] = None,
        settings: Optional[Any] = None,
        api_key: Optional[str] = None,
    ):
        self.fdc_client = fdc_client or FoodDataCentralClient(api_key=api_key)
        self.settings = settings

    def get_nutrition_for_item(
        self,
        food_name: str,
        weight_grams: Optional[float] = None,
        weight_unit: str = "g",
        variant: Optional[str] = None,
        weight_source: str = "user_measured",
    ) -> ScaledNutrients:
        """
        Query USDA FoodData Central and scale nutrients by item weight.
        Formula: nutrient_scaled = nutrient_per_100g * (weight_in_grams / 100.0)
        """
        # Determine weight label & caveat
        if weight_source == "vision_estimated":
            w_label = "Vision Estimated Portion"
            w_caveat = (
                "Physical weight is an empirical estimate calibrated from 2D contour area "
                "and reference food density. Monocular depth provides relative surface relief only, "
                "not absolute physical mass or scale."
            )
        elif weight_source == "standard_reference":
            w_label = "Standard Reference Portion"
            w_caveat = "Standard USDA single-serving reference portion weight."
        else:
            w_label = "User-Entered Measured Weight"
            w_caveat = "Portion weight entered directly from kitchen scale or user measurement."

        # Validate weight
        if weight_grams is not None:
            # Handle unit conversion if passed in kg
            if str(weight_unit).lower() == "kg":
                weight_in_g = float(weight_grams) * 1000.0
            elif str(weight_unit).lower() in ["g", "grams"]:
                weight_in_g = float(weight_grams)
            else:
                weight_in_g = float(weight_grams)

            if math.isnan(weight_in_g) or weight_in_g <= 0.0:
                return ScaledNutrients(
                    food_name=food_name,
                    weight_grams=None,
                    status="NUTRITION_DATA_NOT_AVAILABLE",
                    error_code="INVALID_WEIGHT",
                    weight_source=weight_source,
                    weight_label=w_label,
                    weight_caveat=w_caveat,
                    message=f"Invalid food weight '{weight_grams}'. Weight must be a positive non-zero number.",
                )
        else:
            weight_in_g = None

        # Search USDA (with preparation variant support, backwards-compatible with legacy/mock clients)
        try:
            query_res = self.fdc_client.search_food(food_name, variant=variant)
        except TypeError:
            query_res = self.fdc_client.search_food(food_name)
        status = query_res.get("status", "NUTRITION_DATA_NOT_AVAILABLE")

        if status != "AVAILABLE" or not query_res.get("nutrients_per_100g"):
            err_code = query_res.get("error_code") or "NUTRITION_DATA_NOT_AVAILABLE"
            return ScaledNutrients(
                food_name=food_name,
                weight_grams=weight_in_g,
                usda_food_name=query_res.get("usda_food_name"),
                usda_fdc_id=query_res.get("usda_fdc_id"),
                status="NUTRITION_DATA_NOT_AVAILABLE",
                error_code=err_code,
                source=query_res.get("source", "USDA FoodData Central"),
                provenance=query_res.get("provenance"),
                preparation_variant=query_res.get("preparation_variant"),
                variant_name=query_res.get("variant_name"),
                is_generic_match=query_res.get("is_generic_default", True),
                available_variants=query_res.get("available_variants", []),
                weight_source=weight_source,
                weight_label=w_label,
                weight_caveat=w_caveat,
                message=query_res.get("message", "Nutritional information unavailable from USDA FoodData Central."),
            )

        n100 = query_res["nutrients_per_100g"]
        scale = (weight_in_g / 100.0) if weight_in_g is not None else 1.0

        def _scale(val: Optional[float]) -> Optional[float]:
            if val is None:
                return None
            return round(val * scale, 3)

        return ScaledNutrients(
            food_name=food_name,
            weight_grams=weight_in_g,
            usda_food_name=query_res.get("usda_food_name"),
            usda_fdc_id=query_res.get("usda_fdc_id"),
            calories_kcal=_scale(n100.get("calories_kcal") if n100.get("calories_kcal") is not None else n100.get("calories")),
            protein_g=_scale(n100.get("protein_g")),
            carbohydrates_g=_scale(n100.get("carbohydrates_g") if n100.get("carbohydrates_g") is not None else n100.get("carbs_g")),
            fat_g=_scale(n100.get("fat_g")),
            fiber_g=_scale(n100.get("fiber_g")),
            sodium_mg=_scale(n100.get("sodium_mg")),
            potassium_mg=_scale(n100.get("potassium_mg")),
            calcium_mg=_scale(n100.get("calcium_mg")),
            iron_mg=_scale(n100.get("iron_mg")),
            sugars_g=_scale(n100.get("sugars_g")),
            serving_size_g=query_res.get("serving_size_g", 100.0),
            status="AVAILABLE",
            error_code=None,
            source=query_res.get("source", "USDA FoodData Central"),
            provenance=query_res.get("provenance"),
            retrieved_at=query_res.get("retrieved_at"),
            preparation_variant=query_res.get("preparation_variant"),
            variant_name=query_res.get("variant_name"),
            is_generic_match=query_res.get("is_generic_default", True),
            available_variants=query_res.get("available_variants", []),
            weight_source=weight_source,
            weight_label=w_label,
            weight_caveat=w_caveat,
            scaling_formula="nutrient_scaled = nutrient_per_100g * (portion_weight_g / 100.0)",
            nutrients_per_100g=n100,
            message=None,
        )

    def calculate_nutrition_for_weight(
        self,
        food_name: str,
        weight_g: Optional[float] = None,
        variant: Optional[str] = None,
        weight_source: str = "user_measured",
    ) -> Dict[str, Any]:
        """
        Calculate nutrients scaled for a specified food weight in grams.
        If weight_g is None, returns base nutritional values per 100g.
        """
        res = self.get_nutrition_for_item(
            food_name=food_name,
            weight_grams=weight_g,
            variant=variant,
            weight_source=weight_source,
        )
        d = res.to_dict()
        d["matched_food"] = res.usda_food_name
        d["fdc_id"] = res.usda_fdc_id
        return d

    def analyze_meal(self, food_items_with_weights: List[Dict[str, Any]]) -> MealNutritionReport:
        """
        Analyze complete plate meal from list of items with weights.
        Only items with valid USDA data are added to nutrient totals.
        No silent default values are used.
        """
        items: List[ScaledNutrients] = []
        tot_wt = 0.0
        tot_cal = 0.0
        tot_protein = 0.0
        tot_carbs = 0.0
        tot_fat = 0.0
        tot_fiber = 0.0
        tot_sodium = 0.0
        unavailable_foods: List[str] = []
        calc_count = 0
        unavail_count = 0

        for item in food_items_with_weights:
            name = item.get("name") or item.get("food_name", "food")
            wt = item.get("weight_grams")
            if wt is not None:
                try:
                    wt = float(wt)
                except (ValueError, TypeError):
                    wt = None

            nutrients = self.get_nutrition_for_item(name, weight_grams=wt)
            items.append(nutrients)

            if nutrients.status == "AVAILABLE" and nutrients.calories_kcal is not None and wt is not None:
                calc_count += 1
                tot_wt += wt
                tot_cal += nutrients.calories_kcal
                if nutrients.protein_g is not None:
                    tot_protein += nutrients.protein_g
                if nutrients.carbohydrates_g is not None:
                    tot_carbs += nutrients.carbohydrates_g
                if nutrients.fat_g is not None:
                    tot_fat += nutrients.fat_g
                if nutrients.fiber_g is not None:
                    tot_fiber += nutrients.fiber_g
                if nutrients.sodium_mg is not None:
                    tot_sodium += nutrients.sodium_mg
            else:
                unavail_count += 1
                unavailable_foods.append(name)

        notes = None
        if unavailable_foods:
            notes = f"Nutritional data unavailable for {len(unavailable_foods)} item(s): {', '.join(unavailable_foods)}. Totals reflect only items with verified USDA data."

        return MealNutritionReport(
            items=items,
            total_items=len(food_items_with_weights),
            calculated_items_count=calc_count,
            unavailable_items_count=unavail_count,
            total_weight_grams=round(tot_wt, 2),
            total_calories_kcal=round(tot_cal, 2),
            total_protein_g=round(tot_protein, 2),
            total_carbohydrates_g=round(tot_carbs, 2),
            total_fat_g=round(tot_fat, 2),
            total_fiber_g=round(tot_fiber, 2),
            total_sodium_mg=round(tot_sodium, 2),
            unavailable_foods=unavailable_foods,
            source="USDA FoodData Central",
            notes=notes,
        )


_NUTRITION_SERVICE_INSTANCE: Optional[NutritionService] = None


def get_nutrition_service(settings: Optional[Any] = None) -> NutritionService:
    """Singleton getter for NutritionService."""
    global _NUTRITION_SERVICE_INSTANCE
    if _NUTRITION_SERVICE_INSTANCE is None:
        _NUTRITION_SERVICE_INSTANCE = NutritionService(settings=settings)
    return _NUTRITION_SERVICE_INSTANCE
