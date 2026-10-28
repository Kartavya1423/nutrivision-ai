"""
NutriVision AI — Food Nutrition Lookup Route
Supports authentic USDA FoodData Central retrieval, portion scaling,
and recipe/preparation variant selection.
"""

from typing import Optional
try:
    from fastapi import APIRouter, Query
    from fastapi.responses import JSONResponse
    router = APIRouter(prefix="/nutrition", tags=["Nutrition"])
except ImportError:
    router = None

from src.nutrition.nutrition_service import NutritionService
from src.nutrition.fooddata_central import FoodDataCentralClient


if router is not None:
    @router.get("/variants/{food_name}")
    def get_variants(food_name: str):
        """Retrieve available recipe and preparation variants for a food class."""
        variants = FoodDataCentralClient.get_available_variants(food_name)
        return JSONResponse(content={
            "food_name": food_name,
            "has_variants": len(variants) > 0,
            "variants": variants,
        })

    @router.get("/{food_name}")
    def lookup_nutrition(
        food_name: str,
        weight_grams: float = Query(100.0, description="Portion weight in grams"),
        variant: Optional[str] = Query(None, description="Recipe or preparation variant key"),
        weight_source: str = Query("user_measured", description="Portion weight source: user_measured or vision_estimated"),
    ):
        """Retrieve nutritional breakdown scaled by weight in grams with variant support."""
        svc = NutritionService()
        nutrients = svc.get_nutrition_for_item(
            food_name=food_name,
            weight_grams=weight_grams,
            variant=variant,
            weight_source=weight_source,
        )
        return JSONResponse(content=nutrients.to_dict())
