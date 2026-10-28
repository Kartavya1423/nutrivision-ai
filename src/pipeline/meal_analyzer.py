"""
NutriVision AI — Meal Analysis Module
Aggregates multi-item meal analysis across detection, segmentation, classification,
volume/weight estimation, and nutritional queries.
Output schema matches requirements:
{
  "foods": [],
  "total_weight_grams": null,
  "total_calories": null,
  "protein_grams": null,
  "carbohydrates_grams": null,
  "fat_grams": null,
  "fiber_grams": null,
  "status": "MODEL_NOT_READY"
}
Does NOT fabricate values.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MealAnalysisReport:
    """Full meal aggregate summary matching specification."""
    foods: List[Dict[str, Any]] = field(default_factory=list)
    total_weight_grams: Optional[float] = None
    total_calories: Optional[float] = None
    protein_grams: Optional[float] = None
    carbohydrates_grams: Optional[float] = None
    fat_grams: Optional[float] = None
    fiber_grams: Optional[float] = None
    status: str = "MODEL_NOT_READY"
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "foods": self.foods,
            "total_weight_grams": round(float(self.total_weight_grams), 1) if self.total_weight_grams is not None else None,
            "total_calories": round(float(self.total_calories), 1) if self.total_calories is not None else None,
            "protein_grams": round(float(self.protein_grams), 1) if self.protein_grams is not None else None,
            "carbohydrates_grams": round(float(self.carbohydrates_grams), 1) if self.carbohydrates_grams is not None else None,
            "fat_grams": round(float(self.fat_grams), 1) if self.fat_grams is not None else None,
            "fiber_grams": round(float(self.fiber_grams), 1) if self.fiber_grams is not None else None,
            "status": self.status,
            "message": self.message,
        }


class MealAnalyzer:
    """Combines individual food detections into a cohesive meal nutritional report."""

    def aggregate_meal(
        self,
        food_items: List[Dict[str, Any]],
        pipeline_status: str = "MODEL_NOT_READY",
    ) -> MealAnalysisReport:
        """
        Aggregate nutrient totals across all food items on a plate.
        If pipeline status is not fully READY, returns status MODEL_NOT_READY with nulls.
        """
        if pipeline_status != "READY" or not food_items:
            return MealAnalysisReport(
                foods=food_items,
                total_weight_grams=None,
                total_calories=None,
                protein_grams=None,
                carbohydrates_grams=None,
                fat_grams=None,
                fiber_grams=None,
                status="MODEL_NOT_READY",
                message="Models not yet trained or calibrated. Returning MODEL_NOT_READY.",
            )

        tot_weight = 0.0
        tot_cal = 0.0
        tot_protein = 0.0
        tot_carbs = 0.0
        tot_fat = 0.0
        tot_fiber = 0.0

        for item in food_items:
            nutr = item.get("nutrition", {})
            w = item.get("estimated_weight_grams")
            if w is not None:
                tot_weight += float(w)

            if nutr and nutr.get("status") == "READY":
                if nutr.get("calories") is not None:
                    tot_cal += float(nutr["calories"])
                if nutr.get("protein_grams") is not None:
                    tot_protein += float(nutr["protein_grams"])
                if nutr.get("carbohydrates_grams") is not None:
                    tot_carbs += float(nutr["carbohydrates_grams"])
                if nutr.get("fat_grams") is not None:
                    tot_fat += float(nutr["fat_grams"])
                if nutr.get("fiber_grams") is not None:
                    tot_fiber += float(nutr["fiber_grams"])

        return MealAnalysisReport(
            foods=food_items,
            total_weight_grams=tot_weight,
            total_calories=tot_cal,
            protein_grams=tot_protein,
            carbohydrates_grams=tot_carbs,
            fat_grams=tot_fat,
            fiber_grams=tot_fiber,
            status="READY",
        )
