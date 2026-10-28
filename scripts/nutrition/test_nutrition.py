"""
NutriVision AI — Single Food Nutrition CLI Test Script (Phase 7)
Usage:
    python scripts/nutrition/test_nutrition.py --food banana --weight 120
    python scripts/nutrition/test_nutrition.py --food apple --weight 150 --unit g
    python scripts/nutrition/test_nutrition.py --food chicken --weight 0.2 --unit kg
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.nutrition.fooddata_central import FoodDataCentralClient
from src.nutrition.nutrition_service import NutritionService


def parse_args():
    parser = argparse.ArgumentParser(description="Test USDA FoodData Central nutritional lookup and weight scaling.")
    parser.add_argument("--food", type=str, required=True, help="Food name or query string (e.g., 'banana', 'apple', 'rice').")
    parser.add_argument("--weight", type=float, required=True, help="Estimated physical weight.")
    parser.add_argument("--unit", type=str, default="g", choices=["g", "kg", "grams"], help="Unit of input weight (default: g).")
    parser.add_argument("--output-dir", type=str, default="outputs/nutrition/tests", help="Directory to save output JSON.")
    parser.add_argument("--no-cache", action="store_true", help="Force live USDA API query without local cache.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = FoodDataCentralClient()
    service = NutritionService(fdc_client=client)

    print(f"\n========================================================")
    print(f" NutriVision AI — USDA Nutritional Analysis (Phase 7)")
    print(f"========================================================")
    print(f" Food Query:   {args.food}")
    print(f" Input Weight: {args.weight} {args.unit}")
    print(f" USDA API Key: {'Configured' if client.is_configured else 'Not Configured'}")
    print(f" Cache Mode:   {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"--------------------------------------------------------")

    result = service.get_nutrition_for_item(
        food_name=args.food,
        weight_grams=args.weight,
        weight_unit=args.unit,
    )

    res_dict = result.to_dict()

    if result.status == "AVAILABLE":
        print(f" STATUS:           {result.status}")
        print(f" USDA Matched:     {result.usda_food_name}")
        print(f" USDA FDC ID:      {result.usda_fdc_id}")
        print(f" Provenance:       {result.provenance}")
        print(f" Scaled Weight:    {result.weight_grams} g")
        print(f"--------------------------------------------------------")
        print(f" Nutrient Breakdown (for {result.weight_grams}g portion):")
        print(f"  • Calories:       {result.calories_kcal} kcal")
        print(f"  • Protein:        {result.protein_g} g")
        print(f"  • Carbohydrates:  {result.carbohydrates_g} g")
        print(f"  • Total Fat:      {result.fat_g} g")
        print(f"  • Dietary Fiber:  {result.fiber_g} g")
        if result.sodium_mg is not None:
            print(f"  • Sodium:         {result.sodium_mg} mg")
        if result.potassium_mg is not None:
            print(f"  • Potassium:      {result.potassium_mg} mg")
        if result.sugars_g is not None:
            print(f"  • Sugars:         {result.sugars_g} g")
    else:
        print(f" STATUS:           {result.status}")
        print(f" Error Code:       {result.error_code}")
        print(f" Message:          {result.message}")

    print(f"========================================================\n")

    # Save output JSON
    clean_food = args.food.lower().replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{clean_food}_{int(args.weight)}g_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    print(f"Result saved to: {out_file}\n")
    return 0 if result.status == "AVAILABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
