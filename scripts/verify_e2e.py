import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.nutrivision_pipeline import NutriVisionPipeline

pipe = NutriVisionPipeline()

test_cases = [
    ("Tomato", "data/processed/classification/test/tomato/foodseg103_00004423_c01.jpg"),
    ("Mango", "data/processed/classification/test/mango/foodseg103_00004514_c00.jpg"),
    ("Apple", "data/processed/classification/test/apple/foodseg103_00004416_c00.jpg"),
]

for label, img_path in test_cases:
    print(f"\n==================== TESTING {label.upper()} ====================")
    full_path = PROJECT_ROOT / img_path
    if not full_path.exists():
        print(f"File not found: {full_path}")
        continue
    res = pipe.process_image(str(full_path))
    print("Pipeline Status:", res.get("status"))
    print("Total Foods Detected:", res.get("total_foods_detected"))
    for idx, item in enumerate(res.get("items", [])):
        print(f"Item {idx}:")
        print("  Food Label:", item.get("food"))
        print("  Classification Conf:", item.get("classification", {}).get("confidence"))
        print("  Estimated Weight:", item.get("weight"))
        nut = item.get("nutrition", {})
        print(f"  Calories: {nut.get('calories_kcal')} kcal")
        print(f"  Protein:  {nut.get('protein_g')} g")
        print(f"  Carbs:    {nut.get('carbohydrates_g')} g")
        print(f"  Fat:      {nut.get('fat_g')} g")
        print(f"  Fiber:    {nut.get('fiber_g')} g")
        print("  Source:   ", nut.get("source"))
