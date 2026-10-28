from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

test_cases = [
    ("Tomato", "data/processed/classification/test/tomato/foodseg103_00004423_c01.jpg"),
    ("Mango", "data/processed/classification/test/mango/foodseg103_00004514_c00.jpg"),
]

for label, rel_path in test_cases:
    full_path = PROJECT_ROOT / rel_path
    with open(full_path, "rb") as f:
        res = requests.post("http://127.0.0.1:8000/analyze", files={"file": f})
    
    data = res.json()
    print(f"\n==================== API RESPONSE FOR {label.upper()} ====================")
    print("HTTP Status Code:", res.status_code)
    print("Pipeline Status:", data.get("status"))
    print("Total Foods Detected:", data.get("total_foods_detected"))
    for idx, item in enumerate(data.get("items", [])):
        print(f"Item {idx}:")
        print("  Food Label:", item.get("food"))
        print("  Confidence:", item.get("classification", {}).get("confidence"))
        print("  Weight:", item.get("weight"))
        nut = item.get("nutrition", {})
        print(f"  Calories: {nut.get('calories_kcal')} kcal")
        print(f"  Protein:  {nut.get('protein_g')} g")
        print(f"  Carbs:    {nut.get('carbohydrates_g')} g")
        print(f"  Fat:      {nut.get('fat_g')} g")
        print(f"  Fiber:    {nut.get('fiber_g')} g")
        print("  Source:   ", nut.get("source"))
    summary = data.get("meal_summary", {})
    print("Meal Summary:")
    print(f"  Total Weight:   {summary.get('total_weight_g')} g")
    print(f"  Total Calories: {summary.get('total_calories_kcal')} kcal")
    print(f"  Total Protein:  {summary.get('total_protein_g')} g")
