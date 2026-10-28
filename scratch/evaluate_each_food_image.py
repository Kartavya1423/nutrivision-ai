import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from src.pipeline.nutrivision_pipeline import NutriVisionPipeline

pipeline = NutriVisionPipeline()

test_classes = [
    'apple', 'banana', 'orange', 'tomato', 'mango', 
    'pineapple', 'broccoli', 'pizza', 'bread', 'chicken',
    'cucumber', 'french_fries', 'egg', 'strawberry', 'watermelon'
]
tested_samples = {}

for cls_name in test_classes:
    candidates = list(Path(f'data/processed/target_foods_v1/val/{cls_name}').glob('*.*'))
    if not candidates:
        candidates = list(Path(f'data/processed/target_foods_v1/test/{cls_name}').glob('*.*'))
    if not candidates:
        candidates = list(Path('data').glob(f'**/*{cls_name}*.*'))
    candidates = [p for p in candidates if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if candidates:
        tested_samples[cls_name] = candidates[0]

print(f"Testing real sample images for {len(tested_samples)} classes...")
print("=" * 80)

summary_results = []

for ground_truth, img_path in tested_samples.items():
    print(f"\nProcessing Ground Truth: [{ground_truth.upper()}] from {img_path}")
    res = pipeline.process_image(str(img_path))
    items = res.get("items", [])
    if len(items) == 0:
        print(f"  ❌ NO DETECTION: YOLO detector found 0 objects.")
        summary_results.append({
            "food": ground_truth,
            "status": "NO_DETECTION",
            "predicted": "None",
            "confidence": 0.0,
            "weight_g": None,
            "calories": None,
            "match": False
        })
    else:
        for idx, it in enumerate(items):
            food = it.get("food")
            det = it.get("detection", {})
            cls_info = it.get("classification", {})
            wgt = it.get("weight", {})
            nut = it.get("nutrition", {})
            fresh = it.get("freshness", {})
            is_match = (food == ground_truth)
            status_icon = "✅" if is_match else "❌"
            
            c_conf = cls_info.get("confidence", 0.0)
            d_conf = det.get("confidence", 0.0)
            alts = [f"{a.get('class')}: {a.get('confidence', 0)*100:.1f}%" for a in cls_info.get("alternative_suggestions", [])]
            
            print(f"  {status_icon} Predicted: '{food}' (Conf: {c_conf*100:.1f}%) | Ground Truth: '{ground_truth}'")
            print(f"     Detector Box: {det.get('class')} ({d_conf*100:.1f}%) bbox={det.get('bbox')}")
            print(f"     Top Alternatives: {', '.join(alts)}")
            print(f"     Weight Estimate: {wgt.get('value_g')} g ({wgt.get('source')})")
            print(f"     Calories: {nut.get('calories_kcal')} kcal | Matched USDA: {nut.get('matched_food')}")
            print(f"     Freshness: {fresh.get('label')} ({fresh.get('confidence', 0)*100:.1f}%)")
            
            summary_results.append({
                "food": ground_truth,
                "status": "MATCH" if is_match else "MISMATCH",
                "predicted": food,
                "confidence": c_conf,
                "weight_g": wgt.get("value_g"),
                "calories": nut.get("calories_kcal"),
                "match": is_match
            })

print("\n" + "=" * 80)
print("FINAL SUMMARY TABLE")
print("=" * 80)
correct_cnt = sum(1 for r in summary_results if r["match"])
total_cnt = len(summary_results)
print(f"Total Evaluated: {total_cnt} | Correct: {correct_cnt} | Accuracy: {correct_cnt/total_cnt*100:.1f}%\n")
for r in summary_results:
    icon = "✅" if r["match"] else "❌"
    print(f"{icon} {r['food']:15s} -> Predicted: {r['predicted']:15s} (Conf: {r['confidence']*100:5.1f}%) | Weight: {r['weight_g']} g | Cals: {r['calories']} kcal")
