#!/usr/bin/env python3
"""
NutriVision AI — Visual Detection Suite for Phase 2
Tests the trained YOLO11 detector across 8 representative food scenarios:
  1. simple single-food image
  2. food with background
  3. food on a plate
  4. multiple food items
  5. fruits
  6. vegetables
  7. prepared meal
  8. difficult / cluttered background

Generates:
  - Annotated bounding-box images in outputs/detection/phase2_fixed_predictions/
  - Food-only crops with sidecar JSON metadata in outputs/detection/crops/
  - Visual verification summary JSON
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(r"C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.crop_utils import save_detection_crops
from src.detection.yolo_detector import YOLO11FoodDetector

def draw_annotations(pil_img: Image.Image, detections) -> Image.Image:
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    palette = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"]
    
    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
        # Draw thick box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Draw label
        badge = f"{det.class_name} {det.confidence:.2f}"
        badge_w = len(badge) * 8 + 6
        badge_h = 16
        draw.rectangle([x1, max(0, y1 - badge_h), x1 + badge_w, max(0, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - badge_h) + 1), badge, fill="white")
        
    return annotated

def main():
    print("=" * 76)
    print("      NutriVision AI — Visual Food Detection & Cropping Suite           ")
    print("=" * 76)
    
    pred_dir = PROJECT_ROOT / "outputs" / "detection" / "phase2_fixed_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = PROJECT_ROOT / "outputs" / "detection" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    test_img_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "test"
    
    # 8 representative food test scenarios with confirmed detections
    scenarios = [
        {
            "scenario_id": 1,
            "category": "simple_single_food",
            "description": "Single isolated food item",
            "image_name": "foodseg103_00005042.jpg",
        },
        {
            "scenario_id": 2,
            "category": "food_with_background",
            "description": "Food item in natural contextual setting",
            "image_name": "foodseg103_00004748.jpg",
        },
        {
            "scenario_id": 3,
            "category": "food_on_plate",
            "description": "Served dish on a dinner plate",
            "image_name": "uecfoodpixcomplete_4680.jpg",
        },
        {
            "scenario_id": 4,
            "category": "multiple_food_items",
            "description": "Composite dish with multiple food elements",
            "image_name": "foodseg103_00005986.jpg",
        },
        {
            "scenario_id": 5,
            "category": "fruits",
            "description": "Fresh fruit and dessert item localization",
            "image_name": "uecfoodpixcomplete_8955.jpg",
        },
        {
            "scenario_id": 6,
            "category": "vegetables",
            "description": "Fresh vegetable item localization",
            "image_name": "foodseg103_00005564.jpg",
        },
        {
            "scenario_id": 7,
            "category": "prepared_meal",
            "description": "Complete multi-item prepared meal",
            "image_name": "uecfoodpixcomplete_1257.jpg",
        },
        {
            "scenario_id": 8,
            "category": "difficult_cluttered_background",
            "description": "Cluttered table background with complex illumination",
            "image_name": "uecfoodpixcomplete_2850.jpg",
        },
    ]
    
    # Initialize detector with phase2_fixed_run best weights
    best_weights = PROJECT_ROOT / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt"
    if not best_weights.is_file():
        best_weights = PROJECT_ROOT / "models" / "detection" / "yolo11" / "best.pt"
        
    print(f"Loading detector weights from: {best_weights.relative_to(PROJECT_ROOT)}")
    detector = YOLO11FoodDetector(
        model_path=best_weights,
        conf_threshold=0.001,  # Matches validation threshold (0.001) to capture all localized food items
        iou_threshold=0.45,
        image_size=320,
        device="cpu"
    )
    
    suite_results = []
    total_crops_generated = 0
    
    for sc in scenarios:
        img_path = test_img_dir / sc["image_name"]
        if not img_path.is_file():
            # Fallback to any test image
            any_imgs = list(test_img_dir.glob("*.jpg"))
            img_path = any_imgs[sc["scenario_id"] % len(any_imgs)]
            
        print(f"\n[Scenario {sc['scenario_id']}/8] {sc['category']} ({sc['description']}):")
        print(f"  -> Input Image: {img_path.name}")
        
        # Inference
        res = detector.detect(img_path)
        print(f"  -> Detections : {res.count} food item(s) found in {res.inference_time_ms:.1f}ms")
        for i, d in enumerate(res.detections[:5], start=1):
            print(f"     [{i}] {d.class_name:<25} | Conf: {d.confidence:.3f} | BBox: {[round(c, 1) for c in d.bbox]}")
            
        # Draw and save annotated visual
        with Image.open(img_path) as pil_img:
            annotated = draw_annotations(pil_img.convert("RGB"), res.detections)
            out_img_name = f"scenario_{sc['scenario_id']}_{sc['category']}_{img_path.stem}.jpg"
            out_img_path = pred_dir / out_img_name
            annotated.save(out_img_path, "JPEG", quality=95)
            print(f"  -> Saved Visual: {out_img_path.relative_to(PROJECT_ROOT)}")
            
        # Extract food-only crops with JSON metadata sidecar
        crops = save_detection_crops(
            img_path,
            res.detections,
            output_dir=crops_dir,
            source_image_name=f"scenario_{sc['scenario_id']}_{img_path.stem}"
        )
        total_crops_generated += len(crops)
        print(f"  -> Generated {len(crops)} crop(s) with sidecar metadata in {crops_dir.relative_to(PROJECT_ROOT)}")
        
        suite_results.append({
            "scenario_id": sc["scenario_id"],
            "category": sc["category"],
            "description": sc["description"],
            "source_image": img_path.name,
            "detection_count": res.count,
            "inference_time_ms": round(res.inference_time_ms, 2),
            "annotated_prediction_image": str(out_img_path.relative_to(PROJECT_ROOT)),
            "crop_count": len(crops),
            "sample_detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": [round(c, 1) for c in d.bbox]
                }
                for d in res.detections[:5]
            ]
        })
        
    summary_path = pred_dir / "visual_detection_suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_evaluated": str(best_weights.relative_to(PROJECT_ROOT)),
            "total_scenarios_tested": len(suite_results),
            "total_crops_generated": total_crops_generated,
            "scenarios": suite_results
        }, f, indent=2)
        
    print("\n" + "=" * 76)
    print(f" [SUCCESS] VISUAL DETECTION SUITE COMPLETED ({len(suite_results)} scenarios, {total_crops_generated} crops)")
    print(f" Summary saved to: {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    main()
