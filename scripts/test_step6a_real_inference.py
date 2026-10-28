"""
NutriVision AI — Step 6A Real Inference Verification Script
Tests API endpoints and direct model inference with real food image.
"""

import json
from pathlib import Path
import sys

# Ensure repository root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import requests
import torch

from src.config.settings import get_settings
from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
from src.models.model_manager import get_model_manager

def test_inference():
    cfg = get_settings()
    mm = get_model_manager(settings=cfg)
    status = mm.get_status()
    print("=== Model Manager Diagnostics ===")
    print(json.dumps(status, indent=2))

    # Test Image 1: 1.jpg (known to have detections with the custom trained YOLO11 checkpoint)
    img_path = Path("data/raw/UECFoodPixComplete/data/UECFoodPIXCOMPLETE/train/img/1.jpg")
    if not img_path.exists():
        print(f"Test image not found at {img_path}")
        return

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"\nLoaded test image: {img_path} ({img_rgb.shape})")

    # Direct YOLO detection test at standard conf (0.25) vs tuned conf (0.015)
    print("\n--- Direct YOLO11 Inference (conf=0.25 standard) ---")
    det_res_std = mm.detector.predict(img_rgb, conf_threshold=0.25)
    print(f"YOLO detections at conf 0.25: {len(det_res_std.detections)}")

    print("\n--- Direct YOLO11 Inference (conf=0.015 custom model sensitive) ---")
    det_res_sens = mm.detector.predict(img_rgb, conf_threshold=0.015)
    print(f"YOLO detections at conf 0.015: {len(det_res_sens.detections)}")
    for d in det_res_sens.detections:
        print(f"  Class: {d.class_name} | Conf: {d.confidence:.4f} | Box: {d.bbox}")

    # Full End-to-End Pipeline Test
    print("\n--- Full 10-Stage Pipeline Inference (using conf_threshold=0.015) ---")
    pipe = NutriVisionPipeline(settings=cfg)
    res = pipe.process_image(
        image=img_rgb,
        enable_gradcam=True,
        enable_nutrition=True,
        confidence_threshold=0.015,
    )
    print(f"Pipeline Status: {res.get('status')}")
    print(f"Total Foods: {res.get('summary', {}).get('total_items', 0)}")
    print(f"Total Weight: {res.get('summary', {}).get('total_weight_g')} g")
    print(f"Total Calories: {res.get('summary', {}).get('total_calories')} kcal")

    for idx, food in enumerate(res.get("foods", [])):
        print(f"\nItem #{idx+1}:")
        print(f"  YOLO Bounding Box: {food.get('bbox')}")
        print(f"  YOLO Detected Label: {food.get('label')} (conf: {food.get('yolo_confidence')})")
        print(f"  SAM2 Mask Pixels: {food.get('mask_area_pixels')}")
        print(f"  Classifier Predicted Label: {food.get('classification_label')} (conf: {food.get('classification_confidence')})")
        print(f"  Freshness: {food.get('freshness', {}).get('status')} (conf: {food.get('freshness', {}).get('confidence')})")
        print(f"  Depth Metric Mean: {food.get('depth', {}).get('mean_depth')}")
        print(f"  Estimated Volume: {food.get('volume_cm3')} cm³")
        print(f"  Estimated Weight: {food.get('weight_g')} g ({food.get('weight_status')})")
        print(f"  Nutritional Breakdown: {food.get('nutrition')}")
        print(f"  Grad-CAM generated: {'gradcam_image' in food or 'gradcam' in food}")

    out_file = Path("outputs/step6a_real_inference_output.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved real inference output to: {out_file}")

if __name__ == "__main__":
    test_inference()
