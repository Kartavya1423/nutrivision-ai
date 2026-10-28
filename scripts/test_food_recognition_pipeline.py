"""
NutriVision AI — Integrated Food Recognition Pipeline CLI (Phase 4)
Orchestrates: Input Image -> YOLO11 -> SAM 2 -> Food Crop -> EfficientNet-B4 -> Food Class

Usage:
    python scripts/test_food_recognition_pipeline.py --image path/to/plate.jpg
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import get_settings
from src.detection.yolo_detector import YOLO11FoodDetector
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.utils.crop import extract_food_region
from src.utils.image_ops import load_image


def main():
    parser = argparse.ArgumentParser(description="Test full YOLO11 -> SAM2 -> EfficientNet-B4 food pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to input food plate image")
    parser.add_argument("--crop_padding", type=int, default=8, help="Crop padding in pixels")
    parser.add_argument("--save_crops", action="store_true", default=True, help="Save extracted RGBA crops")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save output JSON")
    args = parser.parse_args()

    cfg = get_settings()

    # Find a test image if not provided
    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        # Search for valid sample image in raw datasets
        cands = list((PROJECT_ROOT / "data" / "raw" / "FoodSeg103" / "Images" / "ann_dir" / "test").glob("*.jpg"))
        if not cands:
            cands = list((PROJECT_ROOT / "data" / "raw" / "UECFoodPixComplete" / "data" / "UECFoodPixComplete" / "test" / "img").glob("*.jpg"))
        if cands:
            image_path = cands[0]

    if not image_path or not image_path.is_file():
        print(f"[!] Error: Test image not found. Provide --image <path>")
        sys.exit(1)

    print(f"\n=======================================================")
    print("NutriVision AI — Integrated Food Recognition Pipeline")
    print("YOLO11 Detection -> SAM 2 Segmentation -> EfficientNet-B4 Classification")
    print(f"=======================================================")
    print(f"Input Image: {image_path}")

    # Load components
    det_p = PROJECT_ROOT / "models" / "detection" / "yolo11" / "best.pt"
    cls_p = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "best.pth"
    detector = YOLO11FoodDetector(model_path=det_p if det_p.is_file() else None, settings=cfg)
    segmenter = SAM2FoodSegmenter(settings=cfg)
    classifier = EfficientNetB4Classifier(model_path=cls_p if cls_p.is_file() else None, settings=cfg)

    print(f"Detector Status:      {detector.status.value}")
    print(f"Segmenter Status:     {segmenter.status.value}")
    print(f"Classifier Status:    {classifier.status.value}")

    # 1. Load image
    np_img, pil_img = load_image(image_path)
    h, w = np_img.shape[:2]
    print(f"Image Resolution:     {w}x{h}")

    t_total_0 = time.perf_counter()

    # 2. YOLO11 Detection
    t_det_0 = time.perf_counter()
    det_res = detector.predict(np_img)
    t_det_ms = (time.perf_counter() - t_det_0) * 1000.0

    if not det_res.is_success or len(det_res.detections) == 0:
        print("[!] No food items detected by YOLO11.")
        # Fallback: treat whole image as single region
        from src.detection.detector import BoundingBox, FoodDetection
        boxes = [[0.0, 0.0, float(w), float(h)]]
        detections = [FoodDetection(bbox=[0.0, 0.0, float(w), float(h)], confidence=1.0, class_id=0, class_name="food_item")]
    else:
        boxes = [d.bbox for d in det_res.detections]
        detections = det_res.detections

    print(f"[+] YOLO11 detected {len(detections)} food region(s) in {t_det_ms:.1f} ms")

    # 3. SAM 2 Instance Segmentation
    t_seg_0 = time.perf_counter()
    if segmenter.is_ready():
        seg_res = segmenter.predict_masks(np_img, bboxes=boxes)
        masks = seg_res.masks if seg_res.is_success else []
    else:
        masks = []
    t_seg_ms = (time.perf_counter() - t_seg_0) * 1000.0

    print(f"[+] SAM 2 generated {len(masks)} instance mask(s) in {t_seg_ms:.1f} ms")

    # 4. Extract Food-Only Crops & Run EfficientNet-B4
    crops_dir = cfg.outputs.crops
    crops_dir.mkdir(parents=True, exist_ok=True)

    food_items_report = []

    for idx, det in enumerate(detections):
        print(f"\n---------------------------------------------")
        print(f"Food #{idx + 1}")
        print(f"---------------------------------------------")
        if hasattr(det.bbox, 'x1'):
            bx1, by1, bx2, by2 = float(det.bbox.x1), float(det.bbox.y1), float(det.bbox.x2), float(det.bbox.y2)
        else:
            bx1, by1, bx2, by2 = float(det.bbox[0]), float(det.bbox[1]), float(det.bbox[2]), float(det.bbox[3])

        bbox_str = f"[{int(bx1)}, {int(by1)}, {int(bx2)}, {int(by2)}]"
        print(f"Detection:       SUCCESS (BBox: {bbox_str}, Confidence: {det.confidence:.2f})")

        mask_obj = masks[idx] if idx < len(masks) else None
        crop_path = crops_dir / f"pipeline_food_{idx}_{int(time.time())}.png"

        if mask_obj is not None:
            crop_arr, crop_pil = extract_food_region(
                image=np_img,
                mask=mask_obj,
                padding=args.crop_padding,
                return_rgba=True,
                save_path=crop_path,
            )
            seg_status_str = "SUCCESS"
            print(f"Segmentation:    SUCCESS (Area: {mask_obj.area_pixels} px, Confidence: {mask_obj.confidence:.2f})")
            print(f"Food-Only Crop:  {crop_path.name}")
        else:
            # Crop via bounding box
            x1, y1 = max(0, int(bx1)), max(0, int(by1))
            x2, y2 = min(w, int(bx2)), min(h, int(by2))
            crop_arr = np_img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop_arr)
            crop_pil.save(crop_path)
            seg_status_str = "FALLBACK_BBOX"
            print(f"Segmentation:    FALLBACK_BBOX (No high-confidence SAM 2 mask)")
            print(f"BBox Crop:       {crop_path.name}")

        # Classification
        t_cls_0 = time.perf_counter()
        cls_res = classifier.classify(crop_pil, top_k=5)
        t_cls_ms = (time.perf_counter() - t_cls_0) * 1000.0

        if cls_res.is_success and cls_res.result:
            out = cls_res.result
            print(f"Classification:  SUCCESS in {t_cls_ms:.1f} ms")
            print(f"  Class:         {out.class_name}")
            print(f"  Confidence:    {out.confidence:.4f}")
            print(f"  Status:        {cls_res.confidence_status}")
            print("  Top-3 Candidates:")
            for cand in out.top_k_candidates[:3]:
                print(f"    - {cand['class_name']:<24} {cand['confidence']:.4f}")
        else:
            print(f"Classification:  {cls_res.status.value} ({cls_res.error_message})")

        item_dict = {
            "food_id": idx + 1,
            "detection": {
                "status": "SUCCESS",
                "bbox": [bx1, by1, bx2, by2],
                "confidence": round(float(det.confidence), 4),
            },
            "segmentation": {
                "status": seg_status_str,
                "area_pixels": mask_obj.area_pixels if mask_obj else int((bx2 - bx1) * (by2 - by1)),
            },
            "crop_path": str(crop_path),
            "classification": cls_res.to_dict(),
        }
        food_items_report.append(item_dict)

    total_time_ms = (time.perf_counter() - t_total_0) * 1000.0
    print(f"\n=======================================================")
    print(f"Pipeline Execution Complete in {total_time_ms:.1f} ms")
    print(f"Processed {len(food_items_report)} food item(s)")
    print(f"=======================================================\n")

    # Output JSON summary
    out_json = Path(args.output_json) if args.output_json else PROJECT_ROOT / "outputs" / "classification" / "pipeline_food_recognition_test.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "image": str(image_path),
        "total_inference_time_ms": round(total_time_ms, 2),
        "detector_time_ms": round(t_det_ms, 2),
        "segmenter_time_ms": round(t_seg_ms, 2),
        "foods": food_items_report,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[*] Pipeline report written to: {out_json}")


if __name__ == "__main__":
    main()
