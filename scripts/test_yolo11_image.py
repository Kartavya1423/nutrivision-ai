#!/usr/bin/env python3
"""
NutriVision AI — Visual YOLO11 Food Detection & Crop CLI
Phase 2: YOLO11 Food Detection Pipeline

Runs object detection on an arbitrary image, overlays labeled bounding boxes,
generates isolated food-only crops, and saves results to outputs/detection/.
Usage:
    python scripts/test_yolo11_image.py --image <path_to_image>
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.crop_utils import save_detection_crops
from src.detection.yolo_detector import YOLO11FoodDetector


def draw_bounding_boxes(image: Image.Image, detections: List[Any]) -> Image.Image:
    """Draws colored bounding boxes and confidence labels on PIL Image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Color palette
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"]

    for idx, det in enumerate(detections):
        color = colors[idx % len(colors)]
        bbox = det.bbox if hasattr(det, "bbox") else det.get("bbox")
        x1, y1, x2, y2 = [int(round(c)) for c in bbox]
        cls_name = det.class_name if hasattr(det, "class_name") else det.get("class_name", "food")
        conf = det.confidence if hasattr(det, "confidence") else det.get("confidence", 0.0)

        # Draw box rectangle (width=3)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label badge
        label_text = f"{cls_name} {conf:.2f}"
        # Approximate text bounding box
        tw = len(label_text) * 8 + 6
        th = 16
        draw.rectangle([x1, max(0, y1 - th), x1 + tw, max(0, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - th) + 1), label_text, fill="white")

    return annotated


def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — Visual YOLO11 Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/detection/yolo11/phase2_fixed_run/weights/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image dimension")
    parser.add_argument("--pred-dir", type=str, default="outputs/detection/phase2_fixed_predictions", help="Output directory for annotated prediction images")
    parser.add_argument("--crops-dir", type=str, default="outputs/detection/crops", help="Output directory for crops")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        # Check relative to project root
        cand = PROJECT_ROOT / args.image
        if cand.is_file():
            img_path = cand
        else:
            print(f"[ERROR] Image file not found: {args.image}")
            sys.exit(1)

    print("=" * 76)
    print("        NutriVision AI — Visual Food Detection & Crop Pipeline           ")
    print("=" * 76)

    # 1. INITIALIZE DETECTOR
    model_path = PROJECT_ROOT / args.model
    if not model_path.is_file():
        alt = PROJECT_ROOT / "models" / "detection" / "yolo11" / "baseline_run" / "weights" / "best.pt"
        if alt.is_file():
            model_path = alt

    detector = YOLO11FoodDetector(
        model_path=model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device="cpu"
    )

    print(f"\n[Step 1/3] Running YOLO11 detection on: {img_path.name}...")
    res = detector.detect(img_path)

    print(f"  -> Model Status      : {res.status.value}")
    print(f"  -> Inference Time    : {res.inference_time_ms:.1f} ms")
    print(f"  -> Food Items Found  : {res.count}")

    for idx, d in enumerate(res.detections, start=1):
        print(f"     [{idx}] {d.class_name:<20} | Conf: {d.confidence:.2f} | BBox: {[round(c, 1) for c in d.bbox]}")

    # 2. ANNOTATE AND SAVE PREDICTION
    outputs_dir = PROJECT_ROOT / "outputs" / "detection"
    pred_dir = PROJECT_ROOT / args.pred_dir
    pred_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = PROJECT_ROOT / args.crops_dir
    crops_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as orig_pil:
        orig_rgb = orig_pil.convert("RGB")
        annotated_pil = draw_bounding_boxes(orig_rgb, res.detections)
        out_pred_path = pred_dir / f"annotated_{img_path.stem}.jpg"
        annotated_pil.save(out_pred_path, "JPEG", quality=95)

    print(f"\n[Step 2/3] Saved annotated visual to: {out_pred_path.relative_to(PROJECT_ROOT)}")

    # 3. EXTRACT AND SAVE CROPS
    print(f"\n[Step 3/3] Generating food-only crops for Phase 3 (SAM2 / Nutrition)...")
    crop_records = save_detection_crops(
        img_path, res.detections, output_dir=crops_dir, source_image_name=img_path.stem
    )
    print(f"  -> Generated {len(crop_records)} food crop(s) in: {crops_dir.relative_to(PROJECT_ROOT)}")
    for cr in crop_records:
        print(f"     -> {cr['crop_file']} (Size: {cr['crop_width']}x{cr['crop_height']} px)")

    print("\n" + "=" * 76)
    print("      [SUCCESS] FOOD DETECTION & CROPPING PIPELINE COMPLETED!           ")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
