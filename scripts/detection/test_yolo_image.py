#!/usr/bin/env python3
"""
NutriVision AI — Single-Image YOLO11 Food Detection CLI
Phase 2.6: Run inference, detect food items, draw bounding boxes with class names
and confidence, save annotated predictions, and return structured detection information.

Usage:
    python scripts/detection/test_yolo_image.py --image <path_to_image>
    python scripts/detection/test_yolo_image.py --image <path_to_image> --conf 0.15
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.crop_utils import save_detection_crops
from src.detection.yolo_detector import YOLO11FoodDetector


def draw_bounding_boxes(image: Image.Image, detections: List[Any]) -> Image.Image:
    """Draws colored bounding boxes and confidence labels on PIL Image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

    for idx, det in enumerate(detections):
        color = colors[idx % len(colors)]
        bbox = det.bbox if hasattr(det, "bbox") else det.get("bbox")
        x1, y1, x2, y2 = [int(round(c)) for c in bbox]
        cls_name = det.class_name if hasattr(det, "class_name") else det.get("class_name", "food")
        conf = det.confidence if hasattr(det, "confidence") else det.get("confidence", 0.0)

        # Draw box rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label badge
        label_text = f"{cls_name} {conf:.2f}"
        tw = len(label_text) * 8 + 6
        th = 16
        draw.rectangle([x1, max(0, y1 - th), x1 + tw, max(0, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - th) + 1), label_text, fill="white")

    return annotated


def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — Single-Image YOLO11 Food Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input food image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/detection/yolo11/best.pt",
        help="Path to trained YOLO11 checkpoint",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image dimension")
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="outputs/detection/predictions",
        help="Output directory for annotated prediction images",
    )
    parser.add_argument(
        "--crops-dir",
        type=str,
        default="outputs/detection/crops",
        help="Output directory for detection crops",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        cand = PROJECT_ROOT / args.image
        if cand.is_file():
            img_path = cand
        else:
            print(f"[ERROR] Image file not found: {args.image}")
            sys.exit(1)

    print("=" * 76)
    print("      NutriVision AI — YOLO11 Food Detection Single-Image CLI           ")
    print("=" * 76)

    # 1. Resolve model checkpoint
    model_path = Path(args.model)
    if not model_path.is_file():
        cand = PROJECT_ROOT / args.model
        if cand.is_file():
            model_path = cand
        else:
            alt = PROJECT_ROOT / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt"
            if alt.is_file():
                model_path = alt
            else:
                alt2 = PROJECT_ROOT / "models" / "detection" / "yolo11" / "baseline_best_backup.pt"
                if alt2.is_file():
                    model_path = alt2

    print(f"Loading detector weights from: {model_path.relative_to(PROJECT_ROOT) if model_path.is_relative_to(PROJECT_ROOT) else model_path}")

    detector = YOLO11FoodDetector(
        model_path=model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device="cpu",
    )

    # 2. Run inference
    t0 = time.time()
    res = detector.detect(img_path)
    inf_time_ms = round((time.time() - t0) * 1000, 2)

    print(f"Detection Status: {res.status.value}")
    print(f"Inference Latency: {inf_time_ms} ms")
    print(f"Food items detected: {res.count}")

    # 3. Create prediction output
    pred_dir = PROJECT_ROOT / args.pred_dir
    pred_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        pil_img = im.convert("RGB")
        w, h = pil_img.size

    annotated = draw_bounding_boxes(pil_img, res.detections)
    annotated_path = pred_dir / f"pred_{img_path.name}"
    annotated.save(annotated_path, quality=92)
    print(f"Annotated image saved: {annotated_path.relative_to(PROJECT_ROOT)}")

    # 4. Save crops
    crops_dir = PROJECT_ROOT / args.crops_dir
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_records = save_detection_crops(pil_img, res.detections, crops_dir, img_path.stem)
    print(f"Exported {len(crop_records)} food crops to: {crops_dir.relative_to(PROJECT_ROOT)}")

    # 5. Build structured return dictionary
    structured_output = {
        "image_file": img_path.name,
        "image_path": str(img_path.resolve()),
        "image_dimensions": {"width": w, "height": h},
        "model_checkpoint": str(model_path),
        "status": res.status.value,
        "food_items_count": res.count,
        "inference_latency_ms": inf_time_ms,
        "annotated_image": str(annotated_path.resolve()),
        "detections": [
            {
                "detection_id": idx + 1,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "bbox_xyxy": [round(float(c), 2) for c in d.bbox],
                "bbox_normalized": [
                    round(d.bbox[0] / w, 4),
                    round(d.bbox[1] / h, 4),
                    round(d.bbox[2] / w, 4),
                    round(d.bbox[3] / h, 4),
                ],
                "crop_path": crop_records[idx]["crop_path"] if idx < len(crop_records) else None,
            }
            for idx, d in enumerate(res.detections)
        ],
    }

    # Save sidecar JSON
    sidecar_json = pred_dir / f"pred_{img_path.stem}.json"
    with open(sidecar_json, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2)

    print("\n--- Structured Detection Results ---")
    for d in structured_output["detections"]:
        print(f"  Item #{d['detection_id']}: {d['class_name']} (conf={d['confidence']:.4f}, box={d['bbox_xyxy']})")

    print("\n" + json.dumps(structured_output, indent=2))
    return structured_output


if __name__ == "__main__":
    main()
