#!/usr/bin/env python3
"""
NutriVision AI — Visual YOLO11 → SAM 2 Segmentation CLI
Phase 3: SAM 2 Food Instance Segmentation

Runs YOLO11 food detection, passes bounding box prompts to SAM 2,
generates pixel-accurate food masks, isolates food regions with transparent
RGBA background removal, exports artifacts and sidecar JSON metadata,
and produces a multi-panel visual comparison.

Usage:
    python scripts/test_sam2_image.py --image <path_to_image>
    python scripts/test_sam2_image.py --image <path_to_image> --conf 0.15
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.detection.yolo_detector import YOLO11FoodDetector
from src.segmentation.crop_export import save_segmentation_artifacts
from src.segmentation.pipeline import YOLOToSAMPipeline
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.visualizer import create_segmentation_comparison_panel
from src.utils.image_ops import load_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="NutriVision AI — YOLO11 → SAM 2 Food Instance Segmentation CLI"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input food image")
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/detection/yolo11/best.pt",
        help="Path to trained YOLO11 detector weights",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="models/segmentation/sam2.1_t.pt",
        help="Path to SAM 2 checkpoint weights",
    )
    parser.add_argument("--conf", type=float, default=0.0001, help="YOLO11 detection confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Inference device (auto, cpu, cuda)")
    parser.add_argument("--crop-padding", type=int, default=10, help="Pixel padding around food crops")
    parser.add_argument("--output-dir", type=str, default="outputs/segmentation", help="Directory for output artifacts")
    parser.add_argument("--boxes", type=str, default=None, help="Optional manual boxes '[[x1,y1,x2,y2],...]' override")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()

    img_path = Path(args.image)
    if not img_path.is_file():
        cand = PROJECT_ROOT / args.image
        if cand.is_file():
            img_path = cand
        else:
            print(f"[ERROR] Image file not found: {args.image}")
            sys.exit(1)

    print("=" * 76)
    print("      NutriVision AI — Phase 3: SAM 2 Food Instance Segmentation        ")
    print("=" * 76)

    # 1. Resolve model paths
    det_model_path = Path(args.detector_model)
    if not det_model_path.is_file():
        cand = PROJECT_ROOT / args.detector_model
        if cand.is_file():
            det_model_path = cand
        else:
            alt = PROJECT_ROOT / "models" / "detection" / "yolo11" / "baseline_run" / "weights" / "best.pt"
            if alt.is_file():
                det_model_path = alt

    sam_model_path = Path(args.sam_model)
    if not sam_model_path.is_file():
        cand = PROJECT_ROOT / args.sam_model
        if cand.is_file():
            sam_model_path = cand
        else:
            cand2 = PROJECT_ROOT / "sam2.1_t.pt"
            if cand2.is_file():
                sam_model_path = cand2

    print(f"\n[Configuration]")
    print(f"  Input Image       : {img_path.name}")
    print(f"  YOLO11 Detector   : {det_model_path.name} (exists={det_model_path.is_file()})")
    print(f"  SAM 2 Checkpoint  : {sam_model_path.name} (exists={sam_model_path.is_file()})")
    print(f"  Device Requested  : {args.device}")
    print(f"  Confidence Thresh : {args.conf}")

    # 2. Initialize Models
    detector = YOLO11FoodDetector(
        model_path=det_model_path,
        conf_threshold=args.conf,
        iou_threshold=0.45,
        image_size=640,
        device="cpu" if args.device == "cpu" else None,
    )
    segmenter = SAM2FoodSegmenter(
        checkpoint_path=sam_model_path,
        device="cpu" if args.device == "cpu" else None,
        min_mask_area=50,
        box_margin=0.05,
        apply_postprocessing=True,
    )

    # Check manual boxes
    manual_boxes = None
    if args.boxes:
        try:
            manual_boxes = json.loads(args.boxes)
            print(f"  Manual Prompt Boxes: {len(manual_boxes)} boxes specified.")
        except Exception as e:
            print(f"  [WARNING] Failed to parse manual boxes: {e}")

    # 3. Execute Detection or Manual Box Prompting
    print(f"\n[Step 1/4] Running Food Detection...")
    np_img, _ = load_image(img_path)
    h, w = np_img.shape[:2]

    if manual_boxes:
        bboxes = manual_boxes
        class_names = [f"food_{i+1}" for i in range(len(bboxes))]
        confidences = [1.0] * len(bboxes)
        det_time_ms = 0.0
    else:
        t_det = time.perf_counter()
        det_result = detector.detect(np_img)
        det_time_ms = (time.perf_counter() - t_det) * 1000.0
        print(f"  -> Detector Status   : {det_result.status.value}")
        print(f"  -> Detections Found  : {det_result.count}")
        print(f"  -> Detection Time    : {det_time_ms:.1f} ms")

        bboxes = [d.bbox for d in det_result.detections]
        class_names = [d.class_name for d in det_result.detections]
        confidences = [d.confidence for d in det_result.detections]

    if not bboxes:
        print(f"\n[Result] No food detections to segment.")
        print(f"  Status: NO_DETECTIONS")
        print("=" * 76)
        return

    # 4. Execute SAM 2 Segmentation
    print(f"\n[Step 2/4] Running SAM 2 Instance Segmentation on {len(bboxes)} box prompt(s)...")
    t_seg = time.perf_counter()
    seg_result = segmenter.predict_masks(
        image=np_img,
        bboxes=bboxes,
        class_names=class_names,
        confidences=confidences,
        crop_padding=args.crop_padding,
        generate_crops=True,
    )
    seg_time_ms = (time.perf_counter() - t_seg) * 1000.0

    print(f"  -> SAM 2 Status      : {seg_result.status_code}")
    print(f"  -> Segmentation Time : {seg_time_ms:.1f} ms")
    print(f"  -> Successful Masks  : {sum(1 for m in seg_result.masks if m.area_pixels > 0)} / {len(seg_result.masks)}")

    # 5. Export Artifacts
    print(f"\n[Step 3/4] Exporting masks, overlays, RGBA crops, and JSON metadata...")
    out_dir = PROJECT_ROOT / args.output_dir
    exported = save_segmentation_artifacts(
        image=np_img,
        result=seg_result,
        source_name=img_path.stem,
        output_dir=out_dir,
    )

    # 6. Generate 5-Panel Visual Comparison
    print(f"\n[Step 4/4] Generating 5-panel visual comparison...")
    panel_path = out_dir / "predictions" / f"{img_path.stem}_segmentation_panel.jpg"
    create_segmentation_comparison_panel(
        image=np_img,
        result=seg_result,
        bboxes=bboxes,
        class_names=class_names,
        confidences=confidences,
        save_path=panel_path,
    )
    print(f"  -> Saved panel comparison to: {panel_path.relative_to(PROJECT_ROOT) if panel_path.is_relative_to(PROJECT_ROOT) else panel_path}")

    # 7. Print Concise Summary
    print("\n" + "=" * 76)
    print(f"Image: {img_path.name} ({w}x{h})")
    print(f"Detections: {len(seg_result.masks)}")
    print("-" * 76)

    for idx, f_mask in enumerate(seg_result.masks):
        conf_val = confidences[idx] if idx < len(confidences) else 0.0
        crop_rel = exported["rgba"][idx] if idx < len(exported["rgba"]) else "N/A"
        try:
            crop_rel_disp = Path(crop_rel).relative_to(PROJECT_ROOT)
        except Exception:
            crop_rel_disp = crop_rel

        print(f"{idx + 1}. {f_mask.class_name}")
        print(f"   YOLO confidence: {conf_val:.4f}")
        print(f"   SAM quality    : {f_mask.confidence:.4f}")
        print(f"   mask area      : {f_mask.area_pixels} px ({f_mask.relative_area * 100:.1f}%)")
        print(f"   crop (RGBA)    : {crop_rel_disp}")
        print(f"   status         : {f_mask.status}")
        print()

    print("=" * 76)
    print(f"[SUCCESS] SAM 2 FOOD INSTANCE SEGMENTATION COMPLETED!")
    print(f"Master metadata: {Path(exported['metadata_json']).relative_to(PROJECT_ROOT) if exported['metadata_json'] else 'N/A'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
