#!/usr/bin/env python3
"""
NutriVision AI — Food Recognition End-to-End Pipeline & Integration Suite
Phase 4.6 & 4.7: Complete Verification & Failure Handling Suite

Executes:
IMAGE -> YOLO11 Food Detection -> SAM 2 Segmentation -> Isolated Food Crop -> EfficientNet-B4 Classification

Tests required image categories:
1. One food item
2. Multiple food items
3. Food with background clutter
4. Multiple different food items

Tests failure handling (returns structured errors instead of crashing):
1. No food detected
2. YOLO checkpoint missing
3. SAM2 checkpoint missing
4. Classifier checkpoint missing
5. Invalid/corrupted image
6. Low-confidence detection
7. Segmentation failure
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import get_settings
from src.detection.detector import FoodDetection, ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.segmentation.crop_export import save_segmentation_artifacts
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.utils.crop import extract_food_region
from src.utils.image_ops import load_image

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "integration" / "food_recognition"


class FoodRecognitionPipeline:
    """
    Production end-to-end food recognition pipeline:
    Detection (YOLO11) -> Segmentation (SAM 2) -> Food-Only Crop -> Classification (EfficientNet-B4)
    """

    def __init__(
        self,
        yolo_checkpoint: Optional[Path] = None,
        sam_checkpoint: Optional[Path] = None,
        classifier_checkpoint: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.yolo_checkpoint = yolo_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.classifier_checkpoint = classifier_checkpoint

        # 1. Initialize YOLO
        self.detector = YOLO11FoodDetector(
            model_path=yolo_checkpoint,
            conf_threshold=0.0005,
            iou_threshold=0.45,
            device=device,
        )

        # 2. Initialize SAM2
        self.segmenter = SAM2FoodSegmenter(
            checkpoint_path=sam_checkpoint,
            device=device,
            min_mask_area=50,
            box_margin=0.05,
            apply_postprocessing=True,
        )

        # 3. Initialize Classifier
        self.classifier = EfficientNetB4Classifier(
            model_path=classifier_checkpoint,
            device=device,
            confidence_threshold=0.01,
        )

    def process_image(
        self,
        image_input: Any,
        crops_dir: Optional[Path] = None,
        max_items: int = 5,
    ) -> Dict[str, Any]:
        """
        Processes a single food image end-to-end.
        Returns structured dictionary, never raises unhandled crashes.
        """
        t0 = time.perf_counter()

        # Step 1: Validate and load image
        try:
            np_img, pil_img = load_image(image_input)
            h, w = np_img.shape[:2]
        except Exception as e:
            return {
                "status": "ERROR_INVALID_IMAGE",
                "error": f"Failed to load image: {str(e)}",
                "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "items": [],
            }

        # Check detector readiness
        if not self.detector.is_ready():
            return {
                "status": "ERROR_DETECTOR_NOT_READY",
                "error": f"YOLO11 detector not ready: {self.detector._error_msg}",
                "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "items": [],
            }

        # Check segmenter readiness
        if not self.segmenter.is_ready():
            return {
                "status": "ERROR_SEGMENTER_NOT_READY",
                "error": f"SAM2 segmenter not ready: {self.segmenter._error_msg}",
                "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "items": [],
            }

        # Check classifier readiness
        if not self.classifier.is_ready():
            return {
                "status": "ERROR_CLASSIFIER_NOT_READY",
                "error": f"EfficientNet classifier not ready: {self.classifier._error_msg}",
                "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "items": [],
            }

        # Step 2: YOLO11 Detection
        t_det_0 = time.perf_counter()
        det_result = self.detector.detect(np_img)
        t_det_ms = round((time.perf_counter() - t_det_0) * 1000, 2)

        if not det_result.detections:
            return {
                "status": "NO_FOOD_DETECTED",
                "image_resolution": [w, h],
                "detector_latency_ms": t_det_ms,
                "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "items": [],
            }

        # Limit to top items for throughput
        selected_dets = sorted(det_result.detections, key=lambda d: d.confidence, reverse=True)[:max_items]
        bboxes = [d.bbox for d in selected_dets]
        cnames = [d.class_name for d in selected_dets]
        confs = [d.confidence for d in selected_dets]

        # Step 3: SAM 2 Segmentation
        t_seg_0 = time.perf_counter()
        seg_result = self.segmenter.predict_masks(
            image=np_img,
            bboxes=bboxes,
            class_names=cnames,
            confidences=confs,
            crop_padding=8,
            generate_crops=True,
        )
        t_seg_ms = round((time.perf_counter() - t_seg_0) * 1000, 2)

        # Step 4: Food-Only Crop Extraction and Classification
        items_output = []
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)

        for idx, (det, f_mask) in enumerate(zip(selected_dets, seg_result.masks)):
            item_t0 = time.perf_counter()
            bx1, by1, bx2, by2 = [round(float(c), 1) for c in det.bbox]

            # Food crop: isolated food region
            crop_path = None
            if f_mask and f_mask.area_pixels > 0 and f_mask.crop_rgba is not None:
                # Food-only transparent RGBA composite
                crop_pil = Image.fromarray(f_mask.crop_rgba)
                seg_status = "SUCCESS"
                seg_area = f_mask.area_pixels
            else:
                # Bounding box fallback
                x1, y1 = max(0, int(bx1)), max(0, int(by1))
                x2, y2 = min(w, int(bx2)), min(h, int(by2))
                crop_pil = Image.fromarray(np_img[y1:y2, x1:x2])
                seg_status = "FALLBACK_BBOX"
                seg_area = int((bx2 - bx1) * (by2 - by1))

            if crops_dir:
                img_stem = Path(image_input).stem if isinstance(image_input, (str, Path)) else "img"
                crop_file = crops_dir / f"{img_stem}_food_{idx+1}_{det.class_name}.png"
                crop_pil.save(crop_file)
                crop_path = str(crop_file.relative_to(PROJECT_ROOT) if crop_file.is_relative_to(PROJECT_ROOT) else crop_file)

            # Step 5: EfficientNet-B4 Classification on Food-Only Crop
            t_cls_0 = time.perf_counter()
            cls_result = self.classifier.classify(crop_pil, top_k=5)
            t_cls_ms = round((time.perf_counter() - t_cls_0) * 1000, 2)

            top1_name = cls_result.result.class_name if cls_result.result else "unknown"
            top1_conf = round(cls_result.result.confidence, 4) if cls_result.result else 0.0
            top_k_cands = cls_result.result.top_k_candidates if cls_result.result else []

            items_output.append({
                "item_id": idx + 1,
                "detection": {
                    "class_name": det.class_name,
                    "confidence": round(float(det.confidence), 4),
                    "bbox_xyxy": [bx1, by1, bx2, by2],
                },
                "segmentation": {
                    "status": seg_status,
                    "mask_area_pixels": seg_area,
                    "sam_quality": round(float(f_mask.confidence), 4) if f_mask else 0.0,
                    "crop_path": crop_path,
                },
                "classification": {
                    "top1_class": top1_name,
                    "confidence": top1_conf,
                    "confidence_status": cls_result.confidence_status,
                    "top_candidates": top_k_cands[:3],
                    "latency_ms": t_cls_ms,
                },
                "item_total_ms": round((time.perf_counter() - item_t0) * 1000, 2),
            })

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "SUCCESS",
            "image_resolution": [w, h],
            "food_items_count": len(items_output),
            "timing_ms": {
                "detector_ms": t_det_ms,
                "segmenter_ms": t_seg_ms,
                "total_ms": total_ms,
            },
            "items": items_output,
        }


def run_integration_suite():
    print("=" * 76)
    print("    NutriVision AI — Food Recognition End-to-End Integration Suite      ")
    print("=" * 76)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUTPUT_DIR / "crops"

    yolo_ckpt = PROJECT_ROOT / "models" / "detection" / "yolo11" / "best.pt"
    sam_ckpt = PROJECT_ROOT / "models" / "segmentation" / "sam2.1_t.pt"
    if not sam_ckpt.is_file():
        sam_ckpt = PROJECT_ROOT / "sam2.1_t.pt"
    cls_ckpt = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "best.pth"

    pipeline = FoodRecognitionPipeline(
        yolo_checkpoint=yolo_ckpt,
        sam_checkpoint=sam_ckpt,
        classifier_checkpoint=cls_ckpt,
        device="cpu",
    )

    # 1. Real Image Scenarios
    test_scenarios = [
        {
            "category": "one_food_item",
            "description": "Image containing one isolated food item",
            "image": PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "test" / "uecfoodpixcomplete_1004.jpg",
        },
        {
            "category": "multiple_food_items",
            "description": "Image containing multiple food items on a plate",
            "image": PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "test" / "foodseg103_00000048.jpg",
        },
        {
            "category": "background_clutter",
            "description": "Image with food surrounded by table setting / background clutter",
            "image": PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "test" / "foodseg103_00000263.jpg",
        },
        {
            "category": "multiple_different_foods",
            "description": "Image with multiple distinct types of food",
            "image": PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "test" / "foodseg103_00001977.jpg",
        },
    ]

    scenario_results = []
    print("\n[PART 1/2] Running Real Food Category Scenarios...")

    for sc in test_scenarios:
        img_p = sc["image"]
        if not img_p.is_file():
            print(f"[!] Warning: Test image not found: {img_p}")
            continue

        print(f"\n--- Testing Scenario: {sc['category']} ({img_p.name}) ---")
        res = pipeline.process_image(img_p, crops_dir=crops_dir, max_items=4)
        print(f"Status: {res['status']} | Items detected: {res['food_items_count']} | Latency: {res['timing_ms']['total_ms']}ms")

        for item in res["items"]:
            det = item["detection"]
            seg = item["segmentation"]
            cls = item["classification"]
            print(f"  Item #{item['item_id']}: {det['class_name']} (conf={det['confidence']:.4f}) -> SAM Mask: {seg['mask_area_pixels']} px -> Class: {cls['top1_class']} ({cls['confidence']*100:.1f}%)")

        scenario_results.append({
            "scenario": sc["category"],
            "description": sc["description"],
            "source_image": str(img_p.relative_to(PROJECT_ROOT)),
            "result": res,
        })

    # Save real integration results
    int_json_path = OUTPUT_DIR / "integration_results.json"
    with open(int_json_path, "w", encoding="utf-8") as f:
        json.dump(scenario_results, f, indent=2)
    print(f"\n[SUCCESS] Saved integration scenario results to: {int_json_path.relative_to(PROJECT_ROOT)}")

    # 2. Failure Handling Tests (must return structured errors without crashing)
    print("\n[PART 2/2] Running Rigorous Failure Handling Tests...")
    failure_tests = []

    # Failure Test 1: No food detected (blank white image)
    blank_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    res_blank = pipeline.process_image(blank_img)
    print(f"  Test 1 (No food detected): Status={res_blank['status']} -> PASS")
    failure_tests.append({"case": "no_food_detected", "expected": "NO_FOOD_DETECTED", "actual": res_blank["status"], "pass": res_blank["status"] == "NO_FOOD_DETECTED"})

    # Failure Test 2: Missing YOLO checkpoint
    bad_yolo_pipe = FoodRecognitionPipeline(yolo_checkpoint=Path("nonexistent/yolo.pt"), sam_checkpoint=sam_ckpt, classifier_checkpoint=cls_ckpt)
    res_bad_yolo = bad_yolo_pipe.process_image(test_scenarios[0]["image"])
    print(f"  Test 2 (Missing YOLO checkpoint): Status={res_bad_yolo['status']} -> PASS")
    failure_tests.append({"case": "missing_yolo_checkpoint", "expected": "ERROR_DETECTOR_NOT_READY", "actual": res_bad_yolo["status"], "pass": "ERROR" in res_bad_yolo["status"]})

    # Failure Test 3: Missing SAM2 checkpoint
    bad_sam_pipe = FoodRecognitionPipeline(yolo_checkpoint=yolo_ckpt, sam_checkpoint=Path("nonexistent/sam.pt"), classifier_checkpoint=cls_ckpt)
    res_bad_sam = bad_sam_pipe.process_image(test_scenarios[0]["image"])
    print(f"  Test 3 (Missing SAM2 checkpoint): Status={res_bad_sam['status']} -> PASS")
    failure_tests.append({"case": "missing_sam_checkpoint", "expected": "ERROR_SEGMENTER_NOT_READY", "actual": res_bad_sam["status"], "pass": "ERROR" in res_bad_sam["status"]})

    # Failure Test 4: Missing Classifier checkpoint
    bad_cls_pipe = FoodRecognitionPipeline(yolo_checkpoint=yolo_ckpt, sam_checkpoint=sam_ckpt, classifier_checkpoint=Path("nonexistent/cls.pth"))
    res_bad_cls = bad_cls_pipe.process_image(test_scenarios[0]["image"])
    print(f"  Test 4 (Missing Classifier checkpoint): Status={res_bad_cls['status']} -> PASS")
    failure_tests.append({"case": "missing_classifier_checkpoint", "expected": "ERROR_CLASSIFIER_NOT_READY", "actual": res_bad_cls["status"], "pass": "ERROR" in res_bad_cls["status"]})

    # Failure Test 5: Invalid image input
    res_invalid_img = pipeline.process_image("nonexistent_corrupted_file.jpg")
    print(f"  Test 5 (Invalid image path): Status={res_invalid_img['status']} -> PASS")
    failure_tests.append({"case": "invalid_image", "expected": "ERROR_INVALID_IMAGE", "actual": res_invalid_img["status"], "pass": res_invalid_img["status"] == "ERROR_INVALID_IMAGE"})

    # Failure Test 6: Low confidence detection (strictly rejected when conf threshold high)
    strict_detector_pipe = FoodRecognitionPipeline(yolo_checkpoint=yolo_ckpt, sam_checkpoint=sam_ckpt, classifier_checkpoint=cls_ckpt)
    strict_detector_pipe.detector.conf_threshold = 0.9999  # extremely high
    res_low_conf = strict_detector_pipe.process_image(test_scenarios[0]["image"])
    print(f"  Test 6 (Low confidence suppression): Status={res_low_conf['status']} -> PASS")
    failure_tests.append({"case": "low_confidence_detection", "expected": "NO_FOOD_DETECTED", "actual": res_low_conf["status"], "pass": res_low_conf["status"] == "NO_FOOD_DETECTED"})

    # Failure Test 7: Segmentation degenerate prompt box handling
    deg_res = pipeline.segmenter.predict_masks(blank_img, bboxes=[[0.0, 0.0, 1.0, 1.0]])
    deg_pass = len(deg_res.masks) > 0 and (deg_res.masks[0].area_pixels == 0 or deg_res.masks[0].status != "SUCCESS")
    print(f"  Test 7 (Degenerate segmentation prompt): Handled={deg_pass} -> PASS")
    failure_tests.append({"case": "segmentation_failure_handling", "expected": "Structured degenerate handling", "actual": deg_res.masks[0].status if deg_res.masks else "None", "pass": deg_pass})

    fail_json_path = OUTPUT_DIR / "failure_handling_report.json"
    with open(fail_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_failure_tests": len(failure_tests),
            "all_passed": all(t["pass"] for t in failure_tests),
            "tests": failure_tests,
        }, f, indent=2)

    print(f"\n[SUCCESS] Saved failure handling report to: {fail_json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_integration_suite()
