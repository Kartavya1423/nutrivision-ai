"""
NutriVision AI — Phase 5 End-to-End Execution & Validation Script
Runs: Image -> YOLO11 -> SAM2 -> Depth Anything V2 -> Volume -> Density -> Weight.
Saves structured JSON results and 7-panel visualizations.

Usage:
    python scripts/phase5/run_phase5_pipeline.py --image data/processed/yolo_detection/images/val/foodseg103_00000015.jpg
    python scripts/phase5/run_phase5_pipeline.py --validate-samples
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.detection.yolo_detector import YOLO11FoodDetector
from src.pipeline.phase5_pipeline import Phase5Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NutriVision AI Phase 5 Food Weight Estimation Pipeline."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input food image.",
    )
    parser.add_argument(
        "--validate-samples",
        action="store_true",
        help="Run comprehensive validation on standard benchmark food images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="YOLO detection confidence threshold (default: 0.001).",
    )
    parser.add_argument(
        "--calibration-cm-per-pixel",
        type=float,
        default=None,
        help="Optional metric calibration scale in cm per pixel (e.g. 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase5",
        help="Directory to save Phase 5 artifacts and visualizations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()

    cand_p2 = [
        PROJECT_ROOT / "models" / "detection" / "yolo11" / "phase2_final_run" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "detection" / "yolo11" / "best.pt",
    ]
    det_ckpt = next((p for p in cand_p2 if p.is_file()), None)
    detector = YOLO11FoodDetector(
        model_path=det_ckpt,
        conf_threshold=args.conf,
        settings=settings,
    )

    pipeline = Phase5Pipeline(
        detector=detector,
        settings=settings,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = None
    if args.calibration_cm_per_pixel is not None:
        calibration = {"cm_per_pixel": args.calibration_cm_per_pixel, "reference_object": "calibration_target"}

    if args.validate_samples:
        print("\n========================================================")
        print(" NutriVision AI — Phase 5 Real Food Image Validation Suite")
        print("========================================================")

        # Diverse food scenarios: single food, two foods, multi-food, dessert, negative sample
        sample_candidates = [
            PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "val" / "foodseg103_00000018.jpg",
            PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "val" / "foodseg103_00000008.jpg",
            PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "val" / "foodseg103_00000057.jpg",
            PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "val" / "foodseg103_00000015.jpg",
            PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "val" / "foodseg103_00000032.jpg",
            PROJECT_ROOT / "data" / "raw" / "FoodSeg103" / "Images" / "img_dir" / "test" / "00000048.jpg",
        ]

        valid_samples = [p for p in sample_candidates if p.is_file()]
        if not valid_samples:
            valid_samples = list(PROJECT_ROOT.glob("data/**/*.jpg"))[:5]

        print(f"Selected {len(valid_samples)} benchmark food images covering single, multi-item, and edge cases.")

        validation_records = []
        for idx, sample_path in enumerate(valid_samples):
            print(f"\n[{idx + 1}/{len(valid_samples)}] Validating: {sample_path.name}")

            # 1. Monocular Relative Depth Scenario (Uncalibrated)
            t_start = time.perf_counter()
            res_uncalib = pipeline.run(
                image=sample_path,
                calibration=None,
                save_visualizations=True,
                output_dir=str(out_dir),
            )
            time_uncalib = (time.perf_counter() - t_start) * 1000.0

            # 2. Calibrated Metric Depth Scenario (with standard reference plate 25cm -> 0.05 cm/px)
            t_start = time.perf_counter()
            res_calib = pipeline.run(
                image=sample_path,
                calibration={"cm_per_pixel": 0.05, "reference_object": "standard_dinner_plate_25cm"},
                save_visualizations=True,
                output_dir=str(out_dir),
            )
            time_calib = (time.perf_counter() - t_start) * 1000.0

            detected_classes = [it.class_name for it in res_calib.items]
            print(f"  Detected Foods:    {len(detected_classes)} -> {detected_classes}")
            print(f"  Uncalibrated State:{res_uncalib.calibration_status}")
            print(f"  Calibrated Volume: {res_calib.total_volume_cm3} cm3")
            print(f"  Estimated Weight:  {res_calib.total_estimated_weight_g} g")
            print(f"  Processing Time:   {res_calib.inference_time_ms:.1f} ms")

            record = {
                "image": str(sample_path.resolve()),
                "filename": sample_path.name,
                "detected_foods": detected_classes,
                "total_items": len(detected_classes),
                "segmentation_status": "READY" if res_calib.total_detected_foods > 0 else "NO_DETECTIONS",
                "depth_status": "READY",
                "uncalibrated_evaluation": {
                    "status": res_uncalib.status,
                    "calibration_status": res_uncalib.calibration_status,
                    "volume_cm3": res_uncalib.total_volume_cm3,
                    "estimated_weight_g": res_uncalib.total_estimated_weight_g,
                    "items": [it.to_dict() for it in res_uncalib.items],
                },
                "calibrated_evaluation": {
                    "status": res_calib.status,
                    "calibration_status": res_calib.calibration_status,
                    "total_volume_cm3": res_calib.total_volume_cm3,
                    "total_estimated_weight_g": res_calib.total_estimated_weight_g,
                    "items": [it.to_dict() for it in res_calib.items],
                },
                "warnings": res_calib.warnings,
                "processing_time_ms": round(res_calib.inference_time_ms, 2),
                "visualizations": res_calib.visualization_files,
            }
            validation_records.append(record)

        # Write outputs/phase5/validation_results.json per Section 5.16
        val_json_path = out_dir / "validation_results.json"
        with open(val_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "validation_suite": "NutriVision AI Phase 5 Food Weight Estimation",
                    "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "depth_model": "Depth Anything V2 (Small/vits)",
                    "depth_checkpoint": "models/depth/depth_anything_v2_vits.pth",
                    "detection_model": "YOLO11 (198 food categories)",
                    "segmentation_model": "SAM2 (sam2.1_t.pt)",
                    "ground_truth_weight_status": "GROUND_TRUTH_WEIGHT_DATA_NOT_AVAILABLE",
                    "note": (
                        "data/raw/weight_dataset contains no ground truth measurement records. "
                        "System reports physical weight as calibrated estimation without claiming unverified accuracy."
                    ),
                    "total_images_evaluated": len(validation_records),
                    "successful_evaluations": sum(1 for r in validation_records if r["calibrated_evaluation"]["status"] == "COMPLETE"),
                    "evaluations": validation_records,
                },
                f,
                indent=2,
            )

        print(f"\n========================================================")
        print(f"Validation successfully completed.")
        print(f"Results written to: {val_json_path}")
        print(f"Visualizations saved to: {out_dir / 'visualizations'}")
        print(f"========================================================\n")
        return

    # Single image mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"Error: Image file not found: {img_path}")
            sys.exit(1)

        print(f"\nRunning Phase 5 pipeline on: {img_path}")
        result = pipeline.run(
            image=img_path,
            calibration=calibration,
            save_visualizations=True,
            output_dir=str(out_dir),
        )

        res_dict = result.to_dict()
        print(json.dumps(res_dict, indent=2))

        res_file = out_dir / f"{img_path.stem}_phase5_result.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
        print(f"\nResult saved to: {res_file}")
    else:
        print("Please provide --image <path> or use --validate-samples.")


if __name__ == "__main__":
    main()
