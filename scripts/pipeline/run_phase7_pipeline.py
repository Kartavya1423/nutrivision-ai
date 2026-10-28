"""
NutriVision AI — Phase 7 End-to-End Pipeline Validation Runner
Executes complete 10-stage pipeline:
YOLO11 -> SAM2 -> Food Crop -> EfficientNet Cls -> Freshness -> Depth/Weight -> USDA Nutrition -> Grad-CAM.

Outputs:
- outputs/phase7/validation_results.json
- outputs/phase7/performance.json
- outputs/phase7/visualizations/
- outputs/phase7/gradcam/
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.pipeline.phase7_pipeline import Phase7Pipeline


BENCHMARK_IMAGES = [
    # Single fruit: fresh apple
    "data/processed/freshness/test/freshapples/freshapples_0006.png",
    # Single fruit: rotten apple
    "data/processed/freshness/test/rottenapples/rottenapples_0006.png",
    # Single fruit: fresh banana
    "data/processed/freshness/test/freshbanana/freshbanana_0006.png",
    # Single fruit: rotten banana
    "data/processed/freshness/test/rottenbanana/rottenbanana_0006.png",
    # Single fruit: fresh orange
    "data/processed/freshness/test/freshoranges/freshoranges_0006.png",
    # Single fruit: rotten orange
    "data/processed/freshness/test/rottenoranges/rottenoranges_0006.png",
    # Multi-food plate scenes from Fruits-360 multi / FoodSeg103
    "data/raw/Fruits360_multi/test-multiple_fruits/apple_1.jpg",
    "data/raw/Fruits360_multi/test-multiple_fruits/apple_banana.jpg",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 7 full end-to-end vision and nutrition pipeline.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image.")
    parser.add_argument("--validate-samples", action="store_true", help="Run full benchmark suite.")
    parser.add_argument("--calibration-cm-per-pixel", type=float, default=0.05, help="Scale calibration.")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation.")
    parser.add_argument("--output-dir", type=str, default="outputs/phase7", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = Phase7Pipeline(calibration_cm_per_pixel=args.calibration_cm_per_pixel)

    images_to_process = []
    if args.image:
        images_to_process.append(Path(args.image))
    elif args.validate_samples:
        for p_str in BENCHMARK_IMAGES:
            p = root_dir / p_str
            if p.is_file():
                images_to_process.append(p)
            else:
                print(f"Warning: Benchmark image not found: {p}")
    else:
        # Default test image
        def_img = root_dir / BENCHMARK_IMAGES[0]
        if def_img.is_file():
            images_to_process.append(def_img)
        else:
            print("Error: Specify --image or --validate-samples.")
            return 1

    print(f"\n================================================================================")
    print(f" NutriVision AI — Phase 7 Full End-to-End Pipeline Validation Suite")
    print(f"================================================================================")
    print(f" Images to process:        {len(images_to_process)}")
    print(f" Grad-CAM explainability:  {'Disabled' if args.no_gradcam else 'Enabled'}")
    print(f" Calibration (cm/pixel):   {args.calibration_cm_per_pixel}")
    print(f" Output directory:         {pipeline.output_dir}")
    print(f"================================================================================\n")

    all_results = []
    performance_records = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_images_processed": len(images_to_process),
        "total_foods_evaluated": 0,
        "latencies_ms": {
            "detection": [],
            "segmentation": [],
            "depth": [],
            "volume_and_weight": [],
            "classification": [],
            "freshness": [],
            "usda_nutrition": [],
            "gradcam": [],
            "total_pipeline": [],
        },
        "averages_ms": {},
    }

    t_suite_start = time.perf_counter()

    for idx, img_p in enumerate(images_to_process, 1):
        print(f"[{idx}/{len(images_to_process)}] Processing: {img_p.name} ...")
        t0 = time.perf_counter()

        res = pipeline.process_image(
            image_path=img_p,
            generate_gradcam=not args.no_gradcam,
            save_visualizations=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        res_dict = res.to_dict()
        all_results.append(res_dict)

        performance_records["total_foods_evaluated"] += res.total_foods_detected
        performance_records["latencies_ms"]["total_pipeline"].append(res.processing_time_ms)
        if "detection_ms" in res.stage_latencies_ms:
            performance_records["latencies_ms"]["detection"].append(res.stage_latencies_ms["detection_ms"])
        if "segmentation_ms" in res.stage_latencies_ms:
            performance_records["latencies_ms"]["segmentation"].append(res.stage_latencies_ms["segmentation_ms"])
        if "depth_ms" in res.stage_latencies_ms:
            performance_records["latencies_ms"]["depth"].append(res.stage_latencies_ms["depth_ms"])

        # Collect item level latencies
        for k, v in res.stage_latencies_ms.items():
            if "classification_ms" in k:
                performance_records["latencies_ms"]["classification"].append(v)
            elif "freshness_ms" in k:
                performance_records["latencies_ms"]["freshness"].append(v)
            elif "nutrition_ms" in k:
                performance_records["latencies_ms"]["usda_nutrition"].append(v)
            elif "gradcam_ms" in k:
                performance_records["latencies_ms"]["gradcam"].append(v)

        print(f"    • Status: {res.overall_status} | Foods Detected: {res.total_foods_detected} | Time: {elapsed:.1f}ms")
        for it in res.items:
            cal = it.nutrition.get("calories_kcal")
            cal_str = f"{cal:.1f} kcal" if cal is not None else "N/A"
            wt_str = f"{it.estimated_weight_g:.1f}g" if it.estimated_weight_g is not None else "N/A"
            print(f"      - {it.food.upper()} | Freshness: {it.freshness} ({it.freshness_confidence*100:.1f}%) | Wt: {wt_str} | Energy: {cal_str}")

        if res.visualization_path:
            print(f"    • Visualization: {res.visualization_path}")

    # Compute averages
    for k, v in performance_records["latencies_ms"].items():
        if v:
            performance_records["averages_ms"][f"avg_{k}_ms"] = round(sum(v) / len(v), 2)
            performance_records["averages_ms"][f"p95_{k}_ms"] = round(sorted(v)[int(len(v) * 0.95)], 2)
            performance_records["averages_ms"][f"min_{k}_ms"] = round(min(v), 2)
            performance_records["averages_ms"][f"max_{k}_ms"] = round(max(v), 2)

    total_suite_s = time.perf_counter() - t_suite_start
    performance_records["total_suite_runtime_seconds"] = round(total_suite_s, 2)

    # Save validation results JSON
    val_out = pipeline.output_dir / "validation_results.json"
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save performance JSON
    perf_out = pipeline.output_dir / "performance.json"
    with open(perf_out, "w", encoding="utf-8") as f:
        json.dump(performance_records, f, indent=2)

    print(f"\n================================================================================")
    print(f" Phase 7 End-to-End Validation Completed!")
    print(f"  • Total Images:               {len(images_to_process)}")
    print(f"  • Total Food Items Tested:    {performance_records['total_foods_evaluated']}")
    print(f"  • Average Pipeline Latency:   {performance_records['averages_ms'].get('avg_total_pipeline_ms', 0.0)} ms")
    print(f"  • Average USDA Lookup:        {performance_records['averages_ms'].get('avg_usda_nutrition_ms', 0.0)} ms")
    print(f"  • Average Grad-CAM Time:      {performance_records['averages_ms'].get('avg_gradcam_ms', 0.0)} ms")
    print(f"  • Validation Results JSON:    {val_out}")
    print(f"  • Performance JSON:           {perf_out}")
    print(f"================================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
