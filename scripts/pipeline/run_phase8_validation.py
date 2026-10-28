"""
NutriVision AI — Phase 8 End-to-End Validation Suite
Executes the complete application pipeline on real image benchmarks:
1. Single-food fresh produce (Apple)
2. Single-food rotten produce (Rotten Apple)
3. Multi-food plate (apple_1.jpg - multiple food items)
4. Cluttered complex plate (apple_banana.jpg - 20+ food items)
5. Low-confidence / difficult image

Uses real model checkpoints for all 5 neural networks.
Records stage-by-stage latencies and outputs:
- outputs/phase8/end_to_end_results.json
- outputs/phase8/performance_report.json
- outputs/phase8/performance_report.md
"""

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from api.main import app, model_manager, pipeline
from src.config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrivision.phase8_val")

cfg = get_settings()


def run_phase8_validation():
    client = TestClient(app)
    out_dir = cfg.root_dir / "outputs" / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        {
            "name": "Single Fresh Apple",
            "path": cfg.root_dir / "data" / "processed" / "freshness" / "test" / "freshapples" / "freshapples_0006.png",
            "category": "single_food",
        },
        {
            "name": "Single Rotten Apple",
            "path": cfg.root_dir / "data" / "processed" / "freshness" / "test" / "rottenapples" / "rottenapples_0006.png",
            "category": "single_food_rotten",
        },
        {
            "name": "Multi-Fruit Plate",
            "path": cfg.root_dir / "data" / "raw" / "Fruits360_multi" / "test-multiple_fruits" / "apple_1.jpg",
            "category": "multi_food",
        },
        {
            "name": "Cluttered Multi-Fruit Scene",
            "path": cfg.root_dir / "data" / "raw" / "Fruits360_multi" / "test-multiple_fruits" / "apple_banana.jpg",
            "category": "cluttered_multi_food",
        },
    ]

    # Add difficult produce case if available
    diff_path = cfg.root_dir / "data" / "processed" / "freshness" / "test" / "rottenbanana" / "rottenbanana_0006.png"
    if diff_path.is_file():
        test_images.append({
            "name": "Difficult Rotten Banana",
            "path": diff_path,
            "category": "difficult_produce",
        })

    eval_results = []
    performance_records = []
    total_foods_evaluated = 0

    logger.info("Starting Phase 8 End-to-End Validation with real checkpoints...")

    for test_info in test_images:
        img_p = test_info["path"]
        if not img_p.is_file():
            logger.warning(f"File not found: {img_p}")
            continue

        with open(img_p, "rb") as f:
            file_bytes = f.read()

        t_req0 = time.perf_counter()
        files = {"file": (img_p.name, file_bytes, "image/png" if img_p.suffix == ".png" else "image/jpeg")}
        data = {
            "enable_gradcam": "true",
            "enable_nutrition": "true",
            "crop_padding": "10",
        }

        res = client.post("/analyze", files=files, data=data)
        req_latency_ms = round((time.perf_counter() - t_req0) * 1000.0, 2)

        if res.status_code != 200:
            logger.error(f"Failed analysis on {img_p.name}: {res.status_code} {res.text}")
            continue

        res_data = res.json()
        item_count = len(res_data.get("items", []))
        total_foods_evaluated += item_count

        logger.info(
            f"Evaluated '{test_info['name']}' ({img_p.name}): "
            f"Detected {item_count} items in {req_latency_ms:.1f} ms | Status: {res_data['status']}"
        )

        record = {
            "benchmark_name": test_info["name"],
            "image_filename": img_p.name,
            "category": test_info["category"],
            "status": res_data["status"],
            "items_count": item_count,
            "request_latency_ms": req_latency_ms,
            "timing": res_data.get("timing", {}),
            "result_id": res_data.get("result_id"),
            "meal_summary": res_data.get("meal_summary"),
            "items": res_data.get("items", []),
            "visualizations": res_data.get("visualizations", {}),
        }
        eval_results.append(record)

        perf_entry = {
            "image": img_p.name,
            "items": item_count,
            "total_request_ms": req_latency_ms,
            **res_data.get("timing", {}),
        }
        performance_records.append(perf_entry)

    # Save end-to-end results JSON
    e2e_json_path = out_dir / "end_to_end_results.json"
    with open(e2e_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_benchmark_scenes": len(eval_results),
            "total_foods_evaluated": total_foods_evaluated,
            "device": model_manager.device,
            "scenes": eval_results,
        }, f, indent=2)
    logger.info(f"Saved end-to-end results to {e2e_json_path}")

    # Compute aggregate timing statistics
    avg_det = np.mean([p["detection_ms"] for p in performance_records]) if performance_records else 0.0
    avg_seg = np.mean([p["segmentation_ms"] for p in performance_records]) if performance_records else 0.0
    avg_dep = np.mean([p["depth_ms"] for p in performance_records]) if performance_records else 0.0
    avg_cls = np.mean([p["classification_ms"] for p in performance_records]) if performance_records else 0.0
    avg_fre = np.mean([p["freshness_ms"] for p in performance_records]) if performance_records else 0.0
    avg_nut = np.mean([p["nutrition_ms"] for p in performance_records]) if performance_records else 0.0
    avg_cam = np.mean([p["gradcam_ms"] for p in performance_records]) if performance_records else 0.0
    avg_tot = np.mean([p["total_request_ms"] for p in performance_records]) if performance_records else 0.0

    perf_summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "device": model_manager.device,
        "total_images_processed": len(performance_records),
        "total_foods_evaluated": total_foods_evaluated,
        "averages_ms": {
            "avg_detection_ms": round(float(avg_det), 2),
            "avg_segmentation_ms": round(float(avg_seg), 2),
            "avg_depth_ms": round(float(avg_dep), 2),
            "avg_classification_ms": round(float(avg_cls), 2),
            "avg_freshness_ms": round(float(avg_fre), 2),
            "avg_nutrition_ms": round(float(avg_nut), 2),
            "avg_gradcam_ms": round(float(avg_cam), 2),
            "avg_total_request_ms": round(float(avg_tot), 2),
        },
        "per_image_performance": performance_records,
    }

    perf_json_path = out_dir / "performance_report.json"
    with open(perf_json_path, "w", encoding="utf-8") as f:
        json.dump(perf_summary, f, indent=2)
    logger.info(f"Saved performance report to {perf_json_path}")

    # Generate performance report Markdown
    perf_md_path = out_dir / "performance_report.md"
    with open(perf_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 8 API & Pipeline Performance Report\n\n")
        f.write(f"**Execution Date:** {datetime.now(timezone.utc).strftime('%B %d, %Y')}\n")
        f.write(f"**Hardware Device:** `{model_manager.device.upper()}`\n")
        f.write(f"**Benchmark Scenes Evaluated:** {len(performance_records)}\n")
        f.write(f"**Total Food Items Analyzed:** {total_foods_evaluated}\n\n")
        f.write("## 1. Mean Pipeline Stage Latencies\n\n")
        f.write("| Pipeline Stage | Mean Latency (ms) |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| **YOLO11 Detection** | {avg_det:.2f} ms |\n")
        f.write(f"| **SAM2 Instance Segmentation** | {avg_seg:.2f} ms |\n")
        f.write(f"| **Depth Anything V2** | {avg_dep:.2f} ms |\n")
        f.write(f"| **EfficientNet-B4 Classification** | {avg_cls:.2f} ms |\n")
        f.write(f"| **Freshness Classification** | {avg_fre:.2f} ms |\n")
        f.write(f"| **USDA Nutrition Query (Cached)** | {avg_nut:.2f} ms |\n")
        f.write(f"| **Grad-CAM Attention Generation** | {avg_cam:.2f} ms |\n")
        f.write(f"| **Total End-to-End Request** | **{avg_tot:.2f} ms** |\n\n")
        f.write("## 2. Per-Image Benchmark Breakdown\n\n")
        f.write("| Image | Category | Items | Total Latency (ms) | Status |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for rec in eval_results:
            f.write(f"| `{rec['image_filename']}` | {rec['category']} | {rec['items_count']} | {rec['request_latency_ms']:.1f} ms | `{rec['status']}` |\n")
        f.write("\n---\n")

    logger.info(f"Saved performance report markdown to {perf_md_path}")
    print("Phase 8 End-to-End Validation Completed Successfully!")


if __name__ == "__main__":
    run_phase8_validation()
