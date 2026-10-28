#!/usr/bin/env python3
"""
NutriVision AI — Freshness Pipeline Runner CLI (Phase 6)
Executes the full Phase 6 pipeline:
    Image -> YOLO11 -> SAM2 -> Food-only Crop -> Freshness Assessment

Usage:
    python scripts/freshness/run_freshness_pipeline.py --image <IMAGE_PATH>
    python scripts/freshness/run_freshness_pipeline.py --benchmark
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.freshness_pipeline import FreshnessPipeline


def main():
    parser = argparse.ArgumentParser(description="Run complete Phase 6 freshness detection pipeline.")
    parser.add_argument("--image", type=str, default=None, help="Path to input food image.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on validation suite.")
    parser.add_argument("--conf_thresh", type=float, default=0.001, help="YOLO detection confidence threshold.")
    parser.add_argument("--fresh_thresh", type=float, default=0.60, help="Freshness confidence threshold.")
    parser.add_argument("--output_dir", type=str, default="outputs/freshness", help="Output directory.")
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("      NutriVision AI — Phase 6 Freshness Pipeline Orchestrator      ")
    print("=" * 72)

    pipeline = FreshnessPipeline(
        conf_threshold=args.conf_thresh,
        freshness_threshold=args.fresh_thresh,
        output_dir=out_root,
    )

    if args.image:
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        if not img_path.is_file():
            print(f"[Error] Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Processing image: {img_path.name}")
        res = pipeline.process_image(img_path)

        print(f"\n[Summary]")
        print(f"  Source Image     : {res.source_image}")
        print(f"  Foods Detected   : {res.total_foods_detected}")
        print(f"  Overall Status   : {res.overall_status}")
        print(f"  Processing Time  : {res.processing_time_ms:.1f} ms")

        for it in res.items:
            print(f"\n  Item #{it.item_index + 1}: {it.food_class.upper()}")
            print(f"    - Detection Conf : {it.detection_confidence*100:.1f}%")
            print(f"    - Freshness      : {it.freshness_state.upper()} ({it.freshness_confidence*100:.1f}% conf)")
            print(f"    - Probabilities  : P(fresh)={it.probabilities.get('fresh', 0)*100:.1f}% | P(rotten)={it.probabilities.get('rotten', 0)*100:.1f}%")
            print(f"    - Notes          : {it.notes}")

        if res.visualization_path:
            print(f"\n  -> Visualization saved: {res.visualization_path}")

        # Save result JSON
        out_json = out_root / f"{img_path.stem}_pipeline_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        print(f"  -> Structured JSON saved: {out_json.relative_to(PROJECT_ROOT)}")

    elif args.benchmark:
        print("[Benchmark] Running on test validation benchmark suite...")
        # Gather sample images from Fruits360_multi and test sets
        sample_candidates = [
            PROJECT_ROOT / "data" / "processed" / "freshness" / "test" / "freshapples" / "freshapples_0006.png",
            PROJECT_ROOT / "data" / "processed" / "freshness" / "test" / "rottenapples" / "rottenapples_0006.png",
            PROJECT_ROOT / "data" / "processed" / "freshness" / "test" / "freshbanana" / "freshbanana_0006.png",
            PROJECT_ROOT / "data" / "processed" / "freshness" / "test" / "rottenbanana" / "rottenbanana_0006.png",
            PROJECT_ROOT / "data" / "processed" / "freshness" / "test" / "freshoranges" / "freshoranges_0006.png",
            PROJECT_ROOT / "data" / "processed" / "freshness" / "test" / "rottenoranges" / "rottenoranges_0006.png",
            PROJECT_ROOT / "data" / "raw" / "Fruits360_multi" / "test-multiple_fruits" / "apple_1.jpg",
            PROJECT_ROOT / "data" / "raw" / "Fruits360_multi" / "test-multiple_fruits" / "banana_1.jpg",
        ]

        benchmark_results = []
        for sample_p in sample_candidates:
            if not sample_p.is_file():
                continue
            print(f"  -> Testing: {sample_p.name}...")
            res = pipeline.process_image(sample_p)
            benchmark_results.append(res.to_dict())

        out_bench = out_root / "validation_results.json"
        with open(out_bench, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\n[Completed] Benchmark results saved to: {out_bench.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
