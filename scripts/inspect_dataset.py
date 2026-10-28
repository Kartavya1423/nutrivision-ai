#!/usr/bin/env python3
"""
NutriVision AI — Dataset Inspection CLI Tool
Module 1: Dataset Inspection and Preparation

Usage:
  python scripts/inspect_dataset.py
  python scripts/inspect_dataset.py --data-dir data/raw
  python scripts/inspect_dataset.py --create-sample
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path so src imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset_inspector import DatasetInspector


def create_sample_benchmark_dataset(target_dir: Path):
    """
    Creates a minimal, synthetic sample dataset structure to test and demonstrate
    all format detection capabilities (YOLO detection, classification, freshness).
    """
    from PIL import Image, ImageDraw

    print(f"\n[Sample Creator] Generating sample benchmark dataset in: {target_dir}...")
    sample_classes = ["fresh_apple", "rotten_apple", "fresh_banana", "rotten_banana", "salad"]
    splits = ["train", "val", "test"]

    for split in splits:
        for cname in sample_classes:
            split_class_dir = target_dir / split / cname
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a synthetic test image
            for idx in range(2):
                img = Image.new("RGB", (320, 240), color=(100 + idx * 30, 150, 80 + idx * 20))
                draw = ImageDraw.Draw(img)
                draw.rectangle([50, 40, 270, 200], outline="red", width=2)
                draw.text((60, 50), f"{cname}_{split}_{idx}", fill="white")
                img_path = split_class_dir / f"{cname}_{split}_{idx}.jpg"
                img.save(img_path)

                # Create corresponding YOLO bounding box annotation
                txt_path = split_class_dir / f"{cname}_{split}_{idx}.txt"
                class_id = sample_classes.index(cname)
                # YOLO format: class_id x_center y_center width height (normalized)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"{class_id} 0.50 0.50 0.68 0.66\n")

    print(f"[Sample Creator] Created 30 images across 5 classes with YOLO bboxes and splits.\n")


def main():
    parser = argparse.ArgumentParser(
        description="NutriVision AI — Dataset Inspection & Format Detector"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Path to raw dataset directory to inspect (default: data/raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "reports"),
        help="Directory to store inspection reports (default: data/reports)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Generate a sample benchmark dataset in data/raw/sample_benchmark to verify scanner"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.create_sample:
        sample_path = data_dir / "sample_benchmark"
        create_sample_benchmark_dataset(sample_path)
        data_dir = sample_path

    print("=" * 70)
    print("        NutriVision AI -- Dataset Inspection & Preparation       ")
    print("=" * 70)
    print(f"Target Directory: {data_dir}")
    print(f"Reports Output  : {output_dir}")
    print("-" * 70)

    inspector = DatasetInspector(raw_dir=str(data_dir))
    results = inspector.run_inspection()

    # Generate output reports
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_report.json"
    txt_path = output_dir / "dataset_report.txt"

    inspector.generate_json_report(str(json_path))
    txt_report = inspector.generate_text_report(str(txt_path))

    # Print human-readable report to console
    print(txt_report)

    # Print exact required summary fields
    print("\n" + "#" * 70)
    print("             EXECUTIVE SUMMARY (NutriVision AI Module 1)          ")
    print("#" * 70)
    print(f"* Total Images                : {results['total_images']}")
    print(f"* Total Classes               : {results['total_classes']}")
    print(f"* Class Names                 : {', '.join(results['class_names']) if results['class_names'] else 'None'}")
    print(f"* Annotation Format           : {', '.join(results['annotation_formats_detected']) if results['annotation_formats_detected'] else 'None'}")
    print(f"* Bounding-Box Availability   : {'AVAILABLE' if results['bounding_box_available'] else 'NOT AVAILABLE'}")
    print(f"* Segmentation-Mask Avail.    : {'AVAILABLE' if results['segmentation_mask_available'] else 'NOT AVAILABLE (Requires SAM 2 zero-shot)'}")
    print(f"* Freshness-Label Avail.      : {'AVAILABLE' if results['freshness_labels_available'] else 'NOT AVAILABLE'}")
    print(f"* Train/Val/Test Splits       : {results['splits_detected'] if results['splits_detected'] else 'None'}")
    print(f"* Missing Data / Gaps         : {'; '.join(results['missing_data_warnings']) if results['missing_data_warnings'] else 'None'}")
    print(f"* Recommended Next Action     : {results['recommended_next_action']}")
    print("#" * 70)
    print(f"\n[Artifacts Saved]:")
    print(f"  JSON Report : {json_path}")
    print(f"  Text Report : {txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
