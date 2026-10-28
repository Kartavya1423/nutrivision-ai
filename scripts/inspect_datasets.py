#!/usr/bin/env python3
"""
NutriVision AI — Dataset Status & Inspection Command
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Usage:
  python scripts/inspect_datasets.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_scanner import DatasetScanner


def main():
    scanner = DatasetScanner()
    scans = scanner.scan_all()

    print("\n# NutriVision AI Dataset Inspection\n")

    # Display ordered target datasets
    targets = [
        "FoodSeg103",
        "UECFoodPixComplete",
        "Fruits360_100x100",
        "Fruits360_multi",
        "weight_dataset",
        "freshness_dataset",
        "sample_benchmark"
    ]

    for name in targets:
        if name not in scans:
            continue
        s = scans[name]
        display_name = "Fruits-360 100x100" if name == "Fruits360_100x100" else ("Fruits-360 multi" if name == "Fruits360_multi" else name)
        print(display_name)
        print(f"Status: {s.status}")
        if s.is_available:
            print(f"Images: {s.image_count}")
            if name == "Fruits360_100x100":
                print(f"Classes: {s.classes_count}")
                tr_count = s.splits_detected.get("Training", 0)
                te_count = s.splits_detected.get("Test", 0)
                print(f"Train: {tr_count}")
                print(f"Test: {te_count}")
                print(f"Validation: {s.validation_status}")
            elif name == "Fruits360_multi":
                print(f"BBoxes: Bounding-box annotations are unavailable.")
                print(f"Masks: None")
                print(f"Validation: {s.validation_status}")
            else:
                print(f"Classes: {s.classes_count}")
                if s.mask_count > 0 or s.mask_available:
                    print(f"Masks: {s.mask_count}")
                elif s.bbox_count > 0 or s.bbox_available:
                    print(f"Annotations: {s.bbox_count}")
                print(f"Validation: {s.validation_status}")
        else:
            if s.notes:
                print(f"Notes: {s.notes}")
        print()

    # Any additional discovered datasets
    extras = [k for k in scans.keys() if k not in targets]
    for name in extras:
        s = scans[name]
        print(name)
        print(f"Status: {s.status}")
        if s.is_available:
            print(f"Images: {s.image_count}")
            print(f"Classes: {s.classes_count}")
            print(f"Validation: {s.validation_status}")
        print()


if __name__ == "__main__":
    main()
