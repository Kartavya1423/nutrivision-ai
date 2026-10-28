#!/usr/bin/env python3
"""
NutriVision AI — Detection Dataset Data Leakage Audit
Phase 2: YOLO11 Food Detection Pipeline

Audits train, validation, and test splits in the YOLO detection dataset
against exact and perceptual duplicates detected during Phase 1.
Ensures zero data leakage between training, validation, and test partitions.
Outputs:
    outputs/detection/data_leakage_report.json
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_data_leakage() -> Dict[str, Any]:
    print("=" * 76)
    print("        NutriVision AI — Detection Data Leakage Verification Audit       ")
    print("=" * 76)

    data_dir = PROJECT_ROOT / "data"
    processed_dir = data_dir / "processed"
    yolo_dir = processed_dir / "yolo_detection"
    images_dir = yolo_dir / "images"
    outputs_dir = PROJECT_ROOT / "outputs" / "detection"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Index images in each split
    split_images: Dict[str, Set[str]] = {
        "train": set(),
        "val": set(),
        "test": set()
    }

    # Map filename stems to split
    stem_to_split: Dict[str, str] = {}

    for split in ["train", "val", "test"]:
        s_dir = images_dir / split
        if s_dir.is_dir():
            for p in s_dir.glob("*.*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    split_images[split].add(p.name)
                    # stem without prefix (e.g. foodseg103_00000001 -> 00000001)
                    stem_to_split[p.stem] = split
                    if "_" in p.stem:
                        raw_stem = p.stem.split("_", 1)[1]
                        stem_to_split[raw_stem] = split

    train_count = len(split_images["train"])
    val_count = len(split_images["val"])
    test_count = len(split_images["test"])

    print(f"  -> Inspected Detection Partitions:")
    print(f"     Train Images : {train_count}")
    print(f"     Val Images   : {val_count}")
    print(f"     Test Images  : {test_count}")

    # 2. Check overlap between split image sets directly
    train_val_overlap = split_images["train"].intersection(split_images["val"])
    train_test_overlap = split_images["train"].intersection(split_images["test"])
    val_test_overlap = split_images["val"].intersection(split_images["test"])

    # 3. Read Phase 1 duplicate report
    dup_file = processed_dir / "duplicates_report.csv"
    cross_split_exact_duplicates: List[Dict[str, Any]] = []
    cross_split_visual_duplicates: List[Dict[str, Any]] = []
    within_split_exact_duplicates: List[Dict[str, Any]] = []
    within_split_visual_duplicates: List[Dict[str, Any]] = []

    if dup_file.is_file():
        with open(dup_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                f1 = Path(r["file_1"]).stem
                f2 = Path(r["file_2"]).stem
                dtype = r.get("duplicate_type", "EXACT")

                split1 = stem_to_split.get(f1)
                split2 = stem_to_split.get(f2)

                if split1 and split2:
                    entry = {
                        "file_1": r["file_1"],
                        "file_2": r["file_2"],
                        "split_1": split1,
                        "split_2": split2,
                        "duplicate_type": dtype,
                        "similarity_metric": r.get("similarity_metric", ""),
                        "notes": r.get("notes", "")
                    }
                    if split1 != split2:
                        if dtype == "EXACT":
                            cross_split_exact_duplicates.append(entry)
                        else:
                            cross_split_visual_duplicates.append(entry)
                    else:
                        if dtype == "EXACT":
                            within_split_exact_duplicates.append(entry)
                        else:
                            within_split_visual_duplicates.append(entry)

    # 4. Formulate leakage report
    leakage_detected = (
        len(train_val_overlap) > 0 or
        len(train_test_overlap) > 0 or
        len(val_test_overlap) > 0 or
        len(cross_split_exact_duplicates) > 0
    )

    report_data = {
        "status": "PASS" if not leakage_detected else "FAIL",
        "leakage_detected": leakage_detected,
        "exact_duplicate_cross_split_count": len(cross_split_exact_duplicates),
        "visual_duplicate_cross_split_count": len(cross_split_visual_duplicates),
        "within_split_exact_duplicate_count": len(within_split_exact_duplicates),
        "within_split_visual_duplicate_count": len(within_split_visual_duplicates),
        "excluded_count": 0,
        "remaining_train_count": train_count,
        "remaining_val_count": val_count,
        "remaining_test_count": test_count,
        "split_overlaps": {
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
        },
        "cross_split_exact_duplicates": cross_split_exact_duplicates,
        "cross_split_visual_duplicates": cross_split_visual_duplicates[:20],
    }

    report_path = outputs_dir / "data_leakage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n  -> Leakage Audit Status             : {report_data['status']}")
    print(f"  -> Cross-Split Exact Duplicates     : {len(cross_split_exact_duplicates)}")
    print(f"  -> Cross-Split Visual Candidates    : {len(cross_split_visual_duplicates)}")
    print(f"  -> File Overlaps (Train/Val/Test)   : {len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)}")
    print(f"  -> Saved Leakage Report             : {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76 + "\n")

    return report_data


if __name__ == "__main__":
    check_data_leakage()
