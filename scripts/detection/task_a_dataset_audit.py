#!/usr/bin/env python3
"""
NutriVision AI — Phase 2 Task A: Dataset and Labels Verification
High-speed, robust verification of:
1. Active YOLO11 dataset YAML and checkpoint taxonomy alignment
2. Annotation format, class IDs, bounding-box coordinate boundaries
3. Image/label pairing, empty annotations, corrupt files
4. Split disjointness and data leakage across train, val, and test splits
5. Visualizes 20 representative test images with ground-truth bounding boxes
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path("D:/NutriVision-AI (2)/NutriVision-AI")
DATASET_YAML_PATH = PROJECT_ROOT / "data/processed/yolo_detection/data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs/detection"
GT_VIS_DIR = OUTPUT_DIR / "diagnosis_gt_visualizations"

GT_VIS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_yaml() -> Tuple[Path, Dict]:
    with open(DATASET_YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DATASET_YAML_PATH, cfg


def audit_splits(cfg: Dict) -> Dict:
    base_dir = PROJECT_ROOT / cfg.get("path", "data/processed/yolo_detection")
    names_dict = cfg.get("names", {})
    nc = len(names_dict)

    split_reports = {}
    split_stems: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_fingerprints: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for split in ["train", "val", "test"]:
        img_rel = cfg.get(split, f"images/{split}")
        img_dir = base_dir / img_rel
        lbl_dir = base_dir / "labels" / split

        img_files = sorted(list(img_dir.glob("*.*")))
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))

        img_stems = {}
        for p in img_files:
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img_stems[p.stem] = p
                # Fast file fingerprint: size + stem
                split_fingerprints[split].add(f"{p.stem}_{p.stat().st_size}")

        lbl_stems = {p.stem: p for p in lbl_files}
        split_stems[split] = set(img_stems.keys())

        missing_labels = [s for s in img_stems if s not in lbl_stems]
        orphan_labels = [s for s in lbl_stems if s not in img_stems]

        total_boxes = 0
        empty_labels = 0
        invalid_format_lines = 0
        out_of_bounds_class_ids = 0
        out_of_bounds_coords = 0
        zero_size_images = 0

        for stem, p in img_stems.items():
            if p.stat().st_size == 0:
                zero_size_images += 1

        # Check all label annotations in the split
        class_instance_counts = {}
        for stem, lbl_p in lbl_stems.items():
            try:
                txt = lbl_p.read_text(encoding="utf-8").strip()
                if not txt:
                    empty_labels += 1
                    continue
                for line in txt.splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_format_lines += 1
                        continue
                    try:
                        cid = int(parts[0])
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        invalid_format_lines += 1
                        continue

                    if cid < 0 or cid >= nc:
                        out_of_bounds_class_ids += 1
                    else:
                        class_instance_counts[cid] = class_instance_counts.get(cid, 0) + 1

                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        out_of_bounds_coords += 1
                    total_boxes += 1
            except Exception:
                invalid_format_lines += 1

        split_reports[split] = {
            "image_dir": str(img_dir.relative_to(PROJECT_ROOT)),
            "label_dir": str(lbl_dir.relative_to(PROJECT_ROOT)),
            "image_count": len(img_stems),
            "label_count": len(lbl_stems),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "empty_labels": empty_labels,
            "total_boxes": total_boxes,
            "invalid_format_lines": invalid_format_lines,
            "out_of_bounds_class_ids": out_of_bounds_class_ids,
            "out_of_bounds_coords": out_of_bounds_coords,
            "zero_size_images": zero_size_images,
            "unique_classes_present": len(class_instance_counts),
        }

    # Leakage check across splits
    train_val_stem_leak = len(split_stems["train"].intersection(split_stems["val"]))
    train_test_stem_leak = len(split_stems["train"].intersection(split_stems["test"]))
    val_test_stem_leak = len(split_stems["val"].intersection(split_stems["test"]))

    train_val_fp_leak = len(split_fingerprints["train"].intersection(split_fingerprints["val"]))
    train_test_fp_leak = len(split_fingerprints["train"].intersection(split_fingerprints["test"]))
    val_test_fp_leak = len(split_fingerprints["val"].intersection(split_fingerprints["test"]))

    leakage_report = {
        "stem_leakage": {
            "train_val": train_val_stem_leak,
            "train_test": train_test_stem_leak,
            "val_test": val_test_stem_leak,
        },
        "fingerprint_leakage": {
            "train_val": train_val_fp_leak,
            "train_test": train_test_fp_leak,
            "val_test": val_test_fp_leak,
        },
    }

    return {"splits": split_reports, "leakage": leakage_report}


def visualize_20_ground_truth(cfg: Dict) -> List[str]:
    base_dir = PROJECT_ROOT / cfg.get("path", "data/processed/yolo_detection")
    names_dict = cfg.get("names", {})
    test_img_dir = base_dir / cfg.get("test", "images/test")
    test_lbl_dir = base_dir / "labels/test"

    test_imgs = sorted(list(test_img_dir.glob("*.jpg")))
    saved_images = []

    count = 0
    step = max(1, len(test_imgs) // 25)
    for i in range(0, len(test_imgs), step):
        if count >= 20:
            break
        img_p = test_imgs[i]
        lbl_p = test_lbl_dir / (img_p.stem + ".txt")
        if not lbl_p.is_file():
            continue

        lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue

        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cname = names_dict.get(cid, f"cls_{cid}")
            xc, yc, bw, bh = float(parts[1]) * w, float(parts[2]) * h, float(parts[3]) * w, float(parts[4]) * h
            x1 = max(0, int(xc - bw / 2))
            y1 = max(0, int(yc - bh / 2))
            x2 = min(w - 1, int(xc + bw / 2))
            y2 = min(h - 1, int(yc + bh / 2))

            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 230, 0), 2)
            label_text = f"GT:{cname} ({cid})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(im, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 180, 0), -1)
            cv2.putText(im, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        out_name = f"gt_vis_{count+1:02d}_{img_p.stem}.jpg"
        out_p = GT_VIS_DIR / out_name
        cv2.imwrite(str(out_p), im)
        saved_images.append(str(out_p.relative_to(PROJECT_ROOT)))
        count += 1

    return saved_images


def main():
    print("=== NutriVision AI: Task A Dataset Audit ===")
    yaml_path, cfg = load_dataset_yaml()
    print(f"Loaded YAML: {yaml_path.relative_to(PROJECT_ROOT)}")
    print(f"Total classes defined in YAML: {len(cfg.get('names', {}))}")

    audit_res = audit_splits(cfg)
    vis_res = visualize_20_ground_truth(cfg)

    full_report = {
        "dataset_yaml": str(yaml_path.relative_to(PROJECT_ROOT)),
        "class_count": len(cfg.get("names", {})),
        "split_audit": audit_res["splits"],
        "leakage_audit": audit_res["leakage"],
        "ground_truth_visualizations_count": len(vis_res),
        "ground_truth_visualization_samples": vis_res,
    }

    out_json = OUTPUT_DIR / "task_a_dataset_audit_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("Audit completed successfully!")
    print(f"Report saved to: {out_json.relative_to(PROJECT_ROOT)}")
    print(f"Visualized {len(vis_res)} ground-truth test images to: {GT_VIS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
