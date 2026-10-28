#!/usr/bin/env python3
"""
NutriVision AI — Segmentation Mask to YOLO Bounding Box Pipeline
Phase 2: YOLO11 Food Detection Pipeline

Orchestrates conversion of FoodSeg103 and UECFoodPixComplete segmentation
masks into normalized YOLO detection annotations.
Preserves official benchmark test sets and writes detection dataset structure:
    data/processed/yolo_detection/
    ├── images/ (train/val/test)
    ├── labels/ (train/val/test)
    ├── dataset.yaml
    └── metadata/
"""

from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detection_config import MaskConversionConfig
from src.detection.mask_to_yolo import MaskToYOLOConverter, YOLOBox, MaskConversionResult


def build_detection_class_mapping(processed_dir: Path) -> Tuple[Dict[str, Dict[int, int]], Dict[int, str], List[Dict[str, Any]]]:
    """
    Builds a deterministic 0-indexed detector class mapping based on Phase 1
    class mappings and confirmed overlap analysis.

    Returns:
        dataset_class_maps: { "FoodSeg103": {native_id: detector_id}, "UECFoodPixComplete": {native_id: detector_id} }
        detector_class_names: { detector_id: class_name }
        records: list of mapping records for CSV export
    """
    class_mapping_csv = processed_dir / "class_mapping.csv"
    class_overlap_csv = processed_dir / "class_overlap.csv"
    dataset_classes_csv = processed_dir / "dataset_classes.csv"

    # Identify confirmed cross-dataset matches
    confirmed_pairs = set()
    if class_overlap_csv.is_file():
        with open(class_overlap_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("match_type") == "CONFIRMED":
                    c_a = r.get("class_a", "").strip().lower().replace(" ", "_")
                    c_b = r.get("class_b", "").strip().lower().replace(" ", "_")
                    confirmed_pairs.add((c_a, c_b))
                    confirmed_pairs.add((c_b, c_a))

    # Read native classes for FoodSeg103 and UECFoodPixComplete
    foodseg_classes: List[Tuple[int, str]] = []
    uec_classes: List[Tuple[int, str]] = []

    if dataset_classes_csv.is_file():
        with open(dataset_classes_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ds = r.get("dataset")
                c_id = int(r.get("original_class_id", 0))
                c_name = r.get("original_class_name", "").strip()
                if ds == "FoodSeg103":
                    foodseg_classes.append((c_id, c_name))
                elif ds == "UECFoodPixComplete":
                    uec_classes.append((c_id, c_name))

    # Build canonical detector dictionary
    # First: map all FoodSeg103 classes (0 to len-1)
    detector_name_to_id: Dict[str, int] = {}
    detector_id_to_name: Dict[int, str] = {}
    current_detector_id = 0

    foodseg_map: Dict[int, int] = {}
    records: List[Dict[str, Any]] = []

    for native_id, name in foodseg_classes:
        norm_name = name.lower().replace(" ", "_")
        if norm_name not in detector_name_to_id:
            detector_name_to_id[norm_name] = current_detector_id
            detector_id_to_name[current_detector_id] = norm_name
            current_detector_id += 1
        det_id = detector_name_to_id[norm_name]
        foodseg_map[native_id] = det_id
        records.append({
            "dataset": "FoodSeg103",
            "native_class_id": native_id,
            "native_class_name": name,
            "detector_class_id": det_id,
            "detector_class_name": norm_name,
            "mapping_status": "NATIVE"
        })

    # Second: map UECFoodPixComplete classes (merging confirmed matches)
    uec_map: Dict[int, int] = {}
    for native_id, name in uec_classes:
        norm_name = name.lower().replace(" ", "_")

        # Check if this class has a confirmed match in FoodSeg103
        matched_det_id = None
        for fs_norm, uec_norm in confirmed_pairs:
            if uec_norm == norm_name and fs_norm in detector_name_to_id:
                matched_det_id = detector_name_to_id[fs_norm]
                break

        if matched_det_id is not None:
            det_id = matched_det_id
            status = "MERGED_CONFIRMED"
            det_name = detector_id_to_name[det_id]
        elif norm_name in detector_name_to_id:
            det_id = detector_name_to_id[norm_name]
            status = "MERGED_EXACT"
            det_name = norm_name
        else:
            det_id = current_detector_id
            detector_name_to_id[norm_name] = current_detector_id
            detector_id_to_name[current_detector_id] = norm_name
            current_detector_id += 1
            status = "NATIVE"
            det_name = norm_name

        uec_map[native_id] = det_id
        records.append({
            "dataset": "UECFoodPixComplete",
            "native_class_id": native_id,
            "native_class_name": name,
            "detector_class_id": det_id,
            "detector_class_name": det_name,
            "mapping_status": status
        })

    dataset_class_maps = {
        "FoodSeg103": foodseg_map,
        "UECFoodPixComplete": uec_map
    }

    # Save data/processed/detection_class_mapping.csv
    out_csv = processed_dir / "detection_class_mapping.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "native_class_id", "native_class_name",
            "detector_class_id", "detector_class_name", "mapping_status"
        ])
        writer.writeheader()
        writer.writerows(records)

    return dataset_class_maps, detector_id_to_name, records


def link_or_copy(src: Path, dst: Path) -> None:
    """Links file using hardlink (0 bytes consumed) with fallback to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def convert_dataset_split(
    dataset_name: str,
    split_name: str,
    image_paths: List[Path],
    mask_dir: Path,
    target_img_dir: Path,
    target_lbl_dir: Path,
    class_map: Dict[int, int],
    converter: MaskToYOLOConverter,
    mask_map: Optional[Dict[str, Path]] = None,
) -> Tuple[List[MaskConversionResult], Counter]:
    """
    Converts a single split of a dataset into YOLO detection format.
    """
    results: List[MaskConversionResult] = []
    boxes_per_class: Counter = Counter()

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        stem = img_path.stem
        # Look up mask
        mask_path = None
        if mask_map and stem in mask_map:
            mask_path = mask_map[stem]
        else:
            cand1 = mask_dir / f"{stem}.png"
            cand2 = mask_dir / f"{stem}_mask.png"
            if cand1.is_file():
                mask_path = cand1
            elif cand2.is_file():
                mask_path = cand2

        if not mask_path or not mask_path.is_file():
            continue

        # Convert mask to YOLO boxes
        conv_res = converter.convert(img_path, mask_path, class_mapping=class_map)
        results.append(conv_res)

        # Unique file naming: e.g. foodseg103_00000001.jpg
        dst_stem = f"{dataset_name.lower()}_{stem}"
        dst_img = target_img_dir / f"{dst_stem}{img_path.suffix}"
        dst_lbl = target_lbl_dir / f"{dst_stem}.txt"

        # Write label file (even if empty to register negative background image)
        converter.write_yolo_label_file(dst_lbl, conv_res.boxes)

        # Link image
        link_or_copy(img_path, dst_img)

        # Record class frequencies
        for b in conv_res.boxes:
            boxes_per_class[b.class_id] += 1

    return results, boxes_per_class


def main():
    print("=" * 76)
    print("   NutriVision AI — Segmentation Mask to YOLO Bounding Box Converter    ")
    print("=" * 76)
    print(" Phase 2: Converting pixel-level food masks to YOLO detection format.")
    print(" All bounding boxes are derived via connected-component analysis.")
    print(" Raw dataset files remain pristine.")
    print("-" * 76)

    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"
    outputs_dir = PROJECT_ROOT / "outputs" / "detection"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    yolo_dir = processed_dir / "yolo_detection"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    meta_dir = yolo_dir / "metadata"

    for d in [images_dir, labels_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. CLASS MAPPING
    print("\n[Step 1/5] Building unified detector class mapping...")
    dataset_class_maps, detector_classes, mapping_records = build_detection_class_mapping(processed_dir)
    total_classes = len(detector_classes)
    print(f"  -> Generated {total_classes} detector classes (0 to {total_classes - 1}).")
    print(f"  -> Saved detection class mapping to {processed_dir / 'detection_class_mapping.csv'}")

    # 2. LOAD PHASE 1 SPLITS
    print("\n[Step 2/5] Loading benchmark-preserving dataset splits from Phase 1...")
    foodseg_split_file = splits_dir / "FoodSeg103_split.json"
    uec_split_file = splits_dir / "UECFoodPixComplete_split.json"

    with open(foodseg_split_file, "r", encoding="utf-8") as f:
        foodseg_raw = json.load(f)

    foodseg_splits = {"train": [], "val": [], "test": []}
    for p_str, split_name in foodseg_raw.get("assignments", {}).items():
        if split_name in foodseg_splits:
            foodseg_splits[split_name].append(Path(p_str))

    with open(uec_split_file, "r", encoding="utf-8") as f:
        uec_raw = json.load(f)

    uec_splits = {"train": [], "val": [], "test": []}
    for p_str, split_name in uec_raw.get("assignments", {}).items():
        if split_name in uec_splits:
            uec_splits[split_name].append(Path(p_str))

    print(f"  -> FoodSeg103 splits: Train={len(foodseg_splits['train'])}, Val={len(foodseg_splits['val'])}, Test={len(foodseg_splits['test'])}")
    print(f"  -> UECFoodPix splits: Train={len(uec_splits['train'])}, Val={len(uec_splits['val'])}, Test={len(uec_splits['test'])}")

    # Index masks for fast lookup
    print("\n[Step 3/5] Indexing segmentation mask archives...")
    foodseg_mask_map: Dict[str, Path] = {}
    foodseg_ann_train = raw_dir / "FoodSeg103" / "Images" / "ann_dir" / "train"
    foodseg_ann_test = raw_dir / "FoodSeg103" / "Images" / "ann_dir" / "test"
    for d in [foodseg_ann_train, foodseg_ann_test]:
        if d.is_dir():
            for p in d.glob("*.png"):
                foodseg_mask_map[p.stem] = p

    uec_mask_map: Dict[str, Path] = {}
    uec_mask_train = raw_dir / "UECFoodPixComplete" / "data" / "UECFoodPIXCOMPLETE" / "train" / "mask"
    uec_mask_test = raw_dir / "UECFoodPixComplete" / "data" / "UECFoodPIXCOMPLETE" / "test" / "mask"
    for d in [uec_mask_train, uec_mask_test]:
        if d.is_dir():
            for p in d.glob("*.png"):
                uec_mask_map[p.stem] = p

    print(f"  -> Indexed {len(foodseg_mask_map)} FoodSeg103 masks.")
    print(f"  -> Indexed {len(uec_mask_map)} UECFoodPixComplete masks.")

    # 3. CONVERT MASKS TO YOLO LABELS
    print("\n[Step 4/5] Executing mask-to-YOLO conversion (connected-components)...")
    converter = MaskToYOLOConverter(MaskConversionConfig(min_area_pixels=16, min_dimension_pixels=4))

    all_conversion_results: List[MaskConversionResult] = []
    split_counts: Dict[str, Dict[str, int]] = {
        "train": {"images": 0, "boxes": 0},
        "val": {"images": 0, "boxes": 0},
        "test": {"images": 0, "boxes": 0}
    }
    dataset_box_counts: Counter = Counter()
    class_box_counts: Counter = Counter()

    datasets_to_process = [
        ("FoodSeg103", foodseg_splits, foodseg_mask_map, dataset_class_maps["FoodSeg103"]),
        ("UECFoodPixComplete", uec_splits, uec_mask_map, dataset_class_maps["UECFoodPixComplete"])
    ]

    for ds_name, splits_data, mask_map, cls_map in datasets_to_process:
        print(f"\n  Processing {ds_name}...")
        for split_key in ["train", "val", "test"]:
            img_paths = [Path(p) for p in splits_data[split_key]]
            t_img_dir = images_dir / split_key
            t_lbl_dir = labels_dir / split_key

            res_list, box_counts = convert_dataset_split(
                dataset_name=ds_name,
                split_name=split_key,
                image_paths=img_paths,
                mask_dir=Path(""),
                target_img_dir=t_img_dir,
                target_lbl_dir=t_lbl_dir,
                class_map=cls_map,
                converter=converter,
                mask_map=mask_map,
            )

            all_conversion_results.extend(res_list)
            num_boxes = sum(len(r.boxes) for r in res_list)
            split_counts[split_key]["images"] += len(res_list)
            split_counts[split_key]["boxes"] += num_boxes
            dataset_box_counts[ds_name] += num_boxes
            class_box_counts.update(box_counts)

            print(f"    -> {split_key:<5}: {len(res_list)} images, {num_boxes} derived food bounding boxes")

    # 4. GENERATE DATASET.YAML
    print("\n[Step 5/5] Generating YOLO dataset.yaml and conversion metadata...")
    dataset_yaml_path = yolo_dir / "dataset.yaml"
    import yaml
    yaml_dict = {
        "path": yolo_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(cid): str(detector_classes[cid]) for cid in sorted(detector_classes.keys())}
    }
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        f.write("# NutriVision AI — YOLO11 Food Detection Dataset Configuration\n")
        f.write("# Auto-generated from FoodSeg103 & UECFoodPixComplete segmentation masks\n\n")
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"  -> Generated dataset.yaml at {dataset_yaml_path.relative_to(PROJECT_ROOT)}")

    # 5. DIAGNOSTICS & METADATA REPORT
    total_images_proc = len(all_conversion_results)
    total_boxes_gen = sum(r.boxes_generated for r in all_conversion_results)
    total_invalid_boxes = sum(r.invalid_boxes for r in all_conversion_results)
    total_clipped_boxes = sum(r.clipped_boxes for r in all_conversion_results)
    total_empty_masks = sum(1 for r in all_conversion_results if r.is_empty_mask)

    report_data = {
        "total_images_processed": total_images_proc,
        "total_masks_processed": total_images_proc,
        "total_bounding_boxes_generated": total_boxes_gen,
        "total_invalid_boxes": total_invalid_boxes,
        "total_clipped_boxes": total_clipped_boxes,
        "total_empty_masks": total_empty_masks,
        "total_classes": total_classes,
        "boxes_per_split": split_counts,
        "boxes_per_dataset": dict(dataset_box_counts),
        "boxes_per_class": {detector_classes.get(cid, str(cid)): count for cid, count in class_box_counts.most_common()},
        "annotation_type": "derived",
        "bbox_source": "derived_from_mask",
        "instance_separation": "connected_components",
    }

    report_file = outputs_dir / "bbox_conversion_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Also save to metadata inside yolo_detection
    with open(meta_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"  -> Saved conversion report to {report_file.relative_to(PROJECT_ROOT)}")
    print("\n" + "=" * 76)
    print("      [SUCCESS] MASK-TO-YOLO CONVERSION COMPLETED SUCCESSFULLY!        ")
    print("=" * 76)
    print(f"  Total Images Processed : {total_images_proc}")
    print(f"  Total Bounding Boxes   : {total_boxes_gen}")
    print(f"  Train Boxes            : {split_counts['train']['boxes']} across {split_counts['train']['images']} images")
    print(f"  Val Boxes              : {split_counts['val']['boxes']} across {split_counts['val']['images']} images")
    print(f"  Test Boxes             : {split_counts['test']['boxes']} across {split_counts['test']['images']} images")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
