#!/usr/bin/env python3
"""
NutriVision AI — Master Dataset Preparation Orchestrator
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Executes the complete Phase 1 workflow idempotently:
1. Scan datasets (FoodSeg103, UECFoodPixComplete, Fruits-360, weight, freshness, etc.)
2. Validate images and segmentation mask pairings
3. Detect exact and visual duplicates (non-destructive)
4. Generate analytical statistics
5. Generate unified class mappings with broad food taxonomy
6. Generate dataset manifest (data/processed/dataset_manifest.json)
7. Generate reproducible train/val/test splits preserving official splits (data/splits/)
8. Compile HTML executive dashboard and JSON reports (outputs/)

Usage:
  python scripts/prepare_datasets.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_scanner import DatasetScanner
from src.data.dataset_validator import DatasetValidator
from src.data.dataset_statistics import DatasetStatistics
from src.data.class_mapper import ClassMapper
from src.data.split_generator import SplitGenerator
from src.data.dataset_report import DatasetReportGenerator


def main():
    print("=" * 76)
    print("      NutriVision AI — Master Dataset Preparation Engine (Phase 1)      ")
    print("=" * 76)
    print(" [POLICY] Phase 1 strictly prepares and validates datasets.")
    print("          No deep learning models (YOLO11, SAM2, EfficientNet) are trained.")
    print("          All raw datasets remain completely unmodified.")
    print("-" * 76)

    # 1. SCAN DATASETS
    print("\n[Step 1/8] Scanning raw dataset repositories in data/raw/...")
    scanner = DatasetScanner()
    scans = scanner.scan_all()
    for name, s in scans.items():
        print(f"  -> {name:<20}: {s.status:<14} ({s.image_count} images, {s.classes_count} classes)")

    # 2. VALIDATE IMAGES & MASKS
    print("\n[Step 2/8] Validating image integrity and segmentation correspondences...")
    validator = DatasetValidator()
    all_images = []
    mask_results = []
    for name, s in scans.items():
        if s.is_available:
            all_images.extend(s.image_paths)
            if s.mask_available and s.mask_paths:
                m_res = validator.validate_segmentation_pairs(name, s.image_paths, s.mask_paths)
                mask_results.extend(m_res)

    img_results = validator.validate_dataset_images(all_images)
    print(f"  -> Verified {len(img_results.get('valid', []))} valid images.")
    print(f"  -> Corrupted / unreadable images : {len(img_results.get('corrupted', []))}")
    print(f"  -> Zero-byte images              : {len(img_results.get('zero_byte', []))}")
    print(f"  -> Mask pairs verified           : {len(mask_results)}")

    # 3. DUPLICATE AUDIT
    print("\n[Step 3/8] Executing non-destructive duplicate detection (SHA-256 + dHash)...")
    duplicates = validator.find_duplicates(all_images) if all_images else []
    print(f"  -> Discovered {len(duplicates)} duplicate pairings (exact or visual).")
    val_files = validator.save_validation_reports(img_results, mask_results, duplicates)

    val_summary = {
        "valid_images_count": len(img_results.get("valid", [])),
        "corrupted_images_count": len(img_results.get("corrupted", [])),
        "zero_byte_count": len(img_results.get("zero_byte", [])),
        "unsupported_format_count": len(img_results.get("unsupported", [])),
        "total_pairs_inspected": len(mask_results),
        "unpaired_images_count": sum(1 for m in mask_results if not m.is_paired),
        "dimension_mismatch_count": sum(1 for m in mask_results if m.is_paired and not m.dimensions_match),
        "duplicate_pairs_count": len(duplicates)
    }

    # 4. DATASET STATISTICS
    print("\n[Step 4/8] Deriving statistical and class imbalance metrics...")
    stats = {}
    for name, s in scans.items():
        stats[name] = DatasetStatistics.compute_summary(s)
        if s.is_available and s.classes_count > 0:
            st = stats[name]
            imb = f"{st.imbalance_ratio:.2f}" if st.imbalance_ratio else "N/A"
            print(f"  -> {name:<20}: Imbalance ratio: {imb} (Min: {st.min_class_count}, Max: {st.max_class_count})")

    # Persist Fruits-360 per-class breakdown CSV
    f360_key = "Fruits360_100x100" if "Fruits360_100x100" in stats else "Fruits360"
    if f360_key in stats and stats[f360_key].fruits360_table:
        f360_csv = DatasetStatistics.save_fruits360_statistics_csv(
            stats[f360_key].fruits360_table,
            PROJECT_ROOT / "data" / "processed" / "fruits360_statistics.csv"
        )
        print(f"  -> Saved Fruits-360 per-class counts to {f360_csv.relative_to(PROJECT_ROOT)}")

    # 5. UNIFIED CLASS MAPPING & OVERLAP
    print("\n[Step 5/8] Constructing unified, reversible food taxonomy mapping...")
    mapper = ClassMapper()
    mapping_records = mapper.generate_mapping(scans)
    class_map_file = mapper.save_mapping_csv(mapping_records)
    dataset_classes_file = mapper.save_dataset_classes_csv(scans)
    overlap = mapper.analyze_cross_dataset_overlap(mapping_records)
    class_overlap_file = mapper.save_class_overlap_csv(overlap, mapping_records)
    print(f"  -> Saved {len(mapping_records)} normalized taxonomy entries to {class_map_file.relative_to(PROJECT_ROOT)}")
    print(f"  -> Saved native dataset class counts to {dataset_classes_file.relative_to(PROJECT_ROOT)}")
    print(f"  -> Saved cross-dataset overlap report to {class_overlap_file.relative_to(PROJECT_ROOT)}")
    print(f"  -> Confirmed cross-dataset matches: {overlap.get('total_confirmed_matches', 0)}")
    print(f"  -> Possible semantic overlaps      : {overlap.get('total_possible_matches', 0)}")

    # 6. SPLIT GENERATION
    print("\n[Step 6/8] Generating deterministic splits and preserving official benchmarks...")
    split_gen = SplitGenerator()
    for name, s in scans.items():
        if s.is_available and s.image_paths:
            s_meta = split_gen.generate_split(name, s.image_paths, official_splits=s.official_splits)
            pres = "YES (Official Test preserved)" if s_meta.official_splits_preserved else "Generated 70/15/15"
            print(f"  -> {name:<20}: Train: {s_meta.train_count}, Val: {s_meta.val_count}, Test: {s_meta.test_count} [{pres}]")

    # 7. MANIFEST & PURPOSE MAPPINGS
    print("\n[Step 7/8] Generating dataset manifest and module purpose mappings...")
    reporter = DatasetReportGenerator()
    purpose_file = reporter.save_dataset_purpose_json()
    manifest_file = reporter.save_dataset_manifest_json(scans)
    print(f"  -> Dataset Manifest : {manifest_file.relative_to(PROJECT_ROOT)}")
    print(f"  -> Purpose Mappings : {purpose_file.relative_to(PROJECT_ROOT)}")

    # 8. HTML & JSON REPORTS
    print("\n[Step 8/8] Rendering visual executive HTML dashboard and analytical JSON...")
    json_file = reporter.save_dataset_report_json(scans, stats, val_summary, overlap, duplicates)
    html_file = reporter.generate_html_report(scans, stats, val_summary, overlap, duplicates)
    print(f"  -> HTML Dashboard   : {html_file.relative_to(PROJECT_ROOT)}")
    print(f"  -> Analytical JSON  : {json_file.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 76)
    print("       [SUCCESS] PHASE 1 DATASET PREPARATION COMPLETED SUCCESSFULLY!       ")
    print("=" * 76)
    print("  All datasets scanned, validated, categorized, split, and documented.")
    print("  Zero model training occurred. All raw files remain pristine.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
