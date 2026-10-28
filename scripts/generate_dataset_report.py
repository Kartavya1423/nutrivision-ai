#!/usr/bin/env python3
"""
NutriVision AI — Dataset Report Generator CLI
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Generates:
  outputs/dataset_report.html
  outputs/dataset_report.json
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
from src.data.dataset_report import DatasetReportGenerator


def main():
    print("=" * 70)
    print("        NutriVision AI — Dataset Report Generation Subsystem       ")
    print("=" * 70)

    # 1. Scan
    print("[*] Scanning raw datasets...")
    scanner = DatasetScanner()
    scans = scanner.scan_all()

    # 2. Validate
    print("[*] Validating discovered assets...")
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
    duplicates = validator.find_duplicates(all_images) if all_images else []
    validation_paths = validator.save_validation_reports(img_results, mask_results, duplicates)

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

    # 3. Statistics
    print("[*] Computing analytical dataset statistics...")
    stats = {}
    for name, s in scans.items():
        stats[name] = DatasetStatistics.compute_summary(s)

    # 4. Class Mapping & Overlap
    print("[*] Performing cross-dataset class taxonomy mapping...")
    mapper = ClassMapper()
    mapping_records = mapper.generate_mapping(scans)
    mapper.save_mapping_csv(mapping_records)
    overlap = mapper.analyze_cross_dataset_overlap(mapping_records)

    # 5. Reports
    print("[*] Compiling executive HTML and JSON reports...")
    reporter = DatasetReportGenerator()
    purpose_path = reporter.save_dataset_purpose_json()
    manifest_path = reporter.save_dataset_manifest_json(scans)
    json_report = reporter.save_dataset_report_json(scans, stats, val_summary, overlap, duplicates)
    html_report = reporter.generate_html_report(scans, stats, val_summary, overlap, duplicates)

    print("\n" + "=" * 70)
    print("                     Generated Reports Summary                     ")
    print("=" * 70)
    print(f"  HTML Executive Dashboard : {html_report.relative_to(PROJECT_ROOT)}")
    print(f"  Machine-Readable JSON    : {json_report.relative_to(PROJECT_ROOT)}")
    print(f"  Dataset Manifest         : {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  Dataset Purpose Map      : {purpose_path.relative_to(PROJECT_ROOT)}")
    print(f"  Class Taxonomy Mapping   : {mapper.output_csv.relative_to(PROJECT_ROOT)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
