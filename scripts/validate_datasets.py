#!/usr/bin/env python3
"""
NutriVision AI — Dataset Validation & Duplicate Detection CLI
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Usage:
  python scripts/validate_datasets.py
  python scripts/validate_datasets.py --check-duplicates
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_scanner import DatasetScanner
from src.data.dataset_validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — Dataset Validation Engine")
    parser.add_argument("--check-duplicates", action="store_true", default=True, help="Run perceptual and exact duplicate check (default: True)")
    args = parser.parse_args()

    print("=" * 70)
    print("      NutriVision AI — Dataset Validation & Integrity Audit       ")
    print("=" * 70)

    scanner = DatasetScanner()
    scans = scanner.scan_all()

    validator = DatasetValidator()

    all_images = []
    mask_results = []

    for name, scan in scans.items():
        if not scan.is_available:
            continue
        print(f"[*] Validating {name} ({scan.image_count} images)...")
        all_images.extend(scan.image_paths)

        if scan.mask_available and scan.mask_paths:
            print(f"    -> Checking image-mask correspondence for {len(scan.mask_paths)} masks...")
            m_res = validator.validate_segmentation_pairs(name, scan.image_paths, scan.mask_paths)
            mask_results.extend(m_res)

    print(f"\n[*] Total images collected for integrity verification: {len(all_images)}")
    img_results = validator.validate_dataset_images(all_images)

    duplicate_results = []
    if args.check_duplicates and all_images:
        print("[*] Performing SHA-256 and dHash duplicate detection...")
        duplicate_results = validator.find_duplicates(all_images)
        print(f"    -> Found {len(duplicate_results)} duplicate pairings.")

    # Save output artifacts
    saved_files = validator.save_validation_reports(img_results, mask_results, duplicate_results)

    print("\n" + "=" * 70)
    print("                     Validation Results Summary                    ")
    print("=" * 70)
    print(f"  Valid Images        : {len(img_results.get('valid', []))}")
    print(f"  Corrupted Images    : {len(img_results.get('corrupted', []))}")
    print(f"  Zero-Byte Files     : {len(img_results.get('zero_byte', []))}")
    print(f"  Unsupported Formats : {len(img_results.get('unsupported', []))}")
    print(f"  Mask Pairs Checked  : {len(mask_results)}")
    print(f"  Duplicate Pairs     : {len(duplicate_results)}")
    print("-" * 70)
    for k, p in saved_files.items():
        print(f"  Report saved: {p.relative_to(PROJECT_ROOT)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
