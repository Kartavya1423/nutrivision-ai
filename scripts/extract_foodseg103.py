#!/usr/bin/env python3
"""
NutriVision AI — FoodSeg103 Extraction Script
Extracts image and segmentation mask files from official Hugging Face Parquet distribution.
"""

import os
import sys
from pathlib import Path
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapper import FOODSEG103_CANONICAL
DOWNLOADS_DIR = Path(r"C:\Users\mdahs\Downloads")
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "FoodSeg103"


def extract_split(parquet_paths, split_name):
    img_out = TARGET_DIR / "Images" / "img_dir" / split_name
    ann_out = TARGET_DIR / "Images" / "ann_dir" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    total_extracted = 0
    for p in parquet_paths:
        print(f"Reading {p.name}...")
        table = pq.read_table(p)
        n = len(table)
        for i in range(n):
            img_data = table["image"][i].as_py()
            lbl_data = table["label"][i].as_py()

            img_bytes = img_data["bytes"]
            lbl_bytes = lbl_data["bytes"]

            filename = img_data.get("path")
            if not filename:
                stem = f"{table['id'][i].as_py():08d}"
                filename = f"{stem}.jpg"
            else:
                stem = Path(filename).stem

            # Save image
            with open(img_out / filename, "wb") as f_img:
                f_img.write(img_bytes)

            # Save mask
            mask_name = f"{stem}.png"
            with open(ann_out / mask_name, "wb") as f_lbl:
                f_lbl.write(lbl_bytes)

            total_extracted += 1
            if total_extracted % 1000 == 0:
                print(f"  Extracted {total_extracted} {split_name} pairs...")

    print(f"Completed {split_name}: {total_extracted} image-mask pairs.")
    return total_extracted


def main():
    print("=" * 60)
    print("Extracting FoodSeg103 from Parquet to data/raw/FoodSeg103...")
    print("=" * 60)

    train_parquets = [
        DOWNLOADS_DIR / "train-00000-of-00003.parquet",
        DOWNLOADS_DIR / "train-00001-of-00003.parquet",
        DOWNLOADS_DIR / "train-00002-of-00003.parquet",
    ]
    val_parquets = [
        DOWNLOADS_DIR / "validation-00000-of-00001.parquet"
    ]

    for p in train_parquets + val_parquets:
        if not p.is_file():
            print(f"ERROR: Missing parquet file: {p}")
            sys.exit(1)

    # In FoodSeg103 benchmark: train is training set, test/val is the evaluation set
    # We will write train into train, and validation into test (and symlink/support val)
    n_train = extract_split(train_parquets, "train")
    n_test = extract_split(val_parquets, "test")

    # Write category_id.txt
    cat_file = TARGET_DIR / "category_id.txt"
    with open(cat_file, "w", encoding="utf-8") as f:
        for cat in FOODSEG103_CANONICAL:
            f.write(f"{cat}\n")
    print(f"Wrote {len(FOODSEG103_CANONICAL)} classes to {cat_file}")

    print(f"\nFoodSeg103 Extraction Complete: {n_train} Train, {n_test} Test. Total: {n_train + n_test}")


if __name__ == "__main__":
    main()
